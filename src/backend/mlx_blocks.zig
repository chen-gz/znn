//! MLX GPU 组合原语 (MLX GPU Composite Primitives)
//!
//! 提取模型无关的高阶 Apple MLX 硬件加速组合块：
//! - 旋转位置编码 (`Rope`)：统一支持标准 RoPE 与比例旋转位置编码
//! - 多头归一化与旋转 (`projectHeads`)：统一单步解码 (t=1) 与预填充批处理 (t>1) 的 Q/K/V 投影归一化、转置与 RoPE 旋转
//! - 门控多层感知机 (`gatedMlp`)：统一门控前馈网络（支持融合 gate_up 与独立 gate/up，可选前后 RMSNorm）
//! - 残差相加与缩放 (`residualScale`)：统一 `(x + residual) * scale` 或 `x + residual`
//! - 最终隐状态提取与归一化 (`finalHidden`)：统一末尾行切片与最终均方根归一化

const std = @import("std");
const mlx = @import("mlx.zig");
const rope_mod = @import("rope.zig");
const activation = @import("activation.zig");
const offload = @import("offload.zig");

pub const Stream = mlx.Stream;
pub const QuantWeights = offload.QuantWeights;

/// 单个注意力层的 GPU 旋转位置编码：partial_rotary_factor 为 1 时为标准 RoPE (base = rope_theta)，
/// 小于 1 时为在完整 head_dim 上旋转的比例旋转位置编码 (常驻 [head_dim / 2] 周期表，由 `RopeOptions.fillPeriods` 生成，
/// 与 CPU `tensor.RopeTable` 共用同一频率公式；`mlx.fast.rope` 以 theta_i = pos / freqs[i] 计算旋转角)
pub const Rope = struct {
    head_dim: usize,
    base: f32,
    traditional: bool,
    freqs: ?mlx.Array = null,

    /// 只支持旋转整行 (rotary_offset = 0) 的偶数 head_dim
    pub fn init(head_dim: usize, options: rope_mod.RopeOptions) !Rope {
        if (head_dim == 0 or head_dim % 2 != 0 or options.rotary_offset != 0) return error.InvalidShape;
        const traditional = options.mode == .interleaved;
        if (options.partial_rotary_factor >= 1.0) return .{ .head_dim = head_dim, .base = options.base, .traditional = traditional };
        const half = head_dim / 2;
        if (half > rope_mod.max_angles) return error.InvalidShape;
        var buf: [rope_mod.max_angles]f32 = undefined;
        options.fillPeriods(head_dim, buf[0..half]);
        return .{
            .head_dim = head_dim,
            .base = options.base,
            .traditional = traditional,
            .freqs = try mlx.Array.fromSlice(buf[0..half], &.{half}),
        };
    }

    pub fn deinit(self: *Rope) void {
        if (self.freqs) |*f| f.deinit();
        self.freqs = null;
    }

    /// 对 [B, heads, T, head_dim] 张量施加旋转位置编码，第 t 个位置的绝对位置为 offset + t
    pub fn apply(self: *const Rope, x: mlx.Array, offset: usize, s: ?Stream) !mlx.Array {
        if (self.freqs) |f| return mlx.Array.fastRoPEWithFreqs(x, self.head_dim, self.traditional, 1.0, offset, f, s);
        return mlx.Array.fastRoPE(x, self.head_dim, self.traditional, self.base, 1.0, offset, s);
    }
};

/// 将投影结果 [1, t, n * head_dim] (或单步解码 [1, n, 1, head_dim]) 按头执行均方根层归一化、维度重排与可选旋转位置编码：
/// - 单步解码 (t == 1): 直接以 [1, n, 1, head_dim] 内存布局就地归一化并施加 RoPE，避免转置；
/// - 预填充/序列 (t > 1): 重塑为 [1, t, n, head_dim] 归一化后转置为 [1, n, t, head_dim] 并施加 RoPE；
/// - 若 rope 为 null (例如无旋转的 V 投影)，则仅执行头归一化与排布。
pub fn projectHeads(
    raw: mlx.Array,
    weight: ?mlx.Array,
    t: usize,
    n: usize,
    head_dim: usize,
    eps: f32,
    rope: ?*const Rope,
    pos: usize,
    s: ?Stream,
) !mlx.Array {
    if (t == 1) {
        var heads_4d = try raw.reshape(&.{ 1, n, 1, head_dim }, s);
        defer heads_4d.deinit();
        var normed = try mlx.Array.fastRmsNorm(heads_4d, weight, eps, s);
        if (rope) |rp| {
            defer normed.deinit();
            return rp.apply(normed, pos, s);
        }
        return normed;
    }
    var heads_4d = try raw.reshape(&.{ 1, t, n, head_dim }, s);
    defer heads_4d.deinit();
    var normed = try mlx.Array.fastRmsNorm(heads_4d, weight, eps, s);
    defer normed.deinit();
    var trans = try normed.transposeAxes(&.{ 0, 2, 1, 3 }, s);
    if (rope) |rp| {
        defer trans.deinit();
        return rp.apply(trans, pos, s);
    }
    return trans;
}

/// 门控多层感知机 (Gated MLP) 组合原语：
/// - 可选前置均方根层归一化 (pre_norm_w)
/// - 门控投影与上升投影 (若 up_proj 为 null 则视为 gate_proj 已在最后一轴水平融合 gate 与 up)
/// - 激活函数 (GeLU / SiLU) 与逐元素相乘
/// - 下降投影 (down_proj) 与可选后置均方根归一化 (post_norm_w)
pub fn gatedMlp(
    x: mlx.Array,
    gate_proj: *const QuantWeights,
    up_proj: ?*const QuantWeights,
    down_proj: *const QuantWeights,
    act: activation.GeluApproximation,
    pre_norm_w: ?mlx.Array,
    post_norm_w: ?mlx.Array,
    eps: f32,
    s: ?Stream,
) !mlx.Array {
    var in_norm: ?mlx.Array = if (pre_norm_w) |pw|
        try mlx.Array.fastRmsNorm(x, pw, eps, s)
    else
        null;
    defer if (in_norm) |*in| in.deinit();
    const mlp_in = in_norm orelse x;

    var gate: mlx.Array = undefined;
    var up: mlx.Array = undefined;
    if (up_proj) |up_p| {
        gate = try gate_proj.forwardArray(mlp_in, s);
        errdefer gate.deinit();
        up = try up_p.forwardArray(mlp_in, s);
    } else {
        var gate_up = try gate_proj.forwardArray(mlp_in, s);
        defer gate_up.deinit();
        var gu_vec = try gate_up.split(2, -1, s);
        defer gu_vec.deinit();
        gate = gu_vec.get(0) orelse return error.ShapeMismatch;
        errdefer gate.deinit();
        up = gu_vec.get(1) orelse return error.ShapeMismatch;
    }
    defer gate.deinit();
    defer up.deinit();

    var act_arr = try gate.gelu(act, s);
    defer act_arr.deinit();
    var hidden = try mlx.Array.multiply(act_arr, up, s);
    defer hidden.deinit();

    var down_out = try down_proj.forwardArray(hidden, s);
    if (post_norm_w) |pnw| {
        defer down_out.deinit();
        return mlx.Array.fastRmsNorm(down_out, pnw, eps, s);
    }
    return down_out;
}

/// 残差相加与层标量缩放融合组合块：`out = (x + residual) * scale`（scale 为 null 时为纯相加）
pub fn residualScale(
    x: mlx.Array,
    residual: mlx.Array,
    scale: ?mlx.Array,
    s: ?Stream,
) !mlx.Array {
    if (scale) |sc| {
        var sum = try mlx.Array.add(x, residual, s);
        defer sum.deinit();
        return mlx.Array.multiply(sum, sc, s);
    }
    return mlx.Array.add(x, residual, s);
}

/// 最终隐状态提取与均方根层归一化组合块：
/// 若指定了 rows，则截取序列尾部的 rows 个词元重塑为 [rows, hidden_size] 后归一化；否则对 x 直接归一化
pub fn finalHidden(
    x: mlx.Array,
    final_norm_w: mlx.Array,
    eps: f32,
    rows: ?usize,
    hidden_size: usize,
    s: ?Stream,
) !mlx.Array {
    if (rows) |r| {
        const t = x.dimSize(1);
        var tail = try x.sliceAxis(1, t - r, t, s);
        defer tail.deinit();
        var tail_2d = try tail.reshape(&.{ r, hidden_size }, s);
        defer tail_2d.deinit();
        return mlx.Array.fastRmsNorm(tail_2d, final_norm_w, eps, s);
    }
    return mlx.Array.fastRmsNorm(x, final_norm_w, eps, s);
}
