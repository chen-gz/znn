//! MLX GPU 组合原语 (MLX GPU Composite Primitives)
//!
//! 提取模型无关的高阶 Apple MLX 硬件加速组合块：
//! - 旋转位置编码 (`Rope`)：统一支持标准 RoPE 与比例旋转位置编码
//! - 多头归一化与旋转 (`projectHeads`)：统一单步解码 (t=1) 与预填充批处理 (t>1) 的 Q/K/V 投影归一化、转置与 RoPE 旋转
//! - 门控多层感知机 (`gatedMlp`)：统一门控前馈网络（支持融合 gate_up 与独立 gate/up，可选前后 RMSNorm）
//! - 残差相加与缩放 (`residualScale`)：统一 `(x + residual) * scale` 或 `x + residual`
//! - 最终隐状态提取与归一化 (`finalHidden`)：统一末尾行切片与最终均方根归一化

const std = @import("std");
const backend = @import("../backend.zig");
const mlx = backend.mlx;
const activation = backend.activation;
const offload = backend.offload;
const rope_mod = @import("../tensor/rope.zig");
const tensor = @import("../tensor.zig");

pub const Stream = mlx.Stream;
pub const QuantWeights = offload.QuantWeights;
pub const LmHead = offload.LmHead;
pub const LmHeadWeights = offload.LmHeadWeights;

/// 词表加性排除掩码 (Additive Token Exclusion Mask)：[vocab_size] 的 BF16 常驻数组，排除词元为 -inf，其余为 0。
/// 与对数几率相加后再求 argmax 或选取随机采样候选，使 GPU 路径与 CPU 排除完全相同的词元集合。
pub const TokenExclusionMask = struct {
    mask: ?mlx.Array = null,
    vocab_size: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        vocab_size: usize,
        excluded: []const tensor.TokenRange,
    ) !TokenExclusionMask {
        if (!offload.isAvailable()) return error.OffloadNotSupported;
        if (vocab_size == 0) return error.InvalidShape;
        var has_excluded = false;
        for (excluded) |r| {
            if (r.intersects(0, vocab_size)) {
                has_excluded = true;
                break;
            }
        }
        if (!has_excluded) return .{ .vocab_size = vocab_size };
        const bits = try allocator.alloc(u16, vocab_size);
        defer allocator.free(bits);
        @memset(bits, 0x0000);
        for (excluded) |r| {
            if (r.first >= bits.len) continue;
            const end = @min(r.last + 1, bits.len);
            @memset(bits[r.first..end], 0xFF80);
        }
        const mask = try mlx.Array.fromSliceBf16(bits, &.{vocab_size});
        return .{ .mask = mask, .vocab_size = vocab_size };
    }

    pub fn deinit(self: *TokenExclusionMask) void {
        if (self.mask) |*m| m.deinit();
        self.mask = null;
    }

    pub fn apply(self: *const TokenExclusionMask, logits: mlx.Array, stream: ?Stream) !mlx.Array {
        if (!offload.isAvailable()) return error.OffloadNotSupported;
        const nd = logits.ndim();
        if (nd == 0 or logits.dimSize(@intCast(nd - 1)) != self.vocab_size) return error.ShapeMismatch;
        const m = self.mask orelse return logits.share();
        return mlx.Array.add(logits, m, stream);
    }

    pub fn argmax(self: *const TokenExclusionMask, logits: mlx.Array, stream: ?Stream) !mlx.Array {
        var masked = try self.apply(logits, stream);
        defer masked.deinit();
        return masked.argmax(@intCast(logits.ndim() - 1), false, stream);
    }
};

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

pub const LogitsProcessorOptions = struct {
    vocab_size: usize,
    final_logit_softcapping: f32 = 0.0,
    sampling_excluded_tokens: []const tensor.TokenRange = &.{},

    pub fn defaultOptions(vocab_size: usize) LogitsProcessorOptions {
        return .{
            .vocab_size = vocab_size,
            .final_logit_softcapping = 0.0,
            .sampling_excluded_tokens = &.{},
        };
    }

    pub fn of(cfg: anytype) LogitsProcessorOptions {
        if (@TypeOf(cfg) == LogitsProcessorOptions) return cfg;
        return .{
            .vocab_size = cfg.vocab_size,
            .final_logit_softcapping = if (@hasField(@TypeOf(cfg), "final_logit_softcapping")) cfg.final_logit_softcapping else 0.0,
            .sampling_excluded_tokens = if (@hasField(@TypeOf(cfg), "sampling_excluded_tokens")) cfg.sampling_excluded_tokens else &.{},
        };
    }
};

pub fn softcapLogits(logits: mlx.Array, cap: f32, s: ?Stream) !mlx.Array {
    var logits_f32 = try logits.astype(.float32, s);
    if (cap <= 0.0) return logits_f32;
    defer logits_f32.deinit();
    var scaled = try logits_f32.mulScalar(1.0 / cap, s);
    defer scaled.deinit();
    var tanh_l = try mlx.Array.tanh(scaled, s);
    defer tanh_l.deinit();
    return tanh_l.mulScalar(cap, s);
}

pub fn penalizeDevice(
    allocator: std.mem.Allocator,
    logits: mlx.Array,
    vocab_size: usize,
    recent_tokens: []const usize,
    repetition_penalty: f32,
    s: ?Stream,
) !?mlx.Array {
    if (!tensor.penaltyActive(repetition_penalty)) return null;
    var count: usize = 0;
    for (recent_tokens) |tok| count += @intFromBool(tok < vocab_size);
    if (count == 0) return null;

    const ids = try allocator.alloc(i32, count);
    defer allocator.free(ids);
    var n: usize = 0;
    for (recent_tokens) |tok| {
        if (tok >= vocab_size) continue;
        ids[n] = @intCast(tok);
        n += 1;
    }
    var idx = try mlx.Array.fromSliceI32(ids, &.{ 1, count });
    defer idx.deinit();
    var gathered = try logits.takeAlongAxis(idx, -1, s);
    defer gathered.deinit();
    var shrunk = try gathered.mulScalar(1.0 / repetition_penalty, s);
    defer shrunk.deinit();
    var grown = try gathered.mulScalar(repetition_penalty, s);
    defer grown.deinit();
    var zero = try mlx.Array.fromScalarF32(0.0);
    defer zero.deinit();
    var positive = try mlx.Array.greater(gathered, zero, s);
    defer positive.deinit();
    var penalized = try mlx.Array.where(positive, shrunk, grown, s);
    defer penalized.deinit();
    return try logits.putAlongAxis(idx, penalized, -1, s);
}

pub const DeviceCandidates = struct {
    ids: mlx.Array,
    logits: mlx.Array,
    log_normalizer: ?mlx.Array = null,

    pub fn deinit(self: *DeviceCandidates) void {
        self.ids.deinit();
        self.logits.deinit();
        if (self.log_normalizer) |*a| a.deinit();
        self.log_normalizer = null;
    }

    pub fn len(self: *const DeviceCandidates) usize {
        return self.ids.dimSize(1);
    }

    pub fn arrays(self: *const DeviceCandidates, out: []mlx.Array) usize {
        out[0] = self.ids;
        out[1] = self.logits;
        if (self.log_normalizer) |a| {
            out[2] = a;
            return 3;
        }
        return 2;
    }

    pub fn read(self: *const DeviceCandidates, out: []tensor.Candidate) !HostCandidates {
        const ids = try self.ids.dataUint32();
        const vals = try self.logits.dataFloat32();
        if (ids.len != vals.len or ids.len > out.len) return error.ShapeMismatch;
        for (out[0..ids.len], ids, vals) |*cand, id, v| cand.* = .{ .id = id, .logit = v };
        const lse: ?f32 = if (self.log_normalizer) |a| (try a.dataFloat32())[0] else null;
        return .{ .candidates = out[0..ids.len], .log_normalizer = lse };
    }
};

pub const HostCandidates = struct {
    candidates: []tensor.Candidate,
    log_normalizer: ?f32,
};

pub fn deviceCandidates(
    allocator: std.mem.Allocator,
    logits: mlx.Array,
    opts: anytype,
    mask: *const TokenExclusionMask,
    recent_tokens: []const usize,
    sampling_opts: tensor.SamplingOptions,
    s: ?Stream,
) !DeviceCandidates {
    const opt_proc = LogitsProcessorOptions.of(opts);
    if (logits.ndim() != 2 or logits.dimSize(0) != 1 or logits.dimSize(1) != opt_proc.vocab_size) return error.ShapeMismatch;
    const recent = if (sampling_opts.repetition_window) |win| sampling_opts.recentTokens(recent_tokens, win) else recent_tokens;
    var capped = try softcapLogits(logits, opt_proc.final_logit_softcapping, s);
    defer capped.deinit();
    var penalized_opt = try penalizeDevice(allocator, capped, opt_proc.vocab_size, recent, sampling_opts.repetition_penalty, s);
    defer if (penalized_opt) |*a| a.deinit();
    var masked = try mask.apply(penalized_opt orelse capped, s);
    defer masked.deinit();

    const k = sampling_opts.candidateCount(opt_proc.vocab_size);
    var neg = try masked.negative(s);
    defer neg.deinit();
    var order = try neg.argpartitionAxis(k - 1, -1, s);
    defer order.deinit();
    var top_view = try order.sliceAxis(1, 0, k, s);
    defer top_view.deinit();
    var ids = try top_view.contiguous(s);
    errdefer ids.deinit();
    var vals = try masked.takeAlongAxis(ids, -1, s);
    errdefer vals.deinit();

    var lse: ?mlx.Array = null;
    if (sampling_opts.needsLogNormalizer()) {
        var scaled = try masked.mulScalar(1.0 / sampling_opts.temperature, s);
        defer scaled.deinit();
        lse = try scaled.logsumexpAxis(-1, false, s);
    }
    return .{ .ids = ids, .logits = vals, .log_normalizer = lse };
}

pub fn selectToken(
    allocator: std.mem.Allocator,
    logits: mlx.Array,
    opts: anytype,
    mask: *const TokenExclusionMask,
    recent_tokens: []const usize,
    sampler: anytype,
    s: ?Stream,
) !usize {
    const opt_proc = LogitsProcessorOptions.of(opts);
    const s_opts = sampler.options;
    const recent = if (s_opts.repetition_window) |win| s_opts.recentTokens(recent_tokens, win) else recent_tokens;
    if (s_opts.isPlainGreedy()) {
        var best = try mask.argmax(logits, s);
        defer best.deinit();
        return @intCast(try best.itemUint32());
    }
    if (s_opts.isGreedy()) {
        var capped = try softcapLogits(logits, opt_proc.final_logit_softcapping, s);
        defer capped.deinit();
        const data = try capped.dataFloat32();
        if (data.len != opt_proc.vocab_size) return error.ShapeMismatch;
        return tensor.argmaxWithPenalty(data, opt_proc.vocab_size, opt_proc.sampling_excluded_tokens, recent, s_opts.repetition_penalty);
    }

    var pending = try deviceCandidates(allocator, logits, opt_proc, mask, recent, s_opts, s);
    defer pending.deinit();
    const buf = try allocator.alloc(tensor.Candidate, pending.len());
    defer allocator.free(buf);
    const host = try pending.read(buf);
    return sampler.sampleCandidates(host.candidates, host.log_normalizer);
}

pub fn selectTokenFromHidden(
    allocator: std.mem.Allocator,
    lm_head: anytype,
    opts: anytype,
    mask: *const TokenExclusionMask,
    h_last: []const f32,
    recent_tokens: []const usize,
    sampler: anytype,
    s: ?Stream,
) !usize {
    const head: LmHead = switch (@TypeOf(lm_head.*)) {
        LmHead => lm_head.*,
        LmHeadWeights => .{ .dense = lm_head.* },
        QuantWeights => .{ .quant = lm_head.* },
        else => @compileError("Unsupported lm_head type"),
    };
    const hid = head.hiddenSize();
    const voc = head.vocabSize();
    const opt_proc = LogitsProcessorOptions.of(opts);
    if (h_last.len != hid or voc != opt_proc.vocab_size) return error.ShapeMismatch;
    var x = try mlx.Array.fromSlice(h_last, &.{ 1, hid });
    defer x.deinit();
    var logits = try head.forwardArray(x, s);
    defer logits.deinit();
    return selectToken(allocator, logits, opt_proc, mask, recent_tokens, sampler, s);
}

