//! 硬件加速统一卸载基础设施 (Hardware Offload Infrastructure)
//!
//! 将计算密集型算子（通用矩阵乘法 GEMM、均方根层归一化 RMSNorm、旋转位置编码 RoPE、
//! 缩放点积注意力 SDPA / FlashAttention、语言模型预测头 LM Head 巨型词表投影）
//! 统一卸载至 Apple 机器学习拓展 (Machine Learning Explore, MLX) 后端在图形处理器 (GPU) 上执行。
//! 遵循分层规范，本模块位于最底层的 backend 包（层级 0），对外暴露解耦的硬件加速卸载入口。

const std = @import("std");
const mlx = @import("mlx.zig");
const activation = @import("activation.zig");
const attention_mask = @import("attention_mask.zig");
const rope_mod = @import("rope.zig");
const sampling = @import("sampling.zig");

/// 当前硬件环境是否支持统一卸载 (macOS Apple Silicon aarch64 且已集成 MLX)
pub fn isAvailable() bool {
    return mlx.has_mlx;
}

pub const Stream = mlx.Stream;
pub const DeviceType = mlx.DeviceType;
pub const Array = mlx.Array;
pub const DeviceArray = mlx.Array;
pub const VectorArray = mlx.VectorArray;
pub const Closure = mlx.Closure;

/// 硬件内存优化与编译控制
pub const configureOptimalMemory = mlx.configureOptimalWiredMemory;
pub const enableCompile = mlx.enableCompile;
pub const setCompileMode = mlx.setCompileMode;

/// 硬件加速算子与内核接口
pub const compileFn = mlx.compileFn;
pub const compileGeluMul = mlx.Array.compileGeluMul;
pub const compileAddMul = mlx.Array.compileAddMul;
pub const fastRmsNorm = mlx.Array.fastRmsNorm;
pub const fastScaledDotProductAttention = mlx.Array.fastScaledDotProductAttention;
pub const fusedGeluMul = mlx.Array.fusedGeluMul;
pub const fusedAddMul = mlx.Array.fusedAddMul;
pub const evalMultiple = mlx.Array.evalMultiple;

pub const OffloadError = mlx.MlxError || error{
    OffloadNotSupported,
    UnsupportedQuantMode,
    InvalidShape,
    ShapeMismatch,
    NonContiguousTensor,
    EmptyTensor,
};

/// 借用或包装数组句柄守卫
pub const ArrayBorrow = struct {
    arr: mlx.Array,
    needs_deinit: bool,

    pub fn deinit(self: *ArrayBorrow) void {
        if (self.needs_deinit) {
            self.arr.deinit();
        }
    }
};

/// 从输入张量或数组借用 MLX 数组句柄 (若已在 GPU 驻留则零开销借用；若在主机内存则零拷贝包装)
pub fn borrowArray(t: anytype) OffloadError!ArrayBorrow {
    const T = @TypeOf(t);
    if (T == mlx.Array) {
        return .{ .arr = t, .needs_deinit = false };
    }
    const info = @typeInfo(T);
    if (info == .pointer and @hasField(info.pointer.child, "device_array")) {
        if (t.device_array) |arr| {
            return .{ .arr = arr, .needs_deinit = false };
        }
        if (t.data.len > 0) {
            const arr = try mlx.Array.fromTensorZeroCopy(t);
            return .{ .arr = arr, .needs_deinit = true };
        }
        return error.EmptyTensor;
    }
    const arr = try mlx.Array.fromTensorZeroCopy(t);
    return .{ .arr = arr, .needs_deinit = true };
}

/// 将计算结果安全存入输出张量 (若输出为 GPU 驻留张量则直接接管句柄；若为 CPU 主机张量则拷回内存)
pub fn storeOutput(out: anytype, res_mlx: mlx.Array) OffloadError!void {
    const T = @TypeOf(out);
    const info = @typeInfo(T);
    if (info == .pointer and @hasField(info.pointer.child, "device_array")) {
        if (out.device == .gpu or out.data.len == 0) {
            if (out.device_array) |*old| old.deinit();
            out.device_array = res_mlx;
            out.device = .gpu;
            return;
        }
    }
    var to_free = res_mlx;
    defer to_free.deinit();
    try res_mlx.copyToTensor(out);
}

/// 通用矩阵乘法硬件卸载 (General Matrix Multiply Offload, GEMM Offload)：C = A * B
/// 支持二维矩阵乘法以及高维张量批处理矩阵乘法 (Batch MatMul)
pub fn matmul(
    A: anytype,
    B: anytype,
    C: anytype,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var a_b = try borrowArray(A);
    defer a_b.deinit();
    var b_b = try borrowArray(B);
    defer b_b.deinit();

    var c_mlx = try mlx.Array.matmul(a_b.arr, b_b.arr, stream);
    errdefer c_mlx.deinit();

    try storeOutput(C, c_mlx);
}

/// 快速均方根层归一化硬件卸载 (Root Mean Square Layer Normalization Offload, RMSNorm Offload)
pub fn rmsNorm(
    X: anytype,
    weight: anytype,
    out: anytype,
    eps: f32,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var x_b = try borrowArray(X);
    defer x_b.deinit();

    var w_b: ?ArrayBorrow = null;
    defer if (w_b) |*w| w.deinit();
    const WT = @TypeOf(weight);
    if (@typeInfo(WT) == .optional) {
        if (weight) |w| {
            w_b = try borrowArray(w);
        }
    } else {
        w_b = try borrowArray(weight);
    }
    const D = X.shape.dims[X.shape.len - 1];
    var w_reshaped: ?mlx.Array = null;
    defer if (w_reshaped) |*wr| wr.deinit();

    const w_arr: ?mlx.Array = if (w_b) |w| blk: {
        if (w.arr.ndim() != 1) {
            w_reshaped = try w.arr.reshape(&.{D}, stream);
            break :blk w_reshaped.?;
        }
        break :blk w.arr;
    } else null;

    var out_mlx = try mlx.Array.fastRmsNorm(x_b.arr, w_arr, eps, stream);
    errdefer out_mlx.deinit();

    try storeOutput(out, out_mlx);
}

/// 快速旋转位置编码硬件卸载 (Rotary Position Embedding Offload, RoPE Offload)：对 `[..., T, head_dim]` 张量的每一行
/// 按共享配置 `RopeOptions` 旋转，第 t 行的绝对位置为 start_pos + t。部分旋转 / 比例旋转经共享周期表 (`fillPeriods`)
/// 交给 MLX `fast.rope`；旋转子空间带偏移 (rotary_offset != 0)、head_dim 为奇数或超出周期表容量时返回
/// `error.OffloadNotSupported` (调用方回退 CPU 内核)
pub fn rope(
    X: anytype,
    out: anytype,
    options: rope_mod.RopeOptions,
    start_pos: usize,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    if (!ropeSupported(X.shape.dims[X.shape.len - 1], options)) return error.OffloadNotSupported;
    const head_dim = X.shape.dims[X.shape.len - 1];
    const traditional = options.mode == .interleaved;
    var x_b = try borrowArray(X);
    defer x_b.deinit();

    var y_mlx = if (options.numAngles(head_dim) == head_dim / 2)
        try mlx.Array.fastRoPE(x_b.arr, head_dim, traditional, options.base, 1.0, start_pos, stream)
    else blk: {
        var periods: [rope_mod.max_angles]f32 = undefined;
        options.fillPeriods(head_dim, periods[0 .. head_dim / 2]);
        var freqs = try mlx.Array.fromSlice(periods[0 .. head_dim / 2], &.{head_dim / 2});
        defer freqs.deinit();
        break :blk try mlx.Array.fastRoPEWithFreqs(x_b.arr, head_dim, traditional, 1.0, start_pos, freqs, stream);
    };
    errdefer y_mlx.deinit();

    try storeOutput(out, y_mlx);
}

/// `rope` 能否在 MLX 上执行：旋转整行 (rotary_offset == 0) 的偶数 head_dim，且周期表不超过 `rope_mod.max_angles`
pub fn ropeSupported(head_dim: usize, options: rope_mod.RopeOptions) bool {
    return options.rotary_offset == 0 and head_dim > 0 and head_dim % 2 == 0 and head_dim / 2 <= rope_mod.max_angles;
}

/// 快速缩放点积注意力硬件卸载 (Scaled Dot-Product Attention Offload, SDPA Offload / FlashAttention)：
/// q [B, q_heads, tq, d]、k [B, kv_heads, tk, d]、v [B, kv_heads, tk, dv] (分组查询注意力 (Grouped-Query Attention, GQA)
/// 由融合内核原生处理)，掩码与位置语义见 `maskedAttention`，结果写入 out [B, q_heads, tq, dv]
pub fn attention(
    allocator: std.mem.Allocator,
    queries: anytype,
    keys: anytype,
    values: anytype,
    out: anytype,
    scale: f32,
    mask: attention_mask.Mask,
    q_pos0: usize,
    k_pos0: usize,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var q_b = try borrowArray(queries);
    defer q_b.deinit();
    var k_b = try borrowArray(keys);
    defer k_b.deinit();
    var v_b = try borrowArray(values);
    defer v_b.deinit();

    var out_mlx = try maskedAttention(allocator, q_b.arr, k_b.arr, v_b.arr, scale, mask, q_pos0, k_pos0, stream);
    errdefer out_mlx.deinit();

    try storeOutput(out, out_mlx);
}

/// 归一化指数函数硬件卸载 (Softmax Offload)
pub fn softmax(
    A: anytype,
    out: anytype,
    axis: c_int,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var a_b = try borrowArray(A);
    defer a_b.deinit();

    var res_mlx = try mlx.Array.softmax(a_b.arr, axis, stream);
    errdefer res_mlx.deinit();

    try storeOutput(out, res_mlx);
}

/// 逐元素加法硬件卸载
pub fn add(
    A: anytype,
    B: anytype,
    C: anytype,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var a_b = try borrowArray(A);
    defer a_b.deinit();
    var b_b = try borrowArray(B);
    defer b_b.deinit();

    var c_mlx = try mlx.Array.add(a_b.arr, b_b.arr, stream);
    errdefer c_mlx.deinit();

    try storeOutput(C, c_mlx);
}

/// 逐元素乘法硬件卸载
pub fn multiply(
    A: anytype,
    B: anytype,
    C: anytype,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var a_b = try borrowArray(A);
    defer a_b.deinit();
    var b_b = try borrowArray(B);
    defer b_b.deinit();

    var c_mlx = try mlx.Array.multiply(a_b.arr, b_b.arr, stream);
    errdefer c_mlx.deinit();

    try storeOutput(C, c_mlx);
}

/// 标量乘法硬件卸载
pub fn mulScalar(
    A: anytype,
    scalar: f32,
    C: anytype,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var a_b = try borrowArray(A);
    defer a_b.deinit();

    var c_mlx = try mlx.Array.mulScalar(a_b.arr, scalar, stream);
    errdefer c_mlx.deinit();

    try storeOutput(C, c_mlx);
}

/// 高斯误差线性单元硬件卸载 (GELU Offload)
pub fn gelu(
    X: anytype,
    out: anytype,
    approx: activation.GeluApproximation,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var x_b = try borrowArray(X);
    defer x_b.deinit();

    var out_mlx = try x_b.arr.gelu(approx, stream);
    errdefer out_mlx.deinit();

    try storeOutput(out, out_mlx);
}

/// 语言模型输出头 (Language Model Head, LM Head) 常驻词表矩阵的量化模式
pub const QuantMode = enum {
    bf16,
    q8,
    q4,

    /// 仿射量化参数 (分组长度与位宽)
    pub const Params = struct {
        group_size: i32,
        bits: i32,
    };

    /// 返回量化模式对应的仿射量化参数；BF16 不量化，返回 null
    pub fn params(self: QuantMode) ?Params {
        return switch (self) {
            .bf16 => null,
            .q8 => .{ .group_size = 64, .bits = 8 },
            .q4 => .{ .group_size = 32, .bits = 4 },
        };
    }
};

/// 语言模型输出头权重 (Language Model Head Weights, LM Head Weights)
/// 预先将巨型词表嵌入矩阵 (如 262,144 维词表) 在 GPU 显存中常驻 (BF16 转置或仿射量化)，
/// 自回归解码每一步仅需下发单步输入向量 [1, hidden_size]，利用统一内存实现 5~10 ms 极速词表打分。
/// 本类型只负责词表投影；截断、重复惩罚与贪婪采样等模型相关策略由上层模型实现。
pub const LmHeadWeights = struct {
    w_t: ?mlx.Array = null,
    w_q: ?mlx.Array = null,
    scales: ?mlx.Array = null,
    biases: ?mlx.Array = null,
    is_quantized: bool = false,
    group_size: i32 = 32,
    bits: i32 = 4,
    vocab_size: usize,
    hidden_size: usize,

    /// 校验量化模式能否作用于 [vocab_size, hidden_size] 词表矩阵：量化模式要求 hidden_size 同时整除分组长度
    /// 与每个 uint32 打包的元素数，不满足时返回 `error.UnsupportedQuantMode`（不会静默退回 BF16）
    pub fn validateMode(mode: QuantMode, vocab_size: usize, hidden_size: usize) OffloadError!void {
        if (vocab_size == 0 or hidden_size == 0) return error.InvalidShape;
        const p = mode.params() orelse return;
        _ = QuantLayout.init(hidden_size, vocab_size, p.group_size, p.bits, .bfloat16) catch return error.UnsupportedQuantMode;
    }

    /// 指定量化模式构建常驻 GPU 词表矩阵 (支持 BF16、Q8 与 Q4)；量化模式与词表形状不兼容时返回 `error.UnsupportedQuantMode`
    pub fn initWithMode(
        embed_u16: []const u16,
        vocab_size: usize,
        hidden_size: usize,
        mode: QuantMode,
        stream: ?Stream,
    ) OffloadError!LmHeadWeights {
        try validateMode(mode, vocab_size, hidden_size);
        if (!isAvailable()) return error.OffloadNotSupported;
        var w = try mlx.Array.fromSliceBf16(embed_u16, &.{ vocab_size, hidden_size });
        defer w.deinit();

        if (mode.params()) |p| {
            var pkg = try mlx.Array.quantize(w, p.group_size, p.bits, "affine", stream);
            errdefer pkg.deinit();
            pkg.weights.eval();
            pkg.scales.eval();
            if (pkg.biases) |*b| b.eval();
            if (stream) |st| st.synchronize();

            return .{
                .w_q = pkg.weights,
                .scales = pkg.scales,
                .biases = pkg.biases,
                .is_quantized = true,
                .group_size = p.group_size,
                .bits = p.bits,
                .vocab_size = vocab_size,
                .hidden_size = hidden_size,
            };
        }

        const w_t = try w.transpose(stream);
        return .{
            .w_t = w_t,
            .is_quantized = false,
            .vocab_size = vocab_size,
            .hidden_size = hidden_size,
        };
    }

    /// 从脑浮点 16 (Brain Floating Point 16, BF16) 词表权重构建常驻 GPU 词表矩阵 (默认标准 BF16 硬件直推，保证 100% 原始输出一致性)
    pub fn init(
        embed_u16: []const u16,
        vocab_size: usize,
        hidden_size: usize,
        stream: ?Stream,
    ) OffloadError!LmHeadWeights {
        return initWithMode(embed_u16, vocab_size, hidden_size, .bf16, stream);
    }

    /// 释放 GPU 常驻权重资源
    pub fn deinit(self: *LmHeadWeights) void {
        if (self.w_t) |*wt| wt.deinit();
        if (self.w_q) |*wq| wq.deinit();
        if (self.scales) |*sc| sc.deinit();
        if (self.biases) |*b| b.deinit();
        self.w_t = null;
        self.w_q = null;
        self.scales = null;
        self.biases = null;
    }

    /// 执行 GPU 词表打分投影计算图调度并保留原生脑浮点 16 (Brain Floating Point 16, BF16) 输出：[..., hidden_size] -> [..., vocab_size]
    pub fn forwardArrayBf16(self: LmHeadWeights, x: mlx.Array, stream: ?Stream) OffloadError!mlx.Array {
        if (!isAvailable()) return error.OffloadNotSupported;
        var x_bf16 = try x.astype(.bfloat16, stream);
        defer x_bf16.deinit();
        if (self.is_quantized) {
            return x_bf16.quantizedMatmul(
                self.w_q.?,
                self.scales.?,
                self.biases,
                true, // transpose = true -> x @ W^T
                self.group_size,
                self.bits,
                "affine",
                stream,
            );
        } else {
            return mlx.Array.matmul(x_bf16, self.w_t.?, stream);
        }
    }

    /// 执行 GPU 词表打分投影计算图调度：[..., hidden_size] -> [..., vocab_size] (单精度输出)
    pub fn forwardArray(self: LmHeadWeights, x: mlx.Array, stream: ?Stream) OffloadError!mlx.Array {
        var logits_bf16 = try self.forwardArrayBf16(x, stream);
        defer logits_bf16.deinit();
        return logits_bf16.astype(.float32, stream);
    }
};

/// 词表加性排除掩码 (Additive Token Exclusion Mask)：[vocab_size] 的 BF16 常驻数组，排除词元为 -inf，其余为 0。
/// 与对数几率相加后再求 argmax 或选取随机采样候选，使 GPU 路径与 CPU `sampling.argmaxWithPenalty` /
/// `sampling.maskExcludedInPlace` 排除完全相同的词元集合。
pub const TokenExclusionMask = struct {
    /// 排除集合与词表不相交时为 null，直接使用原始对数几率
    mask: ?mlx.Array = null,
    vocab_size: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        vocab_size: usize,
        excluded: []const sampling.TokenRange,
    ) OffloadError!TokenExclusionMask {
        if (!isAvailable()) return error.OffloadNotSupported;
        if (vocab_size == 0) return error.InvalidShape;
        if (!sampling.anyExcluded(excluded, 0, vocab_size)) return .{ .vocab_size = vocab_size };
        const bits = try allocator.alloc(u16, vocab_size);
        defer allocator.free(bits);
        sampling.fillExclusionMaskBf16(bits, excluded);
        // mlx_array_new_data 立即复制数据，临时缓冲区可在返回前释放
        const mask = try mlx.Array.fromSliceBf16(bits, &.{vocab_size});
        return .{ .mask = mask, .vocab_size = vocab_size };
    }

    pub fn deinit(self: *TokenExclusionMask) void {
        if (self.mask) |*m| m.deinit();
        self.mask = null;
    }

    /// 对 logits [..., vocab_size] 叠加排除掩码，返回新数组 (排除词元为 -inf)；排除集合与词表不相交时返回共享同一数据的新句柄
    pub fn apply(self: *const TokenExclusionMask, logits: mlx.Array, stream: ?Stream) OffloadError!mlx.Array {
        if (!isAvailable()) return error.OffloadNotSupported;
        const nd = logits.ndim();
        if (nd == 0 or logits.dimSize(@intCast(nd - 1)) != self.vocab_size) return error.ShapeMismatch;
        const m = self.mask orelse return logits.share();
        return mlx.Array.add(logits, m, stream);
    }

    /// 沿最后一轴对 logits [..., vocab_size] 求未被排除词元中的最大值下标 (uint32，形状去掉最后一轴)
    pub fn argmax(self: *const TokenExclusionMask, logits: mlx.Array, stream: ?Stream) OffloadError!mlx.Array {
        var masked = try self.apply(logits, stream);
        defer masked.deinit();
        return masked.argmax(@intCast(logits.ndim() - 1), false, stream);
    }
};

/// MLX 仿射量化权重的内存布局：w_q 为 [out, in * bits / 32] 的 uint32 紧凑数组，scales / biases 为 [out, in / group_size]
pub const QuantLayout = struct {
    w_dims: [2]usize,
    sb_dims: [2]usize,
    w_q_bytes: usize,
    sb_bytes: usize,

    pub fn totalBytes(self: QuantLayout) usize {
        return self.w_q_bytes + self.sb_bytes * 2;
    }

    /// 校验量化参数（位宽须整除 32，输入维度须同时整除分组长度与每个 uint32 打包的元素数）并计算各缓冲区形状与字节数
    pub fn init(in_features: usize, out_features: usize, group_size: i32, bits: i32, sb_dtype: mlx.DType) OffloadError!QuantLayout {
        if (bits <= 0 or bits > 32 or @rem(32, bits) != 0 or group_size <= 0) return error.InvalidShape;
        const per_word: usize = @intCast(@divExact(32, bits));
        const group: usize = @intCast(group_size);
        if (in_features % per_word != 0 or in_features % group != 0) return error.InvalidShape;
        const sb_elem_bytes: usize = switch (sb_dtype) {
            .float16, .bfloat16 => 2,
            .float32 => 4,
            else => return error.InvalidShape,
        };
        const w_dims = [2]usize{ out_features, in_features / per_word };
        const sb_dims = [2]usize{ out_features, in_features / group };
        return .{
            .w_dims = w_dims,
            .sb_dims = sb_dims,
            .w_q_bytes = w_dims[0] * w_dims[1] * @sizeOf(u32),
            .sb_bytes = sb_dims[0] * sb_dims[1] * sb_elem_bytes,
        };
    }

    pub fn initDefault(in_features: usize, out_features: usize, group_size: i32, bits: i32) QuantLayout {
        return init(in_features, out_features, group_size, bits, .bfloat16) catch unreachable;
    }
};

/// 将多个字节切片依次串联写入 dst 开头，总长度必须恰好等于 expected_bytes
fn concatInto(dst: []u8, slices: []const []const u8, expected_bytes: usize) OffloadError!void {
    var offset: usize = 0;
    for (slices) |part| {
        if (part.len > dst.len - offset) return error.ShapeMismatch;
        @memcpy(dst[offset..][0..part.len], part);
        offset += part.len;
    }
    if (offset != expected_bytes) return error.ShapeMismatch;
}

/// 统一分块量化矩阵乘法硬件加速权重 (Quantized Matrix Multiply Weights, QuantWeights)
/// 统一覆盖 4-bit (Q4) 与 8-bit (Q8) 等仿射量化权重，打包为 Apple MLX 原生 uint32 紧凑排列常驻 GPU 显存。
pub const QuantWeights = struct {
    w_q: mlx.Array,
    scales: mlx.Array,
    biases: mlx.Array,
    in_features: usize,
    out_features: usize,
    group_size: i32 = 32,
    bits: i32 = 4,

    /// 沿第 0 轴根据词元索引向量提取对应行并执行反量化与可选标量缩放：返回 [indices.len, in_features]
    pub fn gatherRows(
        self: *const QuantWeights,
        ids: mlx.Array,
        scale: ?mlx.Array,
        s: ?Stream,
    ) OffloadError!mlx.Array {
        var w_sel = try self.w_q.takeAxis(ids, 0, s);
        defer w_sel.deinit();
        var sc_sel = try self.scales.takeAxis(ids, 0, s);
        defer sc_sel.deinit();
        var bi_sel = try self.biases.takeAxis(ids, 0, s);
        defer bi_sel.deinit();

        var deq = try mlx.Array.dequantize(
            w_sel,
            sc_sel,
            bi_sel,
            self.group_size,
            self.bits,
            "affine",
            s,
        );
        if (scale) |sc| {
            defer deq.deinit();
            return mlx.Array.multiply(deq, sc, s);
        }
        return deq;
    }

    /// 将 18 字节/块的原始量化切片解包并重排写入直接内存缓冲区 (w_q, scales, biases)
    pub fn packBlocksToDirectBuffers(
        block_slices: []const []const u8,
        out_features_list: []const usize,
        in_features: usize,
        w_q_out: []u32,
        scales_out: []f16,
        biases_out: []f16,
    ) void {
        const blocks_per_row = in_features / 32;
        var global_blk_idx: usize = 0;
        for (block_slices, 0..) |slice, s_idx| {
            const slice_blocks = out_features_list[s_idx] * blocks_per_row;
            for (0..slice_blocks) |b_idx| {
                const raw = slice[b_idx * 18 .. (b_idx + 1) * 18];
                const scale_bits = std.mem.readInt(u16, raw[0..2][0..2], .little);
                const scale_f16: f16 = @bitCast(scale_bits);
                const scale_f32: f32 = @floatCast(scale_f16);
                const bias_f16: f16 = @floatCast(-8.0 * scale_f32);

                scales_out[global_blk_idx] = scale_f16;
                biases_out[global_blk_idx] = bias_f16;

                const qs = raw[2..18];
                var w0: u32 = 0;
                var w1: u32 = 0;
                var w2: u32 = 0;
                var w3: u32 = 0;
                for (0..8) |j| {
                    const shift: u5 = @intCast(j * 4);
                    w0 |= @as(u32, qs[j] & 0x0F) << shift;
                    w1 |= @as(u32, qs[j + 8] & 0x0F) << shift;
                    w2 |= @as(u32, qs[j] >> 4) << shift;
                    w3 |= @as(u32, qs[j + 8] >> 4) << shift;
                }
                w_q_out[global_blk_idx * 4 + 0] = w0;
                w_q_out[global_blk_idx * 4 + 1] = w1;
                w_q_out[global_blk_idx * 4 + 2] = w2;
                w_q_out[global_blk_idx * 4 + 3] = w3;

                global_blk_idx += 1;
            }
        }
    }

    /// 从预重排直接内存缓冲区构建 GPU 量化权重 (Direct Layout Fast Path)
    /// 无需任何运行时 CPU 解包与动态内存分配，直接将预重排内存映射至 MLX 显存，实现毫秒级加载
    pub fn initFromDirectBuffers(
        w_q_bytes: []const u8,
        scales_bytes: []const u8,
        biases_bytes: []const u8,
        in_features: usize,
        out_features: usize,
        stream: ?Stream,
    ) OffloadError!Q4Weights {
        return initFromDirectBuffersCustom(
            w_q_bytes,
            scales_bytes,
            biases_bytes,
            in_features,
            out_features,
            32,
            4,
            .float16,
            stream,
        );
    }

    /// 从任意量化配置（支持 4-bit / 8-bit 及 float16 / bfloat16 缩放偏移）直接构建 GPU 量化权重
    pub fn initFromDirectBuffersCustom(
        w_q_bytes: []const u8,
        scales_bytes: []const u8,
        biases_bytes: []const u8,
        in_features: usize,
        out_features: usize,
        group_size: i32,
        bits: i32,
        sb_dtype: mlx.DType,
        stream: ?Stream,
    ) OffloadError!Q4Weights {
        _ = stream;
        if (!isAvailable()) return error.OffloadNotSupported;

        const layout = try QuantLayout.init(in_features, out_features, group_size, bits, sb_dtype);
        // mlx_array_new_data 按形状从原始指针复制数据：缓冲区不足时会越界读取，必须先校验字节长度
        if (w_q_bytes.len < layout.w_q_bytes or scales_bytes.len < layout.sb_bytes or biases_bytes.len < layout.sb_bytes) {
            return error.ShapeMismatch;
        }
        const w_dims = layout.w_dims;
        const sb_dims = layout.sb_dims;

        var w_q = try mlx.Array.fromRawData(@ptrCast(w_q_bytes.ptr), &w_dims, .uint32);
        errdefer w_q.deinit();
        var scales = try mlx.Array.fromRawData(@ptrCast(scales_bytes.ptr), &sb_dims, sb_dtype);
        errdefer scales.deinit();
        var biases = try mlx.Array.fromRawData(@ptrCast(biases_bytes.ptr), &sb_dims, sb_dtype);
        errdefer biases.deinit();

        return .{
            .w_q = w_q,
            .scales = scales,
            .biases = biases,
            .in_features = in_features,
            .out_features = out_features,
            .group_size = group_size,
            .bits = bits,
        };
    }

    /// 从多个已打包的直接字节切片沿输出维度 (axis 0) 串联构建单一融合 GPU 量化权重矩阵 (Fused Q4Weights)
    /// 例如：将 Q/K/V 三个投影或 Gate/Up 两个投影合并为单一矩阵，使用复用的 `scratch_buf` 一次构建对齐的 MLX 张量
    pub fn initFusedFromDirectBuffersCustom(
        w_q_slices: []const []const u8,
        scales_slices: []const []const u8,
        biases_slices: []const []const u8,
        scratch_buf: []u8,
        in_features: usize,
        total_out_features: usize,
        group_size: i32,
        bits: i32,
        sb_dtype: mlx.DType,
        stream: ?Stream,
    ) OffloadError!Q4Weights {
        _ = stream;
        if (!isAvailable()) return error.OffloadNotSupported;
        if (w_q_slices.len == 0 or w_q_slices.len != scales_slices.len or w_q_slices.len != biases_slices.len) {
            return error.ShapeMismatch;
        }

        const layout = try QuantLayout.init(in_features, total_out_features, group_size, bits, sb_dtype);
        const w_dims = layout.w_dims;
        const sb_dims = layout.sb_dims;

        // 每组切片串联后的总字节数必须恰好等于融合矩阵所需字节数；mlx_array_new_data 会立即复制数据，因此 scratch_buf 可逐组复用
        try concatInto(scratch_buf, w_q_slices, layout.w_q_bytes);
        var w_q = try mlx.Array.fromRawData(@ptrCast(scratch_buf.ptr), &w_dims, .uint32);
        errdefer w_q.deinit();

        try concatInto(scratch_buf, scales_slices, layout.sb_bytes);
        var scales = try mlx.Array.fromRawData(@ptrCast(scratch_buf.ptr), &sb_dims, sb_dtype);
        errdefer scales.deinit();

        try concatInto(scratch_buf, biases_slices, layout.sb_bytes);
        var biases = try mlx.Array.fromRawData(@ptrCast(scratch_buf.ptr), &sb_dims, sb_dtype);
        errdefer biases.deinit();

        return .{
            .w_q = w_q,
            .scales = scales,
            .biases = biases,
            .in_features = in_features,
            .out_features = total_out_features,
            .group_size = group_size,
            .bits = bits,
        };
    }

    /// 从多个 Q4Block 权重切片直接沿输出维度 (axis 0) 串联构建融合 GPU 硬件加速量化权重矩阵 (Fused Q4Weights)
    /// 例如：同时传入 gate_proj 与 up_proj 的切片，直接构建单一大矩阵 [gate_out + up_out, in_features]
    pub fn initFromBlockSlices(
        allocator: std.mem.Allocator,
        block_slices: []const []const u8,
        in_features: usize,
        out_features_list: []const usize,
        stream: ?Stream,
    ) OffloadError!Q4Weights {
        if (!isAvailable()) return error.OffloadNotSupported;
        if (block_slices.len != out_features_list.len or block_slices.len == 0) return error.ShapeMismatch;
        if (in_features == 0 or in_features % 32 != 0) return error.InvalidShape;

        const blocks_per_row = in_features / 32;
        var total_out_features: usize = 0;
        var total_blocks: usize = 0;
        for (out_features_list, 0..) |out_f, i| {
            const expected_bytes = out_f * blocks_per_row * 18;
            if (block_slices[i].len < expected_bytes) return error.ShapeMismatch;
            total_out_features += out_f;
            total_blocks += out_f * blocks_per_row;
        }

        const w_q_buf = try allocator.alloc(u32, total_blocks * 4);
        defer allocator.free(w_q_buf);
        const scales_buf = try allocator.alloc(f16, total_blocks);
        defer allocator.free(scales_buf);
        const biases_buf = try allocator.alloc(f16, total_blocks);
        defer allocator.free(biases_buf);

        packBlocksToDirectBuffers(block_slices, out_features_list, in_features, w_q_buf, scales_buf, biases_buf);

        return initFromDirectBuffers(
            std.mem.sliceAsBytes(w_q_buf),
            std.mem.sliceAsBytes(scales_buf),
            std.mem.sliceAsBytes(biases_buf),
            in_features,
            total_out_features,
            stream,
        );
    }

    /// 从单个原始 Q4Block 切片 (18 字节/块) 构建 GPU 硬件加速量化权重矩阵
    pub fn initFromBlocks(
        allocator: std.mem.Allocator,
        blocks_bytes: []const u8,
        in_features: usize,
        out_features: usize,
        stream: ?Stream,
    ) OffloadError!Q4Weights {
        return initFromBlockSlices(
            allocator,
            &.{blocks_bytes},
            in_features,
            &.{out_features},
            stream,
        );
    }

    /// 释放图形处理器 (Graphics Processing Unit, GPU) 权重资源
    pub fn deinit(self: *Q4Weights) void {
        self.w_q.deinit();
        self.scales.deinit();
        self.biases.deinit();
    }

    /// 惰性量化矩阵乘法数组分发 (Quantized Matrix Multiplication, QMM)
    /// 保持结果在图形处理器 (Graphics Processing Unit, GPU) 显存中，支持算子融合与流水线批处理调度
    pub fn forwardArray(
        self: Q4Weights,
        x_mlx: mlx.Array,
        stream: ?Stream,
    ) OffloadError!mlx.Array {
        if (!isAvailable()) return error.OffloadNotSupported;
        return x_mlx.quantizedMatmul(
            self.w_q,
            self.scales,
            self.biases,
            true, // transpose
            self.group_size,
            self.bits,
            "affine",
            stream,
        );
    }

    /// 前向量化矩阵乘法 (Quantized Matrix Multiplication, QMM)：out = X * W^T
    pub fn forward(
        self: Q4Weights,
        X: anytype,
        out: anytype,
        stream: ?Stream,
    ) OffloadError!void {
        if (!isAvailable()) return error.OffloadNotSupported;
        var total_elements: usize = 1;
        for (0..X.shape.len) |d| {
            total_elements *= X.shape.dims[d];
        }
        const in_f = self.in_features;
        if (total_elements % in_f != 0) return error.ShapeMismatch;
        const rows = total_elements / in_f;

        var x_b = try borrowArray(X);
        defer x_b.deinit();

        var x_2d = if (x_b.arr.ndim() == 2)
            x_b.arr
        else
            try x_b.arr.reshape(&.{ rows, in_f }, stream);
        defer if (x_b.arr.ndim() != 2) x_2d.deinit();

        var res_mlx = try self.forwardArray(x_2d, stream);
        errdefer res_mlx.deinit();

        try storeOutput(out, res_mlx);
    }
};

pub const Q4Weights = QuantWeights;

/// 语言模型词表输出头统一门面：支持稠密转置权重 (LmHeadWeights) 与分块量化权重 (QuantWeights)
pub const LmHead = union(enum) {
    dense: LmHeadWeights,
    quant: QuantWeights,

    pub fn vocabSize(self: *const LmHead) usize {
        return switch (self.*) {
            .dense => |*d| d.vocab_size,
            .quant => |*q| q.out_features,
        };
    }

    pub fn hiddenSize(self: *const LmHead) usize {
        return switch (self.*) {
            .dense => |*d| d.hidden_size,
            .quant => |*q| q.in_features,
        };
    }

    pub fn forwardArray(self: *const LmHead, x: mlx.Array, s: ?Stream) OffloadError!mlx.Array {
        return switch (self.*) {
            .dense => |*d| d.forwardArrayBf16(x, s),
            .quant => |*q| q.forwardArray(x, s),
        };
    }

    pub fn deinit(self: *LmHead) void {
        switch (self.*) {
            .dense => |*d| d.deinit(),
            .quant => |*q| q.deinit(),
        }
    }
};

/// 量化矩阵乘法硬件加速便捷入口 (Quantized Matrix Multiplication, QMM)
pub fn q4MatMul(
    X: anytype,
    out: anytype,
    weights: *const QuantWeights,
    stream: ?Stream,
) OffloadError!void {
    return weights.forward(X, out, stream);
}

pub const quantMatMul = q4MatMul;

/// 带掩码的分组查询缩放点积注意力 (Masked Grouped-Query Scaled Dot-Product Attention)
/// q: [B, num_heads, tq, head_dim]，k / v: [B, num_kv_heads, tk, head_dim]（num_heads 为 num_kv_heads 的整数倍，dtype 任意且一致）；
/// 第 i 个查询的绝对位置为 q_pos0 + i，第 j 个键的绝对位置为 k_pos0 + j，可见性见 `attention_mask.Mask`。
/// 掩码先在 `[tq, tk]` 网格上化简 (`Mask.simplify`)：无掩码与右下对齐的因果掩码直接使用融合内核的内置模式，
/// 其余情况在主机端构建加性掩码 (`fillAdditiveMask`)，按查询的 dtype 交给融合内核 (`mask_mode = "array"`)。
/// 打分在 Softmax 前乘以 scale。
pub fn maskedAttention(
    allocator: std.mem.Allocator,
    q: mlx.Array,
    k: mlx.Array,
    v: mlx.Array,
    scale: f32,
    mask: attention_mask.Mask,
    q_pos0: usize,
    k_pos0: usize,
    stream: ?Stream,
) OffloadError!mlx.Array {
    if (!isAvailable()) return error.OffloadNotSupported;
    if (q.ndim() != 4 or k.ndim() != 4) return error.InvalidShape;
    const tq = q.dimSize(2);
    const tk = k.dimSize(2);

    switch (mask.simplify(tq, tk, q_pos0, k_pos0)) {
        .none => return mlx.Array.fastScaledDotProductAttention(q, k, v, scale, .none, stream),
        // 融合内核的因果模式按右下角对齐：第 i 个查询可见前 i + tk - tq + 1 个键
        .causal => if (q_pos0 + tq == k_pos0 + tk) return mlx.Array.fastScaledDotProductAttention(q, k, v, scale, .causal, stream),
        .sliding => {},
    }

    const mask_buf = try allocator.alloc(f32, tq * tk);
    defer allocator.free(mask_buf);
    attention_mask.fillAdditiveMask(mask_buf, tq, tk, q_pos0, k_pos0, mask);
    var mask_f32 = try mlx.Array.fromSlice(mask_buf, &.{ 1, 1, tq, tk });
    defer mask_f32.deinit();
    // 加性掩码的 dtype 须可提升至查询的 dtype (如 bfloat16)，-1e9 在 bfloat16 中仍为有限的极大负值
    const q_dtype = q.dtype();
    var mask_cast: ?mlx.Array = if (q_dtype != .float32) try mask_f32.astype(q_dtype, stream) else null;
    defer if (mask_cast) |*m| m.deinit();

    return mlx.Array.fastScaledDotProductAttention(q, k, v, scale, .{ .additive = mask_cast orelse mask_f32 }, stream);
}

/// 键值缓存一步前向的注意力：键值视图须为 `GpuKvCache.writeStep` 返回的视图 (或与其覆盖相同绝对位置的视图)，
/// 视图第 0 个键的绝对位置为 `visibleKeyRange` 的起点；查询位于 [pos, pos + seq_len)。
/// 掩码化简后：可见键范围随查询变化时施加滑动窗口加性掩码，多词元步使用右下对齐因果掩码，
/// 单词元解码步不需要掩码 (视图恰为该查询的可见键，含环形覆写的完整缓存)
pub fn cachedAttention(
    allocator: std.mem.Allocator,
    q: mlx.Array,
    k: mlx.Array,
    v: mlx.Array,
    scale: f32,
    is_sliding: bool,
    sliding_window: usize,
    pos: usize,
    seq_len: usize,
    stream: ?Stream,
) OffloadError!mlx.Array {
    const range = GpuKvCache.visibleKeyRange(is_sliding, sliding_window, pos, seq_len);
    const mask: attention_mask.Mask = if (is_sliding) .{ .sliding = sliding_window } else .causal;
    // 环形覆写的完整缓存不按绝对位置排列，但它恰为单查询的可见键 [pos + 1 - sliding_window, pos]，
    // 按这一绝对位置区间描述时掩码化简为无掩码，与槽位顺序无关
    return maskedAttention(allocator, q, k, v, scale, mask, pos, range.start, stream);
}

/// 块式预分配显存键值缓存 (Pre-allocated Chunked GPU Key-Value Cache, GPU KVCache)
pub const GpuKvCache = struct {
    k: mlx.Array,
    v: mlx.Array,
    step_eff_k: ?mlx.Array = null,
    step_eff_v: ?mlx.Array = null,
    capacity: usize,
    num_kv: usize,
    head_dim: usize,
    dtype: mlx.DType = .bfloat16,
    curr_len: usize = 0,

    pub fn init(capacity: usize, num_kv: usize, head_dim: usize, dtype: mlx.DType, stream: Stream) !GpuKvCache {
        var k = try mlx.Array.zeros(&.{ 1, num_kv, capacity, head_dim }, dtype, stream);
        errdefer k.deinit();
        const v = try mlx.Array.zeros(&.{ 1, num_kv, capacity, head_dim }, dtype, stream);
        return .{
            .k = k,
            .v = v,
            .step_eff_k = null,
            .step_eff_v = null,
            .capacity = capacity,
            .num_kv = num_kv,
            .head_dim = head_dim,
            .dtype = dtype,
            .curr_len = 0,
        };
    }

    pub fn initDefault(capacity: usize, num_kv: usize, head_dim: usize, stream: Stream) !GpuKvCache {
        return init(capacity, num_kv, head_dim, .bfloat16, stream);
    }

    pub fn truncate(self: *GpuKvCache, new_len: usize) void {
        self.clearStepSlices();
        if (new_len < self.curr_len) {
            self.curr_len = new_len;
        }
    }

    pub fn reset(self: *GpuKvCache) void {
        self.clearStepSlices();
        self.curr_len = 0;
    }

    pub fn clearStepSlices(self: *GpuKvCache) void {
        if (self.step_eff_k) |*ek| {
            ek.deinit();
            self.step_eff_k = null;
        }
        if (self.step_eff_v) |*ev| {
            ev.deinit();
            self.step_eff_v = null;
        }
    }

    pub fn deinit(self: *GpuKvCache) void {
        self.clearStepSlices();
        self.k.deinit();
        self.v.deinit();
    }

    pub fn importWindow(
        self: *GpuKvCache,
        k_data: []const f32,
        v_data: []const f32,
        win: usize,
        stream: Stream,
    ) !void {
        try self.ensureCapacity(win, stream);
        var k_arr = try mlx.Array.fromSlice(k_data, &.{ 1, self.num_kv, win, self.head_dim });
        defer k_arr.deinit();
        var v_arr = try mlx.Array.fromSlice(v_data, &.{ 1, self.num_kv, win, self.head_dim });
        defer v_arr.deinit();

        var next_k = try self.k.sliceUpdate4dAxis2(k_arr, self.num_kv, 0, win, self.head_dim, stream);
        errdefer next_k.deinit();
        const next_v = try self.v.sliceUpdate4dAxis2(v_arr, self.num_kv, 0, win, self.head_dim, stream);
        self.k.deinit();
        self.v.deinit();
        self.k = next_k;
        self.v = next_v;
        self.curr_len = win;
    }

    pub fn exportWindow(
        self: *const GpuKvCache,
        dst_k: []f32,
        dst_v: []f32,
        win: usize,
        stream: ?Stream,
    ) !void {
        var sliced_k = try self.k.slice4dAxis2(self.num_kv, 0, win, self.head_dim, stream);
        defer sliced_k.deinit();
        var sliced_v = try self.v.slice4dAxis2(self.num_kv, 0, win, self.head_dim, stream);
        defer sliced_v.deinit();
        try sliced_k.copyToSlice(dst_k);
        try sliced_v.copyToSlice(dst_v);
    }

    /// 本步查询 [pos, pos + seq_len) 共同需要的键的绝对位置范围 [start, stop)
    pub const KeyRange = struct {
        start: usize,
        stop: usize,
        /// 各查询的可见键范围不同 (键数超过滑动窗口)，须施加滑动窗口掩码；为 false 时右下对齐因果掩码即精确
        needs_window_mask: bool,
    };

    /// 计算本步查询共同需要的键范围：全局注意力层为 [0, pos + seq_len)；滑动窗口层为
    /// [max(0, pos + 1 - sliding_window), pos + seq_len)，即首个查询窗口的起点到末个查询自身
    /// (与 mlx-lm `RotatingKVCache` 多词元更新保留最近 sliding_window - 1 个历史键的规则一致)
    pub fn visibleKeyRange(is_sliding: bool, sliding_window: usize, pos: usize, seq_len: usize) KeyRange {
        const stop = pos + seq_len;
        if (!is_sliding) return .{ .start = 0, .stop = stop, .needs_window_mask = false };
        const start = if (pos + 1 > sliding_window) pos + 1 - sliding_window else 0;
        return .{ .start = start, .stop = stop, .needs_window_mask = stop - start > sliding_window };
    }

    pub const WriteRange = struct {
        write_start: usize,
        write_stop: usize,
        slice_start: usize,
        slice_stop: usize,
        use_full_buffer: bool,
    };

    pub const WriteRangeError = error{
        /// 滑动窗口层的缓存已进入环形覆写布局 (槽位不再按绝对位置排列)，无法再追加多词元步；须先重置缓存再预填充
        SlidingWindowCacheWrapped,
    };

    /// 计算键值缓存写入与注意力读取范围（支持局部滑动窗口层的固定容量环形覆写 RotatingKVCache）
    /// - 当局部滑动窗口层处于单步解码 (`seq_len == 1`) 且序列位置达到窗口容量上限 (`pos >= sliding_window`) 时，
    ///   直接以 `pos % sliding_window` 原地环形覆写最旧词元槽位并直接复用完整缓存张量，消除扩容拷贝与切片视图分配开销
    /// - 其余情况按绝对位置线性写入 [pos, pos + seq_len)，读取 `visibleKeyRange` 给出的键范围：多词元步保留首个查询窗口内的全部历史键，
    ///   由注意力的滑动窗口掩码区分各查询的可见键
    pub fn computeWriteRange(
        capacity: usize,
        is_sliding: bool,
        sliding_window: usize,
        pos: usize,
        seq_len: usize,
    ) WriteRangeError!WriteRange {
        const total_len = pos + seq_len;
        const ring_capacity = is_sliding and capacity == sliding_window;
        if (ring_capacity and seq_len == 1 and pos >= sliding_window) {
            const write_idx = pos % sliding_window;
            return .{
                .write_start = write_idx,
                .write_stop = write_idx + 1,
                .slice_start = 0,
                .slice_stop = sliding_window,
                .use_full_buffer = true,
            };
        }
        // 容量恰为窗口且 pos 已越过窗口，说明位置 sliding_window 处已发生过环形覆写
        if (ring_capacity and seq_len > 1 and pos > sliding_window) return error.SlidingWindowCacheWrapped;
        const range = visibleKeyRange(is_sliding, sliding_window, pos, seq_len);
        return .{
            .write_start = pos,
            .write_stop = total_len,
            .slice_start = range.start,
            .slice_stop = range.stop,
            .use_full_buffer = (range.start == 0 and total_len == capacity),
        };
    }

    /// 本步注意力读取的键值视图
    pub const StepView = struct {
        k: mlx.Array,
        v: mlx.Array,
        /// 为 true 时视图是新建的切片 (调用方负责释放或转交)；为 false 时视图即缓存张量本身 (由缓存持有)
        is_slice: bool,
    };

    /// 将本步旋转后的键与值 (`[1, num_kv, seq_len, head_dim]`) 按 `computeWriteRange` 写入缓存 (必要时扩容)，
    /// 返回注意力应读取的键值视图 (覆盖 `visibleKeyRange` 的绝对位置，或环形覆写时的完整缓存)
    pub fn writeStep(
        self: *GpuKvCache,
        k_new: mlx.Array,
        v_new: mlx.Array,
        is_sliding: bool,
        sliding_window: usize,
        pos: usize,
        seq_len: usize,
        stream: Stream,
    ) !StepView {
        const range = try computeWriteRange(self.capacity, is_sliding, sliding_window, pos, seq_len);
        try self.ensureCapacity(range.write_stop, stream);
        {
            var next_k = try self.k.sliceUpdate4dAxis2(k_new, self.num_kv, range.write_start, range.write_stop, self.head_dim, stream);
            errdefer next_k.deinit();
            const next_v = try self.v.sliceUpdate4dAxis2(v_new, self.num_kv, range.write_start, range.write_stop, self.head_dim, stream);
            self.k.deinit();
            self.v.deinit();
            self.k = next_k;
            self.v = next_v;
            self.curr_len = range.write_stop;
        }
        if (range.use_full_buffer) return .{ .k = self.k, .v = self.v, .is_slice = false };
        var sliced_k = try self.k.slice4dAxis2(self.num_kv, range.slice_start, range.slice_stop, self.head_dim, stream);
        errdefer sliced_k.deinit();
        const sliced_v = try self.v.slice4dAxis2(self.num_kv, range.slice_start, range.slice_stop, self.head_dim, stream);
        return .{ .k = sliced_k, .v = sliced_v, .is_slice = true };
    }

    pub fn ensureCapacity(self: *GpuKvCache, required: usize, stream: Stream) !void {
        if (required <= self.capacity) return;
        const add_cap: usize = ((required - self.capacity + 255) / 256) * 256;
        const new_cap = self.capacity + add_cap;
        var pad_k = try mlx.Array.zeros(&.{ 1, self.num_kv, add_cap, self.head_dim }, self.dtype, stream);
        defer pad_k.deinit();
        var pad_v = try mlx.Array.zeros(&.{ 1, self.num_kv, add_cap, self.head_dim }, self.dtype, stream);
        defer pad_v.deinit();

        var exp_k = try mlx.Array.concatenateAxis(&.{ self.k, pad_k }, 2, stream);
        errdefer exp_k.deinit();
        const exp_v = try mlx.Array.concatenateAxis(&.{ self.v, pad_v }, 2, stream);

        self.k.deinit();
        self.v.deinit();
        self.k = exp_k;
        self.v = exp_v;
        self.capacity = new_cap;
    }
};

// ============================================================================
// 共享 MLX 组合块与对数几率处理器转导出
// ============================================================================

pub const mlx_blocks = @import("mlx_blocks.zig");
pub const blocks = mlx_blocks;
pub const Rope = mlx_blocks.Rope;
pub const projectHeads = mlx_blocks.projectHeads;
pub const gatedMlp = mlx_blocks.gatedMlp;
pub const residualScale = mlx_blocks.residualScale;
pub const finalHidden = mlx_blocks.finalHidden;

pub const logits_processor = @import("logits_processor.zig");
pub const softcapLogits = logits_processor.softcapLogits;
pub const penalizeDevice = logits_processor.penalizeDevice;
pub const deviceCandidates = logits_processor.deviceCandidates;
pub const selectToken = logits_processor.selectToken;
pub const selectTokenFromHidden = logits_processor.selectTokenFromHidden;
pub const DeviceCandidates = logits_processor.DeviceCandidates;
pub const HostCandidates = logits_processor.HostCandidates;
pub const LogitsProcessorOptions = logits_processor.LogitsProcessorOptions;
pub const LogitsProcessor = logits_processor.LogitsProcessor;
