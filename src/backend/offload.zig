//! 硬件加速统一卸载基础设施 (Hardware Offload Infrastructure)
//!
//! 将计算密集型算子（通用矩阵乘法 GEMM、均方根层归一化 RMSNorm、旋转位置编码 RoPE、
//! 缩放点积注意力 SDPA / FlashAttention、语言模型预测头 LM Head 巨型词表投影）
//! 统一卸载至 Apple 机器学习拓展 (Machine Learning Explore, MLX) 后端在图形处理器 (GPU) 上执行。
//! 遵循分层规范，本模块位于最底层的 backend 包（层级 0），对外暴露解耦的硬件加速卸载入口。

const std = @import("std");
const mlx = @import("mlx.zig");
const activation = @import("activation.zig");
const sampling = @import("sampling.zig");

/// 当前硬件环境是否支持统一卸载 (macOS Apple Silicon aarch64 且已集成 MLX)
pub fn isAvailable() bool {
    return mlx.has_mlx;
}

pub const Stream = mlx.Stream;
pub const DeviceType = mlx.DeviceType;

pub const OffloadError = mlx.MlxError || error{
    OffloadNotSupported,
    UnsupportedQuantMode,
    InvalidShape,
    ShapeMismatch,
    NonContiguousTensor,
};

/// 通用矩阵乘法硬件卸载 (General Matrix Multiply Offload, GEMM Offload)：C = A * B
/// 支持二维矩阵乘法以及高维张量批处理矩阵乘法 (Batch MatMul)
pub fn matmul(
    A: anytype,
    B: anytype,
    C: anytype,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var a_mlx = try mlx.Array.fromTensorZeroCopy(A);
    defer a_mlx.deinit();
    var b_mlx = try mlx.Array.fromTensorZeroCopy(B);
    defer b_mlx.deinit();

    var c_mlx = try mlx.Array.matmul(a_mlx, b_mlx, stream);
    defer c_mlx.deinit();

    try c_mlx.copyToTensor(C);
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
    var x_mlx = try mlx.Array.fromTensorZeroCopy(X);
    defer x_mlx.deinit();

    var w_mlx: ?mlx.Array = null;
    defer if (w_mlx) |*w| w.deinit();
    const WT = @TypeOf(weight);
    if (@typeInfo(WT) == .optional) {
        if (weight) |w| {
            const D = X.shape.dims[X.shape.len - 1];
            w_mlx = try mlx.Array.fromSliceZeroCopy(w.data[0..D], &.{D});
        }
    } else {
        const D = X.shape.dims[X.shape.len - 1];
        w_mlx = try mlx.Array.fromSliceZeroCopy(weight.data[0..D], &.{D});
    }

    var out_mlx = try mlx.Array.fastRmsNorm(x_mlx, w_mlx, eps, stream);
    defer out_mlx.deinit();

    try out_mlx.copyToTensor(out);
}

/// 快速旋转位置编码硬件卸载 (Rotary Position Embedding Offload, RoPE Offload)
pub fn rope(
    X: anytype,
    out: anytype,
    dims: usize,
    traditional: bool,
    base: ?f32,
    scale: f32,
    offset: usize,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var x_mlx = try mlx.Array.fromTensorZeroCopy(X);
    defer x_mlx.deinit();

    var y_mlx = try mlx.Array.fastRoPE(x_mlx, dims, traditional, base, scale, offset, stream);
    defer y_mlx.deinit();

    try y_mlx.copyToTensor(out);
}

/// 快速缩放点积注意力硬件卸载 (Scaled Dot-Product Attention Offload, SDPA Offload / FlashAttention)
pub fn scaledDotProductAttention(
    queries: anytype,
    keys: anytype,
    values: anytype,
    out: anytype,
    scale: f32,
    is_causal: bool,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var q_mlx = try mlx.Array.fromTensor(queries);
    defer q_mlx.deinit();
    var k_mlx = try mlx.Array.fromTensor(keys);
    defer k_mlx.deinit();
    var v_mlx = try mlx.Array.fromTensor(values);
    defer v_mlx.deinit();

    var out_mlx = try mlx.Array.fastScaledDotProductAttention(
        q_mlx,
        k_mlx,
        v_mlx,
        scale,
        is_causal,
        stream,
    );
    defer out_mlx.deinit();

    try out_mlx.copyToTensor(out);
}

/// 归一化指数函数硬件卸载 (Softmax Offload)
pub fn softmax(
    A: anytype,
    out: anytype,
    axis: c_int,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var a_mlx = try mlx.Array.fromTensorZeroCopy(A);
    defer a_mlx.deinit();

    var res_mlx = try mlx.Array.softmax(a_mlx, axis, stream);
    defer res_mlx.deinit();

    try res_mlx.copyToTensor(out);
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

/// 4-bit 分块量化矩阵乘法硬件加速权重 (Quantized Matrix Multiply Weights, Q4Weights)
/// 将 Q4Block (18 字节/32 权重，含 16 字节数据与 1 个 f16 scale) 打包转置为
/// Apple 机器学习拓展 (Machine Learning Explore, MLX) 原生 uint32 紧凑排列并在图形处理器 (Graphics Processing Unit, GPU) 显存中常驻。
/// 在自回归单步解码时，通过统一内存零拷贝直接分发至硬件量化内核，消除 CPU 解包与带宽瓶颈。
pub const Q4Weights = struct {
    w_q: mlx.Array,
    scales: mlx.Array,
    biases: mlx.Array,
    in_features: usize,
    out_features: usize,
    group_size: i32 = 32,
    bits: i32 = 4,

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
        const out_f = self.out_features;
        if (total_elements % in_f != 0) return error.ShapeMismatch;
        const rows = total_elements / in_f;
        if (out.data.len < rows * out_f) return error.ShapeMismatch;

        var x_mlx = try mlx.Array.fromSliceZeroCopy(X.data[0..total_elements], &.{ rows, in_f });
        defer x_mlx.deinit();

        var res_mlx = try self.forwardArray(x_mlx, stream);
        defer res_mlx.deinit();
        res_mlx.eval();

        const res_data = try res_mlx.dataFloat32();
        if (res_data.len != rows * out_f) return error.ShapeMismatch;
        @memcpy(out.data[0 .. rows * out_f], res_data);
    }
};

/// 门控多层感知机硬件算子融合与异步执行 (Multilayer Perceptron, MLP)
/// 在图形处理器 (Graphics Processing Unit, GPU) 上连续调度门控投影 (gate_proj)、上升投影 (up_proj)、
/// 高斯误差线性单元激活 (Gaussian Error Linear Unit, GELU，形式由 `gelu_approximation` 指定)、逐元素点乘与下降投影 (down_proj)。
/// 中间激活值常驻显存，将三次独立内核同步合并为一次最终求值，消除驱动管线等待。
pub fn mlpForwardOffload(
    gate_qw: *const Q4Weights,
    up_qw: *const Q4Weights,
    down_qw: *const Q4Weights,
    gelu_approximation: activation.GeluApproximation,
    X: anytype,
    out: anytype,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var total_elements: usize = 1;
    for (0..X.shape.len) |d| {
        total_elements *= X.shape.dims[d];
    }
    const in_f = gate_qw.in_features;
    if (total_elements % in_f != 0) return error.ShapeMismatch;
    const rows = total_elements / in_f;
    const out_f = down_qw.out_features;
    if (out.data.len < rows * out_f) return error.ShapeMismatch;

    var x_mlx = try mlx.Array.fromSliceZeroCopy(X.data[0..total_elements], &.{ rows, in_f });
    defer x_mlx.deinit();

    var gate_arr = try gate_qw.forwardArray(x_mlx, stream);
    defer gate_arr.deinit();

    var up_arr = try up_qw.forwardArray(x_mlx, stream);
    defer up_arr.deinit();

    var act_arr = try gate_arr.gelu(gelu_approximation, stream);
    defer act_arr.deinit();

    var fused_arr = try mlx.Array.multiply(act_arr, up_arr, stream);
    defer fused_arr.deinit();

    var down_arr = try down_qw.forwardArray(fused_arr, stream);
    defer down_arr.deinit();

    down_arr.eval();

    const res_data = try down_arr.dataFloat32();
    if (res_data.len != rows * out_f) return error.ShapeMismatch;
    @memcpy(out.data[0 .. rows * out_f], res_data);
}

/// 查询键值多头投影硬件并发执行 (Query-Key-Value, QKV)
/// 在图形处理器 (Graphics Processing Unit, GPU) 上统一批处理分发 Q、K、V 三路量化投影，
/// 并通过多张量统一求值一次性完成计算，消除重复的主机-设备同步停顿。
pub fn qkvForwardOffload(
    q_qw: *const Q4Weights,
    k_qw: *const Q4Weights,
    v_qw: ?*const Q4Weights,
    X: anytype,
    q_out: anytype,
    k_out: anytype,
    v_out: anytype,
    stream: ?Stream,
) OffloadError!void {
    if (!isAvailable()) return error.OffloadNotSupported;
    var total_elements: usize = 1;
    for (0..X.shape.len) |d| {
        total_elements *= X.shape.dims[d];
    }
    const in_f = q_qw.in_features;
    if (total_elements % in_f != 0) return error.ShapeMismatch;
    const rows = total_elements / in_f;

    var x_mlx = try mlx.Array.fromSliceZeroCopy(X.data[0..total_elements], &.{ rows, in_f });
    defer x_mlx.deinit();

    var q_arr = try q_qw.forwardArray(x_mlx, stream);
    defer q_arr.deinit();

    var k_arr = try k_qw.forwardArray(x_mlx, stream);
    defer k_arr.deinit();

    if (v_qw) |vw| {
        var v_arr = try vw.forwardArray(x_mlx, stream);
        defer v_arr.deinit();

        mlx.Array.evalMultiple(&.{ q_arr, k_arr, v_arr });

        const q_data = try q_arr.dataFloat32();
        const k_data = try k_arr.dataFloat32();
        const v_data = try v_arr.dataFloat32();
        try copyRows(q_out.data, q_data, rows * q_qw.out_features);
        try copyRows(k_out.data, k_data, rows * k_qw.out_features);
        try copyRows(v_out.data, v_data, rows * vw.out_features);
    } else {
        mlx.Array.evalMultiple(&.{ q_arr, k_arr });

        const q_data = try q_arr.dataFloat32();
        const k_data = try k_arr.dataFloat32();
        // 无独立 V 投影时 K 与 V 共享同一投影结果
        try copyRows(q_out.data, q_data, rows * q_qw.out_features);
        try copyRows(k_out.data, k_data, rows * k_qw.out_features);
        try copyRows(v_out.data, k_data, rows * k_qw.out_features);
    }
}

/// 将设备端结果复制到输出缓冲区前缀：结果长度必须恰好为 len，输出缓冲区至少为 len
fn copyRows(dest: []f32, src: []const f32, len: usize) OffloadError!void {
    if (src.len != len or dest.len < len) return error.ShapeMismatch;
    @memcpy(dest[0..len], src);
}

/// 4-bit 量化矩阵乘法硬件加速便捷入口 (Quantized Matrix Multiplication, QMM)
pub fn q4MatMul(
    X: anytype,
    out: anytype,
    weights: *const Q4Weights,
    stream: ?Stream,
) OffloadError!void {
    return weights.forward(X, out, stream);
}

/// 分组查询注意力 (Grouped-Query Attention, GQA) 的头几何规格
pub const AttentionHeads = struct {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
};

/// 带滑动窗口掩码的分组查询缩放点积注意力 (Sliding Window Grouped-Query Scaled Dot-Product Attention)
/// q: [1, num_heads, tq, head_dim]，k / v: [1, num_kv_heads, tk, head_dim]；第 i 个查询的绝对位置为 q_pos0 + i，
/// 第 j 个键的绝对位置为 k_pos0 + j，仅当 0 <= q_pos - k_pos < sliding_window 时可见（不可见位置施加 -1e9 加性掩码）。
/// 打分在 Softmax 前乘以 scale（scale 为 1 时省略乘法）。
pub fn slidingWindowAttention(
    allocator: std.mem.Allocator,
    heads: AttentionHeads,
    q: mlx.Array,
    k: mlx.Array,
    v: mlx.Array,
    scale: f32,
    sliding_window: usize,
    tq: usize,
    tk: usize,
    q_pos0: usize,
    k_pos0: usize,
    stream: ?Stream,
) OffloadError!mlx.Array {
    if (!isAvailable()) return error.OffloadNotSupported;
    if (heads.num_kv_heads == 0 or heads.num_heads % heads.num_kv_heads != 0) return error.InvalidShape;
    const groups = heads.num_heads / heads.num_kv_heads;
    var k_trans = try k.transposeAxes(&.{ 0, 1, 3, 2 }, stream);
    defer k_trans.deinit();
    // 同一键值头的 groups 个查询头在头维度上相邻，重排为 [1, num_kv_heads, groups * tq, head_dim] 后共享同一组键值
    var q_grouped = try q.reshape(&.{ 1, heads.num_kv_heads, groups * tq, heads.head_dim }, stream);
    defer q_grouped.deinit();
    var scores_grouped = try mlx.Array.matmul(q_grouped, k_trans, stream);
    defer scores_grouped.deinit();
    var scores_raw = try scores_grouped.reshape(&.{ 1, heads.num_heads, tq, tk }, stream);
    defer scores_raw.deinit();
    var scaled_scores: ?mlx.Array = if (scale != 1.0) try scores_raw.mulScalar(scale, stream) else null;
    defer if (scaled_scores) |*sc| sc.deinit();
    const scores = scaled_scores orelse scores_raw;

    const mask_buf = try allocator.alloc(f32, tq * tk);
    defer allocator.free(mask_buf);
    for (0..tq) |i| {
        const q_pos = q_pos0 + i;
        for (0..tk) |j| {
            const k_pos = k_pos0 + j;
            const visible = k_pos <= q_pos and q_pos - k_pos < sliding_window;
            mask_buf[i * tk + j] = if (visible) 0.0 else -1e9;
        }
    }
    var mask_arr = try mlx.Array.fromSlice(mask_buf, &.{ 1, 1, tq, tk });
    defer mask_arr.deinit();
    var masked_scores = try mlx.Array.add(scores, mask_arr, stream);
    defer masked_scores.deinit();
    var probs = try mlx.Array.softmax(masked_scores, 3, stream);
    defer probs.deinit();

    var probs_grouped = try probs.reshape(&.{ 1, heads.num_kv_heads, groups * tq, tk }, stream);
    defer probs_grouped.deinit();
    var out_grouped = try mlx.Array.matmul(probs_grouped, v, stream);
    defer out_grouped.deinit();
    return out_grouped.reshape(&.{ 1, heads.num_heads, tq, heads.head_dim }, stream);
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

    pub fn init(capacity: usize, num_kv: usize, head_dim: usize, stream: Stream) !GpuKvCache {
        var k = try mlx.Array.zeros(&.{ 1, num_kv, capacity, head_dim }, .bfloat16, stream);
        errdefer k.deinit();
        const v = try mlx.Array.zeros(&.{ 1, num_kv, capacity, head_dim }, .bfloat16, stream);
        return .{
            .k = k,
            .v = v,
            .step_eff_k = null,
            .step_eff_v = null,
            .capacity = capacity,
            .num_kv = num_kv,
            .head_dim = head_dim,
        };
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

    pub const WriteRange = struct {
        write_start: usize,
        write_stop: usize,
        slice_start: usize,
        slice_stop: usize,
        use_full_buffer: bool,
    };

    /// 计算键值缓存写入与注意力读取范围（支持局部滑动窗口层的固定容量环形覆写 RotatingKVCache）
    /// 当局部滑动窗口层处于单步解码 (`seq_len == 1`) 且序列位置达到窗口容量上限 (`pos >= sliding_window`) 时，
    /// 直接以 `pos % sliding_window` 原地环形覆写最旧词元槽位并直接复用完整缓存张量，消除扩容拷贝与切片视图分配开销
    pub fn computeWriteRange(
        capacity: usize,
        is_sliding: bool,
        sliding_window: usize,
        pos: usize,
        seq_len: usize,
    ) WriteRange {
        const total_len = pos + seq_len;
        if (is_sliding and seq_len == 1 and pos >= sliding_window and capacity == sliding_window) {
            const write_idx = pos % sliding_window;
            return .{
                .write_start = write_idx,
                .write_stop = write_idx + 1,
                .slice_start = 0,
                .slice_stop = sliding_window,
                .use_full_buffer = true,
            };
        }
        var start_k: usize = 0;
        if (is_sliding and total_len > sliding_window) {
            start_k = total_len - sliding_window;
        }
        return .{
            .write_start = pos,
            .write_stop = total_len,
            .slice_start = start_k,
            .slice_stop = total_len,
            .use_full_buffer = (start_k == 0 and total_len == capacity),
        };
    }

    pub fn ensureCapacity(self: *GpuKvCache, required: usize, stream: Stream) !void {
        if (required <= self.capacity) return;
        const add_cap: usize = ((required - self.capacity + 255) / 256) * 256;
        const new_cap = self.capacity + add_cap;
        var pad_k = try mlx.Array.zeros(&.{ 1, self.num_kv, add_cap, self.head_dim }, .bfloat16, stream);
        defer pad_k.deinit();
        var pad_v = try mlx.Array.zeros(&.{ 1, self.num_kv, add_cap, self.head_dim }, .bfloat16, stream);
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
