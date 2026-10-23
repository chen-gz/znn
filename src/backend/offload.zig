//! 硬件加速统一卸载基础设施 (Hardware Offload Infrastructure)
//!
//! 将计算密集型算子（通用矩阵乘法 GEMM、均方根层归一化 RMSNorm、旋转位置编码 RoPE、
//! 缩放点积注意力 SDPA / FlashAttention、语言模型预测头 LM Head 巨型词表投影采样）
//! 统一卸载至 Apple 机器学习拓展 (Machine Learning Explore, MLX) 后端在图形处理器 (GPU) 上执行。
//! 遵循分层规范，本模块位于最底层的 backend 包（层级 0），对外暴露解耦的硬件加速卸载入口。

const std = @import("std");
const mlx = @import("mlx.zig");

/// 当前硬件环境是否支持统一卸载 (macOS Apple Silicon aarch64 且已集成 MLX)
pub fn isAvailable() bool {
    return mlx.has_mlx;
}

pub const Stream = mlx.Stream;
pub const DeviceType = mlx.DeviceType;

pub const OffloadError = mlx.MlxError || error{
    OffloadNotSupported,
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

pub const QuantMode = enum {
    bf16,
    q8,
    q4,
};

/// 语言模型输出头权重与采样加速器 (Language Model Head Accelerator, LM Head Accelerator)
/// 预先将巨型词表嵌入矩阵 (如 Gemma 4 的 262,144 维词表) 在 GPU 显存中转置常驻，
/// 自回归解码每一步仅需下发单步输入向量 [1, hidden_size]，利用统一内存实现 5~10 ms 极速词表打分与采样。
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

    /// 指定量化模式构建常驻 GPU 词表矩阵 (支持 BF16、Q8 与 Q4)
    pub fn initWithMode(
        embed_u16: []const u16,
        vocab_size: usize,
        hidden_size: usize,
        mode: QuantMode,
        stream: ?Stream,
    ) OffloadError!LmHeadWeights {
        if (!isAvailable()) return error.OffloadNotSupported;
        var w = try mlx.Array.fromSliceBf16(embed_u16, &.{ vocab_size, hidden_size });
        defer w.deinit();

        switch (mode) {
            .q4 => {
                if (hidden_size >= 32 and hidden_size % 32 == 0) {
                    var q4_pkg = try mlx.Array.quantize(w, 32, 4, "affine", stream);
                    q4_pkg.weights.eval();
                    q4_pkg.scales.eval();
                    if (q4_pkg.biases) |*b| b.eval();
                    if (stream) |st| st.synchronize();

                    return .{
                        .w_q = q4_pkg.weights,
                        .scales = q4_pkg.scales,
                        .biases = q4_pkg.biases,
                        .is_quantized = true,
                        .group_size = 32,
                        .bits = 4,
                        .vocab_size = vocab_size,
                        .hidden_size = hidden_size,
                    };
                }
            },
            .q8 => {
                if (hidden_size >= 64 and hidden_size % 64 == 0) {
                    var q8_pkg = try mlx.Array.quantize(w, 64, 8, "affine", stream);
                    q8_pkg.weights.eval();
                    q8_pkg.scales.eval();
                    if (q8_pkg.biases) |*b| b.eval();
                    if (stream) |st| st.synchronize();

                    return .{
                        .w_q = q8_pkg.weights,
                        .scales = q8_pkg.scales,
                        .biases = q8_pkg.biases,
                        .is_quantized = true,
                        .group_size = 64,
                        .bits = 8,
                        .vocab_size = vocab_size,
                        .hidden_size = hidden_size,
                    };
                }
            },
            .bf16 => {},
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

    /// 执行 GPU 词表打分投影计算图调度：[..., hidden_size] -> [..., vocab_size]
    pub fn forwardArray(self: LmHeadWeights, x: mlx.Array, stream: ?Stream) OffloadError!mlx.Array {
        var logits_bf16 = try self.forwardArrayBf16(x, stream);
        defer logits_bf16.deinit();
        return logits_bf16.astype(.float32, stream);
    }

    /// 贪婪采样时始终排除的特殊词元：0 号填充词元 (Padding Token) 与 Gemma 4 词表 [258880, 258884] 区间内的控制词元
    pub const excluded_token_range_lo: usize = 258880;
    pub const excluded_token_range_hi: usize = 258884;

    fn isExcludedToken(v: usize) bool {
        return v == 0 or (v >= excluded_token_range_lo and v <= excluded_token_range_hi);
    }

    /// 词元区间 [begin, end) 是否包含任一排除词元
    fn rangeHasExcludedToken(begin: usize, end: usize) bool {
        return begin == 0 or (begin <= excluded_token_range_hi and end > excluded_token_range_lo);
    }

    /// 最近词元位图窗口的字数：每个窗口覆盖 4096 * 64 = 262,144 个词元，位图保存在栈上，不进行堆分配
    const penalty_bitset_words: usize = 4096;
    const penalty_window_tokens: usize = penalty_bitset_words * 64;

    /// 在已计算的词表对数几率切片上执行分块位图重复惩罚与贪婪采样 (Chunked Bitset Penalty & Fast-Path Argmax)
    /// 词表按 262,144 词元的窗口分段，每段在栈上重建最近词元位图；绝大多数 64 词块位图为 0，
    /// 直接走单指令多数据流 (Single Instruction, Multiple Data, SIMD) 向量化块最大值快车道，仅在命中词块内执行位运算与惩罚。
    /// 结果与逐元素严格大于比较的顺序扫描一致：返回首个取得最大惩罚后对数几率的非排除词元（NaN 被忽略）。
    pub fn argmaxWithPenalty(
        logits_slice: []const f32,
        vocab_size: usize,
        recent_tokens: []const usize,
        repetition_penalty: f32,
    ) usize {
        std.debug.assert(logits_slice.len >= vocab_size);
        const apply_penalty = repetition_penalty > 1.0;
        const V = @Vector(16, f32);

        var bitset: [penalty_bitset_words]u64 = undefined;
        var max_val: f32 = -std.math.inf(f32);
        var max_id: usize = 0;

        var win_base: usize = 0;
        while (win_base < vocab_size) : (win_base += penalty_window_tokens) {
            const win_end = @min(win_base + penalty_window_tokens, vocab_size);
            const words = bitset[0 .. (win_end - win_base + 63) / 64];
            @memset(words, 0);
            if (apply_penalty) {
                for (recent_tokens) |rec| {
                    if (rec >= win_base and rec < win_end) {
                        const off = rec - win_base;
                        words[off / 64] |= @as(u64, 1) << @intCast(off % 64);
                    }
                }
            }

            for (words, 0..) |word, w| {
                const base_v = win_base + w * 64;
                const end_v = @min(base_v + 64, win_end);
                const check_excluded = rangeHasExcludedToken(base_v, end_v);

                if (word == 0 and !check_excluded and end_v - base_v == 64) {
                    // 快车道 (Fast Path)：整块 64 个词元既无重复词元也无排除词元，先向量化求块最大值，仅当刷新全局最大值时才定位下标
                    const block = logits_slice[base_v..][0..64];
                    var vmax: V = block[0..16].*;
                    inline for (1..4) |q| vmax = @max(vmax, @as(V, block[q * 16 ..][0..16].*));
                    const block_max = @reduce(.Max, vmax);
                    if (block_max > max_val) {
                        max_val = block_max;
                        max_id = base_v + std.mem.indexOfScalar(f32, block, block_max).?;
                    }
                    continue;
                }

                // 精准道：逐元素过滤排除词元，并对位图命中的最近词元施加重复惩罚
                for (base_v..end_v) |v| {
                    if (check_excluded and isExcludedToken(v)) continue;
                    var val = logits_slice[v];
                    if ((word >> @intCast(v - base_v)) & 1 != 0) {
                        val = if (val > 0.0) val / repetition_penalty else val * repetition_penalty;
                    }
                    if (val > max_val) {
                        max_val = val;
                        max_id = v;
                    }
                }
            }
        }

        return max_id;
    }

    /// 执行单步词表投影并贪婪采样最佳词元 ID
    pub fn sampleNext(
        self: LmHeadWeights,
        h_last: []const f32,
        recent_tokens: []const usize,
        repetition_penalty: f32,
        stream: ?Stream,
    ) OffloadError!usize {
        if (!isAvailable()) return error.OffloadNotSupported;
        if (h_last.len != self.hidden_size) return error.ShapeMismatch;

        var x_vec = try mlx.Array.fromSlice(h_last, &.{ 1, self.hidden_size });
        defer x_vec.deinit();

        var logits = try self.forwardArray(x_vec, stream);
        defer logits.deinit();
        logits.eval();

        const logits_slice = logits.dataFloat32();
        if (logits_slice.len != self.vocab_size) return error.ShapeMismatch;

        return argmaxWithPenalty(logits_slice, self.vocab_size, recent_tokens, repetition_penalty);
    }
};

/// 一次性词表投影与贪婪采样（未缓存权重的便捷卸载入口）
pub fn lmHeadArgmax(
    h_last: []const f32,
    embed_u16: []const u16,
    vocab_size: usize,
    hidden_size: usize,
    recent_tokens: []const usize,
    repetition_penalty: f32,
    stream: ?Stream,
) OffloadError!usize {
    var weights = try LmHeadWeights.init(embed_u16, vocab_size, hidden_size, stream);
    defer weights.deinit();
    return weights.sampleNext(h_last, recent_tokens, repetition_penalty, stream);
}

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

        const res_data = res_mlx.dataFloat32();
        if (res_data.len != rows * out_f) return error.ShapeMismatch;
        @memcpy(out.data[0..rows * out_f], res_data);
    }
};

/// 门控多层感知机硬件算子融合与异步执行 (Multilayer Perceptron, MLP)
/// 在图形处理器 (Graphics Processing Unit, GPU) 上连续调度门控投影 (gate_proj)、上升投影 (up_proj)、
/// 高斯误差线性单元激活 (Gaussian Error Linear Unit, GELU)、逐元素点乘与下降投影 (down_proj)。
/// 中间激活值常驻显存，将三次独立内核同步合并为一次最终求值，消除驱动管线等待。
pub fn mlpForwardOffload(
    gate_qw: *const Q4Weights,
    up_qw: *const Q4Weights,
    down_qw: *const Q4Weights,
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

    var act_arr = try gate_arr.gelu(stream);
    defer act_arr.deinit();

    var fused_arr = try mlx.Array.multiply(act_arr, up_arr, stream);
    defer fused_arr.deinit();

    var down_arr = try down_qw.forwardArray(fused_arr, stream);
    defer down_arr.deinit();

    down_arr.eval();

    const res_data = down_arr.dataFloat32();
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

        const q_data = q_arr.dataFloat32();
        const k_data = k_arr.dataFloat32();
        const v_data = v_arr.dataFloat32();
        try copyRows(q_out.data, q_data, rows * q_qw.out_features);
        try copyRows(k_out.data, k_data, rows * k_qw.out_features);
        try copyRows(v_out.data, v_data, rows * vw.out_features);
    } else {
        mlx.Array.evalMultiple(&.{ q_arr, k_arr });

        const q_data = q_arr.dataFloat32();
        const k_data = k_arr.dataFloat32();
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

/// 图形处理器 (Graphics Processing Unit, GPU) 加速解码单个 Transformer 层的常驻权重与超参数
pub const GpuLayerWeights = struct {
    fused_qkv: Q4Weights,
    o_proj: Q4Weights,
    fused_gate_up: Q4Weights,
    down_proj: Q4Weights,

    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    intermediate_size: usize,
    rot_dim: usize,
    attn_dim: usize,

    input_norm_w: mlx.Array,
    q_norm_w: mlx.Array,
    k_norm_w: mlx.Array,
    post_attn_w: mlx.Array,
    pre_mlp_w: mlx.Array,
    post_mlp_w: mlx.Array,

    layer_scalar: f32,
    attn_type: u8, // 0 = 滑动窗口注意力 (Sliding Window Attention), 1 = 全局全注意力 (Full Attention)
    sliding_window: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    partial_rotary_factor: f32,
    rms_norm_eps: f32,
    compiled_mlp: ?mlx.Closure = null,

    pub fn initFromBlocks(
        allocator: std.mem.Allocator,
        q_bytes: []const u8,
        k_bytes: []const u8,
        v_bytes: ?[]const u8,
        o_bytes: []const u8,
        gate_bytes: []const u8,
        up_bytes: []const u8,
        down_bytes: []const u8,
        hidden_size: usize,
        intermediate_size: usize,
        q_dim: usize,
        kv_dim: usize,
        in_w_data: []const f32,
        qn_w_data: []const f32,
        kn_w_data: []const f32,
        pa_w_data: []const f32,
        pm_w_data: []const f32,
        pom_w_data: []const f32,
        layer_scalar: f32,
        attn_type: u8,
        sliding_window: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        partial_rotary_factor: f32,
        rms_norm_eps: f32,
        stream: ?Stream,
    ) !GpuLayerWeights {
        var in_w = try mlx.Array.fromSlice(in_w_data, &.{in_w_data.len});
        errdefer in_w.deinit();
        var qn_w = try mlx.Array.fromSlice(qn_w_data, &.{qn_w_data.len});
        errdefer qn_w.deinit();
        var kn_w = try mlx.Array.fromSlice(kn_w_data, &.{kn_w_data.len});
        errdefer kn_w.deinit();
        var pa_w = try mlx.Array.fromSlice(pa_w_data, &.{pa_w_data.len});
        errdefer pa_w.deinit();
        var pm_w = try mlx.Array.fromSlice(pm_w_data, &.{pm_w_data.len});
        errdefer pm_w.deinit();
        var pom_w = try mlx.Array.fromSlice(pom_w_data, &.{pom_w_data.len});
        errdefer pom_w.deinit();

        const v_dim = if (v_bytes != null) kv_dim else 0;
        var fused_qkv = if (v_bytes) |vb|
            try Q4Weights.initFromBlockSlices(
                allocator,
                &.{ q_bytes, k_bytes, vb },
                hidden_size,
                &.{ q_dim, kv_dim, kv_dim },
                stream,
            )
        else
            try Q4Weights.initFromBlockSlices(
                allocator,
                &.{ q_bytes, k_bytes },
                hidden_size,
                &.{ q_dim, kv_dim },
                stream,
            );
        errdefer fused_qkv.deinit();

        var o_proj = try Q4Weights.initFromBlocks(
            allocator,
            o_bytes,
            q_dim,
            hidden_size,
            stream,
        );
        errdefer o_proj.deinit();

        var fused_gate_up = try Q4Weights.initFromBlockSlices(
            allocator,
            &.{ gate_bytes, up_bytes },
            hidden_size,
            &.{ intermediate_size, intermediate_size },
            stream,
        );
        errdefer fused_gate_up.deinit();

        var down_proj = try Q4Weights.initFromBlocks(
            allocator,
            down_bytes,
            intermediate_size,
            hidden_size,
            stream,
        );
        errdefer down_proj.deinit();

        const rot_dim = @as(usize, @intFromFloat(@as(f32, @floatFromInt(head_dim)) * partial_rotary_factor));
        const attn_dim = num_heads * head_dim;

        return .{
            .fused_qkv = fused_qkv,
            .o_proj = o_proj,
            .fused_gate_up = fused_gate_up,
            .down_proj = down_proj,
            .q_dim = q_dim,
            .k_dim = kv_dim,
            .v_dim = v_dim,
            .intermediate_size = intermediate_size,
            .rot_dim = rot_dim,
            .attn_dim = attn_dim,
            .input_norm_w = in_w,
            .q_norm_w = qn_w,
            .k_norm_w = kn_w,
            .post_attn_w = pa_w,
            .pre_mlp_w = pm_w,
            .post_mlp_w = pom_w,
            .layer_scalar = layer_scalar,
            .attn_type = attn_type,
            .sliding_window = sliding_window,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .rope_theta = rope_theta,
            .partial_rotary_factor = partial_rotary_factor,
            .rms_norm_eps = rms_norm_eps,
            .compiled_mlp = null,
        };
    }

    /// 从预重排直接内存缓冲区构建 GPU 硬件加速层权重 (Direct Layout Fast Path)
    /// 无需 CPU 解包与动态内存分配，实现毫秒级快速初始化
    pub fn initFromDirect(
        fused_qkv_wq: []const u8,
        fused_qkv_scales: []const u8,
        fused_qkv_biases: []const u8,
        o_proj_wq: []const u8,
        o_proj_scales: []const u8,
        o_proj_biases: []const u8,
        fused_gate_up_wq: []const u8,
        fused_gate_up_scales: []const u8,
        fused_gate_up_biases: []const u8,
        down_proj_wq: []const u8,
        down_proj_scales: []const u8,
        down_proj_biases: []const u8,
        hidden_size: usize,
        intermediate_size: usize,
        q_dim: usize,
        kv_dim: usize,
        v_dim: usize,
        in_w_data: []const f32,
        qn_w_data: []const f32,
        kn_w_data: []const f32,
        pa_w_data: []const f32,
        pm_w_data: []const f32,
        pom_w_data: []const f32,
        layer_scalar: f32,
        attn_type: u8,
        sliding_window: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        partial_rotary_factor: f32,
        rms_norm_eps: f32,
        stream: ?Stream,
    ) OffloadError!GpuLayerWeights {
        if (!isAvailable()) return error.OffloadNotSupported;

        var in_w = try mlx.Array.fromSlice(in_w_data, &.{in_w_data.len});
        errdefer in_w.deinit();
        var qn_w = try mlx.Array.fromSlice(qn_w_data, &.{qn_w_data.len});
        errdefer qn_w.deinit();
        var kn_w = try mlx.Array.fromSlice(kn_w_data, &.{kn_w_data.len});
        errdefer kn_w.deinit();
        var pa_w = try mlx.Array.fromSlice(pa_w_data, &.{pa_w_data.len});
        errdefer pa_w.deinit();
        var pm_w = try mlx.Array.fromSlice(pm_w_data, &.{pm_w_data.len});
        errdefer pm_w.deinit();
        var pom_w = try mlx.Array.fromSlice(pom_w_data, &.{pom_w_data.len});
        errdefer pom_w.deinit();

        var fused_qkv = try Q4Weights.initFromDirectBuffers(
            fused_qkv_wq,
            fused_qkv_scales,
            fused_qkv_biases,
            hidden_size,
            q_dim + kv_dim + v_dim,
            stream,
        );
        errdefer fused_qkv.deinit();

        var o_proj = try Q4Weights.initFromDirectBuffers(
            o_proj_wq,
            o_proj_scales,
            o_proj_biases,
            q_dim,
            hidden_size,
            stream,
        );
        errdefer o_proj.deinit();

        var fused_gate_up = try Q4Weights.initFromDirectBuffers(
            fused_gate_up_wq,
            fused_gate_up_scales,
            fused_gate_up_biases,
            hidden_size,
            intermediate_size * 2,
            stream,
        );
        errdefer fused_gate_up.deinit();

        var down_proj = try Q4Weights.initFromDirectBuffers(
            down_proj_wq,
            down_proj_scales,
            down_proj_biases,
            intermediate_size,
            hidden_size,
            stream,
        );
        errdefer down_proj.deinit();

        const rot_dim = @as(usize, @intFromFloat(@as(f32, @floatFromInt(head_dim)) * partial_rotary_factor));
        const attn_dim = num_heads * head_dim;

        return .{
            .fused_qkv = fused_qkv,
            .o_proj = o_proj,
            .fused_gate_up = fused_gate_up,
            .down_proj = down_proj,
            .q_dim = q_dim,
            .k_dim = kv_dim,
            .v_dim = v_dim,
            .intermediate_size = intermediate_size,
            .rot_dim = rot_dim,
            .attn_dim = attn_dim,
            .input_norm_w = in_w,
            .q_norm_w = qn_w,
            .k_norm_w = kn_w,
            .post_attn_w = pa_w,
            .pre_mlp_w = pm_w,
            .post_mlp_w = pom_w,
            .layer_scalar = layer_scalar,
            .attn_type = attn_type,
            .sliding_window = sliding_window,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .rope_theta = rope_theta,
            .partial_rotary_factor = partial_rotary_factor,
            .rms_norm_eps = rms_norm_eps,
            .compiled_mlp = null,
        };
    }

    /// 门控多层感知机残差块：x2 = (x1 + post_mlp_norm(down_proj(GeGLU(fused_gate_up(pre_mlp_norm(x1)))))) * layer_scalar
    pub fn mlpResidual(self: *const GpuLayerWeights, x1: mlx.Array, stream: ?Stream) !mlx.Array {
        var pre_mlp = try mlx.Array.fastRmsNorm(x1, self.pre_mlp_w, self.rms_norm_eps, stream);
        defer pre_mlp.deinit();
        var gate_up = try self.fused_gate_up.forwardArray(pre_mlp, stream);
        defer gate_up.deinit();
        var fused = try gate_up.geglu(self.intermediate_size, stream);
        defer fused.deinit();
        var down = try self.down_proj.forwardArray(fused, stream);
        defer down.deinit();
        var post_mlp = try mlx.Array.fastRmsNorm(down, self.post_mlp_w, self.rms_norm_eps, stream);
        defer post_mlp.deinit();
        var x2 = try mlx.Array.add(x1, post_mlp, stream);
        if (self.layer_scalar == 1.0) return x2;
        defer x2.deinit();
        return x2.mulScalar(self.layer_scalar, stream);
    }

    /// 执行门控多层感知机残差块：已编译宏内核闭包时直接派发单次融合调用，否则逐算子调度 `mlpResidual`
    fn runMlp(self: *const GpuLayerWeights, x1: mlx.Array, stream: ?Stream) !mlx.Array {
        if (self.compiled_mlp) |c_mlp| {
            var in_vec = mlx.VectorArray.fromArrays(&.{x1});
            defer in_vec.deinit();
            var out_vec = try c_mlp.apply(in_vec);
            defer out_vec.deinit();
            return out_vec.get(0) orelse error.MlxOperationFailed;
        }
        return self.mlpResidual(x1, stream);
    }

    /// 使用 Apple MLX Metal 即时编译 (Just-In-Time Compilation, JIT) 编译门控多层感知机 (Multi-Layer Perceptron, MLP) 宏内核闭包
    pub fn initCompiledMlp(self: *GpuLayerWeights) !void {
        if (!isAvailable()) return;
        if (self.compiled_mlp != null) return;

        const Helper = struct {
            fn compute(vres: [*c]mlx.c.mlx_vector_array, in: mlx.c.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
                if (!mlx.has_mlx) return 1;
                const lw: *const GpuLayerWeights = @alignCast(@ptrCast(payload.?));

                var x_handle: mlx.c.mlx_array = .{ .ctx = null };
                if (mlx.c.mlx_vector_array_get(&x_handle, in, 0) != 0 or x_handle.ctx == null) return 1;
                // mlx_vector_array_get 为取出的元素分配了新的数组句柄，用完后需释放
                var x1 = mlx.Array{ .handle = x_handle };
                defer x1.deinit();

                var res_arr = lw.mlpResidual(x1, null) catch return 1;
                defer res_arr.deinit();

                const res_vec = mlx.c.mlx_vector_array_new_data(&[_]mlx.c.mlx_array{res_arr.handle}, 1);
                _ = mlx.c.mlx_vector_array_set(vres, res_vec);
                _ = mlx.c.mlx_vector_array_free(res_vec);
                return 0;
            }
        };

        self.compiled_mlp = try mlx.compile(Helper.compute, self, false);
    }

    pub fn deinit(self: *GpuLayerWeights) void {
        if (self.compiled_mlp) |*c_cls| {
            c_cls.deinit();
            self.compiled_mlp = null;
        }
        self.fused_qkv.deinit();
        self.o_proj.deinit();
        self.fused_gate_up.deinit();
        self.down_proj.deinit();
        self.input_norm_w.deinit();
        self.q_norm_w.deinit();
        self.k_norm_w.deinit();
        self.post_attn_w.deinit();
        self.pre_mlp_w.deinit();
        self.post_mlp_w.deinit();
    }
};

/// 图形处理器 (Graphics Processing Unit, GPU) 端到端整网自回归增量解码上下文 (End-to-End GPU Decode Context)
/// 将全部 Transformer 解码层、均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)、
/// 旋转位置编码 (Rotary Position Embedding, RoPE)、自回归键值缓存 (Key-Value Cache, KVCache) 与
/// 语言模型输出头 (Language Model Head, LM Head) 完全常驻在 Apple Silicon 统一内存显存中。
/// 每生成一个词元仅触发一次最终求值 (Single Evaluation)，消除 145 次 CPU-GPU 往返同步开销。
pub const GpuModelContext = struct {
    allocator: std.mem.Allocator,
    layers: []GpuLayerWeights,
    final_norm_w: mlx.Array,
    lm_head: LmHeadWeights,
    final_eps: f32,
    hidden_size: usize,
    vocab_size: usize,
    max_seq_len: usize,
    alloc_seq_len: usize,
    stream: Stream,

    gpu_k: []?mlx.Array,
    gpu_v: []?mlx.Array,
    curr_len: usize = 0,
    sqrt_d: f32 = 0.0,

    pub fn init(
        allocator: std.mem.Allocator,
        layers: []GpuLayerWeights,
        final_norm_data: []const f32,
        final_eps: f32,
        lm_head: LmHeadWeights,
        hidden_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
    ) !GpuModelContext {
        if (!isAvailable()) return error.OffloadNotSupported;

        var final_norm_w = try mlx.Array.fromSlice(final_norm_data, &.{hidden_size});
        errdefer final_norm_w.deinit();
        final_norm_w.eval();

        const gpu_k = try allocator.alloc(?mlx.Array, layers.len);
        errdefer allocator.free(gpu_k);
        const gpu_v = try allocator.alloc(?mlx.Array, layers.len);
        errdefer allocator.free(gpu_v);
        @memset(gpu_k, null);
        @memset(gpu_v, null);

        const stream = try Stream.defaultGpu();

        mlx.enableCompile();
        mlx.setCompileMode(.enabled);

        for (layers) |*lw| {
            lw.initCompiledMlp() catch {};
        }

        return GpuModelContext{
            .allocator = allocator,
            .layers = layers,
            .final_norm_w = final_norm_w,
            .lm_head = lm_head,
            .final_eps = final_eps,
            .hidden_size = hidden_size,
            .vocab_size = vocab_size,
            .max_seq_len = max_seq_len,
            .alloc_seq_len = max_seq_len,
            .stream = stream,
            .gpu_k = gpu_k,
            .gpu_v = gpu_v,
            .curr_len = 0,
            .sqrt_d = @sqrt(@as(f32, @floatFromInt(hidden_size))),
        };
    }

    pub fn deinit(self: *GpuModelContext) void {
        for (self.layers) |*l| {
            l.deinit();
        }
        self.allocator.free(self.layers);
        self.final_norm_w.deinit();
        self.resetCache();
        self.stream.deinit();
        self.allocator.free(self.gpu_k);
        self.allocator.free(self.gpu_v);
    }

    /// 重置 GPU 显存中的全部键值缓存
    pub fn resetCache(self: *GpuModelContext) void {
        for (self.gpu_k, self.gpu_v) |*k_opt, *v_opt| {
            if (k_opt.*) |*k| k.deinit();
            if (v_opt.*) |*v| v.deinit();
            k_opt.* = null;
            v_opt.* = null;
        }
        self.curr_len = 0;
    }

    /// 将词元序列的脑浮点 16 (Brain Floating Point 16, BF16) 嵌入行展开为单精度浮点并乘以 sqrt(hidden_size)，
    /// 写入 [tokens.len, hidden_size] 行主序缓冲区
    fn embedTokens(self: *const GpuModelContext, tokens: []const usize, embed_u16: []const u16, out: []f32) !void {
        const h = self.hidden_size;
        std.debug.assert(out.len == tokens.len * h);
        for (tokens, 0..) |tok, t| {
            if (tok >= self.vocab_size or (tok + 1) * h > embed_u16.len) return error.TokenOutOfBounds;
            const emb_row = embed_u16[tok * h ..][0..h];
            for (out[t * h ..][0..h], emb_row) |*d, e| {
                d.* = @as(f32, @bitCast(@as(u32, e) << 16)) * self.sqrt_d;
            }
        }
    }

    /// 将新增键值 [1, num_kv_heads, t_new, head_dim] 沿序列轴 (axis 2) 追加至第 l 层 GPU 键值缓存
    fn appendKv(self: *GpuModelContext, l: usize, k_new: mlx.Array, v_new: mlx.Array, s: Stream) !void {
        if (self.gpu_k[l]) |old_k| {
            var new_k = try mlx.Array.concatenateAxis(&.{ old_k, k_new }, 2, s);
            errdefer new_k.deinit();
            const new_v = try mlx.Array.concatenateAxis(&.{ self.gpu_v[l].?, v_new }, 2, s);
            self.replaceKv(l, new_k, new_v);
        } else {
            var cloned_k = try k_new.mulScalar(1.0, s);
            errdefer cloned_k.deinit();
            const cloned_v = try v_new.mulScalar(1.0, s);
            self.replaceKv(l, cloned_k, cloned_v);
        }
    }

    /// 仅保留第 l 层键值缓存序列轴上 [start, stop) 区间的词元
    fn sliceKv(self: *GpuModelContext, l: usize, start: usize, stop: usize, s: Stream) !void {
        var new_k = try self.gpu_k[l].?.sliceAxis(2, start, stop, s);
        errdefer new_k.deinit();
        const new_v = try self.gpu_v[l].?.sliceAxis(2, start, stop, s);
        self.replaceKv(l, new_k, new_v);
    }

    /// 以新数组替换第 l 层键值缓存并释放旧数组
    fn replaceKv(self: *GpuModelContext, l: usize, new_k: mlx.Array, new_v: mlx.Array) void {
        if (self.gpu_k[l]) |*old_k| old_k.deinit();
        if (self.gpu_v[l]) |*old_v| old_v.deinit();
        self.gpu_k[l] = new_k;
        self.gpu_v[l] = new_v;
    }

    /// 滑动窗口注意力层的键值缓存只保留最近 sliding_window 个词元（截取序列尾部，而非固定的 [1, sliding_window + 1) 区间）
    fn trimSlidingKv(self: *GpuModelContext, l: usize, lw: *const GpuLayerWeights, s: Stream) !void {
        if (lw.attn_type != 0) return;
        const len = self.gpu_k[l].?.dimSize(2);
        if (len > lw.sliding_window) try self.sliceKv(l, len - lw.sliding_window, len, s);
    }

    /// 带滑动窗口掩码的分组查询缩放点积注意力 (Sliding Window Grouped-Query Scaled Dot-Product Attention)
    /// q: [1, num_heads, tq, head_dim]，k / v: [1, num_kv_heads, tk, head_dim]；第 i 个查询的绝对位置为 q_pos0 + i，
    /// 第 j 个键的绝对位置为 k_pos0 + j，仅当 0 <= q_pos - k_pos < sliding_window 时可见（不可见位置施加 -1e9 加性掩码）。
    fn slidingWindowAttention(
        self: *const GpuModelContext,
        lw: *const GpuLayerWeights,
        q: mlx.Array,
        k: mlx.Array,
        v: mlx.Array,
        tq: usize,
        tk: usize,
        q_pos0: usize,
        k_pos0: usize,
        s: Stream,
    ) !mlx.Array {
        const groups = lw.num_heads / lw.num_kv_heads;
        var k_trans = try k.transposeAxes(&.{ 0, 1, 3, 2 }, s);
        defer k_trans.deinit();
        // 同一键值头的 groups 个查询头在头维度上相邻，重排为 [1, num_kv_heads, groups * tq, head_dim] 后共享同一组键值
        var q_grouped = try q.reshape(&.{ 1, lw.num_kv_heads, groups * tq, lw.head_dim }, s);
        defer q_grouped.deinit();
        var scores_grouped = try mlx.Array.matmul(q_grouped, k_trans, s);
        defer scores_grouped.deinit();
        var scores = try scores_grouped.reshape(&.{ 1, lw.num_heads, tq, tk }, s);
        defer scores.deinit();

        const mask_buf = try self.allocator.alloc(f32, tq * tk);
        defer self.allocator.free(mask_buf);
        for (0..tq) |i| {
            const q_pos = q_pos0 + i;
            for (0..tk) |j| {
                const k_pos = k_pos0 + j;
                const visible = k_pos <= q_pos and q_pos - k_pos < lw.sliding_window;
                mask_buf[i * tk + j] = if (visible) 0.0 else -1e9;
            }
        }
        var mask_arr = try mlx.Array.fromSlice(mask_buf, &.{ 1, 1, tq, tk });
        defer mask_arr.deinit();
        var masked_scores = try mlx.Array.add(scores, mask_arr, s);
        defer masked_scores.deinit();
        var probs = try mlx.Array.softmax(masked_scores, 3, s);
        defer probs.deinit();

        var probs_grouped = try probs.reshape(&.{ 1, lw.num_kv_heads, groups * tq, tk }, s);
        defer probs_grouped.deinit();
        var out_grouped = try mlx.Array.matmul(probs_grouped, v, s);
        defer out_grouped.deinit();
        return out_grouped.reshape(&.{ 1, lw.num_heads, tq, lw.head_dim }, s);
    }

    /// 从 CPU 预填充阶段 (Prefill) 的键值缓存同步导入至 GPU 显存
    pub fn syncCacheFromCpu(
        self: *GpuModelContext,
        k_slices: []const []const f32,
        v_slices: []const []const f32,
        cached_len: usize,
    ) !void {
        self.resetCache();
        if (cached_len == 0) return;
        if (k_slices.len < self.layers.len or v_slices.len < self.layers.len) return error.ShapeMismatch;

        for (self.layers, 0..) |lw, l_idx| {
            const num_kv = lw.num_kv_heads;
            const hd = lw.head_dim;
            const start_pos: usize = if (lw.attn_type == 0 and cached_len > lw.sliding_window)
                cached_len - lw.sliding_window
            else
                0;
            const eff_len = cached_len - start_pos;
            const packed_len = num_kv * eff_len * hd;

            const cpu_k = k_slices[l_idx];
            const cpu_v = v_slices[l_idx];
            const cache_stride = cpu_k.len / num_kv;
            if (cpu_v.len != cpu_k.len or cached_len * hd > cache_stride) return error.ShapeMismatch;

            const pack_buf_k = try self.allocator.alloc(f32, packed_len);
            defer self.allocator.free(pack_buf_k);
            const pack_buf_v = try self.allocator.alloc(f32, packed_len);
            defer self.allocator.free(pack_buf_v);

            for (0..num_kv) |kv_h| {
                const src_k = cpu_k[kv_h * cache_stride + start_pos * hd .. kv_h * cache_stride + cached_len * hd];
                const dest_k = pack_buf_k[kv_h * eff_len * hd .. (kv_h + 1) * eff_len * hd];
                @memcpy(dest_k, src_k);

                const src_v = cpu_v[kv_h * cache_stride + start_pos * hd .. kv_h * cache_stride + cached_len * hd];
                const dest_v = pack_buf_v[kv_h * eff_len * hd .. (kv_h + 1) * eff_len * hd];
                @memcpy(dest_v, src_v);
            }

            var k_arr = try mlx.Array.fromSlice(pack_buf_k, &.{ 1, num_kv, eff_len, hd });
            errdefer k_arr.deinit();
            k_arr.eval();
            var v_arr = try mlx.Array.fromSlice(pack_buf_v, &.{ 1, num_kv, eff_len, hd });
            v_arr.eval();

            self.replaceKv(l_idx, k_arr, v_arr);
        }
        self.curr_len = cached_len;
    }

    /// 执行全 GPU 序列级并行预填充 (Prefill)，一次性完成整个 Prompt 词元序列的矩阵乘法与 KVCache 初始化
    pub fn prefill(
        self: *GpuModelContext,
        prompt_tokens: []const usize,
        embed_u16: []const u16,
        repetition_penalty: f32,
        stream: ?Stream,
    ) !usize {
        const T = prompt_tokens.len;
        if (T == 0) return 0;
        if (T >= self.alloc_seq_len) return error.ContextWindowExceeded;

        const s = if (stream) |st| st else self.stream;

        // 重置 GPU 键值缓存 (Reset GPU KVCache)
        self.resetCache();

        // 1. 构建 Prompt 词元嵌入 [1, T, hidden_size]
        const tok_emb_buf = try self.allocator.alloc(f32, T * self.hidden_size);
        defer self.allocator.free(tok_emb_buf);
        try self.embedTokens(prompt_tokens, embed_u16, tok_emb_buf);
        var x_cur = try mlx.Array.fromSlice(tok_emb_buf, &.{ 1, T, self.hidden_size });
        errdefer x_cur.deinit();

        for (self.layers, 0..) |lw, l| {
            // (1) 输入均方根层归一化 [1, T, hidden_size]
            var in_norm = try mlx.Array.fastRmsNorm(x_cur, lw.input_norm_w, lw.rms_norm_eps, s);
            defer in_norm.deinit();

            // (2) QKV 融合量化投影 [1, T, total_dim] (Fused GEMM: 1 个内核完成全部 Q, K, V 投影)
            var qkv_raw = try lw.fused_qkv.forwardArray(in_norm, s);
            defer qkv_raw.deinit();

            var q_raw = try qkv_raw.sliceAxis(2, 0, lw.q_dim, s);
            defer q_raw.deinit();
            var k_raw = try qkv_raw.sliceAxis(2, lw.q_dim, lw.q_dim + lw.k_dim, s);
            defer k_raw.deinit();
            var v_raw = if (lw.v_dim > 0)
                try qkv_raw.sliceAxis(2, lw.q_dim + lw.k_dim, lw.q_dim + lw.k_dim + lw.v_dim, s)
            else
                try k_raw.sliceAxis(2, 0, lw.k_dim, s);
            defer v_raw.deinit();

            // (3) 重构形状、在连续内存上执行 Q-Norm/K-Norm/V-Norm，再转置至 [1, heads, T, head_dim]
            var q_4d_raw = try q_raw.reshape(&.{ 1, T, lw.num_heads, lw.head_dim }, s);
            defer q_4d_raw.deinit();
            var k_4d_raw = try k_raw.reshape(&.{ 1, T, lw.num_kv_heads, lw.head_dim }, s);
            defer k_4d_raw.deinit();
            var v_4d_raw = try v_raw.reshape(&.{ 1, T, lw.num_kv_heads, lw.head_dim }, s);
            defer v_4d_raw.deinit();

            var q_normed_raw = try mlx.Array.fastRmsNorm(q_4d_raw, lw.q_norm_w, lw.rms_norm_eps, s);
            defer q_normed_raw.deinit();
            var k_normed_raw = try mlx.Array.fastRmsNorm(k_4d_raw, lw.k_norm_w, lw.rms_norm_eps, s);
            defer k_normed_raw.deinit();
            var v_normed_raw = try mlx.Array.fastRmsNorm(v_4d_raw, null, lw.rms_norm_eps, s);
            defer v_normed_raw.deinit();

            var q_normed = try q_normed_raw.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer q_normed.deinit();
            var k_normed = try k_normed_raw.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer k_normed.deinit();
            var v_normed = try v_normed_raw.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer v_normed.deinit();

            // (5) 旋转位置编码 (Rotary Position Embedding, RoPE)
            var q_rot = try mlx.Array.fastRoPE(q_normed, lw.rot_dim, false, lw.rope_theta, 1.0, 0, s);
            defer q_rot.deinit();
            var k_rot = try mlx.Array.fastRoPE(k_normed, lw.rot_dim, false, lw.rope_theta, 1.0, 0, s);
            defer k_rot.deinit();

            // (6) 写入 GPU 键值缓存 (Initial KVCache on GPU)
            if (lw.attn_type == 0 and T > lw.sliding_window) {
                self.gpu_k[l] = try k_rot.sliceAxis(2, T - lw.sliding_window, T, s);
                self.gpu_v[l] = try v_normed.sliceAxis(2, T - lw.sliding_window, T, s);
            } else {
                self.gpu_k[l] = try k_rot.mulScalar(1.0, s);
                self.gpu_v[l] = try v_normed.mulScalar(1.0, s);
            }

            // (7) 带因果与滑动窗口掩码的缩放点积注意力 (Causal / Sliding Window SDPA)
            var attn_out = if (lw.attn_type == 1 or T <= lw.sliding_window)
                try mlx.Array.fastScaledDotProductAttention(q_rot, k_rot, v_normed, 1.0, true, s)
            else
                try self.slidingWindowAttention(&lw, q_rot, k_rot, v_normed, T, T, 0, 0, s);
            defer attn_out.deinit();

            // (8) 输出投影 (O-projection)
            var attn_t = try attn_out.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer attn_t.deinit();
            var attn_2d = try attn_t.reshape(&.{ 1, T, lw.attn_dim }, s);
            defer attn_2d.deinit();
            var o_out = try lw.o_proj.forwardArray(attn_2d, s);
            defer o_out.deinit();

            // (9) 注意力后层归一化与残差相加
            var post_attn = try mlx.Array.fastRmsNorm(o_out, lw.post_attn_w, lw.rms_norm_eps, s);
            defer post_attn.deinit();
            var x1 = try mlx.Array.add(x_cur, post_attn, s);
            defer x1.deinit();

            // (10) 门控多层感知机残差块 (Fused Gate-Up GEMM + GeGLU)、前馈后层归一化、残差相加与层缩放
            const x2 = try lw.mlpResidual(x1, s);
            var old_x = x_cur;
            x_cur = x2;
            old_x.deinit();
        }

        // 3. 提取最终词元向量并执行 LM Head 投影
        var last_row_3d = try x_cur.sliceAxis(1, T - 1, T, s);
        defer last_row_3d.deinit();
        var last_row_2d = try last_row_3d.reshape(&.{ 1, self.hidden_size }, s);
        defer last_row_2d.deinit();
        x_cur.deinit();

        var h_norm = try mlx.Array.fastRmsNorm(last_row_2d, self.final_norm_w, self.final_eps, s);
        defer h_norm.deinit();

        const max_id: usize = if (repetition_penalty == 1.0 and self.vocab_size > 1) blk: {
            var logits_bf16 = try self.lm_head.forwardArrayBf16(h_norm, s);
            defer logits_bf16.deinit();
            var sliced = try logits_bf16.sliceAxis(1, 1, self.vocab_size, s);
            defer sliced.deinit();
            var best_tok_arr = try sliced.argmax(1, false, s);
            defer best_tok_arr.deinit();
            best_tok_arr.eval();
            break :blk @intCast(best_tok_arr.itemUint32() + 1);
        } else blk: {
            var logits = try self.lm_head.forwardArray(h_norm, s);
            defer logits.deinit();
            logits.eval();
            const logits_slice = logits.dataFloat32();
            if (logits_slice.len != self.vocab_size) return error.ShapeMismatch;
            break :blk LmHeadWeights.argmaxWithPenalty(logits_slice, self.vocab_size, prompt_tokens, repetition_penalty);
        };
        self.curr_len = T;
        return max_id;
    }

    /// 执行单步端到端整网自回归增量解码 (Decode Step)，全程在 GPU 计算图中完成惰性调度
    pub fn decodeStep(
        self: *GpuModelContext,
        emb_row: []const f32,
        recent_tokens: []const usize,
        repetition_penalty: f32,
        stream: ?Stream,
    ) !usize {
        const x_cur = try mlx.Array.fromSlice(emb_row, &.{ 1, self.hidden_size });
        return self.decodeStepWithArray(x_cur, recent_tokens, repetition_penalty, stream);
    }

    /// 基于已初始化的词元嵌入数组执行单步端到端整网自回归增量解码 (Array-based Decode Step)
    pub fn decodeStepWithArray(
        self: *GpuModelContext,
        x_in: mlx.Array,
        recent_tokens: []const usize,
        repetition_penalty: f32,
        stream: ?Stream,
    ) !usize {
        const pos = self.curr_len;
        if (pos >= self.alloc_seq_len) {
            var to_free = x_in;
            to_free.deinit();
            return error.ContextWindowExceeded;
        }

        const s = if (stream) |st| st else self.stream;

        var x_cur = x_in;
        errdefer x_cur.deinit();

        for (self.layers, 0..) |lw, l| {
            // 1. 输入均方根层归一化 (Input RMSNorm)
            var in_norm = try mlx.Array.fastRmsNorm(x_cur, lw.input_norm_w, lw.rms_norm_eps, s);
            defer in_norm.deinit();

            // 2. QKV 融合量化投影 (Fused QKV GEMV: 1 个内核完成全部 Q, K, V 投影)
            var qkv_raw = try lw.fused_qkv.forwardArray(in_norm, s);
            defer qkv_raw.deinit();

            var q_raw = try qkv_raw.slice2dAxis1(0, lw.q_dim, s);
            defer q_raw.deinit();
            var k_raw = try qkv_raw.slice2dAxis1(lw.q_dim, lw.q_dim + lw.k_dim, s);
            defer k_raw.deinit();
            var v_raw = if (lw.v_dim > 0)
                try qkv_raw.slice2dAxis1(lw.q_dim + lw.k_dim, lw.q_dim + lw.k_dim + lw.v_dim, s)
            else
                try k_raw.slice2dAxis1(0, lw.k_dim, s);
            defer v_raw.deinit();

            // 3. 重构形状至 [1, heads, 1, head_dim]
            var q_4d = try q_raw.reshape(&.{ 1, lw.num_heads, 1, lw.head_dim }, s);
            defer q_4d.deinit();
            var k_4d = try k_raw.reshape(&.{ 1, lw.num_kv_heads, 1, lw.head_dim }, s);
            defer k_4d.deinit();
            var v_4d = try v_raw.reshape(&.{ 1, lw.num_kv_heads, 1, lw.head_dim }, s);
            defer v_4d.deinit();

            // 4. Q-Norm, K-Norm, 与无参数 V-Norm
            var q_normed = try mlx.Array.fastRmsNorm(q_4d, lw.q_norm_w, lw.rms_norm_eps, s);
            defer q_normed.deinit();
            var k_normed = try mlx.Array.fastRmsNorm(k_4d, lw.k_norm_w, lw.rms_norm_eps, s);
            defer k_normed.deinit();
            var v_normed = try mlx.Array.fastRmsNorm(v_4d, null, lw.rms_norm_eps, s);
            defer v_normed.deinit();

            // 5. 旋转位置编码 (Rotary Position Embedding, RoPE)
            var q_rot = try mlx.Array.fastRoPE(q_normed, lw.rot_dim, false, lw.rope_theta, 1.0, pos, s);
            defer q_rot.deinit();
            var k_rot = try mlx.Array.fastRoPE(k_normed, lw.rot_dim, false, lw.rope_theta, 1.0, pos, s);
            defer k_rot.deinit();

            // 6. 增量写入 GPU 键值缓存 (KVCache Along Axis 2)，滑动窗口层随后只保留最近 sliding_window 个词元
            try self.appendKv(l, k_rot, v_normed, s);
            try self.trimSlidingKv(l, &lw, s);

            // 7. 快速缩放点积注意力 (Scaled Dot-Product Attention, SDPA / GQA)
            var attn_out = try mlx.Array.fastScaledDotProductAttention(
                q_rot,
                self.gpu_k[l].?,
                self.gpu_v[l].?,
                1.0,
                false,
                s,
            );
            defer attn_out.deinit();

            // 8. 输出投影 (O-projection)
            var attn_2d = try attn_out.reshape(&.{ 1, lw.attn_dim }, s);
            defer attn_2d.deinit();
            var o_out = try lw.o_proj.forwardArray(attn_2d, s);
            defer o_out.deinit();

            // 9. 注意力后层归一化与残差相加
            var post_attn = try mlx.Array.fastRmsNorm(o_out, lw.post_attn_w, lw.rms_norm_eps, s);
            defer post_attn.deinit();
            var x1 = try mlx.Array.add(x_cur, post_attn, s);
            defer x1.deinit();

            // 10. 门控多层感知机 (Fused Gate-Up GEMV + GeGLU 融合激活 + 静态编译宏内核)、前馈后层归一化、残差相加与层缩放
            const x2 = try lw.runMlp(x1, s);
            var old_x = x_cur;
            x_cur = x2;
            old_x.deinit();
        }

        // 最终均方根层归一化 (Final RMSNorm)
        var h_norm = try mlx.Array.fastRmsNorm(x_cur, self.final_norm_w, self.final_eps, s);
        defer h_norm.deinit();
        x_cur.deinit();

        // 最终词表投影 (LM Head MatMul: 自动走 Q8 量化加速或密集矩阵乘，当无重复惩罚时直接在 GPU 上执行 BF16 argmax)
        const max_id: usize = if (repetition_penalty == 1.0 and self.vocab_size > 1) blk: {
            var logits_bf16 = try self.lm_head.forwardArrayBf16(h_norm, s);
            defer logits_bf16.deinit();
            var sliced = try logits_bf16.sliceAxis(1, 1, self.vocab_size, s);
            defer sliced.deinit();
            var best_tok_arr = try sliced.argmax(1, false, s);
            defer best_tok_arr.deinit();
            best_tok_arr.eval();
            break :blk @intCast(best_tok_arr.itemUint32() + 1);
        } else blk: {
            var logits = try self.lm_head.forwardArray(h_norm, s);
            defer logits.deinit();
            logits.eval();
            const logits_slice = logits.dataFloat32();
            if (logits_slice.len != self.vocab_size) return error.ShapeMismatch;
            break :blk LmHeadWeights.argmaxWithPenalty(logits_slice, self.vocab_size, recent_tokens, repetition_penalty);
        };
        self.curr_len += 1;
        return max_id;
    }

    /// 执行投机采样批量验证前向传播 (Speculative Draft Verification Step)
    /// 一次性将基础词元 (input_tokens[0]) 与投机候选词元 (input_tokens[1..M])
    /// 在图形处理器 (GPU) 上完成因果注意力与全网前向传播（仅单次遍历读取 7.59 GB 权重），
    /// 并逐位贪婪校验接受候选词元。
    /// 返回值：实际被接受并写入 KVCache 的词元数量 (1 <= num_accepted <= input_tokens.len)
    pub fn decodeSpeculative(
        self: *GpuModelContext,
        input_tokens: []const usize,
        embed_u16: []const u16,
        recent_tokens: []const usize,
        repetition_penalty: f32,
        stream: ?Stream,
        accepted_tokens: []usize,
    ) !usize {
        const M = input_tokens.len;
        if (M == 0) return 0;
        if (accepted_tokens.len < M) return error.ShapeMismatch;
        const pos = self.curr_len;
        if (pos + M >= self.alloc_seq_len) return error.ContextWindowExceeded;

        const s = if (stream) |st| st else self.stream;

        // 1. 构建 M 个词元的嵌入向量 [1, M, hidden_size]
        const tok_emb_buf = try self.allocator.alloc(f32, M * self.hidden_size);
        defer self.allocator.free(tok_emb_buf);
        try self.embedTokens(input_tokens, embed_u16, tok_emb_buf);

        var x_cur = try mlx.Array.fromSlice(tok_emb_buf, &.{ 1, M, self.hidden_size });
        errdefer x_cur.deinit();

        for (self.layers, 0..) |lw, l| {
            // (1) 输入均方根层归一化 [1, M, hidden_size]
            var in_norm = try mlx.Array.fastRmsNorm(x_cur, lw.input_norm_w, lw.rms_norm_eps, s);
            defer in_norm.deinit();

            // (2) QKV 融合量化投影 [1, M, total_dim]
            var qkv_raw = try lw.fused_qkv.forwardArray(in_norm, s);
            defer qkv_raw.deinit();

            var q_raw = try qkv_raw.sliceAxis(2, 0, lw.q_dim, s);
            defer q_raw.deinit();
            var k_raw = try qkv_raw.sliceAxis(2, lw.q_dim, lw.q_dim + lw.k_dim, s);
            defer k_raw.deinit();
            var v_raw = if (lw.v_dim > 0)
                try qkv_raw.sliceAxis(2, lw.q_dim + lw.k_dim, lw.q_dim + lw.k_dim + lw.v_dim, s)
            else
                try k_raw.sliceAxis(2, 0, lw.k_dim, s);
            defer v_raw.deinit();

            // (3) 重构形状与转置至 [1, heads, M, head_dim]
            var q_4d_raw = try q_raw.reshape(&.{ 1, M, lw.num_heads, lw.head_dim }, s);
            defer q_4d_raw.deinit();
            var q_4d = try q_4d_raw.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer q_4d.deinit();

            var k_4d_raw = try k_raw.reshape(&.{ 1, M, lw.num_kv_heads, lw.head_dim }, s);
            defer k_4d_raw.deinit();
            var k_4d = try k_4d_raw.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer k_4d.deinit();

            var v_4d_raw = try v_raw.reshape(&.{ 1, M, lw.num_kv_heads, lw.head_dim }, s);
            defer v_4d_raw.deinit();
            var v_4d = try v_4d_raw.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer v_4d.deinit();

            // (4) Q-Norm, K-Norm 与无参数 V-Norm
            var q_normed = try mlx.Array.fastRmsNorm(q_4d, lw.q_norm_w, lw.rms_norm_eps, s);
            defer q_normed.deinit();
            var k_normed = try mlx.Array.fastRmsNorm(k_4d, lw.k_norm_w, lw.rms_norm_eps, s);
            defer k_normed.deinit();
            var v_normed = try mlx.Array.fastRmsNorm(v_4d, null, lw.rms_norm_eps, s);
            defer v_normed.deinit();

            // (5) 旋转位置编码 (Rotary Position Embedding, RoPE)
            var q_rot = try mlx.Array.fastRoPE(q_normed, lw.rot_dim, false, lw.rope_theta, 1.0, pos, s);
            defer q_rot.deinit();
            var k_rot = try mlx.Array.fastRoPE(k_normed, lw.rot_dim, false, lw.rope_theta, 1.0, pos, s);
            defer k_rot.deinit();

            // (6) 增量追加至 GPU 键值缓存（滑动窗口层在本步结束后再截取最近 sliding_window 个词元）
            try self.appendKv(l, k_rot, v_normed, s);
            const kv_len = self.gpu_k[l].?.dimSize(2);

            // (7) 因果注意力机制 (Causal Scaled Dot-Product Attention)：缓存中的第 j 个键对应绝对位置 pos + M - kv_len + j；
            //     滑动窗口层的缓存超过窗口长度时，较早的查询会看到窗口之外的键，必须改用显式滑动窗口掩码
            var attn_out = if (lw.attn_type == 1 or kv_len <= lw.sliding_window)
                try mlx.Array.fastScaledDotProductAttention(q_rot, self.gpu_k[l].?, self.gpu_v[l].?, 1.0, true, s)
            else
                try self.slidingWindowAttention(&lw, q_rot, self.gpu_k[l].?, self.gpu_v[l].?, M, kv_len, pos, pos + M - kv_len, s);
            defer attn_out.deinit();

            // (8) 输出投影 (O-projection)
            var attn_t = try attn_out.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer attn_t.deinit();
            var attn_2d = try attn_t.reshape(&.{ 1, M, lw.attn_dim }, s);
            defer attn_2d.deinit();
            var o_out = try lw.o_proj.forwardArray(attn_2d, s);
            defer o_out.deinit();

            // (9) 注意力后层归一化与残差相加
            var post_attn = try mlx.Array.fastRmsNorm(o_out, lw.post_attn_w, lw.rms_norm_eps, s);
            defer post_attn.deinit();
            var x1 = try mlx.Array.add(x_cur, post_attn, s);
            defer x1.deinit();

            // (10) 门控多层感知机 (Fused Gate-Up GEMV + GeGLU 融合激活 + 静态编译宏内核)、前馈后层归一化、残差相加与层缩放
            const x2 = try lw.runMlp(x1, s);
            var old_x = x_cur;
            x_cur = x2;
            old_x.deinit();
        }

        // 2. 最终层 RMSNorm 与 LM Head 词表投影
        var h_norm = try mlx.Array.fastRmsNorm(x_cur, self.final_norm_w, self.final_eps, s);
        defer h_norm.deinit();
        x_cur.deinit();

        var logits = try self.lm_head.forwardArray(h_norm, s);
        defer logits.deinit();

        logits.eval();

        const logits_slice = logits.dataFloat32();
        if (logits_slice.len != M * self.vocab_size) return error.ShapeMismatch;

        // 3. 逐位贪婪校验候选词元 (Greedy Verification with Penalty)
        var verified_tokens = try std.ArrayList(usize).initCapacity(self.allocator, recent_tokens.len + M);
        defer verified_tokens.deinit(self.allocator);
        try verified_tokens.appendSlice(self.allocator, recent_tokens);

        var num_accepted: usize = 0;
        for (0..M) |step_i| {
            const row = logits_slice[step_i * self.vocab_size .. (step_i + 1) * self.vocab_size];
            const rec_window = if (verified_tokens.items.len > 64) verified_tokens.items[verified_tokens.items.len - 64 ..] else verified_tokens.items;
            const pred = LmHeadWeights.argmaxWithPenalty(row, self.vocab_size, rec_window, repetition_penalty);
            accepted_tokens[step_i] = pred;
            num_accepted += 1;
            try verified_tokens.append(self.allocator, pred);

            if (step_i + 1 < M) {
                // 投机命中校验：若预测词元与下一个投机候选词元不一致，则停止接受后续投机词元
                if (pred != input_tokens[step_i + 1]) {
                    break;
                }
            }
        }

        // 4. 回滚未被接受的投机词元 KVCache（从缓存尾部丢弃，滑动窗口层的缓存并不从位置 0 开始），
        //    随后滑动窗口层只保留最近 sliding_window 个词元 (惰性图切片，零同步开销)
        const rejected = M - num_accepted;
        for (self.layers, 0..) |*lw, l| {
            if (rejected > 0) {
                const len = self.gpu_k[l].?.dimSize(2);
                try self.sliceKv(l, 0, len - rejected, s);
            }
            try self.trimSlidingKv(l, lw, s);
        }

        self.curr_len = pos + num_accepted;
        return num_accepted;
    }

    /// 执行 Metal 计算管线预热 (Pipeline Pre-warming)，通过微型前向传递触发 MSL JIT 编译并构建管线状态对象 (PSO)
    pub fn warmup(self: *GpuModelContext, embed_u16: []const u16, stream: ?Stream) !void {
        if (self.vocab_size < 3) return;
        const dummy_prompt = [_]usize{ 1, 2 };
        _ = try self.prefill(&dummy_prompt, embed_u16, 1.0, stream);

        var dummy_row: [4096]f32 = undefined;
        if (self.hidden_size <= dummy_row.len) {
            const row_slice = dummy_row[0..self.hidden_size];
            @memset(row_slice, 0.0);
            _ = try self.decodeStep(row_slice, &dummy_prompt, 1.0, stream);
        }

        const dummy_spec = [_]usize{ 1, 2 };
        var accepted_dummy: [2]usize = undefined;
        _ = self.decodeSpeculative(&dummy_spec, embed_u16, &dummy_prompt, 1.0, stream, &accepted_dummy) catch {};

        self.resetCache();
        if (stream) |s| s.synchronize();
    }
};

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
