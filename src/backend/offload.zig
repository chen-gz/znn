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
        w_t.eval();
        if (stream) |st| st.synchronize();

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

    /// 执行 GPU 词表打分投影计算图调度：[..., hidden_size] -> [..., vocab_size]
    pub fn forwardArray(self: LmHeadWeights, x: mlx.Array, stream: ?Stream) OffloadError!mlx.Array {
        if (!isAvailable()) return error.OffloadNotSupported;
        if (self.is_quantized) {
            var x_bf16 = try x.astype(.bfloat16, stream);
            defer x_bf16.deinit();
            var logits_bf16 = try x_bf16.quantizedMatmul(
                self.w_q.?,
                self.scales.?,
                self.biases,
                true, // transpose = true -> x @ W^T
                self.group_size,
                self.bits,
                "affine",
                stream,
            );
            defer logits_bf16.deinit();
            return logits_bf16.astype(.float32, stream);
        } else {
            var x_bf16 = try x.astype(.bfloat16, stream);
            defer x_bf16.deinit();
            var logits_bf16 = try mlx.Array.matmul(x_bf16, self.w_t.?, stream);
            defer logits_bf16.deinit();
            return logits_bf16.astype(.float32, stream);
        }
    }

    /// 在已计算的词表对数几率切片上执行分块位图重复惩罚与贪婪采样 (Chunked Bitset Penalty & Fast-Path Argmax)
    /// 99.8% 的 64 词块 word == 0，直接走纯向量化比较快车道，仅在命中词块内执行位运算，兼具纯只读安全与极速
    pub fn argmaxWithPenalty(
        logits_slice: []const f32,
        vocab_size: usize,
        recent_tokens: []const usize,
        repetition_penalty: f32,
    ) usize {
        var stack_bitset = [_]u64{0} ** 4096;
        const num_words = (vocab_size + 63) / 64;
        const recent_bitset: []u64 = if (num_words <= stack_bitset.len)
            stack_bitset[0..num_words]
        else
            std.heap.c_allocator.alloc(u64, num_words) catch stack_bitset[0..];
        defer if (num_words > stack_bitset.len and recent_bitset.ptr != &stack_bitset) std.heap.c_allocator.free(recent_bitset);
        if (num_words > stack_bitset.len and recent_bitset.ptr != &stack_bitset) {
            @memset(recent_bitset, 0);
        }

        if (repetition_penalty > 1.0) {
            for (recent_tokens) |rec| {
                if (rec < vocab_size and (rec / 64) < recent_bitset.len) {
                    recent_bitset[rec / 64] |= (@as(u64, 1) << @intCast(rec % 64));
                }
            }
        }

        var max_val: f32 = -1e30;
        var max_id: usize = 0;

        for (0..num_words) |w| {
            const word = recent_bitset[w];
            const base_v = w * 64;
            const end_v = @min(base_v + 64, vocab_size);

            if (word == 0) {
                // 快车道 (Fast Path): 该 64 词块内无任何重复词元
                if (w != 0 and w != 4045) {
                    // 无特殊排除词元，直接进行纯向量化极速最大值查找（零条件分支）
                    for (base_v..end_v) |v| {
                        const dot = logits_slice[v];
                        if (dot > max_val) {
                            max_val = dot;
                            max_id = v;
                        }
                    }
                } else {
                    for (base_v..end_v) |v| {
                        if (v == 0 or (v >= 258880 and v <= 258884)) continue;
                        const dot = logits_slice[v];
                        if (dot > max_val) {
                            max_val = dot;
                            max_id = v;
                        }
                    }
                }
            } else {
                // 精准道: 仅对包含最近词元的少数几个词块执行位掩码测试与重复惩罚
                if (w != 0 and w != 4045) {
                    for (base_v..end_v) |v| {
                        var dot = logits_slice[v];
                        if ((word & (@as(u64, 1) << @intCast(v % 64))) != 0) {
                            dot = if (dot > 0.0) dot / repetition_penalty else dot * repetition_penalty;
                        }
                        if (dot > max_val) {
                            max_val = dot;
                            max_id = v;
                        }
                    }
                } else {
                    for (base_v..end_v) |v| {
                        if (v == 0 or (v >= 258880 and v <= 258884)) continue;
                        var dot = logits_slice[v];
                        if ((word & (@as(u64, 1) << @intCast(v % 64))) != 0) {
                            dot = if (dot > 0.0) dot / repetition_penalty else dot * repetition_penalty;
                        }
                        if (dot > max_val) {
                            max_val = dot;
                            max_id = v;
                        }
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

        var global_blk_idx: usize = 0;
        for (block_slices, 0..) |slice, s_idx| {
            const slice_blocks = out_features_list[s_idx] * blocks_per_row;
            for (0..slice_blocks) |b_idx| {
                const raw = slice[b_idx * 18 .. (b_idx + 1) * 18];
                const scale_bits = std.mem.readInt(u16, raw[0..2][0..2], .little);
                const scale_f16: f16 = @bitCast(scale_bits);
                const scale_f32: f32 = @floatCast(scale_f16);
                const bias_f16: f16 = @floatCast(-8.0 * scale_f32);

                scales_buf[global_blk_idx] = scale_f16;
                biases_buf[global_blk_idx] = bias_f16;

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
                w_q_buf[global_blk_idx * 4 + 0] = w0;
                w_q_buf[global_blk_idx * 4 + 1] = w1;
                w_q_buf[global_blk_idx * 4 + 2] = w2;
                w_q_buf[global_blk_idx * 4 + 3] = w3;

                global_blk_idx += 1;
            }
        }

        const w_dims = [_]usize{ total_out_features, in_features / 8 };
        const sb_dims = [_]usize{ total_out_features, in_features / 32 };

        var w_q = try mlx.Array.fromRawData(@ptrCast(w_q_buf.ptr), &w_dims, .uint32);
        errdefer w_q.deinit();
        var scales = try mlx.Array.fromRawData(@ptrCast(scales_buf.ptr), &sb_dims, .float16);
        errdefer scales.deinit();
        var biases = try mlx.Array.fromRawData(@ptrCast(biases_buf.ptr), &sb_dims, .float16);
        errdefer biases.deinit();

        w_q.eval();
        scales.eval();
        biases.eval();
        if (stream) |s| s.synchronize();

        return .{
            .w_q = w_q,
            .scales = scales,
            .biases = biases,
            .in_features = in_features,
            .out_features = total_out_features,
        };
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
            32,
            4,
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
        const rows = total_elements / in_f;

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
    const rows = total_elements / in_f;

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
    if (res_data.len != rows * in_f) return error.ShapeMismatch;
    @memcpy(out.data[0..rows * in_f], res_data);
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
        @memcpy(q_out.data[0..rows * q_qw.out_features], q_data);
        @memcpy(k_out.data[0..rows * k_qw.out_features], k_data);
        @memcpy(v_out.data[0..rows * vw.out_features], v_data);
    } else {
        mlx.Array.evalMultiple(&.{ q_arr, k_arr });

        const q_data = q_arr.dataFloat32();
        const k_data = k_arr.dataFloat32();
        @memcpy(q_out.data[0..rows * q_qw.out_features], q_data);
        @memcpy(k_out.data[0..rows * k_qw.out_features], k_data);
        @memcpy(v_out.data[0..rows * k_qw.out_features], k_data);
    }
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
        in_w.eval();
        var qn_w = try mlx.Array.fromSlice(qn_w_data, &.{qn_w_data.len});
        errdefer qn_w.deinit();
        qn_w.eval();
        var kn_w = try mlx.Array.fromSlice(kn_w_data, &.{kn_w_data.len});
        errdefer kn_w.deinit();
        kn_w.eval();
        var pa_w = try mlx.Array.fromSlice(pa_w_data, &.{pa_w_data.len});
        errdefer pa_w.deinit();
        pa_w.eval();
        var pm_w = try mlx.Array.fromSlice(pm_w_data, &.{pm_w_data.len});
        errdefer pm_w.deinit();
        pm_w.eval();
        var pom_w = try mlx.Array.fromSlice(pom_w_data, &.{pom_w_data.len});
        errdefer pom_w.deinit();
        pom_w.eval();

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
        };
    }

    pub fn deinit(self: *GpuLayerWeights) void {
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

    /// 从 CPU 预填充阶段 (Prefill) 的键值缓存同步导入至 GPU 显存
    pub fn syncCacheFromCpu(
        self: *GpuModelContext,
        k_slices: []const []const f32,
        v_slices: []const []const f32,
        cached_len: usize,
    ) !void {
        self.resetCache();
        if (cached_len == 0) return;

        for (self.layers, 0..) |lw, l_idx| {
            const num_kv = lw.num_kv_heads;
            const hd = lw.head_dim;
            const start_pos: usize = if (lw.attn_type == 0 and cached_len > lw.sliding_window)
                cached_len - lw.sliding_window
            else
                0;
            const eff_len = cached_len - start_pos;
            const packed_len = num_kv * eff_len * hd;
            const pack_buf_k = try self.allocator.alloc(f32, packed_len);
            defer self.allocator.free(pack_buf_k);
            const pack_buf_v = try self.allocator.alloc(f32, packed_len);
            defer self.allocator.free(pack_buf_v);

            const cpu_k = k_slices[l_idx];
            const cpu_v = v_slices[l_idx];
            const cache_stride = cpu_k.len / num_kv;
            for (0..num_kv) |kv_h| {
                const src_k = cpu_k[kv_h * cache_stride + start_pos * hd .. kv_h * cache_stride + cached_len * hd];
                const dest_k = pack_buf_k[kv_h * eff_len * hd .. (kv_h + 1) * eff_len * hd];
                @memcpy(dest_k, src_k);

                const src_v = cpu_v[kv_h * cache_stride + start_pos * hd .. kv_h * cache_stride + cached_len * hd];
                const dest_v = pack_buf_v[kv_h * eff_len * hd .. (kv_h + 1) * eff_len * hd];
                @memcpy(dest_v, src_v);
            }

            var k_arr = try mlx.Array.fromSlice(pack_buf_k, &.{ 1, num_kv, eff_len, hd });
            k_arr.eval();
            var v_arr = try mlx.Array.fromSlice(pack_buf_v, &.{ 1, num_kv, eff_len, hd });
            v_arr.eval();

            self.gpu_k[l_idx] = k_arr;
            self.gpu_v[l_idx] = v_arr;
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

        for (prompt_tokens, 0..) |tok, t_idx| {
            const emb_row = embed_u16[tok * self.hidden_size .. (tok + 1) * self.hidden_size];
            const dest = tok_emb_buf[t_idx * self.hidden_size .. (t_idx + 1) * self.hidden_size];
            for (0..self.hidden_size) |d| {
                const u: u32 = @as(u32, emb_row[d]) << 16;
                dest[d] = @as(f32, @bitCast(u)) * self.sqrt_d;
            }
        }
        var x_cur = try mlx.Array.fromSlice(tok_emb_buf, &.{ 1, T, self.hidden_size });

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

            // (3) 重构形状与转置至 [1, heads, T, head_dim]
            var q_4d_raw = try q_raw.reshape(&.{ 1, T, lw.num_heads, lw.head_dim }, s);
            defer q_4d_raw.deinit();
            var q_4d = try q_4d_raw.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer q_4d.deinit();

            var k_4d_raw = try k_raw.reshape(&.{ 1, T, lw.num_kv_heads, lw.head_dim }, s);
            defer k_4d_raw.deinit();
            var k_4d = try k_4d_raw.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer k_4d.deinit();

            var v_4d_raw = try v_raw.reshape(&.{ 1, T, lw.num_kv_heads, lw.head_dim }, s);
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
                try mlx.Array.fastScaledDotProductAttention(
                    q_rot,
                    k_rot,
                    v_normed,
                    1.0,
                    true,
                    s,
                )
            else blk: {
                var k_trans = try k_rot.transposeAxes(&.{ 0, 1, 3, 2 }, s);
                defer k_trans.deinit();
                const groups = lw.num_heads / lw.num_kv_heads;
                var q_grouped = if (groups > 1)
                    try q_rot.reshape(&.{ 1, lw.num_kv_heads, groups * T, lw.head_dim }, s)
                else
                    try q_rot.mulScalar(1.0, s);
                defer q_grouped.deinit();

                var scores_grouped = try mlx.Array.matmul(q_grouped, k_trans, s);
                defer scores_grouped.deinit();
                var scores_4d = if (groups > 1)
                    try scores_grouped.reshape(&.{ 1, lw.num_heads, T, T }, s)
                else
                    try scores_grouped.mulScalar(1.0, s);
                defer scores_4d.deinit();

                const mask_buf = try self.allocator.alloc(f32, T * T);
                defer self.allocator.free(mask_buf);
                for (0..T) |i| {
                    for (0..T) |j| {
                        if (j > i or i >= j + lw.sliding_window) {
                            mask_buf[i * T + j] = -1e9;
                        } else {
                            mask_buf[i * T + j] = 0.0;
                        }
                    }
                }
                var mask_arr = try mlx.Array.fromSlice(mask_buf, &.{ 1, 1, T, T });
                defer mask_arr.deinit();
                var masked_scores = try mlx.Array.add(scores_4d, mask_arr, s);
                defer masked_scores.deinit();
                var probs = try mlx.Array.softmax(masked_scores, 3, s);
                defer probs.deinit();

                var probs_grouped = if (groups > 1)
                    try probs.reshape(&.{ 1, lw.num_kv_heads, groups * T, T }, s)
                else
                    try probs.mulScalar(1.0, s);
                defer probs_grouped.deinit();
                var out_grouped = try mlx.Array.matmul(probs_grouped, v_normed, s);
                defer out_grouped.deinit();
                break :blk if (groups > 1)
                    try out_grouped.reshape(&.{ 1, lw.num_heads, T, lw.head_dim }, s)
                else
                    try out_grouped.mulScalar(1.0, s);
            };
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

            // (10) 门控多层感知机 (Fused Gate-Up GEMV: 1 个内核完成 Gate 与 Up 投影)
            var pre_mlp = try mlx.Array.fastRmsNorm(x1, lw.pre_mlp_w, lw.rms_norm_eps, s);
            defer pre_mlp.deinit();

            var gate_up = try lw.fused_gate_up.forwardArray(pre_mlp, s);
            defer gate_up.deinit();

            var fused = try gate_up.geglu(lw.intermediate_size, s);
            defer fused.deinit();
            var down = try lw.down_proj.forwardArray(fused, s);
            defer down.deinit();

            // (11) 前馈后层归一化、残差相加与层缩放
            var post_mlp = try mlx.Array.fastRmsNorm(down, lw.post_mlp_w, lw.rms_norm_eps, s);
            defer post_mlp.deinit();
            var x2 = try mlx.Array.add(x1, post_mlp, s);

            var old_x = x_cur;
            if (lw.layer_scalar != 1.0) {
                x_cur = try x2.mulScalar(lw.layer_scalar, s);
                x2.deinit();
            } else {
                x_cur = x2;
            }
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

        var logits = try self.lm_head.forwardArray(h_norm, s);
        defer logits.deinit();

        logits.eval();

        const logits_slice = logits.dataFloat32();
        if (logits_slice.len != self.vocab_size) return error.ShapeMismatch;

        const max_id = LmHeadWeights.argmaxWithPenalty(logits_slice, self.vocab_size, prompt_tokens, repetition_penalty);
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

            // 6. 增量写入 GPU 键值缓存 (KVCache Along Axis 2)
            if (self.gpu_k[l]) |old_k| {
                var new_k = try mlx.Array.concatenateAxis(&.{ old_k, k_rot }, 2, s);
                var new_v = try mlx.Array.concatenateAxis(&.{ self.gpu_v[l].?, v_normed }, 2, s);
                var k_to_free = old_k;
                var v_to_free = self.gpu_v[l].?;

                const k_len = pos + 1;
                if (lw.attn_type == 0 and k_len > lw.sliding_window) {
                    const sliced_k = try new_k.sliceAxis(2, 1, lw.sliding_window + 1, s);
                    const sliced_v = try new_v.sliceAxis(2, 1, lw.sliding_window + 1, s);
                    new_k.deinit();
                    new_v.deinit();
                    new_k = sliced_k;
                    new_v = sliced_v;
                }
                self.gpu_k[l] = new_k;
                self.gpu_v[l] = new_v;
                k_to_free.deinit();
                v_to_free.deinit();
            } else {
                const cloned_k = try k_rot.mulScalar(1.0, s);
                const cloned_v = try v_normed.mulScalar(1.0, s);
                self.gpu_k[l] = cloned_k;
                self.gpu_v[l] = cloned_v;
            }

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

            // 10. 门控多层感知机 (Fused Gate-Up GEMV + GeGLU 融合激活)
            var pre_mlp = try mlx.Array.fastRmsNorm(x1, lw.pre_mlp_w, lw.rms_norm_eps, s);
            defer pre_mlp.deinit();

            var gate_up = try lw.fused_gate_up.forwardArray(pre_mlp, s);
            defer gate_up.deinit();

            var fused = try gate_up.geglu(lw.intermediate_size, s);
            defer fused.deinit();

            var down = try lw.down_proj.forwardArray(fused, s);
            defer down.deinit();

            // 11. 前馈后层归一化、残差相加与层缩放
            var post_mlp = try mlx.Array.fastRmsNorm(down, lw.post_mlp_w, lw.rms_norm_eps, s);
            defer post_mlp.deinit();
            var x2 = try mlx.Array.add(x1, post_mlp, s);

            var old_x = x_cur;
            if (lw.layer_scalar != 1.0) {
                x_cur = try x2.mulScalar(lw.layer_scalar, s);
                x2.deinit();
            } else {
                x_cur = x2;
            }
            old_x.deinit();
        }

        // 最终均方根层归一化 (Final RMSNorm)
        var h_norm = try mlx.Array.fastRmsNorm(x_cur, self.final_norm_w, self.final_eps, s);
        defer h_norm.deinit();
        x_cur.deinit();

        // 最终词表投影 (LM Head MatMul: 自动走 Q8 量化加速或密集矩阵乘)
        var logits = try self.lm_head.forwardArray(h_norm, s);
        defer logits.deinit();

        // 单一终点强制求值 (Single Evaluation): 全网 48 层 + RMSNorm + LM Head 一次性在 Metal GPU 流水线中完成
        logits.eval();

        const logits_slice = logits.dataFloat32();
        if (logits_slice.len != self.vocab_size) return error.ShapeMismatch;

        const max_id = LmHeadWeights.argmaxWithPenalty(logits_slice, self.vocab_size, recent_tokens, repetition_penalty);
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
        const pos = self.curr_len;
        if (pos + M >= self.alloc_seq_len) return error.ContextWindowExceeded;

        const s = if (stream) |st| st else self.stream;

        // 1. 构建 M 个词元的嵌入向量 [1, M, hidden_size]
        const tok_emb_buf = try self.allocator.alloc(f32, M * self.hidden_size);
        defer self.allocator.free(tok_emb_buf);

        for (input_tokens, 0..) |tok, i| {
            if (tok >= self.vocab_size) return error.TokenOutOfBounds;
            const emb_row = embed_u16[tok * self.hidden_size .. (tok + 1) * self.hidden_size];
            const dest = tok_emb_buf[i * self.hidden_size .. (i + 1) * self.hidden_size];
            for (0..self.hidden_size) |d| {
                const u: u32 = @as(u32, emb_row[d]) << 16;
                dest[d] = @as(f32, @bitCast(u)) * self.sqrt_d;
            }
        }

        var x_cur = try mlx.Array.fromSlice(tok_emb_buf, &.{ 1, M, self.hidden_size });

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

            // (6) 增量追加至 GPU 键值缓存 [1, num_kv_heads, pos + M, head_dim]
            if (self.gpu_k[l]) |old_k| {
                const new_k = try mlx.Array.concatenateAxis(&.{ old_k, k_rot }, 2, s);
                const new_v = try mlx.Array.concatenateAxis(&.{ self.gpu_v[l].?, v_normed }, 2, s);
                var k_to_free = old_k;
                var v_to_free = self.gpu_v[l].?;
                self.gpu_k[l] = new_k;
                self.gpu_v[l] = new_v;
                k_to_free.deinit();
                v_to_free.deinit();
            } else {
                self.gpu_k[l] = try k_rot.mulScalar(1.0, s);
                self.gpu_v[l] = try v_normed.mulScalar(1.0, s);
            }

            // (7) 因果注意力机制 (Causal Scaled Dot-Product Attention)
            var attn_out = try mlx.Array.fastScaledDotProductAttention(
                q_rot,
                self.gpu_k[l].?,
                self.gpu_v[l].?,
                1.0,
                true,
                s,
            );
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

            // (10) 门控多层感知机 (Fused Gate-Up GEMV)
            var pre_mlp = try mlx.Array.fastRmsNorm(x1, lw.pre_mlp_w, lw.rms_norm_eps, s);
            defer pre_mlp.deinit();

            var gate_up = try lw.fused_gate_up.forwardArray(pre_mlp, s);
            defer gate_up.deinit();

            var fused = try gate_up.geglu(lw.intermediate_size, s);
            defer fused.deinit();
            var down = try lw.down_proj.forwardArray(fused, s);
            defer down.deinit();

            // (11) 前馈后层归一化、残差相加与层缩放
            var post_mlp = try mlx.Array.fastRmsNorm(down, lw.post_mlp_w, lw.rms_norm_eps, s);
            defer post_mlp.deinit();
            var x2 = try mlx.Array.add(x1, post_mlp, s);

            var old_x = x_cur;
            if (lw.layer_scalar != 1.0) {
                x_cur = try x2.mulScalar(lw.layer_scalar, s);
                x2.deinit();
            } else {
                x_cur = x2;
            }
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

        // 4. 回滚未被接受的投机词元 KVCache
        if (num_accepted < M) {
            const target_len = pos + num_accepted;
            for (self.layers, 0..) |_, l| {
                var old_k = self.gpu_k[l].?;
                var old_v = self.gpu_v[l].?;
                self.gpu_k[l] = try old_k.sliceAxis(2, 0, target_len, s);
                self.gpu_v[l] = try old_v.sliceAxis(2, 0, target_len, s);
                old_k.deinit();
                old_v.deinit();
            }
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

test "backend offload infrastructure" {
    if (!isAvailable()) return;

    var stream = try Stream.defaultGpu();
    defer stream.deinit();

    // 1. 验证 LmHeadWeights 词表投影卸载
    // 构造 4 个词元、维度为 2 的测试词表 (BF16)
    // token 0: [1.0, 0.0] -> 0x3f80, 0x0000
    // token 1: [0.0, 2.0] -> 0x0000, 0x4000
    // token 2: [3.0, 3.0] -> 0x4040, 0x4040 (最大值目标)
    // token 3: [0.5, 0.5] -> 0x3f00, 0x3f00
    const embed_u16 = [_]u16{
        0x3f80, 0x0000,
        0x0000, 0x4000,
        0x4040, 0x4040,
        0x3f00, 0x3f00,
    };

    var lm_head = try LmHeadWeights.init(&embed_u16, 4, 2, stream);
    defer lm_head.deinit();

    const h_last = [_]f32{ 1.0, 1.0 };
    // 无重复惩罚时，token 2 得分 1.0*3.0 + 1.0*3.0 = 6.0 最高
    const best_tok = try lm_head.sampleNext(&h_last, &.{}, 1.0, stream);
    try std.testing.expectEqual(@as(usize, 2), best_tok);

    // 对 token 2 施加极端惩罚 10.0，使得分变为 0.6，此时 token 1 得分 2.0 胜出
    const penalized_tok = try lm_head.sampleNext(&h_last, &.{2}, 10.0, stream);
    try std.testing.expectEqual(@as(usize, 1), penalized_tok);

    // 2. 验证 Q4Weights 硬件量化矩阵乘法卸载
    var raw_block: [18]u8 = undefined;
    const scale_bits: u16 = 0x3c00; // f16 = 1.0
    std.mem.writeInt(u16, raw_block[0..2], scale_bits, .little);
    // 所有权重置为 (10 - 8) = 2.0 (低 4 位为 10 即 0xA，高 4 位为 10 即 0xA)
    @memset(raw_block[2..18], 0xAA);

    var q4_w = try Q4Weights.initFromBlocks(std.testing.allocator, &raw_block, 32, 1, stream);
    defer q4_w.deinit();

    var x_vec: [32]f32 = undefined;
    for (&x_vec) |*v| v.* = 1.0;
    var out_res: [1]f32 = undefined;

    const DummyTensor = struct {
        data: []f32,
        shape: struct {
            dims: [2]usize,
            len: usize = 2,
        },
    };
    var x_t = DummyTensor{ .data = &x_vec, .shape = .{ .dims = .{ 1, 32 }, .len = 2 } };
    var out_t = DummyTensor{ .data = &out_res, .shape = .{ .dims = .{ 1, 1 }, .len = 2 } };

    try q4MatMul(&x_t, &out_t, &q4_w, stream);
    stream.synchronize();

    // 32 个 1.0 与 2.0 的点积 = 64.0
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), out_res[0], 1e-2);

    // 3. 验证融合量化矩阵 (Fused Q4Weights: initFromBlockSlices)
    var raw_block_a: [18]u8 = undefined;
    std.mem.writeInt(u16, raw_block_a[0..2], scale_bits, .little);
    @memset(raw_block_a[2..18], 0xAA); // weight = 2.0

    var raw_block_b: [18]u8 = undefined;
    std.mem.writeInt(u16, raw_block_b[0..2], scale_bits, .little);
    @memset(raw_block_b[2..18], 0xBB); // weight = 11 - 8 = 3.0

    var fused_q4 = try Q4Weights.initFromBlockSlices(
        std.testing.allocator,
        &.{ &raw_block_a, &raw_block_b },
        32,
        &.{ 1, 1 },
        stream,
    );
    defer fused_q4.deinit();

    var in_arr = try mlx.Array.fromSlice(&x_vec, &.{ 1, 32 });
    defer in_arr.deinit();

    var in_bf16 = try in_arr.astype(.bfloat16, stream);
    defer in_bf16.deinit();

    var norm_bf16 = try mlx.Array.fastRmsNorm(in_bf16, null, 1e-6, stream);
    defer norm_bf16.deinit();
    norm_bf16.eval();
    stream.synchronize();

    var fused_bf16_out = try fused_q4.forwardArray(in_bf16, stream);
    defer fused_bf16_out.deinit();
    fused_bf16_out.eval();
    stream.synchronize();

    var fused_out = try fused_q4.forwardArray(in_arr, stream);
    defer fused_out.deinit();

    var shape_buf: [8]usize = undefined;
    const fused_nd = try fused_out.shape(&shape_buf);
    try std.testing.expectEqual(@as(usize, 2), fused_nd);
    try std.testing.expectEqual(@as(usize, 1), shape_buf[0]);
    try std.testing.expectEqual(@as(usize, 2), shape_buf[1]);

    // 切片验证: sliceAxis 产生子张量并支持后续算子
    var slice_a = try fused_out.sliceAxis(1, 0, 1, stream);
    defer slice_a.deinit();
    var slice_b = try fused_out.sliceAxis(1, 1, 2, stream);
    defer slice_b.deinit();

    // 验证 gelu 与 multiply
    var act = try slice_a.gelu(stream);
    defer act.deinit();
    var mul = try mlx.Array.multiply(act, slice_b, stream);
    defer mul.deinit();

    fused_out.eval();
    const f_data = fused_out.dataFloat32();
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), f_data[0], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 96.0), f_data[1], 1e-2);
}

test "LmHeadWeights argmaxWithPenalty dynamic vocab and negative logits" {
    const logits = [_]f32{ 1.0, 2.0, 6.0, -0.5 };
    // 无重复惩罚时，token 2 (6.0) 最大
    try std.testing.expectEqual(@as(usize, 2), LmHeadWeights.argmaxWithPenalty(&logits, 4, &.{}, 1.0));
    // 对 token 2 施加 10.0 惩罚 -> 6.0 / 10.0 = 0.6，此时 token 1 (2.0) 最大
    try std.testing.expectEqual(@as(usize, 1), LmHeadWeights.argmaxWithPenalty(&logits, 4, &.{2}, 10.0));
    // 对正数与负数同时施加惩罚，验证负数乘以 penalty 的分支与越界 token 过滤
    const neg_logits = [_]f32{ -2.0, -1.0, -0.5, -3.0 };
    try std.testing.expectEqual(@as(usize, 1), LmHeadWeights.argmaxWithPenalty(&neg_logits, 4, &.{ 2, 999999 }, 4.0));
}
