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

/// 语言模型输出头权重与采样加速器 (Language Model Head Accelerator, LM Head Accelerator)
/// 预先将巨型词表嵌入矩阵 (如 Gemma 4 的 262,144 维词表) 在 GPU 显存中转置常驻，
/// 自回归解码每一步仅需下发单步输入向量 [1, hidden_size]，利用统一内存实现 5~10 ms 极速词表打分与采样。
pub const LmHeadWeights = struct {
    w_t: mlx.Array,
    vocab_size: usize,
    hidden_size: usize,

    /// 从脑浮点 16 (Brain Floating Point 16, BF16) 词表权重构建常驻 GPU 词表矩阵
    pub fn init(
        embed_u16: []const u16,
        vocab_size: usize,
        hidden_size: usize,
        stream: ?Stream,
    ) OffloadError!LmHeadWeights {
        if (!isAvailable()) return error.OffloadNotSupported;
        var w = try mlx.Array.fromSliceBf16(embed_u16, &.{ vocab_size, hidden_size });
        defer w.deinit();

        const w_t = try w.transpose(stream);
        w_t.eval();

        return .{
            .w_t = w_t,
            .vocab_size = vocab_size,
            .hidden_size = hidden_size,
        };
    }

    /// 释放 GPU 常驻权重资源
    pub fn deinit(self: *LmHeadWeights) void {
        self.w_t.deinit();
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

        // 1. 创建单行隐状态向量 [1, hidden_size]
        var x_vec = try mlx.Array.fromSlice(h_last, &.{ 1, self.hidden_size });
        defer x_vec.deinit();

        // 2. GPU 矩阵乘法：[1, hidden_size] @ [hidden_size, vocab_size] -> [1, vocab_size]
        var logits = try mlx.Array.matmul(x_vec, self.w_t, stream);
        defer logits.deinit();
        logits.eval();

        const logits_slice = logits.dataFloat32();
        if (logits_slice.len != self.vocab_size) return error.ShapeMismatch;

        // 3. 在统一内存中读取未受惩罚的最大分值与索引，并应用重复惩罚 (Repetition Penalty)
        var max_val: f32 = -1e30;
        var max_id: usize = 0;

        for (0..self.vocab_size) |v| {
            // 屏蔽特殊保留词元（Gemma 4 约定：0 号 padding，258880..258884 为控制词元）
            if (v == 0 or (v >= 258880 and v <= 258884)) continue;

            var dot = logits_slice[v];
            if (repetition_penalty > 1.0) {
                for (recent_tokens) |rec| {
                    if (rec == v) {
                        if (dot > 0.0) {
                            dot /= repetition_penalty;
                        } else {
                            dot *= repetition_penalty;
                        }
                        break;
                    }
                }
            }

            if (dot > max_val) {
                max_val = dot;
                max_id = v;
            }
        }

        return max_id;
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

    /// 从原始 Q4Block 切片 (18 字节/块) 构建 GPU 硬件加速量化权重矩阵
    pub fn initFromBlocks(
        allocator: std.mem.Allocator,
        blocks_bytes: []const u8,
        in_features: usize,
        out_features: usize,
        stream: ?Stream,
    ) OffloadError!Q4Weights {
        if (!isAvailable()) return error.OffloadNotSupported;
        const blocks_per_row = in_features / 32;
        const total_blocks = out_features * blocks_per_row;
        if (blocks_bytes.len < total_blocks * 18) return error.ShapeMismatch;

        const w_q_buf = try allocator.alloc(u32, total_blocks * 4);
        defer allocator.free(w_q_buf);
        const scales_buf = try allocator.alloc(f16, total_blocks);
        defer allocator.free(scales_buf);
        const biases_buf = try allocator.alloc(f16, total_blocks);
        defer allocator.free(biases_buf);

        for (0..total_blocks) |blk_idx| {
            const raw = blocks_bytes[blk_idx * 18 .. (blk_idx + 1) * 18];
            const scale_bits = std.mem.readInt(u16, raw[0..2][0..2], .little);
            const scale_f16: f16 = @bitCast(scale_bits);
            const scale_f32: f32 = @floatCast(scale_f16);
            const bias_f16: f16 = @floatCast(-8.0 * scale_f32);

            scales_buf[blk_idx] = scale_f16;
            biases_buf[blk_idx] = bias_f16;

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
            w_q_buf[blk_idx * 4 + 0] = w0;
            w_q_buf[blk_idx * 4 + 1] = w1;
            w_q_buf[blk_idx * 4 + 2] = w2;
            w_q_buf[blk_idx * 4 + 3] = w3;
        }

        const w_shape = [_]c_int{ @intCast(out_features), @intCast(in_features / 8) };
        const sb_shape = [_]c_int{ @intCast(out_features), @intCast(in_features / 32) };

        const w_arr = mlx.c.mlx_array_new_data(@ptrCast(w_q_buf.ptr), &w_shape, 2, @intFromEnum(mlx.DType.uint32));
        var w_q = mlx.Array{ .handle = w_arr };
        const s_arr = mlx.c.mlx_array_new_data(@ptrCast(scales_buf.ptr), &sb_shape, 2, @intFromEnum(mlx.DType.float16));
        var scales = mlx.Array{ .handle = s_arr };
        const b_arr = mlx.c.mlx_array_new_data(@ptrCast(biases_buf.ptr), &sb_shape, 2, @intFromEnum(mlx.DType.float16));
        var biases = mlx.Array{ .handle = b_arr };

        w_q.eval();
        scales.eval();
        biases.eval();
        if (stream) |s| s.synchronize();

        return .{
            .w_q = w_q,
            .scales = scales,
            .biases = biases,
            .in_features = in_features,
            .out_features = out_features,
        };
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
}
