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

    stream.synchronize();
}
