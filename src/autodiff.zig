const std = @import("std");
const c = @import("cblas.zig");
const tensor = @import("tensor.zig");

extern fn erff(x: f32) f32;

pub const Tensor = tensor.Tensor;
pub const Shape = tensor.Shape;
pub const computeContiguousStrides = tensor.computeContiguousStrides;
pub const transposeShape = tensor.transposeShape;


// 支持的算子类型枚举
pub const OpType = enum {
    // --- 基础数学与张量逐元素运算 (Basic Math & Element-wise Ops) ---
    Add,                 // 张量逐元素加法
    AddBias,             // 偏置项加法（广播机制）
    AddScalar,           // 标量加法
    Mul,                 // 张量逐元素乘法
    MulScalar,           // 标量乘法（张量缩放）
    MatMul,              // 矩阵乘法
    BatchMatMul,         // 批量矩阵乘法 (Batched Matrix Multiplication)

    // --- 形状与维度变换 (Shape & Dimension Transforms) ---
    Reshape,             // 形状变换
    Transpose,           // 维度转置

    // --- 激活函数与非线性变换 (Activation Functions) ---
    Relu,                // 激活函数 ReLU
    LeakyRelu,           // 激活函数 LeakyReLU
    Gelu,                // 激活函数 GELU
    Silu,                // 激活函数 SiLU / Swish
    Sigmoid,             // 激活函数 Sigmoid
    Tanh,                // 激活函数 Tanh
    Softmax,             // 独立 Softmax (Standalone Softmax)

    // --- 神经网络层与结构运算 (Neural Network Layers & Structural Ops) ---
    Conv2D,              // 二维卷积
    MaxPool2D,           // 二维最大池化
    RmsNorm,             // RMSNorm 归一化
    Embedding,           // 嵌入查找 (Embedding Lookup)

    // --- 损失函数与正则化 (Loss Functions & Regularization) ---
    MseLoss,             // 均方误差损失函数 (MSE Loss)
    BceLoss,             // 二元交叉熵损失函数 (BCE Loss)
    BceWithLogitsLoss,   // 二元交叉熵带 Logits 损失函数
    SigmoidCrossEntropy, // Sigmoid 交叉熵损失 (BCE with logits)
    SoftmaxCrossEntropy, // 结合 Softmax 与交叉熵损失（数值稳定性更好）
    L1Loss,              // L1 正则化 / Lasso Loss: lambda * sum(|w|)
    L2Loss,              // L2 正则化 / Ridge Loss: 0.5 * lambda * sum(w^2)
};

// 各算子反向传播所需的上下文信息（如 Softmax 的概率输出与 Target 类别）
pub const OpContext = union(enum) {
    // --- 基础数学与张量逐元素运算 ---
    Add: void,
    AddBias: void,
    AddScalar: struct {
        val: f32,
    },
    Mul: void,
    MulScalar: struct {
        val: f32,
    },
    MatMul: void,
    BatchMatMul: void,

    // --- 形状与维度变换 ---
    Reshape: void,
    Transpose: struct {
        dim0: usize,
        dim1: usize,
    },

    // --- 激活函数与非线性变换 ---
    Relu: void,
    LeakyRelu: struct {
        alpha: f32,
    },
    Gelu: void,
    Silu: void,
    Sigmoid: void,
    Tanh: void,
    Softmax: void,

    // --- 神经网络层与结构运算 ---
    Conv2D: void,
    MaxPool2D: struct {
        pool_size: usize,
        stride: usize,
    },
    RmsNorm: struct {
        eps: f32,
    },
    Embedding: void,

    // --- 损失函数与正则化 ---
    MseLoss: void,
    BceLoss: struct {
        eps: f32,
    },
    BceWithLogitsLoss: void,
    SigmoidCrossEntropy: void,
    SoftmaxCrossEntropy: struct {
        probs: []f32,
        targets: []const u8,
    },
    L1Loss: struct {
        lambda: f32,
    },
    L2Loss: struct {
        lambda: f32,
    },
};





// 计算图中的算子节点（Op）结构体
// 存储算子的操作类型、输入输出张量指针，并定义了如何对该操作执行求导（backward）
pub const Op = struct {
    op_type: OpType,        // 算子类别（如 MatMul, Relu）
    inputs: []*Tensor,      // 输入张量数组
    outputs: []*Tensor,     // 输出张量数组
    context: OpContext,     // 算子特有的运行时上下文数据

    // 重新执行该算子的前向计算，根据最新输入更新输出张量的数据
    pub fn forward(self: *Op, allocator: std.mem.Allocator) !void {
        _ = allocator;
        switch (self.op_type) {
            .MatMul => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];
                const M = A.shape.dims[0];
                const K = A.shape.dims[1];
                const N = B.shape.dims[1];
                c.cblas_sgemm(
                    c.CblasRowMajor,
                    c.CblasNoTrans,
                    c.CblasNoTrans,
                    @intCast(M),
                    @intCast(N),
                    @intCast(K),
                    1.0,
                    A.data.ptr,
                    @intCast(K),
                    B.data.ptr,
                    @intCast(N),
                    0.0,
                    C.data.ptr,
                    @intCast(N),
                );
            },
            .AddBias => {
                const A = self.inputs[0];
                const bias = self.inputs[1];
                const C = self.outputs[0];
                const M = A.shape.dims[0];
                const N = A.shape.dims[1];
                for (0..M) |i| {
                    const a_row = A.data[i * N .. (i + 1) * N];
                    const c_row = C.data[i * N .. (i + 1) * N];
                    for (0..N) |j| {
                        c_row[j] = a_row[j] + bias.data[j];
                    }
                }
            },
            .Relu => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                for (C.data, A.data) |*c_val, a_val| {
                    c_val.* = if (a_val > 0.0) a_val else 0.0;
                }
            },
            // 前向公式: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            .Gelu => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const sqrt_2 = @sqrt(@as(f32, 2.0));
                for (C.data, A.data) |*c_val, a_val| {
                    // 使用标准库中的误差函数 erff 进行计算
                    const erf_val = erff(a_val / sqrt_2);
                    c_val.* = 0.5 * a_val * (1.0 + erf_val);
                }
            },
            .Sigmoid => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                for (C.data, A.data) |*c_val, a_val| {
                    if (a_val >= 0.0) {
                        c_val.* = 1.0 / (1.0 + @exp(-a_val));
                    } else {
                        const e = @exp(a_val);
                        c_val.* = e / (1.0 + e);
                    }
                }
            },
            .Tanh => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                for (C.data, A.data) |*c_val, a_val| {
                    c_val.* = std.math.tanh(a_val);
                }
            },
            .LeakyRelu => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const alpha = self.context.LeakyRelu.alpha;
                for (C.data, A.data) |*c_val, a_val| {
                    c_val.* = if (a_val > 0.0) a_val else alpha * a_val;
                }
            },
            .Silu => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                for (C.data, A.data) |*c_val, a_val| {
                    const sig = if (a_val >= 0.0) 1.0 / (1.0 + @exp(-a_val)) else @exp(a_val) / (1.0 + @exp(a_val));
                    c_val.* = a_val * sig;
                }
            },
            // 前向公式:
            // 1. Softmax 归一化概率: p_j = e^{x_j - max} / sum(e^{x_i - max})
            // 2. 交叉熵损失计算: Loss = -1/B * sum( log(p_target) )
            .SoftmaxCrossEntropy => {
                const logits = self.inputs[0];
                const loss = self.outputs[0];
                const targets = self.context.SoftmaxCrossEntropy.targets;
                const probs = self.context.SoftmaxCrossEntropy.probs;
 
                const B_size = logits.shape.dims[0];
                const D = logits.shape.dims[1];
                var loss_sum: f32 = 0.0;
 
                for (0..B_size) |i| {
                    const row = logits.data[i * D .. (i + 1) * D];
                    
                    // 1) 寻找最大值 max_val 用于减法防止指数溢出 (Softmax 数值稳定性)
                    var max_val = row[0];
                    for (row) |val| {
                        if (val > max_val) max_val = val;
                    }
                    
                    // 2) 计算非归一化指数项 e^{x_j - max} 并求和
                    var sum: f32 = 0.0;
                    const row_probs = probs[i * D .. (i + 1) * D];
                    for (0..D) |j| {
                        const e = @exp(row[j] - max_val);
                        row_probs[j] = e;
                        sum += e;
                    }
                    
                    // 3) 归一化计算概率
                    for (0..D) |j| {
                        row_probs[j] /= sum;
                    }
                    
                    // 4) 提取 Target 类别的预测概率计算负对数似然损失
                    const target_idx = targets[i];
                    const prob = row_probs[target_idx];
                    const clipped = @max(prob, 1e-15); // 防止 log(0)
                    loss_sum += -@log(clipped);
                }
                
                // 5) 计算 Batch 范围内的均值 Loss
                loss.data[0] = loss_sum / @as(f32, @floatFromInt(B_size));
            },
            .BceWithLogitsLoss, .SigmoidCrossEntropy => {
                const logits = self.inputs[0];
                const targets = self.inputs[1];
                const loss = self.outputs[0];
                const N = logits.data.len;
                var loss_sum: f32 = 0.0;
                for (0..N) |i| {
                    const x = logits.data[i];
                    const y = targets.data[i];
                    const max_x = @max(x, 0.0);
                    const abs_x = @abs(x);
                    const l = max_x - x * y + @log(1.0 + @exp(-abs_x));
                    loss_sum += l;
                }
                loss.data[0] = loss_sum / @as(f32, @floatFromInt(N));
            },
            .BceLoss => {
                const probs = self.inputs[0];
                const targets = self.inputs[1];
                const loss = self.outputs[0];
                const eps = self.context.BceLoss.eps;
                const N = probs.data.len;
                var loss_sum: f32 = 0.0;
                for (0..N) |i| {
                    const p = probs.data[i];
                    const y = targets.data[i];
                    const p_clip = @max(p, eps);
                    const one_minus_p_clip = @max(1.0 - p, eps);
                    const l = -(y * @log(p_clip) + (1.0 - y) * @log(one_minus_p_clip));
                    loss_sum += l;
                }
                loss.data[0] = loss_sum / @as(f32, @floatFromInt(N));
            },
            .Reshape => {
                // A.data and C.data point to the same memory buffer (aliased).
                // Do not use @memcpy as it panics on overlapping identical memory.
            },
            .Transpose => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const ctx = self.context.Transpose;
                const strides_trans = transposeShape(A.strides, ctx.dim0, ctx.dim1);

                var indices = [_]usize{0} ** 8;
                const len = C.shape.len;
                const total_size = C.data.len;
                for (0..total_size) |dest_flat_idx| {
                    var src_flat_idx: usize = 0;
                    for (0..len) |d| {
                        src_flat_idx += indices[d] * strides_trans.dims[d];
                    }
                    C.data[dest_flat_idx] = A.data[src_flat_idx];

                    var d: usize = len;
                    while (d > 0) {
                        d -= 1;
                        indices[d] += 1;
                        if (indices[d] < C.shape.dims[d]) {
                            break;
                        }
                        indices[d] = 0;
                    }
                }
            },
            .MseLoss => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];
                const N = A.data.len;
                var loss_sum: f32 = 0.0;
                for (0..N) |i| {
                    const diff = A.data[i] - B.data[i];
                    loss_sum += diff * diff;
                }
                C.data[0] = loss_sum / @as(f32, @floatFromInt(N));
            },
            .MulScalar => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const val = self.context.MulScalar.val;
                for (C.data, A.data) |*c_val, a_val| {
                    c_val.* = a_val * val;
                }
            },
            .AddScalar => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const val = self.context.AddScalar.val;
                for (C.data, A.data) |*c_val, a_val| {
                    c_val.* = a_val + val;
                }
            },
            .Add => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];
                for (C.data, A.data, B.data) |*c_val, a_val, b_val| {
                    c_val.* = a_val + b_val;
                }
            },
            .Mul => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];
                for (C.data, A.data, B.data) |*c_val, a_val, b_val| {
                    c_val.* = a_val * b_val;
                }
            },
            .Conv2D => {
                const A = self.inputs[0];
                const W = self.inputs[1];
                const C = self.outputs[0];
                const bias = if (self.inputs.len > 2) self.inputs[2] else null;
                
                const N = A.shape.dims[0];
                const C_in = A.shape.dims[1];
                const C_out = W.shape.dims[0];
                const KH = W.shape.dims[2];
                const KW = W.shape.dims[3];
                const H_out = C.shape.dims[2];
                const W_out = C.shape.dims[3];

                const s_n = A.strides.dims[0];
                const s_c = A.strides.dims[1];
                const s_h = A.strides.dims[2];
                const s_w = A.strides.dims[3];

                const w_co = W.strides.dims[0];
                const w_ci = W.strides.dims[1];
                const w_kh = W.strides.dims[2];
                const w_kw = W.strides.dims[3];

                const o_n = C.strides.dims[0];
                const o_c = C.strides.dims[1];
                const o_h = C.strides.dims[2];
                const o_w = C.strides.dims[3];

                for (0..N) |n| {
                    for (0..C_out) |co| {
                        const b_val = if (bias) |b| b.data[co] else 0.0;
                        for (0..H_out) |h| {
                            for (0..W_out) |w| {
                                var sum: f32 = b_val;
                                for (0..C_in) |ci| {
                                    for (0..KH) |kh| {
                                        for (0..KW) |kw| {
                                            const input_val = A.data[n * s_n + ci * s_c + (h + kh) * s_h + (w + kw) * s_w];
                                            const weight_val = W.data[co * w_co + ci * w_ci + kh * w_kh + kw * w_kw];
                                            sum += input_val * weight_val;
                                        }
                                    }
                                }
                                C.data[n * o_n + co * o_c + h * o_h + w * o_w] = sum;
                            }
                        }
                    }
                }
            },
            .MaxPool2D => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const pool_size = self.context.MaxPool2D.pool_size;
                const stride = self.context.MaxPool2D.stride;
                
                const N = A.shape.dims[0];
                const C_ch = A.shape.dims[1];
                const H = A.shape.dims[2];
                const W = A.shape.dims[3];
                const H_out = C.shape.dims[2];
                const W_out = C.shape.dims[3];

                const s_n = A.strides.dims[0];
                const s_c = A.strides.dims[1];
                const s_h = A.strides.dims[2];
                const s_w = A.strides.dims[3];

                const o_n = C.strides.dims[0];
                const o_c = C.strides.dims[1];
                const o_h = C.strides.dims[2];
                const o_w = C.strides.dims[3];

                for (0..N) |n| {
                    for (0..C_ch) |c_| {
                        for (0..H_out) |h| {
                            for (0..W_out) |w| {
                                var max_val = A.data[n * s_n + c_ * s_c + (h * stride) * s_h + (w * stride) * s_w];
                                for (0..pool_size) |ph| {
                                    for (0..pool_size) |pw| {
                                        const ih = h * stride + ph;
                                        const iw = w * stride + pw;
                                        if (ih < H and iw < W) {
                                            const val = A.data[n * s_n + c_ * s_c + ih * s_h + iw * s_w];
                                            if (val > max_val) {
                                                max_val = val;
                                            }
                                        }
                                    }
                                }
                                C.data[n * o_n + c_ * o_c + h * o_h + w * o_w] = max_val;
                            }
                        }
                    }
                }
            },
            .Softmax => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const D = A.shape.dims[A.shape.len - 1];
                const M = A.data.len / D;

                for (0..M) |i| {
                    const row_in = A.data[i * D .. (i + 1) * D];
                    const row_out = C.data[i * D .. (i + 1) * D];

                    var max_val = row_in[0];
                    for (row_in[1..]) |val| {
                        if (val > max_val) max_val = val;
                    }

                    var sum: f32 = 0.0;
                    for (row_in, row_out) |val, *p| {
                        const exp_val = @exp(val - max_val);
                        p.* = exp_val;
                        sum += exp_val;
                    }

                    for (row_out) |*p| {
                        p.* /= sum;
                    }
                }
            },
            .RmsNorm => {
                const X = self.inputs[0];
                const G = self.inputs[1];
                const Y = self.outputs[0];
                const eps = self.context.RmsNorm.eps;
                const D = X.shape.dims[X.shape.len - 1];
                const M = X.data.len / D;

                for (0..M) |i| {
                    const row_in = X.data[i * D .. (i + 1) * D];
                    const row_out = Y.data[i * D .. (i + 1) * D];

                    var sum_x2: f32 = 0.0;
                    for (row_in) |val| {
                        sum_x2 += val * val;
                    }
                    const rms = @sqrt(sum_x2 / @as(f32, @floatFromInt(D)) + eps);

                    for (row_in, row_out, G.data) |x_val, *y_val, g_val| {
                        y_val.* = x_val / rms * g_val;
                    }
                }
            },
            .BatchMatMul => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];

                std.debug.assert(A.shape.len == 4);
                std.debug.assert(B.shape.len == 4);
                std.debug.assert(C.shape.len == 4);

                const batch_size = A.shape.dims[0];
                const num_heads = A.shape.dims[1];
                const M = A.shape.dims[2];
                const K = A.shape.dims[3];
                const N = B.shape.dims[3];

                const sA_b = A.strides.dims[0];
                const sA_h = A.strides.dims[1];
                const sB_b = B.strides.dims[0];
                const sB_h = B.strides.dims[1];
                const sC_b = C.strides.dims[0];
                const sC_h = C.strides.dims[1];

                for (0..batch_size) |b| {
                    for (0..num_heads) |h| {
                        const ptrA = A.data.ptr + b * sA_b + h * sA_h;
                        const ptrB = B.data.ptr + b * sB_b + h * sB_h;
                        const ptrC = C.data.ptr + b * sC_b + h * sC_h;

                        c.cblas_sgemm(
                            c.CblasRowMajor,
                            c.CblasNoTrans,
                            c.CblasNoTrans,
                            @intCast(M),
                            @intCast(N),
                            @intCast(K),
                            1.0,
                            ptrA,
                            @intCast(K),
                            ptrB,
                            @intCast(N),
                            0.0,
                            ptrC,
                            @intCast(N),
                        );
                    }
                }
            },
            .Embedding => {
                const W = self.inputs[0];
                const X = self.inputs[1];
                const Y = self.outputs[0];

                const B = X.shape.dims[0];
                const T = X.shape.dims[1];
                const D = W.shape.dims[1];
                const VocabSize = W.shape.dims[0];

                for (0..B) |b| {
                    for (0..T) |t| {
                        const idx_f = X.data[b * T + t];
                        const idx = @as(usize, @intFromFloat(idx_f));
                        std.debug.assert(idx < VocabSize);

                        const w_row = W.data[idx * D .. (idx + 1) * D];
                        const y_row = Y.data[(b * T + t) * D .. (b * T + t + 1) * D];
                        @memcpy(y_row, w_row);
                    }
                }
            },
            .L2Loss => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const lambda = self.context.L2Loss.lambda;
                var sum_sq: f32 = 0.0;
                for (A.data) |v| {
                    sum_sq += v * v;
                }
                C.data[0] = 0.5 * lambda * sum_sq;
            },
            .L1Loss => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const lambda = self.context.L1Loss.lambda;
                var sum_abs: f32 = 0.0;
                for (A.data) |v| {
                    sum_abs += @abs(v);
                }
                C.data[0] = lambda * sum_abs;
            },
        }
    }


    // 执行该算子的反向传播计算，更新其输入节点的梯度
    pub fn backward(self: *Op) !void {
        switch (self.op_type) {
            // ====================================================================
            // 1. 矩阵乘法反向传播 (MatMul Backward)
            // ====================================================================
            // 前向公式: C = A * B，其中 A (M x K), B (K x N), C (M x N)
            // 数学推导:
            // 设损失标量为 L，我们拥有对输出的梯度 dC = ∂L/∂C (M x N)。
            // 根据矩阵微积分链式法则：
            //   1. 对左输入 A 的导数: dA = ∂L/∂A = dC * B^T
            //      维度匹配: (M x N) * (N x K) -> (M x K)
            //   2. 对右输入 B 的导数: dB = ∂L/∂B = A^T * dC
            //      维度匹配: (K x M) * (M x N) -> (K x N)
            // 注意: 在深度学习中，梯度是累加的 (+=)，所以我们传入 beta = 1.0 给 cblas_sgemm。
            .MatMul => {
                const A = self.inputs[0]; // 形状为 M x K
                const B = self.inputs[1]; // 形状为 K x N
                const C = self.outputs[0]; // 形状为 M x N
                const M = A.shape.dims[0];
                const K = A.shape.dims[1];
                const N = B.shape.dims[1];

                // 1. 计算对左乘矩阵 A 的梯度: dA += dC * B^T
                if (A.requires_grad) {
                    // 使用 CPU Apple Accelerate (AMX) sgemm 矩阵乘法
                    c.cblas_sgemm(
                        c.CblasRowMajor,
                        c.CblasNoTrans, // C.grad 不转置
                        c.CblasTrans,   // B.data 需转置为 B^T
                        @intCast(M),
                        @intCast(K),
                        @intCast(N),
                        1.0,            // alpha = 1.0
                        C.grad.ptr,
                        @intCast(N),
                        B.data.ptr,
                        @intCast(N),
                        1.0,            // beta = 1.0 表示累加到 A.grad，不覆盖已有值
                        A.grad.ptr,
                        @intCast(K),
                    );
                }

                // 2. 计算对右乘矩阵 B 的梯度: dB += A^T * dC
                if (B.requires_grad) {
                    c.cblas_sgemm(
                        c.CblasRowMajor,
                        c.CblasTrans,   // A.data 需转置为 A^T
                        c.CblasNoTrans, // C.grad 不转置
                        @intCast(K),
                        @intCast(N),
                        @intCast(M),
                        1.0,            // alpha = 1.0
                        A.data.ptr,
                        @intCast(K),
                        C.grad.ptr,
                        @intCast(N),
                        1.0,            // beta = 1.0 同样进行累加
                        B.grad.ptr,
                        @intCast(N),
                    );
                }
            },
            // ====================================================================
            // 2. 偏置项加法反向传播 (AddBias Backward)
            // ====================================================================
            // 前向公式: C[i, j] = A[i, j] + bias[j]，其中 A (M x N), bias (1 x N), C (M x N)
            // 数学推导:
            //   1. 对输入 A 的偏导数: ∂L/∂A[i, j] = ∂L/∂C[i, j]
            //      因此 dA += dC (逐元素直接累加)。
            //   2. 对偏置 bias 的偏导数: ∂L/∂bias[j] = sum_{i=0}^{M-1} (∂L/∂C[i, j])
            //      这是因为偏置项在行维度 (批量样本维度) 进行了广播复制。
            //      因此对偏置的梯度为对输出梯度 dC 在第 0 维（行）上的降维累加和。
            .AddBias => {
                const A = self.inputs[0];
                const bias = self.inputs[1];
                const C = self.outputs[0];
                const M = A.shape.dims[0];
                const N = A.shape.dims[1];

                for (0..N) |n| {
                    var bias_sum: f32 = 0.0;
                    for (0..M) |m| {
                        const grad_val = C.grad[m * N + n];
                        if (A.requires_grad) {
                            A.grad[m * N + n] += grad_val;
                        }
                        bias_sum += grad_val;
                    }
                    if (bias.requires_grad) {
                        bias.grad[n] += bias_sum;
                    }
                }
            },
            // ====================================================================
            // 3. ReLU 激活函数反向传播 (ReLU Backward)
            // ====================================================================
            // 前向公式: C = max(0, A)，逐元素操作
            // 数学推导:
            //   对于每个元素：
            //   若 A[i] > 0，则该点斜率为 1.0 -> dA[i] += dC[i]
            //   若 A[i] <= 0，则该点斜率为 0.0 -> dA[i] += 0.0
            .Relu => {
                const A = self.inputs[0];
                const C = self.outputs[0];

                if (A.requires_grad) {
                    const total = A.data.len;
                    for (0..total) |i| {
                        A.grad[i] += if (A.data[i] > 0.0) C.grad[i] else 0.0;
                    }
                }
            },
            // ====================================================================
            // 3.5. GELU 激活函数反向传播 (GELU Backward)
            // ====================================================================
            // 前向公式: C = 0.5 * A * (1 + erf(A / sqrt(2)))
            // 数学推导:
            //   dC/dA = 0.5 * (1 + erf(A / sqrt(2))) + A * (1 / sqrt(2 * pi)) * e^{-A^2 / 2}
            //   dA[i] += dC[i] * (dC/dA)
            .Gelu => {
                const A = self.inputs[0];
                const C = self.outputs[0];

                if (A.requires_grad) {
                    const total = A.data.len;
                    const sqrt_2 = @sqrt(@as(f32, 2.0));
                    const inv_sqrt_2pi = 1.0 / @sqrt(@as(f32, 2.0 * std.math.pi));
                    for (0..total) |i| {
                        const x = A.data[i];
                        const erf_val = erff(x / sqrt_2);
                        const cdf = 0.5 * (1.0 + erf_val);
                        const pdf = inv_sqrt_2pi * @exp(-0.5 * x * x);
                        const deriv = cdf + x * pdf;
                        A.grad[i] += C.grad[i] * deriv;
                    }
                }
            },
            .Sigmoid => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    for (A.grad, C.grad, C.data) |*a_g, c_g, c_val| {
                        a_g.* += c_g * c_val * (1.0 - c_val);
                    }
                }
            },
            .Tanh => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    for (A.grad, C.grad, C.data) |*a_g, c_g, c_val| {
                        a_g.* += c_g * (1.0 - c_val * c_val);
                    }
                }
            },
            .LeakyRelu => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const alpha = self.context.LeakyRelu.alpha;
                if (A.requires_grad) {
                    for (A.grad, C.grad, A.data) |*a_g, c_g, a_val| {
                        const slope: f32 = if (a_val > 0.0) 1.0 else alpha;
                        a_g.* += c_g * slope;
                    }
                }
            },
            .Silu => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    for (A.grad, C.grad, A.data) |*a_g, c_g, a_val| {
                        const sig = if (a_val >= 0.0) 1.0 / (1.0 + @exp(-a_val)) else @exp(a_val) / (1.0 + @exp(a_val));
                        const d_silu = sig * (1.0 + a_val * (1.0 - sig));
                        a_g.* += c_g * d_silu;
                    }
                }
            },
            // ====================================================================
            // 4. Softmax + Cross Entropy 损失函数反向传播 (SoftmaxCrossEntropy Backward)
            // ====================================================================
            // 前向公式:
            //   设输入的 Logits 矩阵为 X (M x N)，真实分类为 label (M x 1)。
            //   对于第 i 行样本，先算 Softmax 概率：probs[i, j] = e^{X[i, j]} / sum_k(e^{X[i, k]})
            //   再算平均交叉熵损失：L = -1/M * sum_i( ln(probs[i, label_i]) )
            // 数学推导:
            //   将 Softmax 和 CrossEntropy 结合后，对输入 Logits X[i, j] 的偏导数具有极佳的数值稳定性：
            //     ∂L/∂X[i, j] = (probs[i, j] - Indicator(j == label_i)) / M
            //   其中 Indicator 在当前类别 j 等于真实类别 label_i 时为 1.0，否则为 0.0。
            //   最后除以样本数 M 得到平均样本梯度。
            .SoftmaxCrossEntropy => {
                const logits = self.inputs[0];
                const M = logits.shape.dims[0];
                const N = logits.shape.dims[1];
                const ctx = &self.context.SoftmaxCrossEntropy;

                const scale = 1.0 / @as(f32, @floatFromInt(M));

                for (0..M) |i| {
                    const label = ctx.targets[i];
                    const p_row = ctx.probs[i * N .. (i + 1) * N];
                    const dLogits_row = logits.grad[i * N .. (i + 1) * N];
                    for (0..N) |j| {
                        dLogits_row[j] += scale * (p_row[j] - (if (j == label) @as(f32, 1.0) else 0.0));
                    }
                }
            },
            .BceWithLogitsLoss, .SigmoidCrossEntropy => {
                const logits = self.inputs[0];
                const targets = self.inputs[1];
                const loss = self.outputs[0];
                const dL = loss.grad[0];
                const N = logits.data.len;
                const scale = dL / @as(f32, @floatFromInt(N));

                if (logits.requires_grad) {
                    for (logits.grad, logits.data, targets.data) |*x_g, x_val, y_val| {
                        const sig: f32 = if (x_val >= 0.0)
                            1.0 / (1.0 + @exp(-x_val))
                        else
                            @exp(x_val) / (1.0 + @exp(x_val));
                        x_g.* += scale * (sig - y_val);
                    }
                }
                if (targets.requires_grad) {
                    for (targets.grad, logits.data) |*y_g, x_val| {
                        y_g.* += scale * (-x_val);
                    }
                }
            },
            .BceLoss => {
                const probs = self.inputs[0];
                const targets = self.inputs[1];
                const loss = self.outputs[0];
                const eps = self.context.BceLoss.eps;
                const dL = loss.grad[0];
                const N = probs.data.len;
                const scale = dL / @as(f32, @floatFromInt(N));

                if (probs.requires_grad) {
                    for (probs.grad, probs.data, targets.data) |*p_g, p_val, y_val| {
                        const p_clip = @max(p_val, eps);
                        const one_minus_p_clip = @max(1.0 - p_val, eps);
                        const grad_p = (p_val - y_val) / (p_clip * one_minus_p_clip);
                        p_g.* += scale * grad_p;
                    }
                }
                if (targets.requires_grad) {
                    for (targets.grad, probs.data, targets.data) |*y_g, p_val, _| {
                        const p_clip = @max(p_val, eps);
                        const one_minus_p_clip = @max(1.0 - p_val, eps);
                        const grad_y = -@log(p_clip) + @log(one_minus_p_clip);
                        y_g.* += scale * grad_y;
                    }
                }
            },
            // ====================================================================
            // 5. 形状变换反向传播 (Reshape Backward)
            // ====================================================================
            // 前向公式: C = reshape(A)，仅改变形状元数据，不修改物理排列
            // 数学推导:
            //   Reshape 没有数学上的参数变换，因此其梯度传递就是将输出梯度 dC
            //   以一维平铺形式直接拷回/累加到输入 A 的梯度 dA 缓冲区中。
            .Reshape => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    for (C.grad, 0..) |g, i| {
                        A.grad[i] += g;
                    }
                }
            },
            // ====================================================================
            // 6. 维度转置反向传播 (Transpose Backward)
            // ====================================================================
            // 前向公式: C = transpose(A, dim0, dim1)
            // 数学推导:
            //   转置算子物理上改变了元素的读取索引。
            //   因此反向传播时，必须通过转置后的 stride 步长定位到 A.grad 中的物理偏移位置，
            //   并将 C.grad 中连续排列的梯度累加进去。
            .Transpose => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    const ctx = self.context.Transpose;
                    const strides_trans = transposeShape(A.strides, ctx.dim0, ctx.dim1);

                    var indices = [_]usize{0} ** 8;
                    const len = C.shape.len;
                    const total_size = C.data.len;
                    for (0..total_size) |dest_flat_idx| {
                        var src_flat_idx: usize = 0;
                        for (0..len) |d| {
                            src_flat_idx += indices[d] * strides_trans.dims[d];
                        }
                        A.grad[src_flat_idx] += C.grad[dest_flat_idx];

                        var d: usize = len;
                        while (d > 0) {
                            d -= 1;
                            indices[d] += 1;
                            if (indices[d] < C.shape.dims[d]) {
                                break;
                            }
                            indices[d] = 0;
                        }
                    }
                }
            },
            .MseLoss => {
                const A = self.inputs[0]; // y_pred
                const B = self.inputs[1]; // y_true
                const C = self.outputs[0]; // loss
                const N = A.data.len;
                const N_f = @as(f32, @floatFromInt(N));

                if (A.requires_grad) {
                    for (0..N) |i| {
                        A.grad[i] += C.grad[0] * (2.0 / N_f) * (A.data[i] - B.data[i]);
                    }
                }
                if (B.requires_grad) {
                    for (0..N) |i| {
                        B.grad[i] += C.grad[0] * (2.0 / N_f) * (B.data[i] - A.data[i]);
                    }
                }
            },
            .MulScalar => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const val = self.context.MulScalar.val;
                if (A.requires_grad) {
                    for (0..A.data.len) |i| {
                        A.grad[i] += C.grad[i] * val;
                    }
                }
            },
            .AddScalar => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    for (0..A.data.len) |i| {
                        A.grad[i] += C.grad[i];
                    }
                }
            },
            .Add => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    for (0..A.data.len) |i| {
                        A.grad[i] += C.grad[i];
                    }
                }
                if (B.requires_grad) {
                    for (0..B.data.len) |i| {
                        B.grad[i] += C.grad[i];
                    }
                }
            },
            .Mul => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    for (0..A.data.len) |i| {
                        A.grad[i] += C.grad[i] * B.data[i];
                    }
                }
                if (B.requires_grad) {
                    for (0..B.data.len) |i| {
                        B.grad[i] += C.grad[i] * A.data[i];
                    }
                }
            },
            .Conv2D => {
                const A = self.inputs[0];
                const W = self.inputs[1];
                const C = self.outputs[0];
                const N = A.shape.dims[0];
                const C_in = A.shape.dims[1];
                const C_out = W.shape.dims[0];
                const KH = W.shape.dims[2];
                const KW = W.shape.dims[3];
                const H_out = C.shape.dims[2];
                const W_out = C.shape.dims[3];

                const s_n = A.strides.dims[0];
                const s_c = A.strides.dims[1];
                const s_h = A.strides.dims[2];
                const s_w = A.strides.dims[3];

                const w_co = W.strides.dims[0];
                const w_ci = W.strides.dims[1];
                const w_kh = W.strides.dims[2];
                const w_kw = W.strides.dims[3];

                const o_n = C.strides.dims[0];
                const o_c = C.strides.dims[1];
                const o_h = C.strides.dims[2];
                const o_w = C.strides.dims[3];

                for (0..N) |n| {
                    for (0..C_out) |co| {
                        for (0..H_out) |h| {
                            for (0..W_out) |w| {
                                const grad_val = C.grad[n * o_n + co * o_c + h * o_h + w * o_w];
                                if (grad_val == 0.0) continue;

                                // Bias gradient
                                if (self.inputs.len > 2) {
                                    const bias = self.inputs[2];
                                    if (bias.requires_grad) {
                                        bias.grad[co] += grad_val;
                                    }
                                }

                                for (0..C_in) |ci| {
                                    for (0..KH) |kh| {
                                        for (0..KW) |kw| {
                                            const ih = h + kh;
                                            const iw = w + kw;

                                            // Weight gradient: dW += dC * A
                                            if (W.requires_grad) {
                                                const input_val = A.data[n * s_n + ci * s_c + ih * s_h + iw * s_w];
                                                W.grad[co * w_co + ci * w_ci + kh * w_kh + kw * w_kw] += grad_val * input_val;
                                            }

                                            // Input gradient: dA += dC * W
                                            if (A.requires_grad) {
                                                const weight_val = W.data[co * w_co + ci * w_ci + kh * w_kh + kw * w_kw];
                                                A.grad[n * s_n + ci * s_c + ih * s_h + iw * s_w] += grad_val * weight_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            .MaxPool2D => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const N = A.shape.dims[0];
                const C_ch = A.shape.dims[1];
                const H = A.shape.dims[2];
                const W = A.shape.dims[3];
                const H_out = C.shape.dims[2];
                const W_out = C.shape.dims[3];

                const pool_size = self.context.MaxPool2D.pool_size;
                const stride = self.context.MaxPool2D.stride;

                const s_n = A.strides.dims[0];
                const s_c = A.strides.dims[1];
                const s_h = A.strides.dims[2];
                const s_w = A.strides.dims[3];

                const o_n = C.strides.dims[0];
                const o_c = C.strides.dims[1];
                const o_h = C.strides.dims[2];
                const o_w = C.strides.dims[3];

                if (A.requires_grad) {
                    for (0..N) |n| {
                        for (0..C_ch) |c_| {
                            for (0..H_out) |h| {
                                for (0..W_out) |w| {
                                    const grad_val = C.grad[n * o_n + c_ * o_c + h * o_h + w * o_w];
                                    if (grad_val == 0.0) continue;

                                    // Find where the max was
                                    var max_val = A.data[n * s_n + c_ * s_c + (h * stride) * s_h + (w * stride) * s_w];
                                    var max_h = h * stride;
                                    var max_w = w * stride;

                                    for (0..pool_size) |ph| {
                                        for (0..pool_size) |pw| {
                                            const ih = h * stride + ph;
                                            const iw = w * stride + pw;
                                            if (ih < H and iw < W) {
                                                const val = A.data[n * s_n + c_ * s_c + ih * s_h + iw * s_w];
                                                if (val > max_val) {
                                                    max_val = val;
                                                    max_h = ih;
                                                    max_w = iw;
                                                }
                                            }
                                        }
                                    }
                                    // Route gradient to max_h, max_w
                                    A.grad[n * s_n + c_ * s_c + max_h * s_h + max_w * s_w] += grad_val;
                                }
                            }
                        }
                    }
                }
            },
            .Softmax => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const D = A.shape.dims[A.shape.len - 1];
                const M = A.data.len / D;

                if (A.requires_grad) {
                    for (0..M) |i| {
                        const row_out = C.data[i * D .. (i + 1) * D];
                        const row_grad_out = C.grad[i * D .. (i + 1) * D];
                        const row_grad_in = A.grad[i * D .. (i + 1) * D];

                        var sum_dy_y: f32 = 0.0;
                        for (row_grad_out, row_out) |dy, y| {
                            sum_dy_y += dy * y;
                        }

                        for (row_grad_in, row_out, row_grad_out) |*da, y, dy| {
                            da.* += y * (dy - sum_dy_y);
                        }
                    }
                }
            },
            .RmsNorm => {
                const X = self.inputs[0];
                const G = self.inputs[1];
                const Y = self.outputs[0];
                const eps = self.context.RmsNorm.eps;
                const D = X.shape.dims[X.shape.len - 1];
                const M = X.data.len / D;

                for (0..M) |i| {
                    const row_in = X.data[i * D .. (i + 1) * D];
                    const row_grad_out = Y.grad[i * D .. (i + 1) * D];

                    var sum_x2: f32 = 0.0;
                    for (row_in) |val| {
                        sum_x2 += val * val;
                    }
                    const scale = 1.0 / @sqrt(sum_x2 / @as(f32, @floatFromInt(D)) + eps);

                    if (G.requires_grad) {
                        for (0..D) |j| {
                            G.grad[j] += row_grad_out[j] * row_in[j] * scale;
                        }
                    }

                    if (X.requires_grad) {
                        const row_grad_in = X.grad[i * D .. (i + 1) * D];

                        var sum_dy_g_x: f32 = 0.0;
                        for (0..D) |j| {
                            sum_dy_g_x += row_grad_out[j] * G.data[j] * row_in[j];
                        }

                        for (0..D) |j| {
                            const term1 = G.data[j] * row_grad_out[j];
                            const term2 = row_in[j] * scale * scale * sum_dy_g_x / @as(f32, @floatFromInt(D));
                            row_grad_in[j] += scale * (term1 - term2);
                        }
                    }
                }
            },
            .BatchMatMul => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];

                const batch_size = A.shape.dims[0];
                const num_heads = A.shape.dims[1];
                const M = A.shape.dims[2];
                const K = A.shape.dims[3];
                const N = B.shape.dims[3];

                const sA_b = A.strides.dims[0];
                const sA_h = A.strides.dims[1];
                const sB_b = B.strides.dims[0];
                const sB_h = B.strides.dims[1];
                const sC_b = C.strides.dims[0];
                const sC_h = C.strides.dims[1];

                for (0..batch_size) |b| {
                    for (0..num_heads) |h| {
                        const ptrA = A.data.ptr + b * sA_b + h * sA_h;
                        const ptrB = B.data.ptr + b * sB_b + h * sB_h;
                        const ptrdC = C.grad.ptr + b * sC_b + h * sC_h;

                        if (A.requires_grad) {
                            const ptrdA = A.grad.ptr + b * sA_b + h * sA_h;
                            c.cblas_sgemm(
                                c.CblasRowMajor,
                                c.CblasNoTrans,
                                c.CblasTrans,
                                @intCast(M),
                                @intCast(K),
                                @intCast(N),
                                1.0,
                                ptrdC,
                                @intCast(N),
                                ptrB,
                                @intCast(N),
                                1.0,
                                ptrdA,
                                @intCast(K),
                            );
                        }

                        if (B.requires_grad) {
                            const ptrdB = B.grad.ptr + b * sB_b + h * sB_h;
                            c.cblas_sgemm(
                                c.CblasRowMajor,
                                c.CblasTrans,
                                c.CblasNoTrans,
                                @intCast(K),
                                @intCast(N),
                                @intCast(M),
                                1.0,
                                ptrA,
                                @intCast(K),
                                ptrdC,
                                @intCast(N),
                                1.0,
                                ptrdB,
                                @intCast(N),
                            );
                        }
                    }
                }
            },
            .Embedding => {
                const W = self.inputs[0];
                const X = self.inputs[1];
                const Y = self.outputs[0];

                const B = X.shape.dims[0];
                const T = X.shape.dims[1];
                const D = W.shape.dims[1];

                if (W.requires_grad) {
                    for (0..B) |b| {
                        for (0..T) |t| {
                            const idx_f = X.data[b * T + t];
                            const idx = @as(usize, @intFromFloat(idx_f));

                            const w_grad_row = W.grad[idx * D .. (idx + 1) * D];
                            const y_grad_row = Y.grad[(b * T + t) * D .. (b * T + t + 1) * D];

                            for (w_grad_row, y_grad_row) |*wg, yg| {
                                wg.* += yg;
                            }
                        }
                    }
                }
            },
            .L2Loss => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const lambda = self.context.L2Loss.lambda;
                if (A.requires_grad) {
                    for (0..A.data.len) |i| {
                        A.grad[i] += C.grad[0] * lambda * A.data[i];
                    }
                }
            },
            .L1Loss => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const lambda = self.context.L1Loss.lambda;
                if (A.requires_grad) {
                    for (0..A.data.len) |i| {
                        const val = A.data[i];
                        const sign: f32 = if (val > 0.0) 1.0 else if (val < 0.0) -1.0 else 0.0;
                        A.grad[i] += C.grad[0] * lambda * sign;
                    }
                }
            },
        }
    }

};

// 计算图（Graph）结构体
// 追踪所有的张量节点与算子节点，管理内存生命周期并负责反向传播调度
pub const Graph = struct {
    backing_allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,       // 使用 Arena 机制，使每次前向/反向生成的中间节点内存可在 batch 结束时一并释放，避免内存碎片和频繁分配
    tensors: std.ArrayList(*Tensor),     // 追踪计算图中的所有张量指针
    ops: std.ArrayList(*Op),             // 追踪计算图中的所有算子指针
    enable_grad: bool,                   // 梯度使能开关（类似 torch.set_grad_enabled），为 false 时不分配梯度缓冲区亦不记录 Op 节点

    // 初始化计算图，传入底层通用内存分配器
    pub fn init(backing_allocator: std.mem.Allocator) Graph {
        return Graph{
            .backing_allocator = backing_allocator,
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .tensors = .empty,
            .ops = .empty,
            .enable_grad = true,
        };
    }

    // 设置梯度追踪开关
    pub fn setGradEnabled(self: *Graph, enabled: bool) void {
        self.enable_grad = enabled;
    }

    // 释放整个计算图的内存（包括所有张量与算子节点的前向/反向缓冲区）
    pub fn deinit(self: *Graph) void {
        self.tensors.deinit(self.backing_allocator);
        self.ops.deinit(self.backing_allocator);
        self.arena.deinit();
    }

    // 在计算图中创建并注册一个新的张量节点
    pub fn tensor(self: *Graph, rows: usize, cols: usize, requires_grad: bool) !*Tensor {
        return self.tensorND(&.{rows, cols}, requires_grad);
    }

    // 创建并注册一个带初始数据的二维张量节点
    pub fn tensorWithData(self: *Graph, rows: usize, cols: usize, initial_data: []const f32, requires_grad: bool) !*Tensor {
        return self.tensorNDWithData(&.{rows, cols}, initial_data, requires_grad);
    }

    // 创建并注册一个带初始数据的多维张量节点
    pub fn tensorNDWithData(self: *Graph, shape_slice: []const usize, initial_data: []const f32, requires_grad: bool) !*Tensor {
        const t = try self.tensorND(shape_slice, requires_grad);
        std.debug.assert(t.data.len == initial_data.len);
        @memcpy(t.data, initial_data);
        return t;
    }

    // NumPy-like API: zeros
    pub fn zeros(self: *Graph, shape_slice: []const usize, requires_grad: bool) !*Tensor {
        return self.tensorND(shape_slice, requires_grad);
    }

    // NumPy-like API: ones
    pub fn ones(self: *Graph, shape_slice: []const usize, requires_grad: bool) !*Tensor {
        const t = try self.tensorND(shape_slice, requires_grad);
        @memset(t.data, 1.0);
        return t;
    }

    // NumPy-like API: array
    pub fn array(self: *Graph, shape_slice: []const usize, initial_data: []const f32, requires_grad: bool) !*Tensor {
        return self.tensorNDWithData(shape_slice, initial_data, requires_grad);
    }

    // NumPy-like API: transpose alias
    pub fn transpose(self: *Graph, A: *Tensor, dim0: usize, dim1: usize) !*Tensor {
        return self.transposeND(A, dim0, dim1);
    }

    // 在计算图中创建并注册一个新的 N 维张量节点
    pub fn tensorND(self: *Graph, shape_slice: []const usize, requires_grad: bool) !*Tensor {
        const allocator = self.arena.allocator();
        const t = try allocator.create(Tensor);
        const shape = Shape.init(shape_slice);
        const strides = computeContiguousStrides(shape);

        var total_size: usize = 1;
        for (shape_slice) |dim| {
            total_size *= dim;
        }

        const effective_req_grad = self.enable_grad and requires_grad;

        t.* = Tensor{
            .data = try allocator.alloc(f32, total_size),
            .grad = if (effective_req_grad) try allocator.alloc(f32, total_size) else &.{},
            .shape = shape,
            .strides = strides,
            .requires_grad = effective_req_grad,
            .creator = null,
        };
        @memset(t.data, 0.0);
        if (effective_req_grad) {
            @memset(t.grad, 0.0);
        }
        try self.tensors.append(self.backing_allocator, t);
        return t;
    }

    // 形状变换算子前向传播
    pub fn reshape(self: *Graph, A: *Tensor, new_shape_slice: []const usize) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try allocator.create(Tensor);
        const shape = Shape.init(new_shape_slice);
        const strides = computeContiguousStrides(shape);

        var old_total: usize = 1;
        for (0..A.shape.len) |i| {
            old_total *= A.shape.dims[i];
        }
        var new_total: usize = 1;
        for (new_shape_slice) |dim| {
            new_total *= dim;
        }
        std.debug.assert(old_total == new_total);

        const req_grad = self.enable_grad and A.requires_grad;

        C.* = Tensor{
            .data = A.data, // 共享前向数据
            .grad = if (req_grad) try allocator.alloc(f32, new_total) else &.{},
            .shape = shape,
            .strides = strides,
            .requires_grad = req_grad,
            .creator = null,
        };
        if (req_grad) {
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Reshape,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Reshape = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 维度转置算子前向传播：交换 dim0 和 dim1
    pub fn transposeND(self: *Graph, A: *Tensor, dim0: usize, dim1: usize) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.transpose(dim0, dim1, allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Transpose,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{
                    .Transpose = .{
                        .dim0 = dim0,
                        .dim1 = dim1,
                    },
                },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 矩阵乘法算子前向传播：C = A * B
    pub fn matmul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.matmul(B, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = A;
            inputs[1] = B;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .MatMul,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .MatMul = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 偏置相加算子前向传播：C = A + bias (按行广播相加)
    pub fn addBias(self: *Graph, A: *Tensor, bias: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.addBias(bias, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or bias.requires_grad);
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = A;
            inputs[1] = bias;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .AddBias,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .AddBias = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 激活函数 ReLU 前向传播：C = max(0, A)
    pub fn relu(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.relu(allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Relu,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Relu = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn gelu(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.gelu(allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Gelu,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Gelu = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn sigmoid(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.sigmoid(allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Sigmoid,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Sigmoid = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn tanh(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.tanh(allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Tanh,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Tanh = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn leakyRelu(self: *Graph, A: *Tensor, alpha: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.leakyRelu(alpha, allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .LeakyRelu,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .LeakyRelu = .{ .alpha = alpha } },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 激活函数 SiLU (Swish) 前向传播：C = A * sigmoid(A)
    pub fn silu(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.silu(allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Silu,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Silu = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 损失函数 Softmax + Cross Entropy 结合前向传播
    // 在 logits 的行维度计算 Softmax 概率分布，并与 targets 分类标签计算交叉熵损失
    pub fn softmaxCrossEntropy(self: *Graph, logits: *Tensor, targets: []const u8) !*Tensor {
        const req_grad = self.enable_grad and logits.requires_grad;
        const loss = try self.tensor(1, 1, req_grad);

        const B = logits.shape.dims[0];
        const N = logits.shape.dims[1];
        const allocator = self.arena.allocator();

        var loss_sum: f32 = 0.0;
        if (req_grad) {
            const probs = try allocator.alloc(f32, B * N);

            // 1. 对每一行计算 Softmax 概率（数值稳定的减去 max 技巧）
            for (0..B) |i| {
                const logits_row = logits.data[i * N .. (i + 1) * N];
                const probs_row = probs[i * N .. (i + 1) * N];

                // 寻找当前行的最大值，避免 @exp() 产生数值上溢（NaN）
                var max_val = logits_row[0];
                for (logits_row[1..]) |val| {
                    if (val > max_val) max_val = val;
                }

                var sum: f32 = 0.0;
                for (logits_row, probs_row) |val, *p| {
                    const exp_val = @exp(val - max_val);
                    p.* = exp_val;
                    sum += exp_val;
                }

                // 归一化为概率分布
                for (probs_row) |*p| {
                    p.* /= sum;
                }
            }

            // 2. 计算平均交叉熵损失值：L = -1/B * sum(log(prob_target))
            for (0..B) |i| {
                const label = targets[i];
                const prob = probs[i * N + label];
                const clipped = @max(prob, 1e-15); // 微小值剪裁，避免 log(0) 产生 -inf
                loss_sum += -@log(clipped);
            }
            loss.data[0] = loss_sum / @as(f32, @floatFromInt(B));

            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = logits;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = loss;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .SoftmaxCrossEntropy,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{
                    .SoftmaxCrossEntropy = .{
                        .probs = probs,
                        .targets = targets,
                    },
                },
            };
            loss.creator = o;
            try self.ops.append(self.backing_allocator, o);
        } else {
            for (0..B) |i| {
                const logits_row = logits.data[i * N .. (i + 1) * N];
                var max_val = logits_row[0];
                for (logits_row[1..]) |val| {
                    if (val > max_val) max_val = val;
                }
                var sum: f32 = 0.0;
                for (logits_row) |val| {
                    sum += @exp(val - max_val);
                }
                const label = targets[i];
                const prob = @exp(logits_row[label] - max_val) / sum;
                const clipped = @max(prob, 1e-15);
                loss_sum += -@log(clipped);
            }
            loss.data[0] = loss_sum / @as(f32, @floatFromInt(B));
        }

        return loss;
    }

    // 均方误差 (MSE) 损失函数：C = 1/N * sum((y_pred - y_true)^2)
    pub fn mseLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and (y_pred.requires_grad or y_true.requires_grad);
        const loss = try self.tensor(1, 1, req_grad);

        const N = y_pred.data.len;
        std.debug.assert(N == y_true.data.len);

        var loss_sum: f32 = 0.0;
        for (0..N) |i| {
            const diff = y_pred.data[i] - y_true.data[i];
            loss_sum += diff * diff;
        }
        loss.data[0] = loss_sum / @as(f32, @floatFromInt(N));

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = y_pred;
            inputs[1] = y_true;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = loss;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .MseLoss,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .MseLoss = {} },
            };
            loss.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return loss;
    }

    pub fn bceWithLogitsLoss(self: *Graph, logits: *Tensor, targets: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and (logits.requires_grad or targets.requires_grad);
        const loss = try self.tensor(1, 1, req_grad);

        const N = logits.data.len;
        std.debug.assert(N == targets.data.len);

        var loss_sum: f32 = 0.0;
        for (0..N) |i| {
            const x = logits.data[i];
            const y = targets.data[i];
            const max_x = @max(x, 0.0);
            const abs_x = @abs(x);
            const l = max_x - x * y + @log(1.0 + @exp(-abs_x));
            loss_sum += l;
        }
        loss.data[0] = loss_sum / @as(f32, @floatFromInt(N));

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = logits;
            inputs[1] = targets;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = loss;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .BceWithLogitsLoss,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .BceWithLogitsLoss = {} },
            };
            loss.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return loss;
    }

    pub fn sigmoidCrossEntropy(self: *Graph, logits: *Tensor, targets: *Tensor) !*Tensor {
        return self.bceWithLogitsLoss(logits, targets);
    }

    pub fn bceLoss(self: *Graph, probs: *Tensor, targets: *Tensor, eps: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and (probs.requires_grad or targets.requires_grad);
        const loss = try self.tensor(1, 1, req_grad);

        const N = probs.data.len;
        std.debug.assert(N == targets.data.len);

        var loss_sum: f32 = 0.0;
        for (0..N) |i| {
            const p = probs.data[i];
            const y = targets.data[i];
            const p_clip = @max(p, eps);
            const one_minus_p_clip = @max(1.0 - p, eps);
            const l = -(y * @log(p_clip) + (1.0 - y) * @log(one_minus_p_clip));
            loss_sum += l;
        }
        loss.data[0] = loss_sum / @as(f32, @floatFromInt(N));

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = probs;
            inputs[1] = targets;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = loss;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .BceLoss,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .BceLoss = .{ .eps = eps } },
            };
            loss.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return loss;
    }

    pub fn randomNormal(self: *Graph, shape_slice: []const usize, random: std.Random, mean: f32, stddev: f32, requires_grad: bool) !*Tensor {
        const t = try self.tensorND(shape_slice, requires_grad);
        t.fillNormal(random, mean, stddev);
        return t;
    }

    pub fn randomUniform(self: *Graph, shape_slice: []const usize, random: std.Random, min: f32, max: f32, requires_grad: bool) !*Tensor {
        const t = try self.tensorND(shape_slice, requires_grad);
        t.fillUniform(random, min, max);
        return t;
    }

    // L2 正则化损失函数：C = 0.5 * lambda * sum(weight_i^2)
    pub fn l2Loss(self: *Graph, weight: *Tensor, lambda: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and weight.requires_grad;
        const loss = try self.tensor(1, 1, req_grad);

        var sum_sq: f32 = 0.0;
        for (weight.data) |v| {
            sum_sq += v * v;
        }
        loss.data[0] = 0.5 * lambda * sum_sq;

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = weight;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = loss;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .L2Loss,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .L2Loss = .{ .lambda = lambda } },
            };
            loss.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return loss;
    }

    // 岭回归 (Ridge) 组合损失函数：Loss = MSE(y_pred, y_true) + 0.5 * lambda * sum(weight_i^2)
    pub fn ridgeLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor, weight: *Tensor, lambda: f32) !*Tensor {
        const mse = try self.mseLoss(y_pred, y_true);
        if (lambda == 0.0) return mse;
        const l2 = try self.l2Loss(weight, lambda);
        return try self.add(mse, l2);
    }

    // L1 正则化损失：Loss = lambda * sum(|weight_i|)
    pub fn l1Loss(self: *Graph, weight: *Tensor, lambda: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and weight.requires_grad;
        const loss = try self.tensor(1, 1, req_grad);

        var sum_abs: f32 = 0.0;
        for (weight.data) |v| {
            sum_abs += @abs(v);
        }
        loss.data[0] = lambda * sum_abs;

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = weight;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = loss;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .L1Loss,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .L1Loss = .{ .lambda = lambda } },
            };
            loss.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return loss;
    }

    // Lasso 组合损失函数：Loss = MSE(y_pred, y_true) + lambda * sum(|weight_i|)
    pub fn lassoLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor, weight: *Tensor, lambda: f32) !*Tensor {
        const mse = try self.mseLoss(y_pred, y_true);
        if (lambda == 0.0) return mse;
        const l1 = try self.l1Loss(weight, lambda);
        return try self.add(mse, l1);
    }

    // Elastic Net 组合损失函数：Loss = MSE(y_pred, y_true) + lambda * rho * ||w||_1 + 0.5 * lambda * (1 - rho) * ||w||_2^2
    pub fn elasticNetLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor, weight: *Tensor, lambda: f32, l1_ratio: f32) !*Tensor {
        const mse = try self.mseLoss(y_pred, y_true);
        if (lambda == 0.0) return mse;
        var total_loss = mse;
        if (l1_ratio > 0.0) {
            const l1 = try self.l1Loss(weight, lambda * l1_ratio);
            total_loss = try self.add(total_loss, l1);
        }
        if (l1_ratio < 1.0) {
            const l2 = try self.l2Loss(weight, lambda * (1.0 - l1_ratio));
            total_loss = try self.add(total_loss, l2);
        }
        return total_loss;
    }

    // 标量乘法（缩放）：C = val * A
    pub fn mulScalar(self: *Graph, A: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.mulScalar(val, allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .MulScalar,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .MulScalar = .{ .val = val } },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 标量加法：C = A + val
    pub fn addScalar(self: *Graph, A: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.addScalar(val, allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .AddScalar,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .AddScalar = .{ .val = val } },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 逐元素张量加法：C = A + B
    pub fn add(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.add(B, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = A;
            inputs[1] = B;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Add,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Add = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 逐元素张量乘法 (Hadamard 积)：C = A * B
    pub fn mul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.mul(B, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = A;
            inputs[1] = B;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Mul,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Mul = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn conv2d(self: *Graph, A: *Tensor, weight: *Tensor, bias: ?*Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.conv2d(weight, bias, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or weight.requires_grad or (bias != null and bias.?.requires_grad));
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const num_inputs: usize = if (bias != null) 3 else 2;
            const inputs = try allocator.alloc(*Tensor, num_inputs);
            inputs[0] = A;
            inputs[1] = weight;
            if (bias) |b| {
                inputs[2] = b;
            }
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Conv2D,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Conv2D = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn maxpool2d(self: *Graph, A: *Tensor, pool_size: usize, stride: usize) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.maxpool2d(pool_size, stride, allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .MaxPool2D,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .MaxPool2D = .{ .pool_size = pool_size, .stride = stride } },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn softmax(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.softmax(allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Softmax,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Softmax = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn rmsNorm(self: *Graph, X: *Tensor, G: *Tensor, eps: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const Y = try X.rmsNorm(G, eps, allocator, null);

        const req_grad = self.enable_grad and (X.requires_grad or G.requires_grad);
        Y.requires_grad = req_grad;
        if (req_grad) {
            Y.grad = try allocator.alloc(f32, Y.data.len);
            @memset(Y.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, Y);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = X;
            inputs[1] = G;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = Y;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .RmsNorm,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .RmsNorm = .{ .eps = eps } },
            };
            Y.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return Y;
    }

    pub fn batchMatMul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.batchMatMul(B, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = A;
            inputs[1] = B;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .BatchMatMul,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .BatchMatMul = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn embedding(self: *Graph, W: *Tensor, X: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const Y = try W.embedding(X, allocator, null);

        const req_grad = self.enable_grad and W.requires_grad;
        Y.requires_grad = req_grad;
        if (req_grad) {
            Y.grad = try allocator.alloc(f32, Y.data.len);
            @memset(Y.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, Y);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = W;
            inputs[1] = X;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = Y;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Embedding,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Embedding = {} },
            };
            Y.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return Y;
    }

    // 执行计算图的反向传播
    pub fn backward(self: *Graph, loss_tensor: *Tensor) !void {
        const allocator = self.arena.allocator();
        var visited = std.AutoHashMap(*Tensor, void).init(allocator);
        defer visited.deinit();
        var sorted_list: std.ArrayList(*Tensor) = .empty;
        defer sorted_list.deinit(allocator);

        // 1. 对计算图进行拓扑排序，以确保节点按正确的计算依赖关系进行链式求导
        try self.topologicalSort(loss_tensor, &visited, &sorted_list);

        // 2. 损失函数节点本身的偏导数设为 1.0 (dL/dL = 1.0)
        loss_tensor.grad[0] = 1.0;

        // 3. 按拓扑排序的逆序执行各算子的 backward 求导函数，由深至浅传导梯度
        var i = sorted_list.items.len;
        while (i > 0) {
            i -= 1;
            const node = sorted_list.items[i];
            if (node.creator) |op| {
                try op.backward();
            }
        }
    }

    // 执行计算图的反向传播，起点张量的梯度已被外部手动预填（常用于自定义损失函数如 MSE）
    pub fn backwardWithGrad(self: *Graph, output_tensor: *Tensor) !void {
        const allocator = self.arena.allocator();
        var visited = std.AutoHashMap(*Tensor, void).init(allocator);
        defer visited.deinit();
        var sorted_list: std.ArrayList(*Tensor) = .empty;
        defer sorted_list.deinit(allocator);

        // 1. 对计算图进行拓扑排序
        try self.topologicalSort(output_tensor, &visited, &sorted_list);

        // 2. 按拓扑排序的逆序执行各算子的 backward 求导，由深至浅传导梯度
        var i = sorted_list.items.len;
        while (i > 0) {
            i -= 1;
            const node = sorted_list.items[i];
            if (node.creator) |op| {
                try op.backward();
            }
        }
    }

    // 拓扑排序辅助函数（深度优先搜索 DFS 实现）
    fn topologicalSort(self: *Graph, node: *Tensor, visited: *std.AutoHashMap(*Tensor, void), list: *std.ArrayList(*Tensor)) !void {
        if (visited.contains(node)) return;
        try visited.put(node, {});

        if (node.creator) |op| {
            for (op.inputs) |input| {
                try self.topologicalSort(input, visited, list);
            }
        }
        try list.append(self.arena.allocator(), node);
    }

    // 运行计算图的前向传播，根据输入更新所有算子节点的值
    pub fn forward(self: *Graph) !void {
        for (self.ops.items) |op| {
            try op.forward(self.backing_allocator);
        }
    }

    // 将计算图中所有注册张量的梯度清零
    pub fn zeroGrad(self: *Graph) void {
        for (self.tensors.items) |t| {
            t.zeroGrad();
        }
    }
};

test "autodiff Graph enable_grad and setGradEnabled" {
    const allocator = std.testing.allocator;

    var graph = Graph.init(allocator);
    defer graph.deinit();

    // 默认 enable_grad = true
    try std.testing.expect(graph.enable_grad);

    const x = try graph.tensorWithData(2, 2, &.{ 1.0, 2.0, 3.0, 4.0 }, false);
    const w = try graph.tensorWithData(2, 2, &.{ 0.5, -0.5, 1.0, 2.0 }, true);

    const y = try graph.matmul(x, w);
    try std.testing.expect(y.requires_grad);
    try std.testing.expectEqual(@as(usize, 4), y.grad.len);
    try std.testing.expect(y.creator != null);
    try std.testing.expectEqual(@as(usize, 1), graph.ops.items.len);

    // 关闭梯度：setGradEnabled(false)
    var graph_nograd = Graph.init(allocator);
    defer graph_nograd.deinit();
    graph_nograd.setGradEnabled(false);
    try std.testing.expect(!graph_nograd.enable_grad);

    const x2 = try graph_nograd.tensorWithData(2, 2, &.{ 1.0, 2.0, 3.0, 4.0 }, false);
    const w2 = try graph_nograd.tensorWithData(2, 2, &.{ 0.5, -0.5, 1.0, 2.0 }, true);
    // 即使 w2 传入了 requires_grad = true，在 graph_nograd 下也应被强制置为 false
    try std.testing.expect(!w2.requires_grad);
    try std.testing.expectEqual(@as(usize, 0), w2.grad.len);

    const y2 = try graph_nograd.matmul(x2, w2);
    // 验证前向数值正确
    try std.testing.expectApproxEqAbs(y.data[0], y2.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(y.data[1], y2.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(y.data[2], y2.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(y.data[3], y2.data[3], 1e-5);

    // 验证无任何梯度内存与 Op 记录
    try std.testing.expect(!y2.requires_grad);
    try std.testing.expectEqual(@as(usize, 0), y2.grad.len);
    try std.testing.expect(y2.creator == null);
    try std.testing.expectEqual(@as(usize, 0), graph_nograd.ops.items.len);
}

