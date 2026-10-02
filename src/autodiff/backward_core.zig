const std = @import("std");
const c = @import("../cblas.zig");
const tensor = @import("../tensor.zig");
const transposeShape = tensor.transposeShape;
const op_mod = @import("op.zig");
const Op = op_mod.Op;

extern fn erff(x: f32) f32;

pub fn backwardCore(self: *Op) !void {
    switch (self.op_type) {
        // ====================================================================
        // 1. 矩阵乘法反向传播 (MatMul Backward)
        // ====================================================================
        .MatMul => {
            const A = self.inputs[0]; // 形状为 M x K
            const B = self.inputs[1]; // 形状为 K x N
            const C = self.outputs[0]; // 形状为 M x N
            const M = A.shape.dims[0];
            const K = A.shape.dims[1];
            const N = B.shape.dims[1];

            // 1. 计算对左乘矩阵 A 的梯度: dA += dC * B^T
            if (A.requires_grad) {
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
        // 2. 修正线性单元 (Rectified Linear Unit, ReLU) 激活函数反向传播 (ReLU Backward)
        // ====================================================================
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
        // 3.5. 高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 激活函数反向传播 (GELU Backward)
        // ====================================================================
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
        .SoftmaxCrossEntropy => {
            const logits = self.inputs[0];
            const loss = self.outputs[0];
            const dL = loss.grad[0];
            const M = logits.shape.dims[0];
            const N = logits.shape.dims[1];
            const ctx = &self.context.SoftmaxCrossEntropy;

            if (logits.requires_grad) {
                const denom = if (ctx.mask != null) ctx.total_weight else @as(f32, @floatFromInt(M));
                if (denom > 0.0) {
                    for (0..M) |i| {
                        const w: f32 = if (ctx.mask) |m| m[i] else 1.0;
                        if (w <= 0.0) continue;
                        const scale = dL * (w / denom);
                        const label = ctx.targets[i];
                        const p_row = ctx.probs[i * N .. (i + 1) * N];
                        const dLogits_row = logits.grad[i * N .. (i + 1) * N];
                        for (0..N) |j| {
                            dLogits_row[j] += scale * (p_row[j] - (if (j == label) @as(f32, 1.0) else 0.0));
                        }
                    }
                }
            }
        },
        .DpoLoss => {
            const pi_chosen = self.inputs[0];
            const pi_rejected = self.inputs[1];
            const loss = self.outputs[0];
            const dL = loss.grad[0];
            const ctx = self.context.DpoLoss;
            const N = pi_chosen.data.len;
            if (N > 0) {
                const inv_n = dL / @as(f32, @floatFromInt(N));
                for (0..N) |i| {
                    const log_ratio_chosen = pi_chosen.data[i] - ctx.ref_chosen[i];
                    const log_ratio_rejected = pi_rejected.data[i] - ctx.ref_rejected[i];
                    const z = ctx.beta * (log_ratio_chosen - log_ratio_rejected);
                    // d/dz (-log(sigmoid(z))) = -sigmoid(-z) = -1 / (1 + exp(z))
                    const sig_neg_z: f32 = if (z >= 0.0)
                        @exp(-z) / (1.0 + @exp(-z))
                    else
                        1.0 / (1.0 + @exp(z));
                    const dz = -inv_n * ctx.beta * sig_neg_z;
                    if (pi_chosen.requires_grad) {
                        pi_chosen.grad[i] += dz;
                    }
                    if (pi_rejected.requires_grad) {
                        pi_rejected.grad[i] -= dz;
                    }
                }
            }
        },
        .GrpoLoss => {
            const old_logps = self.inputs[0];
            const new_logps = self.inputs[1];
            const loss = self.outputs[0];
            const dL = loss.grad[0];
            const ctx = self.context.GrpoLoss;
            const N = old_logps.data.len;
            if (N > 0 and new_logps.requires_grad) {
                const inv_n = dL / @as(f32, @floatFromInt(N));
                for (0..N) |i| {
                    const ratio = @exp(new_logps.data[i] - old_logps.data[i]);
                    const adv = ctx.advantages[i];
                    var d_surrogate: f32 = 0.0;
                    if (adv >= 0.0) {
                        if (ratio <= 1.0 + ctx.clip_eps) {
                            d_surrogate = ratio * adv;
                        }
                    } else {
                        if (ratio >= 1.0 - ctx.clip_eps) {
                            d_surrogate = ratio * adv;
                        }
                    }

                    var d_kl: f32 = 0.0;
                    if (ctx.beta > 0.0) {
                        const ref = if (ctx.ref_logps) |refs| refs[i] else old_logps.data[i];
                        d_kl = 1.0 - @exp(ref - new_logps.data[i]);
                    }

                    new_logps.grad[i] += -inv_n * (d_surrogate - ctx.beta * d_kl);
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
        .Reshape => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                if (A.isContiguous() and A.grad.len >= C.grad.len) {
                    for (C.grad, 0..) |g, i| {
                        A.grad[i] += g;
                    }
                } else {
                    var coord = [_]usize{0} ** 8;
                    const len = A.shape.len;
                    for (C.grad) |g| {
                        var src_idx: usize = 0;
                        for (0..len) |d| {
                            src_idx += coord[d] * A.strides.dims[d];
                        }
                        A.grad[src_idx] += g;

                        var d = len;
                        while (d > 0) {
                            d -= 1;
                            coord[d] += 1;
                            if (coord[d] < A.shape.dims[d]) break;
                            coord[d] = 0;
                        }
                    }
                }
            }
        },
        // ====================================================================
        // 6. 维度转置反向传播 (Transpose Backward)
        // ====================================================================
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
        .Concat => {
            const dim = self.context.Concat.dim;
            const out = self.outputs[0];
            const rank = out.shape.len;
            const concat_dim_total = out.shape.dims[dim];

            var outer_size: usize = 1;
            for (0..dim) |d| {
                outer_size *= out.shape.dims[d];
            }
            var inner_size: usize = 1;
            for (dim + 1..rank) |d| {
                inner_size *= out.shape.dims[d];
            }

            for (0..outer_size) |outer| {
                const out_base = outer * concat_dim_total * inner_size;
                var offset_dim: usize = 0;
                for (self.inputs) |t| {
                    const d_k = t.shape.dims[dim];
                    const src_base = outer * d_k * inner_size;
                    const dest_base = out_base + offset_dim * inner_size;
                    const copy_len = d_k * inner_size;

                    if (t.requires_grad) {
                        for (0..copy_len) |j| {
                            t.grad[src_base + j] += out.grad[dest_base + j];
                        }
                    }
                    offset_dim += d_k;
                }
            }
        },
        .Split => {
            const dim = self.context.Split.dim;
            const in = self.inputs[0];
            const rank = in.shape.len;
            const dim_size = in.shape.dims[dim];
            const num_splits = self.outputs.len;
            const split_dim_size = self.outputs[0].shape.dims[dim];

            if (in.requires_grad) {
                var outer_size: usize = 1;
                for (0..dim) |d| {
                    outer_size *= in.shape.dims[d];
                }
                var inner_size: usize = 1;
                for (dim + 1..rank) |d| {
                    inner_size *= in.shape.dims[d];
                }

                for (0..outer_size) |outer| {
                    const src_base = outer * dim_size * inner_size;
                    for (0..num_splits) |k| {
                        const dest_base = outer * split_dim_size * inner_size;
                        const src_offset = src_base + k * split_dim_size * inner_size;
                        const copy_len = split_dim_size * inner_size;
                        for (0..copy_len) |j| {
                            in.grad[src_offset + j] += self.outputs[k].grad[dest_base + j];
                        }
                    }
                }
            }
        },
        .RepeatKV => {
            const groups = self.context.RepeatKV.groups;
            const X = self.inputs[0];
            const Y = self.outputs[0];
            if (X.requires_grad) {
                const B = X.shape.dims[0];
                const n_kv = X.shape.dims[1];
                const T = X.shape.dims[2];
                const hs = X.shape.dims[3];
                const head_bytes = T * hs;

                for (0..B) |b| {
                    for (0..n_kv) |kv_h| {
                        const x_grad = X.grad[((b * n_kv + kv_h) * head_bytes) .. ((b * n_kv + kv_h + 1) * head_bytes)];
                        for (0..groups) |g| {
                            const h = kv_h * groups + g;
                            const y_grad = Y.grad[((b * (n_kv * groups) + h) * head_bytes) .. ((b * (n_kv * groups) + h + 1) * head_bytes)];
                            for (x_grad, y_grad) |*xg, yg| {
                                xg.* += yg;
                            }
                        }
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
        .DivScalar => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const val = self.context.DivScalar.val;
            if (A.requires_grad) {
                for (0..A.data.len) |i| {
                    A.grad[i] += C.grad[i] / val;
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
        .SubScalar => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                for (0..A.data.len) |i| {
                    A.grad[i] += C.grad[i];
                }
            }
        },
        .Add, .AddBias => {
            const A = self.inputs[0];
            const B = self.inputs[1];
            const C = self.outputs[0];

            if (A.shape.eq(B.shape) and A.isContiguous() and B.isContiguous() and
                (!A.requires_grad or A.grad.len >= C.grad.len) and
                (!B.requires_grad or B.grad.len >= C.grad.len))
            {
                if (A.requires_grad) {
                    for (A.grad[0..C.grad.len], C.grad) |*a_g, c_g| {
                        a_g.* += c_g;
                    }
                }
                if (B.requires_grad) {
                    for (B.grad[0..C.grad.len], C.grad) |*b_g, c_g| {
                        b_g.* += c_g;
                    }
                }
            } else {
                const a_strides = tensor.computeBroadcastStrides(A.shape, A.strides, C.shape);
                const b_strides = tensor.computeBroadcastStrides(B.shape, B.strides, C.shape);
                const len = C.shape.len;
                var coord = [_]usize{0} ** 8;

                for (C.grad) |c_g| {
                    var a_idx: usize = 0;
                    var b_idx: usize = 0;
                    for (0..len) |d| {
                        a_idx += coord[d] * a_strides.dims[d];
                        b_idx += coord[d] * b_strides.dims[d];
                    }

                    if (A.requires_grad) {
                        A.grad[a_idx] += c_g;
                    }
                    if (B.requires_grad) {
                        B.grad[b_idx] += c_g;
                    }

                    var d = len;
                    while (d > 0) {
                        d -= 1;
                        coord[d] += 1;
                        if (coord[d] < C.shape.dims[d]) {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
            }
        },
        .Sub => {
            const A = self.inputs[0];
            const B = self.inputs[1];
            const C = self.outputs[0];

            if (A.shape.eq(B.shape) and A.isContiguous() and B.isContiguous() and
                (!A.requires_grad or A.grad.len >= C.grad.len) and
                (!B.requires_grad or B.grad.len >= C.grad.len))
            {
                if (A.requires_grad) {
                    for (A.grad[0..C.grad.len], C.grad) |*a_g, c_g| {
                        a_g.* += c_g;
                    }
                }
                if (B.requires_grad) {
                    for (B.grad[0..C.grad.len], C.grad) |*b_g, c_g| {
                        b_g.* -= c_g;
                    }
                }
            } else {
                const a_strides = tensor.computeBroadcastStrides(A.shape, A.strides, C.shape);
                const b_strides = tensor.computeBroadcastStrides(B.shape, B.strides, C.shape);
                const len = C.shape.len;
                var coord = [_]usize{0} ** 8;

                for (C.grad) |c_g| {
                    var a_idx: usize = 0;
                    var b_idx: usize = 0;
                    for (0..len) |d| {
                        a_idx += coord[d] * a_strides.dims[d];
                        b_idx += coord[d] * b_strides.dims[d];
                    }

                    if (A.requires_grad) {
                        A.grad[a_idx] += c_g;
                    }
                    if (B.requires_grad) {
                        B.grad[b_idx] -= c_g;
                    }

                    var d = len;
                    while (d > 0) {
                        d -= 1;
                        coord[d] += 1;
                        if (coord[d] < C.shape.dims[d]) {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
            }
        },
        .Mul => {
            const A = self.inputs[0];
            const B = self.inputs[1];
            const C = self.outputs[0];

            if (A.shape.eq(B.shape) and A.isContiguous() and B.isContiguous() and
                A.data.len >= C.grad.len and B.data.len >= C.grad.len and
                (!A.requires_grad or A.grad.len >= C.grad.len) and
                (!B.requires_grad or B.grad.len >= C.grad.len))
            {
                if (A.requires_grad) {
                    for (A.grad[0..C.grad.len], C.grad, B.data[0..C.grad.len]) |*a_g, c_g, b_val| {
                        a_g.* += c_g * b_val;
                    }
                }
                if (B.requires_grad) {
                    for (B.grad[0..C.grad.len], C.grad, A.data[0..C.grad.len]) |*b_g, c_g, a_val| {
                        b_g.* += c_g * a_val;
                    }
                }
            } else {
                const a_strides = tensor.computeBroadcastStrides(A.shape, A.strides, C.shape);
                const b_strides = tensor.computeBroadcastStrides(B.shape, B.strides, C.shape);
                const len = C.shape.len;
                var coord = [_]usize{0} ** 8;

                for (C.grad) |c_g| {
                    var a_idx: usize = 0;
                    var b_idx: usize = 0;
                    for (0..len) |d| {
                        a_idx += coord[d] * a_strides.dims[d];
                        b_idx += coord[d] * b_strides.dims[d];
                    }

                    if (A.requires_grad) {
                        A.grad[a_idx] += c_g * B.data[b_idx];
                    }
                    if (B.requires_grad) {
                        B.grad[b_idx] += c_g * A.data[a_idx];
                    }

                    var d = len;
                    while (d > 0) {
                        d -= 1;
                        coord[d] += 1;
                        if (coord[d] < C.shape.dims[d]) {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
            }
        },
        .Div => {
            const A = self.inputs[0];
            const B = self.inputs[1];
            const C = self.outputs[0];

            if (A.shape.eq(B.shape) and A.isContiguous() and B.isContiguous() and
                A.data.len >= C.grad.len and B.data.len >= C.grad.len and
                (!A.requires_grad or A.grad.len >= C.grad.len) and
                (!B.requires_grad or B.grad.len >= C.grad.len))
            {
                if (A.requires_grad) {
                    for (A.grad[0..C.grad.len], C.grad, B.data[0..C.grad.len]) |*a_g, c_g, b_val| {
                        a_g.* += c_g / b_val;
                    }
                }
                if (B.requires_grad) {
                    for (B.grad[0..C.grad.len], C.grad, A.data[0..C.grad.len], B.data[0..C.grad.len]) |*b_g, c_g, a_val, b_val| {
                        b_g.* -= c_g * a_val / (b_val * b_val);
                    }
                }
            } else {
                const a_strides = tensor.computeBroadcastStrides(A.shape, A.strides, C.shape);
                const b_strides = tensor.computeBroadcastStrides(B.shape, B.strides, C.shape);
                const len = C.shape.len;
                var coord = [_]usize{0} ** 8;

                for (C.grad) |c_g| {
                    var a_idx: usize = 0;
                    var b_idx: usize = 0;
                    for (0..len) |d| {
                        a_idx += coord[d] * a_strides.dims[d];
                        b_idx += coord[d] * b_strides.dims[d];
                    }

                    const b_val = B.data[b_idx];
                    if (A.requires_grad) {
                        A.grad[a_idx] += c_g / b_val;
                    }
                    if (B.requires_grad) {
                        const a_val = A.data[a_idx];
                        B.grad[b_idx] -= c_g * a_val / (b_val * b_val);
                    }

                    var d = len;
                    while (d > 0) {
                        d -= 1;
                        coord[d] += 1;
                        if (coord[d] < C.shape.dims[d]) {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
            }
        },
        else => unreachable,
    }
}
