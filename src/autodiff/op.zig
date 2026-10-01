const std = @import("std");
const c = @import("../cblas.zig");
const tensor_mod = @import("../tensor.zig");
pub const tensor = tensor_mod;
const Tensor = tensor_mod.Tensor;
const Shape = tensor_mod.Shape;
const transposeShape = tensor_mod.transposeShape;
const types = @import("types.zig");
pub const OpType = types.OpType;
pub const OpContext = types.OpContext;

extern fn erff(x: f32) f32;

// 计算图中的算子节点（Op）结构体
// 存储算子的操作类型、输入输出张量指针，并定义了如何对该操作执行求导（backward）
pub const Op = struct {
    op_type: OpType,        // 算子类别（如 MatMul, Relu）
    inputs: []*Tensor,      // 输入张量数组
    outputs: []*Tensor,     // 输出张量数组
    context: OpContext,     // 算子特有的运行时上下文数据
    scope: []const u8 = "", // 创建该算子时所处的模块作用域完整路径（由 Graph 作用域栈记录，"" 表示图的根作用域）

    fn copyFromEager(dest: *Tensor, tmp: *Tensor, allocator: std.mem.Allocator) void {
        defer tmp.deinit(allocator);
        @memcpy(dest.data, tmp.data);
    }

    // 重新执行该算子的前向计算，根据最新输入更新输出张量的数据
    pub fn forward(self: *Op, allocator: std.mem.Allocator) !void {
        switch (self.op_type) {
            .MatMul => {
                copyFromEager(self.outputs[0], try self.inputs[0].matmul(self.inputs[1], allocator), allocator);
            },
            .Relu => {
                copyFromEager(self.outputs[0], try self.inputs[0].relu(allocator), allocator);
            },
            .Gelu => {
                copyFromEager(self.outputs[0], try self.inputs[0].gelu(allocator), allocator);
            },
            .Sigmoid => {
                copyFromEager(self.outputs[0], try self.inputs[0].sigmoid(allocator), allocator);
            },
            .Tanh => {
                copyFromEager(self.outputs[0], try self.inputs[0].tanh(allocator), allocator);
            },
            .LeakyRelu => {
                copyFromEager(self.outputs[0], try self.inputs[0].leakyRelu(self.context.LeakyRelu.alpha, allocator), allocator);
            },
            .Silu => {
                copyFromEager(self.outputs[0], try self.inputs[0].silu(allocator), allocator);
            },
            .SoftmaxCrossEntropy => {
                const logits = self.inputs[0];
                const loss = self.outputs[0];
                const ctx = &self.context.SoftmaxCrossEntropy;
                const targets = ctx.targets;
                const probs = ctx.probs;

                const B_size = logits.shape.dims[0];
                const D = logits.shape.dims[1];
                var loss_sum: f32 = 0.0;
                var total_w: f32 = 0.0;

                for (0..B_size) |i| {
                    const w: f32 = if (ctx.mask) |m| m[i] else 1.0;
                    if (w <= 0.0) continue;

                    const row = logits.data[i * D .. (i + 1) * D];
                    var max_val = row[0];
                    for (row[1..]) |val| {
                        if (val > max_val) max_val = val;
                    }

                    var sum: f32 = 0.0;
                    if (probs.len == B_size * D) {
                        const row_probs = probs[i * D .. (i + 1) * D];
                        for (0..D) |j| {
                            const e = @exp(row[j] - max_val);
                            row_probs[j] = e;
                            sum += e;
                        }
                        for (0..D) |j| {
                            row_probs[j] /= sum;
                        }
                    } else {
                        for (row) |val| {
                            sum += @exp(val - max_val);
                        }
                    }

                    const target_idx = targets[i];
                    if (ctx.mask != null) {
                        const log_sum_exp = max_val + @log(sum);
                        loss_sum += (log_sum_exp - row[target_idx]) * w;
                    } else {
                        const prob = if (probs.len == B_size * D) probs[i * D + target_idx] else (@exp(row[target_idx] - max_val) / sum);
                        const clipped = @max(prob, 1e-15);
                        loss_sum += -@log(clipped);
                    }
                    total_w += w;
                }

                ctx.total_weight = total_w;
                loss.data[0] = if (total_w > 0.0) loss_sum / total_w else 0.0;
            },
            .DpoLoss => {
                const pi_chosen = self.inputs[0];
                const pi_rejected = self.inputs[1];
                const loss = self.outputs[0];
                const ctx = self.context.DpoLoss;
                const N = pi_chosen.data.len;

                var total_loss: f32 = 0.0;
                for (0..N) |i| {
                    const log_ratio_chosen = pi_chosen.data[i] - ctx.ref_chosen[i];
                    const log_ratio_rejected = pi_rejected.data[i] - ctx.ref_rejected[i];
                    const z = ctx.beta * (log_ratio_chosen - log_ratio_rejected);
                    const loss_i = if (z > 0.0)
                        @log(1.0 + @exp(-z))
                    else
                        -z + @log(1.0 + @exp(z));
                    total_loss += loss_i;
                }
                loss.data[0] = if (N > 0) total_loss / @as(f32, @floatFromInt(N)) else 0.0;
            },
            .GrpoLoss => {
                const old_logps = self.inputs[0];
                const new_logps = self.inputs[1];
                const loss = self.outputs[0];
                const ctx = self.context.GrpoLoss;
                const N = old_logps.data.len;

                var total_obj: f32 = 0.0;
                for (0..N) |i| {
                    const ratio = @exp(new_logps.data[i] - old_logps.data[i]);
                    const adv = ctx.advantages[i];
                    const s1 = ratio * adv;
                    const clipped_ratio = std.math.clamp(ratio, 1.0 - ctx.clip_eps, 1.0 + ctx.clip_eps);
                    const s2 = clipped_ratio * adv;
                    const surrogate = @min(s1, s2);

                    var kl: f32 = 0.0;
                    if (ctx.beta > 0.0) {
                        const ref = if (ctx.ref_logps) |refs| refs[i] else old_logps.data[i];
                        const u = ref - new_logps.data[i];
                        kl = @exp(u) - u - 1.0;
                    }
                    total_obj += (surrogate - ctx.beta * kl);
                }
                loss.data[0] = if (N > 0) -(total_obj / @as(f32, @floatFromInt(N))) else 0.0;
            },
            .BceWithLogitsLoss, .SigmoidCrossEntropy => {
                copyFromEager(self.outputs[0], try self.inputs[0].bceWithLogitsLoss(self.inputs[1], allocator), allocator);
            },
            .BceLoss => {
                copyFromEager(self.outputs[0], try self.inputs[0].bceLoss(self.inputs[1], self.context.BceLoss.eps, allocator), allocator);
            },
            .Reshape => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (!C.is_view) {
                    copyFromEager(C, try A.reshape(C.shape.dims[0..C.shape.len], allocator), allocator);
                }
            },
            .Transpose => {
                const ctx = self.context.Transpose;
                copyFromEager(self.outputs[0], try self.inputs[0].transpose(ctx.dim0, ctx.dim1, allocator), allocator);
            },
            .Concat => {
                copyFromEager(self.outputs[0], try tensor_mod.concat(allocator, self.inputs, self.context.Concat.dim), allocator);
            },
            .Split => {
                const tmp_outs = try tensor_mod.split(allocator, self.inputs[0], self.outputs.len, self.context.Split.dim);
                defer {
                    for (tmp_outs) |t| t.deinit(allocator);
                    allocator.free(tmp_outs);
                }
                for (self.outputs, tmp_outs) |dest, src| {
                    @memcpy(dest.data, src.data);
                }
            },
            .RepeatKV => {
                copyFromEager(self.outputs[0], try self.inputs[0].repeatKV(self.context.RepeatKV.groups, allocator), allocator);
            },
            .MseLoss => {
                copyFromEager(self.outputs[0], try self.inputs[0].mseLoss(self.inputs[1], allocator), allocator);
            },
            .MulScalar => {
                copyFromEager(self.outputs[0], try self.inputs[0].mulScalar(self.context.MulScalar.val, allocator), allocator);
            },
            .DivScalar => {
                copyFromEager(self.outputs[0], try self.inputs[0].divScalar(self.context.DivScalar.val, allocator), allocator);
            },
            .AddScalar => {
                copyFromEager(self.outputs[0], try self.inputs[0].addScalar(self.context.AddScalar.val, allocator), allocator);
            },
            .SubScalar => {
                copyFromEager(self.outputs[0], try self.inputs[0].subScalar(self.context.SubScalar.val, allocator), allocator);
            },
            .Add, .AddBias => {
                copyFromEager(self.outputs[0], try self.inputs[0].add(self.inputs[1], allocator), allocator);
            },
            .Sub => {
                copyFromEager(self.outputs[0], try self.inputs[0].sub(self.inputs[1], allocator), allocator);
            },
            .Mul => {
                copyFromEager(self.outputs[0], try self.inputs[0].mul(self.inputs[1], allocator), allocator);
            },
            .Div => {
                copyFromEager(self.outputs[0], try self.inputs[0].div(self.inputs[1], allocator), allocator);
            },
            .Conv2D => {
                const bias = if (self.inputs.len > 2) self.inputs[2] else null;
                const ctx = self.context.Conv2D;
                copyFromEager(self.outputs[0], try self.inputs[0].conv2dWithConfig(self.inputs[1], bias, ctx.stride, ctx.padding, allocator), allocator);
            },
            .ConvTranspose2D => {
                const bias = if (self.inputs.len > 2) self.inputs[2] else null;
                const ctx = self.context.ConvTranspose2D;
                copyFromEager(self.outputs[0], try self.inputs[0].convTranspose2d(self.inputs[1], bias, ctx.stride, ctx.padding, allocator), allocator);
            },
            .MaxPool2D => {
                const ctx = self.context.MaxPool2D;
                copyFromEager(self.outputs[0], try self.inputs[0].maxpool2d(ctx.pool_size, ctx.stride, allocator), allocator);
            },
            .AvgPool2D => {
                const ctx = self.context.AvgPool2D;
                copyFromEager(self.outputs[0], try self.inputs[0].avgpool2d(ctx.kernel_size, ctx.stride, allocator), allocator);
            },
            .Softmax => {
                copyFromEager(self.outputs[0], try self.inputs[0].softmax(allocator), allocator);
            },
            .RmsNorm => {
                copyFromEager(self.outputs[0], try self.inputs[0].rmsNorm(self.inputs[1], self.context.RmsNorm.eps, allocator), allocator);
            },
            .LayerNorm => {
                copyFromEager(self.outputs[0], try self.inputs[0].layerNorm(self.inputs[1], self.inputs[2], self.context.LayerNorm.eps, allocator), allocator);
            },
            .BatchNorm2d => {
                const X = self.inputs[0];
                const G = self.inputs[1];
                const B = self.inputs[2];
                const Y = self.outputs[0];
                const ctx = self.context.BatchNorm2d;

                const N = X.shape.dims[0];
                const C = X.shape.dims[1];
                const H = X.shape.dims[2];
                const W = X.shape.dims[3];
                const spatial_size = H * W;

                for (0..C) |c_| {
                    const mean_val = ctx.save_mean[c_];
                    const inv_std = ctx.save_inv_std[c_];
                    const g_val = G.data[c_];
                    const b_val = B.data[c_];

                    for (0..N) |n| {
                        const in_slice = X.data[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                        const out_slice = Y.data[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                        for (in_slice, out_slice) |val, *o| {
                            o.* = (val - mean_val) * inv_std * g_val + b_val;
                        }
                    }
                }
            },
            .Dropout => {
                const X = self.inputs[0];
                const Y = self.outputs[0];
                const mask_scale = self.context.Dropout.mask_scale;
                for (X.data, Y.data, mask_scale) |val, *o, m| {
                    o.* = val * m;
                }
            },
            .RoPE => {
                const ctx = self.context.RoPE;
                copyFromEager(self.outputs[0], try self.inputs[0].ropeOffset(ctx.start_pos, ctx.rotary_offset, allocator), allocator);
            },
            .BatchMatMul => {
                copyFromEager(self.outputs[0], try self.inputs[0].batchMatMul(self.inputs[1], allocator), allocator);
            },
            .Embedding => {
                copyFromEager(self.outputs[0], try self.inputs[0].embedding(self.inputs[1], allocator), allocator);
            },
            .L2Loss => {
                copyFromEager(self.outputs[0], try self.inputs[0].l2Loss(self.context.L2Loss.lambda, allocator), allocator);
            },
            .L1Loss => {
                copyFromEager(self.outputs[0], try self.inputs[0].l1Loss(self.context.L1Loss.lambda, allocator), allocator);
            },
            .Sqrt => {
                copyFromEager(self.outputs[0], try self.inputs[0].sqrt(allocator), allocator);
            },
            .Exp => {
                copyFromEager(self.outputs[0], try self.inputs[0].exp(allocator), allocator);
            },
            .Log => {
                copyFromEager(self.outputs[0], try self.inputs[0].log(allocator), allocator);
            },
            .Abs => {
                copyFromEager(self.outputs[0], try self.inputs[0].abs(allocator), allocator);
            },
            .Sum => {
                const ctx = self.context.Sum;
                copyFromEager(self.outputs[0], try self.inputs[0].sum(ctx.axis, ctx.keepdims, allocator), allocator);
            },
            .Mean => {
                const ctx = self.context.Mean;
                copyFromEager(self.outputs[0], try self.inputs[0].mean(ctx.axis, ctx.keepdims, allocator), allocator);
            },
            .Where => {
                const X = self.inputs[0];
                const Y = self.inputs[1];
                const C = self.outputs[0];
                const mask = self.context.Where.mask;
                const x_strides = tensor_mod.computeBroadcastStrides(X.shape, X.strides, C.shape);
                const y_strides = tensor_mod.computeBroadcastStrides(Y.shape, Y.strides, C.shape);
                const rank = C.shape.len;
                var indices = [_]usize{0} ** 8;
                for (0..C.data.len) |c_flat| {
                    var x_flat: usize = 0;
                    var y_flat: usize = 0;
                    for (0..rank) |d| {
                        x_flat += indices[d] * x_strides.dims[d];
                        y_flat += indices[d] * y_strides.dims[d];
                    }
                    C.data[c_flat] = if (mask[c_flat]) X.data[x_flat] else Y.data[y_flat];
                    var d = rank;
                    while (d > 0) {
                        d -= 1;
                        indices[d] += 1;
                        if (indices[d] < C.shape.dims[d]) break;
                        indices[d] = 0;
                    }
                }
            },
            .MaskedFill => {
                const X = self.inputs[0];
                const C = self.outputs[0];
                const ctx = self.context.MaskedFill;
                var coord = [_]usize{0} ** 8;
                const len = X.shape.len;
                for (0..C.data.len) |i| {
                    if (ctx.mask[i]) {
                        C.data[i] = ctx.value;
                    } else {
                        var src_idx: usize = 0;
                        for (0..len) |d| src_idx += coord[d] * X.strides.dims[d];
                        C.data[i] = X.data[src_idx];
                    }
                    var d = len;
                    while (d > 0) {
                        d -= 1;
                        coord[d] += 1;
                        if (coord[d] < X.shape.dims[d]) break;
                        coord[d] = 0;
                    }
                }
            },
            .Slice => {
                const X = self.inputs[0];
                const C = self.outputs[0];
                const ctx = self.context.Slice;
                var coord = [_]usize{0} ** 8;
                for (0..C.data.len) |dest_i| {
                    var src_idx: usize = ctx.offset;
                    for (0..ctx.rank) |d| src_idx += coord[d] * ctx.strides[d];
                    C.data[dest_i] = X.data[src_idx];
                    var d = ctx.rank;
                    while (d > 0) {
                        d -= 1;
                        coord[d] += 1;
                        if (coord[d] < C.shape.dims[d]) break;
                        coord[d] = 0;
                    }
                }
            },
        }
    }

    fn reduceSumMeanBackward(A: *Tensor, C: *Tensor, axis: ?usize, keepdims: bool, scale: f32) void {
        if (!A.requires_grad) return;
        if (axis) |ax| {
            const reduce_size = A.shape.dims[ax];
            var outer_size: usize = 1;
            for (0..ax) |d| outer_size *= A.shape.dims[d];
            var inner_size: usize = 1;
            for ((ax + 1)..A.shape.len) |d| inner_size *= A.shape.dims[d];

            if (A.isContiguous()) {
                for (0..outer_size) |outer| {
                    const out_base = outer * inner_size;
                    const src_base = outer * reduce_size * inner_size;
                    for (0..inner_size) |inner| {
                        const g = C.grad[out_base + inner] * scale;
                        for (0..reduce_size) |k| {
                            A.grad[src_base + k * inner_size + inner] += g;
                        }
                    }
                }
            } else {
                const out_rank = C.shape.len;
                var out_indices = [_]usize{0} ** 8;
                for (0..C.grad.len) |out_idx| {
                    var tmp = out_idx;
                    var d: usize = out_rank;
                    while (d > 0) {
                        d -= 1;
                        out_indices[d] = tmp % C.shape.dims[d];
                        tmp /= C.shape.dims[d];
                    }

                    var src_indices = [_]usize{0} ** 8;
                    if (keepdims) {
                        for (0..A.shape.len) |idx_d| {
                            src_indices[idx_d] = out_indices[idx_d];
                        }
                    } else {
                        var src_d: usize = 0;
                        for (0..A.shape.len) |idx_d| {
                            if (idx_d == ax) continue;
                            src_indices[idx_d] = out_indices[src_d];
                            src_d += 1;
                        }
                    }

                    const g = C.grad[out_idx] * scale;
                    for (0..reduce_size) |k| {
                        src_indices[ax] = k;
                        A.grad[A.getFlatIndex(src_indices[0..A.shape.len])] += g;
                    }
                }
            }
        } else {
            const g = C.grad[0] * scale;
            const elem_count = A.shape.numel();
            if (A.isContiguous() and A.grad.len >= elem_count) {
                for (A.grad[0..elem_count]) |*ag| {
                    ag.* += g;
                }
            } else {
                var coord = [_]usize{0} ** 8;
                const len = A.shape.len;
                for (0..elem_count) |_| {
                    var src_idx: usize = 0;
                    for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
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
            // 2. ReLU 激活函数反向传播 (ReLU Backward)
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
            // 前向公式: C = reshape(A)
            // 数学推导:
            //   Reshape 没有数学上的参数变换，若 A 为连续布局则直接逐元素累加，
            //   若 A 为非连续跨步视图则按 A.strides 累加回对应的物理偏移位置。
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
            .Conv2D => {
                const A = self.inputs[0];
                const W = self.inputs[1];
                const C = self.outputs[0];
                const stride = self.context.Conv2D.stride;
                const padding = self.context.Conv2D.padding;

                const N = A.shape.dims[0];
                const C_in = A.shape.dims[1];
                const H = A.shape.dims[2];
                const W_in = A.shape.dims[3];
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

                // Bias gradient: db[co] += sum_{n, h_out, w_out} dC[n, co, h_out, w_out]
                if (self.inputs.len > 2) {
                    const bias = self.inputs[2];
                    if (bias.requires_grad) {
                        const b_stride = bias.strides.dims[0];
                        for (0..N) |n| {
                            for (0..C_out) |co| {
                                var acc: f32 = 0.0;
                                for (0..H_out) |h_out| {
                                    for (0..W_out) |w_out| {
                                        acc += C.grad[n * o_n + co * o_c + h_out * o_h + w_out * o_w];
                                    }
                                }
                                bias.grad[co * b_stride] += acc;
                            }
                        }
                    }
                }

                const K_col = C_in * KH * KW;
                const L_out = H_out * W_out;
                var used_sgemm = false;

                if ((A.requires_grad or W.requires_grad) and A.isContiguous() and W.isContiguous() and C.isContiguous() and K_col > 0 and L_out > 0 and C_out > 0) {
                    var stack_buf: [4096]f32 = undefined;
                    const need_len = K_col * L_out;
                    const heap_buf: ?[]f32 = if (need_len > stack_buf.len)
                        (std.heap.c_allocator.alloc(f32, need_len) catch null)
                    else
                        null;
                    defer if (heap_buf) |hb| std.heap.c_allocator.free(hb);

                    const col_opt: ?[]f32 = if (need_len <= stack_buf.len) stack_buf[0..need_len] else heap_buf;
                    if (col_opt) |col_buf| {
                        used_sgemm = true;
                        for (0..N) |n| {
                            const dC_n = C.grad[n * C_out * L_out .. (n + 1) * C_out * L_out];

                            if (W.requires_grad) {
                                // im2col(A_n) -> col_buf [K_col, L_out]
                                for (0..C_in) |ci| {
                                    for (0..KH) |kh| {
                                        for (0..KW) |kw| {
                                            const k_row = (ci * KH + kh) * KW + kw;
                                            const col_row = col_buf[k_row * L_out .. (k_row + 1) * L_out];
                                            for (0..H_out) |h_out| {
                                                const ih_signed: isize = @as(isize, @intCast(h_out * stride + kh)) - @as(isize, @intCast(padding));
                                                if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) {
                                                    @memset(col_row[h_out * W_out .. (h_out + 1) * W_out], 0.0);
                                                    continue;
                                                }
                                                const ih: usize = @intCast(ih_signed);
                                                for (0..W_out) |w_out| {
                                                    const iw_s: isize = @as(isize, @intCast(w_out * stride + kw)) - @as(isize, @intCast(padding));
                                                    if (iw_s >= 0 and iw_s < @as(isize, @intCast(W_in))) {
                                                        const iw: usize = @intCast(iw_s);
                                                        col_row[h_out * W_out + w_out] = A.data[n * s_n + ci * s_c + ih * s_h + iw * s_w];
                                                    } else {
                                                        col_row[h_out * W_out + w_out] = 0.0;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // dW += dC_n * col_buf^T
                                c.cblas_sgemm(
                                    c.CblasRowMajor,
                                    c.CblasNoTrans,
                                    c.CblasTrans,
                                    @intCast(C_out),
                                    @intCast(K_col),
                                    @intCast(L_out),
                                    1.0,
                                    dC_n.ptr,
                                    @intCast(L_out),
                                    col_buf.ptr,
                                    @intCast(L_out),
                                    1.0,
                                    W.grad.ptr,
                                    @intCast(K_col),
                                );
                            }

                            if (A.requires_grad) {
                                // col_buf = W^T * dC_n -> [K_col, L_out]
                                c.cblas_sgemm(
                                    c.CblasRowMajor,
                                    c.CblasTrans,
                                    c.CblasNoTrans,
                                    @intCast(K_col),
                                    @intCast(L_out),
                                    @intCast(C_out),
                                    1.0,
                                    W.data.ptr,
                                    @intCast(K_col),
                                    dC_n.ptr,
                                    @intCast(L_out),
                                    0.0,
                                    col_buf.ptr,
                                    @intCast(L_out),
                                );

                                // col2im(col_buf) -> A.grad[n]
                                for (0..C_in) |ci| {
                                    for (0..KH) |kh| {
                                        for (0..KW) |kw| {
                                            const k_row = (ci * KH + kh) * KW + kw;
                                            const col_row = col_buf[k_row * L_out .. (k_row + 1) * L_out];
                                            for (0..H_out) |h_out| {
                                                const ih_signed: isize = @as(isize, @intCast(h_out * stride + kh)) - @as(isize, @intCast(padding));
                                                if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) continue;
                                                const ih: usize = @intCast(ih_signed);
                                                for (0..W_out) |w_out| {
                                                    const iw_s: isize = @as(isize, @intCast(w_out * stride + kw)) - @as(isize, @intCast(padding));
                                                    if (iw_s >= 0 and iw_s < @as(isize, @intCast(W_in))) {
                                                        const iw: usize = @intCast(iw_s);
                                                        A.grad[n * s_n + ci * s_c + ih * s_h + iw * s_w] += col_row[h_out * W_out + w_out];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (!used_sgemm and (A.requires_grad or W.requires_grad)) {
                    for (0..N) |n| {
                        for (0..C_out) |co| {
                            for (0..H_out) |h_out| {
                                for (0..W_out) |w_out| {
                                    const grad_val = C.grad[n * o_n + co * o_c + h_out * o_h + w_out * o_w];
                                    if (grad_val == 0.0) continue;

                                    for (0..C_in) |ci| {
                                        for (0..KH) |kh| {
                                            const ih_signed: isize = @as(isize, @intCast(h_out * stride + kh)) - @as(isize, @intCast(padding));
                                            if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) continue;
                                            const ih: usize = @intCast(ih_signed);
                                            for (0..KW) |kw| {
                                                const iw_signed: isize = @as(isize, @intCast(w_out * stride + kw)) - @as(isize, @intCast(padding));
                                                if (iw_signed < 0 or iw_signed >= @as(isize, @intCast(W_in))) continue;
                                                const iw: usize = @intCast(iw_signed);

                                                if (W.requires_grad) {
                                                    const input_val = A.data[n * s_n + ci * s_c + ih * s_h + iw * s_w];
                                                    W.grad[co * w_co + ci * w_ci + kh * w_kh + kw * w_kw] += grad_val * input_val;
                                                }

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
                }
            },
            .ConvTranspose2D => {
                const A = self.inputs[0];
                const W = self.inputs[1];
                const C = self.outputs[0];
                const bias = if (self.inputs.len > 2) self.inputs[2] else null;
                const stride = self.context.ConvTranspose2D.stride;
                const padding = self.context.ConvTranspose2D.padding;

                const N = A.shape.dims[0];
                const C_in = A.shape.dims[1];
                const H_in = A.shape.dims[2];
                const W_in = A.shape.dims[3];

                const C_out = W.shape.dims[1];
                const KH = W.shape.dims[2];
                const KW = W.shape.dims[3];

                const H_out = C.shape.dims[2];
                const W_out = C.shape.dims[3];

                // Bias gradient
                if (bias) |b| {
                    if (b.requires_grad) {
                        for (0..N) |n| {
                            for (0..C_out) |co| {
                                for (0..H_out) |h| {
                                    for (0..W_out) |w| {
                                        b.grad[co] += C.grad[n * (C_out * H_out * W_out) + co * (H_out * W_out) + h * W_out + w];
                                    }
                                }
                            }
                        }
                    }
                }

                for (0..N) |n| {
                    for (0..C_in) |ci| {
                        for (0..H_in) |h| {
                            for (0..W_in) |w| {
                                const in_idx = n * (C_in * H_in * W_in) + ci * (H_in * W_in) + h * W_in + w;
                                const input_val = A.data[in_idx];

                                for (0..C_out) |co| {
                                    for (0..KH) |kh| {
                                        const out_h_raw = h * stride + kh;
                                        if (out_h_raw < padding) continue;
                                        const out_h = out_h_raw - padding;
                                        if (out_h >= H_out) continue;

                                        for (0..KW) |kw| {
                                            const out_w_raw = w * stride + kw;
                                            if (out_w_raw < padding) continue;
                                            const out_w = out_w_raw - padding;
                                            if (out_w >= W_out) continue;

                                            const out_idx = n * (C_out * H_out * W_out) + co * (H_out * W_out) + out_h * W_out + out_w;
                                            const grad_out = C.grad[out_idx];
                                            if (grad_out == 0.0) continue;

                                            const w_idx = ci * (C_out * KH * KW) + co * (KH * KW) + kh * KW + kw;

                                            if (W.requires_grad) {
                                                W.grad[w_idx] += grad_out * input_val;
                                            }

                                            if (A.requires_grad) {
                                                A.grad[in_idx] += grad_out * W.data[w_idx];
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
            .AvgPool2D => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const N = A.shape.dims[0];
                const C_ch = A.shape.dims[1];
                const H = A.shape.dims[2];
                const W = A.shape.dims[3];
                const H_out = C.shape.dims[2];
                const W_out = C.shape.dims[3];

                const kernel_size = self.context.AvgPool2D.kernel_size;
                const stride = self.context.AvgPool2D.stride;
                const pool_area = @as(f32, @floatFromInt(kernel_size * kernel_size));

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
                            for (0..H_out) |oh| {
                                for (0..W_out) |ow| {
                                    const grad_val = C.grad[n * o_n + c_ * o_c + oh * o_h + ow * o_w];
                                    if (grad_val == 0.0) continue;
                                    const distributed_grad = grad_val / pool_area;

                                    for (0..kernel_size) |kh| {
                                        for (0..kernel_size) |kw| {
                                            const ih = oh * stride + kh;
                                            const iw = ow * stride + kw;
                                            if (ih < H and iw < W) {
                                                A.grad[n * s_n + c_ * s_c + ih * s_h + iw * s_w] += distributed_grad;
                                            }
                                        }
                                    }
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
            .LayerNorm => {
                const X = self.inputs[0];
                const G = self.inputs[1];
                const B = self.inputs[2];
                const Y = self.outputs[0];
                const eps = self.context.LayerNorm.eps;
                const D = X.shape.dims[X.shape.len - 1];
                const M = X.data.len / D;
                const d_f = @as(f32, @floatFromInt(D));

                for (0..M) |i| {
                    const row_in = X.data[i * D .. (i + 1) * D];
                    const row_grad_out = Y.grad[i * D .. (i + 1) * D];

                    var sum_x: f32 = 0.0;
                    for (row_in) |val| sum_x += val;
                    const mean_val = sum_x / d_f;

                    var var_sum: f32 = 0.0;
                    for (row_in) |val| {
                        const diff = val - mean_val;
                        var_sum += diff * diff;
                    }
                    const inv_std = 1.0 / @sqrt(var_sum / d_f + eps);

                    if (B.requires_grad) {
                        for (0..D) |j| {
                            B.grad[j] += row_grad_out[j];
                        }
                    }

                    if (G.requires_grad) {
                        for (0..D) |j| {
                            const x_hat = (row_in[j] - mean_val) * inv_std;
                            G.grad[j] += row_grad_out[j] * x_hat;
                        }
                    }

                    if (X.requires_grad) {
                        const row_grad_in = X.grad[i * D .. (i + 1) * D];
                        var sum_dx_hat: f32 = 0.0;
                        var sum_dx_hat_x_hat: f32 = 0.0;
                        for (0..D) |j| {
                            const x_hat = (row_in[j] - mean_val) * inv_std;
                            const dx_hat = row_grad_out[j] * G.data[j];
                            sum_dx_hat += dx_hat;
                            sum_dx_hat_x_hat += dx_hat * x_hat;
                        }
                        for (0..D) |j| {
                            const x_hat = (row_in[j] - mean_val) * inv_std;
                            const dx_hat = row_grad_out[j] * G.data[j];
                            row_grad_in[j] += (inv_std / d_f) * (d_f * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat);
                        }
                    }
                }
            },
            .BatchNorm2d => {
                const X = self.inputs[0];
                const G = self.inputs[1];
                const B = self.inputs[2];
                const Y = self.outputs[0];
                const ctx = self.context.BatchNorm2d;

                const N = X.shape.dims[0];
                const C = X.shape.dims[1];
                const H = X.shape.dims[2];
                const W = X.shape.dims[3];
                const spatial_size = H * W;
                const m_f = @as(f32, @floatFromInt(N * spatial_size));

                for (0..C) |c_| {
                    const mean_val = ctx.save_mean[c_];
                    const inv_std = ctx.save_inv_std[c_];
                    const g_val = G.data[c_];

                    var dbeta: f32 = 0.0;
                    var dgamma: f32 = 0.0;
                    for (0..N) |n| {
                        const in_slice = X.data[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                        const dy_slice = Y.grad[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                        for (in_slice, dy_slice) |x_val, dy_val| {
                            const x_hat = (x_val - mean_val) * inv_std;
                            dbeta += dy_val;
                            dgamma += dy_val * x_hat;
                        }
                    }

                    if (B.requires_grad) {
                        B.grad[c_] += dbeta;
                    }
                    if (G.requires_grad) {
                        G.grad[c_] += dgamma;
                    }

                    if (X.requires_grad) {
                        if (ctx.training) {
                            const factor = (g_val * inv_std) / m_f;
                            for (0..N) |n| {
                                const in_slice = X.data[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                                const dy_slice = Y.grad[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                                const dx_slice = X.grad[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                                for (in_slice, dy_slice, dx_slice) |x_val, dy_val, *dx_val| {
                                    const x_hat = (x_val - mean_val) * inv_std;
                                    dx_val.* += factor * (m_f * dy_val - dbeta - x_hat * dgamma);
                                }
                            }
                        } else {
                            const factor = g_val * inv_std;
                            for (0..N) |n| {
                                const dy_slice = Y.grad[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                                const dx_slice = X.grad[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                                for (dy_slice, dx_slice) |dy_val, *dx_val| {
                                    dx_val.* += factor * dy_val;
                                }
                            }
                        }
                    }
                }
            },
            .Dropout => {
                const X = self.inputs[0];
                const Y = self.outputs[0];
                const mask_scale = self.context.Dropout.mask_scale;
                if (X.requires_grad) {
                    for (X.grad, Y.grad, mask_scale) |*dx, dy, m| {
                        dx.* += dy * m;
                    }
                }
            },
            .RoPE => {
                const X = self.inputs[0];
                const Y = self.outputs[0];
                if (X.requires_grad) {
                    const start_pos = self.context.RoPE.start_pos;
                    const rotary_offset = self.context.RoPE.rotary_offset;
                    const D = X.shape.dims[X.shape.len - 1];
                    const T = if (X.shape.len >= 2) X.shape.dims[X.shape.len - 2] else 1;
                    const outer = X.data.len / (T * D);
                    const rot_dim = D - rotary_offset;
                    const half = rot_dim / 2;
                    const rot_dim_f = @as(f32, @floatFromInt(rot_dim));

                    for (0..outer) |o| {
                        for (0..T) |t| {
                            const dx_row = X.grad[(o * T + t) * D .. (o * T + t + 1) * D];
                            const dy_row = Y.grad[(o * T + t) * D .. (o * T + t + 1) * D];
                            for (0..rotary_offset) |j| {
                                dx_row[j] += dy_row[j];
                            }
                            const pos_f = @as(f32, @floatFromInt(start_pos + t));
                            for (0..half) |i| {
                                const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / rot_dim_f);
                                const theta = pos_f * freq;
                                const cos_t = @cos(theta);
                                const sin_t = @sin(theta);
                                const dy0 = dy_row[rotary_offset + 2 * i];
                                const dy1 = dy_row[rotary_offset + 2 * i + 1];
                                dx_row[rotary_offset + 2 * i] += dy0 * cos_t + dy1 * sin_t;
                                dx_row[rotary_offset + 2 * i + 1] += -dy0 * sin_t + dy1 * cos_t;
                            }
                            if (2 * half < rot_dim) {
                                dx_row[D - 1] += dy_row[D - 1];
                            }
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

                const D = W.shape.dims[1];
                const num_indices = X.shape.numel();

                if (W.requires_grad) {
                    if (X.isContiguous() and X.data.len >= num_indices) {
                        for (0..num_indices) |i| {
                            const idx = @as(usize, @intFromFloat(X.data[i]));
                            const w_grad_row = W.grad[idx * D .. (idx + 1) * D];
                            const y_grad_row = Y.grad[i * D .. (i + 1) * D];

                            for (w_grad_row, y_grad_row) |*wg, yg| {
                                wg.* += yg;
                            }
                        }
                    } else {
                        var coord = [_]usize{0} ** 8;
                        const rank = X.shape.len;
                        for (0..num_indices) |i| {
                            var x_flat: usize = 0;
                            for (0..rank) |d| x_flat += coord[d] * X.strides.dims[d];
                            const idx = @as(usize, @intFromFloat(X.data[x_flat]));
                            const w_grad_row = W.grad[idx * D .. (idx + 1) * D];
                            const y_grad_row = Y.grad[i * D .. (i + 1) * D];

                            for (w_grad_row, y_grad_row) |*wg, yg| {
                                wg.* += yg;
                            }
                            var d = rank;
                            while (d > 0) {
                                d -= 1;
                                coord[d] += 1;
                                if (coord[d] < X.shape.dims[d]) break;
                                coord[d] = 0;
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
            .Sqrt => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    if (A.isContiguous() and A.grad.len >= C.grad.len) {
                        for (C.grad, C.data, 0..) |cg, c_val, i| {
                            A.grad[i] += cg / (2.0 * @max(c_val, 1e-12));
                        }
                    } else {
                        var coord = [_]usize{0} ** 8;
                        const len = A.shape.len;
                        for (C.grad, C.data) |cg, c_val| {
                            var src_idx: usize = 0;
                            for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                            A.grad[src_idx] += cg / (2.0 * @max(c_val, 1e-12));
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
            .Exp => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    if (A.isContiguous() and A.grad.len >= C.grad.len) {
                        for (C.grad, C.data, 0..) |cg, c_val, i| {
                            A.grad[i] += cg * c_val;
                        }
                    } else {
                        var coord = [_]usize{0} ** 8;
                        const len = A.shape.len;
                        for (C.grad, C.data) |cg, c_val| {
                            var src_idx: usize = 0;
                            for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                            A.grad[src_idx] += cg * c_val;
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
            .Log => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    if (A.isContiguous() and A.grad.len >= C.grad.len) {
                        for (C.grad, A.data[0..C.grad.len], 0..) |cg, a_val, i| {
                            A.grad[i] += cg / a_val;
                        }
                    } else {
                        var coord = [_]usize{0} ** 8;
                        const len = A.shape.len;
                        for (C.grad) |cg| {
                            var src_idx: usize = 0;
                            for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                            A.grad[src_idx] += cg / A.data[src_idx];
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
            .Abs => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    if (A.isContiguous() and A.grad.len >= C.grad.len) {
                        for (C.grad, A.data[0..C.grad.len], 0..) |cg, a_val, i| {
                            const sign: f32 = if (a_val > 0.0) 1.0 else if (a_val < 0.0) -1.0 else 0.0;
                            A.grad[i] += cg * sign;
                        }
                    } else {
                        var coord = [_]usize{0} ** 8;
                        const len = A.shape.len;
                        for (C.grad) |cg| {
                            var src_idx: usize = 0;
                            for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                            const a_val = A.data[src_idx];
                            const sign: f32 = if (a_val > 0.0) 1.0 else if (a_val < 0.0) -1.0 else 0.0;
                            A.grad[src_idx] += cg * sign;
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
            .Sum => {
                const ctx = self.context.Sum;
                reduceSumMeanBackward(self.inputs[0], self.outputs[0], ctx.axis, ctx.keepdims, 1.0);
            },
            .Mean => {
                const A = self.inputs[0];
                const ctx = self.context.Mean;
                const count = if (ctx.axis) |ax| @as(f32, @floatFromInt(A.shape.dims[ax])) else @as(f32, @floatFromInt(A.shape.numel()));
                reduceSumMeanBackward(A, self.outputs[0], ctx.axis, ctx.keepdims, 1.0 / count);
            },
            .Where => {
                const X = self.inputs[0];
                const Y = self.inputs[1];
                const C = self.outputs[0];
                const mask = self.context.Where.mask;
                if (X.requires_grad or Y.requires_grad) {
                    const x_strides = tensor_mod.computeBroadcastStrides(X.shape, X.strides, C.shape);
                    const y_strides = tensor_mod.computeBroadcastStrides(Y.shape, Y.strides, C.shape);
                    const rank = C.shape.len;
                    var indices = [_]usize{0} ** 8;
                    for (0..C.grad.len) |c_flat| {
                        const g = C.grad[c_flat];
                        if (mask[c_flat]) {
                            if (X.requires_grad) {
                                var x_flat: usize = 0;
                                for (0..rank) |d| x_flat += indices[d] * x_strides.dims[d];
                                X.grad[x_flat] += g;
                            }
                        } else {
                            if (Y.requires_grad) {
                                var y_flat: usize = 0;
                                for (0..rank) |d| y_flat += indices[d] * y_strides.dims[d];
                                Y.grad[y_flat] += g;
                            }
                        }
                        var d = rank;
                        while (d > 0) {
                            d -= 1;
                            indices[d] += 1;
                            if (indices[d] < C.shape.dims[d]) break;
                            indices[d] = 0;
                        }
                    }
                }
            },
            .MaskedFill => {
                const X = self.inputs[0];
                const C = self.outputs[0];
                const ctx = self.context.MaskedFill;
                if (X.requires_grad) {
                    if (X.isContiguous() and X.grad.len >= C.grad.len) {
                        for (C.grad, ctx.mask, 0..) |g, m, i| {
                            if (!m) X.grad[i] += g;
                        }
                    } else {
                        var coord = [_]usize{0} ** 8;
                        const len = X.shape.len;
                        for (C.grad, ctx.mask) |g, m| {
                            if (!m) {
                                var src_idx: usize = 0;
                                for (0..len) |d| src_idx += coord[d] * X.strides.dims[d];
                                X.grad[src_idx] += g;
                            }
                            var d = len;
                            while (d > 0) {
                                d -= 1;
                                coord[d] += 1;
                                if (coord[d] < X.shape.dims[d]) break;
                                coord[d] = 0;
                            }
                        }
                    }
                }
            },
            .Slice => {
                const X = self.inputs[0];
                const C = self.outputs[0];
                const ctx = self.context.Slice;
                if (X.requires_grad) {
                    var coord = [_]usize{0} ** 8;
                    for (0..C.grad.len) |dest_i| {
                        var src_idx: usize = ctx.offset;
                        for (0..ctx.rank) |d| src_idx += coord[d] * ctx.strides[d];
                        X.grad[src_idx] += C.grad[dest_i];
                        var d = ctx.rank;
                        while (d > 0) {
                            d -= 1;
                            coord[d] += 1;
                            if (coord[d] < C.shape.dims[d]) break;
                            coord[d] = 0;
                        }
                    }
                }
            },
        }
    }

};

