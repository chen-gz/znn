const std = @import("std");
const tensor_mod = @import("../tensor.zig");
pub const tensor = tensor_mod;
const Tensor = tensor_mod.Tensor;
const types = @import("types.zig");
pub const OpType = types.OpType;
pub const OpContext = types.OpContext;
const backward_core = @import("backward_core.zig");
const backward_nn = @import("backward_nn.zig");
const backward_math = @import("backward_math.zig");

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
            .Conv1D => {
                const bias = if (self.inputs.len > 2) self.inputs[2] else null;
                const ctx = self.context.Conv1D;
                copyFromEager(self.outputs[0], try self.inputs[0].conv1d(self.inputs[1], bias, .{ .stride = ctx.stride, .padding = ctx.padding }, allocator), allocator);
            },
            .Conv2D => {
                const bias = if (self.inputs.len > 2) self.inputs[2] else null;
                const ctx = self.context.Conv2D;
                copyFromEager(self.outputs[0], try self.inputs[0].conv2d(self.inputs[1], bias, .{ .stride = ctx.stride, .padding = ctx.padding }, allocator), allocator);
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
                copyFromEager(self.outputs[0], try self.inputs[0].ropeOffsetWithTheta(ctx.start_pos, ctx.rotary_offset, ctx.rope_theta, allocator), allocator);
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

    // 执行该算子的反向传播计算，更新其输入节点的梯度
    pub fn backward(self: *Op) !void {
        switch (self.op_type) {
            .MatMul,
            .Relu,
            .Gelu,
            .Sigmoid,
            .Tanh,
            .LeakyRelu,
            .Silu,
            .SoftmaxCrossEntropy,
            .DpoLoss,
            .GrpoLoss,
            .BceWithLogitsLoss,
            .SigmoidCrossEntropy,
            .BceLoss,
            .Reshape,
            .Transpose,
            .Concat,
            .Split,
            .RepeatKV,
            .MseLoss,
            .MulScalar,
            .DivScalar,
            .AddScalar,
            .SubScalar,
            .Add,
            .AddBias,
            .Sub,
            .Mul,
            .Div,
            => try backward_core.backwardCore(self),

            .Conv1D,
            .Conv2D,
            .ConvTranspose2D,
            .MaxPool2D,
            .AvgPool2D,
            .Softmax,
            .RmsNorm,
            .LayerNorm,
            .BatchNorm2d,
            .Dropout,
            .RoPE,
            .BatchMatMul,
            .Embedding,
            => try backward_nn.backwardNN(self),

            .L2Loss,
            .L1Loss,
            .Sqrt,
            .Exp,
            .Log,
            .Abs,
            .Sum,
            .Mean,
            .Where,
            .MaskedFill,
            .Slice,
            => try backward_math.backwardMath(self),
        }
    }
};
