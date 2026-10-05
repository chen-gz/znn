const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const types = @import("types.zig");
const OpType = types.OpType;
const OpContext = types.OpContext;
const op_mod = @import("op.zig");
const Op = op_mod.Op;
const Graph = @import("graph.zig").Graph;

// 损失函数 Softmax + Cross Entropy 结合前向传播
// 在 logits 的行维度计算 Softmax 概率分布，并与 targets 分类标签（支持 u8/u32/usize 等任意整型切片）计算交叉熵损失
pub fn softmaxCrossEntropy(self: *Graph, logits: *Tensor, targets: anytype) !*Tensor {
    const B = logits.shape.dims[0];
    const N = logits.shape.dims[1];
    if (targets.len != B) return error.ShapeMismatch;

    const allocator = self.arena.allocator();
    const targets_copy = try allocator.alloc(usize, B);
    for (0..B) |i| {
        const label: usize = @intCast(targets[i]);
        if (label >= N) return error.IndexOutOfBounds;
        targets_copy[i] = label;
    }

    const req_grad = self.enable_grad and logits.requires_grad;
    const loss = try self.tensor(1, 1, req_grad);

    var empty_probs: [0]f32 = .{};
    const probs: []f32 = if (req_grad) try allocator.alloc(f32, B * N) else &empty_probs;

    return self.runAndRecordPreallocatedOp(
        loss,
        &.{logits},
        .SoftmaxCrossEntropy,
        .{
            .SoftmaxCrossEntropy = .{
                .probs = probs,
                .targets = targets_copy,
                .mask = null,
                .total_weight = @as(f32, @floatFromInt(B)),
            },
        },
        req_grad,
    );
}

// 监督微调 (Supervised Fine-Tuning, SFT) 掩码交叉熵损失：仅对 mask[i] > 0 的位置计算交叉熵并支持自动微分 (Automatic Differentiation, Autograd) 反向传播
pub fn maskedCrossEntropyLoss(self: *Graph, logits: *Tensor, targets: anytype, mask: []const f32) !*Tensor {
    const B = logits.shape.dims[0];
    const N = logits.shape.dims[1];
    if (targets.len != B or mask.len != B) return error.ShapeMismatch;

    const allocator = self.arena.allocator();
    const targets_copy = try allocator.alloc(usize, B);
    for (0..B) |i| {
        const label: usize = @intCast(targets[i]);
        if (label >= N) return error.IndexOutOfBounds;
        targets_copy[i] = label;
    }
    const mask_copy = try allocator.alloc(f32, B);
    @memcpy(mask_copy, mask);

    const req_grad = self.enable_grad and logits.requires_grad;
    const loss = try self.tensor(1, 1, req_grad);

    var empty_probs: [0]f32 = .{};
    const probs: []f32 = if (req_grad) try allocator.alloc(f32, B * N) else &empty_probs;
    if (req_grad) @memset(probs, 0.0);

    return self.runAndRecordPreallocatedOp(
        loss,
        &.{logits},
        .SoftmaxCrossEntropy,
        .{
            .SoftmaxCrossEntropy = .{
                .probs = probs,
                .targets = targets_copy,
                .mask = mask_copy,
                .total_weight = 0.0,
            },
        },
        req_grad,
    );
}

// 直接偏好优化 (Direct Preference Optimization, DPO) 损失函数：支持对策略模型对数概率 pi_chosen_logps / pi_rejected_logps 的计算图反向传播
pub fn dpoLoss(
    self: *Graph,
    pi_chosen_logps: *Tensor,
    pi_rejected_logps: *Tensor,
    ref_chosen_logps: []const f32,
    ref_rejected_logps: []const f32,
    beta: f32,
) !*Tensor {
    const N = pi_chosen_logps.data.len;
    if (pi_rejected_logps.data.len != N or ref_chosen_logps.len != N or ref_rejected_logps.len != N) {
        return error.ShapeMismatch;
    }

    const allocator = self.arena.allocator();
    const ref_c_copy = try allocator.alloc(f32, N);
    @memcpy(ref_c_copy, ref_chosen_logps);
    const ref_r_copy = try allocator.alloc(f32, N);
    @memcpy(ref_r_copy, ref_rejected_logps);

    const req_grad = self.enable_grad and (pi_chosen_logps.requires_grad or pi_rejected_logps.requires_grad);
    const loss = try self.tensor(1, 1, req_grad);

    return self.runAndRecordPreallocatedOp(
        loss,
        &.{ pi_chosen_logps, pi_rejected_logps },
        .DpoLoss,
        .{
            .DpoLoss = .{
                .ref_chosen = ref_c_copy,
                .ref_rejected = ref_r_copy,
                .beta = beta,
            },
        },
        req_grad,
    );
}

// 群组相对策略优化 (Group Relative Policy Optimization, GRPO) 损失函数：支持在计算图中对 new_logps 自动微分求导
pub fn grpoLoss(
    self: *Graph,
    old_logps: *Tensor,
    new_logps: *Tensor,
    advantages: []const f32,
    ref_logps: ?[]const f32,
    beta: f32,
    clip_eps: f32,
) !*Tensor {
    const N = old_logps.data.len;
    if (new_logps.data.len != N or advantages.len != N) return error.ShapeMismatch;
    if (ref_logps) |refs| {
        if (refs.len != N) return error.ShapeMismatch;
    }

    const allocator = self.arena.allocator();
    const adv_copy = try allocator.alloc(f32, N);
    @memcpy(adv_copy, advantages);

    var ref_copy: ?[]const f32 = null;
    if (ref_logps) |refs| {
        const rc = try allocator.alloc(f32, N);
        @memcpy(rc, refs);
        ref_copy = rc;
    }

    const req_grad = self.enable_grad and new_logps.requires_grad;
    const loss = try self.tensor(1, 1, req_grad);

    return self.runAndRecordPreallocatedOp(
        loss,
        &.{ old_logps, new_logps },
        .GrpoLoss,
        .{
            .GrpoLoss = .{
                .advantages = adv_copy,
                .ref_logps = ref_copy,
                .beta = beta,
                .clip_eps = clip_eps,
            },
        },
        req_grad,
    );
}

// 均方误差 (Mean Squared Error, MSE) 损失函数：C = 1/N * sum((y_pred - y_true)^2)
pub fn mseLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor) !*Tensor {
    const req_grad = self.enable_grad and (y_pred.requires_grad or y_true.requires_grad);
    const loss = try self.tensor(1, 1, req_grad);
    return self.runAndRecordPreallocatedOp(
        loss,
        &.{ y_pred, y_true },
        .MseLoss,
        .{ .MseLoss = {} },
        req_grad,
    );
}

pub fn bceWithLogitsLoss(self: *Graph, logits: *Tensor, targets: *Tensor) !*Tensor {
    const req_grad = self.enable_grad and (logits.requires_grad or targets.requires_grad);
    const loss = try self.tensor(1, 1, req_grad);
    return self.runAndRecordPreallocatedOp(
        loss,
        &.{ logits, targets },
        .BceWithLogitsLoss,
        .{ .BceWithLogitsLoss = {} },
        req_grad,
    );
}

pub fn sigmoidCrossEntropy(self: *Graph, logits: *Tensor, targets: *Tensor) !*Tensor {
    return self.bceWithLogitsLoss(logits, targets);
}

pub fn bceLoss(self: *Graph, probs: *Tensor, targets: *Tensor, eps: f32) !*Tensor {
    const req_grad = self.enable_grad and (probs.requires_grad or targets.requires_grad);
    const loss = try self.tensor(1, 1, req_grad);
    return self.runAndRecordPreallocatedOp(
        loss,
        &.{ probs, targets },
        .BceLoss,
        .{ .BceLoss = .{ .eps = eps } },
        req_grad,
    );
}

pub fn randomNormal(self: *Graph, shape_slice: []const usize, random: std.Random, mean_val: f32, stddev: f32, requires_grad: bool) !*Tensor {
    const t = try self.tensorND(shape_slice, requires_grad);
    t.fillNormal(random, mean_val, stddev);
    return t;
}

pub fn randomUniform(self: *Graph, shape_slice: []const usize, random: std.Random, min: f32, max: f32, requires_grad: bool) !*Tensor {
    const t = try self.tensorND(shape_slice, requires_grad);
    t.fillUniform(random, min, max);
    return t;
}

// 二阶范数 (L2 Norm) 正则化损失函数：C = 0.5 * lambda * sum(weight_i^2)
pub fn l2Loss(self: *Graph, weight: *Tensor, lambda: f32) !*Tensor {
    const req_grad = self.enable_grad and weight.requires_grad;
    const loss = try self.tensor(1, 1, req_grad);
    return self.runAndRecordPreallocatedOp(
        loss,
        &.{weight},
        .L2Loss,
        .{ .L2Loss = .{ .lambda = lambda } },
        req_grad,
    );
}

// 岭回归 (Ridge) 组合损失函数：Loss = MSE(y_pred, y_true) + 0.5 * lambda * sum(weight_i^2)
pub fn ridgeLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor, weight: *Tensor, lambda: f32) !*Tensor {
    const mse = try self.mseLoss(y_pred, y_true);
    if (lambda == 0.0) return mse;
    const l2 = try self.l2Loss(weight, lambda);
    return try self.add(mse, l2);
}

// 一阶范数 (L1 Norm) 正则化损失：Loss = lambda * sum(|weight_i|)
pub fn l1Loss(self: *Graph, weight: *Tensor, lambda: f32) !*Tensor {
    const req_grad = self.enable_grad and weight.requires_grad;
    const loss = try self.tensor(1, 1, req_grad);
    return self.runAndRecordPreallocatedOp(
        loss,
        &.{weight},
        .L1Loss,
        .{ .L1Loss = .{ .lambda = lambda } },
        req_grad,
    );
}

// 最小绝对收缩和选择算子 (Least Absolute Shrinkage and Selection Operator, LASSO) 组合损失函数：Loss = MSE(y_pred, y_true) + lambda * sum(|weight_i|)
pub fn lassoLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor, weight: *Tensor, lambda: f32) !*Tensor {
    const mse = try self.mseLoss(y_pred, y_true);
    if (lambda == 0.0) return mse;
    const l1 = try self.l1Loss(weight, lambda);
    return try self.add(mse, l1);
}

// 弹性网络 (Elastic Net) 组合损失函数：Loss = MSE(y_pred, y_true) + lambda * rho * ||w||_1 + 0.5 * lambda * (1 - rho) * ||w||_2^2
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

pub fn conv1d(
    self: *Graph,
    A: *Tensor,
    weight: *Tensor,
    bias: ?*Tensor,
    options: tensor_mod.ConvOptions,
) !*Tensor {
    const allocator = self.arena.allocator();
    const C = try A.conv1d(weight, bias, options, allocator);
    const req_grad = self.enable_grad and (A.requires_grad or weight.requires_grad or (bias != null and bias.?.requires_grad));
    const ctx: OpContext = .{ .Conv1D = .{ .stride = options.stride, .padding = options.padding } };
    if (bias) |b| {
        return self.registerSingleOutputOp(C, &.{ A, weight, b }, .Conv1D, ctx, req_grad);
    } else {
        return self.registerSingleOutputOp(C, &.{ A, weight }, .Conv1D, ctx, req_grad);
    }
}

pub fn conv2d(
    self: *Graph,
    A: *Tensor,
    weight: *Tensor,
    bias: ?*Tensor,
    options: tensor_mod.ConvOptions,
) !*Tensor {
    const allocator = self.arena.allocator();
    const C = try A.conv2d(weight, bias, options, allocator);
    const req_grad = self.enable_grad and (A.requires_grad or weight.requires_grad or (bias != null and bias.?.requires_grad));
    const ctx: OpContext = .{ .Conv2D = .{ .stride = options.stride, .padding = options.padding } };
    if (bias) |b| {
        return self.registerSingleOutputOp(C, &.{ A, weight, b }, .Conv2D, ctx, req_grad);
    } else {
        return self.registerSingleOutputOp(C, &.{ A, weight }, .Conv2D, ctx, req_grad);
    }
}

pub fn convTranspose1d(
    self: *Graph,
    A: *Tensor,
    weight: *Tensor,
    bias: ?*Tensor,
    options: tensor_mod.ConvOptions,
) !*Tensor {
    const allocator = self.arena.allocator();
    const C = try A.convTranspose1d(weight, bias, options, allocator);
    const req_grad = self.enable_grad and (A.requires_grad or weight.requires_grad or (bias != null and bias.?.requires_grad));
    const ctx: OpContext = .{ .ConvTranspose1D = .{ .stride = options.stride, .padding = options.padding } };
    if (bias) |b| {
        return self.registerSingleOutputOp(C, &.{ A, weight, b }, .ConvTranspose1D, ctx, req_grad);
    } else {
        return self.registerSingleOutputOp(C, &.{ A, weight }, .ConvTranspose1D, ctx, req_grad);
    }
}

pub fn convTranspose2d(
    self: *Graph,
    A: *Tensor,
    weight: *Tensor,
    bias: ?*Tensor,
    options: tensor_mod.ConvOptions,
) !*Tensor {
    const allocator = self.arena.allocator();
    const C = try A.convTranspose2d(weight, bias, options, allocator);
    const req_grad = self.enable_grad and (A.requires_grad or weight.requires_grad or (bias != null and bias.?.requires_grad));
    const ctx: OpContext = .{ .ConvTranspose2D = .{ .stride = options.stride, .padding = options.padding } };
    if (bias) |b| {
        return self.registerSingleOutputOp(C, &.{ A, weight, b }, .ConvTranspose2D, ctx, req_grad);
    } else {
        return self.registerSingleOutputOp(C, &.{ A, weight }, .ConvTranspose2D, ctx, req_grad);
    }
}

pub const convTranspose2D = convTranspose2d;

pub fn maxpool1d(self: *Graph, A: *Tensor, pool_size: usize, options: tensor_mod.PoolOptions) !*Tensor {
    const allocator = self.arena.allocator();
    const C = try A.maxpool1d(pool_size, options, allocator);
    return self.registerSingleOutputOp(
        C,
        &.{A},
        .MaxPool1D,
        .{ .MaxPool1D = .{
            .pool_size = pool_size,
            .stride = options.resolveStride(pool_size),
            .padding = options.padding,
        } },
        self.enable_grad and A.requires_grad,
    );
}

pub fn maxpool2d(self: *Graph, A: *Tensor, pool_size: usize, options: tensor_mod.PoolOptions) !*Tensor {
    const allocator = self.arena.allocator();
    const C = try A.maxpool2d(pool_size, options, allocator);
    return self.registerSingleOutputOp(
        C,
        &.{A},
        .MaxPool2D,
        .{ .MaxPool2D = .{
            .pool_size = pool_size,
            .stride = options.resolveStride(pool_size),
            .padding = options.padding,
        } },
        self.enable_grad and A.requires_grad,
    );
}

pub fn avgpool1d(self: *Graph, A: *Tensor, kernel_size: usize, options: tensor_mod.PoolOptions) !*Tensor {
    const allocator = self.arena.allocator();
    const C = try A.avgpool1d(kernel_size, options, allocator);
    return self.registerSingleOutputOp(
        C,
        &.{A},
        .AvgPool1D,
        .{ .AvgPool1D = .{
            .kernel_size = kernel_size,
            .stride = options.resolveStride(kernel_size),
            .padding = options.padding,
        } },
        self.enable_grad and A.requires_grad,
    );
}

pub fn avgpool2d(self: *Graph, A: *Tensor, kernel_size: usize, options: tensor_mod.PoolOptions) !*Tensor {
    const allocator = self.arena.allocator();
    const C = try A.avgpool2d(kernel_size, options, allocator);
    return self.registerSingleOutputOp(
        C,
        &.{A},
        .AvgPool2D,
        .{ .AvgPool2D = .{
            .kernel_size = kernel_size,
            .stride = options.resolveStride(kernel_size),
            .padding = options.padding,
        } },
        self.enable_grad and A.requires_grad,
    );
}

pub fn softmax(self: *Graph, A: *Tensor) !*Tensor {
    const allocator = self.arena.allocator();
    const C = try A.softmax(allocator);
    return self.registerSingleOutputOp(
        C,
        &.{A},
        .Softmax,
        .{ .Softmax = {} },
        self.enable_grad and A.requires_grad,
    );
}

pub fn rmsNorm(self: *Graph, X: *Tensor, G: *Tensor, eps: f32) !*Tensor {
    const allocator = self.arena.allocator();
    const Y = try X.rmsNorm(G, eps, allocator);
    return self.registerSingleOutputOp(
        Y,
        &.{ X, G },
        .RmsNorm,
        .{ .RmsNorm = .{ .eps = eps } },
        self.enable_grad and (X.requires_grad or G.requires_grad),
    );
}

pub fn layerNorm(self: *Graph, X: *Tensor, G: *Tensor, B: *Tensor, eps: f32) !*Tensor {
    const allocator = self.arena.allocator();
    const Y = try X.layerNorm(G, B, eps, allocator);
    return self.registerSingleOutputOp(
        Y,
        &.{ X, G, B },
        .LayerNorm,
        .{ .LayerNorm = .{ .eps = eps } },
        self.enable_grad and (X.requires_grad or G.requires_grad or B.requires_grad),
    );
}

pub fn batchNorm2d(
    self: *Graph,
    X: *Tensor,
    G: *Tensor,
    B: *Tensor,
    running_mean: *Tensor,
    running_var: *Tensor,
    eps: f32,
    momentum: f32,
    training: bool,
) !*Tensor {
    if (X.shape.len != 4) return error.IncompatibleDimensions;
    const N = X.shape.dims[0];
    const C = X.shape.dims[1];
    const H = X.shape.dims[2];
    const W = X.shape.dims[3];
    if (G.data.len != C or B.data.len != C or running_mean.data.len != C or running_var.data.len != C) {
        return error.ShapeMismatch;
    }

    const allocator = self.arena.allocator();
    const req_grad = self.enable_grad and (X.requires_grad or G.requires_grad or B.requires_grad);
    const Y = try self.tensorND(&.{ N, C, H, W }, req_grad);

    const save_mean = try allocator.alloc(f32, C);
    const save_inv_std = try allocator.alloc(f32, C);

    const spatial_size = H * W;
    const total_samples_f = @as(f32, @floatFromInt(N * spatial_size));

    for (0..C) |c| {
        var mean_val: f32 = 0.0;
        var var_val: f32 = 0.0;

        if (training) {
            var sum_val: f32 = 0.0;
            for (0..N) |n| {
                const c_slice = X.data[(n * C + c) * spatial_size .. (n * C + c + 1) * spatial_size];
                for (c_slice) |val| sum_val += val;
            }
            mean_val = sum_val / total_samples_f;

            var var_sum: f32 = 0.0;
            for (0..N) |n| {
                const c_slice = X.data[(n * C + c) * spatial_size .. (n * C + c + 1) * spatial_size];
                for (c_slice) |val| {
                    const diff = val - mean_val;
                    var_sum += diff * diff;
                }
            }
            var_val = var_sum / total_samples_f;

            running_mean.data[c] = (1.0 - momentum) * running_mean.data[c] + momentum * mean_val;
            running_var.data[c] = (1.0 - momentum) * running_var.data[c] + momentum * var_val;
        } else {
            mean_val = running_mean.data[c];
            var_val = running_var.data[c];
        }

        const inv_std = 1.0 / @sqrt(var_val + eps);
        save_mean[c] = mean_val;
        save_inv_std[c] = inv_std;

        const g_val = G.data[c];
        const b_val = B.data[c];

        for (0..N) |n| {
            const in_slice = X.data[(n * C + c) * spatial_size .. (n * C + c + 1) * spatial_size];
            const out_slice = Y.data[(n * C + c) * spatial_size .. (n * C + c + 1) * spatial_size];
            for (in_slice, out_slice) |val, *o| {
                o.* = (val - mean_val) * inv_std * g_val + b_val;
            }
        }
    }

    if (self.enable_grad) {
        const inputs = try allocator.alloc(*Tensor, 3);
        inputs[0] = X;
        inputs[1] = G;
        inputs[2] = B;
        const outputs = try allocator.alloc(*Tensor, 1);
        outputs[0] = Y;

        const o = try allocator.create(Op);
        o.* = Op{
            .op_type = .BatchNorm2d,
            .inputs = inputs,
            .outputs = outputs,
            .context = .{ .BatchNorm2d = .{
                .eps = eps,
                .training = training,
                .save_mean = save_mean,
                .save_inv_std = save_inv_std,
            } },
        };
        Y.creator = o;
        try self.recordOp(o);
    }

    return Y;
}

pub fn dropout(self: *Graph, X: *Tensor, p: f32, random: std.Random) !*Tensor {
    const allocator = self.arena.allocator();
    const req_grad = self.enable_grad and X.requires_grad;
    const Y = try self.tensorND(X.shape.dims[0..X.shape.len], req_grad);

    const mask_scale = try allocator.alloc(f32, X.data.len);
    const scale = 1.0 / (1.0 - p);

    for (X.data, Y.data, mask_scale) |val, *o, *m| {
        if (random.float(f32) < p) {
            m.* = 0.0;
            o.* = 0.0;
        } else {
            m.* = scale;
            o.* = val * scale;
        }
    }

    if (self.enable_grad) {
        const inputs = try allocator.alloc(*Tensor, 1);
        inputs[0] = X;
        const outputs = try allocator.alloc(*Tensor, 1);
        outputs[0] = Y;

        const o = try allocator.create(Op);
        o.* = Op{
            .op_type = .Dropout,
            .inputs = inputs,
            .outputs = outputs,
            .context = .{ .Dropout = .{ .mask_scale = mask_scale } },
        };
        Y.creator = o;
        try self.recordOp(o);
    }

    return Y;
}

pub fn rope(self: *Graph, X: *Tensor, start_pos: usize) !*Tensor {
    return self.ropeOffset(X, start_pos, 0);
}

pub fn ropeOffset(self: *Graph, X: *Tensor, start_pos: usize, rotary_offset: usize) !*Tensor {
    const allocator = self.arena.allocator();
    const Y = try X.ropeOffset(start_pos, rotary_offset, allocator);
    return self.registerSingleOutputOp(
        Y,
        &.{X},
        .RoPE,
        .{ .RoPE = .{ .start_pos = start_pos, .rotary_offset = rotary_offset } },
        self.enable_grad and X.requires_grad,
    );
}

pub fn batchMatMul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
    const allocator = self.arena.allocator();
    const C = try A.batchMatMul(B, allocator);
    return self.registerSingleOutputOp(
        C,
        &.{ A, B },
        .BatchMatMul,
        .{ .BatchMatMul = {} },
        self.enable_grad and (A.requires_grad or B.requires_grad),
    );
}

pub fn embedding(self: *Graph, W: *Tensor, X: anytype) !*Tensor {
    const allocator = self.arena.allocator();
    const XT = @TypeOf(X);
    const x_tensor: *Tensor = if (XT == *Tensor or XT == *const Tensor)
        @constCast(X)
    else blk: {
        const ptr_info = @typeInfo(XT);
        if (ptr_info == .pointer and ptr_info.pointer.size == .one and
            @typeInfo(ptr_info.pointer.child) == .@"struct" and
            @hasField(ptr_info.pointer.child, "shape"))
        {
            const t = try self.tensorND(X.shape.dims[0..X.shape.len], false);
            const num_elem = X.shape.numel();
            if (X.isContiguous() and X.data.len >= num_elem) {
                for (0..num_elem) |i| {
                    t.data[i] = tensor_mod.convertScalar(f32, @TypeOf(X.data[0]), X.data[i]);
                }
            } else {
                var coord = [_]usize{0} ** 8;
                const rank = X.shape.len;
                for (0..num_elem) |i| {
                    var flat: usize = 0;
                    for (0..rank) |d| flat += coord[d] * X.strides.dims[d];
                    t.data[i] = tensor_mod.convertScalar(f32, @TypeOf(X.data[0]), X.data[flat]);
                    var d = rank;
                    while (d > 0) {
                        d -= 1;
                        coord[d] += 1;
                        if (coord[d] < X.shape.dims[d]) break;
                        coord[d] = 0;
                    }
                }
            }
            break :blk t;
        } else {
            const t = try self.tensorND(&.{X.len}, false);
            for (0..X.len) |i| {
                t.data[i] = tensor_mod.convertScalar(f32, @TypeOf(X[0]), X[i]);
            }
            break :blk t;
        }
    };

    const Y = try W.embedding(x_tensor, allocator);
    return self.registerSingleOutputOp(
        Y,
        &.{ W, x_tensor },
        .Embedding,
        .{ .Embedding = {} },
        self.enable_grad and W.requires_grad,
    );
}
