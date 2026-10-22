const std = @import("std");
const backend = @import("../backend.zig");
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const types = @import("types.zig");
const OpType = types.OpType;
const OpContext = types.OpContext;
const Op = types.Op;
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

pub fn adaptiveAvgPool1d(self: *Graph, A: *Tensor, output_size: usize) !*Tensor {
    const allocator = self.arena.allocator();
    const C = try A.adaptiveAvgPool1d(output_size, allocator);
    return self.registerSingleOutputOp(
        C,
        &.{A},
        .AdaptiveAvgPool1D,
        .{ .AdaptiveAvgPool1D = .{ .output_size = output_size } },
        self.enable_grad and A.requires_grad,
    );
}

pub fn adaptiveAvgPool2d(self: *Graph, A: *Tensor, output_size: [2]usize) !*Tensor {
    const allocator = self.arena.allocator();
    const C = try A.adaptiveAvgPool2d(output_size, allocator);
    return self.registerSingleOutputOp(
        C,
        &.{A},
        .AdaptiveAvgPool2D,
        .{ .AdaptiveAvgPool2D = .{ .output_size = output_size } },
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
    if (self.options.device == .gpu and backend.offload.isAvailable() and !self.enable_grad) {
        const Y = try self.tensorND(X.shape.dims[0..X.shape.len], false);
        try backend.offload.rmsNorm(X, G, Y, eps, self.options.mlx_stream);
        return Y;
    }

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

/// 一维批量归一化节点 (1-Dimensional Batch Normalization, BatchNorm1d)
/// 支持二维 `[N, C]` 与三维 `[N, C, L]` 输入张量
pub fn batchNorm1d(
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
    if (X.shape.len != 2 and X.shape.len != 3) return error.IncompatibleDimensions;
    const C = X.shape.dims[1];
    if (G.data.len != C or B.data.len != C or running_mean.data.len != C or running_var.data.len != C) {
        return error.ShapeMismatch;
    }

    const allocator = self.arena.allocator();
    const save_mean = try allocator.alloc(f32, C);
    const save_inv_std = try allocator.alloc(f32, C);

    const Y = try X.batchNorm1d(
        G,
        B,
        running_mean,
        running_var,
        training,
        eps,
        momentum,
        save_mean,
        save_inv_std,
        allocator,
    );
    const req_grad = self.enable_grad and (X.requires_grad or G.requires_grad or B.requires_grad);
    return self.registerSingleOutputOp(
        Y,
        &.{ X, G, B },
        .BatchNorm1d,
        .{ .BatchNorm1d = .{
            .eps = eps,
            .training = training,
            .save_mean = save_mean,
            .save_inv_std = save_inv_std,
        } },
        req_grad,
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
    const C = X.shape.dims[1];
    if (G.data.len != C or B.data.len != C or running_mean.data.len != C or running_var.data.len != C) {
        return error.ShapeMismatch;
    }

    const allocator = self.arena.allocator();
    const save_mean = try allocator.alloc(f32, C);
    const save_inv_std = try allocator.alloc(f32, C);

    const Y = try X.batchNorm2d(
        G,
        B,
        running_mean,
        running_var,
        training,
        eps,
        momentum,
        save_mean,
        save_inv_std,
        allocator,
    );
    const req_grad = self.enable_grad and (X.requires_grad or G.requires_grad or B.requires_grad);
    return self.registerSingleOutputOp(
        Y,
        &.{ X, G, B },
        .BatchNorm2d,
        .{ .BatchNorm2d = .{
            .eps = eps,
            .training = training,
            .save_mean = save_mean,
            .save_inv_std = save_inv_std,
        } },
        req_grad,
    );
}

/// 分组归一化节点 (Group Normalization, GroupNorm)
/// 支持任意维度 `>= 2` 的输入张量 `[N, C, ...]`，要求 `C % num_groups == 0`
pub fn groupNorm(
    self: *Graph,
    X: *Tensor,
    G: *Tensor,
    B: *Tensor,
    num_groups: usize,
    eps: f32,
) !*Tensor {
    if (X.shape.len < 2) return error.IncompatibleDimensions;
    const N = X.shape.dims[0];
    const C = X.shape.dims[1];
    if (num_groups == 0 or C == 0 or C % num_groups != 0) return error.ShapeMismatch;
    if (G.data.len != C or B.data.len != C) return error.ShapeMismatch;

    const allocator = self.arena.allocator();
    const save_mean = try allocator.alloc(f32, N * num_groups);
    const save_inv_std = try allocator.alloc(f32, N * num_groups);

    const Y = try X.groupNorm(
        G,
        B,
        num_groups,
        eps,
        save_mean,
        save_inv_std,
        allocator,
    );
    const req_grad = self.enable_grad and (X.requires_grad or G.requires_grad or B.requires_grad);
    return self.registerSingleOutputOp(
        Y,
        &.{ X, G, B },
        .GroupNorm,
        .{ .GroupNorm = .{
            .num_groups = num_groups,
            .eps = eps,
            .save_mean = save_mean,
            .save_inv_std = save_inv_std,
        } },
        req_grad,
    );
}

pub fn dropout(self: *Graph, X: *Tensor, p: f32, random: std.Random) !*Tensor {
    const allocator = self.arena.allocator();
    const mask_scale = try allocator.alloc(f32, X.numel());
    const scale = 1.0 / (1.0 - p);

    for (mask_scale) |*m| {
        if (random.float(f32) < p) {
            m.* = 0.0;
        } else {
            m.* = scale;
        }
    }

    const Y = try X.applyDropoutMask(mask_scale, allocator);
    return self.registerSingleOutputOp(
        Y,
        &.{X},
        .Dropout,
        .{ .Dropout = .{ .mask_scale = mask_scale } },
        self.enable_grad and X.requires_grad,
    );
}

pub fn rope(self: *Graph, X: *Tensor, start_pos: usize, options: tensor_mod.RopeOptions) !*Tensor {
    if (self.options.device == .gpu and backend.offload.isAvailable() and !self.enable_grad and options.rotary_offset == 0 and options.partial_rotary_factor == 1.0) {
        const D = X.shape.dims[X.shape.len - 1];
        if (D % 2 == 0) {
            const traditional = (options.mode == .interleaved);
            const Y = try self.tensorND(X.shape.dims[0..X.shape.len], false);
            try backend.offload.rope(X, Y, D, traditional, options.base, 1.0, start_pos, self.options.mlx_stream);
            return Y;
        }
    }

    const allocator = self.arena.allocator();
    const Y = try X.rope(start_pos, options, allocator);
    return self.registerSingleOutputOp(
        Y,
        &.{X},
        .RoPE,
        .{ .RoPE = .{
            .start_pos = start_pos,
            .rotary_offset = options.rotary_offset,
            .base = options.base,
            .mode = options.mode,
            .partial_rotary_factor = options.partial_rotary_factor,
        } },
        self.enable_grad and X.requires_grad,
    );
}

pub fn batchMatMul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
    const allocator = self.arena.allocator();
    if (self.options.device == .gpu and backend.offload.isAvailable() and !self.enable_grad) {
        const ndim = A.shape.len;
        var out_shape: [8]usize = undefined;
        for (0..ndim - 2) |i| {
            out_shape[i] = A.shape.dims[i];
        }
        out_shape[ndim - 2] = A.shape.dims[ndim - 2];
        out_shape[ndim - 1] = B.shape.dims[B.shape.len - 1];

        const C = try self.tensorND(out_shape[0..ndim], false);
        try backend.offload.matmul(A, B, C, self.options.mlx_stream);
        return C;
    }

    const C = try A.batchMatMul(B, allocator);
    return self.registerSingleOutputOp(
        C,
        &.{ A, B },
        .BatchMatMul,
        .{ .BatchMatMul = {} },
        self.enable_grad and (A.requires_grad or B.requires_grad),
    );
}

/// 缩放点积注意力 (Scaled Dot-Product Attention, SDPA) 算子：
/// 在开启 Apple Silicon GPU 硬件加速且无需求导时，卸载至 MLX FlashAttention Metal 融合内核执行；
/// 否则回退至通用多维计算图分解求值（支持端到端自动微分 Autograd）。
pub fn scaledDotProductAttention(
    self: *Graph,
    q: *Tensor,
    k: *Tensor,
    v: *Tensor,
    scale: f32,
    is_causal: bool,
) !*Tensor {
    if (self.options.device == .gpu and backend.offload.isAvailable() and !self.enable_grad) {
        const B = q.shape.dims[0];
        const H = q.shape.dims[1];
        const T = q.shape.dims[2];
        const d_v = v.shape.dims[3];
        const out_tensor = try self.tensorND(&.{ B, H, T, d_v }, false);
        try backend.offload.scaledDotProductAttention(q, k, v, out_tensor, scale, is_causal, self.options.mlx_stream);
        return out_tensor;
    }

    // 通用计算图分解路径
    const k_t = try self.transposeND(k, 2, 3);
    const att = try self.batchMatMul(q, k_t);
    var scores = try self.mulScalar(att, scale);
    const T = q.shape.dims[2];
    if (is_causal) {
        const mask_node = try self.tensorND(&.{ 1, 1, T, T }, false);
        mask_node.is_buffer = true;
        @memset(mask_node.data, 0.0);
        for (0..T) |i| {
            for (i + 1..T) |j| {
                mask_node.data[i * T + j] = -1e9;
            }
        }
        const scope = self.currentScope();
        if (scope.len > 0) {
            mask_node.name = try std.fmt.allocPrint(self.arenaAllocator(), "{s}.causal_mask", .{scope});
        }
        scores = try self.add(scores, mask_node);
    }
    const att_sm = try self.softmax(scores);
    return try self.batchMatMul(att_sm, v);
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
                    X.shape.incrementCoord(&coord);
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

/// 4-bit 量化矩阵乘法 (4-bit Quantized Matrix Multiplication) 前向：经计算图注入的执行器 (`GraphOptions.io`) 多核并行，
/// 仅用于推理（反向传播返回 `error.UnsupportedBackward`）
pub fn q4MatMul(
    self: *Graph,
    x: *Tensor,
    blocks: []const tensor_mod.quant.Q4Block,
    in_features: usize,
    out_features: usize,
) !*Tensor {
    const allocator = self.arena.allocator();
    const Y = try tensor_mod.quant.q4MatMul(self.executor(), x, blocks, in_features, out_features, allocator);
    const ctx: OpContext = .{
        .Q4MatMul = .{
            .blocks = blocks,
            .in_features = in_features,
            .out_features = out_features,
        },
    };
    return self.registerSingleOutputOp(
        Y,
        &.{x},
        .Q4MatMul,
        ctx,
        false,
    );
}

/// 无分支前 K 大元素与下标选择算子 (Branchless Top-K Selection Operator, TopK)：
/// 沿输入张量最后一维选出前 `k` 大元素 `values` 与对应专家下标 `indices`，
/// 全程无数据依赖 `if` 跳转，反向传播将 `values.grad` 按 `indices` 路由回原位置。
pub fn topK(self: *Graph, X: *Tensor, k: usize) !Tensor.TopKResult {
    const allocator = self.arena.allocator();
    const res = try X.topK(k, allocator);
    const req_grad = self.enable_grad and X.requires_grad;

    res.values.scope = self.currentScope();
    res.values.requires_grad = req_grad;
    if (req_grad) {
        res.values.grad = try allocator.alloc(f32, res.values.data.len);
        @memset(res.values.grad, 0.0);
    }
    try self.tensors.append(self.backing_allocator, res.values);

    res.indices.scope = self.currentScope();
    res.indices.requires_grad = false;
    try self.tensors.append(self.backing_allocator, res.indices);

    if (self.enable_grad) {
        const inps = try allocator.alloc(*Tensor, 1);
        inps[0] = X;
        const outs = try allocator.alloc(*Tensor, 2);
        outs[0] = res.values;
        outs[1] = res.indices;

        const o = try allocator.create(Op);
        o.* = Op{
            .op_type = .TopK,
            .inputs = inps,
            .outputs = outs,
            .context = .{ .TopK = .{ .k = k } },
        };
        res.values.creator = o.handle();
        res.indices.creator = o.handle();
        try self.recordOp(o);
    }

    return res;
}

/// 索引收集专家矩阵乘法算子 (Indexed Gather Matrix Multiplication Operator, GatherMatMul)：
/// 根据 `indices` (`[N, K]`) 对每个词元选中的 `K` 个专家权重 `weights[e]`（及可选偏置 `biases[e]`）
/// 直接寻址并执行矩阵乘法，仅激活 `K` 个被选中专家且无任何主机端或数据依赖 `if` 分支。
pub fn gatherMatMul(
    self: *Graph,
    X: *Tensor,
    indices: *Tensor,
    weights: []const *Tensor,
    biases: ?[]const *Tensor,
) !*Tensor {
    const allocator = self.arena.allocator();
    const E = weights.len;
    const has_bias = biases != null;
    const Y = try X.gatherMatMul(indices, weights, biases, allocator);

    var req_grad = false;
    if (self.enable_grad) {
        if (X.requires_grad) req_grad = true;
        for (weights) |w| {
            if (w.requires_grad) req_grad = true;
        }
        if (biases) |bs| {
            for (bs) |b| {
                if (b.requires_grad) req_grad = true;
            }
        }
    }

    const total_inputs = 2 + E + (if (has_bias) E else 0);
    const inps = try allocator.alloc(*Tensor, total_inputs);
    inps[0] = X;
    inps[1] = indices;
    for (weights, 0..) |w, i| {
        inps[2 + i] = w;
    }
    if (biases) |bs| {
        for (bs, 0..) |b, i| {
            inps[2 + E + i] = b;
        }
    }

    return self.registerSingleOutputOp(
        Y,
        inps,
        .GatherMatMul,
        .{
            .GatherMatMul = .{
                .num_experts = E,
                .has_bias = has_bias,
            },
        },
        req_grad,
    );
}
