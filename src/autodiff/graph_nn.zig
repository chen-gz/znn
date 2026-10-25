const std = @import("std");
const backend = @import("../backend.zig");
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const types = @import("types.zig");
const OpType = types.OpType;
const OpContext = types.OpContext;
const Op = types.Op;
const Graph = @import("graph.zig").Graph;

/// 交叉熵类损失要求 logits 为连续的二维张量 `[B, num_classes]` 且类别数非零 (三维 logits 需先 reshape 为 `[B * T, V]`)
fn validateClassLogits(logits: *const Tensor) !void {
    if (logits.shape.len != 2) return error.IncompatibleDimensions;
    if (logits.shape.dims[1] == 0) return error.InvalidDimension;
    if (!logits.isContiguous()) return error.NonContiguousTensor;
}

// 损失函数 Softmax + Cross Entropy 结合前向传播
// 在 logits 的行维度计算 Softmax 概率分布，并与 targets 分类标签（支持 u8/u32/usize 等任意整型切片）计算交叉熵损失
pub fn softmaxCrossEntropy(self: *Graph, logits: *Tensor, targets: anytype) !*Tensor {
    try validateClassLogits(logits);
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
    );
}

// 监督微调 (Supervised Fine-Tuning, SFT) 掩码交叉熵损失：仅对 mask[i] > 0 的位置计算交叉熵并支持自动微分 (Automatic Differentiation, Autograd) 反向传播
pub fn maskedCrossEntropyLoss(self: *Graph, logits: *Tensor, targets: anytype, mask: []const f32) !*Tensor {
    try validateClassLogits(logits);
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
    );
}

/// 校验直接偏好优化 (Direct Preference Optimization, DPO) 的温度系数 beta：必须是有限正数
pub fn validateDpoBeta(beta: f32) error{InvalidArgument}!void {
    if (!std.math.isFinite(beta) or beta <= 0.0) return error.InvalidArgument;
}

/// 校验群组相对策略优化 (Group Relative Policy Optimization, GRPO) 的超参数：
/// 库尔贝克-莱布勒散度 (Kullback-Leibler Divergence, KL Divergence) 惩罚系数 beta 与裁剪半径 clip_eps 都必须是有限非负数 (负的 clip_eps 会使裁剪区间 [1 - eps, 1 + eps] 为空)
pub fn validateGrpoHyperparams(beta: f32, clip_eps: f32) error{InvalidArgument}!void {
    if (!std.math.isFinite(beta) or beta < 0.0) return error.InvalidArgument;
    if (!std.math.isFinite(clip_eps) or clip_eps < 0.0) return error.InvalidArgument;
}

// 直接偏好优化 (Direct Preference Optimization, DPO) 损失函数：支持对策略模型对数概率 pi_chosen_logps / pi_rejected_logps 的计算图反向传播
pub fn dpoLoss(
    self: *Graph,
    pi_chosen_in: *Tensor,
    pi_rejected_in: *Tensor,
    ref_chosen_logps: []const f32,
    ref_rejected_logps: []const f32,
    beta: f32,
) !*Tensor {
    const pi_chosen_logps = try self.contiguous(pi_chosen_in);
    const pi_rejected_logps = try self.contiguous(pi_rejected_in);
    const N = pi_chosen_logps.data.len;
    if (pi_rejected_logps.data.len != N or ref_chosen_logps.len != N or ref_rejected_logps.len != N) {
        return error.ShapeMismatch;
    }
    try validateDpoBeta(beta);

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
    );
}

// 群组相对策略优化 (Group Relative Policy Optimization, GRPO) 损失函数：支持在计算图中对 new_logps 自动微分求导
pub fn grpoLoss(
    self: *Graph,
    old_logps_in: *Tensor,
    new_logps_in: *Tensor,
    advantages: []const f32,
    ref_logps: ?[]const f32,
    beta: f32,
    clip_eps: f32,
) !*Tensor {
    const old_logps = try self.contiguous(old_logps_in);
    const new_logps = try self.contiguous(new_logps_in);
    const N = old_logps.data.len;
    if (new_logps.data.len != N or advantages.len != N) return error.ShapeMismatch;
    if (ref_logps) |refs| {
        if (refs.len != N) return error.ShapeMismatch;
    }
    try validateGrpoHyperparams(beta, clip_eps);

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
    );
}

// 均方误差 (Mean Squared Error, MSE) 损失函数：C = 1/N * sum((y_pred - y_true)^2)
pub fn mseLoss(self: *Graph, y_pred_in: *Tensor, y_true_in: *Tensor) !*Tensor {
    const y_pred = try self.contiguous(y_pred_in);
    const y_true = try self.contiguous(y_true_in);
    const req_grad = self.enable_grad and (y_pred.requires_grad or y_true.requires_grad);
    const loss = try self.tensor(1, 1, req_grad);
    return self.runAndRecordPreallocatedOp(
        loss,
        &.{ y_pred, y_true },
        .MseLoss,
        .{ .MseLoss = {} },
    );
}

pub fn bceWithLogitsLoss(self: *Graph, logits_in: *Tensor, targets_in: *Tensor) !*Tensor {
    const logits = try self.contiguous(logits_in);
    const targets = try self.contiguous(targets_in);
    const req_grad = self.enable_grad and (logits.requires_grad or targets.requires_grad);
    const loss = try self.tensor(1, 1, req_grad);
    return self.runAndRecordPreallocatedOp(
        loss,
        &.{ logits, targets },
        .BceWithLogitsLoss,
        .{ .BceWithLogitsLoss = {} },
    );
}

pub fn sigmoidCrossEntropy(self: *Graph, logits: *Tensor, targets: *Tensor) !*Tensor {
    return self.bceWithLogitsLoss(logits, targets);
}

pub fn bceLoss(self: *Graph, probs_in: *Tensor, targets_in: *Tensor, eps: f32) !*Tensor {
    const probs = try self.contiguous(probs_in);
    const targets = try self.contiguous(targets_in);
    const req_grad = self.enable_grad and (probs.requires_grad or targets.requires_grad);
    const loss = try self.tensor(1, 1, req_grad);
    return self.runAndRecordPreallocatedOp(
        loss,
        &.{ probs, targets },
        .BceLoss,
        .{ .BceLoss = .{ .eps = eps } },
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
pub fn l2Loss(self: *Graph, weight_in: *Tensor, lambda: f32) !*Tensor {
    const weight = try self.contiguous(weight_in);
    const req_grad = self.enable_grad and weight.requires_grad;
    const loss = try self.tensor(1, 1, req_grad);
    return self.runAndRecordPreallocatedOp(
        loss,
        &.{weight},
        .L2Loss,
        .{ .L2Loss = .{ .lambda = lambda } },
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
pub fn l1Loss(self: *Graph, weight_in: *Tensor, lambda: f32) !*Tensor {
    const weight = try self.contiguous(weight_in);
    const req_grad = self.enable_grad and weight.requires_grad;
    const loss = try self.tensor(1, 1, req_grad);
    return self.runAndRecordPreallocatedOp(
        loss,
        &.{weight},
        .L1Loss,
        .{ .L1Loss = .{ .lambda = lambda } },
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
    A_in: *Tensor,
    weight_in: *Tensor,
    bias_in: ?*Tensor,
    options: tensor_mod.ConvOptions,
) !*Tensor {
    const allocator = self.arena.allocator();
    const A = try self.contiguous(A_in);
    const weight = try self.contiguous(weight_in);
    const bias: ?*Tensor = if (bias_in) |b| try self.contiguous(b) else null;
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
    A_in: *Tensor,
    weight_in: *Tensor,
    bias_in: ?*Tensor,
    options: tensor_mod.ConvOptions,
) !*Tensor {
    const allocator = self.arena.allocator();
    const A = try self.contiguous(A_in);
    const weight = try self.contiguous(weight_in);
    const bias: ?*Tensor = if (bias_in) |b| try self.contiguous(b) else null;
    const C = try A.convTranspose2d(weight, bias, options, allocator);
    const req_grad = self.enable_grad and (A.requires_grad or weight.requires_grad or (bias != null and bias.?.requires_grad));
    const ctx: OpContext = .{ .ConvTranspose2D = .{ .stride = options.stride, .padding = options.padding } };
    if (bias) |b| {
        return self.registerSingleOutputOp(C, &.{ A, weight, b }, .ConvTranspose2D, ctx, req_grad);
    } else {
        return self.registerSingleOutputOp(C, &.{ A, weight }, .ConvTranspose2D, ctx, req_grad);
    }
}

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

/// 沿最后一维的数值稳定归一化指数函数 (Softmax)；统一设备分派：GPU 计算图且输出不需要梯度时由 MLX 执行前向
pub fn softmax(self: *Graph, A_in: *Tensor) !*Tensor {
    const allocator = self.arena.allocator();
    const A = try self.contiguous(A_in);
    const req_grad = self.enable_grad and A.requires_grad;
    const C = if (self.useDevice(req_grad)) blk: {
        if (A.shape.len == 0) return error.IncompatibleDimensions;
        const out = try tensor_mod.zeros(allocator, A.shape.dims[0..A.shape.len]);
        try backend.offload.softmax(A, out, -1, self.options.mlx_stream);
        break :blk out;
    } else try A.softmax(allocator);
    return self.registerSingleOutputOp(C, &.{A}, .Softmax, .{ .Softmax = {} }, req_grad);
}

/// 均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)：y = x / sqrt(mean(x^2) + eps) * (w + weight_offset)，
/// 沿最后一维归一化；W 为 null 时不缩放 (无参数 RMSNorm)。需要梯度时在算子上下文中保存每行的均方根倒数，
/// 反向传播直接复用。统一设备分派：GPU 计算图且输出不需要梯度时由 MLX 快速 RMSNorm 执行 (weight_offset 为 0 时)。
pub fn rmsNorm(self: *Graph, X_in: *Tensor, W_in: ?*Tensor, options: tensor_mod.RmsNormOptions) !*Tensor {
    const allocator = self.arena.allocator();
    const X = try self.contiguous(X_in);
    const W: ?*Tensor = if (W_in) |w| try self.contiguous(w) else null;
    if (X.shape.len == 0) return error.IncompatibleDimensions;
    const D = X.shape.dims[X.shape.len - 1];
    if (D == 0) return error.DivisionByZero;
    const req_grad = self.enable_grad and (X.requires_grad or (W != null and W.?.requires_grad));
    const inv_rms: []f32 = if (req_grad) try allocator.alloc(f32, X.shape.numel() / D) else &.{};

    const Y = if (self.useDevice(req_grad) and (W == null or options.weight_offset == 0.0)) blk: {
        if (W) |w| if (w.shape.numel() != D) return error.ShapeMismatch;
        const out = try self.tensorND(X.shape.dims[0..X.shape.len], req_grad);
        try backend.offload.rmsNorm(X, W, out, options.eps, self.options.mlx_stream);
        break :blk out;
    } else try X.rmsNorm(W, options, if (req_grad) inv_rms else null, allocator);

    const ctx: OpContext = .{ .RmsNorm = .{ .options = options, .inv_rms = inv_rms } };
    if (W) |w| return self.registerSingleOutputOp(Y, &.{ X, w }, .RmsNorm, ctx, req_grad);
    return self.registerSingleOutputOp(Y, &.{X}, .RmsNorm, ctx, req_grad);
}

/// 层归一化 (Layer Normalization, LayerNorm)：需要梯度时在算子上下文中保存每行的均值与标准差倒数，反向传播直接复用
pub fn layerNorm(self: *Graph, X_in: *Tensor, G_in: *Tensor, B_in: *Tensor, eps: f32) !*Tensor {
    const allocator = self.arena.allocator();
    const X = try self.contiguous(X_in);
    const G = try self.contiguous(G_in);
    const B = try self.contiguous(B_in);
    if (X.shape.len == 0) return error.IncompatibleDimensions;
    const D = X.shape.dims[X.shape.len - 1];
    if (D == 0) return error.DivisionByZero;
    const req_grad = self.enable_grad and (X.requires_grad or G.requires_grad or B.requires_grad);
    const rows = X.shape.numel() / D;
    const save_mean: []f32 = if (req_grad) try allocator.alloc(f32, rows) else &.{};
    const save_inv_std: []f32 = if (req_grad) try allocator.alloc(f32, rows) else &.{};
    const Y = try X.layerNorm(G, B, eps, if (req_grad) save_mean else null, if (req_grad) save_inv_std else null, allocator);
    return self.registerSingleOutputOp(
        Y,
        &.{ X, G, B },
        .LayerNorm,
        .{ .LayerNorm = .{ .eps = eps, .save_mean = save_mean, .save_inv_std = save_inv_std } },
        req_grad,
    );
}

/// 一维批量归一化节点 (1-Dimensional Batch Normalization, BatchNorm1d)
/// 支持二维 `[N, C]` 与三维 `[N, C, L]` 输入张量
pub fn batchNorm1d(
    self: *Graph,
    X_in: *Tensor,
    G_in: *Tensor,
    B_in: *Tensor,
    running_mean: *Tensor,
    running_var: *Tensor,
    eps: f32,
    momentum: f32,
    training: bool,
) !*Tensor {
    const X = try self.contiguous(X_in);
    const G = try self.contiguous(G_in);
    const B = try self.contiguous(B_in);
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
            .momentum = momentum,
            .training = training,
            .running_mean = running_mean,
            .running_var = running_var,
            .save_mean = save_mean,
            .save_inv_std = save_inv_std,
        } },
        req_grad,
    );
}

pub fn batchNorm2d(
    self: *Graph,
    X_in: *Tensor,
    G_in: *Tensor,
    B_in: *Tensor,
    running_mean: *Tensor,
    running_var: *Tensor,
    eps: f32,
    momentum: f32,
    training: bool,
) !*Tensor {
    const X = try self.contiguous(X_in);
    const G = try self.contiguous(G_in);
    const B = try self.contiguous(B_in);
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
            .momentum = momentum,
            .training = training,
            .running_mean = running_mean,
            .running_var = running_var,
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
    X_in: *Tensor,
    G_in: *Tensor,
    B_in: *Tensor,
    num_groups: usize,
    eps: f32,
) !*Tensor {
    const X = try self.contiguous(X_in);
    const G = try self.contiguous(G_in);
    const B = try self.contiguous(B_in);
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

pub fn dropout(self: *Graph, X_in: *Tensor, p: f32, random: std.Random) !*Tensor {
    if (!(p >= 0.0 and p <= 1.0)) return error.InvalidProbability;
    const allocator = self.arena.allocator();
    const X = try self.contiguous(X_in);
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

/// 旋转位置编码 (Rotary Position Embedding, RoPE)：第 t 行的绝对位置为 start_pos + t。
/// 统一设备分派：GPU 计算图且输出不需要梯度时由 MLX 执行 (部分旋转 / 比例旋转经共享周期表，旋转子空间带偏移时回退 CPU)
pub fn rope(self: *Graph, X_in: *Tensor, start_pos: usize, options: tensor_mod.RopeOptions) !*Tensor {
    const allocator = self.arena.allocator();
    const X = try self.contiguous(X_in);
    if (X.shape.len == 0) return error.IncompatibleDimensions;
    const req_grad = self.enable_grad and X.requires_grad;
    const D = X.shape.dims[X.shape.len - 1];
    const Y = if (self.useDevice(req_grad) and backend.offload.ropeSupported(D, options)) blk: {
        const out = try self.tensorND(X.shape.dims[0..X.shape.len], req_grad);
        try backend.offload.rope(X, out, options, start_pos, self.options.mlx_stream);
        break :blk out;
    } else try X.rope(start_pos, options, allocator);
    return self.registerSingleOutputOp(
        Y,
        &.{X},
        .RoPE,
        .{ .RoPE = .{ .start_pos = start_pos, .options = options } },
        req_grad,
    );
}

/// 批量矩阵乘法 [B, H, M, K] x [B, H, K, N]；统一设备分派：GPU 计算图且输出不需要梯度时由 MLX 执行
pub fn batchMatMul(self: *Graph, A_in: *Tensor, B_in: *Tensor) !*Tensor {
    const allocator = self.arena.allocator();
    const A = try self.contiguous(A_in);
    const B = try self.contiguous(B_in);
    const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
    const C = if (self.useDevice(req_grad)) blk: {
        if (A.shape.len != 4 or B.shape.len != 4) return error.IncompatibleDimensions;
        if (A.shape.dims[0] != B.shape.dims[0] or A.shape.dims[1] != B.shape.dims[1] or A.shape.dims[3] != B.shape.dims[2]) return error.ShapeMismatch;
        const out = try self.tensorND(&.{ A.shape.dims[0], A.shape.dims[1], A.shape.dims[2], B.shape.dims[3] }, req_grad);
        try backend.offload.matmul(A, B, out, self.options.mlx_stream);
        break :blk out;
    } else try A.batchMatMul(B, allocator);
    return self.registerSingleOutputOp(C, &.{ A, B }, .BatchMatMul, .{ .BatchMatMul = {} }, req_grad);
}

/// 缩放点积注意力 (Scaled Dot-Product Attention, SDPA) 算子：O = softmax(scale * Q K^T + M) V。
/// q [B, q_heads, tq, d]、k [B, kv_heads, tk, d]、v [B, kv_heads, tk, dv]，q_heads = kv_heads * kv_groups：
/// 分组查询注意力 (Grouped-Query Attention, GQA) 直接按组共享键值，不物化复制后的键值；查询数与键数可以不同，
/// 第 i 个查询 / 第 j 个键的绝对位置为 q_pos0 + i / k_pos0 + j，掩码可见性由 `tensor.AttentionMask` 定义。
/// 需要梯度时保存注意力概率，反向传播 (`backward_nn`) 直接复用。
/// 统一设备分派：GPU 计算图且输出不需要梯度时由 MLX 融合缩放点积注意力执行 (掩码化简后使用内置因果模式或加性掩码)。
pub fn attention(self: *Graph, q_in: *Tensor, k_in: *Tensor, v_in: *Tensor, options: tensor_mod.AttentionOptions) !*Tensor {
    const allocator = self.arena.allocator();
    const q = try self.contiguous(q_in);
    const k = try self.contiguous(k_in);
    const v = try self.contiguous(v_in);
    const shape = try tensor_mod.attention_kernels.Shape.init(q, k, v, options);
    const req_grad = self.enable_grad and (q.requires_grad or k.requires_grad or v.requires_grad);

    const out = try self.tensorND(&.{ shape.batch, shape.q_heads, shape.tq, shape.v_dim }, req_grad);
    const probs: []f32 = if (req_grad) try allocator.alloc(f32, shape.probsLen()) else &.{};
    if (self.useDevice(req_grad)) {
        try backend.offload.attention(self.backing_allocator, q, k, v, out, options.resolveScale(shape.head_dim), options.mask, options.q_pos0, options.k_pos0, self.options.mlx_stream);
    } else {
        try tensor_mod.attention_kernels.attentionInto(q, k, v, out, if (req_grad) probs else null, options, self.backing_allocator);
    }
    return self.registerSingleOutputOp(
        out,
        &.{ q, k, v },
        .Attention,
        .{ .Attention = .{ .options = options, .probs = probs } },
        req_grad,
    );
}

pub fn embedding(self: *Graph, W_in: *Tensor, X: anytype) !*Tensor {
    const allocator = self.arena.allocator();
    const W = try self.contiguous(W_in);
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
                var coord: [8]usize = @splat(0);
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

    const req_grad = self.enable_grad and W.requires_grad;
    if (self.useDevice(req_grad) and (W.device == .gpu or W.device_array != null)) {
        var w_b = try backend.offload.borrowArray(W);
        defer w_b.deinit();
        var x_b = try backend.offload.borrowArray(x_tensor);
        defer x_b.deinit();
        var x_i32 = try x_b.arr.astype(.int32, self.options.mlx_stream);
        defer x_i32.deinit();
        const res = try w_b.arr.takeAxis(x_i32, 0, self.options.mlx_stream);
        var out_shape_dims: [8]usize = undefined;
        for (0..x_tensor.shape.len) |d| {
            out_shape_dims[d] = x_tensor.shape.dims[d];
        }
        out_shape_dims[x_tensor.shape.len] = W.shape.dims[1];
        const shape = tensor_mod.Shape.init(out_shape_dims[0 .. x_tensor.shape.len + 1]);
        const Y = try allocator.create(Tensor);
        Y.* = Tensor{
            .data = &.{},
            .grad = &.{},
            .shape = shape,
            .strides = tensor_mod.computeContiguousStrides(shape),
            .requires_grad = false,
            .creator = null,
            .device = .gpu,
            .device_array = res,
            .is_graph_owned = true,
            .scope = self.currentScope(),
        };
        try self.tensors.append(self.backing_allocator, Y);
        return self.registerSingleOutputOp(
            Y,
            &.{ W, x_tensor },
            .Embedding,
            .{ .Embedding = {} },
            false,
        );
    }

    const Y = try W.embedding(x_tensor, allocator);
    return self.registerSingleOutputOp(
        Y,
        &.{ W, x_tensor },
        .Embedding,
        .{ .Embedding = {} },
        req_grad,
    );
}

/// 4-bit 量化矩阵乘法 (4-bit Quantized Matrix Multiplication) 前向：经计算图注入的执行器 (`GraphOptions.io`) 多核并行；
/// 量化权重冻结，反向传播只对输入 `x` 求梯度 (可在量化骨干网络之上训练低秩自适应 (Low-Rank Adaptation, LoRA) 等可训练模块)
pub fn q4MatMul(
    self: *Graph,
    x_in: *Tensor,
    blocks: []const tensor_mod.quant.Q4Block,
    in_features: usize,
    out_features: usize,
) !*Tensor {
    const allocator = self.arena.allocator();
    const x = try self.contiguous(x_in);
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
        self.enable_grad and x.requires_grad,
    );
}

/// 无分支前 K 大元素与下标选择算子 (Branchless Top-K Selection Operator, TopK)：
/// 沿输入张量最后一维选出前 `k` 大元素 `values` 与对应专家下标 `indices`，
/// 全程无数据依赖 `if` 跳转，反向传播将 `values.grad` 按 `indices` 路由回原位置。
pub fn topK(self: *Graph, X_in: *Tensor, k: usize) !Tensor.TopKResult {
    const allocator = self.arena.allocator();
    const X = try self.contiguous(X_in);
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
    X_in: *Tensor,
    indices_in: *Tensor,
    weights_in: []const *Tensor,
    biases_in: ?[]const *Tensor,
) !*Tensor {
    const allocator = self.arena.allocator();
    const X = try self.contiguous(X_in);
    const indices = try self.contiguous(indices_in);
    const weights = try allocator.alloc(*Tensor, weights_in.len);
    for (weights, weights_in) |*dst, src| dst.* = try self.contiguous(src);
    var biases: ?[]const *Tensor = null;
    if (biases_in) |bs_in| {
        const bs = try allocator.alloc(*Tensor, bs_in.len);
        for (bs, bs_in) |*dst, src| dst.* = try self.contiguous(src);
        biases = bs;
    }
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
