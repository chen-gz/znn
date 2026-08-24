const std = @import("std");
const tensor = @import("tensor.zig");
const autodiff = @import("autodiff.zig");
const dataset = @import("dataset.zig");
const nn = @import("nn.zig");

pub const ClassificationStepResult = struct {
    loss: f32,
    accuracy: f32,
    batch_size: usize,
};

pub const ClassificationEpochResult = struct {
    loss: f32,
    accuracy: f32,
    num_batches: usize,
};

// 兼容别名
pub const StepResult = ClassificationStepResult;
pub const EpochResult = ClassificationEpochResult;

/// 计算多分类批次的预测准确率 (Top-1 Accuracy)
pub fn computeAccuracy(logits: *tensor.Tensor, targets: []const u8, allocator: std.mem.Allocator) !f32 {
    const preds = try logits.argmax(1, allocator);
    defer tensor.free(allocator, preds);

    var correct: usize = 0;
    for (preds.data, 0..) |pred_float, i| {
        const pred = @as(usize, @intFromFloat(pred_float));
        if (pred == targets[i]) {
            correct += 1;
        }
    }
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(preds.data.len));
}

/// 通用分类训练单步 (Classification Train Step)
/// 自动从 targets.len 推导 batch_size，并从 x_data.len / targets.len 推导 input_dim。
/// 流程：
/// 1. 构建局部 Batch 计算图 (Graph)
/// 2. 前向传播计算 Logits
/// 3. 计算 Softmax 交叉熵损失与分类准确率
/// 4. 反向传播计算梯度并调用优化器 step() 更新权重
pub fn trainClassificationStep(
    allocator: std.mem.Allocator,
    model: anytype,
    optimizer: anytype,
    x_data: []const f32,
    targets: []const u8,
) !ClassificationStepResult {
    const batch_size = targets.len;
    std.debug.assert(batch_size > 0);
    std.debug.assert(x_data.len % batch_size == 0);
    const input_dim = x_data.len / batch_size;

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x_tensor = try graph.tensor(batch_size, input_dim, false);
    @memcpy(x_tensor.data, x_data);

    const logits = try model.forward(allocator, &graph, x_tensor);
    const loss = try graph.softmaxCrossEntropy(logits, targets);

    const batch_loss = loss.data[0];
    const batch_acc = try computeAccuracy(logits, targets, allocator);

    model.zeroGrad();
    try graph.backward(loss);
    optimizer.step();

    return ClassificationStepResult{
        .loss = batch_loss,
        .accuracy = batch_acc,
        .batch_size = batch_size,
    };
}

/// 通用分类评估单步 (Classification Eval Step)
/// 仅执行前向传播和指标计算，不计算梯度与更新权重
pub fn evalClassificationStep(
    allocator: std.mem.Allocator,
    model: anytype,
    x_data: []const f32,
    targets: []const u8,
) !ClassificationStepResult {
    const batch_size = targets.len;
    std.debug.assert(batch_size > 0);
    std.debug.assert(x_data.len % batch_size == 0);
    const input_dim = x_data.len / batch_size;

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x_tensor = try graph.tensor(batch_size, input_dim, false);
    @memcpy(x_tensor.data, x_data);

    const logits = try model.forward(allocator, &graph, x_tensor);
    const loss = try graph.softmaxCrossEntropy(logits, targets);

    const batch_loss = loss.data[0];
    const batch_acc = try computeAccuracy(logits, targets, allocator);

    return ClassificationStepResult{
        .loss = batch_loss,
        .accuracy = batch_acc,
        .batch_size = batch_size,
    };
}

/// 训练整个 DataLoader 的一个完整 Epoch (Classification)
/// 自动从 loader.dataset 获取单样本输入维度 (input_dim = rows * cols)
pub fn trainClassificationEpoch(
    allocator: std.mem.Allocator,
    model: anytype,
    optimizer: anytype,
    loader: *dataset.DataLoader,
    progress_callback: ?*const fn (batch_idx: usize, loss: f32, acc: f32) void,
) !ClassificationEpochResult {
    loader.reset();
    var total_loss: f32 = 0.0;
    var total_acc: f32 = 0.0;
    var num_batches: usize = 0;

    const input_dim = loader.dataset.images.rows * loader.dataset.images.cols;
    const max_batch_size = loader.batch_size;
    const x_buffer = try allocator.alloc(f32, max_batch_size * input_dim);
    defer allocator.free(x_buffer);
    const y_buffer = try allocator.alloc(u8, max_batch_size);
    defer allocator.free(y_buffer);

    while (true) {
        const actual_batch_size = loader.peekNextBatchSize();
        if (actual_batch_size == 0) break;

        _ = loader.nextInto(x_buffer, y_buffer);

        const step_res = try trainClassificationStep(
            allocator,
            model,
            optimizer,
            x_buffer[0 .. actual_batch_size * input_dim],
            y_buffer[0..actual_batch_size],
        );

        total_loss += step_res.loss;
        total_acc += step_res.accuracy;
        num_batches += 1;

        if (progress_callback) |cb| {
            cb(num_batches, step_res.loss, step_res.accuracy);
        }
    }

    if (num_batches == 0) return ClassificationEpochResult{ .loss = 0, .accuracy = 0, .num_batches = 0 };

    return ClassificationEpochResult{
        .loss = total_loss / @as(f32, @floatFromInt(num_batches)),
        .accuracy = total_acc / @as(f32, @floatFromInt(num_batches)),
        .num_batches = num_batches,
    };
}

/// 评估整个 DataLoader (Classification)
/// 自动从 loader.dataset 获取单样本输入维度 (input_dim = rows * cols)
pub fn evaluateClassification(
    allocator: std.mem.Allocator,
    model: anytype,
    loader: *dataset.DataLoader,
) !ClassificationEpochResult {
    loader.reset();
    var total_loss: f32 = 0.0;
    var total_acc: f32 = 0.0;
    var num_batches: usize = 0;

    const input_dim = loader.dataset.images.rows * loader.dataset.images.cols;
    const max_batch_size = loader.batch_size;
    const x_buffer = try allocator.alloc(f32, max_batch_size * input_dim);
    defer allocator.free(x_buffer);
    const y_buffer = try allocator.alloc(u8, max_batch_size);
    defer allocator.free(y_buffer);

    while (true) {
        const actual_batch_size = loader.peekNextBatchSize();
        if (actual_batch_size == 0) break;

        _ = loader.nextInto(x_buffer, y_buffer);

        const step_res = try evalClassificationStep(
            allocator,
            model,
            x_buffer[0 .. actual_batch_size * input_dim],
            y_buffer[0..actual_batch_size],
        );

        total_loss += step_res.loss;
        total_acc += step_res.accuracy;
        num_batches += 1;
    }

    if (num_batches == 0) return ClassificationEpochResult{ .loss = 0, .accuracy = 0, .num_batches = 0 };

    return ClassificationEpochResult{
        .loss = total_loss / @as(f32, @floatFromInt(num_batches)),
        .accuracy = total_acc / @as(f32, @floatFromInt(num_batches)),
        .num_batches = num_batches,
    };
}

// 别名导出
pub const trainStep = trainClassificationStep;
pub const evalStep = evalClassificationStep;
pub const trainEpoch = trainClassificationEpoch;
pub const evaluate = evaluateClassification;

test "engine trainClassificationStep and evalClassificationStep" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const SimpleMLP = struct {
        fc: nn.Linear,

        pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
            return .{
                .fc = try nn.Linear.init(alloc, 4, 2, rnd),
            };
        }

        pub fn forward(self: *const @This(), alloc: std.mem.Allocator, graph: ?*autodiff.Graph, x: *tensor.Tensor) !*tensor.Tensor {
            return try self.fc.forward(alloc, graph, x);
        }
    };

    const Model = nn.Module(SimpleMLP);
    var model = Model.init(arena, try SimpleMLP.init(arena, random));
    defer model.deinit();

    var optim = try @import("optim.zig").SGDOptimizer.init(arena, &model, .{ .lr = 0.1 });
    defer optim.deinit();

    const x_mock = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 }; // 2 samples of dim 4
    const y_mock = [_]u8{ 0, 1 };

    // 不需要显式传递 batch_size 和 input_dim，全部自动推导
    const step_res = try trainClassificationStep(arena, &model, &optim, &x_mock, &y_mock);
    try std.testing.expect(step_res.loss > 0);
    try std.testing.expect(step_res.accuracy >= 0 and step_res.accuracy <= 1.0);
    try std.testing.expectEqual(@as(usize, 2), step_res.batch_size);

    const eval_res = try evalClassificationStep(arena, &model, &x_mock, &y_mock);
    try std.testing.expect(eval_res.loss > 0);
    try std.testing.expect(eval_res.accuracy >= 0 and eval_res.accuracy <= 1.0);
    try std.testing.expectEqual(@as(usize, 2), eval_res.batch_size);
}
