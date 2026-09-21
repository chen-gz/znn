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

pub const GradClipConfig = @import("optim.zig").GradClipConfig;

/// 通用分类训练单步 (支持可选的梯度裁剪)
pub fn trainClassificationStepWithClip(
    allocator: std.mem.Allocator,
    model: anytype,
    optimizer: anytype,
    x_data: []const f32,
    targets: []const u8,
    clip_config: ?GradClipConfig,
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

    if (clip_config) |cfg| {
        _ = @import("optim.zig").clipGradients(optimizer.params, cfg);
    }

    optimizer.step();

    return ClassificationStepResult{
        .loss = batch_loss,
        .accuracy = batch_acc,
        .batch_size = batch_size,
    };
}

/// 通用分类训练单步 (Classification Train Step)
pub fn trainClassificationStep(
    allocator: std.mem.Allocator,
    model: anytype,
    optimizer: anytype,
    x_data: []const f32,
    targets: []const u8,
) !ClassificationStepResult {
    return trainClassificationStepWithClip(allocator, model, optimizer, x_data, targets, null);
}

/// 通用分类评估单步 (Classification Eval Step)
/// 纯 Eager 前向推理模式：使用 ArenaAllocator 一次性管理评估内存，传入 graph = null，零梯度开销
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

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    const x_tensor = try tensor.array(arena_allocator, &.{ batch_size, input_dim }, x_data);

    const logits = try model.forward(arena_allocator, null, x_tensor);
    const loss = try logits.softmaxCrossEntropy(targets, arena_allocator, null);

    const batch_loss = loss.data[0];
    const batch_acc = try computeAccuracy(logits, targets, arena_allocator);

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

    // 测试带梯度裁剪的单步训练
    const step_clip_res = try trainClassificationStepWithClip(arena, &model, &optim, &x_mock, &y_mock, .{ .norm = 1.0 });
    try std.testing.expect(step_clip_res.loss > 0);
    try std.testing.expectEqual(@as(usize, 2), step_clip_res.batch_size);
}

test "engine computeAccuracy edge cases" {
    const arena = std.testing.allocator;

    // 1. All correct: logits max at target indices
    const logits_perfect = try tensor.array(arena, &.{ 2, 3 }, &[_]f32{
        10.0, 0.0, 0.0, // target 0
        0.0, 10.0, 0.0, // target 1
    });
    defer tensor.free(arena, logits_perfect);

    const targets_perfect = [_]u8{ 0, 1 };
    const acc_perfect = try computeAccuracy(logits_perfect, &targets_perfect, arena);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), acc_perfect, 1e-6);

    // 2. All wrong: logits max at incorrect indices
    const targets_wrong = [_]u8{ 1, 0 };
    const acc_wrong = try computeAccuracy(logits_perfect, &targets_wrong, arena);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), acc_wrong, 1e-6);

    // 3. Half correct
    const targets_half = [_]u8{ 0, 0 };
    const acc_half = try computeAccuracy(logits_perfect, &targets_half, arena);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), acc_half, 1e-6);
}

test "engine trainClassificationEpoch and evaluateClassification with DataLoader" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(123);
    const random = prng.random();

    const SimpleMLP = struct {
        fc: nn.Linear,
        pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
            return .{ .fc = try nn.Linear.init(alloc, 4, 2, rnd) };
        }
        pub fn forward(self: *const @This(), alloc: std.mem.Allocator, graph: ?*autodiff.Graph, x: *tensor.Tensor) !*tensor.Tensor {
            return try self.fc.forward(alloc, graph, x);
        }
    };

    const Model = nn.Module(SimpleMLP);
    var model = Model.init(arena, try SimpleMLP.init(arena, random));
    defer model.deinit();

    var optim = try @import("optim.zig").AdamOptimizer.init(arena, &model, .{ .lr = 0.01 });
    defer optim.deinit();

    var img_data = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    var lbl_data = [_]u8{ 0, 1, 0, 1 };
    const mock_ds = dataset.Dataset{
        .images = .{ .num_images = 4, .rows = 2, .cols = 2, .data = &img_data },
        .labels = .{ .num_items = 4, .data = &lbl_data },
    };

    var loader = try dataset.DataLoader.init(arena, mock_ds, 2, .{ .shuffle = false, .drop_last = false });
    defer loader.deinit(arena);

    const Context = struct {
        var count: usize = 0;
        fn cb(_: usize, _: f32, _: f32) void {
            count += 1;
        }
    };
    Context.count = 0;

    const train_res = try trainClassificationEpoch(arena, &model, &optim, &loader, Context.cb);
    try std.testing.expectEqual(@as(usize, 2), Context.count);
    try std.testing.expectEqual(@as(usize, 2), train_res.num_batches);
    try std.testing.expect(train_res.loss > 0.0);

    const eval_res = try evaluateClassification(arena, &model, &loader);
    try std.testing.expectEqual(@as(usize, 2), eval_res.num_batches);
    try std.testing.expect(eval_res.loss > 0.0);
}

