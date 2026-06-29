const std = @import("std");
const zig_ml = @import("zig_ml");
const dataset = zig_ml.dataset;
const nn = zig_ml.nn;
const autodiff = zig_ml.autodiff;
const tensor = zig_ml.tensor;

const CLASS_NAMES = [10][]const u8{
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
};

pub const MLP = struct {
    // 定义模型结构
    fc1: nn.Linear,
    fc2: nn.Linear,
    fc3: nn.Linear,

    // 定义初始化每一层参数的规则
    pub fn init(allocator: std.mem.Allocator, seed: u64) !MLP {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();

        const fc1 = try nn.Linear.init(allocator, 784, 128, random);
        errdefer fc1.deinit(allocator);

        const fc2 = try nn.Linear.init(allocator, 128, 64, random);
        errdefer fc2.deinit(allocator);

        const fc3 = try nn.Linear.init(allocator, 64, 10, random);
        errdefer fc3.deinit(allocator);

        return MLP{
            .fc1 = fc1,
            .fc2 = fc2,
            .fc3 = fc3,
        };
    }

    // 用户只需专注定义前向传播逻辑
    pub fn forward(self: *const MLP, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *tensor.Tensor) !*tensor.Tensor {
        if (graph == null) {
            // Eager 模式：使用局部 ArenaAllocator 自动管理中间临时 Tensor 内存，避免频繁手写 defer free
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const arena_allocator = arena.allocator();

            const x1 = try self.fc1.forward(arena_allocator, null, x);
            const a1 = try x1.relu(arena_allocator, null);
            const x2 = try self.fc2.forward(arena_allocator, null, a1);
            const a2 = try x2.relu(arena_allocator, null);
            const out_arena = try self.fc3.forward(arena_allocator, null, a2);

            // 将最终结果克隆到外部的 allocator 中返回，出作用域后局部 arena 内的临时变量会被一并释放
            return try zig_ml.tensor.array(allocator, out_arena.shape.dims[0..out_arena.shape.len], out_arena.data);
        }

        const x1 = try self.fc1.forward(allocator, graph, x);
        const a1 = try x1.relu(allocator, graph);
        const x2 = try self.fc2.forward(allocator, graph, a1);
        const a2 = try x2.relu(allocator, graph);
        return try self.fc3.forward(allocator, graph, a2);
    }
};

pub const NeuralNetwork = nn.Module(MLP);

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    std.debug.print("Running on CPU (Accelerate CBLAS sgemm)...\n", .{});

    std.debug.print("Loading dataset...\n", .{});
    var train_dataset = try dataset.loadDataset(io, arena, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    defer train_dataset.deinit(arena);

    var test_dataset = try dataset.loadDataset(io, arena, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    defer test_dataset.deinit(arena);

    std.debug.print("Loaded {} training images, {} test images.\n", .{ train_dataset.images.num_images, test_dataset.images.num_images });

    std.debug.print("Initializing Standard Model (3-layer MLP: 784 -> 128 -> 64 -> 10)...\n", .{});
    var model = NeuralNetwork.init(arena, try MLP.init(arena, 42));
    defer model.deinit();
    try runTraining(&model, arena, io, train_dataset, test_dataset);
    try printPredictions(&model, arena, test_dataset, 5);
}

fn runTraining(
    model: anytype,
    arena: std.mem.Allocator,
    io: std.Io,
    train_dataset: dataset.Dataset,
    test_dataset: dataset.Dataset,
) !void {
    const input_dim = model.inner.fc1.weight.shape.dims[0];
    const batch_size = 64;
    const epochs = 15;
    var lr: f32 = 0.05;
    const beta: f32 = 0.9; // SGD momentum factor

    var train_loader = try dataset.DataLoader.init(arena, train_dataset, batch_size, .{
        .shuffle = true,
        .seed = 1337,
        .drop_last = true,
    });
    defer train_loader.deinit(arena);

    // Allocate label batch buffers (since images are wrapped directly into Tensors, we only need buffers for targets)
    const y_batch = try arena.alloc(u8, batch_size);
    defer arena.free(y_batch);

    std.debug.print("Starting training (3-layer NN with dynamic autodiff)...\n", .{});

    // 开始主循环（15 个 Epoch 的 Fashion MNIST 训练）
    for (0..epochs) |epoch| {
        var epoch_label_buf: [32]u8 = undefined;
        const epoch_label = try std.fmt.bufPrint(&epoch_label_buf, "Epoch {d:2}/{d:2}", .{ epoch + 1, epochs });
        const timer = zig_ml.ProfileBlock.start(epoch_label);
        defer timer.end();

        // Reset loader at start of epoch (this will shuffle)
        train_loader.reset();

        var epoch_loss: f32 = 0.0;
        var epoch_acc: f32 = 0.0;
        var num_batches: usize = 0;

        while (true) {
            const actual_batch_size = train_loader.peekNextBatchSize();
            if (actual_batch_size == 0) break;

            // 关键：初始化一个新的、生命周期处于当前 batch 内的局部计算图。
            // 使用 Arena 分配器，当前 batch 结束后通过 `defer graph.deinit()` 一次性自动释放所有中间层 Tensor 内存。
            var graph = autodiff.Graph.init(arena);
            defer graph.deinit();

            // 将 Batch 的输入数据直接封装为计算图中的 Tensor 节点并分配内存（注意输入数据 requires_grad = false）
            const x_tensor = try graph.tensor(actual_batch_size, input_dim, false);
            _ = train_loader.nextInto(x_tensor.data, y_batch);
            const targets = y_batch[0..actual_batch_size];

            // 执行前向传播构建动态计算图（类似于 PyTorch 的 model(x)）
            const logits = try model.forward(arena, &graph, x_tensor);

            // 损失函数：交叉熵损失节点，附带 Softmax 概率
            const loss = try graph.softmaxCrossEntropy(logits, targets);

            // 提取批次的 Loss 标量与准确率
            const batch_loss = loss.data[0];
            const batch_acc = try computeAccuracy(logits, targets, arena);

            epoch_loss += batch_loss;
            epoch_acc += batch_acc;
            num_batches += 1;

            // 1. Zero out gradients of the model (equivalent to optimizer.zero_grad() in PyTorch)
            model.zeroGrad();

            // 2. Compute gradients through backpropagation (equivalent to loss.backward() in PyTorch)
            try graph.backward(loss);

            // 3. Update weights using SGD with momentum (equivalent to optimizer.step() in PyTorch)
            model.updateWeights(lr, beta);
        }

        epoch_loss /= @as(f32, @floatFromInt(num_batches));
        epoch_acc /= @as(f32, @floatFromInt(num_batches));

        // Evaluate on test dataset
        const eval_res = try evaluateModel(model, arena, test_dataset);
        const test_loss = eval_res.loss;
        const test_acc = eval_res.acc;

        std.debug.print("Epoch {d:2}/{d:2} | Train Loss: {d:.4} | Train Acc: {d:.2}% | Test Loss: {d:.4} | Test Acc: {d:.2}% | ", .{
            epoch + 1,
            epochs,
            epoch_loss,
            epoch_acc * 100.0,
            test_loss,
            test_acc * 100.0,
        });

        // Learning rate decay
        lr *= 0.90;
    }

    // Save model parameters
    std.debug.print("\nSaving trained model to 'model.bin'...\n", .{});
    model.save(io, "model.bin") catch |err| {
        std.debug.print("Failed to save model: {}\n", .{err});
    };
}

fn computeAccuracy(logits: *tensor.Tensor, y: []const u8, allocator: std.mem.Allocator) !f32 {
    const preds = try logits.argmax(1, allocator);
    defer zig_ml.tensor.free(allocator, preds);

    var correct: usize = 0;
    for (preds.data, 0..) |pred_float, i| {
        const pred = @as(usize, @intFromFloat(pred_float));
        if (pred == y[i]) {
            correct += 1;
        }
    }
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(preds.data.len));
}

const EvalResult = struct {
    loss: f32,
    acc: f32,
};

fn evaluateModel(
    model: anytype,
    arena: std.mem.Allocator,
    test_dataset: dataset.Dataset,
) !EvalResult {
    const input_dim = model.inner.fc1.weight.shape.dims[0];
    const test_batch_size = 100;

    var test_loader = try dataset.DataLoader.init(arena, test_dataset, test_batch_size, .{
        .shuffle = false,
        .drop_last = false,
    });
    defer test_loader.deinit(arena);

    const eval_y_batch = try arena.alloc(u8, test_batch_size);
    defer arena.free(eval_y_batch);

    var test_loss: f32 = 0.0;
    var test_acc: f32 = 0.0;
    var batch_count: usize = 0;

    while (true) {
        const actual_batch_size = test_loader.peekNextBatchSize();
        if (actual_batch_size == 0) break;

        var graph = autodiff.Graph.init(arena);
        defer graph.deinit();

        const x_tensor = try graph.tensor(actual_batch_size, input_dim, false);
        _ = test_loader.nextInto(x_tensor.data, eval_y_batch);
        const targets = eval_y_batch[0..actual_batch_size];

        const logits = try model.forward(arena, &graph, x_tensor);

        const loss = try graph.softmaxCrossEntropy(logits, targets);

        test_loss += loss.data[0];
        test_acc += try computeAccuracy(logits, targets, arena);
        batch_count += 1;
    }

    return EvalResult{
        .loss = test_loss / @as(f32, @floatFromInt(batch_count)),
        .acc = test_acc / @as(f32, @floatFromInt(batch_count)),
    };
}

fn printPredictions(
    model: anytype,
    arena: std.mem.Allocator,
    test_dataset: dataset.Dataset,
    count: usize,
) !void {
    const input_dim = model.inner.fc1.weight.shape.dims[0];

    std.debug.print("\nSample Predictions from Test Set:\n", .{});
    for (0..count) |idx| {
        const img_slice = test_dataset.images.data[idx * input_dim .. (idx + 1) * input_dim];
        const actual_label = test_dataset.labels.data[idx];

        var graph = autodiff.Graph.init(arena);
        defer graph.deinit();

        const x_tensor = try graph.tensor(1, input_dim, false);
        @memcpy(x_tensor.data, img_slice);

        const logits = try model.forward(arena, &graph, x_tensor);

        const loss = try graph.softmaxCrossEntropy(logits, &[1]u8{actual_label});
        const preds = try logits.argmax(1, arena);
        const pred = @as(usize, @intFromFloat(preds.data[0]));
        const probs = loss.creator.?.context.SoftmaxCrossEntropy.probs;
        const max_val = probs[pred];

        const is_correct = (pred == actual_label);
        const status = if (is_correct) "✅ CORRECT" else "❌ INCORRECT";
        std.debug.print("Sample #{d:5}: Pred: {s} ({d:.2}%) | Actual: {s} | {s}\n", .{
            idx,
            CLASS_NAMES[pred],
            max_val * 100.0,
            CLASS_NAMES[actual_label],
            status,
        });
    }
}

test "MLP model initialization and forward passes (Eager & Graph)" {
    const allocator = std.testing.allocator;

    var model = NeuralNetwork.init(allocator, try MLP.init(allocator, 42));
    defer model.deinit();

    const x_data = try allocator.alloc(f32, 2 * 784);
    defer allocator.free(x_data);
    @memset(x_data, 0.1);

    // Test Eager Mode (graph == null)
    {
        const x_tensor = try tensor.array(allocator, &.{ 2, 784 }, x_data);
        defer tensor.free(allocator, x_tensor);

        const logits = try model.forward(allocator, null, x_tensor);
        defer tensor.free(allocator, logits);

        try std.testing.expectEqualSlices(usize, &.{ 2, 10 }, logits.shape.dims[0..logits.shape.len]);
    }

    // Test Graph Mode (graph != null)
    {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();

        var graph = autodiff.Graph.init(arena_allocator);
        defer graph.deinit();

        const x_tensor = try graph.tensor(2, 784, false);
        @memcpy(x_tensor.data, x_data);

        const logits = try model.forward(arena_allocator, &graph, x_tensor);

        try std.testing.expectEqualSlices(usize, &.{ 2, 10 }, logits.shape.dims[0..logits.shape.len]);
    }
}
