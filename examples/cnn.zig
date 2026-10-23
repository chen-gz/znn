const std = @import("std");
const zig_ml = @import("zig_ml");
const dataset = zig_ml.dataset;
const nn = zig_ml.nn;
const engine = zig_ml.engine;
const autodiff = zig_ml.autodiff;
const tensor = zig_ml.tensor;

const CLASS_NAMES = [10][]const u8{
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
};

pub const CNN = struct {
    conv1: nn.Conv2D,
    conv2: nn.Conv2D,
    conv3: nn.Conv2D,
    fc1: nn.Linear,

    /// 只分配各层参数内存，参数数值在建立前向计算图后由 nn.initModel / Module.initParameters 统一初始化
    pub fn init(allocator: std.mem.Allocator) !CNN {
        return .{
            .conv1 = try nn.Conv2D.init(allocator, 1, 4, 3, .{}),
            .conv2 = try nn.Conv2D.init(allocator, 4, 8, 3, .{}),
            .conv3 = try nn.Conv2D.init(allocator, 8, 16, 3, .{}),
            .fc1 = try nn.Linear.init(allocator, 144, 10),
        };
    }

    // 【内存管理说明】：前向计算产生的中间张量全部挂载在传入的计算图上，由调用方在批次结束调用 graph.deinit() 统一释放。
    // 训练时使用 `Graph.initDefault` 记录反向传播所需的算子；推理时使用 `Graph.init(allocator, .{ .enable_grad = false })`，只执行前向计算，不分配梯度缓冲区。
    pub fn forward(self: *const CNN, graph: *autodiff.Graph, x: *tensor.Tensor) !*tensor.Tensor {
        const batch_size = x.shape.dims[0];
        const x_reshaped = try graph.reshape(x, &.{ batch_size, 1, 28, 28 });

        // Layer 1: Conv -> ReLU -> MaxPool
        const x1 = try self.conv1.forward(graph, x_reshaped);
        const a1 = try graph.relu(x1);
        const p1 = try graph.maxpool2d(a1, 2, .{});

        // Layer 2: Conv -> ReLU -> MaxPool
        const x2 = try self.conv2.forward(graph, p1);
        const a2 = try graph.relu(x2);
        const p2 = try graph.maxpool2d(a2, 2, .{});

        // Layer 3: Conv -> ReLU
        const x3 = try self.conv3.forward(graph, p2);
        const a3 = try graph.relu(x3);

        // Flatten -> Linear
        const flat = try graph.reshape(a3, &.{ batch_size, 144 });
        return try self.fc1.forward(graph, flat);
    }
};

pub const NeuralNetwork = nn.Module(CNN);

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const io = init.io;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    std.debug.print("Running CNN example on CPU...\n", .{});

    std.debug.print("Loading dataset...\n", .{});
    var train_dataset = try dataset.loadDataset(arena, io, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    defer train_dataset.deinit(arena);

    var test_dataset = try dataset.loadDataset(arena, io, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    defer test_dataset.deinit(arena);

    std.debug.print("Loaded {} training images, {} test images.\n", .{ train_dataset.images.num_images, test_dataset.images.num_images });

    std.debug.print("Initializing 3-Layer CNN Model (Conv1 1->4, Conv2 4->8, Conv3 8->16, FC 144->10)...\n", .{});
    var model = try NeuralNetwork.init(arena, try CNN.init(arena));
    defer model.deinit();
    // 用一个样本批次建立前向计算图，依据计算图与激活函数初始化参数
    var prng = std.Random.DefaultPrng.init(42);
    const init_sample = try tensor.ones(arena, &.{ 1, 784 });
    try model.initParametersWithSample(prng.random(), .{init_sample});

    try runTraining(&model, io, allocator, arena, train_dataset, test_dataset);
    try printPredictions(&model, allocator, test_dataset, 5);
}

/// 每 100 个批次打印一次训练进度
fn printBatchProgress(batch_idx: usize, loss: f32, acc: f32) void {
    if (batch_idx % 100 == 0) {
        std.debug.print("  Batch {d:4} | Loss: {d:.4} | Acc: {d:.2}%\n", .{ batch_idx, loss, acc * 100.0 });
    }
}

/// `allocator` 用于每个批次的计算图：计算图在批次结束时释放，内存会真正归还；
/// `arena` 只承载生命周期贯穿整个训练过程的对象 (优化器状态、数据加载器)。
/// 若把逐批次的计算图也挂在外层 arena 上，arena 无法回收中间释放的内存，训练期间内存会持续增长。
fn runTraining(
    model: anytype,
    io: std.Io,
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    train_dataset: dataset.Dataset,
    test_dataset: dataset.Dataset,
) !void {
    const batch_size = 64;
    const epochs = 3; // CNN is slower on CPU, so run for fewer epochs in example

    var optimizer = try zig_ml.optim.SGDOptimizer.init(arena, model, .{
        .lr = 0.02,
        .momentum = 0.9,
    });
    defer optimizer.deinit();

    var train_loader = try dataset.DataLoader.init(arena, train_dataset, batch_size, .{
        .shuffle = true,
        .seed = 1337,
        .drop_last = true,
    });
    defer train_loader.deinit(arena);

    std.debug.print("Starting training (CNN with Conv2D + MaxPool2D + Linear)...\n", .{});

    for (0..epochs) |epoch| {
        var epoch_label_buf: [32]u8 = undefined;
        const epoch_label = try std.fmt.bufPrint(&epoch_label_buf, "Epoch {d:2}/{d:2}", .{ epoch + 1, epochs });
        const timer = zig_ml.ProfileBlock.start(epoch_label);
        defer timer.end();

        // 通用分类训练 Epoch (自动从 targets.len 推导 batch_size，从数据集推导 input_dim)
        const train_res = try engine.trainClassificationEpoch(allocator, model, &optimizer, &train_loader, printBatchProgress);

        const eval_res = try evaluateModel(model, allocator, test_dataset);

        std.debug.print("Epoch {d:2}/{d:2} | Train Loss: {d:.4} | Train Acc: {d:.2}% | Test Loss: {d:.4} | Test Acc: {d:.2}%\n", .{
            epoch + 1,
            epochs,
            train_res.loss,
            train_res.accuracy * 100.0,
            eval_res.loss,
            eval_res.accuracy * 100.0,
        });

        optimizer.lr *= 0.90;
    }

    std.debug.print("\nSaving trained CNN model to 'cnn_model.bin'...\n", .{});
    model.save(io, "cnn_model.bin") catch |err| {
        std.debug.print("Failed to save model: {}\n", .{err});
    };
}

fn evaluateModel(
    model: anytype,
    allocator: std.mem.Allocator,
    test_dataset: dataset.Dataset,
) !engine.ClassificationEpochResult {
    const test_batch_size = 100;

    var test_loader = try dataset.DataLoader.init(allocator, test_dataset, test_batch_size, .{
        .shuffle = false,
        .drop_last = false,
    });
    defer test_loader.deinit(allocator);

    return try engine.evaluateClassification(allocator, model, &test_loader);
}

fn printPredictions(
    model: anytype,
    allocator: std.mem.Allocator,
    test_dataset: dataset.Dataset,
    count: usize,
) !void {
    const input_dim = 784;

    std.debug.print("\nSample CNN Predictions from Test Set:\n", .{});
    for (0..count) |idx| {
        const img_slice = test_dataset.images.data[idx * input_dim .. (idx + 1) * input_dim];
        const actual_label = test_dataset.labels.data[idx];

        // 推理只需前向计算：无梯度计算图，中间张量 (包括 argmax 结果) 都由计算图的 arena 统一释放
        var graph = autodiff.Graph.init(allocator, .{ .enable_grad = false });
        defer graph.deinit();

        const x_tensor = try graph.array(&.{ 1, input_dim }, img_slice, false);
        const logits = try model.forward(&graph, x_tensor);

        const preds = try logits.argmax(1, graph.arenaAllocator());
        const pred = @as(usize, @intFromFloat(preds.data[0]));
        const probs = try logits.softmax(graph.arenaAllocator());
        const max_val = probs.data[pred];

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

test "CNN model initialization and forward passes (no-grad & gradient graphs)" {
    const allocator = std.testing.allocator;

    var model = try NeuralNetwork.init(allocator, try CNN.init(allocator));
    defer model.deinit();

    const x_data = try allocator.alloc(f32, 2 * 784);
    defer allocator.free(x_data);
    @memset(x_data, 0.1);

    var prng = std.Random.DefaultPrng.init(42);
    const init_sample = try tensor.array(allocator, &.{ 2, 784 }, x_data);
    defer init_sample.deinit(allocator);
    try model.initParametersWithSample(prng.random(), .{init_sample});

    // Inference mode (no-grad graph)
    {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();

        const x_tensor = try tensor.array(arena_allocator, &.{ 2, 784 }, x_data);
        var logits_graph = autodiff.Graph.init(arena_allocator, .{ .enable_grad = false });
        defer logits_graph.deinit();
        const logits = try model.forward(&logits_graph, x_tensor);

        try std.testing.expectEqualSlices(usize, &.{ 2, 10 }, logits.shape.dims[0..logits.shape.len]);
    }

    // Training mode (gradient-recording graph)
    {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();

        var graph = autodiff.Graph.initDefault(arena_allocator);
        defer graph.deinit();

        const x_tensor = try graph.tensor(2, 784, false);
        @memcpy(x_tensor.data, x_data);

        const logits = try model.forward(&graph, x_tensor);

        try std.testing.expectEqualSlices(usize, &.{ 2, 10 }, logits.shape.dims[0..logits.shape.len]);
    }
}
