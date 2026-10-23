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

pub const MLP = struct {
    // 定义模型结构
    fc1: nn.Linear,
    fc2: nn.Linear,
    fc3: nn.Linear,

    // 定义模型结构：init 只负责分配各层参数内存，参数数值在建立前向计算图后由 nn.initModel / Module.initParameters 统一初始化
    pub fn init(allocator: std.mem.Allocator) !MLP {
        return .{
            .fc1 = try nn.Linear.init(allocator, 784, 128),
            .fc2 = try nn.Linear.init(allocator, 128, 64),
            .fc3 = try nn.Linear.init(allocator, 64, 10),
        };
    }


    // 用户只需专注定义前向数据流逻辑，所有算子都通过传入的计算图执行。
    // 【内存管理说明】：前向计算产生的中间张量（x1, a1, x2, a2）全部挂载在计算图上，由调用方在批次结束调用 graph.deinit() 统一释放。
    // 训练时使用 `Graph.initDefault` 记录反向传播所需的算子；推理时使用 `Graph.init(allocator, .{ .enable_grad = false })`，只执行前向计算，不分配梯度缓冲区。
    pub fn forward(self: *const MLP, graph: *autodiff.Graph, x: *tensor.Tensor) !*tensor.Tensor {
        const x1 = try self.fc1.forward(graph, x);
        const a1 = try graph.relu(x1);
        const x2 = try self.fc2.forward(graph, a1);
        const a2 = try graph.relu(x2);
        return try self.fc3.forward(graph, a2);
    }


};

pub const NeuralNetwork = nn.Module(MLP);

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const io = init.io;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    std.debug.print("Running on CPU...\n", .{});

    std.debug.print("Loading dataset...\n", .{});
    var train_dataset = dataset.loadDataset(arena, io, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte") catch |err| {
        std.debug.print("❌ Error loading training dataset: {}\n", .{err});
        std.debug.print("💡 Hint: Please make sure Fashion MNIST files are present in the 'data/' directory (e.g. data/train-images-idx3-ubyte).\n", .{});
        return err;
    };
    defer train_dataset.deinit(arena);

    var test_dataset = dataset.loadDataset(arena, io, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte") catch |err| {
        std.debug.print("❌ Error loading test dataset: {}\n", .{err});
        std.debug.print("💡 Hint: Please make sure Fashion MNIST files are present in the 'data/' directory (e.g. data/t10k-images-idx3-ubyte).\n", .{});
        return err;
    };
    defer test_dataset.deinit(arena);

    std.debug.print("Loaded {} training images, {} test images.\n", .{ train_dataset.images.num_images, test_dataset.images.num_images });

    std.debug.print("Initializing Standard Model (3-layer MLP: 784 -> 128 -> 64 -> 10)...\n", .{});
    var model = try NeuralNetwork.init(arena, try MLP.init(arena));
    defer model.deinit();
    // 用一个样本批次建立前向计算图，依据计算图与激活函数初始化参数
    var prng = std.Random.DefaultPrng.init(42);
    const init_sample = try tensor.ones(arena, &.{ 1, 784 });
    try model.initParametersWithSample(prng.random(), .{init_sample});
    try runTraining(&model, io, allocator, arena, train_dataset, test_dataset);
    try printPredictions(&model, allocator, test_dataset, 5);
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
    const epochs = 15;

    var optimizer = try zig_ml.optim.SGDOptimizer.init(arena, model, .{
        .lr = 0.05,
        .momentum = 0.9,
    });
    defer optimizer.deinit();

    var train_loader = try dataset.DataLoader.init(arena, train_dataset, batch_size, .{
        .shuffle = true,
        .seed = 1337,
        .drop_last = true,
    });
    defer train_loader.deinit(arena);

    std.debug.print("Starting training (3-layer NN with dynamic autodiff)...\n", .{});

    // 开始主循环（15 个 Epoch 的 Fashion MNIST 训练）
    for (0..epochs) |epoch| {
        var epoch_label_buf: [32]u8 = undefined;
        const epoch_label = try std.fmt.bufPrint(&epoch_label_buf, "Epoch {d:2}/{d:2}", .{ epoch + 1, epochs });
        const timer = zig_ml.ProfileBlock.start(epoch_label);
        defer timer.end();

        // 通用分类训练 Epoch：每个 Epoch 开始时重置 (并重新打乱) 数据加载器，自动推导 batch_size 和 input_dim
        const train_res = try engine.trainClassificationEpoch(allocator, model, &optimizer, &train_loader, null);

        // Evaluate on test dataset
        const eval_res = try evaluateModel(model, allocator, test_dataset);

        std.debug.print("Epoch {d:2}/{d:2} | Train Loss: {d:.4} | Train Acc: {d:.2}% | Test Loss: {d:.4} | Test Acc: {d:.2}% | ", .{
            epoch + 1,
            epochs,
            train_res.loss,
            train_res.accuracy * 100.0,
            eval_res.loss,
            eval_res.accuracy * 100.0,
        });

        // Learning rate decay
        optimizer.lr *= 0.90;
    }

    // Save model parameters
    std.debug.print("\nSaving trained model to 'model.bin'...\n", .{});
    model.save(io, "model.bin") catch |err| {
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
    const input_dim = model.inner.fc1.weight.shape.dims[0];

    std.debug.print("\nSample Predictions from Test Set:\n", .{});
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

test "MLP model initialization and forward passes (no-grad & gradient graphs)" {
    const allocator = std.testing.allocator;

    var model = try NeuralNetwork.init(allocator, try MLP.init(allocator));
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
