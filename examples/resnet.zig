const std = @import("std");
const zig_ml = @import("zig_ml");
const dataset = zig_ml.dataset;
const nn = zig_ml.nn;
const engine = zig_ml.engine;
const autodiff = zig_ml.autodiff;
const tensor = zig_ml.tensor;
const models = zig_ml.models;

const CLASS_NAMES = [10][]const u8{
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
};

pub const NeuralNetwork = models.DefaultResNet;

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const io = init.io;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    std.debug.print("============================================================\n", .{});
    std.debug.print("        ZNN - ResNet (Residual Network) Example             \n", .{});
    std.debug.print("============================================================\n\n", .{});

    // 1. 构建 ResNet 模型配置
    // 采用轻量化残差网络结构 (Stem 1->16, Stage 1: 16->16, Stage 2: 16->32, Stage 3: 32->64, 全局自适应均值池化 -> FC 10)
    const resnet_config = models.ResNetConfig.resnet9(1, 10);
    std.debug.print("Initializing ResNet model (Stem: 1->16, Stages: [16, 32, 64], Head: 64->10)...\n", .{});

    var model = try NeuralNetwork.init(arena, try models.ResNet.init(arena, resnet_config));
    defer model.deinit();

    // 2. 参数初始化
    var prng = std.Random.DefaultPrng.init(42);
    const init_sample = try tensor.ones(arena, &.{ 1, 1, 28, 28 });
    try model.initParametersWithSample(prng.random(), .{init_sample});

    const param_count = model.numParameters();
    std.debug.print("Model initialized successfully. Total trainable parameters: {d}\n\n", .{param_count});

    // 3. 数据集加载与容灾回退
    std.debug.print("Attempting to load Fashion-MNIST dataset from 'data/'...\n", .{});
    const train_dataset_opt = dataset.loadDataset(arena, io, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte") catch |err| blk: {
        std.debug.print("  [!] Dataset files not found ({s}). Switching to synthetic image batches for demonstration.\n", .{@errorName(err)});
        std.debug.print("  [i] (Run 'zig build download_data -- fashion_mnist' to fetch the real dataset)\n\n", .{});
        break :blk null;
    };

    if (train_dataset_opt) |train_dataset| {
        var td = train_dataset;
        defer td.deinit(arena);
        var test_dataset = try dataset.loadDataset(arena, io, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
        defer test_dataset.deinit(arena);

        std.debug.print("Loaded {d} training images, {d} test images.\n", .{ td.images.num_images, test_dataset.images.num_images });
        try runDatasetTraining(&model, allocator, arena, td, test_dataset);
        try printDatasetPredictions(&model, allocator, test_dataset, 5);
    } else {
        // 使用合成图像数据运行端到端训练与反向传播验证
        try runSyntheticTraining(&model, allocator, arena, prng.random());
    }
}

/// 每 50 个批次打印一次训练进度
fn printBatchProgress(batch_idx: usize, loss: f32, acc: f32) void {
    if (batch_idx % 50 == 0) {
        std.debug.print("  Batch {d:4} | Loss: {d:.4} | Acc: {d:.2}%\n", .{ batch_idx, loss, acc * 100.0 });
    }
}

fn runDatasetTraining(
    model: anytype,
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    train_dataset: dataset.Dataset,
    test_dataset: dataset.Dataset,
) !void {
    const batch_size = 64;
    const epochs = 2;

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

    std.debug.print("Starting ResNet training on Fashion-MNIST...\n", .{});

    for (0..epochs) |epoch| {
        var epoch_label_buf: [32]u8 = undefined;
        const epoch_label = try std.fmt.bufPrint(&epoch_label_buf, "Epoch {d:2}/{d:2}", .{ epoch + 1, epochs });
        const timer = zig_ml.ProfileBlock.start(epoch_label);
        defer timer.end();

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
}

fn runSyntheticTraining(
    model: anytype,
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    random: std.Random,
) !void {
    std.debug.print("Executing ResNet end-to-end forward, loss, backward & optimization on synthetic data...\n", .{});

    const batch_size = 16;
    const steps = 5;

    var optimizer = try zig_ml.optim.SGDOptimizer.init(arena, model, .{
        .lr = 0.02,
        .momentum = 0.9,
    });
    defer optimizer.deinit();

    for (0..steps) |step| {
        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();

        // 构造合成四维输入 [N, 1, 28, 28] 与随机类别目标 [N]
        const x = try g.tensorND(&.{ batch_size, 1, 28, 28 }, false);
        for (x.data) |*p| p.* = random.float(f32) * 2.0 - 1.0;

        const targets = try allocator.alloc(usize, batch_size);
        defer allocator.free(targets);
        for (targets) |*t| t.* = random.uintLessThan(usize, 10);

        model.zeroGrad();

        // 前向传播
        const logits = try model.forward(&g, x);

        // 计算交叉熵损失
        const loss = try g.softmaxCrossEntropy(logits, targets);
        const loss_val = loss.data[0];

        // 准确率统计
        var correct: usize = 0;
        for (0..batch_size) |i| {
            var max_val = logits.data[i * 10];
            var pred_c: usize = 0;
            for (1..10) |c| {
                const val = logits.data[i * 10 + c];
                if (val > max_val) {
                    max_val = val;
                    pred_c = c;
                }
            }
            if (pred_c == targets[i]) correct += 1;
        }
        const acc = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(batch_size));

        // 反向传播与优化器参数更新
        try g.backward(loss);
        optimizer.step();

        std.debug.print("  Step {d:2}/{d:2} | Synthetic Loss: {d:.4} | Batch Acc: {d:.2}%\n", .{
            step + 1,
            steps,
            loss_val,
            acc * 100.0,
        });
    }

    std.debug.print("\n[✓] ResNet forward, backward autograd, and optimizer step verified successfully!\n", .{});
}

fn evaluateModel(
    model: anytype,
    allocator: std.mem.Allocator,
    test_dataset: dataset.Dataset,
) !engine.ClassificationEpochResult {
    model.eval();
    defer model.train();

    const test_batch_size = 100;
    var test_loader = try dataset.DataLoader.init(allocator, test_dataset, test_batch_size, .{
        .shuffle = false,
        .drop_last = false,
    });
    defer test_loader.deinit(allocator);

    return try engine.evaluateClassification(allocator, model, &test_loader);
}

fn printDatasetPredictions(
    model: anytype,
    allocator: std.mem.Allocator,
    test_dataset: dataset.Dataset,
    num_samples: usize,
) !void {
    model.eval();
    defer model.train();

    const input_dim = 784;
    std.debug.print("\nSample Predictions from Test Set:\n", .{});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});
    std.debug.print("{s:<8} | {s:<18} | {s:<18} | {s:<12} | {s}\n", .{ "Index", "True Label", "Predicted", "Confidence", "Result" });
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    const actual_samples = @min(num_samples, test_dataset.images.num_images);
    for (0..actual_samples) |i| {
        const img_slice = test_dataset.images.data[i * input_dim .. (i + 1) * input_dim];
        const true_label = test_dataset.labels.data[i];

        var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
        defer g.deinit();

        const x = try g.array(&.{ 1, 1, 28, 28 }, img_slice, false);
        const logits = try model.forward(&g, x);

        const preds = try logits.argmax(1, g.arenaAllocator());
        const pred_class = @as(usize, @intFromFloat(preds.data[0]));
        const probs = try logits.softmax(g.arenaAllocator());
        const conf = probs.data[pred_class];

        const match = pred_class == true_label;
        const result_str = if (match) "✓ CORRECT" else "✗ INCORRECT";

        std.debug.print("{d:<8} | {s:<18} | {s:<18} | {d:>6.2}%     | {s}\n", .{
            i,
            CLASS_NAMES[true_label],
            CLASS_NAMES[pred_class],
            conf * 100.0,
            result_str,
        });
    }
    std.debug.print("--------------------------------------------------------------------------------\n", .{});
}
