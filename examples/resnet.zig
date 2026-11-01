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

pub const CliOptions = struct {
    epochs: usize = 1,
    batch_size: usize = 64,
    max_batches: usize = 100, // 默认跑 100 个批次 (6400 张图) 以便快速完成基准评估；设为 0 则跑满全量 937 个批次
    lr: f32 = 0.05,
    momentum: f32 = 0.9,
    arch: []const u8 = "resnet9", // resnet9, resnet18, resnet20
    dataset: []const u8 = "fashion_mnist", // fashion_mnist, tiny_imagenet
    test_eval_batches: usize = 20, // 评估时测 20 个批次 (2000 张图)
};

pub fn parseCli(allocator: std.mem.Allocator, args: anytype) !CliOptions {
    var opts = CliOptions{};
    var iter = try args.iterateAllocator(allocator);
    defer iter.deinit();
    _ = iter.next(); // skip binary path

    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--epochs") or std.mem.eql(u8, arg, "-e")) {
            if (iter.next()) |val| opts.epochs = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--batch-size") or std.mem.eql(u8, arg, "-b")) {
            if (iter.next()) |val| opts.batch_size = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--max-batches") or std.mem.eql(u8, arg, "-m")) {
            if (iter.next()) |val| opts.max_batches = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--lr")) {
            if (iter.next()) |val| opts.lr = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, arg, "--arch") or std.mem.eql(u8, arg, "-a")) {
            if (iter.next()) |val| opts.arch = val;
        } else if (std.mem.eql(u8, arg, "--dataset") or std.mem.eql(u8, arg, "-d")) {
            if (iter.next()) |val| opts.dataset = val;
        } else if (std.mem.eql(u8, arg, "--full")) {
            opts.max_batches = 0; // 全量 60000 张图
            opts.test_eval_batches = 0; // 全量 10000 张测试图
        }
    }
    return opts;
}

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const io = init.io;

    const cli = try parseCli(allocator, init.minimal.args);

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    std.debug.print("================================================================================\n", .{});
    std.debug.print("            ZNN - ResNet (Residual Network) Training & Benchmark                \n", .{});
    std.debug.print("================================================================================\n\n", .{});

    const is_tiny_imagenet = std.mem.eql(u8, cli.dataset, "tiny_imagenet") or std.mem.eql(u8, cli.dataset, "tiny-imagenet");
    const in_channels: usize = if (is_tiny_imagenet) 3 else 1;
    const num_classes: usize = if (is_tiny_imagenet) 200 else 10;
    const spatial_size: usize = if (is_tiny_imagenet) 64 else 28;

    // 1. 构建 ResNet 模型配置
    const resnet_config = if (std.mem.eql(u8, cli.arch, "resnet18"))
        models.ResNetConfig.resnet18(in_channels, num_classes)
    else if (std.mem.eql(u8, cli.arch, "resnet20"))
        models.ResNetConfig.resnet20(in_channels, num_classes)
    else
        models.ResNetConfig.resnet9(in_channels, num_classes);

    std.debug.print("Building ResNet architecture ({s}: in_channels={d}, num_classes={d}, spatial={d}x{d})...\n", .{
        cli.arch,
        in_channels,
        num_classes,
        spatial_size,
        spatial_size,
    });

    var model = try NeuralNetwork.init(arena, try models.ResNet.init(arena, resnet_config));
    defer model.deinit();

    // 2. 参数初始化
    var prng = std.Random.DefaultPrng.init(42);
    const init_sample = try tensor.ones(arena, &.{ 1, in_channels, spatial_size, spatial_size });
    try model.initParametersWithSample(prng.random(), .{init_sample});

    const param_count = model.numParameters();
    std.debug.print("Model initialized. Trainable parameters: {d} (~{d:.2} KB in f32)\n\n", .{
        param_count,
        @as(f64, @floatFromInt(param_count * @sizeOf(f32))) / 1024.0,
    });

    // 3. 数据集加载
    if (is_tiny_imagenet) {
        std.debug.print("Loading Tiny ImageNet (3 channels RGB, 64x64, 200 classes)...\n", .{});
        var train_ds = dataset.loadTinyImageNetBin(arena, io, "data/tiny_imagenet_train.bin", null) catch |err| blk: {
            std.debug.print("  [!] data/tiny_imagenet_train.bin not found ({s}). Generating synthetic Tiny ImageNet (500 train, 100 test samples)...\n", .{@errorName(err)});
            break :blk try dataset.createSyntheticTinyImageNet(arena, 500);
        };
        defer train_ds.deinit(arena);

        var test_ds = dataset.loadTinyImageNetBin(arena, io, "data/tiny_imagenet_val.bin", null) catch blk: {
            break :blk try dataset.createSyntheticTinyImageNet(arena, 100);
        };
        defer test_ds.deinit(arena);

        std.debug.print("  [✓] Loaded {d} train images, {d} test images (RGB 3x64x64, 200 classes).\n\n", .{ train_ds.images.num_images, test_ds.images.num_images });

        try runBenchmarkTraining(&model, allocator, arena, train_ds, test_ds, cli);
        try printPredictions(&model, allocator, test_ds, in_channels, spatial_size, 5);
    } else {
        std.debug.print("Loading Fashion-MNIST dataset from 'data/'...\n", .{});
        const train_dataset_opt = dataset.loadDataset(arena, io, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte") catch |err| blk: {
            std.debug.print("  [!] Dataset files not found ({s}). Switching to synthetic data fallback.\n", .{@errorName(err)});
            break :blk null;
        };

        if (train_dataset_opt) |train_dataset| {
            var td = train_dataset;
            defer td.deinit(arena);
            var test_dataset = try dataset.loadDataset(arena, io, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
            defer test_dataset.deinit(arena);

            std.debug.print("  [✓] Loaded {d} train images, {d} test images.\n\n", .{ td.images.num_images, test_dataset.images.num_images });

            try runBenchmarkTraining(&model, allocator, arena, td, test_dataset, cli);
            try printPredictions(&model, allocator, test_dataset, in_channels, spatial_size, 5);
        } else {
            try runSyntheticTraining(&model, allocator, arena, prng.random());
        }
    }
}

pub const BenchmarkMetrics = struct {
    total_samples: usize = 0,
    total_time_s: f64 = 0.0,
    samples_per_sec: f64 = 0.0,
    avg_step_ms: f64 = 0.0,
    final_train_loss: f32 = 0.0,
    final_train_acc: f32 = 0.0,
    test_loss: f32 = 0.0,
    test_acc: f32 = 0.0,
};

fn runBenchmarkTraining(
    model: anytype,
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    train_dataset: dataset.Dataset,
    test_dataset: dataset.Dataset,
    cli: CliOptions,
) !void {
    var optimizer = try zig_ml.optim.SGDOptimizer.init(arena, model, .{
        .lr = cli.lr,
        .momentum = cli.momentum,
    });
    defer optimizer.deinit();

    var train_loader = try dataset.DataLoader.init(arena, train_dataset, cli.batch_size, .{
        .shuffle = true,
        .seed = 1337,
        .drop_last = true,
    });
    defer train_loader.deinit(arena);

    const input_dim = train_dataset.images.sampleSize();
    const x_buffer = try allocator.alloc(f32, cli.batch_size * input_dim);
    defer allocator.free(x_buffer);
    const y_buffer = try allocator.alloc(u8, cli.batch_size);
    defer allocator.free(y_buffer);

    std.debug.print("Starting Benchmark Training:\n", .{});
    std.debug.print("  - Batch Size:   {d}\n", .{cli.batch_size});
    std.debug.print("  - Batches/Epoch:{s}\n", .{if (cli.max_batches == 0) " full (937)" else " capped"});
    std.debug.print("  - Epochs:       {d}\n", .{cli.epochs});
    std.debug.print("  - Optimizer:    SGD (lr={d:.4}, momentum={d:.2})\n\n", .{ cli.lr, cli.momentum });

    var total_trained_samples: usize = 0;
    var total_train_time_ns: u64 = 0;
    var last_train_loss: f32 = 0.0;
    var last_train_acc: f32 = 0.0;

    for (0..cli.epochs) |epoch| {
        train_loader.reset();
        var batch_idx: usize = 0;
        var epoch_loss_sum: f32 = 0.0;
        var epoch_correct: usize = 0;
        var epoch_samples: usize = 0;

        const epoch_start = zig_ml.bench.getTimeNs();

        while (train_loader.nextInto(x_buffer, y_buffer)) |actual_batch| {
            if (cli.max_batches > 0 and batch_idx >= cli.max_batches) break;

            const step_res = try engine.trainClassificationStep(
                allocator,
                model,
                &optimizer,
                x_buffer[0 .. actual_batch * input_dim],
                y_buffer[0..actual_batch],
            );

            epoch_loss_sum += step_res.loss * @as(f32, @floatFromInt(actual_batch));
            epoch_correct += @as(usize, @intFromFloat(step_res.accuracy * @as(f32, @floatFromInt(actual_batch))));
            epoch_samples += actual_batch;
            batch_idx += 1;

            if (batch_idx % 20 == 0 or (cli.max_batches > 0 and batch_idx == cli.max_batches)) {
                const cur_loss = epoch_loss_sum / @as(f32, @floatFromInt(epoch_samples));
                const cur_acc = @as(f32, @floatFromInt(epoch_correct)) / @as(f32, @floatFromInt(epoch_samples));
                std.debug.print("  [Epoch {d}/{d}] Batch {d:3} | Running Loss: {d:.4} | Acc: {d:5.2}%\n", .{
                    epoch + 1,
                    cli.epochs,
                    batch_idx,
                    cur_loss,
                    cur_acc * 100.0,
                });
            }
        }

        const epoch_elapsed_ns = zig_ml.bench.getTimeNs() - epoch_start;
        total_train_time_ns += epoch_elapsed_ns;
        total_trained_samples += epoch_samples;

        last_train_loss = epoch_loss_sum / @as(f32, @floatFromInt(epoch_samples));
        last_train_acc = @as(f32, @floatFromInt(epoch_correct)) / @as(f32, @floatFromInt(epoch_samples));

        const epoch_s = @as(f64, @floatFromInt(epoch_elapsed_ns)) / 1e9;
        const epoch_fps = @as(f64, @floatFromInt(epoch_samples)) / epoch_s;

        std.debug.print("  -> Epoch {d} Complete: {d} samples in {d:.2}s ({d:.1} img/s) | Train Loss: {d:.4} | Train Acc: {d:.2}%\n\n", .{
            epoch + 1,
            epoch_samples,
            epoch_s,
            epoch_fps,
            last_train_loss,
            last_train_acc * 100.0,
        });

        optimizer.lr *= 0.90;
    }

    // 运行测试集评估
    std.debug.print("Evaluating model on test dataset...\n", .{});
    const test_start = zig_ml.bench.getTimeNs();
    const eval_res = try evaluateModelCapped(model, allocator, test_dataset, cli.test_eval_batches);
    const test_elapsed_s = @as(f64, @floatFromInt(zig_ml.bench.getTimeNs() - test_start)) / 1e9;

    const total_train_s = @as(f64, @floatFromInt(total_train_time_ns)) / 1e9;
    const overall_fps = @as(f64, @floatFromInt(total_trained_samples)) / total_train_s;
    const avg_step_ms = (total_train_s / @as(f64, @floatFromInt(total_trained_samples / cli.batch_size))) * 1000.0;

    // 打印综合 Benchmark 报告看板
    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    ZNN ResNet Performance Benchmark Report                     \n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("  Metric                           | Value                                      \n", .{});
    std.debug.print("-----------------------------------+--------------------------------------------\n", .{});
    std.debug.print("  Network Architecture             | ResNet-9 (Conv2D + BN + Residual + Pool)   \n", .{});
    std.debug.print("  Total Trainable Parameters       | {d:10}                                \n", .{model.numParameters()});
    std.debug.print("  Batch Size                       | {d:10}                                \n", .{cli.batch_size});
    std.debug.print("  Total Trained Samples            | {d:10} images                         \n", .{total_trained_samples});
    std.debug.print("  Training Duration                | {d:10.2} seconds                        \n", .{total_train_s});
    std.debug.print("  Training Throughput              | {d:10.2} images/second                  \n", .{overall_fps});
    std.debug.print("  Average Step Latency             | {d:10.2} ms/step                        \n", .{avg_step_ms});
    std.debug.print("-----------------------------------+--------------------------------------------\n", .{});
    std.debug.print("  Initial (Random) Accuracy        |      10.00 %                               \n", .{});
    std.debug.print("  Training Accuracy (Top-1)        | {d:10.2} %                               \n", .{last_train_acc * 100.0});
    std.debug.print("  Training Loss (Softmax-CE)       | {d:10.4}                                \n", .{last_train_loss});
    std.debug.print("  Test Accuracy (Top-1)            | {d:10.2} %                               \n", .{eval_res.accuracy * 100.0});
    std.debug.print("  Test Loss (Softmax-CE)           | {d:10.4}                                \n", .{eval_res.loss});
    std.debug.print("  Inference Speed (Test Eval)      | {d:10.2} images/second (in {d:.2}s)      \n", .{
        @as(f64, @floatFromInt(eval_res.num_samples)) / test_elapsed_s,
        test_elapsed_s,
    });
    std.debug.print("================================================================================\n\n", .{});
}

fn evaluateModelCapped(
    model: anytype,
    allocator: std.mem.Allocator,
    test_dataset: dataset.Dataset,
    max_eval_batches: usize,
) !engine.ClassificationEpochResult {
    model.eval();
    defer model.train();

    const test_batch_size = 100;
    var test_loader = try dataset.DataLoader.init(allocator, test_dataset, test_batch_size, .{
        .shuffle = false,
        .drop_last = false,
    });
    defer test_loader.deinit(allocator);

    if (max_eval_batches == 0) {
        return try engine.evaluateClassification(allocator, model, &test_loader);
    }

    test_loader.reset();
    var acc = struct {
        total_loss: f64 = 0.0,
        total_acc: f64 = 0.0,
        num_batches: usize = 0,
        total_samples: usize = 0,
    }{};

    const input_dim = test_dataset.images.sampleSize();
    const x_buf = try allocator.alloc(f32, test_batch_size * input_dim);
    defer allocator.free(x_buf);
    const y_buf = try allocator.alloc(u8, test_batch_size);
    defer allocator.free(y_buf);

    var b: usize = 0;
    while (test_loader.nextInto(x_buf, y_buf)) |actual_batch| {
        if (b >= max_eval_batches) break;
        const res = try engine.evalClassificationStep(allocator, model, x_buf[0 .. actual_batch * input_dim], y_buf[0..actual_batch]);
        acc.total_loss += res.loss * @as(f64, @floatFromInt(actual_batch));
        acc.total_acc += res.accuracy * @as(f64, @floatFromInt(actual_batch));
        acc.num_batches += 1;
        acc.total_samples += actual_batch;
        b += 1;
    }

    const n = @as(f32, @floatFromInt(acc.total_samples));
    return engine.ClassificationEpochResult{
        .loss = @as(f32, @floatCast(acc.total_loss / n)),
        .accuracy = @as(f32, @floatCast(acc.total_acc / n)),
        .num_batches = acc.num_batches,
        .num_samples = acc.total_samples,
    };
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

        const x = try g.tensorND(&.{ batch_size, 1, 28, 28 }, false);
        for (x.data) |*p| p.* = random.float(f32) * 2.0 - 1.0;

        const targets = try allocator.alloc(usize, batch_size);
        defer allocator.free(targets);
        for (targets) |*t| t.* = random.uintLessThan(usize, 10);

        model.zeroGrad();

        const logits = try model.forward(&g, x);
        const loss = try g.softmaxCrossEntropy(logits, targets);
        const loss_val = loss.data[0];

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

fn printPredictions(
    model: anytype,
    allocator: std.mem.Allocator,
    test_dataset: dataset.Dataset,
    in_channels: usize,
    spatial_size: usize,
    num_samples: usize,
) !void {
    model.eval();
    defer model.train();

    const input_dim = test_dataset.images.sampleSize();
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

        const x = try g.array(&.{ 1, in_channels, spatial_size, spatial_size }, img_slice, false);
        const logits = try model.forward(&g, x);
        const num_classes = logits.shape.dims[1];

        const preds = try logits.argmax(1, g.arenaAllocator());
        const pred_class = @as(usize, @intFromFloat(preds.data[0]));
        const probs = try logits.softmax(g.arenaAllocator());
        const conf = probs.data[pred_class];

        const match = pred_class == true_label;
        const result_str = if (match) "✓ CORRECT" else "✗ INCORRECT";

        if (num_classes == 10 and true_label < CLASS_NAMES.len and pred_class < CLASS_NAMES.len) {
            std.debug.print("{d:<8} | {s:<18} | {s:<18} | {d:>6.2}%     | {s}\n", .{
                i,
                CLASS_NAMES[true_label],
                CLASS_NAMES[pred_class],
                conf * 100.0,
                result_str,
            });
        } else {
            var true_buf: [32]u8 = undefined;
            var pred_buf: [32]u8 = undefined;
            const true_str = std.fmt.bufPrint(&true_buf, "Class {d}", .{true_label}) catch "Class ?";
            const pred_str = std.fmt.bufPrint(&pred_buf, "Class {d}", .{pred_class}) catch "Class ?";
            std.debug.print("{d:<8} | {s:<18} | {s:<18} | {d:>6.2}%     | {s}\n", .{
                i,
                true_str,
                pred_str,
                conf * 100.0,
                result_str,
            });
        }
    }
    std.debug.print("--------------------------------------------------------------------------------\n\n", .{});
}
