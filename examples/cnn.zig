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

pub const CNN = struct {
    conv1: nn.Conv2D,
    conv2: nn.Conv2D,
    conv3: nn.Conv2D,
    fc1: nn.Linear,

    pub fn init(allocator: std.mem.Allocator, seed: u64) !CNN {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();

        return .{
            .conv1 = try nn.Conv2D.init(allocator, 1, 4, 3, random),
            .conv2 = try nn.Conv2D.init(allocator, 4, 8, 3, random),
            .conv3 = try nn.Conv2D.init(allocator, 8, 16, 3, random),
            .fc1 = try nn.Linear.init(allocator, 144, 10, random),
        };
    }


    // 【内存管理说明】：前向计算产生的中间张量生命周期由外部调用方统一管理：
    // - 训练模式（graph != null）：中间节点挂载在计算图上，由外部在批次结束调用 graph.deinit() 统一一键释放；
    // - 纯推理模式（graph == null）：由外部调用方传入的 ArenaAllocator 在当前作用域结束时统一批量释放。
    pub fn forward(self: *const CNN, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *tensor.Tensor) !*tensor.Tensor {
        const batch_size = x.shape.dims[0];
        const x_reshaped = try x.reshape(&.{ batch_size, 1, 28, 28 }, allocator, graph);

        // Layer 1: Conv -> ReLU -> MaxPool
        const x1 = try self.conv1.forward(allocator, graph, x_reshaped);
        const a1 = try x1.relu(allocator, graph);
        const p1 = try a1.maxpool2d(2, 2, allocator, graph);

        // Layer 2: Conv -> ReLU -> MaxPool
        const x2 = try self.conv2.forward(allocator, graph, p1);
        const a2 = try x2.relu(allocator, graph);
        const p2 = try a2.maxpool2d(2, 2, allocator, graph);

        // Layer 3: Conv -> ReLU
        const x3 = try self.conv3.forward(allocator, graph, p2);
        const a3 = try x3.relu(allocator, graph);

        // Flatten -> Linear
        const flat = try a3.reshape(&.{ batch_size, 144 }, allocator, graph);
        return try self.fc1.forward(allocator, graph, flat);
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
    var model = NeuralNetwork.init(arena, try CNN.init(arena, 42));
    defer model.deinit();

    try runTraining(&model, io, arena, train_dataset, test_dataset);
    try printPredictions(&model, arena, test_dataset, 5);
}

fn runTraining(
    model: anytype,
    io: std.Io,
    arena: std.mem.Allocator,
    train_dataset: dataset.Dataset,
    test_dataset: dataset.Dataset,
) !void {
    const input_dim = 784; // 28 * 28
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

    const x_batch = try arena.alloc(f32, batch_size * input_dim);
    defer arena.free(x_batch);
    const y_batch = try arena.alloc(u8, batch_size);
    defer arena.free(y_batch);

    std.debug.print("Starting training (CNN with Conv2D + MaxPool2D + Linear)...\n", .{});

    for (0..epochs) |epoch| {
        var epoch_label_buf: [32]u8 = undefined;
        const epoch_label = try std.fmt.bufPrint(&epoch_label_buf, "Epoch {d:2}/{d:2}", .{ epoch + 1, epochs });
        const timer = zig_ml.ProfileBlock.start(epoch_label);
        defer timer.end();

        train_loader.reset();

        var epoch_loss: f32 = 0.0;
        var epoch_acc: f32 = 0.0;
        var num_batches: usize = 0;

        while (true) {
            const actual_batch_size = train_loader.peekNextBatchSize();
            if (actual_batch_size == 0) break;

            _ = train_loader.nextInto(x_batch, y_batch);

            // 使用通用分类训练单步 (自动从 targets.len 推导 batch_size，从 x_batch 长度推导 input_dim)
            const step_res = try nn.trainStep(
                arena,
                model,
                &optimizer,
                x_batch[0 .. actual_batch_size * input_dim],
                y_batch[0..actual_batch_size],
            );


            epoch_loss += step_res.loss;
            epoch_acc += step_res.accuracy;
            num_batches += 1;

            // Print batch progress every 100 batches
            if (num_batches % 100 == 0) {
                std.debug.print("  Batch {d:4} | Loss: {d:.4} | Acc: {d:.2}%\n", .{
                    num_batches,
                    step_res.loss,
                    step_res.accuracy * 100.0,
                });
            }
        }

        epoch_loss /= @as(f32, @floatFromInt(num_batches));
        epoch_acc /= @as(f32, @floatFromInt(num_batches));

        const eval_res = try evaluateModel(model, arena, test_dataset);

        std.debug.print("Epoch {d:2}/{d:2} | Train Loss: {d:.4} | Train Acc: {d:.2}% | Test Loss: {d:.4} | Test Acc: {d:.2}%\n", .{
            epoch + 1,
            epochs,
            epoch_loss,
            epoch_acc * 100.0,
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
    arena: std.mem.Allocator,
    test_dataset: dataset.Dataset,
) !nn.EpochResult {
    const test_batch_size = 100;


    var test_loader = try dataset.DataLoader.init(arena, test_dataset, test_batch_size, .{
        .shuffle = false,
        .drop_last = false,
    });
    defer test_loader.deinit(arena);

    return try nn.evaluate(arena, model, &test_loader);
}


fn printPredictions(
    model: anytype,
    arena: std.mem.Allocator,
    test_dataset: dataset.Dataset,
    count: usize,
) !void {
    const input_dim = 784;

    std.debug.print("\nSample CNN Predictions from Test Set:\n", .{});
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

test "CNN model initialization and forward passes (Eager & Graph)" {
    const allocator = std.testing.allocator;

    var model = NeuralNetwork.init(allocator, try CNN.init(allocator, 42));
    defer model.deinit();

    const x_data = try allocator.alloc(f32, 2 * 784);
    defer allocator.free(x_data);
    @memset(x_data, 0.1);

    // Test Eager Mode (graph == null)
    {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();

        const x_tensor = try tensor.array(arena_allocator, &.{ 2, 784 }, x_data);
        const logits = try model.forward(arena_allocator, null, x_tensor);

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
