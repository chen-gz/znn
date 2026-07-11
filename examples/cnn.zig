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

        // Layer 1: Input [Batch, 1, 28, 28] -> Conv1 (1->4, kernel 3) -> [Batch, 4, 26, 26] -> Pool -> [Batch, 4, 13, 13]
        const conv1 = try nn.Conv2D.init(allocator, 1, 4, 3, random);
        errdefer conv1.deinit(allocator);

        // Layer 2: Input [Batch, 4, 13, 13] -> Conv2 (4->8, kernel 3) -> [Batch, 8, 11, 11] -> Pool -> [Batch, 8, 5, 5]
        const conv2 = try nn.Conv2D.init(allocator, 4, 8, 3, random);
        errdefer conv2.deinit(allocator);

        // Layer 3: Input [Batch, 8, 5, 5] -> Conv3 (8->16, kernel 3) -> [Batch, 16, 3, 3] -> (No pooling) -> Flatten -> [Batch, 144]
        const conv3 = try nn.Conv2D.init(allocator, 8, 16, 3, random);
        errdefer conv3.deinit(allocator);

        // FC1: 144 input features -> 10 output classes.
        const fc1 = try nn.Linear.init(allocator, 144, 10, random);
        errdefer fc1.deinit(allocator);

        return CNN{
            .conv1 = conv1,
            .conv2 = conv2,
            .conv3 = conv3,
            .fc1 = fc1,
        };
    }

    pub fn forward(self: *const CNN, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *tensor.Tensor) !*tensor.Tensor {
        if (graph == null) {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const arena_allocator = arena.allocator();

            const batch_size = x.shape.dims[0];
            const x_reshaped = try x.reshape(&.{ batch_size, 1, 28, 28 }, arena_allocator, null);

            // Layer 1
            const x1 = try self.conv1.forward(arena_allocator, null, x_reshaped);
            const a1 = try x1.relu(arena_allocator, null);
            const p1 = try a1.maxpool2d(2, 2, arena_allocator, null);

            // Layer 2
            const x2 = try self.conv2.forward(arena_allocator, null, p1);
            const a2 = try x2.relu(arena_allocator, null);
            const p2 = try a2.maxpool2d(2, 2, arena_allocator, null);

            // Layer 3
            const x3 = try self.conv3.forward(arena_allocator, null, p2);
            const a3 = try x3.relu(arena_allocator, null);

            // Flatten and Linear
            const flat = try a3.reshape(&.{ batch_size, 144 }, arena_allocator, null);
            const out_arena = try self.fc1.forward(arena_allocator, null, flat);

            return try tensor.array(allocator, out_arena.shape.dims[0..out_arena.shape.len], out_arena.data);
        }

        const batch_size = x.shape.dims[0];
        const x_reshaped = try graph.?.reshape(x, &.{ batch_size, 1, 28, 28 });

        // Layer 1
        const x1 = try self.conv1.forward(allocator, graph, x_reshaped);
        const a1 = try graph.?.relu(x1);
        const p1 = try graph.?.maxpool2d(a1, 2, 2);

        // Layer 2
        const x2 = try self.conv2.forward(allocator, graph, p1);
        const a2 = try graph.?.relu(x2);
        const p2 = try graph.?.maxpool2d(a2, 2, 2);

        // Layer 3
        const x3 = try self.conv3.forward(allocator, graph, p2);
        const a3 = try graph.?.relu(x3);

        // Flatten and Linear
        const flat = try graph.?.reshape(a3, &.{ batch_size, 144 });
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
    var lr: f32 = 0.02;
    const beta: f32 = 0.9; // SGD momentum

    var train_loader = try dataset.DataLoader.init(arena, train_dataset, batch_size, .{
        .shuffle = true,
        .seed = 1337,
        .drop_last = true,
    });
    defer train_loader.deinit(arena);

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

            var graph = autodiff.Graph.init(arena);
            defer graph.deinit();

            const x_tensor = try graph.tensor(actual_batch_size, input_dim, false);
            _ = train_loader.nextInto(x_tensor.data, y_batch);
            const targets = y_batch[0..actual_batch_size];

            const logits = try model.forward(arena, &graph, x_tensor);
            const loss = try graph.softmaxCrossEntropy(logits, targets);

            const batch_loss = loss.data[0];
            const batch_acc = try computeAccuracy(logits, targets, arena);

            epoch_loss += batch_loss;
            epoch_acc += batch_acc;
            num_batches += 1;

            model.zeroGrad();
            try graph.backward(loss);
            model.updateWeights(lr, beta);

            // Print batch progress every 100 batches
            if (num_batches % 100 == 0) {
                std.debug.print("  Batch {d:4} | Loss: {d:.4} | Acc: {d:.2}%\n", .{
                    num_batches,
                    batch_loss,
                    batch_acc * 100.0,
                });
            }
        }

        epoch_loss /= @as(f32, @floatFromInt(num_batches));
        epoch_acc /= @as(f32, @floatFromInt(num_batches));

        const eval_res = try evaluateModel(model, arena, test_dataset);
        const test_loss = eval_res.loss;
        const test_acc = eval_res.acc;

        std.debug.print("Epoch {d:2}/{d:2} | Train Loss: {d:.4} | Train Acc: {d:.2}% | Test Loss: {d:.4} | Test Acc: {d:.2}%\n", .{
            epoch + 1,
            epochs,
            epoch_loss,
            epoch_acc * 100.0,
            test_loss,
            test_acc * 100.0,
        });

        lr *= 0.90;
    }

    std.debug.print("\nSaving trained CNN model to 'cnn_model.bin'...\n", .{});
    model.save(io, "cnn_model.bin") catch |err| {
        std.debug.print("Failed to save model: {}\n", .{err});
    };
}

fn computeAccuracy(logits: *tensor.Tensor, y: []const u8, allocator: std.mem.Allocator) !f32 {
    const preds = try logits.argmax(1, allocator);
    defer tensor.free(allocator, preds);

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
    const input_dim = 784;
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
