const std = @import("std");
const zig_ml = @import("zig_ml");
const dataset = zig_ml.dataset;
const nn = zig_ml.nn;
const autodiff = zig_ml.autodiff;

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
    pub fn init(allocator: std.mem.Allocator, ni: usize, nh1: usize, nh2: usize, no: usize, seed: u64) !MLP {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();

        const fc1 = try nn.Linear.init(allocator, ni, nh1, random);
        errdefer fc1.deinit(allocator);

        const fc2 = try nn.Linear.init(allocator, nh1, nh2, random);
        errdefer fc2.deinit(allocator);

        const fc3 = try nn.Linear.init(allocator, nh2, no, random);
        errdefer fc3.deinit(allocator);

        return MLP{
            .fc1 = fc1,
            .fc2 = fc2,
            .fc3 = fc3,
        };
    }

    // 用户只需专注定义前向传播逻辑
    pub fn forward(self: *const MLP, graph: *autodiff.Graph, x: *autodiff.Tensor) !*autodiff.Tensor {
        const x1 = try self.fc1.forward(graph, x);
        const a1 = try graph.relu(x1);

        const x2 = try self.fc2.forward(graph, a1);
        const a2 = try graph.relu(x2);

        return try self.fc3.forward(graph, a2);
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

    const input_dim = train_dataset.images.rows * train_dataset.images.cols;
    const num_classes = CLASS_NAMES.len;

    std.debug.print("Initializing Standard Model (3-layer MLP: 784 -> 128 -> 64 -> 10)...\n", .{});
    var model = try NeuralNetwork.init(arena, input_dim, 128, 64, num_classes, 42);
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
    const num_classes = model.inner.fc3.weight.shape.dims[1];
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
            const logits = try model.forward(&graph, x_tensor);

            // 损失函数：交叉熵损失节点，附带 Softmax 概率
            const loss = try graph.softmaxCrossEntropy(logits, targets);

            // 提取批次的 Loss 标量与准确率
            const batch_loss = loss.data[0];
            const probs = loss.creator.?.context.SoftmaxCrossEntropy.probs;
            const batch_acc = computeAccuracy(actual_batch_size, num_classes, probs, targets);

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



fn computeAccuracy(B: usize, num_classes: usize, a3: []const f32, y: []const u8) f32 {
    var correct: usize = 0;
    for (0..B) |i| {
        const label = y[i];
        const a3_row = a3[i * num_classes .. (i + 1) * num_classes];
        var max_val = a3_row[0];
        var pred: usize = 0;
        for (1..num_classes) |j| {
            if (a3_row[j] > max_val) {
                max_val = a3_row[j];
                pred = j;
            }
        }
        if (pred == label) {
            correct += 1;
        }
    }
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(B));
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
    const num_classes = model.inner.fc3.weight.shape.dims[1];
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

        const logits = try model.forward(&graph, x_tensor);

        const loss = try graph.softmaxCrossEntropy(logits, targets);

        test_loss += loss.data[0];
        const probs = loss.creator.?.context.SoftmaxCrossEntropy.probs;
        test_acc += computeAccuracy(actual_batch_size, num_classes, probs, targets);
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
    const num_classes = model.inner.fc3.weight.shape.dims[1];

    std.debug.print("\nSample Predictions from Test Set:\n", .{});
    for (0..count) |idx| {
        const img_slice = test_dataset.images.data[idx * input_dim .. (idx + 1) * input_dim];
        const actual_label = test_dataset.labels.data[idx];

        var graph = autodiff.Graph.init(arena);
        defer graph.deinit();

        const x_tensor = try graph.tensor(1, input_dim, false);
        @memcpy(x_tensor.data, img_slice);

        const logits = try model.forward(&graph, x_tensor);

        const loss = try graph.softmaxCrossEntropy(logits, &[1]u8{actual_label});
        const probs = loss.creator.?.context.SoftmaxCrossEntropy.probs;

        var max_val = probs[0];
        var pred: usize = 0;
        for (1..num_classes) |j| {
            if (probs[j] > max_val) {
                max_val = probs[j];
                pred = j;
            }
        }

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
