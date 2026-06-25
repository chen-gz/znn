const std = @import("std");
const zig_ml = @import("zig_ml");
const dataset = zig_ml.dataset;
const nn = zig_ml.nn;
const autodiff = zig_ml.autodiff;

const CLASS_NAMES = [10][]const u8{
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    std.debug.print("Running on CPU (Accelerate CBLAS sgemm)...\n", .{});

    std.debug.print("Loading dataset...\n", .{});
    var train_images = try dataset.loadImages(io, arena, "data/train-images-idx3-ubyte");
    defer train_images.deinit(arena);

    var train_labels = try dataset.loadLabels(io, arena, "data/train-labels-idx1-ubyte");
    defer train_labels.deinit(arena);

    var test_images = try dataset.loadImages(io, arena, "data/t10k-images-idx3-ubyte");
    defer test_images.deinit(arena);

    var test_labels = try dataset.loadLabels(io, arena, "data/t10k-labels-idx1-ubyte");
    defer test_labels.deinit(arena);

    std.debug.print("Loaded {} training images, {} test images.\n", .{ train_images.num_images, test_images.num_images });

    const input_dim = train_images.rows * train_images.cols;
    const num_classes = CLASS_NAMES.len;

    std.debug.print("Initializing Standard Model (3-layer MLP: 784 -> 128 -> 64 -> 10)...\n", .{});
    var model = try nn.NeuralNetwork.init(arena, input_dim, 128, 64, num_classes, 42);
    defer model.deinit();
    try runTraining(&model, arena, io, train_images, train_labels, test_images, test_labels);
}

fn runTraining(
    model: anytype,
    arena: std.mem.Allocator,
    io: std.Io,
    train_images: dataset.ImageDataset,
    train_labels: dataset.LabelDataset,
    test_images: dataset.ImageDataset,
    test_labels: dataset.LabelDataset,
) !void {
    const input_dim = train_images.rows * train_images.cols;
    const num_classes = CLASS_NAMES.len;
    const batch_size = 64;
    const test_batch_size = 100;
    const epochs = 15;
    var lr: f32 = 0.05;
    const beta: f32 = 0.9; // SGD momentum factor

    // Indices for shuffling
    const num_train = train_images.num_images;
    var train_indices = try arena.alloc(usize, num_train);
    defer arena.free(train_indices);
    for (0..num_train) |i| {
        train_indices[i] = i;
    }

    var prng = std.Random.DefaultPrng.init(1337);
    const random = prng.random();

    // Allocate batch buffers (we copy images into these contiguous buffers before wrapping as tensors)
    var x_batch = try arena.alloc(f32, batch_size * input_dim);
    defer arena.free(x_batch);
    var y_batch = try arena.alloc(u8, batch_size);
    defer arena.free(y_batch);

    var eval_x_batch = try arena.alloc(f32, test_batch_size * input_dim);
    defer arena.free(eval_x_batch);
    var eval_y_batch = try arena.alloc(u8, test_batch_size);
    defer arena.free(eval_y_batch);

    std.debug.print("Starting training (3-layer NN with dynamic autodiff)...\n", .{});

    // 开始主循环（15 个 Epoch 的 Fashion MNIST 训练）
    for (0..epochs) |epoch| {
        const start_time = std.Io.Clock.awake.now(io);

        // 随机打乱训练数据集的索引顺序
        shuffle(random, train_indices);

        var epoch_loss: f32 = 0.0;
        var epoch_acc: f32 = 0.0;
        const num_batches = num_train / batch_size; // 丢弃最后一个非整 Batch

        for (0..num_batches) |b| {
            const batch_start = b * batch_size;

            // 从数据集抓取并拼接一个 Batch 的输入数据和目标标签
            for (0..batch_size) |j| {
                const idx = train_indices[batch_start + j];
                @memcpy(x_batch[j * input_dim .. (j + 1) * input_dim], train_images.data[idx * input_dim .. (idx + 1) * input_dim]);
                y_batch[j] = train_labels.data[idx];
            }

            // 关键：初始化一个新的、生命周期处于当前 batch 内的局部计算图。
            // 使用 Arena 分配器，当前 batch 结束后通过 `defer graph.deinit()` 一次性自动释放所有中间层 Tensor 内存。
            var graph = autodiff.Graph.init(arena);
            defer graph.deinit();

            // 将 Batch 的输入数据封装为计算图中的 Tensor 节点（注意输入数据 requires_grad = false）
            const x_tensor = try graph.tensor(batch_size, input_dim, false);
            @memcpy(x_tensor.data, x_batch);

            // 执行前向传播构建动态计算图（类似于 PyTorch 的 model(x)）
            const logits = try model.forward(&graph, x_tensor);

            // 损失函数：交叉熵损失节点，附带 Softmax 概率
            const loss = try graph.softmaxCrossEntropy(logits, y_batch);

            // 提取批次的 Loss 标量与准确率
            const batch_loss = loss.data[0];
            const probs = loss.creator.?.context.SoftmaxCrossEntropy.probs;
            const batch_acc = computeAccuracy(batch_size, num_classes, probs, y_batch);

            epoch_loss += batch_loss;
            epoch_acc += batch_acc;

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
        var test_loss: f32 = 0.0;
        var test_acc: f32 = 0.0;
        const num_test = test_images.num_images;
        const num_test_batches = num_test / test_batch_size;

        for (0..num_test_batches) |b| {
            const batch_start = b * test_batch_size;

            // Prepare batch
            for (0..test_batch_size) |j| {
                const idx = batch_start + j;
                @memcpy(eval_x_batch[j * input_dim .. (j + 1) * input_dim], test_images.data[idx * input_dim .. (idx + 1) * input_dim]);
                eval_y_batch[j] = test_labels.data[idx];
            }

            var graph = autodiff.Graph.init(arena);
            defer graph.deinit();

            const x_tensor = try graph.tensor(test_batch_size, input_dim, false);
            @memcpy(x_tensor.data, eval_x_batch);

            const logits = try model.forward(&graph, x_tensor);

            const loss = try graph.softmaxCrossEntropy(logits, eval_y_batch);

            test_loss += loss.data[0];
            const probs = loss.creator.?.context.SoftmaxCrossEntropy.probs;
            test_acc += computeAccuracy(test_batch_size, num_classes, probs, eval_y_batch);
        }

        test_loss /= @as(f32, @floatFromInt(num_test_batches));
        test_acc /= @as(f32, @floatFromInt(num_test_batches));

        const elapsed_s = @as(f32, @floatFromInt(start_time.untilNow(io, .awake).toNanoseconds())) / 1_000_000_000.0;

        std.debug.print("Epoch {d:2}/{d:2} | Train Loss: {d:.4} | Train Acc: {d:.2}% | Test Loss: {d:.4} | Test Acc: {d:.2}% | Time: {d:.2}s\n", .{
            epoch + 1,
            epochs,
            epoch_loss,
            epoch_acc * 100.0,
            test_loss,
            test_acc * 100.0,
            elapsed_s,
        });

        // Learning rate decay
        lr *= 0.90;
    }

    // Save model parameters
    std.debug.print("\nSaving trained model to 'model.bin'...\n", .{});
    model.save(io, "model.bin") catch |err| {
        std.debug.print("Failed to save model: {}\n", .{err});
    };

    // Print sample predictions
    std.debug.print("\nSample Predictions from Test Set:\n", .{});
    for (0..5) |i| {
        _ = i;
        const idx = random.intRangeLessThan(usize, 0, test_images.num_images);
        const img_slice = test_images.data[idx * input_dim .. (idx + 1) * input_dim];
        const actual_label = test_labels.data[idx];

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

fn shuffle(random: std.Random, indices: []usize) void {
    var i: usize = indices.len - 1;
    while (i > 0) : (i -= 1) {
        const j = random.intRangeLessThan(usize, 0, i + 1);
        const temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
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
