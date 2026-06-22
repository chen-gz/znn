const std = @import("std");
const zig_ml = @import("zig_ml");
const dataset = zig_ml.dataset;
const nn = zig_ml.nn;
const autodiff = zig_ml.autodiff;

const CLASS_NAMES = [10][]const u8{
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
};

pub fn main(init: std.process.Init) !void {
    // Zig 进程初始化提供的 arena 分配器，用于进程级的内存分配
    const arena = init.arena.allocator();
    const io = init.io;

    // 解析命令行参数，检查是否传入了 --gpu 开关
    const args = try init.minimal.args.toSlice(arena);
    var run_gpu = false;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--gpu")) {
            run_gpu = true;
        }
    }
    // 设置 autodiff 引擎中的全局模式标志
    autodiff.use_gpu = run_gpu;

    if (run_gpu) {
        // 如果是 GPU 模式，初始化 macOS Metal Compute Shader 后端
        const rc = autodiff.metal_init();
        if (rc != 0) {
            std.debug.print("Failed to initialize Metal GPU backend: {}\n", .{rc});
            return error.MetalInitializationFailed;
        }
        std.debug.print("Running on GPU (Metal)...\n", .{});
    } else {
        // 否则，默认运行在 CPU 上，链接 Apple Accelerate 框架的 CBLAS 高性能库
        std.debug.print("Running on CPU (Accelerate CBLAS)...\n", .{});
    }

    // 初始化多线程池（用于加速需要并行的 CPU 算子，尽管本框架大部分高性能算子已交给 CBLAS/GPU，此设计依然提供了线程管理模板）
    try autodiff.initThreadPool();
    defer autodiff.deinitThreadPool();

    std.debug.print("Loading dataset...\n", .{});
    var train_images = dataset.loadImages(io, arena, "data/train-images-idx3-ubyte") catch |err| {
        std.debug.print("Failed to load training images: {}\n", .{err});
        return err;
    };
    defer train_images.deinit(arena);

    var train_labels = dataset.loadLabels(io, arena, "data/train-labels-idx1-ubyte") catch |err| {
        std.debug.print("Failed to load training labels: {}\n", .{err});
        return err;
    };
    defer train_labels.deinit(arena);

    var test_images = dataset.loadImages(io, arena, "data/t10k-images-idx3-ubyte") catch |err| {
        std.debug.print("Failed to load test images: {}\n", .{err});
        return err;
    };
    defer test_images.deinit(arena);

    var test_labels = dataset.loadLabels(io, arena, "data/t10k-labels-idx1-ubyte") catch |err| {
        std.debug.print("Failed to load test labels: {}\n", .{err});
        return err;
    };
    defer test_labels.deinit(arena);

    std.debug.print("Loaded {} training images, {} test images.\n", .{train_images.num_images, test_images.num_images});

    const ni = 784; // 28 * 28 pixels
    const nh1 = 128;
    const nh2 = 64;
    const no = 10; // 10 classes
    const batch_size = 64;
    const test_batch_size = 100;
    const epochs = 15;
    var lr: f32 = 0.05;
    const beta: f32 = 0.9; // SGD momentum factor

    var model = try nn.NeuralNetwork.init(arena, ni, nh1, nh2, no, 42);
    defer model.deinit();

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
    var x_batch = try arena.alloc(f32, batch_size * ni);
    defer arena.free(x_batch);
    var y_batch = try arena.alloc(u8, batch_size);
    defer arena.free(y_batch);

    var eval_x_batch = try arena.alloc(f32, test_batch_size * ni);
    defer arena.free(eval_x_batch);
    var eval_y_batch = try arena.alloc(u8, test_batch_size);
    defer arena.free(eval_y_batch);

    std.debug.print("Starting training (3-layer NN with dynamic autodiff: {} -> {} -> {} -> {})...\n", .{ni, nh1, nh2, no});

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
                @memcpy(
                    x_batch[j * ni .. (j + 1) * ni],
                    train_images.data[idx * ni .. (idx + 1) * ni]
                );
                y_batch[j] = train_labels.data[idx];
            }

            // 关键：初始化一个新的、生命周期处于当前 batch 内的局部计算图。
            // 使用 Arena 分配器，当前 batch 结束后通过 `defer graph.deinit()` 一次性自动释放所有中间层 Tensor 内存。
            var graph = autodiff.Graph.init(arena);
            defer graph.deinit();

            // 将 Batch 的输入数据封装为计算图中的 Tensor 节点（注意输入数据 requires_grad = false）
            const x_tensor = try graph.tensor(batch_size, ni, false);
            @memcpy(x_tensor.data, x_batch);

            // 执行前向传播构建动态计算图：
            // 第一层：线性变换 MatMul -> 偏置加法 AddBias -> 激活函数 ReLU
            const z1 = try graph.matmul(x_tensor, model.w1);
            const z1_bias = try graph.addBias(z1, model.b1);
            const a1 = try graph.relu(z1_bias);

            // 第二层：线性变换 MatMul -> 偏置加法 AddBias -> 激活函数 ReLU
            const z2 = try graph.matmul(a1, model.w2);
            const z2_bias = try graph.addBias(z2, model.b2);
            const a2 = try graph.relu(z2_bias);

            // 第三层：线性输出层 MatMul -> 偏置加法 AddBias (Logits)
            const z3 = try graph.matmul(a2, model.w3);
            const logits = try graph.addBias(z3, model.b3);

            // 损失函数：交叉熵损失节点，附带 Softmax 概率
            const loss = try graph.softmaxCrossEntropy(logits, y_batch);

            // 提取批次的 Loss 标量与准确率
            const batch_loss = loss.data[0];
            const probs = loss.creator.?.context.SoftmaxCrossEntropy.probs;
            const batch_acc = computeAccuracy(batch_size, no, probs, y_batch);

            epoch_loss += batch_loss;
            epoch_acc += batch_acc;

            // 1. 在反向传播前，清空神经网络模型中持久化权重的梯度值 (w.grad = 0)
            model.zeroGrad();

            // 2. 触发反向传播，自 loss 节点开始，通过拓扑排序逆序逐个算子求导，填充各 Tensor 的 grad 梯度缓冲区
            try graph.backward(loss);

            // 3. 执行权重更新：采用带 Momentum 动量的随机梯度下降（SGD）更新网络权重与偏置项
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
                @memcpy(
                    eval_x_batch[j * ni .. (j + 1) * ni],
                    test_images.data[idx * ni .. (idx + 1) * ni]
                );
                eval_y_batch[j] = test_labels.data[idx];
            }

            var graph = autodiff.Graph.init(arena);
            defer graph.deinit();

            const x_tensor = try graph.tensor(test_batch_size, ni, false);
            @memcpy(x_tensor.data, eval_x_batch);

            const z1 = try graph.matmul(x_tensor, model.w1);
            const z1_bias = try graph.addBias(z1, model.b1);
            const a1 = try graph.relu(z1_bias);

            const z2 = try graph.matmul(a1, model.w2);
            const z2_bias = try graph.addBias(z2, model.b2);
            const a2 = try graph.relu(z2_bias);

            const z3 = try graph.matmul(a2, model.w3);
            const logits = try graph.addBias(z3, model.b3);

            const loss = try graph.softmaxCrossEntropy(logits, eval_y_batch);

            test_loss += loss.data[0];
            const probs = loss.creator.?.context.SoftmaxCrossEntropy.probs;
            test_acc += computeAccuracy(test_batch_size, no, probs, eval_y_batch);
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
        const img_slice = test_images.data[idx * ni .. (idx + 1) * ni];
        const actual_label = test_labels.data[idx];

        var graph = autodiff.Graph.init(arena);
        defer graph.deinit();

        const x_tensor = try graph.tensor(1, ni, false);
        @memcpy(x_tensor.data, img_slice);

        const z1 = try graph.matmul(x_tensor, model.w1);
        const z1_bias = try graph.addBias(z1, model.b1);
        const a1 = try graph.relu(z1_bias);

        const z2 = try graph.matmul(a1, model.w2);
        const z2_bias = try graph.addBias(z2, model.b2);
        const a2 = try graph.relu(z2_bias);

        const z3 = try graph.matmul(a2, model.w3);
        const logits = try graph.addBias(z3, model.b3);

        const loss = try graph.softmaxCrossEntropy(logits, &[1]u8{actual_label});
        const probs = loss.creator.?.context.SoftmaxCrossEntropy.probs;

        var max_val = probs[0];
        var pred: usize = 0;
        for (1..no) |j| {
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

fn computeAccuracy(B: usize, no: usize, a3: []const f32, y: []const u8) f32 {
    var correct: usize = 0;
    for (0..B) |i| {
        const label = y[i];
        const a3_row = a3[i * no .. (i + 1) * no];
        var max_val = a3_row[0];
        var pred: usize = 0;
        for (1..no) |j| {
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
