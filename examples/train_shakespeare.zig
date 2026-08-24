const std = @import("std");
const zig_ml = @import("zig_ml");
const nn = zig_ml.nn;
const autodiff = zig_ml.autodiff;
const dataset = zig_ml.dataset;
const optim = zig_ml.optim;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    std.debug.print("=========================================================\n", .{});
    std.debug.print("   Training Mini GPT on TinyShakespeare with znn         \n", .{});
    std.debug.print("=========================================================\n\n", .{});

    // 1. 读取 TinyShakespeare 文本
    const file_path = "data/tinyshakespeare.txt";
    const cwd = std.Io.Dir.cwd();
    var file = cwd.openFile(io, file_path, .{}) catch |err| {
        std.debug.print("⚠️  Could not find '{s}' ({s}).\n", .{ file_path, @errorName(err) });
        std.debug.print("👉 Please run: zig build download-dataset -- tinyshakespeare\n", .{});
        return;
    };
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);
    const reader = &file_reader.interface;
    const file_len = try file.length(io);
    const text = try allocator.alloc(u8, file_len);
    defer allocator.free(text);
    try reader.readSliceAll(text);

    std.debug.print("📖 Loaded corpus: {} bytes from {s}\n", .{ text.len, file_path });

    // 2. 初始化 Byte-level BPE 分词器 (纯 256 字节词表)
    std.debug.print("🔤 Tokenizing corpus with Byte-level BPETokenizer (vocab_size=256)...\n", .{});
    var tokenizer = try dataset.BPETokenizer.init(allocator);
    defer tokenizer.deinit();

    const sample_len = @min(text.len, 100_000);
    const tokens = try tokenizer.encode(allocator, text[0..sample_len]);
    defer allocator.free(tokens);
    std.debug.print("✨ Encoded {} tokens from {} bytes\n\n", .{ tokens.len, sample_len });

    // 3. 构建模型配置与初始化
    const block_size: usize = 64;
    const batch_size: usize = 8;
    const vocab_size: usize = 256; // 256 raw byte tokens

    const gpt_config = nn.GPTConfig{
        .vocab_size = vocab_size,
        .block_size = block_size,
        .n_embd = 64,
        .n_head = 4,
        .n_layer = 4,
    };

    std.debug.print("🏗️  Initializing GPT (vocab={}, block={}, embd={}, head={}, layers={})...\n", .{
        gpt_config.vocab_size,
        gpt_config.block_size,
        gpt_config.n_embd,
        gpt_config.n_head,
        gpt_config.n_layer,
    });

    const GPTModule = nn.Module(nn.GPT(gpt_config));
    var model = GPTModule.init(allocator, try nn.GPT(gpt_config).init(allocator, random));
    defer model.deinit();

    // 4. 优化器与学习率调度器
    var optimizer = try optim.AdamWOptimizer.init(allocator, &model, .{
        .lr = 2e-3,
        .beta1 = 0.9,
        .beta2 = 0.95,
        .eps = 1e-8,
        .weight_decay = 0.01,
    });
    defer optimizer.deinit();

    const max_steps: usize = 200;
    const scheduler = optim.CosineScheduler.init(2e-3, 2e-4, 15, max_steps);

    var train_dataset = dataset.BinaryMmapDataset.fromSlice(tokens, block_size);
    defer train_dataset.close();

    std.debug.print("🚀 Starting training for {} steps (Batch Size={}, Context Length={})...\n\n", .{ max_steps, batch_size, block_size });

    var start_ts: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &start_ts);

    const x_input = try allocator.alloc(f32, batch_size * block_size);
    defer allocator.free(x_input);
    const y_target = try allocator.alloc(u8, batch_size * block_size);
    defer allocator.free(y_target);

    for (0..max_steps) |step| {
        const offset = (step * batch_size * block_size) % (tokens.len - batch_size * block_size - 2);
        const batch = train_dataset.getBatch(offset, batch_size);

        for (0..batch_size * block_size) |i| {
            x_input[i] = @as(f32, @floatFromInt(batch.x[i]));
            y_target[i] = @as(u8, @intCast(batch.y[i] % vocab_size));
        }

        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();

        const x_node = try graph.tensorND(&.{ batch_size, block_size }, false);
        @memcpy(x_node.data, x_input);

        // 前向传播
        const logits = try model.forward(allocator, &graph, x_node);
        const logits_2d = try graph.reshape(logits, &.{ batch_size * block_size, vocab_size });
        const loss = try graph.softmaxCrossEntropy(logits_2d, y_target);

        // 反向传播
        model.zeroGrad();
        try graph.backward(loss);

        _ = optim.clipGradNorm(optimizer.params, 1.0);
        const cur_lr = scheduler.getLR(step);
        optimizer.stepWithLR(cur_lr);

        if ((step + 1) % 25 == 0 or step == 0) {
            std.debug.print("  Step {:3}/{} | Loss: {d:.4} | LR: {d:.5}\n", .{
                step + 1,
                max_steps,
                loss.data[0],
                cur_lr,
            });
        }
    }

    var end_ts: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &end_ts);
    const elapsed_s = @as(f64, @floatFromInt(end_ts.sec - start_ts.sec)) + @as(f64, @floatFromInt(end_ts.nsec - start_ts.nsec)) / 1e9;
    std.debug.print("\n✅ Training finished in {d:.2}s!\n\n", .{elapsed_s});

    // 5. 实时文本生成 (Autoregressive Generation with Top-P Sampling)
    std.debug.print("🎭 Sampling text from trained model (Prompt: \"KING:\")...\n", .{});
    const prompt_text = "KING:";
    const prompt_tokens = try tokenizer.encode(allocator, prompt_text);
    defer allocator.free(prompt_tokens);

    var gen_tokens: std.ArrayList(u32) = .empty;
    defer gen_tokens.deinit(allocator);
    try gen_tokens.appendSlice(allocator, prompt_tokens);

    for (0..120) |_| {
        const cur_len = @min(gen_tokens.items.len, block_size);
        const start_idx = gen_tokens.items.len - cur_len;
        const cur_slice = gen_tokens.items[start_idx..];

        var g = autodiff.Graph.init(allocator);
        defer g.deinit();

        const x_eval = try g.tensorND(&.{ 1, cur_len }, false);
        for (cur_slice, 0..) |tok, i| {
            x_eval.data[i] = @as(f32, @floatFromInt(tok));
        }

        const logits = try model.forward(allocator, &g, x_eval);
        const last_logits = logits.data[(cur_len - 1) * vocab_size .. cur_len * vocab_size];

        const next_token = try nn.sampleTopP(last_logits, vocab_size, 0.7, 0.85, random, allocator);
        try gen_tokens.append(allocator, next_token);
    }

    const generated_text = try tokenizer.decode(allocator, gen_tokens.items);
    defer allocator.free(generated_text);
    std.debug.print("\n--- Generated Output ---\n{s}\n------------------------\n", .{generated_text});
}
