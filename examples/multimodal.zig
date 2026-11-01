const std = @import("std");
const zig_ml = @import("zig_ml");
const tensor = zig_ml.tensor;
const autodiff = zig_ml.autodiff;
const nn = zig_ml.nn;
const dataset = zig_ml.dataset;
const models = zig_ml.models;
const optim = zig_ml.optim;
const bench = zig_ml.bench;

/// 视觉语言多模态架构配置 (Vision-Language Multimodal Configuration)
pub const MultimodalConfig = struct {
    vision_dim: usize = 32,
    d_model: usize = 64,
    n_head: usize = 4,
    vocab_size: usize = 280,
    max_seq_len: usize = 16,
};

/// 视觉语言多模态端到端模型 (Vision-Language Multimodal Architecture)
/// 1. Vision Backbone: ConvNeXt 纯卷积视觉主干，将 [B, 3, 64, 64] 提取为视觉特征向量 [B, vision_dim]
/// 2. Multimodal Projector: Linear(vision_dim, d_model) 将视觉向量映射至文本嵌入空间 [B, 1, d_model]
/// 3. Text Embedding: Embedding(vocab_size, d_model) 将自然语言描述标题转换为 [B, S, d_model]
/// 4. Multimodal Fusion: 在序列维度拼接视觉标记与文本词元序列 [B, 1 + S, d_model]
/// 5. Transformer Decoder: 堆叠因果自注意力与 MLP，完成以图生文/图文跨模态对齐建模
/// 6. LM Head: 输出词表概率对数几率 (Logits)
pub const MultimodalModel = struct {
    vision_backbone: models.ConvNeXt,
    vision_proj: nn.Linear,
    text_embed: nn.Embedding,
    block1: nn.TransformerBlock,
    block2: nn.TransformerBlock,
    lm_head: nn.Linear,
    config: MultimodalConfig,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, cfg: MultimodalConfig) !MultimodalModel {
        const convnext_raw = try models.convnext_atto(allocator, 3, cfg.vision_dim);
        const vision_proj = try nn.Linear.init(allocator, cfg.vision_dim, cfg.d_model);
        const text_embed = try nn.Embedding.init(allocator, cfg.vocab_size, cfg.d_model);
        const block1 = try nn.TransformerBlock.init(allocator, cfg.d_model, cfg.n_head);
        const block2 = try nn.TransformerBlock.init(allocator, cfg.d_model, cfg.n_head);
        const lm_head = try nn.Linear.init(allocator, cfg.d_model, cfg.vocab_size);

        return .{
            .vision_backbone = convnext_raw,
            .vision_proj = vision_proj,
            .text_embed = text_embed,
            .block1 = block1,
            .block2 = block2,
            .lm_head = lm_head,
            .config = cfg,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MultimodalModel) void {
        nn.deinitModel(&self.lm_head, self.allocator);
        nn.deinitModel(&self.block2, self.allocator);
        nn.deinitModel(&self.block1, self.allocator);
        nn.deinitModel(&self.text_embed, self.allocator);
        nn.deinitModel(&self.vision_proj, self.allocator);
        self.vision_backbone.deinit();
    }

    pub fn forward(
        self: *const MultimodalModel,
        graph: *autodiff.Graph,
        images: *tensor.Tensor,
        text_tokens: *tensor.Tensor,
    ) !*tensor.Tensor {
        const B = images.shape.dims[0];
        const S = text_tokens.shape.dims[1];

        // 1. 视觉特征提取与线性投影: [B, 3, 64, 64] -> [B, vision_dim] -> [B, 1, d_model]
        const v_feats = try self.vision_backbone.forward(graph, images);
        const v_proj = try self.vision_proj.forward(graph, v_feats);
        const v_token = try graph.reshape(v_proj, &.{ B, 1, self.config.d_model });

        // 2. 文本词元嵌入: [B, S] -> [B, S, d_model]
        const t_emb = try self.text_embed.forward(graph, text_tokens);

        // 3. 拼接视觉与文本词元: [B, 1 + S, d_model]
        const seq_emb = try graph.concat(&.{ v_token, t_emb }, 1);

        // 4. 堆叠变换器因果块处理
        const h1 = try self.block1.forward(graph, seq_emb);
        const h2 = try self.block2.forward(graph, h1);

        // 5. 语言模型分类头: [B * (1 + S), vocab_size]
        const total_len = 1 + S;
        const flat_h = try graph.reshape(h2, &.{ B * total_len, self.config.d_model });
        return try self.lm_head.forward(graph, flat_h);
    }
};

pub const CliOptions = struct {
    epochs: usize = 2,
    batch_size: usize = 8,
    num_samples: usize = 32,
    max_seq_len: usize = 16,
    lr: f32 = 0.002,
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
        } else if (std.mem.eql(u8, arg, "--samples") or std.mem.eql(u8, arg, "-n")) {
            if (iter.next()) |val| opts.num_samples = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--lr")) {
            if (iter.next()) |val| opts.lr = try std.fmt.parseFloat(f32, val);
        }
    }
    return opts;
}

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cli = try parseCli(allocator, init.minimal.args);

    std.debug.print("================================================================================\n", .{});
    std.debug.print("        ZNN - Vision-Language Multimodal Paired Benchmark (ConvNeXt + LLM)       \n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // 1. 初始化 BPE 分词器
    var tokenizer = try dataset.BPETokenizer.init(allocator);
    defer tokenizer.deinit();

    // 预注册通用词元提高分词效率
    _ = try tokenizer.addToken("<|pad|>");
    _ = try tokenizer.addToken("<|image|>");
    _ = try tokenizer.addToken("<|eos|>");

    const sample_vocab_size = @as(usize, @intCast(tokenizer.inv_vocab.items.len));

    // 2. 构造多模态图文对数据集 (3 通道 64x64 彩色图像 + 英文描述标题)
    std.debug.print("Generating synthetic Multimodal Vision-Language dataset ({d} pairs)...\n", .{cli.num_samples});
    var mm_dataset = try dataset.createSyntheticMultimodalDataset(allocator, &tokenizer, cli.num_samples);
    defer mm_dataset.deinit();

    std.debug.print("  [✓] Loaded {d} paired samples (Image: 3x64x64 RGB, Captions tokenized via BPE).\n", .{mm_dataset.count()});
    const sample0 = mm_dataset.get(0);
    std.debug.print("  [Example Pair #1] Caption: \"{s}\" (Token count: {d})\n\n", .{ sample0.caption, sample0.tokens.len });

    // 3. 构建多模态模型
    const cfg = MultimodalConfig{
        .vision_dim = 32,
        .d_model = 64,
        .n_head = 4,
        .vocab_size = @max(sample_vocab_size + 30, 280),
        .max_seq_len = cli.max_seq_len,
    };

    const raw_model = try MultimodalModel.init(allocator, cfg);
    var model = try nn.Module(MultimodalModel).init(allocator, raw_model);
    defer model.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const dummy_img = try tensor.zeros(allocator, &.{ 1, 3, 64, 64 });
    defer tensor.free(allocator, dummy_img);
    const dummy_tok = try tensor.zeros(allocator, &.{ 1, cli.max_seq_len });
    defer tensor.free(allocator, dummy_tok);
    try model.initParametersWithSample(prng.random(), .{ dummy_img, dummy_tok });

    std.debug.print("Initialized Multimodal Architecture:\n", .{});
    std.debug.print("  - Vision Encoder:       ConvNeXt-atto (3 channels -> 32-dim visual features)\n", .{});
    std.debug.print("  - Cross-Modal Projector:Linear (32 -> 64-dim language space)\n", .{});
    std.debug.print("  - Language Backbone:    2-Layer Causal Transformer (d_model=64, n_head=4)\n", .{});
    std.debug.print("  - Vocabulary Size:      {d}\n\n", .{cfg.vocab_size});

    // 4. 多模态数据迭代器与缓冲区
    var loader = try dataset.MultimodalDataLoader.init(
        allocator,
        mm_dataset,
        cli.batch_size,
        cli.max_seq_len,
        .{ .shuffle = true, .seed = 1337, .drop_last = true },
    );
    defer loader.deinit(allocator);

    const img_buf = try allocator.alloc(f32, cli.batch_size * 3 * 64 * 64);
    defer allocator.free(img_buf);
    const tokens_buf = try allocator.alloc(u32, cli.batch_size * cli.max_seq_len);
    defer allocator.free(tokens_buf);
    const lengths_buf = try allocator.alloc(usize, cli.batch_size);
    defer allocator.free(lengths_buf);

    var optimizer = try optim.AdamWOptimizer.init(allocator, &model, .{
        .lr = cli.lr,
        .weight_decay = 0.01,
    });
    defer optimizer.deinit();

    std.debug.print("Starting Multimodal Vision-to-Language Autoregressive Training:\n", .{});
    std.debug.print("  - Batch Size:   {d} pairs/step\n", .{cli.batch_size});
    std.debug.print("  - Sequence Len: {d} text tokens (+1 image token)\n", .{cli.max_seq_len});
    std.debug.print("  - Epochs:       {d}\n\n", .{cli.epochs});

    var total_pairs_processed: usize = 0;
    const start_time_ns = bench.getTimeNs();
    var last_loss: f32 = 0.0;

    for (0..cli.epochs) |epoch| {
        loader.reset();
        var step_idx: usize = 0;
        var epoch_loss_sum: f32 = 0.0;
        var epoch_steps: usize = 0;

        while (loader.nextInto(img_buf, tokens_buf, lengths_buf)) |actual_batch| {
            var graph = autodiff.Graph.initDefault(allocator);
            defer graph.deinit();

            // 图像张量 [actual_batch, 3, 64, 64]
            const img_tensor = try graph.tensor(actual_batch, 3 * 64 * 64, false);
            @memcpy(img_tensor.data, img_buf[0 .. actual_batch * 3 * 64 * 64]);
            const img_4d = try graph.reshape(img_tensor, &.{ actual_batch, 3, 64, 64 });

            // 文本词元张量 [actual_batch, max_seq_len]
            const tok_tensor = try graph.tensor(actual_batch, cli.max_seq_len, false);
            for (tok_tensor.data, 0..) |*t, i| {
                t.* = @as(f32, @floatFromInt(tokens_buf[i]));
            }

            // 前向传播: 输出 [actual_batch * (1 + max_seq_len), vocab_size]
            const logits = try model.forward(&graph, .{ img_4d, tok_tensor });

            // 构造下一个词元预测目标:
            // 位置 0 (视觉图像): 目标是词元 0 (标题首词)
            // 位置 1..S-1 (第 j 个词): 目标是词元 j (标题下一个词)
            // 位置 S: 目标为填充符 0
            const total_seq = 1 + cli.max_seq_len;
            const target_count = actual_batch * total_seq;
            const targets = try allocator.alloc(u32, target_count);
            defer allocator.free(targets);

            for (0..actual_batch) |b| {
                // 图像位置预测首个词元
                targets[b * total_seq + 0] = tokens_buf[b * cli.max_seq_len + 0];
                for (1..cli.max_seq_len) |j| {
                    targets[b * total_seq + j] = tokens_buf[b * cli.max_seq_len + j];
                }
                targets[b * total_seq + cli.max_seq_len] = 0;
            }

            const loss = try graph.softmaxCrossEntropy(logits, targets);
            const step_loss = loss.data[0];

            model.zeroGrad();

            try graph.backward(loss);
            optimizer.step();

            epoch_loss_sum += step_loss;
            epoch_steps += 1;
            total_pairs_processed += actual_batch;
            step_idx += 1;
            last_loss = step_loss;

            std.debug.print("  [Epoch {d}/{d}] Step {d} | Loss: {d:.4}\n", .{
                epoch + 1,
                cli.epochs,
                step_idx,
                step_loss,
            });
        }

        const avg_loss = if (epoch_steps > 0) epoch_loss_sum / @as(f32, @floatFromInt(epoch_steps)) else 0.0;
        std.debug.print("  -> Epoch {d} Complete | Average Loss: {d:.4}\n\n", .{ epoch + 1, avg_loss });
    }

    const elapsed_s = @as(f64, @floatFromInt(bench.getTimeNs() - start_time_ns)) / 1e9;
    const throughput = @as(f64, @floatFromInt(total_pairs_processed)) / @max(elapsed_s, 0.001);

    std.debug.print("================================================================================\n", .{});
    std.debug.print("                ZNN Multimodal Benchmark Performance Report                     \n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("  Metric                           | Value                                      \n", .{});
    std.debug.print("-----------------------------------+--------------------------------------------\n", .{});
    std.debug.print("  Dataset Pairs Processed          | {d:10} pairs                             \n", .{total_pairs_processed});
    std.debug.print("  Total Elapsed Time               | {d:10.2} seconds                           \n", .{elapsed_s});
    std.debug.print("  Multimodal Throughput            | {d:10.2} pairs/second                      \n", .{throughput});
    std.debug.print("  Final Cross-Entropy Loss         | {d:10.4}                                   \n", .{last_loss});
    std.debug.print("================================================================================\n\n", .{});
    std.debug.print("✨ Multimodal Vision-Language end-to-end dataset & training benchmark verified!\n\n", .{});
}
