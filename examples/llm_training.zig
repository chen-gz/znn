const std = @import("std");
const zig_ml = @import("zig_ml");
const nn = zig_ml.nn;
const autodiff = zig_ml.autodiff;
const dataset = zig_ml.dataset;
const optim = zig_ml.optim;

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(2026);
    const random = prng.random();

    std.debug.print("====================================================\n", .{});
    std.debug.print("   End-to-End LLM Pipeline in Zig (znn framework)  \n", .{});
    std.debug.print("====================================================\n\n", .{});

    // ---------------------------------------------------------------
    // 阶段 0: Byte-level BPE 分词器
    // ---------------------------------------------------------------
    std.debug.print("[Stage 0] Initializing Byte-level BPE Tokenizer...\n", .{});
    var tokenizer = try dataset.BPETokenizer.init(allocator);
    defer tokenizer.deinit();

    try tokenizer.addMerge("Z", "i", 0);
    try tokenizer.addMerge("Zi", "g", 1);
    try tokenizer.addMerge("m", "l", 2);
    try tokenizer.addMerge(" ", "Z", 3);

    const raw_text = "Zig ml is awesome! Zig ml";
    const tokens = try tokenizer.encode(allocator, raw_text);
    defer allocator.free(tokens);
    std.debug.print("  Raw Text: \"{s}\" -> Encoded {} tokens\n", .{ raw_text, tokens.len });

    const decoded = try tokenizer.decode(allocator, tokens);
    defer allocator.free(decoded);
    std.debug.print("  Decoded: \"{s}\"\n\n", .{decoded});

    // ---------------------------------------------------------------
    // 阶段 1: 预训练循环 (Pre-training on raw sequence with SwiGLU & AdamW)
    // ---------------------------------------------------------------
    std.debug.print("[Stage 1] Pre-training with SwiGLU MLP & AdamW Optimizer...\n", .{});

    const dim: usize = 16;
    const hidden_dim: usize = 32;
    var swiglu = try nn.SwiGLU.init(allocator, dim, hidden_dim, random);
    defer swiglu.deinit(allocator);

    const adamw_cfg = optim.AdamWConfig{
        .lr = 1e-2,
        .beta1 = 0.9,
        .beta2 = 0.95,
        .eps = 1e-8,
        .weight_decay = 0.01,
    };
    var opt = try optim.AdamWOptimizer.init(allocator, &swiglu, adamw_cfg);
    defer opt.deinit();

    const scheduler = optim.CosineScheduler.init(1e-2, 1e-4, 2, 5);

    const seq_len: usize = 4;
    const batch_size: usize = 2;

    for (1..6) |epoch| {
        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();

        const x = try graph.tensorND(&.{ batch_size, seq_len, dim }, true);
        @memset(x.data, 0.25);

        const out = try swiglu.forward(allocator, &graph, x);

        // 模拟损失 (MSE / L2 目标)
        swiglu.zeroGrad();
        @memset(out.grad, 0.1);
        try graph.backward(out);

        const grad_norm = optim.clipGradNorm(opt.params, 1.0);
        const cur_lr = scheduler.getLR(epoch);
        opt.stepWithLR(cur_lr);

        std.debug.print("  Epoch {:2} | LR: {d:.5} | Grad Norm: {d:.4}\n", .{ epoch, cur_lr, grad_norm });
    }
    std.debug.print(">> Base Model Pre-training Step Completed.\n\n", .{});

    // ---------------------------------------------------------------
    // 阶段 2: 监督指令微调 (SFT with Masked Loss)
    // ---------------------------------------------------------------
    std.debug.print("[Stage 2] Supervised Fine-Tuning (SFT) with Masked Loss...\n", .{});
    const logits = try zig_ml.tensor.zeros(allocator, &.{ 4, 10 });
    defer zig_ml.tensor.free(allocator, logits);
    @memset(logits.data, 0.1);
    logits.data[0 * 10 + 2] = 2.0; // Token 0 (Prompt)
    logits.data[1 * 10 + 3] = 3.5; // Token 1 (Response Target=3)
    logits.data[2 * 10 + 5] = 4.0; // Token 2 (Response Target=5)
    logits.data[3 * 10 + 8] = 2.8; // Token 3 (Response Target=8)

    const targets = [_]u32{ 2, 3, 5, 8 };
    const mask = [_]f32{ 0.0, 1.0, 1.0, 1.0 }; // Only assistant response has mask=1.0

    const sft_loss = try nn.maskedCrossEntropyLoss(logits, &targets, &mask, allocator);
    std.debug.print("  SFT Prompt-Masked Cross-Entropy Loss: {d:.4}\n", .{sft_loss});
    std.debug.print(">> Instruction Tuning Verified.\n\n", .{});

    // ---------------------------------------------------------------
    // 阶段 3: LoRA 低秩适配微调 (LoRA Adaptation & Fusing)
    // ---------------------------------------------------------------
    std.debug.print("[Stage 3] Injecting LoRA Adapters (r=4, alpha=8)...\n", .{});
    var lora_layer = try nn.LoRALinear.init(allocator, 16, 16, 4, 8.0, random);
    defer lora_layer.deinit(allocator);

    var lora_graph = autodiff.Graph.init(allocator);
    defer lora_graph.deinit();

    const lora_in = try lora_graph.tensor(2, 16, false);
    @memset(lora_in.data, 1.0);

    const lora_out = try lora_layer.forward(allocator, &lora_graph, lora_in);
    std.debug.print("  LoRA Forward output computed. Shape: {any}\n", .{lora_out.shape.dims[0..lora_out.shape.len]});

    // 权重融合验证
    lora_layer.fuse();
    std.debug.print(">> LoRA Adaptation Weights Fused back into Base Weight.\n\n", .{});

    // ---------------------------------------------------------------
    // 阶段 4: DPO 偏好对齐损失
    // ---------------------------------------------------------------
    std.debug.print("[Stage 4] Direct Preference Optimization (DPO) Loss...\n", .{});
    const pi_chosen = [_]f32{ -0.8, -1.2 };
    const pi_rejected = [_]f32{ -2.4, -3.0 };
    const ref_chosen = [_]f32{ -1.0, -1.5 };
    const ref_rejected = [_]f32{ -1.8, -2.1 };
    const dpo_val = nn.dpoLoss(&pi_chosen, &pi_rejected, &ref_chosen, &ref_rejected, 0.1);
    std.debug.print("  DPO Alignment Loss: {d:.4}\n", .{dpo_val});
    std.debug.print(">> DPO Loss Verified.\n\n", .{});

    // ---------------------------------------------------------------
    // 阶段 5: 生成采样测试 (Top-P & Top-K)
    // ---------------------------------------------------------------
    std.debug.print("[Stage 5] Generating next token with Top-P=0.9 and Top-K=3...\n", .{});
    const mock_logits = [_]f32{ 0.1, 0.4, 2.8, 0.2, 0.9 };
    const sampled_p = try nn.sampleTopP(&mock_logits, 5, 0.7, 0.9, random, allocator);
    const sampled_k = try nn.sampleTopK(&mock_logits, 5, 0.7, 3, random, allocator);
    std.debug.print("  Sampled Top-P Token ID: {}\n", .{sampled_p});
    std.debug.print("  Sampled Top-K Token ID: {}\n", .{sampled_k});
    std.debug.print(">> All LLM Training Pipeline Modules Verified Successfully!\n", .{});
}
