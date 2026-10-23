const std = @import("std");
const backend = @import("../backend.zig");
const autodiff = @import("../autodiff.zig");
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");
const models = @import("../models.zig");

const Tensor = tensor.Tensor;
const gemma4 = models.gemma4;
const gemma4_q4 = models.gemma4_q4;
const gemma4_e4b = models.gemma4_e4b;
const gemma4_gpu = models.gemma4_gpu;
const sampling = backend.sampling;
const Gemma4E4BModel = gemma4_e4b.Gemma4E4BModel;

test "Gemma 4 specialized operators, modules, and CausalLM forward" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // 1. GemmaRMSNorm 单元测试 (含 unit offset 1.0 + weight 特性)
    {
        var norm = try gemma4.GemmaRMSNorm.init(allocator, 4, 1e-6);
        defer nn.deinitModel(&norm, allocator);
        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();

        const x = try g.tensorNDWithData(&.{ 1, 4 }, &.{ 2.0, 2.0, 2.0, 2.0 }, false);
        const y = try norm.forward(&g, x);
        // x rms = 2.0, x / rms = 1.0, weight 为 1.0, 结果为 1.0
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), y.data[0], 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), y.data[3], 1e-4);
    }

    // 2. Gemma 专用 Split-Half RoPE 算子单测
    {
        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 1, 2, 8 }, false);
        const x_rot = try gemma4.ropeSplitHalf(&g, x, 0, 1.0, 10000.0);
        try std.testing.expectEqual(@as(usize, 3), x_rot.shape.len);
        try std.testing.expectEqual(@as(usize, 8), x_rot.shape.dims[2]);
    }

    // 3. Gemma4MLP 单元测试
    {
        var mlp = try gemma4.Gemma4MLP.init(allocator, 8, 16);
        mlp.gate_proj.resetParameters(random, .{});
        mlp.up_proj.resetParameters(random, .{});
        mlp.down_proj.resetParameters(random, .{});
        defer nn.deinitModel(&mlp, allocator);

        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 2, 8 }, false);
        const out = try mlp.forward(&g, x);
        try std.testing.expectEqual(@as(usize, 2), out.shape.len);
        try std.testing.expectEqual(@as(usize, 2), out.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 8), out.shape.dims[1]);
    }

    // 4. Gemma4Attention 单元测试 (含 Q/K-Norm, V-Norm 与 GQA)
    {
        var attn = try gemma4.Gemma4Attention.init(
            allocator,
            .sliding_attention,
            16, // hidden_size
            4,  // num_heads
            2,  // num_kv_heads
            4,  // head_dim
            2,  // sliding_window
            1e-6,
        );
        attn.q_proj.resetParameters(random, .{});
        attn.k_proj.resetParameters(random, .{});
        if (attn.v_proj) |*vp| vp.resetParameters(random, .{});
        attn.o_proj.resetParameters(random, .{});
        defer nn.deinitModel(&attn, allocator);

        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 1, 3, 16 }, false);
        const y = try attn.forward(&g, x);
        try std.testing.expectEqual(@as(usize, 3), y.shape.len);
        try std.testing.expectEqual(@as(usize, 1), y.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 3), y.shape.dims[1]);
        try std.testing.expectEqual(@as(usize, 16), y.shape.dims[2]);
    }

    // 5. 端到端 TinyGemma4 模型测试
    {
        var model = try gemma4.TinyGemma4.init(allocator);
        defer nn.deinitModel(&model, allocator);

        var names = std.heap.ArenaAllocator.init(allocator);
        defer names.deinit();
        try nn.nameModules(&model, names.allocator(), "gemma4");

        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();

        const token_ids = try g.zeros(&.{ 2, 4 }, false);
        for (token_ids.data, 0..) |*val, idx| {
            val.* = @as(f32, @floatFromInt(idx % 10));
        }

        const logits = try model.forward(&g, token_ids);
        try std.testing.expectEqual(@as(usize, 3), logits.shape.len);
        try std.testing.expectEqual(@as(usize, 2), logits.shape.dims[0]); // B
        try std.testing.expectEqual(@as(usize, 4), logits.shape.dims[1]); // T
        try std.testing.expectEqual(@as(usize, 128), logits.shape.dims[2]); // V (vocab_size)

        // 验证 Logit softcapping 幅度不超过 30.0
        for (logits.data) |v| {
            try std.testing.expect(v >= -30.01 and v <= 30.01);
        }

        // 验证 Apple MLX GPU 加速模式输出精度等价性
        if (backend.mlx.has_mlx) {
            var g_gpu = autodiff.Graph.init(allocator, .{ .device = .gpu });
            defer g_gpu.deinit();

            const token_ids_gpu = try g_gpu.zeros(&.{ 2, 4 }, false);
            @memcpy(token_ids_gpu.data, token_ids.data);

            const logits_gpu = try model.forward(&g_gpu, token_ids_gpu);
            try std.testing.expectEqual(@as(usize, 3), logits_gpu.shape.len);
            try std.testing.expectEqual(@as(usize, 2), logits_gpu.shape.dims[0]);
            try std.testing.expectEqual(@as(usize, 4), logits_gpu.shape.dims[1]);
            try std.testing.expectEqual(@as(usize, 128), logits_gpu.shape.dims[2]);

            for (logits.data, logits_gpu.data) |c_val, g_val| {
                try std.testing.expectApproxEqAbs(c_val, g_val, 1e-3);
            }
        }
    }

    // 6. 4-bit (Q4) 分块量化与 Q4Linear 推理测试
    {
        var q4_layer = try gemma4_q4.Q4Linear.init(allocator, 32, 64);
        defer q4_layer.deinit(allocator);

        // 构造浮点权重并量化
        const weights = try allocator.alloc(f32, 32 * 64);
        defer allocator.free(weights);
        for (weights, 0..) |*w_val, i| {
            w_val.* = @as(f32, @floatFromInt(i % 10)) * 0.1 - 0.5;
        }
        q4_layer.quantizeFromF32(weights);

        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();

        const x = try g.ones(&.{ 1, 32 }, false);
        const y = try q4_layer.forward(&g, x);
        try std.testing.expectEqual(@as(usize, 2), y.shape.len);
        try std.testing.expectEqual(@as(usize, 1), y.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 64), y.shape.dims[1]);
    }

    // 7. 端到端 TinyQ4Gemma4 模型推理测试
    {
        var model = try gemma4_q4.TinyQ4Gemma4.init(allocator);
        defer model.deinit(allocator);

        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();

        const token_ids = try g.zeros(&.{ 1, 4 }, false);
        for (token_ids.data, 0..) |*val, idx| {
            val.* = @as(f32, @floatFromInt(idx % 10));
        }

        const logits = try model.forward(&g, token_ids);
        try std.testing.expectEqual(@as(usize, 3), logits.shape.len);
        try std.testing.expectEqual(@as(usize, 1), logits.shape.dims[0]); // B
        try std.testing.expectEqual(@as(usize, 4), logits.shape.dims[1]); // T
        try std.testing.expectEqual(@as(usize, 128), logits.shape.dims[2]); // V (vocab_size)

        // 验证 TinyQ4Gemma4 在 Apple MLX GPU 模式下的等价性
        if (backend.mlx.has_mlx) {
            var g_gpu = autodiff.Graph.init(allocator, .{ .device = .gpu });
            defer g_gpu.deinit();

            const token_ids_gpu = try g_gpu.zeros(&.{ 1, 4 }, false);
            @memcpy(token_ids_gpu.data, token_ids.data);

            const logits_gpu = try model.forward(&g_gpu, token_ids_gpu);
            try std.testing.expectEqual(@as(usize, 3), logits_gpu.shape.len);
            try std.testing.expectEqual(@as(usize, 1), logits_gpu.shape.dims[0]);
            try std.testing.expectEqual(@as(usize, 4), logits_gpu.shape.dims[1]);
            try std.testing.expectEqual(@as(usize, 128), logits_gpu.shape.dims[2]);

            for (logits.data, logits_gpu.data) |c_val, g_val| {
                try std.testing.expectApproxEqAbs(c_val, g_val, 1e-3);
            }
        }
    }

    // 8. 验证 Gemma4KVCache 键值缓存初始化、预填充与增量解码
    {
            var cache = try gemma4.Gemma4KVCache.initDefault(allocator, gemma4_q4.TinyQ4Gemma4.config);
            defer cache.deinit(allocator);
            try std.testing.expectEqual(@as(usize, 2), cache.layers.len);
            try std.testing.expectEqual(@as(usize, 0), cache.layers[0].curr_len);

            // 构造合法的 TinyQ4Gemma4 二进制内存切片
            const expected_bytes = gemma4_q4.TinyQ4Gemma4.computeExpectedFileSize();
            const mock_mmap = try allocator.alloc(u8, expected_bytes);
            defer allocator.free(mock_mmap);
            @memset(mock_mmap, 0);
            @memcpy(mock_mmap[0..8], "ZNNQ4G02");

            var mmap_model = try gemma4_q4.TinyQ4Gemma4.loadFromMmapWithOptions(allocator, mock_mmap, .{ .lm_head_mode = .bf16 });
            defer mmap_model.deinit(allocator);

            var g_prefill = autodiff.Graph.initDefault(allocator);
            defer g_prefill.deinit();

            const prompt = [_]usize{ 2, 10, 25 };
            const next_1 = try mmap_model.prefill(&g_prefill, &prompt, &cache, mock_mmap, 1.0);
            try std.testing.expectEqual(@as(usize, 3), cache.layers[0].curr_len);
            try std.testing.expect(next_1 < gemma4_q4.TinyQ4Gemma4.config.vocab_size);

            // 验证 mmap 加载后的模型同样可安全调用 forward() 接口
            const tok_tensor = try g_prefill.tensorNDWithData(&.{ 1, 3 }, &.{ 2.0, 10.0, 25.0 }, false);
            const mmap_logits = try mmap_model.forward(&g_prefill, tok_tensor);
            try std.testing.expectEqual(@as(usize, 3), mmap_logits.shape.len);
            try std.testing.expectEqual(@as(usize, 128), mmap_logits.shape.dims[2]);

            var g_decode = autodiff.Graph.initDefault(allocator);
            defer g_decode.deinit();

            const next_2 = try mmap_model.decodeStep(&g_decode, next_1, &cache, mock_mmap, &prompt, 1.0);
            try std.testing.expectEqual(@as(usize, 4), cache.layers[0].curr_len);
            try std.testing.expect(next_2 < gemma4_q4.TinyQ4Gemma4.config.vocab_size);

            // 验证在 Apple MLX GPU 硬件加速模式下的预填充与增量解码等价性
            if (backend.offload.isAvailable()) {
                cache.reset();
                var g_gpu_prefill = autodiff.Graph.init(allocator, .{ .device = .gpu });
                defer g_gpu_prefill.deinit();

                const next_gpu_1 = try mmap_model.prefill(&g_gpu_prefill, &prompt, &cache, mock_mmap, 1.0);
                try std.testing.expectEqual(@as(usize, 3), cache.layers[0].curr_len);
                try std.testing.expectEqual(next_1, next_gpu_1);

                var g_gpu_decode = autodiff.Graph.init(allocator, .{ .device = .gpu });
                defer g_gpu_decode.deinit();

                const next_gpu_2 = try mmap_model.decodeStep(&g_gpu_decode, next_gpu_1, &cache, mock_mmap, &prompt, 1.0);
                try std.testing.expectEqual(@as(usize, 4), cache.layers[0].curr_len);
                try std.testing.expectEqual(next_2, next_gpu_2);

                // 验证在 GPU 模式下的投机采样多词元校验
                var accepted_buf: [3]usize = undefined;
                var g_gpu_spec = autodiff.Graph.init(allocator, .{ .device = .gpu });
                defer g_gpu_spec.deinit();

                const input_spec = [_]usize{ next_gpu_2, 10, 25 };
                const num_acc = try mmap_model.decodeSpeculative(&g_gpu_spec, &input_spec, &cache, mock_mmap, &prompt, 1.0, &accepted_buf);
                try std.testing.expect(num_acc >= 1);
            }

            cache.reset();
            try std.testing.expectEqual(@as(usize, 0), cache.layers[0].curr_len);
        }
    }

test "Gemma 4 promptLookup N-gram speculative sampling" {
    // 1. 短序列测试：无法构成 N-gram 返回 null
    const short_seq = [_]usize{ 1, 2 };
    try std.testing.expectEqual(@as(?[]const usize, null), gemma4.promptLookup(&short_seq, 2, 4, 3));

    // 2. 匹配测试：重复 2-gram 提取后续候选
    const seq1 = [_]usize{ 10, 20, 30, 40, 50, 10, 20 };
    const draft1 = gemma4.promptLookup(&seq1, 2, 4, 3);
    try std.testing.expect(draft1 != null);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 30, 40, 50 }, draft1.?);

    // 3. 最大推测长度限制测试
    const draft1_max2 = gemma4.promptLookup(&seq1, 2, 4, 2);
    try std.testing.expect(draft1_max2 != null);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 30, 40 }, draft1_max2.?);

    // 4. 无重复序列测试
    const seq2 = [_]usize{ 1, 2, 3, 4, 5, 6, 7 };
    try std.testing.expectEqual(@as(?[]const usize, null), gemma4.promptLookup(&seq2, 2, 4, 3));

    // 5. 优先最长 3-gram 测试
    const seq3 = [_]usize{ 99, 10, 20, 30, 88, 10, 20, 77, 10, 20, 30 };
    const draft3 = gemma4.promptLookup(&seq3, 2, 4, 2);
    try std.testing.expect(draft3 != null);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 88, 10 }, draft3.?);
}

test "Gemma 4 SpeculativeDraftEngine Markov and N-gram proposal" {
    const allocator = std.testing.allocator;
    var engine = gemma4.SpeculativeDraftEngine.init(allocator);
    defer engine.deinit();

    // 1. 种子注入转移
    const prompt_seq = [_]usize{ 100, 200, 300, 400 };
    try engine.seedFromTokens(&prompt_seq);

    // 2. 预测转移：100 -> 200 -> 300 -> 400
    var buf: [3]usize = undefined;
    const cur_seq1 = [_]usize{ 999, 100 };
    const draft1 = engine.proposeDraft(&cur_seq1, &buf);
    try std.testing.expect(draft1 != null);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 200, 300, 400 }, draft1.?);

    // 3. 动态记录新转移
    try engine.recordTransition(500, 600);
    const cur_seq2 = [_]usize{ 888, 500 };
    const draft2 = engine.proposeDraft(&cur_seq2, &buf);
    try std.testing.expect(draft2 != null);
    try std.testing.expectEqualSlices(usize, &[_]usize{600}, draft2.?);
}

test "Gemma4E4BModel direct binary weight cache layout and reader verification" {
    const cfg = gemma4.Gemma4Config.e4b;
    const expected_bytes = Gemma4E4BModel.computeExpectedDirectCacheSize(cfg);
    // Gemma 4 E4B 4-bit 注意力/PLE + 8-bit MLP 混合量化权重总大小约为 5.95 GB (高于 5.5 GB，低于 6.5 GB)
    try std.testing.expect(expected_bytes > 5_500_000_000);
    try std.testing.expect(expected_bytes < 6_500_000_000);

    var mock_buf: [32]u8 = undefined;
    @memcpy(mock_buf[0..8], Gemma4E4BModel.DIRECT_CACHE_MAGIC);
    std.mem.writeInt(u64, mock_buf[8..16], Gemma4E4BModel.DIRECT_CACHE_VERSION, .little);
    @memset(mock_buf[16..32], 0xAB);

    var reader = try Gemma4E4BModel.DirectCacheReader.init(&mock_buf, 32);
    const payload = try reader.readBytes(16);
    try std.testing.expectEqual(@as(usize, 16), payload.len);
    try std.testing.expectEqual(@as(u8, 0xAB), payload[0]);
    try std.testing.expectError(error.InvalidDirectCache, reader.readBytes(1));
}


// ============================================================================
// 回归测试：Gemma 4 推理路径正确性、资源回收与权重布局
// ============================================================================

fn fillNormal(data: []f32, random: std.Random, scale: f32) void {
    for (data) |*v| v.* = random.floatNorm(f32) * scale;
}

fn randomizeAttention(attn: *gemma4.Gemma4Attention, random: std.Random) void {
    attn.q_proj.resetParameters(random, .{});
    attn.k_proj.resetParameters(random, .{});
    if (attn.v_proj) |*vp| vp.resetParameters(random, .{});
    attn.o_proj.resetParameters(random, .{});
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0;
    for (a, b) |x, y| m = @max(m, @abs(x - y));
    return m;
}

test "Gemma4DecoderLayer applies layer_scalar identically in forward, prefill and decode" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(1);
    const random = prng.random();

    var layer = try gemma4.Gemma4DecoderLayer.init(allocator, .sliding_attention, 16, 32, 4, 2, 4, 8, 1e-6);
    defer nn.deinitModel(&layer, allocator);
    randomizeAttention(&layer.self_attn, random);
    layer.mlp.gate_proj.resetParameters(random, .{});
    layer.mlp.up_proj.resetParameters(random, .{});
    layer.mlp.down_proj.resetParameters(random, .{});
    layer.layer_scalar.data[0] = 0.5;

    // 层缩放标量与官方实现一致为不可训练缓冲区
    try std.testing.expect(layer.layer_scalar.is_buffer);
    try std.testing.expect(!layer.layer_scalar.requires_grad);

    var cache = try gemma4.Gemma4KVCache.LayerCache.init(allocator, .sliding_attention, 2, 4, 8, 16);
    defer cache.deinit(allocator);

    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();
    const x = try g.tensorND(&.{ 1, 3, 16 }, false);
    fillNormal(x.data, random, 0.5);

    const y_fwd = try layer.forward(&g, x);
    const y_pre = try layer.forwardPrefill(&g, x, &cache);
    try std.testing.expect(maxAbsDiff(y_fwd.data, y_pre.data) < 1e-5);
}

fn checkAttentionPrefillDecodeParity(attn_type: gemma4.Gemma4AttentionType, head_dim: usize, n_kv: usize) !void {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();
    const C: usize = 16;
    const T: usize = 7; // 超过滑动窗口 W，覆盖带掩码的分解注意力分支
    const W: usize = 2;

    var attn = try gemma4.Gemma4Attention.init(allocator, attn_type, C, 4, n_kv, head_dim, W, 1e-6);
    defer nn.deinitModel(&attn, allocator);
    randomizeAttention(&attn, random);

    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();
    const x = try g.tensorND(&.{ 1, T, C }, false);
    fillNormal(x.data, random, 0.5);

    var cache_a = try gemma4.Gemma4KVCache.LayerCache.init(allocator, attn_type, n_kv, head_dim, W, 32);
    defer cache_a.deinit(allocator);
    const y_all = try attn.forwardPrefill(&g, x, &cache_a);
    try std.testing.expectEqual(T, cache_a.curr_len);

    var cache_b = try gemma4.Gemma4KVCache.LayerCache.init(allocator, attn_type, n_kv, head_dim, W, 32);
    defer cache_b.deinit(allocator);
    const x0 = try g.tensorNDWithData(&.{ 1, 1, C }, x.data[0..C], false);
    _ = try attn.forwardPrefill(&g, x0, &cache_b);
    for (1..T) |t| {
        const xt = try g.tensorNDWithData(&.{ 1, 1, C }, x.data[t * C .. (t + 1) * C], false);
        const yt = try attn.forwardDecode(&g, xt, &cache_b);
        try std.testing.expect(maxAbsDiff(yt.data, y_all.data[t * C .. (t + 1) * C]) < 1e-4);
    }
    // 预填充 (M = T) 与逐词元解码 (M = 1) 走不同的矩阵乘法分块路径，累加顺序不同，只要求浮点容差内一致
    try std.testing.expect(maxAbsDiff(cache_a.k[0 .. T * head_dim], cache_b.k[0 .. T * head_dim]) < 1e-5);
}

test "Gemma4Attention prefill matches incremental decode beyond the sliding window" {
    try checkAttentionPrefillDecodeParity(.sliding_attention, 4, 2);
    try checkAttentionPrefillDecodeParity(.full_attention, 16, 1);
}

test "Gemma4Q4Attention prefill matches incremental decode beyond the sliding window" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const random = prng.random();
    const C: usize = 32;
    const T: usize = 6;

    var attn = try gemma4_q4.Gemma4Q4Attention.init(allocator, .sliding_attention, C, 4, 2, 8, 3, 1e-6);
    defer attn.deinit(allocator);
    inline for (.{ &attn.q_proj, &attn.k_proj, &attn.v_proj.?, &attn.o_proj }) |lin| {
        const w = try allocator.alloc(f32, lin.in_features * lin.out_features);
        defer allocator.free(w);
        fillNormal(w, random, 0.2);
        lin.quantizeFromF32(w);
    }

    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();
    const x = try g.tensorND(&.{ 1, T, C }, false);
    fillNormal(x.data, random, 0.5);

    var cache_a = try gemma4.Gemma4KVCache.LayerCache.init(allocator, .sliding_attention, 2, 8, 3, 16);
    defer cache_a.deinit(allocator);
    const y_all = try attn.forwardPrefill(&g, x, &cache_a);

    var cache_b = try gemma4.Gemma4KVCache.LayerCache.init(allocator, .sliding_attention, 2, 8, 3, 16);
    defer cache_b.deinit(allocator);
    const x0 = try g.tensorNDWithData(&.{ 1, 1, C }, x.data[0..C], false);
    _ = try attn.forwardPrefill(&g, x0, &cache_b);
    for (1..T) |t| {
        const xt = try g.tensorNDWithData(&.{ 1, 1, C }, x.data[t * C .. (t + 1) * C], false);
        const yt = try attn.forwardDecode(&g, xt, &cache_b);
        try std.testing.expect(maxAbsDiff(yt.data, y_all.data[t * C .. (t + 1) * C]) < 1e-4);
    }
}

test "Gemma 4 inference entry points reject invalid prompts instead of reading out of bounds" {
    const allocator = std.testing.allocator;
    var model = try gemma4.TinyGemma4.initInference(allocator);
    defer nn.deinitModel(&model, allocator);

    var cache = try gemma4.Gemma4KVCache.init(allocator, gemma4.TinyGemma4.config, .{ .max_seq_len = 4 });
    defer cache.deinit(allocator);

    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();

    // 提示词超出键值缓存容量
    try std.testing.expectError(error.ContextWindowExceeded, model.prefill(&g, &.{ 2, 3, 4, 5, 6 }, &cache, 1.0));
    // 词元 ID 越界与空提示词
    try std.testing.expectError(error.InvalidTokenId, model.prefill(&g, &.{ 2, 128 }, &cache, 1.0));
    try std.testing.expectError(error.EmptyPrompt, model.prefill(&g, &.{}, &cache, 1.0));

    // 缓存写满后增量解码返回错误
    _ = try model.prefill(&g, &.{ 2, 3, 4 }, &cache, 1.0);
    _ = try model.decodeStep(&g, 5, &cache, &.{ 2, 3, 4 }, 1.0);
    try std.testing.expectEqual(@as(usize, 4), cache.layers[0].curr_len);
    try std.testing.expectError(error.ContextWindowExceeded, model.decodeStep(&g, 6, &cache, &.{ 2, 3, 4, 5 }, 1.0));
    try std.testing.expectError(error.InvalidTokenId, model.decodeStep(&g, 1000, &cache, &.{2}, 1.0));
}

/// 对构造函数的每一次内存分配依次注入失败，断言构造返回 OutOfMemory 且已分配内存全部释放。
/// 析构使用不注入失败的测试分配器 (nn.deinitModel 内部会分配去重表)。
fn expectConstructionReleasesOnOom(ctx: anytype) !void {
    const backing = std.testing.allocator;
    var counting = std.testing.FailingAllocator.init(backing, .{});
    var model = try ctx.construct(counting.allocator());
    const needed = counting.alloc_index;
    @TypeOf(ctx).destroy(&model, backing);

    for (0..needed) |fail_index| {
        var failing = std.testing.FailingAllocator.init(backing, .{ .fail_index = fail_index });
        if (ctx.construct(failing.allocator())) |constructed| {
            var leaked = constructed;
            @TypeOf(ctx).destroy(&leaked, backing);
            return error.SwallowedOutOfMemoryError;
        } else |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        }
    }
}

const TinyGemma4Ctor = struct {
    fn construct(_: @This(), allocator: std.mem.Allocator) !gemma4.TinyGemma4 {
        return gemma4.TinyGemma4.init(allocator);
    }
    fn destroy(model: *gemma4.TinyGemma4, allocator: std.mem.Allocator) void {
        nn.deinitModel(model, allocator);
    }
};

const TinyQ4Gemma4Ctor = struct {
    fn construct(_: @This(), allocator: std.mem.Allocator) !gemma4_q4.TinyQ4Gemma4 {
        return gemma4_q4.TinyQ4Gemma4.init(allocator);
    }
    fn destroy(model: *gemma4_q4.TinyQ4Gemma4, allocator: std.mem.Allocator) void {
        model.deinit(allocator);
    }
};

const TinyQ4MmapCtor = struct {
    bytes: []const u8,
    fn construct(self: @This(), allocator: std.mem.Allocator) !gemma4_q4.TinyQ4Gemma4 {
        return gemma4_q4.TinyQ4Gemma4.loadFromMmapWithOptions(allocator, self.bytes, .{ .lm_head_mode = .bf16, .warmup = false });
    }
    fn destroy(model: *gemma4_q4.TinyQ4Gemma4, allocator: std.mem.Allocator) void {
        model.deinit(allocator);
    }
};

const KVCacheCtor = struct {
    fn construct(_: @This(), allocator: std.mem.Allocator) !gemma4.Gemma4KVCache {
        return gemma4.Gemma4KVCache.init(allocator, gemma4.Gemma4Config.tiny_test, .{ .max_seq_len = 8 });
    }
    fn destroy(cache: *gemma4.Gemma4KVCache, allocator: std.mem.Allocator) void {
        cache.deinit(allocator);
    }
};

test "Gemma 4 model construction releases every allocation on out-of-memory" {
    try expectConstructionReleasesOnOom(TinyGemma4Ctor{});
    try expectConstructionReleasesOnOom(TinyQ4Gemma4Ctor{});
    try expectConstructionReleasesOnOom(KVCacheCtor{});

    if (backend.offload.isAvailable()) return;
    const allocator = std.testing.allocator;
    const M = gemma4_q4.TinyQ4Gemma4;
    const bytes = try allocator.alloc(u8, M.computeExpectedFileSize());
    defer allocator.free(bytes);
    @memset(bytes, 0);
    @memcpy(bytes[0..8], M.Q4_MAGIC);
    try expectConstructionReleasesOnOom(TinyQ4MmapCtor{ .bytes = bytes });
}

test "Gemma4Q4 mmap loader reads per-layer norms and layer_scalar in storage order" {
    const allocator = std.testing.allocator;
    const M = gemma4_q4.TinyQ4Gemma4;
    const cfg = M.config;
    const bytes = try allocator.alloc(u8, M.computeExpectedFileSize());
    defer allocator.free(bytes);
    @memset(bytes, 0);
    @memcpy(bytes[0..8], M.Q4_MAGIC);

    // 第 1 层：input_layernorm[0] = 2.0 (BF16 0x4000)，layer_scalar = 0.5 (BF16 0x3F00)
    const offsets = M.computeLayerOffsets().offsets;
    const layer1 = offsets[1];
    std.mem.writeInt(u16, bytes[layer1..][0..2], 0x4000, .little);
    const scalar_off = layer1 + M.layerNormsBytes(1) - 2;
    std.mem.writeInt(u16, bytes[scalar_off..][0..2], 0x3F00, .little);
    try std.testing.expectEqual(M.computeExpectedFileSize(), offsets[cfg.num_hidden_layers]);

    var model = try M.loadFromMmapWithOptions(allocator, bytes, .{ .lm_head_mode = .bf16, .warmup = false });
    defer model.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 2.0), model.layers[1].input_layernorm.weight.data[0]);
    try std.testing.expectEqual(@as(f32, 0.5), model.layers[1].layer_scalar.data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), model.layers[0].layer_scalar.data[0]);
    try std.testing.expectEqual(cfg.hidden_size * cfg.num_attention_heads * cfg.head_dim / 32, model.layers[0].self_attn.q_proj.blocks.len);

    // 词元 ID 越界时返回错误而非越界读取嵌入矩阵
    var cache = try gemma4.Gemma4KVCache.initDefault(allocator, cfg);
    defer cache.deinit(allocator);
    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();
    try std.testing.expectError(error.InvalidTokenId, model.prefill(&g, &.{ 2, cfg.vocab_size }, &cache, bytes, 1.0));
    try std.testing.expectError(error.EmptyPrompt, model.prefill(&g, &.{}, &cache, bytes, 1.0));
}

test "Gemma4Q4 direct layout keeps packed u32 weights 4-byte aligned" {
    inline for (.{ gemma4_q4.DefaultGemma4Q4, gemma4_q4.TinyQ4Gemma4 }) |M| {
        const offsets = M.computeDirectLayerOffsets().offsets;
        for (0..M.config.num_hidden_layers) |i| {
            try std.testing.expectEqual(@as(usize, 0), offsets[i] % 4);
            try std.testing.expectEqual(@as(usize, 0), (offsets[i] + M.directLayerNormsBytes(i)) % 4);
            try std.testing.expect(M.directLayerNormsBytes(i) >= M.layerNormsBytes(i));
        }
        try std.testing.expectEqual(M.computeExpectedDirectFileSize(), offsets[M.config.num_hidden_layers]);
    }
}

test "Gemma4Q4 direct layout requires hardware offload and never leaks on rejection" {
    if (backend.offload.isAvailable()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const M = gemma4_q4.TinyQ4Gemma4;
    const bytes = try allocator.alloc(u8, M.computeExpectedDirectFileSize());
    defer allocator.free(bytes);
    @memset(bytes, 0);
    @memcpy(bytes[0..8], M.DIRECT_MAGIC);
    try std.testing.expectError(error.DirectLayoutRequiresOffload, M.loadFromMmapWithOptions(allocator, bytes, .{ .lm_head_mode = .bf16, .warmup = false }));
}

test "promptLookup tolerates a zero minimum n-gram length" {
    const seq = [_]usize{ 5, 6, 7, 5, 6 };
    const draft = gemma4.promptLookup(&seq, 0, 3, 2);
    try std.testing.expect(draft != null);
    try std.testing.expectEqualSlices(usize, &.{ 7, 5 }, draft.?);
    try std.testing.expectEqual(@as(?[]const usize, null), gemma4.promptLookup(&.{1}, 0, 3, 2));
}

test "Gemma 4 SIMD dot kernels match scalar references" {
    var prng = std.Random.DefaultPrng.init(3);
    const random = prng.random();
    var a: [37]f32 = undefined;
    var b: [37]f32 = undefined;
    var b_bits: [37]u16 = undefined;
    fillNormal(&a, random, 1.0);
    fillNormal(&b, random, 1.0);
    for (&b_bits, &b) |*bits, *v| {
        bits.* = @truncate(@as(u32, @bitCast(v.*)) >> 16);
        v.* = @bitCast(@as(u32, bits.*) << 16);
    }
    inline for (.{ 0, 1, 16, 31, 37 }) |n| {
        var ref: f32 = 0;
        for (a[0..n], b[0..n]) |x, y| ref += x * y;
        try std.testing.expectApproxEqAbs(ref, gemma4.dotF32(a[0..n], b[0..n]), 1e-4);
        try std.testing.expectApproxEqAbs(ref, gemma4.dotF32Bf16(a[0..n], b_bits[0..n]), 1e-4);
    }
}

/// 以 `mlx.fast.rope(dims = hd, freqs)` 的语义 (配对 (i, i + hd/2)，theta_i = pos / freqs[i]) 在 CPU 上旋转单个头向量
fn ropeWithFreqsReference(x: []const f32, freqs: []const f32, pos: usize, out: []f32) void {
    const half = x.len / 2;
    @memcpy(out, x);
    for (freqs, 0..) |f, i| {
        const theta = @as(f32, @floatFromInt(pos)) / f;
        const a = x[i];
        const b = x[i + half];
        out[i] = a * @cos(theta) - b * @sin(theta);
        out[i + half] = b * @cos(theta) + a * @sin(theta);
    }
}

fn checkProportionalRopeTable(comptime hd: usize, pos: usize, attn_type: gemma4.Gemma4AttentionType) !void {
    const allocator = std.testing.allocator;
    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();
    const base = attn_type.ropeTheta();
    const factor = attn_type.partialRotaryFactor();
    const x = try g.tensorND(&.{ 1, 1, 1, hd }, false);
    for (x.data, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i + 1))) * 0.5;
    const y_ref = try gemma4.ropeSplitHalf(&g, x, pos, factor, base);

    var freqs: [hd / 2]f32 = undefined;
    gemma4_gpu.proportionalRopeFreqs(&freqs, hd, factor, base);
    var y: [hd]f32 = undefined;
    ropeWithFreqsReference(x.data[0..hd], &freqs, pos, &y);
    try std.testing.expect(maxAbsDiff(y_ref.data[0..hd], &y) < 1e-4);
    // 未参与旋转的角度周期为 +inf (保持恒等)
    const num_angles: usize = @intFromFloat(@as(f32, @floatFromInt(hd / 2)) * factor);
    for (freqs, 0..) |f, i| try std.testing.expectEqual(i >= num_angles, std.math.isInf(f));
}

test "Gemma 4 proportional RoPE frequency table matches the F32 CPU path for E4B and 12B global layers" {
    // 小维度、E4B / 12B 全局注意力层 (global_head_dim = 512, partial_rotary_factor = 0.25) 与滑动窗口层 (head_dim = 256, 全旋转)
    try checkProportionalRopeTable(16, 5, .full_attention);
    try checkProportionalRopeTable(gemma4.Gemma4Config.default.global_head_dim, 37, .full_attention);
    try checkProportionalRopeTable(gemma4.Gemma4Config.e4b.global_head_dim, 1000, .full_attention);
    try checkProportionalRopeTable(gemma4.Gemma4Config.default.head_dim, 9, .sliding_attention);
}

test "Gemma4E4B KV-shared layers reuse the last non-shared layer of the same attention type" {
    const cfg = gemma4.Gemma4Config.e4b;
    const expected = [42]usize{
        0,  1,  2,  3,  4,  5,
        6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23,
        22, 22, 22, 22, 22, 23,
        22, 22, 22, 22, 22, 23,
        22, 22, 22, 22, 22, 23,
    };
    for (expected, 0..) |src, idx| try std.testing.expectEqual(src, gemma4_e4b.kvSourceLayer(cfg, idx));
    for (0..gemma4_e4b.num_kv_layers) |idx| {
        try std.testing.expectEqual(idx == 22 or idx == 23, gemma4_e4b.isKvSharedSource(cfg, idx));
    }
}

// ============================================================================
// 回归测试：采样排除词元、截断后惩罚、键值缓存窗口同步与 LM Head 量化模式校验
// ============================================================================

const tiny_sampling_excluded = [_]sampling.TokenRange{ .single(0), .{ .first = 5, .last = 6 } };

/// 8 词元、2 维隐状态、截断上限 2.0 的采样测试配置
const tiny_sampling_config = blk: {
    var c = gemma4.Gemma4Config.tiny_test;
    c.vocab_size = 8;
    c.hidden_size = 2;
    c.final_logit_softcapping = 2.0;
    c.sampling_excluded_tokens = &tiny_sampling_excluded;
    break :blk c;
};

test "Gemma 4 CPU greedy LM head skips configured tokens and penalizes after softcapping" {
    const allocator = std.testing.allocator;
    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();

    // h = [1, 0] 时每个词元的原始对数几率即嵌入行首元素
    const raw = [_]f32{ 100.0, 1.5, 10.0, -1.0, 0.5, 50.0, 40.0, 1.0 };
    var embed_f32: [16]f32 = undefined;
    var embed_bf16: [16]u16 = undefined;
    for (raw, 0..) |r, v| {
        embed_f32[v * 2] = r;
        embed_f32[v * 2 + 1] = 0.0;
        embed_bf16[v * 2] = @truncate(@as(u32, @bitCast(r)) >> 16);
        embed_bf16[v * 2 + 1] = 0;
    }
    const h = [_]f32{ 1.0, 0.0 };
    const cfg = tiny_sampling_config;

    inline for (.{ f32, u16 }) |Elem| {
        const embed: []const Elem = if (Elem == f32) &embed_f32 else &embed_bf16;
        // 0、5、6 号词元得分最高但属于配置中的排除集合
        try std.testing.expectEqual(@as(usize, 2), try gemma4.greedyLmHeadCpu(cfg, Elem, &g, embed, &h, &.{}, 1.0));
        // 截断后惩罚：2 * tanh(10 / 2) / 2 ≈ 1.0 < 2 * tanh(1.5 / 2) ≈ 1.27，词元 1 胜出；
        // 若先惩罚再截断，2 * tanh(5 / 2) ≈ 1.97 仍会选中词元 2
        try std.testing.expectEqual(@as(usize, 1), try gemma4.greedyLmHeadCpu(cfg, Elem, &g, embed, &h, &.{2}, 2.0));
    }

    const after = gemma4.samplingLogit(10.0, 2.0, true, 2.0);
    const before = gemma4.softcapLogit(10.0 / 2.0, 2.0);
    try std.testing.expectApproxEqAbs(2.0 * std.math.tanh(@as(f32, 5.0)) / 2.0, after, 1e-6);
    try std.testing.expect(before > gemma4.samplingLogit(1.5, 2.0, false, 2.0));
    try std.testing.expect(after < gemma4.samplingLogit(1.5, 2.0, false, 2.0));
    // 负值惩罚同样作用在截断之后：乘以惩罚系数
    try std.testing.expectApproxEqAbs(2.0 * std.math.tanh(@as(f32, -0.5)) * 3.0, gemma4.samplingLogit(-1.0, 2.0, true, 3.0), 1e-6);
    try std.testing.expectEqual(@as(f32, 7.0), gemma4.softcapLogit(7.0, 0.0));
}

test "Gemma 4 sampling exclusions come from the config and agree between CPU argmax and the GPU mask" {
    const allocator = std.testing.allocator;
    const cfg = gemma4.Gemma4Config.default;
    try std.testing.expect(cfg.isSamplingExcluded(0));
    for (258880..258885) |v| try std.testing.expect(cfg.isSamplingExcluded(v));
    try std.testing.expect(!cfg.isSamplingExcluded(258879));
    try std.testing.expect(!cfg.isSamplingExcluded(258885));
    // E4B 与 12B 共用同一份词表排除定义
    try std.testing.expectEqual(cfg.sampling_excluded_tokens.ptr, gemma4.Gemma4Config.e4b.sampling_excluded_tokens.ptr);

    const logits = try allocator.alloc(f32, cfg.vocab_size);
    defer allocator.free(logits);
    @memset(logits, -1.0);
    logits[0] = 10.0;
    logits[258882] = 9.0;
    logits[77] = 3.0;
    try std.testing.expectEqual(@as(usize, 77), sampling.argmaxWithPenalty(logits, cfg.vocab_size, cfg.sampling_excluded_tokens, &.{}, 1.0));
    try std.testing.expectEqual(@as(usize, 77), sampling.argmaxWithPenalty(logits, cfg.vocab_size, cfg.sampling_excluded_tokens, &.{5}, 1.5));

    // GPU 贪婪路径叠加的 BF16 加性掩码恰好屏蔽同一组词元
    const mask = try allocator.alloc(u16, cfg.vocab_size);
    defer allocator.free(mask);
    sampling.fillExclusionMaskBf16(mask, cfg.sampling_excluded_tokens);
    for (mask, 0..) |bits, v| {
        try std.testing.expectEqual(cfg.isSamplingExcluded(v), bits == sampling.bf16_neg_inf_bits);
    }
}

test "Gemma4KVCache layer window pack and restore keep incremental decode identical" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(23);
    const random = prng.random();
    const n_kv: usize = 2;
    const hd: usize = 4;
    const nh: usize = 4;
    const W: usize = 3;
    const T: usize = 6;

    inline for (.{ gemma4.Gemma4AttentionType.sliding_attention, gemma4.Gemma4AttentionType.full_attention }) |attn_type| {
        var cache_a = try gemma4.Gemma4KVCache.LayerCache.init(allocator, attn_type, n_kv, hd, W, 16);
        defer cache_a.deinit(allocator);
        var k: [n_kv * T * hd]f32 = undefined;
        var v: [n_kv * T * hd]f32 = undefined;
        fillNormal(&k, random, 1.0);
        fillNormal(&v, random, 1.0);
        try gemma4.writePrefillKVCache(&cache_a, &k, &v, 1, T);

        // 打包可见窗口 (滑动窗口层只保留最近 W 个位置)，写回一个全新的缓存 (模拟 GPU -> CPU 同步)
        const win = cache_a.windowLen();
        try std.testing.expectEqual(if (attn_type == .sliding_attention) W else T, win);
        var packed_k: [n_kv * T * hd]f32 = undefined;
        var packed_v: [n_kv * T * hd]f32 = undefined;
        try cache_a.packWindow(packed_k[0 .. n_kv * win * hd], packed_v[0 .. n_kv * win * hd]);

        var cache_b = try gemma4.Gemma4KVCache.LayerCache.init(allocator, attn_type, n_kv, hd, W, 16);
        defer cache_b.deinit(allocator);
        try cache_b.writeWindow(packed_k[0 .. n_kv * win * hd], packed_v[0 .. n_kv * win * hd], win, T);
        try std.testing.expectEqual(T, cache_b.curr_len);
        // 全局注意力层必须恢复完整历史，过短的窗口被拒绝
        if (attn_type == .full_attention) {
            try std.testing.expectError(error.ShapeMismatch, cache_b.writeWindow(packed_k[0 .. n_kv * hd], packed_v[0 .. n_kv * hd], 1, T));
        }

        // 两个缓存上的下一步增量解码结果逐位一致
        var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
        defer g.deinit();
        var q: [nh * hd]f32 = undefined;
        var k1: [n_kv * hd]f32 = undefined;
        var v1: [n_kv * hd]f32 = undefined;
        fillNormal(&q, random, 1.0);
        fillNormal(&k1, random, 1.0);
        fillNormal(&v1, random, 1.0);
        const y_a = try gemma4.decodeAttention(&g, &cache_a, attn_type, W, nh, &q, &k1, &v1);
        const y_b = try gemma4.decodeAttention(&g, &cache_b, attn_type, W, nh, &q, &k1, &v1);
        try std.testing.expectEqualSlices(f32, y_a.data, y_b.data);
    }
}

test "Gemma4Q4 loader rejects an LM head quantization mode the hidden size cannot support" {
    const allocator = std.testing.allocator;
    const M = gemma4_q4.TinyQ4Gemma4;
    const bytes = try allocator.alloc(u8, M.computeExpectedFileSize());
    defer allocator.free(bytes);
    @memset(bytes, 0);
    @memcpy(bytes[0..8], M.Q4_MAGIC);
    // hidden_size = 32 不整除 Q8 分组长度 64：显式报错而不是静默退回 BF16
    try std.testing.expectError(error.UnsupportedQuantMode, M.loadFromMmapWithOptions(allocator, bytes, .{ .lm_head_mode = .q8, .warmup = false }));
    var model = try M.loadFromMmapWithOptions(allocator, bytes, .{ .lm_head_mode = .q4, .warmup = false });
    model.deinit(allocator);
}
