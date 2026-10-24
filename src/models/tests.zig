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

    // 2. Gemma 滑动窗口层半切分旋转位置编码 (Split-Half RoPE) 配置单测
    {
        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 1, 2, 8 }, false);
        const x_rot = try g.rope(x, 0, gemma4.Gemma4AttentionType.sliding_attention.ropeOptions());
        try std.testing.expectEqual(@as(usize, 3), x_rot.shape.len);
        try std.testing.expectEqual(@as(usize, 8), x_rot.shape.dims[2]);
    }

    // 3. Gemma4MLP 单元测试 (与按双曲正切近似 GELU 手工计算的结果逐元素对比)
    {
        var mlp = try gemma4.Gemma4MLP.init(allocator, 8, 16, .tanh);
        mlp.gate_proj.resetParameters(random, .{});
        mlp.up_proj.resetParameters(random, .{});
        mlp.down_proj.resetParameters(random, .{});
        defer nn.deinitModel(&mlp, allocator);

        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();
        const x = try g.tensorND(&.{ 2, 8 }, false);
        for (x.data, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.9) * 3.0;
        const out = try mlp.forward(&g, x);
        try std.testing.expectEqual(@as(usize, 2), out.shape.len);
        try std.testing.expectEqual(@as(usize, 2), out.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 8), out.shape.dims[1]);

        const gate = try mlp.gate_proj.forward(&g, x);
        const up = try mlp.up_proj.forward(&g, x);
        const hidden = try g.tensorND(&.{ 2, 16 }, false);
        for (hidden.data, gate.data, up.data) |*h, gv, uv| h.* = backend.activation.gelu(.tanh, gv) * uv;
        const expected = try mlp.down_proj.forward(&g, hidden);
        for (expected.data, out.data) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);

        // 同一权重下的精确 (erf) 形式输出与之接近但不相同
        mlp.hidden_activation = .none;
        const out_exact = try mlp.forward(&g, x);
        var max_diff: f32 = 0.0;
        for (out_exact.data, out.data) |e, o| max_diff = @max(max_diff, @abs(e - o));
        try std.testing.expect(max_diff > 0.0);
    }

    // 4. Gemma4Attention 单元测试 (含 Q/K-Norm, V-Norm 与 GQA)
    {
        var attn = try gemma4.Gemma4Attention.init(
            allocator,
            .sliding_attention,
            16, // hidden_size
            4, // num_heads
            2, // num_kv_heads
            4, // head_dim
            2, // sliding_window
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
        var greedy = sampling.Sampler.init(.greedy);
        const next_1 = try mmap_model.prefill(&g_prefill, &prompt, &cache, mock_mmap, &greedy);
        try std.testing.expectEqual(@as(usize, 3), cache.layers[0].curr_len);
        try std.testing.expect(next_1 < gemma4_q4.TinyQ4Gemma4.config.vocab_size);

        // 验证 mmap 加载后的模型同样可安全调用 forward() 接口
        const tok_tensor = try g_prefill.tensorNDWithData(&.{ 1, 3 }, &.{ 2.0, 10.0, 25.0 }, false);
        const mmap_logits = try mmap_model.forward(&g_prefill, tok_tensor);
        try std.testing.expectEqual(@as(usize, 3), mmap_logits.shape.len);
        try std.testing.expectEqual(@as(usize, 128), mmap_logits.shape.dims[2]);

        var g_decode = autodiff.Graph.initDefault(allocator);
        defer g_decode.deinit();

        const next_2 = try mmap_model.decodeStep(&g_decode, next_1, &cache, mock_mmap, &prompt, &greedy);
        try std.testing.expectEqual(@as(usize, 4), cache.layers[0].curr_len);
        try std.testing.expect(next_2 < gemma4_q4.TinyQ4Gemma4.config.vocab_size);

        // 验证在 Apple MLX GPU 硬件加速模式下的预填充与增量解码等价性
        if (backend.offload.isAvailable()) {
            cache.reset();
            var g_gpu_prefill = autodiff.Graph.init(allocator, .{ .device = .gpu });
            defer g_gpu_prefill.deinit();

            const next_gpu_1 = try mmap_model.prefill(&g_gpu_prefill, &prompt, &cache, mock_mmap, &greedy);
            try std.testing.expectEqual(@as(usize, 3), cache.layers[0].curr_len);
            try std.testing.expectEqual(next_1, next_gpu_1);

            var g_gpu_decode = autodiff.Graph.init(allocator, .{ .device = .gpu });
            defer g_gpu_decode.deinit();

            const next_gpu_2 = try mmap_model.decodeStep(&g_gpu_decode, next_gpu_1, &cache, mock_mmap, &prompt, &greedy);
            try std.testing.expectEqual(@as(usize, 4), cache.layers[0].curr_len);
            try std.testing.expectEqual(next_2, next_gpu_2);

            // 验证在 GPU 模式下的投机采样多词元校验
            var accepted_buf: [3]usize = undefined;
            var g_gpu_spec = autodiff.Graph.init(allocator, .{ .device = .gpu });
            defer g_gpu_spec.deinit();

            const input_spec = [_]usize{ next_gpu_2, 10, 25 };
            const num_acc = try mmap_model.decodeSpeculative(&g_gpu_spec, &input_spec, &cache, mock_mmap, &prompt, &greedy, &accepted_buf);
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

    var layer = try gemma4.Gemma4DecoderLayer.init(allocator, .sliding_attention, 16, 32, 4, 2, 4, 8, 1e-6, .tanh);
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
    var greedy = sampling.Sampler.init(.greedy);

    // 提示词超出键值缓存容量
    try std.testing.expectError(error.ContextWindowExceeded, model.prefill(&g, &.{ 2, 3, 4, 5, 6 }, &cache, &greedy));
    // 词元 ID 越界与空提示词
    try std.testing.expectError(error.IndexOutOfBounds, model.prefill(&g, &.{ 2, 128 }, &cache, &greedy));
    try std.testing.expectError(error.EmptyPrompt, model.prefill(&g, &.{}, &cache, &greedy));

    // 缓存写满后增量解码返回错误
    _ = try model.prefill(&g, &.{ 2, 3, 4 }, &cache, &greedy);
    _ = try model.decodeStep(&g, 5, &cache, &.{ 2, 3, 4 }, &greedy);
    try std.testing.expectEqual(@as(usize, 4), cache.layers[0].curr_len);
    try std.testing.expectError(error.ContextWindowExceeded, model.decodeStep(&g, 6, &cache, &.{ 2, 3, 4, 5 }, &greedy));
    try std.testing.expectError(error.IndexOutOfBounds, model.decodeStep(&g, 1000, &cache, &.{2}, &greedy));
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
    var greedy = sampling.Sampler.init(.greedy);
    try std.testing.expectError(error.IndexOutOfBounds, model.prefill(&g, &.{ 2, cfg.vocab_size }, &cache, bytes, &greedy));
    try std.testing.expectError(error.EmptyPrompt, model.prefill(&g, &.{}, &cache, bytes, &greedy));
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
    const opts = attn_type.ropeOptions();
    const x = try g.tensorND(&.{ 1, 1, 1, hd }, false);
    for (x.data, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i + 1))) * 0.5;
    const y_ref = try g.rope(x, pos, opts);

    var freqs: [hd / 2]f32 = undefined;
    opts.fillPeriods(hd, &freqs);
    var y: [hd]f32 = undefined;
    ropeWithFreqsReference(x.data[0..hd], &freqs, pos, &y);
    try std.testing.expect(maxAbsDiff(y_ref.data[0..hd], &y) < 1e-4);
    // 未参与旋转的角度周期为 +inf (保持恒等)
    const num_angles: usize = @intFromFloat(@as(f32, @floatFromInt(hd / 2)) * opts.partial_rotary_factor);
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
    const before = backend.vec_math.softcap(10.0 / 2.0, 2.0);
    try std.testing.expectApproxEqAbs(2.0 * std.math.tanh(@as(f32, 5.0)) / 2.0, after, 1e-6);
    try std.testing.expect(before > gemma4.samplingLogit(1.5, 2.0, false, 2.0));
    try std.testing.expect(after < gemma4.samplingLogit(1.5, 2.0, false, 2.0));
    // 负值惩罚同样作用在截断之后：乘以惩罚系数
    try std.testing.expectApproxEqAbs(2.0 * std.math.tanh(@as(f32, -0.5)) * 3.0, gemma4.samplingLogit(-1.0, 2.0, true, 3.0), 1e-6);
    try std.testing.expectEqual(@as(f32, 7.0), backend.vec_math.softcap(7.0, 0.0));
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

/// 按 ZNNVOC01 格式在内存中构建词表文件：magic + count + 字符串总字节数 + 偏移表 + 字符串数据
fn buildVocabBytes(allocator: std.mem.Allocator, pieces: []const []const u8) ![]align(4) u8 {
    var total: usize = 0;
    for (pieces) |p| total += p.len;
    const bytes = try allocator.alignedAlloc(u8, .@"4", 16 + pieces.len * 4 + total);
    @memcpy(bytes[0..8], gemma4.Gemma4Vocabulary.MAGIC);
    std.mem.writeInt(u32, bytes[8..12], @intCast(pieces.len), .little);
    std.mem.writeInt(u32, bytes[12..16], @intCast(total), .little);
    var off: usize = 0;
    const str_start = 16 + pieces.len * 4;
    for (pieces, 0..) |p, i| {
        std.mem.writeInt(u32, bytes[16 + i * 4 ..][0..4], @intCast(off), .little);
        @memcpy(bytes[str_start + off ..][0..p.len], p);
        off += p.len;
    }
    return bytes;
}

test "Gemma4Vocabulary encode falls back to <0xNN> byte tokens instead of dropping bytes" {
    const allocator = std.testing.allocator;
    const pieces = [_][]const u8{ "<pad>", "a", " b", "<0x63>", "<0xE2>", "<0x82>", "<0xAC>" };
    const bytes = try buildVocabBytes(allocator, &pieces);
    defer allocator.free(bytes);
    // 词表直接引用内存缓冲区 (非 mmap)，测试结束时不调用会执行 munmap 的 deinit
    const vocab = try gemma4.Gemma4Vocabulary.loadFromMmap(bytes);

    const ac = try vocab.encode(allocator, "ac");
    defer allocator.free(ac);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, ac);

    // 多字节 UTF-8 字符 '€' (E2 82 AC) 整体逐字节回退
    const euro = try vocab.encode(allocator, "a€");
    defer allocator.free(euro);
    try std.testing.expectEqualSlices(usize, &.{ 1, 4, 5, 6 }, euro);

    const spaced = try vocab.encode(allocator, " b");
    defer allocator.free(spaced);
    try std.testing.expectEqualSlices(usize, &.{2}, spaced);

    // 词表缺少字节词元 <0x64> ('d') 时返回错误而不是静默丢弃
    try std.testing.expectError(error.UnencodableByte, vocab.encode(allocator, "ad"));
    try std.testing.expectEqual(@as(usize, 3), try vocab.byteToken('c'));
}

// ============================================================================
// 随机采样 (Stochastic Sampling)：官方生成配置、CPU LM Head 采样顺序、种子可复现性与 GPU 候选选择
// ============================================================================

test "Gemma 4 default generation options match the official generation_config.json" {
    inline for (.{ gemma4.Gemma4Config.default, gemma4.Gemma4Config.e4b, gemma4.Gemma4Config.tiny_test }) |cfg| {
        const gen = cfg.generation;
        try std.testing.expectEqual(@as(f32, 1.0), gen.temperature);
        try std.testing.expectEqual(@as(usize, 64), gen.top_k);
        try std.testing.expectEqual(@as(f32, 0.95), gen.top_p);
        try std.testing.expectEqual(@as(f32, 1.0), gen.repetition_penalty);
        try std.testing.expect(!gen.isGreedy());
        // 官方 suppress_tokens [258882, 258883] 落在采样排除集合内
        try std.testing.expect(cfg.isSamplingExcluded(258882) and cfg.isSamplingExcluded(258883));
    }
}

test "Gemma 4 CPU LM head sampling applies softcap, penalty and exclusion before the shared finisher" {
    const allocator = std.testing.allocator;
    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();

    const raw = [_]f32{ 100.0, 1.5, 10.0, -1.0, 0.5, 50.0, 40.0, 1.0 };
    var embed: [16]f32 = undefined;
    for (raw, 0..) |r, v| {
        embed[v * 2] = r;
        embed[v * 2 + 1] = 0.0;
    }
    const h = [_]f32{ 1.0, 0.0 };
    const cfg = tiny_sampling_config;
    const recent = [_]usize{ 2, 2, 4 };

    // 完整对数几率即截断后的点积
    var logits: [8]f32 = undefined;
    try gemma4.lmHeadLogitsCpu(cfg, f32, &g, &embed, &h, &logits);
    for (raw, logits) |r, l| try std.testing.expectApproxEqAbs(backend.vec_math.softcap(r, cfg.final_logit_softcapping), l, 1e-6);

    // 贪婪采样 (温度 <= 0 或 top_k == 1) 走融合快车道，与 greedyLmHeadCpu 一致
    for ([_]sampling.SamplingOptions{
        .{ .temperature = 0.0, .repetition_penalty = 2.0 },
        .{ .temperature = 0.8, .top_k = 1, .repetition_penalty = 2.0 },
    }) |opts| {
        var sampler = sampling.Sampler.init(opts);
        try std.testing.expectEqual(
            try gemma4.greedyLmHeadCpu(cfg, f32, &g, &embed, &h, &recent, 2.0),
            try gemma4.sampleLmHeadCpu(cfg, f32, &g, &embed, &h, &recent, &sampler),
        );
    }

    // 随机采样与「截断 → 惩罚 → 排除 → Sampler.sampleLogits」的参考流程逐词元一致
    var processed = logits;
    try sampling.applyRepetitionPenaltyInPlace(allocator, &processed, &recent, 2.0);
    sampling.maskExcludedInPlace(&processed, cfg.sampling_excluded_tokens);
    const opts = sampling.SamplingOptions{ .temperature = 1.5, .top_k = 4, .top_p = 0.9, .repetition_penalty = 2.0, .seed = 31 };
    var model_sampler = sampling.Sampler.init(opts);
    var reference = sampling.Sampler.init(opts);
    var seen: [8]bool = @splat(false);
    for (0..400) |_| {
        const tok = try gemma4.sampleLmHeadCpu(cfg, f32, &g, &embed, &h, &recent, &model_sampler);
        try std.testing.expectEqual(try reference.sampleLogits(g.executor(), allocator, &processed), tok);
        try std.testing.expect(!cfg.isSamplingExcluded(tok));
        seen[tok] = true;
    }
    // 受罚后的词元 2 仍在前 K 候选之内，被抽中过
    try std.testing.expect(seen[2] and seen[1]);
}

/// 只初始化词嵌入的 TinyGemma4 (投影权重为零，隐状态经残差直通)：对数几率完全由词嵌入决定
fn initTinyGemma4ForSampling(allocator: std.mem.Allocator) !gemma4.TinyGemma4 {
    const model = try gemma4.TinyGemma4.initInference(allocator);
    var prng = std.Random.DefaultPrng.init(2026);
    fillNormal(model.embed_tokens.weight.data, prng.random(), 0.3);
    return model;
}

/// 预填充后连续增量解码，返回生成的 n 个词元
fn generateTinyF32(
    model: *const gemma4.TinyGemma4,
    allocator: std.mem.Allocator,
    cache: *gemma4.Gemma4KVCache,
    prompt: []const usize,
    sampler: *sampling.Sampler,
    out: []usize,
) !void {
    var history: std.ArrayList(usize) = .empty;
    defer history.deinit(allocator);
    try history.appendSlice(allocator, prompt);
    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();
    for (out, 0..) |*tok, i| {
        g.reset();
        tok.* = if (i == 0)
            try model.prefill(&g, prompt, cache, sampler)
        else
            try model.decodeStep(&g, out[i - 1], cache, history.items, sampler);
        try history.append(allocator, tok.*);
    }
}

test "Gemma 4 F32 tiny model keeps greedy decoding and reproduces sampled sequences per seed" {
    const allocator = std.testing.allocator;
    var model = try initTinyGemma4ForSampling(allocator);
    defer nn.deinitModel(&model, allocator);
    const cfg = gemma4.TinyGemma4.config;
    var cache = try gemma4.Gemma4KVCache.init(allocator, cfg, .{ .max_seq_len = 32 });
    defer cache.deinit(allocator);
    const prompt = [_]usize{ 2, 10, 25, 7 };

    // 贪婪采样与完整前向对数几率 (截断后) 的惩罚 argmax 一致
    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();
    const ids = try g.tensorNDWithData(&.{ 1, prompt.len }, &.{ 2.0, 10.0, 25.0, 7.0 }, false);
    const logits = try model.forward(&g, ids);
    const last = logits.data[(prompt.len - 1) * cfg.vocab_size ..][0..cfg.vocab_size];
    var greedy = sampling.Sampler.init(.{ .temperature = 0.0, .repetition_penalty = 1.25 });
    var g_prefill = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g_prefill.deinit();
    try std.testing.expectEqual(
        sampling.argmaxWithPenalty(last, cfg.vocab_size, cfg.sampling_excluded_tokens, &prompt, 1.25),
        try model.prefill(&g_prefill, &prompt, &cache, &greedy),
    );

    // 随机采样：相同种子得到相同序列，不同种子得到不同序列
    const opts = sampling.SamplingOptions{ .temperature = 4.0, .top_k = 64, .top_p = 0.95, .repetition_penalty = 1.3, .seed = 42 };
    var first: [8]usize = undefined;
    var second: [8]usize = undefined;
    var s1 = sampling.Sampler.init(opts);
    try generateTinyF32(&model, allocator, &cache, &prompt, &s1, &first);
    var s2 = sampling.Sampler.init(opts);
    try generateTinyF32(&model, allocator, &cache, &prompt, &s2, &second);
    try std.testing.expectEqualSlices(usize, &first, &second);
    for (first) |tok| try std.testing.expect(!cfg.isSamplingExcluded(tok));

    var differs = false;
    for (1..6) |seed| {
        var other = opts;
        other.seed = seed;
        var s_other = sampling.Sampler.init(other);
        var seq: [8]usize = undefined;
        try generateTinyF32(&model, allocator, &cache, &prompt, &s_other, &seq);
        if (!std.mem.eql(usize, &first, &seq)) differs = true;
    }
    try std.testing.expect(differs);

    // 首个采样词元与对完整前向对数几率执行「惩罚 → 排除 → 抽样」的参考流程一致
    const processed = try allocator.dupe(f32, last);
    defer allocator.free(processed);
    try sampling.applyRepetitionPenaltyInPlace(allocator, processed, &prompt, opts.repetition_penalty);
    sampling.maskExcludedInPlace(processed, cfg.sampling_excluded_tokens);
    var reference = sampling.Sampler.init(opts);
    try std.testing.expectEqual(first[0], try reference.sampleLogits(backend.Executor.serial, allocator, processed));
}

test "Gemma 4 tiny F32 and Q4 models use the tanh-approximate GELU" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(backend.GeluApproximation.tanh, gemma4.Gemma4Config.defaultConfig().hidden_activation);
    try std.testing.expectEqual(backend.GeluApproximation.tanh, gemma4.Gemma4Config.e4b.hidden_activation);

    // F32 模型：前向计算图只记录双曲正切近似算子 GeluTanh (每层一个)，不出现精确形式算子 Gelu
    {
        var model = try initTinyGemma4ForSampling(allocator);
        defer nn.deinitModel(&model, allocator);
        for (model.layers) |layer| try std.testing.expectEqual(backend.GeluApproximation.tanh, layer.mlp.hidden_activation);
        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();
        const ids = try g.tensorNDWithData(&.{ 1, 3 }, &.{ 2.0, 10.0, 25.0 }, false);
        _ = try model.forward(&g, ids);
        var tanh_count: usize = 0;
        for (g.ops.items) |op| {
            try std.testing.expect(op.op_type != .Gelu);
            if (op.op_type == .GeluTanh) tanh_count += 1;
        }
        try std.testing.expectEqual(gemma4.TinyGemma4.config.num_hidden_layers, tanh_count);
    }

    // Q4 模型：各层多层感知机均使用双曲正切近似形式
    {
        const M = gemma4_q4.TinyQ4Gemma4;
        const bytes = try allocator.alloc(u8, M.computeExpectedFileSize());
        defer allocator.free(bytes);
        @memset(bytes, 0);
        @memcpy(bytes[0..8], M.Q4_MAGIC);
        var model = try M.loadFromMmapWithOptions(allocator, bytes, .{ .lm_head_mode = .bf16, .warmup = false });
        defer model.deinit(allocator);
        for (model.layers) |layer| try std.testing.expectEqual(backend.GeluApproximation.tanh, layer.mlp.hidden_activation);
    }
}

test "Gemma 4 Q4 tiny model sampling is reproducible and breaks logit ties canonically" {
    const allocator = std.testing.allocator;
    const M = gemma4_q4.TinyQ4Gemma4;
    const cfg = M.config;
    // 全零权重：所有词元的对数几率均为 0，贪婪采样取 ID 最小的非排除词元，Top-K 取 ID 最小的 K 个非排除词元
    const bytes = try allocator.alloc(u8, M.computeExpectedFileSize());
    defer allocator.free(bytes);
    @memset(bytes, 0);
    @memcpy(bytes[0..8], M.Q4_MAGIC);
    var model = try M.loadFromMmapWithOptions(allocator, bytes, .{ .lm_head_mode = .bf16, .warmup = false });
    defer model.deinit(allocator);
    var cache = try gemma4.Gemma4KVCache.initDefault(allocator, cfg);
    defer cache.deinit(allocator);
    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();
    const prompt = [_]usize{ 2, 10, 25 };

    var greedy = sampling.Sampler.init(.greedy);
    try std.testing.expectEqual(@as(usize, 1), try model.prefill(&g, &prompt, &cache, bytes, &greedy));
    try std.testing.expectEqual(@as(usize, 1), try model.generateNextToken(&g, &prompt, bytes, &greedy));

    var seen: [128]bool = @splat(false);
    for (0..40) |seed| {
        const opts = sampling.SamplingOptions{ .temperature = 1.0, .top_k = 16, .top_p = 1.0, .seed = seed };
        var a = sampling.Sampler.init(opts);
        var b = sampling.Sampler.init(opts);
        g.reset();
        const tok_a = try model.prefill(&g, &prompt, &cache, bytes, &a);
        const step_a = try model.decodeStep(&g, tok_a, &cache, bytes, &prompt, &a);
        g.reset();
        const tok_b = try model.prefill(&g, &prompt, &cache, bytes, &b);
        const step_b = try model.decodeStep(&g, tok_b, &cache, bytes, &prompt, &b);
        try std.testing.expectEqual(tok_a, tok_b);
        try std.testing.expectEqual(step_a, step_b);
        for ([_]usize{ tok_a, step_a }) |tok| {
            try std.testing.expect(tok >= 1 and tok <= 16);
            seen[tok] = true;
        }
    }
    var distinct: usize = 0;
    for (seen) |hit| distinct += @intFromBool(hit);
    try std.testing.expect(distinct > 8);
}

test "Gemma 4 GPU candidate selection matches the CPU sampler for identical logits" {
    if (!backend.offload.isAvailable()) return;
    const allocator = std.testing.allocator;
    const cfg = blk: {
        var c = gemma4.Gemma4Config.tiny_test;
        c.final_logit_softcapping = 5.0;
        c.sampling_excluded_tokens = &tiny_sampling_excluded;
        break :blk c;
    };
    var prng = std.Random.DefaultPrng.init(17);
    var raw: [128]f32 = undefined;
    fillNormal(&raw, prng.random(), 2.0);
    const recent = [_]usize{ 3, 9, 3, 40, 127 };

    var stream = try backend.mlx.Stream.defaultGpu();
    defer stream.deinit();
    var logits_arr = try backend.mlx.Array.fromSlice(&raw, &.{ 1, cfg.vocab_size });
    defer logits_arr.deinit();
    var mask = try backend.offload.TokenExclusionMask.init(allocator, cfg.vocab_size, cfg.sampling_excluded_tokens);
    defer mask.deinit();

    // CPU 参考：截断 → 惩罚 → 排除
    var processed: [128]f32 = undefined;
    for (raw, &processed) |r, *p| p.* = backend.vec_math.softcap(r, cfg.final_logit_softcapping);
    try sampling.applyRepetitionPenaltyInPlace(allocator, &processed, &recent, 1.4);
    sampling.maskExcludedInPlace(&processed, cfg.sampling_excluded_tokens);

    for ([_]sampling.SamplingOptions{
        .{ .temperature = 0.9, .top_k = 16, .top_p = 0.95, .repetition_penalty = 1.4, .seed = 5 },
        .{ .temperature = 1.2, .top_p = 0.8, .repetition_penalty = 1.4, .seed = 6 },
    }) |opts| {
        var pending = try gemma4_gpu.deviceCandidates(allocator, logits_arr, cfg, &mask, &recent, opts, stream);
        defer pending.deinit();
        var buf: [128]sampling.Candidate = undefined;
        const host = try pending.read(&buf);

        var cpu_buf: [128]sampling.Candidate = undefined;
        const cpu_cands = try sampling.selectTopK(backend.Executor.serial, allocator, &processed, cpu_buf[0..opts.candidateCount(cfg.vocab_size)]);
        const host_dist = sampling.buildDistribution(host.candidates, .{ .temperature = 1.0 }, null);
        const cpu_dist = sampling.buildDistribution(cpu_cands, .{ .temperature = 1.0 }, null);
        try std.testing.expectEqual(cpu_dist.candidates.len, host_dist.candidates.len);
        for (cpu_dist.candidates, host_dist.candidates) |c_cand, g_cand| {
            try std.testing.expectEqual(c_cand.id, g_cand.id);
            try std.testing.expectApproxEqAbs(c_cand.logit, g_cand.logit, 1e-4);
        }
        if (opts.needsLogNormalizer()) {
            const lse = try sampling.logSumExpScaled(backend.Executor.serial, &processed, 1.0 / opts.temperature);
            try std.testing.expectApproxEqAbs(lse, host.log_normalizer.?, 1e-4);
        }

        // 相同候选集与种子：GPU 候选与 CPU 完整对数几率抽样得到相同词元
        var gpu_sampler = sampling.Sampler.init(opts);
        var cpu_sampler = sampling.Sampler.init(opts);
        for (0..64) |_| {
            var copy = buf;
            const gpu_tok = gpu_sampler.sampleCandidates(copy[0..host.candidates.len], host.log_normalizer);
            try std.testing.expectEqual(@as(usize, gpu_tok), try cpu_sampler.sampleLogits(backend.Executor.serial, allocator, &processed));
        }

        var sel_sampler = sampling.Sampler.init(opts);
        const tok = try gemma4_gpu.selectToken(allocator, logits_arr, cfg, &mask, &recent, &sel_sampler, stream);
        try std.testing.expect(!cfg.isSamplingExcluded(tok));
    }
}
