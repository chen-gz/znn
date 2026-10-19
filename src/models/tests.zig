const std = @import("std");
const backend = @import("../backend.zig");
const autodiff = @import("../autodiff.zig");
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");
const models = @import("../models.zig");

const Tensor = tensor.Tensor;
const gemma4 = models.gemma4;
const gemma4_q4 = models.gemma4_q4;

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

            var mmap_model = try gemma4_q4.TinyQ4Gemma4.loadFromMmap(allocator, mock_mmap);
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
            }

            cache.reset();
            try std.testing.expectEqual(@as(usize, 0), cache.layers[0].curr_len);
        }
    }

