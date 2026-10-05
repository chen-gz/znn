const std = @import("std");
const autodiff = @import("../autodiff.zig");
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");
const models = @import("../models.zig");

const Tensor = tensor.Tensor;
const gemma4 = models.gemma4;

test "Gemma 4 specialized operators, modules, and CausalLM forward" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // 1. GemmaRMSNorm 单元测试 (含 unit offset 1.0 + weight 特性)
    {
        var norm = try gemma4.GemmaRMSNorm.init(allocator, 4, 1e-6);
        defer nn.deinitModel(&norm, allocator);
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();

        const x = try g.tensorNDWithData(&.{ 1, 4 }, &.{ 2.0, 2.0, 2.0, 2.0 }, false);
        const y = try norm.forward(&g, x);
        // x rms = 2.0, x / rms = 1.0, weight 为 1.0, 结果为 1.0
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), y.data[0], 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), y.data[3], 1e-4);
    }

    // 2. Gemma 专用 Split-Half RoPE 算子单测
    {
        var g = autodiff.Graph.init(allocator);
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

        var g = autodiff.Graph.init(allocator);
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

        var g = autodiff.Graph.init(allocator);
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

        var g = autodiff.Graph.init(allocator);
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
    }

    // 6. 4-bit (Q4) 分块量化与 Q4Linear 推理测试
    {
        var q4_layer = try gemma4.Q4Linear.init(allocator, 32, 64);
        defer q4_layer.deinit(allocator);

        // 构造浮点权重并量化
        const weights = try allocator.alloc(f32, 32 * 64);
        defer allocator.free(weights);
        for (weights, 0..) |*w_val, i| {
            w_val.* = @as(f32, @floatFromInt(i % 10)) * 0.1 - 0.5;
        }
        q4_layer.quantizeFromF32(weights);

        var g = autodiff.Graph.init(allocator);
        defer g.deinit();

        const x = try g.ones(&.{ 1, 32 }, false);
        const y = try q4_layer.forward(&g, x);
        try std.testing.expectEqual(@as(usize, 2), y.shape.len);
        try std.testing.expectEqual(@as(usize, 1), y.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 64), y.shape.dims[1]);
    }

    // 7. 端到端 TinyQ4Gemma4 模型推理测试
    {
        var model = try gemma4.TinyQ4Gemma4.init(allocator);
        defer model.deinit(allocator);

        var g = autodiff.Graph.init(allocator);
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
    }
}
