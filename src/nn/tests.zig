const std = @import("std");
const testing_init = @import("testing_init.zig");
const autodiff = @import("../autodiff.zig");
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;

const core = nn.core;
const activations = nn.activations;
const normalization = nn.normalization;
const recurrent = nn.recurrent;
const transformer = nn.transformer;
const serialization = nn.serialization;
const visualization = nn.visualization;
const graph_ir = nn.graph_ir;
const generateJson = nn.generateJson;
const exportJson = nn.exportJson;
const NodeKind = nn.NodeKind;
const NodeStatus = nn.NodeStatus;
const FlowNodeKind = nn.FlowNodeKind;
const EdgeKind = nn.EdgeKind;
const NodeData = nn.NodeData;

const init_mod = nn.init_mod;
const Nonlinearity = nn.Nonlinearity;
const calculateGain = nn.calculateGain;
const InitMethod = nn.InitMethod;
const InitOptions = nn.InitOptions;
const normalRandom = nn.normalRandom;
const initWeights = nn.initWeights;
const createPersistentTensor = nn.createPersistentTensor;
const freePersistentTensor = nn.freePersistentTensor;
const Linear = nn.Linear;
const Conv1D = nn.Conv1D;
const Conv2D = nn.Conv2D;
const ConvTranspose1D = nn.ConvTranspose1D;
const ConvTranspose2D = nn.ConvTranspose2D;
const Module = nn.Module;
const deinitModel = nn.deinitModel;
const zeroGradModel = nn.zeroGradModel;
const parameters = nn.parameters;
const Sequential = nn.Sequential;
const sequential = nn.sequential;

const ReLU = nn.ReLU;
const GELU = nn.GELU;
const Sigmoid = nn.Sigmoid;
const Tanh = nn.Tanh;
const LeakyReLU = nn.LeakyReLU;
const SiLU = nn.SiLU;
const Swish = nn.Swish;

const RMSNorm = nn.RMSNorm;
const LayerNorm = nn.LayerNorm;
const BatchNorm2d = nn.BatchNorm2d;
const Dropout = nn.Dropout;
const AvgPool2D = nn.AvgPool2D;

const RNNCell = nn.RNNCell;
const RNN = nn.RNN;
const LSTMState = nn.LSTMState;
const LSTMCell = nn.LSTMCell;
const LSTM = nn.LSTM;
const StackedLSTM = nn.StackedLSTM;
const GRUCell = nn.GRUCell;
const GRU = nn.GRU;

const Embedding = nn.Embedding;
const KVCache = nn.KVCache;
const MLP = nn.MLP;
const SwiGLU = nn.SwiGLU;
const MoELayer = nn.MoELayer;
const CausalSelfAttention = nn.CausalSelfAttention;
const applyRope1D = nn.applyRope1D;
const MLACache = nn.MLACache;
const MLALayer = nn.MLALayer;
const TransformerBlock = nn.TransformerBlock;
const TransformerDecoder = nn.TransformerDecoder;
const GPTConfig = nn.GPTConfig;
const GPT = nn.GPT;
const LoRALinear = nn.LoRALinear;
const maskedCrossEntropyLoss = nn.maskedCrossEntropyLoss;
const maskedCrossEntropyLossGraph = nn.maskedCrossEntropyLossGraph;
const dpoLoss = nn.dpoLoss;
const dpoLossGraph = nn.dpoLossGraph;
const computeGroupAdvantages = nn.computeGroupAdvantages;
const computeGRPOLoss = nn.computeGRPOLoss;
const grpoLoss = nn.grpoLoss;
const grpoLossGraph = nn.grpoLossGraph;
const sampleTopP = nn.sampleTopP;
const sampleTopK = nn.sampleTopK;

const saveModel = nn.saveModel;
const loadModel = nn.loadModel;


// 单元测试套件 (Unit Tests)
// ============================================================================

test {
    _ = @import("tests_init.zig");
    _ = @import("tests_module.zig");
    _ = @import("tests_vis.zig");
}

test "Embedding Module" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var emb = try Embedding.init(arena, 10, 4);
    emb.resetParameters(random, .{});
    defer nn.deinitModel(&emb, arena);

    var x = try createPersistentTensor(arena, 2, 3, false);
    defer freePersistentTensor(arena, x);
    x.data[0] = 0; x.data[1] = 1; x.data[2] = 2;
    x.data[3] = 3; x.data[4] = 4; x.data[5] = 5;

    var y_eager_graph = autodiff.Graph.init(arena, .{ .enable_grad = false });
    defer y_eager_graph.deinit();
    const y_eager = try emb.forward(&y_eager_graph, x);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 4}, y_eager.shape.dims[0..y_eager.shape.len]);
}

test "Embedding Module Graph Mode" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var emb = try Embedding.init(arena, 10, 4);
    emb.resetParameters(random, .{});
    defer nn.deinitModel(&emb, arena);

    var graph = autodiff.Graph.initDefault(arena);
    defer graph.deinit();

    const x = try graph.tensorND(&.{2, 3}, false);
    x.data[0] = 0; x.data[1] = 1; x.data[2] = 2;
    x.data[3] = 3; x.data[4] = 4; x.data[5] = 5;

    const y = try emb.forward(&graph, x);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 4}, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    for (0..6) |i| {
        for (0..4) |j| {
            try std.testing.expectEqual(@as(f32, 1.0), emb.weight.grad[i * 4 + j]);
        }
    }
    for (6..10) |i| {
        for (0..4) |j| {
            try std.testing.expectEqual(@as(f32, 0.0), emb.weight.grad[i * 4 + j]);
        }
    }
}

test "RMSNorm Module" {
    const arena = std.testing.allocator;
    var norm = try RMSNorm.init(arena, 4, 1e-5);
    defer nn.deinitModel(&norm, arena);

    var x = try createPersistentTensor(arena, 2, 4, false);
    defer freePersistentTensor(arena, x);
    x.data[0] = 1.0; x.data[1] = 2.0; x.data[2] = 3.0; x.data[3] = 4.0;
    x.data[4] = 5.0; x.data[5] = 6.0; x.data[6] = 7.0; x.data[7] = 8.0;

    var y_eager_graph = autodiff.Graph.init(arena, .{ .enable_grad = false });
    defer y_eager_graph.deinit();
    const y_eager = try norm.forward(&y_eager_graph, x);
    try std.testing.expectEqualSlices(usize, &.{2, 4}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.initDefault(arena);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 4}, true);
    @memcpy(x_node.data, x.data);

    const y = try norm.forward(&graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{2, 4}, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    var norm_g_grad_sum: f32 = 0.0;
    for (norm.weight.grad) |g| norm_g_grad_sum += @abs(g);
    try std.testing.expect(norm_g_grad_sum > 0.0);

    var x_grad_sum: f32 = 0.0;
    for (x_node.grad) |g| x_grad_sum += @abs(g);
    try std.testing.expect(x_grad_sum > 0.0);
}

test "MLP Module" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var mlp = try MLP.init(arena, 4, 8);
    try testing_init.initFromOnes(&mlp, arena, random, &.{ 2, 4 });
    defer nn.deinitModel(&mlp, arena);

    const x_3d = try arena.create(Tensor);
    const shape = Shape.init(&.{2, 3, 4});
    x_3d.* = Tensor{
        .data = try arena.alloc(f32, 24),
        .grad = &.{},
        .shape = shape,
        .strides = tensor.computeContiguousStrides(shape),
        .requires_grad = false,
        .creator = null,
    };
    defer {
        arena.free(x_3d.data);
        arena.destroy(x_3d);
    }
    for (x_3d.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    var y_eager_graph = autodiff.Graph.init(arena, .{ .enable_grad = false });
    defer y_eager_graph.deinit();
    const y_eager = try mlp.forward(&y_eager_graph, x_3d);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 4}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.initDefault(arena);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 3, 4}, true);
    @memcpy(x_node.data, x_3d.data);

    const y = try mlp.forward(&graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 4}, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    var w1_grad_sum: f32 = 0.0;
    for (mlp.c_fc.weight.grad) |g| w1_grad_sum += @abs(g);
    try std.testing.expect(w1_grad_sum > 0.0);
}

test "CausalSelfAttention Module" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var attn = try CausalSelfAttention.init(arena, 8, 2);
    try testing_init.initFromOnes(&attn, arena, random, &.{ 1, 2, 8 });
    defer nn.deinitModel(&attn, arena);

    const x_3d = try arena.create(Tensor);
    const shape = Shape.init(&.{2, 3, 8});
    x_3d.* = Tensor{
        .data = try arena.alloc(f32, 48),
        .grad = &.{},
        .shape = shape,
        .strides = tensor.computeContiguousStrides(shape),
        .requires_grad = false,
        .creator = null,
    };
    defer {
        arena.free(x_3d.data);
        arena.destroy(x_3d);
    }
    for (x_3d.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    var y_eager_graph = autodiff.Graph.init(arena, .{ .enable_grad = false });
    defer y_eager_graph.deinit();
    const y_eager = try attn.forward(&y_eager_graph, x_3d);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 8}, y_eager.shape.dims[0..y_eager.shape.len]);

    const backend = @import("../backend.zig");
    if (backend.mlx.has_mlx) {
        var y_gpu_graph = autodiff.Graph.init(arena, .{ .enable_grad = false, .device = .gpu });
        defer y_gpu_graph.deinit();
        const y_gpu = try attn.forward(&y_gpu_graph, x_3d);
        try std.testing.expectEqualSlices(usize, &.{2, 3, 8}, y_gpu.shape.dims[0..y_gpu.shape.len]);
        for (y_eager.data, y_gpu.data) |e_val, g_val| {
            try std.testing.expectApproxEqAbs(e_val, g_val, 1e-3);
        }
    }

    var graph = autodiff.Graph.initDefault(arena);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 3, 8}, true);
    @memcpy(x_node.data, x_3d.data);

    const y = try attn.forward(&graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 8}, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    var q_grad_sum: f32 = 0.0;
    for (attn.q_attn.weight.grad) |g| q_grad_sum += @abs(g);
    try std.testing.expect(q_grad_sum > 0.0);
}

test "GPT Module" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const config = GPTConfig{
        .vocab_size = 10,
        .block_size = 5,
        .n_embd = 8,
        .n_head = 2,
        .n_layer = 2,
    };

    var gpt = try GPT(config).init(arena);
    try testing_init.initFromOnes(&gpt, arena, random, &.{ 1, 2 });
    defer deinitModel(&gpt, arena);

    const x = try arena.create(Tensor);
    const shape = Shape.init(&.{2, 3});
    x.* = Tensor{
        .data = try arena.alloc(f32, 6),
        .grad = &.{},
        .shape = shape,
        .strides = tensor.computeContiguousStrides(shape),
        .requires_grad = false,
        .creator = null,
    };
    defer {
        arena.free(x.data);
        arena.destroy(x);
    }
    x.data[0] = 0; x.data[1] = 1; x.data[2] = 2;
    x.data[3] = 3; x.data[4] = 4; x.data[5] = 5;

    var y_eager_graph = autodiff.Graph.init(arena, .{ .enable_grad = false });
    defer y_eager_graph.deinit();
    const y_eager = try gpt.forward(&y_eager_graph, x);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 10}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.initDefault(arena);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 3}, false);
    @memcpy(x_node.data, x.data);

    const y = try gpt.forward(&graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 10}, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    var token_embedding_grad_sum: f32 = 0.0;
    for (gpt.token_embedding.weight.grad) |g| token_embedding_grad_sum += @abs(g);
    try std.testing.expect(token_embedding_grad_sum > 0.0);
}

test "GPT Module Save and Load" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const config = GPTConfig{
        .vocab_size = 10,
        .block_size = 5,
        .n_embd = 8,
        .n_head = 2,
        .n_layer = 2,
    };

    var gpt = try GPT(config).init(arena);
    try testing_init.initFromOnes(&gpt, arena, random, &.{ 1, 2 });
    defer deinitModel(&gpt, arena);

    try saveModel(&gpt, std.testing.io, "test_gpt_model.safetensors", arena);
    defer {
        std.Io.Dir.cwd().deleteFile(std.testing.io, "test_gpt_model.safetensors") catch {};
    }

    var gpt2 = try GPT(config).init(arena);
    try testing_init.initFromOnes(&gpt2, arena, random, &.{ 1, 2 });
    defer deinitModel(&gpt2, arena);

    try loadModel(&gpt2, std.testing.io, "test_gpt_model.safetensors", arena);

    for (gpt.token_embedding.weight.data, gpt2.token_embedding.weight.data) |w1, w2| {
        try std.testing.expectEqual(w1, w2);
    }
}

test "Sequential container chaining" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var seq = sequential(.{
        try Linear.init(allocator, 10, 20),
        ReLU{},
        try Linear.init(allocator, 20, 5),
    });
    defer nn.deinitModel(&seq, allocator);
    try testing_init.initFromOnes(&seq, allocator, random, &.{ 2, 10 });

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const x = try graph.tensor(2, 10, false);
    @memset(x.data, 0.5);

    const y = try seq.forward(&graph, x);
    try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    var grad_sum: f32 = 0.0;
    for (seq.layers.@"0".weight.grad) |g| grad_sum += @abs(g);
    try std.testing.expect(grad_sum > 0.0);

    // Test parameter collection on Sequential container
    const params = try parameters(&seq, allocator);
    defer allocator.free(params);
    // Two Linear layers each with weight and bias = 4 parameter tensors
    try std.testing.expectEqual(@as(usize, 4), params.len);

    // Inference forward on a no-grad graph without memory leak
    const eager_in = try createPersistentTensor(allocator, 2, 10, false);
    defer freePersistentTensor(allocator, eager_in);
    @memset(eager_in.data, 0.5);

    var eager_out_graph = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer eager_out_graph.deinit();
    const eager_out = try seq.forward(&eager_out_graph, eager_in);
    try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, eager_out.shape.dims[0..eager_out.shape.len]);
}

test "SwiGLU forward and backward autograd" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var swiglu = try SwiGLU.init(allocator, 4, 8);
    try testing_init.initFromOnes(&swiglu, allocator, random, &.{ 2, 4 });
    defer nn.deinitModel(&swiglu, allocator);

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const x = try graph.tensor(2, 4, true);
    @memset(x.data, 0.5);

    const y = try swiglu.forward(&graph, x);
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    var gate_grad: f32 = 0.0;
    for (swiglu.w_gate.weight.grad) |g| gate_grad += @abs(g);
    try std.testing.expect(gate_grad > 0.0);

    var up_grad: f32 = 0.0;
    for (swiglu.w_up.weight.grad) |g| up_grad += @abs(g);
    try std.testing.expect(up_grad > 0.0);

    var down_grad: f32 = 0.0;
    for (swiglu.w_down.weight.grad) |g| down_grad += @abs(g);
    try std.testing.expect(down_grad > 0.0);
}

test "LoRALinear forward and fuse" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var lora = try LoRALinear.init(allocator, 4, 4, 2, 4.0);
    lora.resetParameters(random, .{});
    defer nn.deinitModel(&lora, allocator);

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const x = try graph.tensor(2, 4, false);
    @memset(x.data, 1.0);

    // Initial forward: since B=0, LoRA output must match Base Weight output exactly
    const y = try lora.forward(&graph, x);
    const base_y = try x.matmul(lora.weight, allocator);
    defer tensor.free(allocator, base_y);

    for (y.data, base_y.data) |y_val, by_val| {
        try std.testing.expectApproxEqAbs(by_val, y_val, 1e-5);
    }

    // Set B to non-zero and test backward
    @memset(lora.lora_b.data, 0.5);
    @memset(y.grad, 1.0);
    try graph.backward(y);

    var lora_a_grad: f32 = 0.0;
    for (lora.lora_a.grad) |g| lora_a_grad += @abs(g);
    try std.testing.expect(lora_a_grad > 0.0);

    // Test fuse
    lora.fuse();
    @memset(lora.lora_b.data, 0.0);
    const fused_out = try x.matmul(lora.weight, allocator);
    defer tensor.free(allocator, fused_out);
    try std.testing.expect(fused_out.data[0] != base_y.data[0]);
}

test "maskedCrossEntropyLoss and dpoLoss" {
    const allocator = std.testing.allocator;
    const logits = try tensor.zeros(allocator, &.{ 3, 4 });
    defer tensor.free(allocator, logits);

    @memcpy(logits.data, &[_]f32{
        2.0, 1.0, 0.1, 0.0, // Token 0 (Prompt, Mask=0)
        0.5, 3.0, 0.2, 0.1, // Token 1 (Response, Mask=1, Target=1)
        0.1, 0.2, 4.0, 0.1, // Token 2 (Response, Mask=1, Target=2)
    });

    const targets = [_]u32{ 0, 1, 2 };
    const mask = [_]f32{ 0.0, 1.0, 1.0 };

    const loss = try maskedCrossEntropyLoss(logits, &targets, &mask, allocator);
    try std.testing.expect(loss > 0.0 and loss < 1.0);

    // Graph-integrated maskedCrossEntropyLoss
    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const g_logits = try graph.tensorNDWithData(&.{ 3, 4 }, logits.data, true);
    const g_loss = try maskedCrossEntropyLossGraph(&graph, g_logits, &targets, &mask);
    try std.testing.expectApproxEqAbs(loss, g_loss.data[0], 1e-5);
    try graph.backward(g_loss);

    // Token 0 has mask=0 -> gradient must be 0
    for (g_logits.grad[0..4]) |g| {
        try std.testing.expectEqual(@as(f32, 0.0), g);
    }
    // Token 1 and 2 have mask=1 -> target class gradient must be negative
    try std.testing.expect(g_logits.grad[1 * 4 + 1] < 0.0);
    try std.testing.expect(g_logits.grad[2 * 4 + 2] < 0.0);

    // Large vocabulary (> 256) softmaxCrossEntropy test with u32 labels
    const big_logits = try graph.zeros(&.{ 2, 300 }, true);
    big_logits.data[0 * 300 + 280] = 5.0;
    big_logits.data[1 * 300 + 299] = 5.0;
    const big_targets = [_]u32{ 280, 299 };
    const big_loss = try graph.softmaxCrossEntropy(big_logits, &big_targets);
    try graph.backward(big_loss);
    try std.testing.expect(big_logits.grad[0 * 300 + 280] < 0.0);
    try std.testing.expect(big_logits.grad[1 * 300 + 299] < 0.0);

    // DPO Loss test (Eager and Graph autograd)
    const pi_w = [_]f32{-1.2};
    const pi_l = [_]f32{-2.8};
    const ref_w = [_]f32{-1.5};
    const ref_l = [_]f32{-2.0};
    const d_loss = dpoLoss(&pi_w, &pi_l, &ref_w, &ref_l, 0.1);
    try std.testing.expect(d_loss > 0.0);

    const g_pi_w = try graph.tensorNDWithData(&.{1}, &pi_w, true);
    const g_pi_l = try graph.tensorNDWithData(&.{1}, &pi_l, true);
    const g_dpo = try dpoLossGraph(&graph, g_pi_w, g_pi_l, &ref_w, &ref_l, 0.1);
    try std.testing.expectApproxEqAbs(d_loss, g_dpo.data[0], 1e-6);
    try graph.backward(g_dpo);
    // Increasing chosen log-prob decreases DPO loss (grad < 0); increasing rejected increases loss (grad > 0)
    try std.testing.expect(g_pi_w.grad[0] < 0.0);
    try std.testing.expect(g_pi_l.grad[0] > 0.0);
    try std.testing.expectApproxEqAbs(-g_pi_w.grad[0], g_pi_l.grad[0], 1e-6);
}

test "LayerNorm forward and backward autograd" {
    const allocator = std.testing.allocator;
    var ln = try LayerNorm.init(allocator, 4, 1e-5);
    defer nn.deinitModel(&ln, allocator);

    const x = try tensor.zeros(allocator, &.{ 2, 4 });
    defer tensor.free(allocator, x);
    @memcpy(x.data, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0 });

    var y_graph = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer y_graph.deinit();
    const y = try ln.forward(&y_graph, x);

    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, y.shape.dims[0..2]);
    var sum: f32 = 0.0;
    for (y.data[0..4]) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sum / 4.0, 1e-4);

    // Graph mode forward + backward
    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const gx = try graph.tensorNDWithData(&.{ 2, 4 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 1.0, 3.0 }, true);
    const gy = try ln.forward(&graph, gx);
    try std.testing.expect(gy.creator != null);
    try std.testing.expectEqual(autodiff.OpType.LayerNorm, autodiff.Op.creatorOf(gy).?.op_type);

    const target = try graph.tensorNDWithData(&.{ 2, 4 }, &[_]f32{ 0.5, -0.5, 1.0, -1.0, -0.5, 0.5, -1.0, 1.0 }, false);
    const loss = try graph.mseLoss(gy, target);
    try graph.backward(loss);

    var x_grad_norm: f32 = 0.0;
    for (gx.grad) |g| x_grad_norm += @abs(g);
    try std.testing.expect(x_grad_norm > 1e-4);

    var w_grad_norm: f32 = 0.0;
    for (ln.weight.grad) |g| w_grad_norm += @abs(g);
    try std.testing.expect(w_grad_norm > 1e-4);

    var b_grad_norm: f32 = 0.0;
    for (ln.bias.grad) |g| b_grad_norm += @abs(g);
    try std.testing.expect(b_grad_norm > 1e-4);
}

test "BatchNorm2d forward and backward autograd" {
    const allocator = std.testing.allocator;
    var bn = try BatchNorm2d.init(allocator, 2, 1e-5, 0.1);
    defer nn.deinitModel(&bn, allocator);

    const x = try tensor.zeros(allocator, &.{ 2, 2, 2, 2 });
    defer tensor.free(allocator, x);
    for (x.data, 0..) |*p, i| p.* = @as(f32, @floatFromInt(i));

    var y_graph = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer y_graph.deinit();
    const y = try bn.forward(&y_graph, x);

    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 2 }, y.shape.dims[0..4]);

    // Graph mode forward + backward
    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const gx = try graph.tensorND(&.{ 2, 2, 2, 2 }, true);
    for (gx.data, 0..) |*p, i| {
        const fi = @as(f32, @floatFromInt(i));
        p.* = @sin(fi * 1.3) + 0.2 * fi;
    }
    const gy = try bn.forward(&graph, gx);
    try std.testing.expect(gy.creator != null);
    try std.testing.expectEqual(autodiff.OpType.BatchNorm2d, autodiff.Op.creatorOf(gy).?.op_type);

    for (gy.grad, 0..) |*g, i| {
        const fi = @as(f32, @floatFromInt(i));
        g.* = @cos(fi * 0.7) - 0.1 * fi;
    }
    try graph.backwardWithGrad(gy);

    var gx_grad_norm: f32 = 0.0;
    for (gx.grad) |g| gx_grad_norm += @abs(g);
    try std.testing.expect(gx_grad_norm > 1e-4);

    var gamma_grad_norm: f32 = 0.0;
    for (bn.gamma.grad) |g| gamma_grad_norm += @abs(g);
    try std.testing.expect(gamma_grad_norm > 1e-4);

    var beta_grad_norm: f32 = 0.0;
    for (bn.beta.grad) |g| beta_grad_norm += @abs(g);
    try std.testing.expect(beta_grad_norm > 1e-4);
}

test "BatchNorm1d and GroupNorm forward and backward autograd" {
    const allocator = std.testing.allocator;
    var bn1 = try nn.BatchNorm1d.init(allocator, 2, .{});
    defer nn.deinitModel(&bn1, allocator);

    var gn = try nn.GroupNorm.init(allocator, 2, 4, .{});
    defer nn.deinitModel(&gn, allocator);

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const gx1 = try graph.tensorND(&.{ 2, 2, 3 }, true);
    for (gx1.data, 0..) |*p, i| {
        const fi = @as(f32, @floatFromInt(i));
        p.* = @sin(fi * 1.1) + 0.3 * fi;
    }
    const gy1 = try bn1.forward(&graph, gx1);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3 }, gy1.shape.dims[0..3]);
    try std.testing.expectEqual(autodiff.OpType.BatchNorm1d, autodiff.Op.creatorOf(gy1).?.op_type);
    for (gy1.grad, 0..) |*g, i| g.* = @as(f32, @floatFromInt(i + 1)) * 0.1;
    try graph.backwardWithGrad(gy1);
    try std.testing.expect(@abs(gx1.grad[0]) + @abs(gx1.grad[1]) > 1e-5);
    try std.testing.expect(@abs(bn1.gamma.grad[0]) > 1e-5);
    try std.testing.expect(@abs(bn1.beta.grad[0]) > 1e-5);

    const gx2 = try graph.tensorND(&.{ 2, 4, 2, 2 }, true);
    for (gx2.data, 0..) |*p, i| {
        const fi = @as(f32, @floatFromInt(i));
        p.* = @cos(fi * 0.9) - 0.15 * fi;
    }
    const gy2 = try gn.forward(&graph, gx2);
    try std.testing.expectEqualSlices(usize, &.{ 2, 4, 2, 2 }, gy2.shape.dims[0..4]);
    try std.testing.expectEqual(autodiff.OpType.GroupNorm, autodiff.Op.creatorOf(gy2).?.op_type);
    for (gy2.grad, 0..) |*g, i| g.* = @as(f32, @floatFromInt(i + 1)) * 0.05;
    try graph.backwardWithGrad(gy2);
    try std.testing.expect(@abs(gx2.grad[0]) + @abs(gx2.grad[1]) > 1e-5);
    try std.testing.expect(@abs(gn.weight.grad[0]) > 1e-5);
    try std.testing.expect(@abs(gn.bias.grad[0]) > 1e-5);
}

test "Dropout and AvgPool2D forward and backward passes" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    const drop = Dropout.init(0.2);
    const x = try tensor.zeros(allocator, &.{ 1, 10 });
    defer tensor.free(allocator, x);
    @memset(x.data, 1.0);

    var y_drop_graph = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer y_drop_graph.deinit();
    const y_drop = try drop.forward(&y_drop_graph, x, rand);
    try std.testing.expectEqual(10, y_drop.data.len);

    const pool = AvgPool2D.init(2, .{});
    const img = try tensor.zeros(allocator, &.{ 1, 1, 4, 4 });
    defer tensor.free(allocator, img);
    @memset(img.data, 4.0);

    var pooled_graph = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer pooled_graph.deinit();
    const pooled = try pool.forward(&pooled_graph, img);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2, 2 }, pooled.shape.dims[0..4]);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), pooled.data[0], 1e-5);

    // Graph mode Dropout + AvgPool2D backward
    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const g_img = try graph.tensorND(&.{ 1, 1, 4, 4 }, true);
    for (g_img.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1));

    const g_dropped = try drop.forward(&graph, g_img, rand);
    try std.testing.expect(g_dropped.creator != null);
    try std.testing.expectEqual(autodiff.OpType.Dropout, autodiff.Op.creatorOf(g_dropped).?.op_type);

    const g_pooled = try pool.forward(&graph, g_dropped);
    try std.testing.expect(g_pooled.creator != null);
    try std.testing.expectEqual(autodiff.OpType.AvgPool2D, autodiff.Op.creatorOf(g_pooled).?.op_type);

    @memset(g_pooled.grad, 1.0);
    try graph.backwardWithGrad(g_pooled);

    var img_grad_sum: f32 = 0.0;
    for (g_img.grad) |g| img_grad_sum += g;
    try std.testing.expect(img_grad_sum > 0.0);

    // Test MaxPool1D, AvgPool1D, and MaxPool2D modules
    const mp1d = nn.MaxPool1D.init(2, .{});
    const ap1d = nn.AvgPool1D.init(2, .{});
    const mp2d = nn.MaxPool2D.init(2, .{});
    const seq1d = try graph.array(&.{ 1, 1, 4 }, &[_]f32{ 1.0, 5.0, 3.0, 7.0 }, true);
    const out_mp1d = try mp1d.forward(&graph, seq1d);
    const out_ap1d = try ap1d.forward(&graph, seq1d);
    const out_mp2d = try mp2d.forward(&graph, g_img);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2 }, out_mp1d.shape.dims[0..3]);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2 }, out_ap1d.shape.dims[0..3]);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2, 2 }, out_mp2d.shape.dims[0..4]);

    const aap1d = nn.AdaptiveAvgPool1D.init(1);
    const aap2d = nn.AdaptiveAvgPool2D.initSquare(1);
    const out_aap1d = try aap1d.forward(&graph, seq1d);
    const out_aap2d = try aap2d.forward(&graph, g_img);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1 }, out_aap1d.shape.dims[0..3]);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1, 1 }, out_aap2d.shape.dims[0..4]);
}

test "KVCache initialization and reset" {
    const allocator = std.testing.allocator;
    var cache = try KVCache.init(allocator, 1, 4, 128, 32);
    defer cache.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 1, 4, 128, 32 }, cache.k.shape.dims[0..4]);
    cache.curr_len = 50;
    cache.reset();
    try std.testing.expectEqual(0, cache.curr_len);
}

test "RNNCell and RNN forward and backward autograd" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(123);
    const random = prng.random();

    // 1. RNNCell test
    var cell = try RNNCell.init(allocator, 4, 3);
    try testing_init.initRecurrent(&cell, allocator, random);
    defer nn.deinitModel(&cell, allocator);

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const x0 = try graph.tensorND(&.{ 2, 4 }, true);
    for (x0.data, 0..) |*p, i| p.* = @as(f32, @floatFromInt(i)) * 0.1;
    const h0 = try graph.zeros(&.{ 2, 3 }, true);

    const h1 = try cell.forward(&graph, x0, h0);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, h1.shape.dims[0..2]);

    @memset(h1.grad, 1.0);
    try graph.backward(h1);

    var grad_sum: f32 = 0.0;
    for (x0.grad) |g| grad_sum += @abs(g);
    try std.testing.expect(grad_sum > 1e-4);

    // 2. RNN sequence container test
    var rnn = try RNN.init(allocator, 4, 3);
    try testing_init.initRecurrent(&rnn, allocator, random);
    defer nn.deinitModel(&rnn, allocator);

    var graph_seq = autodiff.Graph.initDefault(allocator);
    defer graph_seq.deinit();

    const x_seq_0 = try graph_seq.tensorND(&.{ 2, 4 }, true);
    const x_seq_1 = try graph_seq.tensorND(&.{ 2, 4 }, true);
    @memset(x_seq_0.data, 0.5);
    @memset(x_seq_1.data, -0.5);

    const inputs = [_]*Tensor{ x_seq_0, x_seq_1 };
    const res = try rnn.forward(&graph_seq, &inputs, null);

    try std.testing.expectEqual(2, res.outputs.len);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, res.h_n.shape.dims[0..2]);

    @memset(res.h_n.grad, 1.0);
    try graph_seq.backward(res.h_n);

    var x_seq_grad_sum: f32 = 0.0;
    for (x_seq_0.grad) |g| x_seq_grad_sum += @abs(g);
    for (x_seq_1.grad) |g| x_seq_grad_sum += @abs(g);
    try std.testing.expect(x_seq_grad_sum > 1e-4);
}

test "LSTMCell and LSTM forward and backward autograd" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(456);
    const random = prng.random();

    // 1. LSTMCell test
    var cell = try LSTMCell.init(allocator, 4, 3);
    cell.resetParameters(random, .{});
    defer nn.deinitModel(&cell, allocator);

    // Verify forget gate bias is initialized to 1.0
    for (cell.w_ih_f.bias.data) |b| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), b, 1e-6);
    }

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const x0 = try graph.tensorND(&.{ 2, 4 }, true);
    for (x0.data, 0..) |*p, i| p.* = @as(f32, @floatFromInt(i)) * 0.1;
    const h0 = try graph.zeros(&.{ 2, 3 }, true);
    const c0 = try graph.zeros(&.{ 2, 3 }, true);

    const state = try cell.forward(&graph, x0, h0, c0);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, state.h.shape.dims[0..2]);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, state.c.shape.dims[0..2]);

    @memset(state.h.grad, 1.0);
    try graph.backward(state.h);

    var grad_sum: f32 = 0.0;
    for (x0.grad) |g| grad_sum += @abs(g);
    try std.testing.expect(grad_sum > 1e-4);

    // 2. LSTM sequence container test
    var lstm = try LSTM.init(allocator, 4, 3);
    try testing_init.initRecurrent(&lstm, allocator, random);
    defer nn.deinitModel(&lstm, allocator);

    var graph_seq = autodiff.Graph.initDefault(allocator);
    defer graph_seq.deinit();

    const x_seq_0 = try graph_seq.tensorND(&.{ 2, 4 }, true);
    const x_seq_1 = try graph_seq.tensorND(&.{ 2, 4 }, true);
    @memset(x_seq_0.data, 0.2);
    @memset(x_seq_1.data, -0.2);

    const inputs = [_]*Tensor{ x_seq_0, x_seq_1 };
    const res = try lstm.forward(&graph_seq, &inputs, null, null);

    try std.testing.expectEqual(2, res.outputs.len);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, res.h_n.shape.dims[0..2]);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, res.c_n.shape.dims[0..2]);

    @memset(res.h_n.grad, 1.0);
    try graph_seq.backward(res.h_n);

    var x_seq_grad_sum: f32 = 0.0;
    for (x_seq_0.grad) |g| x_seq_grad_sum += @abs(g);
    for (x_seq_1.grad) |g| x_seq_grad_sum += @abs(g);
    try std.testing.expect(x_seq_grad_sum > 1e-4);
}

test "StackedLSTM forward and backward autograd" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(789);
    const random = prng.random();

    var stacked = try StackedLSTM.init(allocator, 4, 3, 2);
    try testing_init.initRecurrent(&stacked, allocator, random);
    defer nn.deinitModel(&stacked, allocator);

    try std.testing.expectEqual(2, stacked.num_layers);

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const x0 = try graph.tensorND(&.{ 2, 4 }, true);
    const x1 = try graph.tensorND(&.{ 2, 4 }, true);
    @memset(x0.data, 0.3);
    @memset(x1.data, -0.3);

    const inputs = [_]*Tensor{ x0, x1 };
    const res = try stacked.forwardSequence(&graph, &inputs, null, null);

    try std.testing.expectEqual(2, res.outputs.len);
    try std.testing.expectEqual(2, res.h_n.len);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, res.outputs[1].shape.dims[0..2]);

    @memset(res.outputs[1].grad, 1.0);
    try graph.backward(res.outputs[1]);

    var x_grad_sum: f32 = 0.0;
    for (x0.grad) |g| x_grad_sum += @abs(g);
    for (x1.grad) |g| x_grad_sum += @abs(g);
    try std.testing.expect(x_grad_sum > 1e-4);
}

test "GRUCell and GRU forward and backward autograd" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(999);
    const random = prng.random();

    // 1. GRUCell test
    var cell = try GRUCell.init(allocator, 4, 3);
    try testing_init.initRecurrent(&cell, allocator, random);
    defer nn.deinitModel(&cell, allocator);

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const x0 = try graph.tensorND(&.{ 2, 4 }, true);
    for (x0.data, 0..) |*p, i| p.* = @as(f32, @floatFromInt(i)) * 0.1;
    const h0 = try graph.zeros(&.{ 2, 3 }, true);

    const h1 = try cell.forward(&graph, x0, h0);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, h1.shape.dims[0..2]);

    @memset(h1.grad, 1.0);
    try graph.backward(h1);

    var grad_sum: f32 = 0.0;
    for (x0.grad) |g| grad_sum += @abs(g);
    try std.testing.expect(grad_sum > 1e-4);

    // 2. GRU sequence container test
    var gru = try GRU.init(allocator, 4, 3);
    try testing_init.initRecurrent(&gru, allocator, random);
    defer nn.deinitModel(&gru, allocator);

    var graph_seq = autodiff.Graph.initDefault(allocator);
    defer graph_seq.deinit();

    const x_seq_0 = try graph_seq.tensorND(&.{ 2, 4 }, true);
    const x_seq_1 = try graph_seq.tensorND(&.{ 2, 4 }, true);
    @memset(x_seq_0.data, 0.4);
    @memset(x_seq_1.data, -0.4);

    const inputs = [_]*Tensor{ x_seq_0, x_seq_1 };
    const res = try gru.forward(&graph_seq, &inputs, null);

    try std.testing.expectEqual(2, res.outputs.len);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, res.h_n.shape.dims[0..2]);

    @memset(res.h_n.grad, 1.0);
    try graph_seq.backward(res.h_n);

    var x_seq_grad_sum: f32 = 0.0;
    for (x_seq_0.grad) |g| x_seq_grad_sum += @abs(g);
    for (x_seq_1.grad) |g| x_seq_grad_sum += @abs(g);
    try std.testing.expect(x_seq_grad_sum > 1e-4);
}

test "sampleTopP and sampleTopK edge cases and deterministic argmax" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Logits: highest at index 3 (value 10.0)
    const logits = [_]f32{ 0.1, 1.0, 2.0, 10.0, 0.5 };

    // 1. sampleTopK with k=1 must be strictly deterministic and return index 3
    for (0..5) |_| {
        const picked_k1 = try sampleTopK(&logits, 5, 0.1, 1, random, allocator);
        try std.testing.expectEqual(@as(u32, 3), picked_k1);
    }

    // 2. sampleTopK with k >= vocab_size
    const picked_kall = try sampleTopK(&logits, 5, 1.0, 10, random, allocator);
    try std.testing.expect(picked_kall < 5);

    // 3. sampleTopP with low temperature and top_p=0.1
    const picked_p_low = try sampleTopP(&logits, 5, 0.01, 0.1, random, allocator);
    try std.testing.expectEqual(@as(u32, 3), picked_p_low);
}

test "computeGroupAdvantages zero-variance and grpoLoss clipping" {
    const allocator = std.testing.allocator;

    // 1. All rewards equal: mean = 2.0, std = 0.0 -> advantages = 0.0 without NaN
    const uniform_rewards = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    const adv = try computeGroupAdvantages(allocator, &uniform_rewards, 4, 1e-6);
    defer allocator.free(adv);

    for (adv) |a| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), a, 1e-4);
    }

    // 2. dpoLoss symmetry: when chosen and rejected logps match exactly, loss = log(2)
    const pi_c = [_]f32{-1.0};
    const pi_r = [_]f32{-1.0};
    const ref_c = [_]f32{-1.0};
    const ref_r = [_]f32{-1.0};
    const sym_dpo = dpoLoss(&pi_c, &pi_r, &ref_c, &ref_r, 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, @log(2.0)), sym_dpo, 1e-5);

    // 3. computeGRPOLoss with clipping
    const old_logps = [_]f32{-1.0};
    const new_logps = [_]f32{-0.5}; // ratio = exp(0.5) ~ 1.6487 > 1 + 0.2 (clip_eps = 0.2)
    const advantages = [_]f32{1.0};
    const grpo_val = computeGRPOLoss(&old_logps, &new_logps, &advantages, null, 0.0, 0.2);
    // Surrogate clamped to (1 + 0.2) * 1.0 = 1.2 -> loss = -1.2
    try std.testing.expectApproxEqAbs(@as(f32, -1.2), grpo_val, 1e-4);
}

test "RMSNorm and LayerNorm zero variance and uniform numerical stability" {
    const allocator = std.testing.allocator;

    // 1. RMSNorm with all zeros: denominator = sqrt(eps), output = 0.0 (no NaN)
    var rms = try RMSNorm.init(allocator, 4, 1e-5);
    defer nn.deinitModel(&rms, allocator);

    const x_zeros = try tensor.zeros(allocator, &.{ 1, 4 });
    defer tensor.free(allocator, x_zeros);

    var rms_out_graph = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer rms_out_graph.deinit();
    const rms_out = try rms.forward(&rms_out_graph, x_zeros);

    for (rms_out.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }

    // 2. LayerNorm with uniform row (all 3.0): mean=3.0, var=0.0 -> output is 0.0 (no NaN)
    var ln = try LayerNorm.init(allocator, 4, 1e-5);
    defer nn.deinitModel(&ln, allocator);

    const x_uniform = try tensor.zeros(allocator, &.{ 1, 4 });
    defer tensor.free(allocator, x_uniform);
    @memset(x_uniform.data, 3.0);

    var ln_out_graph = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer ln_out_graph.deinit();
    const ln_out = try ln.forward(&ln_out_graph, x_uniform);

    for (ln_out.data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-4);
    }

    // 3. Dropout with p=0.0 (returns x directly) and high p
    var prng = std.Random.DefaultPrng.init(11);
    const drop0 = Dropout.init(0.0);
    const drop_heavy = Dropout.init(0.9999);

    const x_test = try tensor.zeros(allocator, &.{ 1, 4 });
    defer tensor.free(allocator, x_test);
    @memset(x_test.data, 2.5);

    var y_drop0_graph = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer y_drop0_graph.deinit();
    const y_drop0 = try drop0.forward(&y_drop0_graph, x_test, prng.random());
    try std.testing.expectEqual(x_test, y_drop0);
    for (y_drop0.data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 2.5), v, 1e-5);
    }

    var y_heavy_graph = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer y_heavy_graph.deinit();
    const y_heavy = try drop_heavy.forward(&y_heavy_graph, x_test, prng.random());
    for (y_heavy.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "Comptime reflection supports slice modules, optional bias, and frozen LoRA base weights" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // 1. MoELayer ([]MLP fields: routed_experts and shared_experts)
    var moe = try MoELayer.init(allocator, 4, 8, 3, 1, 2);
    try testing_init.initFromOnes(&moe, allocator, random, &.{ 2, 4 });
    defer deinitModel(&moe, allocator);
    const moe_params = try parameters(&moe, allocator);
    defer allocator.free(moe_params);
    // gate (w + b = 2) + 3 routed experts (each c_fc.w, c_fc.b, c_proj.w, c_proj.b = 4) + 1 shared expert (4) = 18
    try std.testing.expectEqual(@as(usize, 18), moe_params.len);
    for (moe_params) |p| p.grad[0] = 1.0;
    zeroGradModel(&moe);
    for (moe_params) |p| try std.testing.expectApproxEqAbs(@as(f32, 0.0), p.grad[0], 1e-6);

    // 2. StackedLSTM ([]LSTMCell field: layers)
    var stacked_lstm = try StackedLSTM.init(allocator, 4, 6, 2);
    try testing_init.initRecurrent(&stacked_lstm, allocator, random);
    defer deinitModel(&stacked_lstm, allocator);
    const lstm_params = try parameters(&stacked_lstm, allocator);
    defer allocator.free(lstm_params);
    // 2 layers * 4 gates * 4 parameters (w_ih/w_hh Linear weights + bias) = 32
    try std.testing.expectEqual(@as(usize, 32), lstm_params.len);

    // 3. ConvTranspose2D (?*Tensor field: bias)
    var deconv = try ConvTranspose2D.init(allocator, 2, 3, 2, .{});
    deconv.resetParameters(random, .{});
    defer deinitModel(&deconv, allocator);
    const deconv_params = try parameters(&deconv, allocator);
    defer allocator.free(deconv_params);
    try std.testing.expectEqual(@as(usize, 2), deconv_params.len);

    // 4. LoRALinear (frozen weight, trainable lora_a, lora_b, and optional bias)
    var lora = try LoRALinear.initWithBias(allocator, 4, 3, 2, 4.0, true);
    lora.resetParameters(random, .{});
    defer deinitModel(&lora, allocator);
    const lora_params = try parameters(&lora, allocator);
    defer allocator.free(lora_params);
    // weight is frozen (requires_grad == false), only lora_a, lora_b, bias collected = 3
    try std.testing.expect(!lora.weight.requires_grad);
    try std.testing.expectEqual(@as(usize, 3), lora_params.len);
}

test "Safetensors serialization supports slice modules, optional bias, and out-of-order offsets" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(101);
    const random = prng.random();

    // 1. Round-trip saveModel / loadModel with MoELayer ([]MLP) and ConvTranspose2D (?*Tensor)
    const CompositeModel = struct {
        moe: MoELayer,
        deconv: ConvTranspose2D,
        lora: LoRALinear,
    };

    var m1 = CompositeModel{
        .moe = try MoELayer.init(allocator, 4, 8, 2, 1, 1),
        .deconv = try ConvTranspose2D.init(allocator, 2, 2, 2, .{}),
        .lora = try LoRALinear.initWithBias(allocator, 4, 3, 2, 2.0, true),
    };
    defer deinitModel(&m1, allocator);
    try testing_init.initFromOnes(&m1.moe, allocator, random, &.{ 2, 4 });
    m1.deconv.resetParameters(random, .{});
    m1.lora.resetParameters(random, .{});
    m1.deconv.bias.?.data[0] = 7.25;
    m1.lora.bias.?.data[1] = -3.5;
    m1.moe.routed_experts[1].c_fc.weight.data[0] = 42.0;

    const path = "test_composite_model.safetensors";
    try saveModel(&m1, std.testing.io, path, allocator);
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    var m2 = CompositeModel{
        .moe = try MoELayer.init(allocator, 4, 8, 2, 1, 1),
        .deconv = try ConvTranspose2D.init(allocator, 2, 2, 2, .{}),
        .lora = try LoRALinear.initWithBias(allocator, 4, 3, 2, 2.0, true),
    };
    defer deinitModel(&m2, allocator);
    m2.deconv.bias.?.data[0] = 0.0;
    m2.lora.bias.?.data[1] = 0.0;
    m2.moe.routed_experts[1].c_fc.weight.data[0] = 0.0;

    try loadModel(&m2, std.testing.io, path, allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 7.25), m2.deconv.bias.?.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -3.5), m2.lora.bias.?.data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), m2.moe.routed_experts[1].c_fc.weight.data[0], 1e-6);

    // 2. Out-of-order physical offsets (bias stored before weight in file) + __metadata__
    var lin = try Linear.init(allocator, 2, 2);
    lin.resetParameters(random, .{});
    defer nn.deinitModel(&lin, allocator);

    const header_json =
        \\{"__metadata":{"format":"pt"},"bias":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[8,24]}}
    ;
    const ooo_path = "test_ooo_linear.safetensors";
    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, ooo_path, .{});
        defer file.close(std.testing.io);
        var buf: [1024]u8 = undefined;
        var fw = file.writer(std.testing.io, &buf);
        const w = &fw.interface;
        const hlen: u64 = header_json.len;
        try w.writeAll(std.mem.asBytes(&hlen));
        try w.writeAll(header_json);
        // Physical payload: bias (2 floats = 8 bytes) FIRST, then weight (4 floats = 16 bytes)
        const payload_floats = [_]f32{ 10.0, 20.0, 1.0, 2.0, 3.0, 4.0 };
        try w.writeAll(std.mem.sliceAsBytes(&payload_floats));
        try w.flush();
    }
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, ooo_path) catch {};

    try loadModel(&lin, std.testing.io, ooo_path, allocator);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 10.0, 20.0 }, lin.bias.data);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, lin.weight.data);
}

test "Recurrent and Transformer modules naming and Graph scope registration" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(1234);
    const random = prng.random();

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    var rnn = try RNN.init(allocator, 4, 4);
    try testing_init.initRecurrent(&rnn, allocator, random);
    defer nn.deinitModel(&rnn, allocator);
    var names = std.heap.ArenaAllocator.init(allocator);
    defer names.deinit();
    try nn.nameModules(&rnn, names.allocator(), "enc_rnn");
    try std.testing.expectEqualStrings("enc_rnn", rnn.name.?);
    try std.testing.expectEqualStrings("enc_rnn.cell", rnn.cell.name.?);

    var lstm = try LSTM.init(allocator, 4, 4);
    try testing_init.initRecurrent(&lstm, allocator, random);
    defer nn.deinitModel(&lstm, allocator);
    try nn.nameModules(&lstm, names.allocator(), "enc_lstm");
    try std.testing.expectEqualStrings("enc_lstm", lstm.name.?);

    var slstm = try StackedLSTM.init(allocator, 4, 4, 2);
    try testing_init.initRecurrent(&slstm, allocator, random);
    defer nn.deinitModel(&slstm, allocator);
    try nn.nameModules(&slstm, names.allocator(), "deep_lstm");
    try std.testing.expectEqualStrings("deep_lstm.layers.0", slstm.layers[0].name.?);

    var gru = try GRU.init(allocator, 4, 4);
    try testing_init.initRecurrent(&gru, allocator, random);
    defer nn.deinitModel(&gru, allocator);
    try nn.nameModules(&gru, names.allocator(), "enc_gru");
    try std.testing.expectEqualStrings("enc_gru", gru.name.?);

    var moe = try MoELayer.init(allocator, 4, 8, 2, 1, 1);
    try testing_init.initFromOnes(&moe, allocator, random, &.{ 2, 4 });
    defer nn.deinitModel(&moe, allocator);
    try nn.nameModules(&moe, names.allocator(), "ffn_moe");
    try std.testing.expectEqualStrings("ffn_moe.gate", moe.gate.name.?);
    try std.testing.expectEqualStrings("ffn_moe.routed_experts.0", moe.routed_experts[0].name.?);

    var mla = try MLALayer.init(allocator, 8, 2, 4, 4, 2);
    try testing_init.initFromOnes(&mla, allocator, random, &.{ 1, 2, 8 });
    defer nn.deinitModel(&mla, allocator);
    try nn.nameModules(&mla, names.allocator(), "attn_mla");
    try std.testing.expectEqualStrings("attn_mla.q_proj", mla.q_proj.name.?);

    var lora = try LoRALinear.initWithBias(allocator, 4, 4, 2, 4.0, true);
    lora.resetParameters(random, .{});
    defer nn.deinitModel(&lora, allocator);
    try nn.nameModules(&lora, names.allocator(), "proj_lora");
    try std.testing.expectEqualStrings("proj_lora.lora_a", lora.lora_a.getName().?);

    const x = try graph.tensor(2, 4, true);
    @memset(x.data, 0.2);
    const out_lora = try lora.forward(&graph, x);
    try std.testing.expectEqualStrings("proj_lora", autodiff.Op.creatorOf(out_lora).?.scope);
    try std.testing.expectEqualStrings(LoRALinear.formula, graph.inferModuleFormula("proj_lora"));
}

test "Conv2D with stride and padding forward and backward (im2col + sgemm)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(2026);
    const random = prng.random();

    var conv = try Conv2D.init(allocator, 1, 2, 3, .{ .stride = 2, .padding = 1 });
    conv.resetParameters(random, .{});
    defer nn.deinitModel(&conv, allocator);
    @memset(conv.weight.data[0..9], 1.0);
    @memset(conv.weight.data[9..18], 2.0);
    conv.bias.data[0] = 0.5;
    conv.bias.data[1] = -0.5;

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    // Input [1, 1, 4, 4] with kernel=3, stride=2, padding=1 -> Output [1, 2, 2, 2]
    const x = try graph.tensor(1, 16, true);
    x.shape = Shape.init(&.{ 1, 1, 4, 4 });
    x.strides = tensor.computeContiguousStrides(x.shape);
    for (x.data, 0..) |*v, idx| {
        v.* = @as(f32, @floatFromInt(idx + 1));
    }

    const out = try conv.forward(&graph, x);
    try std.testing.expectEqual(@as(usize, 4), out.shape.len);
    try std.testing.expectEqual(@as(usize, 1), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), out.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 2), out.shape.dims[2]);
    try std.testing.expectEqual(@as(usize, 2), out.shape.dims[3]);

    // Top-left window (h_out=0, w_out=0) with padding=1 samples x[0..2, 0..2] = {1, 2, 5, 6}, sum = 14
    try std.testing.expectApproxEqAbs(@as(f32, 14.5), out.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 27.5), out.data[4], 1e-4);

    const loss = try graph.sum(out, null, false);
    try graph.backward(loss);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), conv.bias.grad[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), conv.bias.grad[1], 1e-4);
    // x[0, 0] is covered only by window (0, 0) at (kh=1, kw=1), so dL/dx[0,0] = w[0,0,1,1] + w[1,0,1,1] = 1 + 2 = 3
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), x.grad[0], 1e-4);
}

test "Conv1D with stride and padding forward and backward (im2col + sgemm)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(2026);
    const random = prng.random();

    var conv = try Conv1D.init(allocator, 1, 2, 3, .{ .stride = 2, .padding = 1 });
    conv.resetParameters(random, .{});
    defer nn.deinitModel(&conv, allocator);
    @memset(conv.weight.data[0..3], 1.0);
    @memset(conv.weight.data[3..6], 2.0);
    conv.bias.data[0] = 0.5;
    conv.bias.data[1] = -0.5;

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    // Input [1, 1, 4] = {1, 2, 3, 4} with kernel=3, stride=2, padding=1 -> Output [1, 2, 2]
    // Window 0 (l_out=0): [0, 1, 2] -> sum = 3
    // Window 1 (l_out=1): [2, 3, 4] -> sum = 9
    const x = try graph.tensorNDWithData(&.{ 1, 1, 4 }, &.{ 1.0, 2.0, 3.0, 4.0 }, true);

    const out = try conv.forward(&graph, x);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 2 }, out.shape.dims[0..out.shape.len]);

    try std.testing.expectApproxEqAbs(@as(f32, 3.5), out.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 9.5), out.data[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), out.data[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 17.5), out.data[3], 1e-4);

    const loss = try graph.sum(out, null, false);
    try graph.backward(loss);

    try std.testing.expectApproxEqAbs(@as(f32, 2.0), conv.bias.grad[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), conv.bias.grad[1], 1e-4);
    // x[0] is covered only by window 0 at k=1 -> dL/dx[0] = w[0,0,1] + w[1,0,1] = 1 + 2 = 3
    // x[1] is covered by window 0 (k=2) and window 1 (k=0) -> dL/dx[1] = 2 * (1 + 2) = 6
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), x.grad[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), x.grad[1], 1e-4);
}

test "ConvTranspose1D with stride and padding forward and backward" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(2027);
    const random = prng.random();

    var deconv = try ConvTranspose1D.init(allocator, 1, 2, 3, .{ .stride = 2, .padding = 1, .use_bias = true });
    deconv.resetParameters(random, .{});
    defer nn.deinitModel(&deconv, allocator);
    // weight shape: [in_c=1, out_c=2, K=3]
    @memset(deconv.weight.data[0..3], 1.0);
    @memset(deconv.weight.data[3..6], 2.0);
    deconv.bias.?.data[0] = 0.5;
    deconv.bias.?.data[1] = -0.5;

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    // Input [1, 1, 2] = {1, 2}, stride=2, padding=1, K=3 -> L_out = (2-1)*2 + 3 - 2 = 3
    // l_in=0 (val=1) contributes to out_l_raw in {0,1,2} -> out_l in {0,1} (k=1,2)
    // l_in=1 (val=2) contributes to out_l_raw in {2,3,4} -> out_l in {1,2} (k=0,1)
    const x = try graph.tensorNDWithData(&.{ 1, 1, 2 }, &.{ 1.0, 2.0 }, true);
    const out = try deconv.forward(&graph, x);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, out.shape.dims[0..out.shape.len]);

    // Channel 0 (w=1, b=0.5): out_l=0 -> 1*1 + 0.5 = 1.5; out_l=1 -> 1*1 + 2*1 + 0.5 = 3.5; out_l=2 -> 2*1 + 0.5 = 2.5
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), out.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), out.data[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), out.data[2], 1e-4);
    // Channel 1 (w=2, b=-0.5): out_l=0 -> 2 - 0.5 = 1.5; out_l=1 -> 2 + 4 - 0.5 = 5.5; out_l=2 -> 4 - 0.5 = 3.5
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), out.data[3], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), out.data[4], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), out.data[5], 1e-4);

    const loss = try graph.sum(out, null, false);
    try graph.backward(loss);

    try std.testing.expectApproxEqAbs(@as(f32, 3.0), deconv.bias.?.grad[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), deconv.bias.?.grad[1], 1e-4);
    // Both l_in=0 and l_in=1 land on 2 valid output positions per channel -> dL/dx[i] = 2*(1 + 2) = 6
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), x.grad[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), x.grad[1], 1e-4);
}

test "MoELayer sparse top-k expert execution skips inactive experts" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(77);
    const random = prng.random();

    // 3 routed experts, 0 shared experts, top_k = 1
    var moe = try MoELayer.init(allocator, 4, 8, 3, 0, 1);
    try testing_init.initFromOnes(&moe, allocator, random, &.{ 2, 4 });
    defer nn.deinitModel(&moe, allocator);

    // Verify all 3 routed experts were initialized by initModel during pre-init graph discovery
    for (moe.routed_experts) |exp| {
        var w_norm: f32 = 0.0;
        for (exp.c_fc.weight.data) |v| w_norm += @abs(v);
        try std.testing.expect(w_norm > 0.0);
    }

    // Force gate weights so expert 0 wins for positive inputs while keeping finite logits
    @memset(moe.gate.weight.data, 0.0);
    @memset(moe.gate.bias.data, 0.0);
    moe.gate.bias.data[0] = 2.0;
    moe.gate.bias.data[1] = -1.0;
    moe.gate.bias.data[2] = -1.0;

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const x = try graph.tensor(2, 4, true);
    @memset(x.data, 1.0);

    // Graph mode forward + backward
    const out_g = try moe.forward(&graph, x);

    // Verify branchless indexed execution: 1 TopK, 2 GatherMatMul (c_fc & c_proj), and 1 Gelu op in the graph
    var topk_count: usize = 0;
    var gather_mm_count: usize = 0;
    var gelu_count: usize = 0;
    for (graph.ops.items) |op| {
        if (op.op_type == .TopK) topk_count += 1;
        if (op.op_type == .GatherMatMul) gather_mm_count += 1;
        if (op.op_type == .Gelu) gelu_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), topk_count);
    try std.testing.expectEqual(@as(usize, 2), gather_mm_count);
    try std.testing.expectEqual(@as(usize, 1), gelu_count);

    const loss = try graph.sum(out_g, null, false);
    try graph.backward(loss);

    // Expert 0 was selected -> non-zero weight gradients; Experts 1 & 2 were skipped -> strictly zero gradients
    var exp0_grad_norm: f32 = 0.0;
    for (moe.routed_experts[0].c_proj.weight.grad) |g| exp0_grad_norm += @abs(g);
    try std.testing.expect(exp0_grad_norm > 0.0);

    for (moe.routed_experts[1].c_proj.weight.grad) |g| {
        try std.testing.expectEqual(@as(f32, 0.0), g);
    }
    for (moe.routed_experts[2].c_proj.weight.grad) |g| {
        try std.testing.expectEqual(@as(f32, 0.0), g);
    }

    // Even with top_k == 1, router gate must receive non-zero gradients (post-softmax Top-K gating)
    var gate_w_grad_norm: f32 = 0.0;
    for (moe.gate.weight.grad) |g| gate_w_grad_norm += @abs(g);
    try std.testing.expect(gate_w_grad_norm > 0.0);

    var gate_b_grad_norm: f32 = 0.0;
    for (moe.gate.bias.grad) |g| gate_b_grad_norm += @abs(g);
    try std.testing.expect(gate_b_grad_norm > 0.0);

    // Inference mode forward should match Graph mode output
    var out_e_graph = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer out_e_graph.deinit();
    const out_e = try moe.forward(&out_e_graph, x);
    for (out_g.data, out_e.data) |vg, ve| {
        try std.testing.expectApproxEqAbs(vg, ve, 1e-4);
    }

    // Static graph replay (`graph.forward()`) when router preference switches from Expert 0 to Expert 2:
    // Because TopK and GatherMatMul have zero host-side `if` graph pruning, replaying the static graph
    // dynamically routes to Expert 2 without rebuilding the graph.
    moe.gate.bias.data[0] = -1.0;
    moe.gate.bias.data[1] = -1.0;
    moe.gate.bias.data[2] = 3.0;
    graph.zeroGrad();
    nn.zeroGradModel(&moe);
    try graph.forward();
    try graph.backward(loss);

    for (moe.routed_experts[0].c_proj.weight.grad) |g| {
        try std.testing.expectEqual(@as(f32, 0.0), g);
    }
    for (moe.routed_experts[1].c_proj.weight.grad) |g| {
        try std.testing.expectEqual(@as(f32, 0.0), g);
    }
    var exp2_grad_norm: f32 = 0.0;
    for (moe.routed_experts[2].c_proj.weight.grad) |g| exp2_grad_norm += @abs(g);
    try std.testing.expect(exp2_grad_norm > 0.0);

    // Verify norm_topk_prob = true with top_k = 2 renormalizes selected expert weights and propagates gradients
    var moe_norm = try MoELayer.init(allocator, 4, 8, 3, 0, 2);
    moe_norm.norm_topk_prob = true;
    try testing_init.initFromOnes(&moe_norm, allocator, random, &.{ 2, 4 });
    defer nn.deinitModel(&moe_norm, allocator);

    var g_norm = autodiff.Graph.initDefault(allocator);
    defer g_norm.deinit();
    const x_norm = try g_norm.tensor(2, 4, true);
    @memset(x_norm.data, 0.5);
    const out_norm = try moe_norm.forward(&g_norm, x_norm);
    const loss_norm = try g_norm.sum(out_norm, null, false);
    try g_norm.backward(loss_norm);
    var norm_gate_grad: f32 = 0.0;
    for (moe_norm.gate.weight.grad) |g| norm_gate_grad += @abs(g);
    try std.testing.expect(norm_gate_grad > 0.0);

    // Verify shape validation errors
    const bad_rank = try g_norm.tensorND(&.{ 1, 2, 2, 4 }, false);
    try std.testing.expectError(error.IncompatibleDimensions, moe.forward(&g_norm, bad_rank));
    const bad_dim = try g_norm.tensor(2, 5, false);
    try std.testing.expectError(error.ShapeMismatch, moe.forward(&g_norm, bad_dim));
}

test "Recurrent zero initial states are named module buffers" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(2468);
    const random = prng.random();

    const expectBuffer = struct {
        fn check(g: *const autodiff.Graph, name: []const u8, scope: []const u8) !void {
            for (g.tensors.items) |t| {
                if (t.name) |n| if (std.mem.eql(u8, n, name)) {
                    try std.testing.expect(t.is_buffer);
                    try std.testing.expect(!t.requires_grad);
                    try std.testing.expectEqualStrings(scope, t.scope);
                    return;
                };
            }
            return error.TestExpectedBufferNotFound;
        }
    };

    {
        var m = try LSTM.init(allocator, 4, 3);
        try testing_init.initRecurrent(&m, allocator, random);
        defer nn.deinitModel(&m, allocator);
        var names = std.heap.ArenaAllocator.init(allocator);
        defer names.deinit();
        try nn.nameModules(&m, names.allocator(), "lstm");
        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();
        var inps = [_]*Tensor{ try g.ones(&.{ 2, 4 }, false), try g.ones(&.{ 2, 4 }, false) };
        _ = try m.forward(&g, &inps, null, null);
        try expectBuffer.check(&g, "lstm.h_0", "lstm");
        try expectBuffer.check(&g, "lstm.c_0", "lstm");
    }

    {
        var m = try StackedLSTM.init(allocator, 4, 3, 2);
        try testing_init.initRecurrent(&m, allocator, random);
        defer nn.deinitModel(&m, allocator);
        var names = std.heap.ArenaAllocator.init(allocator);
        defer names.deinit();
        try nn.nameModules(&m, names.allocator(), "stacked_lstm");
        var g = autodiff.Graph.initDefault(allocator);
        defer g.deinit();
        var inps = [_]*Tensor{ try g.ones(&.{ 2, 4 }, false), try g.ones(&.{ 2, 4 }, false) };
        _ = try m.forwardSequence(&g, &inps, null, null);
        for ([_][]const u8{ "stacked_lstm.h_0_0", "stacked_lstm.c_0_0", "stacked_lstm.h_0_1", "stacked_lstm.c_0_1" }) |name| {
            try expectBuffer.check(&g, name, "stacked_lstm");
        }
    }
}

test "ScaledDotProductAttention standalone causal and non-causal forward and backward" {
    const allocator = std.testing.allocator;
    var g = autodiff.Graph.initDefault(allocator);
    defer g.deinit();

    const q = try g.tensorNDWithData(&.{ 1, 1, 2, 2 }, &.{ 1.0, 0.0, 0.0, 1.0 }, true);
    const k = try g.tensorNDWithData(&.{ 1, 1, 2, 2 }, &.{ 1.0, 0.0, 0.0, 1.0 }, true);
    const v = try g.tensorNDWithData(&.{ 1, 1, 2, 2 }, &.{ 2.0, 4.0, 6.0, 8.0 }, true);

    var sdpa_causal = transformer.ScaledDotProductAttention.initDefault();
    var names = std.heap.ArenaAllocator.init(allocator);
    defer names.deinit();
    try nn.nameModules(&sdpa_causal, names.allocator(), "sdpa");
    const out_causal = try sdpa_causal.forward(&g, q, k, v);
    try std.testing.expectEqual(@as(usize, 4), out_causal.shape.len);
    // Position 0 only attends to position 0 (causal mask blocks position 1): output[0] == v[0] = [2.0, 4.0]
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out_causal.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out_causal.data[1], 1e-4);

    const sdpa_bidir = transformer.ScaledDotProductAttention.init(.{ .causal = false });
    const out_bidir = try sdpa_bidir.forward(&g, q, k, v);
    // Without causal mask, position 0 also attends to position 1 with non-zero weight, so output[0] > v[0]
    try std.testing.expect(out_bidir.data[0] > 2.5);
    try std.testing.expect(out_bidir.data[1] > 4.5);

    const loss = try g.sum(out_causal, null, false);
    try g.backward(loss);
    try std.testing.expect(v.grad[0] > 0.0);
}

test "Safetensors BF16 deserialization and loadModel" {
    const allocator = std.testing.allocator;
    const tmp_file = "test_bf16_roundtrip.safetensors";
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_file) catch {};

    // 手工写一个标准 Safetensors BF16 文件
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.createFile(std.testing.io, tmp_file, .{});
    defer file.close(std.testing.io);

    const header_json = "{\"weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]}}";
    const padding = (8 - (header_json.len % 8)) % 8;
    const total_header_len = header_json.len + padding;

    var buf: [1024]u8 = undefined;
    var file_writer = file.writer(std.testing.io, &buf);
    const writer = &file_writer.interface;

    const len_u64 = @as(u64, total_header_len);
    try writer.writeAll(std.mem.asBytes(&len_u64));
    try writer.writeAll(header_json);
    for (0..padding) |_| try writer.writeByte(' ');

    // 写入 4 个 BF16 数值: 1.0, 2.0, -1.5, 0.5
    const bf16_vals: [4]u16 = .{
        tensor.bf16.fromF32(1.0).bits,
        tensor.bf16.fromF32(2.0).bits,
        tensor.bf16.fromF32(-1.5).bits,
        tensor.bf16.fromF32(0.5).bits,
    };
    try writer.writeAll(std.mem.sliceAsBytes(&bf16_vals));
    try writer.flush();

    const DummyModule = struct {
        weight: *Tensor,
    };
    const w = try core.createPersistentTensor(allocator, 2, 2, false);
    defer core.freePersistentTensor(allocator, w);
    var dummy = DummyModule{ .weight = w };

    try serialization.loadModel(&dummy, std.testing.io, tmp_file, allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), w.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), w.data[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), w.data[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), w.data[3], 1e-4);
}

test "Linear forward supports 3D and 1D inputs with autograd" {
    const allocator = std.testing.allocator;
    var linear = try Linear.init(allocator, 3, 2);
    defer nn.deinitModel(&linear, allocator);

    // Initialize weights and biases
    @memcpy(linear.weight.data, &[_]f32{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    });
    @memcpy(linear.bias.data, &[_]f32{ 0.1, 0.2 });

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    // 1. 3D input: [B=2, T=2, C=3]
    const x_3d = try graph.tensorND(&.{ 2, 2, 3 }, true);
    for (x_3d.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1));
    }

    const y_3d = try linear.forward(&graph, x_3d);
    try std.testing.expectEqual(@as(usize, 3), y_3d.shape.len);
    try std.testing.expectEqual(@as(usize, 2), y_3d.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), y_3d.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 2), y_3d.shape.dims[2]);

    // Backward pass
    const loss = try graph.sum(y_3d, null, false);
    try graph.backward(loss);
    try std.testing.expect(x_3d.grad.len == 12);

    // 2. 1D input: [C=3]
    var g1 = autodiff.Graph.initDefault(allocator);
    defer g1.deinit();
    const x_1d = try g1.tensorND(&.{3}, false);
    @memcpy(x_1d.data, &[_]f32{ 1.0, 1.0, 1.0 });

    const y_1d = try linear.forward(&g1, x_1d);
    try std.testing.expectEqual(@as(usize, 1), y_1d.shape.len);
    try std.testing.expectEqual(@as(usize, 2), y_1d.shape.dims[0]);
    // y = [1, 1, 1] * W + [0.1, 0.2] = [1+3+5 + 0.1, 2+4+6 + 0.2] = [9.1, 12.2]
    try std.testing.expectApproxEqAbs(@as(f32, 9.1), y_1d.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12.2), y_1d.data[1], 1e-5);
}

test "sampleTopP samples from the nucleus of highest-probability tokens" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(1);
    const random = prng.random();

    // p ≈ [0.043, 0.043, 0.043, 0.87]：top_p = 0.05 时核只含词元 3
    const peaked = [_]f32{ 0, 0, 0, 3 };
    for (0..200) |_| {
        try std.testing.expectEqual(@as(u32, 3), try sampleTopP(&peaked, 4, 1.0, 0.05, random, allocator));
    }

    // 均匀分布 + top_p = 0.5：核为下标最小的两个词元，二者近似各占一半
    const uniform = [_]f32{ 0, 0, 0, 0 };
    var counts: [4]usize = @splat(0);
    for (0..4000) |_| counts[try sampleTopP(&uniform, 4, 1.0, 0.5, random, allocator)] += 1;
    try std.testing.expectEqual(@as(usize, 0), counts[2] + counts[3]);
    try std.testing.expect(counts[0] > 1600 and counts[1] > 1600);

    // top_p = 1.0：全词表采样，每个词元都会出现
    @memset(&counts, 0);
    for (0..4000) |_| counts[try sampleTopP(&uniform, 4, 1.0, 1.0, random, allocator)] += 1;
    for (counts) |c| try std.testing.expect(c > 700);

    // top_p = 0：贪心解码
    const logits = [_]f32{ 0.1, 1.0, 2.0, 10.0, 0.5 };
    try std.testing.expectEqual(@as(u32, 3), try sampleTopP(&logits, 5, 1.0, 0.0, random, allocator));
    try std.testing.expectError(error.InvalidArgument, sampleTopP(&logits, 0, 1.0, 0.9, random, allocator));
}

test "sampleTopK restricts sampling to the k most likely tokens and rejects k = 0" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const random = prng.random();

    const logits = [_]f32{ 0.1, 1.0, 2.0, 10.0, 0.5 };
    var counts: [5]usize = @splat(0);
    // 高温使前两名概率接近，确保两者都会被采到且其余词元从不出现
    for (0..2000) |_| counts[try sampleTopK(&logits, 5, 100.0, 2, random, allocator)] += 1;
    try std.testing.expectEqual(@as(usize, 0), counts[0] + counts[1] + counts[4]);
    try std.testing.expect(counts[2] > 500 and counts[3] > 500);

    try std.testing.expectError(error.InvalidArgument, sampleTopK(&logits, 5, 1.0, 0, random, allocator));
}

test "maskedCrossEntropyLoss eager gradient matches the graph version" {
    const allocator = std.testing.allocator;
    const values = [_]f32{
        2.0, 1.0, 0.1, 0.0,
        0.5, 3.0, 0.2, 0.1,
        0.1, 0.2, 4.0, 0.1,
    };
    const targets = [_]u32{ 0, 3, 2 };
    const mask = [_]f32{ 0.0, 0.5, 1.5 };

    const logits = try createPersistentTensor(allocator, 3, 4, true);
    defer freePersistentTensor(allocator, logits);
    @memcpy(logits.data, &values);
    const loss = try maskedCrossEntropyLoss(logits, &targets, &mask, allocator);

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();
    const g_logits = try graph.tensorNDWithData(&.{ 3, 4 }, &values, true);
    const g_loss = try maskedCrossEntropyLossGraph(&graph, g_logits, &targets, &mask);
    try graph.backward(g_loss);

    try std.testing.expectApproxEqAbs(g_loss.data[0], loss, 1e-5);
    for (g_logits.grad, logits.grad) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-5);
    }

    // 越界目标下标在写入任何梯度之前报错
    @memset(logits.grad, 0.0);
    const bad_targets = [_]u32{ 0, 1, 4 };
    try std.testing.expectError(error.IndexOutOfBounds, maskedCrossEntropyLoss(logits, &bad_targets, &mask, allocator));
    for (logits.grad) |g| try std.testing.expectEqual(@as(f32, 0.0), g);
}

test "LoRALinear fuse preserves the adapted forward output" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const random = prng.random();

    var lora = try LoRALinear.init(allocator, 5, 3, 2, 4.0);
    defer nn.deinitModel(&lora, allocator);
    for (lora.weight.data) |*v| v.* = random.floatNorm(f32);
    for (lora.lora_a.data) |*v| v.* = random.floatNorm(f32);
    for (lora.lora_b.data) |*v| v.* = random.floatNorm(f32);

    var graph = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer graph.deinit();
    const x = try graph.tensorND(&.{ 2, 5 }, false);
    for (x.data) |*v| v.* = random.floatNorm(f32);

    const before = try lora.forward(&graph, x);
    lora.fuse();
    for (lora.lora_b.data) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
    const after = try lora.forward(&graph, x);
    for (before.data, after.data) |b, a| try std.testing.expectApproxEqAbs(b, a, 1e-4);
}

test "computeGroupAdvantages rejects invalid group sizes" {
    const allocator = std.testing.allocator;
    const rewards = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectError(error.InvalidArgument, computeGroupAdvantages(allocator, &rewards, 0, 1e-6));
    try std.testing.expectError(error.InvalidArgument, computeGroupAdvantages(allocator, &rewards, 2, 1e-6));
}

/// 以随机参数 (含全部偏置) 对比多头潜在注意力 (Multi-Head Latent Attention, MLA) 增量推理与全序列前向的最大误差。
/// batch_2d = true 时以二维输入 [B, C] 批量执行单步推理
fn mlaIncrementalMaxDiff(batch_2d: bool) !f32 {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();
    const dim = 8;
    const T = 4;
    const B = 2;
    var mla = try MLALayer.init(allocator, dim, 2, 4, 4, 2);
    defer nn.deinitModel(&mla, allocator);
    const params = try nn.parameters(&mla, allocator);
    defer allocator.free(params);
    for (params) |p| {
        for (p.data) |*v| v.* = random.floatNorm(f32) * 0.3;
    }

    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();
    const x = try g.tensorND(&.{ B, T, dim }, false);
    for (x.data) |*v| v.* = random.floatNorm(f32);
    const y_full = try mla.forward(&g, x);

    var cache = try MLACache.init(allocator, B, T, 4, 2);
    defer cache.deinit(allocator);
    var max_diff: f32 = 0;
    const step = try allocator.alloc(f32, B * dim);
    defer allocator.free(step);
    for (0..T) |t| {
        for (0..B) |b| @memcpy(step[b * dim .. (b + 1) * dim], x.data[(b * T + t) * dim ..][0..dim]);
        const shape: []const usize = if (batch_2d) &.{ B, dim } else &.{ B, 1, dim };
        const xt = try tensor.array(allocator, shape, step);
        defer tensor.free(allocator, xt);
        const yt = try mla.forwardInference(allocator, xt, &cache);
        defer tensor.free(allocator, yt);
        try std.testing.expectEqual(B, yt.shape.dims[0]);
        for (0..B) |b| {
            for (0..dim) |c| {
                max_diff = @max(max_diff, @abs(yt.data[b * dim + c] - y_full.data[(b * T + t) * dim + c]));
            }
        }
    }
    return max_diff;
}

test "MLALayer forwardInference matches full forward with biases and batched inputs" {
    try std.testing.expect(try mlaIncrementalMaxDiff(false) < 1e-4);
    try std.testing.expect(try mlaIncrementalMaxDiff(true) < 1e-4);
}

test "CausalSelfAttention forwardInference matches full forward and reports a full cache" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const random = prng.random();
    const C = 12;
    const T = 3;
    const B = 2;
    var attn = try CausalSelfAttention.initGQA(allocator, C, 4, 2);
    defer nn.deinitModel(&attn, allocator);
    const params = try nn.parameters(&attn, allocator);
    defer allocator.free(params);
    for (params) |p| {
        for (p.data) |*v| v.* = random.floatNorm(f32) * 0.3;
    }

    var g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
    defer g.deinit();
    const x = try g.tensorND(&.{ B, T, C }, false);
    for (x.data) |*v| v.* = random.floatNorm(f32);
    const y_full = try attn.forward(&g, x);

    var cache = try KVCache.init(allocator, B, 2, T, C / 4);
    defer cache.deinit(allocator);
    var step: [B * C]f32 = undefined;
    for (0..T) |t| {
        for (0..B) |b| @memcpy(step[b * C .. (b + 1) * C], x.data[(b * T + t) * C ..][0..C]);
        const xt = try tensor.array(allocator, &.{ B, 1, C }, &step);
        defer tensor.free(allocator, xt);
        const yt = try attn.forwardInference(allocator, xt, &cache);
        defer tensor.free(allocator, yt);
        for (0..B) |b| {
            for (0..C) |c| {
                try std.testing.expectApproxEqAbs(y_full.data[(b * T + t) * C + c], yt.data[b * C + c], 1e-4);
            }
        }
    }

    const xt = try tensor.array(allocator, &.{ B, 1, C }, &step);
    defer tensor.free(allocator, xt);
    try std.testing.expectError(error.ContextWindowExceeded, attn.forwardInference(allocator, xt, &cache));

    var mla = try MLALayer.init(allocator, 8, 2, 4, 4, 2);
    defer nn.deinitModel(&mla, allocator);
    var mla_cache = try MLACache.init(allocator, 1, 1, 4, 2);
    defer mla_cache.deinit(allocator);
    mla_cache.curr_len = 1;
    const x_mla = try tensor.zeros(allocator, &.{ 1, 8 });
    defer tensor.free(allocator, x_mla);
    try std.testing.expectError(error.ContextWindowExceeded, mla.forwardInference(allocator, x_mla, &mla_cache));
}

test "KVCache and MLACache init release partial allocations on failure" {
    // 每个无梯度缓存张量分配两次 (张量结构体与数据)，逐一让第 1..3 次分配失败
    for (1..4) |fail_index| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        try std.testing.expectError(error.OutOfMemory, KVCache.init(failing.allocator(), 1, 2, 4, 4));
        var failing_mla = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        try std.testing.expectError(error.OutOfMemory, MLACache.init(failing_mla.allocator(), 1, 4, 4, 2));
    }
}
