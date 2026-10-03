const std = @import("std");
const testing_init = @import("testing_init.zig");
const autodiff = @import("../autodiff.zig");
const tensor = @import("../tensor.zig");
const engine = @import("../engine.zig");
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
const Conv2D = nn.Conv2D;
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

const trainClassificationStep = nn.trainClassificationStep;
const evalClassificationStep = nn.evalClassificationStep;
const trainClassificationEpoch = nn.trainClassificationEpoch;
const evaluateClassification = nn.evaluateClassification;
const computeAccuracy = nn.computeAccuracy;
const ClassificationStepResult = nn.ClassificationStepResult;
const ClassificationEpochResult = nn.ClassificationEpochResult;
const trainStep = nn.trainStep;
const evalStep = nn.evalStep;
const trainEpoch = nn.trainEpoch;
const evaluate = nn.evaluate;
const StepResult = nn.StepResult;
const EpochResult = nn.EpochResult;

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

    var y_eager_graph = autodiff.Graph.initNoGrad(arena);
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

    var graph = autodiff.Graph.init(arena);
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

    var y_eager_graph = autodiff.Graph.initNoGrad(arena);
    defer y_eager_graph.deinit();
    const y_eager = try norm.forward(&y_eager_graph, x);
    try std.testing.expectEqualSlices(usize, &.{2, 4}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.init(arena);
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

    var y_eager_graph = autodiff.Graph.initNoGrad(arena);
    defer y_eager_graph.deinit();
    const y_eager = try mlp.forward(&y_eager_graph, x_3d);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 4}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.init(arena);
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

    var y_eager_graph = autodiff.Graph.initNoGrad(arena);
    defer y_eager_graph.deinit();
    const y_eager = try attn.forward(&y_eager_graph, x_3d);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 8}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.init(arena);
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

    var y_eager_graph = autodiff.Graph.initNoGrad(arena);
    defer y_eager_graph.deinit();
    const y_eager = try gpt.forward(&y_eager_graph, x);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 10}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.init(arena);
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

    var graph = autodiff.Graph.init(allocator);
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

    var eager_out_graph = autodiff.Graph.initNoGrad(allocator);
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

    var graph = autodiff.Graph.init(allocator);
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

    var graph = autodiff.Graph.init(allocator);
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
    var graph = autodiff.Graph.init(allocator);
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

    var y_graph = autodiff.Graph.initNoGrad(allocator);
    defer y_graph.deinit();
    const y = try ln.forward(&y_graph, x);

    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, y.shape.dims[0..2]);
    var sum: f32 = 0.0;
    for (y.data[0..4]) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sum / 4.0, 1e-4);

    // Graph mode forward + backward
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const gx = try graph.tensorNDWithData(&.{ 2, 4 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 1.0, 3.0 }, true);
    const gy = try ln.forward(&graph, gx);
    try std.testing.expect(gy.creator != null);
    try std.testing.expectEqual(autodiff.OpType.LayerNorm, gy.creator.?.op_type);

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

    var y_graph = autodiff.Graph.initNoGrad(allocator);
    defer y_graph.deinit();
    const y = try bn.forward(&y_graph, x);

    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 2 }, y.shape.dims[0..4]);

    // Graph mode forward + backward
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const gx = try graph.tensorND(&.{ 2, 2, 2, 2 }, true);
    for (gx.data, 0..) |*p, i| {
        const fi = @as(f32, @floatFromInt(i));
        p.* = @sin(fi * 1.3) + 0.2 * fi;
    }
    const gy = try bn.forward(&graph, gx);
    try std.testing.expect(gy.creator != null);
    try std.testing.expectEqual(autodiff.OpType.BatchNorm2d, gy.creator.?.op_type);

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

test "Dropout and AvgPool2D forward and backward passes" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    const drop = Dropout.init(0.2);
    const x = try tensor.zeros(allocator, &.{ 1, 10 });
    defer tensor.free(allocator, x);
    @memset(x.data, 1.0);

    var y_drop_graph = autodiff.Graph.initNoGrad(allocator);
    defer y_drop_graph.deinit();
    const y_drop = try drop.forward(&y_drop_graph, x, rand);
    try std.testing.expectEqual(10, y_drop.data.len);

    const pool = AvgPool2D.init(2, 2);
    const img = try tensor.zeros(allocator, &.{ 1, 1, 4, 4 });
    defer tensor.free(allocator, img);
    @memset(img.data, 4.0);

    var pooled_graph = autodiff.Graph.initNoGrad(allocator);
    defer pooled_graph.deinit();
    const pooled = try pool.forward(&pooled_graph, img);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2, 2 }, pooled.shape.dims[0..4]);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), pooled.data[0], 1e-5);

    // Graph mode Dropout + AvgPool2D backward
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const g_img = try graph.tensorND(&.{ 1, 1, 4, 4 }, true);
    for (g_img.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1));

    const g_dropped = try drop.forward(&graph, g_img, rand);
    try std.testing.expect(g_dropped.creator != null);
    try std.testing.expectEqual(autodiff.OpType.Dropout, g_dropped.creator.?.op_type);

    const g_pooled = try pool.forward(&graph, g_dropped);
    try std.testing.expect(g_pooled.creator != null);
    try std.testing.expectEqual(autodiff.OpType.AvgPool2D, g_pooled.creator.?.op_type);

    @memset(g_pooled.grad, 1.0);
    try graph.backwardWithGrad(g_pooled);

    var img_grad_sum: f32 = 0.0;
    for (g_img.grad) |g| img_grad_sum += g;
    try std.testing.expect(img_grad_sum > 0.0);
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

    var graph = autodiff.Graph.init(allocator);
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

    var graph_seq = autodiff.Graph.init(allocator);
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

    var graph = autodiff.Graph.init(allocator);
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

    var graph_seq = autodiff.Graph.init(allocator);
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

    var graph = autodiff.Graph.init(allocator);
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

    var graph = autodiff.Graph.init(allocator);
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

    var graph_seq = autodiff.Graph.init(allocator);
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

    var rms_out_graph = autodiff.Graph.initNoGrad(allocator);
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

    var ln_out_graph = autodiff.Graph.initNoGrad(allocator);
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

    var y_drop0_graph = autodiff.Graph.initNoGrad(allocator);
    defer y_drop0_graph.deinit();
    const y_drop0 = try drop0.forward(&y_drop0_graph, x_test, prng.random());
    try std.testing.expectEqual(x_test, y_drop0);
    for (y_drop0.data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 2.5), v, 1e-5);
    }

    var y_heavy_graph = autodiff.Graph.initNoGrad(allocator);
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
    var deconv = try ConvTranspose2D.init(allocator, 2, 3, 2, 1, 0, true);
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
        .deconv = try ConvTranspose2D.init(allocator, 2, 2, 2, 1, 0, true),
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
        .deconv = try ConvTranspose2D.init(allocator, 2, 2, 2, 1, 0, true),
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

    var graph = autodiff.Graph.init(allocator);
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
    try std.testing.expectEqualStrings("proj_lora", out_lora.creator.?.scope);
    try std.testing.expectEqualStrings(LoRALinear.formula, graph.inferModuleFormula("proj_lora"));
}

test "Conv2D with stride and padding forward and backward (im2col + sgemm)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(2026);
    const random = prng.random();

    var conv = try Conv2D.initWithConfig(allocator, 1, 2, 3, 2, 1);
    conv.resetParameters(random, .{});
    defer nn.deinitModel(&conv, allocator);
    @memset(conv.weight.data[0..9], 1.0);
    @memset(conv.weight.data[9..18], 2.0);
    conv.bias.data[0] = 0.5;
    conv.bias.data[1] = -0.5;

    var graph = autodiff.Graph.init(allocator);
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

test "MoELayer sparse top-k expert execution skips inactive experts" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(77);
    const random = prng.random();

    // 3 routed experts, 0 shared experts, top_k = 1
    var moe = try MoELayer.init(allocator, 4, 8, 3, 0, 1);
    try testing_init.initFromOnes(&moe, allocator, random, &.{ 2, 4 });
    defer nn.deinitModel(&moe, allocator);

    // Force gate weights so expert 0 always wins for positive inputs
    @memset(moe.gate.weight.data, 0.0);
    @memset(moe.gate.bias.data, 0.0);
    moe.gate.bias.data[0] = 10.0;
    moe.gate.bias.data[1] = -10.0;
    moe.gate.bias.data[2] = -10.0;

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x = try graph.tensor(2, 4, true);
    @memset(x.data, 1.0);

    // Graph mode forward + backward
    const out_g = try moe.forward(&graph, x);
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

    // Eager mode forward should match Graph mode output
    var out_e_graph = autodiff.Graph.initNoGrad(allocator);
    defer out_e_graph.deinit();
    const out_e = try moe.forward(&out_e_graph, x);
    for (out_g.data, out_e.data) |vg, ve| {
        try std.testing.expectApproxEqAbs(vg, ve, 1e-4);
    }
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
        var g = autodiff.Graph.init(allocator);
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
        var g = autodiff.Graph.init(allocator);
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
    var g = autodiff.Graph.init(allocator);
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

