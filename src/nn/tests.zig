const std = @import("std");
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
const initializeWeights = nn.initializeWeights;
const createPersistentTensor = nn.createPersistentTensor;
const freePersistentTensor = nn.freePersistentTensor;
const Linear = nn.Linear;
const Conv2D = nn.Conv2D;
const ConvTranspose2D = nn.ConvTranspose2D;
const Module = nn.Module;
const deinitModel = nn.deinitModel;
const zeroGradModel = nn.zeroGradModel;
const collectParameters = nn.collectParameters;
const Sequential = nn.Sequential;
const sequential = nn.sequential;
const autoSequential = nn.autoSequential;
const detectNextActivation = nn.detectNextActivation;

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
const swigluForward = nn.swigluForward;
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
const sftCrossEntropyLoss = nn.sftCrossEntropyLoss;
const sftCrossEntropyLossGraph = nn.sftCrossEntropyLossGraph;
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

test "Embedding Module" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var emb = try Embedding.init(arena, 10, 4, random);
    defer emb.deinit(arena);

    var x = try createPersistentTensor(arena, 2, 3, false);
    defer freePersistentTensor(arena, x);
    x.data[0] = 0; x.data[1] = 1; x.data[2] = 2;
    x.data[3] = 3; x.data[4] = 4; x.data[5] = 5;

    const y_eager = try emb.forward(arena, null, x);
    defer tensor.free(arena, y_eager);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 4}, y_eager.shape.dims[0..y_eager.shape.len]);
}

test "Embedding Module Graph Mode" {
    const arena = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var emb = try Embedding.init(arena, 10, 4, random);
    defer emb.deinit(arena);

    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const x = try graph.tensorND(&.{2, 3}, false);
    x.data[0] = 0; x.data[1] = 1; x.data[2] = 2;
    x.data[3] = 3; x.data[4] = 4; x.data[5] = 5;

    const y = try emb.forward(arena, &graph, x);
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
    defer norm.deinit(arena);

    var x = try createPersistentTensor(arena, 2, 4, false);
    defer freePersistentTensor(arena, x);
    x.data[0] = 1.0; x.data[1] = 2.0; x.data[2] = 3.0; x.data[3] = 4.0;
    x.data[4] = 5.0; x.data[5] = 6.0; x.data[6] = 7.0; x.data[7] = 8.0;

    const y_eager = try norm.forward(arena, null, x);
    defer tensor.free(arena, y_eager);
    try std.testing.expectEqualSlices(usize, &.{2, 4}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 4}, true);
    @memcpy(x_node.data, x.data);

    const y = try norm.forward(arena, &graph, x_node);
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

    var mlp = try MLP.init(arena, 4, 8, random);
    defer mlp.deinit(arena);

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

    const y_eager = try mlp.forward(arena, null, x_3d);
    defer tensor.free(arena, y_eager);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 4}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 3, 4}, true);
    @memcpy(x_node.data, x_3d.data);

    const y = try mlp.forward(arena, &graph, x_node);
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

    var attn = try CausalSelfAttention.init(arena, 8, 2, random);
    defer attn.deinit(arena);

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

    const y_eager = try attn.forward(arena, null, x_3d);
    defer tensor.free(arena, y_eager);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 8}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 3, 8}, true);
    @memcpy(x_node.data, x_3d.data);

    const y = try attn.forward(arena, &graph, x_node);
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

    var gpt = try GPT(config).init(arena, random);
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

    const y_eager = try gpt.forward(arena, null, x);
    defer tensor.free(arena, y_eager);
    try std.testing.expectEqualSlices(usize, &.{2, 3, 10}, y_eager.shape.dims[0..y_eager.shape.len]);

    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 3}, false);
    @memcpy(x_node.data, x.data);

    const y = try gpt.forward(arena, &graph, x_node);
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

    var gpt = try GPT(config).init(arena, random);
    defer deinitModel(&gpt, arena);

    try saveModel(&gpt, std.testing.io, "test_gpt_model.safetensors", arena);
    defer {
        std.Io.Dir.cwd().deleteFile(std.testing.io, "test_gpt_model.safetensors") catch {};
    }

    var gpt2 = try GPT(config).init(arena, random);
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
        try Linear.init(allocator, 10, 20, random),
        ReLU{},
        try Linear.init(allocator, 20, 5, random),
    });
    defer seq.deinit(allocator);

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x = try graph.tensor(2, 10, false);
    @memset(x.data, 0.5);

    const y = try seq.forward(allocator, &graph, x);
    try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    var grad_sum: f32 = 0.0;
    for (seq.layers.@"0".weight.grad) |g| grad_sum += @abs(g);
    try std.testing.expect(grad_sum > 0.0);

    // Test parameter collection on Sequential container
    const params = try collectParameters(&seq, allocator);
    defer allocator.free(params);
    // Two Linear layers each with weight and bias = 4 parameter tensors
    try std.testing.expectEqual(@as(usize, 4), params.len);

    // Test eager forward (graph == null) without memory leak
    const eager_in = try createPersistentTensor(allocator, 2, 10, false);
    defer freePersistentTensor(allocator, eager_in);
    @memset(eager_in.data, 0.5);

    const eager_out = try seq.forward(allocator, null, eager_in);
    defer freePersistentTensor(allocator, eager_out);
    try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, eager_out.shape.dims[0..eager_out.shape.len]);
}

test "SwiGLU forward and backward autograd" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var swiglu = try SwiGLU.init(allocator, 4, 8, random);
    defer swiglu.deinit(allocator);

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x = try graph.tensor(2, 4, true);
    @memset(x.data, 0.5);

    const y = try swiglu.forward(allocator, &graph, x);
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

    var lora = try LoRALinear.init(allocator, 4, 4, 2, 4.0, random);
    defer lora.deinit(allocator);

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x = try graph.tensor(2, 4, false);
    @memset(x.data, 1.0);

    // Initial forward: since B=0, LoRA output must match Base Weight output exactly
    const y = try lora.forward(allocator, &graph, x);
    const base_y = try x.matmul(lora.weight, allocator, null);
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
    const fused_out = try x.matmul(lora.weight, allocator, null);
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
    defer ln.deinit(allocator);

    const x = try tensor.zeros(allocator, &.{ 2, 4 });
    defer tensor.free(allocator, x);
    @memcpy(x.data, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0 });

    const y = try ln.forward(allocator, null, x);
    defer tensor.free(allocator, y);

    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, y.shape.dims[0..2]);
    var sum: f32 = 0.0;
    for (y.data[0..4]) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sum / 4.0, 1e-4);

    // Graph mode forward + backward
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const gx = try graph.tensorNDWithData(&.{ 2, 4 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 1.0, 3.0 }, true);
    const gy = try ln.forward(allocator, &graph, gx);
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
    defer bn.deinit(allocator);

    const x = try tensor.zeros(allocator, &.{ 2, 2, 2, 2 });
    defer tensor.free(allocator, x);
    for (x.data, 0..) |*p, i| p.* = @as(f32, @floatFromInt(i));

    const y = try bn.forward(allocator, null, x);
    defer tensor.free(allocator, y);

    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 2 }, y.shape.dims[0..4]);

    // Graph mode forward + backward
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const gx = try graph.tensorND(&.{ 2, 2, 2, 2 }, true);
    for (gx.data, 0..) |*p, i| {
        const fi = @as(f32, @floatFromInt(i));
        p.* = @sin(fi * 1.3) + 0.2 * fi;
    }
    const gy = try bn.forward(allocator, &graph, gx);
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

    const y_drop = try drop.forward(allocator, null, x, rand);
    defer tensor.free(allocator, y_drop);
    try std.testing.expectEqual(10, y_drop.data.len);

    const pool = AvgPool2D.init(2, 2);
    const img = try tensor.zeros(allocator, &.{ 1, 1, 4, 4 });
    defer tensor.free(allocator, img);
    @memset(img.data, 4.0);

    const pooled = try pool.forward(allocator, null, img);
    defer tensor.free(allocator, pooled);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2, 2 }, pooled.shape.dims[0..4]);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), pooled.data[0], 1e-5);

    // Graph mode Dropout + AvgPool2D backward
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const g_img = try graph.tensorND(&.{ 1, 1, 4, 4 }, true);
    for (g_img.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1));

    const g_dropped = try drop.forward(allocator, &graph, g_img, rand);
    try std.testing.expect(g_dropped.creator != null);
    try std.testing.expectEqual(autodiff.OpType.Dropout, g_dropped.creator.?.op_type);

    const g_pooled = try pool.forward(allocator, &graph, g_dropped);
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
    var cell = try RNNCell.init(allocator, 4, 3, random);
    defer cell.deinit(allocator);

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x0 = try graph.tensorND(&.{ 2, 4 }, true);
    for (x0.data, 0..) |*p, i| p.* = @as(f32, @floatFromInt(i)) * 0.1;
    const h0 = try graph.zeros(&.{ 2, 3 }, true);

    const h1 = try cell.forward(allocator, &graph, x0, h0);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, h1.shape.dims[0..2]);

    @memset(h1.grad, 1.0);
    try graph.backward(h1);

    var grad_sum: f32 = 0.0;
    for (x0.grad) |g| grad_sum += @abs(g);
    try std.testing.expect(grad_sum > 1e-4);

    // 2. RNN sequence container test
    var rnn = try RNN.init(allocator, 4, 3, random);
    defer rnn.deinit(allocator);

    var graph_seq = autodiff.Graph.init(allocator);
    defer graph_seq.deinit();

    const x_seq_0 = try graph_seq.tensorND(&.{ 2, 4 }, true);
    const x_seq_1 = try graph_seq.tensorND(&.{ 2, 4 }, true);
    @memset(x_seq_0.data, 0.5);
    @memset(x_seq_1.data, -0.5);

    const inputs = [_]*Tensor{ x_seq_0, x_seq_1 };
    const res = try rnn.forward(allocator, &graph_seq, &inputs, null);
    defer allocator.free(res.outputs);

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
    var cell = try LSTMCell.init(allocator, 4, 3, random);
    defer cell.deinit(allocator);

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

    const state = try cell.forward(allocator, &graph, x0, h0, c0);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, state.h.shape.dims[0..2]);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, state.c.shape.dims[0..2]);

    @memset(state.h.grad, 1.0);
    try graph.backward(state.h);

    var grad_sum: f32 = 0.0;
    for (x0.grad) |g| grad_sum += @abs(g);
    try std.testing.expect(grad_sum > 1e-4);

    // 2. LSTM sequence container test
    var lstm = try LSTM.init(allocator, 4, 3, random);
    defer lstm.deinit(allocator);

    var graph_seq = autodiff.Graph.init(allocator);
    defer graph_seq.deinit();

    const x_seq_0 = try graph_seq.tensorND(&.{ 2, 4 }, true);
    const x_seq_1 = try graph_seq.tensorND(&.{ 2, 4 }, true);
    @memset(x_seq_0.data, 0.2);
    @memset(x_seq_1.data, -0.2);

    const inputs = [_]*Tensor{ x_seq_0, x_seq_1 };
    const res = try lstm.forward(allocator, &graph_seq, &inputs, null, null);
    defer allocator.free(res.outputs);

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

    var stacked = try StackedLSTM.init(allocator, 4, 3, 2, random);
    defer stacked.deinit(allocator);

    try std.testing.expectEqual(2, stacked.num_layers);

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x0 = try graph.tensorND(&.{ 2, 4 }, true);
    const x1 = try graph.tensorND(&.{ 2, 4 }, true);
    @memset(x0.data, 0.3);
    @memset(x1.data, -0.3);

    const inputs = [_]*Tensor{ x0, x1 };
    const res = try stacked.forwardSequence(allocator, &graph, &inputs, null, null);
    defer allocator.free(res.outputs);
    defer allocator.free(res.h_n);
    defer allocator.free(res.c_n);

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
    var cell = try GRUCell.init(allocator, 4, 3, random);
    defer cell.deinit(allocator);

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x0 = try graph.tensorND(&.{ 2, 4 }, true);
    for (x0.data, 0..) |*p, i| p.* = @as(f32, @floatFromInt(i)) * 0.1;
    const h0 = try graph.zeros(&.{ 2, 3 }, true);

    const h1 = try cell.forward(allocator, &graph, x0, h0);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, h1.shape.dims[0..2]);

    @memset(h1.grad, 1.0);
    try graph.backward(h1);

    var grad_sum: f32 = 0.0;
    for (x0.grad) |g| grad_sum += @abs(g);
    try std.testing.expect(grad_sum > 1e-4);

    // 2. GRU sequence container test
    var gru = try GRU.init(allocator, 4, 3, random);
    defer gru.deinit(allocator);

    var graph_seq = autodiff.Graph.init(allocator);
    defer graph_seq.deinit();

    const x_seq_0 = try graph_seq.tensorND(&.{ 2, 4 }, true);
    const x_seq_1 = try graph_seq.tensorND(&.{ 2, 4 }, true);
    @memset(x_seq_0.data, 0.4);
    @memset(x_seq_1.data, -0.4);

    const inputs = [_]*Tensor{ x_seq_0, x_seq_1 };
    const res = try gru.forward(allocator, &graph_seq, &inputs, null);
    defer allocator.free(res.outputs);

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
    defer rms.deinit(allocator);

    const x_zeros = try tensor.zeros(allocator, &.{ 1, 4 });
    defer tensor.free(allocator, x_zeros);

    const rms_out = try rms.forward(allocator, null, x_zeros);
    defer tensor.free(allocator, rms_out);

    for (rms_out.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }

    // 2. LayerNorm with uniform row (all 3.0): mean=3.0, var=0.0 -> output is 0.0 (no NaN)
    var ln = try LayerNorm.init(allocator, 4, 1e-5);
    defer ln.deinit(allocator);

    const x_uniform = try tensor.zeros(allocator, &.{ 1, 4 });
    defer tensor.free(allocator, x_uniform);
    @memset(x_uniform.data, 3.0);

    const ln_out = try ln.forward(allocator, null, x_uniform);
    defer tensor.free(allocator, ln_out);

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

    const y_drop0 = try drop0.forward(allocator, null, x_test, prng.random());
    try std.testing.expectEqual(x_test, y_drop0);
    for (y_drop0.data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 2.5), v, 1e-5);
    }

    const y_heavy = try drop_heavy.forward(allocator, null, x_test, prng.random());
    defer tensor.free(allocator, y_heavy);
    for (y_heavy.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "Weight initialization methods and Linear initWithOptions" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n_elements: usize = 20000;
    const buf = try allocator.alloc(f32, n_elements);
    defer allocator.free(buf);

    // 1. Zeros
    initWeights(random, buf, 100, 100, .zeros);
    for (buf) |v| try std.testing.expectEqual(@as(f32, 0.0), v);

    // 2. Ones
    initWeights(random, buf, 100, 100, .ones);
    for (buf) |v| try std.testing.expectEqual(@as(f32, 1.0), v);

    // 3. Constant
    initWeights(random, buf, 100, 100, .{ .constant = 3.14 });
    for (buf) |v| try std.testing.expectApproxEqAbs(@as(f32, 3.14), v, 1e-5);

    // Helper to calculate mean and variance
    const calcStats = struct {
        fn run(slice: []const f32) struct { mean: f32, variance: f32 } {
            var sum: f64 = 0.0;
            for (slice) |v| sum += v;
            const mean: f64 = sum / @as(f64, @floatFromInt(slice.len));
            var var_sum: f64 = 0.0;
            for (slice) |v| {
                const diff = @as(f64, v) - mean;
                var_sum += diff * diff;
            }
            const variance: f64 = var_sum / @as(f64, @floatFromInt(slice.len));
            return .{ .mean = @floatCast(mean), .variance = @floatCast(variance) };
        }
    }.run;

    // 4. He Normal: Var = gain^2 / fan_in = 2 / 100 = 0.02
    initWeights(random, buf, 100, 100, .{ .he_normal = .{} });
    const he_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), he_stats.mean, 0.015);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), he_stats.variance, 0.003);

    // 5. Xavier Normal: Var = gain^2 * 2 / (fan_in + fan_out) = 2 / 200 = 0.01
    initWeights(random, buf, 100, 100, .{ .xavier_normal = .{} });
    const xavier_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), xavier_stats.mean, 0.015);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), xavier_stats.variance, 0.002);

    // 6. LeCun Normal: Var = 1 / fan_in = 1 / 100 = 0.01
    initWeights(random, buf, 100, 100, .lecun_normal);
    const lecun_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lecun_stats.mean, 0.015);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), lecun_stats.variance, 0.002);

    // 7. Linear with customInit specifying nonlinearity
    var lin_tanh = try Linear.initClean(allocator, 100, 100);
    defer lin_tanh.deinit(allocator);
    lin_tanh.customInit(random, .{
        .nonlinearity = .tanh, // Gain = 5/3 ~ 1.6667 -> Xavier Normal with Gain
        .bias_init = .{ .constant = 0.5 },
    });

    const tanh_stats = calcStats(lin_tanh.weight.data);
    // Var = gain^2 * 2 / (100 + 100) = (25 / 9) * 2 / 200 = 25 / 900 ~ 0.02778
    try std.testing.expectApproxEqAbs(@as(f32, 0.02778), tanh_stats.variance, 0.005);
    for (lin_tanh.bias.data) |b| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.5), b, 1e-5);
    }
    try std.testing.expect(lin_tanh.weight.is_custom_initialized);
    try std.testing.expect(lin_tanh.bias.is_custom_initialized);
}

test "Sequential autoInit and detectNextActivation" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(1234);
    const random = prng.random();

    // 自动检测网络：
    // fc1 -> ReLU (自动探测为 .relu -> He Normal, gain=sqrt(2))
    // fc2 -> Tanh (自动探测为 .tanh -> Xavier Normal, gain=5/3)
    // fc3 -> 无激活函数 (自动探测为 .linear, gain=1.0)
    var model = autoSequential(.{
        try Linear.initUninitialized(allocator, 100, 100),
        ReLU{},
        try Linear.initUninitialized(allocator, 100, 100),
        Tanh{},
        try Linear.initUninitialized(allocator, 100, 10),
    }, random);
    defer model.deinit(allocator);

    const calcVar = struct {
        fn run(slice: []const f32) f32 {
            var sum: f64 = 0.0;
            for (slice) |v| sum += v;
            const mean = sum / @as(f64, @floatFromInt(slice.len));
            var var_sum: f64 = 0.0;
            for (slice) |v| {
                const diff = @as(f64, v) - mean;
                var_sum += diff * diff;
            }
            return @floatCast(var_sum / @as(f64, @floatFromInt(slice.len)));
        }
    }.run;

    // 1. 验证编译期类型推导
    const TupleT = @TypeOf(model.layers);
    try std.testing.expectEqual(Nonlinearity.relu, detectNextActivation(TupleT, 0));
    try std.testing.expectEqual(Nonlinearity.tanh, detectNextActivation(TupleT, 2));
    try std.testing.expectEqual(Nonlinearity.linear, detectNextActivation(TupleT, 4));

    // 2. 统计方差验证
    // fc1 (ReLU): Var = 2 / 100 = 0.02
    const var_fc1 = calcVar(model.layers.@"0".weight.data);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), var_fc1, 0.003);

    // fc2 (Tanh): Var = (5/3)^2 * 2 / 200 = 25 / 900 ~ 0.02778
    const var_fc2 = calcVar(model.layers.@"2".weight.data);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02778), var_fc2, 0.004);

    // fc3 (Linear/Logits): Var = 1.0^2 / 100 = 0.01 (无放大！)
    const var_fc3 = calcVar(model.layers.@"4".weight.data);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), var_fc3, 0.003);
}

test "Graph.initWeights dynamically infers activations and respects customInit" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(999);
    const random = prng.random();

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    // 1. 各层通过干净的 initClean 创建（无随机数，只分配内存）并设置人类可读名字
    var fc_relu = try Linear.initClean(allocator, 100, 100);
    defer fc_relu.deinit(allocator);
    fc_relu.setName("dense_relu_1");

    var fc_tanh = try Linear.initClean(allocator, 100, 100);
    defer fc_tanh.deinit(allocator);
    fc_tanh.setName("dense_tanh_2");

    var fc_custom = try Linear.initClean(allocator, 100, 10);
    defer fc_custom.deinit(allocator);
    fc_custom.setName("special_head");

    // 2. 特殊层显式调用 customInit：指定常数偏置 3.14，并随机初始化权重
    fc_custom.customInit(random, .{
        .nonlinearity = .linear,
        .bias_init = .{ .constant = 3.14 },
    });
    // 记录 custom 权重切片的一个样本以验证后续不被 Graph 篡改重写
    const custom_weight_sample = fc_custom.weight.data[0];

    // 3. 在构造/连接期通过各类 Operation 将图自然动态串联起来
    const x = try graph.zeros(&.{ 2, 100 }, false);
    x.setName("features_input");

    const z1 = try graph.addBias(try graph.matmul(x, fc_relu.weight), fc_relu.bias);
    const a1 = try graph.relu(z1); // 后续接 ReLU
    a1.setName("relu_activation_1");

    const z2 = try graph.addBias(try graph.matmul(a1, fc_tanh.weight), fc_tanh.bias);
    const a2 = try graph.tanh(z2); // 后续接 Tanh

    var fc_out = try Linear.initClean(allocator, 10, 2);
    defer fc_out.deinit(allocator);
    fc_out.setName("logits_out");

    const logits = try graph.addBias(try graph.matmul(a2, fc_custom.weight), fc_custom.bias);
    logits.setName("custom_head_logits");

    const final_out = try graph.addBias(try graph.matmul(logits, fc_out.weight), fc_out.bias);
    final_out.setName("network_final_out");

    // 4. 一键初始化全图！
    graph.initWeights(random);

    const calcVar = struct {
        fn run(slice: []const f32) f32 {
            var sum: f64 = 0.0;
            for (slice) |v| sum += v;
            const mean = sum / @as(f64, @floatFromInt(slice.len));
            var var_sum: f64 = 0.0;
            for (slice) |v| {
                const diff = @as(f64, v) - mean;
                var_sum += diff * diff;
            }
            return @floatCast(var_sum / @as(f64, @floatFromInt(slice.len)));
        }
    }.run;

    // 5. 校验：
    // fc_relu 后面探查到 ReLU -> 自动赋 He Normal (Var = 2 / 100 = 0.02)
    const var_relu = calcVar(fc_relu.weight.data);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), var_relu, 0.003);

    // fc_tanh 后面探查到 Tanh -> 自动赋 Xavier Normal (Var = (5/3)^2 * 2 / 200 ~ 0.02778)
    const var_tanh = calcVar(fc_tanh.weight.data);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02778), var_tanh, 0.004);

    // fc_custom 已经过 customInit -> 绝对不会被 Graph 重写覆盖！
    try std.testing.expectEqual(custom_weight_sample, fc_custom.weight.data[0]);
    for (fc_custom.bias.data) |b| {
        try std.testing.expectApproxEqAbs(@as(f32, 3.14), b, 1e-5);
    }

    // 6. 测试 formatInitReport 能够正常输出自定义名称、非参数节点 (Input, Activation) 和各种初始化状态
    const report_str = try graph.formatInitReport(allocator);
    defer allocator.free(report_str);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "dense_relu_1.weight") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "dense_relu_1.bias") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "dense_tanh_2.weight") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "special_head.weight") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "logits_out.weight") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "features_input") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "relu_activation_1") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "custom_head_logits") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "network_final_out") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "Input") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "Activation") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "Param") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "CUSTOM_INIT") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "AUTO_GRAPH") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "OP_OUTPUT") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "ReLU") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "Tanh") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "Linear (None)") != null);
}

test "Comprehensive coverage of all InitMethod strategies" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(54321);
    const random = prng.random();

    const n = 10000;
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);

    const calcStats = struct {
        fn run(slice: []const f32) struct { mean: f32, variance: f32, min: f32, max: f32 } {
            var sum: f64 = 0.0;
            var min_v: f32 = slice[0];
            var max_v: f32 = slice[0];
            for (slice) |v| {
                sum += v;
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }
            const mean = sum / @as(f64, @floatFromInt(slice.len));
            var var_sum: f64 = 0.0;
            for (slice) |v| {
                const diff = @as(f64, v) - mean;
                var_sum += diff * diff;
            }
            return .{
                .mean = @floatCast(mean),
                .variance = @floatCast(var_sum / @as(f64, @floatFromInt(slice.len))),
                .min = min_v,
                .max = max_v,
            };
        }
    }.run;

    // 1. Zeros, Ones, Constant
    initWeights(random, buf, 100, 100, .zeros);
    for (buf) |v| try std.testing.expectEqual(@as(f32, 0.0), v);

    initWeights(random, buf, 100, 100, .ones);
    for (buf) |v| try std.testing.expectEqual(@as(f32, 1.0), v);

    initWeights(random, buf, 100, 100, .{ .constant = 42.0 });
    for (buf) |v| try std.testing.expectEqual(@as(f32, 42.0), v);

    // 2. Normal (mean=2.0, std=0.5)
    initWeights(random, buf, 100, 100, .{ .normal = .{ .mean = 2.0, .std = 0.5 } });
    const norm_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), norm_stats.mean, 0.03);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), norm_stats.variance, 0.02);

    // 3. Uniform (min=-1.0, max=3.0) -> mean=1.0, var=(3 - (-1))^2 / 12 = 16/12 ~ 1.3333
    initWeights(random, buf, 100, 100, .{ .uniform = .{ .min = -1.0, .max = 3.0 } });
    const uni_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), uni_stats.mean, 0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 1.3333), uni_stats.variance, 0.05);
    try std.testing.expect(uni_stats.min >= -1.0);
    try std.testing.expect(uni_stats.max <= 3.0);

    // 4. Xavier Uniform: limit = gain * sqrt(6 / (100 + 100)) = sqrt(6/200) = sqrt(0.03) ~ 0.1732
    // Var = limit^2 / 3 = 0.03 / 3 = 0.01
    initWeights(random, buf, 100, 100, .{ .xavier_uniform = .{} });
    const xu_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), xu_stats.mean, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), xu_stats.variance, 0.002);
    try std.testing.expect(xu_stats.max <= 0.1733);
    try std.testing.expect(xu_stats.min >= -0.1733);

    // 5. He Uniform: limit = gain * sqrt(3 / 100) = sqrt(2) * sqrt(0.03) = sqrt(0.06) ~ 0.2449
    // Var = limit^2 / 3 = 0.06 / 3 = 0.02
    initWeights(random, buf, 100, 100, .{ .he_uniform = .{} });
    const hu_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), hu_stats.mean, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), hu_stats.variance, 0.003);

    // 6. LeCun Uniform: limit = sqrt(3 / 100) ~ 0.1732, Var = 0.03 / 3 = 0.01
    initWeights(random, buf, 100, 100, .lecun_uniform);
    const lu_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lu_stats.mean, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), lu_stats.variance, 0.002);
}

test "Hierarchical module naming and interactive HTML report export" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(13579);
    const random = prng.random();

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    // 1. 创建 Embedding 模块并设置顶级层次命名
    var emb = try transformer.Embedding.init(allocator, 1000, 64, random);
    defer emb.deinit(allocator);
    emb.setName("gpt.wte");

    // 2. 创建 TransformerBlock 并分层命名为 "gpt.layers.0"
    var block = try transformer.TransformerBlock.init(allocator, 64, 4, random);
    defer block.deinit(allocator);
    block.setName("gpt.layers.0");
    try graph.registerModuleType("gpt", "GPT");

    // 验证子层参数名称是否按层次正确拼接
    try std.testing.expectEqualStrings("gpt.wte.weight", emb.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.ln_1.weight", block.ln_1.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.attn.q_attn.weight", block.attn.q_attn.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.attn.k_attn.weight", block.attn.k_attn.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.attn.v_attn.weight", block.attn.v_attn.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.attn.c_proj.weight", block.attn.c_proj.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.mlp.c_fc.weight", block.mlp.c_fc.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.mlp.c_proj.weight", block.mlp.c_proj.weight.name.?);

    // 3. 构建前向计算图并命名输入与中间激活节点
    const input_tokens = try graph.zeros(&.{ 2, 8, 64 }, false);
    input_tokens.setName("inputs.token_embeddings");

    const block_out = try block.forward(allocator, &graph, input_tokens);
    block_out.setName("activations.block_0_out");

    // 4. 生成递归结构 JSON (后端生成递归数据结构，直接提供给前端解析)
    const json_data = try graph.formatJson(allocator);
    defer allocator.free(json_data);

    // 校验 JSON 结构可正常被解析且包含完整的递归模型树
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_data, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value == .object);

    const root_obj = parsed.value.object.get("root").?.object;
    try std.testing.expectEqualStrings("root", root_obj.get("name").?.string);
    try std.testing.expect(root_obj.get("total_params").?.integer > 0);
    try std.testing.expect(root_obj.get("children").?.array.items.len > 0);

    // 校验根与子模块直接内嵌包含 formula, module_type, parameters, ops 与 edges
    // 图输入归属于 root.nodes，不再生成 "inputs" 伪模块，因此 gpt 是唯一的根子模块
    try std.testing.expectEqual(@as(usize, 1), root_obj.get("children").?.array.items.len);
    const gpt_child = root_obj.get("children").?.array.items[0].object; // "gpt"
    try std.testing.expectEqualStrings("GPT", gpt_child.get("module_type").?.string);
    try std.testing.expect(gpt_child.get("formula") != null);

    const layer0_obj = gpt_child.get("children").?.array.items[0].object.get("children").?.array.items[0].object; // "gpt.layers.0"
    try std.testing.expectEqualStrings("TransformerBlock", layer0_obj.get("module_type").?.string);
    try std.testing.expect(std.mem.indexOf(u8, layer0_obj.get("formula").?.string, "TransformerBlock") != null);
    try std.testing.expect(std.mem.indexOf(u8, layer0_obj.get("formula").?.string, "Attention") != null);
    try std.testing.expect(std.mem.indexOf(u8, layer0_obj.get("formula").?.string, "MLP") != null);
    try std.testing.expect(layer0_obj.get("ops").?.array.items.len > 0);
    try std.testing.expect(layer0_obj.get("edges").?.array.items.len > 0);

    const q_attn_obj = layer0_obj.get("children").?.array.items[1].object.get("children").?.array.items[0].object; // "q_attn"
    try std.testing.expectEqualStrings("Linear", q_attn_obj.get("module_type").?.string);
    try std.testing.expectEqualStrings("y = x W^T + b", q_attn_obj.get("formula").?.string);
    try std.testing.expectEqual(2, q_attn_obj.get("parameters").?.array.items.len);
    try std.testing.expect(q_attn_obj.get("ops").?.array.items.len >= 2);

    const summary_obj = parsed.value.object.get("summary").?.object;
    try std.testing.expect(summary_obj.get("total_params").?.integer > 0);
    try std.testing.expect(summary_obj.get("param_nodes").?.integer >= 8);

    // 校验 schema 2.0 顶层结构：nodes, ops, formulas, edges 均已就近集成进模块树，
    // 顶层仅保留 default_scope 指示前端初始展开的作用域
    try std.testing.expectEqualStrings("2.0", parsed.value.object.get("version").?.string);
    try std.testing.expect(parsed.value.object.get("nodes") == null);
    try std.testing.expect(parsed.value.object.get("edges") == null);
    try std.testing.expect(parsed.value.object.get("ops") == null);
    try std.testing.expect(parsed.value.object.get("formulas") == null);
    try std.testing.expectEqualStrings("gpt", parsed.value.object.get("default_scope").?.string);

    // 校验 Block0 级别的拓扑边 (Edges) 正确性：
    // 1) 包含前向流 ln_1 -> attn 以及残差边 inputs.token_embeddings -> residual_attn (is_skip = true)
    // 2) 严禁包含子模块内部 Q/K/V 指向 attn 的泄露边
    var found_ln1_to_attn = false;
    var found_skip_to_res = false;
    var leaked_qkv_to_attn = false;
    for (layer0_obj.get("edges").?.array.items) |e_val| {
        const edge = e_val.object;
        const from_s = edge.get("from").?.string;
        const to_s = edge.get("to").?.string;
        const is_skip = edge.get("is_skip").?.bool;

        if (std.mem.indexOf(u8, from_s, "ln_1") != null and std.mem.indexOf(u8, to_s, "attn") != null and !is_skip) {
            found_ln1_to_attn = true;
        }
        if (std.mem.indexOf(u8, to_s, "residual_attn") != null and is_skip) {
            found_skip_to_res = true;
        }
        if ((std.mem.eql(u8, from_s, "q_attn") or std.mem.eql(u8, from_s, "k_attn") or std.mem.eql(u8, from_s, "v_attn")) and std.mem.eql(u8, to_s, "attn")) {
            leaked_qkv_to_attn = true;
        }
    }
    try std.testing.expect(found_ln1_to_attn);
    try std.testing.expect(found_skip_to_res);
    try std.testing.expect(!leaked_qkv_to_attn);

    // 校验参数矩阵守恒与内存指标递归一致性 (Conservation Check)
    const ParamCounter = struct {
        fn countParams(obj: std.json.ObjectMap) usize {
            var sum: usize = 0;
            if (obj.get("parameters")) |p_val| {
                for (p_val.array.items) |p_item| {
                    if (p_item.object.get("elements")) |el| {
                        sum += @as(usize, @intCast(el.integer));
                    }
                }
            }
            if (obj.get("children")) |c_val| {
                for (c_val.array.items) |c_item| {
                    sum += countParams(c_item.object);
                }
            }
            return sum;
        }
    };
    const total_recursed_params = ParamCounter.countParams(root_obj);
    try std.testing.expectEqual(summary_obj.get("total_params").?.integer, @as(i64, @intCast(total_recursed_params)));

    // 校验算子（Ops）拓扑与张量流转维度非空完整性
    for (q_attn_obj.get("ops").?.array.items) |op_val| {
        const op_obj = op_val.object;
        try std.testing.expect(op_obj.get("op_type") != null);
        try std.testing.expect(op_obj.get("input_shape") != null);
        try std.testing.expect(op_obj.get("output_shape") != null);
        try std.testing.expect(op_obj.get("elements").?.integer > 0);
        try std.testing.expect(op_obj.get("bytes").?.integer > 0);
    }

    // 5. 测试将递归 JSON 导出到临时测试文件
    const tmp_json_path = "tmp_test_model_graph.json";
    try graph.exportJson(tmp_json_path);

    const json_z = try allocator.dupeZ(u8, tmp_json_path);
    defer allocator.free(json_z);
    const c_api = struct { extern "c" fn remove(filename: [*:0]const u8) c_int; };
    defer _ = c_api.remove(json_z.ptr);
    const fj = std.c.fopen(json_z.ptr, "rb") orelse return error.CannotOpenFile;
    defer _ = std.c.fclose(fj);
    var check_json_buf: [1024]u8 = undefined;
    const json_bytes_read = std.c.fread(&check_json_buf, 1, check_json_buf.len, fj);
    try std.testing.expect(json_bytes_read > 200);

    // 6. 测试直接从 ModelHierarchyGraph 序列化 JSON (与 Graph 实例解耦)
    var model_hierarchy = try graph_ir.build(&graph, allocator);
    defer model_hierarchy.deinit();
    const json_from_struct = try graph_ir.serializeJson(&model_hierarchy, allocator);
    defer allocator.free(json_from_struct);
    try std.testing.expect(std.mem.indexOf(u8, json_from_struct, "\"module_type\": \"GPT\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_from_struct, "\"gpt.layers.0.attn.q_attn.weight\"") != null);
}

test "End-to-End Multi-layer GPT JSON Graph Topology and Cross-layer Connectivity" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(54321);
    const random = prng.random();

    // 1. 配置 2 层标准 GPT 模型
    const config = transformer.GPTConfig{
        .vocab_size = 128,
        .block_size = 16,
        .n_embd = 32,
        .n_head = 2,
        .n_layer = 2,
    };

    var gpt = try transformer.GPT(config).init(allocator, random);
    defer gpt.deinit(allocator);
    gpt.setName("gpt");

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    // 2. 构造输入张量并执行前向传播
    const batch_size: usize = 2;
    const seq_len: usize = 8;
    const token_data = try allocator.alloc(f32, batch_size * seq_len);
    defer allocator.free(token_data);
    for (token_data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 50));

    const input_tokens = try graph.tensorNDWithData(&.{ batch_size, seq_len }, token_data, false);
    input_tokens.setName("inputs.token_ids");

    const logits = try gpt.forward(allocator, &graph, input_tokens);
    logits.setName("outputs.logits");

    // 3. 构建 ModelHierarchyGraph 与 JSON
    const json_data = try graph.formatJson(allocator);
    defer allocator.free(json_data);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_data, .{});
    defer parsed.deinit();

    const root_obj = parsed.value.object.get("root").?.object;
    const gpt_obj = root_obj.get("children").?.array.items[0].object; // "gpt"
    const layers_obj = gpt_obj.get("children").?.array.items[2].object; // "layers"
    try std.testing.expectEqualStrings("TransformerDecoder", layers_obj.get("module_type").?.string);

    // 4. 校验跨层连接 (Cross-layer continuity between Layer 0 and Layer 1)
    // layers 容器内部必须正确记录 0 -> 1 的前向流以及 0 到 1 的残差连接
    var found_layer0_to_1 = false;
    for (layers_obj.get("edges").?.array.items) |e_item| {
        const edge = e_item.object;
        const from_s = edge.get("from").?.string;
        const to_s = edge.get("to").?.string;
        if (std.mem.eql(u8, from_s, "0") and std.mem.eql(u8, to_s, "1")) {
            found_layer0_to_1 = true;
        }
    }
    try std.testing.expect(found_layer0_to_1);

    // 5. 校验 Layer 1 内部的残差汇聚结构与公式
    const layer1_obj = layers_obj.get("children").?.array.items[1].object; // "gpt.layers.1"
    try std.testing.expectEqualStrings("TransformerBlock", layer1_obj.get("module_type").?.string);
    try std.testing.expect(layer1_obj.get("edges").?.array.items.len > 0);

    var found_layer1_res_skip = false;
    for (layer1_obj.get("edges").?.array.items) |e_item| {
        const edge = e_item.object;
        const to_s = edge.get("to").?.string;
        const is_skip = edge.get("is_skip").?.bool;
        if (std.mem.indexOf(u8, to_s, "residual_attn") != null and is_skip) {
            found_layer1_res_skip = true;
        }
    }
    try std.testing.expect(found_layer1_res_skip);
}

/// 可视化测试辅助：按完整路径查找模块节点，并以 "from->to[ skip| buffer]" 形式比较局部边集合
const VisTestUtil = struct {
    fn findModule(node: std.json.ObjectMap, path: []const u8) ?std.json.ObjectMap {
        if (std.mem.eql(u8, node.get("path").?.string, path)) return node;
        if (node.get("children")) |children| {
            for (children.array.items) |child| {
                if (findModule(child.object, path)) |m| return m;
            }
        }
        return null;
    }

    fn expectEdgeSet(allocator: std.mem.Allocator, root: std.json.ObjectMap, path: []const u8, expected: []const []const u8) !void {
        const module = findModule(root, path) orelse return error.ModuleNotFound;
        var actual: std.ArrayList([]u8) = .empty;
        defer {
            for (actual.items) |s| allocator.free(s);
            actual.deinit(allocator);
        }
        for (module.get("edges").?.array.items) |e_val| {
            const edge = e_val.object;
            const tag: []const u8 = if (edge.get("is_skip").?.bool)
                " skip"
            else if (std.mem.eql(u8, edge.get("kind").?.string, "buffer"))
                " buffer"
            else
                "";
            try actual.append(allocator, try std.fmt.allocPrint(allocator, "{s}->{s}{s}", .{
                edge.get("from").?.string, edge.get("to").?.string, tag,
            }));
        }
        errdefer {
            std.debug.print("edge set mismatch in scope '{s}', actual edges:\n", .{path});
            for (actual.items) |s| std.debug.print("  {s}\n", .{s});
        }
        try std.testing.expectEqual(expected.len, actual.items.len);
        for (expected) |want| {
            var found = false;
            for (actual.items) |have| {
                if (std.mem.eql(u8, want, have)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std.debug.print("missing edge: {s}\n", .{want});
                return error.MissingEdge;
            }
        }
    }

    fn findEdge(root: std.json.ObjectMap, path: []const u8, from: []const u8, to: []const u8) !std.json.ObjectMap {
        const module = findModule(root, path) orelse return error.ModuleNotFound;
        for (module.get("edges").?.array.items) |e_val| {
            const edge = e_val.object;
            if (std.mem.eql(u8, edge.get("from").?.string, from) and std.mem.eql(u8, edge.get("to").?.string, to)) return edge;
        }
        return error.EdgeNotFound;
    }

    fn expectPortRef(root: std.json.ObjectMap, path: []const u8, direction: []const u8, index: usize, ref: []const u8) !void {
        const module = findModule(root, path) orelse return error.ModuleNotFound;
        const ports = module.get("ports").?.object.get(direction).?.array.items;
        try std.testing.expect(index < ports.len);
        try std.testing.expectEqualStrings(ref, ports[index].object.get("ref").?.string);
    }
};

test "Explicit module scopes attribute ops and tensors to the executing module" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(24680);
    const random = prng.random();

    const config = transformer.GPTConfig{ .vocab_size = 128, .block_size = 16, .n_embd = 32, .n_head = 2, .n_layer = 2 };
    var gpt = try transformer.GPT(config).init(allocator, random);
    defer gpt.deinit(allocator);
    gpt.setName("gpt");

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    var token_data: [2 * 8]f32 = undefined;
    for (&token_data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 50));
    const input_tokens = try graph.tensorNDWithData(&.{ 2, 8 }, &token_data, false);
    input_tokens.setName("inputs.token_ids");
    _ = try gpt.forward(allocator, &graph, input_tokens);

    // forward 结束后作用域栈必须完全弹出
    try std.testing.expectEqualStrings("", graph.currentScope());

    // 图输入在任何模块之外创建，归属根作用域
    try std.testing.expectEqualStrings("", input_tokens.scope);

    var gelu_scope: ?[]const u8 = null;
    var core_transpose_found = false;
    var attn_reshape_found = false;
    for (graph.ops.items) |o| {
        switch (o.op_type) {
            .Gelu => if (gelu_scope == null) {
                gelu_scope = o.scope;
            },
            .Transpose => if (std.mem.eql(u8, o.scope, "gpt.layers.0.attn.core")) {
                core_transpose_found = true;
            },
            .Reshape => if (std.mem.eql(u8, o.scope, "gpt.layers.0.attn")) {
                attn_reshape_found = true;
            },
            else => {},
        }
        // 叶子模块 ln_1 只执行 RMSNorm 计算，不应拥有任何 Reshape
        if (o.op_type == .Reshape) try std.testing.expect(!std.mem.eql(u8, o.scope, "gpt.layers.0.ln_1"));
    }
    try std.testing.expectEqualStrings("gpt.layers.0.mlp", gelu_scope.?);
    try std.testing.expect(core_transpose_found);
    try std.testing.expect(attn_reshape_found);

    // 输出 logits 的最终 Reshape 由 GPT 自身执行，归属 "gpt"
    const last_op = graph.ops.items[graph.ops.items.len - 1];
    try std.testing.expectEqual(autodiff.OpType.Reshape, last_op.op_type);
    try std.testing.expectEqualStrings("gpt", last_op.scope);

    // pos_indices 是 GPT 内部创建的静态缓冲区
    var pos_found = false;
    for (graph.tensors.items) |t| {
        if (t.name) |n| if (std.mem.eql(u8, n, "gpt.pos_indices")) {
            pos_found = true;
            try std.testing.expect(t.is_buffer);
            try std.testing.expectEqualStrings("gpt", t.scope);
        };
    }
    try std.testing.expect(pos_found);

    // 模块类型由 enterModule 自动注册
    try std.testing.expectEqualStrings("CausalSelfAttention", graph.module_types.get("gpt.layers.0.attn").?);
    try std.testing.expectEqualStrings("ScaledDotProductAttention", graph.module_types.get("gpt.layers.0.attn.core").?);
}

test "Scoped local graph export matches golden edge sets (schema 2.0)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(97531);
    const random = prng.random();

    const config = transformer.GPTConfig{ .vocab_size = 128, .block_size = 16, .n_embd = 32, .n_head = 2, .n_layer = 2 };
    var gpt = try transformer.GPT(config).init(allocator, random);
    defer gpt.deinit(allocator);
    gpt.setName("gpt");

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    var token_data: [2 * 8]f32 = undefined;
    for (&token_data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 50));
    const input_tokens = try graph.tensorNDWithData(&.{ 2, 8 }, &token_data, false);
    input_tokens.setName("inputs.token_ids");
    const logits = try gpt.forward(allocator, &graph, input_tokens);
    logits.setName("outputs.logits");

    const json_data = try graph.formatJson(allocator);
    defer allocator.free(json_data);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_data, .{});
    defer parsed.deinit();

    const top = parsed.value.object;
    try std.testing.expectEqualStrings("2.0", top.get("version").?.string);
    try std.testing.expectEqualStrings("gpt", top.get("default_scope").?.string);
    try std.testing.expect(top.get("edges") == null);
    const summary = top.get("summary").?.object;
    try std.testing.expectEqual(@as(i64, 1), summary.get("input_nodes").?.integer);
    // pos_indices + 每层一个 causal_mask
    try std.testing.expectEqual(@as(i64, 3), summary.get("buffer_nodes").?.integer);

    const root = top.get("root").?.object;

    // 根作用域：图输入/输出端口与 gpt 相连
    try VisTestUtil.expectEdgeSet(allocator, root, "", &.{ "@in0->gpt", "gpt->@out0" });
    try VisTestUtil.expectPortRef(root, "", "inputs", 0, "inputs.token_ids");
    try VisTestUtil.expectPortRef(root, "", "outputs", 0, "outputs.logits");

    // gpt：pos_indices 为缓冲区边，而非模型输入；Reshape 折叠进边的 transforms
    try VisTestUtil.expectEdgeSet(allocator, root, "gpt", &.{
        "@in0->wte",
        "pos_indices->wpe buffer",
        "wte->embeddings_sum",
        "wpe->embeddings_sum",
        "embeddings_sum->layers",
        "layers->lm_head",
        "lm_head->@out0",
    });
    const to_head = try VisTestUtil.findEdge(root, "gpt", "layers", "lm_head");
    try std.testing.expectEqualStrings("[2, 8, 32]", to_head.get("shape").?.string);
    try std.testing.expectEqualStrings("[16, 32]", to_head.get("dst_shape").?.string);
    try std.testing.expectEqual(@as(usize, 1), to_head.get("transforms").?.array.items.len);

    // gpt.layers：层间串联，无虚假残差
    try VisTestUtil.expectEdgeSet(allocator, root, "gpt.layers", &.{ "@in0->0", "0->1", "1->ln_f", "ln_f->@out0" });

    // 每个 Block：两条残差边都在 Block 自身作用域内，且端口引用在最近公共作用域中解析
    const block_edges = [_][]const u8{
        "@in0->ln_1",
        "ln_1->attn",
        "@in0->residual_attn skip",
        "attn->residual_attn",
        "residual_attn->ln_2",
        "ln_2->mlp",
        "residual_attn->residual_mlp skip",
        "mlp->residual_mlp",
        "residual_mlp->@out0",
    };
    try VisTestUtil.expectEdgeSet(allocator, root, "gpt.layers.0", &block_edges);
    try VisTestUtil.expectEdgeSet(allocator, root, "gpt.layers.1", &block_edges);
    try VisTestUtil.expectPortRef(root, "gpt.layers.0", "inputs", 0, "gpt.embeddings_sum");
    try VisTestUtil.expectPortRef(root, "gpt.layers.0", "outputs", 0, "gpt.layers.1");
    try VisTestUtil.expectPortRef(root, "gpt.layers.1", "outputs", 0, "gpt.layers.ln_f");

    // 注意力：共享输入扇出到 q/k/v，核心计算封装在 core 子作用域
    try VisTestUtil.expectEdgeSet(allocator, root, "gpt.layers.0.attn", &.{
        "@in0->q_attn",
        "@in0->k_attn",
        "@in0->v_attn",
        "q_attn->core",
        "k_attn->core",
        "v_attn->core",
        "core->c_proj",
        "c_proj->@out0",
    });

    // core：无参数多算子叶子导出算子级局部图，causal_mask 以缓冲区边接入
    const core_mod = VisTestUtil.findModule(root, "gpt.layers.0.attn.core").?;
    try std.testing.expectEqualStrings("ScaledDotProductAttention", core_mod.get("module_type").?.string);
    try std.testing.expectEqual(@as(usize, 3), core_mod.get("ports").?.object.get("inputs").?.array.items.len);
    var mask_edges: usize = 0;
    for (core_mod.get("edges").?.array.items) |e_val| {
        const edge = e_val.object;
        if (std.mem.eql(u8, edge.get("from").?.string, "causal_mask")) {
            try std.testing.expectEqualStrings("buffer", edge.get("kind").?.string);
            mask_edges += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 1), mask_edges);

    // MLP：激活函数作为命名算子节点出现
    try VisTestUtil.expectEdgeSet(allocator, root, "gpt.layers.0.mlp", &.{ "@in0->c_fc", "c_fc->gelu", "gelu->c_proj", "c_proj->@out0" });

    // 带参数的叶子模块 (Linear / RMSNorm) 不导出算子级局部图
    for ([_][]const u8{ "gpt.layers.0.attn.q_attn", "gpt.layers.0.ln_1" }) |leaf_path| {
        const leaf = VisTestUtil.findModule(root, leaf_path).?;
        if (leaf.get("edges")) |e| try std.testing.expectEqual(@as(usize, 0), e.array.items.len);
    }
}

/// 可视化测试辅助：按 model_graph.schema.json 所用的 JSON Schema 关键字子集校验 JSON
/// ($ref, type, const, enum, minimum, required, properties, additionalProperties, items)
const SchemaCheck = struct {
    defs: std.json.ObjectMap,
    /// 首个违规所在的字段名 (未违规时为空)
    field: []const u8 = "",

    fn init(schema: std.json.Value) SchemaCheck {
        return .{ .defs = schema.object.get("$defs").?.object };
    }

    fn resolve(self: SchemaCheck, schema: std.json.ObjectMap) std.json.ObjectMap {
        if (schema.get("$ref")) |r| {
            const prefix = "#/$defs/";
            std.debug.assert(std.mem.startsWith(u8, r.string, prefix));
            return self.defs.get(r.string[prefix.len..]).?.object;
        }
        return schema;
    }

    fn typeMatches(name: []const u8, v: std.json.Value) bool {
        const eql = std.mem.eql;
        return switch (v) {
            .null => eql(u8, name, "null"),
            .bool => eql(u8, name, "boolean"),
            .integer => eql(u8, name, "integer") or eql(u8, name, "number"),
            .float, .number_string => eql(u8, name, "number"),
            .string => eql(u8, name, "string"),
            .array => eql(u8, name, "array"),
            .object => eql(u8, name, "object"),
        };
    }

    fn check(self: *SchemaCheck, schema_in: std.json.ObjectMap, v: std.json.Value) !void {
        const schema = self.resolve(schema_in);
        if (schema.get("type")) |t| {
            const ok = switch (t) {
                .string => |name| typeMatches(name, v),
                .array => |names| blk: {
                    for (names.items) |name| {
                        if (typeMatches(name.string, v)) break :blk true;
                    }
                    break :blk false;
                },
                else => return error.InvalidSchema,
            };
            if (!ok) return error.SchemaTypeMismatch;
        }
        if (schema.get("const")) |c| {
            if (v != .string or !std.mem.eql(u8, c.string, v.string)) return error.SchemaConstMismatch;
        }
        if (schema.get("enum")) |e| {
            if (v != .string) return error.SchemaEnumMismatch;
            for (e.array.items) |allowed| {
                if (std.mem.eql(u8, allowed.string, v.string)) break;
            } else return error.SchemaEnumMismatch;
        }
        if (schema.get("minimum")) |m| {
            if (v == .integer and v.integer < m.integer) return error.SchemaBelowMinimum;
        }
        switch (v) {
            .object => |obj| {
                if (schema.get("required")) |req| {
                    for (req.array.items) |key| {
                        if (obj.get(key.string) == null) {
                            self.field = key.string;
                            return error.SchemaMissingField;
                        }
                    }
                }
                const closed = if (schema.get("additionalProperties")) |ap| ap == .bool and !ap.bool else false;
                const props = schema.get("properties");
                var it = obj.iterator();
                while (it.next()) |entry| {
                    const key = entry.key_ptr.*;
                    if (props) |p| {
                        if (p.object.get(key)) |sub| {
                            self.check(sub.object, entry.value_ptr.*) catch |err| {
                                if (self.field.len == 0) self.field = key;
                                return err;
                            };
                            continue;
                        }
                    }
                    if (closed) {
                        self.field = key;
                        return error.SchemaUnknownField;
                    }
                }
            },
            .array => |arr| {
                if (schema.get("items")) |items| {
                    for (arr.items) |item| try self.check(items.object, item);
                }
            },
            else => {},
        }
    }

    fn run(allocator: std.mem.Allocator, json_text: []const u8, report: bool) !void {
        var schema = try std.json.parseFromSlice(std.json.Value, allocator, visualization.SCHEMA_JSON, .{});
        defer schema.deinit();
        var doc = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
        defer doc.deinit();
        var checker = SchemaCheck.init(schema.value);
        checker.check(schema.value.object, doc.value) catch |err| {
            if (report) std.debug.print("schema violation {s} at field \"{s}\"\n", .{ @errorName(err), checker.field });
            return err;
        };
    }

    /// 校验并返回首个违规错误 (不输出诊断)
    fn validate(allocator: std.mem.Allocator, json_text: []const u8) !void {
        return run(allocator, json_text, false);
    }

    /// 校验导出结果；违规时输出出错字段名
    fn expectConforms(allocator: std.mem.Allocator, json_text: []const u8) !void {
        return run(allocator, json_text, true);
    }
};

test "Model graph JSON export conforms to the published JSON Schema" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(24680);
    const random = prng.random();

    // 1. 多层 GPT：覆盖端口、缓冲区边、折叠变换与残差边
    {
        const config = transformer.GPTConfig{ .vocab_size = 64, .block_size = 8, .n_embd = 16, .n_head = 2, .n_layer = 2 };
        var gpt = try transformer.GPT(config).init(allocator, random);
        defer gpt.deinit(allocator);
        gpt.setName("gpt");

        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();
        var token_data: [2 * 4]f32 = undefined;
        for (&token_data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 50));
        const input_tokens = try graph.tensorNDWithData(&.{ 2, 4 }, &token_data, false);
        input_tokens.setName("inputs.token_ids");
        const logits = try gpt.forward(allocator, &graph, input_tokens);
        logits.setName("outputs.logits");

        const json_data = try graph.formatJson(allocator);
        defer allocator.free(json_data);
        try SchemaCheck.expectConforms(allocator, json_data);
    }

    // 2. 单个 Linear：最小模型
    {
        var linear = try core.Linear.init(allocator, 8, 4, random);
        defer linear.deinit(allocator);
        linear.setName("linear");

        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();
        var x_data: [2 * 8]f32 = undefined;
        for (&x_data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i)) * 0.1;
        const x = try graph.tensorNDWithData(&.{ 2, 8 }, &x_data, false);
        x.setName("inputs.x");
        const y = try linear.forward(allocator, &graph, x);
        y.setName("outputs.y");

        const json_data = try graph.formatJson(allocator);
        defer allocator.free(json_data);
        try SchemaCheck.expectConforms(allocator, json_data);
    }

    // 3. 校验器自身：未声明字段与错误版本均被拒绝
    try std.testing.expectError(error.SchemaConstMismatch, SchemaCheck.validate(allocator,
        \\{"version": "1.0", "summary": {}, "default_scope": "", "root": {}}
    ));
    try std.testing.expectError(error.SchemaUnknownField, SchemaCheck.validate(allocator,
        \\{"version": "2.0", "extra": 1, "summary": {"total_params": 0, "total_bytes": 0, "param_nodes": 0, "input_nodes": 0, "buffer_nodes": 0, "activation_nodes": 0, "custom_init_count": 0, "auto_graph_count": 0, "total_nodes": 0},
        \\ "default_scope": "", "root": {"name": "root", "path": "", "kind": "module", "module_type": "Model", "formula": null, "total_params": 0, "total_bytes": 0, "param_count": 0, "node_count": 0,
        \\ "children": [], "parameters": [], "ops": [], "ports": {"inputs": [], "outputs": []}, "flow_nodes": [], "edges": [], "nodes": []}}
    ));
}

test "NodeKind enum conversions and NodeData typing" {
    try std.testing.expectEqualStrings("Param", NodeKind.Param.asString());
    try std.testing.expectEqualStrings("Input", NodeKind.Input.asString());
    try std.testing.expectEqualStrings("Buffer", NodeKind.Buffer.asString());
    try std.testing.expectEqualStrings("Activation", NodeKind.Activation.asString());

    try std.testing.expectEqual(NodeKind.Param, NodeKind.fromString("Param"));
    try std.testing.expectEqual(NodeKind.Input, NodeKind.fromString("Input"));
    try std.testing.expectEqual(NodeKind.Buffer, NodeKind.fromString("Buffer"));
    try std.testing.expectEqual(NodeKind.Activation, NodeKind.fromString("Activation"));

    try std.testing.expectEqual(NodeKind.Param, NodeKind.fromString("param"));
    try std.testing.expectEqual(NodeKind.Input, NodeKind.fromString("input"));
    try std.testing.expectEqual(NodeKind.Buffer, NodeKind.fromString("buffer"));
    try std.testing.expectEqual(NodeKind.Activation, NodeKind.fromString("activation"));

    try std.testing.expect(NodeKind.fromString("Unknown") == null);
}

test "Visualization enums match the enum lists of the published JSON Schema" {
    const allocator = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, visualization.SCHEMA_JSON, .{});
    defer parsed.deinit();
    const defs = parsed.value.object.get("$defs").?.object;

    const Check = struct {
        fn prop(d: std.json.ObjectMap, def: []const u8, field: []const u8) std.json.ObjectMap {
            return d.get(def).?.object.get("properties").?.object.get(field).?.object;
        }

        /// schema 的 enum 列表与 Zig 枚举的标签逐一对应 (顺序一致)
        fn same(comptime E: type, values: []const std.json.Value) !void {
            const fields = @typeInfo(E).@"enum".fields;
            try std.testing.expectEqual(fields.len, values.len);
            inline for (fields, 0..) |f, i| {
                try std.testing.expectEqualStrings(f.name, values[i].string);
            }
        }
    };

    try Check.same(NodeKind, Check.prop(defs, "TensorNode", "kind").get("enum").?.array.items);
    try Check.same(NodeStatus, Check.prop(defs, "TensorNode", "status").get("enum").?.array.items);
    try Check.same(FlowNodeKind, Check.prop(defs, "FlowNode", "kind").get("enum").?.array.items);
    try Check.same(EdgeKind, Check.prop(defs, "Edge", "kind").get("enum").?.array.items);

    // 参数只可能是 CUSTOM_INIT / AUTO_GRAPH
    const param_status = Check.prop(defs, "ParamEntry", "status").get("enum").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), param_status.len);
    try std.testing.expectEqualStrings(NodeStatus.CUSTOM_INIT.asString(), param_status[0].string);
    try std.testing.expectEqualStrings(NodeStatus.AUTO_GRAPH.asString(), param_status[1].string);

    try std.testing.expectEqualStrings("module", Check.prop(defs, "ModuleNode", "kind").get("const").?.string);
}

test "Comptime reflection supports slice modules, optional bias, and frozen LoRA base weights" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // 1. MoELayer ([]MLP fields: routed_experts and shared_experts)
    var moe = try MoELayer.init(allocator, 4, 8, 3, 1, 2, random);
    defer deinitModel(&moe, allocator);
    const moe_params = try collectParameters(&moe, allocator);
    defer allocator.free(moe_params);
    // gate (w + b = 2) + 3 routed experts (each c_fc.w, c_fc.b, c_proj.w, c_proj.b = 4) + 1 shared expert (4) = 18
    try std.testing.expectEqual(@as(usize, 18), moe_params.len);
    for (moe_params) |p| p.grad[0] = 1.0;
    zeroGradModel(&moe);
    for (moe_params) |p| try std.testing.expectApproxEqAbs(@as(f32, 0.0), p.grad[0], 1e-6);

    // 2. StackedLSTM ([]LSTMCell field: layers)
    var stacked_lstm = try StackedLSTM.init(allocator, 4, 6, 2, random);
    defer deinitModel(&stacked_lstm, allocator);
    const lstm_params = try collectParameters(&stacked_lstm, allocator);
    defer allocator.free(lstm_params);
    // 2 layers * 4 gates * 4 parameters (w_ih/w_hh Linear weights + bias) = 32
    try std.testing.expectEqual(@as(usize, 32), lstm_params.len);

    // 3. ConvTranspose2D (?*Tensor field: bias)
    var deconv = try ConvTranspose2D.init(allocator, 2, 3, 2, 1, 0, true, random);
    defer deinitModel(&deconv, allocator);
    const deconv_params = try collectParameters(&deconv, allocator);
    defer allocator.free(deconv_params);
    try std.testing.expectEqual(@as(usize, 2), deconv_params.len);

    // 4. LoRALinear (frozen weight, trainable lora_a, lora_b, and optional bias)
    var lora = try LoRALinear.initWithBias(allocator, 4, 3, 2, 4.0, true, random);
    defer deinitModel(&lora, allocator);
    const lora_params = try collectParameters(&lora, allocator);
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
        .moe = try MoELayer.init(allocator, 4, 8, 2, 1, 1, random),
        .deconv = try ConvTranspose2D.init(allocator, 2, 2, 2, 1, 0, true, random),
        .lora = try LoRALinear.initWithBias(allocator, 4, 3, 2, 2.0, true, random),
    };
    defer deinitModel(&m1, allocator);
    m1.deconv.bias.?.data[0] = 7.25;
    m1.lora.bias.?.data[1] = -3.5;
    m1.moe.routed_experts[1].c_fc.weight.data[0] = 42.0;

    const path = "test_composite_model.safetensors";
    try saveModel(&m1, std.testing.io, path, allocator);
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    var m2 = CompositeModel{
        .moe = try MoELayer.init(allocator, 4, 8, 2, 1, 1, random),
        .deconv = try ConvTranspose2D.init(allocator, 2, 2, 2, 1, 0, true, random),
        .lora = try LoRALinear.initWithBias(allocator, 4, 3, 2, 2.0, true, random),
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
    var lin = try Linear.init(allocator, 2, 2, random);
    defer lin.deinit(allocator);

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

    var rnn = try RNN.init(allocator, 4, 4, random);
    defer rnn.deinit(allocator);
    rnn.setName("enc_rnn");
    try std.testing.expectEqualStrings("enc_rnn", rnn.getName().?);
    try std.testing.expectEqualStrings("enc_rnn.cell", rnn.cell.getName().?);

    var lstm = try LSTM.init(allocator, 4, 4, random);
    defer lstm.deinit(allocator);
    lstm.setName("enc_lstm");
    try std.testing.expectEqualStrings("enc_lstm", lstm.getName().?);

    var slstm = try StackedLSTM.init(allocator, 4, 4, 2, random);
    defer slstm.deinit(allocator);
    slstm.setName("deep_lstm");
    try std.testing.expectEqualStrings("deep_lstm.layer_0", slstm.layers[0].getName().?);

    var gru = try GRU.init(allocator, 4, 4, random);
    defer gru.deinit(allocator);
    gru.setName("enc_gru");
    try std.testing.expectEqualStrings("enc_gru", gru.getName().?);

    var moe = try MoELayer.init(allocator, 4, 8, 2, 1, 1, random);
    defer moe.deinit(allocator);
    moe.setName("ffn_moe");
    try std.testing.expectEqualStrings("ffn_moe.gate", moe.gate.getName().?);
    try std.testing.expectEqualStrings("ffn_moe.routed_0", moe.routed_experts[0].getName().?);

    var mla = try MLALayer.init(allocator, 8, 2, 4, 4, 2, random);
    defer mla.deinit(allocator);
    mla.setName("attn_mla");
    try std.testing.expectEqualStrings("attn_mla.q_proj", mla.q_proj.getName().?);

    var lora = try LoRALinear.initWithBias(allocator, 4, 4, 2, 4.0, true, random);
    defer lora.deinit(allocator);
    lora.setName("proj_lora");
    try std.testing.expectEqualStrings("proj_lora.lora_a", lora.lora_a.getName().?);

    const x = try graph.tensor(2, 4, true);
    @memset(x.data, 0.2);
    const out_lora = try lora.forward(allocator, &graph, x);
    try std.testing.expectEqualStrings("proj_lora", out_lora.creator.?.scope);
    try std.testing.expectEqualStrings(LoRALinear.formula, graph.inferModuleFormula("proj_lora"));
}

test "Conv2D with stride and padding forward and backward (im2col + sgemm)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(2026);
    const random = prng.random();

    var conv = try Conv2D.initWithConfig(allocator, 1, 2, 3, 2, 1, random);
    defer conv.deinit(allocator);
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

    const out = try conv.forward(allocator, &graph, x);
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
    var moe = try MoELayer.init(allocator, 4, 8, 3, 0, 1, random);
    defer moe.deinit(allocator);

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
    const out_g = try moe.forward(allocator, &graph, x);
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
    const out_e = try moe.forward(allocator, null, x);
    defer out_e.deinit(allocator);
    for (out_g.data, out_e.data) |vg, ve| {
        try std.testing.expectApproxEqAbs(vg, ve, 1e-4);
    }
}


