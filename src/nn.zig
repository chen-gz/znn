const std = @import("std");
const autodiff = @import("autodiff.zig");
const tensor = @import("tensor.zig");
const engine = @import("engine.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;

// ============================================================================
// 领域驱动子模块 (Domain-driven Submodules)
// ============================================================================

pub const core = @import("nn/core.zig");
pub const activations = @import("nn/activations.zig");
pub const normalization = @import("nn/normalization.zig");
pub const recurrent = @import("nn/recurrent.zig");
pub const transformer = @import("nn/transformer.zig");
pub const serialization = @import("nn/serialization.zig");

// ============================================================================
// 门面层导出 (Facade Re-exports) - 确保 100% 向上兼容
// ============================================================================

// 1. 核心与容器 (Core & Containers)
pub const init_mod = core.init_mod;
pub const Nonlinearity = core.Nonlinearity;
pub const calculateGain = core.calculateGain;
pub const InitMethod = core.InitMethod;
pub const InitOptions = core.InitOptions;
pub const normalRandom = core.normalRandom;
pub const initWeights = core.initWeights;
pub const initializeWeights = core.initializeWeights;
pub const createPersistentTensor = core.createPersistentTensor;
pub const freePersistentTensor = core.freePersistentTensor;
pub const Linear = core.Linear;
pub const Conv2D = core.Conv2D;
pub const ConvTranspose2D = core.ConvTranspose2D;
pub const Module = core.Module;
pub const deinitModel = core.deinitModel;
pub const collectParameters = core.collectParameters;
pub const Sequential = core.Sequential;
pub const sequential = core.sequential;
pub const autoSequential = core.autoSequential;
pub const detectNextActivation = core.detectNextActivation;

// 2. 激活函数 (Activations)
pub const ReLU = activations.ReLU;
pub const GELU = activations.GELU;
pub const Sigmoid = activations.Sigmoid;
pub const Tanh = activations.Tanh;
pub const LeakyReLU = activations.LeakyReLU;
pub const SiLU = activations.SiLU;
pub const Swish = activations.Swish;

// 3. 归一化与池化 (Normalization & Pooling)
pub const RMSNorm = normalization.RMSNorm;
pub const LayerNorm = normalization.LayerNorm;
pub const BatchNorm2d = normalization.BatchNorm2d;
pub const Dropout = normalization.Dropout;
pub const AvgPool2D = normalization.AvgPool2D;

// 4. 循环神经网络 (Recurrent Neural Networks)
pub const RNNCell = recurrent.RNNCell;
pub const RNN = recurrent.RNN;
pub const RNNResult = recurrent.RNNResult;
pub const LSTMState = recurrent.LSTMState;
pub const LSTMCell = recurrent.LSTMCell;
pub const LSTM = recurrent.LSTM;
pub const LSTMResult = recurrent.LSTMResult;
pub const StackedLSTM = recurrent.StackedLSTM;
pub const StackedLSTMResult = recurrent.StackedLSTMResult;
pub const GRUCell = recurrent.GRUCell;
pub const GRU = recurrent.GRU;
pub const GRUResult = recurrent.GRUResult;

// 5. Transformer、注意力与生成对齐 (Transformer, Attention & Alignment)
pub const Embedding = transformer.Embedding;
pub const KVCache = transformer.KVCache;
pub const MLP = transformer.MLP;
pub const swigluForward = transformer.swigluForward;
pub const SwiGLU = transformer.SwiGLU;
pub const MoELayer = transformer.MoELayer;
pub const CausalSelfAttention = transformer.CausalSelfAttention;
pub const applyRope1D = transformer.applyRope1D;
pub const MLACache = transformer.MLACache;
pub const MLALayer = transformer.MLALayer;
pub const TransformerBlock = transformer.TransformerBlock;
pub const TransformerDecoder = transformer.TransformerDecoder;
pub const GPTConfig = transformer.GPTConfig;
pub const GPT = transformer.GPT;
pub const LoRALinear = transformer.LoRALinear;
pub const maskedCrossEntropyLoss = transformer.maskedCrossEntropyLoss;
pub const sftCrossEntropyLoss = transformer.sftCrossEntropyLoss;
pub const dpoLoss = transformer.dpoLoss;
pub const computeGroupAdvantages = transformer.computeGroupAdvantages;
pub const computeGRPOLoss = transformer.computeGRPOLoss;
pub const grpoLoss = transformer.grpoLoss;
pub const sampleTopP = transformer.sampleTopP;
pub const sampleTopK = transformer.sampleTopK;

// 6. 权重序列化 (Serialization - Safetensors)
pub const saveModel = serialization.saveModel;
pub const loadModel = serialization.loadModel;

// 7. 引擎执行与评估 (Training & Evaluation Engine)
pub const trainClassificationStep = engine.trainClassificationStep;
pub const evalClassificationStep = engine.evalClassificationStep;
pub const trainClassificationEpoch = engine.trainClassificationEpoch;
pub const evaluateClassification = engine.evaluateClassification;
pub const computeAccuracy = engine.computeAccuracy;
pub const ClassificationStepResult = engine.ClassificationStepResult;
pub const ClassificationEpochResult = engine.ClassificationEpochResult;
pub const trainStep = engine.trainStep;
pub const evalStep = engine.evalStep;
pub const trainEpoch = engine.trainEpoch;
pub const evaluate = engine.evaluate;
pub const StepResult = engine.StepResult;
pub const EpochResult = engine.EpochResult;

// ============================================================================
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

    // DPO Loss test
    const pi_w = [_]f32{-1.2};
    const pi_l = [_]f32{-2.8};
    const ref_w = [_]f32{-1.5};
    const ref_l = [_]f32{-2.0};
    const d_loss = dpoLoss(&pi_w, &pi_l, &ref_w, &ref_l, 0.1);
    try std.testing.expect(d_loss > 0.0);
}

test "LayerNorm forward pass" {
    const allocator = std.testing.allocator;
    var ln = try LayerNorm.init(allocator, 4, 1e-5);
    defer ln.deinit(allocator);

    const x = try tensor.zeros(allocator, &.{ 2, 4 });
    defer tensor.free(allocator, x);
    @memcpy(x.data, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0 });

    const y = try ln.forward(allocator, null, x);
    defer tensor.free(allocator, y);

    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, y.shape.dims[0..2]);
    // Mean of normalized output should be close to 0.0
    var sum: f32 = 0.0;
    for (y.data[0..4]) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sum / 4.0, 1e-4);
}

test "BatchNorm2d forward pass" {
    const allocator = std.testing.allocator;
    var bn = try BatchNorm2d.init(allocator, 2, 1e-5, 0.1);
    defer bn.deinit(allocator);

    const x = try tensor.zeros(allocator, &.{ 2, 2, 2, 2 });
    defer tensor.free(allocator, x);
    for (x.data, 0..) |*p, i| p.* = @as(f32, @floatFromInt(i));

    const y = try bn.forward(allocator, null, x);
    defer tensor.free(allocator, y);

    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 2 }, y.shape.dims[0..4]);
}

test "Dropout and AvgPool2D forward passes" {
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

    // 1. 各层通过干净的 initClean 创建（无随机数，只分配内存）
    var fc_relu = try Linear.initClean(allocator, 100, 100);
    defer fc_relu.deinit(allocator);
    var fc_tanh = try Linear.initClean(allocator, 100, 100);
    defer fc_tanh.deinit(allocator);
    var fc_custom = try Linear.initClean(allocator, 100, 10);
    defer fc_custom.deinit(allocator);

    // 2. 特殊层显式调用 customInit：指定常数偏置 3.14，并随机初始化权重
    fc_custom.customInit(random, .{
        .nonlinearity = .linear,
        .bias_init = .{ .constant = 3.14 },
    });
    // 记录 custom 权重切片的一个样本以验证后续不被 Graph 篡改重写
    const custom_weight_sample = fc_custom.weight.data[0];

    // 3. 在构造/连接期通过各类 Operation 将图自然动态串联起来
    const x = try graph.zeros(&.{ 2, 100 }, false);
    const z1 = try graph.addBias(try graph.matmul(x, fc_relu.weight), fc_relu.bias);
    const a1 = try graph.relu(z1); // 后续接 ReLU

    const z2 = try graph.addBias(try graph.matmul(a1, fc_tanh.weight), fc_tanh.bias);
    const a2 = try graph.tanh(z2); // 后续接 Tanh

    var fc_out = try Linear.initClean(allocator, 10, 2);
    defer fc_out.deinit(allocator);

    const logits = try graph.addBias(try graph.matmul(a2, fc_custom.weight), fc_custom.bias);
    const final_out = try graph.addBias(try graph.matmul(logits, fc_out.weight), fc_out.bias);
    _ = final_out;

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

    // 6. 测试 formatInitReport 能够正常输出并展示 CUSTOM_INIT 状态
    const report_str = try graph.formatInitReport(allocator);
    defer allocator.free(report_str);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "CUSTOM_INIT") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "AUTO_GRAPH") != null);
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





