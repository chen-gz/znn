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
pub const normalRandom = core.normalRandom;
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
