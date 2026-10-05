const root = @import("root.zig");
const testing_init = @import("nn/testing_init.zig");
const VERSION = root.VERSION;
const version = root.version;
const tensor = root.tensor;
const nn = root.nn;
const dataset = root.dataset;
const autodiff = root.autodiff;
const optim = root.optim;
const regression = root.regression;
const manifold = root.manifold;
const cv = root.cv;
const engine = root.engine;
const bench = root.bench;
const TSNE = root.TSNE;
const TSNEOptions = root.TSNEOptions;
const tsne = root.tsne;
const tsneDefault = root.tsneDefault;
const Nonlinearity = root.Nonlinearity;
const calculateGain = root.calculateGain;
const InitMethod = root.InitMethod;
const InitOptions = root.InitOptions;
const initWeights = root.initWeights;
const GenericTensor = root.GenericTensor;
const TensorOf = root.TensorOf;
const FloatTensor = root.FloatTensor;
const DoubleTensor = root.DoubleTensor;
const IntTensor = root.IntTensor;
const LongTensor = root.LongTensor;
const BoolTensor = root.BoolTensor;
const BFloat16Tensor = root.BFloat16Tensor;
const bf16 = root.bf16;
const DType = root.DType;
const SliceRange = root.SliceRange;
const SGDConfig = root.SGDConfig;
const AdamConfig = root.AdamConfig;
const AdamWConfig = root.AdamWConfig;
const CosineScheduler = root.CosineScheduler;
const StepLRScheduler = root.StepLRScheduler;
const LinearWarmupScheduler = root.LinearWarmupScheduler;
const ExponentialLRScheduler = root.ExponentialLRScheduler;
const LRScheduler = root.LRScheduler;
const GradClipConfig = root.GradClipConfig;
const measureTime = root.measureTime;
const ProfileBlock = root.ProfileBlock;
const ScopeTimer = root.ScopeTimer;

test "basic imports and struct definitions" {
    const std = @import("std");
    _ = @import("optim.zig");
    try std.testing.expectEqualStrings("0.2.7", VERSION);
    try std.testing.expectEqual(@as(u32, 0), version.major);
    try std.testing.expectEqual(@as(u32, 2), version.minor);
    try std.testing.expectEqual(@as(u32, 7), version.patch);
    try std.testing.expect(@TypeOf(nn.Linear) == type);
    try std.testing.expect(@TypeOf(nn.SwiGLU) == type);
    try std.testing.expect(@TypeOf(nn.LoRALinear) == type);
    try std.testing.expect(@TypeOf(nn.TransformerDecoder) == fn(comptime usize) type);
    try std.testing.expect(@TypeOf(autodiff.Graph) == type);
    try std.testing.expect(@TypeOf(tensor.Tensor) == type);
    try std.testing.expect(@TypeOf(optim.SGDOptimizer) == type);
    try std.testing.expect(@TypeOf(optim.AdamOptimizer) == type);
    try std.testing.expect(@TypeOf(optim.AdamWOptimizer) == type);
    try std.testing.expect(@TypeOf(optim.StepLRScheduler) == type);
    try std.testing.expect(@TypeOf(optim.LinearWarmupScheduler) == type);
    try std.testing.expect(@TypeOf(optim.ExponentialLRScheduler) == type);
    try std.testing.expect(@TypeOf(optim.LRScheduler) == type);
    try std.testing.expect(@TypeOf(optim.GradClipConfig) == type);
    try std.testing.expect(@TypeOf(dataset.BPETokenizer) == type);
    try std.testing.expect(@TypeOf(nn.LayerNorm) == type);
    try std.testing.expect(@TypeOf(nn.BatchNorm2d) == type);
    try std.testing.expect(@TypeOf(nn.Dropout) == type);
    try std.testing.expect(@TypeOf(nn.AvgPool2D) == type);
    try std.testing.expect(@TypeOf(nn.KVCache) == type);
    try std.testing.expect(@TypeOf(nn.RNNCell) == type);
    try std.testing.expect(@TypeOf(nn.RNN) == type);
    try std.testing.expect(@TypeOf(nn.LSTMCell) == type);
    try std.testing.expect(@TypeOf(nn.LSTM) == type);
    try std.testing.expect(@TypeOf(nn.StackedLSTM) == type);
    try std.testing.expect(@TypeOf(nn.GRUCell) == type);
    try std.testing.expect(@TypeOf(nn.GRU) == type);
    _ = @import("manifold.zig");
    try std.testing.expect(@TypeOf(manifold.TSNE) == type);
    try std.testing.expect(@typeInfo(@TypeOf(tsne)) == .@"fn");
}

test "measureTime utility" {
    const std = @import("std");
    const helper = struct {
        fn add(a: i32, b: i32) i32 {
            var i: i32 = 0;
            while (i < 1000) : (i += 1) {
                std.mem.doNotOptimizeAway(i);
            }
            return a + b;
        }
    };
    const timed = try measureTime(helper.add, .{ 5, 10 });
    try std.testing.expectEqual(@as(i32, 15), timed.result);
    try std.testing.expect(timed.elapsed_ns > 0);
}


test "ProfileBlock utility" {
    const p = ProfileBlock.start("test_block");
    defer p.end();
}

test "ScopeTimer utility" {
    const std = @import("std");
    var elapsed: u64 = 0;
    {
        const t = ScopeTimer.start(&elapsed);
        defer t.end();
        var i: i32 = 0;
        while (i < 1000) : (i += 1) {
            std.mem.doNotOptimizeAway(i);
        }
    }
    try std.testing.expect(elapsed > 0);
}

test "Tensor ND reshape and transpose autograd" {
    const std = @import("std");

    const arena = std.testing.allocator;
    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    // Create a 2x3 tensor
    const A = try graph.tensorND(&.{2, 3}, true);
    A.data[0] = 1.0; A.data[1] = 2.0; A.data[2] = 3.0;
    A.data[3] = 4.0; A.data[4] = 5.0; A.data[5] = 6.0;

    // Transpose it to 3x2
    const B = try graph.transposeND(A, 0, 1);
    try std.testing.expectEqualSlices(usize, &.{3, 2}, B.shape.dims[0..B.shape.len]);
    try std.testing.expectEqual(@as(f32, 1.0), B.data[0]); // A[0,0]
    try std.testing.expectEqual(@as(f32, 4.0), B.data[1]); // A[1,0]
    try std.testing.expectEqual(@as(f32, 2.0), B.data[2]); // A[0,1]
    try std.testing.expectEqual(@as(f32, 5.0), B.data[3]); // A[1,1]

    // Reshape it to 1x6
    const C = try graph.reshape(B, &.{1, 6});
    try std.testing.expectEqualSlices(usize, &.{1, 6}, C.shape.dims[0..C.shape.len]);

    // Let's set some gradients in C.grad and backward
    C.grad[0] = 10.0;
    C.grad[1] = 20.0;
    C.grad[2] = 30.0;
    C.grad[3] = 40.0;
    C.grad[4] = 50.0;
    C.grad[5] = 60.0;

    // Run backward on C (usually we call graph.backward(loss), but here we manually backward C's creator)
    if (C.creator) |op| {
        try op.backward();
    }
    if (B.creator) |op| {
        try op.backward();
    }

    // Check A.grad
    try std.testing.expectEqual(@as(f32, 10.0), A.grad[0]); // A[0,0]
    try std.testing.expectEqual(@as(f32, 30.0), A.grad[1]); // A[0,1]
    try std.testing.expectEqual(@as(f32, 50.0), A.grad[2]); // A[0,2]
    try std.testing.expectEqual(@as(f32, 20.0), A.grad[3]); // A[1,0]
    try std.testing.expectEqual(@as(f32, 40.0), A.grad[4]); // A[1,1]
    try std.testing.expectEqual(@as(f32, 60.0), A.grad[5]); // A[1,2]
}

test "Tensor matrix multiplication and bias addition autograd example" {
    const std = @import("std");

    const arena = std.testing.allocator;
    // 1. Initialize the computation graph
    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    // 2. Create input tensor A (2x3) and weight B (3x2)
    // A represents a batch of 2 samples with 3 features each
    const A = try graph.array(&.{2, 3}, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, true);

    // B represents weights mapping 3 features to 2 outputs
    const B = try graph.array(&.{3, 2}, &[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, true);

    // 3. Matrix Multiplication: C = A * B (resulting in 2x2)
    const C = try graph.matmul(A, B);
    try std.testing.expectEqualSlices(usize, &.{2, 2}, C.shape.dims[0..C.shape.len]);

    // Verify C values:
    // C[0, 0] = 1.0*0.1 + 2.0*0.3 + 3.0*0.5 = 2.2
    // C[0, 1] = 1.0*0.2 + 2.0*0.4 + 3.0*0.6 = 2.8
    // C[1, 0] = 4.0*0.1 + 5.0*0.3 + 6.0*0.5 = 4.9
    // C[1, 1] = 4.0*0.2 + 5.0*0.4 + 6.0*0.6 = 6.4
    try std.testing.expectApproxEqAbs(@as(f32, 2.2), C.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.8), C.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.9), C.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.4), C.data[3], 1e-5);

    // 4. Bias Addition: D = C + bias (1x2 bias broadcasted to 2x2 C)
    const bias = try graph.array(&.{1, 2}, &[_]f32{ 0.5, 1.0 }, true);

    const D = try graph.addBias(C, bias);
    try std.testing.expectEqualSlices(usize, &.{2, 2}, D.shape.dims[0..D.shape.len]);

    // D[0, 0] = C[0, 0] + bias[0] = 2.2 + 0.5 = 2.7
    // D[0, 1] = C[0, 1] + bias[1] = 2.8 + 1.0 = 3.8
    // D[1, 0] = C[1, 0] + bias[0] = 4.9 + 0.5 = 5.4
    // D[1, 1] = C[1, 1] + bias[1] = 6.4 + 1.0 = 7.4
    try std.testing.expectApproxEqAbs(@as(f32, 2.7), D.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.8), D.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.4), D.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 7.4), D.data[3], 1e-5);

    // 5. Backpropagation: compute gradients dD/dA, dD/dB, dD/dbias
    @memset(D.grad, 1.0);

    try graph.backward(D);

    // Verify bias gradient: dD/dbias = sum over rows of D.grad
    try std.testing.expectEqual(@as(f32, 2.0), bias.grad[0]);
    try std.testing.expectEqual(@as(f32, 2.0), bias.grad[1]);

    // Verify weight gradient: dD/dB = A^T * D.grad
    try std.testing.expectEqual(@as(f32, 5.0), B.grad[0]);
    try std.testing.expectEqual(@as(f32, 5.0), B.grad[1]);
    try std.testing.expectEqual(@as(f32, 7.0), B.grad[2]);

    // Verify input gradient: dD/dA = D.grad * B^T
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), A.grad[0], 1e-5);
}

test "Conv2D autograd" {
    const std = @import("std");
    const arena = std.testing.allocator;
    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const A = try graph.array(&.{ 1, 1, 3, 3 }, &[_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    }, true);

    const W = try graph.array(&.{ 1, 1, 2, 2 }, &[_]f32{
        1.0, 0.0,
        0.0, 1.0,
    }, true);

    const bias = try graph.array(&.{1}, &[_]f32{0.5}, true);

    const C = try graph.conv2d(A, W, bias, .{});
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2, 2 }, C.shape.dims[0..C.shape.len]);

    try std.testing.expectApproxEqAbs(@as(f32, 6.5), C.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.5), C.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12.5), C.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 14.5), C.data[3], 1e-5);

    @memset(C.grad, 1.0);
    try graph.backward(C);

    // Verify bias grad
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), bias.grad[0], 1e-5);

    // Verify weight grad
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), W.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), W.grad[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), W.grad[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 28.0), W.grad[3], 1e-5);

    // Verify input grad
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[0], 1e-5); // A[0,0]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[1], 1e-5); // A[0,1]
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), A.grad[2], 1e-5); // A[0,2]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[3], 1e-5); // A[1,0]
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[4], 1e-5); // A[1,1]
}

test "MaxPool2D autograd" {
    const std = @import("std");
    const arena = std.testing.allocator;
    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const A = try graph.array(&.{ 1, 1, 4, 4 }, &[_]f32{
        1.0, 2.0, 5.0, 3.0,
        4.0, 3.0, 0.0, 2.0,
        8.0, 7.0, 1.0, 2.0,
        6.0, 5.0, 3.0, 4.0,
    }, true);

    const C = try graph.maxpool2d(A, 2, .{});
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2, 2 }, C.shape.dims[0..C.shape.len]);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), C.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), C.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), C.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), C.data[3], 1e-5);

    @memset(C.grad, 1.0);
    try graph.backward(C);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[4], 1e-5); // A[1,0] (4.0)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[2], 1e-5); // A[0,2] (5.0)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[8], 1e-5); // A[2,0] (8.0)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[15], 1e-5); // A[3,3] (4.0)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), A.grad[0], 1e-5);  // A[0,0]
}

test "MaxPool1D and AvgPool1D autograd" {
    const std = @import("std");
    const arena = std.testing.allocator;
    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const A = try graph.array(&.{ 1, 1, 4 }, &[_]f32{ 1.0, 4.0, 2.0, 6.0 }, true);
    const mp = try graph.maxpool1d(A, 2, .{});
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2 }, mp.shape.dims[0..mp.shape.len]);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), mp.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), mp.data[1], 1e-5);

    @memset(mp.grad, 1.0);
    try graph.backward(mp);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), A.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), A.grad[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[3], 1e-5);

    const B = try graph.array(&.{ 1, 1, 4 }, &[_]f32{ 1.0, 3.0, 2.0, 6.0 }, true);
    const ap = try graph.avgpool1d(B, 2, .{});
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2 }, ap.shape.dims[0..ap.shape.len]);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), ap.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), ap.data[1], 1e-5);

    @memset(ap.grad, 1.0);
    try graph.backward(ap);
    for (B.grad) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.5), g, 1e-5);
    }
}

test "AdaptiveAvgPool1D and AdaptiveAvgPool2D autograd" {
    const std = @import("std");
    const arena = std.testing.allocator;
    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const A = try graph.array(&.{ 1, 1, 4 }, &[_]f32{ 1.0, 3.0, 2.0, 6.0 }, true);
    const aap1 = try graph.adaptiveAvgPool1d(A, 2);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2 }, aap1.shape.dims[0..aap1.shape.len]);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), aap1.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), aap1.data[1], 1e-5);

    @memset(aap1.grad, 1.0);
    try graph.backward(aap1);
    for (A.grad) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.5), g, 1e-5);
    }

    const B = try graph.array(&.{ 1, 1, 4, 4 }, &[_]f32{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    }, true);
    const aap2 = try graph.adaptiveAvgPool2d(B, .{ 1, 1 });
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1, 1 }, aap2.shape.dims[0..aap2.shape.len]);
    try std.testing.expectApproxEqAbs(@as(f32, 8.5), aap2.data[0], 1e-5);

    @memset(aap2.grad, 1.0);
    try graph.backward(aap2);
    for (B.grad) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 16.0), g, 1e-5);
    }
}

test "Sigmoid and Tanh autograd" {
    const std = @import("std");
    const arena = std.testing.allocator;
    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const A = try graph.array(&.{ 1, 2 }, &[_]f32{ 0.0, 2.0 }, true);

    const sig = try graph.sigmoid(A);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sig.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.880797), sig.data[1], 1e-5);

    @memset(sig.grad, 1.0);
    try graph.backward(sig);

    try std.testing.expectApproxEqAbs(@as(f32, 0.25), A.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1049935), A.grad[1], 1e-5);

    const t = try graph.tanh(A);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), t.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9640275), t.data[1], 1e-5);
}

test "LeakyReLU autograd" {
    const std = @import("std");
    const arena = std.testing.allocator;
    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const A = try graph.array(&.{ 1, 2 }, &[_]f32{ -2.0, 3.0 }, true);
    const C = try graph.leakyRelu(A, 0.2);

    try std.testing.expectApproxEqAbs(@as(f32, -0.4), C.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), C.data[1], 1e-5);

    @memset(C.grad, 1.0);
    try graph.backward(C);

    try std.testing.expectApproxEqAbs(@as(f32, 0.2), A.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[1], 1e-5);
}

test "BCEWithLogitsLoss autograd" {
    const std = @import("std");
    const arena = std.testing.allocator;
    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    const logits = try graph.array(&.{ 1, 2 }, &[_]f32{ 0.0, 2.0 }, true);
    const targets = try graph.array(&.{ 1, 2 }, &[_]f32{ 1.0, 0.0 }, false);

    const loss = try graph.bceWithLogitsLoss(logits, targets);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4100375), loss.data[0], 1e-4);

    @memset(loss.grad, 1.0);
    try graph.backward(loss);

    try std.testing.expectApproxEqAbs(@as(f32, -0.25), logits.grad[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4403985), logits.grad[1], 1e-4);
}

test "AdamOptimizer model parameter updates" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(42);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    var opt = try optim.AdamOptimizer.init(allocator, &linear, .{
        .lr = 0.01,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .eps = 1e-8,
    });
    defer opt.deinit();

    linear.weight.grad[0] = 1.0;
    linear.weight.grad[1] = -1.0;

    const w0_before = linear.weight.data[0];
    opt.step();
    const w0_after = linear.weight.data[0];

    try std.testing.expect(w0_after < w0_before);
}

test "L2Loss and RidgeLoss autograd" {
    const std = @import("std");
    const arena = std.testing.allocator;
    var graph = autodiff.Graph.init(arena);
    defer graph.deinit();

    // Weight tensor: [2.0, -3.0]
    const W = try graph.array(&.{ 2, 1 }, &[_]f32{ 2.0, -3.0 }, true);
    const lambda: f32 = 0.5;

    // L2 Loss: 0.5 * lambda * (2^2 + (-3)^2) = 0.5 * 0.5 * (4 + 9) = 3.25
    const l2 = try graph.l2Loss(W, lambda);
    try std.testing.expectApproxEqAbs(@as(f32, 3.25), l2.data[0], 1e-5);

    l2.grad[0] = 1.0;
    try graph.backward(l2);

    // Gradient dL/dW = lambda * W = 0.5 * [2.0, -3.0] = [1.0, -1.5]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), W.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), W.grad[1], 1e-5);
}

test "solveLinearSystem Gauss-Jordan elimination" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    // System:
    // 2*x0 + 1*x1 = 5
    // 1*x0 + 3*x1 = 10
    // Exact solution: x0 = 1, x1 = 3
    const A = [_]f32{
        2.0, 1.0,
        1.0, 3.0,
    };
    const b = [_]f32{ 5.0, 10.0 };
    var x = [_]f32{ 0.0, 0.0 };

    try tensor.solveLinearSystem(allocator, &A, &b, 2, &x);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), x[1], 1e-5);
}

test "solveRidgeAnalytical convergence check" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    // 1D test: y = 2.0 * x + 1.0 with zero noise
    const x = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    const y = [_]f32{ -3.0, -1.0, 1.0, 3.0, 5.0 };
    var w = [_]f32{0.0};
    var b: f32 = 0.0;

    // With lambda = 0 (OLS), w = 2.0, b = 1.0
    try tensor.solveRidgeAnalytical(allocator, &x, &y, 5, 1, 0.0, &w, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), w[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), b, 1e-5);

    // With lambda = 10.0 (Sum of dx^2 = 4 + 1 + 0 + 1 + 4 = 10):
    // w = 20 / (10 + 10) = 1.0, b = 1.0 - 1.0 * 0.0 = 1.0
    try tensor.solveRidgeAnalytical(allocator, &x, &y, 5, 1, 10.0, &w, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), w[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), b, 1e-5);
}

test "L1Loss and LassoLoss autograd" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const w = try graph.tensorWithData(1, 3, &[_]f32{ -2.0, 0.0, 3.0 }, true);
    const l1_loss = try graph.l1Loss(w, 2.0);

    // Forward: 2.0 * (| -2 | + | 0 | + | 3 |) = 2.0 * 5.0 = 10.0
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), l1_loss.data[0], 1e-5);

    try graph.backward(l1_loss);

    // Gradients: 2.0 * sign(w) = [-2.0, 0.0, 2.0]
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), w.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), w.grad[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), w.grad[2], 1e-5);
}

test "regression module Ridge, Lasso, and ElasticNet models" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    const x = [_]f32{
        -2.0,  1.0,
        -1.0, -1.0,
         0.0,  0.0,
         1.0, -1.0,
         2.0,  1.0,
    };
    // y = 2.0 * x0 + 0.0 * x1 + 1.0
    const y = [_]f32{ -3.0, -1.0, 1.0, 3.0, 5.0 };

    var ridge = try regression.solveRidge(allocator, &x, &y, 5, 2, 0.1);
    defer ridge.deinit();
    try std.testing.expect(@abs(ridge.intercept - 1.0) < 0.1);
    try std.testing.expect(@abs(ridge.weights[0] - 2.0) < 0.2);

    var lasso = try regression.solveLasso(allocator, &x, &y, 5, 2, 0.05, 500, 1e-5);
    defer lasso.deinit();
    try std.testing.expect(@abs(lasso.intercept - 1.0) < 0.1);
    try std.testing.expect(@abs(lasso.weights[0] - 2.0) < 0.2);

    var enet = try regression.solveElasticNet(allocator, &x, &y, 5, 2, 0.05, 0.5, 500, 1e-5);
    defer enet.deinit();
    try std.testing.expect(@abs(enet.intercept - 1.0) < 0.1);
    try std.testing.expect(@abs(enet.weights[0] - 2.0) < 0.2);
}

test "regression module solveAnalytical" {
    const std = @import("std");
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y = [_]f32{ 3.0, 5.0, 7.0, 9.0, 11.0 }; // y = 2x + 1

    const res = regression.solveAnalytical(&x, &y);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), res.w, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), res.b, 1e-5);
}


test "cross_validation module searchLasso" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    const N: usize = 20;
    const P: usize = 2;

    var X: [N * P]f32 = undefined;
    var y: [N]f32 = undefined;

    for (0..N) |i| {
        const fi = @as(f32, @floatFromInt(i));
        X[i * P + 0] = fi;
        X[i * P + 1] = fi * 0.5;
        y[i] = 2.0 * fi + 1.0;
    }

    var cv_search = cv.CrossValidationGridSearch.init(allocator, 5);
    defer cv_search.deinit();

    const alphas = [_]f32{ 0.001, 0.01, 0.1, 1.0, 10.0 };
    try cv_search.searchLasso(&X, &y, N, P, &alphas, 42);

    try std.testing.expectEqual(@as(usize, 5), cv_search.results.items.len);
    try std.testing.expect(cv_search.getBestMinAlpha() <= 0.1);
}

test "SiLU and Mul autograd in Graph" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const A = try graph.tensor(2, 2, true);
    const B = try graph.tensor(2, 2, true);
    @memcpy(A.data, &[_]f32{ 0.0, 1.0, -1.0, 2.0 });
    @memcpy(B.data, &[_]f32{ 2.0, 3.0, 4.0, 5.0 });

    // C = silu(A)
    const C = try graph.silu(A);
    // D = C * B (element-wise mul)
    const D = try graph.mul(C, B);

    try graph.forward();

    // Check C[0] = 0.0 * sig(0) = 0.0, D[0] = 0.0 * 2.0 = 0.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), C.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), D.data[0], 1e-5);

    // C[1] = 1.0 / (1 + e^-1) = 0.73105858, D[1] = 0.73105858 * 3.0 = 2.1931757
    try std.testing.expectApproxEqAbs(@as(f32, 0.73105858 * 3.0), D.data[1], 1e-4);

    @memset(D.grad, 1.0);
    try graph.backward(D);

    // dD/dB = C
    for (B.grad, C.data) |b_g, c_val| {
        try std.testing.expectApproxEqAbs(c_val, b_g, 1e-5);
    }
    // dD/dA = B * d(silu(A))
    // for i=0: A=0 -> sig=0.5 -> d_silu = 0.5 * (1 + 0) = 0.5 -> grad = 2.0 * 0.5 = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[0], 1e-5);
}

test "SwiGLU forward operator" {
    const std = @import("std");
    const allocator = std.testing.allocator;
    var graph = autodiff.Graph.initNoGrad(allocator);
    defer graph.deinit();

    const gate = try graph.tensorNDWithData(&.{ 1, 2 }, &[_]f32{ 0.0, 2.0 }, false);
    const up = try graph.tensorNDWithData(&.{ 1, 2 }, &[_]f32{ 3.0, 4.0 }, false);
    const act = try graph.silu(gate);
    const out = try graph.mul(act, up);
    // out[0] = (0 * sig(0)) * 3 = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.data[0], 1e-5);
    // out[1] = (2 * sig(2)) * 4 = 2 * (1 / (1 + e^-2)) * 4 = 8 * 0.880797 = 7.046376
    try std.testing.expectApproxEqAbs(@as(f32, 7.046376), out.data[1], 1e-4);
}

test "End-to-End LLM Pipeline integration demo" {
    const std = @import("std");
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(2026);
    const random = prng.random();

    // 1. BPETokenizer test
    var tokenizer = try dataset.BPETokenizer.init(allocator);
    defer tokenizer.deinit();
    try tokenizer.addMerge("Z", "i", 0);
    try tokenizer.addMerge("Zi", "g", 1);

    const encoded = try tokenizer.encode(allocator, "Zig");
    defer allocator.free(encoded);
    try std.testing.expectEqual(@as(usize, 1), encoded.len);

    const decoded = try tokenizer.decode(allocator, encoded);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("Zig", decoded);

    // 2. SwiGLU MLP
    var swiglu = try nn.SwiGLU.init(allocator, 8, 16);
    try testing_init.initFromOnes(&swiglu, allocator, random, &.{ 2, 8 });
    defer nn.deinitModel(&swiglu, allocator);

    // 3. AdamW Optimizer with Cosine Scheduler
    var opt = try optim.AdamWOptimizer.init(allocator, &swiglu, .{
        .lr = 1e-3,
        .weight_decay = 0.01,
    });
    defer opt.deinit();

    const sched = optim.CosineScheduler.init(1e-3, 1e-5, 5, 20);

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x = try graph.tensorND(&.{ 2, 4, 8 }, true);
    @memset(x.data, 0.1);

    const out = try swiglu.forward(&graph, x);
    try std.testing.expectEqualSlices(usize, &.{ 2, 4, 8 }, out.shape.dims[0..out.shape.len]);

    @memset(out.grad, 1.0);
    try graph.backward(out);

    _ = optim.clipGradNorm(opt.params, 1.0);
    const current_lr = sched.getLR(1);
    opt.stepWithLR(current_lr);

    // 4. Sampling test
    const mock_logits = [_]f32{ 0.1, 0.4, 2.5, 0.2, 0.8 };
    const sampled_top_p = try nn.sampleTopP(&mock_logits, 5, 0.7, 0.9, random, allocator);
    try std.testing.expect(sampled_top_p < 5);

    const sampled_top_k = try nn.sampleTopK(&mock_logits, 5, 0.7, 2, random, allocator);
    try std.testing.expect(sampled_top_k < 5);
}

test "Tensor concat and split autograd" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    // 1. Test Concat on dim 1 for 2D tensors [2, 2] and [2, 3] -> [2, 5]
    const A = try graph.tensorND(&.{ 2, 2 }, true);
    const B = try graph.tensorND(&.{ 2, 3 }, true);
    @memcpy(A.data, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    @memcpy(B.data, &[_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 });

    const C = try graph.concat(&.{ A, B }, 1);
    try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, C.shape.dims[0..C.shape.len]);
    try std.testing.expectEqual(@as(f32, 1.0), C.data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), C.data[1]);
    try std.testing.expectEqual(@as(f32, 10.0), C.data[2]);
    try std.testing.expectEqual(@as(f32, 30.0), C.data[4]);
    try std.testing.expectEqual(@as(f32, 3.0), C.data[5]);
    try std.testing.expectEqual(@as(f32, 60.0), C.data[9]);

    // Backward on Concat
    @memset(C.grad, 2.0);
    try graph.backwardWithGrad(C);
    for (A.grad) |g| {
        try std.testing.expectEqual(@as(f32, 2.0), g);
    }
    for (B.grad) |g| {
        try std.testing.expectEqual(@as(f32, 2.0), g);
    }

    // 2. Test Split on dim 2 for 3D tensor [1, 2, 6] -> 3 x [1, 2, 2]
    var graph2 = autodiff.Graph.init(allocator);
    defer graph2.deinit();

    const X = try graph2.tensorND(&.{ 1, 2, 6 }, true);
    for (0..12) |i| {
        X.data[i] = @as(f32, @floatFromInt(i + 1));
    }

    const splits = try graph2.split(X, 3, 2);
    try std.testing.expectEqual(@as(usize, 3), splits.len);
    for (splits) |sp| {
        try std.testing.expectEqualSlices(usize, &.{ 1, 2, 2 }, sp.shape.dims[0..sp.shape.len]);
    }

    // splits[0]: row0 = [1, 2], row1 = [7, 8]
    try std.testing.expectEqual(@as(f32, 1.0), splits[0].data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), splits[0].data[1]);
    try std.testing.expectEqual(@as(f32, 7.0), splits[0].data[2]);
    try std.testing.expectEqual(@as(f32, 8.0), splits[0].data[3]);

    // splits[1]: row0 = [3, 4], row1 = [9, 10]
    try std.testing.expectEqual(@as(f32, 3.0), splits[1].data[0]);
    try std.testing.expectEqual(@as(f32, 4.0), splits[1].data[1]);

    // splits[2]: row0 = [5, 6], row1 = [11, 12]
    try std.testing.expectEqual(@as(f32, 5.0), splits[2].data[0]);
    try std.testing.expectEqual(@as(f32, 6.0), splits[2].data[1]);

    // Backward on Split: splits[0] grad 1.0, splits[1] grad 2.0, splits[2] grad 3.0
    @memset(splits[0].grad, 1.0);
    @memset(splits[1].grad, 2.0);
    @memset(splits[2].grad, 3.0);

    // Call backward on each split's creator or through graph
    if (splits[0].creator) |op| {
        try op.backward();
    }

    try std.testing.expectEqual(@as(f32, 1.0), X.grad[0]); // [0, 0, 0] in splits[0]
    try std.testing.expectEqual(@as(f32, 1.0), X.grad[1]); // [0, 0, 1] in splits[0]
    try std.testing.expectEqual(@as(f32, 2.0), X.grad[2]); // [0, 0, 2] in splits[1]
    try std.testing.expectEqual(@as(f32, 2.0), X.grad[3]); // [0, 0, 3] in splits[1]
    try std.testing.expectEqual(@as(f32, 3.0), X.grad[4]); // [0, 0, 4] in splits[2]
    try std.testing.expectEqual(@as(f32, 3.0), X.grad[5]); // [0, 0, 5] in splits[2]
}

test "GQA CausalSelfAttention and KVCache forwardInference" {
    const std = @import("std");
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(123);
    const random = prng.random();

    const n_embd: usize = 16;
    const n_head: usize = 4;
    const num_kv_heads: usize = 2; // GQA: 4 query heads, 2 KV heads (groups = 2)
    const head_dim = n_embd / n_head; // 4

    var gqa_attn = try nn.CausalSelfAttention.initGQA(allocator, n_embd, n_head, num_kv_heads);
    try testing_init.initFromOnes(&gqa_attn, allocator, random, &.{ 1, 2, n_embd });
    defer nn.deinitModel(&gqa_attn, allocator);

    // 1. Test Autograd Forward and Backward with GQA
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const B: usize = 2;
    const T: usize = 3;
    const x_node = try graph.tensorND(&.{ B, T, n_embd }, true);
    for (x_node.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 10)) * 0.1;
    }

    const y = try gqa_attn.forward(&graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{ B, T, n_embd }, y.shape.dims[0..y.shape.len]);

    @memset(y.grad, 1.0);
    try graph.backward(y);

    var q_grad_sum: f32 = 0.0;
    for (gqa_attn.q_attn.weight.grad) |g| q_grad_sum += @abs(g);
    var k_grad_sum: f32 = 0.0;
    for (gqa_attn.k_attn.weight.grad) |g| k_grad_sum += @abs(g);
    var v_grad_sum: f32 = 0.0;
    for (gqa_attn.v_attn.weight.grad) |g| v_grad_sum += @abs(g);

    try std.testing.expect(q_grad_sum > 0.0);
    try std.testing.expect(k_grad_sum > 0.0);
    try std.testing.expect(v_grad_sum > 0.0);

    // 2. Test Eager Forward
    const x_eager = try tensor.zeros(allocator, &.{ B, T, n_embd });
    defer tensor.free(allocator, x_eager);
    @memcpy(x_eager.data, x_node.data);

    var y_eager_graph = autodiff.Graph.initNoGrad(allocator);
    defer y_eager_graph.deinit();
    const y_eager = try gqa_attn.forward(&y_eager_graph, x_eager);
    try std.testing.expectEqualSlices(usize, &.{ B, T, n_embd }, y_eager.shape.dims[0..y_eager.shape.len]);

    // 3. Test KVCache forwardInference (3 autoregressive steps)
    const max_seq_len: usize = 10;
    var cache = try nn.KVCache.init(allocator, 1, num_kv_heads, max_seq_len, head_dim);
    defer cache.deinit(allocator);

    for (0..3) |step| {
        const token_emb = try tensor.zeros(allocator, &.{ 1, 1, n_embd });
        defer tensor.free(allocator, token_emb);
        for (token_emb.data, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(step + i)) * 0.05;
        }

        const out_step = try gqa_attn.forwardInference(allocator, token_emb, &cache);
        defer tensor.free(allocator, out_step);

        try std.testing.expectEqualSlices(usize, &.{ 1, 1, n_embd }, out_step.shape.dims[0..out_step.shape.len]);
        try std.testing.expectEqual(@as(usize, step + 1), cache.curr_len);
    }
}

test "GRPO group advantages and loss" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    // 1. Test computeGroupAdvantages
    const rewards = [_]f32{ 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0 };
    const advs = try nn.computeGroupAdvantages(allocator, &rewards, 4, 1e-6);
    defer allocator.free(advs);

    try std.testing.expectEqual(@as(usize, 8), advs.len);

    // Group 1 mean = 2.5, std = sqrt(1.25) ~ 1.118034
    // adv[0] < adv[1] < adv[2] < adv[3]
    try std.testing.expect(advs[0] < 0.0);
    try std.testing.expect(advs[1] < 0.0);
    try std.testing.expect(advs[2] > 0.0);
    try std.testing.expect(advs[3] > 0.0);
    var group1_sum: f32 = 0.0;
    for (advs[0..4]) |a| group1_sum += a;
    try std.testing.expect(@abs(group1_sum) < 1e-5);

    // Group 2 mean = 25.0, sum of normalized should also be ~0
    var group2_sum: f32 = 0.0;
    for (advs[4..8]) |a| group2_sum += a;
    try std.testing.expect(@abs(group2_sum) < 1e-5);

    // 2. Test computeGRPOLoss
    const old_logps = [_]f32{ -1.0, -1.5, -2.0, -0.5 };
    const new_logps = [_]f32{ -0.9, -1.4, -2.1, -0.6 };
    const sample_advs = [_]f32{ 1.0, 0.5, -0.5, -1.0 };
    const ref_logps = [_]f32{ -1.0, -1.5, -2.0, -0.5 };

    const loss_eval = nn.computeGRPOLoss(&old_logps, &new_logps, &sample_advs, &ref_logps, 0.05, 0.2);
    try std.testing.expect(!std.math.isNan(loss_eval));

    // 3. Test grpoLoss with autograd and verify gradient direction
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const old_t = try graph.tensorND(&.{4}, false);
    @memcpy(old_t.data, &old_logps);

    const new_t = try graph.tensorND(&.{4}, true);
    @memcpy(new_t.data, &new_logps);
    @memset(new_t.grad, 0.0);

    const loss_val = nn.grpoLoss(old_t, new_t, &sample_advs, &ref_logps, 0.05, 0.2);
    try std.testing.expectApproxEqAbs(loss_eval, loss_val, 1e-5);

    // Verify finite difference gradient check on new_logps
    const eps: f32 = 1e-3;
    for (0..4) |i| {
        var perturbed_plus = new_logps;
        perturbed_plus[i] += eps;
        const loss_plus = nn.computeGRPOLoss(&old_logps, &perturbed_plus, &sample_advs, &ref_logps, 0.05, 0.2);

        var perturbed_minus = new_logps;
        perturbed_minus[i] -= eps;
        const loss_minus = nn.computeGRPOLoss(&old_logps, &perturbed_minus, &sample_advs, &ref_logps, 0.05, 0.2);

        const numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);
        try std.testing.expectApproxEqAbs(numerical_grad, new_t.grad[i], 1e-3);
    }

    // 4. Test Graph-integrated grpoLossGraph with graph.backward
    const new_t_graph = try graph.tensorNDWithData(&.{4}, &new_logps, true);
    const scaled_new = try graph.mulScalar(new_t_graph, 1.0);
    const g_grpo_loss = try nn.grpoLossGraph(&graph, old_t, scaled_new, &sample_advs, &ref_logps, 0.05, 0.2);
    try std.testing.expectApproxEqAbs(loss_eval, g_grpo_loss.data[0], 1e-5);
    try graph.backward(g_grpo_loss);
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(new_t.grad[i], new_t_graph.grad[i], 1e-5);
    }
}

test "MoELayer Top-K routing and autograd" {
    const std = @import("std");
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const dim: usize = 8;
    const hidden_dim: usize = 16;
    const num_routed_experts: usize = 4;
    const num_shared_experts: usize = 1;
    const top_k: usize = 2;

    var moe = try nn.MoELayer.init(
        allocator,
        dim,
        hidden_dim,
        num_routed_experts,
        num_shared_experts,
        top_k,
    );
    try testing_init.initFromOnes(&moe, allocator, random, &.{ 2, dim });
    defer nn.deinitModel(&moe, allocator);

    // 1. Eager mode test on 3D input [2, 3, 8]
    const x_eager = try tensor.zeros(allocator, &.{ 2, 3, dim });
    defer tensor.free(allocator, x_eager);
    for (x_eager.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 5)) * 0.2;
    }

    var y_eager_graph = autodiff.Graph.initNoGrad(allocator);
    defer y_eager_graph.deinit();
    const y_eager = try moe.forward(&y_eager_graph, x_eager);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, dim }, y_eager.shape.dims[0..y_eager.shape.len]);

    // 2. Autograd Graph mode test on 3D input [2, 3, 8]
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{ 2, 3, dim }, true);
    @memcpy(x_node.data, x_eager.data);

    const y = try moe.forward(&graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, dim }, y.shape.dims[0..y.shape.len]);

    // Backward pass
    @memset(y.grad, 1.0);
    try graph.backward(y);

    // Verify gradients propagated to input x
    var x_grad_sum: f32 = 0.0;
    for (x_node.grad) |g| x_grad_sum += @abs(g);
    try std.testing.expect(x_grad_sum > 0.0);

    // Verify gradients on gate weights
    var gate_grad_sum: f32 = 0.0;
    for (moe.gate.weight.grad) |g| gate_grad_sum += @abs(g);
    try std.testing.expect(gate_grad_sum > 0.0);

    // Verify gradients on shared expert
    var shared_grad_sum: f32 = 0.0;
    for (moe.shared_experts[0].c_fc.weight.grad) |g| shared_grad_sum += @abs(g);
    try std.testing.expect(shared_grad_sum > 0.0);

    // Verify gradients on at least one routed expert (due to top-k selection)
    var routed_grad_sum: f32 = 0.0;
    for (moe.routed_experts) |exp| {
        for (exp.c_fc.weight.grad) |g| routed_grad_sum += @abs(g);
    }
    try std.testing.expect(routed_grad_sum > 0.0);
}

test "MLALayer with MLACache matrix absorption inference" {
    const std = @import("std");
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(100);
    const random = prng.random();

    const dim: usize = 16;
    const n_head: usize = 4;
    const head_dim: usize = 4;
    const d_c: usize = 8;
    const d_r: usize = 4;

    var mla = try nn.MLALayer.init(allocator, dim, n_head, head_dim, d_c, d_r);
    try testing_init.initFromOnes(&mla, allocator, random, &.{ 1, 2, dim });
    defer nn.deinitModel(&mla, allocator);

    // 1. Eager mode full forward
    const x_eager = try tensor.zeros(allocator, &.{ 2, 3, dim });
    defer tensor.free(allocator, x_eager);
    for (x_eager.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
    }

    var y_eager_graph = autodiff.Graph.initNoGrad(allocator);
    defer y_eager_graph.deinit();
    const y_eager = try mla.forward(&y_eager_graph, x_eager);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, dim }, y_eager.shape.dims[0..y_eager.shape.len]);

    // 2. Autograd Graph mode full forward & backward
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{ 2, 3, dim }, true);
    @memcpy(x_node.data, x_eager.data);

    const y = try mla.forward(&graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, dim }, y.shape.dims[0..y.shape.len]);

    for (y_eager.data, y.data) |ve, vg| {
        try std.testing.expectApproxEqAbs(ve, vg, 1e-5);
    }

    // Use position-weighted upstream gradients so attention weights have non-zero gradients
    for (y.grad, 0..) |*g, i| {
        const fi = @as(f32, @floatFromInt(i));
        g.* = @sin(fi * 0.37) + 0.1 * fi;
    }
    try graph.backwardWithGrad(y);

    var x_grad_sum: f32 = 0.0;
    for (x_node.grad) |g| x_grad_sum += @abs(g);
    try std.testing.expect(x_grad_sum > 1e-5);

    const checkLinearGrad = struct {
        fn run(lin: nn.Linear) !void {
            var w_sum: f32 = 0.0;
            for (lin.weight.grad) |g| w_sum += @abs(g);
            try std.testing.expect(w_sum > 1e-6);
        }
    }.run;
    try checkLinearGrad(mla.q_proj);
    try checkLinearGrad(mla.w_dkv);
    try checkLinearGrad(mla.w_kr);
    try checkLinearGrad(mla.w_uk);
    try checkLinearGrad(mla.w_uv);
    try checkLinearGrad(mla.o_proj);

    // 3. Autoregressive inference with MLACache and Matrix Absorption (matches full forward!)
    var cache = try nn.MLACache.init(allocator, 1, 10, d_c, d_r);
    defer cache.deinit(allocator);

    const seq_x = try tensor.zeros(allocator, &.{ 1, 3, dim });
    defer tensor.free(allocator, seq_x);
    for (0..3) |step| {
        for (0..dim) |i| {
            seq_x.data[step * dim + i] = @as(f32, @floatFromInt(step + i)) * 0.05;
        }
    }
    var full_out_graph = autodiff.Graph.initNoGrad(allocator);
    defer full_out_graph.deinit();
    const full_out = try mla.forward(&full_out_graph, seq_x);

    for (0..3) |step| {
        const token_x = try tensor.zeros(allocator, &.{ 1, 1, dim });
        defer tensor.free(allocator, token_x);
        @memcpy(token_x.data, seq_x.data[step * dim .. (step + 1) * dim]);

        const out_step = try mla.forwardInference(allocator, token_x, &cache);
        defer tensor.free(allocator, out_step);

        try std.testing.expectEqualSlices(usize, &.{ 1, 1, dim }, out_step.shape.dims[0..out_step.shape.len]);
        try std.testing.expectEqual(@as(usize, step + 1), cache.curr_len);

        for (out_step.data, full_out.data[step * dim .. (step + 1) * dim]) |v_inf, v_full| {
            try std.testing.expectApproxEqAbs(v_full, v_inf, 1e-4);
        }
    }
}

test "ConvTranspose2D eager and autograd backward" {
    const std = @import("std");
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const in_channels: usize = 1;
    const out_channels: usize = 2;
    const kernel_size: usize = 3;
    const stride: usize = 1;
    const padding: usize = 0;

    var conv_t = try nn.ConvTranspose2D.init(
        allocator,
        in_channels,
        out_channels,
        kernel_size,
        .{ .stride = stride, .padding = padding, .use_bias = true },
    );
    conv_t.resetParameters(random, .{});
    defer nn.deinitModel(&conv_t, allocator);

    // 1. Eager mode on input [1, 1, 2, 2] -> expected [1, 2, 4, 4]
    const x_eager = try tensor.zeros(allocator, &.{ 1, in_channels, 2, 2 });
    defer tensor.free(allocator, x_eager);
    @memcpy(x_eager.data, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });

    var y_eager_graph = autodiff.Graph.initNoGrad(allocator);
    defer y_eager_graph.deinit();
    const y_eager = try conv_t.forward(&y_eager_graph, x_eager);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 4, 4 }, y_eager.shape.dims[0..y_eager.shape.len]);

    // 2. Autograd Graph mode
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{ 1, in_channels, 2, 2 }, true);
    @memcpy(x_node.data, x_eager.data);

    const y = try conv_t.forward(&graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 4, 4 }, y.shape.dims[0..y.shape.len]);

    // Check numerical match between eager and graph forward
    for (y_eager.data, y.data) |ve, vg| {
        try std.testing.expectApproxEqAbs(ve, vg, 1e-5);
    }

    // 3. Backward pass
    @memset(y.grad, 1.0);
    try graph.backward(y);

    var x_grad_sum: f32 = 0.0;
    for (x_node.grad) |g| x_grad_sum += @abs(g);
    try std.testing.expect(x_grad_sum > 0.0);

    var w_grad_sum: f32 = 0.0;
    for (conv_t.weight.grad) |g| w_grad_sum += @abs(g);
    try std.testing.expect(w_grad_sum > 0.0);

    if (conv_t.bias) |b| {
        var b_grad_sum: f32 = 0.0;
        for (b.grad) |g| b_grad_sum += @abs(g);
        try std.testing.expect(b_grad_sum > 0.0);
    }

    // 4. Test 2x upsampling configuration: stride=2, padding=1, kernel=4
    // H_out = (2 - 1) * 2 + 4 - 2 * 1 = 4 (exact 2x upsampling)
    var upsample_conv = try nn.ConvTranspose2D.init(
        allocator,
        1,
        1,
        4,
        .{ .stride = 2, .padding = 1, .use_bias = false },
    );
    upsample_conv.resetParameters(random, .{});
    defer nn.deinitModel(&upsample_conv, allocator);

    const x_up = try tensor.zeros(allocator, &.{ 1, 1, 3, 3 });
    defer tensor.free(allocator, x_up);
    @memset(x_up.data, 1.0);

    var y_up_graph = autodiff.Graph.initNoGrad(allocator);
    defer y_up_graph.deinit();
    const y_up = try upsample_conv.forward(&y_up_graph, x_up);
    // H_out = (3 - 1) * 2 + 4 - 2 = 6, W_out = 6
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 6, 6 }, y_up.shape.dims[0..y_up.shape.len]);
}

test "GAN adversarial training step" {
    const std = @import("std");
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // 1. 定义轻量 Generator: Linear(2, 8) -> LeakyReLU -> Linear(8, 2)
    const TinyGenerator = struct {
        l1: nn.Linear,
        act: nn.LeakyReLU,
        l2: nn.Linear,

        pub fn init(alloc: std.mem.Allocator) !@This() {
            return .{
                .l1 = try nn.Linear.init(alloc, 2, 8),
                .act = .{ .alpha = 0.2 },
                .l2 = try nn.Linear.init(alloc, 8, 2),
            };
        }
        pub fn deinit(self: @This(), alloc: std.mem.Allocator) void {
            nn.deinitModel(&self.l1, alloc);
            nn.deinitModel(&self.l2, alloc);
        }
        pub fn zeroGrad(self: *@This()) void {
            nn.zeroGradModel(&self.l1);
            nn.zeroGradModel(&self.l2);
        }
        pub fn forward(self: *@This(), g: *autodiff.Graph, z: *tensor.Tensor) !*tensor.Tensor {
            const h = try self.l1.forward(g, z);
            const a = try self.act.forward(g, h);
            return try self.l2.forward(g, a);
        }
    };

    // 2. 定义轻量 Discriminator: Linear(2, 8) -> LeakyReLU -> Linear(8, 1)
    const TinyDiscriminator = struct {
        l1: nn.Linear,
        act: nn.LeakyReLU,
        l2: nn.Linear,

        pub fn init(alloc: std.mem.Allocator) !@This() {
            return .{
                .l1 = try nn.Linear.init(alloc, 2, 8),
                .act = .{ .alpha = 0.2 },
                .l2 = try nn.Linear.init(alloc, 8, 1),
            };
        }
        pub fn deinit(self: @This(), alloc: std.mem.Allocator) void {
            nn.deinitModel(&self.l1, alloc);
            nn.deinitModel(&self.l2, alloc);
        }
        pub fn zeroGrad(self: *@This()) void {
            nn.zeroGradModel(&self.l1);
            nn.zeroGradModel(&self.l2);
        }
        pub fn forward(self: *@This(), g: *autodiff.Graph, x: *tensor.Tensor) !*tensor.Tensor {
            const h = try self.l1.forward(g, x);
            const a = try self.act.forward(g, h);
            return try self.l2.forward(g, a);
        }
    };

    var gen = try TinyGenerator.init(allocator);
    try testing_init.initFromOnes(&gen, allocator, random, &.{ 2, 2 });
    defer gen.deinit(allocator);

    var disc = try TinyDiscriminator.init(allocator);
    try testing_init.initFromOnes(&disc, allocator, random, &.{ 2, 2 });
    defer disc.deinit(allocator);

    var opt_g = try optim.AdamOptimizer.init(allocator, &gen, .{ .lr = 0.01, .beta1 = 0.5, .beta2 = 0.999 });
    defer opt_g.deinit();

    var opt_d = try optim.AdamOptimizer.init(allocator, &disc, .{ .lr = 0.01, .beta1 = 0.5, .beta2 = 0.999 });
    defer opt_d.deinit();

    const batch_size: usize = 16;

    // 执行 5 步对抗训练
    for (0..5) |_| {
        // Step D: 训练判别器
        var graph_d = autodiff.Graph.init(allocator);
        defer graph_d.deinit();

        const real_data = try graph_d.randomNormal(&.{ batch_size, 2 }, random, 3.0, 0.5, false);
        const real_targets = try graph_d.ones(&.{ batch_size, 1 }, false);

        const noise_d = try graph_d.randomNormal(&.{ batch_size, 2 }, random, 0.0, 1.0, false);
        var fake_eager_graph = autodiff.Graph.initNoGrad(allocator);
        defer fake_eager_graph.deinit();
        const fake_eager = try gen.forward(&fake_eager_graph, noise_d);

        const fake_data = try graph_d.array(&.{ batch_size, 2 }, fake_eager.data, false);
        const fake_targets = try graph_d.zeros(&.{ batch_size, 1 }, false);

        const real_logits = try disc.forward(&graph_d, real_data);
        const fake_logits = try disc.forward(&graph_d, fake_data);

        const loss_real = try graph_d.bceWithLogitsLoss(real_logits, real_targets);
        const loss_fake = try graph_d.bceWithLogitsLoss(fake_logits, fake_targets);
        const loss_d = try graph_d.add(loss_real, loss_fake);

        disc.zeroGrad();
        @memset(loss_d.grad, 1.0);
        try graph_d.backward(loss_d);
        opt_d.step();

        try std.testing.expect(!std.math.isNan(loss_d.data[0]));

        // Step G: 训练生成器
        var graph_g = autodiff.Graph.init(allocator);
        defer graph_g.deinit();

        const noise_g = try graph_g.randomNormal(&.{ batch_size, 2 }, random, 0.0, 1.0, false);
        const gen_out = try gen.forward(&graph_g, noise_g);
        const g_targets = try graph_g.ones(&.{ batch_size, 1 }, false);

        const g_logits = try disc.forward(&graph_g, gen_out);
        const loss_g = try graph_g.bceWithLogitsLoss(g_logits, g_targets);

        gen.zeroGrad();
        @memset(loss_g.grad, 1.0);
        try graph_g.backward(loss_g);
        opt_g.step();

        try std.testing.expect(!std.math.isNan(loss_g.data[0]));
    }
}

test "setTrainingModel, trainModel, and evalModel recursive reflection" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    const CompositeModel = struct {
        bn: nn.BatchNorm2d,
        drop: nn.Dropout,
        sub_bns: [2]nn.BatchNorm2d,
    };

    var m = CompositeModel{
        .bn = try nn.BatchNorm2d.init(allocator, 4, 1e-5, 0.1),
        .drop = nn.Dropout.initDefault(),
        .sub_bns = .{
            try nn.BatchNorm2d.init(allocator, 2, 1e-5, 0.1),
            try nn.BatchNorm2d.init(allocator, 2, 1e-5, 0.1),
        },
    };
    defer nn.deinitModel(&m, allocator);

    try std.testing.expect(m.bn.training);
    try std.testing.expect(m.drop.training);
    try std.testing.expect(m.sub_bns[0].training);
    try std.testing.expect(m.sub_bns[1].training);

    root.evalModel(&m);
    try std.testing.expect(!m.bn.training);
    try std.testing.expect(!m.drop.training);
    try std.testing.expect(!m.sub_bns[0].training);
    try std.testing.expect(!m.sub_bns[1].training);

    root.trainModel(&m);
    try std.testing.expect(m.bn.training);
    try std.testing.expect(m.drop.training);
    try std.testing.expect(m.sub_bns[0].training);
    try std.testing.expect(m.sub_bns[1].training);
}

test "Fixed array [N]*Tensor reflection and Safetensors serialization" {
    const std = @import("std");
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const ArrayTensorModel = struct {
        bank: [2]*tensor.Tensor,
    };

    var m1 = ArrayTensorModel{
        .bank = .{
            try nn.createPersistentTensor(allocator, 1, 3, true),
            try nn.createPersistentTensor(allocator, 1, 2, true),
        },
    };
    defer nn.deinitModel(&m1, allocator);

    m1.bank[0].data[0] = 1.5;
    m1.bank[0].data[1] = -2.5;
    m1.bank[0].data[2] = 3.5;
    m1.bank[1].data[0] = 4.25;
    m1.bank[1].data[1] = -5.75;

    const params = try nn.parameters(&m1, allocator);
    defer allocator.free(params);
    try std.testing.expectEqual(@as(usize, 2), params.len);

    m1.bank[0].grad[0] = 9.0;
    m1.bank[1].grad[1] = 8.0;
    nn.zeroGradModel(&m1);
    try std.testing.expectEqual(@as(f32, 0.0), m1.bank[0].grad[0]);
    try std.testing.expectEqual(@as(f32, 0.0), m1.bank[1].grad[1]);

    const tmp_path = "/tmp/znn_test_array_tensor_model.safetensors";
    defer std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};

    try nn.saveModel(&m1, io, tmp_path, allocator);

    var m2 = ArrayTensorModel{
        .bank = .{
            try nn.createPersistentTensor(allocator, 1, 3, true),
            try nn.createPersistentTensor(allocator, 1, 2, true),
        },
    };
    defer nn.deinitModel(&m2, allocator);

    try nn.loadModel(&m2, io, tmp_path, allocator);
    try std.testing.expectEqualSlices(f32, m1.bank[0].data, m2.bank[0].data);
    try std.testing.expectEqualSlices(f32, m1.bank[1].data, m2.bank[1].data);
}

test "nameModules copies the root name and names survive moving the module" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    var names = std.heap.ArenaAllocator.init(allocator);
    defer names.deinit();

    var lin = try nn.Linear.init(allocator, 4, 2);
    defer nn.deinitModel(&lin, allocator);

    {
        var stack_buf: [16]u8 = undefined;
        const dynamic_name = try std.fmt.bufPrint(&stack_buf, "layer_{d}", .{7});
        try nn.nameModules(&lin, names.allocator(), dynamic_name);
        @memset(&stack_buf, 'X');
    }

    // 名称存放在 arena 中而非模块自身，按值复制 (如放入 nn.sequential) 后仍然有效
    const seq = nn.sequential(.{lin});
    try std.testing.expectEqualStrings("layer_7", seq.layers[0].name.?);
    try std.testing.expectEqualStrings("layer_7.weight", seq.layers[0].weight.getName().?);
    try std.testing.expectEqualStrings("layer_7.bias", lin.bias.getName().?);
}

test "Config defaults and initDefault ergonomics across modules" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    var cv_default = cv.CrossValidationGridSearch.initDefault(allocator);
    defer cv_default.deinit();
    try std.testing.expectEqual(@as(usize, 5), cv_default.k_splits);
    try std.testing.expectEqual(@as(usize, 5), root.CrossValidationOptions.defaultOptions().k_splits);

    const drop = nn.Dropout.initDefault();
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), drop.p, 1e-6);
    try std.testing.expect(nn.Dropout.defaultOptions().training);

    const lrelu = nn.LeakyReLU.initDefault();
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), lrelu.alpha, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), nn.LeakyReLU.defaultOptions().alpha, 1e-6);

    const cos = CosineScheduler.initDefault();
    try std.testing.expect(cos.max_lr > cos.min_lr);
    const step_s = StepLRScheduler.initDefault();
    try std.testing.expect(step_s.step_size > 0);
    const warm_s = LinearWarmupScheduler.initDefault();
    try std.testing.expect(warm_s.warmup_steps > 0);
    const exp_s = ExponentialLRScheduler.initDefault();
    try std.testing.expect(exp_s.gamma > 0.0);
    const lr_s = LRScheduler.defaultConfig();
    try std.testing.expect(lr_s.getLR(0) >= 0.0);

    try std.testing.expect(Nonlinearity.defaultOptions() == .relu);
    try std.testing.expect(InitMethod.defaultOptions() == .he_normal);
    try std.testing.expectEqual(@as(usize, 1), nn.Conv1D.Options.defaultOptions().stride);
    try std.testing.expectEqual(@as(usize, 0), nn.Conv1D.Options.defaultOptions().padding);
    try std.testing.expectEqual(@as(usize, 1), nn.Conv2D.Options.defaultOptions().stride);
    try std.testing.expectEqual(@as(usize, 0), nn.Conv2D.Options.defaultOptions().padding);
    try std.testing.expect(@TypeOf(root.DefaultGPT) == type);
}

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
