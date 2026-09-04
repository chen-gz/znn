pub const tensor = @import("tensor.zig");
pub const nn = @import("nn.zig");
pub const dataset = @import("dataset.zig");
pub const autodiff = @import("autodiff.zig");
pub const optim = @import("optim.zig");
pub const regression = @import("regression.zig");
pub const cv = @import("cross_validation.zig");
pub const engine = @import("engine.zig");


pub fn measureTime(comptime func: anytype, args: anytype) !struct {
    result: @TypeOf(@call(.auto, func, args)),
    elapsed_ns: u64,
} {
    const std = @import("std");
    var start_ts: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &start_ts);

    const result = @call(.auto, func, args);

    var end_ts: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &end_ts);

    const start_ns = @as(u64, @intCast(start_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(start_ts.nsec));
    const end_ns = @as(u64, @intCast(end_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(end_ts.nsec));
    return .{
        .result = result,
        .elapsed_ns = end_ns - start_ns,
    };
}

test "basic imports and struct definitions" {
    const std = @import("std");
    _ = @import("optim.zig");
    try std.testing.expect(@TypeOf(nn.Linear) == type);
    try std.testing.expect(@TypeOf(nn.SwiGLU) == type);
    try std.testing.expect(@TypeOf(nn.LoRALinear) == type);
    try std.testing.expect(@TypeOf(nn.TransformerDecoder) == fn(comptime usize) type);
    try std.testing.expect(@TypeOf(autodiff.Tensor) == type);
    try std.testing.expect(@TypeOf(tensor.Tensor) == type);
    try std.testing.expect(@TypeOf(optim.SGDOptimizer) == type);
    try std.testing.expect(@TypeOf(optim.AdamOptimizer) == type);
    try std.testing.expect(@TypeOf(optim.AdamWOptimizer) == type);
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

pub const ProfileBlock = struct {
    label: []const u8,
    start_ts: @import("std").posix.system.timespec,

    pub fn start(label: []const u8) ProfileBlock {
        const std = @import("std");
        var start_ts: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &start_ts);
        return .{
            .label = label,
            .start_ts = start_ts,
        };
    }

    pub fn end(self: ProfileBlock) void {
        const std = @import("std");
        var end_ts: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &end_ts);
        const start_ns = @as(u64, @intCast(self.start_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(self.start_ts.nsec));
        const end_ns = @as(u64, @intCast(end_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(end_ts.nsec));
        const elapsed_ms = @as(f64, @floatFromInt(end_ns - start_ns)) / 1_000_000.0;
        std.debug.print("[PROFILE] {s} took {d:.3}ms\n", .{ self.label, elapsed_ms });
    }
};

pub const ScopeTimer = struct {
    start_ts: @import("std").posix.system.timespec,
    elapsed_ns_ptr: *u64,

    pub fn start(elapsed_ns_ptr: *u64) ScopeTimer {
        const std = @import("std");
        var start_ts: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &start_ts);
        return .{
            .start_ts = start_ts,
            .elapsed_ns_ptr = elapsed_ns_ptr,
        };
    }

    pub fn end(self: ScopeTimer) void {
        const std = @import("std");
        var end_ts: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &end_ts);
        const start_ns = @as(u64, @intCast(self.start_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(self.start_ts.nsec));
        const end_ns = @as(u64, @intCast(end_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(end_ts.nsec));
        self.elapsed_ns_ptr.* = end_ns - start_ns;
    }
};

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

    // Test print
    std.debug.print("\nTesting print function for 2x3 tensor:\n", .{});
    A.print();

    // Transpose it to 3x2
    const B = try graph.transposeND(A, 0, 1);
    try std.testing.expectEqualSlices(usize, &.{3, 2}, B.shape.dims[0..B.shape.len]);
    try std.testing.expectEqual(@as(f32, 1.0), B.data[0]); // A[0,0]
    try std.testing.expectEqual(@as(f32, 4.0), B.data[1]); // A[1,0]
    try std.testing.expectEqual(@as(f32, 2.0), B.data[2]); // A[0,1]
    try std.testing.expectEqual(@as(f32, 5.0), B.data[3]); // A[1,1]

    std.debug.print("Testing print function for transposed 3x2 tensor:\n", .{});
    B.print();

    // Reshape it to 1x6
    const C = try graph.reshape(B, &.{1, 6});
    try std.testing.expectEqualSlices(usize, &.{1, 6}, C.shape.dims[0..C.shape.len]);

    std.debug.print("Testing print function for reshaped 1x6 tensor:\n", .{});
    C.print();

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

    const C = try graph.conv2d(A, W, bias);
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

    const C = try graph.maxpool2d(A, 2, 2);
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

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
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
    var linear = try nn.Linear.init(allocator, 2, 2, prng.random());
    defer linear.deinit(allocator);

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
    var out: [2]f32 = undefined;
    const gate = [_]f32{ 0.0, 2.0 };
    const up = [_]f32{ 3.0, 4.0 };

    nn.swigluForward(&out, &gate, &up);
    // out[0] = (0 * sig(0)) * 3 = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-5);
    // out[1] = (2 * sig(2)) * 4 = 2 * (1 / (1 + e^-2)) * 4 = 8 * 0.880797 = 7.046376
    try std.testing.expectApproxEqAbs(@as(f32, 7.046376), out[1], 1e-4);
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
    var swiglu = try nn.SwiGLU.init(allocator, 8, 16, random);
    defer swiglu.deinit(allocator);

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

    const out = try swiglu.forward(allocator, &graph, x);
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









