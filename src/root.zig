pub const tensor = @import("tensor.zig");
pub const nn = @import("nn.zig");
pub const dataset = @import("dataset.zig");
pub const autodiff = @import("autodiff.zig");

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
    try std.testing.expect(@TypeOf(nn.Linear) == type);
    try std.testing.expect(@TypeOf(nn.TransformerDecoder) == fn(comptime usize) type);
    try std.testing.expect(@TypeOf(autodiff.Tensor) == type);
    try std.testing.expect(@TypeOf(tensor.Tensor) == type);
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

test "json testing" {
    const std = @import("std");
    const json_str = "{\"a\": 123}";
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json_str, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.object.get("a").?.integer == 123);
}






