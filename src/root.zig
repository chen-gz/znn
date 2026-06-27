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

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}


