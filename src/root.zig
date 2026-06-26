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

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}


