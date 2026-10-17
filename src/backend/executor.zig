//! 基于 Zig 0.16 输入输出接口 (Input/Output, Io) 的数据并行执行器 (Executor)。
//! 库本身不创建全局运行时或后台线程：线程池由应用创建 (`std.Io.Threaded`) 并经 `GraphOptions.io` 注入。

const std = @import("std");

/// 数据并行执行器 (Executor)
///
/// 遵循 AGENTS.md §4 规范：所有字段均有默认值，可以直接使用 `.{}` 或 `defaultOptions()` 构造；
/// `Executor.serial` 提供无并发的零开销串行默认实例。
pub const Executor = struct {
    /// 注入的输入输出并发接口；为 null 时完全串行执行，不引入任何多线程开销
    io: ?std.Io = null,
    /// 最大并发任务数上限；为 null 时自动查询逻辑核心数
    max_tasks: ?usize = null,

    pub const serial: Executor = .{};

    pub fn defaultOptions() Executor {
        return .{};
    }

    /// 将区间 [0, n) 按固定粒度 `grain` 切块并行调用 `body(ctx, begin, end)`。
    /// 块边界只取决于 n 与 grain（与线程数无关），因此结果跨线程数逐位可复现。
    pub fn parallelFor(
        self: Executor,
        n: usize,
        grain: usize,
        ctx: anytype,
        comptime body: fn (@TypeOf(ctx), usize, usize) void,
    ) std.Io.Cancelable!void {
        if (n == 0) return;
        const effective_grain = if (grain == 0) n else grain;
        const chunks = (n + effective_grain - 1) / effective_grain;

        if (self.io == null or chunks <= 1) {
            for (0..chunks) |c| {
                const start = c * effective_grain;
                const end = @min(start + effective_grain, n);
                body(ctx, start, end);
            }
            return;
        }

        const io = self.io.?;
        const cpu_count = std.Thread.getCpuCount() catch 1;
        const max_allowed = self.max_tasks orelse cpu_count;
        const tasks = @max(1, @min(chunks, max_allowed));

        if (tasks <= 1) {
            for (0..chunks) |c| {
                const start = c * effective_grain;
                const end = @min(start + effective_grain, n);
                body(ctx, start, end);
            }
            return;
        }

        var next_chunk = std.atomic.Value(usize).init(0);
        const Worker = struct {
            fn run(
                next_ptr: *std.atomic.Value(usize),
                total_chunks: usize,
                g_size: usize,
                total_n: usize,
                worker_ctx: @TypeOf(ctx),
            ) void {
                while (true) {
                    const c = next_ptr.fetchAdd(1, .monotonic);
                    if (c >= total_chunks) break;
                    const start = c * g_size;
                    const end = @min(start + g_size, total_n);
                    body(worker_ctx, start, end);
                }
            }
        };

        var group: std.Io.Group = .init;
        for (1..tasks) |_| {
            group.async(io, Worker.run, .{ &next_chunk, chunks, effective_grain, n, ctx });
        }
        Worker.run(&next_chunk, chunks, effective_grain, n, ctx);
        try group.await(io);
    }

    /// 有序归约：每块写入 `partials[chunk]`，由调用线程按块序合并，保证确定性。
    pub fn parallelReduce(
        self: Executor,
        comptime T: type,
        n: usize,
        grain: usize,
        ctx: anytype,
        comptime map_fn: fn (@TypeOf(ctx), usize, usize) T,
        comptime reduce_fn: fn (T, T) T,
        identity: T,
    ) (std.Io.Cancelable || error{OutOfMemory})!T {
        if (n == 0) return identity;
        const effective_grain = if (grain == 0) n else grain;
        const chunks = (n + effective_grain - 1) / effective_grain;

        if (self.io == null or chunks <= 1) {
            var acc = identity;
            for (0..chunks) |c| {
                const start = c * effective_grain;
                const end = @min(start + effective_grain, n);
                acc = reduce_fn(acc, map_fn(ctx, start, end));
            }
            return acc;
        }

        var stack_buf: [256]T = undefined;
        var heap_buf: ?[]T = null;
        var partials: []T = undefined;

        if (chunks <= stack_buf.len) {
            partials = stack_buf[0..chunks];
        } else {
            heap_buf = try std.heap.page_allocator.alloc(T, chunks);
            partials = heap_buf.?;
        }
        defer if (heap_buf) |h| std.heap.page_allocator.free(h);

        const io = self.io.?;
        const cpu_count = std.Thread.getCpuCount() catch 1;
        const max_allowed = self.max_tasks orelse cpu_count;
        const tasks = @max(1, @min(chunks, max_allowed));

        var next_chunk = std.atomic.Value(usize).init(0);
        const Worker = struct {
            fn run(
                next_ptr: *std.atomic.Value(usize),
                total_chunks: usize,
                g_size: usize,
                total_n: usize,
                worker_ctx: @TypeOf(ctx),
                out_partials: []T,
            ) void {
                while (true) {
                    const c = next_ptr.fetchAdd(1, .monotonic);
                    if (c >= total_chunks) break;
                    const start = c * g_size;
                    const end = @min(start + g_size, total_n);
                    out_partials[c] = map_fn(worker_ctx, start, end);
                }
            }
        };

        var group: std.Io.Group = .init;
        for (1..tasks) |_| {
            group.async(io, Worker.run, .{ &next_chunk, chunks, effective_grain, n, ctx, partials });
        }
        Worker.run(&next_chunk, chunks, effective_grain, n, ctx, partials);
        try group.await(io);

        var final_acc = identity;
        for (partials) |p| {
            final_acc = reduce_fn(final_acc, p);
        }
        return final_acc;
    }
};

test "Executor serial execution" {
    const exec = Executor.serial;
    var data = [_]u32{0} ** 100;
    const Ctx = struct {
        arr: []u32,
        fn fill(ctx: @This(), begin: usize, end: usize) void {
            for (begin..end) |i| {
                ctx.arr[i] = @intCast(i * 2);
            }
        }
    };
    try exec.parallelFor(100, 10, Ctx{ .arr = &data }, Ctx.fill);
    for (0..100) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i * 2)), data[i]);
    }

    const MapReduce = struct {
        arr: []const u32,
        fn map(ctx: @This(), begin: usize, end: usize) u64 {
            var s: u64 = 0;
            for (begin..end) |i| s += ctx.arr[i];
            return s;
        }
        fn add(a: u64, b: u64) u64 {
            return a + b;
        }
    };
    const total = try exec.parallelReduce(u64, 100, 10, MapReduce{ .arr = &data }, MapReduce.map, MapReduce.add, 0);
    // sum of i*2 for i in 0..99 = 99 * 100 = 9900
    try std.testing.expectEqual(@as(u64, 9900), total);
}

test "Executor multi-task execution with std.Io.Threaded" {
    const allocator = std.testing.allocator;
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();

    const exec = Executor{ .io = threaded.io(), .max_tasks = 4 };
    var data = [_]u32{0} ** 1000;
    const Ctx = struct {
        arr: []u32,
        fn fill(ctx: @This(), begin: usize, end: usize) void {
            for (begin..end) |i| {
                ctx.arr[i] = @intCast(i + 1);
            }
        }
    };
    try exec.parallelFor(1000, 25, Ctx{ .arr = &data }, Ctx.fill);
    for (0..1000) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i + 1)), data[i]);
    }

    const MapReduce = struct {
        arr: []const u32,
        fn map(ctx: @This(), begin: usize, end: usize) u64 {
            var s: u64 = 0;
            for (begin..end) |i| s += ctx.arr[i];
            return s;
        }
        fn add(a: u64, b: u64) u64 {
            return a + b;
        }
    };
    const total = try exec.parallelReduce(u64, 1000, 25, MapReduce{ .arr = &data }, MapReduce.map, MapReduce.add, 0);
    // sum of 1..1000 = 1000 * 1001 / 2 = 500500
    try std.testing.expectEqual(@as(u64, 500500), total);
}
