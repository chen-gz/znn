//! 基于 Zig 0.16 输入输出接口 (Input/Output, Io) 的数据并行执行器 (Executor)。
//! 库本身不创建全局运行时或后台线程：线程池由应用创建 (`std.Io.Threaded`) 并经 `GraphOptions.io` 注入。

const std = @import("std");

/// 逻辑核心数缓存：首次查询后复用，避免每次派发都发起系统调用
var cached_cpu_count = std.atomic.Value(usize).init(0);

fn cpuCount() usize {
    const cached = cached_cpu_count.load(.monotonic);
    if (cached != 0) return cached;
    const n = @max(1, std.Thread.getCpuCount() catch 1);
    cached_cpu_count.store(n, .monotonic);
    return n;
}

/// 数据并行执行器 (Executor)
///
/// 遵循 AGENTS.md §4 规范：所有字段均有默认值，可以直接使用 `.{}` 或 `defaultOptions()` 构造；
/// `Executor.serial` 提供无并发的零开销串行默认实例。
pub const Executor = struct {
    /// 注入的输入输出并发接口；为 null 时完全串行执行，不引入任何多线程开销
    io: ?std.Io = null,
    /// 最大并发任务数上限；为 null 时使用逻辑核心数
    max_tasks: ?usize = null,

    pub const serial: Executor = .{};

    pub fn defaultOptions() Executor {
        return .{};
    }

    /// `parallelReduce` 每批最多并行计算的块数（局部结果保存在栈上，不进行堆分配）
    pub const max_reduce_batch: usize = 256;

    fn taskLimit(self: Executor) usize {
        if (self.io == null) return 1;
        return @max(1, self.max_tasks orelse cpuCount());
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
        const tasks = @min(chunks, self.taskLimit());

        if (tasks <= 1) {
            for (0..chunks) |c| {
                const start = c * effective_grain;
                body(ctx, start, @min(start + effective_grain, n));
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
                    body(worker_ctx, start, @min(start + g_size, total_n));
                }
            }
        };

        const io = self.io.?;
        var group: std.Io.Group = .init;
        for (1..tasks) |_| {
            group.async(io, Worker.run, .{ &next_chunk, chunks, effective_grain, n, ctx });
        }
        Worker.run(&next_chunk, chunks, effective_grain, n, ctx);
        try group.await(io);
    }

    /// 有序归约：块边界只取决于 n 与 grain；各块结果按块序由调用线程合并，保证结果与线程数无关、逐位可复现。
    /// 块数超过 `max_reduce_batch` 时分批并行计算，每批的局部结果保存在栈上，不进行堆分配。
    pub fn parallelReduce(
        self: Executor,
        comptime T: type,
        n: usize,
        grain: usize,
        ctx: anytype,
        comptime map_fn: fn (@TypeOf(ctx), usize, usize) T,
        comptime reduce_fn: fn (T, T) T,
        identity: T,
    ) std.Io.Cancelable!T {
        if (n == 0) return identity;
        const effective_grain = if (grain == 0) n else grain;
        const chunks = (n + effective_grain - 1) / effective_grain;

        var acc = identity;
        if (chunks <= 1 or self.taskLimit() <= 1) {
            for (0..chunks) |c| {
                const start = c * effective_grain;
                acc = reduce_fn(acc, map_fn(ctx, start, @min(start + effective_grain, n)));
            }
            return acc;
        }

        const Batch = struct {
            map_ctx: @TypeOf(ctx),
            g_size: usize,
            total_n: usize,
            first_chunk: usize,
            out_partials: []T,

            fn run(batch: @This(), begin: usize, end: usize) void {
                for (begin..end) |i| {
                    const start = (batch.first_chunk + i) * batch.g_size;
                    batch.out_partials[i] = map_fn(batch.map_ctx, start, @min(start + batch.g_size, batch.total_n));
                }
            }
        };

        var partials: [max_reduce_batch]T = undefined;
        var first: usize = 0;
        while (first < chunks) : (first += max_reduce_batch) {
            const count = @min(max_reduce_batch, chunks - first);
            const batch = Batch{
                .map_ctx = ctx,
                .g_size = effective_grain,
                .total_n = n,
                .first_chunk = first,
                .out_partials = partials[0..count],
            };
            try self.parallelFor(count, 1, batch, Batch.run);
            for (partials[0..count]) |p| acc = reduce_fn(acc, p);
        }
        return acc;
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

test "Executor parallelReduce with more chunks than one batch stays deterministic" {
    const allocator = std.testing.allocator;
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();

    const data = try allocator.alloc(f32, 10_000);
    defer allocator.free(data);
    for (data, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i))) * 3.0;

    const MapReduce = struct {
        arr: []const f32,
        fn map(ctx: @This(), begin: usize, end: usize) f32 {
            var s: f32 = 0.0;
            for (ctx.arr[begin..end]) |v| s += v;
            return s;
        }
        fn add(a: f32, b: f32) f32 {
            return a + b;
        }
    };
    // grain = 7 → 1429 块，超过单批上限 256，需要分批归约
    const serial_sum = try Executor.serial.parallelReduce(f32, data.len, 7, MapReduce{ .arr = data }, MapReduce.map, MapReduce.add, 0.0);
    const threaded_sum = try (Executor{ .io = threaded.io(), .max_tasks = 6 }).parallelReduce(f32, data.len, 7, MapReduce{ .arr = data }, MapReduce.map, MapReduce.add, 0.0);
    try std.testing.expectEqual(serial_sum, threaded_sum);
}
