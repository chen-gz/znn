//! 基准测试运行器 (Benchmark Runner)：计时、统计与结果表格输出。
//! 被基准门面 `bench.zig` 与套件 `bench/suites.zig` 共同依赖，自身只依赖标准库。
const std = @import("std");

pub const BenchmarkStats = struct {
    name: []const u8,
    category: []const u8,
    iterations: usize,
    min_ns: u64,
    avg_ns: u64,
    max_ns: u64,
    stddev_ns: u64,
    flops: ?f64 = null,
    bytes_processed: ?u64 = null,
    items_count: ?usize = null,
    items_unit: []const u8 = "",

    pub fn minMs(self: BenchmarkStats) f64 {
        return @as(f64, @floatFromInt(self.min_ns)) / 1_000_000.0;
    }

    pub fn avgMs(self: BenchmarkStats) f64 {
        return @as(f64, @floatFromInt(self.avg_ns)) / 1_000_000.0;
    }

    pub fn maxMs(self: BenchmarkStats) f64 {
        return @as(f64, @floatFromInt(self.max_ns)) / 1_000_000.0;
    }

    pub fn stddevMs(self: BenchmarkStats) f64 {
        return @as(f64, @floatFromInt(self.stddev_ns)) / 1_000_000.0;
    }

    pub fn gflops(self: BenchmarkStats) ?f64 {
        if (self.flops) |f| {
            if (self.avg_ns == 0) return 0.0;
            const secs = @as(f64, @floatFromInt(self.avg_ns)) / 1_000_000_000.0;
            return (f / secs) / 1e9;
        }
        return null;
    }

    pub fn throughputGBs(self: BenchmarkStats) ?f64 {
        if (self.bytes_processed) |b| {
            if (self.avg_ns == 0) return 0.0;
            const secs = @as(f64, @floatFromInt(self.avg_ns)) / 1_000_000_000.0;
            return (@as(f64, @floatFromInt(b)) / secs) / 1e9;
        }
        return null;
    }

    pub fn throughputItems(self: BenchmarkStats) ?f64 {
        if (self.items_count) |items| {
            if (self.avg_ns == 0) return 0.0;
            const secs = @as(f64, @floatFromInt(self.avg_ns)) / 1_000_000_000.0;
            return @as(f64, @floatFromInt(items)) / secs;
        }
        return null;
    }
};

pub fn getTimeNs() u64 {
    var ts: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub const BenchmarkConfig = struct {
    warmup: usize = 3,
    iterations: usize = 10,
    filter: ?[]const u8 = null,
    suite: ?[]const u8 = null,
    quiet: bool = false,

    pub const default: BenchmarkConfig = .{};
    pub fn defaultConfig() BenchmarkConfig {
        return .{};
    }
};

pub const BenchmarkRunner = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayList(BenchmarkStats),
    config: BenchmarkConfig,

    pub fn initDefault(allocator: std.mem.Allocator) BenchmarkRunner {
        return init(allocator, BenchmarkConfig.default);
    }

    pub fn init(allocator: std.mem.Allocator, config: BenchmarkConfig) BenchmarkRunner {
        return .{
            .allocator = allocator,
            .results = .empty,
            .config = config,
        };
    }

    pub fn deinit(self: *BenchmarkRunner) void {
        for (self.results.items) |st| {
            self.allocator.free(st.name);
            self.allocator.free(st.category);
            if (st.items_unit.len > 0) {
                self.allocator.free(st.items_unit);
            }
        }
        self.results.deinit(self.allocator);
    }

    pub fn shouldRun(self: *const BenchmarkRunner, name: []const u8, category: []const u8) bool {
        if (self.config.suite) |s| {
            if (!std.ascii.eqlIgnoreCase(s, "all")) {
                var cat_buf: [64]u8 = undefined;
                const c_len = @min(category.len, cat_buf.len);
                for (category[0..c_len], 0..) |c, i| {
                    cat_buf[i] = std.ascii.toLower(c);
                }
                const lower_cat = cat_buf[0..c_len];

                var suite_buf: [64]u8 = undefined;
                const s_len = @min(s.len, suite_buf.len);
                for (s[0..s_len], 0..) |c, i| {
                    suite_buf[i] = std.ascii.toLower(c);
                }
                const lower_suite = suite_buf[0..s_len];

                if (std.mem.indexOf(u8, lower_cat, lower_suite) == null) {
                    return false;
                }
            }
        }

        if (self.config.filter) |f| {
            var lower_name_buf: [128]u8 = undefined;
            const n_len = @min(name.len, lower_name_buf.len);
            for (name[0..n_len], 0..) |c, i| {
                lower_name_buf[i] = std.ascii.toLower(c);
            }
            const lower_name = lower_name_buf[0..n_len];

            var lower_filt_buf: [128]u8 = undefined;
            const f_len = @min(f.len, lower_filt_buf.len);
            for (f[0..f_len], 0..) |c, i| {
                lower_filt_buf[i] = std.ascii.toLower(c);
            }
            const lower_filt = lower_filt_buf[0..f_len];

            if (std.mem.indexOf(u8, lower_name, lower_filt) == null) {
                return false;
            }
        }
        return true;
    }

    pub fn benchmark(
        self: *BenchmarkRunner,
        name: []const u8,
        category: []const u8,
        flops: ?f64,
        bytes_processed: ?u64,
        items_count: ?usize,
        items_unit: []const u8,
        context: anytype,
    ) !void {
        return self.benchmarkWithCap(
            name,
            category,
            flops,
            bytes_processed,
            items_count,
            items_unit,
            self.config.warmup,
            self.config.iterations,
            context,
        );
    }

    pub fn benchmarkWithCap(
        self: *BenchmarkRunner,
        name: []const u8,
        category: []const u8,
        flops: ?f64,
        bytes_processed: ?u64,
        items_count: ?usize,
        items_unit: []const u8,
        max_warmup: usize,
        max_iterations: usize,
        context: anytype,
    ) !void {
        if (!self.shouldRun(name, category)) return;

        const warmup = @min(self.config.warmup, max_warmup);
        const iters = @max(@as(usize, 1), @min(self.config.iterations, max_iterations));

        // Warmup
        for (0..warmup) |_| {
            try context.run();
        }

        // Measurement iterations
        var times = try self.allocator.alloc(u64, iters);
        defer self.allocator.free(times);

        var total_ns: u64 = 0;
        var min_ns: u64 = std.math.maxInt(u64);
        var max_ns: u64 = 0;

        for (0..iters) |i| {
            const start = getTimeNs();
            try context.run();
            const end = getTimeNs();
            const elapsed = end - start;

            times[i] = elapsed;
            total_ns += elapsed;
            if (elapsed < min_ns) min_ns = elapsed;
            if (elapsed > max_ns) max_ns = elapsed;
        }

        const avg_ns = if (iters > 0) total_ns / iters else 0;

        // Standard deviation
        var variance_sum: f64 = 0.0;
        const avg_f = @as(f64, @floatFromInt(avg_ns));
        for (times) |t| {
            const diff = @as(f64, @floatFromInt(t)) - avg_f;
            variance_sum += diff * diff;
        }
        const stddev_ns = if (iters > 0)
            @as(u64, @intFromFloat(@sqrt(variance_sum / @as(f64, @floatFromInt(iters)))))
        else
            0;

        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_cat = try self.allocator.dupe(u8, category);
        errdefer self.allocator.free(owned_cat);
        const owned_unit = if (items_unit.len > 0)
            try self.allocator.dupe(u8, items_unit)
        else
            "";
        errdefer if (owned_unit.len > 0) self.allocator.free(owned_unit);

        const stats = BenchmarkStats{
            .name = owned_name,
            .category = owned_cat,
            .iterations = iters,
            .min_ns = min_ns,
            .avg_ns = avg_ns,
            .max_ns = max_ns,
            .stddev_ns = stddev_ns,
            .flops = flops,
            .bytes_processed = bytes_processed,
            .items_count = items_count,
            .items_unit = owned_unit,
        };

        try self.results.append(self.allocator, stats);

        if (!self.config.quiet) {
            printProgressLine(stats);
        }
    }

    pub fn printSummary(self: *const BenchmarkRunner) void {
        std.debug.print("\n", .{});
        std.debug.print("=" ** 106 ++ "\n", .{});
        std.debug.print(" {s:<36} {s:<14} {s:>5}  {s:>9}  {s:>9}  {s:>9}  {s:>16}\n", .{
            "Benchmark Name", "Category", "Iter", "Min(ms)", "Avg(ms)", "Max(ms)", "Throughput",
        });
        std.debug.print("-" ** 106 ++ "\n", .{});

        var current_cat: ?[]const u8 = null;
        for (self.results.items) |st| {
            if (current_cat == null or !std.mem.eql(u8, current_cat.?, st.category)) {
                if (current_cat != null) {
                    std.debug.print("-" ** 106 ++ "\n", .{});
                }
                current_cat = st.category;
            }

            var tp_buf: [32]u8 = undefined;
            const tp_str = formatThroughput(st, &tp_buf);

            std.debug.print(" {s:<36} {s:<14} {d:>5}  {d:>9.3}  {d:>9.3}  {d:>9.3}  {s:>16}\n", .{
                st.name,
                st.category,
                st.iterations,
                st.minMs(),
                st.avgMs(),
                st.maxMs(),
                tp_str,
            });
        }

        std.debug.print("=" ** 106 ++ "\n", .{});
        std.debug.print(" Total benchmarks completed: {d}\n\n", .{self.results.items.len});
    }
};

fn printProgressLine(st: BenchmarkStats) void {
    var tp_buf: [32]u8 = undefined;
    const tp_str = formatThroughput(st, &tp_buf);

    std.debug.print("  [✓] {s:<34} avg: {d:>7.3} ms (min: {d:>7.3} ms) -> {s}\n", .{
        st.name,
        st.avgMs(),
        st.minMs(),
        tp_str,
    });
}

fn formatThroughput(st: BenchmarkStats, buf: *[32]u8) []const u8 {
    if (st.gflops()) |g| {
        return std.fmt.bufPrint(buf, "{d:.2} GFLOPS", .{g}) catch "N/A";
    } else if (st.throughputItems()) |it| {
        if (st.items_unit.len > 0) {
            return std.fmt.bufPrint(buf, "{d:.1} {s}", .{ it, st.items_unit }) catch "N/A";
        } else {
            return std.fmt.bufPrint(buf, "{d:.1} items/s", .{it}) catch "N/A";
        }
    } else if (st.throughputGBs()) |bw| {
        if (bw >= 0.01) {
            return std.fmt.bufPrint(buf, "{d:.2} GB/s", .{bw}) catch "N/A";
        } else if (bw * 1000.0 >= 0.01) {
            return std.fmt.bufPrint(buf, "{d:.2} MB/s", .{bw * 1000.0}) catch "N/A";
        } else {
            return std.fmt.bufPrint(buf, "{d:.2} KB/s", .{bw * 1_000_000.0}) catch "N/A";
        }
    }
    return "-";
}
