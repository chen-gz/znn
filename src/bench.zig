const std = @import("std");
const builtin = @import("builtin");
const tensor = @import("tensor.zig");
const nn = @import("nn.zig");
const autodiff = @import("autodiff.zig");
const optim = @import("optim.zig");
const dataset = @import("dataset.zig");
const engine = @import("engine.zig");

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
        if (!self.shouldRun(name, category)) return;

        // Warmup
        for (0..self.config.warmup) |_| {
            try context.run();
        }

        // Measurement iterations
        const iters = self.config.iterations;
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

// ============================================================================
// Benchmark Suites
// ============================================================================

/// Suite 1: BLAS / GEMM Benchmarks
pub fn runGemmBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    const GemmContext = struct {
        allocator: std.mem.Allocator,
        arena: std.heap.ArenaAllocator,
        A: *tensor.Tensor,
        B: *tensor.Tensor,

        pub fn init(alloc: std.mem.Allocator, M: usize, K: usize, N: usize) !@This() {
            var prng = std.Random.DefaultPrng.init(42);
            const A = try tensor.zeros(alloc, &.{ M, K });
            errdefer tensor.free(alloc, A);
            const B = try tensor.zeros(alloc, &.{ K, N });
            errdefer tensor.free(alloc, B);

            A.fillNormal(prng.random(), 0.0, 1.0);
            B.fillNormal(prng.random(), 0.0, 1.0);

            return .{
                .allocator = alloc,
                .arena = std.heap.ArenaAllocator.init(alloc),
                .A = A,
                .B = B,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.arena.deinit();
            tensor.free(self.allocator, self.A);
            tensor.free(self.allocator, self.B);
        }

        pub fn run(self: *@This()) !void {
            _ = self.arena.reset(.retain_capacity);
            const C = try self.A.matmul(self.B, self.arena.allocator(), null);
            std.mem.doNotOptimizeAway(C.data.ptr);
        }
    };

    const sizes = [_][3]usize{
        .{ 64, 128, 64 },
        .{ 256, 256, 256 },
        .{ 512, 512, 512 },
        .{ 1024, 1024, 1024 },
    };

    for (sizes) |s| {
        const M = s[0];
        const K = s[1];
        const N = s[2];
        var ctx = try GemmContext.init(allocator, M, K, N);
        defer ctx.deinit();

        var name_buf: [48]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "GEMM [{d}x{d}x{d}]", .{ M, K, N });

        const flops: f64 = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(K)) * @as(f64, @floatFromInt(N));
        const bytes: u64 = @as(u64, @intCast((M * K + K * N + M * N) * @sizeOf(f32)));

        try runner.benchmark(name, "BLAS/GEMM", flops, bytes, null, "", &ctx);
    }
}

/// Suite 2: Tensor Element-wise & Broadcasting Ops
pub fn runTensorOpBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    // 1. Vector Add 100K & 1M
    const elem_sizes = [_]usize{ 100_000, 1_000_000 };
    for (elem_sizes) |n| {
        const ElemAddContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            A: *tensor.Tensor,
            B: *tensor.Tensor,

            pub fn init(alloc: std.mem.Allocator, size: usize) !@This() {
                var prng = std.Random.DefaultPrng.init(1234);
                const A = try tensor.zeros(alloc, &.{size});
                errdefer tensor.free(alloc, A);
                const B = try tensor.zeros(alloc, &.{size});
                errdefer tensor.free(alloc, B);

                A.fillNormal(prng.random(), 0.0, 1.0);
                B.fillNormal(prng.random(), 0.0, 1.0);

                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .A = A,
                    .B = B,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                tensor.free(self.allocator, self.A);
                tensor.free(self.allocator, self.B);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const C = try self.A.add(self.B, self.arena.allocator(), null);
                std.mem.doNotOptimizeAway(C.data.ptr);
            }
        };

        var ctx = try ElemAddContext.init(allocator, n);
        defer ctx.deinit();

        var name_buf: [48]u8 = undefined;
        const name = if (n >= 1_000_000)
            try std.fmt.bufPrint(&name_buf, "Add [{d}M floats]", .{n / 1_000_000})
        else
            try std.fmt.bufPrint(&name_buf, "Add [{d}K floats]", .{n / 1_000});

        const bytes: u64 = @as(u64, @intCast(n * 3 * @sizeOf(f32)));
        try runner.benchmark(name, "Tensor/Ops", null, bytes, null, "", &ctx);
    }

    // 2. Vector Mul 1M
    {
        const ElemMulContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            A: *tensor.Tensor,
            B: *tensor.Tensor,

            pub fn init(alloc: std.mem.Allocator, size: usize) !@This() {
                const A = try tensor.zeros(alloc, &.{size});
                errdefer tensor.free(alloc, A);
                const B = try tensor.zeros(alloc, &.{size});
                errdefer tensor.free(alloc, B);
                @memset(A.data, 1.5);
                @memset(B.data, 2.0);

                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .A = A,
                    .B = B,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                tensor.free(self.allocator, self.A);
                tensor.free(self.allocator, self.B);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const C = try self.A.mul(self.B, self.arena.allocator(), null);
                std.mem.doNotOptimizeAway(C.data.ptr);
            }
        };

        var ctx = try ElemMulContext.init(allocator, 1_000_000);
        defer ctx.deinit();

        const bytes: u64 = 1_000_000 * 3 * @sizeOf(f32);
        try runner.benchmark("Mul [1M floats]", "Tensor/Ops", null, bytes, null, "", &ctx);
    }

    // 3. Broadcasting 2D: [128, 768] + [1, 768]
    {
        const Broad2DContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            A: *tensor.Tensor,
            B: *tensor.Tensor,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const A = try tensor.zeros(alloc, &.{ 128, 768 });
                errdefer tensor.free(alloc, A);
                const B = try tensor.zeros(alloc, &.{ 1, 768 });
                errdefer tensor.free(alloc, B);
                @memset(A.data, 1.0);
                @memset(B.data, 0.5);

                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .A = A,
                    .B = B,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                tensor.free(self.allocator, self.A);
                tensor.free(self.allocator, self.B);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const C = try self.A.add(self.B, self.arena.allocator(), null);
                std.mem.doNotOptimizeAway(C.data.ptr);
            }
        };

        var ctx = try Broad2DContext.init(allocator);
        defer ctx.deinit();

        const bytes: u64 = (128 * 768 + 768 + 128 * 768) * @sizeOf(f32);
        try runner.benchmark("Broadcast Add [128,768]+[1,768]", "Tensor/Ops", null, bytes, null, "", &ctx);
    }

    // 4. Broadcasting 4D: [2, 4, 32, 64] + [1, 4, 1, 64]
    {
        const Broad4DContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            A: *tensor.Tensor,
            B: *tensor.Tensor,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const A = try tensor.zeros(alloc, &.{ 2, 4, 32, 64 });
                errdefer tensor.free(alloc, A);
                const B = try tensor.zeros(alloc, &.{ 1, 4, 1, 64 });
                errdefer tensor.free(alloc, B);
                @memset(A.data, 1.0);
                @memset(B.data, 0.5);

                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .A = A,
                    .B = B,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                tensor.free(self.allocator, self.A);
                tensor.free(self.allocator, self.B);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const C = try self.A.add(self.B, self.arena.allocator(), null);
                std.mem.doNotOptimizeAway(C.data.ptr);
            }
        };

        var ctx = try Broad4DContext.init(allocator);
        defer ctx.deinit();

        const out_elem = 2 * 4 * 32 * 64;
        const bytes: u64 = (out_elem + (1 * 4 * 1 * 64) + out_elem) * @sizeOf(f32);
        try runner.benchmark("Broadcast Add 4D [2,4,32,64]+[1,4,1,64]", "Tensor/Ops", null, bytes, null, "", &ctx);
    }

    // 5. Transpose 2D [512, 512]
    {
        const Transpose2DContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            A: *tensor.Tensor,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const A = try tensor.zeros(alloc, &.{ 512, 512 });
                @memset(A.data, 1.0);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .A = A,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                tensor.free(self.allocator, self.A);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const B = try self.A.transpose(0, 1, self.arena.allocator(), null);
                std.mem.doNotOptimizeAway(B.data.ptr);
            }
        };

        var ctx = try Transpose2DContext.init(allocator);
        defer ctx.deinit();

        const bytes: u64 = (512 * 512 * 2) * @sizeOf(f32);
        try runner.benchmark("Transpose 2D [512, 512]", "Tensor/Ops", null, bytes, null, "", &ctx);
    }

    // 6. Transpose 4D [4, 8, 64, 32] -> [4, 64, 8, 32]
    {
        const Transpose4DContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            A: *tensor.Tensor,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const A = try tensor.zeros(alloc, &.{ 4, 8, 64, 32 });
                @memset(A.data, 1.0);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .A = A,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                tensor.free(self.allocator, self.A);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const B = try self.A.transpose(1, 2, self.arena.allocator(), null);
                std.mem.doNotOptimizeAway(B.data.ptr);
            }
        };

        var ctx = try Transpose4DContext.init(allocator);
        defer ctx.deinit();

        const num_elems = 4 * 8 * 64 * 32;
        const bytes: u64 = num_elems * 2 * @sizeOf(f32);
        try runner.benchmark("Transpose 4D [4,8,64,32]->[4,64,8,32]", "Tensor/Ops", null, bytes, null, "", &ctx);
    }
}

/// Suite 3: Activations & Normalization Benchmarks
pub fn runActivationBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    const size = 100_000;
    const ActKind = enum { relu, silu, gelu, sigmoid };

    // Helper for unary elementwise activations
    const ActContext = struct {
        allocator: std.mem.Allocator,
        arena: std.heap.ArenaAllocator,
        A: *tensor.Tensor,
        act_type: ActKind,

        pub fn init(alloc: std.mem.Allocator, act: ActKind) !@This() {
            var prng = std.Random.DefaultPrng.init(42);
            const A = try tensor.zeros(alloc, &.{size});
            A.fillNormal(prng.random(), 0.0, 1.0);
            return .{
                .allocator = alloc,
                .arena = std.heap.ArenaAllocator.init(alloc),
                .A = A,
                .act_type = act,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.arena.deinit();
            tensor.free(self.allocator, self.A);
        }

        pub fn run(self: *@This()) !void {
            _ = self.arena.reset(.retain_capacity);
            const alloc = self.arena.allocator();
            const out = switch (self.act_type) {
                .relu => try self.A.relu(alloc, null),
                .silu => try self.A.silu(alloc, null),
                .gelu => try self.A.gelu(alloc, null),
                .sigmoid => try self.A.sigmoid(alloc, null),
            };
            std.mem.doNotOptimizeAway(out.data.ptr);
        }
    };

    const acts = [_]struct { name: []const u8, kind: ActKind }{
        .{ .name = "ReLU [100K floats]", .kind = .relu },
        .{ .name = "GELU [100K floats]", .kind = .gelu },
        .{ .name = "SiLU [100K floats]", .kind = .silu },
        .{ .name = "Sigmoid [100K floats]", .kind = .sigmoid },
    };

    for (acts) |a| {
        var ctx = try ActContext.init(allocator, a.kind);
        defer ctx.deinit();
        const bytes: u64 = size * 2 * @sizeOf(f32);
        try runner.benchmark(a.name, "Activations", null, bytes, null, "", &ctx);
    }

    // Softmax [32, 1024]
    {
        const SoftmaxContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            A: *tensor.Tensor,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                var prng = std.Random.DefaultPrng.init(42);
                const A = try tensor.zeros(alloc, &.{ 32, 1024 });
                A.fillNormal(prng.random(), 0.0, 1.0);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .A = A,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                tensor.free(self.allocator, self.A);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const out = try self.A.softmax(self.arena.allocator(), null);
                std.mem.doNotOptimizeAway(out.data.ptr);
            }
        };

        var ctx = try SoftmaxContext.init(allocator);
        defer ctx.deinit();
        const bytes: u64 = 32 * 1024 * 2 * @sizeOf(f32);
        try runner.benchmark("Softmax [32, 1024]", "Activations", null, bytes, null, "", &ctx);
    }

    // LayerNorm Forward + Backward [32, 512]
    {
        const LNContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            ln: nn.LayerNorm,
            x_data: []f32,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const ln = try nn.LayerNorm.init(alloc, 512, 1e-5);
                const x_data = try alloc.alloc(f32, 32 * 512);
                @memset(x_data, 0.5);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .ln = ln,
                    .x_data = x_data,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.ln.deinit(self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 32, 512 }, self.x_data, true);
                const out = try self.ln.forward(alloc, &graph, x);
                @memset(out.grad, 1.0);
                try graph.backward(out);
                std.mem.doNotOptimizeAway(x.grad.ptr);
            }
        };

        var ctx = try LNContext.init(allocator);
        defer ctx.deinit();
        try runner.benchmark("LayerNorm Fwd+Bwd [32, 512]", "Activations", null, (32 * 512 * 4) * @sizeOf(f32), null, "", &ctx);
    }

    // RMSNorm Forward + Backward [32, 512]
    {
        const RMSContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            rms: nn.RMSNorm,
            x_data: []f32,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const rms = try nn.RMSNorm.init(alloc, 512, 1e-5);
                const x_data = try alloc.alloc(f32, 32 * 512);
                @memset(x_data, 0.5);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .rms = rms,
                    .x_data = x_data,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.rms.deinit(self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 32, 512 }, self.x_data, true);
                const out = try self.rms.forward(alloc, &graph, x);
                @memset(out.grad, 1.0);
                try graph.backward(out);
                std.mem.doNotOptimizeAway(x.grad.ptr);
            }
        };

        var ctx = try RMSContext.init(allocator);
        defer ctx.deinit();
        try runner.benchmark("RMSNorm Fwd+Bwd [32, 512]", "Activations", null, (32 * 512 * 4) * @sizeOf(f32), null, "", &ctx);
    }
}

/// Suite 4: Layers (Forward & Backward)
pub fn runLayerBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(2026);
    const random = prng.random();

    // 1. Linear Forward [64, 784 -> 128]
    {
        const LinFwdContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            linear: nn.Linear,
            x: *tensor.Tensor,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                const linear = try nn.Linear.init(alloc, 784, 128, rnd);
                const x = try tensor.zeros(alloc, &.{ 64, 784 });
                @memset(x.data, 0.1);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .linear = linear,
                    .x = x,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.linear.deinit(self.allocator);
                tensor.free(self.allocator, self.x);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const out = try self.linear.forward(self.arena.allocator(), null, self.x);
                std.mem.doNotOptimizeAway(out.data.ptr);
            }
        };

        var ctx = try LinFwdContext.init(allocator, random);
        defer ctx.deinit();
        const flops: f64 = 2.0 * 64.0 * 784.0 * 128.0;
        try runner.benchmark("Linear Fwd [64, 784->128]", "Layers", flops, null, null, "", &ctx);
    }

    // 2. Linear Forward + Backward [64, 784 -> 128]
    {
        const LinTrainContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            linear: nn.Linear,
            x_data: []f32,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                const linear = try nn.Linear.init(alloc, 784, 128, rnd);
                const x_data = try alloc.alloc(f32, 64 * 784);
                @memset(x_data, 0.1);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .linear = linear,
                    .x_data = x_data,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.linear.deinit(self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 64, 784 }, self.x_data, true);
                const out = try self.linear.forward(alloc, &graph, x);
                @memset(out.grad, 1.0);
                self.linear.zeroGrad();
                try graph.backward(out);
                std.mem.doNotOptimizeAway(self.linear.weight.grad.ptr);
            }
        };

        var ctx = try LinTrainContext.init(allocator, random);
        defer ctx.deinit();
        // Fwd + Bwd = 3 matrix multiplies (fwd, grad_weight, grad_input)
        const flops: f64 = 3.0 * (2.0 * 64.0 * 784.0 * 128.0);
        try runner.benchmark("Linear Fwd+Bwd [64, 784->128]", "Layers", flops, null, null, "", &ctx);
    }

    // 3. Conv2D Forward [32, 1, 28x28 -> 16 3x3]
    {
        const ConvFwdContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            conv: nn.Conv2D,
            x: *tensor.Tensor,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                const conv = try nn.Conv2D.init(alloc, 1, 16, 3, rnd);
                const x = try tensor.zeros(alloc, &.{ 32, 1, 28, 28 });
                @memset(x.data, 0.1);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .conv = conv,
                    .x = x,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.conv.deinit(self.allocator);
                tensor.free(self.allocator, self.x);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const out = try self.conv.forward(self.arena.allocator(), null, self.x);
                std.mem.doNotOptimizeAway(out.data.ptr);
            }
        };

        var ctx = try ConvFwdContext.init(allocator, random);
        defer ctx.deinit();
        // Conv2D FLOPs: 2 * B * out_c * out_h * out_w * in_c * k_h * k_w
        // out_h = 28 - 3 + 1 = 26, out_w = 26
        const flops: f64 = 2.0 * 32.0 * 16.0 * 26.0 * 26.0 * 1.0 * 3.0 * 3.0;
        try runner.benchmark("Conv2D Fwd [32, 1, 28x28, 16 3x3]", "Layers", flops, null, null, "", &ctx);
    }

    // 4. Conv2D Forward + Backward [32, 1, 28x28 -> 16 3x3]
    {
        const ConvTrainContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            conv: nn.Conv2D,
            x_data: []f32,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                const conv = try nn.Conv2D.init(alloc, 1, 16, 3, rnd);
                const x_data = try alloc.alloc(f32, 32 * 1 * 28 * 28);
                @memset(x_data, 0.1);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .conv = conv,
                    .x_data = x_data,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.conv.deinit(self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 32, 1, 28, 28 }, self.x_data, true);
                const out = try self.conv.forward(alloc, &graph, x);
                @memset(out.grad, 1.0);
                self.conv.zeroGrad();
                try graph.backward(out);
                std.mem.doNotOptimizeAway(self.conv.weight.grad.ptr);
            }
        };

        var ctx = try ConvTrainContext.init(allocator, random);
        defer ctx.deinit();
        const flops: f64 = 3.0 * (2.0 * 32.0 * 16.0 * 26.0 * 26.0 * 1.0 * 3.0 * 3.0);
        try runner.benchmark("Conv2D Fwd+Bwd [32, 1, 28, 16]", "Layers", flops, null, null, "", &ctx);
    }

    // 5. MaxPool2D Forward + Backward [32, 16, 26x26, 2x2]
    {
        const PoolContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            x_data: []f32,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const x_data = try alloc.alloc(f32, 32 * 16 * 26 * 26);
                @memset(x_data, 0.5);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .x_data = x_data,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 32, 16, 26, 26 }, self.x_data, true);
                const out = try x.maxpool2d(2, 2, alloc, &graph);
                @memset(out.grad, 1.0);
                try graph.backward(out);
                std.mem.doNotOptimizeAway(x.grad.ptr);
            }
        };

        var ctx = try PoolContext.init(allocator);
        defer ctx.deinit();
        const bytes: u64 = (32 * 16 * 26 * 26 * 2) * @sizeOf(f32);
        try runner.benchmark("MaxPool2D Fwd+Bwd [32, 16, 26x26]", "Layers", null, bytes, null, "", &ctx);
    }

    // 6. SwiGLU Forward + Backward [4, 64, 128 -> hidden 256]
    {
        const SwiGLUContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            swiglu: nn.SwiGLU,
            x_data: []f32,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                const swiglu = try nn.SwiGLU.init(alloc, 128, 256, rnd);
                const x_data = try alloc.alloc(f32, 4 * 64 * 128);
                @memset(x_data, 0.2);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .swiglu = swiglu,
                    .x_data = x_data,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.swiglu.deinit(self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 4, 64, 128 }, self.x_data, true);
                const out = try self.swiglu.forward(alloc, &graph, x);
                @memset(out.grad, 1.0);
                self.swiglu.zeroGrad();
                try graph.backward(out);
                std.mem.doNotOptimizeAway(self.swiglu.w_gate.weight.grad.ptr);
            }
        };

        var ctx = try SwiGLUContext.init(allocator, random);
        defer ctx.deinit();
        const flops: f64 = 3.0 * (6.0 * 4.0 * 64.0 * 128.0 * 256.0);
        try runner.benchmark("SwiGLU Fwd+Bwd [4, 64, 128->256]", "Layers", flops, null, null, "", &ctx);
    }

    // 7. CausalSelfAttention Forward [B=4, S=64, D=128, H=4]
    {
        const AttnFwdContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            attn: nn.CausalSelfAttention,
            x: *tensor.Tensor,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                const attn = try nn.CausalSelfAttention.init(alloc, 128, 4, rnd);
                const x = try tensor.zeros(alloc, &.{ 4, 64, 128 });
                @memset(x.data, 0.1);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .attn = attn,
                    .x = x,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.attn.deinit(self.allocator);
                tensor.free(self.allocator, self.x);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const out = try self.attn.forward(self.arena.allocator(), null, self.x);
                std.mem.doNotOptimizeAway(out.data.ptr);
            }
        };

        var ctx = try AttnFwdContext.init(allocator, random);
        defer ctx.deinit();
        const flops_fwd: f64 = 8.0 * 4.0 * 64.0 * 128.0 * 128.0 + 4.0 * 4.0 * 64.0 * 64.0 * 128.0;
        try runner.benchmark("SelfAttention Fwd [4, 64, 128, 4]", "Layers", flops_fwd, null, null, "", &ctx);
    }

    // 8. CausalSelfAttention Forward + Backward [B=4, S=64, D=128, H=4]
    {
        const AttnTrainContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            attn: nn.CausalSelfAttention,
            x_data: []f32,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                const attn = try nn.CausalSelfAttention.init(alloc, 128, 4, rnd);
                const x_data = try alloc.alloc(f32, 4 * 64 * 128);
                @memset(x_data, 0.1);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .attn = attn,
                    .x_data = x_data,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.attn.deinit(self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 4, 64, 128 }, self.x_data, true);
                const out = try self.attn.forward(alloc, &graph, x);
                @memset(out.grad, 1.0);
                self.attn.zeroGrad();
                try graph.backward(out);
                std.mem.doNotOptimizeAway(self.attn.q_attn.weight.grad.ptr);
            }
        };

        var ctx = try AttnTrainContext.init(allocator, random);
        defer ctx.deinit();
        const flops_train: f64 = 3.0 * (8.0 * 4.0 * 64.0 * 128.0 * 128.0 + 4.0 * 4.0 * 64.0 * 64.0 * 128.0);
        try runner.benchmark("SelfAttention Fwd+Bwd [4, 64, 128]", "Layers", flops_train, null, null, "", &ctx);
    }

    // 9. TransformerBlock Forward + Backward [B=4, S=64, D=128, H=4]
    {
        const BlockTrainContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            block: nn.TransformerBlock,
            x_data: []f32,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                const block = try nn.TransformerBlock.init(alloc, 128, 4, rnd);
                const x_data = try alloc.alloc(f32, 4 * 64 * 128);
                @memset(x_data, 0.1);
                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .block = block,
                    .x_data = x_data,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.block.deinit(self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 4, 64, 128 }, self.x_data, true);
                const out = try self.block.forward(alloc, &graph, x);
                @memset(out.grad, 1.0);
                self.block.zeroGrad();
                try graph.backward(out);
                std.mem.doNotOptimizeAway(self.block.attn.q_attn.weight.grad.ptr);
            }
        };

        var ctx = try BlockTrainContext.init(allocator, random);
        defer ctx.deinit();
        const attn_fwd: f64 = 8.0 * 4.0 * 64.0 * 128.0 * 128.0 + 4.0 * 4.0 * 64.0 * 64.0 * 128.0;
        const mlp_fwd: f64 = 6.0 * 4.0 * 64.0 * 128.0 * 256.0;
        const flops_block: f64 = 3.0 * (attn_fwd + mlp_fwd);
        try runner.benchmark("TransformerBlock Fwd+Bwd [4, 64, 128]", "Layers", flops_block, null, null, "", &ctx);
    }
}

/// Suite 5: End-to-End Model Training Pipeline Step Benchmarks
pub fn runModelBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // 1. MLP Full Step (Batch=64, FashionMNIST architecture: 784 -> 128 -> 64 -> 10 + AdamW)
    {
        const MlpStepContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            fc1: nn.Linear,
            fc2: nn.Linear,
            fc3: nn.Linear,
            opt: optim.AdamWOptimizer,
            x_data: []f32,
            targets: [64]u8,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                var fc1 = try nn.Linear.init(alloc, 784, 128, rnd);
                errdefer fc1.deinit(alloc);
                var fc2 = try nn.Linear.init(alloc, 128, 64, rnd);
                errdefer fc2.deinit(alloc);
                var fc3 = try nn.Linear.init(alloc, 64, 10, rnd);
                errdefer fc3.deinit(alloc);

                const ModelWrap = struct {
                    fc1: nn.Linear,
                    fc2: nn.Linear,
                    fc3: nn.Linear,
                };
                var model = ModelWrap{ .fc1 = fc1, .fc2 = fc2, .fc3 = fc3 };
                const opt = try optim.AdamWOptimizer.init(alloc, &model, .{ .lr = 1e-3 });

                const x_data = try alloc.alloc(f32, 64 * 784);
                @memset(x_data, 0.2);

                var targets: [64]u8 = undefined;
                for (&targets, 0..) |*t, i| {
                    t.* = @as(u8, @intCast(i % 10));
                }

                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .fc1 = fc1,
                    .fc2 = fc2,
                    .fc3 = fc3,
                    .opt = opt,
                    .x_data = x_data,
                    .targets = targets,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.opt.deinit();
                self.fc1.deinit(self.allocator);
                self.fc2.deinit(self.allocator);
                self.fc3.deinit(self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensor(64, 784, false);
                @memcpy(x.data, self.x_data);

                // Forward
                const h1 = try self.fc1.forward(alloc, &graph, x);
                const a1 = try h1.relu(alloc, &graph);
                const h2 = try self.fc2.forward(alloc, &graph, a1);
                const a2 = try h2.relu(alloc, &graph);
                const logits = try self.fc3.forward(alloc, &graph, a2);

                // Loss
                const loss = try graph.softmaxCrossEntropy(logits, &self.targets);

                // Backward & Optimizer
                self.fc1.zeroGrad();
                self.fc2.zeroGrad();
                self.fc3.zeroGrad();
                try graph.backward(loss);
                self.opt.step();

                std.mem.doNotOptimizeAway(self.fc1.weight.data.ptr);
            }
        };

        var ctx = try MlpStepContext.init(allocator, random);
        defer ctx.deinit();
        try runner.benchmark("MLP Step [B=64, 784-128-64-10]", "Models", null, null, 64, "samples/s", &ctx);
    }

    // 2. CNN Step (Batch=32, FashionMNIST architecture: Conv 4 -> Conv 8 -> Conv 16 -> FC + AdamW)
    {
        const CnnStepContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            conv1: nn.Conv2D,
            conv2: nn.Conv2D,
            conv3: nn.Conv2D,
            fc1: nn.Linear,
            opt: optim.AdamWOptimizer,
            x_data: []f32,
            targets: [32]u8,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                var conv1 = try nn.Conv2D.init(alloc, 1, 4, 3, rnd);
                errdefer conv1.deinit(alloc);
                var conv2 = try nn.Conv2D.init(alloc, 4, 8, 3, rnd);
                errdefer conv2.deinit(alloc);
                var conv3 = try nn.Conv2D.init(alloc, 8, 16, 3, rnd);
                errdefer conv3.deinit(alloc);
                var fc1 = try nn.Linear.init(alloc, 144, 10, rnd);
                errdefer fc1.deinit(alloc);

                const ModelWrap = struct {
                    conv1: nn.Conv2D,
                    conv2: nn.Conv2D,
                    conv3: nn.Conv2D,
                    fc1: nn.Linear,
                };
                var model = ModelWrap{ .conv1 = conv1, .conv2 = conv2, .conv3 = conv3, .fc1 = fc1 };
                const opt = try optim.AdamWOptimizer.init(alloc, &model, .{ .lr = 1e-3 });

                const x_data = try alloc.alloc(f32, 32 * 784);
                @memset(x_data, 0.2);

                var targets: [32]u8 = undefined;
                for (&targets, 0..) |*t, i| {
                    t.* = @as(u8, @intCast(i % 10));
                }

                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .conv1 = conv1,
                    .conv2 = conv2,
                    .conv3 = conv3,
                    .fc1 = fc1,
                    .opt = opt,
                    .x_data = x_data,
                    .targets = targets,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.opt.deinit();
                self.conv1.deinit(self.allocator);
                self.conv2.deinit(self.allocator);
                self.conv3.deinit(self.allocator);
                self.fc1.deinit(self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensor(32, 784, false);
                @memcpy(x.data, self.x_data);

                const x_reshaped = try x.reshape(&.{ 32, 1, 28, 28 }, alloc, &graph);

                // Layer 1
                const x1 = try self.conv1.forward(alloc, &graph, x_reshaped);
                const a1 = try x1.relu(alloc, &graph);
                const p1 = try a1.maxpool2d(2, 2, alloc, &graph);

                // Layer 2
                const x2 = try self.conv2.forward(alloc, &graph, p1);
                const a2 = try x2.relu(alloc, &graph);
                const p2 = try a2.maxpool2d(2, 2, alloc, &graph);

                // Layer 3
                const x3 = try self.conv3.forward(alloc, &graph, p2);
                const a3 = try x3.relu(alloc, &graph);

                // Flatten -> Linear
                const flat = try a3.reshape(&.{ 32, 144 }, alloc, &graph);
                const logits = try self.fc1.forward(alloc, &graph, flat);

                // Loss
                const loss = try graph.softmaxCrossEntropy(logits, &self.targets);

                // Backward & Optimizer
                self.conv1.zeroGrad();
                self.conv2.zeroGrad();
                self.conv3.zeroGrad();
                self.fc1.zeroGrad();
                try graph.backward(loss);
                self.opt.step();

                std.mem.doNotOptimizeAway(self.fc1.weight.data.ptr);
            }
        };

        var ctx = try CnnStepContext.init(allocator, random);
        defer ctx.deinit();
        try runner.benchmark("CNN Step [B=32, FashionMNIST]", "Models", null, null, 32, "samples/s", &ctx);
    }

    // 3. TransformerBlock Full Step (Batch=4, SeqLen=64, Dim=128 + AdamW)
    {
        const BlockStepContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            block: nn.TransformerBlock,
            opt: optim.AdamWOptimizer,
            x_data: []f32,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                var block = try nn.TransformerBlock.init(alloc, 128, 4, rnd);
                errdefer block.deinit(alloc);

                const ModelWrap = struct {
                    block: nn.TransformerBlock,
                };
                var model = ModelWrap{ .block = block };
                const opt = try optim.AdamWOptimizer.init(alloc, &model, .{ .lr = 1e-3 });

                const x_data = try alloc.alloc(f32, 4 * 64 * 128);
                @memset(x_data, 0.1);

                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .block = block,
                    .opt = opt,
                    .x_data = x_data,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.opt.deinit();
                self.block.deinit(self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 4, 64, 128 }, self.x_data, true);
                const out = try self.block.forward(alloc, &graph, x);

                @memset(out.grad, 1.0);
                self.block.zeroGrad();
                try graph.backward(out);
                self.opt.step();

                std.mem.doNotOptimizeAway(self.block.attn.q_attn.weight.data.ptr);
            }
        };

        var ctx = try BlockStepContext.init(allocator, random);
        defer ctx.deinit();
        const total_tokens = 4 * 64;
        try runner.benchmark("TransformerBlock Step [B=4, S=64]", "Models", null, null, total_tokens, "tokens/s", &ctx);
    }
}

/// Suite 6: Optimizers Benchmarks
pub fn runOptimizerBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    const num_params: usize = 1_000_000;

    // 1. AdamW Optimizer Step
    {
        const AdamWContext = struct {
            allocator: std.mem.Allocator,
            opt: optim.AdamWOptimizer,
            linear: nn.Linear,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                var prng = std.Random.DefaultPrng.init(42);
                var linear = try nn.Linear.init(alloc, 1000, 1000, prng.random());
                errdefer linear.deinit(alloc);

                @memset(linear.weight.grad, 0.05);
                @memset(linear.bias.grad, 0.01);

                const opt = try optim.AdamWOptimizer.init(alloc, &linear, .{
                    .lr = 1e-3,
                    .weight_decay = 0.01,
                });

                return .{
                    .allocator = alloc,
                    .opt = opt,
                    .linear = linear,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.opt.deinit();
                self.linear.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                self.opt.step();
                std.mem.doNotOptimizeAway(self.linear.weight.data.ptr);
            }
        };

        var ctx = try AdamWContext.init(allocator);
        defer ctx.deinit();
        // AdamW updates w, m, v reading grad: ~4 floats read/write per param
        const bytes: u64 = num_params * 4 * @sizeOf(f32);
        try runner.benchmark("AdamW Step [1M params]", "Optimizers", null, bytes, null, "", &ctx);
    }

    // 2. SGD with Momentum Optimizer Step
    {
        const SgdContext = struct {
            allocator: std.mem.Allocator,
            opt: optim.SGDOptimizer,
            linear: nn.Linear,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                var prng = std.Random.DefaultPrng.init(42);
                var linear = try nn.Linear.init(alloc, 1000, 1000, prng.random());
                errdefer linear.deinit(alloc);

                @memset(linear.weight.grad, 0.05);
                @memset(linear.bias.grad, 0.01);

                const opt = try optim.SGDOptimizer.init(alloc, &linear, .{
                    .lr = 0.01,
                    .momentum = 0.9,
                });

                return .{
                    .allocator = alloc,
                    .opt = opt,
                    .linear = linear,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.opt.deinit();
                self.linear.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                self.opt.step();
                std.mem.doNotOptimizeAway(self.linear.weight.data.ptr);
            }
        };

        var ctx = try SgdContext.init(allocator);
        defer ctx.deinit();
        const bytes: u64 = num_params * 3 * @sizeOf(f32);
        try runner.benchmark("SGD Momentum Step [1M params]", "Optimizers", null, bytes, null, "", &ctx);
    }
}

/// Suite 7: Tokenizer Benchmarks
pub fn runTokenizerBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    const sample_text =
        \\First Citizen:
        \\Before we proceed any further, hear me speak.
        \\
        \\All:
        \\Speak, speak.
        \\
        \\First Citizen:
        \\You are all resolved rather to die than to famish?
        \\
        \\All:
        \\Resolved. resolved.
        \\
        \\First Citizen:
        \\First, you know Caius Marcius is chief enemy to the people.
        \\
        \\All:
        \\We know't, we know't.
        \\
        \\First Citizen:
        \\Let us kill him, and we'll have corn at our own price.
        \\Is't a verdict?
        \\
        \\All:
        \\No more talking on't; let it be done: away, away!
        \\
        \\Second Citizen:
        \\One word, good citizens.
    ;

    var tok = try dataset.BPETokenizer.init(allocator);
    defer tok.deinit();

    // Populate common English subword merges
    const common_merges = [_][2][]const u8{
        .{ "t", "h" }, .{ "th", "e" }, .{ "i", "n" }, .{ "e", "r" },
        .{ "a", "n" }, .{ "r", "e" },  .{ "o", "n" }, .{ "a", "t" },
        .{ "e", "n" }, .{ "e", "s" },  .{ "o", "r" }, .{ "t", "e" },
        .{ " ", "t" }, .{ " ", "a" },  .{ " ", "w" }, .{ " ", "b" },
        .{ "C", "i" }, .{ "Ci", "t" }, .{ "Cit", "i" }, .{ "Citi", "z" },
        .{ "Citiz", "e" }, .{ "Citize", "n" },
    };
    for (common_merges, 0..) |m, rank| {
        try tok.addMerge(m[0], m[1], @as(u32, @intCast(rank)));
    }

    // 1. Encode Benchmark
    {
        const EncodeContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            tokenizer: *const dataset.BPETokenizer,
            text: []const u8,

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const tokens = try self.tokenizer.encode(self.arena.allocator(), self.text);
                std.mem.doNotOptimizeAway(tokens.ptr);
            }
        };

        var ctx = EncodeContext{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tokenizer = &tok,
            .text = sample_text,
        };
        defer ctx.arena.deinit();

        const bytes: u64 = sample_text.len;
        try runner.benchmark("BPETokenizer Encode [Sample Text]", "Tokenizer", null, bytes, null, "", &ctx);
    }

    // 2. Decode Benchmark
    {
        const encoded_tokens = try tok.encode(allocator, sample_text);
        defer allocator.free(encoded_tokens);

        const DecodeContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            tokenizer: *const dataset.BPETokenizer,
            tokens: []const dataset.TokenId,

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const decoded = try self.tokenizer.decode(self.arena.allocator(), self.tokens);
                std.mem.doNotOptimizeAway(decoded.ptr);
            }
        };

        var ctx = DecodeContext{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tokenizer = &tok,
            .tokens = encoded_tokens,
        };
        defer ctx.arena.deinit();

        const bytes: u64 = sample_text.len;
        try runner.benchmark("BPETokenizer Decode [Sample Text]", "Tokenizer", null, bytes, null, "", &ctx);
    }
}

/// Run all benchmark suites
pub fn runAllBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    try runGemmBenchmarks(runner, allocator);
    try runTensorOpBenchmarks(runner, allocator);
    try runActivationBenchmarks(runner, allocator);
    try runLayerBenchmarks(runner, allocator);
    try runModelBenchmarks(runner, allocator);
    try runOptimizerBenchmarks(runner, allocator);
    try runTokenizerBenchmarks(runner, allocator);
}

test "bench runner basic execution" {
    const test_alloc = std.testing.allocator;
    var runner = BenchmarkRunner.init(test_alloc, .{
        .warmup = 1,
        .iterations = 2,
        .filter = "GEMM [64x128x64]",
        .quiet = true,
    });
    defer runner.deinit();

    try runGemmBenchmarks(&runner, test_alloc);
    try std.testing.expectEqual(@as(usize, 1), runner.results.items.len);
    const res = runner.results.items[0];
    try std.testing.expect(res.min_ns > 0);
    try std.testing.expect(res.avg_ns >= res.min_ns);
    try std.testing.expect(res.gflops() != null);
}
