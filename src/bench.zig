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
            const C = try self.A.matmul(self.B, self.arena.allocator());
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
                const C = try self.A.add(self.B, self.arena.allocator());
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
                const C = try self.A.mul(self.B, self.arena.allocator());
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
                const C = try self.A.add(self.B, self.arena.allocator());
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
                const C = try self.A.add(self.B, self.arena.allocator());
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
                const B = try self.A.transpose(0, 1, self.arena.allocator());
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
                const B = try self.A.transpose(1, 2, self.arena.allocator());
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
                .relu => try self.A.relu(alloc),
                .silu => try self.A.silu(alloc),
                .gelu => try self.A.gelu(alloc),
                .sigmoid => try self.A.sigmoid(alloc),
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
                const out = try self.A.softmax(self.arena.allocator());
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
                const out = try self.ln.forward(&graph, x);
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
                const out = try self.rms.forward(&graph, x);
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

pub const suites = @import("bench/suites.zig");
pub const runLayerBenchmarks = suites.runLayerBenchmarks;
pub const runModelBenchmarks = suites.runModelBenchmarks;
pub const runOptimizerBenchmarks = suites.runOptimizerBenchmarks;
pub const runTokenizerBenchmarks = suites.runTokenizerBenchmarks;

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
