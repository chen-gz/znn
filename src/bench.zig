const std = @import("std");
const builtin = @import("builtin");
const tensor = @import("tensor.zig");
const nn = @import("nn.zig");
const autodiff = @import("autodiff.zig");
const optim = @import("optim.zig");
const dataset = @import("dataset.zig");
const engine = @import("engine.zig");

const runner_mod = @import("bench/runner.zig");
pub const BenchmarkStats = runner_mod.BenchmarkStats;
pub const getTimeNs = runner_mod.getTimeNs;
pub const BenchmarkConfig = runner_mod.BenchmarkConfig;
pub const BenchmarkRunner = runner_mod.BenchmarkRunner;

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
                nn.deinitModel(&self.ln, self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.initDefault(alloc);
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
                nn.deinitModel(&self.rms, self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.initDefault(alloc);
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
pub const runExecutorBenchmarks = suites.runExecutorBenchmarks;
pub const runQ4Benchmarks = suites.runQ4Benchmarks;
pub const runDecodeBenchmarks = suites.runDecodeBenchmarks;

/// Run all benchmark suites
pub fn runAllBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    try runGemmBenchmarks(runner, allocator);
    try runTensorOpBenchmarks(runner, allocator);
    try runActivationBenchmarks(runner, allocator);
    try runLayerBenchmarks(runner, allocator);
    try runModelBenchmarks(runner, allocator);
    try runOptimizerBenchmarks(runner, allocator);
    try runTokenizerBenchmarks(runner, allocator);
    try runExecutorBenchmarks(runner, allocator);
    try runQ4Benchmarks(runner, allocator);
    try runDecodeBenchmarks(runner, allocator);
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
