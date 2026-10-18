//! Apple 机器学习拓展 (Machine Learning Explore, MLX) 硬件加速基准测试套件
//! 测量图形处理器 (Graphics Processing Unit, GPU) 矩阵乘法、注意力与归一化算子性能。

const std = @import("std");
const backend = @import("../backend.zig");
const runner_mod = @import("runner.zig");
const BenchmarkRunner = runner_mod.BenchmarkRunner;

/// Suite 11: Apple 机器学习拓展 (Machine Learning Explore, MLX) GPU 加速基准测试
pub fn runMlxBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    _ = allocator;
    if (!backend.mlx.has_mlx) return;

    var stream = backend.mlx.Stream.defaultGpu() catch return;
    defer stream.deinit();

    // 1. MLX GPU GEMM [1024x1024x1024]
    {
        const MlxGemmCtx = struct {
            stream: backend.mlx.Stream,
            a: backend.mlx.Array,
            b: backend.mlx.Array,
            raw_a: []f32,
            raw_b: []f32,
            alloc: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator, s: backend.mlx.Stream) !@This() {
                const n = 1024;
                const raw_a = try alloc.alloc(f32, n * n);
                errdefer alloc.free(raw_a);
                @memset(raw_a, 0.5);

                const raw_b = try alloc.alloc(f32, n * n);
                errdefer alloc.free(raw_b);
                @memset(raw_b, 0.25);

                const a = try backend.mlx.Array.fromSlice(raw_a, &.{ n, n });
                const b = try backend.mlx.Array.fromSlice(raw_b, &.{ n, n });
                a.eval();
                b.eval();

                return .{
                    .stream = s,
                    .a = a,
                    .b = b,
                    .raw_a = raw_a,
                    .raw_b = raw_b,
                    .alloc = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.a.deinit();
                self.b.deinit();
                self.alloc.free(self.raw_a);
                self.alloc.free(self.raw_b);
            }

            pub fn run(self: *@This()) !void {
                var c = try backend.mlx.Array.matmul(self.a, self.b, self.stream);
                defer c.deinit();
                c.eval();
                self.stream.synchronize();
                const out = c.dataFloat32();
                std.mem.doNotOptimizeAway(out.ptr);
            }
        };

        var ctx = try MlxGemmCtx.init(std.heap.page_allocator, stream);
        defer ctx.deinit();
        const flops: f64 = 2.0 * 1024.0 * 1024.0 * 1024.0;
        try runner.benchmark("MLX GEMM [1024x1024x1024, GPU]", "MLX", flops, null, null, "", &ctx);
    }

    // 2. MLX Fast SDPA [B=1, H=8, L=512, D=64]
    {
        const MlxSdpaCtx = struct {
            stream: backend.mlx.Stream,
            q: backend.mlx.Array,
            k: backend.mlx.Array,
            v: backend.mlx.Array,
            raw: []f32,
            alloc: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator, s: backend.mlx.Stream) !@This() {
                const total = 1 * 8 * 512 * 64;
                const raw = try alloc.alloc(f32, total);
                errdefer alloc.free(raw);
                @memset(raw, 0.05);

                const q = try backend.mlx.Array.fromSlice(raw, &.{ 1, 8, 512, 64 });
                const k = try backend.mlx.Array.fromSlice(raw, &.{ 1, 8, 512, 64 });
                const v = try backend.mlx.Array.fromSlice(raw, &.{ 1, 8, 512, 64 });
                q.eval();
                k.eval();
                v.eval();

                return .{
                    .stream = s,
                    .q = q,
                    .k = k,
                    .v = v,
                    .raw = raw,
                    .alloc = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.q.deinit();
                self.k.deinit();
                self.v.deinit();
                self.alloc.free(self.raw);
            }

            pub fn run(self: *@This()) !void {
                const scale: f32 = 1.0 / @sqrt(64.0);
                var out = try backend.mlx.Array.fastScaledDotProductAttention(self.q, self.k, self.v, scale, true, self.stream);
                defer out.deinit();
                out.eval();
                self.stream.synchronize();
                const ptr = out.dataFloat32();
                std.mem.doNotOptimizeAway(ptr.ptr);
            }
        };

        var ctx = try MlxSdpaCtx.init(std.heap.page_allocator, stream);
        defer ctx.deinit();
        const flops: f64 = 4.0 * 1.0 * 8.0 * 512.0 * 512.0 * 64.0;
        try runner.benchmark("MLX SDPA Causal [1, 8, 512, 64, GPU]", "MLX", flops, null, 512, "tokens/s", &ctx);
    }

    // 3. MLX Fast RMSNorm [1024, 2048]
    {
        const MlxRmsCtx = struct {
            stream: backend.mlx.Stream,
            x: backend.mlx.Array,
            raw: []f32,
            alloc: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator, s: backend.mlx.Stream) !@This() {
                const total = 1024 * 2048;
                const raw = try alloc.alloc(f32, total);
                errdefer alloc.free(raw);
                @memset(raw, 0.33);

                const x = try backend.mlx.Array.fromSlice(raw, &.{ 1024, 2048 });
                x.eval();

                return .{
                    .stream = s,
                    .x = x,
                    .raw = raw,
                    .alloc = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.x.deinit();
                self.alloc.free(self.raw);
            }

            pub fn run(self: *@This()) !void {
                var out = try backend.mlx.Array.fastRmsNorm(self.x, null, 1e-5, self.stream);
                defer out.deinit();
                out.eval();
                self.stream.synchronize();
                const ptr = out.dataFloat32();
                std.mem.doNotOptimizeAway(ptr.ptr);
            }
        };

        var ctx = try MlxRmsCtx.init(std.heap.page_allocator, stream);
        defer ctx.deinit();
        const bytes: u64 = 1024 * 2048 * @sizeOf(f32) * 2;
        try runner.benchmark("MLX RMSNorm [1024x2048, GPU]", "MLX", null, bytes, null, "", &ctx);
    }
}
