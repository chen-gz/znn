//! Apple 机器学习拓展 (Machine Learning Explore, MLX) 硬件加速基准测试套件
//! 对比中央处理器 (Central Processing Unit, CPU) 与统一内存图形处理器 (Graphics Processing Unit, GPU) 性能。

const std = @import("std");
const backend = @import("../backend.zig");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const nn = @import("../nn.zig");
const testing_init = @import("../nn/testing_init.zig");
const runner_mod = @import("runner.zig");
const BenchmarkRunner = runner_mod.BenchmarkRunner;
const BenchmarkStats = runner_mod.BenchmarkStats;

pub const ComparisonRow = struct {
    name: []const u8,
    cpu_ms: f64,
    gpu_ms: f64,
    speedup: f64,
    gpu_metric: []const u8,
    max_error: f32,
    passed: bool,
};

/// Suite 11: Apple 机器学习拓展 (Machine Learning Explore, MLX) 对比基准测试
pub fn runMlxBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    if (!backend.mlx.has_mlx) return;

    var stream = backend.mlx.Stream.defaultGpu() catch return;
    defer stream.deinit();

    var prng = std.Random.DefaultPrng.init(2026);
    const random = prng.random();

    var comparisons = std.ArrayList(ComparisonRow).empty;
    defer comparisons.deinit(allocator);

    // =========================================================================
    // 1. GEMM 矩阵乘法对比 [1024x1024x1024]
    // =========================================================================
    {
        const n: usize = 1024;
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(n * n * n));

        const CpuGemmCtx = struct {
            raw_a: []f32,
            raw_b: []f32,
            raw_c: []f32,
            alloc: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const raw_a = try alloc.alloc(f32, n * n);
                errdefer alloc.free(raw_a);
                @memset(raw_a, 0.5);

                const raw_b = try alloc.alloc(f32, n * n);
                errdefer alloc.free(raw_b);
                @memset(raw_b, 0.25);

                const raw_c = try alloc.alloc(f32, n * n);
                errdefer alloc.free(raw_c);

                return .{
                    .raw_a = raw_a,
                    .raw_b = raw_b,
                    .raw_c = raw_c,
                    .alloc = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.alloc.free(self.raw_a);
                self.alloc.free(self.raw_b);
                self.alloc.free(self.raw_c);
            }

            pub fn run(self: *@This()) !void {
                backend.cblas.cblas_sgemm(
                    backend.cblas.CblasRowMajor,
                    backend.cblas.CblasNoTrans,
                    backend.cblas.CblasNoTrans,
                    @intCast(n),
                    @intCast(n),
                    @intCast(n),
                    1.0,
                    self.raw_a.ptr,
                    @intCast(n),
                    self.raw_b.ptr,
                    @intCast(n),
                    0.0,
                    self.raw_c.ptr,
                    @intCast(n),
                );
                std.mem.doNotOptimizeAway(self.raw_c.ptr);
            }
        };

        const GpuGemmCtx = struct {
            stream: backend.mlx.Stream,
            a: backend.mlx.Array,
            b: backend.mlx.Array,
            raw_a: []f32,
            raw_b: []f32,
            alloc: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator, s: backend.mlx.Stream) !@This() {
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
                const out = try c.dataFloat32();
                std.mem.doNotOptimizeAway(out.ptr);
            }
        };

        var cpu_ctx = try CpuGemmCtx.init(allocator);
        defer cpu_ctx.deinit();
        var gpu_ctx = try GpuGemmCtx.init(allocator, stream);
        defer gpu_ctx.deinit();

        // 预先数值一致性校验 (Numerical Correctness Check)
        try cpu_ctx.run();
        var c_gpu = try backend.mlx.Array.matmul(gpu_ctx.a, gpu_ctx.b, stream);
        defer c_gpu.deinit();
        c_gpu.eval();
        stream.synchronize();
        var max_diff: f32 = 0.0;
        for (cpu_ctx.raw_c, try c_gpu.dataFloat32()) |vc, vg| {
            const diff = @abs(vc - vg);
            if (diff > max_diff) max_diff = diff;
        }

        const idx_start = runner.results.items.len;
        try runner.benchmark("CPU GEMM [1024x1024, CBLAS]", "MLX", flops, null, null, "", &cpu_ctx);
        try runner.benchmark("MLX GPU GEMM [1024x1024, GPU]", "MLX", flops, null, null, "", &gpu_ctx);

        if (runner.results.items.len >= idx_start + 2) {
            const cpu_st = runner.results.items[idx_start];
            const gpu_st = runner.results.items[idx_start + 1];
            const speedup = if (gpu_st.avgMs() > 0) cpu_st.avgMs() / gpu_st.avgMs() else 1.0;
            const gflops_val = gpu_st.gflops() orelse 0.0;
            const metric_str = try std.fmt.allocPrint(allocator, "{d:.1} GFLOPS", .{gflops_val});

            try comparisons.append(allocator, .{
                .name = "GEMM [1024x1024x1024]",
                .cpu_ms = cpu_st.avgMs(),
                .gpu_ms = gpu_st.avgMs(),
                .speedup = speedup,
                .gpu_metric = metric_str,
                .max_error = max_diff,
                .passed = (max_diff < 1e-4),
            });
        }
    }

    // =========================================================================
    // 2. 缩放点积注意力因果对比 (SDPA Causal Attention) [1, 8, 512, 64]
    // =========================================================================
    {
        const B = 1;
        const H = 8;
        const L = 512;
        const D = 64;
        const total = B * H * L * D;
        const flops: f64 = 4.0 * @as(f64, @floatFromInt(B * H * L * L * D));
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));

        const SdpaCtx = struct {
            alloc: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            stream: backend.mlx.Stream,
            device: backend.mlx.DeviceType,
            q_raw: []f32,
            k_raw: []f32,
            v_raw: []f32,

            pub fn init(alloc: std.mem.Allocator, s: backend.mlx.Stream, dev: backend.mlx.DeviceType) !@This() {
                const q_raw = try alloc.alloc(f32, total);
                errdefer alloc.free(q_raw);
                const k_raw = try alloc.alloc(f32, total);
                errdefer alloc.free(k_raw);
                const v_raw = try alloc.alloc(f32, total);
                errdefer alloc.free(v_raw);

                for (0..total) |i| {
                    const val = @sin(@as(f32, @floatFromInt(i)) * 0.03);
                    q_raw[i] = val;
                    k_raw[i] = val * 0.9;
                    v_raw[i] = val * 1.1;
                }

                return .{
                    .alloc = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .stream = s,
                    .device = dev,
                    .q_raw = q_raw,
                    .k_raw = k_raw,
                    .v_raw = v_raw,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.alloc.free(self.q_raw);
                self.alloc.free(self.k_raw);
                self.alloc.free(self.v_raw);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var g = autodiff.Graph.init(self.arena.allocator(), .{
                    .enable_grad = false,
                    .device = self.device,
                    .mlx_stream = if (self.device == .gpu) self.stream else null,
                });
                defer g.deinit();

                const q = try g.tensorNDWithData(&.{ B, H, L, D }, self.q_raw, false);
                const k = try g.tensorNDWithData(&.{ B, H, L, D }, self.k_raw, false);
                const v = try g.tensorNDWithData(&.{ B, H, L, D }, self.v_raw, false);

                const out = try g.scaledDotProductAttention(q, k, v, scale, true);
                if (self.device == .gpu) self.stream.synchronize();
                std.mem.doNotOptimizeAway(out.data.ptr);
            }
        };

        var cpu_sdpa = try SdpaCtx.init(allocator, stream, .cpu);
        defer cpu_sdpa.deinit();
        var gpu_sdpa = try SdpaCtx.init(allocator, stream, .gpu);
        defer gpu_sdpa.deinit();

        // 预先数值一致性校验
        try cpu_sdpa.run();
        try gpu_sdpa.run();

        // 从单次运行提取两者误差
        var g_c = autodiff.Graph.init(allocator, .{ .enable_grad = false, .device = .cpu });
        defer g_c.deinit();
        var g_g = autodiff.Graph.init(allocator, .{ .enable_grad = false, .device = .gpu, .mlx_stream = stream });
        defer g_g.deinit();
        const qc = try g_c.tensorNDWithData(&.{ B, H, L, D }, cpu_sdpa.q_raw, false);
        const kc = try g_c.tensorNDWithData(&.{ B, H, L, D }, cpu_sdpa.k_raw, false);
        const vc = try g_c.tensorNDWithData(&.{ B, H, L, D }, cpu_sdpa.v_raw, false);
        const qg = try g_g.tensorNDWithData(&.{ B, H, L, D }, gpu_sdpa.q_raw, false);
        const kg = try g_g.tensorNDWithData(&.{ B, H, L, D }, gpu_sdpa.k_raw, false);
        const vg = try g_g.tensorNDWithData(&.{ B, H, L, D }, gpu_sdpa.v_raw, false);
        const out_c = try g_c.scaledDotProductAttention(qc, kc, vc, scale, true);
        const out_g = try g_g.scaledDotProductAttention(qg, kg, vg, scale, true);
        stream.synchronize();

        var max_diff: f32 = 0.0;
        for (out_c.data, out_g.data) |v1, v2| {
            const diff = @abs(v1 - v2);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }

        const idx_start = runner.results.items.len;
        try runner.benchmark("CPU SDPA Causal [1, 8, 512, 64]", "MLX", flops, null, L, "tokens/s", &cpu_sdpa);
        try runner.benchmark("MLX GPU SDPA Causal [1, 8, 512, 64, GPU]", "MLX", flops, null, L, "tokens/s", &gpu_sdpa);

        if (runner.results.items.len >= idx_start + 2) {
            const cpu_st = runner.results.items[idx_start];
            const gpu_st = runner.results.items[idx_start + 1];
            const speedup = if (gpu_st.avgMs() > 0) cpu_st.avgMs() / gpu_st.avgMs() else 1.0;
            const gflops_val = gpu_st.gflops() orelse 0.0;
            const metric_str = try std.fmt.allocPrint(allocator, "{d:.1} GFLOPS", .{gflops_val});

            try comparisons.append(allocator, .{
                .name = "SDPA Causal [1, 8, 512, 64]",
                .cpu_ms = cpu_st.avgMs(),
                .gpu_ms = gpu_st.avgMs(),
                .speedup = speedup,
                .gpu_metric = metric_str,
                .max_error = max_diff,
                .passed = (max_diff < 1e-3),
            });
        }
    }

    // =========================================================================
    // 3. 长上下文注意力对比 (SDPA Long-Context) [1, 8, 1024, 64]
    // =========================================================================
    {
        const B = 1;
        const H = 8;
        const L = 1024;
        const D = 64;
        const total = B * H * L * D;
        const flops: f64 = 4.0 * @as(f64, @floatFromInt(B * H * L * L * D));
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));

        const SdpaLongCtx = struct {
            alloc: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            stream: backend.mlx.Stream,
            device: backend.mlx.DeviceType,
            q_raw: []f32,
            k_raw: []f32,
            v_raw: []f32,

            pub fn init(alloc: std.mem.Allocator, s: backend.mlx.Stream, dev: backend.mlx.DeviceType) !@This() {
                const q_raw = try alloc.alloc(f32, total);
                errdefer alloc.free(q_raw);
                const k_raw = try alloc.alloc(f32, total);
                errdefer alloc.free(k_raw);
                const v_raw = try alloc.alloc(f32, total);
                errdefer alloc.free(v_raw);

                for (0..total) |i| {
                    const val = @sin(@as(f32, @floatFromInt(i)) * 0.02);
                    q_raw[i] = val;
                    k_raw[i] = val * 0.9;
                    v_raw[i] = val * 1.1;
                }

                return .{
                    .alloc = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .stream = s,
                    .device = dev,
                    .q_raw = q_raw,
                    .k_raw = k_raw,
                    .v_raw = v_raw,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.alloc.free(self.q_raw);
                self.alloc.free(self.k_raw);
                self.alloc.free(self.v_raw);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var g = autodiff.Graph.init(self.arena.allocator(), .{
                    .enable_grad = false,
                    .device = self.device,
                    .mlx_stream = if (self.device == .gpu) self.stream else null,
                });
                defer g.deinit();

                const q = try g.tensorNDWithData(&.{ B, H, L, D }, self.q_raw, false);
                const k = try g.tensorNDWithData(&.{ B, H, L, D }, self.k_raw, false);
                const v = try g.tensorNDWithData(&.{ B, H, L, D }, self.v_raw, false);

                const out = try g.scaledDotProductAttention(q, k, v, scale, true);
                if (self.device == .gpu) self.stream.synchronize();
                std.mem.doNotOptimizeAway(out.data.ptr);
            }
        };

        var cpu_long = try SdpaLongCtx.init(allocator, stream, .cpu);
        defer cpu_long.deinit();
        var gpu_long = try SdpaLongCtx.init(allocator, stream, .gpu);
        defer gpu_long.deinit();

        // 预先数值一致性校验
        var g_c = autodiff.Graph.init(allocator, .{ .enable_grad = false, .device = .cpu });
        defer g_c.deinit();
        var g_g = autodiff.Graph.init(allocator, .{ .enable_grad = false, .device = .gpu, .mlx_stream = stream });
        defer g_g.deinit();
        const qc = try g_c.tensorNDWithData(&.{ B, H, L, D }, cpu_long.q_raw, false);
        const kc = try g_c.tensorNDWithData(&.{ B, H, L, D }, cpu_long.k_raw, false);
        const vc = try g_c.tensorNDWithData(&.{ B, H, L, D }, cpu_long.v_raw, false);
        const qg = try g_g.tensorNDWithData(&.{ B, H, L, D }, gpu_long.q_raw, false);
        const kg = try g_g.tensorNDWithData(&.{ B, H, L, D }, gpu_long.k_raw, false);
        const vg = try g_g.tensorNDWithData(&.{ B, H, L, D }, gpu_long.v_raw, false);
        const out_c = try g_c.scaledDotProductAttention(qc, kc, vc, scale, true);
        const out_g = try g_g.scaledDotProductAttention(qg, kg, vg, scale, true);
        stream.synchronize();

        var max_diff: f32 = 0.0;
        for (out_c.data, out_g.data) |v1, v2| {
            const diff = @abs(v1 - v2);
            if (diff > max_diff) max_diff = diff;
        }

        const idx_start = runner.results.items.len;
        try runner.benchmark("CPU SDPA Long [1, 8, 1024, 64]", "MLX", flops, null, L, "tokens/s", &cpu_long);
        try runner.benchmark("MLX GPU SDPA Long [1, 8, 1024, 64, GPU]", "MLX", flops, null, L, "tokens/s", &gpu_long);

        if (runner.results.items.len >= idx_start + 2) {
            const cpu_st = runner.results.items[idx_start];
            const gpu_st = runner.results.items[idx_start + 1];
            const speedup = if (gpu_st.avgMs() > 0) cpu_st.avgMs() / gpu_st.avgMs() else 1.0;
            const gflops_val = gpu_st.gflops() orelse 0.0;
            const metric_str = try std.fmt.allocPrint(allocator, "{d:.1} GFLOPS", .{gflops_val});

            try comparisons.append(allocator, .{
                .name = "SDPA Long-Context [T=1024]",
                .cpu_ms = cpu_st.avgMs(),
                .gpu_ms = gpu_st.avgMs(),
                .speedup = speedup,
                .gpu_metric = metric_str,
                .max_error = max_diff,
                .passed = (max_diff < 1e-3),
            });
        }
    }

    // =========================================================================
    // 4. RMSNorm 归一化算子对比 [1024x2048]
    // =========================================================================
    {
        const M = 1024;
        const D = 2048;
        const total = M * D;
        const bytes_val: u64 = total * @sizeOf(f32) * 2;

        const RmsCtx = struct {
            alloc: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            stream: backend.mlx.Stream,
            device: backend.mlx.DeviceType,
            x_raw: []f32,
            w_raw: []f32,

            pub fn init(alloc: std.mem.Allocator, s: backend.mlx.Stream, dev: backend.mlx.DeviceType) !@This() {
                const x_raw = try alloc.alloc(f32, total);
                errdefer alloc.free(x_raw);
                const w_raw = try alloc.alloc(f32, D);
                errdefer alloc.free(w_raw);

                for (0..total) |i| {
                    x_raw[i] = @as(f32, @floatFromInt(i % 17)) * 0.1;
                }
                @memset(w_raw, 1.0);

                return .{
                    .alloc = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .stream = s,
                    .device = dev,
                    .x_raw = x_raw,
                    .w_raw = w_raw,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.alloc.free(self.x_raw);
                self.alloc.free(self.w_raw);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var g = autodiff.Graph.init(self.arena.allocator(), .{
                    .enable_grad = false,
                    .device = self.device,
                    .mlx_stream = if (self.device == .gpu) self.stream else null,
                });
                defer g.deinit();

                const x = try g.tensorNDWithData(&.{ M, D }, self.x_raw, false);
                const w = try g.tensorNDWithData(&.{D}, self.w_raw, false);
                const out = try g.rmsNorm(x, w, 1e-5);
                if (self.device == .gpu) self.stream.synchronize();
                std.mem.doNotOptimizeAway(out.data.ptr);
            }
        };

        var cpu_rms = try RmsCtx.init(allocator, stream, .cpu);
        defer cpu_rms.deinit();
        var gpu_rms = try RmsCtx.init(allocator, stream, .gpu);
        defer gpu_rms.deinit();

        // 校验数值误差
        var g_c = autodiff.Graph.init(allocator, .{ .enable_grad = false, .device = .cpu });
        defer g_c.deinit();
        var g_g = autodiff.Graph.init(allocator, .{ .enable_grad = false, .device = .gpu, .mlx_stream = stream });
        defer g_g.deinit();
        const xc = try g_c.tensorNDWithData(&.{ M, D }, cpu_rms.x_raw, false);
        const wc = try g_c.tensorNDWithData(&.{D}, cpu_rms.w_raw, false);
        const xg = try g_g.tensorNDWithData(&.{ M, D }, gpu_rms.x_raw, false);
        const wg = try g_g.tensorNDWithData(&.{D}, gpu_rms.w_raw, false);
        const out_c = try g_c.rmsNorm(xc, wc, 1e-5);
        const out_g = try g_g.rmsNorm(xg, wg, 1e-5);
        stream.synchronize();

        var max_diff: f32 = 0.0;
        for (out_c.data, out_g.data) |v1, v2| {
            const diff = @abs(v1 - v2);
            if (diff > max_diff) max_diff = diff;
        }

        const idx_start = runner.results.items.len;
        try runner.benchmark("CPU RMSNorm [1024x2048]", "MLX", null, bytes_val, null, "", &cpu_rms);
        try runner.benchmark("MLX GPU RMSNorm [1024x2048, GPU]", "MLX", null, bytes_val, null, "", &gpu_rms);

        if (runner.results.items.len >= idx_start + 2) {
            const cpu_st = runner.results.items[idx_start];
            const gpu_st = runner.results.items[idx_start + 1];
            const speedup = if (gpu_st.avgMs() > 0) cpu_st.avgMs() / gpu_st.avgMs() else 1.0;
            const gb_s = gpu_st.throughputGBs() orelse 0.0;
            const metric_str = try std.fmt.allocPrint(allocator, "{d:.1} GB/s", .{gb_s});

            try comparisons.append(allocator, .{
                .name = "RMSNorm [1024x2048]",
                .cpu_ms = cpu_st.avgMs(),
                .gpu_ms = gpu_st.avgMs(),
                .speedup = speedup,
                .gpu_metric = metric_str,
                .max_error = max_diff,
                .passed = (max_diff < 1e-4),
            });
        }
    }

    // =========================================================================
    // 5. 端到端注意力模块对比 (CausalSelfAttention Fwd) [B=4, T=128, D=256, H=4]
    // =========================================================================
    {
        const B = 4;
        const T = 128;
        const D = 256;
        const H = 4;
        const total_tokens = B * T;
        const flops_attn: f64 = 8.0 * @as(f64, @floatFromInt(B * T * D * D)) + 4.0 * @as(f64, @floatFromInt(B * H * T * T * (D / H)));

        var attn = try nn.CausalSelfAttention.init(allocator, D, H);
        defer nn.deinitModel(&attn, allocator);
        try testing_init.initFromOnes(&attn, allocator, random, &.{ 1, 2, D });

        const x = try tensor.zeros(allocator, &.{ B, T, D });
        defer tensor.free(allocator, x);
        for (x.data, 0..) |*val, i| {
            val.* = @sin(@as(f32, @floatFromInt(i)) * 0.04);
        }

        const AttnModCtx = struct {
            arena: std.heap.ArenaAllocator,
            attn: *const nn.CausalSelfAttention,
            stream: backend.mlx.Stream,
            device: backend.mlx.DeviceType,
            x: *tensor.Tensor,

            pub fn init(alloc: std.mem.Allocator, a: *const nn.CausalSelfAttention, in_x: *tensor.Tensor, s: backend.mlx.Stream, dev: backend.mlx.DeviceType) @This() {
                return .{
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .attn = a,
                    .stream = s,
                    .device = dev,
                    .x = in_x,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var g = autodiff.Graph.init(self.arena.allocator(), .{
                    .enable_grad = false,
                    .device = self.device,
                    .mlx_stream = if (self.device == .gpu) self.stream else null,
                });
                defer g.deinit();

                const y = try self.attn.forward(&g, self.x);
                if (self.device == .gpu) self.stream.synchronize();
                std.mem.doNotOptimizeAway(y.data.ptr);
            }
        };

        var cpu_mod = AttnModCtx.init(allocator, &attn, x, stream, .cpu);
        defer cpu_mod.deinit();
        var gpu_mod = AttnModCtx.init(allocator, &attn, x, stream, .gpu);
        defer gpu_mod.deinit();

        // 校验整层端到端数值精度
        var g_c = autodiff.Graph.init(allocator, .{ .enable_grad = false, .device = .cpu });
        defer g_c.deinit();
        var g_g = autodiff.Graph.init(allocator, .{ .enable_grad = false, .device = .gpu, .mlx_stream = stream });
        defer g_g.deinit();
        const y_c = try attn.forward(&g_c, x);
        const y_g = try attn.forward(&g_g, x);
        stream.synchronize();

        var max_diff: f32 = 0.0;
        for (y_c.data, y_g.data) |v1, v2| {
            const diff = @abs(v1 - v2);
            if (diff > max_diff) max_diff = diff;
        }

        const idx_start = runner.results.items.len;
        try runner.benchmark("CPU CausalSelfAttn [4, 128, 256]", "MLX", flops_attn, null, total_tokens, "tokens/s", &cpu_mod);
        try runner.benchmark("MLX GPU CausalSelfAttn [4, 128, 256, GPU]", "MLX", flops_attn, null, total_tokens, "tokens/s", &gpu_mod);

        if (runner.results.items.len >= idx_start + 2) {
            const cpu_st = runner.results.items[idx_start];
            const gpu_st = runner.results.items[idx_start + 1];
            const speedup = if (gpu_st.avgMs() > 0) cpu_st.avgMs() / gpu_st.avgMs() else 1.0;
            const tps = gpu_st.throughputItems() orelse 0.0;
            const metric_str = try std.fmt.allocPrint(allocator, "{d:.0} tokens/s", .{tps});

            try comparisons.append(allocator, .{
                .name = "CausalSelfAttention [B=4, T=128, D=256]",
                .cpu_ms = cpu_st.avgMs(),
                .gpu_ms = gpu_st.avgMs(),
                .speedup = speedup,
                .gpu_metric = metric_str,
                .max_error = max_diff,
                .passed = (max_diff < 1e-3),
            });
        }
    }

    // =========================================================================
    // 6. 输出端到端 CPU vs MLX GPU 速度与精度加速汇总对照表
    // =========================================================================
    if (!runner.config.quiet and comparisons.items.len > 0) {
        std.debug.print("\n========================================================================================================================\n", .{});
        std.debug.print("                              Apple Silicon CPU (AMX CBLAS) vs MLX GPU 性能对比与精度验证表                              \n", .{});
        std.debug.print("========================================================================================================================\n", .{});
        std.debug.print(" {s:<40} | {s:>10} | {s:>10} | {s:>9} | {s:>16} | {s:>10} | {s:>6}\n", .{
            "Workload / Operator", "CPU (ms)", "GPU (ms)", "Speedup", "Peak Throughput", "Max Error", "Status",
        });
        std.debug.print("------------------------------------------------------------------------------------------------------------------------\n", .{});

        for (comparisons.items) |row| {
            std.debug.print(" {s:<40} | {d:>10.3} | {d:>10.3} | {d:>8.2}x | {s:>16} | {d:>10.6} | {s:>6}\n", .{
                row.name,
                row.cpu_ms,
                row.gpu_ms,
                row.speedup,
                row.gpu_metric,
                row.max_error,
                if (row.passed) "PASS" else "FAIL",
            });
            allocator.free(row.gpu_metric);
        }
        std.debug.print("========================================================================================================================\n\n", .{});
    }
}
