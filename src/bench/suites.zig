const std = @import("std");
const testing_init = @import("../nn/testing_init.zig");
const BenchmarkRunner = @import("runner.zig").BenchmarkRunner;
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");
const autodiff = @import("../autodiff.zig");
const optim = @import("../optim.zig");
const dataset = @import("../dataset.zig");
const backend = @import("../backend.zig");
const models = @import("../models.zig");

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
                var linear = try nn.Linear.init(alloc, 784, 128);
                linear.resetParameters(rnd, .{});
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
                nn.deinitModel(&self.linear, self.allocator);
                tensor.free(self.allocator, self.x);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var out_graph = autodiff.Graph.init(self.arena.allocator(), .{ .enable_grad = false });
                defer out_graph.deinit();
                const out = try self.linear.forward(&out_graph, self.x);
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
                var linear = try nn.Linear.init(alloc, 784, 128);
                linear.resetParameters(rnd, .{});
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
                nn.deinitModel(&self.linear, self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.initDefault(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 64, 784 }, self.x_data, true);
                const out = try self.linear.forward(&graph, x);
                @memset(out.grad, 1.0);
                nn.zeroGradModel(&self.linear);
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
                var conv = try nn.Conv2D.init(alloc, 1, 16, 3, .{});
                conv.resetParameters(rnd, .{});
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
                nn.deinitModel(&self.conv, self.allocator);
                tensor.free(self.allocator, self.x);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var out_graph = autodiff.Graph.init(self.arena.allocator(), .{ .enable_grad = false });
                defer out_graph.deinit();
                const out = try self.conv.forward(&out_graph, self.x);
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
                var conv = try nn.Conv2D.init(alloc, 1, 16, 3, .{});
                conv.resetParameters(rnd, .{});
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
                nn.deinitModel(&self.conv, self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.initDefault(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 32, 1, 28, 28 }, self.x_data, true);
                const out = try self.conv.forward(&graph, x);
                @memset(out.grad, 1.0);
                nn.zeroGradModel(&self.conv);
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
                var graph = autodiff.Graph.initDefault(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 32, 16, 26, 26 }, self.x_data, true);
                const out = try graph.maxpool2d(x, 2, .{});
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
                var swiglu = try nn.SwiGLU.init(alloc, 128, 256);
                try testing_init.initFromOnes(&swiglu, alloc, rnd, &.{ 2, 128 });
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
                nn.deinitModel(&self.swiglu, self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.initDefault(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 4, 64, 128 }, self.x_data, true);
                const out = try self.swiglu.forward(&graph, x);
                @memset(out.grad, 1.0);
                nn.zeroGradModel(&self.swiglu);
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
                var attn = try nn.CausalSelfAttention.init(alloc, 128, 4);
                try testing_init.initFromOnes(&attn, alloc, rnd, &.{ 1, 2, 128 });
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
                nn.deinitModel(&self.attn, self.allocator);
                tensor.free(self.allocator, self.x);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var out_graph = autodiff.Graph.init(self.arena.allocator(), .{ .enable_grad = false });
                defer out_graph.deinit();
                const out = try self.attn.forward(&out_graph, self.x);
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
                var attn = try nn.CausalSelfAttention.init(alloc, 128, 4);
                try testing_init.initFromOnes(&attn, alloc, rnd, &.{ 1, 2, 128 });
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
                nn.deinitModel(&self.attn, self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.initDefault(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 4, 64, 128 }, self.x_data, true);
                const out = try self.attn.forward(&graph, x);
                @memset(out.grad, 1.0);
                nn.zeroGradModel(&self.attn);
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
                var block = try nn.TransformerBlock.init(alloc, 128, 4);
                try testing_init.initFromOnes(&block, alloc, rnd, &.{ 1, 2, 128 });
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
                nn.deinitModel(&self.block, self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.initDefault(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 4, 64, 128 }, self.x_data, true);
                const out = try self.block.forward(&graph, x);
                @memset(out.grad, 1.0);
                nn.zeroGradModel(&self.block);
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

    // 10. Scaled Dot-Product Attention Causal [B=1, H=8, L=256, D=64]
    {
        const SdpaCausalContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            q_raw: []f32,
            k_raw: []f32,
            v_raw: []f32,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                const total = 1 * 8 * 256 * 64;
                const q_raw = try alloc.alloc(f32, total);
                errdefer alloc.free(q_raw);
                const k_raw = try alloc.alloc(f32, total);
                errdefer alloc.free(k_raw);
                const v_raw = try alloc.alloc(f32, total);
                errdefer alloc.free(v_raw);

                for (0..total) |i| {
                    q_raw[i] = (rnd.float(f32) - 0.5) * 0.2;
                    k_raw[i] = (rnd.float(f32) - 0.5) * 0.2;
                    v_raw[i] = (rnd.float(f32) - 0.5) * 0.2;
                }

                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .q_raw = q_raw,
                    .k_raw = k_raw,
                    .v_raw = v_raw,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.allocator.free(self.q_raw);
                self.allocator.free(self.k_raw);
                self.allocator.free(self.v_raw);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var g = autodiff.Graph.init(self.arena.allocator(), .{ .enable_grad = false });
                defer g.deinit();

                const q = try g.tensorNDWithData(&.{ 1, 8, 256, 64 }, self.q_raw, false);
                const k = try g.tensorNDWithData(&.{ 1, 8, 256, 64 }, self.k_raw, false);
                const v = try g.tensorNDWithData(&.{ 1, 8, 256, 64 }, self.v_raw, false);
                const out = try g.scaledDotProductAttention(q, k, v, 0.125, true);
                std.mem.doNotOptimizeAway(out.data.ptr);
            }
        };

        var ctx = try SdpaCausalContext.init(allocator, random);
        defer ctx.deinit();
        const flops_sdpa: f64 = 4.0 * 1.0 * 8.0 * 256.0 * 256.0 * 64.0;
        try runner.benchmark("SDPA Causal [1, 8, 256, 64]", "Layers", flops_sdpa, null, null, "", &ctx);
    }

    // 11. Gemma4Attention Sliding-Window GQA Forward [B=1, T=32, D=256, H=8, KV=2]
    {
        const Gemma4AttnContext = struct {
            allocator: std.mem.Allocator,
            arena: std.heap.ArenaAllocator,
            attn: models.gemma4.Gemma4Attention,
            x: *tensor.Tensor,

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                var attn = try models.gemma4.Gemma4Attention.init(
                    alloc,
                    .sliding_attention,
                    256,
                    8,
                    2,
                    32,
                    16,
                    1e-6,
                );
                errdefer nn.deinitModel(&attn, alloc);
                attn.q_proj.resetParameters(rnd, .{});
                attn.k_proj.resetParameters(rnd, .{});
                if (attn.v_proj) |*vp| vp.resetParameters(rnd, .{});
                attn.o_proj.resetParameters(rnd, .{});

                const x = try tensor.zeros(alloc, &.{ 1, 32, 256 });
                x.fillNormal(rnd, 0.0, 0.2);

                return .{
                    .allocator = alloc,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .attn = attn,
                    .x = x,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                nn.deinitModel(&self.attn, self.allocator);
                tensor.free(self.allocator, self.x);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var g = autodiff.Graph.init(self.arena.allocator(), .{ .enable_grad = false });
                defer g.deinit();

                const out = try self.attn.forward(&g, self.x);
                std.mem.doNotOptimizeAway(out.data.ptr);
            }
        };

        var ctx = try Gemma4AttnContext.init(allocator, random);
        defer ctx.deinit();
        try runner.benchmark("Gemma4Attention GQA [1, 32, 256, 8/2]", "Layers", null, null, 32, "tokens/s", &ctx);
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
                var fc1 = try nn.Linear.init(alloc, 784, 128);
                fc1.resetParameters(rnd, .{});
                errdefer nn.deinitModel(&fc1, alloc);
                var fc2 = try nn.Linear.init(alloc, 128, 64);
                fc2.resetParameters(rnd, .{});
                errdefer nn.deinitModel(&fc2, alloc);
                var fc3 = try nn.Linear.init(alloc, 64, 10);
                fc3.resetParameters(rnd, .{});
                errdefer nn.deinitModel(&fc3, alloc);

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
                nn.deinitModel(&self.fc1, self.allocator);
                nn.deinitModel(&self.fc2, self.allocator);
                nn.deinitModel(&self.fc3, self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.initDefault(alloc);
                defer graph.deinit();

                const x = try graph.tensor(64, 784, false);
                @memcpy(x.data, self.x_data);

                // Forward
                const h1 = try self.fc1.forward(&graph, x);
                const a1 = try graph.relu(h1);
                const h2 = try self.fc2.forward(&graph, a1);
                const a2 = try graph.relu(h2);
                const logits = try self.fc3.forward(&graph, a2);

                // Loss
                const loss = try graph.softmaxCrossEntropy(logits, &self.targets);

                // Backward & Optimizer
                nn.zeroGradModel(&self.fc1);
                nn.zeroGradModel(&self.fc2);
                nn.zeroGradModel(&self.fc3);
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
                var conv1 = try nn.Conv2D.init(alloc, 1, 4, 3, .{});
                conv1.resetParameters(rnd, .{});
                errdefer nn.deinitModel(&conv1, alloc);
                var conv2 = try nn.Conv2D.init(alloc, 4, 8, 3, .{});
                conv2.resetParameters(rnd, .{});
                errdefer nn.deinitModel(&conv2, alloc);
                var conv3 = try nn.Conv2D.init(alloc, 8, 16, 3, .{});
                conv3.resetParameters(rnd, .{});
                errdefer nn.deinitModel(&conv3, alloc);
                var fc1 = try nn.Linear.init(alloc, 144, 10);
                fc1.resetParameters(rnd, .{});
                errdefer nn.deinitModel(&fc1, alloc);

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
                nn.deinitModel(&self.conv1, self.allocator);
                nn.deinitModel(&self.conv2, self.allocator);
                nn.deinitModel(&self.conv3, self.allocator);
                nn.deinitModel(&self.fc1, self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.initDefault(alloc);
                defer graph.deinit();

                const x = try graph.tensor(32, 784, false);
                @memcpy(x.data, self.x_data);

                const x_reshaped = try graph.reshape(x, &.{ 32, 1, 28, 28 });

                // Layer 1
                const x1 = try self.conv1.forward(&graph, x_reshaped);
                const a1 = try graph.relu(x1);
                const p1 = try graph.maxpool2d(a1, 2, .{ .stride = 2 });

                // Layer 2
                const x2 = try self.conv2.forward(&graph, p1);
                const a2 = try graph.relu(x2);
                const p2 = try graph.maxpool2d(a2, 2, .{ .stride = 2 });

                // Layer 3
                const x3 = try self.conv3.forward(&graph, p2);
                const a3 = try graph.relu(x3);

                // Flatten -> Linear
                const flat = try graph.reshape(a3, &.{ 32, 144 });
                const logits = try self.fc1.forward(&graph, flat);

                // Loss
                const loss = try graph.softmaxCrossEntropy(logits, &self.targets);

                // Backward & Optimizer
                nn.zeroGradModel(&self.conv1);
                nn.zeroGradModel(&self.conv2);
                nn.zeroGradModel(&self.conv3);
                nn.zeroGradModel(&self.fc1);
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
                var block = try nn.TransformerBlock.init(alloc, 128, 4);
                try testing_init.initFromOnes(&block, alloc, rnd, &.{ 1, 2, 128 });
                errdefer nn.deinitModel(&block, alloc);

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
                nn.deinitModel(&self.block, self.allocator);
                self.allocator.free(self.x_data);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                const alloc = self.arena.allocator();
                var graph = autodiff.Graph.initDefault(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 4, 64, 128 }, self.x_data, true);
                const out = try self.block.forward(&graph, x);

                @memset(out.grad, 1.0);
                nn.zeroGradModel(&self.block);
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
                var linear = try nn.Linear.init(alloc, 1000, 1000);
                linear.resetParameters(prng.random(), .{});
                errdefer nn.deinitModel(&linear, alloc);

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
                nn.deinitModel(&self.linear, self.allocator);
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
                var linear = try nn.Linear.init(alloc, 1000, 1000);
                linear.resetParameters(prng.random(), .{});
                errdefer nn.deinitModel(&linear, alloc);

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
                nn.deinitModel(&self.linear, self.allocator);
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

/// Suite 8: Executor (ParallelFor & ParallelReduce Dispatch Overhead)
pub fn runExecutorBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    // 1. parallelFor serial [10k items]
    {
        const ForSerialCtx = struct {
            data: []u32,
            allocator: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const data = try alloc.alloc(u32, 10_000);
                @memset(data, 1);
                return .{ .data = data, .allocator = alloc };
            }

            pub fn deinit(self: *@This()) void {
                self.allocator.free(self.data);
            }

            fn kernel(slice: []u32, start: usize, end: usize) void {
                for (start..end) |i| {
                    slice[i] = slice[i] *% 3 +% 1;
                }
            }

            pub fn run(self: *@This()) !void {
                try backend.Executor.serial.parallelFor(self.data.len, 100, self.data, kernel);
                std.mem.doNotOptimizeAway(self.data.ptr);
            }
        };

        var ctx = try ForSerialCtx.init(allocator);
        defer ctx.deinit();
        try runner.benchmark("parallelFor serial [10k, grain=100]", "Executor", null, 10_000 * @sizeOf(u32), 10_000, "items/s", &ctx);
    }

    // 2. parallelFor threaded [10k items, 4 workers]
    {
        const ForThreadedCtx = struct {
            data: []u32,
            threaded: std.Io.Threaded,
            allocator: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const data = try alloc.alloc(u32, 10_000);
                @memset(data, 1);
                return .{
                    .data = data,
                    .threaded = std.Io.Threaded.init(alloc, .{}),
                    .allocator = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.threaded.deinit();
                self.allocator.free(self.data);
            }

            fn kernel(slice: []u32, start: usize, end: usize) void {
                for (start..end) |i| {
                    slice[i] = slice[i] *% 3 +% 1;
                }
            }

            pub fn run(self: *@This()) !void {
                const exec = backend.Executor{ .io = self.threaded.io(), .max_tasks = 4 };
                try exec.parallelFor(self.data.len, 2500, self.data, kernel);
                std.mem.doNotOptimizeAway(self.data.ptr);
            }
        };

        var ctx = try ForThreadedCtx.init(allocator);
        defer ctx.deinit();
        try runner.benchmark("parallelFor threaded [10k, 4 workers]", "Executor", null, 10_000 * @sizeOf(u32), 10_000, "items/s", &ctx);
    }

    // 3. parallelReduce serial [100k items]
    {
        const ReduceSerialCtx = struct {
            data: []u32,
            allocator: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const data = try alloc.alloc(u32, 100_000);
                for (data, 0..) |*v, i| v.* = @as(u32, @intCast(i % 100));
                return .{ .data = data, .allocator = alloc };
            }

            pub fn deinit(self: *@This()) void {
                self.allocator.free(self.data);
            }

            fn map(slice: []const u32, start: usize, end: usize) u64 {
                var s: u64 = 0;
                for (start..end) |i| s += slice[i];
                return s;
            }

            fn add(a: u64, b: u64) u64 {
                return a + b;
            }

            pub fn run(self: *@This()) !void {
                var total = try backend.Executor.serial.parallelReduce(u64, self.data.len, 1000, self.data, map, add, 0);
                std.mem.doNotOptimizeAway(&total);
            }
        };

        var ctx = try ReduceSerialCtx.init(allocator);
        defer ctx.deinit();
        try runner.benchmark("parallelReduce serial [100k, grain=1000]", "Executor", null, 100_000 * @sizeOf(u32), 100_000, "items/s", &ctx);
    }

    // 4. parallelReduce threaded [100k items, 4 workers]
    {
        const ReduceThreadedCtx = struct {
            data: []u32,
            threaded: std.Io.Threaded,
            allocator: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const data = try alloc.alloc(u32, 100_000);
                for (data, 0..) |*v, i| v.* = @as(u32, @intCast(i % 100));
                return .{
                    .data = data,
                    .threaded = std.Io.Threaded.init(alloc, .{}),
                    .allocator = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.threaded.deinit();
                self.allocator.free(self.data);
            }

            fn map(slice: []const u32, start: usize, end: usize) u64 {
                var s: u64 = 0;
                for (start..end) |i| s += slice[i];
                return s;
            }

            fn add(a: u64, b: u64) u64 {
                return a + b;
            }

            pub fn run(self: *@This()) !void {
                const exec = backend.Executor{ .io = self.threaded.io(), .max_tasks = 4 };
                var total = try exec.parallelReduce(u64, self.data.len, 10_000, self.data, map, add, 0);
                std.mem.doNotOptimizeAway(&total);
            }
        };

        var ctx = try ReduceThreadedCtx.init(allocator);
        defer ctx.deinit();
        try runner.benchmark("parallelReduce threaded [100k, 4 workers]", "Executor", null, 100_000 * @sizeOf(u32), 100_000, "items/s", &ctx);
    }
}

/// Suite 9: Q4 (Quantized GEMV & GEMM Throughput)
pub fn runQ4Benchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    // 1. Q4 GEMV serial [1x2048x2048]
    {
        const Q4GemvSerialCtx = struct {
            blocks: []tensor.Q4Block,
            x: *tensor.Tensor,
            out: *tensor.Tensor,
            allocator: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator, in_f: usize, out_f: usize) !@This() {
                const num_blocks = (in_f * out_f) / tensor.Q4Block.QK;
                const raw_w = try alloc.alloc(f32, in_f * out_f);
                defer alloc.free(raw_w);
                for (raw_w, 0..) |*w, i| w.* = @sin(@as(f32, @floatFromInt(i))) * 1.5;

                const blocks = try alloc.alloc(tensor.Q4Block, num_blocks);
                errdefer alloc.free(blocks);
                tensor.quant.quantizeSlice(blocks, raw_w);

                const x = try tensor.zeros(alloc, &.{ 1, in_f });
                errdefer x.deinit(alloc);
                for (x.data, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i))) * 0.8;

                const out = try tensor.zeros(alloc, &.{ 1, out_f });
                errdefer out.deinit(alloc);

                return .{
                    .blocks = blocks,
                    .x = x,
                    .out = out,
                    .allocator = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.allocator.free(self.blocks);
                self.x.deinit(self.allocator);
                self.out.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                try tensor.quant.q4MatMulInto(.serial, self.out, self.x, self.blocks, 2048, 2048);
                std.mem.doNotOptimizeAway(self.out.data.ptr);
            }
        };

        var ctx = try Q4GemvSerialCtx.init(allocator, 2048, 2048);
        defer ctx.deinit();
        const flops: f64 = 2.0 * 1.0 * 2048.0 * 2048.0;
        const weight_bytes: u64 = ((2048 * 2048) / tensor.Q4Block.QK) * @sizeOf(tensor.Q4Block);
        try runner.benchmark("Q4 GEMV serial [1x2048x2048]", "Q4", flops, weight_bytes, null, "", &ctx);
    }

    // 2. Q4 GEMV threaded [1x2048x2048, 4 workers]
    {
        const Q4GemvThreadedCtx = struct {
            blocks: []tensor.Q4Block,
            x: *tensor.Tensor,
            out: *tensor.Tensor,
            threaded: std.Io.Threaded,
            allocator: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator, in_f: usize, out_f: usize) !@This() {
                const num_blocks = (in_f * out_f) / tensor.Q4Block.QK;
                const raw_w = try alloc.alloc(f32, in_f * out_f);
                defer alloc.free(raw_w);
                for (raw_w, 0..) |*w, i| w.* = @sin(@as(f32, @floatFromInt(i))) * 1.5;

                const blocks = try alloc.alloc(tensor.Q4Block, num_blocks);
                errdefer alloc.free(blocks);
                tensor.quant.quantizeSlice(blocks, raw_w);

                const x = try tensor.zeros(alloc, &.{ 1, in_f });
                errdefer x.deinit(alloc);
                for (x.data, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i))) * 0.8;

                const out = try tensor.zeros(alloc, &.{ 1, out_f });
                errdefer out.deinit(alloc);

                return .{
                    .blocks = blocks,
                    .x = x,
                    .out = out,
                    .threaded = std.Io.Threaded.init(alloc, .{}),
                    .allocator = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.threaded.deinit();
                self.allocator.free(self.blocks);
                self.x.deinit(self.allocator);
                self.out.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                const exec = backend.Executor{ .io = self.threaded.io(), .max_tasks = 4 };
                try tensor.quant.q4MatMulInto(exec, self.out, self.x, self.blocks, 2048, 2048);
                std.mem.doNotOptimizeAway(self.out.data.ptr);
            }
        };

        var ctx = try Q4GemvThreadedCtx.init(allocator, 2048, 2048);
        defer ctx.deinit();
        const flops: f64 = 2.0 * 1.0 * 2048.0 * 2048.0;
        const weight_bytes: u64 = ((2048 * 2048) / tensor.Q4Block.QK) * @sizeOf(tensor.Q4Block);
        try runner.benchmark("Q4 GEMV threaded [1x2048x2048, 4 workers]", "Q4", flops, weight_bytes, null, "", &ctx);
    }

    // 2b. Q4 GEMV through Graph.q4MatMul (the Q4Linear.forward path) [1x2048x2048, 4 workers]
    {
        const Q4GraphCtx = struct {
            blocks: []tensor.Q4Block,
            x: *tensor.Tensor,
            threaded: std.Io.Threaded,
            arena: std.heap.ArenaAllocator,
            allocator: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator, in_f: usize, out_f: usize) !@This() {
                const num_blocks = (in_f * out_f) / tensor.Q4Block.QK;
                const raw_w = try alloc.alloc(f32, in_f * out_f);
                defer alloc.free(raw_w);
                for (raw_w, 0..) |*w, i| w.* = @sin(@as(f32, @floatFromInt(i))) * 1.5;

                const blocks = try alloc.alloc(tensor.Q4Block, num_blocks);
                errdefer alloc.free(blocks);
                tensor.quant.quantizeSlice(blocks, raw_w);

                const x = try tensor.zeros(alloc, &.{ 1, in_f });
                errdefer x.deinit(alloc);
                for (x.data, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i))) * 0.8;

                return .{
                    .blocks = blocks,
                    .x = x,
                    .threaded = std.Io.Threaded.init(alloc, .{}),
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .allocator = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.threaded.deinit();
                self.allocator.free(self.blocks);
                self.x.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var g = autodiff.Graph.init(self.arena.allocator(), .{ .enable_grad = false, .io = self.threaded.io(), .max_tasks = 4 });
                defer g.deinit();
                const y = try g.q4MatMul(self.x, self.blocks, 2048, 2048);
                std.mem.doNotOptimizeAway(y.data.ptr);
            }
        };

        var ctx = try Q4GraphCtx.init(allocator, 2048, 2048);
        defer ctx.deinit();
        const flops: f64 = 2.0 * 1.0 * 2048.0 * 2048.0;
        const weight_bytes: u64 = ((2048 * 2048) / tensor.Q4Block.QK) * @sizeOf(tensor.Q4Block);
        try runner.benchmark("Q4 GEMV Graph.q4MatMul [1x2048x2048, 4 workers]", "Q4", flops, weight_bytes, null, "", &ctx);
    }

    // 3. Q4 GEMV serial [1x4096x4096]
    {
        const Q4GemvLargeCtx = struct {
            blocks: []tensor.Q4Block,
            x: *tensor.Tensor,
            out: *tensor.Tensor,
            allocator: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator, in_f: usize, out_f: usize) !@This() {
                const num_blocks = (in_f * out_f) / tensor.Q4Block.QK;
                const raw_w = try alloc.alloc(f32, in_f * out_f);
                defer alloc.free(raw_w);
                for (raw_w, 0..) |*w, i| w.* = @sin(@as(f32, @floatFromInt(i))) * 1.5;

                const blocks = try alloc.alloc(tensor.Q4Block, num_blocks);
                errdefer alloc.free(blocks);
                tensor.quant.quantizeSlice(blocks, raw_w);

                const x = try tensor.zeros(alloc, &.{ 1, in_f });
                errdefer x.deinit(alloc);
                for (x.data, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i))) * 0.8;

                const out = try tensor.zeros(alloc, &.{ 1, out_f });
                errdefer out.deinit(alloc);

                return .{
                    .blocks = blocks,
                    .x = x,
                    .out = out,
                    .allocator = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.allocator.free(self.blocks);
                self.x.deinit(self.allocator);
                self.out.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                try tensor.quant.q4MatMulInto(.serial, self.out, self.x, self.blocks, 4096, 4096);
                std.mem.doNotOptimizeAway(self.out.data.ptr);
            }
        };

        var ctx = try Q4GemvLargeCtx.init(allocator, 4096, 4096);
        defer ctx.deinit();
        const flops: f64 = 2.0 * 1.0 * 4096.0 * 4096.0;
        const weight_bytes: u64 = ((4096 * 4096) / tensor.Q4Block.QK) * @sizeOf(tensor.Q4Block);
        try runner.benchmark("Q4 GEMV serial [1x4096x4096]", "Q4", flops, weight_bytes, null, "", &ctx);
    }

    // 4. Q4 GEMM batched [16x2048x2048, 4 workers]
    {
        const Q4GemmBatchCtx = struct {
            blocks: []tensor.Q4Block,
            x: *tensor.Tensor,
            out: *tensor.Tensor,
            threaded: std.Io.Threaded,
            allocator: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator, m: usize, in_f: usize, out_f: usize) !@This() {
                const num_blocks = (in_f * out_f) / tensor.Q4Block.QK;
                const raw_w = try alloc.alloc(f32, in_f * out_f);
                defer alloc.free(raw_w);
                for (raw_w, 0..) |*w, i| w.* = @sin(@as(f32, @floatFromInt(i))) * 1.5;

                const blocks = try alloc.alloc(tensor.Q4Block, num_blocks);
                errdefer alloc.free(blocks);
                tensor.quant.quantizeSlice(blocks, raw_w);

                const x = try tensor.zeros(alloc, &.{ m, in_f });
                errdefer x.deinit(alloc);
                for (x.data, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i))) * 0.8;

                const out = try tensor.zeros(alloc, &.{ m, out_f });
                errdefer out.deinit(alloc);

                return .{
                    .blocks = blocks,
                    .x = x,
                    .out = out,
                    .threaded = std.Io.Threaded.init(alloc, .{}),
                    .allocator = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.threaded.deinit();
                self.allocator.free(self.blocks);
                self.x.deinit(self.allocator);
                self.out.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                const exec = backend.Executor{ .io = self.threaded.io(), .max_tasks = 4 };
                try tensor.quant.q4MatMulInto(exec, self.out, self.x, self.blocks, 2048, 2048);
                std.mem.doNotOptimizeAway(self.out.data.ptr);
            }
        };

        var ctx = try Q4GemmBatchCtx.init(allocator, 16, 2048, 2048);
        defer ctx.deinit();
        const flops: f64 = 2.0 * 16.0 * 2048.0 * 2048.0;
        const weight_bytes: u64 = ((2048 * 2048) / tensor.Q4Block.QK) * @sizeOf(tensor.Q4Block);
        try runner.benchmark("Q4 GEMM batched [16x2048x2048, 4 workers]", "Q4", flops, weight_bytes, null, "", &ctx);
    }
}

/// 构造符合 ZNNQ4G02 二进制布局规范的合成随机量化模型内存切片（无需外部磁盘权重文件）
fn allocSyntheticGemma4Q4Mmap(
    comptime cfg: models.gemma4.Gemma4Config,
    allocator: std.mem.Allocator,
    seed: u64,
) ![]align(std.heap.page_size_min) u8 {
    const ModelType = models.gemma4_q4.Gemma4Q4ForCausalLM(cfg);
    const expected_bytes = ModelType.computeExpectedFileSize();
    const buf = try allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), expected_bytes);
    errdefer allocator.free(buf);

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();

    @memcpy(buf[0..8], "ZNNQ4G02");
    var cursor: usize = 8;

    // 1. 最终层 RMSNorm (BF16)
    const final_norm_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, buf[cursor .. cursor + cfg.hidden_size * 2]));
    for (final_norm_u16) |*w| {
        w.* = tensor.bf16.fromF32((rnd.float(f32) - 0.5) * 0.1).bits;
    }
    cursor += cfg.hidden_size * 2;

    // 2. 词表嵌入矩阵 (BF16)
    const embed_len = cfg.vocab_size * cfg.hidden_size;
    const embed_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, buf[cursor .. cursor + embed_len * 2]));
    for (embed_u16) |*w| {
        w.* = tensor.bf16.fromF32((rnd.float(f32) - 0.5) * 0.1).bits;
    }
    cursor += embed_len * 2;

    // 3. 逐层填充 RMSNorm、layer_scalar 与 Q4Block 权重
    for (0..cfg.num_hidden_layers) |layer_idx| {
        const is_full = ((layer_idx + 1) % 6 == 0);
        const head_dim = if (is_full) cfg.global_head_dim else cfg.head_dim;
        const num_kv = if (is_full) cfg.num_global_key_value_heads else cfg.num_key_value_heads;

        const norms_count = cfg.hidden_size * 4 + head_dim * 2;
        const norms_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, buf[cursor .. cursor + norms_count * 2]));
        for (norms_u16) |*w| {
            w.* = tensor.bf16.fromF32((rnd.float(f32) - 0.5) * 0.1).bits;
        }
        cursor += norms_count * 2;

        // layer_scalar = 1.0 (BF16 0x3F80)
        std.mem.writeInt(u16, buf[cursor .. cursor + 2][0..2], 0x3F80, .little);
        cursor += 2;

        const q_blocks = (cfg.hidden_size * cfg.num_attention_heads * head_dim) / tensor.Q4Block.QK;
        const k_blocks = (cfg.hidden_size * num_kv * head_dim) / tensor.Q4Block.QK;
        const v_blocks = if (is_full) 0 else (cfg.hidden_size * num_kv * head_dim) / tensor.Q4Block.QK;
        const o_blocks = (cfg.num_attention_heads * head_dim * cfg.hidden_size) / tensor.Q4Block.QK;
        const mlp_blocks = (cfg.hidden_size * cfg.intermediate_size * 3) / tensor.Q4Block.QK;
        const total_blocks = q_blocks + k_blocks + v_blocks + o_blocks + mlp_blocks;

        for (0..total_blocks) |_| {
            std.mem.writeInt(u16, buf[cursor .. cursor + 2][0..2], 0x2800, .little);
            rnd.bytes(buf[cursor + 2 .. cursor + 18]);
            cursor += @sizeOf(tensor.Q4Block);
        }
    }

    std.debug.assert(cursor == expected_bytes);
    return buf;
}

/// Suite 10: Decode (Autoregressive Single-Step & End-to-End Whole-Model Generation Throughput)
pub fn runDecodeBenchmarks(runner: *BenchmarkRunner, allocator: std.mem.Allocator) !void {
    // 1. TinyGemma4 Decode Step [B=1, T=1]
    {
        const TinyGemmaDecodeCtx = struct {
            model: models.gemma4.TinyGemma4,
            token_ids: *tensor.Tensor,
            arena: std.heap.ArenaAllocator,
            allocator: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const model = try models.gemma4.TinyGemma4.init(alloc);
                const token_ids = try tensor.zeros(alloc, &.{ 1, 1 });
                token_ids.data[0] = 42.0;
                return .{
                    .model = model,
                    .token_ids = token_ids,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .allocator = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                nn.deinitModel(&self.model, self.allocator);
                self.token_ids.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var g = autodiff.Graph.init(self.arena.allocator(), .{ .enable_grad = false });
                defer g.deinit();
                const logits = try self.model.forward(&g, self.token_ids);
                std.mem.doNotOptimizeAway(logits.data.ptr);
            }
        };

        var ctx = try TinyGemmaDecodeCtx.init(allocator);
        defer ctx.deinit();
        try runner.benchmark("TinyGemma4 Decode Step [B=1, T=1]", "Decode", null, null, 1, "tokens/s", &ctx);
    }

    // 2. TinyQ4Gemma4 Decode Step [B=1, T=1]
    {
        const TinyQ4DecodeCtx = struct {
            model: models.gemma4_q4.TinyQ4Gemma4,
            token_ids: *tensor.Tensor,
            arena: std.heap.ArenaAllocator,
            allocator: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const model = try models.gemma4_q4.TinyQ4Gemma4.init(alloc);
                const token_ids = try tensor.zeros(alloc, &.{ 1, 1 });
                token_ids.data[0] = 42.0;
                return .{
                    .model = model,
                    .token_ids = token_ids,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .allocator = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.model.deinit(self.allocator);
                self.token_ids.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var g = autodiff.Graph.init(self.arena.allocator(), .{ .enable_grad = false });
                defer g.deinit();
                const logits = try self.model.forward(&g, self.token_ids);
                std.mem.doNotOptimizeAway(logits.data.ptr);
            }
        };

        var ctx = try TinyQ4DecodeCtx.init(allocator);
        defer ctx.deinit();
        try runner.benchmark("TinyQ4Gemma4 Decode Step [B=1, T=1]", "Decode", null, null, 1, "tokens/s", &ctx);
    }

    // 3. TinyQ4Gemma4 Prefill Step [B=1, T=16]
    {
        const TinyQ4PrefillCtx = struct {
            model: models.gemma4_q4.TinyQ4Gemma4,
            token_ids: *tensor.Tensor,
            arena: std.heap.ArenaAllocator,
            allocator: std.mem.Allocator,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                const model = try models.gemma4_q4.TinyQ4Gemma4.init(alloc);
                const token_ids = try tensor.zeros(alloc, &.{ 1, 16 });
                for (token_ids.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 128));
                return .{
                    .model = model,
                    .token_ids = token_ids,
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .allocator = alloc,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.model.deinit(self.allocator);
                self.token_ids.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var g = autodiff.Graph.init(self.arena.allocator(), .{ .enable_grad = false });
                defer g.deinit();
                const logits = try self.model.forward(&g, self.token_ids);
                std.mem.doNotOptimizeAway(logits.data.ptr);
            }
        };

        var ctx = try TinyQ4PrefillCtx.init(allocator);
        defer ctx.deinit();
        try runner.benchmark("TinyQ4Gemma4 Prefill Step [B=1, T=16]", "Decode", null, null, 16, "tokens/s", &ctx);
    }

    // 4~9. 6 层异构整网模型对比：未量化全精度 (F32) vs 4-bit 量化 (Q4)
    const BenchGemma4Config = models.gemma4.Gemma4Config{
        .vocab_size = 4096,
        .hidden_size = 256,
        .intermediate_size = 1024,
        .num_hidden_layers = 6,
        .num_attention_heads = 8,
        .num_key_value_heads = 4,
        .num_global_key_value_heads = 1,
        .head_dim = 32,
        .global_head_dim = 64,
        .sliding_window = 16,
        .rms_norm_eps = 1e-6,
        .final_logit_softcapping = 30.0,
        .max_position_embeddings = 128,
    };
    const BenchF32Model = models.gemma4.Gemma4ForCausalLM(BenchGemma4Config);
    const BenchQ4Model = models.gemma4_q4.Gemma4Q4ForCausalLM(BenchGemma4Config);

    // 4~6. 未量化全精度 (F32) 6 层异构整网基准测试 (Prefill / DecodeStep / E2E Generate)
    {
        const WholeF32BenchCtx = struct {
            model: BenchF32Model,
            cache: models.gemma4.Gemma4KVCache,
            threaded: std.Io.Threaded,
            arena: std.heap.ArenaAllocator,
            allocator: std.mem.Allocator,
            prompt16: [16]usize,
            prompt8: [8]usize,

            pub fn init(alloc: std.mem.Allocator) !@This() {
                var model = try BenchF32Model.init(alloc);
                errdefer nn.deinitModel(&model, alloc);

                var prng = std.Random.DefaultPrng.init(20261019);
                const rnd = prng.random();
                model.embed_tokens.weight.fillNormal(rnd, 0.0, 0.02);
                for (model.layers) |*layer| {
                    layer.self_attn.q_proj.resetParameters(rnd, .{});
                    layer.self_attn.k_proj.resetParameters(rnd, .{});
                    if (layer.self_attn.v_proj) |*vp| vp.resetParameters(rnd, .{});
                    layer.self_attn.o_proj.resetParameters(rnd, .{});
                    layer.mlp.gate_proj.resetParameters(rnd, .{});
                    layer.mlp.up_proj.resetParameters(rnd, .{});
                    layer.mlp.down_proj.resetParameters(rnd, .{});
                }

                var cache = try models.gemma4.Gemma4KVCache.init(alloc, BenchGemma4Config, .{ .max_seq_len = 64 });
                errdefer cache.deinit(alloc);

                var p16: [16]usize = undefined;
                for (&p16, 0..) |*t, i| t.* = (i * 17 + 3) % BenchGemma4Config.vocab_size;
                var p8: [8]usize = undefined;
                for (&p8, 0..) |*t, i| t.* = (i * 23 + 2) % BenchGemma4Config.vocab_size;

                return .{
                    .model = model,
                    .cache = cache,
                    .threaded = std.Io.Threaded.init(alloc, .{}),
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .allocator = alloc,
                    .prompt16 = p16,
                    .prompt8 = p8,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.threaded.deinit();
                self.cache.deinit(self.allocator);
                nn.deinitModel(&self.model, self.allocator);
            }
        };

        var shared = try WholeF32BenchCtx.init(allocator);
        defer shared.deinit();

        const F32PrefillRunner = struct {
            s: *WholeF32BenchCtx,
            pub fn run(self: *@This()) !void {
                _ = self.s.arena.reset(.retain_capacity);
                var g = autodiff.Graph.init(self.s.arena.allocator(), .{
                    .enable_grad = false,
                    .io = self.s.threaded.io(),
                    .max_tasks = 4,
                });
                defer g.deinit();
                var next_tok = try self.s.model.prefill(&g, &self.s.prompt16, &self.s.cache, 1.25);
                std.mem.doNotOptimizeAway(&next_tok);
            }
        };
        var prefill_r = F32PrefillRunner{ .s = &shared };
        try runner.benchmark("Gemma4F32-6L Prefill KVCache [T=16]", "Decode", null, null, 16, "tokens/s", &prefill_r);

        // 预先填充 8 个词元供单步 DecodeStep 测试
        {
            _ = shared.arena.reset(.retain_capacity);
            var g_init = autodiff.Graph.init(shared.arena.allocator(), .{ .enable_grad = false });
            defer g_init.deinit();
            _ = try shared.model.prefill(&g_init, &shared.prompt8, &shared.cache, 1.25);
        }

        const F32DecodeRunner = struct {
            s: *WholeF32BenchCtx,
            pub fn run(self: *@This()) !void {
                _ = self.s.arena.reset(.retain_capacity);
                for (self.s.cache.layers) |*cl| cl.curr_len = 8;
                var g = autodiff.Graph.init(self.s.arena.allocator(), .{
                    .enable_grad = false,
                    .io = self.s.threaded.io(),
                    .max_tasks = 4,
                });
                defer g.deinit();
                var next_tok = try self.s.model.decodeStep(&g, 42, &self.s.cache, &self.s.prompt16, 1.25);
                std.mem.doNotOptimizeAway(&next_tok);
            }
        };
        var decode_r = F32DecodeRunner{ .s = &shared };
        try runner.benchmark("Gemma4F32-6L DecodeStep KVCache [T=1]", "Decode", null, null, 1, "tokens/s", &decode_r);

        const F32E2ERunner = struct {
            s: *WholeF32BenchCtx,
            pub fn run(self: *@This()) !void {
                const gen_tokens: usize = 16;
                var history_buf: [8 + gen_tokens]usize = undefined;
                @memcpy(history_buf[0..8], &self.s.prompt8);
                var history_len: usize = 8;

                _ = self.s.arena.reset(.retain_capacity);
                {
                    var g_pre = autodiff.Graph.init(self.s.arena.allocator(), .{
                        .enable_grad = false,
                        .io = self.s.threaded.io(),
                        .max_tasks = 4,
                    });
                    defer g_pre.deinit();
                    const first_tok = try self.s.model.prefill(&g_pre, history_buf[0..history_len], &self.s.cache, 1.25);
                    history_buf[history_len] = first_tok;
                    history_len += 1;
                }

                var cur_tok = history_buf[history_len - 1];
                for (1..gen_tokens) |_| {
                    _ = self.s.arena.reset(.retain_capacity);
                    var g_dec = autodiff.Graph.init(self.s.arena.allocator(), .{
                        .enable_grad = false,
                        .io = self.s.threaded.io(),
                        .max_tasks = 4,
                    });
                    defer g_dec.deinit();
                    cur_tok = try self.s.model.decodeStep(&g_dec, cur_tok, &self.s.cache, history_buf[0..history_len], 1.25);
                    history_buf[history_len] = cur_tok;
                    history_len += 1;
                }
                std.mem.doNotOptimizeAway(&cur_tok);
            }
        };
        var e2e_r = F32E2ERunner{ .s = &shared };
        try runner.benchmark("Gemma4F32-6L E2E Generate [Gen=16]", "Decode", null, null, 16, "tokens/s", &e2e_r);
    }

    const synthetic_mmap = try allocSyntheticGemma4Q4Mmap(BenchGemma4Config, allocator, 20261019);
    defer allocator.free(synthetic_mmap);

    // 4. 整网 KVCache 预填充吞吐测试 (Gemma4Q4-6L Prefill KVCache [Prompt=16, 4 workers])
    {
        const WholePrefillCtx = struct {
            model: BenchQ4Model,
            cache: models.gemma4.Gemma4KVCache,
            mmap_bytes: []const u8,
            threaded: std.Io.Threaded,
            arena: std.heap.ArenaAllocator,
            allocator: std.mem.Allocator,
            prompt: [16]usize,

            pub fn init(alloc: std.mem.Allocator, mmap_slice: []const u8) !@This() {
                var model = try BenchQ4Model.loadFromMmap(alloc, mmap_slice);
                errdefer model.deinit(alloc);
                var cache = try models.gemma4.Gemma4KVCache.init(alloc, BenchGemma4Config, .{ .max_seq_len = 64 });
                errdefer cache.deinit(alloc);

                var prompt: [16]usize = undefined;
                for (&prompt, 0..) |*t, i| t.* = (i * 17 + 3) % BenchGemma4Config.vocab_size;

                return .{
                    .model = model,
                    .cache = cache,
                    .mmap_bytes = mmap_slice,
                    .threaded = std.Io.Threaded.init(alloc, .{}),
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .allocator = alloc,
                    .prompt = prompt,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.threaded.deinit();
                self.cache.deinit(self.allocator);
                self.model.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                var g = autodiff.Graph.init(self.arena.allocator(), .{
                    .enable_grad = false,
                    .io = self.threaded.io(),
                    .max_tasks = 4,
                });
                defer g.deinit();
                var next_tok = try self.model.prefill(&g, &self.prompt, &self.cache, self.mmap_bytes, 1.25);
                std.mem.doNotOptimizeAway(&next_tok);
            }
        };

        var ctx = try WholePrefillCtx.init(allocator, synthetic_mmap);
        defer ctx.deinit();
        try runner.benchmark("Gemma4Q4-6L Prefill KVCache [T=16]", "Decode", null, null, 16, "tokens/s", &ctx);
    }

    // 5. 整网 KVCache 单步增量解码吞吐测试 (Gemma4Q4-6L DecodeStep KVCache [T=1, 4 workers])
    {
        const WholeDecodeStepCtx = struct {
            model: BenchQ4Model,
            cache: models.gemma4.Gemma4KVCache,
            mmap_bytes: []const u8,
            threaded: std.Io.Threaded,
            arena: std.heap.ArenaAllocator,
            allocator: std.mem.Allocator,
            recent_tokens: [16]usize,

            pub fn init(alloc: std.mem.Allocator, mmap_slice: []const u8) !@This() {
                var model = try BenchQ4Model.loadFromMmap(alloc, mmap_slice);
                errdefer model.deinit(alloc);
                var cache = try models.gemma4.Gemma4KVCache.init(alloc, BenchGemma4Config, .{ .max_seq_len = 64 });
                errdefer cache.deinit(alloc);

                var recent: [16]usize = undefined;
                for (&recent, 0..) |*t, i| t.* = (i * 13 + 5) % BenchGemma4Config.vocab_size;

                var g_init = autodiff.Graph.init(alloc, .{ .enable_grad = false });
                defer g_init.deinit();
                _ = try model.prefill(&g_init, recent[0..8], &cache, mmap_slice, 1.25);

                return .{
                    .model = model,
                    .cache = cache,
                    .mmap_bytes = mmap_slice,
                    .threaded = std.Io.Threaded.init(alloc, .{}),
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .allocator = alloc,
                    .recent_tokens = recent,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.threaded.deinit();
                self.cache.deinit(self.allocator);
                self.model.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                _ = self.arena.reset(.retain_capacity);
                // 保持缓存游标稳定在 8 个上下文词元处，持续测试单步增量解码延迟
                for (self.cache.layers) |*cl| cl.curr_len = 8;
                var g = autodiff.Graph.init(self.arena.allocator(), .{
                    .enable_grad = false,
                    .io = self.threaded.io(),
                    .max_tasks = 4,
                });
                defer g.deinit();
                var next_tok = try self.model.decodeStep(&g, 42, &self.cache, self.mmap_bytes, &self.recent_tokens, 1.25);
                std.mem.doNotOptimizeAway(&next_tok);
            }
        };

        var ctx = try WholeDecodeStepCtx.init(allocator, synthetic_mmap);
        defer ctx.deinit();
        try runner.benchmark("Gemma4Q4-6L DecodeStep KVCache [T=1]", "Decode", null, null, 1, "tokens/s", &ctx);
    }

    // 6. 整网端到端连续多 Token 自回归生成测试 (Prompt=8 + 连续生成 16 个 Token，覆盖滑动窗口淘汰与全局注意力)
    {
        const WholeE2EGenerateCtx = struct {
            model: BenchQ4Model,
            cache: models.gemma4.Gemma4KVCache,
            mmap_bytes: []const u8,
            threaded: std.Io.Threaded,
            arena: std.heap.ArenaAllocator,
            allocator: std.mem.Allocator,
            prompt: [8]usize,

            pub fn init(alloc: std.mem.Allocator, mmap_slice: []const u8) !@This() {
                var model = try BenchQ4Model.loadFromMmap(alloc, mmap_slice);
                errdefer model.deinit(alloc);
                var cache = try models.gemma4.Gemma4KVCache.init(alloc, BenchGemma4Config, .{ .max_seq_len = 64 });
                errdefer cache.deinit(alloc);

                var prompt: [8]usize = undefined;
                for (&prompt, 0..) |*t, i| t.* = (i * 23 + 2) % BenchGemma4Config.vocab_size;

                return .{
                    .model = model,
                    .cache = cache,
                    .mmap_bytes = mmap_slice,
                    .threaded = std.Io.Threaded.init(alloc, .{}),
                    .arena = std.heap.ArenaAllocator.init(alloc),
                    .allocator = alloc,
                    .prompt = prompt,
                };
            }

            pub fn deinit(self: *@This()) void {
                self.arena.deinit();
                self.threaded.deinit();
                self.cache.deinit(self.allocator);
                self.model.deinit(self.allocator);
            }

            pub fn run(self: *@This()) !void {
                const gen_tokens: usize = 16;
                var history_buf: [8 + gen_tokens]usize = undefined;
                @memcpy(history_buf[0..8], &self.prompt);
                var history_len: usize = 8;

                // 1) Prefill 生成第 1 个新词元
                _ = self.arena.reset(.retain_capacity);
                {
                    var g_pre = autodiff.Graph.init(self.arena.allocator(), .{
                        .enable_grad = false,
                        .io = self.threaded.io(),
                        .max_tasks = 4,
                    });
                    defer g_pre.deinit();
                    const first_tok = try self.model.prefill(&g_pre, history_buf[0..history_len], &self.cache, self.mmap_bytes, 1.25);
                    history_buf[history_len] = first_tok;
                    history_len += 1;
                }

                // 2) 连续 15 步增量 Decode 生成剩余 15 个词元（共生成 16 个词元）
                var cur_tok = history_buf[history_len - 1];
                for (1..gen_tokens) |_| {
                    _ = self.arena.reset(.retain_capacity);
                    var g_dec = autodiff.Graph.init(self.arena.allocator(), .{
                        .enable_grad = false,
                        .io = self.threaded.io(),
                        .max_tasks = 4,
                    });
                    defer g_dec.deinit();
                    cur_tok = try self.model.decodeStep(&g_dec, cur_tok, &self.cache, self.mmap_bytes, history_buf[0..history_len], 1.25);
                    history_buf[history_len] = cur_tok;
                    history_len += 1;
                }
                std.mem.doNotOptimizeAway(&cur_tok);
            }
        };

        var ctx = try WholeE2EGenerateCtx.init(allocator, synthetic_mmap);
        defer ctx.deinit();
        try runner.benchmark("Gemma4Q4-6L E2E Generate [Gen=16]", "Decode", null, null, 16, "tokens/s", &ctx);
    }
}
