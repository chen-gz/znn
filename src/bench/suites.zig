const std = @import("std");
const testing_init = @import("../nn/testing_init.zig");
const bench = @import("../bench.zig");
const BenchmarkRunner = bench.BenchmarkRunner;
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");
const autodiff = @import("../autodiff.zig");
const optim = @import("../optim.zig");
const dataset = @import("../dataset.zig");

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
                var out_graph = autodiff.Graph.initNoGrad(self.arena.allocator());
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
                var graph = autodiff.Graph.init(alloc);
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
                var conv = try nn.Conv2D.init(alloc, 1, 16, 3);
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
                var out_graph = autodiff.Graph.initNoGrad(self.arena.allocator());
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
                var conv = try nn.Conv2D.init(alloc, 1, 16, 3);
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
                var graph = autodiff.Graph.init(alloc);
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
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensorNDWithData(&.{ 32, 16, 26, 26 }, self.x_data, true);
                const out = try graph.maxpool2d(x, 2, 2);
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
                var graph = autodiff.Graph.init(alloc);
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
                var out_graph = autodiff.Graph.initNoGrad(self.arena.allocator());
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
                var graph = autodiff.Graph.init(alloc);
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
                var graph = autodiff.Graph.init(alloc);
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
                var graph = autodiff.Graph.init(alloc);
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
                var conv1 = try nn.Conv2D.init(alloc, 1, 4, 3);
                conv1.resetParameters(rnd, .{});
                errdefer nn.deinitModel(&conv1, alloc);
                var conv2 = try nn.Conv2D.init(alloc, 4, 8, 3);
                conv2.resetParameters(rnd, .{});
                errdefer nn.deinitModel(&conv2, alloc);
                var conv3 = try nn.Conv2D.init(alloc, 8, 16, 3);
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
                var graph = autodiff.Graph.init(alloc);
                defer graph.deinit();

                const x = try graph.tensor(32, 784, false);
                @memcpy(x.data, self.x_data);

                const x_reshaped = try graph.reshape(x, &.{ 32, 1, 28, 28 });

                // Layer 1
                const x1 = try self.conv1.forward(&graph, x_reshaped);
                const a1 = try graph.relu(x1);
                const p1 = try graph.maxpool2d(a1, 2, 2);

                // Layer 2
                const x2 = try self.conv2.forward(&graph, p1);
                const a2 = try graph.relu(x2);
                const p2 = try graph.maxpool2d(a2, 2, 2);

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
                var graph = autodiff.Graph.init(alloc);
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
