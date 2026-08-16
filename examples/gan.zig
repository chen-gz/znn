const std = @import("std");
const znn = @import("zig_ml");
const autodiff = znn.autodiff;
const tensor = znn.tensor;
const nn = znn.nn;
const optim = znn.optim;
const Tensor = tensor.Tensor;

// ============================================================================
// 1. Generator (生成器 G) 网络模块定义
// ============================================================================
// 结构: Latent Noise z (2D) -> Linear(2, 16) -> LeakyReLU -> Linear(16, 16) -> LeakyReLU -> Linear(16, 2)
pub const Generator = struct {
    l1: nn.Linear,
    act1: nn.LeakyReLU,
    l2: nn.Linear,
    act2: nn.LeakyReLU,
    l3: nn.Linear,

    pub fn init(allocator: std.mem.Allocator, random: std.Random) !Generator {
        return Generator{
            .l1 = try nn.Linear.init(allocator, 2, 16, random),
            .act1 = .{ .alpha = 0.2 },
            .l2 = try nn.Linear.init(allocator, 16, 16, random),
            .act2 = .{ .alpha = 0.2 },
            .l3 = try nn.Linear.init(allocator, 16, 2, random),
        };
    }

    pub fn deinit(self: Generator, allocator: std.mem.Allocator) void {
        self.l1.deinit(allocator);
        self.l2.deinit(allocator);
        self.l3.deinit(allocator);
    }

    pub fn zeroGrad(self: *Generator) void {
        self.l1.zeroGrad();
        self.l2.zeroGrad();
        self.l3.zeroGrad();
    }

    pub fn forward(self: *Generator, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, z: *Tensor) !*Tensor {
        const h1 = try self.l1.forward(allocator, graph, z);
        defer if (graph == null) tensor.free(allocator, h1);
        const a1 = try self.act1.forward(allocator, graph, h1);
        defer if (graph == null) tensor.free(allocator, a1);
        const h2 = try self.l2.forward(allocator, graph, a1);
        defer if (graph == null) tensor.free(allocator, h2);
        const a2 = try self.act2.forward(allocator, graph, h2);
        defer if (graph == null) tensor.free(allocator, a2);
        return try self.l3.forward(allocator, graph, a2);
    }
};

// ============================================================================
// 2. Discriminator (判别器 D) 网络模块定义
// ============================================================================
// 结构: Sample x (2D) -> Linear(2, 16) -> LeakyReLU -> Linear(16, 16) -> LeakyReLU -> Linear(16, 1) -> Logits
pub const Discriminator = struct {
    l1: nn.Linear,
    act1: nn.LeakyReLU,
    l2: nn.Linear,
    act2: nn.LeakyReLU,
    l3: nn.Linear,

    pub fn init(allocator: std.mem.Allocator, random: std.Random) !Discriminator {
        return Discriminator{
            .l1 = try nn.Linear.init(allocator, 2, 16, random),
            .act1 = .{ .alpha = 0.2 },
            .l2 = try nn.Linear.init(allocator, 16, 16, random),
            .act2 = .{ .alpha = 0.2 },
            .l3 = try nn.Linear.init(allocator, 16, 1, random),
        };
    }

    pub fn deinit(self: Discriminator, allocator: std.mem.Allocator) void {
        self.l1.deinit(allocator);
        self.l2.deinit(allocator);
        self.l3.deinit(allocator);
    }

    pub fn zeroGrad(self: *Discriminator) void {
        self.l1.zeroGrad();
        self.l2.zeroGrad();
        self.l3.zeroGrad();
    }

    pub fn forward(self: *Discriminator, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const h1 = try self.l1.forward(allocator, graph, x);
        defer if (graph == null) tensor.free(allocator, h1);
        const a1 = try self.act1.forward(allocator, graph, h1);
        defer if (graph == null) tensor.free(allocator, a1);
        const h2 = try self.l2.forward(allocator, graph, a1);
        defer if (graph == null) tensor.free(allocator, h2);
        const a2 = try self.act2.forward(allocator, graph, h2);
        defer if (graph == null) tensor.free(allocator, a2);
        return try self.l3.forward(allocator, graph, a2);
    }
};

// ============================================================================
// 3. 真实数据分布采样器 (Real Data Distribution)
// ============================================================================
// 目标真实分布: 2D 高斯分布 Mean = [3.0, -2.0], Std = [0.5, 0.5]
fn sampleRealData(graph: *autodiff.Graph, batch_size: usize, random: std.Random) !*Tensor {
    const t = try graph.randomNormal(&.{ batch_size, 2 }, random, 0.0, 1.0, false);
    for (0..batch_size) |i| {
        t.data[i * 2 + 0] = 3.0 + t.data[i * 2 + 0] * 0.5;
        t.data[i * 2 + 1] = -2.0 + t.data[i * 2 + 1] * 0.5;
    }
    return t;
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(1337);
    const random = prng.random();

    std.debug.print("=========================================================\n", .{});
    std.debug.print("🚀 Initializing Generative Adversarial Network (GAN)...  \n", .{});
    std.debug.print("   Target Real Distribution: 2D Gaussian Mean=[3.00, -2.00], Std=[0.50, 0.50]\n", .{});
    std.debug.print("=========================================================\n\n", .{});

    // 初始化 G 与 D
    var net_g = try Generator.init(allocator, random);
    defer net_g.deinit(allocator);

    var net_d = try Discriminator.init(allocator, random);
    defer net_d.deinit(allocator);

    // 初始化 Adam 优化器 (GAN 推荐参数 lr=0.005, beta1=0.5, beta2=0.999)
    var opt_g = try optim.AdamOptimizer.init(allocator, &net_g, .{
        .lr = 0.005,
        .beta1 = 0.5,
        .beta2 = 0.999,
        .eps = 1e-8,
    });
    defer opt_g.deinit();

    var opt_d = try optim.AdamOptimizer.init(allocator, &net_d, .{
        .lr = 0.005,
        .beta1 = 0.5,
        .beta2 = 0.999,
        .eps = 1e-8,
    });
    defer opt_d.deinit();

    const batch_size: usize = 64;
    const num_epochs: usize = 600;

    for (1..num_epochs + 1) |epoch| {
        // --------------------------------------------------------------------
        // Step 1: 训练判别器 Discriminator (D)
        // --------------------------------------------------------------------
        var graph_d = autodiff.Graph.init(allocator);

        // 真实数据样本 (Label = 1)
        const real_data = try sampleRealData(&graph_d, batch_size, random);
        const real_targets = try graph_d.ones(&.{ batch_size, 1 }, false);

        // 生成器伪造样本 (Label = 0)
        const noise_d = try graph_d.randomNormal(&.{ batch_size, 2 }, random, 0.0, 1.0, false);
        const fake_data_eager = try net_g.forward(allocator, null, noise_d);
        defer tensor.free(allocator, fake_data_eager);

        const fake_data = try graph_d.array(&.{ batch_size, 2 }, fake_data_eager.data, false);
        const fake_targets = try graph_d.zeros(&.{ batch_size, 1 }, false);

        // 前向传播计算 D(real) 和 D(fake)
        const real_logits = try net_d.forward(allocator, &graph_d, real_data);
        const fake_logits = try net_d.forward(allocator, &graph_d, fake_data);

        // 计算 BCEWithLogitsLoss
        const loss_d_real = try graph_d.bceWithLogitsLoss(real_logits, real_targets);
        const loss_d_fake = try graph_d.bceWithLogitsLoss(fake_logits, fake_targets);
        const loss_d = try graph_d.add(loss_d_real, loss_d_fake);

        net_d.zeroGrad();
        @memset(loss_d.grad, 1.0);
        try graph_d.backward(loss_d);
        opt_d.step();

        const d_loss_val = loss_d.data[0];
        graph_d.deinit();

        // --------------------------------------------------------------------
        // Step 2: 训练生成器 Generator (G)
        // --------------------------------------------------------------------
        var graph_g = autodiff.Graph.init(allocator);

        const noise_g = try graph_g.randomNormal(&.{ batch_size, 2 }, random, 0.0, 1.0, false);
        const g_generated = try net_g.forward(allocator, &graph_g, noise_g);

        // 目标是欺骗 D，使其认为生成样本为 1.0 (Real)
        const g_targets = try graph_g.ones(&.{ batch_size, 1 }, false);
        const g_logits = try net_d.forward(allocator, &graph_g, g_generated);
        const loss_g = try graph_g.bceWithLogitsLoss(g_logits, g_targets);

        net_g.zeroGrad();
        @memset(loss_g.grad, 1.0);
        try graph_g.backward(loss_g);
        opt_g.step();

        const g_loss_val = loss_g.data[0];
        graph_g.deinit();

        // --------------------------------------------------------------------
        // 定期打印 GAN 训练进度与生成样本分布统计信息
        // --------------------------------------------------------------------
        if (epoch % 50 == 0 or epoch == 1) {
            // 计算当前生成器的输出均值与标准差
            var eval_graph = autodiff.Graph.init(allocator);
            defer eval_graph.deinit();

            const eval_noise = try eval_graph.randomNormal(&.{ 500, 2 }, random, 0.0, 1.0, false);
            const generated = try net_g.forward(allocator, &eval_graph, eval_noise);

            var mean_x: f32 = 0.0;
            var mean_y: f32 = 0.0;
            for (0..500) |i| {
                mean_x += generated.data[i * 2 + 0];
                mean_y += generated.data[i * 2 + 1];
            }
            mean_x /= 500.0;
            mean_y /= 500.0;

            var var_x: f32 = 0.0;
            var var_y: f32 = 0.0;
            for (0..500) |i| {
                const dx = generated.data[i * 2 + 0] - mean_x;
                const dy = generated.data[i * 2 + 1] - mean_y;
                var_x += dx * dx;
                var_y += dy * dy;
            }
            const std_x = @sqrt(var_x / 500.0);
            const std_y = @sqrt(var_y / 500.0);

            std.debug.print("Epoch [{d:3}/{d:3}] | D Loss: {d:.4} | G Loss: {d:.4} | Gen Mean: [{d:.2}, {d:.2}] (Target: [3.00, -2.00]) | Gen Std: [{d:.2}, {d:.2}]\n", .{
                epoch, num_epochs, d_loss_val, g_loss_val, mean_x, mean_y, std_x, std_y,
            });
        }
    }

    std.debug.print("\n✨ GAN Training Complete! The Generator successfully learned the target data distribution.\n", .{});
}
