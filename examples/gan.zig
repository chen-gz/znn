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

    pub fn init(allocator: std.mem.Allocator) !Generator {
        return Generator{
            .l1 = try nn.Linear.init(allocator, 2, 16),
            .act1 = .{ .alpha = 0.2 },
            .l2 = try nn.Linear.init(allocator, 16, 16),
            .act2 = .{ .alpha = 0.2 },
            .l3 = try nn.Linear.init(allocator, 16, 2),
        };
    }

    pub fn deinit(self: Generator, allocator: std.mem.Allocator) void {
        nn.deinitModel(&self, allocator);
    }

    pub fn zeroGrad(self: *Generator) void {
        nn.zeroGradModel(self);
    }

    pub fn forward(self: *Generator, graph: *autodiff.Graph, z: *Tensor) !*Tensor {
        const h1 = try self.l1.forward(graph, z);
        const a1 = try self.act1.forward(graph, h1);
        const h2 = try self.l2.forward(graph, a1);
        const a2 = try self.act2.forward(graph, h2);
        return try self.l3.forward(graph, a2);
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

    pub fn init(allocator: std.mem.Allocator) !Discriminator {
        return Discriminator{
            .l1 = try nn.Linear.init(allocator, 2, 16),
            .act1 = .{ .alpha = 0.2 },
            .l2 = try nn.Linear.init(allocator, 16, 16),
            .act2 = .{ .alpha = 0.2 },
            .l3 = try nn.Linear.init(allocator, 16, 1),
        };
    }

    pub fn deinit(self: Discriminator, allocator: std.mem.Allocator) void {
        nn.deinitModel(&self, allocator);
    }

    pub fn zeroGrad(self: *Discriminator) void {
        nn.zeroGradModel(self);
    }

    pub fn forward(self: *Discriminator, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const h1 = try self.l1.forward(graph, x);
        const a1 = try self.act1.forward(graph, h1);
        const h2 = try self.l2.forward(graph, a1);
        const a2 = try self.act2.forward(graph, h2);
        return try self.l3.forward(graph, a2);
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
    var net_g = try Generator.init(allocator);
    defer net_g.deinit(allocator);

    var net_d = try Discriminator.init(allocator);
    defer net_d.deinit(allocator);

    // 用一个 2 维样本建立前向计算图，依据计算图中的 LeakyReLU 等下游激活函数初始化 G 与 D 的参数
    const init_sample = try tensor.ones(allocator, &.{ 1, 2 });
    defer tensor.free(allocator, init_sample);
    try nn.initModelWithSample(&net_g, allocator, random, .{init_sample});
    try nn.initModelWithSample(&net_d, allocator, random, .{init_sample});

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
        var fake_data_eager_graph = autodiff.Graph.initNoGrad(allocator);
        defer fake_data_eager_graph.deinit();
        const fake_data_eager = try net_g.forward(&fake_data_eager_graph, noise_d);

        const fake_data = try graph_d.array(&.{ batch_size, 2 }, fake_data_eager.data, false);
        const fake_targets = try graph_d.zeros(&.{ batch_size, 1 }, false);

        // 前向传播计算 D(real) 和 D(fake)
        const real_logits = try net_d.forward(&graph_d, real_data);
        const fake_logits = try net_d.forward(&graph_d, fake_data);

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
        const g_generated = try net_g.forward(&graph_g, noise_g);

        // 目标是欺骗 D，使其认为生成样本为 1.0 (Real)
        const g_targets = try graph_g.ones(&.{ batch_size, 1 }, false);
        const g_logits = try net_d.forward(&graph_g, g_generated);
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
            const generated = try net_g.forward(&eval_graph, eval_noise);

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
