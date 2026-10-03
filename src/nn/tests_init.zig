const std = @import("std");
const testing_init = @import("testing_init.zig");
const Tensor = @import("../tensor.zig").Tensor;
const autodiff = @import("../autodiff.zig");
const nn = @import("../nn.zig");

const Nonlinearity = nn.Nonlinearity;
const initWeights = nn.initWeights;
const Linear = nn.Linear;
const ReLU = nn.ReLU;
const Tanh = nn.Tanh;
const sequential = nn.sequential;

test "Weight initialization methods and Linear initWithOptions" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n_elements: usize = 20000;
    const buf = try allocator.alloc(f32, n_elements);
    defer allocator.free(buf);

    // 1. Zeros
    initWeights(random, buf, 100, 100, .zeros);
    for (buf) |v| try std.testing.expectEqual(@as(f32, 0.0), v);

    // 2. Ones
    initWeights(random, buf, 100, 100, .ones);
    for (buf) |v| try std.testing.expectEqual(@as(f32, 1.0), v);

    // 3. Constant
    initWeights(random, buf, 100, 100, .{ .constant = 3.14 });
    for (buf) |v| try std.testing.expectApproxEqAbs(@as(f32, 3.14), v, 1e-5);

    // Helper to calculate mean and variance
    const calcStats = struct {
        fn run(slice: []const f32) struct { mean: f32, variance: f32 } {
            var sum: f64 = 0.0;
            for (slice) |v| sum += v;
            const mean: f64 = sum / @as(f64, @floatFromInt(slice.len));
            var var_sum: f64 = 0.0;
            for (slice) |v| {
                const diff = @as(f64, v) - mean;
                var_sum += diff * diff;
            }
            const variance: f64 = var_sum / @as(f64, @floatFromInt(slice.len));
            return .{ .mean = @floatCast(mean), .variance = @floatCast(variance) };
        }
    }.run;

    // 4. He Normal: Var = gain^2 / fan_in = 2 / 100 = 0.02
    initWeights(random, buf, 100, 100, .{ .he_normal = .{} });
    const he_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), he_stats.mean, 0.015);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), he_stats.variance, 0.003);

    // 5. Xavier Normal: Var = gain^2 * 2 / (fan_in + fan_out) = 2 / 200 = 0.01
    initWeights(random, buf, 100, 100, .{ .xavier_normal = .{} });
    const xavier_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), xavier_stats.mean, 0.015);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), xavier_stats.variance, 0.002);

    // 6. LeCun Normal: Var = 1 / fan_in = 1 / 100 = 0.01
    initWeights(random, buf, 100, 100, .lecun_normal);
    const lecun_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lecun_stats.mean, 0.015);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), lecun_stats.variance, 0.002);

    // 7. Linear built-in resetParameters specifying nonlinearity
    var lin_tanh = try Linear.init(allocator, 100, 100);
    defer nn.deinitModel(&lin_tanh, allocator);
    lin_tanh.resetParameters(random, .{
        .nonlinearity = .tanh, // Gain = 5/3 ~ 1.6667 -> Xavier Normal with Gain
        .bias_init = .{ .constant = 0.5 },
    });

    const tanh_stats = calcStats(lin_tanh.weight.data);
    // Var = gain^2 * 2 / (100 + 100) = (25 / 9) * 2 / 200 = 25 / 900 ~ 0.02778
    try std.testing.expectApproxEqAbs(@as(f32, 0.02778), tanh_stats.variance, 0.005);
    for (lin_tanh.bias.data) |b| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.5), b, 1e-5);
    }
    // 库内置初始化不标记自定义初始化
    try std.testing.expect(!lin_tanh.weight.is_custom_initialized);
    try std.testing.expect(!lin_tanh.bias.is_custom_initialized);
}

test "initModel infers per-layer initialization from the forward graph" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(1234);
    const random = prng.random();

    // 依据前向计算图自动推导：
    // fc1 -> ReLU (下游为 ReLU -> He Normal, gain=sqrt(2))
    // fc2 -> Tanh (下游为 Tanh -> Xavier Normal, gain=5/3)
    // fc3 -> 无激活函数 (Linear, gain=1.0)
    var model = sequential(.{
        try Linear.init(allocator, 100, 100),
        ReLU{},
        try Linear.init(allocator, 100, 100),
        Tanh{},
        try Linear.init(allocator, 100, 10),
    });
    defer nn.deinitModel(&model, allocator);

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();
    _ = try model.forward(&graph, try graph.ones(&.{ 2, 100 }, false));
    try nn.initModel(&model, &graph, random);

    const calcVar = struct {
        fn run(slice: []const f32) f32 {
            var sum: f64 = 0.0;
            for (slice) |v| sum += v;
            const mean = sum / @as(f64, @floatFromInt(slice.len));
            var var_sum: f64 = 0.0;
            for (slice) |v| {
                const diff = @as(f64, v) - mean;
                var_sum += diff * diff;
            }
            return @floatCast(var_sum / @as(f64, @floatFromInt(slice.len)));
        }
    }.run;

    // 统计方差验证
    // fc1 (ReLU): Var = 2 / 100 = 0.02
    const var_fc1 = calcVar(model.layers.@"0".weight.data);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), var_fc1, 0.003);

    // fc2 (Tanh): Var = (5/3)^2 * 2 / 200 = 25 / 900 ~ 0.02778
    const var_fc2 = calcVar(model.layers.@"2".weight.data);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02778), var_fc2, 0.004);

    // fc3 (Linear/Logits): Var = 1.0^2 / 100 = 0.01 (无放大！)
    const var_fc3 = calcVar(model.layers.@"4".weight.data);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), var_fc3, 0.003);
}

test "Graph.computeParamFans derives fan_in and fan_out from the consuming op" {
    const allocator = std.testing.allocator;

    var linear = try Linear.init(allocator, 5, 7);
    defer nn.deinitModel(&linear, allocator);
    var conv = try nn.Conv2D.init(allocator, 3, 8, 3);
    defer nn.deinitModel(&conv, allocator);
    var deconv = try nn.ConvTranspose2D.init(allocator, 6, 2, 3, 1, 0, true);
    defer nn.deinitModel(&deconv, allocator);

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    // MatMul 右操作数 y = x W，W: [in=5, out=7]
    _ = try linear.forward(&graph, try graph.ones(&.{ 2, 5 }, false));
    // MatMul 左操作数 y = W x，W: [out=6, in=4]
    const w_left = try graph.tensorND(&.{ 6, 4 }, true);
    _ = try graph.matmul(w_left, try graph.ones(&.{ 4, 2 }, false));
    // 先转置再投影 y = x W^T，W: [out=7, in=5]
    const w_t = try graph.tensorND(&.{ 7, 5 }, true);
    _ = try graph.matmul(try graph.ones(&.{ 2, 5 }, false), try graph.transpose(w_t, 0, 1));
    // Conv2D 卷积核 [out_c=8, in_c=3, 3, 3]
    _ = try conv.forward(&graph, try graph.ones(&.{ 1, 3, 6, 6 }, false));
    // ConvTranspose2D 卷积核 [in_c=6, out_c=2, 3, 3]
    _ = try deconv.forward(&graph, try graph.ones(&.{ 1, 6, 4, 4 }, false));

    const expectFans = struct {
        fn run(fans: autodiff.graph_init.ParamFans, fan_in: usize, fan_out: usize) !void {
            try std.testing.expectEqual(fan_in, fans.fan_in);
            try std.testing.expectEqual(fan_out, fans.fan_out);
        }
    }.run;
    try expectFans(graph.computeParamFans(linear.weight), 5, 7);
    try expectFans(graph.computeParamFans(w_left), 4, 6);
    try expectFans(graph.computeParamFans(w_t), 5, 7);
    try expectFans(graph.computeParamFans(conv.weight), 3 * 9, 8 * 9);
    try expectFans(graph.computeParamFans(deconv.weight), 6 * 9, 2 * 9);
}

test "initModel uses the transposed-convolution kernel layout for ConvTranspose2D fans" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();

    // 卷积核 [in_c=32, out_c=4, 3, 3]：fan_in = 32·9 = 288，下游无激活函数 -> He Normal (gain=1)，Var = 1/288
    var deconv = try nn.ConvTranspose2D.init(allocator, 32, 4, 3, 1, 0, true);
    defer nn.deinitModel(&deconv, allocator);

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();
    _ = try deconv.forward(&graph, try graph.ones(&.{ 1, 32, 4, 4 }, false));
    try nn.initModel(&deconv, &graph, random);

    var sum: f64 = 0.0;
    for (deconv.weight.data) |v| sum += v;
    const mean = sum / @as(f64, @floatFromInt(deconv.weight.data.len));
    var var_sum: f64 = 0.0;
    for (deconv.weight.data) |v| {
        const diff = @as(f64, v) - mean;
        var_sum += diff * diff;
    }
    const variance: f32 = @floatCast(var_sum / @as(f64, @floatFromInt(deconv.weight.data.len)));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 288.0), variance, 0.0006);
}

test "Graph.initWeights dynamically infers activations and respects customInit" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(999);
    const random = prng.random();

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    // 1. 各层通过 init 创建（只分配内存，参数保持全零）并设置人类可读名字
    var fc_relu = try Linear.init(allocator, 100, 100);
    defer nn.deinitModel(&fc_relu, allocator);
    var names = std.heap.ArenaAllocator.init(allocator);
    defer names.deinit();
    try nn.nameModules(&fc_relu, names.allocator(), "dense_relu_1");

    var fc_tanh = try Linear.init(allocator, 100, 100);
    defer nn.deinitModel(&fc_tanh, allocator);
    try nn.nameModules(&fc_tanh, names.allocator(), "dense_tanh_2");

    // 库外用户模块：定义 customInit，由 nn.initModel 调用并标记为自定义初始化
    const SpecialHead = struct {
        head: Linear,

        pub fn customInit(self: *@This(), rnd: std.Random) void {
            self.head.resetParameters(rnd, .{
                .nonlinearity = .linear,
                .bias_init = .{ .constant = 3.14 },
            });
        }
    };

    var special = SpecialHead{ .head = try Linear.init(allocator, 100, 10) };
    defer nn.deinitModel(&special.head, allocator);
    try nn.nameModules(&special.head, names.allocator(), "special_head");
    const fc_custom = &special.head;

    // 2. 在构造/连接期通过各类 Operation 将图自然动态串联起来
    const x = try graph.zeros(&.{ 2, 100 }, false);
    x.setName("features_input");

    const z1 = try graph.addBias(try graph.matmul(x, fc_relu.weight), fc_relu.bias);
    const a1 = try graph.relu(z1); // 后续接 ReLU
    a1.setName("relu_activation_1");

    const z2 = try graph.addBias(try graph.matmul(a1, fc_tanh.weight), fc_tanh.bias);
    const a2 = try graph.tanh(z2); // 后续接 Tanh

    var fc_out = try Linear.init(allocator, 10, 2);
    defer nn.deinitModel(&fc_out, allocator);
    try nn.nameModules(&fc_out, names.allocator(), "logits_out");

    const logits = try graph.addBias(try graph.matmul(a2, fc_custom.weight), fc_custom.bias);
    logits.setName("custom_head_logits");

    const final_out = try graph.addBias(try graph.matmul(logits, fc_out.weight), fc_out.bias);
    final_out.setName("network_final_out");

    // 3. 图建立之后初始化：外部模块定义了 customInit -> initModel 调用之并标记其参数；
    //    其余参数由 graph.initWeights 依据下游激活函数推导初始化，不覆盖 customInit 的结果
    try nn.initModel(&special, &graph, random);
    try std.testing.expect(fc_custom.weight.is_custom_initialized);
    try std.testing.expect(fc_custom.bias.is_custom_initialized);
    const custom_weight_sample = fc_custom.weight.data[0];
    graph.initWeights(random);

    const calcVar = struct {
        fn run(slice: []const f32) f32 {
            var sum: f64 = 0.0;
            for (slice) |v| sum += v;
            const mean = sum / @as(f64, @floatFromInt(slice.len));
            var var_sum: f64 = 0.0;
            for (slice) |v| {
                const diff = @as(f64, v) - mean;
                var_sum += diff * diff;
            }
            return @floatCast(var_sum / @as(f64, @floatFromInt(slice.len)));
        }
    }.run;

    // 5. 校验：
    // fc_relu 后面探查到 ReLU -> 自动赋 He Normal (Var = 2 / 100 = 0.02)
    const var_relu = calcVar(fc_relu.weight.data);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), var_relu, 0.003);

    // fc_tanh 后面探查到 Tanh -> 自动赋 Xavier Normal (Var = (5/3)^2 * 2 / 200 ~ 0.02778)
    const var_tanh = calcVar(fc_tanh.weight.data);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02778), var_tanh, 0.004);

    // fc_custom 已经过 customInit -> 绝对不会被 Graph 重写覆盖！
    try std.testing.expectEqual(custom_weight_sample, fc_custom.weight.data[0]);
    for (fc_custom.bias.data) |b| {
        try std.testing.expectApproxEqAbs(@as(f32, 3.14), b, 1e-5);
    }

    // 6. 测试 formatInitReport 能够正常输出自定义名称、非参数节点 (Input, Activation) 和各种初始化状态
    const report_str = try graph.formatInitReport(allocator);
    defer allocator.free(report_str);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "dense_relu_1.weight") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "dense_relu_1.bias") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "dense_tanh_2.weight") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "special_head.weight") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "logits_out.weight") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "features_input") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "relu_activation_1") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "custom_head_logits") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "network_final_out") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "Input") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "Activation") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "Param") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "CUSTOM_INIT") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "AUTO_GRAPH") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "OP_OUTPUT") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "ReLU") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "Tanh") != null);
    try std.testing.expect(std.mem.indexOf(u8, report_str, "Linear (None)") != null);
}

test "Comprehensive coverage of all InitMethod strategies" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(54321);
    const random = prng.random();

    const n = 10000;
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);

    const calcStats = struct {
        fn run(slice: []const f32) struct { mean: f32, variance: f32, min: f32, max: f32 } {
            var sum: f64 = 0.0;
            var min_v: f32 = slice[0];
            var max_v: f32 = slice[0];
            for (slice) |v| {
                sum += v;
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }
            const mean = sum / @as(f64, @floatFromInt(slice.len));
            var var_sum: f64 = 0.0;
            for (slice) |v| {
                const diff = @as(f64, v) - mean;
                var_sum += diff * diff;
            }
            return .{
                .mean = @floatCast(mean),
                .variance = @floatCast(var_sum / @as(f64, @floatFromInt(slice.len))),
                .min = min_v,
                .max = max_v,
            };
        }
    }.run;

    // 1. Zeros, Ones, Constant
    initWeights(random, buf, 100, 100, .zeros);
    for (buf) |v| try std.testing.expectEqual(@as(f32, 0.0), v);

    initWeights(random, buf, 100, 100, .ones);
    for (buf) |v| try std.testing.expectEqual(@as(f32, 1.0), v);

    initWeights(random, buf, 100, 100, .{ .constant = 42.0 });
    for (buf) |v| try std.testing.expectEqual(@as(f32, 42.0), v);

    // 2. Normal (mean=2.0, std=0.5)
    initWeights(random, buf, 100, 100, .{ .normal = .{ .mean = 2.0, .std = 0.5 } });
    const norm_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), norm_stats.mean, 0.03);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), norm_stats.variance, 0.02);

    // 3. Uniform (min=-1.0, max=3.0) -> mean=1.0, var=(3 - (-1))^2 / 12 = 16/12 ~ 1.3333
    initWeights(random, buf, 100, 100, .{ .uniform = .{ .min = -1.0, .max = 3.0 } });
    const uni_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), uni_stats.mean, 0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 1.3333), uni_stats.variance, 0.05);
    try std.testing.expect(uni_stats.min >= -1.0);
    try std.testing.expect(uni_stats.max <= 3.0);

    // 4. Xavier Uniform: limit = gain * sqrt(6 / (100 + 100)) = sqrt(6/200) = sqrt(0.03) ~ 0.1732
    // Var = limit^2 / 3 = 0.03 / 3 = 0.01
    initWeights(random, buf, 100, 100, .{ .xavier_uniform = .{} });
    const xu_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), xu_stats.mean, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), xu_stats.variance, 0.002);
    try std.testing.expect(xu_stats.max <= 0.1733);
    try std.testing.expect(xu_stats.min >= -0.1733);

    // 5. He Uniform: limit = gain * sqrt(3 / 100) = sqrt(2) * sqrt(0.03) = sqrt(0.06) ~ 0.2449
    // Var = limit^2 / 3 = 0.06 / 3 = 0.02
    initWeights(random, buf, 100, 100, .{ .he_uniform = .{} });
    const hu_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), hu_stats.mean, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), hu_stats.variance, 0.003);

    // 6. LeCun Uniform: limit = sqrt(3 / 100) ~ 0.1732, Var = 0.03 / 3 = 0.01
    initWeights(random, buf, 100, 100, .lecun_uniform);
    const lu_stats = calcStats(buf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lu_stats.mean, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), lu_stats.variance, 0.002);
}

test "Built-in library initialization never marks is_custom_initialized" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(2026);
    const random = prng.random();

    var lin = try Linear.init(allocator, 8, 4);
    defer nn.deinitModel(&lin, allocator);
    lin.resetParameters(random, .{});
    try std.testing.expect(!lin.weight.is_custom_initialized);
    try std.testing.expect(!lin.bias.is_custom_initialized);

    var conv = try nn.Conv2D.init(allocator, 3, 4, 3);
    defer nn.deinitModel(&conv, allocator);
    conv.resetParameters(random, .{});
    try std.testing.expect(!conv.weight.is_custom_initialized);
    try std.testing.expect(!conv.bias.is_custom_initialized);

    var deconv = try nn.ConvTranspose2D.init(allocator, 4, 3, 3, 1, 0, true);
    defer nn.deinitModel(&deconv, allocator);
    deconv.resetParameters(random, .{});
    try std.testing.expect(!deconv.weight.is_custom_initialized);
    try std.testing.expect(!deconv.bias.?.is_custom_initialized);

    var emb = try nn.Embedding.init(allocator, 32, 8);
    defer nn.deinitModel(&emb, allocator);
    emb.resetParameters(random, .{});
    try std.testing.expect(!emb.weight.is_custom_initialized);

    var lora = try nn.LoRALinear.initWithBias(allocator, 8, 4, 2, 4.0, true);
    defer nn.deinitModel(&lora, allocator);
    lora.resetParameters(random, .{});
    try std.testing.expect(!lora.weight.is_custom_initialized);
    try std.testing.expect(!lora.lora_a.is_custom_initialized);
    try std.testing.expect(!lora.lora_b.is_custom_initialized);
    try std.testing.expect(!lora.bias.?.is_custom_initialized);

    // 依据计算图的默认初始化同样不标记
    var lstm = try nn.LSTM.init(allocator, 8, 8);
    defer nn.deinitModel(&lstm, allocator);
    try testing_init.initRecurrent(&lstm, allocator, random);
    try std.testing.expect(!lstm.cell.w_ih_f.weight.is_custom_initialized);
    try std.testing.expect(!lstm.cell.w_ih_f.bias.is_custom_initialized);
}

test "initModel runs external customInit first and initializes the rest from the forward graph" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(31415);
    const random = prng.random();

    // 0. 库层 init 只分配内存：参数全零，未消耗任何随机数
    var bare = try Linear.init(allocator, 4, 3);
    defer nn.deinitModel(&bare, allocator);
    for (bare.weight.data) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
    for (bare.bias.data) |v| try std.testing.expectEqual(@as(f32, 0.0), v);

    // 1. 全部参数由确定性 customInit(self) 指定：不依赖计算图内容 (空图即可)，不消耗随机数，参数被标记为自定义初始化
    const CustomBlock = struct {
        fc: Linear,
        norm: nn.LayerNorm,

        pub fn customInit(self: *@This()) void {
            @memset(self.fc.weight.data, 0.25);
            @memset(self.fc.bias.data, -1.0);
        }
    };
    var custom = CustomBlock{
        .fc = try Linear.init(allocator, 4, 3),
        .norm = try nn.LayerNorm.init(allocator, 3, 1e-5),
    };
    defer nn.deinitModel(&custom.fc, allocator);
    defer nn.deinitModel(&custom.norm, allocator);
    {
        var empty_graph = autodiff.Graph.init(allocator);
        defer empty_graph.deinit();
        try nn.initModel(&custom, &empty_graph, random);
    }
    for (custom.fc.weight.data) |v| try std.testing.expectEqual(@as(f32, 0.25), v);
    for (custom.fc.bias.data) |v| try std.testing.expectEqual(@as(f32, -1.0), v);
    try std.testing.expect(custom.fc.weight.is_custom_initialized);
    try std.testing.expect(custom.norm.weight.is_custom_initialized);
    try std.testing.expect(custom.norm.bias.is_custom_initialized);

    // 2. 外部未指定：依据计算图初始化 (Linear 下游为 Tanh -> Xavier；LSTM 遗忘门偏置 -> 1.0)，不标记自定义初始化
    const PlainBlock = struct {
        fc: Linear,
        cell: nn.LSTMCell,
    };
    var plain = PlainBlock{
        .fc = try Linear.init(allocator, 4, 4),
        .cell = try nn.LSTMCell.init(allocator, 4, 4),
    };
    defer nn.deinitModel(&plain.fc, allocator);
    defer nn.deinitModel(&plain.cell, allocator);
    {
        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();
        const x = try graph.ones(&.{ 1, 4 }, false);
        const h0 = try graph.zeros(&.{ 1, 4 }, false);
        const c0 = try graph.zeros(&.{ 1, 4 }, false);
        const feat = try graph.tanh(try plain.fc.forward(&graph, x));
        _ = try plain.cell.forward(&graph, feat, h0, c0);
        try nn.initModel(&plain, &graph, random);
    }
    var nonzero = false;
    for (plain.fc.weight.data) |v| {
        if (v != 0.0) nonzero = true;
    }
    try std.testing.expect(nonzero);
    for (plain.cell.w_ih_f.bias.data) |v| try std.testing.expectEqual(@as(f32, 1.0), v);
    try std.testing.expect(!plain.fc.weight.is_custom_initialized);
    try std.testing.expect(!plain.cell.w_ih_f.bias.is_custom_initialized);

    // 3. Sequential：定义了 customInit 的外部层被调用并标记，库层依据计算图中的下游激活函数初始化
    const ConstLayer = struct {
        fc: Linear,

        pub fn customInit(self: *@This()) void {
            @memset(self.fc.weight.data, 0.5);
            @memset(self.fc.bias.data, 0.0);
        }

        pub fn forward(self: @This(), graph: *autodiff.Graph, x: *Tensor) !*Tensor {
            return self.fc.forward(graph, x);
        }

        pub fn deinit(self: @This(), alloc: std.mem.Allocator) void {
            self.fc.deinit(alloc);
        }
    };
    var model = sequential(.{
        try Linear.init(allocator, 4, 4),
        ReLU{},
        ConstLayer{ .fc = try Linear.init(allocator, 4, 2) },
    });
    defer nn.deinitModel(&model, allocator);
    try testing_init.initFromOnes(&model, allocator, random, &.{ 2, 4 });
    try std.testing.expect(!model.layers.@"0".weight.is_custom_initialized);
    try std.testing.expect(model.layers.@"0".weight.data[0] != 0.0 or model.layers.@"0".weight.data[1] != 0.0);
    for (model.layers.@"2".fc.weight.data) |v| try std.testing.expectEqual(@as(f32, 0.5), v);
    try std.testing.expect(model.layers.@"2".fc.weight.is_custom_initialized);
    try std.testing.expect(model.layers.@"2".fc.bias.is_custom_initialized);

    // 4. 嵌套：未定义 customInit 的父模块递归查找，子模块的 customInit (含随机数签名) 在任意层级被调用
    const RandomHead = struct {
        fc: Linear,

        pub fn customInit(self: *@This(), rnd: std.Random) void {
            for (self.fc.weight.data) |*w| w.* = 2.0 + rnd.float(f32);
            @memset(self.fc.bias.data, 0.0);
        }
    };
    const Outer = struct {
        backbone: Linear,
        head: RandomHead,
    };
    var outer = Outer{
        .backbone = try Linear.init(allocator, 4, 4),
        .head = .{ .fc = try Linear.init(allocator, 4, 2) },
    };
    defer nn.deinitModel(&outer.backbone, allocator);
    defer nn.deinitModel(&outer.head.fc, allocator);
    {
        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();
        const x = try graph.ones(&.{ 1, 4 }, false);
        _ = try outer.head.fc.forward(&graph, try graph.relu(try outer.backbone.forward(&graph, x)));
        try nn.initModel(&outer, &graph, random);
    }
    try std.testing.expect(!outer.backbone.weight.is_custom_initialized);
    for (outer.head.fc.weight.data) |v| try std.testing.expect(v >= 2.0 and v < 3.0);
    try std.testing.expect(outer.head.fc.weight.is_custom_initialized);

    // 5. 未开启梯度记录的计算图不记录算子，无法据此推导初始化
    {
        var no_grad = autodiff.Graph.initNoGrad(allocator);
        defer no_grad.deinit();
        try std.testing.expectError(error.GraphNotRecordingOps, nn.initModel(&outer, &no_grad, random));
    }
}

test "inspectParameterInit detects models whose weight matrices were never initialized" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();

    const Block = struct {
        fc: Linear,
        norm: nn.LayerNorm,
        lora: nn.LoRALinear,
    };
    var block = Block{
        .fc = try Linear.init(allocator, 4, 4),
        .norm = try nn.LayerNorm.init(allocator, 4, 1e-5),
        .lora = try nn.LoRALinear.initWithBias(allocator, 4, 4, 2, 4.0, true),
    };
    defer nn.deinitModel(&block, allocator);

    // 1. 只调用 init：LayerNorm γ = 1 与偏置等向量形参数不参与判断，全部权重矩阵为 0 -> 判定为未初始化
    const params = try nn.parameters(&block, allocator);
    defer allocator.free(params);
    const before = nn.inspectParameterInit(params);
    try std.testing.expectEqual(@as(usize, 3), before.weight_tensors); // fc.weight, lora_a, lora_b
    try std.testing.expectEqual(@as(usize, 3), before.zero_weight_tensors);
    try std.testing.expect(before.looksUninitialized());

    // 2. 初始化之后：LoRA 旁路 B 按设计仍为 0，但其余权重矩阵非 0 -> 不判定为未初始化
    block.fc.resetParameters(random, .{});
    block.lora.resetParameters(random, .{});
    const after = nn.inspectParameterInit(params);
    try std.testing.expectEqual(@as(usize, 1), after.zero_weight_tensors);
    try std.testing.expect(!after.looksUninitialized());
}
