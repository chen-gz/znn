const std = @import("std");
const nn = @import("../nn.zig");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const optim = @import("../optim.zig");

const SGDOptimizer = optim.SGDOptimizer;
const SGDConfig = optim.SGDConfig;
const AdamOptimizer = optim.AdamOptimizer;
const AdamConfig = optim.AdamConfig;
const AdamWOptimizer = optim.AdamWOptimizer;
const AdamWConfig = optim.AdamWConfig;
const CosineScheduler = optim.CosineScheduler;
const StepLRScheduler = optim.StepLRScheduler;
const LinearWarmupScheduler = optim.LinearWarmupScheduler;
const ExponentialLRScheduler = optim.ExponentialLRScheduler;
const LRScheduler = optim.LRScheduler;
const clipGradNorm = optim.clipGradNorm;
const clipGradValue = optim.clipGradValue;
const clipGradients = optim.clipGradients;
const GradClipConfig = optim.GradClipConfig;
const OPTIMIZER_MAGIC = optim.OPTIMIZER_MAGIC;
const OPTIMIZER_VERSION = optim.OPTIMIZER_VERSION;
const CheckpointError = optim.CheckpointError;
const CHECKPOINT_HEADER_BYTES = optim.checkpoint.CHECKPOINT_HEADER_BYTES;


test "SGDOptimizer basic and momentum updates" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(42);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    // 1. 测试无动量 SGD
    var opt_plain = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1, .momentum = 0.0 });
    defer opt_plain.deinit();

    linear.weight.data[0] = 1.0;
    linear.weight.grad[0] = 0.5;

    opt_plain.step();
    // w = 1.0 - 0.1 * 0.5 = 0.95
    try testing.expectApproxEqAbs(@as(f32, 0.95), linear.weight.data[0], 1e-5);

    // 2. 测试带动量 SGD
    var opt_mom = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1, .momentum = 0.9 });
    defer opt_mom.deinit();

    linear.weight.data[0] = 1.0;
    linear.weight.grad[0] = 0.5;

    // Step 1: vel = 0.9*0 + 0.1*0.5 = 0.05, w = 1.0 - 0.05 = 0.95
    opt_mom.step();
    try testing.expectApproxEqAbs(@as(f32, 0.95), linear.weight.data[0], 1e-5);

    // Step 2: vel = 0.9*0.05 + 0.1*0.5 = 0.045 + 0.05 = 0.095, w = 0.95 - 0.095 = 0.855
    opt_mom.step();
    try testing.expectApproxEqAbs(@as(f32, 0.855), linear.weight.data[0], 1e-5);
}

test "AdamOptimizer multi-step parameter updates" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(42);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    var opt = try AdamOptimizer.init(allocator, &linear, .{
        .lr = 0.01,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .eps = 1e-8,
    });
    defer opt.deinit();

    linear.weight.data[0] = 1.0;
    linear.weight.grad[0] = 0.2;

    opt.step();
    try testing.expect(linear.weight.data[0] < 1.0);

    const after_step1 = linear.weight.data[0];
    opt.step();
    try testing.expect(linear.weight.data[0] < after_step1);
}

test "AdamWOptimizer weight decay and step" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(42);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    var opt = try AdamWOptimizer.init(allocator, &linear, .{
        .lr = 0.01,
        .beta1 = 0.9,
        .beta2 = 0.95,
        .eps = 1e-8,
        .weight_decay = 0.1,
    });
    defer opt.deinit();

    linear.weight.data[0] = 1.0;
    linear.weight.grad[0] = 0.0; // 即使梯度为 0，解耦权重衰减也应该减小权重

    opt.step();
    try testing.expect(linear.weight.data[0] < 1.0);
}

test "CosineScheduler warmup and decay" {
    const testing = std.testing;
    const sched = CosineScheduler.init(1e-3, 1e-4, 10, 100);

    // Warmup: step 0 -> 0, step 5 -> 0.5 * max_lr, step 10 -> max_lr
    try testing.expectApproxEqAbs(@as(f32, 0.0), sched.getLR(0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 5e-4), sched.getLR(5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1e-3), sched.getLR(10), 1e-6);

    // Midpoint: step 55 -> halfway between max_lr and min_lr
    const mid_lr = sched.getLR(55);
    try testing.expect(mid_lr < 1e-3 and mid_lr > 1e-4);

    // End: step 100 -> min_lr
    try testing.expectApproxEqAbs(@as(f32, 1e-4), sched.getLR(100), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1e-4), sched.getLR(120), 1e-6);
}

test "StepLR and ExponentialLR schedulers" {
    const testing = std.testing;

    const step_sched = StepLRScheduler.init(0.1, 10, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.1), step_sched.getLR(0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), step_sched.getLR(9), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.05), step_sched.getLR(10), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.05), step_sched.getLR(19), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.025), step_sched.getLR(20), 1e-6);

    const exp_sched = ExponentialLRScheduler.init(0.1, 0.9);
    try testing.expectApproxEqAbs(@as(f32, 0.1), exp_sched.getLR(0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.09), exp_sched.getLR(1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.081), exp_sched.getLR(2), 1e-6);
}

test "LinearWarmupScheduler and LRScheduler union" {
    const testing = std.testing;

    const warmup = LinearWarmupScheduler.init(0.01, 0.1, 10);
    try testing.expectApproxEqAbs(@as(f32, 0.01), warmup.getLR(0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.055), warmup.getLR(5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), warmup.getLR(10), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), warmup.getLR(20), 1e-6);

    const sched_union = LRScheduler{ .warmup = warmup };
    try testing.expectApproxEqAbs(@as(f32, 0.055), sched_union.getLR(5), 1e-6);
}

test "clipGradNorm and clipGradValue" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var t1 = try tensor.zeros(allocator, &.{ 2, 2 });
    defer tensor.free(allocator, t1);
    t1.requires_grad = true;
    t1.grad = try allocator.alloc(f32, 4);

    @memcpy(t1.grad, &[_]f32{ 3.0, 4.0, 0.0, 0.0 }); // norm = sqrt(9 + 16) = 5.0

    var params = [_]*Tensor{t1};
    const norm = clipGradNorm(&params, 2.5);
    try testing.expectApproxEqAbs(@as(f32, 5.0), norm, 1e-5);
    // After clipping with max_norm=2.5, norm should be scaled by 2.5/5.0 = 0.5
    try testing.expectApproxEqAbs(@as(f32, 1.5), t1.grad[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), t1.grad[1], 1e-5);

    // Value clipping test
    @memcpy(t1.grad, &[_]f32{ 5.0, -10.0, 0.5, -0.2 });
    clipGradValue(&params, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), t1.grad[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -1.0), t1.grad[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), t1.grad[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -0.2), t1.grad[3], 1e-5);
}

test "Optimizer checkpoint serialization and resumption" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(42);
    var linear = try nn.Linear.init(allocator, 3, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    // 1. Test SGDOptimizer checkpointing
    {
        var opt1 = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.05, .momentum = 0.9 });
        defer opt1.deinit();

        linear.weight.data[0] = 2.0;
        linear.weight.grad[0] = 1.0;
        opt1.step(); // updates momentum buffer and step_count

        const ckpt_path = "test_sgd_ckpt.bin";
        try opt1.saveCheckpoint(testing.io, ckpt_path);

        var opt2 = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.01, .momentum = 0.0 });
        defer opt2.deinit();

        try opt2.loadCheckpoint(testing.io, ckpt_path);
        try testing.expectEqual(opt1.step_count, opt2.step_count);
        try testing.expectApproxEqAbs(opt1.lr, opt2.lr, 1e-6);
        try testing.expect(opt2.velocities != null);
        try testing.expectApproxEqAbs(opt1.velocities.?[0][0], opt2.velocities.?[0][0], 1e-6);

        std.Io.Dir.cwd().deleteFile(testing.io, ckpt_path) catch {};
    }

    // 2. Test AdamWOptimizer checkpointing
    {
        var opt1 = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.002, .weight_decay = 0.05 });
        defer opt1.deinit();

        linear.weight.data[0] = 1.5;
        linear.weight.grad[0] = 0.3;
        opt1.step();
        opt1.step();

        const ckpt_path = "test_adamw_ckpt.bin";
        try opt1.saveCheckpoint(testing.io, ckpt_path);

        var opt2 = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.001 });
        defer opt2.deinit();

        try opt2.loadCheckpoint(testing.io, ckpt_path);
        try testing.expectEqual(opt1.step_count, opt2.step_count);
        try testing.expectApproxEqAbs(opt1.config.lr, opt2.config.lr, 1e-6);
        try testing.expectApproxEqAbs(opt1.m[0][0], opt2.m[0][0], 1e-6);
        try testing.expectApproxEqAbs(opt1.v[0][0], opt2.v[0][0], 1e-6);

        std.Io.Dir.cwd().deleteFile(testing.io, ckpt_path) catch {};
    }

    // 3. Test AdamOptimizer checkpointing
    {
        var opt1 = try AdamOptimizer.init(allocator, &linear, .{ .lr = 0.003 });
        defer opt1.deinit();

        linear.weight.data[0] = 2.5;
        linear.weight.grad[0] = 0.4;
        opt1.step();

        const ckpt_path = "test_adam_ckpt.bin";
        try opt1.saveCheckpoint(testing.io, ckpt_path);

        var opt2 = try AdamOptimizer.init(allocator, &linear, .{ .lr = 0.001 });
        defer opt2.deinit();

        try opt2.loadCheckpoint(testing.io, ckpt_path);
        try testing.expectEqual(opt1.step_count, opt2.step_count);
        try testing.expectApproxEqAbs(opt1.lr, opt2.lr, 1e-6);
        try testing.expectApproxEqAbs(opt1.m[0][0], opt2.m[0][0], 1e-6);
        try testing.expectApproxEqAbs(opt1.v[0][0], opt2.v[0][0], 1e-6);

        std.Io.Dir.cwd().deleteFile(testing.io, ckpt_path) catch {};
    }
}

test "Optimizer checkpoint error conditions and corruption resilience" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(99);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    // 1. Test corrupt magic header
    {
        const corrupt_file = "test_corrupt_magic.bin";
        var file = try std.Io.Dir.cwd().createFile(testing.io, corrupt_file, .{});
        defer file.close(testing.io);
        defer std.Io.Dir.cwd().deleteFile(testing.io, corrupt_file) catch {};

        var buf: [64]u8 = undefined;
        var file_writer = file.writer(testing.io, &buf);
        try file_writer.interface.writeAll("BADMAGICDATA12345678");
        try file_writer.interface.flush();

        var opt = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.01 });
        defer opt.deinit();

        try testing.expectError(error.InvalidCheckpointMagic, opt.loadCheckpoint(testing.io, corrupt_file));
    }

    // 2. Test optimizer type mismatch (saved as SGD, load with AdamW)
    {
        const sgd_file = "test_type_mismatch.bin";
        var opt_sgd = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.01 });
        defer opt_sgd.deinit();
        try opt_sgd.saveCheckpoint(testing.io, sgd_file);
        defer std.Io.Dir.cwd().deleteFile(testing.io, sgd_file) catch {};

        var opt_adamw = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.01 });
        defer opt_adamw.deinit();

        try testing.expectError(error.OptimizerTypeMismatch, opt_adamw.loadCheckpoint(testing.io, sgd_file));
    }

    // 3. Test parameter count mismatch
    {
        const param_file = "test_param_mismatch.bin";
        var opt_sgd = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.01 });
        defer opt_sgd.deinit();
        try opt_sgd.saveCheckpoint(testing.io, param_file);
        defer std.Io.Dir.cwd().deleteFile(testing.io, param_file) catch {};

        // Different model with more layers (4 parameters instead of 2)
        var seq = nn.sequential(.{
            try nn.Linear.init(allocator, 2, 2),
            try nn.Linear.init(allocator, 2, 2),
        });
        defer nn.deinitModel(&seq, allocator);
        {
            var init_graph = @import("../autodiff.zig").Graph.initDefault(allocator);
            defer init_graph.deinit();
            _ = try seq.forward(&init_graph, try init_graph.ones(&.{ 1, 2 }, false));
            try nn.initModel(&seq, &init_graph, prng.random());
        }

        var opt_diff = try SGDOptimizer.init(allocator, &seq, .{ .lr = 0.01 });
        defer opt_diff.deinit();

        try testing.expectError(error.ParamCountMismatch, opt_diff.loadCheckpoint(testing.io, param_file));
    }
}

test "clipGradNorm and clipGradValue extreme and empty edge cases" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 1. Empty parameter slice
    var empty_params = [_]*Tensor{};
    const empty_norm = clipGradNorm(&empty_params, 1.0);
    try testing.expectEqual(@as(f32, 0.0), empty_norm);
    clipGradValue(&empty_params, 1.0);
    const empty_cfg_norm = clipGradients(&empty_params, .{ .norm = 1.0 });
    try testing.expectEqual(@as(f32, 0.0), empty_cfg_norm);
    const empty_cfg_none = clipGradients(&empty_params, .none);
    try testing.expect(empty_cfg_none == null);

    // 2. All zero gradients: norm = 0, ensure no NaN or division by zero
    var t_zero = try tensor.zeros(allocator, &.{ 2, 2 });
    defer tensor.free(allocator, t_zero);
    t_zero.requires_grad = true;
    t_zero.grad = try allocator.alloc(f32, 4);
    @memset(t_zero.grad, 0.0);

    var zero_params = [_]*Tensor{t_zero};
    const zero_norm = clipGradNorm(&zero_params, 1.0);
    try testing.expectEqual(@as(f32, 0.0), zero_norm);
    for (t_zero.grad) |g| {
        try testing.expectEqual(@as(f32, 0.0), g);
    }

    // 3. Non-clipping condition: actual norm <= max_norm
    @memcpy(t_zero.grad, &[_]f32{ 0.3, 0.4, 0.0, 0.0 }); // norm = 0.5 <= 1.0
    const unclipped_norm = clipGradNorm(&zero_params, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 0.5), unclipped_norm, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.3), t_zero.grad[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.4), t_zero.grad[1], 1e-5);

    // 4. max_norm <= 0.0 (scales all gradients to zero)
    const zero_scaled = clipGradNorm(&zero_params, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 0.5), zero_scaled, 1e-5);
    try testing.expectEqual(@as(f32, 0.0), t_zero.grad[0]);
    try testing.expectEqual(@as(f32, 0.0), t_zero.grad[1]);
}

test "Optimizer interface uniformity and scheduler stepping" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(77);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    // Test getLR / setLR / stepWithLR across all 3 optimizers
    var sgd = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1 });
    defer sgd.deinit();
    try testing.expectApproxEqAbs(@as(f32, 0.1), sgd.getLR(), 1e-6);
    sgd.setLR(0.05);
    try testing.expectApproxEqAbs(@as(f32, 0.05), sgd.getLR(), 1e-6);
    sgd.stepWithLR(0.02);
    try testing.expectEqual(@as(u64, 1), sgd.step_count);

    var adam = try AdamOptimizer.init(allocator, &linear, .{ .lr = 0.01 });
    defer adam.deinit();
    try testing.expectApproxEqAbs(@as(f32, 0.01), adam.getLR(), 1e-6);
    adam.setLR(0.005);
    try testing.expectApproxEqAbs(@as(f32, 0.005), adam.getLR(), 1e-6);
    adam.stepWithLR(0.002);
    try testing.expectEqual(@as(u64, 1), adam.step_count);

    var adamw = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.02 });
    defer adamw.deinit();
    try testing.expectApproxEqAbs(@as(f32, 0.02), adamw.getLR(), 1e-6);
    adamw.setLR(0.008);
    try testing.expectApproxEqAbs(@as(f32, 0.008), adamw.getLR(), 1e-6);
    adamw.stepWithLR(0.004);
    try testing.expectEqual(@as(u64, 1), adamw.step_count);

    // Test polymorphic LRScheduler.step()
    var sched = LRScheduler{ .step_lr = StepLRScheduler.init(0.1, 5, 0.5) };
    const lr0 = sched.step(&sgd, 0); // step 0 -> lr=0.1
    try testing.expectApproxEqAbs(@as(f32, 0.1), lr0, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), sgd.getLR(), 1e-6);
    const lr5 = sched.step(&sgd, 5); // step 5 -> lr=0.05
    try testing.expectApproxEqAbs(@as(f32, 0.05), lr5, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.05), sgd.getLR(), 1e-6);
}

test "Adam and AdamW vectorized updates match the scalar reference formulas" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 13 个权重元素：同时覆盖 SIMD 主循环之外的标量尾部
    var prng = std.Random.DefaultPrng.init(5);
    var linear = try nn.Linear.init(allocator, 13, 1);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    const n = linear.weight.data.len;
    const w0 = try allocator.dupe(f32, linear.weight.data);
    defer allocator.free(w0);
    for (linear.weight.grad, 0..) |*g, i| g.* = 0.1 * @as(f32, @floatFromInt(i)) - 0.6;

    const ref_w = try allocator.alloc(f32, n);
    defer allocator.free(ref_w);
    const ref_m = try allocator.alloc(f32, n);
    defer allocator.free(ref_m);
    const ref_v = try allocator.alloc(f32, n);
    defer allocator.free(ref_v);

    // 1. AdamW: w -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)
    {
        const cfg = AdamWConfig{ .lr = 0.01, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-8, .weight_decay = 0.1 };
        @memcpy(linear.weight.data, w0);
        var opt = try AdamWOptimizer.init(allocator, &linear, cfg);
        defer opt.deinit();
        @memcpy(ref_w, w0);
        @memset(ref_m, 0.0);
        @memset(ref_v, 0.0);
        for (1..4) |t| {
            opt.step();
            const tf: f32 = @floatFromInt(t);
            const bc1 = 1.0 - std.math.pow(f32, cfg.beta1, tf);
            const bc2 = 1.0 - std.math.pow(f32, cfg.beta2, tf);
            for (0..n) |i| {
                const g = linear.weight.grad[i];
                ref_m[i] = cfg.beta1 * ref_m[i] + (1.0 - cfg.beta1) * g;
                ref_v[i] = cfg.beta2 * ref_v[i] + (1.0 - cfg.beta2) * g * g;
                const step_val = (ref_m[i] / bc1) / (@sqrt(ref_v[i] / bc2) + cfg.eps);
                ref_w[i] -= cfg.lr * (step_val + cfg.weight_decay * ref_w[i]);
            }
        }
        for (0..n) |i| {
            try testing.expectApproxEqAbs(ref_w[i], linear.weight.data[i], 1e-5);
            try testing.expectApproxEqAbs(ref_m[i], opt.m[0][i], 1e-6);
            try testing.expectApproxEqAbs(ref_v[i], opt.v[0][i], 1e-6);
        }
    }

    // 2. Adam: w -= lr * sqrt(1 - beta2^t) / (1 - beta1^t) * m / (sqrt(v) + eps)
    {
        const cfg = AdamConfig{ .lr = 0.01, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8 };
        @memcpy(linear.weight.data, w0);
        var opt = try AdamOptimizer.init(allocator, &linear, cfg);
        defer opt.deinit();
        @memcpy(ref_w, w0);
        @memset(ref_m, 0.0);
        @memset(ref_v, 0.0);
        for (1..4) |t| {
            opt.step();
            const tf: f32 = @floatFromInt(t);
            const lr_t = cfg.lr * @sqrt(1.0 - std.math.pow(f32, cfg.beta2, tf)) / (1.0 - std.math.pow(f32, cfg.beta1, tf));
            for (0..n) |i| {
                const g = linear.weight.grad[i];
                ref_m[i] = cfg.beta1 * ref_m[i] + (1.0 - cfg.beta1) * g;
                ref_v[i] = cfg.beta2 * ref_v[i] + (1.0 - cfg.beta2) * g * g;
                ref_w[i] -= lr_t * ref_m[i] / (@sqrt(ref_v[i]) + cfg.eps);
            }
        }
        for (0..n) |i| {
            try testing.expectApproxEqAbs(ref_w[i], linear.weight.data[i], 1e-5);
        }
    }
}

test "AdamOptimizer step counter keeps increasing past 2^24 steps" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(8);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    var opt = try AdamOptimizer.init(allocator, &linear, .{});
    defer opt.deinit();
    // f32 步数在 2^24 处 t + 1 == t，偏差修正会停止推进；整数步数不受影响
    opt.step_count = 1 << 24;
    opt.step();
    opt.step();
    try testing.expectEqual(@as(u64, (1 << 24) + 2), opt.step_count);
}

test "Optimizer loadCheckpoint leaves state untouched on mismatch or truncation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const io = testing.io;

    var prng = std.Random.DefaultPrng.init(21);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    const path = "test_ckpt_untouched.bin";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    // 1. 参数个数不匹配：学习率与步数不应被检查点中的值覆盖
    {
        var src = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.5 });
        defer src.deinit();
        src.step();
        src.step();
        try src.saveCheckpoint(io, path);

        var seq = nn.sequential(.{
            try nn.Linear.init(allocator, 2, 2),
            try nn.Linear.init(allocator, 2, 2),
        });
        defer nn.deinitModel(&seq, allocator);
        for ([_]*Tensor{ seq.layers[0].weight, seq.layers[0].bias.?, seq.layers[1].weight, seq.layers[1].bias.? }) |t| {
            @memset(t.data, 0.1);
        }

        var dst = try SGDOptimizer.init(allocator, &seq, .{ .lr = 0.01 });
        defer dst.deinit();
        try testing.expectError(error.ParamCountMismatch, dst.loadCheckpoint(io, path));
        try testing.expectEqual(@as(f32, 0.01), dst.lr);
        try testing.expectEqual(@as(u64, 0), dst.step_count);
    }

    // 2. 截断的 AdamW 检查点：报告长度不匹配，且一阶/二阶矩与学习率保持原值
    {
        var src = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.002 });
        defer src.deinit();
        @memset(linear.weight.grad, 0.3);
        src.step();
        try src.saveCheckpoint(io, path);

        const full = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(full);
        {
            var file = try std.Io.Dir.cwd().createFile(io, path, .{});
            defer file.close(io);
            var buf: [256]u8 = undefined;
            var fw = file.writer(io, &buf);
            try fw.interface.writeAll(full[0 .. full.len - 4]);
            try fw.interface.flush();
        }

        var dst = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.001 });
        defer dst.deinit();
        try testing.expectError(error.CheckpointSizeMismatch, dst.loadCheckpoint(io, path));
        try testing.expectEqual(@as(f32, 0.001), dst.config.lr);
        try testing.expectEqual(@as(u64, 0), dst.step_count);
        try testing.expectEqual(@as(f32, 0.0), dst.m[0][0]);
    }

    // 3. 不含动量缓冲的 SGD 检查点：加载后已有动量缓冲被清零
    {
        var src = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1 });
        defer src.deinit();
        try src.saveCheckpoint(io, path);

        var dst = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1, .momentum = 0.9 });
        defer dst.deinit();
        @memset(linear.weight.grad, 1.0);
        dst.step();
        try testing.expect(dst.velocities.?[0][0] != 0.0);
        try dst.loadCheckpoint(io, path);
        for (dst.velocities.?) |v| {
            for (v) |x| try testing.expectEqual(@as(f32, 0.0), x);
        }
    }
}

/// 测试辅助：把字节内容整体写入文件
fn writeTestFile(io: std.Io, path: []const u8, bytes: []const u8) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [256]u8 = undefined;
    var fw = file.writer(io, &buf);
    try fw.interface.writeAll(bytes);
    try fw.interface.flush();
}

test "Optimizer checkpoint is fully little-endian and round-trips exactly" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const io = testing.io;

    var prng = std.Random.DefaultPrng.init(31);
    var linear = try nn.Linear.init(allocator, 3, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    const path = "test_ckpt_little_endian.bin";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var src = try AdamOptimizer.init(allocator, &linear, .{ .lr = 0.25 });
    defer src.deinit();
    for (src.m, src.v, 0..) |m_p, v_p, p| {
        for (m_p, v_p, 0..) |*m, *v, j| {
            m.* = @as(f32, @floatFromInt(p * 100 + j)) * 0.5 - 1.25;
            v.* = std.math.floatEps(f32) * @as(f32, @floatFromInt(j + 1));
        }
    }
    // 第一个参数的一阶矩首元素设为 1.5，其 IEEE 754 位模式为 0x3FC00000
    src.m[0][0] = 1.5;
    src.step_count = 0x0102030405060708;
    try src.saveCheckpoint(io, path);

    // 字节布局与主机字节序无关：版本、步数、学习率与状态缓冲都按小端序写出
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    try testing.expectEqualSlices(u8, &OPTIMIZER_MAGIC, bytes[0..4]);
    try testing.expectEqualSlices(u8, &.{ 2, 0, 0, 0 }, bytes[4..8]);
    try testing.expectEqual(OPTIMIZER_VERSION, std.mem.readInt(u32, bytes[4..8], .little));
    try testing.expectEqualSlices(u8, &.{ 8, 7, 6, 5, 4, 3, 2, 1 }, bytes[12..20]);
    try testing.expectEqualSlices(u8, &.{ 0x00, 0x00, 0x80, 0x3E }, bytes[20..24]); // 0.25
    const first_state = CHECKPOINT_HEADER_BYTES + 8;
    try testing.expectEqualSlices(u8, &.{ 0x00, 0x00, 0xC0, 0x3F }, bytes[first_state..][0..4]);

    // 往返恢复：所有状态按位一致
    var dst = try AdamOptimizer.init(allocator, &linear, .{});
    defer dst.deinit();
    try dst.loadCheckpoint(io, path);
    try testing.expectEqual(src.step_count, dst.step_count);
    try testing.expectEqual(src.lr, dst.lr);
    for (src.m, dst.m, src.v, dst.v) |sm, dm, sv, dv| {
        try testing.expectEqualSlices(f32, sm, dm);
        try testing.expectEqualSlices(f32, sv, dv);
    }
}

test "Optimizer loadCheckpoint is transactional on per-parameter length mismatch" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const io = testing.io;

    var prng = std.Random.DefaultPrng.init(37);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    const path = "test_ckpt_transactional.bin";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    // 1. AdamW：头部与文件总长度都正确，但第二个参数的长度字段被篡改。
    //    第一个参数的状态缓冲在文件中完好，也不得被写入优化器。
    {
        var src = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.002 });
        defer src.deinit();
        @memset(linear.weight.grad, 0.3);
        @memset(linear.bias.?.grad, -0.2);
        src.step();
        src.step();
        try src.saveCheckpoint(io, path);

        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(bytes);
        const second_len_offset = CHECKPOINT_HEADER_BYTES + 8 + 2 * src.params[0].data.len * @sizeOf(f32);
        const second_len = std.mem.readInt(u64, bytes[second_len_offset..][0..8], .little);
        try testing.expectEqual(@as(u64, src.params[1].data.len), second_len);
        std.mem.writeInt(u64, bytes[second_len_offset..][0..8], second_len + 1, .little);
        try writeTestFile(io, path, bytes);

        var dst = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.001 });
        defer dst.deinit();
        @memset(linear.weight.grad, 1.0);
        @memset(linear.bias.?.grad, 1.0);
        dst.step();
        const m_before = try allocator.dupe(f32, dst.m[0]);
        defer allocator.free(m_before);
        const v_before = try allocator.dupe(f32, dst.v[0]);
        defer allocator.free(v_before);
        const m_ptr = dst.m.ptr;

        try testing.expectError(error.BufferSizeMismatch, dst.loadCheckpoint(io, path));
        try testing.expectEqual(@as(f32, 0.001), dst.config.lr);
        try testing.expectEqual(@as(u64, 1), dst.step_count);
        try testing.expectEqual(m_ptr, dst.m.ptr);
        try testing.expectEqualSlices(f32, m_before, dst.m[0]);
        try testing.expectEqualSlices(f32, v_before, dst.v[0]);
    }

    // 2. SGD：无动量缓冲的优化器加载被篡改的带动量检查点，失败后仍不分配动量缓冲
    {
        var src = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1, .momentum = 0.9 });
        defer src.deinit();
        src.step();
        try src.saveCheckpoint(io, path);

        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(bytes);
        const second_len_offset = CHECKPOINT_HEADER_BYTES + 1 + 8 + src.params[0].data.len * @sizeOf(f32);
        std.mem.writeInt(u64, bytes[second_len_offset..][0..8], 0, .little);
        try writeTestFile(io, path, bytes);

        var dst = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.01 });
        defer dst.deinit();
        try testing.expectError(error.BufferSizeMismatch, dst.loadCheckpoint(io, path));
        try testing.expect(dst.velocities == null);
        try testing.expectEqual(@as(f32, 0.01), dst.lr);
        try testing.expectEqual(@as(u64, 0), dst.step_count);
    }

    // 3. SGD：动量标志只能是 0 或 1
    {
        var src = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1 });
        defer src.deinit();
        try src.saveCheckpoint(io, path);

        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(bytes);
        bytes[CHECKPOINT_HEADER_BYTES] = 2;
        try writeTestFile(io, path, bytes);

        var dst = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.01, .momentum = 0.9 });
        defer dst.deinit();
        @memset(linear.weight.grad, 1.0);
        dst.step();
        const vel_before = dst.velocities.?[0][0];
        try testing.expectError(error.InvalidCheckpointFlag, dst.loadCheckpoint(io, path));
        try testing.expectEqual(vel_before, dst.velocities.?[0][0]);
        try testing.expectEqual(@as(u64, 1), dst.step_count);
    }
}

test "clipGradNorm accumulates in higher precision and StepLR tolerates step_size 0" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 一个大分量 (平方 1e8) 加 1e6 个单位分量：单个 f32 累加器会把每个 +1 舍入掉 (1e8 处的单位最小精度为 8)
    const n: usize = 1_000_000 + 1;
    var t = try tensor.zeros(allocator, &.{n});
    defer tensor.free(allocator, t);
    t.requires_grad = true;
    t.grad = try allocator.alloc(f32, n);
    @memset(t.grad, 1.0);
    t.grad[0] = 1e4;

    var params = [_]*Tensor{t};
    const norm = clipGradNorm(&params, 1e9);
    try testing.expectApproxEqRel(@as(f32, @floatCast(@sqrt(1.01e8))), norm, 1e-5);

    const sched = StepLRScheduler{ .base_lr = 1.0, .step_size = 0, .gamma = 0.5 };
    try testing.expectApproxEqAbs(@as(f32, 0.25), sched.getLR(2), 1e-6);
}
