const std = @import("std");
const nn = @import("nn.zig");
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

pub const SGDOptimizer = struct {
    allocator: std.mem.Allocator,
    params: []*Tensor,
    velocities: ?[][]f32, // Only allocated if momentum > 0
    lr: f32,
    momentum: f32,

    pub fn init(allocator: std.mem.Allocator, model: anytype, config: struct { lr: f32, momentum: f32 = 0.0 }) !SGDOptimizer {
        const params = try nn.collectParameters(model, allocator);
        errdefer allocator.free(params);

        var velocities: ?[][]f32 = null;
        if (config.momentum > 0.0) {
            const v_list = try allocator.alloc([]f32, params.len);
            errdefer allocator.free(v_list);
            var initialized: usize = 0;
            errdefer {
                for (0..initialized) |j| allocator.free(v_list[j]);
            }
            for (params) |param| {
                v_list[initialized] = try allocator.alloc(f32, param.data.len);
                @memset(v_list[initialized], 0.0);
                initialized += 1;
            }
            velocities = v_list;
        }

        return SGDOptimizer{
            .allocator = allocator,
            .params = params,
            .velocities = velocities,
            .lr = config.lr,
            .momentum = config.momentum,
        };
    }

    pub fn deinit(self: SGDOptimizer) void {
        if (self.velocities) |v_list| {
            for (v_list) |v| self.allocator.free(v);
            self.allocator.free(v_list);
        }
        self.allocator.free(self.params);
    }

    pub fn step(self: SGDOptimizer) void {
        for (self.params, 0..) |param, i| {
            const w = param.data;
            const dw = param.grad;
            if (self.velocities) |v_list| {
                const v = v_list[i];
                for (w, dw, v) |*weight, grad, *vel| {
                    vel.* = self.momentum * vel.* + self.lr * grad;
                    weight.* -= vel.*;
                }
            } else {
                for (w, dw) |*weight, grad| {
                    weight.* -= self.lr * grad;
                }
            }
        }
    }
};

pub const AdamOptimizer = struct {
    allocator: std.mem.Allocator,
    params: []*Tensor,
    m: [][]f32,
    v: [][]f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: f32, // Timestep

    pub fn init(allocator: std.mem.Allocator, model: anytype, config: struct {
        lr: f32 = 0.001,
        beta1: f32 = 0.9,
        beta2: f32 = 0.999,
        eps: f32 = 1e-8,
    }) !AdamOptimizer {
        const params = try nn.collectParameters(model, allocator);
        errdefer allocator.free(params);

        const m = try allocator.alloc([]f32, params.len);
        errdefer allocator.free(m);
        const v = try allocator.alloc([]f32, params.len);
        errdefer allocator.free(v);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |i| {
                allocator.free(m[i]);
                allocator.free(v[i]);
            }
        }

        for (params) |param| {
            m[initialized] = try allocator.alloc(f32, param.data.len);
            v[initialized] = try allocator.alloc(f32, param.data.len);
            @memset(m[initialized], 0.0);
            @memset(v[initialized], 0.0);
            initialized += 1;
        }

        return AdamOptimizer{
            .allocator = allocator,
            .params = params,
            .m = m,
            .v = v,
            .lr = config.lr,
            .beta1 = config.beta1,
            .beta2 = config.beta2,
            .eps = config.eps,
            .t = 0.0,
        };
    }

    pub fn deinit(self: AdamOptimizer) void {
        for (0..self.params.len) |i| {
            self.allocator.free(self.m[i]);
            self.allocator.free(self.v[i]);
        }
        self.allocator.free(self.m);
        self.allocator.free(self.v);
        self.allocator.free(self.params);
    }

    pub fn step(self: *AdamOptimizer) void {
        self.t += 1.0;
        const correction1 = 1.0 - std.math.pow(f32, self.beta1, self.t);
        const correction2 = 1.0 - std.math.pow(f32, self.beta2, self.t);
        const lr_t = self.lr * @sqrt(correction2) / correction1;

        for (self.params, 0..) |param, i| {
            const w = param.data;
            const dw = param.grad;
            const m_t = self.m[i];
            const v_t = self.v[i];

            for (w, dw, m_t, v_t) |*weight, grad, *m_i, *v_i| {
                m_i.* = self.beta1 * m_i.* + (1.0 - self.beta1) * grad;
                v_i.* = self.beta2 * v_i.* + (1.0 - self.beta2) * grad * grad;
                weight.* -= lr_t * m_i.* / (@sqrt(v_i.*) + self.eps);
            }
        }
    }
};

pub const AdamWConfig = struct {
    lr: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.95,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.01,
};

/// 具备解耦权重衰减 (Decoupled Weight Decay) 的 AdamW 优化器
pub const AdamWOptimizer = struct {
    allocator: std.mem.Allocator,
    params: []*Tensor,
    m: [][]f32,
    v: [][]f32,
    config: AdamWConfig,
    step_count: u64,

    pub fn init(allocator: std.mem.Allocator, model: anytype, config: AdamWConfig) !AdamWOptimizer {
        const params = try nn.collectParameters(model, allocator);
        errdefer allocator.free(params);

        const m = try allocator.alloc([]f32, params.len);
        errdefer allocator.free(m);
        const v = try allocator.alloc([]f32, params.len);
        errdefer allocator.free(v);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |i| {
                allocator.free(m[i]);
                allocator.free(v[i]);
            }
        }

        for (params) |param| {
            m[initialized] = try allocator.alloc(f32, param.data.len);
            v[initialized] = try allocator.alloc(f32, param.data.len);
            @memset(m[initialized], 0.0);
            @memset(v[initialized], 0.0);
            initialized += 1;
        }

        return AdamWOptimizer{
            .allocator = allocator,
            .params = params,
            .m = m,
            .v = v,
            .config = config,
            .step_count = 0,
        };
    }

    pub fn deinit(self: AdamWOptimizer) void {
        for (0..self.params.len) |i| {
            self.allocator.free(self.m[i]);
            self.allocator.free(self.v[i]);
        }
        self.allocator.free(self.m);
        self.allocator.free(self.v);
        self.allocator.free(self.params);
    }

    pub fn stepWithLR(self: *AdamWOptimizer, current_lr: f32) void {
        self.step_count += 1;
        const t_f32 = @as(f32, @floatFromInt(self.step_count));
        const beta1 = self.config.beta1;
        const beta2 = self.config.beta2;
        const eps = self.config.eps;
        const wd = self.config.weight_decay;

        const bias_correction1 = 1.0 - std.math.pow(f32, beta1, t_f32);
        const bias_correction2 = 1.0 - std.math.pow(f32, beta2, t_f32);

        for (self.params, 0..) |param, i| {
            const w = param.data;
            const dw = param.grad;
            const m_t = self.m[i];
            const v_t = self.v[i];

            for (w, dw, m_t, v_t) |*weight, grad, *m_val, *v_val| {
                // 1. 更新一阶和二阶动量
                m_val.* = beta1 * m_val.* + (1.0 - beta1) * grad;
                v_val.* = beta2 * v_val.* + (1.0 - beta2) * (grad * grad);

                // 2. 无偏估计
                const m_hat = m_val.* / bias_correction1;
                const v_hat = v_val.* / bias_correction2;

                // 3. 自适应动量项
                const step_val = m_hat / (@sqrt(v_hat) + eps);

                // 4. 解耦权重衰减更新: w = w - lr * (step_val + wd * w)
                weight.* -= current_lr * (step_val + wd * weight.*);
            }
        }
    }

    pub fn step(self: *AdamWOptimizer) void {
        self.stepWithLR(self.config.lr);
    }
};

/// 带有线性预热 (Linear Warmup) 的余弦退火学习率调度器
pub const CosineScheduler = struct {
    max_lr: f32,
    min_lr: f32,
    warmup_steps: u64,
    max_steps: u64,

    pub fn init(max_lr: f32, min_lr: f32, warmup_steps: u64, max_steps: u64) CosineScheduler {
        return .{
            .max_lr = max_lr,
            .min_lr = min_lr,
            .warmup_steps = warmup_steps,
            .max_steps = max_steps,
        };
    }

    pub fn getLR(self: CosineScheduler, current_step: u64) f32 {
        if (self.warmup_steps > 0 and current_step < self.warmup_steps) {
            const progress = @as(f32, @floatFromInt(current_step)) / @as(f32, @floatFromInt(self.warmup_steps));
            return self.max_lr * progress;
        }
        if (current_step >= self.max_steps) {
            return self.min_lr;
        }

        const current_f = @as(f32, @floatFromInt(current_step - self.warmup_steps));
        const total_f = @as(f32, @floatFromInt(self.max_steps - self.warmup_steps));
        const progress = current_f / total_f;
        const cosine_decay = 0.5 * (1.0 + @cos(progress * std.math.pi));

        return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay;
    }
};

/// 全局梯度 L2 范数裁剪 (Gradient Norm Clipping)
pub fn clipGradNorm(params: []*Tensor, max_norm: f32) f32 {
    var total_norm_sq: f32 = 0.0;
    for (params) |param| {
        for (param.grad) |g| {
            total_norm_sq += g * g;
        }
    }
    const total_norm = @sqrt(total_norm_sq);
    if (total_norm > max_norm and total_norm > 0.0) {
        const scale = max_norm / (total_norm + 1e-6);
        for (params) |param| {
            for (param.grad) |*g| {
                g.* *= scale;
            }
        }
    }
    return total_norm;
}

test "SGDOptimizer basic and momentum updates" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(42);
    var linear = try nn.Linear.init(allocator, 2, 2, prng.random());
    defer linear.deinit(allocator);

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
    var linear = try nn.Linear.init(allocator, 2, 2, prng.random());
    defer linear.deinit(allocator);

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
    var linear = try nn.Linear.init(allocator, 2, 2, prng.random());
    defer linear.deinit(allocator);

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

test "clipGradNorm gradient scaling" {
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
}

