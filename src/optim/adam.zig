const std = @import("std");
const nn = @import("../nn.zig");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const checkpoint = @import("checkpoint.zig");
const CheckpointHeader = checkpoint.CheckpointHeader;
const writeCheckpoint = checkpoint.writeCheckpoint;
const readCheckpoint = checkpoint.readCheckpoint;
const replaceStates = checkpoint.replaceStates;
const allocZeroedStates = checkpoint.allocZeroedStates;
const freeStates = checkpoint.freeStates;

/// Adam 系列优化器的逐元素更新系数：
///   m = beta1 * m + (1 - beta1) * g
///   v = beta2 * v + (1 - beta2) * g^2
///   w = w - decay * w - step_size * m / (sqrt(v) * denom_scale + eps)
/// 偏差修正 (Bias Correction) 的除法均在每步开始时折算进 `step_size` 与 `denom_scale`，
/// 内层循环不再逐元素做两次除法。
pub const AdamUpdate = struct {
    beta1: f32,
    one_minus_beta1: f32,
    beta2: f32,
    one_minus_beta2: f32,
    step_size: f32,
    denom_scale: f32,
    eps: f32,
    decay: f32,

    const vec_len = std.simd.suggestVectorLength(f32) orelse 4;
    const Vec = @Vector(vec_len, f32);

    pub fn apply(self: AdamUpdate, w: []f32, g: []const f32, m: []f32, v: []f32) void {
        std.debug.assert(g.len == w.len and m.len == w.len and v.len == w.len);
        const b1: Vec = @splat(self.beta1);
        const omb1: Vec = @splat(self.one_minus_beta1);
        const b2: Vec = @splat(self.beta2);
        const omb2: Vec = @splat(self.one_minus_beta2);
        const step_size: Vec = @splat(self.step_size);
        const denom_scale: Vec = @splat(self.denom_scale);
        const eps: Vec = @splat(self.eps);
        const keep: Vec = @splat(1.0 - self.decay);

        var i: usize = 0;
        while (i + vec_len <= w.len) : (i += vec_len) {
            const gv: Vec = g[i..][0..vec_len].*;
            const mv = b1 * @as(Vec, m[i..][0..vec_len].*) + omb1 * gv;
            const vv = b2 * @as(Vec, v[i..][0..vec_len].*) + omb2 * (gv * gv);
            const wv: Vec = w[i..][0..vec_len].*;
            m[i..][0..vec_len].* = mv;
            v[i..][0..vec_len].* = vv;
            w[i..][0..vec_len].* = keep * wv - step_size * mv / (@sqrt(vv) * denom_scale + eps);
        }
        while (i < w.len) : (i += 1) {
            const gi = g[i];
            const mi = self.beta1 * m[i] + self.one_minus_beta1 * gi;
            const vi = self.beta2 * v[i] + self.one_minus_beta2 * (gi * gi);
            m[i] = mi;
            v[i] = vi;
            w[i] = (1.0 - self.decay) * w[i] - self.step_size * mi / (@sqrt(vi) * self.denom_scale + self.eps);
        }
    }
};

/// 自适应矩估计 (Adaptive Moment Estimation, Adam) 优化器配置
pub const AdamConfig = struct {
    lr: f32 = 0.001,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,

    pub const default: AdamConfig = .{};
    pub fn defaultConfig() AdamConfig {
        return .{};
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
    /// 已执行的更新步数；整数计数避免 f32 步数在 2^24 步后停止增长
    step_count: u64 = 0,

    pub fn initDefault(allocator: std.mem.Allocator, model: anytype) !AdamOptimizer {
        return init(allocator, model, AdamConfig.default);
    }

    pub fn init(allocator: std.mem.Allocator, model: anytype, config: AdamConfig) !AdamOptimizer {
        const params = try nn.parameters(model, allocator);
        nn.warnIfParametersUninitialized(params);
        errdefer allocator.free(params);

        const m = try allocZeroedStates(allocator, params);
        errdefer freeStates(allocator, m);
        const v = try allocZeroedStates(allocator, params);

        return AdamOptimizer{
            .allocator = allocator,
            .params = params,
            .m = m,
            .v = v,
            .lr = config.lr,
            .beta1 = config.beta1,
            .beta2 = config.beta2,
            .eps = config.eps,
            .step_count = 0,
        };
    }

    pub fn deinit(self: AdamOptimizer) void {
        freeStates(self.allocator, self.m);
        freeStates(self.allocator, self.v);
        self.allocator.free(self.params);
    }

    pub fn getLR(self: AdamOptimizer) f32 {
        return self.lr;
    }

    pub fn setLR(self: *AdamOptimizer, lr: f32) void {
        self.lr = lr;
    }

    pub fn stepWithLR(self: *AdamOptimizer, current_lr: f32) void {
        self.step_count += 1;
        const t: f32 = @floatFromInt(self.step_count);

        const bias_correction1 = 1.0 - std.math.pow(f32, self.beta1, t);
        const bias_correction2 = 1.0 - std.math.pow(f32, self.beta2, t);

        // w = w - lr * m_hat / (sqrt(v_hat) + eps)，其中
        // m_hat = m / bias_correction1，v_hat = v / bias_correction2
        const update = AdamUpdate{
            .beta1 = self.beta1,
            .one_minus_beta1 = 1.0 - self.beta1,
            .beta2 = self.beta2,
            .one_minus_beta2 = 1.0 - self.beta2,
            .step_size = current_lr / bias_correction1,
            .denom_scale = 1.0 / @sqrt(bias_correction2),
            .eps = self.eps,
            .decay = 0.0,
        };

        for (self.params, self.m, self.v) |param, m_t, v_t| {
            update.apply(param.data, param.grad, m_t, v_t);
            param.clearDeviceArray();
        }
    }

    pub fn step(self: *AdamOptimizer) void {
        self.stepWithLR(self.lr);
    }

    pub fn saveCheckpoint(self: AdamOptimizer, io: std.Io, file_path: []const u8) !void {
        const header: CheckpointHeader = .{ .step = self.step_count, .lr = self.lr };
        try writeCheckpoint(io, file_path, .adam, header, self.params, &.{ self.m, self.v });
    }

    /// 事务式加载：任何错误都让优化器保持加载前的状态
    pub fn loadCheckpoint(self: *AdamOptimizer, io: std.Io, file_path: []const u8) !void {
        const loaded = try readCheckpoint(self.allocator, io, file_path, .adam, self.params);

        // 提交阶段：以下操作均不会失败
        replaceStates(self.allocator, &self.m, loaded.states[0]);
        replaceStates(self.allocator, &self.v, loaded.states[1]);
        self.step_count = loaded.header.step;
        self.lr = loaded.header.lr;
    }
};

pub const AdamWConfig = struct {
    lr: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.95,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.01,

    pub const default: AdamWConfig = .{};
    pub fn defaultConfig() AdamWConfig {
        return .{};
    }
};

/// 具备解耦权重衰减的自适应矩估计优化器 (Adam with Decoupled Weight Decay, AdamW)
pub const AdamWOptimizer = struct {
    allocator: std.mem.Allocator,
    params: []*Tensor,
    m: [][]f32,
    v: [][]f32,
    config: AdamWConfig,
    step_count: u64,

    pub fn initDefault(allocator: std.mem.Allocator, model: anytype) !AdamWOptimizer {
        return init(allocator, model, AdamWConfig.default);
    }

    pub fn init(allocator: std.mem.Allocator, model: anytype, config: AdamWConfig) !AdamWOptimizer {
        const params = try nn.parameters(model, allocator);
        nn.warnIfParametersUninitialized(params);
        errdefer allocator.free(params);

        const m = try allocZeroedStates(allocator, params);
        errdefer freeStates(allocator, m);
        const v = try allocZeroedStates(allocator, params);

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
        freeStates(self.allocator, self.m);
        freeStates(self.allocator, self.v);
        self.allocator.free(self.params);
    }

    pub fn getLR(self: AdamWOptimizer) f32 {
        return self.config.lr;
    }

    pub fn setLR(self: *AdamWOptimizer, lr: f32) void {
        self.config.lr = lr;
    }

    pub fn stepWithLR(self: *AdamWOptimizer, current_lr: f32) void {
        self.step_count += 1;
        const t: f32 = @floatFromInt(self.step_count);
        const beta1 = self.config.beta1;
        const beta2 = self.config.beta2;

        const bias_correction1 = 1.0 - std.math.pow(f32, beta1, t);
        const bias_correction2 = 1.0 - std.math.pow(f32, beta2, t);

        // w = w - lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)，其中
        // m_hat = m / bias_correction1，v_hat = v / bias_correction2
        const update = AdamUpdate{
            .beta1 = beta1,
            .one_minus_beta1 = 1.0 - beta1,
            .beta2 = beta2,
            .one_minus_beta2 = 1.0 - beta2,
            .step_size = current_lr / bias_correction1,
            .denom_scale = 1.0 / @sqrt(bias_correction2),
            .eps = self.config.eps,
            .decay = current_lr * self.config.weight_decay,
        };

        for (self.params, self.m, self.v) |param, m_t, v_t| {
            update.apply(param.data, param.grad, m_t, v_t);
            param.clearDeviceArray();
        }
    }

    pub fn step(self: *AdamWOptimizer) void {
        self.stepWithLR(self.config.lr);
    }

    pub fn saveCheckpoint(self: AdamWOptimizer, io: std.Io, file_path: []const u8) !void {
        const header: CheckpointHeader = .{ .step = self.step_count, .lr = self.config.lr };
        try writeCheckpoint(io, file_path, .adamw, header, self.params, &.{ self.m, self.v });
    }

    /// 事务式加载：任何错误都让优化器保持加载前的状态
    pub fn loadCheckpoint(self: *AdamWOptimizer, io: std.Io, file_path: []const u8) !void {
        const loaded = try readCheckpoint(self.allocator, io, file_path, .adamw, self.params);

        // 提交阶段：以下操作均不会失败
        replaceStates(self.allocator, &self.m, loaded.states[0]);
        replaceStates(self.allocator, &self.v, loaded.states[1]);
        self.step_count = loaded.header.step;
        self.config.lr = loaded.header.lr;
    }
};
