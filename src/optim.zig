const std = @import("std");
const nn = @import("nn.zig");
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

pub const OPTIMIZER_MAGIC = [4]u8{ 'Z', 'N', 'N', 'O' };
pub const OPTIMIZER_VERSION: u32 = 1;

pub const OptimizerTypeTag = enum(u32) {
    sgd = 0,
    adam = 1,
    adamw = 2,
};

pub const CheckpointError = error{
    InvalidCheckpointMagic,
    UnsupportedCheckpointVersion,
    OptimizerTypeMismatch,
    ParamCountMismatch,
    BufferSizeMismatch,
};

fn writeU32(writer: anytype, val: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, val, .little);
    try writer.writeAll(&b);
}

fn writeU64(writer: anytype, val: u64) !void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, val, .little);
    try writer.writeAll(&b);
}

fn writeF32(writer: anytype, val: f32) !void {
    const bits: u32 = @bitCast(val);
    try writeU32(writer, bits);
}

fn readU32(reader: anytype) !u32 {
    var b: [4]u8 = undefined;
    try reader.readSliceAll(&b);
    return std.mem.readInt(u32, &b, .little);
}

fn readU64(reader: anytype) !u64 {
    var b: [8]u8 = undefined;
    try reader.readSliceAll(&b);
    return std.mem.readInt(u64, &b, .little);
}

fn readF32(reader: anytype) !f32 {
    const bits = try readU32(reader);
    return @bitCast(bits);
}

pub const SGDOptimizer = struct {
    allocator: std.mem.Allocator,
    params: []*Tensor,
    velocities: ?[][]f32, // Only allocated if momentum > 0
    lr: f32,
    momentum: f32,
    step_count: u64 = 0,

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
            .step_count = 0,
        };
    }

    pub fn deinit(self: SGDOptimizer) void {
        if (self.velocities) |v_list| {
            for (v_list) |v| self.allocator.free(v);
            self.allocator.free(v_list);
        }
        self.allocator.free(self.params);
    }

    pub fn getLR(self: SGDOptimizer) f32 {
        return self.lr;
    }

    pub fn setLR(self: *SGDOptimizer, lr: f32) void {
        self.lr = lr;
    }

    pub fn stepWithLR(self: *SGDOptimizer, current_lr: f32) void {
        self.step_count += 1;
        for (self.params, 0..) |param, i| {
            const w = param.data;
            const dw = param.grad;
            if (self.velocities) |v_list| {
                const v = v_list[i];
                for (w, dw, v) |*weight, grad, *vel| {
                    vel.* = self.momentum * vel.* + current_lr * grad;
                    weight.* -= vel.*;
                }
            } else {
                for (w, dw) |*weight, grad| {
                    weight.* -= current_lr * grad;
                }
            }
        }
    }

    pub fn step(self: *SGDOptimizer) void {
        self.stepWithLR(self.lr);
    }

    pub fn saveCheckpoint(self: SGDOptimizer, io: std.Io, file_path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.createFile(io, file_path, .{});
        defer file.close(io);

        var buf: [65536]u8 = undefined;
        var file_writer = file.writer(io, &buf);
        const writer = &file_writer.interface;

        try writer.writeAll(&OPTIMIZER_MAGIC);
        try writeU32(writer, OPTIMIZER_VERSION);
        try writeU32(writer, @intFromEnum(OptimizerTypeTag.sgd));
        try writeU64(writer, self.step_count);
        try writeF32(writer, self.lr);
        try writeU64(writer, @as(u64, self.params.len));

        const has_vel: u8 = if (self.velocities != null) 1 else 0;
        try writer.writeAll(&[_]u8{has_vel});

        for (self.params, 0..) |param, i| {
            try writeU64(writer, @as(u64, param.data.len));
            if (self.velocities) |v_list| {
                try writer.writeAll(std.mem.sliceAsBytes(v_list[i]));
            }
        }
        try writer.flush();
    }

    pub fn loadCheckpoint(self: *SGDOptimizer, io: std.Io, file_path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.openFile(io, file_path, .{});
        defer file.close(io);

        var buf: [65536]u8 = undefined;
        var file_reader = file.reader(io, &buf);
        const reader = &file_reader.interface;

        var magic: [4]u8 = undefined;
        try reader.readSliceAll(&magic);
        if (!std.mem.eql(u8, &magic, &OPTIMIZER_MAGIC)) return error.InvalidCheckpointMagic;

        const version = try readU32(reader);
        if (version != OPTIMIZER_VERSION) return error.UnsupportedCheckpointVersion;

        const type_tag = try readU32(reader);
        if (type_tag != @intFromEnum(OptimizerTypeTag.sgd)) return error.OptimizerTypeMismatch;

        self.step_count = try readU64(reader);
        self.lr = try readF32(reader);
        const param_count = try readU64(reader);
        if (param_count != self.params.len) return error.ParamCountMismatch;

        var has_vel_byte: [1]u8 = undefined;
        try reader.readSliceAll(&has_vel_byte);
        const has_vel = has_vel_byte[0] == 1;

        if (has_vel and self.velocities == null) {
            const v_list = try self.allocator.alloc([]f32, self.params.len);
            errdefer self.allocator.free(v_list);
            var init_count: usize = 0;
            errdefer {
                for (0..init_count) |j| self.allocator.free(v_list[j]);
            }
            for (self.params) |param| {
                v_list[init_count] = try self.allocator.alloc(f32, param.data.len);
                init_count += 1;
            }
            self.velocities = v_list;
        }

        for (self.params, 0..) |param, i| {
            const param_len = try readU64(reader);
            if (param_len != param.data.len) return error.BufferSizeMismatch;
            if (has_vel) {
                try reader.readSliceAll(std.mem.sliceAsBytes(self.velocities.?[i]));
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

    pub fn getLR(self: AdamOptimizer) f32 {
        return self.lr;
    }

    pub fn setLR(self: *AdamOptimizer, lr: f32) void {
        self.lr = lr;
    }

    pub fn stepWithLR(self: *AdamOptimizer, current_lr: f32) void {
        self.t += 1.0;
        const correction1 = 1.0 - std.math.pow(f32, self.beta1, self.t);
        const correction2 = 1.0 - std.math.pow(f32, self.beta2, self.t);
        const lr_t = current_lr * @sqrt(correction2) / correction1;

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

    pub fn step(self: *AdamOptimizer) void {
        self.stepWithLR(self.lr);
    }

    pub fn saveCheckpoint(self: AdamOptimizer, io: std.Io, file_path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.createFile(io, file_path, .{});
        defer file.close(io);

        var buf: [65536]u8 = undefined;
        var file_writer = file.writer(io, &buf);
        const writer = &file_writer.interface;

        try writer.writeAll(&OPTIMIZER_MAGIC);
        try writeU32(writer, OPTIMIZER_VERSION);
        try writeU32(writer, @intFromEnum(OptimizerTypeTag.adam));
        try writeU64(writer, @as(u64, @intFromFloat(self.t)));
        try writeF32(writer, self.lr);
        try writeU64(writer, @as(u64, self.params.len));

        for (self.params, 0..) |param, i| {
            try writeU64(writer, @as(u64, param.data.len));
            try writer.writeAll(std.mem.sliceAsBytes(self.m[i]));
            try writer.writeAll(std.mem.sliceAsBytes(self.v[i]));
        }
        try writer.flush();
    }

    pub fn loadCheckpoint(self: *AdamOptimizer, io: std.Io, file_path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.openFile(io, file_path, .{});
        defer file.close(io);

        var buf: [65536]u8 = undefined;
        var file_reader = file.reader(io, &buf);
        const reader = &file_reader.interface;

        var magic: [4]u8 = undefined;
        try reader.readSliceAll(&magic);
        if (!std.mem.eql(u8, &magic, &OPTIMIZER_MAGIC)) return error.InvalidCheckpointMagic;

        const version = try readU32(reader);
        if (version != OPTIMIZER_VERSION) return error.UnsupportedCheckpointVersion;

        const type_tag = try readU32(reader);
        if (type_tag != @intFromEnum(OptimizerTypeTag.adam)) return error.OptimizerTypeMismatch;

        const step_val = try readU64(reader);
        self.t = @as(f32, @floatFromInt(step_val));
        self.lr = try readF32(reader);
        const param_count = try readU64(reader);
        if (param_count != self.params.len) return error.ParamCountMismatch;

        for (self.params, 0..) |param, i| {
            const param_len = try readU64(reader);
            if (param_len != param.data.len) return error.BufferSizeMismatch;
            try reader.readSliceAll(std.mem.sliceAsBytes(self.m[i]));
            try reader.readSliceAll(std.mem.sliceAsBytes(self.v[i]));
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

    pub fn getLR(self: AdamWOptimizer) f32 {
        return self.config.lr;
    }

    pub fn setLR(self: *AdamWOptimizer, lr: f32) void {
        self.config.lr = lr;
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

    pub fn saveCheckpoint(self: AdamWOptimizer, io: std.Io, file_path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.createFile(io, file_path, .{});
        defer file.close(io);

        var buf: [65536]u8 = undefined;
        var file_writer = file.writer(io, &buf);
        const writer = &file_writer.interface;

        try writer.writeAll(&OPTIMIZER_MAGIC);
        try writeU32(writer, OPTIMIZER_VERSION);
        try writeU32(writer, @intFromEnum(OptimizerTypeTag.adamw));
        try writeU64(writer, self.step_count);
        try writeF32(writer, self.config.lr);
        try writeU64(writer, @as(u64, self.params.len));

        for (self.params, 0..) |param, i| {
            try writeU64(writer, @as(u64, param.data.len));
            try writer.writeAll(std.mem.sliceAsBytes(self.m[i]));
            try writer.writeAll(std.mem.sliceAsBytes(self.v[i]));
        }
        try writer.flush();
    }

    pub fn loadCheckpoint(self: *AdamWOptimizer, io: std.Io, file_path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.openFile(io, file_path, .{});
        defer file.close(io);

        var buf: [65536]u8 = undefined;
        var file_reader = file.reader(io, &buf);
        const reader = &file_reader.interface;

        var magic: [4]u8 = undefined;
        try reader.readSliceAll(&magic);
        if (!std.mem.eql(u8, &magic, &OPTIMIZER_MAGIC)) return error.InvalidCheckpointMagic;

        const version = try readU32(reader);
        if (version != OPTIMIZER_VERSION) return error.UnsupportedCheckpointVersion;

        const type_tag = try readU32(reader);
        if (type_tag != @intFromEnum(OptimizerTypeTag.adamw)) return error.OptimizerTypeMismatch;

        self.step_count = try readU64(reader);
        self.config.lr = try readF32(reader);
        const param_count = try readU64(reader);
        if (param_count != self.params.len) return error.ParamCountMismatch;

        for (self.params, 0..) |param, i| {
            const param_len = try readU64(reader);
            if (param_len != param.data.len) return error.BufferSizeMismatch;
            try reader.readSliceAll(std.mem.sliceAsBytes(self.m[i]));
            try reader.readSliceAll(std.mem.sliceAsBytes(self.v[i]));
        }
    }
};

// =========================================================================
// 学习率调度器 (Learning Rate Schedulers)
// =========================================================================

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

/// 固定步长阶梯衰减学习率调度器 (StepLR)
pub const StepLRScheduler = struct {
    base_lr: f32,
    step_size: u64,
    gamma: f32,

    pub fn init(base_lr: f32, step_size: u64, gamma: f32) StepLRScheduler {
        std.debug.assert(step_size > 0);
        return .{
            .base_lr = base_lr,
            .step_size = step_size,
            .gamma = gamma,
        };
    }

    pub fn getLR(self: StepLRScheduler, current_step: u64) f32 {
        const factor = std.math.pow(f32, self.gamma, @as(f32, @floatFromInt(current_step / self.step_size)));
        return self.base_lr * factor;
    }
};

/// 纯线性预热学习率调度器 (Linear Warmup)
pub const LinearWarmupScheduler = struct {
    start_lr: f32,
    target_lr: f32,
    warmup_steps: u64,

    pub fn init(start_lr: f32, target_lr: f32, warmup_steps: u64) LinearWarmupScheduler {
        std.debug.assert(warmup_steps > 0);
        return .{
            .start_lr = start_lr,
            .target_lr = target_lr,
            .warmup_steps = warmup_steps,
        };
    }

    pub fn getLR(self: LinearWarmupScheduler, current_step: u64) f32 {
        if (current_step >= self.warmup_steps) {
            return self.target_lr;
        }
        const progress = @as(f32, @floatFromInt(current_step)) / @as(f32, @floatFromInt(self.warmup_steps));
        return self.start_lr + (self.target_lr - self.start_lr) * progress;
    }
};

/// 指数衰减学习率调度器 (ExponentialLR)
pub const ExponentialLRScheduler = struct {
    base_lr: f32,
    gamma: f32,

    pub fn init(base_lr: f32, gamma: f32) ExponentialLRScheduler {
        return .{
            .base_lr = base_lr,
            .gamma = gamma,
        };
    }

    pub fn getLR(self: ExponentialLRScheduler, current_step: u64) f32 {
        return self.base_lr * std.math.pow(f32, self.gamma, @as(f32, @floatFromInt(current_step)));
    }
};

/// 统一的多态学习率调度器包装
pub const LRScheduler = union(enum) {
    cosine: CosineScheduler,
    step_lr: StepLRScheduler,
    warmup: LinearWarmupScheduler,
    exponential: ExponentialLRScheduler,

    pub fn getLR(self: LRScheduler, current_step: u64) f32 {
        return switch (self) {
            .cosine => |s| s.getLR(current_step),
            .step_lr => |s| s.getLR(current_step),
            .warmup => |s| s.getLR(current_step),
            .exponential => |s| s.getLR(current_step),
        };
    }

    pub fn step(self: LRScheduler, optimizer: anytype, current_step: u64) f32 {
        const lr = self.getLR(current_step);
        optimizer.setLR(lr);
        return lr;
    }
};

// =========================================================================
// 梯度裁剪 (Gradient Clipping)
// =========================================================================

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

/// 梯度分量数值截断 (Gradient Value Clipping)
pub fn clipGradValue(params: []*Tensor, clip_value: f32) void {
    const abs_clip = @abs(clip_value);
    for (params) |param| {
        for (param.grad) |*g| {
            g.* = std.math.clamp(g.*, -abs_clip, abs_clip);
        }
    }
}

/// 梯度裁剪配置类型
pub const GradClipConfig = union(enum) {
    norm: f32,
    value: f32,
    none: void,
};

/// 统一梯度裁剪执行函数
pub fn clipGradients(params: []*Tensor, config: GradClipConfig) ?f32 {
    return switch (config) {
        .norm => |max_norm| clipGradNorm(params, max_norm),
        .value => |clip_val| {
            clipGradValue(params, clip_val);
            return null;
        },
        .none => null,
    };
}

// =========================================================================
// 单元测试
// =========================================================================

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
    var linear = try nn.Linear.init(allocator, 3, 2, prng.random());
    defer linear.deinit(allocator);

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
}


