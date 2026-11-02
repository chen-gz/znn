const std = @import("std");

pub const CosineScheduler = struct {
    max_lr: f32 = 1e-3,
    min_lr: f32 = 1e-5,
    warmup_steps: u64 = 100,
    max_steps: u64 = 1000,

    pub const default: CosineScheduler = .{};
    pub fn defaultOptions() CosineScheduler {
        return .{};
    }

    pub fn initDefault() CosineScheduler {
        return .{};
    }

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

/// 固定步长阶梯衰减学习率调度器 (Step Learning Rate, StepLR)
pub const StepLRScheduler = struct {
    base_lr: f32 = 1e-3,
    step_size: u64 = 100,
    gamma: f32 = 0.1,

    pub const default: StepLRScheduler = .{};
    pub fn defaultOptions() StepLRScheduler {
        return .{};
    }

    pub fn initDefault() StepLRScheduler {
        return .{};
    }

    pub fn init(base_lr: f32, step_size: u64, gamma: f32) StepLRScheduler {
        std.debug.assert(step_size > 0);
        return .{
            .base_lr = base_lr,
            .step_size = step_size,
            .gamma = gamma,
        };
    }

    pub fn getLR(self: StepLRScheduler, current_step: u64) f32 {
        // step_size 为 0 时视为每步衰减一次，避免整数除零
        const decays = current_step / @max(self.step_size, 1);
        const factor = std.math.pow(f32, self.gamma, @as(f32, @floatFromInt(decays)));
        return self.base_lr * factor;
    }
};

/// 纯线性预热学习率调度器 (Linear Warmup)
pub const LinearWarmupScheduler = struct {
    start_lr: f32 = 0.0,
    target_lr: f32 = 1e-3,
    warmup_steps: u64 = 100,

    pub const default: LinearWarmupScheduler = .{};
    pub fn defaultOptions() LinearWarmupScheduler {
        return .{};
    }

    pub fn initDefault() LinearWarmupScheduler {
        return .{};
    }

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

/// 指数衰减学习率调度器 (Exponential Learning Rate, ExponentialLR)
pub const ExponentialLRScheduler = struct {
    base_lr: f32 = 1e-3,
    gamma: f32 = 0.95,

    pub const default: ExponentialLRScheduler = .{};
    pub fn defaultOptions() ExponentialLRScheduler {
        return .{};
    }

    pub fn initDefault() ExponentialLRScheduler {
        return .{};
    }

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

/// 统一的多态学习率调度器 (Learning Rate Scheduler, LRScheduler) 包装
pub const LRScheduler = union(enum) {
    cosine: CosineScheduler,
    step_lr: StepLRScheduler,
    warmup: LinearWarmupScheduler,
    exponential: ExponentialLRScheduler,

    pub const default: LRScheduler = .{ .cosine = .{} };

    pub fn defaultConfig() LRScheduler {
        return default;
    }

    pub fn defaultOptions() LRScheduler {
        return default;
    }

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
