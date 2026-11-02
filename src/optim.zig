//! 优化器与学习率调度器顶层公共门面 (Optimizers & Schedulers Facade)
//!
//! 聚合导出 SGD、Adam、AdamW、LRScheduler 调度器族、梯度裁剪与二进制检查点持久化机制。

pub const types = @import("optim/types.zig");
pub const checkpoint = @import("optim/checkpoint.zig");
pub const sgd = @import("optim/sgd.zig");
pub const adam = @import("optim/adam.zig");
pub const lr_scheduler = @import("optim/lr_scheduler.zig");

// --- 检查点与魔数 ---
pub const OPTIMIZER_MAGIC = checkpoint.OPTIMIZER_MAGIC;
pub const OPTIMIZER_VERSION = checkpoint.OPTIMIZER_VERSION;
pub const OptimizerTypeTag = checkpoint.OptimizerTypeTag;
pub const CheckpointError = checkpoint.CheckpointError;

// --- 优化器 ---
pub const SGDConfig = sgd.SGDConfig;
pub const SGDOptimizer = sgd.SGDOptimizer;

pub const AdamConfig = adam.AdamConfig;
pub const AdamOptimizer = adam.AdamOptimizer;
pub const AdamWConfig = adam.AdamWConfig;
pub const AdamWOptimizer = adam.AdamWOptimizer;
pub const AdamUpdate = adam.AdamUpdate;

// --- 学习率调度器 ---
pub const CosineScheduler = lr_scheduler.CosineScheduler;
pub const StepLRScheduler = lr_scheduler.StepLRScheduler;
pub const LinearWarmupScheduler = lr_scheduler.LinearWarmupScheduler;
pub const ExponentialLRScheduler = lr_scheduler.ExponentialLRScheduler;
pub const LRScheduler = lr_scheduler.LRScheduler;

// --- 梯度裁剪 ---
pub const clipGradNorm = types.clipGradNorm;
pub const clipGradValue = types.clipGradValue;
pub const GradClipConfig = types.GradClipConfig;
pub const clipGradients = types.clipGradients;

test {
    _ = @import("optim/tests.zig");
}
