//! Backend - 计算后端、底层执行器与硬件加速基础设施
//! 统一管理数据并行执行器 (Executor)、系统级 BLAS (Accelerate / 回退 GEMM) 与未来的 Apple Silicon Metal 加速。

const std = @import("std");

pub const activation = @import("backend/activation.zig");
pub const GeluApproximation = activation.GeluApproximation;

pub const attention_mask = @import("backend/attention_mask.zig");

pub const executor = @import("backend/executor.zig");
pub const Executor = executor.Executor;

pub const cblas = @import("backend/cblas.zig");

pub const mlx = @import("backend/mlx.zig");

pub const offload = @import("backend/offload.zig");

pub const rope = @import("backend/rope.zig");

pub const sampling = @import("backend/sampling.zig");

pub const vec_math = @import("backend/vec_math.zig");

test {
    std.testing.refAllDecls(@This());
    _ = @import("backend/activation.zig");
    _ = @import("backend/attention_mask.zig");
    _ = @import("backend/executor.zig");
    _ = @import("backend/cblas.zig");
    _ = @import("backend/mlx.zig");
    _ = @import("backend/offload.zig");
    _ = @import("backend/rope.zig");
    _ = @import("backend/sampling.zig");
    _ = @import("backend/vec_math.zig");
    _ = @import("backend/tests.zig");
}
