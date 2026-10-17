//! Backend - 计算后端、底层执行器与硬件加速基础设施
//! 统一管理数据并行执行器 (Executor)、系统级 BLAS (Accelerate / 回退 GEMM) 与未来的 Apple Silicon Metal 加速。

const std = @import("std");

pub const executor = @import("backend/executor.zig");
pub const Executor = executor.Executor;

pub const cblas = @import("backend/cblas.zig");

test {
    std.testing.refAllDecls(@This());
    _ = @import("backend/executor.zig");
    _ = @import("backend/cblas.zig");
}
