//! Models - 专用深度学习模型与专属架构库
//! 收敛具体模型架构 (如 Gemma 4) 及其特定非通用算子与端到端推理实现。

const std = @import("std");

pub const gemma4 = @import("models/gemma4.zig");

test {
    std.testing.refAllDecls(@This());
    _ = @import("models/tests.zig");
}
