const std = @import("std");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
pub const checkpoint = @import("checkpoint.zig");

pub const OPTIMIZER_MAGIC = checkpoint.OPTIMIZER_MAGIC;
pub const OPTIMIZER_VERSION = checkpoint.OPTIMIZER_VERSION;
pub const OptimizerTypeTag = checkpoint.OptimizerTypeTag;
pub const CheckpointError = checkpoint.CheckpointError;

// =========================================================================
// 梯度裁剪 (Gradient Clipping)
// =========================================================================

/// 全局梯度欧几里得二范数 (L2 Norm) 裁剪 (Gradient Norm Clipping)
pub fn clipGradNorm(params: []*Tensor, max_norm: f32) f32 {
    var total_norm_sq: f64 = 0.0;
    for (params) |param| {
        total_norm_sq += sumSquares(param.grad);
    }
    const total_norm: f32 = @floatCast(@sqrt(total_norm_sq));
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

/// 平方和：块内使用两路 SIMD 向量 (`@Vector`) 部分和并行累加 (严格浮点语义下标量累加无法被自动向量化)，
/// 每块结果再以 f64 汇总，避免百万级元素在单个 f32 累加器上累积舍入误差
pub fn sumSquares(values: []const f32) f64 {
    const vec_len = std.simd.suggestVectorLength(f32) orelse 4;
    const Vec = @Vector(vec_len, f32);
    const block_len = 2 * vec_len * 256;

    var total: f64 = 0.0;
    var i: usize = 0;
    while (i + 2 * vec_len <= values.len) {
        const block_end = i + @min(block_len, (values.len - i) / (2 * vec_len) * (2 * vec_len));
        var acc0: Vec = @splat(0.0);
        var acc1: Vec = @splat(0.0);
        while (i < block_end) : (i += 2 * vec_len) {
            const a: Vec = values[i..][0..vec_len].*;
            const b: Vec = values[i + vec_len ..][0..vec_len].*;
            acc0 += a * a;
            acc1 += b * b;
        }
        const block_sum: @Vector(vec_len, f64) = @floatCast(acc0 + acc1);
        total += @reduce(.Add, block_sum);
    }
    while (i < values.len) : (i += 1) {
        const x: f64 = values[i];
        total += x * x;
    }
    return total;
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

    pub const default: GradClipConfig = .{ .none = {} };
    pub fn defaultConfig() GradClipConfig {
        return .{ .none = {} };
    }
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
