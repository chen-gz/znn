//! 张量 (Tensor) 与 Apple 机器学习拓展 (Machine Learning Explore, MLX) 桥接转换层
//!
//! 提供 Tensor 与 `backend.mlx.Array` 之间的零拷贝 (Zero-Copy) 内存桥接、切片复制与生命周期转换。

const std = @import("std");
const backend = @import("../backend.zig");
const core = @import("core.zig");
const ops = @import("ops.zig");
const Tensor = core.Tensor;

pub const BridgeError = error{
    MlxNotSupported,
    NonContiguousTensor,
    InvalidShape,
    ShapeMismatch,
    OutOfMemory,
};

/// 将只读张量复制转换为 MLX 数组 (Apple Machine Learning Explore, MLX)
pub fn toMlxArray(self: *const Tensor) !backend.mlx.Array {
    if (!backend.mlx.has_mlx) return error.MlxNotSupported;
    if (!self.isContiguous()) return error.NonContiguousTensor;
    return try backend.mlx.Array.fromSlice(self.data[0..self.shape.numel()], self.shape.dims[0..self.shape.len]);
}

/// 将张量以零拷贝 (Zero-Copy) 方式映射为 MLX 数组 (统一内存架构 Unified Memory Architecture, UMA)
pub fn toMlxArrayZeroCopy(self: *Tensor) !backend.mlx.Array {
    if (!backend.mlx.has_mlx) return error.MlxNotSupported;
    if (!self.isContiguous()) return error.NonContiguousTensor;
    return try backend.mlx.Array.fromSliceZeroCopy(self.data[0..self.shape.numel()], self.shape.dims[0..self.shape.len]);
}

/// 将 MLX 数组计算完成后的数据拷贝回张量
pub fn copyFromMlxArray(self: *Tensor, arr: backend.mlx.Array) !void {
    if (!backend.mlx.has_mlx) return error.MlxNotSupported;
    if (!self.isContiguous()) return error.NonContiguousTensor;
    try arr.copyToSlice(self.data[0..self.shape.numel()]);
}

/// 从已求值的 MLX 数组新建分配一个张量
pub fn fromMlxArray(allocator: std.mem.Allocator, arr: backend.mlx.Array) !*Tensor {
    if (!backend.mlx.has_mlx) return error.MlxNotSupported;
    var shape_buf: [8]usize = undefined;
    const n = try arr.shape(&shape_buf);
    const t = try ops.zeros(allocator, shape_buf[0..n]);
    errdefer ops.free(allocator, t);
    try arr.copyToSlice(t.data);
    return t;
}
