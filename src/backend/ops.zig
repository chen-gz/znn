//! 多后端算子统一分派器 (Unified Multi-Backend Operator Dispatcher)
//!
//! 提供跨硬件计算后端的统一算子调用入口（CPU SIMD/Accelerate、Apple MLX/Metal、
//! 以及未来的 NVIDIA CUDA / Vulkan 等），上层（autodiff、nn、models、engine）
//! 无需手工书写针对不同硬件后端的特定分支。

const std = @import("std");
const device_mod = @import("device.zig");
const DeviceType = device_mod.DeviceType;
const offload = @import("offload.zig");
const cblas = @import("cblas.zig");
const vec_math = @import("vec_math.zig");
const activation = @import("activation.zig");

pub const OpError = offload.OffloadError || error{
    DeviceNotSupported,
    ShapeMismatch,
    IncompatibleDimensions,
};

/// 通用矩阵乘法多后端分派：C = A * B
pub fn matmul(
    A: anytype,
    B: anytype,
    C: anytype,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev == .gpu and device_mod.isAvailable(.gpu)) {
        return offload.matmul(A, B, C, stream);
    }

    // CPU 后备分派路径 (使用 cblas_sgemm: macOS 上调用 Accelerate，Linux/Windows 上调用纯 Zig SGEMM)
    const M: usize = A.shape.dims[0];
    const K: usize = A.shape.dims[1];
    const K2: usize = B.shape.dims[0];
    const N: usize = B.shape.dims[1];
    if (K != K2) return error.ShapeMismatch;

    cblas.cblas_sgemm(
        cblas.CblasRowMajor,
        cblas.CblasNoTrans,
        cblas.CblasNoTrans,
        @intCast(M),
        @intCast(N),
        @intCast(K),
        1.0,
        A.data.ptr,
        @intCast(K),
        B.data.ptr,
        @intCast(N),
        0.0,
        C.data.ptr,
        @intCast(N),
    );
}

/// 均方根层归一化多后端分派 (RMSNorm Dispatcher)
pub fn rmsNorm(
    X: anytype,
    weight: anytype,
    out: anytype,
    eps: f32,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev == .gpu and device_mod.isAvailable(.gpu)) {
        return offload.rmsNorm(X, weight, out, eps, stream);
    }

    // CPU 后备分派路径
    const dim = X.shape.dims[X.shape.len - 1];
    const w_slice: ?[]const f32 = blk: {
        const WT = @TypeOf(weight);
        if (@typeInfo(WT) == .optional) {
            if (weight) |w| break :blk w.data;
            break :blk null;
        } else if (@typeInfo(WT) == .null) {
            break :blk null;
        } else {
            break :blk weight.data;
        }
    };

    vec_math.rmsNormRows(X.data, w_slice, out.data, null, dim, eps, 0.0);
}

/// 逐元素加法多后端分派：C = A + B
pub fn add(
    A: anytype,
    B: anytype,
    C: anytype,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev == .gpu and device_mod.isAvailable(.gpu)) {
        return offload.add(A, B, C, stream);
    }

    // CPU 向量化逐元素相加
    std.debug.assert(A.data.len == B.data.len and C.data.len == A.data.len);
    for (A.data, B.data, C.data) |a, b, *c| {
        c.* = a + b;
    }
}

/// 逐元素乘法多后端分派：C = A * B
pub fn multiply(
    A: anytype,
    B: anytype,
    C: anytype,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev == .gpu and device_mod.isAvailable(.gpu)) {
        return offload.multiply(A, B, C, stream);
    }

    // CPU 向量化逐元素相乘
    std.debug.assert(A.data.len == B.data.len and C.data.len == A.data.len);
    for (A.data, B.data, C.data) |a, b, *c| {
        c.* = a * b;
    }
}

/// 标量乘法多后端分派：C = A * scalar
pub fn mulScalar(
    A: anytype,
    scalar: f32,
    C: anytype,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev == .gpu and device_mod.isAvailable(.gpu)) {
        return offload.mulScalar(A, scalar, C, stream);
    }

    // CPU 标量缩放拷贝
    vec_math.scaleCopy(C.data, A.data, scalar);
}

/// 激活函数多后端分派 (GELU Dispatcher)
pub fn gelu(
    A: anytype,
    out: anytype,
    approximation: activation.GeluApproximation,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev == .gpu and device_mod.isAvailable(.gpu)) {
        return offload.gelu(A, out, approximation, stream);
    }

    // CPU 激活函数逐元素映射
    std.debug.assert(A.data.len == out.data.len);
    switch (approximation) {
        .none => {
            for (A.data, out.data) |a, *o| {
                o.* = activation.gelu(a);
            }
        },
        .tanh => {
            for (A.data, out.data) |a, *o| {
                o.* = activation.geluTanh(a);
            }
        },
    }
}

/// 归一化指数函数多后端分派 (Softmax Dispatcher)
pub fn softmax(
    A: anytype,
    out: anytype,
    axis: c_int,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev == .gpu and device_mod.isAvailable(.gpu)) {
        return offload.softmax(A, out, axis, stream);
    }

    // CPU 单行/多行 Softmax 映射
    const last_dim = A.shape.dims[A.shape.len - 1];
    const rows = A.data.len / last_dim;
    for (0..rows) |i| {
        vec_math.softmaxRow(
            A.data[i * last_dim .. (i + 1) * last_dim],
            out.data[i * last_dim .. (i + 1) * last_dim],
        );
    }
}

/// 键值缓存注意力多后端分派入口
pub fn cachedAttention(
    allocator: std.mem.Allocator,
    q: anytype,
    k: anytype,
    v: anytype,
    scale: f32,
    is_sliding: bool,
    sliding_window: usize,
    pos: usize,
    seq_len: usize,
    stream: anytype,
) OpError!@TypeOf(q) {
    return offload.cachedAttention(allocator, q, k, v, scale, is_sliding, sliding_window, pos, seq_len, stream);
}
