//! 多后端算子统一分派器 (Unified Multi-Backend Operator Dispatcher)
//!
//! 提供跨硬件计算后端的统一算子调用入口（中央处理器 CPU SIMD/Accelerate、Apple 机器学习拓展 MLX/Metal、
//! 以及 NVIDIA 统一计算设备架构 CUDA），上层（autodiff、nn、models、engine）无需手工书写针对不同硬件后端的特定分支。

const std = @import("std");
const device_mod = @import("device.zig");
const DeviceType = device_mod.DeviceType;
const offload = @import("offload.zig");
const cuda = @import("cuda.zig");
const cblas = @import("cblas.zig");
const vec_math = @import("vec_math.zig");
const activation = @import("activation.zig");
const rope_mod = @import("rope.zig");

pub const OpError = offload.OffloadError || cuda.CudaError || error{
    DeviceNotSupported,
    ShapeMismatch,
    IncompatibleDimensions,
};

pub const BorrowedCudaArray = struct {
    arr: cuda.Array,
    needs_deinit: bool,

    pub fn deinit(self: *BorrowedCudaArray) void {
        if (self.needs_deinit) {
            self.arr.deinit();
        }
    }
};

/// 借用张量的 CUDA 显存数组句柄 (若张量已在 CUDA 显存常驻则直接复用，实现零拷贝借用；若为 CPU 主机张量则分配显存并拷贝)
pub fn borrowCudaArray(t: anytype) cuda.CudaError!BorrowedCudaArray {
    const T = @TypeOf(t);
    const info = @typeInfo(T);
    if (info == .pointer and @hasField(info.pointer.child, "device_array")) {
        if (t.device_array) |da| {
            if (da.asCuda()) |arr| {
                return .{ .arr = arr, .needs_deinit = false };
            }
        }
        if (t.data.len > 0) {
            const arr = try cuda.Array.fromTensor(t);
            return .{ .arr = arr, .needs_deinit = true };
        }
        return error.EmptyTensor;
    }
    const arr = try cuda.Array.fromTensor(t);
    return .{ .arr = arr, .needs_deinit = true };
}

/// 将计算结果安全存入输出张量 (若输出为 GPU 驻留张量则直接接管 CUDA 显存句柄；若为 CPU 主机张量则拷回内存)
pub fn storeCudaOutput(out: anytype, res_cuda: cuda.Array) cuda.CudaError!void {
    const T = @TypeOf(out);
    const info = @typeInfo(T);
    if (info == .pointer and @hasField(info.pointer.child, "device_array")) {
        if (out.device.isGpu() or out.data.len == 0) {
            if (out.device_array) |*old| old.deinit();
            out.device_array = .{ .cuda = res_cuda };
            out.device = .cuda;
            return;
        }
    }
    var to_free = res_cuda;
    defer to_free.deinit();
    try res_cuda.copyToTensor(out);
}

/// 通用矩阵乘法多后端分派 (General Matrix Multiply Dispatcher, GEMM Dispatcher)：C = A * B
pub fn matmul(
    A: anytype,
    B: anytype,
    C: anytype,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev.isGpu() and device_mod.isAvailable(dev)) {
        const backend = device_mod.activeBackend(dev);
        if (backend == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            var b_b = try borrowCudaArray(B);
            defer b_b.deinit();
            const c_arr = try cuda.Array.matmul(a_b.arr, b_b.arr, null);
            try storeCudaOutput(C, c_arr);
            return;
        } else if (backend == .mlx) {
            return offload.matmul(A, B, C, stream);
        }
    }

    // 中央处理器 (Central Processing Unit, CPU) 后备分派路径 (使用 cblas_sgemm)
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

/// 均方根层归一化多后端分派 (Root Mean Square Layer Normalization Dispatcher, RMSNorm Dispatcher)
pub fn rmsNorm(
    X: anytype,
    weight: anytype,
    out: anytype,
    eps: f32,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev.isGpu() and device_mod.isAvailable(dev)) {
        const backend = device_mod.activeBackend(dev);
        if (backend == .cuda) {
            var x_b = try borrowCudaArray(X);
            defer x_b.deinit();
            var w_b: ?BorrowedCudaArray = null;
            defer if (w_b) |*w| w.deinit();
            const WT = @TypeOf(weight);
            if (@typeInfo(WT) == .optional) {
                if (weight) |w| w_b = try borrowCudaArray(w);
            } else if (@typeInfo(WT) != .null) {
                w_b = try borrowCudaArray(weight);
            }
            const w_arr: ?cuda.Array = if (w_b) |w| w.arr else null;
            const out_arr = try cuda.Array.rmsNorm(x_b.arr, w_arr, eps, null);
            try storeCudaOutput(out, out_arr);
            return;
        } else if (backend == .mlx) {
            return offload.rmsNorm(X, weight, out, eps, stream);
        }
    }

    // 中央处理器 (Central Processing Unit, CPU) 后备分派路径
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

/// 逐元素加法多后端分派 (Elementwise Addition Dispatcher)：C = A + B
pub fn add(
    A: anytype,
    B: anytype,
    C: anytype,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev.isGpu() and device_mod.isAvailable(dev)) {
        const backend = device_mod.activeBackend(dev);
        if (backend == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            var b_b = try borrowCudaArray(B);
            defer b_b.deinit();
            const c_arr = try cuda.Array.add(a_b.arr, b_b.arr, null);
            try storeCudaOutput(C, c_arr);
            return;
        } else if (backend == .mlx) {
            return offload.add(A, B, C, stream);
        }
    }

    // 中央处理器 (Central Processing Unit, CPU) 向量化逐元素相加
    std.debug.assert(A.data.len == B.data.len and C.data.len == A.data.len);
    for (A.data, B.data, C.data) |a, b, *c| {
        c.* = a + b;
    }
}

/// 逐元素乘法多后端分派 (Elementwise Multiply Dispatcher)：C = A * B
pub fn multiply(
    A: anytype,
    B: anytype,
    C: anytype,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev.isGpu() and device_mod.isAvailable(dev)) {
        const backend = device_mod.activeBackend(dev);
        if (backend == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            var b_b = try borrowCudaArray(B);
            defer b_b.deinit();
            const c_arr = try cuda.Array.multiply(a_b.arr, b_b.arr, null);
            try storeCudaOutput(C, c_arr);
            return;
        } else if (backend == .mlx) {
            return offload.multiply(A, B, C, stream);
        }
    }

    // 中央处理器 (Central Processing Unit, CPU) 向量化逐元素相乘
    std.debug.assert(A.data.len == B.data.len and C.data.len == A.data.len);
    for (A.data, B.data, C.data) |a, b, *c| {
        c.* = a * b;
    }
}

/// 标量乘法多后端分派 (Scalar Multiply Dispatcher)：C = A * scalar
pub fn mulScalar(
    A: anytype,
    scalar: f32,
    C: anytype,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev.isGpu() and device_mod.isAvailable(dev)) {
        const backend = device_mod.activeBackend(dev);
        if (backend == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            const c_arr = try cuda.Array.mulScalar(a_b.arr, scalar, null);
            try storeCudaOutput(C, c_arr);
            return;
        } else if (backend == .mlx) {
            return offload.mulScalar(A, scalar, C, stream);
        }
    }

    // 中央处理器 (Central Processing Unit, CPU) 标量缩放拷贝
    vec_math.scaleCopy(C.data, A.data, scalar);
}

/// 高斯误差线性单元多后端分派 (Gaussian Error Linear Unit Dispatcher, GELU Dispatcher)
pub fn gelu(
    A: anytype,
    out: anytype,
    approximation: activation.GeluApproximation,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev.isGpu() and device_mod.isAvailable(dev)) {
        const backend = device_mod.activeBackend(dev);
        if (backend == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            const out_arr = try cuda.Array.gelu(a_b.arr, approximation, null);
            try storeCudaOutput(out, out_arr);
            return;
        } else if (backend == .mlx) {
            return offload.gelu(A, out, approximation, stream);
        }
    }

    // 中央处理器 (Central Processing Unit, CPU) 激活函数逐元素映射
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
    if (dev.isGpu() and device_mod.isAvailable(dev)) {
        const backend = device_mod.activeBackend(dev);
        if (backend == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            const out_arr = try cuda.Array.softmax(a_b.arr, axis, null);
            try storeCudaOutput(out, out_arr);
            return;
        } else if (backend == .mlx) {
            return offload.softmax(A, out, axis, stream);
        }
    }

    // 中央处理器 (Central Processing Unit, CPU) 单行/多行 Softmax 映射
    const last_dim = A.shape.dims[A.shape.len - 1];
    const rows = A.data.len / last_dim;
    for (0..rows) |i| {
        vec_math.softmaxRow(
            A.data[i * last_dim .. (i + 1) * last_dim],
            out.data[i * last_dim .. (i + 1) * last_dim],
        );
    }
}

/// 旋转位置编码多后端分派 (Rotary Position Embedding Dispatcher, RoPE Dispatcher)
pub fn rope(
    X: anytype,
    out: anytype,
    options: rope_mod.RopeOptions,
    start_pos: usize,
    dev: DeviceType,
    stream: anytype,
) OpError!void {
    if (dev.isGpu() and device_mod.isAvailable(dev)) {
        const backend = device_mod.activeBackend(dev);
        if (backend == .cuda) {
            var x_b = try borrowCudaArray(X);
            defer x_b.deinit();
            const head_dim = X.shape.dims[X.shape.len - 1];
            const traditional = options.mode == .interleaved;
            const out_arr = try cuda.Array.fastRoPE(x_b.arr, head_dim, traditional, options.base, 1.0, start_pos, null);
            try storeCudaOutput(out, out_arr);
            return;
        } else if (backend == .mlx) {
            return offload.rope(X, out, options, start_pos, stream);
        }
    }

    return error.DeviceNotSupported;
}

/// 键值缓存注意力多后端分派入口 (Key-Value Cache Attention Dispatcher, KVCache Attention Dispatcher)
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

test "ops: multi-backend dispatcher tests" {
    // 构造测试张量模拟结构体
    const MockTensor = struct {
        data: []f32,
        shape: struct {
            dims: [4]usize,
            len: usize,
        },
    };

    var a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b_data = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var c_data = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    var A = MockTensor{ .data = &a_data, .shape = .{ .dims = .{ 2, 2, 0, 0 }, .len = 2 } };
    var B = MockTensor{ .data = &b_data, .shape = .{ .dims = .{ 2, 2, 0, 0 }, .len = 2 } };
    var C = MockTensor{ .data = &c_data, .shape = .{ .dims = .{ 2, 2, 0, 0 }, .len = 2 } };

    // CPU 分派验证
    try matmul(&A, &B, &C, .cpu, null);
    try std.testing.expectEqual(@as(f32, 1.0), C.data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), C.data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), C.data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), C.data[3]);

    // GPU / CUDA 分派验证 (若可用)
    if (device_mod.isAvailable(.gpu)) {
        C.data[0] = 0;
        C.data[1] = 0;
        C.data[2] = 0;
        C.data[3] = 0;
        try matmul(&A, &B, &C, .gpu, null);
        try std.testing.expectEqual(@as(f32, 1.0), C.data[0]);
        try std.testing.expectEqual(@as(f32, 2.0), C.data[1]);
        try std.testing.expectEqual(@as(f32, 3.0), C.data[2]);
        try std.testing.expectEqual(@as(f32, 4.0), C.data[3]);

        // 验证 Add 分派
        try add(&A, &A, &C, .gpu, null);
        try std.testing.expectEqual(@as(f32, 2.0), C.data[0]);
        try std.testing.expectEqual(@as(f32, 8.0), C.data[3]);

        // 验证 Multiply 分派
        try multiply(&A, &B, &C, .gpu, null);
        try std.testing.expectEqual(@as(f32, 1.0), C.data[0]);
        try std.testing.expectEqual(@as(f32, 0.0), C.data[1]);

        // 验证 MulScalar 分派
        try mulScalar(&A, 5.0, &C, .gpu, null);
        try std.testing.expectEqual(@as(f32, 5.0), C.data[0]);
        try std.testing.expectEqual(@as(f32, 20.0), C.data[3]);

        // 验证 RMSNorm 分派
        try rmsNorm(&A, null, &C, 1e-5, .gpu, null);
        try std.testing.expect(C.data[0] > 0.0);

        // 验证 Softmax 分派
        try softmax(&A, &C, -1, .gpu, null);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), C.data[0] + C.data[1], 0.001);
    }
}
