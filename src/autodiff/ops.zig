//! 多后端算子统一分派器 (Unified Multi-Backend Operator Dispatcher)
//!
//! 提供跨硬件计算后端的统一算子调用入口（中央处理器 CPU SIMD/Accelerate、Apple 机器学习拓展 MLX/Metal、
//! 以及 NVIDIA 统一计算设备架构 CUDA），上层（autodiff、nn、models、engine）无需手工书写针对不同硬件后端的特定分支。
//! 遵循分层规范，位于 autodiff 包（层级 2），使用具体 `*const Tensor` / `*Tensor` 类型，零鸭子类型。

const std = @import("std");
const backend = @import("../backend.zig");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const Device = backend.Device;
const DeviceType = backend.DeviceType;
const cblas = backend.cblas;
const vec_math = backend.vec_math;
const activation = backend.activation;
const cuda = backend.cuda;
const mlx = backend.mlx;
const offload = backend.offload;

pub const OpError = cuda.CudaError || mlx.MlxError || error{
    DeviceNotSupported,
    ShapeMismatch,
    IncompatibleDimensions,
    EmptyTensor,
    OffloadNotSupported,
    OutOfMemory,
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

/// 借用张量的 CUDA 显存数组句柄
pub fn borrowCudaArray(t: *const Tensor) cuda.CudaError!BorrowedCudaArray {
    if (t.device_array) |da| {
        if (da.asCuda()) |arr| {
            return .{ .arr = arr, .needs_deinit = false };
        }
    }
    if (t.data.len > 0) {
        const arr = try cuda.Array.fromSlice(t.data, t.shape.dims[0..t.shape.len]);
        return .{ .arr = arr, .needs_deinit = true };
    }
    return error.EmptyTensor;
}

/// 将计算结果安全存入输出张量 (若输出为 GPU 驻留张量则直接接管 CUDA 显存句柄；若为 CPU 主机张量则拷回内存)
pub fn storeCudaOutput(out: *Tensor, res_cuda: cuda.Array) cuda.CudaError!void {
    if (out.device.isGpu() or out.data.len == 0) {
        if (out.device_array) |*old| old.deinit();
        out.device_array = .{ .cuda = res_cuda };
        out.device = .{ .type = .cuda };
        return;
    }
    var to_free = res_cuda;
    defer to_free.deinit();
    try res_cuda.copyToSlice(out.data);
}

pub const BorrowedMlxArray = struct {
    arr: mlx.Array,
    needs_deinit: bool,

    pub fn deinit(self: *BorrowedMlxArray) void {
        if (self.needs_deinit) {
            self.arr.deinit();
        }
    }
};

/// 借用张量的 MLX 数组句柄
pub fn borrowMlxArray(t: *const Tensor) !BorrowedMlxArray {
    if (t.device_array) |da| {
        if (da.asMlx()) |arr| {
            return .{ .arr = arr, .needs_deinit = false };
        }
    }
    if (t.data.len > 0) {
        const arr = try mlx.Array.fromSlice(t.data, t.shape.dims[0..t.shape.len]);
        return .{ .arr = arr, .needs_deinit = true };
    }
    return error.EmptyTensor;
}

/// 将 MLX 计算结果安全存入输出张量
pub fn storeMlxOutput(out: *Tensor, res_mlx: mlx.Array) !void {
    if (out.device.isGpu() or out.data.len == 0) {
        if (out.device_array) |*old| old.deinit();
        out.device_array = .{ .mlx = res_mlx };
        out.device = .{ .type = .metal };
        return;
    }
    var to_free = res_mlx;
    defer to_free.deinit();
    try res_mlx.copyToSlice(out.data);
}

/// 通用矩阵乘法多后端分派 (General Matrix Multiply Dispatcher, GEMM Dispatcher)：C = A * B
pub fn matmul(
    A: *const Tensor,
    B: *const Tensor,
    C: *Tensor,
    dev: Device,
    stream: ?mlx.Stream,
) OpError!void {
    if (dev.isGpu() and backend.device.isAvailable(dev)) {
        const backend_type = backend.device.activeBackend(dev);
        if (backend_type == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            var b_b = try borrowCudaArray(B);
            defer b_b.deinit();
            const c_arr = try cuda.Array.matmul(a_b.arr, b_b.arr, null);
            try storeCudaOutput(C, c_arr);
            return;
        } else if (backend_type == .mlx) {
            var a_b = try borrowMlxArray(A);
            defer a_b.deinit();
            var b_b = try borrowMlxArray(B);
            defer b_b.deinit();
            var c_mlx = try mlx.Array.matmul(a_b.arr, b_b.arr, stream);
            errdefer c_mlx.deinit();
            try storeMlxOutput(C, c_mlx);
            return;
        }
    }

    // CPU 后备分派路径 (使用 cblas_sgemm)
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
    X: *const Tensor,
    weight: ?*const Tensor,
    out: *Tensor,
    eps: f32,
    dev: Device,
    stream: ?mlx.Stream,
) OpError!void {
    if (dev.isGpu() and backend.device.isAvailable(dev)) {
        const backend_type = backend.device.activeBackend(dev);
        if (backend_type == .cuda) {
            var x_b = try borrowCudaArray(X);
            defer x_b.deinit();
            var w_b: ?BorrowedCudaArray = null;
            defer if (w_b) |*w| w.deinit();
            if (weight) |w| w_b = try borrowCudaArray(w);
            const w_arr: ?cuda.Array = if (w_b) |w| w.arr else null;
            const out_arr = try cuda.Array.rmsNorm(x_b.arr, w_arr, eps, null);
            try storeCudaOutput(out, out_arr);
            return;
        } else if (backend_type == .mlx) {
            var x_b = try borrowMlxArray(X);
            defer x_b.deinit();
            var w_b: ?BorrowedMlxArray = null;
            defer if (w_b) |*w| w.deinit();
            if (weight) |w| w_b = try borrowMlxArray(w);
            const D = X.shape.dims[X.shape.len - 1];
            var w_reshaped: ?mlx.Array = null;
            defer if (w_reshaped) |*wr| wr.deinit();
            const w_arr: ?mlx.Array = if (w_b) |w| blk: {
                if (w.arr.ndim() != 1) {
                    w_reshaped = try w.arr.reshape(&.{D}, stream);
                    break :blk w_reshaped.?;
                }
                break :blk w.arr;
            } else null;
            var out_mlx = try mlx.Array.fastRmsNorm(x_b.arr, w_arr, eps, stream);
            errdefer out_mlx.deinit();
            try storeMlxOutput(out, out_mlx);
            return;
        }
    }

    // CPU 后备分派路径
    const dim = X.shape.dims[X.shape.len - 1];
    const w_slice: ?[]const f32 = if (weight) |w| w.data else null;
    vec_math.rmsNormRows(X.data, w_slice, out.data, null, dim, eps, 0.0);
}

/// 逐元素加法多后端分派 (Elementwise Addition Dispatcher)：C = A + B
pub fn add(
    A: *const Tensor,
    B: *const Tensor,
    C: *Tensor,
    dev: Device,
    stream: ?mlx.Stream,
) OpError!void {
    if (dev.isGpu() and backend.device.isAvailable(dev)) {
        const backend_type = backend.device.activeBackend(dev);
        if (backend_type == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            var b_b = try borrowCudaArray(B);
            defer b_b.deinit();
            const c_arr = try cuda.Array.add(a_b.arr, b_b.arr, null);
            try storeCudaOutput(C, c_arr);
            return;
        } else if (backend_type == .mlx) {
            var a_b = try borrowMlxArray(A);
            defer a_b.deinit();
            var b_b = try borrowMlxArray(B);
            defer b_b.deinit();
            var c_mlx = try mlx.Array.add(a_b.arr, b_b.arr, stream);
            errdefer c_mlx.deinit();
            try storeMlxOutput(C, c_mlx);
            return;
        }
    }

    // CPU 向量化逐元素相加
    std.debug.assert(A.data.len == B.data.len and C.data.len == A.data.len);
    for (A.data, B.data, C.data) |a, b, *c| {
        c.* = a + b;
    }
}

/// 逐元素乘法多后端分派 (Elementwise Multiply Dispatcher)：C = A * B
pub fn multiply(
    A: *const Tensor,
    B: *const Tensor,
    C: *Tensor,
    dev: Device,
    stream: ?mlx.Stream,
) OpError!void {
    if (dev.isGpu() and backend.device.isAvailable(dev)) {
        const backend_type = backend.device.activeBackend(dev);
        if (backend_type == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            var b_b = try borrowCudaArray(B);
            defer b_b.deinit();
            const c_arr = try cuda.Array.multiply(a_b.arr, b_b.arr, null);
            try storeCudaOutput(C, c_arr);
            return;
        } else if (backend_type == .mlx) {
            var a_b = try borrowMlxArray(A);
            defer a_b.deinit();
            var b_b = try borrowMlxArray(B);
            defer b_b.deinit();
            var c_mlx = try mlx.Array.multiply(a_b.arr, b_b.arr, stream);
            errdefer c_mlx.deinit();
            try storeMlxOutput(C, c_mlx);
            return;
        }
    }

    // CPU 向量化逐元素相乘
    std.debug.assert(A.data.len == B.data.len and C.data.len == A.data.len);
    for (A.data, B.data, C.data) |a, b, *c| {
        c.* = a * b;
    }
}

/// 标量乘法多后端分派 (Scalar Multiply Dispatcher)：C = A * scalar
pub fn mulScalar(
    A: *const Tensor,
    scalar: f32,
    C: *Tensor,
    dev: Device,
    stream: ?mlx.Stream,
) OpError!void {
    if (dev.isGpu() and backend.device.isAvailable(dev)) {
        const backend_type = backend.device.activeBackend(dev);
        if (backend_type == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            const c_arr = try cuda.Array.mulScalar(a_b.arr, scalar, null);
            try storeCudaOutput(C, c_arr);
            return;
        } else if (backend_type == .mlx) {
            var a_b = try borrowMlxArray(A);
            defer a_b.deinit();
            var c_mlx = try mlx.Array.mulScalar(a_b.arr, scalar, stream);
            errdefer c_mlx.deinit();
            try storeMlxOutput(C, c_mlx);
            return;
        }
    }

    // CPU 标量缩放拷贝
    vec_math.scaleCopy(C.data, A.data, scalar);
}

/// 高斯误差线性单元多后端分派 (Gaussian Error Linear Unit Dispatcher, GELU Dispatcher)
pub fn gelu(
    A: *const Tensor,
    out: *Tensor,
    approximation: activation.GeluApproximation,
    dev: Device,
    stream: ?mlx.Stream,
) OpError!void {
    if (dev.isGpu() and backend.device.isAvailable(dev)) {
        const backend_type = backend.device.activeBackend(dev);
        if (backend_type == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            const out_arr = try cuda.Array.gelu(a_b.arr, approximation, null);
            try storeCudaOutput(out, out_arr);
            return;
        } else if (backend_type == .mlx) {
            var a_b = try borrowMlxArray(A);
            defer a_b.deinit();
            var out_mlx = try a_b.arr.gelu(approximation, stream);
            errdefer out_mlx.deinit();
            try storeMlxOutput(out, out_mlx);
            return;
        }
    }

    // CPU 激活函数逐元素映射
    std.debug.assert(A.data.len == out.data.len);
    switch (approximation) {
        .none => {
            for (A.data, out.data) |a, *o| {
                o.* = activation.gelu(.none, a);
            }
        },
        .tanh => {
            for (A.data, out.data) |a, *o| {
                o.* = activation.gelu(.tanh, a);
            }
        },
    }
}

/// 归一化指数函数多后端分派 (Softmax Dispatcher)
pub fn softmax(
    A: *const Tensor,
    out: *Tensor,
    axis: c_int,
    dev: Device,
    stream: ?mlx.Stream,
) OpError!void {
    if (dev.isGpu() and backend.device.isAvailable(dev)) {
        const backend_type = backend.device.activeBackend(dev);
        if (backend_type == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            const out_arr = try cuda.Array.softmax(a_b.arr, axis, null);
            try storeCudaOutput(out, out_arr);
            return;
        } else if (backend_type == .mlx) {
            var a_b = try borrowMlxArray(A);
            defer a_b.deinit();
            var res_mlx = try mlx.Array.softmax(a_b.arr, axis, stream);
            errdefer res_mlx.deinit();
            try storeMlxOutput(out, res_mlx);
            return;
        }
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

/// 检查旋转位置编码配置在当前硬件上是否支持
pub fn ropeSupported(head_dim: usize, options: tensor.RopeOptions) bool {
    return options.rotary_offset == 0 and head_dim > 0 and head_dim % 2 == 0 and head_dim / 2 <= tensor.rope.max_angles;
}

/// 旋转位置编码多后端分派 (Rotary Position Embedding Dispatcher, RoPE Dispatcher)
pub fn rope(
    X: *const Tensor,
    out: *Tensor,
    options: tensor.RopeOptions,
    start_pos: usize,
    dev: Device,
    stream: ?mlx.Stream,
) OpError!void {
    if (dev.isGpu() and backend.device.isAvailable(dev)) {
        const backend_type = backend.device.activeBackend(dev);
        if (backend_type == .cuda) {
            var x_b = try borrowCudaArray(X);
            defer x_b.deinit();
            const head_dim = X.shape.dims[X.shape.len - 1];
            const traditional = options.mode == .interleaved;
            const out_arr = try cuda.Array.fastRoPE(x_b.arr, head_dim, traditional, options.base, 1.0, start_pos, null);
            try storeCudaOutput(out, out_arr);
            return;
        } else if (backend_type == .mlx) {
            const head_dim = X.shape.dims[X.shape.len - 1];
            const traditional = options.mode == .interleaved;
            var x_b = try borrowMlxArray(X);
            defer x_b.deinit();

            var y_mlx = if (options.numAngles(head_dim) == head_dim / 2)
                try mlx.Array.fastRoPE(x_b.arr, head_dim, traditional, options.base, 1.0, start_pos, stream)
            else blk: {
                var periods: [tensor.rope.max_angles]f32 = undefined;
                options.fillPeriods(head_dim, periods[0 .. head_dim / 2]);
                var freqs = try mlx.Array.fromSlice(periods[0 .. head_dim / 2], &.{head_dim / 2});
                defer freqs.deinit();
                break :blk try mlx.Array.fastRoPEWithFreqs(x_b.arr, head_dim, traditional, 1.0, start_pos, freqs, stream);
            };
            errdefer y_mlx.deinit();
            try storeMlxOutput(out, y_mlx);
            return;
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
    stream: ?mlx.Stream,
) OpError!@TypeOf(q) {
    const QType = @TypeOf(q);
    if (QType == cuda.Array) {
        return try cuda.Array.cachedAttention(q, k, v, scale, is_sliding, sliding_window, pos, seq_len, stream);
    } else if (QType == backend.device.DeviceArray) {
        switch (q) {
            .mlx => |mq| {
                const mk = k.asMlx() orelse return error.ShapeMismatch;
                const mv = v.asMlx() orelse return error.ShapeMismatch;
                const res = try offload.cachedAttention(allocator, mq, mk, mv, scale, is_sliding, sliding_window, pos, seq_len, stream);
                return .{ .mlx = res };
            },
            .cuda => |cq| {
                const ck = k.asCuda() orelse return error.ShapeMismatch;
                const cv = v.asCuda() orelse return error.ShapeMismatch;
                const res = try cuda.Array.cachedAttention(cq, ck, cv, scale, is_sliding, sliding_window, pos, seq_len, stream);
                return .{ .cuda = res };
            },
        }
    } else {
        return offload.cachedAttention(allocator, q, k, v, scale, is_sliding, sliding_window, pos, seq_len, stream);
    }
}

/// 词元嵌入查表多后端分派 (Token Embedding Lookup Dispatcher)
pub fn embedding(
    W: *const Tensor,
    X: *const Tensor,
    out: *Tensor,
    dev: Device,
    stream: ?mlx.Stream,
) OpError!void {
    if (dev.isGpu() and backend.device.isAvailable(dev)) {
        const backend_type = backend.device.activeBackend(dev);
        if (backend_type == .cuda) {
            var w_b = try borrowCudaArray(W);
            defer w_b.deinit();
            var x_b = try borrowCudaArray(X);
            defer x_b.deinit();
            const res = try cuda.Array.embedding(w_b.arr, x_b.arr, null);
            try storeCudaOutput(out, res);
            return;
        } else if (backend_type == .mlx) {
            var w_b = try borrowMlxArray(W);
            defer w_b.deinit();
            var x_b = try borrowMlxArray(X);
            defer x_b.deinit();
            var x_i32 = try x_b.arr.astype(.int32, stream);
            defer x_i32.deinit();
            var res = try w_b.arr.takeAxis(x_i32, 0, stream);
            errdefer res.deinit();
            try storeMlxOutput(out, res);
            return;
        }
    }

    // CPU 查表回退路径
    const D = W.shape.dims[1];
    const V = W.shape.dims[0];
    for (0..X.data.len) |i| {
        const idx: usize = @intFromFloat(X.data[i]);
        if (idx < V) {
            @memcpy(out.data[i * D .. (i + 1) * D], W.data[idx * D .. (idx + 1) * D]);
        } else {
            @memset(out.data[i * D .. (i + 1) * D], 0.0);
        }
    }
}

/// 软截断算子多后端分派 (Soft Capping Dispatcher): y = cap * tanh(x / cap)
pub fn softcap(
    A: *const Tensor,
    out: *Tensor,
    cap: f32,
    dev: Device,
    stream: ?mlx.Stream,
) OpError!void {
    if (cap <= 0.0) return;
    if (dev.isGpu() and backend.device.isAvailable(dev)) {
        const backend_type = backend.device.activeBackend(dev);
        if (backend_type == .cuda) {
            var a_b = try borrowCudaArray(A);
            defer a_b.deinit();
            const res = try cuda.Array.softcap(a_b.arr, cap, null);
            try storeCudaOutput(out, res);
            return;
        } else if (backend_type == .mlx) {
            var a_b = try borrowMlxArray(A);
            defer a_b.deinit();
            var scaled = try a_b.arr.mulScalar(1.0 / cap, stream);
            defer scaled.deinit();
            var tanh_l = try mlx.Array.tanh(scaled, stream);
            defer tanh_l.deinit();
            var res = try tanh_l.mulScalar(cap, stream);
            errdefer res.deinit();
            try storeMlxOutput(out, res);
            return;
        }
    }

    // CPU 软截断回退路径
    for (0..A.data.len) |i| {
        out.data[i] = cap * std.math.tanh(A.data[i] / cap);
    }
}

/// 缩放点积注意力多后端分派 (Scaled Dot-Product Attention Dispatcher, SDPA Dispatcher)
pub fn attention(
    allocator: std.mem.Allocator,
    queries: *const Tensor,
    keys: *const Tensor,
    values: *const Tensor,
    out: *Tensor,
    scale: f32,
    mask: tensor.Mask,
    q_pos0: usize,
    k_pos0: usize,
    dev: Device,
    stream: ?mlx.Stream,
) OpError!void {
    if (dev.isGpu() and backend.device.isAvailable(dev)) {
        const backend_type = backend.device.activeBackend(dev);
        if (backend_type == .mlx) {
            var q_b = try borrowMlxArray(queries);
            defer q_b.deinit();
            var k_b = try borrowMlxArray(keys);
            defer k_b.deinit();
            var v_b = try borrowMlxArray(values);
            defer v_b.deinit();

            var out_mlx = offload.maskedAttention(allocator, q_b.arr, k_b.arr, v_b.arr, scale, mask, q_pos0, k_pos0, stream) catch |err| switch (err) {
                error.UnsupportedQuantMode => return error.OperationNotSupported,
                else => |e| return e,
            };
            errdefer out_mlx.deinit();

            try storeMlxOutput(out, out_mlx);
            return;
        }
    }
    return error.DeviceNotSupported;
}
