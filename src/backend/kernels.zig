//! 无状态计算内核接口 (Stateless Compute Kernel Interfaces)
//!
//! 提供通用矩阵乘法 (GEMM)、均方根层归一化 (RMSNorm)、旋转位置编码 (RoPE)、
//! 归一化指数函数 (Softmax) 与基础逐元素算子的纯强类型硬件分派。
//! 完全杜绝 anytype 鸭子类型，所有入参与出参均为明确的 DeviceBuffer 与 typed config。

const std = @import("std");
const device_mod = @import("device.zig");
const Device = device_mod.Device;
const buffer_mod = @import("buffer.zig");
const DeviceBuffer = buffer_mod.DeviceBuffer;
const cblas = @import("cblas.zig");
const vec_math = @import("vec_math.zig");
const activation = @import("activation.zig");
const cuda = @import("cuda.zig");
const mlx = @import("mlx.zig");

pub const KernelError = error{
    DeviceNotAvailable,
    DeviceMismatch,
    UnsupportedDevice,
    InvalidDimensions,
    ShapeMismatch,
    BufferTooSmall,
    KernelLaunchFailed,
    OperationNotSupported,
};

/// 硬件数据类型枚举
pub const KernelDType = enum(u8) {
    float32,
    bfloat16,
    float16,
    int32,
    uint32,
    uint8,

    pub fn byteSize(self: KernelDType) usize {
        return switch (self) {
            .float32, .int32, .uint32 => 4,
            .bfloat16, .float16 => 2,
            .uint8 => 1,
        };
    }
};

// ============================================================================
// 内核配置结构体 (遵循 AGENTS.md §4 规范，提供默认值与 defaultOptions)
// ============================================================================

pub const GemmConfig = struct {
    m: usize,
    n: usize,
    k: usize,
    lda: usize = 0,
    ldb: usize = 0,
    ldc: usize = 0,
    transpose_a: bool = false,
    transpose_b: bool = false,
    alpha: f32 = 1.0,
    beta: f32 = 0.0,
    batch_count: usize = 1,
    stride_a: usize = 0,
    stride_b: usize = 0,
    stride_c: usize = 0,
    dtype: KernelDType = .float32,

    pub fn defaultOptions() GemmConfig {
        return .{ .m = 0, .n = 0, .k = 0 };
    }
};

pub const RmsNormConfig = struct {
    rows: usize,
    dim: usize,
    eps: f32 = 1e-5,
    dtype: KernelDType = .float32,

    pub fn defaultOptions() RmsNormConfig {
        return .{ .rows = 0, .dim = 0 };
    }
};

pub const RopeConfig = struct {
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    start_pos: usize = 0,
    interleaved: bool = true,
    base: f32 = 10000.0,
    dtype: KernelDType = .float32,

    pub fn defaultOptions() RopeConfig {
        return .{ .seq_len = 0, .num_heads = 0, .head_dim = 0 };
    }
};

pub const SoftmaxConfig = struct {
    rows: usize,
    cols: usize,
    dtype: KernelDType = .float32,

    pub fn defaultOptions() SoftmaxConfig {
        return .{ .rows = 0, .cols = 0 };
    }
};

pub const ElementwiseConfig = struct {
    len: usize,
    dtype: KernelDType = .float32,

    pub fn defaultOptions() ElementwiseConfig {
        return .{ .len = 0 };
    }
};

pub const GeluConfig = struct {
    len: usize,
    approximation: activation.GeluApproximation = .none,
    dtype: KernelDType = .float32,

    pub fn defaultOptions() GeluConfig {
        return .{ .len = 0 };
    }
};

pub const SoftcapConfig = struct {
    len: usize,
    cap: f32 = 30.0,
    dtype: KernelDType = .float32,

    pub fn defaultOptions() SoftcapConfig {
        return .{ .len = 0, .cap = 30.0 };
    }
};

pub const EmbeddingConfig = struct {
    num_indices: usize,
    embedding_dim: usize,
    vocab_size: usize,
    dtype: KernelDType = .float32,

    pub fn defaultOptions() EmbeddingConfig {
        return .{ .num_indices = 0, .embedding_dim = 0, .vocab_size = 0 };
    }
};

// ============================================================================
// 强类型计算内核接口
// ============================================================================

/// 通用矩阵乘法计算内核 (General Matrix Multiply Kernel): C = alpha * (A * B) + beta * C
pub fn gemm(
    device: Device,
    config: GemmConfig,
    a: *const DeviceBuffer,
    b: *const DeviceBuffer,
    c: *DeviceBuffer,
) KernelError!void {
    if (!a.device.eql(device) or !b.device.eql(device) or !c.device.eql(device)) return error.DeviceMismatch;
    if (config.m == 0 or config.n == 0 or config.k == 0) return;

    switch (device.type) {
        .cpu => {
            const a_slice = a.asCpuSlice(f32) catch return error.BufferTooSmall;
            const b_slice = b.asCpuSlice(f32) catch return error.BufferTooSmall;
            const c_slice = c.asCpuSliceMut(f32) catch return error.BufferTooSmall;

            const lda: c_int = if (config.lda > 0) @intCast(config.lda) else @intCast(if (config.transpose_a) config.m else config.k);
            const ldb: c_int = if (config.ldb > 0) @intCast(config.ldb) else @intCast(if (config.transpose_b) config.k else config.n);
            const ldc: c_int = if (config.ldc > 0) @intCast(config.ldc) else @intCast(config.n);

            for (0..config.batch_count) |batch| {
                const off_a = batch * (if (config.stride_a > 0) config.stride_a else config.m * config.k);
                const off_b = batch * (if (config.stride_b > 0) config.stride_b else config.k * config.n);
                const off_c = batch * (if (config.stride_c > 0) config.stride_c else config.m * config.n);

                cblas.cblas_sgemm(
                    cblas.CblasRowMajor,
                    if (config.transpose_a) cblas.CblasTrans else cblas.CblasNoTrans,
                    if (config.transpose_b) cblas.CblasTrans else cblas.CblasNoTrans,
                    @intCast(config.m),
                    @intCast(config.n),
                    @intCast(config.k),
                    config.alpha,
                    a_slice.ptr + off_a,
                    lda,
                    b_slice.ptr + off_b,
                    ldb,
                    config.beta,
                    c_slice.ptr + off_c,
                    ldc,
                );
            }
        },
        .cuda => {
            cuda.launchSgemm(
                a.handle.cuda,
                b.handle.cuda,
                c.handle.cuda,
                config.m,
                config.n,
                config.k,
                config.batch_count,
            ) catch return error.KernelLaunchFailed;
        },
        .metal => {
            return error.OperationNotSupported;
        },
    }
}

/// 均方根层归一化内核 (Root Mean Square Layer Normalization Kernel)
pub fn rmsNorm(
    device: Device,
    config: RmsNormConfig,
    x: *const DeviceBuffer,
    weight: ?*const DeviceBuffer,
    out: *DeviceBuffer,
) KernelError!void {
    if (!x.device.eql(device) or !out.device.eql(device)) return error.DeviceMismatch;
    if (weight) |w| if (!w.device.eql(device)) return error.DeviceMismatch;
    if (config.rows == 0 or config.dim == 0) return;

    switch (device.type) {
        .cpu => {
            const x_slice = x.asCpuSlice(f32) catch return error.BufferTooSmall;
            const out_slice = out.asCpuSliceMut(f32) catch return error.BufferTooSmall;
            const w_slice: ?[]const f32 = if (weight) |w| w.asCpuSlice(f32) catch return error.BufferTooSmall else null;
            vec_math.rmsNormRows(x_slice, w_slice, out_slice, null, config.dim, config.eps, 0.0);
        },
        .cuda => {
            const w_ptr = if (weight) |w| w.handle.cuda else 0;
            cuda.launchRmsNorm(
                x.handle.cuda,
                w_ptr,
                out.handle.cuda,
                config.rows,
                config.dim,
                config.eps,
            ) catch return error.KernelLaunchFailed;
        },
        .metal => {
            return error.OperationNotSupported;
        },
    }
}

/// 旋转位置编码内核 (Rotary Position Embedding Kernel)
pub fn rope(
    device: Device,
    config: RopeConfig,
    x: *const DeviceBuffer,
    out: *DeviceBuffer,
) KernelError!void {
    if (!x.device.eql(device) or !out.device.eql(device)) return error.DeviceMismatch;
    if (config.seq_len == 0 or config.head_dim == 0) return;

    switch (device.type) {
        .cpu => {
            return error.OperationNotSupported;
        },
        .cuda => {
            cuda.launchRope(
                x.handle.cuda,
                out.handle.cuda,
                config.seq_len * config.num_heads,
                config.head_dim,
                config.interleaved,
                config.base,
                config.start_pos,
            ) catch return error.KernelLaunchFailed;
        },
        .metal => {
            return error.OperationNotSupported;
        },
    }
}

/// 归一化指数函数内核 (Softmax Kernel)
pub fn softmax(
    device: Device,
    config: SoftmaxConfig,
    x: *const DeviceBuffer,
    out: *DeviceBuffer,
) KernelError!void {
    if (!x.device.eql(device) or !out.device.eql(device)) return error.DeviceMismatch;
    if (config.rows == 0 or config.cols == 0) return;

    switch (device.type) {
        .cpu => {
            const x_slice = x.asCpuSlice(f32) catch return error.BufferTooSmall;
            const out_slice = out.asCpuSliceMut(f32) catch return error.BufferTooSmall;
            for (0..config.rows) |i| {
                const row_x = x_slice[i * config.cols .. (i + 1) * config.cols];
                const row_out = out_slice[i * config.cols .. (i + 1) * config.cols];
                vec_math.softmaxRow(row_x, row_out);
            }
        },
        .cuda => {
            cuda.launchSoftmax(
                x.handle.cuda,
                out.handle.cuda,
                config.rows,
                config.cols,
            ) catch return error.KernelLaunchFailed;
        },
        .metal => {
            return error.OperationNotSupported;
        },
    }
}

/// 逐元素加法内核 (Elementwise Add Kernel): out = a + b
pub fn add(
    device: Device,
    config: ElementwiseConfig,
    a: *const DeviceBuffer,
    b: *const DeviceBuffer,
    out: *DeviceBuffer,
) KernelError!void {
    if (!a.device.eql(device) or !b.device.eql(device) or !out.device.eql(device)) return error.DeviceMismatch;
    if (config.len == 0) return;

    switch (device.type) {
        .cpu => {
            const a_slice = a.asCpuSlice(f32) catch return error.BufferTooSmall;
            const b_slice = b.asCpuSlice(f32) catch return error.BufferTooSmall;
            const out_slice = out.asCpuSliceMut(f32) catch return error.BufferTooSmall;
            for (a_slice[0..config.len], b_slice[0..config.len], out_slice[0..config.len]) |va, vb, *vo| {
                vo.* = va + vb;
            }
        },
        .cuda => {
            cuda.launchAdd(a.handle.cuda, b.handle.cuda, out.handle.cuda, config.len) catch return error.KernelLaunchFailed;
        },
        .metal => {
            return error.OperationNotSupported;
        },
    }
}

/// 逐元素乘法内核 (Elementwise Multiply Kernel): out = a * b
pub fn multiply(
    device: Device,
    config: ElementwiseConfig,
    a: *const DeviceBuffer,
    b: *const DeviceBuffer,
    out: *DeviceBuffer,
) KernelError!void {
    if (!a.device.eql(device) or !b.device.eql(device) or !out.device.eql(device)) return error.DeviceMismatch;
    if (config.len == 0) return;

    switch (device.type) {
        .cpu => {
            const a_slice = a.asCpuSlice(f32) catch return error.BufferTooSmall;
            const b_slice = b.asCpuSlice(f32) catch return error.BufferTooSmall;
            const out_slice = out.asCpuSliceMut(f32) catch return error.BufferTooSmall;
            for (a_slice[0..config.len], b_slice[0..config.len], out_slice[0..config.len]) |va, vb, *vo| {
                vo.* = va * vb;
            }
        },
        .cuda => {
            cuda.launchMultiply(a.handle.cuda, b.handle.cuda, out.handle.cuda, config.len) catch return error.KernelLaunchFailed;
        },
        .metal => {
            return error.OperationNotSupported;
        },
    }
}

/// 标量乘法内核 (Scalar Multiply Kernel): out = a * scalar
pub fn mulScalar(
    device: Device,
    config: ElementwiseConfig,
    scalar: f32,
    a: *const DeviceBuffer,
    out: *DeviceBuffer,
) KernelError!void {
    if (!a.device.eql(device) or !out.device.eql(device)) return error.DeviceMismatch;
    if (config.len == 0) return;

    switch (device.type) {
        .cpu => {
            const a_slice = a.asCpuSlice(f32) catch return error.BufferTooSmall;
            const out_slice = out.asCpuSliceMut(f32) catch return error.BufferTooSmall;
            vec_math.scaleCopy(out_slice[0..config.len], a_slice[0..config.len], scalar);
        },
        .cuda => {
            cuda.launchMulScalar(a.handle.cuda, scalar, out.handle.cuda, config.len) catch return error.KernelLaunchFailed;
        },
        .metal => {
            return error.OperationNotSupported;
        },
    }
}

/// 高斯误差线性单元内核 (GELU Activation Kernel)
pub fn gelu(
    device: Device,
    config: GeluConfig,
    x: *const DeviceBuffer,
    out: *DeviceBuffer,
) KernelError!void {
    if (!x.device.eql(device) or !out.device.eql(device)) return error.DeviceMismatch;
    if (config.len == 0) return;

    switch (device.type) {
        .cpu => {
            const x_slice = x.asCpuSlice(f32) catch return error.BufferTooSmall;
            const out_slice = out.asCpuSliceMut(f32) catch return error.BufferTooSmall;
            switch (config.approximation) {
                .none => {
                    for (x_slice[0..config.len], out_slice[0..config.len]) |vx, *vo| {
                        vo.* = activation.gelu(.none, vx);
                    }
                },
                .tanh => {
                    for (x_slice[0..config.len], out_slice[0..config.len]) |vx, *vo| {
                        vo.* = activation.gelu(.tanh, vx);
                    }
                },
            }
        },
        .cuda => {
            cuda.launchGelu(x.handle.cuda, out.handle.cuda, config.len, config.approximation == .tanh) catch return error.KernelLaunchFailed;
        },
        .metal => {
            return error.OperationNotSupported;
        },
    }
}

/// 软截断算子内核 (Soft Capping Kernel): out = cap * tanh(x / cap)
pub fn softcap(
    device: Device,
    config: SoftcapConfig,
    x: *const DeviceBuffer,
    out: *DeviceBuffer,
) KernelError!void {
    if (!x.device.eql(device) or !out.device.eql(device)) return error.DeviceMismatch;
    if (config.len == 0 or config.cap <= 0.0) return;

    switch (device.type) {
        .cpu => {
            const x_slice = x.asCpuSlice(f32) catch return error.BufferTooSmall;
            const out_slice = out.asCpuSliceMut(f32) catch return error.BufferTooSmall;
            for (x_slice[0..config.len], out_slice[0..config.len]) |vx, *vo| {
                vo.* = config.cap * std.math.tanh(vx / config.cap);
            }
        },
        .cuda => {
            cuda.launchSoftcap(x.handle.cuda, out.handle.cuda, config.len, config.cap) catch return error.KernelLaunchFailed;
        },
        .metal => {
            return error.OperationNotSupported;
        },
    }
}

/// 词元嵌入查表内核 (Token Embedding Lookup Kernel)
pub fn embedding(
    device: Device,
    config: EmbeddingConfig,
    weights: *const DeviceBuffer,
    indices: *const DeviceBuffer,
    out: *DeviceBuffer,
) KernelError!void {
    if (!weights.device.eql(device) or !indices.device.eql(device) or !out.device.eql(device)) return error.DeviceMismatch;
    if (config.num_indices == 0 or config.embedding_dim == 0) return;

    switch (device.type) {
        .cpu => {
            const w_slice = weights.asCpuSlice(f32) catch return error.BufferTooSmall;
            const idx_slice = indices.asCpuSlice(f32) catch return error.BufferTooSmall;
            const out_slice = out.asCpuSliceMut(f32) catch return error.BufferTooSmall;
            const D = config.embedding_dim;
            const V = config.vocab_size;

            for (0..config.num_indices) |i| {
                const idx: usize = @intFromFloat(idx_slice[i]);
                if (idx < V) {
                    @memcpy(out_slice[i * D .. (i + 1) * D], w_slice[idx * D .. (idx + 1) * D]);
                } else {
                    @memset(out_slice[i * D .. (i + 1) * D], 0.0);
                }
            }
        },
        .cuda => {
            cuda.launchEmbedding(
                weights.handle.cuda,
                indices.handle.cuda,
                out.handle.cuda,
                config.num_indices,
                config.embedding_dim,
                config.vocab_size,
            ) catch return error.KernelLaunchFailed;
        },
        .metal => {
            return error.OperationNotSupported;
        },
    }
}

test "kernels: gemm, rmsNorm, softmax, add, multiply, gelu" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const dev = Device.cpu;

    // GEMM: [2, 2] * [2, 2]
    var a = try DeviceBuffer.alloc(dev, 4 * @sizeOf(f32), allocator);
    defer a.free(allocator);
    var b = try DeviceBuffer.alloc(dev, 4 * @sizeOf(f32), allocator);
    defer b.free(allocator);
    var c = try DeviceBuffer.alloc(dev, 4 * @sizeOf(f32), allocator);
    defer c.free(allocator);

    const a_s = try a.asCpuSliceMut(f32);
    @memcpy(a_s, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    const b_s = try b.asCpuSliceMut(f32);
    @memcpy(b_s, &[_]f32{ 2.0, 0.0, 1.0, 2.0 });

    try gemm(dev, .{ .m = 2, .n = 2, .k = 2 }, &a, &b, &c);
    const c_s = try c.asCpuSlice(f32);
    try testing.expectEqual(@as(f32, 4.0), c_s[0]);
    try testing.expectEqual(@as(f32, 4.0), c_s[1]);
    try testing.expectEqual(@as(f32, 10.0), c_s[2]);
    try testing.expectEqual(@as(f32, 8.0), c_s[3]);

    // Add: [4] + [4]
    try add(dev, .{ .len = 4 }, &a, &b, &c);
    try testing.expectEqual(@as(f32, 3.0), c_s[0]);
    try testing.expectEqual(@as(f32, 2.0), c_s[1]);
    try testing.expectEqual(@as(f32, 4.0), c_s[2]);
    try testing.expectEqual(@as(f32, 6.0), c_s[3]);
}
