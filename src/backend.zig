//! Backend - 计算后端、底层执行器与硬件加速基础设施
//! 统一管理数据并行执行器 (Executor)、系统级 BLAS (Accelerate / 回退 GEMM)、Apple Silicon MLX 与 NVIDIA CUDA 硬件加速。

const std = @import("std");

pub const activation = @import("backend/activation.zig");
pub const GeluApproximation = activation.GeluApproximation;

pub const executor = @import("backend/executor.zig");
pub const Executor = executor.Executor;

pub const cblas = @import("backend/cblas.zig");

pub const device = @import("backend/device.zig");
pub const Device = device.Device;
pub const DeviceType = device.DeviceType;
pub const DeviceArray = device.DeviceArray;
pub const DeviceKvCache = device.DeviceKvCache;
pub const KvCache = device.KvCache;
pub const BackendType = device.BackendType;
pub const isAvailable = device.isAvailable;
pub const isTypeAvailable = device.isTypeAvailable;
pub const defaultDevice = device.defaultDevice;
pub const defaultGpu = device.defaultGpu;
pub const activeBackend = device.activeBackend;

pub const buffer = @import("backend/buffer.zig");
pub const DeviceBuffer = buffer.DeviceBuffer;
pub const BufferHandle = buffer.BufferHandle;
pub const BufferError = buffer.BufferError;

pub const kernels = @import("backend/kernels.zig");
pub const KernelError = kernels.KernelError;
pub const KernelDType = kernels.KernelDType;
pub const GemmConfig = kernels.GemmConfig;
pub const RmsNormConfig = kernels.RmsNormConfig;
pub const RopeConfig = kernels.RopeConfig;
pub const SoftmaxConfig = kernels.SoftmaxConfig;
pub const ElementwiseConfig = kernels.ElementwiseConfig;
pub const GeluConfig = kernels.GeluConfig;
pub const SoftcapConfig = kernels.SoftcapConfig;
pub const EmbeddingConfig = kernels.EmbeddingConfig;

pub const mlx = @import("backend/mlx.zig");
pub const cuda = @import("backend/cuda.zig");
pub const CudaKvCache = cuda.CudaKvCache;
pub const cuda_kernels = @import("backend/cuda_kernels.zig");

pub const offload = @import("backend/offload.zig");
pub const GpuKvCache = offload.GpuKvCache;

pub const vec_math = @import("backend/vec_math.zig");

test {
    std.testing.refAllDecls(@This());
    _ = @import("backend/device.zig");
    _ = @import("backend/buffer.zig");
    _ = @import("backend/kernels.zig");
    _ = @import("backend/activation.zig");
    _ = @import("backend/executor.zig");
    _ = @import("backend/cblas.zig");
    _ = @import("backend/mlx.zig");
    _ = @import("backend/cuda.zig");
    _ = @import("backend/cuda_kernels.zig");
    _ = @import("backend/offload.zig");
    _ = @import("backend/vec_math.zig");
    _ = @import("backend/tests.zig");
}
