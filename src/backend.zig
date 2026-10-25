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
pub const Device = mlx.DeviceType;
pub const DeviceType = mlx.DeviceType;

pub const mlx_blocks = @import("backend/mlx_blocks.zig");
pub const Rope = mlx_blocks.Rope;
pub const projectHeads = mlx_blocks.projectHeads;
pub const gatedMlp = mlx_blocks.gatedMlp;
pub const residualScale = mlx_blocks.residualScale;
pub const finalHidden = mlx_blocks.finalHidden;

pub const logits_processor = @import("backend/logits_processor.zig");
pub const LogitsProcessor = logits_processor.LogitsProcessor;
pub const LogitsProcessorOptions = logits_processor.LogitsProcessorOptions;
pub const softcapLogits = logits_processor.softcapLogits;
pub const penalizeDevice = logits_processor.penalizeDevice;
pub const deviceCandidates = logits_processor.deviceCandidates;
pub const selectToken = logits_processor.selectToken;
pub const selectTokenFromHidden = logits_processor.selectTokenFromHidden;
pub const DeviceCandidates = logits_processor.DeviceCandidates;
pub const HostCandidates = logits_processor.HostCandidates;

pub const offload = @import("backend/offload.zig");

pub const rope = @import("backend/rope.zig");

pub const safetensors = @import("backend/safetensors.zig");
pub const SafetensorsArchive = safetensors.SafetensorsArchive;
pub const TensorMeta = safetensors.TensorMeta;

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
    _ = @import("backend/safetensors.zig");
    _ = @import("backend/rope.zig");
    _ = @import("backend/sampling.zig");
    _ = @import("backend/vec_math.zig");
    _ = @import("backend/tests.zig");
}
