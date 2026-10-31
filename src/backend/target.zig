//! 编译期目标架构与硬件加速链检测 (Comptime Target Architecture & Acceleration Toolchain)
//!
//! 在编译期内省 CPU 架构、操作系统特征与 SIMD 指令集扩展（NEON、AVX2、AVX-512、AMX、FMA），
//! 自动计算最优单精度向量宽度，并直接判定编译期使能的加速工具链（Accelerate、MLX、CUDA）。

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

/// 编译期目标处理器与操作系统特征 (Comptime Target Architecture)
pub const TargetArch = struct {
    pub const arch = builtin.target.cpu.arch;
    pub const os = builtin.target.os.tag;
    pub const is_darwin = os == .macos or os == .ios or os == .tvos or os == .watchos;
    pub const is_linux = os == .linux;
    pub const is_windows = os == .windows;

    /// 是否具备 ARM64 NEON 向量指令集
    pub const has_neon = arch.isAARCH64() and std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon);

    /// 是否具备 x86 Advanced Vector Extensions 2 (AVX2)
    pub const has_avx2 = arch.isX86() and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);

    /// 是否具备 x86 AVX-512 Foundation (AVX-512F)
    pub const has_avx512 = arch.isX86() and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f);

    /// 是否具备硬件融合乘加 (Fused Multiply-Add, FMA)
    pub const has_fma = switch (arch) {
        .x86, .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
        .aarch64, .aarch64_be => has_neon,
        else => false,
    };

    /// 编译期自适应计算最优单精度浮点 SIMD 向量长度 (Lanes)
    pub const optimal_vector_len: comptime_int = if (has_avx512)
        16
    else if (has_avx2)
        8
    else if (has_neon)
        4
    else
        4;

    /// 目标硬件原生单精度 SIMD 向量类型
    pub const SimdVector = @Vector(optimal_vector_len, f32);
};

/// 编译期激活的硬件加速工具链 (Comptime Active Toolchains)
pub const ActiveToolchains = struct {
    /// Apple Accelerate 框架 (单精度 GEMM 与 vDSP 向量运算)
    pub const has_accelerate: bool = TargetArch.is_darwin;

    /// Apple Silicon MLX 统一显存加速链
    pub const has_mlx: bool = build_options.enable_mlx and TargetArch.is_darwin and TargetArch.arch == .aarch64;

    /// NVIDIA CUDA 驱动 API 运行时加速链 (Linux / Windows)
    pub const has_cuda: bool = TargetArch.is_linux or TargetArch.is_windows;
};

test "target architecture and toolchain inspection" {
    try std.testing.expect(TargetArch.optimal_vector_len >= 4);
    if (TargetArch.is_darwin and TargetArch.arch == .aarch64) {
        try std.testing.expect(TargetArch.has_neon);
        try std.testing.expect(ActiveToolchains.has_accelerate);
    }
}
