//! 逐元素激活函数的标量数学定义：CPU 张量内核、自动微分反向传播与 MLX 图形处理器路径共用的唯一公式来源。

const std = @import("std");

extern fn erff(x: f32) f32;

/// 高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 的近似方式 (与 PyTorch `approximate` 参数同名同义)
pub const GeluApproximation = enum {
    /// 精确形式：GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    none,
    /// 双曲正切 (Hyperbolic Tangent, Tanh) 近似形式：GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))，
    /// 即 Hugging Face `gelu_pytorch_tanh` 与 mlx-lm `nn.gelu_approx`
    tanh,

    pub const default: GeluApproximation = .none;
    pub fn defaultOptions() GeluApproximation {
        return .none;
    }
};

/// 双曲正切近似中的三次项系数
pub const gelu_tanh_cubic_coeff: f32 = 0.044715;
/// 双曲正切近似中的缩放系数 sqrt(2 / pi)
pub const gelu_tanh_scale: f32 = 0.7978845608028654;
/// 精确形式中的缩放系数 1 / sqrt(2)
pub const gelu_erf_scale: f32 = 0.7071067811865476;

/// 按指定近似方式计算标量 GELU(x)
pub fn gelu(comptime approximation: GeluApproximation, x: f32) f32 {
    return switch (approximation) {
        .none => 0.5 * x * (1.0 + erff(x * gelu_erf_scale)),
        .tanh => 0.5 * x * (1.0 + std.math.tanh(gelu_tanh_scale * (x + gelu_tanh_cubic_coeff * x * x * x))),
    };
}

/// 按指定近似方式计算标量导数 dGELU(x)/dx
/// - 精确形式：Phi(x) + x * phi(x)，其中 Phi 为标准正态累积分布函数，phi 为标准正态概率密度函数
/// - 双曲正切近似：0.5 * (1 + t) + 0.5 * x * (1 - t^2) * sqrt(2 / pi) * (1 + 3 * 0.044715 * x^2)，其中 t = tanh(u)
pub fn geluDerivative(comptime approximation: GeluApproximation, x: f32) f32 {
    switch (approximation) {
        .none => {
            const inv_sqrt_2pi: f32 = 0.3989422804014327;
            const cdf = 0.5 * (1.0 + erff(x * gelu_erf_scale));
            const pdf = inv_sqrt_2pi * @exp(-0.5 * x * x);
            return cdf + x * pdf;
        },
        .tanh => {
            const x2 = x * x;
            const t = std.math.tanh(gelu_tanh_scale * (x + gelu_tanh_cubic_coeff * x2 * x));
            const du_dx = gelu_tanh_scale * (1.0 + 3.0 * gelu_tanh_cubic_coeff * x2);
            return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx;
        },
    }
}

test "gelu scalar forms match reference values" {
    // 精确形式参考值：GELU(1) = 0.8413447，GELU(-1) = -0.1586553
    try std.testing.expectApproxEqAbs(@as(f32, 0.8413447), gelu(.none, 1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1586553), gelu(.none, -1.0), 1e-6);
    // 双曲正切近似参考值 (PyTorch `gelu(approximate="tanh")`)：GELU(1) = 0.8411920，GELU(-1) = -0.1588080
    try std.testing.expectApproxEqAbs(@as(f32, 0.8411920), gelu(.tanh, 1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1588080), gelu(.tanh, -1.0), 1e-6);
    inline for (.{ GeluApproximation.none, GeluApproximation.tanh }) |approx| {
        try std.testing.expectEqual(@as(f32, 0.0), gelu(approx, 0.0));
        try std.testing.expectApproxEqAbs(@as(f32, 0.5), geluDerivative(approx, 0.0), 1e-7);
        // 大幅值输入：正半轴趋于恒等，负半轴趋于 0 (含三次项溢出为无穷的情形)
        try std.testing.expectApproxEqAbs(@as(f32, 20.0), gelu(approx, 20.0), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), gelu(approx, -20.0), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 1e20), gelu(approx, 1e20), 1e14);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), gelu(approx, -1e20), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), geluDerivative(approx, 20.0), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), geluDerivative(approx, -20.0), 1e-5);
    }
}
