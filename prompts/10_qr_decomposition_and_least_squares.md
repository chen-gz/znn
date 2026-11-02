# 正交三角分解 (QR Decomposition) 与高精度最小二乘求解支持任务规划

## 1. 任务背景与目标 (Context & Motivation)
在数值线性代数与科学计算中，正交三角分解（QR Decomposition）是求解线性方程组、奇异值分解前置、特征值迭代算法（QR 算法）以及超定最小二乘拟合（Ordinary Least Squares, OLS）的核心支柱。

相比直接求解正规方程（Normal Equations, $X^T X w = X^T y$）导致的条件数平方灾难（$\kappa(X^T X) = \kappa(X)^2$ 与浮点精度丢失），QR 分解直接对数据矩阵进行正交变换（$X = Q R$），将最小二乘问题转化为上三角回代求解（$R w = Q^T y$），将条件数严格保持在 $\kappa(X)$，在病态数据拟合中具有极其优越的数值稳定性。

本项目目前在 `src/tensor/linalg.zig` 中拥有精简版 QR 分解原型，但缺少：
1. 完整的配置结构体 `QROptions`（遵循 `AGENTS.md` §4 规范，提供 `defaultOptions()` 与 `.{}` 零配置初始化）；
2. 完整模式（Complete QR, $Q \in \mathbb{R}^{M \times M}$, $R \in \mathbb{R}^{M \times N}$）支持；
3. 上三角线性系统回代求解器 `solveTriangularUpper`；
4. 基于 QR 分解的张量级通用最小二乘求解器 `qrSolve`；
5. 在回归层 `src/regression.zig` 中集成 `solveLeastSquaresQR`，提供高精度 OLS 求解入口；
6. 完善的张量与回归单测集，验证正交性、重建误差与数值稳定性。

---

## 2. 接口设计与契约 (Interface Design)

### 2.1 配置结构体与模式定义 (`src/tensor/linalg.zig`)
```zig
/// 正交三角分解计算模式
pub const QRMode = enum {
    reduced,  // 精简模式：M >= N 时 Q 为 [M, N], R 为 [N, N]；M < N 时 Q 为 [M, M], R 为 [M, N]
    complete, // 完整模式：对于任意 M x N，Q 为 [M, M] 完整正交矩阵，R 为 [M, N] 上梯形矩阵
};

/// 正交三角分解配置选项
pub const QROptions = struct {
    mode: QRMode = .reduced,

    pub fn defaultOptions() QROptions {
        return .{};
    }
};
```

### 2.2 函数签名与扩展
```zig
/// 带选项的正交三角分解 (QR Decomposition with Options)
pub fn qrWithOptions(self: Tensor, allocator: std.mem.Allocator, options: QROptions) !QRResult

/// 默认选项正交三角分解 (开箱即用)
pub fn qr(self: Tensor, allocator: std.mem.Allocator) !QRResult {
    return qrWithOptions(self, allocator, .{});
}

/// 上三角线性系统回代求解器 (Back Substitution: R * x = b)
pub fn solveTriangularUpper(R: Tensor, b: Tensor, allocator: std.mem.Allocator) !*Tensor

/// 基于 QR 分解的最小二乘求解器 (Solve A * x \approx b via QR)
pub fn qrSolve(A: Tensor, b: Tensor, allocator: std.mem.Allocator) !*Tensor
```

### 2.3 回归求解扩展 (`src/regression.zig`)
```zig
/// 利用正交三角分解求解普通最小二乘回归 (高数值稳定性)
pub fn solveLeastSquaresQR(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    n_samples: usize,
    n_features: usize,
) !ModelResult
```

---

## 3. 验收标准与实际验证结果 (Acceptance Criteria & Verification)
1. `zig build check-layers` 严格通过，91 个源码文件层级依赖严格单向，零越层引用；
2. `zig build test` 全量通过，张量算子、回归层与端到端测试均 0 内存泄漏；
3. 单测覆盖：
   - 验证精简模式与完整模式 QR 分解（方阵、高瘦矩阵与矮胖矩阵）；
   - 验证完整模式下 $Q^T Q = I_4$ 与 $Q Q^T = I_4$，上梯形 $R$ 底行全零，重建误差 $< 10^{-4}$；
   - 验证上三角回代求解器 `solveTriangularUpper` 与最小二乘求解器 `qrSolve` 准确性；
   - 验证 `solveLeastSquaresQR` 在多元回归参数估计与截距恢复中与解析真值完全对齐；
4. `CHANGELOG.md` 与根导出同步完成。
5. 状态：**✅ 已完成并通过全量测试验证 (Completed & Verified)**。
