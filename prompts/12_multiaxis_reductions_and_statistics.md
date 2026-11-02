# Milestone 12: 多轴联合归约、前缀扫描与进阶统计算子体系

## 🎯 任务背景与核心价值 (Context & Objective)

在当前 ZNN 架构中，大部分归约（Reductions）算子仅支持单轴 (`axis: ?usize`) 或全张量展平归约。
而在现代深度学习模型（如自注意力多头批次统计、Transformer 中对特定序列维与隐层维的同时归约、强化学习中的奖励折扣累加 `cumsum`、正则化中对任意通道的 `norm` 范数约束）中，多轴联合归约与前缀扫描（Cumulative Operations）是核心基础设施。

本 Milestone 目标是完整补齐对齐 PyTorch / NumPy 的进阶统计与归约体系：
1. 多轴切片联合归约：`sumAxes`, `meanAxes`, `varianceAxes`, `stdAxes`（支持 `axes: []const usize` 与 `keepdims: bool`）；
2. 累乘与前缀扫描：`prod`, `cumsum`, `cumprod`（含零分配目标变体 `*Into`）；
3. 通用矩阵与向量范数算子：`norm`（支持 $p=1, 2, \infty$ 及 Frobenius 矩阵范数）；
4. 统计极值与分位数：`median`, `quantile`（基于快速选择算法 QuickSelect）；
5. Autograd 计算图闭环：接入 `OpType.CumSum`, `OpType.Norm`, `OpType.Max`/`Min` 梯度路由及数值稳定的 `logSumExp`。

---

## 📐 接口契约与分层规范 (API Contracts)

### 1. Layer 1: 多轴联合归约与前缀扫描 (`src/tensor/reductions.zig`)

```zig
pub fn sumAxes(self: *const Tensor, axes: []const usize, keepdims: bool, allocator: Allocator) !*Tensor;
pub fn meanAxes(self: *const Tensor, axes: []const usize, keepdims: bool, allocator: Allocator) !*Tensor;
pub fn norm(self: *const Tensor, p: ?f32, dim: ?usize, keepdim: bool, allocator: Allocator) !*Tensor;
pub fn cumsumInto(dest: *Tensor, self: *const Tensor, dim: usize) !void;
pub fn cumprodInto(dest: *Tensor, self: *const Tensor, dim: usize) !void;
pub fn quantile(self: *const Tensor, q: f32, dim: ?usize, keepdim: bool, allocator: Allocator) !*Tensor;
```

### 2. Layer 2: Autograd 梯度传导与计算图算子 (`src/autodiff/`)

- `cumsum` 反向传播：$\nabla_x \text{cumsum}(x, d) = \text{cumsum}(\nabla_y, d)_{\text{reverse}}$ (逆序累加)；
- `norm` ($L_2$) 反向传播：$\nabla_x \|x\|_2 = \frac{x}{\|x\|_2} \odot \nabla_y$；
- `logSumExp` 反向传播：$\nabla_x \text{LSE}(x) = \text{softmax}(x) \odot \nabla_y$；
- 扩展 `src/autodiff/types.zig`、`op.zig`、`backward_math.zig` 与 `graph.zig`。

---

## 📋 分阶段推进计划与验收标准 (Acceptance Criteria)

- [ ] **Phase 1 (Layer 1 算子与核心算法)**：
  - 实现通用 `axes` 掩码坐标映射与多轴归约；
  - 实现带步长的高效前缀累加累乘 `cumsum` / `cumprod`；
  - 实现基于 IntroSelect / QuickSelect 的 `quantile` 与 `median`。
- [ ] **Phase 2 (单元测试与数值边界)**：
  - 在 `src/tensor/tests_reductions.zig` 中覆盖 1D~4D 边界测试（包含 `keepdims` 形状保真、极端分位数、负轴转换）。
- [ ] **Phase 3 (Autograd 计算图与梯度闭环)**：
  - 接入 `Graph.cumsum`, `Graph.norm`, `Graph.logSumExp`；
  - 编写有限差分数值梯度检验（`gradcheck`）。
- [ ] **Phase 4 (全量工程门禁)**：
  - 运行 `zig build check-layers` 保证 0 依赖违规；
  - 运行 `zig build test` 确保全量 440+ 测试通过且 0 内存泄漏；
  - 导出全部 18 个经典模型 JSON 架构图并通过 Web 前端测试。
