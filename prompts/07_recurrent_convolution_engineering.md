# Milestone 1: 循环卷积工程落地与乒乓双缓冲零动态分配体系 (Recurrent Convolution & Ping-Pong Buffering)

## 📌 阶段基本信息
- **阶段编号**：Milestone 1 (Recurrent Conv & Ping-Pong Double Buffering)
- **优先级**：P0
- **状态**：✅ 已完成 (Completed)
- **代码分支**：`feat/recurrent-conv-research`
- **所有权文件**：
  - `src/models/recurrent_conv.zig` (新建核心模型文件)
  - `src/models.zig` (公共导出注册)
  - `prompts/07_recurrent_convolution_engineering.md` (本计划与验收档案)

---

## 🎯 目标与背景 (Context & Motivation)

基于前期理论综述报告（`doc/recurrent_convolution_research_report.md`）中的数学动力学与系统优化分析，本里程碑在 ZNN 深度学习框架中落地循环/权重共享卷积（Recurrent / Tied Convolution）的核心工程组件：
1. **显式内存控制与 $O(1)$ 空间零动态分配**：针对推理阶段，利用预分配的 `RecurrentWorkspace`（持有双缓冲 `buf_a`、`buf_b`、静态输入投影 `buf_u`、中间卷积缓冲 `buf_conv` 与展平展开内存 `col_buf`），通过奇偶时间步交替读写（Ping-Pong Double Buffering），彻底消除循环迭代期间的任何堆内存申请（`alloc` / `malloc` 次数严格为 0）；
2. **训练模式计算图时间反向传播 (BPTT)**：构建动态计算图展开逻辑，跨 $T$ 步共享卷积核 $W_r$，利用 ZNN 自动微分引擎的天然累加特性（$\beta = 1.0$ / `+=`）实现精确的 BPTT 梯度反向累积；
3. **自适应计算深度早停 (ACT)**：基于维度归一化残差均方根范数（RMSE）判定 $\delta_t = \frac{\|h_t - h_{t-1}\|_2}{\sqrt{C \cdot H \cdot W}} < \epsilon$ 与发散守卫，实现样本粒度的动态收敛退出；
4. **端到端分类网络 (`RecurrentConvNet`)**：集成 Stem 卷积、GroupNorm 归一化、循环卷积块、全局自适应平均池化（GAP）与线性分类头，提供与前向训练模式数值等价的零动态分配推理入口；
5. **规范契约完全合规**：严格遵循 ZNN `AGENTS.md`：
   - 依赖单向分层（Layer 4 仅依赖 Layer $\le 3$），100% 通过 `zig build check-layers`；
   - 配置结构体支持默认匿名结构字面量 `.{}` 与 `defaultOptions()`；
   - 模块暴露 `initDefault()` 便捷初始化函数；
   - 完善的单元测试覆盖前向、反向 BPTT 梯度、乒乓零分配及 GPA 内存无泄漏验证。

---

## 🛠️ 模块架构与接口设计 (Architectural Blueprint & Interfaces)

### 1. `RecurrentConvOptions` (配置选项结构体)
```zig
pub const RecurrentConvOptions = struct {
    kernel_size: usize = 3,
    stride: usize = 1,
    padding: usize = 1,
    groups: usize = 1,
    num_norm_groups: usize = 8,
    t_min: usize = 2,
    t_max: usize = 8,
    epsilon: f32 = 1e-3,
    alpha_damping: f32 = 0.8,
    use_residual: bool = true,

    pub const default: RecurrentConvOptions = .{};
    pub fn defaultOptions() RecurrentConvOptions {
        return .{};
    }
};
```

### 2. `RecurrentConvResult` (推理结果结构体)
```zig
pub const RecurrentConvResult = struct {
    output: *Tensor,
    steps_taken: usize,
    converged: bool,
    final_residual: f32,
};
```

### 3. `RecurrentWorkspace` (零动态分配专用工作区)
```zig
pub const RecurrentWorkspace = struct {
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    buf_a: *Tensor,
    buf_b: *Tensor,
    buf_u: *Tensor,
    buf_conv: *Tensor,
    col_buf: []f32,

    pub fn init(allocator: std.mem.Allocator, batch_size: usize, channels: usize, height: usize, width: usize, kernel_size: usize) !RecurrentWorkspace;
    pub fn deinit(self: *RecurrentWorkspace, allocator: std.mem.Allocator) void;
};
```

### 4. `RecurrentConvBlock` (核心循环卷积块)
- **数据流拓扑**：
  - 训练模式：$u = \text{conv\_f}(x), h_0 = u$；在 $t=1..T$ 中，$z = \text{conv\_r}(h_{t-1}) + u$（若 `use_residual`），$a = \text{ReLU}(z)$，$n = \text{norm}(a)$，$h_t = (1-\alpha)h_{t-1} + \alpha n$；返回 $h_T$。
  - 推理模式：利用 `FixedBufferAllocator.init(std.mem.sliceAsBytes(ws.col_buf))` 消除 `conv2dInto` 的堆分配，交替读写 `buf_a` 与 `buf_b`，计算 RMSE 范数 $\delta_t$，触发早停。
- **公共接口**：
  - `pub fn init(allocator: std.mem.Allocator, channels: usize, options: RecurrentConvOptions) !RecurrentConvBlock`
  - `pub fn initDefault(allocator: std.mem.Allocator, channels: usize) !RecurrentConvBlock`
  - `pub fn forward(self: *const RecurrentConvBlock, graph: *autodiff.Graph, x: *Tensor) !*Tensor`
  - `pub fn forwardO1(self: *const RecurrentConvBlock, ws: *RecurrentWorkspace, x: *Tensor) !RecurrentConvResult`
  - `pub fn resetParameters(self: *RecurrentConvBlock, random: std.Random, options: nn.InitOptions) void`
  - `pub fn zeroGrad(self: *RecurrentConvBlock) void`

### 5. `RecurrentConvNetConfig` & `RecurrentConvNet` (完整图像分类网络)
- **网络拓扑**：
  - Stem: `Conv2D(in_channels -> hidden_channels, 3x3, s=1, p=1)` -> `GroupNorm(num_norm_groups, hidden_channels)` -> `ReLU`
  - Recurrent Block: `RecurrentConvBlock(hidden_channels, options)`
  - Pool: `AdaptiveAvgPool2D(1, 1)`
  - Head: `Linear(hidden_channels -> num_classes)`
- **公共接口**：
  - `pub fn init(allocator: std.mem.Allocator, config: RecurrentConvNetConfig) !RecurrentConvNet`
  - `pub fn initDefault(allocator: std.mem.Allocator) !RecurrentConvNet`
  - `pub fn forward(self: *const RecurrentConvNet, graph: *autodiff.Graph, x: *Tensor) !*Tensor`
  - `pub fn forwardO1(self: *const RecurrentConvNet, ws: *RecurrentWorkspace, x: *Tensor) !RecurrentConvResult`
  - `pub fn resetParameters(self: *RecurrentConvNet, random: std.Random, options: nn.InitOptions) void`
  - `pub fn zeroGrad(self: *RecurrentConvNet) void`
  - `pub fn deinit(self: *RecurrentConvNet, allocator: std.mem.Allocator) void`

---

## 🧪 验收标准与单元测试设计 (Acceptance Criteria & Test Matrix)

- [x] **AC 1: 配置与默认值规范 (AGENTS.md §4)**
  - `RecurrentConvOptions` 与 `RecurrentConvNetConfig` 支持 `.{}`、`defaultOptions()` 与 `.default` 常量，默认字段完全一致。
  - 对应测试：`test "RecurrentConvOptions and default options compliance"` 通过。
- [x] **AC 2: 工作区初始化与安全释放 (GPA 零内存泄漏)**
  - `RecurrentWorkspace.init` 正确分配 4 个持久化缓冲区张量与连续展平切片 `col_buf`；
  - `ws.deinit` 干净释放所有缓冲区与张量句柄，通过 `std.testing.allocator` 校验 0 泄漏。
  - 对应测试：`test "RecurrentWorkspace init and deinit clean memory"` 通过。
- [x] **AC 3: 乒乓双缓冲零动态分配 (Zero Heap Allocation in `forwardO1`)**
  - 在固定分配器包装与预分配工作区支持下，`forwardO1` 循环迭代 $T$ 步期间对系统堆（GPA / OS heap）的申请调用严格为 0；
  - 验证双缓冲在偶数/奇数时间步的切换正确性与残差 RMSE 单调性。
  - 对应测试：`test "RecurrentConvBlock forwardO1 zero heap allocation ping-pong"` 通过。
- [x] **AC 4: 计算图 BPTT 反向梯度多步累加**
  - 展开 $T=4$ 步前向传播，通过标量损失调用 `graph.backward`；
  - 验证共享权重 `conv_r.weight.grad` 正确累积来自 4 个不同时间步的偏导数；
  - 验证 `zeroGrad()` 能够完整清空累加梯度。
  - 对应测试：`test "RecurrentConvBlock forward training BPTT and gradient accumulation"` 通过。
- [x] **AC 5: 端到端分类网络与工作区复用及数值等价性验证**
  - 构建 `RecurrentConvNet`，在同一预分配 `RecurrentWorkspace` 上连续执行多次 `forwardO1` 推理，验证形状恢复逻辑保证无 `ShapeMismatch`，输出分类 Logits 形状一致（`[B, num_classes]`）、数值全部有限（非 NaN/Inf）且与计算图模式完全等价（绝对差小于 `1e-4`）。
  - 对应测试：`test "RecurrentConvNet forward and forwardO1 equivalence"` 经 3 轮连续推理循环通过。
- [x] **AC 6: 残差 RMSE 与早退出 NaN 弹性韧性防伪早停**
  - 在输入含 NaN 异常值时，`computeResidualRmse` 严格返回 NaN，`forwardO1` 守卫判定退出且置 `converged = false`，杜绝 `@max(0.0, NaN)` 引发假收敛。
  - 对应测试：`test "computeResidualRmse and forwardO1 NaN resilience"` 通过。
- [x] **AC 7: 架构分层校验与全量测试套件通过**
  - 执行 `zig build check-layers`，107 个文件 100% 单向分层无违规；
  - 执行 `zig build test`，全仓 25/25 步骤测试全部通过。

---

## 📅 执行状态与实测记录 (Execution Record & Verification Evidence)

- **迭代 2 整改要点 (Milestone 1 Iteration 2)**：
  1. 修复 `RecurrentConvNet.forwardO1` 工作区形状状态污染：在入口处完整恢复 `buf_conv`、`buf_a`、`buf_b`、`buf_u` 的原始 4D 形状与连续步长，保障连续多批次推理无 `ShapeMismatch`；增加分类头缓冲区容量检查 `b_size * num_classes <= buf_conv.data.len`。
  2. 修复 `computeResidualRmse` 与 `forwardO1` 的 NaN 处理：当局部 RMSE 为 NaN 时直接返回 NaN；`forwardO1` 显式拦截 NaN 并判定未收敛退出，防止假收敛。
  3. 扩充单元测试：连续 3 次复用 `RecurrentWorkspace` 推理测试，以及全 NaN 输入测试。
- **分层校验**：
  `zig build check-layers` 输出：`check-layers: 107 files, all package dependencies are one-way, facades enforced, zero duck-typing` (Exit 0)。
- **全量测试**：
  `zig build test` 输出：`Build Summary: 25/25 steps succeeded` (Exit 0)。
- **交付结论**：
  Milestone 1 Iteration 2 整改完成，零动态分配乒乓双缓冲、工作区复用安全性、BPTT 自动微分求导与 NaN 数值弹性全部 100% 达成验收标准。

---

# Milestone 2 & 3: 四组对齐基准评测、自适应早停、鲁棒性探索与实验报告交付

## 📌 阶段基本信息
- **阶段编号**：Milestones 2 & 3 (Aligned Benchmarks, Dynamic ACT, Input Robustness & Experiment Report)
- **优先级**：P0
- **状态**：✅ 已完成 (Completed & Verified)
- **代码分支**：`feat/recurrent-conv-research`
- **所有权文件**：
  - `examples/recurrent_benchmark.zig` (新建综合评测与探索套件)
  - `build.zig` (注册 `run-recurrent` 可执行目标与测试套件 `recurrent_bench_tests`)
  - `doc/recurrent_convolution_experiment_report.md` (交付完整的结构化实验报告)
  - `prompts/07_recurrent_convolution_engineering.md` (本阶段执行与验收记录)

---

## 🎯 核心目标与交付范围 (Mission & Deliverables)

1. **4 组严格对齐基准网络实现（FashionMNIST $1\times 28\times 28$, 10 分类）**：
   - **B1 (Shallow Baseline)**：3 层 Conv2D ($1 \to 32 \to 40 \to 48$), GroupNorm (8 组) + ReLU, GAP + Linear ($48 \to 10$)。总参数量：**29,938**。
   - **B2 (Deep CNN Baseline)**：8 层 Conv2D ($1 \to 64$, 随后 7 层 $64 \to 64$), GroupNorm (8 组) + ReLU, GAP + Linear ($64 \to 10$)。总参数量：**260,810**。
   - **B3 (Proposed Recurrent CNN)**：Stem Conv2D ($1 \to 64$, GroupNorm 8 组) + 绑定 Recurrent Conv ($64 \to 64$, GroupNorm 8 组, 递归展开 $T=8$ 步, 凸阻尼 $\alpha=0.8$), GAP + Linear ($64 \to 10$)。总参数量：**38,474**。
   - **B4 (Equal-Param Wide Baseline)**：2 层 Conv2D ($1 \to 64$ Stem, 随后 $64 \to 64$), GroupNorm (8 组) + ReLU, GAP + Linear ($64 \to 10$)。总参数量：**38,474** — 与 B3 保持严格 bit-exact 位级完全一致！
2. **多维度基准度量采集体系**：
   - Top-1 测试集分类准确率 (%)；
   - 单样本前向理论 FLOPs 计算与验证（$2 \cdot B \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot C_{\text{in}} \cdot K^2$）；
   - 单样本推理前向延迟（ms/img，基于 `zig_ml.bench.getTimeNs()` 高精度时钟）；
   - 显存/内存峰值度量：验证 B3 Ping-Pong 推理在 $O(1)$ 空间上的严格零动态分配（Zero Allocations）。
3. **消融实验体系 (Ablation Suites)**：
   - 递归展开步数消融：$T \in \{1, 2, 4, 8, 16\}$；
   - 凸阻尼平滑系数消融：$\alpha \in \{0.5, 0.8, 1.0\}$；
   - 归一化机制消融：GroupNorm vs 无归一化 / LayerNorm。
4. **动态自适应早停探索 (Dynamic Early-Exit ACT)**：
   - 尺度无关残差均方根（Dimension-Normalized RMSE）准则扫描：$\epsilon \in \{10^{-4}, 5\times 10^{-4}, 10^{-3}, 2\times 10^{-3}, 5\times 10^{-3}\}$；
   - 统计平均计算步数 $\bar{T}$、算力节约百分比与 Top-1 精度保留率。
5. **输入鲁棒性与自去噪/抗遮挡探索**：
   - 加性高斯白噪声 (AWGN)：$\sigma \in \{0.1, 0.2, 0.3\}$；
   - 椒盐脉冲噪声：$p \in \{0.05, 0.10, 0.20\}$；
   - 中心矩形遮挡 (Center Cutout)：$8 \times 8$ 与 $12 \times 12$；
   - 评估 B3 在各时间步 $T \in \{1, 2, 3, 4, 5, 6, 7, 8\}$ 下的特征恢复与准确率跃迁。
6. **最终实验报告交付**：
   - 交付结构完整、符合 KaTeX 与 CommonMark 排版规范的科学研究报告至 `doc/recurrent_convolution_experiment_report.md`。

---

## 🧪 验收标准与测试矩阵 (Acceptance Criteria & Results)

- [x] **AC 1: 4 组基线参数量严格位对齐 (Bit-Exact Alignment)**
  - B1 总参数量严格为 29,938；
  - B2 总参数量严格为 260,810；
  - B3 总参数量严格为 38,474；
  - B4 总参数量严格为 38,474；
  - 启动阶段执行 `std.testing.expectEqual(b3_params, b4_params)` 绝对位对齐断言通过（delta = 0 bits）。
  - 对应单元测试：`test "4 baselines exact parameter count alignment"` 通过。
- [x] **AC 2: 4 组基线前向计算与张量维度一致性**
  - 验证 B1, B2, B3, B4 在输入张量 `[B, 784]` 下均输出标准 Logits `[B, 10]`，且各层特征维度严格符合规格表。
  - 对应单元测试：`test "4 baselines forward pass and shape consistency"` 通过。
- [x] **AC 3: B3 forwardO1 乒乓推理零堆内存申请**
  - 在固定分配器与预分配工作区支持下，`B3.forwardO1` 循环迭代 $T$ 步期间无任何动态堆申请，输出与计算图模式完全等价。
  - 对应单元测试：`test "B3 forwardO1 ping-pong zero heap allocation"` 通过。
- [x] **AC 4: 扰动算子与理论 FLOPs 计算验证**
  - 高斯噪声、椒盐噪声与中心遮挡算子数值范围严格处于 $[0.0, 1.0]$，中心清零区域定位精确；
  - 理论 FLOPs 公式严格匹配 Table 1 规格。
  - 对应单元测试：`test "Perturbation functions (Gaussian, Salt-and-Pepper, Cutout)"` 与 `test "Theoretical FLOPs calculation consistency"` 通过。
- [x] **AC 5: 注册 build.zig 并通过 check-layers 与全量测试**
  - `build.zig` 注册 `recurrent_benchmark` 目标（运行步 `run-recurrent`，测试名 `recurrent_bench_tests`）；
  - `zig build check-layers` 107 个文件无架构分层违规；
  - `zig build test` 全仓测试 100% 顺利通过。
- [x] **AC 6: 完整交付 Markdown 实验研究报告**
  - `doc/recurrent_convolution_experiment_report.md` 包含执行摘要、4 基线对比表、参数对齐数学证明、消融实验、动态早停与鲁棒性分析。
