# Milestone 11: 高级张量索引、跨步切片视图与离散累加体系 (Advanced Indexing, Strided Views & Scatter Operations)

> **Target Version**: `v0.5.0`  
> **Status**: 🔥 **准备就绪 / 待启动 (Queued / Ready for Execution)**  
> **Priority**: **P1 (核心深度学习与离散求导底座)**  
> **Dependencies**: Milestone 2 (张量统一存储与 `*Into` 算子), Milestone 3 (执行图重放与分配器掌控), Milestone 4 (`nn.Module` 协议), Milestone 10 (QR 与线性代数基础)  
> **Target Packages**:  
> - `src/tensor/` (Layer 1: `indexing.zig` [新建], `core.zig`, `reductions.zig`, `tests.zig`)  
> - `src/autodiff/` (Layer 2: `types.zig`, `op.zig`, `graph.zig`, `backward_math.zig`, `tests.zig`)  
> - `src/root.zig` (Layer 7: 顶层门面导出)  
> **Enforcement Policy**: Strict refactoring with zero backward-compatibility baggage (`AGENTS.md` §3)

---

## 1. 任务背景与核心动机 (Background & Technical Rationale)

在现代深度学习与数值计算中，张量元素的操作主要分为两类：
1. **规则结构化计算**（Dense Regular Computations）：如矩阵乘法（GEMM）、逐元素运算（Element-wise）、连续张量切片（Slicing）与空间卷积（Conv2D）。此类操作具有固定的内存步长与规则的循环上界；
2. **离散非结构化索引与散布**（Discrete Irregular Indexing & Scattering）：如按照动态概率或离散决策提取目标特征、图结构中的邻居节点聚合、稀疏注意力掩码以及强化学习动作价值提取。

### 1.1 为什么当前 ZNN 架构急需通用 `gather` 与 `scatterAdd`？
目前 `znn` 仅支持基于连续/正步长的切片视图（`SliceRange`）以及词表行查表（`embeddingLookup`）。当面对以下典型场景时，代码库出现明显的功能断层：
- **大语言模型变长终止与动态生成**：在批量自回归解码中，不同样本的终止符（`<eos>`）位置各不相同。若要从序列输出 `[batch_size, seq_len, hidden_dim]` 中提取各样本最后一个有效 Token 的隐藏状态向量，PyTorch 标准写法为 `hidden.gather(1, end_indices.unsqueeze(-1).expand(-1, -1, hidden_dim))`。在 `znn` 中目前只能通过复杂的 CPU 循环逐行切片，打断了计算图与流水线；
- **强化学习动作值提取 (Q-Learning / SAC)**：在价值网络输出所有动作价值 $Q(s) \in \mathbb{R}^{B \times A}$ 后，需要根据离散动作索引 $a \in \mathbb{R}^{B \times 1}$ 提取对应的预测值 $Q(s, a)$，即 $y = \text{gather}(Q(s), 1, a)$；
- **图神经网络 (GNN) 与稀疏分发**：图拓扑节点更新公式为 $h_v^{(l+1)} = \sum_{u \in \mathcal{N}(v)} W h_u^{(l)}$。其核心计算正是通过 `scatterAdd` 将源节点变换特征沿着目标节点索引进行散布累加；
- **混合专家网络 (MoE) 动态路由**：专家路由门控决定 Token 分发至哪个专家处理，反向传播时专家输出的梯度必须通过 `scatterAdd` 累加回原始 Token 梯度缓冲区。

### 1.2 自动微分 (Autograd) 的对偶反向闭环
在数学上，`gather` 与 `scatterAdd` 构成严格的伴随对偶算子（Adjoint Dual Operators）：
- 设前向计算为轴向元素抽取：
  $$y = \text{gather}(x, d, I) \iff y_{i_0, \dots, k, \dots, i_{R-1}} = x_{i_0, \dots, I(i_0, \dots, k, \dots, i_{R-1}), \dots, i_{R-1}}$$
- 其标量损失 $L$ 对输入张量 $x$ 的梯度满足：
  $$\frac{\partial L}{\partial x_{i_0, \dots, j, \dots, i_{R-1}}} = \sum_{k: I(i_0, \dots, k, \dots, i_{R-1}) = j} \frac{\partial L}{\partial y_{i_0, \dots, k, \dots, i_{R-1}}}$$
- 即：
  $$\nabla_x \text{gather}(x, d, I) = \text{scatterAdd}(\mathbf{0}, d, I, \nabla_y)$$
**结论**：缺乏通用 `scatterAdd`，不仅前向图无法表达稀疏累加，更直接导致 `gather` 无法在 Autograd 计算图中完成梯度反向传播。

### 1.3 跨步切片步长语义与无符号 Shape 内存模型 (Strided Slicing & Memory Model)
在当前 `znn` 架构中，`Tensor.strides` 采用无符号静态数组 `Shape.dims: [8]usize` 进行紧凑存储与高效步长迭代（如 `RowIter`、`gatherStrided` 与广播内核）。为保证全库底层内存安全，切片采用清晰划分的内存模型：
- **正步长切片 (`step > 0`)**：通过重算 `shape`、`strides` 与物理内存 `offset`，保持原生零拷贝视图（Zero-Copy Strided View，`is_view = true`），时间复杂度为 $O(1)$ 且无堆分配；
- **负步长切片 (`step < 0`)**：由于无符号 `usize` 步长无法在不破坏现有类型体系的前提下表示逆向物理地址跨度（直接传递负数会触发 64 位无符号整数溢出 Panic，且 `self.data[offset..]` 无法逆向寻址），当检测到 `step < 0` 时，切片算子统一采用**显式逆序物化拷贝**（Reverse Materialization Copy，`is_view = false`），安全分配紧凑正向内存并按逆序复制元素（对标 NumPy `arr[::-1]` 语义）。此举既保证了对数组翻转与反向切片的高层 API 完备对齐，又杜绝了修改底层 `Shape` 结构体带来的全库破坏性风险。

---

## 2. 接口设计契约与架构规范 (Interface Contracts)

本里程碑遵循 `AGENTS.md` 规范：
- 严格遵循 8 层单向依赖拓扑；
- 贯彻 Layer 1 零分配目标写入 `*Into` 算子设计；
- 提供带 sensible defaults 的配置结构体。

### 2.1 Layer 1: 张量高级索引算子 (`src/tensor/indexing.zig`)

新建 `src/tensor/indexing.zig`，由 `src/tensor/core.zig` 门面重导出：

```zig
const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("core.zig").Tensor;
const Shape = @import("shape.zig").Shape;
const TensorError = @import("core.zig").TensorError;

/// 通用轴向索引收集：沿着指定维度 dim 按照 index 张量中的坐标抽取元素
/// 要求：
/// 1. self.shape.rank == index.shape.rank
/// 2. 除了 dim 维度之外，self 与 index 在所有其他维度上的大小必须相同
/// 3. 输出形状与 index 的形状严格一致
pub fn gather(
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
    allocator: Allocator,
) !*Tensor;

/// 零堆分配目标写入 gather 算子
pub fn gatherInto(
    dest: *Tensor,
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
) !void;

/// 通用轴向离散覆写 (Scatter)：将 src 中的元素按照 index 散布覆写至 dest
pub fn scatter(
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
    src: *Tensor,
    allocator: Allocator,
) !*Tensor;

/// 零堆分配目标写入覆写 scatter 算子 (5 参数：dest 为目标缓冲区，首先复制 base，随后按 index 写入 src)
pub fn scatterInto(
    dest: *Tensor,
    base: *const Tensor,
    dim: usize,
    index: *const Tensor,
    src: *Tensor,
) !void;

/// 通用轴向离散累加 (Scatter-Add)：将 src 中的元素按照 index 散布并累加至 dest
/// 核心数学逻辑：dest[...][index[...]][...] += src[...][...][...]
pub fn scatterAdd(
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
    src: *Tensor,
    allocator: Allocator,
) !*Tensor;

/// 零堆分配原位离散累加算子 (4 参数：dest 原位累加，满足梯度无分配累加与无别名规则)
pub fn scatterAddInto(
    dest: *Tensor,
    dim: usize,
    index: *const Tensor,
    src: *Tensor,
) !void;

/// 一维子张量索引切片提取：沿指定 dim 提取 index (1D) 指定的超平面
/// 输出形状：原 shape 中 dim 维替换为 index.shape.dims[0]
pub fn indexSelect(
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
    allocator: Allocator,
) !*Tensor;

pub fn indexSelectInto(
    dest: *Tensor,
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
) !void;

/// 一维子张量索引切片累加 (Index-Add)：将 src 的切片按照一维 index 坐标累加至 dest 沿 dim 维度的对应超平面
/// 核心数学逻辑：dest[..., index[i], ...] += src[..., i, ...]
/// 与 indexSelect 互为伴随对偶，是 OpType.IndexSelect 零分配反向传播的基础原语 (4 参数原位累加)
pub fn indexAdd(
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
    src: *Tensor,
    allocator: Allocator,
) !*Tensor;

pub fn indexAddInto(
    dest: *Tensor,
    dim: usize,
    index: *const Tensor,
    src: *Tensor,
) !void;

/// 布尔掩码一维压缩提取：按照布尔掩码 mask 提取所有非零/真值元素，返回 1D 连续扁平张量
pub fn maskedSelect(
    self: *Tensor,
    mask: *const Tensor,
    allocator: Allocator,
) !*Tensor;
```

### 2.1.1 统一索引张量类型系统与兼容辅助封装 (Unified Index Typing Contract)
- **核心标准类型**：Layer 1 核心索引算子与 Layer 2 Autograd 计算图**统一以 `*const Tensor` 为主标准类型 (Primary Index Typing)**。索引在物理上存储于 `backend.DeviceBuffer` 统一内存中，算子内核通过现有的 `indexScalarToUsize` 转换函数安全校验非负整数与值域上界（越界严格返回 `error.IndexOutOfBounds`），与自动微分计算图及梯度缓存完全同构；
- **泛型与原生切片辅助封装**：为兼顾纯 CPU 离线计算与无图数据处理流水线，在 `src/tensor/indexing.zig` 中配套提供泛型/切片辅助函数（如 `gatherSlice`、`gatherGeneric` 或 `anytype` 入口封装），直接接受 `*const GenericTensor(usize)`、`*const GenericTensor(bool)` 及 `[]const usize`，内部无缝转接或轻量索引，消除数据结构断层。


### 2.2 跨步切片负步长升级 (`src/tensor/reductions.zig`)

重构 `SliceRange` 结构体，支持带符号步长与边界自动反转：

```zig
pub const SliceRange = struct {
    start: ?isize = null,
    end: ?isize = null,
    step: isize = 1,

    pub const default: SliceRange = .{};
    pub fn defaultOptions() SliceRange {
        return .{};
    }
};
```

**切片步长与视图计算规范 (Slicing Stride & Memory Model Contract)**：
张量步长底层存储为 `Tensor.strides: Shape`（内部字段为无符号数组 `dims: [8]usize`）。
- 当 `step > 0` 时 —— **零拷贝跨步视图 (`is_view = true`)**：
  - 默认 `actual_start = start orelse 0`；
  - 默认 `actual_end = end orelse @intCast(dim_len)`；
  - 切片长度 $L = \max(0, \lfloor \frac{\text{actual\_end} - \text{actual\_start} + \text{step} - 1}{\text{step}} \rfloor)$；
  - 新步长 $\text{new\_stride} = \text{orig\_stride} \times @as(usize, @intCast(\text{step}))$；
  - 起始偏移增加 $\text{actual\_start} \times \text{orig\_stride}$；
  - 标记 `is_view = true`，享受 $O(1)$ 时间与零堆内存分配；
- 当 `step < 0` 时 —— **显式逆序物化拷贝 (`is_view = false`)**：
  - 默认 `actual_start = start orelse (@intCast(dim_len) - 1)`；
  - 默认 `actual_end = end orelse -1`；
  - 切片长度 $L = \max(0, \lfloor \frac{\text{actual\_start} - \text{actual\_end} + (-\text{step}) - 1}{-\text{step}} \rfloor)$；
  - **无符号步长约束防御**：鉴于 `Shape.dims: [8]usize` 无法容纳负物理跨度且下游内核（`RowIter`、广播、`validateOutput`）假定正向指针偏移，算子为切片分配大小为 $L$ 的新物理内存存储，按逆序步长将元素复制到新连续缓冲区中；
  - 标记 `is_view = false`，保证与 NumPy `arr[::-1]` 语义 100% 对齐的同时彻底消除无符号溢出风险。

### 2.3 Layer 2: 自动求导计算图集成 (`src/autodiff/`)

#### 扩展算子类型 (`src/autodiff/types.zig`)
```zig
pub const OpType = enum {
    // ... 既有算子 ...
    Gather,
    Scatter,
    ScatterAdd,
    IndexSelect,
};
```

#### 计算图前向接口暴露 (`src/autodiff/graph.zig`)
```zig
pub fn gather(self: *Graph, x: *Tensor, dim: usize, index: *const Tensor) !*Tensor;
pub fn scatter(self: *Graph, x: *Tensor, dim: usize, index: *const Tensor, src: *Tensor) !*Tensor;
pub fn scatterAdd(self: *Graph, x: *Tensor, dim: usize, index: *const Tensor, src: *Tensor) !*Tensor;
pub fn indexSelect(self: *Graph, x: *Tensor, dim: usize, index: *const Tensor) !*Tensor;
```

#### 反向自动微分数学闭环 (`src/autodiff/backward_math.zig`)
1. **`OpType.Gather` 求导**：
   - 前向：$y = \text{gather}(x, \text{dim}, \text{index})$
   - 输入节点：`x = op.inputs[0]`，索引：`index = op.inputs[1]`，输出梯度：$\nabla_y = \text{output.grad}$；
   - 伴随梯度累加：`x.grad` 累加 $\text{scatterAdd}(\mathbf{0}, \text{dim}, \text{index}, \nabla_y)$（零分配路径直接调用 `scatterAddInto(x.grad, dim, index, output.grad)`）。
2. **`OpType.Scatter` 求导 (覆写型散布求导与梯度置零)**：
   - 前向：$y = \text{scatter}(x, \text{dim}, \text{index}, \text{src})$
   - 输入节点：`x = op.inputs[0]`，`index = op.inputs[1]`，`src = op.inputs[2]`，输出梯度：$\nabla_y = \text{output.grad}$；
   - **求导数学闭环**：因为 $src$ 覆写了 $x$ 在 $index$ 处的原值，原 $x$ 在覆写位置对输出与损失的偏导数为 0（梯度被屏蔽置零），未被覆写的位置梯度则全额透传 $\nabla_y$；而 $src$ 获得对应位置的抽取梯度：
     $$\nabla_x = \text{scatter}(\nabla_y, \text{dim}, \text{index}, \mathbf{0})$$
     $$\nabla_{\text{src}} = \text{gather}(\nabla_y, \text{dim}, \text{index})$$
   - 零堆分配执行路径：
     - 源输入梯度：调用 `gatherInto(src.grad, output.grad, dim, index)` 累加至 `src.grad`；
     - 基底输入梯度：在将 $\nabla_y$ 累加至 `x.grad` 时，将被覆写位置的梯度置零（或将 $\nabla_y$ 拷贝后对 `index` 位置清零再累加至 `x.grad`），保证梯度传播严格正确。
3. **`OpType.ScatterAdd` 求导**：
   - 前向：$y = \text{scatterAdd}(x, \text{dim}, \text{index}, \text{src})$
   - 输入节点：`x = op.inputs[0]`，`src = op.inputs[2]`；
   - 累加到基底梯度：`x.grad += output.grad`（直通累加）；
   - 累加到源梯度：`src.grad += gather(output.grad, dim, index)`（或通过 `gatherInto` 累加）。
4. **`OpType.IndexSelect` 求导**：
   - 前向：$y = \text{indexSelect}(x, \text{dim}, \text{index})$
   - 累加到输入梯度：使用伴随对偶算子 `indexAddInto(x.grad, dim, index, output.grad)` 将输出切片梯度散布累加回原维度对应超平面。

---

## 3. 分阶段执行路线图 (Phased Implementation Roadmap)

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Layer 1 基础切片与索引内核实现                         │
│ - 升级 reductions.zig SliceRange 支持带符号步长与逆序物化拷贝   │
│ - 新建 indexing.zig 实现 gather, scatter, scatterAdd, select,   │
│   indexAdd 及对应 Into 算子 (4 参数 scatterAddInto/5 参数 scatterInto) │
│ - 在 core.zig 暴露统一 *const Tensor 门面与泛型辅助包装         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│ Phase 2: Layer 1 单元测试与边界鲁棒性加固                       │
│ - 跨步正步长零拷贝视图与负步长逆序物化拷贝正确性验证            │
│ - 1D~4D 密集与稀疏下标 gather / scatter / scatterAdd / indexAdd  │
│ - 下标越界防御断言 (error.IndexOutOfBounds) 审计               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│ Phase 3: Layer 2 Autograd 计算图节点接入与反向梯度求导          │
│ - types.zig 扩展 OpType 枚举 (Gather, Scatter, ScatterAdd, etc.)│
│ - backward_math.zig 实现 Gather, Scatter (梯度置零),           │
│   ScatterAdd 与 IndexSelect (indexAddInto) 伴随对偶梯度传导     │
│ - graph.zig 封装声明式 API，无缝融入批次 Arena 内存生命周期     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│ Phase 4: 全量闭环验证与发布 (Layer 0–7 门禁)                    │
│ - gradcheck 数值梯度自动对比验证 (100% 对齐)                    │
│ - zig build check-layers (91+ 源码文件 0 越层违规)              │
│ - zig build test (0 错误，0 内存泄漏)                           │
│ - 18 个经典模型结构图导出验证 (run-book-models)                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 破坏性改动声明 (Breaking Change Policy)

- **完全无破坏性改动 (Non-Breaking / Purely Additive)**：
  - 本里程碑引入的所有接口（`gather`, `scatter`, `scatterAdd`, `indexSelect`, `indexAdd`, `maskedSelect`）均为向后纯新增算子；
  - `SliceRange` 的 `step` 扩展兼容默认值 `1`，对既有正步长切片代码 100% 保持源码级透明兼容；`step < 0` 采用显式逆序物化拷贝（`is_view = false`），安全保护底层无符号 `Shape.dims: [8]usize` 免受溢出风险；
  - 现有的 `embeddingLookupInto` 与 `gatherMatMulInto` 保持现有高效路径不变，未来可无缝重构为基于通用 `gatherInto` 的特化实现。

---

## 5. 验收标准与客观验证命令 (Acceptance Criteria & Verification)

### 5.1 验收清单 (Checklist)
- [ ] **切片负步长与逆序物化**：`tensor.slice(&.{.{ .step = -1 }})` 在 `step < 0` 时安全产出逆序物化拷贝（`is_view == false`），正步长切片保持零拷贝视图（`is_view == true`），数值与 NumPy `arr[::-1]` 严格对齐，安全规避无符号 `Shape.dims: [8]usize` 整数溢出；
- [ ] **通用 Gather 正确性**：支持 1D、2D、3D、4D 张量沿任意指定 `dim` 收集元素，数值与 PyTorch `torch.gather` 严格一致；
- [ ] **离散 ScatterAdd 与 Scatter 覆写**：支持 4 参数 `scatterAddInto` 多重重叠下标累加与 5 参数 `scatterInto` 基础底模替换，数值与 PyTorch `torch.scatter_add` / `torch.scatter` 严格一致；
- [ ] **子张量切片与索引累加**：`indexSelect`、`indexAdd` / `indexAddInto` 与 `maskedSelect` 正确提取与累加超平面切片及布尔掩码压缩；
- [ ] **越界安全防御**：传入越界索引时，安全返回 `error.IndexOutOfBounds`，严禁触发任何 Segfault 或越界读写；
- [ ] **Autograd 伴随梯度验证**：`Graph.gather`、`Graph.scatter`（含被覆写梯度置零）、`Graph.scatterAdd` 与 `Graph.indexSelect`（通过 `indexAddInto` 累加）在反向传播时通过一阶有限差分数值梯度检验（`gradcheck` 误差 $< 10^{-4}$）；
- [ ] **内存安全**：在 Debug 模式与 ReleaseSafe 模式下，全量测试套件 GPA 报告 0 内存泄漏。

### 5.2 客观独立验证命令 (Verification Commands)

```bash
# 1. 验证 8 层架构依赖严格单向合规性 (Layer 0 -> Layer 7)
zig build check-layers

# 2. 运行全量单元测试与内存检测 (默认测试集)
zig build test

# 3. 运行纯 CPU 模式全量测试 (验证便携回退路径)
zig build test -Dmlx=false

# 4. 验证 18 个经典模型架构图 JSON 导出完整性
zig build run-book-models

# 5. 验证前端 Web 可视化工具单元测试
cd web && npm test
```
