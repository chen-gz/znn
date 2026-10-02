# 🔬 ZNN vs. Modern NumPy (NumPy 2.x) 对比分析与补齐规划

本文档系统性地将 **ZNN (Zig Neural Network)** 的张量与数值计算底座与现代 **NumPy (NumPy 2.x)** 进行全面对比，诊断当前存在的关键功能短板，并制定逐步对齐与演进的工程规划。

---

## 🧭 1. 定位与设计哲学对比

| 维度 | 现代 NumPy (NumPy 2.x) | ZNN (当前项目) |
| :--- | :--- | :--- |
| **核心定位** | **通用多维数组科学计算底座**<br>(General-purpose N-Dimensional Array Computing) | **面向深度学习的张量与计算图微框架**<br>(Deep Learning & Autograd Framework akin to PyTorch/tinygrad) |
| **计算图与自动微分** | ❌ 无内置 Autograd，纯数值前向求值（需借助 JAX / Autograd） | ✅ **原生内置动态反向传播**（`autodiff.Graph`），张量天生具备 `grad` 与计算图拓扑 |
| **神经网络与训练原语** | ❌ 纯数组库，无神经网络层、损失函数或优化器 | ✅ **开箱即用深度学习算子库**（Transformer, GQA, MLA, KV-Cache, AdamW, DPO, GRPO） |
| **运行时依赖** | 依赖 Python 解释器、C-API、动态库运行时 | ✅ **Pure Zig 编写**，单二进制独立静态编译，零运行时依赖 |

---

## 📊 2. 核心功能对比矩阵与演进状态

| 功能大类 | 现代 NumPy (NumPy 2.x) | ZNN 当前已实现能力 (`v0.2.7`) | 剩余差距与补齐优先级 |
| :--- | :--- | :--- | :---: |
| **① 标量与数据类型系统 (Dtypes)** | `float16/32/64/128`, `int8~64`, `uint8~64`, `complex`, `bool`，支持类型提升与可扩展 DType API | `GenericTensor(T)` 支持 `f32`, `f64`, `bf16`, `i32`, `i64`, `usize`, `bool` 及 `to(DestT)` / `fromGeneric`；`StaticTensor(T, dims)` 支持编译期形状校验；Autograd `Tensor` 为 `f32` | **P1**（待扩展 `f16`/`bf16` 自动混合精度求导与 INT8/INT4 量化） |
| **② 切片与多维索引 (Indexing)** | 基础切片（零拷贝 View）、步长 (`arr[::-1]`)、省略号 (`...`)、花式索引 (`arr[[0, 2]]`)、布尔掩码 (`arr[mask]`) | 支持 `SliceRange { start, end, step }` 跨步零拷贝视图 (`slice`, `is_view`, `contiguous`)、条件选择 `where`、布尔/条件掩码填充 `maskedFill`、`split`、`concat`、`stack` | **P1**（待补充通用 `gather` / `scatterAdd` 与布尔掩码一维过滤提取） |
| **③ 归约与统计分析算子 (Reductions)** | `sum`, `prod`, `mean`, `std`, `var`, `min`, `max`, `ptp`, `quantile`，支持多轴 `axis=(0, 1)` 与 `keepdims` | 已实现沿指定轴或全局的 `sum`, `mean`, `variance`, `stdDev` (支持 `keepdims` 与 `ddof`)、`max`, `argmax`, `any`, `all`，且 `sum`/`mean` 已接入 Autograd 反向传播 | **P2**（待补充多轴元组 `axes: []const usize` 同时归约与 `quantile`/`median`） |
| **④ 条件选择与查找排序 (Searching & Sorting)** | `where(condition, x, y)`, `nonzero`, `sort`, `argsort`, `searchsorted`, `clip` | 已完整实现多维广播 `where`、`nonzero`、沿任意轴 `sort` / `argsort`（支持升序/降序）、`clip` / `clip_`、以及 LLM `sampleTopK` / `sampleTopP` | **已基本对齐**（仅缺 `searchsorted` / `unique`） |
| **⑤ 数组形态操纵 (Array Manipulation)** | `squeeze`, `expand_dims`, `tile`, `repeat`, `pad`, `flip`, `roll`, `stack`, `meshgrid` | 已实现 `reshape`, `transpose` / `transposeView`, `squeeze`, `unsqueeze`, `repeat`, `tile`, `concat`, `split`, `stack` | **P2**（待补充通用 `pad`、`flip`、`roll`、`meshgrid`） |
| **⑥ 通用函数系统 (ufuncs)** | 上百种标准 ufunc，统一支持广播及 `.reduce()`, `.accumulate()`, `.outer()`, `.at()` 方法 | 已实现广播四则运算 (`add`, `sub`, `mul`, `div`)、比较算子 (`eq`, `ne`, `gt`, `ge`, `lt`, `le`)、激活函数与初等超越函数 (`sqrt`, `exp`, `log`, `abs` 含 Autograd) | **P2**（待补充三角/反三角函数 `sin`/`cos` 与幂函数 `pow`） |
| **⑦ 现代随机数生成 (PRNG & Distributions)** | 现代 `Generator` 体系 (PCG64/Philox)，独立生成器对象，30+ 种概率分布与随机抽样 | 基于 `std.Random` 提供均匀分布 `fillUniform`/`randomUniform`、Box-Muller 正态分布 `fillNormal`/`randomNormal`、He/Xavier/LeCun 初始化与 Top-K/Top-P 采样 | **P2**（待补充泊松、指数、Dirichlet 分布与 `permutation`） |
| **⑧ 高级线性代数 (`linalg`)** | SVD, QR, Cholesky 分解, 特征值/特征向量 (`eig/eigh`), 矩阵求逆/伪逆, 行列式, 范数, `einsum` | 已实现 BLAS/SIMD `matmul`、`batchMatMul`、高斯-若尔当 `solveLinearSystem`、闭式岭回归 `solveRidgeAnalytical`、Householder `qr`、Jacobi `svd` 与实对称特征分解 `symeig` | **P1**（待补充正定矩阵 `cholesky`、显式求逆 `inv`、行列式 `det` 与 `einsum`） |
| **⑨ 专业科学计算模块** | 离散傅里叶变换 (`np.fft`)、多项式拟合 (`np.polynomial`)、文件读写 (`.npy`, `.npz`) | 专注于深度学习与统计学习：`SafeTensors` 模型序列化、优化器二进制检查点、`BinaryMmapDataset`、OLS/Ridge/Lasso/ElasticNet、K-Fold CV、t-SNE | **按需补充** |

---

## 🔍 3. 核心能力现状与剩余差距深度诊断

### 3.1 标量类型系统、泛型张量与编译期静态张量（Dtype & Static Shape System）
* **已实现能力**：
  * [`src/tensor/types.zig`](../src/tensor/types.zig) 提供了泛型 [`GenericTensor(comptime T: type)`](../src/tensor/types.zig) 及类型别名（`FloatTensor`, `DoubleTensor`, `IntTensor`, `LongTensor`, `BoolTensor`, `BFloat16Tensor`, `UsizeTensor`），原生支持 `bf16` 与 `f32` IEEE 754 双向转换以及跨类型转换 `to(DestT)` / `fromGeneric`。
  * [`src/tensor/static.zig`](../src/tensor/static.zig) 提供了编译期维度校验包装器 [`StaticTensor(comptime ElemT: type, comptime dims: anytype)`](../src/tensor/static.zig)，在编译阶段拦截维度不匹配的矩阵乘法、加法与重排。
* **剩余差距与演进目标**：
  * 动态自动微分引擎（[`Tensor`](../src/tensor/core.zig) 与 [`Graph`](../src/autodiff/graph.zig)）目前仍固定使用 `f32` 存储激活与梯度；下一步需支持 `bf16`/`f16` 混合精度前向/反向传播与 `GradScaler`。

### 3.2 进阶切片与花式索引机制（Slicing & Fancy Indexing）
* **已实现能力**：
  * **零拷贝跨步切片视图 (`SliceRange`)**：[`Tensor.slice`](../src/tensor/reductions.zig) 与 [`GenericTensor.slice`](../src/tensor/types.zig) 支持通过 `SliceRange { start, end, step }` 调整 `shape` 与 `strides` 实现 $O(1)$ 零拷贝视图，并由 `is_view` 与 `contiguous()` 保障内存安全；同时 [`Graph.slice`](../src/autodiff/graph.zig) 支持切片视图的反向传播梯度回传。
  * **条件掩码与选择**：已实现支持多维广播与 `BoolTensor` 掩码的 [`where`](../src/tensor/reductions.zig) 与 [`maskedFill`](../src/tensor/reductions.zig) / `maskedFill_`。
  * **词表嵌入索引**：[`Embedding.forward`](../src/nn/transformer.zig) 与 [`Graph.embedding`](../src/autodiff/graph_nn.zig) 支持直接传入整型切片或整型 `GenericTensor`。
* **剩余差距与演进目标**：
  * 补充沿任意维度的通用整数张量花式索引 `gather(dim, index_tensor)` 与 `scatterAdd(dim, index_tensor, src)`。

### 3.3 通用多轴归约算子（Multi-axis Reductions）
* **已实现能力**：
  * [`src/tensor/reductions.zig`](../src/tensor/reductions.zig) 已封装通用轴向与全局归约接口：`sum(axis, keepdims)`, `mean(axis, keepdims)`, `variance(axis, keepdims, ddof)`, `stdDev(axis, keepdims, ddof)`, `max`, `argmax`，且 `Graph.sum`、`Graph.mean`、`Graph.variance` 已支持计算图自动求导。
* **剩余差距与演进目标**：
  * 扩展 `axis` 参数以支持同时沿多个轴（如 `axes: []const usize`）进行联合归约，并补充 `min`、`argmin`、`prod`、`cumsum`。

### 3.4 高级线性代数与爱因斯坦求和 (`linalg` & `einsum`)
* **已实现能力**：
  * [`src/tensor/core.zig`](../src/tensor/core.zig) 已实现基于 Householder 变换的 QR 分解 (`Tensor.qr`)、奇异值分解 (`Tensor.svd`) 以及实对称矩阵 Jacobi 特征值分解 (`Tensor.symeig`)，并在 [`src/tensor/ops.zig`](../src/tensor/ops.zig) 中提供 `solveLinearSystem` 与 `solveRidgeAnalytical`。
* **剩余差距与演进目标**：
  * 引入对称正定矩阵 **Cholesky 分解 ($LL^T$)**、显式矩阵求逆 `inv(A)` 与行列式 `det(A)`。
  * 引入轻量级编译期/运行期 **Einsum 算子**（如 `einsum("bshd,bthd->bhst", q, k)`）。

---

## 🗺️ 4. 补齐演进规划 (Actionable Roadmap)

### Phase 1: 基础核心增强 (P0 - Quick Wins & Fundamentals)
- [x] **多轴归约函数库**：
  - [x] 实现通用 `sum(axis, keepdims)`, `mean(axis, keepdims)`, `variance/stdDev(axis, keepdims, ddof)`，支持连续与任意多维跨步张量及 Autograd 求导。
- [x] **条件选择与掩码填充**：
  - [x] 实现 `where(condition, x, y)` 支持多维广播对齐，与 `maskedFill(mask, value)` / `maskedFill_(mask, value)`（包含计算图安全防护）。
- [x] **多维形态补充算子**：
  - [x] 实现 `squeeze(axis)` (单轴/全轴为 1 维度压缩)、`unsqueeze(dim)` (新维度扩充)、`repeat`、`tile` 与 `stack`。

### Phase 2: 泛型张量、静态形状与切片视图 (P1 - Architectural Evolution)
- [x] **泛型张量与编译期静态张量系统**：
  - [x] 实现 `GenericTensor(comptime T: type)`，支持 `f32`, `f64`, `i32`, `i64`, `usize`, `bool`, `bf16` 与跨类型互转 `to(DestT)` / `fromGeneric`。
  - [x] 实现编译期形状校验张量 `StaticTensor(comptime ElemT: type, comptime dims: anytype)` (`src/tensor/static.zig`)。
  - [x] 实现 `bf16` 原生浮点结构体与 `f32` IEEE 754 互转及 `DType` 大小自省。
- [x] **跨步零拷贝切片 (Strided View Slicing)**：
  - [x] 引入 `SliceRange { start, end, step }`，支持对任意维度进行跨步切片且复用底层数据指针（`is_view` 内存生命周期保护），提供 `contiguous()` 紧凑拷贝。
- [x] **排序、检索与截断算子**：
  - [x] 实现 `sort(axis, ascending)`, `argsort(axis, ascending)`, `nonzero()`, `clip(min, max)` 与 `clip_`。

### Phase 3: 科学计算扩展与高性能线性代数 (P2 - Scientific Extensions)
- [ ] **现代随机数发生器体系**：
  - [ ] 抽象可配置种子的独立分布采样器，支持泊松分布、指数分布、随机排列 `permutation` 与无放回抽样。
- [ ] **高级线性代数库 (`src/tensor/`)**：
  - [x] 实现 `qr(A)` (`Tensor.qr`)、`svd(A)` (`Tensor.svd`) 与实对称特征值分解 `symeig(A)` (`Tensor.symeig`)。
  - [ ] 实现正定矩阵分解 `cholesky(A)`、矩阵求逆 `inv(A)` 与行列式 `det(A)`。
- [ ] **轻量级 Einsum 引擎**：
  - [ ] 支持类似 `einsum("bshd,bthd->bhst", q, k)` 的字符串解析与张量缩并执行。
