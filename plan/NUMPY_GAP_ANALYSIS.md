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

## 📊 2. 核心功能对比矩阵与缺失分析

| 功能大类 | 现代 NumPy (NumPy 2.x) | ZNN 当前实现 | 补齐优先级 |
| :--- | :--- | :--- | :---: |
| **① 标量与数据类型系统 (Dtypes)** | `float16/32/64/128`, `int8~64`, `uint8~64`, `complex`, `bool`，支持类型提升 (Type Promotion) 与可扩展 DType API | 核心数据硬编码为单精度 `f32` (`data: []f32`) | **P0** (关键短板) |
| **② 切片与多维索引 (Indexing)** | 基础切片（零拷贝 View）、步长 (`arr[::-1]`)、省略号 (`...`)、花式索引 (`arr[[0, 2]]`)、布尔掩码 (`arr[mask]`) | 仅支持标量坐标查询 `get/set`、`split`、`concat` | **P0** (关键短板) |
| **③ 归约与统计分析算子 (Reductions)** | `sum`, `prod`, `mean`, `std`, `var`, `min`, `max`, `ptp`, `quantile`，支持多轴 `axis=(0, 1)` 与 `keepdims` | 仅支持单轴 `argmax` 与 `max`，均值/方差仅作为 Norm 算子私有内部逻辑 | **P0** (高频需求) |
| **④ 条件选择与查找排序 (Searching & Sorting)** | `where(condition, x, y)`, `nonzero`, `sort`, `argsort`, `searchsorted`, `clip` | 实现了 Top-K/Top-P 采样与梯度剪裁，缺少通用的条件选择与全量快速排序 | **P1** |
| **⑤ 数组形态操纵 (Array Manipulation)** | `squeeze`, `expand_dims`, `tile`, `repeat`, `pad`, `flip`, `roll`, `stack`, `meshgrid` | 具备 `reshape`, `transpose` (任意双轴交换), `concat`, `split` | **P1** |
| **⑥ 通用函数系统 (ufuncs)** | 上百种标准 ufunc，统一支持广播及 `.reduce()`, `.accumulate()`, `.outer()`, `.at()` 方法 | 实现了常见激活函数与基础四则运算广播，缺乏统一的 ufunc 抽象与超越函数族 | **P2** |
| **⑦ 现代随机数生成 (PRNG & Distributions)** | 现代 `Generator` 体系 (PCG64/Philox)，独立生成器对象，30+ 种概率分布与随机抽样 | 基础均匀分布 `rand`、正态分布 `fillNormal`、Top-K/Top-P 采样 | **P2** |
| **⑧ 高级线性代数 (`linalg`)** | SVD, QR, Cholesky 分解, 特征值/特征向量 (`eig/eigh`), 矩阵求逆/伪逆, 行列式, 范数, `einsum` | 具备 BLAS SGEMM 矩阵乘法、批量矩阵乘 `batchMatMul`、高斯-约旦消元解方程 `solveLinearSystem` | **P2** |
| **⑨ 专业科学计算模块** | 离散傅里叶变换 (`np.fft`)、多项式拟合 (`np.polynomial`)、文件读写 (`.npy`, `.npz`) | 专注于深度学习格式：支持 `SafeTensors`、`ZNNO` 检查点序列化、`BinaryMmapDataset` | **按需补充** |

---

## 🔍 3. 关键缺失点深度诊断

### 3.1 标量类型系统与多精度支持（Dtype System）
* **现状痛点**：
  * [`src/tensor.zig`](file:///Users/guangzong/Documents/znn/src/tensor.zig) 中的数据缓冲区直接定义为 `data: []f32` 与 `grad: []f32`。
  * **后果**：无法原生支持混合精度训练（FP16 / BF16 / FP32 AMP）与量化推理（INT8 / INT4）；词表索引等离散特征需要借助 `f32` 强转存储。
* **对齐目标**：
  * 将 `Tensor` 重构为泛型数据结构：`pub fn Tensor(comptime T: type) type`。
  * 优先支持：`f32`, `f64`, `f16`, `bf16`, `i32`, `i64`, `u8`, `bool`。
  * 支持安全的类型转换（如 `tensor.to(f16)`）与类型提升规则。

### 3.2 进阶切片与索引机制（Slicing & Indexing）
* **现状痛点**：
  * 目前仅有单一坐标读取 `tensor.get(&.{0, 1})`，缺少切片视图。
  * **后果**：在提取张分子块、序列截断、通道提取时必须分配新内存并通过循环或 `split` + `concat` 拼接，代码冗长且带来显著内存分配开销。
* **对齐目标**：
  * **零拷贝切片视图 (Strided Slice View)**：提供 `tensor.slice(ranges: []const SliceRange) !*Tensor`，通过调整 `offset`、`shape` 与 `strides` 实现 $O(1)$ 零拷贝视图。
  * **条件掩码与布尔索引**：实现 `maskedFill(mask: *Tensor(bool), val: T)` 与 `where(cond, x, y)`，极大简化 Transformer 注意力掩码与分类损失计算。
  * **高级花式索引**：支持整数数组索引 `gather(dim, index_tensor)`。

### 3.3 通用多轴归约算子（Multi-axis Reductions）
* **现状痛点**：
  * 框架缺少通用的多维求和与统计接口，只在局部神经网络层（如 `RMSNorm`, `LayerNorm`）中硬编码了局部方差和均值。
* **对齐目标**：
  * 封装通用轴向归约接口：
    ```zig
    pub fn sum(self: *Tensor, axis: ?usize, keepdims: bool, allocator: Allocator) !*Tensor;
    pub fn mean(self: *Tensor, axis: ?usize, keepdims: bool, allocator: Allocator) !*Tensor;
    pub fn var(self: *Tensor, axis: ?usize, keepdims: bool, ddof: usize, allocator: Allocator) !*Tensor;
    pub fn std(self: *Tensor, axis: ?usize, keepdims: bool, ddof: usize, allocator: Allocator) !*Tensor;
    ```
  * 同步支持计算图反向传播求导（广播逆向扩展累加）。

### 3.4 高级线性代数与爱因斯坦求和 (`linalg` & `einsum`)
* **现状痛点**：
  * 目前仅有最小二乘法用的高斯消元法 `solveLinearSystem` 和基础 SGEMM 矩阵乘法。
* **对齐目标**：
  * 引入数值稳定的正定矩阵求解：**Cholesky 分解 ($LL^T$)** 与 **QR 分解**。
  * 矩阵特征值与奇异值分解：**SVD 分解**（主成分分析、低秩近似基础）。
  * 引入轻量级 **Einsum 算子**：统一处理多头注意力变换与张量缩并。

---

## 🗺️ 4. 补齐演进规划 (Actionable Roadmap)

### Phase 1: 基础核心增强 (P0 - Quick Wins & Fundamentals)
- [x] **多轴归约函数库**：
  - [x] 实现通用 `sum(axis, keepdims)`, `mean(axis, keepdims)`, `variance/stdDev(axis, keepdims, ddof)`，支持连续与任意多维跨步张量。
- [x] **条件选择与掩码填充**：
  - [x] 实现 `where(condition, x, y)` 支持多维广播对齐，与 `maskedFill(mask, value)` / `maskedFill_(mask, value)`（包含计算图安全防护）。
- [x] **多维形态补充算子**：
  - [x] 实现 `squeeze(axis)` (单轴/全轴为 1 维度压缩) 与 `unsqueeze(dim)` (新维度扩充)。

### Phase 2: 泛型张量与切片视图 (P1 - Architectural Evolution)
- [ ] **泛型张量重构**：
  - 将 `Tensor` 升级为 `Tensor(comptime T: type)`，首批完整支持 `f32`, `f64`, `i32`, `bool`。
- [ ] **跨步零拷贝切片 (Strided View Slicing)**：
  - 引入 `SliceRange { start, end, step }`，支持对任意维度进行跨步切片且复用底层数据指针。
- [ ] **排序与检索算子**：
  - 实现 `sort(axis)`, `argsort(axis)`, `nonzero()`.

### Phase 3: 科学计算扩展与高性能线性代数 (P2 - Scientific Extensions)
- [ ] **现代随机数发生器体系**：
  - 抽象可配置种子的 `RNG` 结构体，支持泊松分布、指数分布、随机排列 `permutation` 与无放回抽样。
- [ ] **高级线性代数库 (`znn.linalg`)**：
  - 实现 `cholesky(A)`, `qr(A)`, `svd(A)`, `inv(A)`, `det(A)`.
- [ ] **轻量级 Einsum 引擎**：
  - 支持类似 `einsum("bshd,bthd->bhst", q, k)` 的字符串解析与执行。
