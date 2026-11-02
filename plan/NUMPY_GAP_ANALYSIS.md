# 🔬 ZNN vs. PyTorch / SciPy / NumPy 2.x 系统性对齐差距分析与架构演进全景报告 (Comprehensive Gap Analysis & Roadmap)

**版本号**：`v0.4.0`  
**审计基准提交**：`39224590` (通过 `zig build check-layers` 91 源码文件单向分层，`zig build test` 433 项测试全通 0 内存泄漏)  
**对标生态系统**：
- **PyTorch 2.5+** (`torch`, `torch.autograd`, `torch.linalg`, `torch.special`, `torch.distributions`)
- **SciPy 1.14+** (`scipy.linalg`, `scipy.stats`, `scipy.special`)
- **NumPy 2.x** (`numpy.linalg`, `numpy.random`, `numpy.fft`)

---

## 1. 定位、设计哲学与执行摘要 (Executive Summary)

### 1.1 定位与设计哲学对比

| 维度 | PyTorch 2.5+ | SciPy 1.14+ / NumPy 2.x | ZNN (Zig Neural Network) |
| :--- | :--- | :--- | :--- |
| **核心定位** | 工业级通用深度学习计算与自动微分引擎 | 科学计算、多维数组与高级数学算法底座 | **基于纯 Zig 构建的高性能深度学习与数值计算微框架** |
| **自动微分 (Autograd)** | 动态反向传播图 (`torch.autograd`)，支持二阶导数 | 无内置 Autograd，纯数值求值（依赖 JAX 或手写梯度） | **原生算子级拓扑排序动态计算图** (`autodiff.Graph`)，张量原生支持 `grad` |
| **内存与所有权模型** | 引用计数 TensorImpl，共享存储 Storage 视图 | 引用计数 ndarray，基础视图共享 Buffer | **显式物理设备缓冲区** (`backend.DeviceBuffer`)，受控生命周期，严格区分 View 与 Contiguous |
| **分层与工程约束** | 复杂 C++/Python 混合绑定，多后端动态分发 | C/Fortran 扩展与 Python 运行时粘合 | **8 层严格单向依赖** (`tools/check_layers.zig`)，零 C 运行时开销，开箱即静态编译 |
| **向后兼容性策略** | 严格保留弃用 API 与兼容垫片 | 渐进式废弃，2.0 经历了破坏性清理 | **重构优先，零历史包袱，废弃代码即刻剔除** (`AGENTS.md` §3) |

### 1.2 核心诊断结论

通过对 `src/tensor/`、`src/autodiff/`、`src/nn/`、`src/backend/` 的全面代码审计，`znn` 已经构建了极其稳固的 8 层单向依赖拓扑和基础张量计算底座（涵盖 Householder QR 分解、Jacobi SVD、SymEig 特征分解、零分配 `*Into` 目标算子体系、PyTorch 风格 `nn.Module` 反射、ConvNeXt 与 ResNet 全套模型族）。

然而，若要对标现代深度学习框架（PyTorch）与科学计算库（SciPy、NumPy 2.x），`znn` 目前存在 **四大系统性功能断层**：
1. **断层 1：缺乏通用离散索引与累加原语**：
   - 现存 `slice` 仅支持正步长（`step > 0` 且 `end >= start`），无法表达负步长零拷贝翻转（`arr[::-1]`）；
   - 张量层完全缺失通用 `gather(dim, index)`、`scatter(dim, index, src)`、`scatterAdd(dim, index, src)`、`indexSelect(dim, index)` 与 `maskedSelect(mask)`；
   - Autograd 计算图缺少离散索引求导节点，直接阻碍了图神经网络（GNN）、稀疏 Token 分发、稀疏注意力和复杂强化学习损失函数的原生表达。
2. **断层 2：多维归约算子维度表达单一且缺少极值 Autograd**：
   - 所有统计归约算子（`sum`、`mean`、`variance`、`stdDev`）的签名固定为 `axis: ?usize`，只接受单轴或全局归约，无法联合多轴归约（如图像特征图空间归约 `axes = &.{2, 3}`）；
   - `max` 与 `min` 未接入 Autograd 计算图（`OpType` 枚举中仅有 `Sum` 与 `Mean`，缺少 `Max` 与 `Min`），导致极值损失函数无法反向求导；
   - 缺失累乘 `prod`、前缀扫描 `cumsum` / `cumprod`、通用范数 `norm`、分位数 `quantile` 与中位数 `median`。
3. **断层 3：数值线性代数缺少对称正定系统与显式求逆**：
   - 求解线性系统仍依赖高斯-若尔当消元（`solveLinearSystem`，开销 $O(n^3)$）；
   - 缺失对称正定矩阵（SPD）的 Cholesky 分解 ($A = LL^T$) 与下三角前代求解器 `solveTriangularLower`，在协方差分析、高斯过程和高维回归中计算量比 Cholesky 多消耗 $100\%$ 浮点操作；
   - 缺失显式矩阵求逆 `inv`、摩尔-彭若斯广义伪逆 `pinv`、数值稳定对数行列式 `slogdet` 与轻量级 `einsum` 缩并引擎。
4. **断层 4：概率计算、随机发生器与特殊函数体系缺位**：
   - 现有随机初始化（`fillNormal`/`fillUniform`）依赖包级静态全局可变变量 `default_prng`，并发或多线程下存在竞态与状态污染；
   - 缺少独立线程安全的 `Generator` 对象体系；
   - 缺乏对象化概率分布体系（`Distribution`），无法通过重参数化技巧（`rsample`）进行路径可微采样（VAE、扩散模型、RL 必备）；
   - 缺失逆误差函数 `erfinv`、对数伽马 `lgamma`、双伽马 `digamma` 以及浮点稳定初等超越函数 `log1p` / `expm1`。

---

## 2. 核心能力全景对比矩阵 (Comprehensive Capability Matrix)

状态标识：
- ✅ **已完全支持 (Supported)**：功能完备，具备高精度数学逻辑与完整单测，符合工程规范。
- ⚠️ **部分支持 (Partially Supported)**：仅覆盖特化场景，或缺乏多维泛化，或未接入 Autograd。
- ❌ **尚未支持 (Missing)**：代码库中完全未实现。

| 模块领域 | 算子 / 功能特性 | PyTorch 2.5+ 等价实现 | NumPy 2.x / SciPy 等价实现 | ZNN 当前实现状态 | 现有代码位置与技术现状 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **1. 索引与切片视图** | 跨步零拷贝切片 | `tensor[start:end:step]` | `arr[start:end:step]` | ⚠️ 部分支持 | `src/tensor/reductions.zig:568` 支持 `SliceRange` 正步长零拷贝视图（`is_view = true`）；负步长 (`step < 0`) 计划通过显式逆序物化拷贝（`is_view = false`）对齐，规避无符号 `Shape.dims: [8]usize` 溢出风险 |
| | 通用轴向收集 | `torch.gather(dim, index)` | `np.take_along_axis(arr, idx, axis)` | ❌ 缺失 | 仅有特化的 `gatherMatMul` (`nn_kernels.zig:736`) 与嵌入查表 `embeddingLookup` |
| | 轴向离散散布/累加 | `torch.scatter` / `scatter_add` | `np.put_along_axis` | ❌ 缺失 | 张量层与 Autograd 缺失通用 `scatter` / `scatterAdd` 及伴随对偶 `index_add` |
| | 维度子张量切片提取 | `torch.index_select(dim, index)` | `np.take(arr, indices, axis)` | ❌ 缺失 | 无法按一维整型下标切片提取特定批次或通道 |
| | 条件布尔压缩过滤 | `torch.masked_select(mask)` | `arr[mask]` (1D Boolean Mask) | ❌ 缺失 | 仅支持标量覆写 `maskedFill` 与三元选择 `where`，无法提取变长有效元素 |
| | 非零元素坐标检索 | `torch.nonzero()` | `np.nonzero(arr)` | ✅ 已支持 | `src/tensor/reductions.zig:791` 返回 `*GenericTensor(usize)` |
| | 轴向稳定全序排序 | `torch.sort` / `argsort` | `np.sort` / `argsort` | ✅ 已支持 | `src/tensor/reductions.zig:680` 原生支持沿任意轴升降序稳定排序 |
| **2. 归约与统计分析** | 多轴联合求和/均值 | `torch.sum(dim=(1, 2))` | `np.sum(axis=(1, 2))` | ⚠️ 部分支持 | `reductions.zig:209` 仅支持单轴 `axis: ?usize` 或全局，**不支持多轴切片 `axes: []const usize`** |
| | 多轴方差与标准差 | `torch.var` / `std` | `np.var` / `std(ddof=1)` | ⚠️ 部分支持 | `reductions.zig:314` 支持 `ddof` 与 Autograd，但仅限单轴或全局 |
| | 极值与极值索引 | `torch.max` / `min` (含计算图) | `np.max` / `min`, `argmax/min` | ⚠️ 部分支持 | `reductions.zig:183` 仅支持单轴且**未接入 Autograd (`OpType` 无 Max/Min)**，无全局极值 |
| | 累乘归约算子 | `torch.prod(dim)` | `np.prod(axis)` | ❌ 缺失 | 缺失前向累乘与反向传播求导 |
| | 累加与累乘前缀扫描 | `torch.cumsum` / `cumprod` | `np.cumsum` / `cumprod` | ❌ 缺失 | 缺失前缀和与前缀积前向及反向求导 |
| | 通用向量与矩阵范数 | `torch.norm` / `linalg.norm` | `np.linalg.norm(ord, axis)` | ❌ 缺失 | 仅有针对权重的 `l1Loss`/`l2Loss`，无通用 L1/L2/Frobenius/无穷范数归约算子 |
| | 分位数与中位数 | `torch.quantile` / `median` | `np.quantile` / `median` | ❌ 缺失 | 缺失快速选择（QuickSelect）分位数分析与中位数提取 |
| | 对数求和指数归约 | `torch.logsumexp(dim)` | `scipy.special.logsumexp` | ⚠️ 部分支持 | `backend/vec_math.zig:336` 仅有底层行内核，张量层与 Autograd 计算图无此一等算子 |
| **3. 进阶线性代数** | 正交三角分解 | `torch.linalg.qr` | `np.linalg.qr` | ✅ 已支持 | `src/tensor/linalg.zig:67` 支持 `.reduced` 与 `.complete` 模式及回代 `qrSolve` |
| | 奇异值分解 | `torch.linalg.svd` | `np.linalg.svd` | ✅ 已支持 | `src/tensor/linalg.zig:643` 基于单边 Jacobi 旋转支持任意 $M \times N$ 矩阵 |
| | 实对称特征分解 | `torch.linalg.eigh` | `np.linalg.eigh` | ✅ 已支持 | `src/tensor/linalg.zig:317` 基于双边 Jacobi 旋转求解升序特征值与特征向量 |
| | 正定矩阵 Cholesky 分解 | `torch.linalg.cholesky` | `scipy.linalg.cholesky` | ❌ 缺失 | 无法高效求解对称正定系统 ($LL^T$) 与高斯过程协方差 |
| | 三角线性系统求解器 | `torch.linalg.solve_triangular` | `scipy.linalg.solve_triangular` | ⚠️ 部分支持 | `linalg.zig:225` 仅支持上三角回代 `solveTriangularUpper`，**缺失下三角前代 `solveTriangularLower`** |
| | 显式矩阵求逆 | `torch.linalg.inv` | `np.linalg.inv` | ❌ 缺失 | 仅有 Gauss-Jordan 解线性系统 `solveLinearSystem`，无显式求逆算子 |
| | 广义伪逆 (Moore-Penrose) | `torch.linalg.pinv` | `np.linalg.pinv` | ❌ 缺失 | 缺失基于现有 SVD 的截断奇异值求逆 |
| | 行列式与对数行列式 | `torch.linalg.det` / `slogdet` | `np.linalg.det` / `slogdet` | ❌ 缺失 | 缺失对数行列式，阻碍多元正态概率计算与连续流模型 |
| | 爱因斯坦求和算子 | `torch.einsum("...", ...)` | `np.einsum("...", ...)` | ❌ 缺失 | 缺失基于标记解析的轻量级多维张量缩并执行引擎 |
| **4. 科学计算与概率分布** | 独立可重现随机发生器 | `torch.Generator` | `np.random.Generator` (PCG64) | ❌ 缺失 | `src/tensor/ops.zig:215` 采用全局可变 `default_prng`，非线程安全且不支持分叉 |
| | 随机排列与洗牌 | `torch.randperm` | `np.random.permutation/shuffle`| ❌ 缺失 | 仅有针对小数组的手写循环，缺少张量级随机洗牌算子 |
| | 可微概率分布对象基类 | `torch.distributions.Distribution` | `scipy.stats.rv_continuous` | ❌ 缺失 | `engine/sampling.zig:311` 的 `Distribution` 仅为离散 Token 候选表，非连续概率分布 |
| | 重参数化路径采样 (`rsample`) | `dist.rsample()` | N/A (需手动建图) | ❌ 缺失 | 缺失将随机隐变量无缝接入 Autograd 梯度的机制（VAE / 扩散模型必备） |
| | 对数概率密度计算 (`log_prob`) | `dist.log_prob(x)` | `dist.logpdf(x)` | ❌ 缺失 | 无法直接计算高斯、均匀、贝塔分布的损失导数 |
| | 误差函数族 | `torch.special.erf` / `erfinv` | `scipy.special.erf` / `erfinv` | ⚠️ 部分支持 | 仅在激活函数内部间接使用标量 `erff`，无张量级 `erf` 与逆误差函数 `erfinv` |
| | 伽马函数族与对数伽马 | `torch.special.gammaln` / `digamma` | `scipy.special.gammaln` / `psi` | ❌ 缺失 | 狄利克雷与伽马分布参数学习必备函数缺失 |
| | 浮点稳定初等超越函数 | `torch.log1p` / `expm1` | `np.log1p` / `expm1` | ❌ 缺失 | 浮点小量截断丢失精度，无张量级实现 |

---

## 3. 四大领域深度诊断与技术剖析

### 3.1 领域一：多维高级索引、跨步切片视图与花式索引体系

#### 现有实现与技术依据
1. **跨步零拷贝切片视图 (`Tensor.slice`)**：
   - 源码位于 `src/tensor/reductions.zig:568–618` 与 `src/autodiff/graph.zig:1114–1202`；
   - 通过 `SliceRange { start: ?usize, end: ?usize, step: usize = 1 }` 在原有张量上重算 `shape`、`strides` 与物理内存 `offset`，标记 `is_view = true`，实现 $O(1)$ 零拷贝创建；
   - `src/autodiff/backward_math.zig:238–251` 实现了 `Slice` 算子的反向传播（将输出切片梯度根据步长与偏移散布累加回输入张量梯度缓冲区）。
2. **条件掩码与布尔选择**：
   - `Tensor.where` (`src/tensor/reductions.zig:350`) 支持多维广播对齐；
   - `Tensor.maskedFill` / `maskedFill_` (`src/tensor/reductions.zig:384, 411`) 允许用指定标量覆写掩码对应元素；
   - 两者均已完整接入 Autograd 计算图 (`src/autodiff/backward_math.zig:186–237`)。
3. **特化嵌入与专家收集**：
   - `src/tensor/nn_kernels.zig:642` 的 `embeddingLookupInto` 支持根据 1D/2D 整型张量提取权重行；
   - `src/tensor/nn_kernels.zig:736` 的 `gatherMatMulInto` 为 MoE 专家并行提供了分组矩阵乘。

#### 核心差距与技术缺陷
1. **缺失通用 `gather(dim, index_tensor)`**：
   - **PyTorch 规范**：`out[i][j][k] = input[index[i][j][k]][j][k]` (当 `dim=0` 时)。输入与输出形状与 `index_tensor` 严格相同。
   - **理论与工程影响**：当前无法在注意力输出后提取变长终止符位置特征、无法进行动态 Top-P 词元收集、无法实现强化学习中的动作价值提取（Q-learning: `Q.gather(1, actions)`）。
2. **缺失通用 `scatter(dim, index_tensor, src)` 与 `scatterAdd(dim, index_tensor, src)`**：
   - **PyTorch 规范**：`scatter`（覆写）与 `scatter_add`（累加）：`self[index[...]][...] += src[...]`。
   - **理论与工程影响**：`scatterAdd` 是图神经网络（Graph Neural Networks, GNN）节点消息聚合、分箱统计（Histogram）、稀疏梯度求导以及 `gather` 反向传播的核心原语；`scatter` 是离散决策覆写与注意力稀疏化的基础原语。目前缺失导致无法实现任意离散反向求导。
3. **缺失一维子张量索引与累加 `indexSelect` / `indexAdd`**：
   - 无法直接按照整型切片沿指定维度批量抽取非连续切片（如 Mini-batch 随机重排、多样本按长度重新排序）；同时缺失伴随对偶原语 `indexAdd`（`indexSelect` 反向求导闭环的基础）。
4. **缺失布尔掩码一维提取 `maskedSelect(mask)`**：
   - NumPy 与 PyTorch 中高频使用的 `arr[mask]`（返回一维压缩张量）在 `znn` 中无法直接表达，迫使上层必须先调用 `nonzero()`，再逐个坐标回读，产生极高的遍历开销。
5. **切片视图无法表达负步长 (`step < 0`) 且受限于无符号 Shape**：
   - `SliceRange` 的字段为 `usize`，且 `reductions.zig:585` 中显式断言 `if (end < start) return error.InvalidSliceRange;`。鉴于底层 `Tensor.strides` 为无符号静态数组 `Shape.dims: [8]usize`，负步长无法在不溢出的前提下作为只读视图表示，因此需通过显式逆序物化拷贝（`is_view = false`）安全实现数组翻转（`arr[::-1]`）与多维坐标镜像。

---

### 3.2 领域二：多轴联合归约与统计分析算子库

#### 现有实现与技术依据
1. **单轴与全局归约算子**：
   - `src/tensor/reductions.zig:209–347` 实现了 `sum(axis, keepdims)`、`mean(axis, keepdims)`、`variance(axis, keepdims, ddof)` 与 `stdDev(axis, keepdims, ddof)`；
   - 支持通过 `axis = null` 进行全局标量归约，或通过 `axis = k` 沿单个维度归约；
   - `src/autodiff/backward_math.zig:13–66` 的 `reduceSumMeanBackward` 通过 `shape.RowIter` 实现了连续梯度回传。
2. **极值归约内核**：
   - `src/tensor/reductions.zig:96–197` 通过 `reduceExtremum` 统一了 `max`、`min`、`argmax`、`argmin` 的单轴扫描。

#### 核心差距与技术缺陷
1. **无法进行多轴联合归约 (Multi-axis Reductions)**：
   - **现状**：函数签名固定为 `axis: ?usize`。如果用户需要对四维图像张量 `[N, C, H, W]` 沿空间维度 `(2, 3)` 计算均值或求和（如全局平均池化、LayerNorm、特征图能量归一化），必须连续调用两次单轴归约，产生中间张量分配与显存搬运开销。
   - **对齐目标**：将接口升级为支持多轴切片 `axes: []const usize`，如 `sumAxes(&.{2, 3}, keepdims)`。
2. **`max` 与 `min` 未接入 Autograd 计算图**：
   - `src/autodiff/types.zig:6–79` 的 `OpType` 仅包含 `Sum` 与 `Mean`，缺少 `Max` 与 `Min`。
   - 在动态图求导中，$\max(x)$ 的次梯度应当精确回传至最大值所在索引下标（若有多个相同最大值，则均匀分配或赋予首个索引）。目前缺失此机制，导致用户无法在损失函数中使用 $\max$（如 Hinge Loss、Margin Ranking Loss、最大池化通用图层）。
3. **缺失累乘与累加扫描算子 (`prod`, `cumsum`, `cumprod`)**：
   - 无法高效计算马尔可夫链状态概率积、离散时间折扣因子序列累积 $\gamma^t$、序列自回归因果前缀和。
4. **缺失通用范数归约 (`norm`)**：
   - 现仅有针对全张量权重的正则化项 `l1Loss`/`l2Loss`。无法沿指定维度计算 $p$ 范数（如特征向量单位化 $L_2$ 归一化：`x / torch.norm(x, p=2, dim=-1, keepdim=True)`）。
5. **缺失分位数与中位数 (`quantile`, `median`)**：
   - 缺乏抗噪统计分析能力，无法进行鲁棒梯度裁剪（基于分位数的 Gradient Clipping）与箱线图离群点过滤。

---

### 3.3 领域三：进阶数值线性代数与爱因斯坦求和 (`linalg` & `einsum`)

#### 现有实现与技术依据
1. **正交三角分解 (`Tensor.qr` / `qrWithOptions`)**：
   - 位于 `src/tensor/linalg.zig:58–214`，基于 Householder 变换，规范化 $R$ 对角线非负；
   - 提供了上三角回代求解器 `solveTriangularUpper` 与最小二乘求解器 `qrSolve`。
2. **实对称矩阵 Jacobi 特征分解 (`Tensor.symeig`)**：
   - 位于 `src/tensor/linalg.zig:317–460`，基于经典雅可比双边旋转法，自动升序排列特征值并同步置换特征向量矩阵。
3. **奇异值分解 (`Tensor.svd`)**：
   - 位于 `src/tensor/linalg.zig:463–686`，采用单边雅可比旋转法，支持任意高瘦与矮胖矩阵分解。

#### 核心差距与技术缺陷
1. **缺失对称正定矩阵 Cholesky 分解 ($LL^T$) 与前代求解器**：
   - **理论优势**：对于对称正定协方差矩阵与核矩阵，Cholesky 分解的计算复杂度仅为高斯消元法或 LU 分解的 **$50\%$**（$\frac{1}{3} n^3$ 对比 $\frac{2}{3} n^3$ 浮点操作），且由于无须选主元，访存连续性与数值稳定性极高。
   - **现状短板**：`src/tensor/ops.zig:520` 的 `solveLinearSystem` 仍使用开销巨大的 Gauss-Jordan 消元；`src/regression.zig` 的高维岭回归在特征数大于样本数时直接求逆受阻。同时缺失下三角前代求解器 `solveTriangularLower`（求解 $L y = b$）。
2. **缺失显式矩阵求逆 (`inv`) 与广义伪逆 (`pinv`)**：
   - **`inv(A)`**：目前无法直接输出非奇异方阵的显式逆矩阵 $A^{-1}$。
   - **`pinv(A)`**：对于病态矩阵或奇异矩阵，通过现有 `svd` 分解（$A = U \Sigma V^T$），截断微小奇异值并计算 $A^+ = V \Sigma^+ U^T$ 是标准工业解法，但目前张量层未封装该高层接口。
3. **缺失行列式与对数绝对值行列式 (`det`, `slogdet`)**：
   - 在高维概率统计中，直接计算 $\det(A)$ 极易发生浮点下溢或上溢（例如 $100 \times 100$ 协方差矩阵）。
   - PyTorch 与 SciPy 统一通过 `slogdet(A)` 返回 $(\text{sign}, \ln|\det(A)|)$。基于 Cholesky 分解计算正定矩阵行列式对数极简：$\ln|\det(A)| = 2 \sum_{i=1}^n \ln(L_{ii})$。目前该能力完全缺失。
4. **缺失爱因斯坦求和算子 (`einsum`)**：
   - 现代深度学习（特别是多头注意力、张量网络与复杂投影）高度依赖字符串声明式缩并，如：
     `einsum("bshd,bthd->bhst", q, k)`（点积注意力矩阵）
     `einsum("bhst,bthd->bshd", attn, v)`（值向量加权合并）
   - 目前 `znn` 必须通过繁琐的手动 `transposeView`、`reshape`、`batchMatMul` 串联组合，代码冗长且增加了中间临时缓冲生命周期管理的负担。

---

### 3.4 领域四：现代科学计算与概率分布可微采样体系

#### 现有实现与技术依据
1. **张量随机数填充**：
   - `src/tensor/core.zig:645–686` 提供了 `fillNormal(random, mean, stddev)`（基于 Box-Muller 变换）与 `fillUniform(random, min, max)`；
   - `src/tensor/weight_init.zig` 提供了深度学习权重初始化器（Xavier, Kaiming, Orthogonal）。
2. **底层向量化超越函数与采样原语**：
   - `src/backend/vec_math.zig:34–88` 实现了基于 Cody-Waite 与 Cephes 多项式逼近的高性能 8 路 SIMD `exp8`；
   - `src/engine/sampling.zig:311` 实现了 LLM 自回归生成的候选词元采样器（Top-K / Top-P / 惩罚系数）。

#### 核心差距与技术缺陷
1. **全局伪随机状态污染与多线程竞态**：
   - `src/tensor/ops.zig:215` 定义了包级静态全局变量：
     ```zig
     var default_prng = std.Random.DefaultPrng.init(12345);
     ```
   - 当在 `std.Io` 多核并行或多 Agent 异步环境下并发创建张量时，共享全局可变 `default_prng` 会引发非确定性结果与数据竞态。必须转向类似于 NumPy 2.x `np.random.Generator` 的显式生成器对象模型。
2. **缺失对象化概率分布体系与重参数化可微采样 (`rsample`)**：
   - **理论机制**：在变分自编码器（Variational Autoencoders, VAE）、流形生成模型、扩散模型（Diffusion Models）以及强化学习策略梯度（PPO / SAC）中，模型需要从概率分布 $z \sim q_\phi(z|x)$ 中采样，并反向传播更新编码器参数 $\phi$。
   - **现状断层**：标准采样 `sample()` 具有随机性不可导，必须使用 **重参数化技巧 (Reparameterization Trick)**：
     $$z = \mu(\phi) + \sigma(\phi) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
   - 由于 `znn` 缺乏一等 `Distribution` 结构体，用户无法通过 `dist.rsample(graph)` 将采样节点自动连接到 Autograd 计算图中。
3. **缺失对数概率密度评估 (`log_prob`) 与散度求解 (`klDivergence`)**：
   - 无法对输入数据计算 $\ln p(x; \theta)$，阻碍了最大似然估计（MLE）、策略优化中的重要性权重比例计算以及先验分布与后验分布之间的解析 KL 散度约束求解。
4. **特殊数学函数库缺位 (`special`)**：
   - 缺失逆误差函数 `erfinv(x)`（从均匀分布到任意高斯分位数的反函数映射基础）；
   - 缺失对数伽马函数 `lgamma(x)` 与双伽马函数 `digamma(x)`（Dirichlet、Beta、Gamma、Student-t 分布的对数概率与熵计算必备）；
   - 缺失数值稳定的小量计算函数 `log1p(x) = \ln(1 + x)` 与 `expm1(x) = e^x - 1`。

---

## 4. 技术难度评估与风险矩阵 (Technical Difficulty Assessment)

下表对四大领域的关键补齐项进行算力复杂度、依赖包层级、内存分配策略与实施难度评级：

| 算子 / 功能特性 | 目标包层级 | 时间复杂度 | 内存分配特性 | 技术难度评级 | 关键技术难点与风险 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`gather` / `gatherInto`** | Layer 1 (`tensor`)<br>Layer 2 (`autodiff`) | $O(N_{\text{idx}})$ | 零分配（`Into` 版）<br>或单次输出分配 | **中等 (Medium)** | 多维坐标与跨步映射；越界下标安全防御；必须支持非连续输入视图 |
| **`scatterAdd` / `scatterAddInto`** | Layer 1 (`tensor`)<br>Layer 2 (`autodiff`) | $O(N_{\text{src}})$ | 原位就地累加<br>或单次输出分配 | **中等 (Medium)** | 同一下标并发累加碰撞；作为 `gather` 反向求导算子时的数值严格对齐 |
| **负步长切片 (`step < 0`)** | Layer 1 (`tensor`) | $O(N)$ (物化拷贝) | 逆序物化拷贝 (`is_view = false`)；正步长零拷贝视图 | **低 (Low)** | 明确 `Shape.dims: [8]usize` 无符号约束；`step > 0` 零拷贝视图，`step < 0` 逆序物化拷贝 |
| **多轴联合归约 (`sumAxes` 等)** | Layer 1 (`tensor`)<br>Layer 2 (`autodiff`) | $O(N)$ | 单次输出分配 | **中等 (Medium)** | 任意非连续轴的复合步长折叠；SIMD 向量化缓存命中优化；多轴反向广播恢复 |
| **极值求导 (`OpType.Max/Min`)** | Layer 2 (`autodiff`) | $O(N)$ | 复用输出缓冲 | **中等 (Medium)** | 极大值次梯度精准路由至首个或均匀分散下标；稀疏梯度回传 |
| **Cholesky 分解 (`cholesky`)** | Layer 1 (`tensor/linalg`) | $O(\frac{1}{3} n^3)$ | 单次方阵分配 | **中等 (Medium)** | 对称正定性校验；平方根前负数断言 (`error.NotPositiveDefinite`)；缓存分块 |
| **下三角求解 (`solveTriangularLower`)**| Layer 1 (`tensor/linalg`) | $O(n^2)$ | 单次输出分配 | **低 (Low)** | 前代消除（Forward Substitution）；除零奇异性防护 |
| **显式求逆与伪逆 (`inv`, `pinv`)** | Layer 1 (`tensor/linalg`) | $O(n^3)$ | 单次输出分配 | **中等 (Medium)** | 奇异矩阵防御；基于现有 SVD 截断微小奇异值；截断阈值 `rcond` 调优 |
| **爱因斯坦求和 (`einsum`)** | Layer 1 (`tensor/einsum`) | 取决于收缩路径 | 动态或规划缓冲 | **高 (High)** | 字符串编译期/运行期解析；最优收缩路径贪心搜索；中间张量生命周期管理 |
| **独立生成器 (`Generator`)** | Layer 1 (`tensor/random`) | $O(1)$ | 栈上结构体 | **低 (Low)** | 彻底淘汰包级全局 `default_prng`；提供无竞态随机性与状态序列化 |
| **可微分布 (`Normal.rsample`)** | Layer 2 (`autodiff`)<br>Layer 3 (`nn`) | $O(N)$ | 计算图节点分配 | **中等 (Medium)** | 噪声张量 $\epsilon$ 的无梯度分离与路径重参数化算子连接；`log_prob` 符号稳定性 |
| **特殊超越函数 (`erfinv`, `lgamma`)** | Layer 1 (`tensor/special`) | $O(N)$ | 单次输出分配 | **高 (High)** | 有理 Chebyshev 多项式逼近与 Lanczos 近似精度；边界发散与下溢防护 |

---

## 5. 优先级排序与实施路线图 (Milestones 11–14 Architecture Proposal)

根据框架基础依赖关系与功能倍增效应，后续演进规划分为严格串行的 4 个里程碑（**P1 为核心基础，P2 为高阶拓展**）：

```
                    ┌─────────────────────────────────────────┐
                    │ Milestone 11: 高级索引、跨步视图与离散累加 │
                    │ (gather, scatter, scatterAdd, select)   │  ◄── P1 首要推进 (Next in Queue)
                    └───────────────────┬─────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────┐
                    │ Milestone 12: 多轴联合归约与进阶统计算子 │
                    │ (axes reduction, prod, cumsum, max/min) │  ◄── P1 基础统计
                    └───────────────────┬─────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────┐
                    │ Milestone 13: 高性能正定代数与求解器     │
                    │ (Cholesky, inv, pinv, det, einsum)      │  ◄── P2 科学计算
                    └───────────────────┬─────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────┐
                    │ Milestone 14: 现代科学计算与可微分布体系 │
                    │ (Generator, dist, rsample, special math)│  ◄── P2 概率深度学习
                    └─────────────────────────────────────────┘
```

---

## 6. 四大里程碑详细规划与接口契约 (Specifications & Contracts)

### 6.1 Milestone 11: 高级张量索引、跨步切片视图与离散累加体系 (Advanced Indexing, Strided Views & Scatter Operations)
- **优先级**：**P1 (首要推进目标 / Next in Queue)**
- **核心定位**：打通非连续张量元素提取、离散散布累加与反向传播闭环，为 GNN、MoE、稀疏注意力和强化学习奠定基础。
- **涉及文件**：
  - `src/tensor/indexing.zig` (新建：核心索引内核与目标写入算子，统一以 `*const Tensor` 为主标准类型并提供泛型/切片辅助封装)
  - `src/tensor/reductions.zig` (升级 `SliceRange` 支持带符号步长：`step > 0` 零拷贝视图，`step < 0` 显式逆序物化拷贝规避无符号溢出)
  - `src/autodiff/types.zig` (扩展 `OpType.Gather`, `OpType.Scatter`, `OpType.ScatterAdd`, `OpType.IndexSelect`)
  - `src/autodiff/backward_math.zig` (实现离散求导反向传播，包含 `Scatter` 覆写位置梯度置零与 `IndexSelect` 的 `indexAddInto` 梯度累加)
  - `src/autodiff/graph.zig` (暴露计算图 API)
- **核心接口契约**：
  ```zig
  /// 索引类型约定：以 *const Tensor 为主标准类型（通过 indexScalarToUsize 统一转换并做非负与越界校验），
  /// 确保与 Autograd 图节点和 DeviceBuffer 统一内存同构；同时配套提供接收 *const GenericTensor(usize) 与 []const usize 的辅助封装。

  /// 通用轴向索引收集：沿着指定 dim 按照 index 张量提取元素
  /// 约束：index.rank == self.rank，且 index.shape[d] <= self.shape[d]
  pub fn gather(self: *Tensor, dim: usize, index: *const Tensor, allocator: std.mem.Allocator) !*Tensor;
  pub fn gatherInto(dest: *Tensor, self: *Tensor, dim: usize, index: *const Tensor) !void;

  /// 通用轴向离散覆写 (Scatter)：将 src 中的元素按照 index 散布覆写至 dest
  pub fn scatter(self: *Tensor, dim: usize, index: *const Tensor, src: *Tensor, allocator: std.mem.Allocator) !*Tensor;
  /// 5 参数目标写入签名：dest 为输出缓冲，先复制 base，再按 index 写入 src
  pub fn scatterInto(dest: *Tensor, base: *const Tensor, dim: usize, index: *const Tensor, src: *Tensor) !void;

  /// 通用轴向离散累加：将 src 的值沿着 dim 维度按照 index 散布并累加至 dest
  pub fn scatterAdd(self: *Tensor, dim: usize, index: *const Tensor, src: *Tensor, allocator: std.mem.Allocator) !*Tensor;
  /// 4 参数原位累加签名：dest 直接原位累加，满足梯度无分配反向累加与无别名契约
  pub fn scatterAddInto(dest: *Tensor, dim: usize, index: *const Tensor, src: *Tensor) !void;

  /// 一维整型下标切片提取 (沿着 dim 维度提取 index 指定的超平面)
  pub fn indexSelect(self: *Tensor, dim: usize, index: *const Tensor, allocator: std.mem.Allocator) !*Tensor;
  pub fn indexSelectInto(dest: *Tensor, self: *Tensor, dim: usize, index: *const Tensor) !void;

  /// 一维整型下标切片累加 (沿着 dim 维度将 src 累加至指定超平面，与 indexSelect 伴随对偶)
  pub fn indexAdd(self: *Tensor, dim: usize, index: *const Tensor, src: *Tensor, allocator: std.mem.Allocator) !*Tensor;
  pub fn indexAddInto(dest: *Tensor, dim: usize, index: *const Tensor, src: *Tensor) !void;

  /// 布尔掩码一维压缩提取 (返回一维扁平有效元素)
  pub fn maskedSelect(self: *Tensor, mask: *const Tensor, allocator: std.mem.Allocator) !*Tensor;
  ```
- **Autograd 闭环设计**：
  - **Gather 求导**：
    - 前向：$y = \text{gather}(x, \text{dim}, \text{index})$
    - 反向：$\frac{\partial L}{\partial x} = \text{scatterAdd}(\mathbf{0}, \text{dim}, \text{index}, \frac{\partial L}{\partial y})$（零分配路径调用 `scatterAddInto(x.grad, dim, index, output.grad)`）
  - **Scatter 求导 (覆写梯度置零)**：
    - 前向：$y = \text{scatter}(x, \text{dim}, \text{index}, \text{src})$
    - 反向：$\frac{\partial L}{\partial x} = \text{scatter}(\frac{\partial L}{\partial y}, \text{dim}, \text{index}, \mathbf{0})$，$\frac{\partial L}{\partial \text{src}} = \text{gather}(\frac{\partial L}{\partial y}, \text{dim}, \text{index})$
    - 覆写位置基底原值偏导为 0（被覆写梯度置零），未覆写位置透传全额梯度；源输入抽取对应位置梯度。
  - **ScatterAdd 求导**：
    - 前向：$y = \text{scatterAdd}(x, \text{dim}, \text{index}, \text{src})$
    - 反向：$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}$（直通累加），$\frac{\partial L}{\partial \text{src}} = \text{gather}(\frac{\partial L}{\partial y}, \text{dim}, \text{index})$
  - **IndexSelect 求导**：
    - 前向：$y = \text{indexSelect}(x, \text{dim}, \text{index})$
    - 反向：$\frac{\partial L}{\partial x} = \text{indexAdd}(\mathbf{0}, \text{dim}, \text{index}, \frac{\partial L}{\partial y})$（零分配路径调用 `indexAddInto(x.grad, dim, index, output.grad)`）

---

### 6.2 Milestone 12: 多轴联合归约与进阶统计算子库 (Multi-Axis Reductions & Statistics)
- **优先级**：**P1 (基础统计完备度)**
- **核心定位**：消除高维视觉与归一化图层的嵌套单轴归约样板代码，补齐极值次梯度求导。
- **涉及文件**：
  - `src/tensor/reductions.zig` (升级多轴签名，增加 `prod`, `cumsum`, `norm`, `quantile`)
  - `src/autodiff/types.zig` (新增 `OpType.Max`, `OpType.Min`, `OpType.Prod`, `OpType.LogSumExp`)
  - `src/autodiff/backward_math.zig` (实现极值与累乘求导)
  - `src/autodiff/graph.zig` (暴露图 API)
- **核心接口契约**：
  ```zig
  /// 多轴联合求和：例如对图像 [N, C, H, W] 沿空间轴 &.{2, 3} 求和
  pub fn sumAxes(self: *Tensor, axes: []const usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor;
  pub fn meanAxes(self: *Tensor, axes: []const usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor;
  pub fn varianceAxes(self: *Tensor, axes: []const usize, keepdims: bool, ddof: usize, allocator: std.mem.Allocator) !*Tensor;

  /// 累乘与前缀累积扫描
  pub fn prod(self: *Tensor, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor;
  pub fn cumsum(self: *Tensor, axis: usize, allocator: std.mem.Allocator) !*Tensor;
  pub fn cumprod(self: *Tensor, axis: usize, allocator: std.mem.Allocator) !*Tensor;

  /// 范数归约 (L1, L2, Frobenius, Inf, p 范数)
  pub const NormOrder = union(enum) { l1, l2, frobenius, inf, p: f32 };
  pub fn norm(self: *Tensor, ord: NormOrder, axes: ?[]const usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor;

  /// 分位数与中位数 (基于 QuickSelect 算法)
  pub fn quantile(self: *Tensor, q: f32, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor;
  pub fn median(self: *Tensor, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor;
  ```

---

### 6.3 Milestone 13: 进阶数值线性代数库与求解器 (Advanced Numerical Linear Algebra & Solvers)
- **优先级**：**P2 (数值优化与科学计算)**
- **核心定位**：提供对称正定高效分解、下三角求解、显式求逆、伪逆与 Einsum 缩并。
- **涉及文件**：
  - `src/tensor/linalg.zig` (新增 `cholesky`, `solveTriangularLower`, `choSolve`, `inv`, `pinv`, `det`, `slogdet`)
  - `src/tensor/einsum.zig` (新建轻量级爱因斯坦求和引擎)
  - `src/regression.zig` (集成 Cholesky 求解器加速高斯过程与岭回归)
- **核心接口契约**：
  ```zig
  /// 对称正定矩阵 Cholesky 分解: A = L * L^T (或 U^T * U)
  pub fn cholesky(A: Tensor, upper: bool, allocator: std.mem.Allocator) !*Tensor;

  /// 下三角线性系统前代求解器 (Forward Substitution: L * x = b)
  pub fn solveTriangularLower(L: Tensor, b: Tensor, allocator: std.mem.Allocator) !*Tensor;

  /// 基于 Cholesky 分解的高精度线性方程求解器: A * x = b
  pub fn choSolve(L: Tensor, b: Tensor, upper: bool, allocator: std.mem.Allocator) !*Tensor;

  /// 显式方阵求逆 A^{-1}
  pub fn inv(A: Tensor, allocator: std.mem.Allocator) !*Tensor;

  /// 摩尔-彭若斯广义伪逆 A^+ (基于现有 SVD 截断微小奇异值)
  pub fn pinv(A: Tensor, rcond: ?f32, allocator: std.mem.Allocator) !*Tensor;

  /// 行列式与对数绝对值行列式
  pub fn det(A: Tensor, allocator: std.mem.Allocator) !f32;
  pub const SLogDetResult = struct { sign: f32, logabsdet: f32 };
  pub fn slogdet(A: Tensor, allocator: std.mem.Allocator) !SLogDetResult;

  /// 轻量级爱因斯坦求和
  pub fn einsum(equation: []const u8, operands: []const *Tensor, allocator: std.mem.Allocator) !*Tensor;
  ```

---

### 6.4 Milestone 14: 现代科学计算与可微概率分布体系 (Modern Scientific Computing & Distributions)
- **优先级**：**P2 (概率深度学习与生成模型)**
- **核心定位**：彻底根除全局可变随机状态，建立对象化概率分布体系与重参数化求导。
- **涉及文件**：
  - `src/tensor/random.zig` (新建独立 `Generator` 对象)
  - `src/tensor/special.zig` (新建特殊数学函数库)
  - `src/tensor/distributions/` (新建概率分布抽象)
- **核心接口契约**：
  ```zig
  /// 独立可重现随机数生成器
  pub const Generator = struct {
      prng: std.Random.DefaultPrng,
      pub fn init(seed: u64) Generator;
      pub fn random(self: *Generator) std.Random;
      pub fn permutation(self: *Generator, n: usize, allocator: std.mem.Allocator) !*GenericTensor(usize);
      pub fn shuffle(self: *Generator, t: *Tensor, axis: usize) void;
  };

  /// 特殊超越函数库 (src/tensor/special.zig)
  pub fn erf(x: *Tensor, allocator: std.mem.Allocator) !*Tensor;
  pub fn erfinv(x: *Tensor, allocator: std.mem.Allocator) !*Tensor;
  pub fn gammaln(x: *Tensor, allocator: std.mem.Allocator) !*Tensor;
  pub fn digamma(x: *Tensor, allocator: std.mem.Allocator) !*Tensor;
  pub fn log1p(x: *Tensor, allocator: std.mem.Allocator) !*Tensor;
  pub fn expm1(x: *Tensor, allocator: std.mem.Allocator) !*Tensor;

  /// 正态分布对象 (支持路径可微重参数化采样)
  pub const Normal = struct {
      loc: *Tensor,    // 均值 mu
      scale: *Tensor,  // 标准差 sigma

      pub fn init(loc: *Tensor, scale: *Tensor) Normal;
      pub fn sample(self: Normal, gen: *Generator, sample_shape: []const usize, allocator: std.mem.Allocator) !*Tensor;
      pub fn rsample(self: Normal, graph: *autodiff.Graph, gen: *Generator, sample_shape: []const usize) !*Tensor;
      pub fn logProb(self: Normal, graph: *autodiff.Graph, value: *Tensor) !*Tensor;
      pub fn entropy(self: Normal, graph: *autodiff.Graph) !*Tensor;
  };
  ```

---

## 7. 结语与接续准则 (Conclusion & Next Steps)

本差距分析确立了 ZNN 迈向完整数值与深度学习计算底座的战略蓝图。在实际工程落地中，必须严格遵循以下原则：
1. **测试驱动与零内存泄漏**：每一个新算子必须提供端到端测试，并通过 GPA 0 内存泄漏验证；
2. **8 层架构禁令**：新增文件必须严格注册在 `tools/check_layers.zig` 对应的层级，严禁反向越层导入；
3. **默认配置与开箱即用**：所有选项结构体必须提供 `defaultOptions()` 并支持 `.{}` 零样板调用；
4. **即刻启动 M11**：首要编写并推进 `prompts/11_advanced_indexing_and_strided_views.md`。
