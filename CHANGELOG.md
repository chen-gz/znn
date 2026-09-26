# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.4] - 2026-09-26

### Added
- **可视化报告 HTML 独立模板文件 (`src/nn/visualization/template.html`)**:
  - 将所有 HTML/CSS 及纯前端 JavaScript 交互渲染引擎从 Zig 代码中彻底抽离为独立的 `template.html` 模板文件，Zig 代码通过 `@embedFile` 仅负责插值注入 JSON 数据包。
- **发布规则强化与 Git Tag 机制 (`AGENTS.md`)**:
  - 明确规定在每次版本号递增升级时，必须在对应的 commit 上创建语义化版本 Git 标签（`git tag vX.Y.Z`），并同步推送到远端仓库。

### Fixed
- **Mermaid 架构流程图嵌套层级修正 (`src/nn/visualization/template.html`)**:
  - 修复多层 Transformer Block 在 Mermaid 中扁平渲染的问题，重构为递归 Scope Tree 解析，将 `gpt.layers.0` 和 `gpt.layers.1` 真正作为内嵌的嵌套子模块 (`subgraph`) 呈现。
- **DAG 架构流图拓扑管道修复 (`src/nn/visualization/template.html` & `src/nn/visualization.zig`)**:
  - 修复 `outputs.logits` 箭头回环指回非目标节点的问题，改为基于严格正向拓扑链条渲染。
  - 修正中间模块（如 `ln_1`, `ln_2`, `ln_f`）的后缀匹配与数据流路径，杜绝虚假中间外壳节点。

## [0.2.3] - 2026-09-26

### Changed
- **可视化报告前后端完全解耦与独立模块化 (`src/nn/visualization.zig` & `src/autodiff/graph.zig`)**:
  - 从 `autodiff.Graph` 中剥离 `formatHtmlReport` 与 `exportHtmlReport` 方法，解除计算图引擎与 HTML/DOM 展现层的耦合。
  - 将可视化子系统拆分为两个职责分明的独立子模块：
    - `graph_ir`: 专注于计算图解析、层次化模块树 (`ModuleNode`) 构建、参数与内存递归汇聚，并提供结构化中间表示及递归 JSON 导出 (`generateJson` / `exportJson`)。
    - `html_report`: 独立的自包含 HTML 报告生成器，仅接收约定的递归 JSON 字符串或 `ModelHierarchyGraph` 强类型结构，完全不依赖 `autodiff.Graph`。
  - 在 `Graph` 上提供轻量级 `formatJson` 与 `exportJson` 方法，便于任何外部工具直接消费模型拓扑。
  - 更新单元测试与示例脚本，支持前后端分离的报告导出与验证。

## [0.2.2] - 2026-09-26

### Fixed
- **可视化报告检查器细化：精确限制 Scaled Dot-Product Attention Core 检查范围 (`src/nn/visualization.zig`)**:
  - 修复在结构树中点击 **Step 2: Scaled Dot-Product Attention Core** 的 `🔍 Inspect` 按钮时误展开整个 Attention 模块（包含全部 33k 参数与 $W_q, W_k, W_v, W_o$ 线性投影）的问题。
  - 将 Step 2 卡片的检查目标指定为 `${prefix}.core`，在 `openInspector` 中自动过滤排除子模块投影算子与权重参数，只展示真正的点积注意力核心计算步骤（$Q \cdot K^T$、$\div \sqrt{d}$、掩码加法、Softmax、$\cdot V$、多头转置及重排）。
  - 为 Attention Core 提供精确的 0 参数（Parameter-free）标识、输入输出维度提取、前驱依赖（Q, K, V Projections 及 Causal Mask）与下游流向（`c_proj` 输出投影）。
  - 在 Step 2 节点卡片上增加直观的数学流向管道提示：`Q·Kᵀ ➔ ÷√d ➔ +Mask ➔ Softmax ➔ ·V ➔ Merge Heads`。

## [0.2.1] - 2026-09-26

### Changed
- **模块化重构：拆分超长核心源文件 (`src/autodiff.zig` & `src/tensor.zig`)**:
  - 将原 3,715 行的 `src/autodiff.zig` 拆分为专用子模块目录 `src/autodiff/`:
    - `src/autodiff/types.zig`: 算子类别枚举 `OpType` 与各算子上下文联合体 `OpContext`。
    - `src/autodiff/op.zig`: `Op` 结构体及前向 `forward` / 反向 `backward` 导数实现。
    - `src/autodiff/graph.zig`: `Graph` 计算图核心引擎、内存 Arena 管理、拓扑排序、权重自动初始化及报告导出方法。
    - `src/autodiff/tests.zig`: 自动微分引擎的完整单元测试集。
    - `src/autodiff.zig` 作为顶层 Facade 门面重新导出全部符号，保持 100% 外部调用向后兼容。
  - 将原 3,867 行的 `src/tensor.zig` 拆分为专用子模块目录 `src/tensor/`:
    - `src/tensor/shape.zig`: `Shape` 结构体、多维跨度推导 `computeContiguousStrides`、形状广播 `broadcastShapes` 等。
    - `src/tensor/types.zig`: 数据类型系统 `DType`、`bf16`、标量转换以及 `GenericTensor` 泛型实现。
    - `src/tensor/core.zig`: 核心 `Tensor` 结构体定义及其全部实例方法。
    - `src/tensor/ops.zig`: NumPy 风格张量工厂与高级运算函数 (`array`, `zeros`, `concat`, `split`, `where`, `svd`, `applyRoPE` 等)。
    - `src/tensor/tests.zig`: 多维张量、切片、广播、降维与线性代数测试集。
    - `src/tensor.zig` 作为顶层 Facade 门面完整重新导出所有张量组件与全局函数，无破坏性变更。

### Added
- **对齐 NumPy Tier 1 核心数组与数学 API (`src/tensor/`)**:
  - **数值序列与矩阵创建**：
    - `tensor.arange(allocator, start, stop, step)`: 生成等差数列张量。
    - `tensor.linspace(allocator, start, stop, num)`: 生成指定点数的线性等分序列。
    - `tensor.eye(allocator, N, M, k)` / `tensor.identity(allocator, n)`: 生成单位矩阵与带偏置对角线矩阵。
    - `tensor.full(allocator, shape, val)`: 快速生成指定标量值填充的张量。
  - **多维堆叠与平铺重复**：
    - `tensor.stack(allocator, inputs, axis)`: 沿新维度堆叠多个形状一致的张量。
    - `tensor.repeat(repeats, axis, allocator)`: 沿轴重复或展平重复张量元素。
    - `tensor.tile(reps, allocator)`: 沿多维复制平铺张量。
  - **逐元素向量化数学超越函数 (ufuncs)**:
    - `tensor.sqrt()`: 逐元素开平方根。
    - `tensor.exp()`: 逐元素自然指数函数。
    - `tensor.log()`: 逐元素自然对数函数。
    - `tensor.abs()`: 逐元素绝对值函数。

---

## [0.2.0] - 2026-09-26

### Added
- **交互式模型架构与计算图 HTML 可视化引擎 (`src/nn/visualization.zig`)**:
  - 提供 `graph.exportHtmlReport(file_path)` 与 `generateHtmlReport(graph, allocator)`，自动从前向图中抽取拓扑节点、张量流向、模块边界与初始化状态，生成自包含、零外部依赖的单个 HTML 报告文件。
  - **模块层级树 (Hierarchical Module Tree)**：高度还原代码编写时的模块封装层级，采用 TensorBoard 风格的紧凑卡片呈现各子模块与层级节点，支持一键展开/折叠与名称搜索过滤。
  - **残差跳跃流结构识别 (Residual Skip Connection)**：针对 Transformer Block 自动将 Attention 与 MLP 子层可视化为 `[⚡ Shortcut (Skip) x] | [⚙️ Transform Branch F(x)] -> [⊕ Residual Add: x + F(x)]` 的并联结构。
  - **独立模块弹窗探查器 (Module Inspector Modal Window)**：
    - **模块总体 I/O 维度看板 (Module Input/Output Shape Banner)**：直观展现该层整体流入与流出的张量维度（如 `[2, 16, 64] -> [2, 16, 256]`）。
    - **向量变换数学公式看板 (Mathematical Vector Transformation Formula)**：支持在代码中通过 `graph.setModuleFormula(path, formula)` 显式绑定数学公式并在界面直接高亮展示（带 `CODE SPECIFIED` 徽标），未显式指定时自动基于算子类型进行推导展示。
    - **逐算子维度流转明细表**：对模块内部执行的每个算子，清晰标注其 `输入 shape`、`输入参数 shape` (如 `weight: [64, 64]`, `bias: [1, 64]`)、`输出 shape`、执行算子类型与显存占用。
    - **入站依赖与出站下游流向追踪**：列出上下游相邻计算节点以及连接方式（顺序流 vs 残差跳线）。

---

## [0.1.0] - 2026-09-20

### Added
- **核心多维张量库 (`src/tensor.zig`)**:
  - 支持多维动态形状管理 (`Shape` 最多 8 维)，提供连续内存步长控制与转置/切片。
  - 集成硬件级 BLAS 加速（macOS Apple Accelerate 框架），支持极速高阶 GEMM (`cblas_sgemm`)。
  - 丰富张量算子集：加减乘除、广播机制、矩阵乘法、批量矩阵乘 (`BatchMatMul`)、Softmax、GELU、SiLU、RMSNorm、LayerNorm 等。
- **自动微分引擎 (`src/autodiff.zig`)**:
  - 反向传播自动微分机制 (Reverse-mode Autodiff / Autograd)，支持完整的计算图动态构图与拓扑排序调度。
  - 采用高效 Arena 内存管理策略，Batch 训练期间中间计算节点免碎片零散开销。
- **经典深度学习模型与网络组件 (`src/nn/`)**:
  - **Transformer 系列**：GPT-2 架构多层 Transformer、因果多头自注意力 (`CausalSelfAttention`)、SwiGLU、KV-Cache、LoRA 低秩微调层。
  - **循环神经网络**：单层/多层双向长短期记忆网络 (`LSTM`, `StackedLSTM`)。
  - **卷积神经网络**：二维卷积 (`Conv2D`)、转置卷积 (`ConvTranspose2D`)、最大池化 (`MaxPool2D`)。
  - **前馈网络**：全连接层 (`Linear`)、多层感知机 (`MLP`)、自定义初始化后门机制。
- **优化器套件 (`src/optim.zig`)**:
  - 实现 SGD (带动量/Nesterov)、Adam 与 AdamW 权重衰减优化器。
  - 二进制检查点持久化 (`saveCheckpoint` / `loadCheckpoint`)，包含魔数头校验 (`ZNNO`) 与张量维度自检。
- **流形学习算法 (`src/manifold.zig`)**:
  - 高效 t-SNE (t-Distributed Stochastic Neighbor Embedding) 高维特征降维可视化算法。
