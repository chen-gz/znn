# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
