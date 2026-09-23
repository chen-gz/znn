# 📋 znn (Zig Neural Network) 现状评估与演进规划 (TODO List)

本文档综合记录了对 `znn` 项目现状的全面审查诊断，以及演进为高性能、生产级深度学习框架的完整 TODO 清单。

---

## 🔍 第一部分：代码库现状诊断与不成熟之处 (Current Architecture Review)

经全面审查，`znn` 已完成基础闭环（包含 N 维张量、动态 Autograd、常用视觉/LLM 算子、AdamW 优化器、BPE 分词器与交叉验证），但在以下 **6 个维度** 仍存在演进空间：

### 1. 核心计算与张量系统 (Tensor Engine & Core Math)
* **单精度硬编码 (Lack of Multi-Precision / Generic DTypes)**：
  在 [`Tensor`](file:///Users/guangzong/Documents/znn/src/tensor.zig)、[`Graph`](file:///Users/guangzong/Documents/znn/src/autodiff.zig) 及所有网络层中，底层浮点类型全部硬编码为 `f32`。缺乏对 `f16`、`bf16`（现代 LLM 训练与推理标配）、`f64` 以及整型与量化类型（如 `int8`、`q4_0`、`q8_0`）的泛型支持 (`Tensor(T)`)。
* **维度上限静态限制与静默截断**：
  [`Shape.init`](file:///Users/guangzong/Documents/znn/src/tensor.zig) 采用固定 `[8]usize` 静态数组，在维度超过 8 维时直接 `break` 静默截断，未返回显式错误。
* **线性代数求解器简单且易退化**：
  [`solveLinearSystem`](file:///Users/guangzong/Documents/znn/src/tensor.zig) 采用 $O(N^3)$ 高斯-若尔当消元（Gauss-Jordan）。当特征维度较大或存在病态矩阵（Ill-conditioned）时，数值稳定性和性能远逊于工业级的 Cholesky 分解 ($LL^T$) 或 QR 分解。

### 2. 自动求导与计算图 (Autodiff & Graph Engine)
* **不支持高阶导数 (No Higher-Order Gradients)**：
  [`Tensor.grad`](file:///Users/guangzong/Documents/znn/src/tensor.zig) 是一维切片 (`[]f32`)，反向传播是由 [`Graph.backward`](file:///Users/guangzong/Documents/znn/src/autodiff.zig) 执行单向链式传导，梯度计算过程本身不会在图上注册为新节点。因此无法对梯度再次求导，不支持二阶优化算法、Hessian 向量积以及带梯度惩罚项的网络（如 WGAN-GP）。
* **就地操作 (In-place Ops) 与计算图安全机制缺失**：
  就地算子（如 [`mulScalar_`](file:///Users/guangzong/Documents/znn/src/tensor.zig)、[`add_`](file:///Users/guangzong/Documents/znn/src/tensor.zig)）直接通过 `assert(!requires_grad and creator == null)` 防御。缺乏计算图版本计数器（Version Counter），如果用户错误地就地修改了已被前向图引用的中间张量，会导致后续反向传播产生错误的静默计算结果。

### 3. 硬件加速与平台生态 (Hardware Acceleration & Parallelism)
* **缺少 GPU / NPU 加速后端**：
  所有前向计算与反向梯度求解均在 CPU 上执行，缺乏 Metal Compute、Vulkan Compute / WebGPU 或 CUDA 后端。
* **跨平台 BLAS 支持不平衡**：
  在 macOS 上直接动态链接 Apple Accelerate 框架以利用 AMX 协处理器 ([`cblas.zig`](file:///Users/guangzong/Documents/znn/src/cblas.zig))；但在 Linux/Windows 上仅退化为纯 Zig 8-way SIMD ([`cblas_sgemm_fallback`](file:///Users/guangzong/Documents/znn/src/cblas.zig))，未提供链接系统级 OpenBLAS、BLIS 或 Intel MKL 的构建选项。
* **多核多线程并行度不足**：
  除了 GEMM 调用了系统 BLAS/SIMD 外，大部分逐元素算子、Softmax、Conv2D、RMSNorm、LayerNorm 均为单线程嵌套 `for` 循环，未接入线程池（Thread Pool）进行多核分块并行（Multi-threaded chunking）。

### 4. 网络架构与现代 LLM 特性完备度 (Model Architectures & Operators)
* **注意力机制未应用 Memory-Efficient / FlashAttention 优化**：
  [`CausalSelfAttention`](file:///Users/guangzong/Documents/znn/src/nn/transformer.zig) 仍采用传统的 $O(T^2)$ 批次矩阵乘法并显式分配了整个注意力矩阵，长序列（Long Context）训练时内存开销大且缓存命中率低。
* **视觉算子库覆盖仍较基础**：
  目前支持基础 2D 卷积 [`Conv2D`](file:///Users/guangzong/Documents/znn/src/nn/core.zig)、[`MaxPool2D`](file:///Users/guangzong/Documents/znn/src/autodiff.zig)、转置卷积 [`ConvTranspose2D`](file:///Users/guangzong/Documents/znn/src/nn/core.zig) 和 [`AvgPool2D`](file:///Users/guangzong/Documents/znn/src/nn/normalization.zig)。仍缺少 `AdaptiveAvgPool2d`、`Conv1D`、`Conv3D` 以及高级插值与填充（Upsampling / Padding）算子。
* **训练工程基建缺失**：
  缺少自动混合精度训练（AMP / `GradScaler`）与多进程/多卡分布式训练原语（Distributed Data Parallel / AllReduce）。

### 5. 分词器与模型格式互操作性 (Tokenizer & Interoperability)
* **BPE 分词器效率与分词规则 (Tokenizer Complexity)**：
  [`BPETokenizer`](file:///Users/guangzong/Documents/znn/src/dataset.zig) 目前在编码时使用分块字符串切片匹配，时间复杂度随合并规则增加而上升，且缺少现代分词器（如 GPT-2/TikToken）使用的 Pre-tokenization 正则表达式预切分（标点、缩写和连续空白分离）。
* **业界模型格式互操作**：
  已提供原生 `SafeTensors` 读写解析器与 `ZNNO` 检查点序列化，仍缺少针对 **ONNX 导出/导入** 与 **GGUF/GGML 解析与加载** 的完整格式互操作。

### 6. 工程健壮性与代码规范 (Engineering & Robustness)
* **断言与类型化错误处理平衡**：
  在张量索引、形状校验等函数中逐步引入了错误拦截，但底层仍有部分代码使用 `std.debug.assert`。在 `ReleaseFast` 优化编译模式下，断言失效可能引发越界风险。

---

## 🎯 第二部分：分阶段开发任务清单 (Roadmap & Actionable TODO List)

| 阶段 / 优先级 | 核心目标 | 涉及模块 | 预估复杂度 |
| :--- | :--- | :--- | :--- |
| **Phase 1 (P0)** | 健壮性增强、类型化错误、自动化测试覆盖与广播系统 | `tensor.zig`, `autodiff.zig`, `scripts/` | 🟡 中等 |
| **Phase 2 (P1)** | 泛型数据类型与 Linux BLAS / 多核加速 | `tensor.zig`, `cblas.zig`, `build.zig` | 🔴 较高 |
| **Phase 3 (P2)** | 现代 LLM 架构与 FlashAttention / 视觉算子 | `nn/`, `autodiff.zig`, `optim.zig` | 🔴 较高 |
| **Phase 4 (P3)** | 分词器升级、格式互操作与 GPU 后端 | `dataset.zig`, `tools/`, `build.zig` | 🟣 复杂 |

---

### Phase 1 (P0): 健壮性增强与通用基础 (Robustness & Core Engine)

- [x] **1.1 全面规范化错误处理机制 (Replace Assertions with Typed Errors)**
  - [x] 移除公共 API、形状推导与张量索引中过度依赖的 `std.debug.assert`，全面替换为类型化错误。
  - [x] 在 [`tensor.zig`](file:///Users/guangzong/Documents/znn/src/tensor.zig) 和 [`autodiff.zig`](file:///Users/guangzong/Documents/znn/src/autodiff.zig) 中补充类型化错误处理（`ShapeMismatch`, `DimensionOutOfBounds`, `IncompatibleDimensions`, `EmptyInputs`, `InvalidSplitCount`, `UnevenSplit`, `KernelBiggerThanInput` 等）。
  - [x] 修复 [`Shape.init`](file:///Users/guangzong/Documents/znn/src/tensor.zig) 维度超过 8 时的静默截断行为，引入 `Shape.fromSlice` 显式校验与 `MaxDimensionsExceeded` 错误拦截。


- [x] **1.2 通用多维张量广播系统 (NumPy-style Multi-Dimensional Broadcasting)**
  - [x] 实现通用的形状对齐与步长映射算法 `broadcastShapes(shape1, shape2) !Shape` 与 `computeBroadcastStrides`。
  - [x] 为逐元素算子（`add`, `sub`, `mul`, `div`）支持任意维度的向后对齐与维度为 1 自动展开（包含 1D 到 8D 的虚拟步长映射）。
  - [x] 在 [`autodiff.zig`](file:///Users/guangzong/Documents/znn/src/autodiff.zig) 中实现广播算子的反向传播（自动沿广播维度进行多维梯度求和累加与降维映射）。

- [x] **1.3 自动化测试与高覆盖率保障 (Automated Coverage & Test Suite)**
  - [x] 补充核心模块极限边缘用例（奇异矩阵、全负 argmax、零方差归一化、损坏 Checkpoint 识别等），全量 104+ 测试 100% 通过。
  - [x] 建立基于 `kcov` 的端到端自动化覆盖率测试流水线（`zig build coverage` / `just coverage`），代码覆盖率超 91.7%。
  - [x] 清理 [`src/root.zig`](file:///Users/guangzong/Documents/znn/src/root.zig) 等测试中的调试输出干扰，确保 `zig build test` 拥有静默清晰的测试输出。


- [x] **1.4 评估管线轻量化改造与无梯度模式 (Lightweight Eval Pipeline & No-Grad)**
  - [x] 重构 [`engine.evaluateClassification`](file:///Users/guangzong/Documents/znn/src/engine.zig)，改用 `ArenaAllocator` + `graph = null` 纯前向模式，消除评估阶段隐式创建图和分配梯度的冗余开销。
  - [x] 在 [`autodiff.Graph`](file:///Users/guangzong/Documents/znn/src/autodiff.zig) 中增加 `graph.enable_grad: bool = true` 开关，支持在图模式下显式关闭梯度缓冲区分配与 Op 追踪。

---

### Phase 2 (P1): 泛型数据类型与跨平台高性能加速 (Generic DTypes & High-Perf Math)

- [ ] **2.1 张量泛型化与多精度支持 (Generic Tensor Type)**
  - [ ] 将核心 [`Tensor`](file:///Users/guangzong/Documents/znn/src/tensor.zig) 重构为泛型结构体 `Tensor(comptime T: type)`，支持 `f32`、`f64`、`f16`、`bf16`。
  - [ ] 在 [`autodiff.zig`](file:///Users/guangzong/Documents/znn/src/autodiff.zig) 计算图与 [`nn/`](file:///Users/guangzong/Documents/znn/src/nn/) 模块中支持泛型类型特化。
  - [ ] 提供基础低精度类型（`bf16` / `f16`）的浮点转换工具函数与 SIMD 指令封装。

- [ ] **2.2 跨平台 BLAS 支持与构建选项**
  - [ ] 在 [`build.zig`](file:///Users/guangzong/Documents/znn/build.zig) 中增加选项 `-Dblas=[accelerate|openblas|mkl|fallback]`。
  - [ ] 完善 Linux / Windows 环境下自动探测并链接系统 `libopenblas` 或 Intel MKL 的配置。
  - [ ] 优化无外部依赖时的纯 Zig Fallback GEMM 内核（进一步利用缓存分块 Cache-blocking 与 AVX2 / NEON 向量化）。

- [ ] **2.3 线性代数算法升级 (Numerical Linear Algebra)**
  - [ ] 引入 Cholesky 分解 ($A = LL^T$) 与前代/回代求解器，替代线性回归/岭回归中现有的高斯-若尔当消元法 [`solveLinearSystem`](file:///Users/guangzong/Documents/znn/src/tensor.zig)。
  - [ ] 增加 QR 分解与奇异值分解（SVD）基础支持，提升病态矩阵求解稳定性。

- [ ] **2.4 多核 CPU 并行化 (Multi-Threading)**
  - [ ] 引入轻量级工作窃取（Work-stealing）或分块线程池调度器。
  - [ ] 将 `Conv2D`、`Softmax`、`LayerNorm`、`RMSNorm` 及大矩阵逐元素算子改写为多线程分块并行。

---

### Phase 3 (P2): 现代 LLM 架构与算子完备度 (Modern LLM & Layer Architecture)

- [ ] **3.1 注意力机制升级 (Memory-Efficient & FlashAttention)**
  - [ ] 实现基于 Tiling 分块与在线 Softmax 统计更新的 **FlashAttention** 前向与反向算子（避免显式存储 $O(T^2)$ 注意力分数矩阵）。
  - [x] 在 [`CausalSelfAttention`](file:///Users/guangzong/Documents/znn/src/nn/transformer.zig) 中支持 **Grouped-Query Attention (GQA)** 与 **Multi-Query Attention (MQA)**。
  - [x] 实现 **Multi-Head Latent Attention (MLA)** 与极简低秩 `MLACache` 矩阵吸收推理。
  - [x] 实现混合专家前馈网络 **MoELayer**（DeepSeekMoE 细粒度路由、共享专家与 Top-K 门控）。
  - [x] 实现强化学习 **GRPO** (Group Relative Policy Optimization) 组内优势函数与带 KL 惩罚损失函数。

- [x] **3.2 解耦优化器与训练基建 (Optimizers & Training Infrastructure)**
  - [x] 实现解耦优化器架构：`SGDOptimizer` (Momentum), `AdamOptimizer`, `AdamWOptimizer` (Decoupled Weight Decay)。
  - [x] 学习率调度器族：`CosineScheduler` (Warmup & Min LR), `StepLRScheduler`, `LinearWarmupScheduler`, `ExponentialLRScheduler` 及多态统一接口。
  - [x] 梯度裁剪组件：全局 L2 范数裁剪 (`clipGradNorm`) 与数值范围裁剪 (`clipGradValue`)。
  - [x] 二进制检查点持久化：支持签名魔数、格式校验与张量元数据验证的 `saveCheckpoint` / `loadCheckpoint`。
  - [ ] 引入计算图版本计数器（Version Counter），增强就地修改（In-place ops）在反向传播时的安全性检测。
  - [ ] 实现自动混合精度训练（AMP）与 `GradScaler`（动态损失缩放，防止 `bf16`/`f16` 下溢）。

- [x] **3.3 经典网络与视觉算子扩展**
  - [x] 实现多维张量拼接与切分算子 `concat` 与 `split` 及 Autograd 反向梯度回传。
  - [x] 实现 `ConvTranspose2D`（转置卷积/反卷积）及其 Autograd 自动求导。
  - [x] 实现 `BatchNorm2d`, `Dropout`, `AvgPool2D`, `RMSNorm`, `LayerNorm`。
  - [x] 实现 `RNN`, `LSTM`, `StackedLSTM`, `GRU` 循环神经网络及 GAN 对抗训练网络。
  - [ ] 实现 `Conv1D`、`Conv3D`。
  - [ ] 实现 `AdaptiveAvgPool2D`、`AdaptiveMaxPool2D`、`GroupNorm`、`PixelShuffle`。

---

### Phase 4 (P3): 分词器、模型互操作与 GPU 加速 (Tokenizer, Interop & GPU)

- [ ] **4.1 生产级 BPE 分词器重构**
  - [x] 实现基础无依赖 `BPETokenizer`（支持字符与字节合并、UTF-8 边界容错编码/解码与无监督语料训练）。
  - [ ] 引入 GPT-2 / Llama 风格的 Pre-tokenization 正则表达式预切分（标点、缩写与连续空格隔离）。
  - [ ] 基于双向链表与优先队列（Min-Heap）重构 BPE 合并算法，将分词时间复杂度降至 $O(N \log N)$。
  - [ ] 支持加载业界主流 `tokenizer.json` / TikToken 词表文件。

- [ ] **4.2 工业级模型格式导入导出 (Model Formats & Interoperability)**
  - [x] 实现原生纯 Zig 零依赖 `SafeTensors` 模型权重读取与持久化存储。
  - [ ] 编写 **GGUF / GGML** 格式解析器与权重加载器（支持直接读取 LLaMA / Qwen 等开源模型权重）。
  - [ ] 提供 Python 脚本工具将 PyTorch `.pt` / `.safetensors` 权重无缝转换为 `znn` 二进制结构。

- [ ] **4.3 异构硬件与 GPU 计算后端探索 (GPU Acceleration)**
  - [ ] 探索通过 Metal Compute（macOS/iOS）执行矩阵乘法与注意力计算。
  - [ ] 探索通过 WebGPU / Vulkan Compute Shaders 实现跨平台纯图形 API 训练与推理后端。
