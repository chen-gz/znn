# 📋 znn (Zig Neural Network) 现状评估与演进规划 (TODO List)

本文档综合记录了对 `znn` 项目现状的全面审查诊断，以及演进为高性能、生产级深度学习框架的完整 TODO 清单。

---

## 🔍 第一部分：代码库现状诊断与演进空间 (Current Architecture Review)

经全面审查与模块化重构（详见 [`doc/framework-review.md`](../doc/framework-review.md)），`znn` 已完成核心架构解耦与全模块精简（所有非测试源码文件均 `< 60 KB`，涵盖 N 维跨步张量、`GenericTensor(T)`、编译期 `StaticTensor(T, dims)`、动态 Autograd、视觉/循环/现代 LLM 算子、AdamW 优化器、BPE 分词器、交叉验证与 Schema 2.0 模型图导出），在以下 **6 个维度** 仍存在进一步演进空间：

### 1. 核心计算与张量系统 (Tensor Engine & Core Math)
* **自动微分多精度与量化张量扩展 (Mixed-Precision Autograd & Quantization)**：
  [`src/tensor/types.zig`](../src/tensor/types.zig) 已实现支持 `f32`、`f64`、`bf16`、`i32`、`i64`、`usize`、`bool` 的泛型 [`GenericTensor(T)`](../src/tensor/types.zig)，且 [`src/tensor/static.zig`](../src/tensor/static.zig) 已提供编译期形状检查的 [`StaticTensor(T, dims)`](../src/tensor/static.zig)；但动态计算图节点 [`Tensor`](../src/tensor/core.zig) 与 [`Graph`](../src/autodiff/graph.zig) 的前向/反向传播目前仍以 `f32` 为主，尚未实现原生 `f16`/`bf16` 自动混合精度梯度流与低比特量化类型（如 `int8`、`q4_0`、`q8_0`）。
* **维度上限与高维扩展**：
  [`Shape.init`](../src/tensor/shape.zig) 与 [`Shape.fromSlice`](../src/tensor/shape.zig) 采用固定 `[8]usize` 栈数组并通过 `MaxDimensionsExceeded` 进行显式边界校验，满足绝大多数深度学习场景（最高 8 维），但尚未支持超过 8 维的任意高维张量缩并。
* **正定线性代数求解器扩展**：
  [`src/tensor/core.zig`](../src/tensor/core.zig) 已支持 QR 分解 (`Tensor.qr`)、奇异值分解 (`Tensor.svd`) 与实对称特征值分解 (`Tensor.symeig`)，但 [`solveLinearSystem`](../src/tensor/ops.zig) 仍采用高斯-若尔当消元（Gauss-Jordan），尚待引入针对对称正定矩阵的 Cholesky 分解 ($LL^T$) 求解器。

### 2. 自动求导与计算图 (Autodiff & Graph Engine)
* **不支持高阶导数 (No Higher-Order Gradients)**：
  [`Tensor.grad`](../src/tensor/core.zig) 是一维切片 (`[]f32`)，反向传播由 [`Graph.backward`](../src/autodiff/graph.zig) 调度 [`backward_core.zig`](../src/autodiff/backward_core.zig)、[`backward_nn.zig`](../src/autodiff/backward_nn.zig) 与 [`backward_math.zig`](../src/autodiff/backward_math.zig) 执行单向链式传导，梯度计算本身不在图上注册为新节点，暂不支持二阶导数或 Hessian 向量积。
* **就地操作 (In-place Ops) 版本计数器与前向零梯度内存优化**：
  就地算子（如 [`mulScalar_`](../src/tensor/reductions.zig)、[`add_`](../src/tensor/reductions.zig)）目前通过 `assert(!requires_grad and creator == null)` 防御，尚缺计算图版本计数器（Version Counter）。此外，无梯度推理图 (`Graph.initNoGrad`) 下中间张量的 `grad` 切片按需跳过分配以及消除 `Op.forward` 中 `copyFromEager` 临时缓冲仍是进一步降低推理内存占用的关键优化点。

### 3. 硬件加速与平台生态 (Hardware Acceleration & Parallelism)
* **缺少 GPU / NPU 加速后端**：
  所有前向计算与反向梯度求解均在 CPU 上执行，缺乏 Metal Compute、Vulkan Compute / WebGPU 或 CUDA 后端。
* **跨平台系统级 BLAS 链接选项**：
  在 macOS 上动态绑定 Apple Accelerate 框架以利用 AMX 协处理器 ([`cblas.zig`](../src/cblas.zig))；在 Linux/Windows 上使用纯 Zig 8 路 `@Vector(8, f32)` SIMD 后备内核，尚待在 `build.zig` 中提供显式链接系统 OpenBLAS、BLIS 或 Intel MKL 的配置选项。
* **多核多线程并行度不足**：
  除 GEMM 外，逐元素算子、Softmax、Conv2D、RMSNorm、LayerNorm 仍以单线程循环执行，尚未接入 `std.Thread.Pool` 进行多核分块并行。

### 4. 网络架构与现代 LLM 特性完备度 (Model Architectures & Operators)
* **注意力机制 FlashAttention 与端到端 GPT KV-Cache 推理**：
  [`CausalSelfAttention`](../src/nn/attention.zig) 与 [`MLALayer`](../src/nn/attention.zig) 已实现单层 `forwardInference`（支持 `KVCache` 与 `MLACache`），但 [`CausalSelfAttention.forward`](../src/nn/attention.zig) 训练时仍物化 $O(T^2)$ 注意力矩阵（待引入 FlashAttention 分块在线 Softmax），且 [`GPT.generate`](../src/nn/transformer.zig) 尚未跨层串联 `[]KVCache` 实现端到端 $O(T)$ 增量解码。
* **视觉算子库高阶扩展**：
  已支持可配置 `stride`/`padding` 的 `im2col` [`Conv2D`](../src/nn/core.zig)、[`MaxPool2D`](../src/nn/core.zig)、[`ConvTranspose2D`](../src/nn/core.zig) 与 [`AvgPool2D`](../src/nn/normalization.zig)，仍待补充 `AdaptiveAvgPool2D`、`Conv1D`、`Conv3D` 与 `GroupNorm`。
* **混合精度与分布式训练原语**：
  缺少自动混合精度训练（AMP / `GradScaler`）与多卡数据并行原语。

### 5. 分词器与模型格式互操作性 (Tokenizer & Interoperability)
* **BPE 分词器复杂度与预切分规则**：
  [`BPETokenizer`](../src/dataset.zig) 编码时按合并表扫描，尚待引入优先队列（Min-Heap）$O(N \log N)$ 合并与 GPT-2/Llama 正则预切分（Pre-tokenization）。
* **跨框架模型格式互操作**：
  已提供原生零依赖 `SafeTensors` 读写解析器 ([`src/nn/serialization.zig`](../src/nn/serialization.zig)) 与优化器二进制检查点 ([`src/optim.zig`](../src/optim.zig))，仍缺少 `BF16`/`F16` Safetensors 权重自动转换、**GGUF/GGML** 加载与 **ONNX** 导出。

### 6. 训练引擎通用性 (Training Engine Generality)
* **N 维输入与多任务训练引擎扩展**：
  [`src/engine.zig`](../src/engine.zig) 目前聚焦于 2D 展平分类特征输入与 `DataLoader` 图像批次，尚待泛化支持任意 N 维张量（如 4D 图像 `[N, C, H, W]` 与 2D 序列 Token `[B, T]`）。

---

## 🎯 第二部分：分阶段开发任务清单 (Roadmap & Actionable TODO List)

| 阶段 / 优先级 | 核心目标 | 涉及模块 | 预估复杂度 |
| :--- | :--- | :--- | :--- |
| **Phase 1 (P0)** | 健壮性增强、类型化错误、自动化测试覆盖、模块化重构 (`< 60 KB`) | `tensor/`, `autodiff/`, `nn/`, `bench/` | 🟡 中等 |
| **Phase 2 (P1)** | 泛型/静态形状张量系统与 Linux BLAS / 多核加速 | `tensor/`, `cblas.zig`, `build.zig` | 🔴 较高 |
| **Phase 3 (P2)** | 现代 LLM 架构、FlashAttention、KV-Cache 解码与视觉算子 | `nn/`, `autodiff/`, `optim.zig` | 🔴 较高 |
| **Phase 4 (P3)** | 分词器升级、格式互操作与 GPU 后端 | `dataset.zig`, `tools/`, `build.zig` | 🟣 复杂 |

---

### Phase 1 (P0): 健壮性增强与通用基础 (Robustness & Core Engine)

- [x] **1.1 全面规范化错误处理机制 (Replace Assertions with Typed Errors)**
  - [x] 移除公共 API、形状推导与张量索引中过度依赖的 `std.debug.assert`，全面替换为类型化错误。
  - [x] 在 [`src/tensor/`](../src/tensor.zig) 和 [`src/autodiff/`](../src/autodiff.zig) 中补充类型化错误处理（`ShapeMismatch`, `DimensionOutOfBounds`, `IncompatibleDimensions`, `EmptyInputs`, `InvalidSplitCount`, `UnevenSplit`, `KernelBiggerThanInput` 等）。
  - [x] 修复 [`Shape.init`](../src/tensor/shape.zig) 维度超过 8 时的静默截断行为，引入 `Shape.fromSlice` 显式校验与 `MaxDimensionsExceeded` 错误拦截。

- [x] **1.2 通用多维张量广播系统 (NumPy-style Multi-Dimensional Broadcasting)**
  - [x] 实现通用的形状对齐与步长映射算法 `broadcastShapes(shape1, shape2) !Shape` 与 `computeBroadcastStrides`。
  - [x] 为逐元素算子（`add`, `sub`, `mul`, `div`）支持任意维度的向后对齐与维度为 1 自动展开（包含 1D 到 8D 的虚拟步长映射）。
  - [x] 在 [`src/autodiff/backward_core.zig`](../src/autodiff/backward_core.zig) 中实现广播算子的反向传播（自动沿广播维度进行多维梯度求和累加与降维映射）。

- [x] **1.3 自动化测试与高覆盖率保障 (Automated Coverage & Test Suite)**
  - [x] 补充核心模块极限边缘用例（奇异矩阵、全负 argmax、零方差归一化、损坏 Checkpoint 识别等），全量测试 100% 通过。
  - [x] 建立基于 `kcov` 的端到端自动化覆盖率测试流水线（`zig build coverage` / `just coverage`，覆盖 `root_tests`、`cnn_tests`、`exe_tests`、`bench_tests`），代码覆盖率超 91.7%。
  - [x] 清理 [`src/root.zig`](../src/root.zig) 等测试中的调试输出干扰，确保 `zig build test` 拥有静默清晰的测试输出。

- [x] **1.4 评估管线轻量化改造与无梯度模式 (Lightweight Eval Pipeline & No-Grad)**
  - [x] 重构 [`engine.evalClassificationStep`](../src/engine.zig) 与 [`engine.evaluateClassification`](../src/engine.zig)，改用 [`autodiff.Graph.initNoGrad(allocator)`](../src/autodiff/graph.zig) 无梯度计算图模式，消除评估阶段记录算子拓扑与反向图的开销。
  - [x] 在 [`autodiff.Graph`](../src/autodiff/graph.zig) 中提供 `Graph.initNoGrad(allocator)` 与 `graph.enable_grad: bool = true` 开关，支持在图模式下显式关闭 Op 追踪。

---

### Phase 1.5 (P0/P1): 框架架构深度审查专项修复 (Architectural Audit & Framework Fixes)

- [x] **1.5.1 补全核心层 Autograd `Op` 注册与反向传播闭环 (`normalization.zig`, `autodiff/`)**
  - [x] 在 [`OpType`](../src/autodiff/types.zig) 中新增 `LayerNorm`、`BatchNorm2d`、`Dropout`、`AvgPool2D`，并在 [`Graph`](../src/autodiff/graph_nn.zig) 与 [`Op`](../src/autodiff/backward_nn.zig) 中实现其前向/反向传播与数学公式推导。
  - [x] 统一 [`LayerNorm.forward`](../src/nn/normalization.zig)、[`BatchNorm2d.forward`](../src/nn/normalization.zig)、[`Dropout.forward`](../src/nn/normalization.zig) 与 [`AvgPool2D.forward`](../src/nn/normalization.zig) 通过 `*autodiff.Graph` 挂载 `Op` 并完成梯度回传。

- [x] **1.5.2 完善 `MLALayer.forward` 训练前向与反向求导路径 (`nn/attention.zig`)**
  - [x] 修复 [`MLALayer.forward`](../src/nn/attention.zig) 未使用 `q_all`、`w_kr` 及因果注意力的早期实现，补全基于计算图的完整潜在多头注意力（Content + RoPE + Scaled Dot-Product Attention）前向与反向传播。

- [x] **1.5.3 大词表交叉熵与 LLM 后训练 Loss 计算图集成 (`autodiff/`, `nn/llm.zig`)**
  - [x] 扩展 [`Graph.softmaxCrossEntropy`](../src/autodiff/graph_nn.zig) 支持 `usize` / `u32` 标签（突破 `[]const u8` 最多 256 类的限制）。
  - [x] 将 [`maskedCrossEntropyLoss`](../src/nn/llm.zig)、[`dpoLoss`](../src/nn/llm.zig) 与 [`grpoLoss`](../src/nn/llm.zig) 接入动态计算图 `Graph`，支持端到端 `graph.backward(loss)`。

- [x] **1.5.4 非连续张量视图（Strided Views）在广播、反向传播与 `reshape` 中的内存安全 (`shape.zig`, `core.zig`, `autodiff/`)**
  - [x] 修复 [`broadcastBinaryOpRaw`](../src/tensor/shape.zig) 及 [`Op.backward`](../src/autodiff/backward_core.zig) 中仅凭 `A_shape.eq(B_shape)` 就跳过步长与 `offset` 检查直接遍历底层切片的越界/读错数据问题。
  - [x] 修复 [`Graph.reshape`](../src/autodiff/graph.zig) 对非连续张量直接共享 `data` 切片且未维护 `is_view` 的隐患。

- [x] **1.5.5 Comptime 模型反射支持动态切片 `[]T`、定长数组 `[N]T` 与可选参数 `?*Tensor` (`nn/core.zig`)**
  - [x] 在 [`collectParametersInternal`](../src/nn/core.zig)、[`deinitModel`](../src/nn/core.zig) 与 [`zeroGradModel`](../src/nn/core.zig) 中支持结构体切片字段（如 `MoELayer` 的 `[]MLP`、`StackedLSTM` 的 `[]LSTMCell`）、定长数组及 `?*Tensor` 字段（如 `ConvTranspose2D.bias`、`LoRALinear.bias`），并正确冻结 `LoRALinear` 基础权重梯度。

- [x] **1.5.6 Safetensors 序列化覆盖度与无序加载兼容性 (`nn/serialization.zig`)**
  - [x] 在 [`writeModelTensors`](../src/nn/serialization.zig)、[`writeModelData`](../src/nn/serialization.zig) 与 [`loadModelTensors`](../src/nn/serialization.zig) 中支持 `?*Tensor`、定长张量数组 `[N]*Tensor` 与子模块切片 `[]T`。
  - [x] 移除 [`loadTensorData`](../src/nn/serialization.zig) 中对文件张量物理存储顺序必须与 Zig 字段顺序一致的强假设，改为按偏移量随机访问读取，兼容外部导出的 Safetensors 文件。

- [x] **1.5.7 消除 `Tensor`、`Graph` 与 `Op.forward` 的算子前向三重重复 (`tensor/core.zig`, `autodiff/graph.zig`, `autodiff/op.zig`)**
  - [x] 复用统一的前向计算实现，消除 [`Op.forward`](../src/autodiff/op.zig) 与 [`Graph`](../src/autodiff/graph.zig) 中重复手写的数百行前向算子代码。

- [x] **1.5.8 收敛 `Tensor` 与 `GenericTensor(T)` 双轨互操作 (`tensor/types.zig`, `tensor/core.zig`, `nn/transformer.zig`)**
  - [x] 打通 `Tensor` 与 `GenericTensor(T)` 的互操作接口：支持 [`Embedding`](../src/nn/transformer.zig) 直接接收整型索引切片/张量（无需先转为 `f32`），支持 [`Tensor.where`](../src/tensor/reductions.zig) / [`Tensor.maskedFill`](../src/tensor/reductions.zig) 接收 `BoolTensor`，并为 `GenericTensor(T)` 补齐核心逐元素与归约方法及修复 `GenericTensor.transposeView`。

- [x] **1.5.9 补全归约、数学初等函数、索引与视图算子的 Autograd 支持 (`tensor/reductions.zig`, `autodiff/`)**
  - [x] 为按轴归约 `sum`、`mean`、`variance`、初等函数 `sqrt`、`exp`、`log`、`abs`、条件选择 `where`、`maskedFill` 以及视图算子 `squeeze`、`unsqueeze`、`slice` 全面接入 `autodiff.Graph` 与反向传播。

- [x] **1.5.10 统一 `nn` 模块元数据、补全可视化 Scope 覆盖并模块化测试与构建 (`nn/`, `root.zig`, `build.zig`)**
  - [x] 为 [`RNN`](../src/nn/recurrent.zig)、[`LSTM`](../src/nn/recurrent.zig)、[`StackedLSTM`](../src/nn/recurrent.zig)、[`GRU`](../src/nn/recurrent.zig)、[`MoELayer`](../src/nn/transformer.zig)、[`MLALayer`](../src/nn/attention.zig)、[`LoRALinear`](../src/nn/llm.zig) 补齐命名接口与 `Graph.enterModule` 作用域追踪。
  - [x] 将内联测试拆分为独立模块化测试文件（`src/tests.zig`、`src/nn/tests.zig`、`src/nn/tests_init.zig`、`src/nn/tests_vis.zig`），并用表驱动循环管理 [`build.zig`](../build.zig) 中全部 18 个示例构建与运行步骤。

- [x] **1.5.11 核心算子性能与功能优化 (`nn/`, `cblas.zig`, `tensor/nn_kernels.zig`, `nn/core.zig`)**
  - [x] **`MoELayer` 稀疏激活**：修复 [`MoELayer.forward`](../src/nn/transformer.zig) 对未选中专家仍执行全量前向计算且未严格置零非 Top-K 门控概率的问题。
  - [x] **`CausalSelfAttention` 掩码广播优化**：将因果掩码从每次分配两份 $[B, n_h, T, T]$ 缩减为单份 $[1, 1, T, T]$ 广播张量。
  - [x] **`cblas_sgemm_fallback` `TransB` SIMD 加速**：为 [`cblas_sgemm_fallback`](../src/cblas.zig) 的 `NoTrans × Trans` 分支实现 8 路 `@Vector(8, f32)` 向量化内积。
  - [x] **`Conv2D` 步长/填充扩展与 `im2col + sgemm` 加速**：为 [`Conv2D`](../src/nn/core.zig) 与 [`Tensor.conv2dWithConfig`](../src/tensor/nn_kernels.zig) 增加可配置 `stride` 与 `padding` 支持，并用 `im2col` / `col2im` + `cblas_sgemm` 加速前向与反向计算。

- [x] **1.6 全库架构评审与超长模块解耦重构 (`< 60 KB` per File)**
  - [x] 输出深度架构评审报告 [`doc/framework-review.md`](../doc/framework-review.md)，覆盖目录组织、核心抽象解耦、训练/推理/优化流水线与文档/测试/基准四大维度。
  - [x] 将所有超 60 KB 的实现文件拆分为高内聚子模块：`src/tensor/{core,nn_kernels,reductions,static}.zig`、`src/autodiff/{op,backward_core,backward_nn,backward_math,graph,graph_nn,graph_init}.zig`、`src/nn/{attention,transformer,llm,visualization,graph_ir}.zig` 与 `src/bench/{suites.zig}`。
  - [x] 在 [`src/nn/core.zig`](../src/nn/core.zig) 中新增全模型训练/评估状态切换函数 `setTrainingModel`、`trainModel` 与 `evalModel`；在 [`src/nn/recurrent.zig`](../src/nn/recurrent.zig) 中定义具名返回结构体 `RNNResult`、`LSTMResult`、`StackedLSTMResult`、`GRUResult`；补齐全部 25 个配置结构体的默认值与 `defaultOptions()` / `initDefault()`。

---

### Phase 2 (P1): 泛型数据类型与跨平台高性能加速 (Generic DTypes & High-Perf Math)

- [x] **2.1 张量泛型化、编译期静态张量与多精度支持 (Generic & Static Tensor Types)**
  - [x] 引入泛型结构体 [`GenericTensor(comptime T: type)`](../src/tensor/types.zig) 与类型别名（`FloatTensor`, `DoubleTensor`, `IntTensor`, `LongTensor`, `BoolTensor`, `BFloat16Tensor`, `UsizeTensor`），完整支持 `f32`、`f64`、`i32`、`i64`、`usize`、`bool`、`bf16`。
  - [x] 在 [`src/tensor/static.zig`](../src/tensor/static.zig) 中实现编译期形状校验包装器 [`StaticTensor(comptime ElemT: type, comptime dims: anytype)`](../src/tensor/static.zig)，并在 [`src/tensor.zig`](../src/tensor.zig) 与 [`src/root.zig`](../src/root.zig) 中导出（配套示例 `zig build run-static`）。
  - [x] 实现原生 `bf16` (Brain Floating Point 16-bit) 浮点格式与 IEEE 754 互转及 `DType` 大小自省。
  - [x] 支持跨标量与张量类型安全提升转换函数 `to(DestT)` / `fromGeneric`。
  - [x] 引入跨步零拷贝切片 [`SliceRange`](../src/tensor/types.zig) 与 `slice()`，通过 `is_view` 保障生命周期安全与 `contiguous()` 紧凑化。
  - [x] 补充 `clip`, `clip_`, `sort`, `argsort`, `nonzero` 等排序检索算子。

- [ ] **2.2 跨平台 BLAS 支持与构建选项**
  - [ ] 在 [`build.zig`](../build.zig) 中增加选项 `-Dblas=[accelerate|openblas|mkl|fallback]`。
  - [ ] 完善 Linux / Windows 环境下自动探测并链接系统 `libopenblas` 或 Intel MKL 的配置。
  - [ ] 优化无外部依赖时的纯 Zig Fallback GEMM 内核（进一步利用缓存分块 Cache-blocking 与 AVX2 / NEON 向量化）。

- [ ] **2.3 线性代数算法升级 (Numerical Linear Algebra)**
  - [ ] 引入 Cholesky 分解 ($A = LL^T$) 与前代/回代求解器，替代线性回归/岭回归中现有的高斯-若尔当消元法 [`solveLinearSystem`](../src/tensor/ops.zig)。
  - [x] 增加 QR 分解 (`Tensor.qr`)、奇异值分解 (`Tensor.svd`) 与实对称特征值分解 (`Tensor.symeig`) 支持，提升病态矩阵求解稳定性。

- [ ] **2.4 多核 CPU 并行化 (Multi-Threading)**
  - [ ] 引入轻量级工作窃取（Work-stealing）或分块线程池调度器。
  - [ ] 将 `Conv2D`、`Softmax`、`LayerNorm`、`RMSNorm` 及大矩阵逐元素算子改写为多线程分块并行。

---

### Phase 3 (P2): 现代 LLM 架构与算子完备度 (Modern LLM & Layer Architecture)

- [ ] **3.1 注意力机制升级 (Memory-Efficient & FlashAttention)**
  - [ ] 实现基于 Tiling 分块与在线 Softmax 统计更新的 **FlashAttention** 前向与反向算子（避免显式存储 $O(T^2)$ 注意力分数矩阵）。
  - [x] 实现通用 **RoPE** 旋转位置编码的前向与反向 Autograd 算子（[`OpType.RoPE`](../src/autodiff/types.zig)、[`Graph.rope`](../src/autodiff/graph_nn.zig)、[`Tensor.rope`](../src/tensor/nn_kernels.zig) 统一接收 [`RopeOptions`](../src/tensor/types.zig)）并接入 [`MLALayer.forward`](../src/nn/attention.zig)。
  - [x] **补全 Split-Half 与 Partial Rotary RoPE 的静态图重演与反向传播支持 (`graph_nn.zig`, `op.zig`, `backward_nn.zig`, `models/gemma4.zig`)**：
    - [x] **修复 `requires_grad` 硬编码为 `false` 导致的训练梯度断流**：在 [`src/autodiff/graph_nn.zig`](../src/autodiff/graph_nn.zig) 中，统一 RoPE 继承 `self.enable_grad and X.requires_grad`，彻底修复反向传播梯度断流问题。
    - [x] **修复 `op.zig` 中 `.RoPE` 静态图重演（`graph.forward()`）结果错误**：在 [`OpContext.RoPE`](../src/autodiff/types.zig) 中保存 `mode` 与 `partial_rotary_factor`，并在 [`Op.forward(.RoPE)`](../src/autodiff/op.zig) 重演时传入完整配置，实现 100% 数值一致的图重演。
    - [x] **补全 `backward_nn.zig` 中 `.RoPE` 反向传播对 `split_half` 与 `partial_rotary_factor` 的伴随正交逆旋转**：在 [`src/autodiff/backward_nn.zig`](../src/autodiff/backward_nn.zig) 中实现正交伴随逆旋转求导内核 $\mathbf{R}^T$，完整支持 `interleaved` 与 `split_half` 模式及部分旋转因子，并通过有限差分数值梯度检验。
  - [ ] 为 [`CausalSelfAttention`](../src/nn/attention.zig) 与 [`GPTConfig`](../src/nn/transformer.zig) 增加可选 RoPE 位置编码开关，并将单层 `KVCache` 串联至 `TransformerBlock` / `GPT.forwardInference` 实现端到端 $O(T)$ 增量生成。
  - [ ] `KVCache` 支持动态扩容与分页块分配（Paged KV Cache），提升自回归解码吞吐。
  - [x] 在 [`CausalSelfAttention`](../src/nn/attention.zig) 中支持 **Grouped-Query Attention (GQA)** 与 **Multi-Query Attention (MQA)**。
  - [x] 实现 **Multi-Head Latent Attention (MLA)** 与极简低秩 `MLACache` 矩阵吸收推理。
  - [x] 实现混合专家前馈网络 **MoELayer**（DeepSeekMoE 细粒度路由、共享专家与 Top-K 门控）。
  - [x] 实现强化学习 **GRPO** (Group Relative Policy Optimization) 组内优势函数与带 KL 惩罚损失函数。

- [x] **3.2 解耦优化器与训练基建 (Optimizers & Training Infrastructure)**
  - [x] 实现解耦优化器架构：`SGDOptimizer` (Momentum), `AdamOptimizer`, `AdamWOptimizer` (Decoupled Weight Decay)。
  - [x] 学习率调度器族：`CosineScheduler` (Warmup & Min LR), `StepLRScheduler`, `LinearWarmupScheduler`, `ExponentialLRScheduler` 及多态统一接口。
  - [x] 梯度裁剪组件：全局 L2 范数裁剪 (`clipGradNorm`) 与数值范围裁剪 (`clipGradValue`)。
  - [x] 二进制检查点持久化：支持签名魔数、格式校验与张量元数据验证的 `saveCheckpoint` / `loadCheckpoint`。
  - [x] 全模型训练与评估模式一键切换：`setTrainingModel`、`trainModel` 与 `evalModel`。
  - [ ] 引入计算图版本计数器（Version Counter），增强就地修改（In-place ops）在反向传播时的安全性检测。
  - [ ] 实现自动混合精度训练（AMP）与 `GradScaler`（动态损失缩放，防止 `bf16`/`f16` 下溢）。

- [x] **3.3 经典网络与视觉算子扩展**
  - [x] 实现多维张量拼接与切分算子 `concat` 与 `split` 及 Autograd 反向梯度回传。
  - [x] 实现一维与二维卷积及转置卷积 `Conv1D`、`Conv2D`、`ConvTranspose1D`、`ConvTranspose2D`（统一基于 `ConvOptions`）及其 Autograd 自动求导与 `computeParamFans` 扇入/扇出推导。
  - [x] 实现一维、二维与自适应池化层 `MaxPool1D`、`AvgPool1D`、`MaxPool2D`、`AvgPool2D`（统一基于 `PoolOptions`）以及 `AdaptiveAvgPool1D`、`AdaptiveAvgPool2D`。
  - [x] 实现归一化层 `BatchNorm1d`（支持 `[N, C]` 与 `[N, C, L]`）、`BatchNorm2d`、`GroupNorm`（支持 `[N, C, ...]`）、`RMSNorm`、`LayerNorm` 与 `Dropout`。
  - [x] 实现 `RNN`, `LSTM`, `StackedLSTM`, `GRU` 循环神经网络及 GAN 对抗训练网络。
  - [ ] 按需扩展 `Conv3D`、`AdaptiveMaxPool2D`、`PixelShuffle`。

- [ ] **3.4 计算图执行与内存优化 (Graph Execution & Memory)**
  - [ ] 激活重计算（Activation / Gradient Checkpointing）：前向释放中间激活，反向时重算，以计算换显存。
  - [ ] 无梯度模式 (`Graph.initNoGrad`) 下跳过中间张量的 `grad` 缓冲区分配，并消除 `Op.forward` 中 `copyFromEager` 的临时堆分配。
  - [ ] 基于计算图生命周期分析的激活缓冲区原地复用（如 ReLU、Dropout 等非分支激活）。
  - [ ] 将 [`StaticTensor`](../src/tensor/static.zig) 融入动态计算图，得到零分配、编译期校验形状的子图。

---

### Phase 4 (P3): 分词器、模型互操作与 GPU 加速 (Tokenizer, Interop & GPU)

- [ ] **4.1 生产级 BPE 分词器重构**
  - [x] 实现基础无依赖 `BPETokenizer`（支持字符与字节合并、UTF-8 边界容错编码/解码与无监督语料训练）。
  - [ ] 引入 GPT-2 / Llama 风格的 Pre-tokenization 正则表达式预切分（标点、缩写与连续空格隔离）。
  - [ ] 基于双向链表与优先队列（Min-Heap）重构 BPE 合并算法，将分词时间复杂度降至 $O(N \log N)$。
  - [ ] 支持加载业界主流 `tokenizer.json` / TikToken 词表文件。
  - [ ] 多线程并行语料分词，加速 GB 级文本预处理。

- [ ] **4.2 工业级模型格式导入导出与推理加载 (Model Formats, Interoperability & LLM Loading)**
  - [x] 实现原生纯 Zig 零依赖 `SafeTensors` 模型权重读取与持久化存储。
  - [x] Safetensors 支持读取 `BF16` / `F16` 格式权重并自动转换为 `F32`，直接兼容 HuggingFace 预训练模型权重。
  - [x] 实现 Gemma 4 架构全套算子与 4-bit (Q4_0 Block) 量化前向推理 (`Q4Linear`, `Gemma4Q4ForCausalLM`)，支持低内存高效推理。
  - [ ] **单层/分块切片权重对齐与验证 (Layer/Block Slice Weight Verification)**：编写针对 Gemma 4 单层 DecoderLayer / Attention / MLP 的 Safetensors 权重切片加载与数值误差对齐验证工具，避免整模全量加载带来的内存压力。
  - [ ] **内存映射懒加载流式权重加载器 (`mmap` Lazy Streaming Weight Loader)**：基于 `posix.mmap` 零拷贝映射 20GB+ Safetensors 权重文件，在推理时按需分页调入各层权重，或流式量化为 Q4 格式，支持在 16GB 消费级内存下运行 12B/27B 级别大模型。
  - [ ] 编写 **GGUF / GGML** 格式解析器与权重加载器（支持直接读取 LLaMA / Qwen 等开源模型权重）。
  - [ ] 提供 Python 脚本工具将 PyTorch `.pt` / `.safetensors` 权重无缝转换为 `znn` 二进制结构。
  - [ ] 导出 ONNX 计算图与权重。

- [ ] **4.3 异构硬件与 GPU 计算后端探索 (GPU Acceleration)**
  - [ ] 探索通过 Metal Compute（macOS/iOS）执行矩阵乘法与注意力计算。
  - [ ] 探索通过 WebGPU / Vulkan Compute Shaders 实现跨平台纯图形 API 训练与推理后端。

- [ ] **4.4 训练可观测性 (Training Telemetry)**
  - [ ] 终端训练面板：实时显示 tokens/s、吞吐、估算 TFLOPS、当前学习率与剩余时间。
