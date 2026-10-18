# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Apple 机器学习拓展 (Machine Learning Explore, MLX) 硬件加速后端与计算图卸载 (`src/backend/mlx.zig`, `src/backend.zig`, `src/tensor/mlx_bridge.zig`, `src/autodiff/graph.zig`, `src/autodiff/graph_nn.zig`, `src/nn/attention.zig`, `deps/mlx-c/`, `build.zig`)**:
  - 新增 `backend.mlx` 模块，通过 C API 直接接入 Apple 官方 MLX 深度学习引擎，提供对 Apple Silicon 统一内存图形处理器 (Graphics Processing Unit, GPU) 的直接硬件加速调用。
  - 将 `mlx-c` 完整源码与预构建 C-API 库集成在 `deps/mlx-c/` 子项目中，配置 `@rpath` 相对路径寻址，实现构建期零外部 Homebrew 依赖。
  - 实现 `MlxStream`（GPU/CPU 流及同步）与 `MlxArray`，支持从切片创建张量、零拷贝内存映射 (`fromSliceZeroCopy`)、数据就绪提取 (`dataFloat32`)、GPU 矩阵乘法 (`matmul`)、逐元素算子及 Softmax。
  - 接入 Apple MLX 官方手写优化的原生快速内核：`fastRmsNorm`、`fastRoPE` 旋转位置编码与 `fastScaledDotProductAttention`（FlashAttention）。
  - 新增 `src/tensor/mlx_bridge.zig` 桥接模块与 `Tensor` 快捷方法：`toMlxArray()`、`toMlxArrayZeroCopy()`（UMA 零拷贝无锁共享）、`copyFromMlxArray()` 与 `fromMlxArray()`。
  - `GraphOptions` 支持硬件设备配置：`device: backend.mlx.DeviceType = .cpu` 与 `mlx_stream: ?backend.mlx.Stream = null`。无求导执行时，`graph.matmul`、`graph.batchMatMul`、`graph.rmsNorm` 以及 `graph.scaledDotProductAttention` 自动下沉至 Apple Silicon GPU 流上并发执行。
  - `ScaledDotProductAttention.forwardCore` 与 `CausalSelfAttention` 模块在 GPU 模式下直通融合 FlashAttention 硬件内核。
  - 新增 `mlx` 基准测试套件 (`src/bench/mlx_suite.zig`)：实测 Apple Silicon GPU 矩阵乘法达到 **1.24 TFLOPS**，因果 SDPA 达到 **891.8 GFLOPS**，RMSNorm 吞吐达到 **24.8 GB/s**。
- **执行器与多核并行 (`src/backend/executor.zig`, `src/backend.zig`)**:
  - 新增 `backend` 包（分层层级 0）及 `Executor` 执行器：支持串行执行器 `Executor.serial` 与基于 Zig 0.16 `std.Io` 的线程化执行器 `Executor{ .io = threaded.io(), .max_tasks = 4 }`。
  - 提供分块数据并行 `parallelFor` 与按 chunk 索引写入、串行确定性规约的 `parallelReduce`，保证不同线程数下 100% 逐位确定性（Bitwise Determinism）。
- **Q4 split-half 块格式与张量算子 (`src/tensor/quant.zig`, `src/tensor.zig`, `src/autodiff/types.zig`, `src/autodiff/op.zig`, `src/autodiff/graph_nn.zig`)**:
  - 新增 split-half 排布的 `Q4Block`（低 4 位存储前 16 个权重，高 4 位存储后 16 个权重，对齐 ggml Q4_0 规范），激活可按连续向量加载、半字节以 16 路向量一次解包。128 核 Linux 虚拟机、Gemma 4 12B MLP 形状（3840×15360，权重 33 MB）实测：单线程 Q4 GEMV 由 v0.3.0 的 19.7 ms（1.7 GB/s）降至约 11.6 ms（2.8 GB/s，约 1.7 倍）；经执行器用满全部核心约 0.9–1.2 ms（28–36 GB/s）。
  - 新增 `Q4MatMul` 算子并挂载至 `Graph.q4MatMul`，前向经计算图注入的执行器分发至 `q4MatMulInto`，反向显式抛出 `error.UnsupportedBackward`。
  - `q4MatMulInto` 按（8 行 × 128 列）二维分块并行，分块内每个权重块只解包一次并复用于各行；每个输出元素由单一分块按固定块序累加，串行、多线程与逐行计算结果逐位一致。实测（同上形状、全部核心）T=4 / 16 / 64 行分别为 1.3 / 2.8 / 7.4 ms（每行 320 / 176 / 116 µs）；单线程多行时每行约 6.7 ms（单行 11.6 ms）。
- **Gemma 4 Q4 模块化拆分 (`src/models/gemma4_q4.zig`, `src/models/gemma4.zig`, `src/models.zig`)**:
  - 将超出 60 KB 限制的 `src/models/gemma4.zig` 拆分为 `src/models/gemma4.zig`（~25 KB）与 `src/models/gemma4_q4.zig`（~34 KB），严格保持所有非测试源码文件 `< 60 KB`。
  - 导出 `Q4Linear`、`Gemma4Q4MLP`、`Gemma4Q4Attention`、`Gemma4Q4DecoderLayer`、`Gemma4Q4ForCausalLM`、`TinyQ4Gemma4` 与 `DefaultGemma4Q4`。
- **计算图选项与执行模式配置 (`src/autodiff/graph.zig`, `src/autodiff.zig`)**:
  - 新增 `GraphOptions`（包含 `enable_grad: bool = true`、`mode: ExecutionMode = .eager` 与 `io: ?std.Io = null`）。
  - 新增 `Graph.init(backing, options)` 与 `Graph.initDefault(backing)`，提供 `graph.executor()` 获取注入的执行器。
- **基准测试套件扩充 (`src/bench/suites.zig`, `src/bench.zig`, `examples/benchmark.zig`)**:
  - 新增 `executor` 套件：测试 `parallelFor` 与 `parallelReduce` 在串行与 4 线程下的派发开销与吞吐。
  - 新增 `q4` 套件：测试单行解码 GEMV 与多行预填充 GEMM 的吞吐量（GB/s 与 GFLOPS）。
  - 新增 `decode` 套件：对比全精度 `TinyGemma4` 与量化 `TinyQ4Gemma4` 的单步自回归解码延迟与吞吐（tokens/s）。
- **异步执行与 Apple Silicon Metal 卸载设计 (`plan/ASYNC_OFFLOAD_DESIGN.md`, `plan/README.md`, `plan/TODO.md`, `doc/README.md`, `README.md`)**:
  - 新增面向 `v0.4.0`–`v0.5.x` 的设计文档：平台优先级为 Apple Silicon Metal > CPU 多核 > 其他 GPU；并发基于 Zig 0.16 `std.Io`，由应用创建 `std.Io.Threaded` 并经 `GraphOptions.io` 注入；`eager` / `deferred` 双执行模式与单一有序命令队列；统一内存零拷贝 `MetalAllocator`、Objective-C 运行时桥接与运行期编译 MSL；目标传入式 (DPS) 内核、`hostSlice` 宿主访问协议、静态内存规划与显式 `Attention` / `Q4MatMul` / `TopK` / `KVCacheAppend` 算子；Phase 0–5 路线图与验收标准。
- **包级分层依赖检查 (`tools/check_layers.zig`, `build.zig`, `AGENTS.md`)**:
  - 新增 `zig build check-layers`（同时纳入 `zig build test`）：扫描 `src/` 下全部 `@import`，要求跨包导入只指向更底层的包（`backend` ← `tensor` / `dataset` ← `autodiff` / `regression` / `manifold` ← `nn` / `cross_validation` ← `optim` / `models` ← `engine` ← `bench` ← `root`），违规时报告文件与行号；`AGENTS.md` 新增第 6 节《包级单向依赖》。

### Changed
- **热路径消除瞬态线程派发 (`src/models/gemma4_q4.zig`)**:
  - 将 `Q4Linear.forward` 与 LM head argmax 中每次调用 `std.Thread.spawn` 的实现替换为经 `GraphOptions.io` 注入的 `Executor.parallelFor` / `Executor.parallelReduce`，消除了高频创建/销毁 OS 线程的系统开销（v0.3.0 每次调用约 1.9 ms）。
- **二进制格式魔数升级 (`src/models/gemma4_q4.zig`, `tools/quantize_gemma4.zig`, `examples/gemma4.zig`)**:
  - 随着 Q4 块排布升级为 split-half，二进制模型文件魔数直接升级为 `ZNNQ4G02`，量化工具与推理加载器同步升级（遵循 AGENTS.md §3 不保留旧版本向后兼容回退）。
- **CBLAS 迁入 backend 包 (`src/backend/cblas.zig`, `tools/check_layers.zig`)**:
  - 原 `src/cblas.zig` 迁入 `src/backend/cblas.zig`；`tools/check_layers.zig` 将 `backend` 注册为分层层级 0。
- **全局统一计算图初始化入口与删除 `initNoGrad`**:
  - 彻底删除 `Graph.initNoGrad`，全库（`src/`、`examples/`、`tools/`）统一迁移为 `Graph.init(allocator, .{ .enable_grad = false })` 或 `Graph.initDefault(allocator)`。
- **消除循环导入，包级依赖改为严格单向 (`src/tensor/`, `src/autodiff/`, `src/nn/`, `src/bench/`, `src/manifold.zig`, `examples/`)**:
  - `Tensor.creator` 的类型由 `?*autodiff.Op` 改为张量层不透明句柄 `?*tensor.OpHandle`；写入使用 `op.handle()`，读取使用 `autodiff.Op.creatorOf(t)`，`tensor` 不再导入 `autodiff`。
  - `autodiff.Op` 数据结构移入 `src/autodiff/types.zig`；`Op.forward` / `Op.backward` 方法改为分发函数 `autodiff.op.forward(op, allocator)` / `autodiff.op.backward(op)`，反向内核文件只依赖 `Op` 数据结构。
  - 权重初始化模块 `src/nn/init.zig` 移至 `src/tensor/weight_init.zig`（`nn.init_mod` 及 `nn.InitMethod` 等导出不变），`Graph` 的自动初始化不再依赖 `nn`。
  - 删除 `Graph.formatJson` / `Graph.exportJson`，计算图 JSON 导出统一使用 `nn.generateJson(&graph, allocator)` / `nn.exportJson(&graph, path, allocator)`；删除 `nn.visualization` 对 `graph_ir` 的转导出与未使用的 `buildModelHierarchy` 别名。
  - 删除 `nn` 对 `engine` 训练与评估函数（`trainClassificationStep`、`evaluateClassification`、`trainStep` 等）的转导出，调用方改用 `engine.*`。
  - `createPersistentTensor` / `freePersistentTensor` 的实现移入 `src/nn/module.zig`（`nn.core` 与 `nn` 导出不变）；基准运行器 `BenchmarkRunner` 等拆分至 `src/bench/runner.zig`（`bench` 门面导出不变）；依赖 `autodiff` 的两个张量互操作测试移至 `src/autodiff/tests.zig`。

### Removed
- 删除了 `src/cblas.zig`（迁入 `src/backend/cblas.zig`）。
- 删除了 `Graph.initNoGrad`（统一为 `Graph.init` / `Graph.initDefault`）。

### Fixed
- **Q4 推理路径的并行与健壮性 (`src/autodiff/graph_nn.zig`, `src/autodiff/op.zig`, `src/tensor/quant.zig`, `src/backend/executor.zig`, `src/models/gemma4_q4.zig`, `examples/gemma4.zig`, `src/bench/suites.zig`)**:
  - `Graph.q4MatMul`（`Q4Linear.forward` 的实际路径）此前固定以串行执行器计算，忽略 `GraphOptions.io`；现改为使用 `graph.executor()`。同一形状下由 11.7 ms 降至约 0.9 ms；新增 `Graph.q4MatMul` 串行/多线程逐位一致测试与 `Q4 GEMV Graph.q4MatMul` 基准，防止回归。
  - `examples/gemma4.zig` 未把 `init.io` 传入 `GraphOptions`，整个生成过程为单线程；现传入 `.io = io`。
  - `q4MatMulInto` 改按 `shape.numel()` 推算行数并校验输入/输出连续性与形状（切片视图的 `data.len` 大于逻辑元素数），非连续输入由分配式 `q4MatMul` 与静态图重演先物化为连续副本；`q4MatMul` 改为接收执行器参数。
  - LM head argmax 改为固定 4096 词的切块粒度（262144 词表约 64 块），不再固定切 8 块。
  - `Executor` 缓存逻辑核心数，避免每次派发发起系统调用；`parallelReduce` 块数超过 256 时分批计算，局部结果保存在栈上，不再使用 `std.heap.page_allocator`（返回错误集不再包含 `OutOfMemory`）。
- **规划与架构文档同代码现状对齐 (`plan/TODO.md`, `plan/README.md`, `doc/architecture.md`)**:
  - 更正 `MoELayer` “稀疏激活”的不实描述：当前仅以 Top-K 掩码置零非选中专家的门控概率，专家前向仍为稠密计算，稀疏 Token 分发另列任务。
  - 标注 `src/models/gemma4.zig`（64,336 B）超出 60 KB 规则，并将拆分列入 Phase 0。
  - 更正多线程现状（Linux GEMM 后备为单线程；Gemma 4 Q4 每次调用临时 spawn 线程），并将 `std.Thread.Pool` 改为 Zig 0.16 `std.Io` 执行器；更正 Accelerate 为构建期链接；无梯度图不分配 `grad` 缓冲已实现，剩余工作为不记录 Op 与 `copyFromEager`；移除不存在的 `GPT.generate` 引用与“毫秒级”解码表述。

## [0.3.0] - 2026-10-16

### Added
- **高级线性代数分解算子 (`src/tensor/linalg.zig`, `src/tensor/core.zig`, `src/tensor/ops.zig`, `src/tensor.zig`, `src/root.zig`, `src/tensor/tests.zig`)**:
  - 新增精简正交三角分解 (`Tensor.qr`, `linalg.qr`)：基于豪斯霍尔德变换 (Householder Transformation)，支持任意形状 $M \times N$ 矩阵，输出具有标准正交列的正交矩阵 $Q \in \mathbb{R}^{M \times K}$ 与对角元素非负的上三角矩阵 $R \in \mathbb{R}^{K \times N}$ ($K = \min(M, N)$)，满足 $A = Q R$。
  - 新增实对称特征值分解 (`Tensor.symeig`, `linalg.symeig`)：基于雅可比旋转法 (Jacobi Rotation Method)，输出按代数值升序排列的特征值向量与标准正交特征向量矩阵 $V \in \mathbb{R}^{N \times N}$，满足 $A V = V \text{diag}(\lambda)$ 与 $V^T V = I$。
  - 新增奇异值分解 (`Tensor.svd`, `linalg.svd`)：基于单边雅可比奇异值分解 (One-Sided Jacobi SVD)，同时支持高瘦 ($M \ge N$) 与扁平 ($M < N$) 任意矩形矩阵，输出标准正交列矩阵 $U \in \mathbb{R}^{M \times K}$、降序排列非负奇异值向量 $S \in \mathbb{R}^K$ 及标准正交行矩阵 $V^T \in \mathbb{R}^{K \times N}$ ($K = \min(M, N)$)，满足 $A = U \text{diag}(S) V^T$。
- **全维度归约极值与索引算子 (`src/tensor/reductions.zig`, `src/tensor/core.zig`, `src/tensor/ops.zig`, `src/tensor.zig`, `src/tensor/tests.zig`)**:
  - 彻底泛化 `argmax` 与 `max` 从此前的 2D 限制扩展至支持任意秩 $N$ 维张量沿任意合法轴 `dim < rank` 归约。
  - 新增对偶算子 `argmin` 与 `min`，并在张量方法、`ops` 算子及顶层门面完成完整导出与测试覆盖。
- **内聚前端模型图可视化器 (`web/`, `doc/visualization-model-edge-design.md`)**:
  - 将 Schema 2.0 模型计算图可视化前端（`web/src/utils/model-graph.ts`、`web/src/utils/flow-edges.ts`、`web/src/utils/json-schema.ts`、`web/src/scripts/visualizer.ts` 及对应的 Vitest 单元测试套件）与完整设计规范 `doc/visualization-model-edge-design.md` 收归至 `znn` 仓库，直接读取 `src/nn/model_graph.schema.json` 与 `examples/` 下的导出模型图，消除跨仓文件复制。
- **专用模型库与 Gemma 4 架构实现 (`src/models/gemma4.zig`, `src/models.zig`, `src/root.zig`, `src/models/tests.zig`, `examples/gemma4.zig`, `tools/quantize_gemma4.zig`)**:
  - 新增顶层 `models` 模块与 Gemma 4 专有架构：`GemmaRMSNorm` (支持 $1 + w$ 仿射缩放)、`GemmaUnscaledRMSNorm` (无权重 RMSNorm)、`Gemma4MLP` (门控 GeLU_tanh 投影)、`Gemma4Attention` (支持混合滑动窗口与全局注意力、头部 Q/K-Norm、无权重缩放 `v_norm`、分组查询注意力 GQA、专用 Split-Half 与 Proportional RoPE 旋转位置编码与局部窗口掩码) 与 `Gemma4DecoderLayer` (前置与后置双 RMSNorm)。
  - 新增 `Gemma4ForCausalLM` 模型定义、`DefaultGemma4` 与 `TinyGemma4`，支持词嵌入缩放 $\sqrt{d_{\text{hidden}}}$、词表权重共享投影 (Tie Word Embeddings)、最终 Logit 软截断平滑 (Logit Softcapping: $c \tanh(\text{logits}/c)$)、262K 词表解码器 `Gemma4Vocabulary` 以及流式 4-bit (Q4_0) 量化推理 (`Q4Linear`, `Gemma4Q4ForCausalLM`, `zig build quantize-gemma4`, `zig build run-gemma4`)。
- **调试构建下的参数未初始化检查 (`src/nn/core.zig`, `src/nn.zig`, `src/optim.zig`, `src/nn/tests_init.zig`)**:
  - 新增 `nn.inspectParameterInit(params) ParameterInitReport` 与 `nn.warnIfParametersUninitialized(params)`：只统计可训练权重矩阵（向量形参数如偏置、归一化 γ / β 不参与判断），所有权重矩阵都全为 0 时判定为未初始化。
  - `SGDOptimizer`、`AdamOptimizer`、`AdamWOptimizer` 构造时调用该检查，Debug 构建下输出 `std.log.warn` 提示先调用 `nn.initModel` / `Module.initParameters` / `Graph.initWeights`；非 Debug 构建下为空操作。
- **独立可调用的缩放点积注意力模块 (`src/nn/attention.zig`, `src/nn.zig`, `src/root.zig`, `src/nn/tests.zig`)**:
  - 将结构体 `ScaledDotProductAttention` 升级为完整的功能模块：内置 `causal: bool = true`、`Options` (`defaultOptions()`)、`init()`、`initDefault()`、`forward()`（独立进入模块作用域）与 `forwardCore()`（在父模块子作用域内执行 $[B, H, T, D_q] \times [B, H, T_{kv}, D_q] \times [B, H, T_{kv}, D_v] \to [B, H, T, D_v]$ 核心计算，兼容 $D_q \neq D_v$）。
  - `CausalSelfAttention` 与 `MLALayer` 均内嵌 `core: ScaledDotProductAttention = .{}` 并复用其核心前向实现，消除两处重复的注意力矩阵乘法、缩放、因果掩码与 Softmax 逻辑。
- **全库注释术语全称与缩写规范 (`src/`, `AGENTS.md`)**:
  - 在 `AGENTS.md` 新增第 5 节《注释缩写规范》，并统一梳理 `src/` 下所有源码注释：凡在注释中使用专业缩写（如 SDPA、MHA、GQA、MQA、MLA、MoE、LoRA、RoPE、KVCache、RMSNorm、BN、MLP、GELU、ReLU、SiLU、RNN、LSTM、GRU、SGD、AdamW、BCE、MSE、DPO、GRPO、OLS、PCA、t-SNE、BLAS、JSON、IR 等），均先写出完整名称再接括号缩写。
- **编译期静态形状张量提升入库 (`src/tensor/static.zig`, `src/tensor.zig`, `src/root.zig`, `build.zig`)**:
  - 将编译期形状校验包装器 `StaticTensor(comptime ElemT: type, comptime dims: anytype)` 提升为正式库模块 `src/tensor/static.zig` 并通过 `tensor.StaticTensor` 与 `zig_ml.StaticTensor` 导出；在 `build.zig` 中接入 `zig build run-static`。
- **全模型训练/评估状态递归切换 (`src/nn/core.zig`, `src/nn.zig`, `src/root.zig`)**:
  - 新增 `setTrainingModel(model, is_training)`、`trainModel(model)` 与 `evalModel(model)` 编译期反射工具函数，支持一键递归切换复合模型中所有 `BatchNorm1d`、`BatchNorm2d` 与 `Dropout` 层的 `.training` 状态。
- **循环网络具名输出结构体 (`src/nn/recurrent.zig`, `src/nn.zig`)**:
  - 定义并导出 `RNNResult`、`LSTMResult`、`StackedLSTMResult` 与 `GRUResult` 具名返回结构体，作为 `RNN.forward`、`LSTM.forward`、`StackedLSTM.forwardSequence` 与 `GRU.forward` 的强类型返回值。
- **配置结构体默认构造与便捷入口补全 (`src/cross_validation.zig`, `src/nn/`, `src/optim.zig`)**:
  - 为 `CrossValidationGridSearch` 补充 `k_splits: usize = 5` 默认值、`default`、`defaultOptions()` 与 `initDefault(allocator)`；为 `GPT(config)` 补充 `initDefault` 与 `DefaultGPT` 别名；为 `Dropout` 与 `LeakyReLU` 补充 `default`、`defaultOptions()` 与 `initDefault()`；为 `CosineScheduler`、`StepLRScheduler`、`LinearWarmupScheduler`、`ExponentialLRScheduler` 补充 `initDefault()`；为 `LRScheduler`、`Nonlinearity`、`InitMethod` 补齐 `default` 与 `defaultConfig()` / `defaultOptions()`。
- **显式模块作用域与局部图导出 (`src/autodiff/graph.zig`, `src/autodiff/op.zig`, `src/tensor/core.zig`, `src/nn/visualization.zig`, `src/nn/graph_ir.zig`)**:
  - `Graph` 新增作用域栈与 `pushScope` / `popScope` / `currentScope`，以及 `Graph.enterModule` / `Graph.enterChildScope` 守卫 API；`Op.scope` 与 `Tensor.scope` 在创建时记录当前模块完整路径。
  - 每个组合模块导出自身的 `ports` / `flow_nodes` / `edges` 局部图；Reshape / Transpose / RepeatKV 折叠进边的 `transforms`；残差边通过局部图可达性判定；无参数但执行算子的叶子 (如 `core`) 导出算子级局部图。
  - 新增作用域归属测试与黄金边集测试 (`src/nn/tests_vis.zig`)。
- **可视化 JSON Schema 与图书全模型导出 (`src/nn/model_graph.schema.json`, `examples/export_model_report.zig`, `examples/export_book_models.zig`, `build.zig`)**:
  - 以 JSON Schema (draft 2020-12) 逐字段描述 schema 2.0 导出格式，通过 `visualization.SCHEMA_JSON` 嵌入；新增 `examples/minimal_model_graph.json` 最小参考 JSON。
  - 新增 `zig build run-book-models` 步骤，导出全部 18 个经典模型（`linear`、`mlp`、`rnn`、`lstm`、`stacked_lstm`、`gru`、`embedding`、`attention`、`transformer_block`、`gpt`、`swiglu`、`lora_linear`、`layernorm`、`mla`、`deepseek_moe`、`gan_generator`、`gan_discriminator`、`conv2d`）的 schema 2.0 计算图 JSON。
- **Autograd 算子扩展与 LLM 后训练 Loss 求导 (`src/autodiff/`, `src/tensor/`, `src/nn/`)**:
  - 新增 `LayerNorm`、`BatchNorm1d`、`BatchNorm2d`、`GroupNorm`、`Dropout`、`Conv1D`、`ConvTranspose1D`、`MaxPool1D`、`AvgPool1D`、`MaxPool2D`、`AvgPool2D`、`AdaptiveAvgPool1D`、`AdaptiveAvgPool2D`、`RoPE`、`MaskedCrossEntropyLoss`、`DpoLoss`、`GrpoLoss`、`Sqrt`、`Exp`、`Log`、`Abs`、`Sum`、`Mean`、`Variance`、`Where`、`MaskedFill`、`Squeeze`、`Unsqueeze`、`Slice` 等算子的计算图前向/反向传播与可视化数学公式。
  - `softmaxCrossEntropy` 与 `maskedCrossEntropyLoss` 支持 `anytype` 整型分类标签切片（`u8`、`u32`、`usize` 等），突破 256 类词表限制。
  - `Conv1D` 与 `Conv2D` 新增可配置 `stride` 与 `padding` 支持，并采用 `im2col` / `col2im` + `cblas_sgemm` 加速前向与反向传播。
- **张量双轨互操作与核心模块单测覆盖扩充 (`src/tensor/types.zig`, `src/tensor/core.zig`, `src/nn/core.zig`, `src/dataset.zig`, `src/engine.zig`, `src/autodiff/tests.zig`)**:
  - `GenericTensor(T)` 新增 `fromSlice`、`zeros`、`ones`、`full`、`reshape`、`transpose`、`add`/`sub`/`mul`、`sum`/`mean`、`eq`/`ne`/`gt`/`lt`、`any`/`all` 及与 `Tensor` 双向转换接口；`Embedding.forward` 支持直接传入整型 `GenericTensor` 或切片；`Tensor.where` 与 `Tensor.maskedFill` 支持直接接收 `BoolTensor`。
  - 补充 `dataset.loadImages`、`engine.evaluateClassification`、`engine.trainClassificationStep` 以及 `Graph.getModuleFormula` 的专项单元测试。

### Changed
- **专用模型与非通用算子模块化解耦 (`src/models/`, `src/models.zig`, `src/tensor/`, `src/autodiff/`, `src/nn/`)**:
  - 将以 Gemma 4 为代表的具体大模型架构与模型专用算子统一收敛至新建的顶层 `models` 模块 (`src/models.zig`, `src/models/gemma4.zig`)，确保 `tensor`、`autodiff` 与 `nn` 命名空间只承载纯粹的通用基础组件。
- **Safetensors 权重格式反序列化增强 (`src/nn/serialization.zig`, `src/nn/tests.zig`)**:
  - `serialization.loadTensorData` 与 `loadModel` 新增对 `BF16` (BFloat16) 与 `F16` (Float16) 浮点数据格式的原生读取与转换解析支持，并支持无序偏移量的 Safetensors 文件加载及定长张量数组字段 (`[N]*Tensor`)。
- **抽取卷积与池化内核子模块、统一 `ConvOptions` / `PoolOptions` / `RopeOptions` 配置 (`src/tensor/conv_pool.zig`, `src/tensor/nn_kernels.zig`, `src/tensor/types.zig`, `src/autodiff/`, `src/nn/`)**:
  - 将 1D/2D 卷积、转置卷积与池化内核拆分至独立子模块 `src/tensor/conv_pool.zig`，并将 `BatchNorm2d` 与 `Dropout` 前向计算统一收敛为单源 `Tensor` 内核。
  - 新增 `tensor.ConvOptions`、`tensor.PoolOptions` 与 `tensor.RopeOptions`（支持 `.interleaved` 与 `.split_half` 模式及 `partial_rotary_factor`），统一所有卷积、转置卷积、池化与 RoPE 算子接口。
- **PyTorch 风格模块协议与按字段路径自动命名 (`src/nn/module.zig`, `src/nn/core.zig`, `src/nn/serialization.zig`, `src/nn.zig`, `src/optim.zig`)** (不兼容变更):
  - 新增统一的编译期字段遍历 `nn.walk(model, visitor)`，`deinitModel`、`zeroGradModel`、`parameters`、`namedParameters`、`numParameters`、`setRequiresGrad`、`setTrainingModel`、`customInit` 分派与 Safetensors 读写全部基于它实现。
  - 新增 `nn.nameModules(&model, arena, root)` 与 `nn.enterModuleScope(graph, self)`，删除全部库层的逐层 `setName`、`deinit`、`zeroGrad` 及 `name_buf` 样板代码。
  - `nn.Module(T)` 包装器内置名称 arena，支持单输入或元组输入 `forward(graph, args)`。
- **两阶段参数构造与基于前向计算图的 `nn.initModel` 初始化分派 (`src/nn/`, `src/autodiff/graph_init.zig`)**:
  - 所有内置库层的 `init` 只分配全零参数内存并设置结构默认值（归一化 γ / β、LSTM 遗忘门偏置 `Tensor.init_constant = 1.0`），不再接收随机数参数；参数重填函数统一更名为 `resetParameters`。
  - 新增 `nn.initModel(model, graph, random)` 与 `nn.initModelWithSample`：先递归执行库外模块的 `customInit`，再调用 `Graph.initWeights(random)` 依据直接消费算子 (`Graph.computeParamFans`) 与下游激活函数自动初始化其余参数。
- **全库超长模块解耦重构（单实现文件 `< 60 KB`）(`src/tensor/`, `src/autodiff/`, `src/nn/`, `src/bench/`)**:
  - 将 `src/tensor/core.zig` 拆分为 `core.zig`、`conv_pool.zig`、`nn_kernels.zig`、`reductions.zig`、`linalg.zig` 与 `static.zig`；
  - 将 `src/autodiff/op.zig` 拆分为 `op.zig`、`backward_core.zig`、`backward_nn.zig` 与 `backward_math.zig`；
  - 将 `src/autodiff/graph.zig` 拆分为 `graph.zig`、`graph_nn.zig` 与 `graph_init.zig`；
  - 将 `src/nn/transformer.zig` 拆分为 `attention.zig`、`transformer.zig` 与 `llm.zig`；
  - 将 `src/nn/visualization.zig` 拆分为 `visualization.zig` 与 `graph_ir.zig`；
  - 将 `src/bench.zig` 拆分为 `src/bench.zig` 与 `src/bench/suites.zig`；
  - 将 `src/nn/tests.zig` 拆分为 `tests.zig`、`tests_init.zig`、`tests_module.zig` 与 `tests_vis.zig`。
- **可视化 Schema 2.0 与强类型枚举 (`src/nn/visualization.zig`, `src/nn/model_graph.schema.json`)** (不兼容 1.0):
  - 将 `NodeKind`、`NodeStatus`、`FlowNodeKind`、`EdgeKind` 全面重构为强类型枚举；移除顶层 `edges`，新增 `default_scope`、`summary.buffer_nodes` 与 `@in<k>` / `@out<k>` 端口模型。
- **Tensor ← Graph ← nn 单向解耦与算子前向去重 (`src/tensor/`, `src/autodiff/`, `src/nn/`, `src/engine.zig`)**:
  - `Tensor` 方法不再接收 `graph: ?*autodiff.Graph` 参数，只保留纯计算内核；`Graph` 新增 `initNoGrad` 与 `arenaAllocator`；所有 `nn` 模块统一为 `forward(self, graph: *autodiff.Graph, x, ...)`；`Op.forward` 复用 `Tensor` 的 Eager 算子实现。
- **核心算子、排序与回归算法性能重构 (`src/tensor/`, `src/nn/`, `src/cblas.zig`, `src/regression.zig`, `src/cross_validation.zig`, `src/manifold.zig`)**:
  - `argsort` 与 `sort` 针对连续内维实现零额外内存分配快路径；提取多维坐标递增辅助函数 `Shape.incrementCoord`。
  - `MoELayer.forward` 实现 Top-K 稀疏门控掩码与活跃专家筛选；`CausalSelfAttention.forward` 因果掩码优化为单份 `[1, 1, T, T]` 广播张量；`cblas_sgemm_fallback` 的 `NoTrans × Trans` 分支新增 8 路 `@Vector(8, f32)` SIMD 向量化快路径。
  - `regression.zig` 提取公共数据中心化逻辑 `centerData`，并将 `solveLasso` 统一委托给 `solveElasticNet(..., l1_ratio = 1.0)`；`cross_validation.zig` 优化 `kFold` 循环顺序消除重复分配与标准化；`manifold.zig` 移除 `binarySearchPerplexity` 中未使用的 `p_row` 临时缓冲区。

### Fixed
- **自动微分多输出算子反向传播重复执行与梯度相乘缺陷 (`src/autodiff/graph.zig`, `src/autodiff/backward_core.zig`, `src/autodiff/tests.zig`)**:
  - 将此前作用在 `*Tensor` 上的拓扑排序重构为以 `*Op` 算子为顶点的有向无环图深度优先遍历 (`topologicalSortOps`)，在 `backward` 与 `backwardWithGrad` 中按反向拓扑序严格保证每个多输出算子（如 `Split`）仅执行一次 `Op.backward()`，消除梯度随消费分支倍增的缺陷。
- **切片视图与非连续张量物理长度匹配与跨步物化 (`src/tensor/core.zig`, `src/tensor/nn_kernels.zig`, `src/tensor/ops.zig`, `src/tensor/shape.zig`, `src/autodiff/`)**:
  - 全面清理直接以底层物理缓冲区长度 `self.data.len` 循环导致的越界或脏数据问题，改以逻辑元素量 `self.shape.numel()` 与 `data[0..numel]` 遍历。
  - 在核心矩阵乘法 (`matmul`)、拼接 (`concat`)、分割 (`split`)、广播二元运算 (`broadcastBinaryOpRaw`)、`Graph.reshape` 及激活/归约前置检查中，对非连续切片视图张量自动物化连续副本，保障底层 BLAS 与 SIMD 访存安全。
- **全连接层 (`Linear.forward`) 任意 N 维张量与 1D 向量输入支持 (`src/nn/core.zig`, `src/nn/tests.zig`)**:
  - 新增对秩大于 2 的张量（如 Transformer 批次序列 `[B, S, D]`）的前置维度自动展平与输出形状恢复，同时支持 1D 单向量 `[in_features]` 直接前向与自动求导。
- **复合模型权重共享 (Tied Weights) 重复释放防护 (`src/nn/module.zig`, `src/nn/tests_module.zig`)**:
  - 在 `deinitModel` 反射遍历器中引入指针去重哈希集 `seen: std.AutoHashMap(*Tensor, void)`，保证共享权重的张量仅被销毁释放一次，杜绝 Double-Free 崩溃。
- **BPE 分词器多字节 UTF-8 分块边界完整性 (`src/dataset.zig`)**:
  - 优化 `BPETokenizer.encode` 分块算法：在分块后半段优先检索空白字符以保持词汇边界；若无空白字符，自动向后回退跳过 UTF-8 延续字节 (`10xxxxxx`)，确保绝不在多字节字符中间截断。
- **旋转位置编码 (RoPE) 梯度断流修复与静态图重演闭环 (`src/tensor/`, `src/autodiff/`, `src/models/gemma4.zig`, `src/autodiff/tests.zig`)**:
  - 修复计算图硬编码 `requires_grad = false` 导致的梯度阻断；在 `backward_nn.zig` 中基于正交伴随矩阵 $\mathbf{R}^T$ 完整实现 `interleaved` 与 `split_half` 的伴随逆旋转 Autograd 反向求导内核；在 `OpContext.RoPE` 中补齐 `mode` 与 `partial_rotary_factor` 使 `Op.forward` 图重演与 eager 计算完全一致。
- **门面导出完整性、训练路径与可视化边界修复 (`src/tensor/`, `src/autodiff.zig`, `src/nn/`, `scripts/coverage.sh`)**:
  - 修复 `GenericTensor(T).transposeView` 编译错误并增加维度越界检查；启用 `std.testing.refAllDecls(@This())` 编译期全量声明校验；在 `scripts/coverage.sh` 中纳入 `bench_tests`。
  - 修复 `MLALayer.forward` 未使用 `q_all`、`w_kr` 及因果注意力的早期实现，补全完整潜在多头注意力训练与求导路径。
  - 将 `markSkips` 中递归 DFS 的 `reaches` 重构为带已访问哈希表的线性 BFS；将 `RNN` / `LSTM` / `StackedLSTM` / `GRU` 的零初始状态 `h_0` / `c_0` 归类为模块常量缓冲区。

### Security
- **前端 Schema 指南 DOMPurify XSS 防护 (`web/src/schema-guide.ts`, `web/package.json`)**:
  - 使用 `DOMPurify.sanitize` 对 `schema-guide.ts` 中写入 `container.innerHTML` 的 HTML 内容进行消毒，消除潜在跨站脚本 (XSS) 风险并升级 `dompurify` 依赖。

### Removed
- 移除遗留向后兼容包装函数与被取代的旧代码/文档（遵循 `AGENTS.md` §3）：删除 `initializeWeights`、`initClean` / `initUninitialized`、`autoSequential` / `Sequential.autoInit`、`sftCrossEntropyLoss` / `sftCrossEntropyLossGraph` 别名、未使用的 `swigluForward`、`src/manifold.zig` 中重复的 `normalRandom` 实现、孤立空目录 `src/nn/visualization/` 以及过时的临时排查/阶段评审文档（`gemma_todo.md`、`doc/framework-review.md`、`plan/VISUALIZATION_GRAPH_EDGE_DESIGN.md`）。

## [0.2.7] - 2026-09-27

### Added
- **检查器多层级面包屑与上一层返回导航 (`web/index.html`, `web/js/app.js`, `web/css/style.css`, `src/nn/visualization/template.html`)**:
  - 在节点检查器 (Inspector) 顶部引入多级面包屑导航栏 (`.insp-breadcrumb-bar`)，实时解析并展示模块的全路径层级（如 `gpt / layers / 0 / attn`）。
  - 用户可随时点击面包屑中的任意前置层级直接跳回上级子模块结构视图；同时新增 `[← Back]` 按钮，一键沿历史浏览栈平滑回退到上一层。

### Fixed
- **基础算子节点数学公式精准绑定修复 (`src/autodiff/graph.zig`, `src/nn/visualization.zig`, `web/js/app.js`)**:
  - 修复注意力层内基础算子节点（如 `gpt.layers.0.attn.act_Add_20`）因前缀匹配错误继承外部 `Attention` 完整注意力公式的问题。
  - 在后端图推导及 JSON 序列化阶段，优先匹配具体算子类型（如 `Add` 匹配为 $C = A + B$），并在公式字典中注册所有基础算子类型的标准数学公式。
  - 前端 `getEffectiveFormula` 调整为算子精准匹配优先于模块前缀模糊继承，确保叶子算子节点公式精准呈现。

## [0.2.6] - 2026-09-27

### Added
- **子模块层级图 DAG 架构流程化渲染 (`web/js/app.js`, `web/css/style.css`, `src/nn/visualization/template.html`)**:
  - 将节点检查器 (Inspector) 模态窗口内的 **Direct Submodules & Internal Hierarchy** 彻底从朴素网格卡片重构为与主页面一致的 **DAG 架构流程结构图**。
  - 支持内部拓扑排序、多分支并行执行（如 Attention 内 `q_attn`、`k_attn`、`v_attn` 并行投影与 Dot-Product 聚合）以及残差跳跃链接（Residual Shortcut Highway、⊕ Converge 汇聚节点）。
  - 子模块流程卡片同样集成形状指示、参数统计，并支持直接点击卡片或按钮 (`📐 Formula`、`📁 Submodules`、`⚙️ Params`) 进行无限级下钻检查。

## [0.2.5] - 2026-09-27

### Added
- **DAG 架构流子模块钻取与参数检查抽屉 (`web/` & `src/nn/visualization/template.html`)**:
  - 在 **Architecture & Skip Connections (DAG 流程视图)** 中为每个并行分支与顺序卡片添加快捷操作按钮（`📐 Formula`、`📁 Submodules`、`⚙️ Params`）。
  - 在检查器 (Inspector) 模态窗口中引入多 Tab 导航结构：
    - `[📐 Formula & Shapes]`: 呈现输入输出张量形状及精准推导/指定的数学公式。
    - `[📁 Submodule Structure]`: 展示当前节点的所有下级子模块层级、参数统计并支持一键向下深入检查（Drill-down）。
    - `[⚙️ Parameters]`: 独立展示该模块下的全部可训练参数矩阵、元素量、内存占用与初始化策略。

### Fixed
- **模块数学公式前缀匹配与残差节点公式错乱修复 (`src/autodiff/graph.zig`, `src/nn/transformer.zig`, `web/js/app.js`)**:
  - 修复 `inferModuleFormula` 与前端 `getEffectiveFormula` 采用粗糙前缀匹配导致深层残差节点（如 `gpt.layers.0.output`）错误继承顶层 `gpt` 的 `logits = GPT(...)` 公式的问题。
  - 改为基于**最长公共前缀匹配 (Longest Prefix Match)**，并在 `TransformerBlock` 前向传播中为残差节点 `residual_attn` 与 `output` 内置精准数学公式：
    - `residual_attn`: $x_1 = x + \text{Attention}(\text{RMSNorm}(x))$
    - `output`: $x_{l+1} = x_1 + \text{MLP}(\text{RMSNorm}(x_1))$
  - 在 `visualization.zig` 中对图中的所有具名张量节点进行全量公式推导与缓存注入。

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
