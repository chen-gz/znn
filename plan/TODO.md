# 📋 znn (Zig Neural Network) 现状评估与演进规划 (TODO List)

本文档综合记录了对 `znn` 项目现状的全面审查诊断，以及演进为高性能、生产级深度学习框架的完整 TODO 清单。

---

## 🔍 第一部分：代码库现状诊断与演进空间 (Current Architecture Review)

经全面审查与模块化重构（详见 [`doc/architecture.md`](../doc/architecture.md)），`znn`（`v0.3.0`）已完成核心架构解耦与模块化清晰划分（涵盖 N 维跨步张量、`GenericTensor(T)`、编译期 `StaticTensor(T, dims)`、算子级拓扑排序动态 Autograd、1D/2D 卷积与自适应池化、循环/现代 LLM 与 Gemma 4 Q4 架构、Apple Silicon MLX GPU 端到端硬件卸载、Householder QR / Jacobi SymEig / 单边 Jacobi SVD、AdamW 优化器、BPE 分词器、交叉验证与内聚的 Schema 2.0 Web 可视化器），在以下 **6 个维度** 仍存在进一步演进空间：

### 1. 核心计算与张量系统 (Tensor Engine & Core Math)
* **自动微分多精度与量化张量扩展 (Mixed-Precision Autograd & Quantization)**：
  [`src/tensor/types.zig`](../src/tensor/types.zig) 已实现支持 `f32`、`f64`、`bf16`、`i32`、`i64`、`usize`、`bool` 的泛型 [`GenericTensor(T)`](../src/tensor/types.zig)，且 [`src/tensor/static.zig`](../src/tensor/static.zig) 已提供编译期形状检查的 [`StaticTensor(T, dims)`](../src/tensor/static.zig)；但动态计算图节点 [`Tensor`](../src/tensor/core.zig) 与 [`Graph`](../src/autodiff/graph.zig) 的前向/反向传播目前仍以 `f32` 为主，尚未实现原生 `f16`/`bf16` 自动混合精度梯度流与低比特量化类型（如 `int8`、`q4_0`、`q8_0`）；低比特量化目前仅以 Gemma 4 专用的 `Q4Block` / `Q4Linear` 形式存在于 [`src/models/gemma4.zig`](../src/models/gemma4.zig)，尚未作为张量层算子提供。
* **维度上限与高维扩展**：
  [`Shape.init`](../src/tensor/shape.zig) 与 [`Shape.fromSlice`](../src/tensor/shape.zig) 采用固定 `[8]usize` 栈数组并通过 `MaxDimensionsExceeded` 进行显式边界校验，满足绝大多数深度学习场景（最高 8 维），但尚未支持超过 8 维的任意高维张量缩并。
* **正定线性代数求解器扩展**：
  [`src/tensor/core.zig`](../src/tensor/core.zig) 与 [`src/tensor/linalg.zig`](../src/tensor/linalg.zig) 已支持基于 Householder 变换的 QR 分解 (`Tensor.qr`)、奇异值分解 (`Tensor.svd`) 与实对称特征值分解 (`Tensor.symeig`)，但 [`solveLinearSystem`](../src/tensor/ops.zig) 仍采用高斯-若尔当消元（Gauss-Jordan），尚待引入针对对称正定矩阵的 Cholesky 分解 ($LL^T$) 求解器。

### 2. 自动求导与计算图 (Autodiff & Graph Engine)
* **不支持高阶导数 (No Higher-Order Gradients)**：
  [`Tensor.grad`](../src/tensor/core.zig) 是一维切片 (`[]f32`)，反向传播由 [`Graph.backward`](../src/autodiff/graph.zig) 调度 [`backward_core.zig`](../src/autodiff/backward_core.zig)、[`backward_nn.zig`](../src/autodiff/backward_nn.zig) 与 [`backward_math.zig`](../src/autodiff/backward_math.zig) 执行单向链式传导，梯度计算本身不在图上注册为新节点，暂不支持二阶导数或 Hessian 向量积。
* **就地操作 (In-place Ops) 版本计数器与前向零梯度内存优化**：
  就地算子（如 [`mulScalar_`](../src/tensor/reductions.zig)、[`add_`](../src/tensor/reductions.zig)）目前通过 `assert(!requires_grad and creator == null)` 防御，尚缺计算图版本计数器（Version Counter）。无梯度推理图 (`Graph.initNoGrad`) 已不为中间张量分配 `grad` 缓冲，但也不记录 `Op` 节点，因此没有可供规划或重放的计算图；同时 `op.forward` 与 `runAndRecordPreallocatedOp` 经 `copyFromEager` 先算到临时张量再拷贝，输出缓冲还会被预先清零，这些是降低推理内存与带宽开销的关键优化点。

### 3. 硬件加速与平台生态 (Hardware Acceleration & Parallelism)
> 📐 **专项设计**：详见 [`plan/ASYNC_OFFLOAD_DESIGN.md`](ASYNC_OFFLOAD_DESIGN.md)（Apple Silicon Metal 优先；并发经 `io: std.Io` 注入；`eager` / `deferred` 双模式；按流有序命令队列；Phase 0–5 路线图），对应任务见第二部分 2.4 与 2.5。
* **缺少 GPU 加速后端**：
  所有前向计算与反向梯度求解均在 CPU 上执行。首要目标为 Apple Silicon Metal（统一内存零拷贝），CUDA / Vulkan / WebGPU 不在近期范围内。
* **跨平台系统级 BLAS 链接选项与 Linux Fallback GEMM 瓶颈**：
  macOS 上通过 `@extern` 与构建期 `linkFramework("Accelerate")` 调用 Accelerate，由其调度 AMX 协处理器 ([`src/backend/cblas.zig`](../src/backend/cblas.zig))；Linux/Windows 上使用单线程纯 Zig 8 路 `@Vector(8, f32)` SIMD 后备内核（`cblas_sgemm_fallback`，实测约 32~34 GFLOPS），既没有 L1/L2/L3 寄存器与缓存分块微内核（Register/Cache Blocking），也未接入 `Executor` 多线程并行或提供 `build.zig` 链接系统 OpenBLAS / BLIS / Intel MKL 的配置选项。
* **多核并行度与超越激活函数标量瓶颈**：
  逐元素算子、Softmax、Conv2D、RMSNorm、LayerNorm 与 Linux 上的 GEMM 后备内核均为单线程；特别是超越激活函数 [`GELU`](../src/tensor/nn_kernels.zig)（逐元素调用 `erff(x)`，实测仅 `0.62 GB/s`）与 [`SiLU`](../src/tensor/nn_kernels.zig) / [`Sigmoid`](../src/tensor/nn_kernels.zig)（逐元素调用 `@exp(-x)`，实测仅 `0.47 GB/s`）因标量 `libm` 调用阻碍了编译器 AVX2/NEON 向量化，吞吐比向量化 `Add`/`ReLU`（`8.4 ~ 11.6 GB/s`）慢近 20 倍，成为 Gemma 4 GeGLU MLP 层的显著 CPU 算力瓶颈。
* **小模型 L3 Cache 容量效应与 Q4 单步 Decode 反量化开销**：
  在 `Prefill` 多行批处理阶段（如 `T=16`），[`q4MatMulInto`](../src/tensor/quant.zig) 借助 `row_tile = 8` 将每个 `Q4Block` 解包一次复用于 8 行输入（实测 `66.44 GFLOPS`，`Gemma4Q4-6L Prefill` 达 `1648.8 tokens/s`，比 F32 的 `1097.1 tokens/s` 快 50%）；但在单步自回归 `DecodeStep`（`T=1` GEMV）且模型体积可完全放入 CPU L3 Cache（如 `Gemma4-6L` 约 24 MB）时，内存带宽不再是瓶颈，每 32 个权重执行一次半字节掩码/移位/`u8->f32` 解包且仅使用 1 次即丢弃的指令开销，使得小模型 Q4 单步 Decode（`390.9 tokens/s`）略低于直接做 FMA 的 F32（`484.0 tokens/s`）。需引入 AVX2/NEON 专用 `int8`/`int4` 整数点积（如 `_mm256_maddubs_epi16` / `vdotq_s32`）与激活量化（`Q8_0`）以消除逐块浮点转换开销。
* **CPU 增量解码中间张量分配与拷贝开销**：
  CPU 路径下的 [`Gemma4Q4ForCausalLM.decodeStep`](../src/models/gemma4_q4.zig) 与 [`Gemma4ForCausalLM.decodeStep`](../src/models/gemma4.zig) 在每步生成时仍为各层动态创建临时 `Tensor` 视图并执行 `contiguous()` 转置与拼接拷贝，尚待引入预分配固定形状的层间推理工作区（Scratchpad Buffer）与原位算子以消除中间缓冲分配和内存搬运。

### 4. 网络架构与现代 LLM 特性完备度 (Model Architectures & Operators)
* **注意力机制 FlashAttention 与端到端 GPT / Gemma 4 KV-Cache 推理**：
  [`CausalSelfAttention`](../src/nn/attention.zig) 与 [`MLALayer`](../src/nn/attention.zig) 已实现单层 `forwardInference`（支持 `KVCache` 与 `MLACache`），但 [`CausalSelfAttention.forward`](../src/nn/attention.zig) 训练时仍物化 $O(T^2)$ 注意力矩阵（待引入 FlashAttention 分块在线 Softmax），且 [`GPT`](../src/nn/transformer.zig)（尚无增量生成接口）与 [`Gemma4ForCausalLM`](../src/models/gemma4.zig) / [`Gemma4Q4ForCausalLM`](../src/models/gemma4.zig) 尚未跨层串联增量 `KVCache` 实现端到端 $O(T)$ 单步解码与交互式多轮对话 CLI (Chat REPL)；[`examples/gemma4.zig`](../examples/gemma4.zig) 每生成一个 Token 都重算整段序列。
* **注意力实现分散**：
  [`Gemma4Attention`](../src/models/gemma4.zig) 未复用 [`ScaledDotProductAttention`](../src/nn/attention.zig)，独立实现掩码与 Softmax，并物化 `repeatKV` 与多次转置。
* **混合专家网络 Token 级 Gather/Scatter 分发**：
  [`MoELayer.forward`](../src/nn/transformer.zig) 已实现专家级稀疏跳过（仅对当前批次内 `expert_active` 为真的被选中专家执行前向与反向计算，完全跳过未命中专家），但在单个活跃专家内部仍对批次内全部 $N$ 个 Token 矩阵乘后再乘门控掩码，后续可进一步实现按专家 Gather 命中的 Token 子集 → 专家前向 → Scatter-Add 加权合并。
* **视觉算子库高阶扩展**：
  已支持可配置 `stride`/`padding` 的 `im2col` [`Conv1D`](../src/nn/core.zig)、[`Conv2D`](../src/nn/core.zig)、[`ConvTranspose1D`](../src/nn/core.zig)、[`ConvTranspose2D`](../src/nn/core.zig)、[`MaxPool1D`](../src/nn/normalization.zig)、[`AvgPool1D`](../src/nn/normalization.zig)、[`MaxPool2D`](../src/nn/normalization.zig)、[`AvgPool2D`](../src/nn/normalization.zig)、[`AdaptiveAvgPool1D`](../src/nn/normalization.zig)、[`AdaptiveAvgPool2D`](../src/nn/normalization.zig)、[`BatchNorm1d`](../src/nn/normalization.zig)、[`BatchNorm2d`](../src/nn/normalization.zig) 与 [`GroupNorm`](../src/nn/normalization.zig)，后续可按需扩展 `Conv3D`、`AdaptiveMaxPool2D` 与 `PixelShuffle`。
* **混合精度与分布式训练原语**：
  缺少自动混合精度训练（AMP / `GradScaler`）与多卡数据并行原语。

### 5. 分词器与模型格式互操作性 (Tokenizer & Interoperability)
* **BPE 分词器复杂度与预切分规则**：
  [`BPETokenizer`](../src/dataset.zig) 已支持多字节 UTF-8 边界保护与词汇边界切分，编码时仍按合并表扫描，尚待引入优先队列（Min-Heap）$O(N \log N)$ 合并与 GPT-2/Llama 正则预切分（Pre-tokenization）。
* **跨框架模型格式互操作**：
  已提供原生零依赖 `SafeTensors`（支持 `F32`、`BF16`、`F16`）读写解析器 ([`src/nn/serialization.zig`](../src/nn/serialization.zig))、Gemma 4 流式 `Q4_0` 量化与 `mmap` 映射加载 ([`src/models/gemma4.zig`](../src/models/gemma4.zig)) 以及优化器二进制检查点 ([`src/optim.zig`](../src/optim.zig))，仍缺少通用 **GGUF/GGML** 加载与 **ONNX** 导出。

### 6. 训练引擎通用性 (Training Engine Generality)
* **N 维输入与多任务训练引擎扩展**：
  [`src/engine.zig`](../src/engine.zig) 目前聚焦于 2D 展平分类特征输入与 `DataLoader` 图像批次，尚待泛化支持任意 N 维张量（如 4D 图像 `[N, C, H, W]` 与 2D 序列 Token `[B, T]`）。

---

## 🎯 第二部分：分阶段开发任务清单 (Roadmap & Actionable TODO List)

| 阶段 / 优先级 | 核心目标 | 涉及模块 | 预估复杂度 |
| :--- | :--- | :--- | :--- |
| **Phase 0 (当前最高优先级 P0)** | Gemma 4 E4B 性能极限对齐与追平 Python MLX 官方基准 (15.5 → 32+ tokens/s) | `examples/gemma4_e4b.zig`, `backend/mlx.zig`, `backend/offload.zig` | 🔴 较高 |
| **Phase 1 (P0 已完成)** | 健壮性增强、类型化错误、自动化测试覆盖、模块化清晰重构 | `tensor/`, `autodiff/`, `nn/`, `bench/` | 🟡 中等 |
| **Phase 2 (P1)** | 泛型/静态形状张量、`io: std.Io` 多核执行器与 Apple Silicon Metal 异步卸载 | `tensor/`, `backend/`, `autodiff/`, `build.zig` | 🔴 较高 |
| **Phase 3 (P2)** | 现代 LLM 架构、FlashAttention、KV-Cache 解码与视觉算子 | `nn/`, `autodiff/`, `optim.zig` | 🔴 较高 |
| **Phase 4 (P3)** | 分词器升级、格式互操作与非 Apple GPU 后端 | `dataset.zig`, `tools/`, `build.zig` | 🟣 复杂 |

---

### 🔥 Phase 0 (当前最高优先级 P0): Gemma 4 E4B 性能极限对齐与追平 Python MLX 官方基准 (15.5 → 32+ tokens/s)

> 📌 **核心目标**：在 100% 官方词表数学精度与位输出完全一致的前提下，将 Zig 原生 Gemma 4 E4B 自回归解码速率由 **15.5 token/s 提升至 32+ token/s**，全面追平甚至反超 Apple 官方 Python MLX 基准。

#### 1. 现状实测对比基准 (Benchmark Snapshot)
| 评测指标 | 官方 Python MLX | Zig (`znn`) 优化后版本 (P0 完成) | 差异对比 |
| :--- | :--- | :--- | :--- |
| **自回归解码速率 (Decode)** | **~21.93 token/s** (单步 ~45.6 ms) | **~20.5 ~ 20.9 token/s** (单步 ~47.8 ms) | 达到官方 Python MLX 的 **95%** |
| **首字延迟 (TTFT / Prefill)** | **~350 毫秒** | **~94.2 毫秒** (127.4 token/s) | 比官方 Python MLX **快 3.7 倍** |
| **冷启动加载耗时** | **~2.8 秒** | **~8.5 秒** | 自动完成 Metal JIT PSO 编译预热 |
| **显存峰值占用** | **~5.8 GB** | **~3.9 GB** | Zig 内存占用更紧凑 |
| **输出精度与内容** | 100% 基准 | 100% 对齐 | **完全位一致 (Bit-exact)** |

#### 2. 核心性能瓶颈深度剖析与优化成果
1. **键值缓存：256 步长预分配（In-Place） vs 逐步动态拼接（Concatenate）——【已解决】**
   - 实现 `GpuKvCache` 结构体：初始预分配 256 步长 $[1, \text{num\_kv}, 256, \text{head\_dim}]$ 显存缓冲。
   - 使用 `sliceUpdate4dAxis2` 与 `slice4dAxis2` 进行原地切片定点更新，彻底消除了单步 48 次 `concatenateAxis` 显存反复重分配与历史数据重拷。
2. **根除 `bfloat16 -> float32` 全网隐式精度污染（消除每步 592 个额外 `astype` 内核与 2x KV 带宽）——【已解决】**
   - 诊断发现 C++ `mlx_array_new_float32` 创建的是强类型 `float32` 张量（不同于 Python `float` 弱类型标量）。此前 `mulScalar` 与 `gelu` 使用 `fromScalarF32` 导致 `bfloat16` 激活从 `getEmbeddings` 起被 `promote_types(bfloat16, float32)` 强制提升为 `float32`。
   - 这一隐式提升导致每步 296 次 `quantizedMatmul` 在 GPU 上动态发射 **592 个 `astype(bfloat16 -> float32)` 内核**转换权重 `scales` 与 `biases`，并将 24 层 `GpuKvCache` 提升为 `float32`。
   - 通过 `fromScalarBf16`、`fromScalarTyped` 与模型初始化期预分配的常驻 `bfloat16` 标量张量，确保全网 42 层、KV 缓存与 LM Head 100% 运行在原生 16-bit `bfloat16` 快速通道。
3. **真正的双级 GPU 异步流水线重叠（统一生成入口 `Gemma4E4BModel.generate` / `Generation.next` 内部自动启用）——【已解决】**
   - 修复此前 `best_tok_arr.asyncEval()` 后紧接 `best_tok_arr.itemUint32()` 导致 CPU 在当前步立刻同步阻塞等待 GPU 完成的假异步问题。
   - 直接将第 $t$ 步的 GPU `argmax` 输出张量（`[1]`, `uint32`）传入第 $t+1$ 步的 `takeAxis` 构建并 `asyncEval` 入队第 $t+1$ 步计算图，随后再提取第 $t$ 步词元 ID，使 CPU 图构建与 GPU 内核执行实现 100% 并发重叠。
4. **开启 MLX 静态图编译与 `compiled_gelu_mul` 算子融合——【已解决】**
   - 启用 `mlx.enableCompile()` 并通过 `compileGeluMul` (`shapeless = true`) 将 MLP 与 PLE 的 6 步 `GELU(gate) * up` 逐元素链融合为单一 Metal 内核，每步减少 **420 次 Metal 内核启动**。
5. **单步解码 (`seq_len == 1`) 零转置与零冗余节点快车道 + `smp_allocator` 极速加载——【已解决】**
   - 在 `seq_len == 1` 时消除每步 42 次 `ple_item.reshape`、132 次 `[1, 1, H, D] <-> [1, H, 1, D]` `transposeAxes` 与 42 次 `mulScalar(1.0)`；主程序改用 `std.heap.smp_allocator`。
6. **预填充阶段 (`T > 1`) 首字延迟 (TTFT) 极限压缩与算子管线预热——【已解决】**
   - **Metal 管线与融合闭包预热 (`Gemma4E4BModel.warmup`)**：此前 `examples/gemma4_e4b.zig` 未在加载阶段执行算子预热，导致首次 `prefill` 计时包含了 GEMM 模式 `quantized_matmul`、因果掩码 `fast_scaled_dot_product_attention` 与 3D/4D `compiled_gelu_mul` 闭包的首次 MSL 着色器 JIT 编译与 PSO 创建开销（约 250~300 ms）。通过在加载期执行 `T = 2` 预填充与 `T = 1` 解码预热并调用 `mlx.clearCache()`，将 JIT 编译开销彻底移出正式推理计时，真实 Prompt 首字延迟由 768ms **骤降至 94.2ms**（127.4 token/s）。
   - **消除 `T > 1` 预填充中 42 次非连续 PLE `reshape` 触发的 GPU 拷贝内核**：当 `T > 1` 时，`ple_inputs.split(42, 2)` 产生的 `[1, T, 1, 256]` 切片在维度 1 上非连续，直接对其调用 `reshape(&.{1, T, 256})` 会迫使 MLX 在每层启动 1 次 GPU 连续化拷贝内核（共 42 次拷贝内核）。通过将连续的 `ple_gate_raw` (`[1, T, 256]`) 零拷贝重塑为 `[1, T, 1, 256]` 并在 `fusedGeluMul` 输出连续张量后零拷贝重塑回 `[1, T, 256]`，消除了全部 42 次非连续拷贝内核。
   - **`GpuKvCache` 初始预分配与跨轮次零分配复用 (`resetCache`)**：在加载期一次性分配 24 个 KV 层的 256 步缓冲区，`resetCache()` 仅将 `curr_len` 归零并保留缓冲区，消除每次 `prefill` 内动态派发 48 次 `mlx_zeros` 内核的开销。
   - **Metal Wired Memory 锁定 (`configureOptimalWiredMemory`)**：调用 `mlx.configureOptimalWiredMemory()` 锁定推荐工作集上限 (`max_recommended_working_set_size`)，防止大模型运行期间显存页被系统换出。
7. **自回归解码 (Decode) 90 次 QMV 矩阵融合、43 次编译残差缩放融合与共享 KV 视图复用（每步再减 133 个 Metal 内核）——【已解决】**
   - **24 层注意力 QKV 融合 (`qkv_proj`) 与 42 层 MLP Gate+Up 融合 (`fused_gate_up`)**：通过 `Q4Weights.initFusedFromDirectBuffersCustom` 在加载期将前 24 层的 `q_proj + k_proj + v_proj`（4-bit）合并为单一矩阵、将全部 42 层的 `gate_proj + up_proj`（8-bit）合并为单一矩阵。每生成一个词元直接减少 **90 次 `quantized_matmul` (QMV) Metal 内核启动**（从 344 次降至 254 次）与 90 次 `in_norm` / `pre_mlp` 重复读取，并解决了小维度 `k_proj`/`v_proj`（`N = 512`）线程组占用率不足的瓶颈。
   - **静态编译残差相加与层缩放融合 (`compileAddMul` / `fusedAddMul`)**：将 42 层末尾的 `(x2 + ple_normed) * layer_scalar_w` 与步首的 `(ple_proj_normed + ple_tokens_arr) * input_scale_arr` 编译融合为单一 Metal 内核，每步减少 **43 次逐元素 Metal 内核启动**与 43 次中间张量显存读写。
   - **18 个键值共享层 (`24..41`) 的单步有效切片缓存 (`step_eff_k` / `step_eff_v`) 与 Prefill → Decode 首步预入队**：在层 22 与层 23 缓存当前步 `slice4dAxis2` 结果供后 18 层直接复用（省去每步 36 次冗余 `mlx_slice` 调用），并在等待预填充完成前提前入队第 1 个 Decode 步计算图，消除解码首步图构建气泡。

#### 3. 落地实施任务清单 (Actionable P0 Tasks)
- [x] **P0.1 块式预分配与切片原地更新 KVCache (`step = 256`)**：
  - 在 `Gemma4E4BModel` 中实现类似 MLX 的块式容量增长：初始分配 256 步容量 $[1, \text{num\_kv}, 256, \text{head\_dim}]$。
  - 单步解码时，通过原地切片写入新词元 KV，只有当 `curr_len % 256 == 0` 时才按需扩容 256 步，彻底消除单步 48 次 `concatenateAxis` 显存反复重分配。
- [x] **P0.2 GPU 原生 Argmax 采样算子与轻量回读**：
  - 在 `src/backend/mlx.zig` 中暴露 `mlx.Array.argmax(axis, keepdims, stream)` 算子与 `dataUint32()` / `itemUint32()`。
  - 在自回归解码主循环中，直接在 Metal GPU 上完成 `argmax` 规约，将 CPU 拷贝数据量从 1 MB (262,144 floats) 骤降至 4 字节 (1 个 uint32 ID)。
- [x] **P0.3 真双级异步执行管线 (`Generation.next` 内部流水线) 消除 GPU 同步停顿**：
  - 改造解码循环，利用 GPU `argmax` 张量直接驱动下一步 `takeAxis` 嵌入查表并先行 `asyncEval()`，随后再同步读取上一步词元，彻底消除步间 CPU-GPU 气泡。
- [x] **P0.4 逐层嵌入输入 (PLE) 步首预切分、单步零转置快车道与原生 `bfloat16` 精度锁定**：
  - 将 `getPleTokens` 与投影输出在进入 42 层循环前使用 `split(42, 2)` 一次性切分出 42 个层级张量引用，并在 `seq_len == 1` 时跳过全部冗余 `reshape` 与 `transposeAxes`。
  - 修复 `mulScalar` / `gelu` 标量类型提升并接入 `compileGeluMul` 静态融合内核，锁死全网 16-bit `bfloat16` 高速通道。
- [x] **P0.5 预填充 (Prefill / TTFT) 零拷贝 4D 门控融合、算子管线预热与输出精度 100% 位一致保障**：
  - 消除 `T > 1` 时 42 层非连续 `ple_item.reshape` 触发的 GPU 拷贝内核，预分配并跨 `resetCache()` 复用 24 层 `GpuKvCache`，增加 `warmup()` 预编译 `T > 1` / `T = 1` Metal 管线，首字延迟压缩至 94.2ms，并确保权重直通分配与 Google 官方 Python MLX 100% 位一致。
- [x] **P0.6 自回归解码 QKV / Gate+Up 量化矩阵融合、编译残差缩放融合与共享 KV 切片复用**：
  - 融合 24 层 `qkv_proj` 与 42 层 `fused_gate_up`、新增 `compileAddMul` 融合 `(a + b) * scale`、缓存层 22/23 单步 KV 切片供 18 个共享层直接复用，并在等待 Prefill 完成前预入队第 1 步 Decode，每步再消除 133 次 Metal GPU 内核启动与 36 次切片调用。

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
  - [x] **`MoELayer` 门控概率严格置零**：以 Top-K 掩码严格置零 [`MoELayer.forward`](../src/nn/transformer.zig) 中非 Top-K 专家的门控概率（专家前向仍为稠密计算，稀疏 Token 分发见 3.1）。
  - [x] **`CausalSelfAttention` 掩码广播优化**：将因果掩码从每次分配两份 $[B, n_h, T, T]$ 缩减为单份 $[1, 1, T, T]$ 广播张量。
  - [x] **`cblas_sgemm_fallback` `TransB` SIMD 加速**：为 [`cblas_sgemm_fallback`](../src/cblas.zig) 的 `NoTrans × Trans` 分支实现 8 路 `@Vector(8, f32)` 向量化内积。
  - [x] **`Conv2D` 步长/填充扩展与 `im2col + sgemm` 加速**：为 [`Conv2D`](../src/nn/core.zig) 与 [`Tensor.conv2dWithConfig`](../src/tensor/nn_kernels.zig) 增加可配置 `stride` 与 `padding` 支持，并用 `im2col` / `col2im` + `cblas_sgemm` 加速前向与反向计算。

- [x] **1.6 全库架构评审与模块化清晰解耦重构 (Modular Architecture Refactoring)**
  - [x] 完成目录组织、核心抽象解耦、训练/推理/优化流水线与文档/测试/基准四大维度的全库审查与文档同步（详见 [`doc/architecture.md`](../doc/architecture.md)）。
  - [x] 将复合实现文件按职责拆分为高内聚子模块：`src/tensor/{core,conv_pool,nn_kernels,reductions,linalg,static}.zig`、`src/autodiff/{op,backward_core,backward_nn,backward_math,graph,graph_nn,graph_init}.zig`、`src/nn/{attention,transformer,llm,visualization,graph_ir}.zig`、`src/models/{gemma4,gemma4_q4}.zig` 与 `src/bench/{suites.zig}`。
  - [x] 在 [`src/nn/core.zig`](../src/nn/core.zig) 中新增全模型训练/评估状态切换函数 `setTrainingModel`、`trainModel` 与 `evalModel`；在 [`src/nn/recurrent.zig`](../src/nn/recurrent.zig) 中定义具名返回结构体 `RNNResult`、`LSTMResult`、`StackedLSTMResult`、`GRUResult`；补齐全部配置结构体的默认值与 `defaultOptions()` / `initDefault()`。

- [x] **1.7 包级单向依赖与分层检查 (Layered Package Dependencies)**
  - [x] 消除全部跨包循环导入：`Tensor.creator` 改为张量层不透明句柄 `tensor.OpHandle`（经 `autodiff.Op.handle()` / `Op.creatorOf()` 转换）；权重初始化下沉为 `src/tensor/weight_init.zig`；计算图 JSON 导出从 `Graph` 移至 `nn.generateJson` / `nn.exportJson`；删除 `nn` 对 `engine` 训练函数的转导出。
  - [x] 消除包内非方法型循环：`nn/core` ↔ `nn/module`（持久化参数张量分配移入 `module.zig`）、`nn/visualization` ↔ `nn/graph_ir`、`autodiff/op` ↔ `backward_*`（`Op` 数据结构移入 `types.zig`，`op.forward` / `op.backward` 改为分发函数）、`bench` ↔ `bench/suites`（运行器拆为 `bench/runner.zig`）；张量层测试不再依赖 `autodiff`。
  - [x] 新增 [`tools/check_layers.zig`](../tools/check_layers.zig) 与 `zig build check-layers`（纳入 `zig build test`），逐文件检查跨包 `@import` 只指向更底层的包。
  - [ ] （可选）以编译期 mixin（`pub fn Methods(comptime T: type) type`）重构 `Tensor` 与 `Graph` 的方法实现文件，消除 `tensor/core` ↔ `ops` / `linalg` / `reductions` / `nn_kernels` / `conv_pool` 与 `autodiff/graph` ↔ `graph_nn` / `graph_init` 的包内文件级循环。

---

### Phase 2 (P1): 泛型数据类型、多核执行器与 Apple Silicon Metal (Generic DTypes, Executor & Metal)

- [x] **2.1 张量泛型化、编译期静态张量与多精度支持 (Generic & Static Tensor Types)**
  - [x] 引入泛型结构体 [`GenericTensor(comptime T: type)`](../src/tensor/types.zig) 与类型别名（`FloatTensor`, `DoubleTensor`, `IntTensor`, `LongTensor`, `BoolTensor`, `BFloat16Tensor`, `UsizeTensor`），完整支持 `f32`、`f64`、`i32`、`i64`、`usize`、`bool`、`bf16`。
  - [x] 在 [`src/tensor/static.zig`](../src/tensor/static.zig) 中实现编译期形状校验包装器 [`StaticTensor(comptime ElemT: type, comptime dims: anytype)`](../src/tensor/static.zig)，并在 [`src/tensor.zig`](../src/tensor.zig) 与 [`src/root.zig`](../src/root.zig) 中导出（配套示例 `zig build run-static`）。
  - [x] 实现原生 `bf16` (Brain Floating Point 16-bit) 浮点格式与 IEEE 754 互转及 `DType` 大小自省。
  - [x] 支持跨标量与张量类型安全提升转换函数 `to(DestT)` / `fromGeneric`。
  - [x] 引入跨步零拷贝切片 [`SliceRange`](../src/tensor/types.zig) 与 `slice()`，通过 `is_view` 保障生命周期安全与 `contiguous()` 紧凑化。
  - [x] 补充 `clip`, `clip_`, `sort`, `argsort`, `nonzero` 等排序检索算子。

- [ ] **2.2 跨平台 BLAS 支持与构建选项（Apple Silicon 优先，本项顺延至设计 Phase 5）**
  - [ ] 在 [`build.zig`](../build.zig) 中增加选项 `-Dblas=[accelerate|openblas|mkl|fallback]`。
  - [ ] 完善 Linux / Windows 环境下自动探测并链接系统 `libopenblas` 或 Intel MKL 的配置。
  - [ ] 优化无外部依赖时的纯 Zig Fallback GEMM 内核（进一步利用缓存分块 Cache-blocking 与 AVX2 / NEON 向量化）。

- [ ] **2.3 线性代数算法升级 (Numerical Linear Algebra)**
  - [ ] 引入 Cholesky 分解 ($A = LL^T$) 与前代/回代求解器，替代线性回归/岭回归中现有的高斯-若尔当消元法 [`solveLinearSystem`](../src/tensor/ops.zig)。
  - [x] 增加 QR 分解 (`Tensor.qr`)、奇异值分解 (`Tensor.svd`) 与实对称特征值分解 (`Tensor.symeig`) 支持，提升病态矩阵求解稳定性。

- [ ] **2.4 多核 CPU 并行化与 SIMD 算子优化 (Multi-Threading & SIMD Kernel Optimization，详见设计 Phase 0 / 1 / 5)**
  - [x] 基于 `std.Io`（应用创建 `std.Io.Threaded`，库内使用 `Io.Group`）实现固定粒度分块的确定性执行器 `parallelFor` / `parallelReduce`，经 `GraphOptions.io` 注入（Phase 0）。
  - [x] 用执行器替换 [`Q4Linear.forward`](../src/models/gemma4_q4.zig) 与 Gemma 4 LM head argmax 中每次调用临时 spawn 线程的实现（Phase 0）。
  - [ ] **超越激活函数 `@Vector(8, f32)` 向量化 (`tensor/nn_kernels.zig`)**：使用高精度多项式/有理函数近似（Pade / `tanh` 近似与快速向量化 `exp`）重写 [`GELU`](../src/tensor/nn_kernels.zig)（当前标量 `erff` 仅 `0.62 GB/s`）、[`SiLU`](../src/tensor/nn_kernels.zig)（当前 `0.47 GB/s`）与 [`Sigmoid`](../src/tensor/nn_kernels.zig)（当前 `0.48 GB/s`），消除逐元素标量 `libm` 调用瓶颈。
  - [ ] **Q4 单步 GEMV 激活量化 (`Q4_0 x Q8_0`) 与整数 SIMD 点积 (`tensor/quant.zig`)**：针对 `T=1` 单步自回归解码（`DecodeStep`）引入输入激活的在线 `int8` 块量化（`Q8_0`），配合 x86 AVX2 / ARM NEON 整数乘加内积指令消除逐块 `u8 -> f32` 解包转换开销。
  - [ ] 将 `Softmax`、`LayerNorm`、`RMSNorm`、`RoPE`、`Conv2D` 及大张量逐元素算子改写为按行/按块并行（Phase 1，随 DPS 内核改造）。
  - [ ] **Linux Fallback GEMM 寄存器/缓存分块微内核与多线程 (`backend/cblas.zig`)**：为 [`cblas_sgemm_fallback`](../src/backend/cblas.zig) 引入 L1/L2 Cache 二维分块（Tiling）、8×8 寄存器微内核与 `Executor.parallelFor` 多核并行（Phase 5）。

- [x] **2.5 Apple Silicon Metal / MLX 硬件加速与端到端 GPU 增量解码（详见 [`ASYNC_OFFLOAD_DESIGN.md`](ASYNC_OFFLOAD_DESIGN.md)）**
  - [x] **Phase 0 执行器与基线**：`GraphOptions` 与 `Graph.init(backing, options)` / `Graph.initDefault`（删除 `initNoGrad` 并完成全库迁移）；拆分 `src/models/gemma4.zig` 为 `src/tensor/quant.zig` 与 `src/models/gemma4_q4.zig`（保持模块边界清晰）；`Q4Block` 与内核下沉为 `src/tensor/quant.zig` 并新增 `Q4MatMul` 算子；Q4 块内 split-half 排布与文件魔数 `ZNNQ4G02`（128 核 Linux 虚拟机、3840×15360：单线程 19.7 ms → 约 11.6 ms，经 `Graph.q4MatMul` 用满全部核心约 0.9 ms）；Q4 矩阵乘按（8 行 × 128 列）二维分块并复用已解包的权重块；`src/cblas.zig` 迁入 `src/backend/` 并更新分层检查；新增 `executor` / `q4` / `decode` 基准。
  - [x] **Phase 1-2 MLX GPU 硬件加速后端与统一卸载基础设施**：
    - [x] 在 `src/backend/mlx.zig` 中接入 Apple 官方 MLX 引擎，提供 `MlxStream`、`MlxArray`、`fastRmsNorm`、`fastRoPE` 与 `fastScaledDotProductAttention`（FlashAttention）。
    - [x] 在 `src/backend/offload.zig` 中新增统一硬件卸载门面，实现 `Q4Weights` 与 `q4MatMul`，模型加载时将全部 48 层共 328 个量化层常驻统一内存，并在推理时自动下沉至 MLX GPU 执行。
    - [x] 实现 `LmHeadWeights` 语言模型输出头常驻转置矩阵，配合 32 KB `[4096]u64` 快速位图索引（Bitset），将单步词表惩罚与贪婪采样延迟从 2500 ms 压缩至 **< 1 ms**。
  - [x] **Phase 3 端到端单次求值 GPU 自回归解码计算图（Single-Evaluation GPU Decode Graph）**：
    - [x] 在 `src/backend/offload.zig` 中实现 `GpuModelContext`：将全网 48 层 Transformer 骨干、KVCache、RMSNorm 与 LM Head 完整串联在单一持久化 GPU 流（`backend.mlx.Stream`）中，消除每步 1,680 次命令流创建/释放开销。
    - [x] 彻底消除原本每步 145 次 CPU-GPU 驱动同步停顿，实现全网单一终点强制求值（Single Evaluation）。
    - [x] 解决巨型词表 LM Head 混合精度隐式上采样开销：输入隐状态通过 `astype(.bfloat16)` 与 2.01 GB 权重对齐，LM Head 延迟从 **235.6 ms 降至 20.9 ms**（提速 **11.3 倍**）。
    - [x] 修复 CPU KVCache 同步动态步长计算，Gemma 4 12B Q4 增量解码速度从 0.51 tokens/s 跃升至 **5.25 tokens/s**（稳态单步 150 ms，达到 **6.55 tokens/s**，真实提速 **超 10.3 倍**）。
  - [ ] **Phase 4 后续性能极致突破路径（推进目标：12B 模型达到 10~15+ tokens/s，E4B 已率先全面落地 A/B/C）**：
    - [x] **路径 A (权重列拼接融合)**：在模型初始化时将注意力 Q/K/V 权重在输出维度水平拼接为一个融合 QKV 量化张量，MLP gate/up 拼接为一个融合 GateUp 张量，全网每步减少 90 次 QMV Metal 内核调度开销（已在 `Gemma4E4BModel` 与 `Q4Weights.initFusedFromDirectBuffersCustom` 落地，并支持 `gemma4_e4b_direct.bin` 预融合顺序缓存）。
    - [x] **路径 B (GPU 预分配静态 KVCache 环形缓冲区)**：在 GPU 端直接分配固定 `[1, num_kv, sliding_window, head_dim]` 连续显存，单步解码超长时通过 `pos % sliding_window` 原地环形覆写（`GpuKvCache.computeWriteRange`），消除动态 `concatenateAxis` 扩容与 `slice4dAxis2` 分配。
    - [x] **路径 C (MLX 图静态编译 `mlx_compile`)**：利用 `deps/mlx-c/include/mlx/c/compile.h` 中的 `mlx_compile` 将步首 Embedding+PLE 子图 (`compiled_ple_prep`) 与 42 层注意力后全尾部子图 (`compiled_tail`) 静态编译为 Metal 融合计算管线，大幅削减运行期 C API 与命令编码器开销。
    - [ ] **路径 D (投机采样 / 草稿模型 Speculative Decoding)**：利用轻量草稿模型（如 1B/2B 级或紧凑蒸馏头）在 CPU 快速推测 4~6 个候选词元，主模型 12B 在 GPU 上单次批处理前向完成多词元并行验证与接收，实现端到端有效吞吐翻倍。


---

### Phase 3 (P2): 现代 LLM 架构与算子完备度 (Modern LLM & Layer Architecture)

- [ ] **3.1 注意力机制升级与共享算子模块化 (Attention & Shared Modular Operators)**
  - [x] **共享注意力掩码与内核 (`backend/attention_mask.zig`, `tensor/attention_kernels.zig`, `autodiff/`)**：
    - [x] 在 [`backend/attention_mask.zig`](../src/backend/attention_mask.zig) 中统一掩码定义 `Mask = union(enum) { none, causal, sliding: usize }`、`visibleRange`、`isSlidingWindowVisible`（$0 \le p - k < W$）与 `fillAdditiveMask`，供 CPU 张量内核、增量解码与 MLX GPU 卸载共同复用，并修复 Gemma 4 E4B 超窗预填充 ($T > W$) 的滑动窗口加性掩码。
    - [x] 在 [`tensor/attention_kernels.zig`](../src/tensor/attention_kernels.zig) 中实现单步注意力 `attendOneQuery` 与批处理 `attentionForward`（支持 `Tq != Tk`、因果/滑动窗口掩码与不物化 `repeatKV` 的分组查询注意力 (Grouped-Query Attention, GQA)）；统一 [`Graph.attention`](../src/autodiff/graph_nn.zig)（`OpType.Attention`）前向与反向传播，替代 `ScaledDotProductAttention.forwardCore` 的多算子拆分、[`gemma4.prefillAttention`](../src/models/gemma4.zig) 与 `decodeAttention` 的重复实现。
    - [x] 统一 [`Tensor.rmsNorm`](../src/tensor/nn_kernels.zig) / [`Graph.rmsNorm`](../src/autodiff/graph_nn.zig) 支持可选权重 `w: ?*Tensor` 与 `weight_offset`；新增一等算子 [`OpType.Softcap`](../src/autodiff/types.zig) / `Graph.softcap` 与 [`OpType.GeluTanh`](../src/autodiff/types.zig)（精确 `erf` 与 `tanh` 近似双版本 [`GeluApproximation`](../src/backend/activation.zig)，Gemma 4 全路径使用 `.tanh`）；统一 `Graph` 各算子 GPU 分支与 CPU 分支一致注册 `Op` 节点，确保计算图导出与设备无关。
  - [ ] 实现基于 Tiling 分块与在线 Softmax 统计更新的 **FlashAttention** 前向与反向算子（避免显式存储 $O(T^2)$ 注意力分数矩阵）。
  - [x] **阶段 P3 — 共享 `nn` 模块抽象 (`src/nn/`)**：
    - [x] 扩展 [`nn.RMSNorm`](../src/nn/normalization.zig) 支持可选权重 `weight: ?*Tensor` 与 `RmsNormOptions{ .eps, .weight_offset }`，删除 [`src/models/gemma4.zig`](../src/models/gemma4.zig) 中的 `GemmaRMSNorm` 与 `GemmaUnscaledRMSNorm`。
    - [x] 为 [`nn.Linear`](../src/nn/core.zig) 增加 `Options{ .bias: bool = true }`（Gemma 4 使用 `.bias = false`，消除零偏置张量与冗余加偏置算子），并移除 `MLP`、`SwiGLU`、`MLALayer` 与 Gemma 4 MLP 中冗余的手动 3D↔2D `reshape`。
    - [x] 将 `Q4Linear` 从 [`src/models/gemma4_q4.zig`](../src/models/gemma4_q4.zig) 下沉至 `src/nn/`（`nn.Q4Linear`），在 [`src/tensor/quant.zig`](../src/tensor/quant.zig) 新增 `quantizeSliceBf16` 替代 `Q4Linear.quantizeFromBF16`，定义 `Linear` 与 `Q4Linear` 共用的投影契约。
    - [x] 新增泛型门控前馈模块 `nn.GatedMLP(comptime Proj)`（支持 `.silu | .gelu | .gelu_tanh`），统一替代 `nn.SwiGLU`、`Gemma4MLP` 与 `Gemma4Q4MLP`，并使 `MoELayer` 支持可配置专家激活函数。
    - [x] 新增泛型分组查询注意力模块 `nn.GQAAttention(comptime Proj)`（支持 `qk_norm`、无缩放 `v_norm`、`rope: RopeOptions`、`scale`、`mask`、`k_eq_v`、`num_kv_heads` 及共享 `splitHeads`/`mergeHeads`），供 `CausalSelfAttention`、`Gemma4Attention` 与 `Gemma4Q4Attention` 复用。
    - [x] 新增 CPU 单层键值缓存原语 `nn.LayerKVCache`（提供 `append`、`writePrefill`、`visibleRange`、`packWindow`/`writeWindow` 及写入前 `ContextWindowExceeded` 校验），使 `nn.KVCache` 与 `Gemma4KVCache.LayerCache` 成为其轻量封装。
    - [x] 删除 [`src/nn/llm.zig`](../src/nn/llm.zig) 中重复的 `sampleTopP` / `sampleTopK`，统一使用 `backend.sampling.Sampler`，并更新 `examples/llm_training.zig` 与 `examples/train_shakespeare.zig`；将 `releaseModuleGrads` 与 `createLayerScalar` 迁入 `nn.core`。
  - [x] **阶段 P4 — 共享 MLX GPU 组合原语 (`src/backend/`)**：
    - [x] 将模型无关的 MLX 组合块（`Rope`、`projectHeads`、`gatedMlp`、`residualScale`、`finalHidden`）提取至 `src/backend/`，消除 `Gemma4GpuLayer.normHeads`/`projectQkv` 与 `gemma4_e4b.zig` 中 4 处内联 QKV 归一化/旋转重复代码；删除 `offload.qkvForwardOffload` / `mlpForwardOffload`。
    - [x] 将 `Q4Weights` 重命名为 `QuantWeights`（统一覆盖 4-bit 与 8-bit），统一量化块布局计算 `QuantLayout`（移除 `Gemma4E4BModel.q4ByteLengths`），新增 `QuantWeights.gatherRows(ids, scale)` 替代 E4B 4 处反量化查表缩放副本，并以 `LmHead = union(enum){ dense, quant }` 统一词表输出头。
    - [x] 统一 `GpuKvCache`（支持 `dtype` 参数、块式扩容、可选环形滑动窗口、投机解码 `truncate` 回滚与 CPU 窗口同步 `exportWindow`/`importWindow`），替换 12B `Gemma4GpuContext` 每步动态 `concatenate` 的 KV 数组。
    - [x] 将设备端对数几率处理（`softcapLogits`、`penalizeDevice`、`deviceCandidates`、`selectToken`、`selectTokenFromHidden`）下沉至 `src/backend/` 并以 `LogitsProcessor` 参数化，解除对 `Gemma4Config` 的依赖。
    - [x] 在 [`src/backend/mlx.zig`](../src/backend/mlx.zig) 新增类型安全的 `compileFn(comptime n_in, comptime n_out, Ctx, fn, shapeless)` 封装，统一 5 处手写 `mlx_vector_array_*` 编译闭包样板，并使 E4B 编译闭包直接调用单一前向函数。
  - [x] **阶段 P5 — 模型层组装与统一生成前端 (`src/models/`)**：
    - [x] 以 `Gemma4AttentionT(Proj)` 与 `Gemma4DecoderLayerT(Attn, Mlp)` 统一 Gemma 4 F32 与 Q4 CPU 模型的前向、预填充、单步解码与残差尾部实现。
    - [x] 基于阶段 P4 的 GPU 组合块统一组装 12B `Gemma4GpuLayer` 与 E4B `GpuLayerE4B`。
    - [x] 将 `Gemma4Config` 改为配置驱动（显式携带 `layer_types`、按注意力类型的 `rope_parameters`、`attention_k_eq_v`，并将 `kvSourceLayer`/`isKvSharedSource` 迁入配置）；在 `SamplingOptions` 中增加统一的 `repetition_window: ?usize = null`。
    - [x] 为 12B `Gemma4Q4ForCausalLM` 提供与 E4B 一致的单一生成入口 `generate(allocator, prompt, GenerateOptions) -> Generation`（`next()`），将 `examples/gemma4.zig` 中的投机解码循环内聚至生成会话中。
    - [x] 将 `GPT` / `GPTConfig` / `TransformerDecoder` / `DefaultGPT` 从 `src/nn/transformer.zig` 迁至 `src/models/`，将 `SpeculativeDraftEngine` / `promptLookup` 下沉至 `src/nn/llm.zig`。
  - [x] **阶段 P6 — 共享损失核与循环层收敛 (`src/tensor/losses.zig`, `src/nn/recurrent.zig`)**：
    - [x] 新增 `src/tensor/losses.zig`（`ceRow`、`dpoTerm`、`grpoTerm`），供 `autodiff/op.zig` 前向与 `backward_core.zig` 反向共用；扩展 `Tensor.softmaxCrossEntropy` 支持任意整型标签；删除 `src/nn/llm.zig` 中手写梯度的即时模式重复实现（`maskedCrossEntropyLoss` 手写梯度分支、即时 `dpoLoss`、`computeGRPOLoss`、`grpoLoss`），并在 `computeGroupAdvantages` 中复用 `backend.vec_math`。
    - [x] 在 [`src/nn/recurrent.zig`](../src/nn/recurrent.zig) 中提取泛型时间步展开器 `Recurrent(comptime Cell)`，供 `RNN` 与 `GRU` 共用。
  - [x] 实现通用 **RoPE** 旋转位置编码的前向与反向 Autograd 算子（[`OpType.RoPE`](../src/autodiff/types.zig)、[`Graph.rope`](../src/autodiff/graph_nn.zig)、[`Tensor.rope`](../src/tensor/nn_kernels.zig) 统一接收 [`RopeOptions`](../src/backend/rope.zig)）并接入 [`MLALayer.forward`](../src/nn/attention.zig)。
  - [x] **共享旋转表 [`RopeTable`](../src/tensor/rope.zig)**：CPU 前向内核、RoPE 反向传播 (逆旋转，不再逐元素重算幂函数与三角函数)、`MLALayer` 增量解码 (频率基数随 `MLALayer.rope` 配置，不再写死 10000) 共用一份旋转实现；频率公式与比例旋转位置编码 (Proportional RoPE) 预设 `RopeOptions.proportional` 位于 [`backend/rope.zig`](../src/backend/rope.zig)，Gemma 4 GPU `GpuRope` 的周期表也由它生成。
  - [x] **补全 Split-Half 与 Partial Rotary RoPE 的静态图重演与反向传播支持 (`graph_nn.zig`, `op.zig`, `backward_nn.zig`, `models/gemma4.zig`)**：
    - [x] **修复 `requires_grad` 硬编码为 `false` 导致的训练梯度断流**：在 [`src/autodiff/graph_nn.zig`](../src/autodiff/graph_nn.zig) 中，统一 RoPE 继承 `self.enable_grad and X.requires_grad`，彻底修复反向传播梯度断流问题。
    - [x] **修复 `op.zig` 中 `.RoPE` 静态图重演（`graph.forward()`）结果错误**：在 [`OpContext.RoPE`](../src/autodiff/types.zig) 中保存 `mode` 与 `partial_rotary_factor`，并在 [`Op.forward(.RoPE)`](../src/autodiff/op.zig) 重演时传入完整配置，实现 100% 数值一致的图重演。
    - [x] **补全 `backward_nn.zig` 中 `.RoPE` 反向传播对 `split_half` 与 `partial_rotary_factor` 的伴随正交逆旋转**：在 [`src/autodiff/backward_nn.zig`](../src/autodiff/backward_nn.zig) 中实现正交伴随逆旋转求导内核 $\mathbf{R}^T$，完整支持 `interleaved` 与 `split_half` 模式及部分旋转因子，并通过有限差分数值梯度检验。
  - [x] **Gemma 4 专用异构连续 `Gemma4KVCache` 与端到端增量自回归解码管线 (`models/gemma4.zig`, `models/gemma4_q4.zig`, `examples/gemma4.zig`)**：
    - [x] 为 40 层滑动窗口注意力 (`head_dim=256, num_kv=8`) 与 8 层全局全注意力 (`head_dim=512, num_kv=1`) 设计专用的连续块内存布局，1024 上下文仅占 ~700 MB 显存/内存。
    - [x] 在 `Gemma4Q4Attention` 与 `Gemma4Q4DecoderLayer` 中实现 `forwardPrefill` 与 `forwardDecode`，在 `Gemma4Q4ForCausalLM` 中暴露高效 `prefill` 与 `decodeStep` 接口。
    - [x] 在 `examples/gemma4.zig` 中接入完整的流式生成管线，通过实测确认输出词元流（`hello world!\n\n* 1.\n* 2.\n*`）与全量重算基线 100% 逐位精确一致。
  - [ ] **3.1.1 Gemma 4 12B 本地推理极致提速路线 (Speedup Roadmap: 0.14 → 15~25+ tokens/s)**：
    - [x] **路径 1：LM Head 262K 词表采样 GPU 卸载（消除 2.3 秒单步延迟）**：
      - 在 `src/backend/offload.zig` 中实现 `LmHeadWeights` 与 `lmHeadArgmax`，将 262,144 个 3840 维 BF16 权重 (1.92 GB) 转置常驻 GPU 显存，自回归每步仅需将 `[1, 3840]` 隐状态向量下发至 MLX GPU 流。
      - 在 Apple Silicon 统一内存架构下 GEMV 耗时仅需 **5~10 ms**（直接取得 200x 加速，完全消除 2.3 秒 CPU 软件扫描延迟），并在 `Gemma4Q4ForCausalLM` 中完成集成与端到端实测验证。
    - [x] **路径 2：Q4 矩阵向量乘 (GEMV) 的 Apple Silicon MLX GPU 硬件加速卸载（实测提速 3.1 倍）**：
      - 在 `src/backend/mlx.zig` 中封装 `Array.quantizedMatmul`，并在 `src/backend/offload.zig` 中实现 `Q4Weights` 与 `q4MatMul`，将 48 层 Transformer 的所有注意力投影与 MLP 门控投影（共 328 个量化投影层，7.59 GB 权重）转换为 MLX 原生紧凑 uint32 排列与 f16 scales/biases 常驻 GPU 显存。
    - [x] **路径 2.1：GPU 异步流式发射与批量规约 (Asynchronous GPU Queueing & Eliminating Sync Roundtrips)**：
      - 在 `src/backend/offload.zig` 中实现 `mlpForwardOffload`（将 `gate_proj`、`up_proj`、`GELU`、逐元素乘与 `down_proj` 全程在 GPU 显存内惰性串联，中间激活零 CPU 传输，单层规约 2 次同步）与 `qkvForwardOffload`（利用 `evalMultiple` 并发发射 Q、K、V 三路量化投影，单层规约 2 次同步）。
      - 全模型单步自回归解码的 GPU 同步停顿从 337 次大幅缩减至 **145 次**（降幅超 **57%**）。
    - [x] **路径 2.2：LM Head 4096-bit 快速位图与 9.6 GB 物理内存激进回收**：
      - 重构 `sampleNext` 与 `ArgmaxContext`：以仅 32 KB L1 缓存的 `[4096]u64` 快速位图索引替代原本 1600 万次嵌套比对循环，过滤延迟从 2500 ms 归零至 **< 1 ms**。
      - 消除 `loadFromMmap` 中未使用的 4.03 GB 冗余 `embed_tokens.weight` 堆张量，并在 GPU 卸载后显式 `madvise(MADV_DONTNEED)` 释放 5.6 GB 磁盘页缓存，彻底消除 16 GB 机器上的虚拟内存换页（Swap Thrashing）。
    - [x] **路径 2.3：端到端单次求值 GPU 解码计算图与 bfloat16 精度对齐（实测提速 10.3 倍，稳态 6.55 tokens/s）**：
      - 在 `src/backend/offload.zig` 中实现 `GpuModelContext`，将 48 层 Transformer 解码层、KVCache、RMSNorm 与 LM Head 全部常驻 GPU 统一内存中，单步彻底规约为**单次求值（Single Evaluation）**，消除原本每步 145 次往返开销。
      - 引入单一持久化 GPU 命令流（`Stream`）避免每步 1,680 次命令流重复创建销毁；解决 LM Head 在 Metal GEMV 内部因 float32 与 bfloat16 混合精度导致的 2.01 GB 权重矩阵隐式上采样开销（耗时从 235 ms 骤降至 20.9 ms，LM Head 提速 11.3 倍）。
      - 实测生成耗时由 27.51 秒缩短至 2.67 秒，解码速度提升至 5.83 ~ 6.55 tokens/s（提速超 10.3 倍）。
    - [x] **路径 2.4：全 GPU 序列级 Prefill 批处理卸载（消除 CPU 逐层调度与 CPU->GPU 缓存拷贝）**：
      - 在 `src/backend/offload.zig` 的 `GpuModelContext` 中实现 `prefill(prompt_tokens)`：将全部 Prompt 词元嵌入 $[1, T, 3840]$ 一次性送入 GPU，48 层内的所有注意力与 MLP 投影转换为批处理矩阵乘法（GEMM），结合带因果掩码的 GPU 原生 SDPA 并行填满 GPU KVCache。
      - 消除 CPU 逐层往返调度，并将 CPU 到 GPU 的 700 MB KVCache 内存拷贝（`syncCacheFromCpu`）彻底移除，实现 Prefill 与 Decode 全程 GPU 闭环，端到端吞吐提升至 4.58 tokens/s。
    - [ ] **路径 2.5：KVCache 预分配静态显存与切片原地更新（消除 96 次/步动态张量重分配）**：
      - 在 `GpuModelContext.init` 中为 48 层的 Key 和 Value 静态分配固定显存上限缓冲 $[1, \text{num\_kv}, \text{max\_seq\_len}, \text{head\_dim}]$。
      - 增量自回归时，以 MLX 官方原生的 `mlx_slice_update` 原地写入当前 `pos` 切片槽位，替代每步 96 次的 `concatenateAxis` 动态数组分配与历史缓存重复拷贝，消除显存碎片与内存带宽浪费。
    - [ ] **路径 2.6：MLX JIT 计算图编译（`mlx_compile` / Metal Pipeline PSO 预热）**：
      - 接入 MLX 的 `mlx_compile`，将单步增量解码前向闭包编译为固定的 Metal Compute Pipeline，消除 CPU 侧每步 700+ 次 C-API 动态节点构建与调度停顿。
    - [ ] **路径 2.7：显存带宽受限算子深度融合（RMSNorm + Q4 GEMV 深度内核融合）**：
      - 将输入 RMSNorm 缩放与后续 Q4Block 反量化矩阵乘在 Metal Shader 层面垂直融合，减少中间激活张量在全局显存中的往返读写，使显存读取效率无限逼近 Apple Silicon 物理带宽上限，向 15~25+ tokens/s 冲刺。
  - [ ] 为 [`examples/gemma4.zig`](../examples/gemma4.zig) 增加多轮对话交互式 CLI (Chat REPL) 与流式 Token 实时打字机输出。
  - [ ] `KVCache` 支持动态扩容与分页块分配（Paged KV Cache），提升自回归解码吞吐。
  - [x] 在 [`CausalSelfAttention`](../src/nn/attention.zig) 中支持 **Grouped-Query Attention (GQA)** 与 **Multi-Query Attention (MQA)**。
  - [x] 实现 **Multi-Head Latent Attention (MLA)** 与极简低秩 `MLACache` 矩阵吸收推理。
  - [x] 实现混合专家前馈网络 **MoELayer**（DeepSeekMoE 细粒度路由、共享专家、无分支 `TopK` + `GatherMatMul` 索引收集矩阵乘法，实现零主机端/数据依赖 `if` 分支的 $O(N \cdot K)$ 稀疏专家执行）。
  - [x] **MoELayer Token 级 Gather/Scatter 分发**：通过一等算子 `TopK` 与 `GatherMatMul` 按 `(n, k)` 索引直接寻址被选中的 $K$ 个专家权重并在静态形状 `[N, K, D]` 上完成前向与反向归约。
  - [x] 实现强化学习 **GRPO** (Group Relative Policy Optimization) 组内优势函数与带 KL 惩罚损失函数。

- [x] **3.2 解耦优化器与训练基建 (Optimizers & Training Infrastructure)**
  - [x] 实现解耦优化器架构：`SGDOptimizer` (Momentum), `AdamOptimizer`, `AdamWOptimizer` (Decoupled Weight Decay)。
  - [x] 学习率调度器族：`CosineScheduler` (Warmup & Min LR), `StepLRScheduler`, `LinearWarmupScheduler`, `ExponentialLRScheduler` 及多态统一接口。
  - [x] 梯度裁剪组件：全局 L2 范数裁剪 (`clipGradNorm`) 与数值范围裁剪 (`clipGradValue`)。
  - [x] 二进制检查点持久化：支持签名魔数、格式校验与张量元数据验证的 `saveCheckpoint` / `loadCheckpoint`。
  - [x] 全模型训练与评估模式一键切换：`setTrainingModel`、`trainModel` 与 `evalModel`。
  - [ ] 引入计算图版本计数器（Version Counter），增强就地修改（In-place ops）在反向传播时的安全性检测（与设计中的缓冲区级冲突跟踪统一，Phase 5）。
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
  - [x] 无梯度模式 (`Graph.initNoGrad`) 下不为中间张量分配 `grad` 缓冲（`tensorND`、`registerSingleOutputOp` 与张量层构造函数仅在需要梯度时分配）。
  - [ ] 消除 `op.forward` 与 `runAndRecordPreallocatedOp` 中 `copyFromEager` 的临时堆分配与输出预清零（并入设计 Phase 1 的 DPS 内核改造）。
  - [ ] **CPU 增量解码预分配工作区（Decode Scratchpad Buffer）**：在 [`Gemma4Q4ForCausalLM`](../src/models/gemma4_q4.zig) 与 [`Gemma4ForCausalLM`](../src/models/gemma4.zig) 的 CPU `decodeStep` 中复用固定形状的层间激活缓冲（`q/k/v`、注意力分数、GeGLU MLP 中间态），消除每步逐层的临时 `Tensor` 分配与 `contiguous()` 拷贝。
  - [ ] 基于计算图生命周期分析的激活缓冲区复用（deferred 模式静态内存规划，设计 Phase 1）。
  - [ ] 将 [`StaticTensor`](../src/tensor/static.zig) 融入动态计算图，得到零分配、编译期校验形状的子图。

---

### Phase 4 (P3): 分词器、模型互操作与其他 GPU 后端 (Tokenizer, Interop & Other GPUs)

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

- [ ] **4.3 非 Apple GPU 计算后端探索 (Other GPU Backends，Apple Silicon Metal 见 2.5)**
  - [ ] 评估 CUDA 后端（`std.DynLib` 动态加载 `libcuda` / `libcublas`，自定义内核经 NVRTC 运行期编译）。
  - [ ] 探索通过 WebGPU / Vulkan Compute Shaders 实现跨平台纯图形 API 训练与推理后端。

- [ ] **4.4 训练可观测性 (Training Telemetry)**
  - [ ] 终端训练面板：实时显示 tokens/s、吞吐、估算 TFLOPS、当前学习率与剩余时间。
