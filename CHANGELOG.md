# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

- **通用模块化重构、模型定义与硬件加速彻底解耦 (`src/models/`, `src/backend/`, `src/autodiff/`, `src/nn/`)**:
  - **纯 PyTorch / JAX 风格模型定义**：`src/models/` 彻底剥离全部专有硬件运行模块（彻底删除 `src/models/gemma4_e4b_gpu.zig` 与 `src/models/gemma4_gpu.zig`），模型架构（`Gemma4ForCausalLM`、`Gemma4E4BForCausalLM`、`Gemma4Q4ForCausalLM`、`GPT`）转为 100% 纯粹的 `nn.Module`，零专有硬件结构（如 `gpu_context`）侵入。
  - **统一计算设备枚举 (`Device.cpu`, `Device.gpu`)**：在 `backend` 统一导出 `pub const Device = mlx.DeviceType;`，并穿透暴露至 `tensor`、`autodiff`、`nn`、`models` 与根包 `zig_ml`；计算图通过 `graph.options.device` 实现全局设备分发。
  - **硬件加速执行流彻底下沉至 Layer 0 (backend)**：
    - 将 Safetensors 解析与零拷贝元数据提取下沉至底层 `src/backend/safetensors.zig`，解耦 `nn` 依赖；
    - 将 Gemma 4 E4B 统一内存直通执行引擎收敛至 `src/backend/e4b_offload.zig`（`Gemma4E4BOffloadEngine`）；
    - 将 Gemma 4 12B 常驻 GPU 显存执行引擎收敛至 `src/backend/gemma4_offload.zig`（`Gemma4GpuContext`、`Gemma4GpuLayer`）；
  - **算子与模块透明加速分发**：上层模块调用 `forward(graph, ...)` 时，由底层 `autodiff.Graph` 与各 `nn.Module` 自动按目标设备派发：
    - 若选择 `.gpu`：透明直通 Apple Silicon MLX Metal 极速内核（GEMM、Q4 矩阵乘法、RMSNorm、RoPE、SDPA、Softmax、GeLU）；
    - 若选择 `.cpu`：透明调用 Accelerate / CBLAS / SIMD 多线程 CPU 内核。
  - **极致性能与架构分层无损验证**：重构后端到端 `gemma4-e4b` 生成速度保持 20.78 token/s，81 个包分层单向校验与全部 392+ 单元测试 100% 通过。

- **Gemma 4 模型解耦、PyTorch/JAX 风格纯模块化重构与硬件执行流剥离 (`src/models/`, `src/models/gemma4_ple.zig`, `src/models/gemma4_e4b.zig`, `src/models/gemma4_e4b_gpu.zig`, `src/models/tests.zig`)**:
  - **解耦逐层嵌入增强为独立模块 (`Gemma4PleLayer`, `Gemma4PleEmbedding`, `src/models/gemma4_ple.zig`)**：遵循 PyTorch / JAX 纯 Module 设计范式，将 Gemma 4 专有的逐层嵌入增强 (PLE) 提取为独立的模块结构，持有子模块与参数张量，通过 `forward(graph, ...)` 纯张量接口调度计算，完全解耦底层硬件细节（零专有数组依赖）。
  - **泛型解码层 PLE 组合支持 (`Gemma4DecoderLayerT(Attn, Mlp, Ple)`)**：扩展泛型变换器解码层支持可选 `Ple` 子模块，并在残差连接中规范化残差加法与层缩放乘法 `(x + ple_normed) * layer_scalar`，消除专有算子硬编码。
  - **Gemma 4 E4B 模块化架构定义与硬件流彻底剥离 (`src/models/gemma4_e4b.zig` vs `src/models/gemma4_e4b_gpu.zig`)**：将 `GpuLayerE4B`、`PleWeights`、`forwardLayer` 与底层 MLX 执行流彻底剥离至专有硬件加速文件 `src/models/gemma4_e4b_gpu.zig`（`Gemma4E4BGpuContext`）；`src/models/gemma4_e4b.zig` 实现零 `backend.mlx` 依赖，纯粹由 `nn.Module` 组装并通过 `autodiff.Graph` 调度前向，性能通过后端 offload 引擎分发并保持 100% 极致吞吐。
  - **模型单元测试全量覆盖 (`src/models/tests.zig`)**：增加 `Gemma4PleLayer`、`Gemma4PleEmbedding` 与 `Gemma4E4BForCausalLM` 端到端前向计算测试，验证纯 Module 拓扑与张量维度的正确性。

- **因果语言模型训练支持与模型直接推理收敛 (`src/engine/causal_lm.zig`, `src/engine.zig`, `src/models/`, `examples/`)**:
  - **因果语言模型训练辅助器 (`engine.causal_lm`)**：在 Layer 5 增加 `trainCausalLMStep` 与 `CausalLMStepResult`，完整保留自回归因果语言模型（如 GPT、Gemma 4）的单步训练、交叉熵损失计算、自动微分反向传播与可选梯度裁剪，确保大模型预训练与微调能力。
  - **模型级直接推理与精简 (`examples/gemma4_e4b.zig`, `examples/gemma4.zig`)**：避免过早抽象跨模型上层推理流水线（各模型词表与特殊控制符各异，如 `Gemma4Vocabulary` 与通用 BPE 词表）；`examples` 统一直接使用各原生模型结构（`Gemma4E4BModel`、`DefaultGemma4Q4`）及其专属词表与 `generate()` 入口执行端到端流式生成与性能测算。
  - **架构分层纯粹化 (`engine` Layer 5 vs `models` Layer 4)**：彻底删除 `src/models/gemma4_e4b.zig` 中的 `Gemma4E4BEngine` 以及 `src/models/gemma4_q4.zig` 中的 `Gemma4Q4Engine`，消除 `models` 包对上层端到端执行流与评测样板代码的耦合；`models` 严格作为纯粹的模型定义库（`Gemma4E4BModel`、`DefaultGemma4Q4`、`GPT`）。


- **Gemma 4 Instruct 对话模板与提示词拼接抽象 (`src/models/gemma4.zig`, `examples/gemma4.zig`, `examples/gemma4_e4b.zig`, `src/models/tests.zig`)**:
  - **结构化对话消息与轮次角色 (`Gemma4Vocabulary.Role`, `Gemma4Vocabulary.Message`)**：在 `Gemma4Vocabulary` 中增加结构化角色（`user`、`model`、`system`）与消息类型，消除手写硬编码词元 ID。
  - **统一模板拼接函数 (`formatInstructPrompt`, `formatChatPrompt`, `appendTurn`)**：提供开箱即用的单轮提示词格式化函数 `formatInstructPrompt(allocator, prompt)` 与多轮对话模板函数 `formatChatPrompt(allocator, messages, add_generation_prompt)`，自动处理 `<bos>`、`<|turn>{role}\n`、`<turn|>\n` 与 `<|turn>model\n`，彻底消除 `examples/gemma4.zig` 与 `examples/gemma4_e4b.zig` 中的手写词元拼接样板代码。


- **共享损失核与泛型循环层收敛 (`src/tensor/losses.zig`, `src/autodiff/`, `src/nn/recurrent.zig`, `src/nn/llm.zig`)**:
  - **共享损失计算核 (`src/tensor/losses.zig`)**：新增 `ceRow`（按对数求和指数 LSE 精确计算单行交叉熵，支持概率写入与原地梯度累加）、`accumulateCeGradRow`、`dpoTerm`（直接偏好优化单样本损失与导数）与 `grpoTerm`（组相对策略优化裁剪代理目标与 KL 散度项及对数概率梯度）；消除 `autodiff/op.zig`、`autodiff/backward_core.zig` 与 `nn/llm.zig` 间三处前向与反向公式重复实现。
  - **整型标签多态支持 (`Tensor.softmaxCrossEntropy`)**：扩展 `Tensor.softmaxCrossEntropy` 支持任意整型切片与元组（`u8`、`u32`、`usize`），复用 `tensor.losses.ceRow` 并实现统一的越界与形状校验。
  - **LLM 对齐损失与组优势计算收敛 (`src/nn/llm.zig`)**：重构 `maskedCrossEntropyLoss`、`dpoLoss`、`computeGRPOLoss` 与 `grpoLoss` 直接调用 `tensor.losses` 共享核；在 `computeGroupAdvantages` 中复用 `backend.vec_math.sum` 与 `sumSquaredDeviations`，提升组内均值与标准差计算效率并消除手写循环。
  - **泛型时间步展开器 (`nn.Recurrent(Cell)`, `nn.RecurrentResult`)**：提取泛型时间步展开容器 `Recurrent(comptime Cell: type)` 与返回结构体 `RecurrentResult`，使经典 `RNN`（`Recurrent(RNNCell)`）与单层门控循环单元 `GRU`（`Recurrent(GRUCell)`）完全复用通用的序列时序展开、隐状态缓冲分配与零状态兜底逻辑。

- **统一 Gemma 4 CPU 泛型模块、配置驱动架构与模型统一生成前端 (`src/models/`, `src/backend/`, `src/nn/`, `examples/`)**:
  - **泛型注意力与解码层原语 (`Gemma4AttentionT`, `Gemma4DecoderLayerT`)**：以 `Gemma4AttentionT(comptime Proj: type)` 与 `Gemma4DecoderLayerT(comptime Attn: type, comptime Mlp: type)` 统一全精度 F32 与 4-bit 量化 Q4 CPU 模型的前向、预填充、增量解码与残差尾部实现；消除 `src/models/gemma4_q4.zig` 中原本 450 行重复手写的注意力机制与解码层样板代码。
  - **配置驱动架构扩展 (`Gemma4Config`, `SamplingOptions`)**：在 `Gemma4Config` 中增加显式 `layer_types`、`sliding_rope`、`full_rope`、`attention_k_eq_v`，并将 `kvSourceLayer` 与 `isKvSharedSource` 深度内聚进配置层；在 `backend.sampling.SamplingOptions` 中增加统一的 `repetition_window: ?usize = null`。
  - **统一自回归生成会话前端 (`Gemma4Q4ForCausalLM.generate -> Generation`)**：为 12B `Gemma4Q4ForCausalLM` 提供与 E4B 一致的统一生成入口 `generate(allocator, prompt, GenerateOptions) -> Generation`（`next()`），将预填充、键值缓存管理、动态 n-gram 搭配转移学习与端到端 GPU 投机采样批量验证（`decodeSpeculative`）彻底内聚于会话内部；重构 `examples/gemma4.zig` 生成主循环。
  - **GPT 模型解耦与草稿推测引擎下沉 (`src/models/gpt.zig`, `src/nn/llm.zig`)**：将 `GPTConfig`、`TransformerDecoder`、`GPT` 与 `DefaultGPT` 从 `src/nn/transformer.zig` 迁至 `src/models/gpt.zig`，消除 `nn` 层次对模型级完整拓扑的耦合；将 `promptLookup` 与通用 `SpeculativeDraftEngine` 下沉至 `src/nn/llm.zig`；移动测试文件至 `src/models/tests_vis.zig` 确保包依赖单向检查完全通过。
  - **基于 MLX 组合块组装 12B `Gemma4GpuLayer` 与 E4B `GpuLayerE4B`**：在 `Gemma4GpuLayer.mlpResidual` 中直接复用 `backend.mlx_blocks.gatedMlp` 与 `residualScale`，进一步收敛 GPU 算子实现。

- **共享 MLX GPU 组合原语与量化抽象下沉 (`src/backend/`, `src/models/`, `src/nn/`)**:
  - **下沉 MLX GPU 组合原语 (`src/backend/mlx_blocks.zig`)**：新增 `Rope`、`projectHeads`、`gatedMlp`、`residualScale` 与 `finalHidden`，统一单步解码与多步预填充的 Q/K/V 头归一化与旋转、门控多层感知机（支持门控/上升融合与独立量化矩阵）以及带缩放残差连接；删除 `offload.qkvForwardOffload` 与 `mlpForwardOffload`。
  - **通用量化权重与量化块布局 (`QuantWeights`, `QuantLayout`, `LmHead`)**：将 `Q4Weights` 重命名并推广为通用 `QuantWeights`，新增 `QuantWeights.gatherRows(ids, scale)` 统一词元嵌入查表与反量化缩放；统一量化布局计算 `QuantLayout` 并替换 `Gemma4E4BModel.q4ByteLengths`；引入统一词表输出头 `LmHead = union(enum) { dense, quant }`。
  - **统一 GPU 键值缓存原语 (`src/backend/offload.zig::GpuKvCache`)**：支持 `dtype` 参数、块式扩容、可选环形滑动窗口、投机解码回滚（`truncate`）与 CPU 双向同步（`importWindow` / `exportWindow`）；在 12B `Gemma4GpuContext` 中替换原先每步动态 `concatenate` 的 KV 数组，并在投机验证中直接调用 `truncate`。
  - **设备端对数几率处理与候选生成 (`src/backend/logits_processor.zig`)**：下沉 `softcapLogits`、`penalizeDevice`、`deviceCandidates`、`selectToken`、`selectTokenFromHidden` 以及面向对象的 `LogitsProcessor`，彻底解耦对特定模型配置 `Gemma4Config` 的依赖。
  - **类型安全 MLX 编译闭包封装 (`src/backend/mlx.zig::compileFn`)**：提供泛型与编译期反射驱动的 `compileFn`，消除手写 `mlx_vector_array_*` 的 C API 样板代码与类型转换开销；使 E4B 步首预计算（`PlePrepContext`）与层尾计算（`GpuLayerE4B.forwardTail`）直接复用同一前向函数。

- **共享 `nn` 模块抽象与架构原语统一 (`src/nn/`, `src/models/`, `src/tensor/quant.zig`, `examples/`)**:
  - **统一可缩放与无缩放 `RMSNorm` (`nn.RMSNorm`)**：扩展 `RMSNorm` 支持可选权重 `weight: ?*Tensor`、偏置偏移 `weight_offset` 与 `initUnscaled`，删除 `src/models/gemma4.zig` 中专有的 `GemmaRMSNorm` 与 `GemmaUnscaledRMSNorm`。
  - **`Linear` 支持可选偏置并消除 3D↔2D 重塑冗余 (`nn.Linear`)**：新增 `Linear.Options{ .bias: bool = true }` 与 `Linear.initWithOptions`；重构 `Linear.forward` 直接接收任意维度张量并在内部利用批量/2D SGEMM 处理，彻底移除 `MLP`、`SwiGLU`、`MLALayer` 与 `Gemma4MLP` 中冗余的手动 3D↔2D `reshape`。
  - **下沉量化线性层与通用量化切片 (`nn.Q4Linear`, `tensor.quantizeSliceBf16`)**：将 `Q4Linear` 从 `src/models/gemma4_q4.zig` 统一迁入 `src/nn/core.zig`，并在 `src/tensor/quant.zig` 提供泛型量化切片 `quantizeSliceBf16` 替代模型私有实现。
  - **泛型门控前馈网络与可配置专家激活 (`nn.GatedMLP(Proj)`, `nn.MoELayer`)**：新增 `GatedActivation` 枚举（支持 `.silu`、`.gelu`、`.gelu_tanh`）与泛型模块 `GatedMLP(comptime Proj)`，支持以 `Linear` 与 `Q4Linear` 实例化；`SwiGLU`、`Gemma4MLP` 与 `Gemma4Q4MLP` 统一收敛为 `GatedMLP` 的特化；`MoELayer` 支持通过 `activation: GatedActivation` 传入可配置激活。
  - **泛型分组查询注意力与张量转置辅助原语 (`nn.GQAAttention(Proj)`, `splitHeads`, `mergeHeads`)**：在 `nn/attention.zig` 中新增张量头重排工具 `splitHeads` 与 `mergeHeads`，并构建泛型 `GQAAttention(comptime Proj)`（支持 QK 归一化、无缩放 V 归一化、可选 RoPE、缩放因子、掩码与键值头复用）；`CausalSelfAttention`、`Gemma4Attention` 与 `Gemma4Q4Attention` 统一复用头转置辅助原语。
  - **单层键值缓存原语与生命周期解耦 (`nn.LayerKVCache`, `nn.KVCache`, `Gemma4KVCache.LayerCache`)**：实现解耦的 CPU 单层键值缓存 `LayerKVCache`（支持 `append`、`writePrefill`、`visibleRange`、`packWindow`/`writeWindow` 及写入前 `ContextWindowExceeded` 保护）；`nn.KVCache` 与 `Gemma4KVCache.LayerCache` 统一代理至 `LayerKVCache`，并修复视图张量结构体与底层切片内存的析构生命周期。
  - **采样原语归一与全局清理 (`src/nn/llm.zig`, `examples/`)**：删除 `src/nn/llm.zig` 中重复的 `sampleTopP` / `sampleTopK`，统一使用 `backend.sampling.Sampler` 并更新 `examples/llm_training.zig` 与 `examples/train_shakespeare.zig`；迁移 `releaseModuleGrads` 与 `createLayerScalar` 至 `nn.core`（`src/nn/module.zig`）。

- **修复基准测试编译与 GPU/CPU 候选采样一致性单测 (`src/bench/mlx_suite.zig`, `src/models/tests.zig`)**:
  - **更新基准测试至统一算子接口 (`src/bench/mlx_suite.zig`)**：将 `mlx_suite.zig` 中遗留的 `Graph.scaledDotProductAttention` 统一更新为 `Graph.attention(q, k, v, .{ .scale = scale, .mask = .causal })`，将 `Graph.rmsNorm` 调用更新为传入 `RmsNormOptions{ .eps = 1e-5 }`，修复 `zig build e4b` / `zig build bench` 编译失败。
  - **修复 GPU 候选采样与 CPU 选取的一致性断言 (`src/models/tests.zig`)**：断言对比经 `sampling.buildDistribution` 剔除不可采样词元并排序后的终结候选分布，消除未截断前原始有界切片与 CPU 堆部分选择的长度与排序偏差。

- **共享注意力、RMSNorm、Softcap 算子与计算图统一设备分发 (`src/backend/attention_mask.zig`, `src/tensor/attention_kernels.zig`, `src/tensor/`, `src/autodiff/`, `src/nn/`, `src/models/`, `examples/models/*.json`, `web/`)**:
  - **共享注意力掩码与内核 (`backend/attention_mask.zig`, `tensor/attention_kernels.zig`, `autodiff/`)**：新增 `backend/attention_mask.zig`（统一 `Mask = union(enum) { none, causal, sliding: usize }`、`visibleRange`、`isSlidingWindowVisible` 与 `fillAdditiveMask`）与 `tensor/attention_kernels.zig`（单查询步 `attendOneQuery` 与批处理 `attentionForward`，支持 `Tq != Tk`、因果/滑动窗口掩码与不物化 `repeatKV` 的分组查询注意力 (Grouped-Query Attention, GQA)）；扩展一等算子 `OpType.Attention` / `Graph.attention` 支持 `AttentionOptions` 及无 `repeatKV` 反向传播与有限差分梯度检查，替换 `ScaledDotProductAttention.forwardCore` 的多算子拆分、`gemma4.prefillAttention` 与 `gemma4.decodeAttention`，并在 `CausalSelfAttention.forwardInference` 中复用 `attendOneQuery`。
  - **统一 RMSNorm 可选权重、偏置偏移与反向上下文保存 (`tensor/nn_kernels.zig`, `autodiff/`)**：`Tensor.rmsNorm` 与 `Graph.rmsNorm` 统一接收可选缩放权重 `w: ?*Tensor` 与 `RmsNormOptions{ .eps, .weight_offset }`，支持无缩放 RMSNorm 与 $(1 + w)$ 偏移模式。
  - **新增一等 `Softcap` 算子 (`OpType.Softcap`, `Tensor.softcap`, `Graph.softcap`)**：将对数几率软截断 $y = c \tanh(x / c)$（$c \le 0$ 时为恒等）收敛为单一算子并补齐导数 $1 - \tanh^2(x/c)$ 与有限差分梯度检查，替换 `gemma4.zig` 与 `gemma4_q4.zig` 中的 `mulScalar -> tanh -> mulScalar` 三步组合。
  - **计算图 GPU 分支统一注册 `Op` 节点 (`autodiff/graph.zig`, `autodiff/graph_nn.zig`, `nn/attention.zig`)**：`matmul`、`batchMatMul`、`rmsNorm`、`rope`（支持部分与比例旋转）、`attention`、`softmax` 的 GPU 分支均与 CPU 分支一致注册 `Op` 节点（无梯度时 `requires_grad = false`），移除 `nn/attention.zig` 中的 GPU 旁路，确保计算图中间表示与 JSON 导出与运行设备无关。

- **共享 SIMD 行原语、旋转位置编码表与嵌入查表 (`src/backend/vec_math.zig`, `src/backend/rope.zig`, `src/tensor/rope.zig`, `src/tensor/`, `src/autodiff/`, `src/nn/`, `src/models/`, `src/regression.zig`)**:
  - **统一 SIMD 行原语 (`backend/vec_math.zig`)**：将 `vec_math.zig` 下沉至第 0 层 `backend` 包并补齐 `dot`、`dotBf16`、`axpy`、`scaleCopy`、`sum`、`max`、`sumSquares`、`sumSquaredDeviations`、`softmaxRow` / `softmaxRowInPlace`、`logSumExpRow` / `logSumExpPartial`、稳定 `sigmoid`、标量 `softcap` 与 BF16 展开（`bf16ToF32` / `widenBf16` / `widenBf16Bytes`），删除 `gemma4`、`nn/attention`、`regression`、`cblas`、`tensor/core`、`backward_core` 中的全部私有副本。
  - **统一旋转位置编码公式与旋转表 (`backend/rope.zig`, `tensor/rope.zig`)**：`RopeOptions` 下沉至 `backend/rope.zig` 并新增 `RopeOptions.proportional(theta, factor)` 预设与频率/周期表生成函数；新增 `tensor.RopeTable`，供 CPU `Tensor.rope`、RoPE 反向传播（正交逆旋转，约 12× 提速）与 `MLALayer.forwardInference`（新增 `MLALayer.rope` 字段，不再硬编码基数 10000）共用，`Gemma4AttentionType.ropeOptions()` 替代 `ropeTheta`/`partialRotaryFactor`；删除 `ops.applyRoPE*`、`nn.rope1DAngles`/`applyRope1D`、`gemma4.ropeSplitHalf*` 与 `gemma4_gpu.proportionalRopeFreqs`。
  - **统一嵌入查表与索引校验 (`tensor.embeddingLookupInto`, `tensor.checkIndices`)**：支持 `f32` 与 BF16 原始位查表及缩放，越界词元 ID 统一返回 `error.IndexOutOfBounds`，修复 `gemma4_q4` BF16 `forward` 静默截断越界词元 ID 的问题。

- **双曲正切近似 GELU 与 Gemma 4 E4B 超窗预填充掩码修复 (`src/backend/activation.zig`, `src/backend/mlx.zig`, `src/backend/offload.zig`, `src/tensor/`, `src/autodiff/`, `src/nn/`, `src/models/`)**:
  - **双版本 GELU 激活 (`backend/activation.zig`, `OpType.GeluTanh`)**：新增 `GeluApproximation = enum { none, tanh }` 标量前向与导数，保留精确 `erf` 形式（`.none`，GPT / MoE 等默认）与 `gelu_pytorch_tanh` 近似形式（`.tanh`），接入 `Tensor.gelu`、`Graph.gelu`（新增 `OpType.GeluTanh` 及反向传播与梯度检查）、`nn.GELU`、MLX `gelu`/`geglu`/`fusedGeluMul`/`compileGeluMul`；Gemma 4 配置 `hidden_activation = .tanh` 并贯穿 12B CPU/Q4/GPU 与 E4B 全路径，消除与官方 HuggingFace / Python MLX 的激活函数偏差。
  - **修复 E4B 滑动窗口预填充超窗掩码 (`backend/offload.zig`, `backend/mlx.zig`, `models/gemma4_e4b.zig`)**：当预填充序列长度超过滑动窗口（如 $T > 512$）时，先写入最后 $W$ 个键值槽位并为 `fastScaledDotProductAttention` 构造精确的 $[1, 1, T, K]$ 滑动窗口加性掩码（`mask_mode = "array"`），消除此前直接传入 `"causal"` 导致旧词元跨窗口泄露或环形覆写错位的问题；`GpuKvCache.computeWriteRange` 对单次超出窗口容量的调用返回 `error.SlidingWindowCacheWrapped`。

- **Gemma 4 E4B 统一生成入口 (`src/models/gemma4_e4b.zig`, `examples/gemma4_e4b.zig`, `plan/TODO.md`)**:
  - 删除对外并存的 `prefillAsync` / `decodeStepAsync` 与 `prefill` / `decodeStep` 两套接口，改为唯一入口 `Gemma4E4BModel.generate(allocator, prompt_tokens, GenerateOptions)` 返回的 `Generation` 会话：`next()` 逐个产出词元，到达停止词元或 `max_new_tokens` 后返回 null；不带重复惩罚的贪婪采样在内部自动启用双阶段 GPU 流水线，其余采样设置逐步同步执行，调用方无需区分。`GenerateOptions` 含 `sampling`（默认官方生成配置）、`max_new_tokens` 与 `stop_tokens`（默认 `<eos>` / `<turn|>`）。

- **Gemma 4 随机采样：温度、前 K 截断、核采样与可复现种子 (`src/backend/sampling.zig`, `src/backend/mlx.zig`, `src/backend/offload.zig`, `src/models/gemma4*.zig`, `examples/gemma4*.zig`)**:
  - **统一采样定义**：`backend.sampling` 新增 `SamplingOptions`（温度 (Temperature)、前 K 截断 (Top-K)、核采样 (Nucleus Sampling, Top-P)、重复惩罚、随机种子，全部带默认值，`temperature <= 0` 或 `top_k == 1` 为贪婪解码）与持有伪随机数生成器的 `Sampler`，同一种子可复现生成序列；处理顺序与 Hugging Face 一致：最终对数几率截断 → 重复惩罚 → 排除词元 → 温度 → Top-K → Top-P → 抽样。
  - **高效候选选择与共享终结步骤**：`selectTopK` 分块有界堆部分选择（O(V log k)，结果与线程数无关），`logSumExpScaled` 提供全词表归一化常数；CPU 与 GPU 共用 `buildDistribution` / `Distribution` 终结步骤，相同候选集与种子得到相同词元。
  - **各推理路径接入**：新增官方生成配置 `Gemma4Config.generation`（temperature 1.0、top_k 64、top_p 0.95、无重复惩罚）；12B F32 / Q4 CPU 路径并行物化截断后的对数几率再采样（贪婪仍走融合快速路径，输出不变）；12B GPU 与 E4B 在设备端完成截断、重复惩罚、排除与 `argpartition` 候选选择，只传回 k 个候选（未启用 Top-K 时固定取 1024 个候选）；E4B 双阶段异步流水线只用于不带惩罚的贪婪解码。
  - **无偏投机解码**：12B GPU 投机解码在随机采样下按标准投机采样规则校验（以目标概率接受草稿、拒绝后去掉草稿重采样、全部接受时抽取额外词元），贪婪模式仍按精确匹配校验。
  - **接口变化**：Gemma 4 各生成入口（F32 / Q4 `prefill`、`decodeStep`、`decodeSpeculative`，`Gemma4GpuContext` 解码入口，E4B `prefill` / `decodeStep`，`gemma4_gpu.selectToken` / `selectTokenFromHidden`）以 `*Sampler` 取代 `repetition_penalty: f32`；Q4 `generateNextToken` 与 `generateNextTokenWithPenalty` 合并为接收 `*Sampler` 的 `generateNextToken`；MLX 绑定新增 `argpartitionAxis`、`takeAlongAxis`、`putAlongAxis`、`logsumexpAxis`、`negative`、`greater`、`where`、`share`。
  - **示例**：`examples/gemma4.zig` 与 `examples/gemma4_e4b.zig` 新增 `--temperature`、`--top-k`、`--top-p`、`--penalty`、`--seed`、`--greedy` 参数，默认采用官方生成配置并打印生效设置；移除 12B 示例中硬编码的 1.25 重复惩罚。

- **开发环境与最低版本切换到 Zig 0.17 (`devenv.nix`, `devenv.lock`, `build.zig.zon`, `README.md`, `doc/README.md`)**:
  - `devenv.nix` 显式固定 `pkgs.zig_0_17`，并更新 `devenv.lock` 中的 `devenv-nixpkgs` 版本；`build.zig.zon` 的 `minimum_zig_version` 改为 `0.17.0`，README 徽章与说明同步为 Zig 0.17.0。

- **审计遗留问题全部修复：Gemma 4 GPU 代码归位、采样语义统一、检查点格式与 MoE 加速 (`src/backend/`, `src/models/`, `src/tensor/`, `src/autodiff/`, `src/nn/`, `src/optim.zig`, `src/dataset.zig`, `src/cross_validation.zig`, `examples/`, `web/`, `doc/`)**:
  - **后端与模型 (`backend`, `models`)**：Gemma 4 专属的 GPU 解码层与上下文 (`Gemma4GpuLayer` / `Gemma4GpuContext`)、比例旋转位置编码 (`GpuRope`) 与词元选取迁移至 `src/models/gemma4_gpu.zig`，`backend.offload` 只保留与模型无关的卸载原语，新增 `backend.sampling`（排除区间、重复惩罚、`argmaxWithPenalty`）与 `TokenExclusionMask`；排除词元统一由 `Gemma4Config.sampling_excluded_tokens` 定义，GPU argmax（12B 与 E4B）不再遗漏 258880–258884 控制词元；12B GPU 全局注意力层改用完整 head_dim 的比例旋转位置编码频率表；全部路径改为先最终对数几率截断 (Softcap) 再施加重复惩罚，与 Hugging Face 一致；Q4 模型从 GPU 切回 CPU 解码前把 GPU 键值缓存同步回 CPU；`Gemma4GpuContext` 独占持有 LM Head；`LmHeadWeights.initWithMode` 遇到不兼容的量化模式返回 `error.UnsupportedQuantMode`，不再静默退回 BF16；未传入命令流的 MLX 算子复用缓存的默认 GPU 命令流（`mlx.releaseDefaultStream` 释放）；`dataFloat32` / `dataUint32` / `itemUint32` 校验数据类型；`Gemma4Vocabulary.encode` 对未登录字符执行 `<0xNN>` 字节回退；直通布局加载器不再向 stderr 打印耗时（GPU 改动仅做 aarch64-macos 类型检查，需在 Apple Silicon 上实测）。
  - **张量 (`tensor`)**：`gatherMatMulInto` 新增 `allocator` 参数，按专家分组 (`nn_kernels.ExpertGroups`) 后每个专家只调用一次 SGEMM（混合专家 (Mixture of Experts, MoE) 前向约 1.4–4.9×）；`fillNormal` / `fillUniform` / `GenericTensor.fill` 在视图上只写可见元素；`StaticTensor.matmul` 不再跳过零元素；`sort` / `argsort` / `topK` 采用 NaN 视为最大值的全序比较，`topKInto` 不再因 Inf / NaN 污染其他槽位并支持跨步视图；`GenericTensor.sum` 对跨步视图逐行遍历；连续视图直接交给损失、Softmax、归一化、RoPE、concat / split、极值归约与 QR / SymEig / SVD 内核，不再额外复制。
  - **自动微分与神经网络 (`autodiff`, `nn`)**：删除从未记录的 `OpType.AddBias` / `OpType.SigmoidCrossEntropy`；Linear、MLP、RNNCell、GRUCell、CausalSelfAttention 公式改为与 `[in, out]` 权重一致的行向量形式 `x W` 并重新生成示例 JSON；`Graph.matmul` 支持左操作数带前导批次维度 (`[..., K] x [K, N]`)；`dpoLoss` / `computeGRPOLoss` / `grpoLoss` 返回错误联合而非断言；`nn.Dropout.init` 校验概率；`maskedCrossEntropyLoss` 去掉未使用的 `allocator` 参数；`collectGraphNodes` 分配失败不再泄漏；删除已弃用别名 `RoPEOptions` / `TensorOf`。性能：GatherMatMul 反向按专家分组 SGEMM（512 词元 × 2 槽位 × 8 专家约 231 → 18 ms）；求和 / 均值反向跨步逐行遍历（转置视图约 20×）；Conv1D / Conv2D 反向按样本块批量 im2col + SGEMM；多头潜在注意力 (Multi-head Latent Attention, MLA) 推理每步只计算一次 RoPE 角度表（约 35×）。
  - **训练与数据 (`optim`, `dataset`, `cross_validation`, `examples`)**：优化器检查点升级到版本 2，状态缓冲全部按小端序 (Little-Endian) 写出，不再读取版本 1；`loadCheckpoint` 改为事务式加载，任何错误都不会留下部分加载的状态（新增 `InvalidCheckpointFlag`）；`CrossValidationOptions` 新增 `max_iter` / `tol`，`CrossValidationGridSearch.init(allocator, options)` 保存配置，`searchLasso` 不再接受 `random_seed` 参数；字节对编码 (Byte Pair Encoding, BPE) 训练频次相同时按 `(left_id, right_id)` 字典序选择合并对；`transformer_gpt` / `train_shakespeare` / `llm_training` 的推理演示改用无梯度计算图。
  - **代码格式**：`src/`、`examples/`、`tools/` 与 `build.zig` 全部通过 `zig fmt`。

- **全仓分包代码审计：正确性修复、内存安全与热点加速 (`src/backend/`, `src/tensor/`, `src/autodiff/`, `src/nn/`, `src/optim.zig`, `src/dataset.zig`, `src/regression.zig`, `src/manifold.zig`, `src/cross_validation.zig`, `src/engine.zig`, `src/bench/`, `src/models/`, `examples/`, `tools/`)**:
  - **后端 (`backend`)**：纯 Zig 后备单精度通用矩阵乘法 (Single-precision General Matrix Multiply, SGEMM) 改为 GotoBLAS 式打包分块与 6×16 寄存器微内核（1024³ 约 34 → 76 GFLOPS），不再跳过 `a == 0` 而丢失 NaN / Inf 传播，并正确支持列主序；`Executor.parallelFor` / `parallelReduce` 修复超大粒度下的整数溢出；MLX 卸载路径修复 `argmaxWithPenalty` 越界读、滑动窗口键值缓存 (Key-Value Cache, KVCache) 裁剪丢失最新词元、投机解码 (Speculative Decoding) 窗口掩码与回滚错误、`Q4Weights` 未校验长度、`copyToSlice` 非连续视图越界、`VectorArray` 超过 64 个数组被截断以及多处错误路径泄漏（MLX 改动仅在 Linux 上做 aarch64-macos 类型检查，需在 Apple Silicon 上实测）。
  - **张量 (`tensor`)**：修复 `zeros` / `concat` / `reshape` / 即时损失 / `qr` / `svd` 等分配失败泄漏；`softmaxCrossEntropy` 改用对数求和指数 (Log-Sum-Exp, LSE)，不再把损失截断在 34.5；原地算子与 `repeat` 只作用于视图可见元素，`batchMatMul` / `matmul` 正确处理转置视图；0 维张量 `numel` 为 1；`bf16.fromF32` 保留 NaN；混合专家 (Mixture of Experts, MoE) 内核校验形状与非法专家下标。性能：广播运算维度折叠（约 3×）、按行归约（约 20×）、向量化指数 `vec_math.exp8` 加速 `softmax`（约 5.4×）、旋转位置编码 (Rotary Position Embedding, RoPE) 三角表复用（约 16×）、转置卷积改为 SGEMM + 列转图像 (Column to Image, col2im)（约 16×）。
  - **自动微分 (`autodiff`)**：新增覆盖全部可微算子与损失的有限差分梯度检查 (Finite-Difference Gradient Check)；`Graph.backward` 每次反向前清零中间梯度（修复重复反向导致的梯度重复累加）、只遍历需要梯度的子图、拓扑排序改为迭代式且不再占用计算图 Arena；MatMul 反向支持跨步视图，新增 `Graph.contiguous` 为仅支持连续布局的算子物化视图输入；BatchNorm 重演时重新计算批统计量；交叉熵统一使用 LSE；`Graph.q4MatMul` 支持对输入求梯度（可在 4-bit 骨干上训练低秩自适应 (Low-Rank Adaptation, LoRA)）。性能：广播 / 转置反向逐行遍历（自注意力训练约快 17%），转置卷积反向改为 im2col + SGEMM（约 18×）。
  - **神经网络 (`nn`)**：修复 `sampleTopP` 未按概率排序取核 (Nucleus) 的错误；多头潜在注意力 (Multi-head Latent Attention, MLA) 增量推理补上 `b_uv` 偏置并支持二维输入；KVCache / MLACache 写满返回 `error.ContextWindowExceeded`；Safetensors 头部全面校验并支持非对齐 BF16 / F16；非法配置与空序列返回错误而非断言；模型图形状字符串缓冲区越界；`deinitModel` 去重集合无法分配时退化为按遍历顺序回扫，权重共享下既不重复释放也不泄漏。增量注意力内层循环使用单指令多数据 (Single Instruction Multiple Data, SIMD) 向量化（约 8×）。
  - **训练与数据 (`optim`, `dataset`, `regression`, `manifold`, `cross_validation`, `engine`)**：检查点加载先校验头部与文件长度再写入状态（新增 `CheckpointSizeMismatch`）；`AdamOptimizer.t` 改为整数 `step_count`；`clipGradNorm` 以 f64 汇总；Adam / AdamW 共享 SIMD 内核（AdamW 1M 参数单步 16.3 ms → 约 1.0 ms）；字节对编码 (Byte Pair Encoding, BPE) 训练使用正确的合并 Token ID；`DataLoader` 校验空数据集、批大小与标签数；t 分布随机邻域嵌入 (t-Distributed Stochastic Neighbor Embedding, t-SNE) 困惑度搜索避免亲和度下溢；交叉验证 (Cross-Validation, CV) 一倍标准误 (One-Standard-Error, 1-SE) 规则与网格顺序无关；分类训练 / 评估按样本数加权平均；下载工具先写临时文件再重命名。
  - **模型 (`models`)**：`Gemma4DecoderLayer.forward` 施加 `layer_scalar`，`layer_scalar` / `v_norm` 改为不可训练缓冲区；预填充超出 KVCache 容量与词元 ID 越界返回错误；Gemma 4 E4B 全局注意力改用比例旋转位置编码 (Proportional RoPE) 频率表；修复 E4B 加载与前向错误路径的重复释放和 Q4 构造泄漏；直通格式按 4 字节对齐（魔数 `ZNNQ4DIR` → `ZNNQ4D02`，需重新运行 `zig build pack-gemma4`）；`quantize-gemma4` 校验张量形状；解码注意力与 CPU 词表投影改用 SIMD 点积（约 10×）。
  - **接口变化**：`AdamOptimizer.t` → `step_count: u64`；`BPETokenizer.addMerge` 返回合并 ID；`ClassificationEpochResult` 新增 `num_samples`；删除 `engine` 兼容别名（`StepResult`、`EpochResult`、`trainStep`、`evalStep`、`trainEpoch`、`evaluate`）、`BinaryMmapDataset.close`、`Tensor.sigmoidCrossEntropy`、`Nonlinearity.defaultNonlinearity`、`InitMethod.defaultConfig` 与 `Graph.convTranspose2D` 重复别名。

- **全面适配 Zig 0.17 编译器标准与语言重大语法升级 (`src/`)**:
  - **移除废弃重复算子 `**` 并迁移至 `@splat` 与编译期内存复制**：针对 Zig 0.17 废弃的数组重复操作符 `**`，将全库张量维度、形状元数据、静态掩码及测试数据初始化统一重构为原生 `@splat(...)`；字符串与语料复用改用编译期静态块内 `@memcpy` 复制。
  - **适配 `@typeInfo` 类型反射并行数组化与属性重构 (`src/nn/module.zig`, `src/tensor/static.zig`, `src/nn/tests_vis.zig`, `src/nn/graph_ir.zig`, `src/nn/core.zig`)**：
    - 将结构体字段反射 `info.@"struct".fields` 迁移至 Zig 0.17 并行数组 `field_names`、`field_types` 与 `field_attrs`，字段属性访问同步更新为 `attrs.@"comptime"` 与 `attrs.@"align"`；
    - 将枚举类型字段反射迁移至 `field_names` 与 `field_values`，算子公式注册采用标准 `std.enums.values`；
    - 将函数形参类型反射 `info.@"fn".params` 迁移至 `info.@"fn".param_types`；
    - 指针只读属性由 `info.pointer.is_const` 迁移至 `info.pointer.attrs.@"const"`；
    - 运行时构建模式由大写 `.Debug` 迁移至 Zig 0.17 的枚举成员 `.debug`。
  - **剥离 `@cImport` 依赖并内置 MLX C 矢量化抽象 (`src/backend/mlx.zig`, `src/backend/mlx_c.zig`)**：适配 Zig 0.17 彻底废除 `@cImport` 语言内置宏的规范，通过翻译模块生成静态自包含的 `src/backend/mlx_c.zig` 绑定，使 `zig test` 与无头 CI 具备即时毫秒级加载能力，彻底消除对系统 Clang AST 在线翻译的开销。
  - **标准库 API 迁移 (`src/nn/core.zig`, `src/nn/graph_ir.zig`, `src/nn/tests_vis.zig`)**：
    - 内存分配器哨兵复制由 `allocator.dupeZ` 迁移至 `allocator.dupeSentinel(u8, ..., 0)`；
    - 元组类型构造由已废弃的 `std.meta.Tuple` 迁移至编译器原生内置函数 `@Tuple(&.{...})`。

- **重构 `MoELayer` 为无分支 `TopK` + `GatherMatMul` 索引专家算子架构：零 `if` 分支实现 $O(N \cdot K)$ 稀疏激活与静态图重演 (`src/tensor/nn_kernels.zig`, `src/tensor/core.zig`, `src/autodiff/types.zig`, `src/autodiff/op.zig`, `src/autodiff/graph_nn.zig`, `src/autodiff/backward_nn.zig`, `src/autodiff/graph_init.zig`, `src/nn/transformer.zig`, `src/nn/visualization.zig`, `examples/models/deepseek_moe.json`)**:
  - **新增无分支前 K 项选择算子 (`OpType.TopK`, `Tensor.topKInto` / `Graph.topK`)**：通过纯算术比较排名累加（`@intFromBool(v_m > v_j) | @intFromBool((v_m == v_j) and (m < j))` 与 `@floatFromInt(@intFromBool(rank == r))`）从 `[N, E]` 门控概率中直接输出 `(values: [N, K], indices: [N, K])`，全程零数据依赖 `if` 跳转，反向传播将 `values.grad` 按 `indices` 路由回原专家列。
  - **新增索引收集专家矩阵乘法算子 (`OpType.GatherMatMul`, `Tensor.gatherMatMulInto` / `Graph.gatherMatMul`)**：对标 Apple MLX 的 `mlx_gather_mm` (`SwitchMLP`)，仅在固定 `[N, K]` 循环内通过 `e = @intFromFloat(indices[n, k])` 直接偏移寻址被选中专家的权重 `weights[e]` 与偏置 `biases[e]` 并调用 BLAS `cblas_sgemm`，前向与反向计算量严格为 $O(N \cdot K)$（未选中专家零计算且梯度严格为 0），同时在 `graph_init.zig` 的 `computeParamFans` 与 `detectConsumerActivation` 中完整支持全体 $E$ 个路由专家的参数扇入/扇出与激活函数推导。
  - **彻底移除 `MoELayer.forward` 中的主机端 `if` 分支与 CPU 掩码构造 (`src/nn/transformer.zig`)**：以 `Softmax -> TopK -> GatherMatMul(c_fc) -> Gelu -> GatherMatMul(c_proj) -> Mul -> Sum` 替代原有的 CPU Top-K 排序循环、`keep_node` 二值掩码、`Split`、`isUninitialized()` 与 `if (!expert_active[e]) continue;`，使计算图拓扑静态固定、支持跨不同路由结果的 `graph.forward()` 原位重演与整图 GPU 派发。
  - **增强多专家参数算子在可视化中的作用域归属与动态形状缓冲 (`src/nn/visualization.zig`)**：将 `opScope` 升级为求算子全部参数所属模块的最近公共祖先作用域（使横跨多个专家子模块的 `GatherMatMul` 归属于 `routed_experts` 容器作用域，保持局部图严格无环 DAG），并将 `collectGraphOps` 中的 `inp_shape` / `param_shape` 格式化缓冲改为动态 `std.ArrayList(u8)`。

- **修复 `MoELayer` 混合专家门控梯度消失缺陷、恢复真稀疏专家跳过与局部图死分支端口过滤 (`src/nn/transformer.zig`, `src/nn/visualization.zig`, `src/nn/tests.zig`, `examples/models/deepseek_moe.json`)**:
  - **修复 `top_k == 1` 时门控网络梯度恒为 0 与双重掩码冗余 (`src/nn/transformer.zig`)**：移除此前先对非 Top-K 位置加 `-1e9` 再执行 Softmax、随后又乘二值 `keep_node` 的双重掩码逻辑（该实现会导致 `top_k == 1` 时胜出专家的 Softmax 输出恒为 `1.0`，由 $\frac{\partial p_i}{\partial z_i} = p_i(1 - p_i) = 0$ 造成门控网络 `gate.weight` 与 `gate.bias` 梯度完全消失，且切断未选中专家的竞争负梯度）。改为对齐 DeepSeekMoE / Switch Transformer 标准范式：先对全量路由专家 logits 执行 `graph.softmax(gate_logits)`，再乘以命名的 `{scope}.top_k_mask` 二值缓冲区（并新增可选 `norm_topk_prob: bool = false` 字段支持在 $K > 1$ 时对选中专家概率重归一化）。
  - **恢复并完善 `expert_active` 专家级真稀疏执行与两阶段初始化兼容 (`src/nn/transformer.zig`)**：在路由专家遍历中恢复 `if (!expert_active[e]) continue;` 稀疏跳过，使未被当前批次任何词元命中的路由专家真正跳过前向与反向计算；同时通过 `isUninitialized()` 检测 `nn.initModel` 之前的全零建图阶段并自动纳入全部路由专家以完成两阶段参数发现与初始化，并为 `MLP`、`SwiGLU` 与 `MoELayer` 补全 `resetParameters`。
  - **消除 Top-K 栈缓冲区硬编码上限并补全输入校验 (`src/nn/transformer.zig`)**：将 `top_indices` 与 `top_vals` 由固定 `[64]` 栈数组改为从 `graph.arenaAllocator()` 动态分配，并在 `MoELayer.init` 与 `MoELayer.forward` 中补全超参数与输入张量维度校验（`error.InvalidConfiguration`、`error.IncompatibleDimensions`、`error.ShapeMismatch`）。
  - **修复多输出算子（`Split`）稀疏未消费分支在可视化局部图中产生幽灵 `@out` 端口 (`src/nn/visualization.zig`)**：在 `LocalGraphBuilder.build` 中过滤多输出算子（`p.outputs.len > 1` 且至少有一个输出已被消费）中未被稀疏分支消费的未命名子输出，避免被跳过的路由专家对应的 `prob_cols[e]` 被误判为模块终端输出端口。

- **修复 Gemma 4 E4B 预填充首字延迟 (Prefill / TTFT) 退化：隔离单步解码形状特化静态编译闭包与动态长度预填充路径 (`src/models/gemma4_e4b.zig`)**:
  - **根因定位**：`compiled_ple_prep`（步首词元嵌入与逐层输入预计算闭包）与 42 层 `GpuLayerE4B.compiled_tail`（注意力后全尾部闭包）因包含依赖序列长度 $T$ 的 `reshape` 与 `split` 算子，以形状特化模式 (`shapeless = false`) 编译。此前在 `T > 1` 的预填充阶段无条件调用这 43 个闭包，导致虽然 `warmup()` 预热了 `T = 2` 与 `T = 1`，但真实推理遇到任意新 Prompt 长度（如 `T = 12`）时会同时触发 **43 个编译闭包的形状缓存未命中 (Shape Cache Miss)**，迫使 MLX 在 TTFT 计时窗口内同步重新追踪 43 张计算图并 JIT 编译 43 个新的 Metal 管线状态对象 (PSO)，使预填充首字延迟从 ~94ms 退化至数倍。
  - **精准分流 (`T == 1` vs `T > 1`)**：在 `forwardTokenArrayToLogits` 与 `forwardLayer` 中严格限定仅在单步自回归解码 (`T == 1` / `seq_len == 1`) 时调用 `compiled_ple_prep` 与 42 层 `compiled_tail`（该路径在 `warmup()` 的 `decodeStepAsync` 中已 100% 完成 `T = 1` 编译缓存）；而在预填充阶段 (`T > 1`) 直接执行基于全局形状无关 (`shapeless = true`) 融合闭包 `compiled_gelu_mul` 与 `compiled_add_mul` 的算子路径（该路径在 `warmup()` 的 `prefillAsync` 中已完成 3D/4D 预热且对任意 Prompt 长度 $T$ 零重编译复用），并同步简化 `initCompiledTail` 中仅针对 3D `ple_slice` (`[1, 1, 256]`) 的门控计算逻辑。

- **系统架构与模块边界重构：下沉 `Gemma4E4BModel` 至 `src/models/`、统一 `SafetensorsArchive` 与 `GpuKvCache`、独立 `src/backend/tests.zig` (`src/models/gemma4_e4b.zig`, `src/nn/serialization.zig`, `src/backend/offload.zig`, `src/backend/tests.zig`, `examples/gemma4_e4b.zig`, `tools/quantize_gemma4.zig`)**:
  - **下沉 `Gemma4E4BModel` 核心模型实现至 `src/models/gemma4_e4b.zig`**：将原本位于示例脚本 `examples/gemma4_e4b.zig` 中的 `PleWeights`、`PlePrepContext`、`GpuLayerE4B` 与 `Gemma4E4BModel` 完整迁移至正式库模块 `src/models/gemma4_e4b.zig`，并在 `src/models.zig` 与 `src/root.zig` 门面层统一导出；`examples/gemma4_e4b.zig` 精简为约 230 行的轻量命令行评测驱动程序。
  - **统一零拷贝 `SafetensorsArchive` 内存映射解析器至 `src/nn/serialization.zig`**：在 `src/nn/serialization.zig` 中统一提供 `TensorMeta` 与 `SafetensorsArchive`（同时支持双分片目录 `open` 与单文件 `openSingleFile`），消除 `src/models/gemma4_e4b.zig` 与 `tools/quantize_gemma4.zig` 中的重复 Safetensors 头部解析与 `mmap` 逻辑。
  - **上提块式预分配与环形滑动窗口 `GpuKvCache` 至 `src/backend/offload.zig`**：将 `GpuKvCache`（含 `WriteRange` 与 `computeWriteRange` 固定容量环形覆写计算）提取为 `src/backend/offload.zig` 的通用 GPU 显存键值缓存原语，理顺 `backend` 与 `models` 的职责边界。
  - **抽离 `backend` 包内联测试至独立 `src/backend/tests.zig`**：将 `src/backend/mlx.zig` 与 `src/backend/offload.zig` 末尾约 730 行内联 `test` 块以及 `GpuKvCache` 单元测试统一收拢至 `src/backend/tests.zig`，并将 `Gemma4E4BModel` 缓存布局测试纳入 `src/models/tests.zig`，使全仓 6 大核心子包（`backend`、`tensor`、`autodiff`、`nn`、`models`、`root`）形成 100% 一致的 `tests.zig` 组织规范。

- **Gemma 4 E4B 全链路图编译、环形滑动窗口 KV 缓存与预融合顺序权重直通缓存 (`src/backend/mlx.zig`, `examples/gemma4_e4b.zig`, `build.zig`, `plan/TODO.md`)**:
  - **42 层注意力后全尾部子图静态编译 (`GpuLayerE4B.initCompiledTail` / `Closure.apply3`)**：在 `GpuLayerE4B` 中将 `o_proj` 之后的完整计算链（`post_attn` RMSNorm → 残差加 `x1` → `pre_mlp` RMSNorm → `fused_gate_up` QMV → `split` → GeGLU → `down_proj` QMV → `post_mlp` RMSNorm → 残差加 `x2` → `ple_gate` QMV → 3D/4D GeGLU → `ple_proj` QMV → `ple_norm` RMSNorm → 残差加 + `layer_scalar` 缩放）通过 `mlx.compile` 编译为每层单一静态闭包 `compiled_tail`。每步解码消除约 **600+ 次细粒度 C API 调用与计算图节点构造**，并由 MLX JIT 自动跨算子融合全部残差、缩放与激活内核。
  - **步首词元嵌入与逐层输入预计算子图静态编译 (`PlePrepContext` / `Closure.apply1To2`)**：将 `embed_tokens` 反量化缩放、`embed_tokens_per_layer` 反量化缩放、`per_layer_model_projection` QMV、`proj_norm` RMSNorm 与 `(proj + tokens) * input_scale` 整合成单一堆常驻上下文静态编译闭包 `compiled_ple_prep`，一次调用直接输出 `(x_cur, ple_inputs)`。
  - **局部滑动窗口层固定容量环形 KV 缓存 (`GpuKvCache.computeWriteRange` / `RotatingKVCache`)**：为 20 个局部滑动窗口 KV 层预分配精确等于 `sliding_window = 512` 的固定容量显存，并在单步解码超过 512 步 (`pos >= 512`) 时以 `pos % 512` 原地环形覆写最旧词元槽位、直接复用整块 `[1, num_kv, 512, head_dim]` 缓冲区而不触发任何 `ensureCapacity` 扩容或 `slice4dAxis2` 视图分配。
  - **预融合顺序二进制权重直通缓存 (`gemma4_e4b_direct.bin`, 魔数 `"ZNN4E4B1"`)**：首次从 Safetensors 加载时自动将全部权重（含预拼接的 24 层 `qkv_proj` 与 42 层 `fused_gate_up`）按精确加载顺序流式写入 `<model_dir>/gemma4_e4b_direct.bin`；后续启动通过 `DirectCacheReader` 与 `MADV.SEQUENTIAL | MADV.WILLNEED` 单遍顺序映射直读，彻底省去双文件 JSON 解析、52 MB `fuse_scratch` 暂存分配与 2.3 GB 权重拼接 `memcpy`。
  - **新增 `gemma4_e4b_tests` 单元测试套件 (`build.zig`, `examples/gemma4_e4b.zig`)**：验证环形滑动窗口 KV 缓存边界回绕逻辑与预融合顺序二进制缓存大小/头部校验。

- **Gemma 4 E4B 自回归解码 (Decode) 极限提速：融合 QKV / Gate+Up 量化矩阵乘、编译残差缩放闭包与共享 KV 切片复用 (`src/backend/mlx.zig`, `src/backend/offload.zig`, `examples/gemma4_e4b.zig`, `plan/TODO.md`)**:
  - **24 层注意力 QKV 投影与 42 层 MLP Gate+Up 投影矩阵融合 (`initFusedFromDirectBuffersCustom`)**：在模型加载阶段通过单一复用的暂存缓冲将前 24 个 KV 拥有层的 `q_proj`、`k_proj`、`v_proj`（4-bit）沿输出维度合并为单一 `qkv_proj` 矩阵，并将全部 42 层的 `gate_proj` 与 `up_proj`（8-bit）沿输出维度合并为单一 `fused_gate_up` 矩阵。每生成一个词元直接消除 **90 次 `quantized_matmul` (QMV) Metal 内核启动**（48 次 QKV + 42 次 Gate/Up），消除 `in_norm` 与 `pre_mlp` 的 90 次重复显存读取，并将原本线程组数量过少的小维度 `k_proj`/`v_proj`（`N = 512`）合并进高占用率大矩阵内核。
  - **静态编译残差相加与缩放融合闭包 (`compileAddMul` / `fusedAddMul`)**：在 `src/backend/mlx.zig` 中新增 `(a + b) * scale` 静态编译融合算子，用于融合每层末尾的 `(x2 + ple_normed) * layer_scalar` 以及步首的 `(ple_proj_normed + ple_tokens) * input_scale`，每生成一个词元再消除 **43 次逐元素 Metal 内核启动**与 43 次全局显存中间往返（连同 QMV 融合每步共减少 **133 次 Metal GPU 内核派发**）。
  - **18 个键值共享层 (`24..41`) 的单步 KV 切片缓存复用 (`step_eff_k` / `step_eff_v`)**：在层 22 与层 23 完成 `slice4dAxis2` 后将当前步有效 KV 视图缓存在 `GpuKvCache` 中，使后续 18 个共享层直接零开销复用同一 `mlx.Array` 节点，每步消除 36 次冗余 `mlx_slice` C API 调用与计算图节点构造。
  - **Prefill → Decode 零气泡首步预入队**：在等待 `prefillAsync` GPU 完成前先行构建并 `asyncEval` 入队第 1 个 Decode 步的计算图，将第 1 步解码的 42 层图构建耗时完全隐藏在 Prefill GPU 执行窗口内。

- **Gemma 4 E4B 预填充首字延迟 (Prefill / TTFT) 优化、算子管线预热与生成稳定性修复 (`src/backend/mlx.zig`, `src/backend/offload.zig`, `examples/gemma4_e4b.zig`, `plan/TODO.md`)**:
  - **修复未对齐权重映射导致的 NaN 与 `<pad>` 词元退化**：定位并修复此前尝试对 `SafetensorsArchive` 非对齐文件偏移使用 `mlx_array_new_data_managed` 直通导致 Metal GPU 内核静默失败、Logits 全为 NaN 触发连续生成 `<pad>` (Token 0) 的缺陷。恢复可靠的 `mlx.Array.fromRawData` 统一内存直通加载，并在权重加载完成后立即调用 `archive.deinit()` 卸载文件映射，保持 100% 位一致输出。
  - **Metal 管线与多维融合闭包离线预热 (`Gemma4E4BModel.warmup`)**：在模型加载收尾阶段自动执行微型预填充 (`T = 2`) 与单步解码 (`T = 1`) 预热传递，提前完成 GEMM 模式 `quantized_matmul`、因果掩码 `fast_scaled_dot_product_attention` 以及 3D/4D `compiled_gelu_mul` 闭包的 Metal Shading Language (MSL) JIT 编译与管线状态对象 (PSO) 缓存，使真实 Prompt 预填充首字延迟 (TTFT) 从 768ms **骤降至 94.23ms**（吞吐率达 127.4 token/s，支持 `--no-warmup` 命令行开关）。
  - **消除预填充阶段 (`T > 1`) 42 层非连续 PLE 张量重塑拷贝内核 (`examples/gemma4_e4b.zig`)**：诊断发现当 `T > 1` 时，`ple_inputs.split(42, 2)` 切分出的 `[1, T, 1, 256]` 视图在维度 1 上非连续，此前逐层调用 `ple_item.reshape(&.{1, T, 256})` 会迫使 MLX 在每层额外派发 1 次 GPU 连续化拷贝内核（全网共 **42 个额外 GPU Copy 内核**）。改为将连续的 `ple_gate_raw` (`[1, T, 256]`) 以 O(1) 零拷贝视图重塑为 `[1, T, 1, 256]`，由 `fusedGeluMul` 直接融合读取跨步视图并输出连续张量后再以 O(1) 视图还原为 `[1, T, 256]`，实现 `T > 1` 预填充阶段零额外拷贝内核。
  - **24 层 `GpuKvCache` 初始预分配与跨轮次零分配复用 (`resetCache`) + Metal Wired Memory 锁定 (`configureOptimalWiredMemory`)**：在模型初始化期一次性预分配 24 个 KV 拥有层的 256 步容量缓冲区，`resetCache()` 仅将 `curr_len` 归零并原位复用已分配显存，彻底消除每次 `prefill` 时动态触发 48 次 `mlx_zeros` 显存分配内核的开销；新增 `mlx.configureOptimalWiredMemory()` 自动锁定 Apple Silicon 推荐工作集上限 (`max_recommended_working_set_size`)。

- **修复全网 bfloat16 隐式 float32 精度提升、启用真双缓冲异步流水线 (`decodeStepAsync`) 与 JIT 编译 GeGLU 宏内核融合 (`src/backend/mlx.zig`, `examples/gemma4_e4b.zig`, `src/backend/offload.zig`, `plan/TODO.md`)**:
  - **彻底根除 `bfloat16 -> float32` 全网隐式精度污染 (`src/backend/mlx.zig`, `examples/gemma4_e4b.zig`)**：定位并修复 C++ `mlx_array_new_float32` 强类型标量在 `mulScalar` 与 `gelu` 中与 `bfloat16` 张量相乘时触发 `promote_types(bfloat16, float32) -> float32` 的核心性能缺陷。此前该隐式类型提升导致从第 0 层嵌入起全网激活变为 `float32`，不仅迫使每步 296 次 `quantizedMatmul` 在 GPU 上额外动态派发 **592 个 `astype(bfloat16 -> float32)` 权重缩放/偏置转换内核**、降级运行 32-bit QMV 内核，还将 24 层 `GpuKvCache` 隐式提升为 `float32`（带宽翻倍）。新增 `f32ToBf16Bits`、`fromScalarBf16`、`fromScalarTyped` 与 `dtype()`，并在模型加载期预分配常驻 `bfloat16` 标量数组（`sqrt_hidden_arr`、`ple` 缩放标量与 42 层 `layer_scalar_w`），确保全网 42 层前向、KV 缓存与 LM Head 100% 运行在原生 16-bit `bfloat16` 快速通道。
  - **实现真正的 CPU-GPU 双级异步流水线重叠 (`prefillAsync` / `decodeStepAsync`)**：修复此前 `best_tok_arr.asyncEval()` 后紧接 `best_tok_arr.itemUint32()` 导致 CPU 在同一步立刻阻塞等待 GPU 完成的假异步问题。新增 `getEmbeddingsFromArray`、`getPleTokensFromArray` 与 `decodeStepAsync`，直接将第 $t$ 步的 GPU `argmax` 输出张量（`[1]`, `uint32`）作为 `takeAxis` 索引构建并异步入队第 $t+1$ 步的 42 层计算图，随后再读取第 $t$ 步词元 ID，实现第 $t+1$ 步 CPU 图构建与第 $t$ 步 GPU 内核执行的 100% 并发重叠。
  - **开启 MLX 静态图编译与 `compiled_gelu_mul` 算子融合 (`src/backend/mlx.zig`, `examples/gemma4_e4b.zig`)**：在 `examples/gemma4_e4b.zig` 中启用 `mlx.enableCompile()`，并通过 `mlx.Array.compileGeluMul` (`shapeless = true`) 将 MLP 与 PLE 门控中的 6 步逐元素激活链 `0.5 * x * (1 + erf(x * 0.7071)) * up` 静态编译为单一 Metal 融合内核，每步解码将 504 次逐元素 Metal 内核启动压减至 84 次（减少 **420 次内核派发/步**）。
  - **单步解码 (`seq_len == 1`) 零转置与零冗余算子快车道 (`examples/gemma4_e4b.zig`, `src/backend/offload.zig`)**：在 `seq_len == 1` 时跳过 `forwardTokenArrayToLogits` 中 42 次 `ple_item.reshape`、跳过 `forwardLayer` 中 `[1, 1, H, D] <-> [1, H, 1, D]` 的 132 次无意义 `transposeAxes` 视图节点构造，并跳过 `layer_scalar == 1.0` 时的 42 次恒等乘法；同时在 `LmHeadWeights.forwardArrayBf16` 中支持 GPU 原生 `bfloat16` `argmax` 采样。
  - **消除 `page_allocator` 系统调用风暴与 Safetensors 双倍内存驻留 (`examples/gemma4_e4b.zig`)**：将 `main` 分配器由每次分配触发内核 `mmap`/`munmap` 的 `std.heap.page_allocator` 切换为多线程池化 `std.heap.smp_allocator`，预分配 `SafetensorsArchive` 哈希表容量，并在权重加载至 Metal 缓存后立即 `archive.deinit()` 释放 3.8 GB 文件映射，大幅压缩冷启动耗时与内存占用。

- **Gemma 4 E4B 性能极限对齐与 Phase 0 (P0) 优化全面落地 (`examples/gemma4_e4b.zig`, `src/backend/mlx.zig`, `plan/TODO.md`)**:
  - 实现块式预分配显存键值缓存 (`GpuKvCache`，`step = 256`)：通过 `sliceUpdate4dAxis2` 与 `slice4dAxis2` 进行原地切片定点更新，彻底消除单步自回归解码中 24 个独立 KV 层共 48 次 `concatenateAxis` 显存反复重分配与历史数据重拷。
  - 引入 GPU 原生 `argmax` 与 4 字节轻量标量回读 (`mlx.Array.argmax`, `dataUint32`, `itemUint32`)：自回归解码直接在 Metal GPU 上完成最大值索引规约，彻底消除每步 1 MB (262,144 个 float32) 全词表 Logits 从 GPU 到 CPU 的同步回读瓶颈。
  - 支持双缓冲异步执行管线与非阻塞入队 (`mlx.Array.asyncEval`, `fromScalarI32`)：移除单步内无谓的 `s.synchronize()` 强同步阻塞，使 Metal GPU 始终保持饱和流水线运行；单词元嵌入索引提取采用栈缓冲区直接构造，消除自回归解码循环中的单步堆内存分配。
  - 优化 Q/K/V 投影归一化内存连续性：将投影后的 FastRMSNorm 提前至张量转置前在连续内存上完成，避免转置后的非连续跨步访存与 Metal 内核复制开销。
  - 步首预切分逐层嵌入增强 (Per-Layer Embeddings, PLE)：通过 `split(42, 2)` 在进入 42 层循环前一次性切分出 42 个层级张量，消除循环体内 84 次切片 C API 调度。
  - 性能与精度实测：在保持与 Google 官方 Python MLX 100% 位一致输出的前提下，解码吞吐稳定在 **16.0 ~ 16.5 token/s**，与 Python MLX 实际 21.93 token/s 的差距由 2.1 倍大幅收窄至 1.3 倍。

- **Gemma 4 E4B (Effective 4B) 4-bit / 8-bit 混合量化模型直通推理、键值缓存跨层共享与逐层嵌入增强 (`examples/gemma4_e4b.zig`, `src/models/gemma4.zig`, `src/backend/mlx.zig`, `src/backend/offload.zig`, `build.zig`)**:
  - 新增 Gemma 4 E4B Instruct 完整端到端推理评测引擎 (`examples/gemma4_e4b.zig`, `zig build run-gemma4-e4b` / `zig build e4b`)，全面支持 Google 原生 42 层 Transformer 混合量化模型架构（包含 35 层局部滑动窗口注意力与 7 层全局全注意力）。
  - 支持多文件 Safetensors 零拷贝内存映射 (`SafetensorsArchive`)：完全绕过 CPU 反量化解包耗时，直接将 4-bit 注意力与 8-bit 门控多层感知机 (MLP) 权重挂载至 Metal 统一内存 (UMA)，冷启动加载压缩至 7.0 ~ 8.2 秒。
  - 完美实现键值缓存共享机制 (Key-Value Cache Shared Layers, `num_kv_shared_layers = 18`)：严格对齐 Google 官方架构设计，前 24 层 (0..23) 具备独立的 K/V 投影与缓存，后 18 层 (24..41) 完全跳过冗余 K/V 计算，分别共享复用滑动窗口注意力层 22 与全局全注意力层 23 的键值缓存，大幅减少计算与显存占用。
  - 在 GPU 上完整实现逐层嵌入增强 (Per-Layer Embeddings, PLE)：通过 `mlx.Array.dequantize` 与 `takeAxis` 结合 Metal GPU 算子流水线，极速计算 `[1, T, 42, 256]` 4D 逐层增强嵌入并与层门控、上投影紧密融合。
  - 实测性能突破：首字延迟 (TTFT / Prefill) 仅 **716 毫秒** (吞吐速率达 **16.7 ~ 25.5 token/s**)，自回归单步解码速率稳定达到 **15.49 ~ 15.75 token/s**，生成自然连贯且完全对齐 Google 官方输出精度。


- **零拷贝直通内存重排格式 (ZNNQ4DIR) 与多核并发解包工具 (`tools/pack_gemma4.zig`, `src/backend/offload.zig`, `src/models/gemma4_q4.zig`, `examples/gemma4.zig`, `build.zig`)**:
  - 设计并实现零拷贝直通权重格式 (`ZNNQ4DIR`)：将 Gemma 4 12B 模型 48 层 Transformer 的量化权重 (`w_q`、`scales`、`biases`) 离线预重排并连续存放，使运行时完全绕过 CPU 单线程解包与数百次内核 `mmap`/`munmap` 内存页清理，通过 `mlx.Array.fromRawData` 直接将内存映射指针挂载至 Apple Silicon Metal 统一内存架构 (Unified Memory Architecture, UMA) 显存中，消除冷启动 40 秒等待，实现秒级极速就绪。
  - 在 `src/backend/offload.zig` 中实现 `packBlocksToDirectBuffers`（零内存分配解包内核）与 `initFromDirectBuffers`，并重构 `initFromBlockSlices` 与 `GpuLayerWeights.initFromDirect`，精准支持全局注意力 (Full Attention) 与滑动窗口注意力 (Sliding Attention) 的异构头维度与非对称投影矩阵重组。
  - 在 `tools/pack_gemma4.zig` (`zig build pack-gemma4`) 中实现多核多线程并发转换工具，将原始 `gemma-4-12B-q4.bin` 极速重排为 8.22 GB 的 `gemma-4-12B-direct.bin`。
  - 在 `src/models/gemma4_q4.zig` 中新增多线程并行层解包回退通道与 `computeExpectedDirectFileSize` 校验，并在 `examples/gemma4.zig` 中实现 `ZNNQ4DIR` 优先探测与自动回退。


- **Apple MLX 闭包与计算图静态编译、门控多层感知机 (MLP) Metal 宏内核融合 (`src/backend/mlx.zig`, `src/backend/offload.zig`)**:
  - 在 `src/backend/mlx.zig` 中完整封装 Apple MLX 的 C 静态编译闭包接口，实现 `VectorArray`（向量容器包装器）、`Closure`（Metal JIT 编译闭包执行器）与 `mlx.compile(fun, payload, shapeless)`，为全栈静态图编译提供完备的基础设施。
  - 在 `src/backend/offload.zig` 的 `GpuLayerWeights` 中引入 `compiled_mlp` 与 `initCompiledMlp`：将 Transformer 层的 7 个微算子（Pre-feedforward 均方根层归一化、Fused Gate-Up 4-bit 量化通用矩阵向量乘法、GeGLU 门控激活融合、Down 4-bit 量化投影、Post-feedforward 均方根层归一化、残差相加与层缩放）在 GPU 加载阶段一次性通过 Metal JIT 编译融合为单一纯函数闭包宏内核。
  - 在 `decodeStepWithArray` 与 `decodeSpeculative` 中全链路启用 `compiled_mlp`，消除自回归解码与投机验证中每步 48 层共 336 次细粒度 Metal 内核往返与运行时调度开销。
  - 在保持 100% 官方词表数学精度与位一致性的同时，通过单元测试与端到端实机验证，首字延迟 (TTFT) 达到 **0.38 秒**，解码生成稳定自然流畅。


- **自适应投机推测草稿引擎、马尔可夫转移缓存与自定义 Prompt 词元编码 (`src/models/gemma4.zig`, `src/models/tests.zig`, `examples/gemma4.zig`, `src/backend/offload.zig`)**:
  - 在 `src/models/gemma4.zig` 中实现 `SpeculativeDraftEngine`，融合多尺度 Prompt N-gram 历史回溯匹配 (`promptLookup`)、在线单步马尔可夫词元转移学习 (`transitions`) 与通用语言高频共现搭配词组缓存 (`seedCommonCollocations`)，在无需引入额外小模型辅助的前提下大幅提升草稿命中率。
  - 在 `src/models/gemma4.zig` 中新增 `Gemma4Vocabulary.buildLookupTable` 与 `encode` 方法，为 262K 官方词表构建高频逆向索引表，支持命令行 `-p` / `--prompt` 输入任意用户 Prompt 文本并动态分词编码。
  - 在 `src/backend/offload.zig` 的 `GpuModelContext.init` 中全面开启 Apple MLX 静态编译模式 (`enableCompile` 与 `setCompileMode(.enabled)`)，并将投机采样回滚阶段优化为惰性计算图切片，消除 96 次同步 Metal 内核等待。
  - 修复 ReleaseFast 优化下切片长度被错误收缩截断为 2-bit 导致投机批量验证触发空切片 SIGSEGV 的编译器范围推断缺陷，实现 100% 数学无损、位一致的自回归投机采样加速。

- **GeGLU 门控激活融合、计算图单例复用与零分支向量化 Argmax 性能优化 (`src/backend/mlx.zig`, `src/backend/offload.zig`, `src/autodiff/graph.zig`, `examples/gemma4.zig`)**:
  - 在 `src/backend/mlx.zig` 中实现 `Array.geglu(intermediate_size)` 门控激活融合算子与 `Array.takeAxis` 沿轴索引抽取算子，每层 Transformer 减少 4 个中间数组分配与 3 次独立 Metal 内核调度，全网 48 层共减少 192 次数组分配与 144 次内核派发，在 `prefill`、`decodeStepWithArray` 与 `decodeSpeculative` 中全链路启用。
  - 在 `src/autodiff/graph.zig` 为 `Graph` 引入 `reset()` 方法（清空 `module_formulas`、`scope_stack`、`tensors`、`ops` 并保留容量，重置 `arena.reset(.retain_capacity)`），并在 `examples/gemma4.zig` 将 `step_graph` 提升至 while 循环外部单例复用，消除每个自回归词元重复分配释放 5 个哈希表与 Arena 内存的开销。
  - 优化 `LmHeadWeights.argmaxWithPenalty` 分块位图算法：将特殊排除词元精确隔离在第 0 块与第 4045 块，消除其余 4094 个 64 词块中内部的全部分支跳转，使 LLVM 能够无条件将 262K 词表对数几率扫描向量化为 ARM NEON `fmax` 指令流。
  - 首字延迟 (TTFT / Prefill) 稳定在 **0.72 秒** (相比基准提速 9.3 倍)，增量解码速度达 **6.32 ~ 6.62 tokens/s**，全局吞吐达 **5.80 ~ 6.05 tokens/s** (相比基准提速 100%+，翻倍)，保持 100% 官方词表位输出一致性。

- **Metal 管线离线预热、Prompt N-gram 投机采样与 MLX 静态编译支持 (`examples/gemma4.zig`, `src/backend/offload.zig`, `src/backend/mlx.zig`, `src/models/gemma4.zig`, `src/models/gemma4_q4.zig`, `src/models/tests.zig`)**:
  - 在 `src/backend/offload.zig` 中新增 `GpuModelContext.warmup`，通过在加载阶段触发微型前向传递预编译 Metal Shading Language (MSL) JIT 着色器并构建管线状态对象 (PSO)，将真实推理的首字延迟 (TTFT / Prefill 耗时) 从 **5.83 秒骤降至 0.74 秒 (提速 7.9 倍)**。
  - 在 `LmHeadWeights.argmaxWithPenalty` 中实现分块位图快车道 (Chunked Bitset Penalty Fast-Path)：99.8% 的 64 词块内无任何重复词元，直接执行纯向量化极速最大值查找，仅在少数命中词块执行位运算，兼顾只读安全性与极速采样。
  - 在 `src/models/gemma4.zig` 与 `src/models/gemma4_q4.zig` 中实现零额外模型参数的 Prompt N-gram 投机采样算法 (`promptLookup` 与 `decodeSpeculative`)：在统一统一显存 GPU 架构下，仅需单次读取 7.59 GB 骨干网络权重，即可一次性完成多个推测候选词元的因果注意力与全网批量验证，突破单字显存带宽高墙，且数学输出与自回归生成保持 100% 精确一致。
  - 在 `src/backend/mlx.zig` 中暴露 Apple MLX 编译接口 (`enableCompile`, `disableCompile`, `setCompileMode`)，为底层算子图静态编译与宏内核融合提供官方工程基础设施。

- **支持 Gemma 4 命令行 `--q4` / `--q8` 词表量化模式与 `loadFromMmapWithOptions` 接口 (`examples/gemma4.zig`, `src/models/gemma4_q4.zig`, `src/backend/offload.zig`, `build.zig`)**:
  - 在 `src/models/gemma4_q4.zig` 中新增符合工程规范默认值的 `LoadMmapOptions` 配置结构体，提供 `loadFromMmapWithOptions` 支持外部指定 `lm_head_mode`（默认 `.bf16`，可显式指定 `.q4` 或 `.q8`）。
  - 在 `examples/gemma4.zig` 中支持 `--q4`、`--q8` 与 `--bf16` 命令行参数解析，并在 `build.zig` 中为 `gemma4` 示例开启 `.forward_args = true`。
  - 使用 Q4 量化 LM Head 后，词表显存占用直降 75%（从 2.01 GB 降至 530 MB），首字延迟从 9.42 秒大幅优化至 2.71 秒，增量解码速度从 5.64 tokens/s 跃升至 **6.91 tokens/s**。

- **修复 GPU 卸载上下文生命周期、长序列滑动窗口 Prefill 与零拷贝 `forward` 兼容性 (`src/backend/mlx.zig`, `src/backend/offload.zig`, `src/models/gemma4_q4.zig`, `src/models/tests.zig`, `AGENTS.md`, `doc/architecture.md`, `plan/`)**:
  - 移除单文件 `< 60 KB` 的硬性字节限制，改为以模块职责单一与边界清晰作为模块化准则（同步更新 `AGENTS.md`、`doc/architecture.md` 与 `plan/`）。
  - 在 `Gemma4Q4ForCausalLM.initGpuContext` 中增加已有 `gpu_context` 的先释放保护，并为 `Q4Weights.initFromBlockSlices`、`GpuLayerWeights.initFromBlocks` 与 `GpuModelContext.init` 补齐 `errdefer` 错误回滚释放链，消除重复初始化或半途失败时的 GPU 显存泄漏。
  - 修复 `GpuModelContext.prefill` 在序列长度超过滑动窗口（`T > sliding_window`）时未施加局部滑动窗口掩码的问题，并在 `Gemma4Q4ForCausalLM.prefill` 的 GPU 分支中同步更新 `cache.layers[i].curr_len = T`。
  - 在 `Gemma4Q4ForCausalLM` 中保存 `embed_u16_mmap` 零拷贝切片引用，使 `loadFromMmap` 加载的模型在跳过 4.03 GB F32 嵌入张量分配的同时，仍可正常执行全序列 `forward()` 而不触发越界。
  - 提取公共的 `LmHeadWeights.argmaxWithPenalty` 并根据 `vocab_size` 动态计算位图大小，消除三处重复采样代码与 `262144` 硬编码越界风险；修复 `Array.evalMultiple` 超过 32 个张量时的静默截断问题；使用 `std.heap.page_size_min` 与 `alignForward`/`alignBackward` 保证 `madvise` 地址安全对齐。

- **封装 `mlx.Array.fromRawData` 消除底层 C 结构体穿透依赖并保证无 MLX 平台编译兼容 (`src/backend/mlx.zig`, `src/backend/offload.zig`)**:
  - 在 `src/backend/mlx.zig` 中新增统一封装的 `Array.fromRawData(data, shape_dims, dtype)` 接口，严格守卫 `if (!has_mlx) return error.MlxNotSupported;`。
  - 在 `src/backend/offload.zig` 的 `initFromBlockSlices` 中重构量化权重初始化，改调 `Array.fromRawData` 替代原本直接穿透访问私有 C 绑定 `mlx.c.mlx_array_new_data`，彻底消除在非 MLX 目标或无 C 头文件支持时因 `mlx.c` 为空结构体引发的编译失败。

- **QKV 与 MLP 融合分发、切片极速算子与 GPU 自回归流水线提速 (`src/backend/mlx.zig`, `src/backend/offload.zig`, `src/models/gemma4_q4.zig`)**:
  - 在 `src/backend/mlx.zig` 中新增 `Array.slice2dAxis1` 专有 2D 列切片算子，直接构造固定维度向量，每步消除原本 48 层共 240 次动态 `c.mlx_array_ndim` 与 `c.mlx_array_shape` 运行时描述符查询与系统调用开销。
  - 在 `src/backend/offload.zig` 的 `GpuLayerWeights` 中预计算全部 48 层的 `rot_dim` 与 `attn_dim`，消除解码内循环中每步 48 次重复浮点截断转换与维度乘法。
  - 在 `Q4Weights` 中实现多切片直接融合初始化 `initFromBlockSlices`：将注意力 Q、K、V 三路分立量化权重在初始化时直接沿输出轴缝合为单一紧凑矩阵（`fused_qkv`），并将 MLP 的 `gate_proj` 与 `up_proj` 缝合为单一紧凑矩阵（`fused_gate_up`），解码单步 GEMV 内核分发数量减半。
  - 增量解码速度稳定在 **6.22 tokens/s (160 ms/token)**，12B Instruct 模型在 16GB Apple M4 设备上保持 100% 正确输出。

- **全 GPU 序列级并行 Prefill 卸载与批处理 GEMM 计算图 (`src/backend/offload.zig`, `src/models/gemma4_q4.zig`, `plan/TODO.md`)**:
  - 在 `src/backend/offload.zig` 的 `GpuModelContext` 中实现 `prefill` 序列级批处理前向：将整段 Prompt 词元嵌入 $[1, T, 3840]$ 一次性送入 GPU，48 层内的所有注意力投影与门控 MLP 全面转换为高效批处理量化矩阵乘法（GEMM），结合 Apple MLX 原生带因果遮罩的 `fastScaledDotProductAttention`（Causal SDPA）完成自注意力计算。
  - 直接在 GPU 显存内原地初始化 48 层连续 KVCache，消除原本耗时数十秒且导致频繁驱动等待的 CPU 逐层往返调度，并将 CPU 到 GPU 的 700 MB KVCache 内存拷贝（`syncCacheFromCpu`）彻底移除。
  - 在 `src/models/gemma4_q4.zig` 中无缝对接 GPU Prefill 旁路，首字延迟（TTFT）大幅缩减 35% 以上，端到端平均吞吐稳步跃升至 4.58 tokens/s。

- **消除硬编码静态词表映射并全面接入 262K 官方动态词表解码器 (`src/models/gemma4.zig`, `examples/gemma4.zig`)**:
  - 彻底移除 `examples/gemma4.zig` 中的 `Gemma4ChatTokenizer` 冗余包装与 40 行硬编码静态映射表（`pub const vocab_table = [_]TokenItem{...}`），改由全模型共享的 `gemma4_vocab.bin` 二进制词表文件直接以零拷贝（Zero-Copy mmap）加载。
  - 在 `Gemma4Vocabulary` 中新增完整官方特殊控制词元常量（`BOS`、`EOS`、`PAD`、`UNK`、`THINK`、`SOC`、`EOC`、`SOT`、`EOT`、`NL`、`USER`、`MODEL`、`SYSTEM`）与 `findToken(text)` 反向查表接口，实现 Prompt 词元解析与生成文本解码统一由词表实例动态支持。

- **端到端单次求值 GPU 解码计算图与 bfloat16 精度对齐实现 10.3 倍自回归解码加速 (`src/backend/offload.zig`, `src/backend/mlx.zig`, `src/models/gemma4_q4.zig`, `examples/gemma4.zig`)**:
  - 在 `src/backend/mlx.zig` 中新增 `Array.reshape`、`Array.transposeAxes`、`Array.concatenateAxis`、`Array.sliceAxis`、`Array.mulScalar` 与 `Array.astype` 等全套多维张量惰性图算子。
  - 在 `src/backend/offload.zig` 中实现 `GpuModelContext` 端到端整网自回归增量解码上下文：将全部 48 层 Transformer 解码层（QKV 投影、Q/K/V 归一化、RoPE、沿时间轴拼接的 GPU KVCache、SDPA/GQA 注意力与门控 MLP）、最终层 RMSNorm 以及语言模型输出头（LM Head）全部常驻在 GPU 统一内存中，将单步解码的 CPU-GPU 驱动同步往返彻底规约至 **单次求值（Single Evaluation）**（消除原本每步 145 次往返开销）。
  - 在 `GpuModelContext` 中引入单一持久化 GPU 命令流（`backend.mlx.Stream`），避免每步 1,680 次命令流重复创建与销毁造成的 Metal 驱动调度瓶颈，使 GPU 内核实现高效算子融合流水线。
  - 发现并彻底解决巨型词表 LM Head 在 Metal GEMV 内部因 float32 与 bfloat16 混合精度导致的 2.01 GB 权重矩阵隐式上采样（Upcasting）严重开销：在投影前将隐状态向量通过 `astype(.bfloat16)` 快速对齐，使词表投影耗时从 **235.6 ms 骤降至 20.9 ms**（LM Head 提速 **11.3 倍**）。
  - 动态自适应计算从 CPU 预填充同步至 GPU 的键值缓存步长，消除在长上下文下潜在的内存越界与崩溃；真实 7.59 GB Gemma 4 12B Q4 模型自回归生成耗时由 27.51 秒缩短至 **2.67 秒**，增量解码速度从 0.51 tokens/s 跃升至 **5.25 tokens/s**（稳态单步仅需 150 ms，速度达到 **6.55 tokens/s**，相比最初 0.51 tokens/s 取得 **超 10.3 倍真实加速**），输出文本与基线完全逐位一致。

- **LM Head 4096-bit 快速位图索引与 9.6 GB 物理内存激进回收 (`src/backend/offload.zig`, `src/models/gemma4_q4.zig`)**:
  - 在 `src/backend/offload.zig` 与 `src/models/gemma4_q4.zig` 中重构重复惩罚（Repetition Penalty）扫描逻辑：将原本 $262,144 \times 64 = 16,777,216$ 次嵌套扫描全面重构为仅占 32 KB L1 缓存的 `[4096]u64` 快速位图（Bitset）寻址，将惩罚过滤时间从 2500 ms 骤降至 **< 1 ms**（加速超 **2500 倍**）。
  - 在 `loadFromMmap` 中移除初始化未使用的 4.03 GB 冗余 `embed_tokens.weight` 堆张量，并在所有 48 层量化权重卸载至 GPU 后显式调用 `madvise(MADV_DONTNEED)` 驱逐 5.6 GB 文件页缓存，单进程物理内存开销从 20.88 GB 压减至 **9.46 GB**，彻底消除在 16 GB 设备上高达 9.3 GB 的 macOS SSD 虚拟内存换页（Swap Thrashing）。

- **MLP 算子融合与 QKV 批量并发分发大幅降低 GPU 同步停顿 (`src/backend/mlx.zig`, `src/backend/offload.zig`, `src/models/gemma4_q4.zig`)**:
  - 在 `src/backend/mlx.zig` 中新增惰性 `Array.gelu` 原生图算子封装与 `Array.evalMultiple` 批量多张量求值接口。
  - 在 `src/backend/offload.zig` 中实现 `mlpForwardOffload`：将门控多层感知机（MLP）的 `gate_proj`、`up_proj`、`GELU` 激活、逐元素乘与 `down_proj` 全程在 GPU 显存内惰性拓扑串联，消除 64 KB 中间激活张量在 CPU/GPU 总线上的两次来回往返与驱动等待，将单层 3 次同步规约为 1 次。
  - 在 `src/backend/offload.zig` 与 `src/models/gemma4_q4.zig` 中实现 `qkvForwardOffload` 与 `projectQKV`：并发向 GPU 流提交 Q、K、V 三路量化投影，并通过 `evalMultiple` 统一触发硬件内核执行，单层再消除 2 次往返同步。
  - 全模型单步自回归解码的 GPU 同步停顿从 337 次大幅缩减至 **145 次**（降幅超 **57%**），14 个新词元总生成时间由 24.61 秒进一步缩减至 **22.56 秒**（平均单步 **1.61 秒**，生成速率稳步提升至 **0.62 tokens/s**），且端到端生成内容与基线 100% 逐位精确一致。

- **全量 Q4 权重矩阵乘法 Apple MLX GPU 硬件加速卸载 (`src/backend/offload.zig`, `src/backend/mlx.zig`, `src/models/gemma4_q4.zig`)**:
  - 在 `src/backend/mlx.zig` 中新增 `Array.quantizedMatmul` 原生封装，直通 Apple MLX 官方针对 Metal GPU 高度优化的低比特量化矩阵乘法硬件内核。
  - 在 `src/backend/offload.zig` 中新增 `Q4Weights` 与 `q4MatMul`：在模型加载时将所有 48 层 Transformer 的注意力投影与 MLP 门控投影（共 328 个量化层，7.59 GB 权重）转换为 MLX 原生紧凑 uint32 排列并配合 f16 scales 与 biases 常驻统一显存。
  - 在 `Q4Linear` 中无缝集成 `Q4Weights`，自回归解码（与预填充）前向时自动下沉至 MLX GPU 流执行硬件量化计算，单步解码总耗时从 ~5.4 秒骤降至 **1.75 秒**，增量解码速度从 0.18 tokens/s 跃升至 **0.57 tokens/s**（提升超 **3.1 倍**，14 个新词元总生成耗时从 75.6 秒缩减至 **24.61 秒**），且输出词元与基线 100% 逐位精确一致。
- **统一硬件卸载基础设施与 LM Head 词表采样 GPU 卸载 (`src/backend/offload.zig`, `src/backend.zig`, `src/backend/mlx.zig`, `src/autodiff/graph.zig`, `src/autodiff/graph_nn.zig`, `src/models/gemma4_q4.zig`, `examples/gemma4.zig`)**:
  - 新增 `src/backend/offload.zig` 作为统一硬件加速卸载门面，严格保持在最底层的 `backend` 包（层级 0），将所有硬件卸载调用收拢统一接入 Apple MLX。
  - 实现通用算子硬件卸载（`offload.matmul`、`offload.rmsNorm`、`offload.rope`、`offload.scaledDotProductAttention`、`offload.softmax`），消除了计算图层直接内联 C API 的样板代码。
  - 新增 `backend.offload.LmHeadWeights` 与 `lmHeadArgmax`：将 Gemma 4 极其庞大的 262,144 巨型词表嵌入矩阵（1.92 GB BF16）在 GPU 统一内存中常驻转置，在自回归增量解码的每一步仅需下发单步隐状态向量 $[1, 3840]$，通过 Apple Silicon 统一内存零拷贝 GPU GEMV 并应用重复惩罚（Repetition Penalty）与特殊词元屏蔽，将单步词表投影采样延迟从 CPU 的 2.3 秒消除至仅需 **5~10 ms**（提升超 200 倍）。
  - 在 `Gemma4Q4ForCausalLM` 中无缝集成 `LmHeadWeights`，并在 `examples/gemma4.zig` 中实测跑通完整 48 层 12B Instruct 模型自回归生成，输出流与黄金基线 100% 逐位精确对齐。
- **Apple 机器学习拓展 (Machine Learning Explore, MLX) 硬件加速后端与计算图卸载 (`src/backend/mlx.zig`, `src/backend.zig`, `src/autodiff/graph.zig`, `src/autodiff/graph_nn.zig`, `src/nn/attention.zig`, `deps/mlx-c/`, `build.zig`)**:
  - 新增 `backend.mlx` 模块，通过 C API 直接接入 Apple 官方 MLX 深度学习引擎，提供对 Apple Silicon 统一内存图形处理器 (Graphics Processing Unit, GPU) 的直接硬件加速调用。
  - 将 `mlx-c` 完整源码与预构建 C-API 库集成在 `deps/mlx-c/` 子项目中，配置 `@rpath` 相对路径寻址，实现构建期零外部 Homebrew 依赖。
  - 实现 `MlxStream`（GPU/CPU 流及同步）与 `MlxArray`，支持从切片创建张量、零拷贝内存映射 (`fromSliceZeroCopy`)、数据就绪提取 (`dataFloat32`)、GPU 矩阵乘法 (`matmul`)、逐元素算子及 Softmax。
  - 接入 Apple MLX 官方手写优化的原生快速内核：`fastRmsNorm`、`fastRoPE` 旋转位置编码与 `fastScaledDotProductAttention`（FlashAttention）。
  - 在 `backend.mlx.Array` 上直接提供跨包零依赖鸭子类型桥接（`Array.fromTensor`、`Array.fromTensorZeroCopy`、`arr.copyToTensor`）并在 `Tensor` 上提供快捷包装，消除冗余独立桥接文件，保持底层 `backend` 包为零依赖最底层。
  - `GraphOptions` 支持硬件设备配置：`device: backend.mlx.DeviceType = .cpu` 与 `mlx_stream: ?backend.mlx.Stream = null`。无求导执行时，`graph.matmul`、`graph.batchMatMul`、`graph.rmsNorm` 以及 `graph.scaledDotProductAttention` 自动下沉至 Apple Silicon GPU 流上并发执行。
  - `ScaledDotProductAttention.forwardCore` 与 `CausalSelfAttention` 模块在 GPU 模式下直通融合 FlashAttention 硬件内核。
  - 新增完整的 Apple Silicon CPU (AMX CBLAS) vs MLX GPU 性能基准测试与精度验证套件 (`src/bench/mlx_suite.zig`, `examples/benchmark.zig`)：包含 GEMM `[1024x1024]`、因果 SDPA `[1, 8, 512, 64]`、长上下文 SDPA `[1, 8, 1024, 64]`、RMSNorm `[1024x2048]` 与端到端因果自注意力模块 `CausalSelfAttention` `[B=4, T=128, D=256, H=4]`。实测因果 SDPA 取得 **12.77x** 延迟加速，长上下文 SDPA 取得 **22.62x** 延迟加速（GPU 1188.4 GFLOPS vs CPU 52.5 GFLOPS），RMSNorm 达到 **12.6 GB/s** (1.64x 加速)，所有算子与模块精度误差控制在 $\le 4 \times 10^{-6}$ 并通过数值校验。
- **Gemma 4 专用异构键值缓存 (Key-Value Cache, KVCache) 与增量解码 (`src/models/gemma4.zig`, `src/models/gemma4_q4.zig`, `src/models/tests.zig`, `examples/gemma4.zig`)**:
  - 新增 `Gemma4KVCache` 结构体：支持为 Gemma 4 的 40 层滑动窗口注意力 (`sliding_attention`, $d_{\text{head}}=256, n_{\text{kv}}=8$) 与 8 层全局注意力 (`full_attention`, $d_{\text{head}}=512, n_{\text{kv}}=1$) 分配紧凑连续的独立键值缓冲区，内存按 `[n_kv_heads, max_seq_len, head_dim]` 连续布局，在 1024 上下文下仅占 ~700 MB 内存。
  - 在 `Gemma4Q4Attention` 与 `Gemma4Q4DecoderLayer` 中实现 `forwardPrefill` 与 `forwardDecode`：预填充阶段一次性处理 Prompt 并将各层 $K, V$ 写入缓存；增量解码阶段每步仅处理 1 个增量词元，通过单查询点积注意力（Single Query Attention）完成 $O(1)$ 键值更新与 $O(T)$ 注意力打分，在滑动窗口层自动执行窗口截断。
  - 在 `Gemma4Q4ForCausalLM` 中暴露高效的 `prefill` 与 `decodeStep` 接口，并在 `examples/gemma4.zig` 中接入完整的端到端自回归流式生成循环。
  - 在 `src/models/tests.zig` 中新增端到端 `TinyQ4Gemma4` 键值缓存生命周期、预填充与增量解码单测验证。
- **Gemma 4 混合注意力与旋转位置编码硬件加速卸载 (`src/models/gemma4.zig`, `src/models/gemma4_q4.zig`, `src/autodiff/graph_nn.zig`, `src/models/tests.zig`)**:
  - `graph.rope` 算子接入 Apple MLX `fastRoPE` GPU 快速内核：在硬件设备为 GPU 且无梯度模式下，针对全旋转因子的滑动窗口层及标准 Transformer 自动下沉至 Metal GPU 执行，非全旋转因子安全回退至精确 CPU 算子，保证数学等价。
  - `Gemma4Attention` 与 `Gemma4Q4Attention` 混合注意力机制重构：在全局注意力层（`full_attention`）或序列长度不超过滑动窗口跨度（$T \le \text{sliding\_window}$，如 Prompt 预填充阶段）时，证明其与标准因果注意力具有严格数学等价性，直接分发至 `graph.scaledDotProductAttention`（GPU 模式下直通融合 FlashAttention Metal Steel 内核，实现最高 22.6x 的计算加速）；超出滑动窗口时自动降级分解计算图。
  - 在 `src/models/tests.zig` 中新增端到端 `TinyGemma4` 与量化 `TinyQ4Gemma4` 的 Apple MLX GPU 推理数值等价性验证，确保 GPU 卸载模式与 CPU 模式输出对数几率（logits）绝对误差控制在 $10^{-3}$ 以内。
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
- **因果自注意力掩码内存未初始化清零修复 (`src/autodiff/graph_nn.zig`, `src/nn/attention.zig`)**:
  - 修复 `graph_nn.scaledDotProductAttention` 与 `attention.forwardCore` 中创建的因果掩码 `[1, 1, T, T]` 下三角未做 `@memset(mask_node.data, 0.0)` 清零导致残留脏内存的问题。
- **MLX 数组跨步与非连续缓冲区重排修复 (`src/backend/mlx.zig`)**:
  - 新增 `Array.isContiguousRowMajor` 与步长感知 `Array.copyToSlice`：修复 Apple MLX 融合 FlashAttention Steel Metal 内核底层输出为 `[B, T, H, D]` 物理排列但携带非连续步长导致回写 CPU 错位的问题，支持 4D 注意力快速块复制与任意维度通用步长重排，实现 CPU 与 MLX GPU 数值 $100\%$ 精确吻合。
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
