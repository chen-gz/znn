# ZNN AI 任务规划与提示词中心 (AI Prompts & Phased Task Hub)

本目录（`prompts/`）专门用于存放由 AI 生成的高阶架构重构提示词 (Prompts)、接口契约、执行蓝图以及分阶段交付清单。

---

## 🎯 设立宗旨与工程规范 (Design & Rules)

1. **Token 限制与持久化断点恢复 (Token-Budget Resilience)**：
   - 现代 LLM 具有单次对话 Token 窗口与配额限制，大型架构重构必然经历多轮会话或由不同 Agent 串行接力。
   - 将结构完整、自包含且带有明确验收标准的 Prompt 持久化在版本库中，确保任何 AI 在开启新会话时能够**秒级恢复完整上下文**，彻底杜绝凭空臆测与重复劳动。

2. **所有 AI 任务开始前必须建档 (Mandatory Pre-Execution Logging)**：
   - 任何 AI Agent 在启动复杂任务（包括多文件重构、包层级调整、性能优化）前，**必须首先在本目录下编写或确认目标阶段的 Prompt 文件**。
   - 文件须明确说明：涉及的文件清单、输入输出类型签名变更、破坏性修改说明、以及客观验证命令（如 `zig build check-layers`、`zig build test`）。

3. **里程碑完成后的归档更新 (Post-Milestone Status Sync)**：
   - 当某个阶段完成并全量测试通过后，推进的 AI 需更新对应 Prompt 的状态（标记为 `[x] Completed` 并附上测试通过指标），并将下一个里程碑的 Prompt 就绪，伴随代码一同提交入库。

---

## 🗺️ 架构演进与优先级索引表 (Master Priority & Phase Index)

根据最新架构再评估，系统将所有后续任务划分为严格的优先级序列（历史已完成里程碑全部合入 `main`，新一阶段以 M11 为首要推进目标）：

| 优先级 / 阶段 | 文件 | 核心任务 | 当前状态 |
| :--- | :--- | :--- | :--- |
| **P0 优先级**<br>*(前置保障)* | [`03_execution_graph_and_buffer_reuse.md`](03_execution_graph_and_buffer_reuse.md) §0 | **后端 Comptime 架构检测、构建链自适应探测与三大运行时缺陷修复**：<br>1. `build.zig` 增加 MLX/CUDA 依赖库物理探测，实现开箱即编译；<br>2. `src/backend/` comptime 目标架构检测与 SIMD 宽度选择；<br>3. 修复 GPU/MLX 模式下 3 大运行时缺陷；<br>4. 清理 `chen-gz.github.io` 25 处本地绝对路径与死链。 | **✅ 已完成并合入 main (`b3237149`, `9ae21b03`)** |
| **P1 优先级**<br>*(M1-M5)* | [`01_backend_isolation_and_clean_hal.md`](01_backend_isolation_and_clean_hal.md) ~ [`05_composable_models_gpt_and_gemma4.md`](05_composable_models_gpt_and_gemma4.md) | **架构分层、张量统一存储、执行图重放、`nn.Module` 协议与核心模型重构**：<br>1. 8 层严格单向依赖与纯物理 `DeviceBuffer`；<br>2. 目标写入 `*Into` 算子体系与零堆分配；<br>3. `MemoryPlanner` 显存复用与 Arena 注入；<br>4. PyTorch 风格模块反射与声明式前向图。 | **✅ 已完成并合入 main (`8f4e87d1` ~ `9a691413`)** |
| **P1 优先级**<br>*(M7/M8/ResNet)* | [`07_convnext_vision_architecture.md`](07_convnext_vision_architecture.md), [`08_comprehensive_datasets_tiny_imagenet_and_multimodal.md`](08_comprehensive_datasets_tiny_imagenet_and_multimodal.md) | **现代纯卷积 ConvNeXt、经典 ResNet 全套模型与多模态数据集**：<br>1. `groups` 深度卷积、`LayerNorm2d`、ConvNeXt 架构族与梯度验证；<br>2. ResNet-18~152 / CIFAR-20~110 残差架构；<br>3. Tiny ImageNet 紧凑二进制、WikiText-103 与多模态图文数据流。 | **✅ 已完成并合入 main (`6b6b9464`, `6a233edb`, `5ee4f515`)** |
| **P1 优先级**<br>*(M10)* | [`10_qr_decomposition_and_least_squares.md`](10_qr_decomposition_and_least_squares.md) | **正交三角分解 (QR) 与高精度最小二乘求解**：<br>1. 完整与精简 Householder QR 分解；<br>2. 上三角回代求解器 `solveTriangularUpper`；<br>3. 通用张量最小二乘 `qrSolve` 与高精度 OLS 回归。 | **✅ 已完成并合入 main (`39224590`)** |
| **P1 优先级**<br>*(M11 首要)* | [`11_advanced_indexing_and_strided_views.md`](11_advanced_indexing_and_strided_views.md) | **高级张量索引、跨步切片视图与离散累加体系**：<br>1. 通用轴向索引收集 `gather(dim, index)`；<br>2. 通用离散散布与累加 `scatter` / `scatterAdd` (含 `scatterAddInto` / `scatterInto`)；<br>3. 子张量一维切片与累加 `indexSelect` / `indexAdd` 与掩码过滤 `maskedSelect`；<br>4. 跨步切片带符号步长与逆序物化 (`step < 0`)；<br>5. 离散索引操作的 Autograd 计算图与梯度回传闭环。 | **🔥 下一阶段待启动 (Next in Queue)** |
| **P1 优先级**<br>*(M12 待办)* | `12_multiaxis_reductions_and_statistics.md` | **多轴联合归约与进阶统计算子库**：<br>1. 多轴切片联合归约 `sumAxes`, `meanAxes`, `varianceAxes`；<br>2. 累乘 `prod`、前缀扫描 `cumsum` / `cumprod`；<br>3. 通用向量与矩阵范数 `norm` (L1, L2, Frobenius, Inf)；<br>4. 分位数 `quantile` 与中位数 `median`；<br>5. `OpType.Max` / `OpType.Min` Autograd 梯度路由与 `logSumExp`。 | **⏳ 待启动 (Queued)** |
| **P2 优先级**<br>*(M13 待办)* | `13_advanced_linear_algebra_and_solvers.md` | **进阶数值线性代数库与求解器**：<br>1. 对称正定矩阵 Cholesky 分解 ($A = LL^T$)；<br>2. 下三角前代求解器 `solveTriangularLower` 与 `choSolve`；<br>3. 显式方阵求逆 `inv` 与基于 SVD 的摩尔-彭若斯伪逆 `pinv`；<br>4. 行列式 `det` 与数值稳定对数绝对值行列式 `slogdet`；<br>5. 轻量级爱因斯坦求和 `einsum` 引擎。 | **⏳ 待启动 (Queued)** |
| **P2 优先级**<br>*(M14 待办)* | `14_modern_scientific_computing_and_distributions.md` | **现代科学计算与可微概率分布体系**：<br>1. 线程安全独立 `Generator` 对象与张量随机排列 `permutation`/`shuffle`；<br>2. 连续与离散概率分布体系 (`Normal`, `Uniform`, `Categorical` 等)；<br>3. 重参数化可微路径采样 `rsample` 与对数概率密度 `log_prob`；<br>4. 特殊数学函数库（`erfinv`, `gammaln`, `digamma`, `log1p`, `expm1`）。 | **⏳ 待启动 (Queued)** |
| **P3 优先级**<br>*(M6 门禁)* | [`06_comprehensive_e2e_verification_and_hardening.md`](06_comprehensive_e2e_verification_and_hardening.md) | **端到端全量验证与对抗性加固**：<br>1. 矩阵自动化测试（全平台、全优化级别 0 内存泄漏）；<br>2. 极端上下文窗口与内存压力 OOM 边界测试。 | **🛡️ 持续测试门禁 (Active QA Gate)** |

---

## 📜 已完成里程碑归档 (Completed Milestones Archive)

- **[Milestone 10] 正交三角分解 (QR Decomposition) 与高精度最小二乘求解** ([`10_qr_decomposition_and_least_squares.md`](10_qr_decomposition_and_least_squares.md))：
  - 交付：`QROptions` 与 `QRMode`（支持 `.reduced` 与 `.complete`），完整 Householder QR 分解 ($Q \in \mathbb{R}^{M \times M}$ 正交基)，上三角回代求解器 `solveTriangularUpper`，通用最小二乘求解器 `qrSolve`，线性回归高精度 QR 求解器 `solveLeastSquaresQR`（条件数严格保持 $O(\kappa)$）；全量 433 项测试通过，0 泄漏。
  - 状态：**✅ Done & Merged to `main` (`39224590`)**。

- **[Milestone 8] 综合测试数据集扩展** ([`08_comprehensive_datasets_tiny_imagenet_and_multimodal.md`](08_comprehensive_datasets_tiny_imagenet_and_multimodal.md))：
  - 交付：3 通道 RGB `ImageDataset`；Tiny ImageNet 紧凑二进制格式解析器与合成数据生成器；WikiText-103 词表与自回归因果数据集；`MultimodalDataset` 图文配对数据集与批次流水线；`download-data` CLI 扩展与交互帮助；`examples/multimodal.zig`（21+ pairs/s），0 泄漏。
  - 状态：**✅ Done & Merged to `main` (`5ee4f515`, `992418e0`)**。

- **[Milestone 7] ConvNeXt 现代纯卷积视觉模型架构** ([`07_convnext_vision_architecture.md`](07_convnext_vision_architecture.md))：
  - 交付：`ConvOptions.groups` 分组卷积与 7x7 深度可分离卷积；反向传播数值梯度检验（`gradcheck` 100% 对齐）；通道优先 `LayerNorm2d`；`ConvNeXtBlock`（反向瓶颈 + 7x7 DWConv + GELU + LayerScale）；全套模型预设（Femto/Pico/Nano/Tiny/Base）；`examples/convnext.zig`（训练 186.97 img/s，推理 635.49 img/s，测试准确率 75.10%）。
  - 状态：**✅ Done & Merged to `main` (`6b6b9464`)**。

- **[论文复现] ResNet 论文经典架构全套实现**：
  - 交付：`models.resnet` 实现 ResNet-18/34 (`BasicBlock`)、ResNet-50/101/152 (`Bottleneck`)、CIFAR-20/32/44/56/110；残差直连与步长投影下采样；`examples/resnet.zig` 训练速度 876+ img/s，验证集准确率 85.05%。
  - 状态：**✅ Done & Merged to `main` (`6a233edb`, `f70130db`, `d22f88a0`)**。

- **[Milestone 5] 核心模型全模块化重构 (`GPT` & `Gemma 4`)** ([`05_composable_models_gpt_and_gemma4.md`](05_composable_models_gpt_and_gemma4.md))：
  - 交付：`autodiff.Graph.arange` 动态无泄漏索引分配；`models.GPT` 位置编码与线性多维张量流重构；`decodeAttention` 全量接入 `graph.attention` 向量化 SGEMM，消除裸切片与指针循环；前端 Web 可视化 203 测试与 18 个模型架构 JSON 导出全通。
  - 状态：**✅ Done & Merged to `main` (`9a691413`, `42050e5a`)**。

- **[Milestone 4] PyTorch 风格 `nn.Module` 协议与图层标准化** ([`04_nn_module_protocol_and_layer_composability.md`](04_nn_module_protocol_and_layer_composability.md))：
  - 交付：编译期 `verifyModule` 断言；`createParameter` 与 `createBuffer` 显式解耦；`BatchNorm` 运行统计量持久化缓冲且免疫 `setRequiresGrad`；`buffers`/`namedBuffers`/`submodules` 反射与 `ModuleList` 容器；428 项测试全通。
  - 状态：**✅ Done & Merged to `main` (`4b2f5d8f`)**。

- **[Milestone 3] 静态执行图优化与缓冲区复用** ([`03_execution_graph_and_buffer_reuse.md`](03_execution_graph_and_buffer_reuse.md))：
  - 交付：消除 51 处 `copyFromEager` 直写输出；反向消除 8 处 `c_allocator`；实现 `MemoryPlanner` 与 `ExecutionGraph`，激活显存降低 $\ge 85\%$，424 项测试全通。
  - 状态：**✅ Done & Merged to `main` (`dc348622`, `dedc47a1`)**。

- **[Milestone 2] 张量物理存储统一与算子零分配体系** ([`02_tensor_storage_and_zero_alloc_ops.md`](02_tensor_storage_and_zero_alloc_ops.md))：
  - 交付：淘汰 `Tensor` 废弃字段（`device` 与 `device_array`），统一在 `storage: backend.DeviceBuffer` 下；新增 6 类零分配目标写入 `*Into` 算子；全量 418 项测试通过，0 内存泄漏；18 个经典模型 JSON 导出通过；Gemma 4 E4B 4-bit 量化实测 127 token/s prefill 与 21.3 token/s decode。
  - 状态：**✅ Done & Merged to `main` (`bc6981a4`, `c82f67a5`)**。

- **[Milestone 1] 硬件后端严格隔离与 HAL 建立** ([`01_backend_isolation_and_clean_hal.md`](01_backend_isolation_and_clean_hal.md))：
  - 交付：纯物理 `DeviceBuffer`、`kernels.zig`、消除 566 行 duck-typing `ops.zig`、迁移 6 大错配模块、层级依赖检查器通过。
  - 状态：**✅ Done & Merged to `main` (`8f4e87d1`, `df3835ec`)**。

- **[P0 前置修复] 编译期架构检测与构建工具链自适应** ([`03_execution_graph_and_buffer_reuse.md`](03_execution_graph_and_buffer_reuse.md) §0)：
  - 交付：`probeMlxAvailable`、`src/backend/target.zig`、修复 3 处 GPU 悬垂指针与转置缺陷、清理文档全部 25 处绝对路径。
  - 状态：**✅ Done & Merged to `main` (`b3237149`, `9ae21b03`)**。
