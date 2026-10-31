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

根据最新架构再评估，系统将所有后续任务划分为严格的优先级序列（**P0 优先级最高，列于最前**）：

| 优先级 / 阶段 | 文件 | 核心任务 | 当前状态 |
| :--- | :--- | :--- | :--- |
| **P0 优先级**<br>*(前置保障)* | [`03_execution_graph_and_buffer_reuse.md`](03_execution_graph_and_buffer_reuse.md) §0 | **后端 Comptime 架构检测、构建链自适应探测与三大运行时缺陷修复**：<br>1. `build.zig` 增加 MLX/CUDA 依赖库物理探测，库缺失时安全回退，杜绝 `-lmlxc` 链接错误，实现开箱即编译；<br>2. `src/backend/` 支持 `comptime` 目标架构检测与 SIMD 向量宽度选择，无用后端代码编译期彻底剔除；<br>3. 彻底修复 GPU/MLX 模式下 3 大运行时缺陷（`write-after-free`、设备错配、转置连续性校验）；<br>4. 清理 `chen-gz.github.io` 18 处本地绝对路径与死链。 | **🔥 准备就绪 (Ready for Execution)** |
| **P1 优先级**<br>*(M3 核心)* | [`03_execution_graph_and_buffer_reuse.md`](03_execution_graph_and_buffer_reuse.md) §1–§5 | **静态执行图、缓冲区复用与自动微分重放 (Milestone 3)**：<br>1. 前向重演彻底根除 51 处 `copyFromEager`，全量接入 `*Into` 算子直写；<br>2. 反向求导彻底根除 8 处 `std.heap.c_allocator`，注入工作区分配器；<br>3. 静态内存生命周期分析与缓冲区复用引擎（`memory_planner.zig`，激活显存下降 90%+）；<br>4. `Tensor.grad` 一等公民重构（GPU 原生驻留、懒分配、Prologue 提升切片累加）。 | **⏳ 待启动 (Queued)** |
| **P2 优先级**<br>*(M3 扩展)* | [`03_execution_graph_and_buffer_reuse.md`](03_execution_graph_and_buffer_reuse.md) §2.5 | **平台中立调度与显存切片抽象**：<br>1. 解耦 `GraphOptions` 中的 `mlx_stream` 为平台中立句柄；<br>2. `DeviceBuffer` 支持零拷贝子显存切片（Sub-Buffer Slicing）。 | **⏳ 待启动 (Queued)** |
| **P3 优先级**<br>*(M4)* | [`04_nn_module_protocol_and_layer_composability.md`](04_nn_module_protocol_and_layer_composability.md) | **PyTorch 风格 `nn.Module` 协议与图层标准化**：<br>1. 编译期模块契约内省（`verifyModule(T)`）；<br>2. 严格区分可训练参数与持久缓冲区（`createParameter` / `createBuffer`）；<br>3. `Q4Linear` 接入 `walk` 反射，统一标准 `KVCache`。 | **⏳ 待启动 (Queued)** |
| **P4 优先级**<br>*(M5)* | [`05_composable_models_gpt_and_gemma4.md`](05_composable_models_gpt_and_gemma4.md) | **核心模型全模块化重构 (`GPT` & `Gemma 4`)**：<br>1. 消除模型层过程式裸指针偏移与裸切片索引；<br>2. 统一纯粹声明式模块调用，推理/训练执行引擎彻底剥离至 `engine/`。 | **⏳ 待启动 (Queued)** |
| **P5 优先级**<br>*(M6)* | [`06_comprehensive_e2e_verification_and_hardening.md`](06_comprehensive_e2e_verification_and_hardening.md) | **端到端全量验证与对抗性加固**：<br>1. 全自动化矩阵测试（全平台、全优化级别 0 内存泄漏）；<br>2. 对抗性边界测试与 Web 可视化器 Schema 2.0 契约对齐。 | **⏳ 待启动 (Queued)** |

---

## 📜 已完成里程碑归档 (Completed Milestones Archive)

- **[Milestone 1] 硬件后端严格隔离与 HAL 建立** ([`01_backend_isolation_and_clean_hal.md`](01_backend_isolation_and_clean_hal.md))：
  - 交付：纯物理 `DeviceBuffer`、`kernels.zig`、消除 566 行 duck-typing `ops.zig`、迁移 6 大错配模块、层级依赖检查器通过。
  - 状态：**✅ Done & Merged to `main`**。
- **[Milestone 2] 张量物理存储统一与算子零分配体系** ([`02_tensor_storage_and_zero_alloc_ops.md`](02_tensor_storage_and_zero_alloc_ops.md))：
  - 交付：淘汰 `Tensor` 废弃字段（`device` 与 `device_array`），统一在 `storage: backend.DeviceBuffer` 下；新增 6 类零分配目标写入 `*Into` 算子；全量 418 项测试通过，0 内存泄漏；18 个经典模型 JSON 导出通过；Gemma 4 E4B 4-bit 量化实测 127 token/s prefill 与 21.3 token/s decode。
  - 状态：**✅ Done & Merged to `main` (Commit `bc6981a4`, `c82f67a5`)**。
