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

## 🗺️ 架构重构演进路线图索引 (Phase 6 Master Index)

| 文件 | 阶段 | 核心任务 | 状态 |
| :--- | :--- | :--- | :--- |
| [`00_master_architecture_overhaul.md`](00_master_architecture_overhaul.md) | **全景总览** | ZNN 架构深度重构总目标、审计发现与 6 大里程碑总体规划 | **基线已建立** |
| [`01_backend_isolation_and_clean_hal.md`](01_backend_isolation_and_clean_hal.md) | **M1** | 硬件后端严格隔离、纯物理 `DeviceBuffer`、`kernels.zig` 与 6 大错配模块重定位 | **✅ Done & Merged** |
| [`02_tensor_storage_and_zero_alloc_ops.md`](02_tensor_storage_and_zero_alloc_ops.md) | **M2** | 统一张量存储 `storage: DeviceBuffer`、目标写入 `*Into` 算子、强类型 `TensorError` | **⏳ In progress (in-place guards implemented; full validation pending)** |
| [`03_autodiff_forward_replay_and_allocator.md`](03_autodiff_forward_replay_and_allocator.md) | **M3** | 计算图前向重演零堆分配（`*Into` 直接写入）、反向传播引入 `scratch_allocator`、清除 `c_allocator` | **⏳ 待启动 (Queued)** |
| [`04_nn_module_protocol_and_layer_composability.md`](04_nn_module_protocol_and_layer_composability.md) | **M4** | PyTorch 风格 `nn.Module` 强类型协议、参数与缓冲区明确划分、反射迭代器、统一 `KVCache` | **⏳ 待启动 (Queued)** |
| [`05_composable_models_gpt_and_gemma4.md`](05_composable_models_gpt_and_gemma4.md) | **M5** | 消除 `GPT` 与 `Gemma 4` 过程式指针偏移与裸切片索引，统一声明式模块调用，剥离执行引擎至 `engine/` | **⏳ 待启动 (Queued)** |
| [`06_comprehensive_e2e_verification_and_hardening.md`](06_comprehensive_e2e_verification_and_hardening.md) | **M6** | 全量自动化测试矩阵、对抗性边界白盒审计、Schema 2.0 Web 可视化器对齐 | **⏳ 待启动 (Queued)** |
