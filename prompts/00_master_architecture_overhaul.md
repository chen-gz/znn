# Master Blueprint: ZNN Architecture Audit and Refactoring

> **Status**: In Progress (Milestone 1 completed and merged to `main`)  
> **Source Prompt**: `prompt_draft.md` (Teamwork Preview Master)  
> **Target Directory**: `/Users/guangzong/Documents/znn`  
> **Enforcement Policy**: Strict refactoring with zero backward-compatibility baggage (`AGENTS.md` §3)

---

## 1. 核心任务定义 (Mission Statement)

全面审查并重构 ZNN (Zig Neural Network) 架构体系：
1. **严格后端隔离**：消除底层硬件代码中的鸭子类型（`anytype`），将计算后端（CPU/CBLAS、NVIDIA CUDA、Apple Metal/MLX）严格隔离在 Layer 0（`src/backend/`），仅向高层暴露强类型物理内存与无状态纯 C-ABI 算子；
2. **算子清晰化与零分配**：厘清张量计算、自动求导与底层内核的职责分工，引入成套目标写入 (`*Into`) 算子，彻底消除前向重演中的内存拷贝与抖动；
3. **PyTorch 风格的模块化体系 (`nn.Module`)**：全面重构神经网络层与核心模型（`GPT`、`Gemma 4`），消除过程式裸指针偏移、硬编码切片索引与非标推理分支，建立标准参数/缓冲区反射与声明式前向图。

---

## 2. 8 层严格单向依赖架构 (Strict 8-Layer Hierarchy)

由 [`tools/check_layers.zig`](../tools/check_layers.zig) 静态保证跨包导入只能单向指向低层级：

```text
Layer 0: backend (HAL, 物理内存 DeviceBuffer, 无状态 kernels, 驱动)
   ▲
Layer 1: tensor, dataset (单一存储张量 Tensor, 连续化, 广播, 目标写入算子, safetensors)
   ▲
Layer 2: autodiff, regression, manifold (Graph, 自动求导 Op, 前向重演, backward, autodiff/ops)
   ▲
Layer 3: nn, cross_validation (nn.Module 协议, 常用层, GQAAttention, KVCache, graph_ir, visualization)
   ▲
Layer 4: models, optim (纯模型定义 GPT / Gemma4, 参数优化器 AdamW / SGD)
   ▲
Layer 5: engine (训练循环 trainCausalLMStep, 采样与对数几率处理 sampling / logits_processor)
   ▲
Layer 6: bench (基准测试套件)
   ▲
Layer 7: root (顶层聚合门面与 tests.zig)
```

---

## 3. 分阶段里程碑全景表 (Phased Milestones)

| Milestone | 目标与范围 | 核心交付成果 | 验收状态 |
| :--- | :--- | :--- | :--- |
| **M1** | 后端严格隔离与 HAL 建立 | 纯物理 `DeviceBuffer`、`kernels.zig`、消除 566 行 duck-typing `ops.zig`、迁移 6 大错配模块、层级检查器静态禁令 | **✅ 已完成 (51 文件, +3788/-2696 行)** |
| **M2** | 张量物理存储统一与算子零分配 | `Tensor.storage: DeviceBuffer` 单一真实来源、引入成套 `*Into` 算子、显式强类型 `TensorError` | **⏳ 下一步 (待启动)** |
| **M3** | 自动微分前向重演与分配器受控 | `op.forward` 废除 `copyFromEager` 直写输出、`backward` 显式传入 `scratch_allocator`、根除 `c_allocator` | **⏳ 待启动** |
| **M4** | `nn.Module` 协议规范化与图层治理 | 编译期 `verifyModule(T)`、`createParameter`/`createBuffer` 划分、`Q4Linear` 接入 `walk`、统一 `KVCache` | **⏳ 待启动** |
| **M5** | 核心模型全模块化重构 | 消除 `GPT` 与 `Gemma 4` 中的切片索引与过程式偏移、模型脱离特定硬件引擎、统一 `forwardWithCache` | **⏳ 待启动** |
| **M6** | 端到端全量验证与对抗加固 | 全自动化测试矩阵（`check-layers`, `test`, `run-book-models`）、白盒压力测试、Web 可视化器 Schema 2.0 对齐 | **⏳ 待启动** |
