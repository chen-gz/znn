# Master Blueprint: ZNN Architecture Audit and Refactoring

> **Status**: In Progress (Milestone 1 & Milestone 2 Completed & Merged to `main`)  
> **Source Prompt**: `prompt_draft.md` (Teamwork Preview Master)  
> **Target Directory**: `/Users/guangzong/Documents/znn`  
> **Enforcement Policy**: Strict refactoring with zero backward-compatibility baggage (`AGENTS.md` §3)

---

## 1. 核心任务定义与架构再评估 (Mission Statement & Re-Evaluation)

全面审查并重构 ZNN (Zig Neural Network) 架构体系：
1. **编译期硬件与工具链自适应探测 (Comptime Architecture Detection & Toolchain Probing)**：
   - 彻底摆脱脆弱的运行时配置与外部手工编译参数（如 `-Dmlx=false`）；
   - 在构建层（`build.zig`）增加底层动态库/SDK 物理探测，按真实存在的库安全默认配置，杜绝链接中断；
   - 在内核层（`src/backend/`）全面基于 Zig 原生 `comptime` 判定目标架构、操作系统与 CPU 特性（NEON、AVX2、AVX-512、AMX、FMA），自动计算最优 SIMD 向量宽度，并在编译期直接决定硬件加速链（Apple Accelerate、Metal MLX、NVIDIA CUDA Driver API 或纯 Zig 便携内核），无用后端代码编译期彻底剔除，实现零开销分发；
2. **算子清晰化与静态执行图零堆分配 (Zero-Allocation Static Execution Graph)**：
   - 厘清张量计算、自动求导与底层内核的职责分工，引入成套目标写入 (`*Into`) 算子，彻底消除前向重演中的 51 处 `copyFromEager` 内存拷贝与抖动；
   - 引入静态内存生命周期分析与缓冲区复用引擎（`src/autodiff/memory_planner.zig`），在推理时降低激活显存达 90%+，在反向时根除 `backward_nn.zig` 中全部 8 处 `std.heap.c_allocator`；
3. **张量数据与梯度一等公民重构 (`Tensor.grad: ?*Tensor = null`)**：
   - 将梯度从脆弱的 CPU 裸切片升级为完整的 `Tensor` 实例，原生支持 GPU 显存驻留与懒分配；
   - 消除裸切片直接访问引发的 `write-after-free` 悬垂指针缺陷，引入受控宿主访问器（`asHostSlice` / `asHostSliceMut`）与内联提升规范（Prologue Hoisting）；
4. **PyTorch 风格的模块化体系 (`nn.Module`) 与核心模型解耦**：
   - 全面重构神经网络层与核心模型（`GPT`、`Gemma 4`），消除过程式裸指针偏移、硬编码切片索引与非标推理分支，建立标准参数/缓冲区反射与声明式前向图，将训练与推理引擎彻底沉淀至 `engine/`。

---

## 2. 8 层严格单向依赖架构 (Strict 8-Layer Hierarchy)

由 [`tools/check_layers.zig`](../tools/check_layers.zig) 静态保证跨包导入只能单向指向低层级：

```text
Layer 0: backend (HAL, 物理内存 DeviceBuffer, 无状态 kernels, comptime 硬件探测与驱动)
   ▲
Layer 1: tensor, dataset (单一存储张量 Tensor, 连续化, 广播, 目标写入算子, safetensors)
   ▲
Layer 2: autodiff, regression, manifold (Graph, 自动求导 Op, 前向重演, memory_planner, backward)
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

## 3. 分阶段实施路线与优先级全景表 (Phased Priority & Roadmap)

根据架构深度再评估，所有后续任务按严格优先级组织，**P0 优先级的开箱即用与前置阻断修复位于最前**：

| 优先级 | Milestone | 目标与范围 | 核心交付成果 | 验收状态 |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | **前置保障** | **Comptime 架构检测、构建链自适应探测与 3 大 GPU 缺陷修复** | 1. `build.zig` 库存在性探测，实现开箱即编译；<br>2. `src/backend/` comptime 目标架构检测、SIMD 宽度选择与无用代码修剪；<br>3. 修复 `write-after-free`、设备错配与转置连续性校验 3 大缺陷；<br>4. 清理 `chen-gz.github.io` 18 处本地绝对路径与死链。 | **🔥 准备就绪 (Ready for Execution)** |
| **P1** | **M3** | **静态执行图、缓冲区复用与自动微分重放** | 1. 彻底根除 51 处 `copyFromEager` 直写输出；<br>2. 彻底根除 8 处 `std.heap.c_allocator`；<br>3. 静态内存生命周期分析与缓冲区复用（`memory_planner.zig`，激活显存降低 90%+）；<br>4. `Tensor.grad` 一等公民重构（GPU 驻留、懒分配、Prologue 提升累加）。 | **⏳ 待启动 (Queued)** |
| **P2** | **M3.x** | **平台中立调度与显存切片抽象** | 1. 解耦 `GraphOptions` 中厂商特定的 `mlx_stream`；<br>2. `DeviceBuffer` 支持零拷贝子显存切片（Sub-Buffer Slicing）。 | **⏳ 待启动 (Queued)** |
| **P3** | **M4** | **`nn.Module` 协议规范化与图层治理** | 1. 编译期 `verifyModule(T)` 强类型断言；<br>2. 严格区分可训练参数与持久缓冲区（`createParameter`/`createBuffer`）；<br>3. `Q4Linear` 接入 `walk` 反射，统一标准 `KVCache`。 | **⏳ 待启动 (Queued)** |
| **P4** | **M5** | **核心模型全模块化重构 (`GPT` & `Gemma 4`)** | 1. 消除模型层过程式裸指针偏移与裸切片索引；<br>2. 统一纯粹声明式模块调用，推理/训练执行引擎彻底剥离至 `engine/`。 | **⏳ 待启动 (Queued)** |
| **P5** | **M6** | **端到端全量验证与对抗加固** | 1. 全自动化矩阵测试（全平台、全优化级别 0 内存泄漏）；<br>2. 对抗性边界测试与 Web 可视化器 Schema 2.0 契约对齐。 | **⏳ 待启动 (Queued)** |

---

## 4. 已完成里程碑交付纪要 (Completed Milestones Summary)

### Milestone 1: 硬件后端严格隔离与 HAL 建立
- **提交与范围**: 51 个文件修改，+3788 / -2696 行代码。
- **核心成果**: 引入统一物理显存容器 `DeviceBuffer`；消除 566 行 duck-typing `ops.zig`；建立跨包单向依赖检查器 `check_layers.zig`；将错配模块下沉至 Layer 0。

### Milestone 2: 张量物理存储统一与算子零分配体系
- **提交与范围**: 核心提交 `bc6981a4`，附加演进提交 `c82f67a5`。
- **核心成果**:
  1. 彻底淘汰 `Tensor` 废弃字段（`device` 与 `device_array`），统一在 `storage: backend.DeviceBuffer` 下；
  2. 新增 6 类零分配目标写入 `*Into` 算子；
  3. `nn/module.zig` 持久化张量（`createPersistentTensor` / `createCacheTensor`）全面改由 `DeviceBuffer.alloc()` 分配，彻底根除双重释放；
  4. 验证指标：全量 418 项单元测试通过，0 内存泄漏；18 个经典模型 JSON 导出通过；Gemma 4 E4B 4-bit 量化实测 127 token/s prefill 与 21.3 token/s decode。
