# Master Blueprint: ZNN Architecture Audit and Refactoring

> **Status**: Completed through Milestone 10 (P0, M1, M2, M3, M4, M5, M7, M8, ResNet, M10 Completed & Merged to `main`)  
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

所有历史里程碑均已通过客观门禁并合入 `main`，后续任务按严格优先级组织推进：

| 优先级 | Milestone | 目标与范围 | 核心交付成果 | 验收状态 |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | **前置保障** | **Comptime 架构检测、构建链自适应探测与 3 大 GPU 缺陷修复** | 1. `build.zig` 库存在性探测，实现开箱即编译；<br>2. `src/backend/` comptime 目标架构检测、SIMD 宽度选择与无用代码修剪；<br>3. 修复 `write-after-free`、设备错配与转置连续性校验 3 大缺陷；<br>4. 清理 `chen-gz.github.io` 25 处本地绝对路径与死链。 | **✅ 已完成并合入 main (`b3237149`, `9ae21b03`)** |
| **P1** | **M1** | **硬件后端严格隔离与 HAL 建立** | 纯物理 `DeviceBuffer`、`kernels.zig`、消除 566 行 duck-typing `ops.zig`、84 文件 `check-layers` 严格单向。 | **✅ 已完成并合入 main (`8f4e87d1`, `df3835ec`)** |
| **P1** | **M2** | **张量物理存储统一与算子零分配体系** | 统一 `storage: backend.DeviceBuffer`，引入目标写入 `*Into` 算子，418 测试通过，18 模型 JSON 导出。 | **✅ 已完成并合入 main (`bc6981a4`, `c82f67a5`)** |
| **P1** | **M3** | **静态执行图、缓冲区复用与自动微分重放** | 消除 51 处 `copyFromEager`，根除 8 处 `c_allocator`，`MemoryPlanner` 显存复用 $\ge 85\%$，424 测试通过。 | **✅ 已完成并合入 main (`dc348622`, `dedc47a1`)** |
| **P1** | **M4** | **`nn.Module` 协议规范化与图层治理** | 编译期 `verifyModule(T)`，参数与持久缓冲区解耦，`ModuleList`，428 测试通过。 | **✅ 已完成并合入 main (`4b2f5d8f`)** |
| **P1** | **M5** | **核心模型全模块化重构 (`GPT` & `Gemma 4`)** | 消除裸指针与切片循环，声明式位置编码与多维张量流，91 文件分层无违规。 | **✅ 已完成并合入 main (`9a691413`, `42050e5a`)** |
| **P1** | **M7** | **ConvNeXt 现代纯卷积视觉架构** | `groups` 分组与 7x7 深度卷积、`LayerNorm2d`、`ConvNeXtBlock`、全套模型预设与基准。 | **✅ 已完成并合入 main (`6b6b9464`)** |
| **P1** | **M8** | **综合测试数据集扩展** | 3 通道 RGB `ImageDataset`、Tiny ImageNet 二进制加载、WikiText-103、多模态图文数据流。 | **✅ 已完成并合入 main (`5ee4f515`, `992418e0`)** |
| **P1** | **ResNet** | **ResNet 论文经典架构族** | `BasicBlock`, `Bottleneck`, ResNet-18~152 & CIFAR-20~110，训练 876+ img/s，测试 85.05%。 | **✅ 已完成并合入 main (`6a233edb`, `f70130db`, `d22f88a0`)** |
| **P1** | **M10** | **正交三角分解与高精度最小二乘求解** | 完整 Householder QR ($Q \in \mathbb{R}^{M \times M}$)、上三角回代 `solveTriangularUpper`、最小二乘 `qrSolve`。 | **✅ 已完成并合入 main (`39224590`)** |
| **P1** | **M11** | **高级张量索引、跨步视图与离散累加体系** | 通用 `gather`、`scatter`、`scatterAdd`、`indexSelect`、`maskedSelect`、负步长切片与 Autograd。 | **🔥 准备就绪 (Next in Queue)** |
| **P1** | **M12** | **多轴联合归约与进阶统计算子库** | 多轴联合归约 (`sumAxes` 等)、`prod`、`cumsum`、`norm`、`quantile` 与 `OpType.Max/Min` 求导。 | **⏳ 待启动 (Queued)** |
| **P2** | **M13** | **进阶数值线性代数库与求解器** | Cholesky 分解 ($LL^T$)、下三角前代 `solveTriangularLower`、`inv`、`pinv`、`det`/`slogdet`、`einsum`。 | **⏳ 待启动 (Queued)** |
| **P2** | **M14** | **现代科学计算与可微概率分布** | 独立 `Generator`、`znn.dist` 分布对象体系、重参数化采样 `rsample`、`log_prob`、特殊函数库。 | **⏳ 待启动 (Queued)** |
| **P3** | **M6** | **端到端对抗加固与压力验证** | 变长输入与极端上下文、内存压力 OOM 回退与安全审计。 | **🛡️ 持续测试门禁 (Active QA Gate)** |

---

## 4. 已完成里程碑交付纪要 (Completed Milestones Summary)

### Milestone 1: 硬件后端严格隔离与 HAL 建立
- **提交与范围**: 51 个文件修改，+3788 / -2696 行代码 (`8f4e87d1`, `df3835ec`)。
- **核心成果**: 引入统一物理显存容器 `DeviceBuffer`；消除 566 行 duck-typing `ops.zig`；建立跨包单向依赖检查器 `check_layers.zig`；将错配模块下沉至 Layer 0。

### Milestone 2: 张量物理存储统一与算子零分配体系
- **提交与范围**: 核心提交 `bc6981a4`，附加演进提交 `c82f67a5`。
- **核心成果**: 淘汰 `Tensor` 废弃字段（`device` 与 `device_array`），统一在 `storage: backend.DeviceBuffer` 下；新增 6 类零分配目标写入 `*Into` 算子；418 项单元测试通过，0 内存泄漏；18 个经典模型 JSON 导出通过；Gemma 4 E4B 4-bit 量化实测 127 token/s prefill 与 21.3 token/s decode。

### P0 前置修复: 编译期架构检测与构建工具链自适应
- **提交与范围**: 提交 `b3237149` 与 `9ae21b03`。
- **核心成果**: `build.zig` 库存在性物理探测 (`probeMlxAvailable`)；`src/backend/target.zig` 编译期目标架构与 SIMD 判定；修复 3 处 GPU 悬垂指针与转置校验缺陷；清理文档 25 处本地绝对路径与死链。

### Milestone 3: 静态执行图优化与缓冲区复用
- **提交与范围**: 核心提交 `dc348622`，基础提交 `dedc47a1`。
- **核心成果**: 消除 51 处 `copyFromEager` 直写输出；反向消除 8 处 `c_allocator`；实现 `MemoryPlanner` 与 `ExecutionGraph`，激活显存降低 $\ge 85\%$，424 项测试全通。

### Milestone 4: PyTorch 风格 `nn.Module` 协议与图层标准化
- **提交与范围**: 核心提交 `4b2f5d8f`。
- **核心成果**: 编译期 `verifyModule` 断言；`createParameter` 与 `createBuffer` 显式解耦；`BatchNorm` 运行统计量持久化缓冲且免疫 `setRequiresGrad`；`buffers`/`namedBuffers`/`submodules` 反射与 `ModuleList` 容器；428 项测试全通。

### Milestone 5: 核心模型全模块化重构 (`GPT` & `Gemma 4`)
- **提交与范围**: 核心提交 `9a691413`，状态同步提交 `42050e5a`。
- **核心成果**: `autodiff.Graph.arange` 动态无泄漏索引分配；`models.GPT` 位置编码与线性多维张量流重构；`decodeAttention` 全量接入 `graph.attention` 向量化 SGEMM，消除裸切片与指针循环；前端 Web 可视化 203 测试与 18 个模型架构 JSON 导出全通。

### Milestone 7: ConvNeXt 现代纯卷积视觉模型架构
- **提交与范围**: 核心提交 `6b6b9464`。
- **核心成果**: `ConvOptions.groups` 分组卷积与 7x7 深度可分离卷积；反向传播数值梯度检验（`gradcheck` 100% 对齐）；通道优先 `LayerNorm2d`；`ConvNeXtBlock`（反向瓶颈 + 7x7 DWConv + GELU + LayerScale）；全套模型预设；`examples/convnext.zig`（训练 186.97 img/s，推理 635.49 img/s，测试准确率 75.10%）。

### Milestone 8: 综合测试数据集扩展
- **提交与范围**: 核心提交 `5ee4f515` 与 `992418e0`。
- **核心成果**: 3 通道 RGB `ImageDataset`；Tiny ImageNet 紧凑二进制格式解析器与合成数据生成器；WikiText-103 词表与自回归因果数据集；`MultimodalDataset` 图文配对数据集与批次流水线；`download-data` CLI 扩展与交互帮助；`examples/multimodal.zig`（21+ pairs/s），0 泄漏。

### ResNet 论文经典架构全套实现
- **提交与范围**: 提交 `6a233edb`, `f70130db`, `d22f88a0`。
- **核心成果**: `models.resnet` 实现 ResNet-18/34 (`BasicBlock`)、ResNet-50/101/152 (`Bottleneck`)、CIFAR-20/32/44/56/110；残差直连与步长投影下采样；`examples/resnet.zig` 训练速度 876+ img/s，验证集准确率 85.05%。

### Milestone 10: 正交三角分解与高精度最小二乘求解
- **提交与范围**: 核心提交 `39224590`。
- **核心成果**: `QROptions` 与 `QRMode`（支持 `.reduced` 与 `.complete`），完整 Householder QR 分解 ($Q \in \mathbb{R}^{M \times M}$ 正交基)，上三角回代求解器 `solveTriangularUpper`，通用最小二乘求解器 `qrSolve`，线性回归高精度 QR 求解器 `solveLeastSquaresQR`（条件数保持 $O(\kappa)$）；全量 433 项测试通过，0 泄漏。
