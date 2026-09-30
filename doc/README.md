# ZNN 文档中心 (Documentation Hub)

欢迎查阅 `znn` (Zig Neural Network) 核心架构与工程规范文档中心。本目录（`doc/`）汇总了 `znn` 的全景系统架构设计、张量与自动微分底座、神经网络算子模型、训练与优化基础设施、计算图可视化协议以及工程演进规范。

---

## 🎯 项目核心使命与设计哲学

`znn` 是一个基于 **Zig 0.16.0** 从零构建的高性能、模块化深度学习与现代大语言模型（LLM）微框架。

项目的核心工程原则：
1. **纯 Zig 实现与零依赖 (100% Pure Zig & Zero External Dependencies)**：不依赖 Python 运行时、PyTorch C-API 或任何外部软件包，单二进制独立静态编译执行。
2. **确定性内存管理 (Deterministic Memory Management)**：基于 `ArenaAllocator` 的批次生命周期垃圾回收，消除训练循环中的内存碎片与动态泄漏；模型权重持久化基于显式分配器掌控。
3. **软硬件协同加速 (Hardware-Aware Acceleration)**：深度融合 macOS Accelerate 框架，无缝调度 Apple Silicon AMX 协处理器；提供基于 Zig 原生 `@Vector(8, f32)` SIMD 向量化跨平台指令级后备内核。
4. **端到端前沿模型覆盖 (Modern Deep Learning Continuum)**：从基础统计回归（OLS、Ridge、Lasso、ElasticNet）到前沿生成式架构（SwiGLU、MLA、MoE、LoRA、BPE 分词器、DPO 对齐损失）。
5. **现代计算图可视化协议 (Schema 2.0 Visualization)**：原生支持模块作用域拓扑图导出，与前端 Web 可视化工具无缝契合。

---

## 📚 文档目录索引 (Documentation Index)

| 模块 | 文档名称 | 核心主题与内容概要 |
| :--- | :--- | :--- |
| **01. 系统全景架构** | [系统全景架构设计 (architecture.md)](architecture.md) | 系统分层模型、张量引擎与跨步内存、反向传播与动态计算图、参数反射、前沿 Transformer 与循环网络、解耦优化器与学习率调度器、Safetensors 与二进制 Checkpoint、内存生命周期与软硬件加速。 |
| **02. 模型图导出** | [模型图导出 (model-graph-visualization.md)](model-graph-visualization.md) | 模块作用域（`enterModule` / `enterChildScope`）、作用域局部图与端口（`@in<k>` / `@out<k>`）、透明算子折叠、JSON Schema 与枚举字段、修改导出格式的步骤。 |
| **03. 技术差距分析** | [现代 NumPy 2.x 差距诊断 (NUMPY_GAP_ANALYSIS.md)](../plan/NUMPY_GAP_ANALYSIS.md) | 9 大能力维度深度对比表（泛型 Dtype、零拷贝切片、多轴归约、线性代数等）、设计权衡与演进阶段规划。 |
| **04. 研发任务清单** | [架构诊断与分阶段任务跟踪 (TODO.md)](../plan/TODO.md) | 6 大核心维度的成熟度诊断与演进状态跟踪、P0 至 P3 分阶段攻坚任务明细。 |
| **05. 版本更新历史** | [版本发布与变更日志 (CHANGELOG.md)](../CHANGELOG.md) | 语义化版本记录、各版本特性增加（`Added`）、架构变更（`Changed`）与修复（`Fixed`）。 |

---

## 🛠️ 构建与运行

构建、测试、覆盖率、基准测试与示例命令见仓库根目录的 [`README.md`](../README.md)。
