# Milestone 10.5: 核心代码库架构整洁度与超长单体模块目录化重构 (Structural Optimization & Package Modularization)

> **Target Version**: `v0.4.2` (前置优化与结构加固)  
> **Status**: 🔥 **最高优先级待启动 (Top of Queue / Immediate Priority)**  
> **Priority**: **P0 (架构解耦与防劣化基石 - 优先于 Milestone 11 执行)**  
> **Target Packages**:  
> - `src/optim/` (Layer 4: 设立目录，重构 `src/optim.zig` 门面并拆分为专用子模块)  
> - `src/dataset/` (Layer 1: 设立目录，重构 `src/dataset.zig` 门面并拆分为多模态/视觉/文本子模块)  
> - `src/tensor/` (Layer 1: 拆分 130KB 超大单体测试 `src/tensor/tests.zig` 为领域测试文件)  
> - `examples/` & `scripts/` (根目录清洁度与 Python 辅助脚本归位)  
> - `doc/architecture.md` (全景架构文档与目录拓扑同步)  
> **Enforcement Policy**: Strict refactoring with zero backward-compatibility baggage (`AGENTS.md` §3)

---

## 1. 任务背景与核心动机 (Background & Technical Rationale)

在近期连续完成 Milestone 7 (ConvNeXt 视觉架构)、Milestone 8 (多模态与综合数据集) 以及 Milestone 10 (QR 分解与正交最小二乘) 之后，`znn` 实现了跨越式的能力升级，全量单元测试达到 433 项。

然而，伴随高强度的功能演进，代码库局部出现了明显的**单体文件膨胀与职责混杂**现象，若在启动庞大的 Milestone 11 (高级索引与离散求导) 和后续科学计算体系前不进行系统性重构，将严重制约后续模块的维护性与编译效率：

### 1.1 现状痛点诊断

1. **`src/optim.zig`（61.5 KB，1,600+ 行）单体严重超载**：
   - 该文件集成了基础优化器接口、SGD (含动量/Nesterov)、Adam/AdamW (含权重衰减解耦与 AMSGrad)、5 种学习率调度器（Linear、CosineAnnealing、StepLR 等）、二进制检查点持久化（`saveCheckpoint`/`loadCheckpoint`，含 `ZNNO` 魔数自检与权重版本映射），以及超过 500 行的内嵌测试用例；
   - 相比于 `nn`、`tensor`、`autodiff`、`models` 均具备独立的目录与清晰的子文件，`optim` 缺乏子目录结构，未来在引入自动混合精度 (AMP) 的 `GradScaler` 或新型优化器（如 Lion、Muon）时将彻底失控。

2. **`src/dataset.zig`（54.9 KB，1,400+ 行）异构数据流混杂**：
   - 随着近期多模态支持的合入，该文件同时混合了通用 `DataLoader` 协议、MNIST/CIFAR 二进制 IDX 动态解析器、Tiny ImageNet 紧凑二进制解析与合成器、WikiText-103 自回归文本数据流、以及图文配对 `MultimodalDataset`；
   - 缺乏领域划分，使得计算机视觉、自然语言与多模态数据加载代码彼此纠缠，修改任一领域数据格式都会引起全量文件重编译。

3. **`src/tensor/tests.zig`（130.8 KB，3,500+ 行）成为全库最大测试单体**：
   - 包含了从张量广播算术、多维切片、卷积池化、SVD 到 QR 分解的所有测试用例；
   - Milestone 11 即将引入大量的 `gather`、`scatter`、`scatterAdd`、`indexSelect` 及步长负切片测试，若继续堆叠在单体文件中，测试代码将突破 5,000 行，极难快速定位失败点。

4. **工程目录整洁度（Hygiene）**：
   - `examples/linear_regression.py` 孤立散落在纯 Zig 运行示例目录中，违背了 `examples/` 保持为纯 runnable Zig 目标的设计原则；
   - `doc/architecture.md` 目录树与最新代码库存在偏差，未反映 `convnext.zig`、`resnet.zig` 以及独立的 `changelog/` 架构。

---

## 2. 重构设计契约与目标结构 (Architectural Design & File Layout)

重构后，`src/optim` 与 `src/dataset` 将完全对齐 `src/nn` 和 `src/tensor` 的“**顶层公共门面 (Facade) + 内聚子模块目录**”标准范式：

```text
src/
├── optim.zig                      # [Layer 4 Facade] 仅重导出 optim/ 下的全部公共结构体与函数
├── optim/
│   ├── types.zig                  # 优化器基础接口、Config 与梯度步进公用类型
│   ├── sgd.zig                    # SGD、SGDConfig、Momentum 与 Nesterov 算法
│   ├── adam.zig                   # Adam、AdamW、AdamWConfig 与状态容器
│   ├── lr_scheduler.zig           # LRScheduler 协议与调度器族 (CosineAnnealing, StepLR, etc.)
│   ├── checkpoint.zig             # saveCheckpoint, loadCheckpoint 与 ZNNO 检查点序列化
│   └── tests.zig                  # 优化器专用单元测试集
│
├── dataset.zig                    # [Layer 1 Facade] 仅重导出 dataset/ 下的全部数据流接口
├── dataset/
│   ├── dataloader.zig             # 通用 DataLoader, Batch, Sampler, Collate 协议
│   ├── vision.zig                 # ImageDataset, IDX 二进制解析 (MNIST/CIFAR), TinyImageNet
│   ├── text.zig                   # TextDataset, WikiText-103 自回归因果文本流
│   ├── multimodal.zig             # MultimodalDataset, MultimodalPair 图文配对流
│   └── tests.zig                  # 数据加载专用单元测试集
│
├── tensor/
│   ├── tests_ops.zig              # 逐元素算术、广播、激活与矩阵乘法测试集
│   ├── tests_linalg.zig           # QR 分解、SVD 奇异值分解与对称特征值测试集
│   ├── tests_conv.zig             # 1D/2D 卷积、转置卷积与空间池化测试集
│   └── tests.zig                  # 汇总聚合测试，保留基础生命周期与边界回归测试
```

### 2.1 分层架构与 Facade 契约保证
- **零破坏性导入**：所有外部调用方（如 `src/root.zig`、`examples/*.zig`、`src/engine.zig`）通过 `@import("optim.zig")` 或 `@import("dataset.zig")` 的引用路径与公用 API **保持 100% 签名一致**，调用方无需任何代码改动；
- **分层依赖检查器 (`tools/check_layers.zig`) 天然兼容**：`check_layers.zig` 的 `packageOf` 逻辑按首段目录名判定，`optim` 保持为 Layer 4，`dataset` 保持为 Layer 1，分层规则不受任何影响。

---

## 3. 四阶段重构实施路线 (Phased Implementation Plan)

### Phase 1: `src/optim/` 模块化目录拆解
1. 创建目录 `src/optim/`；
2. 提取 `types.zig`、`sgd.zig`、`adam.zig`、`lr_scheduler.zig`、`checkpoint.zig` 与 `tests.zig`；
3. 将 `src/optim.zig` 精简为纯门面；
4. 运行 `zig build test` 确保所有优化器测试无回归通过。

### Phase 2: `src/dataset/` 模块化目录拆解
1. 创建目录 `src/dataset/`；
2. 提取 `dataloader.zig`、`vision.zig`、`text.zig`、`multimodal.zig` 与 `tests.zig`；
3. 将 `src/dataset.zig` 精简为纯门面；
4. 运行 `zig build test` 与 `zig build run-multimodal` 验证数据流吞吐与零泄漏。

### Phase 3: `src/tensor/tests.zig` 领域测试解耦与脚本归位
1. 依据数学领域将 `src/tensor/tests.zig` 分流解耦为 `tests_ops.zig`、`tests_linalg.zig` 与 `tests_conv.zig`；
2. 将 `examples/linear_regression.py` 迁移至 `scripts/benchmarks/linear_regression.py`，保持 `examples/` 纯粹性；
3. 预留 `src/tensor/indexing.zig` 与 `src/tensor/tests_indexing.zig` 桩位供 Milestone 11 开箱即用。

### Phase 4: 全局文档同步与全量回归验收
1. 更新 `doc/architecture.md`，同步反映最新的模块结构、包拓扑图与 `changelog/` 架构；
2. 运行 `zig build check-layers`（强制 0 违规）；
3. 运行 `zig build test`（433+ 测试全通，0 内存泄漏）；
4. 运行 `cd web && npm test` 确保可视化测试全通；
5. 按 `AGENTS.md` 规范使用 Jujutsu 创建独立分支并推送。

---

## 4. 客观验收标准与验证命令 (Acceptance Criteria)

- [ ] **优化器模块拆分完成**：`src/optim/` 包含 5 个职责分明的子模块与独立 `tests.zig`，`src/optim.zig` 体积降低至 `< 2KB` 纯门面；
- [ ] **数据集模块拆分完成**：`src/dataset/` 包含视觉、文本、多模态与独立 `tests.zig`，`src/dataset.zig` 体积降低至 `< 2KB` 纯门面；
- [ ] **测试文件减负**：`src/tensor/tests.zig` 成功按计算领域解耦，单个测试文件体积控制在合理范围；
- [ ] **目录整洁度**：`examples/linear_regression.py` 成功迁移至 `scripts/`，`examples/` 全部为纯 Zig 运行程序；
- [ ] **架构文档一致性**：`doc/architecture.md` 目录树与最新代码库 100% 对齐；
- [ ] **分层检查通过**：`zig build check-layers` 扫描所有 90+ 源文件，0 循环依赖、0 层级违规；
- [ ] **全量测试通过**：`zig build test` 所有测试 100% 成功，0 失败，GPA 内存泄漏检测为 0；
- [ ] **应用示例实跑**：`zig build run-multimodal` 与 `zig build run-convnext` 正常运行且输出指标吻合；
- [ ] **版本与分支管理**：在独立分支上完成代码提交并即时推送到 GitHub。
