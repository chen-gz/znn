# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **显式模块作用域 (`src/autodiff/graph.zig`, `src/autodiff/op.zig`, `src/tensor/core.zig`)**:
  - `Graph` 新增作用域栈与 `pushScope` / `popScope` / `currentScope`，以及 `Graph.enterModule` / `Graph.enterChildScope` 守卫 API；`Op.scope` 与 `Tensor.scope` 在创建时记录当前模块完整路径。
  - 所有内置模块 (`Linear`, `Conv2D`, `ConvTranspose2D`, `RMSNorm`, `LayerNorm`, `BatchNorm2d`, `Embedding`, `MLP`, `SwiGLU`, `CausalSelfAttention`, `TransformerBlock`, `TransformerDecoder`, `GPT`) 在 `forward` 入口进入自身作用域并自动注册模块类型。
  - `CausalSelfAttention` 将 $QK^T$、缩放、掩码、softmax 与加权求和封装在 `core` 子作用域 (`ScaledDotProductAttention`)。
- **作用域局部图导出 (`src/nn/visualization.zig`)**: 每个组合模块导出自身的 `ports` / `flow_nodes` / `edges` 局部图；Reshape / Transpose / RepeatKV 折叠进边的 `transforms`；残差边通过局部图可达性判定；无参数但执行算子的叶子 (如 `core`) 导出算子级局部图。
- 新增作用域归属测试与黄金边集测试 (`src/nn.zig`)。
- **可视化 JSON Schema (`src/nn/model_graph.schema.json`)**: 以 JSON Schema (draft 2020-12) 逐字段描述 schema 2.0 导出格式 (所有对象 `additionalProperties: false`，字段均带 `description`)，通过 `visualization.SCHEMA_JSON` 嵌入；新增一致性测试，用 GPT 与单个 `Linear` 的导出结果校验 schema，并验证未声明字段与错误版本会被拒绝。
- `examples/export_model_report.zig` 额外导出单个 `Linear` 层的最小参考 JSON `examples/minimal_model_graph.json` (可视化器 JSON 格式指南中的模板)。
- **图书全模型计算图导出 (`examples/export_book_models.zig`, `build.zig`)**: 新增 `zig build run-book-models` 步骤，导出专著各章节涉及的全部 18 个经典模型（`linear`、`mlp`、`rnn`、`lstm`、`stacked_lstm`、`gru`、`embedding`、`attention`、`transformer_block`、`gpt`、`swiglu`、`lora_linear`、`layernorm`、`mla`、`deepseek_moe`、`gan_generator`、`gan_discriminator`、`conv2d`）的 schema 2.0 计算图 JSON。

- **Autograd 算子扩展与 LLM 后训练 Loss 求导 (`src/autodiff/`, `src/tensor/core.zig`, `src/nn/`)**:
  - 新增 `LayerNorm`、`BatchNorm2d`、`Dropout`、`AvgPool2D`、`RoPE`、`MaskedCrossEntropyLoss`、`DpoLoss`、`GrpoLoss`、`Sqrt`、`Exp`、`Log`、`Abs`、`Sum`、`Mean`、`Variance`、`Where`、`MaskedFill`、`Squeeze`、`Unsqueeze`、`Slice` 等算子的计算图前向/反向传播与可视化数学公式。
  - `softmaxCrossEntropy` 与 `maskedCrossEntropyLoss` 支持 `anytype` 整型分类标签切片（`u8`、`u32`、`usize` 等），突破 256 类词表限制。
  - `Conv2D` 与 `Tensor.conv2dWithConfig` / `Graph.conv2dWithConfig` 新增可配置 `stride` 与 `padding` 支持，并采用 `im2col` / `col2im` + `cblas_sgemm` 加速前向与反向传播。
- **张量双轨互操作 (`src/tensor/types.zig`, `src/tensor/core.zig`, `src/nn/core.zig`)**:
  - `GenericTensor(T)` 新增 `fromSlice`、`zeros`、`ones`、`full`、`reshape`、`transpose`、`add`/`sub`/`mul`、`sum`/`mean`、`eq`/`ne`/`gt`/`lt`、`any`/`all` 及与 `Tensor` 双向转换接口。
  - `Embedding.forward` 支持直接传入 `GenericTensor(u32)` / `GenericTensor(usize)` / `GenericTensor(i32)` 或整型切片；`Tensor.where` 与 `Tensor.maskedFill` 支持直接接收 `BoolTensor`。
- **全模块命名与可视化作用域覆盖 (`src/nn/recurrent.zig`, `src/nn/transformer.zig`)**:
  - 为 `RNNCell`、`RNN`、`LSTMCell`、`LSTM`、`StackedLSTM`、`GRUCell`、`GRU`、`MoELayer`、`MLALayer`、`LoRALinear` 补齐 `setName` / `setNameFormatted` / `getName` / `formula` / `registerFormula` 与 `Graph.enterModule` 作用域追踪。

### Changed
- **节点分类强类型枚举 (`src/nn/visualization.zig`, `src/nn.zig`)**: 将 `NodeData.kind` 从弱类型字符串切片 (`[]const u8`) 重构为强类型枚举 `NodeKind` (`.Param`, `.Input`, `.Buffer`, `.Activation`)，消除 `std.mem.eql` 字符串比较并利用 `switch` 提供编译期完备性检查；序列化与反序列化通过 `asString()` 与 `fromString()` 保持 JSON Schema 2.0 规格完全一致。
- **其余取值字段同样改为枚举 (`src/nn/visualization.zig`, `src/nn.zig`)**: `NodeData.status` → `NodeStatus` (`CUSTOM_INIT` / `AUTO_GRAPH` / `INPUT` / `BUFFER` / `OP_OUTPUT`)，`FlowNode.kind` → `FlowNodeKind` (`port_in` / `port_out` / `module` / `op` / `buffer`)，`EdgeData.kind` → `EdgeKind` (`data` / `buffer`)；`summary` 的初始化计数改为穷举 `switch`；删除恒为 `"module"` 的 `ModuleNode.kind` 字段 (序列化仍固定输出 `"kind": "module"`)。新增测试断言这些枚举的标签与 `model_graph.schema.json` 中对应的 `enum` 列表逐一一致；导出的 JSON 逐字节不变。
- **可视化 JSON 升级为 schema 2.0 (不兼容 1.0)**: 移除顶层 `edges`，新增 `default_scope`；端口以 `@in<k>` / `@out<k>` 命名并携带 `ref`；`summary` 新增 `buffer_nodes`；图输入归属 `root.nodes`，不再生成 `inputs` / `outputs` 伪模块。
- **Schema 字段说明补全 (`src/nn/model_graph.schema.json`)**: `FlowNode.id` 写明本地 id 的来源、自动命名与重名后缀规则，`Edge.is_skip` 写明残差判定条件，`PortEntry.id` / `ref` 与 `ModuleNode` 写明端口编号、外部端点解析与局部图导出范围；模型图导出的设计摘要移至 `doc/model-graph-visualization.md`。
- **文档整理 (`README.md`, `doc/`, `plan/`)**: README 精简为概览、文档索引、快速上手与示例命令表；README 中的路线图条目并入 `plan/TODO.md`，NumPy 对比与已完成里程碑分别由 `plan/NUMPY_GAP_ANALYSIS.md` 与本文件承载；`doc/model-graph-visualization.md` 按当前 schema 重写并删除 `plan/VISUALIZATION_GRAPH_EDGE_DESIGN.md`；`plan/` 中的绝对路径链接改为相对链接。
- `TransformerBlock` 第二个残差加法节点由 `output` 更名为 `residual_mlp`；MLP 激活输出命名为 `gelu`。
- `GPT` 的位置索引张量改为 `{gpt}.pos_indices` 静态缓冲区 (`is_buffer = true`)，不再作为模型输入出现。
- `TransformerBlock.formula` 改用 `aligned` 环境分两行排版 (注意力残差与 MLP 残差各占一行)，不再以 `\quad` 拼接在同一行。
- **算子前向去重与构建/测试模块化 (`src/autodiff/op.zig`, `src/tests.zig`, `src/nn/tests.zig`, `build.zig`)**:
  - `Op.forward` 复用 `Tensor` 的 Eager 算子实现，消除 `Tensor`、`Graph` 与 `Op.forward` 三处重复的前向计算逻辑。
  - 将 `src/root.zig` 与 `src/nn.zig` 中的内联测试拆分为独立测试文件 `src/tests.zig` 与 `src/nn/tests.zig`；`build.zig` 中 16 个示例构建与运行步骤统一收敛为表驱动循环。
- **算子性能优化 (`src/nn/transformer.zig`, `src/cblas.zig`)**:
  - `MoELayer.forward` 实现 Top-K 稀疏门控掩码与活跃专家筛选，跳过未命中专家的前向计算。
  - `CausalSelfAttention.forward` 将因果掩码从每次分配两份 `[B, nh, T, T]` 优化为单份 `[1, 1, T, T]` 广播张量。
  - `cblas_sgemm_fallback` 的 `NoTrans × Trans` 分支新增 8 路 `@Vector(8, f32)` SIMD 向量化内积快路径。
- **Tensor ← Graph ← nn 单向解耦 (`src/tensor/`, `src/autodiff/`, `src/nn/`, `src/engine.zig`, `examples/`)**:
  - `Tensor` 方法与 `tensor.concat` / `tensor.split` 不再接收 `graph: ?*autodiff.Graph` 参数，只保留纯计算内核；`Tensor` 对 autodiff 仅保留数据字段 `creator: ?*Op`。新增 `Tensor.squeezedShape` / `unsqueezedShape` 供 `Graph.squeeze` / `unsqueeze` 复用；删除 `tensorSplit` 别名。
  - `Graph` 新增 `initNoGrad` (推理用无梯度图) 与 `arenaAllocator`；`enterModule` / `enterChildScope` 改为 `Graph` 方法。
  - 所有 nn 模块统一为 `forward(self, graph: *autodiff.Graph, x, ...)`，移除 Eager/Graph 双分支、`graph == null` 时的手动释放与 `free_*` 标记；`RNN` / `LSTM` / `StackedLSTM` / `GRU` 的输出切片分配在图 arena 中。
  - `CausalSelfAttention.forwardInference` 与 `MLALayer.forwardInference` 内部使用局部无梯度图；`engine.evalClassificationStep` 改用无梯度图计算损失与准确率。
  - `src/tensor/ops.zig` 中依赖 `Graph` 的测试迁移至 `src/autodiff/tests.zig`；`examples/sample_model_graph.json` 重新生成 (因果掩码为 `[1, 1, 16, 16]`)。
  - `src/autodiff.zig` 只导出自身 API (`Graph`、`Op`、`OpType`、`OpContext`)；不再转出 `tensor`、`Tensor`、`Shape`、`computeContiguousStrides`、`transposeShape`，`types` / `op` / `graph` 子模块改为私有。张量类型统一从 `tensor` 模块引用。

### Fixed
- 修复算子依据输入推断归属导致的模块错配、`.core` 伪节点合成、端口名与真实节点冲突、残差判定依赖边顺序、根作用域边与顶层 `edges` 层级错位等问题 (详见 chen-gz.github.io `doc/visualization-model-edge-design.md`)。
- **非连续视图内存安全 (`src/tensor/shape.zig`, `src/autodiff/op.zig`, `src/autodiff/graph.zig`)**: 修复 `broadcastBinaryOpRaw`、`Op.backward` 与 `Graph.reshape` 在处理非连续步长视图 (`!isContiguous()`) 或带 `offset` 子视图时的越界与错读问题。
- **训练路径与反射/序列化完整性 (`src/nn/transformer.zig`, `src/nn/core.zig`, `src/nn/serialization.zig`)**:
  - 修复 `MLALayer.forward` 未使用 `q_all`、`w_kr` 及因果注意力的占位实现，补全完整潜在多头注意力训练与求导路径。
  - 修复 `collectParameters`、`deinitModel`、`zeroGradModel` 及 Safetensors 序列化对动态切片字段 (`[]T`) 与可选张量 (`?*Tensor`) 的遗漏，支持无序偏移量的 Safetensors 文件加载，并默认冻结 `LoRALinear` 基础权重梯度。
- **可视化残差可达性判定 BFS 优化 (`src/nn/visualization.zig`)**: 将 `markSkips` 中递归 DFS 的 `reaches` 重构为带已访问哈希表 (`std.StringHashMap(void)`) 的线性 BFS，彻底解决多分支复合模块（如 LSTM、StackedLSTM、MoE）可达性检查指数爆炸卡死的问题。

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
