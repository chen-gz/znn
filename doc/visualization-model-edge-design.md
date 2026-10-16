# ZNN 模型图可视化设计规范
# (ZNN Model Graph Visualization Specification)

| 项 | 内容 |
| :--- | :--- |
| 范围 | `znn` 导出模型图 JSON，`web/` 可视化前端渲染 |
| 格式版本 | schema `2.0` |
| 格式定义 | `src/nn/model_graph.schema.json`（JSON Schema draft 2020-12），见 §4 |
| 样例 | `examples/sample_model_graph.json`（2 层 GPT）、`examples/minimal_model_graph.json`（单个 `Linear`），均由 `zig build run-report` 导出 |

---

## 1. 概览 (Overview)

```text
Module.forward ──(作用域栈)──▶ autodiff.Graph ──graph_ir.build──▶ ModelHierarchyGraph ──serializeJson──▶ JSON
                                                                                                         │
                          web/ ◀──────(按 schema 校验)────── model_graph.schema.json ◀────────────────────┘
```

* 每个模块在 `forward` 中进入自己的作用域，算子与张量在创建时记录所在作用域（§2）。
* 导出端为模块树中的每个模块生成一张**局部图**：节点是该模块的直接子模块、自身执行的算子、常量缓冲区和边界端口，边是它们之间的数据依赖（§3）。
* JSON 的每个字段由 schema 定义（§4）；渲染端按局部图逐层展示，任何模块都可以单独查看（§5）。

| 位置 | 文件 | 职责 |
| :--- | :--- | :--- |
| `znn` | `src/autodiff/graph.zig` | 作用域栈、`Op.scope` / `Tensor.scope` |
| `znn` | `src/nn/visualization.zig` & `src/nn/graph_ir.zig` | 局部图构建、枚举类型、模块树、JSON 序列化与导出入口 (`nn.generateJson` / `nn.exportJson`) |
| `znn` | `src/nn/model_graph.schema.json` | JSON Schema |
| `znn/web` | `web/src/utils/model-graph.ts` | 数据层：局部 DAG、路径解析、残差识别、分层布局 |
| `znn/web` | `web/src/utils/flow-edges.ts` | 连线路由（不依赖 DOM） |
| `znn/web` | `web/src/utils/json-schema.ts` | schema 校验与格式指南的表格生成 |
| `znn/web` | `web/src/scripts/visualizer.ts` | 页面渲染与交互 |
| `znn` | `examples/` | 样例（`sample_model_graph.json`、`minimal_model_graph.json` 与 `examples/models/*.json`） |

---

## 2. 模块作用域 (Module Scopes)

`autodiff.Graph` 维护一个作用域栈，模块归属由**执行位置**决定（训练图 `Graph.init(allocator)` 与无梯度推理图 `Graph.initNoGrad(allocator)` 均一致记录作用域）：

```zig
// 每个模块 forward 的第一行（模块未命名时返回空守卫）
const module_scope = try enterModuleScope(graph, self); // push + 注册模块类型与数学公式
defer module_scope.exit();                              // pop
```

* API：`pushScope(path, ?module_type)` / `popScope()` / `currentScope()`；`graph.enterModule` 与 `graph.enterChildScope` 返回 `ScopeGuard`，`enterModule` 同时注册模块类型，`nn.enterModuleScope` 额外自动注册模块声明的 `pub const formula`。
* 接入作用域的内置 `module_type`：`root`、`container`、`ModuleList`、`GPT`、`TransformerDecoder`、`TransformerBlock`、`CausalSelfAttention`、`ScaledDotProductAttention`、`MLALayer`、`MoELayer`、`MLP`、`SwiGLU`、`Linear`、`LoRALinear`、`Embedding`、`RMSNorm`、`LayerNorm`、`BatchNorm1d`、`BatchNorm2d`、`GroupNorm`、`Conv1D`、`Conv2D`、`ConvTranspose1D`、`ConvTranspose2D`、`MaxPool1D`、`AvgPool1D`、`MaxPool2D`、`AvgPool2D`、`AdaptiveAvgPool1D`、`AdaptiveAvgPool2D`、`Dropout`、`RNN`、`RNNCell`、`LSTM`、`LSTMCell`、`StackedLSTM`、`GRU`、`GRUCell`。
* 每个 `Op` 在 `Graph.recordOp` 时记录 `op.scope`（栈顶模块的完整路径），每个张量在创建时记录 `Tensor.scope`。
* 归属规则：
  * 算子：`op.scope`；若某个参数输入的名字前缀是 `op.scope` 更深的后代（未接入作用域的自定义模块），取该前缀。
  * 张量：`creator` 算子的作用域 → 参数名字前缀 → `Tensor.scope`。缓冲区取创建时的作用域，图输入属于 root（`""`）。
* 张量名只是显示标签，不参与归属。例如 `logits.setName("outputs.logits")` 不改变产生它的 Reshape 所在的作用域。
* 注意力核心：`CausalSelfAttention.forward` 用 `try graph.enterChildScope("core", "ScaledDotProductAttention")` 开启子作用域，覆盖 `K^T → QK^T → scale → +mask → softmax → ·V`，得到加权结果后关闭；head split / merge 属于 `attn` 本身。

---

## 3. 局部图模型 (Scoped Local Graph)

### 3.1 节点

对任意模块 $M$（包括 root），局部图 $G(M) = (V_M, E_M)$ 的节点由 `flow_nodes` 声明，类型由 `kind` 给出：

| `kind` | 含义 | 本地 id | 例子 |
| :--- | :--- | :--- | :--- |
| `port_in` / `port_out` | 进入 / 离开 $M$ 的每一个不同的边界张量 | `@in0`、`@in1`、`@out0` … | `decoder.h.0` 的 `@in0` |
| `module` | $M$ 的直接子模块 | 子模块名 | `ln_1`、`attn` |
| `op` | 在 $M$ 作用域内直接执行的非透明算子 | 输出张量名的最后一段；未命名时为自动名 `act_<OpType>_<idx>` | `residual_attn`（Add）、`gelu`、`embeddings_sum`、`act_Softmax_21` |
| `buffer` | 在 $M$ 作用域内创建的常量 | 张量名的最后一段 | `pos_indices`、`causal_mask` |

* `@` 是端口的保留前缀，端口 id 不会与子模块或算子重名。端口按在遍历中首次出现的顺序编号。
* 其他 id 若在同一模块内重复，依次加后缀 `_2`、`_3` 等。
* `ref` 是节点的全局引用：`module` 为子模块路径，`op` / `buffer` 为张量名，端口见 §3.3。
* `op` 节点带 `op_type`，`module` 节点带 `module_type`；端口、算子与缓冲区带张量 `shape`，模块的 `shape` 为 `""`。

```json
"flow_nodes": [
  { "id": "@in0", "kind": "port_in", "ref": "gpt.embeddings_sum", "shape": "[2, 16, 64]" },
  { "id": "attn", "kind": "module", "ref": "gpt.decoder.h.0.attn", "module_type": "CausalSelfAttention", "shape": "" },
  { "id": "residual_attn", "kind": "op", "ref": "gpt.decoder.h.0.residual_attn", "op_type": "Add", "shape": "[2, 16, 64]" }
]
```

### 3.2 边与透明算子

`edges` 中的 `from` / `to` 都是同一模块 `flow_nodes` 中的本地 id。**透明算子**（`Reshape`、`Transpose`、`RepeatKV`，只改变布局、不做计算）不作为节点，而是按执行顺序折叠进经过它的边：

```json
{
  "from": "@in0", "to": "q_attn",
  "shape": "[2, 16, 64]",
  "dst_shape": "[32, 64]",
  "transforms": ["Reshape [2, 16, 64] -> [32, 64]"],
  "is_skip": false,
  "kind": "data"
}
```

* `shape` 为生产端的张量形状；`dst_shape` 为经过透明算子后到达消费端的形状，与 `shape` 相同时省略；`transforms` 为空时省略。
* `kind`：`data` 为激活数据流，`buffer` 为从常量缓冲区出发的边。
* 同一对 `(from, to)` 只保留一条边，`shape` 始终取生产端张量，与遍历顺序无关。

### 3.3 端口

每个模块导出 `ports`，内容与 `flow_nodes` 中的 `port_in` / `port_out` 一一对应，用于把端口解析回全局路径：

```json
"ports": {
  "inputs":  [{ "id": "@in0",  "ref": "gpt.embeddings_sum", "shape": "[2, 16, 64]" }],
  "outputs": [{ "id": "@out0", "ref": "gpt.decoder.h.1",    "shape": "[2, 16, 64]" }]
}
```

* 端口以边界张量为键，同一个张量只生成一个端口；同一个张量扇出到多个子节点时是一个端口加多条边。
* `ref` 是 $M$ 外部那一端在其与 $M$ 的**最近公共祖先**作用域中可见的节点：输入端口为生产者，输出端口为第一个消费者。图的最终输出没有消费者，`ref` 为该张量的名字（例如 root 的 `@out0` 为 `outputs.logits`）。

### 3.4 导出范围

root、有子模块的模块，以及没有参数但在自身作用域内执行了算子的模块（例如 `core`）导出局部图。`Linear`、`RMSNorm`、`Embedding` 这类带参数的叶子模块是原子节点，只导出 `parameters`、`ops`、`nodes`，其 `ports` / `flow_nodes` / `edges` 为空数组。

### 3.5 封闭性

一条原始数据依赖 $(u \to v)$ 只在 $u$ 与 $v$ 的最近公共祖先作用域中表示为“子节点 → 子节点”的边；在两者之间的每个中间作用域中，它表现为 `@inK → child` 或 `child → @outK`。因此每张局部图只包含本模块可见的端点，边不会越过模块边界，也不会丢失。

### 3.6 残差标记

`is_skip` 在每个 $G(M)$ 构建完成后在局部图内计算：

> 边 $(u \to t)$ 标记为 skip，当且仅当 $t$ 是 `kind = op` 且 `op_type = Add` 的节点，且 $t$ 另有一条入边 $(w \to t)$，使得在 $G(M)$ 内 $u$ 可达 $w$。

汇聚用的 Add 不在本模块内时不会出现 skip 边，例如 `gpt.decoder.h` 中的 `0 → 1`。两个输入互不可达的 Add（`gpt` 中的 `embeddings_sum`，`core` 中的 `act_Add_20`）也没有 skip 边。

### 3.7 常量与根层级

* 常量缓冲区（`is_buffer = true`）在创建它的模块中是 `buffer` 节点：`gpt.pos_indices` 是 `gpt` 中的 `pos_indices → position_embedding`，`causal_mask` 是 `core` 中的 `causal_mask → act_Add_20`，两条边的 `kind` 都是 `buffer`。`pos_indices` 与 `token_ids` 之间只有形状关系、没有数据依赖，所以没有边。
* 图输入 / 输出是 **root 的端口**；图输入张量记录在 `root.nodes` 中。
* `default_scope`：root 只有一个子模块时为该子模块路径（例如 `"gpt"`），否则为 `""`（root）。页面默认展示该作用域的局部图。

---

## 4. JSON 格式 (JSON Format)

### 4.1 规范定义

JSON 的完整格式由 `model_graph.schema.json` 定义，它是字段含义与取值范围的唯一依据；本文中的 JSON 片段只是示例。

* 源文件：`src/nn/model_graph.schema.json`，由 `visualization.SCHEMA_JSON` 嵌入导出端，并由 `web/` 直接读取用于客户端校验与 JSON Format Guide 生成。
* 内容：顶层 `ModelHierarchyGraph` 与 9 个定义，共 71 个字段，每个字段都有 `description`。所有对象为 `additionalProperties: false`；`version` 固定为 `"2.0"`。
* 可省略的字段只有 `OpEntry.formula`、`FlowNode.op_type` / `module_type`、`Edge.dst_shape` / `transforms`，其余字段都必须出现。

结构概要（字段说明见 schema 或页面上的 JSON Format Guide）：

```text
ModelHierarchyGraph
├── version        "2.0"
├── summary        Summary        全局统计
├── default_scope  string         默认展示的模块路径
└── root           ModuleNode     路径为 "" 的根模块
    ModuleNode
    ├── name / path / kind("module") / module_type / formula
    ├── total_params / total_bytes / param_count / node_count
    ├── children     ModuleNode[]  直接子模块
    ├── parameters   ParamEntry[]  本模块的可学习张量
    ├── ops          OpEntry[]     本模块作用域内执行的算子
    ├── nodes        TensorNode[]  本模块拥有的全部张量
    ├── ports        Ports         { inputs: PortEntry[], outputs: PortEntry[] }
    ├── flow_nodes   FlowNode[]    局部图节点（§3.1）
    └── edges        Edge[]        局部图边（§3.2）
```

### 4.2 枚举字段

schema 中的每个 `enum` 在两端代码中都是枚举类型，测试断言两者逐一一致（包括顺序）：

| schema 字段 | 取值 | `znn`（Zig `enum`，经 `@tagName` 序列化） | `znn/web`（`as const` 列表与联合类型） |
| :--- | :--- | :--- | :--- |
| `TensorNode.kind` | `Param`、`Input`、`Buffer`、`Activation` | `NodeKind` | `NODE_KINDS` / `NodeKind` |
| `TensorNode.status` | `CUSTOM_INIT`、`AUTO_GRAPH`、`INPUT`、`BUFFER`、`OP_OUTPUT` | `NodeStatus` | `NODE_STATUSES` / `NodeStatus` |
| `ParamEntry.status` | `CUSTOM_INIT`、`AUTO_GRAPH` | `NodeStatus` 的子集 | `NodeStatus` 的子集 |
| `FlowNode.kind` | `port_in`、`port_out`、`module`、`op`、`buffer` | `FlowNodeKind` | `FLOW_NODE_KINDS` / `FlowNodeKind` |
| `Edge.kind` | `data`、`buffer` | `EdgeKind` | `EDGE_KINDS` / `EdgeKind` |

`ModuleNode.kind` 是常量 `"module"`，由序列化直接写出。

### 4.3 校验

schema 只使用以下关键字：`$ref`（指向 `$defs`）、`type`、`const`、`enum`、`minimum`、`required`、`properties`、`additionalProperties`、`items`。两端各有一个只支持这些关键字的校验器，不引入依赖：

* `znn`：测试用它校验 GPT 与单个 `Linear` 的导出结果。
* `znn/web`：`validateJson`（`web/src/utils/json-schema.ts`）校验上传的文件；不符合时在页头列出前几条违规及其 JSON 路径（例如 `$.root.flow_nodes[0].kind: expected one of ...`），不做部分渲染。

### 4.4 格式指南与模板

页面上的 “JSON Format Guide” 面板（`web/src/schema-guide.ts`）由 schema 生成：顶层和每个定义各一张表（字段、类型、是否必填、说明），引用类型可点击跳到对应的表。面板还提供 schema、最小模板与完整样例的下载，以及可折叠的完整 schema 文本。最小模板 `minimal_model_graph.json` 是单个 `Linear` 层（输入 `[2, 8]`，输出 `[2, 4]`）的真实导出结果。

### 4.5 修改流程

修改导出格式时，在同一变更中完成：

1. 修改 `serializeJson` 与 `model_graph.schema.json`（包括枚举与 schema `enum` 的对应关系）。
2. 运行 `zig build test`、`zig build run-report` 与 `zig build run-book-models`，重新生成样例模型图。
3. 同步修改 `web/` 渲染端与本文档，运行 `cd web && npm test`。
4. 在 `znn` 的 `CHANGELOG.md` `Unreleased` 小节记录。

---

## 5. 前端渲染 (Frontend Rendering)

图逻辑放在不依赖 DOM 的 `web/src/utils/model-graph.ts` 与 `web/src/utils/flow-edges.ts`，`web/src/scripts/visualizer.ts` 只负责渲染与交互。

1. **数据层**：`buildLocalGraph(mod)` 由 `flow_nodes` / `edges` 构建局部 DAG（入边、出边、稳定拓扑序）。节点类型只来自 `kind`、`module_type`、`op_type`，不从名字推断。
2. **路径解析**：`resolveCanonical(mod, id)` 返回 flow node 的 `ref`；缺少 flow node 时查 `ports`，再退回 `mod.path + "." + id`。卡片、Mermaid 节点与面包屑都用规范路径跳转，通过 `data-inspect` 属性和一个委托点击处理器触发。
3. **分层布局**：`layoutPipeline` 按最长路径分层，每层横向排成一行（不换行）。
   * 没有前驱的源节点（输入端口、buffer）放在其第一个消费者的上一层，例如 `core` 中 `causal_mask` 位于 `act_Add_20` 上方。
   * 同层单元按前驱在上层中的平均位置（重心）排序以减少交叉，下沉的源节点排在右侧。
   * 每个 flow node 只画一张卡片，共享输入只出现一次。卡片列出自己的入边（来源、`shape → dst_shape`、折叠的透明算子）。
4. **逐边连线**：渲染后测量卡片位置，由 `routeEdges` 为局部图的**每一条边**生成一条 SVG 连线，扇出、汇入与并行路径都按真实连接显示。容器尺寸变化时由 `ResizeObserver` 重新测量并重绘。
   * `data` / `buffer` 边从来源卡片底部连到目标卡片顶部，优先画成竖直直线：两张卡片横向有重叠时，取较窄卡片的中心（限制在重叠区间内）作为共同的 x。卡片没有横向重叠，或同一侧的锚点间距过近时，该侧锚点沿边均匀分布（按另一端的横坐标排序），连线在目标上方的间隙处拐弯，画成圆角直角折线。连线会穿过其他卡片时，改走局部图右侧的独立通道。
   * `skip` 边走残差块左侧的通道：一条细长的橙色条带，skip 边是其中从来源卡片底部直达 ⊕ 顶部的竖直箭头。⊕ 放在 F(x) 框下方、与来源卡片左对齐。来源或 ⊕ 未覆盖通道位置时（例如残差块位于多列行内），画成“来源底部 → 通道 → 从左侧连入 ⊕”的折线。
   * 样式：`data` 为实线，`skip` 为橙色虚线，`buffer` 为灰色点线，均带箭头；与 Mermaid 视图一致。
5. **残差块**：对 skip 边 $(s \to t)$，`residualBranch` 取 $\text{Reach}^{+}(s) \cap \left(\text{Reach}^{-}(w) \cup \{w\}\right)$（$w$ 为 $t$ 的非 skip 前驱），按拓扑序排列作为分支 F(x)。残差块（分支加汇聚节点）作为一个整体参与分层，分支内部递归调用 `layoutPipeline`，非链状分支与嵌套残差都能完整渲染。F(x) 由橙色虚线框包住，框左侧为 skip 通道，⊕ 位于框下方。
6. **⊕ 只表示加法**：只有 `op_type` 为 `Add` 的 `op` 节点显示为 ⊕；其他多入边节点（如 `core`）按普通卡片渲染，并标注输入数量。
7. **Mermaid**：`buildMermaidLocalGraph(mod)` 渲染一个作用域的局部图，每个直接子模块是一个节点。Mermaid 页默认展示 `default_scope`：点击有局部图的子模块进入其作用域，点击其他节点在 Inspector 页中打开，面包屑返回上层。边标签显示 `shape`，有 `dst_shape` 时显示 `shape → dst_shape`。
8. **Inspector**：检查只在 “Module Inspector & Submodules” 页内进行，页面不使用弹出窗口。架构图卡片、张量树、Mermaid 节点、子模块卡片、面包屑与 Quick Switch 的点击都经 `navigateInspector` 切换到该页并原地渲染目标，`← Back` 按访问历史返回。
   * 模块：Structure 页显示局部数据流、Mermaid 局部图与子模块卡片；节点、参数、算子由 `collectSubtree` 按模块树收集。
   * 张量：按全名精确匹配，Structure 页显示张量信息与产生它的算子。
   * 面包屑只包含真实存在的模块。
9. **其他**：KPI 的 Buffers 显示 `summary.buffer_nodes`；张量树按节点的 `module` 字段分组；JSON 解析或校验失败时在页头原地显示错误；复制按钮以按钮文字反馈结果。

---

## 6. GPT 样例的边集 (Golden Edge Sets)

配置为 `n_layer = 2`、`B = 2`、`T = 16`、`D = 64`、`V = 256`（`sample_model_graph.json`）。下列边集是导出端的验收标准。

**root**

| from | to | shape |
| :--- | :--- | :--- |
| `@in0` (inputs.token_ids) | `gpt` | `[2, 16]` |
| `gpt` | `@out0` (outputs.logits) | `[2, 16, 256]` |

**gpt**

| from | to | shape / dst_shape | 备注 |
| :--- | :--- | :--- | :--- |
| `@in0` | `token_embedding` | `[2, 16]` | |
| `pos_indices` | `position_embedding` | `[2, 16]` | `kind: buffer` |
| `token_embedding` | `embeddings_sum` | `[2, 16, 64]` | Add 节点 |
| `position_embedding` | `embeddings_sum` | `[2, 16, 64]` | 两条都不是 skip（互不可达） |
| `embeddings_sum` | `decoder` | `[2, 16, 64]` | |
| `decoder` | `lm_head` | `[2, 16, 64]` → `[32, 64]` | Reshape 属于 `gpt`，折叠进边 |
| `lm_head` | `@out0` | `[32, 256]` → `[2, 16, 256]` | |

**gpt.decoder**（TransformerDecoder）：`@in0 → h → ln_f → @out0`；**gpt.decoder.h**（ModuleList，各层 block 的容器）：`@in0 → 0 → 1 → @out0`。形状均为 `[2, 16, 64]`。

**gpt.decoder.h.0**（TransformerBlock）

| from | to | shape | skip |
| :--- | :--- | :--- | :--- |
| `@in0` | `ln_1` | `[2, 16, 64]` | |
| `@in0` | `residual_attn` | `[2, 16, 64]` | ✔ |
| `ln_1` | `attn` | `[2, 16, 64]` | |
| `attn` | `residual_attn` | `[2, 16, 64]` | |
| `residual_attn` | `ln_2` | `[2, 16, 64]` | |
| `residual_attn` | `residual_mlp` | `[2, 16, 64]` | ✔ |
| `ln_2` | `mlp` | `[2, 16, 64]` | |
| `mlp` | `residual_mlp` | `[2, 16, 64]` | |
| `residual_mlp` | `@out0` | `[2, 16, 64]` | |

端口 `ref`：`gpt.decoder.h.0` 的 `@in0` 为 `gpt.embeddings_sum`，`@out0` 为 `gpt.decoder.h.1`；`gpt.decoder.h.1` 的 `@out0` 为 `gpt.decoder.ln_f`。

**gpt.decoder.h.0.attn**

| from | to | shape → dst_shape | transforms |
| :--- | :--- | :--- | :--- |
| `@in0` | `q_attn` / `k_attn` / `v_attn` | `[2, 16, 64]` → `[32, 64]` | Reshape（同一个张量扇出到 3 条边） |
| `q_attn` | `core` | `[32, 64]` → `[2, 4, 16, 16]` | Reshape, Transpose(1,2) |
| `k_attn` | `core` | `[32, 64]` → `[2, 4, 16, 16]` | Reshape, Transpose(1,2)；$K^T$ 转置在 `core` 内部 |
| `v_attn` | `core` | `[32, 64]` → `[2, 4, 16, 16]` | Reshape, Transpose(1,2) |
| `core` | `c_proj` | `[2, 4, 16, 16]` → `[32, 64]` | Transpose(1,2), Reshape, Reshape |
| `c_proj` | `@out0` | `[32, 64]` → `[2, 16, 64]` | Reshape |

**gpt.decoder.h.0.attn.core**（ScaledDotProductAttention，无参数的多算子叶子）：`@in0(q)`、`@in1(k)`、`@in2(v)`、`causal_mask` → `BatchMatMul → MulScalar → Add → Softmax → BatchMatMul → @out0`。算子节点使用自动名（如 `act_BatchMatMul_18`）；$K^T$ 转置折叠进 `@in1 → act_BatchMatMul_18` 的 `transforms`；`causal_mask → act_Add_20` 为 `kind: buffer`。`act_Add_20` 的两个输入互不可达，没有 skip 边。

**gpt.decoder.h.0.mlp**

| from | to | shape → dst_shape |
| :--- | :--- | :--- |
| `@in0` | `c_fc` | `[2, 16, 64]` → `[32, 64]` |
| `c_fc` | `gelu` | `[32, 256]` |
| `gelu` | `c_proj` | `[32, 256]` |
| `c_proj` | `@out0` | `[32, 64]` → `[2, 16, 64]` |

---

## 7. 测试 (Tests)

**`znn`（`src/nn/visualization.zig`、`src/nn/graph_ir.zig`、`src/nn/tests_vis.zig`，`zig build test`）**

* `Explicit module scopes attribute ops and tensors to the executing module`：父模块 `forward` 中的 reshape、transpose、激活函数属于父模块，注意力核心属于 `core`。
* `Scoped local graph export matches golden edge sets (schema 2.0)`：以 `B = 2`、`T = 8`、`D = 32`、`V = 128` 的配置断言 §6 的边集、skip / buffer 标记与端口 `ref`。
* `Model graph JSON export conforms to the published JSON Schema`：GPT 与单个 `Linear` 的导出结果符合 schema；未声明字段与错误版本被拒绝。
* `Visualization enums match the enum lists of the published JSON Schema`：§4.2 的枚举与 schema 一致。

**`znn/web`（`cd web && npm test`，`vitest`）**

* `model-graph.test.ts`：局部 DAG、路径解析、残差分支、分层结果（包括重心排序与源节点下沉）。
* `flow-edges.test.ts`：竖直直线优先、锚点冲突时的直角折线、绕行通道、skip 直线通道及其折线形式。
* `json-schema.test.ts`：样例与最小模板符合 schema；每个字段都有说明、每个对象都是封闭的；违规报告带 JSON 路径；TypeScript 枚举列表与 schema 一致。
* `visualizer.test.ts`：前端页面渲染、模块导航与检查器交互测试。
