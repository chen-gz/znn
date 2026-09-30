# ZNN 模型图导出 (Model Graph Export)

`znn` 把一次 `forward` 的计算图导出为分层的模型图 JSON，供 chen-gz.github.io 的 `/visualizer` 页面渲染。完整规范（局部图模型、端口解析、残差判定、前端渲染与 GPT 黄金边集）见 chen-gz.github.io 仓库的 `doc/visualization-model-edge-design.md`。

## 1. 流程

```text
Module.forward ──(作用域栈)──▶ autodiff.Graph ──graph_ir.build──▶ ModelHierarchyGraph ──serializeJson──▶ JSON
```

* **模块作用域**：每个模块在 `forward` 开头调用 `Graph.enterModule(graph, self.name, self.module_type)`，并以 `defer scope.exit()` 退出；模块内部的子逻辑用 `Graph.enterChildScope`（例如注意力核心 `core`）。算子与张量在创建时记录所在作用域（`Op.scope`、`Tensor.scope`）。
* **局部图**：root、有子模块的模块，以及无参数但在自身作用域内执行了算子的模块，各自导出 `ports`、`flow_nodes`、`edges`。节点是直接子模块、自身算子、常量缓冲区与边界端口 `@in<k>` / `@out<k>`；端口的 `ref` 在最近公共祖先作用域中解析。
* **透明算子**：`Reshape`、`Transpose`、`RepeatKV` 不作为节点，按执行顺序折叠进边的 `transforms`。
* **残差**：终点为 Add、且起点在局部图内可达该 Add 另一条入边来源的边标记 `is_skip`。

## 2. JSON 格式

* 格式由 [`src/nn/model_graph.schema.json`](../src/nn/model_graph.schema.json)（JSON Schema draft 2020-12，schema `2.0`）定义，每个字段都有 `description`，所有对象不允许未声明字段；导出端通过 `visualization.SCHEMA_JSON` 嵌入。
* 顶层字段：`version`、`summary`、`default_scope`、`root`；`root` 是递归的 `ModuleNode`（`children`、`parameters`、`ops`、`nodes`、`ports`、`flow_nodes`、`edges` 等）。
* 取值受限的字段在代码中是枚举类型，测试断言其标签与 schema 的 `enum` 列表一致：

| schema 字段 | Zig 枚举 |
| :--- | :--- |
| `TensorNode.kind` | `NodeKind` |
| `TensorNode.status` / `ParamEntry.status` | `NodeStatus` |
| `FlowNode.kind` | `FlowNodeKind` |
| `Edge.kind` | `EdgeKind` |

## 3. 代码与测试

| 位置 | 内容 |
| :--- | :--- |
| `src/autodiff/graph.zig` | 作用域栈、`enterModule` / `enterChildScope` |
| `src/nn/visualization.zig` | 模块树、局部图构建、枚举、JSON 序列化 |
| `src/nn.zig` | 作用域归属、黄金边集、schema 一致性与枚举一致性测试 |
| `examples/export_model_report.zig` | `zig build run-report` 导出 `examples/sample_model_graph.json`（2 层 GPT）与 `examples/minimal_model_graph.json`（单个 `Linear`） |

## 4. 修改导出格式

在同一变更中完成（与 [`AGENTS.md`](../AGENTS.md) §3 一致）：

1. 修改导出端与 `model_graph.schema.json`。
2. 运行 `zig build test` 与 `zig build run-report`。
3. 把 schema 与两个样例逐字节复制到 chen-gz.github.io 的 `public/tools/visualizer/`，同步修改渲染端与规范文档。
4. 在 `CHANGELOG.md` 的 `Unreleased` 小节记录。
