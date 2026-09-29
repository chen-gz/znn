# ZNN 计算图拓扑边与层次化可视化导出设计

本设计已由 v2 规范取代并实施完成，权威文档位于 chen-gz.github.io 仓库：

- `doc/visualization-model-edge-design.md` (Visualization Model Edge Design v2)

v2 要点：

- 模块作用域由 `Graph.enterModule` / `Graph.enterChildScope` 显式记录 (`Op.scope`, `Tensor.scope`)，不再依据输入推断。
- 每个组合模块导出自身局部图 (`ports`, `flow_nodes`, `edges`)，端口 `@in<k>` / `@out<k>` 通过 `ref` 在最近公共作用域中解析。
- 可视化 JSON schema 2.0：移除顶层 `edges`，新增 `default_scope` 与 `summary.buffer_nodes`。
- 实现位于 `src/nn/visualization.zig`，黄金边集测试位于 `src/nn.zig`。
