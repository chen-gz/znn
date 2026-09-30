# ZNN 模型图可视化导出

完整规范位于 chen-gz.github.io 仓库的 `doc/visualization-model-edge-design.md`（ZNN 模型图可视化设计规范）。

要点：

- 模块在 `forward` 中通过 `Graph.enterModule` / `Graph.enterChildScope` 进入作用域，算子与张量记录所在作用域（`Op.scope`、`Tensor.scope`）。
- 每个导出局部图的模块给出 `ports`、`flow_nodes`、`edges`；端口 `@in<k>` / `@out<k>` 的 `ref` 在最近公共祖先作用域中解析；透明算子（Reshape、Transpose、RepeatKV）折叠进边的 `transforms`。
- JSON 格式由 `src/nn/model_graph.schema.json`（schema `2.0`）定义，每个字段都有说明；`TensorNode.kind`、`status`、`FlowNode.kind`、`Edge.kind` 在代码中是枚举类型（`NodeKind`、`NodeStatus`、`FlowNodeKind`、`EdgeKind`）。
- 实现位于 `src/nn/visualization.zig`；作用域、黄金边集、schema 与枚举一致性的测试位于 `src/nn.zig`。
- 修改导出格式时，同步更新 schema、`examples/` 中的样例、站点的 `public/tools/visualizer/` 副本与规范文档。
