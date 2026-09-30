# ZNN 计算图拓扑边与层次化可视化导出设计 (Visualization Model Edge Design v2)

本文档定义 `znn` 模型计算图导出与 Web 端交互式可视化（Schema 2.0）的权威数据协议与实现规范。

---

## 1. 架构目标与 v1 问题诊断

在早期的计算图导出设计（v1 / Schema 1.0）中，存在若干结构性缺陷：
1. **模块内部细节向外泄露**：子模块内部的微观算子与临时张量节点，在顶层视图中被错误展开为全局扁平边，造成图谱极端臃肿；
2. **作用域推断不可靠**：早期的算子作用域依赖输入张量所属模块进行启发式猜测，当跨模块张量传递时极易误判；
3. **缺少端口抽象**：无法清晰表达一个复合模块的输入（Inputs）与输出（Outputs）边界；
4. **残差连接拓扑混乱**：跳跃连接（Skip Connection）在展开后跨越多个层级，导致连线严重交叉重叠。

在 **Schema 2.0** 中，上述问题通过“**局部封闭律 (Local Encapsulation)**”与“**端口化抽象 (Port Abstraction)**”彻底解决。

---

## 2. 核心设计原则 (Design Principles)

1. **显式作用域记录 (Explicit Module Scoping)**：
   - 依赖 `Graph.enterModule` 与 `Graph.enterChildScope` 运行栈；
   - 算子与张量在创建瞬间立即绑定所属模块路径 (`Op.scope`, `Tensor.scope`)，彻底废弃基于输入的启发式推断。
2. **每个模块携带独立的局部图 (Scoped Local Graph)**：
   - 每个复合模块导出自己的 `ports`、`flow_nodes` 与 `edges`；
   - 外部调用者仅看到模块的黑盒端口，点击进入后才展开内部流图。
3. **最近公共祖先拓扑收归 (LCA Edge Routing)**：
   - 跨越不同模块的边，自动在两者的最近公共祖先 (Lowest Common Ancestor) 作用域中收归为对应的端口连线。
4. **透明算子折叠 (Transparent Operator Folding)**：
   - 针对无参数、纯形状变换的透明算子（如 `Reshape`、`Transpose`），支持在导出时折叠或作为边的变换属性（`transforms: ["reshape", "transpose"]`）依附于数据边上。
5. **张量节点分类强类型化 (`NodeKind`)**：
   - 明确分为 `Param`、`Input`、`Buffer` 与 `Activation` 四种形态。

---

## 3. Schema 2.0 数据结构规范

导出的 JSON 数据结构遵循 `src/nn/model_graph.schema.json` 定义：

```typescript
export interface ModelHierarchyGraph {
  schema_version: "2.0";
  model_name: string;
  default_scope: string; // 默认展示的顶层作用域 (通常为 "")
  summary: {
    total_params: number;
    total_memory_bytes: number;
    activation_nodes: number;
    buffer_nodes: number;
  };
  modules: Record<string, ModuleNode>; // 键为全路径，如 "", "gpt.layers.0", "gpt.layers.0.attn"
  nodes: NodeData[];                   // 所有底层张量元数据列表
  ops: OpData[];                       // 所有算子元数据列表
}

export interface ModuleNode {
  path: string;
  name: string;
  module_type: string;
  param_count: number;
  memory_bytes: number;
  formula?: string; // 显式数学公式，如 "y = x W^T + b"
  submodules: string[]; // 直接子模块列表
  ports: {
    inputs: PortEntry[];   // 虚拟输入端口，如 id: "@in0", ref: "gpt.x"
    outputs: PortEntry[];  // 虚拟输出端口，如 id: "@out0", ref: "gpt.layers.11.out"
  };
  flow_nodes: FlowNode[];  // 当前局部图中的节点 (子模块、内部算子或端口)
  edges: EdgeData[];       // 当前局部图中的有向连线
}
```

---

## 4. 与 Web 前端 (chen-gz.github.io/visualizer) 的契约与协同升级

遵循项目全局规范中的“**生产端与消费端同步升级 (Upgrade Producer and Consumer Together)**”原则：
* `znn` 作为生产端 (`src/nn/visualization.zig`)；
* `chen-gz.github.io` 作为消费端 (`src/utils/model-graph.ts`, `src/scripts/visualizer.ts`, `src/pages/visualizer.astro`)；
* 共享权威 Schema 文件：`model_graph.schema.json`；
* 共享基准样本文件：`examples/sample_model_graph.json` 与 `public/tools/visualizer/sample_model_graph.json` 保持 100% 字节级一致。
