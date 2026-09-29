# 📐 ZNN 计算图拓扑边与层次化可视化导出设计规范
# (ZNN Computation Graph Edges & Hierarchical Visualization Export Design)

## 1. 背景与现状诊断 (Background & Diagnostics)

`znn` 提供了将前向计算图及模块层次树导出为 JSON（`graph.exportJson(output_path)`）的功能，供前端可视化工具（如 `chen-gz.github.io/visualizer`）进行动态交互展示。

在审查当前的计算图边采集机制（[`src/nn/visualization.zig`](file:///usr/local/google/home/guangzong/Documents/znn/src/nn/visualization.zig)）时，发现以下关键缺陷导致导出的边数据异常：

1. **子模块边越界污染（Boundary Leakage）**：
   `collectGraphEdges` 将计算图中的所有跨模块边归属到 `module_map` 中的各个模块时，使用了条件：
   ```zig
   if (!from_in and !to_in) continue;
   ```
   导致起点在当前模块但终点在下一模块的跨层连接（例如 `gpt.layers.0.output -> gpt.layers.1.ln_1`）被强行塞入了 `gpt.layers.0.edges` 中。
2. **顶层宏观边收拢粒度断层（Macro Scope Inconsistency）**：
   `isSublayerModuleType` 仅定义了部分复合算子（`CausalSelfAttention`, `MLP` 等），未涵盖顶层主要模块（`TransformerBlock`, `RMSNorm`）。导致顶层 `edges` 既未收敛至纯模块间宏观流转，又破坏了原本细粒度连线的可追溯性。
3. **初始嵌入层孤立输入缺失**：
   位置嵌入 `gpt.wpe` 依赖的位置索引张量被判定为 buffer 忽略，导致 `gpt.wpe` 在图中成为入度为 0 的孤立起点，而未与模型全局输入连接。
4. **命名截断破坏全局索引链接**：
   子模块边为了相对简短使用了截断名称（如 `q_attn`），但在节点树中所有节点和参数均使用全路径命名（如 `gpt.layers.0.attn.q_attn`），导致前端通过边端点进行逆向 Inspector 查找时无法匹配。

---

## 2. 拓扑边层次封闭规范 (Hierarchical Edge Enclosure Specification)

为确保计算图在任意模块层次上的自洽与封闭性，确立以下设计规范：

### 2.1 局部作用域封闭律 (Scope Enclosure Law)
* 模块 $M$ 的 `m.edges` 集合中，**必须且仅能包含端点完全由 $M$ 直接管辖的边**：
  1. $M$ 内部子模块之间的连接（如 $M.\text{ln\_1} \to M.\text{attn}$）；
  2. $M$ 内部的残差跳跃旁路（如 $M.\text{inputs} \to M.\text{residual\_attn}$）；
  3. $M$ 内部变换结果向汇聚节点的连接（如 $M.\text{attn} \to M.\text{residual\_attn}$）；
  4. 最终汇聚节点向 $M$ 输出端口的连接（如 $M.\text{mlp} \to M.\text{output}$）。
* **跨模块连线一律上浮至两者的最近公共祖先模块（LCA）管理**。例如 `layers.0.output -> layers.1.ln_1` 只能存在于父级 `gpt.layers` 或 `root` 的边集合中，绝对禁止出现在 `layers.0` 的局部边中。

### 2.2 权威全路径与命名标准 (Canonical Path Standard)
* 全局边集合（`graph_ir.edges`）一律保存绝对全路径（Canonical Path，例如 `gpt.layers.0.ln_1`）。
* 模块内部边（`m.edges`）中的内部连接使用统一的直接子级标识（如 `ln_1`, `attn`, `residual_attn`），输入输出接口统一定义为 `inputs` 与 `output`。
* 保证前端解析器可以无歧义地通过 `mod.path + "." + e.to` 拼装出全局唯一节点路径。

---

## 3. 实现计划 (Implementation Plan)

1. **重构 `src/nn/visualization.zig` 中的边划分逻辑**：
   * 严格限定只有当 `from_in and to_in` 同时成立，且没有更深层共同子模块承载时，才归入当前模块的 `m.edges`。
   * 对于模块输入接口，引入清晰的 `inputs -> first_child` 虚拟入边；对于模块输出接口，引入 `last_child -> output` 的接口边。
2. **规范顶层宏观边生成**：
   * 消除散装无归属节点，保证 `root.edges` 清晰展现模型主线流水。
3. **重新导出权威样本**：
   * 运行 `zig build run-report` 生成规范的 `examples/sample_model_graph.json`。
   * 同步至前端网站仓库 `chen-gz.github.io/public/tools/visualizer/sample_model_graph.json`。
