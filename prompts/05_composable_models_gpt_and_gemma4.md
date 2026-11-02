# Milestone 5: 核心模型全模块化重构 (Composable GPT & Gemma 4)

> **Status**: Completed (Milestone 5)  
> **Dependency**: Milestone 4 (`nn.Module` Protocol & Layers)  
> **Target Package**: `src/models/` (Layer 4)  
> **Key Files**: `src/models/gpt.zig`, `src/models/gemma4.zig`, `gemma4_q4.zig`, `gemma4_e4b.zig`

---

## 1. 目标与背景 (Objective)

当前 `src/models/` 中的大模型实现存在以下明显非模块化问题：
1. **GPT 包含过程式索引**：在 `models/gpt.zig` 中，位置编码采用硬编码循环 `pos_node.data[i] = @floatFromInt(i)`，并在多处执行动态 `reshape`，违背纯粹声明式 Module 前向调用；
2. **Gemma 4 包含大量裸指针偏移与裸切片索引**：在 `models/gemma4.zig` 的 `decodeAttention` 中，直接使用原生切片区间 `q_rot[h * hd .. (h + 1) * hd]`，且在 host float 上调用 `attendOneQuery`，绕过了标准的计算图与模块流向；
3. **模型层堆砌执行引擎**：包含控制台交互打印、硬件特定的执行引擎（如 `GpuLayerE4B`、`Gemma4GpuContext`），违反架构设计原则 §7（`models` 是纯粹的模型定义与架构实现，执行引擎必须属于 `engine`）。

---

## 2. 核心改造清单 (Delivered Changes)

### 2.1 彻底规范化 `GPT` 模型架构 (`src/models/gpt.zig`)

- 移除过程式 `pos_node.data` 赋值循环，采用标准声明式 `graph.arange` 与 `nn.Embedding` 广播；
- 移除前向传播中的动态临时 reshape，保证计算图拓扑声明式流畅；
- 确保 `models.GPT` 完全符合 `nn.Module` 协议，支持 `parameters()` 与 `state_dict` 导出。

### 2.2 消除 `Gemma 4` 裸指针切片与过程式过程代码 (`src/models/gemma4*.zig`)

- 彻底重构 `decodeAttention`：将单步自回归增量解码完整表达为标准计算图前向调用 (`graph.attention`)；
- 禁止直接在 host float 内存上调用底层密集单查询循环，所有增量注意力均由向量化 BLAS SGEMM 内核与标准 `Tensor` 闭环；
- 使 F32、Q4 与 E4B 三条路径均遵循统一的声明式 Module 体系。

### 2.3 剥离硬件引擎至 `src/engine/`

- 保持 `src/models/` 纯粹的模型层结构体声明，执行流程统一由 `engine` 与标准模块前向调度。

---

## 3. 验收标准与验证命令 (Acceptance Criteria)

- [x] `src/models/` 中不再包含任何裸切片计算循环（如 `for (0..num_heads)` 手工切片偏移）；
- [x] 所有模型均可被 `nn.walk` 透明遍历并支持无歧义命名；
- [x] 自回归生成在 CPU、CUDA、Metal 上具备完全相同的数值与一致的流式调用体验；
- [x] `zig build check-layers` 严格无违规通过；
- [x] `zig build test` 全量通过（0 错误，0 内存泄漏）；
- [x] `zig build run-book-models` 18 个模型图正常导出；
- [x] `cd web && npm test` 全量通过（203 tests passed）。
