# Milestone 4: PyTorch 风格 `nn.Module` 协议与图层标准化

> **Status**: Queued (Milestone 4)  
> **Dependency**: Milestone 2, Milestone 3  
> **Target Package**: `src/nn/` (Layer 3)  
> **Key Files**: `src/nn/module.zig`, `src/nn/core.zig`, `src/nn/attention.zig`, `src/nn/normalization.zig`

---

## 1. 目标与背景 (Objective)

当前 `znn` 的神经网络模块基于简单的结构体反射遍历（`nn.walk`），但在使用体验和工程健壮性上与 PyTorch `nn.Module` 仍有明显差距：
1. **缺少编译期模块协议校验**：任何结构体只要有字段就能算模块，缺少对 `init`, `deinit`, `forward`, `resetParameters` 的编译期契约断言；
2. **参数与缓冲区生命周期混淆**：`BatchNorm` 的 `running_mean` 与 `running_var` 等非训练参数缺乏第一公民的 Buffer 支持；`setRequiresGrad` 容易错误覆写超参数缓冲区；
3. **量化层被反射遗漏**：`Q4Linear` 因采用自定义量化权重结构，无法被 `nn.walk` 自动递归遍历与参数计数；
4. **注意力算子实现碎片化**：存在多个 attention 实现，缺乏统一标准的多头/分组查询注意力 (GQA) 与统一 KV 缓存。

---

## 2. 核心改造清单 (Delivered Changes)

### 2.1 编译期 `nn.Module` 协议强类型验证 (`src/nn/module.zig`)

提供编译期内省工具 `verifyModule(comptime T: type) void`，强制所有网络层符合标准模块形态：
- 具备 `init(allocator: Allocator, ...) !Self`
- 具备 `deinit(self: *Self, allocator: Allocator) void`
- 具备 `forward(self: *const Self, graph: *Graph, x: *Tensor) !*Tensor`
- 具备可选的 `resetParameters(self: *Self, random: Random, options: InitOptions) void`

### 2.2 显式区分可训练参数与持久缓冲区 (`Parameters` vs. `Buffers`)

- 引入两类明确的持久化张量分配器：
  - `pub fn createParameter(allocator: Allocator, shape: []const usize, requires_grad: bool) !*Tensor`（标记 `.is_buffer = false`）
  - `pub fn createBuffer(allocator: Allocator, shape: []const usize) !*Tensor`（标记 `.is_buffer = true`, `.requires_grad = false`）
- `nn.setRequiresGrad` 遇到 `is_buffer == true` 的张量严格跳过；
- 优化器在收集可优化参数列表时，自动过滤所有 `is_buffer == true` 的持久化缓冲区。

### 2.3 完善反射迭代器与容器支持

- 在 `src/nn/module.zig` 中新增：
  - `pub fn buffers(model: anytype, allocator: Allocator) ![]*Tensor`
  - `pub fn namedBuffers(model: anytype, allocator: Allocator) !NamedParameterList`
  - `pub fn submodules(model: anytype, allocator: Allocator) ![]NamedModule`
- 适配 `Q4Linear`：重构量化结构体字段，使 `walk` 能透明访问量化权重元数据与缩放因子。
- 提供泛型容器：
  - `nn.ModuleList(comptime T: type)`
  - `nn.Sequential(comptime LayersTuple: type)`

### 2.4 统一注意力机制与键值缓存

- 收敛注意力实现为单一权威的 `nn.GQAAttention`（支持 MHA、GQA、MQA、滑动窗口与头归一化）；
- 统一 `nn.KVCache` 为基于张量存储的标准模块，消除各模型私有缓存结构。

---

## 3. 验收标准与验证命令 (Acceptance Criteria)

- [ ] `src/nn/` 中所有基础层均通过 `verifyModule` 编译期协议断言；
- [ ] `BatchNorm` 与归一化层的缓冲区与学习参数生命周期完全解耦，`setRequiresGrad(false)` 不破坏缓冲区；
- [ ] `Q4Linear` 能被 `nn.parameters` 与 `nn.namedParameters` 正确遍历；
- [ ] `zig build check-layers` 严格无违规通过；
- [ ] `zig build test` 全量通过（0 错误，0 内存泄漏）；
- [ ] `zig build run-book-models` 18 个模型图正常导出。
