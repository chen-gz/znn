# Milestone 3: 自动微分重放与分配器受控体系 (Autodiff Replay & Allocator Discipline)

> **Status**: Queued (Milestone 3)  
> **Dependency**: Milestone 2 (`Tensor.storage` & `*Into` Ops)  
> **Target Package**: `src/autodiff/` (Layer 2)  
> **Key Files**: `src/autodiff/op.zig`, `src/autodiff/graph.zig`, `src/autodiff/backward_*.zig`

---

## 1. 目标与背景 (Objective)

当前计算图执行存在两处内存与性能痛点：
1. **前向重演内存抖动 (Forward Memory Thrashing)**：`op.forward` 与 `runAndRecordPreallocatedOp` 内部采用 `copyFromEager`——即每次通过 eager 函数在堆上重新分配临时张量计算，然后再拷贝回预先分配的目标节点，甚至在重演前还多余地执行显式清零；
2. **反向求导原生分配器失控**：部分复杂的反向算子（如卷积、池化或高维归约）直接硬编码调用 `std.heap.c_allocator` 分配临时缓冲区，未接入计算图绑定的生命周期管理，引发隐式内存碎片。

---

## 2. 核心改造清单 (Delivered Changes)

### 2.1 消除 `copyFromEager`，前向直写输出 (`src/autodiff/op.zig`)

```zig
pub fn forward(self: *Op, allocator: std.mem.Allocator) !void {
    _ = allocator; // 零堆分配前向重演
    const out = self.outputs[0];
    switch (self.op_type) {
        .matmul => try tensor.matmulInto(out, self.inputs[0], self.inputs[1]),
        .add => try tensor.addInto(out, self.inputs[0], self.inputs[1]),
        .mul => try tensor.mulInto(out, self.inputs[0], self.inputs[1]),
        .rms_norm => try tensor.rmsNormInto(out, self.inputs[0], if (self.inputs.len > 1) self.inputs[1] else null, self.context.rms_norm),
        // ...其余全部通过 *Into 直写预分配节点
    }
}
```

### 2.2 显式反向工作区分配器 (`src/autodiff/backward_*.zig`)

- 为所有反向算子引入显式分配器参数签名：
  `pub fn backward(self: *Op, scratch_allocator: std.mem.Allocator) !void`
- 遍历并彻底清除 `backward_core.zig`、`backward_nn.zig`、`backward_math.zig` 中所有的 `std.heap.c_allocator`；
- 所有临时工作缓冲（如反向矩阵转置、临时掩码切片）全部从计算图绑定的 `graph.arenaAllocator()` 中开辟，反向结束后随图生命周期整体回收。

### 2.3 `GraphOptions` 清理与损失函数去重

- 彻底清除 `GraphOptions` 中遗留的厂商硬件特定字段（如 `mlx_stream`），使计算图配置保持纯粹平台中立；
- 统一收敛损失函数节点注册接口，通过 `src/autodiff/ops.zig` 统一分发。

---

## 3. 验收标准与验证命令 (Acceptance Criteria)

- [ ] `op.forward` 中彻底废除 `copyFromEager`，前向计算不产生任何多余临时张量与无谓拷贝；
- [ ] 全文搜索 `src/autodiff/`，`std.heap.c_allocator` 出现次数严格为 0；
- [ ] `zig build check-layers` 严格无违规通过；
- [ ] `zig build test` 全量通过（0 错误，0 内存泄漏）；
- [ ] `zig build run-book-models` 18 个模型图正常导出。
