# Milestone 2: 张量存储统一与算子零分配体系 (Tensor Storage & Destination Ops)

> **Status**: In Progress (in-place guard delivery implemented; full M2 pending)  
> **Target Package**: `src/tensor/` (Layer 1) & `src/backend/buffer.zig` (Layer 0)  
> **Key Files**: `src/tensor/core.zig`, `src/tensor/ops.zig`, `src/tensor/types.zig`

---

## 1. 目标与背景 (Objective)

当前 `Tensor` 结构体中同时维护了 `data: []f32`（CPU 宿主切片）与 `device_array: ?DeviceArray`（GPU/CUDA 显存句柄）。这种双重状态带来了严重的内存失步风险（例如 CPU 梯度更新后未能及时重置 GPU 句柄，或者 GPU 前向计算后 CPU 端切片为陈旧数据）。

同时，张量层的大多数算子均采用开辟新内存的方式（Eager Allocating），在计算图前向重演中需要依赖 `copyFromEager` 进行二次内存拷贝。

本里程碑的目标是：
1. **单一真实来源 (Single Source of Truth)**：`Tensor` 内部仅持有 `storage: backend.DeviceBuffer`，所有数据访问通过受控方法瞬时映射；
2. **目标写入算子体系 (Destination `*Into` Ops)**：为所有核心张量算子提供零分配的 `*Into` 变体；
3. **标准化强类型错误分类 (TensorError Taxonomy)**：淘汰在 ReleaseFast 下被剥离的 `assert`。

---

## 2. 接口契约与具体改造清单 (Interface Contracts)

### 2.1 重构 `Tensor` 核心结构体 (`src/tensor/core.zig`)

```zig
pub const Tensor = struct {
    storage: backend.DeviceBuffer,      // 单一真实来源的底层物理缓冲区
    shape: Shape,
    strides: Shape,
    dtype: DType = .f32,
    requires_grad: bool = false,
    grad: ?*Tensor = null,              // 梯度张量本身也是标准 Tensor
    creator: ?*OpHandle = null,
    is_buffer: bool = false,            // 区分可训练参数与非参数缓冲区
    name: ?[]const u8 = null,

    /// 获取 CPU 连续宿主切片（若张量位于 GPU，则自动触发物理缓冲区同步）
    pub fn asHostSlice(self: *const Tensor) []const f32;
    pub fn asHostSliceMut(self: *Tensor) []f32;

    /// 显存句柄访问
    pub fn deviceBuffer(self: *const Tensor) *const backend.DeviceBuffer {
        return &self.storage;
    }
};
```

### 2.2 目标写入算子体系 (`src/tensor/ops.zig` & `core.zig`)

为所有核心算子补充预分配写入版本，要求 `dest` 形状与步长已对齐，内部直接调用底层 `kernels` 或 SIMD 原语，**绝对零动态内存分配**：
- `pub fn matmulInto(dest: *Tensor, a: *const Tensor, b: *const Tensor) !void`
- `pub fn addInto(dest: *Tensor, a: *const Tensor, b: *const Tensor) !void`
- `pub fn mulInto(dest: *Tensor, a: *const Tensor, b: *const Tensor) !void`
- `pub fn rmsNormInto(dest: *Tensor, x: *const Tensor, weight: ?*const Tensor, options: RmsNormOptions) !void`
- `pub fn softmaxInto(dest: *Tensor, x: *const Tensor, dim: ?usize) !void`
- `pub fn geluInto(dest: *Tensor, x: *const Tensor, approx: GeluApproximation) !void`

### 2.3 规范化强类型错误体系 (`src/tensor/types.zig`)

```zig
pub const TensorError = error{
    InPlaceOpOnGraphTensor,    // 严禁对 requires_grad 或具有 creator 的图张量执行原地修改
    ShapeMismatch,
    StrideMismatch,
    DeviceMismatch,
    NonContiguousLayout,
    UnsupportedDType,
    DimensionOutOfBounds,
};
```

---

## 3. 验收标准与验证命令 (Acceptance Criteria)

- [ ] `Tensor` 结构体中不再包含独立的 `device_array` 与裸 `data: []f32` 字段，全局统一为 `storage: backend.DeviceBuffer`；
- [ ] 核心算子均具备对应的 `*Into` 签名，并在单元测试中验证零内存分配；
- [ ] 原地操作保护机制由运行时 `TensorError.InPlaceOpOnGraphTensor` 显式抛出并有测试覆盖；
- [ ] `zig build check-layers` 严格无违规通过；
- [ ] `zig build test` 全量通过（0 错误，0 内存泄漏）；
- [ ] `zig build run-book-models` 18 个模型图正常导出。

## 4. Execution checkpoint (2026-10-26)

First delivery targets the existing in-place arithmetic guards. `mulScalar_`,
`addScalar_`, and `add_` currently use debug assertions, so optimized builds can
silently mutate graph tensors. Replace these guards with explicit
`TensorError.InPlaceOpOnGraphTensor` returns before any write. Scalar methods
become fallible; update every repository caller, including regression examples.

Files: `src/tensor/types.zig`, `src/tensor.zig`, `src/tensor/reductions.zig`,
`src/tensor/core.zig`, `src/tensor/tests.zig`, and the linear, logistic, and ridge regression examples.
Validate leaf tensors requiring gradients, creator-backed tensors, strided views,
shape mismatch atomicity, and ordinary eager operations. Run the tensor tests in
Debug and ReleaseFast, the full test suite, the layer checker, and model exports.
The storage migration and destination operations remain pending; this delivery
does not complete M2. Next work should establish storage ownership and view
lifetimes before replacing the existing host/device fields.

Review extension: slice and transpose views must retain a separate
`is_graph_view` mutation-protection bit without claiming to be differentiable
view operators. Propagate it through nested views, and enforce it in all five
existing guarded in-place operations (including `clip_` and `maskedFill_`).
Independent contiguous copies remain mutable. This does not add storage version
counters or retroactively track changes to the parent's graph membership.

### Validation and continuation

- Passed `zig build check-layers` (84 source files).
- The two new regression test bodies passed in Debug and ReleaseFast against
  the actual tensor module via an isolated test entry point. The entry point
  used the same test bodies and a separate named tensor module to avoid linking
  unrelated graph execution code. This is focused validation, not a full-suite run.
- Full native `zig build test` and model exports remain unverified because
  `deps/mlx-c/lib` is absent from this checkout. The x86_64 cross-build additionally
  exposes an existing unguarded `mlx.c.mlx_array_data_uint8` reference in
  `src/backend/buffer.zig` on a target without MLX; this host has no Rosetta runtime.
- An independent agent reviewed the guards, view propagation, and callers and
  found no blocking defect in this bounded change.

Next: restore a supported full-test environment, run the complete suite and
`run-book-models`, then define ownership, synchronization, and view lifetime
contracts for the `DeviceBuffer` migration. Keep M2 and M3 acceptance checkboxes
open until their complete validation matrices pass.
