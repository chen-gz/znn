# Milestone 2: 张量存储统一与算子零分配体系 (Tensor Storage & Destination Ops)

> **Status**: In Progress (CPU ownership, typed GPU adoption, and six destination APIs implemented; Tensor migration pending)  
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

    /// CPU-only host access, without implicit allocation or device transfer.
    pub fn asHostSlice(self: *const Tensor) TensorError![]const f32;
    pub fn asHostSliceMut(self: *Tensor) TensorError![]f32;

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
- [x] CPU 核心算子均具备对应的 `*Into` 签名，并在单元测试中验证零内存分配；
- [x] 原地操作保护机制由运行时 `TensorError.InPlaceOpOnGraphTensor` 显式抛出并有测试覆盖；
- [x] `zig build check-layers` 严格无违规通过；
- [x] CPU `zig build test -Dmlx=false` 全量通过（0 错误，0 内存泄漏）；
- [x] CPU `zig build run-book-models -Dmlx=false` 18 个模型图正常导出。

## 4. Current implementation and continuation

Implemented and independently reviewed:

- In-place arithmetic returns explicit errors before graph data is modified in
  Debug and ReleaseFast. Nested slice/transpose views preserve mutation guards.
- CPU-only builds use `-Dmlx=false`, with optional `-Dmacos-sdk` lookup.
- All six destination APIs are implemented for CPU storage. Inputs may be
  strided; output must be independent and contiguous. Shape, layout, storage,
  residency, and alias checks run before writes. No device transfer is implicit.
- Matrix multiplication supports rank-two inputs and a zero contraction dimension.
  Softmax accepts any axis. Root mean square normalization accepts optional
  strided weights. GELU supports both approximation modes and scalar inputs.
- CPU DeviceBuffer allocations retain allocator provenance and an atomic shared
  owner. `retain()` and byte-range `slice()` acquire ownership explicitly; plain
  descriptor assignment does not. Either parent or view may be released first.
  `free()` releases through the stored allocator and resets the descriptor.
- Empty CPU buffers/views allocate nothing. Allocation bases remain 64-byte
  aligned; typed view access validates element alignment. CPU copies are safe
  with overlapping ranges. Allocation failure cleanup has regression coverage.

### Remaining storage migration contract

Do not mechanically replace Tensor fields before resolving these requirements:

1. Implement fallible host-only accessors and separate explicit synchronization.
   Device transfers
   may fail or allocate and must not be hidden in an accessor.
2. Integrate retained typed GPU buffers into Tensor factories and accessors.
   Adoption supports contiguous float32 arrays; other types/layouts are rejected.
   GPU byte slicing and raw GPU-buffer retention remain unsupported. Raw Metal
   buffer kernels are not yet implemented.
3. Validate Metal transfers and CUDA retained reshape lifetimes on supported
   GPU runtimes. Host control-record tests cannot validate device execution.
4. Migrate Tensor factories, slices, transpose views, graph reshape views, graph
   arena allocations, external parameters, and cleanup together. Preserve empty
   tensors and scalar storage. Remove obsolete data/device_array fields and all
   consumers in the same cohesive migration.
5. Route graph replay through destination operations only after ownership and
   residency contracts are established. M3 execution-buffer reuse is still pending.

### Validation checkpoint

Validated on 2026-10-26 with Zig 0.17.0 on Apple Silicon:

- Full CPU Debug and ReleaseFast suites each passed 417 tests, with five GPU
  runtime tests explicitly skipped (422 total).
- Layer checks passed for 86 source files; all 18 canonical model graphs exported
  successfully after the host-access and transition changes.
- GPU-enabled and x86_64 Linux source passed compile-only checks.
- Independent reviews found no blocking ownership or destination-operation defects.
- Metal and CUDA execution were not exercised. These results do not complete M2.

Reproduce with `zig build test -Dmlx=false`, repeat with
`-Doptimize=ReleaseFast`, and run `zig build run-book-models -Dmlx=false`.
Supply `-Dmacos-sdk` with the local SDK path if framework discovery needs it.

## 5. Buffer transfer capacity implementation

Metal prefix uploads and device copies preserve destination capacity and suffix
bytes, matching the CPU/CUDA prefix-copy contract. Empty copies are no-ops after
device/bounds validation. A fresh slice-update result is evaluated before the old
handle is released. Byte-array type, rank, contiguity, and physical capacity are
validated before nonempty transfers; malformed descriptors fail explicitly.
Allocation sizes beyond Metal's signed shape range fail before narrowing casts.

CPU prefix-copy and empty-transfer contracts pass in Debug and ReleaseFast.
The conditional Metal regression covers uploads, self/device copies, suffix
preservation, and malformed storage. It compiles with Metal enabled but was not
executed because the optional libraries are unavailable. CPU/Linux compilation
also passes. Runtime Metal validation remains required before GPU adoption.

## 6. CUDA reshape ownership implementation

Every CUDA Array producer now creates a shared atomic allocation record, storing
its device pointer and host allocator. Retain and reshape acquire references;
plain descriptor copies are borrowed. Reshape validates shape products before
retention and may outlive its parent. Clone remains a deep copy. Deinitialization
resets the descriptor and frees the device allocation only on final release.
Existing allocation error paths release device bytes if owner creation fails.

Tensor.toCudaArray returns an owned reference in both resident and upload paths.
Graph reshape releases a newly retained device array if allocation or tensor-list
registration fails; after registration the graph owns its cleanup.

CPU tests exercise nested views, both release orders, allocation failure,
invalid/overflowing shapes, and reference-count overflow using null device
pointers. A conditional CUDA test covers real allocation lifetime and clone
independence. It is skipped on this host, as is the Metal runtime regression.

## 7. Typed GPU buffer integration implementation

DeviceBuffer.fromDeviceArray validates and retains contiguous float32 CUDA/Metal
arrays, the current Tensor representation. Shape/element/byte counts and ownership
are checked before retention; CUDA control records also retain allocation size.
Unsupported dtypes/layouts fail explicitly. The typed driver array is the owner;
the raw kernel handle is a borrowed alias. free() releases the owner once and
retain() acquires another independent reference. GPU byte slicing is unsupported.

borrowedDeviceArray exposes a borrowed descriptor for typed kernels; callers must
not deinitialize it. copyToHostF32 transfers explicitly into exact-length caller
storage. Raw byte copies and setZero reject adopted storage so typed Metal arrays
cannot be replaced by byte arrays. Existing raw buffer behavior is unchanged.
DeviceArray.retain uses CUDA retention or MLX shared handles; Metal clone returns
an independent copy instead of aliasing a handle's ownership.

Tests cover invalid ownership, dtype and metadata, physical allocation-size
mismatch, and invalid host destinations. Conditional CUDA/Metal tests cover
source release before buffer use, retained buffers, clones, typed transfer, and
raw-byte-operation rejection. Four GPU runtime tests are skipped on this host.

### Next Tensor migration delivery

Migrate Tensor factories, views, and Graph cleanup together onto DeviceBuffer.
Replace direct host fields with fallible host access and explicit synchronization;
retain storage for slices/transpose/reshape views. Preserve graph arenas, external
parameter ownership, zero elements, and scalar storage. Update all consumers in
one cohesive migration; do not keep duplicate authoritative host/device state.
The typed adoption path needs GPU runtime validation before claiming M2 complete.

## 8. Host access and transition implementation

Tensor now exposes fallible contiguous CPU host accessors with checked rank,
strides, counts, and bounds. Mutable access retains graph mutation guards. These
accessors never trigger transfer or allocation. The storage fields themselves
remain unchanged pending the cohesive Tensor/Graph migration.

Host synchronization stages data before changing the tensor. Failure leaves the
old data, device, and array ownership untouched. Successful synchronization releases
the old device array so later CPU writes cannot be hidden by a stale cached handle.
Growing host storage requires an explicit allocator; graph callers must use the
graph arena for graph-owned host allocations. Borrowed-view materialization is rejected
until shared storage is migrated. Device upload commits only after creating an owned copy of host bytes and rejects implicit
cross-device transitions. Tests cover validation and failure atomicity without
GPU libraries, plus conditional transfer/re-upload tests.
