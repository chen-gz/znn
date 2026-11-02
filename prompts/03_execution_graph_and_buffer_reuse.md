# Milestone 3 & P0 前置规划: 编译期后端架构自适应、执行图优化与缓冲区复用 (Comptime HAL, Execution Graph & Buffer Reuse)

> **Status**: **Completed & Verified** (P0 & Milestone 3 100% Completed, Rebased & Merged to `main` at `dc348622`)  
> **Verification Summary**:
> - P0.1 Comptime Architecture Detection & Toolchain Auto-Probing: Verified (`probeMlxAvailable`, `src/backend/target.zig`).
> - P0.2 GPU/MLX Runtime Defect Resolutions: Verified (toDevice dangling pointer, double-alloc leak, row contiguity).
> - P0.3 Documentation URL Clean-up: Verified (all 25 leaked file URLs replaced, bib:check 302/302).
> - P1.1 Forward Replay & Elimination of copyFromEager: Verified (all 51 operators transitioned to in-place destination *Into).
> - P1.2 Elimination of c_allocator in Backward: Verified (scratch allocator injected, 0 std.heap.c_allocator in backward_nn.zig).
> - P1.3 Static Memory Planner & Buffer Reuse: Verified (FFD memory coloring, >= 85% activation savings, ExecutionGraph replay numerical parity).
> - Test Suites: `zig build check-layers` (89 files, 0 violations), `zig build test` (424/424 passed), `zig build test -Dmlx=false` (passed), `zig build run-book-models` (18/18 models passed).
> **Priority Hierarchy**: **P0 (开箱即用与前置阻断修复)** ➔ **P1 (M3 静态执行图与内存复用)** ➔ **P2 (平台中立调度与显存切片)**  
> **Target Packages**:  
> - `build.zig` (构建工具链自动探测与链接判定)  
> - `src/backend/` (Layer 0: `target.zig`, `buffer.zig`, `device.zig`, `cblas.zig`, `mlx.zig`)  
> - `src/tensor/` (Layer 1: `core.zig`, `destination.zig`, `tests.zig`)  
> - `src/autodiff/` (Layer 2: `graph.zig`, `op.zig`, `memory_planner.zig`, `execution_graph.zig`, `backward_*.zig`, `tests.zig`)  
> - `src/optim/` (Layer 4: `optim.zig`)  
> - `chen-gz.github.io` (文档中心：外链与死链清理、34 处 `Graph.init` 接口同步)  
> **Enforcement Policy**: Strict refactoring with zero backward-compatibility baggage (`AGENTS.md` §3)  
> **Dependencies**: Milestone 1 (`src/backend/` HAL 隔离) & Milestone 2 (`DeviceBuffer` 存储统一)

---

## 0. P0 优先级：架构再评估、编译期后端自适应与前置阻断修复 (P0: Comptime Architecture & Pre-M3 Blockers)

在启动大规模计算图执行引擎重构之前，必须首先解决构建与运行时的一致性与稳定性阻断项。这些项被评定为 **P0 优先级**，具有最高的执行先验性：

---

### P0.1: Comptime 编译期后端架构检测与构建工具链自适应探测 (Comptime Architecture Detection & Auto-Probing Toolchains)

#### 1. 痛点与问题根因 (Problem Statement)
- **手工 `-Dmlx=false` 痛点**：当前 `build.zig` 对 Apple MLX 动态库的链接逻辑基于简单的系统架构硬编码判定：
  ```zig
  const supports_mlx = target.result.os.tag == .macos and target.result.cpu.arch == .aarch64;
  const enable_mlx = b.option(bool, "mlx", "Enable Apple MLX acceleration") orelse supports_mlx;
  ```
  如果用户在 macOS Apple Silicon 上进行全新克隆，或者预编译动态库（`deps/mlx-c/lib/libmlxc.dylib`、`libmlx.dylib`）未放入目录，`build.zig` 默认盲目开启 `enable_mlx = true`，导致链接器直接报错崩溃：
  ```text
  error: library not found for -lmlxc
  error: library not found for -lmlx
  ```
  外部调用者或测试脚本被迫必须手动传递 `-Dmlx=false` 才能运行，破坏了“开箱即用”原则。
- **缺失统一的 Comptime 目标架构总线**：在 `src/backend/` 中，系统缺乏统一的 `comptime` 硬件与架构探测总线。不同模块碎片化地读取 `builtin.os.tag` 或 `builtin.cpu.arch`（如 `cblas.zig` 读取 OS，`vec_math.zig` 硬编码向量长度为 8 或 16）。
- **未启用后端的死代码残留**：当某个加速后端被禁用时（如 Linux 无 MLX，或 macOS 无 CUDA），相关代码未能通过 `comptime` 彻底剔除，导致遗留无用符号和伪分支。

#### 2. 构建层物理探测机制 (`build.zig`)
在 `build.zig` 中，增加针对依赖库的物理存在性探测逻辑：
```zig
fn probeMlxAvailable(b: *std.Build, target: std.Build.ResolvedTarget) bool {
    if (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64) return false;
    // 1. 优先探测项目根目录下的预编译库 deps/mlx-c/lib/libmlxc.dylib
    const local_lib = b.path("deps/mlx-c/lib/libmlxc.dylib").getPath(b);
    if (std.fs.accessAbsolute(local_lib, .{})) return true else |_| {}
    // 2. 备选探测系统级 Homebrew 安装路径
    if (std.fs.accessAbsolute("/opt/homebrew/lib/libmlxc.dylib", .{})) return true else |_| {}
    return false;
}
```
`enable_mlx` 默认值变更为物理探测结果：
```zig
const mlx_detected = probeMlxAvailable(b, target);
const enable_mlx = b.option(bool, "mlx", "Enable Apple MLX acceleration (requires Apple Silicon and MLX libraries)") orelse mlx_detected;
```
这样在任何环境下 `zig build` 均能自适应感知硬件与依赖库，库存在则自动启用 GPU 加速，缺失则优雅回退至 CPU Accelerate / SIMD，彻底杜绝链接中断，实现真正的“开箱即编译”。

#### 3. 内核层 Comptime 硬件特征与向量宽度判定 (`src/backend/target.zig`)
引入独立的 `src/backend/target.zig`，由 `comptime` 统一计算目标架构信息：
```zig
//! 编译期目标架构与硬件加速链检测 (Comptime Target Architecture & Acceleration Toolchain)

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

pub const TargetArch = struct {
    pub const arch = builtin.target.cpu.arch;
    pub const os = builtin.target.os.tag;
    pub const is_darwin = os == .macos or os == .ios or os == .tvos or os == .watchos;

    // SIMD 指令集编译期检测
    pub const has_neon = arch.isAARCH64() and std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon);
    pub const has_avx2 = arch.isX86() and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);
    pub const has_avx512 = arch.isX86() and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f);
    pub const has_fma = switch (arch) {
        .x86, .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
        .aarch64, .aarch64_be => has_neon,
        else => false,
    };

    /// 编译期自适应计算最优单精度 SIMD 向量长度
    pub const optimal_vector_len: comptime_int = if (has_avx512)
        16
    else if (has_avx2)
        8
    else if (has_neon)
        4
    else
        4;

    pub const SimdVector = @Vector(optimal_vector_len, f32);
};

/// 编译期启用的硬件加速工具链
pub const ActiveToolchains = struct {
    pub const has_accelerate: bool = TargetArch.is_darwin;
    pub const has_mlx: bool = build_options.enable_mlx and TargetArch.is_darwin and TargetArch.arch == .aarch64;
    pub const has_cuda: bool = (builtin.target.os.tag == .linux or builtin.target.os.tag == .windows);
};
```

#### 4. 编译期零开销工具链分发与死代码修剪 (Comptime Toolchain Dispatch)
在 `src/backend/` 各模块中，通过 `comptime if` 确保未启用的后端在编译期被彻底剪除：
- 当 `ActiveToolchains.has_mlx == false` 时，`DeviceArray` 中的 `mlx: mlx.Array` 与 MLX 动态 C 链接在编译期被替换为不可达/零大小结构体，零二进制膨胀；
- 矩阵乘法分发统一通过 `comptime` 内联决定是链接 Apple Accelerate `cblas_sgemm`、系统 OpenBLAS 还是采用自适应向量宽度的纯 Zig 块状 GEMM。

---

### P0.2: 消除 GPU/MLX 模式下的 3 大运行时阻断缺陷 (Fix 3 Critical GPU/MLX Runtime Defects)

在 Milestone 2 完成后，默认开启 MLX 的全量测试仍有 3 处缺陷阻碍了 GPU CI 的通过：

#### 缺陷 1: `src/tensor/tests.zig:2718` 的 `write-after-free` 悬垂指针
- **故障现象**：在开启 MLX 时执行 `zig build test`，在 `tests.zig:2718` 处触发 `SIGABRT` panic 崩溃。
- **根本原因**：`Tensor.toDevice()` 将 CPU 物理缓冲区释放并替换为 GPU 显存，但未清空外部持有的 CPU `data: []f32` 裸切片；测试用例执行 `t.data[0] = 99;` 直接向已释放内存写入，被安全分配器捕获。
- **修复方案**：
  1. `toDevice` 在释放原有 CPU 缓冲后，将 `self.data = &.{}` 立即置空；
  2. 废除直接读写 `t.data`，全面改用受控访问器 `t.asHostSliceMut()`；当张量驻留在 GPU 时，`asHostSliceMut()` 抛出 `error.DeviceMismatch`，彻底杜绝静默写入或悬垂指针。

#### 缺陷 2: `src/autodiff/tests.zig:1547` 的测试张量设备错配与 `storeMlxOutput` 宿主回退隐患
- **故障现象**：在 GPU 计算图测试中，部分测试辅助函数（如 `tensorND`）默认开辟 CPU 张量，而后续算子在 GPU 上调度，引发 `DeviceMismatch` 错误；在 `storeMlxOutput` 中，当 GPU 计算图产生非预期输出时，错误处理回退不完整。
- **根本原因**：测试脚手架缺乏显式设备感知（Device-Aware Test Harness），未能向测试张量传递当前执行图的目标设备。
- **修复方案**：更新测试套件中的工厂辅助函数，使 `tensorNDDevice(..., device)` 显式继承图设备；`storeMlxOutput` 强化强类型错误拦截与传播。

#### 缺陷 3: `src/backend/buffer.zig:682` 的 MLX 矩阵转置非连续步长与行主序校验不一致
- **故障现象**：对 MLX 数组执行转置（`transposeAxes`）后，`buffer.zig` 中的 `f32ByteSize` 或布局校验函数调用 `arr.isContiguousRowMajor()` 返回 `false`，导致后续操作抛出 `error.NonContiguousLayout`。
- **根本原因**：转置操作在 MLX 中为零拷贝视图（Strided View），其步长不再是标准行主序连续。在需要物理连续缓冲区的算子中，直接对转置视图进行行主序断言会导致假阳性错误。
- **修复方案**：在 `DeviceArray` 与 `DeviceBuffer` 的布局校验中，区分逻辑连续与物理连续；对于转置视图，若算子要求物理连续，显式调用 MLX 的连续化操作（如 `contiguous()` / `clone()`）进行物化，否则允许非连续步长视图正常传递。

---

### P0.3: 专著与文档中心外链与死链清障 (Scrub Leaked Local URLs & Broken Links in `chen-gz.github.io`)

#### 1. 清理 18 处泄漏本地绝对路径的 Markdown 章节
在 `chen-gz.github.io` 仓库中，经静态审查发现共有 9 篇专著章节（中英文共计 18 处）泄漏了开发机本地文件协议路径（`file:///Users/guangzong/Documents/znn/...`）：
- `src/content/ml/book/chapter 4-recurrent-neural-networks/` 下 4 个章节（6 处）；
- `src/content/ml/book/chapter 7-advanced-systems-and-acceleration/` 下 1 个章节（2 处）；
- `src/content/ml/book/chapter 8-deepseek-architecture-and-systems/` 下 3 个章节（8 处）；
- `src/content/ml/book/chapter 9-generative-models/` 下 1 个章节（2 处）。
**修复规范**：将其全部统一替换为合法的 GitHub 仓库公共链接或规范相对引用（如 `https://github.com/chen-gz/znn` 与 `https://github.com/chen-gz/znn/blob/main/src/...`）。

#### 2. 修复 `doc/README.md:34` 指向丢失文档的链接
`doc/README.md:34` 引用了 `visualization-model-edge-design.md`，该文档实际存在于 `znn/doc/` 仓库中。将其更新为对齐文档或在 `chen-gz.github.io` 中补充完整说明。

---

## 1. P1 优先级：Milestone 3 目标与重构使命 (Executive Summary & Mission)

在大语言模型（Large Language Model, LLM，如 GPT、Gemma 4）的训练与推理执行中，内存吞吐与执行抖动是系统的核心瓶颈。目前 ZNN 在计算图执行与内存管理上存在四大系统性痛点：

1. **前向重演内存严重抖动 (Forward Memory Thrashing)**：  
   `src/autodiff/op.zig` 中的 `forward` 方法在重放执行时，全量依赖 `copyFromEager`（全局共计 **51 处调用**，覆盖 30+ 种算子类型）。每次前向计算都在堆上临时开辟全新的 `Tensor`，执行 eager 运算后再通过 `@memcpy` 拷回预分配的目标节点，计算结束后立即销毁临时张量。这造成了大量无意义的堆内存分配、释放与内存拷贝开销。
2. **反向求导原生分配器失控 (Uncontrolled `c_allocator` in Backward)**：  
   `src/autodiff/backward_nn.zig` 中仍存在 **8 处硬编码的 `std.heap.c_allocator`**（涉及 `conv2d`、`maxPool2d`、`avgPool2d`、`layerNorm`、`batchNorm2d`、`convTranspose2d` 的临时重排与展开缓冲区）。这些临时分配脱离了计算图的生命周期管理，引发隐式内存碎片化并破坏了可控内存分配协议。
3. **中间激活未进行生命周期复用 (Unbounded Activation Footprint)**：  
   当前的 `Graph` 为每一个中间算子输出独立开辟物理内存，中间激活的总显存/内存开销等于整个网络前向所有节点体积的代数和（$\sum_{i} \text{size}(T_i)$）。在静态执行模式（`deferred`）下，中间张量完全可以通过**生命周期区间分析**（Liveness Interval Analysis）进行内存着色与偏移复用，将峰值内存降至**同时存活集合的最大上界**。
4. **张量状态双轨制与梯度非一等公民 (Dual-State Hazards & Raw Slice Gradients)**：  
   `Tensor` 结构体中仍挂载非所有权切片 `data: []f32` 与 `grad: []f32`。梯度无法在图形处理器（GPU）上物理驻留，迫使 GPU 训练在反向前将所有数据同步回 CPU。

### 核心使命 (Core Mission)
本里程碑（Milestone 3）将构建自包含、确定性且零动态堆分配的执行图与自动微分体系：
- **静态执行图 (Static Execution Graph)**：彻底消除 `op.forward` 中全部 51 处 `copyFromEager`，全面接入 Layer 1 的 `*Into` 目标写入算子，实现前向重放**零堆分配**（Zero Heap Allocation）；
- **生命周期复用规划器 (Static Memory Planner & Buffer Reuse)**：基于计算图拓扑排序与严格的反向活跃区间分析，实现紧凑的内存着色算法，将中间张量绑定至统一的物理共享缓冲区（`shared_arena`），在多层前向推理时降低激活显存峰值达 90%–95%，并在训练时消除前向内部碎片；
- **反向工作区分配器注入 (Backward Scratch Allocator)**：为 `Op.backward` 全面注入 `scratch_allocator: std.mem.Allocator`，彻底根除 `backward_nn.zig` 中全部 8 处 `std.heap.c_allocator`；
- **张量数据与梯度双层重构 (`Tensor` Data & Grad Refactoring)**：将梯度彻底重构为一等公民 `grad: ?*Tensor = null`，支持原生 GPU 驻留与懒分配；废除不安全的 `.data` 裸切片直写，改用强类型校验的 `asHostSlice()` / `asHostSliceMut()`；提供兼顾融合循环的原位切片累加器 `hostGradSliceMut` 与图节点级累加原语 `accumulateGrad`；
- **专著与教程 34 处历史接口同步升级**：同步迁移 `chen-gz.github.io` 中所有 `Graph.init(allocator)` 签名。

---

## 2. 深度架构规格说明 (Architectural Specifications)

### 2.1 张量数据与梯度重构 (`Tensor` Data & Grad Transition)

#### 体系重构方案
1. **一等公民梯度 (`grad: ?*Tensor = null`)**：
   - 梯度不再是生硬的 CPU 裸切片 `[]f32`，而是完整的 `Tensor` 实例，由自身的 `storage: backend.DeviceBuffer` 承载；
   - 梯度原生支持位于 CPU、Apple Metal 或 NVIDIA CUDA；
   - **懒分配机制 (Lazy Gradient Allocation)**：在构建计算图时，即便节点标记 `requires_grad = true`，也不预先分配梯度显存；仅在反向传播实际传导到该节点时，才按需创建或从静态工作区复用。
2. **受控宿主切片访问器 (`asHostSlice` / `asHostSliceMut`)**：
   - 严禁外部代码直接解引用裸切片；所有 CPU 访问必须通过 `asHostSlice()`（只读）或 `asHostSliceMut()`（可写）；
   - 访问器在返回切片前严格执行三项断言校验：
     1. `storage.device.isCpu()`：必须位于 CPU，否则抛出 `error.DeviceMismatch`；
     2. `storage.typed_owner == null`：显存数组不可处于未同步状态；
     3. 步长与形状为严格行主序连续（`isContiguousStrides`）。
   - **性能与提升规范 (Hoisting Mandate)**：所有反向核函数必须遵循提升规范，在函数 Prologue 执行一次切片解构后直接传给循环，确保在 ReleaseFast 下性能无损。
3. **双层梯度累加接口设计 (Dual-Level Gradient Interfaces)**：
   - **高层图节点级累加器 (`accumulateGrad`)**：用于多分支图节点梯度汇聚；
   - **底层原位切片获取器 (`getOrAllocGrad` 与 `hostGradSliceMut`)**：用于反向算子内核直接在宿主内存中累加。

   ```zig
   /// 1. 获取或按需懒分配梯度张量 (零初始化，保持 Tensor 实体)
   pub fn getOrAllocGrad(self: *Tensor, allocator: std.mem.Allocator) !*Tensor;

   /// 2. 获取用于 CPU 反向求导的可变切片 (提升到核函数 Prologue 执行)
   pub fn hostGradSliceMut(self: *Tensor, allocator: std.mem.Allocator) ![]f32;
   ```

---

### 2.2 前向重演引擎与直写目标算子 (Forward Replay Engine)

#### 废除 `copyFromEager`
在 `src/autodiff/op.zig` 中，将 51 处 `copyFromEager` 全面替换为直接向 `self.outputs[0]` 写入的目标算子：

```zig
pub fn forward(self: *Op, scratch_allocator: std.mem.Allocator) !void {
    _ = scratch_allocator;
    const out = self.outputs[0];
    switch (self.op_type) {
        .matmul => try tensor.matmulInto(out, self.inputs[0], self.inputs[1]),
        .add => try tensor.addInto(out, self.inputs[0], self.inputs[1]),
        .mul => try tensor.mulInto(out, self.inputs[0], self.inputs[1]),
        .sub => try tensor.subInto(out, self.inputs[0], self.inputs[1]),
        .div => try tensor.divInto(out, self.inputs[0], self.inputs[1]),
        .neg => try tensor.negInto(out, self.inputs[0]),
        .exp => try tensor.expInto(out, self.inputs[0]),
        .log => try tensor.logInto(out, self.inputs[0]),
        .sqrt => try tensor.sqrtInto(out, self.inputs[0]),
        .tanh => try tensor.tanhInto(out, self.inputs[0]),
        .relu => try tensor.reluInto(out, self.inputs[0]),
        .gelu => try tensor.geluInto(out, self.inputs[0]),
        .silu => try tensor.siluInto(out, self.inputs[0]),
        .sigmoid => try tensor.sigmoidInto(out, self.inputs[0]),
        .softmax => try tensor.softmaxInto(out, self.inputs[0], self.context.softmax.dim),
        .layer_norm => try tensor.layerNormInto(out, self.inputs[0], self.inputs[1], self.inputs[2], self.context.layer_norm.eps),
        .rms_norm => try tensor.rmsNormInto(out, self.inputs[0], if (self.inputs.len > 1) self.inputs[1] else null, self.context.rms_norm.eps),
        // ...其余全部直写目标
    }
}
```

#### 目标写入算子库扩展清单 (`src/tensor/destination.zig`)
需在 Layer 1 中扩充的目标算子清单：
- **基础逐元素数学算子**：`subInto`, `divInto`, `negInto`, `expInto`, `logInto`, `sqrtInto`, `powInto`, `clampInto`, `absInto`；
- **激活函数**：`reluInto`, `geluInto`, `siluInto`, `sigmoidInto`, `tanhInto`, `leakyReluInto`；
- **归一化与归约算子**：`softmaxInto`, `layerNormInto`, `rmsNormInto`, `sumDimInto`, `meanDimInto`, `maxDimInto`；
- **形状操作**：`concatInto`, `splitInto`, `tileInto`, `gatherInto`。

---

### 2.3 静态内存规划与缓冲区复用算法 (Static Memory Planning & Buffer Reuse)

在静态编译模式（`deferred`）下，引入 `src/autodiff/memory_planner.zig`：
1. **拓扑排序与时间步划分 (Topological Ordering)**：确定每个算子的严格执行时间步 $t \in [0, T-1]$。
2. **反向活跃区间分析 (Corrected Liveness Interval Analysis)**：
   - 产生时间：生成张量的算子步 $t_{\text{create}}$；
   - 销毁时间：最后一次前向读取步 $t_{\text{last\_fwd}}$；
   - **反向生命周期扩展**：若张量被标记为反向求导所需（`saved_for_backward`），其活跃生命期必须延长至对应的反向求导算子步 $t_{\text{bwd}}$。
3. **内存着色与峰值地址分配算法 (1D Memory Coloring & Peak Sizing)**：
   - 采用 First-Fit Decreasing 内存区间着色算法；
   - 为不重叠的张量分配相同的偏移量，使它们在物理共享内存池（`shared_arena: backend.DeviceBuffer`）中共用内存；
   - 强制执行 64 字节对齐。
4. **显存节约指标**：
   - 多层 Transformer / CNN 推理场景下，前向激活内存峰值可降低 **90%–95%**；
   - 训练模式下消除前向内部碎片。

---

### 2.4 反向工作区临时分配器 (Backward Scratch Allocator)

#### 消除 `c_allocator`
1. 为反向求导接口增加显式分配器参数：
   ```zig
   pub fn backward(self: *Op, scratch_allocator: std.mem.Allocator) !void
   ```
2. 在 `Graph.runBackward` 调度循环中：
   从计算图自带的 `self.arenaAllocator()` 中获取线性 Arena 分配器作为 `scratch_allocator` 传入；
3. 反向传播期间的所有临时展开矩阵、瓦片缓冲均从 Arena 中分配；反向结束后随计算图一并重置或释放，彻底杜绝操作系统级内存碎片并实现零泄漏保证。

---

## 3. P2 优先级：平台中立调度与显存切片抽象 (Platform-Neutral Scheduling & Slicing)

### 3.1 平台中立的 `GraphOptions`
遵循 `AGENTS.md` §4 与 §6：
- 彻底剔除 `GraphOptions` 中暴露的特定硬件句柄（如 `mlx_stream: ?backend.mlx.Stream = null`）；
- 硬件流封装在 `backend.Device` 与 `backend.DeviceBuffer` 内部处理；
- 结构体支持匿名字面量 `.{}` 并暴露 `defaultOptions()`：

```zig
pub const GraphOptions = struct {
    io: ?std.Io = null,                     // 输入输出并行执行器接口；null 表示单线程串行
    enable_grad: bool = true,               // 是否使能自动求导与计算图记录
    mode: ExecutionMode = .eager,           // 执行模式：.eager (即时) 或 .deferred (静态规划重放)
    max_tasks: ?usize = null,               // 最大并发工作任务数
    device: backend.Device = .cpu,          // 目标计算设备：.cpu (默认) 或 .metal / .cuda
    memory_reuse: bool = true,              // 静态模式下是否开启内存复用优化

    pub const default: GraphOptions = .{};
    pub fn defaultOptions() GraphOptions {
        return .{};
    }
};
```

### 3.2 `DeviceBuffer` 的零拷贝子显存切片 (Sub-Buffer Slicing)
为使静态内存规划器能够无缝下发显存偏移给 GPU 后端，`DeviceBuffer` 需支持原生切片视图（Sub-Buffer Views）：
- 在 CPU 上：直接切片 CPU 物理内存；
- 在 Metal / MLX 上：支持从主显存池开辟指定 byte offset 的子视图，避免小张量频繁向驱动申请内存。

---

## 4. 具体 Zig 类型定义与接口契约 (Concrete Zig Types & Signatures)

### 4.1 内存规划器核心类型 (`src/autodiff/memory_planner.zig`)

```zig
const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const backend = @import("../backend.zig");
const types = @import("types.zig");
const Op = types.Op;

/// 描述张量在拓扑执行序列中的活跃时间区间
pub const LivenessInterval = struct {
    tensor_id: usize,
    tensor: *Tensor,
    start_op: usize,             // 生成该张量的算子索引 (对于外部输入/参数为 0)
    end_op: usize,               // 最后读取该张量的算子索引 (已包含反向引用扩展)
    byte_size: usize,            // 64 字节对齐后的物理内存需求
    is_persistent: bool,         // 是否为模型参数/权重或终端图输出 (不可复用)
    is_saved_for_backward: bool, // 是否被反向求导算子引用
    cluster_id: ?usize = null,   // 视图张量所属的存储别名等价簇编号

    pub fn overlaps(self: LivenessInterval, other: LivenessInterval) bool {
        if (self.is_persistent or other.is_persistent) return true;
        return !(self.end_op < other.start_op or other.end_op < self.start_op);
    }
};

/// 静态物理缓冲区内部的张量子切片分配信息
pub const BufferAssignment = struct {
    tensor: *Tensor,
    offset_bytes: usize,      // 在 shared_arena 中的起始字节偏移量
    size_bytes: usize,        // 实际物理尺寸 (已向上对齐至 64 字节)
    alignment: usize = 64,    // 强制 64 字节对齐
};

/// 整个静态计算图的显存复用规划方案
pub const MemoryPlan = struct {
    allocator: std.mem.Allocator,
    total_arena_bytes: usize, // 共享物理缓冲区必须分配的峰值字节数 (max(offset + size))
    baseline_bytes: usize,    // 若不复用，所有中间张量总字节数代数和
    assignments: []BufferAssignment,
    intervals: []LivenessInterval,

    pub fn deinit(self: *MemoryPlan) void {
        self.allocator.free(self.assignments);
        self.allocator.free(self.intervals);
    }

    pub fn savingsRatio(self: MemoryPlan) f32 {
        if (self.baseline_bytes == 0) return 0.0;
        if (self.total_arena_bytes >= self.baseline_bytes) return 0.0;
        const saved = self.baseline_bytes - self.total_arena_bytes;
        return @as(f32, @floatFromInt(saved)) / @as(f32, @floatFromInt(self.baseline_bytes));
    }
};

pub const MemoryPlanner = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) MemoryPlanner {
        return .{ .allocator = allocator };
    }

    pub fn buildPlan(
        self: *MemoryPlanner,
        ops: []const *Op,
        tensors: []const *Tensor,
        graph_outputs: []const *Tensor,
    ) !MemoryPlan;
};
```

### 4.2 静态执行图与重放引擎 (`src/autodiff/execution_graph.zig`)

```zig
const std = @import("std");
const backend = @import("../backend.zig");
const Graph = @import("graph.zig").Graph;
const Op = @import("types.zig").Op;
const MemoryPlan = @import("memory_planner.zig").MemoryPlan;

pub const ExecutionGraph = struct {
    allocator: std.mem.Allocator,
    device: backend.Device,
    shared_arena: backend.DeviceBuffer, // 唯一的全图共享物理连续内存池
    plan: MemoryPlan,
    ordered_ops: []*Op,
    outputs: []*Tensor,
    is_compiled: bool = false,

    pub fn compile(
        graph: *Graph,
        device: backend.Device,
        graph_outputs: []const *Tensor,
        allocator: std.mem.Allocator,
    ) !ExecutionGraph;

    pub fn replay(self: *ExecutionGraph, scratch_allocator: std.mem.Allocator) !void;
    pub fn deinit(self: *ExecutionGraph) void;
};
```

### 4.3 算子接口新签名 (`src/autodiff/op.zig`)

```zig
/// 零堆分配执行算子前向重演 (消灭全部 51 处 copyFromEager)
pub fn forward(self: *Op, scratch_allocator: std.mem.Allocator) !void;

/// 接入统一临时分配器的反向传播求导 (消除 8 处 c_allocator)
pub fn backward(self: *Op, scratch_allocator: std.mem.Allocator) !void;
```

### 4.4 张量核心接口重构 (`src/tensor/core.zig`)

```zig
pub const Tensor = struct {
    storage: backend.DeviceBuffer = backend.DeviceBuffer.emptyCpu(.cpu),
    shape: Shape,
    strides: Shape,
    dtype: DType = .f32,
    requires_grad: bool = false,
    grad: ?*Tensor = null,              // 梯度为一等公民 Tensor，支持 GPU 显存驻留与懒分配
    creator: ?*OpHandle = null,
    is_buffer: bool = false,
    name: ?[]const u8 = null,

    /// 校验连续性与 CPU 驻留的宿主切片只读访问 (零拷贝，非分配)
    pub fn asHostSlice(self: *const Tensor) TensorError![]const f32;

    /// 校验连续性与 CPU 驻留的宿主切片可变访问 (零拷贝，非分配)
    pub fn asHostSliceMut(self: *Tensor) TensorError![]f32;

    /// 获取或按需懒分配梯度张量 (零初始化，不强制创建临时对象)
    pub fn getOrAllocGrad(self: *Tensor, allocator: std.mem.Allocator) !*Tensor;

    /// 获取用于反向求导原位累加的可变 CPU 切片 (提升至核函数 Prologue 执行)
    pub fn hostGradSliceMut(self: *Tensor, allocator: std.mem.Allocator) ![]f32;

    /// 梯度就地累加：自动处理首次赋值克隆与后续 addInto 累积
    pub fn accumulateGrad(self: *Tensor, g: *const Tensor, allocator: std.mem.Allocator) !void;

    /// 深度克隆张量元数据与底层存储
    pub fn clone(self: *const Tensor, allocator: std.mem.Allocator) !*Tensor;

    /// 原位清零梯度缓冲区 (若 grad != null 则清零，严禁释放显存)
    pub fn zeroGrad(self: *Tensor) void;

    /// 底层物理缓冲区句柄
    pub fn deviceBuffer(self: *const Tensor) *const backend.DeviceBuffer {
        return &self.storage;
    }
};
```

---

## 5. 分阶段实施路线图 (Phased Implementation Roadmap)

```
[Phase 0.1: 构建链自适应与 Comptime 架构总线] ──┐
[Phase 0.2: 消除 3 处 GPU/MLX 运行时缺陷]       ├──► [P0 验收通过: 默认全量测试 0 故障]
[Phase 0.3: 清理文档本地泄漏绝对路径与死链]   ──┘           │
                                                             ▼
[Phase 1.1: 目标算子库扩充与消除 copyFromEager] ──┐
[Phase 1.2: 反向工作区分配器注入消除 c_allocator] ├──► [P1 验收通过: 静态执行图零堆分配]
[Phase 1.3: 静态内存规划与缓冲区复用引擎]       ──┤
[Phase 1.4: Tensor.grad 一等公民与 Prologue 提升] ┘           │
                                                             ▼
[Phase 2.0: 平台中立调度与显存切片抽象] ───────────────► [P2 验收通过: 跨硬件完全对称]
```

### Phase 0: P0 前置阻断修复与自适应探测 (P0 Blockers & HAL Comptime)
- **Phase 0.1: 构建系统自适应与 Comptime 架构总线**：
  1. `build.zig` 落地 `probeMlxAvailable`，无动态库时安全默认回退，杜绝链接错误；
  2. 实现 `src/backend/target.zig`，统一 `comptime` SIMD 向量长度与加速链剪枝。
- **Phase 0.2: 修复 3 处 GPU/MLX 运行时缺陷**：
  1. 修复 `src/tensor/tests.zig:2718` 的悬垂指针（`toDevice` 置空 CPU 切片）；
  2. 修复 `src/autodiff/tests.zig:1547` 的测试张量设备感知与 `storeMlxOutput` 错误传播；
  3. 修复 `src/backend/buffer.zig:682` 的 MLX 矩阵转置非连续步长误报。
- **Phase 0.3: 清理文档本地泄漏绝对路径与死链**：
  1. 替换 `chen-gz.github.io/src/content/ml/book/` 下 9 个章节的 18 处 `file:///` 绝对路径为 GitHub 公共 URL；
  2. 修复 `doc/README.md:34` 指向的模型图可视化规范链接。

### Phase 1: P1 Milestone 3 核心执行图与内存复用 (M3 Core Engine)
- **Phase 1.1: 目标写入算子库扩展与前向迁移**：
  1. 在 `src/tensor/destination.zig` 中补齐全部逐元素与归一化 `*Into` 算子；
  2. 遍历并彻底消除 `src/autodiff/op.zig` 中全部 51 处 `copyFromEager`。
- **Phase 1.2: 反向工作区分配器注入与 `c_allocator` 彻底根除**：
  1. 为 `Op.backward` 全面增加 `scratch_allocator: std.mem.Allocator`；
  2. 消除 `backward_nn.zig` 中 8 处 `std.heap.c_allocator`。
- **Phase 1.3: 静态内存规划与缓冲区复用引擎**：
  1. 编写 `src/autodiff/memory_planner.zig`，实现反向扩展的生命周期区间分析与 1D 内存着色；
  2. 编写 `src/autodiff/execution_graph.zig`，实现静态编译与零分配重放。
- **Phase 1.4: `Tensor.grad` 一等公民重构与双层累加接口**：
  1. 将 `grad` 字段迁移至 `?*Tensor = null`；
  2. 废除直接读写裸切片，改用 `asHostSlice()` / `asHostSliceMut()`；
  3. 实现 `getOrAllocGrad` 与 `hostGradSliceMut`，反向算子落实 Prologue 提升规范。
- **Phase 1.5: 专著与博客 34 处 `Graph.init` 接口全量同步**：
  1. 同步迁移 `chen-gz.github.io` 中的计算图初始化代码。

### Phase 2: P2 平台中立调度与显存切片抽象 (Platform Neutrality)
- 解耦 `GraphOptions` 中的 `mlx_stream`；
- 在 `DeviceBuffer` 中实现子显存切片（Sub-Buffer Views）。

---

## 6. 验收标准与客观验证命令 (Acceptance Criteria & Verification Commands)

### 验收检查清单 (Checklist)
- [x] **P0.1 构建自适应**: 在无任何命令行参数情况下，`zig build` 与 `zig build test` 在任何机器上均能自动探测依赖并开箱即编译通过。
- [x] **P0.2 缺陷彻底消除**: 默认开启 MLX GPU 模式下，`src/tensor/tests.zig:2718` 等 3 处运行时缺陷彻底修复，全量测试 100% 通过。
- [x] **P0.3 文档清洁**: `chen-gz.github.io` 中 18 处 `file:///Users/guangzong/...` 路径完全清零，死链完全修复。
- [x] **P1.1 零堆重放**: `op.forward` 中 51 处 `copyFromEager` 完全清零，前向重放 0 次动态堆分配。
- [x] **P1.2 分配器受控**: `backward_nn.zig` 中 8 处 `c_allocator` 完全清零，反向内存受 Arena 管理。
- [x] **P1.3 显存复用达标**: 静态执行图下，中间激活张量峰值显存降低 $\ge 85\%$（目标 $90\%$–$95\%$）。
- [x] **P1.4 梯度一等公民**: `Tensor.grad` 支持 GPU 显存驻留，切片访问受设备与连续性强类型防护。

### 客观独立验证命令 (Verification Commands)

```bash
# 1. 验证架构包层级严格单向依赖 (Layer 0 -> Layer 7)
zig build check-layers

# 2. 验证默认自动探测全量测试 (含 Comptime 探测、零错误、零内存泄漏)
zig build test

# 3. 验证显式 CPU 模式全量测试 (零错误、零内存泄漏)
zig build test -Dmlx=false

# 4. 验证 18 个经典模型计算图生成与导出
zig build run-book-models

# 5. 验证大模型 4-bit 量化增量解码基准
zig build e4b -Doptimize=ReleaseFast -- -n 24

# 6. 代码洁净度静态检查：验证已彻底杜绝 copyFromEager 与 c_allocator
git grep "copyFromEager" src/autodiff/op.zig          # 输出必须为空
git grep "std.heap.c_allocator" src/autodiff/backward_nn.zig # 输出必须为空
git grep "file:///Users/guangzong" /Users/guangzong/Documents/chen-gz.github.io/src/ # 输出必须为空
```
