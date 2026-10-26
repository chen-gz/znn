# Milestone 1: 严格后端隔离与硬件抽象层 (HAL) 彻底重构

> **Status**: Completed & Merged into `main` (`commit 8f4e87d1` & `df3835ec`)  
> **Target Files**: `src/backend/buffer.zig`, `src/backend/kernels.zig`, `src/backend/device.zig`, `tools/check_layers.zig`, etc.

---

## 1. 任务背景与核心问题 (Motivation)

在重构前，代码库存在严重的硬件抽象泄漏与包层级倒挂：
1. **隐式绕过层级检查**：`src/backend/ops.zig` 拥有 566 行代码，大量使用 `anytype` 接收上层 `*Tensor`，通过编译期反射字段（如 `@hasField(..., "device_array")`）在 Layer 0 操作 Layer 1 的张量结构；
2. **领域模块归属倒错**：高层逻辑（如 Safetensors 格式解析、词元采样 Sampler、Transformer 块组合、注意力掩码）被错误地塞入底层 `src/backend/`；
3. **高层穿透依赖**：模型和网络层直接导入 `backend` 内部的私有子文件（如 `@import("../backend/rope.zig")`）。

---

## 2. 交付与重构清单 (Delivered Changes)

1. **彻底清除 Duck-Typing 鸭子类型**：
   - 永久删除 `src/backend/ops.zig`；
   - 彻底清理 `src/backend/offload.zig` 中的所有泛型反射与隐式解包；
   - 统一由 `src/autodiff/ops.zig` (Layer 2) 承载具名 `*const Tensor` / `*Tensor` 强类型调度。

2. **构建纯物理设备缓冲区 (`src/backend/buffer.zig`)**：
   - 提供 64 字节对齐的 `DeviceBuffer` 与 `BufferHandle`；
   - 严格管理 CPU 宿主内存、CUDA 显存指针与 MLX 设备内存生命周期；
   - 严禁在 Layer 0 触碰形状、步长、切片或 autograd 上下文。

3. **构建无状态底层计算算子引擎 (`src/backend/kernels.zig`)**：
   - 新增统一 C-ABI 算子执行契约：`gemm`, `rmsNorm`, `rope`, `softmax`, `elementwise`, `gelu`, `softcap`, `embedding`；
   - 跨 CPU SIMD、NVIDIA CUDA、Apple Metal/MLX 统一调度。

4. **6 大领域模块精准重定向**：
   - `logits_processor.zig` 与 `sampling.zig` $\rightarrow$ `src/engine/` (Layer 5)
   - `attention_mask.zig` (重命名为 `attention.zig`) 与 `safetensors.zig` $\rightarrow$ `src/tensor/` (Layer 1)
   - `rope.zig` 合并收敛至 `src/tensor/rope.zig` (Layer 1)
   - `mlx_blocks.zig` 升级为 `src/nn/transformer_blocks.zig` (Layer 3)

5. **强化静态层级检查器 (`tools/check_layers.zig`)**：
   - 静态扫描 AST 禁令：Layer 0 禁止出现任何 `anytype` 鸭子类型代码；
   - 跨包导入禁令：上层导入 `backend` 只能指向公共门面 `src/backend.zig`。

---

## 3. 验收验证数据 (Verification Report)

- **`zig build check-layers`**: 84 个核心源文件验证，100% 保持严格单向拓扑，0 违规，0 鸭子类型。
- **`zig build test`**: 387 个单元测试通过，1 个跳过 (CUDA 探针)，0 失败，0 内存泄漏。
- **`zig build run-book-models`**: 18 个经典网络模型 JSON 拓扑导出完全正常。
