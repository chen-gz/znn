# Milestone 15: GPU 加速张量索引与进阶多维布尔掩码赋值体系

## 🎯 任务背景与核心价值 (Context & Objective)

在 Milestone 11 中，ZNN 完整构建了 Layer 1/2 的高级张量索引、轴向收集散布（`gather` / `scatter` / `scatterAdd` / `indexSelect`）与 Autograd 伴随对偶梯度闭环。
在审查与大模型实测过程中，发现以下待改进与高阶演进项（依 `AGENTS.md` §9 规范即刻建档规划）：

1. **MLX / Metal 硬件加速索引 (GPU Stream Acceleration)**：
   当前索引与散布算子主要运行于 CPU。对于超大词表嵌入（Large Vocabulary Embeddings）、MoE 专家路由（DeepSeek-V3 稀疏激活）及密集多模态视觉特征提取，CPU 与 GPU 之间来回同步成为吞吐瓶颈。需要将 `gather` 映射至 MLX 原生 `mlx.core.take_along_axis`，并为 `scatterAdd` 补充 GPU 端原子累加（Metal atomic floating-point accumulation）。
2. **多维布尔掩码散布与赋值 (`maskedScatter` & `Tensor[mask] = src`)**：
   目前已支持提取 1D 压缩张量的 `maskedSelect`，但尚未提供类似 PyTorch `masked_scatter` 的写操作，即把源张量中的连续元素按布尔掩码中的 `true` 位置原位写回目标张量中。
3. **高阶多维切片语法糖 (`Ellipsis` 与 `NewAxis`)**：
   支持类似 NumPy / PyTorch 的 `arr[..., 0]` 省略号与新增虚拟维度，使跨步张量操作更加简洁直观。

---

## 📐 接口规范与设计契约 (Interface Design)

### 1. GPU 算子适配与分发 (`src/backend/mlx/` & `src/autodiff/graph.zig`)

```zig
// 在 backend.mlx 中暴露 take_along_axis 与 scatter 绑定
pub fn gatherMlx(x: *const Tensor, dim: usize, index: *const Tensor, stream: Stream) !*Tensor;
pub fn scatterAddMlx(x: *const Tensor, dim: usize, index: *const Tensor, src: *const Tensor, stream: Stream) !*Tensor;
```

### 2. 多维布尔掩码散布算子 (`src/tensor/indexing.zig`)

```zig
/// 将 src 中的元素依次按 mask 中为 true 的位置写回 dest 中
pub fn maskedScatterInto(dest: *Tensor, mask: *const BoolTensor, src: *const Tensor) !void;
pub fn maskedScatter(self: *const Tensor, mask: *const BoolTensor, src: *const Tensor, allocator: Allocator) !*Tensor;
```

### 3. 多维复合切片说明符 (`src/tensor/types.zig`)

```zig
pub const IndexSpec = union(enum) {
    slice: SliceRange,
    index: usize,
    newaxis,
    ellipsis,
};

pub fn indexMulti(self: *const Tensor, specs: []const IndexSpec, allocator: Allocator) !*Tensor;
```

---

## 📋 交付与验收清单 (Acceptance Criteria)

- [ ] `maskedScatter` / `maskedScatterInto` 算子实现并覆盖 1D~4D 边界单元测试；
- [ ] 探通 MLX GPU `take_along_axis` 内核调用，在 Apple Silicon 上启用无缝 GPU 索引收集；
- [ ] 验证 CPU 与 GPU 结果数值误差严格满足 $10^{-6}$；
- [ ] 运行 `zig build check-layers` 保证 0 依赖违规；
- [ ] 全量 `zig build test` 保持 0 泄漏。
