# Milestone: ConvNeXt 现代纯卷积视觉模型架构 (A ConvNet for the 2020s)

> **Status**: Completed  
> **Target Package**: `src/tensor/`, `src/autodiff/`, `src/nn/`, `src/models/`  
> **Key Files**: `src/tensor/conv_pool.zig`, `src/autodiff/graph_nn.zig`, `src/autodiff/backward_nn.zig`, `src/nn/core.zig`, `src/nn/normalization.zig`, `src/models/convnext.zig`, `examples/convnext.zig`

---

## 1. 目标与背景 (Objective)

ConvNeXt (Liu et al. 2022, "A ConvNet for the 2020s", CVPR 2022) 是一类完全由标准卷积构建、吸收了 Vision Transformer (ViT / Swin) 现代设计理念的现代化纯卷积视觉骨干网络。

为使 `znn` 具备现代视觉骨干网络（ConvNeXt 系列），需要打通其所需的基础算子支持与完整模型架构装配：
1. **分组卷积与深度卷积 (Grouped & Depthwise Convolution)**：
   - 在 `ConvOptions` 中增加 `groups: usize = 1` 选项；
   - 在 `src/tensor/conv_pool.zig` 中升级 `conv2d` 与 `conv2dInto`，支持 `groups > 1`（特别是 `groups == in_channels` 的 $7\times 7$ 深度可分离卷积）；
   - 在 `src/autodiff/backward_nn.zig` 中实现严格的分组卷积反向自动微分求导，并通过梯度的数值检查（gradcheck）；
2. **2D 图像通道层归一化 (LayerNorm2d / Channels-First LayerNorm)**：
   - 在 `src/nn/normalization.zig` 中实现 `LayerNorm2d`（或通过转置视图 `[N, H, W, C]` 复用高精度 `LayerNorm`），支持对四维图像张量通道维度的正规化；
3. **ConvNeXt 核心残差块与分阶段规格架构 (`src/models/convnext.zig`)**：
   - 实现 `ConvNeXtBlock`：
     - $7\times 7$ 深度卷积 (`depthwise conv`, `kernel_size=7`, `padding=3`, `groups=dim`)；
     - 层归一化 (`LayerNorm`)；
     - $1\times 1$ 点卷积扩展（$4\times \text{dim}$）；
     - 高斯误差线性单元 (`GELU`) 激活函数；
     - $1\times 1$ 点卷积还原（$\text{dim}$）；
     - 可学习层级缩放系数 (`LayerScale`, $\gamma$ 参数向量，初始值 $10^{-6}$)；
     - 恒等残差连接 (`Residual Add`)；
   - 下采样过渡层 (`Downsample Layer`)：
     - `LayerNorm2d` + $2\times 2$ 步长为 2 的卷积下采样；
   - 论文标准预设规格：
     - `ConvNeXt-Tiny`: `depths: [3, 3, 9, 3]`, `channels: [96, 192, 384, 768]`
     - `ConvNeXt-Small`: `depths: [3, 3, 27, 3]`, `channels: [96, 192, 384, 768]`
     - `ConvNeXt-Base`: `depths: [3, 3, 27, 3]`, `channels: [128, 256, 512, 1024]`
     - `ConvNeXt-Large`: `depths: [3, 3, 27, 3]`, `channels: [192, 384, 768, 1536]`
     - `ConvNeXt-Atto / Femto / Nano` 与 `cifar` 轻量评测规格；
4. **端到端测试与性能基准应用 (`examples/convnext.zig`)**：
   - 提供开箱即用的运行与训练评测程序，验证训练收敛性与吞吐速度。

---

## 2. 接口设计契约 (Interface Design Contract)

### 2.1 `ConvOptions` 升级
```zig
pub const ConvOptions = struct {
    stride: usize = 1,
    padding: usize = 0,
    groups: usize = 1,

    pub const default: ConvOptions = .{};
    pub fn defaultOptions() ConvOptions { return .{}; }
};
```

### 2.2 `ConvNeXtBlock`
```zig
pub const ConvNeXtBlock = struct {
    dwconv: nn.Conv2D,          // 7x7 depthwise conv, groups = dim
    norm: nn.LayerNorm,         // LayerNorm over channel dimension
    pwconv1: nn.Linear,         // dim -> 4 * dim
    pwconv2: nn.Linear,         // 4 * dim -> dim
    gamma: ?*tensor.Tensor,     // LayerScale: [dim]
    dim: usize,

    pub fn init(allocator: std.mem.Allocator, dim: usize, options: Options) !ConvNeXtBlock;
    pub fn forward(self: *const ConvNeXtBlock, graph: *autodiff.Graph, x: *tensor.Tensor) !*tensor.Tensor;
};
```

---

## 3. 验收标准与验证 (Acceptance Criteria)

- [x] `groups > 1` 的 `conv2d` 前向与反向在 `gradcheck_tests.zig` 中 100% 通过数值梯度对比；
- [x] `ConvNeXtBlock` 与 `ConvNeXt` 模型在 `src/models/tests.zig` 中前向与反向测试通过，无内存泄漏；
- [x] `zig build check-layers` 严格无违规（91 个文件分层合规）；
- [x] `zig build test` 全量通过；
- [x] `examples/convnext.zig` 可独立运行并输出速度基准：
  - 训练吞吐：186.97 images/sec (342.31 ms/step, AdamW, 7x7 DWConv)；
  - 评估推理吞吐：635.49 images/sec；
  - 收敛精度：2 个 Epoch 内从 10% 迅速收敛至 75.10% (5/5 采样预测完全正确)；
- [x] 按照规范提交至分支并即时推送到远端。
