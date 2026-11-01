# Milestone: 综合测试数据集扩展 (Tiny ImageNet & 多模态大语言模型数据集)

> **Status**: Completed  
> **Target Package**: `src/dataset/` (`src/dataset.zig`), `tools/download_data.zig`, `examples/`  
> **Key Files**: `src/dataset.zig`, `tools/download_data.zig`, `examples/convnext.zig`, `examples/resnet.zig`, `examples/multimodal.zig`

---

## 1. 目标与背景 (Objective)

此前框架仅支持单通道灰度低分辨率分类数据集（Fashion-MNIST / MNIST），缺少高分辨率、多通道彩色以及图文多模态与长程文本建模数据集。

为满足更全面的测试需求，本阶段引入两类核心基准数据集：
1. **Tiny ImageNet (高维度彩色视觉基准)**：
   - 3 通道 RGB，分辨率 $64 \times 64$；
   - 200 个语义类别，100,000 张训练图，10,000 张验证图；
   - 紧凑二进制格式（`tiny-imagenet-bin`）与流式高效数据加载器；
   - 适配 `ResNet` 与 `ConvNeXt` 的端到端训练与性能评估。
2. **多模态与大规模语言模型数据集 (Multimodal & LLM Benchmarks)**：
   - **WikiText-103**：~180MB 真实高质量 Wikipedia 全文，包含 100M+ tokens，是检验自回归语言模型长程语义、上下文注意力和困惑度（Perplexity）的标准学术数据集；
   - **多模态图文对数据集 (Multimodal Vision-Language Dataset)**：实现 `MultimodalDataset` 结构，提供 `(Image [3, H, W], Caption [seq_len])` 配对数据流，为 `Gemma 4` 多模态模型与视觉语言特征提供测试载体；
   - **统一多数据集下载器与自动化准备工具**。

---

## 2. 接口设计契约 (Interface Design Contract)

### 2.1 升级 `ImageDataset` 支持任意多通道
```zig
pub const ImageDataset = struct {
    num_images: u32,
    channels: u32 = 1, // 1 (灰度), 3 (RGB 彩色)
    rows: u32,
    cols: u32,
    data: []f32, // 归一化浮点像素 [num_images * channels * rows * cols]

    pub fn deinit(self: *ImageDataset, allocator: std.mem.Allocator) void;
};
```

### 2.2 `Tiny ImageNet` 加载器
```zig
/// 加载 Tiny ImageNet 紧凑二进制数据集 (每张图: 2 字节 label + 3*64*64=12288 字节 RGB)
pub fn loadTinyImageNetBin(
    allocator: std.mem.Allocator,
    io: std.Io,
    file_path: []const u8,
    max_samples: usize,
) !Dataset;
```

### 2.3 `MultimodalDataset` 图文多模态数据集
```zig
pub const MultimodalPair = struct {
    image: []const f32, // [channels * H * W]
    caption: []const u8,
    tokens: []const u32,
};

pub const MultimodalDataset = struct {
    images: ImageDataset,
    captions: []const []const u8,
    tokens: []const []const u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *MultimodalDataset) void;
    pub fn get(self: MultimodalDataset, idx: usize) MultimodalPair;
};
```

---

## 3. 验收标准与验证 (Acceptance Criteria)

- [x] `ImageDataset` 完整支持 `channels = 3` 与任意多通道，`DataLoader` 打包与遍历无缝自适应；
- [x] `Tiny ImageNet` 紧凑二进制格式解析器与合成数据/真实数据加载测试通过 (`loadTinyImageNetBin`, `createSyntheticTinyImageNet`)；
- [x] `MultimodalDataset` 能够成功打包图文多模态样本并为模型提供训练输入 (`MultimodalDataLoader`)；
- [x] `tools/download_data.zig` 扩充对 `tiny_imagenet` 与 `wikitext103` 的纯 Zig 自动下载与解压；
- [x] `examples/convnext.zig` 与 `examples/resnet.zig` 支持 `--dataset tiny_imagenet`（3 通道 64x64，200 类别）；
- [x] `examples/multimodal.zig` 验证端到端图文多模态自回归训练，达成 21+ pairs/sec 吞吐量且 0 内存泄漏；
- [x] `zig build check-layers` 严格无违规（91 个文件层级依赖单向合法）；
- [x] `zig build test` 全量通过；
- [x] 按照规范提交至分支并即时推送到远端。
