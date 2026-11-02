//! 数据集与数据加载器顶层公共门面 (Datasets & DataLoaders Facade)
//!
//! 聚合导出计算机视觉 (Image/Label/TinyImageNet)、文本自然语言 (BPE Tokenizer/WikiText)
//! 与多模态图文对齐数据流 (MultimodalDataset)，以及通用的批处理 DataLoader。

pub const vision = @import("dataset/vision.zig");
pub const dataloader = @import("dataset/dataloader.zig");
pub const text = @import("dataset/text.zig");
pub const multimodal = @import("dataset/multimodal.zig");

// --- 计算机视觉数据集 (Vision Datasets) ---
pub const ImageDataset = vision.ImageDataset;
pub const LabelDataset = vision.LabelDataset;
pub const loadImages = vision.loadImages;
pub const loadLabels = vision.loadLabels;
pub const Dataset = vision.Dataset;
pub const loadDataset = vision.loadDataset;
pub const loadTinyImageNetBin = vision.loadTinyImageNetBin;
pub const createSyntheticTinyImageNet = vision.createSyntheticTinyImageNet;

// --- 通用数据加载器 (DataLoaders) ---
pub const DataLoader = dataloader.DataLoader;
pub const DataLoaderOptions = dataloader.DataLoaderOptions;

// --- 文本分词与因果流 (BPE Tokenizer & Text Stream) ---
pub const TokenId = text.TokenId;
pub const MergePair = text.MergePair;
pub const MergeRule = text.MergeRule;
pub const BPETokenizer = text.BPETokenizer;
pub const BinaryMmapDataset = text.BinaryMmapDataset;

// --- 多模态图文数据流 (Multimodal Datasets) ---
pub const MultimodalPair = multimodal.MultimodalPair;
pub const MultimodalDataset = multimodal.MultimodalDataset;
pub const MultimodalDataLoaderOptions = multimodal.MultimodalDataLoaderOptions;
pub const MultimodalDataLoader = multimodal.MultimodalDataLoader;
pub const createSyntheticMultimodalDataset = multimodal.createSyntheticMultimodalDataset;

test {
    _ = @import("dataset/tests.zig");
}
