# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **QR Decomposition, Upper Triangular Back-Substitution & QR Least Squares Solver**:
  - Added `QRMode` (`.reduced` vs `.complete`) and `QROptions` with `defaultOptions()` in `src/tensor/linalg.zig`, conforming to configuration struct guidelines.
  - Added `qrWithOptions` supporting complete QR decomposition ($M > N \implies Q \in \mathbb{R}^{M \times M}, R \in \mathbb{R}^{M \times N}$, forming a complete orthonormal basis $Q Q^T = Q^T Q = I_M$).
  - Implemented upper-triangular back-substitution solver `solveTriangularUpper` for $R x = b$ in $O(K^2)$ time with zero additional allocations.
  - Implemented `qrSolve` for solving overdetermined least-squares systems $A x \approx b$ via $R x = Q^T b$, completely bypassing the condition-squaring normal equations ($A^T A$).
  - Added `solveLeastSquaresQR` in `src/regression.zig` to provide high-stability ordinary least-squares regression with $O(\kappa)$ condition number.
  - Re-exported all new functions and types through `tensor` and `root.zig`.
  - Added comprehensive unit test coverage in `src/tensor/tests.zig` and `src/regression.zig`.

- **Comprehensive Multi-Domain Dataset Suite (Tiny ImageNet, WikiText-103 & Multimodal Vision-Language)**:
  - Upgraded `ImageDataset` in `src/dataset.zig` to support arbitrary channel depth (`channels: u32 = 1` for grayscale, `channels = 3` for RGB) and added `sampleSize() = channels * rows * cols`.
  - Extended `loadImages` to parse both 3D IDX (`0x00000803`) and 4D IDX (`0x00000804` NCHW) tensor files.
  - Implemented Tiny ImageNet packed binary parser `loadTinyImageNetBin` (3-channel $64 \times 64$ RGB, 200 classes) and instant in-memory generator `createSyntheticTinyImageNet`.
  - Added `scripts/pack_tiny_imagenet.py` to pack extracted Tiny ImageNet JPEG folders into high-speed binary blobs.
  - Expanded `tools/download_data.zig` with automated download targets for `tiny_imagenet` and `wikitext103`.
  - Implemented `MultimodalDataset`, `MultimodalPair`, and `MultimodalDataLoader` with batching, token padding, and shuffling in `src/dataset.zig`.
  - Added `createSyntheticMultimodalDataset` producing paired high-resolution RGB images and BPE-tokenized descriptive captions.
  - Created end-to-end vision-language benchmark in `examples/multimodal.zig` (`zig build run-multimodal`) integrating ConvNeXt-atto visual feature extraction, cross-modal linear projection, and causal Transformer autoregressive decoder, processing 21+ pairs/sec with zero memory leaks.
  - Upgraded `examples/convnext.zig` and `examples/resnet.zig` to support `--dataset tiny_imagenet` (3-channel $64 \times 64$ input, 200 output classes).

- **Modern ConvNeXt Pure Convolutional Vision Architecture Suite (Liu et al. CVPR 2022)**:
  - Added support for grouped convolution and depthwise convolution (`groups: usize = 1` in `ConvOptions`, `conv2d`, `conv2dInto`, `conv1d`, and `nn.Conv2D`).
  - Implemented exact grouped and depthwise convolution backward automatic differentiation in `src/autodiff/backward_nn.zig`, fully validated with numerical gradient checking (`gradcheck`).
  - Added `LayerNorm2d` (Channels-First 2D Layer Normalization) in `src/nn/normalization.zig` with `Options` and `initDefault`.
  - Implemented `ConvNeXtBlock` in `src/models/convnext.zig` featuring 7x7 Depthwise Conv, channels-last LayerNorm, 1x1 pointwise expansion (4x), GELU, 1x1 pointwise projection, learnable LayerScale ($\gamma$), and residual connection.
  - Implemented `DownsampleLayer` (LayerNorm2d + 2x2 Conv stride=2) and complete `ConvNeXt` model with stem patchify, multi-stage block sequences, global adaptive average pooling, and linear classification head.
  - Added constructors and presets for canonical ConvNeXt specifications:
    - ImageNet series: `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`.
    - Edge & Benchmark series: `convnext_femto`, `convnext_atto`, `convnext_pico`.
  - Added executable training and benchmark application in `examples/convnext.zig` (`zig build run-convnext`), achieving 186.97 img/s training throughput, 635.49 img/s inference throughput, and 75.10% test accuracy on Fashion-MNIST within 2 epochs.

- **Complete ResNet Paper Architecture Suite (He et al. 2015)**:
  - Implemented both core residual block structures:
    - `BasicBlock` (2-layer 3x3 Conv + BN + ReLU, expansion = 1) for ResNet-18/34 and CIFAR-10 series.
    - `Bottleneck` (3-layer 1x1 -> 3x3 -> 1x1 Conv + BN + ReLU, expansion = 4) with channel compression and restoration for ResNet-50/101/152.
  - Implemented generic modular `GenericResNet(comptime Block: type)` supporting `BasicBlock` and `Bottleneck` with stem convolution, customizable stages via `nn.ModuleList(Block)`, adaptive average pooling, and linear classification head.
  - Added constructors and presets for all canonical architectures from the original paper:
    - ImageNet series (Table 1): `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`.
    - CIFAR-10 series (§4.2): `resnet20`, `resnet32`, `resnet44`, `resnet56`, `resnet110`.
    - Lightweight demo: `resnet9`.
  - Added full training, evaluation, and performance benchmark application in `examples/resnet.zig` (`zig build run-resnet`), achieving 876+ img/s training throughput, 3,400+ img/s inference throughput, and 85.05% test accuracy on Fashion-MNIST.
  - Upgraded `BatchNorm2d.init` with standard `Options` and `initDefault`, conforming to configuration conventions.


---

## Releases

Detailed release notes for past versions are archived in individual files under [`changelog/`](changelog/):

| Version | Release Date | Summary | Release Note File |
| :--- | :--- | :--- | :--- |
| **0.4.0** | 2026-10-31 | Unified tensor storage on `DeviceBuffer`, zero-allocation destination ops, static memory planner & execution graph replay, PyTorch-style `nn.Module` protocol, modular `GPT` and `Gemma 4` models | [`changelog/0.4.0.md`](changelog/0.4.0.md) |
| **0.3.0** | 2026-10-16 | DeepSeek MLA multi-head latent attention, SwiGLU, rotary position embedding (RoPE), Q4 quantization engine, Householder QR and symmetric eigenvalue decomposition | [`changelog/0.3.0.md`](changelog/0.3.0.md) |
| **0.2.7** | 2026-09-27 | Inspector multi-level breadcrumb navigation and back history navigation in web visualization | [`changelog/0.2.7.md`](changelog/0.2.7.md) |
| **0.2.6** | 2026-09-27 | DAG architecture flow visualization for submodules and internal hierarchy in model inspector | [`changelog/0.2.6.md`](changelog/0.2.6.md) |
| **0.2.5** | 2026-09-27 | DAG architecture flow submodule drill-down, parameters inspection drawers, multi-tab modal window | [`changelog/0.2.5.md`](changelog/0.2.5.md) |
| **0.2.4** | 2026-09-26 | Visualizer HTML standalone template extraction, release rules and Git tag guidelines | [`changelog/0.2.4.md`](changelog/0.2.4.md) |
| **0.2.3** | 2026-09-26 | Full decoupling of visualization report frontend from autodiff engine, dedicated submodules | [`changelog/0.2.3.md`](changelog/0.2.3.md) |
| **0.2.2** | 2026-09-26 | Scaled Dot-Product Attention Core inspection scoping fixes in visualization report | [`changelog/0.2.2.md`](changelog/0.2.2.md) |
| **0.2.1** | 2026-09-26 | Modular source file splitting for autodiff and tensor packages, package layering foundations | [`changelog/0.2.1.md`](changelog/0.2.1.md) |
| **0.2.0** | 2026-09-26 | Interactive model architecture and execution graph HTML visualization engine, hierarchical module tree | [`changelog/0.2.0.md`](changelog/0.2.0.md) |
| **0.1.0** | 2026-09-20 | Initial release: core multi-dimensional tensor library, reverse-mode autodiff, neural network modules, optimizers, t-SNE | [`changelog/0.1.0.md`](changelog/0.1.0.md) |
