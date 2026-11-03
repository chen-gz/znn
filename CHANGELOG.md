# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Zero-Config Adaptive Hardware Acceleration (Milestone 11.5)**:
  - `Device.defaultOptions()` and `GraphOptions.device` automatically resolve to the optimal available accelerator (`Metal (Apple Silicon) > CUDA > CPU`), enabling zero-boilerplate hardware acceleration out of the box.
  - Apple Silicon Unified Memory Architecture (UMA) zero-copy bridge: `ops.storeMlxOutput` directly borrows contiguous float32 device memory into `out.data` without staging copies.
  - Transparent CPU fallback engine: automatic `ensureHost` guards synchronize host data on demand when non-GPU ops execute in GPU-default graphs, safely dispatching to CPU SIMD / Accelerate kernels and re-promoting to GPU as needed.
  - Standard host data accessors: `tensor.item(allocator) !f32` (PyTorch alignment) and `tensor.toHost(allocator) ![]const f32`.
  - Hardware introspection API: `backend.accelerationSummary(allocator) ![]const u8` formats active device, backend driver, and SIMD hardware details.
  - Automatic `gatherStrided` materialization for non-contiguous views passed to device ops and graph reshapes, preserving exact mathematical autograd back-scattering.

### Fixed
- Fixed `error.BorrowedStorage` and potential dangling host slices during `syncHost` on views and unified memory arrays.
- Fixed `op_mod.forward` for `.Reshape` to guarantee data synchronization whenever destination and source memory addresses differ.
- Added head dimension equivalence check in `offload.maskedAttention` to prevent illegal FlashAttention calls on MLA asymmetric key/value configurations, cleanly falling back to exact CPU attention.

---

## Releases

Detailed release notes for past versions are archived in individual files under [`changelog/`](changelog/):

| Version | Release Date | Summary | Release Note File |
| :--- | :--- | :--- | :--- |
| **0.5.0** | 2026-11-02 | Advanced tensor indexing, reverse strided slicing, scatter/gather operations, and Autograd adjoint dual differentiation (Milestone 11) | [`changelog/0.5.0.md`](changelog/0.5.0.md) |
| **0.4.2** | 2026-11-02 | Modularization of `src/optim/` and `src/dataset/`, test decoupling in `src/tensor/`, Python script hygiene, Milestone 11 indexing foundation | [`changelog/0.4.2.md`](changelog/0.4.2.md) |
| **0.4.1** | 2026-11-01 | QR decomposition & least squares (M10), multi-domain datasets (Tiny ImageNet, WikiText-103, Multimodal) (M8), ConvNeXt suite (M7), full ResNet paper architecture suite | [`changelog/0.4.1.md`](changelog/0.4.1.md) |
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
