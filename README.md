# Zig ML (`znn`) `v0.1.0`

This project is a high-performance, modular deep learning and modern LLM library built entirely from scratch in **Zig 0.16.0**.

It spans the full continuum of machine learning:
1. **Classical Machine Learning**: Closed-form & iterative Linear Regression, Ridge, Lasso, ElasticNet, and K-Fold Cross-Validation.
2. **Computer Vision & Generative Models**: MLPs, 2D CNNs, Transposed Convolutions, and Generative Adversarial Networks (GANs).
3. **Sequential & Recurrent Architectures**: Standard RNN, LSTM, Stacked LSTM, and GRU cells/networks.
4. **Modern Transformer & Generative LLMs**: Transformer blocks, Causal Self-Attention, SwiGLU, Mixture-of-Experts (MoE), Multi-Head Latent Attention (MLA), LoRA fine-tuning, BPE Tokenizer, and DPO alignment.
5. **Autograd Mechanics & Systems Programming**: Dynamic backward automatic differentiation with topological sorting, compile-time (`comptime`) reflection, custom memory arena recycling, and zero-overhead C interoperability (macOS Accelerate CBLAS / Apple Silicon AMX coprocessor / pure Zig SIMD `@Vector` fallback).

---

## 🚀 Key Features

1. **N-Dimensional Tensor Library**:
   * Supports arbitrary-dimensional tensors with native logical `Shape` and contiguous layouts computed via `strides`.
   * Custom multi-dimensional accessors: `get`, `set`, `getGrad`, and `setGrad` with automatic stride mapping.
   * Recursive, nested pretty-printing of N-dimensional structures (similar to NumPy / PyTorch default representations).
   * Fully-featured `reshape` (zero-copy forward) and `transposeND` (physical transposition to contiguous layout) operators with complete backpropagation support.
   * Compile-time statically shaped tensor experiment (`StaticTensor`) ensuring zero runtime overhead and compile-time shape safety.

2. **Dynamic Autodiff Engine**:
   * Automatic backward propagation using depth-first search (DFS) topological sorting to build computation dependencies.
   * Rich operator zoo: `MatMul`, `AddBias`, `Add`, `Sub`, `Mul`, `Div`, `Pow`, `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `SiLU/Swish`, `SoftmaxCrossEntropy`, `MseLoss`, `BceWithLogitsLoss`, `Reshape`, `Transpose`, `LayerNorm`, `RMSNorm`, and masked causal losses.
   * Advanced memory recycling using `ArenaAllocator` to allocate intermediate tensor values and gradients per batch and release them in a single batch-level deallocation.

3. **Decoupled Optimizer Framework**:
   * Fully decoupled optimizer abstractions separated from network layer logic:
     * **`SGDOptimizer`**: Classic SGD with optional velocity-based momentum.
     * **`AdamOptimizer`**: First and second moment estimation with bias corrections.
     * **`AdamWOptimizer`**: Decoupled weight decay update rule with step-level learning rate control (`stepWithLR`).
   * Generic parameter collection via `nn.collectParameters(model, allocator)` utilizing `comptime` reflection.

4. **Modern Transformer, LLM & Vision Modules**:
   * **CV & Convolutions**: `Linear`, `Conv2D`, `ConvTranspose2D`, `AvgPool2D`, `BatchNorm2d`, `Dropout`.
   * **Norms & Activations**: `RMSNorm`, `LayerNorm`, `GELU`, `SiLU` / `Swish`.
   * **LLM Core**: `Embedding`, `KVCache`, `CausalSelfAttention`, `MLALayer` / `MLACache` (Multi-Head Latent Attention), `SwiGLU`, `MoELayer` (Mixture of Experts), and full `TransformerBlock`.
   * **Fine-Tuning & Alignment**: `LoRALinear` for low-rank parameter-efficient fine-tuning, masked SFT loss, and DPO (Direct Preference Optimization) pair loss.
   * **Recurrent Suite**: `RNNCell`, `RNN`, `LSTMCell`, `LSTM`, `StackedLSTM`, `GRUCell`, `GRU`.

5. **Accelerated CPU Math & SIMD Fallback**:
   * Integrates macOS `Accelerate` CBLAS library to execute matrix operations on Apple Silicon's AMX coprocessor.
   * Cross-platform fallback GEMM vectorized with Zig native `@Vector(8, f32)` SIMD instructions for Linux, Windows, and WebAssembly.

6. **100% Pure Zig & Zero Dependencies**:
   * Builds into a completely self-contained binary without Python, PyTorch runtime wheels, or foreign package managers.

---

## 📂 Codebase Directory Structure

* **`src/` (Core Library Modules)**:
  * **[src/tensor.zig](src/tensor.zig)**: N-Dimensional Tensor library. Implements shape, logical strides, memory layout mapping, and vectorized math.
  * **[src/autodiff.zig](src/autodiff.zig)**: Core Automatic Differentiation engine. Dynamic computation `Graph`, `Node`, operator zoo, and DFS topological sorting.
  * **[src/nn.zig](src/nn.zig)** & **`src/nn/`**: Modular Neural Network subsystem & backward-compatible facade:
    * `src/nn/core.zig`: Core containers & operators (`Linear`, `Conv2D`, `ConvTranspose2D`, `Module`, `Sequential`, `collectParameters`).
    * `src/nn/activations.zig`: Elementwise activations (`ReLU`, `GELU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `SiLU`, `Swish`).
    * `src/nn/normalization.zig`: Normalization & pooling layers (`RMSNorm`, `LayerNorm`, `BatchNorm2d`, `Dropout`, `AvgPool2D`).
    * `src/nn/recurrent.zig`: Recurrent neural networks (`RNN`, `LSTM`, `StackedLSTM`, `GRU`).
    * `src/nn/transformer.zig`: Modern LLM & attention architectures (`Embedding`, `KVCache`, `MLP`, `SwiGLU`, `MoELayer`, `CausalSelfAttention`, `MLALayer`, `TransformerBlock`, `TransformerDecoder`, `GPT`, `LoRALinear`, DPO/GRPO losses, Top-P/Top-K samplers).
    * `src/nn/serialization.zig`: Zero-dependency Safetensors persistence (`saveModel`, `loadModel`).
  * **[src/optim.zig](src/optim.zig)**: Decoupled Optimizer Framework (`SGDOptimizer`, `AdamOptimizer`, `AdamWOptimizer`).
  * **[src/engine.zig](src/engine.zig)**: High-level classification & regression training/evaluation loops, step runners, and metric evaluators.
  * **[src/regression.zig](src/regression.zig)**: Classical statistical regression (OLS, Ridge, Lasso, ElasticNet) with closed-form and iterative solvers.
  * **[src/cross_validation.zig](src/cross_validation.zig)**: K-Fold cross-validation splitters, hyperparameter grid search, and evaluation metrics.
  * **[src/dataset.zig](src/dataset.zig)**: Binary parsers for MNIST/Fashion-MNIST IDX format, and Byte-Pair Encoding (`BPETokenizer`).
  * **[src/cblas.zig](src/cblas.zig)**: System CBLAS C-bindings for macOS Accelerate framework and pure Zig `@Vector` SIMD GEMM fallback.
  * **[src/bench.zig](src/bench.zig)**: Performance benchmarking harness, timing statistics, and suites.
  * **[src/root.zig](src/root.zig)**: Module exports, unit tests, and runtime benchmarking/profiling utilities.

* **`examples/` (Executable Binaries & Workflows)**:
  * **[examples/benchmark.zig](examples/benchmark.zig)**: Comprehensive performance benchmarking tool and CLI runner.
  * **[examples/fashion_mnist.zig](examples/fashion_mnist.zig)**: 3-layer MLP on Fashion MNIST dataset.
  * **[examples/cnn.zig](examples/cnn.zig)**: 2D Convolutional Neural Network (Conv2D + MaxPool2D + Linear).
  * **[examples/gan.zig](examples/gan.zig)**: Generative Adversarial Network (Generator + Discriminator) training with BCEWithLogitsLoss.
  * **[examples/train_shakespeare.zig](examples/train_shakespeare.zig)**: Mini-GPT pretraining on TinyShakespeare with live autoregressive text generation.
  * **[examples/llm_training.zig](examples/llm_training.zig)**: End-to-End LLM training pipeline (BPE Tokenizer + SwiGLU + AdamW + SFT + LoRA + DPO).
  * **[examples/transformer_embedding.zig](examples/transformer_embedding.zig)**: Token + positional embedding demo.
  * **[examples/transformer_attention.zig](examples/transformer_attention.zig)**: Causal multi-head self-attention demo.
  * **[examples/transformer_block.zig](examples/transformer_block.zig)**: Transformer decoder block end-to-end forward step.
  * **[examples/transformer_gpt.zig](examples/transformer_gpt.zig)**: Full GPT architecture initialization and forward step.
  * **[examples/linear_regression.zig](examples/linear_regression.zig)**: Linear regression (OLS closed-form vs. iterative autograd GD).
  * **[examples/logistic_regression.zig](examples/logistic_regression.zig)**: Logistic regression with Sigmoid BCE loss.
  * **[examples/ridge_regression.zig](examples/ridge_regression.zig)**: Ridge regression ($L_2$ regularization) with multicollinearity analysis.
  * **[examples/regularized_regression.zig](examples/regularized_regression.zig)**: Comparative benchmark of Ridge, Lasso ($L_1$), and ElasticNet.
  * **[examples/cross_validation.zig](examples/cross_validation.zig)**: 5-Fold cross-validation hyperparameter search.
  * **[examples/comptime_static_tensor.zig](examples/comptime_static_tensor.zig)**: Experimental compile-time statically shaped tensor verification.

---

## 🛠️ Build and Execution

### 1. Download Datasets
```bash
# Vision datasets
zig build download-dataset                  # Fashion MNIST (default)
zig build download-dataset -- mnist         # Classic MNIST

# LLM Text datasets
zig build download-dataset -- tinyshakespeare # TinyShakespeare (~1.1MB pure text)
zig build download-dataset -- wikitext2       # WikiText-2 (train/valid/test ~12MB)
zig build download-dataset -- tinystories     # TinyStories validation slice (~19MB)
zig build download-dataset -- alpaca          # Stanford Alpaca SFT dataset (~22MB JSON)
zig build download-dataset -- all_llm         # Download all 4 LLM datasets
```

### 2. Classical ML, Regression & Cross-Validation
```bash
# Linear regression (OLS vs autograd GD)
zig build run-lr

# Logistic regression (Sigmoid BCE)
zig build run-logr

# Ridge regression (L2 regularization & multicollinearity analysis)
zig build run-ridge

# Regularized regression comparison (Ridge vs Lasso vs ElasticNet)
zig build run-reg

# 5-Fold cross-validation hyperparameter grid search
zig build run-cv
```

### 3. Vision Models & Generative Models
```bash
# Run MLP training on Fashion MNIST
zig build run -Doptimize=ReleaseFast

# Run CNN training on Fashion MNIST
zig build run-cnn

# Run GAN (Generative Adversarial Network) training
zig build run-gan
```

### 4. Transformer Components & LLM Pipeline
```bash
# Run Transformer components
zig build run-emb     # Embedding layer
zig build run-att     # Causal self-attention
zig build run-block   # Transformer block
zig build run-gpt     # GPT decoder model

# Train Mini GPT on TinyShakespeare corpus and generate text live
zig build run-shakespeare

# End-to-End LLM Pipeline demo (BPE + SwiGLU + AdamW + SFT + LoRA + DPO + Top-P)
zig build run-llm
```

### 5. Run Unit Tests & Automated Coverage
```bash
# Run all unit tests across the entire codebase
zig build test

# Run automated code coverage analysis using kcov (outputs summary table & HTML report)
zig build coverage

# Or pass custom arguments (e.g., fail under 90% threshold, or auto-open browser)
zig build coverage -- --fail-under=90
zig build coverage -- --open

# Alternatively, using just:
just coverage
just coverage-open
```

### 6. Run Performance Benchmark Suite
Execute comprehensive performance microbenchmarks and training pipeline benchmarks (measuring GFLOPS, memory bandwidth GB/s, and training step throughput):
```bash
# Run full benchmark suite with ReleaseFast optimization
just bench

# Run specific suite (gemm, ops, activations, layers, models, optimizers, tokenizer)
just bench --suite gemm
just bench --suite models

# Filter benchmarks by name pattern
just bench --filter conv
just bench --filter attention

# Customize iterations and warmup
just bench -i 20 -w 5

# Direct zig build execution
zig build bench -Doptimize=ReleaseFast -- --filter gemm
```

---

## 🔬 Comparison with Modern NumPy (NumPy 2.x)

While `znn` is architected as an **autograd & deep learning micro-framework** (akin to PyTorch or tinygrad), its foundational tensor layer addresses many of the same problems as **NumPy**. 

Here is a high-level comparison between `znn` and modern NumPy 2.x:

| Capability Dimension | Modern NumPy (NumPy 2.x) | ZNN Current Implementation | Parity Status & Roadmap |
| :--- | :--- | :--- | :--- |
| **Dtypes & Multi-Precision** | Rich scalar system (`float16/32/64/128`, `int8~64`, `bool`, extensible DType API) | Hardcoded `f32` (`data: []f32`, `grad: []f32`) | 🔴 Planned: Generic `Tensor(comptime T: type)` |
| **Strides & Slicing Views** | C/Fortran orders, $O(1)$ strided views, step (`arr[::-1]`), fancy & boolean masks | Row-major with strides, scalar `get/set`, `split`, `concat` | 🔴 Planned: Strided slice views & boolean masking |
| **Reductions & Statistics** | `sum`, `mean`, `std`, `var`, `min`, `max` with arbitrary `axis=(...)` & `keepdims` | Single-axis `argmax`/`max`; mean/var private to Norm layers | 🔴 Planned: Generic multi-axis reduction API |
| **Searching & Sorting** | `where(cond, x, y)`, `nonzero`, `sort`, `argsort`, `searchsorted` | Top-K / Top-P sampling, gradient clipping | 🟡 Planned: `where`, `sort`, `nonzero` |
| **Linear Algebra (`linalg`)** | SVD, QR, Cholesky, eigenvalues (`eig/eigh`), matrix inverse, `einsum` | BLAS SGEMM, `batchMatMul`, Gauss-Jordan `solveLinearSystem` | 🟡 Planned: Cholesky, QR, and SVD decomposition |
| **Random (`random`)** | Modern `Generator` (PCG64/Philox), 30+ distributions, shuffle/choice | Uniform `rand`, normal `fillNormal`, Top-K/Top-P | 🟡 Planned: Modular RNG & extended distributions |
| **Autograd & DL Layers** | ❌ None (pure numerical array library) | ✅ **Native dynamic backward graph, LLM layers, AdamW, DPO/GRPO** | 🟢 Core ZNN advantage |
| **Zero Dependencies** | ❌ Requires Python interpreter & C-API | ✅ **Pure Zig, standalone single-binary compilation** | 🟢 Core ZNN advantage |

> 📖 **Detailed Gap Analysis & Technical Roadmap**: For a deep-dive breakdown of every missing NumPy feature, design trade-offs, and actionable implementation phases, see **[plan/NUMPY_GAP_ANALYSIS.md](plan/NUMPY_GAP_ANALYSIS.md)**.

---

## 🗺️ Roadmap & Future Milestones

> 📋 For the full architectural diagnosis and actionable development checklist, see **[plan/TODO.md](plan/TODO.md)**. For the overarching engineering roadmap and index, see **[plan/README.md](plan/README.md)**.

### ✅ Completed Milestones

1. **Decoupled Optimizer Framework**:
   * Extracted parameter update states out of neural layers into [src/optim.zig](src/optim.zig).
   * Implemented `SGDOptimizer` (with Momentum), `AdamOptimizer`, and `AdamWOptimizer` (with decoupled weight decay).
   * Generalized parameter extraction via `nn.collectParameters` using `comptime` reflection.
2. **Training Infrastructure, Schedulers & Gradient Control**:
   * **Learning Rate Schedulers**: Implemented `CosineScheduler` (with warmup & min_lr), `StepLRScheduler`, `LinearWarmupScheduler`, `ExponentialLRScheduler`, and a polymorphic `LRScheduler` union.
   * **Gradient Clipping Suite**: Implemented global L2 norm clipping (`clipGradNorm`), value clipping (`clipGradValue`), and `GradClipConfig` integration into `engine.trainClassificationStepWithClip`.
   * **Optimizer State Checkpointing**: Implemented binary serialization (`saveCheckpoint` / `loadCheckpoint`) with signature headers (`ZNNO`), versioning, and tensor dimension integrity validation for SGD, Adam, and AdamW.
3. **SIMD Vectorization for CPU Fallback Math**:
   * Implemented `@Vector(8, f32)` SIMD kernel in `cblas_sgemm_fallback` inside [src/cblas.zig](src/cblas.zig).
4. **Core Operators & Layer Zoo Expansion**:
   * Implemented `LayerNorm`, `RMSNorm`, `BatchNorm2d`, `Dropout`, `AvgPool2D`, `ConvTranspose2D`.
   * Implemented `MseLoss` and `BceWithLogitsLoss`.
5. **Modern LLM, Recurrent & Classical ML Extensions**:
   * Added `SwiGLU`, `MoELayer` (Mixture of Experts), `MLALayer` (Multi-Head Latent Attention), `LoRALinear`, `BPETokenizer`, and `DPO` loss.
   * Added `RNN`, `LSTM`, `StackedLSTM`, `GRU`, GANs, and statistical regularized regression with Cross-Validation.

---

### 🚀 Next Steps & Future Roadmap

To push `znn` towards a production-ready, high-throughput deep learning and LLM engine in pure Zig, the following high-value areas are targeted across 5 core dimensions:

#### 1. CPU Multi-Core Parallelism & High-Performance Math (High Priority)
* **Multi-Threaded Tiled GEMM**: Replace the current single-threaded SIMD fallback with a cache-friendly, tiled, multithreaded GEMM using Zig's `std.Thread` pool (M-blocking & N-blocking) to saturate all CPU cores on Linux and Windows.
* **Operator-Level Parallelism**: Multi-thread elementwise tensor math, normalization layers (RMSNorm, LayerNorm), and forward/backward convolutional passes.
* **External BLAS Linkage**: Add build script integration options (`-Dblas=openblas` / `-Dblas=mkl`) to link optimized system BLAS libraries on Linux and non-macOS environments.

#### 2. Modern Generative LLM Architecture Alignment
* **RoPE (Rotary Position Embedding)**: Implement native forward and backward operators for rotary position embeddings, aligning `znn` with modern open-source architectures (Llama 3, Qwen 2.5, DeepSeek V2/V3).
* **Online Softmax & FlashAttention Principles**: Replace full $O(S^2)$ attention matrix materialization with chunked online softmax, eliminating quadratic memory bottlenecks for long context sequences.
* **Dynamic & Paged KV Cache**: Upgrade the static KV Cache to dynamically expandable buffers and chunked/paged block allocation to maximize autoregressive decoding throughput.

#### 3. Model Interoperability & Open-Source Ecosystem
* **Standard Hugging Face Safetensors Parser & Dequantization**: Pure Zig parser for official `.safetensors` model weights with automatic conversion/dequantization for BF16/F16 tensors into F32, enabling zero-dependency inference with pretrained models directly from Hugging Face.
* **GGUF Format Reader**: Implement parser for the llama.cpp GGUF format to enable direct loading and execution of quantized model weights.
* **Checkpoint & Graph Export**: Export trained model weights and dynamic computation graphs into standardized formats (Safetensors / ONNX).

#### 4. Graph Execution & Memory Optimization
* **Activation Checkpointing (Gradient Checkpointing)**: Trade minimal compute for memory by freeing intermediate activations during forward passes and recomputing them dynamically during backward passes, enabling $3\times\text{--}5\times$ longer training context lengths.
* **In-Place Operation Reuse**: Static computation graph lifetime analysis to reuse tensor buffers in-place for non-branching activations (e.g. ReLU, Dropout).
* **Compile-Time Static Graph Fusion**: Deeper unification of `StaticTensor` into the dynamic autograd engine for zero-allocation, compile-time verified subgraphs.

#### 5. Data Pipeline Throughput & Developer Experience
* **Multi-Threaded BPE Tokenization**: Parallelize corpus tokenization across CPU worker threads for high-speed preprocessing of multi-gigabyte text datasets.
* **Memory-Mapped (mmap) Streaming Datasets**: Stream pre-tokenized binary datasets via POSIX `mmap`, avoiding memory-exhaustion when handling massive corpora.
* **Terminal Training Dashboard & Telemetry**: Rich terminal interface tracking real-time tokens/second, throughput, estimated TFLOPS, dynamic learning rate schedules, and estimated time of arrival (ETA).
* **GPU & Compute Shader Backend**: Long-term exploration of **WebGPU** (`wgpu-native` / Dawn) or **Vulkan** compute pipelines for cross-platform GPU training and inference directly from pure Zig.

