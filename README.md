# Zig ML: Modern Deep Learning & LLM Library in Zig

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
  * **[src/nn.zig](src/nn.zig)**: Complete Neural Network Module zoo (Linear, Conv2D, ConvTranspose2D, RMSNorm, LayerNorm, BatchNorm2d, Dropout, SwiGLU, MoE, MLA, LoRA, TransformerBlock, RNN, LSTM, GRU).
  * **[src/optim.zig](src/optim.zig)**: Decoupled Optimizer Framework (`SGDOptimizer`, `AdamOptimizer`, `AdamWOptimizer`).
  * **[src/engine.zig](src/engine.zig)**: High-level classification & regression training/evaluation loops, step runners, and metric evaluators.
  * **[src/regression.zig](src/regression.zig)**: Classical statistical regression (OLS, Ridge, Lasso, ElasticNet) with closed-form and iterative solvers.
  * **[src/cross_validation.zig](src/cross_validation.zig)**: K-Fold cross-validation splitters, hyperparameter grid search, and evaluation metrics.
  * **[src/dataset.zig](src/dataset.zig)**: Binary parsers for MNIST/Fashion-MNIST IDX format, and Byte-Pair Encoding (`BPETokenizer`).
  * **[src/cblas.zig](src/cblas.zig)**: System CBLAS C-bindings for macOS Accelerate framework and pure Zig `@Vector` SIMD GEMM fallback.
  * **[src/root.zig](src/root.zig)**: Module exports, unit tests, and runtime benchmarking/profiling utilities.

* **`examples/` (Executable Binaries & Workflows)**:
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

### 5. Run Unit Tests & Benchmarks
```bash
# Run all unit tests across the entire codebase
zig build test
```

---

## 🗺️ Roadmap & Future Milestones

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

To push `znn` towards a production-ready, high-throughput deep learning engine in Zig, the following high-value areas are targeted:

1. **CPU Parallelism & Multi-Threading Support (High Priority)**:
   * **Multi-Threaded Tiled GEMM**: Replace the single-threaded SIMD fallback with a cache-friendly, tiled, multithreaded GEMM using Zig's `std.Thread` pool for Linux and Windows.
   * **Operator-Level Parallelism**: Multi-thread elementwise ops, norm layers, and convolutional forward/backward passes.

2. **External BLAS Linkage on Linux**:
   * Add build script integration options (`-Dblas=openblas` / `-Dblas=mkl`) to link optimized external BLAS backends on non-macOS systems.

3. **Model Serialization & Interoperability**:
   * **Safetensors / GGUF Parser**: Native pure Zig parser for `.safetensors` and `.gguf` binary weights, enabling zero-dependency loading of open-source models (Llama, Qwen, Mistral) directly from Hugging Face.
   * **Checkpoint Export**: Direct export of model weights and computation graphs to standardized formats (Safetensors / ONNX).

4. **GPU / Compute Shader Acceleration**:
   * Explore a compute shader execution backend using **WebGPU** (`wgpu-native` / Dawn) or **Vulkan** compute pipelines for cross-platform GPU training and inference directly from Zig.

5. **Static Graph Optimization & Memory In-Place Execution**:
   * **Activation Checkpointing (Gradient Checkpointing)**: Trade computation for memory during backward passes to enable training significantly longer sequence lengths.
   * **In-Place Operations**: Analyze computation graph lifespans to allow buffer reuse for non-branching activations.
   * **Comptime Tensor Type Safety**: Deeper unification of `StaticTensor` into the dynamic autograd engine for zero-allocation, statically verified subgraphs.
