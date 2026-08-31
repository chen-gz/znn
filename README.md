# Zig ML: Minimal Deep Learning Library in Zig

This project is a minimal 3-layer Feedforward Neural Network (MLP) library built entirely from scratch in **Zig 0.16.0** for Fashion MNIST image classification.

It serves as a clean, production-grade reference for:
1. **Autograd Mechanics**: Understanding how dynamic backward automatic differentiation engines construct computation graphs and compute gradients.
2. **Zig Systems Programming**: Utilizing Zig's memory allocators, type reflection, memory safety, and `comptime` compile-time meta-programming.
3. **C Interoperability**: Directly binding and executing high-performance system-level C libraries (macOS Accelerate CBLAS / Apple Silicon AMX coprocessor) from Zig.

---

## 🚀 Key Features

1. **N-Dimensional Tensor Library**:
   * Supports arbitrary-dimensional tensors with native logical `Shape` and contiguous layouts computed via `strides`.
   * Custom multi-dimensional accessors: `get`, `set`, `getGrad`, and `setGrad` with automatic stride mapping.
   * Recursive, nested pretty-printing of N-dimensional structures (similar to NumPy or PyTorch's default representation).
   * Fully-featured `reshape` (zero-copy forward) and `transposeND` (physical transposition to contiguous layout) operators with complete backpropagation support.
   * Zero legacy matrix field overhead (no `rows` and `cols` fields on `Tensor`; dimensions are indexed directly from `shape`).

2. **Dynamic Autodiff Engine**:
   * Automatic backward propagation using depth-first search (DFS) topological sorting to build computation dependencies.
   * Core operators implemented: `MatMul`, `AddBias`, `ReLU`, `SoftmaxCrossEntropy`, `Reshape`, and `Transpose`.
   * Advanced memory recycling using `ArenaAllocator` to allocate intermediate tensor values and gradients per batch and release them in a single batch-level deallocation.

3. **Elegant PyTorch-like API**:
   * High-readability forward propagation interface: `logits = try model.forward(&graph, x_tensor)`.
   * Compile-time reflection (`comptime`) to automatically manage parameter lifetime, model serialization (`save` / `load`), SGD momentum updates (`updateWeights`), and gradient flushing (`zeroGrad`).

4. **Accelerated CPU Math**:
   * Integrates macOS `Accelerate` CBLAS library to execute single-threaded matrix operations on Apple Silicon's AMX coprocessor.
   * Avoids unnecessary complexity of thread pools and GPU scheduling, yielding exceptional runtime efficiency and minimal code foot-print.

5. **100% Pure Zig & Zero Dependencies**:
   * Builds into a completely self-contained binary. No python virtualenv, heavy PyTorch wheels, or third-party packages required.

---

## 📂 Codebase Directory Structure

* **[examples/cnn.zig](examples/cnn.zig)**: 2D Convolutional Neural Network (CNN) binary target using Conv2D, MaxPool2D, and Linear layers for Fashion MNIST image classification.
* **[examples/fashion_mnist.zig](examples/fashion_mnist.zig)**: 3-layer Feedforward Neural Network (MLP) binary target. Responsible for dataset loading, training loops, evaluation, and test predictions.
* **[examples/linear_regression.zig](examples/linear_regression.zig)**: Linear regression binary target. Compares OLS analytical closed-form solution with iterative autograd-based gradient descent.
* **[examples/logistic_regression.zig](examples/logistic_regression.zig)**: Logistic regression binary classification target using Sigmoid and SigmoidCrossEntropy loss.
* **[examples/ridge_regression.zig](examples/ridge_regression.zig)**: Ridge regression ($L_2$ regularization) binary target. Compares closed-form matrix solution with autograd gradient descent and demonstrates multicollinearity mitigation.
* **[src/tensor.zig](src/tensor.zig)**: N-Dimensional Tensor library. Implements shape, logical strides, memory layout mapping, and vectorized math.
* **[src/autodiff.zig](src/autodiff.zig)**: Core Automatic Differentiation engine. Implements the dynamic computation `Graph`, `Node`, operators, and DFS topological sorting.
* **[src/nn.zig](src/nn.zig)**: Neural Network Modules. Implements the `Linear` and `Conv2D` modules, activation functions, and `Module` wrapper for comptime reflection parameter management.
* **[src/cblas.zig](src/cblas.zig)**: System CBLAS C-bindings for macOS Accelerate framework matrix operations.
* **[src/dataset.zig](src/dataset.zig)**: Custom binary parser for Fashion MNIST IDX format files.
* **[src/root.zig](src/root.zig)**: Module exports and compile-time unit tests.
* **[build.zig](build.zig)**: Compilation build script detailing target configurations, Accelerate framework linking, and test runner tasks.

---

## 🛠️ Build and Execution

### 1. Download Datasets
You can download computer vision and LLM text datasets with a single command:
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
The downloaded datasets will automatically be saved and extracted into the `data/` directory.

### 2. Compile and Run Linear Regression & Ridge Regression
Run the 1D linear regression or ridge regression example:
```bash
# Linear regression (OLS vs autograd GD)
zig build run-lr

# Logistic regression (Sigmoid BCE)
zig build run-logr

# Ridge regression (L2 regularization, multicollinearity mitigation & Ridge trace)
zig build run-ridge
```

### 3. Compile and Run LLM & GPT Model Training
Run end-to-end LLM pretraining, LoRA fine-tuning, and Shakespeare generation:
```bash
# End-to-End LLM Pipeline demo (BPE + SwiGLU + AdamW + SFT + LoRA + DPO + Top-P)
zig build run-llm

# Train Mini GPT on TinyShakespeare corpus and generate text live
zig build run-shakespeare
```

### 4. Compile and Run Vision Neural Network Training (MLP & CNN)
```bash
# Run MLP training on Fashion MNIST
zig build run -Doptimize=ReleaseFast

# Run CNN training
zig build run-cnn
```

### 5. Run Unit Tests
Execute the test suites containing autograd, SwiGLU, AdamW, LoRA, BPE, and dataset validation:
```bash
zig build test
```

---

## 🗺️ Future Improvements & Roadmap

To make `znn` a more complete and high-performance library, the following areas have been identified for improvement:

1. **Decoupled Optimizer Framework (High Priority)**
   * Currently, SGD with Momentum is hardcoded directly inside layer structures (e.g., `nn.Linear`). We plan to extract this state into a dedicated `Optimizer` abstraction.
   * Add support for more optimizers, specifically **Adam** and **AdamW**, which are critical for training modern Transformer-based architectures efficiently.
   * See [optimizer_design_plan.md](file:///usr/local/google/home/guangzong/.gemini/jetski/brain/c4a0dd41-6e44-4379-877d-925d1eae24d6/optimizer_design_plan.md) for the detailed design.

2. **SIMD Vectorization for CPU Fallback Math**
   * The fallback GEMM (`cblas_sgemm_fallback` in `src/cblas.zig`) is a naive, unoptimized $O(N^3)$ implementation.
   * Optimize it using Zig's native `@Vector` types to enable SIMD acceleration on platforms like Linux and Windows without external dependencies.

3. **External BLAS Support on Linux**
   * Support linking to optimized C BLAS libraries (like OpenBLAS or Intel MKL) on Linux, matching the Accelerate framework integration on macOS.

4. **GPU / WebGPU Acceleration**
   * Integrate WebGPU or Vulkan compute shaders to allow compiling and running neural network training on GPUs from pure Zig.

5. **More Core Operators & Layers**
   * Add common neural network blocks: **BatchNorm2d**, **LayerNorm** (in addition to RMSNorm), **Dropout**, and average pooling.
   * Implement additional loss functions like **MSELoss** and **BCEWithLogitsLoss**.

6. **Comptime Shape Checking**
   * Leverage Zig's `comptime` capabilities to validate tensor shapes and compile-time dimensions where possible, failing compilation early on incompatible matrix operations.

