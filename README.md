# Zig ML (Educational): Minimal Deep Learning Library in Zig

This project is a minimal, educational 3-layer Feedforward Neural Network (MLP) library built entirely from scratch in **Zig 0.16.0** for Fashion MNIST image classification.

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
* **[src/tensor.zig](src/tensor.zig)**: N-Dimensional Tensor library. Implements shape, logical strides, memory layout mapping, and vectorized math.
* **[src/autodiff.zig](src/autodiff.zig)**: Core Automatic Differentiation engine. Implements the dynamic computation `Graph`, `Node`, operators, and DFS topological sorting.
* **[src/nn.zig](src/nn.zig)**: Neural Network Modules. Implements the `Linear` and `Conv2D` modules, activation functions, and `Module` wrapper for comptime reflection parameter management.
* **[src/cblas.zig](src/cblas.zig)**: System CBLAS C-bindings for macOS Accelerate framework matrix operations.
* **[src/dataset.zig](src/dataset.zig)**: Custom binary parser for Fashion MNIST IDX format files.
* **[src/root.zig](src/root.zig)**: Module exports and compile-time unit tests.
* **[build.zig](build.zig)**: Compilation build script detailing target configurations, Accelerate framework linking, and test runner tasks.

---

## 🛠️ Build and Execution

### 1. Download Dataset
Create a `data/` directory in the project root and download/extract the [Fashion MNIST IDX format files](https://github.com/zalandoresearch/fashion-mnist):
* `train-images-idx3-ubyte`
* `train-labels-idx1-ubyte`
* `t10k-images-idx3-ubyte`
* `t10k-labels-idx1-ubyte`

### 2. Compile and Run Linear Regression
Run the simple 1D linear regression example (analytical vs autograd GD):
```bash
zig build run-lr
```

### 3. Compile and Run Neural Network Model Training (MLP)
Run the training pipeline in high-performance Release mode:
```bash
# Run training in optimized ReleaseFast mode
zig build run -Doptimize=ReleaseFast

# Run training in Debug mode with full runtime safety checks
zig build run
```

### 4. Compile and Run CNN Model Training
Run the CNN model training pipeline (Conv2D + MaxPool2D + ReLU + Linear):
```bash
# Run CNN training
zig build run-cnn
```

### 5. Run Unit Tests
Execute the test suites containing autograd, reshape, transpose, and dataset parser validation:
```bash
zig build test
```

---

## 🗺️ Future Improvements & Roadmap

To make `znn` a more complete and high-performance educational library, the following areas have been identified for improvement:

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

