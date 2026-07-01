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

* **[src/fashion_mnist.zig](src/fashion_mnist.zig)**: 3-layer Feedforward Neural Network (MLP) binary target. Responsible for dataset loading, training loops, evaluation, and test predictions.
* **[src/linear_regression.zig](src/linear_regression.zig)**: Linear regression binary target. Compares OLS analytical closed-form solution with iterative autograd-based gradient descent.
* **[src/tensor.zig](src/tensor.zig)**: N-Dimensional Tensor library. Implements shape, logical strides, memory layout mapping, and vectorized math.
* **[src/autodiff.zig](src/autodiff.zig)**: Core Automatic Differentiation engine. Implements the dynamic computation `Graph`, `Node`, operators, and DFS topological sorting.
* **[src/nn.zig](src/nn.zig)**: Neural Network Modules. Implements the `Linear` module, activation functions, and `Module` wrapper for comptime reflection parameter management.
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

### 3. Compile and Run Neural Network Model Training
Run the training pipeline in high-performance Release mode:
```bash
# Run training in optimized ReleaseFast mode
zig build run -Doptimize=ReleaseFast

# Run training in Debug mode with full runtime safety checks
zig build run
```

### 4. Run Unit Tests
Execute the test suites containing autograd, reshape, transpose, and dataset parser validation:
```bash
zig build test
```
