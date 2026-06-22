# Zig ML: 基于 Zig 0.16.0 与双硬件加速的高性能深度学习框架

本项目是一个使用 **Zig 0.16.0** 从零实现的三层前馈神经网络（MLP），用于 Fashion MNIST 图像分类。它内置了完整的**动态逆向自动微分引擎（Dynamic Autodiff）**，并在 macOS 上提供了高性能的 CPU（Apple AMX 协处理器）与 GPU（Metal Compute Shader）双加速计算后端。

---

## 🚀 核心特性

1. **动态逆向自动微分引擎**：
   * 支持 `MatMul`（矩阵乘法）、`AddBias`（偏置相加）、`ReLU`（激活函数）和 `SoftmaxCrossEntropy`（损失函数）等核心算子的动态求导。
   * 采用 **DFS 拓扑排序** 算法，自动规划反向传播求导链条。
   * 基于 **ArenaAllocator** 的计算图生命周期管理，每次 Batch 结束后一键释放所有临时层节点内存，无内存泄露和碎片。

2. **极简 PyTorch-like API**：
   * 封装了 `forward` 接口，在 Zig 中以接近 PyTorch 的可读性（`logits = model.forward(&graph, x)`）定义并构建多层复杂神经网络。

3. **双硬件加速后端**：
   * **CPU 加速（AMX）**：通过 macOS `Accelerate` 动态链接库直接调用系统 `cblas_sgemm`，深度压榨 Apple Silicon 芯片中的 AMX 矩阵计算硬件。
   * **GPU 加速（Metal）**：通过 Objective-C++ 联合编译，利用 Metal Compute Shader 在苹果 GPU 上执行矩阵并行计算。

4. **无外部依赖 & 极致轻量化**：
   * 除 macOS 系统自带的核心框架（`Accelerate`、`Metal`、`Foundation`）外，100% 纯 Zig 实现，编译产出的 Release 独立运行二进制文件仅为 **~420KB**。

---

## 📊 性能表现（Apple Silicon）

在相同的硬件环境下，训练 15 个 Epoch 的 Fashion MNIST（Batch Size = 64），平均每个 Epoch 的训练耗时对比如下：

| 运行环境/版本 | 底层芯片与后端 | 平均 Epoch 耗时 | 备注 |
| :--- | :--- | :--- | :--- |
| **自研 Zig (CBLAS 加速版)** 🚀 | **苹果 CPU (AMX)** | **~0.18 秒** | 缓存友好，无额外调用延迟，表现最快。 |
| **自研 Zig (Metal GPU 版)** ⚡ | **苹果 GPU (Metal)** | **~2.20 秒** | 适合大规模矩阵，小矩阵受设备指令编码与调度开销限制。 |
| **PyTorch CPU (多线程)** | CPU 多线程 (MKL/OpenMP) | **~1.70 秒** | 工业级优化，但受 Python 解释器及封装影响慢于原生 Zig CBLAS。 |
| **PyTorch MPS (GPU)** | 苹果 GPU (MPS) | **~2.75 秒** | PyTorch 官方的苹果 GPU 加速通道。 |

---

## 🛠️ 编译与运行

### 1. 准备数据集
在项目根目录下创建 `data/` 文件夹，并下载/解压 [Fashion MNIST 数据集](https://github.com/zalandoresearch/fashion-mnist) 的 idx 二进制文件：
* `train-images-idx3-ubyte`
* `train-labels-idx1-ubyte`
* `t10k-images-idx3-ubyte`
* `t10k-labels-idx1-ubyte`

### 2. 编译并运行 (CPU - CBLAS / AMX)
```bash
# 运行高性能 Release 模式
zig build run -Doptimize=ReleaseFast

# 运行 Debug 模式（含越界与安全检查）
zig build run
```

### 3. 编译并运行 (GPU - Metal)
```bash
# 运行高性能 Release 模式并使用 GPU
zig build run -Doptimize=ReleaseFast -- --gpu

# 运行 Debug 模式并使用 GPU
zig build run -- --gpu
```

---

## 📂 项目结构说明

* **[src/main.zig](file:///Users/guangzong/Documents/zig_ml/src/main.zig)**：框架的执行入口。负责解析参数、加载数据集、配置神经网络大小、运行训练 Epoch 循环、输出评测指标并保存模型。
* **[src/autodiff.zig](file:///Users/guangzong/Documents/zig_ml/src/autodiff.zig)**：自动微分引擎的实现。定义了张量节点（`Tensor`）、计算图（`Graph`）的拓扑排序、图分配和反向回传求导逻辑。
* **[src/cblas.zig](file:///Users/guangzong/Documents/zig_ml/src/cblas.zig)**：底层 C 语言绑定接口。隔离平台相关的 C 动态链接配置，封装了 Accelerate Framework 的 BLAS 接口及 Metal GPU 的调用接口。
* **[src/nn.zig](file:///Users/guangzong/Documents/zig_ml/src/nn.zig)**：神经网络模型定义。包括 Kaiming (He) 参数初始化、带有 Momentum 动量的 SGD 权重更新机制、模型的序列化二进制保存与加载、以及 PyTorch 风格的 `forward` 前向接口。
* **[src/dataset.zig](file:///Users/guangzong/Documents/zig_ml/src/dataset.zig)**：Fashion MNIST idx 格式数据集的自定义解析器，解析为一维的 `f32` 浮点数切片。
* **[src/metal_backend.mm](file:///Users/guangzong/Documents/zig_ml/src/metal_backend.mm)**：Objective-C++ 源码。编写了 Metal Pipeline、Compute Shader 的载入、Unified Memory 缓冲区管理，以及调用自定义 Shader 实现的矩阵计算。
* **[build.zig](file:///Users/guangzong/Documents/zig_ml/build.zig)**：Zig 构建描述脚本。配置了 C 语言编译环境、Metal/Accelerate 系统框架链接及自动编译管线。
