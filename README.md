# Zig ML (Educational): 极简 Zig 深度学习教学项目

本项目是一个为教学目的设计的、使用 **Zig 0.16.0** 从零实现的三层前馈神经网络（MLP）框架，用于 Fashion MNIST 图像分类。

它非常适合作为学习以下内容的实战案例：
1. **自动微分原理**：理解动态逆向自动微分引擎（Dynamic Autodiff）的运行方式。
2. **Zig 语言实战**：学习 Zig 的内存分配（Allocator）、所有权托管与编译期反射等语法特性。
3. **C 语言互操作**：掌握如何在 Zig 中直接链接并调用底层的系统 C 语言库（CBLAS/AMX 矩阵加速）。

---

## 🚀 核心教学特性

1. **从零实现动态自动微分引擎 (Dynamic Autodiff)**：
   * 支持 `MatMul`（矩阵乘法）、`AddBias`（偏置相加）、`ReLU`（激活函数）和 `SoftmaxCrossEntropy`（损失函数结合）等核心算子的求导逻辑。
   * 采用 **DFS 拓扑排序** 算法，自动寻找并构建反向传播求导链条。
   * 使用 **ArenaAllocator** 托管计算图生命周期，每次 Batch 结束后一键释放所有求导中间层的 Tensor 节点内存，展示了 Zig 极简而安全的内存管理技巧。

2. **精巧的仿 PyTorch 式 API (PyTorch-like Wrapper)**：
   * 提供了高可读性的前向接口：`logits = try model.forward(&graph, x)`。
   * 通过 Zig 的编译期反射（`comptime`），实现了一个轻量级的 `Module` 包装器，自动路由和实现模型的参数初始化、梯度清零 (`zeroGrad`)、梯度更新 (`updateWeights`) 和模型的二进制存取 (`save` & `load`)。

3. **极致简单的 CPU 矩阵乘法加速 (CBLAS sgemm)**：
   * 摒弃了复杂的 GPU (Metal/MLX) 依赖和多线程调度逻辑（移除了不必要的 ThreadPool）。
   * 直接通过 macOS 的 `Accelerate` 框架链接系统 CBLAS 库，底层自动使用 Apple Silicon 的 AMX 协处理器执行单线程矩阵乘法，性能卓越且代码高度简化。

4. **100% 纯 Zig & 零第三方包依赖**：
   * 独立运行二进制文件，除了 macOS 自带的 `Accelerate` CBLAS 动态库外，无任何第三方包依赖。

---

## 📂 项目结构说明

* **[src/main.zig](file:///Users/guangzong/Documents/zig_ml/src/main.zig)**：框架的教学执行入口。负责加载数据集、构建标准模型、运行 Epoch 训练循环，并展示最终的预测结果与模型保存。
* **[src/autodiff.zig](file:///Users/guangzong/Documents/zig_ml/src/autodiff.zig)**：核心自动微分引擎实现。定义了张量节点（`Tensor`）、计算图结构（`Graph`）及其算子反向传播机制。
* **[src/nn.zig](file:///Users/guangzong/Documents/zig_ml/src/nn.zig)**：神经网络层定义。包括标准 MLP 的三层模型结构设计、Kaiming (He) 参数初始化、带有 Momentum 动量的 SGD 权重更新机制、以及 `Module` 包装器。
* **[src/cblas.zig](file:///Users/guangzong/Documents/zig_ml/src/cblas.zig)**：系统 C 语言加速库接口绑定，声明了 CBLAS 矩阵乘法接口。
* **[src/dataset.zig](file:///Users/guangzong/Documents/zig_ml/src/dataset.zig)**：自定义的 Fashion MNIST idx 格式文件二进制解析器。
* **[src/root.zig](file:///Users/guangzong/Documents/zig_ml/src/root.zig)**：项目的导出根模块以及存放基础编译期单元测试的地方。
* **[build.zig](file:///Users/guangzong/Documents/zig_ml/build.zig)**：Zig 构建描述脚本。配置了 C 语言编译环境与 macOS Accelerate 系统框架链接。

---

## 🛠️ 编译与运行

### 1. 准备数据集
在项目根目录下创建 `data/` 文件夹，并下载/解压 [Fashion MNIST 数据集](https://github.com/zalandoresearch/fashion-mnist) 的 idx 二进制文件：
* `train-images-idx3-ubyte`
* `train-labels-idx1-ubyte`
* `t10k-images-idx3-ubyte`
* `t10k-labels-idx1-ubyte`

### 2. 编译并运行
运行命令行：
```bash
# 运行高性能 Release 模式进行模型训练
zig build run -Doptimize=ReleaseFast

# 运行 Debug 模式（含完整运行时安全与越界检查）
zig build run
```

### 3. 运行单元测试
```bash
# 运行项目自带的单元测试
zig build test
```
