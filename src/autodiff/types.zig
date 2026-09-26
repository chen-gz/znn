// 支持的算子类型枚举
pub const OpType = enum {
    // --- 基础数学与张量逐元素运算 (Basic Math & Element-wise Ops) ---
    Add, // 张量逐元素加法（支持多维广播）
    AddBias, // 偏置项加法（广播机制）
    AddScalar, // 标量加法
    Sub, // 张量逐元素减法（支持多维广播）
    SubScalar, // 标量减法
    Mul, // 张量逐元素乘法（支持多维广播）
    MulScalar, // 标量乘法（张量缩放）
    Div, // 张量逐元素除法（支持多维广播）
    DivScalar, // 标量除法
    MatMul, // 矩阵乘法
    BatchMatMul, // 批量矩阵乘法 (Batched Matrix Multiplication)

    // --- 形状与维度变换 (Shape & Dimension Transforms) ---
    Reshape, // 形状变换
    Transpose, // 维度转置
    Concat, // 张量沿指定维度拼接
    Split, // 张量沿指定维度切分
    RepeatKV, // GQA 注意力中沿 Head 维度复制广播 Key/Value 张量

    // --- 激活函数与非线性变换 (Activation Functions) ---
    Relu, // 激活函数 ReLU
    LeakyRelu, // 激活函数 LeakyReLU
    Gelu, // 激活函数 GELU
    Silu, // 激活函数 SiLU / Swish
    Sigmoid, // 激活函数 Sigmoid
    Tanh, // 激活函数 Tanh
    Softmax, // 独立 Softmax (Standalone Softmax)

    // --- 神经网络层与结构运算 (Neural Network Layers & Structural Ops) ---
    Conv2D, // 二维卷积
    ConvTranspose2D, // 二维转置卷积 / 反卷积
    MaxPool2D, // 二维最大池化
    RmsNorm, // RMSNorm 归一化
    Embedding, // 嵌入查找 (Embedding Lookup)

    // --- 损失函数与正则化 (Loss Functions & Regularization) ---
    MseLoss, // 均方误差损失函数 (MSE Loss)
    BceLoss, // 二元交叉熵损失函数 (BCE Loss)
    BceWithLogitsLoss, // 二元交叉熵带 Logits 损失函数
    SigmoidCrossEntropy, // Sigmoid 交叉熵损失 (BCE with logits)
    SoftmaxCrossEntropy, // 结合 Softmax 与交叉熵损失（数值稳定性更好）
    L1Loss, // L1 正则化 / Lasso Loss: lambda * sum(|w|)
    L2Loss, // L2 正则化 / Ridge Loss: 0.5 * lambda * sum(w^2)
};

// 各算子反向传播所需的上下文信息（如 Softmax 的概率输出与 Target 类别）
pub const OpContext = union(enum) {
    // --- 基础数学与张量逐元素运算 ---
    Add: void,
    AddBias: void,
    AddScalar: struct {
        val: f32,
    },
    Sub: void,
    SubScalar: struct {
        val: f32,
    },
    Mul: void,
    MulScalar: struct {
        val: f32,
    },
    Div: void,
    DivScalar: struct {
        val: f32,
    },
    MatMul: void,
    BatchMatMul: void,

    // --- 形状与维度变换 ---
    Reshape: void,
    Transpose: struct {
        dim0: usize,
        dim1: usize,
    },
    Concat: struct {
        dim: usize,
    },
    Split: struct {
        dim: usize,
    },
    RepeatKV: struct {
        groups: usize,
    },

    // --- 激活函数与非线性变换 ---
    Relu: void,
    LeakyRelu: struct {
        alpha: f32,
    },
    Gelu: void,
    Silu: void,
    Sigmoid: void,
    Tanh: void,
    Softmax: void,

    // --- 神经网络层与结构运算 ---
    Conv2D: void,
    ConvTranspose2D: struct {
        stride: usize,
        padding: usize,
    },
    MaxPool2D: struct {
        pool_size: usize,
        stride: usize,
    },
    RmsNorm: struct {
        eps: f32,
    },
    Embedding: void,

    // --- 损失函数与正则化 ---
    MseLoss: void,
    BceLoss: struct {
        eps: f32,
    },
    BceWithLogitsLoss: void,
    SigmoidCrossEntropy: void,
    SoftmaxCrossEntropy: struct {
        probs: []f32,
        targets: []const u8,
    },
    L1Loss: struct {
        lambda: f32,
    },
    L2Loss: struct {
        lambda: f32,
    },
};
