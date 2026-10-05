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
    Sqrt, // 逐元素平方根
    Exp, // 逐元素自然指数
    Log, // 逐元素自然对数
    Abs, // 逐元素绝对值
    Sum, // 张量按轴或全局求和归约
    Mean, // 张量按轴或全局均值归约
    Where, // 按条件张量逐元素选择
    MaskedFill, // 按掩码张量填充标量值
    MatMul, // 矩阵乘法
    BatchMatMul, // 批量矩阵乘法 (Batched Matrix Multiplication)

    // --- 形状与维度变换 (Shape & Dimension Transforms) ---
    Reshape, // 形状变换
    Transpose, // 维度转置
    Concat, // 张量沿指定维度拼接
    Split, // 张量沿指定维度切分
    RepeatKV, // 分组查询注意力 (Grouped-Query Attention, GQA) 中沿注意力头维度复制广播键/值 (Key/Value, KV) 张量
    Slice, // 多维跨步切片

    // --- 激活函数与非线性变换 (Activation Functions) ---
    Relu, // 修正线性单元 (Rectified Linear Unit, ReLU) 激活函数
    LeakyRelu, // 带泄露修正线性单元 (Leaky Rectified Linear Unit, LeakyReLU) 激活函数
    Gelu, // 高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 激活函数
    Silu, // Sigmoid 线性单元 (Sigmoid Linear Unit, SiLU / Swish) 激活函数
    Sigmoid, // 激活函数 Sigmoid
    Tanh, // 双曲正切函数 (Hyperbolic Tangent, Tanh) 激活函数
    Softmax, // 独立归一化指数函数 (Standalone Softmax)

    // --- 神经网络层与结构运算 (Neural Network Layers & Structural Ops) ---
    Conv1D, // 一维卷积 (1D Convolution, Conv1D)
    Conv2D, // 二维卷积 (2D Convolution, Conv2D)
    ConvTranspose2D, // 二维转置卷积 / 反卷积 (2D Transposed Convolution, ConvTranspose2D)
    MaxPool2D, // 二维最大池化 (2D Max Pooling, MaxPool2D)
    AvgPool2D, // 二维平均池化 (2D Average Pooling, AvgPool2D)
    RmsNorm, // 均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)
    LayerNorm, // 层归一化 (Layer Normalization, LayerNorm)
    BatchNorm2d, // 二维批量归一化 (2D Batch Normalization, BatchNorm2d)
    Dropout, // 随机丢弃正则化
    RoPE, // 旋转位置编码 (Rotary Position Embedding, RoPE)
    Embedding, // 嵌入查找 (Embedding Lookup)

    // --- 损失函数与正则化 (Loss Functions & Regularization) ---
    MseLoss, // 均方误差损失函数 (Mean Squared Error, MSE Loss)
    BceLoss, // 二元交叉熵损失函数 (Binary Cross-Entropy, BCE Loss)
    BceWithLogitsLoss, // 二元交叉熵带对数几率损失函数 (Binary Cross-Entropy with Logits, BCEWithLogitsLoss)
    SigmoidCrossEntropy, // Sigmoid 二元交叉熵损失 (Binary Cross-Entropy, BCE with Logits)
    SoftmaxCrossEntropy, // 结合 Softmax 与交叉熵损失（数值稳定性更好）
    DpoLoss, // 直接偏好优化损失 (Direct Preference Optimization, DPO Loss)
    GrpoLoss, // 群组相对策略优化损失 (Group Relative Policy Optimization, GRPO Loss)
    L1Loss, // 一阶范数 (L1 Norm) 正则化 / 最小绝对收缩和选择算子 (Least Absolute Shrinkage and Selection Operator, LASSO) 损失: lambda * sum(|w|)
    L2Loss, // 二阶范数 (L2 Norm) 正则化 / 岭回归 (Ridge) 损失: 0.5 * lambda * sum(w^2)

    /// 算子内置的标准 LaTeX 数学表达式
    pub fn getFormula(self: OpType) []const u8 {
        return switch (self) {
            .Add => "C = A + B",
            .AddBias => "y = x + b",
            .AddScalar => "y = x + c",
            .Sub => "C = A - B",
            .SubScalar => "y = x - c",
            .Mul => "C = A \\odot B",
            .MulScalar => "y = c \\cdot x",
            .Div => "C = A \\oslash B",
            .DivScalar => "y = \\frac{x}{c}",
            .Sqrt => "y = \\sqrt{x}",
            .Exp => "y = e^x",
            .Log => "y = \\ln(x)",
            .Abs => "y = |x|",
            .Sum => "y = \\sum x",
            .Mean => "y = \\frac{1}{N} \\sum x",
            .Where => "y = \\text{where}(c, x, y)",
            .MaskedFill => "y_i = \\begin{cases} v & m_i \\\\ x_i & \\neg m_i \\end{cases}",
            .MatMul => "C = A \\cdot B",
            .BatchMatMul => "C_{b, h} = A_{b, h} \\cdot B_{b, h}",
            .Reshape => "y = \\text{reshape}(x, \\text{new\\_shape})",
            .Transpose => "y = x^T",
            .Concat => "y = [x_1, x_2, \\dots, x_k]",
            .Split => "[y_1, y_2, \\dots, y_k] = \\text{split}(x)",
            .RepeatKV => "y = \\text{repeat\\_kv}(x, \\text{groups})",
            .Slice => "y = x[\\text{ranges}]",
            .Relu => "y = \\max(0, x)",
            .LeakyRelu => "y = \\max(\\alpha x, x)",
            .Gelu => "y = 0.5 x \\left(1 + \\text{erf}\\left(\\frac{x}{\\sqrt{2}}\\right)\\right)",
            .Silu => "y = x \\cdot \\sigma(x)",
            .Sigmoid => "y = \\frac{1}{1 + e^{-x}}",
            .Tanh => "y = \\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}",
            .Softmax => "P_i = \\frac{e^{z_i - \\max(z)}}{\\sum_j e^{z_j - \\max(z)}}",
            .Conv1D, .Conv2D => "y = x \\ast W + b",
            .ConvTranspose2D => "y = x \\ast_{\\text{deconv}} W + b",
            .MaxPool2D => "y = \\max_{k \\times k}(x)",
            .AvgPool2D => "y = \\frac{1}{k^2} \\sum_{k \\times k} x",
            .RmsNorm => "y = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum x_i^2 + \\epsilon}} \\odot \\gamma",
            .LayerNorm => "y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\gamma + \\beta",
            .BatchNorm2d => "y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}} \\odot \\gamma + \\beta",
            .Dropout => "y = \\frac{m \\odot x}{1 - p}",
            .RoPE => "y_t = R_{\\Theta, t} x_t",
            .Embedding => "y = W_e[\\text{indices}]",
            .MseLoss => "\\mathcal{L} = \\frac{1}{N} \\sum (y - \\hat{y})^2",
            .BceLoss => "\\mathcal{L} = -\\frac{1}{N} \\sum [y \\log \\hat{y} + (1-y) \\log(1-\\hat{y})]",
            .BceWithLogitsLoss => "\\mathcal{L} = \\max(x, 0) - x \\cdot y + \\log(1 + e^{-|x|})",
            .SigmoidCrossEntropy => "\\mathcal{L} = \\max(x, 0) - x \\cdot y + \\log(1 + e^{-|x|})",
            .SoftmaxCrossEntropy => "\\mathcal{L} = -\\log \\left( \\frac{e^{z_y}}{\\sum_j e^{z_j}} \\right)",
            .DpoLoss => "\\mathcal{L}_{\\text{DPO}} = -\\log \\sigma(\\beta (\\log \\frac{\\pi_\\theta(y_w)}{\\pi_{\\text{ref}}(y_w)} - \\log \\frac{\\pi_\\theta(y_l)}{\\pi_{\\text{ref}}(y_l)}))",
            .GrpoLoss => "\\mathcal{L}_{\\text{GRPO}} = -\\frac{1}{N} \\sum [\\min(r_t A_t, \\text{clip}(r_t) A_t) - \\beta D_{\\text{KL}}]",
            .L1Loss => "\\mathcal{L}_{\\text{reg}} = \\lambda \\sum |w|",
            .L2Loss => "\\mathcal{L}_{\\text{reg}} = \\frac{1}{2} \\lambda \\sum w^2",
        };
    }
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
    Sqrt: void,
    Exp: void,
    Log: void,
    Abs: void,
    Sum: struct {
        axis: ?usize,
        keepdims: bool,
    },
    Mean: struct {
        axis: ?usize,
        keepdims: bool,
    },
    Where: struct {
        mask: []bool,
    },
    MaskedFill: struct {
        mask: []bool,
        value: f32,
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
    Slice: struct {
        offset: usize,
        strides: [8]usize,
        rank: usize,
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
    Conv1D: struct {
        stride: usize = 1,
        padding: usize = 0,
    },
    Conv2D: struct {
        stride: usize = 1,
        padding: usize = 0,
    },
    ConvTranspose2D: struct {
        stride: usize,
        padding: usize,
    },
    MaxPool2D: struct {
        pool_size: usize,
        stride: usize,
    },
    AvgPool2D: struct {
        kernel_size: usize,
        stride: usize,
    },
    RmsNorm: struct {
        eps: f32,
    },
    LayerNorm: struct {
        eps: f32,
    },
    BatchNorm2d: struct {
        eps: f32,
        training: bool,
        save_mean: []f32,
        save_inv_std: []f32,
    },
    Dropout: struct {
        mask_scale: []f32,
    },
    RoPE: struct {
        start_pos: usize,
        rotary_offset: usize,
        rope_theta: f32 = 10000.0,
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
        targets: []const usize,
        mask: ?[]const f32 = null,
        total_weight: f32 = 0.0,
    },
    DpoLoss: struct {
        ref_chosen: []const f32,
        ref_rejected: []const f32,
        beta: f32,
    },
    GrpoLoss: struct {
        advantages: []const f32,
        ref_logps: ?[]const f32,
        beta: f32,
        clip_eps: f32,
    },
    L1Loss: struct {
        lambda: f32,
    },
    L2Loss: struct {
        lambda: f32,
    },
};
