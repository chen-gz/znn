const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const Tensor = tensor.Tensor;

/// 线性整流单元 (Rectified Linear Unit, ReLU) 激活层
pub const ReLU = struct {
    pub fn forward(_: ReLU, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return try graph.relu(x);
    }
};

/// 高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 激活层
pub const GELU = struct {
    pub fn forward(_: GELU, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return try graph.gelu(x);
    }
};

/// S 型激活函数 (Sigmoid Activation, Sigmoid) 层
pub const Sigmoid = struct {
    pub fn forward(_: Sigmoid, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return try graph.sigmoid(x);
    }
};

/// 双曲正切激活函数 (Hyperbolic Tangent Activation, Tanh) 层
pub const Tanh = struct {
    pub fn forward(_: Tanh, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return try graph.tanh(x);
    }
};

/// 带泄露线性整流单元 (Leaky Rectified Linear Unit, LeakyReLU) 激活层
pub const LeakyReLU = struct {
    alpha: f32 = 0.2,

    pub const default: LeakyReLU = .{};

    pub fn defaultOptions() LeakyReLU {
        return .{};
    }

    pub fn initDefault() LeakyReLU {
        return .{};
    }

    pub fn forward(self: LeakyReLU, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return try graph.leakyRelu(x, self.alpha);
    }
};

/// Sigmoid 线性单元 (Sigmoid Linear Unit, SiLU) 激活层
pub const SiLU = struct {
    pub fn forward(_: SiLU, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return try graph.silu(x);
    }
};

/// Swish 激活函数别名，等价于 Sigmoid 线性单元 (Sigmoid Linear Unit, SiLU)
pub const Swish = SiLU;
