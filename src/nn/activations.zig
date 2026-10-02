const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const Tensor = tensor.Tensor;

pub const ReLU = struct {
    pub fn forward(_: ReLU, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return try graph.relu(x);
    }
};

pub const GELU = struct {
    pub fn forward(_: GELU, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return try graph.gelu(x);
    }
};

pub const Sigmoid = struct {
    pub fn forward(_: Sigmoid, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return try graph.sigmoid(x);
    }
};

pub const Tanh = struct {
    pub fn forward(_: Tanh, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return try graph.tanh(x);
    }
};

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

pub const SiLU = struct {
    pub fn forward(_: SiLU, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return try graph.silu(x);
    }
};

pub const Swish = SiLU;
