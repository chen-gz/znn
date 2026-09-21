const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const Tensor = tensor.Tensor;

pub const ReLU = struct {
    pub fn forward(_: ReLU, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        return try x.relu(allocator, graph);
    }
};

pub const GELU = struct {
    pub fn forward(_: GELU, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        return try x.gelu(allocator, graph);
    }
};

pub const Sigmoid = struct {
    pub fn forward(_: Sigmoid, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        return try x.sigmoid(allocator, graph);
    }
};

pub const Tanh = struct {
    pub fn forward(_: Tanh, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        return try x.tanh(allocator, graph);
    }
};

pub const LeakyReLU = struct {
    alpha: f32 = 0.2,

    pub fn forward(self: LeakyReLU, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        return try x.leakyRelu(self.alpha, allocator, graph);
    }
};

pub const SiLU = struct {
    pub fn forward(_: SiLU, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        return try x.silu(allocator, graph);
    }
};

pub const Swish = SiLU;
