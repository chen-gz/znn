const std = @import("std");

pub const types = @import("autodiff/types.zig");
pub const op = @import("autodiff/op.zig");
pub const graph = @import("autodiff/graph.zig");

// Re-export core types & components for 100% backward compatibility
pub const OpType = types.OpType;
pub const OpContext = types.OpContext;
pub const Op = op.Op;
pub const Graph = graph.Graph;

// Re-export tensor types referenced in autodiff
pub const tensor = @import("tensor.zig");
pub const Tensor = tensor.Tensor;
pub const Shape = tensor.Shape;
pub const computeContiguousStrides = tensor.computeContiguousStrides;
pub const transposeShape = tensor.transposeShape;

test {
    _ = @import("autodiff/tests.zig");
}
