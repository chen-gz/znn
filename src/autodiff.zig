const std = @import("std");

pub const types = @import("autodiff/types.zig");
pub const op = @import("autodiff/op.zig");
pub const backward_core = @import("autodiff/backward_core.zig");
pub const backward_nn = @import("autodiff/backward_nn.zig");
pub const backward_math = @import("autodiff/backward_math.zig");
pub const graph = @import("autodiff/graph.zig");
pub const graph_nn = @import("autodiff/graph_nn.zig");
pub const graph_init = @import("autodiff/graph_init.zig");

pub const OpType = types.OpType;
pub const OpContext = types.OpContext;
pub const Op = types.Op;
pub const Graph = graph.Graph;
pub const ExecutionMode = graph.ExecutionMode;
pub const GraphOptions = graph.GraphOptions;

test {
    std.testing.refAllDecls(@This());
    _ = @import("autodiff/tests.zig");
}

