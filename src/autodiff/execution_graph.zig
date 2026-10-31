//! 静态计算图编译与重放引擎 (Static Execution Graph & Replay Engine)
const std = @import("std");
const backend = @import("../backend.zig");
const Graph = @import("graph.zig").Graph;
const Op = @import("types.zig").Op;
const op_mod = @import("op.zig");
const Tensor = @import("../tensor.zig").Tensor;
const memory_planner = @import("memory_planner.zig");
const MemoryPlan = memory_planner.MemoryPlan;
const MemoryPlanner = memory_planner.MemoryPlanner;

pub const ExecutionGraph = struct {
    allocator: std.mem.Allocator,
    device: backend.Device,
    shared_arena: backend.DeviceBuffer, // 唯一的全图共享物理连续内存池
    plan: MemoryPlan,
    ordered_ops: []*Op,
    outputs: []*Tensor,
    is_compiled: bool = false,

    pub fn compile(
        graph: *Graph,
        device: backend.Device,
        graph_outputs: []const *Tensor,
        allocator: std.mem.Allocator,
    ) !ExecutionGraph {
        var planner = MemoryPlanner.init(allocator);
        var plan = try planner.buildPlan(graph.ops.items, graph.tensors.items, graph_outputs);
        errdefer plan.deinit();

        var shared_arena = if (plan.total_arena_bytes > 0)
            try backend.DeviceBuffer.alloc(device, plan.total_arena_bytes, allocator)
        else
            backend.DeviceBuffer.emptyCpu(device);
        errdefer shared_arena.free();

        // 将参与复用的中间张量底层存储重定向到 shared_arena 的切片视图
        for (plan.assignments) |assign| {
            const t = assign.tensor;
            if (device.type == .cpu) {
                const sub_slice = try shared_arena.slice(assign.offset_bytes, assign.size_bytes);
                t.storage.free();
                t.storage = sub_slice;
                t.data = @as([*]f32, @ptrCast(@alignCast(shared_arena.handle.cpu + assign.offset_bytes)))[0..t.shape.numel()];
                t.is_view = true;
            }
        }

        const ordered_ops = try allocator.dupe(*Op, graph.ops.items);
        errdefer allocator.free(ordered_ops);

        const outputs = try allocator.dupe(*Tensor, graph_outputs);
        errdefer allocator.free(outputs);

        return ExecutionGraph{
            .allocator = allocator,
            .device = device,
            .shared_arena = shared_arena,
            .plan = plan,
            .ordered_ops = ordered_ops,
            .outputs = outputs,
            .is_compiled = true,
        };
    }

    pub fn replay(self: *ExecutionGraph, scratch_allocator: std.mem.Allocator) !void {
        for (self.ordered_ops) |op| {
            try op_mod.forward(op, scratch_allocator);
        }
    }

    pub fn deinit(self: *ExecutionGraph) void {
        self.shared_arena.free();
        self.plan.deinit();
        self.allocator.free(self.ordered_ops);
        self.allocator.free(self.outputs);
    }
};
