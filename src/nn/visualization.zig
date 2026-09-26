const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const init_mod = @import("init.zig");

const Tensor = tensor.Tensor;
const Graph = autodiff.Graph;

// ============================================================================
// 1. 中间数据结构与契约定义 (Agreed Intermediate Data Contract)
// ============================================================================

/// 单个计算图节点/张量的详细元数据
pub const NodeData = struct {
    id: usize,
    name: []const u8,
    leaf_name: []const u8,
    path: []const u8,
    kind: []const u8, // "Param", "Input", "Activation"
    shape_str: []const u8,
    shape_dims: []const usize,
    elements: usize,
    bytes: usize,
    status: []const u8, // "AUTO_GRAPH", "CUSTOM_INIT", "INPUT", "OP_OUTPUT"
    inferred_act: []const u8,
    strategy: []const u8,
    op: ?[]const u8 = null,
    inputs: std.ArrayList([]const u8),
    outputs: std.ArrayList([]const u8),
};

/// 递归模型层级模块组节点 (Recursive Module Node)
pub const ModuleNode = struct {
    name: []const u8,
    path: []const u8,
    kind: []const u8 = "module",
    total_params: usize = 0,
    total_bytes: usize = 0,
    param_count: usize = 0,
    node_count: usize = 0,
    children: std.ArrayList(*ModuleNode),
    nodes: std.ArrayList(NodeData),

    pub fn init(allocator: std.mem.Allocator, name: []const u8, path: []const u8) !*ModuleNode {
        const node = try allocator.create(ModuleNode);
        node.* = .{
            .name = name,
            .path = path,
            .kind = "module",
            .total_params = 0,
            .total_bytes = 0,
            .param_count = 0,
            .node_count = 0,
            .children = .empty,
            .nodes = .empty,
        };
        return node;
    }
};

/// 计算图连通性有向边 (Edge)
pub const GraphEdge = struct {
    from: []const u8,
    to: []const u8,
    op: []const u8,
    shape: []const u8,
};

/// 模型全局摘要统计指标 (KPI Summary)
pub const Summary = struct {
    total_params: usize = 0,
    total_bytes: usize = 0,
    param_nodes: usize = 0,
    input_nodes: usize = 0,
    activation_nodes: usize = 0,
    custom_init_count: usize = 0,
    auto_graph_count: usize = 0,
    total_nodes: usize = 0,
};

/// 包含完整递归树、DAG有向图连通关系与全局指标的模型结构中间数据
pub const ModelHierarchyGraph = struct {
    arena: std.heap.ArenaAllocator,
    summary: Summary,
    root: *ModuleNode,
    edges: std.ArrayList(GraphEdge),
    all_nodes: std.ArrayList(NodeData),

    pub fn deinit(self: *ModelHierarchyGraph) void {
        self.arena.deinit();
    }
};

// ============================================================================
// 2. 模块一：模型计算图与模块层级提取器 (Graph / Model Hierarchy Extractor)
//    - 输入: *autodiff.Graph
//    - 输出: 约定的中间数据结构 ModelHierarchyGraph 或 递归 JSON 字符串
// ============================================================================
pub const graph_ir = struct {
    fn formatShapeStr(allocator: std.mem.Allocator, t: *Tensor) ![]const u8 {
        var shape_buf: [64]u8 = undefined;
        var shape_len: usize = 0;
        shape_buf[0] = '[';
        shape_len += 1;
        for (0..t.shape.len) |d| {
            if (d > 0) {
                shape_buf[shape_len] = ',';
                shape_buf[shape_len + 1] = ' ';
                shape_len += 2;
            }
            const part = std.fmt.bufPrint(shape_buf[shape_len..], "{d}", .{t.shape.dims[d]}) catch "";
            shape_len += part.len;
        }
        shape_buf[shape_len] = ']';
        shape_len += 1;
        return allocator.dupe(u8, shape_buf[0..shape_len]);
    }

    fn buildNodeData(
        graph: *Graph,
        t: *Tensor,
        id: usize,
        param_idx: *usize,
        input_idx: *usize,
        op_idx: *usize,
        arena: std.mem.Allocator,
    ) !NodeData {
        const shape_str = try formatShapeStr(arena, t);
        const shape_dims = try arena.dupe(usize, t.shape.dims[0..t.shape.len]);
        const elements = t.data.len;
        const bytes = elements * @sizeOf(f32);

        // 1. 算子中间输出/激活值
        if (t.creator) |creator_op| {
            const op_name = @tagName(creator_op.op_type);
            var name_buf: [64]u8 = undefined;
            const name = if (t.name) |n| try arena.dupe(u8, n) else try arena.dupe(u8, std.fmt.bufPrint(&name_buf, "Node_{s}_{d}", .{ op_name, op_idx.* }) catch "Node_Op");
            op_idx.* += 1;

            var strat_buf: [64]u8 = undefined;
            const strat = try arena.dupe(u8, std.fmt.bufPrint(&strat_buf, "produced by {s}", .{op_name}) catch "op output");

            return .{
                .id = id,
                .name = name,
                .leaf_name = "",
                .path = "",
                .kind = "Activation",
                .shape_str = shape_str,
                .shape_dims = shape_dims,
                .elements = elements,
                .bytes = bytes,
                .status = "OP_OUTPUT",
                .inferred_act = try arena.dupe(u8, op_name),
                .strategy = strat,
                .op = try arena.dupe(u8, op_name),
                .inputs = .empty,
                .outputs = .empty,
            };
        }

        // 2. 外部输入张量
        if (!t.requires_grad) {
            var name_buf: [64]u8 = undefined;
            const name = if (t.name) |n| try arena.dupe(u8, n) else try arena.dupe(u8, std.fmt.bufPrint(&name_buf, "Input_{d}", .{input_idx.*}) catch "Input");
            input_idx.* += 1;

            return .{
                .id = id,
                .name = name,
                .leaf_name = "",
                .path = "",
                .kind = "Input",
                .shape_str = shape_str,
                .shape_dims = shape_dims,
                .elements = elements,
                .bytes = bytes,
                .status = "INPUT",
                .inferred_act = "N/A",
                .strategy = "user input / constant",
                .op = null,
                .inputs = .empty,
                .outputs = .empty,
            };
        }

        // 3. 模型可学习参数 (Weights & Biases)
        var name_buf: [64]u8 = undefined;
        const name = if (t.name) |n| try arena.dupe(u8, n) else try arena.dupe(u8, std.fmt.bufPrint(&name_buf, "Param_{d}", .{param_idx.*}) catch "Param");
        param_idx.* += 1;

        if (t.is_custom_initialized) {
            return .{
                .id = id,
                .name = name,
                .leaf_name = "",
                .path = "",
                .kind = "Param",
                .shape_str = shape_str,
                .shape_dims = shape_dims,
                .elements = elements,
                .bytes = bytes,
                .status = "CUSTOM_INIT",
                .inferred_act = "N/A",
                .strategy = "user-defined customInit",
                .op = null,
                .inputs = .empty,
                .outputs = .empty,
            };
        }

        if (t.shape.len == 1 or (t.shape.len == 2 and t.shape.dims[0] == 1)) {
            return .{
                .id = id,
                .name = name,
                .leaf_name = "",
                .path = "",
                .kind = "Param",
                .shape_str = shape_str,
                .shape_dims = shape_dims,
                .elements = elements,
                .bytes = bytes,
                .status = "AUTO_GRAPH",
                .inferred_act = "bias",
                .strategy = "zeros (0.0)",
                .op = null,
                .inputs = .empty,
                .outputs = .empty,
            };
        }

        const act = graph.detectConsumerActivation(t);
        const gain = init_mod.calculateGain(act);
        const act_name = switch (act) {
            .relu => "ReLU",
            .tanh => "Tanh",
            .sigmoid => "Sigmoid",
            .gelu => "GELU",
            .silu => "SiLU",
            .selu => "SELU",
            .leaky_relu => "LeakyReLU",
            .linear => "Linear (None)",
        };

        var strat_buf: [64]u8 = undefined;
        const strat = switch (act) {
            .tanh, .sigmoid => try arena.dupe(u8, std.fmt.bufPrint(&strat_buf, "Xavier Normal (gain={d:.3})", .{gain}) catch "Xavier Normal"),
            .selu => try arena.dupe(u8, "LeCun Normal"),
            else => try arena.dupe(u8, std.fmt.bufPrint(&strat_buf, "He Normal (gain={d:.3})", .{gain}) catch "He Normal"),
        };

        return .{
            .id = id,
            .name = name,
            .leaf_name = "",
            .path = "",
            .kind = "Param",
            .shape_str = shape_str,
            .shape_dims = shape_dims,
            .elements = elements,
            .bytes = bytes,
            .status = "AUTO_GRAPH",
            .inferred_act = try arena.dupe(u8, act_name),
            .strategy = strat,
            .op = null,
            .inputs = .empty,
            .outputs = .empty,
        };
    }

    fn aggregateMetrics(node: *ModuleNode) void {
        var total_params: usize = 0;
        var total_bytes: usize = 0;
        var param_count: usize = 0;
        var node_count: usize = node.nodes.items.len;

        for (node.nodes.items) |leaf| {
            total_bytes += leaf.bytes;
            if (std.mem.eql(u8, leaf.kind, "Param")) {
                total_params += leaf.elements;
                param_count += 1;
            }
        }

        for (node.children.items) |child| {
            aggregateMetrics(child);
            total_params += child.total_params;
            total_bytes += child.total_bytes;
            node_count += child.node_count;
        }

        node.total_params = total_params;
        node.total_bytes = total_bytes;
        node.param_count = param_count;
        node.node_count = node_count;
    }

    /// 解析 Graph 并构建约定的中间数据结构 ModelHierarchyGraph
    pub fn build(graph: *Graph, allocator: std.mem.Allocator) !ModelHierarchyGraph {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const arena_alloc = arena.allocator();

        var all_nodes: std.ArrayList(NodeData) = .empty;
        var edges: std.ArrayList(GraphEdge) = .empty;

        var visited = std.AutoHashMap(*Tensor, usize).init(arena_alloc);
        var param_idx: usize = 0;
        var input_idx: usize = 0;
        var op_idx: usize = 0;

        // 1. 优先按前向计算图执行拓扑收集节点
        for (graph.ops.items) |op| {
            for (op.inputs) |t| {
                if (visited.contains(t)) continue;
                const id = all_nodes.items.len;
                const node = try buildNodeData(graph, t, id, &param_idx, &input_idx, &op_idx, arena_alloc);
                try all_nodes.append(arena_alloc, node);
                try visited.put(t, id);
            }
            for (op.outputs) |t| {
                if (visited.contains(t)) continue;
                const id = all_nodes.items.len;
                const node = try buildNodeData(graph, t, id, &param_idx, &input_idx, &op_idx, arena_alloc);
                try all_nodes.append(arena_alloc, node);
                try visited.put(t, id);
            }
        }

        // 补充尚未参与计算的悬空权重或参数
        for (graph.tensors.items) |t| {
            if (visited.contains(t)) continue;
            const id = all_nodes.items.len;
            const node = try buildNodeData(graph, t, id, &param_idx, &input_idx, &op_idx, arena_alloc);
            try all_nodes.append(arena_alloc, node);
            try visited.put(t, id);
        }

        // 2. 提取计算图连通关系有向边 (Edges)
        for (graph.ops.items) |op| {
            const op_name = @tagName(op.op_type);
            for (op.inputs) |in_t| {
                for (op.outputs) |out_t| {
                    const in_id = visited.get(in_t) orelse continue;
                    const out_id = visited.get(out_t) orelse continue;

                    const in_node = &all_nodes.items[in_id];
                    const out_node = &all_nodes.items[out_id];

                    try edges.append(arena_alloc, .{
                        .from = in_node.name,
                        .to = out_node.name,
                        .op = op_name,
                        .shape = out_node.shape_str,
                    });

                    try in_node.outputs.append(arena_alloc, out_node.name);
                    try out_node.inputs.append(arena_alloc, in_node.name);
                    if (out_node.op == null) {
                        out_node.op = op_name;
                    }
                }
            }
        }

        // 3. 构建递归模块树 (Hierarchical Module Tree)
        const root = try ModuleNode.init(arena_alloc, "root", "");

        for (all_nodes.items) |*node| {
            var it = std.mem.splitScalar(u8, node.name, '.');
            var parts: std.ArrayList([]const u8) = .empty;
            while (it.next()) |part| {
                try parts.append(arena_alloc, part);
            }

            if (parts.items.len > 1) {
                var curr = root;
                for (parts.items[0 .. parts.items.len - 1]) |part| {
                    var found: ?*ModuleNode = null;
                    for (curr.children.items) |child| {
                        if (std.mem.eql(u8, child.name, part)) {
                            found = child;
                            break;
                        }
                    }
                    if (found) |child| {
                        curr = child;
                    } else {
                        const child_path = if (curr.path.len == 0)
                            try arena_alloc.dupe(u8, part)
                        else
                            try std.fmt.allocPrint(arena_alloc, "{s}.{s}", .{ curr.path, part });
                        const new_child = try ModuleNode.init(arena_alloc, try arena_alloc.dupe(u8, part), child_path);
                        try curr.children.append(arena_alloc, new_child);
                        curr = new_child;
                    }
                }
                node.leaf_name = parts.items[parts.items.len - 1];
                node.path = curr.path;
                try curr.nodes.append(arena_alloc, node.*);
            } else {
                var found_unscoped: ?*ModuleNode = null;
                for (root.children.items) |child| {
                    if (std.mem.eql(u8, child.name, "(unscoped)")) {
                        found_unscoped = child;
                        break;
                    }
                }
                const unscoped = if (found_unscoped) |u| u else blk: {
                    const new_u = try ModuleNode.init(arena_alloc, "(unscoped)", "(unscoped)");
                    try root.children.append(arena_alloc, new_u);
                    break :blk new_u;
                };
                node.leaf_name = node.name;
                node.path = "(unscoped)";
                try unscoped.nodes.append(arena_alloc, node.*);
            }
        }

        // 4. 后序递归汇聚各模块子树的参数量和内存开销
        aggregateMetrics(root);

        // 5. 生成全局指标概览 (Summary)
        var summary = Summary{
            .total_params = root.total_params,
            .total_bytes = root.total_bytes,
            .total_nodes = all_nodes.items.len,
        };

        for (all_nodes.items) |n| {
            if (std.mem.eql(u8, n.kind, "Param")) {
                summary.param_nodes += 1;
                if (std.mem.eql(u8, n.status, "CUSTOM_INIT")) {
                    summary.custom_init_count += 1;
                } else if (std.mem.eql(u8, n.status, "AUTO_GRAPH")) {
                    summary.auto_graph_count += 1;
                }
            } else if (std.mem.eql(u8, n.kind, "Input")) {
                summary.input_nodes += 1;
            } else {
                summary.activation_nodes += 1;
            }
        }

        return .{
            .arena = arena,
            .summary = summary,
            .root = root,
            .edges = edges,
            .all_nodes = all_nodes,
        };
    }

    fn writeJsonString(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, str: []const u8) !void {
        try buf.append(allocator, '"');
        for (str) |c| {
            switch (c) {
                '"' => try buf.appendSlice(allocator, "\\\""),
                '\\' => try buf.appendSlice(allocator, "\\\\"),
                '\n' => try buf.appendSlice(allocator, "\\n"),
                '\r' => try buf.appendSlice(allocator, "\\r"),
                '\t' => try buf.appendSlice(allocator, "\\t"),
                else => try buf.append(allocator, c),
            }
        }
        try buf.append(allocator, '"');
    }

    fn serializeLeafNode(leaf: *const NodeData, buf: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        try buf.appendSlice(allocator, "{");
        try buf.print(allocator, "\"id\": {d},", .{leaf.id});
        try buf.appendSlice(allocator, "\"name\": ");
        try writeJsonString(buf, allocator, leaf.name);
        try buf.appendSlice(allocator, ",\"leaf_name\": ");
        try writeJsonString(buf, allocator, leaf.leaf_name);
        try buf.appendSlice(allocator, ",\"path\": ");
        try writeJsonString(buf, allocator, leaf.path);
        try buf.appendSlice(allocator, ",\"kind\": ");
        try writeJsonString(buf, allocator, leaf.kind);
        try buf.appendSlice(allocator, ",\"shape\": ");
        try writeJsonString(buf, allocator, leaf.shape_str);
        try buf.appendSlice(allocator, ",\"dims\": [");
        for (leaf.shape_dims, 0..) |dim, d| {
            if (d > 0) try buf.appendSlice(allocator, ",");
            try buf.print(allocator, "{d}", .{dim});
        }
        try buf.appendSlice(allocator, "],");
        try buf.print(allocator, "\"elements\": {d},", .{leaf.elements});
        try buf.print(allocator, "\"bytes\": {d},", .{leaf.bytes});
        try buf.appendSlice(allocator, "\"status\": ");
        try writeJsonString(buf, allocator, leaf.status);
        try buf.appendSlice(allocator, ",\"act\": ");
        try writeJsonString(buf, allocator, leaf.inferred_act);
        try buf.appendSlice(allocator, ",\"strategy\": ");
        try writeJsonString(buf, allocator, leaf.strategy);
        if (leaf.op) |op| {
            try buf.appendSlice(allocator, ",\"op\": ");
            try writeJsonString(buf, allocator, op);
        } else {
            try buf.appendSlice(allocator, ",\"op\": null");
        }
        try buf.appendSlice(allocator, ",\"inputs\": [");
        for (leaf.inputs.items, 0..) |inp, j| {
            if (j > 0) try buf.appendSlice(allocator, ",");
            try writeJsonString(buf, allocator, inp);
        }
        try buf.appendSlice(allocator, "],\"outputs\": [");
        for (leaf.outputs.items, 0..) |outp, j| {
            if (j > 0) try buf.appendSlice(allocator, ",");
            try writeJsonString(buf, allocator, outp);
        }
        try buf.appendSlice(allocator, "]}");
    }

    fn serializeModuleNode(node: *const ModuleNode, buf: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        try buf.appendSlice(allocator, "{");
        try buf.appendSlice(allocator, "\"name\": ");
        try writeJsonString(buf, allocator, node.name);
        try buf.appendSlice(allocator, ",\"path\": ");
        try writeJsonString(buf, allocator, node.path);
        try buf.appendSlice(allocator, ",\"kind\": \"module\",");
        try buf.print(allocator, "\"total_params\": {d},", .{node.total_params});
        try buf.print(allocator, "\"total_bytes\": {d},", .{node.total_bytes});
        try buf.print(allocator, "\"param_count\": {d},", .{node.param_count});
        try buf.print(allocator, "\"node_count\": {d},", .{node.node_count});

        try buf.appendSlice(allocator, "\"children\": [");
        for (node.children.items, 0..) |child, i| {
            if (i > 0) try buf.appendSlice(allocator, ",");
            try serializeModuleNode(child, buf, allocator);
        }
        try buf.appendSlice(allocator, "],");

        try buf.appendSlice(allocator, "\"nodes\": [");
        for (node.nodes.items, 0..) |leaf, i| {
            if (i > 0) try buf.appendSlice(allocator, ",");
            try serializeLeafNode(&leaf, buf, allocator);
        }
        try buf.appendSlice(allocator, "]");
        try buf.appendSlice(allocator, "}");
    }

    /// 将 ModelHierarchyGraph 中间结构序列化为递归的 JSON 数据字符串
    pub fn serializeJson(model_graph: *const ModelHierarchyGraph, allocator: std.mem.Allocator) ![]const u8 {
        var json_buf: std.ArrayList(u8) = .empty;
        errdefer json_buf.deinit(allocator);

        try json_buf.appendSlice(allocator, "{\n  \"version\": \"1.0\",\n  \"summary\": {");
        try json_buf.print(allocator,
            \\ "total_params": {d}, "total_bytes": {d}, "param_nodes": {d}, "input_nodes": {d}, "activation_nodes": {d}, "custom_init_count": {d}, "auto_graph_count": {d}, "total_nodes": {d}
        , .{
            model_graph.summary.total_params,
            model_graph.summary.total_bytes,
            model_graph.summary.param_nodes,
            model_graph.summary.input_nodes,
            model_graph.summary.activation_nodes,
            model_graph.summary.custom_init_count,
            model_graph.summary.auto_graph_count,
            model_graph.summary.total_nodes,
        });
        try json_buf.appendSlice(allocator, "},\n  \"root\": ");
        try serializeModuleNode(model_graph.root, &json_buf, allocator);
        try json_buf.appendSlice(allocator, ",\n  \"edges\": [");
        for (model_graph.edges.items, 0..) |edge, i| {
            if (i > 0) try json_buf.appendSlice(allocator, ",");
            try json_buf.appendSlice(allocator, "\n    {\"from\": ");
            try writeJsonString(&json_buf, allocator, edge.from);
            try json_buf.appendSlice(allocator, ", \"to\": ");
            try writeJsonString(&json_buf, allocator, edge.to);
            try json_buf.appendSlice(allocator, ", \"op\": ");
            try writeJsonString(&json_buf, allocator, edge.op);
            try json_buf.appendSlice(allocator, ", \"shape\": ");
            try writeJsonString(&json_buf, allocator, edge.shape);
            try json_buf.appendSlice(allocator, "}");
        }
        try json_buf.appendSlice(allocator, "\n  ]\n}");

        return json_buf.toOwnedSlice(allocator);
    }

    /// 从 Graph 直接生成递归 JSON
    pub fn generateJson(graph: *Graph, allocator: std.mem.Allocator) ![]const u8 {
        var model_graph = try build(graph, allocator);
        defer model_graph.deinit();
        return serializeJson(&model_graph, allocator);
    }

    /// 将计算图结构直接导出为独立的递归 JSON 文件
    pub fn exportJson(graph: *Graph, file_path: []const u8, allocator: std.mem.Allocator) !void {
        const json_content = try graph_ir.generateJson(graph, allocator);
        defer allocator.free(json_content);

        const path_z = try allocator.dupeZ(u8, file_path);
        defer allocator.free(path_z);

        const file = std.c.fopen(path_z.ptr, "wb") orelse return error.CannotOpenFile;
        defer _ = std.c.fclose(file);

        const written = std.c.fwrite(json_content.ptr, 1, json_content.len, file);
        if (written < json_content.len) return error.WriteFailed;
    }
};

// ============================================================================
// 3. 模块二：独立 HTML 可视化报告生成器 (Standalone HTML Report Visualizer)
//    - 输入: 约定的中间数据结构 (json_str: []const u8 或 *const ModelHierarchyGraph)
//    - 输出: 自包含的前后端分离 HTML 报告文档 (纯前端 JS 解析渲染)
//    - 完全独立，不挂在 Graph 下，亦不依赖 autodiff.Graph
// ============================================================================
pub const html_report = struct {
    /// 依据约定的递归 JSON 字符串生成自包含 HTML 报告文档
    pub fn renderFromJson(json_str: []const u8, allocator: std.mem.Allocator) ![]const u8 {
        var html_buf: std.ArrayList(u8) = .empty;
        errdefer html_buf.deinit(allocator);

        // 1. HTML Header & Styling
        try html_buf.appendSlice(allocator,
            \\<!DOCTYPE html>
            \\<html lang="zh-CN">
            \\<head>
            \\  <meta charset="UTF-8">
            \\  <meta name="viewport" content="width=device-width, initial-scale=1.0">
            \\  <title>ZNN Model Architecture & Graph Visualizer</title>
            \\  <style>
            \\    :root {
            \\      --bg-primary: #090d16;
            \\      --bg-secondary: #0f172a;
            \\      --bg-card: #1e293b;
            \\      --bg-card-hover: #273549;
            \\      --border-color: #334155;
            \\      --border-focus: #38bdf8;
            \\      --text-main: #f8fafc;
            \\      --text-sub: #94a3b8;
            \\      --badge-param: #0284c7;
            \\      --badge-input: #d97706;
            \\      --badge-act: #7c3aed;
            \\      --badge-auto: #059669;
            \\      --badge-custom: #e11d48;
            \\      --badge-op: #6366f1;
            \\      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            \\      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            \\    }
            \\    * { box-sizing: border-box; margin: 0; padding: 0; }
            \\    body {
            \\      background-color: var(--bg-primary);
            \\      color: var(--text-main);
            \\      font-family: var(--font-sans);
            \\      padding: 32px 24px;
            \\      line-height: 1.5;
            \\    }
            \\    .container { max-width: 1440px; margin: 0 auto; }
            \\    header { margin-bottom: 24px; }
            \\    .title-row { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px; margin-bottom: 8px; }
            \\    h1 { font-size: 26px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 10px; }
            \\    .subtitle { color: var(--text-sub); font-size: 14px; }
            \\    .header-actions { display: flex; gap: 10px; }
            \\    
            \\    /* KPI Grid */
            \\    .stats-grid {
            \\      display: grid;
            \\      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            \\      gap: 16px;
            \\      margin-bottom: 24px;
            \\    }
            \\    .stat-card {
            \\      background: var(--bg-secondary);
            \\      border: 1px solid var(--border-color);
            \\      border-radius: 10px;
            \\      padding: 16px 20px;
            \\      display: flex;
            \\      flex-direction: column;
            \\      gap: 4px;
            \\      transition: transform 0.2s, border-color 0.2s;
            \\    }
            \\    .stat-card:hover { transform: translateY(-2px); border-color: #475569; }
            \\    .stat-label { font-size: 12px; text-transform: uppercase; color: var(--text-sub); font-weight: 600; letter-spacing: 0.5px; }
            \\    .stat-value { font-size: 24px; font-weight: 700; color: var(--text-main); font-family: var(--font-mono); }
            \\    .stat-sub { font-size: 12px; color: #38bdf8; }
            \\
            \\    /* Controls Bar */
            \\    .controls-bar {
            \\      background: var(--bg-secondary);
            \\      border: 1px solid var(--border-color);
            \\      border-radius: 10px;
            \\      padding: 12px 18px;
            \\      display: flex;
            \\      align-items: center;
            \\      justify-content: space-between;
            \\      flex-wrap: wrap;
            \\      gap: 12px;
            \\      margin-bottom: 20px;
            \\    }
            \\    .search-box { position: relative; flex: 1; min-width: 280px; }
            \\    .search-input {
            \\      width: 100%;
            \\      background: var(--bg-card);
            \\      border: 1px solid var(--border-color);
            \\      border-radius: 6px;
            \\      padding: 8px 12px 8px 34px;
            \\      color: var(--text-main);
            \\      font-size: 13px;
            \\      outline: none;
            \\      transition: border-color 0.2s;
            \\    }
            \\    .search-input:focus { border-color: var(--border-focus); }
            \\    .search-icon { position: absolute; left: 10px; top: 9px; color: var(--text-sub); font-size: 14px; }
            \\    .btn-group { display: flex; gap: 8px; flex-wrap: wrap; }
            \\    .btn {
            \\      background: var(--bg-card);
            \\      border: 1px solid var(--border-color);
            \\      color: var(--text-main);
            \\      padding: 7px 14px;
            \\      border-radius: 6px;
            \\      font-size: 12px;
            \\      font-weight: 500;
            \\      cursor: pointer;
            \\      transition: all 0.2s;
            \\      display: inline-flex;
            \\      align-items: center;
            \\      gap: 6px;
            \\    }
            \\    .btn:hover { background: var(--bg-card-hover); border-color: #475569; }
            \\    .btn.active { background: #0284c7; border-color: #38bdf8; color: #fff; }
            \\    .btn-secondary { background: rgba(51, 65, 85, 0.4); border-color: #334155; }
            \\
            \\    /* View Switcher Tabs */
            \\    .view-tabs { display: flex; gap: 8px; margin-bottom: 16px; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }
            \\    .view-tab {
            \\      background: transparent;
            \\      border: none;
            \\      color: var(--text-sub);
            \\      font-size: 14px;
            \\      font-weight: 600;
            \\      padding: 6px 14px;
            \\      cursor: pointer;
            \\      border-radius: 6px;
            \\      transition: all 0.2s;
            \\    }
            \\    .view-tab.active { background: var(--bg-card); color: #38bdf8; }
            \\
            \\    /* Recursive Tree & Module Group */
            \\    .hierarchy-container { display: flex; flex-direction: column; gap: 12px; }
            \\    details.module-group {
            \\      background: var(--bg-secondary);
            \\      border: 1px solid var(--border-color);
            \\      border-radius: 8px;
            \\      overflow: hidden;
            \\      transition: border-color 0.2s;
            \\    }
            \\    details.module-group[open] { border-color: #475569; }
            \\    summary.module-header {
            \\      background: var(--bg-card);
            \\      padding: 10px 16px;
            \\      cursor: pointer;
            \\      font-weight: 600;
            \\      display: flex;
            \\      align-items: center;
            \\      justify-content: space-between;
            \\      user-select: none;
            \\      list-style: none;
            \\    }
            \\    summary.module-header::-webkit-details-marker { display: none; }
            \\    .module-title-box { display: flex; align-items: center; gap: 8px; }
            \\    .chevron { transition: transform 0.2s; font-size: 11px; color: var(--text-sub); }
            \\    details[open] > summary .chevron { transform: rotate(90deg); }
            \\    .module-name { color: #38bdf8; font-family: var(--font-mono); font-size: 14px; }
            \\    .module-path { color: var(--text-sub); font-size: 12px; font-family: var(--font-mono); margin-left: 6px; opacity: 0.8; }
            \\    .module-meta { font-size: 12px; color: var(--text-sub); display: flex; gap: 10px; font-family: var(--font-mono); align-items: center; }
            \\    .module-content { padding: 12px 16px; display: flex; flex-direction: column; gap: 10px; }
            \\
            \\    /* Node Table */
            \\    table.node-table { width: 100%; border-collapse: collapse; font-size: 13px; }
            \\    th {
            \\      background: #131d2e;
            \\      color: var(--text-sub);
            \\      font-weight: 600;
            \\      text-align: left;
            \\      padding: 8px 12px;
            \\      border-bottom: 1px solid var(--border-color);
            \\      font-size: 11px;
            \\      text-transform: uppercase;
            \\      letter-spacing: 0.5px;
            \\    }
            \\    td { padding: 9px 12px; border-bottom: 1px solid rgba(51, 65, 85, 0.4); vertical-align: middle; }
            \\    tr:last-child td { border-bottom: none; }
            \\    tr:hover td { background: rgba(56, 189, 248, 0.04); }
            \\    .node-name { font-family: var(--font-mono); font-weight: 600; color: #f1f5f9; cursor: pointer; }
            \\    .node-name:hover { color: #38bdf8; text-decoration: underline; }
            \\    .node-shape { font-family: var(--font-mono); color: #cbd5e1; }
            \\    .badge {
            \\      display: inline-block;
            \\      padding: 2px 8px;
            \\      border-radius: 4px;
            \\      font-size: 11px;
            \\      font-weight: 600;
            \\      font-family: var(--font-mono);
            \\      text-align: center;
            \\    }
            \\    .badge-param { background: rgba(2, 132, 199, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
            \\    .badge-input { background: rgba(217, 119, 6, 0.2); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
            \\    .badge-act { background: rgba(124, 58, 237, 0.2); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.3); }
            \\    .badge-auto { background: rgba(5, 150, 105, 0.2); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
            \\    .badge-custom { background: rgba(225, 29, 72, 0.2); color: #fb7185; border: 1px solid rgba(251, 113, 133, 0.3); }
            \\    .badge-op { background: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid rgba(129, 140, 248, 0.3); }
            \\    .strategy-col { font-family: var(--font-mono); font-size: 12px; color: #94a3b8; }
            \\    .io-tag { font-size: 11px; color: var(--text-sub); font-family: var(--font-mono); background: #1e293b; padding: 2px 6px; border-radius: 4px; }
            \\
            \\    /* Graph Pipeline Flow View */
            \\    .graph-flow-container {
            \\      background: var(--bg-secondary);
            \\      border: 1px solid var(--border-color);
            \\      border-radius: 8px;
            \\      padding: 20px;
            \\      display: flex;
            \\      flex-direction: column;
            \\      gap: 14px;
            \\    }
            \\    .edge-card {
            \\      background: var(--bg-card);
            \\      border: 1px solid var(--border-color);
            \\      border-radius: 6px;
            \\      padding: 10px 16px;
            \\      display: flex;
            \\      align-items: center;
            \\      justify-content: space-between;
            \\      font-family: var(--font-mono);
            \\      font-size: 13px;
            \\    }
            \\    .edge-from { color: #fbbf24; }
            \\    .edge-arrow { color: #94a3b8; display: flex; align-items: center; gap: 8px; }
            \\    .edge-to { color: #c084fc; font-weight: 600; }
            \\
            \\    /* Inspector Modal */
            \\    .modal-overlay {
            \\      position: fixed;
            \\      top: 0; left: 0; right: 0; bottom: 0;
            \\      background: rgba(0, 0, 0, 0.7);
            \\      display: flex;
            \\      align-items: center;
            \\      justify-content: center;
            \\      z-index: 1000;
            \\      backdrop-filter: blur(4px);
            \\    }
            \\    .modal-box {
            \\      background: var(--bg-secondary);
            \\      border: 1px solid var(--border-focus);
            \\      border-radius: 12px;
            \\      width: 90%;
            \\      max-width: 680px;
            \\      padding: 24px;
            \\      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            \\    }
            \\    .modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
            \\    .modal-title { font-size: 18px; font-weight: 700; color: #38bdf8; font-family: var(--font-mono); }
            \\    .modal-close { background: none; border: none; color: var(--text-sub); font-size: 20px; cursor: pointer; }
            \\    .modal-body { display: flex; flex-direction: column; gap: 12px; font-size: 13px; }
            \\    .detail-row { display: flex; justify-content: space-between; border-bottom: 1px solid rgba(51, 65, 85, 0.4); padding: 8px 0; }
            \\    .detail-key { color: var(--text-sub); }
            \\    .detail-val { font-family: var(--font-mono); color: var(--text-main); font-weight: 600; }
            \\
            \\    footer { margin-top: 36px; text-align: center; font-size: 12px; color: #64748b; }
            \\  </style>
            \\</head>
            \\<body>
            \\<div class="container">
            \\  <header>
            \\    <div class="title-row">
            \\      <div>
            \\        <h1>⚡ ZNN Model Architecture & Graph Visualizer</h1>
            \\        <div class="subtitle">Decoupled Interactive Graph Inspection · Recursive Module Tree · Autodiff DAG</div>
            \\      </div>
            \\      <div class="header-actions">
            \\        <button class="btn btn-secondary" onclick="copyModelJson()">📋 Copy JSON</button>
            \\        <button class="btn btn-secondary" onclick="downloadModelJson()">💾 Download JSON</button>
            \\      </div>
            \\    </div>
            \\  </header>
            \\
            \\  <!-- KPI Grid (Populated dynamically by Frontend from JSON) -->
            \\  <div class="stats-grid" id="kpi-grid"></div>
            \\
            \\  <!-- View Switcher -->
            \\  <div class="view-tabs">
            \\    <button class="view-tab active" id="tab-tree" onclick="switchView('tree')">📁 Module Hierarchy Tree</button>
            \\    <button class="view-tab" id="tab-graph" onclick="switchView('graph')">🔗 Computation Graph & Edges</button>
            \\  </div>
            \\
            \\  <!-- Controls -->
            \\  <div class="controls-bar">
            \\    <div class="search-box">
            \\      <span class="search-icon">🔍</span>
            \\      <input type="text" id="search-input" class="search-input" placeholder="Search by module path, node name, op, shape, or strategy..." oninput="onSearchInput()">
            \\    </div>
            \\    <div class="btn-group">
            \\      <button class="btn active" data-kind="all" onclick="setKindFilter('all', this)">All Nodes (<span id="count-all">0</span>)</button>
            \\      <button class="btn" data-kind="Param" onclick="setKindFilter('Param', this)">Params (<span id="count-params">0</span>)</button>
            \\      <button class="btn" data-kind="Input" onclick="setKindFilter('Input', this)">Inputs (<span id="count-inputs">0</span>)</button>
            \\      <button class="btn" data-kind="Activation" onclick="setKindFilter('Activation', this)">Activations (<span id="count-acts">0</span>)</button>
            \\      <button class="btn" onclick="expandAll()">Expand All</button>
            \\      <button class="btn" onclick="collapseAll()">Collapse All</button>
            \\    </div>
            \\  </div>
            \\
            \\  <!-- Tree Container (Populated dynamically by Frontend) -->
            \\  <div class="hierarchy-container" id="tree-container"></div>
            \\
            \\  <!-- Graph Pipeline Flow Container -->
            \\  <div class="graph-flow-container" id="graph-flow-container" style="display: none;"></div>
            \\
            \\  <!-- Node Inspector Modal -->
            \\  <div class="modal-overlay" id="node-modal" style="display: none;" onclick="closeModal(event)">
            \\    <div class="modal-box" onclick="event.stopPropagation()">
            \\      <div class="modal-header">
            \\        <div class="modal-title" id="modal-node-name">Node Inspector</div>
            \\        <button class="modal-close" onclick="closeModal()">&times;</button>
            \\      </div>
            \\      <div class="modal-body" id="modal-content"></div>
            \\    </div>
            \\  </div>
            \\
            \\  <footer>Generated automatically by ZNN Autodiff Engine · Pure Frontend-Backend Decoupled Architecture</footer>
            \\</div>
            \\
            \\<!-- BACKEND PAYLOAD: Fully Recursive Model Graph JSON -->
            \\<script id="znn-model-graph-data" type="application/json">
        );

        // 2. 注入约定的递归结构 JSON 数据包
        try html_buf.appendSlice(allocator, json_str);

        // 3. 独立的前端解析引擎 JavaScript
        try html_buf.appendSlice(allocator,
            \\
            \\</script>
            \\
            \\<script>
            \\/**
            \\ * ZNN Frontend Visualizer Engine
            \\ * Pure client-side parsing of the recursive model graph structure.
            \\ */
            \\const RAW_DATA = JSON.parse(document.getElementById('znn-model-graph-data').textContent);
            \\let currentKindFilter = 'all';
            \\let currentQuery = '';
            \\let activeView = 'tree';
            \\
            \\function formatNumber(num) {
            \\  return (num || 0).toLocaleString();
            \\}
            \\
            \\function formatBytes(bytes) {
            \\  if (!bytes || bytes === 0) return '0 B';
            \\  if (bytes < 1024) return bytes + ' B';
            \\  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            \\  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
            \\}
            \\
            \\// 1. 渲染全局 KPI 指标卡片
            \\function renderKPIs() {
            \\  const s = RAW_DATA.summary || {};
            \\  const mbEst = ((s.total_bytes || 0) / (1024 * 1024)).toFixed(2);
            \\  const kpiContainer = document.getElementById('kpi-grid');
            \\  kpiContainer.innerHTML = `
            \\    <div class="stat-card">
            \\      <div class="stat-label">Total Parameters</div>
            \\      <div class="stat-value">${formatNumber(s.total_params)}</div>
            \\      <div class="stat-sub">~${mbEst} MB Total Footprint</div>
            \\    </div>
            \\    <div class="stat-card">
            \\      <div class="stat-label">Param Tensors</div>
            \\      <div class="stat-value">${formatNumber(s.param_nodes)}</div>
            \\      <div class="stat-sub">${s.auto_graph_count || 0} Auto-Graph · ${s.custom_init_count || 0} Custom</div>
            \\    </div>
            \\    <div class="stat-card">
            \\      <div class="stat-label">Graph Nodes</div>
            \\      <div class="stat-value">${formatNumber(s.total_nodes)}</div>
            \\      <div class="stat-sub">${s.input_nodes || 0} Inputs · ${s.activation_nodes || 0} Activations</div>
            \\    </div>
            \\    <div class="stat-card">
            \\      <div class="stat-label">Memory Tracked</div>
            \\      <div class="stat-value">${formatBytes(s.total_bytes)}</div>
            \\      <div class="stat-sub">Tracked by Graph Arena</div>
            \\    </div>
            \\  `;
            \\
            \\  document.getElementById('count-all').textContent = s.total_nodes || 0;
            \\  document.getElementById('count-params').textContent = s.param_nodes || 0;
            \\  document.getElementById('count-inputs').textContent = s.input_nodes || 0;
            \\  document.getElementById('count-acts').textContent = s.activation_nodes || 0;
            \\}
            \\
            \\// 2. 递归过滤与匹配
            \\function nodeMatches(node) {
            \\  const matchesKind = (currentKindFilter === 'all' || node.kind === currentKindFilter);
            \\  if (!matchesKind) return false;
            \\  if (!currentQuery) return true;
            \\  const q = currentQuery.toLowerCase();
            \\  return (
            \\    node.name.toLowerCase().includes(q) ||
            \\    node.leaf_name.toLowerCase().includes(q) ||
            \\    node.shape.toLowerCase().includes(q) ||
            \\    (node.act && node.act.toLowerCase().includes(q)) ||
            \\    (node.strategy && node.strategy.toLowerCase().includes(q)) ||
            \\    (node.op && node.op.toLowerCase().includes(q))
            \\  );
            \\}
            \\
            \\function filterModule(mod) {
            \\  const filteredNodes = (mod.nodes || []).filter(nodeMatches);
            \\  const filteredChildren = [];
            \\  let totalParams = 0;
            \\  let totalBytes = 0;
            \\
            \\  filteredNodes.forEach(n => {
            \\    if (n.kind === 'Param') totalParams += n.elements;
            \\    totalBytes += n.bytes;
            \\  });
            \\
            \\  for (const child of (mod.children || [])) {
            \\    const filteredChild = filterModule(child);
            \\    if (filteredChild) {
            \\      filteredChildren.push(filteredChild);
            \\      totalParams += filteredChild.total_params;
            \\      totalBytes += filteredChild.total_bytes;
            \\    }
            \\  }
            \\
            \\  if (filteredNodes.length > 0 || filteredChildren.length > 0) {
            \\    return {
            \\      name: mod.name,
            \\      path: mod.path,
            \\      kind: mod.kind,
            \\      total_params: totalParams,
            \\      total_bytes: totalBytes,
            \\      children: filteredChildren,
            \\      nodes: filteredNodes,
            \\    };
            \\  }
            \\  return null;
            \\}
            \\
            \\// 3. 递归生成 DOM 代码
            \\function renderTableRows(nodes) {
            \\  return nodes.map(n => {
            \\    const kindBadge = n.kind === 'Param' ? 'badge-param' : (n.kind === 'Input' ? 'badge-input' : 'badge-act');
            \\    const statusBadge = n.status === 'CUSTOM_INIT' ? 'badge-custom' : (n.status === 'AUTO_GRAPH' ? 'badge-auto' : 'badge-op');
            \\    const ioInfo = `${(n.inputs || []).length} in / ${(n.outputs || []).length} out`;
            \\    const nodeJson = encodeURIComponent(JSON.stringify(n));
            \\    return `
            \\      <tr class="node-row" data-name="${n.name.toLowerCase()}" data-kind="${n.kind}">
            \\        <td><span class="node-name" onclick="openInspector('${nodeJson}')">${n.leaf_name || n.name}</span></td>
            \\        <td style="color: var(--text-sub); font-family: var(--font-mono); font-size: 11px;">${n.name}</td>
            \\        <td><span class="badge ${kindBadge}">${n.kind}</span></td>
            \\        <td class="node-shape">${n.shape}</td>
            \\        <td style="font-family: var(--font-mono);">${formatNumber(n.elements)}</td>
            \\        <td><span class="badge ${statusBadge}">${n.status}</span></td>
            \\        <td style="font-family: var(--font-mono);">${n.act || 'N/A'}</td>
            \\        <td class="strategy-col">${n.strategy || ''}</td>
            \\        <td><span class="io-tag">${ioInfo}</span></td>
            \\      </tr>
            \\    `;
            \\  }).join('');
            \\}
            \\
            \\function renderModuleHtml(mod) {
            \\  const hasChildren = mod.children && mod.children.length > 0;
            \\  const hasNodes = mod.nodes && mod.nodes.length > 0;
            \\  const title = mod.name === 'root' ? 'root (Model Root)' : mod.name;
            \\  const pathDisplay = mod.path ? `(${mod.path})` : '';
            \\  const metaInfo = `${formatNumber(mod.total_params)} params · ${formatBytes(mod.total_bytes)}`;
            \\
            \\  let innerHtml = '';
            \\  if (hasNodes) {
            \\    innerHtml += `
            \\      <table class="node-table">
            \\        <thead>
            \\          <tr>
            \\            <th>Leaf Name</th>
            \\            <th>Full Path</th>
            \\            <th>Kind</th>
            \\            <th>Shape</th>
            \\            <th>Elements</th>
            \\            <th>Status</th>
            \\            <th>Inferred Act / Op</th>
            \\            <th>Init Strategy</th>
            \\            <th>Edges</th>
            \\          </tr>
            \\        </thead>
            \\        <tbody>
            \\          ${renderTableRows(mod.nodes)}
            \\        </tbody>
            \\      </table>
            \\    `;
            \\  }
            \\
            \\  if (hasChildren) {
            \\    for (const child of mod.children) {
            \\      innerHtml += renderModuleHtml(child);
            \\    }
            \\  }
            \\
            \\  return `
            \\    <details class="module-group" open data-path="${mod.path || ''}">
            \\      <summary class="module-header">
            \\        <div class="module-title-box">
            \\          <span class="chevron">▶</span>
            \\          <span class="module-name">📁 ${title}</span>
            \\          <span class="module-path">${pathDisplay}</span>
            \\        </div>
            \\        <div class="module-meta">
            \\          <span>${metaInfo}</span>
            \\        </div>
            \\      </summary>
            \\      <div class="module-content">
            \\        ${innerHtml}
            \\      </div>
            \\    </details>
            \\  `;
            \\}
            \\
            \\function renderTree() {
            \\  const container = document.getElementById('tree-container');
            \\  const filteredRoot = filterModule(RAW_DATA.root);
            \\  if (!filteredRoot) {
            \\    container.innerHTML = '<div style="text-align: center; padding: 48px; color: var(--text-sub);">No nodes matched the filter criteria.</div>';
            \\    return;
            \\  }
            \\  container.innerHTML = renderModuleHtml(filteredRoot);
            \\}
            \\
            \\// 4. 渲染计算图连通边 (Graph Flow View)
            \\function renderGraphFlow() {
            \\  const container = document.getElementById('graph-flow-container');
            \\  const edges = RAW_DATA.edges || [];
            \\  if (edges.length === 0) {
            \\    container.innerHTML = '<div style="text-align: center; padding: 36px; color: var(--text-sub);">No graph computation edges recorded.</div>';
            \\    return;
            \\  }
            \\  container.innerHTML = edges.map((e, idx) => `
            \\    <div class="edge-card">
            \\      <span class="edge-from">🔹 ${e.from}</span>
            \\      <div class="edge-arrow">
            \\        <span class="badge badge-op">${e.op}</span>
            \\        <span>──(${e.shape})──►</span>
            \\      </div>
            \\      <span class="edge-to">🔸 ${e.to}</span>
            \\    </div>
            \\  `).join('');
            \\}
            \\
            \\// 5. 交互控制器与视图切换
            \\function switchView(view) {
            \\  activeView = view;
            \\  document.getElementById('tab-tree').classList.toggle('active', view === 'tree');
            \\  document.getElementById('tab-graph').classList.toggle('active', view === 'graph');
            \\  document.getElementById('tree-container').style.display = view === 'tree' ? 'flex' : 'none';
            \\  document.getElementById('graph-flow-container').style.display = view === 'graph' ? 'flex' : 'none';
            \\}
            \\
            \\function setKindFilter(kind, btn) {
            \\  currentKindFilter = kind;
            \\  document.querySelectorAll('.btn-group .btn[data-kind]').forEach(b => b.classList.remove('active'));
            \\  btn.classList.add('active');
            \\  renderTree();
            \\}
            \\
            \\function onSearchInput() {
            \\  currentQuery = document.getElementById('search-input').value.trim();
            \\  renderTree();
            \\}
            \\
            \\function expandAll() {
            \\  document.querySelectorAll('details.module-group').forEach(d => d.open = true);
            \\}
            \\
            \\function collapseAll() {
            \\  document.querySelectorAll('details.module-group').forEach(d => d.open = false);
            \\}
            \\
            \\function copyModelJson() {
            \\  const str = JSON.stringify(RAW_DATA, null, 2);
            \\  navigator.clipboard.writeText(str).then(() => {
            \\    alert('✅ Model Graph JSON copied to clipboard!');
            \\  }).catch(() => {
            \\    alert('Failed to copy to clipboard.');
            \\  });
            \\}
            \\
            \\function downloadModelJson() {
            \\  const blob = new Blob([JSON.stringify(RAW_DATA, null, 2)], { type: 'application/json' });
            \\  const url = URL.createObjectURL(blob);
            \\  const a = document.createElement('a');
            \\  a.href = url;
            \\  a.download = 'model_graph.json';
            \\  a.click();
            \\  URL.revokeObjectURL(url);
            \\}
            \\
            \\function openInspector(encodedJson) {
            \\  const node = JSON.parse(decodeURIComponent(encodedJson));
            \\  document.getElementById('modal-node-name').textContent = node.name;
            \\  const body = document.getElementById('modal-content');
            \\  body.innerHTML = `
            \\    <div class="detail-row"><span class="detail-key">Full Name</span><span class="detail-val">${node.name}</span></div>
            \\    <div class="detail-row"><span class="detail-key">Module Path</span><span class="detail-val">${node.path || '(Root)'}</span></div>
            \\    <div class="detail-row"><span class="detail-key">Kind</span><span class="detail-val">${node.kind}</span></div>
            \\    <div class="detail-row"><span class="detail-key">Shape</span><span class="detail-val">${node.shape}</span></div>
            \\    <div class="detail-row"><span class="detail-key">Elements</span><span class="detail-val">${formatNumber(node.elements)}</span></div>
            \\    <div class="detail-row"><span class="detail-key">Memory Bytes</span><span class="detail-val">${formatBytes(node.bytes)} (${node.bytes} B)</span></div>
            \\    <div class="detail-row"><span class="detail-key">Status</span><span class="detail-val">${node.status}</span></div>
            \\    <div class="detail-row"><span class="detail-key">Inferred Act / Op</span><span class="detail-val">${node.act || 'None'}</span></div>
            \\    <div class="detail-row"><span class="detail-key">Initialization Strategy</span><span class="detail-val">${node.strategy || 'N/A'}</span></div>
            \\    <div class="detail-row"><span class="detail-key">Upstream Producer Inputs</span><span class="detail-val">${(node.inputs || []).join(', ') || 'None'}</span></div>
            \\    <div class="detail-row"><span class="detail-key">Downstream Consumer Outputs</span><span class="detail-val">${(node.outputs || []).join(', ') || 'None'}</span></div>
            \\  `;
            \\  document.getElementById('node-modal').style.display = 'flex';
            \\}
            \\
            \\function closeModal() {
            \\  document.getElementById('node-modal').style.display = 'none';
            \\}
            \\
            \\// 初始化执行前端解析与界面渲染
            \\renderKPIs();
            \\renderTree();
            \\renderGraphFlow();
            \\</script>
            \\</body>
            \\</html>
        );

        return html_buf.toOwnedSlice(allocator);
    }

    /// 依据约定的递归 JSON 字符串导出 HTML 报告文件
    pub fn exportFromJson(json_str: []const u8, file_path: []const u8, allocator: std.mem.Allocator) !void {
        const html_content = try renderFromJson(json_str, allocator);
        defer allocator.free(html_content);

        const path_z = try allocator.dupeZ(u8, file_path);
        defer allocator.free(path_z);

        const file = std.c.fopen(path_z.ptr, "wb") orelse return error.CannotOpenFile;
        defer _ = std.c.fclose(file);

        const written = std.c.fwrite(html_content.ptr, 1, html_content.len, file);
        if (written < html_content.len) return error.WriteFailed;
    }

    /// 依据约定的 ModelHierarchyGraph 结构体渲染自包含 HTML 报告文档
    pub fn renderFromHierarchy(model_graph: *const ModelHierarchyGraph, allocator: std.mem.Allocator) ![]const u8 {
        const json_str = try graph_ir.serializeJson(model_graph, allocator);
        defer allocator.free(json_str);
        return renderFromJson(json_str, allocator);
    }

    /// 依据约定的 ModelHierarchyGraph 结构体导出 HTML 报告文件
    pub fn exportFromHierarchy(model_graph: *const ModelHierarchyGraph, file_path: []const u8, allocator: std.mem.Allocator) !void {
        const json_str = try graph_ir.serializeJson(model_graph, allocator);
        defer allocator.free(json_str);
        try exportFromJson(json_str, file_path, allocator);
    }

    /// 从 Graph 直接生成自包含 HTML 报告 (组合 generateJson + renderFromJson)
    pub fn generateHtmlReport(graph: *Graph, allocator: std.mem.Allocator) ![]const u8 {
        const json_str = try graph_ir.generateJson(graph, allocator);
        defer allocator.free(json_str);
        return renderFromJson(json_str, allocator);
    }

    /// 从 Graph 直接导出 HTML 报告文件 (组合 generateJson + exportFromJson)
    pub fn exportHtmlReport(graph: *Graph, file_path: []const u8, allocator: std.mem.Allocator) !void {
        const json_str = try graph_ir.generateJson(graph, allocator);
        defer allocator.free(json_str);
        try exportFromJson(json_str, file_path, allocator);
    }
};

// ============================================================================
// 4. 便捷顶层重导出 (Top-Level Re-Exports)
// ============================================================================
pub const buildModelHierarchy = graph_ir.build;
pub const generateJson = graph_ir.generateJson;
pub const exportJson = graph_ir.exportJson;

pub const renderHtmlReport = html_report.renderFromJson;
pub const exportHtmlReport = html_report.exportFromJson;
pub const generateHtmlReport = html_report.generateHtmlReport;
