const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const init_mod = @import("init.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Graph = autodiff.Graph;

/// 单个计算图节点的详细可视化元数据
pub const NodeData = struct {
    name: []const u8,
    kind: []const u8, // "Param", "Input", "Activation"
    shape_str: []const u8,
    elements: usize,
    bytes: usize,
    status: []const u8, // "AUTO_GRAPH", "CUSTOM_INIT", "INPUT", "OP_OUTPUT"
    inferred_act: []const u8,
    strategy: []const u8,
};

/// 从节点名称中提取父模块路径 (以 '.' 分隔)
fn extractModuleScope(name: []const u8) ?[]const u8 {
    if (std.mem.lastIndexOfScalar(u8, name, '.')) |dot_idx| {
        if (dot_idx > 0) return name[0..dot_idx];
    }
    return null;
}

/// 计算两个模块路径在点号边界处的最长公共前缀
fn getCommonModulePrefix(a: []const u8, b: []const u8) []const u8 {
    if (std.mem.eql(u8, a, b)) return a;
    const min_len = @min(a.len, b.len);
    var matched_len: usize = 0;
    while (matched_len < min_len and a[matched_len] == b[matched_len]) : (matched_len += 1) {}

    if (matched_len == min_len) {
        if (a.len > min_len and a[min_len] == '.') return b;
        if (b.len > min_len and b[min_len] == '.') return a;
    }

    var i = matched_len;
    while (i > 0) : (i -= 1) {
        if (a[i - 1] == '.') {
            return a[0 .. i - 1];
        }
    }
    return "";
}

/// 收集计算图中所有节点的详细元数据
pub fn collectGraphNodes(graph: *Graph, allocator: std.mem.Allocator) !std.ArrayList(NodeData) {
    var nodes: std.ArrayList(NodeData) = .empty;
    errdefer nodes.deinit(allocator);

    const arena_alloc = graph.arena.allocator();
    var visited = std.AutoHashMap(*Tensor, void).init(arena_alloc);
    defer visited.deinit();

    var scopes = std.AutoHashMap(*Tensor, []const u8).init(arena_alloc);
    defer scopes.deinit();

    // 1. 预扫描：注册所有显式命名且包含点号分层的张量作用域
    for (graph.tensors.items) |t| {
        if (t.name) |n| {
            if (extractModuleScope(n)) |s| {
                scopes.put(t, s) catch {};
            }
        }
    }
    for (graph.ops.items) |op| {
        for (op.inputs) |t| {
            if (t.name) |n| {
                if (extractModuleScope(n)) |s| {
                    scopes.put(t, s) catch {};
                }
            }
        }
        for (op.outputs) |t| {
            if (t.name) |n| {
                if (extractModuleScope(n)) |s| {
                    scopes.put(t, s) catch {};
                }
            }
        }
    }

    // 2. 拓扑扫描：自动将算子输入参数的模块前缀推导并级联传播至各中间激活节点
    for (graph.ops.items) |op| {
        var op_scope: ?[]const u8 = null;
        // 优先从输入中的底层参数（叶子节点，如 weight, bias）继承最具体的子模块层级所属
        for (op.inputs) |inp| {
            if (inp.creator == null and inp.name != null) {
                if (extractModuleScope(inp.name.?)) |p_scope| {
                    if (op_scope == null or p_scope.len > op_scope.?.len) {
                        op_scope = p_scope;
                    }
                }
            }
        }
        // 若无直接参数输入，则寻找所有已有作用域输入的公共最长模块前缀
        if (op_scope == null) {
            for (op.inputs) |inp| {
                if (scopes.get(inp)) |inp_scope| {
                    if (op_scope == null) {
                        op_scope = inp_scope;
                    } else {
                        const common = getCommonModulePrefix(op_scope.?, inp_scope);
                        if (common.len > 0) {
                            op_scope = common;
                        }
                    }
                }
            }
        }

        if (op_scope) |scope| {
            for (op.outputs) |out| {
                if (!scopes.contains(out)) {
                    scopes.put(out, scope) catch {};
                }
            }
        }
    }

    var param_idx: usize = 0;
    var input_idx: usize = 0;
    var op_idx: usize = 0;

    for (graph.ops.items) |op| {
        for (op.inputs) |t| {
            if (visited.contains(t)) continue;
            visited.put(t, {}) catch continue;
            try appendNodeData(graph, t, &param_idx, &input_idx, &op_idx, &scopes, &nodes, allocator);
        }
        for (op.outputs) |t| {
            if (visited.contains(t)) continue;
            visited.put(t, {}) catch continue;
            try appendNodeData(graph, t, &param_idx, &input_idx, &op_idx, &scopes, &nodes, allocator);
        }
    }

    for (graph.tensors.items) |t| {
        if (visited.contains(t)) continue;
        visited.put(t, {}) catch continue;
        try appendNodeData(graph, t, &param_idx, &input_idx, &op_idx, &scopes, &nodes, allocator);
    }

    return nodes;
}

fn appendNodeData(
    graph: *Graph,
    t: *Tensor,
    param_idx: *usize,
    input_idx: *usize,
    op_idx: *usize,
    scopes: *const std.AutoHashMap(*Tensor, []const u8),
    nodes: *std.ArrayList(NodeData),
    allocator: std.mem.Allocator,
) !void {
    // 1. 形状格式化
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
    const shape_str = try allocator.dupe(u8, shape_buf[0..shape_len]);

    const elements = t.data.len;
    const bytes = elements * @sizeOf(f32);

    // 2. 判断节点分类
    if (t.creator) |creator_op| {
        const op_name = @tagName(creator_op.op_type);
        const name = if (t.name) |n|
            try allocator.dupe(u8, n)
        else if (scopes.get(t)) |scope|
            try std.fmt.allocPrint(allocator, "{s}.act_{s}_{d}", .{ scope, op_name, op_idx.* })
        else
            try std.fmt.allocPrint(allocator, "computation_graph.{s}_{d}", .{ op_name, op_idx.* });
        op_idx.* += 1;

        var strat_buf: [64]u8 = undefined;
        const strat = try allocator.dupe(u8, std.fmt.bufPrint(&strat_buf, "produced by {s}", .{op_name}) catch "op output");

        try nodes.append(allocator, .{
            .name = name,
            .kind = "Activation",
            .shape_str = shape_str,
            .elements = elements,
            .bytes = bytes,
            .status = "OP_OUTPUT",
            .inferred_act = try allocator.dupe(u8, op_name),
            .strategy = strat,
        });
        return;
    }

    if (!t.requires_grad) {
        const name = if (t.name) |n|
            try allocator.dupe(u8, n)
        else if (scopes.get(t)) |scope|
            try std.fmt.allocPrint(allocator, "{s}.input_{d}", .{ scope, input_idx.* })
        else
            try std.fmt.allocPrint(allocator, "inputs.input_{d}", .{input_idx.*});
        input_idx.* += 1;

        try nodes.append(allocator, .{
            .name = name,
            .kind = "Input",
            .shape_str = shape_str,
            .elements = elements,
            .bytes = bytes,
            .status = "INPUT",
            .inferred_act = "N/A",
            .strategy = "user input / constant",
        });
        return;
    }

    // 模型可学习参数
    const name = if (t.name) |n|
        try allocator.dupe(u8, n)
    else if (scopes.get(t)) |scope|
        try std.fmt.allocPrint(allocator, "{s}.param_{d}", .{ scope, param_idx.* })
    else
        try std.fmt.allocPrint(allocator, "parameters.param_{d}", .{param_idx.*});
    param_idx.* += 1;

    if (t.is_custom_initialized) {
        try nodes.append(allocator, .{
            .name = name,
            .kind = "Param",
            .shape_str = shape_str,
            .elements = elements,
            .bytes = bytes,
            .status = "CUSTOM_INIT",
            .inferred_act = "N/A",
            .strategy = "user-defined customInit",
        });
        return;
    }

    if (t.shape.len == 1 or (t.shape.len == 2 and t.shape.dims[0] == 1)) {
        try nodes.append(allocator, .{
            .name = name,
            .kind = "Param",
            .shape_str = shape_str,
            .elements = elements,
            .bytes = bytes,
            .status = "AUTO_GRAPH",
            .inferred_act = "bias",
            .strategy = "zeros (0.0)",
        });
        return;
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
        .tanh, .sigmoid => try allocator.dupe(u8, std.fmt.bufPrint(&strat_buf, "Xavier Normal (gain={d:.3})", .{gain}) catch "Xavier Normal"),
        .selu => try allocator.dupe(u8, "LeCun Normal"),
        else => try allocator.dupe(u8, std.fmt.bufPrint(&strat_buf, "He Normal (gain={d:.3})", .{gain}) catch "He Normal"),
    };

    try nodes.append(allocator, .{
        .name = name,
        .kind = "Param",
        .shape_str = shape_str,
        .elements = elements,
        .bytes = bytes,
        .status = "AUTO_GRAPH",
        .inferred_act = try allocator.dupe(u8, act_name),
        .strategy = strat,
    });
}

/// 释放节点元数据内存
pub fn freeGraphNodes(nodes: *std.ArrayList(NodeData), allocator: std.mem.Allocator) void {
    for (nodes.items) |n| {
        allocator.free(n.name);
        allocator.free(n.shape_str);
        if (std.mem.startsWith(u8, n.strategy, "Xavier") or std.mem.startsWith(u8, n.strategy, "He") or std.mem.startsWith(u8, n.strategy, "produced by")) {
            allocator.free(n.strategy);
        }
        if (std.mem.eql(u8, n.kind, "Activation") or std.mem.eql(u8, n.kind, "Param")) {
            if (!std.mem.eql(u8, n.inferred_act, "N/A") and !std.mem.eql(u8, n.inferred_act, "bias")) {
                allocator.free(n.inferred_act);
            }
        }
    }
    nodes.deinit(allocator);
}

/// 计算图中逻辑模块之间的数据流转边
pub const EdgeData = struct {
    from: []const u8,
    to: []const u8,
    shape: []const u8,
    is_skip: bool = false,
};

fn getLogicalModule(name: []const u8) []const u8 {
    if (std.mem.endsWith(u8, name, ".residual_attn")) return name;
    if (std.mem.endsWith(u8, name, ".output")) return name;
    if (std.mem.endsWith(u8, name, ".embeddings_sum")) return name;
    if (std.mem.startsWith(u8, name, "inputs.")) return name;
    if (std.mem.startsWith(u8, name, "outputs.")) return name;

    if (std.mem.indexOf(u8, name, ".attn.")) |idx| {
        return name[0 .. idx + 5];
    }
    if (std.mem.indexOf(u8, name, ".mlp.")) |idx| {
        return name[0 .. idx + 4];
    }
    if (std.mem.indexOf(u8, name, ".ln_1.")) |idx| {
        return name[0 .. idx + 5];
    }
    if (std.mem.indexOf(u8, name, ".ln_2.")) |idx| {
        return name[0 .. idx + 5];
    }
    if (std.mem.indexOf(u8, name, ".ln_f.")) |idx| {
        return name[0 .. idx + 5];
    }
    if (std.mem.indexOf(u8, name, ".wte.")) |idx| {
        return name[0 .. idx + 4];
    }
    if (std.mem.indexOf(u8, name, ".wpe.")) |idx| {
        return name[0 .. idx + 4];
    }
    if (std.mem.indexOf(u8, name, ".lm_head.")) |idx| {
        return name[0 .. idx + 8];
    }

    if (extractModuleScope(name)) |p| return p;
    return name;
}

pub fn collectGraphEdges(graph: *Graph, allocator: std.mem.Allocator) !std.ArrayList(EdgeData) {
    var edges: std.ArrayList(EdgeData) = .empty;
    errdefer freeGraphEdges(&edges, allocator);

    const arena_alloc = graph.arena.allocator();
    var edge_set = std.StringHashMap(void).init(arena_alloc);
    defer edge_set.deinit();

    for (graph.ops.items) |op| {
        for (op.outputs) |out| {
            const out_name = out.name orelse continue;
            const to_mod = getLogicalModule(out_name);

            for (op.inputs) |inp| {
                const inp_name = inp.name orelse continue;
                if (inp.creator == null and inp.requires_grad) continue;
                if (std.mem.endsWith(u8, inp_name, ".causal_mask")) continue;
                if (std.mem.endsWith(u8, inp_name, ".pos_indices") and std.mem.eql(u8, to_mod, "gpt.wpe")) continue;

                const from_mod = getLogicalModule(inp_name);
                if (std.mem.eql(u8, from_mod, to_mod)) continue;

                var shape_buf: [64]u8 = undefined;
                var shape_len: usize = 0;
                shape_buf[0] = '[';
                shape_len += 1;
                for (0..inp.shape.len) |d| {
                    if (d > 0) {
                        shape_buf[shape_len] = ',';
                        shape_buf[shape_len + 1] = ' ';
                        shape_len += 2;
                    }
                    const part = std.fmt.bufPrint(shape_buf[shape_len..], "{d}", .{inp.shape.dims[d]}) catch "";
                    shape_len += part.len;
                }
                shape_buf[shape_len] = ']';
                shape_len += 1;
                const shape_str = try allocator.dupe(u8, shape_buf[0..shape_len]);

                const is_skip = (std.mem.endsWith(u8, to_mod, ".residual_attn") and !std.mem.endsWith(u8, from_mod, ".attn")) or
                                (std.mem.endsWith(u8, to_mod, ".output") and !std.mem.endsWith(u8, from_mod, ".mlp"));

                var key_buf: [256]u8 = undefined;
                const key = std.fmt.bufPrint(&key_buf, "{s}->{s}", .{ from_mod, to_mod }) catch continue;
                if (edge_set.contains(key)) {
                    allocator.free(shape_str);
                    continue;
                }
                try edge_set.put(try arena_alloc.dupe(u8, key), {});

                try edges.append(allocator, .{
                    .from = try allocator.dupe(u8, from_mod),
                    .to = try allocator.dupe(u8, to_mod),
                    .shape = shape_str,
                    .is_skip = is_skip,
                });
            }
        }
    }

    return edges;
}

pub fn freeGraphEdges(edges: *std.ArrayList(EdgeData), allocator: std.mem.Allocator) void {
    for (edges.items) |e| {
        allocator.free(e.from);
        allocator.free(e.to);
        allocator.free(e.shape);
    }
    edges.deinit(allocator);
}

fn formatShapeAlloc(allocator: std.mem.Allocator, s: Shape) ![]const u8 {
    var buf: [64]u8 = undefined;
    var len: usize = 0;
    buf[0] = '[';
    len += 1;
    for (0..s.len) |d| {
        if (d > 0) {
            buf[len] = ',';
            buf[len + 1] = ' ';
            len += 2;
        }
        const part = std.fmt.bufPrint(buf[len..], "{d}", .{s.dims[d]}) catch "";
        len += part.len;
    }
    buf[len] = ']';
    len += 1;
    return try allocator.dupe(u8, buf[0..len]);
}

/// 计算图中单个算子/节点的维度流转与参数信息
pub const OpData = struct {
    op_type: []const u8,
    name: []const u8,
    module: []const u8,
    input_shape: []const u8,
    param_shape: []const u8,
    output_shape: []const u8,
    elements: usize,
    bytes: usize,
};

pub fn collectGraphOps(graph: *Graph, allocator: std.mem.Allocator) !std.ArrayList(OpData) {
    var ops_list: std.ArrayList(OpData) = .empty;
    errdefer freeGraphOps(&ops_list, allocator);

    const arena_alloc = graph.arena.allocator();
    var scopes = std.AutoHashMap(*Tensor, []const u8).init(arena_alloc);
    defer scopes.deinit();

    for (graph.tensors.items) |t| {
        if (t.name) |n| {
            if (extractModuleScope(n)) |s| {
                scopes.put(t, s) catch {};
            }
        }
    }
    for (graph.ops.items) |op| {
        for (op.inputs) |t| {
            if (t.name) |n| {
                if (extractModuleScope(n)) |s| {
                    scopes.put(t, s) catch {};
                }
            }
        }
        for (op.outputs) |t| {
            if (t.name) |n| {
                if (extractModuleScope(n)) |s| {
                    scopes.put(t, s) catch {};
                }
            }
        }
    }

    for (graph.ops.items) |op| {
        var op_scope: ?[]const u8 = null;
        for (op.inputs) |inp| {
            if (inp.creator == null and inp.name != null) {
                if (extractModuleScope(inp.name.?)) |p_scope| {
                    if (op_scope == null or p_scope.len > op_scope.?.len) {
                        op_scope = p_scope;
                    }
                }
            }
        }
        if (op_scope == null) {
            for (op.inputs) |inp| {
                if (scopes.get(inp)) |inp_scope| {
                    if (op_scope == null) {
                        op_scope = inp_scope;
                    } else {
                        const common = getCommonModulePrefix(op_scope.?, inp_scope);
                        if (common.len > 0) op_scope = common;
                    }
                }
            }
        }
        if (op_scope) |scope| {
            for (op.outputs) |out| {
                if (!scopes.contains(out)) {
                    scopes.put(out, scope) catch {};
                }
            }
        }
    }

    var op_counter: usize = 0;
    for (graph.ops.items) |op| {
        if (op.outputs.len == 0) continue;
        const out = op.outputs[0];
        const op_tag = @tagName(op.op_type);

        const mod_scope = if (out.name) |n|
            (extractModuleScope(n) orelse (scopes.get(out) orelse "graph"))
        else
            (scopes.get(out) orelse "graph");

        const op_name = if (out.name) |n|
            try allocator.dupe(u8, n)
        else
            try std.fmt.allocPrint(allocator, "{s}.act_{s}_{d}", .{ mod_scope, op_tag, op_counter });
        op_counter += 1;

        const out_shape = try formatShapeAlloc(allocator, out.shape);
        const elements = out.data.len;
        const bytes = elements * @sizeOf(f32);

        var inp_shape_buf: [256]u8 = undefined;
        var inp_shape_len: usize = 0;

        var param_shape_buf: [256]u8 = undefined;
        var param_shape_len: usize = 0;

        for (op.inputs) |inp| {
            var s_buf: [64]u8 = undefined;
            var s_len: usize = 0;
            s_buf[0] = '[';
            s_len += 1;
            for (0..inp.shape.len) |d| {
                if (d > 0) {
                    s_buf[s_len] = ',';
                    s_buf[s_len + 1] = ' ';
                    s_len += 2;
                }
                const part = std.fmt.bufPrint(s_buf[s_len..], "{d}", .{inp.shape.dims[d]}) catch "";
                s_len += part.len;
            }
            s_buf[s_len] = ']';
            s_len += 1;
            const single_shape = s_buf[0..s_len];

            if (inp.creator == null and inp.requires_grad) {
                const p_name = if (inp.name) |pn| (if (std.mem.lastIndexOf(u8, pn, ".")) |dot| pn[dot + 1 ..] else pn) else "param";
                if (param_shape_len > 0) {
                    param_shape_buf[param_shape_len] = ',';
                    param_shape_buf[param_shape_len + 1] = ' ';
                    param_shape_len += 2;
                }
                const formatted = std.fmt.bufPrint(param_shape_buf[param_shape_len..], "{s}: {s}", .{ p_name, single_shape }) catch "";
                param_shape_len += formatted.len;
            } else {
                if (inp_shape_len > 0) {
                    inp_shape_buf[inp_shape_len] = ',';
                    inp_shape_buf[inp_shape_len + 1] = ' ';
                    inp_shape_len += 2;
                }
                const formatted = std.fmt.bufPrint(inp_shape_buf[inp_shape_len..], "{s}", .{single_shape}) catch "";
                inp_shape_len += formatted.len;
            }
        }

        const input_shape = if (inp_shape_len > 0)
            try allocator.dupe(u8, inp_shape_buf[0..inp_shape_len])
        else
            try allocator.dupe(u8, "None");

        const param_shape = if (param_shape_len > 0)
            try allocator.dupe(u8, param_shape_buf[0..param_shape_len])
        else
            try allocator.dupe(u8, "None (Stateless)");

        try ops_list.append(allocator, .{
            .op_type = try allocator.dupe(u8, op_tag),
            .name = op_name,
            .module = try allocator.dupe(u8, mod_scope),
            .input_shape = input_shape,
            .param_shape = param_shape,
            .output_shape = out_shape,
            .elements = elements,
            .bytes = bytes,
        });
    }

    return ops_list;
}

pub fn freeGraphOps(ops_list: *std.ArrayList(OpData), allocator: std.mem.Allocator) void {
    for (ops_list.items) |o| {
        allocator.free(o.op_type);
        allocator.free(o.name);
        allocator.free(o.module);
        allocator.free(o.input_shape);
        allocator.free(o.param_shape);
        allocator.free(o.output_shape);
    }
    ops_list.deinit(allocator);
}

fn escapeJsonString(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (input) |c| {
        switch (c) {
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '"' => try out.appendSlice(allocator, "\\\""),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => try out.append(allocator, c),
        }
    }
    return out.toOwnedSlice(allocator);
}

/// 将计算图结构与各层初始化详情格式化为可交互、层级展开的 HTML 网页文档
pub fn generateHtmlReport(graph: *Graph, allocator: std.mem.Allocator) ![]const u8 {
    var nodes = try collectGraphNodes(graph, allocator);
    defer freeGraphNodes(&nodes, allocator);

    var edges = try collectGraphEdges(graph, allocator);
    defer freeGraphEdges(&edges, allocator);

    var ops = try collectGraphOps(graph, allocator);
    defer freeGraphOps(&ops, allocator);

    var total_params: usize = 0;
    var total_bytes: usize = 0;
    var param_nodes: usize = 0;
    var input_nodes: usize = 0;
    var act_nodes: usize = 0;
    var custom_init_count: usize = 0;
    var auto_graph_count: usize = 0;

    for (nodes.items) |n| {
        total_bytes += n.bytes;
        if (std.mem.eql(u8, n.kind, "Param")) {
            total_params += n.elements;
            param_nodes += 1;
            if (std.mem.eql(u8, n.status, "CUSTOM_INIT")) {
                custom_init_count += 1;
            } else if (std.mem.eql(u8, n.status, "AUTO_GRAPH")) {
                auto_graph_count += 1;
            }
        } else if (std.mem.eql(u8, n.kind, "Input")) {
            input_nodes += 1;
        } else {
            act_nodes += 1;
        }
    }

    var html_buf: std.ArrayList(u8) = .empty;
    errdefer html_buf.deinit(allocator);

    // 1. HTML Header & Embedded Styling
    try html_buf.appendSlice(allocator,
        \\<!DOCTYPE html>
        \\<html lang="zh-CN">
        \\<head>
        \\  <meta charset="UTF-8">
        \\  <meta name="viewport" content="width=device-width, initial-scale=1.0">
        \\  <title>ZNN Computation Graph & Model Hierarchy Report</title>
        \\  <!-- KaTeX for high-performance LaTeX math formula rendering -->
        \\  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
        \\  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
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
        \\    .container { max-width: 1380px; margin: 0 auto; }
        \\    header { margin-bottom: 28px; }
        \\    .title-row { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px; margin-bottom: 8px; }
        \\    h1 { font-size: 26px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 10px; }
        \\    .subtitle { color: var(--text-sub); font-size: 14px; }
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
        \\    }
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
        \\    .search-box {
        \\      position: relative;
        \\      flex: 1;
        \\      min-width: 260px;
        \\    }
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
        \\    }
        \\    .btn:hover { background: var(--bg-card-hover); border-color: #475569; }
        \\    .btn.active { background: #0284c7; border-color: #38bdf8; color: #fff; }
        \\
        \\    /* Tree & Group Styling */
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
        \\    .module-meta { font-size: 12px; color: var(--text-sub); display: flex; gap: 12px; font-family: var(--font-mono); }
        \\    .module-content { padding: 10px 16px; display: flex; flex-direction: column; gap: 8px; overflow-x: auto; }
        \\
        \\    /* Table */
        \\    table.node-table {
        \\      width: 100%;
        \\      border-collapse: collapse;
        \\      font-size: 13px;
        \\      margin-top: 4px;
        \\    }
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
        \\    td {
        \\      padding: 9px 12px;
        \\      border-bottom: 1px solid rgba(51, 65, 85, 0.4);
        \\      vertical-align: middle;
        \\    }
        \\    tr:last-child td { border-bottom: none; }
        \\    tr:hover td { background: rgba(56, 189, 248, 0.04); }
        \\    .node-name { font-family: var(--font-mono); font-weight: 600; color: #f1f5f9; }
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
        \\
        \\    /* Sequential Flow Connectors in Module Tree */
        \\    .tree-flow-connector {
        \\      display: flex;
        \\      align-items: center;
        \\      gap: 8px;
        \\      padding: 6px 12px;
        \\      margin: 2px 0;
        \\      color: #64748b;
        \\      font-size: 11px;
        \\      font-family: var(--font-mono);
        \\    }
        \\    .tree-flow-connector .flow-line {
        \\      width: 2px;
        \\      height: 14px;
        \\      background: #334155;
        \\      margin-left: 12px;
        \\    }
        \\    .tree-flow-connector .flow-arrow-text {
        \\      display: flex;
        \\      align-items: center;
        \\      gap: 6px;
        \\      background: rgba(15, 23, 42, 0.7);
        \\      padding: 2px 8px;
        \\      border-radius: 4px;
        \\      border: 1px solid rgba(51, 65, 85, 0.4);
        \\      color: #94a3b8;
        \\    }
        \\    .tree-flow-connector .flow-arrow-text span.arrow-symbol {
        \\      color: #38bdf8;
        \\      font-weight: 700;
        \\    }
        \\
        \\    /* In-Tree Residual Parallel Branch Styling */
        \\    details.tree-sublayer-block {
        \\      background: rgba(15, 23, 42, 0.6);
        \\      border: 1px dashed #334155;
        \\      border-radius: 8px;
        \\      margin-bottom: 8px;
        \\      padding: 0;
        \\      overflow: hidden;
        \\      transition: border-color 0.2s;
        \\    }
        \\    details.tree-sublayer-block[open] {
        \\      border-color: #64748b;
        \\    }
        \\    details.tree-sublayer-block > summary.tree-sublayer-header {
        \\      background: rgba(30, 41, 59, 0.7);
        \\      padding: 10px 14px;
        \\      cursor: pointer;
        \\      display: flex;
        \\      align-items: center;
        \\      justify-content: space-between;
        \\      user-select: none;
        \\      list-style: none;
        \\      font-size: 12px;
        \\      font-weight: 700;
        \\      letter-spacing: 0.5px;
        \\      color: #e2e8f0;
        \\      border-bottom: 1px dashed transparent;
        \\      transition: background 0.2s;
        \\    }
        \\    details.tree-sublayer-block > summary.tree-sublayer-header:hover {
        \\      background: rgba(51, 65, 85, 0.8);
        \\    }
        \\    details.tree-sublayer-block[open] > summary.tree-sublayer-header {
        \\      border-bottom-color: rgba(51, 65, 85, 0.6);
        \\    }
        \\    details.tree-sublayer-block > summary.tree-sublayer-header::-webkit-details-marker {
        \\      display: none;
        \\    }
        \\    .tree-sublayer-body {
        \\      padding: 12px;
        \\      display: flex;
        \\      flex-direction: column;
        \\      gap: 10px;
        \\    }
        \\    .tree-branches-row {
        \\      display: flex;
        \\      gap: 12px;
        \\      align-items: stretch;
        \\    }
        \\    .tree-branch-col {
        \\      flex: 1;
        \\      min-width: 0;
        \\      background: var(--bg-card);
        \\      border: 1px solid var(--border-color);
        \\      border-radius: 8px;
        \\      padding: 10px 12px;
        \\      position: relative;
        \\      display: flex;
        \\      flex-direction: column;
        \\    }
        \\    .tree-branch-skip {
        \\      border-style: dashed;
        \\      border-color: #38bdf8;
        \\      background: rgba(2, 132, 199, 0.05);
        \\      flex: 0 0 280px;
        \\    }
        \\    .tree-branch-transform {
        \\      border-color: #818cf8;
        \\      background: rgba(99, 102, 241, 0.05);
        \\    }
        \\    .tree-branch-badge {
        \\      position: absolute;
        \\      top: -9px;
        \\      left: 12px;
        \\      font-size: 9px;
        \\      font-weight: 700;
        \\      font-family: var(--font-mono);
        \\      padding: 1px 6px;
        \\      border-radius: 3px;
        \\      text-transform: uppercase;
        \\    }
        \\    .tree-branch-badge.skip {
        \\      background: #0284c7;
        \\      color: #fff;
        \\      border: 1px solid #38bdf8;
        \\    }
        \\    .tree-branch-badge.transform {
        \\      background: #4f46e5;
        \\      color: #fff;
        \\      border: 1px solid #818cf8;
        \\    }
        \\    .tree-divider {
        \\      display: flex;
        \\      align-items: center;
        \\      justify-content: center;
        \\      color: #475569;
        \\      font-weight: 700;
        \\      font-size: 16px;
        \\    }
        \\    .tree-converge-box {
        \\      display: flex;
        \\      flex-direction: column;
        \\      align-items: center;
        \\      gap: 4px;
        \\      padding-top: 6px;
        \\      border-top: 1px dashed rgba(51, 65, 85, 0.5);
        \\    }
        \\    .tree-converge-card {
        \\      background: #1e1e38;
        \\      border: 1px solid #818cf8;
        \\      border-radius: 6px;
        \\      padding: 6px 14px;
        \\      font-family: var(--font-mono);
        \\      font-size: 12px;
        \\      font-weight: 600;
        \\      color: #a5b4fc;
        \\      display: flex;
        \\      align-items: center;
        \\      gap: 8px;
        \\    }
        \\
        \\    /* Tab Navigation */
        \\    .nav-tabs {
        \\      display: flex;
        \\      gap: 10px;
        \\      margin-bottom: 20px;
        \\      border-bottom: 1px solid var(--border-color);
        \\      padding-bottom: 8px;
        \\    }
        \\    .tab-btn {
        \\      background: var(--bg-card);
        \\      border: 1px solid var(--border-color);
        \\      color: var(--text-sub);
        \\      padding: 10px 20px;
        \\      border-radius: 8px;
        \\      font-size: 14px;
        \\      font-weight: 600;
        \\      cursor: pointer;
        \\      display: flex;
        \\      align-items: center;
        \\      gap: 8px;
        \\      transition: all 0.2s;
        \\    }
        \\    .tab-btn:hover { background: var(--bg-card-hover); color: var(--text-main); }
        \\    .tab-btn.active {
        \\      background: #0284c7;
        \\      border-color: #38bdf8;
        \\      color: #fff;
        \\      box-shadow: 0 0 12px rgba(56, 189, 248, 0.25);
        \\    }
        \\    .tab-view { display: none; }
        \\    .tab-view.active { display: block; }
        \\
        \\    /* DAG Architecture Flow & Skip Connection Styling */
        \\    .dag-flow-container {
        \\      display: flex;
        \\      flex-direction: column;
        \\      align-items: center;
        \\      gap: 0px;
        \\      max-width: 900px;
        \\      margin: 0 auto;
        \\      padding: 12px 0 32px 0;
        \\    }
        \\    .flow-card {
        \\      background: var(--bg-secondary);
        \\      border: 1px solid var(--border-color);
        \\      border-radius: 10px;
        \\      padding: 14px 20px;
        \\      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        \\      transition: all 0.2s;
        \\    }
        \\    .flow-card:hover { border-color: var(--border-focus); transform: translateY(-1px); }
        \\    .flow-card-linear { width: 100%; max-width: 440px; }
        \\    .flow-card-header {
        \\      display: flex;
        \\      align-items: center;
        \\      justify-content: space-between;
        \\      margin-bottom: 6px;
        \\    }
        \\    .flow-card-title {
        \\      font-family: var(--font-mono);
        \\      font-weight: 700;
        \\      font-size: 14px;
        \\      color: #38bdf8;
        \\    }
        \\    .flow-card-body {
        \\      font-size: 12px;
        \\      color: var(--text-sub);
        \\      display: flex;
        \\      justify-content: space-between;
        \\      align-items: center;
        \\    }
        \\    .flow-card-shape {
        \\      font-family: var(--font-mono);
        \\      background: rgba(15, 23, 42, 0.8);
        \\      padding: 2px 8px;
        \\      border-radius: 4px;
        \\      border: 1px solid rgba(51, 65, 85, 0.6);
        \\      color: #cbd5e1;
        \\    }
        \\
        \\    /* Downward Flow Arrow */
        \\    .flow-arrow-down {
        \\      display: flex;
        \\      flex-direction: column;
        \\      align-items: center;
        \\      padding: 6px 0;
        \\      color: #64748b;
        \\      font-size: 14px;
        \\    }
        \\    .flow-arrow-line {
        \\      width: 2px;
        \\      height: 18px;
        \\      background: #334155;
        \\    }
        \\    .flow-arrow-head {
        \\      color: #64748b;
        \\      line-height: 1;
        \\      font-size: 12px;
        \\    }
        \\    .flow-arrow-label {
        \\      font-family: var(--font-mono);
        \\      font-size: 10px;
        \\      color: #94a3b8;
        \\      background: #0f172a;
        \\      padding: 1px 6px;
        \\      border-radius: 4px;
        \\      border: 1px solid #334155;
        \\      margin: 2px 0;
        \\    }
        \\
        \\    /* Parallel Branch Stage: [Module A] | [Module B] */
        \\    .flow-stage-parallel {
        \\      width: 100%;
        \\      background: rgba(15, 23, 42, 0.45);
        \\      border: 1px dashed #334155;
        \\      border-radius: 12px;
        \\      padding: 16px;
        \\      box-sizing: border-box;
        \\    }
        \\    .parallel-header {
        \\      font-size: 12px;
        \\      font-weight: 600;
        \\      text-transform: uppercase;
        \\      letter-spacing: 0.5px;
        \\      color: var(--text-sub);
        \\      margin-bottom: 12px;
        \\      display: flex;
        \\      align-items: center;
        \\      justify-content: space-between;
        \\    }
        \\    .parallel-branches-row {
        \\      display: flex;
        \\      align-items: stretch;
        \\      gap: 16px;
        \\    }
        \\    .parallel-branch-col {
        \\      flex: 1;
        \\      display: flex;
        \\      flex-direction: column;
        \\      background: var(--bg-card);
        \\      border: 1px solid var(--border-color);
        \\      border-radius: 10px;
        \\      padding: 14px;
        \\      position: relative;
        \\    }
        \\    .parallel-branch-col.skip-branch {
        \\      border-style: dashed;
        \\      border-color: #38bdf8;
        \\      background: rgba(2, 132, 199, 0.04);
        \\    }
        \\    .parallel-branch-col.transform-branch {
        \\      border-color: #818cf8;
        \\      background: rgba(99, 102, 241, 0.04);
        \\    }
        \\    .branch-badge {
        \\      position: absolute;
        \\      top: -10px;
        \\      left: 14px;
        \\      font-size: 10px;
        \\      font-weight: 700;
        \\      font-family: var(--font-mono);
        \\      padding: 2px 8px;
        \\      border-radius: 4px;
        \\      text-transform: uppercase;
        \\    }
        \\    .branch-badge.skip {
        \\      background: #0284c7;
        \\      color: #fff;
        \\      border: 1px solid #38bdf8;
        \\    }
        \\    .branch-badge.transform {
        \\      background: #4f46e5;
        \\      color: #fff;
        \\      border: 1px solid #818cf8;
        \\    }
        \\    .branch-divider {
        \\      display: flex;
        \\      align-items: center;
        \\      justify-content: center;
        \\      font-weight: 700;
        \\      color: #64748b;
        \\      font-size: 18px;
        \\      padding: 0 4px;
        \\    }
        \\    .converge-arrow-box {
        \\      display: flex;
        \\      flex-direction: column;
        \\      align-items: center;
        \\      margin-top: 14px;
        \\      padding-top: 10px;
        \\      border-top: 1px dashed #334155;
        \\      width: 100%;
        \\    }
        \\    .converge-card {
        \\      width: 100%;
        \\      max-width: 480px;
        \\      background: #1e1e38;
        \\      border: 1px solid #818cf8;
        \\      border-radius: 10px;
        \\      padding: 12px 18px;
        \\      text-align: center;
        \\    }
        \\    .converge-title {
        \\      font-family: var(--font-mono);
        \\      font-weight: 700;
        \\      font-size: 14px;
        \\      color: #a5b4fc;
        \\      display: flex;
        \\      align-items: center;
        \\      justify-content: center;
        \\      gap: 8px;
        \\    }
        \\
        \\    /* Mermaid container */
        \\    .mermaid-box {
        \\      background: var(--bg-secondary);
        \\      border: 1px solid var(--border-color);
        \\      border-radius: 10px;
        \\      padding: 24px;
        \\      font-family: var(--font-mono);
        \\      font-size: 13px;
        \\      color: #f8fafc;
        \\      white-space: pre-wrap;
        \\      line-height: 1.6;
        \\      overflow-x: auto;
        \\    }
        \\
        \\    /* TensorBoard-like Node Styling */
        \\    .tb-node-card {
        \\      background: #1e293b;
        \\      border: 1px solid #334155;
        \\      border-radius: 8px;
        \\      padding: 10px 14px;
        \\      transition: all 0.2s ease;
        \\      cursor: pointer;
        \\      position: relative;
        \\    }
        \\    .tb-node-card:hover {
        \\      border-color: #38bdf8;
        \\      box-shadow: 0 4px 16px rgba(56, 189, 248, 0.15);
        \\      transform: translateY(-1px);
        \\    }
        \\    .tb-node-header {
        \\      display: flex;
        \\      align-items: center;
        \\      justify-content: space-between;
        \\      gap: 10px;
        \\    }
        \\    .tb-node-title {
        \\      display: flex;
        \\      align-items: center;
        \\      gap: 8px;
        \\      font-family: var(--font-mono);
        \\      font-weight: 700;
        \\      font-size: 13px;
        \\      color: #f1f5f9;
        \\    }
        \\    .tb-stage-card {
        \\      background: rgba(15, 23, 42, 0.55);
        \\      border: 1px dashed #334155;
        \\      border-radius: 10px;
        \\      padding: 12px;
        \\      display: flex;
        \\      flex-direction: column;
        \\      gap: 10px;
        \\      margin-bottom: 8px;
        \\    }
        \\    .tb-stage-header {
        \\      display: flex;
        \\      align-items: center;
        \\      justify-content: space-between;
        \\      padding: 2px 4px 8px 4px;
        \\      border-bottom: 1px solid rgba(51, 65, 85, 0.5);
        \\    }
        \\    .tb-stage-badge {
        \\      font-size: 10px;
        \\      font-weight: 800;
        \\      font-family: var(--font-mono);
        \\      background: #4f46e5;
        \\      color: #fff;
        \\      padding: 2px 7px;
        \\      border-radius: 4px;
        \\      text-transform: uppercase;
        \\      letter-spacing: 0.5px;
        \\    }
        \\    .tb-stage-name {
        \\      font-size: 12px;
        \\      font-weight: 700;
        \\      color: #cbd5e1;
        \\      margin-left: 8px;
        \\      flex: 1;
        \\    }
        \\    .tb-op-icon {
        \\      display: inline-flex;
        \\      align-items: center;
        \\      justify-content: center;
        \\      width: 22px;
        \\      height: 22px;
        \\      border-radius: 4px;
        \\      font-size: 12px;
        \\      font-weight: 700;
        \\    }
        \\    .tb-op-icon.norm { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
        \\    .tb-op-icon.linear { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
        \\    .tb-op-icon.attn { background: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); }
        \\    .tb-op-icon.mlp { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
        \\    .tb-op-icon.add { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
        \\    .tb-op-icon.emb { background: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.4); }
        \\    .tb-type-pill {
        \\      font-size: 10px;
        \\      font-weight: 700;
        \\      font-family: var(--font-mono);
        \\      text-transform: uppercase;
        \\      padding: 1px 6px;
        \\      border-radius: 3px;
        \\      background: rgba(148, 163, 184, 0.15);
        \\      color: #cbd5e1;
        \\      border: 1px solid rgba(148, 163, 184, 0.3);
        \\    }
        \\    .tb-param-chip {
        \\      font-size: 11px;
        \\      font-family: var(--font-mono);
        \\      color: #94a3b8;
        \\      background: rgba(15, 23, 42, 0.6);
        \\      padding: 2px 7px;
        \\      border-radius: 4px;
        \\      border: 1px solid rgba(51, 65, 85, 0.5);
        \\    }
        \\    .tb-shape-chip {
        \\      font-size: 11px;
        \\      font-family: var(--font-mono);
        \\      color: #38bdf8;
        \\      background: rgba(2, 132, 199, 0.12);
        \\      padding: 2px 7px;
        \\      border-radius: 4px;
        \\      border: 1px solid rgba(56, 189, 248, 0.3);
        \\    }
        \\    .tb-inspect-btn {
        \\      background: rgba(56, 189, 248, 0.1);
        \\      border: 1px solid #38bdf8;
        \\      color: #38bdf8;
        \\      font-size: 11px;
        \\      font-weight: 600;
        \\      padding: 3px 8px;
        \\      border-radius: 4px;
        \\      cursor: pointer;
        \\      display: inline-flex;
        \\      align-items: center;
        \\      gap: 4px;
        \\      transition: all 0.2s;
        \\    }
        \\    .tb-inspect-btn:hover {
        \\      background: #0284c7;
        \\      color: #fff;
        \\    }
        \\
        \\    /* Standalone Inspector Modal Window */
        \\    .inspector-overlay {
        \\      position: fixed;
        \\      top: 0;
        \\      left: 0;
        \\      width: 100vw;
        \\      height: 100vh;
        \\      background: rgba(4, 7, 15, 0.75);
        \\      backdrop-filter: blur(8px);
        \\      z-index: 9999;
        \\      display: none;
        \\      align-items: center;
        \\      justify-content: center;
        \\      padding: 24px;
        \\      box-sizing: border-box;
        \\      opacity: 0;
        \\      transition: opacity 0.2s ease;
        \\    }
        \\    .inspector-overlay.open {
        \\      display: flex;
        \\      opacity: 1;
        \\    }
        \\    .inspector-modal {
        \\      background: #0f172a;
        \\      border: 1px solid #38bdf8;
        \\      border-radius: 12px;
        \\      width: 100%;
        \\      max-width: 960px;
        \\      max-height: 88vh;
        \\      display: flex;
        \\      flex-direction: column;
        \\      box-shadow: 0 16px 48px rgba(0, 0, 0, 0.7), 0 0 24px rgba(56, 189, 248, 0.25);
        \\      overflow: hidden;
        \\      transform: translateY(12px);
        \\      transition: transform 0.2s ease;
        \\    }
        \\    .inspector-overlay.open .inspector-modal {
        \\      transform: translateY(0);
        \\    }
        \\    .inspector-header {
        \\      background: #1e293b;
        \\      padding: 16px 20px;
        \\      border-bottom: 1px solid #334155;
        \\      display: flex;
        \\      align-items: center;
        \\      justify-content: space-between;
        \\    }
        \\    .inspector-title-box {
        \\      display: flex;
        \\      align-items: center;
        \\      gap: 12px;
        \\    }
        \\    .inspector-icon {
        \\      font-size: 22px;
        \\      line-height: 1;
        \\    }
        \\    .inspector-title {
        \\      font-family: var(--font-mono);
        \\      font-size: 16px;
        \\      font-weight: 700;
        \\      color: #38bdf8;
        \\    }
        \\    .inspector-subtitle {
        \\      font-size: 12px;
        \\      color: #94a3b8;
        \\      margin-top: 3px;
        \\    }
        \\    .inspector-close-btn {
        \\      background: transparent;
        \\      border: 1px solid #475569;
        \\      color: #94a3b8;
        \\      font-size: 20px;
        \\      line-height: 1;
        \\      width: 32px;
        \\      height: 32px;
        \\      border-radius: 6px;
        \\      cursor: pointer;
        \\      display: flex;
        \\      align-items: center;
        \\      justify-content: center;
        \\      transition: all 0.2s;
        \\    }
        \\    .inspector-close-btn:hover {
        \\      background: #334155;
        \\      color: #fff;
        \\      border-color: #38bdf8;
        \\    }
        \\    .inspector-body {
        \\      padding: 20px;
        \\      overflow-y: auto;
        \\      flex: 1;
        \\      display: flex;
        \\      flex-direction: column;
        \\      gap: 16px;
        \\    }
        \\    .insp-section {
        \\      background: #131d2e;
        \\      border: 1px solid var(--border-color);
        \\      border-radius: 8px;
        \\      padding: 14px 16px;
        \\    }
        \\    .insp-section-title {
        \\      font-size: 13px;
        \\      font-weight: 700;
        \\      color: #f1f5f9;
        \\      margin-bottom: 10px;
        \\      display: flex;
        \\      align-items: center;
        \\      justify-content: space-between;
        \\    }
        \\    .insp-io-banner {
        \\      display: flex;
        \\      align-items: center;
        \\      gap: 16px;
        \\      background: #0b1120;
        \\      border: 1px solid #1e293b;
        \\      border-radius: 10px;
        \\      padding: 14px 18px;
        \\      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        \\    }
        \\    .insp-io-card {
        \\      flex: 1;
        \\      display: flex;
        \\      flex-direction: column;
        \\      gap: 4px;
        \\    }
        \\    .insp-io-label {
        \\      font-size: 11px;
        \\      font-weight: 700;
        \\      text-transform: uppercase;
        \\      letter-spacing: 0.5px;
        \\      color: #94a3b8;
        \\    }
        \\    .insp-io-shape {
        \\      font-family: var(--font-mono);
        \\      font-size: 18px;
        \\      font-weight: 800;
        \\    }
        \\    .insp-io-card.input .insp-io-shape { color: #38bdf8; }
        \\    .insp-io-card.output .insp-io-shape { color: #34d399; }
        \\    .insp-io-sub {
        \\      font-size: 11px;
        \\      color: #64748b;
        \\    }
        \\    .insp-io-arrow {
        \\      font-size: 20px;
        \\      color: #64748b;
        \\      font-weight: 700;
        \\      padding: 0 4px;
        \\    }
        \\    .insp-formula-box {
        \\      background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.7));
        \\      border: 1px solid rgba(56, 189, 248, 0.25);
        \\      border-left: 4px solid #38bdf8;
        \\      border-radius: 8px;
        \\      padding: 12px 18px;
        \\      display: flex;
        \\      flex-direction: column;
        \\      gap: 6px;
        \\      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        \\    }
        \\    .insp-formula-header {
        \\      display: flex;
        \\      align-items: center;
        \\      justify-content: space-between;
        \\      font-size: 11px;
        \\      font-weight: 700;
        \\      text-transform: uppercase;
        \\      letter-spacing: 0.5px;
        \\      color: #94a3b8;
        \\    }
        \\    .insp-formula-badge {
        \\      font-size: 10px;
        \\      font-weight: 600;
        \\      padding: 2px 8px;
        \\      border-radius: 4px;
        \\      font-family: var(--font-mono);
        \\    }
        \\    .insp-formula-badge.specified {
        \\      background: rgba(16, 185, 129, 0.18);
        \\      color: #34d399;
        \\      border: 1px solid rgba(52, 211, 153, 0.4);
        \\    }
        \\    .insp-formula-badge.inferred {
        \\      background: rgba(56, 189, 248, 0.15);
        \\      color: #38bdf8;
        \\      border: 1px solid rgba(56, 189, 248, 0.3);
        \\    }
        \\    .insp-formula-display {
        \\      background: #090d16;
        \\      border: 1px solid #1e293b;
        \\      border-radius: 6px;
        \\      padding: 12px 16px;
        \\      font-size: 15px;
        \\      color: #f8fafc;
        \\      display: flex;
        \\      align-items: center;
        \\      gap: 12px;
        \\      overflow-x: auto;
        \\    }
        \\    .insp-formula-display .katex {
        \\      font-size: 1.15em;
        \\    }
        \\    .insp-formula-desc {
        \\      font-size: 11.5px;
        \\      color: #94a3b8;
        \\      line-height: 1.4;
        \\    }
        \\    tr.node-row { cursor: pointer; }
        \\
        \\    footer { margin-top: 36px; text-align: center; font-size: 12px; color: #64748b; }
        \\  </style>
        \\</head>
        \\<body>
        \\<div class="container">
        \\  <header>
        \\    <div class="title-row">
        \\      <h1>⚡ ZNN Model Architecture & Computation Graph</h1>
        \\      <span class="subtitle">Interactive Hierarchical Inspection & Weight Initialization Report</span>
        \\    </div>
        \\  </header>
        \\
        \\  <!-- KPI Stats Grid -->
        \\  <div class="stats-grid">
    );

    // KPI Cards
    const mb_est = @as(f64, @floatFromInt(total_bytes)) / (1024.0 * 1024.0);
    try html_buf.print(allocator,
        \\    <div class="stat-card">
        \\      <div class="stat-label">Total Parameters</div>
        \\      <div class="stat-value">{d}</div>
        \\      <div class="stat-sub">~{d:.2} MB Memory Footprint</div>
        \\    </div>
        \\    <div class="stat-card">
        \\      <div class="stat-label">Param Tensors</div>
        \\      <div class="stat-value">{d}</div>
        \\      <div class="stat-sub">{d} Auto-Graph / {d} Custom</div>
        \\    </div>
        \\    <div class="stat-card">
        \\      <div class="stat-label">Graph Nodes</div>
        \\      <div class="stat-value">{d}</div>
        \\      <div class="stat-sub">{d} Inputs · {d} Activations</div>
        \\    </div>
        \\    <div class="stat-card">
        \\      <div class="stat-label">Memory Buffers</div>
        \\      <div class="stat-value">{d}</div>
        \\      <div class="stat-sub">Tracked by Graph Arena</div>
        \\    </div>
        \\  </div>
        \\
        \\  <!-- Navigation Tabs -->
        \\  <div class="nav-tabs">
        \\    <button class="tab-btn active" id="btn-tab-dag" onclick="switchMainTab('dag')">🔀 Architecture & Skip Connections</button>
        \\    <button class="tab-btn" id="btn-tab-tree" onclick="switchMainTab('tree')">📋 Hierarchical Module Tree</button>
        \\    <button class="tab-btn" id="btn-tab-mermaid" onclick="switchMainTab('mermaid')">📊 Mermaid Flowchart</button>
        \\  </div>
        \\
        \\  <!-- View 1: Architecture DAG with Skip Connections -->
        \\  <div class="tab-view active" id="view-dag">
        \\    <div class="dag-flow-container" id="dag-container">
        \\    </div>
        \\  </div>
        \\
        \\  <!-- View 2: Hierarchical Tree & Node Inspection -->
        \\  <div class="tab-view" id="view-tree">
        \\    <!-- Controls -->
        \\    <div class="controls-bar">
        \\      <div class="search-box">
        \\        <span class="search-icon">🔍</span>
        \\        <input type="text" id="search-input" class="search-input" placeholder="Search by module path, node name, op, or shape..." oninput="filterNodes()">
        \\      </div>
        \\      <div class="btn-group">
        \\        <button class="btn active" onclick="setKindFilter('all', this)">All Nodes ({d})</button>
        \\        <button class="btn" onclick="setKindFilter('Param', this)">Params ({d})</button>
        \\        <button class="btn" onclick="setKindFilter('Input', this)">Inputs ({d})</button>
        \\        <button class="btn" onclick="setKindFilter('Activation', this)">Activations ({d})</button>
        \\        <button class="btn" onclick="expandAll()">Expand All</button>
        \\        <button class="btn" onclick="collapseAll()">Collapse All</button>
        \\      </div>
        \\    </div>
        \\
        \\    <!-- Hierarchical Container -->
        \\    <div class="hierarchy-container" id="tree-container">
        \\    </div>
        \\  </div>
        \\
        \\  <!-- View 3: Mermaid Diagram -->
        \\  <div class="tab-view" id="view-mermaid">
        \\    <div class="mermaid-box" id="mermaid-code"></div>
        \\  </div>
        \\
        \\  <footer>Generated automatically by ZNN Autodiff Engine</footer>
        \\</div>
        \\
        \\<!-- Standalone Node Inspector Modal Window -->
        \\<div id="inspector-overlay" class="inspector-overlay" onclick="closeInspector(event)">
        \\  <div class="inspector-modal" onclick="event.stopPropagation()">
        \\    <div class="inspector-header">
        \\      <div class="inspector-title-box">
        \\        <span class="inspector-icon" id="insp-icon">🔍</span>
        \\        <div>
        \\          <div class="inspector-title" id="insp-title">Node Inspector</div>
        \\          <div class="inspector-subtitle" id="insp-subtitle">Detailed Inputs, Parameters & Activations</div>
        \\        </div>
        \\      </div>
        \\      <button class="inspector-close-btn" onclick="closeInspector()">&times;</button>
        \\    </div>
        \\    <div class="inspector-body" id="insp-body"></div>
        \\  </div>
        \\</div>
        \\
    , .{
        total_params,
        mb_est,
        param_nodes,
        auto_graph_count,
        custom_init_count,
        nodes.items.len,
        input_nodes,
        act_nodes,
        nodes.items.len,
        nodes.items.len,
        param_nodes,
        input_nodes,
        act_nodes,
    });

    // 2. Embedded JSON data array
    try html_buf.appendSlice(allocator,
        \\<script>
        \\const NODES_DATA = [
    );

    for (nodes.items, 0..) |n, i| {
        if (i > 0) try html_buf.appendSlice(allocator, ",\n");
        try html_buf.print(allocator,
            \\  {{ "name": "{s}", "kind": "{s}", "shape": "{s}", "elements": {d}, "bytes": {d}, "status": "{s}", "act": "{s}", "strategy": "{s}" }}
        , .{
            n.name,
            n.kind,
            n.shape_str,
            n.elements,
            n.bytes,
            n.status,
            n.inferred_act,
            n.strategy,
        });
    }

    // 3. Client-side Interactive Tree Builder & Search Script
    try html_buf.appendSlice(allocator,
        \\
        \\];
        \\
        \\const EDGES_DATA = [
    );

    for (edges.items, 0..) |e, i| {
        if (i > 0) try html_buf.appendSlice(allocator, ",\n");
        try html_buf.print(allocator,
            \\  {{ "from": "{s}", "to": "{s}", "shape": "{s}", "is_skip": {s} }}
        , .{
            e.from,
            e.to,
            e.shape,
            if (e.is_skip) "true" else "false",
        });
    }

    try html_buf.appendSlice(allocator,
        \\
        \\];
        \\
        \\const OPS_DATA = [
    );

    for (ops.items, 0..) |op, i| {
        if (i > 0) try html_buf.appendSlice(allocator, ",\n");
        try html_buf.print(allocator,
            \\  {{ "op_type": "{s}", "name": "{s}", "module": "{s}", "input_shape": "{s}", "param_shape": "{s}", "output_shape": "{s}", "elements": {d}, "bytes": {d} }}
        , .{
            op.op_type,
            op.name,
            op.module,
            op.input_shape,
            op.param_shape,
            op.output_shape,
            op.elements,
            op.bytes,
        });
    }

    try html_buf.appendSlice(allocator,
        \\
        \\];
        \\
        \\const FORMULAS_DATA = {
    );

    var formula_it = graph.module_formulas.iterator();
    var f_idx: usize = 0;
    while (formula_it.next()) |entry| {
        if (f_idx > 0) try html_buf.appendSlice(allocator, ",\n");
        const escaped_key = try escapeJsonString(allocator, entry.key_ptr.*);
        defer allocator.free(escaped_key);
        const escaped_val = try escapeJsonString(allocator, entry.value_ptr.*);
        defer allocator.free(escaped_val);
        try html_buf.print(allocator,
            \\  "{s}": "{s}"
        , .{ escaped_key, escaped_val });
        f_idx += 1;
    }

    try html_buf.appendSlice(allocator,
        \\
        \\};
        \\
        \\let currentKindFilter = 'all';
        \\let currentQuery = '';
        \\
        \\function switchMainTab(tab) {
        \\  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        \\  document.querySelectorAll('.tab-view').forEach(v => v.classList.remove('active'));
        \\  const btn = document.getElementById('btn-tab-' + tab);
        \\  const view = document.getElementById('view-' + tab);
        \\  if (btn) btn.classList.add('active');
        \\  if (view) view.classList.add('active');
        \\}
        \\
        \\function formatNumber(num) {
        \\  return num.toLocaleString();
        \\}
        \\
        \\function formatBytes(bytes) {
        \\  if (!bytes || bytes === 0) return '0 B';
        \\  if (bytes < 1024) return bytes + ' B';
        \\  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        \\  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
        \\}
        \\
        \\function getNodeOpType(name) {
        \\  const n = name.toLowerCase();
        \\  if (n.includes('ln_') || n.includes('norm')) return { icon: '📐', type: 'RMSNorm / LayerNorm', cls: 'norm' };
        \\  if (n.includes('wte') || n.includes('wpe') || n.includes('emb')) return { icon: '🔲', type: 'Embedding Table', cls: 'emb' };
        \\  if (n.includes('attn') && !n.includes('q_') && !n.includes('k_') && !n.includes('v_')) return { icon: '🔀', type: 'Causal Self-Attention', cls: 'attn' };
        \\  if (n.includes('mlp')) return { icon: '⚡', type: 'MLP / SwiGLU Block', cls: 'mlp' };
        \\  if (n.includes('q_attn') || n.includes('k_attn') || n.includes('v_attn')) return { icon: '⚙️', type: 'Multi-Head Linear Proj', cls: 'linear' };
        \\  if (n.includes('c_proj') || n.includes('c_fc') || n.includes('lm_head') || n.includes('linear')) return { icon: '⚙️', type: 'Linear (Dense)', cls: 'linear' };
        \\  if (n.includes('add') || n.includes('residual')) return { icon: '⊕', type: 'Residual Add', cls: 'add' };
        \\  return { icon: '📦', type: 'Module Block', cls: 'linear' };
        \\}
        \\
        \\function getEffectiveFormula(key) {
        \\  if (FORMULAS_DATA[key]) return { formula: FORMULAS_DATA[key], source: 'CODE_SPECIFIED' };
        \\  for (const k in FORMULAS_DATA) {
        \\    if (key.startsWith(k) || k.startsWith(key)) return { formula: FORMULAS_DATA[k], source: 'CODE_SPECIFIED' };
        \\  }
        \\  const k = key.toLowerCase();
        \\  if (k.includes('wte')) return { formula: 'y = Embedding(TokenIDs; W_e \\in \\mathbb{R}^{V \\times D}) \\rightarrow [B, T, D]', source: 'INFERRED' };
        \\  if (k.includes('wpe')) return { formula: 'y = Embedding(PosIDs; W_p \\in \\mathbb{R}^{T_{max} \\times D}) \\rightarrow [B, T, D]', source: 'INFERRED' };
        \\  if (k.includes('embeddings_sum') || (k.includes('embeddings') && k.includes('add'))) return { formula: 'x_0 = wte(tokens) + wpe(positions) \\rightarrow [B, T, D]', source: 'INFERRED' };
        \\  if (k.includes('ln_1') || k.includes('ln_2') || k.includes('ln_f') || k.includes('rms')) return { formula: 'y = \\frac{x}{\\sqrt{\\frac{1}{D} \\sum x_i^2 + \\epsilon}} \\odot \\gamma \\rightarrow [B, T, D]', source: 'INFERRED' };
        \\  if (k.includes('q_attn')) return { formula: 'Q = x \\cdot W_q^T + b_q \\quad (x \\in [B\\cdot T, D], W_q \\in [D, D]) \\rightarrow [B, nh, T, d_k]', source: 'INFERRED' };
        \\  if (k.includes('k_attn')) return { formula: 'K = x \\cdot W_k^T + b_k \\quad (x \\in [B\\cdot T, D], W_k \\in [D, D]) \\rightarrow [B, nh, T, d_k]', source: 'INFERRED' };
        \\  if (k.includes('v_attn')) return { formula: 'V = x \\cdot W_v^T + b_v \\quad (x \\in [B\\cdot T, D], W_v \\in [D, D]) \\rightarrow [B, nh, T, d_v]', source: 'INFERRED' };
        \\  if (k.includes('c_proj') && k.includes('attn')) return { formula: 'O = (A \\cdot V) \\cdot W_o^T + b_o \\quad (W_o \\in [D, D]) \\rightarrow [B, T, D]', source: 'INFERRED' };
        \\  if (k.includes('attn')) return { formula: 'A = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V \\rightarrow [B, T, D]', source: 'INFERRED' };
        \\  if (k.includes('c_fc') && k.includes('mlp')) return { formula: 'h = \\text{GELU}(x \\cdot W_{fc}^T + b_{fc}) \\quad (W_{fc} \\in [4D, D]) \\rightarrow [B, T, 4D]', source: 'INFERRED' };
        \\  if (k.includes('c_proj') && k.includes('mlp')) return { formula: 'y = h \\cdot W_{proj}^T + b_{proj} \\quad (W_{proj} \\in [D, 4D]) \\rightarrow [B, T, D]', source: 'INFERRED' };
        \\  if (k.includes('mlp')) return { formula: 'y = \\text{GELU}(x W_1 + b_1) W_2 + b_2 \\rightarrow [B, T, D]', source: 'INFERRED' };
        \\  if (k.includes('residual') || k.includes('add')) return { formula: 'x_{l+1} = x_l + \\text{Sublayer}(x_l) \\rightarrow [B, T, D]', source: 'INFERRED' };
        \\  if (k.includes('lm_head')) return { formula: 'logits = x \\cdot W_{head}^T \\quad (W_{head} \\in [V, D]) \\rightarrow [B, T, V]', source: 'INFERRED' };
        \\  return { formula: 'y = f(x; \\theta) \\rightarrow [B, T, D]', source: 'DEFAULT' };
        \\}
        \\
        \\function openInspector(key) {
        \\  const modal = document.getElementById('inspector-overlay');
        \\  const titleEl = document.getElementById('insp-title');
        \\  const subEl = document.getElementById('insp-subtitle');
        \\  const iconEl = document.getElementById('insp-icon');
        \\  const bodyEl = document.getElementById('insp-body');
        \\  if (!modal || !bodyEl) return;
        \\
        \\  const matchedNodes = NODES_DATA.filter(n => n.name === key || n.name.startsWith(key + '.'));
        \\  const matchedOps = OPS_DATA.filter(o => o.module === key || o.module.startsWith(key + '.') || o.name.startsWith(key + '.'));
        \\  const opInfo = getNodeOpType(key);
        \\  if (iconEl) iconEl.textContent = opInfo.icon;
        \\  if (titleEl) titleEl.textContent = key;
        \\
        \\  let totalParams = 0;
        \\  let totalBytes = 0;
        \\  matchedNodes.forEach(n => {
        \\    if (n.kind === 'Param') totalParams += n.elements;
        \\    totalBytes += n.bytes;
        \\  });
        \\
        \\  const paramStr = totalParams > 0 ? `${formatNumber(totalParams)} params (${formatBytes(totalBytes)})` : '0 params (Parameter-free)';
        \\  if (subEl) subEl.textContent = `${opInfo.type} · ${paramStr}`;
        \\
        \\  // 0. Module Overall Input & Output Shape Banner
        \\  const incEdges = EDGES_DATA.filter(e => e.to === key || e.to.startsWith(key + '.'));
        \\  const outEdges = EDGES_DATA.filter(e => e.from === key || e.from.startsWith(key + '.'));
        \\
        \\  let modInpShape = incEdges.length > 0 ? incEdges[0].shape : (matchedOps.length > 0 ? matchedOps[0].input_shape : 'Unknown');
        \\  let modOutShape = outEdges.length > 0 ? outEdges[outEdges.length - 1].shape : (matchedOps.length > 0 ? matchedOps[matchedOps.length - 1].output_shape : 'Unknown');
        \\
        \\  if (modInpShape === 'Unknown') {
        \\    if (key.includes('wte') || key.includes('wpe')) modInpShape = '[2, 16] (Token/Pos IDs)';
        \\    else if (key.includes('ln_1') || key.includes('attn') || key.includes('ln_2') || key.includes('mlp') || key.includes('ln_f') || key.includes('lm_head')) modInpShape = '[2, 16, 64] (Hidden State)';
        \\  }
        \\  if (modOutShape === 'Unknown') {
        \\    if (key.includes('wte') || key.includes('wpe') || key.includes('ln_1') || key.includes('attn') || key.includes('ln_2') || key.includes('mlp') || key.includes('ln_f')) modOutShape = '[2, 16, 64] (Hidden State)';
        \\    else if (key.includes('lm_head')) modOutShape = '[2, 16, 50257] (Logits)';
        \\  }
        \\
        \\  let ioBannerHtml = `
        \\    <div class="insp-io-banner">
        \\      <div class="insp-io-card input">
        \\        <div class="insp-io-label">📥 Module Input Shape (模块输入维度)</div>
        \\        <div class="insp-io-shape">${modInpShape}</div>
        \\        <div class="insp-io-sub">Incoming tensor fed into ${key}</div>
        \\      </div>
        \\      <div class="insp-io-arrow">➔</div>
        \\      <div class="insp-io-card output">
        \\        <div class="insp-io-label">📤 Module Output Shape (模块输出维度)</div>
        \\        <div class="insp-io-shape">${modOutShape}</div>
        \\        <div class="insp-io-sub">Final tensor output from ${key}</div>
        \\      </div>
        \\    </div>
        \\  `;
        \\
        \\  // 0.5 Mathematical Vector Transformation Formula Section (代码中指定的数学运算公式)
        \\  const formObj = getEffectiveFormula(key);
        \\  const isSpecified = formObj.source === 'CODE_SPECIFIED';
        \\  const badgeLabel = isSpecified ? 'CODE SPECIFIED (代码显式定义)' : 'INFERRED (框架推导公式)';
        \\  const badgeCls = isSpecified ? 'insp-formula-badge specified' : 'insp-formula-badge inferred';
        \\  let formulaHtml = `
        \\    <div class="insp-formula-box">
        \\      <div class="insp-formula-header">
        \\        <span>📐 向量变换与运算数学公式 (Mathematical Vector Transformation Formula)</span>
        \\        <span class="${badgeCls}">${badgeLabel}</span>
        \\      </div>
        \\      <div class="insp-formula-display">
        \\        <span style="font-size:18px;">📐</span>
        \\        <span id="insp-katex-target" style="flex:1;">${formObj.formula}</span>
        \\      </div>
        \\      <div class="insp-formula-desc">此公式清晰反映本模块在正向传播时对输入张量所执行的线性投影、非线性激活或注意力加权变换关系。</div>
        \\    </div>
        \\  `;
        \\
        \\  // 1. Detailed Operation / Node Shape Table (Each Op's Input, Param, Output Shape)
        \\  let opsHtml = '';
        \\  if (matchedOps.length > 0) {
        \\    opsHtml = `
        \\      <div class="insp-section">
        \\        <div class="insp-section-title">
        \\          <span>⚡ Operations & Layer Dimensions (每个算子/节点的输入·参数·输出 Shape)</span>
        \\          <span style="font-size: 11px; font-weight: normal; color: #94a3b8;">${matchedOps.length} Operations executed</span>
        \\        </div>
        \\        <table class="node-table">
        \\          <thead>
        \\            <tr>
        \\              <th>#</th>
        \\              <th>Operation / Node Name</th>
        \\              <th>Op Type</th>
        \\              <th style="color:#38bdf8;">📥 输入 shape (Input)</th>
        \\              <th style="color:#fbbf24;">⚙️ 输入参数 shape (Params)</th>
        \\              <th style="color:#34d399;">📤 输出 shape (Output)</th>
        \\              <th>Memory</th>
        \\            </tr>
        \\          </thead>
        \\          <tbody>
        \\    `;
        \\    matchedOps.forEach((op, idx) => {
        \\      const shortName = op.name.startsWith(key + '.') ? op.name.slice(key.length + 1) : op.name;
        \\      opsHtml += `
        \\        <tr>
        \\          <td style="font-family:var(--font-mono); color:#64748b;">${idx + 1}</td>
        \\          <td class="node-name" title="${op.name}">${shortName}</td>
        \\          <td><span class="badge badge-op">${op.op_type}</span></td>
        \\          <td class="node-shape" style="color:#38bdf8;">${op.input_shape}</td>
        \\          <td class="node-shape" style="color:#fbbf24; font-size:11px;">${op.param_shape}</td>
        \\          <td class="node-shape" style="color:#34d399; font-weight:600;">${op.output_shape}</td>
        \\          <td style="font-family:var(--font-mono); color:#94a3b8;">${formatBytes(op.bytes)}</td>
        \\        </tr>
        \\      `;
        \\    });
        \\    opsHtml += '</tbody></table></div>';
        \\  }
        \\
        \\  // 2. Parameters Section
        \\  const paramNodes = matchedNodes.filter(n => n.kind === 'Param');
        \\  let paramsHtml = `<div class="insp-section"><div class="insp-section-title"><span>⚙️ Parameters (权重与偏置参数详情)</span><span style="font-size: 11px; font-weight: normal; color: #94a3b8;">Total: ${formatNumber(totalParams)}</span></div>`;
        \\  if (paramNodes.length > 0) {
        \\    paramsHtml += '<table class="node-table"><thead><tr><th>Parameter Name</th><th>Shape</th><th>Elements</th><th>Memory</th><th>Init Strategy</th><th>Status</th></tr></thead><tbody>';
        \\    paramNodes.forEach(p => {
        \\      const statusBadge = p.status === 'CUSTOM_INIT' ? 'badge-custom' : 'badge-auto';
        \\      paramsHtml += `<tr><td class="node-name">${p.name}</td><td class="node-shape">${p.shape}</td><td style="font-family:var(--font-mono);">${formatNumber(p.elements)}</td><td style="font-family:var(--font-mono);">${formatBytes(p.bytes)}</td><td class="strategy-col">${p.strategy}</td><td><span class="badge ${statusBadge}">${p.status}</span></td></tr>`;
        \\    });
        \\    paramsHtml += '</tbody></table>';
        \\  } else {
        \\    paramsHtml += '<div style="font-size: 12px; color: #94a3b8; font-style: italic;">No trainable parameters (stateless / parameter-free operation).</div>';
        \\  }
        \\  paramsHtml += '</div>';
        \\
        \\  // 3. Inputs Section (Incoming Graph Edges)
        \\  const incoming = EDGES_DATA.filter(e => e.to === key || e.to.startsWith(key + '.') || (key.startsWith(e.to) && e.to.length > 5));
        \\  const inputNodes = matchedNodes.filter(n => n.kind === 'Input');
        \\
        \\  let inputsHtml = '<div class="insp-section"><div class="insp-section-title"><span>📥 Inbound Connections (外部输入依赖流)</span></div>';
        \\  if (incoming.length > 0 || inputNodes.length > 0) {
        \\    inputsHtml += '<table class="node-table"><thead><tr><th>Source Tensor / Predecessor</th><th>Target Port</th><th>Shape</th><th>Connection Type</th></tr></thead><tbody>';
        \\    incoming.forEach(e => {
        \\      const typeBadge = e.is_skip 
        \\        ? '<span class="badge" style="background:#0284c7;color:#fff;">⚡ Residual Skip Highway</span>'
        \\        : '<span class="badge" style="background:#334155;color:#94a3b8;">▼ Sequential Flow</span>';
        \\      inputsHtml += `<tr><td class="node-name">${e.from}</td><td class="node-name">${e.to}</td><td class="node-shape">${e.shape}</td><td>${typeBadge}</td></tr>`;
        \\    });
        \\    inputNodes.forEach(inp => {
        \\      inputsHtml += `<tr><td class="node-name">${inp.name}</td><td class="node-name">${key}</td><td class="node-shape">${inp.shape}</td><td><span class="badge badge-input">User / Model Input</span></td></tr>`;
        \\    });
        \\    inputsHtml += '</tbody></table>';
        \\  } else {
        \\    let impliedInp = 'Inherits sequential output from predecessor module in forward graph.';
        \\    if (key.includes('ln_1')) impliedInp = 'Input tensor <code>x</code> (Transformer block input [2, 16, 64])';
        \\    else if (key.includes('attn')) impliedInp = 'Normalized tensor from <code>ln_1</code> [2, 16, 64]';
        \\    else if (key.includes('ln_2')) impliedInp = 'Stage 1 attention residual sum <code>x1</code> [2, 16, 64]';
        \\    else if (key.includes('mlp')) impliedInp = 'Normalized tensor from <code>ln_2</code> [2, 16, 64]';
        \\    else if (key.includes('wte') || key.includes('wpe')) impliedInp = 'Integer token IDs / Position indices <code>[2, 16]</code>';
        \\    else if (key.includes('ln_f')) impliedInp = 'Final transformer block output [2, 16, 64]';
        \\    else if (key.includes('lm_head')) impliedInp = 'Normalized representation from <code>ln_f</code> [2, 16, 64]';
        \\    inputsHtml += `<div style="font-size: 12px; color: #cbd5e1; padding: 6px 0;">${impliedInp}</div>`;
        \\  }
        \\  inputsHtml += '</div>';
        \\
        \\  // 4. Outputs & Activations Section
        \\  const actNodes = matchedNodes.filter(n => n.kind === 'Activation' || n.kind === 'Output');
        \\  let actsHtml = `<div class="insp-section"><div class="insp-section-title"><span>📤 Output Tensors & Intermediate Activations</span><span style="font-size: 11px; font-weight: normal; color: #94a3b8;">${actNodes.length} tensors</span></div>`;
        \\  if (actNodes.length > 0) {
        \\    actsHtml += '<table class="node-table"><thead><tr><th>Tensor Name</th><th>Shape</th><th>Elements</th><th>Inferred Op</th><th>Memory</th></tr></thead><tbody>';
        \\    actNodes.forEach(a => {
        \\      actsHtml += `<tr><td class="node-name">${a.name}</td><td class="node-shape">${a.shape}</td><td style="font-family:var(--font-mono);">${formatNumber(a.elements)}</td><td style="font-family:var(--font-mono);"><span class="badge badge-op">${a.act}</span></td><td style="font-family:var(--font-mono);">${formatBytes(a.bytes)}</td></tr>`;
        \\    });
        \\    actsHtml += '</tbody></table>';
        \\  } else {
        \\    actsHtml += '<div style="font-size: 12px; color: #94a3b8; font-style: italic;">Outputs flow directly to subsequent operator without standalone intermediate caching.</div>';
        \\  }
        \\  actsHtml += '</div>';
        \\
        \\  // 5. Downstream Connections
        \\  const outgoing = EDGES_DATA.filter(e => e.from === key || e.from.startsWith(key + '.'));
        \\  let outHtml = '';
        \\  if (outgoing.length > 0) {
        \\    outHtml += '<div class="insp-section"><div class="insp-section-title"><span>🔀 Downstream Flow (下游流向)</span></div><table class="node-table"><thead><tr><th>Consumer Node</th><th>Shape</th><th>Connection Type</th></tr></thead><tbody>';
        \\    outgoing.forEach(e => {
        \\      const typeBadge = e.is_skip 
        \\        ? '<span class="badge" style="background:#0284c7;color:#fff;">⚡ Residual Skip Highway</span>'
        \\        : '<span class="badge" style="background:#334155;color:#94a3b8;">▼ Sequential Flow</span>';
        \\      outHtml += `<tr><td class="node-name">${e.to}</td><td class="node-shape">${e.shape}</td><td>${typeBadge}</td></tr>`;
        \\    });
        \\    outHtml += '</tbody></table></div>';
        \\  }
        \\
        \\  bodyEl.innerHTML = ioBannerHtml + formulaHtml + opsHtml + paramsHtml + inputsHtml + actsHtml + outHtml;
        \\  modal.classList.add('open');
        \\
        \\  // KaTeX LaTeX Math Rendering
        \\  try {
        \\    const target = document.getElementById('insp-katex-target');
        \\    if (target && window.katex) {
        \\      katex.render(formObj.formula, target, {
        \\        throwOnError: false,
        \\        displayMode: true
        \\      });
        \\    }
        \\  } catch(err) {
        \\    console.warn('KaTeX render fallback:', err);
        \\  }
        \\}
        \\
        \\function closeInspector(e) {
        \\  if (e && e.target && e.target.id !== 'inspector-overlay' && !e.target.classList.contains('inspector-close-btn')) return;
        \\  const modal = document.getElementById('inspector-overlay');
        \\  if (modal) modal.classList.remove('open');
        \\}
        \\
        \\document.addEventListener('keydown', (e) => {
        \\  if (e.key === 'Escape') {
        \\    const modal = document.getElementById('inspector-overlay');
        \\    if (modal) modal.classList.remove('open');
        \\  }
        \\});
        \\
        \\function renderDagView() {
        \\  const container = document.getElementById('dag-container');
        \\  if (!container) return;
        \\
        \\  // Map module to total parameters and primary tensor info
        \\  const modParams = {};
        \\  NODES_DATA.forEach(n => {
        \\    const mod = n.name.split('.').slice(0, -1).join('.');
        \\    if (!modParams[mod]) modParams[mod] = 0;
        \\    if (n.kind === 'Param') modParams[mod] += n.elements;
        \\  });
        \\
        \\  // Group edges into incoming lists per target
        \\  const incoming = {};
        \\  EDGES_DATA.forEach(e => {
        \\    if (!incoming[e.to]) incoming[e.to] = [];
        \\    incoming[e.to].push(e);
        \\  });
        \\
        \\  // Step sequence of key target checkpoints in topological order
        \\  const stageTargets = [
        \\    'gpt.wte',
        \\    'gpt.embeddings_sum',
        \\    'gpt.layers.0.residual_attn',
        \\    'gpt.layers.0.output',
        \\    'gpt.layers.1.residual_attn',
        \\    'gpt.layers.1.output',
        \\    'gpt.ln_f',
        \\    'gpt.lm_head',
        \\    'outputs.logits'
        \\  ];
        \\
        \\  // If model has different layers, dynamically find all convergence/checkpoint nodes
        \\  const seenTargets = new Set();
        \\  const stages = [];
        \\  stageTargets.forEach(st => {
        \\    if (incoming[st] || EDGES_DATA.some(e => e.from === st)) {
        \\      stages.push(st);
        \\      seenTargets.add(st);
        \\    }
        \\  });
        \\  EDGES_DATA.forEach(e => {
        \\    if (!seenTargets.has(e.to)) {
        \\      stages.push(e.to);
        \\      seenTargets.add(e.to);
        \\    }
        \\  });
        \\
        \\  let html = '';
        \\  const renderedModules = new Set();
        \\
        \\  stages.forEach((target, sIdx) => {
        \\    const inc = incoming[target] || [];
        \\    const hasSkip = inc.some(e => e.is_skip);
        \\    const isMultiBranch = inc.length > 1;
        \\
        \\    if (isMultiBranch && hasSkip) {
        \\      // Render parallel branch stage: [Module A (Skip / Shortcut)] | [Module B (Transform)]
        \\      const skipEdge = inc.find(e => e.is_skip);
        \\      const transformEdge = inc.find(e => !e.is_skip);
        \\
        \\      const isAttnBlock = target.endsWith('.residual_attn');
        \\      const blockLabel = isAttnBlock ? 'Residual Attention Sub-Layer' : 'Residual Feed-Forward (MLP) Sub-Layer';
        \\      const layerMatch = target.match(/layers\.(\d+)/);
        \\      const layerNum = layerMatch ? `Layer ${layerMatch[1]}` : 'Block';
        \\
        \\      html += `
        \\        <div class="flow-stage-parallel">
        \\          <div class="parallel-header">
        \\            <span>${layerNum} · ${blockLabel}</span>
        \\            <span class="flow-card-shape">${skipEdge ? skipEdge.shape : ''}</span>
        \\          </div>
        \\          <div class="parallel-branches-row">
        \\            <!-- Branch 1: Skip Connection -->
        \\            <div class="parallel-branch-col skip-branch">
        \\              <span class="branch-badge skip">⚡ Shortcut (Skip)</span>
        \\              <div style="margin-top: 8px;">
        \\                <div class="flow-card-title">${skipEdge ? skipEdge.from : 'Identity'}</div>
        \\                <div style="font-size: 11px; color: #38bdf8; margin-top: 4px;">Direct Identity Connection</div>
        \\                <div style="font-size: 11px; color: var(--text-sub); margin-top: 2px;">Preserves gradient highway (x)</div>
        \\              </div>
        \\            </div>
        \\
        \\            <div class="branch-divider">|</div>
        \\
        \\            <!-- Branch 2: Transform Sub-Layer -->
        \\            <div class="parallel-branch-col transform-branch">
        \\              <span class="branch-badge transform">⚙️ Transform Branch</span>
        \\              <div style="margin-top: 8px;">
        \\                <div class="flow-card-title">${transformEdge ? transformEdge.from : 'Sublayer'}</div>
        \\                <div style="font-size: 11px; color: #a5b4fc; margin-top: 4px;">LayerNorm + ${isAttnBlock ? 'Multi-Head Attention' : 'SwiGLU / MLP'}</div>
        \\                <div style="font-size: 11px; color: var(--text-sub); margin-top: 2px;">F(x) Feature Extraction</div>
        \\              </div>
        \\            </div>
        \\          </div>
        \\
        \\          <!-- Convergence / Merge Downward -->
        \\          <div class="converge-arrow-box">
        \\            <div class="flow-arrow-down">
        \\              <span class="flow-arrow-head">▼</span>
        \\              <span class="flow-arrow-label">⊕ Element-wise Add (x + F(x))</span>
        \\            </div>
        \\            <div class="converge-card">
        \\              <div class="converge-title">
        \\                <span>⊕</span>
        \\                <span>${target}</span>
        \\              </div>
        \\              <div style="font-size: 11px; color: var(--text-sub); margin-top: 4px;">
        \\                Residual Sum Output · Shape: <span class="flow-card-shape">${skipEdge ? skipEdge.shape : ''}</span>
        \\              </div>
        \\            </div>
        \\          </div>
        \\        </div>
        \\      `;
        \\      renderedModules.add(target);
        \\    } else if (target === 'gpt.embeddings_sum') {
        \\      // Embedding combine stage: Token Embeddings | Positional Embeddings -> embeddings_sum
        \\      const wteEdge = inc.find(e => e.from.includes('wte'));
        \\      const wpeEdge = inc.find(e => e.from.includes('wpe'));
        \\      html += `
        \\        <div class="flow-stage-parallel">
        \\          <div class="parallel-header">
        \\            <span>Embedding Stage · Input Projection</span>
        \\            <span class="flow-card-shape">${wteEdge ? wteEdge.shape : ''}</span>
        \\          </div>
        \\          <div class="parallel-branches-row">
        \\            <div class="parallel-branch-col">
        \\              <span class="branch-badge transform">Token Embedding</span>
        \\              <div style="margin-top: 8px;">
        \\                <div class="flow-card-title">gpt.wte</div>
        \\                <div style="font-size: 11px; color: var(--text-sub); margin-top: 4px;">Vocabulary lookup table</div>
        \\              </div>
        \\            </div>
        \\            <div class="branch-divider">|</div>
        \\            <div class="parallel-branch-col">
        \\              <span class="branch-badge transform">Positional Embedding</span>
        \\              <div style="margin-top: 8px;">
        \\                <div class="flow-card-title">gpt.wpe</div>
        \\                <div style="font-size: 11px; color: var(--text-sub); margin-top: 4px;">Learned position table</div>
        \\              </div>
        \\            </div>
        \\          </div>
        \\          <div class="converge-arrow-box">
        \\            <div class="flow-arrow-down">
        \\              <span class="flow-arrow-head">▼</span>
        \\              <span class="flow-arrow-label">⊕ Sum (token + pos)</span>
        \\            </div>
        \\            <div class="converge-card">
        \\              <div class="converge-title">
        \\                <span>⊕</span>
        \\                <span>gpt.embeddings_sum</span>
        \\              </div>
        \\              <div style="font-size: 11px; color: var(--text-sub); margin-top: 4px;">Combined Latent Representation</div>
        \\            </div>
        \\          </div>
        \\        </div>
        \\      `;
        \\      renderedModules.add(target);
        \\    } else {
        \\      // Linear stage card
        \\      const pCount = modParams[target] || 0;
        \\      const pMeta = pCount > 0 ? `${formatNumber(pCount)} params` : '';
        \\      const edge = inc[0];
        \\      const shapeText = edge ? edge.shape : '';
        \\
        \\      html += `
        \\        <div class="flow-card flow-card-linear">
        \\          <div class="flow-card-header">
        \\            <span class="flow-card-title">${target}</span>
        \\            ${shapeText ? `<span class="flow-card-shape">${shapeText}</span>` : ''}
        \\          </div>
        \\          <div class="flow-card-body">
        \\            <span>${pMeta ? `⚙️ ${pMeta}` : 'Forward checkpoint'}</span>
        \\            <span style="font-size: 11px; color: #38bdf8;">Sequential</span>
        \\          </div>
        \\        </div>
        \\      `;
        \\      renderedModules.add(target);
        \\    }
        \\
        \\    // Downward arrow connector to next stage
        \\    if (sIdx < stages.length - 1) {
        \\      html += `
        \\        <div class="flow-arrow-down">
        \\          <div class="flow-arrow-line"></div>
        \\          <span class="flow-arrow-head">▼</span>
        \\        </div>
        \\      `;
        \\    }
        \\  });
        \\
        \\  container.innerHTML = html;
        \\}
        \\
        \\function renderMermaidView() {
        \\  const container = document.getElementById('mermaid-code');
        \\  if (!container) return;
        \\
        \\  let code = 'flowchart TD\n';
        \\  code += '  subgraph Inputs["Inputs"]\n';
        \\  code += '    token_ids["inputs.token_ids"]\n';
        \\  code += '  end\n\n';
        \\
        \\  const edgesSeen = new Set();
        \\  EDGES_DATA.forEach(e => {
        \\    const fromSafe = e.from.replace(/[^a-zA-Z0-9_]/g, '_');
        \\    const toSafe = e.to.replace(/[^a-zA-Z0-9_]/g, '_');
        \\    const edgeKey = fromSafe + '->' + toSafe;
        \\    if (edgesSeen.has(edgeKey)) return;
        \\    edgesSeen.add(edgeKey);
        \\
        \\    if (e.is_skip) {
        \\      code += `  ${fromSafe}["${e.from}"] -.->|⚡ Skip Shortcut| ${toSafe}["${e.to}"]\n`;
        \\    } else {
        \\      code += `  ${fromSafe}["${e.from}"] -->|${e.shape}| ${toSafe}["${e.to}"]\n`;
        \\    }
        \\  });
        \\
        \\  container.textContent = code;
        \\}
        \\
        \\function buildHierarchy(data) {
        \\  const root = { _children: {}, _nodes: [], _paramCount: 0, _bytes: 0 };
        \\
        \\  data.forEach(item => {
        \\    const parts = item.name.split('.');
        \\    if (parts.length === 1) {
        \\      root._nodes.push(item);
        \\      if (item.kind === 'Param') root._paramCount += item.elements;
        \\      root._bytes += item.bytes;
        \\      return;
        \\    }
        \\
        \\    let curr = root;
        \\    for (let i = 0; i < parts.length - 1; i++) {
        \\      const p = parts[i];
        \\      if (!curr._children[p]) {
        \\        curr._children[p] = { _children: {}, _nodes: [], _paramCount: 0, _bytes: 0 };
        \\      }
        \\      if (item.kind === 'Param') curr._children[p]._paramCount += item.elements;
        \\      curr._children[p]._bytes += item.bytes;
        \\      curr = curr._children[p];
        \\    }
        \\    curr._nodes.push(item);
        \\  });
        \\
        \\  return root;
        \\}
        \\
        \\function renderBranch(prefix, node) {
        \\  let html = '';
        \\  const childKeys = Object.keys(node._children);
        \\  
        \\  if (node._nodes.length > 0 || childKeys.length > 0) {
        \\    const hasParams = node._paramCount > 0;
        \\    const metaInfo = hasParams ? `${formatNumber(node._paramCount)} params` : `${node._nodes.length} nodes`;
        \\    const isRoot = prefix.includes('Root') || prefix.includes('Global');
        \\    const titleDisplay = isRoot ? `🌐 ${prefix}` : `📁 ${prefix}`;
        \\    
        \\    html += `
        \\      <details class="module-group" open data-module="${prefix.toLowerCase()}">
        \\        <summary class="module-header">
        \\          <div class="module-title-box">
        \\            <span class="chevron">▶</span>
        \\            <span class="module-name">${titleDisplay}</span>
        \\          </div>
        \\          <div class="module-meta">
        \\            <span>${metaInfo}</span>
        \\          </div>
        \\        </summary>
        \\        <div class="module-content">
        \\    `;
        \\
        \\    const isDecoderLayer = /^gpt\.layers\.\d+$/.test(prefix) || /layers\.\d+$/.test(prefix);
        \\    const isAttentionLayer = prefix.endsWith('.attn') || (node._children['q_attn'] && node._children['k_attn'] && node._children['v_attn']);
        \\    const isLeafOrSpecial = !isDecoderLayer && !isAttentionLayer && (node._nodes.length > 0 || childKeys.length === 0);
        \\
        \\    // TensorBoard node card for leaf/op modules (No static table: inspect in standalone modal)
        \\    if (isLeafOrSpecial && !isRoot) {
        \\      const opInfo = getNodeOpType(prefix);
        \\      html += `
        \\        <div class="tb-node-card" onclick="openInspector('${prefix}')" title="Click to view inputs, parameters & outputs in standalone window" style="margin-bottom: 10px;">
        \\          <div class="tb-node-header">
        \\            <div class="tb-node-title">
        \\              <span class="tb-op-icon ${opInfo.cls}">${opInfo.icon}</span>
        \\              <span>${prefix}</span>
        \\            </div>
        \\            <div style="display: flex; align-items: center; gap: 8px;">
        \\              <span class="tb-type-pill">${opInfo.type}</span>
        \\              ${node._paramCount > 0 ? `<span class="tb-param-chip">⚙️ ${formatNumber(node._paramCount)} params</span>` : ''}
        \\              <button class="tb-inspect-btn" onclick="event.stopPropagation(); openInspector('${prefix}')">🔍 Inspect</button>
        \\            </div>
        \\          </div>
        \\        </div>
        \\      `;
        \\    }
        \\
        \\    if (isRoot && node._nodes.length > 0) {
        \\      node._nodes.forEach(n => {
        \\        const opInfo = getNodeOpType(n.name);
        \\        html += `
        \\          <div class="tb-node-card" onclick="openInspector('${n.name}')" title="Click to view inputs, parameters & outputs in standalone window" style="margin-bottom: 8px;">
        \\            <div class="tb-node-header">
        \\              <div class="tb-node-title">
        \\                <span class="tb-op-icon ${opInfo.cls}">${opInfo.icon}</span>
        \\                <span>${n.name}</span>
        \\              </div>
        \\              <div style="display: flex; align-items: center; gap: 8px;">
        \\                <span class="tb-type-pill">${n.kind}</span>
        \\                <span class="tb-shape-chip">${n.shape}</span>
        \\                <button class="tb-inspect-btn" onclick="event.stopPropagation(); openInspector('${n.name}')">🔍 Inspect</button>
        \\              </div>
        \\            </div>
        \\          </div>
        \\        `;
        \\      });
        \\    }
        \\
        \\    if (isAttentionLayer) {
        \\      const qChild = node._children['q_attn'];
        \\      const kChild = node._children['k_attn'];
        \\      const vChild = node._children['v_attn'];
        \\      const cprojChild = node._children['c_proj'];
        \\
        \\      html += `
        \\        <div class="tb-node-card" onclick="openInspector('${prefix}')" title="Click to inspect Causal Self-Attention module" style="margin-bottom: 12px;">
        \\          <div class="tb-node-header">
        \\            <div class="tb-node-title">
        \\              <span class="tb-op-icon attn">🔀</span>
        \\              <span>${prefix}</span>
        \\            </div>
        \\            <div style="display: flex; align-items: center; gap: 8px;">
        \\              <span class="tb-type-pill">Causal Self-Attention</span>
        \\              ${node._paramCount > 0 ? `<span class="tb-param-chip">⚙️ ${formatNumber(node._paramCount)} params</span>` : ''}
        \\              <button class="tb-inspect-btn" onclick="event.stopPropagation(); openInspector('${prefix}')">🔍 Inspect</button>
        \\            </div>
        \\          </div>
        \\        </div>
        \\      `;
        \\
        \\      if (qChild || kChild || vChild) {
        \\        html += `
        \\          <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin: 4px 0 6px 0; letter-spacing: 0.5px; display: flex; align-items: center; gap: 6px;">
        \\            <span>🔀 Step 1: Parallel Multi-Head Projections (x ➔ Q, K, V)</span>
        \\          </div>
        \\          <div class="tree-branches-row" style="gap: 10px;">
        \\            ${qChild ? `
        \\              <div class="tree-branch-col" style="flex: 1; min-width: 0; border-color: #38bdf8; background: rgba(56, 189, 248, 0.03);">
        \\                <span class="tree-branch-badge" style="background: #0284c7; color: #fff; border: 1px solid #38bdf8;">Query (Q)</span>
        \\                <div style="margin-top: 8px;">
        \\                  ${renderBranch(`${prefix}.q_attn`, qChild)}
        \\                </div>
        \\              </div>
        \\            ` : ''}
        \\
        \\            ${qChild && kChild ? `<div class="tree-divider" style="color: #64748b;">|</div>` : ''}
        \\
        \\            ${kChild ? `
        \\              <div class="tree-branch-col" style="flex: 1; min-width: 0; border-color: #38bdf8; background: rgba(56, 189, 248, 0.03);">
        \\                <span class="tree-branch-badge" style="background: #0284c7; color: #fff; border: 1px solid #38bdf8;">Key (K)</span>
        \\                <div style="margin-top: 8px;">
        \\                  ${renderBranch(`${prefix}.k_attn`, kChild)}
        \\                </div>
        \\              </div>
        \\            ` : ''}
        \\
        \\            ${kChild && vChild ? `<div class="tree-divider" style="color: #64748b;">|</div>` : ''}
        \\
        \\            ${vChild ? `
        \\              <div class="tree-branch-col" style="flex: 1; min-width: 0; border-color: #38bdf8; background: rgba(56, 189, 248, 0.03);">
        \\                <span class="tree-branch-badge" style="background: #0284c7; color: #fff; border: 1px solid #38bdf8;">Value (V)</span>
        \\                <div style="margin-top: 8px;">
        \\                  ${renderBranch(`${prefix}.v_attn`, vChild)}
        \\                </div>
        \\              </div>
        \\            ` : ''}
        \\          </div>
        \\
        \\          <div class="tree-flow-connector" style="justify-content: center; margin: 8px 0;">
        \\            <span class="flow-arrow-text">
        \\              <span class="arrow-symbol">▼</span>
        \\              <span>Parallel Q, K, V representations converge into Scaled Dot-Product Attention</span>
        \\            </span>
        \\          </div>
        \\        `;
        \\      }
        \\
        \\      if (node._nodes.length > 0) {
        \\        html += `
        \\          <div class="tb-node-card" onclick="openInspector('${prefix}')" title="Click to inspect attention core activations in standalone window" style="border-color: #818cf8; background: rgba(99, 102, 241, 0.06); margin: 6px 0;">
        \\            <div class="tb-node-header">
        \\              <div class="tb-node-title">
        \\                <span class="tb-op-icon attn">🎯</span>
        \\                <span style="color: #a5b4fc; font-size: 13px;">Step 2: Scaled Dot-Product Attention Core</span>
        \\              </div>
        \\              <div style="display: flex; align-items: center; gap: 8px;">
        \\                <span class="tb-type-pill" style="border-color: #818cf8; color: #a5b4fc;">Softmax((Q·Kᵀ)/√d + Mask)·V</span>
        \\                <button class="tb-inspect-btn" onclick="event.stopPropagation(); openInspector('${prefix}')">🔍 Inspect</button>
        \\              </div>
        \\            </div>
        \\          </div>
        \\        `;
        \\      }
        \\
        \\      if (cprojChild) {
        \\        html += `
        \\          <div class="tree-flow-connector" style="justify-content: center; margin: 8px 0;">
        \\            <span class="flow-arrow-text">
        \\              <span class="arrow-symbol">▼</span>
        \\              <span>Step 3: Multi-Head Attention representation feeds into Linear Output Projection</span>
        \\            </span>
        \\          </div>
        \\          ${renderBranch(`${prefix}.c_proj`, cprojChild)}
        \\        `;
        \\      }
        \\    }
        \\
        \\    if (isDecoderLayer) {
        \\      const attnChild = node._children['attn'];
        \\      const ln1Child = node._children['ln_1'];
        \\      const mlpChild = node._children['mlp'];
        \\      const ln2Child = node._children['ln_2'];
        \\
        \\      const resAttnNode = node._nodes.find(n => n.name.endsWith('.residual_attn'));
        \\      const outputNode = node._nodes.find(n => n.name.endsWith('.output'));
        \\
        \\      if (attnChild) {
        \\        const attnShape = resAttnNode ? resAttnNode.shape : '[2, 16, 64]';
        \\        html += `
        \\          <div class="tb-stage-card">
        \\            <div class="tb-stage-header">
        \\              <div style="display: flex; align-items: center; gap: 8px;">
        \\                <span class="tb-stage-badge">Stage 1</span>
        \\                <span class="tb-stage-name">Self-Attention Residual Sublayer</span>
        \\              </div>
        \\              <span class="tb-shape-chip">${attnShape}</span>
        \\            </div>
        \\            <div class="tree-branches-row">
        \\              <!-- Branch A: Shortcut (Skip) x -->
        \\              <div class="tree-branch-col tree-branch-skip" onclick="openInspector('${prefix}.residual_attn')" title="Click to inspect Shortcut connection" style="cursor: pointer;">
        \\                <span class="tree-branch-badge skip">⚡ Shortcut (Skip) x</span>
        \\                <div style="font-family: var(--font-mono); font-weight: 700; color: #38bdf8; font-size: 12px; margin-top: 6px;">
        \\                  Identity Highway
        \\                </div>
        \\                <div style="font-size: 11px; color: var(--text-sub); margin-top: 3px;">
        \\                  Preserves input tensor <code>x</code>
        \\                </div>
        \\                <div style="margin-top: auto; padding-top: 8px;">
        \\                  <span class="tb-shape-chip">${attnShape}</span>
        \\                </div>
        \\              </div>
        \\
        \\              <div class="tree-divider">|</div>
        \\
        \\              <!-- Branch B: Transform Branch F(x) (ln_1 -> attn) -->
        \\              <div class="tree-branch-col tree-branch-transform">
        \\                <span class="tree-branch-badge transform">⚙️ Transform Branch F(x)</span>
        \\                <div style="font-size: 11px; color: #a5b4fc; margin: 4px 0 8px 0; font-family: var(--font-mono);">
        \\                  (RMSNorm ➔ Attention)
        \\                </div>
        \\                <div style="display: flex; flex-direction: column; gap: 8px;">
        \\                  ${ln1Child ? renderBranch(`${prefix}.ln_1`, ln1Child) : ''}
        \\                  <div class="tree-flow-connector" style="padding: 2px 8px;">
        \\                    <span class="flow-arrow-text"><span class="arrow-symbol">▼</span> Normalized Activation to Attention</span>
        \\                  </div>
        \\                  ${attnChild ? renderBranch(`${prefix}.attn`, attnChild) : ''}
        \\                </div>
        \\              </div>
        \\            </div>
        \\
        \\            <!-- Downward Convergence: x + F(x) -->
        \\            <div class="tree-converge-box" onclick="openInspector('${resAttnNode ? resAttnNode.name : prefix + '.residual_attn'}')" style="cursor: pointer;">
        \\              <div class="flow-arrow-down" style="padding: 2px 0;">
        \\                <span class="flow-arrow-head">▼</span>
        \\                <span class="flow-arrow-label">⊕ Element-wise Add: x + F(x)</span>
        \\              </div>
        \\              <div class="tree-converge-card" style="width: 100%; justify-content: space-between;">
        \\                <div style="display: flex; align-items: center; gap: 8px;">
        \\                  <span>⊕</span>
        \\                  <span>Residual Add: x + F(x)</span>
        \\                  <span class="flow-card-shape">${attnShape}</span>
        \\                </div>
        \\                <button class="tb-inspect-btn" onclick="event.stopPropagation(); openInspector('${resAttnNode ? resAttnNode.name : prefix + '.residual_attn'}')">🔍 Inspect</button>
        \\              </div>
        \\            </div>
        \\          </div>
        \\        `;
        \\      }
        \\
        \\      if (attnChild && mlpChild) {
        \\        html += `
        \\          <div class="tree-flow-connector" style="justify-content: center; margin: 8px 0;">
        \\            <span class="flow-arrow-text">
        \\              <span class="arrow-symbol">▼</span>
        \\              <span>Output x feeds into Stage 2 (MLP Block)</span>
        \\            </span>
        \\          </div>
        \\        `;
        \\      }
        \\
        \\      if (mlpChild) {
        \\        const mlpShape = outputNode ? outputNode.shape : '[2, 16, 64]';
        \\        html += `
        \\          <div class="tb-stage-card">
        \\            <div class="tb-stage-header">
        \\              <div style="display: flex; align-items: center; gap: 8px;">
        \\                <span class="tb-stage-badge" style="background: #7c3aed;">Stage 2</span>
        \\                <span class="tb-stage-name">Feed-Forward (MLP) Residual Sublayer</span>
        \\              </div>
        \\              <span class="tb-shape-chip">${mlpShape}</span>
        \\            </div>
        \\            <div class="tree-branches-row">
        \\              <!-- Branch A: Shortcut (Skip) x -->
        \\              <div class="tree-branch-col tree-branch-skip" onclick="openInspector('${prefix}.output')" title="Click to inspect Shortcut connection" style="cursor: pointer;">
        \\                <span class="tree-branch-badge skip">⚡ Shortcut (Skip) x</span>
        \\                <div style="font-family: var(--font-mono); font-weight: 700; color: #38bdf8; font-size: 12px; margin-top: 6px;">
        \\                  Attn Residual Highway
        \\                </div>
        \\                <div style="font-size: 11px; color: var(--text-sub); margin-top: 3px;">
        \\                  Stage 1 output tensor <code>x</code>
        \\                </div>
        \\                <div style="margin-top: auto; padding-top: 8px;">
        \\                  <span class="tb-shape-chip">${mlpShape}</span>
        \\                </div>
        \\              </div>
        \\
        \\              <div class="tree-divider">|</div>
        \\
        \\              <!-- Branch B: Transform Branch F(x) (ln_2 -> mlp) -->
        \\              <div class="tree-branch-col tree-branch-transform">
        \\                <span class="tree-branch-badge transform">⚙️ Transform Branch F(x)</span>
        \\                <div style="font-size: 11px; color: #a5b4fc; margin: 4px 0 8px 0; font-family: var(--font-mono);">
        \\                  (RMSNorm ➔ MLP)
        \\                </div>
        \\                <div style="display: flex; flex-direction: column; gap: 8px;">
        \\                  ${ln2Child ? renderBranch(`${prefix}.ln_2`, ln2Child) : ''}
        \\                  <div class="tree-flow-connector" style="padding: 2px 8px;">
        \\                    <span class="flow-arrow-text"><span class="arrow-symbol">▼</span> Normalized Activation to MLP</span>
        \\                  </div>
        \\                  ${mlpChild ? renderBranch(`${prefix}.mlp`, mlpChild) : ''}
        \\                </div>
        \\              </div>
        \\            </div>
        \\
        \\            <!-- Downward Convergence: x + F(x) -->
        \\            <div class="tree-converge-box" onclick="openInspector('${outputNode ? outputNode.name : prefix + '.output'}')" style="cursor: pointer;">
        \\              <div class="flow-arrow-down" style="padding: 2px 0;">
        \\                <span class="flow-arrow-head">▼</span>
        \\                <span class="flow-arrow-label">⊕ Element-wise Add: x + F(x)</span>
        \\              </div>
        \\              <div class="tree-converge-card" style="width: 100%; justify-content: space-between;">
        \\                <div style="display: flex; align-items: center; gap: 8px;">
        \\                  <span>⊕</span>
        \\                  <span>Residual Add: x + F(x)</span>
        \\                  <span class="flow-card-shape">${mlpShape}</span>
        \\                </div>
        \\                <button class="tb-inspect-btn" onclick="event.stopPropagation(); openInspector('${outputNode ? outputNode.name : prefix + '.output'}')">🔍 Inspect</button>
        \\              </div>
        \\            </div>
        \\          </div>
        \\        `;
        \\      }
        \\    }
        \\
        \\    let renderedChildCount = 0;
        \\    for (const childKey of childKeys) {
        \\      if (isDecoderLayer && (childKey === 'ln_1' || childKey === 'attn' || childKey === 'ln_2' || childKey === 'mlp')) {
        \\        continue;
        \\      }
        \\      if (isAttentionLayer && (childKey === 'q_attn' || childKey === 'k_attn' || childKey === 'v_attn' || childKey === 'c_proj')) {
        \\        continue;
        \\      }
        \\      const fullChildName = isRoot ? childKey : `${prefix}.${childKey}`;
        \\      if (renderedChildCount > 0 && !isRoot) {
        \\        html += `
        \\          <div class="tree-flow-connector">
        \\            <div class="flow-line"></div>
        \\            <span class="flow-arrow-text"><span class="arrow-symbol">▼</span> Sequential Data Flow</span>
        \\          </div>
        \\        `;
        \\      }
        \\      html += renderBranch(fullChildName, node._children[childKey]);
        \\      renderedChildCount++;
        \\    }
        \\
        \\    html += `
        \\        </div>
        \\      </details>
        \\    `;
        \\  }
        \\  return html;
        \\}
        \\
        \\function renderTree() {
        \\  const filtered = NODES_DATA.filter(item => {
        \\    const matchesKind = (currentKindFilter === 'all' || item.kind === currentKindFilter);
        \\    const matchesQuery = (currentQuery === '' || 
        \\      item.name.toLowerCase().includes(currentQuery) ||
        \\      item.shape.toLowerCase().includes(currentQuery) ||
        \\      item.act.toLowerCase().includes(currentQuery) ||
        \\      item.strategy.toLowerCase().includes(currentQuery)
        \\    );
        \\    return matchesKind && matchesQuery;
        \\  });
        \\
        \\  const container = document.getElementById('tree-container');
        \\  if (filtered.length === 0) {
        \\    container.innerHTML = '<div style="text-align: center; padding: 48px; color: var(--text-sub);">No nodes matched the filter criteria.</div>';
        \\    return;
        \\  }
        \\
        \\  const hierarchy = buildHierarchy(filtered);
        \\  let outputHtml = '';
        \\
        \\  if (hierarchy._nodes.length > 0) {
        \\    outputHtml += renderBranch('(Root / Global Scope)', { _children: {}, _nodes: hierarchy._nodes, _paramCount: hierarchy._paramCount, _bytes: hierarchy._bytes });
        \\  }
        \\
        \\  const childKeys = Object.keys(hierarchy._children);
        \\  childKeys.forEach((modKey, idx) => {
        \\    if (idx > 0 || hierarchy._nodes.length > 0) {
        \\      outputHtml += `
        \\        <div class="tree-flow-connector" style="justify-content: center; margin: 4px 0 8px 0;">
        \\          <span class="flow-arrow-text">
        \\            <span class="arrow-symbol">▼</span>
        \\            <span>Default Sequential Data Flow: Output feeds as next module Input</span>
        \\          </span>
        \\        </div>
        \\      `;
        \\    }
        \\    outputHtml += renderBranch(modKey, hierarchy._children[modKey]);
        \\  });
        \\
        \\  container.innerHTML = outputHtml;
        \\}
        \\
        \\function setKindFilter(kind, btn) {
        \\  currentKindFilter = kind;
        \\  document.querySelectorAll('.btn-group .btn').forEach(b => b.classList.remove('active'));
        \\  btn.classList.add('active');
        \\  renderTree();
        \\}
        \\
        \\function filterNodes() {
        \\  currentQuery = document.getElementById('search-input').value.trim().toLowerCase();
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
        \\// Initial Render
        \\renderDagView();
        \\renderTree();
        \\renderMermaidView();
        \\</script>
        \\</body>
        \\</html>
    );

    return html_buf.toOwnedSlice(allocator);
}

/// 将计算图结构与各层初始化详情输出并保存为独立的 HTML 报告文件
pub fn exportHtmlReport(graph: *Graph, file_path: []const u8, allocator: std.mem.Allocator) !void {
    const html_content = try generateHtmlReport(graph, allocator);
    defer allocator.free(html_content);

    const path_z = try allocator.dupeZ(u8, file_path);
    defer allocator.free(path_z);

    const file = std.c.fopen(path_z.ptr, "wb") orelse return error.CannotOpenFile;
    defer _ = std.c.fclose(file);

    const written = std.c.fwrite(html_content.ptr, 1, html_content.len, file);
    if (written < html_content.len) return error.WriteFailed;
}
