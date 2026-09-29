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
            var effective_scope = scope;
            if (graph.getModuleType(scope)) |mtype| {
                if (std.mem.eql(u8, mtype, "CausalSelfAttention")) {
                    effective_scope = std.fmt.allocPrint(arena_alloc, "{s}.core", .{scope}) catch scope;
                }
            }
            for (op.outputs) |out| {
                if (!scopes.contains(out)) {
                    scopes.put(out, effective_scope) catch {};
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
        if (t.name == null) t.name = try graph.arena.allocator().dupe(u8, name);
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
        if (t.name == null) t.name = try graph.arena.allocator().dupe(u8, name);
        input_idx.* += 1;

        try nodes.append(allocator, .{
            .name = name,
            .kind = "Input",
            .shape_str = shape_str,
            .elements = elements,
            .bytes = bytes,
            .status = "INPUT",
            .inferred_act = try allocator.dupe(u8, "N/A"),
            .strategy = try allocator.dupe(u8, "user input / constant"),
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
    if (t.name == null) t.name = try graph.arena.allocator().dupe(u8, name);
    param_idx.* += 1;

    if (t.is_custom_initialized) {
        try nodes.append(allocator, .{
            .name = name,
            .kind = "Param",
            .shape_str = shape_str,
            .elements = elements,
            .bytes = bytes,
            .status = "CUSTOM_INIT",
            .inferred_act = try allocator.dupe(u8, "N/A"),
            .strategy = try allocator.dupe(u8, "user-defined customInit"),
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
            .inferred_act = try allocator.dupe(u8, "bias"),
            .strategy = try allocator.dupe(u8, "zeros (0.0)"),
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
        allocator.free(n.strategy);
        allocator.free(n.inferred_act);
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

/// 检查 target_t 是否为 start_t 的拓扑祖先（即存在前向数据通路 target_t ~> start_t）
fn isAncestor(start_t: *Tensor, target_t: *Tensor, visited: *std.AutoHashMap(*Tensor, void)) bool {
    if (start_t == target_t) return true;
    const op = start_t.creator orelse return false;
    for (op.inputs) |inp| {
        // 跳过参数矩阵与偏置，只沿数据流追溯
        if (inp.creator == null and inp.requires_grad) continue;
        if (inp == target_t) return true;
        if (!visited.contains(inp)) {
            visited.put(inp, {}) catch continue;
            if (isAncestor(inp, target_t, visited)) return true;
        }
    }
    return false;
}

/// 判断一个算子是否为跨分支汇聚算子（即非参数数据输入 >= 2，且输入来自不同的模块/分支）
fn isConvergenceOp(op: *autodiff.Op, scopes: *const std.AutoHashMap(*Tensor, []const u8)) bool {
    var data_in_count: usize = 0;
    var first_scope: ?[]const u8 = null;
    var has_different_scopes = false;

    for (op.inputs) |inp| {
        // 排除参数与常量/缓冲区
        if (inp.creator == null and inp.requires_grad) continue;
        if (inp.is_buffer) continue;

        data_in_count += 1;
        const s = scopes.get(inp) orelse (if (inp.name) |n| n else "");
        if (s.len > 0) {
            if (first_scope == null) {
                first_scope = s;
            } else if (!std.mem.eql(u8, first_scope.?, s)) {
                has_different_scopes = true;
            }
        }
    }

    return data_in_count >= 2 and has_different_scopes;
}

/// 从计算图张量直接获取其所属的模块作用域或宏观接口名 (完全基于图拓扑属性，无字符串硬编码启发式)
fn getTensorModule(
    graph: *const Graph,
    t: *Tensor,
    scopes: *const std.AutoHashMap(*Tensor, []const u8),
    consumed_set: *const std.AutoHashMap(*Tensor, void),
) []const u8 {
    // 1. 图原生源节点 (Graph Inputs): 入度为 0 且非参数张量
    if (t.creator == null and !t.requires_grad) {
        if (t.name) |n| return n;
    }

    // 2. 如果属于显式构造的子模块作用域（如注意力核心 ScaledDotProductAttention），优先归属该子模块
    if (scopes.get(t)) |scope| {
        if (graph.getModuleType(scope)) |mtype| {
            if (std.mem.eql(u8, mtype, "ScaledDotProductAttention")) {
                return scope;
            }
        }
    }

    // 3. 图原生汇聚节点 (Convergence Junctions): 跨分支汇聚算子（多数据输入来自不同分支）产生的显式命名张量
    if (t.creator) |op| {
        if (isConvergenceOp(op, scopes)) {
            if (t.name) |n| {
                if (std.mem.indexOf(u8, n, ".act_") == null) return n;
            }
        }
    }

    // 4. 图原生终端输出节点 (Graph Sink Outputs): 出度为 0 且显式命名的模型输出张量
    if (t.creator != null and !consumed_set.contains(t)) {
        if (t.name) |n| return n;
    }

    // 5. 属于具体子模块的内部张量/激活值
    if (scopes.get(t)) |scope| {
        return scope;
    }

    // 6. 模型权重参数 (creator == null 且 requires_grad)
    if (t.creator == null and t.requires_grad and t.name != null) {
        if (extractModuleScope(t.name.?)) |s| return s;
    }

    // 7. 回退到张量原生名称
    if (t.name) |n| return n;

    return "unknown";
}

pub fn collectGraphEdges(graph: *Graph, allocator: std.mem.Allocator) !std.ArrayList(EdgeData) {
    var edges: std.ArrayList(EdgeData) = .empty;
    errdefer freeGraphEdges(&edges, allocator);

    const arena_alloc = graph.arena.allocator();
    var edge_set = std.StringHashMap(void).init(arena_alloc);
    defer edge_set.deinit();

    var scopes = std.AutoHashMap(*Tensor, []const u8).init(arena_alloc);
    defer scopes.deinit();

    for (graph.tensors.items) |t| {
        if (t.name) |n| {
            if (extractModuleScope(n)) |s| scopes.put(t, s) catch {};
        }
    }
    for (graph.ops.items) |op| {
        for (op.inputs) |t| {
            if (t.name) |n| {
                if (extractModuleScope(n)) |s| scopes.put(t, s) catch {};
            }
        }
        for (op.outputs) |t| {
            if (t.name) |n| {
                if (extractModuleScope(n)) |s| scopes.put(t, s) catch {};
            }
        }
    }
    for (graph.ops.items) |op| {
        var op_scope: ?[]const u8 = null;
        for (op.inputs) |inp| {
            if (inp.creator == null and inp.name != null) {
                if (extractModuleScope(inp.name.?)) |p_scope| {
                    if (op_scope == null or p_scope.len > op_scope.?.len) op_scope = p_scope;
                }
            }
        }
        if (op_scope == null) {
            for (op.inputs) |inp| {
                if (scopes.get(inp)) |inp_scope| {
                    if (op_scope == null) op_scope = inp_scope else {
                        const common = getCommonModulePrefix(op_scope.?, inp_scope);
                        if (common.len > 0) op_scope = common;
                    }
                }
            }
        }
        if (op_scope) |scope| {
            var effective_scope = scope;
            if (graph.getModuleType(scope)) |mtype| {
                if (std.mem.eql(u8, mtype, "CausalSelfAttention")) {
                    effective_scope = std.fmt.allocPrint(arena_alloc, "{s}.core", .{scope}) catch scope;
                }
            }
            for (op.outputs) |out| {
                if (!scopes.contains(out)) scopes.put(out, effective_scope) catch {};
            }
        }
    }

    // 预统计计算图中所有被下游算子消费的张量集合（用于准确判断图终端 Sink 输出节点）
    var consumed_set = std.AutoHashMap(*Tensor, void).init(arena_alloc);
    defer consumed_set.deinit();
    for (graph.ops.items) |op| {
        for (op.inputs) |inp| {
            consumed_set.put(inp, {}) catch {};
        }
    }

    const addEdge = struct {
        fn add(
            list: *std.ArrayList(EdgeData),
            set: *std.StringHashMap(void),
            arena: std.mem.Allocator,
            from: []const u8,
            to: []const u8,
            shape: []const u8,
            is_skip: bool,
        ) !void {
            if (from.len == 0 or to.len == 0 or std.mem.eql(u8, from, to)) return;
            var key_buf: [384]u8 = undefined;
            const key = std.fmt.bufPrint(&key_buf, "{s}->{s}", .{ from, to }) catch return;
            if (set.contains(key)) return;
            try set.put(try arena.dupe(u8, key), {});
            try list.append(arena, .{
                .from = try arena.dupe(u8, from),
                .to = try arena.dupe(u8, to),
                .shape = try arena.dupe(u8, shape),
                .is_skip = is_skip,
            });
        }
    }.add;

    for (graph.ops.items) |op| {
        if (op.outputs.len == 0) continue;
        const out = op.outputs[0];

        for (op.inputs) |inp| {
            // 1. 图原生叶子判断：严格跳过内部权重/偏置参数与静态缓冲张量
            if (inp.creator == null and inp.requires_grad) continue;
            if (inp.is_buffer) continue;

            // 2. 格式化张量形状
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
            const shape_str = shape_buf[0..shape_len];

            // 3. 直接通过张量在计算图中的拓扑属性与模块作用域获取所属模块
            const from_mod = getTensorModule(graph, inp, &scopes, &consumed_set);
            const to_mod = getTensorModule(graph, out, &scopes, &consumed_set);

            if (!std.mem.eql(u8, from_mod, to_mod)) {
                // 图原生跳跃连接判定：如果汇聚算子的某个输入在拓扑上是另一个输入的祖先，则该输入为跳跃旁路
                var is_skip = false;
                if (out.creator) |c_op| {
                    if (isConvergenceOp(c_op, &scopes)) {
                        for (c_op.inputs) |other| {
                            if (other == inp) continue;
                            if (other.creator == null and other.requires_grad) continue;
                            var visited = std.AutoHashMap(*Tensor, void).init(arena_alloc);
                            defer visited.deinit();
                            if (isAncestor(other, inp, &visited)) {
                                is_skip = true;
                                break;
                            }
                        }
                    }
                }
                try addEdge(&edges, &edge_set, arena_alloc, from_mod, to_mod, shape_str, is_skip);
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
    formula: ?[]const u8 = null,
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
                    // 如果公共前缀是 .attn 且输出没有明确指定具体名字，标为 .attn (整个注意力模块的内部激活)
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


/// 递归模型层级模块组节点 (Recursive Module Node)
pub const ModuleNode = struct {
    name: []const u8,
    path: []const u8,
    kind: []const u8 = "module",
    module_type: []const u8 = "Module",
    formula: ?[]const u8 = null,
    total_params: usize = 0,
    total_bytes: usize = 0,
    param_count: usize = 0,
    node_count: usize = 0,
    children: std.ArrayList(*ModuleNode),
    nodes: std.ArrayList(NodeData),
    parameters: std.ArrayList(NodeData),
    ops: std.ArrayList(OpData),
    edges: std.ArrayList(EdgeData),

    pub fn init(allocator: std.mem.Allocator, name: []const u8, path: []const u8) !*ModuleNode {
        const node = try allocator.create(ModuleNode);
        node.* = .{
            .name = name,
            .path = path,
            .kind = "module",
            .module_type = "Module",
            .formula = null,
            .total_params = 0,
            .total_bytes = 0,
            .param_count = 0,
            .node_count = 0,
            .children = .empty,
            .nodes = .empty,
            .parameters = .empty,
            .ops = .empty,
            .edges = .empty,
        };
        return node;
    }
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
    nodes: std.ArrayList(NodeData),
    edges: std.ArrayList(EdgeData),
    ops: std.ArrayList(OpData),
    formulas: std.StringHashMap([]const u8),

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

    /// 判断模块类型是否为复合子层 (其内部有独立的微观子层/算子，如注意力 QKV/Core，前馈网络门控/投影等)
    fn isSublayerModuleType(mtype: []const u8) bool {
        return std.mem.eql(u8, mtype, "CausalSelfAttention") or
            std.mem.eql(u8, mtype, "MLP") or
            std.mem.eql(u8, mtype, "SwiGLU") or
            std.mem.eql(u8, mtype, "MoELayer") or
            std.mem.eql(u8, mtype, "MLALayer");
    }

    /// 将包含内部微观零件的路径截断/收拢至最外层复合子层作用域 (例如 gpt.layers.0.attn.q_attn -> gpt.layers.0.attn)
    fn getMacroModuleScope(graph_inst: *const Graph, path: []const u8) []const u8 {
        var it = std.mem.splitScalar(u8, path, '.');
        var current_len: usize = 0;
        while (it.next()) |part| {
            if (current_len > 0) current_len += 1; // '.'
            current_len += part.len;
            const prefix = path[0..current_len];
            if (graph_inst.getModuleType(prefix)) |mtype| {
                if (isSublayerModuleType(mtype)) {
                    return prefix;
                }
            }
        }
        return path;
    }

    /// 解析 Graph 并构建约定的中间数据结构 ModelHierarchyGraph
    pub fn build(graph: *Graph, allocator: std.mem.Allocator) !ModelHierarchyGraph {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const arena_alloc = arena.allocator();

        const nodes = try collectGraphNodes(graph, arena_alloc);
        const edges = try collectGraphEdges(graph, arena_alloc);
        const ops_list = try collectGraphOps(graph, arena_alloc);

        var formulas = std.StringHashMap([]const u8).init(arena_alloc);
        var f_it = graph.module_formulas.iterator();
        while (f_it.next()) |entry| {
            try formulas.put(try arena_alloc.dupe(u8, entry.key_ptr.*), try arena_alloc.dupe(u8, entry.value_ptr.*));
        }

        // 0. 注入所有标准基础算子类型 (OpType) 的标准 LaTeX 公式字典
        inline for (std.meta.fields(@import("../autodiff/types.zig").OpType)) |field| {
            const op_enum: @import("../autodiff/types.zig").OpType = @enumFromInt(field.value);
            const op_name = field.name;
            const op_form = op_enum.getFormula();
            try formulas.put(try arena_alloc.dupe(u8, op_name), try arena_alloc.dupe(u8, op_form));
        }

        // 1. 自动遍历所有算子实例，将其对应的具体算子公式直接绑定到 op.name
        for (ops_list.items) |*op| {
            if (!formulas.contains(op.name)) {
                // 如果存在对应的算子类型标准公式，直接使用算子公式
                if (formulas.get(op.op_type)) |op_form| {
                    try formulas.put(try arena_alloc.dupe(u8, op.name), try arena_alloc.dupe(u8, op_form));
                } else {
                    const inferred = graph.inferModuleFormula(op.name);
                    try formulas.put(try arena_alloc.dupe(u8, op.name), try arena_alloc.dupe(u8, inferred));
                }
            }
            if (!formulas.contains(op.module)) {
                const inferred = graph.inferModuleFormula(op.module);
                try formulas.put(try arena_alloc.dupe(u8, op.module), try arena_alloc.dupe(u8, inferred));
            }
            op.formula = formulas.get(op.name) orelse (formulas.get(op.op_type) orelse null);
        }

        // 2. 自动遍历所有图节点，为尚未显式注册公式的模块和算子从图与内置库中推导并注入公式
        for (nodes.items) |node| {
            if (!formulas.contains(node.name)) {
                if (std.mem.eql(u8, node.kind, "Activation")) {
                    if (formulas.get(node.inferred_act)) |act_form| {
                        try formulas.put(try arena_alloc.dupe(u8, node.name), try arena_alloc.dupe(u8, act_form));
                        continue;
                    }
                }
                const inferred = graph.inferModuleFormula(node.name);
                try formulas.put(try arena_alloc.dupe(u8, node.name), try arena_alloc.dupe(u8, inferred));
            }
            if (extractModuleScope(node.name)) |scope| {
                if (!formulas.contains(scope)) {
                    const inferred = graph.inferModuleFormula(scope);
                    try formulas.put(try arena_alloc.dupe(u8, scope), try arena_alloc.dupe(u8, inferred));
                }
            }
        }

        // 构建递归模块树
        const root = try ModuleNode.init(arena_alloc, "root", "");
        root.module_type = "Model";
        root.formula = graph.getModuleFormula("root") orelse (graph.getModuleFormula("") orelse null);

        var module_map = std.StringHashMap(*ModuleNode).init(arena_alloc);
        try module_map.put("", root);
        try module_map.put("root", root);

        for (nodes.items) |*node| {
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
                        new_child.module_type = graph.getModuleType(child_path) orelse "Module";
                        new_child.formula = formulas.get(child_path) orelse graph.inferModuleFormula(child_path);

                        try curr.children.append(arena_alloc, new_child);
                        try module_map.put(child_path, new_child);
                        curr = new_child;
                    }
                }
                try curr.nodes.append(arena_alloc, node.*);
                if (std.mem.eql(u8, node.kind, "Param")) {
                    try curr.parameters.append(arena_alloc, node.*);
                }
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
                    new_u.module_type = "Unscoped";
                    try root.children.append(arena_alloc, new_u);
                    try module_map.put("(unscoped)", new_u);
                    break :blk new_u;
                };
                try unscoped.nodes.append(arena_alloc, node.*);
                if (std.mem.eql(u8, node.kind, "Param")) {
                    try unscoped.parameters.append(arena_alloc, node.*);
                }
            }
        }

        // 将算子分类关联到对应模块内部
        for (ops_list.items) |op| {
            if (module_map.get(op.module)) |m| {
                try m.ops.append(arena_alloc, op);
            }
        }

        // 将边分类关联到对应模块内部 (依据直接子模块与进出接口精准归属)
        for (edges.items) |e| {
            var it = module_map.iterator();
            while (it.next()) |entry| {
                const mod_path = entry.key_ptr.*;
                if (mod_path.len == 0 or std.mem.eql(u8, mod_path, "root")) continue;
                const m = entry.value_ptr.*;

                const from_in = std.mem.eql(u8, e.from, mod_path) or
                    (std.mem.startsWith(u8, e.from, mod_path) and e.from.len > mod_path.len and e.from[mod_path.len] == '.');
                const to_in = std.mem.eql(u8, e.to, mod_path) or
                    (std.mem.startsWith(u8, e.to, mod_path) and e.to.len > mod_path.len and e.to[mod_path.len] == '.');

                if (!from_in and !to_in) continue;

                // 1. 如果起点和终点都在该模块内部：
                if (from_in and to_in) {
                    // 检查是否完全属于某个更深层的子模块（例如 q_attn -> c_proj 属于 attn，不属于 layers.0）
                    var has_deeper_child = false;
                    for (m.children.items) |child| {
                        const from_child = std.mem.eql(u8, e.from, child.path) or
                            (std.mem.startsWith(u8, e.from, child.path) and e.from.len > child.path.len and e.from[child.path.len] == '.');
                        const to_child = std.mem.eql(u8, e.to, child.path) or
                            (std.mem.startsWith(u8, e.to, child.path) and e.to.len > child.path.len and e.to[child.path.len] == '.');
                        if (from_child and to_child) {
                            has_deeper_child = true;
                            break;
                        }
                    }
                    if (has_deeper_child) continue;
                }

                // 2. 如果目标模块有子模块，但当前边的一端指向自身（例如 e.to == mod_path 且由某个内部子模块发出），
                // 那么这并不是发给父模块的边，而是子模块间的内部连接，应该归入能够承载两者的最小子容器中
                if (m.children.items.len > 0 and std.mem.eql(u8, e.to, mod_path) and from_in) {
                    continue;
                }


                // 2. 提取相对于当前模块的子节点或直接子模块名
                var from_name = e.from;
                if (from_in and e.from.len > mod_path.len + 1) {
                    const rel = e.from[mod_path.len + 1 ..];
                    if (std.mem.indexOfScalar(u8, rel, '.')) |dot| {
                        from_name = rel[0..dot];
                    } else {
                        from_name = rel;
                    }
                }

                var to_name = e.to;
                if (to_in and e.to.len > mod_path.len + 1) {
                    const rel = e.to[mod_path.len + 1 ..];
                    if (std.mem.indexOfScalar(u8, rel, '.')) |dot| {
                        to_name = rel[0..dot];
                    } else {
                        to_name = rel;
                    }
                }

                // 如果两端折叠后成为同一个名字（例如内部局部微观流动），跳过
                if (std.mem.eql(u8, from_name, to_name)) continue;

                // 检查是否已经添加过相同的一对 from -> to
                var already_has = false;
                for (m.edges.items) |existing| {
                    if (std.mem.eql(u8, existing.from, from_name) and std.mem.eql(u8, existing.to, to_name)) {
                        already_has = true;
                        break;
                    }
                }
                if (already_has) continue;

                try m.edges.append(arena_alloc, .{
                    .from = try arena_alloc.dupe(u8, from_name),
                    .to = try arena_alloc.dupe(u8, to_name),
                    .shape = try arena_alloc.dupe(u8, e.shape),
                    .is_skip = e.is_skip,
                });
            }
        }

        aggregateMetrics(root);

        var summary = Summary{
            .total_params = root.total_params,
            .total_bytes = root.total_bytes,
            .total_nodes = nodes.items.len,
        };

        for (nodes.items) |n| {
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

        // 6. 构造顶层宏观拓扑边 (Macro Edges):
        // 消除穿透子模块内部器官的微观边 (如 ln_1 -> q_attn, q_attn -> core)，
        // 将连接端点收拢至复合子层边界 (如 ln_1 -> attn, attn -> residual_attn)，
        // 仅保留跨模块的宏观流转，内部零件流动保留在各模块自身的 m.edges 中
        var macro_edges: std.ArrayList(EdgeData) = .empty;
        var macro_seen = std.StringHashMap(void).init(arena_alloc);

        for (edges.items) |e| {
            const macro_from = getMacroModuleScope(graph, e.from);
            const macro_to = getMacroModuleScope(graph, e.to);

            if (std.mem.eql(u8, macro_from, macro_to)) continue;

            var key_buf: [384]u8 = undefined;
            const key = std.fmt.bufPrint(&key_buf, "{s}->{s}", .{ macro_from, macro_to }) catch continue;
            if (macro_seen.contains(key)) continue;
            try macro_seen.put(try arena_alloc.dupe(u8, key), {});

            try macro_edges.append(arena_alloc, .{
                .from = try arena_alloc.dupe(u8, macro_from),
                .to = try arena_alloc.dupe(u8, macro_to),
                .shape = try arena_alloc.dupe(u8, e.shape),
                .is_skip = e.is_skip,
            });
        }

        return .{
            .arena = arena,
            .summary = summary,
            .root = root,
            .nodes = nodes,
            .edges = macro_edges,
            .ops = ops_list,
            .formulas = formulas,
        };
    }

    fn writeEscapedJsonString(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, str: []const u8) !void {
        try buf.append(allocator, '"');
        for (str) |c| {
            switch (c) {
                '"' => try buf.appendSlice(allocator, "\\\""),
                '\\' => try buf.appendSlice(allocator, "\\\\"),
                '\n' => try buf.appendSlice(allocator, "\\n"),
                '\r' => try buf.appendSlice(allocator, "\\r"),
                '\t' => try buf.appendSlice(allocator, "\\t"),
                0x00...0x08, 0x0B, 0x0C, 0x0E...0x1F => try buf.print(allocator, "\\u00{x:0>2}", .{c}),
                else => try buf.append(allocator, c),
            }
        }
        try buf.append(allocator, '"');
    }

    fn serializeModuleTree(node: *const ModuleNode, buf: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        try buf.appendSlice(allocator, "{");
        try buf.appendSlice(allocator, "\"name\": ");
        try writeEscapedJsonString(buf, allocator, node.name);
        try buf.appendSlice(allocator, ",\"path\": ");
        try writeEscapedJsonString(buf, allocator, node.path);
        try buf.appendSlice(allocator, ",\"kind\": \"module\"");
        try buf.appendSlice(allocator, ",\"module_type\": ");
        try writeEscapedJsonString(buf, allocator, node.module_type);

        if (node.formula) |f| {
            try buf.appendSlice(allocator, ",\"formula\": ");
            try writeEscapedJsonString(buf, allocator, f);
        } else {
            try buf.appendSlice(allocator, ",\"formula\": null");
        }

        try buf.print(allocator, ",\"total_params\": {d},", .{node.total_params});
        try buf.print(allocator, "\"total_bytes\": {d},", .{node.total_bytes});
        try buf.print(allocator, "\"param_count\": {d},", .{node.param_count});
        try buf.print(allocator, "\"node_count\": {d},", .{node.node_count});

        try buf.appendSlice(allocator, "\"children\": [");
        for (node.children.items, 0..) |child, i| {
            if (i > 0) try buf.appendSlice(allocator, ",");
            try serializeModuleTree(child, buf, allocator);
        }
        try buf.appendSlice(allocator, "],\"parameters\": [");
        for (node.parameters.items, 0..) |p, i| {
            if (i > 0) try buf.appendSlice(allocator, ",");
            try buf.appendSlice(allocator, "{\"name\": ");
            try writeEscapedJsonString(buf, allocator, p.name);
            try buf.appendSlice(allocator, ",\"shape\": ");
            try writeEscapedJsonString(buf, allocator, p.shape_str);
            try buf.print(allocator, ",\"elements\": {d},\"bytes\": {d},", .{ p.elements, p.bytes });
            try buf.appendSlice(allocator, "\"status\": ");
            try writeEscapedJsonString(buf, allocator, p.status);
            try buf.appendSlice(allocator, ",\"strategy\": ");
            try writeEscapedJsonString(buf, allocator, p.strategy);
            try buf.appendSlice(allocator, "}");
        }
        try buf.appendSlice(allocator, "],\"ops\": [");
        for (node.ops.items, 0..) |op, i| {
            if (i > 0) try buf.appendSlice(allocator, ",");
            try buf.appendSlice(allocator, "{\"op_type\": ");
            try writeEscapedJsonString(buf, allocator, op.op_type);
            try buf.appendSlice(allocator, ",\"name\": ");
            try writeEscapedJsonString(buf, allocator, op.name);
            try buf.appendSlice(allocator, ",\"input_shape\": ");
            try writeEscapedJsonString(buf, allocator, op.input_shape);
            try buf.appendSlice(allocator, ",\"param_shape\": ");
            try writeEscapedJsonString(buf, allocator, op.param_shape);
            try buf.appendSlice(allocator, ",\"output_shape\": ");
            try writeEscapedJsonString(buf, allocator, op.output_shape);
            if (op.formula) |f| {
                try buf.appendSlice(allocator, ",\"formula\": ");
                try writeEscapedJsonString(buf, allocator, f);
            }
            try buf.print(allocator, ",\"elements\": {d},\"bytes\": {d}}}", .{ op.elements, op.bytes });
        }
        try buf.appendSlice(allocator, "],\"edges\": [");
        for (node.edges.items, 0..) |e, i| {
            if (i > 0) try buf.appendSlice(allocator, ",");
            try buf.appendSlice(allocator, "{\"from\": ");
            try writeEscapedJsonString(buf, allocator, e.from);
            try buf.appendSlice(allocator, ",\"to\": ");
            try writeEscapedJsonString(buf, allocator, e.to);
            try buf.appendSlice(allocator, ",\"shape\": ");
            try writeEscapedJsonString(buf, allocator, e.shape);
            try buf.print(allocator, ",\"is_skip\": {s}}}", .{ if (e.is_skip) "true" else "false" });
        }
        try buf.appendSlice(allocator, "],\"nodes\": [");
        for (node.nodes.items, 0..) |leaf, i| {
            if (i > 0) try buf.appendSlice(allocator, ",");
            try buf.appendSlice(allocator, "{\"name\": ");
            try writeEscapedJsonString(buf, allocator, leaf.name);
            try buf.appendSlice(allocator, ",\"kind\": ");
            try writeEscapedJsonString(buf, allocator, leaf.kind);
            try buf.appendSlice(allocator, ",\"shape\": ");
            try writeEscapedJsonString(buf, allocator, leaf.shape_str);
            try buf.print(allocator, ",\"elements\": {d},\"bytes\": {d},", .{ leaf.elements, leaf.bytes });
            try buf.appendSlice(allocator, "\"status\": ");
            try writeEscapedJsonString(buf, allocator, leaf.status);
            try buf.appendSlice(allocator, ",\"act\": ");
            try writeEscapedJsonString(buf, allocator, leaf.inferred_act);
            try buf.appendSlice(allocator, ",\"strategy\": ");
            try writeEscapedJsonString(buf, allocator, leaf.strategy);
            try buf.appendSlice(allocator, "}");
        }
        try buf.appendSlice(allocator, "]}");
    }

    /// 将 ModelHierarchyGraph 中间结构序列化为约定的完整递归 JSON 数据字符串
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
        try serializeModuleTree(model_graph.root, &json_buf, allocator);

        try json_buf.appendSlice(allocator, ",\n  \"edges\": [");
        for (model_graph.edges.items, 0..) |e, i| {
            if (i > 0) try json_buf.appendSlice(allocator, ",\n    ");
            if (i == 0) try json_buf.appendSlice(allocator, "\n    ");
            try json_buf.appendSlice(allocator, "{\"from\": ");
            try writeEscapedJsonString(&json_buf, allocator, e.from);
            try json_buf.appendSlice(allocator, ", \"to\": ");
            try writeEscapedJsonString(&json_buf, allocator, e.to);
            try json_buf.appendSlice(allocator, ", \"shape\": ");
            try writeEscapedJsonString(&json_buf, allocator, e.shape);
            try json_buf.print(allocator, ", \"is_skip\": {s}}}", .{if (e.is_skip) "true" else "false"});
        }
        try json_buf.appendSlice(allocator, "\n  ]");

        try json_buf.appendSlice(allocator, "\n}");

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
// 3. 便捷顶层重导出 (Top-Level Re-Exports)
// ============================================================================
pub const buildModelHierarchy = graph_ir.build;
pub const generateJson = graph_ir.generateJson;
pub const exportJson = graph_ir.exportJson;

