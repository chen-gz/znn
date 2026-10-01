const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const init_mod = @import("init.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Graph = autodiff.Graph;
const Op = autodiff.Op;

/// 导出 JSON 的 schema 版本
pub const SCHEMA_VERSION = "2.0";

/// 导出 JSON 的 JSON Schema (draft 2020-12)，逐字段描述 serializeJson 的输出
pub const SCHEMA_JSON = @embedFile("model_graph.schema.json");

/// 张量节点分类枚举 (Tensor Node Kind)
pub const NodeKind = enum {
    Param,
    Input,
    Buffer,
    Activation,

    pub fn asString(self: NodeKind) []const u8 {
        return @tagName(self);
    }

    pub fn fromString(str: []const u8) ?NodeKind {
        if (std.meta.stringToEnum(NodeKind, str)) |k| return k;
        if (std.ascii.eqlIgnoreCase(str, "param")) return .Param;
        if (std.ascii.eqlIgnoreCase(str, "input")) return .Input;
        if (std.ascii.eqlIgnoreCase(str, "buffer")) return .Buffer;
        if (std.ascii.eqlIgnoreCase(str, "activation")) return .Activation;
        return null;
    }
};

/// 张量来源 / 初始化状态枚举 (Tensor Node Status)
pub const NodeStatus = enum {
    /// 参数：由模块代码显式初始化 (customInit)
    CUSTOM_INIT,
    /// 参数：依据消费端激活函数自动选择初始化策略
    AUTO_GRAPH,
    /// 图输入
    INPUT,
    /// 常量缓冲区
    BUFFER,
    /// 算子输出 (激活)
    OP_OUTPUT,

    pub fn asString(self: NodeStatus) []const u8 {
        return @tagName(self);
    }
};

/// 单个计算图节点的详细可视化元数据
pub const NodeData = struct {
    pub const Kind = NodeKind;
    pub const Status = NodeStatus;

    name: []const u8,
    kind: NodeKind,
    module: []const u8, // 节点所属模块的完整路径 ("" 表示根)
    shape_str: []const u8,
    elements: usize,
    bytes: usize,
    status: NodeStatus,
    inferred_act: []const u8,
    strategy: []const u8,
};

// ============================================================================
// 1. 模块作用域解析 (Module Scope Resolution)
//    归属完全由 Graph 作用域栈在算子/张量创建时记录的 scope 决定，不从输入推断。
// ============================================================================

/// 从节点名称中提取父模块路径 (以 '.' 分隔)
fn extractModuleScope(name: []const u8) ?[]const u8 {
    if (std.mem.lastIndexOfScalar(u8, name, '.')) |dot_idx| {
        if (dot_idx > 0) return name[0..dot_idx];
    }
    return null;
}

/// 判断 path 是否位于 scope 之内 (包含 scope 自身)；空 scope 表示根作用域，包含一切路径
fn isWithinScope(path: []const u8, scope: []const u8) bool {
    if (scope.len == 0) return true;
    if (std.mem.eql(u8, path, scope)) return true;
    return path.len > scope.len and std.mem.startsWith(u8, path, scope) and path[scope.len] == '.';
}

/// 两个作用域路径在点号边界上的最近公共祖先
fn commonScope(a: []const u8, b: []const u8) []const u8 {
    if (isWithinScope(a, b)) return b;
    if (isWithinScope(b, a)) return a;
    var last_dot: ?usize = null;
    var i: usize = 0;
    while (i < a.len and i < b.len and a[i] == b[i]) : (i += 1) {
        if (a[i] == '.') last_dot = i;
    }
    return if (last_dot) |d| a[0..d] else "";
}

/// scope 相对于 parent 的直接子模块名 (要求 scope 严格位于 parent 之内)
fn childSegment(scope: []const u8, parent: []const u8) []const u8 {
    const rel = if (parent.len == 0) scope else scope[parent.len + 1 ..];
    if (std.mem.indexOfScalar(u8, rel, '.')) |d| return rel[0..d];
    return rel;
}

fn joinScope(allocator: std.mem.Allocator, parent: []const u8, seg: []const u8) ![]const u8 {
    if (parent.len == 0) return allocator.dupe(u8, seg);
    return std.fmt.allocPrint(allocator, "{s}.{s}", .{ parent, seg });
}

fn isParam(t: *const Tensor) bool {
    return t.creator == null and t.requires_grad;
}

/// 算子所属模块：以创建时记录的作用域为准；若算子的参数属于该作用域更深层的子模块
/// (子模块 forward 未接入作用域栈)，则以参数所属模块为准。
pub fn opScope(op: *const Op) []const u8 {
    var s = op.scope;
    for (op.inputs) |inp| {
        if (!isParam(inp)) continue;
        const n = inp.name orelse continue;
        const ps = extractModuleScope(n) orelse continue;
        if (ps.len > s.len and isWithinScope(ps, s)) s = ps;
    }
    return s;
}

/// 张量所属模块：算子输出取其算子作用域；参数取其名称前缀；输入与常量取创建时的作用域
pub fn tensorScope(t: *const Tensor) []const u8 {
    if (t.creator) |op| return opScope(op);
    if (t.requires_grad) {
        if (t.name) |n| {
            if (extractModuleScope(n)) |s| return s;
        }
    }
    return t.scope;
}

/// 透明算子：只改变张量布局、不做数值计算，在局部图中折叠进边而不作为节点
fn isTransparentOp(op: *const Op) bool {
    return switch (op.op_type) {
        .Reshape, .Transpose, .RepeatKV => true,
        else => false,
    };
}

fn firstDataInput(op: *const Op) ?*Tensor {
    for (op.inputs) |inp| {
        if (!isParam(inp)) return inp;
    }
    return null;
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

fn shapeEql(a: Shape, b: Shape) bool {
    if (a.len != b.len) return false;
    for (0..a.len) |i| {
        if (a.dims[i] != b.dims[i]) return false;
    }
    return true;
}

// ============================================================================
// 2. 节点、算子元数据采集 (Node & Op Metadata)
// ============================================================================

/// 收集计算图中所有节点的详细元数据 (未命名张量按所属作用域自动命名)
pub fn collectGraphNodes(graph: *Graph, allocator: std.mem.Allocator) !std.ArrayList(NodeData) {
    var nodes: std.ArrayList(NodeData) = .empty;
    errdefer freeGraphNodes(&nodes, allocator);

    const arena_alloc = graph.arena.allocator();
    var visited = std.AutoHashMap(*Tensor, void).init(arena_alloc);
    defer visited.deinit();

    var counters = NodeCounters{};

    for (graph.ops.items) |op| {
        for (op.inputs) |t| {
            if (visited.contains(t)) continue;
            try visited.put(t, {});
            try appendNodeData(graph, t, &counters, &nodes, allocator);
        }
        for (op.outputs) |t| {
            if (visited.contains(t)) continue;
            try visited.put(t, {});
            try appendNodeData(graph, t, &counters, &nodes, allocator);
        }
    }

    for (graph.tensors.items) |t| {
        if (visited.contains(t)) continue;
        try visited.put(t, {});
        try appendNodeData(graph, t, &counters, &nodes, allocator);
    }

    return nodes;
}

const NodeCounters = struct {
    param: usize = 0,
    input: usize = 0,
    op: usize = 0,
};

fn appendNodeData(
    graph: *Graph,
    t: *Tensor,
    counters: *NodeCounters,
    nodes: *std.ArrayList(NodeData),
    allocator: std.mem.Allocator,
) !void {
    const shape_str = try formatShapeAlloc(allocator, t.shape);
    const elements = t.data.len;
    const bytes = elements * @sizeOf(f32);
    const scope = tensorScope(t);
    const module = try allocator.dupe(u8, scope);

    // 1. 算子输出激活值
    if (t.creator) |creator_op| {
        const op_name = @tagName(creator_op.op_type);
        const name = if (t.name) |n|
            try allocator.dupe(u8, n)
        else if (scope.len > 0)
            try std.fmt.allocPrint(allocator, "{s}.act_{s}_{d}", .{ scope, op_name, counters.op })
        else
            try std.fmt.allocPrint(allocator, "graph.act_{s}_{d}", .{ op_name, counters.op });
        if (t.name == null) t.name = try graph.arena.allocator().dupe(u8, name);
        counters.op += 1;

        var strat_buf: [64]u8 = undefined;
        const strat = try allocator.dupe(u8, std.fmt.bufPrint(&strat_buf, "produced by {s}", .{op_name}) catch "op output");

        try nodes.append(allocator, .{
            .name = name,
            .kind = .Activation,
            .module = module,
            .shape_str = shape_str,
            .elements = elements,
            .bytes = bytes,
            .status = .OP_OUTPUT,
            .inferred_act = try allocator.dupe(u8, op_name),
            .strategy = strat,
        });
        return;
    }

    // 2. 图输入与常量缓冲区
    if (!t.requires_grad) {
        const prefix = if (t.is_buffer) "buffer" else "input";
        const name = if (t.name) |n|
            try allocator.dupe(u8, n)
        else if (scope.len > 0)
            try std.fmt.allocPrint(allocator, "{s}.{s}_{d}", .{ scope, prefix, counters.input })
        else
            try std.fmt.allocPrint(allocator, "inputs.{s}_{d}", .{ prefix, counters.input });
        if (t.name == null) t.name = try graph.arena.allocator().dupe(u8, name);
        counters.input += 1;

        try nodes.append(allocator, .{
            .name = name,
            .kind = if (t.is_buffer) .Buffer else .Input,
            .module = module,
            .shape_str = shape_str,
            .elements = elements,
            .bytes = bytes,
            .status = if (t.is_buffer) .BUFFER else .INPUT,
            .inferred_act = try allocator.dupe(u8, "N/A"),
            .strategy = try allocator.dupe(u8, if (t.is_buffer) "constant buffer" else "user input"),
        });
        return;
    }

    // 3. 模型可学习参数
    const name = if (t.name) |n|
        try allocator.dupe(u8, n)
    else if (scope.len > 0)
        try std.fmt.allocPrint(allocator, "{s}.param_{d}", .{ scope, counters.param })
    else
        try std.fmt.allocPrint(allocator, "parameters.param_{d}", .{counters.param});
    if (t.name == null) t.name = try graph.arena.allocator().dupe(u8, name);
    counters.param += 1;

    if (t.is_custom_initialized) {
        try nodes.append(allocator, .{
            .name = name,
            .kind = .Param,
            .module = module,
            .shape_str = shape_str,
            .elements = elements,
            .bytes = bytes,
            .status = .CUSTOM_INIT,
            .inferred_act = try allocator.dupe(u8, "N/A"),
            .strategy = try allocator.dupe(u8, "user-defined customInit"),
        });
        return;
    }

    if (t.shape.len == 1 or (t.shape.len == 2 and t.shape.dims[0] == 1)) {
        try nodes.append(allocator, .{
            .name = name,
            .kind = .Param,
            .module = module,
            .shape_str = shape_str,
            .elements = elements,
            .bytes = bytes,
            .status = .AUTO_GRAPH,
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
        .kind = .Param,
        .module = module,
        .shape_str = shape_str,
        .elements = elements,
        .bytes = bytes,
        .status = .AUTO_GRAPH,
        .inferred_act = try allocator.dupe(u8, act_name),
        .strategy = strat,
    });
}

/// 释放节点元数据内存
pub fn freeGraphNodes(nodes: *std.ArrayList(NodeData), allocator: std.mem.Allocator) void {
    for (nodes.items) |n| {
        allocator.free(n.name);
        allocator.free(n.module);
        allocator.free(n.shape_str);
        allocator.free(n.strategy);
        allocator.free(n.inferred_act);
    }
    nodes.deinit(allocator);
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

    var op_counter: usize = 0;
    for (graph.ops.items) |op| {
        if (op.outputs.len == 0) continue;
        const out = op.outputs[0];
        const op_tag = @tagName(op.op_type);
        const mod_scope = opScope(op);

        const op_name = if (out.name) |n|
            try allocator.dupe(u8, n)
        else if (mod_scope.len > 0)
            try std.fmt.allocPrint(allocator, "{s}.act_{s}_{d}", .{ mod_scope, op_tag, op_counter })
        else
            try std.fmt.allocPrint(allocator, "graph.act_{s}_{d}", .{ op_tag, op_counter });
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

            if (isParam(inp)) {
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

// ============================================================================
// 3. 模块局部图 (Scoped Local Graph)
// ============================================================================

/// 局部图节点类型枚举 (Flow Node Kind)
pub const FlowNodeKind = enum {
    /// 进入模块的边界张量
    port_in,
    /// 离开模块的边界张量
    port_out,
    /// 直接子模块
    module,
    /// 在本作用域内直接执行的非透明算子
    op,
    /// 在本作用域内创建的常量缓冲区
    buffer,

    pub fn asString(self: FlowNodeKind) []const u8 {
        return @tagName(self);
    }
};

/// 局部图边类型枚举 (Edge Kind)
pub const EdgeKind = enum {
    /// 激活数据流
    data,
    /// 从常量缓冲区出发的边
    buffer,

    pub fn asString(self: EdgeKind) []const u8 {
        return @tagName(self);
    }
};

/// 模块局部图中的节点
pub const FlowNode = struct {
    id: []const u8, // 局部 id；端口使用保留前缀 "@in" / "@out"
    kind: FlowNodeKind,
    ref: []const u8, // 全局引用：子模块路径、张量名，或端口在外部可见的另一端
    op_type: ?[]const u8 = null,
    module_type: ?[]const u8 = null,
    shape: []const u8 = "",
};

/// 模块局部图中的一条数据流边
pub const EdgeData = struct {
    from: []const u8,
    to: []const u8,
    shape: []const u8, // 生产端张量形状
    dst_shape: ?[]const u8 = null, // 经过透明算子后到达消费端的形状 (与 shape 相同时为 null)
    transforms: []const []const u8 = &.{}, // 折叠进该边的透明算子 (按执行顺序)
    is_skip: bool = false,
    kind: EdgeKind = .data,
};

/// 递归模型层级模块组节点 (Recursive Module Node)；序列化时 "kind" 固定为 "module"
pub const ModuleNode = struct {
    name: []const u8,
    path: []const u8,
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
    flow_nodes: std.ArrayList(FlowNode),
    edges: std.ArrayList(EdgeData),

    pub fn init(allocator: std.mem.Allocator, name: []const u8, path: []const u8) !*ModuleNode {
        const node = try allocator.create(ModuleNode);
        node.* = .{
            .name = name,
            .path = path,
            .children = .empty,
            .nodes = .empty,
            .parameters = .empty,
            .ops = .empty,
            .flow_nodes = .empty,
            .edges = .empty,
        };
        return node;
    }

    /// 根模块、复合模块与无参数的多算子叶子模块 (如注意力核心) 导出局部图；
    /// 带参数的叶子层 (Linear、RMSNorm、Embedding 等) 视为原子节点。
    fn wantsLocalGraph(self: *const ModuleNode) bool {
        if (self.path.len == 0) return true;
        if (self.children.items.len > 0) return true;
        return self.parameters.items.len == 0 and self.ops.items.len > 0;
    }
};

const Vertex = union(enum) {
    external,
    transparent,
    child: []const u8,
    own_op: *Op,
    buffer: *Tensor,
    graph_input: *Tensor,
};

const ConsumerMap = std.AutoHashMap(*Tensor, std.ArrayList(*Op));

const LocalGraphBuilder = struct {
    arena: std.mem.Allocator,
    graph: *Graph,
    module: *ModuleNode,
    module_map: *const std.StringHashMap(*ModuleNode),
    consumers: *const ConsumerMap,
    node_ids: std.StringHashMap([]const u8),
    used_ids: std.StringHashMap(void),
    edge_keys: std.StringHashMap(void),
    n_in: usize = 0,
    n_out: usize = 0,

    fn init(
        arena: std.mem.Allocator,
        graph: *Graph,
        module: *ModuleNode,
        module_map: *const std.StringHashMap(*ModuleNode),
        consumers: *const ConsumerMap,
    ) LocalGraphBuilder {
        return .{
            .arena = arena,
            .graph = graph,
            .module = module,
            .module_map = module_map,
            .consumers = consumers,
            .node_ids = std.StringHashMap([]const u8).init(arena),
            .used_ids = std.StringHashMap(void).init(arena),
            .edge_keys = std.StringHashMap(void).init(arena),
        };
    }

    fn classifyOp(self: *const LocalGraphBuilder, op: *Op) Vertex {
        const path = self.module.path;
        const s = opScope(op);
        if (std.mem.eql(u8, s, path)) {
            return if (isTransparentOp(op)) .transparent else .{ .own_op = op };
        }
        if (isWithinScope(s, path)) return .{ .child = childSegment(s, path) };
        return .external;
    }

    fn classifyLeaf(self: *const LocalGraphBuilder, t: *Tensor) Vertex {
        const path = self.module.path;
        const s = t.scope;
        if (std.mem.eql(u8, s, path)) {
            if (!t.is_buffer and path.len == 0) return .{ .graph_input = t };
            return .{ .buffer = t };
        }
        if (isWithinScope(s, path)) return .{ .child = childSegment(s, path) };
        return .external;
    }

    fn producerVertex(self: *const LocalGraphBuilder, t: *Tensor) Vertex {
        if (t.creator) |op| return self.classifyOp(op);
        return self.classifyLeaf(t);
    }

    const Walk = struct {
        origin: *Tensor,
        transforms: []const []const u8,
    };

    /// 沿当前作用域内的透明算子向上回溯，得到真正的生产端张量与折叠的变换序列
    fn walkBack(self: *LocalGraphBuilder, t: *Tensor) !Walk {
        var cur = t;
        var rev: std.ArrayList([]const u8) = .empty;
        while (cur.creator) |op| {
            if (!isTransparentOp(op)) break;
            if (!std.mem.eql(u8, opScope(op), self.module.path)) break;
            const src = firstDataInput(op) orelse break;
            const desc = try std.fmt.allocPrint(self.arena, "{s} {s} -> {s}", .{
                @tagName(op.op_type),
                try formatShapeAlloc(self.arena, src.shape),
                try formatShapeAlloc(self.arena, cur.shape),
            });
            try rev.append(self.arena, desc);
            cur = src;
        }
        std.mem.reverse([]const u8, rev.items);
        return .{ .origin = cur, .transforms = rev.items };
    }

    /// 算子在作用域 scope 的局部图中对应的全局引用 (子模块路径或算子输出张量名)
    fn opRefIn(self: *LocalGraphBuilder, op: *Op, scope: []const u8) ![]const u8 {
        const s = opScope(op);
        if (std.mem.eql(u8, s, scope)) return op.outputs[0].name orelse "op";
        return joinScope(self.arena, scope, childSegment(s, scope));
    }

    /// 输入端口的外部来源：越过透明算子找到真正的生产者，并在最近公共祖先作用域中解析
    fn producerRef(self: *LocalGraphBuilder, t0: *Tensor) ![]const u8 {
        var cur = t0;
        while (cur.creator) |op| {
            if (!isTransparentOp(op)) break;
            cur = firstDataInput(op) orelse break;
        }
        const op = cur.creator orelse return cur.name orelse "input";
        return self.opRefIn(op, commonScope(opScope(op), self.module.path));
    }

    /// 输出端口的外部去向：越过透明算子找到真正的消费者，并在最近公共祖先作用域中解析
    fn consumerRef(self: *LocalGraphBuilder, c: *Op) ![]const u8 {
        var cur = c;
        while (isTransparentOp(cur)) {
            const out = cur.outputs[0];
            const list = self.consumers.get(out) orelse return out.name orelse "output";
            if (list.items.len == 0) return out.name orelse "output";
            cur = list.items[0];
        }
        return self.opRefIn(cur, commonScope(opScope(cur), self.module.path));
    }

    /// 局部 id：去掉当前模块路径前缀后的剩余名；仍含点号时取最后一段
    fn localId(self: *const LocalGraphBuilder, name: []const u8) []const u8 {
        const path = self.module.path;
        var rel = name;
        if (path.len > 0 and name.len > path.len + 1 and std.mem.startsWith(u8, name, path) and name[path.len] == '.') {
            rel = name[path.len + 1 ..];
        }
        if (std.mem.lastIndexOfScalar(u8, rel, '.')) |d| rel = rel[d + 1 ..];
        return rel;
    }

    fn uniqueId(self: *LocalGraphBuilder, preferred: []const u8) ![]const u8 {
        var candidate = preferred;
        var n: usize = 2;
        while (self.used_ids.contains(candidate)) : (n += 1) {
            candidate = try std.fmt.allocPrint(self.arena, "{s}_{d}", .{ preferred, n });
        }
        try self.used_ids.put(candidate, {});
        return candidate;
    }

    fn addNode(self: *LocalGraphBuilder, key: []const u8, preferred_id: []const u8, node: FlowNode) ![]const u8 {
        if (self.node_ids.get(key)) |id| return id;
        const id = try self.uniqueId(preferred_id);
        var n = node;
        n.id = id;
        try self.module.flow_nodes.append(self.arena, n);
        try self.node_ids.put(key, id);
        return id;
    }

    fn vertexId(self: *LocalGraphBuilder, v: Vertex) ![]const u8 {
        switch (v) {
            .child => |seg| {
                const key = try std.fmt.allocPrint(self.arena, "module:{s}", .{seg});
                const ref = try joinScope(self.arena, self.module.path, seg);
                const mtype = if (self.module_map.get(ref)) |m| m.module_type else "Module";
                return self.addNode(key, seg, .{ .id = "", .kind = .module, .ref = ref, .module_type = mtype });
            },
            .own_op => |op| {
                const out = op.outputs[0];
                const name = out.name orelse "op";
                const key = try std.fmt.allocPrint(self.arena, "op:{x}", .{@intFromPtr(op)});
                return self.addNode(key, self.localId(name), .{
                    .id = "",
                    .kind = .op,
                    .ref = name,
                    .op_type = @tagName(op.op_type),
                    .shape = try formatShapeAlloc(self.arena, out.shape),
                });
            },
            .buffer => |t| {
                const name = t.name orelse "buffer";
                const key = try std.fmt.allocPrint(self.arena, "buffer:{x}", .{@intFromPtr(t)});
                return self.addNode(key, self.localId(name), .{
                    .id = "",
                    .kind = .buffer,
                    .ref = name,
                    .shape = try formatShapeAlloc(self.arena, t.shape),
                });
            },
            .graph_input => |t| return self.portIn(t, t.name orelse "input"),
            .external, .transparent => unreachable,
        }
    }

    fn portIn(self: *LocalGraphBuilder, t: *Tensor, ref: []const u8) ![]const u8 {
        const key = try std.fmt.allocPrint(self.arena, "in:{x}", .{@intFromPtr(t)});
        if (self.node_ids.get(key)) |id| return id;
        const preferred = try std.fmt.allocPrint(self.arena, "@in{d}", .{self.n_in});
        self.n_in += 1;
        return self.addNode(key, preferred, .{
            .id = "",
            .kind = .port_in,
            .ref = ref,
            .shape = try formatShapeAlloc(self.arena, t.shape),
        });
    }

    fn portOut(self: *LocalGraphBuilder, t: *Tensor, ref: []const u8) ![]const u8 {
        const key = try std.fmt.allocPrint(self.arena, "out:{x}", .{@intFromPtr(t)});
        if (self.node_ids.get(key)) |id| return id;
        const preferred = try std.fmt.allocPrint(self.arena, "@out{d}", .{self.n_out});
        self.n_out += 1;
        return self.addNode(key, preferred, .{
            .id = "",
            .kind = .port_out,
            .ref = ref,
            .shape = try formatShapeAlloc(self.arena, t.shape),
        });
    }

    fn addEdge(self: *LocalGraphBuilder, from: []const u8, to: []const u8, origin: *Tensor, arrival: *Tensor, transforms: []const []const u8) !void {
        if (std.mem.eql(u8, from, to)) return;
        const key = try std.fmt.allocPrint(self.arena, "{s}->{s}", .{ from, to });
        if (self.edge_keys.contains(key)) return;
        try self.edge_keys.put(key, {});
        try self.module.edges.append(self.arena, .{
            .from = from,
            .to = to,
            .shape = try formatShapeAlloc(self.arena, origin.shape),
            .dst_shape = if (shapeEql(origin.shape, arrival.shape)) null else try formatShapeAlloc(self.arena, arrival.shape),
            .transforms = transforms,
            .kind = if (origin.creator == null and origin.is_buffer) .buffer else .data,
        });
    }

    fn build(self: *LocalGraphBuilder) !void {
        // 1. 每条 (张量 -> 非透明消费者) 数据依赖在当前作用域中的投影
        for (self.graph.ops.items) |c| {
            const cv = self.classifyOp(c);
            if (cv == .transparent) continue;
            for (c.inputs) |t| {
                if (isParam(t)) continue;
                const walk = try self.walkBack(t);
                const origin = walk.origin;
                if (isParam(origin)) continue;
                const pv = self.producerVertex(origin);
                if (pv == .transparent) continue;
                if (pv == .external and cv == .external) continue;

                const from_id = if (pv == .external)
                    try self.portIn(origin, try self.producerRef(origin))
                else
                    try self.vertexId(pv);
                const to_id = if (cv == .external)
                    try self.portOut(t, try self.consumerRef(c))
                else
                    try self.vertexId(cv);
                try self.addEdge(from_id, to_id, origin, t, walk.transforms);
            }
        }

        // 2. 图终端输出 (没有任何消费者的算子输出)
        for (self.graph.ops.items) |p| {
            for (p.outputs) |t| {
                if (self.consumers.contains(t)) continue;
                const walk = try self.walkBack(t);
                const pv = self.producerVertex(walk.origin);
                if (pv == .external or pv == .transparent) continue;
                const from_id = try self.vertexId(pv);
                const to_id = try self.portOut(t, t.name orelse "output");
                try self.addEdge(from_id, to_id, walk.origin, t, walk.transforms);
            }
        }

        // 3. 在局部图内重新计算残差跳跃标记
        self.markSkips();
    }

    fn findNode(self: *const LocalGraphBuilder, id: []const u8) ?FlowNode {
        for (self.module.flow_nodes.items) |n| {
            if (std.mem.eql(u8, n.id, id)) return n;
        }
        return null;
    }

    fn reaches(self: *LocalGraphBuilder, from: []const u8, target: []const u8) bool {
        var queue: std.ArrayList([]const u8) = .empty;
        var visited = std.StringHashMap(void).init(self.arena);
        queue.append(self.arena, from) catch return false;
        visited.put(from, {}) catch return false;

        var head: usize = 0;
        while (head < queue.items.len) : (head += 1) {
            const curr = queue.items[head];
            for (self.module.edges.items) |e| {
                if (!std.mem.eql(u8, e.from, curr)) continue;
                if (std.mem.eql(u8, e.to, target)) return true;
                if (!visited.contains(e.to)) {
                    visited.put(e.to, {}) catch return false;
                    queue.append(self.arena, e.to) catch return false;
                }
            }
        }
        return false;
    }

    /// 边 (u -> t) 为 skip，当且仅当 t 是加法汇聚节点，且 t 另有入边 (w -> t) 使得 u 在局部图内可达 w
    fn markSkips(self: *LocalGraphBuilder) void {
        const edges = self.module.edges.items;
        for (edges, 0..) |*e, i| {
            const target = self.findNode(e.to) orelse continue;
            if (target.kind != .op) continue;
            const op_type = target.op_type orelse continue;
            if (!std.mem.eql(u8, op_type, "Add")) continue;
            for (edges, 0..) |other, j| {
                if (i == j or !std.mem.eql(u8, other.to, e.to)) continue;
                if (std.mem.eql(u8, other.from, e.from)) continue;
                if (self.reaches(e.from, other.from)) {
                    e.is_skip = true;
                    break;
                }
            }
        }
    }
};

// ============================================================================
// 4. 模型层级中间结构 (Model Hierarchy IR)
// ============================================================================

/// 模型全局摘要统计指标 (KPI Summary)
pub const Summary = struct {
    total_params: usize = 0,
    total_bytes: usize = 0,
    param_nodes: usize = 0,
    input_nodes: usize = 0,
    buffer_nodes: usize = 0,
    activation_nodes: usize = 0,
    custom_init_count: usize = 0,
    auto_graph_count: usize = 0,
    total_nodes: usize = 0,
};

/// 包含完整递归模块树 (每个模块自带局部图) 与全局指标的模型结构中间数据
pub const ModelHierarchyGraph = struct {
    arena: std.heap.ArenaAllocator,
    summary: Summary,
    root: *ModuleNode,
    default_scope: []const u8,
    nodes: std.ArrayList(NodeData),
    ops: std.ArrayList(OpData),
    formulas: std.StringHashMap([]const u8),

    pub fn deinit(self: *ModelHierarchyGraph) void {
        self.arena.deinit();
    }
};

// ============================================================================
// 5. 模型计算图与模块层级提取器 (Graph / Model Hierarchy Extractor)
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
            if (leaf.kind == .Param) {
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

    const TreeContext = struct {
        arena: std.mem.Allocator,
        graph: *Graph,
        formulas: *const std.StringHashMap([]const u8),
        module_map: *std.StringHashMap(*ModuleNode),
    };

    /// 获取 (必要时逐级创建) 指定路径的模块节点
    fn ensureModule(ctx: *const TreeContext, path: []const u8) !*ModuleNode {
        if (ctx.module_map.get(path)) |m| return m;
        const parent_path = extractModuleScope(path) orelse "";
        const parent = try ensureModule(ctx, parent_path);

        const path_copy = try ctx.arena.dupe(u8, path);
        const name = if (std.mem.lastIndexOfScalar(u8, path_copy, '.')) |d| path_copy[d + 1 ..] else path_copy;
        const m = try ModuleNode.init(ctx.arena, name, path_copy);
        m.module_type = ctx.graph.getModuleType(path) orelse "Module";
        m.formula = ctx.formulas.get(path) orelse ctx.graph.inferModuleFormula(path);
        try parent.children.append(ctx.arena, m);
        try ctx.module_map.put(path_copy, m);
        return m;
    }

    fn buildLocalGraphs(
        arena: std.mem.Allocator,
        graph: *Graph,
        node: *ModuleNode,
        module_map: *const std.StringHashMap(*ModuleNode),
        consumers: *const ConsumerMap,
    ) !void {
        if (node.wantsLocalGraph()) {
            var builder = LocalGraphBuilder.init(arena, graph, node, module_map, consumers);
            try builder.build();
        }
        for (node.children.items) |child| {
            try buildLocalGraphs(arena, graph, child, module_map, consumers);
        }
    }

    /// 解析 Graph 并构建约定的中间数据结构 ModelHierarchyGraph
    pub fn build(graph: *Graph, allocator: std.mem.Allocator) !ModelHierarchyGraph {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const arena_alloc = arena.allocator();

        const nodes = try collectGraphNodes(graph, arena_alloc);
        const ops_list = try collectGraphOps(graph, arena_alloc);

        var formulas = std.StringHashMap([]const u8).init(arena_alloc);
        var f_it = graph.module_formulas.iterator();
        while (f_it.next()) |entry| {
            try formulas.put(try arena_alloc.dupe(u8, entry.key_ptr.*), try arena_alloc.dupe(u8, entry.value_ptr.*));
        }

        // 0. 注入所有标准基础算子类型 (OpType) 的标准 LaTeX 公式字典
        inline for (std.meta.fields(autodiff.OpType)) |field| {
            const op_enum: autodiff.OpType = @enumFromInt(field.value);
            try formulas.put(try arena_alloc.dupe(u8, field.name), try arena_alloc.dupe(u8, op_enum.getFormula()));
        }

        // 1. 为每个算子实例绑定其算子类型的标准公式
        for (ops_list.items) |*op| {
            if (!formulas.contains(op.name)) {
                if (formulas.get(op.op_type)) |op_form| {
                    try formulas.put(try arena_alloc.dupe(u8, op.name), try arena_alloc.dupe(u8, op_form));
                } else {
                    const inferred = graph.inferModuleFormula(op.name);
                    try formulas.put(try arena_alloc.dupe(u8, op.name), try arena_alloc.dupe(u8, inferred));
                }
            }
            if (op.module.len > 0 and !formulas.contains(op.module)) {
                const inferred = graph.inferModuleFormula(op.module);
                try formulas.put(try arena_alloc.dupe(u8, op.module), try arena_alloc.dupe(u8, inferred));
            }
            op.formula = formulas.get(op.name) orelse (formulas.get(op.op_type) orelse null);
        }

        // 2. 为尚未显式注册公式的节点与其所属模块注入推导公式
        for (nodes.items) |node| {
            if (!formulas.contains(node.name)) {
                if (node.kind == .Activation) {
                    if (formulas.get(node.inferred_act)) |act_form| {
                        try formulas.put(try arena_alloc.dupe(u8, node.name), try arena_alloc.dupe(u8, act_form));
                        continue;
                    }
                }
                const inferred = graph.inferModuleFormula(node.name);
                try formulas.put(try arena_alloc.dupe(u8, node.name), try arena_alloc.dupe(u8, inferred));
            }
            if (node.module.len > 0 and !formulas.contains(node.module)) {
                const inferred = graph.inferModuleFormula(node.module);
                try formulas.put(try arena_alloc.dupe(u8, node.module), try arena_alloc.dupe(u8, inferred));
            }
        }

        // 3. 按作用域构建递归模块树：节点与算子放入其所属模块
        const root = try ModuleNode.init(arena_alloc, "root", "");
        root.module_type = "Model";
        root.formula = graph.getModuleFormula("root") orelse (graph.getModuleFormula("") orelse null);

        var module_map = std.StringHashMap(*ModuleNode).init(arena_alloc);
        try module_map.put("", root);
        const ctx = TreeContext{
            .arena = arena_alloc,
            .graph = graph,
            .formulas = &formulas,
            .module_map = &module_map,
        };

        for (nodes.items) |node| {
            const m = try ensureModule(&ctx, node.module);
            try m.nodes.append(arena_alloc, node);
            if (node.kind == .Param) {
                try m.parameters.append(arena_alloc, node);
            }
        }
        for (ops_list.items) |op| {
            const m = try ensureModule(&ctx, op.module);
            try m.ops.append(arena_alloc, op);
        }

        // 4. 为每个模块构建局部图
        var consumers = ConsumerMap.init(arena_alloc);
        for (graph.ops.items) |op| {
            for (op.inputs) |inp| {
                const gop = try consumers.getOrPut(inp);
                if (!gop.found_existing) gop.value_ptr.* = .empty;
                try gop.value_ptr.append(arena_alloc, op);
            }
        }
        try buildLocalGraphs(arena_alloc, graph, root, &module_map, &consumers);

        aggregateMetrics(root);

        var summary = Summary{
            .total_params = root.total_params,
            .total_bytes = root.total_bytes,
            .total_nodes = nodes.items.len,
        };
        for (nodes.items) |n| {
            switch (n.kind) {
                .Param => {
                    summary.param_nodes += 1;
                    switch (n.status) {
                        .CUSTOM_INIT => summary.custom_init_count += 1,
                        .AUTO_GRAPH => summary.auto_graph_count += 1,
                        .INPUT, .BUFFER, .OP_OUTPUT => {},
                    }
                },
                .Input => summary.input_nodes += 1,
                .Buffer => summary.buffer_nodes += 1,
                .Activation => summary.activation_nodes += 1,
            }
        }

        // 5. 默认展示作用域：根只包裹一个模型容器时展示该容器，否则展示根
        const default_scope: []const u8 = if (root.children.items.len == 1) root.children.items[0].path else "";

        return .{
            .arena = arena,
            .summary = summary,
            .root = root,
            .default_scope = default_scope,
            .nodes = nodes,
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

    fn serializePorts(node: *const ModuleNode, buf: *std.ArrayList(u8), allocator: std.mem.Allocator, kind: FlowNodeKind) !void {
        var first = true;
        for (node.flow_nodes.items) |n| {
            if (n.kind != kind) continue;
            if (!first) try buf.appendSlice(allocator, ",");
            first = false;
            try buf.appendSlice(allocator, "{\"id\": ");
            try writeEscapedJsonString(buf, allocator, n.id);
            try buf.appendSlice(allocator, ",\"ref\": ");
            try writeEscapedJsonString(buf, allocator, n.ref);
            try buf.appendSlice(allocator, ",\"shape\": ");
            try writeEscapedJsonString(buf, allocator, n.shape);
            try buf.appendSlice(allocator, "}");
        }
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
            try writeEscapedJsonString(buf, allocator, p.status.asString());
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
            try buf.appendSlice(allocator, ",\"module\": ");
            try writeEscapedJsonString(buf, allocator, op.module);
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
        try buf.appendSlice(allocator, "],\"ports\": {\"inputs\": [");
        try serializePorts(node, buf, allocator, .port_in);
        try buf.appendSlice(allocator, "],\"outputs\": [");
        try serializePorts(node, buf, allocator, .port_out);
        try buf.appendSlice(allocator, "]},\"flow_nodes\": [");
        for (node.flow_nodes.items, 0..) |n, i| {
            if (i > 0) try buf.appendSlice(allocator, ",");
            try buf.appendSlice(allocator, "{\"id\": ");
            try writeEscapedJsonString(buf, allocator, n.id);
            try buf.appendSlice(allocator, ",\"kind\": ");
            try writeEscapedJsonString(buf, allocator, n.kind.asString());
            try buf.appendSlice(allocator, ",\"ref\": ");
            try writeEscapedJsonString(buf, allocator, n.ref);
            if (n.op_type) |t| {
                try buf.appendSlice(allocator, ",\"op_type\": ");
                try writeEscapedJsonString(buf, allocator, t);
            }
            if (n.module_type) |t| {
                try buf.appendSlice(allocator, ",\"module_type\": ");
                try writeEscapedJsonString(buf, allocator, t);
            }
            try buf.appendSlice(allocator, ",\"shape\": ");
            try writeEscapedJsonString(buf, allocator, n.shape);
            try buf.appendSlice(allocator, "}");
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
            if (e.dst_shape) |ds| {
                try buf.appendSlice(allocator, ",\"dst_shape\": ");
                try writeEscapedJsonString(buf, allocator, ds);
            }
            if (e.transforms.len > 0) {
                try buf.appendSlice(allocator, ",\"transforms\": [");
                for (e.transforms, 0..) |tr, ti| {
                    if (ti > 0) try buf.appendSlice(allocator, ",");
                    try writeEscapedJsonString(buf, allocator, tr);
                }
                try buf.appendSlice(allocator, "]");
            }
            try buf.print(allocator, ",\"is_skip\": {s}", .{if (e.is_skip) "true" else "false"});
            try buf.appendSlice(allocator, ",\"kind\": ");
            try writeEscapedJsonString(buf, allocator, e.kind.asString());
            try buf.appendSlice(allocator, "}");
        }
        try buf.appendSlice(allocator, "],\"nodes\": [");
        for (node.nodes.items, 0..) |leaf, i| {
            if (i > 0) try buf.appendSlice(allocator, ",");
            try buf.appendSlice(allocator, "{\"name\": ");
            try writeEscapedJsonString(buf, allocator, leaf.name);
            try buf.appendSlice(allocator, ",\"kind\": ");
            try writeEscapedJsonString(buf, allocator, leaf.kind.asString());
            try buf.appendSlice(allocator, ",\"module\": ");
            try writeEscapedJsonString(buf, allocator, leaf.module);
            try buf.appendSlice(allocator, ",\"shape\": ");
            try writeEscapedJsonString(buf, allocator, leaf.shape_str);
            try buf.print(allocator, ",\"elements\": {d},\"bytes\": {d},", .{ leaf.elements, leaf.bytes });
            try buf.appendSlice(allocator, "\"status\": ");
            try writeEscapedJsonString(buf, allocator, leaf.status.asString());
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

        try json_buf.appendSlice(allocator, "{\n  \"version\": \"" ++ SCHEMA_VERSION ++ "\",\n  \"summary\": {");
        try json_buf.print(allocator,
            \\ "total_params": {d}, "total_bytes": {d}, "param_nodes": {d}, "input_nodes": {d}, "buffer_nodes": {d}, "activation_nodes": {d}, "custom_init_count": {d}, "auto_graph_count": {d}, "total_nodes": {d}
        , .{
            model_graph.summary.total_params,
            model_graph.summary.total_bytes,
            model_graph.summary.param_nodes,
            model_graph.summary.input_nodes,
            model_graph.summary.buffer_nodes,
            model_graph.summary.activation_nodes,
            model_graph.summary.custom_init_count,
            model_graph.summary.auto_graph_count,
            model_graph.summary.total_nodes,
        });
        try json_buf.appendSlice(allocator, "},\n  \"default_scope\": ");
        try writeEscapedJsonString(&json_buf, allocator, model_graph.default_scope);
        try json_buf.appendSlice(allocator, ",\n  \"root\": ");
        try serializeModuleTree(model_graph.root, &json_buf, allocator);
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
// 6. 便捷顶层重导出 (Top-Level Re-Exports)
// ============================================================================
pub const buildModelHierarchy = graph_ir.build;
pub const generateJson = graph_ir.generateJson;
pub const exportJson = graph_ir.exportJson;
