const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const init_mod = @import("init.zig");

const Tensor = tensor.Tensor;
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

/// 将计算图结构与各层初始化详情格式化为可交互、层级展开的 HTML 网页文档
pub fn generateHtmlReport(graph: *Graph, allocator: std.mem.Allocator) ![]const u8 {
    var nodes = try collectGraphNodes(graph, allocator);
    defer freeGraphNodes(&nodes, allocator);

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
        \\    .module-content { padding: 10px 16px; display: flex; flex-direction: column; gap: 8px; }
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
        \\  <!-- Controls -->
        \\  <div class="controls-bar">
        \\    <div class="search-box">
        \\      <span class="search-icon">🔍</span>
        \\      <input type="text" id="search-input" class="search-input" placeholder="Search by module path, node name, op, or shape..." oninput="filterNodes()">
        \\    </div>
        \\    <div class="btn-group">
        \\      <button class="btn active" onclick="setKindFilter('all', this)">All Nodes ({d})</button>
        \\      <button class="btn" onclick="setKindFilter('Param', this)">Params ({d})</button>
        \\      <button class="btn" onclick="setKindFilter('Input', this)">Inputs ({d})</button>
        \\      <button class="btn" onclick="setKindFilter('Activation', this)">Activations ({d})</button>
        \\      <button class="btn" onclick="expandAll()">Expand All</button>
        \\      <button class="btn" onclick="collapseAll()">Collapse All</button>
        \\    </div>
        \\  </div>
        \\
        \\  <!-- Hierarchical Container -->
        \\  <div class="hierarchy-container" id="tree-container">
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
        \\  </div>
        \\  <footer>Generated automatically by ZNN Autodiff Engine</footer>
        \\</div>
        \\
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
        \\let currentKindFilter = 'all';
        \\let currentQuery = '';
        \\
        \\function formatNumber(num) {
        \\  return num.toLocaleString();
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
        \\function renderTableRows(nodes) {
        \\  return nodes.map(n => {
        \\    const kindBadge = n.kind === 'Param' ? 'badge-param' : (n.kind === 'Input' ? 'badge-input' : 'badge-act');
        \\    const statusBadge = n.status === 'CUSTOM_INIT' ? 'badge-custom' : (n.status === 'AUTO_GRAPH' ? 'badge-auto' : 'badge-op');
        \\    return `
        \\      <tr class="node-row" data-name="${n.name.toLowerCase()}" data-kind="${n.kind}" data-act="${n.act.toLowerCase()}" data-shape="${n.shape}">
        \\        <td class="node-name">${n.name}</td>
        \\        <td><span class="badge ${kindBadge}">${n.kind}</span></td>
        \\        <td class="node-shape">${n.shape}</td>
        \\        <td style="font-family: var(--font-mono);">${formatNumber(n.elements)}</td>
        \\        <td><span class="badge ${statusBadge}">${n.status}</span></td>
        \\        <td style="font-family: var(--font-mono);">${n.act}</td>
        \\        <td class="strategy-col">${n.strategy}</td>
        \\      </tr>
        \\    `;
        \\  }).join('');
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
        \\    if (node._nodes.length > 0) {
        \\      html += `
        \\        <table class="node-table">
        \\          <thead>
        \\            <tr>
        \\              <th>Node Name</th>
        \\              <th>Kind</th>
        \\              <th>Shape</th>
        \\              <th>Elements</th>
        \\              <th>Status</th>
        \\              <th>Inferred Act / Op</th>
        \\              <th>Initialization Strategy</th>
        \\            </tr>
        \\          </thead>
        \\          <tbody>
        \\            ${renderTableRows(node._nodes)}
        \\          </tbody>
        \\        </table>
        \\      `;
        \\    }
        \\
        \\    for (const childKey of childKeys) {
        \\      const fullChildName = isRoot ? childKey : `${prefix}.${childKey}`;
        \\      html += renderBranch(fullChildName, node._children[childKey]);
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
        \\  for (const modKey of Object.keys(hierarchy._children)) {
        \\    outputHtml += renderBranch(modKey, hierarchy._children[modKey]);
        \\  }
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
        \\renderTree();
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
