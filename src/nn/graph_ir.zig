const std = @import("std");
const autodiff = @import("../autodiff.zig");
const vis = @import("visualization.zig");

const Graph = autodiff.Graph;
const SCHEMA_VERSION = vis.SCHEMA_VERSION;
const extractModuleScope = vis.extractModuleScope;
const collectGraphNodes = vis.collectGraphNodes;
const collectGraphOps = vis.collectGraphOps;
const FlowNodeKind = vis.FlowNodeKind;
const ModuleNode = vis.ModuleNode;
const ConsumerMap = vis.ConsumerMap;
const LocalGraphBuilder = vis.LocalGraphBuilder;
const Summary = vis.Summary;
const ModelHierarchyGraph = vis.ModelHierarchyGraph;

// ============================================================================
// 模型计算图与模块层级中间表示提取器 (Graph / Model Hierarchy Intermediate Representation, IR Extractor)
// - 输入: *autodiff.Graph
// - 输出: 约定的中间数据结构 ModelHierarchyGraph 或 递归 JavaScript 对象表示法 (JavaScript Object Notation, JSON) 字符串
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

        // 5. 默认展示作用域：根只包裹一个模型容器且该容器包含局部边时展示该容器，否则展示根
        const default_scope: []const u8 = if (root.children.items.len == 1 and root.children.items[0].edges.items.len > 0) root.children.items[0].path else "";

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
