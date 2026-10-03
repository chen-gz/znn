const std = @import("std");
const autodiff = @import("../autodiff.zig");
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");

const Tensor = tensor.Tensor;
const core = nn.core;
const transformer = nn.transformer;
const visualization = nn.visualization;
const graph_ir = nn.graph_ir;
const NodeKind = nn.NodeKind;
const NodeStatus = nn.NodeStatus;
const FlowNodeKind = nn.FlowNodeKind;
const EdgeKind = nn.EdgeKind;

const Linear = nn.Linear;
const Conv2D = nn.Conv2D;
const sequential = nn.sequential;
const LeakyReLU = nn.LeakyReLU;
const LayerNorm = nn.LayerNorm;
const RNN = nn.RNN;
const LSTM = nn.LSTM;
const StackedLSTM = nn.StackedLSTM;
const GRU = nn.GRU;
const Embedding = nn.Embedding;
const SwiGLU = nn.SwiGLU;
const MoELayer = nn.MoELayer;
const CausalSelfAttention = nn.CausalSelfAttention;
const MLALayer = nn.MLALayer;
const TransformerBlock = nn.TransformerBlock;
const GPTConfig = nn.GPTConfig;
const GPT = nn.GPT;
const LoRALinear = nn.LoRALinear;

test "Hierarchical module naming and interactive HTML report export" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(13579);
    const random = prng.random();

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    // 1. 创建 Embedding 模块并设置顶级层次命名
    var emb = try transformer.Embedding.init(allocator, 1000, 64, random);
    defer emb.deinit(allocator);
    emb.setName("gpt.wte");

    // 2. 创建 TransformerBlock 并分层命名为 "gpt.layers.0"
    var block = try transformer.TransformerBlock.init(allocator, 64, 4, random);
    defer block.deinit(allocator);
    block.setName("gpt.layers.0");
    try graph.registerModuleType("gpt", "GPT");

    // 验证子层参数名称是否按层次正确拼接
    try std.testing.expectEqualStrings("gpt.wte.weight", emb.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.ln_1.weight", block.ln_1.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.attn.q_attn.weight", block.attn.q_attn.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.attn.k_attn.weight", block.attn.k_attn.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.attn.v_attn.weight", block.attn.v_attn.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.attn.c_proj.weight", block.attn.c_proj.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.mlp.c_fc.weight", block.mlp.c_fc.weight.name.?);
    try std.testing.expectEqualStrings("gpt.layers.0.mlp.c_proj.weight", block.mlp.c_proj.weight.name.?);

    // 3. 构建前向计算图并命名输入与中间激活节点
    const input_tokens = try graph.zeros(&.{ 2, 8, 64 }, false);
    input_tokens.setName("inputs.token_embeddings");

    const block_out = try block.forward(&graph, input_tokens);
    block_out.setName("activations.block_0_out");

    // 4. 生成递归结构 JSON (后端生成递归数据结构，直接提供给前端解析)
    const json_data = try graph.formatJson(allocator);
    defer allocator.free(json_data);

    // 校验 JSON 结构可正常被解析且包含完整的递归模型树
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_data, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value == .object);

    const root_obj = parsed.value.object.get("root").?.object;
    try std.testing.expectEqualStrings("root", root_obj.get("name").?.string);
    try std.testing.expect(root_obj.get("total_params").?.integer > 0);
    try std.testing.expect(root_obj.get("children").?.array.items.len > 0);

    // 校验根与子模块直接内嵌包含 formula, module_type, parameters, ops 与 edges
    // 图输入归属于 root.nodes，不再生成 "inputs" 伪模块，因此 gpt 是唯一的根子模块
    try std.testing.expectEqual(@as(usize, 1), root_obj.get("children").?.array.items.len);
    const gpt_child = root_obj.get("children").?.array.items[0].object; // "gpt"
    try std.testing.expectEqualStrings("GPT", gpt_child.get("module_type").?.string);
    try std.testing.expect(gpt_child.get("formula") != null);

    const layer0_obj = gpt_child.get("children").?.array.items[0].object.get("children").?.array.items[0].object; // "gpt.layers.0"
    try std.testing.expectEqualStrings("TransformerBlock", layer0_obj.get("module_type").?.string);
    try std.testing.expect(std.mem.indexOf(u8, layer0_obj.get("formula").?.string, "TransformerBlock") != null);
    try std.testing.expect(std.mem.indexOf(u8, layer0_obj.get("formula").?.string, "Attention") != null);
    try std.testing.expect(std.mem.indexOf(u8, layer0_obj.get("formula").?.string, "MLP") != null);
    try std.testing.expect(layer0_obj.get("ops").?.array.items.len > 0);
    try std.testing.expect(layer0_obj.get("edges").?.array.items.len > 0);

    const q_attn_obj = layer0_obj.get("children").?.array.items[1].object.get("children").?.array.items[0].object; // "q_attn"
    try std.testing.expectEqualStrings("Linear", q_attn_obj.get("module_type").?.string);
    try std.testing.expectEqualStrings("y = x W^T + b", q_attn_obj.get("formula").?.string);
    try std.testing.expectEqual(2, q_attn_obj.get("parameters").?.array.items.len);
    try std.testing.expect(q_attn_obj.get("ops").?.array.items.len >= 2);

    const summary_obj = parsed.value.object.get("summary").?.object;
    try std.testing.expect(summary_obj.get("total_params").?.integer > 0);
    try std.testing.expect(summary_obj.get("param_nodes").?.integer >= 8);

    // 校验 schema 2.0 顶层结构：nodes, ops, formulas, edges 均已就近集成进模块树，
    // 顶层仅保留 default_scope 指示前端初始展开的作用域
    try std.testing.expectEqualStrings("2.0", parsed.value.object.get("version").?.string);
    try std.testing.expect(parsed.value.object.get("nodes") == null);
    try std.testing.expect(parsed.value.object.get("edges") == null);
    try std.testing.expect(parsed.value.object.get("ops") == null);
    try std.testing.expect(parsed.value.object.get("formulas") == null);
    try std.testing.expectEqualStrings("gpt", parsed.value.object.get("default_scope").?.string);

    // 校验 Block0 级别的拓扑边 (Edges) 正确性：
    // 1) 包含前向流 ln_1 -> attn 以及残差边 inputs.token_embeddings -> residual_attn (is_skip = true)
    // 2) 严禁包含子模块内部 Q/K/V 指向 attn 的泄露边
    var found_ln1_to_attn = false;
    var found_skip_to_res = false;
    var leaked_qkv_to_attn = false;
    for (layer0_obj.get("edges").?.array.items) |e_val| {
        const edge = e_val.object;
        const from_s = edge.get("from").?.string;
        const to_s = edge.get("to").?.string;
        const is_skip = edge.get("is_skip").?.bool;

        if (std.mem.indexOf(u8, from_s, "ln_1") != null and std.mem.indexOf(u8, to_s, "attn") != null and !is_skip) {
            found_ln1_to_attn = true;
        }
        if (std.mem.indexOf(u8, to_s, "residual_attn") != null and is_skip) {
            found_skip_to_res = true;
        }
        if ((std.mem.eql(u8, from_s, "q_attn") or std.mem.eql(u8, from_s, "k_attn") or std.mem.eql(u8, from_s, "v_attn")) and std.mem.eql(u8, to_s, "attn")) {
            leaked_qkv_to_attn = true;
        }
    }
    try std.testing.expect(found_ln1_to_attn);
    try std.testing.expect(found_skip_to_res);
    try std.testing.expect(!leaked_qkv_to_attn);

    // 校验参数矩阵守恒与内存指标递归一致性 (Conservation Check)
    const ParamCounter = struct {
        fn countParams(obj: std.json.ObjectMap) usize {
            var sum: usize = 0;
            if (obj.get("parameters")) |p_val| {
                for (p_val.array.items) |p_item| {
                    if (p_item.object.get("elements")) |el| {
                        sum += @as(usize, @intCast(el.integer));
                    }
                }
            }
            if (obj.get("children")) |c_val| {
                for (c_val.array.items) |c_item| {
                    sum += countParams(c_item.object);
                }
            }
            return sum;
        }
    };
    const total_recursed_params = ParamCounter.countParams(root_obj);
    try std.testing.expectEqual(summary_obj.get("total_params").?.integer, @as(i64, @intCast(total_recursed_params)));

    // 校验算子（Ops）拓扑与张量流转维度非空完整性
    for (q_attn_obj.get("ops").?.array.items) |op_val| {
        const op_obj = op_val.object;
        try std.testing.expect(op_obj.get("op_type") != null);
        try std.testing.expect(op_obj.get("input_shape") != null);
        try std.testing.expect(op_obj.get("output_shape") != null);
        try std.testing.expect(op_obj.get("elements").?.integer > 0);
        try std.testing.expect(op_obj.get("bytes").?.integer > 0);
    }

    // 5. 测试将递归 JSON 导出到临时测试文件
    const tmp_json_path = "tmp_test_model_graph.json";
    try graph.exportJson(tmp_json_path);

    const json_z = try allocator.dupeZ(u8, tmp_json_path);
    defer allocator.free(json_z);
    const c_api = struct { extern "c" fn remove(filename: [*:0]const u8) c_int; };
    defer _ = c_api.remove(json_z.ptr);
    const fj = std.c.fopen(json_z.ptr, "rb") orelse return error.CannotOpenFile;
    defer _ = std.c.fclose(fj);
    var check_json_buf: [1024]u8 = undefined;
    const json_bytes_read = std.c.fread(&check_json_buf, 1, check_json_buf.len, fj);
    try std.testing.expect(json_bytes_read > 200);

    // 6. 测试直接从 ModelHierarchyGraph 序列化 JSON (与 Graph 实例解耦)
    var model_hierarchy = try graph_ir.build(&graph, allocator);
    defer model_hierarchy.deinit();
    const json_from_struct = try graph_ir.serializeJson(&model_hierarchy, allocator);
    defer allocator.free(json_from_struct);
    try std.testing.expect(std.mem.indexOf(u8, json_from_struct, "\"module_type\": \"GPT\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_from_struct, "\"gpt.layers.0.attn.q_attn.weight\"") != null);
}

test "End-to-End Multi-layer GPT JSON Graph Topology and Cross-layer Connectivity" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(54321);
    const random = prng.random();

    // 1. 配置 2 层标准 GPT 模型
    const config = transformer.GPTConfig{
        .vocab_size = 128,
        .block_size = 16,
        .n_embd = 32,
        .n_head = 2,
        .n_layer = 2,
    };

    var gpt = try transformer.GPT(config).init(allocator, random);
    defer gpt.deinit(allocator);
    gpt.setName("gpt");

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    // 2. 构造输入张量并执行前向传播
    const batch_size: usize = 2;
    const seq_len: usize = 8;
    const token_data = try allocator.alloc(f32, batch_size * seq_len);
    defer allocator.free(token_data);
    for (token_data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 50));

    const input_tokens = try graph.tensorNDWithData(&.{ batch_size, seq_len }, token_data, false);
    input_tokens.setName("inputs.token_ids");

    const logits = try gpt.forward(&graph, input_tokens);
    logits.setName("outputs.logits");

    // 3. 构建 ModelHierarchyGraph 与 JSON
    const json_data = try graph.formatJson(allocator);
    defer allocator.free(json_data);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_data, .{});
    defer parsed.deinit();

    const root_obj = parsed.value.object.get("root").?.object;
    const gpt_obj = root_obj.get("children").?.array.items[0].object; // "gpt"
    const layers_obj = gpt_obj.get("children").?.array.items[2].object; // "layers"
    try std.testing.expectEqualStrings("TransformerDecoder", layers_obj.get("module_type").?.string);

    // 4. 校验跨层连接 (Cross-layer continuity between Layer 0 and Layer 1)
    // layers 容器内部必须正确记录 0 -> 1 的前向流以及 0 到 1 的残差连接
    var found_layer0_to_1 = false;
    for (layers_obj.get("edges").?.array.items) |e_item| {
        const edge = e_item.object;
        const from_s = edge.get("from").?.string;
        const to_s = edge.get("to").?.string;
        if (std.mem.eql(u8, from_s, "0") and std.mem.eql(u8, to_s, "1")) {
            found_layer0_to_1 = true;
        }
    }
    try std.testing.expect(found_layer0_to_1);

    // 5. 校验 Layer 1 内部的残差汇聚结构与公式
    const layer1_obj = layers_obj.get("children").?.array.items[1].object; // "gpt.layers.1"
    try std.testing.expectEqualStrings("TransformerBlock", layer1_obj.get("module_type").?.string);
    try std.testing.expect(layer1_obj.get("edges").?.array.items.len > 0);

    var found_layer1_res_skip = false;
    for (layer1_obj.get("edges").?.array.items) |e_item| {
        const edge = e_item.object;
        const to_s = edge.get("to").?.string;
        const is_skip = edge.get("is_skip").?.bool;
        if (std.mem.indexOf(u8, to_s, "residual_attn") != null and is_skip) {
            found_layer1_res_skip = true;
        }
    }
    try std.testing.expect(found_layer1_res_skip);
}

/// 可视化测试辅助：按完整路径查找模块节点，并以 "from->to[ skip| buffer]" 形式比较局部边集合
const VisTestUtil = struct {
    fn findModule(node: std.json.ObjectMap, path: []const u8) ?std.json.ObjectMap {
        if (std.mem.eql(u8, node.get("path").?.string, path)) return node;
        if (node.get("children")) |children| {
            for (children.array.items) |child| {
                if (findModule(child.object, path)) |m| return m;
            }
        }
        return null;
    }

    fn expectEdgeSet(allocator: std.mem.Allocator, root: std.json.ObjectMap, path: []const u8, expected: []const []const u8) !void {
        const module = findModule(root, path) orelse return error.ModuleNotFound;
        var actual: std.ArrayList([]u8) = .empty;
        defer {
            for (actual.items) |s| allocator.free(s);
            actual.deinit(allocator);
        }
        for (module.get("edges").?.array.items) |e_val| {
            const edge = e_val.object;
            const tag: []const u8 = if (edge.get("is_skip").?.bool)
                " skip"
            else if (std.mem.eql(u8, edge.get("kind").?.string, "buffer"))
                " buffer"
            else
                "";
            try actual.append(allocator, try std.fmt.allocPrint(allocator, "{s}->{s}{s}", .{
                edge.get("from").?.string, edge.get("to").?.string, tag,
            }));
        }
        errdefer {
            std.debug.print("edge set mismatch in scope '{s}', actual edges:\n", .{path});
            for (actual.items) |s| std.debug.print("  {s}\n", .{s});
        }
        try std.testing.expectEqual(expected.len, actual.items.len);
        for (expected) |want| {
            var found = false;
            for (actual.items) |have| {
                if (std.mem.eql(u8, want, have)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std.debug.print("missing edge: {s}\n", .{want});
                return error.MissingEdge;
            }
        }
    }

    fn findEdge(root: std.json.ObjectMap, path: []const u8, from: []const u8, to: []const u8) !std.json.ObjectMap {
        const module = findModule(root, path) orelse return error.ModuleNotFound;
        for (module.get("edges").?.array.items) |e_val| {
            const edge = e_val.object;
            if (std.mem.eql(u8, edge.get("from").?.string, from) and std.mem.eql(u8, edge.get("to").?.string, to)) return edge;
        }
        return error.EdgeNotFound;
    }

    fn expectPortRef(root: std.json.ObjectMap, path: []const u8, direction: []const u8, index: usize, ref: []const u8) !void {
        const module = findModule(root, path) orelse return error.ModuleNotFound;
        const ports = module.get("ports").?.object.get(direction).?.array.items;
        try std.testing.expect(index < ports.len);
        try std.testing.expectEqualStrings(ref, ports[index].object.get("ref").?.string);
    }
};

test "Explicit module scopes attribute ops and tensors to the executing module" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(24680);
    const random = prng.random();

    const config = transformer.GPTConfig{ .vocab_size = 128, .block_size = 16, .n_embd = 32, .n_head = 2, .n_layer = 2 };
    var gpt = try transformer.GPT(config).init(allocator, random);
    defer gpt.deinit(allocator);
    gpt.setName("gpt");

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    var token_data: [2 * 8]f32 = undefined;
    for (&token_data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 50));
    const input_tokens = try graph.tensorNDWithData(&.{ 2, 8 }, &token_data, false);
    input_tokens.setName("inputs.token_ids");
    _ = try gpt.forward(&graph, input_tokens);

    // forward 结束后作用域栈必须完全弹出
    try std.testing.expectEqualStrings("", graph.currentScope());

    // 图输入在任何模块之外创建，归属根作用域
    try std.testing.expectEqualStrings("", input_tokens.scope);

    var gelu_scope: ?[]const u8 = null;
    var core_transpose_found = false;
    var attn_reshape_found = false;
    for (graph.ops.items) |o| {
        switch (o.op_type) {
            .Gelu => if (gelu_scope == null) {
                gelu_scope = o.scope;
            },
            .Transpose => if (std.mem.eql(u8, o.scope, "gpt.layers.0.attn.core")) {
                core_transpose_found = true;
            },
            .Reshape => if (std.mem.eql(u8, o.scope, "gpt.layers.0.attn")) {
                attn_reshape_found = true;
            },
            else => {},
        }
        // 叶子模块 ln_1 只执行 RMSNorm 计算，不应拥有任何 Reshape
        if (o.op_type == .Reshape) try std.testing.expect(!std.mem.eql(u8, o.scope, "gpt.layers.0.ln_1"));
    }
    try std.testing.expectEqualStrings("gpt.layers.0.mlp", gelu_scope.?);
    try std.testing.expect(core_transpose_found);
    try std.testing.expect(attn_reshape_found);

    // 输出 logits 的最终 Reshape 由 GPT 自身执行，归属 "gpt"
    const last_op = graph.ops.items[graph.ops.items.len - 1];
    try std.testing.expectEqual(autodiff.OpType.Reshape, last_op.op_type);
    try std.testing.expectEqualStrings("gpt", last_op.scope);

    // pos_indices 是 GPT 内部创建的静态缓冲区
    var pos_found = false;
    for (graph.tensors.items) |t| {
        if (t.name) |n| if (std.mem.eql(u8, n, "gpt.pos_indices")) {
            pos_found = true;
            try std.testing.expect(t.is_buffer);
            try std.testing.expectEqualStrings("gpt", t.scope);
        };
    }
    try std.testing.expect(pos_found);

    // 模块类型由 enterModule 自动注册
    try std.testing.expectEqualStrings("CausalSelfAttention", graph.module_types.get("gpt.layers.0.attn").?);
    try std.testing.expectEqualStrings("ScaledDotProductAttention", graph.module_types.get("gpt.layers.0.attn.core").?);
}

test "Scoped local graph export matches golden edge sets (schema 2.0)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(97531);
    const random = prng.random();

    const config = transformer.GPTConfig{ .vocab_size = 128, .block_size = 16, .n_embd = 32, .n_head = 2, .n_layer = 2 };
    var gpt = try transformer.GPT(config).init(allocator, random);
    defer gpt.deinit(allocator);
    gpt.setName("gpt");

    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    var token_data: [2 * 8]f32 = undefined;
    for (&token_data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 50));
    const input_tokens = try graph.tensorNDWithData(&.{ 2, 8 }, &token_data, false);
    input_tokens.setName("inputs.token_ids");
    const logits = try gpt.forward(&graph, input_tokens);
    logits.setName("outputs.logits");

    const json_data = try graph.formatJson(allocator);
    defer allocator.free(json_data);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_data, .{});
    defer parsed.deinit();

    const top = parsed.value.object;
    try std.testing.expectEqualStrings("2.0", top.get("version").?.string);
    try std.testing.expectEqualStrings("gpt", top.get("default_scope").?.string);
    try std.testing.expect(top.get("edges") == null);
    const summary = top.get("summary").?.object;
    try std.testing.expectEqual(@as(i64, 1), summary.get("input_nodes").?.integer);
    // pos_indices + 每层一个 causal_mask
    try std.testing.expectEqual(@as(i64, 3), summary.get("buffer_nodes").?.integer);

    const root = top.get("root").?.object;

    // 根作用域：图输入/输出端口与 gpt 相连
    try VisTestUtil.expectEdgeSet(allocator, root, "", &.{ "@in0->gpt", "gpt->@out0" });
    try VisTestUtil.expectPortRef(root, "", "inputs", 0, "inputs.token_ids");
    try VisTestUtil.expectPortRef(root, "", "outputs", 0, "outputs.logits");

    // gpt：pos_indices 为缓冲区边，而非模型输入；Reshape 折叠进边的 transforms
    try VisTestUtil.expectEdgeSet(allocator, root, "gpt", &.{
        "@in0->wte",
        "pos_indices->wpe buffer",
        "wte->embeddings_sum",
        "wpe->embeddings_sum",
        "embeddings_sum->layers",
        "layers->lm_head",
        "lm_head->@out0",
    });
    const to_head = try VisTestUtil.findEdge(root, "gpt", "layers", "lm_head");
    try std.testing.expectEqualStrings("[2, 8, 32]", to_head.get("shape").?.string);
    try std.testing.expectEqualStrings("[16, 32]", to_head.get("dst_shape").?.string);
    try std.testing.expectEqual(@as(usize, 1), to_head.get("transforms").?.array.items.len);

    // gpt.layers：层间串联，无虚假残差
    try VisTestUtil.expectEdgeSet(allocator, root, "gpt.layers", &.{ "@in0->0", "0->1", "1->ln_f", "ln_f->@out0" });

    // 每个 Block：两条残差边都在 Block 自身作用域内，且端口引用在最近公共作用域中解析
    const block_edges = [_][]const u8{
        "@in0->ln_1",
        "ln_1->attn",
        "@in0->residual_attn skip",
        "attn->residual_attn",
        "residual_attn->ln_2",
        "ln_2->mlp",
        "residual_attn->residual_mlp skip",
        "mlp->residual_mlp",
        "residual_mlp->@out0",
    };
    try VisTestUtil.expectEdgeSet(allocator, root, "gpt.layers.0", &block_edges);
    try VisTestUtil.expectEdgeSet(allocator, root, "gpt.layers.1", &block_edges);
    try VisTestUtil.expectPortRef(root, "gpt.layers.0", "inputs", 0, "gpt.embeddings_sum");
    try VisTestUtil.expectPortRef(root, "gpt.layers.0", "outputs", 0, "gpt.layers.1");
    try VisTestUtil.expectPortRef(root, "gpt.layers.1", "outputs", 0, "gpt.layers.ln_f");

    // 注意力：共享输入扇出到 q/k/v，核心计算封装在 core 子作用域
    try VisTestUtil.expectEdgeSet(allocator, root, "gpt.layers.0.attn", &.{
        "@in0->q_attn",
        "@in0->k_attn",
        "@in0->v_attn",
        "q_attn->core",
        "k_attn->core",
        "v_attn->core",
        "core->c_proj",
        "c_proj->@out0",
    });

    // core：无参数多算子叶子导出算子级局部图，causal_mask 以缓冲区边接入
    const core_mod = VisTestUtil.findModule(root, "gpt.layers.0.attn.core").?;
    try std.testing.expectEqualStrings("ScaledDotProductAttention", core_mod.get("module_type").?.string);
    try std.testing.expectEqual(@as(usize, 3), core_mod.get("ports").?.object.get("inputs").?.array.items.len);
    var mask_edges: usize = 0;
    for (core_mod.get("edges").?.array.items) |e_val| {
        const edge = e_val.object;
        if (std.mem.eql(u8, edge.get("from").?.string, "causal_mask")) {
            try std.testing.expectEqualStrings("buffer", edge.get("kind").?.string);
            mask_edges += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 1), mask_edges);

    // MLP：激活函数作为命名算子节点出现
    try VisTestUtil.expectEdgeSet(allocator, root, "gpt.layers.0.mlp", &.{ "@in0->c_fc", "c_fc->gelu", "gelu->c_proj", "c_proj->@out0" });

    // 带参数的叶子模块 (Linear / RMSNorm) 不导出算子级局部图
    for ([_][]const u8{ "gpt.layers.0.attn.q_attn", "gpt.layers.0.ln_1" }) |leaf_path| {
        const leaf = VisTestUtil.findModule(root, leaf_path).?;
        if (leaf.get("edges")) |e| try std.testing.expectEqual(@as(usize, 0), e.array.items.len);
    }
}

/// 可视化测试辅助：按 model_graph.schema.json 所用的 JSON Schema 关键字子集校验 JSON
/// ($ref, type, const, enum, minimum, required, properties, additionalProperties, items)
const SchemaCheck = struct {
    defs: std.json.ObjectMap,
    /// 首个违规所在的字段名 (未违规时为空)
    field: []const u8 = "",

    fn init(schema: std.json.Value) SchemaCheck {
        return .{ .defs = schema.object.get("$defs").?.object };
    }

    fn resolve(self: SchemaCheck, schema: std.json.ObjectMap) std.json.ObjectMap {
        if (schema.get("$ref")) |r| {
            const prefix = "#/$defs/";
            std.debug.assert(std.mem.startsWith(u8, r.string, prefix));
            return self.defs.get(r.string[prefix.len..]).?.object;
        }
        return schema;
    }

    fn typeMatches(name: []const u8, v: std.json.Value) bool {
        const eql = std.mem.eql;
        return switch (v) {
            .null => eql(u8, name, "null"),
            .bool => eql(u8, name, "boolean"),
            .integer => eql(u8, name, "integer") or eql(u8, name, "number"),
            .float, .number_string => eql(u8, name, "number"),
            .string => eql(u8, name, "string"),
            .array => eql(u8, name, "array"),
            .object => eql(u8, name, "object"),
        };
    }

    fn check(self: *SchemaCheck, schema_in: std.json.ObjectMap, v: std.json.Value) !void {
        const schema = self.resolve(schema_in);
        if (schema.get("type")) |t| {
            const ok = switch (t) {
                .string => |name| typeMatches(name, v),
                .array => |names| blk: {
                    for (names.items) |name| {
                        if (typeMatches(name.string, v)) break :blk true;
                    }
                    break :blk false;
                },
                else => return error.InvalidSchema,
            };
            if (!ok) return error.SchemaTypeMismatch;
        }
        if (schema.get("const")) |c| {
            if (v != .string or !std.mem.eql(u8, c.string, v.string)) return error.SchemaConstMismatch;
        }
        if (schema.get("enum")) |e| {
            if (v != .string) return error.SchemaEnumMismatch;
            for (e.array.items) |allowed| {
                if (std.mem.eql(u8, allowed.string, v.string)) break;
            } else return error.SchemaEnumMismatch;
        }
        if (schema.get("minimum")) |m| {
            if (v == .integer and v.integer < m.integer) return error.SchemaBelowMinimum;
        }
        switch (v) {
            .object => |obj| {
                if (schema.get("required")) |req| {
                    for (req.array.items) |key| {
                        if (obj.get(key.string) == null) {
                            self.field = key.string;
                            return error.SchemaMissingField;
                        }
                    }
                }
                const closed = if (schema.get("additionalProperties")) |ap| ap == .bool and !ap.bool else false;
                const props = schema.get("properties");
                var it = obj.iterator();
                while (it.next()) |entry| {
                    const key = entry.key_ptr.*;
                    if (props) |p| {
                        if (p.object.get(key)) |sub| {
                            self.check(sub.object, entry.value_ptr.*) catch |err| {
                                if (self.field.len == 0) self.field = key;
                                return err;
                            };
                            continue;
                        }
                    }
                    if (closed) {
                        self.field = key;
                        return error.SchemaUnknownField;
                    }
                }
            },
            .array => |arr| {
                if (schema.get("items")) |items| {
                    for (arr.items) |item| try self.check(items.object, item);
                }
            },
            else => {},
        }
    }

    fn run(allocator: std.mem.Allocator, json_text: []const u8, report: bool) !void {
        var schema = try std.json.parseFromSlice(std.json.Value, allocator, visualization.SCHEMA_JSON, .{});
        defer schema.deinit();
        var doc = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
        defer doc.deinit();
        var checker = SchemaCheck.init(schema.value);
        checker.check(schema.value.object, doc.value) catch |err| {
            if (report) std.debug.print("schema violation {s} at field \"{s}\"\n", .{ @errorName(err), checker.field });
            return err;
        };
    }

    /// 校验并返回首个违规错误 (不输出诊断)
    fn validate(allocator: std.mem.Allocator, json_text: []const u8) !void {
        return run(allocator, json_text, false);
    }

    /// 校验导出结果；违规时输出出错字段名
    fn expectConforms(allocator: std.mem.Allocator, json_text: []const u8) !void {
        return run(allocator, json_text, true);
    }
};

test "Model graph JSON export conforms to the published JSON Schema" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(24680);
    const random = prng.random();

    // 1. 多层 GPT：覆盖端口、缓冲区边、折叠变换与残差边
    {
        const config = transformer.GPTConfig{ .vocab_size = 64, .block_size = 8, .n_embd = 16, .n_head = 2, .n_layer = 2 };
        var gpt = try transformer.GPT(config).init(allocator, random);
        defer gpt.deinit(allocator);
        gpt.setName("gpt");

        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();
        var token_data: [2 * 4]f32 = undefined;
        for (&token_data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 50));
        const input_tokens = try graph.tensorNDWithData(&.{ 2, 4 }, &token_data, false);
        input_tokens.setName("inputs.token_ids");
        const logits = try gpt.forward(&graph, input_tokens);
        logits.setName("outputs.logits");

        const json_data = try graph.formatJson(allocator);
        defer allocator.free(json_data);
        try SchemaCheck.expectConforms(allocator, json_data);
    }

    // 2. 单个 Linear：最小模型
    {
        var linear = try core.Linear.init(allocator, 8, 4, random);
        defer linear.deinit(allocator);
        linear.setName("linear");

        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();
        var x_data: [2 * 8]f32 = undefined;
        for (&x_data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i)) * 0.1;
        const x = try graph.tensorNDWithData(&.{ 2, 8 }, &x_data, false);
        x.setName("inputs.x");
        const y = try linear.forward(&graph, x);
        y.setName("outputs.y");

        const json_data = try graph.formatJson(allocator);
        defer allocator.free(json_data);
        try SchemaCheck.expectConforms(allocator, json_data);
    }

    // 3. 校验器自身：未声明字段与错误版本均被拒绝
    try std.testing.expectError(error.SchemaConstMismatch, SchemaCheck.validate(allocator,
        \\{"version": "1.0", "summary": {}, "default_scope": "", "root": {}}
    ));
    try std.testing.expectError(error.SchemaUnknownField, SchemaCheck.validate(allocator,
        \\{"version": "2.0", "extra": 1, "summary": {"total_params": 0, "total_bytes": 0, "param_nodes": 0, "input_nodes": 0, "buffer_nodes": 0, "activation_nodes": 0, "custom_init_count": 0, "auto_graph_count": 0, "total_nodes": 0},
        \\ "default_scope": "", "root": {"name": "root", "path": "", "kind": "module", "module_type": "Model", "formula": null, "total_params": 0, "total_bytes": 0, "param_count": 0, "node_count": 0,
        \\ "children": [], "parameters": [], "ops": [], "ports": {"inputs": [], "outputs": []}, "flow_nodes": [], "edges": [], "nodes": []}}
    ));
}

test "All 18 canonical models export conforming schema and valid ports" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(13579);
    const random = prng.random();

    const verifyModelJson = struct {
        fn check(alloc: std.mem.Allocator, json_data: []const u8, model_name: []const u8) !void {
            // 1. Schema conformance
            try SchemaCheck.expectConforms(alloc, json_data);

            // 2. Parse and inspect
            var parsed = try std.json.parseFromSlice(std.json.Value, alloc, json_data, .{});
            defer parsed.deinit();

            const summary = parsed.value.object.get("summary").?.object;
            try std.testing.expectEqual(@as(i64, 0), summary.get("custom_init_count").?.integer);
            try std.testing.expectEqual(summary.get("param_nodes").?.integer, summary.get("auto_graph_count").?.integer);

            const root = parsed.value.object.get("root").?.object;
            const ports = root.get("ports").?.object;
            const in_ports = ports.get("inputs").?.array;
            const edges = root.get("edges").?.array;

            // At least one input port in root.ports.inputs
            try std.testing.expect(in_ports.items.len >= 1);

            // Every input port connected in root.edges
            for (in_ports.items) |inp_val| {
                const port_id = inp_val.object.get("id").?.string;
                var connected = false;
                for (edges.items) |edge_val| {
                    const from_id = edge_val.object.get("from").?.string;
                    if (std.mem.eql(u8, from_id, port_id)) {
                        connected = true;
                        break;
                    }
                }
                if (!connected) {
                    std.debug.print("Model {s}: port {s} not connected in root.edges\n", .{ model_name, port_id });
                    return error.PortNotConnected;
                }
            }

            // No graph input tensor has kind: "buffer" or status: "BUFFER"
            if (root.get("nodes")) |nodes_val| {
                for (nodes_val.array.items) |node_val| {
                    const node_obj = node_val.object;
                    const name = node_obj.get("name").?.string;
                    if (std.mem.startsWith(u8, name, "inputs.")) {
                        if (node_obj.get("kind")) |k| {
                            if (std.ascii.eqlIgnoreCase(k.string, "buffer")) return error.InputMarkedAsBuffer;
                        }
                        if (node_obj.get("status")) |s| {
                            if (std.ascii.eqlIgnoreCase(s.string, "buffer")) return error.InputMarkedAsBuffer;
                        }
                    }
                }
            }
        }
    };

    // 1. linear
    {
        var m = try Linear.init(allocator, 16, 10, random);
        defer m.deinit(allocator);
        m.setName("linear");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 2, 16 }, false);
        x.setName("inputs.x");
        const y = try m.forward(&g, x);
        y.setName("outputs.y");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "linear");
    }

    // 2. mlp
    {
        const TestMLP = struct {
            fc1: Linear,
            fc2: Linear,
            fc3: Linear,
            name: ?[]const u8 = "mlp",
            module_type: []const u8 = "MLP",

            pub fn init(alloc: std.mem.Allocator, rnd: std.Random) !@This() {
                const fc1 = try Linear.init(alloc, 16, 32, rnd);
                const fc2 = try Linear.init(alloc, 32, 16, rnd);
                const fc3 = try Linear.init(alloc, 16, 10, rnd);
                var mlp = @This(){ .fc1 = fc1, .fc2 = fc2, .fc3 = fc3 };
                mlp.setName("mlp");
                return mlp;
            }

            pub fn deinit(self: @This(), alloc: std.mem.Allocator) void {
                self.fc1.deinit(alloc);
                self.fc2.deinit(alloc);
                self.fc3.deinit(alloc);
            }

            pub fn setName(self: *@This(), name: []const u8) void {
                self.name = name;
                self.fc1.setName("mlp.fc1");
                self.fc2.setName("mlp.fc2");
                self.fc3.setName("mlp.fc3");
            }

            pub fn forward(self: *const @This(), g: *autodiff.Graph, x: *Tensor) !*Tensor {
                const scope = try g.enterModule(self.name, self.module_type);
                defer scope.exit();
                const x1 = try self.fc1.forward(g, x);
                const a1 = try g.relu(x1);
                const x2 = try self.fc2.forward(g, a1);
                const a2 = try g.relu(x2);
                return try self.fc3.forward(g, a2);
            }
        };

        var m = try TestMLP.init(allocator, random);
        defer m.deinit(allocator);
        m.setName("mlp");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 2, 16 }, false);
        x.setName("inputs.x");
        const y = try m.forward(&g, x);
        y.setName("outputs.logits");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "mlp");
    }

    // 3. rnn
    {
        var m = try RNN.init(allocator, 16, 32, random);
        defer m.deinit(allocator);
        m.setName("rnn");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        var inps: [4]*Tensor = undefined;
        for (0..4) |t| {
            inps[t] = try g.ones(&.{ 2, 16 }, false);
            inps[t].setNameFormatted("inputs.x_{d}", .{t});
        }
        const res = try m.forward(&g, &inps, null);
        res.h_n.setName("outputs.h_n");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "rnn");
    }

    // 4. lstm
    {
        var m = try LSTM.init(allocator, 16, 32, random);
        defer m.deinit(allocator);
        m.setName("lstm");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        var inps: [4]*Tensor = undefined;
        for (0..4) |t| {
            inps[t] = try g.ones(&.{ 2, 16 }, false);
            inps[t].setNameFormatted("inputs.x_{d}", .{t});
        }
        const res = try m.forward(&g, &inps, null, null);
        res.h_n.setName("outputs.h_n");
        res.c_n.setName("outputs.c_n");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "lstm");
    }

    // 5. stacked_lstm
    {
        var m = try StackedLSTM.init(allocator, 16, 32, 2, random);
        defer m.deinit(allocator);
        m.setName("stacked_lstm");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        var inps: [4]*Tensor = undefined;
        for (0..4) |t| {
            inps[t] = try g.ones(&.{ 2, 16 }, false);
            inps[t].setNameFormatted("inputs.x_{d}", .{t});
        }
        const res = try m.forwardSequence(&g, &inps, null, null);
        for (res.h_n, 0..) |h, l| h.setNameFormatted("outputs.h_l{d}", .{l});
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "stacked_lstm");
    }

    // 6. gru
    {
        var m = try GRU.init(allocator, 16, 32, random);
        defer m.deinit(allocator);
        m.setName("gru");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        var inps: [4]*Tensor = undefined;
        for (0..4) |t| {
            inps[t] = try g.ones(&.{ 2, 16 }, false);
            inps[t].setNameFormatted("inputs.x_{d}", .{t});
        }
        const res = try m.forward(&g, &inps, null);
        res.h_n.setName("outputs.h_n");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "gru");
    }

    // 7. embedding
    {
        var m = try Embedding.init(allocator, 128, 32, random);
        defer m.deinit(allocator);
        m.setName("embedding");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const tokens = try g.zeros(&.{ 2, 8 }, false);
        tokens.setName("inputs.token_ids");
        for (tokens.data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 16));
        const out = try m.forward(&g, tokens);
        out.setName("outputs.embeddings");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "embedding");
    }

    // 8. attention
    {
        var m = try CausalSelfAttention.init(allocator, 32, 4, random);
        defer m.deinit(allocator);
        m.setName("causal_self_attention");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 2, 4, 32 }, false);
        x.setName("inputs.x");
        const out = try m.forward(&g, x);
        out.setName("outputs.context");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "attention");
    }

    // 9. transformer_block
    {
        var m = try TransformerBlock.init(allocator, 32, 4, random);
        defer m.deinit(allocator);
        m.setName("transformer_block");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 2, 4, 32 }, false);
        x.setName("inputs.x");
        const out = try m.forward(&g, x);
        out.setName("outputs.block_out");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "transformer_block");
    }

    // 10. gpt
    {
        const cfg = GPTConfig{
            .vocab_size = 256,
            .block_size = 32,
            .n_embd = 64,
            .n_head = 4,
            .n_layer = 2,
        };
        var m = try GPT(cfg).init(allocator, random);
        defer m.deinit(allocator);
        m.setName("gpt");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const tokens = try g.zeros(&.{ 2, 16 }, false);
        tokens.setName("inputs.token_ids");
        for (tokens.data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 50));
        const logits = try m.forward(&g, tokens);
        logits.setName("outputs.logits");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "gpt");
    }

    // 11. swiglu
    {
        var m = try SwiGLU.init(allocator, 32, 64, random);
        defer m.deinit(allocator);
        m.setName("swiglu");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 2, 4, 32 }, false);
        x.setName("inputs.x");
        const out = try m.forward(&g, x);
        out.setName("outputs.swiglu_out");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "swiglu");
    }

    // 12. lora_linear
    {
        var m = try LoRALinear.init(allocator, 32, 32, 4, 8.0, random);
        defer m.deinit(allocator);
        m.setName("lora_linear");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 4, 32 }, false);
        x.setName("inputs.x");
        const out = try m.forward(&g, x);
        out.setName("outputs.adapted_out");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "lora_linear");
    }

    // 13. layernorm
    {
        var m = try LayerNorm.init(allocator, 32, 1e-5);
        defer m.deinit(allocator);
        m.setName("layernorm");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 2, 4, 32 }, false);
        x.setName("inputs.x");
        const out = try m.forward(&g, x);
        out.setName("outputs.normed");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "layernorm");
    }

    // 14. mla
    {
        var m = try MLALayer.init(allocator, 32, 4, 8, 16, 8, random);
        defer m.deinit(allocator);
        m.setName("mla");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 2, 4, 32 }, false);
        x.setName("inputs.x");
        const out = try m.forward(&g, x);
        out.setName("outputs.mla_out");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "mla");
    }

    // 15. deepseek_moe
    {
        var m = try MoELayer.init(allocator, 32, 64, 4, 1, 2, random);
        defer m.deinit(allocator);
        m.setName("deepseek_moe");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 4, 32 }, false);
        x.setName("inputs.x");
        const out = try m.forward(&g, x);
        out.setName("outputs.moe_out");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "deepseek_moe");
    }

    // 16. gan_generator
    {
        var l1 = try Linear.init(allocator, 2, 16, random);
        l1.setName("generator.fc1");
        var l2 = try Linear.init(allocator, 16, 16, random);
        l2.setName("generator.fc2");
        var l3 = try Linear.init(allocator, 16, 2, random);
        l3.setName("generator.fc3");
        var net_g = sequential(.{
            l1,
            LeakyReLU{ .alpha = 0.2 },
            l2,
            LeakyReLU{ .alpha = 0.2 },
            l3,
        });
        defer net_g.deinit(allocator);
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const z = try g.ones(&.{ 4, 2 }, false);
        z.setName("inputs.z");
        const scope = try g.enterModule("generator", "Generator");
        defer scope.exit();
        const fake_x = try net_g.forward(&g, z);
        fake_x.setName("outputs.fake_x");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "gan_generator");
    }

    // 17. gan_discriminator
    {
        var d1 = try Linear.init(allocator, 2, 16, random);
        d1.setName("discriminator.fc1");
        var d2 = try Linear.init(allocator, 16, 16, random);
        d2.setName("discriminator.fc2");
        var d3 = try Linear.init(allocator, 16, 1, random);
        d3.setName("discriminator.fc3");
        var net_d = sequential(.{
            d1,
            LeakyReLU{ .alpha = 0.2 },
            d2,
            LeakyReLU{ .alpha = 0.2 },
            d3,
        });
        defer net_d.deinit(allocator);
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 4, 2 }, false);
        x.setName("inputs.x");
        const scope = try g.enterModule("discriminator", "Discriminator");
        defer scope.exit();
        const logits = try net_d.forward(&g, x);
        logits.setName("outputs.logits");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "gan_discriminator");
    }

    // 18. conv2d
    {
        var m = try Conv2D.init(allocator, 1, 4, 3, random);
        defer m.deinit(allocator);
        m.setName("conv2d");
        var g = autodiff.Graph.init(allocator);
        defer g.deinit();
        const x = try g.ones(&.{ 1, 1, 8, 8 }, false);
        x.setName("inputs.x");
        const out = try m.forward(&g, x);
        out.setName("outputs.feature_map");
        const json = try g.formatJson(allocator);
        defer allocator.free(json);
        try verifyModelJson.check(allocator, json, "conv2d");
    }
}

test "NodeKind enum conversions and NodeData typing" {
    try std.testing.expectEqualStrings("Param", NodeKind.Param.asString());
    try std.testing.expectEqualStrings("Input", NodeKind.Input.asString());
    try std.testing.expectEqualStrings("Buffer", NodeKind.Buffer.asString());
    try std.testing.expectEqualStrings("Activation", NodeKind.Activation.asString());

    try std.testing.expectEqual(NodeKind.Param, NodeKind.fromString("Param"));
    try std.testing.expectEqual(NodeKind.Input, NodeKind.fromString("Input"));
    try std.testing.expectEqual(NodeKind.Buffer, NodeKind.fromString("Buffer"));
    try std.testing.expectEqual(NodeKind.Activation, NodeKind.fromString("Activation"));

    try std.testing.expectEqual(NodeKind.Param, NodeKind.fromString("param"));
    try std.testing.expectEqual(NodeKind.Input, NodeKind.fromString("input"));
    try std.testing.expectEqual(NodeKind.Buffer, NodeKind.fromString("buffer"));
    try std.testing.expectEqual(NodeKind.Activation, NodeKind.fromString("activation"));

    try std.testing.expect(NodeKind.fromString("Unknown") == null);
}

test "Visualization enums match the enum lists of the published JSON Schema" {
    const allocator = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, visualization.SCHEMA_JSON, .{});
    defer parsed.deinit();
    const defs = parsed.value.object.get("$defs").?.object;

    const Check = struct {
        fn prop(d: std.json.ObjectMap, def: []const u8, field: []const u8) std.json.ObjectMap {
            return d.get(def).?.object.get("properties").?.object.get(field).?.object;
        }

        /// schema 的 enum 列表与 Zig 枚举的标签逐一对应 (顺序一致)
        fn same(comptime E: type, values: []const std.json.Value) !void {
            const fields = @typeInfo(E).@"enum".fields;
            try std.testing.expectEqual(fields.len, values.len);
            inline for (fields, 0..) |f, i| {
                try std.testing.expectEqualStrings(f.name, values[i].string);
            }
        }
    };

    try Check.same(NodeKind, Check.prop(defs, "TensorNode", "kind").get("enum").?.array.items);
    try Check.same(NodeStatus, Check.prop(defs, "TensorNode", "status").get("enum").?.array.items);
    try Check.same(FlowNodeKind, Check.prop(defs, "FlowNode", "kind").get("enum").?.array.items);
    try Check.same(EdgeKind, Check.prop(defs, "Edge", "kind").get("enum").?.array.items);

    // 参数只可能是 CUSTOM_INIT / AUTO_GRAPH
    const param_status = Check.prop(defs, "ParamEntry", "status").get("enum").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), param_status.len);
    try std.testing.expectEqualStrings(NodeStatus.CUSTOM_INIT.asString(), param_status[0].string);
    try std.testing.expectEqualStrings(NodeStatus.AUTO_GRAPH.asString(), param_status[1].string);

    try std.testing.expectEqualStrings("module", Check.prop(defs, "ModuleNode", "kind").get("const").?.string);
}
