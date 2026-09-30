const std = @import("std");
const zig_ml = @import("zig_ml");
const nn = zig_ml.nn;
const autodiff = zig_ml.autodiff;
const tensor = zig_ml.tensor;

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    std.debug.print("\n=================================================================\n", .{});
    std.debug.print("  ZNN - Interactive Model Architecture & Graph HTML Export Demo  \n", .{});
    std.debug.print("=================================================================\n\n", .{});

    // 1. 配置标准多层 GPT 模型结构 (2 层 Transformer Decoder)
    const config = nn.GPTConfig{
        .vocab_size = 256,
        .block_size = 32,
        .n_embd = 64,
        .n_head = 4,
        .n_layer = 2,
    };

    std.debug.print("[Step 1/4] Initializing GPT Model...\n", .{});
    std.debug.print("  * Vocabulary Size: {d}\n", .{config.vocab_size});
    std.debug.print("  * Context Window:  {d}\n", .{config.block_size});
    std.debug.print("  * Embedding Dim:   {d}\n", .{config.n_embd});
    std.debug.print("  * Attention Heads: {d}\n", .{config.n_head});
    std.debug.print("  * Decoder Layers:  {d}\n\n", .{config.n_layer});

    var gpt = try nn.GPT(config).init(allocator, random);
    defer gpt.deinit(allocator);

    // 2. 赋予顶级层次化模块命名 (自动递归设置至所有子模块与参数)
    gpt.setName("gpt");

    // 3. 构建计算图并执行前向推理
    std.debug.print("[Step 2/4] Constructing Autodiff Graph and Running Forward Pass...\n", .{});
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    // 构造批次输入张量 [batch_size=2, seq_len=16]
    const batch_size: usize = 2;
    const seq_len: usize = 16;
    const input_data = try allocator.alloc(f32, batch_size * seq_len);
    defer allocator.free(input_data);
    for (input_data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i % 50));
    }
    const input_tokens = try graph.tensorNDWithData(&.{ batch_size, seq_len }, input_data, false);
    input_tokens.setName("inputs.token_ids");

    // 前向传播
    const logits = try gpt.forward(allocator, &graph, input_tokens);
    logits.setName("outputs.logits");


    std.debug.print("  * Input shape:       [{d}, {d}]\n", .{ batch_size, seq_len });
    std.debug.print("  * Output shape:      [{d}, {d}, {d}]\n", .{ logits.shape.dims[0], logits.shape.dims[1], logits.shape.dims[2] });
    std.debug.print("  * Graph Total Nodes: {d}\n\n", .{graph.tensors.items.len});

    // 4. 导出计算图为独立的 JSON 文件
    const output_path = "examples/sample_model_graph.json";
    std.debug.print("[Step 3/4] Exporting Model Hierarchy & Graph JSON...\n", .{});
    try graph.exportJson(output_path);
    std.debug.print("  * Graph written to: {s}\n\n", .{output_path});

    // 5. 导出单个 Linear 层的最小参考 JSON (可视化器 JSON 格式指南中的模板)
    std.debug.print("[Step 4/4] Exporting minimal single-Linear reference JSON...\n", .{});
    var linear = try nn.Linear.init(allocator, 8, 4, random);
    defer linear.deinit(allocator);
    linear.setName("linear");

    var min_graph = autodiff.Graph.init(allocator);
    defer min_graph.deinit();
    var x_data: [2 * 8]f32 = undefined;
    for (&x_data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i)) * 0.1;
    const x = try min_graph.tensorNDWithData(&.{ 2, 8 }, &x_data, false);
    x.setName("inputs.x");
    const y = try linear.forward(allocator, &min_graph, x);
    y.setName("outputs.y");

    const minimal_path = "examples/minimal_model_graph.json";
    try min_graph.exportJson(minimal_path);
    std.debug.print("  * Graph written to: {s}\n\n", .{minimal_path});
}

