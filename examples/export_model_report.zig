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

    // 显式在代码中为各层模块配置向量变换数学公式 (直接写入代码，界面将直接精准展示)
    try graph.setModuleFormula("gpt.wte", "y = TokenEmbedding(ids; W_e \\in \\mathbb{R}^{256 \\times 64}) \\rightarrow [2, 16, 64]");
    try graph.setModuleFormula("gpt.wpe", "y = PosEmbedding(pos; W_p \\in \\mathbb{R}^{32 \\times 64}) \\rightarrow [2, 16, 64]");
    try graph.setModuleFormula("gpt.layers.0.ln_1", "y = RMSNorm(x; \\gamma \\in \\mathbb{R}^{64}, \\epsilon=1e-5) \\rightarrow [2, 16, 64]");
    try graph.setModuleFormula("gpt.layers.0.attn", "A = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V \\cdot W_o^T + b_o \\rightarrow [2, 16, 64]");
    try graph.setModuleFormula("gpt.layers.0.attn.q_attn", "Q = x_{2D} W_q^T + b_q \\quad ([32, 64] \\times [64, 64]^T \\rightarrow [2, 4, 16, 16])");
    try graph.setModuleFormula("gpt.layers.0.attn.k_attn", "K = x_{2D} W_k^T + b_k \\quad ([32, 64] \\times [64, 64]^T \\rightarrow [2, 4, 16, 16])");
    try graph.setModuleFormula("gpt.layers.0.attn.v_attn", "V = x_{2D} W_v^T + b_v \\quad ([32, 64] \\times [64, 64]^T \\rightarrow [2, 4, 16, 16])");
    try graph.setModuleFormula("gpt.layers.0.attn.c_proj", "y = (\\text{HeadsConcat}) \\cdot W_{proj}^T + b_{proj} \\rightarrow [2, 16, 64]");
    try graph.setModuleFormula("gpt.layers.0.ln_2", "y = RMSNorm(x_1; \\gamma \\in \\mathbb{R}^{64}, \\epsilon=1e-5) \\rightarrow [2, 16, 64]");
    try graph.setModuleFormula("gpt.layers.0.mlp", "y = \\text{GELU}(x W_{fc}^T + b_{fc}) W_{proj}^T + b_{proj} \\rightarrow [2, 16, 64]");
    try graph.setModuleFormula("gpt.layers.ln_f", "y = RMSNorm(x_{last}; \\gamma \\in \\mathbb{R}^{64}) \\rightarrow [2, 16, 64]");
    try graph.setModuleFormula("gpt.lm_head", "\\text{logits} = x_{norm} \\cdot W_{head}^T \\quad ([2, 16, 64] \\times [256, 64]^T \\rightarrow [2, 16, 256])");

    std.debug.print("  * Input shape:       [{d}, {d}]\n", .{ batch_size, seq_len });
    std.debug.print("  * Output shape:      [{d}, {d}, {d}]\n", .{ logits.shape.dims[0], logits.shape.dims[1], logits.shape.dims[2] });
    std.debug.print("  * Graph Total Nodes: {d}\n\n", .{graph.tensors.items.len});

    // 4. 导出交互式、树形可折叠的 HTML 架构报告
    const output_path = "examples/gpt_model_report.html";
    std.debug.print("[Step 3/4] Exporting Interactive HTML Report...\n", .{});
    try graph.exportHtmlReport(output_path);

    std.debug.print("[Step 4/4] Done! Report written to: {s}\n\n", .{output_path});
    std.debug.print("-----------------------------------------------------------------\n", .{});
    std.debug.print("To view the interactive model report in your browser, run:\n", .{});
    std.debug.print("  open {s}\n", .{output_path});
    std.debug.print("-----------------------------------------------------------------\n\n", .{});
}
