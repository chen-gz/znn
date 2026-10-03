const std = @import("std");
const zig_ml = @import("zig_ml");
const nn = zig_ml.nn;
const autodiff = zig_ml.autodiff;
const tensor = zig_ml.tensor;

const Tensor = tensor.Tensor;

const ThreeLayerMLP = struct {
    fc1: nn.Linear,
    fc2: nn.Linear,
    fc3: nn.Linear,
    name: ?[]const u8 = null,
    module_type: []const u8 = "MLP",

    pub const formula = "h_1 = \\text{ReLU}(x W_1^T + b_1), \\; h_2 = \\text{ReLU}(h_1 W_2^T + b_2), \\; y = h_2 W_3^T + b_3";

    /// 只分配各层参数内存；参数数值在建立前向计算图后由 nn.initModel 依据计算图初始化
    pub fn init(allocator: std.mem.Allocator) !ThreeLayerMLP {
        const fc1 = try nn.Linear.init(allocator, 16, 32);
        errdefer nn.deinitModel(&fc1, allocator);
        const fc2 = try nn.Linear.init(allocator, 32, 16);
        errdefer nn.deinitModel(&fc2, allocator);
        const fc3 = try nn.Linear.init(allocator, 16, 10);
        return .{ .fc1 = fc1, .fc2 = fc2, .fc3 = fc3 };
    }

    pub fn forward(self: *const ThreeLayerMLP, g: *autodiff.Graph, x: *Tensor) !*Tensor {
        const scope = try nn.enterModuleScope(g, self);
        defer scope.exit();
        const x1 = try self.fc1.forward(g, x);
        const a1 = try g.relu(x1);
        const x2 = try self.fc2.forward(g, a1);
        const a2 = try g.relu(x2);
        return try self.fc3.forward(g, a2);
    }
};

fn exportModel(
    allocator: std.mem.Allocator,
    name: []const u8,
    buildAndExportFn: *const fn (allocator: std.mem.Allocator, random: std.Random, graph: *autodiff.Graph, file_path: []const u8) anyerror!void,
    random: std.Random,
    output_dir: []const u8,
) !void {
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    var path_buf: [256]u8 = undefined;
    const file_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}.json", .{ output_dir, name });
    // 计算图引用模型参数张量，因此必须在模型释放之前 (即构建函数内部) 完成导出
    try buildAndExportFn(allocator, random, &graph, file_path);
    std.debug.print("  [✓] Exported {s} -> {s}\n", .{ name, file_path });
}

/// Usage: `zig build run-book-models [-- <output_dir>...]`
/// Without arguments the graphs are written to `examples/models`.
pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();
    _ = args.next(); // executable path

    var target_dirs: std.ArrayList([]const u8) = .empty;
    defer target_dirs.deinit(allocator);
    while (args.next()) |arg| try target_dirs.append(allocator, arg);
    if (target_dirs.items.len == 0) try target_dirs.append(allocator, "examples/models");

    std.debug.print("\n============================================================\n", .{});
    std.debug.print("       ZNN - Exporting All Book Models to JSON Graphs        \n", .{});
    std.debug.print("============================================================\n\n", .{});

    for (target_dirs.items) |dir| {
        std.debug.print("Writing graphs to target directory: {s}\n", .{dir});

        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();

        // 1. linear
        try exportModel(allocator, "linear", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var linear = try nn.Linear.init(alloc, 16, 8);
                defer nn.deinitModel(&linear, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&linear, names.allocator(), "linear");

                const x = try g.ones(&.{ 2, 16 }, false);
                x.setName("inputs.x");
                const y = try linear.forward(g, x);
                y.setName("outputs.y");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&linear, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 2. mlp
        try exportModel(allocator, "mlp", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var model = try ThreeLayerMLP.init(alloc);
                defer nn.deinitModel(&model, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&model, names.allocator(), "mlp");

                const x = try g.ones(&.{ 2, 16 }, false);
                x.setName("inputs.x");
                const y = try model.forward(g, x);
                y.setName("outputs.logits");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&model, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 3. rnn
        try exportModel(allocator, "rnn", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var rnn = try nn.RNN.init(alloc, 16, 32);
                defer nn.deinitModel(&rnn, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&rnn, names.allocator(), "rnn");

                var inputs: [4]*Tensor = undefined;
                for (0..4) |t| {
                    inputs[t] = try g.ones(&.{ 2, 16 }, false);
                    inputs[t].setNameFormatted("inputs.x_{d}", .{t});
                }
                const res = try rnn.forward(g, &inputs, null);
                res.h_n.setName("outputs.h_n");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&rnn, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 4. lstm
        try exportModel(allocator, "lstm", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var lstm = try nn.LSTM.init(alloc, 16, 32);
                defer nn.deinitModel(&lstm, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&lstm, names.allocator(), "lstm");

                var inputs: [4]*Tensor = undefined;
                for (0..4) |t| {
                    inputs[t] = try g.ones(&.{ 2, 16 }, false);
                    inputs[t].setNameFormatted("inputs.x_{d}", .{t});
                }
                const res = try lstm.forward(g, &inputs, null, null);
                res.h_n.setName("outputs.h_n");
                res.c_n.setName("outputs.c_n");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&lstm, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 5. stacked_lstm
        try exportModel(allocator, "stacked_lstm", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var slstm = try nn.StackedLSTM.init(alloc, 16, 32, 2);
                defer nn.deinitModel(&slstm, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&slstm, names.allocator(), "stacked_lstm");

                var inputs: [4]*Tensor = undefined;
                for (0..4) |t| {
                    inputs[t] = try g.ones(&.{ 2, 16 }, false);
                    inputs[t].setNameFormatted("inputs.x_{d}", .{t});
                }
                const res = try slstm.forwardSequence(g, &inputs, null, null);
                for (res.h_n, 0..) |h, l| {
                    h.setNameFormatted("outputs.h_l{d}", .{l});
                }
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&slstm, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 6. gru
        try exportModel(allocator, "gru", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var gru = try nn.GRU.init(alloc, 16, 32);
                defer nn.deinitModel(&gru, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&gru, names.allocator(), "gru");

                var inputs: [4]*Tensor = undefined;
                for (0..4) |t| {
                    inputs[t] = try g.ones(&.{ 2, 16 }, false);
                    inputs[t].setNameFormatted("inputs.x_{d}", .{t});
                }
                const res = try gru.forward(g, &inputs, null);
                res.h_n.setName("outputs.h_n");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&gru, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 7. embedding
        try exportModel(allocator, "embedding", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var emb = try nn.Embedding.init(alloc, 128, 32);
                defer nn.deinitModel(&emb, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&emb, names.allocator(), "embedding");

                const tokens = try g.zeros(&.{ 2, 8 }, false);
                tokens.setName("inputs.token_ids");
                for (tokens.data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 16));
                const out = try emb.forward(g, tokens);
                out.setName("outputs.embeddings");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&emb, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 8. attention
        try exportModel(allocator, "attention", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var attn = try nn.CausalSelfAttention.init(alloc, 32, 4);
                defer nn.deinitModel(&attn, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&attn, names.allocator(), "causal_self_attention");

                const x = try g.ones(&.{ 2, 4, 32 }, false);
                x.setName("inputs.x");
                const out = try attn.forward(g, x);
                out.setName("outputs.context");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&attn, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 9. transformer_block
        try exportModel(allocator, "transformer_block", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var block = try nn.TransformerBlock.init(alloc, 32, 4);
                defer nn.deinitModel(&block, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&block, names.allocator(), "transformer_block");

                const x = try g.ones(&.{ 2, 4, 32 }, false);
                x.setName("inputs.x");
                const out = try block.forward(g, x);
                out.setName("outputs.block_out");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&block, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 10. gpt
        try exportModel(allocator, "gpt", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                const cfg = nn.GPTConfig{
                    .vocab_size = 256,
                    .block_size = 32,
                    .n_embd = 64,
                    .n_head = 4,
                    .n_layer = 2,
                };
                var gpt = try nn.GPT(cfg).init(alloc);
                defer nn.deinitModel(&gpt, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&gpt, names.allocator(), "gpt");

                const tokens = try g.zeros(&.{ 2, 16 }, false);
                tokens.setName("inputs.token_ids");
                for (tokens.data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 50));
                const logits = try gpt.forward(g, tokens);
                logits.setName("outputs.logits");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&gpt, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 11. swiglu
        try exportModel(allocator, "swiglu", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var swiglu = try nn.SwiGLU.init(alloc, 32, 64);
                defer nn.deinitModel(&swiglu, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&swiglu, names.allocator(), "swiglu");

                const x = try g.ones(&.{ 2, 4, 32 }, false);
                x.setName("inputs.x");
                const out = try swiglu.forward(g, x);
                out.setName("outputs.swiglu_out");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&swiglu, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 12. lora_linear
        try exportModel(allocator, "lora_linear", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var lora = try nn.LoRALinear.init(alloc, 32, 32, 4, 8.0);
                defer nn.deinitModel(&lora, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&lora, names.allocator(), "lora_linear");

                const x = try g.ones(&.{ 4, 32 }, false);
                x.setName("inputs.x");
                const out = try lora.forward(g, x);
                out.setName("outputs.adapted_out");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&lora, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 13. layernorm
        try exportModel(allocator, "layernorm", struct {
            fn run(alloc: std.mem.Allocator, _: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var ln = try nn.LayerNorm.init(alloc, 32, 1e-5);
                defer nn.deinitModel(&ln, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&ln, names.allocator(), "layernorm");

                const x = try g.ones(&.{ 2, 4, 32 }, false);
                x.setName("inputs.x");
                const out = try ln.forward(g, x);
                out.setName("outputs.normed");
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 14. mla
        try exportModel(allocator, "mla", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var mla = try nn.MLALayer.init(alloc, 32, 4, 8, 16, 8);
                defer nn.deinitModel(&mla, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&mla, names.allocator(), "mla");

                const x = try g.ones(&.{ 2, 4, 32 }, false);
                x.setName("inputs.x");
                const out = try mla.forward(g, x);
                out.setName("outputs.mla_out");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&mla, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 15. deepseek_moe
        try exportModel(allocator, "deepseek_moe", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var moe = try nn.MoELayer.init(alloc, 32, 64, 4, 1, 2);
                defer nn.deinitModel(&moe, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&moe, names.allocator(), "deepseek_moe");

                const x = try g.ones(&.{ 4, 32 }, false);
                x.setName("inputs.x");
                const out = try moe.forward(g, x);
                out.setName("outputs.moe_out");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&moe, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 16. gan_generator
        try exportModel(allocator, "gan_generator", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var l1 = try nn.Linear.init(alloc, 2, 16);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&l1, names.allocator(), "generator.fc1");
                var l2 = try nn.Linear.init(alloc, 16, 16);
                try nn.nameModules(&l2, names.allocator(), "generator.fc2");
                var l3 = try nn.Linear.init(alloc, 16, 2);
                try nn.nameModules(&l3, names.allocator(), "generator.fc3");

                var net_g = nn.sequential(.{
                    l1,
                    nn.LeakyReLU{ .alpha = 0.2 },
                    l2,
                    nn.LeakyReLU{ .alpha = 0.2 },
                    l3,
                });
                defer nn.deinitModel(&net_g, alloc);

                const z = try g.ones(&.{ 4, 2 }, false);
                z.setName("inputs.z");
                const scope = try g.enterModule("generator", "Generator");
                defer scope.exit();
                const fake_x = try net_g.forward(g, z);
                fake_x.setName("outputs.fake_x");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&net_g, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 17. gan_discriminator
        try exportModel(allocator, "gan_discriminator", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var d1 = try nn.Linear.init(alloc, 2, 16);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&d1, names.allocator(), "discriminator.fc1");
                var d2 = try nn.Linear.init(alloc, 16, 16);
                try nn.nameModules(&d2, names.allocator(), "discriminator.fc2");
                var d3 = try nn.Linear.init(alloc, 16, 1);
                try nn.nameModules(&d3, names.allocator(), "discriminator.fc3");

                var net_d = nn.sequential(.{
                    d1,
                    nn.LeakyReLU{ .alpha = 0.2 },
                    d2,
                    nn.LeakyReLU{ .alpha = 0.2 },
                    d3,
                });
                defer nn.deinitModel(&net_d, alloc);

                const x = try g.ones(&.{ 4, 2 }, false);
                x.setName("inputs.x");
                const scope = try g.enterModule("discriminator", "Discriminator");
                defer scope.exit();
                const logits = try net_d.forward(g, x);
                logits.setName("outputs.logits");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&net_d, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        // 18. conv2d
        try exportModel(allocator, "conv2d", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph, file_path: []const u8) !void {
                var conv = try nn.Conv2D.init(alloc, 1, 4, 3, .{});
                defer nn.deinitModel(&conv, alloc);
                var names = std.heap.ArenaAllocator.init(alloc);
                defer names.deinit();
                try nn.nameModules(&conv, names.allocator(), "conv2d");

                const x = try g.ones(&.{ 1, 1, 8, 8 }, false);
                x.setName("inputs.x");
                const out = try conv.forward(g, x);
                out.setName("outputs.feature_map");
                // 前向计算图建立后，依据图中各参数的下游激活函数初始化参数
                try nn.initModel(&conv, g, rnd);
                try g.exportJson(file_path);
            }
        }.run, random, dir);

        std.debug.print("\n", .{});
    }

    std.debug.print("All 18 canonical models exported successfully!\n", .{});
}
