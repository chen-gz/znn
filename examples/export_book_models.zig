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
    name: ?[]const u8 = "mlp",
    module_type: []const u8 = "MLP",

    pub fn init(allocator: std.mem.Allocator, random: std.Random) !ThreeLayerMLP {
        var fc1 = try nn.Linear.init(allocator, 16, 32, random);
        var fc2 = try nn.Linear.init(allocator, 32, 16, random);
        var fc3 = try nn.Linear.init(allocator, 16, 10, random);
        fc1.setName("mlp.fc1");
        fc2.setName("mlp.fc2");
        fc3.setName("mlp.fc3");
        return .{ .fc1 = fc1, .fc2 = fc2, .fc3 = fc3 };
    }

    pub fn deinit(self: ThreeLayerMLP, allocator: std.mem.Allocator) void {
        self.fc1.deinit(allocator);
        self.fc2.deinit(allocator);
        self.fc3.deinit(allocator);
    }

    pub fn forward(self: *const ThreeLayerMLP, g: *autodiff.Graph, x: *Tensor) !*Tensor {
        const scope = try g.enterModule(self.name, self.module_type);
        defer scope.exit();
        try g.setModuleFormula("mlp", "h_1 = \\text{ReLU}(x W_1^T + b_1), \\; h_2 = \\text{ReLU}(h_1 W_2^T + b_2), \\; y = h_2 W_3^T + b_3");
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
    buildAndForwardFn: *const fn (allocator: std.mem.Allocator, random: std.Random, graph: *autodiff.Graph) anyerror!void,
    random: std.Random,
    output_dir: []const u8,
) !void {
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    try buildAndForwardFn(allocator, random, &graph);

    var path_buf: [256]u8 = undefined;
    const file_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}.json", .{ output_dir, name });
    try graph.exportJson(file_path);
    std.debug.print("  [✓] Exported {s} -> {s}\n", .{ name, file_path });
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    std.debug.print("\n============================================================\n", .{});
    std.debug.print("       ZNN - Exporting All Book Models to JSON Graphs        \n", .{});
    std.debug.print("============================================================\n\n", .{});

    const target_dirs = [_][]const u8{
        "examples/models",
        "/Users/guangzong/Documents/chen-gz.github.io/public/tools/visualizer/models",
    };

    for (target_dirs) |dir| {
        std.debug.print("Writing graphs to target directory: {s}\n", .{dir});

        // 1. linear
        try exportModel(allocator, "linear", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var linear = try nn.Linear.init(alloc, 16, 8, rnd);
                defer linear.deinit(alloc);
                linear.setName("linear");

                const x = try g.ones(&.{ 2, 16 }, false);
                x.setName("inputs.x");
                const y = try linear.forward(g, x);
                y.setName("outputs.y");
            }
        }.run, random, dir);

        // 2. mlp
        try exportModel(allocator, "mlp", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var model = try ThreeLayerMLP.init(alloc, rnd);
                defer model.deinit(alloc);

                const x = try g.ones(&.{ 2, 16 }, false);
                x.setName("inputs.x");
                const y = try model.forward(g, x);
                y.setName("outputs.logits");
            }
        }.run, random, dir);

        // 3. rnn
        try exportModel(allocator, "rnn", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var rnn = try nn.RNN.init(alloc, 16, 32, rnd);
                defer rnn.deinit(alloc);
                rnn.setName("rnn");

                var inputs: [4]*Tensor = undefined;
                for (0..4) |t| {
                    inputs[t] = try g.ones(&.{ 2, 16 }, false);
                    inputs[t].setNameFormatted("inputs.x_{d}", .{t});
                }
                const res = try rnn.forward(g, &inputs, null);
                res.h_n.setName("outputs.h_n");
            }
        }.run, random, dir);

        // 4. lstm
        try exportModel(allocator, "lstm", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var lstm = try nn.LSTM.init(alloc, 16, 32, rnd);
                defer lstm.deinit(alloc);
                lstm.setName("lstm");

                var inputs: [4]*Tensor = undefined;
                for (0..4) |t| {
                    inputs[t] = try g.ones(&.{ 2, 16 }, false);
                    inputs[t].setNameFormatted("inputs.x_{d}", .{t});
                }
                const res = try lstm.forward(g, &inputs, null, null);
                res.h_n.setName("outputs.h_n");
                res.c_n.setName("outputs.c_n");
            }
        }.run, random, dir);

        // 5. stacked_lstm
        try exportModel(allocator, "stacked_lstm", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var slstm = try nn.StackedLSTM.init(alloc, 16, 32, 2, rnd);
                defer slstm.deinit(alloc);
                slstm.setName("stacked_lstm");

                var inputs: [4]*Tensor = undefined;
                for (0..4) |t| {
                    inputs[t] = try g.ones(&.{ 2, 16 }, false);
                    inputs[t].setNameFormatted("inputs.x_{d}", .{t});
                }
                const res = try slstm.forwardSequence(g, &inputs, null, null);
                for (res.h_n, 0..) |h, l| {
                    h.setNameFormatted("outputs.h_l{d}", .{l});
                }
            }
        }.run, random, dir);

        // 6. gru
        try exportModel(allocator, "gru", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var gru = try nn.GRU.init(alloc, 16, 32, rnd);
                defer gru.deinit(alloc);
                gru.setName("gru");

                var inputs: [4]*Tensor = undefined;
                for (0..4) |t| {
                    inputs[t] = try g.ones(&.{ 2, 16 }, false);
                    inputs[t].setNameFormatted("inputs.x_{d}", .{t});
                }
                const res = try gru.forward(g, &inputs, null);
                res.h_n.setName("outputs.h_n");
            }
        }.run, random, dir);

        // 7. embedding
        try exportModel(allocator, "embedding", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var emb = try nn.Embedding.init(alloc, 128, 32, rnd);
                defer emb.deinit(alloc);
                emb.setName("embedding");

                const tokens = try g.zeros(&.{ 2, 8 }, false);
                tokens.setName("inputs.token_ids");
                for (tokens.data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 16));
                const out = try emb.forward(g, tokens);
                out.setName("outputs.embeddings");
            }
        }.run, random, dir);

        // 8. attention
        try exportModel(allocator, "attention", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var attn = try nn.CausalSelfAttention.init(alloc, 32, 4, rnd);
                defer attn.deinit(alloc);
                attn.setName("causal_self_attention");

                const x = try g.ones(&.{ 2, 4, 32 }, false);
                x.setName("inputs.x");
                const out = try attn.forward(g, x);
                out.setName("outputs.context");
            }
        }.run, random, dir);

        // 9. transformer_block
        try exportModel(allocator, "transformer_block", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var block = try nn.TransformerBlock.init(alloc, 32, 4, rnd);
                defer block.deinit(alloc);
                block.setName("transformer_block");

                const x = try g.ones(&.{ 2, 4, 32 }, false);
                x.setName("inputs.x");
                const out = try block.forward(g, x);
                out.setName("outputs.block_out");
            }
        }.run, random, dir);

        // 10. gpt
        try exportModel(allocator, "gpt", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                const cfg = nn.GPTConfig{
                    .vocab_size = 256,
                    .block_size = 32,
                    .n_embd = 64,
                    .n_head = 4,
                    .n_layer = 2,
                };
                var gpt = try nn.GPT(cfg).init(alloc, rnd);
                defer gpt.deinit(alloc);
                gpt.setName("gpt");

                const tokens = try g.zeros(&.{ 2, 16 }, false);
                tokens.setName("inputs.token_ids");
                for (tokens.data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % 50));
                const logits = try gpt.forward(g, tokens);
                logits.setName("outputs.logits");
            }
        }.run, random, dir);

        // 11. swiglu
        try exportModel(allocator, "swiglu", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var swiglu = try nn.SwiGLU.init(alloc, 32, 64, rnd);
                defer swiglu.deinit(alloc);
                swiglu.setName("swiglu");

                const x = try g.ones(&.{ 2, 4, 32 }, false);
                x.setName("inputs.x");
                const out = try swiglu.forward(g, x);
                out.setName("outputs.swiglu_out");
            }
        }.run, random, dir);

        // 12. lora_linear
        try exportModel(allocator, "lora_linear", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var lora = try nn.LoRALinear.init(alloc, 32, 32, 4, 8.0, rnd);
                defer lora.deinit(alloc);
                lora.setName("lora_linear");

                const x = try g.ones(&.{ 4, 32 }, false);
                x.setName("inputs.x");
                const out = try lora.forward(g, x);
                out.setName("outputs.adapted_out");
            }
        }.run, random, dir);

        // 13. layernorm
        try exportModel(allocator, "layernorm", struct {
            fn run(alloc: std.mem.Allocator, _: std.Random, g: *autodiff.Graph) !void {
                var ln = try nn.LayerNorm.init(alloc, 32, 1e-5);
                defer ln.deinit(alloc);
                ln.setName("layernorm");

                const x = try g.ones(&.{ 2, 4, 32 }, false);
                x.setName("inputs.x");
                const out = try ln.forward(g, x);
                out.setName("outputs.normed");
            }
        }.run, random, dir);

        // 14. mla
        try exportModel(allocator, "mla", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var mla = try nn.MLALayer.init(alloc, 32, 4, 8, 16, 8, rnd);
                defer mla.deinit(alloc);
                mla.setName("mla");

                const x = try g.ones(&.{ 2, 4, 32 }, false);
                x.setName("inputs.x");
                const out = try mla.forward(g, x);
                out.setName("outputs.mla_out");
            }
        }.run, random, dir);

        // 15. deepseek_moe
        try exportModel(allocator, "deepseek_moe", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var moe = try nn.MoELayer.init(alloc, 32, 64, 4, 1, 2, rnd);
                defer moe.deinit(alloc);
                moe.setName("deepseek_moe");

                const x = try g.ones(&.{ 4, 32 }, false);
                x.setName("inputs.x");
                const out = try moe.forward(g, x);
                out.setName("outputs.moe_out");
            }
        }.run, random, dir);

        // 16. gan_generator
        try exportModel(allocator, "gan_generator", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var l1 = try nn.Linear.init(alloc, 2, 16, rnd);
                l1.setName("generator.fc1");
                var l2 = try nn.Linear.init(alloc, 16, 16, rnd);
                l2.setName("generator.fc2");
                var l3 = try nn.Linear.init(alloc, 16, 2, rnd);
                l3.setName("generator.fc3");

                var net_g = nn.sequential(.{
                    l1,
                    nn.LeakyReLU{ .alpha = 0.2 },
                    l2,
                    nn.LeakyReLU{ .alpha = 0.2 },
                    l3,
                });
                defer net_g.deinit(alloc);

                const z = try g.ones(&.{ 4, 2 }, false);
                z.setName("inputs.z");
                const fake_x = try net_g.forward(g, z);
                fake_x.setName("outputs.fake_x");
            }
        }.run, random, dir);

        // 17. gan_discriminator
        try exportModel(allocator, "gan_discriminator", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var d1 = try nn.Linear.init(alloc, 2, 16, rnd);
                d1.setName("discriminator.fc1");
                var d2 = try nn.Linear.init(alloc, 16, 16, rnd);
                d2.setName("discriminator.fc2");
                var d3 = try nn.Linear.init(alloc, 16, 1, rnd);
                d3.setName("discriminator.fc3");

                var net_d = nn.sequential(.{
                    d1,
                    nn.LeakyReLU{ .alpha = 0.2 },
                    d2,
                    nn.LeakyReLU{ .alpha = 0.2 },
                    d3,
                });
                defer net_d.deinit(alloc);

                const x = try g.ones(&.{ 4, 2 }, false);
                x.setName("inputs.x");
                const logits = try net_d.forward(g, x);
                logits.setName("outputs.logits");
            }
        }.run, random, dir);

        // 18. conv2d
        try exportModel(allocator, "conv2d", struct {
            fn run(alloc: std.mem.Allocator, rnd: std.Random, g: *autodiff.Graph) !void {
                var conv = try nn.Conv2D.init(alloc, 1, 4, 3, rnd);
                defer conv.deinit(alloc);
                conv.setName("conv2d");

                const x = try g.ones(&.{ 1, 1, 8, 8 }, false);
                x.setName("inputs.x");
                const out = try conv.forward(g, x);
                out.setName("outputs.feature_map");
            }
        }.run, random, dir);

        std.debug.print("\n", .{});
    }

    // Also update sample_model_graph.json and minimal_model_graph.json copies
    std.debug.print("All 18 canonical models exported successfully!\n", .{});
}
