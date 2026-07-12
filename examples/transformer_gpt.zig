const std = @import("std");
const zig_ml = @import("zig_ml");
const nn = zig_ml.nn;
const autodiff = zig_ml.autodiff;
const tensor = zig_ml.tensor;

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    std.debug.print("=== Mini GPT Training Example ===\n", .{});
    std.debug.print("Training a mini GPT model to memorize 'hello hello'\n\n", .{});

    const text = "hello hello";
    _ = text;
    // Char to ID map:
    // 'h'->0, 'e'->1, 'l'->2, 'o'->3, ' '->4
    const vocab_size = 5;
    const block_size = 6; // Context length

    const config = nn.GPTConfig{
        .vocab_size = vocab_size,
        .block_size = block_size,
        .n_embd = 16,
        .n_head = 2,
        .n_layer = 2,
    };

    // Wrap GPT in Module for easy training
    const GPTModel = nn.Module(nn.GPT(config));
    var model = GPTModel.init(allocator, try nn.GPT(config).init(allocator, random));
    defer model.deinit();

    const lr: f32 = 0.01;

    var optimizer = try zig_ml.optim.SGDOptimizer.init(allocator, &model, .{
        .lr = lr,
        .momentum = 0.9,
    });
    defer optimizer.deinit();

    // Prepare training data
    // Text: [0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3] (length 11)
    const data = [_]u8{ 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3 };

    // We can extract multiple overlapping windows of size block_size+1
    // Sample 1: [0,1,2,2,3,4] -> target [1,2,2,3,4,0]
    // Sample 2: [1,2,2,3,4,0] -> target [2,2,3,4,0,1]
    // Sample 3: [2,2,3,4,0,1] -> target [2,3,4,0,1,2]
    // Sample 4: [2,3,4,0,1,2] -> target [3,4,0,1,2,2]
    // Sample 5: [3,4,0,1,2,2] -> target [4,0,1,2,2,3]
    // Total 5 samples.
    // Batch size = 5 (train on all samples at once).

    const batch_size = 5;
    const x_data = try allocator.alloc(f32, batch_size * block_size);
    defer allocator.free(x_data);
    const y_data = try allocator.alloc(u8, batch_size * block_size);
    defer allocator.free(y_data);

    for (0..batch_size) |i| {
        for (0..block_size) |j| {
            x_data[i * block_size + j] = @as(f32, @floatFromInt(data[i + j]));
            y_data[i * block_size + j] = data[i + j + 1];
        }
    }

    std.debug.print("Training Samples:\n", .{});
    for (0..batch_size) |i| {
        std.debug.print("  Sample {}: In={any} -> Out={any}\n", .{ i, x_data[i * block_size .. (i + 1) * block_size], y_data[i * block_size .. (i + 1) * block_size] });
    }

    const epochs = 100;

    std.debug.print("\nStarting training for {} epochs...\n", .{epochs});
    for (0..epochs) |epoch| {
        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();

        const x_node = try graph.tensorND(&.{ batch_size, block_size }, false);
        @memcpy(x_node.data, x_data);

        // Forward
        const logits = try model.forward(allocator, &graph, x_node); // Shape [5, 6, 5]

        const logits_reshaped = try graph.reshape(logits, &.{ batch_size * block_size, vocab_size });

        const loss = try graph.softmaxCrossEntropy(logits_reshaped, y_data);

        model.zeroGrad();
        try graph.backward(loss);
        optimizer.step();

        if ((epoch + 1) % 10 == 0 or epoch == 0) {
            std.debug.print("  Epoch {:3} | Loss: {d:.6}\n", .{ epoch + 1, loss.data[0] });
        }
    }

    // Generation test
    std.debug.print("\n=== Generation Test ===\n", .{});
    std.debug.print("Prompt: 'hell'\n", .{});
    var gen_buf: std.ArrayList(u8) = .empty;
    defer gen_buf.deinit(allocator);
    try gen_buf.appendSlice(allocator, &.{ 0, 1, 2, 2 }); // 'h', 'e', 'l', 'l'

    const decode = [_]u8{ 'h', 'e', 'l', 'o', ' ' };

    std.debug.print("Generated: ", .{});
    for (gen_buf.items) |id| std.debug.print("{c}", .{decode[id]});

    // Generate 10 tokens
    for (0..10) |_| {
        const seq_len = @min(gen_buf.items.len, block_size);
        const start_idx = gen_buf.items.len - seq_len;
        const input_slice = gen_buf.items[start_idx..];

        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();

        const x_node = try graph.tensorND(&.{ 1, seq_len }, false);
        for (input_slice, 0..) |val, idx| {
            x_node.data[idx] = @as(f32, @floatFromInt(val));
        }

        const logits = try model.forward(allocator, &graph, x_node); // Shape [1, T, V]
        
        const start = (@as(usize, seq_len) - 1) * @as(usize, vocab_size);
        const end = @as(usize, seq_len) * @as(usize, vocab_size);
        const last_token_logits = logits.data[start..end];

        var max_val: f32 = -1e9;
        var next_token: u8 = 0;
        for (last_token_logits, 0..) |val, idx| {
            if (val > max_val) {
                max_val = val;
                next_token = @as(u8, @intCast(idx));
            }
        }

        try gen_buf.append(allocator, next_token);
        std.debug.print("{c}", .{decode[next_token]});
    }
    std.debug.print("\n", .{});
}
