const std = @import("std");
const zig_ml = @import("zig_ml");
const nn = zig_ml.nn;
const autodiff = zig_ml.autodiff;
const tensor = zig_ml.tensor;

fn printGrad(t: *tensor.Tensor) void {
    if (t.grad.len == 0) {
        std.debug.print("No gradients\n", .{});
        return;
    }
    const temp = t.data;
    t.data = @constCast(t.grad);
    defer t.data = temp;
    t.print();
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    std.debug.print("=== Transformer Causal Self Attention Example ===\n", .{});
    std.debug.print("Attention allows tokens to focus on other tokens in the sequence.\n", .{});
    std.debug.print("Causal Self Attention uses a mask to ensure tokens can only attend to previous tokens.\n\n", .{});

    // Config: n_embd = 8, n_head = 2 (so head_dim = 4), block_size (max seq len) = 4
    // We will test with batch = 1, seq_len = 3.
    var att = try nn.CausalSelfAttention.init(allocator, 8, 2, random);
    defer att.deinit(allocator);

    // Input: Shape [1, 3, 8] (B=1, T=3, C=8)
    std.debug.print("Input Tensor (Shape: [1, 3, 8]):\n", .{});
    const x = try allocator.create(tensor.Tensor);
    const shape = tensor.Shape.init(&.{1, 3, 8});
    x.* = tensor.Tensor{
        .data = try allocator.alloc(f32, 24),
        .grad = &.{},
        .shape = shape,
        .strides = tensor.computeContiguousStrides(shape),
        .requires_grad = false,
        .creator = null,
    };
    defer {
        allocator.free(x.data);
        allocator.destroy(x);
    }
    // Fill with some values
    for (x.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.1;
    }
    x.print();

    std.debug.print("\n--- Graph Mode Forward ---\n", .{});
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{1, 3, 8}, false);
    @memcpy(x_node.data, x.data);

    const y = try att.forward(allocator, &graph, x_node);
    std.debug.print("Output (Shape: [1, 3, 8]):\n", .{});
    y.print();

    std.debug.print("\nWe set output gradients to 1.0 and run backward to compute gradients of projection weights.\n", .{});
    @memset(y.grad, 1.0);
    try graph.backward(y);

    std.debug.print("\nq_attn weights gradients (q_attn.weight.grad, Shape: [8, 8]):\n", .{});
    printGrad(att.q_attn.weight);

    std.debug.print("\nc_proj (output projection) weights gradients (c_proj.weight.grad, Shape: [8, 8]):\n", .{});
    printGrad(att.c_proj.weight);
}
