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
    // Cast const grad to mutable data for printing
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

    std.debug.print("=== Transformer Embedding Component Example ===\n", .{});
    std.debug.print("Embedding maps integer token IDs to continuous vectors.\n", .{});
    std.debug.print("We will initialize an Embedding layer with vocab_size = 10, embedding_dim = 4.\n\n", .{});

    var emb = try nn.Embedding.init(allocator, 10, 4);
    defer emb.deinit(allocator);

    // Input: Batch = 2, SeqLen = 3
    std.debug.print("\nInput Token IDs (Shape: [2, 3]):\n", .{});
    const x = try allocator.create(tensor.Tensor);
    const shape = tensor.Shape.init(&.{2, 3});
    x.* = tensor.Tensor{
        .data = try allocator.alloc(f32, 6),
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
    x.data[0] = 0; x.data[1] = 1; x.data[2] = 2;
    x.data[3] = 3; x.data[4] = 4; x.data[5] = 5;
    x.print();

    // 用样本输入建立一次前向计算图，依据计算图与下游激活函数初始化参数
    try nn.initModelWithSample(&emb, allocator, random, .{x});
    std.debug.print("\nInitial Embedding Weights (Shape: [10, 4]):\n", .{});
    emb.weight.print();

    std.debug.print("\n--- 1. Inference Forward (no-grad graph) ---\n", .{});
    var y_eager_graph = autodiff.Graph.initNoGrad(allocator);
    defer y_eager_graph.deinit();
    const y_eager = try emb.forward(&y_eager_graph, x);
    std.debug.print("Output (Shape: [2, 3, 4]):\n", .{});
    y_eager.print();

    std.debug.print("\n--- 2. Graph Mode Forward & Backward ---\n", .{});
    var graph = autodiff.Graph.init(allocator);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{2, 3}, false);
    @memcpy(x_node.data, x.data);

    const y = try emb.forward(&graph, x_node);
    std.debug.print("Graph Output (Shape: [2, 3, 4]):\n", .{});
    y.print();

    std.debug.print("\nWe set output gradients to 1.0 and run backward to compute gradients of embedding weights.\n", .{});
    @memset(y.grad, 1.0);
    try graph.backward(y);

    std.debug.print("\nEmbedding Weights Gradients (W.grad, Shape: [10, 4]):\n", .{});
    printGrad(emb.weight);
    std.debug.print("\nNotice that only the rows corresponding to input token IDs (0..5) have non-zero gradients (value 1.0).\n", .{});
}
