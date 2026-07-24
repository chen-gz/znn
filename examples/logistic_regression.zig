const std = @import("std");
const zig_ml = @import("zig_ml");
const tensor = zig_ml.tensor;
const autodiff = zig_ml.autodiff;

pub const FitResult = struct {
    w1: f32,
    w2: f32,
    b: f32,
};

/// Solve logistic regression iteratively using Gradient Descent on the autograd graph
pub fn solveGradientDescent(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    lr: f32,
    epochs: usize,
) !FitResult {
    const N = x.len / 2;
    // Persist parameter data across graph iterations
    var w_data = [_]f32{ 0.0, 0.0 };
    var b_data = [_]f32{0.0};

    for (1..(epochs + 1)) |epoch| {
        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();

        const x_node = try graph.tensorWithData(N, 2, x, false);
        const y_node = try graph.tensorWithData(N, 1, y, false);
        // Weight is 2x1
        const w_node = try graph.tensorWithData(2, 1, &w_data, true);
        const b_node = try graph.tensorNDWithData(&.{1}, &b_data, true);

        // Forward: logits = x * w + b
        const x_w = try x_node.matmul(w_node, allocator, &graph);
        const logits = try x_w.addBias(b_node, allocator, &graph);

        // Compute loss: SigmoidCrossEntropy (BCE with logits)
        const loss_node = try logits.sigmoidCrossEntropy(y_node, allocator, &graph);
        const loss = loss_node.data[0];

        // Backward
        try graph.backward(loss_node);

        // Update weights: param -= lr * grad
        w_data[0] -= lr * w_node.grad[0];
        w_data[1] -= lr * w_node.grad[1];
        b_data[0] -= lr * b_node.grad[0];

        if (epoch == 1 or epoch % 20 == 0) {
            std.debug.print("Epoch {d:3}/{d}: Loss = {d:.6} | w = [{d:.4}, {d:.4}] | b = {d:.4}\n", .{
                epoch,
                epochs,
                loss,
                w_data[0],
                w_data[1],
                b_data[0],
            });
        }
    }

    return FitResult{ .w1 = w_data[0], .w2 = w_data[1], .b = b_data[0] };
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=========================================\n", .{});
    std.debug.print("Logistic Regression using Autodiff Engine\n", .{});
    std.debug.print("=========================================\n\n", .{});

    // 1. Generate synthetic data:
    // We have 2 features: x1, x2
    // True weights: w1 = 1.5, w2 = -2.0, b = 0.5
    const N = 100;
    const true_w1: f32 = 1.5;
    const true_w2: f32 = -2.0;
    const true_b: f32 = 0.5;

    tensor.manualSeed(12345);

    // X shape is 100 x 2
    const x_tensor = (try tensor.rand(allocator, &.{ N, 2 })).mulScalar_(4.0).addScalar_(-2.0);
    defer tensor.free(allocator, x_tensor);

    const y_tensor = try tensor.zeros(allocator, &.{ N, 1 });
    defer tensor.free(allocator, y_tensor);

    var prng = std.Random.DefaultPrng.init(54321);
    const random = prng.random();

    for (0..N) |i| {
        const x1 = x_tensor.data[i * 2 + 0];
        const x2 = x_tensor.data[i * 2 + 1];
        const logit = true_w1 * x1 + true_w2 * x2 + true_b;
        const prob = 1.0 / (1.0 + @exp(-logit));
        y_tensor.data[i] = if (random.float(f32) < prob) 1.0 else 0.0;
    }

    // 2. Solve using Gradient Descent
    std.debug.print("--- Running Gradient Descent ---\n", .{});
    const gd_res = try solveGradientDescent(allocator, x_tensor.data, y_tensor.data, 0.5, 100);

    std.debug.print("\nTraining complete!\n", .{});
    std.debug.print("Final Learned Model: w = [{d:.4}, {d:.4}], b = {d:.4}\n", .{ gd_res.w1, gd_res.w2, gd_res.b });
    std.debug.print("Ground Truth Model:  w = [{d:.2}, {d:.2}], b = {d:.2}\n", .{ true_w1, true_w2, true_b });
}
