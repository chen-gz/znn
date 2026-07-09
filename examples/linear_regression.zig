const std = @import("std");
const zig_ml = @import("zig_ml");
const tensor = zig_ml.tensor;
const autodiff = zig_ml.autodiff;

pub const FitResult = struct {
    w: f32,
    b: f32,
};

/// Solve linear regression analytically using least-squares closed-form formula:
/// w = cov(x, y) / var(x)
/// b = mean_y - w * mean_x
pub fn solveAnalytical(x: []const f32, y: []const f32) FitResult {
    const N = x.len;
    var sum_x: f32 = 0.0;
    var sum_y: f32 = 0.0;
    for (0..N) |i| {
        sum_x += x[i];
        sum_y += y[i];
    }
    const mean_x = sum_x / @as(f32, @floatFromInt(N));
    const mean_y = sum_y / @as(f32, @floatFromInt(N));

    var num: f32 = 0.0;
    var den: f32 = 0.0;
    for (0..N) |i| {
        const dx = x[i] - mean_x;
        const dy = y[i] - mean_y;
        num += dx * dy;
        den += dx * dx;
    }
    const w = num / den;
    const b = mean_y - w * mean_x;
    return FitResult{ .w = w, .b = b };
}

/// Solve linear regression iteratively using Gradient Descent on the autograd graph
pub fn solveGradientDescent(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    lr: f32,
    epochs: usize,
) !FitResult {
    const N = x.len;
    // Persist parameter data across graph iterations
    var w_data = [_]f32{0.0};
    var b_data = [_]f32{0.0};

    for (1..(epochs + 1)) |epoch| {
        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();

        const x_node = try graph.tensorWithData(N, 1, x, false);
        const y_node = try graph.tensorWithData(N, 1, y, false);
        const w_node = try graph.tensorWithData(1, 1, &w_data, true);
        const b_node = try graph.tensorNDWithData(&.{1}, &b_data, true);

        // Forward: y_pred = x * w + b
        const x_w = try x_node.matmul(w_node, allocator, &graph);
        const y_pred = try x_w.addBias(b_node, allocator, &graph);

        // Compute loss: MSE
        const loss_node = try graph.mseLoss(y_pred, y_node);
        const loss = loss_node.data[0];

        // Backward
        try graph.backward(loss_node);

        // Update weights: param -= lr * grad
        w_data[0] -= lr * w_node.grad[0];
        b_data[0] -= lr * b_node.grad[0];

        if (epoch == 1 or epoch % 10 == 0) {
            std.debug.print("Epoch {d:3}/{d}: Loss = {d:.6} | w = {d:.4} | b = {d:.4}\n", .{
                epoch,
                epochs,
                loss,
                w_data[0],
                b_data[0],
            });
        }
    }

    return FitResult{ .w = w_data[0], .b = b_data[0] };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=========================================\n", .{});
    std.debug.print("Linear Regression using Autodiff Engine\n", .{});
    std.debug.print("=========================================\n\n", .{});

    // 1. Generate synthetic data: y = 2.5 * x - 1.2 + noise
    const N = 100;
    const true_w: f32 = 2.5;
    const true_b: f32 = -1.2;

    // Set random seed globally for reproducibility (matches PyTorch's torch.manual_seed)
    tensor.manualSeed(12345);

    // NumPy-like data generation using our new Tensor vectorization APIs!
    const x_tensor = (try tensor.rand(allocator, &.{ N, 1 })).mulScalar_(4.0).addScalar_(-2.0);
    defer tensor.free(allocator, x_tensor);

    const noise_tensor = (try tensor.rand(allocator, &.{ N, 1 })).mulScalar_(0.1).addScalar_(-0.05);
    defer tensor.free(allocator, noise_tensor);

    const y_tensor = (try x_tensor.clone(allocator)).mulScalar_(true_w).addScalar_(true_b);
    defer tensor.free(allocator, y_tensor);
    _ = try y_tensor.add_(noise_tensor);

    // 2. Solve using Gradient Descent
    std.debug.print("--- Running Gradient Descent ---\n", .{});
    const gd_res = try solveGradientDescent(allocator, x_tensor.data, y_tensor.data, 0.05, 100);

    // 3. Solve using Analytical Formula
    const analytical_res = solveAnalytical(x_tensor.data, y_tensor.data);

    std.debug.print("\nTraining complete!\n", .{});
    std.debug.print("Final Learned Model (GD):   y = {d:.4} * x + {d:.4}\n", .{ gd_res.w, gd_res.b });
    std.debug.print("Analytical Model (Formula): y = {d:.4} * x + {d:.4}\n", .{ analytical_res.w, analytical_res.b });
    std.debug.print("Ground Truth Model:         y = {d:.2} * x + {d:.2}\n", .{ true_w, true_b });
}
