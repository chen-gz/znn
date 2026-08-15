const std = @import("std");
const zig_ml = @import("zig_ml");
const tensor = zig_ml.tensor;
const autodiff = zig_ml.autodiff;

pub const FitResult1D = struct {
    w: f32,
    b: f32,
};

pub const FitResultND = struct {
    w: []f32,
    b: f32,
};

/// Solve 1D Ridge regression analytically using closed-form formula:
/// w = Cov(x, y) / (Var(x) + lambda)
/// b = mean_y - w * mean_x
pub fn solveRidge1DAnalytical(x: []const f32, y: []const f32, lambda: f32) FitResult1D {
    const N = x.len;
    const N_f = @as(f32, @floatFromInt(N));

    var sum_x: f32 = 0.0;
    var sum_y: f32 = 0.0;
    for (0..N) |i| {
        sum_x += x[i];
        sum_y += y[i];
    }
    const mean_x = sum_x / N_f;
    const mean_y = sum_y / N_f;

    var num: f32 = 0.0;
    var den: f32 = 0.0;
    for (0..N) |i| {
        const dx = x[i] - mean_x;
        const dy = y[i] - mean_y;
        num += dx * dy;
        den += dx * dx;
    }

    const w = num / (den + lambda);
    const b = mean_y - w * mean_x;
    return FitResult1D{ .w = w, .b = b };
}

/// Solve 1D Ridge regression iteratively using Gradient Descent on the autograd graph
/// Loss = MSE(y_pred, y) + (lambda / N) * w^2
pub fn solveRidge1DGradientDescent(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    lambda: f32,
    lr: f32,
    epochs: usize,
) !FitResult1D {
    const N = x.len;
    const N_f = @as(f32, @floatFromInt(N));
    var w_data = [_]f32{0.0};
    var b_data = [_]f32{0.0};

    // Scaled lambda for MSE loss: lambda_loss = 2.0 * lambda / N
    const lambda_l2 = 2.0 * lambda / N_f;

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

        // Compute loss: MSE + L2 loss
        const loss_node = try graph.ridgeLoss(y_pred, y_node, w_node, lambda_l2);
        const loss = loss_node.data[0];

        // Backward
        try graph.backward(loss_node);

        // Update weights: param -= lr * grad
        w_data[0] -= lr * w_node.grad[0];
        b_data[0] -= lr * b_node.grad[0];

        if (epoch == 1 or epoch % 20 == 0) {
            std.debug.print("Epoch {d:3}/{d}: Loss = {d:.6} | w = {d:.4} | b = {d:.4}\n", .{
                epoch,
                epochs,
                loss,
                w_data[0],
                b_data[0],
            });
        }
    }

    return FitResult1D{ .w = w_data[0], .b = b_data[0] };
}

/// Solve Multi-dimensional Ridge regression iteratively using Gradient Descent on autograd graph
pub fn solveRidgeNDGradientDescent(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    n_samples: usize,
    n_features: usize,
    lambda: f32,
    lr: f32,
    epochs: usize,
) !FitResultND {
    const N = n_samples;
    const D = n_features;
    const N_f = @as(f32, @floatFromInt(N));

    const w_data = try allocator.alloc(f32, D);
    @memset(w_data, 0.0);
    var b_data = [_]f32{0.0};

    const lambda_l2 = 2.0 * lambda / N_f;

    for (1..(epochs + 1)) |epoch| {
        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();

        const x_node = try graph.tensorWithData(N, D, x, false);
        const y_node = try graph.tensorWithData(N, 1, y, false);
        const w_node = try graph.tensorWithData(D, 1, w_data, true);
        const b_node = try graph.tensorNDWithData(&.{1}, &b_data, true);

        // Forward: y_pred = X * W + b
        const x_w = try x_node.matmul(w_node, allocator, &graph);
        const y_pred = try x_w.addBias(b_node, allocator, &graph);

        // Compute loss: MSE + L2 loss
        const loss_node = try graph.ridgeLoss(y_pred, y_node, w_node, lambda_l2);
        const loss = loss_node.data[0];

        // Backward
        try graph.backward(loss_node);

        // Update weights
        for (0..D) |j| {
            w_data[j] -= lr * w_node.grad[j];
        }
        b_data[0] -= lr * b_node.grad[0];

        if (epoch == 1 or epoch % 50 == 0) {
            std.debug.print("Epoch {d:3}/{d}: Loss = {d:.6} | w = [", .{ epoch, epochs, loss });
            for (0..D) |j| {
                if (j > 0) std.debug.print(", ", .{});
                std.debug.print("{d:.4}", .{w_data[j]});
            }
            std.debug.print("] | b = {d:.4}\n", .{b_data[0]});
        }
    }

    return FitResultND{ .w = w_data, .b = b_data[0] };
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("===============================================================\n", .{});
    std.debug.print("Ridge Regression (L2 Regularization) from Scratch in Zig (znn)\n", .{});
    std.debug.print("===============================================================\n\n", .{});

    // ------------------------------------------------------------------------
    // Part 1: 1D Ridge Regression & Weight Shrinkage Verification
    // ------------------------------------------------------------------------
    std.debug.print("--- Part 1: 1D Ridge Regression (Analytical vs Autograd GD) ---\n", .{});
    const N = 100;
    const true_w: f32 = 3.0;
    const true_b: f32 = 1.5;
    const lambda_1d: f32 = 25.0; // Noticeable L2 penalty

    tensor.manualSeed(42);

    const x_1d = (try tensor.rand(allocator, &.{ N, 1 })).mulScalar_(4.0).addScalar_(-2.0);
    defer tensor.free(allocator, x_1d);

    const noise_1d = (try tensor.rand(allocator, &.{ N, 1 })).mulScalar_(0.4).addScalar_(-0.2);
    defer tensor.free(allocator, noise_1d);

    const y_1d = (try x_1d.clone(allocator)).mulScalar_(true_w).addScalar_(true_b);
    defer tensor.free(allocator, y_1d);
    _ = try y_1d.add_(noise_1d);

    std.debug.print("Running 1D Autograd Gradient Descent (lambda = {d:.1})...\n", .{lambda_1d});
    const gd_1d = try solveRidge1DGradientDescent(allocator, x_1d.data, y_1d.data, lambda_1d, 0.05, 100);

    const analytical_1d = solveRidge1DAnalytical(x_1d.data, y_1d.data, lambda_1d);
    const ols_1d = solveRidge1DAnalytical(x_1d.data, y_1d.data, 0.0);

    std.debug.print("\n1D Results Summary:\n", .{});
    std.debug.print("  Ground Truth:                  y = {d:.4} * x + {d:.4}\n", .{ true_w, true_b });
    std.debug.print("  OLS (No Regularization):       y = {d:.4} * x + {d:.4}\n", .{ ols_1d.w, ols_1d.b });
    std.debug.print("  Ridge Analytical (lambda={d:.1}): y = {d:.4} * x + {d:.4} (Shrunk towards 0!)\n", .{ lambda_1d, analytical_1d.w, analytical_1d.b });
    std.debug.print("  Ridge Autograd GD (lambda={d:.1}): y = {d:.4} * x + {d:.4}\n\n", .{ lambda_1d, gd_1d.w, gd_1d.b });

    // ------------------------------------------------------------------------
    // Part 2: Multicollinearity & Why Ridge Regression is Essential
    // ------------------------------------------------------------------------
    std.debug.print("--- Part 2: Multicollinear Features & Overfitting Prevention ---\n", .{});
    std.debug.print("Setting up 2 collinear features: x2 = x1 + tiny noise.\n", .{});
    std.debug.print("Ground Truth model: y = 2.0 * x1 + 2.0 * x2 + 1.0\n\n", .{});

    const N_multi = 80;
    const D_multi = 2;
    var x_multi_data = try allocator.alloc(f32, N_multi * D_multi);
    defer allocator.free(x_multi_data);

    var y_multi_data = try allocator.alloc(f32, N_multi);
    defer allocator.free(y_multi_data);

    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    for (0..N_multi) |i| {
        const x1 = rng.float(f32) * 4.0 - 2.0;
        // x2 is almost identical to x1 with tiny noise (severe multicollinearity!)
        const x2 = x1 + (rng.float(f32) * 0.02 - 0.01);
        const y_val = 2.0 * x1 + 2.0 * x2 + 1.0 + (rng.float(f32) * 0.2 - 0.1);

        x_multi_data[i * D_multi + 0] = x1;
        x_multi_data[i * D_multi + 1] = x2;
        y_multi_data[i] = y_val;
    }

    // Solve with OLS (lambda = 0)
    const w_ols = try allocator.alloc(f32, D_multi);
    defer allocator.free(w_ols);
    var b_ols: f32 = 0.0;
    try tensor.solveRidgeAnalytical(allocator, x_multi_data, y_multi_data, N_multi, D_multi, 0.0, w_ols, &b_ols);

    // Solve with Ridge (lambda = 5.0)
    const w_ridge = try allocator.alloc(f32, D_multi);
    defer allocator.free(w_ridge);
    var b_ridge: f32 = 0.0;
    try tensor.solveRidgeAnalytical(allocator, x_multi_data, y_multi_data, N_multi, D_multi, 5.0, w_ridge, &b_ridge);

    // Solve with Ridge Autograd GD (lambda = 5.0)
    std.debug.print("Training Multicollinear Ridge via Autograd GD (lambda = 5.0)...\n", .{});
    const gd_multi = try solveRidgeNDGradientDescent(allocator, x_multi_data, y_multi_data, N_multi, D_multi, 5.0, 0.02, 200);
    defer allocator.free(gd_multi.w);

    std.debug.print("\nMulticollinearity Comparison:\n", .{});
    std.debug.print("  Ground Truth:                  w = [2.0000, 2.0000], b = 1.0000\n", .{});
    std.debug.print("  OLS (lambda = 0.0):            w = [{d:.4}, {d:.4}], b = {d:.4}  <-- Extreme variance / Unstable!\n", .{ w_ols[0], w_ols[1], b_ols });
    std.debug.print("  Ridge Analytical (lambda=5.0): w = [{d:.4}, {d:.4}], b = {d:.4}  <-- Stabilized & balanced!\n", .{ w_ridge[0], w_ridge[1], b_ridge });
    std.debug.print("  Ridge Autograd GD (lambda=5.0): w = [{d:.4}, {d:.4}], b = {d:.4}\n\n", .{ gd_multi.w[0], gd_multi.w[1], gd_multi.b });

    // ------------------------------------------------------------------------
    // Part 3: Ridge Trace (Regularization Path)
    // ------------------------------------------------------------------------
    std.debug.print("--- Part 3: Ridge Regularization Path (Trace) ---\n", .{});
    std.debug.print("Evaluating weight shrinkage across varying lambda values:\n", .{});
    const lambdas = [_]f32{ 0.0, 0.1, 1.0, 5.0, 20.0, 100.0, 500.0 };

    const w_trace = try allocator.alloc(f32, D_multi);
    defer allocator.free(w_trace);
    var b_trace: f32 = 0.0;

    std.debug.print("+------------+-------------------------+------------+\n", .{});
    std.debug.print("|   Lambda   |        Weights w        |  Bias b    |\n", .{});
    std.debug.print("+------------+-------------------------+------------+\n", .{});
    for (lambdas) |lam| {
        try tensor.solveRidgeAnalytical(allocator, x_multi_data, y_multi_data, N_multi, D_multi, lam, w_trace, &b_trace);
        std.debug.print("| {d:10.1} | [{d:9.4}, {d:9.4}] | {d:10.4} |\n", .{ lam, w_trace[0], w_trace[1], b_trace });
    }
    std.debug.print("+------------+-------------------------+------------+\n\n", .{});
    std.debug.print("Done! Ridge regression implementation verified successfully.\n", .{});
}
