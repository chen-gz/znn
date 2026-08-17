const std = @import("std");
const zig_ml = @import("zig_ml");
const tensor = zig_ml.tensor;
const regression = zig_ml.regression;

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=================================================================\n", .{});
    std.debug.print("  Regularized Regression in Zig: Ridge vs. Lasso vs. Elastic Net \n", .{});
    std.debug.print("=================================================================\n\n", .{});

    const N: usize = 120;
    const P: usize = 10;

    tensor.manualSeed(42);

    // Ground truth: first 3 weights active, rest are 0 (sparse)
    const true_w = [_]f32{ 3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    const X = try allocator.alloc(f32, N * P);
    defer allocator.free(X);
    const y = try allocator.alloc(f32, N);
    defer allocator.free(y);

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..N) |i| {
        for (0..P) |j| {
            X[i * P + j] = random.floatNorm(f32);
        }
        // Inject high collinearity between feature 0 and feature 1
        X[i * P + 1] = X[i * P + 0] * 0.95 + random.floatNorm(f32) * 0.05;

        var dot: f32 = 0.0;
        for (0..P) |j| {
            dot += X[i * P + j] * true_w[j];
        }
        const noise = random.floatNorm(f32) * 0.2;
        y[i] = dot + noise;
    }

    // 1. Solve Ridge using library API
    var ridge = try regression.solveRidge(allocator, X, y, N, P, 0.1);
    defer ridge.deinit();

    // 2. Solve Lasso using library API
    var lasso = try regression.solveLasso(allocator, X, y, N, P, 0.1, 1000, 1e-5);
    defer lasso.deinit();

    // 3. Solve Elastic Net using library API
    var enet = try regression.solveElasticNet(allocator, X, y, N, P, 0.1, 0.5, 1000, 1e-5);
    defer enet.deinit();

    std.debug.print("Ground Truth Weights:   [ ", .{});
    for (true_w) |w| std.debug.print("{d:6.3} ", .{w});
    std.debug.print("]\n\n", .{});

    std.debug.print("Ridge (L2, alpha=0.1):  [ ", .{});
    for (ridge.weights) |w| std.debug.print("{d:6.3} ", .{w});
    std.debug.print("] (Intercept = {d:.4})\n", .{ridge.intercept});

    std.debug.print("Lasso (L1, alpha=0.1):  [ ", .{});
    for (lasso.weights) |w| std.debug.print("{d:6.3} ", .{w});
    std.debug.print("] (Intercept = {d:.4})\n", .{lasso.intercept});

    std.debug.print("ElasticNet (alpha=0.1): [ ", .{});
    for (enet.weights) |w| std.debug.print("{d:6.3} ", .{w});
    std.debug.print("] (Intercept = {d:.4})\n\n", .{enet.intercept});

    var zeros_ridge: usize = 0;
    var zeros_lasso: usize = 0;
    var zeros_enet: usize = 0;

    for (0..P) |j| {
        if (@abs(ridge.weights[j]) < 1e-3) zeros_ridge += 1;
        if (@abs(lasso.weights[j]) < 1e-3) zeros_lasso += 1;
        if (@abs(enet.weights[j]) < 1e-3) zeros_enet += 1;
    }

    std.debug.print("Sparsity Check (Zeroed weights out of {d}):\n", .{P});
    std.debug.print("  - Ridge (L2):       {d}/{d} zero weights\n", .{ zeros_ridge, P });
    std.debug.print("  - Lasso (L1):       {d}/{d} zero weights (High Sparsity)\n", .{ zeros_lasso, P });
    std.debug.print("  - ElasticNet (L1+L2): {d}/{d} zero weights (Sparse + Grouping)\n", .{ zeros_enet, P });
}
