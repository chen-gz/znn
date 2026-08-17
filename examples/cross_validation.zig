const std = @import("std");
const zig_ml = @import("zig_ml");
const regression = zig_ml.regression;
const cv = zig_ml.cv;

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=================================================================\n", .{});
    std.debug.print("  5-Fold Cross-Validation for Hyperparameter Tuning in Zig       \n", .{});
    std.debug.print("=================================================================\n\n", .{});

    const N: usize = 150;
    const P: usize = 15;

    var prng = std.Random.DefaultPrng.init(101);
    const random = prng.random();

    // 1. Generate Synthetic Data
    const true_w = [_]f32{ 2.5, -3.0, 1.8, -1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    const X = try allocator.alloc(f32, N * P);
    defer allocator.free(X);
    const y = try allocator.alloc(f32, N);
    defer allocator.free(y);

    for (0..N) |i| {
        for (0..P) |j| {
            X[i * P + j] = random.floatNorm(f32);
        }
        var dot: f32 = 0.0;
        for (0..P) |j| {
            dot += X[i * P + j] * true_w[j];
        }
        const noise = random.floatNorm(f32) * 0.4;
        y[i] = dot + noise;
    }

    // 2. Perform 5-Fold Grid Search using library's CrossValidationGridSearch
    const candidate_alphas = [_]f32{ 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0 };

    var cv_search = cv.CrossValidationGridSearch.init(allocator, 5);
    defer cv_search.deinit();

    try cv_search.searchLasso(X, y, N, P, &candidate_alphas, 101);

    // 3. Output Report
    std.debug.print("{s:<10} | {s:<18} | {s:<18} | {s}\n", .{ "Alpha", "Mean 5-Fold MSE", "Standard Error", "Status" });
    std.debug.print("-----------------------------------------------------------------\n", .{});

    const best_min_idx = cv_search.best_min_index;
    const best_1se_idx = cv_search.best_1se_index;

    for (cv_search.results.items, 0..) |res, idx| {
        const is_min = (idx == best_min_idx);
        const is_1se = (idx == best_1se_idx);
        const status = if (is_min) "<- Optimal (Min Rule)" else if (is_1se) "<- Optimal (1-SE Rule)" else "";
        std.debug.print("{d:<10.4} | {d:<18.5} | {d:<18.5} | {s}\n", .{
            res.alpha,
            res.mean_mse,
            res.se_mse,
            status,
        });
    }
    std.debug.print("-----------------------------------------------------------------\n", .{});
    std.debug.print("Best Alpha (Min Rule):  {d:.4}\n", .{cv_search.getBestMinAlpha()});
    std.debug.print("Best Alpha (1-SE Rule): {d:.4}\n\n", .{cv_search.getBest1SEAlpha()});

    // 4. Retrain on full dataset with best alpha using library's solveLasso
    var final_model = try regression.solveLasso(allocator, X, y, N, P, cv_search.getBestMinAlpha(), 1000, 1e-5);
    defer final_model.deinit();

    std.debug.print("Final Trained Lasso Model Weights:\n[ ", .{});
    for (final_model.weights) |w| std.debug.print("{d:6.3} ", .{w});
    std.debug.print("]\n", .{});
}
