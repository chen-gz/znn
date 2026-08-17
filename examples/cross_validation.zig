const std = @import("std");
const zig_ml = @import("zig_ml");

pub fn softThreshold(z: f32, gamma: f32) f32 {
    if (z > gamma) {
        return z - gamma;
    } else if (z < -gamma) {
        return z + gamma;
    } else {
        return 0.0;
    }
}

pub const ModelResult = struct {
    weights: []f32,
    intercept: f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ModelResult) void {
        self.allocator.free(self.weights);
    }

    pub fn predict(self: ModelResult, x_row: []const f32) f32 {
        var sum = self.intercept;
        for (0..self.weights.len) |j| {
            sum += x_row[j] * self.weights[j];
        }
        return sum;
    }
};

pub fn solveLasso(
    allocator: std.mem.Allocator,
    N: usize,
    P: usize,
    X: []const f32,
    y: []const f32,
    alpha: f32,
    max_iter: usize,
    tol: f32,
) !ModelResult {
    const mean_x = try allocator.alloc(f32, P);
    defer allocator.free(mean_x);
    @memset(mean_x, 0.0);

    var sum_y: f32 = 0.0;
    for (0..N) |i| {
        sum_y += y[i];
        for (0..P) |j| {
            mean_x[j] += X[i * P + j];
        }
    }
    const mean_y = sum_y / @as(f32, @floatFromInt(N));
    for (0..P) |j| {
        mean_x[j] /= @as(f32, @floatFromInt(N));
    }

    const X_c = try allocator.alloc(f32, N * P);
    defer allocator.free(X_c);
    const y_c = try allocator.alloc(f32, N);
    defer allocator.free(y_c);

    for (0..N) |i| {
        y_c[i] = y[i] - mean_y;
        for (0..P) |j| {
            X_c[i * P + j] = X[i * P + j] - mean_x[j];
        }
    }

    const L = try allocator.alloc(f32, P);
    defer allocator.free(L);
    const n_f = @as(f32, @floatFromInt(N));
    for (0..P) |j| {
        var sum_sq: f32 = 0.0;
        for (0..N) |i| {
            const val = X_c[i * P + j];
            sum_sq += val * val;
        }
        L[j] = sum_sq / n_f;
    }

    const weights = try allocator.alloc(f32, P);
    @memset(weights, 0.0);

    const r = try allocator.alloc(f32, N);
    defer allocator.free(r);
    @memcpy(r, y_c);

    for (0..max_iter) |_| {
        var max_diff: f32 = 0.0;

        for (0..P) |j| {
            if (L[j] < 1e-12) continue;

            var dot_xr: f32 = 0.0;
            for (0..N) |i| {
                dot_xr += X_c[i * P + j] * r[i];
            }
            const c_j = (dot_xr / n_f) + L[j] * weights[j];
            const w_new = softThreshold(c_j, alpha) / L[j];

            const diff = w_new - weights[j];
            if (@abs(diff) > 0.0) {
                for (0..N) |i| {
                    r[i] -= X_c[i * P + j] * diff;
                }
                weights[j] = w_new;
                if (@abs(diff) > max_diff) {
                    max_diff = @abs(diff);
                }
            }
        }

        if (max_diff < tol) break;
    }

    var dot_mw: f32 = 0.0;
    for (0..P) |j| {
        dot_mw += mean_x[j] * weights[j];
    }
    const intercept = mean_y - dot_mw;

    return ModelResult{
        .weights = weights,
        .intercept = intercept,
        .allocator = allocator,
    };
}

pub const Fold = struct {
    train_indices: []usize,
    val_indices: []usize,

    pub fn deinit(self: *Fold, allocator: std.mem.Allocator) void {
        allocator.free(self.train_indices);
        allocator.free(self.val_indices);
    }
};

pub fn kFoldSplit(
    allocator: std.mem.Allocator,
    n_samples: usize,
    k_splits: usize,
    random: std.Random,
) ![]Fold {
    const indices = try allocator.alloc(usize, n_samples);
    defer allocator.free(indices);
    for (0..n_samples) |i| indices[i] = i;

    // Fisher-Yates shuffle
    var i = n_samples;
    while (i > 1) {
        i -= 1;
        const j = random.uintLessThan(usize, i + 1);
        const tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    const folds = try allocator.alloc(Fold, k_splits);
    const base_size = n_samples / k_splits;
    const remainder = n_samples % k_splits;

    var current_idx: usize = 0;
    for (0..k_splits) |fold_idx| {
        const val_size = base_size + if (fold_idx < remainder) @as(usize, 1) else @as(usize, 0);
        const train_size = n_samples - val_size;

        const val_indices = try allocator.alloc(usize, val_size);
        const train_indices = try allocator.alloc(usize, train_size);

        @memcpy(val_indices, indices[current_idx..(current_idx + val_size)]);

        var tr_cnt: usize = 0;
        for (0..current_idx) |idx| {
            train_indices[tr_cnt] = indices[idx];
            tr_cnt += 1;
        }
        for ((current_idx + val_size)..n_samples) |idx| {
            train_indices[tr_cnt] = indices[idx];
            tr_cnt += 1;
        }

        folds[fold_idx] = Fold{
            .train_indices = train_indices,
            .val_indices = val_indices,
        };

        current_idx += val_size;
    }

    return folds;
}

pub const CVResult = struct {
    alpha: f32,
    mean_mse: f32,
    se_mse: f32,
    fold_mses: [5]f32,
};

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=================================================================\n", .{});
    std.debug.print("  5-Fold Cross-Validation for Hyperparameter Tuning in Zig       \n", .{});
    std.debug.print("=================================================================\n\n", .{});

    const N: usize = 150;
    const P: usize = 15;
    const K: usize = 5;

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

    // 2. Generate 5 Folds
    const folds = try kFoldSplit(allocator, N, K, random);
    defer {
        for (folds) |*f| f.deinit(allocator);
        allocator.free(folds);
    }

    // 3. Grid Search over Alphas
    const candidate_alphas = [_]f32{ 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0 };
    var results = try allocator.alloc(CVResult, candidate_alphas.len);
    defer allocator.free(results);

    var best_min_idx: usize = 0;
    var min_mse: f32 = 1e9;

    for (candidate_alphas, 0..) |alpha, a_idx| {
        var fold_mses: [5]f32 = undefined;

        for (0..K) |k| {
            const tr_indices = folds[k].train_indices;
            const va_indices = folds[k].val_indices;
            const N_tr = tr_indices.len;
            const N_va = va_indices.len;

            // Extract Fold Train data
            const X_tr = try allocator.alloc(f32, N_tr * P);
            defer allocator.free(X_tr);
            const y_tr = try allocator.alloc(f32, N_tr);
            defer allocator.free(y_tr);

            for (tr_indices, 0..) |orig_i, local_i| {
                y_tr[local_i] = y[orig_i];
                for (0..P) |j| {
                    X_tr[local_i * P + j] = X[orig_i * P + j];
                }
            }

            // Extract Fold Val data
            const X_va = try allocator.alloc(f32, N_va * P);
            defer allocator.free(X_va);
            const y_va = try allocator.alloc(f32, N_va);
            defer allocator.free(y_va);

            for (va_indices, 0..) |orig_i, local_i| {
                y_va[local_i] = y[orig_i];
                for (0..P) |j| {
                    X_va[local_i * P + j] = X[orig_i * P + j];
                }
            }

            // Standardize within fold (Leakage-free!)
            const tr_mean = try allocator.alloc(f32, P);
            defer allocator.free(tr_mean);
            const tr_std = try allocator.alloc(f32, P);
            defer allocator.free(tr_std);
            @memset(tr_mean, 0.0);

            for (0..N_tr) |i| {
                for (0..P) |j| tr_mean[j] += X_tr[i * P + j];
            }
            for (0..P) |j| tr_mean[j] /= @as(f32, @floatFromInt(N_tr));

            for (0..P) |j| {
                var sum_sq: f32 = 0.0;
                for (0..N_tr) |i| {
                    const diff = X_tr[i * P + j] - tr_mean[j];
                    sum_sq += diff * diff;
                }
                var s = @sqrt(sum_sq / @as(f32, @floatFromInt(N_tr)));
                if (s < 1e-6) s = 1.0;
                tr_std[j] = s;
            }

            for (0..N_tr) |i| {
                for (0..P) |j| {
                    X_tr[i * P + j] = (X_tr[i * P + j] - tr_mean[j]) / tr_std[j];
                }
            }
            for (0..N_va) |i| {
                for (0..P) |j| {
                    X_va[i * P + j] = (X_va[i * P + j] - tr_mean[j]) / tr_std[j];
                }
            }

            // Fit Lasso
            var model = try solveLasso(allocator, N_tr, P, X_tr, y_tr, alpha, 1000, 1e-4);
            defer model.deinit();

            // Evaluate on Validation fold
            var total_sq_err: f32 = 0.0;
            for (0..N_va) |i| {
                const x_row = X_va[(i * P)..((i + 1) * P)];
                const pred = model.predict(x_row);
                const err = y_va[i] - pred;
                total_sq_err += err * err;
            }
            fold_mses[k] = total_sq_err / @as(f32, @floatFromInt(N_va));
        }

        var sum_mse: f32 = 0.0;
        for (fold_mses) |m| sum_mse += m;
        const mean_mse = sum_mse / @as(f32, @floatFromInt(K));

        var var_sum: f32 = 0.0;
        for (fold_mses) |m| {
            const diff = m - mean_mse;
            var_sum += diff * diff;
        }
        const se_mse = @sqrt(var_sum / @as(f32, @floatFromInt(K - 1))) / @sqrt(@as(f32, @floatFromInt(K)));

        results[a_idx] = CVResult{
            .alpha = alpha,
            .mean_mse = mean_mse,
            .se_mse = se_mse,
            .fold_mses = fold_mses,
        };

        if (mean_mse < min_mse) {
            min_mse = mean_mse;
            best_min_idx = a_idx;
        }
    }

    // 4. Determine 1-SE Rule
    const threshold_1se = results[best_min_idx].mean_mse + results[best_min_idx].se_mse;
    var best_1se_idx: usize = best_min_idx;
    for (results, 0..) |res, idx| {
        if (res.mean_mse <= threshold_1se) {
            best_1se_idx = idx; // Pick highest alpha within 1-SE
        }
    }

    // 5. Output Report
    std.debug.print("{s:<10} | {s:<18} | {s:<18} | {s}\n", .{ "Alpha", "Mean 5-Fold MSE", "Standard Error", "Status" });
    std.debug.print("-----------------------------------------------------------------\n", .{});
    for (results, 0..) |res, idx| {
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
    std.debug.print("Best Alpha (Min Rule):  {d:.4}\n", .{results[best_min_idx].alpha});
    std.debug.print("Best Alpha (1-SE Rule): {d:.4}\n\n", .{results[best_1se_idx].alpha});

    // 6. Retrain on full dataset with best alpha
    var final_model = try solveLasso(allocator, N, P, X, y, results[best_min_idx].alpha, 1000, 1e-5);
    defer final_model.deinit();

    std.debug.print("Final Trained Lasso Model Weights:\n[ ", .{});
    for (final_model.weights) |w| std.debug.print("{d:6.3} ", .{w});
    std.debug.print("]\n", .{});
}
