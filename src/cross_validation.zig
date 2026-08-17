const std = @import("std");
const regression = @import("regression.zig");

pub const Fold = struct {
    train_indices: []usize,
    val_indices: []usize,

    pub fn deinit(self: *Fold, allocator: std.mem.Allocator) void {
        allocator.free(self.train_indices);
        allocator.free(self.val_indices);
    }
};

/// Splits dataset of `n_samples` into `k_splits` folds using Fisher-Yates shuffle
pub fn kFoldSplit(
    allocator: std.mem.Allocator,
    n_samples: usize,
    k_splits: usize,
    random: std.Random,
) ![]Fold {
    std.debug.assert(k_splits > 1);
    std.debug.assert(n_samples >= k_splits);

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

/// Standardizes features to mean=0, std=1 (Leakage-free inside each fold)
pub const StandardScaler = struct {
    mean_: []f32,
    std_: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n_features: usize) !StandardScaler {
        const mean_ = try allocator.alloc(f32, n_features);
        const std_ = try allocator.alloc(f32, n_features);
        @memset(mean_, 0.0);
        @memset(std_, 1.0);
        return .{
            .mean_ = mean_,
            .std_ = std_,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *StandardScaler) void {
        self.allocator.free(self.mean_);
        self.allocator.free(self.std_);
    }

    pub fn fitTransform(self: *StandardScaler, X: []const f32, n_samples: usize, n_features: usize, out_scaled: []f32) void {
        std.debug.assert(X.len == n_samples * n_features);
        std.debug.assert(out_scaled.len == n_samples * n_features);

        @memset(self.mean_, 0.0);
        const N_f = @as(f32, @floatFromInt(n_samples));

        // 1. Mean
        for (0..n_samples) |i| {
            for (0..n_features) |j| {
                self.mean_[j] += X[i * n_features + j];
            }
        }
        for (0..n_features) |j| {
            self.mean_[j] /= N_f;
        }

        // 2. Standard deviation
        for (0..n_features) |j| {
            var sum_sq: f32 = 0.0;
            for (0..n_samples) |i| {
                const diff = X[i * n_features + j] - self.mean_[j];
                sum_sq += diff * diff;
            }
            var s = @sqrt(sum_sq / N_f);
            if (s < 1e-7) s = 1.0;
            self.std_[j] = s;
        }

        // 3. Transform
        self.transform(X, n_samples, n_features, out_scaled);
    }

    pub fn transform(self: StandardScaler, X: []const f32, n_samples: usize, n_features: usize, out_scaled: []f32) void {
        for (0..n_samples) |i| {
            for (0..n_features) |j| {
                out_scaled[i * n_features + j] = (X[i * n_features + j] - self.mean_[j]) / self.std_[j];
            }
        }
    }
};

/// Cross-Validation Result for a specific hyperparameter configuration
pub const CVResult = struct {
    alpha: f32,
    l1_ratio: f32 = 1.0,
    mean_mse: f32,
    se_mse: f32,
    fold_mses: []f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CVResult) void {
        self.allocator.free(self.fold_mses);
    }
};

/// 5-Fold Grid Search Engine for Lasso and Elastic Net Hyperparameter Tuning
pub const CrossValidationGridSearch = struct {
    allocator: std.mem.Allocator,
    k_splits: usize,
    results: std.ArrayList(CVResult),
    best_min_index: usize = 0,
    best_1se_index: usize = 0,

    pub fn init(allocator: std.mem.Allocator, k_splits: usize) CrossValidationGridSearch {
        return .{
            .allocator = allocator,
            .k_splits = k_splits,
            .results = std.ArrayList(CVResult).empty,
        };
    }

    pub fn deinit(self: *CrossValidationGridSearch) void {
        for (self.results.items) |*res| {
            res.deinit();
        }
        self.results.deinit(self.allocator);
    }

    /// Evaluates Lasso across a list of candidate alpha values
    pub fn searchLasso(
        self: *CrossValidationGridSearch,
        X: []const f32,
        y: []const f32,
        n_samples: usize,
        n_features: usize,
        alphas: []const f32,
        random_seed: u64,
    ) !void {
        var prng = std.Random.DefaultPrng.init(random_seed);
        const random = prng.random();

        const folds = try kFoldSplit(self.allocator, n_samples, self.k_splits, random);
        defer {
            for (folds) |*f| f.deinit(self.allocator);
            self.allocator.free(folds);
        }

        var min_mse: f32 = 1e12;

        for (alphas, 0..) |alpha, a_idx| {
            const fold_mses = try self.allocator.alloc(f32, self.k_splits);

            for (0..self.k_splits) |k| {
                const tr_indices = folds[k].train_indices;
                const va_indices = folds[k].val_indices;
                const N_tr = tr_indices.len;
                const N_va = va_indices.len;

                // Extract train data
                const X_tr = try self.allocator.alloc(f32, N_tr * n_features);
                defer self.allocator.free(X_tr);
                const y_tr = try self.allocator.alloc(f32, N_tr);
                defer self.allocator.free(y_tr);

                for (tr_indices, 0..) |orig_i, local_i| {
                    y_tr[local_i] = y[orig_i];
                    for (0..n_features) |j| {
                        X_tr[local_i * n_features + j] = X[orig_i * n_features + j];
                    }
                }

                // Extract val data
                const X_va = try self.allocator.alloc(f32, N_va * n_features);
                defer self.allocator.free(X_va);
                const y_va = try self.allocator.alloc(f32, N_va);
                defer self.allocator.free(y_va);

                for (va_indices, 0..) |orig_i, local_i| {
                    y_va[local_i] = y[orig_i];
                    for (0..n_features) |j| {
                        X_va[local_i * n_features + j] = X[orig_i * n_features + j];
                    }
                }

                // Standardize strictly inside fold
                var scaler = try StandardScaler.init(self.allocator, n_features);
                defer scaler.deinit();

                const X_tr_scaled = try self.allocator.alloc(f32, N_tr * n_features);
                defer self.allocator.free(X_tr_scaled);
                scaler.fitTransform(X_tr, N_tr, n_features, X_tr_scaled);

                const X_va_scaled = try self.allocator.alloc(f32, N_va * n_features);
                defer self.allocator.free(X_va_scaled);
                scaler.transform(X_va, N_va, n_features, X_va_scaled);

                // Fit Lasso
                var model = try regression.solveLasso(self.allocator, X_tr_scaled, y_tr, N_tr, n_features, alpha, 1000, 1e-4);
                defer model.deinit();

                // Compute validation MSE
                fold_mses[k] = model.computeMSE(X_va_scaled, y_va, N_va, n_features);
            }

            var sum_mse: f32 = 0.0;
            for (fold_mses) |m| sum_mse += m;
            const mean_mse = sum_mse / @as(f32, @floatFromInt(self.k_splits));

            var var_sum: f32 = 0.0;
            for (fold_mses) |m| {
                const diff = m - mean_mse;
                var_sum += diff * diff;
            }
            const se_mse = @sqrt(var_sum / @as(f32, @floatFromInt(self.k_splits - 1))) / @sqrt(@as(f32, @floatFromInt(self.k_splits)));

            const res = CVResult{
                .alpha = alpha,
                .l1_ratio = 1.0,
                .mean_mse = mean_mse,
                .se_mse = se_mse,
                .fold_mses = fold_mses,
                .allocator = self.allocator,
            };

            try self.results.append(self.allocator, res);

            if (mean_mse < min_mse) {
                min_mse = mean_mse;
                self.best_min_index = a_idx;
            }
        }

        // Determine 1-SE Rule
        const threshold_1se = self.results.items[self.best_min_index].mean_mse + self.results.items[self.best_min_index].se_mse;
        self.best_1se_index = self.best_min_index;
        for (self.results.items, 0..) |res, idx| {
            if (res.mean_mse <= threshold_1se) {
                self.best_1se_index = idx;
            }
        }
    }

    pub fn getBestMinAlpha(self: CrossValidationGridSearch) f32 {
        return self.results.items[self.best_min_index].alpha;
    }

    pub fn getBest1SEAlpha(self: CrossValidationGridSearch) f32 {
        return self.results.items[self.best_1se_index].alpha;
    }
};

test "kFoldSplit basic integrity" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const folds = try kFoldSplit(allocator, 100, 5, prng.random());
    defer {
        for (folds) |*f| f.deinit(allocator);
        allocator.free(folds);
    }

    try std.testing.expectEqual(@as(usize, 5), folds.len);
    for (folds) |f| {
        try std.testing.expectEqual(@as(usize, 20), f.val_indices.len);
        try std.testing.expectEqual(@as(usize, 80), f.train_indices.len);
    }
}

test "StandardScaler integrity" {
    const allocator = std.testing.allocator;
    var scaler = try StandardScaler.init(allocator, 2);
    defer scaler.deinit();

    const X = [_]f32{
        1.0, 10.0,
        2.0, 20.0,
        3.0, 30.0,
    };
    const X_scaled = try allocator.alloc(f32, 6);
    defer allocator.free(X_scaled);

    scaler.fitTransform(&X, 3, 2, X_scaled);
    try std.testing.expect(@abs(scaler.mean_[0] - 2.0) < 1e-4);
    try std.testing.expect(@abs(scaler.mean_[1] - 20.0) < 1e-4);
}
