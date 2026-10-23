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
    errdefer allocator.free(folds);
    var built: usize = 0;
    errdefer for (folds[0..built]) |*f| f.deinit(allocator);

    const base_size = n_samples / k_splits;
    const remainder = n_samples % k_splits;

    var current_idx: usize = 0;
    for (0..k_splits) |fold_idx| {
        const val_size = base_size + if (fold_idx < remainder) @as(usize, 1) else @as(usize, 0);
        const train_size = n_samples - val_size;

        const val_indices = try allocator.alloc(usize, val_size);
        errdefer allocator.free(val_indices);
        const train_indices = try allocator.alloc(usize, train_size);

        @memcpy(val_indices, indices[current_idx..(current_idx + val_size)]);
        @memcpy(train_indices[0..current_idx], indices[0..current_idx]);
        @memcpy(train_indices[current_idx..], indices[(current_idx + val_size)..]);

        folds[fold_idx] = Fold{
            .train_indices = train_indices,
            .val_indices = val_indices,
        };
        built += 1;

        current_idx += val_size;
    }

    return folds;
}

/// Copies the rows `indices` of the row-major `X` / `y` into contiguous `X_out` / `y_out`
fn gatherRows(X: []const f32, y: []const f32, n_features: usize, indices: []const usize, X_out: []f32, y_out: []f32) void {
    for (indices, 0..) |orig_i, local_i| {
        y_out[local_i] = y[orig_i];
        @memcpy(X_out[local_i * n_features .. (local_i + 1) * n_features], X[orig_i * n_features .. (orig_i + 1) * n_features]);
    }
}

/// Standardizes features to mean=0, std=1 (Leakage-free inside each fold)
pub const StandardScaler = struct {
    mean_: []f32,
    std_: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n_features: usize) !StandardScaler {
        const mean_ = try allocator.alloc(f32, n_features);
        errdefer allocator.free(mean_);
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

        // 2. Standard deviation (row-major sweep: contiguous reads, std_ used as the accumulator)
        @memset(self.std_, 0.0);
        for (0..n_samples) |i| {
            const row = X[i * n_features .. (i + 1) * n_features];
            for (self.std_, self.mean_, row) |*acc, m, v| {
                const diff = v - m;
                acc.* += diff * diff;
            }
        }
        for (self.std_) |*s| {
            const sd = @sqrt(s.* / N_f);
            s.* = if (sd < 1e-7) 1.0 else sd;
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

/// Cross-Validation (CV) Result for a specific hyperparameter configuration
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

/// Configuration options for K-Fold Cross-Validation (CV) grid search.
/// The defaults reproduce a 5-fold split with seed 42 and a coordinate descent budget of
/// 1000 sweeps with tolerance 1e-4 for each fitted model.
pub const CrossValidationOptions = struct {
    /// Number of folds; must be greater than 1 and no larger than the sample count
    k_splits: usize = 5,
    /// Seed of the Fisher-Yates shuffle that assigns samples to folds
    random_seed: u64 = 42,
    /// Maximum number of coordinate descent sweeps per fitted model
    max_iter: usize = 1000,
    /// Coordinate descent convergence tolerance (see `regression.solveElasticNet`)
    tol: f32 = 1e-4,

    pub const default: CrossValidationOptions = .{};

    pub fn defaultOptions() CrossValidationOptions {
        return .{};
    }
};

/// K-Fold Grid Search Engine for Least Absolute Shrinkage and Selection Operator (LASSO) Hyperparameter Tuning
pub const CrossValidationGridSearch = struct {
    allocator: std.mem.Allocator,
    options: CrossValidationOptions = .{},
    results: std.ArrayList(CVResult),
    best_min_index: usize = 0,
    best_1se_index: usize = 0,

    pub fn initDefault(allocator: std.mem.Allocator) CrossValidationGridSearch {
        return init(allocator, .{});
    }

    pub fn init(allocator: std.mem.Allocator, options: CrossValidationOptions) CrossValidationGridSearch {
        return .{
            .allocator = allocator,
            .options = options,
            .results = std.ArrayList(CVResult).empty,
        };
    }

    pub fn deinit(self: *CrossValidationGridSearch) void {
        for (self.results.items) |*res| {
            res.deinit();
        }
        self.results.deinit(self.allocator);
    }

    /// Evaluates Least Absolute Shrinkage and Selection Operator (LASSO) across a list of candidate alpha values,
    /// using the fold count, shuffle seed and solver budget from `options`.
    /// Results of a previous search are discarded, so `results`, `best_min_index` and `best_1se_index`
    /// always describe the latest grid.
    pub fn searchLasso(
        self: *CrossValidationGridSearch,
        X: []const f32,
        y: []const f32,
        n_samples: usize,
        n_features: usize,
        alphas: []const f32,
    ) !void {
        std.debug.assert(alphas.len > 0);
        for (self.results.items) |*res| res.deinit();
        self.results.clearRetainingCapacity();
        self.best_min_index = 0;
        self.best_1se_index = 0;

        const k_splits = self.options.k_splits;
        var prng = std.Random.DefaultPrng.init(self.options.random_seed);
        const random = prng.random();

        const folds = try kFoldSplit(self.allocator, n_samples, k_splits, random);
        defer {
            for (folds) |*f| f.deinit(self.allocator);
            self.allocator.free(folds);
        }

        // Ownership of each fold_mses slice moves to `results` once appended; until then it is freed here on error
        const all_fold_mses = try self.allocator.alloc([]f32, alphas.len);
        defer self.allocator.free(all_fold_mses);
        var allocated: usize = 0;
        var moved: usize = 0;
        errdefer for (all_fold_mses[moved..allocated]) |m| self.allocator.free(m);
        for (all_fold_mses) |*m| {
            m.* = try self.allocator.alloc(f32, k_splits);
            allocated += 1;
        }

        // Per-fold buffers sized for the largest fold, reused across folds
        const max_tr = n_samples - n_samples / k_splits;
        const max_va = (n_samples + k_splits - 1) / k_splits;
        const X_tr = try self.allocator.alloc(f32, max_tr * n_features);
        defer self.allocator.free(X_tr);
        const y_tr = try self.allocator.alloc(f32, max_tr);
        defer self.allocator.free(y_tr);
        const X_va = try self.allocator.alloc(f32, max_va * n_features);
        defer self.allocator.free(X_va);
        const y_va = try self.allocator.alloc(f32, max_va);
        defer self.allocator.free(y_va);
        const X_tr_scaled = try self.allocator.alloc(f32, max_tr * n_features);
        defer self.allocator.free(X_tr_scaled);
        const X_va_scaled = try self.allocator.alloc(f32, max_va * n_features);
        defer self.allocator.free(X_va_scaled);

        var scaler = try StandardScaler.init(self.allocator, n_features);
        defer scaler.deinit();

        for (0..k_splits) |k| {
            const tr_indices = folds[k].train_indices;
            const va_indices = folds[k].val_indices;
            const N_tr = tr_indices.len;
            const N_va = va_indices.len;

            gatherRows(X, y, n_features, tr_indices, X_tr[0 .. N_tr * n_features], y_tr[0..N_tr]);
            gatherRows(X, y, n_features, va_indices, X_va[0 .. N_va * n_features], y_va[0..N_va]);

            // Standardize strictly inside fold
            scaler.fitTransform(X_tr[0 .. N_tr * n_features], N_tr, n_features, X_tr_scaled[0 .. N_tr * n_features]);
            scaler.transform(X_va[0 .. N_va * n_features], N_va, n_features, X_va_scaled[0 .. N_va * n_features]);

            for (alphas, 0..) |alpha, a_idx| {
                // Fit Least Absolute Shrinkage and Selection Operator (LASSO)
                var model = try regression.solveLasso(self.allocator, X_tr_scaled[0 .. N_tr * n_features], y_tr[0..N_tr], N_tr, n_features, alpha, self.options.max_iter, self.options.tol);
                defer model.deinit();

                // Compute validation Mean Squared Error (MSE)
                all_fold_mses[a_idx][k] = model.computeMSE(X_va_scaled[0 .. N_va * n_features], y_va[0..N_va], N_va, n_features);
            }
        }

        try self.results.ensureUnusedCapacity(self.allocator, alphas.len);
        var min_mse = std.math.inf(f32);
        for (alphas, 0..) |alpha, a_idx| {
            const fold_mses = all_fold_mses[a_idx];
            var sum_mse: f32 = 0.0;
            for (fold_mses) |m| sum_mse += m;
            const mean_mse = sum_mse / @as(f32, @floatFromInt(k_splits));

            var var_sum: f32 = 0.0;
            for (fold_mses) |m| {
                const diff = m - mean_mse;
                var_sum += diff * diff;
            }
            const se_mse = @sqrt(var_sum / @as(f32, @floatFromInt(k_splits - 1))) / @sqrt(@as(f32, @floatFromInt(k_splits)));

            self.results.appendAssumeCapacity(.{
                .alpha = alpha,
                .l1_ratio = 1.0,
                .mean_mse = mean_mse,
                .se_mse = se_mse,
                .fold_mses = fold_mses,
                .allocator = self.allocator,
            });
            moved += 1;

            if (mean_mse < min_mse) {
                min_mse = mean_mse;
                self.best_min_index = a_idx;
            }
        }

        // One-Standard-Error (1-SE) Rule: the most regularized (largest alpha) model whose
        // mean MSE lies within one standard error of the minimum; independent of the grid order
        const best = self.results.items[self.best_min_index];
        const threshold_1se = best.mean_mse + best.se_mse;
        self.best_1se_index = self.best_min_index;
        for (self.results.items, 0..) |res, idx| {
            if (res.mean_mse <= threshold_1se and res.alpha > self.results.items[self.best_1se_index].alpha) {
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

test "kFoldSplit uneven sample distribution and disjoint partitions" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(123);
    const n_samples: usize = 7;
    const k: usize = 3;

    const folds = try kFoldSplit(allocator, n_samples, k, prng.random());
    defer {
        for (folds) |*f| f.deinit(allocator);
        allocator.free(folds);
    }

    try std.testing.expectEqual(k, folds.len);

    var val_seen: [n_samples]bool = @splat(false);
    var total_val_items: usize = 0;

    for (folds) |f| {
        try std.testing.expectEqual(n_samples, f.train_indices.len + f.val_indices.len);
        for (f.val_indices) |v_idx| {
            try std.testing.expect(!val_seen[v_idx]); // each index in val set exactly once
            val_seen[v_idx] = true;
            total_val_items += 1;
        }
    }
    try std.testing.expectEqual(n_samples, total_val_items);
}

test "StandardScaler zero-variance constant feature safety" {
    const allocator = std.testing.allocator;
    var scaler = try StandardScaler.init(allocator, 2);
    defer scaler.deinit();

    // Feature 0 is constant (5.0), Feature 1 varies (1.0, 2.0, 3.0)
    const X = [_]f32{
        5.0, 1.0,
        5.0, 2.0,
        5.0, 3.0,
    };
    const X_scaled = try allocator.alloc(f32, 6);
    defer allocator.free(X_scaled);

    scaler.fitTransform(&X, 3, 2, X_scaled);

    // Feature 0: mean=5.0, std fallback to 1.0 to prevent division by zero
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), scaler.mean_[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scaler.std_[0], 1e-4);
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), X_scaled[i * 2], 1e-4);
    }
}

fn makeLinearData(comptime N: usize, comptime P: usize, seed: u64, X: *[N * P]f32, y: *[N]f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (0..N) |i| {
        for (0..P) |j| X[i * P + j] = r.floatNorm(f32);
        y[i] = 2.0 * X[i * P] - X[i * P + 1] + 0.1 * r.floatNorm(f32);
    }
}

test "searchLasso replaces previous results and the 1-SE choice ignores grid order" {
    const allocator = std.testing.allocator;
    const N = 40;
    const P = 3;
    var X: [N * P]f32 = undefined;
    var y: [N]f32 = undefined;
    makeLinearData(N, P, 3, &X, &y);

    var search = CrossValidationGridSearch.init(allocator, .{ .random_seed = 1 });
    defer search.deinit();

    try search.searchLasso(&X, &y, N, P, &.{ 0.001, 0.01, 0.1 });
    const ascending_1se = search.getBest1SEAlpha();

    // 第二次搜索：结果只包含新网格，最优下标指向新网格
    try search.searchLasso(&X, &y, N, P, &.{ 5.0, 10.0, 0.001 });
    try std.testing.expectEqual(@as(usize, 3), search.results.items.len);
    try std.testing.expectEqual(@as(f32, 0.001), search.getBestMinAlpha());

    // 同一网格倒序：1-SE 规则选择的 alpha 不随顺序改变
    try search.searchLasso(&X, &y, N, P, &.{ 0.1, 0.01, 0.001 });
    try std.testing.expectEqual(ascending_1se, search.getBest1SEAlpha());
    try std.testing.expectEqual(@as(f32, 0.001), search.getBestMinAlpha());
}

fn searchLassoForAllocationFailures(allocator: std.mem.Allocator) !void {
    const N = 12;
    const P = 2;
    var X: [N * P]f32 = undefined;
    var y: [N]f32 = undefined;
    makeLinearData(N, P, 9, &X, &y);
    var search = CrossValidationGridSearch.init(allocator, .{ .k_splits = 3, .random_seed = 2 });
    defer search.deinit();
    try search.searchLasso(&X, &y, N, P, &.{ 0.01, 0.1 });
}

test "kFoldSplit, StandardScaler and searchLasso do not leak on allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, searchLassoForAllocationFailures, .{});
}

test "searchLasso honours every CrossValidationOptions field" {
    const allocator = std.testing.allocator;
    const N = 40;
    const P = 3;
    var X: [N * P]f32 = undefined;
    var y: [N]f32 = undefined;
    makeLinearData(N, P, 5, &X, &y);
    const alphas = [_]f32{ 0.001, 0.1 };

    var baseline = CrossValidationGridSearch.initDefault(allocator);
    defer baseline.deinit();
    try baseline.searchLasso(&X, &y, N, P, &alphas);
    try std.testing.expectEqual(CrossValidationOptions{}, baseline.options);

    // 显式写出旧的硬编码取值 (5 折、种子 42、1000 轮、容差 1e-4)：结果与默认配置逐位一致
    var explicit = CrossValidationGridSearch.init(allocator, .{ .k_splits = 5, .random_seed = 42, .max_iter = 1000, .tol = 1e-4 });
    defer explicit.deinit();
    try explicit.searchLasso(&X, &y, N, P, &alphas);
    for (baseline.results.items, explicit.results.items) |a, b| {
        try std.testing.expectEqualSlices(f32, a.fold_mses, b.fold_mses);
    }

    // k_splits 决定每个 alpha 的折数
    var three_fold = CrossValidationGridSearch.init(allocator, .{ .k_splits = 3 });
    defer three_fold.deinit();
    try three_fold.searchLasso(&X, &y, N, P, &alphas);
    try std.testing.expectEqual(@as(usize, 3), three_fold.results.items[0].fold_mses.len);

    // random_seed 决定样本分折
    var reseeded = CrossValidationGridSearch.init(allocator, .{ .random_seed = 7 });
    defer reseeded.deinit();
    try reseeded.searchLasso(&X, &y, N, P, &alphas);
    try std.testing.expect(!std.mem.eql(f32, baseline.results.items[0].fold_mses, reseeded.results.items[0].fold_mses));

    // max_iter = 1 只做一轮坐标下降：小 alpha 下尚未收敛，验证误差与默认配置不同
    var one_sweep = CrossValidationGridSearch.init(allocator, .{ .max_iter = 1 });
    defer one_sweep.deinit();
    try one_sweep.searchLasso(&X, &y, N, P, &alphas);
    try std.testing.expect(!std.mem.eql(f32, baseline.results.items[0].fold_mses, one_sweep.results.items[0].fold_mses));

    // 极大的 tol 在第一轮后即判定收敛：与 max_iter = 1 逐位一致
    var loose_tol = CrossValidationGridSearch.init(allocator, .{ .tol = 1e9 });
    defer loose_tol.deinit();
    try loose_tol.searchLasso(&X, &y, N, P, &alphas);
    for (one_sweep.results.items, loose_tol.results.items) |a, b| {
        try std.testing.expectEqualSlices(f32, a.fold_mses, b.fold_mses);
    }
}
