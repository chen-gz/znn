const std = @import("std");
const tensor = @import("tensor.zig");

/// Soft-thresholding operator: S_gamma(z) = sign(z) * max(|z| - gamma, 0)
pub fn softThreshold(z: f32, gamma: f32) f32 {
    if (z > gamma) {
        return z - gamma;
    } else if (z < -gamma) {
        return z + gamma;
    } else {
        return 0.0;
    }
}

/// Result of fitting a linear regression model
pub const ModelResult = struct {
    weights: []f32,
    intercept: f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ModelResult) void {
        self.allocator.free(self.weights);
    }

    pub fn predict(self: ModelResult, x_row: []const f32) f32 {
        std.debug.assert(x_row.len == self.weights.len);
        var sum = self.intercept;
        for (0..self.weights.len) |j| {
            sum += x_row[j] * self.weights[j];
        }
        return sum;
    }

    pub fn predictBatch(self: ModelResult, allocator: std.mem.Allocator, X: []const f32, n_samples: usize, n_features: usize) ![]f32 {
        std.debug.assert(X.len == n_samples * n_features);
        std.debug.assert(n_features == self.weights.len);

        const preds = try allocator.alloc(f32, n_samples);
        for (0..n_samples) |i| {
            const row = X[i * n_features .. (i + 1) * n_features];
            preds[i] = self.predict(row);
        }
        return preds;
    }

    pub fn computeMSE(self: ModelResult, X: []const f32, y: []const f32, n_samples: usize, n_features: usize) f32 {
        std.debug.assert(X.len == n_samples * n_features);
        std.debug.assert(y.len == n_samples);

        var total_sq: f32 = 0.0;
        for (0..n_samples) |i| {
            const row = X[i * n_features .. (i + 1) * n_features];
            const diff = y[i] - self.predict(row);
            total_sq += diff * diff;
        }
        return total_sq / @as(f32, @floatFromInt(n_samples));
    }
};

/// Solves Ordinary Least Squares (OLS) or Ridge regression analytically:
/// min (1/2N) * ||Xw + b - y||_2^2 + (alpha/2) * ||w||_2^2
pub fn solveRidge(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    n_samples: usize,
    n_features: usize,
    alpha: f32,
) !ModelResult {
    std.debug.assert(x.len == n_samples * n_features);
    std.debug.assert(y.len == n_samples);

    const N = n_samples;
    const D = n_features;
    const N_f = @as(f32, @floatFromInt(N));

    // 1. Calculate column means
    const mean_x = try allocator.alloc(f32, D);
    defer allocator.free(mean_x);
    @memset(mean_x, 0.0);

    var sum_y: f32 = 0.0;
    for (0..N) |i| {
        sum_y += y[i];
        for (0..D) |j| {
            mean_x[j] += x[i * D + j];
        }
    }
    const mean_y = sum_y / N_f;
    for (0..D) |j| {
        mean_x[j] /= N_f;
    }

    // 2. Center X and y
    const X_c = try allocator.alloc(f32, N * D);
    defer allocator.free(X_c);
    const y_c = try allocator.alloc(f32, N);
    defer allocator.free(y_c);

    for (0..N) |i| {
        y_c[i] = y[i] - mean_y;
        for (0..D) |j| {
            X_c[i * D + j] = x[i * D + j] - mean_x[j];
        }
    }

    // 3. Compute normal matrix M = (1/N) * X_c^T X_c + alpha * I, and v = (1/N) * X_c^T y_c
    const M = try allocator.alloc(f32, D * D);
    defer allocator.free(M);
    @memset(M, 0.0);

    const v = try allocator.alloc(f32, D);
    defer allocator.free(v);
    @memset(v, 0.0);

    for (0..D) |j1| {
        for (0..D) |j2| {
            var dot: f32 = 0.0;
            for (0..N) |i| {
                dot += X_c[i * D + j1] * X_c[i * D + j2];
            }
            M[j1 * D + j2] = dot / N_f;
        }
        M[j1 * D + j1] += alpha;

        var dot_y: f32 = 0.0;
        for (0..N) |i| {
            dot_y += X_c[i * D + j1] * y_c[i];
        }
        v[j1] = dot_y / N_f;
    }

    // 4. Solve linear system M * w = v
    const weights = try allocator.alloc(f32, D);
    try tensor.solveLinearSystem(allocator, M, v, D, weights);

    // 5. Intercept: b = mean_y - mean_x^T * w
    var dot_mw: f32 = 0.0;
    for (0..D) |j| {
        dot_mw += mean_x[j] * weights[j];
    }
    const intercept = mean_y - dot_mw;

    return ModelResult{
        .weights = weights,
        .intercept = intercept,
        .allocator = allocator,
    };
}

/// Solves Ordinary Least Squares (alpha = 0)
pub fn solveOLS(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    n_samples: usize,
    n_features: usize,
) !ModelResult {
    return solveRidge(allocator, x, y, n_samples, n_features, 0.0);
}

/// Solves Lasso Regression using Coordinate Descent:
/// min (1/2N) * ||Xw + b - y||_2^2 + alpha * ||w||_1
pub fn solveLasso(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    n_samples: usize,
    n_features: usize,
    alpha: f32,
    max_iter: usize,
    tol: f32,
) !ModelResult {
    std.debug.assert(x.len == n_samples * n_features);
    std.debug.assert(y.len == n_samples);

    const N = n_samples;
    const D = n_features;
    const N_f = @as(f32, @floatFromInt(N));

    const mean_x = try allocator.alloc(f32, D);
    defer allocator.free(mean_x);
    @memset(mean_x, 0.0);

    var sum_y: f32 = 0.0;
    for (0..N) |i| {
        sum_y += y[i];
        for (0..D) |j| {
            mean_x[j] += x[i * D + j];
        }
    }
    const mean_y = sum_y / N_f;
    for (0..D) |j| {
        mean_x[j] /= N_f;
    }

    const X_c = try allocator.alloc(f32, N * D);
    defer allocator.free(X_c);
    const y_c = try allocator.alloc(f32, N);
    defer allocator.free(y_c);

    for (0..N) |i| {
        y_c[i] = y[i] - mean_y;
        for (0..D) |j| {
            X_c[i * D + j] = x[i * D + j] - mean_x[j];
        }
    }

    const L = try allocator.alloc(f32, D);
    defer allocator.free(L);
    for (0..D) |j| {
        var sum_sq: f32 = 0.0;
        for (0..N) |i| {
            const val = X_c[i * D + j];
            sum_sq += val * val;
        }
        L[j] = sum_sq / N_f;
    }

    const weights = try allocator.alloc(f32, D);
    @memset(weights, 0.0);

    const r = try allocator.alloc(f32, N);
    defer allocator.free(r);
    @memcpy(r, y_c);

    for (0..max_iter) |_| {
        var max_diff: f32 = 0.0;

        for (0..D) |j| {
            if (L[j] < 1e-12) continue;

            var dot_xr: f32 = 0.0;
            for (0..N) |i| {
                dot_xr += X_c[i * D + j] * r[i];
            }
            const c_j = (dot_xr / N_f) + L[j] * weights[j];
            const w_new = softThreshold(c_j, alpha) / L[j];

            const diff = w_new - weights[j];
            if (@abs(diff) > 0.0) {
                for (0..N) |i| {
                    r[i] -= X_c[i * D + j] * diff;
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
    for (0..D) |j| {
        dot_mw += mean_x[j] * weights[j];
    }
    const intercept = mean_y - dot_mw;

    return ModelResult{
        .weights = weights,
        .intercept = intercept,
        .allocator = allocator,
    };
}

/// Solves Elastic Net Regression using Coordinate Descent:
/// min (1/2N) * ||Xw + b - y||_2^2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||_2^2
pub fn solveElasticNet(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    n_samples: usize,
    n_features: usize,
    alpha: f32,
    l1_ratio: f32,
    max_iter: usize,
    tol: f32,
) !ModelResult {
    std.debug.assert(x.len == n_samples * n_features);
    std.debug.assert(y.len == n_samples);

    const N = n_samples;
    const D = n_features;
    const N_f = @as(f32, @floatFromInt(N));

    const mean_x = try allocator.alloc(f32, D);
    defer allocator.free(mean_x);
    @memset(mean_x, 0.0);

    var sum_y: f32 = 0.0;
    for (0..N) |i| {
        sum_y += y[i];
        for (0..D) |j| {
            mean_x[j] += x[i * D + j];
        }
    }
    const mean_y = sum_y / N_f;
    for (0..D) |j| {
        mean_x[j] /= N_f;
    }

    const X_c = try allocator.alloc(f32, N * D);
    defer allocator.free(X_c);
    const y_c = try allocator.alloc(f32, N);
    defer allocator.free(y_c);

    for (0..N) |i| {
        y_c[i] = y[i] - mean_y;
        for (0..D) |j| {
            X_c[i * D + j] = x[i * D + j] - mean_x[j];
        }
    }

    const L = try allocator.alloc(f32, D);
    defer allocator.free(L);
    for (0..D) |j| {
        var sum_sq: f32 = 0.0;
        for (0..N) |i| {
            const val = X_c[i * D + j];
            sum_sq += val * val;
        }
        L[j] = sum_sq / N_f;
    }

    const l1_penalty = alpha * l1_ratio;
    const l2_penalty = alpha * (1.0 - l1_ratio);

    const weights = try allocator.alloc(f32, D);
    @memset(weights, 0.0);

    const r = try allocator.alloc(f32, N);
    defer allocator.free(r);
    @memcpy(r, y_c);

    for (0..max_iter) |_| {
        var max_diff: f32 = 0.0;

        for (0..D) |j| {
            const denom = L[j] + l2_penalty;
            if (denom < 1e-12) continue;

            var dot_xr: f32 = 0.0;
            for (0..N) |i| {
                dot_xr += X_c[i * D + j] * r[i];
            }
            const c_j = (dot_xr / N_f) + L[j] * weights[j];
            const w_new = softThreshold(c_j, l1_penalty) / denom;

            const diff = w_new - weights[j];
            if (@abs(diff) > 0.0) {
                for (0..N) |i| {
                    r[i] -= X_c[i * D + j] * diff;
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
    for (0..D) |j| {
        dot_mw += mean_x[j] * weights[j];
    }
    const intercept = mean_y - dot_mw;

    return ModelResult{
        .weights = weights,
        .intercept = intercept,
        .allocator = allocator,
    };
}

/// Ridge Regression Model Struct
pub const RidgeRegression = struct {
    alpha: f32 = 1.0,
    model: ?ModelResult = null,

    pub fn init(alpha: f32) RidgeRegression {
        return .{ .alpha = alpha };
    }

    pub fn deinit(self: *RidgeRegression) void {
        if (self.model) |*m| {
            m.deinit();
            self.model = null;
        }
    }

    pub fn fit(self: *RidgeRegression, allocator: std.mem.Allocator, X: []const f32, y: []const f32, n_samples: usize, n_features: usize) !void {
        self.deinit();
        self.model = try solveRidge(allocator, X, y, n_samples, n_features, self.alpha);
    }

    pub fn predict(self: RidgeRegression, x_row: []const f32) f32 {
        if (self.model) |m| {
            return m.predict(x_row);
        }
        @panic("RidgeRegression model has not been fitted yet");
    }
};

/// Lasso Regression Model Struct
pub const LassoRegression = struct {
    alpha: f32 = 1.0,
    max_iter: usize = 1000,
    tol: f32 = 1e-4,
    model: ?ModelResult = null,

    pub fn init(alpha: f32, max_iter: usize, tol: f32) LassoRegression {
        return .{
            .alpha = alpha,
            .max_iter = max_iter,
            .tol = tol,
        };
    }

    pub fn deinit(self: *LassoRegression) void {
        if (self.model) |*m| {
            m.deinit();
            self.model = null;
        }
    }

    pub fn fit(self: *LassoRegression, allocator: std.mem.Allocator, X: []const f32, y: []const f32, n_samples: usize, n_features: usize) !void {
        self.deinit();
        self.model = try solveLasso(allocator, X, y, n_samples, n_features, self.alpha, self.max_iter, self.tol);
    }

    pub fn predict(self: LassoRegression, x_row: []const f32) f32 {
        if (self.model) |m| {
            return m.predict(x_row);
        }
        @panic("LassoRegression model has not been fitted yet");
    }
};

/// Elastic Net Regression Model Struct
pub const ElasticNetRegression = struct {
    alpha: f32 = 1.0,
    l1_ratio: f32 = 0.5,
    max_iter: usize = 1000,
    tol: f32 = 1e-4,
    model: ?ModelResult = null,

    pub fn init(alpha: f32, l1_ratio: f32, max_iter: usize, tol: f32) ElasticNetRegression {
        return .{
            .alpha = alpha,
            .l1_ratio = l1_ratio,
            .max_iter = max_iter,
            .tol = tol,
        };
    }

    pub fn deinit(self: *ElasticNetRegression) void {
        if (self.model) |*m| {
            m.deinit();
            self.model = null;
        }
    }

    pub fn fit(self: *ElasticNetRegression, allocator: std.mem.Allocator, X: []const f32, y: []const f32, n_samples: usize, n_features: usize) !void {
        self.deinit();
        self.model = try solveElasticNet(allocator, X, y, n_samples, n_features, self.alpha, self.l1_ratio, self.max_iter, self.tol);
    }

    pub fn predict(self: ElasticNetRegression, x_row: []const f32) f32 {
        if (self.model) |m| {
            return m.predict(x_row);
        }
        @panic("ElasticNetRegression model has not been fitted yet");
    }
};

test "softThreshold unit test" {
    try std.testing.expectEqual(@as(f32, 2.0), softThreshold(3.0, 1.0));
    try std.testing.expectEqual(@as(f32, -2.0), softThreshold(-3.0, 1.0));
    try std.testing.expectEqual(@as(f32, 0.0), softThreshold(0.5, 1.0));
    try std.testing.expectEqual(@as(f32, 0.0), softThreshold(-0.5, 1.0));
}

test "Ridge and Lasso solvers" {
    const allocator = std.testing.allocator;
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y = [_]f32{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    var ridge = try solveRidge(allocator, &x, &y, 5, 1, 0.0);
    defer ridge.deinit();
    try std.testing.expect(@abs(ridge.weights[0] - 2.0) < 1e-3);
    try std.testing.expect(@abs(ridge.intercept - 0.0) < 1e-3);

    var lasso = try solveLasso(allocator, &x, &y, 5, 1, 0.01, 500, 1e-5);
    defer lasso.deinit();
    try std.testing.expect(@abs(lasso.weights[0] - 2.0) < 0.05);

    var enet = try solveElasticNet(allocator, &x, &y, 5, 1, 0.01, 0.5, 500, 1e-5);
    defer enet.deinit();
    try std.testing.expect(@abs(enet.weights[0] - 2.0) < 0.05);
}
