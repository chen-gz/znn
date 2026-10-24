const std = @import("std");
const tensor = @import("tensor.zig");
const dot = @import("backend.zig").vec_math.dot;

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

/// Result of fitting a 1D linear regression model analytically
pub const FitResult = struct {
    w: f32,
    b: f32,
};

/// Solve 1D linear regression analytically using least-squares closed-form formula:
/// w = cov(x, y) / var(x)
/// b = mean_y - w * mean_x
pub fn solveAnalytical(x: []const f32, y: []const f32) FitResult {
    std.debug.assert(x.len == y.len);
    std.debug.assert(x.len > 0);
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

/// Result of centering the dataset
const CenterResult = struct {
    mean_x: []f32,
    mean_y: f32,
    /// Centered features stored column-major (feature j occupies `X_c[j * N .. (j + 1) * N]`),
    /// so the per-feature dot products and residual updates of the solvers walk contiguous memory.
    X_c: []f32,
    y_c: []f32,

    pub fn deinit(self: *CenterResult, allocator: std.mem.Allocator) void {
        allocator.free(self.mean_x);
        allocator.free(self.X_c);
        allocator.free(self.y_c);
    }

    fn column(self: CenterResult, j: usize, N: usize) []const f32 {
        return self.X_c[j * N .. (j + 1) * N];
    }
};

/// Helper function to compute means and center the dataset for regression solvers.
/// `x` is row-major (N x D); the centered copy is returned column-major (see `CenterResult.X_c`).
fn centerData(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    N: usize,
    D: usize,
) !CenterResult {
    const N_f = @as(f32, @floatFromInt(N));

    const mean_x = try allocator.alloc(f32, D);
    errdefer allocator.free(mean_x);
    @memset(mean_x, 0.0);

    var sum_y: f32 = 0.0;
    for (0..N) |i| {
        sum_y += y[i];
        const row = x[i * D .. (i + 1) * D];
        for (mean_x, row) |*m, v| m.* += v;
    }
    const mean_y = sum_y / N_f;
    for (mean_x) |*m| m.* /= N_f;

    const X_c = try allocator.alloc(f32, N * D);
    errdefer allocator.free(X_c);

    const y_c = try allocator.alloc(f32, N);
    errdefer allocator.free(y_c);

    for (0..N) |i| {
        y_c[i] = y[i] - mean_y;
        const row = x[i * D .. (i + 1) * D];
        for (0..D) |j| {
            X_c[j * N + i] = row[j] - mean_x[j];
        }
    }

    return CenterResult{
        .mean_x = mean_x,
        .mean_y = mean_y,
        .X_c = X_c,
        .y_c = y_c,
    };
}

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

    var center_res = try centerData(allocator, x, y, N, D);
    defer center_res.deinit(allocator);
    const mean_x = center_res.mean_x;
    const mean_y = center_res.mean_y;

    // 3. Compute normal matrix M = (1/N) * X_c^T X_c + alpha * I, and v = (1/N) * X_c^T y_c.
    //    M is symmetric: compute the upper triangle from contiguous columns and mirror it.
    const M = try allocator.alloc(f32, D * D);
    defer allocator.free(M);

    const v = try allocator.alloc(f32, D);
    defer allocator.free(v);

    for (0..D) |j1| {
        const col1 = center_res.column(j1, N);
        for (j1..D) |j2| {
            const m = dot(col1, center_res.column(j2, N)) / N_f;
            M[j1 * D + j2] = m;
            M[j2 * D + j1] = m;
        }
        M[j1 * D + j1] += alpha;
        v[j1] = dot(col1, center_res.y_c) / N_f;
    }

    // 4. Solve linear system M * w = v
    const weights = try allocator.alloc(f32, D);
    errdefer allocator.free(weights);
    try tensor.solveLinearSystem(allocator, M, v, D, weights);

    // 5. Intercept: b = mean_y - mean_x^T * w
    const intercept = mean_y - dot(mean_x, weights);

    return ModelResult{
        .weights = weights,
        .intercept = intercept,
        .allocator = allocator,
    };
}

/// Solves Ordinary Least Squares (OLS) (alpha = 0)
pub fn solveOLS(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    n_samples: usize,
    n_features: usize,
) !ModelResult {
    return solveRidge(allocator, x, y, n_samples, n_features, 0.0);
}

/// Solves Least Absolute Shrinkage and Selection Operator (LASSO) Regression using Coordinate Descent:
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
    return solveElasticNet(allocator, x, y, n_samples, n_features, alpha, 1.0, max_iter, tol);
}

/// Solves Elastic Net Regression using Coordinate Descent:
/// min (1/2N) * ||Xw + b - y||_2^2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||_2^2
///
/// Convergence: stops once the largest coordinate update of a sweep is below
/// `tol * max(1, max_j |w_j|)`, i.e. `tol` is absolute for small weights and relative for large
/// ones (an absolute threshold alone can be unreachable in f32 when the weights are large).
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

    var center_res = try centerData(allocator, x, y, N, D);
    defer center_res.deinit(allocator);
    const mean_x = center_res.mean_x;
    const mean_y = center_res.mean_y;

    const L = try allocator.alloc(f32, D);
    defer allocator.free(L);
    for (0..D) |j| {
        const col = center_res.column(j, N);
        L[j] = dot(col, col) / N_f;
    }

    const l1_penalty = alpha * l1_ratio;
    const l2_penalty = alpha * (1.0 - l1_ratio);

    const r = try allocator.alloc(f32, N);
    defer allocator.free(r);
    @memcpy(r, center_res.y_c);

    const weights = try allocator.alloc(f32, D);
    errdefer allocator.free(weights);
    @memset(weights, 0.0);

    for (0..max_iter) |_| {
        var max_diff: f32 = 0.0;
        var max_weight: f32 = 0.0;

        for (0..D) |j| {
            const denom = L[j] + l2_penalty;
            if (denom < 1e-12) continue;

            const col = center_res.column(j, N);
            const c_j = (dot(col, r) / N_f) + L[j] * weights[j];
            const w_new = softThreshold(c_j, l1_penalty) / denom;

            const diff = w_new - weights[j];
            if (diff != 0.0) {
                for (r, col) |*r_i, x_ij| {
                    r_i.* -= x_ij * diff;
                }
                weights[j] = w_new;
                max_diff = @max(max_diff, @abs(diff));
            }
            max_weight = @max(max_weight, @abs(weights[j]));
        }

        if (max_diff < tol * @max(max_weight, 1.0)) break;
    }

    const intercept = mean_y - dot(mean_x, weights);

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

    pub const default: RidgeRegression = .{};
    pub fn defaultOptions() RidgeRegression {
        return .{};
    }

    pub fn initDefault() RidgeRegression {
        return .{};
    }

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

    pub const default: LassoRegression = .{};
    pub fn defaultOptions() LassoRegression {
        return .{};
    }

    pub fn initDefault() LassoRegression {
        return .{};
    }

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

    pub const default: ElasticNetRegression = .{};
    pub fn defaultOptions() ElasticNetRegression {
        return .{};
    }

    pub fn initDefault() ElasticNetRegression {
        return .{};
    }

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

test "solveAnalytical unit test" {
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y = [_]f32{ 3.0, 5.0, 7.0, 9.0, 11.0 }; // y = 2x + 1

    const res = solveAnalytical(&x, &y);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), res.w, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), res.b, 1e-5);
}

test "solveRidge and solveElasticNet recover multi-feature coefficients" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const random = prng.random();

    const N = 64;
    const D = 3;
    var x: [N * D]f32 = undefined;
    var y: [N]f32 = undefined;
    for (0..N) |i| {
        for (0..D) |j| x[i * D + j] = random.floatNorm(f32);
        y[i] = 1.5 * x[i * D] - 2.0 * x[i * D + 1] + 0.5 * x[i * D + 2] + 3.0;
    }

    var ols = try solveOLS(allocator, &x, &y, N, D);
    defer ols.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), ols.weights[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), ols.weights[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), ols.weights[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), ols.intercept, 1e-4);

    var enet = try solveElasticNet(allocator, &x, &y, N, D, 1e-4, 0.5, 2000, 1e-7);
    defer enet.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), enet.weights[0], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), enet.weights[1], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), enet.weights[2], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), enet.intercept, 1e-2);
}

test "solveRidge frees the weights buffer when the normal matrix is singular" {
    // 第二个特征为常数：中心化后整列为 0，alpha = 0 时正规方程矩阵奇异；testing.allocator 会检测泄漏
    const x = [_]f32{ 1, 5, 2, 5, 3, 5, 4, 5 };
    const y = [_]f32{ 1, 2, 3, 4 };
    try std.testing.expectError(error.SingularMatrix, solveOLS(std.testing.allocator, &x, &y, 4, 2));
}

fn fitElasticNetForAllocationFailures(allocator: std.mem.Allocator) !void {
    const x = [_]f32{ 1, 0, 2, 1, 3, 1, 4, 3 };
    const y = [_]f32{ 1, 2, 3, 4 };
    var model = try solveElasticNet(allocator, &x, &y, 4, 2, 0.01, 0.5, 50, 1e-4);
    model.deinit();
}

test "solveElasticNet does not leak on allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, fitElasticNetForAllocationFailures, .{});
}
