const std = @import("std");
const zig_ml = @import("zig_ml");
const tensor = zig_ml.tensor;

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

/// Gaussian elimination solver for A * x = b (P x P)
pub fn solveLinearSystem(allocator: std.mem.Allocator, P: usize, A_in: []const f32, b_in: []const f32) ![]f32 {
    const A = try allocator.alloc(f32, P * P);
    defer allocator.free(A);
    @memcpy(A, A_in);

    const b = try allocator.alloc(f32, P);
    defer allocator.free(b);
    @memcpy(b, b_in);

    const x = try allocator.alloc(f32, P);

    // Forward elimination with partial pivoting
    for (0..P) |i| {
        // Find pivot
        var max_row = i;
        var max_val = @abs(A[i * P + i]);
        for ((i + 1)..P) |k| {
            const val = @abs(A[k * P + i]);
            if (val > max_val) {
                max_val = val;
                max_row = k;
            }
        }

        if (max_row != i) {
            // Swap rows in A and b
            for (0..P) |col| {
                const tmp = A[i * P + col];
                A[i * P + col] = A[max_row * P + col];
                A[max_row * P + col] = tmp;
            }
            const tmp_b = b[i];
            b[i] = b[max_row];
            b[max_row] = tmp_b;
        }

        const pivot = A[i * P + i];
        if (@abs(pivot) < 1e-9) {
            return error.SingularMatrix;
        }

        for ((i + 1)..P) |k| {
            const factor = A[k * P + i] / pivot;
            for (i..P) |col| {
                A[k * P + col] -= factor * A[i * P + col];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    var i_idx: usize = P;
    while (i_idx > 0) {
        i_idx -= 1;
        const i = i_idx;
        var sum: f32 = b[i];
        for ((i + 1)..P) |j| {
            sum -= A[i * P + j] * x[j];
        }
        x[i] = sum / A[i * P + i];
    }

    return x;
}

/// Ridge Regression solver: min 1/(2N) ||y - Xw - b||^2 + (alpha/2) ||w||_2^2
pub fn solveRidge(
    allocator: std.mem.Allocator,
    N: usize,
    P: usize,
    X: []const f32,
    y: []const f32,
    alpha: f32,
) !ModelResult {
    // 1. Calculate means
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

    // 2. Center X and y
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

    // 3. Compute A = (1/N) * X_c^T X_c + alpha * I, b_vec = (1/N) * X_c^T y_c
    const A = try allocator.alloc(f32, P * P);
    defer allocator.free(A);
    @memset(A, 0.0);

    const b_vec = try allocator.alloc(f32, P);
    defer allocator.free(b_vec);
    @memset(b_vec, 0.0);

    const n_f = @as(f32, @floatFromInt(N));
    for (0..P) |j1| {
        for (0..P) |j2| {
            var dot: f32 = 0.0;
            for (0..N) |i| {
                dot += X_c[i * P + j1] * X_c[i * P + j2];
            }
            A[j1 * P + j2] = dot / n_f;
        }
        A[j1 * P + j1] += alpha; // Add L2 penalty to diagonal

        var dot_y: f32 = 0.0;
        for (0..N) |i| {
            dot_y += X_c[i * P + j1] * y_c[i];
        }
        b_vec[j1] = dot_y / n_f;
    }

    // 4. Solve linear system A * w = b_vec
    const weights = try solveLinearSystem(allocator, P, A, b_vec);

    // 5. Compute intercept: b = mean_y - mean_x^T * w
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

/// Lasso Regression solver using Coordinate Descent: min 1/(2N) ||y - Xw - b||^2 + alpha * ||w||_1
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
    // 1. Calculate means
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

    // 2. Center X and y
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

    // 3. Precompute L_j = (1/N) * sum_i (X_ij^2)
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

    // 4. Initialize weights and residual
    const weights = try allocator.alloc(f32, P);
    @memset(weights, 0.0);

    const r = try allocator.alloc(f32, N);
    defer allocator.free(r);
    @memcpy(r, y_c); // Initial residual = y_c - X_c * 0 = y_c

    // 5. Coordinate descent iterations
    for (0..max_iter) |_| {
        var max_diff: f32 = 0.0;

        for (0..P) |j| {
            if (L[j] < 1e-12) continue;

            // dot(X[:, j], r)
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

/// Elastic Net Regression solver using Coordinate Descent:
/// min 1/(2N) ||y - Xw - b||^2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||_2^2
pub fn solveElasticNet(
    allocator: std.mem.Allocator,
    N: usize,
    P: usize,
    X: []const f32,
    y: []const f32,
    alpha: f32,
    l1_ratio: f32,
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

    const l1_penalty = alpha * l1_ratio;
    const l2_penalty = alpha * (1.0 - l1_ratio);

    const weights = try allocator.alloc(f32, P);
    @memset(weights, 0.0);

    const r = try allocator.alloc(f32, N);
    defer allocator.free(r);
    @memcpy(r, y_c);

    for (0..max_iter) |_| {
        var max_diff: f32 = 0.0;

        for (0..P) |j| {
            const denom = L[j] + l2_penalty;
            if (denom < 1e-12) continue;

            var dot_xr: f32 = 0.0;
            for (0..N) |i| {
                dot_xr += X_c[i * P + j] * r[i];
            }
            const c_j = (dot_xr / n_f) + L[j] * weights[j];
            const w_new = softThreshold(c_j, l1_penalty) / denom;

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

    // 1. Solve Ridge
    var ridge = try solveRidge(allocator, N, P, X, y, 0.1);
    defer ridge.deinit();

    // 2. Solve Lasso
    var lasso = try solveLasso(allocator, N, P, X, y, 0.1, 1000, 1e-5);
    defer lasso.deinit();

    // 3. Solve Elastic Net
    var enet = try solveElasticNet(allocator, N, P, X, y, 0.1, 0.5, 1000, 1e-5);
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
