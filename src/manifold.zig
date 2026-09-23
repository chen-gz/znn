const std = @import("std");

/// Configuration options for t-SNE algorithm
pub const TSNEOptions = struct {
    n_components: usize = 2,
    perplexity: f32 = 30.0,
    n_iter: usize = 1000,
    lr: f32 = 200.0,
    early_exaggeration: f32 = 4.0,
    early_exaggeration_iter: usize = 250,
    min_gain: f32 = 0.01,
    seed: u64 = 42,

    /// Default configuration for t-SNE (2D embedding, perplexity 30, lr 200, 1000 iterations)
    pub const default: TSNEOptions = .{};

    /// Callable function to obtain default TSNEOptions
    pub fn defaultOptions() TSNEOptions {
        return .{};
    }
};

/// Normal random generator using Box-Muller transform
pub fn normalRandom(random: std.Random) f32 {
    var u_1: f32 = random.float(f32);
    while (u_1 == 0.0) {
        u_1 = random.float(f32);
    }
    const u_2: f32 = random.float(f32);
    return @sqrt(-2.0 * @log(u_1)) * @cos(2.0 * std.math.pi * u_2);
}

/// Compute pairwise squared Euclidean distances: D_ij = ||x_i - x_j||^2
/// Returns an N x N row-major slice allocated with `allocator`.
pub fn computePairwiseDistances(allocator: std.mem.Allocator, X: []const f32, N: usize, D: usize) ![]f32 {
    std.debug.assert(X.len == N * D);
    const dist = try allocator.alloc(f32, N * N);

    for (0..N) |i| {
        dist[i * N + i] = 0.0;
        const row_i = X[i * D .. (i + 1) * D];
        for (i + 1..N) |j| {
            const row_j = X[j * D .. (j + 1) * D];
            var d_sq: f32 = 0.0;
            for (0..D) |k| {
                const diff = row_i[k] - row_j[k];
                d_sq += diff * diff;
            }
            dist[i * N + j] = d_sq;
            dist[j * N + i] = d_sq;
        }
    }
    return dist;
}

/// Binary search on Gaussian precision beta_i = 1 / (2 * sigma_i^2)
/// such that Shannon entropy matches log2(target_perplexity).
/// Returns an N x N conditional affinity matrix P_cond (p_{j|i}).
pub fn binarySearchPerplexity(
    allocator: std.mem.Allocator,
    dist_sq: []const f32,
    N: usize,
    target_perplexity: f32,
    tol: f32,
    max_iter: usize,
) ![]f32 {
    const P = try allocator.alloc(f32, N * N);
    @memset(P, 0.0);

    const target_entropy = @log2(target_perplexity);
    const ln2 = @log(2.0);

    // Scratch buffers for probabilities per sample
    const p_row = try allocator.alloc(f32, N);
    defer allocator.free(p_row);

    for (0..N) |i| {
        var beta_min: f32 = -std.math.inf(f32);
        var beta_max: f32 = std.math.inf(f32);
        var beta: f32 = 1.0;

        const row_dist = dist_sq[i * N .. (i + 1) * N];

        for (0..max_iter) |_| {
            var sum_p: f32 = 0.0;
            var sum_dp: f32 = 0.0;

            for (0..N) |j| {
                if (j == i) {
                    p_row[j] = 0.0;
                } else {
                    const pj = @exp(-row_dist[j] * beta);
                    p_row[j] = pj;
                    sum_p += pj;
                    sum_dp += row_dist[j] * pj;
                }
            }

            if (sum_p == 0.0) sum_p = 1e-12;

            // Shannon entropy H = log2(sum_p) + (beta / ln(2)) * (sum_dp / sum_p)
            const H = @log2(sum_p) + (beta / ln2) * (sum_dp / sum_p);
            const p_diff = H - target_entropy;

            if (@abs(p_diff) < tol) break;

            if (p_diff > 0.0) {
                beta_min = beta;
                if (std.math.isInf(beta_max)) {
                    beta *= 2.0;
                } else {
                    beta = (beta + beta_max) * 0.5;
                }
            } else {
                beta_max = beta;
                if (std.math.isInf(beta_min)) {
                    beta *= 0.5;
                } else {
                    beta = (beta + beta_min) * 0.5;
                }
            }
        }

        // Final normalization for sample i
        var sum_final: f32 = 0.0;
        for (0..N) |j| {
            if (j == i) {
                p_row[j] = 0.0;
            } else {
                const pj = @exp(-row_dist[j] * beta);
                p_row[j] = pj;
                sum_final += pj;
            }
        }
        if (sum_final == 0.0) sum_final = 1e-12;

        for (0..N) |j| {
            P[i * N + j] = if (j == i) 0.0 else p_row[j] / sum_final;
        }
    }

    return P;
}

/// Symmetrize conditional probabilities: P_ij = (P_{j|i} + P_{i|j}) / (2 * N)
pub fn symmetrizeAffinities(P_cond: []const f32, N: usize, out_P: []f32) void {
    const inv_2n = 1.0 / (2.0 * @as(f32, @floatFromInt(N)));
    for (0..N) |i| {
        out_P[i * N + i] = 0.0;
        for (i + 1..N) |j| {
            const p_sym = (P_cond[i * N + j] + P_cond[j * N + i]) * inv_2n;
            out_P[i * N + j] = p_sym;
            out_P[j * N + i] = p_sym;
        }
    }
}

/// t-SNE algorithm implementation
pub const TSNE = struct {
    options: TSNEOptions,

    pub fn init(options: TSNEOptions) TSNE {
        return .{ .options = options };
    }

    /// Initialize TSNE model with default options
    pub fn initDefault() TSNE {
        return init(TSNEOptions.default);
    }

    /// Fits the t-SNE model on data X (N x D) and returns embedded coordinates Y (N x n_components).
    pub fn fitTransform(self: TSNE, allocator: std.mem.Allocator, X: []const f32, N: usize, D: usize) ![]f32 {
        const d_out = self.options.n_components;
        std.debug.assert(X.len == N * D);

        // Stage 1: Compute pairwise distances and high-dimensional affinities
        const dist_sq = try computePairwiseDistances(allocator, X, N, D);
        defer allocator.free(dist_sq);

        const P_cond = try binarySearchPerplexity(allocator, dist_sq, N, self.options.perplexity, 1e-5, 50);
        defer allocator.free(P_cond);

        const P = try allocator.alloc(f32, N * N);
        defer allocator.free(P);
        symmetrizeAffinities(P_cond, N, P);

        // Early exaggeration
        for (P) |*p_val| {
            p_val.* = @max(p_val.* * self.options.early_exaggeration, 1e-12);
        }

        // Stage 2: Initialize low-dimensional embedding Y ~ N(0, 1e-4)
        var prng = std.Random.DefaultPrng.init(self.options.seed);
        const random = prng.random();

        const Y = try allocator.alloc(f32, N * d_out);
        const Y_prev = try allocator.alloc(f32, N * d_out);
        defer allocator.free(Y_prev);

        const gains = try allocator.alloc(f32, N * d_out);
        defer allocator.free(gains);

        for (0..N * d_out) |idx| {
            const init_val = normalRandom(random) * 1e-4;
            Y[idx] = init_val;
            Y_prev[idx] = init_val;
            gains[idx] = 1.0;
        }

        // Scratch buffers for optimization loop
        const num = try allocator.alloc(f32, N * N);
        defer allocator.free(num);

        const Q = try allocator.alloc(f32, N * N);
        defer allocator.free(Q);

        const grad = try allocator.alloc(f32, N * d_out);
        defer allocator.free(grad);

        const dY = try allocator.alloc(f32, N * d_out);
        defer allocator.free(dY);
        @memset(dY, 0.0);

        // Stages 3-5: Gradient descent optimization loop
        for (0..self.options.n_iter) |step| {
            // Remove early exaggeration after specified iterations
            if (step == self.options.early_exaggeration_iter) {
                const inv_exag = 1.0 / self.options.early_exaggeration;
                for (P) |*p_val| {
                    p_val.* *= inv_exag;
                }
            }

            // Stage 3: Low-dimensional Student-t kernel similarities
            var num_sum: f32 = 0.0;
            for (0..N) |i| {
                num[i * N + i] = 0.0;
                const yi = Y[i * d_out .. (i + 1) * d_out];
                for (i + 1..N) |j| {
                    const yj = Y[j * d_out .. (j + 1) * d_out];
                    var dist_y_sq: f32 = 0.0;
                    for (0..d_out) |c| {
                        const diff = yi[c] - yj[c];
                        dist_y_sq += diff * diff;
                    }
                    const w = 1.0 / (1.0 + dist_y_sq);
                    num[i * N + j] = w;
                    num[j * N + i] = w;
                    num_sum += 2.0 * w;
                }
            }

            if (num_sum == 0.0) num_sum = 1e-12;
            const inv_num_sum = 1.0 / num_sum;

            for (0..N) |i| {
                Q[i * N + i] = 0.0;
                for (0..N) |j| {
                    if (i != j) {
                        Q[i * N + j] = @max(num[i * N + j] * inv_num_sum, 1e-12);
                    }
                }
            }

            // Stage 4: Gradient computation: dC/dy_i = 4 * sum_j (p_ij - q_ij) * w_ij * (y_i - y_j)
            @memset(grad, 0.0);
            for (0..N) |i| {
                const yi = Y[i * d_out .. (i + 1) * d_out];
                const grad_i = grad[i * d_out .. (i + 1) * d_out];

                for (0..N) |j| {
                    if (i == j) continue;
                    const mult = 4.0 * (P[i * N + j] - Q[i * N + j]) * num[i * N + j];
                    const yj = Y[j * d_out .. (j + 1) * d_out];
                    for (0..d_out) |c| {
                        grad_i[c] += mult * (yi[c] - yj[c]);
                    }
                }
            }

            // Stage 5: Momentum and adaptive learning rate update
            const momentum: f32 = if (step < 250) 0.5 else 0.8;

            for (0..N * d_out) |idx| {
                const g = grad[idx];
                const delta = Y[idx] - Y_prev[idx];

                // Adaptive gain schedule: increase gain if direction changes, decrease if consistent
                if ((g > 0.0) != (delta > 0.0)) {
                    gains[idx] += 0.2;
                } else {
                    gains[idx] = @max(gains[idx] * 0.8, self.options.min_gain);
                }

                // Momentum step: dY = momentum * delta - lr * gains * grad
                const step_dY = momentum * delta - self.options.lr * gains[idx] * g;
                dY[idx] = step_dY;
                Y_prev[idx] = Y[idx];
                Y[idx] += step_dY;
            }

            // Zero-center the embedding coordinates Y
            for (0..d_out) |c| {
                var mean_c: f32 = 0.0;
                for (0..N) |i| {
                    mean_c += Y[i * d_out + c];
                }
                mean_c /= @as(f32, @floatFromInt(N));
                for (0..N) |i| {
                    Y[i * d_out + c] -= mean_c;
                }
            }
        }

        return Y;
    }
};

/// High-level convenience function to run t-SNE
pub fn tsne(allocator: std.mem.Allocator, X: []const f32, N: usize, D: usize, options: TSNEOptions) ![]f32 {
    const model = TSNE.init(options);
    return model.fitTransform(allocator, X, N, D);
}

/// Convenience function to run t-SNE with default options (TSNEOptions.default)
pub fn tsneDefault(allocator: std.mem.Allocator, X: []const f32, N: usize, D: usize) ![]f32 {
    return tsne(allocator, X, N, D, TSNEOptions.default);
}

test "t-SNE pairwise distances computation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const X = [_]f32{
        0.0, 0.0,
        3.0, 4.0,
        1.0, 1.0,
    };
    const dist = try computePairwiseDistances(allocator, &X, 3, 2);
    defer allocator.free(dist);

    try testing.expectApproxEqAbs(@as(f32, 0.0), dist[0 * 3 + 0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 25.0), dist[0 * 3 + 1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), dist[0 * 3 + 2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 25.0), dist[1 * 3 + 0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 13.0), dist[1 * 3 + 2], 1e-5);
}

test "t-SNE perplexity search and affinity symmetrization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const X = [_]f32{
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        5.0, 5.0,
        5.0, 6.0,
    };
    const dist = try computePairwiseDistances(allocator, &X, 5, 2);
    defer allocator.free(dist);

    const P_cond = try binarySearchPerplexity(allocator, dist, 5, 2.0, 1e-4, 50);
    defer allocator.free(P_cond);

    // Check each conditional distribution row sums to approximately 1.0
    for (0..5) |i| {
        var row_sum: f32 = 0.0;
        for (0..5) |j| {
            row_sum += P_cond[i * 5 + j];
        }
        try testing.expectApproxEqAbs(@as(f32, 1.0), row_sum, 1e-3);
    }

    const P = try allocator.alloc(f32, 25);
    defer allocator.free(P);
    symmetrizeAffinities(P_cond, 5, P);

    // Check symmetry and global normalization
    var total_sum: f32 = 0.0;
    for (0..5) |i| {
        for (0..5) |j| {
            try testing.expectApproxEqAbs(P[i * 5 + j], P[j * 5 + i], 1e-5);
            total_sum += P[i * 5 + j];
        }
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), total_sum, 1e-3);
}

test "t-SNE fitTransform on synthetic cluster points" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 6 points in 3 distinct clusters: (0,0), (10,10), (-10,-10)
    const X = [_]f32{
        0.0, 0.1,
        0.1, 0.0,
        10.0, 10.1,
        10.1, 10.0,
        -10.0, -9.9,
        -9.9, -10.0,
    };

    const options = TSNEOptions{
        .n_components = 2,
        .perplexity = 2.0,
        .n_iter = 100,
        .early_exaggeration_iter = 30,
        .seed = 42,
    };

    const Y = try tsne(allocator, &X, 6, 2, options);
    defer allocator.free(Y);

    try testing.expectEqual(@as(usize, 12), Y.len);

    // Check all values are finite (no NaN or Inf)
    for (Y) |y_val| {
        try testing.expect(!std.math.isNan(y_val));
        try testing.expect(!std.math.isInf(y_val));
    }

    // Check coordinates are zero-centered
    var mean_x: f32 = 0.0;
    var mean_y: f32 = 0.0;
    for (0..6) |i| {
        mean_x += Y[i * 2 + 0];
        mean_y += Y[i * 2 + 1];
    }
    try testing.expectApproxEqAbs(@as(f32, 0.0), mean_x / 6.0, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.0), mean_y / 6.0, 1e-4);
}
