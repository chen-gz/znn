const std = @import("std");

pub const CenteredData = struct {
    mean_x: []f32,
    mean_y: f32,
    X_c: []f32,
    y_c: []f32,

    pub fn deinit(self: CenteredData, allocator: std.mem.Allocator) void {
        allocator.free(self.mean_x);
        allocator.free(self.X_c);
        allocator.free(self.y_c);
    }
};

/// Computes the column-wise means of X, the mean of y, and centers both X and y.
/// Returns a CenteredData struct containing the means and the centered data.
pub fn computeCenteredData(allocator: std.mem.Allocator, x: []const f32, y: []const f32, n_samples: usize, n_features: usize) !CenteredData {
    const N = n_samples;
    const D = n_features;
    const N_f = @as(f32, @floatFromInt(N));

    const mean_x = try allocator.alloc(f32, D);
    errdefer allocator.free(mean_x);
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
    errdefer allocator.free(X_c);
    const y_c = try allocator.alloc(f32, N);
    errdefer allocator.free(y_c);

    for (0..N) |i| {
        y_c[i] = y[i] - mean_y;
        for (0..D) |j| {
            X_c[i * D + j] = x[i * D + j] - mean_x[j];
        }
    }

    return CenteredData{
        .mean_x = mean_x,
        .mean_y = mean_y,
        .X_c = X_c,
        .y_c = y_c,
    };
}
