const std = @import("std");
const autodiff = @import("../autodiff.zig");
const c = @import("../cblas.zig");
const shape_mod = @import("shape.zig");
pub const Shape = shape_mod.Shape;
pub const computeContiguousStrides = shape_mod.computeContiguousStrides;
pub const transposeShape = shape_mod.transposeShape;
pub const broadcastShapes = shape_mod.broadcastShapes;
pub const computeBroadcastStrides = shape_mod.computeBroadcastStrides;
pub const broadcastBinaryOpRaw = shape_mod.broadcastBinaryOpRaw;

const types_mod = @import("types.zig");
pub const DType = types_mod.DType;
pub const bf16 = types_mod.bf16;
pub const SliceRange = types_mod.SliceRange;
pub const GenericTensor = types_mod.GenericTensor;

const core_mod = @import("core.zig");
pub const Tensor = core_mod.Tensor;




// ============================================================================
// NumPy-like raw tensor creation APIs (independent of Graph)
// ============================================================================

pub fn array(allocator: std.mem.Allocator, shape_slice: []const usize, initial_data: []const f32) !*Tensor {
    const shape = try Shape.fromSlice(shape_slice);
    const strides = computeContiguousStrides(shape);
    var total_size: usize = 1;
    for (shape_slice) |dim| {
        total_size *= dim;
    }
    if (total_size != initial_data.len) return error.ShapeMismatch;

    const t = try allocator.create(Tensor);
    t.* = Tensor{
        .data = try allocator.alloc(f32, total_size),
        .grad = &.{},
        .shape = shape,
        .strides = strides,
        .requires_grad = false,
        .creator = null,
    };
    @memcpy(t.data, initial_data);
    return t;
}

pub fn zeros(allocator: std.mem.Allocator, shape_slice: []const usize) !*Tensor {
    const shape = try Shape.fromSlice(shape_slice);
    const strides = computeContiguousStrides(shape);
    var total_size: usize = 1;
    for (shape_slice) |dim| {
        total_size *= dim;
    }

    const t = try allocator.create(Tensor);
    t.* = Tensor{
        .data = try allocator.alloc(f32, total_size),
        .grad = &.{},
        .shape = shape,
        .strides = strides,
        .requires_grad = false,
        .creator = null,
    };
    @memset(t.data, 0.0);
    return t;
}

pub fn ones(allocator: std.mem.Allocator, shape_slice: []const usize) !*Tensor {
    const shape = try Shape.fromSlice(shape_slice);
    const strides = computeContiguousStrides(shape);
    var total_size: usize = 1;
    for (shape_slice) |dim| {
        total_size *= dim;
    }

    const t = try allocator.create(Tensor);
    t.* = Tensor{
        .data = try allocator.alloc(f32, total_size),
        .grad = &.{},
        .shape = shape,
        .strides = strides,
        .requires_grad = false,
        .creator = null,
    };
    @memset(t.data, 1.0);
    return t;
}

/// 创建所有元素初始化为指定标量值的张量 (np.full)
pub fn full(allocator: std.mem.Allocator, shape_slice: []const usize, val: f32) !*Tensor {
    const t = try zeros(allocator, shape_slice);
    @memset(t.data, val);
    return t;
}

/// 生成等差数列张量 (np.arange)
/// 如果只传 1 个参数（step 默认为 1.0），生成 [0, stop)；
/// 如果传 start、stop、step，生成从 start 到 stop 步长为 step 的数列。
pub fn arange(allocator: std.mem.Allocator, start: f32, stop: ?f32, step: ?f32) !*Tensor {
    var actual_start: f32 = 0.0;
    var actual_stop: f32 = start;
    const actual_step: f32 = step orelse 1.0;

    if (actual_step == 0.0) return error.InvalidStep;

    if (stop) |st| {
        actual_start = start;
        actual_stop = st;
    }

    if ((actual_step > 0.0 and actual_start >= actual_stop) or (actual_step < 0.0 and actual_start <= actual_stop)) {
        return try zeros(allocator, &.{0});
    }

    const count_f = @ceil((actual_stop - actual_start) / actual_step);
    const count: usize = @intFromFloat(@max(0.0, count_f));

    const t = try zeros(allocator, &.{count});
    var cur = actual_start;
    for (0..count) |i| {
        t.data[i] = cur;
        cur += actual_step;
    }
    return t;
}

/// 生成指定区间内均匀间隔的浮点序列 (np.linspace)
/// num 为生成的点数 (默认需 >= 2，若为 1 则返回 start)
pub fn linspace(allocator: std.mem.Allocator, start: f32, stop: f32, num: usize) !*Tensor {
    if (num == 0) return try zeros(allocator, &.{0});
    const t = try zeros(allocator, &.{num});
    if (num == 1) {
        t.data[0] = start;
        return t;
    }

    const step = (stop - start) / @as(f32, @floatFromInt(num - 1));
    for (0..num) |i| {
        if (i == num - 1) {
            t.data[i] = stop; // 避免累加浮点精度误差
        } else {
            t.data[i] = start + @as(f32, @floatFromInt(i)) * step;
        }
    }
    return t;
}

/// 生成单位矩阵或指定对角线偏置矩阵 (np.eye)
/// N 为行数，M 为列数 (若传 null 则与 N 相同)，k 为对角线偏置 (0 为主对角线，正数为主对角线上方，负数为主对角线下方)
pub fn eye(allocator: std.mem.Allocator, N: usize, M: ?usize, k: ?i32) !*Tensor {
    const cols = M orelse N;
    const diag_offset = k orelse 0;

    const t = try zeros(allocator, &.{ N, cols });
    for (0..N) |r| {
        const c_idx: i64 = @as(i64, @intCast(r)) + @as(i64, diag_offset);
        if (c_idx >= 0 and c_idx < @as(i64, @intCast(cols))) {
            t.data[r * cols + @as(usize, @intCast(c_idx))] = 1.0;
        }
    }
    return t;
}

/// 生成方阵单位矩阵 (np.identity)
pub fn identity(allocator: std.mem.Allocator, n: usize) !*Tensor {
    return eye(allocator, n, n, 0);
}


var default_prng = std.Random.DefaultPrng.init(12345);

pub fn manualSeed(seed: u64) void {
    default_prng = std.Random.DefaultPrng.init(seed);
}

pub fn rand(allocator: std.mem.Allocator, shape_slice: []const usize) !*Tensor {
    const t = try zeros(allocator, shape_slice);
    const random = default_prng.random();
    for (t.data) |*val| {
        val.* = random.float(f32);
    }
    return t;
}

pub fn free(allocator: std.mem.Allocator, t: *Tensor) void {
    t.deinit(allocator);
}

/// 沿指定维度拼接多个张量 (Concat)
pub fn concat(allocator: std.mem.Allocator, inputs: []const *Tensor, dim: usize, graph: ?*autodiff.Graph) anyerror!*Tensor {
    if (graph) |g| {
        return try g.concat(inputs, dim);
    }
    if (inputs.len == 0) return error.EmptyInputs;
    const rank = inputs[0].shape.len;
    if (dim >= rank) return error.DimensionOutOfBounds;

    var out_shape = inputs[0].shape;
    var concat_dim_total: usize = 0;

    for (inputs) |t| {
        if (t.shape.len != rank) return error.IncompatibleDimensions;
        for (0..rank) |d| {
            if (d != dim) {
                if (t.shape.dims[d] != inputs[0].shape.dims[d]) return error.ShapeMismatch;
            }
        }
        concat_dim_total += t.shape.dims[dim];
    }
    out_shape.dims[dim] = concat_dim_total;

    const out = try zeros(allocator, out_shape.dims[0..rank]);

    var outer_size: usize = 1;
    for (0..dim) |d| {
        outer_size *= out_shape.dims[d];
    }
    var inner_size: usize = 1;
    for (dim + 1..rank) |d| {
        inner_size *= out_shape.dims[d];
    }

    for (0..outer_size) |outer| {
        const out_base = outer * concat_dim_total * inner_size;
        var offset_dim: usize = 0;
        for (inputs) |t| {
            const d_k = t.shape.dims[dim];
            const src_base = outer * d_k * inner_size;
            const dest_base = out_base + offset_dim * inner_size;
            const copy_len = d_k * inner_size;
            @memcpy(out.data[dest_base .. dest_base + copy_len], t.data[src_base .. src_base + copy_len]);
            offset_dim += d_k;
        }
    }

    return out;
}

/// 沿新维度堆叠多个张量 (np.stack)
/// axis 取值范围为 0..rank+1，所有输入张量必须具有完全相同的形状
pub fn stack(allocator: std.mem.Allocator, inputs: []const *Tensor, axis: usize) !*Tensor {
    if (inputs.len == 0) return error.EmptyInputs;
    const base_shape = inputs[0].shape;
    const in_rank = base_shape.len;
    if (axis > in_rank) return error.DimensionOutOfBounds;
    if (in_rank >= 8) return error.MaxDimensionsExceeded;

    for (inputs[1..]) |t| {
        if (!t.shape.eq(base_shape)) return error.ShapeMismatch;
    }

    // 首先对每一个输入张量在其 axis 处执行 unsqueeze
    const unsqueezed = try allocator.alloc(*Tensor, inputs.len);
    defer allocator.free(unsqueezed);

    for (inputs, 0..) |t, i| {
        unsqueezed[i] = try t.unsqueeze(axis, allocator);
    }
    defer {
        for (unsqueezed) |u| {
            free(allocator, u);
        }
    }

    // 沿 axis 拼接
    return try concat(allocator, unsqueezed, axis, null);
}

pub fn repeat(t: *Tensor, repeats: usize, axis: ?usize, allocator: std.mem.Allocator) !*Tensor {
    return t.repeat(repeats, axis, allocator);
}

pub fn tile(t: *Tensor, reps: []const usize, allocator: std.mem.Allocator) !*Tensor {
    return t.tile(reps, allocator);
}

pub fn sqrt(t: *Tensor, allocator: std.mem.Allocator) !*Tensor {
    return t.sqrt(allocator);
}

pub fn exp(t: *Tensor, allocator: std.mem.Allocator) !*Tensor {
    return t.exp(allocator);
}

pub fn log(t: *Tensor, allocator: std.mem.Allocator) !*Tensor {
    return t.log(allocator);
}

pub fn abs(t: *Tensor, allocator: std.mem.Allocator) !*Tensor {
    return t.abs(allocator);
}

/// 沿指定维度将张量均等切分为 num_splits 个子张量 (Split)
pub fn split(allocator: std.mem.Allocator, input: *Tensor, num_splits: usize, dim: usize, graph: ?*autodiff.Graph) anyerror![]*Tensor {
    if (graph) |g| {
        return try g.split(input, num_splits, dim);
    }
    if (num_splits == 0) return error.InvalidSplitCount;
    const rank = input.shape.len;
    if (dim >= rank) return error.DimensionOutOfBounds;
    const dim_size = input.shape.dims[dim];
    if (dim_size % num_splits != 0) return error.UnevenSplit;

    const split_dim_size = dim_size / num_splits;

    var split_shape = input.shape;
    split_shape.dims[dim] = split_dim_size;

    const outputs = try allocator.alloc(*Tensor, num_splits);
    for (0..num_splits) |k| {
        outputs[k] = try zeros(allocator, split_shape.dims[0..rank]);
    }

    var outer_size: usize = 1;
    for (0..dim) |d| {
        outer_size *= input.shape.dims[d];
    }
    var inner_size: usize = 1;
    for (dim + 1..rank) |d| {
        inner_size *= input.shape.dims[d];
    }

    for (0..outer_size) |outer| {
        const src_base = outer * dim_size * inner_size;
        for (0..num_splits) |k| {
            const dest_base = outer * split_dim_size * inner_size;
            const src_offset = src_base + k * split_dim_size * inner_size;
            const copy_len = split_dim_size * inner_size;
            @memcpy(outputs[k].data[dest_base .. dest_base + copy_len], input.data[src_offset .. src_offset + copy_len]);
        }
    }

    return outputs;
}

const tensorSplit = split;

pub const where = Tensor.where;

pub fn sum(t: *Tensor, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor {
    return t.sum(axis, keepdims, allocator);
}

pub fn mean(t: *Tensor, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor {
    return t.mean(axis, keepdims, allocator);
}

pub fn variance(t: *Tensor, axis: ?usize, keepdims: bool, ddof: usize, allocator: std.mem.Allocator) !*Tensor {
    return t.variance(axis, keepdims, ddof, allocator);
}

pub fn stdDev(t: *Tensor, axis: ?usize, keepdims: bool, ddof: usize, allocator: std.mem.Allocator) !*Tensor {
    return t.stdDev(axis, keepdims, ddof, allocator);
}

pub fn squeeze(t: *Tensor, axis: ?usize, allocator: std.mem.Allocator) !*Tensor {
    return t.squeeze(axis, allocator);
}

pub fn unsqueeze(t: *Tensor, dim: usize, allocator: std.mem.Allocator) !*Tensor {
    return t.unsqueeze(dim, allocator);
}

pub fn slice(t: *Tensor, ranges: []const SliceRange, allocator: std.mem.Allocator) !*Tensor {
    return t.slice(ranges, allocator);
}

pub fn clip(t: *Tensor, min_val: f32, max_val: f32, allocator: std.mem.Allocator) !*Tensor {
    return t.clip(min_val, max_val, allocator);
}

pub fn sort(t: *Tensor, axis: ?usize, ascending: bool, allocator: std.mem.Allocator) !*Tensor {
    return t.sort(axis, ascending, allocator);
}

pub fn argsort(t: *Tensor, axis: ?usize, ascending: bool, allocator: std.mem.Allocator) !*GenericTensor(usize) {
    return t.argsort(axis, ascending, allocator);
}

pub fn nonzero(t: Tensor, allocator: std.mem.Allocator) !*GenericTensor(usize) {
    return t.nonzero(allocator);
}



/// Solves linear system A * x = b using Gauss-Jordan elimination with partial pivoting.
/// A is an n x n row-major matrix slice, b is an n-element vector, out_x is an n-element output slice.
pub fn solveLinearSystem(allocator: std.mem.Allocator, A_data: []const f32, b_data: []const f32, n: usize, out_x: []f32) !void {
    if (A_data.len != n * n or b_data.len != n or out_x.len != n) return error.ShapeMismatch;

    if (n == 0) return;
    if (n == 1) {
        if (@abs(A_data[0]) < 1e-12) return error.SingularMatrix;
        out_x[0] = b_data[0] / A_data[0];
        return;
    }

    // Augmented matrix [A | b] of dimensions n x (n + 1)
    const cols = n + 1;
    const aug = try allocator.alloc(f32, n * cols);
    defer allocator.free(aug);

    for (0..n) |i| {
        for (0..n) |j| {
            aug[i * cols + j] = A_data[i * n + j];
        }
        aug[i * cols + n] = b_data[i];
    }

    // Gauss-Jordan elimination with partial pivoting
    for (0..n) |col| {
        // Find pivot
        var max_val: f32 = @abs(aug[col * cols + col]);
        var pivot_row: usize = col;
        for ((col + 1)..n) |r| {
            const val = @abs(aug[r * cols + col]);
            if (val > max_val) {
                max_val = val;
                pivot_row = r;
            }
        }

        if (max_val < 1e-12) {
            return error.SingularMatrix;
        }

        // Swap current row with pivot row
        if (pivot_row != col) {
            for (0..cols) |j| {
                const tmp = aug[col * cols + j];
                aug[col * cols + j] = aug[pivot_row * cols + j];
                aug[pivot_row * cols + j] = tmp;
            }
        }

        // Normalize pivot row
        const pivot = aug[col * cols + col];
        for (col..cols) |j| {
            aug[col * cols + j] /= pivot;
        }

        // Eliminate column entries in other rows
        for (0..n) |r| {
            if (r == col) continue;
            const factor = aug[r * cols + col];
            if (factor == 0.0) continue;
            for (col..cols) |j| {
                aug[r * cols + j] -= factor * aug[col * cols + j];
            }
        }
    }

    // Extract solution
    for (0..n) |i| {
        out_x[i] = aug[i * cols + n];
    }
}

/// Solves Ridge Regression analytically:
/// min ||X*w + b*1 - y||_2^2 + lambda * ||w||_2^2
/// Using centered formulation: w = (X_c^T * X_c + lambda * I)^(-1) * X_c^T * y_c
/// b = mean(y) - sum(mean(x_j) * w_j)
pub fn solveRidgeAnalytical(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    n_samples: usize,
    n_features: usize,
    lambda: f32,
    out_w: []f32,
    out_b: *f32,
) !void {
    if (x.len != n_samples * n_features or y.len != n_samples or out_w.len != n_features) {
        return error.ShapeMismatch;
    }


    const N = n_samples;
    const D = n_features;
    const N_f = @as(f32, @floatFromInt(N));

    // 1. Compute means
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

    // 2. Build normal matrix M = X_c^T * X_c + lambda * I, and vector v = X_c^T * y_c
    const M = try allocator.alloc(f32, D * D);
    defer allocator.free(M);
    @memset(M, 0.0);

    const v = try allocator.alloc(f32, D);
    defer allocator.free(v);
    @memset(v, 0.0);

    for (0..N) |i| {
        const dy = y[i] - mean_y;
        for (0..D) |j| {
            const dx_j = x[i * D + j] - mean_x[j];
            v[j] += dx_j * dy;
            for (0..D) |k| {
                const dx_k = x[i * D + k] - mean_x[k];
                M[j * D + k] += dx_j * dx_k;
            }
        }
    }

    // Add L2 penalty lambda to the diagonal
    for (0..D) |j| {
        M[j * D + j] += lambda;
    }

    // 3. Solve M * w = v
    try solveLinearSystem(allocator, M, v, D, out_w);

    // 4. Compute intercept b = mean_y - w^T * mean_x
    var dot_w_mean_x: f32 = 0.0;
    for (0..D) |j| {
        dot_w_mean_x += out_w[j] * mean_x[j];
    }
    out_b.* = mean_y - dot_w_mean_x;
}

test "Shape and strides helpers" {
    // Test Shape init & eq
    const s1 = Shape.init(&.{2, 3, 4});
    try std.testing.expectEqual(@as(usize, 3), s1.len);
    try std.testing.expectEqual(@as(usize, 2), s1.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), s1.dims[1]);
    try std.testing.expectEqual(@as(usize, 4), s1.dims[2]);

    const s2 = Shape.init(&.{2, 3, 4});
    try std.testing.expect(s1.eq(s2));

    const s3 = Shape.init(&.{2, 3, 5});
    try std.testing.expect(!s1.eq(s3));

    // Test computeContiguousStrides
    const strides1 = computeContiguousStrides(s1);
    try std.testing.expectEqual(@as(usize, 12), strides1.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), strides1.dims[1]);
    try std.testing.expectEqual(@as(usize, 1), strides1.dims[2]);

    // Test transposeShape
    const s_trans = transposeShape(s1, 0, 1);
    try std.testing.expectEqual(@as(usize, 3), s_trans.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), s_trans.dims[1]);
    try std.testing.expectEqual(@as(usize, 4), s_trans.dims[2]);
}

test "Tensor indexing and gradient operations" {
    const allocator = std.testing.allocator;
    const shape = Shape.init(&.{2, 3});
    const strides = computeContiguousStrides(shape);

    const data = try allocator.alloc(f32, 6);
    defer allocator.free(data);
    const grad = try allocator.alloc(f32, 6);
    defer allocator.free(grad);

    var t = Tensor{
        .data = data,
        .grad = grad,
        .shape = shape,
        .strides = strides,
        .requires_grad = true,
        .creator = null,
    };

    // Test indexing
    t.set(&.{0, 0}, 1.0);
    t.set(&.{0, 1}, 2.0);
    t.set(&.{0, 2}, 3.0);
    t.set(&.{1, 0}, 4.0);
    t.set(&.{1, 1}, 5.0);
    t.set(&.{1, 2}, 6.0);

    try std.testing.expectEqual(@as(f32, 1.0), t.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 6.0), t.get(&.{1, 2}));
    try std.testing.expectEqual(@as(usize, 5), t.getFlatIndex(&.{1, 2}));

    // Test grad operations
    t.setGrad(&.{0, 1}, 10.0);
    try std.testing.expectEqual(@as(f32, 10.0), t.getGrad(&.{0, 1}));

    t.zeroGrad();
    try std.testing.expectEqual(@as(f32, 0.0), t.getGrad(&.{0, 1}));
}

test "NumPy-like raw tensor creation" {
    const allocator = std.testing.allocator;

    // Test array creation
    const t_arr = try array(allocator, &.{2, 3}, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer free(allocator, t_arr);
    try std.testing.expectEqual(@as(f32, 1.0), t_arr.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 6.0), t_arr.get(&.{1, 2}));

    // Test zeros creation
    const t_zeros = try zeros(allocator, &.{2, 2});
    defer free(allocator, t_zeros);
    try std.testing.expectEqual(@as(f32, 0.0), t_zeros.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 0.0), t_zeros.get(&.{1, 1}));

    // Test ones creation
    const t_ones = try ones(allocator, &.{3, 1});
    defer free(allocator, t_ones);
    try std.testing.expectEqual(@as(f32, 1.0), t_ones.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), t_ones.get(&.{2, 0}));
}

test "Direct tensor operations (eager and graph)" {
    const allocator = std.testing.allocator;

    // Eager Mode Test
    {
        const A = try array(allocator, &.{2, 3}, &[_]f32{ 1, 2, 3, 4, 5, 6 });
        defer free(allocator, A);
        const B = try array(allocator, &.{3, 2}, &[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
        defer free(allocator, B);

        // Matmul
        const C = try A.matmul(B, allocator, null);
        defer free(allocator, C);
        try std.testing.expectApproxEqAbs(@as(f32, 2.2), C.get(&.{0, 0}), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 6.4), C.get(&.{1, 1}), 1e-5);

        // AddBias
        const bias = try array(allocator, &.{1, 2}, &[_]f32{ 0.5, 1.0 });
        defer free(allocator, bias);
        const D = try C.addBias(bias, allocator, null);
        defer free(allocator, D);
        try std.testing.expectApproxEqAbs(@as(f32, 2.7), D.get(&.{0, 0}), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 7.4), D.get(&.{1, 1}), 1e-5);

        // Relu
        const E = try D.relu(allocator, null);
        defer free(allocator, E);
        try std.testing.expectApproxEqAbs(@as(f32, 2.7), E.get(&.{0, 0}), 1e-5);

        // SoftmaxCrossEntropy
        const loss = try E.softmaxCrossEntropy(&[2]u8{ 0, 1 }, allocator, null);
        defer free(allocator, loss);
        try std.testing.expect(loss.get(&.{0, 0}) > 0.0);

        // Reshape
        const F = try E.reshape(&.{1, 4}, allocator, null);
        defer free(allocator, F);
        try std.testing.expectEqualSlices(usize, &.{1, 4}, F.shape.dims[0..F.shape.len]);

        // Transpose
        const G = try F.transpose(0, 1, allocator, null);
        defer free(allocator, G);
        try std.testing.expectEqualSlices(usize, &.{4, 1}, G.shape.dims[0..G.shape.len]);
    }

    // Graph Mode Test
    {
        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();

        const A = try graph.array(&.{2, 3}, &[_]f32{ 1, 2, 3, 4, 5, 6 }, true);
        const B = try graph.array(&.{3, 2}, &[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, true);

        // Matmul
        const C = try A.matmul(B, allocator, &graph);
        try std.testing.expectApproxEqAbs(@as(f32, 2.2), C.get(&.{0, 0}), 1e-5);

        // AddBias
        const bias = try graph.array(&.{1, 2}, &[_]f32{ 0.5, 1.0 }, true);
        const D = try C.addBias(bias, allocator, &graph);
        try std.testing.expectApproxEqAbs(@as(f32, 2.7), D.get(&.{0, 0}), 1e-5);

        // Relu
        const E = try D.relu(allocator, &graph);

        // SoftmaxCrossEntropy
        const loss = try E.softmaxCrossEntropy(&[2]u8{ 0, 1 }, allocator, &graph);
        try std.testing.expect(loss.get(&.{0, 0}) > 0.0);

        // Reshape
        const F = try E.reshape(&.{1, 4}, allocator, &graph);

        // Transpose
        const G = try F.transpose(0, 1, allocator, &graph);
        try std.testing.expectEqualSlices(usize, &.{4, 1}, G.shape.dims[0..G.shape.len]);
    }
}

test "Tensor argmax and max reductions" {
    const allocator = std.testing.allocator;

    const A = try array(allocator, &.{2, 3}, &[_]f32{ 1.0, 5.0, 3.0, 9.0, 2.0, 6.0 });
    defer free(allocator, A);

    // Test argmax along dim 1
    const idx1 = try A.argmax(1, allocator);
    defer free(allocator, idx1);
    try std.testing.expectEqual(@as(f32, 1.0), idx1.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 0.0), idx1.get(&.{1, 0}));

    // Test max along dim 1
    const val1 = try A.max(1, allocator);
    defer free(allocator, val1);
    try std.testing.expectEqual(@as(f32, 5.0), val1.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 9.0), val1.get(&.{1, 0}));

    // Test argmax along dim 0
    const idx0 = try A.argmax(0, allocator);
    defer free(allocator, idx0);
    try std.testing.expectEqual(@as(f32, 1.0), idx0.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 0.0), idx0.get(&.{0, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), idx0.get(&.{0, 2}));

    // Test max along dim 0
    const val0 = try A.max(0, allocator);
    defer free(allocator, val0);
    try std.testing.expectEqual(@as(f32, 9.0), val0.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 5.0), val0.get(&.{0, 1}));
    try std.testing.expectEqual(@as(f32, 6.0), val0.get(&.{0, 2}));
}

test "Tensor MSE loss forward and backward" {
    const allocator = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const y_pred = try graph.array(&.{2, 1}, &[_]f32{ 1.5, 2.5 }, true);
    const y_true = try graph.array(&.{2, 1}, &[_]f32{ 1.0, 3.0 }, false);

    const loss = try graph.mseLoss(y_pred, y_true);
    // loss = 0.5 * ((1.5 - 1.0)^2 + (2.5 - 3.0)^2) = 0.5 * (0.25 + 0.25) = 0.25
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), loss.data[0], 1e-5);

    try graph.backward(loss);

    // grad of y_pred = 2/N * (y_pred - y_true) = 2/2 * (y_pred - y_true) = y_pred - y_true
    // dy_pred_0 = 1.5 - 1.0 = 0.5
    // dy_pred_1 = 2.5 - 3.0 = -0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), y_pred.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), y_pred.grad[1], 1e-5);
}

test "Tensor mulScalar and add autograd" {
    const allocator = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const A = try graph.array(&.{2, 2}, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);
    const B = try graph.array(&.{2, 2}, &[_]f32{ 5.0, 6.0, 7.0, 8.0 }, true);

    // C = A.mulScalar(2.0)
    const C = try A.mulScalar(2.0, arena_allocator, &graph);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), C.get(&.{0, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), C.get(&.{1, 1}), 1e-5);

    // D = C + B
    const D = try C.add(B, arena_allocator, &graph);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), D.get(&.{0, 0}), 1e-5); // 2.0 + 5.0 = 7.0
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), D.get(&.{1, 1}), 1e-5); // 8.0 + 8.0 = 16.0

    // E = D.addScalar(10.0)
    const E = try D.addScalar(10.0, arena_allocator, &graph);
    try std.testing.expectApproxEqAbs(@as(f32, 17.0), E.get(&.{0, 0}), 1e-5); // 7.0 + 10.0 = 17.0
    try std.testing.expectApproxEqAbs(@as(f32, 26.0), E.get(&.{1, 1}), 1e-5); // 16.0 + 10.0 = 26.0

    // Set gradients of E to 1.0 to backpropagate
    for (E.grad) |*g| {
        g.* = 1.0;
    }

    try graph.backward(E);

    // Since E = D + 10, dE/dD = 1
    // Since D = C + B, dD/dB = 1 => B.grad = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), B.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), B.grad[3], 1e-5);

    // Since E = D + 10, dE/dD = 1
    // Since D = C + B, dD/dC = 1
    // Since C = A * 2, dC/dA = 2
    // By chain rule, dE/dA = 1 * 1 * 2 = 2.0 => A.grad = 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[3], 1e-5);
}

test "Tensor static graph forward and backward" {
    const allocator = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    // 1. Build the static graph template once
    const A = try graph.array(&.{2, 2}, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);
    const B = try graph.array(&.{2, 2}, &[_]f32{ 5.0, 6.0, 7.0, 8.0 }, true);
    const C = try A.mulScalar(2.0, arena_allocator, &graph);
    const D = try C.add(B, arena_allocator, &graph);

    // 2. First Run: set inputs
    A.data[0] = 1.0; A.data[1] = 2.0; A.data[2] = 3.0; A.data[3] = 4.0;
    B.data[0] = 5.0; B.data[1] = 6.0; B.data[2] = 7.0; B.data[3] = 8.0;

    // Execute forward pass
    try graph.forward();
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), D.get(&.{0, 0}), 1e-5); // 2*1 + 5 = 7
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), D.get(&.{1, 1}), 1e-5); // 2*4 + 8 = 16

    // Execute backward pass
    graph.zeroGrad(); // Clear all gradients in the graph!
    @memset(D.grad, 1.0);
    try graph.backward(D);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), B.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[0], 1e-5);

    // 3. Second Run: change input data
    A.data[0] = 10.0; A.data[1] = 20.0; A.data[2] = 30.0; A.data[3] = 40.0;
    B.data[0] = 100.0; B.data[1] = 200.0; B.data[2] = 300.0; B.data[3] = 400.0;

    // Recompute forward pass on the exact same graph structure!
    try graph.forward();
    try std.testing.expectApproxEqAbs(@as(f32, 120.0), D.get(&.{0, 0}), 1e-5); // 2*10 + 100 = 120
    try std.testing.expectApproxEqAbs(@as(f32, 480.0), D.get(&.{1, 1}), 1e-5); // 2*40 + 400 = 480

    // Recompute backward pass
    graph.zeroGrad(); // Clear gradients again!
    @memset(D.grad, 1.0);
    try graph.backward(D);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), B.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[0], 1e-5);
}

test "Softmax forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    // Input shape [2, 3]
    const X = try graph.array(&.{2, 3}, &[_]f32{
        1.0, 2.0, 3.0,
        1.0, 1.0, 1.0,
    }, true);

    const Y = try X.softmax(arena_allocator, &graph);

    try graph.forward();

    // Check forward
    // Row 0: exp(1), exp(2), exp(3) -> sum = 2.718 + 7.389 + 20.085 = 30.192
    // exp(1)/sum = 0.0900, exp(2)/sum = 0.2447, exp(3)/sum = 0.6652
    try std.testing.expectApproxEqAbs(@as(f32, 0.09003057), Y.get(&.{0, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.24472847), Y.get(&.{0, 1}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.66524096), Y.get(&.{0, 2}), 1e-5);
    // Row 1: exp(1), exp(1), exp(1) -> 1/3, 1/3, 1/3
    try std.testing.expectApproxEqAbs(@as(f32, 0.33333333), Y.get(&.{1, 0}), 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(Y.grad, 1.0); // dL/dY = 1.0
    // dX_i = Y_i * (dY_i - sum_j dY_j Y_j)
    // Since dY_j = 1.0, sum_j dY_j Y_j = sum_j Y_j = 1.0 (since softmax sums to 1)
    // So dX_i = Y_i * (1.0 - 1.0) = 0.0
    try graph.backward(Y);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), X.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), X.grad[5], 1e-5);

    // Try another grad
    graph.zeroGrad();
    Y.grad[0] = 1.0;
    Y.grad[1] = 0.0;
    Y.grad[2] = 0.0;
    // Row 0: sum_dy_y = 1.0 * Y_0 = Y_0
    // dX_0 = Y_0 * (1.0 - Y_0) = Y_0 * (1 - Y_0)
    // dX_1 = Y_1 * (0.0 - Y_0) = - Y_1 * Y_0
    // dX_2 = Y_2 * (0.0 - Y_0) = - Y_2 * Y_0
    try graph.backward(Y);
    const y0 = Y.get(&.{0, 0});
    const y1 = Y.get(&.{0, 1});
    try std.testing.expectApproxEqAbs(y0 * (1.0 - y0), X.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(-y1 * y0, X.grad[1], 1e-5);
}

test "RMSNorm forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const X = try graph.array(&.{2, 3}, &[_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    }, true);
    const G = try graph.array(&.{3}, &[_]f32{ 1.0, 2.0, 3.0 }, true);

    const Y = try X.rmsNorm(G, 1e-5, arena_allocator, &graph);

    try graph.forward();

    // Row 0: mean(x^2) = (1+4+9)/3 = 14/3 = 4.666666
    // rms = sqrt(4.666666) = 2.1602468
    // Y_0 = 1 / rms * 1 = 0.46291
    // Y_1 = 2 / rms * 2 = 1.85164
    // Y_2 = 3 / rms * 3 = 4.16619
    try std.testing.expectApproxEqAbs(@as(f32, 0.46291), Y.get(&.{0, 0}), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.85164), Y.get(&.{0, 1}), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.16619), Y.get(&.{0, 2}), 1e-4);

    // Backward
    graph.zeroGrad();
    @memset(Y.grad, 1.0);
    try graph.backward(Y);

    // We can verify gradients numerically or just check they are non-zero and reasonable.
    // Let's verify G.grad: dG_j = sum_i (dY_i * X_i * scale)
    // Row 0 scale = 1/2.1602468 = 0.46291
    // Row 1: mean(x^2) = (16+25+36)/3 = 77/3 = 25.6666
    // Row 1 scale = 1/sqrt(25.6666) = 1/5.066228 = 0.197385
    // dG_0 = 1.0 * 1.0 * 0.46291 + 1.0 * 4.0 * 0.197385 = 0.46291 + 0.78954 = 1.25245
    try std.testing.expectApproxEqAbs(@as(f32, 1.25245), G.grad[0], 1e-4);
}

test "Embedding forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const W = try graph.array(&.{3, 4}, &[_]f32{
        0.1, 0.2, 0.3, 0.4,
        1.1, 1.2, 1.3, 1.4,
        2.1, 2.2, 2.3, 2.4,
    }, true);

    const X = try graph.array(&.{2, 2}, &[_]f32{
        0.0, 2.0,
        1.0, 0.0,
    }, false);

    const Y = try W.embedding(X, arena_allocator, &graph);

    try graph.forward();

    // Check forward
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), Y.get(&.{0, 0, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.3), Y.get(&.{0, 1, 2}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4), Y.get(&.{1, 0, 3}), 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(Y.grad, 1.0);
    try graph.backward(Y);

    // W.grad should accumulate gradients
    // X has:
    // (0,0) -> 0.0
    // (0,1) -> 2.0
    // (1,0) -> 1.0
    // (1,1) -> 0.0
    // So row 0 of W is selected twice, row 1 once, row 2 once.
    // Since dY is all 1.0, W.grad row 0 should be 2.0, row 1 should be 1.0, row 2 should be 1.0.
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), W.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), W.grad[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), W.grad[8], 1e-5);
}

test "BatchMatMul forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    // Shape [2, 2, 2, 3]
    const A = try graph.array(&.{2, 2, 2, 3}, &[_]f32{
        // batch 0, head 0
        1, 2, 3,
        4, 5, 6,
        // batch 0, head 1
        1, 1, 1,
        2, 2, 2,
        // batch 1, head 0
        0, 1, 0,
        1, 0, 1,
        // batch 1, head 1
        2, 0, 2,
        0, 2, 0,
    }, true);

    // Shape [2, 2, 3, 2]
    const B = try graph.array(&.{2, 2, 3, 2}, &[_]f32{
        // batch 0, head 0
        1, 0,
        0, 1,
        1, 1,
        // batch 0, head 1
        2, 2,
        2, 2,
        2, 2,
        // batch 1, head 0
        1, 2,
        3, 4,
        5, 6,
        // batch 1, head 1
        1, 1,
        1, 1,
        1, 1,
    }, true);

    const C = try A.batchMatMul(B, arena_allocator, &graph);

    try graph.forward();

    // Check forward
    // Batch 0, Head 0:
    // [1, 2, 3]   [1, 0]   [4, 5]
    // [4, 5, 6] * [0, 1] = [10, 11]
    //             [1, 1]
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), C.get(&.{0, 0, 0, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), C.get(&.{0, 0, 0, 1}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), C.get(&.{0, 0, 1, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), C.get(&.{0, 0, 1, 1}), 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(C.grad, 1.0);
    try graph.backward(C);

    // We can verify some gradients.
    // dA = dC * B^T
    // For Batch 0, Head 0:
    // dC_slice = [1, 1]
    //            [1, 1]
    // B_slice^T = [1, 0, 1]
    //             [0, 1, 1]
    // dA_slice = dC_slice * B_slice^T = [1, 1, 2]
    //                                   [1, 1, 2]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[2], 1e-5);
}

test "GELU forward and backward" {
    const arena_allocator = std.testing.allocator;
    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const A = try graph.tensorNDWithData(&.{2, 2}, &.{ -1.0, 0.0, 1.0, 2.0 }, true);
    const C = try A.gelu(arena_allocator, &graph);

    try graph.forward();

    // Check forward
    try std.testing.expectApproxEqAbs(@as(f32, -0.158655), C.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), C.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.841345), C.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.954500), C.data[3], 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(C.grad, 1.0);
    try graph.backward(C);

    // Check gradients
    try std.testing.expectApproxEqAbs(@as(f32, -0.083316), A.grad[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), A.grad[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.083316), A.grad[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.085232), A.grad[3], 1e-4);
}

test "Sigmoid forward and backward" {
    const arena_allocator = std.testing.allocator;
    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const A = try graph.tensorNDWithData(&.{2, 2}, &.{ -1.0, 0.0, 1.0, 2.0 }, true);
    const C = try A.sigmoid(arena_allocator, &graph);

    try graph.forward();

    // Check forward: sigmoid(x) = 1 / (1 + exp(-x))
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / (1.0 + @exp(@as(f32, 1.0)))), C.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), C.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / (1.0 + @exp(@as(f32, -1.0)))), C.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / (1.0 + @exp(@as(f32, -2.0)))), C.data[3], 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(C.grad, 1.0);
    try graph.backward(C);

    // Check gradients: grad = C * (1 - C)
    for (A.grad, C.data) |g_val, c_val| {
        try std.testing.expectApproxEqAbs(c_val * (1.0 - c_val), g_val, 1e-5);
    }
}

test "SigmoidCrossEntropy forward and backward" {
    const arena_allocator = std.testing.allocator;
    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const logits = try graph.tensorNDWithData(&.{3}, &.{ -1.0, 0.0, 2.0 }, true);
    const targets = try graph.tensorNDWithData(&.{3}, &.{ 0.0, 1.0, 1.0 }, false);
    const loss = try logits.sigmoidCrossEntropy(targets, arena_allocator, &graph);

    try graph.forward();

    // Check forward
    // x = -1, y = 0 -> loss = max(-1, 0) - 0 + log(1 + exp(-1)) = log(1 + e^-1) = log(1.367879) = 0.31326168
    // x = 0, y = 1 -> loss = max(0, 0) - 0 + log(1 + exp(0)) = log(2) = 0.69314718
    // x = 2, y = 1 -> loss = max(2, 0) - 2 + log(1 + exp(-2)) = log(1 + e^-2) = log(1.135335) = 0.126928
    // mean loss = (0.31326168 + 0.69314718 + 0.126928) / 3 = 1.13333686 / 3 = 0.37777895
    try std.testing.expectApproxEqAbs(@as(f32, 0.37777895), loss.data[0], 1e-5);

    // Backward
    graph.zeroGrad();
    loss.grad[0] = 1.0;
    try graph.backward(loss);

    // Check gradients:
    // grad = 1/3 * (sig(x) - y)
    // x = -1, y = 0 -> grad = 1/3 * (1/(1+e) - 0) = 1/3 * 0.268941 = 0.089647
    // x = 0, y = 1 -> grad = 1/3 * (0.5 - 1) = -1/6 = -0.166667
    // x = 2, y = 1 -> grad = 1/3 * (1/(1+e^-2) - 1) = 1/3 * (0.880797 - 1) = -0.039734
    try std.testing.expectApproxEqAbs(@as(f32, 0.089647), logits.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.166667), logits.grad[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.039734), logits.grad[2], 1e-5);
}

/// 旋转位置编码应用算子 (RoPE)
/// x: 输入张量切片，形状 [seq_len, n_head, head_dim]
/// head_dim 必须为偶数
pub fn applyRoPE(
    x: []f32,
    seq_len: usize,
    n_head: usize,
    head_dim: usize,
    base_freq: f32,
) void {
    std.debug.assert(head_dim % 2 == 0);
    const half_dim = head_dim / 2;

    for (0..seq_len) |m| {
        const m_f32 = @as(f32, @floatFromInt(m));

        for (0..half_dim) |i| {
            const i_f32 = @as(f32, @floatFromInt(i));
            const theta = 1.0 / std.math.pow(f32, base_freq, (2.0 * i_f32) / @as(f32, @floatFromInt(head_dim)));
            const freq = m_f32 * theta;
            const cos_val = @cos(freq);
            const sin_val = @sin(freq);

            for (0..n_head) |h| {
                const offset = (m * n_head + h) * head_dim + i * 2;
                const x1 = x[offset];
                const x2 = x[offset + 1];

                // 2D 旋转矩阵变换
                x[offset] = x1 * cos_val - x2 * sin_val;
                x[offset + 1] = x1 * sin_val + x2 * cos_val;
            }
        }
    }
}

/// 对 3D [T, n_head, head_dim] 或 4D [B, n_head, T, head_dim] 张量执行 RoPE 旋转
pub fn applyRoPETensor(t: *Tensor, base_freq: f32) void {
    if (t.shape.len == 3) {
        const seq_len = t.shape.dims[0];
        const n_head = t.shape.dims[1];
        const head_dim = t.shape.dims[2];
        applyRoPE(t.data, seq_len, n_head, head_dim, base_freq);
    } else if (t.shape.len == 4) {
        // [B, n_head, T, head_dim] -> 遍历每个 batch
        const B = t.shape.dims[0];
        const n_head = t.shape.dims[1];
        const T = t.shape.dims[2];
        const head_dim = t.shape.dims[3];
        const half_dim = head_dim / 2;

        for (0..B) |b| {
            for (0..T) |m| {
                const m_f32 = @as(f32, @floatFromInt(m));
                for (0..half_dim) |i| {
                    const i_f32 = @as(f32, @floatFromInt(i));
                    const theta = 1.0 / std.math.pow(f32, base_freq, (2.0 * i_f32) / @as(f32, @floatFromInt(head_dim)));
                    const freq = m_f32 * theta;
                    const cos_val = @cos(freq);
                    const sin_val = @sin(freq);

                    for (0..n_head) |h| {
                        const offset = ((b * n_head + h) * T + m) * head_dim + i * 2;
                        const x1 = t.data[offset];
                        const x2 = t.data[offset + 1];
                        t.data[offset] = x1 * cos_val - x2 * sin_val;
                        t.data[offset + 1] = x1 * sin_val + x2 * cos_val;
                    }
                }
            }
        }
    }
}
