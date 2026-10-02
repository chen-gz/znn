const std = @import("std");
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
pub fn concat(allocator: std.mem.Allocator, inputs: []const *Tensor, dim: usize) !*Tensor {
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
    return try concat(allocator, unsqueezed, axis);
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
pub fn split(allocator: std.mem.Allocator, input: *Tensor, num_splits: usize, dim: usize) ![]*Tensor {
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


pub fn where(cond: anytype, x: *Tensor, y: *Tensor, allocator: std.mem.Allocator) !*Tensor {
    return Tensor.where(cond, x, y, allocator);
}

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

pub fn contiguous(t: *Tensor, allocator: std.mem.Allocator) !*Tensor {
    return t.contiguous(allocator);
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

/// 旋转位置编码 (Rotary Position Embedding, RoPE) 应用算子
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

/// 对三维 [T, n_head, head_dim] 或四维 [B, n_head, T, head_dim] 张量执行旋转位置编码 (Rotary Position Embedding, RoPE) 旋转
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
