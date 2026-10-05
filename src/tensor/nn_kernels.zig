const std = @import("std");
const c = @import("../cblas.zig");
const core = @import("core.zig");
const types = @import("types.zig");
const conv_pool = @import("conv_pool.zig");
const Tensor = core.Tensor;
const ConvOptions = types.ConvOptions;
const PoolOptions = types.PoolOptions;
const RopeOptions = types.RopeOptions;
const ops_mod = @import("ops.zig");
const zeros = ops_mod.zeros;
const free = ops_mod.free;

pub const conv1d = conv_pool.conv1d;
pub const conv2d = conv_pool.conv2d;
pub const convTranspose1d = conv_pool.convTranspose1d;
pub const convTranspose2d = conv_pool.convTranspose2d;
pub const maxpool1d = conv_pool.maxpool1d;
pub const maxpool2d = conv_pool.maxpool2d;
pub const avgpool1d = conv_pool.avgpool1d;
pub const avgpool2d = conv_pool.avgpool2d;
pub const adaptiveAvgPool1d = conv_pool.adaptiveAvgPool1d;
pub const adaptiveAvgPool2d = conv_pool.adaptiveAvgPool2d;

pub fn softmax(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
    const D = self.shape.dims[self.shape.len - 1];
    const M = self.data.len / D;
    const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);

    for (0..M) |i| {
        const row_in = self.data[i * D .. (i + 1) * D];
        const row_out = C.data[i * D .. (i + 1) * D];

        var max_val = row_in[0];
        for (row_in[1..]) |val| {
            if (val > max_val) max_val = val;
        }

        var exp_sum: f32 = 0.0;
        for (row_in, row_out) |val, *p| {
            const exp_val = @exp(val - max_val);
            p.* = exp_val;
            exp_sum += exp_val;
        }

        for (row_out) |*p| {
            p.* /= exp_sum;
        }
    }
    return C;
}

/// 均方根层归一化前向核函数 (Root Mean Square Layer Normalization, RMSNorm)
pub fn rmsNorm(self: *Tensor, G: *Tensor, eps: f32, allocator: std.mem.Allocator) !*Tensor {
    const D = self.shape.dims[self.shape.len - 1];
    const M = self.data.len / D;
    const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);

    for (0..M) |i| {
        const row_in = self.data[i * D .. (i + 1) * D];
        const row_out = Y.data[i * D .. (i + 1) * D];

        var sum_x2: f32 = 0.0;
        for (row_in) |val| {
            sum_x2 += val * val;
        }
        const rms = @sqrt(sum_x2 / @as(f32, @floatFromInt(D)) + eps);

        for (row_in, row_out, G.data) |x_val, *y_val, g_val| {
            y_val.* = x_val / rms * g_val;
        }
    }
    return Y;
}

/// 标准层归一化前向核函数 (Layer Normalization, LayerNorm)
pub fn layerNorm(self: *Tensor, G: *Tensor, B: *Tensor, eps: f32, allocator: std.mem.Allocator) !*Tensor {
    const D = self.shape.dims[self.shape.len - 1];
    if (G.data.len != D or B.data.len != D) return error.ShapeMismatch;
    const M = self.data.len / D;
    const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);
    const d_f = @as(f32, @floatFromInt(D));

    for (0..M) |i| {
        const row_in = self.data[i * D .. (i + 1) * D];
        const row_out = Y.data[i * D .. (i + 1) * D];

        var sum_x: f32 = 0.0;
        for (row_in) |val| sum_x += val;
        const mean_val = sum_x / d_f;

        var var_sum: f32 = 0.0;
        for (row_in) |val| {
            const diff = val - mean_val;
            var_sum += diff * diff;
        }
        const inv_std = 1.0 / @sqrt(var_sum / d_f + eps);

        for (0..D) |j| {
            row_out[j] = (row_in[j] - mean_val) * inv_std * G.data[j] + B.data[j];
        }
    }
    return Y;
}

/// 一维批量归一化统一前向核函数 (1-Dimensional Batch Normalization, BatchNorm1d)
/// 支持二维 `[N, C]` 与三维 `[N, C, L]` 张量
pub fn batchNorm1d(
    self: *Tensor,
    gamma: *Tensor,
    beta: *Tensor,
    running_mean: ?*Tensor,
    running_var: ?*Tensor,
    training: bool,
    eps: f32,
    momentum: f32,
    saved_mean_out: ?[]f32,
    saved_inv_std_out: ?[]f32,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 2 and self.shape.len != 3) return error.IncompatibleDimensions;
    const N = self.shape.dims[0];
    const C = self.shape.dims[1];
    const L = if (self.shape.len == 3) self.shape.dims[2] else 1;
    if (gamma.data.len != C or beta.data.len != C) return error.ShapeMismatch;
    if (running_mean) |rm| {
        if (rm.data.len != C) return error.ShapeMismatch;
    }
    if (running_var) |rv| {
        if (rv.data.len != C) return error.ShapeMismatch;
    }

    const x_contig = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (!self.isContiguous()) x_contig.deinit(allocator);

    const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);
    errdefer Y.deinit(allocator);

    const m = N * L;
    const m_f = @as(f32, @floatFromInt(m));

    for (0..C) |c_| {
        var mean_val: f32 = 0.0;
        var inv_std: f32 = 1.0;

        if (training) {
            var sum_x: f32 = 0.0;
            for (0..N) |n| {
                const start_idx = (n * C + c_) * L;
                for (x_contig.data[start_idx .. start_idx + L]) |val| {
                    sum_x += val;
                }
            }
            mean_val = sum_x / m_f;

            var var_sum: f32 = 0.0;
            for (0..N) |n| {
                const start_idx = (n * C + c_) * L;
                for (x_contig.data[start_idx .. start_idx + L]) |val| {
                    const diff = val - mean_val;
                    var_sum += diff * diff;
                }
            }
            const var_val = var_sum / m_f;
            inv_std = 1.0 / @sqrt(var_val + eps);

            if (running_mean) |rm| {
                rm.data[c_] = (1.0 - momentum) * rm.data[c_] + momentum * mean_val;
            }
            if (running_var) |rv| {
                const unbiased_var = if (m > 1) var_sum / @as(f32, @floatFromInt(m - 1)) else var_val;
                rv.data[c_] = (1.0 - momentum) * rv.data[c_] + momentum * unbiased_var;
            }
        } else {
            mean_val = if (running_mean) |rm| rm.data[c_] else if (saved_mean_out) |sm| sm[c_] else 0.0;
            const var_val = if (running_var) |rv| rv.data[c_] else 1.0;
            inv_std = if (running_var != null) 1.0 / @sqrt(var_val + eps) else if (saved_inv_std_out) |si| si[c_] else 1.0 / @sqrt(1.0 + eps);
        }

        if (saved_mean_out) |sm| sm[c_] = mean_val;
        if (saved_inv_std_out) |si| si[c_] = inv_std;

        const g = gamma.data[c_];
        const b = beta.data[c_];
        for (0..N) |n| {
            const start_idx = (n * C + c_) * L;
            for (0..L) |i| {
                Y.data[start_idx + i] = (x_contig.data[start_idx + i] - mean_val) * inv_std * g + b;
            }
        }
    }
    return Y;
}

/// 二维批量归一化统一前向核函数 (2-Dimensional Batch Normalization, BatchNorm2d)
/// 输入形状: `[N, C, H, W]`
pub fn batchNorm2d(
    self: *Tensor,
    gamma: *Tensor,
    beta: *Tensor,
    running_mean: ?*Tensor,
    running_var: ?*Tensor,
    training: bool,
    eps: f32,
    momentum: f32,
    saved_mean_out: ?[]f32,
    saved_inv_std_out: ?[]f32,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 4) return error.IncompatibleDimensions;
    const N = self.shape.dims[0];
    const C = self.shape.dims[1];
    const H = self.shape.dims[2];
    const W = self.shape.dims[3];
    if (gamma.data.len != C or beta.data.len != C) return error.ShapeMismatch;
    if (running_mean) |rm| {
        if (rm.data.len != C) return error.ShapeMismatch;
    }
    if (running_var) |rv| {
        if (rv.data.len != C) return error.ShapeMismatch;
    }

    const x_contig = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (!self.isContiguous()) x_contig.deinit(allocator);

    const Y = try zeros(allocator, &.{ N, C, H, W });
    errdefer Y.deinit(allocator);

    const spatial_size = H * W;
    const m = N * spatial_size;
    const m_f = @as(f32, @floatFromInt(m));

    for (0..C) |c_| {
        var mean_val: f32 = 0.0;
        var inv_std: f32 = 1.0;

        if (training) {
            var sum_x: f32 = 0.0;
            for (0..N) |n| {
                const start_idx = (n * C + c_) * spatial_size;
                for (x_contig.data[start_idx .. start_idx + spatial_size]) |val| {
                    sum_x += val;
                }
            }
            mean_val = sum_x / m_f;

            var var_sum: f32 = 0.0;
            for (0..N) |n| {
                const start_idx = (n * C + c_) * spatial_size;
                for (x_contig.data[start_idx .. start_idx + spatial_size]) |val| {
                    const diff = val - mean_val;
                    var_sum += diff * diff;
                }
            }
            const var_val = var_sum / m_f;
            inv_std = 1.0 / @sqrt(var_val + eps);

            if (running_mean) |rm| {
                rm.data[c_] = (1.0 - momentum) * rm.data[c_] + momentum * mean_val;
            }
            if (running_var) |rv| {
                const unbiased_var = if (m > 1) var_sum / @as(f32, @floatFromInt(m - 1)) else var_val;
                rv.data[c_] = (1.0 - momentum) * rv.data[c_] + momentum * unbiased_var;
            }
        } else {
            mean_val = if (running_mean) |rm| rm.data[c_] else if (saved_mean_out) |sm| sm[c_] else 0.0;
            const var_val = if (running_var) |rv| rv.data[c_] else 1.0;
            inv_std = if (running_var != null) 1.0 / @sqrt(var_val + eps) else if (saved_inv_std_out) |si| si[c_] else 1.0 / @sqrt(1.0 + eps);
        }

        if (saved_mean_out) |sm| sm[c_] = mean_val;
        if (saved_inv_std_out) |si| si[c_] = inv_std;

        const g = gamma.data[c_];
        const b = beta.data[c_];
        for (0..N) |n| {
            const start_idx = (n * C + c_) * spatial_size;
            for (0..spatial_size) |i| {
                Y.data[start_idx + i] = (x_contig.data[start_idx + i] - mean_val) * inv_std * g + b;
            }
        }
    }
    return Y;
}

/// 分组归一化统一前向核函数 (Group Normalization, GroupNorm)
/// 支持任意维度 `>= 2` 的输入张量 `[N, C, ...]`，要求 `C % num_groups == 0`
pub fn groupNorm(
    self: *Tensor,
    gamma: *Tensor,
    beta: *Tensor,
    num_groups: usize,
    eps: f32,
    saved_mean_out: ?[]f32,
    saved_inv_std_out: ?[]f32,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len < 2) return error.IncompatibleDimensions;
    const N = self.shape.dims[0];
    const C = self.shape.dims[1];
    if (num_groups == 0 or C == 0 or C % num_groups != 0) return error.ShapeMismatch;
    if (gamma.data.len != C or beta.data.len != C) return error.ShapeMismatch;

    const x_contig = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (!self.isContiguous()) x_contig.deinit(allocator);

    const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);
    errdefer Y.deinit(allocator);

    var spatial_size: usize = 1;
    for (2..self.shape.len) |d| {
        spatial_size *= self.shape.dims[d];
    }
    const c_per_g = C / num_groups;
    const group_elems = c_per_g * spatial_size;
    const group_elems_f = @as(f32, @floatFromInt(group_elems));

    for (0..N) |n| {
        for (0..num_groups) |g| {
            const group_offset = (n * num_groups + g) * group_elems;
            const group_slice = x_contig.data[group_offset .. group_offset + group_elems];

            var sum_x: f32 = 0.0;
            for (group_slice) |val| sum_x += val;
            const mean_val = sum_x / group_elems_f;

            var var_sum: f32 = 0.0;
            for (group_slice) |val| {
                const diff = val - mean_val;
                var_sum += diff * diff;
            }
            const inv_std = 1.0 / @sqrt(var_sum / group_elems_f + eps);

            const ng_idx = n * num_groups + g;
            if (saved_mean_out) |sm| sm[ng_idx] = mean_val;
            if (saved_inv_std_out) |si| si[ng_idx] = inv_std;

            for (0..c_per_g) |cg| {
                const c_idx = g * c_per_g + cg;
                const g_val = gamma.data[c_idx];
                const b_val = beta.data[c_idx];
                const ch_offset = (n * C + c_idx) * spatial_size;
                for (0..spatial_size) |s| {
                    Y.data[ch_offset + s] = (x_contig.data[ch_offset + s] - mean_val) * inv_std * g_val + b_val;
                }
            }
        }
    }
    return Y;
}

/// 随机失活掩码前向核函数 (Dropout Mask Application)
pub fn applyDropoutMask(self: *Tensor, mask: []const f32, allocator: std.mem.Allocator) !*Tensor {
    const x_contig = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (!self.isContiguous()) x_contig.deinit(allocator);

    if (mask.len != x_contig.data.len) return error.ShapeMismatch;
    const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);
    for (x_contig.data, Y.data, mask) |x_val, *y_val, m_val| {
        y_val.* = x_val * m_val;
    }
    return Y;
}

/// 旋转位置编码前向核函数 (Rotary Position Embedding, RoPE)
pub fn rope(self: *Tensor, start_pos: usize, options: RopeOptions, allocator: std.mem.Allocator) !*Tensor {
    const rotary_offset = options.rotary_offset;
    const base = options.base;
    const D = self.shape.dims[self.shape.len - 1];
    if (rotary_offset > D) return error.DimensionOutOfBounds;
    const T = if (self.shape.len >= 2) self.shape.dims[self.shape.len - 2] else 1;
    const outer = self.data.len / (T * D);
    const rot_dim = D - rotary_offset;
    const half = rot_dim / 2;
    const rot_dim_f = @as(f32, @floatFromInt(rot_dim));

    const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);
    for (0..outer) |o| {
        for (0..T) |t| {
            const row_in = self.data[(o * T + t) * D .. (o * T + t + 1) * D];
            const row_out = Y.data[(o * T + t) * D .. (o * T + t + 1) * D];
            if (rotary_offset > 0) {
                @memcpy(row_out[0..rotary_offset], row_in[0..rotary_offset]);
            }
            const pos_f = @as(f32, @floatFromInt(start_pos + t));
            for (0..half) |i| {
                const freq = 1.0 / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / rot_dim_f);
                const theta = pos_f * freq;
                const cos_t = @cos(theta);
                const sin_t = @sin(theta);
                const x0 = row_in[rotary_offset + 2 * i];
                const x1 = row_in[rotary_offset + 2 * i + 1];
                row_out[rotary_offset + 2 * i] = x0 * cos_t - x1 * sin_t;
                row_out[rotary_offset + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
            }
            if (2 * half < rot_dim) {
                row_out[D - 1] = row_in[D - 1];
            }
        }
    }
    return Y;
}

pub fn repeatKV(self: *Tensor, groups: usize, allocator: std.mem.Allocator) !*Tensor {
    if (self.shape.len != 4) return error.IncompatibleDimensions;
    const B = self.shape.dims[0];
    const n_kv = self.shape.dims[1];
    const T = self.shape.dims[2];
    const hs = self.shape.dims[3];
    const nh = n_kv * groups;

    const Y = try zeros(allocator, &.{ B, nh, T, hs });
    const head_bytes = T * hs;
    for (0..B) |b| {
        for (0..n_kv) |kv_h| {
            const src = self.data[((b * n_kv + kv_h) * head_bytes) .. ((b * n_kv + kv_h + 1) * head_bytes)];
            for (0..groups) |g| {
                const h = kv_h * groups + g;
                const dest = Y.data[((b * nh + h) * head_bytes) .. ((b * nh + h + 1) * head_bytes)];
                @memcpy(dest, src);
            }
        }
    }
    return Y;
}

pub fn batchMatMul(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator) !*Tensor {
    if (self.shape.len != 4 or other.shape.len != 4) {
        return error.IncompatibleDimensions;
    }
    if (self.shape.dims[0] != other.shape.dims[0] or self.shape.dims[1] != other.shape.dims[1] or self.shape.dims[3] != other.shape.dims[2]) {
        return error.ShapeMismatch;
    }

    const batch_size = self.shape.dims[0];
    const num_heads = self.shape.dims[1];
    const M = self.shape.dims[2];
    const K = self.shape.dims[3];
    const N = other.shape.dims[3];

    const C = try zeros(allocator, &.{ batch_size, num_heads, M, N });

    const sA_b = self.strides.dims[0];
    const sA_h = self.strides.dims[1];
    const sB_b = other.strides.dims[0];
    const sB_h = other.strides.dims[1];
    const sC_b = C.strides.dims[0];
    const sC_h = C.strides.dims[1];

    for (0..batch_size) |b| {
        for (0..num_heads) |h| {
            const ptrA = self.data.ptr + b * sA_b + h * sA_h;
            const ptrB = other.data.ptr + b * sB_b + h * sB_h;
            const ptrC = C.data.ptr + b * sC_b + h * sC_h;

            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasNoTrans,
                @intCast(M),
                @intCast(N),
                @intCast(K),
                1.0,
                ptrA,
                @intCast(K),
                ptrB,
                @intCast(N),
                0.0,
                ptrC,
                @intCast(N),
            );
        }
    }
    return C;
}

inline fn indexScalarToUsize(val: anytype, vocab_size: usize) !usize {
    return types.indexScalarToUsize(@TypeOf(val), val, vocab_size);
}

pub fn embedding(self: *Tensor, indices: anytype, allocator: std.mem.Allocator) !*Tensor {
    if (self.shape.len != 2) return error.IncompatibleDimensions;
    const VocabSize = self.shape.dims[0];
    const D = self.shape.dims[1];

    const IdxT = @TypeOf(indices);
    const idx_info = @typeInfo(IdxT);
    const is_tensor_like = idx_info == .pointer and idx_info.pointer.size == .one and
        @typeInfo(idx_info.pointer.child) == .@"struct" and
        @hasField(idx_info.pointer.child, "shape");

    if (is_tensor_like) {
        const in_rank = indices.shape.len;
        if (in_rank == 0 or in_rank >= 8) return error.MaxDimensionsExceeded;
        var out_dims = [_]usize{0} ** 8;
        for (0..in_rank) |d| out_dims[d] = indices.shape.dims[d];
        out_dims[in_rank] = D;

        const Y = try zeros(allocator, out_dims[0 .. in_rank + 1]);
        errdefer free(allocator, Y);

        const num_indices = indices.shape.numel();
        if (indices.isContiguous() and indices.data.len >= num_indices) {
            for (0..num_indices) |i| {
                const idx = try indexScalarToUsize(indices.data[i], VocabSize);
                const w_row = self.data[idx * D .. (idx + 1) * D];
                const y_row = Y.data[i * D .. (i + 1) * D];
                @memcpy(y_row, w_row);
            }
        } else {
            var coord = [_]usize{0} ** 8;
            for (0..num_indices) |i| {
                var flat_idx: usize = 0;
                for (0..in_rank) |d| flat_idx += coord[d] * indices.strides.dims[d];
                const idx = try indexScalarToUsize(indices.data[flat_idx], VocabSize);
                const w_row = self.data[idx * D .. (idx + 1) * D];
                const y_row = Y.data[i * D .. (i + 1) * D];
                @memcpy(y_row, w_row);

                var d = in_rank;
                while (d > 0) {
                    d -= 1;
                    coord[d] += 1;
                    if (coord[d] < indices.shape.dims[d]) break;
                    coord[d] = 0;
                }
            }
        }
        return Y;
    } else {
        const T_len = indices.len;
        const Y = try zeros(allocator, &.{ T_len, D });
        errdefer free(allocator, Y);
        for (0..T_len) |i| {
            const idx = try indexScalarToUsize(indices[i], VocabSize);
            const w_row = self.data[idx * D .. (idx + 1) * D];
            const y_row = Y.data[i * D .. (i + 1) * D];
            @memcpy(y_row, w_row);
        }
        return Y;
    }
}
