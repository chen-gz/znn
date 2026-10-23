const std = @import("std");
const c = @import("../backend/cblas.zig");
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
    const x = if (self.isContiguous() and !self.is_view) self else try self.contiguous(allocator);
    defer if (!self.isContiguous() or self.is_view) x.deinit(allocator);

    const D = x.shape.dims[x.shape.len - 1];
    if (D == 0) return error.DivisionByZero;
    const M = x.shape.numel() / D;
    const C = try zeros(allocator, x.shape.dims[0..x.shape.len]);

    for (0..M) |i| {
        const row_in = x.data[i * D .. (i + 1) * D];
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
    const x = if (self.isContiguous() and !self.is_view) self else try self.contiguous(allocator);
    defer if (!self.isContiguous() or self.is_view) x.deinit(allocator);

    const g_c = if (G.isContiguous() and !G.is_view) G else try G.contiguous(allocator);
    defer if (!G.isContiguous() or G.is_view) g_c.deinit(allocator);

    const D = x.shape.dims[x.shape.len - 1];
    if (D == 0) return error.DivisionByZero;
    if (g_c.shape.numel() != D) return error.ShapeMismatch;
    const M = x.shape.numel() / D;
    const Y = try zeros(allocator, x.shape.dims[0..x.shape.len]);

    for (0..M) |i| {
        const row_in = x.data[i * D .. (i + 1) * D];
        const row_out = Y.data[i * D .. (i + 1) * D];

        var sum_x2: f32 = 0.0;
        for (row_in) |val| {
            sum_x2 += val * val;
        }
        const rms = @sqrt(sum_x2 / @as(f32, @floatFromInt(D)) + eps);

        for (row_in, row_out, g_c.data[0..D]) |x_val, *y_val, g_val| {
            y_val.* = x_val / rms * g_val;
        }
    }
    return Y;
}

/// 标准层归一化前向核函数 (Layer Normalization, LayerNorm)
pub fn layerNorm(self: *Tensor, G: *Tensor, B: *Tensor, eps: f32, allocator: std.mem.Allocator) !*Tensor {
    const x = if (self.isContiguous() and !self.is_view) self else try self.contiguous(allocator);
    defer if (!self.isContiguous() or self.is_view) x.deinit(allocator);

    const g_c = if (G.isContiguous() and !G.is_view) G else try G.contiguous(allocator);
    defer if (!G.isContiguous() or G.is_view) g_c.deinit(allocator);

    const b_c = if (B.isContiguous() and !B.is_view) B else try B.contiguous(allocator);
    defer if (!B.isContiguous() or B.is_view) b_c.deinit(allocator);

    const D = x.shape.dims[x.shape.len - 1];
    if (D == 0) return error.DivisionByZero;
    if (g_c.shape.numel() != D or b_c.shape.numel() != D) return error.ShapeMismatch;
    const M = x.shape.numel() / D;
    const Y = try zeros(allocator, x.shape.dims[0..x.shape.len]);
    const d_f = @as(f32, @floatFromInt(D));

    for (0..M) |i| {
        const row_in = x.data[i * D .. (i + 1) * D];
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
            row_out[j] = (row_in[j] - mean_val) * inv_std * g_c.data[j] + b_c.data[j];
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
    const total = self.shape.numel();
    if (mask.len != total) return error.ShapeMismatch;
    const x_contig = if (self.isContiguous() and !self.is_view) self else try self.contiguous(allocator);
    defer if (!self.isContiguous() or self.is_view) x_contig.deinit(allocator);

    const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);
    for (x_contig.data[0..total], Y.data, mask) |x_val, *y_val, m_val| {
        y_val.* = x_val * m_val;
    }
    return Y;
}

/// 旋转位置编码统一前向核函数 (Rotary Position Embedding, RoPE)
/// 支持经典相邻配对 (Interleaved) 与主流半切分配对 (Split-Half) 两种拓扑，
/// 以及任意频率基数 (base)、旋转子空间偏移 (rotary_offset) 与部分旋转因子 (partial_rotary_factor)。
pub fn rope(self: *Tensor, start_pos: usize, options: RopeOptions, allocator: std.mem.Allocator) !*Tensor {
    const x = if (self.isContiguous() and !self.is_view) self else try self.contiguous(allocator);
    defer if (!self.isContiguous() or self.is_view) x.deinit(allocator);

    const rotary_offset = options.rotary_offset;
    const base = options.base;
    const D = x.shape.dims[x.shape.len - 1];
    if (rotary_offset > D) return error.DimensionOutOfBounds;
    const T = if (x.shape.len >= 2) x.shape.dims[x.shape.len - 2] else 1;
    const outer = x.shape.numel() / (T * D);
    const rot_dim = D - rotary_offset;
    const half = rot_dim / 2;
    const rot_dim_f = @as(f32, @floatFromInt(rot_dim));

    const factor = std.math.clamp(options.partial_rotary_factor, 0.0, 1.0);
    const num_angles = @as(usize, @intFromFloat(@as(f32, @floatFromInt(half)) * factor));

    const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);
    for (0..outer) |o| {
        for (0..T) |t| {
            const row_in = x.data[(o * T + t) * D .. (o * T + t + 1) * D];
            const row_out = Y.data[(o * T + t) * D .. (o * T + t + 1) * D];

            // 先复制整行，后续仅更新参与旋转的维度
            @memcpy(row_out, row_in);

            const pos_f = @as(f32, @floatFromInt(start_pos + t));
            switch (options.mode) {
                .interleaved => {
                    for (0..num_angles) |i| {
                        const freq = 1.0 / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / rot_dim_f);
                        const theta = pos_f * freq;
                        const cos_t = @cos(theta);
                        const sin_t = @sin(theta);
                        const x0 = row_in[rotary_offset + 2 * i];
                        const x1 = row_in[rotary_offset + 2 * i + 1];
                        row_out[rotary_offset + 2 * i] = x0 * cos_t - x1 * sin_t;
                        row_out[rotary_offset + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
                    }
                },
                .split_half => {
                    for (0..num_angles) |i| {
                        const freq = 1.0 / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / rot_dim_f);
                        const theta = pos_f * freq;
                        const cos_t = @cos(theta);
                        const sin_t = @sin(theta);
                        const x0 = row_in[rotary_offset + i];
                        const x1 = row_in[rotary_offset + i + half];
                        row_out[rotary_offset + i] = x0 * cos_t - x1 * sin_t;
                        row_out[rotary_offset + i + half] = x1 * cos_t + x0 * sin_t;
                    }
                },
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
        var out_dims: [8]usize = @splat(0);
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
            var coord: [8]usize = @splat(0);
            for (0..num_indices) |i| {
                var flat_idx: usize = 0;
                for (0..in_rank) |d| flat_idx += coord[d] * indices.strides.dims[d];
                const idx = try indexScalarToUsize(indices.data[flat_idx], VocabSize);
                const w_row = self.data[idx * D .. (idx + 1) * D];
                const y_row = Y.data[i * D .. (i + 1) * D];
                @memcpy(y_row, w_row);

                indices.shape.incrementCoord(&coord);
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

/// 前 K 大元素选择结果 (Top-K Selection Result)
pub const TopKResult = struct {
    values: *Tensor,
    indices: *Tensor,
};

/// 无分支前 K 大元素原地选择内核 (Branchless Top-K Selection In-Place Kernel)：
/// 沿最后一维 `E` 选出最大的 `k` 个元素及其下标，全程使用无分支比较算术累加（零数据依赖 `if` 跳转），
/// 相同数值按较小列下标优先打破平局，确保每个排名 `r in 0..k` 严格对应唯一专家下标。
pub fn topKInto(self: *const Tensor, k: usize, values_out: *Tensor, indices_out: *Tensor) !void {
    if (self.shape.len == 0) return error.IncompatibleDimensions;
    const E = self.shape.dims[self.shape.len - 1];
    if (k == 0 or k > E) return error.InvalidArgument;
    const N = self.shape.numel() / E;

    @memset(values_out.data, 0.0);
    @memset(indices_out.data, 0.0);

    for (0..N) |n| {
        const row_in = self.data[n * E .. (n + 1) * E];
        const row_vals = values_out.data[n * k .. (n + 1) * k];
        const row_idxs = indices_out.data[n * k .. (n + 1) * k];

        for (0..E) |j| {
            const v_j = row_in[j];
            var rank: usize = 0;
            for (0..E) |m| {
                const v_m = row_in[m];
                const greater: usize = @intFromBool(v_m > v_j);
                const tie_earlier: usize = @intFromBool((v_m == v_j) and (m < j));
                rank += greater | tie_earlier;
            }
            for (0..k) |r| {
                const match_f: f32 = @floatFromInt(@intFromBool(rank == r));
                row_vals[r] += v_j * match_f;
                row_idxs[r] += @as(f32, @floatFromInt(j)) * match_f;
            }
        }
    }
}

/// 沿最后一维选出前 K 大元素及其对应下标 (Top-K Selection)
pub fn topK(self: *Tensor, k: usize, allocator: std.mem.Allocator) !TopKResult {
    if (self.shape.len == 0) return error.IncompatibleDimensions;
    const E = self.shape.dims[self.shape.len - 1];
    if (k == 0 or k > E) return error.InvalidArgument;

    var out_dims: [8]usize = @splat(0);
    for (0..self.shape.len - 1) |d| {
        out_dims[d] = self.shape.dims[d];
    }
    out_dims[self.shape.len - 1] = k;

    const values = try zeros(allocator, out_dims[0..self.shape.len]);
    errdefer free(allocator, values);
    const indices = try zeros(allocator, out_dims[0..self.shape.len]);
    errdefer free(allocator, indices);

    try topKInto(self, k, values, indices);
    return .{ .values = values, .indices = indices };
}

/// 索引收集矩阵乘法原地内核 (Indexed Gather Matrix Multiplication In-Place Kernel, GatherMatMul)：
/// 针对每个词元 `n in 0..N` 与选中的前 `K` 个槽位 `k in 0..K`，直接通过下标 `e = indices[n, k]`
/// 寻址第 `e` 个专家的权重矩阵 `weights[e]`（及可选偏置 `biases[e]`）并调用基础线性代数子程序 (Basic Linear Algebra Subprograms, BLAS)
/// 单精度通用矩阵乘法 (Single-precision General Matrix Multiply, SGEMM)，仅激活每词元选中的 `K` 个专家且无数据依赖分支。
pub fn gatherMatMulInto(
    self: *const Tensor,
    indices: *const Tensor,
    weights: []const *Tensor,
    biases: ?[]const *Tensor,
    out: *Tensor,
) !void {
    if (indices.shape.len != 2 or weights.len == 0) return error.IncompatibleDimensions;
    if (self.shape.len != 2 and self.shape.len != 3) return error.IncompatibleDimensions;

    const N = indices.shape.dims[0];
    const K = indices.shape.dims[1];
    const E = weights.len;
    const in_dim = weights[0].shape.dims[0];
    const out_dim = weights[0].shape.dims[1];

    if (self.shape.dims[0] != N or self.shape.dims[self.shape.len - 1] != in_dim) return error.ShapeMismatch;
    if (self.shape.len == 3 and self.shape.dims[1] != K) return error.ShapeMismatch;
    if (biases) |bs| {
        if (bs.len != E) return error.ShapeMismatch;
    }

    const x_n_stride: usize = self.strides.dims[0];
    const x_k_stride: usize = (self.shape.len - 2) * in_dim;

    for (0..N) |n| {
        for (0..K) |k| {
            const e_raw: usize = @intFromFloat(indices.data[n * K + k]);
            const e: usize = @min(e_raw, E - 1);
            const x_ptr = self.data.ptr + n * x_n_stride + k * x_k_stride;
            const w_ptr = weights[e].data.ptr;
            const y_offset = (n * K + k) * out_dim;
            const y_slice = out.data[y_offset .. y_offset + out_dim];

            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasNoTrans,
                1,
                @intCast(out_dim),
                @intCast(in_dim),
                1.0,
                x_ptr,
                @intCast(in_dim),
                w_ptr,
                @intCast(out_dim),
                0.0,
                y_slice.ptr,
                @intCast(out_dim),
            );

            if (biases) |bs| {
                const b_slice = bs[e].data[0..out_dim];
                for (y_slice, b_slice) |*y_val, b_val| {
                    y_val.* += b_val;
                }
            }
        }
    }
}

/// 索引收集矩阵乘法 (Indexed Gather Matrix Multiplication, GatherMatMul)：
/// 输入 `self` 为 `[N, in_dim]` 或 `[N, K, in_dim]`，`indices` 为 `[N, K]`，返回 `[N, K, out_dim]`。
pub fn gatherMatMul(
    self: *Tensor,
    indices: *Tensor,
    weights: []const *Tensor,
    biases: ?[]const *Tensor,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (indices.shape.len != 2 or weights.len == 0) return error.IncompatibleDimensions;
    const N = indices.shape.dims[0];
    const K = indices.shape.dims[1];
    const out_dim = weights[0].shape.dims[1];

    const Y = try zeros(allocator, &.{ N, K, out_dim });
    errdefer free(allocator, Y);
    try gatherMatMulInto(self, indices, weights, biases, Y);
    return Y;
}

