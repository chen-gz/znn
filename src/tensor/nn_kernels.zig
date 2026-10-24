const std = @import("std");
const c = @import("../backend/cblas.zig");
const core = @import("core.zig");
const types = @import("types.zig");
const conv_pool = @import("conv_pool.zig");
const Tensor = core.Tensor;
const ConvOptions = types.ConvOptions;
const PoolOptions = types.PoolOptions;
const rope_mod = @import("../backend/rope.zig");
const RopeOptions = rope_mod.RopeOptions;
const RopeTable = @import("rope.zig").RopeTable;
const ops_mod = @import("ops.zig");
const zeros = ops_mod.zeros;
const free = ops_mod.free;
const shape_mod = @import("shape.zig");
const vec_math = @import("../backend/vec_math.zig");

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

/// 沿最后一维的数值稳定 Softmax：每行先减去行最大值再求指数，避免上溢
pub fn softmax(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
    if (self.shape.len == 0) return error.IncompatibleDimensions;
    const x = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (x != self) x.deinit(allocator);

    const D = x.shape.dims[x.shape.len - 1];
    if (D == 0) return error.DivisionByZero;
    const M = x.shape.numel() / D;
    const C = try zeros(allocator, x.shape.dims[0..x.shape.len]);

    for (0..M) |i| {
        vec_math.softmaxRow(x.data[i * D .. (i + 1) * D], C.data[i * D .. (i + 1) * D]);
    }
    return C;
}

/// 均方根层归一化前向核函数 (Root Mean Square Layer Normalization, RMSNorm)
pub fn rmsNorm(self: *Tensor, G: *Tensor, eps: f32, allocator: std.mem.Allocator) !*Tensor {
    if (self.shape.len == 0) return error.IncompatibleDimensions;
    const x = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (x != self) x.deinit(allocator);

    const g_c = if (G.isContiguous()) G else try G.contiguous(allocator);
    defer if (g_c != G) g_c.deinit(allocator);

    const D = x.shape.dims[x.shape.len - 1];
    if (D == 0) return error.DivisionByZero;
    if (g_c.shape.numel() != D) return error.ShapeMismatch;
    const M = x.shape.numel() / D;
    const Y = try zeros(allocator, x.shape.dims[0..x.shape.len]);

    for (0..M) |i| {
        const row_in = x.data[i * D .. (i + 1) * D];
        const row_out = Y.data[i * D .. (i + 1) * D];

        const sum_x2 = vec_math.sumSquares(row_in);
        const inv_rms = 1.0 / @sqrt(sum_x2 / @as(f32, @floatFromInt(D)) + eps);

        for (row_in, row_out, g_c.data[0..D]) |x_val, *y_val, g_val| {
            y_val.* = x_val * inv_rms * g_val;
        }
    }
    return Y;
}

/// 标准层归一化前向核函数 (Layer Normalization, LayerNorm)
pub fn layerNorm(self: *Tensor, G: *Tensor, B: *Tensor, eps: f32, allocator: std.mem.Allocator) !*Tensor {
    if (self.shape.len == 0) return error.IncompatibleDimensions;
    const x = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (x != self) x.deinit(allocator);

    const g_c = if (G.isContiguous()) G else try G.contiguous(allocator);
    defer if (g_c != G) g_c.deinit(allocator);

    const b_c = if (B.isContiguous()) B else try B.contiguous(allocator);
    defer if (b_c != B) b_c.deinit(allocator);

    const D = x.shape.dims[x.shape.len - 1];
    if (D == 0) return error.DivisionByZero;
    if (g_c.shape.numel() != D or b_c.shape.numel() != D) return error.ShapeMismatch;
    const M = x.shape.numel() / D;
    const Y = try zeros(allocator, x.shape.dims[0..x.shape.len]);
    const d_f = @as(f32, @floatFromInt(D));

    for (0..M) |i| {
        const row_in = x.data[i * D .. (i + 1) * D];
        const row_out = Y.data[i * D .. (i + 1) * D];

        const mean_val = vec_math.sum(row_in) / d_f;
        const var_sum = vec_math.sumSquaredDeviations(row_in, mean_val);
        const inv_std = 1.0 / @sqrt(var_sum / d_f + eps);

        for (row_in, row_out, g_c.data[0..D], b_c.data[0..D]) |x_val, *y_val, g_val, b_val| {
            y_val.* = (x_val - mean_val) * inv_std * g_val + b_val;
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
    return batchNormChannels(self, gamma, beta, running_mean, running_var, training, eps, momentum, saved_mean_out, saved_inv_std_out, allocator);
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
    return batchNormChannels(self, gamma, beta, running_mean, running_var, training, eps, momentum, saved_mean_out, saved_inv_std_out, allocator);
}

/// 批量归一化 (Batch Normalization, BatchNorm) 共享内核：输入视为 `[N, C, S]`，
/// 其中 `S` 为第 2 维之后所有空间维度之积（BatchNorm1d 的 `L` 或 BatchNorm2d 的 `H * W`），按通道 `C` 统计
fn batchNormChannels(
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
    const N = self.shape.dims[0];
    const C = self.shape.dims[1];
    var spatial_size: usize = 1;
    for (2..self.shape.len) |d| spatial_size *= self.shape.dims[d];
    if (gamma.data.len != C or beta.data.len != C) return error.ShapeMismatch;
    if (running_mean) |rm| {
        if (rm.data.len != C) return error.ShapeMismatch;
    }
    if (running_var) |rv| {
        if (rv.data.len != C) return error.ShapeMismatch;
    }
    if (saved_mean_out) |sm| {
        if (sm.len < C) return error.ShapeMismatch;
    }
    if (saved_inv_std_out) |si| {
        if (si.len < C) return error.ShapeMismatch;
    }

    const x_contig = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (!self.isContiguous()) x_contig.deinit(allocator);

    const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);
    errdefer Y.deinit(allocator);

    const m = N * spatial_size;
    const m_f = @as(f32, @floatFromInt(m));

    for (0..C) |c_| {
        var mean_val: f32 = 0.0;
        var inv_std: f32 = 1.0;

        if (training) {
            var sum_x: f32 = 0.0;
            for (0..N) |n| {
                const start_idx = (n * C + c_) * spatial_size;
                sum_x += vec_math.sum(x_contig.data[start_idx .. start_idx + spatial_size]);
            }
            mean_val = sum_x / m_f;

            var var_sum: f32 = 0.0;
            for (0..N) |n| {
                const start_idx = (n * C + c_) * spatial_size;
                var_sum += vec_math.sumSquaredDeviations(x_contig.data[start_idx .. start_idx + spatial_size], mean_val);
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

            const mean_val = vec_math.sum(group_slice) / group_elems_f;
            const var_sum = vec_math.sumSquaredDeviations(group_slice, mean_val);
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
    const x_contig = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (x_contig != self) x_contig.deinit(allocator);

    const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);
    for (x_contig.data[0..total], Y.data, mask) |x_val, *y_val, m_val| {
        y_val.* = x_val * m_val;
    }
    return Y;
}

/// 旋转位置编码统一前向核函数 (Rotary Position Embedding, RoPE)
/// 支持经典相邻配对 (Interleaved) 与主流半切分配对 (Split-Half) 两种拓扑，
/// 以及任意频率基数 (base)、旋转子空间偏移 (rotary_offset) 与部分旋转因子 (partial_rotary_factor)。
/// 旋转由共享的 `RopeTable` 完成：每个位置只计算一次三角函数表并复用于全部 outer 行（批次与注意力头）。
pub fn rope(self: *Tensor, start_pos: usize, options: RopeOptions, allocator: std.mem.Allocator) !*Tensor {
    if (self.shape.len == 0) return error.IncompatibleDimensions;
    const x = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (x != self) x.deinit(allocator);

    const D = x.shape.dims[x.shape.len - 1];
    const T = if (x.shape.len >= 2) x.shape.dims[x.shape.len - 2] else 1;
    const table = try RopeTable.init(D, options);

    const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);
    errdefer free(allocator, Y);
    if (T * D == 0) return Y;
    const outer = x.shape.numel() / (T * D);

    var cos_buf: [rope_mod.max_angles]f32 = undefined;
    var sin_buf: [rope_mod.max_angles]f32 = undefined;
    const cos_tab = cos_buf[0..table.num_angles];
    const sin_tab = sin_buf[0..table.num_angles];
    for (0..T) |t| {
        table.angles(start_pos + t, cos_tab, sin_tab);
        for (0..outer) |o| {
            // 先复制整行 (仍在一级缓存中)，再原地旋转参与旋转的维度
            const row = Y.data[(o * T + t) * D ..][0..D];
            @memcpy(row, x.data[(o * T + t) * D ..][0..D]);
            table.applyRow(row, cos_tab, sin_tab, false);
        }
    }
    return Y;
}

/// 分组查询注意力 (Grouped-Query Attention, GQA) 键值头复制：`[B, n_kv, T, hs]` -> `[B, n_kv * groups, T, hs]`
pub fn repeatKV(self: *Tensor, groups: usize, allocator: std.mem.Allocator) !*Tensor {
    if (self.shape.len != 4) return error.IncompatibleDimensions;
    const x = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (x != self) x.deinit(allocator);
    const B = self.shape.dims[0];
    const n_kv = self.shape.dims[1];
    const T = self.shape.dims[2];
    const hs = self.shape.dims[3];
    const nh = n_kv * groups;

    const Y = try zeros(allocator, &.{ B, nh, T, hs });
    const head_bytes = T * hs;
    for (0..B) |b| {
        for (0..n_kv) |kv_h| {
            const src = x.data[((b * n_kv + kv_h) * head_bytes)..((b * n_kv + kv_h + 1) * head_bytes)];
            for (0..groups) |g| {
                const h = kv_h * groups + g;
                const dest = Y.data[((b * nh + h) * head_bytes)..((b * nh + h + 1) * head_bytes)];
                @memcpy(dest, src);
            }
        }
    }
    return Y;
}

/// 二维矩阵块在基础线性代数子程序 (Basic Linear Algebra Subprograms, BLAS) 中的行优先描述：
/// 是否以转置方式读取，以及主维跨度 (Leading Dimension, ld)
pub const BlasMatrix = struct { trans: c_int, ld: usize };

/// 判断跨度为 (`s_row`, `s_col`) 的 `[rows, cols]` 矩阵块能否直接交给 BLAS：
/// 行优先连续行（列跨度为 1）按非转置读取，列优先连续列（如 `transposeView` 得到的转置视图，行跨度为 1）按转置读取；
/// 两者都不满足时返回 null，由调用方物化为连续副本
pub fn blasMatrix(rows: usize, cols: usize, s_row: usize, s_col: usize) ?BlasMatrix {
    if (cols <= 1 or s_col == 1) {
        if (rows <= 1) return .{ .trans = c.CblasNoTrans, .ld = @max(cols, 1) };
        if (s_row >= cols) return .{ .trans = c.CblasNoTrans, .ld = @max(s_row, 1) };
    }
    if (rows <= 1 or s_row == 1) {
        if (cols <= 1) return .{ .trans = c.CblasTrans, .ld = @max(rows, 1) };
        if (s_col >= rows) return .{ .trans = c.CblasTrans, .ld = @max(s_col, 1) };
    }
    return null;
}

/// 四维批量矩阵乘法 `[B, H, M, K] x [B, H, K, N] -> [B, H, M, N]`：
/// 支持任意批次/头跨度，以及最后两维为行优先或转置（列优先）布局的视图；其余布局先物化为连续副本
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

    const a_layout = blasMatrix(M, K, self.strides.dims[2], self.strides.dims[3]);
    const a = if (a_layout != null) self else try self.contiguous(allocator);
    defer if (a != self) a.deinit(allocator);
    const b_layout = blasMatrix(K, N, other.strides.dims[2], other.strides.dims[3]);
    const b = if (b_layout != null) other else try other.contiguous(allocator);
    defer if (b != other) b.deinit(allocator);
    const a_mat = a_layout orelse BlasMatrix{ .trans = c.CblasNoTrans, .ld = @max(K, 1) };
    const b_mat = b_layout orelse BlasMatrix{ .trans = c.CblasNoTrans, .ld = @max(N, 1) };

    const C = try zeros(allocator, &.{ batch_size, num_heads, M, N });
    if (M == 0 or N == 0) return C;

    const sA_b = a.strides.dims[0];
    const sA_h = a.strides.dims[1];
    const sB_b = b.strides.dims[0];
    const sB_h = b.strides.dims[1];
    const sC_b = C.strides.dims[0];
    const sC_h = C.strides.dims[1];

    for (0..batch_size) |bi| {
        for (0..num_heads) |h| {
            const ptrA = a.data.ptr + bi * sA_b + h * sA_h;
            const ptrB = b.data.ptr + bi * sB_b + h * sB_h;
            const ptrC = C.data.ptr + bi * sC_b + h * sC_h;

            c.cblas_sgemm(
                c.CblasRowMajor,
                a_mat.trans,
                b_mat.trans,
                @intCast(M),
                @intCast(N),
                @intCast(K),
                1.0,
                ptrA,
                @intCast(a_mat.ld),
                ptrB,
                @intCast(b_mat.ld),
                0.0,
                ptrC,
                @intCast(N),
            );
        }
    }
    return C;
}

/// 嵌入查表 (Embedding Lookup) 内核：out[t, :] = table[ids[t], :] * scale，写入 [ids.len, dim] 行主序缓冲区。
/// - `Elem` 为 f32 (单精度嵌入表) 或 u16 (脑浮点 16 (Brain Floating Point 16, BF16) 原始位，展开为单精度后缩放)；
/// - `ids` 为任意整数或浮点元素类型的切片，词表大小为 table.len / dim，越界、负数或非整数 ID 返回 `error.IndexOutOfBounds`；
/// - scale = 1 时为纯查表 (如 Gemma 的 sqrt(hidden_size) 嵌入缩放即通过 scale 传入)。
pub fn embeddingLookupInto(comptime Elem: type, table: []const Elem, ids: anytype, dim: usize, scale: f32, out: []f32) !void {
    comptime std.debug.assert(Elem == f32 or Elem == u16);
    if (dim == 0 or out.len != ids.len * dim) return error.ShapeMismatch;
    const vocab = table.len / dim;
    const IdxT = std.meta.Elem(@TypeOf(ids));
    for (ids, 0..) |raw, t| {
        const id = try types.indexScalarToUsize(IdxT, raw, vocab);
        const row = table[id * dim ..][0..dim];
        const dst = out[t * dim ..][0..dim];
        if (Elem == u16) {
            vec_math.widenBf16(dst, row, scale);
        } else if (scale == 1.0) {
            @memcpy(dst, row);
        } else {
            vec_math.scaleCopy(dst, row, scale);
        }
    }
}

pub fn embedding(self: *Tensor, indices: anytype, allocator: std.mem.Allocator) !*Tensor {
    if (self.shape.len != 2) return error.IncompatibleDimensions;
    const VocabSize = self.shape.dims[0];
    const D = self.shape.dims[1];
    // 按 `idx * D` 整行拷贝要求嵌入表行优先连续
    if (!self.isContiguous()) {
        const table = try self.contiguous(allocator);
        defer free(allocator, table);
        return embedding(table, indices, allocator);
    }
    const table = self.data[0 .. VocabSize * D];

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
            try embeddingLookupInto(f32, table, indices.data[0..num_indices], D, 1.0, Y.data);
        } else {
            const ElemT = std.meta.Elem(@TypeOf(indices.data));
            var coord: [8]usize = @splat(0);
            for (0..num_indices) |i| {
                var flat_idx: usize = 0;
                for (0..in_rank) |d| flat_idx += coord[d] * indices.strides.dims[d];
                const idx = try types.indexScalarToUsize(ElemT, indices.data[flat_idx], VocabSize);
                @memcpy(Y.data[i * D .. (i + 1) * D], table[idx * D .. (idx + 1) * D]);

                indices.shape.incrementCoord(&coord);
            }
        }
        return Y;
    } else {
        const Y = try zeros(allocator, &.{ indices.len, D });
        errdefer free(allocator, Y);
        try embeddingLookupInto(f32, table, indices, D, 1.0, Y.data);
        return Y;
    }
}

/// 前 K 大元素选择结果 (Top-K Selection Result)
pub const TopKResult = struct {
    values: *Tensor,
    indices: *Tensor,
};

/// 无分支前 K 大元素原地选择内核 (Branchless Top-K Selection In-Place Kernel)：
/// 沿最后一维 `E` 选出最大的 `k` 个元素及其下标。元素按 `reductions.totalLessThan` 全序比较
/// （非数 (Not a Number, NaN) 视为大于一切数值，与降序 `sort` 一致），相同数值按较小列下标优先打破平局，
/// 因此每行的排名构成 `0..E` 的一个排列，每个排名 `r in 0..k` 严格对应唯一下标。
/// 排名由无分支比较计数得到，结果以条件选择写入对应槽位（不使用乘以 0/1 掩码的累加，Inf 与 NaN 不会污染其他槽位）。
/// 输入可为任意跨步视图；输出 `values_out` / `indices_out` 须为连续张量。
pub fn topKInto(self: *const Tensor, k: usize, values_out: *Tensor, indices_out: *Tensor) !void {
    if (self.shape.len == 0) return error.IncompatibleDimensions;
    const E = self.shape.dims[self.shape.len - 1];
    if (k == 0 or k > E) return error.InvalidArgument;
    const N = self.shape.numel() / E;
    if (values_out.shape.numel() != N * k or indices_out.shape.numel() != N * k) return error.ShapeMismatch;
    if (!values_out.isContiguous() or !indices_out.isContiguous()) return error.NonContiguousTensor;

    // 按行优先顺序遍历除最后一维外的所有坐标，每行的元素 `j` 位于 `base + j * s_last`
    const s_last = self.strides.dims[self.shape.len - 1];
    var outer_shape = self.shape;
    outer_shape.len -= 1;
    var outer_strides = self.strides;
    outer_strides.len -= 1;
    var it = shape_mod.RowIter(1).init(outer_shape, .{outer_strides});
    const s_outer = it.inner_strides[0];
    var n: usize = 0;
    while (it.next()) |offs| {
        for (0..it.inner) |i| {
            topKRow(self.data, offs[0] + i * s_outer, s_last, E, values_out.data[n * k ..][0..k], indices_out.data[n * k ..][0..k]);
            n += 1;
        }
    }
}

/// 单行前 K 大选择：行内元素 `j` 位于 `data[base + j * stride]`
fn topKRow(data: []const f32, base: usize, stride: usize, E: usize, row_vals: []f32, row_idxs: []f32) void {
    for (0..E) |j| {
        const v_j = data[base + j * stride];
        const nan_j: usize = @intFromBool(std.math.isNan(v_j));
        var rank: usize = 0;
        for (0..E) |m| {
            const v_m = data[base + m * stride];
            const nan_m: usize = @intFromBool(std.math.isNan(v_m));
            // 全序下 v_m 严格大于 v_j，或两者等价（同为 NaN 或数值相等）且 m 更靠前
            const greater = (1 - nan_j) & (nan_m | @intFromBool(v_m > v_j));
            const equivalent = @intFromBool(v_m == v_j) | (nan_m & nan_j);
            rank += greater | (equivalent & @intFromBool(m < j));
        }
        const j_f: f32 = @floatFromInt(j);
        for (row_vals, row_idxs, 0..) |*val, *idx, r| {
            const hit = rank == r;
            val.* = if (hit) v_j else val.*;
            idx.* = if (hit) j_f else idx.*;
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
/// 对每个词元 `n in 0..N` 与选中的前 `K` 个槽位 `k in 0..K`，计算 `out[n, k] = x[n, k] * weights[e] (+ biases[e])`，
/// 其中 `e = indices[n, k]`，仅激活每词元选中的 `K` 个专家。
/// 实现按专家分组：先以稳定计数排序把 `N * K` 个 (词元, 槽位) 行按专家归并，再把每组输入行收集到暂存缓冲区，
/// 对每个专家只调用一次基础线性代数子程序 (Basic Linear Algebra Subprograms, BLAS) 单精度通用矩阵乘法
/// (Single-precision General Matrix Multiply, SGEMM)，最后把结果行（加上偏置）散射回 `out` 的原位置。
/// `allocator` 只用于分组下标与暂存缓冲区（大小为最大专家组行数 * (in_dim + out_dim)），返回前全部释放。
pub fn gatherMatMulInto(
    self: *const Tensor,
    indices: *const Tensor,
    weights: []const *Tensor,
    biases: ?[]const *Tensor,
    out: *Tensor,
    allocator: std.mem.Allocator,
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
    // 每个词元 (及槽位) 的输入向量必须连续存放，批次维与槽位维可为任意跨度
    if (self.strides.dims[self.shape.len - 1] != 1 and in_dim > 1) return error.NonContiguousTensor;
    if (!indices.isContiguous() or !out.isContiguous()) return error.NonContiguousTensor;
    if (out.shape.numel() != N * K * out_dim) return error.ShapeMismatch;
    for (weights) |w| {
        if (w.shape.len != 2 or w.shape.dims[0] != in_dim or w.shape.dims[1] != out_dim or !w.isContiguous()) return error.ShapeMismatch;
    }
    if (biases) |bs| {
        if (bs.len != E) return error.ShapeMismatch;
        for (bs) |b| {
            if (b.shape.numel() != out_dim or !b.isContiguous()) return error.ShapeMismatch;
        }
    }
    // 下标须为有限非负数；越界下标按设计截断到最后一个专家（与 `topKInto` 的输出范围一致）
    for (indices.data[0 .. N * K]) |raw| {
        if (!std.math.isFinite(raw) or raw < 0.0) return error.IndexOutOfBounds;
    }

    const rows = N * K;
    if (rows == 0 or out_dim == 0) return;

    const x_n_stride: usize = self.strides.dims[0];
    const x_k_stride: usize = if (self.shape.len == 3) self.strides.dims[1] else 0;

    // 1. 按专家分组行号 `r = n * K + k`
    var groups = try ExpertGroups.init(allocator, indices.data[0..rows], E);
    defer groups.deinit(allocator);

    // 2. 每个专家：收集输入行 -> 一次 SGEMM -> 散射结果行并加偏置
    const scratch = try allocator.alloc(f32, groups.max_group * (in_dim + out_dim));
    defer allocator.free(scratch);
    const x_buf = scratch[0 .. groups.max_group * in_dim];
    const y_buf = scratch[groups.max_group * in_dim ..];

    for (0..E) |e| {
        const group = groups.rowsOf(e);
        if (group.len == 0) continue;

        for (group, 0..) |r, g| {
            const x_ptr = self.data.ptr + (r / K) * x_n_stride + (r % K) * x_k_stride;
            @memcpy(x_buf[g * in_dim ..][0..in_dim], x_ptr[0..in_dim]);
        }

        c.cblas_sgemm(
            c.CblasRowMajor,
            c.CblasNoTrans,
            c.CblasNoTrans,
            @intCast(group.len),
            @intCast(out_dim),
            @intCast(in_dim),
            1.0,
            x_buf.ptr,
            @intCast(@max(in_dim, 1)),
            weights[e].data.ptr,
            @intCast(out_dim),
            0.0,
            y_buf.ptr,
            @intCast(out_dim),
        );

        for (group, 0..) |r, g| {
            const y_row = out.data[r * out_dim ..][0..out_dim];
            const src_row = y_buf[g * out_dim ..][0..out_dim];
            if (biases) |bs| {
                for (y_row, src_row, bs[e].data[0..out_dim]) |*y_val, s_val, b_val| y_val.* = s_val + b_val;
            } else {
                @memcpy(y_row, src_row);
            }
        }
    }
}

/// 把已校验为有限非负的浮点专家下标转换为专家序号：先在浮点域截断到最后一个专家再转换，
/// 避免超出 usize 范围的下标触发非法转换
inline fn expertOf(raw: f32, e_max: f32) usize {
    return @intFromFloat(@min(raw, e_max));
}

/// 按专家分组的行号表 (GatherMatMul 前向与反向共用)：对 `[N * K]` 个已校验为有限非负的专家下标做稳定计数排序，
/// `order` 按专家升序存放行号 `r = n * K + k`（同一专家内保持原顺序），越界下标截断到最后一个专家
pub const ExpertGroups = struct {
    /// `meta[0 .. E + 1]` 为各专家在 `order` 中的边界，其后为 `order`
    meta: []usize,
    num_experts: usize,
    /// 单个专家的最大行数，用于确定收集缓冲区大小
    max_group: usize,

    pub fn init(allocator: std.mem.Allocator, expert_indices: []const f32, num_experts: usize) !ExpertGroups {
        std.debug.assert(num_experts > 0);
        const rows = expert_indices.len;
        const meta = try allocator.alloc(usize, num_experts + 1 + rows);
        const bounds = meta[0 .. num_experts + 1];
        const order = meta[num_experts + 1 ..];
        const e_max: f32 = @floatFromInt(num_experts - 1);

        // 填充时把 `bounds[e]` 当作写游标，填充完毕后 `bounds[e]` 恰为专家 `e` 的结束位置；
        // 最后整体右移一位，使 `bounds[e] .. bounds[e + 1]` 为专家 `e` 的区间
        @memset(bounds, 0);
        for (expert_indices) |raw| bounds[expertOf(raw, e_max) + 1] += 1;
        var max_group: usize = 0;
        for (1..num_experts + 1) |e| {
            max_group = @max(max_group, bounds[e]);
            bounds[e] += bounds[e - 1];
        }
        for (expert_indices, 0..) |raw, r| {
            const e = expertOf(raw, e_max);
            order[bounds[e]] = r;
            bounds[e] += 1;
        }
        var e = num_experts;
        while (e > 0) : (e -= 1) bounds[e] = bounds[e - 1];
        bounds[0] = 0;

        return .{ .meta = meta, .num_experts = num_experts, .max_group = max_group };
    }

    pub fn deinit(self: *ExpertGroups, allocator: std.mem.Allocator) void {
        allocator.free(self.meta);
        self.* = undefined;
    }

    /// 专家 `e` 负责的行号 (升序)
    pub fn rowsOf(self: *const ExpertGroups, e: usize) []const usize {
        const bounds = self.meta[0 .. self.num_experts + 1];
        const order = self.meta[self.num_experts + 1 ..];
        return order[bounds[e]..bounds[e + 1]];
    }
};

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
    try gatherMatMulInto(self, indices, weights, biases, Y, allocator);
    return Y;
}
