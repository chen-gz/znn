const std = @import("std");
const c = @import("../backend/cblas.zig");
const core = @import("core.zig");
const types = @import("types.zig");
const Tensor = core.Tensor;
const ConvOptions = types.ConvOptions;
const PoolOptions = types.PoolOptions;
const ops_mod = @import("ops.zig");
const zeros = ops_mod.zeros;

/// 二维卷积几何描述：一维卷积按 `H = KH = 1`、高度方向填充为 0 的二维卷积处理
const ConvGeometry = struct {
    N: usize,
    C_in: usize,
    H: usize,
    W: usize,
    C_out: usize,
    KH: usize,
    KW: usize,
    stride: usize,
    pad_h: usize,
    pad_w: usize,
    H_out: usize,
    W_out: usize,
};

/// 输入下标 `i * stride + k - padding` 落在 `[0, limit)` 内的 `i` 取值区间 `[lo, hi)`（`i < count`）
fn validRange(count: usize, limit: usize, k: usize, stride: usize, padding: usize) [2]usize {
    const lo = if (k >= padding) 0 else (padding - k + stride - 1) / stride;
    const bound = limit + padding;
    const hi = if (bound > k) @min(count, (bound - k + stride - 1) / stride) else 0;
    return .{ @min(lo, hi), hi };
}

/// 图像转列 (Image to Column, im2col) + 单精度通用矩阵乘法 (Single-precision General Matrix Multiply, SGEMM) 卷积内核：
/// 输入 `self` 可为任意跨步布局，`w_data` 为连续的 `[C_out, C_in * KH * KW]` 权重矩阵；
/// 每个样本的感受野窗口展平为 `[C_in * KH * KW, H_out * W_out]` 列矩阵后与权重相乘，偏置预先写入输出再以 beta = 1 累加
fn convIm2col(self: *const Tensor, w_data: []const f32, bias: ?*Tensor, g: ConvGeometry, out: *Tensor, allocator: std.mem.Allocator) !void {
    const K_col = g.C_in * g.KH * g.KW;
    const L_out = g.H_out * g.W_out;
    if (L_out == 0 or g.C_out == 0) return;

    // 四维 [N, C, H, W] 跨度；一维输入 [N, C, L] 的高度跨度为 0
    const s_n = self.strides.dims[0];
    const s_c = self.strides.dims[1];
    const s_h = if (self.shape.len == 4) self.strides.dims[2] else 0;
    const s_w = self.strides.dims[self.shape.len - 1];

    const col_buf = try allocator.alloc(f32, K_col * L_out);
    defer allocator.free(col_buf);

    for (0..g.N) |n| {
        for (0..g.C_in) |ci| {
            for (0..g.KH) |kh| {
                for (0..g.KW) |kw| {
                    const k_row = (ci * g.KH + kh) * g.KW + kw;
                    const col_row = col_buf[k_row * L_out .. (k_row + 1) * L_out];
                    const w_range = validRange(g.W_out, g.W, kw, g.stride, g.pad_w);
                    for (0..g.H_out) |h_out| {
                        const dst = col_row[h_out * g.W_out .. (h_out + 1) * g.W_out];
                        const ih_raw = h_out * g.stride + kh;
                        if (ih_raw < g.pad_h or ih_raw - g.pad_h >= g.H) {
                            @memset(dst, 0.0);
                            continue;
                        }
                        const ih = ih_raw - g.pad_h;
                        @memset(dst[0..w_range[0]], 0.0);
                        @memset(dst[w_range[1]..], 0.0);
                        if (w_range[1] > w_range[0]) {
                            // 有效区间内首个输入列下标
                            const iw0 = w_range[0] * g.stride + kw - g.pad_w;
                            const base = n * s_n + ci * s_c + ih * s_h + iw0 * s_w;
                            const valid = dst[w_range[0]..w_range[1]];
                            const step = g.stride * s_w;
                            if (step == 1) {
                                @memcpy(valid, self.data[base..][0..valid.len]);
                            } else {
                                for (valid, 0..) |*v, j| v.* = self.data[base + j * step];
                            }
                        }
                    }
                }
            }
        }

        const out_n = out.data[n * g.C_out * L_out .. (n + 1) * g.C_out * L_out];
        if (bias) |b| {
            const b_stride = b.strides.dims[0];
            for (0..g.C_out) |co| {
                @memset(out_n[co * L_out .. (co + 1) * L_out], b.data[co * b_stride]);
            }
        }
        if (K_col == 0) continue;

        c.cblas_sgemm(
            c.CblasRowMajor,
            c.CblasNoTrans,
            c.CblasNoTrans,
            @intCast(g.C_out),
            @intCast(L_out),
            @intCast(K_col),
            1.0,
            w_data.ptr,
            @intCast(K_col),
            col_buf.ptr,
            @intCast(L_out),
            if (bias != null) @as(f32, 1.0) else @as(f32, 0.0),
            out_n.ptr,
            @intCast(L_out),
        );
    }
}

/// 一维卷积前向计算 (1-Dimensional Convolution, Conv1D)
/// 输入形状: `[N, C_in, L]`，卷积核形状: `[C_out, C_in, K]`，可选偏置形状: `[C_out]`
pub fn conv1d(
    self: *Tensor,
    weight: *Tensor,
    bias: ?*Tensor,
    options: ConvOptions,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 3 or weight.shape.len != 3) {
        return error.IncompatibleDimensions;
    }
    const stride = options.stride;
    const padding = options.padding;
    if (stride == 0) {
        return error.InvalidStride;
    }
    const N = self.shape.dims[0];
    const C_in = self.shape.dims[1];
    const L = self.shape.dims[2];

    const C_out = weight.shape.dims[0];
    if (weight.shape.dims[1] != C_in) return error.ShapeMismatch;
    const K = weight.shape.dims[2];

    if (bias) |b| {
        if (b.shape.len != 1 or b.shape.dims[0] != C_out) return error.ShapeMismatch;
    }

    const L_padded = L + 2 * padding;
    if (L_padded < K) return error.KernelBiggerThanInput;

    const L_out = (L_padded - K) / stride + 1;

    const out = try zeros(allocator, &.{ N, C_out, L_out });
    errdefer out.deinit(allocator);

    const w = if (weight.isContiguous()) weight else try weight.contiguous(allocator);
    defer if (w != weight) w.deinit(allocator);

    try convIm2col(self, w.data, bias, .{
        .N = N,
        .C_in = C_in,
        .H = 1,
        .W = L,
        .C_out = C_out,
        .KH = 1,
        .KW = K,
        .stride = stride,
        .pad_h = 0,
        .pad_w = padding,
        .H_out = 1,
        .W_out = L_out,
    }, out, allocator);
    return out;
}

/// 二维卷积前向计算 (2-Dimensional Convolution, Conv2D)
/// 输入形状: `[N, C_in, H, W]`，卷积核形状: `[C_out, C_in, KH, KW]`，可选偏置形状: `[C_out]`
pub fn conv2d(
    self: *Tensor,
    weight: *Tensor,
    bias: ?*Tensor,
    options: ConvOptions,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 4 or weight.shape.len != 4) {
        return error.IncompatibleDimensions;
    }
    const stride = options.stride;
    const padding = options.padding;
    if (stride == 0) {
        return error.InvalidStride;
    }
    const N = self.shape.dims[0];
    const C_in = self.shape.dims[1];
    const H = self.shape.dims[2];
    const W = self.shape.dims[3];

    const C_out = weight.shape.dims[0];
    if (weight.shape.dims[1] != C_in) return error.ShapeMismatch;
    const KH = weight.shape.dims[2];
    const KW = weight.shape.dims[3];

    if (bias) |b| {
        if (b.shape.len != 1 or b.shape.dims[0] != C_out) return error.ShapeMismatch;
    }

    const H_padded = H + 2 * padding;
    const W_padded = W + 2 * padding;
    if (H_padded < KH or W_padded < KW) return error.KernelBiggerThanInput;

    const H_out = (H_padded - KH) / stride + 1;
    const W_out = (W_padded - KW) / stride + 1;

    const out = try zeros(allocator, &.{ N, C_out, H_out, W_out });
    errdefer out.deinit(allocator);

    const w = if (weight.isContiguous()) weight else try weight.contiguous(allocator);
    defer if (w != weight) w.deinit(allocator);

    try convIm2col(self, w.data, bias, .{
        .N = N,
        .C_in = C_in,
        .H = H,
        .W = W,
        .C_out = C_out,
        .KH = KH,
        .KW = KW,
        .stride = stride,
        .pad_h = padding,
        .pad_w = padding,
        .H_out = H_out,
        .W_out = W_out,
    }, out, allocator);
    return out;
}

/// 转置卷积 SGEMM + 列转图像 (Column to Image, col2im) 内核：
/// 对每个样本先计算列矩阵 `cols[C_out * KH * KW, H * W] = W^T * X_n`（权重视为连续的 `[C_in, C_out * KH * KW]` 矩阵），
/// 再把每一列散射累加回输出平面；输出需预先写入偏置。
fn convTransposeGemm(x_data: []const f32, w_data: []const f32, g: ConvGeometry, out: *Tensor, allocator: std.mem.Allocator) !void {
    const HW_in = g.H * g.W;
    const K_rows = g.C_out * g.KH * g.KW;
    const plane = g.H_out * g.W_out;
    if (HW_in == 0 or K_rows == 0 or plane == 0 or g.C_in == 0) return;

    const cols = try allocator.alloc(f32, K_rows * HW_in);
    defer allocator.free(cols);

    for (0..g.N) |n| {
        c.cblas_sgemm(
            c.CblasRowMajor,
            c.CblasTrans,
            c.CblasNoTrans,
            @intCast(K_rows),
            @intCast(HW_in),
            @intCast(g.C_in),
            1.0,
            w_data.ptr,
            @intCast(K_rows),
            x_data.ptr + n * g.C_in * HW_in,
            @intCast(HW_in),
            0.0,
            cols.ptr,
            @intCast(HW_in),
        );

        const out_n = out.data[n * g.C_out * plane .. (n + 1) * g.C_out * plane];
        for (0..g.C_out) |co| {
            const out_plane = out_n[co * plane .. (co + 1) * plane];
            for (0..g.KH) |kh| {
                const h_range = validRange(g.H, g.H_out, kh, g.stride, g.pad_h);
                for (0..g.KW) |kw| {
                    const w_range = validRange(g.W, g.W_out, kw, g.stride, g.pad_w);
                    if (w_range[1] <= w_range[0]) continue;
                    const col_row = cols[((co * g.KH + kh) * g.KW + kw) * HW_in ..][0..HW_in];
                    for (h_range[0]..h_range[1]) |h| {
                        const out_h = h * g.stride + kh - g.pad_h;
                        const src = col_row[h * g.W + w_range[0] .. h * g.W + w_range[1]];
                        const ow0 = w_range[0] * g.stride + kw - g.pad_w;
                        const dst_base = out_h * g.W_out + ow0;
                        if (g.stride == 1) {
                            for (out_plane[dst_base..][0..src.len], src) |*o, v| o.* += v;
                        } else {
                            for (src, 0..) |v, j| out_plane[dst_base + j * g.stride] += v;
                        }
                    }
                }
            }
        }
    }
}

/// 把 `[N, C_out, ...]` 输出的每个通道平面填充为对应偏置
fn fillBiasPlanes(out: *Tensor, bias: ?*Tensor, N: usize, C_out: usize, plane: usize) void {
    const b = bias orelse return;
    for (0..N) |n| {
        for (0..C_out) |co| {
            const start = (n * C_out + co) * plane;
            @memset(out.data[start .. start + plane], b.data[co * b.strides.dims[0]]);
        }
    }
}

/// 一维转置卷积前向计算 (1-Dimensional Transposed Convolution, ConvTranspose1D)
/// 输入形状: `[N, C_in, L_in]`，权重形状: `[C_in, C_out, K]`，可选偏置形状: `[C_out]`
/// 输出长度: `L_out = (L_in - 1) * stride + K - 2 * padding`
pub fn convTranspose1d(
    self: *Tensor,
    weight: *Tensor,
    bias: ?*Tensor,
    options: ConvOptions,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 3 or weight.shape.len != 3) {
        return error.IncompatibleDimensions;
    }
    const stride = options.stride;
    const padding = options.padding;
    if (stride == 0) {
        return error.InvalidStride;
    }
    const N = self.shape.dims[0];
    const C_in = self.shape.dims[1];
    const L_in = self.shape.dims[2];
    if (L_in == 0) return error.InvalidDimension;

    if (weight.shape.dims[0] != C_in) return error.ShapeMismatch;
    const C_out = weight.shape.dims[1];
    const K = weight.shape.dims[2];

    if (bias) |b| {
        if (b.shape.len != 1 or b.shape.dims[0] != C_out) return error.ShapeMismatch;
    }

    const raw_len = (L_in - 1) * stride + K;
    if (raw_len <= 2 * padding) return error.KernelBiggerThanInput;
    const L_out = raw_len - 2 * padding;

    const out = try zeros(allocator, &.{ N, C_out, L_out });
    errdefer out.deinit(allocator);
    fillBiasPlanes(out, bias, N, C_out, L_out);

    const x = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (x != self) x.deinit(allocator);
    const w = if (weight.isContiguous()) weight else try weight.contiguous(allocator);
    defer if (w != weight) w.deinit(allocator);

    try convTransposeGemm(x.data, w.data, .{
        .N = N,
        .C_in = C_in,
        .H = 1,
        .W = L_in,
        .C_out = C_out,
        .KH = 1,
        .KW = K,
        .stride = stride,
        .pad_h = 0,
        .pad_w = padding,
        .H_out = 1,
        .W_out = L_out,
    }, out, allocator);
    return out;
}

/// 二维转置卷积前向计算 (2-Dimensional Transposed Convolution, ConvTranspose2D)
/// 输入形状: `[N, C_in, H_in, W_in]`，权重形状: `[C_in, C_out, KH, KW]`，可选偏置形状: `[C_out]`
pub fn convTranspose2d(
    self: *Tensor,
    weight: *Tensor,
    bias: ?*Tensor,
    options: ConvOptions,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 4 or weight.shape.len != 4) {
        return error.IncompatibleDimensions;
    }
    const stride = options.stride;
    const padding = options.padding;
    if (stride == 0) {
        return error.InvalidStride;
    }
    const N = self.shape.dims[0];
    const C_in = self.shape.dims[1];
    const H_in = self.shape.dims[2];
    const W_in = self.shape.dims[3];
    if (H_in == 0 or W_in == 0) return error.InvalidDimension;

    if (weight.shape.dims[0] != C_in) return error.ShapeMismatch;
    const C_out = weight.shape.dims[1];
    const KH = weight.shape.dims[2];
    const KW = weight.shape.dims[3];

    if (bias) |b| {
        if (b.shape.len != 1 or b.shape.dims[0] != C_out) return error.ShapeMismatch;
    }

    const raw_h = (H_in - 1) * stride + KH;
    const raw_w = (W_in - 1) * stride + KW;
    if (raw_h <= 2 * padding or raw_w <= 2 * padding) return error.KernelBiggerThanInput;

    const H_out = raw_h - 2 * padding;
    const W_out = raw_w - 2 * padding;

    const out = try zeros(allocator, &.{ N, C_out, H_out, W_out });
    errdefer out.deinit(allocator);
    fillBiasPlanes(out, bias, N, C_out, H_out * W_out);

    const x = if (self.isContiguous()) self else try self.contiguous(allocator);
    defer if (x != self) x.deinit(allocator);
    const w = if (weight.isContiguous()) weight else try weight.contiguous(allocator);
    defer if (w != weight) w.deinit(allocator);

    try convTransposeGemm(x.data, w.data, .{
        .N = N,
        .C_in = C_in,
        .H = H_in,
        .W = W_in,
        .C_out = C_out,
        .KH = KH,
        .KW = KW,
        .stride = stride,
        .pad_h = padding,
        .pad_w = padding,
        .H_out = H_out,
        .W_out = W_out,
    }, out, allocator);
    return out;
}

/// 一维最大池化前向计算 (1-Dimensional Max Pooling, MaxPool1D)
/// 输入形状: `[N, C, L]`，输出长度: `L_out = (L + 2 * padding - pool_size) / stride + 1`
pub fn maxpool1d(
    self: *Tensor,
    pool_size: usize,
    options: PoolOptions,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 3) return error.IncompatibleDimensions;
    const stride = options.resolveStride(pool_size);
    const padding = options.padding;
    if (stride == 0 or pool_size == 0) return error.InvalidStride;

    const N = self.shape.dims[0];
    const C = self.shape.dims[1];
    const L = self.shape.dims[2];

    const L_padded = L + 2 * padding;
    if (L_padded < pool_size) return error.KernelBiggerThanInput;

    const L_out = (L_padded - pool_size) / stride + 1;
    const out = try zeros(allocator, &.{ N, C, L_out });
    errdefer out.deinit(allocator);

    const s_n = self.strides.dims[0];
    const s_c = self.strides.dims[1];
    const s_l = self.strides.dims[2];

    const o_n = out.strides.dims[0];
    const o_c = out.strides.dims[1];
    const o_l = out.strides.dims[2];

    for (0..N) |n| {
        for (0..C) |c_| {
            for (0..L_out) |ol| {
                var max_val: f32 = -std.math.inf(f32);
                var found = false;
                for (0..pool_size) |pl| {
                    const il_signed: isize = @as(isize, @intCast(ol * stride + pl)) - @as(isize, @intCast(padding));
                    if (il_signed >= 0 and il_signed < @as(isize, @intCast(L))) {
                        const il: usize = @intCast(il_signed);
                        const val = self.data[n * s_n + c_ * s_c + il * s_l];
                        if (!found or val > max_val) {
                            max_val = val;
                            found = true;
                        }
                    }
                }
                out.data[n * o_n + c_ * o_c + ol * o_l] = if (found) max_val else 0.0;
            }
        }
    }
    return out;
}

/// 二维最大池化前向计算 (2-Dimensional Max Pooling, MaxPool2D)
/// 输入形状: `[N, C, H, W]`，输出高宽: `H_out = (H + 2 * padding - pool_size) / stride + 1`
pub fn maxpool2d(
    self: *Tensor,
    pool_size: usize,
    options: PoolOptions,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 4) return error.IncompatibleDimensions;
    const stride = options.resolveStride(pool_size);
    const padding = options.padding;
    if (stride == 0 or pool_size == 0) return error.InvalidStride;

    const N = self.shape.dims[0];
    const C = self.shape.dims[1];
    const H = self.shape.dims[2];
    const W = self.shape.dims[3];

    const H_padded = H + 2 * padding;
    const W_padded = W + 2 * padding;
    if (H_padded < pool_size or W_padded < pool_size) return error.KernelBiggerThanInput;

    const H_out = (H_padded - pool_size) / stride + 1;
    const W_out = (W_padded - pool_size) / stride + 1;

    const out = try zeros(allocator, &.{ N, C, H_out, W_out });
    errdefer out.deinit(allocator);

    const s_n = self.strides.dims[0];
    const s_c = self.strides.dims[1];
    const s_h = self.strides.dims[2];
    const s_w = self.strides.dims[3];

    const o_n = out.strides.dims[0];
    const o_c = out.strides.dims[1];
    const o_h = out.strides.dims[2];
    const o_w = out.strides.dims[3];

    for (0..N) |n| {
        for (0..C) |c_| {
            for (0..H_out) |h| {
                for (0..W_out) |w| {
                    var max_val: f32 = -std.math.inf(f32);
                    var found = false;
                    for (0..pool_size) |ph| {
                        const ih_signed: isize = @as(isize, @intCast(h * stride + ph)) - @as(isize, @intCast(padding));
                        if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) continue;
                        const ih: usize = @intCast(ih_signed);
                        for (0..pool_size) |pw| {
                            const iw_signed: isize = @as(isize, @intCast(w * stride + pw)) - @as(isize, @intCast(padding));
                            if (iw_signed < 0 or iw_signed >= @as(isize, @intCast(W))) continue;
                            const iw: usize = @intCast(iw_signed);
                            const val = self.data[n * s_n + c_ * s_c + ih * s_h + iw * s_w];
                            if (!found or val > max_val) {
                                max_val = val;
                                found = true;
                            }
                        }
                    }
                    out.data[n * o_n + c_ * o_c + h * o_h + w * o_w] = if (found) max_val else 0.0;
                }
            }
        }
    }
    return out;
}

/// 一维平均池化前向计算 (1-Dimensional Average Pooling, AvgPool1D)
/// 输入形状: `[N, C, L]`，输出长度: `L_out = (L + 2 * padding - kernel_size) / stride + 1`
pub fn avgpool1d(
    self: *Tensor,
    kernel_size: usize,
    options: PoolOptions,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 3) return error.IncompatibleDimensions;
    const stride = options.resolveStride(kernel_size);
    const padding = options.padding;
    if (stride == 0 or kernel_size == 0) return error.InvalidStride;

    const N = self.shape.dims[0];
    const C = self.shape.dims[1];
    const L = self.shape.dims[2];

    const L_padded = L + 2 * padding;
    if (L_padded < kernel_size) return error.KernelBiggerThanInput;

    const out_l = (L_padded - kernel_size) / stride + 1;
    const out = try zeros(allocator, &.{ N, C, out_l });
    errdefer out.deinit(allocator);
    const pool_len = @as(f32, @floatFromInt(kernel_size));

    const s_n = self.strides.dims[0];
    const s_c = self.strides.dims[1];
    const s_l = self.strides.dims[2];

    const o_n = out.strides.dims[0];
    const o_c = out.strides.dims[1];
    const o_l = out.strides.dims[2];

    for (0..N) |n| {
        for (0..C) |c_| {
            for (0..out_l) |ol| {
                var sum_val: f32 = 0.0;
                for (0..kernel_size) |kl| {
                    const il_signed: isize = @as(isize, @intCast(ol * stride + kl)) - @as(isize, @intCast(padding));
                    if (il_signed >= 0 and il_signed < @as(isize, @intCast(L))) {
                        const il: usize = @intCast(il_signed);
                        sum_val += self.data[n * s_n + c_ * s_c + il * s_l];
                    }
                }
                out.data[n * o_n + c_ * o_c + ol * o_l] = sum_val / pool_len;
            }
        }
    }
    return out;
}

/// 二维平均池化前向计算 (2-Dimensional Average Pooling, AvgPool2D)
/// 输入形状: `[N, C, H, W]`，输出高宽: `H_out = (H + 2 * padding - kernel_size) / stride + 1`
pub fn avgpool2d(
    self: *Tensor,
    kernel_size: usize,
    options: PoolOptions,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 4) return error.IncompatibleDimensions;
    const stride = options.resolveStride(kernel_size);
    const padding = options.padding;
    if (stride == 0 or kernel_size == 0) return error.InvalidStride;

    const N = self.shape.dims[0];
    const C = self.shape.dims[1];
    const H = self.shape.dims[2];
    const W = self.shape.dims[3];

    const H_padded = H + 2 * padding;
    const W_padded = W + 2 * padding;
    if (H_padded < kernel_size or W_padded < kernel_size) return error.KernelBiggerThanInput;

    const out_h = (H_padded - kernel_size) / stride + 1;
    const out_w = (W_padded - kernel_size) / stride + 1;
    const out = try zeros(allocator, &.{ N, C, out_h, out_w });
    errdefer out.deinit(allocator);
    const pool_area = @as(f32, @floatFromInt(kernel_size * kernel_size));

    const s_n = self.strides.dims[0];
    const s_c = self.strides.dims[1];
    const s_h = self.strides.dims[2];
    const s_w = self.strides.dims[3];

    const o_n = out.strides.dims[0];
    const o_c = out.strides.dims[1];
    const o_h = out.strides.dims[2];
    const o_w = out.strides.dims[3];

    for (0..N) |n| {
        for (0..C) |c_| {
            for (0..out_h) |oh| {
                for (0..out_w) |ow| {
                    var sum_val: f32 = 0.0;
                    for (0..kernel_size) |kh| {
                        const ih_signed: isize = @as(isize, @intCast(oh * stride + kh)) - @as(isize, @intCast(padding));
                        if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) continue;
                        const ih: usize = @intCast(ih_signed);
                        for (0..kernel_size) |kw| {
                            const iw_signed: isize = @as(isize, @intCast(ow * stride + kw)) - @as(isize, @intCast(padding));
                            if (iw_signed < 0 or iw_signed >= @as(isize, @intCast(W))) continue;
                            const iw: usize = @intCast(iw_signed);
                            sum_val += self.data[n * s_n + c_ * s_c + ih * s_h + iw * s_w];
                        }
                    }
                    out.data[n * o_n + c_ * o_c + oh * o_h + ow * o_w] = sum_val / pool_area;
                }
            }
        }
    }
    return out;
}

/// 一维自适应平均池化前向计算 (1-Dimensional Adaptive Average Pooling, AdaptiveAvgPool1D)
/// 输入形状: `[N, C, L]`，输出形状: `[N, C, output_size]`
/// 第 `i` 个输出位置对应区间 `[floor(i * L / output_size), ceil((i + 1) * L / output_size))`
pub fn adaptiveAvgPool1d(
    self: *Tensor,
    output_size: usize,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 3) return error.IncompatibleDimensions;
    if (output_size == 0) return error.InvalidDimension;

    const N = self.shape.dims[0];
    const C = self.shape.dims[1];
    const L = self.shape.dims[2];
    if (L == 0) return error.InvalidDimension;

    const out = try zeros(allocator, &.{ N, C, output_size });
    errdefer out.deinit(allocator);

    const s_n = self.strides.dims[0];
    const s_c = self.strides.dims[1];
    const s_l = self.strides.dims[2];

    const o_n = out.strides.dims[0];
    const o_c = out.strides.dims[1];
    const o_l = out.strides.dims[2];

    for (0..N) |n| {
        for (0..C) |c_| {
            for (0..output_size) |ol| {
                const l_start = (ol * L) / output_size;
                const l_end = ((ol + 1) * L + output_size - 1) / output_size;
                const count = l_end - l_start;
                var sum_val: f32 = 0.0;
                for (l_start..l_end) |il| {
                    sum_val += self.data[n * s_n + c_ * s_c + il * s_l];
                }
                out.data[n * o_n + c_ * o_c + ol * o_l] = sum_val / @as(f32, @floatFromInt(count));
            }
        }
    }
    return out;
}

/// 二维自适应平均池化前向计算 (2-Dimensional Adaptive Average Pooling, AdaptiveAvgPool2D)
/// 输入形状: `[N, C, H, W]`，输出形状: `[N, C, output_size[0], output_size[1]]`
pub fn adaptiveAvgPool2d(
    self: *Tensor,
    output_size: [2]usize,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 4) return error.IncompatibleDimensions;
    const out_h = output_size[0];
    const out_w = output_size[1];
    if (out_h == 0 or out_w == 0) return error.InvalidDimension;

    const N = self.shape.dims[0];
    const C = self.shape.dims[1];
    const H = self.shape.dims[2];
    const W = self.shape.dims[3];
    if (H == 0 or W == 0) return error.InvalidDimension;

    const out = try zeros(allocator, &.{ N, C, out_h, out_w });
    errdefer out.deinit(allocator);

    const s_n = self.strides.dims[0];
    const s_c = self.strides.dims[1];
    const s_h = self.strides.dims[2];
    const s_w = self.strides.dims[3];

    const o_n = out.strides.dims[0];
    const o_c = out.strides.dims[1];
    const o_h = out.strides.dims[2];
    const o_w = out.strides.dims[3];

    for (0..N) |n| {
        for (0..C) |c_| {
            for (0..out_h) |oh| {
                const h_start = (oh * H) / out_h;
                const h_end = ((oh + 1) * H + out_h - 1) / out_h;
                for (0..out_w) |ow| {
                    const w_start = (ow * W) / out_w;
                    const w_end = ((ow + 1) * W + out_w - 1) / out_w;
                    const area = (h_end - h_start) * (w_end - w_start);
                    var sum_val: f32 = 0.0;
                    for (h_start..h_end) |ih| {
                        for (w_start..w_end) |iw| {
                            sum_val += self.data[n * s_n + c_ * s_c + ih * s_h + iw * s_w];
                        }
                    }
                    out.data[n * o_n + c_ * o_c + oh * o_h + ow * o_w] = sum_val / @as(f32, @floatFromInt(area));
                }
            }
        }
    }
    return out;
}
