const std = @import("std");
const c = @import("../cblas.zig");
const core = @import("core.zig");
const types = @import("types.zig");
const Tensor = core.Tensor;
const ConvOptions = types.ConvOptions;
const PoolOptions = types.PoolOptions;
const ops_mod = @import("ops.zig");
const zeros = ops_mod.zeros;

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

    const s_n = self.strides.dims[0];
    const s_c = self.strides.dims[1];
    const s_l = self.strides.dims[2];

    const K_col = C_in * K;

    if (weight.isContiguous() and K_col > 0 and L_out > 0 and C_out > 0) {
        const col_buf = try allocator.alloc(f32, K_col * L_out);
        defer allocator.free(col_buf);

        for (0..N) |n| {
            // 图像转列 (Image to Column, im2col): 展平当前样本的一维感受野窗口为 [K_col, L_out] 矩阵
            for (0..C_in) |ci| {
                for (0..K) |k| {
                    const k_row = ci * K + k;
                    const col_row = col_buf[k_row * L_out .. (k_row + 1) * L_out];
                    for (0..L_out) |l_out| {
                        const il_signed: isize = @as(isize, @intCast(l_out * stride + k)) - @as(isize, @intCast(padding));
                        if (il_signed >= 0 and il_signed < @as(isize, @intCast(L))) {
                            const il: usize = @intCast(il_signed);
                            col_row[l_out] = self.data[n * s_n + ci * s_c + il * s_l];
                        } else {
                            col_row[l_out] = 0.0;
                        }
                    }
                }
            }

            const out_n = out.data[n * C_out * L_out .. (n + 1) * C_out * L_out];
            if (bias) |b| {
                const b_stride = b.strides.dims[0];
                for (0..C_out) |co| {
                    @memset(out_n[co * L_out .. (co + 1) * L_out], b.data[co * b_stride]);
                }
            }

            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasNoTrans,
                @intCast(C_out),
                @intCast(L_out),
                @intCast(K_col),
                1.0,
                weight.data.ptr,
                @intCast(K_col),
                col_buf.ptr,
                @intCast(L_out),
                if (bias != null) @as(f32, 1.0) else @as(f32, 0.0),
                out_n.ptr,
                @intCast(L_out),
            );
        }
        return out;
    }

    const w_co = weight.strides.dims[0];
    const w_ci = weight.strides.dims[1];
    const w_k = weight.strides.dims[2];

    const o_n = out.strides.dims[0];
    const o_c = out.strides.dims[1];
    const o_l = out.strides.dims[2];

    for (0..N) |n| {
        for (0..C_out) |co| {
            const b_val = if (bias) |b| b.data[co * b.strides.dims[0]] else 0.0;
            for (0..L_out) |l_out| {
                var acc: f32 = b_val;
                for (0..C_in) |ci| {
                    for (0..K) |k| {
                        const il_signed: isize = @as(isize, @intCast(l_out * stride + k)) - @as(isize, @intCast(padding));
                        if (il_signed < 0 or il_signed >= @as(isize, @intCast(L))) continue;
                        const il: usize = @intCast(il_signed);
                        const input_val = self.data[n * s_n + ci * s_c + il * s_l];
                        const weight_val = weight.data[co * w_co + ci * w_ci + k * w_k];
                        acc += input_val * weight_val;
                    }
                }
                out.data[n * o_n + co * o_c + l_out * o_l] = acc;
            }
        }
    }
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

    const s_n = self.strides.dims[0];
    const s_c = self.strides.dims[1];
    const s_h = self.strides.dims[2];
    const s_w = self.strides.dims[3];

    const K_col = C_in * KH * KW;
    const L_out = H_out * W_out;

    if (weight.isContiguous() and K_col > 0 and L_out > 0 and C_out > 0) {
        const col_buf = try allocator.alloc(f32, K_col * L_out);
        defer allocator.free(col_buf);

        for (0..N) |n| {
            // 图像转列 (Image to Column, im2col): 展平当前样本的所有感受野窗口为 [K_col, L_out] 矩阵
            for (0..C_in) |ci| {
                for (0..KH) |kh| {
                    for (0..KW) |kw| {
                        const k_row = (ci * KH + kh) * KW + kw;
                        const col_row = col_buf[k_row * L_out .. (k_row + 1) * L_out];
                        for (0..H_out) |h_out| {
                            const ih_signed: isize = @as(isize, @intCast(h_out * stride + kh)) - @as(isize, @intCast(padding));
                            if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) {
                                @memset(col_row[h_out * W_out .. (h_out + 1) * W_out], 0.0);
                                continue;
                            }
                            const ih: usize = @intCast(ih_signed);
                            for (0..W_out) |w_out| {
                                const iw_signed: isize = @as(isize, @intCast(w_out * stride + kw)) - @as(isize, @intCast(padding));
                                if (iw_signed >= 0 and iw_signed < @as(isize, @intCast(W))) {
                                    const iw: usize = @intCast(iw_signed);
                                    col_row[h_out * W_out + w_out] = self.data[n * s_n + ci * s_c + ih * s_h + iw * s_w];
                                } else {
                                    col_row[h_out * W_out + w_out] = 0.0;
                                }
                            }
                        }
                    }
                }
            }

            const out_n = out.data[n * C_out * L_out .. (n + 1) * C_out * L_out];
            if (bias) |b| {
                const b_stride = b.strides.dims[0];
                for (0..C_out) |co| {
                    @memset(out_n[co * L_out .. (co + 1) * L_out], b.data[co * b_stride]);
                }
            }

            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasNoTrans,
                @intCast(C_out),
                @intCast(L_out),
                @intCast(K_col),
                1.0,
                weight.data.ptr,
                @intCast(K_col),
                col_buf.ptr,
                @intCast(L_out),
                if (bias != null) @as(f32, 1.0) else @as(f32, 0.0),
                out_n.ptr,
                @intCast(L_out),
            );
        }
        return out;
    }

    const w_co = weight.strides.dims[0];
    const w_ci = weight.strides.dims[1];
    const w_kh = weight.strides.dims[2];
    const w_kw = weight.strides.dims[3];

    const o_n = out.strides.dims[0];
    const o_c = out.strides.dims[1];
    const o_h = out.strides.dims[2];
    const o_w = out.strides.dims[3];

    for (0..N) |n| {
        for (0..C_out) |co| {
            const b_val = if (bias) |b| b.data[co * b.strides.dims[0]] else 0.0;
            for (0..H_out) |h_out| {
                for (0..W_out) |w_out| {
                    var acc: f32 = b_val;
                    for (0..C_in) |ci| {
                        for (0..KH) |kh| {
                            const ih_signed: isize = @as(isize, @intCast(h_out * stride + kh)) - @as(isize, @intCast(padding));
                            if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) continue;
                            const ih: usize = @intCast(ih_signed);
                            for (0..KW) |kw| {
                                const iw_signed: isize = @as(isize, @intCast(w_out * stride + kw)) - @as(isize, @intCast(padding));
                                if (iw_signed < 0 or iw_signed >= @as(isize, @intCast(W))) continue;
                                const iw: usize = @intCast(iw_signed);
                                const input_val = self.data[n * s_n + ci * s_c + ih * s_h + iw * s_w];
                                const weight_val = weight.data[co * w_co + ci * w_ci + kh * w_kh + kw * w_kw];
                                acc += input_val * weight_val;
                            }
                        }
                    }
                    out.data[n * o_n + co * o_c + h_out * o_h + w_out * o_w] = acc;
                }
            }
        }
    }
    return out;
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

    const s_n = self.strides.dims[0];
    const s_c = self.strides.dims[1];
    const s_l = self.strides.dims[2];

    const w_ci = weight.strides.dims[0];
    const w_co = weight.strides.dims[1];
    const w_k = weight.strides.dims[2];

    for (0..N) |n| {
        for (0..C_out) |co| {
            const b_val = if (bias) |b| b.data[co * b.strides.dims[0]] else 0.0;
            if (b_val != 0.0) {
                @memset(out.data[(n * C_out + co) * L_out .. (n * C_out + co + 1) * L_out], b_val);
            }
        }
    }

    for (0..N) |n| {
        for (0..C_in) |ci| {
            for (0..L_in) |l_in| {
                const input_val = self.data[n * s_n + ci * s_c + l_in * s_l];
                if (input_val == 0.0) continue;

                for (0..C_out) |co| {
                    for (0..K) |k| {
                        const out_l_raw = l_in * stride + k;
                        if (out_l_raw < padding) continue;
                        const out_l = out_l_raw - padding;
                        if (out_l >= L_out) continue;

                        const weight_val = weight.data[ci * w_ci + co * w_co + k * w_k];
                        out.data[(n * C_out + co) * L_out + out_l] += input_val * weight_val;
                    }
                }
            }
        }
    }

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

    const s_n = self.strides.dims[0];
    const s_c = self.strides.dims[1];
    const s_h = self.strides.dims[2];
    const s_w = self.strides.dims[3];

    const w_ci = weight.strides.dims[0];
    const w_co = weight.strides.dims[1];
    const w_kh = weight.strides.dims[2];
    const w_kw = weight.strides.dims[3];

    for (0..N) |n| {
        for (0..C_out) |co| {
            const b_val = if (bias) |b| b.data[co * b.strides.dims[0]] else 0.0;
            if (b_val != 0.0) {
                const plane_start = (n * C_out + co) * (H_out * W_out);
                @memset(out.data[plane_start .. plane_start + H_out * W_out], b_val);
            }
        }
    }

    for (0..N) |n| {
        for (0..C_in) |ci| {
            for (0..H_in) |h| {
                for (0..W_in) |w| {
                    const input_val = self.data[n * s_n + ci * s_c + h * s_h + w * s_w];
                    if (input_val == 0.0) continue;

                    for (0..C_out) |co| {
                        for (0..KH) |kh| {
                            const out_h_raw = h * stride + kh;
                            if (out_h_raw < padding) continue;
                            const out_h = out_h_raw - padding;
                            if (out_h >= H_out) continue;

                            for (0..KW) |kw| {
                                const out_w_raw = w * stride + kw;
                                if (out_w_raw < padding) continue;
                                const out_w = out_w_raw - padding;
                                if (out_w >= W_out) continue;

                                const weight_val = weight.data[ci * w_ci + co * w_co + kh * w_kh + kw * w_kw];
                                out.data[n * (C_out * H_out * W_out) + co * (H_out * W_out) + out_h * W_out + out_w] += input_val * weight_val;
                            }
                        }
                    }
                }
            }
        }
    }

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

