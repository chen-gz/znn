const std = @import("std");
const c = @import("../cblas.zig");
const core = @import("core.zig");
const Tensor = core.Tensor;
const ops_mod = @import("ops.zig");
const zeros = ops_mod.zeros;
const free = ops_mod.free;

pub fn conv2d(self: *Tensor, weight: *Tensor, bias: ?*Tensor, allocator: std.mem.Allocator) !*Tensor {
    return self.conv2dWithConfig(weight, bias, 1, 0, allocator);
}

pub fn conv2dWithConfig(
    self: *Tensor,
    weight: *Tensor,
    bias: ?*Tensor,
    stride: usize,
    padding: usize,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 4 or weight.shape.len != 4) {
        return error.IncompatibleDimensions;
    }
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
            // im2col: 展平当前样本的所有感受野窗口为 [K_col, L_out] 矩阵
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

pub fn convTranspose2d(
    self: *Tensor,
    weight: *Tensor,
    bias: ?*Tensor,
    stride: usize,
    padding: usize,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (self.shape.len != 4 or weight.shape.len != 4) {
        return error.IncompatibleDimensions;
    }
    const N = self.shape.dims[0];
    const C_in = self.shape.dims[1];
    const H_in = self.shape.dims[2];
    const W_in = self.shape.dims[3];

    if (weight.shape.dims[0] != C_in) return error.ShapeMismatch;
    const C_out = weight.shape.dims[1];
    const KH = weight.shape.dims[2];
    const KW = weight.shape.dims[3];

    if (bias) |b| {
        if (b.shape.len != 1 or b.shape.dims[0] != C_out) return error.ShapeMismatch;
    }

    const H_out = (H_in - 1) * stride + KH - 2 * padding;
    const W_out = (W_in - 1) * stride + KW - 2 * padding;

    const out = try zeros(allocator, &.{ N, C_out, H_out, W_out });

    for (0..N) |n| {
        for (0..C_out) |co| {
            const b_val = if (bias) |b| b.data[co] else 0.0;
            for (0..H_out) |h| {
                for (0..W_out) |w| {
                    out.data[n * (C_out * H_out * W_out) + co * (H_out * W_out) + h * W_out + w] = b_val;
                }
            }
        }
    }

    for (0..N) |n| {
        for (0..C_in) |ci| {
            for (0..H_in) |h| {
                for (0..W_in) |w| {
                    const input_val = self.data[n * (C_in * H_in * W_in) + ci * (H_in * W_in) + h * W_in + w];
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

                                const weight_val = weight.data[ci * (C_out * KH * KW) + co * (KH * KW) + kh * KW + kw];
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

pub fn maxpool2d(self: *Tensor, pool_size: usize, stride: usize, allocator: std.mem.Allocator) !*Tensor {
    if (self.shape.len != 4) return error.IncompatibleDimensions;
    if (stride == 0 or pool_size == 0) return error.InvalidStride;
    const N = self.shape.dims[0];
    const C = self.shape.dims[1];
    const H = self.shape.dims[2];
    const W = self.shape.dims[3];

    const H_out = H / stride;
    const W_out = W / stride;

    const out = try zeros(allocator, &.{ N, C, H_out, W_out });

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
                    var max_val = self.data[n * s_n + c_ * s_c + (h * stride) * s_h + (w * stride) * s_w];
                    for (0..pool_size) |ph| {
                        for (0..pool_size) |pw| {
                            const ih = h * stride + ph;
                            const iw = w * stride + pw;
                            if (ih < H and iw < W) {
                                const val = self.data[n * s_n + c_ * s_c + ih * s_h + iw * s_w];
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    out.data[n * o_n + c_ * o_c + h * o_h + w * o_w] = max_val;
                }
            }
        }
    }
    return out;
}

pub fn avgpool2d(self: *Tensor, kernel_size: usize, stride: usize, allocator: std.mem.Allocator) !*Tensor {
    if (self.shape.len != 4) return error.IncompatibleDimensions;
    if (stride == 0 or kernel_size == 0) return error.InvalidStride;
    const N = self.shape.dims[0];
    const C = self.shape.dims[1];
    const H = self.shape.dims[2];
    const W = self.shape.dims[3];
    if (kernel_size > H or kernel_size > W) return error.KernelBiggerThanInput;

    const out_h = (H - kernel_size) / stride + 1;
    const out_w = (W - kernel_size) / stride + 1;
    const out = try zeros(allocator, &.{ N, C, out_h, out_w });
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
                    const ih_start = oh * stride;
                    const iw_start = ow * stride;
                    var sum_val: f32 = 0.0;

                    for (0..kernel_size) |kh| {
                        for (0..kernel_size) |kw| {
                            const ih = ih_start + kh;
                            const iw = iw_start + kw;
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

pub fn rope(self: *Tensor, start_pos: usize, allocator: std.mem.Allocator) !*Tensor {
    return self.ropeOffset(start_pos, 0, allocator);
}

pub fn ropeOffset(self: *Tensor, start_pos: usize, rotary_offset: usize, allocator: std.mem.Allocator) !*Tensor {
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
                const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / rot_dim_f);
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
    const V = @TypeOf(val);
    if (@typeInfo(V) == .float) {
        if (val < 0.0) return error.IndexOutOfBounds;
        const idx = @as(usize, @intFromFloat(val));
        if (idx >= vocab_size) return error.IndexOutOfBounds;
        return idx;
    } else if (@typeInfo(V) == .int or @typeInfo(V) == .comptime_int) {
        if (val < 0) return error.IndexOutOfBounds;
        const idx = @as(usize, @intCast(val));
        if (idx >= vocab_size) return error.IndexOutOfBounds;
        return idx;
    } else {
        @compileError("Unsupported embedding index scalar type: " ++ @typeName(V));
    }
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
