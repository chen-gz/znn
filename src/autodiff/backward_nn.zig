const std = @import("std");
const backend = @import("../backend.zig");
const c = backend.cblas;
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const Op = @import("types.zig").Op;

/// 4-bit 量化矩阵乘法 (4-bit Quantized Matrix Multiplication) 反向时每次反量化的权重行数
const q4_backward_row_tile: usize = 64;

/// 一维 / 二维转置卷积 (Transposed Convolution, ConvTranspose) 反向传播 (一维视为高度为 1 的二维)。
/// 前向关系 `Y[n, co, h * s + kh - p, w * s + kw - p] += X[n, ci, h, w] * W[ci, co, kh, kw]`，
/// 把上游梯度按输入位置展开为列矩阵 `col[(co, kh, kw), (h, w)] = dY[n, co, h * s + kh - p, w * s + kw - p]` (越界为 0) 后：
///   dX_n[C_in, H*W]            += W[C_in, C_out*KH*KW] * col
///   dW[C_in, C_out*KH*KW]      += X_n[C_in, H*W] * col^T
/// 两者都交给单精度通用矩阵乘法 (Single-precision General Matrix Multiply, SGEMM)。输入由 `Graph.contiguous` 保证连续。
fn convTransposeBackward(self: *Op, stride: usize, padding: usize) !void {
    const A = self.inputs[0];
    const W = self.inputs[1];
    const C = self.outputs[0];
    const bias = if (self.inputs.len > 2) self.inputs[2] else null;
    const is_2d = A.shape.len == 4;

    const N = A.shape.dims[0];
    const C_in = A.shape.dims[1];
    const H_in = if (is_2d) A.shape.dims[2] else 1;
    const W_in = A.shape.dims[A.shape.len - 1];
    const C_out = W.shape.dims[1];
    const KH = if (is_2d) W.shape.dims[2] else 1;
    const KW = W.shape.dims[W.shape.len - 1];
    const H_out = if (is_2d) C.shape.dims[2] else 1;
    const W_out = C.shape.dims[C.shape.len - 1];
    const out_plane = H_out * W_out;
    // 一维时高度方向既无步长也无填充
    const stride_h = if (is_2d) stride else 1;
    const pad_h = if (is_2d) padding else 0;

    if (bias) |b| {
        if (b.requires_grad) {
            for (0..N) |n| {
                for (0..C_out) |co| {
                    const plane = C.grad[(n * C_out + co) * out_plane ..][0..out_plane];
                    var acc: f32 = 0.0;
                    for (plane) |g| acc += g;
                    b.grad[co] += acc;
                }
            }
        }
    }
    if (!A.requires_grad and !W.requires_grad) return;

    const in_plane = H_in * W_in;
    const K_col = C_out * KH * KW;
    if (in_plane == 0 or K_col == 0 or C_in == 0) return;
    const col = try std.heap.c_allocator.alloc(f32, K_col * in_plane);
    defer std.heap.c_allocator.free(col);

    for (0..N) |n| {
        const dY_n = C.grad[n * C_out * out_plane ..][0 .. C_out * out_plane];
        for (0..C_out) |co| {
            for (0..KH) |kh| {
                for (0..KW) |kw| {
                    const col_row = col[((co * KH + kh) * KW + kw) * in_plane ..][0..in_plane];
                    for (0..H_in) |h| {
                        const dst = col_row[h * W_in ..][0..W_in];
                        const oh_raw = h * stride_h + kh;
                        if (oh_raw < pad_h or oh_raw - pad_h >= H_out) {
                            @memset(dst, 0.0);
                            continue;
                        }
                        const src_row = dY_n[(co * H_out + (oh_raw - pad_h)) * W_out ..][0..W_out];
                        for (dst, 0..) |*d, w| {
                            const ow_raw = w * stride + kw;
                            d.* = if (ow_raw >= padding and ow_raw - padding < W_out) src_row[ow_raw - padding] else 0.0;
                        }
                    }
                }
            }
        }

        if (A.requires_grad) {
            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasNoTrans,
                @intCast(C_in),
                @intCast(in_plane),
                @intCast(K_col),
                1.0,
                W.data.ptr,
                @intCast(K_col),
                col.ptr,
                @intCast(in_plane),
                1.0,
                A.grad.ptr + n * C_in * in_plane,
                @intCast(in_plane),
            );
        }
        if (W.requires_grad) {
            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasTrans,
                @intCast(C_in),
                @intCast(K_col),
                @intCast(in_plane),
                1.0,
                A.data.ptr + n * C_in * in_plane,
                @intCast(in_plane),
                col.ptr,
                @intCast(in_plane),
                1.0,
                W.grad.ptr,
                @intCast(K_col),
            );
        }
    }
}

/// 卷积反向传播中图像转列 (Image to Column, im2col) 列缓冲区的元素上限 (4M 个 f32，即 16 MiB)：
/// 按样本分块时每块的列矩阵不超过该大小 (至少容纳一个样本)
const conv_col_budget: usize = 1 << 22;

/// 卷积反向传播的几何参数 (一维卷积取 H = KH = H_out = 1、stride_h = 1、pad_h = 0)
const ConvGeometry = struct {
    N: usize,
    C_in: usize,
    H: usize,
    W: usize,
    C_out: usize,
    KH: usize,
    KW: usize,
    H_out: usize,
    W_out: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
};

/// 卷积核位置 (kh, kw) 与输出坐标对应的输入坐标；落在填充区域时返回 null
inline fn convInputIndex(out_idx: usize, stride: usize, k: usize, pad: usize, size: usize) ?usize {
    const raw = out_idx * stride + k;
    if (raw < pad or raw - pad >= size) return null;
    return raw - pad;
}

/// 把单个样本 x[C_in, H, W] 展开到列矩阵 col 的第 [col_off, col_off + L_out) 列 (列矩阵行跨度为 ld)
fn im2colSample(col: []f32, ld: usize, col_off: usize, x: []const f32, g: ConvGeometry) void {
    const in_plane = g.H * g.W;
    for (0..g.C_in) |ci| {
        for (0..g.KH) |kh| {
            for (0..g.KW) |kw| {
                const k_row = (ci * g.KH + kh) * g.KW + kw;
                const dst_row = col[k_row * ld + col_off ..][0 .. g.H_out * g.W_out];
                for (0..g.H_out) |h_out| {
                    const dst = dst_row[h_out * g.W_out ..][0..g.W_out];
                    const ih = convInputIndex(h_out, g.stride_h, kh, g.pad_h, g.H) orelse {
                        @memset(dst, 0.0);
                        continue;
                    };
                    const src_row = x[ci * in_plane + ih * g.W ..][0..g.W];
                    for (dst, 0..) |*d, w_out| {
                        d.* = if (convInputIndex(w_out, g.stride_w, kw, g.pad_w, g.W)) |iw| src_row[iw] else 0.0;
                    }
                }
            }
        }
    }
}

/// 把列矩阵 col 的第 [col_off, col_off + L_out) 列按图像位置累加回单个样本的梯度 dx[C_in, H, W] (列转图像 (Column to Image, col2im))
fn col2imSample(dx: []f32, col: []const f32, ld: usize, col_off: usize, g: ConvGeometry) void {
    const in_plane = g.H * g.W;
    for (0..g.C_in) |ci| {
        for (0..g.KH) |kh| {
            for (0..g.KW) |kw| {
                const k_row = (ci * g.KH + kh) * g.KW + kw;
                const src_row = col[k_row * ld + col_off ..][0 .. g.H_out * g.W_out];
                for (0..g.H_out) |h_out| {
                    const ih = convInputIndex(h_out, g.stride_h, kh, g.pad_h, g.H) orelse continue;
                    const dst_row = dx[ci * in_plane + ih * g.W ..][0..g.W];
                    for (src_row[h_out * g.W_out ..][0..g.W_out], 0..) |v, w_out| {
                        if (convInputIndex(w_out, g.stride_w, kw, g.pad_w, g.W)) |iw| dst_row[iw] += v;
                    }
                }
            }
        }
    }
}

/// 卷积 (Convolution) 反向传播的批量图像转列 (Image to Column, im2col) 加单精度通用矩阵乘法
/// (Single-precision General Matrix Multiply, SGEMM) 路径；输入、权重与输出梯度必须连续。
/// 样本按块处理，每块 nb 个样本 (列缓冲区不超过 `conv_col_budget`)：
///   col[K_col, nb * L_out]    = 块内各样本 im2col 列矩阵按列拼接 (K_col = C_in * KH * KW，L_out = H_out * W_out)
///   dC_blk[C_out, nb * L_out] = 输出梯度 [nb, C_out, L_out] 重排为按输出通道的行
///   dW[C_out, K_col]         += dC_blk * col^T   (每块一次 SGEMM，beta = 1 跨块累加)
///   col                       = W^T * dC_blk     (每块一次 SGEMM)，再按样本列转图像 (col2im) 累加到 dX
/// 相比逐样本调用 SGEMM，内积维度从 L_out 增大为 nb * L_out，权重梯度块只在每块 (而非每个样本) 读写一次。
/// `col_budget` 为列缓冲区元素上限 (调用方传入 `conv_col_budget`)。任一维度为 0 或缓冲区分配失败时返回 false，由调用方改走逐元素路径。
fn convIm2colBackward(A: *Tensor, Wt: *Tensor, C: *Tensor, g: ConvGeometry, col_budget: usize) bool {
    const K_col = g.C_in * g.KH * g.KW;
    const L_out = g.H_out * g.W_out;
    if (g.N == 0 or K_col == 0 or L_out == 0 or g.C_out == 0) return false;

    const nb = std.math.clamp(col_budget / (K_col * L_out), 1, g.N);
    const allocator = std.heap.c_allocator;
    const col = allocator.alloc(f32, K_col * nb * L_out) catch return false;
    defer allocator.free(col);
    // 单样本块直接使用该样本连续的输出梯度 [C_out, L_out]，不需要重排缓冲区
    const dC_buf: []f32 = if (nb > 1) (allocator.alloc(f32, g.C_out * nb * L_out) catch return false) else &.{};
    defer if (nb > 1) allocator.free(dC_buf);

    const in_sample = g.C_in * g.H * g.W;
    const out_sample = g.C_out * L_out;
    var n0: usize = 0;
    while (n0 < g.N) : (n0 += nb) {
        const cnt = @min(nb, g.N - n0);
        const ld = cnt * L_out;
        const dC_blk: []const f32 = if (cnt == 1) C.grad[n0 * out_sample ..][0..out_sample] else blk: {
            for (0..g.C_out) |co| {
                for (0..cnt) |j| {
                    @memcpy(dC_buf[co * ld + j * L_out ..][0..L_out], C.grad[(n0 + j) * out_sample + co * L_out ..][0..L_out]);
                }
            }
            break :blk dC_buf[0 .. g.C_out * ld];
        };

        if (Wt.requires_grad) {
            for (0..cnt) |j| im2colSample(col, ld, j * L_out, A.data[(n0 + j) * in_sample ..][0..in_sample], g);
            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasTrans,
                @intCast(g.C_out),
                @intCast(K_col),
                @intCast(ld),
                1.0,
                dC_blk.ptr,
                @intCast(ld),
                col.ptr,
                @intCast(ld),
                1.0,
                Wt.grad.ptr,
                @intCast(K_col),
            );
        }

        if (A.requires_grad) {
            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasTrans,
                c.CblasNoTrans,
                @intCast(K_col),
                @intCast(ld),
                @intCast(g.C_out),
                1.0,
                Wt.data.ptr,
                @intCast(K_col),
                dC_blk.ptr,
                @intCast(ld),
                0.0,
                col.ptr,
                @intCast(ld),
            );
            for (0..cnt) |j| col2imSample(A.grad[(n0 + j) * in_sample ..][0..in_sample], col, ld, j * L_out, g);
        }
    }
    return true;
}

/// 缩放点积注意力 (Scaled Dot-Product Attention, SDPA) 反向传播：复用前向保存的注意力概率 P。
/// 与前向一样按键值头分组 (组内 groups * tq 个查询行连续)，记 dO 为输出梯度、s 为打分缩放系数：
///   dV   += P^T dO
///   dP    = dO V^T，dS = s * P ⊙ (dP - rowsum(P ⊙ dP))   (掩码位置 P = 0，梯度自然为 0)
///   dQ   += dS K，dK += dS^T Q
/// 全部矩阵乘法交给单精度通用矩阵乘法 (Single-precision General Matrix Multiply, SGEMM)，分组查询注意力
/// (Grouped-Query Attention, GQA) 的键值梯度在组内累加。
fn attentionBackward(self: *Op) !void {
    const Q = self.inputs[0];
    const K = self.inputs[1];
    const V = self.inputs[2];
    const O = self.outputs[0];
    const ctx = self.context.Attention;
    const shape = try tensor_mod.attention_kernels.Shape.init(Q, K, V, ctx.options);
    std.debug.assert(ctx.probs.len == shape.probsLen());
    const rows = shape.groups() * shape.tq;
    const d = shape.head_dim;
    const dv = shape.v_dim;
    const tk = shape.tk;
    if (rows == 0 or tk == 0 or dv == 0) return;
    const scale = ctx.options.resolveScale(d);
    const need_scores = Q.requires_grad or K.requires_grad;

    const allocator = std.heap.c_allocator;
    const ds: []f32 = if (need_scores) try allocator.alloc(f32, rows * tk) else &.{};
    defer if (need_scores) allocator.free(ds);
    const ld_d: c_int = @intCast(@max(d, 1));

    for (0..shape.batch) |b| {
        for (0..shape.kv_heads) |kv_h| {
            const q_row = shape.groupRow(b, kv_h);
            const k_row = shape.kvRow(b, kv_h);
            const p_blk = ctx.probs[q_row * tk ..][0 .. rows * tk];
            const do_blk = O.grad[q_row * dv ..][0 .. rows * dv];

            if (V.requires_grad) {
                const dv_blk = V.grad[k_row * dv ..][0 .. tk * dv];
                c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans, @intCast(tk), @intCast(dv), @intCast(rows), 1.0, p_blk.ptr, @intCast(tk), do_blk.ptr, @intCast(dv), 1.0, dv_blk.ptr, @intCast(dv));
            }
            if (!need_scores) continue;

            const v_blk = V.data[k_row * dv ..][0 .. tk * dv];
            c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans, @intCast(rows), @intCast(tk), @intCast(dv), 1.0, do_blk.ptr, @intCast(dv), v_blk.ptr, @intCast(dv), 0.0, ds.ptr, @intCast(tk));
            for (0..rows) |r| {
                const p_row = p_blk[r * tk .. (r + 1) * tk];
                const ds_row = ds[r * tk .. (r + 1) * tk];
                var delta: f32 = 0.0;
                for (p_row, ds_row) |p, dp| delta += p * dp;
                for (p_row, ds_row) |p, *g| g.* = scale * p * (g.* - delta);
            }

            if (Q.requires_grad) {
                const k_blk = K.data[k_row * d ..][0 .. tk * d];
                const dq_blk = Q.grad[q_row * d ..][0 .. rows * d];
                c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans, @intCast(rows), @intCast(d), @intCast(tk), 1.0, ds.ptr, @intCast(tk), k_blk.ptr, ld_d, 1.0, dq_blk.ptr, ld_d);
            }
            if (K.requires_grad) {
                const q_blk = Q.data[q_row * d ..][0 .. rows * d];
                const dk_blk = K.grad[k_row * d ..][0 .. tk * d];
                c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans, @intCast(tk), @intCast(d), @intCast(rows), 1.0, ds.ptr, @intCast(tk), q_blk.ptr, ld_d, 1.0, dk_blk.ptr, ld_d);
            }
        }
    }
}

pub fn backwardNN(self: *Op) !void {
    switch (self.op_type) {
        .Conv1D => {
            const A = self.inputs[0];
            const W = self.inputs[1];
            const C = self.outputs[0];
            const stride = self.context.Conv1D.stride;
            const padding = self.context.Conv1D.padding;

            const N = A.shape.dims[0];
            const C_in = A.shape.dims[1];
            const L = A.shape.dims[2];
            const C_out = W.shape.dims[0];
            const K = W.shape.dims[2];
            const L_out = C.shape.dims[2];

            const s_n = A.strides.dims[0];
            const s_c = A.strides.dims[1];
            const s_l = A.strides.dims[2];

            const w_co = W.strides.dims[0];
            const w_ci = W.strides.dims[1];
            const w_k = W.strides.dims[2];

            const o_n = C.strides.dims[0];
            const o_c = C.strides.dims[1];
            const o_l = C.strides.dims[2];

            if (self.inputs.len > 2) {
                const bias = self.inputs[2];
                if (bias.requires_grad) {
                    const b_stride = bias.strides.dims[0];
                    for (0..N) |n| {
                        for (0..C_out) |co| {
                            var acc: f32 = 0.0;
                            for (0..L_out) |l_out| {
                                acc += C.grad[n * o_n + co * o_c + l_out * o_l];
                            }
                            bias.grad[co * b_stride] += acc;
                        }
                    }
                }
            }

            const used_sgemm = (A.requires_grad or W.requires_grad) and A.isContiguous() and W.isContiguous() and C.isContiguous() and
                convIm2colBackward(A, W, C, .{
                    .N = N,
                    .C_in = C_in,
                    .H = 1,
                    .W = L,
                    .C_out = C_out,
                    .KH = 1,
                    .KW = K,
                    .H_out = 1,
                    .W_out = L_out,
                    .stride_h = 1,
                    .stride_w = stride,
                    .pad_h = 0,
                    .pad_w = padding,
                }, conv_col_budget);

            if (!used_sgemm and (A.requires_grad or W.requires_grad)) {
                for (0..N) |n| {
                    for (0..C_out) |co| {
                        for (0..L_out) |l_out| {
                            const grad_val = C.grad[n * o_n + co * o_c + l_out * o_l];
                            if (grad_val == 0.0) continue;

                            for (0..C_in) |ci| {
                                for (0..K) |k| {
                                    const il_signed: isize = @as(isize, @intCast(l_out * stride + k)) - @as(isize, @intCast(padding));
                                    if (il_signed < 0 or il_signed >= @as(isize, @intCast(L))) continue;
                                    const il: usize = @intCast(il_signed);

                                    if (W.requires_grad) {
                                        const input_val = A.data[n * s_n + ci * s_c + il * s_l];
                                        W.grad[co * w_co + ci * w_ci + k * w_k] += grad_val * input_val;
                                    }

                                    if (A.requires_grad) {
                                        const weight_val = W.data[co * w_co + ci * w_ci + k * w_k];
                                        A.grad[n * s_n + ci * s_c + il * s_l] += grad_val * weight_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        .Conv2D => {
            const A = self.inputs[0];
            const W = self.inputs[1];
            const C = self.outputs[0];
            const stride = self.context.Conv2D.stride;
            const padding = self.context.Conv2D.padding;

            const N = A.shape.dims[0];
            const C_in = A.shape.dims[1];
            const H = A.shape.dims[2];
            const W_in = A.shape.dims[3];
            const C_out = W.shape.dims[0];
            const KH = W.shape.dims[2];
            const KW = W.shape.dims[3];
            const H_out = C.shape.dims[2];
            const W_out = C.shape.dims[3];

            const s_n = A.strides.dims[0];
            const s_c = A.strides.dims[1];
            const s_h = A.strides.dims[2];
            const s_w = A.strides.dims[3];

            const w_co = W.strides.dims[0];
            const w_ci = W.strides.dims[1];
            const w_kh = W.strides.dims[2];
            const w_kw = W.strides.dims[3];

            const o_n = C.strides.dims[0];
            const o_c = C.strides.dims[1];
            const o_h = C.strides.dims[2];
            const o_w = C.strides.dims[3];

            // Bias gradient: db[co] += sum_{n, h_out, w_out} dC[n, co, h_out, w_out]
            if (self.inputs.len > 2) {
                const bias = self.inputs[2];
                if (bias.requires_grad) {
                    const b_stride = bias.strides.dims[0];
                    for (0..N) |n| {
                        for (0..C_out) |co| {
                            var acc: f32 = 0.0;
                            for (0..H_out) |h_out| {
                                for (0..W_out) |w_out| {
                                    acc += C.grad[n * o_n + co * o_c + h_out * o_h + w_out * o_w];
                                }
                            }
                            bias.grad[co * b_stride] += acc;
                        }
                    }
                }
            }

            const used_sgemm = (A.requires_grad or W.requires_grad) and A.isContiguous() and W.isContiguous() and C.isContiguous() and
                convIm2colBackward(A, W, C, .{
                    .N = N,
                    .C_in = C_in,
                    .H = H,
                    .W = W_in,
                    .C_out = C_out,
                    .KH = KH,
                    .KW = KW,
                    .H_out = H_out,
                    .W_out = W_out,
                    .stride_h = stride,
                    .stride_w = stride,
                    .pad_h = padding,
                    .pad_w = padding,
                }, conv_col_budget);

            if (!used_sgemm and (A.requires_grad or W.requires_grad)) {
                for (0..N) |n| {
                    for (0..C_out) |co| {
                        for (0..H_out) |h_out| {
                            for (0..W_out) |w_out| {
                                const grad_val = C.grad[n * o_n + co * o_c + h_out * o_h + w_out * o_w];
                                if (grad_val == 0.0) continue;

                                for (0..C_in) |ci| {
                                    for (0..KH) |kh| {
                                        const ih_signed: isize = @as(isize, @intCast(h_out * stride + kh)) - @as(isize, @intCast(padding));
                                        if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) continue;
                                        const ih: usize = @intCast(ih_signed);
                                        for (0..KW) |kw| {
                                            const iw_signed: isize = @as(isize, @intCast(w_out * stride + kw)) - @as(isize, @intCast(padding));
                                            if (iw_signed < 0 or iw_signed >= @as(isize, @intCast(W_in))) continue;
                                            const iw: usize = @intCast(iw_signed);

                                            if (W.requires_grad) {
                                                const input_val = A.data[n * s_n + ci * s_c + ih * s_h + iw * s_w];
                                                W.grad[co * w_co + ci * w_ci + kh * w_kh + kw * w_kw] += grad_val * input_val;
                                            }

                                            if (A.requires_grad) {
                                                const weight_val = W.data[co * w_co + ci * w_ci + kh * w_kh + kw * w_kw];
                                                A.grad[n * s_n + ci * s_c + ih * s_h + iw * s_w] += grad_val * weight_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        .ConvTranspose1D => {
            const ctx = self.context.ConvTranspose1D;
            try convTransposeBackward(self, ctx.stride, ctx.padding);
        },
        .ConvTranspose2D => {
            const ctx = self.context.ConvTranspose2D;
            try convTransposeBackward(self, ctx.stride, ctx.padding);
        },
        .MaxPool1D => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const N = A.shape.dims[0];
            const C_ch = A.shape.dims[1];
            const L = A.shape.dims[2];
            const L_out = C.shape.dims[2];

            const pool_size = self.context.MaxPool1D.pool_size;
            const stride = self.context.MaxPool1D.stride;
            const padding = self.context.MaxPool1D.padding;

            const s_n = A.strides.dims[0];
            const s_c = A.strides.dims[1];
            const s_l = A.strides.dims[2];

            const o_n = C.strides.dims[0];
            const o_c = C.strides.dims[1];
            const o_l = C.strides.dims[2];

            if (A.requires_grad) {
                for (0..N) |n| {
                    for (0..C_ch) |c_| {
                        for (0..L_out) |ol| {
                            const grad_val = C.grad[n * o_n + c_ * o_c + ol * o_l];
                            if (grad_val == 0.0) continue;

                            var max_val: f32 = -std.math.inf(f32);
                            var max_l: usize = 0;
                            var found = false;

                            for (0..pool_size) |pl| {
                                const il_signed: isize = @as(isize, @intCast(ol * stride + pl)) - @as(isize, @intCast(padding));
                                if (il_signed >= 0 and il_signed < @as(isize, @intCast(L))) {
                                    const il: usize = @intCast(il_signed);
                                    const val = A.data[n * s_n + c_ * s_c + il * s_l];
                                    if (!found or val > max_val) {
                                        max_val = val;
                                        max_l = il;
                                        found = true;
                                    }
                                }
                            }
                            if (found) {
                                A.grad[n * s_n + c_ * s_c + max_l * s_l] += grad_val;
                            }
                        }
                    }
                }
            }
        },
        .MaxPool2D => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const N = A.shape.dims[0];
            const C_ch = A.shape.dims[1];
            const H = A.shape.dims[2];
            const W = A.shape.dims[3];
            const H_out = C.shape.dims[2];
            const W_out = C.shape.dims[3];

            const pool_size = self.context.MaxPool2D.pool_size;
            const stride = self.context.MaxPool2D.stride;
            const padding = self.context.MaxPool2D.padding;

            const s_n = A.strides.dims[0];
            const s_c = A.strides.dims[1];
            const s_h = A.strides.dims[2];
            const s_w = A.strides.dims[3];

            const o_n = C.strides.dims[0];
            const o_c = C.strides.dims[1];
            const o_h = C.strides.dims[2];
            const o_w = C.strides.dims[3];

            if (A.requires_grad) {
                for (0..N) |n| {
                    for (0..C_ch) |c_| {
                        for (0..H_out) |h| {
                            for (0..W_out) |w| {
                                const grad_val = C.grad[n * o_n + c_ * o_c + h * o_h + w * o_w];
                                if (grad_val == 0.0) continue;

                                var max_val: f32 = -std.math.inf(f32);
                                var max_h: usize = 0;
                                var max_w: usize = 0;
                                var found = false;

                                for (0..pool_size) |ph| {
                                    const ih_signed: isize = @as(isize, @intCast(h * stride + ph)) - @as(isize, @intCast(padding));
                                    if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) continue;
                                    const ih: usize = @intCast(ih_signed);
                                    for (0..pool_size) |pw| {
                                        const iw_signed: isize = @as(isize, @intCast(w * stride + pw)) - @as(isize, @intCast(padding));
                                        if (iw_signed < 0 or iw_signed >= @as(isize, @intCast(W))) continue;
                                        const iw: usize = @intCast(iw_signed);
                                        const val = A.data[n * s_n + c_ * s_c + ih * s_h + iw * s_w];
                                        if (!found or val > max_val) {
                                            max_val = val;
                                            max_h = ih;
                                            max_w = iw;
                                            found = true;
                                        }
                                    }
                                }
                                if (found) {
                                    A.grad[n * s_n + c_ * s_c + max_h * s_h + max_w * s_w] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        },
        .AvgPool1D => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const N = A.shape.dims[0];
            const C_ch = A.shape.dims[1];
            const L = A.shape.dims[2];
            const L_out = C.shape.dims[2];

            const kernel_size = self.context.AvgPool1D.kernel_size;
            const stride = self.context.AvgPool1D.stride;
            const padding = self.context.AvgPool1D.padding;
            const pool_len = @as(f32, @floatFromInt(kernel_size));

            const s_n = A.strides.dims[0];
            const s_c = A.strides.dims[1];
            const s_l = A.strides.dims[2];

            const o_n = C.strides.dims[0];
            const o_c = C.strides.dims[1];
            const o_l = C.strides.dims[2];

            if (A.requires_grad) {
                for (0..N) |n| {
                    for (0..C_ch) |c_| {
                        for (0..L_out) |ol| {
                            const grad_val = C.grad[n * o_n + c_ * o_c + ol * o_l];
                            if (grad_val == 0.0) continue;
                            const distributed_grad = grad_val / pool_len;

                            for (0..kernel_size) |kl| {
                                const il_signed: isize = @as(isize, @intCast(ol * stride + kl)) - @as(isize, @intCast(padding));
                                if (il_signed >= 0 and il_signed < @as(isize, @intCast(L))) {
                                    const il: usize = @intCast(il_signed);
                                    A.grad[n * s_n + c_ * s_c + il * s_l] += distributed_grad;
                                }
                            }
                        }
                    }
                }
            }
        },
        .AvgPool2D => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const N = A.shape.dims[0];
            const C_ch = A.shape.dims[1];
            const H = A.shape.dims[2];
            const W = A.shape.dims[3];
            const H_out = C.shape.dims[2];
            const W_out = C.shape.dims[3];

            const kernel_size = self.context.AvgPool2D.kernel_size;
            const stride = self.context.AvgPool2D.stride;
            const padding = self.context.AvgPool2D.padding;
            const pool_area = @as(f32, @floatFromInt(kernel_size * kernel_size));

            const s_n = A.strides.dims[0];
            const s_c = A.strides.dims[1];
            const s_h = A.strides.dims[2];
            const s_w = A.strides.dims[3];

            const o_n = C.strides.dims[0];
            const o_c = C.strides.dims[1];
            const o_h = C.strides.dims[2];
            const o_w = C.strides.dims[3];

            if (A.requires_grad) {
                for (0..N) |n| {
                    for (0..C_ch) |c_| {
                        for (0..H_out) |oh| {
                            for (0..W_out) |ow| {
                                const grad_val = C.grad[n * o_n + c_ * o_c + oh * o_h + ow * o_w];
                                if (grad_val == 0.0) continue;
                                const distributed_grad = grad_val / pool_area;

                                for (0..kernel_size) |kh| {
                                    const ih_signed: isize = @as(isize, @intCast(oh * stride + kh)) - @as(isize, @intCast(padding));
                                    if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) continue;
                                    const ih: usize = @intCast(ih_signed);
                                    for (0..kernel_size) |kw| {
                                        const iw_signed: isize = @as(isize, @intCast(ow * stride + kw)) - @as(isize, @intCast(padding));
                                        if (iw_signed < 0 or iw_signed >= @as(isize, @intCast(W))) continue;
                                        const iw: usize = @intCast(iw_signed);
                                        A.grad[n * s_n + c_ * s_c + ih * s_h + iw * s_w] += distributed_grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        .AdaptiveAvgPool1D => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const N = A.shape.dims[0];
            const C_ch = A.shape.dims[1];
            const L = A.shape.dims[2];
            const output_size = self.context.AdaptiveAvgPool1D.output_size;

            const s_n = A.strides.dims[0];
            const s_c = A.strides.dims[1];
            const s_l = A.strides.dims[2];

            const o_n = C.strides.dims[0];
            const o_c = C.strides.dims[1];
            const o_l = C.strides.dims[2];

            if (A.requires_grad) {
                for (0..N) |n| {
                    for (0..C_ch) |c_| {
                        for (0..output_size) |ol| {
                            const grad_val = C.grad[n * o_n + c_ * o_c + ol * o_l];
                            if (grad_val == 0.0) continue;
                            const l_start = (ol * L) / output_size;
                            const l_end = ((ol + 1) * L + output_size - 1) / output_size;
                            const count = l_end - l_start;
                            const distributed_grad = grad_val / @as(f32, @floatFromInt(count));
                            for (l_start..l_end) |il| {
                                A.grad[n * s_n + c_ * s_c + il * s_l] += distributed_grad;
                            }
                        }
                    }
                }
            }
        },
        .AdaptiveAvgPool2D => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const N = A.shape.dims[0];
            const C_ch = A.shape.dims[1];
            const H = A.shape.dims[2];
            const W = A.shape.dims[3];
            const out_h = self.context.AdaptiveAvgPool2D.output_size[0];
            const out_w = self.context.AdaptiveAvgPool2D.output_size[1];

            const s_n = A.strides.dims[0];
            const s_c = A.strides.dims[1];
            const s_h = A.strides.dims[2];
            const s_w = A.strides.dims[3];

            const o_n = C.strides.dims[0];
            const o_c = C.strides.dims[1];
            const o_h = C.strides.dims[2];
            const o_w = C.strides.dims[3];

            if (A.requires_grad) {
                for (0..N) |n| {
                    for (0..C_ch) |c_| {
                        for (0..out_h) |oh| {
                            const h_start = (oh * H) / out_h;
                            const h_end = ((oh + 1) * H + out_h - 1) / out_h;
                            for (0..out_w) |ow| {
                                const grad_val = C.grad[n * o_n + c_ * o_c + oh * o_h + ow * o_w];
                                if (grad_val == 0.0) continue;
                                const w_start = (ow * W) / out_w;
                                const w_end = ((ow + 1) * W + out_w - 1) / out_w;
                                const area = (h_end - h_start) * (w_end - w_start);
                                const distributed_grad = grad_val / @as(f32, @floatFromInt(area));
                                for (h_start..h_end) |ih| {
                                    for (w_start..w_end) |iw| {
                                        A.grad[n * s_n + c_ * s_c + ih * s_h + iw * s_w] += distributed_grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        .Softmax => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const D = A.shape.dims[A.shape.len - 1];
            const M = A.shape.numel() / D;

            if (A.requires_grad) {
                for (0..M) |i| {
                    const row_out = C.data[i * D .. (i + 1) * D];
                    const row_grad_out = C.grad[i * D .. (i + 1) * D];
                    const row_grad_in = A.grad[i * D .. (i + 1) * D];

                    var sum_dy_y: f32 = 0.0;
                    for (row_grad_out, row_out) |dy, y| {
                        sum_dy_y += dy * y;
                    }

                    for (row_grad_in, row_out, row_grad_out) |*da, y, dy| {
                        da.* += y * (dy - sum_dy_y);
                    }
                }
            }
        },
        .RmsNorm => {
            // y = x * r * g，r = 1 / sqrt(mean(x^2) + eps) 取自前向保存的 inv_rms，g = w + weight_offset (无权重时 g = 1)：
            // dg_j += dy_j * x_j * r；dx_j += r * (g_j * dy_j - x_j * r^2 * Σ_k(dy_k * g_k * x_k) / D)
            const X = self.inputs[0];
            const W: ?*Tensor = if (self.inputs.len > 1) self.inputs[1] else null;
            const Y = self.outputs[0];
            const ctx = self.context.RmsNorm;
            const offset = ctx.options.weight_offset;
            const D = X.shape.dims[X.shape.len - 1];
            const M = X.shape.numel() / D;
            const d_f = @as(f32, @floatFromInt(D));
            for (0..M) |i| {
                const row_in = X.data[i * D .. (i + 1) * D];
                const row_grad_out = Y.grad[i * D .. (i + 1) * D];
                const r = if (ctx.inv_rms.len == M and ctx.inv_rms[i] > 0.0)
                    ctx.inv_rms[i]
                else blk: {
                    var sum_sq: f32 = 0.0;
                    for (row_in) |x| sum_sq += x * x;
                    break :blk 1.0 / @sqrt(sum_sq / d_f + ctx.options.eps);
                };

                if (W) |w| if (w.requires_grad) {
                    for (w.grad[0..D], row_grad_out, row_in) |*wg, dy, x| wg.* += dy * x * r;
                };

                if (X.requires_grad) {
                    const row_grad_in = X.grad[i * D .. (i + 1) * D];
                    var sum_dy_g_x: f32 = 0.0;
                    for (row_grad_out, row_in, 0..) |dy, x, j| {
                        const g = if (W) |w| w.data[j] + offset else 1.0;
                        sum_dy_g_x += dy * g * x;
                    }
                    const coef = r * r * sum_dy_g_x / d_f;
                    for (row_grad_in, row_grad_out, row_in, 0..) |*dx, dy, x, j| {
                        const g = if (W) |w| w.data[j] + offset else 1.0;
                        dx.* += r * (g * dy - x * coef);
                    }
                }
            }
        },
        .LayerNorm => {
            // 均值与标准差倒数取自前向保存的统计量，与前向使用同一求和顺序
            const X = self.inputs[0];
            const G = self.inputs[1];
            const B = self.inputs[2];
            const Y = self.outputs[0];
            const ctx = self.context.LayerNorm;
            const D = X.shape.dims[X.shape.len - 1];
            const M = X.shape.numel() / D;
            const d_f = @as(f32, @floatFromInt(D));
            std.debug.assert(ctx.save_mean.len == M and ctx.save_inv_std.len == M);

            for (0..M) |i| {
                const row_in = X.data[i * D .. (i + 1) * D];
                const row_grad_out = Y.grad[i * D .. (i + 1) * D];
                const mean_val = ctx.save_mean[i];
                const inv_std = ctx.save_inv_std[i];

                if (B.requires_grad) {
                    for (0..D) |j| {
                        B.grad[j] += row_grad_out[j];
                    }
                }

                if (G.requires_grad) {
                    for (0..D) |j| {
                        const x_hat = (row_in[j] - mean_val) * inv_std;
                        G.grad[j] += row_grad_out[j] * x_hat;
                    }
                }

                if (X.requires_grad) {
                    const row_grad_in = X.grad[i * D .. (i + 1) * D];
                    var sum_dx_hat: f32 = 0.0;
                    var sum_dx_hat_x_hat: f32 = 0.0;
                    for (0..D) |j| {
                        const x_hat = (row_in[j] - mean_val) * inv_std;
                        const dx_hat = row_grad_out[j] * G.data[j];
                        sum_dx_hat += dx_hat;
                        sum_dx_hat_x_hat += dx_hat * x_hat;
                    }
                    for (0..D) |j| {
                        const x_hat = (row_in[j] - mean_val) * inv_std;
                        const dx_hat = row_grad_out[j] * G.data[j];
                        row_grad_in[j] += (inv_std / d_f) * (d_f * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat);
                    }
                }
            }
        },
        .BatchNorm1d => {
            const X = self.inputs[0];
            const G = self.inputs[1];
            const B = self.inputs[2];
            const Y = self.outputs[0];
            const ctx = self.context.BatchNorm1d;

            const N = X.shape.dims[0];
            const C = X.shape.dims[1];
            const L = if (X.shape.len == 3) X.shape.dims[2] else 1;
            const m_f = @as(f32, @floatFromInt(N * L));

            for (0..C) |c_| {
                const mean_val = ctx.save_mean[c_];
                const inv_std = ctx.save_inv_std[c_];
                const g_val = G.data[c_];

                var dbeta: f32 = 0.0;
                var dgamma: f32 = 0.0;
                for (0..N) |n| {
                    const in_slice = X.data[(n * C + c_) * L .. (n * C + c_ + 1) * L];
                    const dy_slice = Y.grad[(n * C + c_) * L .. (n * C + c_ + 1) * L];
                    for (in_slice, dy_slice) |x_val, dy_val| {
                        const x_hat = (x_val - mean_val) * inv_std;
                        dbeta += dy_val;
                        dgamma += dy_val * x_hat;
                    }
                }

                if (B.requires_grad) {
                    B.grad[c_] += dbeta;
                }
                if (G.requires_grad) {
                    G.grad[c_] += dgamma;
                }

                if (X.requires_grad) {
                    if (ctx.training) {
                        const factor = (g_val * inv_std) / m_f;
                        for (0..N) |n| {
                            const in_slice = X.data[(n * C + c_) * L .. (n * C + c_ + 1) * L];
                            const dy_slice = Y.grad[(n * C + c_) * L .. (n * C + c_ + 1) * L];
                            const dx_slice = X.grad[(n * C + c_) * L .. (n * C + c_ + 1) * L];
                            for (in_slice, dy_slice, dx_slice) |x_val, dy_val, *dx_val| {
                                const x_hat = (x_val - mean_val) * inv_std;
                                dx_val.* += factor * (m_f * dy_val - dbeta - x_hat * dgamma);
                            }
                        }
                    } else {
                        const factor = g_val * inv_std;
                        for (0..N) |n| {
                            const dy_slice = Y.grad[(n * C + c_) * L .. (n * C + c_ + 1) * L];
                            const dx_slice = X.grad[(n * C + c_) * L .. (n * C + c_ + 1) * L];
                            for (dy_slice, dx_slice) |dy_val, *dx_val| {
                                dx_val.* += factor * dy_val;
                            }
                        }
                    }
                }
            }
        },
        .BatchNorm2d => {
            const X = self.inputs[0];
            const G = self.inputs[1];
            const B = self.inputs[2];
            const Y = self.outputs[0];
            const ctx = self.context.BatchNorm2d;

            const N = X.shape.dims[0];
            const C = X.shape.dims[1];
            const H = X.shape.dims[2];
            const W = X.shape.dims[3];
            const spatial_size = H * W;
            const m_f = @as(f32, @floatFromInt(N * spatial_size));

            for (0..C) |c_| {
                const mean_val = ctx.save_mean[c_];
                const inv_std = ctx.save_inv_std[c_];
                const g_val = G.data[c_];

                var dbeta: f32 = 0.0;
                var dgamma: f32 = 0.0;
                for (0..N) |n| {
                    const in_slice = X.data[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                    const dy_slice = Y.grad[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                    for (in_slice, dy_slice) |x_val, dy_val| {
                        const x_hat = (x_val - mean_val) * inv_std;
                        dbeta += dy_val;
                        dgamma += dy_val * x_hat;
                    }
                }

                if (B.requires_grad) {
                    B.grad[c_] += dbeta;
                }
                if (G.requires_grad) {
                    G.grad[c_] += dgamma;
                }

                if (X.requires_grad) {
                    if (ctx.training) {
                        const factor = (g_val * inv_std) / m_f;
                        for (0..N) |n| {
                            const in_slice = X.data[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                            const dy_slice = Y.grad[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                            const dx_slice = X.grad[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                            for (in_slice, dy_slice, dx_slice) |x_val, dy_val, *dx_val| {
                                const x_hat = (x_val - mean_val) * inv_std;
                                dx_val.* += factor * (m_f * dy_val - dbeta - x_hat * dgamma);
                            }
                        }
                    } else {
                        const factor = g_val * inv_std;
                        for (0..N) |n| {
                            const dy_slice = Y.grad[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                            const dx_slice = X.grad[(n * C + c_) * spatial_size .. (n * C + c_ + 1) * spatial_size];
                            for (dy_slice, dx_slice) |dy_val, *dx_val| {
                                dx_val.* += factor * dy_val;
                            }
                        }
                    }
                }
            }
        },
        .GroupNorm => {
            const X = self.inputs[0];
            const G = self.inputs[1];
            const B = self.inputs[2];
            const Y = self.outputs[0];
            const ctx = self.context.GroupNorm;

            const N = X.shape.dims[0];
            const C = X.shape.dims[1];
            var spatial_size: usize = 1;
            for (2..X.shape.len) |d| {
                spatial_size *= X.shape.dims[d];
            }
            const c_per_g = C / ctx.num_groups;
            const group_elems = c_per_g * spatial_size;
            const m_f = @as(f32, @floatFromInt(group_elems));

            for (0..N) |n| {
                for (0..ctx.num_groups) |g| {
                    const ng_idx = n * ctx.num_groups + g;
                    const mean_val = ctx.save_mean[ng_idx];
                    const inv_std = ctx.save_inv_std[ng_idx];

                    var sum_dx_hat: f32 = 0.0;
                    var sum_dx_hat_x_hat: f32 = 0.0;

                    for (0..c_per_g) |cg| {
                        const c_idx = g * c_per_g + cg;
                        const g_val = G.data[c_idx];
                        const ch_offset = (n * C + c_idx) * spatial_size;

                        var dbeta: f32 = 0.0;
                        var dgamma: f32 = 0.0;
                        for (0..spatial_size) |s| {
                            const x_hat = (X.data[ch_offset + s] - mean_val) * inv_std;
                            const dy_val = Y.grad[ch_offset + s];
                            const dx_hat = dy_val * g_val;
                            sum_dx_hat += dx_hat;
                            sum_dx_hat_x_hat += dx_hat * x_hat;
                            dbeta += dy_val;
                            dgamma += dy_val * x_hat;
                        }
                        if (B.requires_grad) B.grad[c_idx] += dbeta;
                        if (G.requires_grad) G.grad[c_idx] += dgamma;
                    }

                    if (X.requires_grad) {
                        const factor = inv_std / m_f;
                        for (0..c_per_g) |cg| {
                            const c_idx = g * c_per_g + cg;
                            const g_val = G.data[c_idx];
                            const ch_offset = (n * C + c_idx) * spatial_size;
                            for (0..spatial_size) |s| {
                                const x_hat = (X.data[ch_offset + s] - mean_val) * inv_std;
                                const dx_hat = Y.grad[ch_offset + s] * g_val;
                                X.grad[ch_offset + s] += factor * (m_f * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat);
                            }
                        }
                    }
                }
            }
        },
        .Dropout => {
            const X = self.inputs[0];
            const Y = self.outputs[0];
            const mask_scale = self.context.Dropout.mask_scale;
            if (X.requires_grad) {
                for (X.grad, Y.grad, mask_scale) |*dx, dy, m| {
                    dx.* += dy * m;
                }
            }
        },
        .RoPE => {
            const X = self.inputs[0];
            const Y = self.outputs[0];
            if (X.requires_grad and X.shape.numel() > 0) {
                // 旋转矩阵是正交矩阵：dX = R^T dY，即对上游梯度逐行施加共享旋转表的逆旋转 (未旋转分量的梯度原样传递)
                const start_pos = self.context.RoPE.start_pos;
                const D = X.shape.dims[X.shape.len - 1];
                const T = if (X.shape.len >= 2) X.shape.dims[X.shape.len - 2] else 1;
                const outer = X.shape.numel() / (T * D);
                const table = try tensor_mod.RopeTable.init(D, self.context.RoPE.options);

                const allocator = std.heap.c_allocator;
                const scratch = try allocator.alloc(f32, D + 2 * table.num_angles);
                defer allocator.free(scratch);
                const row = scratch[0..D];
                const cos_tab = scratch[D..][0..table.num_angles];
                const sin_tab = scratch[D + table.num_angles ..][0..table.num_angles];
                for (0..T) |t| {
                    table.angles(start_pos + t, cos_tab, sin_tab);
                    for (0..outer) |o| {
                        const off = (o * T + t) * D;
                        @memcpy(row, Y.grad[off..][0..D]);
                        table.applyRow(row, cos_tab, sin_tab, true);
                        for (X.grad[off..][0..D], row) |*dx, g| dx.* += g;
                    }
                }
            }
        },
        .BatchMatMul => {
            const A = self.inputs[0];
            const B = self.inputs[1];
            const C = self.outputs[0];

            const batch_size = A.shape.dims[0];
            const num_heads = A.shape.dims[1];
            const M = A.shape.dims[2];
            const K = A.shape.dims[3];
            const N = B.shape.dims[3];

            const sA_b = A.strides.dims[0];
            const sA_h = A.strides.dims[1];
            const sB_b = B.strides.dims[0];
            const sB_h = B.strides.dims[1];
            const sC_b = C.strides.dims[0];
            const sC_h = C.strides.dims[1];

            for (0..batch_size) |b| {
                for (0..num_heads) |h| {
                    const ptrA = A.data.ptr + b * sA_b + h * sA_h;
                    const ptrB = B.data.ptr + b * sB_b + h * sB_h;
                    const ptrdC = C.grad.ptr + b * sC_b + h * sC_h;

                    if (A.requires_grad) {
                        const ptrdA = A.grad.ptr + b * sA_b + h * sA_h;
                        c.cblas_sgemm(
                            c.CblasRowMajor,
                            c.CblasNoTrans,
                            c.CblasTrans,
                            @intCast(M),
                            @intCast(K),
                            @intCast(N),
                            1.0,
                            ptrdC,
                            @intCast(N),
                            ptrB,
                            @intCast(N),
                            1.0,
                            ptrdA,
                            @intCast(K),
                        );
                    }

                    if (B.requires_grad) {
                        const ptrdB = B.grad.ptr + b * sB_b + h * sB_h;
                        c.cblas_sgemm(
                            c.CblasRowMajor,
                            c.CblasTrans,
                            c.CblasNoTrans,
                            @intCast(K),
                            @intCast(N),
                            @intCast(M),
                            1.0,
                            ptrA,
                            @intCast(K),
                            ptrdC,
                            @intCast(N),
                            1.0,
                            ptrdB,
                            @intCast(N),
                        );
                    }
                }
            }
        },
        .Attention => try attentionBackward(self),
        .Embedding => {
            const W = self.inputs[0];
            const X = self.inputs[1];
            const Y = self.outputs[0];

            const D = W.shape.dims[1];
            const num_indices = X.shape.numel();

            if (W.requires_grad) {
                if (X.isContiguous() and X.data.len >= num_indices) {
                    for (0..num_indices) |i| {
                        const idx = @as(usize, @intFromFloat(X.data[i]));
                        const w_grad_row = W.grad[idx * D .. (idx + 1) * D];
                        const y_grad_row = Y.grad[i * D .. (i + 1) * D];

                        for (w_grad_row, y_grad_row) |*wg, yg| {
                            wg.* += yg;
                        }
                    }
                } else {
                    var coord: [8]usize = @splat(0);
                    const rank = X.shape.len;
                    for (0..num_indices) |i| {
                        var x_flat: usize = 0;
                        for (0..rank) |d| x_flat += coord[d] * X.strides.dims[d];
                        const idx = @as(usize, @intFromFloat(X.data[x_flat]));
                        const w_grad_row = W.grad[idx * D .. (idx + 1) * D];
                        const y_grad_row = Y.grad[i * D .. (i + 1) * D];

                        for (w_grad_row, y_grad_row) |*wg, yg| {
                            wg.* += yg;
                        }
                        X.shape.incrementCoord(&coord);
                    }
                }
            }
        },
        .TopK => {
            const X = self.inputs[0];
            const values = self.outputs[0];
            const indices = self.outputs[1];
            if (X.requires_grad) {
                const E = X.shape.dims[X.shape.len - 1];
                const K = self.context.TopK.k;
                const N = X.shape.numel() / E;
                for (0..N) |n| {
                    for (0..K) |k| {
                        const e_raw: usize = @intFromFloat(indices.data[n * K + k]);
                        const e: usize = @min(e_raw, E - 1);
                        X.grad[n * E + e] += values.grad[n * K + k];
                    }
                }
            }
        },
        .GatherMatMul => {
            const X = self.inputs[0];
            const indices = self.inputs[1];
            const Y = self.outputs[0];
            const ctx = self.context.GatherMatMul;
            const E = ctx.num_experts;
            const weights = self.inputs[2 .. 2 + E];
            const biases = if (ctx.has_bias) self.inputs[2 + E .. 2 + 2 * E] else null;

            const N = indices.shape.dims[0];
            const K = indices.shape.dims[1];
            const in_dim = weights[0].shape.dims[0];
            const out_dim = weights[0].shape.dims[1];
            const rows = N * K;
            if (rows == 0 or in_dim == 0 or out_dim == 0) return;
            const x_n_stride: usize = X.strides.dims[0];
            const x_k_stride: usize = if (X.shape.len == 3) X.strides.dims[1] else 0;

            // 与前向一致按专家分组：每个专家把所属行的 X / dY 收集成稠密块，
            // 用一次 SGEMM 求 dW_e += X_gᵀ dY_g、一次 SGEMM 求 dX_g = dY_g W_eᵀ，代替逐行 M = 1 的矩阵乘法
            const allocator = std.heap.c_allocator;
            var groups = try tensor_mod.nn_kernels.ExpertGroups.init(allocator, indices.data[0..rows], E);
            defer groups.deinit(allocator);
            const G = groups.max_group;
            const scratch = try allocator.alloc(f32, G * (2 * in_dim + out_dim));
            defer allocator.free(scratch);
            const x_buf = scratch[0 .. G * in_dim];
            const dx_buf = scratch[G * in_dim .. 2 * G * in_dim];
            const dy_buf = scratch[2 * G * in_dim ..];

            for (0..E) |e| {
                const group = groups.rowsOf(e);
                if (group.len == 0) continue;
                const w_e = weights[e];
                const bias_grad: ?[]f32 = if (biases) |bs| (if (bs[e].requires_grad) bs[e].grad[0..out_dim] else null) else null;
                if (!X.requires_grad and !w_e.requires_grad and bias_grad == null) continue;
                const m: c_int = @intCast(group.len);

                for (group, 0..) |r, g| {
                    @memcpy(dy_buf[g * out_dim ..][0..out_dim], Y.grad[r * out_dim ..][0..out_dim]);
                    if (w_e.requires_grad) {
                        const x_off = (r / K) * x_n_stride + (r % K) * x_k_stride;
                        @memcpy(x_buf[g * in_dim ..][0..in_dim], X.data[x_off..][0..in_dim]);
                    }
                }

                if (w_e.requires_grad) {
                    c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans, @intCast(in_dim), @intCast(out_dim), m, 1.0, x_buf.ptr, @intCast(in_dim), dy_buf.ptr, @intCast(out_dim), 1.0, w_e.grad.ptr, @intCast(out_dim));
                }

                if (X.requires_grad) {
                    c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans, m, @intCast(in_dim), @intCast(out_dim), 1.0, dy_buf.ptr, @intCast(out_dim), w_e.data.ptr, @intCast(out_dim), 0.0, dx_buf.ptr, @intCast(in_dim));
                    for (group, 0..) |r, g| {
                        const x_off = (r / K) * x_n_stride + (r % K) * x_k_stride;
                        for (X.grad[x_off..][0..in_dim], dx_buf[g * in_dim ..][0..in_dim]) |*dst, src| dst.* += src;
                    }
                }

                if (bias_grad) |bg| {
                    for (0..group.len) |g| {
                        for (bg, dy_buf[g * out_dim ..][0..out_dim]) |*dst, dy_val| dst.* += dy_val;
                    }
                }
            }
        },
        .Q4MatMul => {
            // 量化权重冻结，只对输入求梯度: dX[rows x in] += dY[rows x out] * W[out x in]
            const X = self.inputs[0];
            const Y = self.outputs[0];
            if (!X.requires_grad) return;
            if (!X.isContiguous()) return error.NonContiguousTensor;
            const ctx = self.context.Q4MatMul;
            const QK = tensor_mod.quant.Q4Block.QK;
            const in_f = ctx.in_features;
            const out_f = ctx.out_features;
            const rows = X.shape.numel() / in_f;
            if (rows == 0 or out_f == 0) return;
            const blocks_per_row = in_f / QK;

            // 按输出特征分块反量化权重行，再交给单精度通用矩阵乘法 (Single-precision General Matrix Multiply, SGEMM)
            const tile = @min(out_f, q4_backward_row_tile);
            const w_buf = try std.heap.c_allocator.alloc(f32, tile * in_f);
            defer std.heap.c_allocator.free(w_buf);
            var j0: usize = 0;
            while (j0 < out_f) : (j0 += tile) {
                const j1 = @min(j0 + tile, out_f);
                const n_rows = j1 - j0;
                tensor_mod.quant.dequantizeSlice(w_buf[0 .. n_rows * in_f], ctx.blocks[j0 * blocks_per_row .. j1 * blocks_per_row]);
                c.cblas_sgemm(
                    c.CblasRowMajor,
                    c.CblasNoTrans,
                    c.CblasNoTrans,
                    @intCast(rows),
                    @intCast(in_f),
                    @intCast(n_rows),
                    1.0,
                    Y.grad.ptr + j0,
                    @intCast(out_f),
                    w_buf.ptr,
                    @intCast(in_f),
                    1.0,
                    X.grad.ptr,
                    @intCast(in_f),
                );
            }
        },
        else => unreachable,
    }
}

test "convIm2colBackward gives the same gradients for any sample block size" {
    const allocator = std.testing.allocator;
    const g: ConvGeometry = .{ .N = 5, .C_in = 2, .H = 6, .W = 5, .C_out = 3, .KH = 3, .KW = 2, .H_out = 3, .W_out = 3, .stride_h = 2, .stride_w = 2, .pad_h = 1, .pad_w = 1 };
    const K_col = g.C_in * g.KH * g.KW;
    const L_out = g.H_out * g.W_out;

    const A = try tensor_mod.zeros(allocator, &.{ g.N, g.C_in, g.H, g.W });
    defer tensor_mod.free(allocator, A);
    const Wt = try tensor_mod.zeros(allocator, &.{ g.C_out, g.C_in, g.KH, g.KW });
    defer tensor_mod.free(allocator, Wt);
    const C = try tensor_mod.zeros(allocator, &.{ g.N, g.C_out, g.H_out, g.W_out });
    defer tensor_mod.free(allocator, C);
    for (A.data, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.37);
    for (Wt.data, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.53);
    const dC = try allocator.alloc(f32, C.data.len);
    defer allocator.free(dC);
    for (dC, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.71 + 0.3);

    // 逐元素参考：dW[co, ci, kh, kw] = sum dC * x，dX[n, ci, ih, iw] = sum dC * W
    const ref_dW = try allocator.alloc(f32, Wt.data.len);
    defer allocator.free(ref_dW);
    const ref_dX = try allocator.alloc(f32, A.data.len);
    defer allocator.free(ref_dX);
    @memset(ref_dW, 0.0);
    @memset(ref_dX, 0.0);
    for (0..g.N) |n| {
        for (0..g.C_out) |co| {
            for (0..g.H_out) |ho| {
                for (0..g.W_out) |wo| {
                    const gv = dC[((n * g.C_out + co) * g.H_out + ho) * g.W_out + wo];
                    for (0..g.C_in) |ci| {
                        for (0..g.KH) |kh| {
                            const ih = convInputIndex(ho, g.stride_h, kh, g.pad_h, g.H) orelse continue;
                            for (0..g.KW) |kw| {
                                const iw = convInputIndex(wo, g.stride_w, kw, g.pad_w, g.W) orelse continue;
                                const x_idx = ((n * g.C_in + ci) * g.H + ih) * g.W + iw;
                                const w_idx = ((co * g.C_in + ci) * g.KH + kh) * g.KW + kw;
                                ref_dW[w_idx] += gv * A.data[x_idx];
                                ref_dX[x_idx] += gv * Wt.data[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // 每块 1、2 个样本 (最后一块不满) 与整批一次处理
    const dX = try allocator.alloc(f32, A.data.len);
    defer allocator.free(dX);
    const dW = try allocator.alloc(f32, Wt.data.len);
    defer allocator.free(dW);
    const saved = .{ A.grad, Wt.grad, C.grad, A.requires_grad, Wt.requires_grad };
    defer {
        A.grad = saved[0];
        Wt.grad = saved[1];
        C.grad = saved[2];
        A.requires_grad = saved[3];
        Wt.requires_grad = saved[4];
    }
    A.grad = dX;
    Wt.grad = dW;
    C.grad = dC;
    A.requires_grad = true;
    Wt.requires_grad = true;
    for ([_]usize{ K_col * L_out, 2 * K_col * L_out, g.N * K_col * L_out }) |budget| {
        @memset(dX, 0.0);
        @memset(dW, 0.0);
        try std.testing.expect(convIm2colBackward(A, Wt, C, g, budget));
        for (ref_dW, dW) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-4);
        for (ref_dX, dX) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-4);
    }
}
