const std = @import("std");
const c = @import("../cblas.zig");
const op_mod = @import("op.zig");
const Op = op_mod.Op;

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

            const K_col = C_in * K;
            var used_sgemm = false;

            if ((A.requires_grad or W.requires_grad) and A.isContiguous() and W.isContiguous() and C.isContiguous() and K_col > 0 and L_out > 0 and C_out > 0) {
                var stack_buf: [4096]f32 = undefined;
                const need_len = K_col * L_out;
                const heap_buf: ?[]f32 = if (need_len > stack_buf.len)
                    (std.heap.c_allocator.alloc(f32, need_len) catch null)
                else
                    null;
                defer if (heap_buf) |hb| std.heap.c_allocator.free(hb);

                const col_opt: ?[]f32 = if (need_len <= stack_buf.len) stack_buf[0..need_len] else heap_buf;
                if (col_opt) |col_buf| {
                    used_sgemm = true;
                    for (0..N) |n| {
                        const dC_n = C.grad[n * C_out * L_out .. (n + 1) * C_out * L_out];

                        if (W.requires_grad) {
                            // 图像转列 (Image to Column, im2col) 展开 A_n -> col_buf [K_col, L_out]
                            for (0..C_in) |ci| {
                                for (0..K) |k| {
                                    const k_row = ci * K + k;
                                    const col_row = col_buf[k_row * L_out .. (k_row + 1) * L_out];
                                    for (0..L_out) |l_out| {
                                        const il_signed: isize = @as(isize, @intCast(l_out * stride + k)) - @as(isize, @intCast(padding));
                                        if (il_signed >= 0 and il_signed < @as(isize, @intCast(L))) {
                                            const il: usize = @intCast(il_signed);
                                            col_row[l_out] = A.data[n * s_n + ci * s_c + il * s_l];
                                        } else {
                                            col_row[l_out] = 0.0;
                                        }
                                    }
                                }
                            }

                            // dW += dC_n * col_buf^T
                            c.cblas_sgemm(
                                c.CblasRowMajor,
                                c.CblasNoTrans,
                                c.CblasTrans,
                                @intCast(C_out),
                                @intCast(K_col),
                                @intCast(L_out),
                                1.0,
                                dC_n.ptr,
                                @intCast(L_out),
                                col_buf.ptr,
                                @intCast(L_out),
                                1.0,
                                W.grad.ptr,
                                @intCast(K_col),
                            );
                        }

                        if (A.requires_grad) {
                            // col_buf = W^T * dC_n -> [K_col, L_out]
                            c.cblas_sgemm(
                                c.CblasRowMajor,
                                c.CblasTrans,
                                c.CblasNoTrans,
                                @intCast(K_col),
                                @intCast(L_out),
                                @intCast(C_out),
                                1.0,
                                W.data.ptr,
                                @intCast(K_col),
                                dC_n.ptr,
                                @intCast(L_out),
                                0.0,
                                col_buf.ptr,
                                @intCast(L_out),
                            );

                            // 列转图像 (Column to Image, col2im) 累加 col_buf -> A.grad[n]
                            for (0..C_in) |ci| {
                                for (0..K) |k| {
                                    const k_row = ci * K + k;
                                    const col_row = col_buf[k_row * L_out .. (k_row + 1) * L_out];
                                    for (0..L_out) |l_out| {
                                        const il_signed: isize = @as(isize, @intCast(l_out * stride + k)) - @as(isize, @intCast(padding));
                                        if (il_signed >= 0 and il_signed < @as(isize, @intCast(L))) {
                                            const il: usize = @intCast(il_signed);
                                            A.grad[n * s_n + ci * s_c + il * s_l] += col_row[l_out];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

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

            const K_col = C_in * KH * KW;
            const L_out = H_out * W_out;
            var used_sgemm = false;

            if ((A.requires_grad or W.requires_grad) and A.isContiguous() and W.isContiguous() and C.isContiguous() and K_col > 0 and L_out > 0 and C_out > 0) {
                var stack_buf: [4096]f32 = undefined;
                const need_len = K_col * L_out;
                const heap_buf: ?[]f32 = if (need_len > stack_buf.len)
                    (std.heap.c_allocator.alloc(f32, need_len) catch null)
                else
                    null;
                defer if (heap_buf) |hb| std.heap.c_allocator.free(hb);

                const col_opt: ?[]f32 = if (need_len <= stack_buf.len) stack_buf[0..need_len] else heap_buf;
                if (col_opt) |col_buf| {
                    used_sgemm = true;
                    for (0..N) |n| {
                        const dC_n = C.grad[n * C_out * L_out .. (n + 1) * C_out * L_out];

                        if (W.requires_grad) {
                            // 图像转列 (Image to Column, im2col) 展开 A_n -> col_buf [K_col, L_out]
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
                                                const iw_s: isize = @as(isize, @intCast(w_out * stride + kw)) - @as(isize, @intCast(padding));
                                                if (iw_s >= 0 and iw_s < @as(isize, @intCast(W_in))) {
                                                    const iw: usize = @intCast(iw_s);
                                                    col_row[h_out * W_out + w_out] = A.data[n * s_n + ci * s_c + ih * s_h + iw * s_w];
                                                } else {
                                                    col_row[h_out * W_out + w_out] = 0.0;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // dW += dC_n * col_buf^T
                            c.cblas_sgemm(
                                c.CblasRowMajor,
                                c.CblasNoTrans,
                                c.CblasTrans,
                                @intCast(C_out),
                                @intCast(K_col),
                                @intCast(L_out),
                                1.0,
                                dC_n.ptr,
                                @intCast(L_out),
                                col_buf.ptr,
                                @intCast(L_out),
                                1.0,
                                W.grad.ptr,
                                @intCast(K_col),
                            );
                        }

                        if (A.requires_grad) {
                            // col_buf = W^T * dC_n -> [K_col, L_out]
                            c.cblas_sgemm(
                                c.CblasRowMajor,
                                c.CblasTrans,
                                c.CblasNoTrans,
                                @intCast(K_col),
                                @intCast(L_out),
                                @intCast(C_out),
                                1.0,
                                W.data.ptr,
                                @intCast(K_col),
                                dC_n.ptr,
                                @intCast(L_out),
                                0.0,
                                col_buf.ptr,
                                @intCast(L_out),
                            );

                            // 列转图像 (Column to Image, col2im) 累加 col_buf -> A.grad[n]
                            for (0..C_in) |ci| {
                                for (0..KH) |kh| {
                                    for (0..KW) |kw| {
                                        const k_row = (ci * KH + kh) * KW + kw;
                                        const col_row = col_buf[k_row * L_out .. (k_row + 1) * L_out];
                                        for (0..H_out) |h_out| {
                                            const ih_signed: isize = @as(isize, @intCast(h_out * stride + kh)) - @as(isize, @intCast(padding));
                                            if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) continue;
                                            const ih: usize = @intCast(ih_signed);
                                            for (0..W_out) |w_out| {
                                                const iw_s: isize = @as(isize, @intCast(w_out * stride + kw)) - @as(isize, @intCast(padding));
                                                if (iw_s >= 0 and iw_s < @as(isize, @intCast(W_in))) {
                                                    const iw: usize = @intCast(iw_s);
                                                    A.grad[n * s_n + ci * s_c + ih * s_h + iw * s_w] += col_row[h_out * W_out + w_out];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

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
            const A = self.inputs[0];
            const W = self.inputs[1];
            const C = self.outputs[0];
            const bias = if (self.inputs.len > 2) self.inputs[2] else null;
            const stride = self.context.ConvTranspose1D.stride;
            const padding = self.context.ConvTranspose1D.padding;

            const N = A.shape.dims[0];
            const C_in = A.shape.dims[1];
            const L_in = A.shape.dims[2];

            const C_out = W.shape.dims[1];
            const K = W.shape.dims[2];
            const L_out = C.shape.dims[2];

            if (bias) |b| {
                if (b.requires_grad) {
                    for (0..N) |n| {
                        for (0..C_out) |co| {
                            for (0..L_out) |l| {
                                b.grad[co] += C.grad[(n * C_out + co) * L_out + l];
                            }
                        }
                    }
                }
            }

            for (0..N) |n| {
                for (0..C_in) |ci| {
                    for (0..L_in) |l_in| {
                        const in_idx = (n * C_in + ci) * L_in + l_in;
                        const input_val = A.data[in_idx];

                        for (0..C_out) |co| {
                            for (0..K) |k| {
                                const out_l_raw = l_in * stride + k;
                                if (out_l_raw < padding) continue;
                                const out_l = out_l_raw - padding;
                                if (out_l >= L_out) continue;

                                const out_idx = (n * C_out + co) * L_out + out_l;
                                const grad_out = C.grad[out_idx];
                                if (grad_out == 0.0) continue;

                                const w_idx = (ci * C_out + co) * K + k;
                                if (W.requires_grad) {
                                    W.grad[w_idx] += grad_out * input_val;
                                }
                                if (A.requires_grad) {
                                    A.grad[in_idx] += grad_out * W.data[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        },
        .ConvTranspose2D => {
            const A = self.inputs[0];
            const W = self.inputs[1];
            const C = self.outputs[0];
            const bias = if (self.inputs.len > 2) self.inputs[2] else null;
            const stride = self.context.ConvTranspose2D.stride;
            const padding = self.context.ConvTranspose2D.padding;

            const N = A.shape.dims[0];
            const C_in = A.shape.dims[1];
            const H_in = A.shape.dims[2];
            const W_in = A.shape.dims[3];

            const C_out = W.shape.dims[1];
            const KH = W.shape.dims[2];
            const KW = W.shape.dims[3];

            const H_out = C.shape.dims[2];
            const W_out = C.shape.dims[3];

            // Bias gradient
            if (bias) |b| {
                if (b.requires_grad) {
                    for (0..N) |n| {
                        for (0..C_out) |co| {
                            for (0..H_out) |h| {
                                for (0..W_out) |w| {
                                    b.grad[co] += C.grad[n * (C_out * H_out * W_out) + co * (H_out * W_out) + h * W_out + w];
                                }
                            }
                        }
                    }
                }
            }

            for (0..N) |n| {
                for (0..C_in) |ci| {
                    for (0..H_in) |h| {
                        for (0..W_in) |w| {
                            const in_idx = n * (C_in * H_in * W_in) + ci * (H_in * W_in) + h * W_in + w;
                            const input_val = A.data[in_idx];

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

                                        const out_idx = n * (C_out * H_out * W_out) + co * (H_out * W_out) + out_h * W_out + out_w;
                                        const grad_out = C.grad[out_idx];
                                        if (grad_out == 0.0) continue;

                                        const w_idx = ci * (C_out * KH * KW) + co * (KH * KW) + kh * KW + kw;

                                        if (W.requires_grad) {
                                            W.grad[w_idx] += grad_out * input_val;
                                        }

                                        if (A.requires_grad) {
                                            A.grad[in_idx] += grad_out * W.data[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
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
            const M = A.data.len / D;

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
            const X = self.inputs[0];
            const G = self.inputs[1];
            const Y = self.outputs[0];
            const eps = self.context.RmsNorm.eps;
            const D = X.shape.dims[X.shape.len - 1];
            const M = X.data.len / D;

            for (0..M) |i| {
                const row_in = X.data[i * D .. (i + 1) * D];
                const row_grad_out = Y.grad[i * D .. (i + 1) * D];

                var sum_x2: f32 = 0.0;
                for (row_in) |val| {
                    sum_x2 += val * val;
                }
                const scale = 1.0 / @sqrt(sum_x2 / @as(f32, @floatFromInt(D)) + eps);

                if (G.requires_grad) {
                    for (0..D) |j| {
                        G.grad[j] += row_grad_out[j] * row_in[j] * scale;
                    }
                }

                if (X.requires_grad) {
                    const row_grad_in = X.grad[i * D .. (i + 1) * D];

                    var sum_dy_g_x: f32 = 0.0;
                    for (0..D) |j| {
                        sum_dy_g_x += row_grad_out[j] * G.data[j] * row_in[j];
                    }

                    for (0..D) |j| {
                        const term1 = G.data[j] * row_grad_out[j];
                        const term2 = row_in[j] * scale * scale * sum_dy_g_x / @as(f32, @floatFromInt(D));
                        row_grad_in[j] += scale * (term1 - term2);
                    }
                }
            }
        },
        .LayerNorm => {
            const X = self.inputs[0];
            const G = self.inputs[1];
            const B = self.inputs[2];
            const Y = self.outputs[0];
            const eps = self.context.LayerNorm.eps;
            const D = X.shape.dims[X.shape.len - 1];
            const M = X.data.len / D;
            const d_f = @as(f32, @floatFromInt(D));

            for (0..M) |i| {
                const row_in = X.data[i * D .. (i + 1) * D];
                const row_grad_out = Y.grad[i * D .. (i + 1) * D];

                var sum_x: f32 = 0.0;
                for (row_in) |val| sum_x += val;
                const mean_val = sum_x / d_f;

                var var_sum: f32 = 0.0;
                for (row_in) |val| {
                    const diff = val - mean_val;
                    var_sum += diff * diff;
                }
                const inv_std = 1.0 / @sqrt(var_sum / d_f + eps);

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
            if (X.requires_grad) {
                const start_pos = self.context.RoPE.start_pos;
                const rotary_offset = self.context.RoPE.rotary_offset;
                const D = X.shape.dims[X.shape.len - 1];
                const T = if (X.shape.len >= 2) X.shape.dims[X.shape.len - 2] else 1;
                const outer = X.data.len / (T * D);
                const rot_dim = D - rotary_offset;
                const half = rot_dim / 2;
                const rot_dim_f = @as(f32, @floatFromInt(rot_dim));

                for (0..outer) |o| {
                    for (0..T) |t| {
                        const dx_row = X.grad[(o * T + t) * D .. (o * T + t + 1) * D];
                        const dy_row = Y.grad[(o * T + t) * D .. (o * T + t + 1) * D];
                        for (0..rotary_offset) |j| {
                            dx_row[j] += dy_row[j];
                        }
                        const pos_f = @as(f32, @floatFromInt(start_pos + t));
                        for (0..half) |i| {
                            const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / rot_dim_f);
                            const theta = pos_f * freq;
                            const cos_t = @cos(theta);
                            const sin_t = @sin(theta);
                            const dy0 = dy_row[rotary_offset + 2 * i];
                            const dy1 = dy_row[rotary_offset + 2 * i + 1];
                            dx_row[rotary_offset + 2 * i] += dy0 * cos_t + dy1 * sin_t;
                            dx_row[rotary_offset + 2 * i + 1] += -dy0 * sin_t + dy1 * cos_t;
                        }
                        if (2 * half < rot_dim) {
                            dx_row[D - 1] += dy_row[D - 1];
                        }
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
                    var coord = [_]usize{0} ** 8;
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
                        var d = rank;
                        while (d > 0) {
                            d -= 1;
                            coord[d] += 1;
                            if (coord[d] < X.shape.dims[d]) break;
                            coord[d] = 0;
                        }
                    }
                }
            }
        },
        else => unreachable,
    }
}
