const std = @import("std");
const tensor = @import("../tensor.zig");
const transposeShape = tensor.transposeShape;
const Op = @import("types.zig").Op;

const backend = @import("../backend.zig");
const c = backend.cblas;
const vec_math = backend.vec_math;

/// 二维矩阵在基础线性代数子程序 (Basic Linear Algebra Subprograms, BLAS) 中的存储描述：
/// 逻辑元素 `X[i][j]` 位于 `ptr[i * ld + j]` (行主序，`trans = false`) 或 `ptr[j * ld + i]` (转置存储，`trans = true`)
const MatLayout = struct {
    ld: usize,
    trans: bool,

    /// 同一块存储解释为逻辑转置矩阵时的描述
    fn transposed(self: MatLayout) MatLayout {
        return .{ .ld = self.ld, .trans = !self.trans };
    }
};

/// 由二维张量的形状与跨步推导 BLAS 存储描述；跨步无法表达为行主序或转置存储 (如带步长的切片) 时返回 null
fn matLayout(t: *const tensor.Tensor) ?MatLayout {
    const rows = t.shape.dims[0];
    const cols = t.shape.dims[1];
    const s0 = t.strides.dims[0];
    const s1 = t.strides.dims[1];
    if ((cols <= 1 or s1 == 1) and (rows <= 1 or s0 >= cols)) {
        return .{ .ld = if (rows <= 1) @max(cols, 1) else s0, .trans = false };
    }
    if ((rows <= 1 or s0 == 1) and (cols <= 1 or s1 >= rows)) {
        return .{ .ld = if (cols <= 1) @max(rows, 1) else s1, .trans = true };
    }
    return null;
}

fn blasOp(trans: bool) c_int {
    return if (trans) c.CblasTrans else c.CblasNoTrans;
}

/// 逻辑矩阵乘加 `C[m x n] += A[m x k] * B[k x n]`，三个操作数各自可以是行主序或转置存储
fn gemmAccumulate(
    m: usize,
    n: usize,
    k: usize,
    a_ptr: [*]const f32,
    a: MatLayout,
    b_ptr: [*]const f32,
    b: MatLayout,
    c_ptr: [*]f32,
    cl: MatLayout,
) void {
    if (m == 0 or n == 0) return;
    if (!cl.trans) {
        c.cblas_sgemm(c.CblasRowMajor, blasOp(a.trans), blasOp(b.trans), @intCast(m), @intCast(n), @intCast(k), 1.0, a_ptr, @intCast(a.ld), b_ptr, @intCast(b.ld), 1.0, c_ptr, @intCast(cl.ld));
    } else {
        // 输出为转置存储时改算 C^T += B^T * A^T
        c.cblas_sgemm(c.CblasRowMajor, blasOp(!b.trans), blasOp(!a.trans), @intCast(n), @intCast(m), @intCast(k), 1.0, b_ptr, @intCast(b.ld), a_ptr, @intCast(a.ld), 1.0, c_ptr, @intCast(cl.ld));
    }
}

/// 按输出张量最内层维度逐行遍历 (行优先)，同时维护两个操作数 (广播或转置跨步) 的行起始偏移：
/// 每个元素的多维坐标换算从 O(rank) 降为每行一次，内层循环只剩固定跨步访问
const RowWalker = struct {
    shape: tensor.Shape,
    a_strides: tensor.Shape,
    b_strides: tensor.Shape,
    coord: [8]usize = @as([8]usize, @splat(0)),
    a_off: usize = 0,
    b_off: usize = 0,

    /// 最内层维度长度 (标量形状视为长度 1)
    fn inner(self: RowWalker) usize {
        return if (self.shape.len == 0) 1 else self.shape.dims[self.shape.len - 1];
    }

    fn aInnerStride(self: RowWalker) usize {
        return if (self.shape.len == 0) 0 else self.a_strides.dims[self.shape.len - 1];
    }

    fn bInnerStride(self: RowWalker) usize {
        return if (self.shape.len == 0) 0 else self.b_strides.dims[self.shape.len - 1];
    }

    /// 行数 = 元素总数 / 最内层维度长度
    fn rows(self: RowWalker, numel: usize) usize {
        const n = self.inner();
        return if (n == 0) 0 else numel / n;
    }

    /// 前进到下一行：递增除最内层外的坐标并增量更新两个操作数的行偏移
    fn next(self: *RowWalker) void {
        if (self.shape.len < 2) return;
        var d = self.shape.len - 1;
        while (d > 0) {
            d -= 1;
            self.coord[d] += 1;
            self.a_off += self.a_strides.dims[d];
            self.b_off += self.b_strides.dims[d];
            if (self.coord[d] < self.shape.dims[d]) return;
            self.a_off -= self.coord[d] * self.a_strides.dims[d];
            self.b_off -= self.coord[d] * self.b_strides.dims[d];
            self.coord[d] = 0;
        }
    }
};

/// 把一行上游梯度 `g` 乘以 `scale` 后累加到 `dst[off + j * stride]`：跨步为 0 (广播轴) 时先求和再写一次
fn accumulateRow(dst: []f32, off: usize, stride: usize, g: []const f32, comptime scale: f32) void {
    switch (stride) {
        0 => dst[off] += scale * vec_math.sum(g),
        1 => for (dst[off..][0..g.len], g) |*d, gv| {
            d.* += scale * gv;
        },
        else => for (g, 0..) |gv, j| {
            dst[off + j * stride] += scale * gv;
        },
    }
}

/// 逐元素乘法梯度的一行：`dst[off + j * stride] += g[j] * other[o_off + j * o_stride]`
fn mulAccumulateRow(dst: []f32, off: usize, stride: usize, g: []const f32, other: []const f32, o_off: usize, o_stride: usize) void {
    if (stride == 0) {
        var acc: f32 = 0.0;
        if (o_stride == 1) {
            for (g, other[o_off..][0..g.len]) |gv, ov| acc += gv * ov;
        } else {
            for (g, 0..) |gv, j| acc += gv * other[o_off + j * o_stride];
        }
        dst[off] += acc;
    } else if (stride == 1 and o_stride == 1) {
        for (dst[off..][0..g.len], g, other[o_off..][0..g.len]) |*d, gv, ov| d.* += gv * ov;
    } else if (stride == 1 and o_stride == 0) {
        const ov = other[o_off];
        for (dst[off..][0..g.len], g) |*d, gv| d.* += gv * ov;
    } else {
        for (g, 0..) |gv, j| dst[off + j * stride] += gv * other[o_off + j * o_stride];
    }
}

pub fn backwardCore(self: *Op) !void {
    switch (self.op_type) {
        // ====================================================================
        // 1. 矩阵乘法反向传播 (MatMul Backward)
        // ====================================================================
        .MatMul => {
            const A = self.inputs[0]; // 形状为 M x K
            const B = self.inputs[1]; // 形状为 K x N
            const C = self.outputs[0]; // 形状为 M x N
            const M = A.shape.dims[0];
            const K = A.shape.dims[1];
            const N = B.shape.dims[1];

            // 梯度缓冲区与数据缓冲区共用同一跨步布局，因此转置视图等非连续操作数也能直接交给 BLAS
            const a_layout = matLayout(A);
            const b_layout = matLayout(B);
            const c_layout = matLayout(C);
            if (a_layout != null and b_layout != null and c_layout != null) {
                const al = a_layout.?;
                const bl = b_layout.?;
                const cl = c_layout.?;
                // 1. 左乘矩阵 A 的梯度: dA += dC * B^T (累加到 A.grad，不覆盖已有值)
                if (A.requires_grad) {
                    gemmAccumulate(M, K, N, C.grad.ptr, cl, B.data.ptr, bl.transposed(), A.grad.ptr, al);
                }
                // 2. 右乘矩阵 B 的梯度: dB += A^T * dC
                if (B.requires_grad) {
                    gemmAccumulate(K, N, M, A.data.ptr, al.transposed(), C.grad.ptr, cl, B.grad.ptr, bl);
                }
            } else {
                // 任意跨步 (如带步长的切片视图) 的通用回退路径
                const sa0 = A.strides.dims[0];
                const sa1 = A.strides.dims[1];
                const sb0 = B.strides.dims[0];
                const sb1 = B.strides.dims[1];
                const sc0 = C.strides.dims[0];
                const sc1 = C.strides.dims[1];
                for (0..M) |i| {
                    for (0..N) |j| {
                        const dc = C.grad[i * sc0 + j * sc1];
                        if (dc == 0.0) continue;
                        for (0..K) |k| {
                            if (A.requires_grad) A.grad[i * sa0 + k * sa1] += dc * B.data[k * sb0 + j * sb1];
                            if (B.requires_grad) B.grad[k * sb0 + j * sb1] += A.data[i * sa0 + k * sa1] * dc;
                        }
                    }
                }
            }
        },
        // ====================================================================
        // 2. 修正线性单元 (Rectified Linear Unit, ReLU) 激活函数反向传播 (ReLU Backward)
        // ====================================================================
        .Relu => {
            const A = self.inputs[0];
            const C = self.outputs[0];

            if (A.requires_grad) {
                const total = A.shape.numel();
                for (0..total) |i| {
                    A.grad[i] += if (A.data[i] > 0.0) C.grad[i] else 0.0;
                }
            }
        },
        // ====================================================================
        // 3.5. 高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 激活函数反向传播 (GELU Backward，精确形式与双曲正切近似形式)
        // ====================================================================
        .Gelu, .GeluTanh => {
            const A = self.inputs[0];
            const C = self.outputs[0];

            if (A.requires_grad) {
                const total = A.shape.numel();
                switch (self.op_type.geluApproximation().?) {
                    inline else => |approx| for (A.grad[0..total], C.grad[0..total], A.data[0..total]) |*a_g, c_g, x| {
                        a_g.* += c_g * backend.activation.geluDerivative(approx, x);
                    },
                }
            }
        },
        .Sigmoid => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                for (A.grad, C.grad, C.data) |*a_g, c_g, c_val| {
                    a_g.* += c_g * c_val * (1.0 - c_val);
                }
            }
        },
        .Tanh => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                for (A.grad, C.grad, C.data) |*a_g, c_g, c_val| {
                    a_g.* += c_g * (1.0 - c_val * c_val);
                }
            }
        },
        .Softcap => {
            // y = cap * tanh(x / cap)，dy/dx = 1 - tanh^2(x / cap)
            const A = self.inputs[0];
            const C = self.outputs[0];
            const cap = self.context.Softcap.cap;
            if (A.requires_grad) {
                for (A.grad, C.grad, A.data) |*a_g, c_g, a_val| {
                    const t = std.math.tanh(a_val / cap);
                    a_g.* += c_g * (1.0 - t * t);
                }
            }
        },
        .LeakyRelu => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const alpha = self.context.LeakyRelu.alpha;
            if (A.requires_grad) {
                for (A.grad, C.grad, A.data) |*a_g, c_g, a_val| {
                    const slope: f32 = if (a_val > 0.0) 1.0 else alpha;
                    a_g.* += c_g * slope;
                }
            }
        },
        .Silu => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                for (A.grad, C.grad, A.data) |*a_g, c_g, a_val| {
                    const sig = vec_math.sigmoid(a_val);
                    const d_silu = sig * (1.0 + a_val * (1.0 - sig));
                    a_g.* += c_g * d_silu;
                }
            }
        },
        // ====================================================================
        // 4. Softmax + Cross Entropy 损失函数反向传播 (SoftmaxCrossEntropy Backward)
        // ====================================================================
        .SoftmaxCrossEntropy => {
            const logits = self.inputs[0];
            const loss = self.outputs[0];
            const dL = loss.grad[0];
            const M = logits.shape.dims[0];
            const N = logits.shape.dims[1];
            const ctx = &self.context.SoftmaxCrossEntropy;

            if (logits.requires_grad) {
                const denom = if (ctx.mask != null) ctx.total_weight else @as(f32, @floatFromInt(M));
                if (denom > 0.0) {
                    for (0..M) |i| {
                        const w: f32 = if (ctx.mask) |m| m[i] else 1.0;
                        if (w <= 0.0) continue;
                        const scale = dL * (w / denom);
                        const label = ctx.targets[i];
                        const p_row = ctx.probs[i * N .. (i + 1) * N];
                        const dLogits_row = logits.grad[i * N .. (i + 1) * N];
                        tensor.losses.accumulateCeGradRow(p_row, label, dLogits_row, scale);
                    }
                }
            }
        },
        .DpoLoss => {
            const pi_chosen = self.inputs[0];
            const pi_rejected = self.inputs[1];
            const loss = self.outputs[0];
            const dL = loss.grad[0];
            const ctx = self.context.DpoLoss;
            const N = pi_chosen.data.len;
            if (N > 0) {
                const inv_n = dL / @as(f32, @floatFromInt(N));
                for (0..N) |i| {
                    var dz: f32 = undefined;
                    _ = tensor.losses.dpoTerm(
                        pi_chosen.data[i],
                        pi_rejected.data[i],
                        ctx.ref_chosen[i],
                        ctx.ref_rejected[i],
                        ctx.beta,
                        &dz,
                    );
                    const scaled_dz = inv_n * dz;
                    if (pi_chosen.requires_grad) {
                        pi_chosen.grad[i] += scaled_dz;
                    }
                    if (pi_rejected.requires_grad) {
                        pi_rejected.grad[i] -= scaled_dz;
                    }
                }
            }
        },
        .GrpoLoss => {
            const old_logps = self.inputs[0];
            const new_logps = self.inputs[1];
            const loss = self.outputs[0];
            const dL = loss.grad[0];
            const ctx = self.context.GrpoLoss;
            const N = old_logps.data.len;
            if (N > 0 and new_logps.requires_grad) {
                const inv_n = dL / @as(f32, @floatFromInt(N));
                for (0..N) |i| {
                    const ref = if (ctx.ref_logps) |refs| refs[i] else null;
                    var d_obj: f32 = undefined;
                    _ = tensor.losses.grpoTerm(
                        new_logps.data[i],
                        old_logps.data[i],
                        ctx.advantages[i],
                        ref,
                        ctx.beta,
                        ctx.clip_eps,
                        &d_obj,
                    );
                    new_logps.grad[i] += -inv_n * d_obj;
                }
            }
        },
        .BceWithLogitsLoss => {
            const logits = self.inputs[0];
            const targets = self.inputs[1];
            const loss = self.outputs[0];
            const dL = loss.grad[0];
            const N = logits.data.len;
            const scale = dL / @as(f32, @floatFromInt(N));

            if (logits.requires_grad) {
                for (logits.grad, logits.data, targets.data) |*x_g, x_val, y_val| {
                    const sig = vec_math.sigmoid(x_val);
                    x_g.* += scale * (sig - y_val);
                }
            }
            if (targets.requires_grad) {
                for (targets.grad, logits.data) |*y_g, x_val| {
                    y_g.* += scale * (-x_val);
                }
            }
        },
        .BceLoss => {
            const probs = self.inputs[0];
            const targets = self.inputs[1];
            const loss = self.outputs[0];
            const eps = self.context.BceLoss.eps;
            const dL = loss.grad[0];
            const N = probs.data.len;
            const scale = dL / @as(f32, @floatFromInt(N));

            if (probs.requires_grad) {
                for (probs.grad, probs.data, targets.data) |*p_g, p_val, y_val| {
                    const p_clip = @max(p_val, eps);
                    const one_minus_p_clip = @max(1.0 - p_val, eps);
                    const grad_p = (p_val - y_val) / (p_clip * one_minus_p_clip);
                    p_g.* += scale * grad_p;
                }
            }
            if (targets.requires_grad) {
                for (targets.grad, probs.data, targets.data) |*y_g, p_val, _| {
                    const p_clip = @max(p_val, eps);
                    const one_minus_p_clip = @max(1.0 - p_val, eps);
                    const grad_y = -@log(p_clip) + @log(one_minus_p_clip);
                    y_g.* += scale * grad_y;
                }
            }
        },
        // ====================================================================
        // 5. 形状变换反向传播 (Reshape Backward)
        // ====================================================================
        .Reshape => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                if (A.isContiguous() and A.grad.len >= C.grad.len) {
                    for (C.grad, 0..) |g, i| {
                        A.grad[i] += g;
                    }
                } else {
                    var coord: [8]usize = @splat(0);
                    const len = A.shape.len;
                    for (C.grad) |g| {
                        var src_idx: usize = 0;
                        for (0..len) |d| {
                            src_idx += coord[d] * A.strides.dims[d];
                        }
                        A.grad[src_idx] += g;

                        A.shape.incrementCoord(&coord);
                    }
                }
            }
        },
        // ====================================================================
        // 6. 维度转置反向传播 (Transpose Backward)
        // ====================================================================
        .Transpose => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                const ctx = self.context.Transpose;
                var walker = RowWalker{
                    .shape = C.shape,
                    .a_strides = transposeShape(A.strides, ctx.dim0, ctx.dim1),
                    .b_strides = C.strides,
                };
                const inner = walker.inner();
                const a_inner = walker.aInnerStride();
                var c_off: usize = 0;
                for (0..walker.rows(C.grad.len)) |_| {
                    accumulateRow(A.grad, walker.a_off, a_inner, C.grad[c_off..][0..inner], 1.0);
                    c_off += inner;
                    walker.next();
                }
            }
        },
        .Concat => {
            const dim = self.context.Concat.dim;
            const out = self.outputs[0];
            const rank = out.shape.len;
            const concat_dim_total = out.shape.dims[dim];

            var outer_size: usize = 1;
            for (0..dim) |d| {
                outer_size *= out.shape.dims[d];
            }
            var inner_size: usize = 1;
            for (dim + 1..rank) |d| {
                inner_size *= out.shape.dims[d];
            }

            for (0..outer_size) |outer| {
                const out_base = outer * concat_dim_total * inner_size;
                var offset_dim: usize = 0;
                for (self.inputs) |t| {
                    const d_k = t.shape.dims[dim];
                    const src_base = outer * d_k * inner_size;
                    const dest_base = out_base + offset_dim * inner_size;
                    const copy_len = d_k * inner_size;

                    if (t.requires_grad) {
                        for (0..copy_len) |j| {
                            t.grad[src_base + j] += out.grad[dest_base + j];
                        }
                    }
                    offset_dim += d_k;
                }
            }
        },
        .Split => {
            const dim = self.context.Split.dim;
            const in = self.inputs[0];
            const rank = in.shape.len;
            const dim_size = in.shape.dims[dim];
            const num_splits = self.outputs.len;
            const split_dim_size = self.outputs[0].shape.dims[dim];

            if (in.requires_grad) {
                var outer_size: usize = 1;
                for (0..dim) |d| {
                    outer_size *= in.shape.dims[d];
                }
                var inner_size: usize = 1;
                for (dim + 1..rank) |d| {
                    inner_size *= in.shape.dims[d];
                }

                for (0..outer_size) |outer| {
                    const src_base = outer * dim_size * inner_size;
                    for (0..num_splits) |k| {
                        if (self.outputs[k].grad.len == 0) continue;
                        const dest_base = outer * split_dim_size * inner_size;
                        const src_offset = src_base + k * split_dim_size * inner_size;
                        const copy_len = split_dim_size * inner_size;
                        for (0..copy_len) |j| {
                            in.grad[src_offset + j] += self.outputs[k].grad[dest_base + j];
                        }
                    }
                }
            }
        },
        .RepeatKV => {
            const groups = self.context.RepeatKV.groups;
            const X = self.inputs[0];
            const Y = self.outputs[0];
            if (X.requires_grad) {
                const B = X.shape.dims[0];
                const n_kv = X.shape.dims[1];
                const T = X.shape.dims[2];
                const hs = X.shape.dims[3];
                const head_bytes = T * hs;

                for (0..B) |b| {
                    for (0..n_kv) |kv_h| {
                        const x_grad = X.grad[((b * n_kv + kv_h) * head_bytes)..((b * n_kv + kv_h + 1) * head_bytes)];
                        for (0..groups) |g| {
                            const h = kv_h * groups + g;
                            const y_grad = Y.grad[((b * (n_kv * groups) + h) * head_bytes)..((b * (n_kv * groups) + h + 1) * head_bytes)];
                            for (x_grad, y_grad) |*xg, yg| {
                                xg.* += yg;
                            }
                        }
                    }
                }
            }
        },
        .MseLoss => {
            const A = self.inputs[0]; // y_pred
            const B = self.inputs[1]; // y_true
            const C = self.outputs[0]; // loss
            const N = A.shape.numel();
            const N_f = @as(f32, @floatFromInt(N));

            if (A.requires_grad) {
                for (0..N) |i| {
                    A.grad[i] += C.grad[0] * (2.0 / N_f) * (A.data[i] - B.data[i]);
                }
            }
            if (B.requires_grad) {
                for (0..N) |i| {
                    B.grad[i] += C.grad[0] * (2.0 / N_f) * (B.data[i] - A.data[i]);
                }
            }
        },
        .MulScalar => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const val = self.context.MulScalar.val;
            if (A.requires_grad) {
                for (0..A.shape.numel()) |i| {
                    A.grad[i] += C.grad[i] * val;
                }
            }
        },
        .DivScalar => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const val = self.context.DivScalar.val;
            if (A.requires_grad) {
                for (0..A.shape.numel()) |i| {
                    A.grad[i] += C.grad[i] / val;
                }
            }
        },
        .AddScalar => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                for (0..A.shape.numel()) |i| {
                    A.grad[i] += C.grad[i];
                }
            }
        },
        .SubScalar => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                for (0..A.shape.numel()) |i| {
                    A.grad[i] += C.grad[i];
                }
            }
        },
        .Add => {
            const A = self.inputs[0];
            const B = self.inputs[1];
            const C = self.outputs[0];

            if (A.shape.eq(B.shape) and A.isContiguous() and B.isContiguous() and
                (!A.requires_grad or A.grad.len >= C.grad.len) and
                (!B.requires_grad or B.grad.len >= C.grad.len))
            {
                if (A.requires_grad) {
                    for (A.grad[0..C.grad.len], C.grad) |*a_g, c_g| {
                        a_g.* += c_g;
                    }
                }
                if (B.requires_grad) {
                    for (B.grad[0..C.grad.len], C.grad) |*b_g, c_g| {
                        b_g.* += c_g;
                    }
                }
            } else {
                var walker = RowWalker{
                    .shape = C.shape,
                    .a_strides = tensor.computeBroadcastStrides(A.shape, A.strides, C.shape),
                    .b_strides = tensor.computeBroadcastStrides(B.shape, B.strides, C.shape),
                };
                const inner = walker.inner();
                const a_inner = walker.aInnerStride();
                const b_inner = walker.bInnerStride();
                var c_off: usize = 0;
                for (0..walker.rows(C.grad.len)) |_| {
                    const g_row = C.grad[c_off..][0..inner];
                    if (A.requires_grad) accumulateRow(A.grad, walker.a_off, a_inner, g_row, 1.0);
                    if (B.requires_grad) accumulateRow(B.grad, walker.b_off, b_inner, g_row, 1.0);
                    c_off += inner;
                    walker.next();
                }
            }
        },
        .Sub => {
            const A = self.inputs[0];
            const B = self.inputs[1];
            const C = self.outputs[0];

            if (A.shape.eq(B.shape) and A.isContiguous() and B.isContiguous() and
                (!A.requires_grad or A.grad.len >= C.grad.len) and
                (!B.requires_grad or B.grad.len >= C.grad.len))
            {
                if (A.requires_grad) {
                    for (A.grad[0..C.grad.len], C.grad) |*a_g, c_g| {
                        a_g.* += c_g;
                    }
                }
                if (B.requires_grad) {
                    for (B.grad[0..C.grad.len], C.grad) |*b_g, c_g| {
                        b_g.* -= c_g;
                    }
                }
            } else {
                var walker = RowWalker{
                    .shape = C.shape,
                    .a_strides = tensor.computeBroadcastStrides(A.shape, A.strides, C.shape),
                    .b_strides = tensor.computeBroadcastStrides(B.shape, B.strides, C.shape),
                };
                const inner = walker.inner();
                const a_inner = walker.aInnerStride();
                const b_inner = walker.bInnerStride();
                var c_off: usize = 0;
                for (0..walker.rows(C.grad.len)) |_| {
                    const g_row = C.grad[c_off..][0..inner];
                    if (A.requires_grad) accumulateRow(A.grad, walker.a_off, a_inner, g_row, 1.0);
                    if (B.requires_grad) accumulateRow(B.grad, walker.b_off, b_inner, g_row, -1.0);
                    c_off += inner;
                    walker.next();
                }
            }
        },
        .Mul => {
            const A = self.inputs[0];
            const B = self.inputs[1];
            const C = self.outputs[0];

            if (A.shape.eq(B.shape) and A.isContiguous() and B.isContiguous() and
                A.data.len >= C.grad.len and B.data.len >= C.grad.len and
                (!A.requires_grad or A.grad.len >= C.grad.len) and
                (!B.requires_grad or B.grad.len >= C.grad.len))
            {
                if (A.requires_grad) {
                    for (A.grad[0..C.grad.len], C.grad, B.data[0..C.grad.len]) |*a_g, c_g, b_val| {
                        a_g.* += c_g * b_val;
                    }
                }
                if (B.requires_grad) {
                    for (B.grad[0..C.grad.len], C.grad, A.data[0..C.grad.len]) |*b_g, c_g, a_val| {
                        b_g.* += c_g * a_val;
                    }
                }
            } else {
                var walker = RowWalker{
                    .shape = C.shape,
                    .a_strides = tensor.computeBroadcastStrides(A.shape, A.strides, C.shape),
                    .b_strides = tensor.computeBroadcastStrides(B.shape, B.strides, C.shape),
                };
                const inner = walker.inner();
                const a_inner = walker.aInnerStride();
                const b_inner = walker.bInnerStride();
                var c_off: usize = 0;
                for (0..walker.rows(C.grad.len)) |_| {
                    const g_row = C.grad[c_off..][0..inner];
                    if (A.requires_grad) mulAccumulateRow(A.grad, walker.a_off, a_inner, g_row, B.data, walker.b_off, b_inner);
                    if (B.requires_grad) mulAccumulateRow(B.grad, walker.b_off, b_inner, g_row, A.data, walker.a_off, a_inner);
                    c_off += inner;
                    walker.next();
                }
            }
        },
        .Div => {
            const A = self.inputs[0];
            const B = self.inputs[1];
            const C = self.outputs[0];

            if (A.shape.eq(B.shape) and A.isContiguous() and B.isContiguous() and
                A.data.len >= C.grad.len and B.data.len >= C.grad.len and
                (!A.requires_grad or A.grad.len >= C.grad.len) and
                (!B.requires_grad or B.grad.len >= C.grad.len))
            {
                if (A.requires_grad) {
                    for (A.grad[0..C.grad.len], C.grad, B.data[0..C.grad.len]) |*a_g, c_g, b_val| {
                        a_g.* += c_g / b_val;
                    }
                }
                if (B.requires_grad) {
                    for (B.grad[0..C.grad.len], C.grad, A.data[0..C.grad.len], B.data[0..C.grad.len]) |*b_g, c_g, a_val, b_val| {
                        b_g.* -= c_g * a_val / (b_val * b_val);
                    }
                }
            } else {
                var walker = RowWalker{
                    .shape = C.shape,
                    .a_strides = tensor.computeBroadcastStrides(A.shape, A.strides, C.shape),
                    .b_strides = tensor.computeBroadcastStrides(B.shape, B.strides, C.shape),
                };
                const inner = walker.inner();
                const a_inner = walker.aInnerStride();
                const b_inner = walker.bInnerStride();
                var c_off: usize = 0;
                for (0..walker.rows(C.grad.len)) |_| {
                    const g_row = C.grad[c_off..][0..inner];
                    for (g_row, 0..) |c_g, j| {
                        const a_idx = walker.a_off + j * a_inner;
                        const b_idx = walker.b_off + j * b_inner;
                        const b_val = B.data[b_idx];
                        if (A.requires_grad) A.grad[a_idx] += c_g / b_val;
                        if (B.requires_grad) B.grad[b_idx] -= c_g * A.data[a_idx] / (b_val * b_val);
                    }
                    c_off += inner;
                    walker.next();
                }
            }
        },
        else => unreachable,
    }
}
