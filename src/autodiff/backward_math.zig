const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const Op = @import("types.zig").Op;

/// 求和 / 均值归约 (Sum / Mean Reduction) 的反向传播：dA[x] += scale * dC[c(x)]。
/// 输出梯度 dC (连续存储) 以广播跨度映射回 A 的形状：被归约维度跨度为 0 (全局归约时全部为 0)，
/// 其余维度沿用 dC 的连续跨度 (keepdims 只多出一个大小为 1 的维度，不改变这些跨度)。
/// 随后用张量层的逐行跨步遍历器 (`shape.RowIter`，内部先做维度折叠 `coalesceDims`) 同时遍历
/// A 的梯度 (可以是任意跨步视图) 与 dC，最内层是一维循环，不再逐元素计算多维平铺下标。
/// 每个元素的梯度互不依赖，因此遍历前先把维度按 A 的跨度从大到小重排，使最内层沿 A 的内存连续方向
/// (例如转置视图按存储顺序而不是逻辑行顺序写入梯度)。
pub fn reduceSumMeanBackward(A: *Tensor, C: *Tensor, axis: ?usize, scale: f32) void {
    if (!A.requires_grad) return;

    var reduced = A.shape;
    if (axis) |ax| {
        reduced.dims[ax] = 1;
    } else {
        @memset(reduced.dims[0..reduced.len], 1);
    }
    var c_strides = tensor_mod.computeContiguousStrides(reduced);
    if (axis) |ax| {
        c_strides.dims[ax] = 0;
    } else {
        @memset(c_strides.dims[0..c_strides.len], 0);
    }

    // 按 A 的跨度从大到小稳定排序维度 (至多 8 维，插入排序)
    var order: [8]usize = undefined;
    for (order[0..A.shape.len], 0..) |*o, d| o.* = d;
    for (1..A.shape.len) |i| {
        var j = i;
        while (j > 0 and A.strides.dims[order[j - 1]] < A.strides.dims[order[j]]) : (j -= 1) {
            std.mem.swap(usize, &order[j - 1], &order[j]);
        }
    }
    var shape = A.shape;
    var a_strides = A.strides;
    var c_perm = c_strides;
    for (order[0..A.shape.len], 0..) |src, d| {
        shape.dims[d] = A.shape.dims[src];
        a_strides.dims[d] = A.strides.dims[src];
        c_perm.dims[d] = c_strides.dims[src];
    }

    var it = tensor_mod.shape.RowIter(2).init(shape, .{ a_strides, c_perm });
    const inner = it.inner;
    const sa = it.inner_strides[0];
    const sc = it.inner_strides[1];
    while (it.next()) |offs| {
        if (sc == 0) {
            // 整行对应同一个输出梯度 (沿最内层归约)
            const g = C.grad[offs[1]] * scale;
            if (sa == 1) {
                for (A.grad[offs[0]..][0..inner]) |*ag| ag.* += g;
            } else {
                for (0..inner) |i| A.grad[offs[0] + i * sa] += g;
            }
        } else if (sa == 1 and sc == 1) {
            for (A.grad[offs[0]..][0..inner], C.grad[offs[1]..][0..inner]) |*ag, cg| ag.* += cg * scale;
        } else {
            for (0..inner) |i| A.grad[offs[0] + i * sa] += C.grad[offs[1] + i * sc] * scale;
        }
    }
}

pub fn backwardMath(self: *Op) !void {
    switch (self.op_type) {
        .L2Loss => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const lambda = self.context.L2Loss.lambda;
            if (A.requires_grad) {
                for (0..A.shape.numel()) |i| {
                    A.grad[i] += C.grad[0] * lambda * A.data[i];
                }
            }
        },
        .L1Loss => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const lambda = self.context.L1Loss.lambda;
            if (A.requires_grad) {
                for (0..A.shape.numel()) |i| {
                    const val = A.data[i];
                    const sign: f32 = if (val > 0.0) 1.0 else if (val < 0.0) -1.0 else 0.0;
                    A.grad[i] += C.grad[0] * lambda * sign;
                }
            }
        },
        .Sqrt => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                if (A.isContiguous() and A.grad.len >= C.grad.len) {
                    for (C.grad, C.data, 0..) |cg, c_val, i| {
                        A.grad[i] += cg / (2.0 * @max(c_val, 1e-12));
                    }
                } else {
                    var coord: [8]usize = @splat(0);
                    const len = A.shape.len;
                    for (C.grad, C.data) |cg, c_val| {
                        var src_idx: usize = 0;
                        for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                        A.grad[src_idx] += cg / (2.0 * @max(c_val, 1e-12));
                        A.shape.incrementCoord(&coord);
                    }
                }
            }
        },
        .Exp => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                if (A.isContiguous() and A.grad.len >= C.grad.len) {
                    for (C.grad, C.data, 0..) |cg, c_val, i| {
                        A.grad[i] += cg * c_val;
                    }
                } else {
                    var coord: [8]usize = @splat(0);
                    const len = A.shape.len;
                    for (C.grad, C.data) |cg, c_val| {
                        var src_idx: usize = 0;
                        for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                        A.grad[src_idx] += cg * c_val;
                        A.shape.incrementCoord(&coord);
                    }
                }
            }
        },
        .Log => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                if (A.isContiguous() and A.grad.len >= C.grad.len) {
                    for (C.grad, A.data[0..C.grad.len], 0..) |cg, a_val, i| {
                        A.grad[i] += cg / a_val;
                    }
                } else {
                    var coord: [8]usize = @splat(0);
                    const len = A.shape.len;
                    for (C.grad) |cg| {
                        var src_idx: usize = 0;
                        for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                        A.grad[src_idx] += cg / A.data[src_idx];
                        A.shape.incrementCoord(&coord);
                    }
                }
            }
        },
        .Abs => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            if (A.requires_grad) {
                if (A.isContiguous() and A.grad.len >= C.grad.len) {
                    for (C.grad, A.data[0..C.grad.len], 0..) |cg, a_val, i| {
                        const sign: f32 = if (a_val > 0.0) 1.0 else if (a_val < 0.0) -1.0 else 0.0;
                        A.grad[i] += cg * sign;
                    }
                } else {
                    var coord: [8]usize = @splat(0);
                    const len = A.shape.len;
                    for (C.grad) |cg| {
                        var src_idx: usize = 0;
                        for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                        const a_val = A.data[src_idx];
                        const sign: f32 = if (a_val > 0.0) 1.0 else if (a_val < 0.0) -1.0 else 0.0;
                        A.grad[src_idx] += cg * sign;
                        A.shape.incrementCoord(&coord);
                    }
                }
            }
        },
        .Sum => {
            const ctx = self.context.Sum;
            reduceSumMeanBackward(self.inputs[0], self.outputs[0], ctx.axis, 1.0);
        },
        .Mean => {
            const A = self.inputs[0];
            const ctx = self.context.Mean;
            const count = if (ctx.axis) |ax| @as(f32, @floatFromInt(A.shape.dims[ax])) else @as(f32, @floatFromInt(A.shape.numel()));
            reduceSumMeanBackward(A, self.outputs[0], ctx.axis, 1.0 / count);
        },
        .Where => {
            const X = self.inputs[0];
            const Y = self.inputs[1];
            const C = self.outputs[0];
            const mask = self.context.Where.mask;
            if (X.requires_grad or Y.requires_grad) {
                const x_strides = tensor_mod.computeBroadcastStrides(X.shape, X.strides, C.shape);
                const y_strides = tensor_mod.computeBroadcastStrides(Y.shape, Y.strides, C.shape);
                const rank = C.shape.len;
                var indices: [8]usize = @splat(0);
                for (0..C.grad.len) |c_flat| {
                    const g = C.grad[c_flat];
                    if (mask[c_flat]) {
                        if (X.requires_grad) {
                            var x_flat: usize = 0;
                            for (0..rank) |d| x_flat += indices[d] * x_strides.dims[d];
                            X.grad[x_flat] += g;
                        }
                    } else {
                        if (Y.requires_grad) {
                            var y_flat: usize = 0;
                            for (0..rank) |d| y_flat += indices[d] * y_strides.dims[d];
                            Y.grad[y_flat] += g;
                        }
                    }
                    C.shape.incrementCoord(&indices);
                }
            }
        },
        .MaskedFill => {
            const X = self.inputs[0];
            const C = self.outputs[0];
            const ctx = self.context.MaskedFill;
            if (X.requires_grad) {
                if (X.isContiguous() and X.grad.len >= C.grad.len) {
                    for (C.grad, ctx.mask, 0..) |g, m, i| {
                        if (!m) X.grad[i] += g;
                    }
                } else {
                    var coord: [8]usize = @splat(0);
                    const len = X.shape.len;
                    for (C.grad, ctx.mask) |g, m| {
                        if (!m) {
                            var src_idx: usize = 0;
                            for (0..len) |d| src_idx += coord[d] * X.strides.dims[d];
                            X.grad[src_idx] += g;
                        }
                        X.shape.incrementCoord(&coord);
                    }
                }
            }
        },
        .Slice => {
            const X = self.inputs[0];
            const C = self.outputs[0];
            const ctx = self.context.Slice;
            if (X.requires_grad) {
                var coord: [8]usize = @splat(0);
                for (0..C.grad.len) |dest_i| {
                    var src_idx: usize = ctx.offset;
                    for (0..ctx.rank) |d| src_idx += coord[d] * ctx.strides[d];
                    X.grad[src_idx] += C.grad[dest_i];
                    C.shape.incrementCoord(&coord);
                }
            }
        },
        else => unreachable,
    }
}
