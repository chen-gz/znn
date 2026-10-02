const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const op_mod = @import("op.zig");
const Op = op_mod.Op;

pub fn reduceSumMeanBackward(A: *Tensor, C: *Tensor, axis: ?usize, keepdims: bool, scale: f32) void {
    if (!A.requires_grad) return;
    if (axis) |ax| {
        const reduce_size = A.shape.dims[ax];
        var outer_size: usize = 1;
        for (0..ax) |d| outer_size *= A.shape.dims[d];
        var inner_size: usize = 1;
        for ((ax + 1)..A.shape.len) |d| inner_size *= A.shape.dims[d];

        if (A.isContiguous()) {
            for (0..outer_size) |outer| {
                const out_base = outer * inner_size;
                const src_base = outer * reduce_size * inner_size;
                for (0..inner_size) |inner| {
                    const g = C.grad[out_base + inner] * scale;
                    for (0..reduce_size) |k| {
                        A.grad[src_base + k * inner_size + inner] += g;
                    }
                }
            }
        } else {
            const out_rank = C.shape.len;
            var out_indices = [_]usize{0} ** 8;
            for (0..C.grad.len) |out_idx| {
                var tmp = out_idx;
                var d: usize = out_rank;
                while (d > 0) {
                    d -= 1;
                    out_indices[d] = tmp % C.shape.dims[d];
                    tmp /= C.shape.dims[d];
                }

                var src_indices = [_]usize{0} ** 8;
                if (keepdims) {
                    for (0..A.shape.len) |idx_d| {
                        src_indices[idx_d] = out_indices[idx_d];
                    }
                } else {
                    var src_d: usize = 0;
                    for (0..A.shape.len) |idx_d| {
                        if (idx_d == ax) continue;
                        src_indices[idx_d] = out_indices[src_d];
                        src_d += 1;
                    }
                }

                const g = C.grad[out_idx] * scale;
                for (0..reduce_size) |k| {
                    src_indices[ax] = k;
                    A.grad[A.getFlatIndex(src_indices[0..A.shape.len])] += g;
                }
            }
        }
    } else {
        const g = C.grad[0] * scale;
        const elem_count = A.shape.numel();
        if (A.isContiguous() and A.grad.len >= elem_count) {
            for (A.grad[0..elem_count]) |*ag| {
                ag.* += g;
            }
        } else {
            var coord = [_]usize{0} ** 8;
            const len = A.shape.len;
            for (0..elem_count) |_| {
                var src_idx: usize = 0;
                for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                A.grad[src_idx] += g;
                var d = len;
                while (d > 0) {
                    d -= 1;
                    coord[d] += 1;
                    if (coord[d] < A.shape.dims[d]) break;
                    coord[d] = 0;
                }
            }
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
                for (0..A.data.len) |i| {
                    A.grad[i] += C.grad[0] * lambda * A.data[i];
                }
            }
        },
        .L1Loss => {
            const A = self.inputs[0];
            const C = self.outputs[0];
            const lambda = self.context.L1Loss.lambda;
            if (A.requires_grad) {
                for (0..A.data.len) |i| {
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
                    var coord = [_]usize{0} ** 8;
                    const len = A.shape.len;
                    for (C.grad, C.data) |cg, c_val| {
                        var src_idx: usize = 0;
                        for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                        A.grad[src_idx] += cg / (2.0 * @max(c_val, 1e-12));
                        var d = len;
                        while (d > 0) {
                            d -= 1;
                            coord[d] += 1;
                            if (coord[d] < A.shape.dims[d]) break;
                            coord[d] = 0;
                        }
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
                    var coord = [_]usize{0} ** 8;
                    const len = A.shape.len;
                    for (C.grad, C.data) |cg, c_val| {
                        var src_idx: usize = 0;
                        for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                        A.grad[src_idx] += cg * c_val;
                        var d = len;
                        while (d > 0) {
                            d -= 1;
                            coord[d] += 1;
                            if (coord[d] < A.shape.dims[d]) break;
                            coord[d] = 0;
                        }
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
                    var coord = [_]usize{0} ** 8;
                    const len = A.shape.len;
                    for (C.grad) |cg| {
                        var src_idx: usize = 0;
                        for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                        A.grad[src_idx] += cg / A.data[src_idx];
                        var d = len;
                        while (d > 0) {
                            d -= 1;
                            coord[d] += 1;
                            if (coord[d] < A.shape.dims[d]) break;
                            coord[d] = 0;
                        }
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
                    var coord = [_]usize{0} ** 8;
                    const len = A.shape.len;
                    for (C.grad) |cg| {
                        var src_idx: usize = 0;
                        for (0..len) |d| src_idx += coord[d] * A.strides.dims[d];
                        const a_val = A.data[src_idx];
                        const sign: f32 = if (a_val > 0.0) 1.0 else if (a_val < 0.0) -1.0 else 0.0;
                        A.grad[src_idx] += cg * sign;
                        var d = len;
                        while (d > 0) {
                            d -= 1;
                            coord[d] += 1;
                            if (coord[d] < A.shape.dims[d]) break;
                            coord[d] = 0;
                        }
                    }
                }
            }
        },
        .Sum => {
            const ctx = self.context.Sum;
            reduceSumMeanBackward(self.inputs[0], self.outputs[0], ctx.axis, ctx.keepdims, 1.0);
        },
        .Mean => {
            const A = self.inputs[0];
            const ctx = self.context.Mean;
            const count = if (ctx.axis) |ax| @as(f32, @floatFromInt(A.shape.dims[ax])) else @as(f32, @floatFromInt(A.shape.numel()));
            reduceSumMeanBackward(A, self.outputs[0], ctx.axis, ctx.keepdims, 1.0 / count);
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
                var indices = [_]usize{0} ** 8;
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
                    var d = rank;
                    while (d > 0) {
                        d -= 1;
                        indices[d] += 1;
                        if (indices[d] < C.shape.dims[d]) break;
                        indices[d] = 0;
                    }
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
                    var coord = [_]usize{0} ** 8;
                    const len = X.shape.len;
                    for (C.grad, ctx.mask) |g, m| {
                        if (!m) {
                            var src_idx: usize = 0;
                            for (0..len) |d| src_idx += coord[d] * X.strides.dims[d];
                            X.grad[src_idx] += g;
                        }
                        var d = len;
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
        .Slice => {
            const X = self.inputs[0];
            const C = self.outputs[0];
            const ctx = self.context.Slice;
            if (X.requires_grad) {
                var coord = [_]usize{0} ** 8;
                for (0..C.grad.len) |dest_i| {
                    var src_idx: usize = ctx.offset;
                    for (0..ctx.rank) |d| src_idx += coord[d] * ctx.strides[d];
                    X.grad[src_idx] += C.grad[dest_i];
                    var d = ctx.rank;
                    while (d > 0) {
                        d -= 1;
                        coord[d] += 1;
                        if (coord[d] < C.shape.dims[d]) break;
                        coord[d] = 0;
                    }
                }
            }
        },
        else => unreachable,
    }
}
