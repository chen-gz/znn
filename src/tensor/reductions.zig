const std = @import("std");
const core = @import("core.zig");
const Tensor = core.Tensor;
const shape_mod = @import("shape.zig");
const Shape = shape_mod.Shape;
const isContiguousStrides = shape_mod.isContiguousStrides;
const broadcastShapes = shape_mod.broadcastShapes;
const computeBroadcastStrides = shape_mod.computeBroadcastStrides;
const types_mod = @import("types.zig");
const SliceRange = types_mod.SliceRange;
const GenericTensor = types_mod.GenericTensor;
const isTruthyScalar = types_mod.isTruthyScalar;
const ops_mod = @import("ops.zig");
const zeros = ops_mod.zeros;
const free = ops_mod.free;

pub fn clone(self: Tensor, allocator: std.mem.Allocator) !*Tensor {
    const t = try allocator.create(Tensor);
    t.* = Tensor{
        .data = try allocator.alloc(f32, self.data.len),
        .grad = if (self.requires_grad) try allocator.alloc(f32, self.grad.len) else &.{},
        .shape = self.shape,
        .strides = self.strides,
        .requires_grad = self.requires_grad,
        .creator = self.creator,
    };
    @memcpy(t.data, self.data);
    if (self.requires_grad) {
        @memcpy(t.grad, self.grad);
    }
    return t;
}

pub fn mulScalar_(self: *Tensor, val: f32) *Tensor {
    std.debug.assert(!self.requires_grad);
    std.debug.assert(self.creator == null);
    for (self.data) |*item| {
        item.* *= val;
    }
    return self;
}

pub fn addScalar_(self: *Tensor, val: f32) *Tensor {
    std.debug.assert(!self.requires_grad);
    std.debug.assert(self.creator == null);
    for (self.data) |*item| {
        item.* += val;
    }
    return self;
}

pub fn add_(self: *Tensor, other: *Tensor) !*Tensor {
    std.debug.assert(!self.requires_grad);
    std.debug.assert(self.creator == null);
    std.debug.assert(self.data.len == other.data.len);
    for (self.data, other.data) |*item, other_val| {
        item.* += other_val;
    }
    return self;
}

pub fn argmax(self: Tensor, dim: usize, allocator: std.mem.Allocator) !*Tensor {
    if (dim >= self.shape.len) return error.DimensionOutOfBounds;
    if (self.shape.len != 2) return error.UnsupportedDimension;
    const M = self.shape.dims[0];
    const N = self.shape.dims[1];

    if (dim == 1) {
        const C = try zeros(allocator, &.{ M, 1 });
        for (0..M) |i| {
            var max_val = self.get(&.{ i, 0 });
            var max_idx: usize = 0;
            for (1..N) |j| {
                const val = self.get(&.{ i, j });
                if (val > max_val) {
                    max_val = val;
                    max_idx = j;
                }
            }
            C.data[i] = @as(f32, @floatFromInt(max_idx));
        }
        return C;
    } else if (dim == 0) {
        const C = try zeros(allocator, &.{ 1, N });
        for (0..N) |j| {
            var max_val = self.get(&.{ 0, j });
            var max_idx: usize = 0;
            for (1..M) |i| {
                const val = self.get(&.{ i, j });
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }
            C.data[j] = @as(f32, @floatFromInt(max_idx));
        }
        return C;
    } else {
        return error.UnsupportedDimension;
    }
}

pub fn max(self: Tensor, dim: usize, allocator: std.mem.Allocator) !*Tensor {
    if (dim >= self.shape.len) return error.DimensionOutOfBounds;
    if (self.shape.len != 2) return error.UnsupportedDimension;
    const M = self.shape.dims[0];
    const N = self.shape.dims[1];

    if (dim == 1) {
        const C = try zeros(allocator, &.{ M, 1 });
        for (0..M) |i| {
            var max_val = self.get(&.{ i, 0 });
            for (1..N) |j| {
                const val = self.get(&.{ i, j });
                if (val > max_val) max_val = val;
            }
            C.data[i] = max_val;
        }
        return C;
    } else if (dim == 0) {
        const C = try zeros(allocator, &.{ 1, N });
        for (0..N) |j| {
            var max_val = self.get(&.{ 0, j });
            for (1..M) |i| {
                const val = self.get(&.{ i, j });
                if (val > max_val) max_val = val;
            }
            C.data[j] = max_val;
        }
        return C;
    } else {
        return error.UnsupportedDimension;
    }
}

pub fn isContiguous(self: Tensor) bool {
    return isContiguousStrides(self.shape, self.strides);
}

pub fn numel(self: Tensor) usize {
    return self.shape.numel();
}

/// 通用多维张量沿指定轴或全局求和归约 (Sum Reduction)
pub fn sum(self: *Tensor, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor {
    if (axis) |ax| {
        if (ax >= self.shape.len) return error.DimensionOutOfBounds;
        const reduce_size = self.shape.dims[ax];

        var out_shape_dims = [_]usize{0} ** 8;
        var out_rank: usize = 0;

        if (keepdims) {
            out_rank = self.shape.len;
            for (0..self.shape.len) |d| {
                out_shape_dims[d] = if (d == ax) 1 else self.shape.dims[d];
            }
        } else {
            if (self.shape.len == 1) {
                out_rank = 1;
                out_shape_dims[0] = 1;
            } else {
                out_rank = self.shape.len - 1;
                var dest_d: usize = 0;
                for (0..self.shape.len) |d| {
                    if (d != ax) {
                        out_shape_dims[dest_d] = self.shape.dims[d];
                        dest_d += 1;
                    }
                }
            }
        }

        const out_shape = try Shape.fromSlice(out_shape_dims[0..out_rank]);
        const C = try zeros(allocator, out_shape.dims[0..out_rank]);

        var outer_size: usize = 1;
        for (0..ax) |d| {
            outer_size *= self.shape.dims[d];
        }
        var inner_size: usize = 1;
        for ((ax + 1)..self.shape.len) |d| {
            inner_size *= self.shape.dims[d];
        }

        if (self.isContiguous()) {
            for (0..outer_size) |outer| {
                const out_base = outer * inner_size;
                const src_base = outer * reduce_size * inner_size;
                for (0..inner_size) |inner| {
                    var acc: f32 = 0.0;
                    for (0..reduce_size) |k| {
                        acc += self.data[src_base + k * inner_size + inner];
                    }
                    C.data[out_base + inner] = acc;
                }
            }
        } else {
            var out_indices = [_]usize{0} ** 8;
            for (0..C.data.len) |out_idx| {
                var tmp = out_idx;
                var d: usize = out_rank;
                while (d > 0) {
                    d -= 1;
                    out_indices[d] = tmp % out_shape.dims[d];
                    tmp /= out_shape.dims[d];
                }

                var src_indices = [_]usize{0} ** 8;
                if (keepdims) {
                    for (0..self.shape.len) |idx_d| {
                        src_indices[idx_d] = out_indices[idx_d];
                    }
                } else {
                    var src_d: usize = 0;
                    for (0..self.shape.len) |idx_d| {
                        if (idx_d == ax) continue;
                        src_indices[idx_d] = out_indices[src_d];
                        src_d += 1;
                    }
                }

                var acc: f32 = 0.0;
                for (0..reduce_size) |k| {
                    src_indices[ax] = k;
                    acc += self.data[self.getFlatIndex(src_indices[0..self.shape.len])];
                }
                C.data[out_idx] = acc;
            }
        }
        return C;
    } else {
        // 全局归约 (Global reduction over all elements)
        var total: f32 = 0.0;
        const elem_count = self.shape.numel();
        if (self.isContiguous() and self.data.len >= elem_count) {
            for (self.data[0..elem_count]) |v| {
                total += v;
            }
        } else {
            var coord = [_]usize{0} ** 8;
            const len = self.shape.len;
            for (0..elem_count) |_| {
                var src_idx: usize = 0;
                for (0..len) |d| {
                    src_idx += coord[d] * self.strides.dims[d];
                }
                total += self.data[src_idx];
                self.shape.incrementCoord(&coord);
            }
        }

        if (keepdims) {
            const out_shape_dims = [_]usize{1} ** 8;
            const out_shape = try Shape.fromSlice(out_shape_dims[0..self.shape.len]);
            const C = try zeros(allocator, out_shape.dims[0..out_shape.len]);
            C.data[0] = total;
            return C;
        } else {
            const C = try zeros(allocator, &.{1});
            C.data[0] = total;
            return C;
        }
    }
}

/// 通用多维张量沿指定轴或全局均值归约 (Mean Reduction)
pub fn mean(self: *Tensor, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor {
    const C = try self.sum(axis, keepdims, allocator);
    const count = if (axis) |ax| @as(f32, @floatFromInt(self.shape.dims[ax])) else @as(f32, @floatFromInt(self.shape.numel()));
    for (C.data) |*val| {
        val.* /= count;
    }
    return C;
}

/// 通用多维张量沿指定轴或全局方差 (Variance Reduction)
pub fn variance(self: *Tensor, axis: ?usize, keepdims: bool, ddof: usize, allocator: std.mem.Allocator) !*Tensor {
    if (axis) |ax| {
        if (ax >= self.shape.len) return error.DimensionOutOfBounds;
    }
    const count = if (axis) |ax| self.shape.dims[ax] else self.shape.numel();
    if (count <= ddof) return error.InvalidDDOF;

    const mean_t = try self.mean(axis, true, allocator);
    defer free(allocator, mean_t);

    const diff = try self.sub(mean_t, allocator);
    defer free(allocator, diff);
    const sq = try diff.mul(diff, allocator);
    defer free(allocator, sq);

    const sum_sq = try sq.sum(axis, keepdims, allocator);
    const denom = @as(f32, @floatFromInt(count - ddof));
    for (sum_sq.data) |*val| {
        val.* /= denom;
    }
    return sum_sq;
}

/// 通用多维张量沿指定轴或全局标准差 (Standard Deviation Reduction)
pub fn stdDev(self: *Tensor, axis: ?usize, keepdims: bool, ddof: usize, allocator: std.mem.Allocator) !*Tensor {
    const var_t = try self.variance(axis, keepdims, ddof, allocator);
    for (var_t.data) |*val| {
        val.* = @sqrt(@max(val.*, 0.0));
    }
    return var_t;
}

/// 依据布尔/条件张量 (支持 `*Tensor` 或 `*BoolTensor` / `*GenericTensor(T)`) 在两个候选张量间进行逐元素选择 (NumPy np.where)
pub fn where(cond: anytype, x: *Tensor, y: *Tensor, allocator: std.mem.Allocator) !*Tensor {
    const s_xy = try broadcastShapes(x.shape, y.shape);
    const target_shape = try broadcastShapes(cond.shape, s_xy);
    const C = try zeros(allocator, target_shape.dims[0..target_shape.len]);

    if (cond.shape.eq(x.shape) and x.shape.eq(y.shape) and cond.isContiguous() and x.isContiguous() and y.isContiguous() and
        cond.data.len >= C.data.len and x.data.len >= C.data.len and y.data.len >= C.data.len)
    {
        for (C.data, cond.data[0..C.data.len], x.data[0..C.data.len], y.data[0..C.data.len]) |*out_v, c_v, x_v, y_v| {
            out_v.* = if (isTruthyScalar(c_v)) x_v else y_v;
        }
        return C;
    }

    const cond_strides = computeBroadcastStrides(cond.shape, cond.strides, target_shape);
    const x_strides = computeBroadcastStrides(x.shape, x.strides, target_shape);
    const y_strides = computeBroadcastStrides(y.shape, y.strides, target_shape);

    const rank = target_shape.len;
    var indices = [_]usize{0} ** 8;
    for (0..C.data.len) |c_flat| {
        var cond_flat: usize = 0;
        var x_flat: usize = 0;
        var y_flat: usize = 0;
        for (0..rank) |d| {
            cond_flat += indices[d] * cond_strides.dims[d];
            x_flat += indices[d] * x_strides.dims[d];
            y_flat += indices[d] * y_strides.dims[d];
        }

        C.data[c_flat] = if (isTruthyScalar(cond.data[cond_flat])) x.data[x_flat] else y.data[y_flat];

        target_shape.incrementCoord(&indices);
    }

    return C;
}

/// 根据 mask (支持 `*Tensor` 或 `*BoolTensor` / `*GenericTensor(T)`) 将满足真值条件的元素赋值为指定标量值（返回新分配副本）
pub fn maskedFill(self: *Tensor, mask: anytype, value: f32, allocator: std.mem.Allocator) !*Tensor {
    if (!self.shape.eq(mask.shape)) return error.ShapeMismatch;
    const C = try self.contiguous(allocator);
    if (mask.isContiguous() and mask.data.len >= C.data.len) {
        for (C.data, mask.data[0..C.data.len]) |*out_v, m_v| {
            if (isTruthyScalar(m_v)) {
                out_v.* = value;
            }
        }
    } else {
        var coord = [_]usize{0} ** 8;
        const len = mask.shape.len;
        for (C.data) |*out_v| {
            var m_idx: usize = 0;
            for (0..len) |d| {
                m_idx += coord[d] * mask.strides.dims[d];
            }
            if (isTruthyScalar(mask.data[m_idx])) {
                out_v.* = value;
            }
            mask.shape.incrementCoord(&coord);
        }
    }
    return C;
}

/// 原地条件掩码填充 (支持 `*Tensor` 或 `*BoolTensor` / `*GenericTensor(T)`)
pub fn maskedFill_(self: *Tensor, mask: anytype, value: f32) !*Tensor {
    if (self.requires_grad or self.creator != null) return error.InPlaceOpOnGraphTensor;
    if (!self.shape.eq(mask.shape)) return error.ShapeMismatch;
    if (self.isContiguous() and mask.isContiguous() and self.data.len >= self.shape.numel() and mask.data.len >= self.shape.numel()) {
        const count = self.shape.numel();
        for (self.data[0..count], mask.data[0..count]) |*out_v, m_v| {
            if (isTruthyScalar(m_v)) {
                out_v.* = value;
            }
        }
    } else {
        var coord = [_]usize{0} ** 8;
        const len = self.shape.len;
        const count = self.shape.numel();
        for (0..count) |_| {
            var s_idx: usize = 0;
            var m_idx: usize = 0;
            for (0..len) |d| {
                s_idx += coord[d] * self.strides.dims[d];
                m_idx += coord[d] * mask.strides.dims[d];
            }
            if (isTruthyScalar(mask.data[m_idx])) {
                self.data[s_idx] = value;
            }
            self.shape.incrementCoord(&coord);
        }
    }
    return self;
}

pub fn compareScalarOp(self: *const Tensor, val: f32, allocator: std.mem.Allocator, comptime cmp_fn: fn (f32, f32) bool) !*GenericTensor(bool) {
    const out = try GenericTensor(bool).init(allocator, self.shape.dims[0..self.shape.len], null);
    errdefer out.deinit(allocator);
    if (self.isContiguous() and self.data.len >= out.data.len) {
        for (self.data[0..out.data.len], out.data) |a, *b| {
            b.* = cmp_fn(a, val);
        }
    } else {
        var coord = [_]usize{0} ** 8;
        const len = self.shape.len;
        for (0..out.data.len) |dest_i| {
            var src_idx: usize = 0;
            for (0..len) |d| src_idx += coord[d] * self.strides.dims[d];
            out.data[dest_i] = cmp_fn(self.data[src_idx], val);
            self.shape.incrementCoord(&coord);
        }
    }
    return out;
}

pub fn gtScalar(self: *const Tensor, val: f32, allocator: std.mem.Allocator) !*GenericTensor(bool) {
    return self.compareScalarOp(val, allocator, struct {
        fn cmp(a: f32, b: f32) bool {
            return a > b;
        }
    }.cmp);
}

pub fn geScalar(self: *const Tensor, val: f32, allocator: std.mem.Allocator) !*GenericTensor(bool) {
    return self.compareScalarOp(val, allocator, struct {
        fn cmp(a: f32, b: f32) bool {
            return a >= b;
        }
    }.cmp);
}

pub fn ltScalar(self: *const Tensor, val: f32, allocator: std.mem.Allocator) !*GenericTensor(bool) {
    return self.compareScalarOp(val, allocator, struct {
        fn cmp(a: f32, b: f32) bool {
            return a < b;
        }
    }.cmp);
}

pub fn leScalar(self: *const Tensor, val: f32, allocator: std.mem.Allocator) !*GenericTensor(bool) {
    return self.compareScalarOp(val, allocator, struct {
        fn cmp(a: f32, b: f32) bool {
            return a <= b;
        }
    }.cmp);
}

pub fn eqScalar(self: *const Tensor, val: f32, allocator: std.mem.Allocator) !*GenericTensor(bool) {
    return self.compareScalarOp(val, allocator, struct {
        fn cmp(a: f32, b: f32) bool {
            return a == b;
        }
    }.cmp);
}

pub fn neScalar(self: *const Tensor, val: f32, allocator: std.mem.Allocator) !*GenericTensor(bool) {
    return self.compareScalarOp(val, allocator, struct {
        fn cmp(a: f32, b: f32) bool {
            return a != b;
        }
    }.cmp);
}

/// 计算 Squeeze 之后的目标形状: 移除所有为 1 的维度，或移除指定为 1 的维度
pub fn squeezedShape(self: *const Tensor, axis: ?usize) !Shape {
    var out = Shape{ .dims = [_]usize{0} ** 8, .len = 0 };
    if (axis) |ax| {
        if (ax >= self.shape.len) return error.DimensionOutOfBounds;
        if (self.shape.dims[ax] != 1) return error.CannotSqueezeDimension;
        for (0..self.shape.len) |d| {
            if (d != ax) {
                out.dims[out.len] = self.shape.dims[d];
                out.len += 1;
            }
        }
    } else {
        for (0..self.shape.len) |d| {
            if (self.shape.dims[d] != 1) {
                out.dims[out.len] = self.shape.dims[d];
                out.len += 1;
            }
        }
    }
    if (out.len == 0) {
        out.dims[0] = 1;
        out.len = 1;
    }
    return out;
}

/// 计算 Unsqueeze 之后的目标形状: 在指定位置插入一个大小为 1 的新维度
pub fn unsqueezedShape(self: *const Tensor, dim: usize) !Shape {
    if (dim > self.shape.len) return error.DimensionOutOfBounds;
    if (self.shape.len >= 8) return error.MaxDimensionsExceeded;
    var out = Shape{ .dims = [_]usize{0} ** 8, .len = self.shape.len + 1 };
    var src_d: usize = 0;
    for (0..out.len) |d| {
        if (d == dim) {
            out.dims[d] = 1;
        } else {
            out.dims[d] = self.shape.dims[src_d];
            src_d += 1;
        }
    }
    return out;
}

/// 压缩单维度 (Squeeze): 移除所有为 1 的维度，或移除指定为 1 的维度
pub fn squeeze(self: *Tensor, axis: ?usize, allocator: std.mem.Allocator) !*Tensor {
    const target = try self.squeezedShape(axis);
    return self.reshape(target.dims[0..target.len], allocator);
}

/// 扩充单维度 (Unsqueeze / expand_dims): 在指定位置插入一个大小为 1 的新维度
pub fn unsqueeze(self: *Tensor, dim: usize, allocator: std.mem.Allocator) !*Tensor {
    const target = try self.unsqueezedShape(dim);
    return self.reshape(target.dims[0..target.len], allocator);
}

/// 跨步零拷贝切片 (Strided View Slicing)
/// 返回一个共享底层内存缓冲区的零拷贝视图张量 (is_view = true)；
/// 需要可微的 Slice 算子时使用 `Graph.slice`
pub fn slice(self: *Tensor, ranges: []const SliceRange, allocator: std.mem.Allocator) !*Tensor {
    if (ranges.len > self.shape.len) return error.DimensionOutOfBounds;

    var new_dims = [_]usize{0} ** 8;
    var new_strides = [_]usize{0} ** 8;
    var offset: usize = 0;

    for (0..self.shape.len) |d| {
        const dim_size = self.shape.dims[d];
        const stride = self.strides.dims[d];
        const range = if (d < ranges.len) ranges[d] else SliceRange{};
        if (range.step == 0) return error.InvalidStep;

        const start = range.start orelse 0;
        const end = range.end orelse dim_size;

        if (start > dim_size or end > dim_size) return error.IndexOutOfBounds;
        if (end < start) return error.InvalidSliceRange;

        const slice_len = if (end > start) (end - start + range.step - 1) / range.step else 0;
        new_dims[d] = slice_len;
        new_strides[d] = stride * range.step;
        offset += start * stride;
    }

    const out = try allocator.create(Tensor);
    out.* = Tensor{
        .data = self.data[offset..],
        .grad = if (self.requires_grad and self.grad.len > offset) self.grad[offset..] else &.{},
        .shape = Shape{ .dims = new_dims, .len = self.shape.len },
        .strides = Shape{ .dims = new_strides, .len = self.shape.len },
        .requires_grad = false,
        .creator = null,
        .is_view = true,
    };
    return out;
}

/// 连续化内存拷贝 (Contiguous copy)
pub fn contiguous(self: Tensor, allocator: std.mem.Allocator) !*Tensor {
    if (self.isContiguous() and !self.is_view) {
        return self.clone(allocator);
    }
    const out = try zeros(allocator, self.shape.dims[0..self.shape.len]);
    errdefer free(allocator, out);

    var coord = [_]usize{0} ** 8;
    const len = self.shape.len;
    for (0..out.data.len) |dest_i| {
        var src_idx: usize = 0;
        for (0..len) |d| {
            src_idx += coord[d] * self.strides.dims[d];
        }
        out.data[dest_i] = self.data[src_idx];

        self.shape.incrementCoord(&coord);
    }
    return out;
}

/// 元素截断操作 (Clip): 将张量元素限制在 [min_val, max_val] 之间
pub fn clip(self: *Tensor, min_val: f32, max_val: f32, allocator: std.mem.Allocator) !*Tensor {
    if (min_val > max_val) return error.InvalidRange;
    const out = try zeros(allocator, self.shape.dims[0..self.shape.len]);
    if (self.isContiguous() and self.data.len >= out.data.len) {
        for (self.data[0..out.data.len], out.data) |x, *y| {
            y.* = std.math.clamp(x, min_val, max_val);
        }
    } else {
        var coord = [_]usize{0} ** 8;
        const len = self.shape.len;
        for (0..out.data.len) |dest_i| {
            var src_idx: usize = 0;
            for (0..len) |d| {
                src_idx += coord[d] * self.strides.dims[d];
            }
            out.data[dest_i] = std.math.clamp(self.data[src_idx], min_val, max_val);

            self.shape.incrementCoord(&coord);
        }
    }
    return out;
}

/// 就地截断操作 (In-place Clip)
pub fn clip_(self: *Tensor, min_val: f32, max_val: f32) !*Tensor {
    if (self.requires_grad or self.creator != null) return error.InPlaceOpOnGraphTensor;
    if (min_val > max_val) return error.InvalidRange;
    for (self.data) |*item| {
        item.* = std.math.clamp(item.*, min_val, max_val);
    }
    return self;
}

/// 沿指定轴对张量进行排序 (Sort)
pub fn sort(self: *Tensor, axis: ?usize, ascending: bool, allocator: std.mem.Allocator) !*Tensor {
    const ax = axis orelse (if (self.shape.len > 0) self.shape.len - 1 else 0);
    if (ax >= self.shape.len) return error.DimensionOutOfBounds;

    const out = try self.contiguous(allocator);
    errdefer free(allocator, out);

    const dim_size = out.shape.dims[ax];
    if (dim_size <= 1) return out;

    var outer_size: usize = 1;
    for (0..ax) |d| outer_size *= out.shape.dims[d];
    var inner_size: usize = 1;
    for (ax + 1..out.shape.len) |d| inner_size *= out.shape.dims[d];

    const temp_buf = try allocator.alloc(f32, dim_size);
    defer allocator.free(temp_buf);

    const stride = inner_size;
    for (0..outer_size) |outer| {
        for (0..inner_size) |inner| {
            const base = outer * dim_size * inner_size + inner;
            for (0..dim_size) |k| {
                temp_buf[k] = out.data[base + k * stride];
            }
            if (ascending) {
                std.mem.sort(f32, temp_buf, {}, struct {
                    fn asc(_: void, a: f32, b: f32) bool {
                        return a < b;
                    }
                }.asc);
            } else {
                std.mem.sort(f32, temp_buf, {}, struct {
                    fn desc(_: void, a: f32, b: f32) bool {
                        return a > b;
                    }
                }.desc);
            }
            for (0..dim_size) |k| {
                out.data[base + k * stride] = temp_buf[k];
            }
        }
    }
    return out;
}

/// 沿指定轴返回排序后的索引 (Argsort)
pub fn argsort(self: *Tensor, axis: ?usize, ascending: bool, allocator: std.mem.Allocator) !*GenericTensor(usize) {
    const ax = axis orelse (if (self.shape.len > 0) self.shape.len - 1 else 0);
    if (ax >= self.shape.len) return error.DimensionOutOfBounds;

    const contig = try self.contiguous(allocator);
    defer free(allocator, contig);

    const out_indices = try GenericTensor(usize).init(allocator, contig.shape.dims[0..contig.shape.len], null);
    errdefer out_indices.deinit(allocator);

    const dim_size = contig.shape.dims[ax];
    if (dim_size == 0) return out_indices;

    var outer_size: usize = 1;
    for (0..ax) |d| outer_size *= contig.shape.dims[d];
    var inner_size: usize = 1;
    for (ax + 1..contig.shape.len) |d| inner_size *= contig.shape.dims[d];

    const temp_vals = try allocator.alloc(f32, dim_size);
    defer allocator.free(temp_vals);
    const temp_idxs = try allocator.alloc(usize, dim_size);
    defer allocator.free(temp_idxs);

    const SortCtx = struct {
        vals: []const f32,
        asc: bool,
        fn cmp(ctx: @This(), a: usize, b: usize) bool {
            if (ctx.asc) {
                return ctx.vals[a] < ctx.vals[b];
            } else {
                return ctx.vals[a] > ctx.vals[b];
            }
        }
    };

    const stride = inner_size;
    for (0..outer_size) |outer| {
        for (0..inner_size) |inner| {
            const base = outer * dim_size * inner_size + inner;
            for (0..dim_size) |k| {
                temp_vals[k] = contig.data[base + k * stride];
                temp_idxs[k] = k;
            }
            std.mem.sort(usize, temp_idxs, SortCtx{ .vals = temp_vals, .asc = ascending }, SortCtx.cmp);
            for (0..dim_size) |k| {
                out_indices.data[base + k * stride] = temp_idxs[k];
            }
        }
    }
    return out_indices;
}

/// 检索非零元素的坐标 (NumPy-like nonzero)
/// 返回形状为 [nonzeros_count, rank] 的 GenericTensor(usize)
pub fn nonzero(self: Tensor, allocator: std.mem.Allocator) !*GenericTensor(usize) {
    var count: usize = 0;
    var coord = [_]usize{0} ** 8;
    const len = self.shape.len;
    const elem_count = self.shape.numel();

    for (0..elem_count) |_| {
        var src_idx: usize = 0;
        for (0..len) |d| {
            src_idx += coord[d] * self.strides.dims[d];
        }
        if (self.data[src_idx] != 0.0) {
            count += 1;
        }
        self.shape.incrementCoord(&coord);
    }

    const out = try GenericTensor(usize).init(allocator, &.{ count, len }, null);
    errdefer out.deinit(allocator);

    @memset(&coord, 0);
    var row: usize = 0;
    for (0..elem_count) |_| {
        var src_idx: usize = 0;
        for (0..len) |d| {
            src_idx += coord[d] * self.strides.dims[d];
        }
        if (self.data[src_idx] != 0.0) {
            for (0..len) |d| {
                out.data[row * len + d] = coord[d];
            }
            row += 1;
        }
        self.shape.incrementCoord(&coord);
    }
    return out;
}
