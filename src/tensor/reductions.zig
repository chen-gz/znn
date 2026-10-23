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
    errdefer allocator.destroy(t);
    const data = try allocator.alloc(f32, self.data.len);
    errdefer allocator.free(data);
    const grad: []f32 = if (self.requires_grad) try allocator.alloc(f32, self.grad.len) else &.{};
    t.* = Tensor{
        .data = data,
        .grad = grad,
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

/// 对张量的每个逻辑元素（视图只覆盖其可见元素，不触碰底层缓冲区的其余部分）按行优先顺序原地执行 `f(elem, arg)`
pub fn mapInPlace(self: *Tensor, arg: anytype, comptime f: fn (*f32, @TypeOf(arg)) void) void {
    const count = self.shape.numel();
    if (self.isContiguous() and self.data.len >= count) {
        for (self.data[0..count]) |*item| f(item, arg);
        return;
    }
    var it = shape_mod.RowIter(1).init(self.shape, .{self.strides});
    const s = it.inner_strides[0];
    while (it.next()) |offs| {
        for (0..it.inner) |i| f(&self.data[offs[0] + i * s], arg);
    }
}

pub fn mulScalar_(self: *Tensor, val: f32) *Tensor {
    std.debug.assert(!self.requires_grad);
    std.debug.assert(self.creator == null);
    mapInPlace(self, val, struct {
        fn f(item: *f32, v: f32) void {
            item.* *= v;
        }
    }.f);
    return self;
}

pub fn addScalar_(self: *Tensor, val: f32) *Tensor {
    std.debug.assert(!self.requires_grad);
    std.debug.assert(self.creator == null);
    mapInPlace(self, val, struct {
        fn f(item: *f32, v: f32) void {
            item.* += v;
        }
    }.f);
    return self;
}

/// 原地逐元素加法 `self += other`：两者必须同形状，支持任意跨步视图
pub fn add_(self: *Tensor, other: *Tensor) !*Tensor {
    std.debug.assert(!self.requires_grad);
    std.debug.assert(self.creator == null);
    if (!self.shape.eq(other.shape)) return error.ShapeMismatch;
    var it = shape_mod.RowIter(2).init(self.shape, .{ self.strides, other.strides });
    const s_self = it.inner_strides[0];
    const s_other = it.inner_strides[1];
    while (it.next()) |offs| {
        if (s_self == 1 and s_other == 1) {
            for (self.data[offs[0]..][0..it.inner], other.data[offs[1]..][0..it.inner]) |*item, other_val| {
                item.* += other_val;
            }
        } else {
            for (0..it.inner) |i| self.data[offs[0] + i * s_self] += other.data[offs[1] + i * s_other];
        }
    }
    return self;
}

/// 沿单轴的极值类归约种类
const ExtremumKind = enum { argmax, argmin, max, min };

/// 沿 `dim` 轴的极值归约通用内核：输出保留被归约轴（大小为 1）。
/// 相同数值保留最先出现的下标；`inner > 1` 时按 (outer, k, inner) 顺序逐行比较，访存连续且可向量化，
/// 每个输出元素的比较顺序与逐列扫描完全一致，因此结果逐位相同。
fn reduceExtremum(self: Tensor, dim: usize, comptime kind: ExtremumKind, allocator: std.mem.Allocator) !*Tensor {
    if (self.shape.len == 0) return error.UnsupportedDimension;
    if (dim >= self.shape.len) return error.DimensionOutOfBounds;
    const contig_ptr: ?*Tensor = if (!self.isContiguous() or self.is_view) try self.contiguous(allocator) else null;
    defer if (contig_ptr) |cp| free(allocator, cp);
    const x_data = if (contig_ptr) |cp| cp.data else self.data;

    var out_dims: [8]usize = @splat(0);
    for (0..self.shape.len) |d| {
        out_dims[d] = if (d == dim) 1 else self.shape.dims[d];
    }
    const C = try zeros(allocator, out_dims[0..self.shape.len]);
    errdefer free(allocator, C);

    const reduce_size = self.shape.dims[dim];
    if (reduce_size == 0) return C;

    var outer_size: usize = 1;
    for (0..dim) |d| {
        outer_size *= self.shape.dims[d];
    }
    var inner_size: usize = 1;
    for ((dim + 1)..self.shape.len) |d| {
        inner_size *= self.shape.dims[d];
    }

    const want_index = comptime (kind == .argmax or kind == .argmin);
    const Cmp = struct {
        inline fn better(a: f32, b: f32) bool {
            return switch (kind) {
                .argmax, .max => a > b,
                .argmin, .min => a < b,
            };
        }
    };

    if (inner_size == 1) {
        for (0..outer_size) |outer| {
            const row = x_data[outer * reduce_size ..][0..reduce_size];
            var best_idx: usize = 0;
            var best_val = row[0];
            for (row[1..], 1..) |val, k| {
                if (Cmp.better(val, best_val)) {
                    best_val = val;
                    best_idx = k;
                }
            }
            C.data[outer] = if (want_index) @floatFromInt(best_idx) else best_val;
        }
        return C;
    }

    // 下标类归约需要额外的当前最优值缓冲区；数值类归约直接在输出行上更新
    const best_buf: []f32 = if (want_index) try allocator.alloc(f32, inner_size) else &.{};
    defer if (want_index) allocator.free(best_buf);

    for (0..outer_size) |outer| {
        const out_row = C.data[outer * inner_size ..][0..inner_size];
        const best_vals = if (want_index) best_buf else out_row;
        const src_base = outer * reduce_size * inner_size;
        @memcpy(best_vals, x_data[src_base..][0..inner_size]);
        for (1..reduce_size) |k| {
            const src_row = x_data[src_base + k * inner_size ..][0..inner_size];
            if (want_index) {
                const k_f: f32 = @floatFromInt(k);
                for (best_vals, src_row, out_row) |*best, val, *idx| {
                    if (Cmp.better(val, best.*)) {
                        best.* = val;
                        idx.* = k_f;
                    }
                }
            } else {
                for (best_vals, src_row) |*best, val| {
                    if (Cmp.better(val, best.*)) best.* = val;
                }
            }
        }
    }
    return C;
}

pub fn argmax(self: Tensor, dim: usize, allocator: std.mem.Allocator) !*Tensor {
    return reduceExtremum(self, dim, .argmax, allocator);
}

pub fn argmin(self: Tensor, dim: usize, allocator: std.mem.Allocator) !*Tensor {
    return reduceExtremum(self, dim, .argmin, allocator);
}

pub fn max(self: Tensor, dim: usize, allocator: std.mem.Allocator) !*Tensor {
    return reduceExtremum(self, dim, .max, allocator);
}

pub fn min(self: Tensor, dim: usize, allocator: std.mem.Allocator) !*Tensor {
    return reduceExtremum(self, dim, .min, allocator);
}

pub fn isContiguous(self: Tensor) bool {
    return isContiguousStrides(self.shape, self.strides);
}

pub fn numel(self: Tensor) usize {
    return self.shape.numel();
}

const V8 = @Vector(8, f32);

/// 连续切片求和：8 路向量累加器打破串行加法的依赖链，比逐元素串行累加更快且舍入误差更小
pub fn sumSlice(values: []const f32) f32 {
    var acc: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= values.len) : (i += 8) {
        acc += values[i..][0..8].*;
    }
    var total: f32 = @reduce(.Add, acc);
    while (i < values.len) : (i += 1) total += values[i];
    return total;
}

/// 连续切片的平方和（8 路向量累加）
pub fn sumSquaresSlice(values: []const f32) f32 {
    var acc: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= values.len) : (i += 8) {
        const v: V8 = values[i..][0..8].*;
        acc += v * v;
    }
    var total: f32 = @reduce(.Add, acc);
    while (i < values.len) : (i += 1) total += values[i] * values[i];
    return total;
}

/// 连续切片相对给定中心的偏差平方和 `sum((x - center)^2)`（8 路向量累加）
pub fn sumSquaredDeviationsSlice(values: []const f32, center: f32) f32 {
    const c_vec: V8 = @splat(center);
    var acc: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= values.len) : (i += 8) {
        const d = @as(V8, values[i..][0..8].*) - c_vec;
        acc += d * d;
    }
    var total: f32 = @reduce(.Add, acc);
    while (i < values.len) : (i += 1) {
        const d = values[i] - center;
        total += d * d;
    }
    return total;
}

/// 非空连续切片的最大值（与逐元素 `if (v > m) m = v` 扫描结果一致，含 NaN 时语义可能不同，调用方仅用于数值稳定化平移）
pub fn maxSlice(values: []const f32) f32 {
    std.debug.assert(values.len > 0);
    var i: usize = 0;
    var m: f32 = values[0];
    if (values.len >= 8) {
        var acc: V8 = values[0..8].*;
        i = 8;
        while (i + 8 <= values.len) : (i += 8) {
            acc = @max(acc, @as(V8, values[i..][0..8].*));
        }
        m = @reduce(.Max, acc);
    }
    while (i < values.len) : (i += 1) m = @max(m, values[i]);
    return m;
}

/// 通用多维张量沿指定轴或全局求和归约 (Sum Reduction)
/// 非连续输入先物化为连续副本；沿轴归约按 (outer, k, inner) 顺序把整行累加到输出行，访存连续且可向量化
pub fn sum(self: *Tensor, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor {
    if (axis) |ax| {
        if (ax >= self.shape.len) return error.DimensionOutOfBounds;
    }
    const contig_ptr: ?*Tensor = if (!self.isContiguous()) try self.contiguous(allocator) else null;
    defer if (contig_ptr) |cp| free(allocator, cp);
    const x_data = (if (contig_ptr) |cp| cp.data else self.data)[0..self.shape.numel()];

    if (axis) |ax| {
        const reduce_size = self.shape.dims[ax];

        var out_shape_dims: [8]usize = @splat(0);
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

        const C = try zeros(allocator, out_shape_dims[0..out_rank]);

        var outer_size: usize = 1;
        for (0..ax) |d| {
            outer_size *= self.shape.dims[d];
        }
        var inner_size: usize = 1;
        for ((ax + 1)..self.shape.len) |d| {
            inner_size *= self.shape.dims[d];
        }

        if (inner_size == 1) {
            for (0..outer_size) |outer| {
                C.data[outer] = sumSlice(x_data[outer * reduce_size ..][0..reduce_size]);
            }
        } else {
            for (0..outer_size) |outer| {
                const out_row = C.data[outer * inner_size ..][0..inner_size];
                const src_base = outer * reduce_size * inner_size;
                for (0..reduce_size) |k| {
                    const src_row = x_data[src_base + k * inner_size ..][0..inner_size];
                    for (out_row, src_row) |*acc, v| acc.* += v;
                }
            }
        }
        return C;
    } else {
        // 全局归约 (Global reduction over all elements)
        const total = sumSlice(x_data);
        if (keepdims) {
            const out_shape_dims: [8]usize = @splat(1);
            const C = try zeros(allocator, out_shape_dims[0..self.shape.len]);
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

    // 偏差平方原地写回 diff，避免额外的平方张量分配
    const diff = try self.sub(mean_t, allocator);
    defer free(allocator, diff);
    for (diff.data) |*v| v.* *= v.*;

    const sum_sq = try diff.sum(axis, keepdims, allocator);
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

    // 广播或非连续路径：折叠维度后逐行遍历，行内按各操作数的最内层跨度寻址
    var it = shape_mod.RowIter(4).init(target_shape, .{
        C.strides,
        computeBroadcastStrides(cond.shape, cond.strides, target_shape),
        computeBroadcastStrides(x.shape, x.strides, target_shape),
        computeBroadcastStrides(y.shape, y.strides, target_shape),
    });
    const s_cond = it.inner_strides[1];
    const s_x = it.inner_strides[2];
    const s_y = it.inner_strides[3];
    while (it.next()) |offs| {
        for (C.data[offs[0]..][0..it.inner], 0..) |*out_v, i| {
            out_v.* = if (isTruthyScalar(cond.data[offs[1] + i * s_cond])) x.data[offs[2] + i * s_x] else y.data[offs[3] + i * s_y];
        }
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
        var coord: [8]usize = @splat(0);
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
        var coord: [8]usize = @splat(0);
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
        var coord: [8]usize = @splat(0);
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
    var out = Shape{ .dims = @splat(0), .len = 0 };
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
    var out = Shape{ .dims = @splat(0), .len = self.shape.len + 1 };
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

    var new_dims: [8]usize = @splat(0);
    var new_strides: [8]usize = @splat(0);
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

    // 空切片不引用任何元素：偏移归零，避免 `start == dim_size` 时累加偏移越过缓冲区末尾
    var is_empty = false;
    for (new_dims[0..self.shape.len]) |dim| {
        if (dim == 0) is_empty = true;
    }
    if (is_empty) offset = 0;

    const out = try allocator.create(Tensor);
    out.* = Tensor{
        .data = if (is_empty) self.data[0..0] else self.data[offset..],
        .grad = if (!is_empty and self.requires_grad and self.grad.len > offset) self.grad[offset..] else &.{},
        .shape = Shape{ .dims = new_dims, .len = self.shape.len },
        .strides = Shape{ .dims = new_strides, .len = self.shape.len },
        .requires_grad = false,
        .creator = null,
        .is_view = true,
    };
    return out;
}

/// 连续化内存拷贝 (Contiguous copy)：返回只含逻辑元素、行优先连续布局的独立数据张量。
/// 结果不携带梯度缓冲区与算子来源 (requires_grad = false, creator = null)，仅作为数值内核的输入副本；
/// 需要完整复制（含梯度）时使用 `clone`。
pub fn contiguous(self: Tensor, allocator: std.mem.Allocator) !*Tensor {
    const out = try ops_mod.allocUninit(allocator, self.shape);
    shape_mod.gatherStrided(out.data, self.data, self.shape, self.strides);
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
        var coord: [8]usize = @splat(0);
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
    mapInPlace(self, [2]f32{ min_val, max_val }, struct {
        fn f(item: *f32, range: [2]f32) void {
            item.* = std.math.clamp(item.*, range[0], range[1]);
        }
    }.f);
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

    const stride = inner_size;

    var temp_buf: ?[]f32 = null;
    defer if (temp_buf) |buf| allocator.free(buf);
    if (stride > 1) {
        temp_buf = try allocator.alloc(f32, dim_size);
    }

    for (0..outer_size) |outer| {
        for (0..inner_size) |inner| {
            const base = outer * dim_size * inner_size + inner;
            if (stride == 1) {
                const cur_slice = out.data[base .. base + dim_size];
                if (ascending) {
                    std.mem.sort(f32, cur_slice, {}, struct {
                        fn asc(_: void, a: f32, b: f32) bool {
                            return a < b;
                        }
                    }.asc);
                } else {
                    std.mem.sort(f32, cur_slice, {}, struct {
                        fn desc(_: void, a: f32, b: f32) bool {
                            return a > b;
                        }
                    }.desc);
                }
            } else {
                const buf = temp_buf.?;
                for (0..dim_size) |k| {
                    buf[k] = out.data[base + k * stride];
                }
                if (ascending) {
                    std.mem.sort(f32, buf, {}, struct {
                        fn asc(_: void, a: f32, b: f32) bool {
                            return a < b;
                        }
                    }.asc);
                } else {
                    std.mem.sort(f32, buf, {}, struct {
                        fn desc(_: void, a: f32, b: f32) bool {
                            return a > b;
                        }
                    }.desc);
                }
                for (0..dim_size) |k| {
                    out.data[base + k * stride] = buf[k];
                }
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

    const stride = inner_size;

    var temp_idxs: ?[]usize = null;
    defer if (temp_idxs) |idxs| allocator.free(idxs);
    if (stride > 1) {
        temp_idxs = try allocator.alloc(usize, dim_size);
    }

    const SortCtx = struct {
        vals: []const f32,
        base: usize,
        stride: usize,
        asc: bool,
        fn cmp(ctx: @This(), a: usize, b: usize) bool {
            const val_a = ctx.vals[ctx.base + a * ctx.stride];
            const val_b = ctx.vals[ctx.base + b * ctx.stride];
            if (ctx.asc) {
                return val_a < val_b;
            } else {
                return val_a > val_b;
            }
        }
    };

    for (0..outer_size) |outer| {
        for (0..inner_size) |inner| {
            const base = outer * dim_size * inner_size + inner;
            if (stride == 1) {
                const cur_slice = out_indices.data[base .. base + dim_size];
                for (0..dim_size) |k| {
                    cur_slice[k] = k;
                }
                std.mem.sort(usize, cur_slice, SortCtx{ .vals = contig.data, .base = base, .stride = stride, .asc = ascending }, SortCtx.cmp);
            } else {
                const idxs = temp_idxs.?;
                for (0..dim_size) |k| {
                    idxs[k] = k;
                }
                std.mem.sort(usize, idxs, SortCtx{ .vals = contig.data, .base = base, .stride = stride, .asc = ascending }, SortCtx.cmp);
                for (0..dim_size) |k| {
                    out_indices.data[base + k * stride] = idxs[k];
                }
            }
        }
    }
    return out_indices;
}

/// 检索非零元素的坐标 (NumPy-like nonzero)
/// 返回形状为 [nonzeros_count, rank] 的 GenericTensor(usize)
pub fn nonzero(self: Tensor, allocator: std.mem.Allocator) !*GenericTensor(usize) {
    var count: usize = 0;
    var coord: [8]usize = @splat(0);
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
