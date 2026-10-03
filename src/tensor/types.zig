const std = @import("std");
const shape_mod = @import("shape.zig");
pub const Shape = shape_mod.Shape;
pub const computeContiguousStrides = shape_mod.computeContiguousStrides;
pub const isContiguousStrides = shape_mod.isContiguousStrides;
pub const transposeShape = shape_mod.transposeShape;
pub const broadcastShapes = shape_mod.broadcastShapes;
pub const computeBroadcastStrides = shape_mod.computeBroadcastStrides;

// ============================================================================
// 2. 数据类型系统与泛型张量 (Data Type, DType System & Generic Tensor)
// ============================================================================

/// 统一标量数据类型枚举 (Data Type, DType)
pub const DType = enum {
    f32,
    f64,
    f16,
    bf16,
    i32,
    i64,
    u8,
    bool,

    pub fn sizeOf(self: DType) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f64, .i64 => 8,
            .f16, .bf16 => 2,
            .u8, .bool => 1,
        };
    }
};

/// 脑浮点数 16 位格式 (Brain Floating Point 16-bit, bfloat16 / bf16)
/// 符号位 1 位，指数位 8 位，尾数位 7 位（与电气与电子工程师协会 (Institute of Electrical and Electronics Engineers, IEEE) 754 标准的 32 位单精度浮点数 f32 动态范围完全相同）
pub const bf16 = packed struct {
    bits: u16,

    pub fn fromF32(val: f32) bf16 {
        const u: u32 = @bitCast(val);
        // Round to nearest even
        const lsb = (u >> 16) & 1;
        const rounding_bias: u32 = 0x7fff + lsb;
        const rounded: u32 = u +% rounding_bias;
        return .{ .bits = @truncate(rounded >> 16) };
    }

    pub fn toF32(self: bf16) f32 {
        const u: u32 = @as(u32, self.bits) << 16;
        return @bitCast(u);
    }
};

/// 跨步切片范围描述 (SliceRange)
pub const SliceRange = struct {
    start: ?usize = null,
    end: ?usize = null,
    step: usize = 1,

    pub const default: SliceRange = .{};
    pub fn defaultOptions() SliceRange {
        return .{};
    }
};

/// 卷积步长与填充配置 (Convolution Options)
pub const ConvOptions = struct {
    stride: usize = 1,
    padding: usize = 0,

    pub const default: ConvOptions = .{};
    pub fn defaultOptions() ConvOptions {
        return .{};
    }
};

/// 通用标量类型转换函数
pub fn convertScalar(comptime DestT: type, comptime SrcT: type, val: SrcT) DestT {
    if (DestT == SrcT) return val;
    if (SrcT == bf16) {
        const f = val.toF32();
        return convertScalar(DestT, f32, f);
    }
    if (DestT == bf16) {
        const f = convertScalar(f32, SrcT, val);
        return bf16.fromF32(f);
    }
    if (DestT == bool) {
        if (@typeInfo(SrcT) == .int) return val != 0;
        if (@typeInfo(SrcT) == .float) return val != 0.0;
    }
    if (SrcT == bool) {
        const int_v: usize = if (val) 1 else 0;
        if (@typeInfo(DestT) == .int) return @intCast(int_v);
        if (@typeInfo(DestT) == .float) return @floatFromInt(int_v);
    }
    if (@typeInfo(SrcT) == .float and @typeInfo(DestT) == .float) {
        return @floatCast(val);
    }
    if (@typeInfo(SrcT) == .float and @typeInfo(DestT) == .int) {
        return @intFromFloat(val);
    }
    if (@typeInfo(SrcT) == .int and @typeInfo(DestT) == .float) {
        return @floatFromInt(val);
    }
    if (@typeInfo(SrcT) == .int and @typeInfo(DestT) == .int) {
        return @intCast(val);
    }
    @compileError("Unsupported type conversion between " ++ @typeName(SrcT) ++ " and " ++ @typeName(DestT));
}

/// 判断任意标量是否为非零/真值（支持 bool, bf16, 整数与浮点数）
pub inline fn isTruthyScalar(val: anytype) bool {
    const V = @TypeOf(val);
    if (V == bool) return val;
    if (V == bf16) return val.toF32() != 0.0;
    if (@typeInfo(V) == .float) return val != 0.0;
    if (@typeInfo(V) == .int) return val != 0;
    @compileError("Unsupported condition/mask scalar type: " ++ @typeName(V));
}

/// 泛型张量结构体 (Generic Tensor)
pub fn GenericTensor(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const ElemType = T;

        data: []T,
        shape: Shape,
        strides: Shape,
        is_view: bool = false,

        pub fn init(allocator: std.mem.Allocator, shape_slice: []const usize, initial_val: ?T) !*Self {
            const shape = try Shape.fromSlice(shape_slice);
            const strides = computeContiguousStrides(shape);
            var total_size: usize = 1;
            for (shape_slice) |dim| {
                total_size *= dim;
            }
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);
            const data = try allocator.alloc(T, total_size);
            if (initial_val) |v| {
                @memset(data, v);
            }
            self.* = Self{
                .data = data,
                .shape = shape,
                .strides = strides,
                .is_view = false,
            };
            return self;
        }

        pub fn zeros(allocator: std.mem.Allocator, shape_slice: []const usize) !*Self {
            return Self.init(allocator, shape_slice, convertScalar(T, usize, 0));
        }

        pub fn ones(allocator: std.mem.Allocator, shape_slice: []const usize) !*Self {
            return Self.init(allocator, shape_slice, convertScalar(T, usize, 1));
        }

        pub fn fromSlice(allocator: std.mem.Allocator, shape_slice: []const usize, slice_data: []const T) !*Self {
            const shape = try Shape.fromSlice(shape_slice);
            const strides = computeContiguousStrides(shape);
            var total_size: usize = 1;
            for (shape_slice) |dim| {
                total_size *= dim;
            }
            if (total_size != slice_data.len) return error.ShapeMismatch;
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);
            const data = try allocator.alloc(T, total_size);
            @memcpy(data, slice_data);
            self.* = Self{
                .data = data,
                .shape = shape,
                .strides = strides,
                .is_view = false,
            };
            return self;
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (!self.is_view) {
                allocator.free(self.data);
            }
            allocator.destroy(self);
        }

        pub fn numel(self: Self) usize {
            return self.shape.numel();
        }

        pub fn fill(self: *Self, val: T) void {
            @memset(self.data, val);
        }

        pub fn isContiguous(self: Self) bool {
            return isContiguousStrides(self.shape, self.strides);
        }

        pub fn getFlatIndexChecked(self: Self, indices: []const usize) !usize {
            if (indices.len != self.shape.len) return error.DimensionMismatch;
            var flat_idx: usize = 0;
            for (indices, 0..) |idx, i| {
                if (idx >= self.shape.dims[i]) return error.IndexOutOfBounds;
                flat_idx += idx * self.strides.dims[i];
            }
            return flat_idx;
        }

        pub fn getFlatIndex(self: Self, indices: []const usize) usize {
            return self.getFlatIndexChecked(indices) catch |err| switch (err) {
                error.DimensionMismatch => @panic("getFlatIndex: dimension mismatch"),
                error.IndexOutOfBounds => @panic("getFlatIndex: index out of bounds"),
            };
        }

        pub fn getChecked(self: Self, indices: []const usize) !T {
            const flat_idx = try self.getFlatIndexChecked(indices);
            return self.data[flat_idx];
        }

        pub fn setChecked(self: *Self, indices: []const usize, val: T) !void {
            const flat_idx = try self.getFlatIndexChecked(indices);
            self.data[flat_idx] = val;
        }

        pub fn get(self: Self, indices: []const usize) T {
            return self.data[self.getFlatIndex(indices)];
        }

        pub fn set(self: *Self, indices: []const usize, val: T) void {
            self.data[self.getFlatIndex(indices)] = val;
        }

        pub fn clone(self: Self, allocator: std.mem.Allocator) !*Self {
            const out = try Self.init(allocator, self.shape.dims[0..self.shape.len], null);
            errdefer out.deinit(allocator);
            if (self.isContiguous() and self.data.len >= out.data.len) {
                @memcpy(out.data, self.data[0..out.data.len]);
            } else {
                var coord = [_]usize{0} ** 8;
                const len = self.shape.len;
                for (0..out.data.len) |dest_i| {
                    var src_idx: usize = 0;
                    for (0..len) |d| {
                        src_idx += coord[d] * self.strides.dims[d];
                    }
                    out.data[dest_i] = self.data[src_idx];
                    var d = len;
                    while (d > 0) {
                        d -= 1;
                        coord[d] += 1;
                        if (coord[d] < self.shape.dims[d]) break;
                        coord[d] = 0;
                    }
                }
            }
            return out;
        }

        pub fn contiguous(self: Self, allocator: std.mem.Allocator) !*Self {
            return self.clone(allocator);
        }

        pub fn reshape(self: *Self, new_shape_slice: []const usize, allocator: std.mem.Allocator) !*Self {
            const new_shape = try Shape.fromSlice(new_shape_slice);
            const total = new_shape.numel();
            if (self.shape.numel() != total) return error.ShapeMismatch;

            if (self.isContiguous() and self.data.len >= total) {
                const out = try allocator.create(Self);
                out.* = Self{
                    .data = self.data[0..total],
                    .shape = new_shape,
                    .strides = computeContiguousStrides(new_shape),
                    .is_view = true,
                };
                return out;
            } else {
                const out = try self.clone(allocator);
                out.shape = new_shape;
                out.strides = computeContiguousStrides(new_shape);
                return out;
            }
        }

        pub fn transposeView(self: *Self, dim0: usize, dim1: usize, allocator: std.mem.Allocator) !*Self {
            if (dim0 >= self.shape.len or dim1 >= self.shape.len) return error.InvalidDimension;
            const new_shape = transposeShape(self.shape, dim0, dim1);
            const new_strides = transposeShape(self.strides, dim0, dim1);
            const out = try allocator.create(Self);
            out.* = Self{
                .data = self.data,
                .shape = new_shape,
                .strides = new_strides,
                .is_view = true,
            };
            return out;
        }

        pub fn slice(self: *Self, ranges: []const SliceRange, allocator: std.mem.Allocator) !*Self {
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

            const out = try allocator.create(Self);
            out.* = Self{
                .data = self.data[offset..],
                .shape = Shape{ .dims = new_dims, .len = self.shape.len },
                .strides = Shape{ .dims = new_strides, .len = self.shape.len },
                .is_view = true,
            };
            return out;
        }

        fn binaryBroadcastOp(
            self: *const Self,
            other: *const Self,
            allocator: std.mem.Allocator,
            comptime OutT: type,
            comptime op_fn: fn (T, T) OutT,
        ) !*GenericTensor(OutT) {
            const target_shape = try broadcastShapes(self.shape, other.shape);
            const out = try GenericTensor(OutT).init(allocator, target_shape.dims[0..target_shape.len], null);
            errdefer out.deinit(allocator);

            if (self.shape.eq(other.shape) and self.isContiguous() and other.isContiguous() and
                self.data.len >= out.data.len and other.data.len >= out.data.len)
            {
                for (out.data, self.data[0..out.data.len], other.data[0..out.data.len]) |*c_val, a_val, b_val| {
                    c_val.* = op_fn(a_val, b_val);
                }
                return out;
            }

            const a_strides = computeBroadcastStrides(self.shape, self.strides, target_shape);
            const b_strides = computeBroadcastStrides(other.shape, other.strides, target_shape);
            const rank = target_shape.len;
            var indices = [_]usize{0} ** 8;

            for (0..out.data.len) |c_flat| {
                var a_flat: usize = 0;
                var b_flat: usize = 0;
                for (0..rank) |d| {
                    a_flat += indices[d] * a_strides.dims[d];
                    b_flat += indices[d] * b_strides.dims[d];
                }
                out.data[c_flat] = op_fn(self.data[a_flat], other.data[b_flat]);

                var d: usize = rank;
                while (d > 0) {
                    d -= 1;
                    indices[d] += 1;
                    if (indices[d] < target_shape.dims[d]) break;
                    indices[d] = 0;
                }
            }
            return out;
        }

        pub fn add(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            return self.binaryBroadcastOp(other, allocator, T, struct {
                fn apply(a: T, b: T) T {
                    if (T == bool) return a or b;
                    if (T == bf16) return bf16.fromF32(a.toF32() + b.toF32());
                    return a + b;
                }
            }.apply);
        }

        pub fn sub(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            return self.binaryBroadcastOp(other, allocator, T, struct {
                fn apply(a: T, b: T) T {
                    if (T == bool) return a and !b;
                    if (T == bf16) return bf16.fromF32(a.toF32() - b.toF32());
                    return a - b;
                }
            }.apply);
        }

        pub fn mul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            return self.binaryBroadcastOp(other, allocator, T, struct {
                fn apply(a: T, b: T) T {
                    if (T == bool) return a and b;
                    if (T == bf16) return bf16.fromF32(a.toF32() * b.toF32());
                    return a * b;
                }
            }.apply);
        }

        pub fn div(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            return self.binaryBroadcastOp(other, allocator, T, struct {
                fn apply(a: T, b: T) T {
                    if (T == bool) return a and b;
                    if (T == bf16) return bf16.fromF32(a.toF32() / b.toF32());
                    if (@typeInfo(T) == .int) return @divTrunc(a, b);
                    return a / b;
                }
            }.apply);
        }

        pub fn eq(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*GenericTensor(bool) {
            return self.binaryBroadcastOp(other, allocator, bool, struct {
                fn apply(a: T, b: T) bool {
                    if (T == bf16) return a.toF32() == b.toF32();
                    return a == b;
                }
            }.apply);
        }

        pub fn ne(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*GenericTensor(bool) {
            return self.binaryBroadcastOp(other, allocator, bool, struct {
                fn apply(a: T, b: T) bool {
                    if (T == bf16) return a.toF32() != b.toF32();
                    return a != b;
                }
            }.apply);
        }

        pub fn gt(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*GenericTensor(bool) {
            return self.binaryBroadcastOp(other, allocator, bool, struct {
                fn apply(a: T, b: T) bool {
                    if (T == bool) return a and !b;
                    if (T == bf16) return a.toF32() > b.toF32();
                    return a > b;
                }
            }.apply);
        }

        pub fn ge(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*GenericTensor(bool) {
            return self.binaryBroadcastOp(other, allocator, bool, struct {
                fn apply(a: T, b: T) bool {
                    if (T == bool) return a or !b;
                    if (T == bf16) return a.toF32() >= b.toF32();
                    return a >= b;
                }
            }.apply);
        }

        pub fn lt(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*GenericTensor(bool) {
            return self.binaryBroadcastOp(other, allocator, bool, struct {
                fn apply(a: T, b: T) bool {
                    if (T == bool) return !a and b;
                    if (T == bf16) return a.toF32() < b.toF32();
                    return a < b;
                }
            }.apply);
        }

        pub fn le(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*GenericTensor(bool) {
            return self.binaryBroadcastOp(other, allocator, bool, struct {
                fn apply(a: T, b: T) bool {
                    if (T == bool) return !a or b;
                    if (T == bf16) return a.toF32() <= b.toF32();
                    return a <= b;
                }
            }.apply);
        }

        pub fn any(self: *const Self) bool {
            const elem_count = self.shape.numel();
            if (self.isContiguous() and self.data.len >= elem_count) {
                for (self.data[0..elem_count]) |v| {
                    if (isTruthyScalar(v)) return true;
                }
                return false;
            }
            var coord = [_]usize{0} ** 8;
            const len = self.shape.len;
            for (0..elem_count) |_| {
                var src_idx: usize = 0;
                for (0..len) |d| src_idx += coord[d] * self.strides.dims[d];
                if (isTruthyScalar(self.data[src_idx])) return true;
                var d = len;
                while (d > 0) {
                    d -= 1;
                    coord[d] += 1;
                    if (coord[d] < self.shape.dims[d]) break;
                    coord[d] = 0;
                }
            }
            return false;
        }

        pub fn all(self: *const Self) bool {
            const elem_count = self.shape.numel();
            if (self.isContiguous() and self.data.len >= elem_count) {
                for (self.data[0..elem_count]) |v| {
                    if (!isTruthyScalar(v)) return false;
                }
                return true;
            }
            var coord = [_]usize{0} ** 8;
            const len = self.shape.len;
            for (0..elem_count) |_| {
                var src_idx: usize = 0;
                for (0..len) |d| src_idx += coord[d] * self.strides.dims[d];
                if (!isTruthyScalar(self.data[src_idx])) return false;
                var d = len;
                while (d > 0) {
                    d -= 1;
                    coord[d] += 1;
                    if (coord[d] < self.shape.dims[d]) break;
                    coord[d] = 0;
                }
            }
            return true;
        }

        pub fn sum(self: *const Self, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*Self {
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
                } else if (self.shape.len == 1) {
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

                const out = try Self.zeros(allocator, out_shape_dims[0..out_rank]);
                errdefer out.deinit(allocator);

                var out_indices = [_]usize{0} ** 8;
                for (0..out.data.len) |out_idx| {
                    var tmp = out_idx;
                    var d: usize = out_rank;
                    while (d > 0) {
                        d -= 1;
                        out_indices[d] = tmp % out.shape.dims[d];
                        tmp /= out.shape.dims[d];
                    }

                    var src_indices = [_]usize{0} ** 8;
                    if (keepdims) {
                        for (0..self.shape.len) |idx_d| src_indices[idx_d] = out_indices[idx_d];
                    } else {
                        var src_d: usize = 0;
                        for (0..self.shape.len) |idx_d| {
                            if (idx_d == ax) continue;
                            src_indices[idx_d] = out_indices[src_d];
                            src_d += 1;
                        }
                    }

                    var acc: f64 = 0.0;
                    for (0..reduce_size) |k| {
                        src_indices[ax] = k;
                        const v = self.data[self.getFlatIndex(src_indices[0..self.shape.len])];
                        acc += convertScalar(f64, T, v);
                    }
                    out.data[out_idx] = convertScalar(T, f64, acc);
                }
                return out;
            } else {
                var acc: f64 = 0.0;
                const elem_count = self.shape.numel();
                if (self.isContiguous() and self.data.len >= elem_count) {
                    for (self.data[0..elem_count]) |v| acc += convertScalar(f64, T, v);
                } else {
                    var coord = [_]usize{0} ** 8;
                    const len = self.shape.len;
                    for (0..elem_count) |_| {
                        var src_idx: usize = 0;
                        for (0..len) |d| src_idx += coord[d] * self.strides.dims[d];
                        acc += convertScalar(f64, T, self.data[src_idx]);
                        var d = len;
                        while (d > 0) {
                            d -= 1;
                            coord[d] += 1;
                            if (coord[d] < self.shape.dims[d]) break;
                            coord[d] = 0;
                        }
                    }
                }
                if (keepdims) {
                    const out_shape_dims = [_]usize{1} ** 8;
                    return Self.init(allocator, out_shape_dims[0..self.shape.len], convertScalar(T, f64, acc));
                } else {
                    return Self.init(allocator, &.{1}, convertScalar(T, f64, acc));
                }
            }
        }

        pub fn mean(self: *const Self, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*GenericTensor(f32) {
            const f32_t = try self.to(f32, allocator);
            defer f32_t.deinit(allocator);
            const sum_t = try f32_t.sum(axis, keepdims, allocator);
            const count = if (axis) |ax| @as(f32, @floatFromInt(self.shape.dims[ax])) else @as(f32, @floatFromInt(self.shape.numel()));
            for (sum_t.data) |*val| {
                val.* /= count;
            }
            return sum_t;
        }

        pub fn to(self: Self, comptime DestT: type, allocator: std.mem.Allocator) !*GenericTensor(DestT) {
            const out = try GenericTensor(DestT).init(allocator, self.shape.dims[0..self.shape.len], null);
            errdefer out.deinit(allocator);

            if (self.isContiguous() and self.data.len >= out.data.len) {
                for (self.data[0..out.data.len], 0..) |val, i| {
                    out.data[i] = convertScalar(DestT, T, val);
                }
            } else {
                var coord = [_]usize{0} ** 8;
                const len = self.shape.len;
                for (0..out.data.len) |dest_i| {
                    var src_idx: usize = 0;
                    for (0..len) |d| {
                        src_idx += coord[d] * self.strides.dims[d];
                    }
                    out.data[dest_i] = convertScalar(DestT, T, self.data[src_idx]);
                    var d = len;
                    while (d > 0) {
                        d -= 1;
                        coord[d] += 1;
                        if (coord[d] < self.shape.dims[d]) break;
                        coord[d] = 0;
                    }
                }
            }
            return out;
        }
    };
}

pub const FloatTensor = GenericTensor(f32);
pub const DoubleTensor = GenericTensor(f64);
pub const IntTensor = GenericTensor(i32);
pub const LongTensor = GenericTensor(i64);
pub const BoolTensor = GenericTensor(bool);
pub const BFloat16Tensor = GenericTensor(bf16);
pub const UsizeTensor = GenericTensor(usize);
pub const TensorOf = GenericTensor;


