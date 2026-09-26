const std = @import("std");
const shape_mod = @import("shape.zig");
pub const Shape = shape_mod.Shape;
pub const computeContiguousStrides = shape_mod.computeContiguousStrides;

// ============================================================================
// 2. 数据类型系统与泛型张量 (DType System & Generic Tensor)
// ============================================================================

/// 统一标量数据类型枚举 (DType)
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

/// Brain Floating Point 16-bit 格式 (bfloat16)
/// 符号位 1 位，指数位 8 位，尾数位 7 位（与 IEEE 754 f32 动态范围完全相同）
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

        pub fn isContiguous(self: Self) bool {
            const c_strides = computeContiguousStrides(self.shape);
            return self.strides.eq(c_strides);
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
            if (self.isContiguous()) {
                @memcpy(out.data, self.data);
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

        pub fn to(self: Self, comptime DestT: type, allocator: std.mem.Allocator) !*GenericTensor(DestT) {
            const out = try GenericTensor(DestT).init(allocator, self.shape.dims[0..self.shape.len], null);
            errdefer out.deinit(allocator);

            if (self.isContiguous()) {
                for (self.data, 0..) |val, i| {
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


