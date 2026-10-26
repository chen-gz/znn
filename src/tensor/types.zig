const std = @import("std");
const backend = @import("../backend.zig");
const vec_math = backend.vec_math;
const shape_mod = @import("shape.zig");
pub const Shape = shape_mod.Shape;
pub const computeContiguousStrides = shape_mod.computeContiguousStrides;
pub const isContiguousStrides = shape_mod.isContiguousStrides;
pub const transposeShape = shape_mod.transposeShape;
pub const broadcastShapes = shape_mod.broadcastShapes;
pub const computeBroadcastStrides = shape_mod.computeBroadcastStrides;

/// Explicit tensor validation errors, retained in every optimization mode.
pub const TensorError = error{
    InPlaceOpOnGraphTensor,
    ShapeMismatch,
    StrideMismatch,
    DeviceMismatch,
    NonContiguousLayout,
    UnsupportedDType,
    DimensionOutOfBounds,
    InvalidStorage,
    AliasedOutput,
};

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
        // 非数 (Not a Number, NaN) 直接截断并强制置位静默位：舍入偏置可能把低位尾数进位到指数位而把 NaN 变成无穷大
        if (std.math.isNan(val)) return .{ .bits = @as(u16, @truncate(u >> 16)) | 0x0040 };
        // Round to nearest even
        const lsb = (u >> 16) & 1;
        const rounding_bias: u32 = 0x7fff + lsb;
        const rounded: u32 = u +% rounding_bias;
        return .{ .bits = @truncate(rounded >> 16) };
    }

    pub fn toF32(self: bf16) f32 {
        return vec_math.bf16ToF32(self.bits);
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

/// 卷积步长与填充配置 (Convolution Options, ConvOptions)
pub const ConvOptions = struct {
    stride: usize = 1,
    padding: usize = 0,

    pub const default: ConvOptions = .{};
    pub fn defaultOptions() ConvOptions {
        return .{};
    }
};

/// 均方根层归一化可选参数配置 (Root Mean Square Layer Normalization Options, RmsNormOptions)：
/// y = x / sqrt(mean(x^2) + eps) * (weight + weight_offset)；不带权重时 y = x / sqrt(mean(x^2) + eps)
pub const RmsNormOptions = struct {
    eps: f32 = 1e-6, // 方差项的数值稳定常数 (加在平方均值上，位于平方根内)
    weight_offset: f32 = 0.0, // 权重偏移量：实际缩放系数为 weight + weight_offset (如 Gemma 1~3 的 1 + w 形式)

    pub const default: RmsNormOptions = .{};
    pub fn defaultOptions() RmsNormOptions {
        return .{};
    }
};

/// 池化算子可选参数配置 (Pooling Operator Options, PoolOptions)
pub const PoolOptions = struct {
    stride: ?usize = null, // 池化滑动步幅，为 null 时默认等于池化窗口大小 (pool_size / kernel_size)
    padding: usize = 0, // 边缘填充大小，默认为 0

    pub const default: PoolOptions = .{};

    pub fn defaultOptions() PoolOptions {
        return .{};
    }

    pub fn resolveStride(self: PoolOptions, pool_size: usize) usize {
        return self.stride orelse pool_size;
    }
};

/// 将任意张量标量元素安全转换为非负索引 (`usize`)，并对越界或非有限值返回 `IndexOutOfBounds`
pub inline fn indexScalarToUsize(comptime IdxT: type, raw: IdxT, bound: usize) !usize {
    const idx: usize = switch (@typeInfo(IdxT)) {
        .float => blk: {
            if (!std.math.isFinite(raw) or raw < 0.0) return error.IndexOutOfBounds;
            const u: usize = @intFromFloat(raw);
            if (@as(IdxT, @floatFromInt(u)) != raw) return error.IndexOutOfBounds;
            break :blk u;
        },
        .int, .comptime_int => blk: {
            if (raw < 0) return error.IndexOutOfBounds;
            break :blk @intCast(raw);
        },
        else => @compileError("Index tensor must have numeric element type"),
    };
    if (idx >= bound) return error.IndexOutOfBounds;
    return idx;
}

/// 校验索引切片的每个元素都是 [0, bound) 内的非负整数值 (元素可为任意整数或浮点类型，规则同 `indexScalarToUsize`)，
/// 否则返回 `error.IndexOutOfBounds`；用于在修改任何状态之前提前校验词元 ID 等索引
pub fn checkIndices(indices: anytype, bound: usize) !void {
    const IdxT = std.meta.Elem(@TypeOf(indices));
    for (indices) |raw| _ = try indexScalarToUsize(IdxT, raw, bound);
}

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

        /// 以常量原地填充张量的全部可见元素（跨步视图只写入自身覆盖的元素，不改写父缓冲区的其余部分）
        pub fn fill(self: *Self, val: T) void {
            const count = self.shape.numel();
            if (self.isContiguous() and self.data.len >= count) {
                @memset(self.data[0..count], val);
                return;
            }
            var it = shape_mod.RowIter(1).init(self.shape, .{self.strides});
            const s = it.inner_strides[0];
            while (it.next()) |offs| {
                for (0..it.inner) |i| self.data[offs[0] + i * s] = val;
            }
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
                var coord: [8]usize = @splat(0);
                const len = self.shape.len;
                for (0..out.data.len) |dest_i| {
                    var src_idx: usize = 0;
                    for (0..len) |d| {
                        src_idx += coord[d] * self.strides.dims[d];
                    }
                    out.data[dest_i] = self.data[src_idx];
                    self.shape.incrementCoord(&coord);
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

            // 空切片不引用任何元素：偏移归零，避免累加偏移越过缓冲区末尾
            var is_empty = false;
            for (new_dims[0..self.shape.len]) |dim| {
                if (dim == 0) is_empty = true;
            }

            const out = try allocator.create(Self);
            out.* = Self{
                .data = if (is_empty) self.data[0..0] else self.data[offset..],
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
            var indices: [8]usize = @splat(0);

            for (0..out.data.len) |c_flat| {
                var a_flat: usize = 0;
                var b_flat: usize = 0;
                for (0..rank) |d| {
                    a_flat += indices[d] * a_strides.dims[d];
                    b_flat += indices[d] * b_strides.dims[d];
                }
                out.data[c_flat] = op_fn(self.data[a_flat], other.data[b_flat]);

                target_shape.incrementCoord(&indices);
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
            var coord: [8]usize = @splat(0);
            const len = self.shape.len;
            for (0..elem_count) |_| {
                var src_idx: usize = 0;
                for (0..len) |d| src_idx += coord[d] * self.strides.dims[d];
                if (isTruthyScalar(self.data[src_idx])) return true;
                self.shape.incrementCoord(&coord);
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
            var coord: [8]usize = @splat(0);
            const len = self.shape.len;
            for (0..elem_count) |_| {
                var src_idx: usize = 0;
                for (0..len) |d| src_idx += coord[d] * self.strides.dims[d];
                if (!isTruthyScalar(self.data[src_idx])) return false;
                self.shape.incrementCoord(&coord);
            }
            return true;
        }

        /// 沿指定轴或全局求和：以 64 位浮点数 (f64) 累加后转换回元素类型。
        /// 任意跨步输入都按折叠维度后的逐行跨步遍历器 (`shape.RowIter`) 顺序读取，不做逐元素多维下标换算；
        /// 每个输出元素的加数顺序与沿轴下标递增顺序一致
        pub fn sum(self: *const Self, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*Self {
            if (axis) |ax| {
                if (ax >= self.shape.len) return error.DimensionOutOfBounds;

                var out_shape_dims: [8]usize = @splat(0);
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

                const out = try Self.init(allocator, out_shape_dims[0..out_rank], null);
                errdefer out.deinit(allocator);

                const acc = try allocator.alloc(f64, out.data.len);
                defer allocator.free(acc);
                @memset(acc, 0.0);

                // 累加器在输入形状下的跨度：保留维度的输出布局为行优先连续，被归约轴跨度为 0
                var kept = self.shape;
                kept.dims[ax] = 1;
                var acc_strides = computeContiguousStrides(kept);
                acc_strides.dims[ax] = 0;

                var it = shape_mod.RowIter(2).init(self.shape, .{ self.strides, acc_strides });
                const s_src = it.inner_strides[0];
                const s_acc = it.inner_strides[1];
                while (it.next()) |offs| {
                    for (0..it.inner) |i| {
                        acc[offs[1] + i * s_acc] += convertScalar(f64, T, self.data[offs[0] + i * s_src]);
                    }
                }
                for (out.data, acc) |*o, a| o.* = convertScalar(T, f64, a);
                return out;
            } else {
                var acc: f64 = 0.0;
                const elem_count = self.shape.numel();
                if (self.isContiguous() and self.data.len >= elem_count) {
                    for (self.data[0..elem_count]) |v| acc += convertScalar(f64, T, v);
                } else {
                    var it = shape_mod.RowIter(1).init(self.shape, .{self.strides});
                    const s_src = it.inner_strides[0];
                    while (it.next()) |offs| {
                        for (0..it.inner) |i| acc += convertScalar(f64, T, self.data[offs[0] + i * s_src]);
                    }
                }
                if (keepdims) {
                    const out_shape_dims: [8]usize = @splat(1);
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
                var coord: [8]usize = @splat(0);
                const len = self.shape.len;
                for (0..out.data.len) |dest_i| {
                    var src_idx: usize = 0;
                    for (0..len) |d| {
                        src_idx += coord[d] * self.strides.dims[d];
                    }
                    out.data[dest_i] = convertScalar(DestT, T, self.data[src_idx]);
                    self.shape.incrementCoord(&coord);
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

// ============================================================================
// 词元采样类型 (Token Range & Sampling Options)
// ============================================================================

/// 词元 ID 闭区间 [first, last]
pub const TokenRange = struct {
    first: usize,
    last: usize,

    /// 只包含单个词元的区间
    pub fn single(id: usize) TokenRange {
        return .{ .first = id, .last = id };
    }

    pub fn contains(self: TokenRange, v: usize) bool {
        return v >= self.first and v <= self.last;
    }

    /// 是否与半开区间 [begin, end) 相交
    pub fn intersects(self: TokenRange, begin: usize, end: usize) bool {
        return begin <= self.last and end > self.first;
    }
};

/// 采样候选词元元组
pub const Candidate = struct {
    id: u32,
    logit: f32,

    pub fn before(a: Candidate, b: Candidate) bool {
        return a.logit > b.logit or (a.logit == b.logit and a.id < b.id);
    }

    pub fn lessThan(_: void, a: Candidate, b: Candidate) bool {
        return a.before(b);
    }
};

pub const bounded_candidate_count: usize = 1024;

/// 采样选项 (Sampling Options)
pub const SamplingOptions = struct {
    temperature: f32 = 1.0,
    top_k: usize = 0,
    top_p: f32 = 1.0,
    repetition_penalty: f32 = 1.0,
    repetition_window: ?usize = null,
    seed: u64 = 0,

    pub const greedy: SamplingOptions = .{ .temperature = 0.0 };

    pub fn defaultOptions() SamplingOptions {
        return .{};
    }

    pub fn isGreedy(self: SamplingOptions) bool {
        return !(self.temperature > 0.0) or self.top_k == 1;
    }

    pub fn isPlainGreedy(self: SamplingOptions) bool {
        const active = self.repetition_penalty != 1.0 and !std.math.isNan(self.repetition_penalty) and self.repetition_penalty > 0;
        return self.isGreedy() and !active;
    }

    pub fn candidateCount(self: SamplingOptions, vocab_size: usize) usize {
        const k: usize = if (self.isGreedy()) 1 else if (self.top_k > 0) self.top_k else bounded_candidate_count;
        return @max(1, @min(k, vocab_size));
    }

    pub fn needsLogNormalizer(self: SamplingOptions) bool {
        return !self.isGreedy() and self.top_k == 0 and self.top_p < 1.0;
    }

    pub fn recentTokens(self: SamplingOptions, tokens: []const usize, default_window: usize) []const usize {
        const win = self.repetition_window orelse default_window;
        return if (tokens.len > win) tokens[tokens.len - win ..] else tokens;
    }
};

pub fn isExcluded(excluded: []const TokenRange, v: usize) bool {
    for (excluded) |r| {
        if (r.contains(v)) return true;
    }
    return false;
}

pub fn anyExcluded(excluded: []const TokenRange, begin: usize, end: usize) bool {
    for (excluded) |r| {
        if (r.intersects(begin, end)) return true;
    }
    return false;
}

pub fn penaltyActive(repetition_penalty: f32) bool {
    return repetition_penalty > 1.0;
}

pub fn applyRepetitionPenalty(logit: f32, repetition_penalty: f32) f32 {
    return if (logit > 0.0) logit / repetition_penalty else logit * repetition_penalty;
}

pub const bf16_neg_inf_bits: u16 = 0xFF80;
pub const bf16_zero_bits: u16 = 0x0000;

pub fn fillExclusionMaskBf16(out: []u16, excluded: []const TokenRange) void {
    @memset(out, bf16_zero_bits);
    for (excluded) |r| {
        if (r.first >= out.len) continue;
        const stop = @min(r.last, out.len - 1);
        @memset(out[r.first .. stop + 1], bf16_neg_inf_bits);
    }
}

pub const penalty_bitset_words: usize = 4096;
pub const penalty_window_tokens: usize = penalty_bitset_words * 64;

pub fn argmaxWithPenalty(
    logits_slice: []const f32,
    vocab_size: usize,
    excluded: []const TokenRange,
    recent_tokens: []const usize,
    repetition_penalty: f32,
) usize {
    std.debug.assert(logits_slice.len >= vocab_size);
    const apply_penalty = penaltyActive(repetition_penalty);
    const V = @Vector(16, f32);

    var bitset: [penalty_bitset_words]u64 = undefined;
    var max_val: f32 = -std.math.inf(f32);
    var max_id: usize = 0;

    var win_base: usize = 0;
    while (win_base < vocab_size) : (win_base += penalty_window_tokens) {
        const win_end = @min(win_base + penalty_window_tokens, vocab_size);
        const words = bitset[0 .. (win_end - win_base + 63) / 64];
        @memset(words, 0);
        if (apply_penalty) {
            for (recent_tokens) |rec| {
                if (rec >= win_base and rec < win_end) {
                    const off = rec - win_base;
                    words[off / 64] |= @as(u64, 1) << @intCast(off % 64);
                }
            }
        }

        for (words, 0..) |word, w| {
            const base_v = win_base + w * 64;
            const end_v = @min(base_v + 64, win_end);
            const check_excluded = anyExcluded(excluded, base_v, end_v);

            if (word == 0 and !check_excluded and end_v - base_v == 64) {
                const chunk: *const [64]f32 = @ptrCast(logits_slice[base_v .. base_v + 64]);
                var m_vec: V = @splat(-std.math.inf(f32));
                inline for (0..4) |unroll| {
                    const sub: *const [16]f32 = @ptrCast(chunk[unroll * 16 ..][0..16]);
                    m_vec = @max(m_vec, @as(V, sub.*));
                }
                const block_max = @reduce(.Max, m_vec);
                if (block_max > max_val) {
                    for (chunk, 0..) |val, i| {
                        if (val > max_val) {
                            max_val = val;
                            max_id = base_v + i;
                        }
                    }
                }
                continue;
            }

            var bit_cursor = word;
            for (base_v..end_v) |v| {
                const bit = bit_cursor & 1;
                bit_cursor >>= 1;
                if (check_excluded and isExcluded(excluded, v)) continue;
                var val = logits_slice[v];
                if (std.math.isNan(val)) continue;
                if (bit == 1) val = applyRepetitionPenalty(val, repetition_penalty);
                if (val > max_val) {
                    max_val = val;
                    max_id = v;
                }
            }
        }
    }
    return max_id;
}

pub fn applyRepetitionPenaltyInPlace(
    allocator: std.mem.Allocator,
    logits: []f32,
    recent_tokens: []const usize,
    repetition_penalty: f32,
) !void {
    if (!penaltyActive(repetition_penalty) or recent_tokens.len == 0) return;
    var seen = try std.DynamicBitSetUnmanaged.initEmpty(allocator, logits.len);
    defer seen.deinit(allocator);
    for (recent_tokens) |tok| {
        if (tok >= logits.len or seen.isSet(tok)) continue;
        seen.set(tok);
        logits[tok] = applyRepetitionPenalty(logits[tok], repetition_penalty);
    }
}

pub fn maskExcludedInPlace(logits: []f32, excluded: []const TokenRange) void {
    for (excluded) |r| {
        if (r.first >= logits.len) continue;
        const stop = @min(r.last, logits.len - 1);
        @memset(logits[r.first .. stop + 1], -std.math.inf(f32));
    }
}

pub fn usableLogit(x: f32) bool {
    return !std.math.isNan(x) and x > -std.math.inf(f32);
}

pub const Distribution = struct {
    candidates: []const Candidate,
    inv_temperature: f32,
    max_scaled: f32,
    mass: f64,
    covers_nucleus: bool,

    pub fn weight(self: Distribution, cand: Candidate) f64 {
        return @exp(@as(f64, cand.logit * self.inv_temperature - self.max_scaled));
    }

    pub fn probability(self: Distribution, id: u32) f64 {
        if (self.mass <= 0.0) return 0.0;
        for (self.candidates) |cand| {
            if (cand.id == id) return self.weight(cand) / self.mass;
        }
        return 0.0;
    }

    pub fn sample(self: Distribution, random: std.Random) u32 {
        return self.sampleExcluding(random, null);
    }

    pub fn sampleExcluding(self: Distribution, random: std.Random, excluded: ?u32) u32 {
        if (self.candidates.len == 0) return 0;
        var total = self.mass;
        var remaining: usize = 0;
        var first_other: ?u32 = null;
        var last_other: u32 = self.candidates[0].id;
        for (self.candidates) |cand| {
            if (excluded) |x| {
                if (cand.id == x) {
                    total -= self.weight(cand);
                    continue;
                }
            }
            remaining += 1;
            if (first_other == null) first_other = cand.id;
            last_other = cand.id;
        }
        const fallback = first_other orelse return self.candidates[0].id;
        if (remaining == 1 or !(total > 0.0)) return fallback;

        const target = random.float(f64) * total;
        var cum: f64 = 0.0;
        for (self.candidates) |cand| {
            if (excluded) |x| {
                if (cand.id == x) continue;
            }
            cum += self.weight(cand);
            if (cum > target) return cand.id;
        }
        return last_other;
    }
};

pub fn buildDistribution(candidates: []Candidate, options: SamplingOptions, log_normalizer: ?f32) Distribution {
    var n: usize = 0;
    for (candidates) |cand| {
        if (!usableLogit(cand.logit)) continue;
        candidates[n] = cand;
        n += 1;
    }
    const sorted = candidates[0..n];
    std.mem.sortUnstable(Candidate, sorted, {}, Candidate.lessThan);

    if (n == 0) return .{ .candidates = sorted, .inv_temperature = 1.0, .max_scaled = 0.0, .mass = 0.0, .covers_nucleus = true };
    if (options.isGreedy()) {
        return .{ .candidates = sorted[0..1], .inv_temperature = 1.0, .max_scaled = sorted[0].logit, .mass = 1.0, .covers_nucleus = true };
    }

    const kept_k = if (options.top_k > 0) sorted[0..@min(options.top_k, n)] else sorted;
    const inv_t = 1.0 / options.temperature;
    var dist = Distribution{
        .candidates = kept_k,
        .inv_temperature = inv_t,
        .max_scaled = kept_k[0].logit * inv_t,
        .mass = 0.0,
        .covers_nucleus = true,
    };

    var candidate_mass: f64 = 0.0;
    for (kept_k) |cand| candidate_mass += dist.weight(cand);
    const use_normalizer = options.top_k == 0 and log_normalizer != null;
    const total: f64 = if (use_normalizer)
        @max(candidate_mass, @exp(@as(f64, log_normalizer.? - dist.max_scaled)))
    else
        candidate_mass;

    if (options.top_p < 1.0) {
        const threshold = @as(f64, options.top_p) * total;
        var cum: f64 = 0.0;
        var kept: usize = 0;
        while (kept < kept_k.len) : (kept += 1) {
            if (kept > 0 and cum >= threshold) break;
            cum += dist.weight(kept_k[kept]);
        }
        dist.candidates = kept_k[0..kept];
        dist.mass = cum;
        dist.covers_nucleus = cum >= threshold;
    } else {
        dist.mass = candidate_mass;
        dist.covers_nucleus = !use_normalizer;
    }
    return dist;
}

pub const DraftVerdict = union(enum) {
    accepted,
    rejected: u32,
};

pub fn verifyDraftToken(dist: Distribution, draft: u32, random: std.Random) DraftVerdict {
    const p = dist.probability(draft);
    if (p >= 1.0) return .accepted;
    if (p > 0.0 and random.float(f64) < p) return .accepted;
    return .{ .rejected = dist.sampleExcluding(random, draft) };
}
