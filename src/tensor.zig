const std = @import("std");
const autodiff = @import("autodiff.zig");
const Op = autodiff.Op;
const c = @import("cblas.zig");

extern fn erff(x: f32) f32;

// ============================================================================
// 1. 维度与形状控制（Shape & Strides Meta-data）
// ============================================================================

/// 多维张量的形状描述体（Shape）
/// 为避免动态内存分配带来的开销，本框架采用静态数组 `[8]usize` 存储各维度大小，最多支持 8 维张量。
pub const Shape = struct {
    dims: [8]usize, // 存储每一维度大小的静态数组，未使用的维度默认为 0
    len: usize,     // 张量的维度个数（Rank，例如 2D 矩阵的 Rank 为 2）

    /// 从切片安全初始化 Shape，超过 8 维返回 error.MaxDimensionsExceeded
    pub fn fromSlice(shape_slice: []const usize) !Shape {
        if (shape_slice.len > 8) return error.MaxDimensionsExceeded;
        var self = Shape{
            .dims = [_]usize{0} ** 8,
            .len = shape_slice.len,
        };
        for (shape_slice, 0..) |dim, i| {
            self.dims[i] = dim;
        }
        return self;
    }

    /// 根据动态传入的切片初始化静态 Shape 结构体（若超过 8 维触发 panic）
    pub fn init(shape_slice: []const usize) Shape {
        return fromSlice(shape_slice) catch |err| switch (err) {
            error.MaxDimensionsExceeded => @panic("Shape.init: maximum dimensions (8) exceeded"),
        };
    }


    /// 校验两个 Shape 是否完全相等（维度个数及每一维大小都匹配）
    pub fn eq(self: Shape, other: Shape) bool {
        if (self.len != other.len) return false;
        for (0..self.len) |i| {
            if (self.dims[i] != other.dims[i]) return false;
        }
        return true;
    }
};

/// 计算行优先（Row-Major）布局下的连续跨度（Contiguous Strides）
/// 数学原理：
/// 假设张量逻辑形状为 [D_0, D_1, ..., D_{n-1}]，对应的行优先连续跨度为 [S_0, S_1, ..., S_{n-1}]。
/// 则任一多维索引 [i_0, i_1, ..., i_{n-1}] 在一维物理缓冲区中的扁平索引偏移计算公式为：
///     FlatIndex = sum_{k=0}^{n-1} (i_k * S_k)
/// 其中跨度递推公式为：
///     S_{n-1} = 1
///     S_k     = S_{k+1} * D_{k+1}  (0 <= k < n-1)
pub fn computeContiguousStrides(shape: Shape) Shape {
    var strides = Shape{
        .dims = [_]usize{0} ** 8,
        .len = shape.len,
    };
    if (shape.len == 0) return strides;

    var s: usize = 1;
    var i: usize = shape.len - 1;
    while (true) {
        strides.dims[i] = s;
        s *= shape.dims[i];
        if (i == 0) break;
        i -= 1;
    }
    return strides;
}

/// 交换指定维度的形状（通常在转置算子中配合 strides 交换实现快速视图变换）
pub fn transposeShape(shape: Shape, dim0: usize, dim1: usize) Shape {
    var new_shape = shape;
    const tmp = new_shape.dims[dim0];
    new_shape.dims[dim0] = new_shape.dims[dim1];
    new_shape.dims[dim1] = tmp;
    return new_shape;
}

/// 通用 NumPy 风格多维形状广播对齐算法 (Broadcasting Shape Inference)
/// 从右向左（尾部对齐，Trailing Dimensions）逐维比对：
/// 1. 若两维度大小相等，输出该维度大小；
/// 2. 若其中一个维度为 1，输出另一个维度的较大值；
/// 3. 若其中一个张量维数较少，高位缺失维度视作 1 并对齐；
/// 4. 若两维度不同且均不为 1，则判定形状不兼容，返回 error.IncompatibleBroadcastShapes。
pub fn broadcastShapes(shape1: Shape, shape2: Shape) !Shape {
    const len1 = shape1.len;
    const len2 = shape2.len;
    const out_len = @max(len1, len2);
    if (out_len > 8) return error.MaxDimensionsExceeded;

    var out_shape = Shape{
        .dims = [_]usize{0} ** 8,
        .len = out_len,
    };

    for (0..out_len) |k| {
        const d1 = if (k < len1) shape1.dims[len1 - 1 - k] else 1;
        const d2 = if (k < len2) shape2.dims[len2 - 1 - k] else 1;

        if (d1 == d2) {
            out_shape.dims[out_len - 1 - k] = d1;
        } else if (d1 == 1) {
            out_shape.dims[out_len - 1 - k] = d2;
        } else if (d2 == 1) {
            out_shape.dims[out_len - 1 - k] = d1;
        } else {
            return error.IncompatibleBroadcastShapes;
        }
    }
    return out_shape;
}

/// 计算输入张量在目标广播形状下的虚拟跨度 (Broadcast Strides)
/// 算法原理：
/// 若某维度大小为 1（或高位缺失），则在遍历该维时不移动底层数据指针，即对应步长（stride）设为 0。
/// 这使得多维索引计算可以通过统一的跨度点积直接映射到输入张量的真实物理偏移，无需物理复制内存。
pub fn computeBroadcastStrides(src_shape: Shape, src_strides: Shape, target_shape: Shape) Shape {
    var b_strides = Shape{
        .dims = [_]usize{0} ** 8,
        .len = target_shape.len,
    };
    const target_len = target_shape.len;
    const src_len = src_shape.len;

    for (0..target_len) |i| {
        const k = target_len - 1 - i;
        if (k < src_len) {
            const src_dim_idx = src_len - 1 - k;
            if (src_shape.dims[src_dim_idx] == 1) {
                b_strides.dims[i] = 0;
            } else {
                b_strides.dims[i] = src_strides.dims[src_dim_idx];
            }
        } else {
            b_strides.dims[i] = 0;
        }
    }
    return b_strides;
}

/// 底层高效通用广播二元算子执行引擎
pub fn broadcastBinaryOpRaw(
    C_data: []f32,
    C_shape: Shape,
    A_data: []const f32,
    A_shape: Shape,
    A_strides: Shape,
    B_data: []const f32,
    B_shape: Shape,
    B_strides: Shape,
    comptime op: fn (f32, f32) f32,
) void {
    // 快速路径：若形状完全相同且连续，直接单层循环 SIMD 扁平迭代
    if (A_shape.eq(B_shape)) {
        for (C_data, A_data, B_data) |*c_val, a_val, b_val| {
            c_val.* = op(a_val, b_val);
        }
        return;
    }

    // 广播路径：基于步长为 0 的虚拟映射执行多维坐标遍历
    const a_strides = computeBroadcastStrides(A_shape, A_strides, C_shape);
    const b_strides = computeBroadcastStrides(B_shape, B_strides, C_shape);
    const len = C_shape.len;
    var coord = [_]usize{0} ** 8;

    for (C_data) |*c_val| {
        var a_idx: usize = 0;
        var b_idx: usize = 0;
        for (0..len) |d| {
            a_idx += coord[d] * a_strides.dims[d];
            b_idx += coord[d] * b_strides.dims[d];
        }

        c_val.* = op(A_data[a_idx], B_data[b_idx]);

        var d = len;
        while (d > 0) {
            d -= 1;
            coord[d] += 1;
            if (coord[d] < C_shape.dims[d]) {
                break;
            }
            coord[d] = 0;
        }
    }
}



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


// ============================================================================
// 3. 基础张量（Tensor）核心定义与元数据
// ============================================================================

/// 张量（Tensor）结构体：承载机器学习网络中所有物理数据与流转拓扑信息
pub const Tensor = struct {
    data: []f32,          // 前向传播的数据缓冲区（行优先存储的一维切片）
    grad: []f32,          // 反向传播的梯度缓冲区（与 data 形状一致，不需梯度的节点可为空）
    shape: Shape,         // 逻辑形状
    strides: Shape,       // 各维度的跨度步长（用于非连续张量及快速视图映射）
    requires_grad: bool,  // 是否需要求梯度（如模型参数为 true，输入数据为 false）
    creator: ?*Op,        // 产生此张量的算子节点（前向图中的父节点，用于追踪计算路径）
    is_view: bool = false, // 是否为零拷贝视图切片（若为 true，deinit 时不释放 data/grad）
    is_custom_initialized: bool = false, // 是否已被层专属自定义初始化 (避免被 Graph 自动初始化重写)
    name: ?[]const u8 = null, // 可选张量调试名称 (如 "fc1.weight", "conv1.bias")
    name_buf: [64]u8 = undefined,

    /// 设置张量的人类可读名称 (用于 Graph.printInitReport 等调试报告)
    pub fn setName(self: *Tensor, name: []const u8) void {
        self.name = name;
    }

    /// 使用格式化模板设置张量的人类可读名称
    pub fn setNameFormatted(self: *Tensor, comptime fmt: []const u8, args: anytype) void {
        if (std.fmt.bufPrint(&self.name_buf, fmt, args)) |s| {
            self.name = s;
        } else |_| {
            self.name = "truncated_name";
        }
    }

    /// 获取张量的人类可读名称
    pub fn getName(self: *const Tensor) ?[]const u8 {
        return self.name;
    }



    // 将梯度缓冲区全部清零，通常在每个 batch 反向传播前调用
    pub fn zeroGrad(self: *Tensor) void {
        if (self.requires_grad) {
            @memset(self.grad, 0.0);
        }
    }

    // 安全获取多维索引对应的扁平化索引（带维度及边界校验）
    pub fn getFlatIndexChecked(self: Tensor, indices: []const usize) !usize {
        if (indices.len != self.shape.len) return error.DimensionMismatch;
        var flat_idx: usize = 0;
        for (indices, 0..) |idx, i| {
            if (idx >= self.shape.dims[i]) return error.IndexOutOfBounds;
            flat_idx += idx * self.strides.dims[i];
        }
        return flat_idx;
    }

    // 获取多维索引对应的扁平化索引
    pub fn getFlatIndex(self: Tensor, indices: []const usize) usize {
        return self.getFlatIndexChecked(indices) catch |err| switch (err) {
            error.DimensionMismatch => @panic("getFlatIndex: dimension mismatch"),
            error.IndexOutOfBounds => @panic("getFlatIndex: index out of bounds"),
        };
    }

    // 安全获取特定多维索引处的值（带边界校验）
    pub fn getChecked(self: Tensor, indices: []const usize) !f32 {
        const flat_idx = try self.getFlatIndexChecked(indices);
        return self.data[flat_idx];
    }

    // 安全设置特定多维索引处的值（带边界校验）
    pub fn setChecked(self: *Tensor, indices: []const usize, val: f32) !void {
        const flat_idx = try self.getFlatIndexChecked(indices);
        self.data[flat_idx] = val;
    }

    // 获取特定多维索引处的值
    pub fn get(self: Tensor, indices: []const usize) f32 {
        return self.data[self.getFlatIndex(indices)];
    }

    // 设置特定多维索引处的值
    pub fn set(self: *Tensor, indices: []const usize, val: f32) void {
        self.data[self.getFlatIndex(indices)] = val;
    }

    // 获取特定多维索引处的梯度值
    pub fn getGrad(self: Tensor, indices: []const usize) f32 {
        std.debug.assert(self.requires_grad);
        return self.grad[self.getFlatIndex(indices)];
    }

    // 设置特定多维索引处的梯度值
    pub fn setGrad(self: *Tensor, indices: []const usize, val: f32) void {
        std.debug.assert(self.requires_grad);
        self.grad[self.getFlatIndex(indices)] = val;
    }

    // 美化输出 N 维 Tensor 的多维表示
    pub fn print(self: Tensor) void {
        self.printND(0, 0);
        std.debug.print("\n", .{});
    }

    fn printND(self: Tensor, dim: usize, offset: usize) void {
        if (self.shape.len == 0) {
            std.debug.print("{d:.4}", .{self.data[offset]});
            return;
        }
        if (dim == self.shape.len - 1) {
            std.debug.print("[", .{});
            const size = self.shape.dims[dim];
            const stride = self.strides.dims[dim];
            for (0..size) |i| {
                std.debug.print("{d:.4}", .{self.data[offset + i * stride]});
                if (i < size - 1) {
                    std.debug.print(", ", .{});
                }
            }
            std.debug.print("]", .{});
            return;
        }

        std.debug.print("[", .{});
        const size = self.shape.dims[dim];
        const stride = self.strides.dims[dim];
        for (0..size) |i| {
            self.printND(dim + 1, offset + i * stride);
            if (i < size - 1) {
                std.debug.print(",\n", .{});
                for (0..dim + 1) |_| {
                    std.debug.print(" ", .{});
                }
            }
        }
        std.debug.print("]", .{});
    }

    // ============================================================================
    // Direct tensor operations (eager or graph-backed)
    // ============================================================================
    pub fn matmul(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.matmul(self, other);
        }
        if (self.shape.len != 2 or other.shape.len != 2) {
            return error.IncompatibleDimensions;
        }
        if (self.shape.dims[1] != other.shape.dims[0]) {
            return error.ShapeMismatch;
        }
        const M = self.shape.dims[0];
        const K = self.shape.dims[1];
        const N = other.shape.dims[1];
        const C = try zeros(allocator, &.{M, N});
        c.cblas_sgemm(
            c.CblasRowMajor,
            c.CblasNoTrans,
            c.CblasNoTrans,
            @intCast(M),
            @intCast(N),
            @intCast(K),
            1.0,
            self.data.ptr,
            @intCast(K),
            other.data.ptr,
            @intCast(N),
            0.0,
            C.data.ptr,
            @intCast(N),
        );
        return C;
    }

    // 偏置相加算子：直接复用多维广播加法
    pub fn addBias(self: *Tensor, bias: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        return self.add(bias, allocator, graph);
    }

    pub fn mulScalar(self: *Tensor, val: f32, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.mulScalar(self, val);
        }
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, s_val| {
            c_val.* = s_val * val;
        }
        return C;
    }

    pub fn addScalar(self: *Tensor, val: f32, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.addScalar(self, val);
        }
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, s_val| {
            c_val.* = s_val + val;
        }
        return C;
    }

    pub fn subScalar(self: *Tensor, val: f32, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.subScalar(self, val);
        }
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, s_val| {
            c_val.* = s_val - val;
        }
        return C;
    }

    pub fn divScalar(self: *Tensor, val: f32, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.divScalar(self, val);
        }
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, s_val| {
            c_val.* = s_val / val;
        }
        return C;
    }

    fn addOp(a: f32, b: f32) f32 { return a + b; }
    fn subOp(a: f32, b: f32) f32 { return a - b; }
    fn mulOp(a: f32, b: f32) f32 { return a * b; }
    fn divOp(a: f32, b: f32) f32 { return a / b; }

    /// 通用多维广播加法：C = self + other
    pub fn add(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.add(self, other);
        }
        const out_shape = try broadcastShapes(self.shape, other.shape);
        const C = try zeros(allocator, out_shape.dims[0..out_shape.len]);
        broadcastBinaryOpRaw(C.data, C.shape, self.data, self.shape, self.strides, other.data, other.shape, other.strides, addOp);
        return C;
    }

    /// 通用多维广播减法：C = self - other
    pub fn sub(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.sub(self, other);
        }
        const out_shape = try broadcastShapes(self.shape, other.shape);
        const C = try zeros(allocator, out_shape.dims[0..out_shape.len]);
        broadcastBinaryOpRaw(C.data, C.shape, self.data, self.shape, self.strides, other.data, other.shape, other.strides, subOp);
        return C;
    }

    /// 通用多维广播乘法 (Hadamard 积)：C = self * other
    pub fn mul(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.mul(self, other);
        }
        const out_shape = try broadcastShapes(self.shape, other.shape);
        const C = try zeros(allocator, out_shape.dims[0..out_shape.len]);
        broadcastBinaryOpRaw(C.data, C.shape, self.data, self.shape, self.strides, other.data, other.shape, other.strides, mulOp);
        return C;
    }

    /// 通用多维广播除法：C = self / other
    pub fn div(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.div(self, other);
        }
        const out_shape = try broadcastShapes(self.shape, other.shape);
        const C = try zeros(allocator, out_shape.dims[0..out_shape.len]);
        broadcastBinaryOpRaw(C.data, C.shape, self.data, self.shape, self.strides, other.data, other.shape, other.strides, divOp);
        return C;
    }

    pub fn silu(self: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.silu(self);
        }
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, a_val| {
            const sig = if (a_val >= 0.0) 1.0 / (1.0 + @exp(-a_val)) else @exp(a_val) / (1.0 + @exp(a_val));
            c_val.* = a_val * sig;
        }
        return C;
    }

    pub fn relu(self: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.relu(self);
        }
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        const total = self.data.len;
        for (0..total) |i| {
            C.data[i] = if (self.data[i] > 0.0) self.data[i] else 0.0;
        }
        return C;
    }

    pub fn gelu(self: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.gelu(self);
        }
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        const total = self.data.len;
        const sqrt_2: f32 = @sqrt(@as(f32, 2.0));
        for (0..total) |i| {
            const x = self.data[i];
            const erf_val = erff(x / sqrt_2);
            C.data[i] = 0.5 * x * (1.0 + erf_val);
        }
        return C;
    }

    pub fn sigmoid(self: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.sigmoid(self);
        }
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, a_val| {
            if (a_val >= 0.0) {
                c_val.* = 1.0 / (1.0 + @exp(-a_val));
            } else {
                const e = @exp(a_val);
                c_val.* = e / (1.0 + e);
            }
        }
        return C;
    }

    pub fn tanh(self: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.tanh(self);
        }
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, a_val| {
            c_val.* = std.math.tanh(a_val);
        }
        return C;
    }

    pub fn leakyRelu(self: *Tensor, alpha: f32, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.leakyRelu(self, alpha);
        }
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, a_val| {
            c_val.* = if (a_val > 0.0) a_val else alpha * a_val;
        }
        return C;
    }

    pub fn bceWithLogitsLoss(self: *Tensor, targets: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.bceWithLogitsLoss(self, targets);
        }
        const loss = try zeros(allocator, &.{ 1, 1 });
        const N = self.data.len;
        if (N != targets.data.len) return error.ShapeMismatch;
        var total_loss: f32 = 0.0;
        for (0..N) |i| {
            const x = self.data[i];
            const y = targets.data[i];
            const max_x = @max(x, 0.0);
            const abs_x = @abs(x);
            total_loss += max_x - x * y + @log(1.0 + @exp(-abs_x));
        }
        loss.data[0] = total_loss / @as(f32, @floatFromInt(N));
        return loss;
    }

    pub fn bceLoss(self: *Tensor, targets: *Tensor, eps: f32, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.bceLoss(self, targets, eps);
        }
        const loss = try zeros(allocator, &.{ 1, 1 });
        const N = self.data.len;
        if (N != targets.data.len) return error.ShapeMismatch;
        var total_loss: f32 = 0.0;
        for (0..N) |i| {
            const p = self.data[i];
            const y = targets.data[i];
            const p_clip = @max(p, eps);
            const one_minus_p_clip = @max(1.0 - p, eps);
            total_loss += -(y * @log(p_clip) + (1.0 - y) * @log(one_minus_p_clip));
        }
        loss.data[0] = total_loss / @as(f32, @floatFromInt(N));
        return loss;
    }

    pub fn fillNormal(self: *Tensor, random: std.Random, mean_val: f32, stddev: f32) void {
        var i: usize = 0;
        const len = self.data.len;
        while (i < len) {
            var u_1: f32 = random.float(f32);
            while (u_1 == 0.0) {
                u_1 = random.float(f32);
            }
            const u_2 = random.float(f32);
            const z0 = @sqrt(-2.0 * @log(u_1)) * @cos(2.0 * std.math.pi * u_2);
            self.data[i] = mean_val + z0 * stddev;
            i += 1;
            if (i < len) {
                const z1 = @sqrt(-2.0 * @log(u_1)) * @sin(2.0 * std.math.pi * u_2);
                self.data[i] = mean_val + z1 * stddev;
                i += 1;
            }
        }
    }

    pub fn fillUniform(self: *Tensor, random: std.Random, min_val: f32, max_val: f32) void {
        const range = max_val - min_val;
        for (self.data) |*val| {
            val.* = min_val + random.float(f32) * range;
        }
    }

    pub fn softmaxCrossEntropy(self: *Tensor, targets: []const u8, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.softmaxCrossEntropy(self, targets);
        }
        if (self.shape.len != 2) return error.IncompatibleDimensions;
        const loss = try zeros(allocator, &.{1, 1});
        const B = self.shape.dims[0];
        const N = self.shape.dims[1];
        if (B != targets.len) return error.ShapeMismatch;

        var loss_sum: f32 = 0.0;
        for (0..B) |i| {
            const logits_row = self.data[i * N .. (i + 1) * N];
            var max_val = logits_row[0];
            for (logits_row[1..]) |val| {
                if (val > max_val) max_val = val;
            }

            var exp_sum: f32 = 0.0;
            for (logits_row) |val| {
                exp_sum += @exp(val - max_val);
            }

            const label = targets[i];
            if (label >= N) return error.IndexOutOfBounds;
            const prob = @exp(logits_row[label] - max_val) / exp_sum;
            const clipped = @max(prob, 1e-15);
            loss_sum += -@log(clipped);
        }
        loss.data[0] = loss_sum / @as(f32, @floatFromInt(B));
        return loss;
    }

    pub fn sigmoidCrossEntropy(self: *Tensor, targets: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.sigmoidCrossEntropy(self, targets);
        }
        const loss = try zeros(allocator, &.{1, 1});
        const N = self.data.len;
        if (N != targets.data.len) return error.ShapeMismatch;

        var loss_sum: f32 = 0.0;
        for (0..N) |i| {
            const x = self.data[i];
            const y = targets.data[i];
            const max_val = @max(x, 0.0);
            const abs_val = @abs(x);
            loss_sum += max_val - x * y + @log(1.0 + @exp(-abs_val));
        }
        loss.data[0] = loss_sum / @as(f32, @floatFromInt(N));
        return loss;
    }

    pub fn l2Loss(self: *Tensor, lambda: f32, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.l2Loss(self, lambda);
        }
        const loss = try zeros(allocator, &.{1, 1});
        var sum_sq: f32 = 0.0;
        for (self.data) |v| {
            sum_sq += v * v;
        }
        loss.data[0] = 0.5 * lambda * sum_sq;
        return loss;
    }

    pub fn reshape(self: *Tensor, new_shape_slice: []const usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.reshape(self, new_shape_slice);
        }
        const shape = try Shape.fromSlice(new_shape_slice);
        const strides = computeContiguousStrides(shape);
        var old_total: usize = 1;
        for (0..self.shape.len) |i| {
            old_total *= self.shape.dims[i];
        }
        var new_total: usize = 1;
        for (new_shape_slice) |dim| {
            new_total *= dim;
        }
        if (old_total != new_total) return error.ShapeMismatch;

        const C = try allocator.create(Tensor);
        C.* = Tensor{
            .data = try allocator.alloc(f32, new_total),
            .grad = &.{},
            .shape = shape,
            .strides = strides,
            .requires_grad = false,
            .creator = null,
        };
        @memcpy(C.data, self.data);
        return C;
    }

    pub fn split(self: *Tensor, num_splits: usize, dim: usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror![]*Tensor {
        return tensorSplit(allocator, self, num_splits, dim, graph);
    }

    pub fn transpose(self: *Tensor, dim0: usize, dim1: usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.transposeND(self, dim0, dim1);
        }
        if (dim0 >= self.shape.len or dim1 >= self.shape.len) {
            return error.DimensionOutOfBounds;
        }


        const shape_trans = transposeShape(self.shape, dim0, dim1);
        const strides_trans = transposeShape(self.strides, dim0, dim1);

        const C_shape = shape_trans;
        const C_strides = computeContiguousStrides(C_shape);

        var total_size: usize = 1;
        for (C_shape.dims[0..C_shape.len]) |dim| {
            total_size *= dim;
        }

        const C = try allocator.create(Tensor);
        C.* = Tensor{
            .data = try allocator.alloc(f32, total_size),
            .grad = &.{},
            .shape = C_shape,
            .strides = C_strides,
            .requires_grad = false,
            .creator = null,
        };

        var indices = [_]usize{0} ** 8;
        const len = C_shape.len;
        for (0..total_size) |dest_flat_idx| {
            var src_flat_idx: usize = 0;
            for (0..len) |d| {
                src_flat_idx += indices[d] * strides_trans.dims[d];
            }
            C.data[dest_flat_idx] = self.data[src_flat_idx];

            var d: usize = len;
            while (d > 0) {
                d -= 1;
                indices[d] += 1;
                if (indices[d] < C_shape.dims[d]) {
                    break;
                }
                indices[d] = 0;
            }
        }
        return C;
    }

    pub fn conv2d(self: *Tensor, weight: *Tensor, bias: ?*Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.conv2d(self, weight, bias);
        }
        if (self.shape.len != 4 or weight.shape.len != 4) {
            return error.IncompatibleDimensions;
        }
        const N = self.shape.dims[0];
        const C_in = self.shape.dims[1];
        const H = self.shape.dims[2];
        const W = self.shape.dims[3];

        const C_out = weight.shape.dims[0];
        if (weight.shape.dims[1] != C_in) return error.ShapeMismatch;
        const KH = weight.shape.dims[2];
        const KW = weight.shape.dims[3];

        if (bias) |b| {
            if (b.shape.len != 1 or b.shape.dims[0] != C_out) return error.ShapeMismatch;
        }

        if (H < KH or W < KW) return error.KernelBiggerThanInput;

        const H_out = H - KH + 1;
        const W_out = W - KW + 1;

        const out = try zeros(allocator, &.{ N, C_out, H_out, W_out });

        const s_n = self.strides.dims[0];
        const s_c = self.strides.dims[1];
        const s_h = self.strides.dims[2];
        const s_w = self.strides.dims[3];

        const w_co = weight.strides.dims[0];
        const w_ci = weight.strides.dims[1];
        const w_kh = weight.strides.dims[2];
        const w_kw = weight.strides.dims[3];

        const o_n = out.strides.dims[0];
        const o_c = out.strides.dims[1];
        const o_h = out.strides.dims[2];
        const o_w = out.strides.dims[3];

        for (0..N) |n| {
            for (0..C_out) |co| {
                const b_val = if (bias) |b| b.data[co] else 0.0;
                for (0..H_out) |h| {
                    for (0..W_out) |w| {
                        var acc: f32 = b_val;
                        for (0..C_in) |ci| {
                            for (0..KH) |kh| {
                                for (0..KW) |kw| {
                                    const input_val = self.data[n * s_n + ci * s_c + (h + kh) * s_h + (w + kw) * s_w];
                                    const weight_val = weight.data[co * w_co + ci * w_ci + kh * w_kh + kw * w_kw];
                                    acc += input_val * weight_val;
                                }
                            }
                        }
                        out.data[n * o_n + co * o_c + h * o_h + w * o_w] = acc;
                    }
                }
            }
        }
        return out;
    }

    pub fn convTranspose2d(
        self: *Tensor,
        weight: *Tensor,
        bias: ?*Tensor,
        stride: usize,
        padding: usize,
        allocator: std.mem.Allocator,
        graph: ?*autodiff.Graph,
    ) anyerror!*Tensor {
        if (graph) |g| {
            return try g.convTranspose2D(self, weight, bias, stride, padding);
        }
        if (self.shape.len != 4 or weight.shape.len != 4) {
            return error.IncompatibleDimensions;
        }
        const N = self.shape.dims[0];
        const C_in = self.shape.dims[1];
        const H_in = self.shape.dims[2];
        const W_in = self.shape.dims[3];

        if (weight.shape.dims[0] != C_in) return error.ShapeMismatch;
        const C_out = weight.shape.dims[1];
        const KH = weight.shape.dims[2];
        const KW = weight.shape.dims[3];

        if (bias) |b| {
            if (b.shape.len != 1 or b.shape.dims[0] != C_out) return error.ShapeMismatch;
        }

        const H_out = (H_in - 1) * stride + KH - 2 * padding;
        const W_out = (W_in - 1) * stride + KW - 2 * padding;

        const out = try zeros(allocator, &.{ N, C_out, H_out, W_out });

        for (0..N) |n| {
            for (0..C_out) |co| {
                const b_val = if (bias) |b| b.data[co] else 0.0;
                for (0..H_out) |h| {
                    for (0..W_out) |w| {
                        out.data[n * (C_out * H_out * W_out) + co * (H_out * W_out) + h * W_out + w] = b_val;
                    }
                }
            }
        }

        for (0..N) |n| {
            for (0..C_in) |ci| {
                for (0..H_in) |h| {
                    for (0..W_in) |w| {
                        const input_val = self.data[n * (C_in * H_in * W_in) + ci * (H_in * W_in) + h * W_in + w];
                        if (input_val == 0.0) continue;

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

                                    const weight_val = weight.data[ci * (C_out * KH * KW) + co * (KH * KW) + kh * KW + kw];
                                    out.data[n * (C_out * H_out * W_out) + co * (H_out * W_out) + out_h * W_out + out_w] += input_val * weight_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        return out;
    }

    pub fn maxpool2d(self: *Tensor, pool_size: usize, stride: usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.maxpool2d(self, pool_size, stride);
        }
        if (self.shape.len != 4) return error.IncompatibleDimensions;
        if (stride == 0 or pool_size == 0) return error.InvalidStride;
        const N = self.shape.dims[0];
        const C = self.shape.dims[1];
        const H = self.shape.dims[2];
        const W = self.shape.dims[3];

        const H_out = H / stride;
        const W_out = W / stride;

        const out = try zeros(allocator, &.{ N, C, H_out, W_out });

        const s_n = self.strides.dims[0];
        const s_c = self.strides.dims[1];
        const s_h = self.strides.dims[2];
        const s_w = self.strides.dims[3];

        const o_n = out.strides.dims[0];
        const o_c = out.strides.dims[1];
        const o_h = out.strides.dims[2];
        const o_w = out.strides.dims[3];

        for (0..N) |n| {
            for (0..C) |c_| {
                for (0..H_out) |h| {
                    for (0..W_out) |w| {
                        var max_val = self.data[n * s_n + c_ * s_c + (h * stride) * s_h + (w * stride) * s_w];
                        for (0..pool_size) |ph| {
                            for (0..pool_size) |pw| {
                                const ih = h * stride + ph;
                                const iw = w * stride + pw;
                                if (ih < H and iw < W) {
                                    const val = self.data[n * s_n + c_ * s_c + ih * s_h + iw * s_w];
                                    if (val > max_val) {
                                        max_val = val;
                                    }
                                }
                            }
                        }
                        out.data[n * o_n + c_ * o_c + h * o_h + w * o_w] = max_val;
                    }
                }
            }
        }
        return out;
    }

    pub fn softmax(self: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.softmax(self);
        }
        const D = self.shape.dims[self.shape.len - 1];
        const M = self.data.len / D;
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);

        for (0..M) |i| {
            const row_in = self.data[i * D .. (i + 1) * D];
            const row_out = C.data[i * D .. (i + 1) * D];

            var max_val = row_in[0];
            for (row_in[1..]) |val| {
                if (val > max_val) max_val = val;
            }

            var exp_sum: f32 = 0.0;
            for (row_in, row_out) |val, *p| {
                const exp_val = @exp(val - max_val);
                p.* = exp_val;
                exp_sum += exp_val;
            }

            for (row_out) |*p| {
                p.* /= exp_sum;
            }
        }
        return C;
    }

    pub fn rmsNorm(self: *Tensor, G: *Tensor, eps: f32, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.rmsNorm(self, G, eps);
        }
        const D = self.shape.dims[self.shape.len - 1];
        const M = self.data.len / D;
        const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);

        for (0..M) |i| {
            const row_in = self.data[i * D .. (i + 1) * D];
            const row_out = Y.data[i * D .. (i + 1) * D];

            var sum_x2: f32 = 0.0;
            for (row_in) |val| {
                sum_x2 += val * val;
            }
            const rms = @sqrt(sum_x2 / @as(f32, @floatFromInt(D)) + eps);

            for (row_in, row_out, G.data) |x_val, *y_val, g_val| {
                y_val.* = x_val / rms * g_val;
            }
        }
        return Y;
    }

    pub fn batchMatMul(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.batchMatMul(self, other);
        }
        if (self.shape.len != 4 or other.shape.len != 4) {
            return error.IncompatibleDimensions;
        }
        if (self.shape.dims[0] != other.shape.dims[0] or self.shape.dims[1] != other.shape.dims[1] or self.shape.dims[3] != other.shape.dims[2]) {
            return error.ShapeMismatch;
        }


        const batch_size = self.shape.dims[0];
        const num_heads = self.shape.dims[1];
        const M = self.shape.dims[2];
        const K = self.shape.dims[3];
        const N = other.shape.dims[3];

        const C = try zeros(allocator, &.{ batch_size, num_heads, M, N });

        const sA_b = self.strides.dims[0];
        const sA_h = self.strides.dims[1];
        const sB_b = other.strides.dims[0];
        const sB_h = other.strides.dims[1];
        const sC_b = C.strides.dims[0];
        const sC_h = C.strides.dims[1];

        for (0..batch_size) |b| {
            for (0..num_heads) |h| {
                const ptrA = self.data.ptr + b * sA_b + h * sA_h;
                const ptrB = other.data.ptr + b * sB_b + h * sB_h;
                const ptrC = C.data.ptr + b * sC_b + h * sC_h;

                c.cblas_sgemm(
                    c.CblasRowMajor,
                    c.CblasNoTrans,
                    c.CblasNoTrans,
                    @intCast(M),
                    @intCast(N),
                    @intCast(K),
                    1.0,
                    ptrA,
                    @intCast(K),
                    ptrB,
                    @intCast(N),
                    0.0,
                    ptrC,
                    @intCast(N),
                );
            }
        }
        return C;
    }

    pub fn embedding(self: *Tensor, indices: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.embedding(self, indices);
        }
        const B = indices.shape.dims[0];
        const T = indices.shape.dims[1];
        const D = self.shape.dims[1];
        const VocabSize = self.shape.dims[0];

        const Y = try zeros(allocator, &.{ B, T, D });

        for (0..B) |b| {
            for (0..T) |t| {
                const idx_f = indices.data[b * T + t];
                const idx = @as(usize, @intFromFloat(idx_f));
                std.debug.assert(idx < VocabSize);

                const w_row = self.data[idx * D .. (idx + 1) * D];
                const y_row = Y.data[(b * T + t) * D .. (b * T + t + 1) * D];
                @memcpy(y_row, w_row);
            }
        }
        return Y;
    }


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
            const C = try zeros(allocator, &.{M, 1});
            for (0..M) |i| {
                var max_val = self.get(&.{i, 0});
                var max_idx: usize = 0;
                for (1..N) |j| {
                    const val = self.get(&.{i, j});
                    if (val > max_val) {
                        max_val = val;
                        max_idx = j;
                    }
                }
                C.data[i] = @as(f32, @floatFromInt(max_idx));
            }
            return C;
        } else if (dim == 0) {
            const C = try zeros(allocator, &.{1, N});
            for (0..N) |j| {
                var max_val = self.get(&.{0, j});
                var max_idx: usize = 0;
                for (1..M) |i| {
                    const val = self.get(&.{i, j});
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
            const C = try zeros(allocator, &.{M, 1});
            for (0..M) |i| {
                var max_val = self.get(&.{i, 0});
                for (1..N) |j| {
                    const val = self.get(&.{i, j});
                    if (val > max_val) max_val = val;
                }
                C.data[i] = max_val;
            }
            return C;
        } else if (dim == 0) {
            const C = try zeros(allocator, &.{1, N});
            for (0..N) |j| {
                var max_val = self.get(&.{0, j});
                for (1..M) |i| {
                    const val = self.get(&.{i, j});
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
        const contig = computeContiguousStrides(self.shape);
        return self.strides.eq(contig);
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
            for (self.data) |v| {
                total += v;
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
        const count = if (axis) |ax| @as(f32, @floatFromInt(self.shape.dims[ax])) else @as(f32, @floatFromInt(self.data.len));
        for (C.data) |*val| {
            val.* /= count;
        }
        return C;
    }

    /// 通用多维张量沿指定轴或全局方差 (Variance Reduction)
    pub fn variance(self: *Tensor, axis: ?usize, keepdims: bool, ddof: usize, allocator: std.mem.Allocator) !*Tensor {
        const mean_t = try self.mean(axis, true, allocator);
        defer free(allocator, mean_t);

        const diff = try self.sub(mean_t, allocator, null);
        defer free(allocator, diff);
        const sq = try diff.mul(diff, allocator, null);
        defer free(allocator, sq);

        const sum_sq = try sq.sum(axis, keepdims, allocator);
        const count = if (axis) |ax| self.shape.dims[ax] else self.data.len;
        if (count <= ddof) {
            free(allocator, sum_sq);
            return error.InvalidDDOF;
        }
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

    /// 依据布尔/条件张量在两个候选张量间进行逐元素选择 (NumPy np.where)
    /// out[i] = if (cond[i] != 0.0) x[i] else y[i]
    pub fn where(cond: *Tensor, x: *Tensor, y: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        const s_xy = try broadcastShapes(x.shape, y.shape);
        const target_shape = try broadcastShapes(cond.shape, s_xy);
        const C = try zeros(allocator, target_shape.dims[0..target_shape.len]);

        if (cond.shape.eq(x.shape) and x.shape.eq(y.shape) and cond.isContiguous() and x.isContiguous() and y.isContiguous()) {
            for (C.data, cond.data, x.data, y.data) |*out_v, c_v, x_v, y_v| {
                out_v.* = if (c_v != 0.0) x_v else y_v;
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

            C.data[c_flat] = if (cond.data[cond_flat] != 0.0) x.data[x_flat] else y.data[y_flat];

            var d: usize = rank;
            while (d > 0) {
                d -= 1;
                indices[d] += 1;
                if (indices[d] < target_shape.dims[d]) break;
                indices[d] = 0;
            }
        }

        return C;
    }

    /// 根据 mask 将满足条件 (mask != 0) 的元素赋值为指定标量值（返回新分配副本）
    pub fn maskedFill(self: *Tensor, mask: *Tensor, value: f32, allocator: std.mem.Allocator) !*Tensor {
        if (!self.shape.eq(mask.shape)) return error.ShapeMismatch;
        const C = try self.clone(allocator);
        for (C.data, mask.data) |*out_v, m_v| {
            if (m_v != 0.0) {
                out_v.* = value;
            }
        }
        return C;
    }

    /// 原地条件掩码填充
    pub fn maskedFill_(self: *Tensor, mask: *Tensor, value: f32) !*Tensor {
        if (self.requires_grad or self.creator != null) return error.InPlaceOpOnGraphTensor;
        if (!self.shape.eq(mask.shape)) return error.ShapeMismatch;
        for (self.data, mask.data) |*out_v, m_v| {
            if (m_v != 0.0) {
                out_v.* = value;
            }
        }
        return self;
    }

    /// 压缩单维度 (Squeeze): 移除所有为 1 的维度，或移除指定为 1 的维度
    pub fn squeeze(self: *Tensor, axis: ?usize, allocator: std.mem.Allocator) !*Tensor {
        if (axis) |ax| {
            if (ax >= self.shape.len) return error.DimensionOutOfBounds;
            if (self.shape.dims[ax] != 1) return error.CannotSqueezeDimension;
            if (self.shape.len == 1) {
                return self.clone(allocator);
            }
            var new_dims = [_]usize{0} ** 8;
            var dest_d: usize = 0;
            for (0..self.shape.len) |d| {
                if (d != ax) {
                    new_dims[dest_d] = self.shape.dims[d];
                    dest_d += 1;
                }
            }
            return self.reshape(new_dims[0..dest_d], allocator, null);
        } else {
            var new_dims = [_]usize{0} ** 8;
            var dest_d: usize = 0;
            for (0..self.shape.len) |d| {
                if (self.shape.dims[d] != 1) {
                    new_dims[dest_d] = self.shape.dims[d];
                    dest_d += 1;
                }
            }
            if (dest_d == 0) {
                new_dims[0] = 1;
                dest_d = 1;
            }
            return self.reshape(new_dims[0..dest_d], allocator, null);
        }
    }

    /// 扩充单维度 (Unsqueeze / expand_dims): 在指定位置插入一个大小为 1 的新维度
    pub fn unsqueeze(self: *Tensor, dim: usize, allocator: std.mem.Allocator) !*Tensor {
        if (dim > self.shape.len) return error.DimensionOutOfBounds;
        if (self.shape.len >= 8) return error.MaxDimensionsExceeded;

        var new_dims = [_]usize{0} ** 8;
        var src_d: usize = 0;
        for (0..(self.shape.len + 1)) |d| {
            if (d == dim) {
                new_dims[d] = 1;
            } else {
                new_dims[d] = self.shape.dims[src_d];
                src_d += 1;
            }
        }
        return self.reshape(new_dims[0..(self.shape.len + 1)], allocator, null);
    }

    /// 跨步零拷贝切片 (Strided View Slicing)
    /// 返回一个共享底层内存缓冲区的零拷贝视图张量 (is_view = true)
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

            var d = len;
            while (d > 0) {
                d -= 1;
                coord[d] += 1;
                if (coord[d] < self.shape.dims[d]) break;
                coord[d] = 0;
            }
        }
        return out;
    }

    /// 元素截断操作 (Clip): 将张量元素限制在 [min_val, max_val] 之间
    pub fn clip(self: *Tensor, min_val: f32, max_val: f32, allocator: std.mem.Allocator) !*Tensor {
        if (min_val > max_val) return error.InvalidRange;
        const out = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        if (self.isContiguous()) {
            for (self.data, out.data) |x, *y| {
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

        for (0..self.data.len) |_| {
            var src_idx: usize = 0;
            for (0..len) |d| {
                src_idx += coord[d] * self.strides.dims[d];
            }
            if (self.data[src_idx] != 0.0) {
                count += 1;
            }
            var d = len;
            while (d > 0) {
                d -= 1;
                coord[d] += 1;
                if (coord[d] < self.shape.dims[d]) break;
                coord[d] = 0;
            }
        }

        const out = try GenericTensor(usize).init(allocator, &.{ count, len }, null);
        errdefer out.deinit(allocator);

        @memset(&coord, 0);
        var row: usize = 0;
        for (0..self.data.len) |_| {
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
            var d = len;
            while (d > 0) {
                d -= 1;
                coord[d] += 1;
                if (coord[d] < self.shape.dims[d]) break;
                coord[d] = 0;
            }
        }
        return out;
    }

    /// 将当前 Tensor 转换为泛型张量 GenericTensor(DestT)
    pub fn to(self: Tensor, comptime DestT: type, allocator: std.mem.Allocator) !*GenericTensor(DestT) {
        const out = try GenericTensor(DestT).init(allocator, self.shape.dims[0..self.shape.len], null);
        errdefer out.deinit(allocator);

        if (self.isContiguous()) {
            for (self.data, 0..) |val, i| {
                out.data[i] = convertScalar(DestT, f32, val);
            }
        } else {
            var coord = [_]usize{0} ** 8;
            const len = self.shape.len;
            for (0..out.data.len) |dest_i| {
                var src_idx: usize = 0;
                for (0..len) |d| {
                    src_idx += coord[d] * self.strides.dims[d];
                }
                out.data[dest_i] = convertScalar(DestT, f32, self.data[src_idx]);
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

    /// 从任意泛型张量创建标准 Autograd Tensor (f32)
    pub fn fromGeneric(comptime SrcT: type, generic_t: *const GenericTensor(SrcT), allocator: std.mem.Allocator) !*Tensor {
        const f32_gen = try generic_t.to(f32, allocator);
        defer f32_gen.deinit(allocator);
        return array(allocator, f32_gen.shape.dims[0..f32_gen.shape.len], f32_gen.data);
    }

    pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
        if (!self.is_view) {
            allocator.free(self.data);
            if (self.requires_grad and self.grad.len > 0) {
                allocator.free(self.grad);
            }
        }
        allocator.destroy(self);
    }
};



// ============================================================================
// NumPy-like raw tensor creation APIs (independent of Graph)
// ============================================================================

pub fn array(allocator: std.mem.Allocator, shape_slice: []const usize, initial_data: []const f32) !*Tensor {
    const shape = try Shape.fromSlice(shape_slice);
    const strides = computeContiguousStrides(shape);
    var total_size: usize = 1;
    for (shape_slice) |dim| {
        total_size *= dim;
    }
    if (total_size != initial_data.len) return error.ShapeMismatch;

    const t = try allocator.create(Tensor);
    t.* = Tensor{
        .data = try allocator.alloc(f32, total_size),
        .grad = &.{},
        .shape = shape,
        .strides = strides,
        .requires_grad = false,
        .creator = null,
    };
    @memcpy(t.data, initial_data);
    return t;
}

pub fn zeros(allocator: std.mem.Allocator, shape_slice: []const usize) !*Tensor {
    const shape = try Shape.fromSlice(shape_slice);
    const strides = computeContiguousStrides(shape);
    var total_size: usize = 1;
    for (shape_slice) |dim| {
        total_size *= dim;
    }

    const t = try allocator.create(Tensor);
    t.* = Tensor{
        .data = try allocator.alloc(f32, total_size),
        .grad = &.{},
        .shape = shape,
        .strides = strides,
        .requires_grad = false,
        .creator = null,
    };
    @memset(t.data, 0.0);
    return t;
}

pub fn ones(allocator: std.mem.Allocator, shape_slice: []const usize) !*Tensor {
    const shape = try Shape.fromSlice(shape_slice);
    const strides = computeContiguousStrides(shape);
    var total_size: usize = 1;
    for (shape_slice) |dim| {
        total_size *= dim;
    }

    const t = try allocator.create(Tensor);
    t.* = Tensor{
        .data = try allocator.alloc(f32, total_size),
        .grad = &.{},
        .shape = shape,
        .strides = strides,
        .requires_grad = false,
        .creator = null,
    };
    @memset(t.data, 1.0);
    return t;
}


var default_prng = std.Random.DefaultPrng.init(12345);

pub fn manualSeed(seed: u64) void {
    default_prng = std.Random.DefaultPrng.init(seed);
}

pub fn rand(allocator: std.mem.Allocator, shape_slice: []const usize) !*Tensor {
    const t = try zeros(allocator, shape_slice);
    const random = default_prng.random();
    for (t.data) |*val| {
        val.* = random.float(f32);
    }
    return t;
}

pub fn free(allocator: std.mem.Allocator, t: *Tensor) void {
    t.deinit(allocator);
}

/// 沿指定维度拼接多个张量 (Concat)
pub fn concat(allocator: std.mem.Allocator, inputs: []const *Tensor, dim: usize, graph: ?*autodiff.Graph) anyerror!*Tensor {
    if (graph) |g| {
        return try g.concat(inputs, dim);
    }
    if (inputs.len == 0) return error.EmptyInputs;
    const rank = inputs[0].shape.len;
    if (dim >= rank) return error.DimensionOutOfBounds;

    var out_shape = inputs[0].shape;
    var concat_dim_total: usize = 0;

    for (inputs) |t| {
        if (t.shape.len != rank) return error.IncompatibleDimensions;
        for (0..rank) |d| {
            if (d != dim) {
                if (t.shape.dims[d] != inputs[0].shape.dims[d]) return error.ShapeMismatch;
            }
        }
        concat_dim_total += t.shape.dims[dim];
    }
    out_shape.dims[dim] = concat_dim_total;

    const out = try zeros(allocator, out_shape.dims[0..rank]);

    var outer_size: usize = 1;
    for (0..dim) |d| {
        outer_size *= out_shape.dims[d];
    }
    var inner_size: usize = 1;
    for (dim + 1..rank) |d| {
        inner_size *= out_shape.dims[d];
    }

    for (0..outer_size) |outer| {
        const out_base = outer * concat_dim_total * inner_size;
        var offset_dim: usize = 0;
        for (inputs) |t| {
            const d_k = t.shape.dims[dim];
            const src_base = outer * d_k * inner_size;
            const dest_base = out_base + offset_dim * inner_size;
            const copy_len = d_k * inner_size;
            @memcpy(out.data[dest_base .. dest_base + copy_len], t.data[src_base .. src_base + copy_len]);
            offset_dim += d_k;
        }
    }

    return out;
}

/// 沿指定维度将张量均等切分为 num_splits 个子张量 (Split)
pub fn split(allocator: std.mem.Allocator, input: *Tensor, num_splits: usize, dim: usize, graph: ?*autodiff.Graph) anyerror![]*Tensor {
    if (graph) |g| {
        return try g.split(input, num_splits, dim);
    }
    if (num_splits == 0) return error.InvalidSplitCount;
    const rank = input.shape.len;
    if (dim >= rank) return error.DimensionOutOfBounds;
    const dim_size = input.shape.dims[dim];
    if (dim_size % num_splits != 0) return error.UnevenSplit;

    const split_dim_size = dim_size / num_splits;

    var split_shape = input.shape;
    split_shape.dims[dim] = split_dim_size;

    const outputs = try allocator.alloc(*Tensor, num_splits);
    for (0..num_splits) |k| {
        outputs[k] = try zeros(allocator, split_shape.dims[0..rank]);
    }

    var outer_size: usize = 1;
    for (0..dim) |d| {
        outer_size *= input.shape.dims[d];
    }
    var inner_size: usize = 1;
    for (dim + 1..rank) |d| {
        inner_size *= input.shape.dims[d];
    }

    for (0..outer_size) |outer| {
        const src_base = outer * dim_size * inner_size;
        for (0..num_splits) |k| {
            const dest_base = outer * split_dim_size * inner_size;
            const src_offset = src_base + k * split_dim_size * inner_size;
            const copy_len = split_dim_size * inner_size;
            @memcpy(outputs[k].data[dest_base .. dest_base + copy_len], input.data[src_offset .. src_offset + copy_len]);
        }
    }

    return outputs;
}

const tensorSplit = split;

pub const where = Tensor.where;

pub fn sum(t: *Tensor, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor {
    return t.sum(axis, keepdims, allocator);
}

pub fn mean(t: *Tensor, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator) !*Tensor {
    return t.mean(axis, keepdims, allocator);
}

pub fn variance(t: *Tensor, axis: ?usize, keepdims: bool, ddof: usize, allocator: std.mem.Allocator) !*Tensor {
    return t.variance(axis, keepdims, ddof, allocator);
}

pub fn stdDev(t: *Tensor, axis: ?usize, keepdims: bool, ddof: usize, allocator: std.mem.Allocator) !*Tensor {
    return t.stdDev(axis, keepdims, ddof, allocator);
}

pub fn squeeze(t: *Tensor, axis: ?usize, allocator: std.mem.Allocator) !*Tensor {
    return t.squeeze(axis, allocator);
}

pub fn unsqueeze(t: *Tensor, dim: usize, allocator: std.mem.Allocator) !*Tensor {
    return t.unsqueeze(dim, allocator);
}

pub fn slice(t: *Tensor, ranges: []const SliceRange, allocator: std.mem.Allocator) !*Tensor {
    return t.slice(ranges, allocator);
}

pub fn clip(t: *Tensor, min_val: f32, max_val: f32, allocator: std.mem.Allocator) !*Tensor {
    return t.clip(min_val, max_val, allocator);
}

pub fn sort(t: *Tensor, axis: ?usize, ascending: bool, allocator: std.mem.Allocator) !*Tensor {
    return t.sort(axis, ascending, allocator);
}

pub fn argsort(t: *Tensor, axis: ?usize, ascending: bool, allocator: std.mem.Allocator) !*GenericTensor(usize) {
    return t.argsort(axis, ascending, allocator);
}

pub fn nonzero(t: Tensor, allocator: std.mem.Allocator) !*GenericTensor(usize) {
    return t.nonzero(allocator);
}



/// Solves linear system A * x = b using Gauss-Jordan elimination with partial pivoting.
/// A is an n x n row-major matrix slice, b is an n-element vector, out_x is an n-element output slice.
pub fn solveLinearSystem(allocator: std.mem.Allocator, A_data: []const f32, b_data: []const f32, n: usize, out_x: []f32) !void {
    if (A_data.len != n * n or b_data.len != n or out_x.len != n) return error.ShapeMismatch;

    if (n == 0) return;
    if (n == 1) {
        if (@abs(A_data[0]) < 1e-12) return error.SingularMatrix;
        out_x[0] = b_data[0] / A_data[0];
        return;
    }

    // Augmented matrix [A | b] of dimensions n x (n + 1)
    const cols = n + 1;
    const aug = try allocator.alloc(f32, n * cols);
    defer allocator.free(aug);

    for (0..n) |i| {
        for (0..n) |j| {
            aug[i * cols + j] = A_data[i * n + j];
        }
        aug[i * cols + n] = b_data[i];
    }

    // Gauss-Jordan elimination with partial pivoting
    for (0..n) |col| {
        // Find pivot
        var max_val: f32 = @abs(aug[col * cols + col]);
        var pivot_row: usize = col;
        for ((col + 1)..n) |r| {
            const val = @abs(aug[r * cols + col]);
            if (val > max_val) {
                max_val = val;
                pivot_row = r;
            }
        }

        if (max_val < 1e-12) {
            return error.SingularMatrix;
        }

        // Swap current row with pivot row
        if (pivot_row != col) {
            for (0..cols) |j| {
                const tmp = aug[col * cols + j];
                aug[col * cols + j] = aug[pivot_row * cols + j];
                aug[pivot_row * cols + j] = tmp;
            }
        }

        // Normalize pivot row
        const pivot = aug[col * cols + col];
        for (col..cols) |j| {
            aug[col * cols + j] /= pivot;
        }

        // Eliminate column entries in other rows
        for (0..n) |r| {
            if (r == col) continue;
            const factor = aug[r * cols + col];
            if (factor == 0.0) continue;
            for (col..cols) |j| {
                aug[r * cols + j] -= factor * aug[col * cols + j];
            }
        }
    }

    // Extract solution
    for (0..n) |i| {
        out_x[i] = aug[i * cols + n];
    }
}

/// Solves Ridge Regression analytically:
/// min ||X*w + b*1 - y||_2^2 + lambda * ||w||_2^2
/// Using centered formulation: w = (X_c^T * X_c + lambda * I)^(-1) * X_c^T * y_c
/// b = mean(y) - sum(mean(x_j) * w_j)
pub fn solveRidgeAnalytical(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    n_samples: usize,
    n_features: usize,
    lambda: f32,
    out_w: []f32,
    out_b: *f32,
) !void {
    if (x.len != n_samples * n_features or y.len != n_samples or out_w.len != n_features) {
        return error.ShapeMismatch;
    }


    const N = n_samples;
    const D = n_features;
    const N_f = @as(f32, @floatFromInt(N));

    // 1. Compute means
    const mean_x = try allocator.alloc(f32, D);
    defer allocator.free(mean_x);
    @memset(mean_x, 0.0);

    var sum_y: f32 = 0.0;
    for (0..N) |i| {
        sum_y += y[i];
        for (0..D) |j| {
            mean_x[j] += x[i * D + j];
        }
    }
    const mean_y = sum_y / N_f;
    for (0..D) |j| {
        mean_x[j] /= N_f;
    }

    // 2. Build normal matrix M = X_c^T * X_c + lambda * I, and vector v = X_c^T * y_c
    const M = try allocator.alloc(f32, D * D);
    defer allocator.free(M);
    @memset(M, 0.0);

    const v = try allocator.alloc(f32, D);
    defer allocator.free(v);
    @memset(v, 0.0);

    for (0..N) |i| {
        const dy = y[i] - mean_y;
        for (0..D) |j| {
            const dx_j = x[i * D + j] - mean_x[j];
            v[j] += dx_j * dy;
            for (0..D) |k| {
                const dx_k = x[i * D + k] - mean_x[k];
                M[j * D + k] += dx_j * dx_k;
            }
        }
    }

    // Add L2 penalty lambda to the diagonal
    for (0..D) |j| {
        M[j * D + j] += lambda;
    }

    // 3. Solve M * w = v
    try solveLinearSystem(allocator, M, v, D, out_w);

    // 4. Compute intercept b = mean_y - w^T * mean_x
    var dot_w_mean_x: f32 = 0.0;
    for (0..D) |j| {
        dot_w_mean_x += out_w[j] * mean_x[j];
    }
    out_b.* = mean_y - dot_w_mean_x;
}

test "Shape and strides helpers" {
    // Test Shape init & eq
    const s1 = Shape.init(&.{2, 3, 4});
    try std.testing.expectEqual(@as(usize, 3), s1.len);
    try std.testing.expectEqual(@as(usize, 2), s1.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), s1.dims[1]);
    try std.testing.expectEqual(@as(usize, 4), s1.dims[2]);

    const s2 = Shape.init(&.{2, 3, 4});
    try std.testing.expect(s1.eq(s2));

    const s3 = Shape.init(&.{2, 3, 5});
    try std.testing.expect(!s1.eq(s3));

    // Test computeContiguousStrides
    const strides1 = computeContiguousStrides(s1);
    try std.testing.expectEqual(@as(usize, 12), strides1.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), strides1.dims[1]);
    try std.testing.expectEqual(@as(usize, 1), strides1.dims[2]);

    // Test transposeShape
    const s_trans = transposeShape(s1, 0, 1);
    try std.testing.expectEqual(@as(usize, 3), s_trans.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), s_trans.dims[1]);
    try std.testing.expectEqual(@as(usize, 4), s_trans.dims[2]);
}

test "Tensor indexing and gradient operations" {
    const allocator = std.testing.allocator;
    const shape = Shape.init(&.{2, 3});
    const strides = computeContiguousStrides(shape);

    const data = try allocator.alloc(f32, 6);
    defer allocator.free(data);
    const grad = try allocator.alloc(f32, 6);
    defer allocator.free(grad);

    var t = Tensor{
        .data = data,
        .grad = grad,
        .shape = shape,
        .strides = strides,
        .requires_grad = true,
        .creator = null,
    };

    // Test indexing
    t.set(&.{0, 0}, 1.0);
    t.set(&.{0, 1}, 2.0);
    t.set(&.{0, 2}, 3.0);
    t.set(&.{1, 0}, 4.0);
    t.set(&.{1, 1}, 5.0);
    t.set(&.{1, 2}, 6.0);

    try std.testing.expectEqual(@as(f32, 1.0), t.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 6.0), t.get(&.{1, 2}));
    try std.testing.expectEqual(@as(usize, 5), t.getFlatIndex(&.{1, 2}));

    // Test grad operations
    t.setGrad(&.{0, 1}, 10.0);
    try std.testing.expectEqual(@as(f32, 10.0), t.getGrad(&.{0, 1}));

    t.zeroGrad();
    try std.testing.expectEqual(@as(f32, 0.0), t.getGrad(&.{0, 1}));
}

test "NumPy-like raw tensor creation" {
    const allocator = std.testing.allocator;

    // Test array creation
    const t_arr = try array(allocator, &.{2, 3}, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer free(allocator, t_arr);
    try std.testing.expectEqual(@as(f32, 1.0), t_arr.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 6.0), t_arr.get(&.{1, 2}));

    // Test zeros creation
    const t_zeros = try zeros(allocator, &.{2, 2});
    defer free(allocator, t_zeros);
    try std.testing.expectEqual(@as(f32, 0.0), t_zeros.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 0.0), t_zeros.get(&.{1, 1}));

    // Test ones creation
    const t_ones = try ones(allocator, &.{3, 1});
    defer free(allocator, t_ones);
    try std.testing.expectEqual(@as(f32, 1.0), t_ones.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 1.0), t_ones.get(&.{2, 0}));
}

test "Direct tensor operations (eager and graph)" {
    const allocator = std.testing.allocator;

    // Eager Mode Test
    {
        const A = try array(allocator, &.{2, 3}, &[_]f32{ 1, 2, 3, 4, 5, 6 });
        defer free(allocator, A);
        const B = try array(allocator, &.{3, 2}, &[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
        defer free(allocator, B);

        // Matmul
        const C = try A.matmul(B, allocator, null);
        defer free(allocator, C);
        try std.testing.expectApproxEqAbs(@as(f32, 2.2), C.get(&.{0, 0}), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 6.4), C.get(&.{1, 1}), 1e-5);

        // AddBias
        const bias = try array(allocator, &.{1, 2}, &[_]f32{ 0.5, 1.0 });
        defer free(allocator, bias);
        const D = try C.addBias(bias, allocator, null);
        defer free(allocator, D);
        try std.testing.expectApproxEqAbs(@as(f32, 2.7), D.get(&.{0, 0}), 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 7.4), D.get(&.{1, 1}), 1e-5);

        // Relu
        const E = try D.relu(allocator, null);
        defer free(allocator, E);
        try std.testing.expectApproxEqAbs(@as(f32, 2.7), E.get(&.{0, 0}), 1e-5);

        // SoftmaxCrossEntropy
        const loss = try E.softmaxCrossEntropy(&[2]u8{ 0, 1 }, allocator, null);
        defer free(allocator, loss);
        try std.testing.expect(loss.get(&.{0, 0}) > 0.0);

        // Reshape
        const F = try E.reshape(&.{1, 4}, allocator, null);
        defer free(allocator, F);
        try std.testing.expectEqualSlices(usize, &.{1, 4}, F.shape.dims[0..F.shape.len]);

        // Transpose
        const G = try F.transpose(0, 1, allocator, null);
        defer free(allocator, G);
        try std.testing.expectEqualSlices(usize, &.{4, 1}, G.shape.dims[0..G.shape.len]);
    }

    // Graph Mode Test
    {
        var graph = autodiff.Graph.init(allocator);
        defer graph.deinit();

        const A = try graph.array(&.{2, 3}, &[_]f32{ 1, 2, 3, 4, 5, 6 }, true);
        const B = try graph.array(&.{3, 2}, &[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, true);

        // Matmul
        const C = try A.matmul(B, allocator, &graph);
        try std.testing.expectApproxEqAbs(@as(f32, 2.2), C.get(&.{0, 0}), 1e-5);

        // AddBias
        const bias = try graph.array(&.{1, 2}, &[_]f32{ 0.5, 1.0 }, true);
        const D = try C.addBias(bias, allocator, &graph);
        try std.testing.expectApproxEqAbs(@as(f32, 2.7), D.get(&.{0, 0}), 1e-5);

        // Relu
        const E = try D.relu(allocator, &graph);

        // SoftmaxCrossEntropy
        const loss = try E.softmaxCrossEntropy(&[2]u8{ 0, 1 }, allocator, &graph);
        try std.testing.expect(loss.get(&.{0, 0}) > 0.0);

        // Reshape
        const F = try E.reshape(&.{1, 4}, allocator, &graph);

        // Transpose
        const G = try F.transpose(0, 1, allocator, &graph);
        try std.testing.expectEqualSlices(usize, &.{4, 1}, G.shape.dims[0..G.shape.len]);
    }
}

test "Tensor argmax and max reductions" {
    const allocator = std.testing.allocator;

    const A = try array(allocator, &.{2, 3}, &[_]f32{ 1.0, 5.0, 3.0, 9.0, 2.0, 6.0 });
    defer free(allocator, A);

    // Test argmax along dim 1
    const idx1 = try A.argmax(1, allocator);
    defer free(allocator, idx1);
    try std.testing.expectEqual(@as(f32, 1.0), idx1.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 0.0), idx1.get(&.{1, 0}));

    // Test max along dim 1
    const val1 = try A.max(1, allocator);
    defer free(allocator, val1);
    try std.testing.expectEqual(@as(f32, 5.0), val1.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 9.0), val1.get(&.{1, 0}));

    // Test argmax along dim 0
    const idx0 = try A.argmax(0, allocator);
    defer free(allocator, idx0);
    try std.testing.expectEqual(@as(f32, 1.0), idx0.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 0.0), idx0.get(&.{0, 1}));
    try std.testing.expectEqual(@as(f32, 1.0), idx0.get(&.{0, 2}));

    // Test max along dim 0
    const val0 = try A.max(0, allocator);
    defer free(allocator, val0);
    try std.testing.expectEqual(@as(f32, 9.0), val0.get(&.{0, 0}));
    try std.testing.expectEqual(@as(f32, 5.0), val0.get(&.{0, 1}));
    try std.testing.expectEqual(@as(f32, 6.0), val0.get(&.{0, 2}));
}

test "Tensor MSE loss forward and backward" {
    const allocator = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const y_pred = try graph.array(&.{2, 1}, &[_]f32{ 1.5, 2.5 }, true);
    const y_true = try graph.array(&.{2, 1}, &[_]f32{ 1.0, 3.0 }, false);

    const loss = try graph.mseLoss(y_pred, y_true);
    // loss = 0.5 * ((1.5 - 1.0)^2 + (2.5 - 3.0)^2) = 0.5 * (0.25 + 0.25) = 0.25
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), loss.data[0], 1e-5);

    try graph.backward(loss);

    // grad of y_pred = 2/N * (y_pred - y_true) = 2/2 * (y_pred - y_true) = y_pred - y_true
    // dy_pred_0 = 1.5 - 1.0 = 0.5
    // dy_pred_1 = 2.5 - 3.0 = -0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), y_pred.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), y_pred.grad[1], 1e-5);
}

test "Tensor mulScalar and add autograd" {
    const allocator = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const A = try graph.array(&.{2, 2}, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);
    const B = try graph.array(&.{2, 2}, &[_]f32{ 5.0, 6.0, 7.0, 8.0 }, true);

    // C = A.mulScalar(2.0)
    const C = try A.mulScalar(2.0, arena_allocator, &graph);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), C.get(&.{0, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), C.get(&.{1, 1}), 1e-5);

    // D = C + B
    const D = try C.add(B, arena_allocator, &graph);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), D.get(&.{0, 0}), 1e-5); // 2.0 + 5.0 = 7.0
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), D.get(&.{1, 1}), 1e-5); // 8.0 + 8.0 = 16.0

    // E = D.addScalar(10.0)
    const E = try D.addScalar(10.0, arena_allocator, &graph);
    try std.testing.expectApproxEqAbs(@as(f32, 17.0), E.get(&.{0, 0}), 1e-5); // 7.0 + 10.0 = 17.0
    try std.testing.expectApproxEqAbs(@as(f32, 26.0), E.get(&.{1, 1}), 1e-5); // 16.0 + 10.0 = 26.0

    // Set gradients of E to 1.0 to backpropagate
    for (E.grad) |*g| {
        g.* = 1.0;
    }

    try graph.backward(E);

    // Since E = D + 10, dE/dD = 1
    // Since D = C + B, dD/dB = 1 => B.grad = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), B.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), B.grad[3], 1e-5);

    // Since E = D + 10, dE/dD = 1
    // Since D = C + B, dD/dC = 1
    // Since C = A * 2, dC/dA = 2
    // By chain rule, dE/dA = 1 * 1 * 2 = 2.0 => A.grad = 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[3], 1e-5);
}

test "Tensor static graph forward and backward" {
    const allocator = std.testing.allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    // 1. Build the static graph template once
    const A = try graph.array(&.{2, 2}, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);
    const B = try graph.array(&.{2, 2}, &[_]f32{ 5.0, 6.0, 7.0, 8.0 }, true);
    const C = try A.mulScalar(2.0, arena_allocator, &graph);
    const D = try C.add(B, arena_allocator, &graph);

    // 2. First Run: set inputs
    A.data[0] = 1.0; A.data[1] = 2.0; A.data[2] = 3.0; A.data[3] = 4.0;
    B.data[0] = 5.0; B.data[1] = 6.0; B.data[2] = 7.0; B.data[3] = 8.0;

    // Execute forward pass
    try graph.forward();
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), D.get(&.{0, 0}), 1e-5); // 2*1 + 5 = 7
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), D.get(&.{1, 1}), 1e-5); // 2*4 + 8 = 16

    // Execute backward pass
    graph.zeroGrad(); // Clear all gradients in the graph!
    @memset(D.grad, 1.0);
    try graph.backward(D);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), B.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[0], 1e-5);

    // 3. Second Run: change input data
    A.data[0] = 10.0; A.data[1] = 20.0; A.data[2] = 30.0; A.data[3] = 40.0;
    B.data[0] = 100.0; B.data[1] = 200.0; B.data[2] = 300.0; B.data[3] = 400.0;

    // Recompute forward pass on the exact same graph structure!
    try graph.forward();
    try std.testing.expectApproxEqAbs(@as(f32, 120.0), D.get(&.{0, 0}), 1e-5); // 2*10 + 100 = 120
    try std.testing.expectApproxEqAbs(@as(f32, 480.0), D.get(&.{1, 1}), 1e-5); // 2*40 + 400 = 480

    // Recompute backward pass
    graph.zeroGrad(); // Clear gradients again!
    @memset(D.grad, 1.0);
    try graph.backward(D);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), B.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[0], 1e-5);
}

test "Softmax forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    // Input shape [2, 3]
    const X = try graph.array(&.{2, 3}, &[_]f32{
        1.0, 2.0, 3.0,
        1.0, 1.0, 1.0,
    }, true);

    const Y = try X.softmax(arena_allocator, &graph);

    try graph.forward();

    // Check forward
    // Row 0: exp(1), exp(2), exp(3) -> sum = 2.718 + 7.389 + 20.085 = 30.192
    // exp(1)/sum = 0.0900, exp(2)/sum = 0.2447, exp(3)/sum = 0.6652
    try std.testing.expectApproxEqAbs(@as(f32, 0.09003057), Y.get(&.{0, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.24472847), Y.get(&.{0, 1}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.66524096), Y.get(&.{0, 2}), 1e-5);
    // Row 1: exp(1), exp(1), exp(1) -> 1/3, 1/3, 1/3
    try std.testing.expectApproxEqAbs(@as(f32, 0.33333333), Y.get(&.{1, 0}), 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(Y.grad, 1.0); // dL/dY = 1.0
    // dX_i = Y_i * (dY_i - sum_j dY_j Y_j)
    // Since dY_j = 1.0, sum_j dY_j Y_j = sum_j Y_j = 1.0 (since softmax sums to 1)
    // So dX_i = Y_i * (1.0 - 1.0) = 0.0
    try graph.backward(Y);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), X.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), X.grad[5], 1e-5);

    // Try another grad
    graph.zeroGrad();
    Y.grad[0] = 1.0;
    Y.grad[1] = 0.0;
    Y.grad[2] = 0.0;
    // Row 0: sum_dy_y = 1.0 * Y_0 = Y_0
    // dX_0 = Y_0 * (1.0 - Y_0) = Y_0 * (1 - Y_0)
    // dX_1 = Y_1 * (0.0 - Y_0) = - Y_1 * Y_0
    // dX_2 = Y_2 * (0.0 - Y_0) = - Y_2 * Y_0
    try graph.backward(Y);
    const y0 = Y.get(&.{0, 0});
    const y1 = Y.get(&.{0, 1});
    try std.testing.expectApproxEqAbs(y0 * (1.0 - y0), X.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(-y1 * y0, X.grad[1], 1e-5);
}

test "RMSNorm forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const X = try graph.array(&.{2, 3}, &[_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    }, true);
    const G = try graph.array(&.{3}, &[_]f32{ 1.0, 2.0, 3.0 }, true);

    const Y = try X.rmsNorm(G, 1e-5, arena_allocator, &graph);

    try graph.forward();

    // Row 0: mean(x^2) = (1+4+9)/3 = 14/3 = 4.666666
    // rms = sqrt(4.666666) = 2.1602468
    // Y_0 = 1 / rms * 1 = 0.46291
    // Y_1 = 2 / rms * 2 = 1.85164
    // Y_2 = 3 / rms * 3 = 4.16619
    try std.testing.expectApproxEqAbs(@as(f32, 0.46291), Y.get(&.{0, 0}), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.85164), Y.get(&.{0, 1}), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.16619), Y.get(&.{0, 2}), 1e-4);

    // Backward
    graph.zeroGrad();
    @memset(Y.grad, 1.0);
    try graph.backward(Y);

    // We can verify gradients numerically or just check they are non-zero and reasonable.
    // Let's verify G.grad: dG_j = sum_i (dY_i * X_i * scale)
    // Row 0 scale = 1/2.1602468 = 0.46291
    // Row 1: mean(x^2) = (16+25+36)/3 = 77/3 = 25.6666
    // Row 1 scale = 1/sqrt(25.6666) = 1/5.066228 = 0.197385
    // dG_0 = 1.0 * 1.0 * 0.46291 + 1.0 * 4.0 * 0.197385 = 0.46291 + 0.78954 = 1.25245
    try std.testing.expectApproxEqAbs(@as(f32, 1.25245), G.grad[0], 1e-4);
}

test "Embedding forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const W = try graph.array(&.{3, 4}, &[_]f32{
        0.1, 0.2, 0.3, 0.4,
        1.1, 1.2, 1.3, 1.4,
        2.1, 2.2, 2.3, 2.4,
    }, true);

    const X = try graph.array(&.{2, 2}, &[_]f32{
        0.0, 2.0,
        1.0, 0.0,
    }, false);

    const Y = try W.embedding(X, arena_allocator, &graph);

    try graph.forward();

    // Check forward
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), Y.get(&.{0, 0, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.3), Y.get(&.{0, 1, 2}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4), Y.get(&.{1, 0, 3}), 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(Y.grad, 1.0);
    try graph.backward(Y);

    // W.grad should accumulate gradients
    // X has:
    // (0,0) -> 0.0
    // (0,1) -> 2.0
    // (1,0) -> 1.0
    // (1,1) -> 0.0
    // So row 0 of W is selected twice, row 1 once, row 2 once.
    // Since dY is all 1.0, W.grad row 0 should be 2.0, row 1 should be 1.0, row 2 should be 1.0.
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), W.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), W.grad[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), W.grad[8], 1e-5);
}

test "BatchMatMul forward and backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    // Shape [2, 2, 2, 3]
    const A = try graph.array(&.{2, 2, 2, 3}, &[_]f32{
        // batch 0, head 0
        1, 2, 3,
        4, 5, 6,
        // batch 0, head 1
        1, 1, 1,
        2, 2, 2,
        // batch 1, head 0
        0, 1, 0,
        1, 0, 1,
        // batch 1, head 1
        2, 0, 2,
        0, 2, 0,
    }, true);

    // Shape [2, 2, 3, 2]
    const B = try graph.array(&.{2, 2, 3, 2}, &[_]f32{
        // batch 0, head 0
        1, 0,
        0, 1,
        1, 1,
        // batch 0, head 1
        2, 2,
        2, 2,
        2, 2,
        // batch 1, head 0
        1, 2,
        3, 4,
        5, 6,
        // batch 1, head 1
        1, 1,
        1, 1,
        1, 1,
    }, true);

    const C = try A.batchMatMul(B, arena_allocator, &graph);

    try graph.forward();

    // Check forward
    // Batch 0, Head 0:
    // [1, 2, 3]   [1, 0]   [4, 5]
    // [4, 5, 6] * [0, 1] = [10, 11]
    //             [1, 1]
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), C.get(&.{0, 0, 0, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), C.get(&.{0, 0, 0, 1}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), C.get(&.{0, 0, 1, 0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), C.get(&.{0, 0, 1, 1}), 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(C.grad, 1.0);
    try graph.backward(C);

    // We can verify some gradients.
    // dA = dC * B^T
    // For Batch 0, Head 0:
    // dC_slice = [1, 1]
    //            [1, 1]
    // B_slice^T = [1, 0, 1]
    //             [0, 1, 1]
    // dA_slice = dC_slice * B_slice^T = [1, 1, 2]
    //                                   [1, 1, 2]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), A.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), A.grad[2], 1e-5);
}

test "GELU forward and backward" {
    const arena_allocator = std.testing.allocator;
    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const A = try graph.tensorNDWithData(&.{2, 2}, &.{ -1.0, 0.0, 1.0, 2.0 }, true);
    const C = try A.gelu(arena_allocator, &graph);

    try graph.forward();

    // Check forward
    try std.testing.expectApproxEqAbs(@as(f32, -0.158655), C.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), C.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.841345), C.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.954500), C.data[3], 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(C.grad, 1.0);
    try graph.backward(C);

    // Check gradients
    try std.testing.expectApproxEqAbs(@as(f32, -0.083316), A.grad[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), A.grad[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.083316), A.grad[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.085232), A.grad[3], 1e-4);
}

test "Sigmoid forward and backward" {
    const arena_allocator = std.testing.allocator;
    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const A = try graph.tensorNDWithData(&.{2, 2}, &.{ -1.0, 0.0, 1.0, 2.0 }, true);
    const C = try A.sigmoid(arena_allocator, &graph);

    try graph.forward();

    // Check forward: sigmoid(x) = 1 / (1 + exp(-x))
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / (1.0 + @exp(@as(f32, 1.0)))), C.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), C.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / (1.0 + @exp(@as(f32, -1.0)))), C.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / (1.0 + @exp(@as(f32, -2.0)))), C.data[3], 1e-5);

    // Backward
    graph.zeroGrad();
    @memset(C.grad, 1.0);
    try graph.backward(C);

    // Check gradients: grad = C * (1 - C)
    for (A.grad, C.data) |g_val, c_val| {
        try std.testing.expectApproxEqAbs(c_val * (1.0 - c_val), g_val, 1e-5);
    }
}

test "SigmoidCrossEntropy forward and backward" {
    const arena_allocator = std.testing.allocator;
    var graph = autodiff.Graph.init(arena_allocator);
    defer graph.deinit();

    const logits = try graph.tensorNDWithData(&.{3}, &.{ -1.0, 0.0, 2.0 }, true);
    const targets = try graph.tensorNDWithData(&.{3}, &.{ 0.0, 1.0, 1.0 }, false);
    const loss = try logits.sigmoidCrossEntropy(targets, arena_allocator, &graph);

    try graph.forward();

    // Check forward
    // x = -1, y = 0 -> loss = max(-1, 0) - 0 + log(1 + exp(-1)) = log(1 + e^-1) = log(1.367879) = 0.31326168
    // x = 0, y = 1 -> loss = max(0, 0) - 0 + log(1 + exp(0)) = log(2) = 0.69314718
    // x = 2, y = 1 -> loss = max(2, 0) - 2 + log(1 + exp(-2)) = log(1 + e^-2) = log(1.135335) = 0.126928
    // mean loss = (0.31326168 + 0.69314718 + 0.126928) / 3 = 1.13333686 / 3 = 0.37777895
    try std.testing.expectApproxEqAbs(@as(f32, 0.37777895), loss.data[0], 1e-5);

    // Backward
    graph.zeroGrad();
    loss.grad[0] = 1.0;
    try graph.backward(loss);

    // Check gradients:
    // grad = 1/3 * (sig(x) - y)
    // x = -1, y = 0 -> grad = 1/3 * (1/(1+e) - 0) = 1/3 * 0.268941 = 0.089647
    // x = 0, y = 1 -> grad = 1/3 * (0.5 - 1) = -1/6 = -0.166667
    // x = 2, y = 1 -> grad = 1/3 * (1/(1+e^-2) - 1) = 1/3 * (0.880797 - 1) = -0.039734
    try std.testing.expectApproxEqAbs(@as(f32, 0.089647), logits.grad[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.166667), logits.grad[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.039734), logits.grad[2], 1e-5);
}

/// 旋转位置编码应用算子 (RoPE)
/// x: 输入张量切片，形状 [seq_len, n_head, head_dim]
/// head_dim 必须为偶数
pub fn applyRoPE(
    x: []f32,
    seq_len: usize,
    n_head: usize,
    head_dim: usize,
    base_freq: f32,
) void {
    std.debug.assert(head_dim % 2 == 0);
    const half_dim = head_dim / 2;

    for (0..seq_len) |m| {
        const m_f32 = @as(f32, @floatFromInt(m));

        for (0..half_dim) |i| {
            const i_f32 = @as(f32, @floatFromInt(i));
            const theta = 1.0 / std.math.pow(f32, base_freq, (2.0 * i_f32) / @as(f32, @floatFromInt(head_dim)));
            const freq = m_f32 * theta;
            const cos_val = @cos(freq);
            const sin_val = @sin(freq);

            for (0..n_head) |h| {
                const offset = (m * n_head + h) * head_dim + i * 2;
                const x1 = x[offset];
                const x2 = x[offset + 1];

                // 2D 旋转矩阵变换
                x[offset] = x1 * cos_val - x2 * sin_val;
                x[offset + 1] = x1 * sin_val + x2 * cos_val;
            }
        }
    }
}

/// 对 3D [T, n_head, head_dim] 或 4D [B, n_head, T, head_dim] 张量执行 RoPE 旋转
pub fn applyRoPETensor(t: *Tensor, base_freq: f32) void {
    if (t.shape.len == 3) {
        const seq_len = t.shape.dims[0];
        const n_head = t.shape.dims[1];
        const head_dim = t.shape.dims[2];
        applyRoPE(t.data, seq_len, n_head, head_dim, base_freq);
    } else if (t.shape.len == 4) {
        // [B, n_head, T, head_dim] -> 遍历每个 batch
        const B = t.shape.dims[0];
        const n_head = t.shape.dims[1];
        const T = t.shape.dims[2];
        const head_dim = t.shape.dims[3];
        const half_dim = head_dim / 2;

        for (0..B) |b| {
            for (0..T) |m| {
                const m_f32 = @as(f32, @floatFromInt(m));
                for (0..half_dim) |i| {
                    const i_f32 = @as(f32, @floatFromInt(i));
                    const theta = 1.0 / std.math.pow(f32, base_freq, (2.0 * i_f32) / @as(f32, @floatFromInt(head_dim)));
                    const freq = m_f32 * theta;
                    const cos_val = @cos(freq);
                    const sin_val = @sin(freq);

                    for (0..n_head) |h| {
                        const offset = ((b * n_head + h) * T + m) * head_dim + i * 2;
                        const x1 = t.data[offset];
                        const x2 = t.data[offset + 1];
                        t.data[offset] = x1 * cos_val - x2 * sin_val;
                        t.data[offset + 1] = x1 * sin_val + x2 * cos_val;
                    }
                }
            }
        }
    }
}

test "applyRoPE rotation properties" {
    var data = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // seq_len=2, n_head=1, head_dim=2
    applyRoPE(&data, 2, 1, 2, 10000.0);
    // m = 0: theta^0 = 1, freq = 0 -> cos(0)=1, sin(0)=0 -> x0=1.0, x1=0.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[1], 1e-5);

    // m = 1: freq = 1.0 -> cos(1), sin(1) for [0.0, 1.0] -> x2 = 0*cos(1) - 1*sin(1) = -sin(1), x3 = 0*sin(1) + 1*cos(1) = cos(1)
    const expected_x2 = -@sin(@as(f32, 1.0));
    const expected_x3 = @cos(@as(f32, 1.0));
    try std.testing.expectApproxEqAbs(expected_x2, data[2], 1e-5);
    try std.testing.expectApproxEqAbs(expected_x3, data[3], 1e-5);
}

test "broadcastShapes inference" {
    // 1. Same shapes
    const s1 = Shape.init(&.{ 2, 3 });
    const s2 = Shape.init(&.{ 2, 3 });
    const out1 = try broadcastShapes(s1, s2);
    try std.testing.expect(out1.eq(Shape.init(&.{ 2, 3 })));

    // 2. Trailing dimensions with 1s
    const s3 = Shape.init(&.{ 4, 1, 5 });
    const s4 = Shape.init(&.{ 3, 5 });
    const out2 = try broadcastShapes(s3, s4);
    try std.testing.expect(out2.eq(Shape.init(&.{ 4, 3, 5 })));

    // 3. Different rank multi-dim broadcasting
    const s5 = Shape.init(&.{ 2, 1, 4, 1 });
    const s6 = Shape.init(&.{ 3, 1, 5 });
    const out3 = try broadcastShapes(s5, s6);
    try std.testing.expect(out3.eq(Shape.init(&.{ 2, 3, 4, 5 })));

    // 4. Incompatible shapes
    const s7 = Shape.init(&.{ 3, 4 });
    const s8 = Shape.init(&.{ 2, 4 });
    try std.testing.expectError(error.IncompatibleBroadcastShapes, broadcastShapes(s7, s8));
}

test "tensor eager broadcasting operations (add, sub, mul, div)" {
    const allocator = std.testing.allocator;

    // A: 2x3 matrix
    const a = try array(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer free(allocator, a);

    // B: 1x3 row vector
    const b = try array(allocator, &.{ 1, 3 }, &.{ 10.0, 20.0, 30.0 });
    defer free(allocator, b);

    // C: 2x1 col vector
    const c_vec = try array(allocator, &.{ 2, 1 }, &.{ 100.0, 200.0 });
    defer free(allocator, c_vec);

    // 1. A + B -> 2x3
    const a_add_b = try a.add(b, allocator, null);
    defer free(allocator, a_add_b);
    try std.testing.expect(a_add_b.shape.eq(Shape.init(&.{ 2, 3 })));
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), a_add_b.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), a_add_b.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), a_add_b.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), a_add_b.data[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), a_add_b.data[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 36.0), a_add_b.data[5], 1e-5);

    // 2. A * C -> 2x3
    const a_mul_c = try a.mul(c_vec, allocator, null);
    defer free(allocator, a_mul_c);
    try std.testing.expect(a_mul_c.shape.eq(Shape.init(&.{ 2, 3 })));
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), a_mul_c.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 200.0), a_mul_c.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), a_mul_c.data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 800.0), a_mul_c.data[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1000.0), a_mul_c.data[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1200.0), a_mul_c.data[5], 1e-5);

    // 3. B - A -> 2x3
    const b_sub_a = try b.sub(a, allocator, null);
    defer free(allocator, b_sub_a);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), b_sub_a.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 18.0), b_sub_a.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 27.0), b_sub_a.data[2], 1e-5);

    // 4. B / A -> 2x3
    const b_div_a = try b.div(a, allocator, null);
    defer free(allocator, b_div_a);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), b_div_a.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), b_div_a.data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), b_div_a.data[2], 1e-5);

    // 5. 4D Broadcasting: [2, 1, 3, 1] + [1, 2, 1, 4] -> [2, 2, 3, 4] (Total 48 elements)
    const t4d_1 = try ones(allocator, &.{ 2, 1, 3, 1 });
    defer free(allocator, t4d_1);
    const t4d_2 = try array(allocator, &.{ 1, 2, 1, 4 }, &.{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    });
    defer free(allocator, t4d_2);

    const t4d_out = try t4d_1.add(t4d_2, allocator, null);
    defer free(allocator, t4d_out);
    try std.testing.expect(t4d_out.shape.eq(Shape.init(&.{ 2, 2, 3, 4 })));
    try std.testing.expectEqual(@as(usize, 48), t4d_out.data.len);
    // Elements should be 1.0 + t4d_2 values
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), t4d_out.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), t4d_out.data[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), t4d_out.data[12], 1e-5);

    // 6. subScalar and divScalar
    const s_sub = try a.subScalar(1.0, allocator, null);
    defer free(allocator, s_sub);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), s_sub.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), s_sub.data[5], 1e-5);

    const s_div = try a.divScalar(2.0, allocator, null);
    defer free(allocator, s_div);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), s_div.data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), s_div.data[5], 1e-5);
}

test "tensor argmax edge cases negative values and unsupported dimension error" {
    const allocator = std.testing.allocator;

    // 2x3 matrix with all negative values
    var t = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, t);

    // Row 0: [-10.0, -2.0, -5.0] -> max index is 1 (-2.0)
    // Row 1: [-1.0, -8.0, -4.0]  -> max index is 0 (-1.0)
    t.set(&.{ 0, 0 }, -10.0);
    t.set(&.{ 0, 1 }, -2.0);
    t.set(&.{ 0, 2 }, -5.0);
    t.set(&.{ 1, 0 }, -1.0);
    t.set(&.{ 1, 1 }, -8.0);
    t.set(&.{ 1, 2 }, -4.0);

    // 1. argmax dim=1
    const idx_col = try t.argmax(1, allocator);
    defer free(allocator, idx_col);
    try std.testing.expectEqual(@as(f32, 1.0), idx_col.data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), idx_col.data[1]);

    // 2. argmax dim=0
    // Col 0: -10 vs -1 -> index 1 (-1)
    // Col 1: -2 vs -8  -> index 0 (-2)
    // Col 2: -5 vs -4  -> index 1 (-4)
    const idx_row = try t.argmax(0, allocator);
    defer free(allocator, idx_row);
    try std.testing.expectEqual(@as(f32, 1.0), idx_row.data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), idx_row.data[1]);
    try std.testing.expectEqual(@as(f32, 1.0), idx_row.data[2]);

    // 3. Unsupported dimension on 3D tensor: dim=2 passes assert(dim < shape.len) and reaches error.UnsupportedDimension
    var t_3d = try zeros(allocator, &.{ 2, 2, 2 });
    defer free(allocator, t_3d);
    try std.testing.expectError(error.UnsupportedDimension, t_3d.argmax(2, allocator));
}

test "solveLinearSystem singular matrix error and n=1 scalar" {
    const allocator = std.testing.allocator;

    // 1. n = 1 non-zero
    const A1 = [_]f32{4.0};
    const b1 = [_]f32{8.0};
    var x1 = [_]f32{0.0};
    try solveLinearSystem(allocator, &A1, &b1, 1, &x1);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), x1[0], 1e-6);

    // 2. n = 1 singular (zero)
    const A_zero = [_]f32{0.0};
    try std.testing.expectError(error.SingularMatrix, solveLinearSystem(allocator, &A_zero, &b1, 1, &x1));

    // 3. n = 2 singular matrix (linearly dependent rows: [1, 2; 2, 4])
    const A_sing = [_]f32{ 1.0, 2.0, 2.0, 4.0 };
    const b2 = [_]f32{ 3.0, 6.0 };
    var x2 = [_]f32{ 0.0, 0.0 };
    try std.testing.expectError(error.SingularMatrix, solveLinearSystem(allocator, &A_sing, &b2, 2, &x2));
}

test "tensor typed error handling and boundary validation" {
    const allocator = std.testing.allocator;

    // 1. Shape.fromSlice and creation functions exceeding max dimensions (8)
    const nine_dims = [_]usize{ 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    try std.testing.expectError(error.MaxDimensionsExceeded, Shape.fromSlice(&nine_dims));
    try std.testing.expectError(error.MaxDimensionsExceeded, zeros(allocator, &nine_dims));
    try std.testing.expectError(error.MaxDimensionsExceeded, ones(allocator, &nine_dims));

    // 2. array data length mismatch
    const data_3 = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectError(error.ShapeMismatch, array(allocator, &.{ 2, 2 }, &data_3));

    // 3. getFlatIndexChecked and safe getChecked/setChecked
    var t2x2 = try zeros(allocator, &.{ 2, 2 });
    defer free(allocator, t2x2);
    try std.testing.expectError(error.DimensionMismatch, t2x2.getFlatIndexChecked(&.{ 0, 0, 0 }));
    try std.testing.expectError(error.IndexOutOfBounds, t2x2.getFlatIndexChecked(&.{ 2, 0 }));
    try std.testing.expectError(error.IndexOutOfBounds, t2x2.getChecked(&.{ 0, 3 }));
    try std.testing.expectError(error.IndexOutOfBounds, t2x2.setChecked(&.{ 3, 0 }, 1.0));
    try t2x2.setChecked(&.{ 1, 1 }, 42.0);
    try std.testing.expectEqual(@as(f32, 42.0), try t2x2.getChecked(&.{ 1, 1 }));

    // 4. matmul error conditions
    const t1d = try zeros(allocator, &.{4});
    defer free(allocator, t1d);
    var t2x3 = try zeros(allocator, &.{ 2, 3 });
    defer free(allocator, t2x3);
    const t4x2 = try zeros(allocator, &.{ 4, 2 });
    defer free(allocator, t4x2);
    // Non-2D inputs
    try std.testing.expectError(error.IncompatibleDimensions, t2x2.matmul(t1d, allocator, null));
    // Inner dimension mismatch (2x3 cannot multiply 4x2)
    try std.testing.expectError(error.ShapeMismatch, t2x3.matmul(t4x2, allocator, null));

    // 5. batchMatMul error conditions
    var t4d_a = try zeros(allocator, &.{ 1, 2, 3, 4 });
    defer free(allocator, t4d_a);
    const t4d_b_bad = try zeros(allocator, &.{ 1, 2, 5, 6 }); // K mismatch (4 != 5)
    defer free(allocator, t4d_b_bad);
    try std.testing.expectError(error.IncompatibleDimensions, t4d_a.batchMatMul(t2x2, allocator, null));
    try std.testing.expectError(error.ShapeMismatch, t4d_a.batchMatMul(t4d_b_bad, allocator, null));

    // 6. reshape element count mismatch
    try std.testing.expectError(error.ShapeMismatch, t2x2.reshape(&.{ 3, 3 }, allocator, null));

    // 7. transpose dimension out of bounds
    try std.testing.expectError(error.DimensionOutOfBounds, t2x2.transpose(0, 3, allocator, null));

    // 8. conv2d error conditions
    const w_bad_c = try zeros(allocator, &.{ 2, 3, 2, 2 }); // C_in mismatch with t4d_a (C_in is 2, weight has 3)
    defer free(allocator, w_bad_c);
    try std.testing.expectError(error.ShapeMismatch, t4d_a.conv2d(w_bad_c, null, allocator, null));
    const w_too_big = try zeros(allocator, &.{ 2, 2, 5, 5 }); // KH/KW > H/W (5 > 3 or 4)
    defer free(allocator, w_too_big);
    try std.testing.expectError(error.KernelBiggerThanInput, t4d_a.conv2d(w_too_big, null, allocator, null));

    // 9. concat error conditions
    try std.testing.expectError(error.EmptyInputs, concat(allocator, &.{}, 0, null));
    const inputs_dim_out = [_]*Tensor{t2x2};
    try std.testing.expectError(error.DimensionOutOfBounds, concat(allocator, &inputs_dim_out, 3, null));
    const inputs_mismatch = [_]*Tensor{ t2x2, t2x3 };
    try std.testing.expectError(error.ShapeMismatch, concat(allocator, &inputs_mismatch, 0, null));

    // 10. split error conditions
    try std.testing.expectError(error.InvalidSplitCount, split(allocator, t2x2, 0, 0, null));
    try std.testing.expectError(error.DimensionOutOfBounds, split(allocator, t2x2, 2, 5, null));
    try std.testing.expectError(error.UnevenSplit, split(allocator, t2x3, 2, 1, null)); // dim 1 has size 3, not divisible by 2

    // 11. solveLinearSystem slice length mismatch
    const bad_A = [_]f32{ 1.0, 2.0 };
    const b = [_]f32{1.0};
    var x = [_]f32{0.0};
    try std.testing.expectError(error.ShapeMismatch, solveLinearSystem(allocator, &bad_A, &b, 1, &x));
}

test "Tensor multi-axis reductions (sum, mean, variance, stdDev)" {
    const allocator = std.testing.allocator;

    // 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    const t = try array(allocator, &.{ 2, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer free(allocator, t);

    // 1. sum over all elements (axis = null)
    {
        const s_all = try t.sum(null, false, allocator);
        defer free(allocator, s_all);
        try std.testing.expectEqual(@as(usize, 1), s_all.shape.len);
        try std.testing.expectEqual(@as(usize, 1), s_all.shape.dims[0]);
        try std.testing.expectEqual(@as(f32, 21.0), s_all.data[0]);

        const s_all_kd = try t.sum(null, true, allocator);
        defer free(allocator, s_all_kd);
        try std.testing.expectEqual(@as(usize, 2), s_all_kd.shape.len);
        try std.testing.expectEqual(@as(usize, 1), s_all_kd.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 1), s_all_kd.shape.dims[1]);
        try std.testing.expectEqual(@as(f32, 21.0), s_all_kd.data[0]);
    }

    // 2. sum over axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
    {
        const s0 = try sum(t, 0, false, allocator);
        defer free(allocator, s0);
        try std.testing.expectEqual(@as(usize, 1), s0.shape.len);
        try std.testing.expectEqual(@as(usize, 3), s0.shape.dims[0]);
        try std.testing.expectEqual(@as(f32, 5.0), s0.data[0]);
        try std.testing.expectEqual(@as(f32, 7.0), s0.data[1]);
        try std.testing.expectEqual(@as(f32, 9.0), s0.data[2]);

        const s0_kd = try sum(t, 0, true, allocator);
        defer free(allocator, s0_kd);
        try std.testing.expectEqual(@as(usize, 2), s0_kd.shape.len);
        try std.testing.expectEqual(@as(usize, 1), s0_kd.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 3), s0_kd.shape.dims[1]);
        try std.testing.expectEqual(@as(f32, 5.0), s0_kd.data[0]);
    }

    // 3. sum over axis 1: [1+2+3, 4+5+6] = [6, 15]
    {
        const s1 = try sum(t, 1, false, allocator);
        defer free(allocator, s1);
        try std.testing.expectEqual(@as(usize, 1), s1.shape.len);
        try std.testing.expectEqual(@as(usize, 2), s1.shape.dims[0]);
        try std.testing.expectEqual(@as(f32, 6.0), s1.data[0]);
        try std.testing.expectEqual(@as(f32, 15.0), s1.data[1]);

        const s1_kd = try sum(t, 1, true, allocator);
        defer free(allocator, s1_kd);
        try std.testing.expectEqual(@as(usize, 2), s1_kd.shape.len);
        try std.testing.expectEqual(@as(usize, 2), s1_kd.shape.dims[0]);
        try std.testing.expectEqual(@as(usize, 1), s1_kd.shape.dims[1]);
        try std.testing.expectEqual(@as(f32, 6.0), s1_kd.data[0]);
        try std.testing.expectEqual(@as(f32, 15.0), s1_kd.data[1]);
    }

    // 4. Non-contiguous sum test: custom strided view
    {
        var t_strided = Tensor{
            .data = t.data,
            .grad = &.{},
            .shape = Shape.init(&.{ 3, 2 }),
            .strides = Shape.init(&.{ 1, 3 }), // transposed strides!
            .requires_grad = false,
            .creator = null,
        };
        try std.testing.expect(!t_strided.isContiguous());
        const s_strided = try t_strided.sum(0, false, allocator);
        defer free(allocator, s_strided);
        // r=0: [1, 4], r=1: [2, 5], r=2: [3, 6]
        // sum along axis 0 gives [1+2+3, 4+5+6] = [6, 15]
        try std.testing.expectEqual(@as(f32, 6.0), s_strided.data[0]);
        try std.testing.expectEqual(@as(f32, 15.0), s_strided.data[1]);
    }

    // 5. mean
    {
        const m_all = try mean(t, null, false, allocator);
        defer free(allocator, m_all);
        try std.testing.expectApproxEqAbs(@as(f32, 3.5), m_all.data[0], 1e-5);

        const m0 = try mean(t, 0, false, allocator);
        defer free(allocator, m0);
        try std.testing.expectApproxEqAbs(@as(f32, 2.5), m0.data[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 3.5), m0.data[1], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 4.5), m0.data[2], 1e-5);

        const m1 = try mean(t, 1, false, allocator);
        defer free(allocator, m1);
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), m1.data[0], 1e-5);
        try std.testing.expectApproxEqAbs(@as(f32, 5.0), m1.data[1], 1e-5);
    }

    // 6. variance and stdDev
    {
        // variance with ddof=0: 17.5 / 6 = 2.9166667
        const v_all = try variance(t, null, false, 0, allocator);
        defer free(allocator, v_all);
        try std.testing.expectApproxEqAbs(@as(f32, 2.9166667), v_all.data[0], 1e-5);

        // variance with ddof=1: 17.5 / 5 = 3.5
        const v_sample = try variance(t, null, false, 1, allocator);
        defer free(allocator, v_sample);
        try std.testing.expectApproxEqAbs(@as(f32, 3.5), v_sample.data[0], 1e-5);

        // stdDev with ddof=0: sqrt(2.9166667) ~= 1.7078251
        const sd_all = try stdDev(t, null, false, 0, allocator);
        defer free(allocator, sd_all);
        try std.testing.expectApproxEqAbs(@as(f32, 1.7078251), sd_all.data[0], 1e-5);

        // stdDev along axis 1:
        // row 0: [1, 2, 3], mean = 2, sq_diff sum = (1-2)^2 + (2-2)^2 + (3-2)^2 = 2.
        // var(ddof=0) = 2/3, stdDev = sqrt(2/3) ~= 0.8164966
        const sd1 = try stdDev(t, 1, false, 0, allocator);
        defer free(allocator, sd1);
        try std.testing.expectApproxEqAbs(@as(f32, 0.8164966), sd1.data[0], 1e-5);
    }

    // 7. Error handling
    try std.testing.expectError(error.DimensionOutOfBounds, t.sum(5, false, allocator));
    try std.testing.expectError(error.DimensionOutOfBounds, t.mean(2, false, allocator));
    try std.testing.expectError(error.InvalidDDOF, t.variance(null, false, 6, allocator));
    try std.testing.expectError(error.InvalidDDOF, t.stdDev(null, false, 10, allocator));
}

test "Tensor where condition and masking operations" {
    const allocator = std.testing.allocator;

    // 1. where with identical shapes
    const cond = try array(allocator, &.{ 2, 2 }, &[_]f32{ 1.0, 0.0, 0.0, 1.0 });
    defer free(allocator, cond);
    const x = try array(allocator, &.{ 2, 2 }, &[_]f32{ 10.0, 20.0, 30.0, 40.0 });
    defer free(allocator, x);
    const y = try array(allocator, &.{ 2, 2 }, &[_]f32{ -1.0, -2.0, -3.0, -4.0 });
    defer free(allocator, y);

    const out = try where(cond, x, y, allocator);
    defer free(allocator, out);
    try std.testing.expectEqual(@as(f32, 10.0), out.data[0]);
    try std.testing.expectEqual(@as(f32, -2.0), out.data[1]);
    try std.testing.expectEqual(@as(f32, -3.0), out.data[2]);
    try std.testing.expectEqual(@as(f32, 40.0), out.data[3]);

    // 2. where with broadcast condition
    // cond shape [2, 1], x shape [2, 2], y shape [2, 2]
    const cond_bc = try array(allocator, &.{ 2, 1 }, &[_]f32{ 1.0, 0.0 });
    defer free(allocator, cond_bc);
    const out_bc = try where(cond_bc, x, y, allocator);
    defer free(allocator, out_bc);
    // Row 0 selects x: [10.0, 20.0]; Row 1 selects y: [-3.0, -4.0]
    try std.testing.expectEqual(@as(f32, 10.0), out_bc.data[0]);
    try std.testing.expectEqual(@as(f32, 20.0), out_bc.data[1]);
    try std.testing.expectEqual(@as(f32, -3.0), out_bc.data[2]);
    try std.testing.expectEqual(@as(f32, -4.0), out_bc.data[3]);

    // 3. maskedFill (out of place)
    const mask = try array(allocator, &.{ 2, 2 }, &[_]f32{ 1.0, 0.0, 1.0, 0.0 });
    defer free(allocator, mask);
    const filled = try x.maskedFill(mask, -999.0, allocator);
    defer free(allocator, filled);
    try std.testing.expectEqual(@as(f32, -999.0), filled.data[0]);
    try std.testing.expectEqual(@as(f32, 20.0), filled.data[1]);
    try std.testing.expectEqual(@as(f32, -999.0), filled.data[2]);
    try std.testing.expectEqual(@as(f32, 40.0), filled.data[3]);
    // Original x should remain unchanged
    try std.testing.expectEqual(@as(f32, 10.0), x.data[0]);

    // 4. maskedFill_ (in place)
    var x_mut = try array(allocator, &.{ 2, 2 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    defer free(allocator, x_mut);
    _ = try x_mut.maskedFill_(mask, 0.0);
    try std.testing.expectEqual(@as(f32, 0.0), x_mut.data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), x_mut.data[1]);
    try std.testing.expectEqual(@as(f32, 0.0), x_mut.data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), x_mut.data[3]);

    // In-place protection for graph tensor
    x_mut.requires_grad = true;
    try std.testing.expectError(error.InPlaceOpOnGraphTensor, x_mut.maskedFill_(mask, 1.0));

    // Shape mismatch
    const bad_mask = try zeros(allocator, &.{3});
    defer free(allocator, bad_mask);
    try std.testing.expectError(error.ShapeMismatch, x.maskedFill(bad_mask, 0.0, allocator));
}

test "Tensor squeeze and unsqueeze" {
    const allocator = std.testing.allocator;

    // 1. Squeeze
    const t_4d = try zeros(allocator, &.{ 1, 2, 1, 3 });
    defer free(allocator, t_4d);

    // Squeeze all size-1 dims
    const t_sq_all = try squeeze(t_4d, null, allocator);
    defer free(allocator, t_sq_all);
    try std.testing.expectEqual(@as(usize, 2), t_sq_all.shape.len);
    try std.testing.expectEqual(@as(usize, 2), t_sq_all.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), t_sq_all.shape.dims[1]);

    // Squeeze specific dim 0
    const t_sq_0 = try squeeze(t_4d, 0, allocator);
    defer free(allocator, t_sq_0);
    try std.testing.expectEqual(@as(usize, 3), t_sq_0.shape.len);
    try std.testing.expectEqual(@as(usize, 2), t_sq_0.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 1), t_sq_0.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), t_sq_0.shape.dims[2]);

    // Squeeze specific dim 2
    const t_sq_2 = try squeeze(t_4d, 2, allocator);
    defer free(allocator, t_sq_2);
    try std.testing.expectEqual(@as(usize, 3), t_sq_2.shape.len);
    try std.testing.expectEqual(@as(usize, 1), t_sq_2.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), t_sq_2.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), t_sq_2.shape.dims[2]);

    // Cannot squeeze non-unit dimension
    try std.testing.expectError(error.CannotSqueezeDimension, t_4d.squeeze(1, allocator));
    try std.testing.expectError(error.DimensionOutOfBounds, t_4d.squeeze(5, allocator));

    // Squeeze on tensor where all dimensions are 1
    const t_1x1 = try zeros(allocator, &.{ 1, 1 });
    defer free(allocator, t_1x1);
    const t_sq_scalar = try t_1x1.squeeze(null, allocator);
    defer free(allocator, t_sq_scalar);
    try std.testing.expectEqual(@as(usize, 1), t_sq_scalar.shape.len);
    try std.testing.expectEqual(@as(usize, 1), t_sq_scalar.shape.dims[0]);

    // 2. Unsqueeze
    const t_2d = try array(allocator, &.{ 2, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer free(allocator, t_2d);

    // Insert at dim 0: [1, 2, 3]
    const u_dim0 = try unsqueeze(t_2d, 0, allocator);
    defer free(allocator, u_dim0);
    try std.testing.expectEqual(@as(usize, 3), u_dim0.shape.len);
    try std.testing.expectEqual(@as(usize, 1), u_dim0.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), u_dim0.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), u_dim0.shape.dims[2]);
    try std.testing.expectEqual(@as(f32, 1.0), u_dim0.data[0]);
    try std.testing.expectEqual(@as(f32, 6.0), u_dim0.data[5]);

    // Insert at dim 1: [2, 1, 3]
    const u_dim1 = try unsqueeze(t_2d, 1, allocator);
    defer free(allocator, u_dim1);
    try std.testing.expectEqual(@as(usize, 3), u_dim1.shape.len);
    try std.testing.expectEqual(@as(usize, 2), u_dim1.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 1), u_dim1.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 3), u_dim1.shape.dims[2]);

    // Insert at dim 2 (end): [2, 3, 1]
    const u_dim2 = try unsqueeze(t_2d, 2, allocator);
    defer free(allocator, u_dim2);
    try std.testing.expectEqual(@as(usize, 3), u_dim2.shape.len);
    try std.testing.expectEqual(@as(usize, 2), u_dim2.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), u_dim2.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 1), u_dim2.shape.dims[2]);

    // Out of bounds
    try std.testing.expectError(error.DimensionOutOfBounds, t_2d.unsqueeze(4, allocator));
}

test "DType, bf16, and scalar type conversion" {
    // 1. DType sizeOf
    try std.testing.expectEqual(@as(usize, 4), DType.f32.sizeOf());
    try std.testing.expectEqual(@as(usize, 8), DType.f64.sizeOf());
    try std.testing.expectEqual(@as(usize, 2), DType.f16.sizeOf());
    try std.testing.expectEqual(@as(usize, 2), DType.bf16.sizeOf());
    try std.testing.expectEqual(@as(usize, 4), DType.i32.sizeOf());
    try std.testing.expectEqual(@as(usize, 8), DType.i64.sizeOf());
    try std.testing.expectEqual(@as(usize, 1), DType.u8.sizeOf());
    try std.testing.expectEqual(@as(usize, 1), DType.bool.sizeOf());

    // 2. bf16 conversions
    const b0 = bf16.fromF32(0.0);
    try std.testing.expectEqual(@as(f32, 0.0), b0.toF32());

    const b1 = bf16.fromF32(1.0);
    try std.testing.expectEqual(@as(f32, 1.0), b1.toF32());

    const bm1 = bf16.fromF32(-1.0);
    try std.testing.expectEqual(@as(f32, -1.0), bm1.toF32());

    const b2_5 = bf16.fromF32(2.5);
    try std.testing.expectEqual(@as(f32, 2.5), b2_5.toF32());

    const b_pi = bf16.fromF32(3.14159);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14159), b_pi.toF32(), 1e-2);

    // 3. convertScalar
    try std.testing.expectEqual(@as(f64, 1.5), convertScalar(f64, f32, 1.5));
    try std.testing.expectEqual(@as(i32, 42), convertScalar(i32, f32, 42.0));
    try std.testing.expectEqual(true, convertScalar(bool, i32, 1));
    try std.testing.expectEqual(false, convertScalar(bool, i32, 0));
    try std.testing.expectEqual(@as(f32, 2.0), convertScalar(f32, bf16, bf16.fromF32(2.0)));
    try std.testing.expectEqual(@as(f32, 2.0), convertScalar(bf16, f32, 2.0).toF32());
}

test "GenericTensor and multi-type tensor manipulation" {
    const allocator = std.testing.allocator;

    // 1. GenericTensor(i32)
    const t_i32 = try GenericTensor(i32).fromSlice(allocator, &.{ 2, 2 }, &[_]i32{ 1, 2, 3, 4 });
    defer t_i32.deinit(allocator);
    try std.testing.expectEqual(@as(i32, 1), t_i32.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(i32, 3), t_i32.get(&.{ 1, 0 }));
    t_i32.set(&.{ 1, 0 }, 30);
    try std.testing.expectEqual(@as(i32, 30), t_i32.get(&.{ 1, 0 }));

    // 2. GenericTensor(bool)
    const t_b = try BoolTensor.fromSlice(allocator, &.{3}, &[_]bool{ true, false, true });
    defer t_b.deinit(allocator);
    try std.testing.expectEqual(true, t_b.get(&.{0}));
    try std.testing.expectEqual(false, t_b.get(&.{1}));
    try std.testing.expectEqual(true, t_b.get(&.{2}));

    // 3. GenericTensor to conversion
    const t_f32_conv = try t_i32.to(f32, allocator);
    defer t_f32_conv.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 1.0), t_f32_conv.data[0]);
    try std.testing.expectEqual(@as(f32, 30.0), t_f32_conv.data[2]);

    // 4. BFloat16Tensor
    const t_bf16 = try BFloat16Tensor.init(allocator, &.{2}, bf16.fromF32(3.5));
    defer t_bf16.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 3.5), t_bf16.data[0].toF32());

    // 5. Bidirectional Tensor <-> GenericTensor
    const t_orig = try array(allocator, &.{2}, &[_]f32{ 10.0, 20.0 });
    defer free(allocator, t_orig);
    const t_int_gen = try t_orig.to(i32, allocator);
    defer t_int_gen.deinit(allocator);
    try std.testing.expectEqual(@as(i32, 10), t_int_gen.data[0]);
    try std.testing.expectEqual(@as(i32, 20), t_int_gen.data[1]);

    const t_back = try Tensor.fromGeneric(i32, t_int_gen, allocator);
    defer free(allocator, t_back);
    try std.testing.expectEqual(@as(f32, 10.0), t_back.data[0]);
    try std.testing.expectEqual(@as(f32, 20.0), t_back.data[1]);
}

test "Tensor strided view slicing, contiguous, clip, sort, argsort, nonzero" {
    const allocator = std.testing.allocator;

    // 1. Slicing on 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    var t = try array(allocator, &.{ 2, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer free(allocator, t);

    // Extract submatrix rows 0..2, cols 1..3 -> [[2, 3], [5, 6]]
    const s = try slice(t, &.{ .{ .start = 0, .end = 2 }, .{ .start = 1, .end = 3 } }, allocator);
    defer free(allocator, s);
    try std.testing.expect(s.is_view);
    try std.testing.expectEqual(@as(usize, 2), s.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), s.shape.dims[1]);
    try std.testing.expectEqual(@as(f32, 2.0), s.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 3.0), s.get(&.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 5.0), s.get(&.{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 6.0), s.get(&.{ 1, 1 }));

    // Zero-copy mutation: mutating slice modifies parent
    s.set(&.{ 0, 0 }, 99.0);
    try std.testing.expectEqual(@as(f32, 99.0), t.get(&.{ 0, 1 }));
    s.set(&.{ 0, 0 }, 2.0); // restore

    // Contiguous copy of non-contiguous slice
    const c_contig = try s.contiguous(allocator);
    defer free(allocator, c_contig);
    try std.testing.expect(!c_contig.is_view);
    try std.testing.expect(c_contig.isContiguous());
    try std.testing.expectEqual(@as(f32, 2.0), c_contig.data[0]);
    try std.testing.expectEqual(@as(f32, 3.0), c_contig.data[1]);
    try std.testing.expectEqual(@as(f32, 5.0), c_contig.data[2]);
    try std.testing.expectEqual(@as(f32, 6.0), c_contig.data[3]);

    // Slice error conditions
    try std.testing.expectError(error.DimensionOutOfBounds, t.slice(&.{ .{}, .{}, .{} }, allocator));
    try std.testing.expectError(error.IndexOutOfBounds, t.slice(&.{ .{ .start = 10 } }, allocator));
    try std.testing.expectError(error.InvalidSliceRange, t.slice(&.{ .{ .start = 2, .end = 1 } }, allocator));
    try std.testing.expectError(error.InvalidStep, t.slice(&.{ .{ .step = 0 } }, allocator));

    // 2. clip and clip_
    var t_clip = try array(allocator, &.{4}, &[_]f32{ -5.0, 0.5, 3.0, 10.0 });
    defer free(allocator, t_clip);
    const clipped = try clip(t_clip, 0.0, 5.0, allocator);
    defer free(allocator, clipped);
    try std.testing.expectEqual(@as(f32, 0.0), clipped.data[0]);
    try std.testing.expectEqual(@as(f32, 0.5), clipped.data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), clipped.data[2]);
    try std.testing.expectEqual(@as(f32, 5.0), clipped.data[3]);

    _ = try t_clip.clip_(0.0, 5.0);
    try std.testing.expectEqual(@as(f32, 0.0), t_clip.data[0]);
    try std.testing.expectEqual(@as(f32, 5.0), t_clip.data[3]);
    try std.testing.expectError(error.InvalidRange, t_clip.clip(5.0, 2.0, allocator));

    // 3. sort and argsort
    const t_unsorted = try array(allocator, &.{4}, &[_]f32{ 3.0, 1.0, 4.0, 2.0 });
    defer free(allocator, t_unsorted);

    const t_sorted = try sort(t_unsorted, 0, true, allocator);
    defer free(allocator, t_sorted);
    try std.testing.expectEqual(@as(f32, 1.0), t_sorted.data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), t_sorted.data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), t_sorted.data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), t_sorted.data[3]);

    const t_desc = try sort(t_unsorted, 0, false, allocator);
    defer free(allocator, t_desc);
    try std.testing.expectEqual(@as(f32, 4.0), t_desc.data[0]);
    try std.testing.expectEqual(@as(f32, 1.0), t_desc.data[3]);

    const t_idxs = try argsort(t_unsorted, 0, true, allocator);
    defer t_idxs.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), t_idxs.data[0]);
    try std.testing.expectEqual(@as(usize, 3), t_idxs.data[1]);
    try std.testing.expectEqual(@as(usize, 0), t_idxs.data[2]);
    try std.testing.expectEqual(@as(usize, 2), t_idxs.data[3]);

    // 4. nonzero
    const t_sparse = try array(allocator, &.{ 2, 3 }, &[_]f32{ 0.0, 5.0, 0.0, 1.0, 0.0, 2.0 });
    defer free(allocator, t_sparse);
    const nz = try nonzero(t_sparse.*, allocator);
    defer nz.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 3), nz.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 2), nz.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 0), nz.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(usize, 1), nz.get(&.{ 0, 1 }));
    try std.testing.expectEqual(@as(usize, 1), nz.get(&.{ 1, 0 }));
    try std.testing.expectEqual(@as(usize, 0), nz.get(&.{ 1, 1 }));
    try std.testing.expectEqual(@as(usize, 1), nz.get(&.{ 2, 0 }));
    try std.testing.expectEqual(@as(usize, 2), nz.get(&.{ 2, 1 }));
}


