const std = @import("std");

// ============================================================================
// 1. 维度与形状控制（Shape & Strides Meta-data）
// ============================================================================

/// 多维张量的形状描述体（Shape）
/// 为避免动态内存分配带来的开销，本框架采用静态数组 `[8]usize` 存储各维度大小，最多支持 8 维张量。
pub const Shape = struct {
    dims: [8]usize, // 存储每一维度大小的静态数组，未使用的维度默认为 0
    len: usize, // 张量的维度个数（Rank，例如 2D 矩阵的 Rank 为 2）

    /// 从切片安全初始化 Shape，超过 8 维返回 error.MaxDimensionsExceeded
    pub fn fromSlice(shape_slice: []const usize) !Shape {
        if (shape_slice.len > 8) return error.MaxDimensionsExceeded;
        var self = Shape{
            .dims = @splat(0),
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

    /// 计算张量逻辑元素总数（0 维标量张量含 1 个元素，与 `zeros(&.{})` 分配的缓冲区长度一致）
    pub fn numel(self: Shape) usize {
        var total: usize = 1;
        for (0..self.len) |i| {
            total *= self.dims[i];
        }
        return total;
    }

    /// Increments an N-dimensional coordinate array in-place, carrying over to the previous dimension if a dimension reaches its limit.
    pub inline fn incrementCoord(self: Shape, coord: *[8]usize) void {
        var d = self.len;
        while (d > 0) {
            d -= 1;
            coord[d] += 1;
            if (coord[d] < self.dims[d]) break;
            coord[d] = 0;
        }
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
        .dims = @splat(0),
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

/// 校验给定的 (shape, strides) 是否满足行优先紧密连续布局
/// 注：若某维度大小为 1，其跨度步长不影响物理偏移，视为连续。
pub fn isContiguousStrides(shape: Shape, strides: Shape) bool {
    if (shape.len != strides.len) return false;
    if (shape.len == 0) return true;
    var s: usize = 1;
    var i: usize = shape.len - 1;
    while (true) {
        if (shape.dims[i] != 1 and strides.dims[i] != s) return false;
        s *= shape.dims[i];
        if (i == 0) break;
        i -= 1;
    }
    return true;
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
        .dims = @splat(0),
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
        .dims = @splat(0),
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

/// 维度折叠 (Dimension Coalescing)：去掉大小为 1 的维度，并把对所有操作数都满足
/// `stride[d] == stride[d + 1] * dims[d + 1]` 的相邻维度合并为一维，
/// 使逐行遍历的最内层循环尽可能长。返回的形状至少有 1 维（全部为 1 时为 `[1]`）。
pub fn coalesceDims(comptime n: usize, shape: Shape, strides: *[n]Shape) Shape {
    var out = Shape{ .dims = @as([8]usize, @splat(0)), .len = 0 };
    var out_strides: [n]Shape = undefined;
    for (&out_strides) |*st| st.* = Shape{ .dims = @as([8]usize, @splat(0)), .len = 0 };

    // 从最内层维度向外扫描，结果先逆序存放
    var d = shape.len;
    while (d > 0) {
        d -= 1;
        const size = shape.dims[d];
        if (size == 1) continue;
        if (out.len > 0) {
            const inner = out.len - 1;
            var mergeable = true;
            inline for (0..n) |k| {
                if (strides[k].dims[d] != out_strides[k].dims[inner] * out.dims[inner]) mergeable = false;
            }
            if (mergeable) {
                out.dims[inner] *= size;
                continue;
            }
        }
        out.dims[out.len] = size;
        inline for (0..n) |k| out_strides[k].dims[out.len] = strides[k].dims[d];
        out.len += 1;
    }
    if (out.len == 0) {
        out.dims[0] = 1;
        out.len = 1;
        inline for (0..n) |k| out_strides[k].dims[0] = 0;
    }

    // 反转为行优先顺序
    std.mem.reverse(usize, out.dims[0..out.len]);
    inline for (0..n) |k| {
        std.mem.reverse(usize, out_strides[k].dims[0..out.len]);
        out_strides[k].len = out.len;
        strides[k] = out_strides[k];
    }
    return out;
}

/// 逐行跨步遍历器 (Strided Row Iterator)：在折叠后的形状上按行优先顺序遍历除最内层外的所有坐标，
/// 每次返回 `n` 个操作数当前行的起始偏移；最内层维度长度为 `inner`，各操作数最内层跨度为 `inner_strides`。
pub fn RowIter(comptime n: usize) type {
    return struct {
        const Self = @This();

        shape: Shape,
        strides: [n]Shape,
        coord: [8]usize = @as([8]usize, @splat(0)),
        offsets: [n]usize = @as([n]usize, @splat(0)),
        rows_left: usize,
        inner: usize,
        inner_strides: [n]usize,

        /// `strides[k]` 为第 k 个操作数在 `shape` 下的（广播）跨度
        pub fn init(shape: Shape, strides_in: [n]Shape) Self {
            var strides = strides_in;
            const folded = coalesceDims(n, shape, &strides);
            const total = shape.numel();
            const inner = folded.dims[folded.len - 1];
            var inner_strides: [n]usize = undefined;
            inline for (0..n) |k| inner_strides[k] = strides[k].dims[folded.len - 1];
            return .{
                .shape = folded,
                .strides = strides,
                .rows_left = if (total == 0) 0 else total / inner,
                .inner = inner,
                .inner_strides = inner_strides,
            };
        }

        pub fn next(self: *Self) ?[n]usize {
            if (self.rows_left == 0) return null;
            self.rows_left -= 1;
            const current = self.offsets;
            var d = self.shape.len - 1;
            while (d > 0) {
                d -= 1;
                self.coord[d] += 1;
                inline for (0..n) |k| self.offsets[k] += self.strides[k].dims[d];
                if (self.coord[d] < self.shape.dims[d]) break;
                inline for (0..n) |k| self.offsets[k] -= self.strides[k].dims[d] * self.shape.dims[d];
                self.coord[d] = 0;
            }
            return current;
        }
    };
}

/// 把任意跨步源张量按行优先顺序收集到连续目标缓冲区 (Strided Gather)；
/// `dest.len` 必须等于 `shape.numel()`，最内层连续时整行 `@memcpy`
pub fn gatherStrided(dest: []f32, src: []const f32, shape: Shape, strides: Shape) void {
    std.debug.assert(dest.len == shape.numel());
    var it = RowIter(2).init(shape, .{ computeContiguousStrides(shape), strides });
    const inner = it.inner;
    const s_src = it.inner_strides[1];
    while (it.next()) |offs| {
        const out_row = dest[offs[0]..][0..inner];
        if (s_src == 1) {
            @memcpy(out_row, src[offs[1]..][0..inner]);
        } else {
            for (out_row, 0..) |*v, i| v.* = src[offs[1] + i * s_src];
        }
    }
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
    // 快速路径：若形状完全相同且均为连续行优先跨度，直接单层循环扁平迭代
    if (A_shape.eq(B_shape) and A_shape.eq(C_shape) and
        isContiguousStrides(A_shape, A_strides) and
        isContiguousStrides(B_shape, B_strides) and
        A_data.len >= C_data.len and B_data.len >= C_data.len)
    {
        for (C_data, A_data[0..C_data.len], B_data[0..C_data.len]) |*c_val, a_val, b_val| {
            c_val.* = op(a_val, b_val);
        }
        return;
    }

    // 广播或非连续跨步路径：折叠维度后逐行遍历，最内层按跨度组合分派到可向量化的专用循环
    std.debug.assert(C_data.len == C_shape.numel());
    var it = RowIter(3).init(C_shape, .{
        computeContiguousStrides(C_shape),
        computeBroadcastStrides(A_shape, A_strides, C_shape),
        computeBroadcastStrides(B_shape, B_strides, C_shape),
    });
    const inner = it.inner;
    const sa = it.inner_strides[1];
    const sb = it.inner_strides[2];
    while (it.next()) |offs| {
        const c_row = C_data[offs[0]..][0..inner];
        if (sa == 1 and sb == 1) {
            for (c_row, A_data[offs[1]..][0..inner], B_data[offs[2]..][0..inner]) |*c_val, a_val, b_val| {
                c_val.* = op(a_val, b_val);
            }
        } else if (sa == 1 and sb == 0) {
            const b_val = B_data[offs[2]];
            for (c_row, A_data[offs[1]..][0..inner]) |*c_val, a_val| c_val.* = op(a_val, b_val);
        } else if (sa == 0 and sb == 1) {
            const a_val = A_data[offs[1]];
            for (c_row, B_data[offs[2]..][0..inner]) |*c_val, b_val| c_val.* = op(a_val, b_val);
        } else {
            for (c_row, 0..) |*c_val, i| c_val.* = op(A_data[offs[1] + i * sa], B_data[offs[2] + i * sb]);
        }
    }
}
