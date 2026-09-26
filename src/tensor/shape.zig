const std = @import("std");

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

