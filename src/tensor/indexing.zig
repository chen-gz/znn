//! Advanced indexing, scatter/gather operations, index selection, and masked selection.
//!
//! PyTorch-aligned multidimensional tensor indexing primitives:
//! - `gather`: along axis `dim`, collects elements indexed by `index` tensor
//! - `gatherInto`: destination-driven zero-allocation variant
//! - `scatter`: writes values from `src` into `self` at positions specified by `index`
//! - `scatterInto`: 5-parameter destination-driven variant: `dest = copy(base); dest[index] = src`
//! - `scatterAdd`: in-place or new tensor addition at positions specified by `index`
//! - `scatterAddInto`: 4-parameter zero-allocation in-place accumulation: `dest[index] += src`
//! - `indexSelect`: extracts slices along 1D index tensor
//! - `indexSelectInto`: destination-driven zero-allocation slice extraction
//! - `indexAdd`: in-place or new tensor addition along 1D index tensor
//! - `indexAddInto`: 4-parameter zero-allocation in-place slice addition
//! - `maskedSelect`: extracts 1D tensor where boolean mask is true

const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const core = @import("core.zig");
const Tensor = core.Tensor;
const Shape = types.Shape;
const GenericTensor = types.GenericTensor;
const indexScalarToUsize = types.indexScalarToUsize;
const ops_mod = @import("ops.zig");
const shape_mod = @import("shape.zig");

pub const IndexingError = error{
    IndexOutOfBounds,
    InvalidIndexDimension,
    IndexShapeMismatch,
    ScatterDimensionMismatch,
    DimensionOutOfBounds,
};

inline fn coordToOffset(coord: []const usize, strides: Shape) usize {
    var off: usize = 0;
    for (coord, 0..) |c, d| {
        off += c * strides.dims[d];
    }
    return off;
}

inline fn advanceCoord(coord: []usize, shape: []const usize) bool {
    var d = coord.len;
    while (d > 0) {
        d -= 1;
        coord[d] += 1;
        if (coord[d] < shape[d]) return true;
        coord[d] = 0;
    }
    return false;
}

/// 通用轴向索引收集：沿着指定维度 dim 按照 index 张量中的坐标抽取元素
/// 要求：
/// 1. self.shape.len == index.shape.len (同秩)
/// 2. 除了 dim 维度之外，index 在所有其他维度上的大小不得大于 self 对应维度
/// 3. 输出形状与 index 的形状严格一致
pub fn gatherInto(
    dest: *Tensor,
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
) !void {
    if (dim >= self.shape.len) return error.DimensionOutOfBounds;
    if (self.shape.len != index.shape.len) return error.IndexShapeMismatch;
    if (!dest.shape.eq(index.shape)) return error.IndexShapeMismatch;

    for (0..self.shape.len) |d| {
        if (d != dim and index.shape.dims[d] > self.shape.dims[d]) {
            return error.IndexShapeMismatch;
        }
    }

    const rank = index.shape.len;
    const total_numel = index.shape.numel();
    if (total_numel == 0) return;

    var coord: [8]usize = @splat(0);
    var self_coord: [8]usize = @splat(0);
    const dim_size = self.shape.dims[dim];

    for (0..total_numel) |_| {
        const idx_offset = coordToOffset(coord[0..rank], index.strides);
        const raw_k = index.data[idx_offset];
        const k = try indexScalarToUsize(f32, raw_k, dim_size);

        for (0..rank) |d| {
            self_coord[d] = if (d == dim) k else coord[d];
        }

        const src_offset = coordToOffset(self_coord[0..rank], self.strides);
        const dest_offset = coordToOffset(coord[0..rank], dest.strides);
        dest.data[dest_offset] = self.data[src_offset];

        _ = advanceCoord(coord[0..rank], index.shape.dims[0..rank]);
    }
}

pub fn gather(
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
    allocator: Allocator,
) !*Tensor {
    if (dim >= self.shape.len) return error.DimensionOutOfBounds;
    if (self.shape.len != index.shape.len) return error.IndexShapeMismatch;
    for (0..self.shape.len) |d| {
        if (d != dim and index.shape.dims[d] > self.shape.dims[d]) {
            return error.IndexShapeMismatch;
        }
    }
    const out = try ops_mod.allocUninit(allocator, index.shape);
    errdefer ops_mod.free(allocator, out);
    try gatherInto(out, self, dim, index);
    return out;
}

/// 通用轴向离散覆写 (Scatter)：将 src 中的元素按照 index 散布覆写至 dest
/// 5 参数目标写入散布：首先拷贝 base 至 dest，随后按照 index 坐标将 src 写入 dest
pub fn scatterInto(
    dest: *Tensor,
    base: *const Tensor,
    dim: usize,
    index: *const Tensor,
    src: *Tensor,
) !void {
    if (dim >= dest.shape.len) return error.DimensionOutOfBounds;
    if (dest.shape.len != base.shape.len or !dest.shape.eq(base.shape)) return error.IndexShapeMismatch;
    if (dest.shape.len != index.shape.len) return error.IndexShapeMismatch;
    if (dest.shape.len != src.shape.len) return error.IndexShapeMismatch;

    for (0..dest.shape.len) |d| {
        if (index.shape.dims[d] > src.shape.dims[d]) return error.IndexShapeMismatch;
        if (d != dim and index.shape.dims[d] > dest.shape.dims[d]) return error.IndexShapeMismatch;
    }

    if (dest.data.ptr != base.data.ptr) {
        if (dest.isContiguous() and base.isContiguous() and dest.data.len == base.data.len) {
            @memcpy(dest.data, base.data);
        } else {
            shape_mod.gatherStrided(dest.data, base.data, base.shape, base.strides);
        }
    }

    const rank = index.shape.len;
    const total_numel = index.shape.numel();
    if (total_numel == 0) return;

    var coord: [8]usize = @splat(0);
    var dest_coord: [8]usize = @splat(0);
    const dim_size = dest.shape.dims[dim];

    for (0..total_numel) |_| {
        const idx_offset = coordToOffset(coord[0..rank], index.strides);
        const raw_k = index.data[idx_offset];
        const k = try indexScalarToUsize(f32, raw_k, dim_size);

        for (0..rank) |d| {
            dest_coord[d] = if (d == dim) k else coord[d];
        }

        const src_offset = coordToOffset(coord[0..rank], src.strides);
        const dest_offset = coordToOffset(dest_coord[0..rank], dest.strides);
        dest.data[dest_offset] = src.data[src_offset];

        _ = advanceCoord(coord[0..rank], index.shape.dims[0..rank]);
    }
}

pub fn scatter(
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
    src: *Tensor,
    allocator: Allocator,
) !*Tensor {
    const out = try ops_mod.allocUninit(allocator, self.shape);
    errdefer ops_mod.free(allocator, out);
    try scatterInto(out, self, dim, index, src);
    return out;
}

/// 通用轴向离散累加 (Scatter-Add)：将 src 中的元素按照 index 散布并累加至 dest
/// 4 参数零分配原位累加算子：dest[...][index[...]][...] += src[...][...][...]
pub fn scatterAddInto(
    dest: *Tensor,
    dim: usize,
    index: *const Tensor,
    src: *Tensor,
) !void {
    if (dim >= dest.shape.len) return error.DimensionOutOfBounds;
    if (dest.shape.len != index.shape.len) return error.IndexShapeMismatch;
    if (dest.shape.len != src.shape.len) return error.IndexShapeMismatch;

    for (0..dest.shape.len) |d| {
        if (index.shape.dims[d] > src.shape.dims[d]) return error.IndexShapeMismatch;
        if (d != dim and index.shape.dims[d] > dest.shape.dims[d]) return error.IndexShapeMismatch;
    }

    const rank = index.shape.len;
    const total_numel = index.shape.numel();
    if (total_numel == 0) return;

    var coord: [8]usize = @splat(0);
    var dest_coord: [8]usize = @splat(0);
    const dim_size = dest.shape.dims[dim];

    for (0..total_numel) |_| {
        const idx_offset = coordToOffset(coord[0..rank], index.strides);
        const raw_k = index.data[idx_offset];
        const k = try indexScalarToUsize(f32, raw_k, dim_size);

        for (0..rank) |d| {
            dest_coord[d] = if (d == dim) k else coord[d];
        }

        const src_offset = coordToOffset(coord[0..rank], src.strides);
        const dest_offset = coordToOffset(dest_coord[0..rank], dest.strides);
        dest.data[dest_offset] += src.data[src_offset];

        _ = advanceCoord(coord[0..rank], index.shape.dims[0..rank]);
    }
}

pub fn scatterAdd(
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
    src: *Tensor,
    allocator: Allocator,
) !*Tensor {
    const out = try ops_mod.allocUninit(allocator, self.shape);
    errdefer ops_mod.free(allocator, out);
    if (out.isContiguous() and self.isContiguous() and out.data.len == self.data.len) {
        @memcpy(out.data, self.data);
    } else {
        shape_mod.gatherStrided(out.data, self.data, self.shape, self.strides);
    }
    try scatterAddInto(out, dim, index, src);
    return out;
}

/// 一维子张量索引切片提取：沿指定 dim 提取 index (1D) 指定的超平面
pub fn indexSelectInto(
    dest: *Tensor,
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
) !void {
    if (dim >= self.shape.len) return error.DimensionOutOfBounds;
    if (index.shape.len != 1) return error.IndexShapeMismatch;
    if (dest.shape.len != self.shape.len) return error.IndexShapeMismatch;
    const K = index.shape.dims[0];
    if (dest.shape.dims[dim] != K) return error.IndexShapeMismatch;
    for (0..self.shape.len) |d| {
        if (d != dim and dest.shape.dims[d] != self.shape.dims[d]) {
            return error.IndexShapeMismatch;
        }
    }

    const rank = dest.shape.len;
    const total_numel = dest.shape.numel();
    if (total_numel == 0) return;

    var out_coord: [8]usize = @splat(0);
    var in_coord: [8]usize = @splat(0);
    const dim_size = self.shape.dims[dim];

    for (0..total_numel) |_| {
        const j = out_coord[dim];
        const raw_k = index.data[j * index.strides.dims[0]];
        const k = try indexScalarToUsize(f32, raw_k, dim_size);

        for (0..rank) |d| {
            in_coord[d] = if (d == dim) k else out_coord[d];
        }

        const src_offset = coordToOffset(in_coord[0..rank], self.strides);
        const dest_offset = coordToOffset(out_coord[0..rank], dest.strides);
        dest.data[dest_offset] = self.data[src_offset];

        _ = advanceCoord(out_coord[0..rank], dest.shape.dims[0..rank]);
    }
}

pub fn indexSelect(
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
    allocator: Allocator,
) !*Tensor {
    if (dim >= self.shape.len) return error.DimensionOutOfBounds;
    if (index.shape.len != 1) return error.IndexShapeMismatch;
    var out_dims: [8]usize = @splat(0);
    for (0..self.shape.len) |d| {
        out_dims[d] = if (d == dim) index.shape.dims[0] else self.shape.dims[d];
    }
    const out_shape = Shape{ .dims = out_dims, .len = self.shape.len };
    const out = try ops_mod.allocUninit(allocator, out_shape);
    errdefer ops_mod.free(allocator, out);
    try indexSelectInto(out, self, dim, index);
    return out;
}

/// 一维子张量索引切片累加 (Index-Add)：将 src 的切片按照一维 index 坐标累加至 dest 沿 dim 维度的对应超平面
/// 4 参数零分配原位累加：dest[..., index[i], ...] += src[..., i, ...]
pub fn indexAddInto(
    dest: *Tensor,
    dim: usize,
    index: *const Tensor,
    src: *Tensor,
) !void {
    if (dim >= dest.shape.len) return error.DimensionOutOfBounds;
    if (index.shape.len != 1) return error.IndexShapeMismatch;
    if (src.shape.len != dest.shape.len) return error.IndexShapeMismatch;
    const K = index.shape.dims[0];
    if (src.shape.dims[dim] != K) return error.IndexShapeMismatch;
    for (0..dest.shape.len) |d| {
        if (d != dim and src.shape.dims[d] != dest.shape.dims[d]) {
            return error.IndexShapeMismatch;
        }
    }

    const rank = src.shape.len;
    const total_numel = src.shape.numel();
    if (total_numel == 0) return;

    var src_coord: [8]usize = @splat(0);
    var dest_coord: [8]usize = @splat(0);
    const dim_size = dest.shape.dims[dim];

    for (0..total_numel) |_| {
        const j = src_coord[dim];
        const raw_k = index.data[j * index.strides.dims[0]];
        const k = try indexScalarToUsize(f32, raw_k, dim_size);

        for (0..rank) |d| {
            dest_coord[d] = if (d == dim) k else src_coord[d];
        }

        const src_offset = coordToOffset(src_coord[0..rank], src.strides);
        const dest_offset = coordToOffset(dest_coord[0..rank], dest.strides);
        dest.data[dest_offset] += src.data[src_offset];

        _ = advanceCoord(src_coord[0..rank], src.shape.dims[0..rank]);
    }
}

pub fn indexAdd(
    self: *Tensor,
    dim: usize,
    index: *const Tensor,
    src: *Tensor,
    allocator: Allocator,
) !*Tensor {
    const out = try ops_mod.allocUninit(allocator, self.shape);
    errdefer ops_mod.free(allocator, out);
    if (out.isContiguous() and self.isContiguous() and out.data.len == self.data.len) {
        @memcpy(out.data, self.data);
    } else {
        shape_mod.gatherStrided(out.data, self.data, self.shape, self.strides);
    }
    try indexAddInto(out, dim, index, src);
    return out;
}

/// 布尔掩码一维压缩提取：按照布尔掩码 mask 提取所有非零/真值元素，返回 1D 连续扁平张量
pub fn maskedSelect(
    self: *Tensor,
    mask: *const Tensor,
    allocator: Allocator,
) !*Tensor {
    if (!self.shape.eq(mask.shape)) return error.IndexShapeMismatch;
    const rank = self.shape.len;
    const total_numel = self.shape.numel();

    // Pass 1: count active mask elements
    var count: usize = 0;
    var coord: [8]usize = @splat(0);
    for (0..total_numel) |_| {
        const mask_off = coordToOffset(coord[0..rank], mask.strides);
        if (mask.data[mask_off] != 0.0) {
            count += 1;
        }
        _ = advanceCoord(coord[0..rank], self.shape.dims[0..rank]);
    }

    // Allocate 1D output
    var out_dims: [8]usize = @splat(0);
    out_dims[0] = count;
    const out_shape = Shape{ .dims = out_dims, .len = 1 };
    const out = try ops_mod.allocUninit(allocator, out_shape);
    errdefer ops_mod.free(allocator, out);
    if (count == 0) return out;

    // Pass 2: copy selected elements
    coord = @splat(0);
    var write_i: usize = 0;
    for (0..total_numel) |_| {
        const mask_off = coordToOffset(coord[0..rank], mask.strides);
        if (mask.data[mask_off] != 0.0) {
            const self_off = coordToOffset(coord[0..rank], self.strides);
            out.data[write_i] = self.data[self_off];
            write_i += 1;
        }
        _ = advanceCoord(coord[0..rank], self.shape.dims[0..rank]);
    }

    return out;
}

/// 泛型切片辅助封装：直接传入 []const usize 索引列表提取一维切片
pub fn gatherSlice(
    self: *Tensor,
    dim: usize,
    indices: []const usize,
    allocator: Allocator,
) !*Tensor {
    const idx_tensor = try ops_mod.zeros(allocator, &.{indices.len});
    defer ops_mod.free(allocator, idx_tensor);
    for (indices, 0..) |v, i| {
        idx_tensor.data[i] = @floatFromInt(v);
    }
    return try indexSelect(self, dim, idx_tensor, allocator);
}
