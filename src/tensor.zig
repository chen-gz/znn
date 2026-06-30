const std = @import("std");
const autodiff = @import("autodiff.zig");
const Op = autodiff.Op;
const c = @import("cblas.zig");

// ============================================================================
// 1. 维度与形状控制（Shape & Strides Meta-data）
// ============================================================================

/// 多维张量的形状描述体（Shape）
/// 为避免动态内存分配带来的开销，本框架采用静态数组 `[8]usize` 存储各维度大小，最多支持 8 维张量。
pub const Shape = struct {
    dims: [8]usize, // 存储每一维度大小的静态数组，未使用的维度默认为 0
    len: usize,     // 张量的维度个数（Rank，例如 2D 矩阵的 Rank 为 2）

    /// 根据动态传入的切片初始化静态 Shape 结构体
    pub fn init(slice: []const usize) Shape {
        var self = Shape{
            .dims = [_]usize{0} ** 8,
            .len = slice.len,
        };
        for (slice, 0..) |dim, i| {
            if (i >= 8) break; // 超过 8 维截断
            self.dims[i] = dim;
        }
        return self;
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

// ============================================================================
// 2. 张量（Tensor）核心定义与元数据
// ============================================================================

/// 张量（Tensor）结构体：承载机器学习网络中所有物理数据与流转拓扑信息
pub const Tensor = struct {
    data: []f32,          // 前向传播的数据缓冲区（行优先存储的一维切片）
    grad: []f32,          // 反向传播的梯度缓冲区（与 data 形状一致，不需梯度的节点可为空）
    shape: Shape,         // 逻辑形状
    strides: Shape,       // 各维度的跨度步长（用于非连续张量及快速视图映射）
    requires_grad: bool,  // 是否需要求梯度（如模型参数为 true，输入数据为 false）
    creator: ?*Op,        // 产生此张量的算子节点（前向图中的父节点，用于追踪计算路径）


    // 将梯度缓冲区全部清零，通常在每个 batch 反向传播前调用
    pub fn zeroGrad(self: *Tensor) void {
        if (self.requires_grad) {
            @memset(self.grad, 0.0);
        }
    }

    // 获取多维索引对应的扁平化索引
    pub fn getFlatIndex(self: Tensor, indices: []const usize) usize {
        std.debug.assert(indices.len == self.shape.len);
        var flat_idx: usize = 0;
        for (indices, 0..) |idx, i| {
            std.debug.assert(idx < self.shape.dims[i]);
            flat_idx += idx * self.strides.dims[i];
        }
        return flat_idx;
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

    pub fn addBias(self: *Tensor, bias: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.addBias(self, bias);
        }
        const M = self.shape.dims[0];
        const N = self.shape.dims[1];
        const C = try zeros(allocator, &.{M, N});
        for (0..M) |i| {
            const a_row = self.data[i * N .. (i + 1) * N];
            const c_row = C.data[i * N .. (i + 1) * N];
            for (0..N) |j| {
                c_row[j] = a_row[j] + bias.data[j];
            }
        }
        return C;
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

    pub fn add(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.add(self, other);
        }
        std.debug.assert(self.data.len == other.data.len);
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data, other.data) |*c_val, s_val, o_val| {
            c_val.* = s_val + o_val;
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

    pub fn softmaxCrossEntropy(self: *Tensor, targets: []const u8, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.softmaxCrossEntropy(self, targets);
        }
        const loss = try zeros(allocator, &.{1, 1});
        const B = self.shape.dims[0];
        const N = self.shape.dims[1];

        var loss_sum: f32 = 0.0;
        for (0..B) |i| {
            const logits_row = self.data[i * N .. (i + 1) * N];
            var max_val = logits_row[0];
            for (logits_row[1..]) |val| {
                if (val > max_val) max_val = val;
            }

            var sum: f32 = 0.0;
            for (logits_row) |val| {
                sum += @exp(val - max_val);
            }

            const label = targets[i];
            const prob = @exp(logits_row[label] - max_val) / sum;
            const clipped = @max(prob, 1e-15);
            loss_sum += -@log(clipped);
        }
        loss.data[0] = loss_sum / @as(f32, @floatFromInt(B));
        return loss;
    }

    pub fn reshape(self: *Tensor, new_shape_slice: []const usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.reshape(self, new_shape_slice);
        }
        const shape = Shape.init(new_shape_slice);
        const strides = computeContiguousStrides(shape);
        var old_total: usize = 1;
        for (0..self.shape.len) |i| {
            old_total *= self.shape.dims[i];
        }
        var new_total: usize = 1;
        for (new_shape_slice) |dim| {
            new_total *= dim;
        }
        std.debug.assert(old_total == new_total);

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

    pub fn transpose(self: *Tensor, dim0: usize, dim1: usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.transposeND(self, dim0, dim1);
        }
        std.debug.assert(dim0 < self.shape.len);
        std.debug.assert(dim1 < self.shape.len);

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
        std.debug.assert(dim < self.shape.len);
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
        std.debug.assert(dim < self.shape.len);
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

    pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        if (self.requires_grad and self.grad.len > 0) {
            allocator.free(self.grad);
        }
        allocator.destroy(self);
    }
};


// ============================================================================
// NumPy-like raw tensor creation APIs (independent of Graph)
// ============================================================================

pub fn array(allocator: std.mem.Allocator, shape_slice: []const usize, initial_data: []const f32) !*Tensor {
    const t = try allocator.create(Tensor);
    const shape = Shape.init(shape_slice);
    const strides = computeContiguousStrides(shape);
    var total_size: usize = 1;
    for (shape_slice) |dim| {
        total_size *= dim;
    }
    std.debug.assert(total_size == initial_data.len);

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
    const t = try allocator.create(Tensor);
    const shape = Shape.init(shape_slice);
    const strides = computeContiguousStrides(shape);
    var total_size: usize = 1;
    for (shape_slice) |dim| {
        total_size *= dim;
    }

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
    const t = try allocator.create(Tensor);
    const shape = Shape.init(shape_slice);
    const strides = computeContiguousStrides(shape);
    var total_size: usize = 1;
    for (shape_slice) |dim| {
        total_size *= dim;
    }

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






