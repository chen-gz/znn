const std = @import("std");
const autodiff = @import("autodiff.zig");
const Op = autodiff.Op;
const c = @import("cblas.zig");

pub const Shape = struct {
    dims: [8]usize,
    len: usize,

    pub fn init(slice: []const usize) Shape {
        var self = Shape{
            .dims = [_]usize{0} ** 8,
            .len = slice.len,
        };
        for (slice, 0..) |dim, i| {
            if (i >= 8) break;
            self.dims[i] = dim;
        }
        return self;
    }

    pub fn eq(self: Shape, other: Shape) bool {
        if (self.len != other.len) return false;
        for (0..self.len) |i| {
            if (self.dims[i] != other.dims[i]) return false;
        }
        return true;
    }
};

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

pub fn transposeShape(shape: Shape, dim0: usize, dim1: usize) Shape {
    var new_shape = shape;
    const tmp = new_shape.dims[dim0];
    new_shape.dims[dim0] = new_shape.dims[dim1];
    new_shape.dims[dim1] = tmp;
    return new_shape;
}

// 张量（Tensor）结构体：物理数据与元数据
pub const Tensor = struct {
    data: []f32,          // 前向传播的数据缓冲区（行优先存储的一维切片）
    grad: []f32,          // 反向传播的梯度缓冲区（与 data 形状一致，不需梯度的节点可为空）
    shape: Shape,         // 逻辑形状
    strides: Shape,       // 各维度的跨度步长
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

pub fn free(allocator: std.mem.Allocator, t: *Tensor) void {
    allocator.free(t.data);
    if (t.requires_grad and t.grad.len > 0) {
        allocator.free(t.grad);
    }
    allocator.destroy(t);
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



