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
        std.debug.assert(N == targets.data.len);
        var sum: f32 = 0.0;
        for (0..N) |i| {
            const x = self.data[i];
            const y = targets.data[i];
            const max_x = @max(x, 0.0);
            const abs_x = @abs(x);
            sum += max_x - x * y + @log(1.0 + @exp(-abs_x));
        }
        loss.data[0] = sum / @as(f32, @floatFromInt(N));
        return loss;
    }

    pub fn bceLoss(self: *Tensor, targets: *Tensor, eps: f32, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.bceLoss(self, targets, eps);
        }
        const loss = try zeros(allocator, &.{ 1, 1 });
        const N = self.data.len;
        std.debug.assert(N == targets.data.len);
        var sum: f32 = 0.0;
        for (0..N) |i| {
            const p = self.data[i];
            const y = targets.data[i];
            const p_clip = @max(p, eps);
            const one_minus_p_clip = @max(1.0 - p, eps);
            sum += -(y * @log(p_clip) + (1.0 - y) * @log(one_minus_p_clip));
        }
        loss.data[0] = sum / @as(f32, @floatFromInt(N));
        return loss;
    }

    pub fn fillNormal(self: *Tensor, random: std.Random, mean: f32, stddev: f32) void {
        var i: usize = 0;
        const len = self.data.len;
        while (i < len) {
            var u_1: f32 = random.float(f32);
            while (u_1 == 0.0) {
                u_1 = random.float(f32);
            }
            const u_2 = random.float(f32);
            const z0 = @sqrt(-2.0 * @log(u_1)) * @cos(2.0 * std.math.pi * u_2);
            self.data[i] = mean + z0 * stddev;
            i += 1;
            if (i < len) {
                const z1 = @sqrt(-2.0 * @log(u_1)) * @sin(2.0 * std.math.pi * u_2);
                self.data[i] = mean + z1 * stddev;
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

    pub fn sigmoidCrossEntropy(self: *Tensor, targets: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.sigmoidCrossEntropy(self, targets);
        }
        const loss = try zeros(allocator, &.{1, 1});
        const N = self.data.len;
        std.debug.assert(N == targets.data.len);

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

    pub fn conv2d(self: *Tensor, weight: *Tensor, bias: ?*Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.conv2d(self, weight, bias);
        }
        std.debug.assert(self.shape.len == 4);
        std.debug.assert(weight.shape.len == 4);
        const N = self.shape.dims[0];
        const C_in = self.shape.dims[1];
        const H = self.shape.dims[2];
        const W = self.shape.dims[3];

        const C_out = weight.shape.dims[0];
        std.debug.assert(weight.shape.dims[1] == C_in);
        const KH = weight.shape.dims[2];
        const KW = weight.shape.dims[3];

        if (bias) |b| {
            std.debug.assert(b.shape.len == 1);
            std.debug.assert(b.shape.dims[0] == C_out);
        }

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
                        var sum: f32 = b_val;
                        for (0..C_in) |ci| {
                            for (0..KH) |kh| {
                                for (0..KW) |kw| {
                                    const input_val = self.data[n * s_n + ci * s_c + (h + kh) * s_h + (w + kw) * s_w];
                                    const weight_val = weight.data[co * w_co + ci * w_ci + kh * w_kh + kw * w_kw];
                                    sum += input_val * weight_val;
                                }
                            }
                        }
                        out.data[n * o_n + co * o_c + h * o_h + w * o_w] = sum;
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
        std.debug.assert(self.shape.len == 4);
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

            var sum: f32 = 0.0;
            for (row_in, row_out) |val, *p| {
                const exp_val = @exp(val - max_val);
                p.* = exp_val;
                sum += exp_val;
            }

            for (row_out) |*p| {
                p.* /= sum;
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
        std.debug.assert(self.shape.len == 4);
        std.debug.assert(other.shape.len == 4);

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

/// Solves linear system A * x = b using Gauss-Jordan elimination with partial pivoting.
/// A is an n x n row-major matrix slice, b is an n-element vector, out_x is an n-element output slice.
pub fn solveLinearSystem(allocator: std.mem.Allocator, A_data: []const f32, b_data: []const f32, n: usize, out_x: []f32) !void {
    std.debug.assert(A_data.len == n * n);
    std.debug.assert(b_data.len == n);
    std.debug.assert(out_x.len == n);

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
    std.debug.assert(x.len == n_samples * n_features);
    std.debug.assert(y.len == n_samples);
    std.debug.assert(out_w.len == n_features);

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













