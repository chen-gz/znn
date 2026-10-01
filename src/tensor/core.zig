const std = @import("std");
const autodiff = @import("../autodiff.zig");
const Op = autodiff.Op;
const c = @import("../cblas.zig");
const shape_mod = @import("shape.zig");
pub const Shape = shape_mod.Shape;
pub const computeContiguousStrides = shape_mod.computeContiguousStrides;
pub const isContiguousStrides = shape_mod.isContiguousStrides;
pub const transposeShape = shape_mod.transposeShape;
pub const broadcastShapes = shape_mod.broadcastShapes;
pub const computeBroadcastStrides = shape_mod.computeBroadcastStrides;
pub const broadcastBinaryOpRaw = shape_mod.broadcastBinaryOpRaw;

const types_mod = @import("types.zig");
pub const DType = types_mod.DType;
pub const bf16 = types_mod.bf16;
pub const SliceRange = types_mod.SliceRange;
pub const GenericTensor = types_mod.GenericTensor;
pub const convertScalar = types_mod.convertScalar;
pub const isTruthyScalar = types_mod.isTruthyScalar;

const ops_mod = @import("ops.zig");
pub const array = ops_mod.array;
pub const zeros = ops_mod.zeros;
pub const free = ops_mod.free;
const tensorSplit = ops_mod.split;

extern fn erff(x: f32) f32;

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
    is_buffer: bool = false, // 是否为静态缓冲区/非学习常量张量 (如因果掩码 causal_mask, 位置索引 pos_indices)
    scope: []const u8 = "", // 张量在计算图中创建时所处的模块作用域完整路径 ("" 表示根作用域或图外创建)
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

    fn unaryMathEager(self: *const Tensor, allocator: std.mem.Allocator, comptime math_fn: fn (f32) f32) !*Tensor {
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        if (self.isContiguous() and self.data.len >= C.data.len) {
            for (C.data, self.data[0..C.data.len]) |*c_val, a_val| {
                c_val.* = math_fn(a_val);
            }
        } else {
            var coord = [_]usize{0} ** 8;
            const len = self.shape.len;
            for (0..C.data.len) |dest_i| {
                var src_idx: usize = 0;
                for (0..len) |d| src_idx += coord[d] * self.strides.dims[d];
                C.data[dest_i] = math_fn(self.data[src_idx]);
                var d = len;
                while (d > 0) {
                    d -= 1;
                    coord[d] += 1;
                    if (coord[d] < self.shape.dims[d]) break;
                    coord[d] = 0;
                }
            }
        }
        return C;
    }

    /// 逐元素开平方 (np.sqrt)
    pub fn sqrt(self: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.sqrt(self);
        }
        return self.unaryMathEager(allocator, struct {
            fn op(x: f32) f32 {
                return @sqrt(x);
            }
        }.op);
    }

    /// 逐元素自然指数 (np.exp)
    pub fn exp(self: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.exp(self);
        }
        return self.unaryMathEager(allocator, struct {
            fn op(x: f32) f32 {
                return @exp(x);
            }
        }.op);
    }

    /// 逐元素自然对数 (np.log)
    pub fn log(self: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.log(self);
        }
        return self.unaryMathEager(allocator, struct {
            fn op(x: f32) f32 {
                return @log(x);
            }
        }.op);
    }

    /// 逐元素绝对值 (np.abs)
    pub fn abs(self: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.abs(self);
        }
        return self.unaryMathEager(allocator, struct {
            fn op(x: f32) f32 {
                return @abs(x);
            }
        }.op);
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
        return self.bceWithLogitsLoss(targets, allocator, graph);
    }

    pub fn mseLoss(self: *Tensor, targets: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.mseLoss(self, targets);
        }
        const N = self.data.len;
        if (N != targets.data.len) return error.ShapeMismatch;
        const loss = try zeros(allocator, &.{ 1, 1 });
        var loss_sum: f32 = 0.0;
        for (0..N) |i| {
            const diff = self.data[i] - targets.data[i];
            loss_sum += diff * diff;
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

    pub fn l1Loss(self: *Tensor, lambda: f32, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.l1Loss(self, lambda);
        }
        const loss = try zeros(allocator, &.{ 1, 1 });
        var sum_abs: f32 = 0.0;
        for (self.data) |v| {
            sum_abs += @abs(v);
        }
        loss.data[0] = lambda * sum_abs;
        return loss;
    }

    pub fn reshape(self: *Tensor, new_shape_slice: []const usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.reshape(self, new_shape_slice);
        }
        const shape = try Shape.fromSlice(new_shape_slice);
        const strides = computeContiguousStrides(shape);
        const old_total = self.shape.numel();
        const new_total = shape.numel();
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
        if (self.isContiguous() and self.data.len >= new_total) {
            @memcpy(C.data, self.data[0..new_total]);
        } else {
            var coord = [_]usize{0} ** 8;
            const len = self.shape.len;
            for (0..new_total) |dest_i| {
                var src_idx: usize = 0;
                for (0..len) |d| {
                    src_idx += coord[d] * self.strides.dims[d];
                }
                C.data[dest_i] = self.data[src_idx];

                var d = len;
                while (d > 0) {
                    d -= 1;
                    coord[d] += 1;
                    if (coord[d] < self.shape.dims[d]) break;
                    coord[d] = 0;
                }
            }
        }
        return C;
    }

    pub fn split(self: *Tensor, num_splits: usize, dim: usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror![]*Tensor {
        return tensorSplit(allocator, self, num_splits, dim, graph);
    }

    /// 零拷贝维度转置视图（仅交换 shape 与 strides，共享底层 data 与 grad 缓冲区）
    pub fn transposeView(self: *Tensor, dim0: usize, dim1: usize, allocator: std.mem.Allocator) !*Tensor {
        if (dim0 >= self.shape.len or dim1 >= self.shape.len) {
            return error.DimensionOutOfBounds;
        }
        const view = try allocator.create(Tensor);
        view.* = Tensor{
            .data = self.data,
            .grad = self.grad,
            .shape = transposeShape(self.shape, dim0, dim1),
            .strides = transposeShape(self.strides, dim0, dim1),
            .requires_grad = self.requires_grad,
            .creator = null,
            .is_view = true,
        };
        return view;
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

    /// 沿指定轴重复张量元素 repeats 次 (np.repeat)
    /// axis 若为 null，则先将张量展平后重复
    pub fn repeat(self: *Tensor, repeats: usize, axis: ?usize, allocator: std.mem.Allocator) !*Tensor {
        if (repeats == 0) return try zeros(allocator, &.{0});
        if (axis) |ax| {
            if (ax >= self.shape.len) return error.DimensionOutOfBounds;
            if (repeats == 1) return self.clone(allocator);

            var out_shape = self.shape;
            out_shape.dims[ax] = self.shape.dims[ax] * repeats;
            const C = try zeros(allocator, out_shape.dims[0..out_shape.len]);

            const dim_size = self.shape.dims[ax];
            var outer_size: usize = 1;
            for (0..ax) |d| outer_size *= self.shape.dims[d];
            var inner_size: usize = 1;
            for (ax + 1..self.shape.len) |d| inner_size *= self.shape.dims[d];

            const src_contig = try self.contiguous(allocator);
            defer free(allocator, src_contig);

            for (0..outer_size) |outer| {
                for (0..dim_size) |idx| {
                    const src_offset = (outer * dim_size + idx) * inner_size;
                    const src_slice = src_contig.data[src_offset .. src_offset + inner_size];
                    for (0..repeats) |r| {
                        const dest_offset = (outer * (dim_size * repeats) + (idx * repeats + r)) * inner_size;
                        @memcpy(C.data[dest_offset .. dest_offset + inner_size], src_slice);
                    }
                }
            }
            return C;
        } else {
            // Flatten first
            const total = self.data.len;
            const C = try zeros(allocator, &.{ total * repeats });
            const src_contig = try self.contiguous(allocator);
            defer free(allocator, src_contig);

            for (0..total) |i| {
                const val = src_contig.data[i];
                for (0..repeats) |r| {
                    C.data[i * repeats + r] = val;
                }
            }
            return C;
        }
    }

    /// 构造通过沿各维度重复 reps 次平铺的新张量 (np.tile)
    pub fn tile(self: *Tensor, reps: []const usize, allocator: std.mem.Allocator) !*Tensor {
        if (reps.len == 0) return self.clone(allocator);
        const rank = @max(self.shape.len, reps.len);
        if (rank > 8) return error.MaxDimensionsExceeded;

        var full_self_shape = [_]usize{1} ** 8;
        var full_reps = [_]usize{1} ** 8;
        var out_shape_dims = [_]usize{1} ** 8;

        const self_offset = rank - self.shape.len;
        for (0..self.shape.len) |i| {
            full_self_shape[self_offset + i] = self.shape.dims[i];
        }

        const reps_offset = rank - reps.len;
        for (0..reps.len) |i| {
            full_reps[reps_offset + i] = reps[i];
        }

        for (0..rank) |d| {
            out_shape_dims[d] = full_self_shape[d] * full_reps[d];
        }

        const out = try zeros(allocator, out_shape_dims[0..rank]);
        const src_contig = try self.contiguous(allocator);
        defer free(allocator, src_contig);

        const src_full_shape = try Shape.fromSlice(full_self_shape[0..rank]);
        const src_strides = computeContiguousStrides(src_full_shape);

        var coord = [_]usize{0} ** 8;
        for (0..out.data.len) |dest_i| {
            var src_flat: usize = 0;
            for (0..rank) |d| {
                const src_dim_idx = coord[d] % full_self_shape[d];
                src_flat += src_dim_idx * src_strides.dims[d];
            }
            out.data[dest_i] = src_contig.data[src_flat];

            var d = rank;
            while (d > 0) {
                d -= 1;
                coord[d] += 1;
                if (coord[d] < out_shape_dims[d]) break;
                coord[d] = 0;
            }
        }
        return out;
    }

    pub fn conv2d(self: *Tensor, weight: *Tensor, bias: ?*Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        return self.conv2dWithConfig(weight, bias, 1, 0, allocator, graph);
    }

    pub fn conv2dWithConfig(
        self: *Tensor,
        weight: *Tensor,
        bias: ?*Tensor,
        stride: usize,
        padding: usize,
        allocator: std.mem.Allocator,
        graph: ?*autodiff.Graph,
    ) anyerror!*Tensor {
        if (graph) |g| {
            return try g.conv2dWithConfig(self, weight, bias, stride, padding);
        }
        if (self.shape.len != 4 or weight.shape.len != 4) {
            return error.IncompatibleDimensions;
        }
        if (stride == 0) {
            return error.InvalidStride;
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

        const H_padded = H + 2 * padding;
        const W_padded = W + 2 * padding;
        if (H_padded < KH or W_padded < KW) return error.KernelBiggerThanInput;

        const H_out = (H_padded - KH) / stride + 1;
        const W_out = (W_padded - KW) / stride + 1;

        const out = try zeros(allocator, &.{ N, C_out, H_out, W_out });
        errdefer out.deinit(allocator);

        const s_n = self.strides.dims[0];
        const s_c = self.strides.dims[1];
        const s_h = self.strides.dims[2];
        const s_w = self.strides.dims[3];

        const K_col = C_in * KH * KW;
        const L_out = H_out * W_out;

        if (weight.isContiguous() and K_col > 0 and L_out > 0 and C_out > 0) {
            const col_buf = try allocator.alloc(f32, K_col * L_out);
            defer allocator.free(col_buf);

            for (0..N) |n| {
                // im2col: 展平当前样本的所有感受野窗口为 [K_col, L_out] 矩阵
                for (0..C_in) |ci| {
                    for (0..KH) |kh| {
                        for (0..KW) |kw| {
                            const k_row = (ci * KH + kh) * KW + kw;
                            const col_row = col_buf[k_row * L_out .. (k_row + 1) * L_out];
                            for (0..H_out) |h_out| {
                                const ih_signed: isize = @as(isize, @intCast(h_out * stride + kh)) - @as(isize, @intCast(padding));
                                if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) {
                                    @memset(col_row[h_out * W_out .. (h_out + 1) * W_out], 0.0);
                                    continue;
                                }
                                const ih: usize = @intCast(ih_signed);
                                for (0..W_out) |w_out| {
                                    const iw_signed: isize = @as(isize, @intCast(w_out * stride + kw)) - @as(isize, @intCast(padding));
                                    if (iw_signed >= 0 and iw_signed < @as(isize, @intCast(W))) {
                                        const iw: usize = @intCast(iw_signed);
                                        col_row[h_out * W_out + w_out] = self.data[n * s_n + ci * s_c + ih * s_h + iw * s_w];
                                    } else {
                                        col_row[h_out * W_out + w_out] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }

                const out_n = out.data[n * C_out * L_out .. (n + 1) * C_out * L_out];
                if (bias) |b| {
                    const b_stride = b.strides.dims[0];
                    for (0..C_out) |co| {
                        @memset(out_n[co * L_out .. (co + 1) * L_out], b.data[co * b_stride]);
                    }
                }

                c.cblas_sgemm(
                    c.CblasRowMajor,
                    c.CblasNoTrans,
                    c.CblasNoTrans,
                    @intCast(C_out),
                    @intCast(L_out),
                    @intCast(K_col),
                    1.0,
                    weight.data.ptr,
                    @intCast(K_col),
                    col_buf.ptr,
                    @intCast(L_out),
                    if (bias != null) @as(f32, 1.0) else @as(f32, 0.0),
                    out_n.ptr,
                    @intCast(L_out),
                );
            }
            return out;
        }

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
                const b_val = if (bias) |b| b.data[co * b.strides.dims[0]] else 0.0;
                for (0..H_out) |h_out| {
                    for (0..W_out) |w_out| {
                        var acc: f32 = b_val;
                        for (0..C_in) |ci| {
                            for (0..KH) |kh| {
                                const ih_signed: isize = @as(isize, @intCast(h_out * stride + kh)) - @as(isize, @intCast(padding));
                                if (ih_signed < 0 or ih_signed >= @as(isize, @intCast(H))) continue;
                                const ih: usize = @intCast(ih_signed);
                                for (0..KW) |kw| {
                                    const iw_signed: isize = @as(isize, @intCast(w_out * stride + kw)) - @as(isize, @intCast(padding));
                                    if (iw_signed < 0 or iw_signed >= @as(isize, @intCast(W))) continue;
                                    const iw: usize = @intCast(iw_signed);
                                    const input_val = self.data[n * s_n + ci * s_c + ih * s_h + iw * s_w];
                                    const weight_val = weight.data[co * w_co + ci * w_ci + kh * w_kh + kw * w_kw];
                                    acc += input_val * weight_val;
                                }
                            }
                        }
                        out.data[n * o_n + co * o_c + h_out * o_h + w_out * o_w] = acc;
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

    pub fn avgpool2d(self: *Tensor, kernel_size: usize, stride: usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.avgpool2d(self, kernel_size, stride);
        }
        if (self.shape.len != 4) return error.IncompatibleDimensions;
        if (stride == 0 or kernel_size == 0) return error.InvalidStride;
        const N = self.shape.dims[0];
        const C = self.shape.dims[1];
        const H = self.shape.dims[2];
        const W = self.shape.dims[3];
        if (kernel_size > H or kernel_size > W) return error.KernelBiggerThanInput;

        const out_h = (H - kernel_size) / stride + 1;
        const out_w = (W - kernel_size) / stride + 1;
        const out = try zeros(allocator, &.{ N, C, out_h, out_w });
        const pool_area = @as(f32, @floatFromInt(kernel_size * kernel_size));

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
                for (0..out_h) |oh| {
                    for (0..out_w) |ow| {
                        const ih_start = oh * stride;
                        const iw_start = ow * stride;
                        var sum_val: f32 = 0.0;

                        for (0..kernel_size) |kh| {
                            for (0..kernel_size) |kw| {
                                const ih = ih_start + kh;
                                const iw = iw_start + kw;
                                sum_val += self.data[n * s_n + c_ * s_c + ih * s_h + iw * s_w];
                            }
                        }
                        out.data[n * o_n + c_ * o_c + oh * o_h + ow * o_w] = sum_val / pool_area;
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

    pub fn layerNorm(self: *Tensor, G: *Tensor, B: *Tensor, eps: f32, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.layerNorm(self, G, B, eps);
        }
        const D = self.shape.dims[self.shape.len - 1];
        if (G.data.len != D or B.data.len != D) return error.ShapeMismatch;
        const M = self.data.len / D;
        const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        const d_f = @as(f32, @floatFromInt(D));

        for (0..M) |i| {
            const row_in = self.data[i * D .. (i + 1) * D];
            const row_out = Y.data[i * D .. (i + 1) * D];

            var sum_x: f32 = 0.0;
            for (row_in) |val| sum_x += val;
            const mean_val = sum_x / d_f;

            var var_sum: f32 = 0.0;
            for (row_in) |val| {
                const diff = val - mean_val;
                var_sum += diff * diff;
            }
            const inv_std = 1.0 / @sqrt(var_sum / d_f + eps);

            for (0..D) |j| {
                row_out[j] = (row_in[j] - mean_val) * inv_std * G.data[j] + B.data[j];
            }
        }
        return Y;
    }

    pub fn rope(self: *Tensor, start_pos: usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        return self.ropeOffset(start_pos, 0, allocator, graph);
    }

    pub fn ropeOffset(self: *Tensor, start_pos: usize, rotary_offset: usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.ropeOffset(self, start_pos, rotary_offset);
        }
        const D = self.shape.dims[self.shape.len - 1];
        if (rotary_offset > D) return error.DimensionOutOfBounds;
        const T = if (self.shape.len >= 2) self.shape.dims[self.shape.len - 2] else 1;
        const outer = self.data.len / (T * D);
        const rot_dim = D - rotary_offset;
        const half = rot_dim / 2;
        const rot_dim_f = @as(f32, @floatFromInt(rot_dim));

        const Y = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (0..outer) |o| {
            for (0..T) |t| {
                const row_in = self.data[(o * T + t) * D .. (o * T + t + 1) * D];
                const row_out = Y.data[(o * T + t) * D .. (o * T + t + 1) * D];
                if (rotary_offset > 0) {
                    @memcpy(row_out[0..rotary_offset], row_in[0..rotary_offset]);
                }
                const pos_f = @as(f32, @floatFromInt(start_pos + t));
                for (0..half) |i| {
                    const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / rot_dim_f);
                    const theta = pos_f * freq;
                    const cos_t = @cos(theta);
                    const sin_t = @sin(theta);
                    const x0 = row_in[rotary_offset + 2 * i];
                    const x1 = row_in[rotary_offset + 2 * i + 1];
                    row_out[rotary_offset + 2 * i] = x0 * cos_t - x1 * sin_t;
                    row_out[rotary_offset + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
                }
                if (2 * half < rot_dim) {
                    row_out[D - 1] = row_in[D - 1];
                }
            }
        }
        return Y;
    }

    pub fn repeatKV(self: *Tensor, groups: usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.repeatKV(self, groups);
        }
        if (self.shape.len != 4) return error.IncompatibleDimensions;
        const B = self.shape.dims[0];
        const n_kv = self.shape.dims[1];
        const T = self.shape.dims[2];
        const hs = self.shape.dims[3];
        const nh = n_kv * groups;

        const Y = try zeros(allocator, &.{ B, nh, T, hs });
        const head_bytes = T * hs;
        for (0..B) |b| {
            for (0..n_kv) |kv_h| {
                const src = self.data[((b * n_kv + kv_h) * head_bytes) .. ((b * n_kv + kv_h + 1) * head_bytes)];
                for (0..groups) |g| {
                    const h = kv_h * groups + g;
                    const dest = Y.data[((b * nh + h) * head_bytes) .. ((b * nh + h + 1) * head_bytes)];
                    @memcpy(dest, src);
                }
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

    inline fn indexScalarToUsize(val: anytype, vocab_size: usize) !usize {
        const V = @TypeOf(val);
        if (@typeInfo(V) == .float) {
            if (val < 0.0) return error.IndexOutOfBounds;
            const idx = @as(usize, @intFromFloat(val));
            if (idx >= vocab_size) return error.IndexOutOfBounds;
            return idx;
        } else if (@typeInfo(V) == .int or @typeInfo(V) == .comptime_int) {
            if (val < 0) return error.IndexOutOfBounds;
            const idx = @as(usize, @intCast(val));
            if (idx >= vocab_size) return error.IndexOutOfBounds;
            return idx;
        } else {
            @compileError("Unsupported embedding index scalar type: " ++ @typeName(V));
        }
    }

    pub fn embedding(self: *Tensor, indices: anytype, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.embedding(self, indices);
        }
        if (self.shape.len != 2) return error.IncompatibleDimensions;
        const VocabSize = self.shape.dims[0];
        const D = self.shape.dims[1];

        const IdxT = @TypeOf(indices);
        const idx_info = @typeInfo(IdxT);
        const is_tensor_like = idx_info == .pointer and idx_info.pointer.size == .one and
            @typeInfo(idx_info.pointer.child) == .@"struct" and
            @hasField(idx_info.pointer.child, "shape");

        if (is_tensor_like) {
            const in_rank = indices.shape.len;
            if (in_rank == 0 or in_rank >= 8) return error.MaxDimensionsExceeded;
            var out_dims = [_]usize{0} ** 8;
            for (0..in_rank) |d| out_dims[d] = indices.shape.dims[d];
            out_dims[in_rank] = D;

            const Y = try zeros(allocator, out_dims[0 .. in_rank + 1]);
            errdefer free(allocator, Y);

            const num_indices = indices.shape.numel();
            if (indices.isContiguous() and indices.data.len >= num_indices) {
                for (0..num_indices) |i| {
                    const idx = try indexScalarToUsize(indices.data[i], VocabSize);
                    const w_row = self.data[idx * D .. (idx + 1) * D];
                    const y_row = Y.data[i * D .. (i + 1) * D];
                    @memcpy(y_row, w_row);
                }
            } else {
                var coord = [_]usize{0} ** 8;
                for (0..num_indices) |i| {
                    var flat_idx: usize = 0;
                    for (0..in_rank) |d| flat_idx += coord[d] * indices.strides.dims[d];
                    const idx = try indexScalarToUsize(indices.data[flat_idx], VocabSize);
                    const w_row = self.data[idx * D .. (idx + 1) * D];
                    const y_row = Y.data[i * D .. (i + 1) * D];
                    @memcpy(y_row, w_row);

                    var d = in_rank;
                    while (d > 0) {
                        d -= 1;
                        coord[d] += 1;
                        if (coord[d] < indices.shape.dims[d]) break;
                        coord[d] = 0;
                    }
                }
            }
            return Y;
        } else {
            const T_len = indices.len;
            const Y = try zeros(allocator, &.{ T_len, D });
            errdefer free(allocator, Y);
            for (0..T_len) |i| {
                const idx = try indexScalarToUsize(indices[i], VocabSize);
                const w_row = self.data[idx * D .. (idx + 1) * D];
                const y_row = Y.data[i * D .. (i + 1) * D];
                @memcpy(y_row, w_row);
            }
            return Y;
        }
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
        return isContiguousStrides(self.shape, self.strides);
    }

    pub fn numel(self: Tensor) usize {
        return self.shape.numel();
    }

    /// 通用多维张量沿指定轴或全局求和归约 (Sum Reduction)
    pub fn sum(self: *Tensor, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.sum(self, axis, keepdims);
        }
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
            const elem_count = self.shape.numel();
            if (self.isContiguous() and self.data.len >= elem_count) {
                for (self.data[0..elem_count]) |v| {
                    total += v;
                }
            } else {
                var coord = [_]usize{0} ** 8;
                const len = self.shape.len;
                for (0..elem_count) |_| {
                    var src_idx: usize = 0;
                    for (0..len) |d| {
                        src_idx += coord[d] * self.strides.dims[d];
                    }
                    total += self.data[src_idx];
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
    pub fn mean(self: *Tensor, axis: ?usize, keepdims: bool, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.mean(self, axis, keepdims);
        }
        const C = try self.sum(axis, keepdims, allocator, null);
        const count = if (axis) |ax| @as(f32, @floatFromInt(self.shape.dims[ax])) else @as(f32, @floatFromInt(self.shape.numel()));
        for (C.data) |*val| {
            val.* /= count;
        }
        return C;
    }

    /// 通用多维张量沿指定轴或全局方差 (Variance Reduction)
    pub fn variance(self: *Tensor, axis: ?usize, keepdims: bool, ddof: usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.variance(self, axis, keepdims, ddof);
        }
        if (axis) |ax| {
            if (ax >= self.shape.len) return error.DimensionOutOfBounds;
        }
        const count = if (axis) |ax| self.shape.dims[ax] else self.shape.numel();
        if (count <= ddof) return error.InvalidDDOF;

        const mean_t = try self.mean(axis, true, allocator, null);
        defer free(allocator, mean_t);

        const diff = try self.sub(mean_t, allocator, null);
        defer free(allocator, diff);
        const sq = try diff.mul(diff, allocator, null);
        defer free(allocator, sq);

        const sum_sq = try sq.sum(axis, keepdims, allocator, null);
        const denom = @as(f32, @floatFromInt(count - ddof));
        for (sum_sq.data) |*val| {
            val.* /= denom;
        }
        return sum_sq;
    }

    /// 通用多维张量沿指定轴或全局标准差 (Standard Deviation Reduction)
    pub fn stdDev(self: *Tensor, axis: ?usize, keepdims: bool, ddof: usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.stdDev(self, axis, keepdims, ddof);
        }
        const var_t = try self.variance(axis, keepdims, ddof, allocator, null);
        for (var_t.data) |*val| {
            val.* = @sqrt(@max(val.*, 0.0));
        }
        return var_t;
    }

    /// 依据布尔/条件张量 (支持 `*Tensor` 或 `*BoolTensor` / `*GenericTensor(T)`) 在两个候选张量间进行逐元素选择 (NumPy np.where)
    pub fn where(cond: anytype, x: *Tensor, y: *Tensor, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.where(cond, x, y);
        }
        const s_xy = try broadcastShapes(x.shape, y.shape);
        const target_shape = try broadcastShapes(cond.shape, s_xy);
        const C = try zeros(allocator, target_shape.dims[0..target_shape.len]);

        if (cond.shape.eq(x.shape) and x.shape.eq(y.shape) and cond.isContiguous() and x.isContiguous() and y.isContiguous() and
            cond.data.len >= C.data.len and x.data.len >= C.data.len and y.data.len >= C.data.len)
        {
            for (C.data, cond.data[0..C.data.len], x.data[0..C.data.len], y.data[0..C.data.len]) |*out_v, c_v, x_v, y_v| {
                out_v.* = if (isTruthyScalar(c_v)) x_v else y_v;
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

            C.data[c_flat] = if (isTruthyScalar(cond.data[cond_flat])) x.data[x_flat] else y.data[y_flat];

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

    /// 根据 mask (支持 `*Tensor` 或 `*BoolTensor` / `*GenericTensor(T)`) 将满足真值条件的元素赋值为指定标量值（返回新分配副本）
    pub fn maskedFill(self: *Tensor, mask: anytype, value: f32, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.maskedFill(self, mask, value);
        }
        if (!self.shape.eq(mask.shape)) return error.ShapeMismatch;
        const C = try self.contiguous(allocator);
        if (mask.isContiguous() and mask.data.len >= C.data.len) {
            for (C.data, mask.data[0..C.data.len]) |*out_v, m_v| {
                if (isTruthyScalar(m_v)) {
                    out_v.* = value;
                }
            }
        } else {
            var coord = [_]usize{0} ** 8;
            const len = mask.shape.len;
            for (C.data) |*out_v| {
                var m_idx: usize = 0;
                for (0..len) |d| {
                    m_idx += coord[d] * mask.strides.dims[d];
                }
                if (isTruthyScalar(mask.data[m_idx])) {
                    out_v.* = value;
                }
                var d = len;
                while (d > 0) {
                    d -= 1;
                    coord[d] += 1;
                    if (coord[d] < mask.shape.dims[d]) break;
                    coord[d] = 0;
                }
            }
        }
        return C;
    }

    /// 原地条件掩码填充 (支持 `*Tensor` 或 `*BoolTensor` / `*GenericTensor(T)`)
    pub fn maskedFill_(self: *Tensor, mask: anytype, value: f32) !*Tensor {
        if (self.requires_grad or self.creator != null) return error.InPlaceOpOnGraphTensor;
        if (!self.shape.eq(mask.shape)) return error.ShapeMismatch;
        if (self.isContiguous() and mask.isContiguous() and self.data.len >= self.shape.numel() and mask.data.len >= self.shape.numel()) {
            const count = self.shape.numel();
            for (self.data[0..count], mask.data[0..count]) |*out_v, m_v| {
                if (isTruthyScalar(m_v)) {
                    out_v.* = value;
                }
            }
        } else {
            var coord = [_]usize{0} ** 8;
            const len = self.shape.len;
            const count = self.shape.numel();
            for (0..count) |_| {
                var s_idx: usize = 0;
                var m_idx: usize = 0;
                for (0..len) |d| {
                    s_idx += coord[d] * self.strides.dims[d];
                    m_idx += coord[d] * mask.strides.dims[d];
                }
                if (isTruthyScalar(mask.data[m_idx])) {
                    self.data[s_idx] = value;
                }
                var d = len;
                while (d > 0) {
                    d -= 1;
                    coord[d] += 1;
                    if (coord[d] < self.shape.dims[d]) break;
                    coord[d] = 0;
                }
            }
        }
        return self;
    }

    fn compareScalarOp(self: *const Tensor, val: f32, allocator: std.mem.Allocator, comptime cmp_fn: fn (f32, f32) bool) !*GenericTensor(bool) {
        const out = try GenericTensor(bool).init(allocator, self.shape.dims[0..self.shape.len], null);
        errdefer out.deinit(allocator);
        if (self.isContiguous() and self.data.len >= out.data.len) {
            for (self.data[0..out.data.len], out.data) |a, *b| {
                b.* = cmp_fn(a, val);
            }
        } else {
            var coord = [_]usize{0} ** 8;
            const len = self.shape.len;
            for (0..out.data.len) |dest_i| {
                var src_idx: usize = 0;
                for (0..len) |d| src_idx += coord[d] * self.strides.dims[d];
                out.data[dest_i] = cmp_fn(self.data[src_idx], val);
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

    pub fn gtScalar(self: *const Tensor, val: f32, allocator: std.mem.Allocator) !*GenericTensor(bool) {
        return self.compareScalarOp(val, allocator, struct {
            fn cmp(a: f32, b: f32) bool {
                return a > b;
            }
        }.cmp);
    }

    pub fn geScalar(self: *const Tensor, val: f32, allocator: std.mem.Allocator) !*GenericTensor(bool) {
        return self.compareScalarOp(val, allocator, struct {
            fn cmp(a: f32, b: f32) bool {
                return a >= b;
            }
        }.cmp);
    }

    pub fn ltScalar(self: *const Tensor, val: f32, allocator: std.mem.Allocator) !*GenericTensor(bool) {
        return self.compareScalarOp(val, allocator, struct {
            fn cmp(a: f32, b: f32) bool {
                return a < b;
            }
        }.cmp);
    }

    pub fn leScalar(self: *const Tensor, val: f32, allocator: std.mem.Allocator) !*GenericTensor(bool) {
        return self.compareScalarOp(val, allocator, struct {
            fn cmp(a: f32, b: f32) bool {
                return a <= b;
            }
        }.cmp);
    }

    pub fn eqScalar(self: *const Tensor, val: f32, allocator: std.mem.Allocator) !*GenericTensor(bool) {
        return self.compareScalarOp(val, allocator, struct {
            fn cmp(a: f32, b: f32) bool {
                return a == b;
            }
        }.cmp);
    }

    pub fn neScalar(self: *const Tensor, val: f32, allocator: std.mem.Allocator) !*GenericTensor(bool) {
        return self.compareScalarOp(val, allocator, struct {
            fn cmp(a: f32, b: f32) bool {
                return a != b;
            }
        }.cmp);
    }

    /// 压缩单维度 (Squeeze): 移除所有为 1 的维度，或移除指定为 1 的维度
    pub fn squeeze(self: *Tensor, axis: ?usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (axis) |ax| {
            if (ax >= self.shape.len) return error.DimensionOutOfBounds;
            if (self.shape.dims[ax] != 1) return error.CannotSqueezeDimension;
            if (self.shape.len == 1) {
                return self.reshape(&.{1}, allocator, graph);
            }
            var new_dims = [_]usize{0} ** 8;
            var dest_d: usize = 0;
            for (0..self.shape.len) |d| {
                if (d != ax) {
                    new_dims[dest_d] = self.shape.dims[d];
                    dest_d += 1;
                }
            }
            return self.reshape(new_dims[0..dest_d], allocator, graph);
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
            return self.reshape(new_dims[0..dest_d], allocator, graph);
        }
    }

    /// 扩充单维度 (Unsqueeze / expand_dims): 在指定位置插入一个大小为 1 的新维度
    pub fn unsqueeze(self: *Tensor, dim: usize, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
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
        return self.reshape(new_dims[0..(self.shape.len + 1)], allocator, graph);
    }

    /// 跨步零拷贝切片 (Strided View Slicing)
    /// 当 graph == null 时返回一个共享底层内存缓冲区的零拷贝视图张量 (is_view = true)；
    /// 当 graph != null 时在计算图中注册可微的 Slice 算子
    pub fn slice(self: *Tensor, ranges: []const SliceRange, allocator: std.mem.Allocator, graph: ?*autodiff.Graph) anyerror!*Tensor {
        if (graph) |g| {
            return try g.slice(self, ranges);
        }
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
        if (self.isContiguous() and self.data.len >= out.data.len) {
            for (self.data[0..out.data.len], out.data) |x, *y| {
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
        const elem_count = self.shape.numel();

        for (0..elem_count) |_| {
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
        for (0..elem_count) |_| {
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

        if (self.isContiguous() and self.data.len >= out.data.len) {
            for (self.data[0..out.data.len], 0..) |val, i| {
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
