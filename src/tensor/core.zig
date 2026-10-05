const std = @import("std");
const Op = @import("../autodiff/op.zig").Op;
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

const nn_kernels = @import("nn_kernels.zig");
const reductions = @import("reductions.zig");

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
    init_constant: ?f32 = null, // 结构性常量初始值 (如 LSTM 遗忘门偏置 1.0)，Graph 自动初始化时以该常量填充，不依赖命名
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
    // Eager tensor operations (pure numerical kernels, no graph recording)
    // ============================================================================
    pub fn matmul(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        if (self.shape.len != 2 or other.shape.len != 2) {
            return error.IncompatibleDimensions;
        }
        if (self.shape.dims[1] != other.shape.dims[0]) {
            return error.ShapeMismatch;
        }
        const M = self.shape.dims[0];
        const K = self.shape.dims[1];
        const N = other.shape.dims[1];
        const C = try zeros(allocator, &.{ M, N });
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
    pub fn addBias(self: *Tensor, bias: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        return self.add(bias, allocator);
    }

    pub fn mulScalar(self: *Tensor, val: f32, allocator: std.mem.Allocator) !*Tensor {
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, s_val| {
            c_val.* = s_val * val;
        }
        return C;
    }

    pub fn addScalar(self: *Tensor, val: f32, allocator: std.mem.Allocator) !*Tensor {
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, s_val| {
            c_val.* = s_val + val;
        }
        return C;
    }

    pub fn subScalar(self: *Tensor, val: f32, allocator: std.mem.Allocator) !*Tensor {
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, s_val| {
            c_val.* = s_val - val;
        }
        return C;
    }

    pub fn divScalar(self: *Tensor, val: f32, allocator: std.mem.Allocator) !*Tensor {
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
    pub fn add(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        const out_shape = try broadcastShapes(self.shape, other.shape);
        const C = try zeros(allocator, out_shape.dims[0..out_shape.len]);
        broadcastBinaryOpRaw(C.data, C.shape, self.data, self.shape, self.strides, other.data, other.shape, other.strides, addOp);
        return C;
    }

    /// 通用多维广播减法：C = self - other
    pub fn sub(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        const out_shape = try broadcastShapes(self.shape, other.shape);
        const C = try zeros(allocator, out_shape.dims[0..out_shape.len]);
        broadcastBinaryOpRaw(C.data, C.shape, self.data, self.shape, self.strides, other.data, other.shape, other.strides, subOp);
        return C;
    }

    /// 通用多维广播乘法 (Hadamard 积)：C = self * other
    pub fn mul(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        const out_shape = try broadcastShapes(self.shape, other.shape);
        const C = try zeros(allocator, out_shape.dims[0..out_shape.len]);
        broadcastBinaryOpRaw(C.data, C.shape, self.data, self.shape, self.strides, other.data, other.shape, other.strides, mulOp);
        return C;
    }

    /// 通用多维广播除法：C = self / other
    pub fn div(self: *Tensor, other: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        const out_shape = try broadcastShapes(self.shape, other.shape);
        const C = try zeros(allocator, out_shape.dims[0..out_shape.len]);
        broadcastBinaryOpRaw(C.data, C.shape, self.data, self.shape, self.strides, other.data, other.shape, other.strides, divOp);
        return C;
    }

    pub fn silu(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, a_val| {
            const sig = if (a_val >= 0.0) 1.0 / (1.0 + @exp(-a_val)) else @exp(a_val) / (1.0 + @exp(a_val));
            c_val.* = a_val * sig;
        }
        return C;
    }

    pub fn relu(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        const total = self.data.len;
        for (0..total) |i| {
            C.data[i] = if (self.data[i] > 0.0) self.data[i] else 0.0;
        }
        return C;
    }

    pub fn gelu(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
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

    pub fn sigmoid(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
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

    pub fn tanh(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        const C = try zeros(allocator, self.shape.dims[0..self.shape.len]);
        for (C.data, self.data) |*c_val, a_val| {
            c_val.* = std.math.tanh(a_val);
        }
        return C;
    }

    pub fn leakyRelu(self: *Tensor, alpha: f32, allocator: std.mem.Allocator) !*Tensor {
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
    pub fn sqrt(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        return self.unaryMathEager(allocator, struct {
            fn op(x: f32) f32 {
                return @sqrt(x);
            }
        }.op);
    }

    /// 逐元素自然指数 (np.exp)
    pub fn exp(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        return self.unaryMathEager(allocator, struct {
            fn op(x: f32) f32 {
                return @exp(x);
            }
        }.op);
    }

    /// 逐元素自然对数 (np.log)
    pub fn log(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        return self.unaryMathEager(allocator, struct {
            fn op(x: f32) f32 {
                return @log(x);
            }
        }.op);
    }

    /// 逐元素绝对值 (np.abs)
    pub fn abs(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        return self.unaryMathEager(allocator, struct {
            fn op(x: f32) f32 {
                return @abs(x);
            }
        }.op);
    }

    pub fn bceWithLogitsLoss(self: *Tensor, targets: *Tensor, allocator: std.mem.Allocator) !*Tensor {
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

    pub fn bceLoss(self: *Tensor, targets: *Tensor, eps: f32, allocator: std.mem.Allocator) !*Tensor {
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

    pub fn softmaxCrossEntropy(self: *Tensor, targets: []const u8, allocator: std.mem.Allocator) !*Tensor {
        if (self.shape.len != 2) return error.IncompatibleDimensions;
        const loss = try zeros(allocator, &.{ 1, 1 });
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

    pub fn sigmoidCrossEntropy(self: *Tensor, targets: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        return self.bceWithLogitsLoss(targets, allocator);
    }

    pub fn mseLoss(self: *Tensor, targets: *Tensor, allocator: std.mem.Allocator) !*Tensor {
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

    pub fn l2Loss(self: *Tensor, lambda: f32, allocator: std.mem.Allocator) !*Tensor {
        const loss = try zeros(allocator, &.{ 1, 1 });
        var sum_sq: f32 = 0.0;
        for (self.data) |v| {
            sum_sq += v * v;
        }
        loss.data[0] = 0.5 * lambda * sum_sq;
        return loss;
    }

    pub fn l1Loss(self: *Tensor, lambda: f32, allocator: std.mem.Allocator) !*Tensor {
        const loss = try zeros(allocator, &.{ 1, 1 });
        var sum_abs: f32 = 0.0;
        for (self.data) |v| {
            sum_abs += @abs(v);
        }
        loss.data[0] = lambda * sum_abs;
        return loss;
    }

    pub fn reshape(self: *Tensor, new_shape_slice: []const usize, allocator: std.mem.Allocator) !*Tensor {
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

    pub fn split(self: *Tensor, num_splits: usize, dim: usize, allocator: std.mem.Allocator) ![]*Tensor {
        return tensorSplit(allocator, self, num_splits, dim);
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

    pub fn transpose(self: *Tensor, dim0: usize, dim1: usize, allocator: std.mem.Allocator) !*Tensor {
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
            const C = try zeros(allocator, &.{total * repeats});
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

    // --- Neural network kernels delegated to nn_kernels.zig ---
    pub const conv1d = nn_kernels.conv1d;
    pub const conv2d = nn_kernels.conv2d;
    pub const convTranspose1d = nn_kernels.convTranspose1d;
    pub const convTranspose2d = nn_kernels.convTranspose2d;
    pub const maxpool1d = nn_kernels.maxpool1d;
    pub const maxpool2d = nn_kernels.maxpool2d;
    pub const avgpool1d = nn_kernels.avgpool1d;
    pub const avgpool2d = nn_kernels.avgpool2d;
    pub const softmax = nn_kernels.softmax;
    pub const rmsNorm = nn_kernels.rmsNorm;
    pub const layerNorm = nn_kernels.layerNorm;
    pub const rope = nn_kernels.rope;
    pub const ropeOffset = nn_kernels.ropeOffset;
    pub const repeatKV = nn_kernels.repeatKV;
    pub const batchMatMul = nn_kernels.batchMatMul;
    pub const embedding = nn_kernels.embedding;

    // --- Reductions, mutations, comparisons, slicing, and sorting delegated to reductions.zig ---
    pub const clone = reductions.clone;
    pub const mulScalar_ = reductions.mulScalar_;
    pub const addScalar_ = reductions.addScalar_;
    pub const add_ = reductions.add_;
    pub const argmax = reductions.argmax;
    pub const max = reductions.max;
    pub const isContiguous = reductions.isContiguous;
    pub const numel = reductions.numel;
    pub const sum = reductions.sum;
    pub const mean = reductions.mean;
    pub const variance = reductions.variance;
    pub const stdDev = reductions.stdDev;
    pub const where = reductions.where;
    pub const maskedFill = reductions.maskedFill;
    pub const maskedFill_ = reductions.maskedFill_;
    pub const compareScalarOp = reductions.compareScalarOp;
    pub const gtScalar = reductions.gtScalar;
    pub const geScalar = reductions.geScalar;
    pub const ltScalar = reductions.ltScalar;
    pub const leScalar = reductions.leScalar;
    pub const eqScalar = reductions.eqScalar;
    pub const neScalar = reductions.neScalar;
    pub const squeezedShape = reductions.squeezedShape;
    pub const unsqueezedShape = reductions.unsqueezedShape;
    pub const squeeze = reductions.squeeze;
    pub const unsqueeze = reductions.unsqueeze;
    pub const slice = reductions.slice;
    pub const contiguous = reductions.contiguous;
    pub const clip = reductions.clip;
    pub const clip_ = reductions.clip_;
    pub const sort = reductions.sort;
    pub const argsort = reductions.argsort;
    pub const nonzero = reductions.nonzero;

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
