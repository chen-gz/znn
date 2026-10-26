const std = @import("std");
const backend = @import("../backend.zig");
const c = backend.cblas;
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
const losses = @import("losses.zig");
pub const linalg = @import("linalg.zig");
pub const QRResult = linalg.QRResult;
pub const SymEigResult = linalg.SymEigResult;
pub const SVDResult = linalg.SVDResult;

// ============================================================================
// 3. 基础张量（Tensor）核心定义与元数据
// ============================================================================

/// 自动微分算子节点的不透明句柄 (Opaque Handle)：张量层只保存指向产生该张量的算子的指针，
/// 具体类型 `autodiff.Op` 由上层定义，并通过 `Op.handle` / `Op.creatorOf` 完成转换，
/// 从而保持 `tensor` 层不依赖 `autodiff` 层（单向依赖）。
pub const OpHandle = opaque {};

/// 张量（Tensor）结构体：承载机器学习网络中所有物理数据与流转拓扑信息
pub const Tensor = struct {
    data: []f32, // 前向传播的数据缓冲区（行优先存储的一维切片）
    grad: []f32, // 反向传播的梯度缓冲区（与 data 形状一致，不需梯度的节点可为空）
    shape: Shape, // 逻辑形状
    strides: Shape, // 各维度的跨度步长（用于非连续张量及快速视图映射）
    requires_grad: bool, // 是否需要求梯度（如模型参数为 true，输入数据为 false）
    creator: ?*OpHandle, // 产生此张量的算子节点句柄（前向图中的父节点，由 `autodiff.Op.creatorOf` 还原为 `*Op`）
    is_view: bool = false, // 是否为零拷贝视图切片（若为 true，deinit 时不释放 data/grad）
    is_custom_initialized: bool = false, // 是否已被层专属自定义初始化 (避免被 Graph 自动初始化重写)
    init_constant: ?f32 = null, // 结构性常量初始值 (如 LSTM 遗忘门偏置 1.0)，Graph 自动初始化时以该常量填充，不依赖命名
    is_buffer: bool = false, // 是否为静态缓冲区/非学习常量张量 (如因果掩码 causal_mask, 位置索引 pos_indices)
    scope: []const u8 = "", // 张量在计算图中创建时所处的模块作用域完整路径 ("" 表示根作用域或图外创建)
    device: backend.Device = .cpu, // 存储所在设备：.cpu (默认) 或 .gpu / .cuda
    device_array: ?backend.DeviceArray = null, // GPU 驻留统一显存数组句柄 (支持 Apple MLX 与 NVIDIA CUDA)
    is_graph_owned: bool = false, // 是否由计算图拥有生命周期 (若为 true，计算图 reset/deinit 时自动释放 device_array)
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
    // Apple MLX 硬件加速桥接 (MLX Bridge)
    // ============================================================================

    /// 判断张量物理数据是否驻留在图形处理器 (Graphics Processing Unit, GPU)
    pub fn isDevice(self: *const Tensor) bool {
        return self.device.isGpu() and self.device_array != null;
    }

    /// 触发 GPU 驻留数组的惰性求值
    pub fn eval(self: *const Tensor) void {
        if (self.device_array) |da| {
            da.eval();
        }
    }

    /// 将 GPU 驻留张量的数据物化同步回主机中央处理器 (Central Processing Unit, CPU) 缓冲区 (data)
    pub fn syncHost(self: *Tensor, allocator: ?std.mem.Allocator) !void {
        if (self.device_array) |da| {
            const n_elements = self.shape.numel();
            if (self.data.len < n_elements) {
                const alloc = allocator orelse std.heap.smp_allocator;
                if (!self.is_view and self.data.len > 0) {
                    alloc.free(self.data);
                }
                self.data = try alloc.alloc(f32, n_elements);
            }
            try da.copyToSlice(self.data[0..n_elements]);
            self.device = .cpu;
        }
    }

    /// 迁移张量至指定设备 (.cpu, .cuda 或 .metal)
    pub fn toDevice(self: *Tensor, target: backend.Device) !void {
        if (self.device.eql(target) and self.device_array != null) return;
        switch (target.type) {
            .cuda, .metal => {
                if (!backend.device.isAvailable(target)) return error.DeviceNotSupported;
                const active = backend.device.activeBackend(target);
                if (active == .cuda) {
                    if (self.device_array == null or !self.device_array.?.isCuda()) {
                        if (self.device_array) |*old| old.deinit();
                        self.device_array = .{ .cuda = try backend.cuda.Array.fromSlice(self.data, self.shape.dims[0..self.shape.len]) };
                    }
                } else if (active == .mlx) {
                    if (self.device_array == null or !self.device_array.?.isMlx()) {
                        if (self.device_array) |*old| old.deinit();
                        self.device_array = .{ .mlx = try backend.mlx.Array.fromSliceZeroCopy(self.data, self.shape.dims[0..self.shape.len]) };
                    }
                }
                self.device = target;
            },
            .cpu => {
                // 若有设备数组且主机缓冲区未物化，需要外部调用方通过 syncHost 提供分配器物化
                self.device = .cpu;
            },
        }
    }

    /// 将张量转换为统一计算设备架构 (Compute Unified Device Architecture, CUDA) 数组
    pub fn toCudaArray(self: *const Tensor) !backend.cuda.Array {
        if (self.device_array) |da| {
            if (da.asCuda()) |arr| return arr;
        }
        return backend.cuda.Array.fromSlice(self.data, self.shape.dims[0..self.shape.len]);
    }

    /// 将只读张量转换为 MLX 数组 (Apple Machine Learning Explore, MLX)
    pub fn toMlxArray(self: *const Tensor) !backend.mlx.Array {
        if (self.device_array) |da| {
            if (da.asMlx()) |arr| return arr;
        }
        return backend.mlx.Array.fromSlice(self.data, self.shape.dims[0..self.shape.len]);
    }

    /// 将张量以零拷贝 (Zero-Copy) 方式映射为 MLX 数组 (统一内存架构 Unified Memory Architecture, UMA)
    pub fn toMlxArrayZeroCopy(self: *Tensor) !backend.mlx.Array {
        if (self.device_array) |da| {
            if (da.asMlx()) |arr| return arr;
        }
        return backend.mlx.Array.fromSliceZeroCopy(self.data, self.shape.dims[0..self.shape.len]);
    }

    /// 将 MLX 数组计算完成后的数据拷贝回张量
    pub fn copyFromMlxArray(self: *Tensor, arr: backend.mlx.Array) !void {
        return arr.copyToSlice(self.data);
    }

    /// 从已求值的 MLX 数组新建分配一个张量
    pub fn fromMlxArray(allocator: std.mem.Allocator, arr: backend.mlx.Array) !*Tensor {
        if (!backend.mlx.has_mlx) return error.MlxNotSupported;
        var shape_buf: [8]usize = undefined;
        const n = try arr.shape(&shape_buf);
        const t = try zeros(allocator, shape_buf[0..n]);
        errdefer free(allocator, t);
        try arr.copyToSlice(t.data);
        return t;
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

        // 行优先或转置（列优先）布局的视图直接以对应的转置标志交给 BLAS，其余布局才物化为连续副本
        const a_layout = nn_kernels.blasMatrix(M, K, self.strides.dims[0], self.strides.dims[1]);
        const a = if (a_layout != null) self else try self.contiguous(allocator);
        defer if (a != self) a.deinit(allocator);
        const b_layout = nn_kernels.blasMatrix(K, N, other.strides.dims[0], other.strides.dims[1]);
        const b = if (b_layout != null) other else try other.contiguous(allocator);
        defer if (b != other) b.deinit(allocator);
        const a_mat = a_layout orelse nn_kernels.BlasMatrix{ .trans = c.CblasNoTrans, .ld = @max(K, 1) };
        const b_mat = b_layout orelse nn_kernels.BlasMatrix{ .trans = c.CblasNoTrans, .ld = @max(N, 1) };

        const C = try zeros(allocator, &.{ M, N });
        if (M == 0 or N == 0) return C;
        c.cblas_sgemm(
            c.CblasRowMajor,
            a_mat.trans,
            b_mat.trans,
            @intCast(M),
            @intCast(N),
            @intCast(K),
            1.0,
            a.data.ptr,
            @intCast(a_mat.ld),
            b.data.ptr,
            @intCast(b_mat.ld),
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

    /// 逐元素映射内核：输出为新分配的连续张量 `C[i] = f(self[i], arg)`；
    /// 连续输入单遍直接映射，非连续输入先按行收集到输出缓冲区再原地映射
    fn mapEager(self: *const Tensor, allocator: std.mem.Allocator, arg: anytype, comptime f: fn (f32, @TypeOf(arg)) f32) !*Tensor {
        const C = try ops_mod.allocUninit(allocator, self.shape);
        const n = C.data.len;
        if (self.isContiguous() and self.data.len >= n) {
            for (C.data, self.data[0..n]) |*c_val, s_val| c_val.* = f(s_val, arg);
        } else {
            shape_mod.gatherStrided(C.data, self.data, self.shape, self.strides);
            for (C.data) |*c_val| c_val.* = f(c_val.*, arg);
        }
        return C;
    }

    fn scalarOpEager(self: *const Tensor, val: f32, allocator: std.mem.Allocator, comptime op: fn (f32, f32) f32) !*Tensor {
        return self.mapEager(allocator, val, op);
    }

    pub fn mulScalar(self: *Tensor, val: f32, allocator: std.mem.Allocator) !*Tensor {
        return self.scalarOpEager(val, allocator, struct {
            fn op(a: f32, b: f32) f32 {
                return a * b;
            }
        }.op);
    }

    pub fn addScalar(self: *Tensor, val: f32, allocator: std.mem.Allocator) !*Tensor {
        return self.scalarOpEager(val, allocator, struct {
            fn op(a: f32, b: f32) f32 {
                return a + b;
            }
        }.op);
    }

    pub fn subScalar(self: *Tensor, val: f32, allocator: std.mem.Allocator) !*Tensor {
        return self.scalarOpEager(val, allocator, struct {
            fn op(a: f32, b: f32) f32 {
                return a - b;
            }
        }.op);
    }

    pub fn divScalar(self: *Tensor, val: f32, allocator: std.mem.Allocator) !*Tensor {
        return self.scalarOpEager(val, allocator, struct {
            fn op(a: f32, b: f32) f32 {
                return a / b;
            }
        }.op);
    }

    fn addOp(a: f32, b: f32) f32 {
        return a + b;
    }
    fn subOp(a: f32, b: f32) f32 {
        return a - b;
    }
    fn mulOp(a: f32, b: f32) f32 {
        return a * b;
    }
    fn divOp(a: f32, b: f32) f32 {
        return a / b;
    }

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
        return self.unaryMathEager(allocator, struct {
            fn op(a_val: f32) f32 {
                return a_val * backend.vec_math.sigmoid(a_val);
            }
        }.op);
    }

    pub fn relu(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        return self.unaryMathEager(allocator, struct {
            fn op(x: f32) f32 {
                return if (x > 0.0) x else 0.0;
            }
        }.op);
    }

    /// 高斯误差线性单元 (Gaussian Error Linear Unit, GELU)：按 `approximation` 选择精确 (erf) 形式或双曲正切 (Hyperbolic Tangent, Tanh) 近似形式
    pub fn gelu(self: *Tensor, approximation: backend.GeluApproximation, allocator: std.mem.Allocator) !*Tensor {
        return switch (approximation) {
            inline else => |approx| self.unaryMathEager(allocator, struct {
                fn op(x: f32) f32 {
                    return backend.activation.gelu(approx, x);
                }
            }.op),
        };
    }

    pub fn sigmoid(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        return self.unaryMathEager(allocator, struct {
            fn op(a_val: f32) f32 {
                return backend.vec_math.sigmoid(a_val);
            }
        }.op);
    }

    pub fn tanh(self: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        return self.unaryMathEager(allocator, struct {
            fn op(a_val: f32) f32 {
                return std.math.tanh(a_val);
            }
        }.op);
    }

    /// 软截断 (Soft Capping)：y = cap * tanh(x / cap) (与 `vec_math.softcap` 逐元素一致)，cap <= 0 时为恒等映射
    pub fn softcap(self: *Tensor, cap: f32, allocator: std.mem.Allocator) !*Tensor {
        return self.mapEager(allocator, cap, struct {
            fn op(a_val: f32, cap_val: f32) f32 {
                return backend.vec_math.softcap(a_val, cap_val);
            }
        }.op);
    }

    pub fn leakyRelu(self: *Tensor, alpha: f32, allocator: std.mem.Allocator) !*Tensor {
        return self.mapEager(allocator, alpha, struct {
            fn op(a_val: f32, slope: f32) f32 {
                return if (a_val > 0.0) a_val else slope * a_val;
            }
        }.op);
    }

    fn unaryMathEager(self: *const Tensor, allocator: std.mem.Allocator, comptime math_fn: fn (f32) f32) !*Tensor {
        return self.mapEager(allocator, {}, struct {
            fn op(a_val: f32, _: void) f32 {
                return math_fn(a_val);
            }
        }.op);
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
        const N = self.shape.numel();
        if (N != targets.shape.numel()) return error.ShapeMismatch;
        const x_c = if (self.isContiguous()) self else try self.contiguous(allocator);
        defer if (x_c != self) x_c.deinit(allocator);
        const y_c = if (targets.isContiguous()) targets else try targets.contiguous(allocator);
        defer if (y_c != targets) y_c.deinit(allocator);

        const loss = try zeros(allocator, &.{ 1, 1 });
        var total_loss: f32 = 0.0;
        for (0..N) |i| {
            const x = x_c.data[i];
            const y = y_c.data[i];
            const max_x = @max(x, 0.0);
            const abs_x = @abs(x);
            total_loss += max_x - x * y + @log(1.0 + @exp(-abs_x));
        }
        loss.data[0] = total_loss / @as(f32, @floatFromInt(N));
        return loss;
    }

    pub fn bceLoss(self: *Tensor, targets: *Tensor, eps: f32, allocator: std.mem.Allocator) !*Tensor {
        const N = self.shape.numel();
        if (N != targets.shape.numel()) return error.ShapeMismatch;
        const x_c = if (self.isContiguous()) self else try self.contiguous(allocator);
        defer if (x_c != self) x_c.deinit(allocator);
        const y_c = if (targets.isContiguous()) targets else try targets.contiguous(allocator);
        defer if (y_c != targets) y_c.deinit(allocator);

        const loss = try zeros(allocator, &.{ 1, 1 });
        var total_loss: f32 = 0.0;
        for (0..N) |i| {
            const p = x_c.data[i];
            const y = y_c.data[i];
            const p_clip = @max(p, eps);
            const one_minus_p_clip = @max(1.0 - p, eps);
            total_loss += -(y * @log(p_clip) + (1.0 - y) * @log(one_minus_p_clip));
        }
        loss.data[0] = total_loss / @as(f32, @floatFromInt(N));
        return loss;
    }

    /// 以正态分布 N(mean_val, stddev^2) 随机数原地填充：使用博克斯-穆勒变换 (Box-Muller Transform) 成对生成，
    /// 按行优先顺序只写入张量的可见元素（跨步视图不会改写父缓冲区中不属于视图的元素）
    pub fn fillNormal(self: *Tensor, random: std.Random, mean_val: f32, stddev: f32) void {
        const NormalFill = struct {
            random: std.Random,
            mean: f32,
            stddev: f32,
            spare: ?f32 = null,

            fn next(item: *f32, state: *@This()) void {
                if (state.spare) |z1| {
                    item.* = state.mean + z1 * state.stddev;
                    state.spare = null;
                    return;
                }
                var u_1: f32 = state.random.float(f32);
                while (u_1 == 0.0) {
                    u_1 = state.random.float(f32);
                }
                const u_2 = state.random.float(f32);
                const radius = @sqrt(-2.0 * @log(u_1));
                const theta = 2.0 * std.math.pi * u_2;
                state.spare = radius * @sin(theta);
                item.* = state.mean + radius * @cos(theta) * state.stddev;
            }
        };
        var state = NormalFill{ .random = random, .mean = mean_val, .stddev = stddev };
        reductions.mapInPlace(self, &state, NormalFill.next);
    }

    /// 以均匀分布 U[min_val, max_val) 随机数原地填充，按行优先顺序只写入张量的可见元素
    pub fn fillUniform(self: *Tensor, random: std.Random, min_val: f32, max_val: f32) void {
        const UniformFill = struct {
            random: std.Random,
            min: f32,
            range: f32,

            fn next(item: *f32, state: *const @This()) void {
                item.* = state.min + state.random.float(f32) * state.range;
            }
        };
        const state = UniformFill{ .random = random, .min = min_val, .range = max_val - min_val };
        reductions.mapInPlace(self, &state, UniformFill.next);
    }

    fn getTargetsLen(targets: anytype) usize {
        const T = @TypeOf(targets);
        switch (@typeInfo(T)) {
            .pointer => |ptr| switch (ptr.size) {
                .slice => return targets.len,
                .one => switch (@typeInfo(ptr.child)) {
                    .array => |a| return a.len,
                    .@"struct" => |s| return s.field_names.len,
                    else => return targets.len,
                },
                else => return targets.len,
            },
            .@"struct" => |s| return s.field_names.len,
            .array => |a| return a.len,
            else => return targets.len,
        }
    }

    fn getTarget(targets: anytype, i: usize) usize {
        const T = @TypeOf(targets);
        switch (@typeInfo(T)) {
            .pointer => |ptr| switch (ptr.size) {
                .slice => return @intCast(targets[i]),
                .one => switch (@typeInfo(ptr.child)) {
                    .array => return @intCast(targets[i]),
                    .@"struct" => |s| {
                        inline for (s.field_names, 0..) |name, idx| {
                            if (idx == i) return @intCast(@field(targets, name));
                        }
                        unreachable;
                    },
                    else => return @intCast(targets[i]),
                },
                else => return @intCast(targets[i]),
            },
            .@"struct" => |s| {
                inline for (s.field_names, 0..) |name, idx| {
                    if (idx == i) return @intCast(@field(targets, name));
                }
                unreachable;
            },
            .array => return @intCast(targets[i]),
            else => return @intCast(targets[i]),
        }
    }

    /// Softmax 交叉熵损失：按对数和指数 (Log-Sum-Exp, LSE) 形式计算
    /// `loss = log(sum(exp(x - max))) + max - x[label]`，即使正确类别概率下溢为 0 也能得到精确损失。
    /// targets 支持 u8 / u32 / usize 等任意整型切片与元组。
    pub fn softmaxCrossEntropy(self: *Tensor, targets: anytype, allocator: std.mem.Allocator) !*Tensor {
        if (self.shape.len != 2) return error.IncompatibleDimensions;
        const B = self.shape.dims[0];
        const N = self.shape.dims[1];
        const num_targets = getTargetsLen(targets);
        if (B != num_targets) return error.ShapeMismatch;
        if (B == 0 or N == 0) return error.EmptyInputs;

        const x_c = if (self.isContiguous()) self else try self.contiguous(allocator);
        defer if (x_c != self) x_c.deinit(allocator);

        var loss_sum: f32 = 0.0;
        for (0..B) |i| {
            const logits_row = x_c.data[i * N .. (i + 1) * N];
            const target = getTarget(targets, i);
            loss_sum += try losses.ceRow(logits_row, target, null, null, 0.0);
        }
        const loss = try zeros(allocator, &.{ 1, 1 });
        loss.data[0] = loss_sum / @as(f32, @floatFromInt(B));
        return loss;
    }

    pub fn mseLoss(self: *Tensor, targets: *Tensor, allocator: std.mem.Allocator) !*Tensor {
        const N = self.shape.numel();
        if (N != targets.shape.numel()) return error.ShapeMismatch;
        const x_c = if (self.isContiguous()) self else try self.contiguous(allocator);
        defer if (x_c != self) x_c.deinit(allocator);
        const y_c = if (targets.isContiguous()) targets else try targets.contiguous(allocator);
        defer if (y_c != targets) y_c.deinit(allocator);

        const loss = try zeros(allocator, &.{ 1, 1 });
        var loss_sum: f32 = 0.0;
        for (0..N) |i| {
            const diff = x_c.data[i] - y_c.data[i];
            loss_sum += diff * diff;
        }
        loss.data[0] = loss_sum / @as(f32, @floatFromInt(N));
        return loss;
    }

    pub fn l2Loss(self: *Tensor, lambda: f32, allocator: std.mem.Allocator) !*Tensor {
        const x_c = if (self.isContiguous()) self else try self.contiguous(allocator);
        defer if (x_c != self) x_c.deinit(allocator);

        var sum_sq: f32 = 0.0;
        for (x_c.data[0..self.shape.numel()]) |v| {
            sum_sq += v * v;
        }
        const loss = try zeros(allocator, &.{ 1, 1 });
        loss.data[0] = 0.5 * lambda * sum_sq;
        return loss;
    }

    pub fn l1Loss(self: *Tensor, lambda: f32, allocator: std.mem.Allocator) !*Tensor {
        const x_c = if (self.isContiguous()) self else try self.contiguous(allocator);
        defer if (x_c != self) x_c.deinit(allocator);

        var sum_abs: f32 = 0.0;
        for (x_c.data[0..self.shape.numel()]) |v| {
            sum_abs += @abs(v);
        }
        const loss = try zeros(allocator, &.{ 1, 1 });
        loss.data[0] = lambda * sum_abs;
        return loss;
    }

    pub fn reshape(self: *Tensor, new_shape_slice: []const usize, allocator: std.mem.Allocator) !*Tensor {
        const shape = try Shape.fromSlice(new_shape_slice);
        const strides = computeContiguousStrides(shape);
        const old_total = self.shape.numel();
        const new_total = shape.numel();
        if (old_total != new_total) return error.ShapeMismatch;

        // 先按原形状物化为连续副本，再原地改写形状元数据
        const C = try self.contiguous(allocator);
        C.shape = shape;
        C.strides = strides;
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

        const C = try zeros(allocator, shape_trans.dims[0..shape_trans.len]);
        shape_mod.gatherStrided(C.data, self.data, shape_trans, strides_trans);
        return C;
    }

    /// 沿指定轴重复张量元素 repeats 次 (np.repeat)
    /// axis 若为 null，则先将张量展平后重复
    pub fn repeat(self: *Tensor, repeats: usize, axis: ?usize, allocator: std.mem.Allocator) !*Tensor {
        if (axis) |ax| {
            if (ax >= self.shape.len) return error.DimensionOutOfBounds;
            if (repeats == 1) return self.contiguous(allocator);

            var out_shape = self.shape;
            out_shape.dims[ax] = self.shape.dims[ax] * repeats;

            const dim_size = self.shape.dims[ax];
            var outer_size: usize = 1;
            for (0..ax) |d| outer_size *= self.shape.dims[d];
            var inner_size: usize = 1;
            for (ax + 1..self.shape.len) |d| inner_size *= self.shape.dims[d];

            const src_contig = try self.contiguous(allocator);
            defer free(allocator, src_contig);
            const C = try zeros(allocator, out_shape.dims[0..out_shape.len]);

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
            // 先展平：按逻辑元素数 (numel) 而非底层缓冲区长度计数，切片视图的 data 可能长于其可见元素
            const total = self.shape.numel();
            const src_contig = try self.contiguous(allocator);
            defer free(allocator, src_contig);
            const C = try zeros(allocator, &.{total * repeats});

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
        if (reps.len == 0) return self.contiguous(allocator);
        const rank = @max(self.shape.len, reps.len);
        if (rank > 8) return error.MaxDimensionsExceeded;

        var full_self_shape: [8]usize = @splat(1);
        var full_reps: [8]usize = @splat(1);
        var out_shape_dims: [8]usize = @splat(1);

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

        const src_contig = try self.contiguous(allocator);
        defer free(allocator, src_contig);
        const out = try zeros(allocator, out_shape_dims[0..rank]);

        const src_full_shape = try Shape.fromSlice(full_self_shape[0..rank]);
        const src_strides = computeContiguousStrides(src_full_shape);

        var coord: [8]usize = @splat(0);
        for (0..out.data.len) |dest_i| {
            var src_flat: usize = 0;
            for (0..rank) |d| {
                const src_dim_idx = coord[d] % full_self_shape[d];
                src_flat += src_dim_idx * src_strides.dims[d];
            }
            out.data[dest_i] = src_contig.data[src_flat];

            out.shape.incrementCoord(&coord);
        }
        return out;
    }

    // --- Neural network kernels delegated to nn_kernels.zig / conv_pool.zig ---
    pub const conv1d = nn_kernels.conv1d;
    pub const conv2d = nn_kernels.conv2d;
    pub const convTranspose1d = nn_kernels.convTranspose1d;
    pub const convTranspose2d = nn_kernels.convTranspose2d;
    pub const maxpool1d = nn_kernels.maxpool1d;
    pub const maxpool2d = nn_kernels.maxpool2d;
    pub const avgpool1d = nn_kernels.avgpool1d;
    pub const avgpool2d = nn_kernels.avgpool2d;
    pub const adaptiveAvgPool1d = nn_kernels.adaptiveAvgPool1d;
    pub const adaptiveAvgPool2d = nn_kernels.adaptiveAvgPool2d;
    pub const softmax = nn_kernels.softmax;
    pub const rmsNorm = nn_kernels.rmsNorm;
    pub const layerNorm = nn_kernels.layerNorm;
    pub const batchNorm1d = nn_kernels.batchNorm1d;
    pub const batchNorm2d = nn_kernels.batchNorm2d;
    pub const groupNorm = nn_kernels.groupNorm;
    pub const applyDropoutMask = nn_kernels.applyDropoutMask;
    pub const rope = nn_kernels.rope;
    pub const repeatKV = nn_kernels.repeatKV;
    pub const batchMatMul = nn_kernels.batchMatMul;
    pub const embedding = nn_kernels.embedding;
    pub const TopKResult = nn_kernels.TopKResult;
    pub const topKInto = nn_kernels.topKInto;
    pub const topK = nn_kernels.topK;
    pub const gatherMatMulInto = nn_kernels.gatherMatMulInto;
    pub const gatherMatMul = nn_kernels.gatherMatMul;

    // --- Reductions, mutations, comparisons, slicing, and sorting delegated to reductions.zig ---
    pub const clone = reductions.clone;
    pub const mulScalar_ = reductions.mulScalar_;
    pub const addScalar_ = reductions.addScalar_;
    pub const add_ = reductions.add_;
    pub const argmax = reductions.argmax;
    pub const argmin = reductions.argmin;
    pub const max = reductions.max;
    pub const min = reductions.min;
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
            var coord: [8]usize = @splat(0);
            const len = self.shape.len;
            for (0..out.data.len) |dest_i| {
                var src_idx: usize = 0;
                for (0..len) |d| {
                    src_idx += coord[d] * self.strides.dims[d];
                }
                out.data[dest_i] = convertScalar(DestT, f32, self.data[src_idx]);
                self.shape.incrementCoord(&coord);
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

    /// 正交三角分解 (QR Decomposition, QR)
    pub fn qr(self: Tensor, allocator: std.mem.Allocator) !linalg.QRResult {
        return linalg.qr(self, allocator);
    }

    /// 实对称特征值分解 (Symmetric Eigenvalue Decomposition, SymEig)
    pub fn symeig(self: Tensor, allocator: std.mem.Allocator) !linalg.SymEigResult {
        return linalg.symeig(self, allocator);
    }

    /// 奇异值分解 (Singular Value Decomposition, SVD)
    pub fn svd(self: Tensor, allocator: std.mem.Allocator) !linalg.SVDResult {
        return linalg.svd(self, allocator);
    }

    pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
        if (self.device_array) |*arr| {
            arr.deinit();
            self.device_array = null;
        }
        if (!self.is_view) {
            if (self.data.len > 0) {
                allocator.free(self.data);
            }
            if (self.requires_grad and self.grad.len > 0) {
                allocator.free(self.grad);
            }
        }
        allocator.destroy(self);
    }
};
