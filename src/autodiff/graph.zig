const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const Shape = tensor_mod.Shape;
const computeContiguousStrides = tensor_mod.computeContiguousStrides;
const types = @import("types.zig");
pub const OpType = types.OpType;
pub const OpContext = types.OpContext;
const op_mod = @import("op.zig");
pub const Op = types.Op;
const graph_nn = @import("graph_nn.zig");
const graph_init = @import("graph_init.zig");
const backend = @import("../backend.zig");
pub const Executor = backend.Executor;

/// 计算图执行模式
pub const ExecutionMode = enum {
    /// 即时模式（默认）：算子在记录时立即派发；CPU 上同步完成
    eager,
    /// 延迟模式：只记录算子，统一规划后派发重放
    deferred,
};

pub const Device = backend.Device;

/// 计算图配置（遵循 AGENTS.md §4：所有字段均有默认值）
pub const GraphOptions = struct {
    io: ?std.Io = null, // 由应用注入的输入输出接口；null 表示串行执行
    enable_grad: bool = true,
    mode: ExecutionMode = .eager,
    max_tasks: ?usize = null,
    device: Device = .cpu, // 目标计算设备：.cpu (默认) 或 .gpu (Apple Silicon MLX 硬件加速)
    mlx_stream: ?backend.mlx.Stream = null, // 可选注入的 MLX 执行流 (Stream)；为 null 时使用默认流

    pub const default: GraphOptions = .{};
    pub fn defaultOptions() GraphOptions {
        return .{};
    }
};

// 计算图（Graph）结构体
// 追踪所有的张量节点与算子节点，管理内存生命周期并负责反向传播调度
pub const Graph = struct {
    backing_allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator, // 使用 Arena 机制，使每次前向/反向生成的中间节点内存可在 batch 结束时一并释放，避免内存碎片和频繁分配
    tensors: std.ArrayList(*Tensor), // 追踪计算图中的所有张量指针
    ops: std.ArrayList(*Op), // 追踪计算图中的所有算子指针
    enable_grad: bool, // 梯度使能开关（类似 torch.set_grad_enabled），为 false 时不分配梯度缓冲区亦不记录 Op 节点
    options: GraphOptions, // 计算图运行选项配置
    module_formulas: std.StringHashMap([]const u8), // 存储模块在代码中声明的显式数学运算公式 (如 "y = x W + b")
    module_types: std.StringHashMap([]const u8), // 存储模块在代码中声明的显式模块类型 (如 "Linear", "RMSNorm", "GPT")
    scope_stack: std.ArrayList([]const u8), // 当前正在执行 forward 的模块作用域栈 (栈顶为最内层模块的完整路径)

    /// 初始化计算图，传入底层通用内存分配器与运行配置选项
    pub fn init(backing_allocator: std.mem.Allocator, options: GraphOptions) Graph {
        return Graph{
            .backing_allocator = backing_allocator,
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .tensors = .empty,
            .ops = .empty,
            .enable_grad = options.enable_grad,
            .options = options,
            .module_formulas = std.StringHashMap([]const u8).init(backing_allocator),
            .module_types = std.StringHashMap([]const u8).init(backing_allocator),
            .scope_stack = .empty,
        };
    }

    /// 使用默认配置初始化计算图：等价于 `Graph.init(backing_allocator, .{})`
    pub fn initDefault(backing_allocator: std.mem.Allocator) Graph {
        return Graph.init(backing_allocator, .{});
    }

    /// 获取当前计算图关联的数据并行执行器
    pub fn executor(self: *const Graph) Executor {
        return .{
            .io = self.options.io,
            .max_tasks = self.options.max_tasks,
        };
    }

    /// 计算图 Arena 分配器：其上分配的张量在 `deinit` 时随计算图一并释放
    pub fn arenaAllocator(self: *Graph) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// 模块作用域守卫：由 `enterModule` / `enterChildScope` 返回，`exit()` 时弹出对应作用域
    pub const ScopeGuard = struct {
        graph: ?*Graph = null,

        pub fn exit(self: ScopeGuard) void {
            if (self.graph) |g| g.popScope();
        }
    };

    /// 压入一个模块作用域 (完整路径)，之后创建的算子与张量都归属该模块
    pub fn pushScope(self: *Graph, path: []const u8, module_type: ?[]const u8) !void {
        const path_copy = try self.arena.allocator().dupe(u8, path);
        try self.scope_stack.append(self.backing_allocator, path_copy);
        if (module_type) |t| try self.registerModuleType(path_copy, t);
    }

    /// 弹出最内层模块作用域
    pub fn popScope(self: *Graph) void {
        _ = self.scope_stack.pop();
    }

    /// 当前最内层模块作用域的完整路径 ("" 表示根作用域)
    pub fn currentScope(self: *const Graph) []const u8 {
        if (self.scope_stack.items.len == 0) return "";
        return self.scope_stack.items[self.scope_stack.items.len - 1];
    }

    /// 在模块 forward 入口处进入该模块的作用域。
    /// 模块未命名时返回空守卫。
    /// 用法: `const scope = try g.enterModule(self.name, self.module_type); defer scope.exit();`
    pub fn enterModule(self: *Graph, name: ?[]const u8, module_type: []const u8) !ScopeGuard {
        const n = name orelse return .{};
        try self.pushScope(n, module_type);
        return .{ .graph = self };
    }

    /// 在当前模块内部开启一个命名子作用域 (如注意力模块内部的 "core")。
    /// 当前处于根作用域 (所属模块未命名) 时返回空守卫。
    pub fn enterChildScope(self: *Graph, local_name: []const u8, module_type: []const u8) !ScopeGuard {
        const parent = self.currentScope();
        if (parent.len == 0) return .{};
        const path = try std.fmt.allocPrint(self.arena.allocator(), "{s}.{s}", .{ parent, local_name });
        try self.pushScope(path, module_type);
        return .{ .graph = self };
    }

    /// 记录一个新创建的算子，并标记其所属的当前模块作用域
    pub fn recordOp(self: *Graph, o: *Op) !void {
        o.scope = self.currentScope();
        try self.ops.append(self.backing_allocator, o);
    }

    /// 在代码中为指定模块路径设置数学公式 (如 graph.setModuleFormula("gpt.layers.0.attn", "A = softmax(QK^T / sqrt(d_k)) V"))
    pub fn setModuleFormula(self: *Graph, module_path: []const u8, formula: []const u8) !void {
        const mod_copy = try self.arena.allocator().dupe(u8, module_path);
        const form_copy = try self.arena.allocator().dupe(u8, formula);
        try self.module_formulas.put(mod_copy, form_copy);
    }

    /// 获取指定模块路径绑定的数学公式
    pub fn getModuleFormula(self: *const Graph, module_path: []const u8) ?[]const u8 {
        return self.module_formulas.get(module_path);
    }

    /// 在代码中为指定模块路径注册显式模块类型 (如 graph.registerModuleType("gpt.layers.0.ln_1", "RMSNorm"))
    pub fn registerModuleType(self: *Graph, module_path: []const u8, mod_type: []const u8) !void {
        const mod_copy = try self.arena.allocator().dupe(u8, module_path);
        const type_copy = try self.arena.allocator().dupe(u8, mod_type);
        try self.module_types.put(mod_copy, type_copy);
    }

    /// 获取指定模块路径绑定的模块类型
    pub fn getModuleType(self: *const Graph, module_path: []const u8) ?[]const u8 {
        return self.module_types.get(module_path);
    }

    pub const inferModuleFormula = graph_init.inferModuleFormula;

    // 设置梯度追踪开关
    pub fn setGradEnabled(self: *Graph, enabled: bool) void {
        self.enable_grad = enabled;
        self.options.enable_grad = enabled;
    }

    // 释放整个计算图的内存（包括所有张量与算子节点的前向/反向缓冲区）
    pub fn deinit(self: *Graph) void {
        for (self.tensors.items) |t| {
            if (t.is_graph_owned) {
                if (t.device_array) |*arr| {
                    arr.deinit();
                    t.device_array = null;
                }
            }
        }
        self.module_formulas.deinit();
        self.module_types.deinit();
        self.scope_stack.deinit(self.backing_allocator);
        self.tensors.deinit(self.backing_allocator);
        self.ops.deinit(self.backing_allocator);
        self.arena.deinit();
    }

    /// 重置计算图状态与 Arena 内存（保留已分配容量以供高效复用）
    pub fn reset(self: *Graph) void {
        for (self.tensors.items) |t| {
            if (t.is_graph_owned) {
                if (t.device_array) |*arr| {
                    arr.deinit();
                    t.device_array = null;
                }
            }
        }
        self.module_formulas.clearRetainingCapacity();
        self.module_types.clearRetainingCapacity();
        self.scope_stack.clearRetainingCapacity();
        self.tensors.clearRetainingCapacity();
        self.ops.clearRetainingCapacity();
        _ = self.arena.reset(.retain_capacity);
    }

    // 在计算图中注册外部持久化参数张量节点 (如 Linear / Conv2D 的权重与偏置)
    pub fn registerParameter(self: *Graph, t: *Tensor) !void {
        try self.tensors.append(self.backing_allocator, t);
    }

    // 在计算图中创建并注册一个新的张量节点
    pub fn tensor(self: *Graph, rows: usize, cols: usize, requires_grad: bool) !*Tensor {
        return self.tensorND(&.{ rows, cols }, requires_grad);
    }

    // 创建并注册一个带初始数据的二维张量节点
    pub fn tensorWithData(self: *Graph, rows: usize, cols: usize, initial_data: []const f32, requires_grad: bool) !*Tensor {
        return self.tensorNDWithData(&.{ rows, cols }, initial_data, requires_grad);
    }

    // 创建并注册一个带初始数据的多维张量节点
    pub fn tensorNDWithData(self: *Graph, shape_slice: []const usize, initial_data: []const f32, requires_grad: bool) !*Tensor {
        const t = try self.tensorND(shape_slice, requires_grad);
        std.debug.assert(t.data.len == initial_data.len);
        @memcpy(t.data, initial_data);
        return t;
    }

    // NumPy-like API: zeros
    pub fn zeros(self: *Graph, shape_slice: []const usize, requires_grad: bool) !*Tensor {
        return self.tensorND(shape_slice, requires_grad);
    }

    // NumPy-like API: ones
    pub fn ones(self: *Graph, shape_slice: []const usize, requires_grad: bool) !*Tensor {
        const t = try self.tensorND(shape_slice, requires_grad);
        @memset(t.data, 1.0);
        return t;
    }

    // NumPy-like API: array
    pub fn array(self: *Graph, shape_slice: []const usize, initial_data: []const f32, requires_grad: bool) !*Tensor {
        return self.tensorNDWithData(shape_slice, initial_data, requires_grad);
    }

    // NumPy-like API: transpose alias
    pub fn transpose(self: *Graph, A: *Tensor, dim0: usize, dim1: usize) !*Tensor {
        return self.transposeND(A, dim0, dim1);
    }

    // 在计算图中创建并注册一个新的 N 维张量节点
    pub fn tensorND(self: *Graph, shape_slice: []const usize, requires_grad: bool) !*Tensor {
        const allocator = self.arena.allocator();
        const t = try allocator.create(Tensor);
        const shape = Shape.init(shape_slice);
        const strides = computeContiguousStrides(shape);

        var total_size: usize = 1;
        for (shape_slice) |dim| {
            total_size *= dim;
        }

        const effective_req_grad = self.enable_grad and requires_grad;
        const is_gpu = self.useDevice();

        t.* = Tensor{
            .data = try allocator.alloc(f32, total_size),
            .grad = if (effective_req_grad) try allocator.alloc(f32, total_size) else &.{},
            .shape = shape,
            .strides = strides,
            .requires_grad = effective_req_grad,
            .creator = null,
            .scope = self.currentScope(),
            .device = if (is_gpu) self.options.device else .cpu,
            .is_graph_owned = true,
        };
        @memset(t.data, 0.0);
        if (effective_req_grad) {
            @memset(t.grad, 0.0);
        }
        try self.tensors.append(self.backing_allocator, t);
        return t;
    }

    // 形状变换算子前向传播
    pub fn reshape(self: *Graph, A: *Tensor, new_shape_slice: []const usize) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try allocator.create(Tensor);
        const shape = try Shape.fromSlice(new_shape_slice);
        const strides = computeContiguousStrides(shape);

        const old_total = A.shape.numel();
        const new_total = shape.numel();
        if (old_total != new_total) return error.ShapeMismatch;

        const req_grad = self.enable_grad and A.requires_grad;

        if (self.useDevice() and A.device_array != null) {
            const arr = try A.device_array.?.reshape(new_shape_slice, self.options.mlx_stream);
            C.* = Tensor{
                .data = if (req_grad) try allocator.alloc(f32, new_total) else &.{},
                .grad = if (req_grad) try allocator.alloc(f32, new_total) else &.{},
                .shape = shape,
                .strides = strides,
                .requires_grad = req_grad,
                .creator = null,
                .device = self.options.device,
                .device_array = arr,
                .is_graph_owned = true,
                .scope = self.currentScope(),
            };
            if (req_grad) @memset(C.grad, 0.0);
            try self.tensors.append(self.backing_allocator, C);
            return self.registerSingleOutputOp(C, &.{A}, .Reshape, .{ .Reshape = {} }, req_grad);
        }

        const can_alias = A.isContiguous() and A.data.len >= new_total;

        C.* = Tensor{
            .data = if (can_alias) A.data[0..new_total] else try allocator.alloc(f32, new_total),
            .grad = if (req_grad) try allocator.alloc(f32, new_total) else &.{},
            .shape = shape,
            .strides = strides,
            .requires_grad = req_grad,
            .creator = null,
            .is_view = can_alias,
        };
        if (!can_alias) {
            var coord: [8]usize = @splat(0);
            const len = A.shape.len;
            for (0..new_total) |dest_i| {
                var src_idx: usize = 0;
                for (0..len) |d| {
                    src_idx += coord[d] * A.strides.dims[d];
                }
                C.data[dest_i] = A.data[src_idx];

                A.shape.incrementCoord(&coord);
            }
        }
        if (req_grad) {
            @memset(C.grad, 0.0);
        }

        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Reshape,
            .{ .Reshape = {} },
            req_grad,
        );
    }

    /// 返回张量的连续 (行优先) 版本：连续张量原样返回；跨步视图 (如 `transposeView`) 物化为连续副本，
    /// 复制记录为 `Reshape` 算子，其反向按原跨步把梯度散射回视图。
    /// 只支持连续布局的算子内核在入口处调用它，从而对视图输入也给出正确的前向结果与梯度。
    pub fn contiguous(self: *Graph, t: *Tensor) !*Tensor {
        if (t.isContiguous()) return t;
        return self.reshape(t, t.shape.dims[0..t.shape.len]);
    }

    /// 将单输出张量 `C` 注册到计算图中，并在满足梯度条件时构建与记录 `Op` 节点
    pub fn registerSingleOutputOp(
        self: *Graph,
        C: *Tensor,
        inputs: []const *Tensor,
        op_type: OpType,
        context: OpContext,
        req_grad: bool,
    ) !*Tensor {
        const allocator = self.arena.allocator();
        C.scope = self.currentScope();
        C.requires_grad = req_grad;
        if (req_grad and C.grad.len == 0) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (self.enable_grad) {
            const inps_copy = try allocator.alloc(*Tensor, inputs.len);
            @memcpy(inps_copy, inputs);
            const outs_copy = try allocator.alloc(*Tensor, 1);
            outs_copy[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = op_type,
                .inputs = inps_copy,
                .outputs = outs_copy,
                .context = context,
            };
            C.creator = o.handle();
            try self.recordOp(o);
        }

        return C;
    }

    /// 对已通过 `self.tensor*` 预分配并注册的输出张量执行 `Op.forward`，并在需要梯度时记录 `Op` 节点
    pub fn runAndRecordPreallocatedOp(
        self: *Graph,
        output: *Tensor,
        inputs: []const *Tensor,
        op_type: OpType,
        context: OpContext,
    ) !*Tensor {
        for (inputs) |in| {
            if (in.device_array != null) {
                try in.syncHost(self.backing_allocator);
            }
        }
        var temp_op = Op{
            .op_type = op_type,
            .inputs = @constCast(inputs),
            .outputs = @constCast(&[_]*Tensor{output}),
            .context = context,
        };
        try op_mod.forward(&temp_op, self.backing_allocator);

        if (self.enable_grad) {
            const allocator = self.arena.allocator();
            const inps_copy = try allocator.alloc(*Tensor, inputs.len);
            @memcpy(inps_copy, inputs);
            const outs_copy = try allocator.alloc(*Tensor, 1);
            outs_copy[0] = output;

            const o = try allocator.create(Op);
            o.* = temp_op;
            o.inputs = inps_copy;
            o.outputs = outs_copy;
            output.creator = o.handle();
            try self.recordOp(o);
        }

        return output;
    }

    // 维度转置算子前向传播：交换 dim0 和 dim1
    pub fn transposeND(self: *Graph, A: *Tensor, dim0: usize, dim1: usize) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and A.requires_grad;

        if (self.useDevice() and A.device_array != null) {
            const rank = A.shape.len;
            if (dim0 >= rank or dim1 >= rank) return error.IndexOutOfBounds;
            var axes: [8]usize = undefined;
            for (0..rank) |i| axes[i] = i;
            axes[dim0] = dim1;
            axes[dim1] = dim0;
            const res = try A.device_array.?.transposeAxes(axes[0..rank], self.options.mlx_stream);
            var out_shape = A.shape;
            out_shape.dims[dim0] = A.shape.dims[dim1];
            out_shape.dims[dim1] = A.shape.dims[dim0];
            const numel = out_shape.numel();
            const C = try allocator.create(Tensor);
            C.* = Tensor{
                .data = if (req_grad) try allocator.alloc(f32, numel) else &.{},
                .grad = if (req_grad) try allocator.alloc(f32, numel) else &.{},
                .shape = out_shape,
                .strides = computeContiguousStrides(out_shape),
                .requires_grad = req_grad,
                .creator = null,
                .device = self.options.device,
                .device_array = res,
                .is_graph_owned = true,
                .scope = self.currentScope(),
            };
            if (req_grad) @memset(C.grad, 0.0);
            try self.tensors.append(self.backing_allocator, C);
            return self.registerSingleOutputOp(
                C,
                &.{A},
                .Transpose,
                .{ .Transpose = .{ .dim0 = dim0, .dim1 = dim1 } },
                req_grad,
            );
        }

        const C = try A.transpose(dim0, dim1, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Transpose,
            .{ .Transpose = .{ .dim0 = dim0, .dim1 = dim1 } },
            req_grad,
        );
    }

    // 沿指定维度拼接张量数组 (Concat)
    pub fn concat(self: *Graph, inputs_in: []const *Tensor, dim: usize) !*Tensor {
        const allocator = self.arena.allocator();
        const inputs = try allocator.alloc(*Tensor, inputs_in.len);
        for (inputs, inputs_in) |*dst, src| dst.* = try self.contiguous(src);
        const C = try tensor_mod.concat(allocator, inputs, dim);

        var req_grad = false;
        if (self.enable_grad) {
            for (inputs) |inp| {
                if (inp.requires_grad) {
                    req_grad = true;
                    break;
                }
            }
        }
        return self.registerSingleOutputOp(
            C,
            inputs,
            .Concat,
            .{ .Concat = .{ .dim = dim } },
            req_grad,
        );
    }

    // 沿指定维度将张量均等切分为 num_splits 份 (Split)
    pub fn split(self: *Graph, input_in: *Tensor, num_splits: usize, dim: usize) ![]*Tensor {
        const allocator = self.arena.allocator();
        const input = try self.contiguous(input_in);
        const outputs = try tensor_mod.split(allocator, input, num_splits, dim);

        const req_grad = self.enable_grad and input.requires_grad;
        for (outputs) |out| {
            out.scope = self.currentScope();
            out.requires_grad = req_grad;
            if (req_grad) {
                out.grad = try allocator.alloc(f32, out.data.len);
                @memset(out.grad, 0.0);
            }
            try self.tensors.append(self.backing_allocator, out);
        }

        if (self.enable_grad) {
            const inps = try allocator.alloc(*Tensor, 1);
            inps[0] = input;
            const outs_copy = try allocator.alloc(*Tensor, num_splits);
            @memcpy(outs_copy, outputs);

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Split,
                .inputs = inps,
                .outputs = outs_copy,
                .context = .{
                    .Split = .{
                        .dim = dim,
                    },
                },
            };
            for (outputs) |out| {
                out.creator = o.handle();
            }
            try self.recordOp(o);
        }

        return outputs;
    }

    // 分组查询注意力 (Grouped-Query Attention, GQA) 中沿注意力头维度复制广播键/值 (Key/Value, KV) 张量 (RepeatKV)
    // 输入 X: [B, num_kv_heads, T, hs]
    // 输出 Y: [B, num_kv_heads * groups, T, hs]
    pub fn repeatKV(self: *Graph, X_in: *Tensor, groups: usize) !*Tensor {
        if (groups == 1) return X_in;

        const allocator = self.arena.allocator();
        const X = try self.contiguous(X_in);
        const Y = try X.repeatKV(groups, allocator);
        return self.registerSingleOutputOp(
            Y,
            &.{X},
            .RepeatKV,
            .{ .RepeatKV = .{ .groups = groups } },
            self.enable_grad and X.requires_grad,
        );
    }

    /// 矩阵乘法算子前向传播：C = A * B，右操作数 B 必须是二维矩阵 [K, N] (如投影权重)。
    /// 左操作数 A 可以带任意前导批次维度 [..., K]：前导维度先展平为 [M, K] 行矩阵 (连续输入为零拷贝重塑视图)，
    /// 与所有批次共享的 B 做一次矩阵乘法后再恢复为 [..., N]。整个变换以 Reshape -> MatMul -> Reshape 算子记录，
    /// 前向重演与反向传播 (dA = dC B^T，dB = A^T dC，按展平后的行累加) 都复用二维算子。
    /// 两侧都带批次维度的 [B, H, M, K] x [B, H, K, N] 乘法使用 `batchMatMul`。
    pub fn matmul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        if (A.shape.len > 2) {
            if (B.shape.len != 2) return error.IncompatibleDimensions;
            const rank = A.shape.len;
            const K = A.shape.dims[rank - 1];
            if (K != B.shape.dims[0]) return error.ShapeMismatch;
            var out_dims: [8]usize = undefined;
            var rows: usize = 1;
            for (A.shape.dims[0 .. rank - 1], out_dims[0 .. rank - 1]) |d, *o| {
                rows *= d;
                o.* = d;
            }
            out_dims[rank - 1] = B.shape.dims[1];
            const A_rows = try self.reshape(A, &.{ rows, K });
            const C_rows = try self.matmul(A_rows, B);
            return self.reshape(C_rows, out_dims[0..rank]);
        }

        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        const C = if (self.useDevice()) blk: {
            if (A.shape.len != 2 or B.shape.len != 2) return error.IncompatibleDimensions;
            if (A.shape.dims[1] != B.shape.dims[0]) return error.ShapeMismatch;
            const out = try self.tensorND(&.{ A.shape.dims[0], B.shape.dims[1] }, req_grad);
            try backend.ops.matmul(A, B, out, self.options.device, self.options.mlx_stream);
            break :blk out;
        } else try A.matmul(B, allocator);
        return self.registerSingleOutputOp(C, &.{ A, B }, .MatMul, .{ .MatMul = {} }, req_grad);
    }

    /// 统一设备分派规则：计算图配置为 GPU 或 CUDA 且对应底层硬件加速驱动可用时，由设备执行该算子的前向计算。
    /// 无论在哪个设备上计算，算子都以相同的 `Op` 类型、输入与上下文注册，因此计算图拓扑与导出的模型图与设备完全无关。
    pub fn useDevice(self: *const Graph) bool {
        return self.options.device.isGpu() and backend.device.isAvailable(self.options.device);
    }

    // 偏置相加算子前向传播：C = A + bias (直接路由到通用广播加法)
    pub fn addBias(self: *Graph, A: *Tensor, bias: *Tensor) !*Tensor {
        return self.add(A, bias);
    }

    // 修正线性单元 (Rectified Linear Unit, ReLU) 激活函数前向传播：C = max(0, A)
    pub fn relu(self: *Graph, A_in: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const A = try self.contiguous(A_in);
        const C = try A.relu(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Relu,
            .{ .Relu = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    /// 高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 算子族：`.none` 记录为精确形式算子 `Gelu`，
    /// `.tanh` 记录为双曲正切 (Hyperbolic Tangent, Tanh) 近似形式算子 `GeluTanh`
    pub fn gelu(self: *Graph, A_in: *Tensor, approximation: backend.GeluApproximation) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and A_in.requires_grad;
        const context: types.OpContext = switch (approximation) {
            .none => .{ .Gelu = {} },
            .tanh => .{ .GeluTanh = {} },
        };
        if (self.useDevice()) {
            const out = try self.tensorND(A_in.shape.slice(), req_grad);
            try backend.ops.gelu(A_in, out, approximation, self.options.device, self.options.mlx_stream);
            return self.registerSingleOutputOp(
                out,
                &.{A_in},
                types.OpType.forGelu(approximation),
                context,
                req_grad,
            );
        }
        const A = try self.contiguous(A_in);
        const C = try A.gelu(approximation, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            types.OpType.forGelu(approximation),
            context,
            req_grad,
        );
    }

    pub fn sigmoid(self: *Graph, A_in: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const A = try self.contiguous(A_in);
        const C = try A.sigmoid(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Sigmoid,
            .{ .Sigmoid = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn tanh(self: *Graph, A_in: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const A = try self.contiguous(A_in);
        const C = try A.tanh(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Tanh,
            .{ .Tanh = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    /// 软截断 (Soft Capping)：y = cap * tanh(x / cap)，把取值平滑限制在 (-cap, cap) 内 (如 Gemma 的最终对数几率截断)；
    /// cap <= 0 时为恒等映射，直接返回输入
    pub fn softcap(self: *Graph, A_in: *Tensor, cap: f32) !*Tensor {
        if (cap <= 0.0) return A_in;
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and A_in.requires_grad;
        if (self.useDevice()) {
            const out = try self.tensorND(A_in.shape.slice(), req_grad);
            try backend.ops.softcap(A_in, out, cap, self.options.device, self.options.mlx_stream);
            return self.registerSingleOutputOp(
                out,
                &.{A_in},
                .Softcap,
                .{ .Softcap = .{ .cap = cap } },
                req_grad,
            );
        }
        const A = try self.contiguous(A_in);
        const C = try A.softcap(cap, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Softcap,
            .{ .Softcap = .{ .cap = cap } },
            req_grad,
        );
    }

    pub fn leakyRelu(self: *Graph, A_in: *Tensor, alpha: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const A = try self.contiguous(A_in);
        const C = try A.leakyRelu(alpha, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .LeakyRelu,
            .{ .LeakyRelu = .{ .alpha = alpha } },
            self.enable_grad and A.requires_grad,
        );
    }

    // Sigmoid 线性单元 (Sigmoid Linear Unit, SiLU / Swish) 激活函数前向传播：C = A * sigmoid(A)
    pub fn silu(self: *Graph, A_in: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const A = try self.contiguous(A_in);
        const C = try A.silu(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Silu,
            .{ .Silu = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    // 损失函数与随机生成绑定 (来自 graph_nn.zig)
    pub const softmaxCrossEntropy = graph_nn.softmaxCrossEntropy;
    pub const maskedCrossEntropyLoss = graph_nn.maskedCrossEntropyLoss;
    pub const dpoLoss = graph_nn.dpoLoss;
    pub const grpoLoss = graph_nn.grpoLoss;
    pub const mseLoss = graph_nn.mseLoss;
    pub const bceWithLogitsLoss = graph_nn.bceWithLogitsLoss;
    pub const sigmoidCrossEntropy = graph_nn.sigmoidCrossEntropy;
    pub const bceLoss = graph_nn.bceLoss;
    pub const randomNormal = graph_nn.randomNormal;
    pub const randomUniform = graph_nn.randomUniform;
    pub const l2Loss = graph_nn.l2Loss;
    pub const ridgeLoss = graph_nn.ridgeLoss;
    pub const l1Loss = graph_nn.l1Loss;
    pub const lassoLoss = graph_nn.lassoLoss;
    pub const elasticNetLoss = graph_nn.elasticNetLoss;

    // 标量乘法（缩放）：C = val * A
    pub fn mulScalar(self: *Graph, A_in: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and A_in.requires_grad;
        if (self.useDevice()) {
            const out = try self.tensorND(A_in.shape.slice(), req_grad);
            try backend.ops.mulScalar(A_in, val, out, self.options.device, self.options.mlx_stream);
            return self.registerSingleOutputOp(
                out,
                &.{A_in},
                .MulScalar,
                .{ .MulScalar = .{ .val = val } },
                req_grad,
            );
        }
        const A = try self.contiguous(A_in);
        const C = try A.mulScalar(val, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .MulScalar,
            .{ .MulScalar = .{ .val = val } },
            req_grad,
        );
    }

    // 标量加法：C = A + val
    pub fn addScalar(self: *Graph, A_in: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const A = try self.contiguous(A_in);
        const C = try A.addScalar(val, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .AddScalar,
            .{ .AddScalar = .{ .val = val } },
            self.enable_grad and A.requires_grad,
        );
    }

    // 标量减法：C = A - val
    pub fn subScalar(self: *Graph, A_in: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const A = try self.contiguous(A_in);
        const C = try A.subScalar(val, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .SubScalar,
            .{ .SubScalar = .{ .val = val } },
            self.enable_grad and A.requires_grad,
        );
    }

    // 标量除法：C = A / val
    pub fn divScalar(self: *Graph, A_in: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const A = try self.contiguous(A_in);
        const C = try A.divScalar(val, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .DivScalar,
            .{ .DivScalar = .{ .val = val } },
            self.enable_grad and A.requires_grad,
        );
    }

    // 逐元素张量加法：C = A + B (支持多维广播)
    pub fn add(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        const C = if (self.useDevice()) blk: {
            const out_shape = try tensor_mod.broadcastShapes(A.shape, B.shape);
            const out = try self.tensorND(out_shape.slice(), req_grad);
            try backend.ops.add(A, B, out, self.options.device, self.options.mlx_stream);
            break :blk out;
        } else try A.add(B, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{ A, B },
            .Add,
            .{ .Add = {} },
            req_grad,
        );
    }

    // 逐元素张量减法：C = A - B (支持多维广播)
    pub fn sub(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.sub(B, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{ A, B },
            .Sub,
            .{ .Sub = {} },
            self.enable_grad and (A.requires_grad or B.requires_grad),
        );
    }

    // 逐元素张量乘法 (Hadamard 积)：C = A * B (支持多维广播)
    pub fn mul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        const C = if (self.useDevice()) blk: {
            const out_shape = try tensor_mod.broadcastShapes(A.shape, B.shape);
            const out = try self.tensorND(out_shape.slice(), req_grad);
            try backend.ops.multiply(A, B, out, self.options.device, self.options.mlx_stream);
            break :blk out;
        } else try A.mul(B, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{ A, B },
            .Mul,
            .{ .Mul = {} },
            req_grad,
        );
    }

    // 逐元素张量除法：C = A / B (支持多维广播)
    pub fn div(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.div(B, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{ A, B },
            .Div,
            .{ .Div = {} },
            self.enable_grad and (A.requires_grad or B.requires_grad),
        );
    }

    // 神经网络层算子绑定 (来自 graph_nn.zig)
    pub const conv1d = graph_nn.conv1d;
    pub const conv2d = graph_nn.conv2d;
    pub const convTranspose1d = graph_nn.convTranspose1d;
    pub const convTranspose2d = graph_nn.convTranspose2d;
    pub const maxpool1d = graph_nn.maxpool1d;
    pub const maxpool2d = graph_nn.maxpool2d;
    pub const avgpool1d = graph_nn.avgpool1d;
    pub const avgpool2d = graph_nn.avgpool2d;
    pub const adaptiveAvgPool1d = graph_nn.adaptiveAvgPool1d;
    pub const adaptiveAvgPool2d = graph_nn.adaptiveAvgPool2d;
    pub const softmax = graph_nn.softmax;
    pub const rmsNorm = graph_nn.rmsNorm;
    pub const layerNorm = graph_nn.layerNorm;
    pub const batchNorm1d = graph_nn.batchNorm1d;
    pub const batchNorm2d = graph_nn.batchNorm2d;
    pub const groupNorm = graph_nn.groupNorm;
    pub const dropout = graph_nn.dropout;
    pub const rope = graph_nn.rope;
    pub const batchMatMul = graph_nn.batchMatMul;
    pub const embedding = graph_nn.embedding;
    pub const topK = graph_nn.topK;
    pub const gatherMatMul = graph_nn.gatherMatMul;
    pub const q4MatMul = graph_nn.q4MatMul;
    pub const attention = graph_nn.attention;

    pub fn sqrt(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.sqrt(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Sqrt,
            .{ .Sqrt = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn exp(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.exp(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Exp,
            .{ .Exp = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn log(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.log(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Log,
            .{ .Log = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn abs(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.abs(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Abs,
            .{ .Abs = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn sum(self: *Graph, A: *Tensor, axis: ?usize, keepdims: bool) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.sum(axis, keepdims, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Sum,
            .{ .Sum = .{ .axis = axis, .keepdims = keepdims } },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn mean(self: *Graph, A: *Tensor, axis: ?usize, keepdims: bool) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.mean(axis, keepdims, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Mean,
            .{ .Mean = .{ .axis = axis, .keepdims = keepdims } },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn variance(self: *Graph, A: *Tensor, axis: ?usize, keepdims: bool, ddof: usize) !*Tensor {
        if (axis) |ax| {
            if (ax >= A.shape.len) return error.DimensionOutOfBounds;
        }
        const count = if (axis) |ax| A.shape.dims[ax] else A.shape.numel();
        if (count <= ddof) return error.InvalidDDOF;

        const mean_t = try self.mean(A, axis, true);
        const diff = try self.sub(A, mean_t);
        const sq = try self.mul(diff, diff);
        const sum_sq = try self.sum(sq, axis, keepdims);
        return try self.divScalar(sum_sq, @as(f32, @floatFromInt(count - ddof)));
    }

    pub fn stdDev(self: *Graph, A: *Tensor, axis: ?usize, keepdims: bool, ddof: usize) !*Tensor {
        const var_t = try self.variance(A, axis, keepdims, ddof);
        return try self.sqrt(var_t);
    }

    pub fn where(self: *Graph, cond: anytype, X: *Tensor, Y: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try Tensor.where(cond, X, Y, allocator);
        const mask_buf = try allocator.alloc(bool, C.data.len);
        const cond_strides = tensor_mod.computeBroadcastStrides(cond.shape, cond.strides, C.shape);
        const rank = C.shape.len;
        var indices: [8]usize = @splat(0);
        for (0..C.data.len) |c_flat| {
            var cond_flat: usize = 0;
            for (0..rank) |d| cond_flat += indices[d] * cond_strides.dims[d];
            mask_buf[c_flat] = tensor_mod.isTruthyScalar(cond.data[cond_flat]);
            C.shape.incrementCoord(&indices);
        }
        return self.registerSingleOutputOp(
            C,
            &.{ X, Y },
            .Where,
            .{ .Where = .{ .mask = mask_buf } },
            self.enable_grad and (X.requires_grad or Y.requires_grad),
        );
    }

    pub fn maskedFill(self: *Graph, X: *Tensor, mask: anytype, value: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try X.maskedFill(mask, value, allocator);
        const mask_buf = try allocator.alloc(bool, C.data.len);
        if (mask.isContiguous() and mask.data.len >= C.data.len) {
            for (mask_buf, mask.data[0..C.data.len]) |*mb, mv| {
                mb.* = tensor_mod.isTruthyScalar(mv);
            }
        } else {
            var coord: [8]usize = @splat(0);
            const len = mask.shape.len;
            for (mask_buf) |*mb| {
                var m_idx: usize = 0;
                for (0..len) |d| m_idx += coord[d] * mask.strides.dims[d];
                mb.* = tensor_mod.isTruthyScalar(mask.data[m_idx]);
                mask.shape.incrementCoord(&coord);
            }
        }
        return self.registerSingleOutputOp(
            C,
            &.{X},
            .MaskedFill,
            .{ .MaskedFill = .{ .mask = mask_buf, .value = value } },
            self.enable_grad and X.requires_grad,
        );
    }

    pub fn squeeze(self: *Graph, A: *Tensor, axis: ?usize) !*Tensor {
        const target = try A.squeezedShape(axis);
        return self.reshape(A, target.dims[0..target.len]);
    }

    pub fn unsqueeze(self: *Graph, A: *Tensor, dim: usize) !*Tensor {
        const target = try A.unsqueezedShape(dim);
        return self.reshape(A, target.dims[0..target.len]);
    }

    pub fn slice(self: *Graph, A: *Tensor, ranges: []const tensor_mod.SliceRange) !*Tensor {
        if (ranges.len > A.shape.len) return error.DimensionOutOfBounds;

        var new_dims: [8]usize = @splat(0);
        var new_strides: [8]usize = @splat(0);
        var offset: usize = 0;

        for (0..A.shape.len) |d| {
            const dim_size = A.shape.dims[d];
            const stride = A.strides.dims[d];
            const range = if (d < ranges.len) ranges[d] else tensor_mod.SliceRange{};
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

        const req_grad = self.enable_grad and A.requires_grad;

        if (self.useDevice() and A.device_array != null) {
            var curr_arr = A.device_array.?;
            var had_slice = false;
            for (0..A.shape.len) |d| {
                const dim_size = A.shape.dims[d];
                const range = if (d < ranges.len) ranges[d] else tensor_mod.SliceRange{};
                const start = range.start orelse 0;
                const end = range.end orelse dim_size;
                if (start != 0 or end != dim_size) {
                    curr_arr = try curr_arr.sliceAxis(d, start, end, self.options.mlx_stream);
                    had_slice = true;
                }
            }
            if (!had_slice) {
                curr_arr = try curr_arr.sliceAxis(0, 0, A.shape.dims[0], self.options.mlx_stream);
            }
            const allocator = self.arena.allocator();
            const C = try allocator.create(Tensor);
            const shape = try Shape.fromSlice(new_dims[0..A.shape.len]);
            const numel = shape.numel();
            C.* = Tensor{
                .data = if (req_grad) try allocator.alloc(f32, numel) else &.{},
                .grad = if (req_grad) try allocator.alloc(f32, numel) else &.{},
                .shape = shape,
                .strides = computeContiguousStrides(shape),
                .requires_grad = req_grad,
                .creator = null,
                .device = self.options.device,
                .device_array = curr_arr,
                .is_graph_owned = true,
                .scope = self.currentScope(),
            };
            if (req_grad) @memset(C.grad, 0.0);
            try self.tensors.append(self.backing_allocator, C);
            return self.registerSingleOutputOp(
                C,
                &.{A},
                .Slice,
                .{
                    .Slice = .{
                        .offset = offset,
                        .strides = new_strides,
                        .rank = A.shape.len,
                    },
                },
                false,
            );
        }

        const C = try self.tensorND(new_dims[0..A.shape.len], req_grad);
        return self.runAndRecordPreallocatedOp(
            C,
            &.{A},
            .Slice,
            .{
                .Slice = .{
                    .offset = offset,
                    .strides = new_strides,
                    .rank = A.shape.len,
                },
            },
        );
    }

    /// 执行计算图的反向传播：把损失张量的梯度种子设为 1.0 (dL/dL = 1.0) 后，
    /// 按拓扑逆序把梯度传导到所有需要梯度的叶子张量 (参数梯度累加，可连续调用以累积多个损失的梯度)。
    /// 损失张量不需要梯度时返回 `error.TensorDoesNotRequireGrad`。
    pub fn backward(self: *Graph, loss_tensor: *Tensor) !void {
        try self.runBackward(loss_tensor, true);
    }

    /// 执行计算图的反向传播，起点张量的梯度已被外部手动预填（常用于自定义损失函数或向量-雅可比积）
    pub fn backwardWithGrad(self: *Graph, output_tensor: *Tensor) !void {
        try self.runBackward(output_tensor, false);
    }

    fn runBackward(self: *Graph, root: *Tensor, seed_root: bool) !void {
        if (!root.requires_grad or root.grad.len == 0) return error.TensorDoesNotRequireGrad;

        const allocator = self.backing_allocator;
        var order: std.ArrayList(*Op) = .empty;
        defer order.deinit(allocator);
        if (Op.creatorOf(root)) |root_op| {
            try self.collectBackwardOps(root_op, &order);
        }

        // 确保参与反向传播计算的所有张量数据从 GPU 同步回主机内存
        for (order.items) |op| {
            for (op.inputs) |in| {
                if (in.device_array != null) {
                    try in.syncHost(allocator);
                }
            }
            for (op.outputs) |out| {
                if (out.device_array != null) {
                    try out.syncHost(allocator);
                }
            }
        }

        // 中间张量的梯度只在本次反向传播内有效：先清零，避免上一次反向残留的梯度被重复传导到参数
        for (order.items) |op| {
            for (op.outputs) |out| {
                if (out != root) @memset(out.grad, 0.0);
            }
        }
        if (seed_root) root.grad[0] = 1.0;

        var i = order.items.len;
        while (i > 0) {
            i -= 1;
            try op_mod.backward(order.items[i]);
        }
    }

    /// 以迭代式深度优先搜索 (Depth-First Search, DFS) 收集从 `root_op` 可达、且确实需要传导梯度的算子，
    /// 按拓扑序 (输入先于消费者) 追加到 `order`。只沿 `requires_grad` 的输入回溯：
    /// 不需要梯度的子图 (冻结的骨干网络、常量分支、仅推理的算子) 不参与反向传播。
    fn collectBackwardOps(self: *Graph, root_op: *Op, order: *std.ArrayList(*Op)) !void {
        const allocator = self.backing_allocator;
        var visited: std.AutoHashMapUnmanaged(*Op, void) = .empty;
        defer visited.deinit(allocator);
        try visited.ensureTotalCapacity(allocator, @intCast(@max(self.ops.items.len, 1)));

        const Frame = struct { op: *Op, next_input: usize };
        var stack: std.ArrayList(Frame) = .empty;
        defer stack.deinit(allocator);

        visited.putAssumeCapacity(root_op, {});
        try stack.append(allocator, .{ .op = root_op, .next_input = 0 });
        while (stack.items.len > 0) {
            const top = &stack.items[stack.items.len - 1];
            if (top.next_input < top.op.inputs.len) {
                const input = top.op.inputs[top.next_input];
                top.next_input += 1;
                if (!input.requires_grad) continue;
                const prev_op = Op.creatorOf(input) orelse continue;
                const entry = try visited.getOrPut(allocator, prev_op);
                if (entry.found_existing) continue;
                try stack.append(allocator, .{ .op = prev_op, .next_input = 0 });
            } else {
                try order.append(allocator, top.op);
                _ = stack.pop();
            }
        }
    }

    // 运行计算图的前向传播，根据输入更新所有算子节点的值
    pub fn forward(self: *Graph) !void {
        for (self.ops.items) |op| {
            try op_mod.forward(op, self.backing_allocator);
        }
    }

    // 将计算图中所有注册张量的梯度清零
    pub fn zeroGrad(self: *Graph) void {
        for (self.tensors.items) |t| {
            t.zeroGrad();
        }
    }

    // 参数初始化与架构报告绑定 (来自 graph_init.zig)
    pub const initWeights = graph_init.initWeights;
    pub const initSingleTensor = graph_init.initSingleTensor;
    pub const describeAutoGraphParam = graph_init.describeAutoGraphParam;
    pub const formatInitReport = graph_init.formatInitReport;
    pub const printInitReport = graph_init.printInitReport;
    pub const appendSingleTensorReport = graph_init.appendSingleTensorReport;
    pub const detectConsumerActivation = graph_init.detectConsumerActivation;
    pub const computeParamFans = graph_init.computeParamFans;
};
