const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const Shape = tensor_mod.Shape;
const computeContiguousStrides = tensor_mod.computeContiguousStrides;
const types = @import("types.zig");
pub const OpType = types.OpType;
pub const OpContext = types.OpContext;
const op_mod = @import("op.zig");
pub const Op = op_mod.Op;
const graph_nn = @import("graph_nn.zig");
const graph_init = @import("graph_init.zig");

// 计算图（Graph）结构体
// 追踪所有的张量节点与算子节点，管理内存生命周期并负责反向传播调度
pub const Graph = struct {
    backing_allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator, // 使用 Arena 机制，使每次前向/反向生成的中间节点内存可在 batch 结束时一并释放，避免内存碎片和频繁分配
    tensors: std.ArrayList(*Tensor), // 追踪计算图中的所有张量指针
    ops: std.ArrayList(*Op), // 追踪计算图中的所有算子指针
    enable_grad: bool, // 梯度使能开关（类似 torch.set_grad_enabled），为 false 时不分配梯度缓冲区亦不记录 Op 节点
    module_formulas: std.StringHashMap([]const u8), // 存储模块在代码中声明的显式数学运算公式 (如 "y = x W^T + b")
    module_types: std.StringHashMap([]const u8), // 存储模块在代码中声明的显式模块类型 (如 "Linear", "RMSNorm", "GPT")
    scope_stack: std.ArrayList([]const u8), // 当前正在执行 forward 的模块作用域栈 (栈顶为最内层模块的完整路径)

    // 初始化计算图，传入底层通用内存分配器
    pub fn init(backing_allocator: std.mem.Allocator) Graph {
        return Graph{
            .backing_allocator = backing_allocator,
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .tensors = .empty,
            .ops = .empty,
            .enable_grad = true,
            .module_formulas = std.StringHashMap([]const u8).init(backing_allocator),
            .module_types = std.StringHashMap([]const u8).init(backing_allocator),
            .scope_stack = .empty,
        };
    }

    /// 初始化一个不记录梯度的计算图 (推理 / 评估模式)：
    /// 算子仍在图的 Arena 中分配并执行前向计算，但不分配梯度缓冲区，也不记录 Op 节点
    pub fn initNoGrad(backing_allocator: std.mem.Allocator) Graph {
        var g = Graph.init(backing_allocator);
        g.enable_grad = false;
        return g;
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
    }

    // 释放整个计算图的内存（包括所有张量与算子节点的前向/反向缓冲区）
    pub fn deinit(self: *Graph) void {
        self.module_formulas.deinit();
        self.module_types.deinit();
        self.scope_stack.deinit(self.backing_allocator);
        self.tensors.deinit(self.backing_allocator);
        self.ops.deinit(self.backing_allocator);
        self.arena.deinit();
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

        t.* = Tensor{
            .data = try allocator.alloc(f32, total_size),
            .grad = if (effective_req_grad) try allocator.alloc(f32, total_size) else &.{},
            .shape = shape,
            .strides = strides,
            .requires_grad = effective_req_grad,
            .creator = null,
            .scope = self.currentScope(),
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
            var coord = [_]usize{0} ** 8;
            const len = A.shape.len;
            for (0..new_total) |dest_i| {
                var src_idx: usize = 0;
                for (0..len) |d| {
                    src_idx += coord[d] * A.strides.dims[d];
                }
                C.data[dest_i] = A.data[src_idx];

                var d = len;
                while (d > 0) {
                    d -= 1;
                    coord[d] += 1;
                    if (coord[d] < A.shape.dims[d]) break;
                    coord[d] = 0;
                }
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
            C.creator = o;
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
        req_grad: bool,
    ) !*Tensor {
        _ = req_grad;
        var temp_op = Op{
            .op_type = op_type,
            .inputs = @constCast(inputs),
            .outputs = @constCast(&[_]*Tensor{output}),
            .context = context,
        };
        try temp_op.forward(self.backing_allocator);

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
            output.creator = o;
            try self.recordOp(o);
        }

        return output;
    }

    // 维度转置算子前向传播：交换 dim0 和 dim1
    pub fn transposeND(self: *Graph, A: *Tensor, dim0: usize, dim1: usize) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.transpose(dim0, dim1, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Transpose,
            .{ .Transpose = .{ .dim0 = dim0, .dim1 = dim1 } },
            self.enable_grad and A.requires_grad,
        );
    }

    // 沿指定维度拼接张量数组 (Concat)
    pub fn concat(self: *Graph, inputs: []const *Tensor, dim: usize) !*Tensor {
        const allocator = self.arena.allocator();
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
    pub fn split(self: *Graph, input: *Tensor, num_splits: usize, dim: usize) ![]*Tensor {
        const allocator = self.arena.allocator();
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
                out.creator = o;
            }
            try self.recordOp(o);
        }

        return outputs;
    }

    // 分组查询注意力 (Grouped-Query Attention, GQA) 中沿注意力头维度复制广播键/值 (Key/Value, KV) 张量 (RepeatKV)
    // 输入 X: [B, num_kv_heads, T, hs]
    // 输出 Y: [B, num_kv_heads * groups, T, hs]
    pub fn repeatKV(self: *Graph, X: *Tensor, groups: usize) !*Tensor {
        if (groups == 1) return X;

        const allocator = self.arena.allocator();
        const Y = try X.repeatKV(groups, allocator);
        return self.registerSingleOutputOp(
            Y,
            &.{X},
            .RepeatKV,
            .{ .RepeatKV = .{ .groups = groups } },
            self.enable_grad and X.requires_grad,
        );
    }

    // 矩阵乘法算子前向传播：C = A * B
    pub fn matmul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.matmul(B, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{ A, B },
            .MatMul,
            .{ .MatMul = {} },
            self.enable_grad and (A.requires_grad or B.requires_grad),
        );
    }

    // 偏置相加算子前向传播：C = A + bias (直接路由到通用广播加法)
    pub fn addBias(self: *Graph, A: *Tensor, bias: *Tensor) !*Tensor {
        return self.add(A, bias);
    }

    // 修正线性单元 (Rectified Linear Unit, ReLU) 激活函数前向传播：C = max(0, A)
    pub fn relu(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.relu(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Relu,
            .{ .Relu = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn gelu(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.gelu(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Gelu,
            .{ .Gelu = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn sigmoid(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.sigmoid(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Sigmoid,
            .{ .Sigmoid = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn tanh(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.tanh(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Tanh,
            .{ .Tanh = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn leakyRelu(self: *Graph, A: *Tensor, alpha: f32) !*Tensor {
        const allocator = self.arena.allocator();
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
    pub fn silu(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
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
    pub fn mulScalar(self: *Graph, A: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.mulScalar(val, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .MulScalar,
            .{ .MulScalar = .{ .val = val } },
            self.enable_grad and A.requires_grad,
        );
    }

    // 标量加法：C = A + val
    pub fn addScalar(self: *Graph, A: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
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
    pub fn subScalar(self: *Graph, A: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
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
    pub fn divScalar(self: *Graph, A: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
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
        const C = try A.add(B, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{ A, B },
            .Add,
            .{ .Add = {} },
            self.enable_grad and (A.requires_grad or B.requires_grad),
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
        const C = try A.mul(B, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{ A, B },
            .Mul,
            .{ .Mul = {} },
            self.enable_grad and (A.requires_grad or B.requires_grad),
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
    pub const convTranspose2D = graph_nn.convTranspose2d;
    pub const maxpool1d = graph_nn.maxpool1d;
    pub const maxpool2d = graph_nn.maxpool2d;
    pub const avgpool1d = graph_nn.avgpool1d;
    pub const avgpool2d = graph_nn.avgpool2d;
    pub const adaptiveAvgPool1d = graph_nn.adaptiveAvgPool1d;
    pub const adaptiveAvgPool2d = graph_nn.adaptiveAvgPool2d;
    pub const softmax = graph_nn.softmax;
    pub const rmsNorm = graph_nn.rmsNorm;
    pub const layerNorm = graph_nn.layerNorm;
    pub const batchNorm2d = graph_nn.batchNorm2d;
    pub const dropout = graph_nn.dropout;
    pub const rope = graph_nn.rope;
    pub const ropeOffset = graph_nn.ropeOffset;
    pub const batchMatMul = graph_nn.batchMatMul;
    pub const embedding = graph_nn.embedding;

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
        var indices = [_]usize{0} ** 8;
        for (0..C.data.len) |c_flat| {
            var cond_flat: usize = 0;
            for (0..rank) |d| cond_flat += indices[d] * cond_strides.dims[d];
            mask_buf[c_flat] = tensor_mod.isTruthyScalar(cond.data[cond_flat]);
            var d = rank;
            while (d > 0) {
                d -= 1;
                indices[d] += 1;
                if (indices[d] < C.shape.dims[d]) break;
                indices[d] = 0;
            }
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
            var coord = [_]usize{0} ** 8;
            const len = mask.shape.len;
            for (mask_buf) |*mb| {
                var m_idx: usize = 0;
                for (0..len) |d| m_idx += coord[d] * mask.strides.dims[d];
                mb.* = tensor_mod.isTruthyScalar(mask.data[m_idx]);
                var d = len;
                while (d > 0) {
                    d -= 1;
                    coord[d] += 1;
                    if (coord[d] < mask.shape.dims[d]) break;
                    coord[d] = 0;
                }
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

        var new_dims = [_]usize{0} ** 8;
        var new_strides = [_]usize{0} ** 8;
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
            req_grad,
        );
    }

    // 执行计算图的反向传播
    pub fn backward(self: *Graph, loss_tensor: *Tensor) !void {
        const allocator = self.arena.allocator();
        var visited = std.AutoHashMap(*Tensor, void).init(allocator);
        defer visited.deinit();
        var sorted_list: std.ArrayList(*Tensor) = .empty;
        defer sorted_list.deinit(allocator);

        // 1. 对计算图进行拓扑排序，以确保节点按正确的计算依赖关系进行链式求导
        try self.topologicalSort(loss_tensor, &visited, &sorted_list);

        // 2. 损失函数节点本身的偏导数设为 1.0 (dL/dL = 1.0)
        loss_tensor.grad[0] = 1.0;

        // 3. 按拓扑排序的逆序执行各算子的 backward 求导函数，由深至浅传导梯度
        var i = sorted_list.items.len;
        while (i > 0) {
            i -= 1;
            const node = sorted_list.items[i];
            if (node.creator) |op| {
                try op.backward();
            }
        }
    }

    // 执行计算图的反向传播，起点张量的梯度已被外部手动预填（常用于自定义损失函数如 MSE）
    pub fn backwardWithGrad(self: *Graph, output_tensor: *Tensor) !void {
        const allocator = self.arena.allocator();
        var visited = std.AutoHashMap(*Tensor, void).init(allocator);
        defer visited.deinit();
        var sorted_list: std.ArrayList(*Tensor) = .empty;
        defer sorted_list.deinit(allocator);

        // 1. 对计算图进行拓扑排序
        try self.topologicalSort(output_tensor, &visited, &sorted_list);

        // 2. 按拓扑排序的逆序执行各算子的 backward 求导，由深至浅传导梯度
        var i = sorted_list.items.len;
        while (i > 0) {
            i -= 1;
            const node = sorted_list.items[i];
            if (node.creator) |op| {
                try op.backward();
            }
        }
    }

    // 拓扑排序辅助函数（深度优先搜索 DFS 实现）
    fn topologicalSort(self: *Graph, node: *Tensor, visited: *std.AutoHashMap(*Tensor, void), list: *std.ArrayList(*Tensor)) !void {
        if (visited.contains(node)) return;
        try visited.put(node, {});

        if (node.creator) |op| {
            for (op.inputs) |input| {
                try self.topologicalSort(input, visited, list);
            }
        }
        try list.append(self.arena.allocator(), node);
    }

    // 运行计算图的前向传播，根据输入更新所有算子节点的值
    pub fn forward(self: *Graph) !void {
        for (self.ops.items) |op| {
            try op.forward(self.backing_allocator);
        }
    }

    // 将计算图中所有注册张量的梯度清零
    pub fn zeroGrad(self: *Graph) void {
        for (self.tensors.items) |t| {
            t.zeroGrad();
        }
    }

    // 参数初始化、架构报告与 JSON 导出绑定 (来自 graph_init.zig)
    pub const initWeights = graph_init.initWeights;
    pub const initSingleTensor = graph_init.initSingleTensor;
    pub const describeAutoGraphParam = graph_init.describeAutoGraphParam;
    pub const formatInitReport = graph_init.formatInitReport;
    pub const printInitReport = graph_init.printInitReport;
    pub const formatJson = graph_init.formatJson;
    pub const exportJson = graph_init.exportJson;
    pub const appendSingleTensorReport = graph_init.appendSingleTensorReport;
    pub const detectConsumerActivation = graph_init.detectConsumerActivation;
    pub const computeParamFans = graph_init.computeParamFans;
};
