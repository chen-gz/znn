const std = @import("std");
const tensor_mod = @import("../tensor.zig");
const Tensor = tensor_mod.Tensor;
const Shape = tensor_mod.Shape;
const computeContiguousStrides = tensor_mod.computeContiguousStrides;
const transposeShape = tensor_mod.transposeShape;
const types = @import("types.zig");
pub const OpType = types.OpType;
pub const OpContext = types.OpContext;
const op_mod = @import("op.zig");
pub const Op = op_mod.Op;

// 计算图（Graph）结构体
// 追踪所有的张量节点与算子节点，管理内存生命周期并负责反向传播调度
pub const Graph = struct {
    backing_allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,       // 使用 Arena 机制，使每次前向/反向生成的中间节点内存可在 batch 结束时一并释放，避免内存碎片和频繁分配
    tensors: std.ArrayList(*Tensor),     // 追踪计算图中的所有张量指针
    ops: std.ArrayList(*Op),             // 追踪计算图中的所有算子指针
    enable_grad: bool,                   // 梯度使能开关（类似 torch.set_grad_enabled），为 false 时不分配梯度缓冲区亦不记录 Op 节点
    module_formulas: std.StringHashMap([]const u8), // 存储模块在代码中声明的显式数学运算公式 (如 "y = x W^T + b")
    module_types: std.StringHashMap([]const u8),    // 存储模块在代码中声明的显式模块类型 (如 "Linear", "RMSNorm", "GPT")
    scope_stack: std.ArrayList([]const u8),         // 当前正在执行 forward 的模块作用域栈 (栈顶为最内层模块的完整路径)

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
    fn recordOp(self: *Graph, o: *Op) !void {
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

    /// 从计算图中自动推导指定模块或节点的数学公式
    pub fn inferModuleFormula(self: *const Graph, module_path: []const u8) []const u8 {
        // 1. 如果有显式指定的公式，直接返回
        if (self.getModuleFormula(module_path)) |form| {
            return form;
        }

        // 2. 检查计算图中的具体算子 (Ops) 是否有输出精确匹配该节点名称
        // （必须先于模块前缀继承，避免像 gpt.layers.0.attn.act_Add_20 被继承为 Attention 的全局公式）
        for (self.ops.items) |op| {
            if (op.outputs.len > 0) {
                if (op.outputs[0].name) |out_name| {
                    if (std.mem.eql(u8, out_name, module_path)) {
                        return op.op_type.getFormula();
                    }
                }
            }
        }

        // 3. 匹配子路径的最长前缀 (例如 "gpt.layers.0.output" -> 优先继承更长的 "gpt.layers.0" 而非根路径 "gpt")
        var best_prefix_match: ?[]const u8 = null;
        var best_prefix_len: usize = 0;
        var it = self.module_formulas.iterator();
        while (it.next()) |entry| {
            const k_len = entry.key_ptr.len;
            if (module_path.len > k_len and
                std.mem.startsWith(u8, module_path, entry.key_ptr.*) and
                module_path[k_len] == '.')
            {
                if (k_len > best_prefix_len) {
                    best_prefix_len = k_len;
                    best_prefix_match = entry.value_ptr.*;
                }
            }
        }
        if (best_prefix_match) |form| {
            return form;
        }

        // 4. 根据模块注册的原生类型推导标准公式 (避免脆弱的字符串模式推测)
        if (self.getModuleType(module_path)) |m_type| {
            if (std.mem.eql(u8, m_type, "Linear")) return "y = x W^T + b";
            if (std.mem.eql(u8, m_type, "Conv2D")) return "y = \\text{Conv2D}(x; W, b)";
            if (std.mem.eql(u8, m_type, "ConvTranspose2D")) return "y = \\text{ConvTranspose2D}(x; W, b)";
            if (std.mem.eql(u8, m_type, "RMSNorm")) return "y = \\text{RMSNorm}(x; \\gamma, \\epsilon)";
            if (std.mem.eql(u8, m_type, "LayerNorm")) return "y = \\text{LayerNorm}(x; \\gamma, \\beta)";
            if (std.mem.eql(u8, m_type, "BatchNorm2d")) return "y = \\text{BatchNorm2d}(x; \\gamma, \\beta)";
            if (std.mem.eql(u8, m_type, "Embedding")) return "y = \\text{Embedding}(x; W)";
            if (std.mem.eql(u8, m_type, "MLP")) return "y = \\text{GELU}(x W_{fc}^T + b_{fc}) W_{proj}^T + b_{proj}";
            if (std.mem.eql(u8, m_type, "SwiGLU")) return "y = (\\text{SiLU}(x W_{\\text{gate}}) \\odot (x W_{\\text{up}})) W_{\\text{down}}";
            if (std.mem.eql(u8, m_type, "CausalSelfAttention")) return "A = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V \\cdot W_o^T + b_o";
            if (std.mem.eql(u8, m_type, "ScaledDotProductAttention")) return "\\text{AttentionCore}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V";
            if (std.mem.eql(u8, m_type, "TransformerBlock")) return "h_l = x_l + \\text{Attention}(\\text{RMSNorm}(x_l)), \\quad x_{l+1} = \\text{TransformerBlock}(x_l) = h_l + \\text{MLP}(\\text{RMSNorm}(h_l))";
            if (std.mem.eql(u8, m_type, "TransformerDecoder")) return "x_L = \\text{DecoderStack}(x_0) = (\\text{Block}_L \\circ \\dots \\circ \\text{Block}_1)(x_0)";
            if (std.mem.eql(u8, m_type, "GPT")) return "\\text{logits} = \\text{GPT}(\\text{TokenIDs}; \\theta) \\rightarrow [B, T, V]";
            if (std.mem.eql(u8, m_type, "RNNCell")) return "h_t = \\tanh(x_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh})";
            if (std.mem.eql(u8, m_type, "RNN")) return "h_{1:T} = \\text{RNN}(x_{1:T}, h_0)";
            if (std.mem.eql(u8, m_type, "LSTMCell")) return "c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t, \\quad h_t = o_t \\odot \\tanh(c_t)";
            if (std.mem.eql(u8, m_type, "LSTM")) return "(h_{1:T}, c_{1:T}) = \\text{LSTM}(x_{1:T}, h_0, c_0)";
            if (std.mem.eql(u8, m_type, "StackedLSTM")) return "h^{(L)}_{1:T} = \\text{StackedLSTM}(x_{1:T})";
            if (std.mem.eql(u8, m_type, "GRUCell")) return "h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tanh(W_h x_t + U_h (r_t \\odot h_{t-1}))";
            if (std.mem.eql(u8, m_type, "GRU")) return "h_{1:T} = \\text{GRU}(x_{1:T}, h_0)";
            if (std.mem.eql(u8, m_type, "MoELayer")) return "y = \\sum_{i \\in \\text{TopK}(g(x))} p_i(x) E_i(x) + \\sum_{j} E^{\\text{shared}}_j(x)";
            if (std.mem.eql(u8, m_type, "MLALayer")) return "c_t^{KV} = x_t W^{DKV}, \\quad y = \\text{MLA}(Q, c^{KV}, k^R) W^O";
            if (std.mem.eql(u8, m_type, "LoRALinear")) return "y = x W_0 + \\frac{\\alpha}{r} (x A) B + b";
        }

        return "y = f(x; \\theta)";
    }

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
        return self.tensorND(&.{rows, cols}, requires_grad);
    }

    // 创建并注册一个带初始数据的二维张量节点
    pub fn tensorWithData(self: *Graph, rows: usize, cols: usize, initial_data: []const f32, requires_grad: bool) !*Tensor {
        return self.tensorNDWithData(&.{rows, cols}, initial_data, requires_grad);
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
    fn registerSingleOutputOp(
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

        if (req_grad) {
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
    fn runAndRecordPreallocatedOp(
        self: *Graph,
        output: *Tensor,
        inputs: []const *Tensor,
        op_type: OpType,
        context: OpContext,
        req_grad: bool,
    ) !*Tensor {
        var temp_op = Op{
            .op_type = op_type,
            .inputs = @constCast(inputs),
            .outputs = @constCast(&[_]*Tensor{output}),
            .context = context,
        };
        try temp_op.forward(self.backing_allocator);

        if (req_grad) {
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

        if (req_grad) {
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

    // GQA 注意力中沿 Head 维度复制广播 Key / Value 张量 (RepeatKV)
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

    // 激活函数 ReLU 前向传播：C = max(0, A)
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

    // 激活函数 SiLU (Swish) 前向传播：C = A * sigmoid(A)
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

    // 损失函数 Softmax + Cross Entropy 结合前向传播
    // 在 logits 的行维度计算 Softmax 概率分布，并与 targets 分类标签（支持 u8/u32/usize 等任意整型切片）计算交叉熵损失
    pub fn softmaxCrossEntropy(self: *Graph, logits: *Tensor, targets: anytype) !*Tensor {
        const B = logits.shape.dims[0];
        const N = logits.shape.dims[1];
        if (targets.len != B) return error.ShapeMismatch;

        const allocator = self.arena.allocator();
        const targets_copy = try allocator.alloc(usize, B);
        for (0..B) |i| {
            const label: usize = @intCast(targets[i]);
            if (label >= N) return error.IndexOutOfBounds;
            targets_copy[i] = label;
        }

        const req_grad = self.enable_grad and logits.requires_grad;
        const loss = try self.tensor(1, 1, req_grad);

        var empty_probs: [0]f32 = .{};
        const probs: []f32 = if (req_grad) try allocator.alloc(f32, B * N) else &empty_probs;

        return self.runAndRecordPreallocatedOp(
            loss,
            &.{logits},
            .SoftmaxCrossEntropy,
            .{
                .SoftmaxCrossEntropy = .{
                    .probs = probs,
                    .targets = targets_copy,
                    .mask = null,
                    .total_weight = @as(f32, @floatFromInt(B)),
                },
            },
            req_grad,
        );
    }

    // 监督微调 (SFT) 掩码交叉熵损失：仅对 mask[i] > 0 的位置计算交叉熵并支持 Autograd 反向传播
    pub fn maskedCrossEntropyLoss(self: *Graph, logits: *Tensor, targets: anytype, mask: []const f32) !*Tensor {
        const B = logits.shape.dims[0];
        const N = logits.shape.dims[1];
        if (targets.len != B or mask.len != B) return error.ShapeMismatch;

        const allocator = self.arena.allocator();
        const targets_copy = try allocator.alloc(usize, B);
        for (0..B) |i| {
            const label: usize = @intCast(targets[i]);
            if (label >= N) return error.IndexOutOfBounds;
            targets_copy[i] = label;
        }
        const mask_copy = try allocator.alloc(f32, B);
        @memcpy(mask_copy, mask);

        const req_grad = self.enable_grad and logits.requires_grad;
        const loss = try self.tensor(1, 1, req_grad);

        var empty_probs: [0]f32 = .{};
        const probs: []f32 = if (req_grad) try allocator.alloc(f32, B * N) else &empty_probs;
        if (req_grad) @memset(probs, 0.0);

        return self.runAndRecordPreallocatedOp(
            loss,
            &.{logits},
            .SoftmaxCrossEntropy,
            .{
                .SoftmaxCrossEntropy = .{
                    .probs = probs,
                    .targets = targets_copy,
                    .mask = mask_copy,
                    .total_weight = 0.0,
                },
            },
            req_grad,
        );
    }

    // 直接偏好优化 (DPO) 损失函数：支持对策略模型对数概率 pi_chosen_logps / pi_rejected_logps 的计算图反向传播
    pub fn dpoLoss(
        self: *Graph,
        pi_chosen_logps: *Tensor,
        pi_rejected_logps: *Tensor,
        ref_chosen_logps: []const f32,
        ref_rejected_logps: []const f32,
        beta: f32,
    ) !*Tensor {
        const N = pi_chosen_logps.data.len;
        if (pi_rejected_logps.data.len != N or ref_chosen_logps.len != N or ref_rejected_logps.len != N) {
            return error.ShapeMismatch;
        }

        const allocator = self.arena.allocator();
        const ref_c_copy = try allocator.alloc(f32, N);
        @memcpy(ref_c_copy, ref_chosen_logps);
        const ref_r_copy = try allocator.alloc(f32, N);
        @memcpy(ref_r_copy, ref_rejected_logps);

        const req_grad = self.enable_grad and (pi_chosen_logps.requires_grad or pi_rejected_logps.requires_grad);
        const loss = try self.tensor(1, 1, req_grad);

        return self.runAndRecordPreallocatedOp(
            loss,
            &.{ pi_chosen_logps, pi_rejected_logps },
            .DpoLoss,
            .{
                .DpoLoss = .{
                    .ref_chosen = ref_c_copy,
                    .ref_rejected = ref_r_copy,
                    .beta = beta,
                },
            },
            req_grad,
        );
    }

    // 组相对策略优化 (GRPO) 损失函数：支持在计算图中对 new_logps 自动微分求导
    pub fn grpoLoss(
        self: *Graph,
        old_logps: *Tensor,
        new_logps: *Tensor,
        advantages: []const f32,
        ref_logps: ?[]const f32,
        beta: f32,
        clip_eps: f32,
    ) !*Tensor {
        const N = old_logps.data.len;
        if (new_logps.data.len != N or advantages.len != N) return error.ShapeMismatch;
        if (ref_logps) |refs| {
            if (refs.len != N) return error.ShapeMismatch;
        }

        const allocator = self.arena.allocator();
        const adv_copy = try allocator.alloc(f32, N);
        @memcpy(adv_copy, advantages);

        var ref_copy: ?[]const f32 = null;
        if (ref_logps) |refs| {
            const rc = try allocator.alloc(f32, N);
            @memcpy(rc, refs);
            ref_copy = rc;
        }

        const req_grad = self.enable_grad and new_logps.requires_grad;
        const loss = try self.tensor(1, 1, req_grad);

        return self.runAndRecordPreallocatedOp(
            loss,
            &.{ old_logps, new_logps },
            .GrpoLoss,
            .{
                .GrpoLoss = .{
                    .advantages = adv_copy,
                    .ref_logps = ref_copy,
                    .beta = beta,
                    .clip_eps = clip_eps,
                },
            },
            req_grad,
        );
    }

    // 均方误差 (MSE) 损失函数：C = 1/N * sum((y_pred - y_true)^2)
    pub fn mseLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor) !*Tensor {
        const req_grad = self.enable_grad and (y_pred.requires_grad or y_true.requires_grad);
        const loss = try self.tensor(1, 1, req_grad);
        return self.runAndRecordPreallocatedOp(
            loss,
            &.{ y_pred, y_true },
            .MseLoss,
            .{ .MseLoss = {} },
            req_grad,
        );
    }

    pub fn bceWithLogitsLoss(self: *Graph, logits: *Tensor, targets: *Tensor) !*Tensor {
        const req_grad = self.enable_grad and (logits.requires_grad or targets.requires_grad);
        const loss = try self.tensor(1, 1, req_grad);
        return self.runAndRecordPreallocatedOp(
            loss,
            &.{ logits, targets },
            .BceWithLogitsLoss,
            .{ .BceWithLogitsLoss = {} },
            req_grad,
        );
    }

    pub fn sigmoidCrossEntropy(self: *Graph, logits: *Tensor, targets: *Tensor) !*Tensor {
        return self.bceWithLogitsLoss(logits, targets);
    }

    pub fn bceLoss(self: *Graph, probs: *Tensor, targets: *Tensor, eps: f32) !*Tensor {
        const req_grad = self.enable_grad and (probs.requires_grad or targets.requires_grad);
        const loss = try self.tensor(1, 1, req_grad);
        return self.runAndRecordPreallocatedOp(
            loss,
            &.{ probs, targets },
            .BceLoss,
            .{ .BceLoss = .{ .eps = eps } },
            req_grad,
        );
    }

    pub fn randomNormal(self: *Graph, shape_slice: []const usize, random: std.Random, mean_val: f32, stddev: f32, requires_grad: bool) !*Tensor {
        const t = try self.tensorND(shape_slice, requires_grad);
        t.fillNormal(random, mean_val, stddev);
        return t;
    }

    pub fn randomUniform(self: *Graph, shape_slice: []const usize, random: std.Random, min: f32, max: f32, requires_grad: bool) !*Tensor {
        const t = try self.tensorND(shape_slice, requires_grad);
        t.fillUniform(random, min, max);
        return t;
    }

    // L2 正则化损失函数：C = 0.5 * lambda * sum(weight_i^2)
    pub fn l2Loss(self: *Graph, weight: *Tensor, lambda: f32) !*Tensor {
        const req_grad = self.enable_grad and weight.requires_grad;
        const loss = try self.tensor(1, 1, req_grad);
        return self.runAndRecordPreallocatedOp(
            loss,
            &.{weight},
            .L2Loss,
            .{ .L2Loss = .{ .lambda = lambda } },
            req_grad,
        );
    }

    // 岭回归 (Ridge) 组合损失函数：Loss = MSE(y_pred, y_true) + 0.5 * lambda * sum(weight_i^2)
    pub fn ridgeLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor, weight: *Tensor, lambda: f32) !*Tensor {
        const mse = try self.mseLoss(y_pred, y_true);
        if (lambda == 0.0) return mse;
        const l2 = try self.l2Loss(weight, lambda);
        return try self.add(mse, l2);
    }

    // L1 正则化损失：Loss = lambda * sum(|weight_i|)
    pub fn l1Loss(self: *Graph, weight: *Tensor, lambda: f32) !*Tensor {
        const req_grad = self.enable_grad and weight.requires_grad;
        const loss = try self.tensor(1, 1, req_grad);
        return self.runAndRecordPreallocatedOp(
            loss,
            &.{weight},
            .L1Loss,
            .{ .L1Loss = .{ .lambda = lambda } },
            req_grad,
        );
    }

    // Lasso 组合损失函数：Loss = MSE(y_pred, y_true) + lambda * sum(|weight_i|)
    pub fn lassoLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor, weight: *Tensor, lambda: f32) !*Tensor {
        const mse = try self.mseLoss(y_pred, y_true);
        if (lambda == 0.0) return mse;
        const l1 = try self.l1Loss(weight, lambda);
        return try self.add(mse, l1);
    }

    // Elastic Net 组合损失函数：Loss = MSE(y_pred, y_true) + lambda * rho * ||w||_1 + 0.5 * lambda * (1 - rho) * ||w||_2^2
    pub fn elasticNetLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor, weight: *Tensor, lambda: f32, l1_ratio: f32) !*Tensor {
        const mse = try self.mseLoss(y_pred, y_true);
        if (lambda == 0.0) return mse;
        var total_loss = mse;
        if (l1_ratio > 0.0) {
            const l1 = try self.l1Loss(weight, lambda * l1_ratio);
            total_loss = try self.add(total_loss, l1);
        }
        if (l1_ratio < 1.0) {
            const l2 = try self.l2Loss(weight, lambda * (1.0 - l1_ratio));
            total_loss = try self.add(total_loss, l2);
        }
        return total_loss;
    }

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

    pub fn conv2d(self: *Graph, A: *Tensor, weight: *Tensor, bias: ?*Tensor) !*Tensor {
        return self.conv2dWithConfig(A, weight, bias, 1, 0);
    }

    pub fn conv2dWithConfig(
        self: *Graph,
        A: *Tensor,
        weight: *Tensor,
        bias: ?*Tensor,
        stride: usize,
        padding: usize,
    ) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.conv2dWithConfig(weight, bias, stride, padding, allocator);
        const req_grad = self.enable_grad and (A.requires_grad or weight.requires_grad or (bias != null and bias.?.requires_grad));
        const ctx: OpContext = .{ .Conv2D = .{ .stride = stride, .padding = padding } };
        if (bias) |b| {
            return self.registerSingleOutputOp(C, &.{ A, weight, b }, .Conv2D, ctx, req_grad);
        } else {
            return self.registerSingleOutputOp(C, &.{ A, weight }, .Conv2D, ctx, req_grad);
        }
    }

    pub fn convTranspose2D(
        self: *Graph,
        A: *Tensor,
        weight: *Tensor,
        bias: ?*Tensor,
        stride: usize,
        padding: usize,
    ) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.convTranspose2d(weight, bias, stride, padding, allocator);
        const req_grad = self.enable_grad and (A.requires_grad or weight.requires_grad or (bias != null and bias.?.requires_grad));
        const ctx: OpContext = .{ .ConvTranspose2D = .{ .stride = stride, .padding = padding } };
        if (bias) |b| {
            return self.registerSingleOutputOp(C, &.{ A, weight, b }, .ConvTranspose2D, ctx, req_grad);
        } else {
            return self.registerSingleOutputOp(C, &.{ A, weight }, .ConvTranspose2D, ctx, req_grad);
        }
    }

    pub fn maxpool2d(self: *Graph, A: *Tensor, pool_size: usize, stride: usize) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.maxpool2d(pool_size, stride, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .MaxPool2D,
            .{ .MaxPool2D = .{ .pool_size = pool_size, .stride = stride } },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn avgpool2d(self: *Graph, A: *Tensor, kernel_size: usize, stride: usize) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.avgpool2d(kernel_size, stride, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .AvgPool2D,
            .{ .AvgPool2D = .{ .kernel_size = kernel_size, .stride = stride } },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn softmax(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.softmax(allocator);
        return self.registerSingleOutputOp(
            C,
            &.{A},
            .Softmax,
            .{ .Softmax = {} },
            self.enable_grad and A.requires_grad,
        );
    }

    pub fn rmsNorm(self: *Graph, X: *Tensor, G: *Tensor, eps: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const Y = try X.rmsNorm(G, eps, allocator);
        return self.registerSingleOutputOp(
            Y,
            &.{ X, G },
            .RmsNorm,
            .{ .RmsNorm = .{ .eps = eps } },
            self.enable_grad and (X.requires_grad or G.requires_grad),
        );
    }

    pub fn layerNorm(self: *Graph, X: *Tensor, G: *Tensor, B: *Tensor, eps: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const Y = try X.layerNorm(G, B, eps, allocator);
        return self.registerSingleOutputOp(
            Y,
            &.{ X, G, B },
            .LayerNorm,
            .{ .LayerNorm = .{ .eps = eps } },
            self.enable_grad and (X.requires_grad or G.requires_grad or B.requires_grad),
        );
    }

    pub fn batchNorm2d(
        self: *Graph,
        X: *Tensor,
        G: *Tensor,
        B: *Tensor,
        running_mean: *Tensor,
        running_var: *Tensor,
        eps: f32,
        momentum: f32,
        training: bool,
    ) !*Tensor {
        if (X.shape.len != 4) return error.IncompatibleDimensions;
        const N = X.shape.dims[0];
        const C = X.shape.dims[1];
        const H = X.shape.dims[2];
        const W = X.shape.dims[3];
        if (G.data.len != C or B.data.len != C or running_mean.data.len != C or running_var.data.len != C) {
            return error.ShapeMismatch;
        }

        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and (X.requires_grad or G.requires_grad or B.requires_grad);
        const Y = try self.tensorND(&.{ N, C, H, W }, req_grad);

        const save_mean = try allocator.alloc(f32, C);
        const save_inv_std = try allocator.alloc(f32, C);

        const spatial_size = H * W;
        const total_samples_f = @as(f32, @floatFromInt(N * spatial_size));

        for (0..C) |c| {
            var mean_val: f32 = 0.0;
            var var_val: f32 = 0.0;

            if (training) {
                var sum_val: f32 = 0.0;
                for (0..N) |n| {
                    const c_slice = X.data[(n * C + c) * spatial_size .. (n * C + c + 1) * spatial_size];
                    for (c_slice) |val| sum_val += val;
                }
                mean_val = sum_val / total_samples_f;

                var var_sum: f32 = 0.0;
                for (0..N) |n| {
                    const c_slice = X.data[(n * C + c) * spatial_size .. (n * C + c + 1) * spatial_size];
                    for (c_slice) |val| {
                        const diff = val - mean_val;
                        var_sum += diff * diff;
                    }
                }
                var_val = var_sum / total_samples_f;

                running_mean.data[c] = (1.0 - momentum) * running_mean.data[c] + momentum * mean_val;
                running_var.data[c] = (1.0 - momentum) * running_var.data[c] + momentum * var_val;
            } else {
                mean_val = running_mean.data[c];
                var_val = running_var.data[c];
            }

            const inv_std = 1.0 / @sqrt(var_val + eps);
            save_mean[c] = mean_val;
            save_inv_std[c] = inv_std;

            const g_val = G.data[c];
            const b_val = B.data[c];

            for (0..N) |n| {
                const in_slice = X.data[(n * C + c) * spatial_size .. (n * C + c + 1) * spatial_size];
                const out_slice = Y.data[(n * C + c) * spatial_size .. (n * C + c + 1) * spatial_size];
                for (in_slice, out_slice) |val, *o| {
                    o.* = (val - mean_val) * inv_std * g_val + b_val;
                }
            }
        }

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 3);
            inputs[0] = X;
            inputs[1] = G;
            inputs[2] = B;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = Y;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .BatchNorm2d,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .BatchNorm2d = .{
                    .eps = eps,
                    .training = training,
                    .save_mean = save_mean,
                    .save_inv_std = save_inv_std,
                } },
            };
            Y.creator = o;
            try self.recordOp(o);
        }

        return Y;
    }

    pub fn dropout(self: *Graph, X: *Tensor, p: f32, random: std.Random) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and X.requires_grad;
        const Y = try self.tensorND(X.shape.dims[0..X.shape.len], req_grad);

        const mask_scale = try allocator.alloc(f32, X.data.len);
        const scale = 1.0 / (1.0 - p);

        for (X.data, Y.data, mask_scale) |val, *o, *m| {
            if (random.float(f32) < p) {
                m.* = 0.0;
                o.* = 0.0;
            } else {
                m.* = scale;
                o.* = val * scale;
            }
        }

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = X;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = Y;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Dropout,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Dropout = .{ .mask_scale = mask_scale } },
            };
            Y.creator = o;
            try self.recordOp(o);
        }

        return Y;
    }

    pub fn rope(self: *Graph, X: *Tensor, start_pos: usize) !*Tensor {
        return self.ropeOffset(X, start_pos, 0);
    }

    pub fn ropeOffset(self: *Graph, X: *Tensor, start_pos: usize, rotary_offset: usize) !*Tensor {
        const allocator = self.arena.allocator();
        const Y = try X.ropeOffset(start_pos, rotary_offset, allocator);
        return self.registerSingleOutputOp(
            Y,
            &.{X},
            .RoPE,
            .{ .RoPE = .{ .start_pos = start_pos, .rotary_offset = rotary_offset } },
            self.enable_grad and X.requires_grad,
        );
    }

    pub fn batchMatMul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.batchMatMul(B, allocator);
        return self.registerSingleOutputOp(
            C,
            &.{ A, B },
            .BatchMatMul,
            .{ .BatchMatMul = {} },
            self.enable_grad and (A.requires_grad or B.requires_grad),
        );
    }

    pub fn embedding(self: *Graph, W: *Tensor, X: anytype) !*Tensor {
        const allocator = self.arena.allocator();
        const XT = @TypeOf(X);
        const x_tensor: *Tensor = if (XT == *Tensor or XT == *const Tensor)
            @constCast(X)
        else blk: {
            const ptr_info = @typeInfo(XT);
            if (ptr_info == .pointer and ptr_info.pointer.size == .one and
                @typeInfo(ptr_info.pointer.child) == .@"struct" and
                @hasField(ptr_info.pointer.child, "shape"))
            {
                const t = try self.tensorND(X.shape.dims[0..X.shape.len], false);
                const num_elem = X.shape.numel();
                if (X.isContiguous() and X.data.len >= num_elem) {
                    for (0..num_elem) |i| {
                        t.data[i] = tensor_mod.convertScalar(f32, @TypeOf(X.data[0]), X.data[i]);
                    }
                } else {
                    var coord = [_]usize{0} ** 8;
                    const rank = X.shape.len;
                    for (0..num_elem) |i| {
                        var flat: usize = 0;
                        for (0..rank) |d| flat += coord[d] * X.strides.dims[d];
                        t.data[i] = tensor_mod.convertScalar(f32, @TypeOf(X.data[0]), X.data[flat]);
                        var d = rank;
                        while (d > 0) {
                            d -= 1;
                            coord[d] += 1;
                            if (coord[d] < X.shape.dims[d]) break;
                            coord[d] = 0;
                        }
                    }
                }
                break :blk t;
            } else {
                const t = try self.tensorND(&.{X.len}, false);
                for (0..X.len) |i| {
                    t.data[i] = tensor_mod.convertScalar(f32, @TypeOf(X[0]), X[i]);
                }
                break :blk t;
            }
        };

        const Y = try W.embedding(x_tensor, allocator);
        return self.registerSingleOutputOp(
            Y,
            &.{ W, x_tensor },
            .Embedding,
            .{ .Embedding = {} },
            self.enable_grad and W.requires_grad,
        );
    }

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

    /// 在图建立完毕后，智能探查参数节点的下游消费者算子并自动初始化权重
    pub fn initWeights(self: *Graph, random: std.Random) void {
        const init_mod = @import("../nn/init.zig");

        // 搜集图内所有 Ops 的输入参数节点与 registered tensors
        const allocator = self.arena.allocator();
        var visited = std.AutoHashMap(*Tensor, void).init(allocator);
        defer visited.deinit();

        // 1. 扫描图中的 Ops 所有输入
        for (self.ops.items) |op| {
            for (op.inputs) |t| {
                if (visited.contains(t)) continue;
                visited.put(t, {}) catch continue;
                self.initSingleTensor(t, random, init_mod);
            }
        }

        // 2. 扫描显式注册的 tensors
        for (self.tensors.items) |t| {
            if (visited.contains(t)) continue;
            visited.put(t, {}) catch continue;
            self.initSingleTensor(t, random, init_mod);
        }
    }

    fn initSingleTensor(self: *Graph, t: *Tensor, random: std.Random, comptime init_mod: type) void {
        // 只对属于可训练参数（requires_grad=true 且非中间激活运算生成）的节点进行初始化
        // 如果已经被层的 customInit 初始化过，则坚决跳过，绝不覆盖！
        if (!t.requires_grad or t.is_custom_initialized or t.creator != null) return;

        // 1. 如果是 1D 参数向量 (归一化缩放因子 gamma 初始化为 1.0，偏置向量初始化为 0.0)
        if (t.shape.len == 1 or (t.shape.len == 2 and t.shape.dims[0] == 1)) {
            for (self.ops.items) |op| {
                switch (op.op_type) {
                    .RmsNorm, .LayerNorm, .BatchNorm2d => {
                        if (op.inputs.len >= 2 and op.inputs[1] == t) {
                            @memset(t.data, 1.0);
                            return;
                        }
                    },
                    else => {},
                }
            }
            @memset(t.data, 0.0);
            return;
        }

        // 2. 如果是高维权重矩阵 (Linear, Conv2D, ConvTranspose2D 等)
        // 沿计算图中的 Ops 向后探测第一个下游消费算子 (Consumer Op)
        const nonlinearity = self.detectConsumerActivation(t);
        const gain = init_mod.calculateGain(nonlinearity);

        var fan_in: usize = 1;
        var fan_out: usize = 1;
        if (t.shape.len == 2) {
            fan_in = t.shape.dims[0];
            fan_out = t.shape.dims[1];
        } else if (t.shape.len == 4) {
            // Conv2D: [out_channels, in_channels, kh, kw]
            const out_c = t.shape.dims[0];
            const in_c = t.shape.dims[1];
            const kh = t.shape.dims[2];
            const kw = t.shape.dims[3];
            fan_in = in_c * kh * kw;
            fan_out = out_c * kh * kw;
        } else {
            for (t.shape.dims[0 .. t.shape.len - 1]) |d| fan_in *= d;
            fan_out = t.shape.dims[t.shape.len - 1];
        }

        const method: init_mod.InitMethod = switch (nonlinearity) {
            .tanh, .sigmoid => .{ .xavier_normal = .{ .gain = gain } },
            .selu => .lecun_normal,
            else => .{ .he_normal = .{ .gain = gain } },
        };

        init_mod.initWeights(random, t.data, fan_in, fan_out, method);
    }

    /// 格式化全图各节点（包括输入数据、模型参数及中间算子输出）的详情与初始化报告为分配的字符串
    pub fn formatInitReport(self: *Graph, allocator: std.mem.Allocator) ![]const u8 {
        const init_mod = @import("../nn/init.zig");
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "\n=== Graph Architecture & Initialization Report ===\n");
        try buf.print(allocator, "{s:<24} {s:<12} {s:<18} {s:<14} {s:<16} {s:<28}\n", .{
            "Node", "Kind", "Shape", "Status", "Inferred / Op", "Strategy / Details",
        });
        try buf.print(allocator, "{s:-<24} {s:-<12} {s:-<18} {s:-<14} {s:-<16} {s:-<28}\n", .{
            "", "", "", "", "", "",
        });

        const arena_alloc = self.arena.allocator();
        var visited = std.AutoHashMap(*Tensor, void).init(arena_alloc);
        defer visited.deinit();

        var param_idx: usize = 0;
        var input_idx: usize = 0;
        var op_idx: usize = 0;

        for (self.ops.items) |op| {
            for (op.inputs) |t| {
                if (visited.contains(t)) continue;
                visited.put(t, {}) catch continue;
                try self.appendSingleTensorReport(t, &param_idx, &input_idx, &op_idx, &buf, allocator, init_mod);
            }
            for (op.outputs) |t| {
                if (visited.contains(t)) continue;
                visited.put(t, {}) catch continue;
                try self.appendSingleTensorReport(t, &param_idx, &input_idx, &op_idx, &buf, allocator, init_mod);
            }
        }

        for (self.tensors.items) |t| {
            if (visited.contains(t)) continue;
            visited.put(t, {}) catch continue;
            try self.appendSingleTensorReport(t, &param_idx, &input_idx, &op_idx, &buf, allocator, init_mod);
        }
        try buf.appendSlice(allocator, "=========================================================================================================\n\n");
        return buf.toOwnedSlice(allocator);
    }

    /// 在标准输出/调试控制台直接打印初始化详情报告
    pub fn printInitReport(self: *Graph) void {
        const report = self.formatInitReport(self.backing_allocator) catch return;
        defer self.backing_allocator.free(report);
        std.debug.print("{s}", .{report});
    }

    /// 将计算图与模块层级结构序列化为递归的 JSON 数据字符串 (供前端直接解析并构建完整模型拓扑)
    pub fn formatJson(self: *Graph, allocator: std.mem.Allocator) ![]const u8 {
        const vis = @import("../nn/visualization.zig");
        return vis.graph_ir.generateJson(self, allocator);
    }

    /// 将计算图与模块层级结构直接导出保存为独立的 JSON 文件 (如 "model_graph.json")
    pub fn exportJson(self: *Graph, file_path: []const u8) !void {
        const vis = @import("../nn/visualization.zig");
        try vis.graph_ir.exportJson(self, file_path, self.backing_allocator);
    }

    fn appendSingleTensorReport(
        self: *Graph,
        t: *Tensor,
        param_idx: *usize,
        input_idx: *usize,
        op_idx: *usize,
        buf: *std.ArrayList(u8),
        allocator: std.mem.Allocator,
        comptime init_mod: type,
    ) !void {
        var shape_buf: [64]u8 = undefined;
        var shape_len: usize = 0;
        shape_buf[0] = '[';
        shape_len += 1;
        for (0..t.shape.len) |d| {
            if (d > 0) {
                shape_buf[shape_len] = ',';
                shape_buf[shape_len + 1] = ' ';
                shape_len += 2;
            }
            const part = std.fmt.bufPrint(shape_buf[shape_len..], "{d}", .{t.shape.dims[d]}) catch "";
            shape_len += part.len;
        }
        shape_buf[shape_len] = ']';
        shape_len += 1;
        const shape_str = shape_buf[0..shape_len];

        // 1. 算子生成的中间计算节点 / 激活输出
        if (t.creator) |creator_op| {
            var name_buf: [48]u8 = undefined;
            const op_name = @tagName(creator_op.op_type);
            const name: []const u8 = if (t.name) |n| n else (std.fmt.bufPrint(&name_buf, "Node_{s}_{d}", .{ op_name, op_idx.* }) catch "Node_Op");
            op_idx.* += 1;

            var detail_buf: [64]u8 = undefined;
            const detail = std.fmt.bufPrint(&detail_buf, "produced by {s}", .{op_name}) catch "op output";

            try buf.print(allocator, "{s:<24} {s:<12} {s:<18} {s:<14} {s:<16} {s:<28}\n", .{
                name, "Activation", shape_str, "OP_OUTPUT", op_name, detail,
            });
            return;
        }

        // 2. 外部输入或常量张量 (非可学习参数)
        if (!t.requires_grad) {
            var name_buf: [48]u8 = undefined;
            const name: []const u8 = if (t.name) |n| n else (std.fmt.bufPrint(&name_buf, "Input_{d}", .{input_idx.*}) catch "Input");
            input_idx.* += 1;

            try buf.print(allocator, "{s:<24} {s:<12} {s:<18} {s:<14} {s:<16} {s:<28}\n", .{
                name, "Input", shape_str, "INPUT", "N/A", "user input / constant",
            });
            return;
        }

        // 3. 模型可学习参数节点
        var name_buf: [48]u8 = undefined;
        const name: []const u8 = if (t.name) |n| n else (std.fmt.bufPrint(&name_buf, "Param_{d}", .{param_idx.*}) catch "Param");
        param_idx.* += 1;

        if (t.is_custom_initialized) {
            try buf.print(allocator, "{s:<24} {s:<12} {s:<18} {s:<14} {s:<16} {s:<28}\n", .{
                name, "Param", shape_str, "CUSTOM_INIT", "N/A", "user-defined customInit",
            });
            return;
        }

        if (t.shape.len == 1 or (t.shape.len == 2 and t.shape.dims[0] == 1)) {
            try buf.print(allocator, "{s:<24} {s:<12} {s:<18} {s:<14} {s:<16} {s:<28}\n", .{
                name, "Param", shape_str, "AUTO_GRAPH", "bias", "zeros (0.0)",
            });
            return;
        }

        const act = self.detectConsumerActivation(t);
        const gain = init_mod.calculateGain(act);
        const act_name = switch (act) {
            .relu => "ReLU",
            .tanh => "Tanh",
            .sigmoid => "Sigmoid",
            .gelu => "GELU",
            .silu => "SiLU",
            .selu => "SELU",
            .leaky_relu => "LeakyReLU",
            .linear => "Linear (None)",
        };

        var strat_buf: [64]u8 = undefined;
        const strat = switch (act) {
            .tanh, .sigmoid => std.fmt.bufPrint(&strat_buf, "Xavier Normal (gain={d:.3})", .{gain}) catch "Xavier Normal",
            .selu => "LeCun Normal",
            else => std.fmt.bufPrint(&strat_buf, "He Normal (gain={d:.3})", .{gain}) catch "He Normal",
        };

        try buf.print(allocator, "{s:<24} {s:<12} {s:<18} {s:<14} {s:<16} {s:<28}\n", .{
            name, "Param", shape_str, "AUTO_GRAPH", act_name, strat,
        });
    }

    /// 顺着张量 t 往后在图的 Ops 列表中探查下游消费者的激活函数类型
    pub fn detectConsumerActivation(self: *Graph, target: *Tensor) @import("../nn/init.zig").Nonlinearity {
        var current: *Tensor = target;

        // BFS / DFS 往后搜寻直到遇到激活函数或多层终点
        while (true) {
            var found_consumer = false;
            for (self.ops.items) |op| {
                for (op.inputs) |inp| {
                    if (inp == current) {
                        found_consumer = true;
                        switch (op.op_type) {
                            .Relu => return .relu,
                            .Tanh => return .tanh,
                            .Sigmoid => return .sigmoid,
                            .Gelu => return .gelu,
                            .Silu => return .silu,
                            .LeakyRelu => return .{ .leaky_relu = op.context.LeakyRelu.alpha },
                            // 如果经过了 MatMul/Conv2D/AddBias/Add 等中间运算，顺着它的 output 继续往后看
                            .MatMul, .BatchMatMul, .Conv2D, .ConvTranspose2D, .AddBias, .Add, .Reshape, .Transpose => {
                                if (op.outputs.len > 0) {
                                    current = op.outputs[0];
                                    break;
                                }
                            },
                            else => {},
                        }
                    }
                }
                if (found_consumer and current != target) break;
            }
            if (!found_consumer or current == target) break;
        }

        // 如果下游没有接激活函数或直接进入输出/损失层，判定为 linear (gain=1.0)
        return .linear;
    }
};

