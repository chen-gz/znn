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

    // 初始化计算图，传入底层通用内存分配器
    pub fn init(backing_allocator: std.mem.Allocator) Graph {
        return Graph{
            .backing_allocator = backing_allocator,
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .tensors = .empty,
            .ops = .empty,
            .enable_grad = true,
            .module_formulas = std.StringHashMap([]const u8).init(backing_allocator),
        };
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

    // 设置梯度追踪开关
    pub fn setGradEnabled(self: *Graph, enabled: bool) void {
        self.enable_grad = enabled;
    }

    // 释放整个计算图的内存（包括所有张量与算子节点的前向/反向缓冲区）
    pub fn deinit(self: *Graph) void {
        self.module_formulas.deinit();
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

        var old_total: usize = 1;
        for (0..A.shape.len) |i| {
            old_total *= A.shape.dims[i];
        }
        var new_total: usize = 1;
        for (new_shape_slice) |dim| {
            new_total *= dim;
        }
        if (old_total != new_total) return error.ShapeMismatch;


        const req_grad = self.enable_grad and A.requires_grad;

        C.* = Tensor{
            .data = A.data, // 共享前向数据
            .grad = if (req_grad) try allocator.alloc(f32, new_total) else &.{},
            .shape = shape,
            .strides = strides,
            .requires_grad = req_grad,
            .creator = null,
        };
        if (req_grad) {
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Reshape,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Reshape = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 维度转置算子前向传播：交换 dim0 和 dim1
    pub fn transposeND(self: *Graph, A: *Tensor, dim0: usize, dim1: usize) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.transpose(dim0, dim1, allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Transpose,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{
                    .Transpose = .{
                        .dim0 = dim0,
                        .dim1 = dim1,
                    },
                },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 沿指定维度拼接张量数组 (Concat)
    pub fn concat(self: *Graph, inputs: []const *Tensor, dim: usize) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try tensor_mod.concat(allocator, inputs, dim, null);

        var req_grad = false;
        if (self.enable_grad) {
            for (inputs) |inp| {
                if (inp.requires_grad) {
                    req_grad = true;
                    break;
                }
            }
        }
        C.requires_grad = req_grad;
        if (req_grad) {
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
                .op_type = .Concat,
                .inputs = inps_copy,
                .outputs = outs_copy,
                .context = .{
                    .Concat = .{
                        .dim = dim,
                    },
                },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 沿指定维度将张量均等切分为 num_splits 份 (Split)
    pub fn split(self: *Graph, input: *Tensor, num_splits: usize, dim: usize) ![]*Tensor {
        const allocator = self.arena.allocator();
        const outputs = try tensor_mod.split(allocator, input, num_splits, dim, null);

        const req_grad = self.enable_grad and input.requires_grad;
        for (outputs) |out| {
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
            try self.ops.append(self.backing_allocator, o);
        }

        return outputs;
    }

    // GQA 注意力中沿 Head 维度复制广播 Key / Value 张量 (RepeatKV)
    // 输入 X: [B, num_kv_heads, T, hs]
    // 输出 Y: [B, num_kv_heads * groups, T, hs]
    pub fn repeatKV(self: *Graph, X: *Tensor, groups: usize) !*Tensor {
        if (groups == 1) return X;

        const allocator = self.arena.allocator();
        const B = X.shape.dims[0];
        const n_kv = X.shape.dims[1];
        const T = X.shape.dims[2];
        const hs = X.shape.dims[3];
        const nh = n_kv * groups;

        const req_grad = self.enable_grad and X.requires_grad;
        const Y = try self.tensorND(&.{ B, nh, T, hs }, req_grad);

        const head_bytes = T * hs;
        for (0..B) |b| {
            for (0..n_kv) |kv_h| {
                const src = X.data[((b * n_kv + kv_h) * head_bytes) .. ((b * n_kv + kv_h + 1) * head_bytes)];
                for (0..groups) |g| {
                    const h = kv_h * groups + g;
                    const dest = Y.data[((b * nh + h) * head_bytes) .. ((b * nh + h + 1) * head_bytes)];
                    @memcpy(dest, src);
                }
            }
        }

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = X;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = Y;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .RepeatKV,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{
                    .RepeatKV = .{
                        .groups = groups,
                    },
                },
            };
            Y.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return Y;
    }

    // 矩阵乘法算子前向传播：C = A * B
    pub fn matmul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.matmul(B, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = A;
            inputs[1] = B;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .MatMul,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .MatMul = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 偏置相加算子前向传播：C = A + bias (直接路由到通用广播加法)
    pub fn addBias(self: *Graph, A: *Tensor, bias: *Tensor) !*Tensor {
        return self.add(A, bias);
    }

    // 激活函数 ReLU 前向传播：C = max(0, A)
    pub fn relu(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.relu(allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Relu,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Relu = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn gelu(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.gelu(allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Gelu,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Gelu = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn sigmoid(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.sigmoid(allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Sigmoid,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Sigmoid = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn tanh(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.tanh(allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Tanh,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Tanh = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn leakyRelu(self: *Graph, A: *Tensor, alpha: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.leakyRelu(alpha, allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .LeakyRelu,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .LeakyRelu = .{ .alpha = alpha } },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 激活函数 SiLU (Swish) 前向传播：C = A * sigmoid(A)
    pub fn silu(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.silu(allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Silu,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Silu = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 损失函数 Softmax + Cross Entropy 结合前向传播
    // 在 logits 的行维度计算 Softmax 概率分布，并与 targets 分类标签计算交叉熵损失
    pub fn softmaxCrossEntropy(self: *Graph, logits: *Tensor, targets: []const u8) !*Tensor {
        const req_grad = self.enable_grad and logits.requires_grad;
        const loss = try self.tensor(1, 1, req_grad);

        const B = logits.shape.dims[0];
        const N = logits.shape.dims[1];
        const allocator = self.arena.allocator();

        var loss_sum: f32 = 0.0;
        if (req_grad) {
            const probs = try allocator.alloc(f32, B * N);

            // 1. 对每一行计算 Softmax 概率（数值稳定的减去 max 技巧）
            for (0..B) |i| {
                const logits_row = logits.data[i * N .. (i + 1) * N];
                const probs_row = probs[i * N .. (i + 1) * N];

                // 寻找当前行的最大值，避免 @exp() 产生数值上溢（NaN）
                var max_val = logits_row[0];
                for (logits_row[1..]) |val| {
                    if (val > max_val) max_val = val;
                }

                var sum: f32 = 0.0;
                for (logits_row, probs_row) |val, *p| {
                    const exp_val = @exp(val - max_val);
                    p.* = exp_val;
                    sum += exp_val;
                }

                // 归一化为概率分布
                for (probs_row) |*p| {
                    p.* /= sum;
                }
            }

            // 2. 计算平均交叉熵损失值：L = -1/B * sum(log(prob_target))
            for (0..B) |i| {
                const label = targets[i];
                const prob = probs[i * N + label];
                const clipped = @max(prob, 1e-15); // 微小值剪裁，避免 log(0) 产生 -inf
                loss_sum += -@log(clipped);
            }
            loss.data[0] = loss_sum / @as(f32, @floatFromInt(B));

            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = logits;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = loss;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .SoftmaxCrossEntropy,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{
                    .SoftmaxCrossEntropy = .{
                        .probs = probs,
                        .targets = targets,
                    },
                },
            };
            loss.creator = o;
            try self.ops.append(self.backing_allocator, o);
        } else {
            for (0..B) |i| {
                const logits_row = logits.data[i * N .. (i + 1) * N];
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
        }

        return loss;
    }

    // 均方误差 (MSE) 损失函数：C = 1/N * sum((y_pred - y_true)^2)
    pub fn mseLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and (y_pred.requires_grad or y_true.requires_grad);
        const loss = try self.tensor(1, 1, req_grad);

        const N = y_pred.data.len;
        std.debug.assert(N == y_true.data.len);

        var loss_sum: f32 = 0.0;
        for (0..N) |i| {
            const diff = y_pred.data[i] - y_true.data[i];
            loss_sum += diff * diff;
        }
        loss.data[0] = loss_sum / @as(f32, @floatFromInt(N));

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = y_pred;
            inputs[1] = y_true;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = loss;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .MseLoss,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .MseLoss = {} },
            };
            loss.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return loss;
    }

    pub fn bceWithLogitsLoss(self: *Graph, logits: *Tensor, targets: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and (logits.requires_grad or targets.requires_grad);
        const loss = try self.tensor(1, 1, req_grad);

        const N = logits.data.len;
        std.debug.assert(N == targets.data.len);

        var loss_sum: f32 = 0.0;
        for (0..N) |i| {
            const x = logits.data[i];
            const y = targets.data[i];
            const max_x = @max(x, 0.0);
            const abs_x = @abs(x);
            const l = max_x - x * y + @log(1.0 + @exp(-abs_x));
            loss_sum += l;
        }
        loss.data[0] = loss_sum / @as(f32, @floatFromInt(N));

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = logits;
            inputs[1] = targets;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = loss;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .BceWithLogitsLoss,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .BceWithLogitsLoss = {} },
            };
            loss.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return loss;
    }

    pub fn sigmoidCrossEntropy(self: *Graph, logits: *Tensor, targets: *Tensor) !*Tensor {
        return self.bceWithLogitsLoss(logits, targets);
    }

    pub fn bceLoss(self: *Graph, probs: *Tensor, targets: *Tensor, eps: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and (probs.requires_grad or targets.requires_grad);
        const loss = try self.tensor(1, 1, req_grad);

        const N = probs.data.len;
        std.debug.assert(N == targets.data.len);

        var loss_sum: f32 = 0.0;
        for (0..N) |i| {
            const p = probs.data[i];
            const y = targets.data[i];
            const p_clip = @max(p, eps);
            const one_minus_p_clip = @max(1.0 - p, eps);
            const l = -(y * @log(p_clip) + (1.0 - y) * @log(one_minus_p_clip));
            loss_sum += l;
        }
        loss.data[0] = loss_sum / @as(f32, @floatFromInt(N));

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = probs;
            inputs[1] = targets;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = loss;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .BceLoss,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .BceLoss = .{ .eps = eps } },
            };
            loss.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return loss;
    }

    pub fn randomNormal(self: *Graph, shape_slice: []const usize, random: std.Random, mean: f32, stddev: f32, requires_grad: bool) !*Tensor {
        const t = try self.tensorND(shape_slice, requires_grad);
        t.fillNormal(random, mean, stddev);
        return t;
    }

    pub fn randomUniform(self: *Graph, shape_slice: []const usize, random: std.Random, min: f32, max: f32, requires_grad: bool) !*Tensor {
        const t = try self.tensorND(shape_slice, requires_grad);
        t.fillUniform(random, min, max);
        return t;
    }

    // L2 正则化损失函数：C = 0.5 * lambda * sum(weight_i^2)
    pub fn l2Loss(self: *Graph, weight: *Tensor, lambda: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and weight.requires_grad;
        const loss = try self.tensor(1, 1, req_grad);

        var sum_sq: f32 = 0.0;
        for (weight.data) |v| {
            sum_sq += v * v;
        }
        loss.data[0] = 0.5 * lambda * sum_sq;

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = weight;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = loss;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .L2Loss,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .L2Loss = .{ .lambda = lambda } },
            };
            loss.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return loss;
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
        const allocator = self.arena.allocator();
        const req_grad = self.enable_grad and weight.requires_grad;
        const loss = try self.tensor(1, 1, req_grad);

        var sum_abs: f32 = 0.0;
        for (weight.data) |v| {
            sum_abs += @abs(v);
        }
        loss.data[0] = lambda * sum_abs;

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = weight;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = loss;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .L1Loss,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .L1Loss = .{ .lambda = lambda } },
            };
            loss.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return loss;
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
        const C = try A.mulScalar(val, allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .MulScalar,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .MulScalar = .{ .val = val } },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 标量加法：C = A + val
    pub fn addScalar(self: *Graph, A: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.addScalar(val, allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .AddScalar,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .AddScalar = .{ .val = val } },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 标量减法：C = A - val
    pub fn subScalar(self: *Graph, A: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.subScalar(val, allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .SubScalar,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .SubScalar = .{ .val = val } },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 标量除法：C = A / val
    pub fn divScalar(self: *Graph, A: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.divScalar(val, allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .DivScalar,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .DivScalar = .{ .val = val } },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 逐元素张量加法：C = A + B (支持多维广播)
    pub fn add(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.add(B, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = A;
            inputs[1] = B;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Add,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Add = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 逐元素张量减法：C = A - B (支持多维广播)
    pub fn sub(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.sub(B, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = A;
            inputs[1] = B;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Sub,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Sub = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 逐元素张量乘法 (Hadamard 积)：C = A * B (支持多维广播)
    pub fn mul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.mul(B, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = A;
            inputs[1] = B;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Mul,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Mul = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    // 逐元素张量除法：C = A / B (支持多维广播)
    pub fn div(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.div(B, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = A;
            inputs[1] = B;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Div,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Div = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn conv2d(self: *Graph, A: *Tensor, weight: *Tensor, bias: ?*Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.conv2d(weight, bias, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or weight.requires_grad or (bias != null and bias.?.requires_grad));
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const num_inputs: usize = if (bias != null) 3 else 2;
            const inputs = try allocator.alloc(*Tensor, num_inputs);
            inputs[0] = A;
            inputs[1] = weight;
            if (bias) |b| {
                inputs[2] = b;
            }
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Conv2D,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Conv2D = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
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
        const C = try A.convTranspose2d(weight, bias, stride, padding, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or weight.requires_grad or (bias != null and bias.?.requires_grad));
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const num_inputs: usize = if (bias != null) 3 else 2;
            const inputs = try allocator.alloc(*Tensor, num_inputs);
            inputs[0] = A;
            inputs[1] = weight;
            if (bias) |b| {
                inputs[2] = b;
            }
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .ConvTranspose2D,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .ConvTranspose2D = .{ .stride = stride, .padding = padding } },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn maxpool2d(self: *Graph, A: *Tensor, pool_size: usize, stride: usize) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.maxpool2d(pool_size, stride, allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .MaxPool2D,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .MaxPool2D = .{ .pool_size = pool_size, .stride = stride } },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn softmax(self: *Graph, A: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.softmax(allocator, null);

        const req_grad = self.enable_grad and A.requires_grad;
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 1);
            inputs[0] = A;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Softmax,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Softmax = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn rmsNorm(self: *Graph, X: *Tensor, G: *Tensor, eps: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const Y = try X.rmsNorm(G, eps, allocator, null);

        const req_grad = self.enable_grad and (X.requires_grad or G.requires_grad);
        Y.requires_grad = req_grad;
        if (req_grad) {
            Y.grad = try allocator.alloc(f32, Y.data.len);
            @memset(Y.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, Y);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = X;
            inputs[1] = G;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = Y;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .RmsNorm,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .RmsNorm = .{ .eps = eps } },
            };
            Y.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return Y;
    }

    pub fn batchMatMul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.batchMatMul(B, allocator, null);

        const req_grad = self.enable_grad and (A.requires_grad or B.requires_grad);
        C.requires_grad = req_grad;
        if (req_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = A;
            inputs[1] = B;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = C;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .BatchMatMul,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .BatchMatMul = {} },
            };
            C.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return C;
    }

    pub fn embedding(self: *Graph, W: *Tensor, X: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const Y = try W.embedding(X, allocator, null);

        const req_grad = self.enable_grad and W.requires_grad;
        Y.requires_grad = req_grad;
        if (req_grad) {
            Y.grad = try allocator.alloc(f32, Y.data.len);
            @memset(Y.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, Y);

        if (req_grad) {
            const inputs = try allocator.alloc(*Tensor, 2);
            inputs[0] = W;
            inputs[1] = X;
            const outputs = try allocator.alloc(*Tensor, 1);
            outputs[0] = Y;

            const o = try allocator.create(Op);
            o.* = Op{
                .op_type = .Embedding,
                .inputs = inputs,
                .outputs = outputs,
                .context = .{ .Embedding = {} },
            };
            Y.creator = o;
            try self.ops.append(self.backing_allocator, o);
        }

        return Y;
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

        // 1. 如果是 1D 偏置向量 (Shape 类似 [out_features] 或 [1, out_features] 且为加法偏置)
        if (t.shape.len == 1 or (t.shape.len == 2 and t.shape.dims[0] == 1)) {
            // 默认置零偏置
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

    /// 将计算图结构与各层初始化详情格式化为可交互、层级展开的 HTML 网页文档
    pub fn formatHtmlReport(self: *Graph, allocator: std.mem.Allocator) ![]const u8 {
        const vis = @import("../nn/visualization.zig");
        return vis.generateHtmlReport(self, allocator);
    }

    /// 将计算图结构与各层初始化详情输出并保存为独立的 HTML 报告文件 (如 "report.html")
    pub fn exportHtmlReport(self: *Graph, file_path: []const u8) !void {
        const vis = @import("../nn/visualization.zig");
        try vis.exportHtmlReport(self, file_path, self.backing_allocator);
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
                            // 如果经过了 MatMul/AddBias/Add 等中间运算，顺着它的 output 继续往后看
                            .MatMul, .BatchMatMul, .AddBias, .Add, .Reshape, .Transpose => {
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

