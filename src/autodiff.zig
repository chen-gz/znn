const std = @import("std");
const c = @import("cblas.zig");


// 张量（Tensor）结构体：自动微分引擎的核心数据单元
// 封装了前向传播数据数据流和反向传播的梯度数据流
pub const Tensor = struct {
    data: []f32,          // 前向传播的数据缓冲区（行优先存储的一维切片）
    grad: []f32,          // 反向传播的梯度缓冲区（与 data 形状一致，不需梯度的节点可为空）
    rows: usize,          // 矩阵的行数
    cols: usize,          // 矩阵的列数
    requires_grad: bool,  // 是否需要求梯度（如模型参数为 true，输入数据为 false）
    creator: ?*Op,        // 产生此张量的算子节点（前向图中的父节点，用于追踪计算路径）

    // 将梯度缓冲区全部清零，通常在每个 batch 反向传播前调用
    pub fn zeroGrad(self: *Tensor) void {
        if (self.requires_grad) {
            @memset(self.grad, 0.0);
        }
    }
};

// 支持的算子类型枚举
pub const OpType = enum {
    MatMul,              // 矩阵乘法
    AddBias,             // 偏置项加法（广播机制）
    Relu,                // 激活函数 ReLU
    SoftmaxCrossEntropy, // 损失函数：结合了 Softmax 与交叉熵（数值稳定性更好）
};

// 各算子反向传播所需的上下文信息（如 Softmax 的概率输出与 Target 类别）
pub const OpContext = union(enum) {
    MatMul: void,
    AddBias: void,
    Relu: void,
    SoftmaxCrossEntropy: struct {
        probs: []f32,
        targets: []const u8,
    },
};




// 计算图中的算子节点（Op）结构体
// 存储算子的操作类型、输入输出张量指针，并定义了如何对该操作执行求导（backward）
pub const Op = struct {
    op_type: OpType,        // 算子类别（如 MatMul, Relu）
    inputs: []*Tensor,      // 输入张量数组
    outputs: []*Tensor,     // 输出张量数组
    context: OpContext,     // 算子特有的运行时上下文数据

    // 执行该算子的反向传播计算，更新其输入节点的梯度
    pub fn backward(self: *Op) !void {
        switch (self.op_type) {
            .MatMul => {
                const A = self.inputs[0]; // 形状为 M x K
                const B = self.inputs[1]; // 形状为 K x N
                const C = self.outputs[0]; // 形状为 M x N
                const M = A.rows;
                const K = A.cols;
                const N = B.cols;

                // 1. 计算对左乘矩阵 A 的梯度: dA += dC * B^T
                // 数学原理: d(A * B)/dA = dC * B^T，形状为 (M x N) * (N x K) -> M x K
                if (A.requires_grad) {
                    // 使用 CPU Apple Accelerate (AMX) sgemm 矩阵乘法
                    c.cblas_sgemm(
                        c.CblasRowMajor,
                        c.CblasNoTrans, // C.grad 不转置
                        c.CblasTrans,   // B.data 需转置为 B^T
                        @intCast(M),
                        @intCast(K),
                        @intCast(N),
                        1.0,            // alpha = 1.0
                        C.grad.ptr,
                        @intCast(N),
                        B.data.ptr,
                        @intCast(N),
                        1.0,            // beta = 1.0 表示累加到 A.grad，不覆盖已有值
                        A.grad.ptr,
                        @intCast(K),
                    );
                }

                // 2. 计算对右乘矩阵 B 的梯度: dB += A^T * dC
                // 数学原理: d(A * B)/dB = A^T * dC，形状为 (K x M) * (M x N) -> K x N
                if (B.requires_grad) {
                    c.cblas_sgemm(
                        c.CblasRowMajor,
                        c.CblasTrans,   // A.data 需转置为 A^T
                        c.CblasNoTrans, // C.grad 不转置
                        @intCast(K),
                        @intCast(N),
                        @intCast(M),
                        1.0,            // alpha = 1.0
                        A.data.ptr,
                        @intCast(K),
                        C.grad.ptr,
                        @intCast(N),
                        1.0,            // beta = 1.0 同样进行累加
                        B.grad.ptr,
                        @intCast(N),
                    );
                }
            },
            .AddBias => {
                const A = self.inputs[0];
                const bias = self.inputs[1];
                const C = self.outputs[0];
                const M = A.rows;
                const N = A.cols;

                // AddBias 反向传播：
                // 1. 关于输入 A 的梯度为 dC，按元素累加到 A.grad
                // 2. 关于偏置 bias (1 x N) 的梯度为 dC 按行累加（降维累加）：
                //    bias_grad[j] = sum_{i=0..M-1} dC[i, j]
                // 保持单线程处理，避免多线程调度和同步造成的开销
                for (0..N) |n| {
                    var bias_sum: f32 = 0.0;
                    for (0..M) |m| {
                        const grad_val = C.grad[m * N + n];
                        if (A.requires_grad) {
                            A.grad[m * N + n] += grad_val;
                        }
                        bias_sum += grad_val;
                    }
                    if (bias.requires_grad) {
                        bias.grad[n] += bias_sum;
                    }
                }
            },
            .Relu => {
                const A = self.inputs[0];
                const C = self.outputs[0];

                // ReLU 反向传播：
                // 如果前向值 A.data[i] > 0，则梯度原样传递：dA[i] += dC[i]
                // 如果前向值 A.data[i] <= 0，则梯度置为 0
                if (A.requires_grad) {
                    const total = A.data.len;
                    for (0..total) |i| {
                        A.grad[i] += if (A.data[i] > 0.0) C.grad[i] else 0.0;
                    }
                }
            },
            .SoftmaxCrossEntropy => {
                const logits = self.inputs[0];
                const M = logits.rows;
                const N = logits.cols;
                const ctx = &self.context.SoftmaxCrossEntropy;

                // 平均梯度缩放因子 (1 / batch_size)
                const scale = 1.0 / @as(f32, @floatFromInt(M));

                // SoftmaxCrossEntropy 反向传播：
                // 针对输入 Logits 的梯度公式：dLogits[i, j] = (probs[i, j] - target_indicator) / batch_size
                // 其中 target_indicator 在 j == target 时为 1.0，否则为 0.0
                for (0..M) |i| {
                    const label = ctx.targets[i];
                    const p_row = ctx.probs[i * N .. (i + 1) * N];
                    const dLogits_row = logits.grad[i * N .. (i + 1) * N];
                    for (0..N) |j| {
                        dLogits_row[j] += scale * (p_row[j] - (if (j == label) @as(f32, 1.0) else 0.0));
                    }
                }
            },
        }
    }
};

// 计算图（Graph）结构体
// 追踪所有的张量节点与算子节点，管理内存生命周期并负责反向传播调度
pub const Graph = struct {
    backing_allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,       // 使用 Arena 机制，使每次前向/反向生成的中间节点内存可在 batch 结束时一并释放，避免内存碎片和频繁分配
    tensors: std.ArrayList(*Tensor),     // 追踪计算图中的所有张量指针
    ops: std.ArrayList(*Op),             // 追踪计算图中的所有算子指针

    // 初始化计算图，传入底层通用内存分配器
    pub fn init(backing_allocator: std.mem.Allocator) Graph {
        return Graph{
            .backing_allocator = backing_allocator,
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .tensors = .empty,
            .ops = .empty,
        };
    }

    // 释放整个计算图的内存（包括所有张量与算子节点的前向/反向缓冲区）
    pub fn deinit(self: *Graph) void {
        self.tensors.deinit(self.backing_allocator);
        self.ops.deinit(self.backing_allocator);
        self.arena.deinit();
    }

    // 在计算图中创建并注册一个新的张量节点
    pub fn tensor(self: *Graph, rows: usize, cols: usize, requires_grad: bool) !*Tensor {
        const allocator = self.arena.allocator();
        const t = try allocator.create(Tensor);
        t.* = Tensor{
            .data = try allocator.alloc(f32, rows * cols),
            .grad = if (requires_grad) try allocator.alloc(f32, rows * cols) else &.{},
            .rows = rows,
            .cols = cols,
            .requires_grad = requires_grad,
            .creator = null,
        };
        @memset(t.data, 0.0);
        if (requires_grad) {
            @memset(t.grad, 0.0);
        }
        try self.tensors.append(self.backing_allocator, t);
        return t;
    }

    // 矩阵乘法算子前向传播：C = A * B
    pub fn matmul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        // 创建输出张量 C。若 A 或 B 任意一个需要梯度，则 C 也需要梯度以完成链式反向传播
        const C = try self.tensor(A.rows, B.cols, A.requires_grad or B.requires_grad);

        const M = A.rows;
        const K = A.cols;
        const N = B.cols;

        // 调用 CPU macOS Accelerate CBLAS 计算
        c.cblas_sgemm(
            c.CblasRowMajor,
            c.CblasNoTrans, // A 不转置
            c.CblasNoTrans, // B 不转置
            @intCast(M),
            @intCast(N),
            @intCast(K),
            1.0,            // alpha = 1.0
            A.data.ptr,
            @intCast(K),
            B.data.ptr,
            @intCast(N),
            0.0,            // beta = 0.0（不覆盖/不累加）
            C.data.ptr,
            @intCast(N),
        );

        // 构建并保存当前算子节点以构成计算图的依赖链
        const allocator = self.arena.allocator();
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

        return C;
    }

    // 偏置相加算子前向传播：C = A + bias (按行广播相加)
    pub fn addBias(self: *Graph, A: *Tensor, bias: *Tensor) !*Tensor {
        const C = try self.tensor(A.rows, A.cols, A.requires_grad or bias.requires_grad);

        const M = A.rows;
        const N = A.cols;

        // 偏置向量形状为 1 x N，按行广播复制相加到每一行
        for (0..M) |i| {
            const a_row = A.data[i * N .. (i + 1) * N];
            const c_row = C.data[i * N .. (i + 1) * N];
            for (0..N) |j| {
                c_row[j] = a_row[j] + bias.data[j];
            }
        }

        const allocator = self.arena.allocator();
        const inputs = try allocator.alloc(*Tensor, 2);
        inputs[0] = A;
        inputs[1] = bias;
        const outputs = try allocator.alloc(*Tensor, 1);
        outputs[0] = C;

        const o = try allocator.create(Op);
        o.* = Op{
            .op_type = .AddBias,
            .inputs = inputs,
            .outputs = outputs,
            .context = .{ .AddBias = {} },
        };
        C.creator = o;
        try self.ops.append(self.backing_allocator, o);

        return C;
    }

    // 激活函数 ReLU 前向传播：C = max(0, A)
    pub fn relu(self: *Graph, A: *Tensor) !*Tensor {
        const C = try self.tensor(A.rows, A.cols, A.requires_grad);

        const total = A.data.len;
        for (0..total) |i| {
            C.data[i] = if (A.data[i] > 0.0) A.data[i] else 0.0;
        }

        const allocator = self.arena.allocator();
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

        return C;
    }

    // 损失函数 Softmax + Cross Entropy 结合前向传播
    // 在 logits 的行维度计算 Softmax 概率分布，并与 targets 分类标签计算交叉熵损失
    pub fn softmaxCrossEntropy(self: *Graph, logits: *Tensor, targets: []const u8) !*Tensor {
        const loss = try self.tensor(1, 1, logits.requires_grad);

        const B = logits.rows;
        const N = logits.cols;
        const allocator = self.arena.allocator();
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
        var loss_sum: f32 = 0.0;
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

        return loss;
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
};
