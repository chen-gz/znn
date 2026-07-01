const std = @import("std");
const c = @import("cblas.zig");
const tensor = @import("tensor.zig");

pub const Tensor = tensor.Tensor;
pub const Shape = tensor.Shape;
pub const computeContiguousStrides = tensor.computeContiguousStrides;
pub const transposeShape = tensor.transposeShape;


// 支持的算子类型枚举
pub const OpType = enum {
    MatMul,              // 矩阵乘法
    AddBias,             // 偏置项加法（广播机制）
    Relu,                // 激活函数 ReLU
    SoftmaxCrossEntropy, // 损失函数：结合了 Softmax 与交叉熵（数值稳定性更好）
    Reshape,             // 形状变换
    Transpose,           // 维度转置
    MseLoss,             // 均方误差损失函数
    MulScalar,           // 标量乘法（张量缩放）
    AddScalar,           // 标量加法
    Add,                 // 张量逐元素加法
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
    Reshape: void,
    Transpose: struct {
        dim0: usize,
        dim1: usize,
    },
    MseLoss: void,
    MulScalar: struct {
        val: f32,
    },
    AddScalar: struct {
        val: f32,
    },
    Add: void,
};




// 计算图中的算子节点（Op）结构体
// 存储算子的操作类型、输入输出张量指针，并定义了如何对该操作执行求导（backward）
pub const Op = struct {
    op_type: OpType,        // 算子类别（如 MatMul, Relu）
    inputs: []*Tensor,      // 输入张量数组
    outputs: []*Tensor,     // 输出张量数组
    context: OpContext,     // 算子特有的运行时上下文数据

    // 重新执行该算子的前向计算，根据最新输入更新输出张量的数据
    pub fn forward(self: *Op, allocator: std.mem.Allocator) !void {
        switch (self.op_type) {
            .MatMul => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];
                const temp = try A.matmul(B, allocator, null);
                defer tensor.free(allocator, temp);
                @memcpy(C.data, temp.data);
            },
            .AddBias => {
                const A = self.inputs[0];
                const bias = self.inputs[1];
                const C = self.outputs[0];
                const temp = try A.addBias(bias, allocator, null);
                defer tensor.free(allocator, temp);
                @memcpy(C.data, temp.data);
            },
            .Relu => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                for (C.data, A.data) |*c_val, a_val| {
                    c_val.* = if (a_val > 0.0) a_val else 0.0;
                }
            },
            .SoftmaxCrossEntropy => {
                const logits = self.inputs[0];
                const loss = self.outputs[0];
                const targets = self.context.SoftmaxCrossEntropy.targets;
                const probs = self.context.SoftmaxCrossEntropy.probs;

                const B_size = logits.shape.dims[0];
                const D = logits.shape.dims[1];
                var loss_sum: f32 = 0.0;

                for (0..B_size) |i| {
                    const row = logits.data[i * D .. (i + 1) * D];
                    var max_val = row[0];
                    for (row) |val| {
                        if (val > max_val) max_val = val;
                    }
                    var sum: f32 = 0.0;
                    const row_probs = probs[i * D .. (i + 1) * D];
                    for (0..D) |j| {
                        const e = @exp(row[j] - max_val);
                        row_probs[j] = e;
                        sum += e;
                    }
                    for (0..D) |j| {
                        row_probs[j] /= sum;
                    }
                    const target_idx = targets[i];
                    const prob = row_probs[target_idx];
                    const clipped = @max(prob, 1e-15);
                    loss_sum += -@log(clipped);
                }
                loss.data[0] = loss_sum / @as(f32, @floatFromInt(B_size));
            },
            .Reshape => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                @memcpy(C.data, A.data);
            },
            .Transpose => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const temp = try A.transpose(self.context.Transpose.dim0, self.context.Transpose.dim1, allocator, null);
                defer tensor.free(allocator, temp);
                @memcpy(C.data, temp.data);
            },
            .MseLoss => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];
                const N = A.data.len;
                var loss_sum: f32 = 0.0;
                for (0..N) |i| {
                    const diff = A.data[i] - B.data[i];
                    loss_sum += diff * diff;
                }
                C.data[0] = loss_sum / @as(f32, @floatFromInt(N));
            },
            .MulScalar => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const val = self.context.MulScalar.val;
                for (C.data, A.data) |*c_val, a_val| {
                    c_val.* = a_val * val;
                }
            },
            .AddScalar => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const val = self.context.AddScalar.val;
                for (C.data, A.data) |*c_val, a_val| {
                    c_val.* = a_val + val;
                }
            },
            .Add => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];
                for (C.data, A.data, B.data) |*c_val, a_val, b_val| {
                    c_val.* = a_val + b_val;
                }
            },
        }
    }

    // 执行该算子的反向传播计算，更新其输入节点的梯度
    pub fn backward(self: *Op) !void {
        switch (self.op_type) {
            // ====================================================================
            // 1. 矩阵乘法反向传播 (MatMul Backward)
            // ====================================================================
            // 前向公式: C = A * B，其中 A (M x K), B (K x N), C (M x N)
            // 数学推导:
            // 设损失标量为 L，我们拥有对输出的梯度 dC = ∂L/∂C (M x N)。
            // 根据矩阵微积分链式法则：
            //   1. 对左输入 A 的导数: dA = ∂L/∂A = dC * B^T
            //      维度匹配: (M x N) * (N x K) -> (M x K)
            //   2. 对右输入 B 的导数: dB = ∂L/∂B = A^T * dC
            //      维度匹配: (K x M) * (M x N) -> (K x N)
            // 注意: 在深度学习中，梯度是累加的 (+=)，所以我们传入 beta = 1.0 给 cblas_sgemm。
            .MatMul => {
                const A = self.inputs[0]; // 形状为 M x K
                const B = self.inputs[1]; // 形状为 K x N
                const C = self.outputs[0]; // 形状为 M x N
                const M = A.shape.dims[0];
                const K = A.shape.dims[1];
                const N = B.shape.dims[1];

                // 1. 计算对左乘矩阵 A 的梯度: dA += dC * B^T
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
            // ====================================================================
            // 2. 偏置项加法反向传播 (AddBias Backward)
            // ====================================================================
            // 前向公式: C[i, j] = A[i, j] + bias[j]，其中 A (M x N), bias (1 x N), C (M x N)
            // 数学推导:
            //   1. 对输入 A 的偏导数: ∂L/∂A[i, j] = ∂L/∂C[i, j]
            //      因此 dA += dC (逐元素直接累加)。
            //   2. 对偏置 bias 的偏导数: ∂L/∂bias[j] = sum_{i=0}^{M-1} (∂L/∂C[i, j])
            //      这是因为偏置项在行维度 (批量样本维度) 进行了广播复制。
            //      因此对偏置的梯度为对输出梯度 dC 在第 0 维（行）上的降维累加和。
            .AddBias => {
                const A = self.inputs[0];
                const bias = self.inputs[1];
                const C = self.outputs[0];
                const M = A.shape.dims[0];
                const N = A.shape.dims[1];

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
            // ====================================================================
            // 3. ReLU 激活函数反向传播 (ReLU Backward)
            // ====================================================================
            // 前向公式: C = max(0, A)，逐元素操作
            // 数学推导:
            //   对于每个元素：
            //   若 A[i] > 0，则该点斜率为 1.0 -> dA[i] += dC[i]
            //   若 A[i] <= 0，则该点斜率为 0.0 -> dA[i] += 0.0
            .Relu => {
                const A = self.inputs[0];
                const C = self.outputs[0];

                if (A.requires_grad) {
                    const total = A.data.len;
                    for (0..total) |i| {
                        A.grad[i] += if (A.data[i] > 0.0) C.grad[i] else 0.0;
                    }
                }
            },
            // ====================================================================
            // 4. Softmax + Cross Entropy 损失函数反向传播 (SoftmaxCrossEntropy Backward)
            // ====================================================================
            // 前向公式:
            //   设输入的 Logits 矩阵为 X (M x N)，真实分类为 label (M x 1)。
            //   对于第 i 行样本，先算 Softmax 概率：probs[i, j] = e^{X[i, j]} / sum_k(e^{X[i, k]})
            //   再算平均交叉熵损失：L = -1/M * sum_i( ln(probs[i, label_i]) )
            // 数学推导:
            //   将 Softmax 和 CrossEntropy 结合后，对输入 Logits X[i, j] 的偏导数具有极佳的数值稳定性：
            //     ∂L/∂X[i, j] = (probs[i, j] - Indicator(j == label_i)) / M
            //   其中 Indicator 在当前类别 j 等于真实类别 label_i 时为 1.0，否则为 0.0。
            //   最后除以样本数 M 得到平均样本梯度。
            .SoftmaxCrossEntropy => {
                const logits = self.inputs[0];
                const M = logits.shape.dims[0];
                const N = logits.shape.dims[1];
                const ctx = &self.context.SoftmaxCrossEntropy;

                const scale = 1.0 / @as(f32, @floatFromInt(M));

                for (0..M) |i| {
                    const label = ctx.targets[i];
                    const p_row = ctx.probs[i * N .. (i + 1) * N];
                    const dLogits_row = logits.grad[i * N .. (i + 1) * N];
                    for (0..N) |j| {
                        dLogits_row[j] += scale * (p_row[j] - (if (j == label) @as(f32, 1.0) else 0.0));
                    }
                }
            },
            // ====================================================================
            // 5. 形状变换反向传播 (Reshape Backward)
            // ====================================================================
            // 前向公式: C = reshape(A)，仅改变形状元数据，不修改物理排列
            // 数学推导:
            //   Reshape 没有数学上的参数变换，因此其梯度传递就是将输出梯度 dC
            //   以一维平铺形式直接拷回/累加到输入 A 的梯度 dA 缓冲区中。
            .Reshape => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    for (C.grad, 0..) |g, i| {
                        A.grad[i] += g;
                    }
                }
            },
            // ====================================================================
            // 6. 维度转置反向传播 (Transpose Backward)
            // ====================================================================
            // 前向公式: C = transpose(A, dim0, dim1)
            // 数学推导:
            //   转置算子物理上改变了元素的读取索引。
            //   因此反向传播时，必须通过转置后的 stride 步长定位到 A.grad 中的物理偏移位置，
            //   并将 C.grad 中连续排列的梯度累加进去。
            .Transpose => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    const ctx = self.context.Transpose;
                    const strides_trans = transposeShape(A.strides, ctx.dim0, ctx.dim1);

                    var indices = [_]usize{0} ** 8;
                    const len = C.shape.len;
                    const total_size = C.data.len;
                    for (0..total_size) |dest_flat_idx| {
                        var src_flat_idx: usize = 0;
                        for (0..len) |d| {
                            src_flat_idx += indices[d] * strides_trans.dims[d];
                        }
                        A.grad[src_flat_idx] += C.grad[dest_flat_idx];

                        var d: usize = len;
                        while (d > 0) {
                            d -= 1;
                            indices[d] += 1;
                            if (indices[d] < C.shape.dims[d]) {
                                break;
                            }
                            indices[d] = 0;
                        }
                    }
                }
            },
            .MseLoss => {
                const A = self.inputs[0]; // y_pred
                const B = self.inputs[1]; // y_true
                const C = self.outputs[0]; // loss
                const N = A.data.len;
                const N_f = @as(f32, @floatFromInt(N));

                if (A.requires_grad) {
                    for (0..N) |i| {
                        A.grad[i] += C.grad[0] * (2.0 / N_f) * (A.data[i] - B.data[i]);
                    }
                }
                if (B.requires_grad) {
                    for (0..N) |i| {
                        B.grad[i] += C.grad[0] * (2.0 / N_f) * (B.data[i] - A.data[i]);
                    }
                }
            },
            .MulScalar => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                const val = self.context.MulScalar.val;
                if (A.requires_grad) {
                    for (0..A.data.len) |i| {
                        A.grad[i] += C.grad[i] * val;
                    }
                }
            },
            .AddScalar => {
                const A = self.inputs[0];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    for (0..A.data.len) |i| {
                        A.grad[i] += C.grad[i];
                    }
                }
            },
            .Add => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];
                if (A.requires_grad) {
                    for (0..A.data.len) |i| {
                        A.grad[i] += C.grad[i];
                    }
                }
                if (B.requires_grad) {
                    for (0..B.data.len) |i| {
                        B.grad[i] += C.grad[i];
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

        t.* = Tensor{
            .data = try allocator.alloc(f32, total_size),
            .grad = if (requires_grad) try allocator.alloc(f32, total_size) else &.{},
            .shape = shape,
            .strides = strides,
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

    // 形状变换算子前向传播
    pub fn reshape(self: *Graph, A: *Tensor, new_shape_slice: []const usize) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try allocator.create(Tensor);
        const shape = Shape.init(new_shape_slice);
        const strides = computeContiguousStrides(shape);

        var old_total: usize = 1;
        for (0..A.shape.len) |i| {
            old_total *= A.shape.dims[i];
        }
        var new_total: usize = 1;
        for (new_shape_slice) |dim| {
            new_total *= dim;
        }
        std.debug.assert(old_total == new_total);

        C.* = Tensor{
            .data = A.data, // 共享前向数据
            .grad = if (A.requires_grad) try allocator.alloc(f32, new_total) else &.{},
            .shape = shape,
            .strides = strides,
            .requires_grad = A.requires_grad,
            .creator = null,
        };
        if (A.requires_grad) {
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

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

        return C;
    }

    // 维度转置算子前向传播：交换 dim0 和 dim1
    pub fn transposeND(self: *Graph, A: *Tensor, dim0: usize, dim1: usize) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.transpose(dim0, dim1, allocator, null);

        C.requires_grad = A.requires_grad;
        if (C.requires_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

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

        return C;
    }

    // 矩阵乘法算子前向传播：C = A * B
    pub fn matmul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.matmul(B, allocator, null);

        C.requires_grad = A.requires_grad or B.requires_grad;
        if (C.requires_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

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
        const allocator = self.arena.allocator();
        const C = try A.addBias(bias, allocator, null);

        C.requires_grad = A.requires_grad or bias.requires_grad;
        if (C.requires_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

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
        const allocator = self.arena.allocator();
        const C = try A.relu(allocator, null);

        C.requires_grad = A.requires_grad;
        if (C.requires_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

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

        const B = logits.shape.dims[0];
        const N = logits.shape.dims[1];
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

    // 均方误差 (MSE) 损失函数：C = 1/N * sum((y_pred - y_true)^2)
    pub fn mseLoss(self: *Graph, y_pred: *Tensor, y_true: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const loss = try self.tensor(1, 1, true); // 标量 loss，支持 requires_grad = true

        const N = y_pred.data.len;
        std.debug.assert(N == y_true.data.len);

        var loss_sum: f32 = 0.0;
        for (0..N) |i| {
            const diff = y_pred.data[i] - y_true.data[i];
            loss_sum += diff * diff;
        }
        loss.data[0] = loss_sum / @as(f32, @floatFromInt(N));

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

        return loss;
    }

    // 标量乘法（缩放）：C = val * A
    pub fn mulScalar(self: *Graph, A: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.mulScalar(val, allocator, null);

        C.requires_grad = A.requires_grad;
        if (C.requires_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

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

        return C;
    }

    // 标量加法：C = A + val
    pub fn addScalar(self: *Graph, A: *Tensor, val: f32) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.addScalar(val, allocator, null);

        C.requires_grad = A.requires_grad;
        if (C.requires_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

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

        return C;
    }

    // 逐元素张量加法：C = A + B
    pub fn add(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const allocator = self.arena.allocator();
        const C = try A.add(B, allocator, null);

        C.requires_grad = A.requires_grad or B.requires_grad;
        if (C.requires_grad) {
            C.grad = try allocator.alloc(f32, C.data.len);
            @memset(C.grad, 0.0);
        }

        try self.tensors.append(self.backing_allocator, C);

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

        return C;
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
};
