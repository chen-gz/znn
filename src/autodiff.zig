const std = @import("std");

pub const Tensor = struct {
    data: []f32,
    grad: []f32,
    rows: usize,
    cols: usize,
    requires_grad: bool,
    creator: ?*Op,

    pub fn zeroGrad(self: *Tensor) void {
        if (self.requires_grad) {
            @memset(self.grad, 0.0);
        }
    }
};

pub const OpType = enum {
    MatMul,
    AddBias,
    Relu,
    SoftmaxCrossEntropy,
};

pub const OpContext = union(enum) {
    MatMul: void,
    AddBias: void,
    Relu: void,
    SoftmaxCrossEntropy: struct {
        probs: []f32,
        targets: []const u8,
    },
};

pub const Op = struct {
    op_type: OpType,
    inputs: []*Tensor,
    outputs: []*Tensor,
    context: OpContext,

    pub fn backward(self: *Op) void {
        switch (self.op_type) {
            .MatMul => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];
                const M = A.rows;
                const K = A.cols;
                const N = B.cols;

                // C = A * B
                // dA += dC * B^T
                if (A.requires_grad) {
                    for (0..M) |m| {
                        for (0..K) |k| {
                            const b_row = B.data[k * N .. (k + 1) * N];
                            const dC_row = C.grad[m * N .. (m + 1) * N];
                            var sum: f32 = 0.0;
                            for (0..N) |n| {
                                sum += dC_row[n] * b_row[n];
                            }
                            A.grad[m * K + k] += sum;
                        }
                    }
                }

                // dB += A^T * dC
                if (B.requires_grad) {
                    for (0..M) |m| {
                        for (0..K) |k| {
                            const a_val = A.data[m * K + k];
                            const dB_row = B.grad[k * N .. (k + 1) * N];
                            const dC_row = C.grad[m * N .. (m + 1) * N];
                            for (0..N) |n| {
                                dB_row[n] += a_val * dC_row[n];
                            }
                        }
                    }
                }
            },
            .AddBias => {
                const A = self.inputs[0];
                const bias = self.inputs[1];
                const C = self.outputs[0];
                const M = A.rows;
                const N = A.cols;

                // C = A + bias
                for (0..M) |m| {
                    const dA_row = A.grad[m * N .. (m + 1) * N];
                    const dBias_row = bias.grad;
                    const dC_row = C.grad[m * N .. (m + 1) * N];
                    for (0..N) |n| {
                        if (A.requires_grad) dA_row[n] += dC_row[n];
                        if (bias.requires_grad) dBias_row[n] += dC_row[n];
                    }
                }
            },
            .Relu => {
                const A = self.inputs[0];
                const C = self.outputs[0];

                if (A.requires_grad) {
                    for (0..A.data.len) |i| {
                        A.grad[i] += if (A.data[i] > 0.0) C.grad[i] else 0.0;
                    }
                }
            },
            .SoftmaxCrossEntropy => {
                const logits = self.inputs[0];
                const M = logits.rows;
                const N = logits.cols;
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
        }
    }
};

pub const Graph = struct {
    backing_allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    tensors: std.ArrayList(*Tensor),
    ops: std.ArrayList(*Op),

    pub fn init(backing_allocator: std.mem.Allocator) Graph {
        return Graph{
            .backing_allocator = backing_allocator,
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .tensors = .empty,
            .ops = .empty,
        };
    }

    pub fn deinit(self: *Graph) void {
        self.tensors.deinit(self.backing_allocator);
        self.ops.deinit(self.backing_allocator);
        self.arena.deinit();
    }

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

    pub fn matmul(self: *Graph, A: *Tensor, B: *Tensor) !*Tensor {
        const C = try self.tensor(A.rows, B.cols, A.requires_grad or B.requires_grad);

        const M = A.rows;
        const K = A.cols;
        const N = B.cols;

        // Perform optimized matrix multiplication C = A * B
        for (0..M) |i| {
            for (0..K) |k| {
                const a_val = A.data[i * K + k];
                const b_row = B.data[k * N .. (k + 1) * N];
                const c_row = C.data[i * N .. (i + 1) * N];
                for (0..N) |j| {
                    c_row[j] += a_val * b_row[j];
                }
            }
        }

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

    pub fn addBias(self: *Graph, A: *Tensor, bias: *Tensor) !*Tensor {
        const C = try self.tensor(A.rows, A.cols, A.requires_grad or bias.requires_grad);

        const M = A.rows;
        const N = A.cols;

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

    pub fn relu(self: *Graph, A: *Tensor) !*Tensor {
        const C = try self.tensor(A.rows, A.cols, A.requires_grad);

        for (0..A.data.len) |i| {
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

    pub fn softmaxCrossEntropy(self: *Graph, logits: *Tensor, targets: []const u8) !*Tensor {
        const loss = try self.tensor(1, 1, logits.requires_grad);

        const B = logits.rows;
        const N = logits.cols;
        const allocator = self.arena.allocator();
        const probs = try allocator.alloc(f32, B * N);

        // Softmax
        for (0..B) |i| {
            const logits_row = logits.data[i * N .. (i + 1) * N];
            const probs_row = probs[i * N .. (i + 1) * N];

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

            for (probs_row) |*p| {
                p.* /= sum;
            }
        }

        // Cross entropy
        var loss_sum: f32 = 0.0;
        for (0..B) |i| {
            const label = targets[i];
            const prob = probs[i * N + label];
            const clipped = @max(prob, 1e-15);
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

    pub fn backward(self: *Graph, loss_tensor: *Tensor) !void {
        const allocator = self.arena.allocator();
        var visited = std.AutoHashMap(*Tensor, void).init(allocator);
        defer visited.deinit();
        var sorted_list: std.ArrayList(*Tensor) = .empty;
        defer sorted_list.deinit(allocator);

        try self.topologicalSort(loss_tensor, &visited, &sorted_list);

        // Loss gradient initialization
        loss_tensor.grad[0] = 1.0;

        // Execute backward pass in reverse topological order
        var i = sorted_list.items.len;
        while (i > 0) {
            i -= 1;
            const node = sorted_list.items[i];
            if (node.creator) |op| {
                op.backward();
            }
        }
    }

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
