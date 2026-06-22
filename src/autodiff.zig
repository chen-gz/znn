const std = @import("std");

const num_threads = 4; // Using 4 threads is the sweet spot for performance and thread spawn overhead

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

// ThreadPool implementation
pub const ThreadPool = struct {
    workers: [num_threads - 1]Worker,
    task: std.atomic.Value(?*const Task),
    barrier: std.atomic.Value(usize),

    const Worker = struct {
        thread: std.Thread,
        state: std.atomic.Value(u32), // 0: idle, 1: busy, 2: exit
        id: usize,
    };

    const Task = struct {
        func: *const fn (ctx: ?*anyopaque, thread_idx: usize) void,
        ctx: ?*anyopaque,
    };

    fn sleepNs(ns: u64) void {
        var ts = std.posix.timespec{
            .sec = @intCast(ns / 1_000_000_000),
            .nsec = @intCast(ns % 1_000_000_000),
        };
        _ = std.posix.system.nanosleep(&ts, null);
    }

    fn workerEntry(self: *ThreadPool, worker_id: usize) void {
        const worker = &self.workers[worker_id];
        while (true) {
            var state = worker.state.load(.acquire);
            var spins: usize = 0;
            while (state == 0) {
                spins += 1;
                if (spins > 1000) {
                    sleepNs(10_000); // 10 microseconds
                } else {
                    std.Thread.yield() catch {};
                }
                state = worker.state.load(.acquire);
            }
            if (state == 2) {
                break;
            }

            if (self.task.load(.acquire)) |task| {
                task.func(task.ctx, worker_id + 1);
            }

            worker.state.store(0, .release);
            _ = self.barrier.fetchAdd(1, .acq_rel);
        }
    }

    pub fn run(self: *ThreadPool, func: *const fn (ctx: ?*anyopaque, thread_idx: usize) void, ctx: ?*anyopaque) void {
        const task = Task{
            .func = func,
            .ctx = ctx,
        };
        self.task.store(&task, .release);
        self.barrier.store(0, .release);

        for (0..num_threads - 1) |i| {
            self.workers[i].state.store(1, .release);
        }

        func(ctx, 0);

        var spins: usize = 0;
        while (self.barrier.load(.acquire) < num_threads - 1) {
            spins += 1;
            if (spins > 1000) {
                std.Thread.yield() catch {};
            }
        }

        self.task.store(null, .release);
    }
};

pub var global_pool: ThreadPool = undefined;
pub var global_pool_initialized: bool = false;

pub fn initThreadPool() !void {
    if (global_pool_initialized) return;
    global_pool = ThreadPool{
        .workers = undefined,
        .task = std.atomic.Value(?*const ThreadPool.Task).init(null),
        .barrier = std.atomic.Value(usize).init(0),
    };
    for (0..num_threads - 1) |i| {
        global_pool.workers[i] = .{
            .thread = undefined,
            .state = std.atomic.Value(u32).init(0),
            .id = i,
        };
    }
    // Spawn threads
    for (0..num_threads - 1) |i| {
        global_pool.workers[i].thread = try std.Thread.spawn(.{}, ThreadPool.workerEntry, .{ &global_pool, i });
    }
    global_pool_initialized = true;
}

pub fn deinitThreadPool() void {
    if (global_pool_initialized) {
        for (0..num_threads - 1) |i| {
            global_pool.workers[i].state.store(2, .release);
        }
        for (0..num_threads - 1) |i| {
            global_pool.workers[i].thread.join();
        }
        global_pool_initialized = false;
    }
}

// Helper contexts and tasks for ThreadPool execution (Only heavy MatMul needs ThreadPool)

const MatMulContext = struct {
    M: usize, N: usize, K: usize,
    A_data: []const f32,
    B_data: []const f32,
    C_data: []f32,
    chunk_size: usize,
};

fn matmulTask(ctx_ptr: ?*anyopaque, thread_idx: usize) void {
    const ctx: *const MatMulContext = @ptrCast(@alignCast(ctx_ptr.?));
    const start_row = thread_idx * ctx.chunk_size;
    if (start_row >= ctx.M) return;
    const end_row = @min(start_row + ctx.chunk_size, ctx.M);

    for (start_row..end_row) |i| {
        for (0..ctx.K) |k| {
            const a_val = ctx.A_data[i * ctx.K + k];
            const b_row = ctx.B_data[k * ctx.N .. (k + 1) * ctx.N];
            const c_row = ctx.C_data[i * ctx.N .. (i + 1) * ctx.N];
            for (0..ctx.N) |j| {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

const dAContext = struct {
    M: usize, K: usize, N: usize,
    B_data: []const f32,
    dC_grad: []const f32,
    dA_grad: []f32,
    chunk_size: usize,
};

fn dATask(ctx_ptr: ?*anyopaque, thread_idx: usize) void {
    const ctx: *const dAContext = @ptrCast(@alignCast(ctx_ptr.?));
    const start_row = thread_idx * ctx.chunk_size;
    if (start_row >= ctx.M) return;
    const end_row = @min(start_row + ctx.chunk_size, ctx.M);

    for (start_row..end_row) |m| {
        for (0..ctx.K) |k| {
            const b_row = ctx.B_data[k * ctx.N .. (k + 1) * ctx.N];
            const dC_row = ctx.dC_grad[m * ctx.N .. (m + 1) * ctx.N];
            var sum: f32 = 0.0;
            for (0..ctx.N) |n| {
                sum += dC_row[n] * b_row[n];
            }
            ctx.dA_grad[m * ctx.K + k] += sum;
        }
    }
}

const dBContext = struct {
    M: usize, K: usize, N: usize,
    A_data: []const f32,
    dC_grad: []const f32,
    dB_grad: []f32,
    chunk_size: usize,
};

fn dBTask(ctx_ptr: ?*anyopaque, thread_idx: usize) void {
    const ctx: *const dBContext = @ptrCast(@alignCast(ctx_ptr.?));
    const start_k = thread_idx * ctx.chunk_size;
    if (start_k >= ctx.K) return;
    const end_k = @min(start_k + ctx.chunk_size, ctx.K);

    for (start_k..end_k) |k| {
        const dB_row = ctx.dB_grad[k * ctx.N .. (k + 1) * ctx.N];
        for (0..ctx.M) |m| {
            const a_val = ctx.A_data[m * ctx.K + k];
            const dC_row = ctx.dC_grad[m * ctx.N .. (m + 1) * ctx.N];
            for (0..ctx.N) |n| {
                dB_row[n] += a_val * dC_row[n];
            }
        }
    }
}

pub const Op = struct {
    op_type: OpType,
    inputs: []*Tensor,
    outputs: []*Tensor,
    context: OpContext,

    pub fn backward(self: *Op) !void {
        switch (self.op_type) {
            .MatMul => {
                const A = self.inputs[0];
                const B = self.inputs[1];
                const C = self.outputs[0];
                const M = A.rows;
                const K = A.cols;
                const N = B.cols;

                // dA += dC * B^T (Parallelized via ThreadPool)
                if (A.requires_grad) {
                    if (global_pool_initialized) {
                        const chunk_size = (M + num_threads - 1) / num_threads;
                        var ctx = dAContext{
                            .M = M, .K = K, .N = N,
                            .B_data = B.data, .dC_grad = C.grad, .dA_grad = A.grad,
                            .chunk_size = chunk_size,
                        };
                        global_pool.run(dATask, &ctx);
                    } else {
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
                }

                // dB += A^T * dC (Parallelized via ThreadPool)
                if (B.requires_grad) {
                    if (global_pool_initialized) {
                        const chunk_size = (K + num_threads - 1) / num_threads;
                        var ctx = dBContext{
                            .M = M, .K = K, .N = N,
                            .A_data = A.data, .dC_grad = C.grad, .dB_grad = B.grad,
                            .chunk_size = chunk_size,
                        };
                        global_pool.run(dBTask, &ctx);
                    } else {
                        for (0..K) |k| {
                            const dB_row = B.grad[k * N .. (k + 1) * N];
                            for (0..M) |m| {
                                const a_val = A.data[m * K + k];
                                const dC_row = C.grad[m * N .. (m + 1) * N];
                                for (0..N) |n| {
                                    dB_row[n] += a_val * dC_row[n];
                                }
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

                // AddBias backward is element-wise + single column summation.
                // Keeping it single-threaded completely eliminates thread synchronization overhead.
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

                if (A.requires_grad) {
                    const total = A.data.len;
                    // ReLU backward is basic element-wise comparisons. Single-threaded is extremely fast.
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

                const scale = 1.0 / @as(f32, @floatFromInt(M));

                // SoftmaxCrossEntropy backward (logits grad calculation) is small.
                // Keep it single-threaded.
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

        if (global_pool_initialized) {
            const chunk_size = (M + num_threads - 1) / num_threads;
            var ctx = MatMulContext{
                .M = M, .N = N, .K = K,
                .A_data = A.data, .B_data = B.data, .C_data = C.data,
                .chunk_size = chunk_size,
            };
            global_pool.run(matmulTask, &ctx);
        } else {
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

        // AddBias is lightweight. Keep it single-threaded.
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

        const total = A.data.len;
        // ReLU forward is lightweight. Keep it single-threaded.
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

    pub fn softmaxCrossEntropy(self: *Graph, logits: *Tensor, targets: []const u8) !*Tensor {
        const loss = try self.tensor(1, 1, logits.requires_grad);

        const B = logits.rows;
        const N = logits.cols;
        const allocator = self.arena.allocator();
        const probs = try allocator.alloc(f32, B * N);

        // Softmax is lightweight. Keep it single-threaded.
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

        // Cross entropy (Single-threaded reduction, extremely fast for batch size 64)
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
                try op.backward();
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
