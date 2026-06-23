const std = @import("std");
const c = @import("cblas.zig");
const build_options = @import("build_options");
const use_mlx_enabled = build_options.use_mlx;

// 运行模式开关：true 表示使用 Metal GPU 运行，false 表示使用 Accelerate CPU 运行
pub var use_gpu: bool = false;
pub var use_mlx: bool = false;

// 重新导出外部 C 绑定以保持 API 兼容性
pub const metal_init = c.metal_init;
pub const metal_matmul = c.metal_matmul;

const mlx = struct {
    const mlx_array = extern struct {
        ctx: ?*anyopaque,
    };
    const mlx_stream = extern struct {
        ctx: ?*anyopaque,
    };
    const MLX_FLOAT32: c_int = 10;

    const mlx_array_new: ?*const fn () callconv(.c) mlx_array = if (use_mlx_enabled) @extern(*const fn () callconv(.c) mlx_array, .{ .name = "mlx_array_new" }) else null;
    const mlx_array_new_data: ?*const fn (data: ?*const anyopaque, shape: [*]const c_int, ndim: c_int, dtype: c_int) callconv(.c) mlx_array = if (use_mlx_enabled) @extern(*const fn (data: ?*const anyopaque, shape: [*]const c_int, ndim: c_int, dtype: c_int) callconv(.c) mlx_array, .{ .name = "mlx_array_new_data" }) else null;
    const mlx_array_new_float32: ?*const fn (val: f32) callconv(.c) mlx_array = if (use_mlx_enabled) @extern(*const fn (val: f32) callconv(.c) mlx_array, .{ .name = "mlx_array_new_float32" }) else null;
    const mlx_transpose: ?*const fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int = if (use_mlx_enabled) @extern(*const fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int, .{ .name = "mlx_transpose" }) else null;
    const mlx_matmul: ?*const fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int = if (use_mlx_enabled) @extern(*const fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int, .{ .name = "mlx_matmul" }) else null;
    const mlx_add: ?*const fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int = if (use_mlx_enabled) @extern(*const fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int, .{ .name = "mlx_add" }) else null;
    const mlx_sum_axis: ?*const fn (res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) callconv(.c) c_int = if (use_mlx_enabled) @extern(*const fn (res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) callconv(.c) c_int, .{ .name = "mlx_sum_axis" }) else null;
    const mlx_maximum: ?*const fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int = if (use_mlx_enabled) @extern(*const fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int, .{ .name = "mlx_maximum" }) else null;
    const mlx_greater: ?*const fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int = if (use_mlx_enabled) @extern(*const fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int, .{ .name = "mlx_greater" }) else null;
    const mlx_multiply: ?*const fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int = if (use_mlx_enabled) @extern(*const fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int, .{ .name = "mlx_multiply" }) else null;
    const mlx_array_eval: ?*const fn (a: mlx_array) callconv(.c) c_int = if (use_mlx_enabled) @extern(*const fn (a: mlx_array) callconv(.c) c_int, .{ .name = "mlx_array_eval" }) else null;
    const mlx_array_data_float32: ?*const fn (a: mlx_array) callconv(.c) [*]f32 = if (use_mlx_enabled) @extern(*const fn (a: mlx_array) callconv(.c) [*]f32, .{ .name = "mlx_array_data_float32" }) else null;
    const mlx_array_free: ?*const fn (a: mlx_array) callconv(.c) void = if (use_mlx_enabled) @extern(*const fn (a: mlx_array) callconv(.c) void, .{ .name = "mlx_array_free" }) else null;
    const mlx_default_gpu_stream_new: ?*const fn () callconv(.c) mlx_stream = if (use_mlx_enabled) @extern(*const fn () callconv(.c) mlx_stream, .{ .name = "mlx_default_gpu_stream_new" }) else null;
    const mlx_stream_free: ?*const fn (s: mlx_stream) callconv(.c) void = if (use_mlx_enabled) @extern(*const fn (s: mlx_stream) callconv(.c) void, .{ .name = "mlx_stream_free" }) else null;
};

fn mlx_matmul_impl(
    transA: bool, transB: bool,
    M: usize, N: usize, K: usize,
    A: []const f32, B: []const f32, C: []f32,
    beta: f32
) void {
    if (!use_mlx_enabled) return;

    const stream = mlx.mlx_default_gpu_stream_new.?();
    defer mlx.mlx_stream_free.?(stream);

    // Dimensions of inputs before transposition:
    // If transA, input A has shape K x M (so we transpose to M x K)
    // If not transA, input A has shape M x K
    const rowsA = if (transA) K else M;
    const colsA = if (transA) M else K;
    const shapeA = [2]c_int{ @intCast(rowsA), @intCast(colsA) };
    const a_arr = mlx.mlx_array_new_data.?(A.ptr, &shapeA, 2, mlx.MLX_FLOAT32);
    defer mlx.mlx_array_free.?(a_arr);

    const rowsB = if (transB) N else K;
    const colsB = if (transB) K else N;
    const shapeB = [2]c_int{ @intCast(rowsB), @intCast(colsB) };
    const b_arr = mlx.mlx_array_new_data.?(B.ptr, &shapeB, 2, mlx.MLX_FLOAT32);
    defer mlx.mlx_array_free.?(b_arr);

    // Apply transpositions if required
    var a_input = a_arr;
    var b_input = b_arr;

    if (transA) {
        var a_trans = mlx.mlx_array_new.?();
        _ = mlx.mlx_transpose.?(&a_trans, a_arr, stream);
        a_input = a_trans;
    }
    defer if (transA) mlx.mlx_array_free.?(a_input);

    if (transB) {
        var b_trans = mlx.mlx_array_new.?();
        _ = mlx.mlx_transpose.?(&b_trans, b_arr, stream);
        b_input = b_trans;
    }
    defer if (transB) mlx.mlx_array_free.?(b_input);

    var res_arr = mlx.mlx_array_new.?();
    defer mlx.mlx_array_free.?(res_arr);

    _ = mlx.mlx_matmul.?(&res_arr, a_input, b_input, stream);
    
    // Evaluate res_arr and get underlying buffer
    _ = mlx.mlx_array_eval.?(res_arr);
    const data_ptr = mlx.mlx_array_data_float32.?(res_arr);

    if (beta == 0.0) {
        @memcpy(C, data_ptr[0 .. M * N]);
    } else {
        for (0..M * N) |i| {
            C[i] = data_ptr[i] + beta * C[i];
        }
    }
}

fn mlx_add_bias_impl(
    M: usize, N: usize,
    A: []const f32, bias: []const f32, C: []f32
) void {
    if (!use_mlx_enabled) return;

    const stream = mlx.mlx_default_gpu_stream_new.?();
    defer mlx.mlx_stream_free.?(stream);

    const shapeA = [2]c_int{ @intCast(M), @intCast(N) };
    const a_arr = mlx.mlx_array_new_data.?(A.ptr, &shapeA, 2, mlx.MLX_FLOAT32);
    defer mlx.mlx_array_free.?(a_arr);

    const shapeB = [2]c_int{ 1, @intCast(N) };
    const b_arr = mlx.mlx_array_new_data.?(bias.ptr, &shapeB, 2, mlx.MLX_FLOAT32);
    defer mlx.mlx_array_free.?(b_arr);

    var res_arr = mlx.mlx_array_new.?();
    defer mlx.mlx_array_free.?(res_arr);

    _ = mlx.mlx_add.?(&res_arr, a_arr, b_arr, stream);
    _ = mlx.mlx_array_eval.?(res_arr);

    const data_ptr = mlx.mlx_array_data_float32.?(res_arr);
    @memcpy(C, data_ptr[0 .. M * N]);
}

fn mlx_add_bias_backward_impl(
    M: usize, N: usize,
    C_grad: []const f32,
    A_grad: ?[]f32,
    bias_grad: ?[]f32
) void {
    if (!use_mlx_enabled) return;

    const stream = mlx.mlx_default_gpu_stream_new.?();
    defer mlx.mlx_stream_free.?(stream);

    const shapeC = [2]c_int{ @intCast(M), @intCast(N) };
    const c_grad_arr = mlx.mlx_array_new_data.?(C_grad.ptr, &shapeC, 2, mlx.MLX_FLOAT32);
    defer mlx.mlx_array_free.?(c_grad_arr);

    if (A_grad) |a_g| {
        const a_grad_arr = mlx.mlx_array_new_data.?(a_g.ptr, &shapeC, 2, mlx.MLX_FLOAT32);
        defer mlx.mlx_array_free.?(a_grad_arr);

        var new_a_grad = mlx.mlx_array_new.?();
        defer mlx.mlx_array_free.?(new_a_grad);

        _ = mlx.mlx_add.?(&new_a_grad, a_grad_arr, c_grad_arr, stream);
        _ = mlx.mlx_array_eval.?(new_a_grad);
        const data_ptr = mlx.mlx_array_data_float32.?(new_a_grad);
        @memcpy(a_g, data_ptr[0 .. M * N]);
    }

    if (bias_grad) |b_g| {
        var sum_arr = mlx.mlx_array_new.?();
        defer mlx.mlx_array_free.?(sum_arr);

        _ = mlx.mlx_sum_axis.?(&sum_arr, c_grad_arr, 0, false, stream);

        const shapeB = [2]c_int{ 1, @intCast(N) };
        const b_grad_arr = mlx.mlx_array_new_data.?(b_g.ptr, &shapeB, 2, mlx.MLX_FLOAT32);
        defer mlx.mlx_array_free.?(b_grad_arr);

        var new_b_grad = mlx.mlx_array_new.?();
        defer mlx.mlx_array_free.?(new_b_grad);

        _ = mlx.mlx_add.?(&new_b_grad, b_grad_arr, sum_arr, stream);
        _ = mlx.mlx_array_eval.?(new_b_grad);
        const data_ptr = mlx.mlx_array_data_float32.?(new_b_grad);
        @memcpy(b_g, data_ptr[0 .. N]);
    }
}

fn mlx_relu_impl(
    total_size: usize,
    A: []const f32, C: []f32
) void {
    if (!use_mlx_enabled) return;

    const stream = mlx.mlx_default_gpu_stream_new.?();
    defer mlx.mlx_stream_free.?(stream);

    const shape = [1]c_int{ @intCast(total_size) };
    const a_arr = mlx.mlx_array_new_data.?(A.ptr, &shape, 1, mlx.MLX_FLOAT32);
    defer mlx.mlx_array_free.?(a_arr);

    const zero = mlx.mlx_array_new_float32.?(0.0);
    defer mlx.mlx_array_free.?(zero);

    var res_arr = mlx.mlx_array_new.?();
    defer mlx.mlx_array_free.?(res_arr);

    _ = mlx.mlx_maximum.?(&res_arr, a_arr, zero, stream);
    _ = mlx.mlx_array_eval.?(res_arr);

    const data_ptr = mlx.mlx_array_data_float32.?(res_arr);
    @memcpy(C, data_ptr[0..total_size]);
}

fn mlx_relu_backward_impl(
    total_size: usize,
    A_data: []const f32,
    C_grad: []const f32,
    A_grad: []f32
) void {
    if (!use_mlx_enabled) return;

    const stream = mlx.mlx_default_gpu_stream_new.?();
    defer mlx.mlx_stream_free.?(stream);

    const shape = [1]c_int{ @intCast(total_size) };
    const a_data_arr = mlx.mlx_array_new_data.?(A_data.ptr, &shape, 1, mlx.MLX_FLOAT32);
    defer mlx.mlx_array_free.?(a_data_arr);

    const c_grad_arr = mlx.mlx_array_new_data.?(C_grad.ptr, &shape, 1, mlx.MLX_FLOAT32);
    defer mlx.mlx_array_free.?(c_grad_arr);

    const zero = mlx.mlx_array_new_float32.?(0.0);
    defer mlx.mlx_array_free.?(zero);

    var cond = mlx.mlx_array_new.?();
    defer mlx.mlx_array_free.?(cond);
    _ = mlx.mlx_greater.?(&cond, a_data_arr, zero, stream);

    var grad_step = mlx.mlx_array_new.?();
    defer mlx.mlx_array_free.?(grad_step);
    _ = mlx.mlx_multiply.?(&grad_step, c_grad_arr, cond, stream);

    const a_grad_arr = mlx.mlx_array_new_data.?(A_grad.ptr, &shape, 1, mlx.MLX_FLOAT32);
    defer mlx.mlx_array_free.?(a_grad_arr);

    var new_grad = mlx.mlx_array_new.?();
    defer mlx.mlx_array_free.?(new_grad);
    _ = mlx.mlx_add.?(&new_grad, a_grad_arr, grad_step, stream);
    _ = mlx.mlx_array_eval.?(new_grad);

    const data_ptr = mlx.mlx_array_data_float32.?(new_grad);
    @memcpy(A_grad, data_ptr[0..total_size]);
}

const num_threads = 4; // Using 4 threads is the sweet spot for performance and thread spawn overhead

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
                    if (use_mlx) {
                        mlx_matmul_impl(false, true, M, K, N, C.grad, B.data, A.grad, 1.0);
                    } else if (use_gpu) {
                        // 使用 GPU Metal 矩阵乘法进行计算，并使用 beta=1.0 累加当前梯度
                        metal_matmul(0, 1, @intCast(M), @intCast(K), @intCast(N), C.grad.ptr, B.data.ptr, A.grad.ptr, 1.0);
                    } else {
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
                }

                // 2. 计算对右乘矩阵 B 的梯度: dB += A^T * dC
                // 数学原理: d(A * B)/dB = A^T * dC，形状为 (K x M) * (M x N) -> K x N
                if (B.requires_grad) {
                    if (use_mlx) {
                        mlx_matmul_impl(true, false, K, N, M, A.data, C.grad, B.grad, 1.0);
                    } else if (use_gpu) {
                        metal_matmul(1, 0, @intCast(K), @intCast(N), @intCast(M), A.data.ptr, C.grad.ptr, B.grad.ptr, 1.0);
                    } else {
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
                if (use_mlx) {
                    mlx_add_bias_backward_impl(
                        M, N,
                        C.grad,
                        if (A.requires_grad) A.grad else null,
                        if (bias.requires_grad) bias.grad else null
                    );
                } else {
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
                }
            },
            .Relu => {
                const A = self.inputs[0];
                const C = self.outputs[0];

                // ReLU 反向传播：
                // 如果前向值 A.data[i] > 0，则梯度原样传递：dA[i] += dC[i]
                // 如果前向值 A.data[i] <= 0，则梯度置为 0
                if (A.requires_grad) {
                    if (use_mlx) {
                        mlx_relu_backward_impl(A.data.len, A.data, C.grad, A.grad);
                    } else {
                        const total = A.data.len;
                        for (0..total) |i| {
                            A.grad[i] += if (A.data[i] > 0.0) C.grad[i] else 0.0;
                        }
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

        // 根据硬件后端类型分流计算
        if (use_mlx) {
            mlx_matmul_impl(false, false, M, N, K, A.data, B.data, C.data, 0.0);
        } else if (use_gpu) {
            // 调用 Metal GPU 计算：无转置矩阵乘法，结果累加因子为 0.0（全新覆盖）
            metal_matmul(0, 0, @intCast(M), @intCast(N), @intCast(K), A.data.ptr, B.data.ptr, C.data.ptr, 0.0);
        } else {
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
        }

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

        if (use_mlx) {
            mlx_add_bias_impl(M, N, A.data, bias.data, C.data);
        } else {
            // 偏置向量形状为 1 x N，按行广播复制相加到每一行
            for (0..M) |i| {
                const a_row = A.data[i * N .. (i + 1) * N];
                const c_row = C.data[i * N .. (i + 1) * N];
                for (0..N) |j| {
                    c_row[j] = a_row[j] + bias.data[j];
                }
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

        if (use_mlx) {
            mlx_relu_impl(A.data.len, A.data, C.data);
        } else {
            const total = A.data.len;
            for (0..total) |i| {
                C.data[i] = if (A.data[i] > 0.0) A.data[i] else 0.0;
            }
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
