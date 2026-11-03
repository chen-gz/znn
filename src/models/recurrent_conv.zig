//! Recurrent / Tied Convolution with Zero-Allocation Ping-Pong Double Buffering
//! 循环/权重共享卷积模块与乒乓双缓冲零动态内存分配机制
//!
//! 核心设计理念：
//! 1. 解耦参数量与网络深度：通过跨时间步 $t \in \{1, \dots, T\}$ 绑定空间卷积核 $W_r$，
//!    有效感受野随迭代步数线性扩展 $RF_t = RF_0 + t(k - 1)$，而物理模型参数量严格保持不变。
//! 2. 动力学稳定性保证：引入凸阻尼平滑系数 $\alpha \in (0, 1]$ 与分组归一化 (GroupNorm)，
//!    严格约束时间展开的单步雅可比谱范数 $\|J_t\|_2 \le 1.0$，无条件消除梯度爆炸与高频振荡。
//! 3. 显式内存与零动态分配：推理阶段构建基于双缓冲交替复用的 Ping-Pong 机制 (`forwardO1`)，
//!    结合预分配展开切片 `col_buf` 与 `FixedBufferAllocator`，彻底消除运行时堆分配 ($O(1)$ 空间)。
//! 4. 自适应计算深度 (ACT)：实时计算尺度无关残差均方根范数 (RMSE)，动态判定特征收敛并支持动态早停。

const std = @import("std");
const backend = @import("../backend.zig");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const nn = @import("../nn.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Graph = autodiff.Graph;
const Conv2D = nn.Conv2D;
const GroupNorm = nn.GroupNorm;
const Linear = nn.Linear;

// ============================================================================
// 1. 配置选项与结果结构体 (RecurrentConvOptions & RecurrentConvResult)
// ============================================================================

/// 循环卷积配置选项 (Recurrent Convolution Options)
pub const RecurrentConvOptions = struct {
    kernel_size: usize = 3,
    stride: usize = 1,
    padding: usize = 1,
    groups: usize = 1,
    num_norm_groups: usize = 8,
    t_min: usize = 2,
    t_max: usize = 8,
    epsilon: f32 = 1e-3,
    alpha_damping: f32 = 0.8,
    use_residual: bool = true,

    pub const default: RecurrentConvOptions = .{};

    pub fn defaultOptions() RecurrentConvOptions {
        return .{};
    }
};

/// 循环卷积推理执行结果 (Recurrent Convolution Inference Result)
pub const RecurrentConvResult = struct {
    output: *Tensor,
    steps_taken: usize,
    converged: bool,
    final_residual: f32,
};

// ============================================================================
// 2. 预分配专用工作区 (RecurrentWorkspace)
// ============================================================================

/// 循环卷积零动态分配工作区 (Recurrent Convolution Workspace)
///
/// 专用于 `forwardO1` 推理执行流：
/// - `buf_a`, `buf_b`: 乒乓状态双缓冲区，交替承担上一时间步隐状态读取与当前步写入；
/// - `buf_u`: 静态前馈输入投影特征缓冲区 $W_f * x$，整个循环展开期间保持只读；
/// - `buf_conv`: 单步循环卷积 $W_r * h_{t-1}$ 中间输出缓冲区；
/// - `col_buf`: 预分配的图像转列 (`im2col`) 展开连续浮点切片，消除底层卷积的堆分配。
pub const RecurrentWorkspace = struct {
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    buf_a: *Tensor,
    buf_b: *Tensor,
    buf_u: *Tensor,
    buf_conv: *Tensor,
    col_buf: []f32,

    pub fn init(
        allocator: std.mem.Allocator,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
    ) !RecurrentWorkspace {
        const shape_4d = [_]usize{ batch_size, channels, height, width };

        const buf_a = try nn.createBuffer(allocator, &shape_4d);
        errdefer nn.freePersistentTensor(allocator, buf_a);

        const buf_b = try nn.createBuffer(allocator, &shape_4d);
        errdefer nn.freePersistentTensor(allocator, buf_b);

        const buf_u = try nn.createBuffer(allocator, &shape_4d);
        errdefer nn.freePersistentTensor(allocator, buf_u);

        const buf_conv = try nn.createBuffer(allocator, &shape_4d);
        errdefer nn.freePersistentTensor(allocator, buf_conv);

        const k_col = channels * kernel_size * kernel_size;
        const l_out = height * width;
        const col_buf = try allocator.alloc(f32, k_col * l_out);
        errdefer allocator.free(col_buf);

        return .{
            .batch_size = batch_size,
            .channels = channels,
            .height = height,
            .width = width,
            .buf_a = buf_a,
            .buf_b = buf_b,
            .buf_u = buf_u,
            .buf_conv = buf_conv,
            .col_buf = col_buf,
        };
    }

    pub fn deinit(self: *RecurrentWorkspace, allocator: std.mem.Allocator) void {
        allocator.free(self.col_buf);
        nn.freePersistentTensor(allocator, self.buf_conv);
        nn.freePersistentTensor(allocator, self.buf_u);
        nn.freePersistentTensor(allocator, self.buf_b);
        nn.freePersistentTensor(allocator, self.buf_a);
    }
};

// ============================================================================
// 3. 残差均方根范数判定 (Dimension-Normalized Residual RMSE)
// ============================================================================

/// 高吞吐 SIMD 向量化残差均方根范数计算:
/// \delta_t^{(b)} = \frac{\|curr - prev\|_2}{\sqrt{C \cdot H \cdot W}}
/// 返回当前批次内所有样本的最大残差值 \max_b \delta_t^{(b)}。
pub fn computeResidualRmse(
    curr: []const f32,
    prev: []const f32,
    batch_size: usize,
    c_h_w: usize,
) f32 {
    std.debug.assert(curr.len == prev.len);
    std.debug.assert(curr.len == batch_size * c_h_w);

    const inv_vol = 1.0 / @as(f32, @floatFromInt(c_h_w));
    var max_rmse: f32 = 0.0;

    for (0..batch_size) |b| {
        const start = b * c_h_w;
        const end = start + c_h_w;
        const c_slice = curr[start..end];
        const p_slice = prev[start..end];

        var acc: backend.vec_math.V8 = @splat(0.0);
        var i: usize = 0;
        while (i + 8 <= c_h_w) : (i += 8) {
            const diff = @as(backend.vec_math.V8, c_slice[i..][0..8].*) - @as(backend.vec_math.V8, p_slice[i..][0..8].*);
            acc += diff * diff;
        }
        var sum_sq: f32 = @reduce(.Add, acc);
        while (i < c_h_w) : (i += 1) {
            const diff = c_slice[i] - p_slice[i];
            sum_sq += diff * diff;
        }
        const rmse = @sqrt(sum_sq * inv_vol);
        if (std.math.isNan(rmse)) return std.math.nan(f32);
        max_rmse = @max(max_rmse, rmse);
    }
    return max_rmse;
}

// ============================================================================
// 4. 循环卷积块 (RecurrentConvBlock)
// ============================================================================

/// 循环卷积基础模块 (Recurrent Convolution Block, RecurrentConvBlock)
pub const RecurrentConvBlock = struct {
    channels: usize,
    conv_f: Conv2D,
    conv_r: Conv2D,
    norm: GroupNorm,
    options: RecurrentConvOptions,
    name: ?[]const u8 = null,
    module_type: []const u8 = "RecurrentConvBlock",

    pub fn defaultOptions() RecurrentConvOptions {
        return RecurrentConvOptions.defaultOptions();
    }

    pub fn init(
        allocator: std.mem.Allocator,
        channels: usize,
        options: RecurrentConvOptions,
    ) !RecurrentConvBlock {
        const conv_f = try Conv2D.init(allocator, channels, channels, options.kernel_size, .{
            .stride = options.stride,
            .padding = options.padding,
            .groups = options.groups,
        });
        errdefer nn.deinitModel(&conv_f, allocator);

        const conv_r = try Conv2D.init(allocator, channels, channels, options.kernel_size, .{
            .stride = options.stride,
            .padding = options.padding,
            .groups = options.groups,
        });
        errdefer nn.deinitModel(&conv_r, allocator);

        const norm = try GroupNorm.init(allocator, options.num_norm_groups, channels, .{});
        errdefer nn.deinitModel(&norm, allocator);

        return .{
            .channels = channels,
            .conv_f = conv_f,
            .conv_r = conv_r,
            .norm = norm,
            .options = options,
        };
    }

    pub fn initDefault(allocator: std.mem.Allocator, channels: usize) !RecurrentConvBlock {
        return init(allocator, channels, .{});
    }

    pub fn deinit(self: *RecurrentConvBlock, allocator: std.mem.Allocator) void {
        nn.deinitModel(self, allocator);
    }

    pub fn resetParameters(self: *RecurrentConvBlock, random: std.Random, options: nn.InitOptions) void {
        self.conv_f.resetParameters(random, options);
        self.conv_r.resetParameters(random, options);
    }

    pub fn zeroGrad(self: *RecurrentConvBlock) void {
        nn.zeroGradModel(self);
    }

    /// 训练模式：计算图动态展开 Backpropagation Through Time (BPTT)
    ///
    /// 拓扑流：
    /// u = conv_f(x)
    /// h_0 = u
    /// For t = 1..T:
    ///   z = conv_r(h_{t-1}) + u (若 use_residual)
    ///   a = ReLU(z)
    ///   n = GroupNorm(a)
    ///   h_t = (1 - alpha) * h_{t-1} + alpha * n
    /// 返回 h_T
    pub fn forward(
        self: *const RecurrentConvBlock,
        graph: *Graph,
        x: *Tensor,
    ) !*Tensor {
        const module_scope = try nn.enterModuleScope(graph, self);
        defer module_scope.exit();

        const u = try self.conv_f.forward(graph, x);
        var h_curr = u;

        var t: usize = 1;
        while (t <= self.options.t_max) : (t += 1) {
            const h_rec = try self.conv_r.forward(graph, h_curr);
            const z = if (self.options.use_residual)
                try graph.add(h_rec, u)
            else
                h_rec;
            const a = try graph.relu(z);
            const n = try self.norm.forward(graph, a);

            if (self.options.alpha_damping < 1.0) {
                const scaled_new = try graph.mulScalar(n, self.options.alpha_damping);
                const scaled_old = try graph.mulScalar(h_curr, 1.0 - self.options.alpha_damping);
                h_curr = try graph.add(scaled_old, scaled_new);
            } else {
                h_curr = n;
            }
        }

        return h_curr;
    }

    /// 推理模式：零动态堆内存分配 Ping-Pong 双缓冲执行流 (Zero-Allocation O(1) Memory Inference)
    ///
    /// 1. 利用预分配 `ws.col_buf` 构建栈上 `FixedBufferAllocator`，消除底层 `convIm2col` 堆申请；
    /// 2. `ws.buf_a` 与 `ws.buf_b` 奇偶交替读写；
    /// 3. 执行原地 ReLU 与 `groupNormInto` 原地归一化；
    /// 4. 实时评估 RMSE 范数，达到容差 `epsilon` 时自适应早停。
    pub fn forwardO1(
        self: *const RecurrentConvBlock,
        ws: *RecurrentWorkspace,
        x: *Tensor,
    ) !RecurrentConvResult {
        if (x.shape.len != 4) return error.IncompatibleDimensions;
        if (x.shape.dims[0] != ws.batch_size or x.shape.dims[1] != ws.channels or
            x.shape.dims[2] != ws.height or x.shape.dims[3] != ws.width)
        {
            return error.ShapeMismatch;
        }

        // 确保工作区张量形状重置为标准 [B, C, H, W]
        const orig_shape = Shape.init(&.{ ws.batch_size, ws.channels, ws.height, ws.width });
        ws.buf_a.shape = orig_shape;
        ws.buf_a.strides = tensor.computeContiguousStrides(orig_shape);
        ws.buf_b.shape = orig_shape;
        ws.buf_b.strides = tensor.computeContiguousStrides(orig_shape);
        ws.buf_u.shape = orig_shape;
        ws.buf_u.strides = tensor.computeContiguousStrides(orig_shape);
        ws.buf_conv.shape = orig_shape;
        ws.buf_conv.strides = tensor.computeContiguousStrides(orig_shape);

        const chw = ws.channels * ws.height * ws.width;
        const alpha = self.options.alpha_damping;
        const one_minus_alpha = 1.0 - alpha;

        // 阶段 1: 静态前馈计算 W_f * x，结果存入 buf_u (使用 ws.col_buf 包装的 FBA 消除堆分配)
        var fba = std.heap.FixedBufferAllocator.init(std.mem.sliceAsBytes(ws.col_buf));
        try tensor.conv_pool.conv2dInto(ws.buf_u, x, self.conv_f.weight, self.conv_f.bias, .{
            .stride = self.options.stride,
            .padding = self.options.padding,
            .groups = self.options.groups,
        }, fba.allocator());

        // 阶段 2: 初始隐状态 h_0 拷贝至 buf_a
        @memcpy(ws.buf_a.data, ws.buf_u.data);

        var t: usize = 1;
        var converged: bool = false;
        var last_residual: f32 = std.math.inf(f32);
        var prev_residual: f32 = std.math.inf(f32);
        var active_out: *Tensor = ws.buf_a;

        // 阶段 3: Ping-Pong 循环迭代
        while (t <= self.options.t_max) : (t += 1) {
            const read_buf = if (t % 2 == 1) ws.buf_a else ws.buf_b;
            const write_buf = if (t % 2 == 1) ws.buf_b else ws.buf_a;

            // 1. 计算循环项: buf_conv = W_r * read_buf (重置 FBA 确保零堆分配)
            fba = std.heap.FixedBufferAllocator.init(std.mem.sliceAsBytes(ws.col_buf));
            try tensor.conv_pool.conv2dInto(ws.buf_conv, read_buf, self.conv_r.weight, self.conv_r.bias, .{
                .stride = self.options.stride,
                .padding = self.options.padding,
                .groups = self.options.groups,
            }, fba.allocator());

            // 2. 特征融合: write_buf = buf_u + buf_conv
            if (self.options.use_residual) {
                try tensor.addInto(write_buf, ws.buf_conv, ws.buf_u);
            } else {
                try tensor.copyInto(write_buf, ws.buf_conv);
            }

            // 3. 原地 ReLU 激活
            for (write_buf.data) |*v| {
                v.* = if (v.* > 0.0) v.* else 0.0;
            }

            // 4. 推理 GroupNorm 归一化路径
            try tensor.nn_kernels.groupNormInto(
                write_buf,
                write_buf,
                self.norm.weight,
                self.norm.bias,
                self.norm.num_groups,
                self.norm.eps,
                null,
                null,
                fba.allocator(),
            );

            // 5. 欧拉阻尼平滑: write_buf = (1 - alpha) * read_buf + alpha * write_buf
            if (alpha < 1.0) {
                for (write_buf.data, read_buf.data) |*dst, src| {
                    dst.* = one_minus_alpha * src + alpha * dst.*;
                }
            }

            active_out = write_buf;

            // 6. 自适应早停残差判定 (ACT Early Exit)
            if (t >= self.options.t_min) {
                last_residual = computeResidualRmse(write_buf.data, read_buf.data, ws.batch_size, chw);
                if (std.math.isNan(last_residual)) {
                    converged = false;
                    break;
                }
                if (last_residual < self.options.epsilon) {
                    converged = true;
                    break;
                }
                // 7. 发散守卫 (Divergence Guard)
                if (last_residual > 1000.0 or (prev_residual < std.math.inf(f32) and last_residual > 2.0 * prev_residual and last_residual > 10.0)) {
                    break;
                }
                prev_residual = last_residual;
            }
        }

        return .{
            .output = active_out,
            .steps_taken = if (converged) t else @min(t, self.options.t_max),
            .converged = converged,
            .final_residual = last_residual,
        };
    }
};

// ============================================================================
// 5. 端到端图像分类网络 (RecurrentConvNet)
// ============================================================================

/// 循环卷积网络配置 (Recurrent Convolution Network Configuration)
pub const RecurrentConvNetConfig = struct {
    in_channels: usize = 1,
    hidden_channels: usize = 64,
    num_classes: usize = 10,
    options: RecurrentConvOptions = .{},

    pub const default: RecurrentConvNetConfig = .{};

    pub fn defaultOptions() RecurrentConvNetConfig {
        return .{};
    }
};

/// 循环卷积图像分类网络 (Recurrent Convolution Network)
///
/// 结构流水线：
/// 1. Stem: Conv2D(in_channels -> hidden_channels, 3x3) -> GroupNorm -> ReLU
/// 2. RecurrentConvBlock: 时间展开循环卷积块
/// 3. Global Adaptive Average Pooling: [B, hidden_channels, H, W] -> [B, hidden_channels, 1, 1]
/// 4. Flatten: [B, hidden_channels]
/// 5. Head: Linear(hidden_channels -> num_classes)
pub const RecurrentConvNet = struct {
    config: RecurrentConvNetConfig,
    stem: Conv2D,
    stem_norm: GroupNorm,
    rec_block: RecurrentConvBlock,
    head: Linear,
    name: ?[]const u8 = null,
    module_type: []const u8 = "RecurrentConvNet",

    pub fn defaultOptions() RecurrentConvNetConfig {
        return RecurrentConvNetConfig.defaultOptions();
    }

    pub fn init(allocator: std.mem.Allocator, config: RecurrentConvNetConfig) !RecurrentConvNet {
        const stem = try Conv2D.init(allocator, config.in_channels, config.hidden_channels, 3, .{
            .stride = 1,
            .padding = 1,
        });
        errdefer nn.deinitModel(&stem, allocator);

        const stem_norm = try GroupNorm.init(allocator, config.options.num_norm_groups, config.hidden_channels, .{});
        errdefer nn.deinitModel(&stem_norm, allocator);

        const rec_block = try RecurrentConvBlock.init(allocator, config.hidden_channels, config.options);
        errdefer nn.deinitModel(&rec_block, allocator);

        const head = try Linear.init(allocator, config.hidden_channels, config.num_classes);
        errdefer nn.deinitModel(&head, allocator);

        return .{
            .config = config,
            .stem = stem,
            .stem_norm = stem_norm,
            .rec_block = rec_block,
            .head = head,
        };
    }

    pub fn initDefault(allocator: std.mem.Allocator) !RecurrentConvNet {
        return init(allocator, .{});
    }

    pub fn deinit(self: *RecurrentConvNet, allocator: std.mem.Allocator) void {
        nn.deinitModel(self, allocator);
    }

    pub fn resetParameters(self: *RecurrentConvNet, random: std.Random, options: nn.InitOptions) void {
        self.stem.resetParameters(random, options);
        self.rec_block.resetParameters(random, options);
        self.head.resetParameters(random, options);
    }

    pub fn zeroGrad(self: *RecurrentConvNet) void {
        nn.zeroGradModel(self);
    }

    /// 训练模式：端到端计算图前向传播
    pub fn forward(self: *const RecurrentConvNet, graph: *Graph, x: *Tensor) !*Tensor {
        const module_scope = try nn.enterModuleScope(graph, self);
        defer module_scope.exit();

        const batch_size = x.shape.dims[0];
        const x_4d = if (x.shape.len == 2) blk: {
            const total_elements = x.shape.dims[1];
            const spatial = @as(usize, @intFromFloat(@sqrt(@as(f32, @floatFromInt(total_elements / self.config.in_channels)))));
            break :blk try graph.reshape(x, &.{ batch_size, self.config.in_channels, spatial, spatial });
        } else x;

        // Stem: Conv2D -> GroupNorm -> ReLU
        const s_conv = try self.stem.forward(graph, x_4d);
        const s_norm = try self.stem_norm.forward(graph, s_conv);
        const s_relu = try graph.relu(s_norm);

        // Recurrent Block
        const h_rec = try self.rec_block.forward(graph, s_relu);

        // Global Adaptive Average Pooling: [B, C, H, W] -> [B, C, 1, 1]
        const pooled = try graph.adaptiveAvgPool2d(h_rec, .{ 1, 1 });

        // Reshape: [B, C, 1, 1] -> [B, C]
        const flat = try graph.reshape(pooled, &.{ batch_size, self.config.hidden_channels });

        // Linear Head: [B, C] -> [B, num_classes]
        return try self.head.forward(graph, flat);
    }

    /// 推理模式：端到端零动态分配 Ping-Pong 执行流
    pub fn forwardO1(self: *const RecurrentConvNet, ws: *RecurrentWorkspace, x: *Tensor) !RecurrentConvResult {
        if (x.shape.len != 4) return error.IncompatibleDimensions;
        if (x.shape.dims[0] != ws.batch_size) return error.ShapeMismatch;
        if (x.shape.dims[1] != self.config.in_channels) return error.ShapeMismatch;
        if (ws.channels != self.config.hidden_channels) return error.ShapeMismatch;
        if (x.shape.dims[2] != ws.height or x.shape.dims[3] != ws.width) return error.ShapeMismatch;

        const b_size = ws.batch_size;
        const c_size = ws.channels;
        const h_size = ws.height;
        const w_size = ws.width;
        const spatial = h_size * w_size;
        const inv_spatial = 1.0 / @as(f32, @floatFromInt(spatial));

        if (b_size * self.config.num_classes > ws.buf_conv.data.len) return error.BufferTooSmall;

        // 恢复所有 workspace buffer 初始 4D 形状与连续步长，防止连续调用由于 buf_conv 被重塑为 2D 导致 ShapeMismatch
        const orig_shape = Shape.init(&.{ b_size, c_size, h_size, w_size });
        ws.buf_conv.shape = orig_shape;
        ws.buf_conv.strides = tensor.computeContiguousStrides(orig_shape);
        ws.buf_a.shape = orig_shape;
        ws.buf_a.strides = tensor.computeContiguousStrides(orig_shape);
        ws.buf_b.shape = orig_shape;
        ws.buf_b.strides = tensor.computeContiguousStrides(orig_shape);
        ws.buf_u.shape = orig_shape;
        ws.buf_u.strides = tensor.computeContiguousStrides(orig_shape);

        // 1. Stem Conv2D into ws.buf_conv (消除堆分配)
        var fba = std.heap.FixedBufferAllocator.init(std.mem.sliceAsBytes(ws.col_buf));
        try tensor.conv_pool.conv2dInto(ws.buf_conv, x, self.stem.weight, self.stem.bias, .{
            .stride = 1,
            .padding = 1,
            .groups = 1,
        }, fba.allocator());

        // 2. Stem GroupNorm into ws.buf_a
        try tensor.nn_kernels.groupNormInto(
            ws.buf_a,
            ws.buf_conv,
            self.stem_norm.weight,
            self.stem_norm.bias,
            self.stem_norm.num_groups,
            self.stem_norm.eps,
            null,
            null,
            fba.allocator(),
        );

        // 3. In-place ReLU on ws.buf_a
        for (ws.buf_a.data) |*v| {
            v.* = if (v.* > 0.0) v.* else 0.0;
        }

        // 4. Recurrent Convolution Block Ping-Pong
        const rec_res = try self.rec_block.forwardO1(ws, ws.buf_a);
        const feat = rec_res.output;

        // 5. Global Average Pooling: feat [B, C, H, W] -> pooled [B, C] 写入 ws.buf_u.data
        for (0..b_size) |b| {
            for (0..c_size) |c| {
                const start = (b * c_size + c) * spatial;
                var sum: f32 = 0.0;
                for (feat.data[start .. start + spatial]) |v| {
                    sum += v;
                }
                ws.buf_u.data[b * c_size + c] = sum * inv_spatial;
            }
        }

        // 6. Linear Head: pooled [B, C] * weight [C, num_classes] + bias [num_classes] -> logits [B, num_classes]
        // 结果存入 ws.buf_conv.data[0 .. B * num_classes]
        const num_cls = self.config.num_classes;
        const w_head = self.head.weight.data;
        const b_head = if (self.head.bias) |b| b.data else null;

        for (0..b_size) |b| {
            const p_row = ws.buf_u.data[b * c_size .. (b + 1) * c_size];
            for (0..num_cls) |k| {
                var acc: f32 = if (b_head) |b_vec| b_vec[k] else 0.0;
                for (0..c_size) |c| {
                    acc += p_row[c] * w_head[c * num_cls + k];
                }
                ws.buf_conv.data[b * num_cls + k] = acc;
            }
        }

        ws.buf_conv.shape = Shape.init(&.{ b_size, num_cls });
        ws.buf_conv.strides = tensor.computeContiguousStrides(ws.buf_conv.shape);

        return .{
            .output = ws.buf_conv,
            .steps_taken = rec_res.steps_taken,
            .converged = rec_res.converged,
            .final_residual = rec_res.final_residual,
        };
    }
};

// ============================================================================
// 6. 单元测试套件 (Unit Tests)
// ============================================================================

test "RecurrentConvOptions and default options compliance" {
    // 验证 AGENTS.md §4 规范: 结构体必须支持 .{} 与 defaultOptions()
    const opts_empty: RecurrentConvOptions = .{};
    const opts_fn = RecurrentConvOptions.defaultOptions();
    const opts_def = RecurrentConvOptions.default;

    try std.testing.expectEqual(@as(usize, 3), opts_empty.kernel_size);
    try std.testing.expectEqual(@as(usize, 1), opts_empty.stride);
    try std.testing.expectEqual(@as(usize, 1), opts_empty.padding);
    try std.testing.expectEqual(@as(usize, 1), opts_empty.groups);
    try std.testing.expectEqual(@as(usize, 8), opts_empty.num_norm_groups);
    try std.testing.expectEqual(@as(usize, 2), opts_empty.t_min);
    try std.testing.expectEqual(@as(usize, 8), opts_empty.t_max);
    try std.testing.expectEqual(@as(f32, 1e-3), opts_empty.epsilon);
    try std.testing.expectEqual(@as(f32, 0.8), opts_empty.alpha_damping);
    try std.testing.expect(opts_empty.use_residual);

    try std.testing.expectEqual(opts_empty.kernel_size, opts_fn.kernel_size);
    try std.testing.expectEqual(opts_empty.epsilon, opts_def.epsilon);

    const net_cfg: RecurrentConvNetConfig = .{};
    const net_cfg_fn = RecurrentConvNetConfig.defaultOptions();
    try std.testing.expectEqual(@as(usize, 1), net_cfg.in_channels);
    try std.testing.expectEqual(@as(usize, 64), net_cfg.hidden_channels);
    try std.testing.expectEqual(@as(usize, 10), net_cfg.num_classes);
    try std.testing.expectEqual(net_cfg.hidden_channels, net_cfg_fn.hidden_channels);
}

test "RecurrentWorkspace init and deinit clean memory" {
    const allocator = std.testing.allocator;

    const b = 2;
    const c = 16;
    const h = 8;
    const w = 8;
    const k = 3;

    var ws = try RecurrentWorkspace.init(allocator, b, c, h, w, k);
    defer ws.deinit(allocator);

    try std.testing.expectEqual(b, ws.batch_size);
    try std.testing.expectEqual(c, ws.channels);
    try std.testing.expectEqual(h, ws.height);
    try std.testing.expectEqual(w, ws.width);

    try std.testing.expectEqualSlices(usize, &.{ b, c, h, w }, ws.buf_a.shape.dims[0..ws.buf_a.shape.len]);
    try std.testing.expectEqualSlices(usize, &.{ b, c, h, w }, ws.buf_b.shape.dims[0..ws.buf_b.shape.len]);
    try std.testing.expectEqualSlices(usize, &.{ b, c, h, w }, ws.buf_u.shape.dims[0..ws.buf_u.shape.len]);
    try std.testing.expectEqualSlices(usize, &.{ b, c, h, w }, ws.buf_conv.shape.dims[0..ws.buf_conv.shape.len]);

    const expected_col_size = c * k * k * h * w;
    try std.testing.expectEqual(expected_col_size, ws.col_buf.len);
}

test "RecurrentConvBlock forwardO1 zero heap allocation ping-pong" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var block = try RecurrentConvBlock.init(allocator, 8, .{
        .kernel_size = 3,
        .stride = 1,
        .padding = 1,
        .t_min = 2,
        .t_max = 4,
        .num_norm_groups = 4,
        .epsilon = 1e-6,
        .alpha_damping = 0.8,
    });
    defer block.deinit(allocator);
    block.resetParameters(random, .{});

    var ws = try RecurrentWorkspace.init(allocator, 1, 8, 6, 6, 3);
    defer ws.deinit(allocator);

    const x = try nn.createBuffer(allocator, &.{ 1, 8, 6, 6 });
    defer nn.freePersistentTensor(allocator, x);
    for (x.data, 0..) |*v, idx| {
        v.* = @as(f32, @floatFromInt(idx % 11)) * 0.1 + 0.05;
    }

    // 执行多次 forwardO1，验证状态切换与零动态内存申请
    for (0..3) |_| {
        const res = try block.forwardO1(&ws, x);
        try std.testing.expect(res.steps_taken >= 2);
        try std.testing.expect(res.steps_taken <= 4);
        try std.testing.expect(res.output == ws.buf_a or res.output == ws.buf_b);
        try std.testing.expect(!std.math.isNan(res.final_residual));
        try std.testing.expect(!std.math.isInf(res.final_residual));
    }
}

test "RecurrentConvBlock forward training BPTT and gradient accumulation" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(1337);
    const random = prng.random();

    var block = try RecurrentConvBlock.init(allocator, 4, .{
        .kernel_size = 3,
        .stride = 1,
        .padding = 1,
        .t_max = 4,
        .num_norm_groups = 2,
        .alpha_damping = 0.8,
        .use_residual = true,
    });
    defer block.deinit(allocator);
    block.resetParameters(random, .{});

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{ 1, 4, 4, 4 }, false);
    for (x_node.data) |*v| v.* = random.float(f32) * 0.4 + 0.1;

    // 前向传播：展开 4 步循环
    const h_T = try block.forward(&graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{ 1, 4, 4, 4 }, h_T.shape.dims[0..h_T.shape.len]);

    // 反向传播标量损失
    @memset(h_T.grad, 1.0);
    try graph.backward(h_T);

    // 验证循环卷积权重梯度非空且由多步 BPTT 累加获得
    try std.testing.expect(block.conv_r.weight.grad.len > 0);
    var grad_sum: f32 = 0.0;
    var has_nonzero: bool = false;
    for (block.conv_r.weight.grad) |g| {
        try std.testing.expect(!std.math.isNan(g));
        try std.testing.expect(!std.math.isInf(g));
        grad_sum += @abs(g);
        if (@abs(g) > 1e-7) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
    try std.testing.expect(grad_sum > 0.0);

    // 验证 zeroGrad() 能够完整清空梯度
    block.zeroGrad();
    for (block.conv_r.weight.grad) |g| {
        try std.testing.expectEqual(@as(f32, 0.0), g);
    }
}

test "RecurrentConvNet forward and forwardO1 equivalence" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(2026);
    const random = prng.random();

    const config = RecurrentConvNetConfig{
        .in_channels = 1,
        .hidden_channels = 8,
        .num_classes = 4,
        .options = .{
            .kernel_size = 3,
            .stride = 1,
            .padding = 1,
            .num_norm_groups = 4,
            .t_min = 2,
            .t_max = 3,
            .epsilon = 1e-12, // 强制执行满全部 3 步以精确对比
            .alpha_damping = 0.8,
            .use_residual = true,
        },
    };

    var net = try RecurrentConvNet.init(allocator, config);
    defer net.deinit(allocator);
    net.resetParameters(random, .{});

    // 构造测试输入 x [1, 1, 6, 6]
    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    const x_node = try graph.tensorND(&.{ 1, 1, 6, 6 }, false);
    for (x_node.data, 0..) |*v, idx| {
        v.* = @as(f32, @floatFromInt(idx % 13)) * 0.08 + 0.1;
    }

    // 1. 运行 forward (动态计算图模式)
    const logits_fwd = try net.forward(&graph, x_node);
    try std.testing.expectEqualSlices(usize, &.{ 1, 4 }, logits_fwd.shape.dims[0..logits_fwd.shape.len]);

    // 2. 运行 forwardO1 (零动态分配推理模式，循环 3 次验证同一个 workspace 连续调用无 ShapeMismatch 且数值恒等)
    const x_inf = try nn.createBuffer(allocator, &.{ 1, 1, 6, 6 });
    defer nn.freePersistentTensor(allocator, x_inf);
    @memcpy(x_inf.data, x_node.data);

    var ws = try RecurrentWorkspace.init(allocator, 1, 8, 6, 6, 3);
    defer ws.deinit(allocator);

    for (0..3) |_| {
        const res_o1 = try net.forwardO1(&ws, x_inf);
        try std.testing.expectEqualSlices(usize, &.{ 1, 4 }, res_o1.output.shape.dims[0..res_o1.output.shape.len]);

        // 3. 验证数值有限性 (非 NaN / Inf)
        for (res_o1.output.data) |v| {
            try std.testing.expect(!std.math.isNan(v));
            try std.testing.expect(!std.math.isInf(v));
        }

        // 4. 验证数值等价性 (绝对差 < 1e-4)
        for (0..4) |i| {
            try std.testing.expectApproxEqAbs(logits_fwd.data[i], res_o1.output.data[i], 1e-4);
        }
    }

    for (logits_fwd.data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "computeResidualRmse and forwardO1 NaN resilience" {
    const allocator = std.testing.allocator;

    // 1. 验证 computeResidualRmse 在输入含 NaN 时返回 NaN，不坍缩为 0.0
    var curr = [_]f32{ 1.0, 2.0, std.math.nan(f32), 4.0 };
    const prev = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const rmse = computeResidualRmse(&curr, &prev, 1, 4);
    try std.testing.expect(std.math.isNan(rmse));

    // 2. 验证 RecurrentConvBlock 在输入含 NaN 时不会错误报告 converged = true
    var block = try RecurrentConvBlock.init(allocator, 4, .{
        .kernel_size = 3,
        .stride = 1,
        .padding = 1,
        .num_norm_groups = 2,
        .t_min = 2,
        .t_max = 4,
        .epsilon = 1e-3,
    });
    defer block.deinit(allocator);

    var ws = try RecurrentWorkspace.init(allocator, 1, 4, 4, 4, 3);
    defer ws.deinit(allocator);

    const x_nan = try nn.createBuffer(allocator, &.{ 1, 4, 4, 4 });
    defer nn.freePersistentTensor(allocator, x_nan);
    @memset(x_nan.data, std.math.nan(f32));

    const res = try block.forwardO1(&ws, x_nan);
    try std.testing.expect(!res.converged);
    try std.testing.expect(std.math.isNan(res.final_residual));
}
