const std = @import("std");
const nn = @import("nn.zig");
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

pub const OPTIMIZER_MAGIC = [4]u8{ 'Z', 'N', 'N', 'O' };
pub const OPTIMIZER_VERSION: u32 = 2;

pub const OptimizerTypeTag = enum(u32) {
    sgd = 0,
    adam = 1,
    adamw = 2,
};

pub const CheckpointError = error{
    InvalidCheckpointMagic,
    UnsupportedCheckpointVersion,
    OptimizerTypeMismatch,
    ParamCountMismatch,
    BufferSizeMismatch,
    CheckpointSizeMismatch,
    InvalidCheckpointFlag,
};

// =========================================================================
// 检查点 (Checkpoint) 公共读写
// =========================================================================
//
// 文件布局 (所有多字节字段均为小端序，f32 按其 IEEE 754 位模式写成小端 u32)：
//   magic[4] | version u32 | type_tag u32 | step u64 | lr f32 | param_count u64
//   [仅 SGD: has_velocity u8 (0 或 1)]
//   每个参数: param_len u64 | 状态缓冲 0 (param_len 个 f32) | 状态缓冲 1 | ...
//
// 加载是事务式的：先校验头部与文件总长度，再把全部状态缓冲读入新分配的临时缓冲并逐参数
// 校验长度，所有可能失败的步骤 (读取、校验、分配) 都完成后才一次性替换优化器状态。
// 任何错误都让优化器保持加载前的状态，不会出现学习率、步数或部分缓冲已被覆盖的半更新。

/// 公共头部字节数：magic + version + type_tag + step + lr + param_count
const CHECKPOINT_HEADER_BYTES: u64 = 4 + 4 + 4 + 8 + 4 + 8;

const CheckpointHeader = struct {
    step: u64,
    lr: f32,
};

fn writeU32(writer: *std.Io.Writer, val: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, val, .little);
    try writer.writeAll(&b);
}

fn writeU64(writer: *std.Io.Writer, val: u64) !void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, val, .little);
    try writer.writeAll(&b);
}

fn writeF32(writer: *std.Io.Writer, val: f32) !void {
    try writeU32(writer, @bitCast(val));
}

fn readU32(reader: *std.Io.Reader) !u32 {
    var b: [4]u8 = undefined;
    try reader.readSliceAll(&b);
    return std.mem.readInt(u32, &b, .little);
}

fn readU64(reader: *std.Io.Reader) !u64 {
    var b: [8]u8 = undefined;
    try reader.readSliceAll(&b);
    return std.mem.readInt(u64, &b, .little);
}

fn readF32(reader: *std.Io.Reader) !f32 {
    return @bitCast(try readU32(reader));
}

fn writeCheckpointHeader(writer: *std.Io.Writer, tag: OptimizerTypeTag, header: CheckpointHeader, param_count: usize) !void {
    try writer.writeAll(&OPTIMIZER_MAGIC);
    try writeU32(writer, OPTIMIZER_VERSION);
    try writeU32(writer, @backingInt(tag));
    try writeU64(writer, header.step);
    try writeF32(writer, header.lr);
    try writeU64(writer, @as(u64, param_count));
}

/// 读取并校验公共头部，只返回解析结果，不修改任何优化器状态
fn readCheckpointHeader(reader: *std.Io.Reader, tag: OptimizerTypeTag, param_count: usize) !CheckpointHeader {
    var magic: [4]u8 = undefined;
    try reader.readSliceAll(&magic);
    if (!std.mem.eql(u8, &magic, &OPTIMIZER_MAGIC)) return error.InvalidCheckpointMagic;

    const version = try readU32(reader);
    if (version != OPTIMIZER_VERSION) return error.UnsupportedCheckpointVersion;

    const type_tag = try readU32(reader);
    if (type_tag != @backingInt(tag)) return error.OptimizerTypeMismatch;

    const step = try readU64(reader);
    const lr = try readF32(reader);
    const saved_count = try readU64(reader);
    if (saved_count != param_count) return error.ParamCountMismatch;

    return .{ .step = step, .lr = lr };
}

/// 状态区字节数：每个参数一个 u64 长度字段，加上 `buffers_per_param` 个与参数等长的 f32 缓冲
fn checkpointStateBytes(params: []const *Tensor, buffers_per_param: usize) u64 {
    var total: u64 = 0;
    for (params) |param| {
        total += 8 + @as(u64, buffers_per_param) * @as(u64, param.data.len) * @sizeOf(f32);
    }
    return total;
}

/// 逐参数写入长度字段与各状态缓冲 (`states[k][i]` 为第 i 个参数的第 k 个状态缓冲)，f32 一律按小端序写出
fn writeStateBuffers(writer: *std.Io.Writer, params: []const *Tensor, states: []const []const []f32) !void {
    for (params, 0..) |param, i| {
        try writeU64(writer, @as(u64, param.data.len));
        for (states) |state| {
            const bits: []const u32 = @ptrCast(state[i]);
            try writer.writeSliceEndian(u32, bits, .little);
        }
    }
}

/// 写出完整检查点文件；`states` 的个数必须与 `readCheckpoint` 对该类型推断出的缓冲个数一致
fn writeCheckpoint(
    io: std.Io,
    file_path: []const u8,
    tag: OptimizerTypeTag,
    header: CheckpointHeader,
    params: []const *Tensor,
    states: []const []const []f32,
) !void {
    var file = try std.Io.Dir.cwd().createFile(io, file_path, .{});
    defer file.close(io);

    var buf: [65536]u8 = undefined;
    var file_writer = file.writer(io, &buf);
    const writer = &file_writer.interface;

    try writeCheckpointHeader(writer, tag, header, params.len);
    if (tag == .sgd) {
        const has_vel: u8 = if (states.len == 1) 1 else 0;
        try writer.writeAll(&[_]u8{has_vel});
    }
    try writeStateBuffers(writer, params, states);
    try writer.flush();
}

/// 单个检查点最多携带的状态缓冲组数 (Adam/AdamW 的一阶矩与二阶矩)
const MAX_STATE_BUFFERS = 2;

/// 已完整读取并校验的检查点：状态缓冲存放在新分配的内存中，由调用方在提交阶段接管
const LoadedCheckpoint = struct {
    header: CheckpointHeader,
    /// 有效状态缓冲组数：SGD 由动量标志决定 (0 或 1)，Adam/AdamW 固定为 2
    buffer_count: usize,
    states: [MAX_STATE_BUFFERS][][]f32,

    fn deinit(self: *LoadedCheckpoint, allocator: std.mem.Allocator) void {
        for (self.states[0..self.buffer_count]) |s| freeStates(allocator, s);
        self.buffer_count = 0;
    }
};

/// 读取整个检查点文件并完成全部校验，不修改任何优化器状态。
/// 返回的状态缓冲为新分配的内存 (加载期间峰值内存约为两份优化器状态)，所有权交给调用方；
/// 任何错误都会释放已分配部分。
fn readCheckpoint(
    allocator: std.mem.Allocator,
    io: std.Io,
    file_path: []const u8,
    tag: OptimizerTypeTag,
    params: []const *Tensor,
) !LoadedCheckpoint {
    var file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
    defer file.close(io);
    const file_len = try file.length(io);

    var buf: [65536]u8 = undefined;
    var file_reader = file.reader(io, &buf);
    const reader = &file_reader.interface;

    const header = try readCheckpointHeader(reader, tag, params.len);
    var prefix_bytes: u64 = CHECKPOINT_HEADER_BYTES;
    const buffer_count: usize = switch (tag) {
        .sgd => blk: {
            var flag: [1]u8 = undefined;
            try reader.readSliceAll(&flag);
            prefix_bytes += 1;
            break :blk switch (flag[0]) {
                0 => 0,
                1 => 1,
                else => return error.InvalidCheckpointFlag,
            };
        },
        .adam, .adamw => 2,
    };
    if (file_len != prefix_bytes + checkpointStateBytes(params, buffer_count)) {
        return error.CheckpointSizeMismatch;
    }

    var loaded = LoadedCheckpoint{ .header = header, .buffer_count = 0, .states = undefined };
    errdefer loaded.deinit(allocator);
    for (0..buffer_count) |k| {
        loaded.states[k] = try allocZeroedStates(allocator, params);
        loaded.buffer_count += 1;
    }

    for (params, 0..) |param, i| {
        const param_len = try readU64(reader);
        if (param_len != param.data.len) return error.BufferSizeMismatch;
        for (loaded.states[0..buffer_count]) |state| {
            const bits: []u32 = @ptrCast(state[i]);
            try reader.readSliceEndian(u32, bits, .little);
        }
    }
    return loaded;
}

/// 用新状态替换旧状态并释放旧缓冲 (不会失败，供事务式加载的提交阶段使用)
fn replaceStates(allocator: std.mem.Allocator, target: *[][]f32, new_states: [][]f32) void {
    freeStates(allocator, target.*);
    target.* = new_states;
}

/// 为每个参数分配一块与参数等长、清零的状态缓冲；失败时释放已分配部分
fn allocZeroedStates(allocator: std.mem.Allocator, params: []const *Tensor) ![][]f32 {
    const states = try allocator.alloc([]f32, params.len);
    errdefer allocator.free(states);
    var initialized: usize = 0;
    errdefer for (states[0..initialized]) |s| allocator.free(s);
    for (params) |param| {
        states[initialized] = try allocator.alloc(f32, param.data.len);
        @memset(states[initialized], 0.0);
        initialized += 1;
    }
    return states;
}

fn freeStates(allocator: std.mem.Allocator, states: []const []f32) void {
    for (states) |s| allocator.free(s);
    allocator.free(states);
}

/// 随机梯度下降 (Stochastic Gradient Descent, SGD) 优化器配置
pub const SGDConfig = struct {
    lr: f32 = 0.01,
    momentum: f32 = 0.0,

    pub const default: SGDConfig = .{};
    pub fn defaultConfig() SGDConfig {
        return .{};
    }
};

/// 随机梯度下降优化器 (Stochastic Gradient Descent Optimizer, SGDOptimizer)
pub const SGDOptimizer = struct {
    allocator: std.mem.Allocator,
    params: []*Tensor,
    velocities: ?[][]f32, // Only allocated if momentum > 0
    lr: f32,
    momentum: f32,
    step_count: u64 = 0,

    pub fn initDefault(allocator: std.mem.Allocator, model: anytype) !SGDOptimizer {
        return init(allocator, model, SGDConfig.default);
    }

    pub fn init(allocator: std.mem.Allocator, model: anytype, config: SGDConfig) !SGDOptimizer {
        const params = try nn.parameters(model, allocator);
        nn.warnIfParametersUninitialized(params);
        errdefer allocator.free(params);

        const velocities: ?[][]f32 = if (config.momentum > 0.0) try allocZeroedStates(allocator, params) else null;

        return SGDOptimizer{
            .allocator = allocator,
            .params = params,
            .velocities = velocities,
            .lr = config.lr,
            .momentum = config.momentum,
            .step_count = 0,
        };
    }

    pub fn deinit(self: SGDOptimizer) void {
        if (self.velocities) |v_list| freeStates(self.allocator, v_list);
        self.allocator.free(self.params);
    }

    pub fn getLR(self: SGDOptimizer) f32 {
        return self.lr;
    }

    pub fn setLR(self: *SGDOptimizer, lr: f32) void {
        self.lr = lr;
    }

    pub fn stepWithLR(self: *SGDOptimizer, current_lr: f32) void {
        self.step_count += 1;
        for (self.params, 0..) |param, i| {
            const w = param.data;
            const dw = param.grad;
            if (self.velocities) |v_list| {
                const v = v_list[i];
                for (w, dw, v) |*weight, grad, *vel| {
                    vel.* = self.momentum * vel.* + current_lr * grad;
                    weight.* -= vel.*;
                }
            } else {
                for (w, dw) |*weight, grad| {
                    weight.* -= current_lr * grad;
                }
            }
        }
    }

    pub fn step(self: *SGDOptimizer) void {
        self.stepWithLR(self.lr);
    }

    pub fn saveCheckpoint(self: SGDOptimizer, io: std.Io, file_path: []const u8) !void {
        const header: CheckpointHeader = .{ .step = self.step_count, .lr = self.lr };
        if (self.velocities) |v_list| {
            try writeCheckpoint(io, file_path, .sgd, header, self.params, &.{v_list});
        } else {
            try writeCheckpoint(io, file_path, .sgd, header, self.params, &.{});
        }
    }

    /// 事务式加载：任何错误都让优化器保持加载前的状态
    pub fn loadCheckpoint(self: *SGDOptimizer, io: std.Io, file_path: []const u8) !void {
        const loaded = try readCheckpoint(self.allocator, io, file_path, .sgd, self.params);

        // 提交阶段：以下操作均不会失败
        if (loaded.buffer_count == 1) {
            if (self.velocities) |*v_list| {
                replaceStates(self.allocator, v_list, loaded.states[0]);
            } else {
                self.velocities = loaded.states[0];
            }
        } else if (self.velocities) |v_list| {
            // 检查点不含动量缓冲：清零现有缓冲，使恢复后的状态与检查点一致
            for (v_list) |v| @memset(v, 0.0);
        }
        self.step_count = loaded.header.step;
        self.lr = loaded.header.lr;
    }
};

/// Adam 系列优化器的逐元素更新系数：
///   m = beta1 * m + (1 - beta1) * g
///   v = beta2 * v + (1 - beta2) * g^2
///   w = w - decay * w - step_size * m / (sqrt(v) * denom_scale + eps)
/// 偏差修正 (Bias Correction) 的除法均在每步开始时折算进 `step_size` 与 `denom_scale`，
/// 内层循环不再逐元素做两次除法。
const AdamUpdate = struct {
    beta1: f32,
    one_minus_beta1: f32,
    beta2: f32,
    one_minus_beta2: f32,
    step_size: f32,
    denom_scale: f32,
    eps: f32,
    decay: f32,

    const vec_len = std.simd.suggestVectorLength(f32) orelse 4;
    const Vec = @Vector(vec_len, f32);

    fn apply(self: AdamUpdate, w: []f32, g: []const f32, m: []f32, v: []f32) void {
        std.debug.assert(g.len == w.len and m.len == w.len and v.len == w.len);
        const b1: Vec = @splat(self.beta1);
        const omb1: Vec = @splat(self.one_minus_beta1);
        const b2: Vec = @splat(self.beta2);
        const omb2: Vec = @splat(self.one_minus_beta2);
        const step_size: Vec = @splat(self.step_size);
        const denom_scale: Vec = @splat(self.denom_scale);
        const eps: Vec = @splat(self.eps);
        const keep: Vec = @splat(1.0 - self.decay);

        var i: usize = 0;
        while (i + vec_len <= w.len) : (i += vec_len) {
            const gv: Vec = g[i..][0..vec_len].*;
            const mv = b1 * @as(Vec, m[i..][0..vec_len].*) + omb1 * gv;
            const vv = b2 * @as(Vec, v[i..][0..vec_len].*) + omb2 * (gv * gv);
            const wv: Vec = w[i..][0..vec_len].*;
            m[i..][0..vec_len].* = mv;
            v[i..][0..vec_len].* = vv;
            w[i..][0..vec_len].* = keep * wv - step_size * mv / (@sqrt(vv) * denom_scale + eps);
        }
        while (i < w.len) : (i += 1) {
            const gi = g[i];
            const mi = self.beta1 * m[i] + self.one_minus_beta1 * gi;
            const vi = self.beta2 * v[i] + self.one_minus_beta2 * (gi * gi);
            m[i] = mi;
            v[i] = vi;
            w[i] = (1.0 - self.decay) * w[i] - self.step_size * mi / (@sqrt(vi) * self.denom_scale + self.eps);
        }
    }
};

/// 自适应矩估计 (Adaptive Moment Estimation, Adam) 优化器配置
pub const AdamConfig = struct {
    lr: f32 = 0.001,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,

    pub const default: AdamConfig = .{};
    pub fn defaultConfig() AdamConfig {
        return .{};
    }
};

/// 自适应矩估计优化器 (Adaptive Moment Estimation Optimizer, AdamOptimizer)
pub const AdamOptimizer = struct {
    allocator: std.mem.Allocator,
    params: []*Tensor,
    m: [][]f32,
    v: [][]f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    /// 已执行的更新步数；整数计数避免 f32 步数在 2^24 步后停止增长
    step_count: u64 = 0,

    pub fn initDefault(allocator: std.mem.Allocator, model: anytype) !AdamOptimizer {
        return init(allocator, model, AdamConfig.default);
    }

    pub fn init(allocator: std.mem.Allocator, model: anytype, config: AdamConfig) !AdamOptimizer {
        const params = try nn.parameters(model, allocator);
        nn.warnIfParametersUninitialized(params);
        errdefer allocator.free(params);

        const m = try allocZeroedStates(allocator, params);
        errdefer freeStates(allocator, m);
        const v = try allocZeroedStates(allocator, params);

        return AdamOptimizer{
            .allocator = allocator,
            .params = params,
            .m = m,
            .v = v,
            .lr = config.lr,
            .beta1 = config.beta1,
            .beta2 = config.beta2,
            .eps = config.eps,
            .step_count = 0,
        };
    }

    pub fn deinit(self: AdamOptimizer) void {
        freeStates(self.allocator, self.m);
        freeStates(self.allocator, self.v);
        self.allocator.free(self.params);
    }

    pub fn getLR(self: AdamOptimizer) f32 {
        return self.lr;
    }

    pub fn setLR(self: *AdamOptimizer, lr: f32) void {
        self.lr = lr;
    }

    pub fn stepWithLR(self: *AdamOptimizer, current_lr: f32) void {
        self.step_count += 1;
        const t: f32 = @floatFromInt(self.step_count);
        const correction1 = 1.0 - std.math.pow(f32, self.beta1, t);
        const correction2 = 1.0 - std.math.pow(f32, self.beta2, t);
        // 原论文的高效形式：lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)，eps 不做偏差修正
        const update = AdamUpdate{
            .beta1 = self.beta1,
            .one_minus_beta1 = 1.0 - self.beta1,
            .beta2 = self.beta2,
            .one_minus_beta2 = 1.0 - self.beta2,
            .step_size = current_lr * @sqrt(correction2) / correction1,
            .denom_scale = 1.0,
            .eps = self.eps,
            .decay = 0.0,
        };

        for (self.params, self.m, self.v) |param, m_t, v_t| {
            update.apply(param.data, param.grad, m_t, v_t);
        }
    }

    pub fn step(self: *AdamOptimizer) void {
        self.stepWithLR(self.lr);
    }

    pub fn saveCheckpoint(self: AdamOptimizer, io: std.Io, file_path: []const u8) !void {
        const header: CheckpointHeader = .{ .step = self.step_count, .lr = self.lr };
        try writeCheckpoint(io, file_path, .adam, header, self.params, &.{ self.m, self.v });
    }

    /// 事务式加载：任何错误都让优化器保持加载前的状态
    pub fn loadCheckpoint(self: *AdamOptimizer, io: std.Io, file_path: []const u8) !void {
        const loaded = try readCheckpoint(self.allocator, io, file_path, .adam, self.params);

        // 提交阶段：以下操作均不会失败
        replaceStates(self.allocator, &self.m, loaded.states[0]);
        replaceStates(self.allocator, &self.v, loaded.states[1]);
        self.step_count = loaded.header.step;
        self.lr = loaded.header.lr;
    }
};

/// 解耦权重衰减自适应矩估计 (Adam with Decoupled Weight Decay, AdamW) 配置
pub const AdamWConfig = struct {
    lr: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.95,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.01,

    pub const default: AdamWConfig = .{};
    pub fn defaultConfig() AdamWConfig {
        return .{};
    }
};

/// 具备解耦权重衰减的自适应矩估计优化器 (Adam with Decoupled Weight Decay, AdamW)
pub const AdamWOptimizer = struct {
    allocator: std.mem.Allocator,
    params: []*Tensor,
    m: [][]f32,
    v: [][]f32,
    config: AdamWConfig,
    step_count: u64,

    pub fn initDefault(allocator: std.mem.Allocator, model: anytype) !AdamWOptimizer {
        return init(allocator, model, AdamWConfig.default);
    }

    pub fn init(allocator: std.mem.Allocator, model: anytype, config: AdamWConfig) !AdamWOptimizer {
        const params = try nn.parameters(model, allocator);
        nn.warnIfParametersUninitialized(params);
        errdefer allocator.free(params);

        const m = try allocZeroedStates(allocator, params);
        errdefer freeStates(allocator, m);
        const v = try allocZeroedStates(allocator, params);

        return AdamWOptimizer{
            .allocator = allocator,
            .params = params,
            .m = m,
            .v = v,
            .config = config,
            .step_count = 0,
        };
    }

    pub fn deinit(self: AdamWOptimizer) void {
        freeStates(self.allocator, self.m);
        freeStates(self.allocator, self.v);
        self.allocator.free(self.params);
    }

    pub fn getLR(self: AdamWOptimizer) f32 {
        return self.config.lr;
    }

    pub fn setLR(self: *AdamWOptimizer, lr: f32) void {
        self.config.lr = lr;
    }

    pub fn stepWithLR(self: *AdamWOptimizer, current_lr: f32) void {
        self.step_count += 1;
        const t: f32 = @floatFromInt(self.step_count);
        const beta1 = self.config.beta1;
        const beta2 = self.config.beta2;

        const bias_correction1 = 1.0 - std.math.pow(f32, beta1, t);
        const bias_correction2 = 1.0 - std.math.pow(f32, beta2, t);

        // w = w - lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)，其中
        // m_hat = m / bias_correction1，v_hat = v / bias_correction2
        const update = AdamUpdate{
            .beta1 = beta1,
            .one_minus_beta1 = 1.0 - beta1,
            .beta2 = beta2,
            .one_minus_beta2 = 1.0 - beta2,
            .step_size = current_lr / bias_correction1,
            .denom_scale = 1.0 / @sqrt(bias_correction2),
            .eps = self.config.eps,
            .decay = current_lr * self.config.weight_decay,
        };

        for (self.params, self.m, self.v) |param, m_t, v_t| {
            update.apply(param.data, param.grad, m_t, v_t);
        }
    }

    pub fn step(self: *AdamWOptimizer) void {
        self.stepWithLR(self.config.lr);
    }

    pub fn saveCheckpoint(self: AdamWOptimizer, io: std.Io, file_path: []const u8) !void {
        const header: CheckpointHeader = .{ .step = self.step_count, .lr = self.config.lr };
        try writeCheckpoint(io, file_path, .adamw, header, self.params, &.{ self.m, self.v });
    }

    /// 事务式加载：任何错误都让优化器保持加载前的状态
    pub fn loadCheckpoint(self: *AdamWOptimizer, io: std.Io, file_path: []const u8) !void {
        const loaded = try readCheckpoint(self.allocator, io, file_path, .adamw, self.params);

        // 提交阶段：以下操作均不会失败
        replaceStates(self.allocator, &self.m, loaded.states[0]);
        replaceStates(self.allocator, &self.v, loaded.states[1]);
        self.step_count = loaded.header.step;
        self.config.lr = loaded.header.lr;
    }
};

// =========================================================================
// 学习率调度器 (Learning Rate, LR Schedulers)
// =========================================================================

/// 带有线性预热 (Linear Warmup) 的余弦退火学习率调度器
pub const CosineScheduler = struct {
    max_lr: f32 = 1e-3,
    min_lr: f32 = 1e-5,
    warmup_steps: u64 = 100,
    max_steps: u64 = 1000,

    pub const default: CosineScheduler = .{};
    pub fn defaultOptions() CosineScheduler {
        return .{};
    }

    pub fn initDefault() CosineScheduler {
        return .{};
    }

    pub fn init(max_lr: f32, min_lr: f32, warmup_steps: u64, max_steps: u64) CosineScheduler {
        return .{
            .max_lr = max_lr,
            .min_lr = min_lr,
            .warmup_steps = warmup_steps,
            .max_steps = max_steps,
        };
    }

    pub fn getLR(self: CosineScheduler, current_step: u64) f32 {
        if (self.warmup_steps > 0 and current_step < self.warmup_steps) {
            const progress = @as(f32, @floatFromInt(current_step)) / @as(f32, @floatFromInt(self.warmup_steps));
            return self.max_lr * progress;
        }
        if (current_step >= self.max_steps) {
            return self.min_lr;
        }

        const current_f = @as(f32, @floatFromInt(current_step - self.warmup_steps));
        const total_f = @as(f32, @floatFromInt(self.max_steps - self.warmup_steps));
        const progress = current_f / total_f;
        const cosine_decay = 0.5 * (1.0 + @cos(progress * std.math.pi));

        return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay;
    }
};

/// 固定步长阶梯衰减学习率调度器 (Step Learning Rate, StepLR)
pub const StepLRScheduler = struct {
    base_lr: f32 = 1e-3,
    step_size: u64 = 100,
    gamma: f32 = 0.1,

    pub const default: StepLRScheduler = .{};
    pub fn defaultOptions() StepLRScheduler {
        return .{};
    }

    pub fn initDefault() StepLRScheduler {
        return .{};
    }

    pub fn init(base_lr: f32, step_size: u64, gamma: f32) StepLRScheduler {
        std.debug.assert(step_size > 0);
        return .{
            .base_lr = base_lr,
            .step_size = step_size,
            .gamma = gamma,
        };
    }

    pub fn getLR(self: StepLRScheduler, current_step: u64) f32 {
        // step_size 为 0 时视为每步衰减一次，避免整数除零
        const decays = current_step / @max(self.step_size, 1);
        const factor = std.math.pow(f32, self.gamma, @as(f32, @floatFromInt(decays)));
        return self.base_lr * factor;
    }
};

/// 纯线性预热学习率调度器 (Linear Warmup)
pub const LinearWarmupScheduler = struct {
    start_lr: f32 = 0.0,
    target_lr: f32 = 1e-3,
    warmup_steps: u64 = 100,

    pub const default: LinearWarmupScheduler = .{};
    pub fn defaultOptions() LinearWarmupScheduler {
        return .{};
    }

    pub fn initDefault() LinearWarmupScheduler {
        return .{};
    }

    pub fn init(start_lr: f32, target_lr: f32, warmup_steps: u64) LinearWarmupScheduler {
        std.debug.assert(warmup_steps > 0);
        return .{
            .start_lr = start_lr,
            .target_lr = target_lr,
            .warmup_steps = warmup_steps,
        };
    }

    pub fn getLR(self: LinearWarmupScheduler, current_step: u64) f32 {
        if (current_step >= self.warmup_steps) {
            return self.target_lr;
        }
        const progress = @as(f32, @floatFromInt(current_step)) / @as(f32, @floatFromInt(self.warmup_steps));
        return self.start_lr + (self.target_lr - self.start_lr) * progress;
    }
};

/// 指数衰减学习率调度器 (Exponential Learning Rate, ExponentialLR)
pub const ExponentialLRScheduler = struct {
    base_lr: f32 = 1e-3,
    gamma: f32 = 0.95,

    pub const default: ExponentialLRScheduler = .{};
    pub fn defaultOptions() ExponentialLRScheduler {
        return .{};
    }

    pub fn initDefault() ExponentialLRScheduler {
        return .{};
    }

    pub fn init(base_lr: f32, gamma: f32) ExponentialLRScheduler {
        return .{
            .base_lr = base_lr,
            .gamma = gamma,
        };
    }

    pub fn getLR(self: ExponentialLRScheduler, current_step: u64) f32 {
        return self.base_lr * std.math.pow(f32, self.gamma, @as(f32, @floatFromInt(current_step)));
    }
};

/// 统一的多态学习率调度器 (Learning Rate Scheduler, LRScheduler) 包装
pub const LRScheduler = union(enum) {
    cosine: CosineScheduler,
    step_lr: StepLRScheduler,
    warmup: LinearWarmupScheduler,
    exponential: ExponentialLRScheduler,

    pub const default: LRScheduler = .{ .cosine = .{} };

    pub fn defaultConfig() LRScheduler {
        return default;
    }

    pub fn defaultOptions() LRScheduler {
        return default;
    }

    pub fn getLR(self: LRScheduler, current_step: u64) f32 {
        return switch (self) {
            .cosine => |s| s.getLR(current_step),
            .step_lr => |s| s.getLR(current_step),
            .warmup => |s| s.getLR(current_step),
            .exponential => |s| s.getLR(current_step),
        };
    }

    pub fn step(self: LRScheduler, optimizer: anytype, current_step: u64) f32 {
        const lr = self.getLR(current_step);
        optimizer.setLR(lr);
        return lr;
    }
};

// =========================================================================
// 梯度裁剪 (Gradient Clipping)
// =========================================================================

/// 全局梯度欧几里得二范数 (L2 Norm) 裁剪 (Gradient Norm Clipping)
pub fn clipGradNorm(params: []*Tensor, max_norm: f32) f32 {
    var total_norm_sq: f64 = 0.0;
    for (params) |param| {
        total_norm_sq += sumSquares(param.grad);
    }
    const total_norm: f32 = @floatCast(@sqrt(total_norm_sq));
    if (total_norm > max_norm and total_norm > 0.0) {
        const scale = max_norm / (total_norm + 1e-6);
        for (params) |param| {
            for (param.grad) |*g| {
                g.* *= scale;
            }
        }
    }
    return total_norm;
}

/// 平方和：块内使用两路 SIMD 向量 (`@Vector`) 部分和并行累加 (严格浮点语义下标量累加无法被自动向量化)，
/// 每块结果再以 f64 汇总，避免百万级元素在单个 f32 累加器上累积舍入误差
fn sumSquares(values: []const f32) f64 {
    const vec_len = std.simd.suggestVectorLength(f32) orelse 4;
    const Vec = @Vector(vec_len, f32);
    const block_len = 2 * vec_len * 256;

    var total: f64 = 0.0;
    var i: usize = 0;
    while (i + 2 * vec_len <= values.len) {
        const block_end = i + @min(block_len, (values.len - i) / (2 * vec_len) * (2 * vec_len));
        var acc0: Vec = @splat(0.0);
        var acc1: Vec = @splat(0.0);
        while (i < block_end) : (i += 2 * vec_len) {
            const a: Vec = values[i..][0..vec_len].*;
            const b: Vec = values[i + vec_len ..][0..vec_len].*;
            acc0 += a * a;
            acc1 += b * b;
        }
        const block_sum: @Vector(vec_len, f64) = @floatCast(acc0 + acc1);
        total += @reduce(.Add, block_sum);
    }
    while (i < values.len) : (i += 1) {
        const x: f64 = values[i];
        total += x * x;
    }
    return total;
}

/// 梯度分量数值截断 (Gradient Value Clipping)
pub fn clipGradValue(params: []*Tensor, clip_value: f32) void {
    const abs_clip = @abs(clip_value);
    for (params) |param| {
        for (param.grad) |*g| {
            g.* = std.math.clamp(g.*, -abs_clip, abs_clip);
        }
    }
}

/// 梯度裁剪配置类型
pub const GradClipConfig = union(enum) {
    norm: f32,
    value: f32,
    none: void,

    pub const default: GradClipConfig = .{ .none = {} };
    pub fn defaultConfig() GradClipConfig {
        return .{ .none = {} };
    }
};

/// 统一梯度裁剪执行函数
pub fn clipGradients(params: []*Tensor, config: GradClipConfig) ?f32 {
    return switch (config) {
        .norm => |max_norm| clipGradNorm(params, max_norm),
        .value => |clip_val| {
            clipGradValue(params, clip_val);
            return null;
        },
        .none => null,
    };
}

// =========================================================================
// 单元测试
// =========================================================================

test "SGDOptimizer basic and momentum updates" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(42);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    // 1. 测试无动量 SGD
    var opt_plain = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1, .momentum = 0.0 });
    defer opt_plain.deinit();

    linear.weight.data[0] = 1.0;
    linear.weight.grad[0] = 0.5;

    opt_plain.step();
    // w = 1.0 - 0.1 * 0.5 = 0.95
    try testing.expectApproxEqAbs(@as(f32, 0.95), linear.weight.data[0], 1e-5);

    // 2. 测试带动量 SGD
    var opt_mom = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1, .momentum = 0.9 });
    defer opt_mom.deinit();

    linear.weight.data[0] = 1.0;
    linear.weight.grad[0] = 0.5;

    // Step 1: vel = 0.9*0 + 0.1*0.5 = 0.05, w = 1.0 - 0.05 = 0.95
    opt_mom.step();
    try testing.expectApproxEqAbs(@as(f32, 0.95), linear.weight.data[0], 1e-5);

    // Step 2: vel = 0.9*0.05 + 0.1*0.5 = 0.045 + 0.05 = 0.095, w = 0.95 - 0.095 = 0.855
    opt_mom.step();
    try testing.expectApproxEqAbs(@as(f32, 0.855), linear.weight.data[0], 1e-5);
}

test "AdamOptimizer multi-step parameter updates" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(42);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    var opt = try AdamOptimizer.init(allocator, &linear, .{
        .lr = 0.01,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .eps = 1e-8,
    });
    defer opt.deinit();

    linear.weight.data[0] = 1.0;
    linear.weight.grad[0] = 0.2;

    opt.step();
    try testing.expect(linear.weight.data[0] < 1.0);

    const after_step1 = linear.weight.data[0];
    opt.step();
    try testing.expect(linear.weight.data[0] < after_step1);
}

test "AdamWOptimizer weight decay and step" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(42);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    var opt = try AdamWOptimizer.init(allocator, &linear, .{
        .lr = 0.01,
        .beta1 = 0.9,
        .beta2 = 0.95,
        .eps = 1e-8,
        .weight_decay = 0.1,
    });
    defer opt.deinit();

    linear.weight.data[0] = 1.0;
    linear.weight.grad[0] = 0.0; // 即使梯度为 0，解耦权重衰减也应该减小权重

    opt.step();
    try testing.expect(linear.weight.data[0] < 1.0);
}

test "CosineScheduler warmup and decay" {
    const testing = std.testing;
    const sched = CosineScheduler.init(1e-3, 1e-4, 10, 100);

    // Warmup: step 0 -> 0, step 5 -> 0.5 * max_lr, step 10 -> max_lr
    try testing.expectApproxEqAbs(@as(f32, 0.0), sched.getLR(0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 5e-4), sched.getLR(5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1e-3), sched.getLR(10), 1e-6);

    // Midpoint: step 55 -> halfway between max_lr and min_lr
    const mid_lr = sched.getLR(55);
    try testing.expect(mid_lr < 1e-3 and mid_lr > 1e-4);

    // End: step 100 -> min_lr
    try testing.expectApproxEqAbs(@as(f32, 1e-4), sched.getLR(100), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1e-4), sched.getLR(120), 1e-6);
}

test "StepLR and ExponentialLR schedulers" {
    const testing = std.testing;

    const step_sched = StepLRScheduler.init(0.1, 10, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.1), step_sched.getLR(0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), step_sched.getLR(9), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.05), step_sched.getLR(10), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.05), step_sched.getLR(19), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.025), step_sched.getLR(20), 1e-6);

    const exp_sched = ExponentialLRScheduler.init(0.1, 0.9);
    try testing.expectApproxEqAbs(@as(f32, 0.1), exp_sched.getLR(0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.09), exp_sched.getLR(1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.081), exp_sched.getLR(2), 1e-6);
}

test "LinearWarmupScheduler and LRScheduler union" {
    const testing = std.testing;

    const warmup = LinearWarmupScheduler.init(0.01, 0.1, 10);
    try testing.expectApproxEqAbs(@as(f32, 0.01), warmup.getLR(0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.055), warmup.getLR(5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), warmup.getLR(10), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), warmup.getLR(20), 1e-6);

    const sched_union = LRScheduler{ .warmup = warmup };
    try testing.expectApproxEqAbs(@as(f32, 0.055), sched_union.getLR(5), 1e-6);
}

test "clipGradNorm and clipGradValue" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var t1 = try tensor.zeros(allocator, &.{ 2, 2 });
    defer tensor.free(allocator, t1);
    t1.requires_grad = true;
    t1.grad = try allocator.alloc(f32, 4);

    @memcpy(t1.grad, &[_]f32{ 3.0, 4.0, 0.0, 0.0 }); // norm = sqrt(9 + 16) = 5.0

    var params = [_]*Tensor{t1};
    const norm = clipGradNorm(&params, 2.5);
    try testing.expectApproxEqAbs(@as(f32, 5.0), norm, 1e-5);
    // After clipping with max_norm=2.5, norm should be scaled by 2.5/5.0 = 0.5
    try testing.expectApproxEqAbs(@as(f32, 1.5), t1.grad[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), t1.grad[1], 1e-5);

    // Value clipping test
    @memcpy(t1.grad, &[_]f32{ 5.0, -10.0, 0.5, -0.2 });
    clipGradValue(&params, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), t1.grad[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -1.0), t1.grad[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), t1.grad[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -0.2), t1.grad[3], 1e-5);
}

test "Optimizer checkpoint serialization and resumption" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(42);
    var linear = try nn.Linear.init(allocator, 3, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    // 1. Test SGDOptimizer checkpointing
    {
        var opt1 = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.05, .momentum = 0.9 });
        defer opt1.deinit();

        linear.weight.data[0] = 2.0;
        linear.weight.grad[0] = 1.0;
        opt1.step(); // updates momentum buffer and step_count

        const ckpt_path = "test_sgd_ckpt.bin";
        try opt1.saveCheckpoint(testing.io, ckpt_path);

        var opt2 = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.01, .momentum = 0.0 });
        defer opt2.deinit();

        try opt2.loadCheckpoint(testing.io, ckpt_path);
        try testing.expectEqual(opt1.step_count, opt2.step_count);
        try testing.expectApproxEqAbs(opt1.lr, opt2.lr, 1e-6);
        try testing.expect(opt2.velocities != null);
        try testing.expectApproxEqAbs(opt1.velocities.?[0][0], opt2.velocities.?[0][0], 1e-6);

        std.Io.Dir.cwd().deleteFile(testing.io, ckpt_path) catch {};
    }

    // 2. Test AdamWOptimizer checkpointing
    {
        var opt1 = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.002, .weight_decay = 0.05 });
        defer opt1.deinit();

        linear.weight.data[0] = 1.5;
        linear.weight.grad[0] = 0.3;
        opt1.step();
        opt1.step();

        const ckpt_path = "test_adamw_ckpt.bin";
        try opt1.saveCheckpoint(testing.io, ckpt_path);

        var opt2 = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.001 });
        defer opt2.deinit();

        try opt2.loadCheckpoint(testing.io, ckpt_path);
        try testing.expectEqual(opt1.step_count, opt2.step_count);
        try testing.expectApproxEqAbs(opt1.config.lr, opt2.config.lr, 1e-6);
        try testing.expectApproxEqAbs(opt1.m[0][0], opt2.m[0][0], 1e-6);
        try testing.expectApproxEqAbs(opt1.v[0][0], opt2.v[0][0], 1e-6);

        std.Io.Dir.cwd().deleteFile(testing.io, ckpt_path) catch {};
    }

    // 3. Test AdamOptimizer checkpointing
    {
        var opt1 = try AdamOptimizer.init(allocator, &linear, .{ .lr = 0.003 });
        defer opt1.deinit();

        linear.weight.data[0] = 2.5;
        linear.weight.grad[0] = 0.4;
        opt1.step();

        const ckpt_path = "test_adam_ckpt.bin";
        try opt1.saveCheckpoint(testing.io, ckpt_path);

        var opt2 = try AdamOptimizer.init(allocator, &linear, .{ .lr = 0.001 });
        defer opt2.deinit();

        try opt2.loadCheckpoint(testing.io, ckpt_path);
        try testing.expectEqual(opt1.step_count, opt2.step_count);
        try testing.expectApproxEqAbs(opt1.lr, opt2.lr, 1e-6);
        try testing.expectApproxEqAbs(opt1.m[0][0], opt2.m[0][0], 1e-6);
        try testing.expectApproxEqAbs(opt1.v[0][0], opt2.v[0][0], 1e-6);

        std.Io.Dir.cwd().deleteFile(testing.io, ckpt_path) catch {};
    }
}

test "Optimizer checkpoint error conditions and corruption resilience" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(99);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    // 1. Test corrupt magic header
    {
        const corrupt_file = "test_corrupt_magic.bin";
        var file = try std.Io.Dir.cwd().createFile(testing.io, corrupt_file, .{});
        defer file.close(testing.io);
        defer std.Io.Dir.cwd().deleteFile(testing.io, corrupt_file) catch {};

        var buf: [64]u8 = undefined;
        var file_writer = file.writer(testing.io, &buf);
        try file_writer.interface.writeAll("BADMAGICDATA12345678");
        try file_writer.interface.flush();

        var opt = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.01 });
        defer opt.deinit();

        try testing.expectError(error.InvalidCheckpointMagic, opt.loadCheckpoint(testing.io, corrupt_file));
    }

    // 2. Test optimizer type mismatch (saved as SGD, load with AdamW)
    {
        const sgd_file = "test_type_mismatch.bin";
        var opt_sgd = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.01 });
        defer opt_sgd.deinit();
        try opt_sgd.saveCheckpoint(testing.io, sgd_file);
        defer std.Io.Dir.cwd().deleteFile(testing.io, sgd_file) catch {};

        var opt_adamw = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.01 });
        defer opt_adamw.deinit();

        try testing.expectError(error.OptimizerTypeMismatch, opt_adamw.loadCheckpoint(testing.io, sgd_file));
    }

    // 3. Test parameter count mismatch
    {
        const param_file = "test_param_mismatch.bin";
        var opt_sgd = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.01 });
        defer opt_sgd.deinit();
        try opt_sgd.saveCheckpoint(testing.io, param_file);
        defer std.Io.Dir.cwd().deleteFile(testing.io, param_file) catch {};

        // Different model with more layers (4 parameters instead of 2)
        var seq = nn.sequential(.{
            try nn.Linear.init(allocator, 2, 2),
            try nn.Linear.init(allocator, 2, 2),
        });
        defer nn.deinitModel(&seq, allocator);
        {
            var init_graph = @import("autodiff.zig").Graph.initDefault(allocator);
            defer init_graph.deinit();
            _ = try seq.forward(&init_graph, try init_graph.ones(&.{ 1, 2 }, false));
            try nn.initModel(&seq, &init_graph, prng.random());
        }

        var opt_diff = try SGDOptimizer.init(allocator, &seq, .{ .lr = 0.01 });
        defer opt_diff.deinit();

        try testing.expectError(error.ParamCountMismatch, opt_diff.loadCheckpoint(testing.io, param_file));
    }
}

test "clipGradNorm and clipGradValue extreme and empty edge cases" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 1. Empty parameter slice
    var empty_params = [_]*Tensor{};
    const empty_norm = clipGradNorm(&empty_params, 1.0);
    try testing.expectEqual(@as(f32, 0.0), empty_norm);
    clipGradValue(&empty_params, 1.0);
    const empty_cfg_norm = clipGradients(&empty_params, .{ .norm = 1.0 });
    try testing.expectEqual(@as(f32, 0.0), empty_cfg_norm);
    const empty_cfg_none = clipGradients(&empty_params, .none);
    try testing.expect(empty_cfg_none == null);

    // 2. All zero gradients: norm = 0, ensure no NaN or division by zero
    var t_zero = try tensor.zeros(allocator, &.{ 2, 2 });
    defer tensor.free(allocator, t_zero);
    t_zero.requires_grad = true;
    t_zero.grad = try allocator.alloc(f32, 4);
    @memset(t_zero.grad, 0.0);

    var zero_params = [_]*Tensor{t_zero};
    const zero_norm = clipGradNorm(&zero_params, 1.0);
    try testing.expectEqual(@as(f32, 0.0), zero_norm);
    for (t_zero.grad) |g| {
        try testing.expectEqual(@as(f32, 0.0), g);
    }

    // 3. Non-clipping condition: actual norm <= max_norm
    @memcpy(t_zero.grad, &[_]f32{ 0.3, 0.4, 0.0, 0.0 }); // norm = 0.5 <= 1.0
    const unclipped_norm = clipGradNorm(&zero_params, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 0.5), unclipped_norm, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.3), t_zero.grad[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.4), t_zero.grad[1], 1e-5);

    // 4. max_norm <= 0.0 (scales all gradients to zero)
    const zero_scaled = clipGradNorm(&zero_params, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 0.5), zero_scaled, 1e-5);
    try testing.expectEqual(@as(f32, 0.0), t_zero.grad[0]);
    try testing.expectEqual(@as(f32, 0.0), t_zero.grad[1]);
}

test "Optimizer interface uniformity and scheduler stepping" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(77);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    // Test getLR / setLR / stepWithLR across all 3 optimizers
    var sgd = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1 });
    defer sgd.deinit();
    try testing.expectApproxEqAbs(@as(f32, 0.1), sgd.getLR(), 1e-6);
    sgd.setLR(0.05);
    try testing.expectApproxEqAbs(@as(f32, 0.05), sgd.getLR(), 1e-6);
    sgd.stepWithLR(0.02);
    try testing.expectEqual(@as(u64, 1), sgd.step_count);

    var adam = try AdamOptimizer.init(allocator, &linear, .{ .lr = 0.01 });
    defer adam.deinit();
    try testing.expectApproxEqAbs(@as(f32, 0.01), adam.getLR(), 1e-6);
    adam.setLR(0.005);
    try testing.expectApproxEqAbs(@as(f32, 0.005), adam.getLR(), 1e-6);
    adam.stepWithLR(0.002);
    try testing.expectEqual(@as(u64, 1), adam.step_count);

    var adamw = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.02 });
    defer adamw.deinit();
    try testing.expectApproxEqAbs(@as(f32, 0.02), adamw.getLR(), 1e-6);
    adamw.setLR(0.008);
    try testing.expectApproxEqAbs(@as(f32, 0.008), adamw.getLR(), 1e-6);
    adamw.stepWithLR(0.004);
    try testing.expectEqual(@as(u64, 1), adamw.step_count);

    // Test polymorphic LRScheduler.step()
    var sched = LRScheduler{ .step_lr = StepLRScheduler.init(0.1, 5, 0.5) };
    const lr0 = sched.step(&sgd, 0); // step 0 -> lr=0.1
    try testing.expectApproxEqAbs(@as(f32, 0.1), lr0, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), sgd.getLR(), 1e-6);
    const lr5 = sched.step(&sgd, 5); // step 5 -> lr=0.05
    try testing.expectApproxEqAbs(@as(f32, 0.05), lr5, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.05), sgd.getLR(), 1e-6);
}

test "Adam and AdamW vectorized updates match the scalar reference formulas" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 13 个权重元素：同时覆盖 SIMD 主循环之外的标量尾部
    var prng = std.Random.DefaultPrng.init(5);
    var linear = try nn.Linear.init(allocator, 13, 1);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    const n = linear.weight.data.len;
    const w0 = try allocator.dupe(f32, linear.weight.data);
    defer allocator.free(w0);
    for (linear.weight.grad, 0..) |*g, i| g.* = 0.1 * @as(f32, @floatFromInt(i)) - 0.6;

    const ref_w = try allocator.alloc(f32, n);
    defer allocator.free(ref_w);
    const ref_m = try allocator.alloc(f32, n);
    defer allocator.free(ref_m);
    const ref_v = try allocator.alloc(f32, n);
    defer allocator.free(ref_v);

    // 1. AdamW: w -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)
    {
        const cfg = AdamWConfig{ .lr = 0.01, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-8, .weight_decay = 0.1 };
        @memcpy(linear.weight.data, w0);
        var opt = try AdamWOptimizer.init(allocator, &linear, cfg);
        defer opt.deinit();
        @memcpy(ref_w, w0);
        @memset(ref_m, 0.0);
        @memset(ref_v, 0.0);
        for (1..4) |t| {
            opt.step();
            const tf: f32 = @floatFromInt(t);
            const bc1 = 1.0 - std.math.pow(f32, cfg.beta1, tf);
            const bc2 = 1.0 - std.math.pow(f32, cfg.beta2, tf);
            for (0..n) |i| {
                const g = linear.weight.grad[i];
                ref_m[i] = cfg.beta1 * ref_m[i] + (1.0 - cfg.beta1) * g;
                ref_v[i] = cfg.beta2 * ref_v[i] + (1.0 - cfg.beta2) * g * g;
                const step_val = (ref_m[i] / bc1) / (@sqrt(ref_v[i] / bc2) + cfg.eps);
                ref_w[i] -= cfg.lr * (step_val + cfg.weight_decay * ref_w[i]);
            }
        }
        for (0..n) |i| {
            try testing.expectApproxEqAbs(ref_w[i], linear.weight.data[i], 1e-5);
            try testing.expectApproxEqAbs(ref_m[i], opt.m[0][i], 1e-6);
            try testing.expectApproxEqAbs(ref_v[i], opt.v[0][i], 1e-6);
        }
    }

    // 2. Adam: w -= lr * sqrt(1 - beta2^t) / (1 - beta1^t) * m / (sqrt(v) + eps)
    {
        const cfg = AdamConfig{ .lr = 0.01, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8 };
        @memcpy(linear.weight.data, w0);
        var opt = try AdamOptimizer.init(allocator, &linear, cfg);
        defer opt.deinit();
        @memcpy(ref_w, w0);
        @memset(ref_m, 0.0);
        @memset(ref_v, 0.0);
        for (1..4) |t| {
            opt.step();
            const tf: f32 = @floatFromInt(t);
            const lr_t = cfg.lr * @sqrt(1.0 - std.math.pow(f32, cfg.beta2, tf)) / (1.0 - std.math.pow(f32, cfg.beta1, tf));
            for (0..n) |i| {
                const g = linear.weight.grad[i];
                ref_m[i] = cfg.beta1 * ref_m[i] + (1.0 - cfg.beta1) * g;
                ref_v[i] = cfg.beta2 * ref_v[i] + (1.0 - cfg.beta2) * g * g;
                ref_w[i] -= lr_t * ref_m[i] / (@sqrt(ref_v[i]) + cfg.eps);
            }
        }
        for (0..n) |i| {
            try testing.expectApproxEqAbs(ref_w[i], linear.weight.data[i], 1e-5);
        }
    }
}

test "AdamOptimizer step counter keeps increasing past 2^24 steps" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(8);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    var opt = try AdamOptimizer.init(allocator, &linear, .{});
    defer opt.deinit();
    // f32 步数在 2^24 处 t + 1 == t，偏差修正会停止推进；整数步数不受影响
    opt.step_count = 1 << 24;
    opt.step();
    opt.step();
    try testing.expectEqual(@as(u64, (1 << 24) + 2), opt.step_count);
}

test "Optimizer loadCheckpoint leaves state untouched on mismatch or truncation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const io = testing.io;

    var prng = std.Random.DefaultPrng.init(21);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    const path = "test_ckpt_untouched.bin";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    // 1. 参数个数不匹配：学习率与步数不应被检查点中的值覆盖
    {
        var src = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.5 });
        defer src.deinit();
        src.step();
        src.step();
        try src.saveCheckpoint(io, path);

        var seq = nn.sequential(.{
            try nn.Linear.init(allocator, 2, 2),
            try nn.Linear.init(allocator, 2, 2),
        });
        defer nn.deinitModel(&seq, allocator);
        for ([_]*Tensor{ seq.layers[0].weight, seq.layers[0].bias, seq.layers[1].weight, seq.layers[1].bias }) |t| {
            @memset(t.data, 0.1);
        }

        var dst = try SGDOptimizer.init(allocator, &seq, .{ .lr = 0.01 });
        defer dst.deinit();
        try testing.expectError(error.ParamCountMismatch, dst.loadCheckpoint(io, path));
        try testing.expectEqual(@as(f32, 0.01), dst.lr);
        try testing.expectEqual(@as(u64, 0), dst.step_count);
    }

    // 2. 截断的 AdamW 检查点：报告长度不匹配，且一阶/二阶矩与学习率保持原值
    {
        var src = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.002 });
        defer src.deinit();
        @memset(linear.weight.grad, 0.3);
        src.step();
        try src.saveCheckpoint(io, path);

        const full = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(full);
        {
            var file = try std.Io.Dir.cwd().createFile(io, path, .{});
            defer file.close(io);
            var buf: [256]u8 = undefined;
            var fw = file.writer(io, &buf);
            try fw.interface.writeAll(full[0 .. full.len - 4]);
            try fw.interface.flush();
        }

        var dst = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.001 });
        defer dst.deinit();
        try testing.expectError(error.CheckpointSizeMismatch, dst.loadCheckpoint(io, path));
        try testing.expectEqual(@as(f32, 0.001), dst.config.lr);
        try testing.expectEqual(@as(u64, 0), dst.step_count);
        try testing.expectEqual(@as(f32, 0.0), dst.m[0][0]);
    }

    // 3. 不含动量缓冲的 SGD 检查点：加载后已有动量缓冲被清零
    {
        var src = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1 });
        defer src.deinit();
        try src.saveCheckpoint(io, path);

        var dst = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1, .momentum = 0.9 });
        defer dst.deinit();
        @memset(linear.weight.grad, 1.0);
        dst.step();
        try testing.expect(dst.velocities.?[0][0] != 0.0);
        try dst.loadCheckpoint(io, path);
        for (dst.velocities.?) |v| {
            for (v) |x| try testing.expectEqual(@as(f32, 0.0), x);
        }
    }
}

/// 测试辅助：把字节内容整体写入文件
fn writeTestFile(io: std.Io, path: []const u8, bytes: []const u8) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [256]u8 = undefined;
    var fw = file.writer(io, &buf);
    try fw.interface.writeAll(bytes);
    try fw.interface.flush();
}

test "Optimizer checkpoint is fully little-endian and round-trips exactly" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const io = testing.io;

    var prng = std.Random.DefaultPrng.init(31);
    var linear = try nn.Linear.init(allocator, 3, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    const path = "test_ckpt_little_endian.bin";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var src = try AdamOptimizer.init(allocator, &linear, .{ .lr = 0.25 });
    defer src.deinit();
    for (src.m, src.v, 0..) |m_p, v_p, p| {
        for (m_p, v_p, 0..) |*m, *v, j| {
            m.* = @as(f32, @floatFromInt(p * 100 + j)) * 0.5 - 1.25;
            v.* = std.math.floatEps(f32) * @as(f32, @floatFromInt(j + 1));
        }
    }
    // 第一个参数的一阶矩首元素设为 1.5，其 IEEE 754 位模式为 0x3FC00000
    src.m[0][0] = 1.5;
    src.step_count = 0x0102030405060708;
    try src.saveCheckpoint(io, path);

    // 字节布局与主机字节序无关：版本、步数、学习率与状态缓冲都按小端序写出
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    try testing.expectEqualSlices(u8, &OPTIMIZER_MAGIC, bytes[0..4]);
    try testing.expectEqualSlices(u8, &.{ 2, 0, 0, 0 }, bytes[4..8]);
    try testing.expectEqual(OPTIMIZER_VERSION, std.mem.readInt(u32, bytes[4..8], .little));
    try testing.expectEqualSlices(u8, &.{ 8, 7, 6, 5, 4, 3, 2, 1 }, bytes[12..20]);
    try testing.expectEqualSlices(u8, &.{ 0x00, 0x00, 0x80, 0x3E }, bytes[20..24]); // 0.25
    const first_state = CHECKPOINT_HEADER_BYTES + 8;
    try testing.expectEqualSlices(u8, &.{ 0x00, 0x00, 0xC0, 0x3F }, bytes[first_state..][0..4]);

    // 往返恢复：所有状态按位一致
    var dst = try AdamOptimizer.init(allocator, &linear, .{});
    defer dst.deinit();
    try dst.loadCheckpoint(io, path);
    try testing.expectEqual(src.step_count, dst.step_count);
    try testing.expectEqual(src.lr, dst.lr);
    for (src.m, dst.m, src.v, dst.v) |sm, dm, sv, dv| {
        try testing.expectEqualSlices(f32, sm, dm);
        try testing.expectEqualSlices(f32, sv, dv);
    }
}

test "Optimizer loadCheckpoint is transactional on per-parameter length mismatch" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const io = testing.io;

    var prng = std.Random.DefaultPrng.init(37);
    var linear = try nn.Linear.init(allocator, 2, 2);
    linear.resetParameters(prng.random(), .{});
    defer nn.deinitModel(&linear, allocator);

    const path = "test_ckpt_transactional.bin";
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    // 1. AdamW：头部与文件总长度都正确，但第二个参数的长度字段被篡改。
    //    第一个参数的状态缓冲在文件中完好，也不得被写入优化器。
    {
        var src = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.002 });
        defer src.deinit();
        @memset(linear.weight.grad, 0.3);
        @memset(linear.bias.grad, -0.2);
        src.step();
        src.step();
        try src.saveCheckpoint(io, path);

        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(bytes);
        const second_len_offset = CHECKPOINT_HEADER_BYTES + 8 + 2 * src.params[0].data.len * @sizeOf(f32);
        const second_len = std.mem.readInt(u64, bytes[second_len_offset..][0..8], .little);
        try testing.expectEqual(@as(u64, src.params[1].data.len), second_len);
        std.mem.writeInt(u64, bytes[second_len_offset..][0..8], second_len + 1, .little);
        try writeTestFile(io, path, bytes);

        var dst = try AdamWOptimizer.init(allocator, &linear, .{ .lr = 0.001 });
        defer dst.deinit();
        @memset(linear.weight.grad, 1.0);
        @memset(linear.bias.grad, 1.0);
        dst.step();
        const m_before = try allocator.dupe(f32, dst.m[0]);
        defer allocator.free(m_before);
        const v_before = try allocator.dupe(f32, dst.v[0]);
        defer allocator.free(v_before);
        const m_ptr = dst.m.ptr;

        try testing.expectError(error.BufferSizeMismatch, dst.loadCheckpoint(io, path));
        try testing.expectEqual(@as(f32, 0.001), dst.config.lr);
        try testing.expectEqual(@as(u64, 1), dst.step_count);
        try testing.expectEqual(m_ptr, dst.m.ptr);
        try testing.expectEqualSlices(f32, m_before, dst.m[0]);
        try testing.expectEqualSlices(f32, v_before, dst.v[0]);
    }

    // 2. SGD：无动量缓冲的优化器加载被篡改的带动量检查点，失败后仍不分配动量缓冲
    {
        var src = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1, .momentum = 0.9 });
        defer src.deinit();
        src.step();
        try src.saveCheckpoint(io, path);

        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(bytes);
        const second_len_offset = CHECKPOINT_HEADER_BYTES + 1 + 8 + src.params[0].data.len * @sizeOf(f32);
        std.mem.writeInt(u64, bytes[second_len_offset..][0..8], 0, .little);
        try writeTestFile(io, path, bytes);

        var dst = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.01 });
        defer dst.deinit();
        try testing.expectError(error.BufferSizeMismatch, dst.loadCheckpoint(io, path));
        try testing.expect(dst.velocities == null);
        try testing.expectEqual(@as(f32, 0.01), dst.lr);
        try testing.expectEqual(@as(u64, 0), dst.step_count);
    }

    // 3. SGD：动量标志只能是 0 或 1
    {
        var src = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.1 });
        defer src.deinit();
        try src.saveCheckpoint(io, path);

        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(bytes);
        bytes[CHECKPOINT_HEADER_BYTES] = 2;
        try writeTestFile(io, path, bytes);

        var dst = try SGDOptimizer.init(allocator, &linear, .{ .lr = 0.01, .momentum = 0.9 });
        defer dst.deinit();
        @memset(linear.weight.grad, 1.0);
        dst.step();
        const vel_before = dst.velocities.?[0][0];
        try testing.expectError(error.InvalidCheckpointFlag, dst.loadCheckpoint(io, path));
        try testing.expectEqual(vel_before, dst.velocities.?[0][0]);
        try testing.expectEqual(@as(u64, 1), dst.step_count);
    }
}

test "clipGradNorm accumulates in higher precision and StepLR tolerates step_size 0" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 一个大分量 (平方 1e8) 加 1e6 个单位分量：单个 f32 累加器会把每个 +1 舍入掉 (1e8 处的单位最小精度为 8)
    const n: usize = 1_000_000 + 1;
    var t = try tensor.zeros(allocator, &.{n});
    defer tensor.free(allocator, t);
    t.requires_grad = true;
    t.grad = try allocator.alloc(f32, n);
    @memset(t.grad, 1.0);
    t.grad[0] = 1e4;

    var params = [_]*Tensor{t};
    const norm = clipGradNorm(&params, 1e9);
    try testing.expectApproxEqRel(@as(f32, @floatCast(@sqrt(1.01e8))), norm, 1e-5);

    const sched = StepLRScheduler{ .base_lr = 1.0, .step_size = 0, .gamma = 0.5 };
    try testing.expectApproxEqAbs(@as(f32, 0.25), sched.getLR(2), 1e-6);
}
