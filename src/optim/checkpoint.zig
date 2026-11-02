const std = @import("std");
const tensor = @import("../tensor.zig");
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

/// 公共头部字节数：magic + version + type_tag + step + lr + param_count
pub const CHECKPOINT_HEADER_BYTES: u64 = 4 + 4 + 4 + 8 + 4 + 8;

pub const CheckpointHeader = struct {
    step: u64,
    lr: f32,
};

pub fn writeU32(writer: *std.Io.Writer, val: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, val, .little);
    try writer.writeAll(&b);
}

pub fn writeU64(writer: *std.Io.Writer, val: u64) !void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, val, .little);
    try writer.writeAll(&b);
}

pub fn writeF32(writer: *std.Io.Writer, val: f32) !void {
    try writeU32(writer, @bitCast(val));
}

pub fn readU32(reader: *std.Io.Reader) !u32 {
    var b: [4]u8 = undefined;
    try reader.readSliceAll(&b);
    return std.mem.readInt(u32, &b, .little);
}

pub fn readU64(reader: *std.Io.Reader) !u64 {
    var b: [8]u8 = undefined;
    try reader.readSliceAll(&b);
    return std.mem.readInt(u64, &b, .little);
}

pub fn readF32(reader: *std.Io.Reader) !f32 {
    return @bitCast(try readU32(reader));
}

pub fn writeCheckpointHeader(writer: *std.Io.Writer, tag: OptimizerTypeTag, header: CheckpointHeader, param_count: usize) !void {
    try writer.writeAll(&OPTIMIZER_MAGIC);
    try writeU32(writer, OPTIMIZER_VERSION);
    try writeU32(writer, @backingInt(tag));
    try writeU64(writer, header.step);
    try writeF32(writer, header.lr);
    try writeU64(writer, @as(u64, param_count));
}

/// 读取并校验公共头部，只返回解析结果，不修改任何优化器状态
pub fn readCheckpointHeader(reader: *std.Io.Reader, tag: OptimizerTypeTag, param_count: usize) !CheckpointHeader {
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
pub fn checkpointStateBytes(params: []const *Tensor, buffers_per_param: usize) u64 {
    var total: u64 = 0;
    for (params) |param| {
        total += 8 + @as(u64, buffers_per_param) * @as(u64, param.data.len) * @sizeOf(f32);
    }
    return total;
}

/// 逐参数写入长度字段与各状态缓冲 (`states[k][i]` 为第 i 个参数的第 k 个状态缓冲)，f32 一律按小端序写出
pub fn writeStateBuffers(writer: *std.Io.Writer, params: []const *Tensor, states: []const []const []f32) !void {
    for (params, 0..) |param, i| {
        try writeU64(writer, @as(u64, param.data.len));
        for (states) |state| {
            const bits: []const u32 = @ptrCast(state[i]);
            try writer.writeSliceEndian(u32, bits, .little);
        }
    }
}

/// 写出完整检查点文件；`states` 的个数必须与 `readCheckpoint` 对该类型推断出的缓冲个数一致
pub fn writeCheckpoint(
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
pub const MAX_STATE_BUFFERS = 2;

/// 已完整读取并校验的检查点：状态缓冲存放在新分配的内存中，由调用方在提交阶段接管
pub const LoadedCheckpoint = struct {
    header: CheckpointHeader,
    /// 有效状态缓冲组数：SGD 由动量标志决定 (0 或 1)，Adam/AdamW 固定为 2
    buffer_count: usize,
    states: [MAX_STATE_BUFFERS][][]f32,

    pub fn deinit(self: *LoadedCheckpoint, allocator: std.mem.Allocator) void {
        for (self.states[0..self.buffer_count]) |s| freeStates(allocator, s);
        self.buffer_count = 0;
    }
};

/// 读取整个检查点文件并完成全部校验，不修改任何优化器状态。
/// 返回的状态缓冲为新分配的内存 (加载期间峰值内存约为两份优化器状态)，所有权交给调用方；
/// 任何错误都会释放已分配部分。
pub fn readCheckpoint(
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
pub fn replaceStates(allocator: std.mem.Allocator, target: *[][]f32, new_states: [][]f32) void {
    freeStates(allocator, target.*);
    target.* = new_states;
}

/// 为每个参数分配一块与参数等长、清零的状态缓冲；失败时释放已分配部分
pub fn allocZeroedStates(allocator: std.mem.Allocator, params: []const *Tensor) ![][]f32 {
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

pub fn freeStates(allocator: std.mem.Allocator, states: []const []f32) void {
    for (states) |s| allocator.free(s);
    allocator.free(states);
}
