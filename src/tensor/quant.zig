//! 4-bit 量化 (Q4) 块格式、量化/反量化与矩阵乘法算子 (Quantized Matrix Multiplication)
//!
//! 采用 split-half 块内排布（低半字节存 0..15，高半字节存 16..31，与 ggml Q4_0 对齐），
//! 使得 CPU 单指令多数据流 (Single Instruction, Multiple Data, SIMD) 向量化与 GPU 能够连续加载激活值。

const std = @import("std");
const backend = @import("../backend.zig");
const Executor = backend.Executor;
const core = @import("core.zig");
const Tensor = core.Tensor;
const Shape = core.Shape;
const ops = @import("ops.zig");

/// 4-bit 分块量化结构体 (Q4Block)
/// 32 个浮点数缩减为 1 个 16 位浮点缩放比例 (scale) 与 16 字节权重半字节 (qs)，共 18 字节 (每权重 4.5 bit)
pub const Q4Block = extern struct {
    scale: f16, // 16 位浮点数缩放比例因子
    qs: [16]u8, // 16 字节：低 4 位存权重 j (0..15)，高 4 位存权重 j+16 (16..31)

    pub const QK: usize = 32; // 每块量化权重数量

    /// 将 32 个 f32 浮点权重原地量化并打包为 split-half Q4Block
    pub fn quantize(src: *const [32]f32) Q4Block {
        var amax: f32 = 0.0;
        for (src) |v| {
            const av = @abs(v);
            if (av > amax) amax = av;
        }
        const scale_f32 = amax / 7.0; // 4-bit 有符号映射至 [-7, +7]
        const inv_scale = if (scale_f32 > 0.0) 1.0 / scale_f32 else 0.0;
        var block: Q4Block = .{
            .scale = @floatCast(scale_f32),
            .qs = undefined,
        };

        for (0..16) |j| {
            const v0 = src[j];
            const v1 = src[j + 16];

            const q0_f = std.math.clamp(std.math.round(v0 * inv_scale) + 8.0, 0.0, 15.0);
            const q1_f = std.math.clamp(std.math.round(v1 * inv_scale) + 8.0, 0.0, 15.0);

            const q0: u8 = @intFromFloat(q0_f);
            const q1: u8 = @intFromFloat(q1_f);

            block.qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
        return block;
    }

    /// 反量化解包为 32 个 f32 浮点权重
    pub fn dequantize(self: Q4Block, dest: *[32]f32) void {
        const d: f32 = @floatCast(self.scale);
        for (0..16) |j| {
            const byte = self.qs[j];
            const q0: f32 = @as(f32, @floatFromInt(byte & 0x0F)) - 8.0;
            const q1: f32 = @as(f32, @floatFromInt(byte >> 4)) - 8.0;
            dest[j] = q0 * d;
            dest[j + 16] = q1 * d;
        }
    }

    /// 连续 32 维向量的高效向量化点积计算
    pub fn dot(self: Q4Block, x_chunk: *const [32]f32) f32 {
        const V8 = @Vector(8, f32);
        const eight: V8 = @splat(8.0);
        const d: f32 = @floatCast(self.scale);

        const x_lo0: V8 = @as(*const [8]f32, @ptrCast(x_chunk[0..8].ptr)).*;
        const x_lo1: V8 = @as(*const [8]f32, @ptrCast(x_chunk[8..16].ptr)).*;
        const x_hi0: V8 = @as(*const [8]f32, @ptrCast(x_chunk[16..24].ptr)).*;
        const x_hi1: V8 = @as(*const [8]f32, @ptrCast(x_chunk[24..32].ptr)).*;

        const q_lo0: V8 = .{
            @floatFromInt(self.qs[0] & 0x0F),
            @floatFromInt(self.qs[1] & 0x0F),
            @floatFromInt(self.qs[2] & 0x0F),
            @floatFromInt(self.qs[3] & 0x0F),
            @floatFromInt(self.qs[4] & 0x0F),
            @floatFromInt(self.qs[5] & 0x0F),
            @floatFromInt(self.qs[6] & 0x0F),
            @floatFromInt(self.qs[7] & 0x0F),
        };
        const q_hi0: V8 = .{
            @floatFromInt(self.qs[0] >> 4),
            @floatFromInt(self.qs[1] >> 4),
            @floatFromInt(self.qs[2] >> 4),
            @floatFromInt(self.qs[3] >> 4),
            @floatFromInt(self.qs[4] >> 4),
            @floatFromInt(self.qs[5] >> 4),
            @floatFromInt(self.qs[6] >> 4),
            @floatFromInt(self.qs[7] >> 4),
        };
        const q_lo1: V8 = .{
            @floatFromInt(self.qs[8] & 0x0F),
            @floatFromInt(self.qs[9] & 0x0F),
            @floatFromInt(self.qs[10] & 0x0F),
            @floatFromInt(self.qs[11] & 0x0F),
            @floatFromInt(self.qs[12] & 0x0F),
            @floatFromInt(self.qs[13] & 0x0F),
            @floatFromInt(self.qs[14] & 0x0F),
            @floatFromInt(self.qs[15] & 0x0F),
        };
        const q_hi1: V8 = .{
            @floatFromInt(self.qs[8] >> 4),
            @floatFromInt(self.qs[9] >> 4),
            @floatFromInt(self.qs[10] >> 4),
            @floatFromInt(self.qs[11] >> 4),
            @floatFromInt(self.qs[12] >> 4),
            @floatFromInt(self.qs[13] >> 4),
            @floatFromInt(self.qs[14] >> 4),
            @floatFromInt(self.qs[15] >> 4),
        };

        const w_lo0 = q_lo0 - eight;
        const w_lo1 = q_lo1 - eight;
        const w_hi0 = q_hi0 - eight;
        const w_hi1 = q_hi1 - eight;

        const sum_vec = (x_lo0 * w_lo0 + x_lo1 * w_lo1) + (x_hi0 * w_hi0 + x_hi1 * w_hi1);
        return @reduce(.Add, sum_vec) * d;
    }
};

/// 批量量化单精度浮点切片为 Q4 块
pub fn quantizeSlice(dest: []Q4Block, src: []const f32) void {
    const num_blocks = src.len / Q4Block.QK;
    std.debug.assert(dest.len >= num_blocks);
    for (0..num_blocks) |b| {
        const chunk: *const [32]f32 = @ptrCast(src[b * 32 .. (b + 1) * 32].ptr);
        dest[b] = Q4Block.quantize(chunk);
    }
}

/// 批量反量化 Q4 块为单精度浮点切片
pub fn dequantizeSlice(dest: []f32, src: []const Q4Block) void {
    std.debug.assert(dest.len >= src.len * Q4Block.QK);
    for (src, 0..) |blk, b| {
        const chunk: *[32]f32 = @ptrCast(dest[b * 32 .. (b + 1) * 32].ptr);
        blk.dequantize(chunk);
    }
}

/// 计算目标传入式 (Destination-Passing Style, DPS) Q4 矩阵乘法：
/// out = x * W^T，其中 W 存储为 Q4Block 切片 (out_features x in_features)
pub fn q4MatMulInto(
    exec: Executor,
    out: *Tensor,
    x: *Tensor,
    blocks: []const Q4Block,
    in_features: usize,
    out_features: usize,
) !void {
    if (in_features % Q4Block.QK != 0) return error.IncompatibleDimensions;
    const blocks_per_row = in_features / Q4Block.QK;
    if (blocks.len < out_features * blocks_per_row) return error.DimensionOutOfBounds;

    const num_rows = x.data.len / in_features;
    if (out.data.len < num_rows * out_features) return error.DimensionOutOfBounds;

    const Context = struct {
        x_data: []const f32,
        out_data: []f32,
        blocks: []const Q4Block,
        blocks_per_row: usize,
        in_features: usize,
        out_features: usize,

        fn runRowCols(ctx: @This(), begin_col: usize, end_col: usize) void {
            // 单行密集 GEMV (如自回归解码单个 Token)
            const x_row = ctx.x_data[0..ctx.in_features];
            for (begin_col..end_col) |out_j| {
                const base_b = out_j * ctx.blocks_per_row;
                var total_sum: f32 = 0.0;
                for (0..ctx.blocks_per_row) |b| {
                    const blk = ctx.blocks[base_b + b];
                    const x_chunk: *const [32]f32 = @ptrCast(x_row[b * 32 .. (b + 1) * 32].ptr);
                    total_sum += blk.dot(x_chunk);
                }
                ctx.out_data[out_j] = total_sum;
            }
        }

        fn runMultiRows(ctx: @This(), begin_row: usize, end_row: usize) void {
            // 多行批处理 GEMM
            for (begin_row..end_row) |r| {
                const x_row = ctx.x_data[r * ctx.in_features .. (r + 1) * ctx.in_features];
                const out_row = ctx.out_data[r * ctx.out_features .. (r + 1) * ctx.out_features];
                for (0..ctx.out_features) |out_j| {
                    const base_b = out_j * ctx.blocks_per_row;
                    var total_sum: f32 = 0.0;
                    for (0..ctx.blocks_per_row) |b| {
                        const blk = ctx.blocks[base_b + b];
                        const x_chunk: *const [32]f32 = @ptrCast(x_row[b * 32 .. (b + 1) * 32].ptr);
                        total_sum += blk.dot(x_chunk);
                    }
                    out_row[out_j] = total_sum;
                }
            }
        }
    };

    const ctx = Context{
        .x_data = x.data,
        .out_data = out.data,
        .blocks = blocks,
        .blocks_per_row = blocks_per_row,
        .in_features = in_features,
        .out_features = out_features,
    };

    if (num_rows == 1) {
        // 单行输入：按输出维度切块并行
        const grain: usize = if (out_features >= 2048) 256 else out_features;
        try exec.parallelFor(out_features, grain, ctx, Context.runRowCols);
    } else {
        // 多行输入：按行切块并行
        const grain: usize = 1;
        try exec.parallelFor(num_rows, grain, ctx, Context.runMultiRows);
    }
}

/// 分配式 Q4 矩阵乘法
pub fn q4MatMul(
    x: *Tensor,
    blocks: []const Q4Block,
    in_features: usize,
    out_features: usize,
    allocator: std.mem.Allocator,
) !*Tensor {
    if (x.shape.len == 0) return error.EmptyInputs;
    const last_dim = x.shape.dims[x.shape.len - 1];
    if (last_dim != in_features) return error.IncompatibleDimensions;

    var out_dims: [8]usize = undefined;
    for (0..x.shape.len - 1) |i| {
        out_dims[i] = x.shape.dims[i];
    }
    out_dims[x.shape.len - 1] = out_features;

    const out = try ops.zeros(allocator, out_dims[0..x.shape.len]);
    errdefer out.deinit(allocator);

    try q4MatMulInto(.serial, out, x, blocks, in_features, out_features);
    return out;
}

test "Q4Block split-half quantize and dequantize" {
    var raw: [32]f32 = undefined;
    for (0..32) |i| {
        raw[i] = (@as(f32, @floatFromInt(i)) - 16.0) * 0.25;
    }

    const blk = Q4Block.quantize(&raw);
    var restored: [32]f32 = undefined;
    blk.dequantize(&restored);

    for (0..32) |i| {
        // 4-bit 量化引入的最大量化阶误差不超过 step / 2
        try std.testing.expectApproxEqAbs(raw[i], restored[i], 0.35);
    }
}

test "Q4Block dot product matches dequantized dot" {
    var raw: [32]f32 = undefined;
    var x: [32]f32 = undefined;
    for (0..32) |i| {
        raw[i] = (@as(f32, @floatFromInt(i)) - 15.0) * 0.3;
        x[i] = (@as(f32, @floatFromInt(i)) * 0.1) - 1.0;
    }

    const blk = Q4Block.quantize(&raw);
    var deq: [32]f32 = undefined;
    blk.dequantize(&deq);

    var expected_dot: f32 = 0.0;
    for (0..32) |i| {
        expected_dot += deq[i] * x[i];
    }

    const actual_dot = blk.dot(&x);
    try std.testing.expectApproxEqAbs(expected_dot, actual_dot, 1e-4);
}

test "q4MatMulInto serial vs threaded bitwise equivalence" {
    const allocator = std.testing.allocator;
    const in_f = 64;
    const out_f = 128;
    const num_blocks = (in_f * out_f) / Q4Block.QK;

    const raw_w = try allocator.alloc(f32, in_f * out_f);
    defer allocator.free(raw_w);
    for (raw_w, 0..) |*w, i| {
        w.* = @sin(@as(f32, @floatFromInt(i))) * 2.0;
    }

    const blocks = try allocator.alloc(Q4Block, num_blocks);
    defer allocator.free(blocks);
    quantizeSlice(blocks, raw_w);

    const x_t = try ops.zeros(allocator, &.{ 1, in_f });
    defer x_t.deinit(allocator);
    for (x_t.data, 0..) |*v, i| {
        v.* = @cos(@as(f32, @floatFromInt(i))) * 1.5;
    }

    const out_serial = try ops.zeros(allocator, &.{ 1, out_f });
    defer out_serial.deinit(allocator);
    try q4MatMulInto(.serial, out_serial, x_t, blocks, in_f, out_f);

    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const threaded_exec = Executor{ .io = threaded.io(), .max_tasks = 4 };

    const out_threaded = try ops.zeros(allocator, &.{ 1, out_f });
    defer out_threaded.deinit(allocator);
    try q4MatMulInto(threaded_exec, out_threaded, x_t, blocks, in_f, out_f);

    // 逐位一致断言！
    for (out_serial.data, out_threaded.data) |s, t| {
        try std.testing.expectEqual(s, t);
    }
}
