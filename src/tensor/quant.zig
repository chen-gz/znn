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

const V8 = @Vector(8, f32);

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

    /// 解包后的 Q4 块：32 个去偏移权重 (q - 8) 按 4 组 8 路向量排列，缩放比例单独保存
    pub const Unpacked = struct {
        lo0: V8, // 权重 0..7
        lo1: V8, // 权重 8..15
        hi0: V8, // 权重 16..23
        hi1: V8, // 权重 24..31
        scale: f32,
    };

    /// 以 16 路向量运算解包全部半字节；多行输入时同一权重块只解包一次并复用于各行
    pub fn unpack(self: Q4Block) Unpacked {
        const q: @Vector(16, u8) = self.qs;
        const low_mask: @Vector(16, u8) = @splat(0x0F);
        const high_shift: @Vector(16, u3) = @splat(4);
        const offset: @Vector(16, f32) = @splat(8.0);
        const lo_f: @Vector(16, f32) = @floatFromInt(q & low_mask);
        const hi_f: @Vector(16, f32) = @floatFromInt(q >> high_shift);
        const lo: [16]f32 = lo_f - offset;
        const hi: [16]f32 = hi_f - offset;
        return .{
            .lo0 = lo[0..8].*,
            .lo1 = lo[8..16].*,
            .hi0 = hi[0..8].*,
            .hi1 = hi[8..16].*,
            .scale = @floatCast(self.scale),
        };
    }

    /// 解包后的权重与连续 32 维激活的点积；与 `dot` 逐位一致
    pub fn dotUnpacked(w: Unpacked, x_chunk: *const [32]f32) f32 {
        const x_lo0: V8 = x_chunk[0..8].*;
        const x_lo1: V8 = x_chunk[8..16].*;
        const x_hi0: V8 = x_chunk[16..24].*;
        const x_hi1: V8 = x_chunk[24..32].*;
        const sum_vec = (x_lo0 * w.lo0 + x_lo1 * w.lo1) + (x_hi0 * w.hi0 + x_hi1 * w.hi1);
        return @reduce(.Add, sum_vec) * w.scale;
    }

    /// 连续 32 维向量的向量化点积计算
    pub fn dot(self: Q4Block, x_chunk: *const [32]f32) f32 {
        return dotUnpacked(self.unpack(), x_chunk);
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

/// 批量量化 BF16 位模式切片为 Q4 块
pub fn quantizeSliceBf16(dest: []Q4Block, src: []const u16) void {
    const num_blocks = src.len / Q4Block.QK;
    std.debug.assert(dest.len >= num_blocks);
    var f32_chunk: [32]f32 = undefined;
    for (0..num_blocks) |b| {
        backend.vec_math.widenBf16(&f32_chunk, src[b * 32 .. (b + 1) * 32], 1.0);
        dest[b] = Q4Block.quantize(&f32_chunk);
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

/// 多行输入时一个分块覆盖的行数：分块内同一权重块只解包一次并复用于这些行
const row_tile: usize = 8;
/// 一个分块覆盖的输出列数
const col_tile: usize = 128;

/// 计算目标传入式 (Destination-Passing Style, DPS) Q4 矩阵乘法：
/// out = x * W^T，其中 W 存储为 Q4Block 切片 (out_features x in_features)。
///
/// 输出按 (行块 x 列块) 二维分块并行；每个输出元素只由一个分块按固定的块序累加，
/// 因此结果与线程数、分块方式无关，多行输入中的每一行也与单行输入的结果逐位一致。
/// `x` 与 `out` 必须是连续张量（允许连续的切片视图），非连续输入请使用分配式的 `q4MatMul`。
pub fn q4MatMulInto(
    exec: Executor,
    out: *Tensor,
    x: *Tensor,
    blocks: []const Q4Block,
    in_features: usize,
    out_features: usize,
) !void {
    if (in_features == 0 or in_features % Q4Block.QK != 0) return error.IncompatibleDimensions;
    if (x.shape.len == 0 or x.shape.dims[x.shape.len - 1] != in_features) return error.IncompatibleDimensions;
    if (!x.isContiguous() or !out.isContiguous()) return error.NonContiguousTensor;
    const blocks_per_row = in_features / Q4Block.QK;
    if (blocks.len < out_features * blocks_per_row) return error.DimensionOutOfBounds;

    const x_numel = x.shape.numel();
    const num_rows = x_numel / in_features;
    if (out.shape.numel() != num_rows * out_features) return error.ShapeMismatch;
    if (num_rows == 0 or out_features == 0) return;

    const col_tiles = (out_features + col_tile - 1) / col_tile;
    const row_tiles = (num_rows + row_tile - 1) / row_tile;

    const Context = struct {
        x_data: []const f32,
        out_data: []f32,
        blocks: []const Q4Block,
        blocks_per_row: usize,
        in_features: usize,
        out_features: usize,
        num_rows: usize,
        col_tiles: usize,

        fn runTiles(ctx: @This(), begin_tile: usize, end_tile: usize) void {
            for (begin_tile..end_tile) |tile| {
                const r0 = (tile / ctx.col_tiles) * row_tile;
                const r1 = @min(r0 + row_tile, ctx.num_rows);
                const c0 = (tile % ctx.col_tiles) * col_tile;
                const c1 = @min(c0 + col_tile, ctx.out_features);
                for (c0..c1) |j| {
                    var sums: [row_tile]f32 = @splat(0.0);
                    const w_row = ctx.blocks[j * ctx.blocks_per_row ..][0..ctx.blocks_per_row];
                    for (w_row, 0..) |blk, b| {
                        const w = blk.unpack();
                        for (r0..r1, 0..) |r, i| {
                            const x_chunk = ctx.x_data[r * ctx.in_features + b * Q4Block.QK ..][0..Q4Block.QK];
                            sums[i] += Q4Block.dotUnpacked(w, x_chunk);
                        }
                    }
                    for (r0..r1, 0..) |r, i| {
                        ctx.out_data[r * ctx.out_features + j] = sums[i];
                    }
                }
            }
        }
    };

    const ctx = Context{
        .x_data = x.data[0..x_numel],
        .out_data = out.data[0 .. num_rows * out_features],
        .blocks = blocks,
        .blocks_per_row = blocks_per_row,
        .in_features = in_features,
        .out_features = out_features,
        .num_rows = num_rows,
        .col_tiles = col_tiles,
    };
    try exec.parallelFor(row_tiles * col_tiles, 1, ctx, Context.runTiles);
}

/// 分配式 Q4 矩阵乘法：输出形状为 `x` 的形状把最后一维换成 `out_features`；
/// 非连续输入先物化为连续副本，计算经 `exec` 并行
pub fn q4MatMul(
    exec: Executor,
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

    const x_contig = if (x.isContiguous()) x else try x.contiguous(allocator);
    defer if (x_contig != x) x_contig.deinit(allocator);

    const out = try ops.zeros(allocator, out_dims[0..x.shape.len]);
    errdefer out.deinit(allocator);

    try q4MatMulInto(exec, out, x_contig, blocks, in_features, out_features);
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

/// 测试辅助：用解量化权重逐元素计算 x * W^T 的参考结果
fn referenceMatMul(allocator: std.mem.Allocator, x: []const f32, blocks: []const Q4Block, rows: usize, in_f: usize, out_f: usize) ![]f32 {
    const w = try allocator.alloc(f32, in_f * out_f);
    defer allocator.free(w);
    dequantizeSlice(w, blocks);
    const ref = try allocator.alloc(f32, rows * out_f);
    for (0..rows) |r| {
        for (0..out_f) |j| {
            var acc: f64 = 0.0;
            for (0..in_f) |k| acc += @as(f64, x[r * in_f + k]) * @as(f64, w[j * in_f + k]);
            ref[r * out_f + j] = @floatCast(acc);
        }
    }
    return ref;
}

test "q4MatMulInto multi-row tiles: serial, threaded and per-row results are bitwise identical" {
    const allocator = std.testing.allocator;
    const in_f = 96;
    const out_f = 300; // 非 col_tile 整数倍，覆盖边缘列块
    const rows = 11; // 非 row_tile 整数倍，覆盖边缘行块

    const raw_w = try allocator.alloc(f32, in_f * out_f);
    defer allocator.free(raw_w);
    for (raw_w, 0..) |*w, i| w.* = @sin(@as(f32, @floatFromInt(i)) * 0.37) * 1.3;
    const blocks = try allocator.alloc(Q4Block, (in_f * out_f) / Q4Block.QK);
    defer allocator.free(blocks);
    quantizeSlice(blocks, raw_w);

    const x = try ops.zeros(allocator, &.{ rows, in_f });
    defer x.deinit(allocator);
    for (x.data, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.11) * 0.9;

    const out_serial = try ops.zeros(allocator, &.{ rows, out_f });
    defer out_serial.deinit(allocator);
    try q4MatMulInto(.serial, out_serial, x, blocks, in_f, out_f);

    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const out_threaded = try ops.zeros(allocator, &.{ rows, out_f });
    defer out_threaded.deinit(allocator);
    try q4MatMulInto(.{ .io = threaded.io(), .max_tasks = 4 }, out_threaded, x, blocks, in_f, out_f);
    try std.testing.expectEqualSlices(f32, out_serial.data, out_threaded.data);

    // 每一行单独作为单行输入计算，必须与多行分块结果逐位一致
    const x_row = try ops.zeros(allocator, &.{ 1, in_f });
    defer x_row.deinit(allocator);
    const out_row = try ops.zeros(allocator, &.{ 1, out_f });
    defer out_row.deinit(allocator);
    for (0..rows) |r| {
        @memcpy(x_row.data, x.data[r * in_f .. (r + 1) * in_f]);
        try q4MatMulInto(.serial, out_row, x_row, blocks, in_f, out_f);
        try std.testing.expectEqualSlices(f32, out_serial.data[r * out_f .. (r + 1) * out_f], out_row.data);
    }

    // 与解量化权重的参考实现数值一致
    const ref = try referenceMatMul(allocator, x.data, blocks, rows, in_f, out_f);
    defer allocator.free(ref);
    for (ref, out_serial.data) |e, a| {
        try std.testing.expectApproxEqAbs(e, a, 1e-3 * @max(1.0, @abs(e)));
    }
}

test "q4MatMul handles slice views and non-contiguous inputs; q4MatMulInto rejects non-contiguous inputs" {
    const allocator = std.testing.allocator;
    const in_f = 64;
    const out_f = 40;

    const raw_w = try allocator.alloc(f32, in_f * out_f);
    defer allocator.free(raw_w);
    for (raw_w, 0..) |*w, i| w.* = @sin(@as(f32, @floatFromInt(i)) * 0.21);
    const blocks = try allocator.alloc(Q4Block, (in_f * out_f) / Q4Block.QK);
    defer allocator.free(blocks);
    quantizeSlice(blocks, raw_w);

    const base = try ops.zeros(allocator, &.{ 4, in_f });
    defer base.deinit(allocator);
    for (base.data, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.07);

    // 1. 连续切片视图 (第 1..3 行)：data.len 大于逻辑元素数，必须按 numel 计算行数
    const view = try base.slice(&.{.{ .start = 1, .end = 3 }}, allocator);
    defer view.deinit(allocator);
    const view_copy = try view.contiguous(allocator);
    defer view_copy.deinit(allocator);
    const y_view = try q4MatMul(.serial, view, blocks, in_f, out_f, allocator);
    defer y_view.deinit(allocator);
    const y_copy = try q4MatMul(.serial, view_copy, blocks, in_f, out_f, allocator);
    defer y_copy.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), y_view.shape.dims[0]);
    try std.testing.expectEqualSlices(f32, y_copy.data, y_view.data);

    // 2. 非连续视图 (转置)：分配式接口物化后计算，DPS 接口直接拒绝
    const cols = try ops.zeros(allocator, &.{ in_f, 2 });
    defer cols.deinit(allocator);
    for (cols.data, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.13);
    const x_t = try cols.transposeView(0, 1, allocator);
    defer x_t.deinit(allocator);
    try std.testing.expect(!x_t.isContiguous());
    const x_t_copy = try x_t.contiguous(allocator);
    defer x_t_copy.deinit(allocator);
    const y_t = try q4MatMul(.serial, x_t, blocks, in_f, out_f, allocator);
    defer y_t.deinit(allocator);
    const y_t_copy = try q4MatMul(.serial, x_t_copy, blocks, in_f, out_f, allocator);
    defer y_t_copy.deinit(allocator);
    try std.testing.expectEqualSlices(f32, y_t_copy.data, y_t.data);
    try std.testing.expectError(error.NonContiguousTensor, q4MatMulInto(.serial, y_t, x_t, blocks, in_f, out_f));
}
