const std = @import("std");
const testing = std.testing;
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const zeros = tensor.zeros;
const free = tensor.free;

fn atCoord(t: *const Tensor, coord: []const usize) f32 {
    var idx: usize = 0;
    for (coord, 0..) |c_, d| idx += c_ * t.strides.dims[d];
    return t.data[idx];
}

fn fillDeterministic(t: *Tensor, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    for (t.data) |*v| {
        v.* = (rand.float(f32) - 0.5) * 2.0;
    }
}

/// 测试辅助：朴素二维卷积参考实现（一维卷积视为 H = KH = 1）
fn referenceConv2d(x: *const Tensor, w: *const Tensor, bias: ?*const Tensor, stride: usize, padding: usize, out: *const Tensor) !void {
    const N = x.shape.dims[0];
    const C_in = x.shape.dims[1];
    const H = x.shape.dims[2];
    const W = x.shape.dims[3];
    const C_out = w.shape.dims[0];
    const KH = w.shape.dims[2];
    const KW = w.shape.dims[3];
    const H_out = out.shape.dims[2];
    const W_out = out.shape.dims[3];
    for (0..N) |n| for (0..C_out) |co| for (0..H_out) |oh| for (0..W_out) |ow| {
        var acc: f32 = if (bias) |b| b.data[co] else 0.0;
        for (0..C_in) |ci| for (0..KH) |kh| for (0..KW) |kw| {
            const ih = @as(isize, @intCast(oh * stride + kh)) - @as(isize, @intCast(padding));
            const iw = @as(isize, @intCast(ow * stride + kw)) - @as(isize, @intCast(padding));
            if (ih < 0 or iw < 0 or ih >= H or iw >= W) continue;
            acc += atCoord(x, &.{ n, ci, @intCast(ih), @intCast(iw) }) * atCoord(w, &.{ co, ci, kh, kw });
        };
        try testing.expectApproxEqAbs(acc, out.data[((n * C_out + co) * H_out + oh) * W_out + ow], 1e-4);
    };
}

/// 测试辅助：朴素二维转置卷积参考实现（散射累加）
fn referenceConvTranspose2d(allocator: std.mem.Allocator, x: *const Tensor, w: *const Tensor, bias: ?*const Tensor, stride: usize, padding: usize, out: *const Tensor) !void {
    const N = x.shape.dims[0];
    const C_in = x.shape.dims[1];
    const H = x.shape.dims[2];
    const W = x.shape.dims[3];
    const C_out = w.shape.dims[1];
    const KH = w.shape.dims[2];
    const KW = w.shape.dims[3];
    const H_out = out.shape.dims[2];
    const W_out = out.shape.dims[3];
    const ref = try allocator.alloc(f32, out.data.len);
    defer allocator.free(ref);
    for (0..N) |n| for (0..C_out) |co| {
        const plane = ref[(n * C_out + co) * H_out * W_out ..][0 .. H_out * W_out];
        @memset(plane, if (bias) |b| b.data[co] else 0.0);
    };
    for (0..N) |n| for (0..C_in) |ci| for (0..H) |h| for (0..W) |wi| for (0..C_out) |co| for (0..KH) |kh| for (0..KW) |kw| {
        const oh = @as(isize, @intCast(h * stride + kh)) - @as(isize, @intCast(padding));
        const ow = @as(isize, @intCast(wi * stride + kw)) - @as(isize, @intCast(padding));
        if (oh < 0 or ow < 0 or oh >= H_out or ow >= W_out) continue;
        const o_idx = ((n * C_out + co) * H_out + @as(usize, @intCast(oh))) * W_out + @as(usize, @intCast(ow));
        ref[o_idx] += atCoord(x, &.{ n, ci, h, wi }) * atCoord(w, &.{ ci, co, kh, kw });
    };
    for (ref, out.data) |r, o| try testing.expectApproxEqAbs(r, o, 1e-4);
}

test "audit: im2col conv and GEMM transposed conv match naive references" {
    const allocator = testing.allocator;
    const Case = struct { stride: usize, padding: usize, kh: usize, kw: usize, bias: bool, transpose_x: bool };
    const cases = [_]Case{
        .{ .stride = 1, .padding = 0, .kh = 3, .kw = 3, .bias = true, .transpose_x = false },
        .{ .stride = 1, .padding = 1, .kh = 3, .kw = 2, .bias = false, .transpose_x = true },
        .{ .stride = 2, .padding = 1, .kh = 3, .kw = 3, .bias = true, .transpose_x = false },
        .{ .stride = 3, .padding = 2, .kh = 2, .kw = 4, .bias = true, .transpose_x = true },
        .{ .stride = 2, .padding = 0, .kh = 1, .kw = 1, .bias = false, .transpose_x = false },
    };
    for (cases, 0..) |cs, ci| {
        // 输入 [2, 3, 6, 7]（可选：由 [2, 3, 7, 6] 的转置视图得到非连续输入）
        const x_base = if (cs.transpose_x) try zeros(allocator, &.{ 2, 3, 7, 6 }) else try zeros(allocator, &.{ 2, 3, 6, 7 });
        defer free(allocator, x_base);
        fillDeterministic(x_base, 200 + ci);
        const x = if (cs.transpose_x) try x_base.transposeView(2, 3, allocator) else x_base;
        defer if (cs.transpose_x) free(allocator, x);

        const w = try zeros(allocator, &.{ 4, 3, cs.kh, cs.kw });
        defer free(allocator, w);
        fillDeterministic(w, 300 + ci);
        const b = try zeros(allocator, &.{4});
        defer free(allocator, b);
        fillDeterministic(b, 400 + ci);
        const bias: ?*Tensor = if (cs.bias) b else null;

        const y = try x.conv2d(w, bias, .{ .stride = cs.stride, .padding = cs.padding }, allocator);
        defer free(allocator, y);
        try referenceConv2d(x, w, bias, cs.stride, cs.padding, y);

        // 转置卷积权重 [C_in = 3, C_out = 4, KH, KW]
        const wt = try zeros(allocator, &.{ 3, 4, cs.kh, cs.kw });
        defer free(allocator, wt);
        fillDeterministic(wt, 500 + ci);
        if ((6 - 1) * cs.stride + cs.kh > 2 * cs.padding and (7 - 1) * cs.stride + cs.kw > 2 * cs.padding) {
            const yt = try x.convTranspose2d(wt, bias, .{ .stride = cs.stride, .padding = cs.padding }, allocator);
            defer free(allocator, yt);
            try referenceConvTranspose2d(allocator, x, wt, bias, cs.stride, cs.padding, yt);
        }

        // 一维卷积 / 转置卷积：取第 0 行作为 [N, C, L] 输入
        const x1 = try zeros(allocator, &.{ 2, 3, 7 });
        defer free(allocator, x1);
        fillDeterministic(x1, 600 + ci);
        const w1 = try zeros(allocator, &.{ 4, 3, cs.kw });
        defer free(allocator, w1);
        fillDeterministic(w1, 700 + ci);
        const y1 = try x1.conv1d(w1, bias, .{ .stride = cs.stride, .padding = cs.padding }, allocator);
        defer free(allocator, y1);
        const x1_4d = try x1.reshape(&.{ 2, 3, 1, 7 }, allocator);
        defer free(allocator, x1_4d);
        const w1_4d = try w1.reshape(&.{ 4, 3, 1, cs.kw }, allocator);
        defer free(allocator, w1_4d);
        const y1_4d = try y1.reshape(&.{ 2, 4, 1, y1.shape.dims[2] }, allocator);
        defer free(allocator, y1_4d);
        // 一维卷积只在长度方向填充
        const N = 2;
        const L_out = y1.shape.dims[2];
        for (0..N) |n| for (0..4) |co| for (0..L_out) |ol| {
            var acc: f32 = if (bias) |bb| bb.data[co] else 0.0;
            for (0..3) |c_in| for (0..cs.kw) |k| {
                const il = @as(isize, @intCast(ol * cs.stride + k)) - @as(isize, @intCast(cs.padding));
                if (il < 0 or il >= 7) continue;
                acc += x1.data[(n * 3 + c_in) * 7 + @as(usize, @intCast(il))] * w1.data[(co * 3 + c_in) * cs.kw + k];
            };
            try testing.expectApproxEqAbs(acc, y1.data[(n * 4 + co) * L_out + ol], 1e-4);
        };

        const wt1 = try zeros(allocator, &.{ 3, 4, cs.kw });
        defer free(allocator, wt1);
        fillDeterministic(wt1, 800 + ci);
        if ((7 - 1) * cs.stride + cs.kw > 2 * cs.padding) {
            const yt1 = try x1.convTranspose1d(wt1, bias, .{ .stride = cs.stride, .padding = cs.padding }, allocator);
            defer free(allocator, yt1);
            const Lt = yt1.shape.dims[2];
            const ref = try allocator.alloc(f32, yt1.data.len);
            defer allocator.free(ref);
            for (0..N) |n| for (0..4) |co| @memset(ref[(n * 4 + co) * Lt ..][0..Lt], if (bias) |bb| bb.data[co] else 0.0);
            for (0..N) |n| for (0..3) |c_in| for (0..7) |l| for (0..4) |co| for (0..cs.kw) |k| {
                const o = @as(isize, @intCast(l * cs.stride + k)) - @as(isize, @intCast(cs.padding));
                if (o < 0 or o >= Lt) continue;
                ref[(n * 4 + co) * Lt + @as(usize, @intCast(o))] += x1.data[(n * 3 + c_in) * 7 + l] * wt1.data[(c_in * 4 + co) * cs.kw + k];
            };
            for (ref, yt1.data) |r, o| try testing.expectApproxEqAbs(r, o, 1e-4);
        }
    }
}
