//! 向量化初等函数 (Vectorized Elementary Functions)：为 Softmax 等热点循环提供 8 路单指令多数据流
//! (Single Instruction, Multiple Data, SIMD) 指数函数，替代逐元素调用的标量 `@exp`。

const std = @import("std");

pub const V8 = @Vector(8, f32);

/// 8 路单精度指数函数 `exp(x)`：Cody-Waite 区间约化 `x = n * ln2 + r` (|r| <= ln2 / 2) 后以 Cephes 多项式逼近 `e^r`，
/// 再分两步乘以 `2^n`（避免 `2^n` 本身上溢或成为非规格化数）。
/// 规格化结果的相对误差约 2 个最后一位单位 (Units in the Last Place, ULP)；
/// `x > ln(FLT_MAX)` 返回 +inf，`x < -104` 返回 0（含 -inf），NaN 原样返回。
pub fn exp8(x: V8) V8 {
    const log2e: V8 = @splat(1.44269504088896341);
    const ln2_hi: V8 = @splat(0.693359375);
    const ln2_lo: V8 = @splat(-2.12194440e-4);
    const lo: V8 = @splat(-104.0);
    const hi: V8 = @splat(88.72283905206835);
    const half: V8 = @splat(0.5);
    const one: V8 = @splat(1.0);

    const xc = @min(@max(x, lo), hi);
    const fx = @floor(xc * log2e + half);
    const r = (xc - fx * ln2_hi) - fx * ln2_lo;

    var y: V8 = @splat(1.9875691500e-4);
    y = y * r + @as(V8, @splat(1.3981999507e-3));
    y = y * r + @as(V8, @splat(8.3334519073e-3));
    y = y * r + @as(V8, @splat(4.1665795894e-2));
    y = y * r + @as(V8, @splat(1.6666665459e-1));
    y = y * r + @as(V8, @splat(5.0000001201e-1));
    y = y * (r * r) + r + one;

    // 2^n = 2^n1 * 2^n2，n1 = floor(n / 2)，两个因子都是规格化浮点数
    const n: @Vector(8, i32) = @intFromFloat(fx);
    const n1 = n >> @splat(1);
    const n2 = n - n1;
    const bias: @Vector(8, i32) = @splat(127);
    const p1: V8 = @bitCast((n1 + bias) << @splat(23));
    const p2: V8 = @bitCast((n2 + bias) << @splat(23));
    y = (y * p1) * p2;

    const zero: V8 = @splat(0.0);
    const inf: V8 = @splat(std.math.inf(f32));
    y = @select(f32, x < lo, zero, y);
    y = @select(f32, x > hi, inf, y);
    return @select(f32, x != x, x, y);
}

/// `out[i] = exp(in[i] - shift)`；尾部不足 8 个的元素补齐为一个向量计算，所有元素使用同一逼近
pub fn expShifted(in: []const f32, out: []f32, shift: f32) void {
    std.debug.assert(out.len == in.len);
    const s: V8 = @splat(shift);
    var i: usize = 0;
    while (i + 8 <= in.len) : (i += 8) {
        const e: [8]f32 = exp8(@as(V8, in[i..][0..8].*) - s);
        out[i..][0..8].* = e;
    }
    if (i < in.len) {
        const rem = in.len - i;
        var buf: [8]f32 = @splat(0.0);
        @memcpy(buf[0..rem], in[i..]);
        const e: [8]f32 = exp8(@as(V8, buf) - s);
        @memcpy(out[i..], e[0..rem]);
    }
}

test "exp8 matches @exp within 2 ULP-scale relative error and handles special values" {
    var x: f32 = -103.0;
    while (x < 88.7) : (x += 0.0137) {
        const v: V8 = @splat(x);
        const got = exp8(v)[0];
        const want = @exp(x);
        if (want >= std.math.floatMin(f32)) {
            try std.testing.expectApproxEqRel(want, got, 3e-7);
        } else {
            try std.testing.expectApproxEqAbs(want, got, 1e-44);
        }
    }
    const special: V8 = .{ 0.0, -std.math.inf(f32), std.math.inf(f32), std.math.nan(f32), 89.0, -105.0, 1.0, -1.0 };
    const r = exp8(special);
    try std.testing.expectEqual(@as(f32, 1.0), r[0]);
    try std.testing.expectEqual(@as(f32, 0.0), r[1]);
    try std.testing.expect(std.math.isPositiveInf(r[2]));
    try std.testing.expect(std.math.isNan(r[3]));
    try std.testing.expect(std.math.isPositiveInf(r[4]));
    try std.testing.expectEqual(@as(f32, 0.0), r[5]);
    try std.testing.expectApproxEqRel(@as(f32, std.math.e), r[6], 3e-7);
    try std.testing.expectApproxEqRel(@as(f32, 1.0 / std.math.e), r[7], 3e-7);
}

test "expShifted covers vector body and scalar tail" {
    var in: [13]f32 = undefined;
    for (&in, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.5 - 3.0;
    var out: [13]f32 = undefined;
    expShifted(&in, &out, 1.5);
    for (in, out) |v, o| try std.testing.expectApproxEqRel(@exp(v - 1.5), o, 3e-7);
}
