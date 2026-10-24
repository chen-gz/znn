//! 向量化行原语 (Vectorized Row Primitives)：单指令多数据流 (Single Instruction, Multiple Data, SIMD) 的
//! 指数、求和 / 最大值归约、点积、缩放累加 (a * x + y, AXPY)、归一化指数函数 (Softmax)、对数求和指数 (Log-Sum-Exp, LSE)、
//! 数值稳定的 Sigmoid、对数几率截断 (Softcap) 与脑浮点 16 (Brain Floating Point 16, BF16) 展开。
//! 位于最底层 backend 包，CPU 张量内核、自动微分、增量解码、采样与模型路径共用这一份定义。

const std = @import("std");
const builtin = @import("builtin");

pub const V8 = @Vector(8, f32);

/// 点积与 BF16 展开使用的向量宽度
const lanes = 16;
const V16 = @Vector(lanes, f32);
const U16x16 = @Vector(lanes, u16);
const U32x16 = @Vector(lanes, u32);

/// 目标平台是否具备硬件融合乘加 (Fused Multiply-Add, FMA) 指令；不具备时 `@mulAdd` 会退化为逐元素软件调用，故改用普通乘加
pub const has_fma = switch (builtin.cpu.arch) {
    .x86, .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
    .aarch64, .aarch64_be => std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon),
    else => false,
};

/// 向量乘加 `a * b + c`：有硬件融合乘加时只舍入一次
pub inline fn mulAdd(comptime V: type, a: V, b: V, c: V) V {
    return if (has_fma) @mulAdd(V, a, b, c) else a * b + c;
}

// ============================================================================
// 指数函数
// ============================================================================

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

/// `out[i] = exp(in[i] - shift)`；尾部不足 8 个的元素补齐为一个向量计算，所有元素使用同一逼近。
/// `out` 可以与 `in` 是同一切片 (原地计算)。
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

// ============================================================================
// 行归约
// ============================================================================

/// 连续切片求和：8 路向量累加器打破串行加法的依赖链，比逐元素串行累加更快且舍入误差更小
pub fn sum(values: []const f32) f32 {
    var acc: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= values.len) : (i += 8) {
        acc += values[i..][0..8].*;
    }
    var total: f32 = @reduce(.Add, acc);
    while (i < values.len) : (i += 1) total += values[i];
    return total;
}

/// 连续切片的平方和（8 路向量累加）
pub fn sumSquares(values: []const f32) f32 {
    var acc: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= values.len) : (i += 8) {
        const v: V8 = values[i..][0..8].*;
        acc += v * v;
    }
    var total: f32 = @reduce(.Add, acc);
    while (i < values.len) : (i += 1) total += values[i] * values[i];
    return total;
}

/// 连续切片相对给定中心的偏差平方和 `sum((x - center)^2)`（8 路向量累加）
pub fn sumSquaredDeviations(values: []const f32, center: f32) f32 {
    const c_vec: V8 = @splat(center);
    var acc: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= values.len) : (i += 8) {
        const d = @as(V8, values[i..][0..8].*) - c_vec;
        acc += d * d;
    }
    var total: f32 = @reduce(.Add, acc);
    while (i < values.len) : (i += 1) {
        const d = values[i] - center;
        total += d * d;
    }
    return total;
}

/// 非空连续切片的最大值（与逐元素 `if (v > m) m = v` 扫描结果一致，含 NaN 时语义可能不同，调用方仅用于数值稳定化平移）
pub fn max(values: []const f32) f32 {
    std.debug.assert(values.len > 0);
    var i: usize = 0;
    var m: f32 = values[0];
    if (values.len >= 8) {
        var acc: V8 = values[0..8].*;
        i = 8;
        while (i + 8 <= values.len) : (i += 8) {
            acc = @max(acc, @as(V8, values[i..][0..8].*));
        }
        m = @reduce(.Max, acc);
    }
    while (i < values.len) : (i += 1) m = @max(m, values[i]);
    return m;
}

// ============================================================================
// 点积与缩放累加
// ============================================================================

/// 单精度浮点 (F32) 向量点积 sum(a[i] * b[i])：两组 16 路累加器隐藏融合乘加延迟，尾部逐元素补齐
pub fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc0: V16 = @splat(0.0);
    var acc1: V16 = @splat(0.0);
    var i: usize = 0;
    while (i + 2 * lanes <= a.len) : (i += 2 * lanes) {
        acc0 = mulAdd(V16, a[i..][0..lanes].*, b[i..][0..lanes].*, acc0);
        acc1 = mulAdd(V16, a[i + lanes ..][0..lanes].*, b[i + lanes ..][0..lanes].*, acc1);
    }
    if (i + lanes <= a.len) {
        acc0 = mulAdd(V16, a[i..][0..lanes].*, b[i..][0..lanes].*, acc0);
        i += lanes;
    }
    var total = @reduce(.Add, acc0 + acc1);
    while (i < a.len) : (i += 1) total += a[i] * b[i];
    return total;
}

/// 16 个 BF16 原始位展开为单精度：BF16 位模式左移 16 位即为等值 F32
inline fn widen16(bits: *const [lanes]u16) V16 {
    const raw: U16x16 = bits.*;
    return @bitCast(@as(U32x16, raw) << @splat(16));
}

/// 单精度向量与 BF16 原始位向量的点积：整段以 SIMD 完成类型提升与乘加，累加结构与 `dot` 相同
pub fn dotBf16(a: []const f32, b_bits: []const u16) f32 {
    std.debug.assert(a.len == b_bits.len);
    var acc0: V16 = @splat(0.0);
    var acc1: V16 = @splat(0.0);
    var i: usize = 0;
    while (i + 2 * lanes <= a.len) : (i += 2 * lanes) {
        acc0 = mulAdd(V16, a[i..][0..lanes].*, widen16(b_bits[i..][0..lanes]), acc0);
        acc1 = mulAdd(V16, a[i + lanes ..][0..lanes].*, widen16(b_bits[i + lanes ..][0..lanes]), acc1);
    }
    if (i + lanes <= a.len) {
        acc0 = mulAdd(V16, a[i..][0..lanes].*, widen16(b_bits[i..][0..lanes]), acc0);
        i += lanes;
    }
    var total = @reduce(.Add, acc0 + acc1);
    while (i < a.len) : (i += 1) total += a[i] * bf16ToF32(b_bits[i]);
    return total;
}

/// 缩放累加 y += alpha * x (逐元素先乘后加，结果与标量循环逐位一致)
pub fn axpy(y: []f32, alpha: f32, x: []const f32) void {
    std.debug.assert(y.len == x.len);
    const va: V16 = @splat(alpha);
    var i: usize = 0;
    while (i + lanes <= y.len) : (i += lanes) {
        const vx: V16 = x[i..][0..lanes].*;
        const vy: V16 = y[i..][0..lanes].*;
        y[i..][0..lanes].* = vy + va * vx;
    }
    while (i < y.len) : (i += 1) y[i] += alpha * x[i];
}

/// 缩放复制 dst = alpha * src
pub fn scaleCopy(dst: []f32, src: []const f32, alpha: f32) void {
    std.debug.assert(dst.len == src.len);
    const va: V16 = @splat(alpha);
    var i: usize = 0;
    while (i + lanes <= dst.len) : (i += lanes) {
        const v: V16 = src[i..][0..lanes].*;
        dst[i..][0..lanes].* = v * va;
    }
    while (i < dst.len) : (i += 1) dst[i] = src[i] * alpha;
}

// ============================================================================
// 归一化指数函数 (Softmax) 与对数求和指数 (Log-Sum-Exp, LSE)
// ============================================================================

/// 数值稳定的单行归一化指数函数 (Softmax)：先减去行最大值再求指数 (`exp8`)，避免上溢。
/// `in` 必须非空；`out` 可以与 `in` 是同一切片。
pub fn softmaxRow(in: []const f32, out: []f32) void {
    std.debug.assert(out.len == in.len);
    const shift = max(in);
    expShifted(in, out, shift);
    const inv_sum: V8 = @splat(1.0 / sum(out));
    var i: usize = 0;
    while (i + 8 <= out.len) : (i += 8) {
        out[i..][0..8].* = @as(V8, out[i..][0..8].*) * inv_sum;
    }
    while (i < out.len) : (i += 1) out[i] *= inv_sum[0];
}

/// 原地单行归一化指数函数 (Softmax)，与 `softmaxRow` 结果一致
pub fn softmaxRowInPlace(row: []f32) void {
    softmaxRow(row, row);
}

/// 对数求和指数遇到非数 (Not a Number, NaN) 时的处理方式
pub const NanPolicy = enum {
    /// 任一元素为 NaN 时结果为 NaN (损失函数等需要暴露数值异常的场景)
    propagate,
    /// 跳过 NaN 元素 (采样：NaN 的概率视为 0)
    skip,
};

/// 缩放行 `v_i = x_i * inv_temp` 的对数求和指数部分归约：max 为最大缩放值，sum 为 Σ exp(v_i - max)。
/// 指数与累加使用双精度 (f64)，长词表 (数十万项) 求和不损失精度；-inf 元素贡献 0，全部为 -inf 时 max 为 -inf。
pub const LogSumExp = struct {
    max: f32 = -std.math.inf(f32),
    sum: f64 = 0.0,

    /// 合并两段的部分归约结果 (满足结合律，可用于有序并行归约)
    pub fn merge(a: LogSumExp, b: LogSumExp) LogSumExp {
        if (std.math.isNan(a.max)) return a;
        if (std.math.isNan(b.max)) return b;
        if (b.max == -std.math.inf(f32)) return a;
        if (a.max == -std.math.inf(f32)) return b;
        const m = @max(a.max, b.max);
        return .{
            .max = m,
            .sum = a.sum * @exp(@as(f64, a.max - m)) + b.sum * @exp(@as(f64, b.max - m)),
        };
    }

    /// log Σ exp(v_i)；没有可用元素时为 -inf
    pub fn value(self: LogSumExp) f32 {
        if (std.math.isNan(self.max)) return self.max;
        if (self.max == -std.math.inf(f32)) return self.max;
        return self.max + @as(f32, @floatCast(@log(self.sum)));
    }
};

/// 计算一段行数据的对数求和指数部分归约结果 (见 `LogSumExp`)
pub fn logSumExpPartial(row: []const f32, inv_temp: f32, comptime nan_policy: NanPolicy) LogSumExp {
    var m: f32 = -std.math.inf(f32);
    for (row) |x| {
        const v = x * inv_temp;
        if (std.math.isNan(v)) switch (nan_policy) {
            .propagate => return .{ .max = v, .sum = std.math.nan(f64) },
            .skip => continue,
        };
        m = @max(m, v);
    }
    if (m == -std.math.inf(f32)) return .{};
    var total: f64 = 0.0;
    for (row) |x| {
        const v = x * inv_temp;
        if (nan_policy == .skip and std.math.isNan(v)) continue;
        total += @exp(@as(f64, v - m));
    }
    return .{ .max = m, .sum = total };
}

/// 单行对数求和指数 log Σ exp(x_i * inv_temp)：NaN 传播，全部为 -inf 时返回 -inf
pub fn logSumExpRow(row: []const f32, inv_temp: f32) f32 {
    return logSumExpPartial(row, inv_temp, .propagate).value();
}

// ============================================================================
// 标量激活与截断
// ============================================================================

/// 数值稳定的 Sigmoid：x >= 0 时为 1 / (1 + e^-x)，否则为 e^x / (1 + e^x)，两个分支都不会上溢
pub fn sigmoid(x: f32) f32 {
    if (x >= 0.0) return 1.0 / (1.0 + @exp(-x));
    const e = @exp(x);
    return e / (1.0 + e);
}

/// 对数几率截断 (Logit Softcapping)：cap * tanh(x / cap)；cap <= 0 时不截断
pub fn softcap(x: f32, cap: f32) f32 {
    if (cap <= 0.0) return x;
    return cap * std.math.tanh(x / cap);
}

// ============================================================================
// 脑浮点 16 (Brain Floating Point 16, BF16) 展开
// ============================================================================

/// 单个 BF16 原始位展开为等值单精度浮点
pub fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// BF16 原始位切片展开为单精度并乘以 alpha：dst[i] = f32(src[i]) * alpha (alpha = 1 时为纯展开)
pub fn widenBf16(dst: []f32, src: []const u16, alpha: f32) void {
    std.debug.assert(dst.len == src.len);
    const va: V16 = @splat(alpha);
    var i: usize = 0;
    while (i + lanes <= dst.len) : (i += lanes) {
        dst[i..][0..lanes].* = widen16(src[i..][0..lanes]) * va;
    }
    while (i < dst.len) : (i += 1) dst[i] = bf16ToF32(src[i]) * alpha;
}

/// 小端序 BF16 原始字节展开为单精度 (逐元素读取，不要求源地址对齐)
pub fn widenBf16Bytes(dst: []f32, bytes: []const u8) void {
    std.debug.assert(bytes.len >= dst.len * 2);
    for (dst, 0..) |*d, idx| d.* = bf16ToF32(std.mem.readInt(u16, bytes[idx * 2 ..][0..2], .little));
}

// ============================================================================
// 单元测试
// ============================================================================

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

/// 测试用的确定性伪随机数据
fn fillTestData(values: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (values) |*v| v.* = random.floatNorm(f32);
}

test "dot, dotBf16, axpy and scaleCopy match scalar references across vector body and tail lengths" {
    var a: [77]f32 = undefined;
    var b: [77]f32 = undefined;
    fillTestData(&a, 1);
    fillTestData(&b, 2);
    var b_bits: [77]u16 = undefined;
    for (&b_bits, &b) |*bits, *v| {
        bits.* = @truncate(@as(u32, @bitCast(v.*)) >> 16);
        v.* = bf16ToF32(bits.*);
    }
    inline for (.{ 0, 1, 15, 16, 17, 31, 32, 33, 48, 77 }) |n| {
        var ref: f64 = 0;
        for (a[0..n], b[0..n]) |x, y| ref += @as(f64, x) * y;
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(ref)), dot(a[0..n], b[0..n]), 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(ref)), dotBf16(a[0..n], b_bits[0..n]), 1e-4);

        var y: [n]f32 = undefined;
        @memcpy(&y, b[0..n]);
        axpy(&y, 0.75, a[0..n]);
        for (y, a[0..n], b[0..n]) |got, x, y0| try std.testing.expectEqual(y0 + 0.75 * x, got);

        var s: [n]f32 = undefined;
        scaleCopy(&s, a[0..n], -1.5);
        for (s, a[0..n]) |got, x| try std.testing.expectEqual(x * -1.5, got);

        var w: [n]f32 = undefined;
        widenBf16(&w, b_bits[0..n], 2.0);
        for (w, b[0..n]) |got, x| try std.testing.expectEqual(x * 2.0, got);
    }
}

test "widenBf16Bytes decodes unaligned little-endian BF16 bytes" {
    const values = [_]f32{ 1.0, -2.5, 0.15625, std.math.inf(f32) };
    var bytes: [1 + 2 * values.len]u8 = undefined;
    for (values, 0..) |v, i| {
        const bits: u16 = @truncate(@as(u32, @bitCast(v)) >> 16);
        std.mem.writeInt(u16, bytes[1 + 2 * i ..][0..2], bits, .little);
    }
    var out: [values.len]f32 = undefined;
    widenBf16Bytes(&out, bytes[1..]);
    try std.testing.expectEqualSlices(f32, &values, &out);
}

test "softmaxRow normalizes, is shift invariant and supports in-place rows" {
    var row: [19]f32 = undefined;
    fillTestData(&row, 3);
    var out: [19]f32 = undefined;
    softmaxRow(&row, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum(&out), 1e-6);
    const m = max(&row);
    var exp_sum: f64 = 0;
    for (row) |v| exp_sum += @exp(@as(f64, v - m));
    for (row, out) |v, p| try std.testing.expectApproxEqRel(@as(f32, @floatCast(@exp(@as(f64, v - m)) / exp_sum)), p, 1e-6);

    var shifted = row;
    for (&shifted) |*v| v.* += 100.0;
    softmaxRowInPlace(&shifted);
    for (out, shifted) |p, q| try std.testing.expectApproxEqAbs(p, q, 1e-6);
}

test "logSumExpRow is stable, honours inv_temp and the NaN policy" {
    var row: [40]f32 = undefined;
    fillTestData(&row, 4);
    row[3] = 500.0;
    row[7] = -std.math.inf(f32);
    var ref: f64 = 0;
    for (row) |v| ref += @exp(@as(f64, v * 0.5) - 250.0);
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(250.0 + @log(ref))), logSumExpRow(&row, 0.5), 1e-6);

    // 分段归约后合并与整行一次归约一致
    const merged = LogSumExp.merge(logSumExpPartial(row[0..13], 0.5, .propagate), logSumExpPartial(row[13..], 0.5, .propagate));
    try std.testing.expectApproxEqRel(logSumExpRow(&row, 0.5), merged.value(), 1e-6);

    const neg_inf = [_]f32{ -std.math.inf(f32), -std.math.inf(f32) };
    try std.testing.expect(std.math.isNegativeInf(logSumExpRow(&neg_inf, 1.0)));

    const with_nan = [_]f32{ 1.0, std.math.nan(f32), 2.0 };
    try std.testing.expect(std.math.isNan(logSumExpRow(&with_nan, 1.0)));
    const skipped = logSumExpPartial(&with_nan, 1.0, .skip).value();
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(@log(@exp(@as(f64, 1.0)) + @exp(@as(f64, 2.0))))), skipped, 1e-6);
}

test "sigmoid is stable at both tails and softcap is the identity for non-positive caps" {
    try std.testing.expectEqual(@as(f32, 0.5), sigmoid(0.0));
    try std.testing.expectEqual(@as(f32, 1.0), sigmoid(200.0));
    try std.testing.expectEqual(@as(f32, 0.0), sigmoid(-200.0));
    try std.testing.expectApproxEqAbs(1.0 - sigmoid(1.7), sigmoid(-1.7), 1e-7);
    try std.testing.expect(std.math.isNan(sigmoid(std.math.nan(f32))));

    try std.testing.expectEqual(@as(f32, 7.0), softcap(7.0, 0.0));
    try std.testing.expectApproxEqAbs(30.0 * std.math.tanh(@as(f32, 50.0 / 30.0)), softcap(50.0, 30.0), 1e-5);
    try std.testing.expect(softcap(1e6, 30.0) <= 30.0);
}
