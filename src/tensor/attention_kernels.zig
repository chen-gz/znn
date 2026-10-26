//! 缩放点积注意力 (Scaled Dot-Product Attention, SDPA) 共享内核：
//! - `attendOneQuery` / `weightedSum`：单查询打分 -> Softmax -> 加权求和，供各模型的键值缓存 (Key-Value Cache, KVCache)
//!   增量解码循环复用；
//! - `attentionForward`：多头 / 分组查询注意力 (Grouped-Query Attention, GQA) 批量前向，供计算图算子 `Graph.attention`
//!   的前向与前向重演使用 (反向内核位于 `autodiff/backward_nn.zig`，复用本文件保存的注意力概率)。
//! 掩码的可见性规则统一由 `tensor/attention.zig` 定义。

const std = @import("std");
const backend = @import("../backend.zig");
const c = backend.cblas;
const attention = @import("attention.zig");
const vec_math = backend.vec_math;
const Tensor = @import("core.zig").Tensor;

pub const Mask = attention.Mask;
pub const KeyRange = attention.KeyRange;
pub const visibleRange = attention.visibleRange;
pub const fillAdditiveMask = attention.fillAdditiveMask;

/// 注意力算子配置 (遵循 AGENTS.md §4：所有字段均有默认值)
pub const Options = struct {
    /// Softmax 之前的打分缩放系数；为 null 时取 1 / sqrt(head_dim)
    scale: ?f32 = null,
    /// 掩码 (无掩码 / 因果 / 滑动窗口)
    mask: Mask = .none,
    /// 每个键值头服务的查询头数 (分组查询注意力 (Grouped-Query Attention, GQA) 的组大小)：q_heads = kv_heads * kv_groups
    kv_groups: usize = 1,
    /// 第 0 个查询的绝对位置
    q_pos0: usize = 0,
    /// 第 0 个键的绝对位置
    k_pos0: usize = 0,

    pub const default: Options = .{};
    pub fn defaultOptions() Options {
        return .{};
    }

    /// 实际使用的打分缩放系数
    pub fn resolveScale(self: Options, head_dim: usize) f32 {
        return self.scale orelse 1.0 / @sqrt(@as(f32, @floatFromInt(@max(head_dim, 1))));
    }
};

/// 注意力输入形状：q [B, q_heads, tq, head_dim]、k [B, kv_heads, tk, head_dim]、v [B, kv_heads, tk, v_dim]
pub const Shape = struct {
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    tq: usize,
    tk: usize,
    head_dim: usize,
    v_dim: usize,

    /// 校验四维输入并给出形状；头数不满足 q_heads = kv_heads * kv_groups 或其余维度不一致时返回 `error.ShapeMismatch`
    pub fn init(q: *const Tensor, k: *const Tensor, v: *const Tensor, options: Options) !Shape {
        if (q.shape.len != 4 or k.shape.len != 4 or v.shape.len != 4) return error.IncompatibleDimensions;
        const s: Shape = .{
            .batch = q.shape.dims[0],
            .q_heads = q.shape.dims[1],
            .kv_heads = k.shape.dims[1],
            .tq = q.shape.dims[2],
            .tk = k.shape.dims[2],
            .head_dim = q.shape.dims[3],
            .v_dim = v.shape.dims[3],
        };
        if (options.kv_groups == 0 or s.kv_heads * options.kv_groups != s.q_heads) return error.ShapeMismatch;
        if (k.shape.dims[0] != s.batch or v.shape.dims[0] != s.batch) return error.ShapeMismatch;
        if (k.shape.dims[3] != s.head_dim or v.shape.dims[1] != s.kv_heads or v.shape.dims[2] != s.tk) return error.ShapeMismatch;
        return s;
    }

    pub fn groups(self: Shape) usize {
        return self.q_heads / self.kv_heads;
    }

    /// 注意力概率 [B, q_heads, tq, tk] 的元素数
    pub fn probsLen(self: Shape) usize {
        return self.batch * self.q_heads * self.tq * self.tk;
    }

    /// 同一键值头服务的全部查询行 (groups * tq 行) 在查询、输出与概率张量中连续存放；返回第 (b, kv_h) 组的首行下标
    pub fn groupRow(self: Shape, b: usize, kv_h: usize) usize {
        return (b * self.q_heads + kv_h * self.groups()) * self.tq;
    }

    /// 第 (b, kv_h) 个键值头在键 / 值张量中的首行下标
    pub fn kvRow(self: Shape, b: usize, kv_h: usize) usize {
        return (b * self.kv_heads + kv_h) * self.tk;
    }
};

/// 加权求和 out = Σ_j weights[j] * values[j]，values 为 [weights.len, out.len] 行主序
pub fn weightedSum(weights: []const f32, values: []const f32, out: []f32) void {
    const dv = out.len;
    std.debug.assert(values.len >= weights.len * dv);
    @memset(out, 0.0);
    for (weights, 0..) |w, j| vec_math.axpy(out, w, values[j * dv .. (j + 1) * dv]);
}

/// 单查询注意力：对键块中可见区间 range 内的键计算 scale * (q · k_j)，做数值稳定 Softmax 后对值加权求和写入 out。
/// keys 为 [k_len, q.len]、values 为 [k_len, out.len] 行主序键块；scores 至少容纳 range.len() 个元素。
/// 没有可见键时 out 置零。
pub fn attendOneQuery(
    q: []const f32,
    keys: []const f32,
    values: []const f32,
    range: KeyRange,
    scale: f32,
    scores: []f32,
    out: []f32,
) void {
    const d = q.len;
    const dv = out.len;
    const n = range.len();
    if (n == 0) {
        @memset(out, 0.0);
        return;
    }
    const s = scores[0..n];
    for (s, range.begin..) |*score, j| score.* = vec_math.dot(q, keys[j * d .. (j + 1) * d]) * scale;
    vec_math.softmaxRowInPlace(s);
    weightedSum(s, values[range.begin * dv .. range.end * dv], out);
}

/// 把一行原始点积打分就地变换为注意力概率：可见区间内乘以 scale 后做 Softmax，区间外置零
pub fn probabilitiesInPlace(row: []f32, range: KeyRange, scale: f32) void {
    @memset(row[0..range.begin], 0.0);
    @memset(row[range.end..], 0.0);
    const visible = row[range.begin..range.end];
    if (visible.len == 0) return;
    for (visible) |*s| s.* *= scale;
    vec_math.softmaxRowInPlace(visible);
}

/// 批量注意力前向：out [B, q_heads, tq, v_dim] = softmax(scale * Q K^T + M) V。
/// 每个键值头与其服务的 groups 个查询头一起计算 (查询行连续)，分组查询注意力 (Grouped-Query Attention, GQA)
/// 不需要物化复制后的键值；两次矩阵乘法交给单精度通用矩阵乘法 (Single-precision General Matrix Multiply, SGEMM)。
/// probs 为 [B, q_heads, tq, tk] 时保存注意力概率供反向传播使用；为 null 时使用 allocator 分配的单组临时缓冲。
/// 没有可见键的查询输出 0。输入须为行主序连续张量。
pub fn attentionForward(
    q: []const f32,
    k: []const f32,
    v: []const f32,
    out: []f32,
    probs: ?[]f32,
    shape: Shape,
    options: Options,
    allocator: std.mem.Allocator,
) !void {
    const groups = shape.groups();
    const rows = groups * shape.tq;
    const d = shape.head_dim;
    const dv = shape.v_dim;
    const tk = shape.tk;
    if (probs) |p| std.debug.assert(p.len == shape.probsLen());
    if (rows == 0 or tk == 0) {
        @memset(out, 0.0);
        if (probs) |p| @memset(p, 0.0);
        return;
    }
    const scale = options.resolveScale(d);
    const scratch: []f32 = if (probs == null) try allocator.alloc(f32, rows * tk) else &.{};
    defer if (probs == null) allocator.free(scratch);

    for (0..shape.batch) |b| {
        for (0..shape.kv_heads) |kv_h| {
            const q_row = shape.groupRow(b, kv_h);
            const k_row = shape.kvRow(b, kv_h);
            const q_blk = q[q_row * d ..][0 .. rows * d];
            const k_blk = k[k_row * d ..][0 .. tk * d];
            const v_blk = v[k_row * dv ..][0 .. tk * dv];
            const o_blk = out[q_row * dv ..][0 .. rows * dv];
            const p_blk = if (probs) |p| p[q_row * tk ..][0 .. rows * tk] else scratch;

            // 打分 S[rows, tk] = Q_g K^T
            c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans, @intCast(rows), @intCast(tk), @intCast(d), 1.0, q_blk.ptr, @intCast(@max(d, 1)), k_blk.ptr, @intCast(@max(d, 1)), 0.0, p_blk.ptr, @intCast(tk));
            for (0..rows) |r| {
                const range = visibleRange(options.q_pos0 + r % shape.tq, options.k_pos0, tk, options.mask);
                probabilitiesInPlace(p_blk[r * tk .. (r + 1) * tk], range, scale);
            }
            // 输出 O[rows, dv] = P V
            if (dv > 0) c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans, @intCast(rows), @intCast(dv), @intCast(tk), 1.0, p_blk.ptr, @intCast(tk), v_blk.ptr, @intCast(dv), 0.0, o_blk.ptr, @intCast(dv));
        }
    }
}

/// 张量入口：校验形状后对连续张量执行 `attentionForward`，out 须为 [B, q_heads, tq, v_dim] 连续张量
pub fn attentionInto(q: *const Tensor, k: *const Tensor, v: *const Tensor, out: *Tensor, probs: ?[]f32, options: Options, allocator: std.mem.Allocator) !void {
    if (!q.isContiguous() or !k.isContiguous() or !v.isContiguous() or !out.isContiguous()) return error.NonContiguousTensor;
    const shape = try Shape.init(q, k, v, options);
    if (out.data.len != shape.batch * shape.q_heads * shape.tq * shape.v_dim) return error.ShapeMismatch;
    try attentionForward(q.data, k.data, v.data, out.data, probs, shape, options, allocator);
}

/// 逐查询参考实现：每个查询独立调用 `attendOneQuery`，与批量前向数学等价
fn referenceAttention(q: []const f32, k: []const f32, v: []const f32, out: []f32, shape: Shape, options: Options, scores: []f32) void {
    const d = shape.head_dim;
    const dv = shape.v_dim;
    const groups = shape.groups();
    for (0..shape.batch) |b| for (0..shape.q_heads) |h| for (0..shape.tq) |i| {
        const row = (b * shape.q_heads + h) * shape.tq + i;
        const k_row = shape.kvRow(b, h / groups);
        const range = visibleRange(options.q_pos0 + i, options.k_pos0, shape.tk, options.mask);
        attendOneQuery(q[row * d ..][0..d], k[k_row * d ..][0 .. shape.tk * d], v[k_row * dv ..][0 .. shape.tk * dv], range, options.resolveScale(d), scores, out[row * dv ..][0..dv]);
    };
}

test "attentionForward matches per-query attendOneQuery for GQA, Tq != Tk and every mask" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    const shape: Shape = .{ .batch = 2, .q_heads = 4, .kv_heads = 2, .tq = 3, .tk = 5, .head_dim = 4, .v_dim = 3 };
    var q: [2 * 4 * 3 * 4]f32 = undefined;
    var k: [2 * 2 * 5 * 4]f32 = undefined;
    var v: [2 * 2 * 5 * 3]f32 = undefined;
    for (&q) |*x| x.* = rand.float(f32) * 2.0 - 1.0;
    for (&k) |*x| x.* = rand.float(f32) * 2.0 - 1.0;
    for (&v) |*x| x.* = rand.float(f32) * 2.0 - 1.0;

    const masks = [_]Mask{ .none, .causal, .{ .sliding = 2 } };
    for (masks) |mask| {
        const options: Options = .{ .mask = mask, .kv_groups = 2, .q_pos0 = 2, .k_pos0 = 0, .scale = 0.7 };
        var out: [2 * 4 * 3 * 3]f32 = undefined;
        var probs: [2 * 4 * 3 * 5]f32 = undefined;
        try attentionForward(&q, &k, &v, &out, &probs, shape, options, allocator);
        var expected: [out.len]f32 = undefined;
        var scores: [5]f32 = undefined;
        referenceAttention(&q, &k, &v, &expected, shape, options, &scores);
        for (out, expected) |a, e| try std.testing.expectApproxEqAbs(e, a, 1e-5);
        // 概率行和为 1 (或无可见键时为 0)，不可见位置精确为 0
        for (0..shape.batch * shape.q_heads * shape.tq) |row| {
            const i = row % shape.tq;
            const range = visibleRange(options.q_pos0 + i, 0, shape.tk, mask);
            var total: f32 = 0.0;
            for (probs[row * 5 .. (row + 1) * 5], 0..) |p, j| {
                if (j < range.begin or j >= range.end) try std.testing.expectEqual(@as(f32, 0.0), p);
                total += p;
            }
            try std.testing.expectApproxEqAbs(@as(f32, if (range.len() > 0) 1.0 else 0.0), total, 1e-5);
        }
        // 不保存概率时结果一致
        var out_noprobs: [out.len]f32 = undefined;
        try attentionForward(&q, &k, &v, &out_noprobs, null, shape, options, allocator);
        try std.testing.expectEqualSlices(f32, &out, &out_noprobs);
    }
}

test "attendOneQuery outputs zero when no key is visible" {
    const q = [_]f32{ 1.0, 2.0 };
    const keys = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const values = [_]f32{ 3.0, 4.0 };
    var scores: [2]f32 = undefined;
    var out = [_]f32{7.0};
    attendOneQuery(&q, &keys, &values, .{ .begin = 1, .end = 1 }, 1.0, &scores, &out);
    try std.testing.expectEqual(@as(f32, 0.0), out[0]);
    attendOneQuery(&q, &keys, &values, .{ .begin = 0, .end = 2 }, 1.0, &scores, &out);
    const p1 = 1.0 / (1.0 + @exp(@as(f32, 1.0)));
    try std.testing.expectApproxEqAbs(3.0 * p1 + 4.0 * (1.0 - p1), out[0], 1e-6);
}
