//! 旋转位置编码表 (Rotary Position Embedding Table, RopeTable)：CPU 前向内核 `tensor.rope`、其反向传播、
//! 多头潜在注意力 (Multi-head Latent Attention, MLA) 增量解码等所有 CPU 路径共用的唯一旋转实现。
//! 包含旋转参数配置 RopeOptions、频率表计算与就地旋转内核。

const std = @import("std");

/// 预计算频率表支持的最大旋转角数 (rot_dim / 2)
pub const max_angles: usize = 1024;

/// 旋转位置编码可选参数配置 (Rotary Position Embedding Options, RopeOptions)。
/// 对长度为 head_dim 的行，旋转子空间为 [rotary_offset, head_dim)，记 rot_dim = head_dim - rotary_offset：
/// - 第 i 个旋转角 (0 <= i < num_angles) 的频率为 1 / base^(2i / rot_dim)，频率分母始终是完整的 rot_dim；
/// - num_angles = floor(rot_dim / 2 * partial_rotary_factor)，其余角度保持恒等 (cos = 1, sin = 0)；
/// - 配对拓扑见 `Mode`，半切分配对的两个分量始终相距 rot_dim / 2 (而不是 num_angles)。
pub const RopeOptions = struct {
    rotary_offset: usize = 0, // 旋转子空间起始偏移量，默认为 0
    base: f32 = 10000.0, // 频率基数，默认为 10000.0
    mode: Mode = .interleaved, // 旋转配对拓扑模式：相邻配对 (interleaved) 或半切分配对 (split_half)
    partial_rotary_factor: f32 = 1.0, // 部分旋转因子 (0.0 < factor <= 1.0)

    pub const Mode = enum {
        interleaved, // 传统 RoPE 相邻配对拓扑 (2i, 2i+1)
        split_half, // 现代 GPT-NeoX / LLaMA / Gemma 规范的半切分配对拓扑 (i, i + D/2)
    };

    pub const default: RopeOptions = .{};

    pub fn defaultOptions() RopeOptions {
        return .{};
    }

    /// 比例旋转位置编码 (Proportional RoPE) 预设，与 Hugging Face `proportional` RoPE 及 mlx-lm `ProportionalRoPE` 一致：
    pub fn proportional(theta: f32, partial_rotary_factor: f32) RopeOptions {
        return .{ .mode = .split_half, .base = theta, .partial_rotary_factor = partial_rotary_factor };
    }

    /// 旋转子空间维度 rot_dim = head_dim - rotary_offset；偏移超出行宽时返回 `error.DimensionOutOfBounds`
    pub fn rotaryDim(self: RopeOptions, head_dim: usize) !usize {
        if (self.rotary_offset > head_dim) return error.DimensionOutOfBounds;
        return head_dim - self.rotary_offset;
    }

    /// 参与旋转的角度数 floor(rot_dim / 2 * partial_rotary_factor) (因子截断到 [0, 1])
    pub fn numAngles(self: RopeOptions, rot_dim: usize) usize {
        const factor = std.math.clamp(self.partial_rotary_factor, 0.0, 1.0);
        return @intFromFloat(@as(f32, @floatFromInt(rot_dim / 2)) * factor);
    }

    /// 第 i 个旋转角的周期 base^(2i / rot_dim)
    pub fn period(self: RopeOptions, rot_dim: usize, i: usize) f32 {
        return std.math.pow(f32, self.base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(rot_dim)));
    }

    /// 第 i 个旋转角的频率 1 / base^(2i / rot_dim)
    pub fn inverseFrequency(self: RopeOptions, rot_dim: usize, i: usize) f32 {
        return 1.0 / self.period(rot_dim, i);
    }

    /// 填充 [rot_dim / 2] 周期表
    pub fn fillPeriods(self: RopeOptions, rot_dim: usize, out: []f32) void {
        std.debug.assert(out.len == rot_dim / 2);
        const num_angles = self.numAngles(rot_dim);
        for (out, 0..) |*f, i| {
            f.* = if (i < num_angles) self.period(rot_dim, i) else std.math.inf(f32);
        }
    }
};

/// 长度为 head_dim 的行在给定 `RopeOptions` 下的旋转表：init 时预计算每个旋转角的频率，
/// `angles` 为一个位置计算 cos / sin (每个位置只算一次三角函数，同一位置的所有行复用)，
/// `applyRow` 原地旋转一行 (inverse = true 时施加逆旋转，即反向传播所需的转置)。
pub const RopeTable = struct {
    head_dim: usize,
    rotary_offset: usize,
    mode: RopeOptions.Mode,
    /// 半切分配对两个分量的间距 rot_dim / 2
    half: usize,
    num_angles: usize,
    inv_freq: [max_angles]f32,

    /// 旋转偏移超出行宽时返回 `error.DimensionOutOfBounds`，旋转角数超过 `max_angles` 时返回 `error.InvalidShape`
    pub fn init(head_dim: usize, options: RopeOptions) !RopeTable {
        const rot_dim = try options.rotaryDim(head_dim);
        const num_angles = options.numAngles(rot_dim);
        if (num_angles > max_angles) return error.InvalidShape;
        var table = RopeTable{
            .head_dim = head_dim,
            .rotary_offset = options.rotary_offset,
            .mode = options.mode,
            .half = rot_dim / 2,
            .num_angles = num_angles,
            .inv_freq = undefined,
        };
        for (table.inv_freq[0..num_angles], 0..) |*f, i| f.* = options.inverseFrequency(rot_dim, i);
        return table;
    }

    /// 位置 pos 处每个旋转角的 cos(pos * f_i) 与 sin(pos * f_i)；cos / sin 的长度必须为 num_angles
    pub fn angles(self: *const RopeTable, pos: usize, cos: []f32, sin: []f32) void {
        std.debug.assert(cos.len == self.num_angles and sin.len == self.num_angles);
        const pos_f: f32 = @floatFromInt(pos);
        for (self.inv_freq[0..self.num_angles], cos, sin) |freq, *c, *s| {
            const theta = pos_f * freq;
            c.* = @cos(theta);
            s.* = @sin(theta);
        }
    }

    /// 用 `angles` 的结果原地旋转长度为 head_dim 的一行；未参与旋转的分量保持不变。
    /// inverse = true 时施加逆旋转 (sin 取负)，即正向旋转矩阵的转置。
    pub fn applyRow(self: *const RopeTable, row: []f32, cos: []const f32, sin: []const f32, comptime inverse: bool) void {
        std.debug.assert(row.len == self.head_dim);
        std.debug.assert(cos.len == self.num_angles and sin.len == self.num_angles);
        const rot = row[self.rotary_offset..];
        switch (self.mode) {
            .interleaved => for (cos, sin, 0..) |c, s_raw, i| {
                const s = if (inverse) -s_raw else s_raw;
                const x0 = rot[2 * i];
                const x1 = rot[2 * i + 1];
                rot[2 * i] = x0 * c - x1 * s;
                rot[2 * i + 1] = x0 * s + x1 * c;
            },
            .split_half => for (cos, sin, 0..) |c, s_raw, i| {
                const s = if (inverse) -s_raw else s_raw;
                const x0 = rot[i];
                const x1 = rot[i + self.half];
                rot[i] = x0 * c - x1 * s;
                rot[i + self.half] = x1 * c + x0 * s;
            },
        }
    }
};

test "RopeOptions angle count, frequencies and the proportional preset" {
    const opts = RopeOptions.proportional(1_000_000.0, 0.25);
    try std.testing.expectEqual(RopeOptions.Mode.split_half, opts.mode);
    try std.testing.expectEqual(@as(usize, 64), opts.numAngles(512));
    try std.testing.expectEqual(@as(usize, 512), try opts.rotaryDim(512));
    try std.testing.expectError(error.DimensionOutOfBounds, (RopeOptions{ .rotary_offset = 9 }).rotaryDim(8));
    try std.testing.expectEqual(@as(f32, 1.0), opts.inverseFrequency(512, 0));
    try std.testing.expectApproxEqRel(1.0 / std.math.pow(f32, 1_000_000.0, 2.0 / 512.0), opts.inverseFrequency(512, 1), 1e-6);

    var periods: [256]f32 = undefined;
    opts.fillPeriods(512, &periods);
    for (periods, 0..) |p, i| {
        try std.testing.expectEqual(i >= 64, std.math.isInf(p));
        if (i < 64) try std.testing.expectEqual(opts.period(512, i), p);
    }
    try std.testing.expectEqual(@as(usize, 4), (RopeOptions{ .partial_rotary_factor = 2.0 }).numAngles(8));
    try std.testing.expectEqual(@as(usize, 0), (RopeOptions{ .partial_rotary_factor = -1.0 }).numAngles(8));
}

test "RopeTable rotation preserves norms, leaves unrotated parts unchanged and inverts exactly" {
    inline for (.{ RopeOptions.Mode.interleaved, RopeOptions.Mode.split_half }) |mode| {
        const opts = RopeOptions{ .mode = mode, .rotary_offset = 2, .partial_rotary_factor = 0.5, .base = 500.0 };
        const table = try RopeTable.init(18, opts);
        try std.testing.expectEqual(@as(usize, 8), table.half);
        try std.testing.expectEqual(@as(usize, 4), table.num_angles);

        var row: [18]f32 = undefined;
        for (&row, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i + 1)));
        const orig = row;
        var cos: [4]f32 = undefined;
        var sin: [4]f32 = undefined;
        table.angles(7, &cos, &sin);
        table.applyRow(&row, &cos, &sin, false);

        // 旋转子空间之前的偏移分量与未参与旋转的角度保持不变
        try std.testing.expectEqualSlices(f32, orig[0..2], row[0..2]);
        for (0..table.half) |i| {
            const a = if (mode == .interleaved) 2 + 2 * i else 2 + i;
            const b = if (mode == .interleaved) a + 1 else a + table.half;
            if (i >= table.num_angles) {
                try std.testing.expectEqual(orig[a], row[a]);
                try std.testing.expectEqual(orig[b], row[b]);
            } else {
                const n0 = orig[a] * orig[a] + orig[b] * orig[b];
                const n1 = row[a] * row[a] + row[b] * row[b];
                try std.testing.expectApproxEqRel(n0, n1, 1e-5);
            }
        }

        // inverse = true 完全还原
        table.applyRow(&row, &cos, &sin, true);
        for (orig, row) |o, r| try std.testing.expectApproxEqAbs(o, r, 1e-5);
    }
}
