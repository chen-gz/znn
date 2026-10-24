//! 旋转位置编码表 (Rotary Position Embedding Table, RopeTable)：CPU 前向内核 `tensor.rope`、其反向传播、
//! 多头潜在注意力 (Multi-head Latent Attention, MLA) 增量解码等所有 CPU 路径共用的唯一旋转实现。
//! 频率与旋转角数量来自 `backend.rope.RopeOptions`，与 GPU 路径共用同一公式。

const std = @import("std");
const backend_rope = @import("../backend/rope.zig");

const RopeOptions = backend_rope.RopeOptions;

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
    inv_freq: [backend_rope.max_angles]f32,

    /// 旋转偏移超出行宽时返回 `error.DimensionOutOfBounds`，旋转角数超过 `backend.rope.max_angles` 时返回 `error.InvalidShape`
    pub fn init(head_dim: usize, options: RopeOptions) !RopeTable {
        const rot_dim = try options.rotaryDim(head_dim);
        const num_angles = options.numAngles(rot_dim);
        if (num_angles > backend_rope.max_angles) return error.InvalidShape;
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
                try std.testing.expectApproxEqAbs(n0, n1, 1e-5);
            }
        }

        table.applyRow(&row, &cos, &sin, true);
        for (orig, row) |o, r| try std.testing.expectApproxEqAbs(o, r, 1e-6);
    }
    try std.testing.expectError(error.DimensionOutOfBounds, RopeTable.init(4, .{ .rotary_offset = 5 }));
}
