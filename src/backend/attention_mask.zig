//! 注意力掩码规则 (Attention Mask Rules)：CPU 注意力内核 (`tensor.attention_kernels`)、计算图注意力算子
//! (`Graph.attention`)、增量解码循环与 Apple 机器学习拓展 (Machine Learning Explore, MLX) 融合缩放点积注意力
//! (Scaled Dot-Product Attention, SDPA) 共用这一份可见性定义。
//!
//! 位置约定：查询与键都用绝对位置描述；`[tq, tk]` 网格中第 i 个查询的绝对位置为 q_pos0 + i，
//! 第 j 个键的绝对位置为 k_pos0 + j。

const std = @import("std");

/// 加性掩码中不可见位置的取值：-1e9 在单精度与脑浮点 16 (Brain Floating Point 16, BF16) 中都是有限的极大负值，
/// Softmax 之后的权重精确为 0
pub const masked_value: f32 = -1e9;

/// 注意力掩码
pub const Mask = union(enum) {
    /// 无掩码：每个查询可见全部键
    none,
    /// 因果掩码：绝对位置为 p 的查询可见满足 k <= p 的键
    causal,
    /// 滑动窗口掩码 (与 mlx-lm `create_causal_mask(window_size=...)` 一致)：绝对位置为 p 的查询
    /// 只可见满足 0 <= p - k < window 的键 (含自身，共最多 window 个)
    sliding: usize,

    pub const default: Mask = .none;

    /// 绝对位置为 q_pos 的查询能否看到绝对位置为 k_pos 的键
    pub fn isVisible(self: Mask, q_pos: usize, k_pos: usize) bool {
        return switch (self) {
            .none => true,
            .causal => k_pos <= q_pos,
            .sliding => |window| k_pos <= q_pos and q_pos - k_pos < window,
        };
    }

    /// 在 `[tq, tk]` 网格上化简为等价的最简掩码：滑动窗口覆盖网格内全部因果可见键时化简为因果掩码，
    /// 因果掩码在网格内处处可见 (末个键不晚于首个查询) 时化简为无掩码
    pub fn simplify(self: Mask, tq: usize, tk: usize, q_pos0: usize, k_pos0: usize) Mask {
        if (tq == 0 or tk == 0) return .none;
        const first_q = q_pos0;
        const last_q = q_pos0 + tq - 1;
        const last_k = k_pos0 + tk - 1;
        return switch (self) {
            .none => .none,
            .causal => if (last_k <= first_q) .none else .causal,
            .sliding => |window| if (last_q < k_pos0 + window)
                Mask.simplify(.causal, tq, tk, q_pos0, k_pos0)
            else
                self,
        };
    }
};

/// 单个查询在键块中的可见区间 [begin, end) (相对键块起点的下标)；三种掩码的可见键都是连续区间
pub const KeyRange = struct {
    begin: usize,
    end: usize,

    pub fn len(self: KeyRange) usize {
        return self.end - self.begin;
    }
};

/// 绝对位置为 q_pos 的查询在键块 (共 k_len 个键，第 j 个键的绝对位置为 k_pos0 + j) 中的可见区间；
/// 没有可见键时返回空区间 (begin == end)
pub fn visibleRange(q_pos: usize, k_pos0: usize, k_len: usize, mask: Mask) KeyRange {
    const k_end_abs = k_pos0 + k_len;
    var begin_abs = k_pos0;
    var end_abs = k_end_abs;
    switch (mask) {
        .none => {},
        .causal => end_abs = @min(end_abs, q_pos + 1),
        .sliding => |window| {
            end_abs = @min(end_abs, q_pos + 1);
            if (q_pos + 1 > window) begin_abs = @max(begin_abs, q_pos + 1 - window);
        },
    }
    if (end_abs <= begin_abs) return .{ .begin = 0, .end = 0 };
    return .{ .begin = begin_abs - k_pos0, .end = end_abs - k_pos0 };
}

/// 在主机端构建 `[tq, tk]` 行主序加性掩码：可见位置为 0，不可见位置为 `masked_value`
pub fn fillAdditiveMask(dst: []f32, tq: usize, tk: usize, q_pos0: usize, k_pos0: usize, mask: Mask) void {
    std.debug.assert(dst.len == tq * tk);
    for (0..tq) |i| {
        const row = dst[i * tk .. (i + 1) * tk];
        const range = visibleRange(q_pos0 + i, k_pos0, tk, mask);
        @memset(row, masked_value);
        @memset(row[range.begin..range.end], 0.0);
    }
}

test "visibleRange and fillAdditiveMask agree with isVisible for every mask" {
    const masks = [_]Mask{ .none, .causal, .{ .sliding = 1 }, .{ .sliding = 3 }, .{ .sliding = 7 } };
    for (masks) |mask| {
        for (0..6) |q_pos0| {
            for (0..6) |k_pos0| {
                const tq = 4;
                const tk = 5;
                var buf: [tq * tk]f32 = undefined;
                fillAdditiveMask(&buf, tq, tk, q_pos0, k_pos0, mask);
                for (0..tq) |i| {
                    const range = visibleRange(q_pos0 + i, k_pos0, tk, mask);
                    for (0..tk) |j| {
                        const visible = mask.isVisible(q_pos0 + i, k_pos0 + j);
                        try std.testing.expectEqual(visible, j >= range.begin and j < range.end);
                        try std.testing.expectEqual(@as(f32, if (visible) 0.0 else masked_value), buf[i * tk + j]);
                    }
                }
            }
        }
    }
}

test "Mask.simplify keeps the visibility of every grid cell" {
    const masks = [_]Mask{ .none, .causal, .{ .sliding = 2 }, .{ .sliding = 4 }, .{ .sliding = 9 } };
    for (masks) |mask| {
        for (0..5) |q_pos0| {
            for (0..5) |k_pos0| {
                for (1..5) |tq| {
                    for (1..6) |tk| {
                        const simple = mask.simplify(tq, tk, q_pos0, k_pos0);
                        for (0..tq) |i| for (0..tk) |j| {
                            try std.testing.expectEqual(mask.isVisible(q_pos0 + i, k_pos0 + j), simple.isVisible(q_pos0 + i, k_pos0 + j));
                        };
                    }
                }
            }
        }
    }
    // 窗口覆盖整个因果三角时化简为因果掩码；单查询看到全部历史键时化简为无掩码
    try std.testing.expectEqual(Mask.causal, (Mask{ .sliding = 8 }).simplify(4, 4, 0, 0));
    try std.testing.expectEqual(Mask.none, (Mask{ .sliding = 8 }).simplify(1, 4, 3, 0));
    try std.testing.expectEqual(Mask{ .sliding = 2 }, (Mask{ .sliding = 2 }).simplify(4, 4, 0, 0));
}
