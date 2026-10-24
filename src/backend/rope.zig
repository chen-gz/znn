//! 旋转位置编码 (Rotary Position Embedding, RoPE) 的配置与频率定义：
//! CPU 内核 (`tensor.RopeTable`)、自动微分反向传播与 Apple 机器学习拓展 (Machine Learning Explore, MLX) 图形处理器路径
//! 共用这一份旋转角数量与频率公式。

const std = @import("std");

/// 旋转位置编码可选参数配置 (Rotary Position Embedding Options, RopeOptions)。
/// 对长度为 head_dim 的行，旋转子空间为 [rotary_offset, head_dim)，记 rot_dim = head_dim - rotary_offset：
/// - 第 i 个旋转角 (0 <= i < num_angles) 的频率为 1 / base^(2i / rot_dim)，频率分母始终是完整的 rot_dim；
/// - num_angles = floor(rot_dim / 2 * partial_rotary_factor)，其余角度保持恒等 (cos = 1, sin = 0)；
/// - 配对拓扑见 `Mode`，半切分配对的两个分量始终相距 rot_dim / 2 (而不是 num_angles)。
pub const RopeOptions = struct {
    rotary_offset: usize = 0, // 旋转子空间起始偏移量，默认为 0 (对全部维度应用旋转位置编码 (Rotary Position Embedding, RoPE))
    base: f32 = 10000.0, // 频率基数，默认为 10000.0
    mode: Mode = .interleaved, // 旋转配对拓扑模式：相邻配对 (interleaved) 或半切分配对 (split_half)
    partial_rotary_factor: f32 = 1.0, // 部分旋转因子 (0.0 < factor <= 1.0)，仅对前该比例的角度进行旋转，其余保持恒等

    pub const Mode = enum {
        interleaved, // 传统 RoPE 相邻配对拓扑 (2i, 2i+1)
        split_half, // 现代 GPT-NeoX / LLaMA / Gemma 规范的半切分配对拓扑 (i, i + D/2)
    };

    pub const default: RopeOptions = .{};

    pub fn defaultOptions() RopeOptions {
        return .{};
    }

    /// 比例旋转位置编码 (Proportional RoPE) 预设，与 Hugging Face `proportional` RoPE 及 mlx-lm `ProportionalRoPE` 一致：
    /// 在完整 head_dim 上半切分配对，只旋转前 floor(head_dim / 2 * partial_rotary_factor) 个角度，频率分母仍为完整 head_dim。
    /// partial_rotary_factor = 1 时即标准半切分 RoPE (Gemma 4 的滑动窗口层与全局注意力层分别使用 1.0 与 0.25)。
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

    /// 第 i 个旋转角的周期 base^(2i / rot_dim)：旋转角为 pos / period (MLX `fast.rope` 的 freqs 语义)
    pub fn period(self: RopeOptions, rot_dim: usize, i: usize) f32 {
        return std.math.pow(f32, self.base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(rot_dim)));
    }

    /// 第 i 个旋转角的频率 1 / base^(2i / rot_dim)：旋转角为 pos * inverseFrequency (CPU 路径)
    pub fn inverseFrequency(self: RopeOptions, rot_dim: usize, i: usize) f32 {
        return 1.0 / self.period(rot_dim, i);
    }

    /// 填充 MLX `fast.rope` 使用的 [rot_dim / 2] 周期表：前 num_angles 个为 `period(i)`，
    /// 其余为 +inf (旋转角恒为 0，保持恒等)
    pub fn fillPeriods(self: RopeOptions, rot_dim: usize, out: []f32) void {
        std.debug.assert(out.len == rot_dim / 2);
        const num_angles = self.numAngles(rot_dim);
        for (out, 0..) |*f, i| {
            f.* = if (i < num_angles) self.period(rot_dim, i) else std.math.inf(f32);
        }
    }
};

/// 预计算频率表支持的最大旋转角数 (rot_dim / 2)
pub const max_angles: usize = 1024;

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
    // 因子截断到 [0, 1]
    try std.testing.expectEqual(@as(usize, 4), (RopeOptions{ .partial_rotary_factor = 2.0 }).numAngles(8));
    try std.testing.expectEqual(@as(usize, 0), (RopeOptions{ .partial_rotary_factor = -1.0 }).numAngles(8));
}
