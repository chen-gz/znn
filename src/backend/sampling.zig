//! 贪婪采样与重复惩罚通用基础设施 (Greedy Sampling & Repetition Penalty)
//!
//! 与具体模型无关：需要排除的词元集合 (例如填充词元与不可生成的控制词元) 由调用方的模型配置以闭区间列表传入，
//! CPU 贪婪采样 (`argmaxWithPenalty`) 与 GPU 加性掩码 (`fillExclusionMaskBf16`) 共用同一份区间定义。

const std = @import("std");

/// 词元 ID 闭区间 [first, last]
pub const TokenRange = struct {
    first: usize,
    last: usize,

    /// 只包含单个词元的区间
    pub fn single(id: usize) TokenRange {
        return .{ .first = id, .last = id };
    }

    pub fn contains(self: TokenRange, v: usize) bool {
        return v >= self.first and v <= self.last;
    }

    /// 是否与半开区间 [begin, end) 相交
    pub fn intersects(self: TokenRange, begin: usize, end: usize) bool {
        return begin <= self.last and end > self.first;
    }
};

/// 词元 v 是否落在任一排除区间内
pub fn isExcluded(excluded: []const TokenRange, v: usize) bool {
    for (excluded) |r| {
        if (r.contains(v)) return true;
    }
    return false;
}

/// 半开区间 [begin, end) 内是否存在排除词元
pub fn anyExcluded(excluded: []const TokenRange, begin: usize, end: usize) bool {
    for (excluded) |r| {
        if (r.intersects(begin, end)) return true;
    }
    return false;
}

/// 重复惩罚是否生效：系数大于 1 时生效，小于等于 1 视为不施加惩罚
pub fn penaltyActive(repetition_penalty: f32) bool {
    return repetition_penalty > 1.0;
}

/// 重复惩罚 (Repetition Penalty)，与 Hugging Face `RepetitionPenaltyLogitsProcessor` 一致：
/// 正对数几率除以惩罚系数，非正对数几率乘以惩罚系数。
/// 模型若带最终对数几率截断 (Final Logit Softcapping)，必须对截断之后的值调用本函数。
pub fn applyRepetitionPenalty(logit: f32, repetition_penalty: f32) f32 {
    return if (logit > 0.0) logit / repetition_penalty else logit * repetition_penalty;
}

/// 脑浮点 16 (Brain Floating Point 16, BF16) 负无穷与正零的位模式
pub const bf16_neg_inf_bits: u16 = 0xFF80;
pub const bf16_zero_bits: u16 = 0x0000;

/// 填充词表加性排除掩码的 BF16 位模式：排除词元为 -inf，其余为 0。
/// 掩码与对数几率相加后再求 argmax，即可在 GPU 上排除与 `argmaxWithPenalty` 完全相同的词元集合。
pub fn fillExclusionMaskBf16(out: []u16, excluded: []const TokenRange) void {
    @memset(out, bf16_zero_bits);
    for (excluded) |r| {
        if (r.first >= out.len) continue;
        const stop = @min(r.last, out.len - 1);
        @memset(out[r.first .. stop + 1], bf16_neg_inf_bits);
    }
}

/// 最近词元位图窗口的字数：每个窗口覆盖 4096 * 64 = 262,144 个词元，位图保存在栈上，不进行堆分配
const penalty_bitset_words: usize = 4096;
const penalty_window_tokens: usize = penalty_bitset_words * 64;

/// 在已计算好的最终对数几率切片上执行分块位图重复惩罚与贪婪采样 (Chunked Bitset Penalty & Fast-Path Argmax)。
/// 词表按 262,144 词元的窗口分段，每段在栈上重建最近词元位图；绝大多数 64 词块位图为 0 且不含排除词元，
/// 直接走单指令多数据流 (Single Instruction, Multiple Data, SIMD) 向量化块最大值快车道，仅在命中词块内逐元素处理。
/// 结果与逐元素严格大于比较的顺序扫描一致：返回首个取得最大惩罚后对数几率的非排除词元（NaN 被忽略）。
pub fn argmaxWithPenalty(
    logits_slice: []const f32,
    vocab_size: usize,
    excluded: []const TokenRange,
    recent_tokens: []const usize,
    repetition_penalty: f32,
) usize {
    std.debug.assert(logits_slice.len >= vocab_size);
    const apply_penalty = penaltyActive(repetition_penalty);
    const V = @Vector(16, f32);

    var bitset: [penalty_bitset_words]u64 = undefined;
    var max_val: f32 = -std.math.inf(f32);
    var max_id: usize = 0;

    var win_base: usize = 0;
    while (win_base < vocab_size) : (win_base += penalty_window_tokens) {
        const win_end = @min(win_base + penalty_window_tokens, vocab_size);
        const words = bitset[0 .. (win_end - win_base + 63) / 64];
        @memset(words, 0);
        if (apply_penalty) {
            for (recent_tokens) |rec| {
                if (rec >= win_base and rec < win_end) {
                    const off = rec - win_base;
                    words[off / 64] |= @as(u64, 1) << @intCast(off % 64);
                }
            }
        }

        for (words, 0..) |word, w| {
            const base_v = win_base + w * 64;
            const end_v = @min(base_v + 64, win_end);
            const check_excluded = anyExcluded(excluded, base_v, end_v);

            if (word == 0 and !check_excluded and end_v - base_v == 64) {
                // 快车道 (Fast Path)：整块 64 个词元既无重复词元也无排除词元，先向量化求块最大值，仅当刷新全局最大值时才定位下标
                const block = logits_slice[base_v..][0..64];
                var vmax: V = block[0..16].*;
                inline for (1..4) |q| vmax = @max(vmax, @as(V, block[q * 16 ..][0..16].*));
                const block_max = @reduce(.Max, vmax);
                if (block_max > max_val) {
                    max_val = block_max;
                    max_id = base_v + std.mem.indexOfScalar(f32, block, block_max).?;
                }
                continue;
            }

            // 精准道：逐元素过滤排除词元，并对位图命中的最近词元施加重复惩罚
            for (base_v..end_v) |v| {
                if (check_excluded and isExcluded(excluded, v)) continue;
                var val = logits_slice[v];
                if ((word >> @intCast(v - base_v)) & 1 != 0) {
                    val = applyRepetitionPenalty(val, repetition_penalty);
                }
                if (val > max_val) {
                    max_val = val;
                    max_id = v;
                }
            }
        }
    }

    return max_id;
}
