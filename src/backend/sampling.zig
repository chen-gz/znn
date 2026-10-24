//! 采样通用基础设施：贪婪采样、重复惩罚与随机采样 (Greedy Sampling, Repetition Penalty & Stochastic Sampling)
//!
//! 与具体模型无关：需要排除的词元集合 (例如填充词元与不可生成的控制词元) 由调用方的模型配置以闭区间列表传入，
//! CPU 贪婪采样 (`argmaxWithPenalty`)、CPU 随机采样 (`maskExcludedInPlace`) 与 GPU 加性掩码 (`fillExclusionMaskBf16`) 共用同一份区间定义。
//!
//! 对数几率处理顺序与 Hugging Face `generate` 一致：
//! 最终对数几率截断 (Final Logit Softcapping，模型侧) → 重复惩罚 (Repetition Penalty) → 排除词元 (Suppress Tokens) →
//! 温度 (Temperature) → 前 K 截断 (Top-K) → 核采样 (Nucleus Sampling, Top-P) → 按概率抽样。
//! 前两步 (以及排除) 由调用方在完整词表上完成 (CPU 就地处理或 GPU 计算图)，之后只把候选集交给同一个
//! `Distribution` 完成温度、Top-K、Top-P 与抽样：CPU 与 GPU 路径对相同候选集与相同种子选出相同词元。
//! 随机性只来自调用方持有的 `Sampler` (内含以 `SamplingOptions.seed` 初始化的伪随机数生成器)，本模块不持有任何全局随机状态。

const std = @import("std");
const Executor = @import("executor.zig").Executor;
const vec_math = @import("vec_math.zig");

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

// ============================================================================
// 随机采样 (Stochastic Sampling)：选项、候选集、终结分布与采样器
// ============================================================================

/// 设备端 (例如 GPU) 在未启用前 K 截断 (Top-K) 时选出并传回主机的有界候选数。
/// 只传回概率最高的 1024 个候选：核采样 (Nucleus Sampling, Top-P) 阈值落在其中时结果精确
/// (设备端同时传回完整词表的对数配分函数 (Log-Partition, logsumexp)，累积概率按完整词表归一化)，
/// 否则 (例如 top_p = 1 且分布非常平坦) 只在这 1024 个候选内按相对概率抽样，忽略其余长尾概率质量。
pub const bounded_candidate_count: usize = 1024;

/// CPU 在只启用核采样时首轮部分选择的候选数；累积概率未达到阈值时按 8 倍扩大重新选择
pub const initial_nucleus_candidates: usize = 256;

/// 采样选项 (Sampling Options)：全部字段带默认值，`.{}` 即按完整 Softmax 分布 (温度 1，无截断，无惩罚) 抽样
pub const SamplingOptions = struct {
    /// 温度 (Temperature)：对数几率除以温度后再归一化；小于等于 0 (或 NaN) 时退化为贪婪采样 (Greedy Decoding)
    temperature: f32 = 1.0,
    /// 前 K 截断 (Top-K)：只在概率最高的 K 个词元中抽样；0 表示不截断，1 等价于贪婪采样
    top_k: usize = 0,
    /// 核采样 (Nucleus Sampling, Top-P)：只保留累积概率刚好达到 top_p 的最小高概率词元集合；大于等于 1 表示不截断，
    /// 极小值 (例如 1e-6) 只保留概率最高的词元
    top_p: f32 = 1.0,
    /// 重复惩罚 (Repetition Penalty) 系数：大于 1 时对最近出现过的词元施加惩罚，小于等于 1 表示不惩罚
    repetition_penalty: f32 = 1.0,
    /// 重复惩罚窗口长度 (Repetition Penalty Window)：仅对最近窗口内的词元施加重复惩罚；null 时使用调用方/模型默认窗口
    repetition_window: ?usize = null,
    /// 伪随机数生成器 (Pseudo-Random Number Generator, PRNG) 种子：相同种子与相同对数几率得到相同的生成序列
    seed: u64 = 0,

    /// 贪婪采样 (不施加重复惩罚)
    pub const greedy: SamplingOptions = .{ .temperature = 0.0 };

    pub fn defaultOptions() SamplingOptions {
        return .{};
    }

    /// 是否退化为贪婪采样：温度小于等于 0 (或 NaN) 或 top_k == 1
    pub fn isGreedy(self: SamplingOptions) bool {
        return !(self.temperature > 0.0) or self.top_k == 1;
    }

    /// 是否为不带重复惩罚的纯贪婪采样 (可直接在设备端求 argmax，不需要回传任何对数几率)
    pub fn isPlainGreedy(self: SamplingOptions) bool {
        return self.isGreedy() and !penaltyActive(self.repetition_penalty);
    }

    /// 设备端需要选出并传回主机的候选数：贪婪采样为 1，启用 Top-K 时为 top_k，否则为 `bounded_candidate_count`；不超过词表大小
    pub fn candidateCount(self: SamplingOptions, vocab_size: usize) usize {
        const k: usize = if (self.isGreedy()) 1 else if (self.top_k > 0) self.top_k else bounded_candidate_count;
        return @max(1, @min(k, vocab_size));
    }

    /// 终结分布是否需要完整词表的对数配分函数：只启用核采样 (未启用 Top-K) 时，累积概率必须按完整词表归一化
    pub fn needsLogNormalizer(self: SamplingOptions) bool {
        return !self.isGreedy() and self.top_k == 0 and self.top_p < 1.0;
    }

    /// 获取给定词元序列在当前重复惩罚配置下的有效词元窗口切片
    pub fn recentTokens(self: SamplingOptions, tokens: []const usize, default_window: usize) []const usize {
        const win = self.repetition_window orelse default_window;
        return if (tokens.len > win) tokens[tokens.len - win ..] else tokens;
    }
};

/// 采样候选：词元 ID 与其已完成重复惩罚与排除处理 (尚未除以温度) 的对数几率
pub const Candidate = struct {
    id: u32,
    logit: f32,

    /// 规范全序 (Canonical Order)：对数几率降序，相同时词元 ID 升序。
    /// 候选集合按该全序排序后与输入顺序、分块方式及线程数无关
    pub fn before(a: Candidate, b: Candidate) bool {
        return a.logit > b.logit or (a.logit == b.logit and a.id < b.id);
    }

    fn lessThan(_: void, a: Candidate, b: Candidate) bool {
        return a.before(b);
    }
};

/// 对数几率是否可参与采样：NaN 与 -inf (排除词元) 的概率为 0
fn usableLogit(logit: f32) bool {
    return !std.math.isNan(logit) and logit != -std.math.inf(f32);
}

/// 有界候选堆 (Bounded Min-Heap)：保留按 `Candidate.before` 最优的 items.len 个候选，堆顶为保留集中最差者
const CandidateHeap = struct {
    items: []Candidate,
    len: usize = 0,

    fn push(self: *CandidateHeap, cand: Candidate) void {
        if (self.items.len == 0) return;
        if (self.len < self.items.len) {
            self.items[self.len] = cand;
            self.siftUp(self.len);
            self.len += 1;
        } else if (cand.before(self.items[0])) {
            self.items[0] = cand;
            self.siftDown(0);
        }
    }

    fn siftUp(self: *CandidateHeap, start: usize) void {
        var i = start;
        while (i > 0) {
            const parent = (i - 1) / 2;
            // 父节点必须不优于子节点 (堆顶最差)
            if (!self.items[parent].before(self.items[i])) break;
            std.mem.swap(Candidate, &self.items[parent], &self.items[i]);
            i = parent;
        }
    }

    fn siftDown(self: *CandidateHeap, start: usize) void {
        var i = start;
        while (true) {
            var worst = i;
            inline for (.{ 1, 2 }) |offset| {
                const child = 2 * i + offset;
                if (child < self.len and self.items[worst].before(self.items[child])) worst = child;
            }
            if (worst == i) break;
            std.mem.swap(Candidate, &self.items[worst], &self.items[i]);
            i = worst;
        }
    }
};

/// 部分选择的最小分块长度：分块边界只取决于词表大小与 k，与线程数无关
const topk_min_chunk: usize = 8192;

/// 前 K 部分选择 (Partial Top-K Selection)：从完整对数几率中选出按规范全序最优的 out.len 个可采样候选，
/// 返回 out 的前缀 (可采样词元不足 out.len 个时更短)，顺序未排序。
/// 词表按固定长度分块，各块经 `Executor` 并行维护大小为 k 的有界最小堆，再由调用线程合并各块的局部结果；
/// 代价 O(V log k)，从不对完整词表排序。规范全序是严格全序，因此结果与分块方式及线程数无关。
pub fn selectTopK(
    executor: Executor,
    allocator: std.mem.Allocator,
    logits: []const f32,
    out: []Candidate,
) ![]Candidate {
    const n = logits.len;
    const k = @min(out.len, n);
    if (k == 0) return out[0..0];
    std.debug.assert(n <= std.math.maxInt(u32));

    const grain = @max(topk_min_chunk, k * 4);
    const chunks = n / grain + @intFromBool(n % grain != 0);
    if (chunks <= 1) {
        var heap = CandidateHeap{ .items = out[0..k] };
        for (logits, 0..) |logit, v| {
            if (usableLogit(logit)) heap.push(.{ .id = @intCast(v), .logit = logit });
        }
        return out[0..heap.len];
    }

    const partials = try allocator.alloc(Candidate, chunks * k);
    defer allocator.free(partials);
    const counts = try allocator.alloc(usize, chunks);
    defer allocator.free(counts);

    const Context = struct {
        logits: []const f32,
        partials: []Candidate,
        counts: []usize,
        k: usize,
        grain: usize,

        fn run(ctx: @This(), begin: usize, end: usize) void {
            for (begin..end) |c| {
                const start = c * ctx.grain;
                const stop = @min(start + ctx.grain, ctx.logits.len);
                var heap = CandidateHeap{ .items = ctx.partials[c * ctx.k ..][0..ctx.k] };
                for (ctx.logits[start..stop], start..) |logit, v| {
                    if (usableLogit(logit)) heap.push(.{ .id = @intCast(v), .logit = logit });
                }
                ctx.counts[c] = heap.len;
            }
        }
    };
    try executor.parallelFor(chunks, 1, Context{
        .logits = logits,
        .partials = partials,
        .counts = counts,
        .k = k,
        .grain = grain,
    }, Context.run);

    var heap = CandidateHeap{ .items = out[0..k] };
    for (counts, 0..) |count, c| {
        for (partials[c * k ..][0..count]) |cand| heap.push(cand);
    }
    return out[0..heap.len];
}

/// 按固定分块有序并行归约整个词表的对数求和指数 (Log-Sum-Exp, LSE) 部分结果 (跳过 NaN)
fn logSumExpParallel(executor: Executor, logits: []const f32, inv_temperature: f32) !vec_math.LogSumExp {
    const Context = struct {
        logits: []const f32,
        inv_temperature: f32,

        fn map(ctx: @This(), begin: usize, end: usize) vec_math.LogSumExp {
            // NaN 的概率视为 0；排除词元 (-inf) 对和式贡献 0
            return vec_math.logSumExpPartial(ctx.logits[begin..end], ctx.inv_temperature, .skip);
        }
    };
    return executor.parallelReduce(
        vec_math.LogSumExp,
        logits.len,
        16384,
        Context{ .logits = logits, .inv_temperature = inv_temperature },
        Context.map,
        vec_math.LogSumExp.merge,
        .{},
    );
}

/// 完整词表的对数配分函数 (Log-Partition)：log Σ_v exp(logit_v * inv_temperature)，跳过排除词元 (-inf) 与 NaN；
/// 全部被排除时返回 -inf。按固定分块有序归约，结果与线程数无关。
pub fn logSumExpScaled(executor: Executor, logits: []const f32, inv_temperature: f32) !f32 {
    return (try logSumExpParallel(executor, logits, inv_temperature)).value();
}

/// 就地施加重复惩罚：recent_tokens 中出现的每个词元 (越界 ID 被忽略) 恰好惩罚一次，重复出现不会叠加
pub fn applyRepetitionPenaltyInPlace(
    allocator: std.mem.Allocator,
    logits: []f32,
    recent_tokens: []const usize,
    repetition_penalty: f32,
) !void {
    if (!penaltyActive(repetition_penalty) or recent_tokens.len == 0) return;
    var seen = try std.DynamicBitSetUnmanaged.initEmpty(allocator, logits.len);
    defer seen.deinit(allocator);
    for (recent_tokens) |tok| {
        if (tok >= logits.len or seen.isSet(tok)) continue;
        seen.set(tok);
        logits[tok] = applyRepetitionPenalty(logits[tok], repetition_penalty);
    }
}

/// 就地将排除词元的对数几率置为 -inf (概率为 0)
pub fn maskExcludedInPlace(logits: []f32, excluded: []const TokenRange) void {
    for (excluded) |r| {
        if (r.first >= logits.len) continue;
        const stop = @min(r.last, logits.len - 1);
        @memset(logits[r.first .. stop + 1], -std.math.inf(f32));
    }
}

/// 终结分布 (Finished Distribution)：温度、前 K 截断与核采样之后保留的候选及其未归一化权重。
/// 权重 w_i = exp((logit_i - logit_0) / temperature)，概率 p_i = w_i / mass；CPU 与 GPU 路径共用。
pub const Distribution = struct {
    /// 保留的候选 (按规范全序排列)；为空表示没有任何可采样词元
    candidates: []const Candidate,
    inv_temperature: f32,
    /// 最优候选的缩放对数几率 (权重的数值稳定基准)
    max_scaled: f32,
    /// 保留候选的权重和
    mass: f64,
    /// 核采样阈值是否在给定候选集内达到；为 false 表示候选集只是完整分布的前缀，未覆盖 top_p 要求的概率质量
    covers_nucleus: bool,

    /// 候选的未归一化权重
    pub fn weight(self: Distribution, cand: Candidate) f64 {
        return @exp(@as(f64, cand.logit * self.inv_temperature - self.max_scaled));
    }

    /// 词元 id 在终结分布中的概率 (被截断或不在候选集中时为 0)
    pub fn probability(self: Distribution, id: u32) f64 {
        if (self.mass <= 0.0) return 0.0;
        for (self.candidates) |cand| {
            if (cand.id == id) return self.weight(cand) / self.mass;
        }
        return 0.0;
    }

    /// 按终结分布抽样一个词元：只有一个候选时不消耗随机数；没有可采样词元时返回 0 (与 `argmaxWithPenalty` 一致)
    pub fn sample(self: Distribution, random: std.Random) u32 {
        return self.sampleExcluding(random, null);
    }

    /// 从去掉词元 excluded 并重新归一化后的终结分布中抽样 (投机采样拒绝后的重采样)。
    /// 只剩一个候选时直接返回且不消耗随机数；去掉之后没有剩余候选时返回 excluded 本身
    pub fn sampleExcluding(self: Distribution, random: std.Random, excluded: ?u32) u32 {
        if (self.candidates.len == 0) return 0;
        var total = self.mass;
        var remaining: usize = 0;
        var first_other: ?u32 = null;
        var last_other: u32 = self.candidates[0].id;
        for (self.candidates) |cand| {
            if (excluded) |x| {
                if (cand.id == x) {
                    total -= self.weight(cand);
                    continue;
                }
            }
            remaining += 1;
            if (first_other == null) first_other = cand.id;
            last_other = cand.id;
        }
        const fallback = first_other orelse return self.candidates[0].id;
        if (remaining == 1 or !(total > 0.0)) return fallback;

        const target = random.float(f64) * total;
        var cum: f64 = 0.0;
        for (self.candidates) |cand| {
            if (excluded) |x| {
                if (cand.id == x) continue;
            }
            cum += self.weight(cand);
            if (cum > target) return cand.id;
        }
        return last_other;
    }
};

/// 终结步骤 (CPU / GPU 共用)：把候选集 (对数几率已完成重复惩罚与排除) 依次施加温度、前 K 截断与核采样，得到终结分布。
/// - candidates 按规范全序就地排序并剔除不可采样项，返回的分布借用其中一段前缀；
/// - 贪婪采样只保留最优候选；
/// - log_normalizer 为完整词表的 log Σ exp(logit / temperature) (见 `logSumExpScaled`)，只在未启用 Top-K 时用于核采样阈值：
///   与 Hugging Face 一致，启用 Top-K 时核采样在 Top-K 截断后的分布上重新归一化计算；为 null 时以候选集自身为完整分布。
/// 核采样保留规则与 Hugging Face `TopPLogitsWarper` 一致：按概率降序保留排在其前面的累积概率严格小于 top_p 的词元 (至少保留 1 个)。
pub fn buildDistribution(candidates: []Candidate, options: SamplingOptions, log_normalizer: ?f32) Distribution {
    var n: usize = 0;
    for (candidates) |cand| {
        if (!usableLogit(cand.logit)) continue;
        candidates[n] = cand;
        n += 1;
    }
    const sorted = candidates[0..n];
    std.mem.sortUnstable(Candidate, sorted, {}, Candidate.lessThan);

    if (n == 0) return .{ .candidates = sorted, .inv_temperature = 1.0, .max_scaled = 0.0, .mass = 0.0, .covers_nucleus = true };
    if (options.isGreedy()) {
        return .{ .candidates = sorted[0..1], .inv_temperature = 1.0, .max_scaled = sorted[0].logit, .mass = 1.0, .covers_nucleus = true };
    }

    const kept_k = if (options.top_k > 0) sorted[0..@min(options.top_k, n)] else sorted;
    const inv_t = 1.0 / options.temperature;
    var dist = Distribution{
        .candidates = kept_k,
        .inv_temperature = inv_t,
        .max_scaled = kept_k[0].logit * inv_t,
        .mass = 0.0,
        .covers_nucleus = true,
    };

    var candidate_mass: f64 = 0.0;
    for (kept_k) |cand| candidate_mass += dist.weight(cand);
    const use_normalizer = options.top_k == 0 and log_normalizer != null;
    const total: f64 = if (use_normalizer)
        @max(candidate_mass, @exp(@as(f64, log_normalizer.? - dist.max_scaled)))
    else
        candidate_mass;

    if (options.top_p < 1.0) {
        const threshold = @as(f64, options.top_p) * total;
        var cum: f64 = 0.0;
        var kept: usize = 0;
        while (kept < kept_k.len) : (kept += 1) {
            if (kept > 0 and cum >= threshold) break;
            cum += dist.weight(kept_k[kept]);
        }
        dist.candidates = kept_k[0..kept];
        dist.mass = cum;
        dist.covers_nucleus = cum >= threshold;
    } else {
        dist.mass = candidate_mass;
        dist.covers_nucleus = !use_normalizer;
    }
    return dist;
}

/// 投机采样 (Speculative Sampling) 对单个草稿词元的校验结果
pub const DraftVerdict = union(enum) {
    /// 接受草稿词元
    accepted,
    /// 拒绝草稿词元，并给出从剩余分布中重新抽样得到的替代词元
    rejected: u32,
};

/// 确定性草稿 (草稿分布 q 在 draft 上的概率为 1) 的标准投机采样校验：以概率 min(1, p(draft) / q(draft)) = p(draft) 接受；
/// 拒绝时从 norm(max(0, p - q)) 抽样，即去掉 draft 后重新归一化的目标分布 p。输出词元的分布恰好等于 p (无偏)。
/// p(draft) 为 0 或 1 时不消耗随机数，因此贪婪终结分布 (只含一个候选) 退化为精确匹配规则。
pub fn verifyDraftToken(dist: Distribution, draft: u32, random: std.Random) DraftVerdict {
    const p = dist.probability(draft);
    if (p >= 1.0) return .accepted;
    if (p > 0.0 and random.float(f64) < p) return .accepted;
    return .{ .rejected = dist.sampleExcluding(random, draft) };
}

/// 采样器 (Sampler)：持有采样选项与以 `options.seed` 初始化的伪随机数生成器，由调用方创建并在整个生成过程中复用；
/// 每次随机抽样消耗确定数量的随机数，因此相同种子、相同模型与相同输入得到相同的生成序列。
pub const Sampler = struct {
    options: SamplingOptions,
    prng: std.Random.DefaultPrng,

    pub fn init(options: SamplingOptions) Sampler {
        return .{ .options = options, .prng = .init(options.seed) };
    }

    /// 以默认选项 (`SamplingOptions.defaultOptions()`) 创建采样器
    pub fn initDefault() Sampler {
        return init(.{});
    }

    pub fn isGreedy(self: *const Sampler) bool {
        return self.options.isGreedy();
    }

    pub fn random(self: *Sampler) std.Random {
        return self.prng.random();
    }

    /// 由候选集构建终结分布 (见 `buildDistribution`)
    pub fn distribution(self: *const Sampler, candidates: []Candidate, log_normalizer: ?f32) Distribution {
        return buildDistribution(candidates, self.options, log_normalizer);
    }

    /// 终结步骤：从候选集抽样一个词元 (CPU 与 GPU 路径共用，相同候选集与相同随机状态得到相同词元)
    pub fn sampleCandidates(self: *Sampler, candidates: []Candidate, log_normalizer: ?f32) u32 {
        return self.distribution(candidates, log_normalizer).sample(self.random());
    }

    /// 投机采样校验草稿词元 (见 `verifyDraftToken`)
    pub fn verifyDraft(self: *Sampler, dist: Distribution, draft: u32) DraftVerdict {
        return verifyDraftToken(dist, draft, self.random());
    }

    /// 从完整词表的已处理对数几率 (重复惩罚已施加，排除词元为 -inf，尚未除以温度) 中抽样一个词元。
    /// - 贪婪采样：返回首个最大值 (NaN 被忽略)；
    /// - 启用 Top-K：`selectTopK` 部分选择 k 个候选，代价 O(V log k)；
    /// - 只启用核采样：先求完整词表的对数配分函数，再从 `initial_nucleus_candidates` 个候选开始部分选择，
    ///   累积概率未达到 top_p 时按 8 倍扩大候选数重选；典型的尖峰分布一轮完成 (O(V log 256))，
    ///   极端平坦分布最坏扩大到整个词表 (O(V log V)，与完整排序同阶)；
    /// - 均未启用：按完整 Softmax 分布做一次 O(V) 逆累积分布扫描，不排序。
    pub fn sampleLogits(self: *Sampler, executor: Executor, allocator: std.mem.Allocator, logits: []const f32) !usize {
        const n = logits.len;
        if (n == 0) return error.EmptyLogits;
        const opts = self.options;
        if (opts.isGreedy()) return argmaxWithPenalty(logits, n, &.{}, &.{}, 1.0);

        if (opts.top_k > 0) {
            const buf = try allocator.alloc(Candidate, @min(opts.top_k, n));
            defer allocator.free(buf);
            const cands = try selectTopK(executor, allocator, logits, buf);
            return self.sampleCandidates(cands, null);
        }

        const inv_t = 1.0 / opts.temperature;
        if (!(opts.top_p < 1.0)) return self.sampleFull(executor, logits, inv_t);

        const log_normalizer = try logSumExpScaled(executor, logits, inv_t);
        var m: usize = @min(initial_nucleus_candidates, n);
        while (true) {
            const buf = try allocator.alloc(Candidate, m);
            defer allocator.free(buf);
            const cands = try selectTopK(executor, allocator, logits, buf);
            const dist = self.distribution(cands, log_normalizer);
            if (dist.covers_nucleus or m >= n or cands.len < m) return dist.sample(self.random());
            m = @min(n, m * 8);
        }
    }

    /// 完整 Softmax 分布的 O(V) 逆累积分布抽样 (Inverse CDF Sampling)：按词元 ID 顺序累加权重
    fn sampleFull(self: *Sampler, executor: Executor, logits: []const f32, inv_temperature: f32) !usize {
        const p = try logSumExpParallel(executor, logits, inv_temperature);
        if (p.max == -std.math.inf(f32)) return 0;
        const target = self.random().float(f64) * p.sum;
        var cum: f64 = 0.0;
        var last: usize = 0;
        for (logits, 0..) |logit, v| {
            if (!usableLogit(logit)) continue;
            cum += @exp(@as(f64, logit * inv_temperature - p.max));
            last = v;
            if (cum > target) return v;
        }
        return last;
    }
};
