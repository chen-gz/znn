const std = @import("std");
const backend = @import("../backend.zig");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const nn = @import("../nn.zig");
const core = nn.core;
const transformer = nn.transformer;

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Linear = core.Linear;
const Embedding = transformer.Embedding;
const RMSNorm = nn.RMSNorm;
const createPersistentTensor = core.createPersistentTensor;
const createLayerScalar = core.createLayerScalar;
const freePersistentTensor = core.freePersistentTensor;
const deinitModel = core.deinitModel;
const releaseModuleGrads = core.releaseModuleGrads;
const enterModuleScope = core.enterModuleScope;
const vec_math = backend.vec_math;
pub const TokenRange = tensor.TokenRange;
pub const SamplingOptions = tensor.SamplingOptions;

const topk_min_chunk: usize = 8192;
const initial_nucleus_candidates: usize = 256;

const CandidateHeap = struct {
    items: []tensor.Candidate,
    len: usize = 0,

    fn push(self: *CandidateHeap, cand: tensor.Candidate) void {
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
            if (!self.items[parent].before(self.items[i])) break;
            std.mem.swap(tensor.Candidate, &self.items[parent], &self.items[i]);
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
            std.mem.swap(tensor.Candidate, &self.items[worst], &self.items[i]);
            i = worst;
        }
    }
};

pub fn selectTopK(
    executor: anytype,
    allocator: std.mem.Allocator,
    logits: []const f32,
    out: []tensor.Candidate,
) ![]tensor.Candidate {
    const n = logits.len;
    const k = @min(out.len, n);
    if (k == 0) return out[0..0];
    std.debug.assert(n <= std.math.maxInt(u32));

    const grain = @max(topk_min_chunk, k * 4);
    const chunks = n / grain + @intFromBool(n % grain != 0);
    if (chunks <= 1) {
        var heap = CandidateHeap{ .items = out[0..k] };
        for (logits, 0..) |logit, v| {
            if (tensor.usableLogit(logit)) heap.push(.{ .id = @intCast(v), .logit = logit });
        }
        return out[0..heap.len];
    }

    const partials = try allocator.alloc(tensor.Candidate, chunks * k);
    defer allocator.free(partials);
    const counts = try allocator.alloc(usize, chunks);
    defer allocator.free(counts);

    const Context = struct {
        logits: []const f32,
        partials: []tensor.Candidate,
        counts: []usize,
        k: usize,
        grain: usize,

        fn run(ctx: @This(), begin: usize, end: usize) void {
            for (begin..end) |c| {
                const start = c * ctx.grain;
                const stop = @min(start + ctx.grain, ctx.logits.len);
                var heap = CandidateHeap{ .items = ctx.partials[c * ctx.k ..][0..ctx.k] };
                for (ctx.logits[start..stop], start..) |logit, v| {
                    if (tensor.usableLogit(logit)) heap.push(.{ .id = @intCast(v), .logit = logit });
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

fn logSumExpParallel(executor: anytype, logits: []const f32, inv_temperature: f32) !vec_math.LogSumExp {
    const Context = struct {
        logits: []const f32,
        inv_temperature: f32,

        fn map(ctx: @This(), begin: usize, end: usize) vec_math.LogSumExp {
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

pub fn logSumExpScaled(executor: anytype, logits: []const f32, inv_temperature: f32) !f32 {
    return (try logSumExpParallel(executor, logits, inv_temperature)).value();
}

/// Gemma 4 模型自带采样器 (Model-Native Sampler)
pub const Sampler = struct {
    options: tensor.SamplingOptions,
    prng: std.Random.DefaultPrng,

    pub fn init(options: tensor.SamplingOptions) Sampler {
        return .{ .options = options, .prng = .init(options.seed) };
    }

    pub fn initDefault() Sampler {
        return init(.{});
    }

    pub fn isGreedy(self: *const Sampler) bool {
        return self.options.isGreedy();
    }

    pub fn random(self: *Sampler) std.Random {
        return self.prng.random();
    }

    pub fn distribution(self: *const Sampler, candidates: []tensor.Candidate, log_normalizer: ?f32) tensor.Distribution {
        return tensor.buildDistribution(candidates, self.options, log_normalizer);
    }

    pub fn sampleCandidates(self: *Sampler, candidates: []tensor.Candidate, log_normalizer: ?f32) u32 {
        return self.distribution(candidates, log_normalizer).sample(self.random());
    }

    pub fn verifyDraft(self: *Sampler, dist: tensor.Distribution, draft: u32) tensor.DraftVerdict {
        return tensor.verifyDraftToken(dist, draft, self.random());
    }

    pub fn sampleLogits(self: *Sampler, executor: anytype, allocator: std.mem.Allocator, logits: []const f32) !usize {
        const n = logits.len;
        if (n == 0) return error.EmptyLogits;
        const opts = self.options;
        if (opts.isGreedy()) return tensor.argmaxWithPenalty(logits, n, &.{}, &.{}, 1.0);

        if (opts.top_k > 0) {
            const buf = try allocator.alloc(tensor.Candidate, @min(opts.top_k, n));
            defer allocator.free(buf);
            const cands = try selectTopK(executor, allocator, logits, buf);
            return self.sampleCandidates(cands, null);
        }

        const inv_t: f32 = 1.0 / opts.temperature;
        if (!(opts.top_p < 1.0)) return self.sampleFull(executor, logits, inv_t);

        const log_normalizer = try logSumExpScaled(executor, logits, inv_t);
        var m: usize = @min(initial_nucleus_candidates, n);
        while (true) {
            const buf = try allocator.alloc(tensor.Candidate, m);
            defer allocator.free(buf);
            const cands = try selectTopK(executor, allocator, logits, buf);
            const dist = self.distribution(cands, log_normalizer);
            if (dist.covers_nucleus or m >= n or cands.len < m) return dist.sample(self.random());
            m = @min(n, m * 8);
        }
    }

    fn sampleFull(self: *Sampler, executor: anytype, logits: []const f32, inv_temperature: f32) !usize {
        const p = try logSumExpParallel(executor, logits, inv_temperature);
        if (p.max == -std.math.inf(f32)) return 0;
        const target = self.random().float(f64) * p.sum;
        var cum: f64 = 0.0;
        var last: usize = 0;
        for (logits, 0..) |logit, v| {
            if (!tensor.usableLogit(logit)) continue;
            cum += @exp(@as(f64, logit * inv_temperature - p.max));
            last = v;
            if (cum > target) return v;
        }
        return last;
    }
};

// ============================================================================
// Gemma 4 混合注意力共享核心 (F32 与 Q4 两条推理路径共用)
// ============================================================================

/// 将预填充阶段旋转后的键 (Key, K) 与值 (Value, V) 头张量 ([1, n_kv, T, hd]，行主序连续) 写入层键值缓存
/// (Key-Value Cache, KVCache)，并将缓存游标设为 T。T 超出缓存容量时返回 `error.ContextWindowExceeded`，不修改缓存。
pub fn writePrefillKVCache(
    cache: *Gemma4KVCache.LayerCache,
    k_rot: []const f32,
    v: []const f32,
    batch: usize,
    seq_len: usize,
) !void {
    if (batch != 1) return error.InvalidBatchSize;
    try cache.core.writePrefill(k_rot, v, seq_len);
}

/// 单查询增量解码注意力：将当前时间步旋转后的键 (k_rot: [n_kv, hd]) 与值 (v: [n_kv, hd]) 追加写入层键值缓存，
/// 随后每个查询头对缓存中的可见键 (`Gemma4AttentionType.mask` 给出的区间，滑动窗口层仅最近 sliding_window 个位置)
/// 执行共享的单查询注意力内核 (`attendOneQuery`，scale = 1.0)。
/// 返回输出投影前的注意力特征 [1, nh * hd]。缓存已满时返回 `error.ContextWindowExceeded`，不修改缓存。
pub fn decodeAttention(
    graph: *autodiff.Graph,
    cache: *Gemma4KVCache.LayerCache,
    attn_type: Gemma4AttentionType,
    sliding_window: usize,
    num_heads: usize,
    q_rot: []const f32,
    k_rot: []const f32,
    v: []const f32,
) !*Tensor {
    _ = attn_type;
    _ = sliding_window;
    const pos = cache.core.curr_len;
    try cache.core.append(k_rot, v);

    const n_kv = cache.core.num_kv_heads;
    const hd = cache.core.head_dim;
    const max_len = cache.core.max_seq_len;
    const groups = num_heads / n_kv;

    const y_2d = try graph.zeros(&.{ 1, num_heads * hd }, false);
    const range = cache.core.visibleRange(pos);
    const scores = try graph.arenaAllocator().alloc(f32, range.len());

    for (0..num_heads) |h| {
        const kv_base = (h / groups) * max_len * hd;
        tensor.attention_kernels.attendOneQuery(
            q_rot[h * hd .. (h + 1) * hd],
            cache.core.k[kv_base..][0 .. cache.core.curr_len * hd],
            cache.core.v[kv_base..][0 .. cache.core.curr_len * hd],
            range,
            1.0,
            scores,
            y_2d.data[h * hd .. (h + 1) * hd],
        );
    }
    return y_2d;
}

/// 词元嵌入查表并施加 Gemma 嵌入缩放 sqrt(hidden_size)，返回 [1, T, hidden_size] (F32 与 Q4 CPU 路径共用)。
/// table 为 [vocab_size, hidden_size] 行主序嵌入表 (单精度 f32 或脑浮点 16 (Brain Floating Point 16, BF16) 原始位 u16)，越界词元 ID 返回 `error.IndexOutOfBounds`。
pub fn embedTokensScaled(comptime Elem: type, graph: *autodiff.Graph, table: []const Elem, hidden_size: usize, tokens: []const usize) !*Tensor {
    const tok_emb = try graph.tensorND(&.{ 1, tokens.len, hidden_size }, false);
    const sqrt_d: f32 = @sqrt(@as(f32, @floatFromInt(hidden_size)));
    try tensor.embeddingLookupInto(Elem, table, tokens, hidden_size, sqrt_d, tok_emb.data);
    return tok_emb;
}

// ============================================================================
// 采样共享核心：最终对数几率截断、重复惩罚与随机采样 (F32 / Q4 CPU 路径与 GPU 路径共用同一顺序)
// ============================================================================

/// 采样使用的最终对数几率：先施加最终对数几率截断，再对最近出现过的词元施加重复惩罚，
/// 与 Hugging Face (logits 处理器作用于截断后的输出) 及 E4B GPU 路径的顺序一致
pub fn samplingLogit(raw: f32, cap: f32, penalized: bool, repetition_penalty: f32) f32 {
    const capped = vec_math.softcap(raw, cap);
    if (!penalized) return capped;
    return if (capped > 0.0) capped / repetition_penalty else capped * repetition_penalty;
}

/// 参与重复惩罚的最近词元窗口长度
pub const repetition_window: usize = 64;

/// 取最近 repetition_window 个词元作为重复惩罚窗口
pub fn recentWindow(tokens: []const usize) []const usize {
    return if (tokens.len > repetition_window) tokens[tokens.len - repetition_window ..] else tokens;
}

/// 按采样选项的 `repetition_window` (若未指定则回退到默认 `repetition_window`) 截取最近词元窗口
pub fn recentWindowFor(tokens: []const usize, options: SamplingOptions) []const usize {
    return options.recentTokens(tokens, repetition_window);
}

/// 词表投影、最终对数几率截断、重复惩罚与贪婪采样融合的 CPU 多核并行实现 (Fused LM Head Projection & Greedy Sampling)，
/// 是贪婪采样的快车道：不物化完整对数几率。
/// embed 为 [vocab_size, hidden_size] 行主序词表嵌入矩阵，元素类型为单精度浮点 (f32) 或脑浮点 16 (Brain Floating Point 16, BF16) 原始位 (u16)。
/// 每个词元的最终得分为 `samplingLogit(点积)` (截断后惩罚，recent_tokens 中的词元全部受罚，窗口由调用方决定)，
/// 跳过 `cfg.sampling_excluded_tokens` 中的词元，返回取得最大最终得分的词元 ID (并列时取较小 ID，NaN 被忽略)。
/// 截断函数单调递增，未受罚词元直接按原始点积比较，每个分块只对块内最优者与受罚词元计算一次截断，避免逐词元 tanh 开销。
pub fn greedyLmHeadCpu(
    comptime cfg: Gemma4Config,
    comptime Elem: type,
    graph: *autodiff.Graph,
    embed: []const Elem,
    h_last: []const f32,
    recent_tokens: []const usize,
    repetition_penalty: f32,
) !usize {
    comptime std.debug.assert(Elem == f32 or Elem == u16);
    if (h_last.len != cfg.hidden_size or embed.len < cfg.vocab_size * cfg.hidden_size) return error.ShapeMismatch;

    const BestToken = struct {
        id: usize,
        val: f32,

        fn better(a: @This(), b: @This()) @This() {
            return if (b.val > a.val or (b.val == a.val and b.id < a.id)) b else a;
        }
    };

    const Context = struct {
        embed: []const Elem,
        h_last: []const f32,
        recent_bitset: []const u64,
        apply_penalty: bool,
        repetition_penalty: f32,

        fn mapFn(ctx: @This(), start_v: usize, end_v: usize) BestToken {
            // 未受罚词元按原始点积取块内最大值 (val 暂存原始值)，受罚词元直接比较截断并惩罚后的最终得分
            var best_plain = BestToken{ .id = std.math.maxInt(usize), .val = -std.math.inf(f32) };
            var best_penalized = BestToken{ .id = std.math.maxInt(usize), .val = -std.math.inf(f32) };
            const hidden = cfg.hidden_size;
            for (start_v..end_v) |v| {
                if (cfg.isSamplingExcluded(v)) continue;
                const row = ctx.embed[v * hidden ..][0..hidden];
                const raw = if (Elem == f32) vec_math.dot(ctx.h_last, row) else vec_math.dotBf16(ctx.h_last, row);
                const penalized = ctx.apply_penalty and (ctx.recent_bitset[v / 64] >> @intCast(v % 64)) & 1 != 0;
                if (penalized) {
                    const val = samplingLogit(raw, cfg.final_logit_softcapping, true, ctx.repetition_penalty);
                    if (val > best_penalized.val) best_penalized = .{ .id = v, .val = val };
                } else if (raw > best_plain.val) {
                    best_plain = .{ .id = v, .val = raw };
                }
            }
            if (best_plain.id != std.math.maxInt(usize)) best_plain.val = vec_math.softcap(best_plain.val, cfg.final_logit_softcapping);
            return best_plain.better(best_penalized);
        }

        fn reduceFn(a: BestToken, b: BestToken) BestToken {
            return a.better(b);
        }
    };

    const bitset_words = (cfg.vocab_size + 63) / 64;
    var recent_bitset: [bitset_words]u64 = @splat(0);
    const apply_penalty = (repetition_penalty != 1.0 and !std.math.isNan(repetition_penalty) and repetition_penalty > 0);
    if (apply_penalty) {
        for (recent_tokens) |rec| {
            if (rec < cfg.vocab_size) recent_bitset[rec / 64] |= @as(u64, 1) << @intCast(rec % 64);
        }
    }

    const ctx = Context{
        .embed = embed,
        .h_last = h_last,
        .recent_bitset = &recent_bitset,
        .apply_penalty = apply_penalty,
        .repetition_penalty = repetition_penalty,
    };
    const best = try graph.executor().parallelReduce(
        BestToken,
        cfg.vocab_size,
        4096,
        ctx,
        Context.mapFn,
        Context.reduceFn,
        BestToken{ .id = std.math.maxInt(usize), .val = -std.math.inf(f32) },
    );
    // 全部词元均被排除或得分为 NaN 时退回 0 号词元
    return if (best.id == std.math.maxInt(usize)) 0 else best.id;
}

/// 词表投影与最终对数几率截断的 CPU 多核并行实现：把截断后的完整对数几率写入 out [vocab_size]。
/// 每个词元的结果只取决于自身的点积，按固定粒度分块经 `graph.executor()` 并行，结果与线程数无关。
pub fn lmHeadLogitsCpu(
    comptime cfg: Gemma4Config,
    comptime Elem: type,
    graph: *autodiff.Graph,
    embed: []const Elem,
    h_last: []const f32,
    out: []f32,
) !void {
    comptime std.debug.assert(Elem == f32 or Elem == u16);
    if (h_last.len != cfg.hidden_size or embed.len < cfg.vocab_size * cfg.hidden_size or out.len != cfg.vocab_size) return error.ShapeMismatch;

    const Context = struct {
        embed: []const Elem,
        h_last: []const f32,
        out: []f32,

        fn run(ctx: @This(), start_v: usize, end_v: usize) void {
            const hidden = cfg.hidden_size;
            for (start_v..end_v) |v| {
                const row = ctx.embed[v * hidden ..][0..hidden];
                const raw = if (Elem == f32) vec_math.dot(ctx.h_last, row) else vec_math.dotBf16(ctx.h_last, row);
                ctx.out[v] = vec_math.softcap(raw, cfg.final_logit_softcapping);
            }
        }
    };
    try graph.executor().parallelFor(cfg.vocab_size, 4096, Context{ .embed = embed, .h_last = h_last, .out = out }, Context.run);
}

/// CPU 词表投影并选取下一个词元：
/// - 贪婪采样走融合快车道 `greedyLmHeadCpu`，结果与此前的贪婪实现逐位一致；
/// - 随机采样先物化截断后的完整对数几率，再交由 sampler 完成重复惩罚、排除与抽样。
pub fn sampleLmHeadCpu(
    comptime cfg: Gemma4Config,
    comptime Elem: type,
    graph: *autodiff.Graph,
    embed: []const Elem,
    h_last: []const f32,
    recent_tokens: []const usize,
    sampler: anytype,
) !usize {
    const penalty = sampler.options.repetition_penalty;
    if (sampler.isGreedy()) return greedyLmHeadCpu(cfg, Elem, graph, embed, h_last, recent_tokens, penalty);

    const arena = graph.arenaAllocator();
    const logits = try arena.alloc(f32, cfg.vocab_size);
    try lmHeadLogitsCpu(cfg, Elem, graph, embed, h_last, logits);
    try tensor.applyRepetitionPenaltyInPlace(arena, logits, recent_tokens, penalty);
    tensor.maskExcludedInPlace(logits, cfg.sampling_excluded_tokens);
    return sampler.sampleLogits(graph.executor(), arena, logits);
}

// ============================================================================
// Gemma 4 架构定义与模型实现
// ============================================================================

/// Gemma 4 注意力层类型枚举
pub const Gemma4AttentionType = enum {
    sliding_attention,
    full_attention,

    /// 旋转位置编码 (Rotary Position Embedding, RoPE) 配置：两类层都在完整 head_dim 上半切分配对；
    /// 滑动窗口层为频率基数 10000 的标准 RoPE，全局注意力层为频率基数 1000000、仅前 25% 旋转角参与旋转的
    /// 比例旋转位置编码 (Proportional RoPE)。CPU 与 GPU 路径共用这一份配置。
    pub fn ropeOptions(self: Gemma4AttentionType) tensor.RopeOptions {
        return switch (self) {
            .sliding_attention => .proportional(10000.0, 1.0),
            .full_attention => .proportional(1000000.0, 0.25),
        };
    }

    /// 注意力掩码：滑动窗口层为窗口 sliding_window 的滑动窗口掩码，全局注意力层为因果掩码。
    /// CPU 预填充 (`Graph.attention`)、CPU 增量解码 (`decodeAttention`) 与 GPU 路径共用这一份掩码。
    pub fn mask(self: Gemma4AttentionType, sliding_window: usize) tensor.AttentionMask {
        return switch (self) {
            .sliding_attention => .{ .sliding = sliding_window },
            .full_attention => .causal,
        };
    }
};

/// Gemma 4 词表中采样时始终排除的词元：0 号填充词元 (Padding Token) 与 [258880, 258884] 区间内文本模型不可生成的控制词元
/// (包含官方 `generation_config.json` 的 `suppress_tokens` [258882, 258883])
pub const default_sampling_excluded_tokens = [_]TokenRange{
    .single(0),
    .{ .first = 258880, .last = 258884 },
};

/// Gemma 4 官方生成配置 (`generation_config.json`：do_sample = true, temperature = 1.0, top_k = 64, top_p = 0.95)，
/// 不施加重复惩罚，随机种子为 0
pub const default_generation = SamplingOptions{
    .temperature = 1.0,
    .top_k = 64,
    .top_p = 0.95,
    .repetition_penalty = 1.0,
    .seed = 0,
};

/// Gemma 4 文本模型与主干配置结构体
pub const Gemma4Config = struct {
    vocab_size: usize = 262144,
    hidden_size: usize = 3840,
    intermediate_size: usize = 15360,
    num_hidden_layers: usize = 48,
    num_attention_heads: usize = 16,
    num_key_value_heads: usize = 8,
    num_global_key_value_heads: usize = 1,
    head_dim: usize = 256,
    global_head_dim: usize = 512,
    sliding_window: usize = 1024,
    rms_norm_eps: f32 = 1e-6,
    final_logit_softcapping: f32 = 30.0,
    max_position_embeddings: usize = 262144,
    hidden_size_per_layer_input: usize = 0,
    num_kv_shared_layers: usize = 0,
    attention_k_eq_v: bool = false,
    sliding_rope: tensor.RopeOptions = Gemma4AttentionType.sliding_attention.ropeOptions(),
    full_rope: tensor.RopeOptions = Gemma4AttentionType.full_attention.ropeOptions(),
    layer_types: ?[]const Gemma4AttentionType = null,
    /// 前馈网络与逐层输入门控的激活函数：Gemma 4 官方配置为 `hidden_activation: gelu_pytorch_tanh`，
    /// 即高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 的双曲正切 (Hyperbolic Tangent, Tanh) 近似形式
    hidden_activation: backend.GeluApproximation = .tanh,

    // 特殊词元 ID
    bos_token_id: u32 = 2,
    eos_token_id: u32 = 1,
    pad_token_id: u32 = 0,

    /// 采样时始终排除的词元闭区间：CPU 贪婪 / 随机采样与 GPU 掩码路径共用这唯一一份定义
    sampling_excluded_tokens: []const TokenRange = &default_sampling_excluded_tokens,
    /// 默认生成 (采样) 选项：与官方 `generation_config.json` 一致
    generation: SamplingOptions = default_generation,

    pub const default: Gemma4Config = .{};
    pub fn defaultConfig() Gemma4Config {
        return .{};
    }

    /// 词元 v 是否属于采样排除集合
    pub fn isSamplingExcluded(self: Gemma4Config, v: usize) bool {
        for (self.sampling_excluded_tokens) |r| {
            if (r.contains(v)) return true;
        }
        return false;
    }

    /// 单个解码层的注意力几何规格
    pub const LayerSpec = struct {
        attn_type: Gemma4AttentionType,
        head_dim: usize,
        num_kv_heads: usize,
    };

    /// 获取解码层的注意力类型与几何规格 (支持显式 layer_types 配置或默认每 6 层全注意力)
    pub fn layerSpec(self: Gemma4Config, layer_idx: usize) LayerSpec {
        const attn_type: Gemma4AttentionType = if (self.layer_types) |types|
            types[layer_idx]
        else if ((layer_idx + 1) % 6 == 0)
            .full_attention
        else
            .sliding_attention;
        const is_full = (attn_type == .full_attention);
        return .{
            .attn_type = attn_type,
            .head_dim = if (is_full) self.global_head_dim else self.head_dim,
            .num_kv_heads = if (is_full) self.num_global_key_value_heads else self.num_key_value_heads,
        };
    }

    /// 获取特定注意力类型的旋转位置编码配置
    pub fn ropeOptions(self: Gemma4Config, attn_type: Gemma4AttentionType) tensor.RopeOptions {
        return switch (attn_type) {
            .sliding_attention => self.sliding_rope,
            .full_attention => self.full_rope,
        };
    }

    /// 键值共享层 (KV-Shared Layers) 复用的源层索引：前 num_hidden_layers - num_kv_shared_layers 层各自持有键值缓存，
    /// 其后各层复用最后一个同类型 (滑动窗口 / 全局) 非共享层的键值缓存
    pub fn kvSourceLayer(self: Gemma4Config, layer_idx: usize) usize {
        const first_shared = self.num_hidden_layers - self.num_kv_shared_layers;
        if (layer_idx < first_shared) return layer_idx;
        const attn_type = self.layerSpec(layer_idx).attn_type;
        var j = first_shared;
        while (j > 0) {
            j -= 1;
            if (self.layerSpec(j).attn_type == attn_type) return j;
        }
        unreachable;
    }

    /// 判断某个持有键值缓存的层是否被后续键值共享层复用
    pub fn isKvSharedSource(self: Gemma4Config, layer_idx: usize) bool {
        const first_shared = self.num_hidden_layers - self.num_kv_shared_layers;
        for (first_shared..self.num_hidden_layers) |shared_idx| {
            if (self.kvSourceLayer(shared_idx) == layer_idx) return true;
        }
        return false;
    }

    /// Gemma 4 E4B (4B 级高效端侧模型配置，42 层混合架构与逐层嵌入增强)
    pub const e4b: Gemma4Config = .{
        .vocab_size = 262144,
        .hidden_size = 2560,
        .intermediate_size = 10240,
        .num_hidden_layers = 42,
        .num_attention_heads = 8,
        .num_key_value_heads = 2,
        .num_global_key_value_heads = 2,
        .head_dim = 256,
        .global_head_dim = 512,
        .sliding_window = 512,
        .rms_norm_eps = 1e-6,
        .final_logit_softcapping = 30.0,
        .max_position_embeddings = 131072,
        .hidden_size_per_layer_input = 256,
        .num_kv_shared_layers = 18,
    };

    /// 小型测试配置（用于单测与快速验证，结构完全对齐 Gemma 4 规范）
    pub const tiny_test: Gemma4Config = .{
        .vocab_size = 128,
        .hidden_size = 32,
        .intermediate_size = 64,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .num_global_key_value_heads = 1,
        .head_dim = 8,
        .global_head_dim = 16,
        .sliding_window = 4,
        .rms_norm_eps = 1e-6,
        .final_logit_softcapping = 30.0,
        .max_position_embeddings = 64,
    };
};

/// Gemma 4 专用键值缓存 (Key-Value Cache, KVCache)
/// 针对滑动窗口注意力 (Sliding Window Attention) 与全局全注意力 (Full Attention) 混合异构架构设计。
/// 每个解码层维护各自独立的键 (Key, K) 与值 (Value, V) 连续缓冲区。
pub const Gemma4KVCache = struct {
    pub const LayerCache = struct {
        core: nn.LayerKVCache,
        attn_type: Gemma4AttentionType,

        pub fn init(
            allocator: std.mem.Allocator,
            attn_type: Gemma4AttentionType,
            num_kv_heads: usize,
            head_dim: usize,
            sliding_window: usize,
            max_seq_len: usize,
        ) !LayerCache {
            const sw: ?usize = if (attn_type == .sliding_attention) sliding_window else null;
            const core_cache = try nn.LayerKVCache.init(allocator, num_kv_heads, head_dim, max_seq_len, sw);
            return LayerCache{
                .core = core_cache,
                .attn_type = attn_type,
            };
        }

        pub fn deinit(self: *LayerCache, allocator: std.mem.Allocator) void {
            self.core.deinit(allocator);
        }

        pub fn reset(self: *LayerCache) void {
            self.core.reset();
        }

        pub inline fn windowLen(self: *const LayerCache) usize {
            return self.core.windowLen();
        }

        pub inline fn packWindow(self: *const LayerCache, out_k: []f32, out_v: []f32) !void {
            return self.core.packWindow(out_k, out_v);
        }

        pub inline fn writeWindow(self: *LayerCache, packed_k: []const f32, packed_v: []const f32, win_len: usize, len: usize) !void {
            return self.core.writeWindow(packed_k, packed_v, win_len, len);
        }

        pub inline fn append(self: *LayerCache, step_k: []const f32, step_v: []const f32) !void {
            return self.core.append(step_k, step_v);
        }

        pub inline fn writePrefill(self: *LayerCache, k_seq: []const f32, v_seq: []const f32, seq_len: usize) !void {
            return self.core.writePrefill(k_seq, v_seq, seq_len);
        }

        pub inline fn visibleRange(self: *const LayerCache, pos: usize) tensor.attention.KeyRange {
            return self.core.visibleRange(pos);
        }
    };

    layers: []LayerCache,
    max_seq_len: usize,

    pub const Options = struct {
        max_seq_len: usize = 2048,

        pub const default: Options = .{};
        pub fn defaultOptions() Options {
            return .{};
        }
    };

    pub fn init(allocator: std.mem.Allocator, config: Gemma4Config, options: Options) !Gemma4KVCache {
        const layers = try allocator.alloc(LayerCache, config.num_hidden_layers);
        errdefer allocator.free(layers);

        var initialized_count: usize = 0;
        errdefer {
            for (layers[0..initialized_count]) |*l| {
                l.deinit(allocator);
            }
        }

        for (0..config.num_hidden_layers) |i| {
            const spec = config.layerSpec(i);
            layers[i] = try LayerCache.init(
                allocator,
                spec.attn_type,
                spec.num_kv_heads,
                spec.head_dim,
                config.sliding_window,
                options.max_seq_len,
            );
            initialized_count += 1;
        }

        return Gemma4KVCache{
            .layers = layers,
            .max_seq_len = options.max_seq_len,
        };
    }

    pub fn initDefault(allocator: std.mem.Allocator, config: Gemma4Config) !Gemma4KVCache {
        return init(allocator, config, Options.default);
    }

    pub fn deinit(self: *Gemma4KVCache, allocator: std.mem.Allocator) void {
        for (self.layers) |*l| {
            l.deinit(allocator);
        }
        allocator.free(self.layers);
    }

    pub fn reset(self: *Gemma4KVCache) void {
        for (self.layers) |*l| {
            l.reset();
        }
    }
};

/// Gemma 4 门控前馈多层感知机 (Gemma4MLP)
pub const Gemma4MLP = nn.GatedMLP(core.Linear);

/// Gemma 4 泛型混合注意力机制 (Gemma4AttentionT: 支持滑动窗口注意力和全局注意力)
pub fn Gemma4AttentionT(comptime Proj: type) type {
    return struct {
        const Self = @This();

        q_proj: Proj,
        k_proj: Proj,
        v_proj: ?Proj,
        o_proj: Proj,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        v_norm: RMSNorm,

        attn_type: Gemma4AttentionType,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        rope: tensor.RopeOptions,

        name: ?[]const u8 = null,
        module_type: []const u8 = if (Proj == Linear) "Gemma4Attention" else "Gemma4Q4Attention",

        pub fn init(
            allocator: std.mem.Allocator,
            attn_type: Gemma4AttentionType,
            hidden_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            sliding_window: usize,
            rms_norm_eps: f32,
        ) !Self {
            const q_dim = num_heads * head_dim;
            const kv_dim = num_kv_heads * head_dim;

            var q_proj = if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, hidden_size, q_dim, .{ .bias = false })
            else
                try Proj.init(allocator, hidden_size, q_dim);
            errdefer deinitProj(&q_proj, allocator);

            var k_proj = if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, hidden_size, kv_dim, .{ .bias = false })
            else
                try Proj.init(allocator, hidden_size, kv_dim);
            errdefer deinitProj(&k_proj, allocator);

            var v_proj: ?Proj = if (attn_type == .full_attention)
                null
            else if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, hidden_size, kv_dim, .{ .bias = false })
            else
                try Proj.init(allocator, hidden_size, kv_dim);
            errdefer if (v_proj) |*vp| deinitProj(vp, allocator);

            var o_proj = if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, q_dim, hidden_size, .{ .bias = false })
            else
                try Proj.init(allocator, q_dim, hidden_size);
            errdefer deinitProj(&o_proj, allocator);

            const q_norm = try RMSNorm.init(allocator, head_dim, rms_norm_eps);
            errdefer deinitModel(&q_norm, allocator);
            const k_norm = try RMSNorm.init(allocator, head_dim, rms_norm_eps);
            errdefer deinitModel(&k_norm, allocator);
            const v_norm = RMSNorm.initUnscaled(.{ .eps = rms_norm_eps });

            return Self{
                .q_proj = q_proj,
                .k_proj = k_proj,
                .v_proj = v_proj,
                .o_proj = o_proj,
                .q_norm = q_norm,
                .k_norm = k_norm,
                .v_norm = v_norm,
                .attn_type = attn_type,
                .hidden_size = hidden_size,
                .num_heads = num_heads,
                .num_kv_heads = num_kv_heads,
                .head_dim = head_dim,
                .sliding_window = sliding_window,
                .rope = attn_type.ropeOptions(),
            };
        }

        pub fn initEmpty(
            allocator: std.mem.Allocator,
            attn_type: Gemma4AttentionType,
            hidden_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            sliding_window: usize,
            rms_norm_eps: f32,
        ) !Self {
            if (!@hasDecl(Proj, "fromBlocks")) {
                @compileError("initEmpty only supported for block-quantized projections");
            }
            const q_dim = num_heads * head_dim;
            const kv_dim = num_kv_heads * head_dim;

            const q_proj = Proj.fromBlocks(hidden_size, q_dim, &.{});
            const k_proj = Proj.fromBlocks(hidden_size, kv_dim, &.{});
            const v_proj = if (attn_type == .full_attention) null else Proj.fromBlocks(hidden_size, kv_dim, &.{});
            const o_proj = Proj.fromBlocks(q_dim, hidden_size, &.{});

            const q_norm = try RMSNorm.init(allocator, head_dim, rms_norm_eps);
            errdefer deinitModel(&q_norm, allocator);
            const k_norm = try RMSNorm.init(allocator, head_dim, rms_norm_eps);
            errdefer deinitModel(&k_norm, allocator);
            const v_norm = RMSNorm.initUnscaled(.{ .eps = rms_norm_eps });

            return Self{
                .q_proj = q_proj,
                .k_proj = k_proj,
                .v_proj = v_proj,
                .o_proj = o_proj,
                .q_norm = q_norm,
                .k_norm = k_norm,
                .v_norm = v_norm,
                .attn_type = attn_type,
                .hidden_size = hidden_size,
                .num_heads = num_heads,
                .num_kv_heads = num_kv_heads,
                .head_dim = head_dim,
                .sliding_window = sliding_window,
                .rope = attn_type.ropeOptions(),
            };
        }

        fn deinitProj(proj: *Proj, allocator: std.mem.Allocator) void {
            if (@hasDecl(Proj, "deinit")) {
                proj.deinit(allocator);
            } else {
                deinitModel(proj, allocator);
            }
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            deinitProj(&self.q_proj, allocator);
            deinitProj(&self.k_proj, allocator);
            if (self.v_proj) |*vp| deinitProj(vp, allocator);
            deinitProj(&self.o_proj, allocator);
            deinitModel(&self.q_norm, allocator);
            deinitModel(&self.k_norm, allocator);
        }

        pub const formula = "\\text{Attn}(Q, K, V) = \\text{softmax}\\left(Q K^T + M\\right) V W_o";

        const QKVResult = struct {
            q: *Tensor,
            k: *Tensor,
            v: *Tensor,
        };

        fn projectQKV(self: *const Self, graph: *autodiff.Graph, x_2d: *Tensor) !QKVResult {
            const q_raw = try self.q_proj.forward(graph, x_2d);
            const k_raw = try self.k_proj.forward(graph, x_2d);
            const v_raw = if (self.v_proj) |*vp| try vp.forward(graph, x_2d) else k_raw;
            return .{ .q = q_raw, .k = k_raw, .v = v_raw };
        }

        pub fn forward(self: *const Self, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
            return self.forwardPrefillOptional(graph, x, null);
        }

        pub fn forwardPrefill(
            self: *const Self,
            graph: *autodiff.Graph,
            x: *Tensor,
            layer_cache: *Gemma4KVCache.LayerCache,
        ) !*Tensor {
            return self.forwardPrefillOptional(graph, x, layer_cache);
        }

        fn forwardPrefillOptional(
            self: *const Self,
            graph: *autodiff.Graph,
            x: *Tensor,
            layer_cache: ?*Gemma4KVCache.LayerCache,
        ) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const B = x.shape.dims[0];
            const T = x.shape.dims[1];
            const C = self.hidden_size;
            const nh = self.num_heads;
            const n_kv = self.num_kv_heads;
            const hd = self.head_dim;
            const groups = nh / n_kv;

            const x_2d = try graph.reshape(x, &.{ B * T, C });

            const qkv = try self.projectQKV(graph, x_2d);

            const q_heads = try graph.reshape(qkv.q, &.{ B * T * nh, hd });
            const q_normed = try self.q_norm.forward(graph, q_heads);
            const k_heads = try graph.reshape(qkv.k, &.{ B * T * n_kv, hd });
            const k_normed = try self.k_norm.forward(graph, k_heads);
            const v_heads = try graph.reshape(qkv.v, &.{ B * T * n_kv, hd });
            const v_normed = try self.v_norm.forward(graph, v_heads);

            const q = try nn.splitHeads(graph, q_normed, B, T, nh, hd);
            const k_t_unrot = try nn.splitHeads(graph, k_normed, B, T, n_kv, hd);
            const v = try nn.splitHeads(graph, v_normed, B, T, n_kv, hd);

            const q_rot = try graph.rope(q, 0, self.rope);
            const k_rot = try graph.rope(k_t_unrot, 0, self.rope);

            if (layer_cache) |cache| try writePrefillKVCache(cache, k_rot.data, v.data, B, T);

            const y_4d = try graph.attention(q_rot, k_rot, v, .{
                .scale = 1.0,
                .mask = self.attn_type.mask(self.sliding_window),
                .kv_groups = groups,
            });

            const y_3d = try nn.mergeHeads(graph, y_4d, B, T, nh, hd);
            return try self.o_proj.forward(graph, y_3d);
        }

        pub fn forwardDecodeAttention(
            self: *const Self,
            graph: *autodiff.Graph,
            x: *Tensor,
            layer_cache: *Gemma4KVCache.LayerCache,
        ) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const C = self.hidden_size;
            const nh = self.num_heads;
            const n_kv = self.num_kv_heads;
            const hd = self.head_dim;
            const pos = layer_cache.core.curr_len;
            if (pos >= layer_cache.core.max_seq_len) return error.ContextWindowExceeded;

            const x_2d = if (x.shape.len == 2) x else try graph.reshape(x, &.{ 1, C });

            const qkv = try self.projectQKV(graph, x_2d);

            const q_heads = try graph.reshape(qkv.q, &.{ nh, hd });
            const q_normed = try self.q_norm.forward(graph, q_heads);
            const k_heads = try graph.reshape(qkv.k, &.{ n_kv, hd });
            const k_normed = try self.k_norm.forward(graph, k_heads);
            const v_heads = try graph.reshape(qkv.v, &.{ n_kv, hd });
            const v_normed = try self.v_norm.forward(graph, v_heads);

            const q_4d = try graph.reshape(q_normed, &.{ 1, nh, 1, hd });
            const k_4d = try graph.reshape(k_normed, &.{ 1, n_kv, 1, hd });
            const q_rot = try graph.rope(q_4d, pos, self.rope);
            const k_rot = try graph.rope(k_4d, pos, self.rope);

            return decodeAttention(graph, layer_cache, self.attn_type, self.sliding_window, nh, q_rot.data, k_rot.data, v_normed.data);
        }

        pub fn forwardDecode(
            self: *const Self,
            graph: *autodiff.Graph,
            x: *Tensor,
            layer_cache: *Gemma4KVCache.LayerCache,
        ) !*Tensor {
            const y_2d = try self.forwardDecodeAttention(graph, x, layer_cache);
            const out_2d = try self.o_proj.forward(graph, y_2d);
            return try graph.reshape(out_2d, &.{ 1, 1, self.hidden_size });
        }
    };
}

pub const ple = @import("gemma4_ple.zig");
pub const Gemma4PleLayer = ple.Gemma4PleLayer;
pub const Gemma4PleEmbedding = ple.Gemma4PleEmbedding;

/// Gemma 4 混合注意力机制 (全精度 F32 线性层特化)
pub const Gemma4Attention = Gemma4AttentionT(Linear);

/// Gemma 4 泛型变换器层 (Gemma4DecoderLayerT: 支持 F32、Q4 与可选 PLE 混合层组装)
pub fn Gemma4DecoderLayerT(comptime Attn: type, comptime Mlp: type, comptime Ple: type) type {
    return struct {
        const Self = @This();
        pub const PleType = if (Ple != void) Ple else struct {};

        input_layernorm: RMSNorm,
        self_attn: Attn,
        post_attention_layernorm: RMSNorm,
        pre_feedforward_layernorm: RMSNorm,
        mlp: Mlp,
        post_feedforward_layernorm: RMSNorm,
        ple: PleType,
        layer_scalar: *Tensor, // [1]

        name: ?[]const u8 = null,
        module_type: []const u8 = "Gemma4DecoderLayer",

        pub fn init(
            allocator: std.mem.Allocator,
            attn_type: Gemma4AttentionType,
            hidden_size: usize,
            intermediate_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            sliding_window: usize,
            rms_norm_eps: f32,
            hidden_activation: backend.GeluApproximation,
        ) !Self {
            return initWithPle(
                allocator,
                attn_type,
                hidden_size,
                intermediate_size,
                num_heads,
                num_kv_heads,
                head_dim,
                sliding_window,
                rms_norm_eps,
                hidden_activation,
                0,
            );
        }

        pub fn initWithPle(
            allocator: std.mem.Allocator,
            attn_type: Gemma4AttentionType,
            hidden_size: usize,
            intermediate_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            sliding_window: usize,
            rms_norm_eps: f32,
            hidden_activation: backend.GeluApproximation,
            ple_dim: usize,
        ) !Self {
            const input_layernorm = try RMSNorm.init(allocator, hidden_size, rms_norm_eps);
            errdefer deinitModel(&input_layernorm, allocator);

            var self_attn = try Attn.init(
                allocator,
                attn_type,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                sliding_window,
                rms_norm_eps,
            );
            errdefer if (@hasDecl(Attn, "deinit")) self_attn.deinit(allocator) else deinitModel(&self_attn, allocator);

            const post_attention_layernorm = try RMSNorm.init(allocator, hidden_size, rms_norm_eps);
            errdefer deinitModel(&post_attention_layernorm, allocator);

            const pre_feedforward_layernorm = try RMSNorm.init(allocator, hidden_size, rms_norm_eps);
            errdefer deinitModel(&pre_feedforward_layernorm, allocator);

            var mlp = try Mlp.initGemma(allocator, hidden_size, intermediate_size, hidden_activation);
            errdefer if (@hasDecl(Mlp, "deinit")) mlp.deinit(allocator) else deinitModel(&mlp, allocator);

            const post_feedforward_layernorm = try RMSNorm.init(allocator, hidden_size, rms_norm_eps);
            errdefer deinitModel(&post_feedforward_layernorm, allocator);

            var ple_sub: PleType = if (Ple != void) blk: {
                const p = try Ple.init(allocator, hidden_size, ple_dim, rms_norm_eps, hidden_activation);
                break :blk p;
            } else .{};
            errdefer if (Ple != void) {
                if (@hasDecl(Ple, "deinit")) ple_sub.deinit(allocator) else deinitModel(&ple_sub, allocator);
            };

            const layer_scalar = try createLayerScalar(allocator);
            errdefer freePersistentTensor(allocator, layer_scalar);

            return Self{
                .input_layernorm = input_layernorm,
                .self_attn = self_attn,
                .post_attention_layernorm = post_attention_layernorm,
                .pre_feedforward_layernorm = pre_feedforward_layernorm,
                .mlp = mlp,
                .post_feedforward_layernorm = post_feedforward_layernorm,
                .ple = ple_sub,
                .layer_scalar = layer_scalar,
            };
        }

        pub fn initEmpty(
            allocator: std.mem.Allocator,
            attn_type: Gemma4AttentionType,
            hidden_size: usize,
            intermediate_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            sliding_window: usize,
            rms_norm_eps: f32,
            hidden_activation: backend.GeluApproximation,
        ) !Self {
            return initEmptyWithPle(
                allocator,
                attn_type,
                hidden_size,
                intermediate_size,
                num_heads,
                num_kv_heads,
                head_dim,
                sliding_window,
                rms_norm_eps,
                hidden_activation,
                0,
            );
        }

        pub fn initEmptyWithPle(
            allocator: std.mem.Allocator,
            attn_type: Gemma4AttentionType,
            hidden_size: usize,
            intermediate_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            sliding_window: usize,
            rms_norm_eps: f32,
            hidden_activation: backend.GeluApproximation,
            ple_dim: usize,
        ) !Self {
            const input_layernorm = try RMSNorm.init(allocator, hidden_size, rms_norm_eps);
            errdefer deinitModel(&input_layernorm, allocator);

            var self_attn = try Attn.initEmpty(
                allocator,
                attn_type,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                sliding_window,
                rms_norm_eps,
            );
            errdefer if (@hasDecl(Attn, "deinit")) self_attn.deinit(allocator) else deinitModel(&self_attn, allocator);

            const post_attention_layernorm = try RMSNorm.init(allocator, hidden_size, rms_norm_eps);
            errdefer deinitModel(&post_attention_layernorm, allocator);

            const pre_feedforward_layernorm = try RMSNorm.init(allocator, hidden_size, rms_norm_eps);
            errdefer deinitModel(&pre_feedforward_layernorm, allocator);

            const mlp = Mlp.initEmpty(hidden_size, intermediate_size, nn.GatedActivation.fromGelu(hidden_activation));

            const post_feedforward_layernorm = try RMSNorm.init(allocator, hidden_size, rms_norm_eps);
            errdefer deinitModel(&post_feedforward_layernorm, allocator);

            var ple_sub: PleType = if (Ple != void) blk: {
                const p = try Ple.initEmpty(allocator, hidden_size, ple_dim, rms_norm_eps, hidden_activation);
                break :blk p;
            } else .{};
            errdefer if (Ple != void) {
                if (@hasDecl(Ple, "deinit")) ple_sub.deinit(allocator) else deinitModel(&ple_sub, allocator);
            };

            const layer_scalar = try createLayerScalar(allocator);
            errdefer freePersistentTensor(allocator, layer_scalar);

            return Self{
                .input_layernorm = input_layernorm,
                .self_attn = self_attn,
                .post_attention_layernorm = post_attention_layernorm,
                .pre_feedforward_layernorm = pre_feedforward_layernorm,
                .mlp = mlp,
                .post_feedforward_layernorm = post_feedforward_layernorm,
                .ple = ple_sub,
                .layer_scalar = layer_scalar,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            deinitModel(&self.input_layernorm, allocator);
            if (@hasDecl(Attn, "deinit")) self.self_attn.deinit(allocator) else deinitModel(&self.self_attn, allocator);
            deinitModel(&self.post_attention_layernorm, allocator);
            deinitModel(&self.pre_feedforward_layernorm, allocator);
            if (@hasDecl(Mlp, "deinit")) self.mlp.deinit(allocator) else deinitModel(&self.mlp, allocator);
            deinitModel(&self.post_feedforward_layernorm, allocator);
            if (Ple != void) {
                if (@hasDecl(Ple, "deinit")) self.ple.deinit(allocator) else deinitModel(&self.ple, allocator);
            }
            freePersistentTensor(allocator, self.layer_scalar);
        }

        pub fn enableHardwareOffload(self: *Self, allocator: std.mem.Allocator) !void {
            if (@hasField(Attn, "q_proj") and @hasDecl(@TypeOf(self.self_attn.q_proj), "enableOffload")) {
                try self.self_attn.q_proj.enableOffload(allocator, null);
                try self.self_attn.k_proj.enableOffload(allocator, null);
                if (self.self_attn.v_proj) |*vp| try vp.enableOffload(allocator, null);
                try self.self_attn.o_proj.enableOffload(allocator, null);
            }
            if (@hasField(Mlp, "gate_proj") and @hasDecl(@TypeOf(self.mlp.gate_proj), "enableOffload")) {
                try self.mlp.gate_proj.enableOffload(allocator, null);
                try self.mlp.up_proj.enableOffload(allocator, null);
                try self.mlp.down_proj.enableOffload(allocator, null);
            }
            if (Ple != void) {
                if (@hasDecl(Ple, "enableHardwareOffload")) {
                    try self.ple.enableHardwareOffload(allocator);
                }
            }
        }

        pub const formula = "\\begin{aligned} x_1 &= x + \\text{PostAttnLN}(\\text{Attn}(\\text{InputLN}(x))) \\\\ x_2 &= s \\cdot \\left(x_1 + \\text{PostMLPLN}(\\text{MLP}(\\text{PreMLPLN}(x_1)))\\right) \\end{aligned}";

        fn residualTail(self: *const Self, graph: *autodiff.Graph, x: *Tensor, attn_out: *Tensor, ple_slice: ?*Tensor) !*Tensor {
            const post_attn = try self.post_attention_layernorm.forward(graph, attn_out);
            const x1 = try graph.add(x, post_attn);

            const pre_mlp = try self.pre_feedforward_layernorm.forward(graph, x1);
            const mlp_out = try self.mlp.forward(graph, pre_mlp);
            const post_mlp = try self.post_feedforward_layernorm.forward(graph, mlp_out);
            const x2 = try graph.add(x1, post_mlp);

            if (Ple != void) {
                if (ple_slice) |ps| {
                    return try self.ple.forward(graph, x2, ps, self.layer_scalar);
                }
            }
            return try graph.mulScalar(x2, self.layer_scalar.data[0]);
        }

        pub fn forward(self: *const Self, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
            return self.forwardWithPle(graph, x, null);
        }

        pub fn forwardWithPle(self: *const Self, graph: *autodiff.Graph, x: *Tensor, ple_slice: ?*Tensor) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const in_norm = try self.input_layernorm.forward(graph, x);
            const attn_out = try self.self_attn.forward(graph, in_norm);
            return self.residualTail(graph, x, attn_out, ple_slice);
        }

        pub fn forwardPrefill(
            self: *const Self,
            graph: *autodiff.Graph,
            x: *Tensor,
            layer_cache: *Gemma4KVCache.LayerCache,
        ) !*Tensor {
            return self.forwardPrefillWithPle(graph, x, layer_cache, null);
        }

        pub fn forwardPrefillWithPle(
            self: *const Self,
            graph: *autodiff.Graph,
            x: *Tensor,
            layer_cache: *Gemma4KVCache.LayerCache,
            ple_slice: ?*Tensor,
        ) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const in_norm = try self.input_layernorm.forward(graph, x);
            const attn_out = try self.self_attn.forwardPrefill(graph, in_norm, layer_cache);
            return self.residualTail(graph, x, attn_out, ple_slice);
        }

        pub fn forwardDecode(
            self: *const Self,
            graph: *autodiff.Graph,
            x: *Tensor,
            layer_cache: *Gemma4KVCache.LayerCache,
        ) !*Tensor {
            return self.forwardDecodeWithPle(graph, x, layer_cache, null);
        }

        pub fn forwardDecodeWithPle(
            self: *const Self,
            graph: *autodiff.Graph,
            x: *Tensor,
            layer_cache: *Gemma4KVCache.LayerCache,
            ple_slice: ?*Tensor,
        ) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const in_norm = try self.input_layernorm.forward(graph, x);
            const attn_out = try self.self_attn.forwardDecode(graph, in_norm, layer_cache);
            return self.residualTail(graph, x, attn_out, ple_slice);
        }
    };
}

/// Gemma 4 变换器层 (全精度特化)
pub const Gemma4DecoderLayer = Gemma4DecoderLayerT(Gemma4Attention, Gemma4MLP, void);

/// Gemma 4 语言模型核心架构 (Gemma4ForCausalLM)
pub fn Gemma4ForCausalLM(comptime cfg: Gemma4Config) type {
    return struct {
        const Self = @This();
        pub const config: Gemma4Config = cfg;

        embed_tokens: Embedding,
        layers: []Gemma4DecoderLayer,
        norm: RMSNorm,
        name: ?[]const u8 = null,
        module_type: []const u8 = "Gemma4ForCausalLM",

        pub fn initDefault(allocator: std.mem.Allocator) !Self {
            return init(allocator);
        }

        pub fn init(allocator: std.mem.Allocator) !Self {
            return initWithOptions(allocator, true);
        }

        /// 推理专用模型初始化：逐层释放未使用的梯度缓冲区 (.grad)，将 12B 全精度模型的内存占用减半 (从 ~91 GB 降至 ~45.5 GB)
        pub fn initInference(allocator: std.mem.Allocator) !Self {
            return initWithOptions(allocator, false);
        }

        fn initWithOptions(allocator: std.mem.Allocator, keep_grad: bool) !Self {
            var embed_tokens = try Embedding.init(allocator, config.vocab_size, config.hidden_size);
            errdefer deinitModel(&embed_tokens, allocator);
            if (!keep_grad) releaseModuleGrads(&embed_tokens, allocator);

            const layers = try allocator.alloc(Gemma4DecoderLayer, config.num_hidden_layers);
            var initialized_layers: usize = 0;
            errdefer {
                for (0..initialized_layers) |idx| {
                    deinitModel(&layers[idx], allocator);
                }
                allocator.free(layers);
            }

            for (0..config.num_hidden_layers) |i| {
                const spec = config.layerSpec(i);
                layers[i] = try Gemma4DecoderLayer.init(
                    allocator,
                    spec.attn_type,
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                    spec.num_kv_heads,
                    spec.head_dim,
                    config.sliding_window,
                    config.rms_norm_eps,
                    config.hidden_activation,
                );
                if (!keep_grad) releaseModuleGrads(&layers[i], allocator);
                initialized_layers += 1;
            }

            var norm = try RMSNorm.init(allocator, config.hidden_size, config.rms_norm_eps);
            errdefer deinitModel(&norm, allocator);
            if (!keep_grad) releaseModuleGrads(&norm, allocator);

            return Self{
                .embed_tokens = embed_tokens,
                .layers = layers,
                .norm = norm,
            };
        }

        pub const formula = "\\text{logits} = \\text{Softcap}\\left(\\text{Norm}(\\text{Layers}(\\text{Embed}(x) \\cdot \\sqrt{d})) W_e^T, c\\right)";

        pub fn forward(self: *const Self, graph: *autodiff.Graph, token_ids: anytype) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const B = token_ids.shape.dims[0];
            const T = token_ids.shape.dims[1];

            // 1. 词嵌入与 Gemma 嵌入缩放 sqrt(hidden_size)
            const tok_emb = try self.embed_tokens.forward(graph, token_ids);
            const scale = @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
            var h = try graph.mulScalar(tok_emb, scale);

            // 2. 逐层前向传播
            for (self.layers) |*layer| {
                h = try layer.forward(graph, h);
            }

            // 3. 最终归一化
            const h_norm = try self.norm.forward(graph, h);

            // 4. 绑定权重投影 (Tie Word Embeddings: h * embed_tokens.weight^T)
            const h_2d = try graph.reshape(h_norm, &.{ B * T, config.hidden_size });
            const embed_t = try graph.transposeND(self.embed_tokens.weight, 0, 1);
            const logits_raw_2d = try graph.matmul(h_2d, embed_t);

            // 5. 最终对数几率软截断 (Final Logit Softcapping): cap * tanh(logits / cap)
            const capped_logits_2d = try graph.softcap(logits_raw_2d, config.final_logit_softcapping);

            return try graph.reshape(capped_logits_2d, &.{ B, T, config.vocab_size });
        }

        /// 对最终隐状态 h_last 选取下一个词元 (仅最近 repetition_window 个词元受重复惩罚)
        fn sampleNextTokenF32(
            self: *const Self,
            h_last: []const f32,
            recent_tokens: []const usize,
            sampler: anytype,
            graph: *autodiff.Graph,
        ) !usize {
            return sampleLmHeadCpu(config, f32, graph, self.embed_tokens.weight.data, h_last, recentWindowFor(recent_tokens, sampler.options), sampler);
        }

        /// 全精度 (F32) 预填充阶段 (Prefill)：处理 Prompt 词元序列并写入 KVCache，按 sampler 选取并返回首个预测词元
        pub fn prefill(
            self: *const Self,
            graph: *autodiff.Graph,
            prompt_tokens: []const usize,
            cache: *Gemma4KVCache,
            sampler: anytype,
        ) !usize {
            const T = prompt_tokens.len;
            if (T == 0) return error.EmptyPrompt;
            const tok_emb = try embedTokensScaled(f32, graph, self.embed_tokens.weight.data, config.hidden_size, prompt_tokens);
            cache.reset();

            var h = tok_emb;
            for (self.layers, 0..) |*layer, l_idx| {
                h = try layer.forwardPrefill(graph, h, &cache.layers[l_idx]);
            }

            const h_norm = try self.norm.forward(graph, h);
            const last_row = h_norm.data[(T - 1) * config.hidden_size .. T * config.hidden_size];
            return self.sampleNextTokenF32(last_row, prompt_tokens, sampler, graph);
        }

        /// 全精度 (F32) 增量解码单步 (Decode Step)：基于 KVCache 处理单个增量词元，按 sampler 选取并返回下一个预测词元
        pub fn decodeStep(
            self: *const Self,
            graph: *autodiff.Graph,
            token: usize,
            cache: *Gemma4KVCache,
            recent_tokens: []const usize,
            sampler: anytype,
        ) !usize {
            const tok_emb = try embedTokensScaled(f32, graph, self.embed_tokens.weight.data, config.hidden_size, &.{token});

            var h = tok_emb;
            for (self.layers, 0..) |*layer, l_idx| {
                h = try layer.forwardDecode(graph, h, &cache.layers[l_idx]);
            }

            const h_norm = try self.norm.forward(graph, h);
            const last_row = h_norm.data[0..config.hidden_size];
            return self.sampleNextTokenF32(last_row, recent_tokens, sampler, graph);
        }
    };
}

/// 默认配置的 Gemma 4 12B 模型类型
pub const DefaultGemma4 = Gemma4ForCausalLM(Gemma4Config.default);

/// 测试配置的微型 Gemma 4 模型类型
pub const TinyGemma4 = Gemma4ForCausalLM(Gemma4Config.tiny_test);

/// Gemma 4 完整 262K 词表零拷贝解码器 (Gemma4Vocabulary)
/// 支持直接映射二进制词表文件，以 O(1) 性能根据 Token ID 返回具体文本内容
pub const Gemma4Vocabulary = struct {
    mmap_ptr: []const u8,
    offsets: []const u32,
    str_data: []const u8,
    count: usize,
    lookup_table: ?std.StringHashMapUnmanaged(usize) = null,
    lookup_allocator: ?std.mem.Allocator = null,

    pub const MAGIC: *const [8]u8 = "ZNNVOC01";

    pub fn loadFromMmap(mmap_bytes: []const u8) !Gemma4Vocabulary {
        if (mmap_bytes.len < 16) return error.InvalidVocabFile;
        if (!std.mem.eql(u8, mmap_bytes[0..8], MAGIC)) return error.InvalidVocabMagic;

        const count = std.mem.readInt(u32, mmap_bytes[8..12], .little);
        const total_str_bytes = std.mem.readInt(u32, mmap_bytes[12..16], .little);

        const offsets_start: usize = 16;
        const offsets_end: usize = offsets_start + count * @sizeOf(u32);
        if (mmap_bytes.len < offsets_end + total_str_bytes) return error.VocabTruncated;

        const offsets_bytes = mmap_bytes[offsets_start..offsets_end];
        const offsets = @as([]const u32, @alignCast(std.mem.bytesAsSlice(u32, offsets_bytes)));
        const str_data = mmap_bytes[offsets_end .. offsets_end + total_str_bytes];

        return Gemma4Vocabulary{
            .mmap_ptr = mmap_bytes,
            .offsets = offsets,
            .str_data = str_data,
            .count = count,
        };
    }

    pub fn openFile(io: std.Io, path: []const u8) !Gemma4Vocabulary {
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.openFile(io, path, .{});
        defer file.close(io);

        const stat = try file.stat(io);
        const mmap_ptr = try std.posix.mmap(
            null,
            @as(usize, @intCast(stat.size)),
            .{ .READ = true },
            .{ .TYPE = .SHARED },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(mmap_ptr);

        return loadFromMmap(mmap_ptr);
    }

    /// 构建高速反向词表哈希索引，将 Token 查找从 O(N) 线性扫描降至 O(1)
    pub fn buildLookupTable(self: *Gemma4Vocabulary, allocator: std.mem.Allocator) !void {
        if (self.lookup_table != null) return;
        var table = std.StringHashMapUnmanaged(usize){};
        errdefer table.deinit(allocator);
        try table.ensureTotalCapacity(allocator, @intCast(self.count));
        for (0..self.count) |i| {
            const piece = self.decode(i);
            if (piece.len > 0) {
                const gop = try table.getOrPut(allocator, piece);
                if (!gop.found_existing) {
                    gop.value_ptr.* = i;
                }
            }
        }
        self.lookup_table = table;
        self.lookup_allocator = allocator;
    }

    pub fn deinit(self: *Gemma4Vocabulary) void {
        if (self.lookup_table) |*table| {
            if (self.lookup_allocator) |alloc| {
                table.deinit(alloc);
            }
            self.lookup_table = null;
        }
        std.posix.munmap(@alignCast(self.mmap_ptr));
    }

    // 官方保留特殊控制词元 (Special Control Tokens)
    pub const PAD: usize = 0; // <pad>
    pub const EOS: usize = 1; // <eos> (End of Sequence)
    pub const BOS: usize = 2; // <bos> (Beginning of Sequence)
    pub const UNK: usize = 3; // <unk>
    pub const THINK: usize = 98; // <|think|>
    pub const SOC: usize = 100; // <|channel> (Start of Channel)
    pub const EOC: usize = 101; // <channel|> (End of Channel)
    pub const SOT: usize = 105; // <|turn> (Start of Turn)
    pub const EOT: usize = 106; // <turn|> (End of Turn)
    pub const NL: usize = 107; // \n
    pub const USER: usize = 2364; // user
    pub const MODEL: usize = 4368; // model
    pub const SYSTEM: usize = 9731; // system

    /// 对话轮次角色 (Conversation Turn Role)
    pub const Role = enum {
        user,
        model,
        system,

        pub fn tokenId(self: Role) usize {
            return switch (self) {
                .user => USER,
                .model => MODEL,
                .system => SYSTEM,
            };
        }
    };

    /// 结构化对话消息 (Chat Message)
    pub const Message = struct {
        role: Role,
        content: []const u8,
    };

    /// O(1) 根据 token_id 获取对应解码文本切片
    pub fn decode(self: *const Gemma4Vocabulary, token_id: usize) []const u8 {
        if (token_id >= self.count) return "";
        const start = self.offsets[token_id];
        const end: usize = if (token_id + 1 < self.count) self.offsets[token_id + 1] else self.str_data.len;
        if (start > end or end > self.str_data.len) return "";
        return self.str_data[start..end];
    }

    /// 在词表中根据确切文本反查对应的 Token ID (若不存在返回 null)
    pub fn findToken(self: *const Gemma4Vocabulary, text: []const u8) ?usize {
        if (self.lookup_table) |table| {
            return table.get(text);
        }
        for (0..self.count) |i| {
            if (std.mem.eql(u8, self.decode(i), text)) return i;
        }
        return null;
    }

    /// 将任意 UTF-8 文本贪婪切分为 Gemma 4 词元序列 (Greedy Longest Matching Tokenization)。
    /// 无法匹配任何词元的字符按 SentencePiece 字节回退规则编码为 `<0xNN>` 字节词元，绝不静默丢弃；
    /// 词表缺少所需字节词元时返回 `error.UnencodableByte`。
    pub fn encode(self: *const Gemma4Vocabulary, allocator: std.mem.Allocator, text: []const u8) ![]usize {
        var tokens: std.ArrayList(usize) = .empty;
        errdefer tokens.deinit(allocator);

        var idx: usize = 0;
        var at_start = true;
        while (idx < text.len) {
            var best_len: usize = 0;
            var best_token: ?usize = null;

            // 1. 尝试匹配以空格 ' ' (0x20) 开头的词元
            var cand_buf: [256]u8 = undefined;
            if (at_start or text[idx] == ' ') {
                const effective_idx = if (text[idx] == ' ') idx + 1 else idx;
                const max_check = @min(effective_idx + 48, text.len);
                var sub_len = max_check - effective_idx;
                while (sub_len > 0) : (sub_len -= 1) {
                    const chunk = text[effective_idx .. effective_idx + sub_len];
                    if (1 + chunk.len <= cand_buf.len) {
                        cand_buf[0] = ' ';
                        @memcpy(cand_buf[1 .. 1 + chunk.len], chunk);
                        if (self.findToken(cand_buf[0 .. 1 + chunk.len])) |tok| {
                            best_token = tok;
                            best_len = (if (text[idx] == ' ') @as(usize, 1) else 0) + sub_len;
                            break;
                        }
                    }
                }
            }

            // 2. 若无前缀   匹配，则尝试纯文本贪婪最长匹配
            if (best_token == null) {
                const max_check = @min(idx + 48, text.len);
                var sub_len = max_check - idx;
                while (sub_len > 0) : (sub_len -= 1) {
                    const chunk = text[idx .. idx + sub_len];
                    if (self.findToken(chunk)) |tok| {
                        best_token = tok;
                        best_len = sub_len;
                        break;
                    }
                }
            }

            if (best_token) |tok| {
                try tokens.append(allocator, tok);
                idx += best_len;
            } else {
                // 字节回退 (Byte Fallback)：与 Gemma / SentencePiece 一致，把无法匹配的整个 UTF-8 字符逐字节编码为 `<0xNN>` 字节词元
                const char_len: usize = std.unicode.utf8ByteSequenceLength(text[idx]) catch 1;
                const end = @min(idx + char_len, text.len);
                for (text[idx..end]) |byte| {
                    try tokens.append(allocator, try self.byteToken(byte));
                }
                idx = end;
            }
            at_start = false;
        }

        return tokens.toOwnedSlice(allocator);
    }

    /// 查找单字节回退词元 `<0xNN>` (两位大写十六进制)；词表不含该字节词元时返回 `error.UnencodableByte`
    pub fn byteToken(self: *const Gemma4Vocabulary, byte: u8) !usize {
        var buf: [6]u8 = undefined;
        const piece = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{byte}) catch unreachable;
        return self.findToken(piece) orelse error.UnencodableByte;
    }

    /// 追加单轮对话片段到词元列表：`<|turn>{role}\n{content}<turn|>\n`
    pub fn appendTurn(
        self: *const Gemma4Vocabulary,
        list: *std.ArrayList(usize),
        allocator: std.mem.Allocator,
        role: Role,
        content: []const u8,
    ) !void {
        try list.append(allocator, SOT);
        try list.append(allocator, role.tokenId());
        try list.append(allocator, NL);
        const tokens = try self.encode(allocator, content);
        defer allocator.free(tokens);
        try list.appendSlice(allocator, tokens);
        try list.append(allocator, EOT);
        try list.append(allocator, NL);
    }

    /// 追加生成引导头：`<|turn>{role}\n`（默认 model）
    pub fn appendGenerationPrompt(
        list: *std.ArrayList(usize),
        allocator: std.mem.Allocator,
        role: Role,
    ) !void {
        try list.append(allocator, SOT);
        try list.append(allocator, role.tokenId());
        try list.append(allocator, NL);
    }

    /// 将多轮对话格式化为符合 Gemma 4 Instruct 标准的词元序列：
    /// `<bos>` + 若干 `[<|turn>{role}\n{content}<turn|>\n]` + 可选 `[<|turn>model\n]`
    pub fn formatChatPrompt(
        self: *const Gemma4Vocabulary,
        allocator: std.mem.Allocator,
        messages: []const Message,
        add_generation_prompt: bool,
    ) ![]usize {
        var list: std.ArrayList(usize) = .empty;
        errdefer list.deinit(allocator);

        try list.append(allocator, BOS);
        for (messages) |msg| {
            try self.appendTurn(&list, allocator, msg.role, msg.content);
        }
        if (add_generation_prompt) {
            try appendGenerationPrompt(&list, allocator, .model);
        }
        return list.toOwnedSlice(allocator);
    }

    /// 便捷方法：将单轮用户输入格式化为完整的 Gemma 4 Instruct 提示词词元切片：
    /// `<bos><|turn>user\n{user_prompt}<turn|>\n<|turn>model\n`
    pub fn formatInstructPrompt(
        self: *const Gemma4Vocabulary,
        allocator: std.mem.Allocator,
        user_prompt: []const u8,
    ) ![]usize {
        return self.formatChatPrompt(
            allocator,
            &.{.{ .role = .user, .content = user_prompt }},
            true,
        );
    }
};

/// 基于上下文 N-gram 匹配的投机采样候选生成器 (Prompt Lookup Decoding / N-gram Speculative Sampling)
// ============================================================================
// 投机采样推测草稿原语下沉至 nn.llm，此处保持重导出
// ============================================================================

pub const promptLookup = nn.promptLookup;
pub const SpeculativeDraftEngine = nn.SpeculativeDraftEngine;

/// 预注入通用语言与编程高频共现搭配 (Collocations)
pub fn seedCommonCollocations(engine: *SpeculativeDraftEngine, vocab: *const Gemma4Vocabulary) !void {
    const pairs = [_][2][]const u8{
        .{ " language", " model" },
        .{ " developed", " by" },
        .{ " artificial", " intelligence" },
        .{ " machine", " learning" },
        .{ " neural", " network" },
        .{ " deep", " learning" },
        .{ " large", " language" },
        .{ " in", " order" },
        .{ " order", " to" },
        .{ " as", " well" },
        .{ " well", " as" },
        .{ " for", " example" },
        .{ " such", " as" },
        .{ " based", " on" },
        .{ " according", " to" },
        .{ " thank", " you" },
        .{ " help", " answer" },
        .{ " answer", " your" },
        .{ " your", " questions" },
        .{ " Open", "AI" },
        .{ "def", " __init__" },
        .{ "pub", " fn" },
        .{ "return", " error" },
    };
    for (pairs) |pair| {
        if (vocab.findToken(pair[0])) |t1| {
            if (vocab.findToken(pair[1])) |t2| {
                try engine.recordTransition(t1, t2);
            }
        }
    }
}
