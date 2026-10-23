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
const createPersistentTensor = core.createPersistentTensor;
const freePersistentTensor = core.freePersistentTensor;
const deinitModel = core.deinitModel;
const enterModuleScope = core.enterModuleScope;
const sampling = backend.sampling;
const TokenRange = sampling.TokenRange;
const Sampler = sampling.Sampler;
const SamplingOptions = sampling.SamplingOptions;

// ============================================================================
// Gemma 4 专用非通用算子 (Split-Half & Proportional RoPE)
// ============================================================================

/// 标准 GPT-NeoX / LLaMA / Gemma 规范的半切分旋转位置编码张量前向核函数 (Split-Half RoPE)
/// 算法原理：将输入维度分为前半段 x1 与后半段 x2，rotate_half(x) = [-x2, x1]
/// 对于每个旋转角 i (0 <= i < num_angles):
/// out[i] = x[i] * cos(theta_i) - x[i + half] * sin(theta_i)
/// out[i + half] = x[i + half] * cos(theta_i) + x[i] * sin(theta_i)
/// 当 partial_rotary_factor < 1.0 (例如 Gemma 4 的 proportional RoPE) 时，
/// 仅前 num_angles = int(partial_rotary_factor * D / 2) 个分量参与旋转，其余分量保持原始值 (cos=1, sin=0)。
pub fn tensorRopeSplitHalf(
    self: *Tensor,
    start_pos: usize,
    partial_rotary_factor: f32,
    rope_theta: f32,
    allocator: std.mem.Allocator,
) !*Tensor {
    return self.rope(start_pos, .{
        .mode = .split_half,
        .partial_rotary_factor = partial_rotary_factor,
        .base = rope_theta,
    }, allocator);
}

/// Gemma 4 计算图级半切分旋转位置编码算子
pub fn ropeSplitHalf(
    graph: *autodiff.Graph,
    X: *Tensor,
    start_pos: usize,
    partial_rotary_factor: f32,
    rope_theta: f32,
) !*Tensor {
    return graph.rope(X, start_pos, .{
        .mode = .split_half,
        .partial_rotary_factor = partial_rotary_factor,
        .base = rope_theta,
    });
}

// ============================================================================
// 单指令多数据 (Single Instruction Multiple Data, SIMD) 点积内核
// ============================================================================

const dot_lanes = 16;
const F32Vec = @Vector(dot_lanes, f32);
const U32Vec = @Vector(dot_lanes, u32);

/// 单精度浮点 (F32) 向量点积：按 16 路 SIMD 累加，尾部逐元素补齐
pub fn dotF32(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc: F32Vec = @splat(0.0);
    var i: usize = 0;
    while (i + dot_lanes <= a.len) : (i += dot_lanes) {
        const va: F32Vec = a[i..][0..dot_lanes].*;
        const vb: F32Vec = b[i..][0..dot_lanes].*;
        acc += va * vb;
    }
    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// 单精度浮点 (F32) 与脑浮点 (Brain Floating Point 16, BF16) 原始位向量点积：
/// BF16 位模式左移 16 位即为等值 F32，整段以 SIMD 完成类型提升与乘加
pub fn dotF32Bf16(a: []const f32, b_bits: []const u16) f32 {
    std.debug.assert(a.len == b_bits.len);
    var acc: F32Vec = @splat(0.0);
    var i: usize = 0;
    while (i + dot_lanes <= a.len) : (i += dot_lanes) {
        const va: F32Vec = a[i..][0..dot_lanes].*;
        const raw: @Vector(dot_lanes, u16) = b_bits[i..][0..dot_lanes].*;
        const widened: U32Vec = @as(U32Vec, raw) << @splat(16);
        const vb: F32Vec = @bitCast(widened);
        acc += va * vb;
    }
    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) {
        const b: tensor.bf16 = .{ .bits = b_bits[i] };
        sum += a[i] * b.toF32();
    }
    return sum;
}

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
    if (seq_len > cache.max_seq_len) return error.ContextWindowExceeded;
    const n_kv = cache.num_kv_heads;
    const hd = cache.head_dim;
    const max_len = cache.max_seq_len;
    std.debug.assert(k_rot.len == n_kv * seq_len * hd and v.len == k_rot.len);
    for (0..n_kv) |kv_h| {
        const dest = (kv_h * max_len) * hd;
        const src = (kv_h * seq_len) * hd;
        @memcpy(cache.k[dest .. dest + seq_len * hd], k_rot[src .. src + seq_len * hd]);
        @memcpy(cache.v[dest .. dest + seq_len * hd], v[src .. src + seq_len * hd]);
    }
    cache.curr_len = seq_len;
}

/// 预填充阶段因果 / 滑动窗口注意力 (官方实现中 scaling = 1.0)
/// q_rot、k、v 形状均为 [B, nh, T, hd] (键值已完成分组查询注意力 (Grouped-Query Attention, GQA) 广播)。
/// 全局注意力层或 T 不超过滑动窗口时等价于标准因果注意力，直接分发至融合缩放点积注意力
/// (Scaled Dot-Product Attention, SDPA)；否则使用附加滑动窗口掩码的分解计算图。
pub fn prefillAttention(
    graph: *autodiff.Graph,
    attn_type: Gemma4AttentionType,
    sliding_window: usize,
    q_rot: *Tensor,
    k: *Tensor,
    v: *Tensor,
) !*Tensor {
    const T = q_rot.shape.dims[2];
    const is_pure_causal = (attn_type == .full_attention or T <= sliding_window);
    if (is_pure_causal) return graph.scaledDotProductAttention(q_rot, k, v, 1.0, true);

    const k_trans = try graph.transposeND(k, 2, 3);
    const scores = try graph.batchMatMul(q_rot, k_trans);

    const mask_node = try graph.tensorND(&.{ 1, 1, T, T }, false);
    mask_node.is_buffer = true;
    for (0..T) |i| {
        const row = mask_node.data[i * T .. (i + 1) * T];
        for (row, 0..) |*m, j| {
            // 未来位置因果遮蔽，以及超出局部滑动窗口的历史位置遮蔽
            const visible = j <= i and i < j + sliding_window;
            m.* = if (visible) 0.0 else -1e9;
        }
    }
    const masked = try graph.add(scores, mask_node);
    const att_sm = try graph.softmax(masked);
    return graph.batchMatMul(att_sm, v);
}

/// 单查询增量解码注意力：将当前时间步旋转后的键 (k_rot: [n_kv, hd]) 与值 (v: [n_kv, hd]) 追加写入层键值缓存，
/// 随后对缓存中可见窗口 (滑动窗口层仅最近 sliding_window 个位置) 执行数值稳定 (减最大值) 的 Softmax 加权聚合。
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
    const n_kv = cache.num_kv_heads;
    const hd = cache.head_dim;
    const max_len = cache.max_seq_len;
    const pos = cache.curr_len;
    if (pos >= max_len) return error.ContextWindowExceeded;
    std.debug.assert(q_rot.len == num_heads * hd and k_rot.len == n_kv * hd and v.len == n_kv * hd);
    const groups = num_heads / n_kv;

    for (0..n_kv) |kv_h| {
        const dest = (kv_h * max_len + pos) * hd;
        @memcpy(cache.k[dest .. dest + hd], k_rot[kv_h * hd .. (kv_h + 1) * hd]);
        @memcpy(cache.v[dest .. dest + hd], v[kv_h * hd .. (kv_h + 1) * hd]);
    }
    cache.curr_len = pos + 1;
    const curr_len = cache.curr_len;

    const y_2d = try graph.zeros(&.{ 1, num_heads * hd }, false);
    const start_idx: usize = if (attn_type == .sliding_attention and curr_len > sliding_window)
        curr_len - sliding_window
    else
        0;
    const scores = try graph.arenaAllocator().alloc(f32, curr_len - start_idx);

    for (0..num_heads) |h| {
        const kv_base = (h / groups) * max_len;
        const q_head = q_rot[h * hd .. (h + 1) * hd];

        var max_score: f32 = -std.math.inf(f32);
        for (scores, start_idx..) |*s, j| {
            s.* = dotF32(q_head, cache.k[(kv_base + j) * hd .. (kv_base + j + 1) * hd]);
            max_score = @max(max_score, s.*);
        }

        var exp_sum: f32 = 0.0;
        for (scores) |*s| {
            s.* = @exp(s.* - max_score);
            exp_sum += s.*;
        }
        const inv_sum = 1.0 / exp_sum;

        const y_head = y_2d.data[h * hd .. (h + 1) * hd];
        for (scores, start_idx..) |s, j| {
            const weight = s * inv_sum;
            const v_cached = cache.v[(kv_base + j) * hd .. (kv_base + j + 1) * hd];
            for (y_head, v_cached) |*yv, vv| yv.* += weight * vv;
        }
    }
    return y_2d;
}

/// 校验词元 ID 均落在词表范围内，防止嵌入查表越界读取
pub fn validateTokenIds(tokens: []const usize, vocab_size: usize) !void {
    for (tokens) |tok| {
        if (tok >= vocab_size) return error.InvalidTokenId;
    }
}

// ============================================================================
// 采样共享核心：最终对数几率截断、重复惩罚与随机采样 (F32 / Q4 CPU 路径与 GPU 路径共用同一顺序)
// ============================================================================

/// 最终对数几率截断 (Final Logit Softcapping)：cap * tanh(x / cap)；cap <= 0 时不截断
pub fn softcapLogit(raw: f32, cap: f32) f32 {
    if (cap <= 0.0) return raw;
    return cap * std.math.tanh(raw / cap);
}

/// 采样使用的最终对数几率：先施加最终对数几率截断，再对最近出现过的词元施加重复惩罚，
/// 与 Hugging Face (logits 处理器作用于截断后的输出) 及 E4B GPU 路径的顺序一致
pub fn samplingLogit(raw: f32, cap: f32, penalized: bool, repetition_penalty: f32) f32 {
    const capped = softcapLogit(raw, cap);
    return if (penalized) sampling.applyRepetitionPenalty(capped, repetition_penalty) else capped;
}

/// 参与重复惩罚的最近词元窗口长度
pub const repetition_window: usize = 64;

/// 取最近 repetition_window 个词元作为重复惩罚窗口
pub fn recentWindow(tokens: []const usize) []const usize {
    return if (tokens.len > repetition_window) tokens[tokens.len - repetition_window ..] else tokens;
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
                const raw = if (Elem == f32) dotF32(ctx.h_last, row) else dotF32Bf16(ctx.h_last, row);
                const penalized = ctx.apply_penalty and (ctx.recent_bitset[v / 64] >> @intCast(v % 64)) & 1 != 0;
                if (penalized) {
                    const val = samplingLogit(raw, cfg.final_logit_softcapping, true, ctx.repetition_penalty);
                    if (val > best_penalized.val) best_penalized = .{ .id = v, .val = val };
                } else if (raw > best_plain.val) {
                    best_plain = .{ .id = v, .val = raw };
                }
            }
            if (best_plain.id != std.math.maxInt(usize)) best_plain.val = softcapLogit(best_plain.val, cfg.final_logit_softcapping);
            return best_plain.better(best_penalized);
        }

        fn reduceFn(a: BestToken, b: BestToken) BestToken {
            return a.better(b);
        }
    };

    const bitset_words = (cfg.vocab_size + 63) / 64;
    var recent_bitset: [bitset_words]u64 = @splat(0);
    const apply_penalty = sampling.penaltyActive(repetition_penalty);
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
    // 全部词元均被排除或得分为 NaN 时退回 0 号词元 (与 `sampling.argmaxWithPenalty` 一致)
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
                const raw = if (Elem == f32) dotF32(ctx.h_last, row) else dotF32Bf16(ctx.h_last, row);
                ctx.out[v] = softcapLogit(raw, cfg.final_logit_softcapping);
            }
        }
    };
    try graph.executor().parallelFor(cfg.vocab_size, 4096, Context{ .embed = embed, .h_last = h_last, .out = out }, Context.run);
}

/// CPU 词表投影并选取下一个词元：
/// - 贪婪采样走融合快车道 `greedyLmHeadCpu`，结果与此前的贪婪实现逐位一致；
/// - 随机采样先物化截断后的完整对数几率，再按 Hugging Face 顺序施加重复惩罚 (recent_tokens 中的词元全部受罚，窗口由调用方决定)、
///   排除 `cfg.sampling_excluded_tokens`，最后交给 `Sampler.sampleLogits` 完成温度、前 K 截断 (Top-K)、核采样 (Nucleus Sampling, Top-P) 与抽样。
pub fn sampleLmHeadCpu(
    comptime cfg: Gemma4Config,
    comptime Elem: type,
    graph: *autodiff.Graph,
    embed: []const Elem,
    h_last: []const f32,
    recent_tokens: []const usize,
    sampler: *Sampler,
) !usize {
    const penalty = sampler.options.repetition_penalty;
    if (sampler.isGreedy()) return greedyLmHeadCpu(cfg, Elem, graph, embed, h_last, recent_tokens, penalty);

    const arena = graph.arenaAllocator();
    const logits = try arena.alloc(f32, cfg.vocab_size);
    try lmHeadLogitsCpu(cfg, Elem, graph, embed, h_last, logits);
    try sampling.applyRepetitionPenaltyInPlace(arena, logits, recent_tokens, penalty);
    sampling.maskExcludedInPlace(logits, cfg.sampling_excluded_tokens);
    return sampler.sampleLogits(graph.executor(), arena, logits);
}

// ============================================================================
// Gemma 4 架构定义与模型实现
// ============================================================================

/// Gemma 4 注意力层类型枚举
pub const Gemma4AttentionType = enum {
    sliding_attention,
    full_attention,

    /// 旋转位置编码频率基数 (Rotary Position Embedding Theta, RoPE Theta)
    pub fn ropeTheta(self: Gemma4AttentionType) f32 {
        return switch (self) {
            .sliding_attention => 10000.0,
            .full_attention => 1000000.0,
        };
    }

    /// 部分旋转因子：全局注意力层采用比例旋转位置编码 (Proportional RoPE)，仅前 25% 的旋转角参与旋转
    pub fn partialRotaryFactor(self: Gemma4AttentionType) f32 {
        return switch (self) {
            .sliding_attention => 1.0,
            .full_attention => 0.25,
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
        return sampling.isExcluded(self.sampling_excluded_tokens, v);
    }

    /// 单个解码层的注意力几何规格
    pub const LayerSpec = struct {
        attn_type: Gemma4AttentionType,
        head_dim: usize,
        num_kv_heads: usize,
    };

    /// 每 6 层中最后一层为全局全注意力 (full_attention)，其余为滑动窗口注意力 (sliding_attention)
    pub fn layerSpec(self: Gemma4Config, layer_idx: usize) LayerSpec {
        const is_full = ((layer_idx + 1) % 6 == 0);
        return .{
            .attn_type = if (is_full) .full_attention else .sliding_attention,
            .head_dim = if (is_full) self.global_head_dim else self.head_dim,
            .num_kv_heads = if (is_full) self.num_global_key_value_heads else self.num_key_value_heads,
        };
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
        k: []f32,
        v: []f32,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        attn_type: Gemma4AttentionType,
        curr_len: usize = 0,
        max_seq_len: usize,

        pub fn init(
            allocator: std.mem.Allocator,
            attn_type: Gemma4AttentionType,
            num_kv_heads: usize,
            head_dim: usize,
            sliding_window: usize,
            max_seq_len: usize,
        ) !LayerCache {
            const numel = num_kv_heads * max_seq_len * head_dim;
            const k = try allocator.alloc(f32, numel);
            errdefer allocator.free(k);
            const v = try allocator.alloc(f32, numel);
            @memset(k, 0.0);
            @memset(v, 0.0);
            return LayerCache{
                .k = k,
                .v = v,
                .num_kv_heads = num_kv_heads,
                .head_dim = head_dim,
                .sliding_window = sliding_window,
                .attn_type = attn_type,
                .curr_len = 0,
                .max_seq_len = max_seq_len,
            };
        }

        pub fn deinit(self: *LayerCache, allocator: std.mem.Allocator) void {
            allocator.free(self.k);
            allocator.free(self.v);
        }

        pub fn reset(self: *LayerCache) void {
            self.curr_len = 0;
        }

        /// 注意力可见的缓存窗口长度：滑动窗口层只读取最近 sliding_window 个位置，全局注意力层读取全部 curr_len 个位置
        pub fn windowLen(self: *const LayerCache) usize {
            return switch (self.attn_type) {
                .sliding_attention => @min(self.curr_len, self.sliding_window),
                .full_attention => self.curr_len,
            };
        }

        /// 将可见窗口 [curr_len - windowLen, curr_len) 内的键值按 [num_kv_heads, windowLen, head_dim] 紧凑打包写入 out_k / out_v
        pub fn packWindow(self: *const LayerCache, out_k: []f32, out_v: []f32) !void {
            const win = self.windowLen();
            const hd = self.head_dim;
            const n = self.num_kv_heads * win * hd;
            if (out_k.len != n or out_v.len != n) return error.ShapeMismatch;
            const start = self.curr_len - win;
            for (0..self.num_kv_heads) |kv_h| {
                const src = (kv_h * self.max_seq_len + start) * hd;
                const dst = kv_h * win * hd;
                @memcpy(out_k[dst..][0 .. win * hd], self.k[src..][0 .. win * hd]);
                @memcpy(out_v[dst..][0 .. win * hd], self.v[src..][0 .. win * hd]);
            }
        }

        /// 将 [num_kv_heads, win_len, head_dim] 紧凑键值写入位置 [len - win_len, len) 并把游标设为 len。
        /// 窗口之前的位置保持原样：滑动窗口层此后的增量解码不会再读取它们，全局注意力层要求 win_len == len。
        pub fn writeWindow(self: *LayerCache, packed_k: []const f32, packed_v: []const f32, win_len: usize, len: usize) !void {
            if (len > self.max_seq_len) return error.ContextWindowExceeded;
            const required = switch (self.attn_type) {
                .sliding_attention => @min(len, self.sliding_window),
                .full_attention => len,
            };
            if (win_len > len or win_len < required) return error.ShapeMismatch;
            const hd = self.head_dim;
            const n = self.num_kv_heads * win_len * hd;
            if (packed_k.len != n or packed_v.len != n) return error.ShapeMismatch;
            const start = len - win_len;
            for (0..self.num_kv_heads) |kv_h| {
                const dst = (kv_h * self.max_seq_len + start) * hd;
                const src = kv_h * win_len * hd;
                @memcpy(self.k[dst..][0 .. win_len * hd], packed_k[src..][0 .. win_len * hd]);
                @memcpy(self.v[dst..][0 .. win_len * hd], packed_v[src..][0 .. win_len * hd]);
            }
            self.curr_len = len;
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

/// Gemma RMSNorm (with learnable scale)
pub const GemmaRMSNorm = struct {
    weight: *Tensor, // [dim]
    eps: f32,
    name: ?[]const u8 = null,
    module_type: []const u8 = "GemmaRMSNorm",

    pub fn init(allocator: std.mem.Allocator, dim: usize, eps: f32) !GemmaRMSNorm {
        const weight = try createPersistentTensor(allocator, 1, dim, true);
        errdefer freePersistentTensor(allocator, weight);
        @memset(weight.data, 1.0);
        weight.shape = Shape.init(&.{dim});
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        return GemmaRMSNorm{
            .weight = weight,
            .eps = eps,
        };
    }

    pub const formula = "y = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum x_i^2 + \\epsilon}} \\odot w";

    pub fn forward(self: *const GemmaRMSNorm, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        return try graph.rmsNorm(x, self.weight, self.eps);
    }
};

/// Gemma 无权重缩放均方根层归一化 (GemmaUnscaledRMSNorm)
/// 数学公式：y = RMSNorm(x, eps) = x / sqrt(mean(x^2) + eps)
/// 专用于 Gemma 4 的 Value states 预注意力归一化 (v_norm, with_scale=False)
pub const GemmaUnscaledRMSNorm = struct {
    weight: *Tensor,
    eps: f32,
    name: ?[]const u8 = null,
    module_type: []const u8 = "GemmaUnscaledRMSNorm",

    pub fn init(allocator: std.mem.Allocator, dim: usize, eps: f32) !GemmaUnscaledRMSNorm {
        const weight = try createPersistentTensor(allocator, 1, dim, false);
        errdefer freePersistentTensor(allocator, weight);
        @memset(weight.data, 1.0);
        weight.shape = Shape.init(&.{dim});
        weight.strides = tensor.computeContiguousStrides(weight.shape);
        // 恒为 1 的常量权重注册为缓冲区，解冻参数时不会被误转为可训练参数
        weight.is_buffer = true;

        return GemmaUnscaledRMSNorm{
            .weight = weight,
            .eps = eps,
        };
    }

    pub fn deinit(self: *GemmaUnscaledRMSNorm, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
    }

    pub const formula = "y = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum x_i^2 + \\epsilon}}";

    pub fn forward(self: *const GemmaUnscaledRMSNorm, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        return try graph.rmsNorm(x, self.weight, self.eps);
    }
};

/// Gemma 4 门控前馈多层感知机 (Gemma4MLP: gate_proj, up_proj, down_proj)
/// 结构：(GELU_tanh(x * W_gate) * (x * W_up)) * W_down
pub const Gemma4MLP = struct {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    name: ?[]const u8 = null,
    module_type: []const u8 = "Gemma4MLP",

    pub fn init(allocator: std.mem.Allocator, hidden_size: usize, intermediate_size: usize) !Gemma4MLP {
        const gate_proj = try Linear.init(allocator, hidden_size, intermediate_size);
        errdefer deinitModel(&gate_proj, allocator);
        const up_proj = try Linear.init(allocator, hidden_size, intermediate_size);
        errdefer deinitModel(&up_proj, allocator);
        const down_proj = try Linear.init(allocator, intermediate_size, hidden_size);
        errdefer deinitModel(&down_proj, allocator);

        return Gemma4MLP{
            .gate_proj = gate_proj,
            .up_proj = up_proj,
            .down_proj = down_proj,
        };
    }

    pub const formula = "y = (\\text{GELU}(x W_{\\text{gate}}) \\odot (x W_{\\text{up}})) W_{\\text{down}}";

    pub fn forward(self: *const Gemma4MLP, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        var x_2d = x;
        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            x_2d = try graph.reshape(x, &.{ B * T, D });
        }

        const gate = try self.gate_proj.forward(graph, x_2d);
        const up = try self.up_proj.forward(graph, x_2d);
        const gelu_gate = try graph.gelu(gate);
        const hidden = try graph.mul(gelu_gate, up);
        const out = try self.down_proj.forward(graph, hidden);

        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            return try graph.reshape(out, &.{ B, T, D });
        }
        return out;
    }
};

/// Gemma 4 混合注意力机制 (Gemma4Attention: 支持滑动窗口注意力和全局注意力)
pub const Gemma4Attention = struct {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: ?Linear,
    o_proj: Linear,
    q_norm: GemmaRMSNorm,
    k_norm: GemmaRMSNorm,
    v_norm: GemmaUnscaledRMSNorm,

    attn_type: Gemma4AttentionType,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    rope_theta: f32,
    partial_rotary_factor: f32,

    name: ?[]const u8 = null,
    module_type: []const u8 = "Gemma4Attention",

    pub fn init(
        allocator: std.mem.Allocator,
        attn_type: Gemma4AttentionType,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        rms_norm_eps: f32,
    ) !Gemma4Attention {
        const q_dim = num_heads * head_dim;
        const kv_dim = num_kv_heads * head_dim;

        const q_proj = try Linear.init(allocator, hidden_size, q_dim);
        errdefer deinitModel(&q_proj, allocator);
        const k_proj = try Linear.init(allocator, hidden_size, kv_dim);
        errdefer deinitModel(&k_proj, allocator);
        const v_proj = if (attn_type == .full_attention) null else try Linear.init(allocator, hidden_size, kv_dim);
        errdefer if (v_proj) |*vp| deinitModel(vp, allocator);
        const o_proj = try Linear.init(allocator, q_dim, hidden_size);
        errdefer deinitModel(&o_proj, allocator);

        const q_norm = try GemmaRMSNorm.init(allocator, head_dim, rms_norm_eps);
        errdefer deinitModel(&q_norm, allocator);
        const k_norm = try GemmaRMSNorm.init(allocator, head_dim, rms_norm_eps);
        errdefer deinitModel(&k_norm, allocator);
        const v_norm = try GemmaUnscaledRMSNorm.init(allocator, head_dim, rms_norm_eps);
        errdefer deinitModel(&v_norm, allocator);

        return Gemma4Attention{
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
            .rope_theta = attn_type.ropeTheta(),
            .partial_rotary_factor = attn_type.partialRotaryFactor(),
        };
    }

    pub const formula = "\\text{Attn}(Q, K, V) = \\text{softmax}\\left(Q K^T + M\\right) V W_o";

    pub fn forward(self: *const Gemma4Attention, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return self.forwardPrefillOptional(graph, x, null);
    }

    pub fn forwardPrefill(
        self: *const Gemma4Attention,
        graph: *autodiff.Graph,
        x: *Tensor,
        layer_cache: *Gemma4KVCache.LayerCache,
    ) !*Tensor {
        return self.forwardPrefillOptional(graph, x, layer_cache);
    }

    fn forwardPrefillOptional(
        self: *const Gemma4Attention,
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

        // 1. 投影 Q, K, V (在 full_attention 下 attention_k_eq_v 为 true，V 与 K 共享投影权重)
        const q_raw = try self.q_proj.forward(graph, x_2d); // [B*T, nh * hd]
        const k_raw = try self.k_proj.forward(graph, x_2d); // [B*T, n_kv * hd]
        const v_raw = if (self.v_proj) |*vp| try vp.forward(graph, x_2d) else k_raw;

        // 2. 对每个头应用 Q-Norm 与 K-Norm，以及无参数 v_norm
        const q_heads = try graph.reshape(q_raw, &.{ B * T * nh, hd });
        const q_normed = try self.q_norm.forward(graph, q_heads);
        const k_heads = try graph.reshape(k_raw, &.{ B * T * n_kv, hd });
        const k_normed = try self.k_norm.forward(graph, k_heads);
        const v_heads = try graph.reshape(v_raw, &.{ B * T * n_kv, hd });
        const v_normed = try self.v_norm.forward(graph, v_heads);

        // 3. 转置为四维头张量: [B, nh, T, hd] 与 [B, n_kv, T, hd]
        const q_4d = try graph.reshape(q_normed, &.{ B, T, nh, hd });
        const k_4d = try graph.reshape(k_normed, &.{ B, T, n_kv, hd });
        const v_4d = try graph.reshape(v_normed, &.{ B, T, n_kv, hd });

        const q = try graph.transposeND(q_4d, 1, 2);
        const k_t_unrot = try graph.transposeND(k_4d, 1, 2);
        const v = try graph.transposeND(v_4d, 1, 2);

        // 4. 施加半切分 RoPE 旋转位置编码 (Split-Half RoPE, Gemma 官方标准)
        const q_rot = try ropeSplitHalf(graph, q, 0, self.partial_rotary_factor, self.rope_theta);
        const k_rot = try ropeSplitHalf(graph, k_t_unrot, 0, self.partial_rotary_factor, self.rope_theta);

        // 5. 写入键值缓存 (Key-Value Cache, KVCache)
        if (layer_cache) |cache| try writePrefillKVCache(cache, k_rot.data, v.data, B, T);

        // 6. GQA 广播扩展至 nh 个头
        var k = k_rot;
        var v_final = v;
        if (groups > 1) {
            k = try graph.repeatKV(k_rot, groups);
            v_final = try graph.repeatKV(v, groups);
        }

        // 7. 注意力得分与因果 / 滑动窗口掩码计算 (官方实现中 scaling = 1.0)
        const y_4d = try prefillAttention(graph, self.attn_type, self.sliding_window, q_rot, k, v_final);

        // 8. 转置并融合输出
        const y_trans = try graph.transposeND(y_4d, 1, 2);
        const y_2d = try graph.reshape(y_trans, &.{ B * T, nh * hd });
        const out_2d = try self.o_proj.forward(graph, y_2d);

        return try graph.reshape(out_2d, &.{ B, T, C });
    }

    pub fn forwardDecode(
        self: *const Gemma4Attention,
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
        const pos = layer_cache.curr_len;
        if (pos >= layer_cache.max_seq_len) return error.ContextWindowExceeded;

        const x_2d = if (x.shape.len == 2) x else try graph.reshape(x, &.{ 1, C });

        const q_raw = try self.q_proj.forward(graph, x_2d);
        const k_raw = try self.k_proj.forward(graph, x_2d);
        const v_raw = if (self.v_proj) |*vp| try vp.forward(graph, x_2d) else k_raw;

        const q_heads = try graph.reshape(q_raw, &.{ nh, hd });
        const q_normed = try self.q_norm.forward(graph, q_heads);
        const k_heads = try graph.reshape(k_raw, &.{ n_kv, hd });
        const k_normed = try self.k_norm.forward(graph, k_heads);
        const v_heads = try graph.reshape(v_raw, &.{ n_kv, hd });
        const v_normed = try self.v_norm.forward(graph, v_heads);

        const q_4d = try graph.reshape(q_normed, &.{ 1, nh, 1, hd });
        const k_4d = try graph.reshape(k_normed, &.{ 1, n_kv, 1, hd });
        const q_rot = try ropeSplitHalf(graph, q_4d, pos, self.partial_rotary_factor, self.rope_theta);
        const k_rot = try ropeSplitHalf(graph, k_4d, pos, self.partial_rotary_factor, self.rope_theta);

        const y_2d = try decodeAttention(graph, layer_cache, self.attn_type, self.sliding_window, nh, q_rot.data, k_rot.data, v_normed.data);

        const out_2d = try self.o_proj.forward(graph, y_2d);
        return try graph.reshape(out_2d, &.{ 1, 1, C });
    }
};

/// Gemma 4 变换器层 (Gemma4DecoderLayer)
pub const Gemma4DecoderLayer = struct {
    input_layernorm: GemmaRMSNorm,
    self_attn: Gemma4Attention,
    post_attention_layernorm: GemmaRMSNorm,
    pre_feedforward_layernorm: GemmaRMSNorm,
    mlp: Gemma4MLP,
    post_feedforward_layernorm: GemmaRMSNorm,
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
    ) !Gemma4DecoderLayer {
        const input_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&input_layernorm, allocator);

        const self_attn = try Gemma4Attention.init(
            allocator,
            attn_type,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            rms_norm_eps,
        );
        errdefer deinitModel(&self_attn, allocator);

        const post_attention_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&post_attention_layernorm, allocator);

        const pre_feedforward_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&pre_feedforward_layernorm, allocator);

        const mlp = try Gemma4MLP.init(allocator, hidden_size, intermediate_size);
        errdefer deinitModel(&mlp, allocator);

        const post_feedforward_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&post_feedforward_layernorm, allocator);

        const layer_scalar = try createLayerScalar(allocator);
        errdefer freePersistentTensor(allocator, layer_scalar);

        return Gemma4DecoderLayer{
            .input_layernorm = input_layernorm,
            .self_attn = self_attn,
            .post_attention_layernorm = post_attention_layernorm,
            .pre_feedforward_layernorm = pre_feedforward_layernorm,
            .mlp = mlp,
            .post_feedforward_layernorm = post_feedforward_layernorm,
            .layer_scalar = layer_scalar,
        };
    }

    pub const formula = "\\begin{aligned} x_1 &= x + \\text{PostAttnLN}(\\text{Attn}(\\text{InputLN}(x))) \\\\ x_2 &= s \\cdot \\left(x_1 + \\text{PostMLPLN}(\\text{MLP}(\\text{PreMLPLN}(x_1)))\\right) \\end{aligned}";

    /// 注意力之后的公共残差尾部：PostAttnLN -> + x -> PreMLPLN -> MLP -> PostMLPLN -> + x1 -> * layer_scalar
    /// 训练前向、预填充与增量解码三条路径共用，保证层缩放标量 (layer_scalar) 处处一致地生效
    fn residualTail(self: *const Gemma4DecoderLayer, graph: *autodiff.Graph, x: *Tensor, attn_out: *Tensor) !*Tensor {
        const post_attn = try self.post_attention_layernorm.forward(graph, attn_out);
        const x1 = try graph.add(x, post_attn);

        const pre_mlp = try self.pre_feedforward_layernorm.forward(graph, x1);
        const mlp_out = try self.mlp.forward(graph, pre_mlp);
        const post_mlp = try self.post_feedforward_layernorm.forward(graph, mlp_out);
        const x2 = try graph.add(x1, post_mlp);
        return try graph.mulScalar(x2, self.layer_scalar.data[0]);
    }

    pub fn forward(self: *const Gemma4DecoderLayer, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const in_norm = try self.input_layernorm.forward(graph, x);
        const attn_out = try self.self_attn.forward(graph, in_norm);
        return self.residualTail(graph, x, attn_out);
    }

    pub fn forwardPrefill(
        self: *const Gemma4DecoderLayer,
        graph: *autodiff.Graph,
        x: *Tensor,
        layer_cache: *Gemma4KVCache.LayerCache,
    ) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const in_norm = try self.input_layernorm.forward(graph, x);
        const attn_out = try self.self_attn.forwardPrefill(graph, in_norm, layer_cache);
        return self.residualTail(graph, x, attn_out);
    }

    pub fn forwardDecode(
        self: *const Gemma4DecoderLayer,
        graph: *autodiff.Graph,
        x: *Tensor,
        layer_cache: *Gemma4KVCache.LayerCache,
    ) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const in_norm = try self.input_layernorm.forward(graph, x);
        const attn_out = try self.self_attn.forwardDecode(graph, in_norm, layer_cache);
        return self.residualTail(graph, x, attn_out);
    }
};

/// 创建层缩放标量 (layer_scalar, 初值 1.0)：与官方实现一致注册为不可训练缓冲区 (buffer)，
/// 避免被优化器当作参数参与权重衰减
pub fn createLayerScalar(allocator: std.mem.Allocator) !*Tensor {
    const layer_scalar = try createPersistentTensor(allocator, 1, 1, false);
    layer_scalar.data[0] = 1.0;
    layer_scalar.shape = Shape.init(&.{1});
    layer_scalar.strides = tensor.computeContiguousStrides(layer_scalar.shape);
    layer_scalar.is_buffer = true;
    return layer_scalar;
}

/// Gemma 4 语言模型核心架构 (Gemma4ForCausalLM)
pub fn Gemma4ForCausalLM(comptime cfg: Gemma4Config) type {
    return struct {
        const Self = @This();
        pub const config: Gemma4Config = cfg;

        embed_tokens: Embedding,
        layers: []Gemma4DecoderLayer,
        norm: GemmaRMSNorm,
        name: ?[]const u8 = null,
        module_type: []const u8 = "Gemma4ForCausalLM",

        fn releaseModuleGrads(module_ptr: anytype, allocator: std.mem.Allocator) void {
            const Visitor = struct {
                alloc: std.mem.Allocator,
                pub fn visitTensor(self: *@This(), _: []const u8, t: *Tensor) !void {
                    if (t.grad.len > 0) {
                        self.alloc.free(t.grad);
                        t.grad = &.{};
                    }
                    t.requires_grad = false;
                }
            };
            var v = Visitor{ .alloc = allocator };
            nn.walk(module_ptr, &v) catch {};
        }

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
                );
                if (!keep_grad) releaseModuleGrads(&layers[i], allocator);
                initialized_layers += 1;
            }

            var norm = try GemmaRMSNorm.init(allocator, config.hidden_size, config.rms_norm_eps);
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

            // 5. Logit Softcapping: 30 * tanh(logits / 30)
            const cap = config.final_logit_softcapping;
            const scaled_logits = try graph.mulScalar(logits_raw_2d, 1.0 / cap);
            const tanh_logits = try graph.tanh(scaled_logits);
            const capped_logits_2d = try graph.mulScalar(tanh_logits, cap);

            return try graph.reshape(capped_logits_2d, &.{ B, T, config.vocab_size });
        }

        /// 对最终隐状态 h_last 选取下一个词元 (仅最近 repetition_window 个词元受重复惩罚)
        fn sampleNextTokenF32(
            self: *const Self,
            h_last: []const f32,
            recent_tokens: []const usize,
            sampler: *Sampler,
            graph: *autodiff.Graph,
        ) !usize {
            return sampleLmHeadCpu(config, f32, graph, self.embed_tokens.weight.data, h_last, recentWindow(recent_tokens), sampler);
        }

        /// 词元嵌入查表并施加 Gemma 嵌入缩放 sqrt(hidden_size)，返回 [1, T, hidden_size]
        fn embedScaled(self: *const Self, graph: *autodiff.Graph, tokens: []const usize) !*Tensor {
            const H = config.hidden_size;
            const tok_emb = try graph.tensorND(&.{ 1, tokens.len, H }, false);
            const sqrt_d: f32 = @sqrt(@as(f32, @floatFromInt(H)));
            const embed_f32 = self.embed_tokens.weight.data;
            for (tokens, 0..) |tok, t_idx| {
                const emb_row = embed_f32[tok * H .. (tok + 1) * H];
                for (tok_emb.data[t_idx * H .. (t_idx + 1) * H], emb_row) |*dst, e| dst.* = e * sqrt_d;
            }
            return tok_emb;
        }

        /// 全精度 (F32) 预填充阶段 (Prefill)：处理 Prompt 词元序列并写入 KVCache，按 sampler 选取并返回首个预测词元
        pub fn prefill(
            self: *const Self,
            graph: *autodiff.Graph,
            prompt_tokens: []const usize,
            cache: *Gemma4KVCache,
            sampler: *Sampler,
        ) !usize {
            const T = prompt_tokens.len;
            if (T == 0) return error.EmptyPrompt;
            try validateTokenIds(prompt_tokens, config.vocab_size);
            cache.reset();

            const tok_emb = try self.embedScaled(graph, prompt_tokens);

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
            sampler: *Sampler,
        ) !usize {
            const tokens = [_]usize{token};
            try validateTokenIds(&tokens, config.vocab_size);
            const tok_emb = try self.embedScaled(graph, &tokens);

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
};

/// 基于上下文 N-gram 匹配的投机采样候选生成器 (Prompt Lookup Decoding / N-gram Speculative Sampling)
/// 在历史序列（Prompt 上下文 + 已生成词元）中反向查找当前尾部匹配的 N-gram，
/// 提取历史上紧随其后的后续词元作为推测候选序列 (Draft Candidates)。
/// 该算法零额外参数模型负担，完全基于自然语言/代码的局部重复性，与大模型一次性并行验证形成吞吐放大。
pub fn promptLookup(
    tokens: []const usize,
    min_n: usize,
    max_n: usize,
    max_speculate: usize,
) ?[]const usize {
    // 空模式 (N = 0) 会平凡匹配任意位置，最小匹配长度至少为 1
    const lo = @max(min_n, 1);
    if (tokens.len < lo + 1) return null;
    const cur_len = tokens.len;

    var n = @min(max_n, cur_len - 1);
    while (n >= lo) : (n -= 1) {
        const pattern = tokens[cur_len - n .. cur_len];
        var search_idx: usize = cur_len - n;
        while (search_idx > 0) {
            search_idx -= 1;
            if (std.mem.eql(usize, tokens[search_idx .. search_idx + n], pattern)) {
                const draft_start = search_idx + n;
                if (draft_start < cur_len) {
                    const draft_end = @min(draft_start + max_speculate, cur_len);
                    if (draft_end > draft_start) {
                        return tokens[draft_start..draft_end];
                    }
                }
            }
        }
    }
    return null;
}

/// 投机采样推测草稿引擎 (Speculative Draft Engine)
/// 结合自适应多尺度 N-gram 历史回溯匹配与高频词元马尔可夫转移缓存 (Markov Transition Cache)
pub const SpeculativeDraftEngine = struct {
    allocator: std.mem.Allocator,
    transitions: std.AutoHashMapUnmanaged(usize, usize),

    pub fn init(allocator: std.mem.Allocator) SpeculativeDraftEngine {
        return .{
            .allocator = allocator,
            .transitions = .{},
        };
    }

    pub fn deinit(self: *SpeculativeDraftEngine) void {
        self.transitions.deinit(self.allocator);
    }

    /// 根据 Prompt 与已知词元预填充转移图
    pub fn seedFromTokens(self: *SpeculativeDraftEngine, tokens: []const usize) !void {
        if (tokens.len < 2) return;
        for (0..tokens.len - 1) |i| {
            try self.transitions.put(self.allocator, tokens[i], tokens[i + 1]);
        }
    }

    /// 记录新生成的词元转移
    pub fn recordTransition(self: *SpeculativeDraftEngine, from: usize, to: usize) !void {
        try self.transitions.put(self.allocator, from, to);
    }

    /// 预注入通用语言与编程高频共现搭配 (Collocations)
    pub fn seedCommonCollocations(self: *SpeculativeDraftEngine, vocab: *const Gemma4Vocabulary) !void {
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
                    try self.transitions.put(self.allocator, t1, t2);
                }
            }
        }
    }

    /// 提出推测草稿候选词元 (Propose Draft Candidates)
    pub fn proposeDraft(
        self: *const SpeculativeDraftEngine,
        current_tokens: []const usize,
        out_buf: []usize,
    ) ?[]const usize {
        if (current_tokens.len == 0 or out_buf.len == 0) return null;

        // 1. 优先使用 N-gram 历史回溯匹配 (N=4, 3, 2)
        if (promptLookup(current_tokens, 2, 4, out_buf.len)) |draft| {
            return draft;
        }

        // 2. 马尔可夫多阶转移预测 (Token-to-Token Markov Chain)
        const cur_tok = current_tokens[current_tokens.len - 1];
        var count: usize = 0;
        var probe = cur_tok;
        while (count < out_buf.len) {
            if (self.transitions.get(probe)) |nxt| {
                if (nxt == probe or nxt == Gemma4Vocabulary.EOS or nxt == Gemma4Vocabulary.EOT) break;
                out_buf[count] = nxt;
                count += 1;
                probe = nxt;
            } else {
                break;
            }
        }
        if (count > 0) {
            return out_buf[0..count];
        }
        return null;
    }
};
