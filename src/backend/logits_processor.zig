//! 设备端对数几率处理与候选词元选择 (Device Logits Processing & Candidate Selection)
//!
//! 提供模型无关的 GPU 端对数几率截断 (Logits Softcapping)、重复惩罚 (Repetition Penalty)、
//! 词元排除掩码 (Token Exclusion Mask)、候选生成与抽样终结 (Sampling)。
//! 遵循分层规范，位于 backend 包（层级 0），彻底解耦模型架构与模型专用配置结构体。

const std = @import("std");
const mlx = @import("mlx.zig");
const sampling = @import("sampling.zig");
const offload = @import("offload.zig");

pub const Stream = mlx.Stream;
pub const TokenExclusionMask = offload.TokenExclusionMask;
pub const LmHead = offload.LmHead;
pub const LmHeadWeights = offload.LmHeadWeights;
pub const QuantWeights = offload.QuantWeights;
pub const Sampler = sampling.Sampler;
pub const SamplingOptions = sampling.SamplingOptions;

/// 对数几率处理器配置
pub const LogitsProcessorOptions = struct {
    vocab_size: usize,
    final_logit_softcapping: f32 = 0.0,
    sampling_excluded_tokens: []const sampling.TokenRange = &.{},

    pub fn defaultOptions(vocab_size: usize) LogitsProcessorOptions {
        return .{
            .vocab_size = vocab_size,
            .final_logit_softcapping = 0.0,
            .sampling_excluded_tokens = &.{},
        };
    }
};

/// 将 GPU 对数几率转换为单精度并施加最终对数几率截断 (Final Logit Softcapping)：cap * tanh(x / cap)；cap <= 0 时只做类型转换
pub fn softcapLogits(logits: mlx.Array, cap: f32, s: ?Stream) !mlx.Array {
    var logits_f32 = try logits.astype(.float32, s);
    if (cap <= 0.0) return logits_f32;
    defer logits_f32.deinit();
    var scaled = try logits_f32.mulScalar(1.0 / cap, s);
    defer scaled.deinit();
    var tanh_l = try mlx.Array.tanh(scaled, s);
    defer tanh_l.deinit();
    return tanh_l.mulScalar(cap, s);
}

/// 设备端重复惩罚 (Repetition Penalty)：在 GPU 上对 logits [1, vocab_size] (单精度，已截断) 中 recent_tokens 的位置
/// 取值、按 Hugging Face 规则惩罚 (正值除以系数，非正值乘以系数) 后写回，返回新数组；
/// 不需要惩罚 (系数不大于 1 或没有落在词表内的最近词元) 时返回 null。
/// 重复出现的词元写回同一个惩罚值，因此恰好惩罚一次。
pub fn penalizeDevice(
    allocator: std.mem.Allocator,
    logits: mlx.Array,
    vocab_size: usize,
    recent_tokens: []const usize,
    repetition_penalty: f32,
    s: ?Stream,
) !?mlx.Array {
    if (!sampling.penaltyActive(repetition_penalty)) return null;
    var count: usize = 0;
    for (recent_tokens) |tok| count += @intFromBool(tok < vocab_size);
    if (count == 0) return null;

    const ids = try allocator.alloc(i32, count);
    defer allocator.free(ids);
    var n: usize = 0;
    for (recent_tokens) |tok| {
        if (tok >= vocab_size) continue;
        ids[n] = @intCast(tok);
        n += 1;
    }
    var idx = try mlx.Array.fromSliceI32(ids, &.{ 1, count });
    defer idx.deinit();
    var gathered = try logits.takeAlongAxis(idx, -1, s);
    defer gathered.deinit();
    var shrunk = try gathered.mulScalar(1.0 / repetition_penalty, s);
    defer shrunk.deinit();
    var grown = try gathered.mulScalar(repetition_penalty, s);
    defer grown.deinit();
    var zero = try mlx.Array.fromScalarF32(0.0);
    defer zero.deinit();
    var positive = try mlx.Array.greater(gathered, zero, s);
    defer positive.deinit();
    var penalized = try mlx.Array.where(positive, shrunk, grown, s);
    defer penalized.deinit();
    return try logits.putAlongAxis(idx, penalized, -1, s);
}

/// 设备端随机采样候选 (尚未求值的惰性计算图)：概率最高的 k 个词元 ID 与其已处理对数几率，
/// 以及按需计算的完整词表对数配分函数 (只启用核采样时)
pub const DeviceCandidates = struct {
    /// [1, k] uint32
    ids: mlx.Array,
    /// [1, k] float32 (截断、重复惩罚与排除之后，尚未除以温度)
    logits: mlx.Array,
    /// [1] float32：log Σ exp(logit / temperature)
    log_normalizer: ?mlx.Array = null,

    pub fn deinit(self: *DeviceCandidates) void {
        self.ids.deinit();
        self.logits.deinit();
        if (self.log_normalizer) |*a| a.deinit();
        self.log_normalizer = null;
    }

    /// 候选数 k
    pub fn len(self: *const DeviceCandidates) usize {
        return self.ids.dimSize(1);
    }

    /// 把全部惰性数组写入 out (至少 3 个槽位)，返回实际数组个数，供多组候选一次性批量求值
    pub fn arrays(self: *const DeviceCandidates, out: []mlx.Array) usize {
        out[0] = self.ids;
        out[1] = self.logits;
        if (self.log_normalizer) |a| {
            out[2] = a;
            return 3;
        }
        return 2;
    }

    /// 读回主机候选集 (触发求值；只传输 k 对 (ID, 对数几率) 与一个标量)，写入 out[0..k]
    pub fn read(self: *const DeviceCandidates, out: []sampling.Candidate) !HostCandidates {
        const ids = try self.ids.dataUint32();
        const vals = try self.logits.dataFloat32();
        if (ids.len != vals.len or ids.len > out.len) return error.ShapeMismatch;
        for (out[0..ids.len], ids, vals) |*cand, id, v| cand.* = .{ .id = id, .logit = v };
        const lse: ?f32 = if (self.log_normalizer) |a| (try a.dataFloat32())[0] else null;
        return .{ .candidates = out[0..ids.len], .log_normalizer = lse };
    }
};

/// 读回主机的候选集
pub const HostCandidates = struct {
    candidates: []sampling.Candidate,
    log_normalizer: ?f32,
};

/// 在 GPU 上为单行对数几率 [1, vocab_size] 构建随机采样候选的计算图：
/// 最终对数几率截断 → 重复惩罚 → 叠加排除掩码 → 部分排序选出 Top-K 候选
pub fn deviceCandidates(
    allocator: std.mem.Allocator,
    logits: mlx.Array,
    opts: LogitsProcessorOptions,
    mask: *const TokenExclusionMask,
    recent_tokens: []const usize,
    sampling_opts: SamplingOptions,
    s: ?Stream,
) !DeviceCandidates {
    if (logits.ndim() != 2 or logits.dimSize(0) != 1 or logits.dimSize(1) != opts.vocab_size) return error.ShapeMismatch;
    var capped = try softcapLogits(logits, opts.final_logit_softcapping, s);
    defer capped.deinit();
    var penalized_opt = try penalizeDevice(allocator, capped, opts.vocab_size, recent_tokens, sampling_opts.repetition_penalty, s);
    defer if (penalized_opt) |*a| a.deinit();
    var masked = try mask.apply(penalized_opt orelse capped, s);
    defer masked.deinit();

    const k = sampling_opts.candidateCount(opts.vocab_size);
    var neg = try masked.negative(s);
    defer neg.deinit();
    var order = try neg.argpartitionAxis(k - 1, -1, s);
    defer order.deinit();
    var top_view = try order.sliceAxis(1, 0, k, s);
    defer top_view.deinit();
    var ids = try top_view.contiguous(s);
    errdefer ids.deinit();
    var vals = try masked.takeAlongAxis(ids, -1, s);
    errdefer vals.deinit();

    var lse: ?mlx.Array = null;
    if (sampling_opts.needsLogNormalizer()) {
        var scaled = try masked.mulScalar(1.0 / sampling_opts.temperature, s);
        defer scaled.deinit();
        lse = try scaled.logsumexpAxis(-1, false, s);
    }
    return .{ .ids = ids, .logits = vals, .log_normalizer = lse };
}

/// 从单行 GPU 对数几率 [1, vocab_size] 中选取下一个词元：
/// - 纯贪婪采样 (temperature=0, penalty=1.0) 直接通过 mask.argmax 在 GPU 上完成
/// - 带重复惩罚的贪婪采样在 GPU 上完成截断后，在 CPU 端施加惩罚并贪婪选取
/// - 随机采样在 GPU 上完成截断、惩罚、排除与候选生成，只回传 k 个候选由 Sampler 终结
pub fn selectToken(
    allocator: std.mem.Allocator,
    logits: mlx.Array,
    opts: LogitsProcessorOptions,
    mask: *const TokenExclusionMask,
    recent_tokens: []const usize,
    sampler: *Sampler,
    s: ?Stream,
) !usize {
    const s_opts = sampler.options;
    if (s_opts.isPlainGreedy()) {
        var best = try mask.argmax(logits, s);
        defer best.deinit();
        return @intCast(try best.itemUint32());
    }
    if (s_opts.isGreedy()) {
        var capped = try softcapLogits(logits, opts.final_logit_softcapping, s);
        defer capped.deinit();
        const data = try capped.dataFloat32();
        if (data.len != opts.vocab_size) return error.ShapeMismatch;
        return sampling.argmaxWithPenalty(data, opts.vocab_size, opts.sampling_excluded_tokens, recent_tokens, s_opts.repetition_penalty);
    }

    var pending = try deviceCandidates(allocator, logits, opts, mask, recent_tokens, s_opts, s);
    defer pending.deinit();
    const buf = try allocator.alloc(sampling.Candidate, pending.len());
    defer allocator.free(buf);
    const host = try pending.read(buf);
    return sampler.sampleCandidates(host.candidates, host.log_normalizer);
}

/// 以语言模型词表输出头 (LmHead) 对最终隐状态 h_last [hidden_size] 打分并选取下一个词元
pub fn selectTokenFromHidden(
    allocator: std.mem.Allocator,
    lm_head: *const LmHead,
    opts: LogitsProcessorOptions,
    mask: *const TokenExclusionMask,
    h_last: []const f32,
    recent_tokens: []const usize,
    sampler: *Sampler,
    s: ?Stream,
) !usize {
    const hid = lm_head.hiddenSize();
    const voc = lm_head.vocabSize();
    if (h_last.len != hid or voc != opts.vocab_size) return error.ShapeMismatch;
    var x = try mlx.Array.fromSlice(h_last, &.{ 1, hid });
    defer x.deinit();
    var logits = try lm_head.forwardArray(x, s);
    defer logits.deinit();
    return selectToken(allocator, logits, opts, mask, recent_tokens, sampler, s);
}

/// 具名聚合对数几率处理器
pub const LogitsProcessor = struct {
    options: LogitsProcessorOptions,
    mask: TokenExclusionMask,

    pub fn init(allocator: std.mem.Allocator, options: LogitsProcessorOptions) !LogitsProcessor {
        const mask = try TokenExclusionMask.init(allocator, options.vocab_size, options.sampling_excluded_tokens);
        return .{
            .options = options,
            .mask = mask,
        };
    }

    pub fn deinit(self: *LogitsProcessor) void {
        self.mask.deinit();
    }

    pub fn selectToken(
        self: *const LogitsProcessor,
        allocator: std.mem.Allocator,
        logits: mlx.Array,
        recent_tokens: []const usize,
        sampler: *Sampler,
        s: ?Stream,
    ) !usize {
        return @import("logits_processor.zig").selectToken(allocator, logits, self.options, &self.mask, recent_tokens, sampler, s);
    }

    pub fn selectTokenFromHidden(
        self: *const LogitsProcessor,
        allocator: std.mem.Allocator,
        lm_head: *const LmHead,
        h_last: []const f32,
        recent_tokens: []const usize,
        sampler: *Sampler,
        s: ?Stream,
    ) !usize {
        return @import("logits_processor.zig").selectTokenFromHidden(allocator, lm_head, self.options, &self.mask, h_last, recent_tokens, sampler, s);
    }
};
