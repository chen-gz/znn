//! 设备端对数几率处理与候选词元选择 (Device Logits Processing & Candidate Selection)
//!
//! 提供模型无关的 GPU 端对数几率截断 (Logits Softcapping)、重复惩罚 (Repetition Penalty)、
//! 词元排除掩码 (Token Exclusion Mask)、候选生成与抽样终结 (Sampling)。
//! 遵循分层规范，位于 engine 包（层级 5），彻底解耦模型架构与模型专用配置结构体。

const std = @import("std");
const backend = @import("../backend.zig");
const mlx = backend.mlx;
const offload = backend.offload;
const sampling = @import("sampling.zig");
const nn = @import("../nn.zig");
const tb = nn.transformer_blocks;

pub const Stream = mlx.Stream;
pub const LmHead = offload.LmHead;
pub const LmHeadWeights = offload.LmHeadWeights;
pub const QuantWeights = offload.QuantWeights;
pub const Sampler = sampling.Sampler;
pub const SamplingOptions = sampling.SamplingOptions;
pub const TokenExclusionMask = nn.TokenExclusionMask;

pub const LogitsProcessorOptions = tb.LogitsProcessorOptions;
pub const softcapLogits = tb.softcapLogits;
pub const penalizeDevice = tb.penalizeDevice;
pub const DeviceCandidates = tb.DeviceCandidates;
pub const HostCandidates = tb.HostCandidates;
pub const deviceCandidates = tb.deviceCandidates;
pub const selectToken = tb.selectToken;
pub const selectTokenFromHidden = tb.selectTokenFromHidden;

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
        sampler: anytype,
        s: ?Stream,
    ) !usize {
        return tb.selectToken(allocator, logits, self.options, &self.mask, recent_tokens, sampler, s);
    }

    pub fn selectTokenFromHidden(
        self: *const LogitsProcessor,
        allocator: std.mem.Allocator,
        lm_head: anytype,
        h_last: []const f32,
        recent_tokens: []const usize,
        sampler: anytype,
        s: ?Stream,
    ) !usize {
        return tb.selectTokenFromHidden(allocator, lm_head, self.options, &self.mask, h_last, recent_tokens, sampler, s);
    }
};
