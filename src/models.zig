//! Models - 专用深度学习模型与专属架构库
//! 收敛具体模型架构 (如 Gemma 4) 及其特定非通用算子与端到端推理实现。

const std = @import("std");

pub const gemma4 = @import("models/gemma4.zig");
pub const Gemma4ForCausalLM = gemma4.Gemma4ForCausalLM;
pub const gemma4_q4 = @import("models/gemma4_q4.zig");
pub const DefaultGemma4Q4 = gemma4_q4.DefaultGemma4Q4;
pub const gemma4_e4b = @import("models/gemma4_e4b.zig");
pub const Gemma4E4BModel = gemma4_e4b.Gemma4E4BModel;
pub const Gemma4E4BDecoderLayer = gemma4_e4b.Gemma4E4BDecoderLayer;
pub const Gemma4E4BForCausalLM = gemma4_e4b.Gemma4E4BForCausalLM;

pub const gpt = @import("models/gpt.zig");
pub const GPTConfig = gpt.GPTConfig;
pub const GPT = gpt.GPT;
pub const TransformerDecoder = gpt.TransformerDecoder;
pub const DefaultGPT = gpt.DefaultGPT;

test {
    std.testing.refAllDecls(@This());
    _ = @import("models/tests.zig");
}
