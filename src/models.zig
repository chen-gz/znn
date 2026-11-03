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

pub const resnet = @import("models/resnet.zig");
pub const BasicBlock = resnet.BasicBlock;
pub const ResidualBlock = resnet.ResidualBlock;
pub const Bottleneck = resnet.Bottleneck;
pub const ResNetConfig = resnet.ResNetConfig;
pub const ResNet = resnet.ResNet;
pub const ResNetBottleneck = resnet.ResNetBottleneck;
pub const DefaultResNet = resnet.DefaultResNet;
pub const DefaultResNetBottleneck = resnet.DefaultResNetBottleneck;
pub const resnet18 = resnet.resnet18;
pub const resnet34 = resnet.resnet34;
pub const resnet50 = resnet.resnet50;
pub const resnet101 = resnet.resnet101;
pub const resnet152 = resnet.resnet152;
pub const resnet20 = resnet.resnet20;
pub const resnet32 = resnet.resnet32;
pub const resnet56 = resnet.resnet56;
pub const resnet110 = resnet.resnet110;

pub const convnext = @import("models/convnext.zig");
pub const ConvNeXtBlock = convnext.ConvNeXtBlock;
pub const DownsampleLayer = convnext.DownsampleLayer;
pub const ConvNeXtConfig = convnext.ConvNeXtConfig;
pub const ConvNeXt = convnext.ConvNeXt;
pub const DefaultConvNeXt = convnext.DefaultConvNeXt;
pub const convnext_tiny = convnext.convnext_tiny;
pub const convnext_small = convnext.convnext_small;
pub const convnext_base = convnext.convnext_base;
pub const convnext_large = convnext.convnext_large;
pub const convnext_femto = convnext.convnext_femto;
pub const convnext_atto = convnext.convnext_atto;
pub const convnext_pico = convnext.convnext_pico;

pub const recurrent_conv = @import("models/recurrent_conv.zig");
pub const RecurrentConvOptions = recurrent_conv.RecurrentConvOptions;
pub const RecurrentConvResult = recurrent_conv.RecurrentConvResult;
pub const RecurrentWorkspace = recurrent_conv.RecurrentWorkspace;
pub const RecurrentConvBlock = recurrent_conv.RecurrentConvBlock;
pub const RecurrentConvNetConfig = recurrent_conv.RecurrentConvNetConfig;
pub const RecurrentConvNet = recurrent_conv.RecurrentConvNet;

test {
    std.testing.refAllDecls(@This());
    _ = @import("models/recurrent_conv.zig");
    _ = @import("models/tests.zig");
}


