//! Gemma 4 E4B (Effective 4B) 4-bit / 8-bit 混合量化模型直通推理与性能评测演示
//!
//! 演示调用 `LLMPipeline` 执行端到端极速流式生成与性能基准测试。

const std = @import("std");
const zig_ml = @import("zig_ml");

const gemma4 = zig_ml.models.gemma4;
const LLMPipeline = zig_ml.engine.LLMPipeline;

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const io = init.io;

    // 默认模型与生成参数直接配置在代码中，开箱即用
    const model_dir = "/Users/guangzong/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit";
    const prompt = "who are you";
    const max_new_tokens: usize = 40;
    const sampling_options = gemma4.Gemma4Config.e4b.generation;

    var pipe = LLMPipeline.load(allocator, io, .{
        .model_path = model_dir,
        .vocab_path = "gemma4_vocab.bin",
        .warmup = true,
    }) catch |err| {
        std.debug.print("❌ 无法从 {s} 加载模型: {s}\n", .{ model_dir, @errorName(err) });
        return;
    };
    defer pipe.deinit();

    std.debug.print("User  : {s}\n", .{prompt});
    std.debug.print("Model : ", .{});

    _ = try pipe.streamPrint(prompt, .{
        .sampling = sampling_options,
        .max_new_tokens = max_new_tokens,
        .verbose = true,
    });
}
