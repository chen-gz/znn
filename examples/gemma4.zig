//! Gemma 4 12B Instruct 模型真实推理与性能评测演示 (Q4 量化)
//!
//! 演示调用 `Gemma4Q4Engine` 执行端到端极速流式生成与性能基准测试。

const std = @import("std");
const zig_ml = @import("zig_ml");

const gemma4_q4 = zig_ml.models.gemma4_q4;
const Gemma4Q4Engine = gemma4_q4.Gemma4Q4Engine;

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const io = init.io;

    // 默认模型与生成参数直接配置在代码中，开箱即用
    const prompt = "who are you";
    const max_new_tokens: usize = 40;
    const sampling_options = gemma4_q4.DefaultGemma4Q4.config.generation;

    var engine = Gemma4Q4Engine.init(allocator, io, .{
        .vocab_path = "gemma4_vocab.bin",
        .warmup = true,
    }) catch |err| {
        std.debug.print("⚠️ 无法初始化 Gemma 4 12B 模型: {s}\n👉 请确认当前目录存在 gemma-4-12B-direct.bin 或 gemma-4-12B-q4.bin\n", .{@errorName(err)});
        return;
    };
    defer engine.deinit();

    std.debug.print("User  : {s}\n", .{prompt});
    std.debug.print("Model : ", .{});

    _ = try engine.streamPrint(prompt, .{
        .sampling = sampling_options,
        .max_new_tokens = max_new_tokens,
        .verbose = true,
    });
}
