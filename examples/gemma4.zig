//! Gemma 4 12B Instruct 模型真实推理与性能评测演示 (Q4 量化)
//!
//! 演示调用 `LLMPipeline` 执行端到端极速流式生成与性能基准测试。

const std = @import("std");
const zig_ml = @import("zig_ml");

const gemma4_q4 = zig_ml.models.gemma4_q4;
const LLMPipeline = zig_ml.engine.LLMPipeline;

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const io = init.io;

    // 默认模型与生成参数直接配置在代码中，开箱即用
    const prompt = "who are you";
    const max_new_tokens: usize = 40;
    const sampling_options = gemma4_q4.DefaultGemma4Q4.config.generation;

    const cwd = std.Io.Dir.cwd();
    const model_path = if (cwd.openFile(io, "gemma-4-12B-direct.bin", .{})) |*f| blk: {
        var file = f.*;
        file.close(io);
        break :blk "gemma-4-12B-direct.bin";
    } else |_| "gemma-4-12B-q4.bin";

    var pipe = LLMPipeline.load(allocator, io, .{
        .model_path = model_path,
        .vocab_path = "gemma4_vocab.bin",
        .warmup = true,
    }) catch |err| {
        std.debug.print("⚠️ 无法初始化 Gemma 4 12B 模型: {s}\n👉 请确认当前目录存在 gemma-4-12B-direct.bin 或 gemma-4-12B-q4.bin\n", .{@errorName(err)});
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
