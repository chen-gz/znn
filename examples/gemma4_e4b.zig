//! Gemma 4 E4B (Effective 4B) 4-bit / 8-bit 混合量化模型直通推理与性能评测演示
//!
//! 演示调用 `Gemma4E4BEngine` 执行端到端极速流式生成与性能基准测试。

const std = @import("std");
const zig_ml = @import("zig_ml");

const gemma4 = zig_ml.models.gemma4;
const gemma4_e4b = zig_ml.models.gemma4_e4b;
const Gemma4E4BEngine = gemma4_e4b.Gemma4E4BEngine;

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const io = init.io;

    // 默认模型与生成参数直接配置在代码中，开箱即用
    const model_dir = "/Users/guangzong/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit";
    const prompt = "who are you";
    const max_new_tokens: usize = 40;
    // 默认采样选项与 Gemma 4 官方 generation_config.json 一致 (temperature 1.0, top_k 64, top_p 0.95)
    const sampling_options = gemma4.Gemma4Config.e4b.generation;

    std.debug.print("================================================================================\n", .{});
    std.debug.print("🤖 Gemma 4 E4B Instruct (4B 等效端侧大模型) GPU 极速推理与真实评测\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // 1. 初始化端到端推理引擎（一站式完成权重直通映射、Metal 融合算子预热与 262K 词表零拷贝就绪）
    std.debug.print("1. [准备模型权重与词表解码器]:\n", .{});
    std.debug.print("   - 模型路径: {s}\n", .{model_dir});

    var engine = Gemma4E4BEngine.init(allocator, io, .{
        .model_dir = model_dir,
        .vocab_path = "gemma4_vocab.bin",
        .warmup = true,
    }) catch |err| {
        std.debug.print("❌ 无法从 {s} 加载模型: {s}\n", .{ model_dir, @errorName(err) });
        return;
    };
    defer engine.deinit();

    std.debug.print("   - 格式架构: ⚡ MLX 原生分块量化 Safetensors (GPU 显存直通映射)\n", .{});
    std.debug.print("   - 模型规模: 42 层 Transformer (35 层局部滑动窗口 + 7 层全局全注意力)\n", .{});
    std.debug.print("   - 逐层增强: 逐层嵌入输入增强 (Per-Layer Embeddings, PLE) 已完全就绪\n", .{});
    std.debug.print("   - 算子预热: ✅ 已完成 Metal 着色器与融合图预编译 (Warmup Enabled)\n", .{});
    std.debug.print("   - 加载耗时: {d:.3} 秒 (⚡ 秒开极速启动！)\n\n", .{engine.load_time_s});

    std.debug.print("2. [加载 262K 官方词表解码器]:\n", .{});
    std.debug.print("   - 词表解码器已就绪: 262,144 个官方 Token 文本，支持 O(1) 实时反向解码。\n\n", .{});

    // 2. 执行端到端流式推理评测（自动套用 Instruct 模板、逐词元流式输出、首字延迟与解码吞吐统计）
    std.debug.print("3. [启动 Gemma 4 E4B 自回归生成推理]:\n", .{});
    std.debug.print("   采样  : 随机采样 temperature={d:.2} top_k={} top_p={d:.2} 重复惩罚={d:.2} seed={}\n", .{
        sampling_options.temperature,
        sampling_options.top_k,
        sampling_options.top_p,
        sampling_options.repetition_penalty,
        sampling_options.seed,
    });
    std.debug.print("   User  : {s}\n", .{prompt});
    std.debug.print("   Model : ", .{});

    _ = try engine.streamPrint(prompt, .{
        .sampling = sampling_options,
        .max_new_tokens = max_new_tokens,
        .verbose = true,
    });
}
