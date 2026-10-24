//! Gemma 4 E4B (Effective 4B) 4-bit / 8-bit 混合量化模型直通推理与性能评测演示
//!
//! 命令行入口，调用 `zig_ml.models.gemma4_e4b.Gemma4E4BModel` 执行端到端流式生成与性能基准测试。

const std = @import("std");
const zig_ml = @import("zig_ml");

const gemma4 = zig_ml.models.gemma4;
const gemma4_e4b = zig_ml.models.gemma4_e4b;
const sampling = zig_ml.backend.sampling;

const Gemma4E4BEngine = gemma4_e4b.Gemma4E4BEngine;

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const io = init.io;

    var custom_prompt: ?[]const u8 = null;
    var max_new_tokens: usize = 40;
    // 默认采样选项与 Gemma 4 官方 generation_config.json 一致 (temperature 1.0, top_k 64, top_p 0.95)
    var sampling_options: sampling.SamplingOptions = gemma4.Gemma4Config.e4b.generation;
    var enable_warmup: bool = true;
    const default_model_path = "/Users/guangzong/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit";
    var model_dir: []const u8 = default_model_path;

    if (init.minimal.args.iterateAllocator(allocator)) |*iter| {
        var it = iter.*;
        defer it.deinit();
        _ = it.next(); // Skip binary
        while (it.next()) |arg| {
            if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--prompt")) {
                if (it.next()) |p| custom_prompt = p;
            } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
                if (it.next()) |m| model_dir = m;
            } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--max-tokens")) {
                if (it.next()) |n_str| {
                    max_new_tokens = std.fmt.parseInt(usize, n_str, 10) catch 40;
                }
            } else if (std.mem.eql(u8, arg, "-r") or std.mem.eql(u8, arg, "--penalty")) {
                if (it.next()) |v| sampling_options.repetition_penalty = try parseFloatArg(arg, v);
            } else if (std.mem.eql(u8, arg, "--temperature")) {
                if (it.next()) |v| sampling_options.temperature = try parseFloatArg(arg, v);
            } else if (std.mem.eql(u8, arg, "--top-k")) {
                if (it.next()) |v| sampling_options.top_k = try parseIntArg(usize, arg, v);
            } else if (std.mem.eql(u8, arg, "--top-p")) {
                if (it.next()) |v| sampling_options.top_p = try parseFloatArg(arg, v);
            } else if (std.mem.eql(u8, arg, "--seed")) {
                if (it.next()) |v| sampling_options.seed = try parseIntArg(u64, arg, v);
            } else if (std.mem.eql(u8, arg, "--greedy")) {
                sampling_options.temperature = 0.0;
            } else if (std.mem.eql(u8, arg, "--no-warmup")) {
                enable_warmup = false;
            } else if (std.mem.eql(u8, arg, "--warmup")) {
                enable_warmup = true;
            }
        }
    } else |_| {}

    std.debug.print("================================================================================\n", .{});
    std.debug.print("🤖 Gemma 4 E4B Instruct (4B 等效端侧大模型) GPU 极速推理与真实评测\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // 1. 初始化端到端推理引擎（一站式完成权重直通映射、Metal 融合算子预热与 262K 词表零拷贝就绪）
    std.debug.print("1. [准备模型权重与词表解码器]:\n", .{});
    std.debug.print("   - 模型路径: {s}\n", .{model_dir});

    var engine = Gemma4E4BEngine.init(allocator, io, .{
        .model_dir = model_dir,
        .vocab_path = "gemma4_vocab.bin",
        .warmup = enable_warmup,
    }) catch |err| {
        std.debug.print("❌ 无法从 {s} 加载模型: {s}\n", .{ model_dir, @errorName(err) });
        return;
    };
    defer engine.deinit();

    std.debug.print("   - 格式架构: ⚡ MLX 原生分块量化 Safetensors (GPU 显存直通映射)\n", .{});
    std.debug.print("   - 模型规模: 42 层 Transformer (35 层局部滑动窗口 + 7 层全局全注意力)\n", .{});
    std.debug.print("   - 逐层增强: 逐层嵌入输入增强 (Per-Layer Embeddings, PLE) 已完全就绪\n", .{});
    std.debug.print("   - 算子预热: {s}\n", .{if (enable_warmup) "✅ 已完成 Metal 着色器与融合图预编译 (Warmup Enabled)" else "⏭️ 已跳过 (--no-warmup)"});
    std.debug.print("   - 加载耗时: {d:.3} 秒 (⚡ 秒开极速启动！)\n\n", .{engine.load_time_s});

    std.debug.print("2. [加载 262K 官方词表解码器]:\n", .{});
    std.debug.print("   - 词表解码器已就绪: 262,144 个官方 Token 文本，支持 O(1) 实时反向解码。\n\n", .{});

    // 2. 执行端到端流式推理评测（自动套用 Instruct 模板、逐词元流式输出、首字延迟与解码吞吐统计）
    const user_prompt_str = custom_prompt orelse "who are you";
    std.debug.print("3. [启动 Gemma 4 E4B 自回归生成推理]:\n", .{});
    printSamplingOptions(sampling_options, sampling_options.isPlainGreedy());
    std.debug.print("   User  : {s}\n", .{user_prompt_str});
    std.debug.print("   Model : ", .{});

    _ = try engine.streamPrint(user_prompt_str, .{
        .sampling = sampling_options,
        .max_new_tokens = max_new_tokens,
        .verbose = true,
    });
}

fn parseFloatArg(flag: []const u8, value: []const u8) !f32 {
    return std.fmt.parseFloat(f32, value) catch {
        std.debug.print("❌ 参数 {s} 需要浮点数，实际为: {s}\n", .{ flag, value });
        return error.InvalidArgument;
    };
}

fn parseIntArg(comptime T: type, flag: []const u8, value: []const u8) !T {
    return std.fmt.parseInt(T, value, 10) catch {
        std.debug.print("❌ 参数 {s} 需要非负整数，实际为: {s}\n", .{ flag, value });
        return error.InvalidArgument;
    };
}

/// 打印生效的采样设置 (命令行参数: --temperature / --top-k / --top-p / -r, --penalty / --seed / --greedy)
fn printSamplingOptions(opts: sampling.SamplingOptions, use_pipeline: bool) void {
    if (opts.isGreedy()) {
        std.debug.print("   采样  : 贪婪采样 (Greedy)，重复惩罚 {d:.2}{s}\n", .{
            opts.repetition_penalty,
            if (use_pipeline) "，双阶段异步流水线" else "",
        });
    } else {
        std.debug.print("   采样  : 随机采样 temperature={d:.2} top_k={} top_p={d:.2} 重复惩罚={d:.2} seed={}\n", .{
            opts.temperature, opts.top_k, opts.top_p, opts.repetition_penalty, opts.seed,
        });
    }
}
