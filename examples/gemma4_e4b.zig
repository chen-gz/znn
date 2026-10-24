//! Gemma 4 E4B (Effective 4B) 4-bit / 8-bit 混合量化模型直通推理与性能评测演示
//!
//! 直接调用 `Gemma4E4BModel` 与 `Gemma4Vocabulary` 执行端到端极速流式生成与性能基准测试。

const std = @import("std");
const zig_ml = @import("zig_ml");

const gemma4 = zig_ml.models.gemma4;
const Gemma4E4BModel = zig_ml.models.Gemma4E4BModel;

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const io = init.io;

    // 默认模型与生成参数直接配置在代码中，开箱即用
    const model_dir = "/Users/guangzong/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit";
    const prompt = "who are you";
    const max_new_tokens: usize = 40;
    const sampling_options = gemma4.Gemma4Config.e4b.generation;

    var timer_load: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_load);
    const load_start_ns = @as(u64, @intCast(timer_load.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_load.nsec));

    var model = Gemma4E4BModel.loadFromDirectoryWithOptions(allocator, io, model_dir, .{ .warmup = true }) catch |err| {
        std.debug.print("❌ 无法从 {s} 加载模型: {s}\n", .{ model_dir, @errorName(err) });
        return;
    };
    defer model.deinit();

    var vocab = try gemma4.Gemma4Vocabulary.openFile(io, "gemma4_vocab.bin");
    defer vocab.deinit();
    try vocab.buildLookupTable(allocator);

    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_load);
    const load_end_ns = @as(u64, @intCast(timer_load.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_load.nsec));
    const load_time_s = @as(f64, @floatFromInt(load_end_ns - load_start_ns)) / 1e9;

    const prompt_tokens = try vocab.formatInstructPrompt(allocator, prompt);
    defer allocator.free(prompt_tokens);

    std.debug.print("User  : {s}\n", .{prompt});
    std.debug.print("Model : ", .{});

    var timer_ts: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
    const start_ns = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));

    var generation = try model.generate(allocator, prompt_tokens, .{
        .sampling = sampling_options,
        .max_new_tokens = max_new_tokens,
    });
    defer generation.deinit();

    var decode_start_ns: u64 = start_ns;
    var prefill_time_ms: f64 = 0.0;
    while (try generation.next()) |tok| {
        if (generation.generated().len == 1) {
            _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
            decode_start_ns = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));
            prefill_time_ms = @as(f64, @floatFromInt(decode_start_ns - start_ns)) / 1e6;
        }
        std.debug.print("{s}", .{vocab.decode(tok)});
    }
    std.debug.print("\n", .{});

    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
    const end_ns = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));
    const total_time_s = @as(f64, @floatFromInt(end_ns - start_ns)) / 1e9;
    const decode_time_s = if (decode_start_ns > 0) @as(f64, @floatFromInt(end_ns - decode_start_ns)) / 1e9 else 0.0;
    const gen_count = generation.generated().len;
    const decode_count = if (gen_count > 1) gen_count - 1 else 0;
    const decode_tps = if (decode_time_s > 0.0) @as(f64, @floatFromInt(decode_count)) / decode_time_s else 0.0;
    const prefill_tps = if (prefill_time_ms > 0.0)
        @as(f64, @floatFromInt(prompt_tokens.len)) / (prefill_time_ms / 1000.0)
    else
        0.0;

    std.debug.print(
        \\
        \\================================================================================
        \\📊 Gemma 4 E4B 性能基准测试报告 (Performance Benchmark):
        \\================================================================================
        \\   - 模型冷启动加载耗时 : {d:.3} 秒 (零拷贝 mmap 直通)
        \\   - Prompt 预填充耗时  : {d:.2} 毫秒 (TTFT 首字延迟)
        \\   - 预填充吞吐速率     : {d:.1} token/s
        \\   - 生成词元总数       : {} tokens
        \\   - 自回归解码速率     : {d:.2} token/s ({s})
        \\   - 端到端总生成耗时   : {d:.2} 秒
        \\================================================================================
        \\
    , .{
        load_time_s,
        prefill_time_ms,
        prefill_tps,
        gen_count,
        decode_tps,
        if (generation.isPipelined()) "纯 Metal GPU 双阶段异步流水线" else "同步解码 + 设备端候选选择",
        total_time_s,
    });
}
