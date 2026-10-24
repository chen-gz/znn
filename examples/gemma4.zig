//! Gemma 4 12B Instruct 模型真实推理与性能评测演示 (Q4 量化)
//!
//! 直接调用 `DefaultGemma4Q4` 与 `Gemma4Vocabulary` 执行端到端极速流式生成与性能基准测试。

const std = @import("std");
const zig_ml = @import("zig_ml");

const gemma4 = zig_ml.models.gemma4;
const gemma4_q4 = zig_ml.models.gemma4_q4;
const DefaultGemma4Q4 = gemma4_q4.DefaultGemma4Q4;

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const io = init.io;

    // 默认模型与生成参数直接配置在代码中，开箱即用
    const prompt = "who are you";
    const max_new_tokens: usize = 40;
    const sampling_options = DefaultGemma4Q4.config.generation;

    const cwd = std.Io.Dir.cwd();
    const model_path = if (cwd.openFile(io, "gemma-4-12B-direct.bin", .{})) |*f| blk: {
        var file = f.*;
        file.close(io);
        break :blk "gemma-4-12B-direct.bin";
    } else |_| "gemma-4-12B-q4.bin";

    var timer_load: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_load);
    const load_start_ns = @as(u64, @intCast(timer_load.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_load.nsec));

    var file = cwd.openFile(io, model_path, .{}) catch |err| {
        std.debug.print("⚠️ 无法打开 Gemma 4 12B 模型权重文件 ({s}): {s}\n👉 请确认当前目录存在 gemma-4-12B-direct.bin 或 gemma-4-12B-q4.bin\n", .{ model_path, @errorName(err) });
        return;
    };
    defer file.close(io);

    const stat = try file.stat(io);
    const mmap_ptr = try std.posix.mmap(
        null,
        @as(usize, @intCast(stat.size)),
        .{ .READ = true },
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
    defer std.posix.munmap(mmap_ptr);

    var model = DefaultGemma4Q4.loadFromMmapWithOptions(allocator, mmap_ptr, .{ .warmup = true }) catch |err| {
        std.debug.print("⚠️ 无法初始化 Gemma 4 12B 模型: {s}\n", .{@errorName(err)});
        return;
    };
    defer model.deinit(allocator);

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

    var gen = try model.generate(allocator, prompt_tokens, .{
        .sampling = sampling_options,
        .max_new_tokens = max_new_tokens,
        .mmap_bytes = mmap_ptr,
    });
    defer gen.deinit();

    var decode_start_ns: u64 = start_ns;
    var prefill_time_ms: f64 = 0.0;
    while (try gen.next()) |tok| {
        if (gen.generated().len == 1) {
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
    const gen_count = gen.generated().len;
    const decode_count = if (gen_count > 1) gen_count - 1 else 0;
    const decode_tps = if (decode_time_s > 0.0) @as(f64, @floatFromInt(decode_count)) / decode_time_s else 0.0;
    const prefill_tps = if (prefill_time_ms > 0.0)
        @as(f64, @floatFromInt(prompt_tokens.len)) / (prefill_time_ms / 1000.0)
    else
        0.0;

    std.debug.print(
        \\
        \\================================================================================
        \\📊 Gemma 4 12B Q4 性能基准测试报告 (Performance Benchmark):
        \\================================================================================
        \\   - 模型冷启动加载耗时 : {d:.3} 秒 (零拷贝 mmap 直通)
        \\   - Prompt 预填充耗时  : {d:.2} 毫秒 (TTFT 首字延迟)
        \\   - 预填充吞吐速率     : {d:.1} token/s
        \\   - 生成词元总数       : {} tokens
        \\   - 增量解码速率       : {d:.2} token/s
        \\   - 端到端总生成耗时   : {d:.2} 秒
        \\================================================================================
        \\
    , .{
        load_time_s,
        prefill_time_ms,
        prefill_tps,
        gen_count,
        decode_tps,
        total_time_s,
    });
}
