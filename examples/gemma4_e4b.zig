//! Gemma 4 E4B (Effective 4B) 4-bit / 8-bit 混合量化模型直通推理与性能评测演示
//!
//! 命令行入口，调用 `zig_ml.models.gemma4_e4b.Gemma4E4BModel` 执行端到端流式生成与性能基准测试。

const std = @import("std");
const zig_ml = @import("zig_ml");

const mlx = zig_ml.backend.mlx;
const gemma4 = zig_ml.models.gemma4;
const gemma4_e4b = zig_ml.models.gemma4_e4b;
const sampling = zig_ml.backend.sampling;

pub const TensorMeta = gemma4_e4b.TensorMeta;
pub const SafetensorsArchive = gemma4_e4b.SafetensorsArchive;
pub const PleWeights = gemma4_e4b.PleWeights;
pub const PlePrepContext = gemma4_e4b.PlePrepContext;
pub const GpuLayerE4B = gemma4_e4b.GpuLayerE4B;
pub const GpuKvCache = gemma4_e4b.GpuKvCache;
pub const Gemma4E4BModel = gemma4_e4b.Gemma4E4BModel;

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
    var sampler = sampling.Sampler.init(sampling_options);
    // 不带重复惩罚的贪婪采样启用双阶段 GPU 异步流水线；随机采样与带惩罚的贪婪采样走同步解码
    const use_pipeline = sampling_options.isPlainGreedy();

    std.debug.print("================================================================================\n", .{});
    std.debug.print("🤖 Gemma 4 E4B Instruct (4B 等效端侧大模型) GPU 极速推理与真实评测\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // 1. 验证模型目录
    std.debug.print("1. [准备模型权重与词表解码器]:\n", .{});
    std.debug.print("   - 模型路径: {s}\n", .{model_dir});

    var timer_load: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_load);
    const load_start_ns = @as(u64, @intCast(timer_load.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_load.nsec));

    var model = Gemma4E4BModel.loadFromDirectoryWithOptions(allocator, io, model_dir, .{ .warmup = enable_warmup }) catch |err| {
        std.debug.print("❌ 无法从 {s} 加载模型: {s}\n", .{ model_dir, @errorName(err) });
        return;
    };
    defer model.deinit();

    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_load);
    const load_end_ns = @as(u64, @intCast(timer_load.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_load.nsec));
    const load_elapsed_s = @as(f64, @floatFromInt(load_end_ns - load_start_ns)) / 1e9;

    std.debug.print("   - 格式架构: ⚡ MLX 原生分块量化 Safetensors (GPU 显存直通映射)\n", .{});
    std.debug.print("   - 模型规模: 42 层 Transformer (35 层局部滑动窗口 + 7 层全局全注意力)\n", .{});
    std.debug.print("   - 逐层增强: 逐层嵌入输入增强 (Per-Layer Embeddings, PLE) 已完全就绪\n", .{});
    std.debug.print("   - 算子预热: {s}\n", .{if (enable_warmup) "✅ 已完成 Metal 着色器与融合图预编译 (Warmup Enabled)" else "⏭️ 已跳过 (--no-warmup)"});
    std.debug.print("   - 加载耗时: {d:.3} 秒 (⚡ 秒开极速启动！)\n\n", .{load_elapsed_s});

    // 2. 加载词表解码器
    var vocab = try gemma4.Gemma4Vocabulary.openFile(io, "gemma4_vocab.bin");
    defer vocab.deinit();
    try vocab.buildLookupTable(allocator);
    std.debug.print("2. [加载 262K 官方词表解码器]:\n", .{});
    std.debug.print("   - 词表解码器已就绪: 262,144 个官方 Token 文本，支持 O(1) 实时反向解码。\n\n", .{});

    // 3. 构建对话上下文
    const user_prompt_str = custom_prompt orelse "who are you";
    var prompt_list: std.ArrayList(usize) = .empty;
    defer prompt_list.deinit(allocator);

    try prompt_list.append(allocator, gemma4.Gemma4Vocabulary.BOS);
    try prompt_list.append(allocator, gemma4.Gemma4Vocabulary.SOT);
    try prompt_list.append(allocator, gemma4.Gemma4Vocabulary.USER);
    try prompt_list.append(allocator, gemma4.Gemma4Vocabulary.NL);

    const user_tokens = try vocab.encode(allocator, user_prompt_str);
    defer allocator.free(user_tokens);
    try prompt_list.appendSlice(allocator, user_tokens);

    try prompt_list.append(allocator, gemma4.Gemma4Vocabulary.EOT);
    try prompt_list.append(allocator, gemma4.Gemma4Vocabulary.NL);
    try prompt_list.append(allocator, gemma4.Gemma4Vocabulary.SOT);
    try prompt_list.append(allocator, gemma4.Gemma4Vocabulary.MODEL);
    try prompt_list.append(allocator, gemma4.Gemma4Vocabulary.NL);

    const prompt_tokens = prompt_list.items;
    std.debug.print("3. [Gemma 4 Instruct 对话上下文与 Prompt]:\n", .{});
    std.debug.print("   - Prompt 词元 ({} 个): [", .{prompt_tokens.len});
    for (prompt_tokens, 0..) |tid, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{}:'{s}'", .{ tid, vocab.decode(tid) });
    }
    std.debug.print("]\n\n", .{});

    // 4. 执行推理评测
    std.debug.print("4. [启动 Gemma 4 E4B 自回归生成推理]:\n", .{});
    printSamplingOptions(sampling_options, use_pipeline);
    std.debug.print("   User  : {s}\n", .{user_prompt_str});
    std.debug.print("   Model : ", .{});

    var timer_ts: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
    const start_ns = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));

    // Prefill 阶段：对齐官方 mlx_lm.generate_step 流水线，在等待 Prefill GPU 执行完成的同时，
    // 提前构建并异步入队第 1 个 Decode 步的计算图，实现 Prefill -> Decode 零气泡无缝衔接
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
    const p_start = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));

    var first_tok_arr_opt: ?mlx.Array = if (use_pipeline)
        try model.prefillAsync(prompt_tokens)
    else
        null;
    defer if (first_tok_arr_opt) |*a| a.deinit();

    var pending_tok_arr_opt: ?mlx.Array = if (first_tok_arr_opt != null and max_new_tokens > 1)
        try model.decodeStepAsync(first_tok_arr_opt.?)
    else
        null;
    errdefer if (pending_tok_arr_opt) |*p| p.deinit();

    const first_tok: usize = if (first_tok_arr_opt) |a|
        @intCast(try a.itemUint32())
    else
        try model.prefill(prompt_tokens, &sampler);

    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
    const p_end = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));
    const prefill_time_ms = @as(f64, @floatFromInt(p_end - p_start)) / 1e6;

    var current_tokens: std.ArrayList(usize) = .empty;
    defer current_tokens.deinit(allocator);
    for (prompt_tokens) |tok| try current_tokens.append(allocator, tok);
    try current_tokens.append(allocator, first_tok);

    var gen_count: usize = 1;
    const piece1 = vocab.decode(first_tok);
    if (piece1.len > 0) {
        std.debug.print("{s}", .{piece1});
    } else {
        std.debug.print("[#{}]", .{first_tok});
    }

    // Decode 循环：不带重复惩罚的贪婪采样启用双阶段 GPU 异步流水线 (2-Stage Async GPU Pipelining)，
    // 使 CPU 构建第 t+1 步计算图与 GPU 执行第 t 步 Metal 内核完全重叠，彻底消除步间气泡
    var current_tok = first_tok;
    const decode_start_ns = p_end;

    if (use_pipeline) {
        if (first_tok_arr_opt) |*a| {
            a.deinit();
            first_tok_arr_opt = null;
        }
        if (gen_count < max_new_tokens and current_tok != gemma4.Gemma4Vocabulary.EOS and current_tok != gemma4.Gemma4Vocabulary.EOT) {
            var pending_tok_arr = pending_tok_arr_opt.?;
            pending_tok_arr_opt = null;

            while (gen_count < max_new_tokens) {
                // 在等待当前步 GPU 完成前，直接以 GPU 词元张量预构建并异步派发下一步计算图
                var next_pending_opt: ?mlx.Array = if (gen_count + 1 < max_new_tokens)
                    try model.decodeStepAsync(pending_tok_arr)
                else
                    null;

                const next_tok: usize = @intCast(try pending_tok_arr.itemUint32());
                pending_tok_arr.deinit();

                try current_tokens.append(allocator, next_tok);
                current_tok = next_tok;
                gen_count += 1;

                const piece = vocab.decode(next_tok);
                if (piece.len > 0) {
                    std.debug.print("{s}", .{piece});
                } else {
                    std.debug.print("[#{}]", .{next_tok});
                }

                if (current_tok == gemma4.Gemma4Vocabulary.EOS or current_tok == gemma4.Gemma4Vocabulary.EOT) {
                    if (next_pending_opt) |*np| np.deinit();
                    break;
                }

                if (next_pending_opt) |np| {
                    pending_tok_arr = np;
                } else {
                    break;
                }
            }
        } else if (pending_tok_arr_opt) |*p| {
            p.deinit();
            pending_tok_arr_opt = null;
        }
    } else {
        while (gen_count < max_new_tokens) {
            if (current_tok == gemma4.Gemma4Vocabulary.EOS or current_tok == gemma4.Gemma4Vocabulary.EOT) {
                break;
            }

            const next_tok = try model.decodeStep(current_tok, current_tokens.items, &sampler);
            try current_tokens.append(allocator, next_tok);
            current_tok = next_tok;
            gen_count += 1;

            const piece = vocab.decode(next_tok);
            if (piece.len > 0) {
                std.debug.print("{s}", .{piece});
            } else {
                std.debug.print("[#{}]", .{next_tok});
            }
        }
    }

    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
    const end_ns = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));

    const total_decode_time_s = @as(f64, @floatFromInt(end_ns - decode_start_ns)) / 1e9;
    const total_time_s = @as(f64, @floatFromInt(end_ns - start_ns)) / 1e9;
    const decode_count = if (gen_count > 1) gen_count - 1 else 0;
    const decode_speed = if (total_decode_time_s > 0.0) @as(f64, @floatFromInt(decode_count)) / total_decode_time_s else 0.0;

    std.debug.print("\n\n================================================================================\n", .{});
    std.debug.print("📊 Gemma 4 E4B 性能基准测试报告 (Performance Benchmark):\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("   - 模型冷启动加载耗时 : {d:.3} 秒 (零拷贝 mmap 直通)\n", .{load_elapsed_s});
    std.debug.print("   - Prompt 预填充耗时  : {d:.2} 毫秒 (TTFT 首字延迟)\n", .{prefill_time_ms});
    std.debug.print("   - 预填充吞吐速率     : {d:.1} token/s\n", .{@as(f64, @floatFromInt(prompt_tokens.len)) / (prefill_time_ms / 1000.0)});
    std.debug.print("   - 生成词元总数       : {} tokens\n", .{gen_count});
    std.debug.print("   - 自回归解码速率     : {d:.2} token/s ({s})\n", .{ decode_speed, if (use_pipeline) "纯 Metal GPU 双阶段异步流水线" else "同步解码 + 设备端候选选择" });
    std.debug.print("   - 端到端总生成耗时   : {d:.2} 秒\n", .{total_time_s});
    std.debug.print("================================================================================\n\n", .{});
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
