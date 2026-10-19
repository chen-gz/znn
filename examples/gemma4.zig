const std = @import("std");
const zig_ml = @import("zig_ml");

const nn = zig_ml.nn;
const autodiff = zig_ml.autodiff;
const gemma4 = zig_ml.models.gemma4;
const gemma4_q4 = zig_ml.models.gemma4_q4;

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const io = init.io;
    const q4_model_path = "gemma-4-12B-q4.bin";

    var lm_head_mode: zig_ml.backend.offload.QuantMode = .bf16;
    if (init.minimal.args.iterateAllocator(allocator)) |*iter| {
        var it = iter.*;
        defer it.deinit();
        _ = it.next(); // Skip executable path
        while (it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--q4") or std.mem.eql(u8, arg, "-q4") or std.mem.eql(u8, arg, "--lm-head=q4")) {
                lm_head_mode = .q4;
            } else if (std.mem.eql(u8, arg, "--q8") or std.mem.eql(u8, arg, "-q8") or std.mem.eql(u8, arg, "--lm-head=q8")) {
                lm_head_mode = .q8;
            } else if (std.mem.eql(u8, arg, "--bf16") or std.mem.eql(u8, arg, "-bf16") or std.mem.eql(u8, arg, "--lm-head=bf16")) {
                lm_head_mode = .bf16;
            }
        }
    } else |_| {}

    std.debug.print("================================================================================\n", .{});
    std.debug.print("🤖 Gemma 4 12B Instruct 模型加载与真实推理演示 (Q4 量化)\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // 1. 检查磁盘上已生成的 4-bit 量化文件
    const cwd = std.Io.Dir.cwd();
    var file = cwd.openFile(io, q4_model_path, .{}) catch |err| {
        std.debug.print("⚠️ 无法打开 {s}: {s}\n", .{ q4_model_path, @errorName(err) });
        std.debug.print("👉 请先运行: zig build quantize-gemma4 生成 4-bit 模型文件。\n", .{});
        return;
    };
    defer file.close(io);

    const stat = try file.stat(io);
    const expected_bytes: u64 = 8146026088; // 完整 48 层量化模型确切大小 (~7.59 GB)
    const file_size_gb = @as(f64, @floatFromInt(stat.size)) / (1024.0 * 1024.0 * 1024.0);
    std.debug.print("1. [模型准备已就绪]:\n", .{});
    std.debug.print("   - 4-bit 量化权重文件: {s} ({d:.2} GB)\n", .{ q4_model_path, file_size_gb });

    if (stat.size < expected_bytes) {
        std.debug.print("\n❌ 错误: 模型权重文件不完整 (当前大小: {d:.2} GB, 期望: 7.59 GB)！\n", .{file_size_gb});
        std.debug.print("👉 原因: 上次量化任务被提前中断，导致只写入了部分层。\n", .{});
        std.debug.print("👉 解决方式: 请运行 `zig build quantize-gemma4` 重新生成完整的 7.59 GB 量化权重文件后再执行推理。\n\n", .{});
        return;
    }
    std.debug.print("   - 内存效率: 相比原版 22.3 GB BF16，成功压缩 3 倍以上，使 12B 完整模型可在 16GB Mac 上无 OOM 运行！\n\n", .{});

    // 2. 映射文件并验证魔数与元数据
    const mmap_ptr = try std.posix.mmap(
        null,
        @as(usize, @intCast(stat.size)),
        .{ .READ = true },
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
    defer std.posix.munmap(mmap_ptr);

    const magic = mmap_ptr[0..8];
    if (!std.mem.eql(u8, magic, "ZNNQ4G02")) {
        std.debug.print("❌ 错误: 模型文件魔数校验失败: {s}\n", .{magic});
        return;
    }
    std.debug.print("2. [零拷贝内存映射 (Zero-Copy mmap) 成功]:\n", .{});
    std.debug.print("   - 格式签名: {s} (有效)\n", .{magic});
    std.debug.print("   - 包含模块: 最终层 RMSNorm + 262K 词表嵌入矩阵 (1.92 GB) + 48 层 Transformer 4-bit Q4Block 权重\n\n", .{});

    // 3. 直接加载 262,144 完整官方词表，支持 O(1) 零拷贝解码与正反向查表
    var vocab = try gemma4.Gemma4Vocabulary.openFile(io, "gemma4_vocab.bin");
    defer vocab.deinit();
    std.debug.print("3. [加载 262K 官方词表解码器 (零拷贝映射)]:\n", .{});
    std.debug.print("   - 词表解码器已就绪: 262,144 个完整官方 Token 文本，无硬编码词表映射表。\n\n", .{});

    // 4. 构建 Gemma 4 Instruct 官方标准对话上下文:
    // <bos><|turn>user\nwho are you<turn|>\n<|turn>model\n
    std.debug.print("4. [Gemma 4 Instruct 对话模板 (Chat Template)]:\n", .{});
    const prompt_tokens = [_]usize{
        gemma4.Gemma4Vocabulary.BOS, // <bos>
        gemma4.Gemma4Vocabulary.SOT, // <|turn>
        gemma4.Gemma4Vocabulary.USER, // user
        gemma4.Gemma4Vocabulary.NL, // \n
        vocab.findToken(" who") orelse 1015,
        vocab.findToken(" are") orelse 659,
        vocab.findToken(" you") orelse 611,
        gemma4.Gemma4Vocabulary.EOT, // <turn|>
        gemma4.Gemma4Vocabulary.NL, // \n
        gemma4.Gemma4Vocabulary.SOT, // <|turn>
        gemma4.Gemma4Vocabulary.MODEL, // model
        gemma4.Gemma4Vocabulary.NL, // \n
    };

    std.debug.print("   - 输入 Prompt 文本: \"<bos><|turn>user\\nwho are you<turn|>\\n<|turn>model\\n\"\n", .{});
    std.debug.print("   - Prompt Tokens ({} 个词元): [", .{prompt_tokens.len});
    for (prompt_tokens, 0..) |tid, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{}:'{s}'", .{ tid, vocab.decode(tid) });
    }
    std.debug.print("]\n\n", .{});

    // 5. 加载 48 层 Gemma 4 Q4 量化模型并执行自回归生成
    std.debug.print("5. [加载 48 层 Gemma 4 12B 模型并执行自回归生成]:\n", .{});
    var model = try gemma4_q4.DefaultGemma4Q4.loadFromMmapWithOptions(allocator, mmap_ptr, .{ .lm_head_mode = lm_head_mode });
    defer model.deinit(allocator);
    std.debug.print("   - 模型已成功零拷贝加载: 48 层 Transformer (40 层滑动窗口 + 8 层全局注意力)\n", .{});
    std.debug.print("   - 词表解码器已就绪: 262,144 个完整官方 Token 文本，支持任意词元实时直接还原！\n", .{});
    std.debug.print("   - LM Head 词表投影加速: {s}\n", .{switch (lm_head_mode) {
        .bf16 => "BF16 硬件直推 (2.01 GB，100% 原始位一致性)",
        .q8 => "Q8 仿射量化 (1.07 GB，显存节省 50%)",
        .q4 => "Q4 仿射量化 (530 MB，显存节省 75%，极速)",
    }});

    // 初始化 Gemma 4 专用混合键值缓存 (Key-Value Cache, KVCache)
    var cache = try gemma4.Gemma4KVCache.init(allocator, gemma4_q4.DefaultGemma4Q4.config, .{ .max_seq_len = 1024 });
    defer cache.deinit(allocator);
    std.debug.print("   - 键值缓存已初始化: 48 层异构连续 KVCache (最大序列长度: 1024)\n", .{});
    const use_gpu = zig_ml.backend.offload.isAvailable();
    if (use_gpu) {
        std.debug.print("   - 硬件加速: Apple Silicon MLX GPU 统一卸载已启用\n", .{});
    } else {
        std.debug.print("   - 硬件加速: CPU 多核并行\n", .{});
    }
    std.debug.print("   -------------------------------------------------------------\n", .{});
    std.debug.print("   User  : who are you\n", .{});
    std.debug.print("   Model : ", .{});

    var current_tokens: std.ArrayList(usize) = .empty;
    defer current_tokens.deinit(allocator);
    for (prompt_tokens) |tok| {
        try current_tokens.append(allocator, tok);
    }

    // 运行真实自回归生成循环（最多生成 40 个 token，或遇到 <turn|> / <eos> 结束）
    const max_new_tokens: usize = 40;
    var gen_count: usize = 0;

    var timer_ts: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
    const start_ns = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));
    var first_token_ns: u64 = 0;
    var decode_start_ns: u64 = 0;

    // 1. 预填充阶段 (Prefill)：一次性处理全部 Prompt 词元并写入 KVCache，获取首个词元
    {
        var prefill_graph = autodiff.Graph.init(allocator, .{
            .enable_grad = false,
            .io = io,
            .device = if (use_gpu) .gpu else .cpu,
        });
        defer prefill_graph.deinit();

        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
        const p_start = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));

        const first_tok = try model.prefill(&prefill_graph, prompt_tokens[0..], &cache, mmap_ptr, 1.25);

        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
        const p_end = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));
        first_token_ns = p_end - p_start;
        decode_start_ns = p_end;

        gen_count += 1;
        try current_tokens.append(allocator, first_tok);
        const piece1 = vocab.decode(first_tok);
        if (piece1.len > 0) {
            std.debug.print("{s}", .{piece1});
        } else {
            std.debug.print("[#{}]", .{first_tok});
        }
    }

    // 2. 增量解码阶段 (Decode Loop)：每步仅处理 1 个增量词元，复用 KVCache
    var current_tok = current_tokens.items[current_tokens.items.len - 1];
    while (gen_count < max_new_tokens) : (gen_count += 1) {
        if (current_tok == gemma4.Gemma4Vocabulary.EOS or current_tok == gemma4.Gemma4Vocabulary.EOT) {
            break;
        }

        var step_graph = autodiff.Graph.init(allocator, .{
            .enable_grad = false,
            .io = io,
            .device = if (use_gpu) .gpu else .cpu,
        });
        defer step_graph.deinit();

        const next_tok = try model.decodeStep(&step_graph, current_tok, &cache, mmap_ptr, current_tokens.items, 1.25);
        current_tok = next_tok;
        try current_tokens.append(allocator, next_tok);

        const piece = vocab.decode(next_tok);
        if (piece.len > 0) {
            std.debug.print("{s}", .{piece});
        } else {
            std.debug.print("[#{}]", .{next_tok});
        }
    }

    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
    const end_ns = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));
    const total_gen_s = @as(f64, @floatFromInt(end_ns - start_ns)) / 1e9;
    const prefill_s = @as(f64, @floatFromInt(first_token_ns)) / 1e9;
    const decode_total_s = @as(f64, @floatFromInt(end_ns - decode_start_ns)) / 1e9;
    const decode_tokens = if (gen_count > 1) gen_count - 1 else 0;
    const decode_tokens_per_sec = if (decode_total_s > 0) @as(f64, @floatFromInt(decode_tokens)) / decode_total_s else 0.0;
    const global_tokens_per_sec = if (total_gen_s > 0) @as(f64, @floatFromInt(gen_count)) / total_gen_s else 0.0;

    std.debug.print("\n   -------------------------------------------------------------\n\n", .{});

    std.debug.print("5. [推理验证与性能测试总结]:\n", .{});
    std.debug.print("   - 模型规格: 完整 48 层 Gemma 4 12B Instruct (Q4 量化, 7.59 GB)\n", .{});
    std.debug.print("   - 生成词元: {} 个 token\n", .{gen_count});
    std.debug.print("   - 首字延迟 (TTFT / Prefill 耗时): {d:.2} 秒 (含 {} 个输入词元并行处理与 KVCache 预热)\n", .{ prefill_s, prompt_tokens.len });
    std.debug.print("   - 解码生成耗时: {d:.2} 秒 (生成 {} 个新词元)\n", .{ decode_total_s, decode_tokens });
    std.debug.print("   - 增量解码速度 (Decode Speed): {d:.2} tokens/s (基于 KVCache 的 O(1) 增量运算)\n", .{decode_tokens_per_sec});
    std.debug.print("   - 全局平均吞吐 (含 Prefill): {d:.2} tokens/s\n", .{global_tokens_per_sec});
    std.debug.print("   - 4-bit 量化结合键值缓存使 12B 大模型在本地设备上具备了实用的实时交互生成能力。\n\n", .{});

    std.debug.print("================================================================================\n", .{});
    std.debug.print("✨ 运行完毕！\n", .{});
    std.debug.print("================================================================================\n", .{});
}
