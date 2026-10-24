//! 统一大语言模型推理流水线 (Unified LLM Inference Pipeline)
//!
//! 自动检测模型架构与权重格式（支持 MLX 分块量化目录与零拷贝直通 .bin 文件），
//! 抹平底层异构模型实现（E4B 与 12B Q4），提供统一的流式生成、文本输入输出与基准评测。

const std = @import("std");
const backend = @import("../backend.zig");
const models = @import("../models.zig");
const stats_mod = @import("stats.zig");

pub const GenerationStats = stats_mod.GenerationStats;
pub const TextResult = stats_mod.TextResult;

const gemma4 = models.gemma4;
const gemma4_e4b = models.gemma4_e4b;
const gemma4_q4 = models.gemma4_q4;

/// 支持的推理模型架构
pub const ModelArchitecture = enum {
    gemma4_e4b,
    gemma4_12b_q4,
};

/// 根据给定的文件或目录路径自动嗅探模型架构
pub fn detectModelArchitecture(io: std.Io, path: []const u8) !ModelArchitecture {
    const cwd = std.Io.Dir.cwd();

    // 1. 若为目录且包含 Safetensors 或 config.json，判定为 E4B / MLX 格式
    if (cwd.openDir(io, path, .{})) |*dir| {
        var d = dir.*;
        d.close(io);
        return .gemma4_e4b;
    } else |_| {}

    // 2. 若为单文件，读取文件头魔数 (Magic) 判定
    var file = try cwd.openFile(io, path, .{});
    defer file.close(io);

    var read_buf: [32]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);
    const reader = &file_reader.interface;

    var magic: [8]u8 = undefined;
    try reader.readSliceAll(&magic);

    if (std.mem.eql(u8, &magic, gemma4_e4b.Gemma4E4BModel.DIRECT_CACHE_MAGIC)) {
        return .gemma4_e4b;
    } else if (std.mem.eql(u8, &magic, gemma4_q4.DefaultGemma4Q4.DIRECT_MAGIC) or
        std.mem.eql(u8, &magic, gemma4_q4.DefaultGemma4Q4.Q4_MAGIC))
    {
        return .gemma4_12b_q4;
    }

    return error.InvalidModelSignature;
}

/// 泛化多态模型容器
pub const AnyModel = union(enum) {
    gemma4_e4b: gemma4_e4b.Gemma4E4BModel,
    gemma4_12b_q4: gemma4_q4.DefaultGemma4Q4,

    pub fn deinit(self: *AnyModel, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .gemma4_e4b => |*m| m.deinit(),
            .gemma4_12b_q4 => |*m| m.deinit(allocator),
        }
    }
};

/// 泛化自回归生成会话
pub const AnyGeneration = union(enum) {
    gemma4_e4b: gemma4_e4b.Gemma4E4BModel.Generation,
    gemma4_12b_q4: gemma4_q4.DefaultGemma4Q4.Generation,

    pub fn next(self: *AnyGeneration) !?usize {
        return switch (self.*) {
            inline else => |*g| try g.next(),
        };
    }

    pub fn generated(self: *const AnyGeneration) []const usize {
        return switch (self.*) {
            inline else => |*g| g.generated(),
        };
    }

    pub fn isPipelined(self: *const AnyGeneration) bool {
        return switch (self.*) {
            .gemma4_e4b => |*g| g.isPipelined(),
            .gemma4_12b_q4 => false,
        };
    }

    pub fn speculativeHits(self: *const AnyGeneration) usize {
        return switch (self.*) {
            .gemma4_e4b => 0,
            .gemma4_12b_q4 => |*g| g.speculative_hits,
        };
    }

    pub fn deinit(self: *AnyGeneration) void {
        switch (self.*) {
            inline else => |*g| g.deinit(),
        }
    }
};

/// 统一的高级文本流式迭代器 (Text Streaming Iterator)
pub const TextStream = struct {
    pipeline: *LLMPipeline,
    generation: AnyGeneration,
    prompt_tokens: []const usize,
    start_ns: u64,
    decode_start_ns: u64 = 0,
    prefill_time_ms: f64 = 0.0,
    finished: bool = false,

    pub fn deinit(self: *TextStream) void {
        self.generation.deinit();
        self.pipeline.allocator.free(self.prompt_tokens);
    }

    /// 获取下一个文本片段（已直接反向解码为 UTF-8 字符串），生成结束返回 null
    pub fn next(self: *TextStream) !?[]const u8 {
        if (self.finished) return null;
        const maybe_tok = try self.generation.next();
        if (maybe_tok) |tok| {
            if (self.generation.generated().len == 1) {
                var ts: std.posix.system.timespec = undefined;
                _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &ts);
                self.decode_start_ns = @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
                self.prefill_time_ms = @as(f64, @floatFromInt(self.decode_start_ns - self.start_ns)) / 1e6;
            }
            return self.pipeline.vocab.decode(tok);
        }
        self.finished = true;
        return null;
    }

    /// 计算并产出当前/已完成的性能基准统计
    pub fn stats(self: *const TextStream) GenerationStats {
        var ts: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &ts);
        const end_ns = @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        const total_s = @as(f64, @floatFromInt(end_ns - self.start_ns)) / 1e9;
        const decode_s = if (self.decode_start_ns > 0) @as(f64, @floatFromInt(end_ns - self.decode_start_ns)) / 1e9 else 0.0;
        const gen_count = self.generation.generated().len;
        const decode_count = if (gen_count > 1) gen_count - 1 else 0;
        const decode_tps = if (decode_s > 0.0) @as(f64, @floatFromInt(decode_count)) / decode_s else 0.0;
        const prefill_tps = if (self.prefill_time_ms > 0.0)
            @as(f64, @floatFromInt(self.prompt_tokens.len)) / (self.prefill_time_ms / 1000.0)
        else
            0.0;

        return .{
            .load_time_s = self.pipeline.load_time_s,
            .prefill_time_ms = self.prefill_time_ms,
            .decode_time_s = decode_s,
            .total_time_s = total_s,
            .prompt_tokens = self.prompt_tokens.len,
            .generated_tokens = gen_count,
            .prefill_tps = prefill_tps,
            .decode_tps = decode_tps,
            .is_pipelined = self.generation.isPipelined(),
            .speculative_hits = self.generation.speculativeHits(),
        };
    }
};

/// 统一大语言模型推理流水线 (LLMPipeline)
pub const LLMPipeline = struct {
    allocator: std.mem.Allocator,
    model: AnyModel,
    vocab: gemma4.Gemma4Vocabulary,
    load_time_s: f64,
    mmap_ptr: ?[]const u8 = null,

    pub const Options = struct {
        model_path: []const u8,
        vocab_path: []const u8 = "gemma4_vocab.bin",
        lm_head_mode: backend.offload.QuantMode = .bf16,
        warmup: bool = true,

        pub fn defaultOptions() Options {
            return .{ .model_path = "" };
        }
    };

    pub const GenerateOptions = struct {
        sampling: backend.sampling.SamplingOptions = gemma4.default_generation,
        max_new_tokens: usize = 256,
        stop_tokens: []const usize = &gemma4_e4b.Gemma4E4BModel.default_stop_tokens,
        enable_speculative: bool = true,
        max_seq_len: usize = 2048,
        apply_chat_template: bool = true,
        verbose: bool = false,

        pub fn defaultOptions() GenerateOptions {
            return .{};
        }
    };

    /// 统一加载入口：自动嗅探路径架构并初始化模型与词表
    pub fn load(allocator: std.mem.Allocator, io: std.Io, options: Options) !LLMPipeline {
        var timer_load: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_load);
        const load_start_ns = @as(u64, @intCast(timer_load.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_load.nsec));

        const arch = try detectModelArchitecture(io, options.model_path);

        var mmap_ptr: ?[]const u8 = null;
        var model: AnyModel = undefined;

        switch (arch) {
            .gemma4_e4b => {
                const m = try gemma4_e4b.Gemma4E4BModel.loadFromDirectoryWithOptions(
                    allocator,
                    io,
                    options.model_path,
                    .{ .warmup = options.warmup },
                );
                model = .{ .gemma4_e4b = m };
            },
            .gemma4_12b_q4 => {
                const cwd = std.Io.Dir.cwd();
                var file = try cwd.openFile(io, options.model_path, .{});
                defer file.close(io);

                const stat = try file.stat(io);
                const bytes = try std.posix.mmap(
                    null,
                    @as(usize, @intCast(stat.size)),
                    .{ .READ = true },
                    .{ .TYPE = .SHARED },
                    file.handle,
                    0,
                );
                errdefer std.posix.munmap(bytes);
                mmap_ptr = bytes;

                const m = try gemma4_q4.DefaultGemma4Q4.loadFromMmapWithOptions(allocator, bytes, .{
                    .lm_head_mode = options.lm_head_mode,
                    .warmup = options.warmup,
                });
                model = .{ .gemma4_12b_q4 = m };
            },
        }

        // 尝试从 options.vocab_path 打开词表，若失败则在 model_path 目录下寻找
        var vocab = gemma4.Gemma4Vocabulary.openFile(io, options.vocab_path) catch |err| blk: {
            var buf: [1024]u8 = undefined;
            const fallback = std.fmt.bufPrint(&buf, "{s}/{s}", .{ options.model_path, options.vocab_path }) catch return err;
            break :blk try gemma4.Gemma4Vocabulary.openFile(io, fallback);
        };
        errdefer vocab.deinit();
        try vocab.buildLookupTable(allocator);

        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_load);
        const load_end_ns = @as(u64, @intCast(timer_load.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_load.nsec));
        const load_time_s = @as(f64, @floatFromInt(load_end_ns - load_start_ns)) / 1e9;

        return .{
            .allocator = allocator,
            .model = model,
            .vocab = vocab,
            .load_time_s = load_time_s,
            .mmap_ptr = mmap_ptr,
        };
    }

    pub fn deinit(self: *LLMPipeline) void {
        self.model.deinit(self.allocator);
        self.vocab.deinit();
        if (self.mmap_ptr) |ptr| {
            std.posix.munmap(@alignCast(ptr));
        }
    }

    /// 创建流式生成会话 (TextStream)
    pub fn stream(self: *LLMPipeline, prompt: []const u8, options: GenerateOptions) !TextStream {
        const prompt_tokens = if (options.apply_chat_template)
            try self.vocab.formatInstructPrompt(self.allocator, prompt)
        else
            try self.vocab.encode(self.allocator, prompt);
        errdefer self.allocator.free(prompt_tokens);

        var timer_ts: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
        const start_ns = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));

        const generation: AnyGeneration = switch (self.model) {
            .gemma4_e4b => |*m| .{
                .gemma4_e4b = try m.generate(self.allocator, prompt_tokens, .{
                    .sampling = options.sampling,
                    .max_new_tokens = options.max_new_tokens,
                    .stop_tokens = options.stop_tokens,
                }),
            },
            .gemma4_12b_q4 => |*m| .{
                .gemma4_12b_q4 = try m.generate(self.allocator, prompt_tokens, .{
                    .sampling = options.sampling,
                    .max_new_tokens = options.max_new_tokens,
                    .stop_tokens = options.stop_tokens,
                    .enable_speculative = options.enable_speculative,
                    .max_seq_len = options.max_seq_len,
                    .mmap_bytes = self.mmap_ptr,
                }),
            },
        };

        return .{
            .pipeline = self,
            .generation = generation,
            .prompt_tokens = prompt_tokens,
            .start_ns = start_ns,
        };
    }

    /// 便捷流式打印：实时将生成的文本输出到标准输出，并可选打印性能报告
    pub fn streamPrint(
        self: *LLMPipeline,
        prompt: []const u8,
        options: GenerateOptions,
    ) !GenerationStats {
        var s = try self.stream(prompt, options);
        defer s.deinit();

        while (try s.next()) |chunk| {
            if (chunk.len > 0) {
                std.debug.print("{s}", .{chunk});
            }
        }

        const gen_stats = s.stats();
        if (options.verbose) {
            gen_stats.printReport();
        }
        return gen_stats;
    }

    /// 端到端文本生成 (Text-in -> Text-out)：一次性返回完整响应字符串
    pub fn generateText(
        self: *LLMPipeline,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        options: GenerateOptions,
    ) !TextResult {
        var s = try self.stream(prompt, options);
        defer s.deinit();

        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(allocator);

        while (try s.next()) |chunk| {
            try buf.appendSlice(allocator, chunk);
        }

        const gen_stats = s.stats();
        if (options.verbose) {
            gen_stats.printReport();
        }
        return .{
            .text = try buf.toOwnedSlice(allocator),
            .stats = gen_stats,
        };
    }
};
