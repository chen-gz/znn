//! LLM 推理性能统计与基准报告模块 (Generation Benchmark & Performance Stats)

const std = @import("std");

/// 端到端文本生成的性能基准测试数据
pub const GenerationStats = struct {
    load_time_s: f64 = 0.0,
    prefill_time_ms: f64 = 0.0,
    decode_time_s: f64 = 0.0,
    total_time_s: f64 = 0.0,
    prompt_tokens: usize = 0,
    generated_tokens: usize = 0,
    prefill_tps: f64 = 0.0,
    decode_tps: f64 = 0.0,
    is_pipelined: bool = false,
    speculative_hits: usize = 0,

    /// 格式化打印与官方基准完全对齐的性能报告
    pub fn printReport(self: GenerationStats) void {
        std.debug.print(
            \\================================================================================
            \\📊 LLM 推理性能基准测试报告 (Performance Benchmark):
            \\================================================================================
            \\   - 模型冷启动加载耗时 : {d:.3} 秒 (零拷贝 mmap 直通)
            \\   - Prompt 预填充耗时  : {d:.2} 毫秒 (TTFT 首字延迟)
            \\   - 预填充吞吐速率     : {d:.1} token/s
            \\   - 生成词元总数       : {} tokens
            \\   - 自回归解码速率     : {d:.2} token/s ({s})
            \\   - 端到端总生成耗时   : {d:.2} 秒
        , .{
            self.load_time_s,
            self.prefill_time_ms,
            self.prefill_tps,
            self.generated_tokens,
            self.decode_tps,
            if (self.is_pipelined) "纯 Metal GPU 双阶段异步流水线" else "同步解码 + 设备端候选选择",
            self.total_time_s,
        });

        if (self.speculative_hits > 0) {
            std.debug.print(
                \\
                \\   - 投机采样加速       : 命中 {} 个推测词元
            , .{self.speculative_hits});
        }

        std.debug.print(
            \\
            \\================================================================================
            \\
        , .{});
    }
};

/// 一次性完整文本生成的结果
pub const TextResult = struct {
    text: []const u8,
    stats: GenerationStats,

    pub fn deinit(self: *TextResult, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
    }
};
