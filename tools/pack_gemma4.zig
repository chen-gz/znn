const std = @import("std");
const znn = @import("zig_ml");

const Q4Weights = znn.backend.offload.Q4Weights;
const DefaultGemma4Q4 = znn.models.gemma4_q4.DefaultGemma4Q4;
const config = DefaultGemma4Q4.config;

const MAGIC_IN = "ZNNQ4G02";
const MAGIC_OUT = "ZNNQ4DIR";

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = std.heap.page_allocator;
    const in_path = "gemma-4-12B-q4.bin";
    const out_path = "gemma-4-12B-direct.bin";

    std.debug.print("======================================================================\n", .{});
    std.debug.print("⚡ ZNN Gemma 4 12B Direct Layout Multi-Threaded Pack Tool\n", .{});
    std.debug.print("======================================================================\n\n", .{});

    var timer_ts: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
    const start_ns = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));

    const cwd = std.Io.Dir.cwd();
    var in_file = cwd.openFile(io, in_path, .{}) catch |err| {
        std.debug.print("❌ 无法打开输入文件 {s}: {s}\n", .{ in_path, @errorName(err) });
        return;
    };
    defer in_file.close(io);

    const in_stat = try in_file.stat(io);
    const in_total_bytes = @as(usize, @intCast(in_stat.size));
    const expected_in_bytes = DefaultGemma4Q4.computeExpectedFileSize();
    if (in_total_bytes < expected_in_bytes) {
        std.debug.print("❌ 错误: 输入模型文件不完整 (大小: {} 字节, 期望: {} 字节)\n", .{ in_total_bytes, expected_in_bytes });
        return;
    }

    std.debug.print("1. [映射输入模型文件]: {s} ({d:.2} GB)\n", .{
        in_path, @as(f64, @floatFromInt(in_total_bytes)) / (1024.0 * 1024.0 * 1024.0),
    });

    const in_mmap_ptr = try std.posix.mmap(
        null,
        in_total_bytes,
        .{ .READ = true },
        .{ .TYPE = .SHARED },
        in_file.handle,
        0,
    );
    defer std.posix.munmap(in_mmap_ptr);

    if (!std.mem.eql(u8, in_mmap_ptr[0..8], MAGIC_IN)) {
        std.debug.print("❌ 错误: 输入文件魔数无效: {s} (期望: {s})\n", .{ in_mmap_ptr[0..8], MAGIC_IN });
        return;
    }

    const total_direct_bytes = DefaultGemma4Q4.computeExpectedDirectFileSize();
    std.debug.print("2. [创建直通目标文件]: {s} ({d:.2} GB)\n", .{
        out_path, @as(f64, @floatFromInt(total_direct_bytes)) / (1024.0 * 1024.0 * 1024.0),
    });

    var out_file = try cwd.createFile(io, out_path, .{ .truncate = true, .read = true });
    defer out_file.close(io);

    // 预分配磁盘空间，支持零缓冲内存直接映射
    try out_file.setLength(io, total_direct_bytes);

    const out_mmap_ptr = try std.posix.mmap(
        null,
        total_direct_bytes,
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .SHARED },
        out_file.handle,
        0,
    );
    defer std.posix.munmap(out_mmap_ptr);

    // 写入格式魔数签名
    @memcpy(out_mmap_ptr[0..8], MAGIC_OUT);

    // 复制最终层 RMSNorm 权重 (3840 BF16 = 7680 bytes)
    const norm_bytes_len: usize = config.hidden_size * 2;
    @memcpy(out_mmap_ptr[8 .. 8 + norm_bytes_len], in_mmap_ptr[8 .. 8 + norm_bytes_len]);

    // 复制词表嵌入矩阵 (262144 * 3840 BF16 = 2,013,265,920 bytes)
    const embed_bytes_len: usize = config.vocab_size * config.hidden_size * 2;
    std.debug.print("3. [零拷贝复制词表嵌入矩阵] ({d:.2} MB)...\n", .{
        @as(f64, @floatFromInt(embed_bytes_len)) / (1024.0 * 1024.0),
    });
    @memcpy(out_mmap_ptr[8 + norm_bytes_len .. 8 + norm_bytes_len + embed_bytes_len], in_mmap_ptr[8 + norm_bytes_len .. 8 + norm_bytes_len + embed_bytes_len]);

    std.debug.print("4. [多核多线程并发解包与直接内存重排] (48 层 Transformer)...\n", .{});

    const cpu_count = @max(1, std.Thread.getCpuCount() catch 1);
    const num_threads = @min(cpu_count, config.num_hidden_layers);

    const LayerOffsets = struct {
        in_offsets: [49]usize,
        out_offsets: [49]usize,
    };

    const computeLayerOffsets = struct {
        fn compute() LayerOffsets {
            var res: LayerOffsets = undefined;
            const base = 8 + config.hidden_size * 2 + config.vocab_size * config.hidden_size * 2;
            res.in_offsets[0] = base;
            res.out_offsets[0] = base;

            for (0..config.num_hidden_layers) |i| {
                const is_full = ((i + 1) % 6 == 0);
                const head_dim = if (is_full) config.global_head_dim else config.head_dim;
                const num_kv = if (is_full) config.num_global_key_value_heads else config.num_key_value_heads;
                const q_dim = config.num_attention_heads * head_dim;
                const kv_dim = num_kv * head_dim;
                const v_dim = if (is_full) 0 else kv_dim;

                const norms_len = (config.hidden_size * 4 + head_dim * 2 + 1) * 2;
                const q_blocks = (q_dim * config.hidden_size) / 32;
                const k_blocks = (kv_dim * config.hidden_size) / 32;
                const v_blocks = (v_dim * config.hidden_size) / 32;
                const o_blocks = (config.hidden_size * q_dim) / 32;
                const gate_blocks = (config.intermediate_size * config.hidden_size) / 32;
                const up_blocks = (config.intermediate_size * config.hidden_size) / 32;
                const down_blocks = (config.hidden_size * config.intermediate_size) / 32;

                const raw_layer_bytes = norms_len + (q_blocks + k_blocks + v_blocks + o_blocks + gate_blocks + up_blocks + down_blocks) * 18;

                const qkv_direct_blocks = ((q_dim + kv_dim + v_dim) * config.hidden_size) / 32;
                const direct_layer_bytes = norms_len + (qkv_direct_blocks + o_blocks + gate_blocks * 2 + down_blocks) * 20;

                res.in_offsets[i + 1] = res.in_offsets[i] + raw_layer_bytes;
                res.out_offsets[i + 1] = res.out_offsets[i] + direct_layer_bytes;
            }
            return res;
        }
    }.compute;

    const Worker = struct {
        in_mmap: []const u8,
        out_mmap: []u8,
        start_layer: usize,
        end_layer: usize,

        fn run(ctx: *@This()) void {
            const offsets = computeLayerOffsets();

            for (ctx.start_layer..ctx.end_layer) |i| {
                const is_full = ((i + 1) % 6 == 0);
                const head_dim = if (is_full) config.global_head_dim else config.head_dim;
                const num_kv = if (is_full) config.num_global_key_value_heads else config.num_key_value_heads;
                const q_dim = config.num_attention_heads * head_dim;
                const kv_dim = num_kv * head_dim;
                const v_dim = if (is_full) 0 else kv_dim;

                const in_cursor_start = offsets.in_offsets[i];
                const out_cursor_start = offsets.out_offsets[i];
                const norms_bytes_len = (config.hidden_size * 4 + head_dim * 2 + 1) * 2;

                // 1. 复制 Layernorm 与 scalar 权重
                @memcpy(
                    ctx.out_mmap[out_cursor_start .. out_cursor_start + norms_bytes_len],
                    ctx.in_mmap[in_cursor_start .. in_cursor_start + norms_bytes_len],
                );

                var in_c = in_cursor_start + norms_bytes_len;
                var out_c = out_cursor_start + norms_bytes_len;

                // 2. 重排并写入 fused_qkv (w_q, scales, biases)
                const q_num_blocks = (q_dim * config.hidden_size) / 32;
                const k_num_blocks = (kv_dim * config.hidden_size) / 32;

                const q_slice = ctx.in_mmap[in_c .. in_c + q_num_blocks * 18]; in_c += q_num_blocks * 18;
                const k_slice = ctx.in_mmap[in_c .. in_c + k_num_blocks * 18]; in_c += k_num_blocks * 18;

                const fused_qkv_blocks = ((q_dim + kv_dim + v_dim) * config.hidden_size) / 32;
                const qkv_wq_len = fused_qkv_blocks * 16;
                const qkv_sc_len = fused_qkv_blocks * 2;
                const qkv_bi_len = fused_qkv_blocks * 2;

                const qkv_wq_out = @as([]u32, @alignCast(std.mem.bytesAsSlice(u32, ctx.out_mmap[out_c .. out_c + qkv_wq_len])));
                const qkv_sc_out = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, ctx.out_mmap[out_c + qkv_wq_len .. out_c + qkv_wq_len + qkv_sc_len])));
                const qkv_bi_out = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, ctx.out_mmap[out_c + qkv_wq_len + qkv_sc_len .. out_c + qkv_wq_len + qkv_sc_len + qkv_bi_len])));

                if (is_full) {
                    Q4Weights.packBlocksToDirectBuffers(&.{ q_slice, k_slice }, &.{ q_dim, kv_dim }, config.hidden_size, qkv_wq_out, qkv_sc_out, qkv_bi_out);
                } else {
                    const v_num_blocks = (v_dim * config.hidden_size) / 32;
                    const v_slice = ctx.in_mmap[in_c .. in_c + v_num_blocks * 18]; in_c += v_num_blocks * 18;
                    Q4Weights.packBlocksToDirectBuffers(&.{ q_slice, k_slice, v_slice }, &.{ q_dim, kv_dim, kv_dim }, config.hidden_size, qkv_wq_out, qkv_sc_out, qkv_bi_out);
                }
                out_c += qkv_wq_len + qkv_sc_len + qkv_bi_len;

                // 3. 重排并写入 o_proj
                const o_num_blocks = (config.hidden_size * q_dim) / 32;
                const o_slice = ctx.in_mmap[in_c .. in_c + o_num_blocks * 18]; in_c += o_num_blocks * 18;

                const o_wq_len = o_num_blocks * 16;
                const o_sc_len = o_num_blocks * 2;
                const o_bi_len = o_num_blocks * 2;

                const o_wq_out = @as([]u32, @alignCast(std.mem.bytesAsSlice(u32, ctx.out_mmap[out_c .. out_c + o_wq_len])));
                const o_sc_out = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, ctx.out_mmap[out_c + o_wq_len .. out_c + o_wq_len + o_sc_len])));
                const o_bi_out = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, ctx.out_mmap[out_c + o_wq_len + o_sc_len .. out_c + o_wq_len + o_sc_len + o_bi_len])));

                Q4Weights.packBlocksToDirectBuffers(&.{o_slice}, &.{config.hidden_size}, q_dim, o_wq_out, o_sc_out, o_bi_out);
                out_c += o_wq_len + o_sc_len + o_bi_len;

                // 4. 重排并写入 fused_gate_up
                const gate_up_single_blocks = (config.intermediate_size * config.hidden_size) / 32;
                const gate_slice = ctx.in_mmap[in_c .. in_c + gate_up_single_blocks * 18];
                const up_slice = ctx.in_mmap[in_c + gate_up_single_blocks * 18 .. in_c + gate_up_single_blocks * 2 * 18];
                in_c += gate_up_single_blocks * 2 * 18;

                const fgu_blocks = gate_up_single_blocks * 2;
                const fgu_wq_len = fgu_blocks * 16;
                const fgu_sc_len = fgu_blocks * 2;
                const fgu_bi_len = fgu_blocks * 2;

                const fgu_wq_out = @as([]u32, @alignCast(std.mem.bytesAsSlice(u32, ctx.out_mmap[out_c .. out_c + fgu_wq_len])));
                const fgu_sc_out = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, ctx.out_mmap[out_c + fgu_wq_len .. out_c + fgu_wq_len + fgu_sc_len])));
                const fgu_bi_out = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, ctx.out_mmap[out_c + fgu_wq_len + fgu_sc_len .. out_c + fgu_wq_len + fgu_sc_len + fgu_bi_len])));

                Q4Weights.packBlocksToDirectBuffers(&.{ gate_slice, up_slice }, &.{ config.intermediate_size, config.intermediate_size }, config.hidden_size, fgu_wq_out, fgu_sc_out, fgu_bi_out);
                out_c += fgu_wq_len + fgu_sc_len + fgu_bi_len;

                // 5. 重排并写入 down_proj
                const down_blocks = (config.hidden_size * config.intermediate_size) / 32;
                const down_slice = ctx.in_mmap[in_c .. in_c + down_blocks * 18]; in_c += down_blocks * 18;

                const down_wq_len = down_blocks * 16;
                const down_sc_len = down_blocks * 2;
                const down_bi_len = down_blocks * 2;

                const down_wq_out = @as([]u32, @alignCast(std.mem.bytesAsSlice(u32, ctx.out_mmap[out_c .. out_c + down_wq_len])));
                const down_sc_out = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, ctx.out_mmap[out_c + down_wq_len .. out_c + down_wq_len + down_sc_len])));
                const down_bi_out = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, ctx.out_mmap[out_c + down_wq_len + down_sc_len .. out_c + down_wq_len + down_sc_len + down_bi_len])));

                Q4Weights.packBlocksToDirectBuffers(&.{down_slice}, &.{config.hidden_size}, config.intermediate_size, down_wq_out, down_sc_out, down_bi_out);
                out_c += down_wq_len + down_sc_len + down_bi_len;
            }
        }
    };

    var workers = try allocator.alloc(Worker, num_threads);
    defer allocator.free(workers);
    var threads = try allocator.alloc(std.Thread, num_threads);
    defer allocator.free(threads);

    const chunk = (config.num_hidden_layers + num_threads - 1) / num_threads;
    var spawned: usize = 0;

    for (0..num_threads) |t_idx| {
        const s = t_idx * chunk;
        const e = @min(s + chunk, config.num_hidden_layers);
        if (s >= e) {
            workers[t_idx] = .{ .in_mmap = in_mmap_ptr, .out_mmap = out_mmap_ptr, .start_layer = 0, .end_layer = 0 };
            continue;
        }
        workers[t_idx] = .{ .in_mmap = in_mmap_ptr, .out_mmap = out_mmap_ptr, .start_layer = s, .end_layer = e };
        threads[spawned] = std.Thread.spawn(.{}, Worker.run, .{&workers[t_idx]}) catch {
            workers[t_idx].run();
            continue;
        };
        spawned += 1;
    }

    for (threads[0..spawned]) |t| {
        t.join();
    }

    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
    const end_ns = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));
    const elapsed_s = @as(f64, @floatFromInt(end_ns - start_ns)) / 1e9;

    std.debug.print("\n✨ 预重排完成！总耗时: {d:.2} 秒。\n", .{elapsed_s});
    std.debug.print("👉 已生成零拷贝直通文件: {s} ({d:.2} GB)\n", .{
        out_path, @as(f64, @floatFromInt(total_direct_bytes)) / (1024.0 * 1024.0 * 1024.0),
    });
    std.debug.print("👉 现在运行 `zig build run-gemma4` 或 `./zig-out/bin/gemma4` 将自动启用 1.5 秒极速秒开！\n\n", .{});
}
