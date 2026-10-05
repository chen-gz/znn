const std = @import("std");
const znn = @import("zig_ml");

const Q4Block = znn.Q4Block;
const bf16 = znn.tensor.bf16;

const MAGIC = "ZNNQ4G01"; // 8 bytes magic header

fn getTensorRange(meta_obj: std.json.ObjectMap, name: []const u8) !struct { start: usize, end: usize, shape: []const usize } {
    const v = meta_obj.get(name) orelse return error.TensorNotFound;
    if (v != .object) return error.InvalidHeader;
    const offsets_v = v.object.get("data_offsets") orelse return error.InvalidHeader;
    if (offsets_v != .array or offsets_v.array.items.len != 2) return error.InvalidHeader;
    const start: usize = @intCast(offsets_v.array.items[0].integer);
    const end: usize = @intCast(offsets_v.array.items[1].integer);
    return .{ .start = start, .end = end, .shape = &.{} };
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = std.heap.page_allocator;
    const model_path = "gemma-4-12B/model.safetensors";
    const out_path = "gemma-4-12B-q4.bin";

    std.debug.print("======================================================================\n", .{});
    std.debug.print("🚀 ZNN Gemma 4 12B -> 4-bit (Q4) Streaming Quantization Tool\n", .{});
    std.debug.print("======================================================================\n\n", .{});

    const cwd = std.Io.Dir.cwd();
    var in_file = try cwd.openFile(io, model_path, .{});
    defer in_file.close(io);

    const stat = try in_file.stat(io);
    const total_bytes = @as(usize, @intCast(stat.size));
    std.debug.print("1. Opening Safetensors file: {s} ({d:.2} GB)\n", .{
        model_path, @as(f64, @floatFromInt(total_bytes)) / (1024.0 * 1024.0 * 1024.0),
    });

    const mmap_ptr = try std.posix.mmap(
        null,
        total_bytes,
        .{ .READ = true },
        .{ .TYPE = .SHARED },
        in_file.handle,
        0,
    );
    defer std.posix.munmap(mmap_ptr);

    const header_len = std.mem.readInt(u64, mmap_ptr[0..8], .little);
    const header_slice = mmap_ptr[8 .. 8 + header_len];
    const data_start = 8 + header_len;
    const payload = mmap_ptr[data_start..];

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_slice, .{});
    defer parsed.deinit();

    const meta = parsed.value.object;
    std.debug.print("2. Parsed Header metadata successfully (Total tensors: {})\n", .{meta.count() - 1});

    var out_file = try cwd.createFile(io, out_path, .{});
    defer out_file.close(io);

    var file_buf: [65536]u8 = undefined;
    var f_writer = out_file.writer(io, &file_buf);
    const writer = &f_writer.interface;

    try writer.writeAll(MAGIC);

    // 写入 norm.weight (3840 floats)
    {
        const norm_meta = try getTensorRange(meta, "model.language_model.norm.weight");
        const norm_bytes = payload[norm_meta.start..norm_meta.end];
        std.debug.print("3. Writing final norm ({d:.2} KB)...\n", .{@as(f64, @floatFromInt(norm_bytes.len)) / 1024.0});
        try writer.writeAll(norm_bytes);
    }

    // 写入 embed_tokens.weight (262144 * 3840 BF16 = 1920 MB)
    {
        const emb_meta = try getTensorRange(meta, "model.language_model.embed_tokens.weight");
        const emb_bytes = payload[emb_meta.start..emb_meta.end];
        std.debug.print("4. Streaming embed_tokens ({d:.2} MB)...\n", .{@as(f64, @floatFromInt(emb_bytes.len)) / (1024.0 * 1024.0)});
        try writer.writeAll(emb_bytes);
        try writer.flush();
    }

    std.debug.print("\n5. Quantizing 48 Transformer Layers to Q4 (Streaming)...\n", .{});
    var f32_chunk: [32]f32 = undefined;

    for (0..48) |layer_idx| {
        const is_full = ((layer_idx + 1) % 6 == 0);
        std.debug.print("   -> Layer {:2}/48 [{s}]...\n", .{
            layer_idx + 1, if (is_full) "Global" else "Sliding",
        });

        // 写入当前层 Layernorms & scalars (BF16 raw)
        const bf16_names = [_][]const u8{
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "layer_scalar",
        };

        var key_buf: [128]u8 = undefined;
        for (bf16_names) |name| {
            const key = try std.fmt.bufPrint(&key_buf, "model.language_model.layers.{}.{s}", .{ layer_idx, name });
            const r = try getTensorRange(meta, key);
            try writer.writeAll(payload[r.start..r.end]);
        }

        // 写入需要量化为 Q4 的投影矩阵
        // q_proj, k_proj, (v_proj), o_proj, gate_proj, up_proj, down_proj
        const full_names: []const []const u8 = &.{
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        };
        const sliding_names: []const []const u8 = &.{
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        };
        const q_names = if (is_full) full_names else sliding_names;

        for (q_names) |name| {
            const key = try std.fmt.bufPrint(&key_buf, "model.language_model.layers.{}.{s}", .{ layer_idx, name });
            const r = try getTensorRange(meta, key);
            const raw_bytes = payload[r.start..r.end];
            const u16_slice: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw_bytes));
            const num_blocks = u16_slice.len / Q4Block.QK;

            for (0..num_blocks) |b| {
                for (0..32) |j| {
                    const b_bits = u16_slice[b * 32 + j];
                    const val: bf16 = .{ .bits = b_bits };
                    f32_chunk[j] = val.toF32();
                }
                const block = Q4Block.quantize(&f32_chunk);
                try writer.writeAll(std.mem.asBytes(&block));
            }
        }
        try writer.flush();
    }

    std.debug.print("\n✨ Quantization finished! Saved to {s}\n", .{out_path});
}
