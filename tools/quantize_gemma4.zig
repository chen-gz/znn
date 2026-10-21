const std = @import("std");
const znn = @import("zig_ml");

const Q4Block = znn.tensor.quant.Q4Block;
const bf16 = znn.tensor.bf16;
const SafetensorsArchive = znn.nn.SafetensorsArchive;

const MAGIC = "ZNNQ4G02"; // 8 bytes magic header

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = std.heap.page_allocator;
    const model_path = "gemma-4-12B/model.safetensors";
    const out_path = "gemma-4-12B-q4.bin";

    std.debug.print("======================================================================\n", .{});
    std.debug.print("🚀 ZNN Gemma 4 12B -> 4-bit (Q4) Streaming Quantization Tool\n", .{});
    std.debug.print("======================================================================\n\n", .{});

    std.debug.print("1. Opening Safetensors file: {s}\n", .{model_path});
    var archive = try SafetensorsArchive.openSingleFile(allocator, io, model_path);
    defer archive.deinit();

    std.debug.print("2. Parsed Header metadata successfully (Total tensors: {})\n", .{archive.tensors.count()});

    const cwd = std.Io.Dir.cwd();
    var out_file = try cwd.createFile(io, out_path, .{});
    defer out_file.close(io);

    var file_buf: [65536]u8 = undefined;
    var f_writer = out_file.writer(io, &file_buf);
    const writer = &f_writer.interface;

    try writer.writeAll(MAGIC);

    // 写入 norm.weight (3840 floats)
    {
        const norm_bytes = archive.getTensorBytes("model.language_model.norm.weight") orelse return error.TensorNotFound;
        std.debug.print("3. Writing final norm ({d:.2} KB)...\n", .{@as(f64, @floatFromInt(norm_bytes.len)) / 1024.0});
        try writer.writeAll(norm_bytes);
    }

    // 写入 embed_tokens.weight (262144 * 3840 BF16 = 1920 MB)
    {
        const emb_bytes = archive.getTensorBytes("model.language_model.embed_tokens.weight") orelse return error.TensorNotFound;
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
            const raw_bytes = archive.getTensorBytes(key) orelse return error.TensorNotFound;
            try writer.writeAll(raw_bytes);
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
            const raw_bytes = archive.getTensorBytes(key) orelse return error.TensorNotFound;
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
