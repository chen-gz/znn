const std = @import("std");
const znn = @import("zig_ml");

const Q4Block = znn.tensor.quant.Q4Block;
const bf16 = znn.tensor.bf16;
const SafetensorsArchive = znn.nn.SafetensorsArchive;

const DefaultGemma4Q4 = znn.models.gemma4_q4.DefaultGemma4Q4;
const config = DefaultGemma4Q4.config;
const MAGIC = DefaultGemma4Q4.Q4_MAGIC; // 8 bytes magic header

/// 读取指定张量并校验其 BF16 字节长度，防止数据类型或形状不符时静默写出错位的二进制布局
fn getBf16Tensor(archive: *const SafetensorsArchive, key: []const u8, numel: usize) ![]const u8 {
    const raw_bytes = archive.getTensorBytes(key) orelse return error.TensorNotFound;
    if (raw_bytes.len != numel * 2) {
        std.debug.print("❌ 张量 {s} 字节长度 {} 与期望的 BF16 [{}] ({} 字节) 不符\n", .{ key, raw_bytes.len, numel, numel * 2 });
        return error.UnexpectedTensorShape;
    }
    return raw_bytes;
}

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
        const norm_bytes = try getBf16Tensor(&archive, "model.language_model.norm.weight", config.hidden_size);
        std.debug.print("3. Writing final norm ({d:.2} KB)...\n", .{@as(f64, @floatFromInt(norm_bytes.len)) / 1024.0});
        try writer.writeAll(norm_bytes);
    }

    // 写入 embed_tokens.weight (262144 * 3840 BF16 = 1920 MB)
    {
        const emb_bytes = try getBf16Tensor(&archive, "model.language_model.embed_tokens.weight", config.vocab_size * config.hidden_size);
        std.debug.print("4. Streaming embed_tokens ({d:.2} MB)...\n", .{@as(f64, @floatFromInt(emb_bytes.len)) / (1024.0 * 1024.0)});
        try writer.writeAll(emb_bytes);
        try writer.flush();
    }

    std.debug.print("\n5. Quantizing {} Transformer Layers to Q4 (Streaming)...\n", .{config.num_hidden_layers});
    var f32_chunk: [32]f32 = undefined;

    for (0..config.num_hidden_layers) |layer_idx| {
        const spec = config.layerSpec(layer_idx);
        const is_full = spec.attn_type == .full_attention;
        std.debug.print("   -> Layer {:2}/{} [{s}]...\n", .{
            layer_idx + 1, config.num_hidden_layers, if (is_full) "Global" else "Sliding",
        });
        const H = config.hidden_size;
        const q_dim = config.num_attention_heads * spec.head_dim;
        const kv_dim = spec.num_kv_heads * spec.head_dim;
        const I = config.intermediate_size;

        // 写入当前层 Layernorms & scalars (BF16 raw)
        const TensorSpec = struct { name: []const u8, numel: usize };
        const bf16_specs = [_]TensorSpec{
            .{ .name = "input_layernorm.weight", .numel = H },
            .{ .name = "post_attention_layernorm.weight", .numel = H },
            .{ .name = "pre_feedforward_layernorm.weight", .numel = H },
            .{ .name = "post_feedforward_layernorm.weight", .numel = H },
            .{ .name = "self_attn.q_norm.weight", .numel = spec.head_dim },
            .{ .name = "self_attn.k_norm.weight", .numel = spec.head_dim },
            .{ .name = "layer_scalar", .numel = 1 },
        };

        var key_buf: [128]u8 = undefined;
        for (bf16_specs) |ts| {
            const key = try std.fmt.bufPrint(&key_buf, "model.language_model.layers.{}.{s}", .{ layer_idx, ts.name });
            try writer.writeAll(try getBf16Tensor(&archive, key, ts.numel));
        }

        // 写入需要量化为 Q4 的投影矩阵
        // q_proj, k_proj, (v_proj), o_proj, gate_proj, up_proj, down_proj
        // 全局注意力层 attention_k_eq_v 为 true，不存储独立的 v_proj
        const full_specs = [_]TensorSpec{
            .{ .name = "self_attn.q_proj.weight", .numel = q_dim * H },
            .{ .name = "self_attn.k_proj.weight", .numel = kv_dim * H },
            .{ .name = "self_attn.o_proj.weight", .numel = H * q_dim },
            .{ .name = "mlp.gate_proj.weight", .numel = I * H },
            .{ .name = "mlp.up_proj.weight", .numel = I * H },
            .{ .name = "mlp.down_proj.weight", .numel = H * I },
        };
        const sliding_specs = [_]TensorSpec{
            .{ .name = "self_attn.q_proj.weight", .numel = q_dim * H },
            .{ .name = "self_attn.k_proj.weight", .numel = kv_dim * H },
            .{ .name = "self_attn.v_proj.weight", .numel = kv_dim * H },
            .{ .name = "self_attn.o_proj.weight", .numel = H * q_dim },
            .{ .name = "mlp.gate_proj.weight", .numel = I * H },
            .{ .name = "mlp.up_proj.weight", .numel = I * H },
            .{ .name = "mlp.down_proj.weight", .numel = H * I },
        };
        const q_specs: []const TensorSpec = if (is_full) &full_specs else &sliding_specs;

        for (q_specs) |ts| {
            std.debug.assert(ts.numel % Q4Block.QK == 0);
            const key = try std.fmt.bufPrint(&key_buf, "model.language_model.layers.{}.{s}", .{ layer_idx, ts.name });
            const raw_bytes = try getBf16Tensor(&archive, key, ts.numel);
            const num_blocks = ts.numel / Q4Block.QK;

            for (0..num_blocks) |b| {
                for (0..32) |j| {
                    const b_bits = std.mem.readInt(u16, raw_bytes[(b * 32 + j) * 2 ..][0..2], .little);
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
