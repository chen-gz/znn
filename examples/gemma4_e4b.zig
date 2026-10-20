//! Gemma 4 E4B (Effective 4B) 4-bit / 8-bit 混合量化模型直通推理与性能评测演示
//!
//! 支持直接零拷贝映射 Safetensors 模型文件、逐层嵌入增强 (Per-Layer Embeddings, PLE)、
//! 异构滑动窗口与全局注意力机制，以及苹果统一内存 (Unified Memory Architecture, UMA) GPU 极速硬件加速。

const std = @import("std");
const zig_ml = @import("zig_ml");

const backend = zig_ml.backend;
const mlx = backend.mlx;
const gemma4 = zig_ml.models.gemma4;
const Gemma4Config = gemma4.Gemma4Config;
const Q4Weights = backend.offload.Q4Weights;
const Stream = backend.offload.Stream;

/// Safetensors 格式中的张量元数据
pub const TensorMeta = struct {
    file_idx: u8, // 1 或 2
    start: usize,
    end: usize,
    shape: [8]usize,
    shape_len: u8,
    dtype: mlx.DType,
};

/// Safetensors 文件头解析与内存映射句柄
pub const SafetensorsArchive = struct {
    allocator: std.mem.Allocator,
    mmap1: []const u8,
    mmap2: []const u8,
    payload1: []const u8,
    payload2: []const u8,
    tensors: std.StringHashMap(TensorMeta),

    pub fn open(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !SafetensorsArchive {
        const file1_path = try std.fs.path.join(allocator, &.{ dir_path, "model-00001-of-00002.safetensors" });
        defer allocator.free(file1_path);
        const file2_path = try std.fs.path.join(allocator, &.{ dir_path, "model-00002-of-00002.safetensors" });
        defer allocator.free(file2_path);

        const cwd = std.Io.Dir.cwd();
        var f1 = try cwd.openFile(io, file1_path, .{});
        defer f1.close(io);
        const stat1 = try f1.stat(io);
        const mmap1 = try std.posix.mmap(
            null,
            @as(usize, @intCast(stat1.size)),
            .{ .READ = true },
            .{ .TYPE = .SHARED },
            f1.handle,
            0,
        );
        errdefer std.posix.munmap(@alignCast(mmap1));

        var f2 = try cwd.openFile(io, file2_path, .{});
        defer f2.close(io);
        const stat2 = try f2.stat(io);
        const mmap2 = try std.posix.mmap(
            null,
            @as(usize, @intCast(stat2.size)),
            .{ .READ = true },
            .{ .TYPE = .SHARED },
            f2.handle,
            0,
        );
        errdefer std.posix.munmap(@alignCast(mmap2));

        // 建议内核预读 (Kernel Pre-fetching via madvise)
        _ = std.posix.madvise(mmap1.ptr, mmap1.len, std.posix.MADV.WILLNEED) catch {};
        _ = std.posix.madvise(mmap2.ptr, mmap2.len, std.posix.MADV.WILLNEED) catch {};

        var tensors = std.StringHashMap(TensorMeta).init(allocator);
        errdefer tensors.deinit();

        const p1 = try parseHeader(allocator, mmap1, 1, &tensors);
        const p2 = try parseHeader(allocator, mmap2, 2, &tensors);

        return SafetensorsArchive{
            .allocator = allocator,
            .mmap1 = mmap1,
            .mmap2 = mmap2,
            .payload1 = p1,
            .payload2 = p2,
            .tensors = tensors,
        };
    }

    pub fn deinit(self: *SafetensorsArchive) void {
        var it = self.tensors.keyIterator();
        while (it.next()) |k| {
            self.allocator.free(k.*);
        }
        self.tensors.deinit();
        std.posix.munmap(@alignCast(self.mmap1));
        std.posix.munmap(@alignCast(self.mmap2));
    }

    fn parseHeader(
        allocator: std.mem.Allocator,
        mmap_slice: []const u8,
        file_idx: u8,
        tensors: *std.StringHashMap(TensorMeta),
    ) ![]const u8 {
        if (mmap_slice.len < 8) return error.InvalidSafetensorsHeader;
        const header_len = std.mem.readInt(u64, mmap_slice[0..8], .little);
        if (8 + header_len > mmap_slice.len) return error.InvalidSafetensorsHeader;

        const header_bytes = mmap_slice[8 .. 8 + header_len];
        const payload = mmap_slice[8 + header_len ..];

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_bytes, .{});
        defer parsed.deinit();

        if (parsed.value != .object) return error.InvalidSafetensorsHeader;
        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
            if (entry.value_ptr.* != .object) continue;
            const obj = entry.value_ptr.*.object;

            const offsets_v = obj.get("data_offsets") orelse continue;
            if (offsets_v != .array or offsets_v.array.items.len != 2) continue;
            const start: usize = @intCast(offsets_v.array.items[0].integer);
            const end: usize = @intCast(offsets_v.array.items[1].integer);

            const dtype_v = obj.get("dtype") orelse continue;
            var dt: mlx.DType = .float32;
            if (dtype_v == .string) {
                if (std.mem.eql(u8, dtype_v.string, "U32")) {
                    dt = .uint32;
                } else if (std.mem.eql(u8, dtype_v.string, "BF16")) {
                    dt = .bfloat16;
                } else if (std.mem.eql(u8, dtype_v.string, "F16")) {
                    dt = .float16;
                } else if (std.mem.eql(u8, dtype_v.string, "F32")) {
                    dt = .float32;
                }
            }

            var shape: [8]usize = undefined;
            var shape_len: u8 = 0;
            if (obj.get("shape")) |sv| {
                if (sv == .array and sv.array.items.len <= 8) {
                    for (sv.array.items, 0..) |dim_v, d_i| {
                        shape[d_i] = @intCast(dim_v.integer);
                    }
                    shape_len = @intCast(sv.array.items.len);
                }
            }

            const key_dup = try allocator.dupe(u8, entry.key_ptr.*);
            try tensors.put(key_dup, .{
                .file_idx = file_idx,
                .start = start,
                .end = end,
                .shape = shape,
                .shape_len = shape_len,
                .dtype = dt,
            });
        }

        return payload;
    }

    pub fn getTensorBytes(self: *const SafetensorsArchive, name: []const u8) ?[]const u8 {
        const meta = self.tensors.get(name) orelse return null;
        const payload = if (meta.file_idx == 1) self.payload1 else self.payload2;
        if (meta.end > payload.len) return null;
        return payload[meta.start..meta.end];
    }

    pub fn getTensorMeta(self: *const SafetensorsArchive, name: []const u8) ?TensorMeta {
        return self.tensors.get(name);
    }
};

/// 逐层嵌入输入结构 (Per-Layer Inputs, PLE)
pub const PleWeights = struct {
    embed_tokens_per_layer: Q4Weights,
    model_proj: Q4Weights,
    proj_norm_w: mlx.Array,

    num_layers: usize,
    ple_dim: usize,
    embed_scale: f32,
    proj_scale: f32,
    input_scale: f32,

    pub fn deinit(self: *PleWeights) void {
        self.embed_tokens_per_layer.deinit();
        self.model_proj.deinit();
        self.proj_norm_w.deinit();
    }
};

/// 单个解码层的 GPU 硬件加速权重结构
pub const GpuLayerE4B = struct {
    // 4-bit 量化注意力投影 (Attention Projections)
    q_proj: Q4Weights,
    k_proj: ?Q4Weights = null,
    v_proj: ?Q4Weights = null,
    o_proj: Q4Weights,

    // 均方根层归一化权重 (RMSNorm Weights)
    input_norm_w: mlx.Array,
    q_norm_w: mlx.Array,
    k_norm_w: ?mlx.Array = null,
    post_attn_w: mlx.Array,
    pre_mlp_w: mlx.Array,
    post_mlp_w: mlx.Array,

    // 8-bit 量化门控前馈多层感知机 (8-bit Quantized Gated MLP Projections)
    gate_proj: Q4Weights,
    up_proj: Q4Weights,
    down_proj: Q4Weights,

    // 逐层嵌入门控与上投影 (Per-Layer Inputs Gating & Projection)
    ple_gate: Q4Weights,
    ple_proj: Q4Weights,
    ple_norm_w: mlx.Array,

    has_kv: bool,
    prev_kv_idx: usize,
    layer_scalar: f32,
    is_sliding: bool,
    sliding_window: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rot_dim: usize,
    rope_theta: f32,
    rms_norm_eps: f32,

    pub fn deinit(self: *GpuLayerE4B) void {
        self.q_proj.deinit();
        if (self.k_proj) |*k| k.deinit();
        if (self.v_proj) |*v| v.deinit();
        self.o_proj.deinit();
        self.input_norm_w.deinit();
        self.q_norm_w.deinit();
        if (self.k_norm_w) |*kn| kn.deinit();
        self.post_attn_w.deinit();
        self.pre_mlp_w.deinit();
        self.post_mlp_w.deinit();
        self.gate_proj.deinit();
        self.up_proj.deinit();
        self.down_proj.deinit();
        self.ple_gate.deinit();
        self.ple_proj.deinit();
        self.ple_norm_w.deinit();
    }
};

/// Gemma 4 E4B 完整模型推理引擎
pub const Gemma4E4BModel = struct {
    allocator: std.mem.Allocator,
    archive: SafetensorsArchive,
    layers: []GpuLayerE4B,
    ple: PleWeights,

    embed_tokens: Q4Weights,
    norm_w: mlx.Array,
    lm_head: Q4Weights,

    config: Gemma4Config,
    stream: Stream,

    // 图形处理器显存键值缓存 (Graphics Processing Unit Key-Value Cache, GPU KVCache)
    gpu_k: []?mlx.Array,
    gpu_v: []?mlx.Array,
    curr_len: usize = 0,
    sqrt_hidden: f32,

    pub fn loadFromDirectory(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !Gemma4E4BModel {
        var archive = try SafetensorsArchive.open(allocator, io, dir_path);
        errdefer archive.deinit();

        const cfg = Gemma4Config.e4b;
        const stream = try Stream.defaultGpu();

        // 1. 词表嵌入与 LM Head 权重
        const emb_wq = archive.getTensorBytes("language_model.model.embed_tokens.weight") orelse return error.TensorNotFound;
        const emb_sc = archive.getTensorBytes("language_model.model.embed_tokens.scales") orelse return error.TensorNotFound;
        const emb_bi = archive.getTensorBytes("language_model.model.embed_tokens.biases") orelse return error.TensorNotFound;

        const embed_tokens = try Q4Weights.initFromDirectBuffersCustom(
            emb_wq,
            emb_sc,
            emb_bi,
            cfg.hidden_size,
            cfg.vocab_size,
            64,
            4,
            .bfloat16,
            stream,
        );

        const norm_bytes = archive.getTensorBytes("language_model.model.norm.weight") orelse return error.TensorNotFound;
        var norm_w = try mlx.Array.fromRawData(@ptrCast(norm_bytes.ptr), &.{cfg.hidden_size}, .bfloat16);
        errdefer norm_w.deinit();

        // 2. 逐层嵌入输入结构 (Per-Layer Embeddings, PLE)
        const ple_tokens_wq = archive.getTensorBytes("language_model.model.embed_tokens_per_layer.weight") orelse return error.TensorNotFound;
        const ple_tokens_sc = archive.getTensorBytes("language_model.model.embed_tokens_per_layer.scales") orelse return error.TensorNotFound;
        const ple_tokens_bi = archive.getTensorBytes("language_model.model.embed_tokens_per_layer.biases") orelse return error.TensorNotFound;

        const ple_tokens = try Q4Weights.initFromDirectBuffersCustom(
            ple_tokens_wq,
            ple_tokens_sc,
            ple_tokens_bi,
            cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
            cfg.vocab_size,
            64,
            4,
            .bfloat16,
            stream,
        );

        const ple_proj_wq = archive.getTensorBytes("language_model.model.per_layer_model_projection.weight") orelse return error.TensorNotFound;
        const ple_proj_sc = archive.getTensorBytes("language_model.model.per_layer_model_projection.scales") orelse return error.TensorNotFound;
        const ple_proj_bi = archive.getTensorBytes("language_model.model.per_layer_model_projection.biases") orelse return error.TensorNotFound;

        const ple_model_proj = try Q4Weights.initFromDirectBuffersCustom(
            ple_proj_wq,
            ple_proj_sc,
            ple_proj_bi,
            cfg.hidden_size,
            cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
            64,
            4,
            .bfloat16,
            stream,
        );

        const ple_pnorm_bytes = archive.getTensorBytes("language_model.model.per_layer_projection_norm.weight") orelse return error.TensorNotFound;
        var ple_pnorm_w = try mlx.Array.fromRawData(@ptrCast(ple_pnorm_bytes.ptr), &.{cfg.hidden_size_per_layer_input}, .bfloat16);
        errdefer ple_pnorm_w.deinit();

        const ple = PleWeights{
            .embed_tokens_per_layer = ple_tokens,
            .model_proj = ple_model_proj,
            .proj_norm_w = ple_pnorm_w,
            .num_layers = cfg.num_hidden_layers,
            .ple_dim = cfg.hidden_size_per_layer_input,
            .embed_scale = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size_per_layer_input))),
            .proj_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.hidden_size))),
            .input_scale = 1.0 / @sqrt(2.0),
        };

        // 3. 构建 42 个解码层
        const layers = try allocator.alloc(GpuLayerE4B, cfg.num_hidden_layers);
        errdefer allocator.free(layers);

        // 预设键值缓存共享映射表 (Key-Value Cache Shared Mapping Table)
        // 当 num_kv_shared_layers = 18 时，前 24 层 (0..23) 具备独立的 K/V 投影；后 18 层 (24..41) 共享复用层 22 (局部滑动窗口) 与层 23 (全局全注意力) 的键值缓存。
        const PREVIOUS_KVS = [42]usize{
            0,  1,  2,  3,  4,  5,
            6,  7,  8,  9,  10, 11,
            12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23,
            22, 22, 22, 22, 22, 23,
            22, 22, 22, 22, 22, 23,
            22, 22, 22, 22, 22, 23,
        };

        for (0..cfg.num_hidden_layers) |idx| {
            var name_buf: [128]u8 = undefined;
            const is_full = ((idx + 1) % 6 == 0);
            const hd = if (is_full) cfg.global_head_dim else cfg.head_dim;
            const out_q = cfg.num_attention_heads * hd;
            const out_kv = cfg.num_key_value_heads * hd;
            const rope_theta: f32 = if (is_full) 1000000.0 else 10000.0;
            const rot_dim: usize = if (is_full) hd / 4 else hd; // full attention 采用 0.25 partial rotary factor
            const has_kv = (idx < cfg.num_hidden_layers - cfg.num_kv_shared_layers);
            const prev_kv_idx = PREVIOUS_KVS[idx];

            // 注意力投影权重绑定 (Attention Projections)
            const q_proj = try loadQ4(archive, &name_buf, idx, "self_attn.q_proj", cfg.hidden_size, out_q, 64, 4, stream);
            var k_proj: ?Q4Weights = null;
            var v_proj: ?Q4Weights = null;
            var kn_w: ?mlx.Array = null;

            if (has_kv) {
                k_proj = try loadQ4(archive, &name_buf, idx, "self_attn.k_proj", cfg.hidden_size, out_kv, 64, 4, stream);
                v_proj = try loadQ4(archive, &name_buf, idx, "self_attn.v_proj", cfg.hidden_size, out_kv, 64, 4, stream);
                kn_w = try loadNorm(archive, &name_buf, idx, "self_attn.k_norm.weight", hd);
            }
            const o_proj = try loadQ4(archive, &name_buf, idx, "self_attn.o_proj", out_q, cfg.hidden_size, 64, 4, stream);

            // 门控前馈网络 (8-bit Quantized MLP)
            const gate_proj = try loadQ4(archive, &name_buf, idx, "mlp.gate_proj", cfg.hidden_size, cfg.intermediate_size, 64, 8, stream);
            const up_proj = try loadQ4(archive, &name_buf, idx, "mlp.up_proj", cfg.hidden_size, cfg.intermediate_size, 64, 8, stream);
            const down_proj = try loadQ4(archive, &name_buf, idx, "mlp.down_proj", cfg.intermediate_size, cfg.hidden_size, 64, 8, stream);

            // 逐层嵌入门控与上投影 (PLE Gating & Projection)
            const ple_gate = try loadQ4(archive, &name_buf, idx, "per_layer_input_gate", cfg.hidden_size, cfg.hidden_size_per_layer_input, 64, 4, stream);
            const ple_proj = try loadQ4(archive, &name_buf, idx, "per_layer_projection", cfg.hidden_size_per_layer_input, cfg.hidden_size, 64, 4, stream);

            // 均方根层归一化权重 (RMSNorm Weights)
            const in_w = try loadNorm(archive, &name_buf, idx, "input_layernorm.weight", cfg.hidden_size);
            const qn_w = try loadNorm(archive, &name_buf, idx, "self_attn.q_norm.weight", hd);
            const pa_w = try loadNorm(archive, &name_buf, idx, "post_attention_layernorm.weight", cfg.hidden_size);
            const pm_w = try loadNorm(archive, &name_buf, idx, "pre_feedforward_layernorm.weight", cfg.hidden_size);
            const pom_w = try loadNorm(archive, &name_buf, idx, "post_feedforward_layernorm.weight", cfg.hidden_size);
            const pln_w = try loadNorm(archive, &name_buf, idx, "post_per_layer_input_norm.weight", cfg.hidden_size);

            // 层缩放标量 (Layer Scalar)
            const scalar_name = std.fmt.bufPrint(&name_buf, "language_model.model.layers.{}.layer_scalar", .{idx}) catch unreachable;
            var l_scalar: f32 = 1.0;
            if (archive.getTensorBytes(scalar_name)) |sb| {
                if (sb.len >= 2) {
                    const u16_val = std.mem.readInt(u16, sb[0..2], .little);
                    const f32_val: f32 = @bitCast(@as(u32, u16_val) << 16);
                    l_scalar = f32_val;
                }
            }

            layers[idx] = .{
                .q_proj = q_proj,
                .k_proj = k_proj,
                .v_proj = v_proj,
                .o_proj = o_proj,
                .input_norm_w = in_w,
                .q_norm_w = qn_w,
                .k_norm_w = kn_w,
                .post_attn_w = pa_w,
                .pre_mlp_w = pm_w,
                .post_mlp_w = pom_w,
                .gate_proj = gate_proj,
                .up_proj = up_proj,
                .down_proj = down_proj,
                .ple_gate = ple_gate,
                .ple_proj = ple_proj,
                .ple_norm_w = pln_w,
                .has_kv = has_kv,
                .prev_kv_idx = prev_kv_idx,
                .layer_scalar = l_scalar,
                .is_sliding = !is_full,
                .sliding_window = cfg.sliding_window,
                .num_heads = cfg.num_attention_heads,
                .num_kv_heads = cfg.num_key_value_heads,
                .head_dim = hd,
                .rot_dim = rot_dim,
                .rope_theta = rope_theta,
                .rms_norm_eps = cfg.rms_norm_eps,
            };
        }

        const gpu_k = try allocator.alloc(?mlx.Array, cfg.num_hidden_layers);
        const gpu_v = try allocator.alloc(?mlx.Array, cfg.num_hidden_layers);
        @memset(gpu_k, null);
        @memset(gpu_v, null);

        return Gemma4E4BModel{
            .allocator = allocator,
            .archive = archive,
            .layers = layers,
            .ple = ple,
            .embed_tokens = embed_tokens,
            .norm_w = norm_w,
            .lm_head = embed_tokens,
            .config = cfg,
            .stream = stream,
            .gpu_k = gpu_k,
            .gpu_v = gpu_v,
            .curr_len = 0,
            .sqrt_hidden = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size))),
        };
    }

    pub fn deinit(self: *Gemma4E4BModel) void {
        self.resetCache();
        for (self.layers) |*l| {
            l.deinit();
        }
        self.allocator.free(self.layers);
        self.ple.deinit();
        self.norm_w.deinit();
        self.embed_tokens.deinit();
        self.stream.deinit();
        self.allocator.free(self.gpu_k);
        self.allocator.free(self.gpu_v);
        self.archive.deinit();
    }

    pub fn resetCache(self: *Gemma4E4BModel) void {
        for (self.gpu_k, self.gpu_v) |*k_opt, *v_opt| {
            if (k_opt.*) |*k| k.deinit();
            if (v_opt.*) |*v| v.deinit();
            k_opt.* = null;
            v_opt.* = null;
        }
        self.curr_len = 0;
    }

    fn loadQ4(
        archive: SafetensorsArchive,
        buf: *[128]u8,
        l_idx: usize,
        sub: []const u8,
        in_f: usize,
        out_f: usize,
        group_size: i32,
        bits: i32,
        stream: Stream,
    ) !Q4Weights {
        const w_name = try std.fmt.bufPrint(buf, "language_model.model.layers.{}.{s}.weight", .{ l_idx, sub });
        const w_bytes = archive.getTensorBytes(w_name) orelse return error.TensorNotFound;
        const s_name = try std.fmt.bufPrint(buf, "language_model.model.layers.{}.{s}.scales", .{ l_idx, sub });
        const s_bytes = archive.getTensorBytes(s_name) orelse return error.TensorNotFound;
        const b_name = try std.fmt.bufPrint(buf, "language_model.model.layers.{}.{s}.biases", .{ l_idx, sub });
        const b_bytes = archive.getTensorBytes(b_name) orelse return error.TensorNotFound;

        return Q4Weights.initFromDirectBuffersCustom(
            w_bytes,
            s_bytes,
            b_bytes,
            in_f,
            out_f,
            group_size,
            bits,
            .bfloat16,
            stream,
        );
    }

    fn loadNorm(archive: SafetensorsArchive, buf: *[128]u8, l_idx: usize, sub: []const u8, dim: usize) !mlx.Array {
        const name = try std.fmt.bufPrint(buf, "language_model.model.layers.{}.{s}", .{ l_idx, sub });
        const bytes = archive.getTensorBytes(name) orelse return error.TensorNotFound;
        return mlx.Array.fromRawData(@ptrCast(bytes.ptr), &.{dim}, .bfloat16);
    }

    /// 通过 GPU Metal 算子直接提取词元嵌入并完成反量化与缩放：[1, T, hidden_size]
    pub fn getEmbeddings(
        self: *const Gemma4E4BModel,
        tokens: []const usize,
        s: Stream,
    ) !mlx.Array {
        const T = tokens.len;
        const tokens_i32 = try self.allocator.alloc(i32, T);
        defer self.allocator.free(tokens_i32);
        for (tokens, 0..) |tok, i| {
            tokens_i32[i] = @intCast(tok);
        }

        var indices_arr = try mlx.Array.fromSliceI32(tokens_i32, &.{T});
        defer indices_arr.deinit();

        var w_sel = try self.embed_tokens.w_q.takeAxis(indices_arr, 0, s);
        defer w_sel.deinit();
        var sc_sel = try self.embed_tokens.scales.takeAxis(indices_arr, 0, s);
        defer sc_sel.deinit();
        var bi_sel = try self.embed_tokens.biases.takeAxis(indices_arr, 0, s);
        defer bi_sel.deinit();

        var deq = try mlx.Array.dequantize(
            w_sel,
            sc_sel,
            bi_sel,
            self.embed_tokens.group_size,
            self.embed_tokens.bits,
            "affine",
            s,
        );
        defer deq.deinit();

        var scaled = try deq.mulScalar(self.sqrt_hidden, s);
        defer scaled.deinit();

        return scaled.reshape(&.{ 1, T, self.config.hidden_size }, s);
    }

    /// 通过 GPU Metal 算子直接提取逐层嵌入并完成反量化与缩放：[1, T, num_layers, ple_dim]
    pub fn getPleTokens(
        self: *const Gemma4E4BModel,
        tokens: []const usize,
        s: Stream,
    ) !mlx.Array {
        const T = tokens.len;
        const tokens_i32 = try self.allocator.alloc(i32, T);
        defer self.allocator.free(tokens_i32);
        for (tokens, 0..) |tok, i| {
            tokens_i32[i] = @intCast(tok);
        }

        var indices_arr = try mlx.Array.fromSliceI32(tokens_i32, &.{T});
        defer indices_arr.deinit();

        var w_sel = try self.ple.embed_tokens_per_layer.w_q.takeAxis(indices_arr, 0, s);
        defer w_sel.deinit();
        var sc_sel = try self.ple.embed_tokens_per_layer.scales.takeAxis(indices_arr, 0, s);
        defer sc_sel.deinit();
        var bi_sel = try self.ple.embed_tokens_per_layer.biases.takeAxis(indices_arr, 0, s);
        defer bi_sel.deinit();

        var deq = try mlx.Array.dequantize(
            w_sel,
            sc_sel,
            bi_sel,
            self.ple.embed_tokens_per_layer.group_size,
            self.ple.embed_tokens_per_layer.bits,
            "affine",
            s,
        );
        defer deq.deinit();

        var scaled = try deq.mulScalar(self.ple.embed_scale, s);
        defer scaled.deinit();

        return scaled.reshape(&.{ 1, T, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input }, s);
    }

    /// 预填充阶段 (Prefill)：处理 Prompt 词元并初始化 GPU KVCache
    pub fn prefill(
        self: *Gemma4E4BModel,
        prompt_tokens: []const usize,
        repetition_penalty: f32,
    ) !usize {
        const T = prompt_tokens.len;
        if (T == 0) return error.EmptyPrompt;
        self.resetCache();

        const s = self.stream;

        // 1. 生成 Prompt 词元基础嵌入 [1, T, hidden_size]
        var x_cur = try self.getEmbeddings(prompt_tokens, s);
        errdefer x_cur.deinit();

        // 2. 计算逐层输入嵌入 (Per-Layer Embeddings, PLE): [1, T, 42, 256]
        var ple_tokens_arr = try self.getPleTokens(prompt_tokens, s);
        defer ple_tokens_arr.deinit();

        // 上下文投影: inputs_embeds -> per_layer_model_projection * proj_scale -> RMSNorm
        var ple_proj_raw = try self.ple.model_proj.forwardArray(x_cur, s);
        defer ple_proj_raw.deinit();
        var ple_proj_scaled = try ple_proj_raw.mulScalar(self.ple.proj_scale, s);
        defer ple_proj_scaled.deinit();
        var ple_proj_4d = try ple_proj_scaled.reshape(&.{ 1, T, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input }, s);
        defer ple_proj_4d.deinit();
        var ple_proj_normed = try mlx.Array.fastRmsNorm(ple_proj_4d, self.ple.proj_norm_w, self.config.rms_norm_eps, s);
        defer ple_proj_normed.deinit();

        // 融合逐层输入: (proj + tokens) * input_scale
        var ple_sum = try mlx.Array.add(ple_proj_normed, ple_tokens_arr, s);
        defer ple_sum.deinit();
        var ple_inputs = try ple_sum.mulScalar(self.ple.input_scale, s);
        defer ple_inputs.deinit();

        // 3. 执行 42 层 Transformer 前向传播
        for (self.layers, 0..) |*l, idx| {
            const x_next = try self.forwardLayer(x_cur, l, idx, ple_inputs, 0, T, s);
            x_cur.deinit();
            x_cur = x_next;
        }
        defer x_cur.deinit();

        // 4. 最终层归一化与词表投影
        var h_norm = try mlx.Array.fastRmsNorm(x_cur, self.norm_w, self.config.rms_norm_eps, s);
        defer h_norm.deinit();

        var last_token = try h_norm.sliceAxis(1, T - 1, T, s);
        defer last_token.deinit();
        var last_token_2d = try last_token.reshape(&.{ 1, self.config.hidden_size }, s);
        defer last_token_2d.deinit();

        var logits_bf16 = try self.lm_head.forwardArray(last_token_2d, s);
        defer logits_bf16.deinit();

        var logits_f32 = try logits_bf16.astype(.float32, s);
        defer logits_f32.deinit();

        const cap = self.config.final_logit_softcapping;
        var logits = if (cap > 0.0) blk: {
            var scaled_l = try logits_f32.mulScalar(1.0 / cap, s);
            defer scaled_l.deinit();
            var tanh_l = try mlx.Array.tanh(scaled_l, s);
            defer tanh_l.deinit();
            break :blk try tanh_l.mulScalar(cap, s);
        } else try logits_f32.mulScalar(1.0, s);
        defer logits.deinit();

        logits.eval();
        s.synchronize();

        const logits_slice = logits.dataFloat32();

        var top_indices: [10]usize = undefined;
        var top_vals: [10]f32 = undefined;
        @memset(&top_vals, -1e9);
        for (logits_slice, 0..) |v, i| {
            if (v > top_vals[9]) {
                var pos_ins: usize = 9;
                while (pos_ins > 0 and v > top_vals[pos_ins - 1]) : (pos_ins -= 1) {
                    top_vals[pos_ins] = top_vals[pos_ins - 1];
                    top_indices[pos_ins] = top_indices[pos_ins - 1];
                }
                top_vals[pos_ins] = v;
                top_indices[pos_ins] = i;
            }
        }
        var logit_min: f32 = 1e9;
        var logit_max: f32 = -1e9;
        for (logits_slice) |v| {
            if (v < logit_min) logit_min = v;
            if (v > logit_max) logit_max = v;
        }

        std.debug.print("\n[Logits Stats]: min={d:.2}, max={d:.2}, len={}\n", .{ logit_min, logit_max, logits_slice.len });
        std.debug.print("  - Logit for ' I' (564): {d:.2}\n", .{logits_slice[564]});
        std.debug.print("  - Logit for ' am' (1006): {d:.2}\n", .{logits_slice[1006]});
        std.debug.print("  - Logit for '一' (237009): {d:.2}\n", .{logits_slice[237009]});

        std.debug.print("\n[Top 10 Prefill Tokens]:\n", .{});
        for (0..10) |rank| {
            const t = top_indices[rank];
            std.debug.print("  #{}: id={}, val={d:.2}\n", .{ rank, t, top_vals[rank] });
        }

        const best_tok = backend.offload.LmHeadWeights.argmaxWithPenalty(logits_slice, self.config.vocab_size, prompt_tokens, repetition_penalty);

        self.curr_len = T;
        return best_tok;
    }

    /// 单步自回归解码 (Autoregressive Decode Step)
    pub fn decodeStep(
        self: *Gemma4E4BModel,
        tok: usize,
        history_tokens: []const usize,
        repetition_penalty: f32,
    ) !usize {
        const s = self.stream;
        const pos = self.curr_len;

        // 1. 单词元嵌入
        var x_cur = try self.getEmbeddings(&.{tok}, s);
        errdefer x_cur.deinit();

        // 2. 单词元 PLE
        var ple_tokens_arr = try self.getPleTokens(&.{tok}, s);
        defer ple_tokens_arr.deinit();

        var ple_proj_raw = try self.ple.model_proj.forwardArray(x_cur, s);
        defer ple_proj_raw.deinit();
        var ple_proj_scaled = try ple_proj_raw.mulScalar(self.ple.proj_scale, s);
        defer ple_proj_scaled.deinit();
        var ple_proj_4d = try ple_proj_scaled.reshape(&.{ 1, 1, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input }, s);
        defer ple_proj_4d.deinit();
        var ple_proj_normed = try mlx.Array.fastRmsNorm(ple_proj_4d, self.ple.proj_norm_w, self.config.rms_norm_eps, s);
        defer ple_proj_normed.deinit();

        var ple_sum = try mlx.Array.add(ple_proj_normed, ple_tokens_arr, s);
        defer ple_sum.deinit();
        var ple_inputs = try ple_sum.mulScalar(self.ple.input_scale, s);
        defer ple_inputs.deinit();

        // 3. 执行 42 层 Transformer 解码步
        for (self.layers, 0..) |*l, idx| {
            const x_next = try self.forwardLayer(x_cur, l, idx, ple_inputs, pos, 1, s);
            x_cur.deinit();
            x_cur = x_next;
        }
        defer x_cur.deinit();

        // 4. 最终层归一化与词表投影
        var h_norm = try mlx.Array.fastRmsNorm(x_cur, self.norm_w, self.config.rms_norm_eps, s);
        defer h_norm.deinit();

        var h_2d = try h_norm.reshape(&.{ 1, self.config.hidden_size }, s);
        defer h_2d.deinit();

        var logits_bf16 = try self.lm_head.forwardArray(h_2d, s);
        defer logits_bf16.deinit();

        var logits_f32 = try logits_bf16.astype(.float32, s);
        defer logits_f32.deinit();

        const cap = self.config.final_logit_softcapping;
        var logits = if (cap > 0.0) blk: {
            var scaled_l = try logits_f32.mulScalar(1.0 / cap, s);
            defer scaled_l.deinit();
            var tanh_l = try mlx.Array.tanh(scaled_l, s);
            defer tanh_l.deinit();
            break :blk try tanh_l.mulScalar(cap, s);
        } else try logits_f32.mulScalar(1.0, s);
        defer logits.deinit();

        logits.eval();
        s.synchronize();

        const logits_slice = logits.dataFloat32();
        const best_tok = backend.offload.LmHeadWeights.argmaxWithPenalty(logits_slice, self.config.vocab_size, history_tokens, repetition_penalty);

        self.curr_len = pos + 1;
        return best_tok;
    }

    fn forwardLayer(
        self: *Gemma4E4BModel,
        x_in: mlx.Array,
        l: *GpuLayerE4B,
        l_idx: usize,
        ple_inputs: mlx.Array,
        pos: usize,
        seq_len: usize,
        s: Stream,
    ) !mlx.Array {
        // (1) 输入层 RMSNorm
        var in_norm = try mlx.Array.fastRmsNorm(x_in, l.input_norm_w, l.rms_norm_eps, s);
        defer in_norm.deinit();

        // (2) 注意力查询投影: Q
        var q_raw = try l.q_proj.forwardArray(in_norm, s);
        defer q_raw.deinit();

        // 重塑形状为 [1, num_heads, seq_len, head_dim]
        var q_4d_raw = try q_raw.reshape(&.{ 1, seq_len, l.num_heads, l.head_dim }, s);
        defer q_4d_raw.deinit();
        var q_4d = try q_4d_raw.transposeAxes(&.{ 0, 2, 1, 3 }, s);
        defer q_4d.deinit();

        // Q RMSNorm
        var q_normed = try mlx.Array.fastRmsNorm(q_4d, l.q_norm_w, l.rms_norm_eps, s);
        defer q_normed.deinit();

        // 旋转位置编码 (Rotary Position Embedding, RoPE)
        var q_rot = try mlx.Array.fastRoPE(q_normed, l.rot_dim, false, l.rope_theta, 1.0, pos, s);
        defer q_rot.deinit();

        // 键值缓存 (Key-Value Cache, KVCache) 维护
        var full_k: mlx.Array = undefined;
        var full_v: mlx.Array = undefined;

        if (l.has_kv) {
            var k_raw = try l.k_proj.?.forwardArray(in_norm, s);
            defer k_raw.deinit();
            var v_raw = try l.v_proj.?.forwardArray(in_norm, s);
            defer v_raw.deinit();

            var k_4d_raw = try k_raw.reshape(&.{ 1, seq_len, l.num_kv_heads, l.head_dim }, s);
            defer k_4d_raw.deinit();
            var k_4d = try k_4d_raw.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer k_4d.deinit();

            var v_4d_raw = try v_raw.reshape(&.{ 1, seq_len, l.num_kv_heads, l.head_dim }, s);
            defer v_4d_raw.deinit();
            var v_4d = try v_4d_raw.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer v_4d.deinit();

            var k_normed = try mlx.Array.fastRmsNorm(k_4d, l.k_norm_w.?, l.rms_norm_eps, s);
            defer k_normed.deinit();
            var v_normed = try mlx.Array.fastRmsNorm(v_4d, null, l.rms_norm_eps, s);
            defer v_normed.deinit();

            var k_rot = try mlx.Array.fastRoPE(k_normed, l.rot_dim, false, l.rope_theta, 1.0, pos, s);
            defer k_rot.deinit();

            if (self.gpu_k[l_idx]) |old_k| {
                const new_k = try mlx.Array.concatenateAxis(&.{ old_k, k_rot }, 2, s);
                const new_v = try mlx.Array.concatenateAxis(&.{ self.gpu_v[l_idx].?, v_normed }, 2, s);
                var k_free = old_k;
                var v_free = self.gpu_v[l_idx].?;
                self.gpu_k[l_idx] = new_k;
                self.gpu_v[l_idx] = new_v;
                k_free.deinit();
                v_free.deinit();
            } else {
                self.gpu_k[l_idx] = try k_rot.mulScalar(1.0, s);
                self.gpu_v[l_idx] = try v_normed.mulScalar(1.0, s);
            }

            full_k = self.gpu_k[l_idx].?;
            full_v = self.gpu_v[l_idx].?;
        } else {
            full_k = self.gpu_k[l.prev_kv_idx].?;
            full_v = self.gpu_v[l.prev_kv_idx].?;
        }
        const total_len = pos + seq_len;

        // 滑动窗口切片 (Sliding Window Mask/Slice)
        var eff_k = full_k;
        var eff_v = full_v;
        var eff_k_free: ?mlx.Array = null;
        var eff_v_free: ?mlx.Array = null;
        defer if (eff_k_free) |*ek| ek.deinit();
        defer if (eff_v_free) |*ev| ev.deinit();

        if (l.is_sliding and total_len > l.sliding_window) {
            const start_slice = total_len - l.sliding_window;
            eff_k = try full_k.sliceAxis(2, start_slice, total_len, s);
            eff_k_free = eff_k;
            eff_v = try full_v.sliceAxis(2, start_slice, total_len, s);
            eff_v_free = eff_v;
        }

        // 缩放点积注意力 (Scaled Dot-Product Attention, SDPA, scaling = 1.0)
        const is_causal = (seq_len > 1);
        var attn_out_4d = try mlx.Array.fastScaledDotProductAttention(q_rot, eff_k, eff_v, 1.0, is_causal, s);
        defer attn_out_4d.deinit();

        var attn_trans = try attn_out_4d.transposeAxes(&.{ 0, 2, 1, 3 }, s);
        defer attn_trans.deinit();
        var attn_2d = try attn_trans.reshape(&.{ 1, seq_len, l.num_heads * l.head_dim }, s);
        defer attn_2d.deinit();

        var o_out = try l.o_proj.forwardArray(attn_2d, s);
        defer o_out.deinit();

        var post_attn = try mlx.Array.fastRmsNorm(o_out, l.post_attn_w, l.rms_norm_eps, s);
        defer post_attn.deinit();
        var x1 = try mlx.Array.add(x_in, post_attn, s);
        defer x1.deinit();

        // (3) 门控多层感知机 (Gated MLP: GeGLU with GELU approx)
        var pre_mlp = try mlx.Array.fastRmsNorm(x1, l.pre_mlp_w, l.rms_norm_eps, s);
        defer pre_mlp.deinit();

        var gate_out = try l.gate_proj.forwardArray(pre_mlp, s);
        defer gate_out.deinit();
        var up_out = try l.up_proj.forwardArray(pre_mlp, s);
        defer up_out.deinit();

        var gelu_gate = try gate_out.gelu(s);
        defer gelu_gate.deinit();
        var hidden = try mlx.Array.multiply(gelu_gate, up_out, s);
        defer hidden.deinit();

        var down_out = try l.down_proj.forwardArray(hidden, s);
        defer down_out.deinit();

        var post_mlp = try mlx.Array.fastRmsNorm(down_out, l.post_mlp_w, l.rms_norm_eps, s);
        defer post_mlp.deinit();
        var x2 = try mlx.Array.add(x1, post_mlp, s);
        defer x2.deinit();

        // (4) 逐层输入嵌入增强 (Per-Layer Inputs, PLE)
        // ple_inputs: [1, seq_len, 42, 256] -> 获取当前层切片 [1, seq_len, 256]
        var ple_layer_slice_4d = try ple_inputs.sliceAxis(2, l_idx, l_idx + 1, s);
        defer ple_layer_slice_4d.deinit();
        var ple_layer_slice = try ple_layer_slice_4d.reshape(&.{ 1, seq_len, l.ple_gate.out_features }, s);
        defer ple_layer_slice.deinit();

        var ple_gate_raw = try l.ple_gate.forwardArray(x2, s);
        defer ple_gate_raw.deinit();
        var ple_gate_act = try ple_gate_raw.gelu(s);
        defer ple_gate_act.deinit();

        var ple_modulated = try mlx.Array.multiply(ple_gate_act, ple_layer_slice, s);
        defer ple_modulated.deinit();
        var ple_projected = try l.ple_proj.forwardArray(ple_modulated, s);
        defer ple_projected.deinit();

        var ple_normed = try mlx.Array.fastRmsNorm(ple_projected, l.ple_norm_w, l.rms_norm_eps, s);
        defer ple_normed.deinit();
        var x3 = try mlx.Array.add(x2, ple_normed, s);
        defer x3.deinit();

        // (5) 层缩放标量 (Layer Scalar)
        if (l.layer_scalar != 1.0) {
            return x3.mulScalar(l.layer_scalar, s);
        } else {
            return x3.mulScalar(1.0, s);
        }
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const io = init.io;

    var custom_prompt: ?[]const u8 = null;
    var max_new_tokens: usize = 40;
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
            }
        }
    } else |_| {}

    std.debug.print("================================================================================\n", .{});
    std.debug.print("🤖 Gemma 4 E4B Instruct (4B 等效端侧大模型) GPU 极速推理与真实评测\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // 1. 验证模型目录
    std.debug.print("1. [准备模型权重与词表解码器]:\n", .{});
    std.debug.print("   - 模型路径: {s}\n", .{model_dir});

    var timer_load: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_load);
    const load_start_ns = @as(u64, @intCast(timer_load.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_load.nsec));

    var model = Gemma4E4BModel.loadFromDirectory(allocator, io, model_dir) catch |err| {
        std.debug.print("❌ 无法从 {s} 加载模型: {s}\n", .{ model_dir, @errorName(err) });
        return;
    };
    defer model.deinit();

    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_load);
    const load_end_ns = @as(u64, @intCast(timer_load.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_load.nsec));
    const load_elapsed_s = @as(f64, @floatFromInt(load_end_ns - load_start_ns)) / 1e9;

    std.debug.print("   - 格式架构: ⚡ MLX 原生分块量化 Safetensors (零拷贝 mmap 直通加载)\n", .{});
    std.debug.print("   - 模型规模: 42 层 Transformer (35 层局部滑动窗口 + 7 层全局全注意力)\n", .{});
    std.debug.print("   - 逐层增强: 逐层嵌入输入增强 (Per-Layer Embeddings, PLE) 已完全就绪\n", .{});
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
    std.debug.print("   User  : {s}\n", .{user_prompt_str});
    std.debug.print("   Model : ", .{});

    var timer_ts: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
    const start_ns = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));

    // Prefill 阶段
    _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &timer_ts);
    const p_start = @as(u64, @intCast(timer_ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(timer_ts.nsec));

    const first_tok = try model.prefill(prompt_tokens, 1.25);

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

    // Decode 循环
    var current_tok = first_tok;
    const decode_start_ns = p_end;

    while (gen_count < max_new_tokens) {
        if (current_tok == gemma4.Gemma4Vocabulary.EOS or current_tok == gemma4.Gemma4Vocabulary.EOT) {
            break;
        }

        const next_tok = try model.decodeStep(current_tok, current_tokens.items, 1.25);
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
    std.debug.print("   - 自回归解码速率     : {d:.2} token/s (纯 Metal GPU 流水线)\n", .{decode_speed});
    std.debug.print("   - 端到端总生成耗时   : {d:.2} 秒\n", .{total_time_s});
    std.debug.print("================================================================================\n\n", .{});
}
