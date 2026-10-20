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
        try tensors.ensureTotalCapacity(800);

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
    embed_scale_arr: mlx.Array,
    proj_scale_arr: mlx.Array,
    input_scale_arr: mlx.Array,

    pub fn deinit(self: *PleWeights) void {
        self.embed_tokens_per_layer.deinit();
        self.model_proj.deinit();
        self.proj_norm_w.deinit();
        self.embed_scale_arr.deinit();
        self.proj_scale_arr.deinit();
        self.input_scale_arr.deinit();
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
    layer_scalar_w: ?mlx.Array = null,
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
        if (self.layer_scalar_w) |*sw| sw.deinit();
    }
};

/// 块式预分配显存键值缓存 (Pre-allocated Chunked GPU Key-Value Cache, GPU KVCache)
pub const GpuKvCache = struct {
    k: mlx.Array,
    v: mlx.Array,
    capacity: usize,
    num_kv: usize,
    head_dim: usize,

    pub fn init(capacity: usize, num_kv: usize, head_dim: usize, stream: Stream) !GpuKvCache {
        var k = try mlx.Array.zeros(&.{ 1, num_kv, capacity, head_dim }, .bfloat16, stream);
        errdefer k.deinit();
        const v = try mlx.Array.zeros(&.{ 1, num_kv, capacity, head_dim }, .bfloat16, stream);
        return .{
            .k = k,
            .v = v,
            .capacity = capacity,
            .num_kv = num_kv,
            .head_dim = head_dim,
        };
    }

    pub fn deinit(self: *GpuKvCache) void {
        self.k.deinit();
        self.v.deinit();
    }

    pub fn ensureCapacity(self: *GpuKvCache, required: usize, stream: Stream) !void {
        if (required <= self.capacity) return;
        const add_cap: usize = ((required - self.capacity + 255) / 256) * 256;
        const new_cap = self.capacity + add_cap;
        var pad_k = try mlx.Array.zeros(&.{ 1, self.num_kv, add_cap, self.head_dim }, .bfloat16, stream);
        defer pad_k.deinit();
        var pad_v = try mlx.Array.zeros(&.{ 1, self.num_kv, add_cap, self.head_dim }, .bfloat16, stream);
        defer pad_v.deinit();

        const exp_k = try mlx.Array.concatenateAxis(&.{ self.k, pad_k }, 2, stream);
        const exp_v = try mlx.Array.concatenateAxis(&.{ self.v, pad_v }, 2, stream);

        self.k.deinit();
        self.v.deinit();
        self.k = exp_k;
        self.v = exp_v;
        self.capacity = new_cap;
    }
};

/// Gemma 4 E4B 完整模型推理引擎
pub const Gemma4E4BModel = struct {
    allocator: std.mem.Allocator,
    layers: []GpuLayerE4B,
    ple: PleWeights,

    embed_tokens: Q4Weights,
    norm_w: mlx.Array,
    lm_head: Q4Weights,

    config: Gemma4Config,
    stream: Stream,

    // 24 个独立 KV 层的块式预分配显存键值缓存 (Pre-allocated Chunked GPU Key-Value Cache)
    gpu_kv: [24]?GpuKvCache = [_]?GpuKvCache{null} ** 24,
    curr_len: usize = 0,
    sqrt_hidden: f32,
    sqrt_hidden_arr: mlx.Array,
    compiled_gelu_mul: ?mlx.Closure = null,

    pub fn loadFromDirectory(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !Gemma4E4BModel {
        var archive = try SafetensorsArchive.open(allocator, io, dir_path);
        defer archive.deinit();

        const cfg = Gemma4Config.e4b;
        const stream = try Stream.defaultGpu();
        errdefer {
            var s_tmp = stream;
            s_tmp.deinit();
        }

        mlx.enableCompile();
        mlx.setCompileMode(.enabled);

        // 1. 词表嵌入与 LM Head 权重
        const emb_wq = archive.getTensorBytes("language_model.model.embed_tokens.weight") orelse return error.TensorNotFound;
        const emb_sc = archive.getTensorBytes("language_model.model.embed_tokens.scales") orelse return error.TensorNotFound;
        const emb_bi = archive.getTensorBytes("language_model.model.embed_tokens.biases") orelse return error.TensorNotFound;

        var embed_tokens = try Q4Weights.initFromDirectBuffersCustom(
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
        errdefer embed_tokens.deinit();

        const norm_bytes = archive.getTensorBytes("language_model.model.norm.weight") orelse return error.TensorNotFound;
        var norm_w = try mlx.Array.fromRawData(@ptrCast(norm_bytes.ptr), &.{cfg.hidden_size}, .bfloat16);
        errdefer norm_w.deinit();

        // 2. 逐层嵌入输入结构 (Per-Layer Embeddings, PLE)
        const ple_tokens_wq = archive.getTensorBytes("language_model.model.embed_tokens_per_layer.weight") orelse return error.TensorNotFound;
        const ple_tokens_sc = archive.getTensorBytes("language_model.model.embed_tokens_per_layer.scales") orelse return error.TensorNotFound;
        const ple_tokens_bi = archive.getTensorBytes("language_model.model.embed_tokens_per_layer.biases") orelse return error.TensorNotFound;

        var ple_tokens = try Q4Weights.initFromDirectBuffersCustom(
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
        errdefer ple_tokens.deinit();

        const ple_proj_wq = archive.getTensorBytes("language_model.model.per_layer_model_projection.weight") orelse return error.TensorNotFound;
        const ple_proj_sc = archive.getTensorBytes("language_model.model.per_layer_model_projection.scales") orelse return error.TensorNotFound;
        const ple_proj_bi = archive.getTensorBytes("language_model.model.per_layer_model_projection.biases") orelse return error.TensorNotFound;

        var ple_model_proj = try Q4Weights.initFromDirectBuffersCustom(
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
        errdefer ple_model_proj.deinit();

        const ple_pnorm_bytes = archive.getTensorBytes("language_model.model.per_layer_projection_norm.weight") orelse return error.TensorNotFound;
        var ple_pnorm_w = try mlx.Array.fromRawData(@ptrCast(ple_pnorm_bytes.ptr), &.{cfg.hidden_size_per_layer_input}, .bfloat16);
        errdefer ple_pnorm_w.deinit();

        const embed_scale_val: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size_per_layer_input)));
        const proj_scale_val: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        const input_scale_val: f32 = 1.0 / @sqrt(2.0);

        var embed_scale_arr = try mlx.Array.fromScalarBf16(embed_scale_val);
        errdefer embed_scale_arr.deinit();
        var proj_scale_arr = try mlx.Array.fromScalarBf16(proj_scale_val);
        errdefer proj_scale_arr.deinit();
        var input_scale_arr = try mlx.Array.fromScalarBf16(input_scale_val);
        errdefer input_scale_arr.deinit();

        var ple = PleWeights{
            .embed_tokens_per_layer = ple_tokens,
            .model_proj = ple_model_proj,
            .proj_norm_w = ple_pnorm_w,
            .num_layers = cfg.num_hidden_layers,
            .ple_dim = cfg.hidden_size_per_layer_input,
            .embed_scale = embed_scale_val,
            .proj_scale = proj_scale_val,
            .input_scale = input_scale_val,
            .embed_scale_arr = embed_scale_arr,
            .proj_scale_arr = proj_scale_arr,
            .input_scale_arr = input_scale_arr,
        };
        errdefer ple.deinit();

        const sqrt_hidden_val: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        var sqrt_hidden_arr = try mlx.Array.fromScalarBf16(sqrt_hidden_val);
        errdefer sqrt_hidden_arr.deinit();

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

            // 层缩放标量 (Layer Scalar, 直接绑定 BF16 标量张量以消除每步标量创建与类型提升)
            const scalar_name = std.fmt.bufPrint(&name_buf, "language_model.model.layers.{}.layer_scalar", .{idx}) catch unreachable;
            var l_scalar: f32 = 1.0;
            var l_scalar_w: ?mlx.Array = null;
            if (archive.getTensorBytes(scalar_name)) |sb| {
                if (sb.len >= 2) {
                    const u16_val = std.mem.readInt(u16, sb[0..2], .little);
                    const f32_val: f32 = @bitCast(@as(u32, u16_val) << 16);
                    l_scalar = f32_val;
                    if (l_scalar != 1.0) {
                        l_scalar_w = try mlx.Array.fromRawData(@ptrCast(sb.ptr), &.{1}, .bfloat16);
                    }
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
                .layer_scalar_w = l_scalar_w,
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

        const compiled_gelu_mul = mlx.Array.compileGeluMul() catch null;

        return Gemma4E4BModel{
            .allocator = allocator,
            .layers = layers,
            .ple = ple,
            .embed_tokens = embed_tokens,
            .norm_w = norm_w,
            .lm_head = embed_tokens,
            .config = cfg,
            .stream = stream,
            .gpu_kv = [_]?GpuKvCache{null} ** 24,
            .curr_len = 0,
            .sqrt_hidden = sqrt_hidden_val,
            .sqrt_hidden_arr = sqrt_hidden_arr,
            .compiled_gelu_mul = compiled_gelu_mul,
        };
    }

    pub fn deinit(self: *Gemma4E4BModel) void {
        self.resetCache();
        if (self.compiled_gelu_mul) |*cls| {
            cls.deinit();
            self.compiled_gelu_mul = null;
        }
        for (self.layers) |*l| {
            l.deinit();
        }
        self.allocator.free(self.layers);
        self.ple.deinit();
        self.norm_w.deinit();
        self.sqrt_hidden_arr.deinit();
        self.embed_tokens.deinit();
        self.stream.deinit();
    }

    pub fn resetCache(self: *Gemma4E4BModel) void {
        for (&self.gpu_kv) |*kv_opt| {
            if (kv_opt.*) |*c| {
                c.deinit();
                kv_opt.* = null;
            }
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

    fn createTokenIndicesArray(self: *const Gemma4E4BModel, tokens: []const usize) !mlx.Array {
        const T = tokens.len;
        const single_tok = [_]i32{if (T > 0) @intCast(tokens[0]) else 0};
        if (T == 1) {
            return mlx.Array.fromSliceI32(&single_tok, &.{1});
        }
        const tokens_i32 = try self.allocator.alloc(i32, T);
        defer self.allocator.free(tokens_i32);
        for (tokens, 0..) |tok, i| {
            tokens_i32[i] = @intCast(tok);
        }
        return mlx.Array.fromSliceI32(tokens_i32, &.{T});
    }

    /// 直接基于 GPU 上的词元索引张量提取词元嵌入并完成反量化与 BF16 缩放：[1, T, hidden_size]
    pub fn getEmbeddingsFromArray(
        self: *const Gemma4E4BModel,
        indices_arr: mlx.Array,
        T: usize,
        s: Stream,
    ) !mlx.Array {
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

        var scaled = try mlx.Array.multiply(deq, self.sqrt_hidden_arr, s);
        defer scaled.deinit();

        return scaled.reshape(&.{ 1, T, self.config.hidden_size }, s);
    }

    /// 通过 GPU Metal 算子直接提取词元嵌入并完成反量化与缩放：[1, T, hidden_size]
    pub fn getEmbeddings(
        self: *const Gemma4E4BModel,
        tokens: []const usize,
        s: Stream,
    ) !mlx.Array {
        var indices_arr = try self.createTokenIndicesArray(tokens);
        defer indices_arr.deinit();
        return self.getEmbeddingsFromArray(indices_arr, tokens.len, s);
    }

    /// 直接基于 GPU 上的词元索引张量提取逐层嵌入并完成反量化与 BF16 缩放：[1, T, num_layers, ple_dim]
    pub fn getPleTokensFromArray(
        self: *const Gemma4E4BModel,
        indices_arr: mlx.Array,
        T: usize,
        s: Stream,
    ) !mlx.Array {
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

        var scaled = try mlx.Array.multiply(deq, self.ple.embed_scale_arr, s);
        defer scaled.deinit();

        return scaled.reshape(&.{ 1, T, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input }, s);
    }

    /// 通过 GPU Metal 算子直接提取逐层嵌入并完成反量化与缩放：[1, T, num_layers, ple_dim]
    pub fn getPleTokens(
        self: *const Gemma4E4BModel,
        tokens: []const usize,
        s: Stream,
    ) !mlx.Array {
        var indices_arr = try self.createTokenIndicesArray(tokens);
        defer indices_arr.deinit();
        return self.getPleTokensFromArray(indices_arr, tokens.len, s);
    }

    /// 基于 GPU 词元索引张量构建整网 42 层 + 最终 RMSNorm + LM Head 的 BF16 对数几率张量 `[1, vocab_size]`
    fn forwardTokenArrayToLogits(
        self: *Gemma4E4BModel,
        indices_arr: mlx.Array,
        pos: usize,
        T: usize,
        s: Stream,
    ) !mlx.Array {
        // 1. 生成词元基础嵌入 [1, T, hidden_size] (BF16)
        var x_cur = try self.getEmbeddingsFromArray(indices_arr, T, s);
        errdefer x_cur.deinit();

        // 2. 计算逐层输入嵌入 (Per-Layer Embeddings, PLE): [1, T, 42, 256] (BF16)
        var ple_tokens_arr = try self.getPleTokensFromArray(indices_arr, T, s);
        defer ple_tokens_arr.deinit();

        var ple_proj_raw = try self.ple.model_proj.forwardArray(x_cur, s);
        defer ple_proj_raw.deinit();
        var ple_proj_scaled = try mlx.Array.multiply(ple_proj_raw, self.ple.proj_scale_arr, s);
        defer ple_proj_scaled.deinit();
        var ple_proj_4d = try ple_proj_scaled.reshape(&.{ 1, T, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input }, s);
        defer ple_proj_4d.deinit();
        var ple_proj_normed = try mlx.Array.fastRmsNorm(ple_proj_4d, self.ple.proj_norm_w, self.config.rms_norm_eps, s);
        defer ple_proj_normed.deinit();

        // 融合逐层输入: (proj + tokens) * input_scale (保持纯 BF16)
        var ple_sum = try mlx.Array.add(ple_proj_normed, ple_tokens_arr, s);
        defer ple_sum.deinit();
        var ple_inputs = try mlx.Array.multiply(ple_sum, self.ple.input_scale_arr, s);
        defer ple_inputs.deinit();

        // 3. 执行 42 层 Transformer 前向传播 (T == 1 时重塑为 [1, 42, 256] 并沿 axis 1 切分，直接获得 [1, 1, 256] 切片，省去 42 次逐层 reshape)
        if (T == 1) {
            var ple_3d = try ple_inputs.reshape(&.{ 1, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input }, s);
            defer ple_3d.deinit();
            var ple_vec = try ple_3d.split(42, 1, s);
            defer ple_vec.deinit();

            for (self.layers, 0..) |*l, idx| {
                var ple_slice = ple_vec.get(idx).?;
                defer ple_slice.deinit();

                const x_next = try self.forwardLayer(x_cur, l, idx, ple_slice, pos, 1, s);
                x_cur.deinit();
                x_cur = x_next;
            }
        } else {
            var ple_vec = try ple_inputs.split(42, 2, s);
            defer ple_vec.deinit();

            for (self.layers, 0..) |*l, idx| {
                var ple_item = ple_vec.get(idx).?;
                defer ple_item.deinit();
                var ple_slice = try ple_item.reshape(&.{ 1, T, self.config.hidden_size_per_layer_input }, s);
                defer ple_slice.deinit();

                const x_next = try self.forwardLayer(x_cur, l, idx, ple_slice, pos, T, s);
                x_cur.deinit();
                x_cur = x_next;
            }
        }
        defer x_cur.deinit();

        // 4. 最终层归一化与词表投影
        var h_norm = try mlx.Array.fastRmsNorm(x_cur, self.norm_w, self.config.rms_norm_eps, s);
        defer h_norm.deinit();

        if (T == 1) {
            var h_2d = try h_norm.reshape(&.{ 1, self.config.hidden_size }, s);
            defer h_2d.deinit();
            return self.lm_head.forwardArray(h_2d, s);
        } else {
            var last_token = try h_norm.sliceAxis(1, T - 1, T, s);
            defer last_token.deinit();
            var last_token_2d = try last_token.reshape(&.{ 1, self.config.hidden_size }, s);
            defer last_token_2d.deinit();
            return self.lm_head.forwardArray(last_token_2d, s);
        }
    }

    /// 异步预填充阶段 (Async Prefill)：构建预填充计算图并返回 GPU 上的首词元 ID 张量 (`shape = [1], dtype = uint32`)
    pub fn prefillAsync(
        self: *Gemma4E4BModel,
        prompt_tokens: []const usize,
    ) !mlx.Array {
        const T = prompt_tokens.len;
        if (T == 0) return error.EmptyPrompt;
        self.resetCache();

        const s = self.stream;
        var indices_arr = try self.createTokenIndicesArray(prompt_tokens);
        defer indices_arr.deinit();

        var logits_bf16 = try self.forwardTokenArrayToLogits(indices_arr, 0, T, s);
        defer logits_bf16.deinit();

        var best_tok_arr = try logits_bf16.argmax(1, false, s);
        best_tok_arr.asyncEval();
        self.curr_len = T;
        return best_tok_arr;
    }

    /// 异步单步解码阶段 (Async Pipelined Decode Step)：直接消费上一步在 GPU 上的词元张量 (`shape = [1]`) 构建下一步计算图并异步派发至 Metal 流水线
    pub fn decodeStepAsync(
        self: *Gemma4E4BModel,
        tok_arr: mlx.Array,
    ) !mlx.Array {
        const s = self.stream;
        const pos = self.curr_len;

        var logits_bf16 = try self.forwardTokenArrayToLogits(tok_arr, pos, 1, s);
        defer logits_bf16.deinit();

        var best_tok_arr = try logits_bf16.argmax(1, false, s);
        best_tok_arr.asyncEval();
        self.curr_len = pos + 1;
        return best_tok_arr;
    }

    /// 预填充阶段 (Prefill)：处理 Prompt 词元并初始化 GPU KVCache
    pub fn prefill(
        self: *Gemma4E4BModel,
        prompt_tokens: []const usize,
        repetition_penalty: f32,
    ) !usize {
        if (repetition_penalty == 1.0) {
            var best_tok_arr = try self.prefillAsync(prompt_tokens);
            defer best_tok_arr.deinit();
            return @intCast(best_tok_arr.itemUint32());
        }

        const T = prompt_tokens.len;
        if (T == 0) return error.EmptyPrompt;
        self.resetCache();

        const s = self.stream;
        var indices_arr = try self.createTokenIndicesArray(prompt_tokens);
        defer indices_arr.deinit();

        var logits_bf16 = try self.forwardTokenArrayToLogits(indices_arr, 0, T, s);
        defer logits_bf16.deinit();

        var logits_f32 = try logits_bf16.astype(.float32, s);
        defer logits_f32.deinit();

        const cap = self.config.final_logit_softcapping;
        var logits = if (cap > 0.0) blk2: {
            var scaled_l = try logits_f32.mulScalar(1.0 / cap, s);
            defer scaled_l.deinit();
            var tanh_l = try mlx.Array.tanh(scaled_l, s);
            defer tanh_l.deinit();
            break :blk2 try tanh_l.mulScalar(cap, s);
        } else try logits_f32.mulScalar(1.0, s);
        defer logits.deinit();

        logits.eval();
        s.synchronize();

        const logits_slice = logits.dataFloat32();
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

        const single_tok = [_]i32{@intCast(tok)};
        var indices_arr = try mlx.Array.fromSliceI32(&single_tok, &.{1});
        defer indices_arr.deinit();

        if (repetition_penalty == 1.0) {
            var best_tok_arr = try self.decodeStepAsync(indices_arr);
            defer best_tok_arr.deinit();
            return @intCast(best_tok_arr.itemUint32());
        }

        var logits_bf16 = try self.forwardTokenArrayToLogits(indices_arr, pos, 1, s);
        defer logits_bf16.deinit();

        var logits_f32 = try logits_bf16.astype(.float32, s);
        defer logits_f32.deinit();

        const cap = self.config.final_logit_softcapping;
        var logits = if (cap > 0.0) blk2: {
            var scaled_l = try logits_f32.mulScalar(1.0 / cap, s);
            defer scaled_l.deinit();
            var tanh_l = try mlx.Array.tanh(scaled_l, s);
            defer tanh_l.deinit();
            break :blk2 try tanh_l.mulScalar(cap, s);
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
        ple_slice: mlx.Array,
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

        // 当 seq_len == 1 (自回归解码步) 时，[1, 1, num_heads, head_dim] 与 [1, num_heads, 1, head_dim] 物理内存完全等价，直接重塑即可省去 transposeAxes
        var q_rot: mlx.Array = if (seq_len == 1) blk: {
            var q_4d_raw = try q_raw.reshape(&.{ 1, l.num_heads, 1, l.head_dim }, s);
            defer q_4d_raw.deinit();
            var q_normed = try mlx.Array.fastRmsNorm(q_4d_raw, l.q_norm_w, l.rms_norm_eps, s);
            defer q_normed.deinit();
            break :blk try mlx.Array.fastRoPE(q_normed, l.rot_dim, false, l.rope_theta, 1.0, pos, s);
        } else blk: {
            var q_4d_raw = try q_raw.reshape(&.{ 1, seq_len, l.num_heads, l.head_dim }, s);
            defer q_4d_raw.deinit();
            var q_normed = try mlx.Array.fastRmsNorm(q_4d_raw, l.q_norm_w, l.rms_norm_eps, s);
            defer q_normed.deinit();
            var q_4d = try q_normed.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer q_4d.deinit();
            break :blk try mlx.Array.fastRoPE(q_4d, l.rot_dim, false, l.rope_theta, 1.0, pos, s);
        };
        defer q_rot.deinit();

        // 键值缓存 (Key-Value Cache, KVCache) 维护
        var eff_k: mlx.Array = undefined;
        var eff_v: mlx.Array = undefined;
        var eff_k_free: ?mlx.Array = null;
        var eff_v_free: ?mlx.Array = null;
        defer if (eff_k_free) |*ek| ek.deinit();
        defer if (eff_v_free) |*ev| ev.deinit();

        const total_len = pos + seq_len;

        if (l.has_kv) {
            var k_raw = try l.k_proj.?.forwardArray(in_norm, s);
            defer k_raw.deinit();
            var v_raw = try l.v_proj.?.forwardArray(in_norm, s);
            defer v_raw.deinit();

            var k_rot: mlx.Array = undefined;
            var v_4d: mlx.Array = undefined;
            if (seq_len == 1) {
                var k_4d_raw = try k_raw.reshape(&.{ 1, l.num_kv_heads, 1, l.head_dim }, s);
                defer k_4d_raw.deinit();
                var v_4d_raw = try v_raw.reshape(&.{ 1, l.num_kv_heads, 1, l.head_dim }, s);
                defer v_4d_raw.deinit();

                var k_normed = try mlx.Array.fastRmsNorm(k_4d_raw, l.k_norm_w.?, l.rms_norm_eps, s);
                defer k_normed.deinit();
                v_4d = try mlx.Array.fastRmsNorm(v_4d_raw, null, l.rms_norm_eps, s);
                errdefer v_4d.deinit();

                k_rot = try mlx.Array.fastRoPE(k_normed, l.rot_dim, false, l.rope_theta, 1.0, pos, s);
            } else {
                var k_4d_raw = try k_raw.reshape(&.{ 1, seq_len, l.num_kv_heads, l.head_dim }, s);
                defer k_4d_raw.deinit();
                var v_4d_raw = try v_raw.reshape(&.{ 1, seq_len, l.num_kv_heads, l.head_dim }, s);
                defer v_4d_raw.deinit();

                var k_normed = try mlx.Array.fastRmsNorm(k_4d_raw, l.k_norm_w.?, l.rms_norm_eps, s);
                defer k_normed.deinit();
                var v_normed = try mlx.Array.fastRmsNorm(v_4d_raw, null, l.rms_norm_eps, s);
                defer v_normed.deinit();

                var k_4d = try k_normed.transposeAxes(&.{ 0, 2, 1, 3 }, s);
                defer k_4d.deinit();
                v_4d = try v_normed.transposeAxes(&.{ 0, 2, 1, 3 }, s);
                errdefer v_4d.deinit();

                k_rot = try mlx.Array.fastRoPE(k_4d, l.rot_dim, false, l.rope_theta, 1.0, pos, s);
            }
            defer v_4d.deinit();
            defer k_rot.deinit();

            if (self.gpu_kv[l_idx] == null) {
                const initial_cap = @max(256, ((total_len + 255) / 256) * 256);
                self.gpu_kv[l_idx] = try GpuKvCache.init(initial_cap, l.num_kv_heads, l.head_dim, s);
            }
            const cache = &self.gpu_kv[l_idx].?;
            try cache.ensureCapacity(total_len, s);
            const next_k = try cache.k.sliceUpdate4dAxis2(k_rot, l.num_kv_heads, pos, total_len, l.head_dim, s);
            const next_v = try cache.v.sliceUpdate4dAxis2(v_4d, l.num_kv_heads, pos, total_len, l.head_dim, s);
            cache.k.deinit();
            cache.v.deinit();
            cache.k = next_k;
            cache.v = next_v;

            var start_k: usize = 0;
            if (l.is_sliding and total_len > l.sliding_window) {
                start_k = total_len - l.sliding_window;
            }
            eff_k = try cache.k.slice4dAxis2(l.num_kv_heads, start_k, total_len, l.head_dim, s);
            eff_v = try cache.v.slice4dAxis2(l.num_kv_heads, start_k, total_len, l.head_dim, s);
            eff_k_free = eff_k;
            eff_v_free = eff_v;
        } else {
            const cache = &self.gpu_kv[l.prev_kv_idx].?;
            var start_k: usize = 0;
            if (l.is_sliding and total_len > l.sliding_window) {
                start_k = total_len - l.sliding_window;
            }
            eff_k = try cache.k.slice4dAxis2(l.num_kv_heads, start_k, total_len, l.head_dim, s);
            eff_v = try cache.v.slice4dAxis2(l.num_kv_heads, start_k, total_len, l.head_dim, s);
            eff_k_free = eff_k;
            eff_v_free = eff_v;
        }

        // 缩放点积注意力 (Scaled Dot-Product Attention, SDPA, scaling = 1.0)
        const is_causal = (seq_len > 1);
        var attn_out_4d = try mlx.Array.fastScaledDotProductAttention(q_rot, eff_k, eff_v, 1.0, is_causal, s);
        defer attn_out_4d.deinit();

        var attn_2d = if (seq_len == 1)
            try attn_out_4d.reshape(&.{ 1, 1, l.num_heads * l.head_dim }, s)
        else blk: {
            var attn_trans = try attn_out_4d.transposeAxes(&.{ 0, 2, 1, 3 }, s);
            defer attn_trans.deinit();
            break :blk try attn_trans.reshape(&.{ 1, seq_len, l.num_heads * l.head_dim }, s);
        };
        defer attn_2d.deinit();

        var o_out = try l.o_proj.forwardArray(attn_2d, s);
        defer o_out.deinit();

        var post_attn = try mlx.Array.fastRmsNorm(o_out, l.post_attn_w, l.rms_norm_eps, s);
        defer post_attn.deinit();
        var x1 = try mlx.Array.add(x_in, post_attn, s);
        defer x1.deinit();

        // (3) 门控多层感知机 (Gated MLP: Fused Compiled GeGLU)
        var pre_mlp = try mlx.Array.fastRmsNorm(x1, l.pre_mlp_w, l.rms_norm_eps, s);
        defer pre_mlp.deinit();

        var gate_out = try l.gate_proj.forwardArray(pre_mlp, s);
        defer gate_out.deinit();
        var up_out = try l.up_proj.forwardArray(pre_mlp, s);
        defer up_out.deinit();

        var hidden = try mlx.Array.fusedGeluMul(gate_out, up_out, self.compiled_gelu_mul, s);
        defer hidden.deinit();

        var down_out = try l.down_proj.forwardArray(hidden, s);
        defer down_out.deinit();

        var post_mlp = try mlx.Array.fastRmsNorm(down_out, l.post_mlp_w, l.rms_norm_eps, s);
        defer post_mlp.deinit();
        var x2 = try mlx.Array.add(x1, post_mlp, s);
        defer x2.deinit();

        // (4) 逐层输入嵌入增强 (Per-Layer Inputs, PLE: Fused Compiled GeGLU)
        var ple_gate_raw = try l.ple_gate.forwardArray(x2, s);
        defer ple_gate_raw.deinit();

        var ple_modulated = try mlx.Array.fusedGeluMul(ple_gate_raw, ple_slice, self.compiled_gelu_mul, s);
        defer ple_modulated.deinit();
        var ple_projected = try l.ple_proj.forwardArray(ple_modulated, s);
        defer ple_projected.deinit();

        var ple_normed = try mlx.Array.fastRmsNorm(ple_projected, l.ple_norm_w, l.rms_norm_eps, s);
        defer ple_normed.deinit();
        var x3 = try mlx.Array.add(x2, ple_normed, s);

        // (5) 层缩放标量 (Layer Scalar, 仅当 != 1.0 时执行 BF16 张量乘法)
        if (l.layer_scalar_w) |sw| {
            defer x3.deinit();
            return mlx.Array.multiply(x3, sw, s);
        }
        return x3;
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const io = init.io;

    var custom_prompt: ?[]const u8 = null;
    var max_new_tokens: usize = 40;
    var repetition_penalty: f32 = 1.0;
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
                if (it.next()) |r_str| {
                    repetition_penalty = std.fmt.parseFloat(f32, r_str) catch 1.0;
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

    var first_tok_arr_opt: ?mlx.Array = if (repetition_penalty == 1.0)
        try model.prefillAsync(prompt_tokens)
    else
        null;
    defer if (first_tok_arr_opt) |*a| a.deinit();

    const first_tok: usize = if (first_tok_arr_opt) |a|
        @intCast(a.itemUint32())
    else
        try model.prefill(prompt_tokens, repetition_penalty);

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

    // Decode 循环：当 repetition_penalty == 1.0 时启用双阶段 GPU 异步流水线 (2-Stage Async GPU Pipelining)，
    // 使 CPU 构建第 t+1 步计算图与 GPU 执行第 t 步 Metal 内核完全重叠，彻底消除步间气泡
    var current_tok = first_tok;
    const decode_start_ns = p_end;

    if (repetition_penalty == 1.0) {
        if (gen_count < max_new_tokens and current_tok != gemma4.Gemma4Vocabulary.EOS and current_tok != gemma4.Gemma4Vocabulary.EOT) {
            var pending_tok_arr = try model.decodeStepAsync(first_tok_arr_opt.?);
            first_tok_arr_opt.?.deinit();
            first_tok_arr_opt = null;

            while (gen_count < max_new_tokens) {
                // 在等待当前步 GPU 完成前，直接以 GPU 词元张量预构建并异步派发下一步计算图
                var next_pending_opt: ?mlx.Array = if (gen_count + 1 < max_new_tokens)
                    try model.decodeStepAsync(pending_tok_arr)
                else
                    null;

                const next_tok: usize = @intCast(pending_tok_arr.itemUint32());
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
        }
    } else {
        while (gen_count < max_new_tokens) {
            if (current_tok == gemma4.Gemma4Vocabulary.EOS or current_tok == gemma4.Gemma4Vocabulary.EOT) {
                break;
            }

            const next_tok = try model.decodeStep(current_tok, current_tokens.items, repetition_penalty);
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
    std.debug.print("   - 自回归解码速率     : {d:.2} token/s (纯 Metal GPU 流水线)\n", .{decode_speed});
    std.debug.print("   - 端到端总生成耗时   : {d:.2} 秒\n", .{total_time_s});
    std.debug.print("================================================================================\n\n", .{});
}
