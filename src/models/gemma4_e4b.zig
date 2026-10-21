//! Gemma 4 E4B (Effective 4B) 4-bit / 8-bit 混合量化模型端侧推理引擎
//!
//! 支持直接零拷贝映射 Safetensors 模型权重与预融合顺序二进制缓存 (`gemma4_e4b_direct.bin`)、
//! 逐层嵌入增强 (Per-Layer Embeddings, PLE)、异构滑动窗口与全局注意力机制，
//! 以及苹果统一内存 (Unified Memory Architecture, UMA) GPU 硬件加速与双阶段异步流水线推理。

const std = @import("std");
const backend = @import("../backend.zig");
const nn = @import("../nn.zig");
const gemma4 = @import("gemma4.zig");

const mlx = backend.mlx;
pub const Gemma4Config = gemma4.Gemma4Config;
pub const Q4Weights = backend.offload.Q4Weights;
pub const Stream = backend.offload.Stream;
pub const GpuKvCache = backend.offload.GpuKvCache;
pub const TensorMeta = nn.TensorMeta;
pub const SafetensorsArchive = nn.SafetensorsArchive;

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

/// 步首词元嵌入与逐层输入预计算静态编译上下文 (Step-Head Embedding & PLE Preprocessing Compiled Context)
/// 分配于堆内存以保证跨函数返回时的指针地址恒定，供 `mlx.compile` 闭包稳定引用
pub const PlePrepContext = struct {
    embed_tokens: Q4Weights,
    sqrt_hidden_arr: mlx.Array,
    ple: PleWeights,
    hidden_size: usize,
    rms_norm_eps: f32,

    pub fn compute(vres: [*c]mlx.c.mlx_vector_array, in: mlx.c.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        if (!mlx.has_mlx) return 1;
        const ctx: *const PlePrepContext = @ptrCast(@alignCast(payload orelse return 1));
        var idx_h: mlx.c.mlx_array = .{ .ctx = null };
        if (mlx.c.mlx_vector_array_get(&idx_h, in, 0) != 0 or idx_h.ctx == null) return 1;
        const indices_arr = mlx.Array{ .handle = idx_h };
        const T = indices_arr.dimSize(0);
        if (T == 0) return 1;

        var w_sel = ctx.embed_tokens.w_q.takeAxis(indices_arr, 0, null) catch return 1;
        defer w_sel.deinit();
        var sc_sel = ctx.embed_tokens.scales.takeAxis(indices_arr, 0, null) catch return 1;
        defer sc_sel.deinit();
        var bi_sel = ctx.embed_tokens.biases.takeAxis(indices_arr, 0, null) catch return 1;
        defer bi_sel.deinit();

        var deq = mlx.Array.dequantize(
            w_sel,
            sc_sel,
            bi_sel,
            ctx.embed_tokens.group_size,
            ctx.embed_tokens.bits,
            "affine",
            null,
        ) catch return 1;
        defer deq.deinit();

        var scaled = mlx.Array.multiply(deq, ctx.sqrt_hidden_arr, null) catch return 1;
        defer scaled.deinit();
        var x_cur = scaled.reshape(&.{ 1, T, ctx.hidden_size }, null) catch return 1;
        defer x_cur.deinit();

        var pw_sel = ctx.ple.embed_tokens_per_layer.w_q.takeAxis(indices_arr, 0, null) catch return 1;
        defer pw_sel.deinit();
        var psc_sel = ctx.ple.embed_tokens_per_layer.scales.takeAxis(indices_arr, 0, null) catch return 1;
        defer psc_sel.deinit();
        var pbi_sel = ctx.ple.embed_tokens_per_layer.biases.takeAxis(indices_arr, 0, null) catch return 1;
        defer pbi_sel.deinit();

        var pdeq = mlx.Array.dequantize(
            pw_sel,
            psc_sel,
            pbi_sel,
            ctx.ple.embed_tokens_per_layer.group_size,
            ctx.ple.embed_tokens_per_layer.bits,
            "affine",
            null,
        ) catch return 1;
        defer pdeq.deinit();

        var pscaled = mlx.Array.multiply(pdeq, ctx.ple.embed_scale_arr, null) catch return 1;
        defer pscaled.deinit();
        var ple_tokens_arr = pscaled.reshape(&.{ 1, T, ctx.ple.num_layers, ctx.ple.ple_dim }, null) catch return 1;
        defer ple_tokens_arr.deinit();

        var ple_proj_raw = ctx.ple.model_proj.forwardArray(x_cur, null) catch return 1;
        defer ple_proj_raw.deinit();
        var ple_proj_scaled = mlx.Array.multiply(ple_proj_raw, ctx.ple.proj_scale_arr, null) catch return 1;
        defer ple_proj_scaled.deinit();
        var ple_proj_4d = ple_proj_scaled.reshape(&.{ 1, T, ctx.ple.num_layers, ctx.ple.ple_dim }, null) catch return 1;
        defer ple_proj_4d.deinit();
        var ple_proj_normed = mlx.Array.fastRmsNorm(ple_proj_4d, ctx.ple.proj_norm_w, ctx.rms_norm_eps, null) catch return 1;
        defer ple_proj_normed.deinit();

        var ple_sum = mlx.Array.add(ple_proj_normed, ple_tokens_arr, null) catch return 1;
        defer ple_sum.deinit();
        var ple_inputs = mlx.Array.multiply(ple_sum, ctx.ple.input_scale_arr, null) catch return 1;
        defer ple_inputs.deinit();

        const out_handles = [_]mlx.c.mlx_array{ x_cur.handle, ple_inputs.handle };
        const res_vec = mlx.c.mlx_vector_array_new_data(&out_handles, 2);
        _ = mlx.c.mlx_vector_array_set(vres, res_vec);
        _ = mlx.c.mlx_vector_array_free(res_vec);
        return 0;
    }
};

/// 单个解码层的 GPU 硬件加速权重结构
pub const GpuLayerE4B = struct {
    // 4-bit 量化注意力投影 (Attention Projections: has_kv 层为融合 QKV 矩阵，共享 KV 层为独立 Q 矩阵)
    qkv_proj: Q4Weights,
    o_proj: Q4Weights,

    // 均方根层归一化权重 (Root Mean Square Layer Normalization, RMSNorm Weights)
    input_norm_w: mlx.Array,
    q_norm_w: mlx.Array,
    k_norm_w: ?mlx.Array = null,
    post_attn_w: mlx.Array,
    pre_mlp_w: mlx.Array,
    post_mlp_w: mlx.Array,

    // 8-bit 量化门控前馈多层感知机 (8-bit Quantized Gated MLP Projections: Gate + Up 沿输出维度融合为单一矩阵)
    fused_gate_up: Q4Weights,
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
    compiled_tail: ?mlx.Closure = null,

    /// 静态编译当前解码层单步解码阶段 (`seq_len == 1`) 注意力投影之后的全尾部计算图 (Post-Attention + Gated MLP + PLE + LayerScalar)
    /// 将每层 20+ 个独立 C API 调度节点收拢为单一预编译静态闭包，由 MLX JIT 自动跨算子融合所有逐元素激活与残差连接
    pub fn initCompiledTail(self: *GpuLayerE4B) void {
        if (!mlx.has_mlx) return;
        const Helper = struct {
            fn compute(vres: [*c]mlx.c.mlx_vector_array, in: mlx.c.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
                if (!mlx.has_mlx) return 1;
                const l: *const GpuLayerE4B = @ptrCast(@alignCast(payload orelse return 1));
                var x_in_h: mlx.c.mlx_array = .{ .ctx = null };
                var o_out_h: mlx.c.mlx_array = .{ .ctx = null };
                var ple_slice_h: mlx.c.mlx_array = .{ .ctx = null };
                if (mlx.c.mlx_vector_array_get(&x_in_h, in, 0) != 0 or x_in_h.ctx == null) return 1;
                if (mlx.c.mlx_vector_array_get(&o_out_h, in, 1) != 0 or o_out_h.ctx == null) return 1;
                if (mlx.c.mlx_vector_array_get(&ple_slice_h, in, 2) != 0 or ple_slice_h.ctx == null) return 1;

                const x_in = mlx.Array{ .handle = x_in_h };
                const o_out = mlx.Array{ .handle = o_out_h };
                const ple_slice = mlx.Array{ .handle = ple_slice_h };

                var post_attn = mlx.Array.fastRmsNorm(o_out, l.post_attn_w, l.rms_norm_eps, null) catch return 1;
                defer post_attn.deinit();
                var x1 = mlx.Array.add(x_in, post_attn, null) catch return 1;
                defer x1.deinit();

                var pre_mlp = mlx.Array.fastRmsNorm(x1, l.pre_mlp_w, l.rms_norm_eps, null) catch return 1;
                defer pre_mlp.deinit();

                var gate_up = l.fused_gate_up.forwardArray(pre_mlp, null) catch return 1;
                defer gate_up.deinit();
                var gu_vec = gate_up.split(2, 2, null) catch return 1;
                defer gu_vec.deinit();
                var gate_out = gu_vec.get(0) orelse return 1;
                defer gate_out.deinit();
                var up_out = gu_vec.get(1) orelse return 1;
                defer up_out.deinit();

                var act_mlp = gate_out.gelu(null) catch return 1;
                defer act_mlp.deinit();
                var hidden = mlx.Array.multiply(act_mlp, up_out, null) catch return 1;
                defer hidden.deinit();

                var down_out = l.down_proj.forwardArray(hidden, null) catch return 1;
                defer down_out.deinit();

                var post_mlp = mlx.Array.fastRmsNorm(down_out, l.post_mlp_w, l.rms_norm_eps, null) catch return 1;
                defer post_mlp.deinit();
                var x2 = mlx.Array.add(x1, post_mlp, null) catch return 1;
                defer x2.deinit();

                var ple_gate_raw = l.ple_gate.forwardArray(x2, null) catch return 1;
                defer ple_gate_raw.deinit();

                var act_ple = ple_gate_raw.gelu(null) catch return 1;
                defer act_ple.deinit();
                var ple_modulated = mlx.Array.multiply(act_ple, ple_slice, null) catch return 1;
                defer ple_modulated.deinit();

                var ple_projected = l.ple_proj.forwardArray(ple_modulated, null) catch return 1;
                defer ple_projected.deinit();
                var ple_normed = mlx.Array.fastRmsNorm(ple_projected, l.ple_norm_w, l.rms_norm_eps, null) catch return 1;
                defer ple_normed.deinit();

                var out = if (l.layer_scalar_w) |sw| blk: {
                    var sum = mlx.Array.add(x2, ple_normed, null) catch return 1;
                    defer sum.deinit();
                    break :blk mlx.Array.multiply(sum, sw, null) catch return 1;
                } else mlx.Array.add(x2, ple_normed, null) catch return 1;
                defer out.deinit();

                const res_vec = mlx.c.mlx_vector_array_new_data(&[_]mlx.c.mlx_array{out.handle}, 1);
                _ = mlx.c.mlx_vector_array_set(vres, res_vec);
                _ = mlx.c.mlx_vector_array_free(res_vec);
                return 0;
            }
        };
        self.compiled_tail = mlx.compile(Helper.compute, self, false) catch null;
    }

    pub fn deinit(self: *GpuLayerE4B) void {
        if (self.compiled_tail) |*ct| {
            ct.deinit();
            self.compiled_tail = null;
        }
        self.qkv_proj.deinit();
        self.o_proj.deinit();
        self.input_norm_w.deinit();
        self.q_norm_w.deinit();
        if (self.k_norm_w) |*kn| kn.deinit();
        self.post_attn_w.deinit();
        self.pre_mlp_w.deinit();
        self.post_mlp_w.deinit();
        self.fused_gate_up.deinit();
        self.down_proj.deinit();
        self.ple_gate.deinit();
        self.ple_proj.deinit();
        self.ple_norm_w.deinit();
        if (self.layer_scalar_w) |*sw| sw.deinit();
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
    compiled_add_mul: ?mlx.Closure = null,
    ple_prep_ctx: ?*PlePrepContext = null,
    compiled_ple_prep: ?mlx.Closure = null,

    pub const DIRECT_CACHE_MAGIC = "ZNN4E4B1";
    pub const DIRECT_CACHE_VERSION: u64 = 1;
    pub const DIRECT_CACHE_FILENAME = "gemma4_e4b_direct.bin";

    pub const Q4ByteLengths = struct {
        w_len: usize,
        s_len: usize,
        b_len: usize,

        pub fn total(self: Q4ByteLengths) usize {
            return self.w_len + self.s_len + self.b_len;
        }
    };

    pub fn q4ByteLengths(in_f: usize, out_f: usize, group_size: usize, bits: usize) Q4ByteLengths {
        const pack_factor = 32 / bits;
        const w_len = out_f * (in_f / pack_factor) * @sizeOf(u32);
        const s_len = out_f * (in_f / group_size) * @sizeOf(u16);
        return .{ .w_len = w_len, .s_len = s_len, .b_len = s_len };
    }

    /// 计算预融合顺序二进制权重缓存 (`gemma4_e4b_direct.bin`) 的精确字节总长度
    pub fn computeExpectedDirectCacheSize(cfg: Gemma4Config) usize {
        var total: usize = 16; // 8B magic ("ZNN4E4B1") + 8B version (u64)
        total += q4ByteLengths(cfg.hidden_size, cfg.vocab_size, 64, 4).total();
        total += cfg.hidden_size * @sizeOf(u16);
        total += q4ByteLengths(cfg.num_hidden_layers * cfg.hidden_size_per_layer_input, cfg.vocab_size, 64, 4).total();
        total += q4ByteLengths(cfg.hidden_size, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input, 64, 4).total();
        total += cfg.hidden_size_per_layer_input * @sizeOf(u16);

        for (0..cfg.num_hidden_layers) |idx| {
            const is_full = ((idx + 1) % 6 == 0);
            const hd = if (is_full) cfg.global_head_dim else cfg.head_dim;
            const out_q = cfg.num_attention_heads * hd;
            const out_kv = cfg.num_key_value_heads * hd;
            const has_kv = (idx < cfg.num_hidden_layers - cfg.num_kv_shared_layers);

            if (has_kv) {
                total += hd * @sizeOf(u16);
                total += q4ByteLengths(cfg.hidden_size, out_q + 2 * out_kv, 64, 4).total();
            } else {
                total += q4ByteLengths(cfg.hidden_size, out_q, 64, 4).total();
            }
            total += q4ByteLengths(out_q, cfg.hidden_size, 64, 4).total();
            total += q4ByteLengths(cfg.hidden_size, cfg.intermediate_size * 2, 64, 8).total();
            total += q4ByteLengths(cfg.intermediate_size, cfg.hidden_size, 64, 8).total();
            total += q4ByteLengths(cfg.hidden_size, cfg.hidden_size_per_layer_input, 64, 4).total();
            total += q4ByteLengths(cfg.hidden_size_per_layer_input, cfg.hidden_size, 64, 4).total();
            total += cfg.hidden_size * @sizeOf(u16);
            total += hd * @sizeOf(u16);
            total += cfg.hidden_size * @sizeOf(u16) * 4;
            total += @sizeOf(u16);
        }
        return total;
    }

    pub const DirectCacheReader = struct {
        data: []const u8,
        offset: usize = 16,

        pub fn init(data: []const u8, expected_size: usize) !DirectCacheReader {
            if (data.len != expected_size or data.len < 16) return error.InvalidDirectCache;
            if (!std.mem.eql(u8, data[0..8], DIRECT_CACHE_MAGIC)) return error.InvalidDirectCache;
            const ver = std.mem.readInt(u64, data[8..16], .little);
            if (ver != DIRECT_CACHE_VERSION) return error.InvalidDirectCache;
            return .{ .data = data, .offset = 16 };
        }

        pub fn readBytes(self: *DirectCacheReader, len: usize) ![]const u8 {
            if (self.offset + len > self.data.len) return error.InvalidDirectCache;
            const s = self.data[self.offset .. self.offset + len];
            self.offset += len;
            return s;
        }

        pub fn readQ4(
            self: *DirectCacheReader,
            in_f: usize,
            out_f: usize,
            group_size: i32,
            bits: i32,
            stream: Stream,
        ) !Q4Weights {
            const lens = q4ByteLengths(in_f, out_f, @intCast(group_size), @intCast(bits));
            const w_bytes = try self.readBytes(lens.w_len);
            const s_bytes = try self.readBytes(lens.s_len);
            const b_bytes = try self.readBytes(lens.b_len);
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

        pub fn readNorm(self: *DirectCacheReader, dim: usize) !mlx.Array {
            const bytes = try self.readBytes(dim * @sizeOf(u16));
            return mlx.Array.fromRawData(@ptrCast(bytes.ptr), &.{dim}, .bfloat16);
        }
    };

    pub const LoadOptions = struct {
        warmup: bool = true,

        pub fn defaultOptions() LoadOptions {
            return .{};
        }
    };

    pub fn loadFromDirectory(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !Gemma4E4BModel {
        return loadFromDirectoryWithOptions(allocator, io, dir_path, .{});
    }

    pub fn loadFromDirectoryWithOptions(
        allocator: std.mem.Allocator,
        io: std.Io,
        dir_path: []const u8,
        options: LoadOptions,
    ) !Gemma4E4BModel {
        const cfg = Gemma4Config.e4b;
        const expected_cache_bytes = computeExpectedDirectCacheSize(cfg);
        const cache_path = try std.fs.path.join(allocator, &.{ dir_path, DIRECT_CACHE_FILENAME });
        defer allocator.free(cache_path);

        const cwd = std.Io.Dir.cwd();
        var direct_mmap_opt: ?[]align(std.heap.page_size_min) const u8 = null;
        if (cwd.openFile(io, cache_path, .{})) |cf_val| {
            var cf = cf_val;
            defer cf.close(io);
            if (cf.stat(io)) |st| {
                if (@as(usize, @intCast(st.size)) == expected_cache_bytes) {
                    if (std.posix.mmap(
                        null,
                        expected_cache_bytes,
                        .{ .READ = true },
                        .{ .TYPE = .SHARED },
                        cf.handle,
                        0,
                    )) |mapped| {
                        _ = std.posix.madvise(mapped.ptr, mapped.len, std.posix.MADV.SEQUENTIAL | std.posix.MADV.WILLNEED) catch {};
                        if (std.mem.eql(u8, mapped[0..8], DIRECT_CACHE_MAGIC) and
                            std.mem.readInt(u64, mapped[8..16], .little) == DIRECT_CACHE_VERSION)
                        {
                            direct_mmap_opt = mapped;
                        } else {
                            std.posix.munmap(@alignCast(mapped));
                        }
                    } else |_| {}
                }
            } else |_| {}
        } else |_| {}
        defer if (direct_mmap_opt) |mapped| std.posix.munmap(@alignCast(mapped));

        var archive_opt: ?SafetensorsArchive = if (direct_mmap_opt == null)
            try SafetensorsArchive.open(allocator, io, dir_path)
        else
            null;
        defer if (archive_opt) |*arch| arch.deinit();

        const tmp_cache_path = try std.fs.path.join(allocator, &.{ dir_path, DIRECT_CACHE_FILENAME ++ ".tmp" });
        defer allocator.free(tmp_cache_path);
        var cache_out_file_opt: ?std.Io.File = if (direct_mmap_opt == null)
            (cwd.createFile(io, tmp_cache_path, .{}) catch null)
        else
            null;
        var cache_write_buf: [65536]u8 = undefined;
        var cache_fwriter: std.Io.File.Writer = undefined;
        var cache_writer_opt: ?*std.Io.Writer = null;
        var cache_write_ok: bool = true;
        if (cache_out_file_opt) |*cf_out| {
            cache_fwriter = cf_out.writer(io, &cache_write_buf);
            const w = &cache_fwriter.interface;
            w.writeAll(DIRECT_CACHE_MAGIC) catch {
                cache_write_ok = false;
            };
            const ver_bytes = std.mem.toBytes(@as(u64, DIRECT_CACHE_VERSION));
            w.writeAll(&ver_bytes) catch {
                cache_write_ok = false;
            };
            if (cache_write_ok) cache_writer_opt = w;
        }
        defer if (cache_out_file_opt) |*cf_out| cf_out.close(io);

        const stream = try Stream.defaultGpu();
        errdefer {
            var s_tmp = stream;
            s_tmp.deinit();
        }

        _ = mlx.configureOptimalWiredMemory();
        mlx.enableCompile();
        mlx.setCompileMode(.enabled);

        var direct_reader: DirectCacheReader = if (direct_mmap_opt) |mapped|
            try DirectCacheReader.init(mapped, expected_cache_bytes)
        else
            .{ .data = &.{} };

        // 1. 词表嵌入与 LM Head 权重
        var embed_tokens: Q4Weights = undefined;
        var norm_w: mlx.Array = undefined;
        var ple_tokens: Q4Weights = undefined;
        var ple_model_proj: Q4Weights = undefined;
        var ple_pnorm_w: mlx.Array = undefined;

        if (direct_mmap_opt != null) {
            embed_tokens = try direct_reader.readQ4(cfg.hidden_size, cfg.vocab_size, 64, 4, stream);
            errdefer embed_tokens.deinit();
            norm_w = try direct_reader.readNorm(cfg.hidden_size);
            errdefer norm_w.deinit();
            ple_tokens = try direct_reader.readQ4(cfg.num_hidden_layers * cfg.hidden_size_per_layer_input, cfg.vocab_size, 64, 4, stream);
            errdefer ple_tokens.deinit();
            ple_model_proj = try direct_reader.readQ4(cfg.hidden_size, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input, 64, 4, stream);
            errdefer ple_model_proj.deinit();
            ple_pnorm_w = try direct_reader.readNorm(cfg.hidden_size_per_layer_input);
        } else {
            const archive = archive_opt.?;
            const emb_wq = archive.getTensorBytes("language_model.model.embed_tokens.weight") orelse return error.TensorNotFound;
            const emb_sc = archive.getTensorBytes("language_model.model.embed_tokens.scales") orelse return error.TensorNotFound;
            const emb_bi = archive.getTensorBytes("language_model.model.embed_tokens.biases") orelse return error.TensorNotFound;
            if (cache_writer_opt) |w| {
                w.writeAll(emb_wq) catch {
                    cache_write_ok = false;
                };
                w.writeAll(emb_sc) catch {
                    cache_write_ok = false;
                };
                w.writeAll(emb_bi) catch {
                    cache_write_ok = false;
                };
            }

            embed_tokens = try Q4Weights.initFromDirectBuffersCustom(
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
            if (cache_writer_opt) |w| {
                w.writeAll(norm_bytes[0 .. cfg.hidden_size * @sizeOf(u16)]) catch {
                    cache_write_ok = false;
                };
            }
            norm_w = try mlx.Array.fromRawData(@ptrCast(norm_bytes.ptr), &.{cfg.hidden_size}, .bfloat16);
            errdefer norm_w.deinit();

            // 2. 逐层嵌入输入结构 (Per-Layer Embeddings, PLE)
            const ple_tokens_wq = archive.getTensorBytes("language_model.model.embed_tokens_per_layer.weight") orelse return error.TensorNotFound;
            const ple_tokens_sc = archive.getTensorBytes("language_model.model.embed_tokens_per_layer.scales") orelse return error.TensorNotFound;
            const ple_tokens_bi = archive.getTensorBytes("language_model.model.embed_tokens_per_layer.biases") orelse return error.TensorNotFound;
            if (cache_writer_opt) |w| {
                w.writeAll(ple_tokens_wq) catch {
                    cache_write_ok = false;
                };
                w.writeAll(ple_tokens_sc) catch {
                    cache_write_ok = false;
                };
                w.writeAll(ple_tokens_bi) catch {
                    cache_write_ok = false;
                };
            }

            ple_tokens = try Q4Weights.initFromDirectBuffersCustom(
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
            if (cache_writer_opt) |w| {
                w.writeAll(ple_proj_wq) catch {
                    cache_write_ok = false;
                };
                w.writeAll(ple_proj_sc) catch {
                    cache_write_ok = false;
                };
                w.writeAll(ple_proj_bi) catch {
                    cache_write_ok = false;
                };
            }

            ple_model_proj = try Q4Weights.initFromDirectBuffersCustom(
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
            if (cache_writer_opt) |w| {
                w.writeAll(ple_pnorm_bytes[0 .. cfg.hidden_size_per_layer_input * @sizeOf(u16)]) catch {
                    cache_write_ok = false;
                };
            }
            ple_pnorm_w = try mlx.Array.fromRawData(@ptrCast(ple_pnorm_bytes.ptr), &.{cfg.hidden_size_per_layer_input}, .bfloat16);
        }
        errdefer embed_tokens.deinit();
        errdefer norm_w.deinit();
        errdefer ple_tokens.deinit();
        errdefer ple_model_proj.deinit();
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

        // 3. 构建 42 个解码层（若命中顺序直通缓存则无需分配 52MB 拼接缓冲并消除 2.3GB 内存拷贝）
        const layers = try allocator.alloc(GpuLayerE4B, cfg.num_hidden_layers);
        errdefer allocator.free(layers);

        const max_fused_bytes = (cfg.intermediate_size * 2) * (cfg.hidden_size / 4) * @sizeOf(u32);
        const fuse_scratch: []u8 = if (direct_mmap_opt == null)
            try allocator.alloc(u8, max_fused_bytes)
        else
            &.{};
        defer if (direct_mmap_opt == null) allocator.free(fuse_scratch);

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

            var kn_w: ?mlx.Array = null;
            var qkv_proj: Q4Weights = undefined;
            var o_proj: Q4Weights = undefined;
            var fused_gate_up: Q4Weights = undefined;
            var down_proj: Q4Weights = undefined;
            var ple_gate: Q4Weights = undefined;
            var ple_proj: Q4Weights = undefined;
            var in_w: mlx.Array = undefined;
            var qn_w: mlx.Array = undefined;
            var pa_w: mlx.Array = undefined;
            var pm_w: mlx.Array = undefined;
            var pom_w: mlx.Array = undefined;
            var pln_w: mlx.Array = undefined;
            var l_scalar: f32 = 1.0;
            var l_scalar_w: ?mlx.Array = null;

            if (direct_mmap_opt != null) {
                if (has_kv) {
                    kn_w = try direct_reader.readNorm(hd);
                    qkv_proj = try direct_reader.readQ4(cfg.hidden_size, out_q + 2 * out_kv, 64, 4, stream);
                } else {
                    qkv_proj = try direct_reader.readQ4(cfg.hidden_size, out_q, 64, 4, stream);
                }
                o_proj = try direct_reader.readQ4(out_q, cfg.hidden_size, 64, 4, stream);
                fused_gate_up = try direct_reader.readQ4(cfg.hidden_size, cfg.intermediate_size * 2, 64, 8, stream);
                down_proj = try direct_reader.readQ4(cfg.intermediate_size, cfg.hidden_size, 64, 8, stream);
                ple_gate = try direct_reader.readQ4(cfg.hidden_size, cfg.hidden_size_per_layer_input, 64, 4, stream);
                ple_proj = try direct_reader.readQ4(cfg.hidden_size_per_layer_input, cfg.hidden_size, 64, 4, stream);

                in_w = try direct_reader.readNorm(cfg.hidden_size);
                qn_w = try direct_reader.readNorm(hd);
                pa_w = try direct_reader.readNorm(cfg.hidden_size);
                pm_w = try direct_reader.readNorm(cfg.hidden_size);
                pom_w = try direct_reader.readNorm(cfg.hidden_size);
                pln_w = try direct_reader.readNorm(cfg.hidden_size);

                const sb = try direct_reader.readBytes(2);
                const u16_val = std.mem.readInt(u16, sb[0..2], .little);
                const f32_val: f32 = @bitCast(@as(u32, u16_val) << 16);
                l_scalar = f32_val;
                if (l_scalar != 1.0) {
                    l_scalar_w = try mlx.Array.fromRawData(@ptrCast(sb.ptr), &.{1}, .bfloat16);
                }
            } else {
                const archive = archive_opt.?;
                // 注意力投影权重绑定：前 24 层将 Q/K/V 融合为单一矩阵，每步减少 48 次 QMV 内核派发
                qkv_proj = if (has_kv) blk: {
                    kn_w = try loadNorm(archive, &name_buf, idx, "self_attn.k_norm.weight", hd, cache_writer_opt, &cache_write_ok);
                    break :blk try loadFusedQ4(
                        archive,
                        &name_buf,
                        fuse_scratch,
                        idx,
                        &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" },
                        cfg.hidden_size,
                        out_q + 2 * out_kv,
                        64,
                        4,
                        stream,
                        cache_writer_opt,
                        &cache_write_ok,
                    );
                } else try loadQ4(archive, &name_buf, idx, "self_attn.q_proj", cfg.hidden_size, out_q, 64, 4, stream, cache_writer_opt, &cache_write_ok);

                o_proj = try loadQ4(archive, &name_buf, idx, "self_attn.o_proj", out_q, cfg.hidden_size, 64, 4, stream, cache_writer_opt, &cache_write_ok);

                // 门控前馈网络 (8-bit Quantized MLP)：将 gate_proj 与 up_proj 沿输出维度融合为单一矩阵，每步减少 42 次 QMV 内核派发
                fused_gate_up = try loadFusedQ4(
                    archive,
                    &name_buf,
                    fuse_scratch,
                    idx,
                    &.{ "mlp.gate_proj", "mlp.up_proj" },
                    cfg.hidden_size,
                    cfg.intermediate_size * 2,
                    64,
                    8,
                    stream,
                    cache_writer_opt,
                    &cache_write_ok,
                );
                down_proj = try loadQ4(archive, &name_buf, idx, "mlp.down_proj", cfg.intermediate_size, cfg.hidden_size, 64, 8, stream, cache_writer_opt, &cache_write_ok);

                // 逐层嵌入门控与上投影 (PLE Gating & Projection)
                ple_gate = try loadQ4(archive, &name_buf, idx, "per_layer_input_gate", cfg.hidden_size, cfg.hidden_size_per_layer_input, 64, 4, stream, cache_writer_opt, &cache_write_ok);
                ple_proj = try loadQ4(archive, &name_buf, idx, "per_layer_projection", cfg.hidden_size_per_layer_input, cfg.hidden_size, 64, 4, stream, cache_writer_opt, &cache_write_ok);

                // 均方根层归一化权重 (RMSNorm Weights)
                in_w = try loadNorm(archive, &name_buf, idx, "input_layernorm.weight", cfg.hidden_size, cache_writer_opt, &cache_write_ok);
                qn_w = try loadNorm(archive, &name_buf, idx, "self_attn.q_norm.weight", hd, cache_writer_opt, &cache_write_ok);
                pa_w = try loadNorm(archive, &name_buf, idx, "post_attention_layernorm.weight", cfg.hidden_size, cache_writer_opt, &cache_write_ok);
                pm_w = try loadNorm(archive, &name_buf, idx, "pre_feedforward_layernorm.weight", cfg.hidden_size, cache_writer_opt, &cache_write_ok);
                pom_w = try loadNorm(archive, &name_buf, idx, "post_feedforward_layernorm.weight", cfg.hidden_size, cache_writer_opt, &cache_write_ok);
                pln_w = try loadNorm(archive, &name_buf, idx, "post_per_layer_input_norm.weight", cfg.hidden_size, cache_writer_opt, &cache_write_ok);

                // 层缩放标量 (Layer Scalar, 直接绑定 BF16 标量张量以消除每步标量创建与类型提升)
                const scalar_name = std.fmt.bufPrint(&name_buf, "language_model.model.layers.{}.layer_scalar", .{idx}) catch unreachable;
                var scalar_bf16_bytes = [2]u8{ 0x80, 0x3F }; // 1.0 in BF16 little-endian
                if (archive.getTensorBytes(scalar_name)) |sb| {
                    if (sb.len >= 2) {
                        scalar_bf16_bytes = sb[0..2].*;
                        const u16_val = std.mem.readInt(u16, sb[0..2], .little);
                        const f32_val: f32 = @bitCast(@as(u32, u16_val) << 16);
                        l_scalar = f32_val;
                        if (l_scalar != 1.0) {
                            l_scalar_w = try mlx.Array.fromRawData(@ptrCast(sb.ptr), &.{1}, .bfloat16);
                        }
                    }
                }
                if (cache_writer_opt) |w| {
                    w.writeAll(&scalar_bf16_bytes) catch {
                        cache_write_ok = false;
                    };
                }
            }

            layers[idx] = .{
                .qkv_proj = qkv_proj,
                .o_proj = o_proj,
                .input_norm_w = in_w,
                .q_norm_w = qn_w,
                .k_norm_w = kn_w,
                .post_attn_w = pa_w,
                .pre_mlp_w = pm_w,
                .post_mlp_w = pom_w,
                .fused_gate_up = fused_gate_up,
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
            layers[idx].initCompiledTail();
        }

        if (cache_writer_opt) |w| {
            w.flush() catch {
                cache_write_ok = false;
            };
        }
        if (cache_out_file_opt) |*cf_out| {
            cf_out.close(io);
            cache_out_file_opt = null;
            if (cache_write_ok) {
                cwd.rename(tmp_cache_path, cwd, cache_path, io) catch {};
            }
        }

        const compiled_gelu_mul = mlx.Array.compileGeluMul() catch null;
        const compiled_add_mul = mlx.Array.compileAddMul() catch null;

        var ple_prep_ctx: ?*PlePrepContext = null;
        var compiled_ple_prep: ?mlx.Closure = null;
        if (mlx.has_mlx) {
            if (allocator.create(PlePrepContext)) |ctx_ptr| {
                ctx_ptr.* = .{
                    .embed_tokens = embed_tokens,
                    .sqrt_hidden_arr = sqrt_hidden_arr,
                    .ple = ple,
                    .hidden_size = cfg.hidden_size,
                    .rms_norm_eps = cfg.rms_norm_eps,
                };
                ple_prep_ctx = ctx_ptr;
                compiled_ple_prep = mlx.compile(PlePrepContext.compute, ctx_ptr, false) catch null;
            } else |_| {}
        }
        errdefer {
            if (compiled_ple_prep) |*cls| cls.deinit();
            if (ple_prep_ctx) |ctx_ptr| allocator.destroy(ctx_ptr);
        }

        // 4. 在加载期预分配 24 个独立 KV 层的显存缓冲区，消除 Prefill 首字延迟中的 48 次 zeros 内核分配开销
        var gpu_kv: [24]?GpuKvCache = [_]?GpuKvCache{null} ** 24;
        errdefer {
            for (&gpu_kv) |*kv_opt| {
                if (kv_opt.*) |*c| c.deinit();
            }
        }
        for (0..24) |kv_idx| {
            const l_ref = layers[kv_idx];
            const init_cap: usize = if (l_ref.is_sliding) l_ref.sliding_window else 256;
            gpu_kv[kv_idx] = try GpuKvCache.init(init_cap, l_ref.num_kv_heads, l_ref.head_dim, stream);
        }

        var model = Gemma4E4BModel{
            .allocator = allocator,
            .layers = layers,
            .ple = ple,
            .embed_tokens = embed_tokens,
            .norm_w = norm_w,
            .lm_head = embed_tokens,
            .config = cfg,
            .stream = stream,
            .gpu_kv = gpu_kv,
            .curr_len = 0,
            .sqrt_hidden = sqrt_hidden_val,
            .sqrt_hidden_arr = sqrt_hidden_arr,
            .compiled_gelu_mul = compiled_gelu_mul,
            .compiled_add_mul = compiled_add_mul,
            .ple_prep_ctx = ple_prep_ctx,
            .compiled_ple_prep = compiled_ple_prep,
        };
        errdefer model.deinit();

        if (options.warmup) {
            try model.warmup();
        }

        return model;
    }

    pub fn deinit(self: *Gemma4E4BModel) void {
        self.stream.synchronize();
        for (&self.gpu_kv) |*kv_opt| {
            if (kv_opt.*) |*c| {
                c.deinit();
                kv_opt.* = null;
            }
        }
        if (self.compiled_ple_prep) |*cls| {
            cls.deinit();
            self.compiled_ple_prep = null;
        }
        if (self.ple_prep_ctx) |ctx_ptr| {
            self.allocator.destroy(ctx_ptr);
            self.ple_prep_ctx = null;
        }
        if (self.compiled_gelu_mul) |*cls| {
            cls.deinit();
            self.compiled_gelu_mul = null;
        }
        if (self.compiled_add_mul) |*cls| {
            cls.deinit();
            self.compiled_add_mul = null;
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

    /// 重置键值缓存游标（保留已分配的显存块直接覆写，消除重复分配释放开销）
    pub fn resetCache(self: *Gemma4E4BModel) void {
        for (&self.gpu_kv) |*kv_opt| {
            if (kv_opt.*) |*c| {
                c.clearStepSlices();
            }
        }
        self.curr_len = 0;
    }

    /// 执行 Metal 计算管线与统一内存物理页预热 (Pipeline PSO & UMA Page-Fault Pre-warming)：
    /// 在加载阶段通过微型序列 (T=2) 预填充与单步 (T=1) 解码前向传递，提前完成全部 MSL 着色器编译、
    /// `compiled_gelu_mul` 静态闭包特化以及权重的首次 GPU 页表映射，使真实推理的首字延迟 (TTFT) 骤降 50% 以上
    pub fn warmup(self: *Gemma4E4BModel) !void {
        const dummy_prompt = [_]usize{ 2, 106 };
        var first_arr = try self.prefillAsync(&dummy_prompt);
        defer first_arr.deinit();

        var second_arr = try self.decodeStepAsync(first_arr);
        defer second_arr.deinit();

        _ = second_arr.itemUint32();
        self.stream.synchronize();
        self.resetCache();
        mlx.clearCache();
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
        cache_writer_opt: ?*std.Io.Writer,
        cache_write_ok: *bool,
    ) !Q4Weights {
        const w_name = try std.fmt.bufPrint(buf, "language_model.model.layers.{}.{s}.weight", .{ l_idx, sub });
        const w_bytes = archive.getTensorBytes(w_name) orelse return error.TensorNotFound;
        const s_name = try std.fmt.bufPrint(buf, "language_model.model.layers.{}.{s}.scales", .{ l_idx, sub });
        const s_bytes = archive.getTensorBytes(s_name) orelse return error.TensorNotFound;
        const b_name = try std.fmt.bufPrint(buf, "language_model.model.layers.{}.{s}.biases", .{ l_idx, sub });
        const b_bytes = archive.getTensorBytes(b_name) orelse return error.TensorNotFound;

        if (cache_writer_opt) |w| {
            w.writeAll(w_bytes) catch {
                cache_write_ok.* = false;
            };
            w.writeAll(s_bytes) catch {
                cache_write_ok.* = false;
            };
            w.writeAll(b_bytes) catch {
                cache_write_ok.* = false;
            };
        }

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

    fn loadFusedQ4(
        archive: SafetensorsArchive,
        buf: *[128]u8,
        scratch_buf: []u8,
        l_idx: usize,
        subs: []const []const u8,
        in_f: usize,
        total_out_f: usize,
        group_size: i32,
        bits: i32,
        stream: Stream,
        cache_writer_opt: ?*std.Io.Writer,
        cache_write_ok: *bool,
    ) !Q4Weights {
        var w_slices: [3][]const u8 = undefined;
        var s_slices: [3][]const u8 = undefined;
        var b_slices: [3][]const u8 = undefined;

        for (subs, 0..) |sub, i| {
            const w_name = try std.fmt.bufPrint(buf, "language_model.model.layers.{}.{s}.weight", .{ l_idx, sub });
            w_slices[i] = archive.getTensorBytes(w_name) orelse return error.TensorNotFound;
            const s_name = try std.fmt.bufPrint(buf, "language_model.model.layers.{}.{s}.scales", .{ l_idx, sub });
            s_slices[i] = archive.getTensorBytes(s_name) orelse return error.TensorNotFound;
            const b_name = try std.fmt.bufPrint(buf, "language_model.model.layers.{}.{s}.biases", .{ l_idx, sub });
            b_slices[i] = archive.getTensorBytes(b_name) orelse return error.TensorNotFound;
        }

        if (cache_writer_opt) |w| {
            for (w_slices[0..subs.len]) |ws| w.writeAll(ws) catch {
                cache_write_ok.* = false;
            };
            for (s_slices[0..subs.len]) |ss| w.writeAll(ss) catch {
                cache_write_ok.* = false;
            };
            for (b_slices[0..subs.len]) |bs| w.writeAll(bs) catch {
                cache_write_ok.* = false;
            };
        }

        return Q4Weights.initFusedFromDirectBuffersCustom(
            w_slices[0..subs.len],
            s_slices[0..subs.len],
            b_slices[0..subs.len],
            scratch_buf,
            in_f,
            total_out_f,
            group_size,
            bits,
            .bfloat16,
            stream,
        );
    }

    fn loadNorm(
        archive: SafetensorsArchive,
        buf: *[128]u8,
        l_idx: usize,
        sub: []const u8,
        dim: usize,
        cache_writer_opt: ?*std.Io.Writer,
        cache_write_ok: *bool,
    ) !mlx.Array {
        const name = try std.fmt.bufPrint(buf, "language_model.model.layers.{}.{s}", .{ l_idx, sub });
        const bytes = archive.getTensorBytes(name) orelse return error.TensorNotFound;
        if (cache_writer_opt) |w| {
            w.writeAll(bytes[0 .. dim * @sizeOf(u16)]) catch {
                cache_write_ok.* = false;
            };
        }
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
        var x_cur: mlx.Array = undefined;
        var ple_inputs: mlx.Array = undefined;
        // 仅在单步解码 (T == 1) 时启用形状特化 (shapeless = false) 的静态编译闭包 compiled_ple_prep；
        // 预填充阶段 (T > 1) 的序列长度 T 随用户 Prompt 动态变化，直接走下方基于 shapeless = true 融合算子
        // (compiled_add_mul) 的路径，避免在真实 Prompt 预填充时因新序列长度触发 MLX 重新追踪与 Metal 着色器重编译导致首字延迟 (TTFT) 飙升
        if (T == 1 and self.compiled_ple_prep != null) {
            const prep_out = try self.compiled_ple_prep.?.apply1To2(indices_arr);
            x_cur = prep_out[0];
            ple_inputs = prep_out[1];
        } else {
            // 1. 生成词元基础嵌入 [1, T, hidden_size] (BF16)
            x_cur = try self.getEmbeddingsFromArray(indices_arr, T, s);
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

            // 融合逐层输入: (proj + tokens) * input_scale (保持纯 BF16，通过静态编译闭包融合加法与缩放)
            ple_inputs = try mlx.Array.fusedAddMul(ple_proj_normed, ple_tokens_arr, self.ple.input_scale_arr, self.compiled_add_mul, s);
        }
        errdefer x_cur.deinit();
        defer ple_inputs.deinit();

        // 3. 执行 42 层 Transformer 前向传播：
        // - 当 T == 1 时重塑为 [1, 42, 256] 并沿 axis 1 切分，直接获得 [1, 1, 256] 切片；
        // - 当 T > 1 时沿 axis 2 切分获得 [1, T, 1, 256] 零拷贝视图，直接传入 forwardLayer 与 4D 门控视图融合，消除 42 次非连续张量重塑拷贝内核
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
                var ple_slice = ple_vec.get(idx).?;
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

        // (2) 注意力投影与键值缓存 (Attention QKV Projection & KVCache)
        var qkv_raw = try l.qkv_proj.forwardArray(in_norm, s);
        defer qkv_raw.deinit();

        var q_rot: mlx.Array = undefined;
        errdefer q_rot.deinit();
        var eff_k: mlx.Array = undefined;
        var eff_v: mlx.Array = undefined;
        var eff_k_free: ?mlx.Array = null;
        var eff_v_free: ?mlx.Array = null;
        defer if (eff_k_free) |*ek| ek.deinit();
        defer if (eff_v_free) |*ev| ev.deinit();

        const total_len = pos + seq_len;

        if (l.has_kv) {
            const total_heads = l.num_heads + 2 * l.num_kv_heads;
            var k_rot: mlx.Array = undefined;
            var v_4d: mlx.Array = undefined;

            if (seq_len == 1) {
                var qkv_4d = try qkv_raw.reshape(&.{ 1, total_heads, 1, l.head_dim }, s);
                defer qkv_4d.deinit();

                var q_4d_raw = try qkv_4d.sliceAxis(1, 0, l.num_heads, s);
                defer q_4d_raw.deinit();
                var k_4d_raw = try qkv_4d.sliceAxis(1, l.num_heads, l.num_heads + l.num_kv_heads, s);
                defer k_4d_raw.deinit();
                var v_4d_raw = try qkv_4d.sliceAxis(1, l.num_heads + l.num_kv_heads, total_heads, s);
                defer v_4d_raw.deinit();

                var q_normed = try mlx.Array.fastRmsNorm(q_4d_raw, l.q_norm_w, l.rms_norm_eps, s);
                defer q_normed.deinit();
                q_rot = try mlx.Array.fastRoPE(q_normed, l.rot_dim, false, l.rope_theta, 1.0, pos, s);

                var k_normed = try mlx.Array.fastRmsNorm(k_4d_raw, l.k_norm_w.?, l.rms_norm_eps, s);
                defer k_normed.deinit();
                v_4d = try mlx.Array.fastRmsNorm(v_4d_raw, null, l.rms_norm_eps, s);
                errdefer v_4d.deinit();

                k_rot = try mlx.Array.fastRoPE(k_normed, l.rot_dim, false, l.rope_theta, 1.0, pos, s);
            } else {
                var qkv_4d = try qkv_raw.reshape(&.{ 1, seq_len, total_heads, l.head_dim }, s);
                defer qkv_4d.deinit();

                var q_4d_raw = try qkv_4d.sliceAxis(2, 0, l.num_heads, s);
                defer q_4d_raw.deinit();
                var k_4d_raw = try qkv_4d.sliceAxis(2, l.num_heads, l.num_heads + l.num_kv_heads, s);
                defer k_4d_raw.deinit();
                var v_4d_raw = try qkv_4d.sliceAxis(2, l.num_heads + l.num_kv_heads, total_heads, s);
                defer v_4d_raw.deinit();

                var q_normed = try mlx.Array.fastRmsNorm(q_4d_raw, l.q_norm_w, l.rms_norm_eps, s);
                defer q_normed.deinit();
                var q_4d = try q_normed.transposeAxes(&.{ 0, 2, 1, 3 }, s);
                defer q_4d.deinit();
                q_rot = try mlx.Array.fastRoPE(q_4d, l.rot_dim, false, l.rope_theta, 1.0, pos, s);

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
                const default_cap: usize = if (l.is_sliding) l.sliding_window else 256;
                const initial_cap = @max(default_cap, ((total_len + 255) / 256) * 256);
                self.gpu_kv[l_idx] = try GpuKvCache.init(initial_cap, l.num_kv_heads, l.head_dim, s);
            }
            const cache = &self.gpu_kv[l_idx].?;
            const range = GpuKvCache.computeWriteRange(cache.capacity, l.is_sliding, l.sliding_window, pos, seq_len);
            try cache.ensureCapacity(range.write_stop, s);
            const next_k = try cache.k.sliceUpdate4dAxis2(k_rot, l.num_kv_heads, range.write_start, range.write_stop, l.head_dim, s);
            const next_v = try cache.v.sliceUpdate4dAxis2(v_4d, l.num_kv_heads, range.write_start, range.write_stop, l.head_dim, s);
            cache.k.deinit();
            cache.v.deinit();
            cache.k = next_k;
            cache.v = next_v;

            if (range.use_full_buffer) {
                eff_k = cache.k;
                eff_v = cache.v;
                if (l_idx == 22 or l_idx == 23) {
                    cache.clearStepSlices();
                }
            } else {
                eff_k = try cache.k.slice4dAxis2(l.num_kv_heads, range.slice_start, range.slice_stop, l.head_dim, s);
                eff_v = try cache.v.slice4dAxis2(l.num_kv_heads, range.slice_start, range.slice_stop, l.head_dim, s);
                if (l_idx == 22 or l_idx == 23) {
                    cache.clearStepSlices();
                    cache.step_eff_k = eff_k;
                    cache.step_eff_v = eff_v;
                } else {
                    eff_k_free = eff_k;
                    eff_v_free = eff_v;
                }
            }
        } else {
            if (seq_len == 1) {
                var q_4d_raw = try qkv_raw.reshape(&.{ 1, l.num_heads, 1, l.head_dim }, s);
                defer q_4d_raw.deinit();
                var q_normed = try mlx.Array.fastRmsNorm(q_4d_raw, l.q_norm_w, l.rms_norm_eps, s);
                defer q_normed.deinit();
                q_rot = try mlx.Array.fastRoPE(q_normed, l.rot_dim, false, l.rope_theta, 1.0, pos, s);
            } else {
                var q_4d_raw = try qkv_raw.reshape(&.{ 1, seq_len, l.num_heads, l.head_dim }, s);
                defer q_4d_raw.deinit();
                var q_normed = try mlx.Array.fastRmsNorm(q_4d_raw, l.q_norm_w, l.rms_norm_eps, s);
                defer q_normed.deinit();
                var q_4d = try q_normed.transposeAxes(&.{ 0, 2, 1, 3 }, s);
                defer q_4d.deinit();
                q_rot = try mlx.Array.fastRoPE(q_4d, l.rot_dim, false, l.rope_theta, 1.0, pos, s);
            }

            const cache = &self.gpu_kv[l.prev_kv_idx].?;
            eff_k = cache.step_eff_k orelse cache.k;
            eff_v = cache.step_eff_v orelse cache.v;
        }
        defer q_rot.deinit();

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

        // 仅在单步自回归解码 (seq_len == 1) 时调用形状特化 (shapeless = false) 的 42 层 compiled_tail 静态闭包；
        // 预填充阶段 (seq_len > 1) 直接走下方基于全局 shapeless = true 融合闭包 (compiled_gelu_mul / compiled_add_mul) 的路径，
        // 彻底避免任意新 Prompt 长度触发 42 个层尾闭包的图重追踪与 Metal 管线状态对象 (Pipeline State Object, PSO) JIT 重编译
        if (seq_len == 1) {
            if (l.compiled_tail) |cls| {
                return cls.apply3(x_in, o_out, ple_slice);
            }
        }

        var post_attn = try mlx.Array.fastRmsNorm(o_out, l.post_attn_w, l.rms_norm_eps, s);
        defer post_attn.deinit();
        var x1 = try mlx.Array.add(x_in, post_attn, s);
        defer x1.deinit();

        // (3) 门控多层感知机 (Gated MLP: Fused Gate+Up QMV + Compiled GeGLU)
        var pre_mlp = try mlx.Array.fastRmsNorm(x1, l.pre_mlp_w, l.rms_norm_eps, s);
        defer pre_mlp.deinit();

        var gate_up = try l.fused_gate_up.forwardArray(pre_mlp, s);
        defer gate_up.deinit();
        var gu_vec = try gate_up.split(2, 2, s);
        defer gu_vec.deinit();
        var gate_out = gu_vec.get(0).?;
        defer gate_out.deinit();
        var up_out = gu_vec.get(1).?;
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

        // 当 seq_len > 1 时，ple_slice 来自 ple_inputs.split(42, 2)，形状为非连续 4D 视图 [1, T, 1, 256]；
        // 将连续的 ple_gate_raw [1, T, 256] 以 O(1) 零拷贝视图重塑为 [1, T, 1, 256]，使 fusedGeluMul 直接融合读取步进视图并输出连续张量，
        // 随后以 O(1) 零拷贝视图恢复为 [1, T, 256]，彻底消除 42 层非连续 reshape 触发的 GPU 拷贝内核！
        var ple_modulated = if (seq_len == 1)
            try mlx.Array.fusedGeluMul(ple_gate_raw, ple_slice, self.compiled_gelu_mul, s)
        else blk: {
            const ple_dim = self.config.hidden_size_per_layer_input;
            var ple_gate_4d = try ple_gate_raw.reshape(&.{ 1, seq_len, 1, ple_dim }, s);
            defer ple_gate_4d.deinit();
            var ple_mod_4d = try mlx.Array.fusedGeluMul(ple_gate_4d, ple_slice, self.compiled_gelu_mul, s);
            defer ple_mod_4d.deinit();
            break :blk try ple_mod_4d.reshape(&.{ 1, seq_len, ple_dim }, s);
        };
        defer ple_modulated.deinit();
        var ple_projected = try l.ple_proj.forwardArray(ple_modulated, s);
        defer ple_projected.deinit();

        var ple_normed = try mlx.Array.fastRmsNorm(ple_projected, l.ple_norm_w, l.rms_norm_eps, s);
        defer ple_normed.deinit();

        // (5) 残差相加与层缩放标量融合 (Fused Residual Add + Layer Scalar Multiply)
        if (l.layer_scalar_w) |sw| {
            return mlx.Array.fusedAddMul(x2, ple_normed, sw, self.compiled_add_mul, s);
        }
        return mlx.Array.add(x2, ple_normed, s);
    }
};
