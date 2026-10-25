//! Gemma 4 E4B (Effective 4B) 语言模型架构
//!
//! 遵循 PyTorch / JAX 纯 Module 设计范式：
//! - 采用泛型变换器解码层组合：Q4 混合注意力 + 8-bit 量化门控 MLP + 逐层嵌入增强 (PLE) 模块；
//! - 完全解耦底层硬件细节，模型架构只持有参数张量与子模块；
//! - 端到端前向传播由 `autodiff.Graph` 与 `Tensor` 统一调度；
//! - 原生集成苹果统一内存 (UMA) GPU 硬件加速推理与自回归流式生成，兼具纯模块的架构清晰度与极致推理吞吐。

const std = @import("std");
const backend = @import("../backend.zig");
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");
const autodiff = @import("../autodiff.zig");
const gemma4 = @import("gemma4.zig");
const gemma4_q4 = @import("gemma4_q4.zig");

const offload = backend.offload;
const logits_processor = offload.logits_processor;
const sampling = backend.sampling;
const safetensors = backend.safetensors;
const activation = backend.activation;
const vec_math = backend.vec_math;
const rope_mod = backend.rope;

const Tensor = tensor.Tensor;
const Embedding = nn.transformer.Embedding;
const RMSNorm = nn.RMSNorm;
const deinitModel = nn.core.deinitModel;
const enterModuleScope = nn.core.enterModuleScope;

pub const Gemma4Config = gemma4.Gemma4Config;
pub const Gemma4Q4Attention = gemma4_q4.Gemma4Q4Attention;
pub const Gemma4Q4MLP = gemma4_q4.Gemma4Q4MLP;
pub const Gemma4PleLayer = gemma4.Gemma4PleLayer(nn.Q4Linear);
pub const Gemma4PleEmbedding = gemma4.Gemma4PleEmbedding(nn.Q4Linear);

pub const Q4Weights = offload.Q4Weights;
pub const QuantLayout = offload.QuantLayout;
pub const Stream = offload.Stream;
pub const GpuKvCache = offload.GpuKvCache;
pub const DeviceKvCache = offload.DeviceKvCache;
pub const KvCache = offload.KvCache;
pub const TokenExclusionMask = offload.TokenExclusionMask;
pub const Sampler = sampling.Sampler;
pub const SamplingOptions = sampling.SamplingOptions;
pub const SafetensorsArchive = safetensors.SafetensorsArchive;
pub const TensorMeta = safetensors.TensorMeta;
pub const GeluApproximation = activation.GeluApproximation;
pub const GpuRope = offload.Rope;

pub const default_generation = gemma4.default_generation;
pub const num_kv_layers: usize = 42 - 18;
pub const default_stop_tokens = [_]usize{ 1, 106, 258880, 258884 };

pub const kvSourceLayer = Gemma4Config.kvSourceLayer;
pub const isKvSharedSource = Gemma4Config.isKvSharedSource;

pub const DIRECT_CACHE_MAGIC = "ZNN4E4B1";
pub const DIRECT_CACHE_VERSION: u64 = 1;
pub const DIRECT_CACHE_FILENAME = "gemma4_e4b_direct.bin";

/// 计算预融合顺序二进制权重缓存 (`gemma4_e4b_direct.bin`) 的精确字节总长度
pub fn computeE4BExpectedDirectCacheSize(raw_cfg: anytype) usize {
    _ = raw_cfg;
    const cfg = Gemma4Config.e4b;
    var total: usize = 16; // 8B magic ("ZNN4E4B1") + 8B version (u64)
    total += QuantLayout.initDefault(cfg.hidden_size, cfg.vocab_size, 64, 4).totalBytes();
    total += cfg.hidden_size * @sizeOf(u16);
    total += QuantLayout.initDefault(cfg.num_hidden_layers * cfg.hidden_size_per_layer_input, cfg.vocab_size, 64, 4).totalBytes();
    total += QuantLayout.initDefault(cfg.hidden_size, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input, 64, 4).totalBytes();
    total += cfg.hidden_size_per_layer_input * @sizeOf(u16);

    for (0..cfg.num_hidden_layers) |idx| {
        const hd = cfg.layerSpec(idx).head_dim;
        const out_q = cfg.num_attention_heads * hd;
        const out_kv = cfg.num_key_value_heads * hd;
        const has_kv = (idx < cfg.num_hidden_layers - cfg.num_kv_shared_layers);

        if (has_kv) {
            total += hd * @sizeOf(u16);
            total += QuantLayout.initDefault(cfg.hidden_size, out_q + 2 * out_kv, 64, 4).totalBytes();
        } else {
            total += QuantLayout.initDefault(cfg.hidden_size, out_q, 64, 4).totalBytes();
        }
        total += QuantLayout.initDefault(out_q, cfg.hidden_size, 64, 4).totalBytes();
        total += QuantLayout.initDefault(cfg.hidden_size, cfg.intermediate_size * 2, 64, 8).totalBytes();
        total += QuantLayout.initDefault(cfg.intermediate_size, cfg.hidden_size, 64, 8).totalBytes();
        total += QuantLayout.initDefault(cfg.hidden_size, cfg.hidden_size_per_layer_input, 64, 4).totalBytes();
        total += QuantLayout.initDefault(cfg.hidden_size_per_layer_input, cfg.hidden_size, 64, 4).totalBytes();
        total += cfg.hidden_size * @sizeOf(u16);
        total += hd * @sizeOf(u16);
        total += cfg.hidden_size * @sizeOf(u16) * 4;
        total += @sizeOf(u16);
    }
    return total;
}
pub const computeExpectedDirectCacheSize = computeE4BExpectedDirectCacheSize;

pub const E4BDirectCacheReader = struct {
    data: []const u8,
    offset: usize = 16,

    pub fn init(data: []const u8, expected_size: usize) !E4BDirectCacheReader {
        if (data.len != expected_size or data.len < 16) return error.InvalidDirectCache;
        if (!std.mem.eql(u8, data[0..8], DIRECT_CACHE_MAGIC)) return error.InvalidDirectCache;
        const ver = std.mem.readInt(u64, data[8..16], .little);
        if (ver != DIRECT_CACHE_VERSION) return error.InvalidDirectCache;
        return .{ .data = data, .offset = 16 };
    }

    pub fn readBytes(self: *E4BDirectCacheReader, len: usize) ![]const u8 {
        if (self.offset + len > self.data.len) return error.InvalidDirectCache;
        const s = self.data[self.offset .. self.offset + len];
        self.offset += len;
        return s;
    }

    pub fn readQ4(
        self: *E4BDirectCacheReader,
        in_f: usize,
        out_f: usize,
        group_size: i32,
        bits: i32,
        stream: Stream,
    ) !Q4Weights {
        const layout = QuantLayout.initDefault(in_f, out_f, group_size, bits);
        const w_bytes = try self.readBytes(layout.w_q_bytes);
        const s_bytes = try self.readBytes(layout.sb_bytes);
        const b_bytes = try self.readBytes(layout.sb_bytes);
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

    pub fn readNorm(self: *E4BDirectCacheReader, dim: usize) !offload.Array {
        const bytes = try self.readBytes(dim * @sizeOf(u16));
        return offload.Array.fromRawData(@ptrCast(bytes.ptr), &.{dim}, .bfloat16);
    }
};
pub const DirectCacheReader = E4BDirectCacheReader;

pub const E4BLoadOptions = struct {
    warmup: bool = true,

    pub fn defaultOptions() E4BLoadOptions {
        return .{};
    }
};
pub const LoadOptions = E4BLoadOptions;

pub const E4BGenerateOptions = struct {
    sampling: SamplingOptions = default_generation,
    max_new_tokens: usize = 256,
    stop_tokens: []const usize = &default_stop_tokens,

    pub fn defaultOptions() E4BGenerateOptions {
        return .{};
    }
};
pub const GenerateOptions = E4BGenerateOptions;

/// 逐层嵌入输入 (PLE) GPU 加速权重结构
pub const PleWeights = struct {
    embed_tokens_per_layer: Q4Weights,
    model_proj: Q4Weights,
    proj_norm_w: offload.Array,
    num_layers: usize,
    ple_dim: usize,
    embed_scale: f32,
    proj_scale: f32,
    input_scale: f32,
    embed_scale_arr: offload.Array,
    proj_scale_arr: offload.Array,
    input_scale_arr: offload.Array,

    pub fn deinit(self: *PleWeights) void {
        self.embed_tokens_per_layer.deinit();
        self.model_proj.deinit();
        self.proj_norm_w.deinit();
        self.embed_scale_arr.deinit();
        self.proj_scale_arr.deinit();
        self.input_scale_arr.deinit();
    }
};

/// 步首词元嵌入与逐层输入预计算静态编译上下文
pub const PlePrepContext = struct {
    embed_tokens: Q4Weights,
    sqrt_hidden_arr: offload.Array,
    ple: PleWeights,
    hidden_size: usize,
    rms_norm_eps: f32,

    pub fn run(ctx: *const PlePrepContext, in_arrs: [1]offload.Array) ![2]offload.Array {
        const indices_arr = in_arrs[0];
        const T = indices_arr.dimSize(0);
        if (T == 0) return error.ShapeMismatch;

        var emb = try ctx.embed_tokens.gatherRows(indices_arr, ctx.sqrt_hidden_arr, null);
        defer emb.deinit();
        var x_cur = try emb.reshape(&.{ 1, T, ctx.hidden_size }, null);
        errdefer x_cur.deinit();

        var pdeq = try ctx.ple.embed_tokens_per_layer.gatherRows(indices_arr, ctx.ple.embed_scale_arr, null);
        defer pdeq.deinit();
        var ple_tokens_arr = try pdeq.reshape(&.{ 1, T, ctx.ple.num_layers, ctx.ple.ple_dim }, null);
        defer ple_tokens_arr.deinit();

        var ple_proj_raw = try ctx.ple.model_proj.forwardArray(x_cur, null);
        defer ple_proj_raw.deinit();
        var ple_proj_scaled = try offload.Array.multiply(ple_proj_raw, ctx.ple.proj_scale_arr, null);
        defer ple_proj_scaled.deinit();
        var ple_proj_4d = try ple_proj_scaled.reshape(&.{ 1, T, ctx.ple.num_layers, ctx.ple.ple_dim }, null);
        defer ple_proj_4d.deinit();
        var ple_proj_normed = try offload.fastRmsNorm(ple_proj_4d, ctx.ple.proj_norm_w, ctx.rms_norm_eps, null);
        defer ple_proj_normed.deinit();

        var ple_sum = try offload.Array.add(ple_proj_normed, ple_tokens_arr, null);
        defer ple_sum.deinit();
        var ple_inputs = try offload.Array.multiply(ple_sum, ctx.ple.input_scale_arr, null);
        errdefer ple_inputs.deinit();

        return .{ x_cur, ple_inputs };
    }
};

/// 单个解码层的 GPU 硬件加速权重结构
pub const GpuLayerE4B = struct {
    qkv_proj: Q4Weights,
    o_proj: Q4Weights,

    input_norm_w: offload.Array,
    q_norm_w: offload.Array,
    k_norm_w: ?offload.Array = null,
    post_attn_w: offload.Array,
    pre_mlp_w: offload.Array,
    post_mlp_w: offload.Array,

    fused_gate_up: Q4Weights,
    down_proj: Q4Weights,

    ple_gate: Q4Weights,
    ple_proj: Q4Weights,
    ple_norm_w: offload.Array,

    has_kv: bool,
    prev_kv_idx: usize,
    is_kv_shared_source: bool,
    layer_scalar: f32,
    layer_scalar_w: ?offload.Array = null,
    is_sliding: bool,
    sliding_window: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: GpuRope,
    rms_norm_eps: f32,
    hidden_activation: GeluApproximation,
    compiled_tail: ?offload.Closure = null,

    pub fn forwardTail(
        self: *const GpuLayerE4B,
        x_in: offload.Array,
        o_out: offload.Array,
        ple_slice: offload.Array,
        s: ?Stream,
    ) !offload.Array {
        var post_attn = try offload.fastRmsNorm(o_out, self.post_attn_w, self.rms_norm_eps, s);
        defer post_attn.deinit();
        var x1 = try offload.residualScale(x_in, post_attn, null, s);
        defer x1.deinit();

        var post_mlp = try offload.gatedMlp(
            x1,
            &self.fused_gate_up,
            null,
            &self.down_proj,
            self.hidden_activation,
            self.pre_mlp_w,
            self.post_mlp_w,
            self.rms_norm_eps,
            s,
        );
        defer post_mlp.deinit();
        var x2 = try offload.residualScale(x1, post_mlp, null, s);
        defer x2.deinit();

        var ple_gate_raw = try self.ple_gate.forwardArray(x2, s);
        defer ple_gate_raw.deinit();
        var act_ple = try ple_gate_raw.gelu(self.hidden_activation, s);
        defer act_ple.deinit();
        var ple_modulated = try offload.Array.multiply(act_ple, ple_slice, s);
        defer ple_modulated.deinit();

        var ple_projected = try self.ple_proj.forwardArray(ple_modulated, s);
        defer ple_projected.deinit();
        var ple_normed = try offload.fastRmsNorm(ple_projected, self.ple_norm_w, self.rms_norm_eps, s);
        defer ple_normed.deinit();

        return offload.residualScale(x2, ple_normed, self.layer_scalar_w, s);
    }

    pub fn tailRun(self: *const GpuLayerE4B, in_arrs: [3]offload.Array) !offload.Array {
        return self.forwardTail(in_arrs[0], in_arrs[1], in_arrs[2], null);
    }

    pub fn initCompiledTail(self: *GpuLayerE4B) void {
        if (!offload.isAvailable()) return;
        self.compiled_tail = offload.compileFn(3, 1, GpuLayerE4B, tailRun, false, self) catch null;
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
        self.rope.deinit();
    }
};

const WeightSource = struct {
    direct: ?*E4BDirectCacheReader,
    archive: ?SafetensorsArchive,
    cache_writer: ?*std.Io.Writer,
    cache_write_ok: *bool,
    scratch: []u8,
    stream: Stream,
    name_buf: [128]u8 = undefined,

    const LayerScalar = struct {
        value: f32,
        arr: ?offload.Array,
    };

    fn writeCache(self: *WeightSource, bytes: []const u8) void {
        if (self.cache_writer) |w| {
            w.writeAll(bytes) catch {
                self.cache_write_ok.* = false;
            };
        }
    }

    fn tensorBytes(self: *WeightSource, comptime fmt: []const u8, args: anytype) ![]const u8 {
        const name = try std.fmt.bufPrint(&self.name_buf, fmt, args);
        return self.archive.?.getTensorBytes(name) orelse error.TensorNotFound;
    }

    fn q4(self: *WeightSource, prefix: []const u8, in_f: usize, out_f: usize, group_size: i32, bits: i32) !Q4Weights {
        if (self.direct) |r| return r.readQ4(in_f, out_f, group_size, bits, self.stream);
        const w_bytes = try self.tensorBytes("{s}.weight", .{prefix});
        const s_bytes = try self.tensorBytes("{s}.scales", .{prefix});
        const b_bytes = try self.tensorBytes("{s}.biases", .{prefix});
        self.writeCache(w_bytes);
        self.writeCache(s_bytes);
        self.writeCache(b_bytes);
        return Q4Weights.initFromDirectBuffersCustom(w_bytes, s_bytes, b_bytes, in_f, out_f, group_size, bits, .bfloat16, self.stream);
    }

    fn norm(self: *WeightSource, name: []const u8, dim: usize) !offload.Array {
        const bytes = if (self.direct) |r| try r.readBytes(dim * @sizeOf(u16)) else blk: {
            const raw = try self.tensorBytes("{s}", .{name});
            if (raw.len < dim * @sizeOf(u16)) return error.InvalidTensorShape;
            self.writeCache(raw[0 .. dim * @sizeOf(u16)]);
            break :blk raw;
        };
        return offload.Array.fromRawData(@ptrCast(bytes.ptr), &.{dim}, .bfloat16);
    }

    fn layerQ4(self: *WeightSource, l_idx: usize, sub: []const u8, in_f: usize, out_f: usize, group_size: i32, bits: i32) !Q4Weights {
        var prefix_buf: [128]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&prefix_buf, "language_model.model.layers.{}.{s}", .{ l_idx, sub });
        return self.q4(prefix, in_f, out_f, group_size, bits);
    }

    fn layerNorm(self: *WeightSource, l_idx: usize, sub: []const u8, dim: usize) !offload.Array {
        var name_buf: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "language_model.model.layers.{}.{s}", .{ l_idx, sub });
        return self.norm(name, dim);
    }

    fn layerFusedQ4(
        self: *WeightSource,
        l_idx: usize,
        comptime subs: []const []const u8,
        in_f: usize,
        total_out_f: usize,
        group_size: i32,
        bits: i32,
    ) !Q4Weights {
        if (self.direct) |r| return r.readQ4(in_f, total_out_f, group_size, bits, self.stream);
        var w_slices: [subs.len][]const u8 = undefined;
        var s_slices: [subs.len][]const u8 = undefined;
        var b_slices: [subs.len][]const u8 = undefined;
        inline for (subs, 0..) |sub, i| {
            w_slices[i] = try self.tensorBytes("language_model.model.layers.{}.{s}.weight", .{ l_idx, sub });
            s_slices[i] = try self.tensorBytes("language_model.model.layers.{}.{s}.scales", .{ l_idx, sub });
            b_slices[i] = try self.tensorBytes("language_model.model.layers.{}.{s}.biases", .{ l_idx, sub });
        }
        for (w_slices) |ws| self.writeCache(ws);
        for (s_slices) |ss| self.writeCache(ss);
        for (b_slices) |bs| self.writeCache(bs);
        return Q4Weights.initFusedFromDirectBuffersCustom(
            &w_slices,
            &s_slices,
            &b_slices,
            self.scratch,
            in_f,
            total_out_f,
            group_size,
            bits,
            .bfloat16,
            self.stream,
        );
    }

    fn layerScalar(self: *WeightSource, l_idx: usize) !LayerScalar {
        var bits: [2]u8 = .{ 0x80, 0x3F }; // 1.0 in BF16 little-endian
        if (self.direct) |r| {
            bits = (try r.readBytes(2))[0..2].*;
        } else {
            if (self.tensorBytes("language_model.model.layers.{}.layer_scalar", .{l_idx})) |sb| {
                if (sb.len >= 2) bits = sb[0..2].*;
            } else |err| switch (err) {
                error.TensorNotFound => {},
                else => return err,
            }
            self.writeCache(&bits);
        }
        const value = vec_math.bf16ToF32(std.mem.readInt(u16, &bits, .little));
        const arr: ?offload.Array = if (value != 1.0) try offload.Array.fromRawData(@ptrCast(&bits), &.{1}, .bfloat16) else null;
        return .{ .value = value, .arr = arr };
    }
};

/// Gemma 4 E4B 变换器解码层 (泛型模块装配：Q4 注意力 + Q4 门控 MLP + Q4 PLE 模块)
pub const Gemma4E4BDecoderLayer = gemma4.Gemma4DecoderLayerT(
    Gemma4Q4Attention,
    Gemma4Q4MLP,
    Gemma4PleLayer,
);

/// Gemma 4 E4B 语言模型核心架构 (Gemma4E4BForCausalLM: 纯 PyTorch / JAX 风格模块化定义，兼具原生 UMA 硬件加速)
pub fn Gemma4E4BForCausalLM(comptime cfg: Gemma4Config) type {
    return struct {
        pub const config: Gemma4Config = cfg;
        const Self = @This();

        allocator: std.mem.Allocator,
        embed_tokens: Embedding,
        embed_tokens_per_layer: Embedding,
        ple_embed: Gemma4PleEmbedding,
        layers: []Gemma4E4BDecoderLayer,
        norm: RMSNorm,

        // GPU 硬件加速推理字段 (加载自 Safetensors / 预融合二进制缓存)
        gpu_layers: ?[]GpuLayerE4B = null,
        ple: ?PleWeights = null,
        embed_tokens_w: ?Q4Weights = null,
        norm_w: ?offload.Array = null,
        lm_head: ?Q4Weights = null,
        stream: Stream = undefined,
        exclusion_mask: ?TokenExclusionMask = null,
        gpu_kv: [num_kv_layers]?GpuKvCache = @splat(null),
        curr_len: usize = 0,
        sqrt_hidden_arr: ?offload.Array = null,
        compiled_gelu_mul: ?offload.Closure = null,
        compiled_add_mul: ?offload.Closure = null,
        ple_prep_ctx: ?*PlePrepContext = null,
        compiled_ple_prep: ?offload.Closure = null,

        name: ?[]const u8 = null,
        module_type: []const u8 = "Gemma4E4BForCausalLM",

        pub const KvCache = GpuKvCache;
        pub const DeviceKvCache = GpuKvCache;

        pub const DIRECT_CACHE_MAGIC = "ZNN4E4B1";
        pub const DIRECT_CACHE_VERSION: u64 = 1;
        pub const DIRECT_CACHE_FILENAME = "gemma4_e4b_direct.bin";
        pub const DirectCacheReader = E4BDirectCacheReader;
        pub const LoadOptions = E4BLoadOptions;
        pub const GenerateOptions = E4BGenerateOptions;

        pub fn computeExpectedDirectCacheSize(raw_cfg: anytype) usize {
            return computeE4BExpectedDirectCacheSize(raw_cfg);
        }

        pub fn init(allocator: std.mem.Allocator) !Self {
            const embed_tokens = try Embedding.init(allocator, config.vocab_size, config.hidden_size);
            errdefer deinitModel(&embed_tokens, allocator);

            const embed_tokens_per_layer = try Embedding.init(
                allocator,
                config.vocab_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
            );
            errdefer deinitModel(&embed_tokens_per_layer, allocator);

            var ple_embed = try Gemma4PleEmbedding.init(
                allocator,
                config.hidden_size,
                config.num_hidden_layers,
                config.hidden_size_per_layer_input,
                config.rms_norm_eps,
            );
            errdefer ple_embed.deinit(allocator);

            const layers = try allocator.alloc(Gemma4E4BDecoderLayer, config.num_hidden_layers);
            errdefer allocator.free(layers);
            var initialized_layers: usize = 0;
            errdefer for (layers[0..initialized_layers]) |*layer| layer.deinit(allocator);

            for (0..config.num_hidden_layers) |i| {
                const spec = config.layerSpec(i);
                layers[i] = try Gemma4E4BDecoderLayer.initWithPle(
                    allocator,
                    spec.attn_type,
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                    spec.num_kv_heads,
                    spec.head_dim,
                    config.sliding_window,
                    config.rms_norm_eps,
                    config.hidden_activation,
                    config.hidden_size_per_layer_input,
                );
                initialized_layers += 1;
            }

            const norm = try RMSNorm.init(allocator, config.hidden_size, config.rms_norm_eps);

            return Self{
                .allocator = allocator,
                .embed_tokens = embed_tokens,
                .embed_tokens_per_layer = embed_tokens_per_layer,
                .ple_embed = ple_embed,
                .layers = layers,
                .norm = norm,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.gpu_layers) |gl_slice| {
                for (gl_slice) |*l| l.deinit();
                self.allocator.free(gl_slice);
                self.gpu_layers = null;
            }
            for (&self.gpu_kv) |*kv_opt| {
                if (kv_opt.*) |*c| c.deinit();
                kv_opt.* = null;
            }
            if (self.embed_tokens_w) |*ew| ew.deinit();
            if (self.norm_w) |*nw| nw.deinit();
            self.lm_head = null;
            if (self.ple) |*p| p.deinit();
            if (self.exclusion_mask) |*em| em.deinit();
            if (self.sqrt_hidden_arr) |*sh| sh.deinit();
            if (self.compiled_gelu_mul) |*cls| cls.deinit();
            if (self.compiled_add_mul) |*cls| cls.deinit();
            if (self.compiled_ple_prep) |*cls| cls.deinit();
            if (self.ple_prep_ctx) |ctx_ptr| self.allocator.destroy(ctx_ptr);

            deinitModel(&self.embed_tokens, self.allocator);
            deinitModel(&self.embed_tokens_per_layer, self.allocator);
            self.ple_embed.deinit(self.allocator);
            for (self.layers) |*layer| {
                layer.deinit(self.allocator);
            }
            self.allocator.free(self.layers);
            deinitModel(&self.norm, self.allocator);
        }

        pub fn enableHardwareOffload(self: *Self) !void {
            try self.ple_embed.enableHardwareOffload(self.allocator);
            for (self.layers) |*layer| {
                try layer.enableHardwareOffload(self.allocator);
            }
        }

        pub fn resetCache(self: *Self) void {
            for (&self.gpu_kv) |*kv_opt| {
                if (kv_opt.*) |*c| c.reset();
            }
            self.curr_len = 0;
        }

        pub fn warmup(self: *Self) !void {
            if (self.gpu_layers == null) return;
            const dummy_tokens = [_]usize{ 2, 106, 108 };
            var tok_arr = try self.prefillOnDevice(&dummy_tokens);
            defer tok_arr.deinit();
            var next_tok = try self.decodeOnDevice(tok_arr);
            defer next_tok.deinit();
            _ = try next_tok.itemUint32();
            self.resetCache();
        }

        pub fn loadFromDirectory(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !Self {
            return loadFromDirectoryWithOptions(allocator, io, dir_path, .{});
        }

        pub fn loadFromDirectoryWithOptions(
            allocator: std.mem.Allocator,
            io: std.Io,
            dir_path: []const u8,
            options: E4BLoadOptions,
        ) !Self {
            var model = try Self.init(allocator);
            errdefer model.deinit();

            try model.loadWeights(allocator, io, dir_path);
            if (options.warmup) try model.warmup();
            return model;
        }

        fn loadWeights(self: *Self, allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !void {
            const expected_cache_bytes = computeE4BExpectedDirectCacheSize(config);
            const cache_path = try std.fs.path.join(allocator, &.{ dir_path, Self.DIRECT_CACHE_FILENAME });
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
                            if (std.mem.eql(u8, mapped[0..8], Self.DIRECT_CACHE_MAGIC) and
                                std.mem.readInt(u64, mapped[8..16], .little) == Self.DIRECT_CACHE_VERSION)
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

            const tmp_cache_path = try std.fs.path.join(allocator, &.{ dir_path, Self.DIRECT_CACHE_FILENAME ++ ".tmp" });
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
                w.writeAll(Self.DIRECT_CACHE_MAGIC) catch {
                    cache_write_ok = false;
                };
                const ver_bytes = std.mem.toBytes(@as(u64, Self.DIRECT_CACHE_VERSION));
                w.writeAll(&ver_bytes) catch {
                    cache_write_ok = false;
                };
                if (cache_write_ok) cache_writer_opt = w;
            }
            defer if (cache_out_file_opt) |*cf_out| cf_out.close(io);

            var stream = try Stream.defaultGpu();
            errdefer stream.deinit();

            _ = offload.configureOptimalMemory();
            offload.enableCompile();
            offload.setCompileMode(.enabled);

            var direct_reader: E4BDirectCacheReader = if (direct_mmap_opt) |mapped|
                try E4BDirectCacheReader.init(mapped, expected_cache_bytes)
            else
                .{ .data = &.{} };

            const max_fused_bytes = (config.intermediate_size * 2) * (config.hidden_size / 4) * @sizeOf(u32);
            const fuse_scratch: []u8 = if (direct_mmap_opt == null) try allocator.alloc(u8, max_fused_bytes) else &.{};
            defer if (direct_mmap_opt == null) allocator.free(fuse_scratch);

            var src = WeightSource{
                .direct = if (direct_mmap_opt != null) &direct_reader else null,
                .archive = archive_opt,
                .cache_writer = cache_writer_opt,
                .cache_write_ok = &cache_write_ok,
                .scratch = fuse_scratch,
                .stream = stream,
            };

            var embed_tokens = try src.q4("language_model.model.embed_tokens", config.hidden_size, config.vocab_size, 64, 4);
            errdefer embed_tokens.deinit();
            var norm_w = try src.norm("language_model.model.norm.weight", config.hidden_size);
            errdefer norm_w.deinit();

            const ple_total = config.num_hidden_layers * config.hidden_size_per_layer_input;
            var ple_tokens = try src.q4("language_model.model.embed_tokens_per_layer", ple_total, config.vocab_size, 64, 4);
            errdefer ple_tokens.deinit();
            var ple_model_proj = try src.q4("language_model.model.per_layer_model_projection", config.hidden_size, ple_total, 64, 4);
            errdefer ple_model_proj.deinit();
            var ple_pnorm_w = try src.norm("language_model.model.per_layer_projection_norm.weight", config.hidden_size_per_layer_input);
            errdefer ple_pnorm_w.deinit();

            const embed_scale_val: f32 = @sqrt(@as(f32, @floatFromInt(config.hidden_size_per_layer_input)));
            const proj_scale_val: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
            const input_scale_val: f32 = 1.0 / @sqrt(2.0);

            var embed_scale_arr = try offload.Array.fromScalarBf16(embed_scale_val);
            errdefer embed_scale_arr.deinit();
            var proj_scale_arr = try offload.Array.fromScalarBf16(proj_scale_val);
            errdefer proj_scale_arr.deinit();
            var input_scale_arr = try offload.Array.fromScalarBf16(input_scale_val);
            errdefer input_scale_arr.deinit();

            const ple = PleWeights{
                .embed_tokens_per_layer = ple_tokens,
                .model_proj = ple_model_proj,
                .proj_norm_w = ple_pnorm_w,
                .num_layers = config.num_hidden_layers,
                .ple_dim = config.hidden_size_per_layer_input,
                .embed_scale = embed_scale_val,
                .proj_scale = proj_scale_val,
                .input_scale = input_scale_val,
                .embed_scale_arr = embed_scale_arr,
                .proj_scale_arr = proj_scale_arr,
                .input_scale_arr = input_scale_arr,
            };

            const sqrt_hidden_val: f32 = @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
            var sqrt_hidden_arr = try offload.Array.fromScalarBf16(sqrt_hidden_val);
            errdefer sqrt_hidden_arr.deinit();

            const gpu_layers = try allocator.alloc(GpuLayerE4B, config.num_hidden_layers);
            errdefer allocator.free(gpu_layers);
            var built_layers: usize = 0;
            errdefer for (gpu_layers[0..built_layers]) |*l| l.deinit();

            for (0..config.num_hidden_layers) |idx| {
                gpu_layers[idx] = try self.loadLayer(&src, idx);
                built_layers += 1;
                gpu_layers[idx].initCompiledTail();
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

            var compiled_gelu_mul: ?offload.Closure = offload.compileGeluMul(config.hidden_activation) catch null;
            errdefer if (compiled_gelu_mul) |*cls| cls.deinit();
            var compiled_add_mul: ?offload.Closure = offload.compileAddMul() catch null;
            errdefer if (compiled_add_mul) |*cls| cls.deinit();

            var ple_prep_ctx: ?*PlePrepContext = null;
            var compiled_ple_prep: ?offload.Closure = null;
            if (offload.isAvailable()) {
                if (allocator.create(PlePrepContext)) |ctx_ptr| {
                    ctx_ptr.* = .{
                        .embed_tokens = embed_tokens,
                        .sqrt_hidden_arr = sqrt_hidden_arr,
                        .ple = ple,
                        .hidden_size = config.hidden_size,
                        .rms_norm_eps = config.rms_norm_eps,
                    };
                    ple_prep_ctx = ctx_ptr;
                    compiled_ple_prep = offload.compileFn(1, 2, PlePrepContext, PlePrepContext.run, false, ctx_ptr) catch null;
                } else |_| {}
            }
            errdefer {
                if (compiled_ple_prep) |*cls| cls.deinit();
                if (ple_prep_ctx) |ctx_ptr| allocator.destroy(ctx_ptr);
            }

            var exclusion_mask = try TokenExclusionMask.init(allocator, config.vocab_size, config.sampling_excluded_tokens);
            errdefer exclusion_mask.deinit();

            var gpu_kv: [num_kv_layers]?GpuKvCache = @splat(null);
            errdefer {
                for (&gpu_kv) |*kv_opt| {
                    if (kv_opt.*) |*c| c.deinit();
                }
            }
            for (&gpu_kv, gpu_layers[0..num_kv_layers]) |*kv_slot, l_ref| {
                const init_cap: usize = if (l_ref.is_sliding) l_ref.sliding_window else 256;
                kv_slot.* = try GpuKvCache.init(init_cap, l_ref.num_kv_heads, l_ref.head_dim, .bfloat16, stream);
            }

            self.stream = stream;
            self.embed_tokens_w = embed_tokens;
            self.lm_head = embed_tokens;
            self.norm_w = norm_w;
            self.ple = ple;
            self.sqrt_hidden_arr = sqrt_hidden_arr;
            self.gpu_layers = gpu_layers;
            self.gpu_kv = gpu_kv;
            self.compiled_gelu_mul = compiled_gelu_mul;
            self.compiled_add_mul = compiled_add_mul;
            self.ple_prep_ctx = ple_prep_ctx;
            self.compiled_ple_prep = compiled_ple_prep;
            self.exclusion_mask = exclusion_mask;
            self.curr_len = 0;
        }

        fn loadLayer(self: *Self, src: *WeightSource, idx: usize) !GpuLayerE4B {
            _ = self;
            const spec = config.layerSpec(idx);
            const hd = spec.head_dim;
            const is_full = (spec.attn_type == .full_attention);
            const H = config.hidden_size;
            const ple_dim = config.hidden_size_per_layer_input;
            const out_q = config.num_attention_heads * hd;
            const out_kv = config.num_key_value_heads * hd;
            const has_kv = idx < num_kv_layers;

            var kn_w: ?offload.Array = if (has_kv) try src.layerNorm(idx, "self_attn.k_norm.weight", hd) else null;
            errdefer if (kn_w) |*kn| kn.deinit();

            var qkv_proj = if (has_kv)
                try src.layerFusedQ4(idx, &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" }, H, out_q + 2 * out_kv, 64, 4)
            else
                try src.layerQ4(idx, "self_attn.q_proj", H, out_q, 64, 4);
            errdefer qkv_proj.deinit();
            var o_proj = try src.layerQ4(idx, "self_attn.o_proj", out_q, H, 64, 4);
            errdefer o_proj.deinit();

            var fused_gate_up = try src.layerFusedQ4(idx, &.{ "mlp.gate_proj", "mlp.up_proj" }, H, config.intermediate_size * 2, 64, 8);
            errdefer fused_gate_up.deinit();
            var down_proj = try src.layerQ4(idx, "mlp.down_proj", config.intermediate_size, H, 64, 8);
            errdefer down_proj.deinit();

            var ple_gate = try src.layerQ4(idx, "per_layer_input_gate", H, ple_dim, 64, 4);
            errdefer ple_gate.deinit();
            var ple_proj = try src.layerQ4(idx, "per_layer_projection", ple_dim, H, 64, 4);
            errdefer ple_proj.deinit();

            var in_w = try src.layerNorm(idx, "input_layernorm.weight", H);
            errdefer in_w.deinit();
            var qn_w = try src.layerNorm(idx, "self_attn.q_norm.weight", hd);
            errdefer qn_w.deinit();
            var pa_w = try src.layerNorm(idx, "post_attention_layernorm.weight", H);
            errdefer pa_w.deinit();
            var pm_w = try src.layerNorm(idx, "pre_feedforward_layernorm.weight", H);
            errdefer pm_w.deinit();
            var pom_w = try src.layerNorm(idx, "post_feedforward_layernorm.weight", H);
            errdefer pom_w.deinit();
            var pln_w = try src.layerNorm(idx, "post_per_layer_input_norm.weight", H);
            errdefer pln_w.deinit();

            var scalar = try src.layerScalar(idx);
            errdefer if (scalar.arr) |*arr| arr.deinit();

            const rope = try GpuRope.init(hd, spec.attn_type.ropeOptions());

            return .{
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
                .prev_kv_idx = config.kvSourceLayer(idx),
                .is_kv_shared_source = has_kv and config.isKvSharedSource(idx),
                .layer_scalar = scalar.value,
                .layer_scalar_w = scalar.arr,
                .is_sliding = !is_full,
                .sliding_window = config.sliding_window,
                .num_heads = config.num_attention_heads,
                .num_kv_heads = config.num_key_value_heads,
                .head_dim = hd,
                .rope = rope,
                .rms_norm_eps = config.rms_norm_eps,
                .hidden_activation = config.hidden_activation,
            };
        }

        fn createTokenIndicesArray(self: *const Self, tokens: []const usize) !offload.Array {
            for (tokens) |tok| {
                if (tok >= config.vocab_size) return error.IndexOutOfBounds;
            }
            const T = tokens.len;
            const single_tok = [_]i32{if (T > 0) @intCast(tokens[0]) else 0};
            if (T == 1) {
                return offload.Array.fromSliceI32(&single_tok, &.{1});
            }
            const tokens_i32 = try self.allocator.alloc(i32, T);
            defer self.allocator.free(tokens_i32);
            for (tokens, 0..) |tok, i| {
                tokens_i32[i] = @intCast(tok);
            }
            return offload.Array.fromSliceI32(tokens_i32, &.{T});
        }

        pub fn getEmbeddingsFromArray(self: *const Self, indices_arr: offload.Array, T: usize, s: Stream) !offload.Array {
            var emb = try self.embed_tokens_w.?.gatherRows(indices_arr, self.sqrt_hidden_arr.?, s);
            defer emb.deinit();
            return emb.reshape(&.{ 1, T, config.hidden_size }, s);
        }

        pub fn getPleTokensFromArray(self: *const Self, indices_arr: offload.Array, T: usize, s: Stream) !offload.Array {
            var pdeq = try self.ple.?.embed_tokens_per_layer.gatherRows(indices_arr, self.ple.?.embed_scale_arr, s);
            defer pdeq.deinit();
            return pdeq.reshape(&.{ 1, T, config.num_hidden_layers, config.hidden_size_per_layer_input }, s);
        }

        fn forwardTokenArrayToLogits(self: *Self, indices_arr: offload.Array, pos: usize, T: usize, s: Stream) !offload.Array {
            var x_cur: offload.Array = undefined;
            var ple_inputs: offload.Array = undefined;
            if (T == 1 and self.compiled_ple_prep != null) {
                const prep_out = try self.compiled_ple_prep.?.apply1To2(indices_arr);
                x_cur = prep_out[0];
                ple_inputs = prep_out[1];
            } else {
                x_cur = try self.getEmbeddingsFromArray(indices_arr, T, s);
                errdefer x_cur.deinit();

                var ple_tokens_arr = try self.getPleTokensFromArray(indices_arr, T, s);
                defer ple_tokens_arr.deinit();

                var ple_proj_raw = try self.ple.?.model_proj.forwardArray(x_cur, s);
                defer ple_proj_raw.deinit();
                var ple_proj_scaled = try offload.Array.multiply(ple_proj_raw, self.ple.?.proj_scale_arr, s);
                defer ple_proj_scaled.deinit();
                var ple_proj_4d = try ple_proj_scaled.reshape(&.{ 1, T, config.num_hidden_layers, config.hidden_size_per_layer_input }, s);
                defer ple_proj_4d.deinit();
                var ple_proj_normed = try offload.fastRmsNorm(ple_proj_4d, self.ple.?.proj_norm_w, config.rms_norm_eps, s);
                defer ple_proj_normed.deinit();

                ple_inputs = try offload.fusedAddMul(ple_proj_normed, ple_tokens_arr, self.ple.?.input_scale_arr, self.compiled_add_mul, s);
            }
            defer x_cur.deinit();
            defer ple_inputs.deinit();
            const num_layers: i32 = @intCast(config.num_hidden_layers);

            if (T == 1) {
                var ple_3d = try ple_inputs.reshape(&.{ 1, config.num_hidden_layers, config.hidden_size_per_layer_input }, s);
                defer ple_3d.deinit();
                var ple_vec = try ple_3d.split(num_layers, 1, s);
                defer ple_vec.deinit();

                for (self.gpu_layers.?, 0..) |*l, idx| {
                    var ple_slice = ple_vec.get(idx).?;
                    defer ple_slice.deinit();

                    const x_next = try self.forwardLayer(x_cur, l, idx, ple_slice, pos, 1, s);
                    x_cur.deinit();
                    x_cur = x_next;
                }
            } else {
                var ple_vec = try ple_inputs.split(num_layers, 2, s);
                defer ple_vec.deinit();

                for (self.gpu_layers.?, 0..) |*l, idx| {
                    var ple_slice = ple_vec.get(idx).?;
                    defer ple_slice.deinit();

                    const x_next = try self.forwardLayer(x_cur, l, idx, ple_slice, pos, T, s);
                    x_cur.deinit();
                    x_cur = x_next;
                }
            }

            var h_norm = try offload.fastRmsNorm(x_cur, self.norm_w.?, config.rms_norm_eps, s);
            defer h_norm.deinit();

            if (T == 1) {
                var h_2d = try h_norm.reshape(&.{ 1, config.hidden_size }, s);
                defer h_2d.deinit();
                return self.lm_head.?.forwardArray(h_2d, s);
            } else {
                var last_token = try h_norm.sliceAxis(1, T - 1, T, s);
                defer last_token.deinit();
                var last_token_2d = try last_token.reshape(&.{ 1, config.hidden_size }, s);
                defer last_token_2d.deinit();
                return self.lm_head.?.forwardArray(last_token_2d, s);
            }
        }

        fn forwardLayer(
            self: *Self,
            x_in: offload.Array,
            l: *GpuLayerE4B,
            l_idx: usize,
            ple_slice: offload.Array,
            pos: usize,
            seq_len: usize,
            s: Stream,
        ) !offload.Array {
            var in_norm = try offload.fastRmsNorm(x_in, l.input_norm_w, l.rms_norm_eps, s);
            defer in_norm.deinit();

            var qkv_raw = try l.qkv_proj.forwardArray(in_norm, s);
            defer qkv_raw.deinit();

            var q_rot_opt: ?offload.Array = null;
            defer if (q_rot_opt) |*q| q.deinit();
            var eff_k: offload.Array = undefined;
            var eff_v: offload.Array = undefined;
            var eff_k_free: ?offload.Array = null;
            var eff_v_free: ?offload.Array = null;
            defer if (eff_k_free) |*ek| ek.deinit();
            defer if (eff_v_free) |*ev| ev.deinit();

            const total_len = pos + seq_len;

            if (l.has_kv) {
                const total_heads = l.num_heads + 2 * l.num_kv_heads;
                const axis: usize = if (seq_len == 1) 1 else 2;
                const qkv_shape = if (seq_len == 1)
                    [4]usize{ 1, total_heads, 1, l.head_dim }
                else
                    [4]usize{ 1, seq_len, total_heads, l.head_dim };
                var qkv_4d = try qkv_raw.reshape(&qkv_shape, s);
                defer qkv_4d.deinit();

                var q_raw_slice = try qkv_4d.sliceAxis(axis, 0, l.num_heads, s);
                defer q_raw_slice.deinit();
                var k_raw_slice = try qkv_4d.sliceAxis(axis, l.num_heads, l.num_heads + l.num_kv_heads, s);
                defer k_raw_slice.deinit();
                var v_raw_slice = try qkv_4d.sliceAxis(axis, l.num_heads + l.num_kv_heads, total_heads, s);
                defer v_raw_slice.deinit();

                q_rot_opt = try offload.projectHeads(q_raw_slice, l.q_norm_w, seq_len, l.num_heads, l.head_dim, l.rms_norm_eps, &l.rope, pos, s);
                var k_rot = try offload.projectHeads(k_raw_slice, l.k_norm_w.?, seq_len, l.num_kv_heads, l.head_dim, l.rms_norm_eps, &l.rope, pos, s);
                defer k_rot.deinit();
                var v_4d = try offload.projectHeads(v_raw_slice, null, seq_len, l.num_kv_heads, l.head_dim, l.rms_norm_eps, null, pos, s);
                defer v_4d.deinit();

                if (self.gpu_kv[l_idx] == null) {
                    const default_cap: usize = if (l.is_sliding) l.sliding_window else 256;
                    const initial_cap = @max(default_cap, ((total_len + 255) / 256) * 256);
                    self.gpu_kv[l_idx] = try GpuKvCache.init(initial_cap, l.num_kv_heads, l.head_dim, .bfloat16, s);
                }
                const cache = &self.gpu_kv[l_idx].?;
                const view = try cache.writeStep(k_rot, v_4d, l.is_sliding, l.sliding_window, pos, seq_len, s);
                eff_k = view.k;
                eff_v = view.v;
                if (!view.is_slice) {
                    if (l.is_kv_shared_source) cache.clearStepSlices();
                } else if (l.is_kv_shared_source) {
                    cache.clearStepSlices();
                    cache.step_eff_k = view.k;
                    cache.step_eff_v = view.v;
                } else {
                    eff_k_free = view.k;
                    eff_v_free = view.v;
                }
            } else {
                q_rot_opt = try offload.projectHeads(qkv_raw, l.q_norm_w, seq_len, l.num_heads, l.head_dim, l.rms_norm_eps, &l.rope, pos, s);
                const cache = &self.gpu_kv[l.prev_kv_idx].?;
                eff_k = cache.step_eff_k orelse cache.k;
                eff_v = cache.step_eff_v orelse cache.v;
            }
            const q_rot = q_rot_opt.?;

            var attn_out_4d = try offload.cachedAttention(self.allocator, q_rot, eff_k, eff_v, 1.0, l.is_sliding, l.sliding_window, pos, seq_len, s);
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

            if (seq_len == 1) {
                if (l.compiled_tail) |cls| {
                    return cls.apply3(x_in, o_out, ple_slice);
                }
                return l.forwardTail(x_in, o_out, ple_slice, s);
            }

            var post_attn = try offload.fastRmsNorm(o_out, l.post_attn_w, l.rms_norm_eps, s);
            defer post_attn.deinit();
            var x1 = try offload.Array.add(x_in, post_attn, s);
            defer x1.deinit();

            var pre_mlp = try offload.fastRmsNorm(x1, l.pre_mlp_w, l.rms_norm_eps, s);
            defer pre_mlp.deinit();

            var gate_up = try l.fused_gate_up.forwardArray(pre_mlp, s);
            defer gate_up.deinit();
            var gu_vec = try gate_up.split(2, 2, s);
            defer gu_vec.deinit();
            var gate_out = gu_vec.get(0).?;
            defer gate_out.deinit();
            var up_out = gu_vec.get(1).?;
            defer up_out.deinit();

            var hidden = try offload.fusedGeluMul(gate_out, up_out, config.hidden_activation, self.compiled_gelu_mul, s);
            defer hidden.deinit();

            var down_out = try l.down_proj.forwardArray(hidden, s);
            defer down_out.deinit();

            var post_mlp = try offload.fastRmsNorm(down_out, l.post_mlp_w, l.rms_norm_eps, s);
            defer post_mlp.deinit();
            var x2 = try offload.Array.add(x1, post_mlp, s);
            defer x2.deinit();

            var ple_gate_raw = try l.ple_gate.forwardArray(x2, s);
            defer ple_gate_raw.deinit();

            var ple_modulated = if (seq_len == 1)
                try offload.fusedGeluMul(ple_gate_raw, ple_slice, config.hidden_activation, self.compiled_gelu_mul, s)
            else blk: {
                const ple_dim = config.hidden_size_per_layer_input;
                var ple_gate_4d = try ple_gate_raw.reshape(&.{ 1, seq_len, 1, ple_dim }, s);
                defer ple_gate_4d.deinit();
                var ple_mod_4d = try offload.fusedGeluMul(ple_gate_4d, ple_slice, config.hidden_activation, self.compiled_gelu_mul, s);
                defer ple_mod_4d.deinit();
                break :blk try ple_mod_4d.reshape(&.{ 1, seq_len, ple_dim }, s);
            };
            defer ple_modulated.deinit();
            var ple_projected = try l.ple_proj.forwardArray(ple_modulated, s);
            defer ple_projected.deinit();

            var ple_normed = try offload.fastRmsNorm(ple_projected, l.ple_norm_w, l.rms_norm_eps, s);
            defer ple_normed.deinit();

            if (l.layer_scalar_w) |sw| {
                return offload.fusedAddMul(x2, ple_normed, sw, self.compiled_add_mul, s);
            }
            return offload.Array.add(x2, ple_normed, s);
        }

        fn prefillOnDevice(self: *Self, prompt_tokens: []const usize) !offload.Array {
            const T = prompt_tokens.len;
            if (T == 0) return error.EmptyPrompt;
            self.resetCache();

            const s = self.stream;
            var indices_arr = try self.createTokenIndicesArray(prompt_tokens);
            defer indices_arr.deinit();

            var logits_bf16 = try self.forwardTokenArrayToLogits(indices_arr, 0, T, s);
            defer logits_bf16.deinit();

            var best_tok_arr = try self.exclusion_mask.?.argmax(logits_bf16, s);
            best_tok_arr.asyncEval();
            self.curr_len = T;
            return best_tok_arr;
        }

        fn decodeOnDevice(self: *Self, tok_arr: offload.Array) !offload.Array {
            const s = self.stream;
            const pos = self.curr_len;

            var logits_bf16 = try self.forwardTokenArrayToLogits(tok_arr, pos, 1, s);
            defer logits_bf16.deinit();

            var best_tok_arr = try self.exclusion_mask.?.argmax(logits_bf16, s);
            best_tok_arr.asyncEval();
            self.curr_len = pos + 1;
            return best_tok_arr;
        }

        fn prefillSampled(self: *Self, prompt_tokens: []const usize, sampler: *Sampler) !usize {
            const T = prompt_tokens.len;
            if (T == 0) return error.EmptyPrompt;
            self.resetCache();

            const s = self.stream;
            var indices_arr = try self.createTokenIndicesArray(prompt_tokens);
            defer indices_arr.deinit();

            var logits_bf16 = try self.forwardTokenArrayToLogits(indices_arr, 0, T, s);
            defer logits_bf16.deinit();

            const best_tok = try logits_processor.selectToken(
                self.allocator,
                logits_bf16,
                .{
                    .vocab_size = config.vocab_size,
                    .final_logit_softcapping = config.final_logit_softcapping,
                    .sampling_excluded_tokens = config.sampling_excluded_tokens,
                },
                &self.exclusion_mask.?,
                prompt_tokens,
                sampler,
                s,
            );
            self.curr_len = T;
            return best_tok;
        }

        fn decodeSampled(self: *Self, tok: usize, history_tokens: []const usize, sampler: *Sampler) !usize {
            const s = self.stream;
            const pos = self.curr_len;

            var indices_arr = try self.createTokenIndicesArray(&.{tok});
            defer indices_arr.deinit();

            var logits_bf16 = try self.forwardTokenArrayToLogits(indices_arr, pos, 1, s);
            defer logits_bf16.deinit();

            const best_tok = try logits_processor.selectToken(
                self.allocator,
                logits_bf16,
                .{
                    .vocab_size = config.vocab_size,
                    .final_logit_softcapping = config.final_logit_softcapping,
                    .sampling_excluded_tokens = config.sampling_excluded_tokens,
                },
                &self.exclusion_mask.?,
                history_tokens,
                sampler,
                s,
            );
            self.curr_len = pos + 1;
            return best_tok;
        }

        pub fn generate(
            self: *Self,
            allocator: std.mem.Allocator,
            prompt_tokens: []const usize,
            options: E4BGenerateOptions,
        ) !Generation {
            if (prompt_tokens.len == 0) return error.EmptyPrompt;
            var history: std.ArrayList(usize) = .empty;
            errdefer history.deinit(allocator);
            try history.appendSlice(allocator, prompt_tokens);
            return .{
                .model = self,
                .allocator = allocator,
                .options = options,
                .sampler = Sampler.init(options.sampling),
                .history = history,
                .prompt_len = prompt_tokens.len,
            };
        }

        pub const Generation = struct {
            model: *Self,
            allocator: std.mem.Allocator,
            options: E4BGenerateOptions,
            sampler: Sampler,
            history: std.ArrayList(usize),
            prompt_len: usize,
            pending: ?offload.Array = null,
            started: bool = false,
            finished: bool = false,

            pub fn deinit(self: *Generation) void {
                if (self.pending) |*p| p.deinit();
                self.pending = null;
                self.history.deinit(self.allocator);
            }

            pub fn isPipelined(self: *const Generation) bool {
                return self.options.sampling.isPlainGreedy();
            }

            pub fn generated(self: *const Generation) []const usize {
                return self.history.items[self.prompt_len..];
            }

            pub fn tokens(self: *const Generation) []const usize {
                return self.history.items;
            }

            pub fn next(self: *Generation) !?usize {
                if (self.finished) return null;
                if (self.options.max_new_tokens == 0) {
                    self.finished = true;
                    return null;
                }
                const model = self.model;

                // GPU 硬件加速路径 (已加载 GPU 权重)
                if (model.gpu_layers != null) {
                    if (!self.isPipelined()) {
                        const tok = if (!self.started)
                            try model.prefillSampled(self.history.items, &self.sampler)
                        else
                            try model.decodeSampled(self.history.items[self.history.items.len - 1], self.history.items, &self.sampler);
                        self.started = true;
                        return try self.emit(tok);
                    }

                    if (!self.started) {
                        self.pending = try model.prefillOnDevice(self.history.items);
                        self.started = true;
                    }
                    var current = self.pending.?;
                    self.pending = null;
                    defer current.deinit();

                    if (self.generated().len + 1 < self.options.max_new_tokens) {
                        self.pending = try model.decodeOnDevice(current);
                    }
                    const tok: usize = @intCast(try current.itemUint32());
                    return try self.emit(tok);
                }

                // CPU 通用计算图回退路径：无 GPU 离载时的跨平台端到端生成
                const tok = try self.stepCpu();
                self.started = true;
                return try self.emit(tok);
            }

            fn stepCpu(self: *Generation) !usize {
                var graph = autodiff.Graph.init(self.allocator, .{ .device = .cpu });
                defer graph.deinit();

                const hist_tokens = self.history.items;
                const T = hist_tokens.len;
                const tok_t = try graph.tensorND(&.{ 1, T }, false);
                for (hist_tokens, 0..) |tok, i| {
                    tok_t.data[i] = @floatFromInt(tok);
                }

                const logits = try self.model.forward(&graph, tok_t);
                const last_logits = logits.data[(T - 1) * config.vocab_size .. T * config.vocab_size];

                const penalty = self.sampler.options.repetition_penalty;
                const recent = gemma4.recentWindowFor(hist_tokens, self.sampler.options);

                if (self.sampler.isGreedy()) {
                    return sampling.argmaxWithPenalty(last_logits, config.vocab_size, config.sampling_excluded_tokens, recent, penalty);
                }

                const arena = graph.arenaAllocator();
                const work_logits = try arena.alloc(f32, config.vocab_size);
                @memcpy(work_logits, last_logits);
                try sampling.applyRepetitionPenaltyInPlace(arena, work_logits, recent, penalty);
                sampling.maskExcludedInPlace(work_logits, config.sampling_excluded_tokens);
                return self.sampler.sampleLogits(graph.executor(), arena, work_logits);
            }

            fn emit(self: *Generation, tok: usize) !usize {
                try self.history.append(self.allocator, tok);
                const stop = std.mem.indexOfScalar(usize, self.options.stop_tokens, tok) != null;
                if (stop or self.generated().len >= self.options.max_new_tokens) {
                    self.finished = true;
                    if (self.pending) |*p| p.deinit();
                    self.pending = null;
                }
                return tok;
            }
        };

        /// 纯 PyTorch / JAX 风格模块前向计算：通过计算图完成，与具体硬件后端完全解耦
        pub fn forward(self: *const Self, graph: *autodiff.Graph, token_ids: anytype) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const B = token_ids.shape.dims[0];
            const T = token_ids.shape.dims[1];

            // 1. 词嵌入与 Gemma 嵌入缩放 sqrt(hidden_size)
            const tok_emb = try self.embed_tokens.forward(graph, token_ids);
            const scale = @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
            var h = try graph.mulScalar(tok_emb, scale);

            // 2. 逐层嵌入输入提取并重塑为 4D [B, T, num_layers, ple_dim]
            const ple_tok_flat = try self.embed_tokens_per_layer.forward(graph, token_ids);
            const ple_tokens = try graph.reshape(ple_tok_flat, &.{ B, T, config.num_hidden_layers, config.hidden_size_per_layer_input });

            // 3. 逐层输入预处理 (PLE Preprocessing)
            const ple_inputs = try self.ple_embed.forward(graph, h, ple_tokens);

            // 4. 逐层前向传播
            for (self.layers, 0..) |*layer, layer_idx| {
                const ple_slice_4d = try graph.slice(ple_inputs, &.{
                    .{},
                    .{},
                    .{ .start = layer_idx, .end = layer_idx + 1 },
                    .{},
                });
                const ple_slice = try graph.squeeze(ple_slice_4d, 2);
                h = try layer.forwardWithPle(graph, h, ple_slice);
            }

            // 5. 最终层 RMSNorm
            const h_norm = try self.norm.forward(graph, h);

            // 6. 绑定权重投影 (Tie Word Embeddings) 与软截断
            const h_2d = try graph.reshape(h_norm, &.{ B * T, config.hidden_size });
            const embed_t = try graph.transposeND(self.embed_tokens.weight, 0, 1);
            const logits_raw_2d = try graph.matmul(h_2d, embed_t);
            const capped_logits_2d = try graph.softcap(logits_raw_2d, config.final_logit_softcapping);

            return try graph.reshape(capped_logits_2d, &.{ B, T, config.vocab_size });
        }
    };
}

/// 默认 Gemma 4 E4B 架构模型
pub const Gemma4E4BModel = Gemma4E4BForCausalLM(Gemma4Config.e4b);
pub const Gemma4E4BGpuContext = Gemma4E4BModel;
