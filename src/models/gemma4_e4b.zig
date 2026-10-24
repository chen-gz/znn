//! Gemma 4 E4B (Effective 4B) 4-bit / 8-bit 混合量化模型端侧推理引擎
//!
//! 支持直接零拷贝映射 Safetensors 模型权重与预融合顺序二进制缓存 (`gemma4_e4b_direct.bin`)、
//! 逐层嵌入增强 (Per-Layer Embeddings, PLE)、异构滑动窗口与全局注意力机制，
//! 以及苹果统一内存 (Unified Memory Architecture, UMA) GPU 硬件加速与双阶段异步流水线推理。

const std = @import("std");
const backend = @import("../backend.zig");
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");
const autodiff = @import("../autodiff.zig");
const gemma4 = @import("gemma4.zig");
const gemma4_q4 = @import("gemma4_q4.zig");
const gemma4_gpu = @import("gemma4_gpu.zig");

const Tensor = tensor.Tensor;
const Embedding = nn.transformer.Embedding;
const RMSNorm = nn.RMSNorm;
const deinitModel = nn.core.deinitModel;
const enterModuleScope = nn.core.enterModuleScope;

pub const Gemma4Q4Attention = gemma4_q4.Gemma4Q4Attention;
pub const Gemma4Q4MLP = gemma4_q4.Gemma4Q4MLP;
pub const Gemma4PleLayer = gemma4.Gemma4PleLayer(nn.Q4Linear);
pub const Gemma4PleEmbedding = gemma4.Gemma4PleEmbedding(nn.Q4Linear);

/// Gemma 4 E4B 变换器解码层 (泛型模块装配：Q4 注意力 + Q4 门控 MLP + Q4 PLE 模块)
pub const Gemma4E4BDecoderLayer = gemma4.Gemma4DecoderLayerT(
    Gemma4Q4Attention,
    Gemma4Q4MLP,
    Gemma4PleLayer,
);

/// Gemma 4 E4B 语言模型核心架构 (Gemma4E4BForCausalLM: 纯 PyTorch / JAX 风格模块化定义，完全解耦硬件后端)
pub fn Gemma4E4BForCausalLM(comptime cfg: Gemma4Config) type {
    return struct {
        pub const config: Gemma4Config = cfg;
        const Self = @This();

        embed_tokens: Embedding,
        embed_tokens_per_layer: Embedding,
        ple_embed: Gemma4PleEmbedding,
        layers: []Gemma4E4BDecoderLayer,
        norm: RMSNorm,

        name: ?[]const u8 = null,
        module_type: []const u8 = "Gemma4E4BForCausalLM",

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
                .embed_tokens = embed_tokens,
                .embed_tokens_per_layer = embed_tokens_per_layer,
                .ple_embed = ple_embed,
                .layers = layers,
                .norm = norm,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            deinitModel(&self.embed_tokens, allocator);
            deinitModel(&self.embed_tokens_per_layer, allocator);
            self.ple_embed.deinit(allocator);
            for (self.layers) |*layer| {
                layer.deinit(allocator);
            }
            allocator.free(self.layers);
            deinitModel(&self.norm, allocator);
        }

        pub fn enableHardwareOffload(self: *Self, allocator: std.mem.Allocator) !void {
            try self.ple_embed.enableHardwareOffload(allocator);
            for (self.layers) |*layer| {
                try layer.enableHardwareOffload(allocator);
            }
        }

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

const mlx = backend.mlx;
pub const Gemma4Config = gemma4.Gemma4Config;
pub const Q4Weights = backend.offload.Q4Weights;
pub const QuantLayout = backend.offload.QuantLayout;
pub const Stream = backend.offload.Stream;
pub const GpuKvCache = backend.offload.GpuKvCache;
const TokenExclusionMask = backend.offload.TokenExclusionMask;
const Sampler = backend.sampling.Sampler;
const SamplingOptions = backend.sampling.SamplingOptions;
pub const TensorMeta = nn.TensorMeta;
pub const SafetensorsArchive = nn.SafetensorsArchive;

/// 持有独立键值缓存的解码层数 (其余 num_kv_shared_layers 层复用前层缓存)
pub const num_kv_layers: usize = Gemma4Config.e4b.num_hidden_layers - Gemma4Config.e4b.num_kv_shared_layers;

pub const kvSourceLayer = Gemma4Config.kvSourceLayer;
pub const isKvSharedSource = Gemma4Config.isKvSharedSource;

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

    pub fn run(ctx: *const PlePrepContext, in_arrs: [1]mlx.Array) ![2]mlx.Array {
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
        var ple_proj_scaled = try mlx.Array.multiply(ple_proj_raw, ctx.ple.proj_scale_arr, null);
        defer ple_proj_scaled.deinit();
        var ple_proj_4d = try ple_proj_scaled.reshape(&.{ 1, T, ctx.ple.num_layers, ctx.ple.ple_dim }, null);
        defer ple_proj_4d.deinit();
        var ple_proj_normed = try mlx.Array.fastRmsNorm(ple_proj_4d, ctx.ple.proj_norm_w, ctx.rms_norm_eps, null);
        defer ple_proj_normed.deinit();

        var ple_sum = try mlx.Array.add(ple_proj_normed, ple_tokens_arr, null);
        defer ple_sum.deinit();
        var ple_inputs = try mlx.Array.multiply(ple_sum, ctx.ple.input_scale_arr, null);
        errdefer ple_inputs.deinit();

        return .{ x_cur, ple_inputs };
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
    /// 当前层的键值缓存是否被后续键值共享层复用 (需保留本步有效键值切片)
    is_kv_shared_source: bool,
    layer_scalar: f32,
    layer_scalar_w: ?mlx.Array = null,
    is_sliding: bool,
    sliding_window: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// 旋转位置编码：滑动窗口层为标准半切分 RoPE，全局注意力层为在完整 head_dim 上旋转的比例旋转位置编码 (Proportional RoPE)
    rope: gemma4_gpu.GpuRope,
    rms_norm_eps: f32,
    /// 门控多层感知机与逐层输入门控的高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 形式 (取自 `Gemma4Config.hidden_activation`)
    hidden_activation: backend.GeluApproximation,
    compiled_tail: ?mlx.Closure = null,

    /// 执行解码层注意力投影之后的全尾部计算 (Post-Attention RMSNorm + Gated MLP + PLE + LayerScalar)
    pub fn forwardTail(
        self: *const GpuLayerE4B,
        x_in: mlx.Array,
        o_out: mlx.Array,
        ple_slice: mlx.Array,
        s: ?Stream,
    ) !mlx.Array {
        var post_attn = try mlx.Array.fastRmsNorm(o_out, self.post_attn_w, self.rms_norm_eps, s);
        defer post_attn.deinit();
        var x1 = try backend.mlx_blocks.residualScale(x_in, post_attn, null, s);
        defer x1.deinit();

        var post_mlp = try backend.mlx_blocks.gatedMlp(
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
        var x2 = try backend.mlx_blocks.residualScale(x1, post_mlp, null, s);
        defer x2.deinit();

        var ple_gate_raw = try self.ple_gate.forwardArray(x2, s);
        defer ple_gate_raw.deinit();
        var act_ple = try ple_gate_raw.gelu(self.hidden_activation, s);
        defer act_ple.deinit();
        var ple_modulated = try mlx.Array.multiply(act_ple, ple_slice, s);
        defer ple_modulated.deinit();

        var ple_projected = try self.ple_proj.forwardArray(ple_modulated, s);
        defer ple_projected.deinit();
        var ple_normed = try mlx.Array.fastRmsNorm(ple_projected, self.ple_norm_w, self.rms_norm_eps, s);
        defer ple_normed.deinit();

        return backend.mlx_blocks.residualScale(x2, ple_normed, self.layer_scalar_w, s);
    }

    pub fn tailRun(self: *const GpuLayerE4B, in_arrs: [3]mlx.Array) !mlx.Array {
        return self.forwardTail(in_arrs[0], in_arrs[1], in_arrs[2], null);
    }

    /// 静态编译当前解码层单步解码阶段 (`seq_len == 1`) 注意力投影之后的全尾部计算图
    pub fn initCompiledTail(self: *GpuLayerE4B) void {
        if (!mlx.has_mlx) return;
        self.compiled_tail = mlx.compileFn(3, 1, GpuLayerE4B, tailRun, false, self) catch null;
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
    /// 与 `config.sampling_excluded_tokens` 一致的 GPU 词表排除掩码 (贪婪 argmax 与随机采样候选选择前叠加)
    exclusion_mask: TokenExclusionMask,

    // 全部独立 KV 层 (num_kv_layers 个) 的块式预分配显存键值缓存 (Pre-allocated Chunked GPU Key-Value Cache)
    gpu_kv: [num_kv_layers]?GpuKvCache = @splat(null),
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

    /// 计算预融合顺序二进制权重缓存 (`gemma4_e4b_direct.bin`) 的精确字节总长度
    pub fn computeExpectedDirectCacheSize(cfg: Gemma4Config) usize {
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
        var model = try loadWeights(allocator, io, dir_path);
        errdefer model.deinit();
        if (options.warmup) try model.warmup();
        return model;
    }

    /// 加载全部权重并构建模型 (不含预热)：任一步骤失败时精确回收此前已构建的全部资源
    fn loadWeights(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !Gemma4E4BModel {
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

        var stream = try Stream.defaultGpu();
        errdefer stream.deinit();

        _ = mlx.configureOptimalWiredMemory();
        mlx.enableCompile();
        mlx.setCompileMode(.enabled);

        var direct_reader: DirectCacheReader = if (direct_mmap_opt) |mapped|
            try DirectCacheReader.init(mapped, expected_cache_bytes)
        else
            .{ .data = &.{} };

        // 若命中顺序直通缓存则无需分配 52MB 拼接缓冲并消除 2.3GB 内存拷贝
        const max_fused_bytes = (cfg.intermediate_size * 2) * (cfg.hidden_size / 4) * @sizeOf(u32);
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

        // 1. 词表嵌入 (与 LM Head 权重绑定) 与最终层归一化
        var embed_tokens = try src.q4("language_model.model.embed_tokens", cfg.hidden_size, cfg.vocab_size, 64, 4);
        errdefer embed_tokens.deinit();
        var norm_w = try src.norm("language_model.model.norm.weight", cfg.hidden_size);
        errdefer norm_w.deinit();

        // 2. 逐层嵌入输入结构 (Per-Layer Embeddings, PLE)
        const ple_total = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input;
        var ple_tokens = try src.q4("language_model.model.embed_tokens_per_layer", ple_total, cfg.vocab_size, 64, 4);
        errdefer ple_tokens.deinit();
        var ple_model_proj = try src.q4("language_model.model.per_layer_model_projection", cfg.hidden_size, ple_total, 64, 4);
        errdefer ple_model_proj.deinit();
        var ple_pnorm_w = try src.norm("language_model.model.per_layer_projection_norm.weight", cfg.hidden_size_per_layer_input);
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

        // PleWeights 仅聚合上方各自带 errdefer 的句柄，此处不再重复注册整体释放，避免出错时二次释放
        const ple = PleWeights{
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

        const sqrt_hidden_val: f32 = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
        var sqrt_hidden_arr = try mlx.Array.fromScalarBf16(sqrt_hidden_val);
        errdefer sqrt_hidden_arr.deinit();

        // 3. 构建全部解码层
        const layers = try allocator.alloc(GpuLayerE4B, cfg.num_hidden_layers);
        errdefer allocator.free(layers);
        var built_layers: usize = 0;
        errdefer for (layers[0..built_layers]) |*l| l.deinit();

        for (0..cfg.num_hidden_layers) |idx| {
            layers[idx] = try loadLayer(&src, cfg, idx);
            built_layers += 1;
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

        var compiled_gelu_mul: ?mlx.Closure = mlx.Array.compileGeluMul(cfg.hidden_activation) catch null;
        errdefer if (compiled_gelu_mul) |*cls| cls.deinit();
        var compiled_add_mul: ?mlx.Closure = mlx.Array.compileAddMul() catch null;
        errdefer if (compiled_add_mul) |*cls| cls.deinit();

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
                compiled_ple_prep = mlx.compileFn(1, 2, PlePrepContext, PlePrepContext.run, false, ctx_ptr) catch null;
            } else |_| {}
        }
        errdefer {
            if (compiled_ple_prep) |*cls| cls.deinit();
            if (ple_prep_ctx) |ctx_ptr| allocator.destroy(ctx_ptr);
        }

        var exclusion_mask = try TokenExclusionMask.init(allocator, cfg.vocab_size, cfg.sampling_excluded_tokens);
        errdefer exclusion_mask.deinit();

        // 4. 在加载期预分配全部独立 KV 层的显存缓冲区，消除 Prefill 首字延迟中的 zeros 内核分配开销
        var gpu_kv: [num_kv_layers]?GpuKvCache = @splat(null);
        errdefer {
            for (&gpu_kv) |*kv_opt| {
                if (kv_opt.*) |*c| c.deinit();
            }
        }
        for (&gpu_kv, layers[0..num_kv_layers]) |*kv_slot, l_ref| {
            const init_cap: usize = if (l_ref.is_sliding) l_ref.sliding_window else 256;
            kv_slot.* = try GpuKvCache.init(init_cap, l_ref.num_kv_heads, l_ref.head_dim, .bfloat16, stream);
        }

        return Gemma4E4BModel{
            .allocator = allocator,
            .layers = layers,
            .ple = ple,
            .embed_tokens = embed_tokens,
            .norm_w = norm_w,
            .lm_head = embed_tokens,
            .config = cfg,
            .stream = stream,
            .exclusion_mask = exclusion_mask,
            .gpu_kv = gpu_kv,
            .curr_len = 0,
            .sqrt_hidden = sqrt_hidden_val,
            .sqrt_hidden_arr = sqrt_hidden_arr,
            .compiled_gelu_mul = compiled_gelu_mul,
            .compiled_add_mul = compiled_add_mul,
            .ple_prep_ctx = ple_prep_ctx,
            .compiled_ple_prep = compiled_ple_prep,
        };
    }

    /// 按直通缓存的存储顺序加载单个解码层：
    /// [k_norm] -> qkv -> o -> gate_up -> down -> ple_gate -> ple_proj -> 6 个层归一化 -> layer_scalar
    fn loadLayer(src: *WeightSource, cfg: Gemma4Config, idx: usize) !GpuLayerE4B {
        const spec = cfg.layerSpec(idx);
        const hd = spec.head_dim;
        const is_full = spec.attn_type == .full_attention;
        const H = cfg.hidden_size;
        const ple_dim = cfg.hidden_size_per_layer_input;
        const out_q = cfg.num_attention_heads * hd;
        const out_kv = cfg.num_key_value_heads * hd;
        const has_kv = idx < num_kv_layers;

        var kn_w: ?mlx.Array = if (has_kv) try src.layerNorm(idx, "self_attn.k_norm.weight", hd) else null;
        errdefer if (kn_w) |*kn| kn.deinit();
        // 注意力投影：持有键值缓存的层将 Q/K/V 融合为单一矩阵，每步减少 QMV 内核派发；共享 KV 层仅需独立 Q 矩阵
        var qkv_proj = if (has_kv)
            try src.layerFusedQ4(idx, &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" }, H, out_q + 2 * out_kv, 64, 4)
        else
            try src.layerQ4(idx, "self_attn.q_proj", H, out_q, 64, 4);
        errdefer qkv_proj.deinit();
        var o_proj = try src.layerQ4(idx, "self_attn.o_proj", out_q, H, 64, 4);
        errdefer o_proj.deinit();

        // 门控前馈网络 (8-bit Quantized MLP)：将 gate_proj 与 up_proj 沿输出维度融合为单一矩阵
        var fused_gate_up = try src.layerFusedQ4(idx, &.{ "mlp.gate_proj", "mlp.up_proj" }, H, cfg.intermediate_size * 2, 64, 8);
        errdefer fused_gate_up.deinit();
        var down_proj = try src.layerQ4(idx, "mlp.down_proj", cfg.intermediate_size, H, 64, 8);
        errdefer down_proj.deinit();

        // 逐层嵌入门控与上投影 (PLE Gating & Projection)
        var ple_gate = try src.layerQ4(idx, "per_layer_input_gate", H, ple_dim, 64, 4);
        errdefer ple_gate.deinit();
        var ple_proj = try src.layerQ4(idx, "per_layer_projection", ple_dim, H, 64, 4);
        errdefer ple_proj.deinit();

        // 均方根层归一化权重 (RMSNorm Weights)
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

        // 层缩放标量 (Layer Scalar, 非 1.0 时直接绑定 BF16 标量张量以消除每步标量创建与类型提升)
        var scalar = try src.layerScalar(idx);
        errdefer if (scalar.arr) |*arr| arr.deinit();

        // 全局注意力层采用 partial_rotary_factor = 0.25 的比例旋转位置编码 (与 12B GPU 路径共用同一实现)
        const rope = try gemma4_gpu.GpuRope.init(hd, spec.attn_type.ropeOptions());

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
            .prev_kv_idx = kvSourceLayer(cfg, idx),
            .is_kv_shared_source = has_kv and isKvSharedSource(cfg, idx),
            .layer_scalar = scalar.value,
            .layer_scalar_w = scalar.arr,
            .is_sliding = !is_full,
            .sliding_window = cfg.sliding_window,
            .num_heads = cfg.num_attention_heads,
            .num_kv_heads = cfg.num_key_value_heads,
            .head_dim = hd,
            .rope = rope,
            .rms_norm_eps = cfg.rms_norm_eps,
            .hidden_activation = cfg.hidden_activation,
        };
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
        self.exclusion_mask.deinit();
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
        var first_arr = try self.prefillOnDevice(&dummy_prompt);
        defer first_arr.deinit();

        var second_arr = try self.decodeOnDevice(first_arr);
        defer second_arr.deinit();

        _ = try second_arr.itemUint32();
        self.stream.synchronize();
        self.resetCache();
        mlx.clearCache();
    }

    /// 解码层与全局权重来源：顺序直通缓存 (Direct Cache)，或 Safetensors 归档 (同时按相同顺序回写直通缓存)
    const WeightSource = struct {
        direct: ?*DirectCacheReader,
        archive: ?SafetensorsArchive,
        cache_writer: ?*std.Io.Writer,
        cache_write_ok: *bool,
        scratch: []u8,
        stream: Stream,
        name_buf: [128]u8 = undefined,

        const LayerScalar = struct {
            value: f32,
            arr: ?mlx.Array,
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

        /// 读取以 `prefix` 命名的量化矩阵 (prefix.weight / prefix.scales / prefix.biases)
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

        /// 读取以 `name` 命名的 BF16 归一化权重 [dim]
        fn norm(self: *WeightSource, name: []const u8, dim: usize) !mlx.Array {
            const bytes = if (self.direct) |r| try r.readBytes(dim * @sizeOf(u16)) else blk: {
                const raw = try self.tensorBytes("{s}", .{name});
                if (raw.len < dim * @sizeOf(u16)) return error.InvalidTensorShape;
                self.writeCache(raw[0 .. dim * @sizeOf(u16)]);
                break :blk raw;
            };
            return mlx.Array.fromRawData(@ptrCast(bytes.ptr), &.{dim}, .bfloat16);
        }

        fn layerQ4(self: *WeightSource, l_idx: usize, sub: []const u8, in_f: usize, out_f: usize, group_size: i32, bits: i32) !Q4Weights {
            var prefix_buf: [128]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&prefix_buf, "language_model.model.layers.{}.{s}", .{ l_idx, sub });
            return self.q4(prefix, in_f, out_f, group_size, bits);
        }

        fn layerNorm(self: *WeightSource, l_idx: usize, sub: []const u8, dim: usize) !mlx.Array {
            var name_buf: [128]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "language_model.model.layers.{}.{s}", .{ l_idx, sub });
            return self.norm(name, dim);
        }

        /// 读取多个同输入维度的量化矩阵并沿输出维度融合为单一矩阵 (直通缓存中已按融合布局存储)
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

        /// 读取层缩放标量 (缺省为 1.0)，非 1.0 时同时构建 BF16 标量张量
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
            const value = backend.vec_math.bf16ToF32(std.mem.readInt(u16, &bits, .little));
            const arr: ?mlx.Array = if (value != 1.0) try mlx.Array.fromRawData(@ptrCast(&bits), &.{1}, .bfloat16) else null;
            return .{ .value = value, .arr = arr };
        }
    };

    fn createTokenIndicesArray(self: *const Gemma4E4BModel, tokens: []const usize) !mlx.Array {
        try tensor.checkIndices(tokens, self.config.vocab_size);
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
        var emb = try self.embed_tokens.gatherRows(indices_arr, self.sqrt_hidden_arr, s);
        defer emb.deinit();
        return emb.reshape(&.{ 1, T, self.config.hidden_size }, s);
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
        var pdeq = try self.ple.embed_tokens_per_layer.gatherRows(indices_arr, self.ple.embed_scale_arr, s);
        defer pdeq.deinit();
        return pdeq.reshape(&.{ 1, T, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input }, s);
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
        // x_cur 在逐层循环中被替换为新层输出，defer 在作用域退出时释放其最终持有的数组 (成功与出错路径各一次)
        defer x_cur.deinit();
        defer ple_inputs.deinit();
        const num_layers: i32 = @intCast(self.config.num_hidden_layers);

        // 3. 执行 42 层 Transformer 前向传播：
        // - 当 T == 1 时重塑为 [1, 42, 256] 并沿 axis 1 切分，直接获得 [1, 1, 256] 切片；
        // - 当 T > 1 时沿 axis 2 切分获得 [1, T, 1, 256] 零拷贝视图，直接传入 forwardLayer 与 4D 门控视图融合，消除 42 次非连续张量重塑拷贝内核
        if (T == 1) {
            var ple_3d = try ple_inputs.reshape(&.{ 1, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input }, s);
            defer ple_3d.deinit();
            var ple_vec = try ple_3d.split(num_layers, 1, s);
            defer ple_vec.deinit();

            for (self.layers, 0..) |*l, idx| {
                var ple_slice = ple_vec.get(idx).?;
                defer ple_slice.deinit();

                const x_next = try self.forwardLayer(x_cur, l, idx, ple_slice, pos, 1, s);
                x_cur.deinit();
                x_cur = x_next;
            }
        } else {
            var ple_vec = try ple_inputs.split(num_layers, 2, s);
            defer ple_vec.deinit();

            for (self.layers, 0..) |*l, idx| {
                var ple_slice = ple_vec.get(idx).?;
                defer ple_slice.deinit();

                const x_next = try self.forwardLayer(x_cur, l, idx, ple_slice, pos, T, s);
                x_cur.deinit();
                x_cur = x_next;
            }
        }

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

    /// 预填充并在设备端贪婪选取首词元：构建预填充计算图，叠加词表排除掩码后 argmax，异步派发并返回 GPU 上的
    /// 词元 ID 张量 (`shape = [1], dtype = uint32`)，调用方可直接把它喂给 `decodeOnDevice` 而无需等待 GPU 完成
    fn prefillOnDevice(self: *Gemma4E4BModel, prompt_tokens: []const usize) !mlx.Array {
        const T = prompt_tokens.len;
        if (T == 0) return error.EmptyPrompt;
        self.resetCache();

        const s = self.stream;
        var indices_arr = try self.createTokenIndicesArray(prompt_tokens);
        defer indices_arr.deinit();

        var logits_bf16 = try self.forwardTokenArrayToLogits(indices_arr, 0, T, s);
        defer logits_bf16.deinit();

        var best_tok_arr = try self.exclusion_mask.argmax(logits_bf16, s);
        best_tok_arr.asyncEval();
        self.curr_len = T;
        return best_tok_arr;
    }

    /// 单步解码并在设备端贪婪选取下一个词元：直接消费上一步在 GPU 上的词元张量构建计算图并异步派发，
    /// 使 CPU 构建第 t+1 步计算图与 GPU 执行第 t 步重叠
    fn decodeOnDevice(self: *Gemma4E4BModel, tok_arr: mlx.Array) !mlx.Array {
        const s = self.stream;
        const pos = self.curr_len;

        var logits_bf16 = try self.forwardTokenArrayToLogits(tok_arr, pos, 1, s);
        defer logits_bf16.deinit();

        var best_tok_arr = try self.exclusion_mask.argmax(logits_bf16, s);
        best_tok_arr.asyncEval();
        self.curr_len = pos + 1;
        return best_tok_arr;
    }

    /// 预填充并按 sampler 在主机端完成选取 (带重复惩罚的贪婪采样与随机采样)，重复惩罚作用于完整的 Prompt 词元
    fn prefillSampled(self: *Gemma4E4BModel, prompt_tokens: []const usize, sampler: *Sampler) !usize {
        const T = prompt_tokens.len;
        if (T == 0) return error.EmptyPrompt;
        self.resetCache();

        const s = self.stream;
        var indices_arr = try self.createTokenIndicesArray(prompt_tokens);
        defer indices_arr.deinit();

        var logits_bf16 = try self.forwardTokenArrayToLogits(indices_arr, 0, T, s);
        defer logits_bf16.deinit();

        const best_tok = try gemma4_gpu.selectToken(self.allocator, logits_bf16, self.config, &self.exclusion_mask, prompt_tokens, sampler, s);
        self.curr_len = T;
        return best_tok;
    }

    /// 单步解码并按 sampler 在主机端完成选取；`history_tokens` 为截至并包含 `tok` 的全部词元，重复惩罚作用于完整历史
    fn decodeSampled(self: *Gemma4E4BModel, tok: usize, history_tokens: []const usize, sampler: *Sampler) !usize {
        const s = self.stream;
        const pos = self.curr_len;

        var indices_arr = try self.createTokenIndicesArray(&.{tok});
        defer indices_arr.deinit();

        var logits_bf16 = try self.forwardTokenArrayToLogits(indices_arr, pos, 1, s);
        defer logits_bf16.deinit();

        const best_tok = try gemma4_gpu.selectToken(self.allocator, logits_bf16, self.config, &self.exclusion_mask, history_tokens, sampler, s);
        self.curr_len = pos + 1;
        return best_tok;
    }

    /// 自回归生成选项 (遵循 AGENTS.md §4：所有字段均有默认值)
    pub const GenerateOptions = struct {
        /// 采样设置，默认采用 Gemma 4 官方生成配置
        sampling: SamplingOptions = gemma4.default_generation,
        /// 最多生成的词元数 (含首个词元)
        max_new_tokens: usize = 256,
        /// 生成到这些词元 (含) 即停止
        stop_tokens: []const usize = &Gemma4E4BModel.default_stop_tokens,

        pub fn defaultOptions() GenerateOptions {
            return .{};
        }
    };

    pub const default_stop_tokens = [_]usize{ gemma4.Gemma4Vocabulary.EOS, gemma4.Gemma4Vocabulary.EOT };

    /// 开始一次自回归生成：重置键值缓存并返回逐词元产出的 `Generation`。同一模型同一时刻只能进行一次生成
    /// (共享键值缓存)；生成对象须在模型之前释放
    pub fn generate(
        self: *Gemma4E4BModel,
        allocator: std.mem.Allocator,
        prompt_tokens: []const usize,
        options: GenerateOptions,
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

    /// 自回归生成会话：`next` 逐个返回生成的词元，生成结束 (遇到停止词元或达到 `max_new_tokens`) 后返回 null。
    /// 不带重复惩罚的贪婪采样自动启用双阶段 GPU 流水线 (在等待第 t 步结果前已派发第 t+1 步，与 mlx-lm `generate_step` 一致；
    /// 因此遇到停止词元时会多算一步并丢弃)；其余采样设置逐步同步执行并在主机端完成选取
    pub const Generation = struct {
        model: *Gemma4E4BModel,
        allocator: std.mem.Allocator,
        options: GenerateOptions,
        sampler: Sampler,
        /// Prompt 与已生成词元
        history: std.ArrayList(usize),
        prompt_len: usize,
        /// 流水线模式下已派发、尚未取回的下一个词元 (GPU 上的 `shape = [1]` 张量)
        pending: ?mlx.Array = null,
        started: bool = false,
        finished: bool = false,

        pub fn deinit(self: *Generation) void {
            if (self.pending) |*p| p.deinit();
            self.pending = null;
            self.history.deinit(self.allocator);
        }

        /// 是否使用双阶段 GPU 流水线 (仅不带重复惩罚的贪婪采样)
        pub fn isPipelined(self: *const Generation) bool {
            return self.options.sampling.isPlainGreedy();
        }

        /// 已生成的词元 (不含 Prompt)
        pub fn generated(self: *const Generation) []const usize {
            return self.history.items[self.prompt_len..];
        }

        /// Prompt 与已生成的全部词元
        pub fn tokens(self: *const Generation) []const usize {
            return self.history.items;
        }

        /// 生成下一个词元；生成已结束时返回 null
        pub fn next(self: *Generation) !?usize {
            if (self.finished) return null;
            if (self.options.max_new_tokens == 0) {
                self.finished = true;
                return null;
            }
            const model = self.model;

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
            // 取回第 t 步结果前先派发第 t+1 步
            if (self.generated().len + 1 < self.options.max_new_tokens) {
                self.pending = try model.decodeOnDevice(current);
            }
            const tok: usize = @intCast(try current.itemUint32());
            return try self.emit(tok);
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

        // 旋转后的查询张量仅在成功构建后登记，作用域退出 (成功或出错) 时恰好释放一次
        var q_rot_opt: ?mlx.Array = null;
        defer if (q_rot_opt) |*q| q.deinit();
        var eff_k: mlx.Array = undefined;
        var eff_v: mlx.Array = undefined;
        var eff_k_free: ?mlx.Array = null;
        var eff_v_free: ?mlx.Array = null;
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

            q_rot_opt = try backend.mlx_blocks.projectHeads(q_raw_slice, l.q_norm_w, seq_len, l.num_heads, l.head_dim, l.rms_norm_eps, &l.rope, pos, s);
            var k_rot = try backend.mlx_blocks.projectHeads(k_raw_slice, l.k_norm_w.?, seq_len, l.num_kv_heads, l.head_dim, l.rms_norm_eps, &l.rope, pos, s);
            defer k_rot.deinit();
            var v_4d = try backend.mlx_blocks.projectHeads(v_raw_slice, null, seq_len, l.num_kv_heads, l.head_dim, l.rms_norm_eps, null, pos, s);
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
                // 本步有效键值切片交由缓存持有，供后续键值共享层复用 (共享层与源层注意力类型相同，可见键范围一致)
                cache.clearStepSlices();
                cache.step_eff_k = view.k;
                cache.step_eff_v = view.v;
            } else {
                eff_k_free = view.k;
                eff_v_free = view.v;
            }
        } else {
            q_rot_opt = try backend.mlx_blocks.projectHeads(qkv_raw, l.q_norm_w, seq_len, l.num_heads, l.head_dim, l.rms_norm_eps, &l.rope, pos, s);
            const cache = &self.gpu_kv[l.prev_kv_idx].?;
            eff_k = cache.step_eff_k orelse cache.k;
            eff_v = cache.step_eff_v orelse cache.v;
        }
        const q_rot = q_rot_opt.?;

        // 缩放点积注意力 (Scaled Dot-Product Attention, SDPA, scaling = 1.0)：单词元解码与键数不超过窗口的多词元步走融合因果内核，
        // 滑动窗口层上键数超过窗口的多词元步 (如长 Prompt 预填充) 施加逐查询滑动窗口掩码
        var attn_out_4d = try backend.offload.cachedAttention(self.allocator, q_rot, eff_k, eff_v, 1.0, l.is_sliding, l.sliding_window, pos, seq_len, s);
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
            return l.forwardTail(x_in, o_out, ple_slice, s);
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

        var hidden = try mlx.Array.fusedGeluMul(gate_out, up_out, self.config.hidden_activation, self.compiled_gelu_mul, s);
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

        // 当 seq_len > 1 时，ple_slice 来自 ple_inputs.split(num_hidden_layers, 2)，形状为非连续 4D 视图 [1, T, 1, 256]；
        // 将连续的 ple_gate_raw [1, T, 256] 以 O(1) 零拷贝视图重塑为 [1, T, 1, 256]，使 fusedGeluMul 直接融合读取步进视图并输出连续张量，
        // 随后以 O(1) 零拷贝视图恢复为 [1, T, 256]，彻底消除 42 层非连续 reshape 触发的 GPU 拷贝内核！
        var ple_modulated = if (seq_len == 1)
            try mlx.Array.fusedGeluMul(ple_gate_raw, ple_slice, self.config.hidden_activation, self.compiled_gelu_mul, s)
        else blk: {
            const ple_dim = self.config.hidden_size_per_layer_input;
            var ple_gate_4d = try ple_gate_raw.reshape(&.{ 1, seq_len, 1, ple_dim }, s);
            defer ple_gate_4d.deinit();
            var ple_mod_4d = try mlx.Array.fusedGeluMul(ple_gate_4d, ple_slice, self.config.hidden_activation, self.compiled_gelu_mul, s);
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
