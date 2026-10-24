//! Gemma 4 E4B (Effective 4B) 语言模型架构
//!
//! 遵循 PyTorch / JAX 纯 Module 设计范式：
//! - 采用泛型变换器解码层组合：Q4 混合注意力 + 8-bit 量化门控 MLP + 逐层嵌入增强 (PLE) 模块；
//! - 完全解耦底层硬件细节（零专有硬件数组依赖），模型定义只持有参数张量与子模块；
//! - 端到端前向传播由 `autodiff.Graph` 与 `Tensor` 统一调度；
//! - 性能通过后端 offload 引擎分发：可选集成底层硬件加速上下文 (`Gemma4E4BGpuContext`)，
//!   兼具纯模块的架构清晰度与苹果统一内存 (UMA) GPU 的极致推理吞吐。

const std = @import("std");
const tensor = @import("../tensor.zig");
const nn = @import("../nn.zig");
const autodiff = @import("../autodiff.zig");
const gemma4 = @import("gemma4.zig");
const gemma4_q4 = @import("gemma4_q4.zig");
pub const gemma4_e4b_gpu = @import("gemma4_e4b_gpu.zig");

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

// GPU 硬件加速引擎导出 (按需调用底层硬件 Runner)
pub const Gemma4E4BGpuContext = gemma4_e4b_gpu.Gemma4E4BGpuContext;
pub const GpuLayerE4B = gemma4_e4b_gpu.GpuLayerE4B;
pub const PleWeights = gemma4_e4b_gpu.PleWeights;
pub const DirectCacheReader = Gemma4E4BGpuContext.DirectCacheReader;
pub const computeExpectedDirectCacheSize = Gemma4E4BGpuContext.computeExpectedDirectCacheSize;
pub const DIRECT_CACHE_MAGIC = Gemma4E4BGpuContext.DIRECT_CACHE_MAGIC;
pub const DIRECT_CACHE_VERSION = Gemma4E4BGpuContext.DIRECT_CACHE_VERSION;
pub const DIRECT_CACHE_FILENAME = Gemma4E4BGpuContext.DIRECT_CACHE_FILENAME;
pub const LoadOptions = Gemma4E4BGpuContext.LoadOptions;
pub const GenerateOptions = Gemma4E4BGpuContext.GenerateOptions;
pub const Generation = Gemma4E4BGpuContext.Generation;
pub const num_kv_layers = gemma4_e4b_gpu.num_kv_layers;
pub const kvSourceLayer = Gemma4Config.kvSourceLayer;
pub const isKvSharedSource = Gemma4Config.isKvSharedSource;

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

        allocator: std.mem.Allocator,
        embed_tokens: Embedding,
        embed_tokens_per_layer: Embedding,
        ple_embed: Gemma4PleEmbedding,
        layers: []Gemma4E4BDecoderLayer,
        norm: RMSNorm,

        /// 可选的底层硬件加速上下文 (独占持有 GPU 异步流水线、KVCache 与零拷贝权重)
        gpu_context: ?Gemma4E4BGpuContext = null,

        name: ?[]const u8 = null,
        module_type: []const u8 = "Gemma4E4BForCausalLM",

        pub const DIRECT_CACHE_MAGIC = Gemma4E4BGpuContext.DIRECT_CACHE_MAGIC;
        pub const DIRECT_CACHE_VERSION = Gemma4E4BGpuContext.DIRECT_CACHE_VERSION;
        pub const DIRECT_CACHE_FILENAME = Gemma4E4BGpuContext.DIRECT_CACHE_FILENAME;
        pub const computeExpectedDirectCacheSize = Gemma4E4BGpuContext.computeExpectedDirectCacheSize;
        pub const DirectCacheReader = Gemma4E4BGpuContext.DirectCacheReader;

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
                .gpu_context = null,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.gpu_context) |*ctx| {
                ctx.deinit();
                self.gpu_context = null;
            }
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

        /// 从 Safetensors / 预融合直接二进制文件装载端到端硬件加速推理模型
        pub fn loadFromDirectory(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !Self {
            return loadFromDirectoryWithOptions(allocator, io, dir_path, .{});
        }

        pub fn loadFromDirectoryWithOptions(
            allocator: std.mem.Allocator,
            io: std.Io,
            dir_path: []const u8,
            options: LoadOptions,
        ) !Self {
            var gpu_ctx = try Gemma4E4BGpuContext.loadFromDirectoryWithOptions(allocator, io, dir_path, options);
            errdefer gpu_ctx.deinit();

            var model = try Self.init(allocator);
            model.gpu_context = gpu_ctx;
            return model;
        }

        pub fn warmup(self: *Self) !void {
            if (self.gpu_context) |*ctx| {
                try ctx.warmup();
            }
        }

        pub fn resetCache(self: *Self) void {
            if (self.gpu_context) |*ctx| {
                ctx.resetCache();
            }
        }

        /// 端到端流式自回归生成：优先分发至高性能 GPU 上下文，无硬件加速时可扩展 CPU 计算图回退
        pub fn generate(
            self: *Self,
            allocator: std.mem.Allocator,
            prompt_tokens: []const usize,
            options: GenerateOptions,
        ) !Generation {
            if (self.gpu_context) |*ctx| {
                return ctx.generate(allocator, prompt_tokens, options);
            }
            return error.HardwareOffloadRequired;
        }

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
