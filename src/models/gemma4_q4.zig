//! Gemma 4 专用 4-bit 量化 (Q4) 模型架构与推理实现 (Gemma 4 Q4 Architecture)
//!
//! 包含 Q4Linear 量化线性层、Gemma4Q4Attention 混合注意力、Gemma4Q4DecoderLayer 与 Gemma4Q4ForCausalLM。
//! 采用 split-half Q4Block，并通过 Executor 进行多核并行，热路径不派生临时 OS 线程。

const std = @import("std");
const backend = @import("../backend.zig");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const nn = @import("../nn.zig");
const gemma4 = @import("gemma4.zig");

const offload = backend.offload;
const Stream = offload.Stream;
const Q4Weights = offload.Q4Weights;
const TokenExclusionMask = nn.TokenExclusionMask;
const transformer_blocks = nn.transformer_blocks;
const softcapLogits = transformer_blocks.softcapLogits;
const penalizeDevice = transformer_blocks.penalizeDevice;
const DeviceCandidates = transformer_blocks.DeviceCandidates;
const HostCandidates = transformer_blocks.HostCandidates;
const GpuRope = nn.Rope;

const Tensor = tensor.Tensor;
const Q4Block = tensor.quant.Q4Block;
const Candidate = tensor.Candidate;
const SamplingOptions = tensor.SamplingOptions;
const TokenRange = tensor.TokenRange;
const Sampler = gemma4.Sampler;
const Gemma4Config = gemma4.Gemma4Config;
const Gemma4AttentionType = gemma4.Gemma4AttentionType;
const Gemma4KVCache = gemma4.Gemma4KVCache;
const LmHeadWeights = transformer_blocks.LmHeadWeights;
const Embedding = nn.transformer.Embedding;
const RMSNorm = nn.RMSNorm;
const vec_math = backend.vec_math;

const freePersistentTensor = nn.core.freePersistentTensor;
const deinitModel = nn.core.deinitModel;
const enterModuleScope = nn.core.enterModuleScope;

pub const Q4Linear = nn.Q4Linear;

/// Gemma 4 门控前馈多层感知机 (Q4 量化版本)
pub const Gemma4Q4MLP = nn.GatedMLP(nn.Q4Linear);

/// Gemma 4 混合注意力机制 (Q4 量化版本)
pub const Gemma4Q4Attention = gemma4.Gemma4AttentionT(Q4Linear);

/// Gemma 4 解码器层 (Q4 量化版本)
pub const Gemma4Q4DecoderLayer = gemma4.Gemma4DecoderLayerT(Gemma4Q4Attention, Gemma4Q4MLP, void);

/// 4-bit 量化因果语言模型 (Gemma4Q4ForCausalLM)
/// 整体内存降低至原始全精度的 1/4 到 1/8，使得 12B 模型能够在 16GB 设备上完整运行！
pub fn Gemma4Q4ForCausalLM(comptime cfg: Gemma4Config) type {
    return struct {
        pub const config: Gemma4Config = cfg;
        const Self = @This();

        embed_tokens: Embedding,
        embed_u16_mmap: ?[]const u16 = null,
        mmap_bytes: ?[]const u8 = null,
        layers: []Gemma4Q4DecoderLayer,
        norm: RMSNorm,
        /// 尚未交给 GPU 解码上下文的独立 LM Head：只在 GPU 上下文不存在 (未启用或构建失败) 时持有；
        /// GPU 上下文构建成功后所有权转移至 `gpu_context` 并将本字段置空，两者绝不同时持有同一份权重
        lm_head_weights: ?LmHeadWeights = null,
        /// 端到端 GPU 解码上下文 (独占持有 GPU 解码层与 LM Head)
        gpu_context: ?Gemma4GpuContext = null,
        /// 键值缓存最新内容的所在位置：GPU 路径推进后 CPU 层缓存只更新游标，内容在切回 CPU 解码前从 GPU 同步回来；
        /// CPU 路径推进后 GPU 缓存在下一次 GPU 解码前从 CPU 同步过去
        kv_owner: KvOwner = .cpu,

        name: ?[]const u8 = null,
        module_type: []const u8 = "Gemma4Q4ForCausalLM",

        pub fn init(allocator: std.mem.Allocator) !Self {
            const embed_tokens = try Embedding.init(allocator, config.vocab_size, config.hidden_size);
            errdefer deinitModel(&embed_tokens, allocator);
            const layers = try allocator.alloc(Gemma4Q4DecoderLayer, config.num_hidden_layers);
            errdefer allocator.free(layers);
            var initialized_layers: usize = 0;
            errdefer for (layers[0..initialized_layers]) |*layer| layer.deinit(allocator);

            for (0..config.num_hidden_layers) |i| {
                const is_full = ((i + 1) % 6 == 0);
                const attn_type: Gemma4AttentionType = if (is_full) .full_attention else .sliding_attention;
                const head_dim = if (is_full) config.global_head_dim else config.head_dim;
                const num_kv = if (is_full) config.num_global_key_value_heads else config.num_key_value_heads;

                layers[i] = try Gemma4Q4DecoderLayer.init(
                    allocator,
                    attn_type,
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                    num_kv,
                    head_dim,
                    config.sliding_window,
                    config.rms_norm_eps,
                    config.hidden_activation,
                );
                initialized_layers += 1;
            }

            const norm = try RMSNorm.init(allocator, config.hidden_size, config.rms_norm_eps);

            return Self{
                .embed_tokens = embed_tokens,
                .layers = layers,
                .norm = norm,
            };
        }

        /// 键值缓存最新内容的所在设备
        pub const KvOwner = enum { cpu, gpu };

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (self.gpu_context) |*ctx| {
                ctx.deinit();
                self.gpu_context = null;
            }
            if (self.lm_head_weights) |*w| {
                w.deinit();
                self.lm_head_weights = null;
            }
            deinitModel(&self.embed_tokens, allocator);
            for (self.layers) |*layer| {
                layer.deinit(allocator);
            }
            allocator.free(self.layers);
            deinitModel(&self.norm, allocator);
        }

        pub fn forward(self: *const Self, graph: *autodiff.Graph, token_ids: anytype) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const B = token_ids.shape.dims[0];
            const T = token_ids.shape.dims[1];

            // 1. 词表嵌入查找
            const tok_emb = if (self.embed_u16_mmap) |embed_u16| blk: {
                const out_emb = try graph.tensorND(&.{ B, T, config.hidden_size }, false);
                try tensor.embeddingLookupInto(u16, embed_u16, token_ids.data[0 .. B * T], config.hidden_size, 1.0, out_emb.data);
                break :blk out_emb;
            } else try self.embed_tokens.forward(graph, token_ids);
            const sqrt_d: f32 = @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
            var h = try graph.mulScalar(tok_emb, sqrt_d);

            // 2. 逐层前向传播
            for (self.layers) |*layer| {
                h = try layer.forward(graph, h);
            }

            // 3. 最终层 RMSNorm
            const h_norm = try self.norm.forward(graph, h);

            // 4. 绑定权重投影 (Tie Word Embeddings: h * embed_tokens.weight^T)
            const h_2d = try graph.reshape(h_norm, &.{ B * T, config.hidden_size });
            const logits_raw_2d = if (self.embed_u16_mmap) |embed_u16| blk: {
                const out_logits = try graph.tensorND(&.{ B * T, config.vocab_size }, false);
                for (0..B * T) |r| {
                    const h_row = h_2d.data[r * config.hidden_size .. (r + 1) * config.hidden_size];
                    const out_row = out_logits.data[r * config.vocab_size .. (r + 1) * config.vocab_size];
                    for (out_row, 0..) |*o, v| {
                        o.* = vec_math.dotBf16(h_row, embed_u16[v * config.hidden_size .. (v + 1) * config.hidden_size]);
                    }
                }
                break :blk out_logits;
            } else blk: {
                const embed_t = try graph.transposeND(self.embed_tokens.weight, 0, 1);
                break :blk try graph.matmul(h_2d, embed_t);
            };

            // 5. 对数几率软截断 (Logit Softcapping, Gemma 官方标尺 30.0)
            const capped_logits_2d = try graph.softcap(logits_raw_2d, config.final_logit_softcapping);

            return try graph.reshape(capped_logits_2d, &.{ B, T, config.vocab_size });
        }

        /// 4-bit 分块量化格式 (Q4 Block Format) 文件魔数
        pub const Q4_MAGIC = "ZNNQ4G02";
        /// 零拷贝直通预重排格式 (Direct Layout) 文件魔数：
        /// 每层层归一化区段按 4 字节对齐填充，保证其后的 u32 打包权重区段自然对齐
        pub const DIRECT_MAGIC = "ZNNQ4D02";

        const magic_bytes: usize = 8;
        /// 文件头之后依次为最终层 RMSNorm (BF16) 与词表嵌入矩阵 (BF16)，随后为逐层权重
        const embed_offset: usize = magic_bytes + config.hidden_size * 2;
        const embed_bytes: usize = config.vocab_size * config.hidden_size * 2;
        const layers_offset: usize = embed_offset + embed_bytes;

        /// 每层层归一化原始字节数：4 个隐藏维度层归一化 + Q/K-Norm + 层缩放标量 (layer_scalar)，均为 BF16
        pub fn layerNormsBytes(layer_idx: usize) usize {
            const spec = config.layerSpec(layer_idx);
            return (config.hidden_size * 4 + spec.head_dim * 2 + 1) * 2;
        }

        /// 直通布局中层归一化区段按 4 字节对齐填充后的字节数
        pub fn directLayerNormsBytes(layer_idx: usize) usize {
            return std.mem.alignForward(usize, layerNormsBytes(layer_idx), 4);
        }

        const LayerDims = struct {
            q_dim: usize,
            kv_dim: usize,
            /// 全局注意力层 attention_k_eq_v 为 true，不存储独立的 V 投影
            v_dim: usize,
        };

        fn layerDims(layer_idx: usize) LayerDims {
            const spec = config.layerSpec(layer_idx);
            const kv_dim = spec.num_kv_heads * spec.head_dim;
            return .{
                .q_dim = config.num_attention_heads * spec.head_dim,
                .kv_dim = kv_dim,
                .v_dim = if (spec.attn_type == .full_attention) 0 else kv_dim,
            };
        }

        /// 单层全部投影矩阵 (Q, K, V, O, Gate, Up, Down) 的 Q4Block 总数
        fn layerQ4Blocks(layer_idx: usize) usize {
            const d = layerDims(layer_idx);
            const H = config.hidden_size;
            return ((d.q_dim + d.kv_dim + d.v_dim) * H + d.q_dim * H + config.intermediate_size * H * 3) / Q4Block.QK;
        }

        /// 直通布局中每个 Q4 block 占 16 (w_q: 4 * u32) + 2 (scales: f16) + 2 (biases: f16) = 20 字节
        const direct_block_bytes: usize = 20;

        pub const LayerOffsets = struct {
            offsets: [config.num_hidden_layers + 1]usize,
        };

        /// 计算 4-bit 分块量化格式 (ZNNQ4G02) 中全部解码层的绝对字节偏移量表 (末项为文件总长度)
        pub fn computeLayerOffsets() LayerOffsets {
            var res: LayerOffsets = undefined;
            res.offsets[0] = layers_offset;
            for (0..config.num_hidden_layers) |i| {
                res.offsets[i + 1] = res.offsets[i] + layerNormsBytes(i) + layerQ4Blocks(i) * @sizeOf(Q4Block);
            }
            return res;
        }

        /// 计算直通预重排格式 (ZNNQ4D02) 中全部解码层的绝对字节偏移量表 (末项为文件总长度)。
        /// 层区段起点与其中各 u32 权重区段均满足 4 字节对齐。
        pub fn computeDirectLayerOffsets() LayerOffsets {
            var res: LayerOffsets = undefined;
            res.offsets[0] = std.mem.alignForward(usize, layers_offset, 4);
            for (0..config.num_hidden_layers) |i| {
                res.offsets[i + 1] = res.offsets[i] + directLayerNormsBytes(i) + layerQ4Blocks(i) * direct_block_bytes;
            }
            return res;
        }

        /// 计算 4-bit 量化二进制文件 (ZNNQ4G02) 的理论字节大小
        pub fn computeExpectedFileSize() usize {
            return computeLayerOffsets().offsets[config.num_hidden_layers];
        }

        /// 计算直通预重排二进制文件 (ZNNQ4D02) 的理论字节大小
        pub fn computeExpectedDirectFileSize() usize {
            return computeDirectLayerOffsets().offsets[config.num_hidden_layers];
        }

        /// 按存储顺序 (input, post_attn, pre_ffn, post_ffn, q_norm, k_norm, layer_scalar) 填充单层归一化参数
        fn loadLayerNorms(layer: *Gemma4Q4DecoderLayer, bytes: []const u8, head_dim: usize) void {
            const H = config.hidden_size;
            var cursor: usize = 0;
            inline for (.{
                &layer.input_layernorm,
                &layer.post_attention_layernorm,
                &layer.pre_feedforward_layernorm,
                &layer.post_feedforward_layernorm,
            }) |ln| {
                vec_math.widenBf16Bytes(ln.weight.?.data, bytes[cursor .. cursor + H * 2]);
                cursor += H * 2;
            }
            inline for (.{ &layer.self_attn.q_norm, &layer.self_attn.k_norm }) |ln| {
                vec_math.widenBf16Bytes(ln.weight.?.data[0..head_dim], bytes[cursor .. cursor + head_dim * 2]);
                cursor += head_dim * 2;
            }
            vec_math.widenBf16Bytes(layer.layer_scalar.data[0..1], bytes[cursor .. cursor + 2]);
        }

        /// 从直通布局字节流中切出单个投影矩阵的三段式切片 (w_q, scales, biases) 并推进游标
        fn takeDirectMatrix(bytes: []const u8, cursor: *usize, blocks: usize) Gemma4GpuLayer.DirectMatrix {
            const wq_len = blocks * 16;
            const sb_len = blocks * 2;
            const start = cursor.*;
            cursor.* += wq_len + 2 * sb_len;
            return .{
                .wq = bytes[start .. start + wq_len],
                .sc = bytes[start + wq_len .. start + wq_len + sb_len],
                .bi = bytes[start + wq_len + sb_len .. start + wq_len + 2 * sb_len],
            };
        }

        /// 单层 CPU 层归一化参数视图 (供 GPU 解码层复制为常驻显存权重)
        fn layerNormWeights(layer: *const Gemma4Q4DecoderLayer) Gemma4GpuLayer.NormWeights {
            const hd = layer.self_attn.head_dim;
            return .{
                .input = layer.input_layernorm.weight.?.data,
                .q = layer.self_attn.q_norm.weight.?.data[0..hd],
                .k = layer.self_attn.k_norm.weight.?.data[0..hd],
                .post_attn = layer.post_attention_layernorm.weight.?.data,
                .pre_mlp = layer.pre_feedforward_layernorm.weight.?.data,
                .post_mlp = layer.post_feedforward_layernorm.weight.?.data,
            };
        }

        /// 由已构建的投影权重组装 GPU 解码层；失败时释放投影权重
        fn assembleGpuLayer(layer: *const Gemma4Q4DecoderLayer, i: usize, projections: Gemma4GpuLayer.Projections) !Gemma4GpuLayer {
            var proj = projections;
            return Gemma4GpuLayer.init(config, i, proj, layerNormWeights(layer), layer.layer_scalar.data[0]) catch |err| {
                proj.deinit();
                return err;
            };
        }

        /// 从直通布局加载单层：CPU 侧仅保留层归一化参数，投影矩阵直接构建为 GPU 常驻量化权重
        fn loadDirectSingleLayer(
            allocator: std.mem.Allocator,
            mmap_bytes: []const u8,
            cursor_start: usize,
            i: usize,
            out_gpu: *Gemma4GpuLayer,
        ) !Gemma4Q4DecoderLayer {
            const spec = config.layerSpec(i);
            const d = ProjectionDims.of(config, i);
            const H = config.hidden_size;

            var layer = try Gemma4Q4DecoderLayer.initEmpty(
                allocator,
                spec.attn_type,
                H,
                config.intermediate_size,
                config.num_attention_heads,
                spec.num_kv_heads,
                spec.head_dim,
                config.sliding_window,
                config.rms_norm_eps,
                config.hidden_activation,
            );
            errdefer layer.deinit(allocator);
            loadLayerNorms(&layer, mmap_bytes[cursor_start..], spec.head_dim);

            var cursor = cursor_start + directLayerNormsBytes(i);
            const direct = Gemma4GpuLayer.DirectSlices{
                .qkv = takeDirectMatrix(mmap_bytes, &cursor, ((d.q_dim + d.kv_dim + d.v_dim) * H) / Q4Block.QK),
                .o = takeDirectMatrix(mmap_bytes, &cursor, (H * d.q_dim) / Q4Block.QK),
                .gate_up = takeDirectMatrix(mmap_bytes, &cursor, (config.intermediate_size * 2 * H) / Q4Block.QK),
                .down = takeDirectMatrix(mmap_bytes, &cursor, (H * config.intermediate_size) / Q4Block.QK),
            };
            out_gpu.* = try assembleGpuLayer(&layer, i, try Gemma4GpuLayer.Projections.fromDirect(config, i, direct));
            return layer;
        }

        /// 从 4-bit 量化二进制映射切片加载时的配置选项结构体
        pub const LoadMmapOptions = struct {
            /// 最终层 LM Head (分类头) 量化模式：
            /// 骨干 48 层始终为 Q4 (7.59 GB)，LM Head 默认采用 .q8 (1.07 GB) 保障 100% 精度一致性并降低 50% 显存；
            /// 也可指定为 .bf16 (2.01 GB) 或 .q4 (530 MB)。
            /// 量化模式与词表形状不兼容 (hidden_size 不整除分组长度) 时加载返回 `error.UnsupportedQuantMode`。
            lm_head_mode: backend.offload.QuantMode = .q8,
            /// 是否在加载阶段自动执行 Metal 计算管线预热，消除推理首字着色器编译延迟 (TTFT)
            warmup: bool = true,

            pub fn defaultOptions() LoadMmapOptions {
                return .{};
            }
        };

        /// 从 4-bit 量化二进制映射切片 (mmap_bytes) 零拷贝加载全部 48 层权重与层归一化参数
        /// 支持魔数 ZNNQ4G02 (分块量化格式) 与 ZNNQ4D02 (直通预重排格式)
        pub fn loadFromMmap(allocator: std.mem.Allocator, mmap_bytes: []const u8) !Self {
            return loadFromMmapWithOptions(allocator, mmap_bytes, .{});
        }

        /// 支持自定义选项从 4-bit 量化二进制映射切片 (mmap_bytes) 零拷贝加载全部 48 层权重与层归一化参数
        pub fn loadFromMmapWithOptions(
            allocator: std.mem.Allocator,
            mmap_bytes: []const u8,
            options: LoadMmapOptions,
        ) !Self {
            if (mmap_bytes.len < magic_bytes) return error.InvalidModelSignature;
            const is_direct = std.mem.eql(u8, mmap_bytes[0..magic_bytes], DIRECT_MAGIC);
            if (!is_direct and !std.mem.eql(u8, mmap_bytes[0..magic_bytes], Q4_MAGIC)) {
                return error.InvalidModelSignature;
            }
            // 直通布局仅保存 GPU 量化内核所需的重排权重，CPU 回退路径无法消费，必须具备硬件加速卸载能力
            if (is_direct and !backend.offload.isAvailable()) return error.DirectLayoutRequiresOffload;

            const expected_len = if (is_direct) computeExpectedDirectFileSize() else computeExpectedFileSize();
            if (mmap_bytes.len < expected_len) return error.IncompleteModelFile;
            try LmHeadWeights.validateMode(options.lm_head_mode, config.vocab_size, config.hidden_size);

            _ = std.posix.system.madvise(@ptrCast(@alignCast(@constCast(mmap_bytes.ptr))), mmap_bytes.len, std.posix.MADV.WILLNEED);

            // 1. 最终层 RMSNorm 权重 (BF16)
            const norm = try RMSNorm.init(allocator, config.hidden_size, config.rms_norm_eps);
            errdefer deinitModel(&norm, allocator);
            vec_math.widenBf16Bytes(norm.weight.?.data, mmap_bytes[magic_bytes..embed_offset]);

            // 2. 词表嵌入矩阵 (BF16) 保持在 mmap 与 GPU 统一显存中，
            // 避免在 CPU 堆内存上分配 4.03 GB 的冗余 F32 Tensor，自回归直接利用零拷贝映射
            const embed_tokens = try Embedding.init(allocator, 1, 1);
            errdefer deinitModel(&embed_tokens, allocator);
            const embed_u16_slice: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, mmap_bytes[embed_offset .. embed_offset + embed_bytes]));

            if (is_direct) {
                // 直通零拷贝极速初始化 (Direct Layout Fast Path, 1.5s 秒开)
                // GPU 上下文构建成功后接管 gpu_layers 与 LM Head，此后不再由本函数释放
                var gpu_owned_here = true;
                const gpu_layers = try allocator.alloc(Gemma4GpuLayer, config.num_hidden_layers);
                errdefer if (gpu_owned_here) allocator.free(gpu_layers);
                const layers = try allocator.alloc(Gemma4Q4DecoderLayer, config.num_hidden_layers);
                errdefer allocator.free(layers);
                var loaded: usize = 0;
                errdefer for (layers[0..loaded]) |*layer| layer.deinit(allocator);
                errdefer if (gpu_owned_here) {
                    for (gpu_layers[0..loaded]) |*gl| gl.deinit();
                };

                const offsets = computeDirectLayerOffsets();
                for (0..config.num_hidden_layers) |i| {
                    layers[i] = try loadDirectSingleLayer(allocator, mmap_bytes, offsets.offsets[i], i, &gpu_layers[i]);
                    loaded += 1;
                }

                // 创建 LM Head (直通布局下模型完全依赖 GPU 上下文，LM Head 构建失败必须向上传播)
                var lm_head_weights = try LmHeadWeights.initWithMode(
                    embed_u16_slice,
                    config.vocab_size,
                    config.hidden_size,
                    options.lm_head_mode,
                    null,
                );
                errdefer if (gpu_owned_here) lm_head_weights.deinit();

                // GPU 上下文成功构建后接管 gpu_layers (及其中各层权重) 与 LM Head 的所有权；失败时由上方 errdefer 释放
                const gpu_context = try Gemma4GpuContext.init(
                    allocator,
                    config,
                    gpu_layers,
                    norm.weight.?.data,
                    lm_head_weights,
                    config.max_position_embeddings,
                );
                gpu_owned_here = false;
                var res = Self{
                    .embed_tokens = embed_tokens,
                    .embed_u16_mmap = embed_u16_slice,
                    .mmap_bytes = mmap_bytes,
                    .layers = layers,
                    .norm = norm,
                    .gpu_context = gpu_context,
                };

                // 预热只是可选的着色器预编译，失败不影响推理正确性
                if (options.warmup) {
                    res.gpu_context.?.warmup(embed_u16_slice, null) catch {};
                }
                return res;
            }

            // 3. 逐层零拷贝绑定 48 层 Transformer 的 Q4Block 分块数组 (空结构 + mmap 绑定，不分配权重缓冲区)
            const layers = try allocator.alloc(Gemma4Q4DecoderLayer, config.num_hidden_layers);
            errdefer allocator.free(layers);
            var loaded: usize = 0;
            errdefer for (layers[0..loaded]) |*layer| layer.deinit(allocator);

            const offsets = computeLayerOffsets();
            for (0..config.num_hidden_layers) |i| {
                const spec = config.layerSpec(i);
                var layer = try Gemma4Q4DecoderLayer.initEmpty(
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
                );
                loadLayerNorms(&layer, mmap_bytes[offsets.offsets[i]..], spec.head_dim);

                var cursor = offsets.offsets[i] + layerNormsBytes(i);
                layer.self_attn.q_proj.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                layer.self_attn.k_proj.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                if (layer.self_attn.v_proj) |*vp| vp.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                layer.self_attn.o_proj.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                layer.mlp.gate_proj.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                layer.mlp.up_proj.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                layer.mlp.down_proj.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                std.debug.assert(cursor == offsets.offsets[i + 1]);

                layers[i] = layer;
                loaded += 1;
            }

            // 量化模式已在上方校验；GPU 资源不足等运行期失败时 LM Head 不可用，sampleNextToken 自动回退至 CPU 投影路径
            var lm_head_weights: ?LmHeadWeights = null;
            if (backend.offload.isAvailable()) {
                lm_head_weights = LmHeadWeights.initWithMode(
                    embed_u16_slice,
                    config.vocab_size,
                    config.hidden_size,
                    options.lm_head_mode,
                    null,
                ) catch null;
            }

            var res = Self{
                .embed_tokens = embed_tokens,
                .embed_u16_mmap = embed_u16_slice,
                .mmap_bytes = mmap_bytes,
                .layers = layers,
                .norm = norm,
                .lm_head_weights = lm_head_weights,
            };
            if (lm_head_weights != null) res.initGpuContext(allocator) catch {};

            if (options.warmup and res.gpu_context != null) {
                res.gpu_context.?.warmup(embed_u16_slice, null) catch {};
            }

            if (res.gpu_context != null) {
                // 通知操作系统内核丢弃已卸载至 GPU 显存的 4-bit 权重文件页缓存，防止 7.59 GB 物理内存重复驻留造成 macOS Swap 严重换页
                const page_size: usize = std.heap.page_size_min;
                const base_addr = @intFromPtr(mmap_bytes.ptr);
                const aligned_start_addr = std.mem.alignForward(usize, base_addr + layers_offset, page_size);
                const aligned_end_addr = std.mem.alignBackward(usize, base_addr + mmap_bytes.len, page_size);
                if (aligned_end_addr > aligned_start_addr) {
                    const ptr: *align(std.heap.page_size_min) anyopaque = @ptrFromInt(aligned_start_addr);
                    _ = std.posix.system.madvise(ptr, aligned_end_addr - aligned_start_addr, std.posix.MADV.DONTNEED);
                }
            }

            return res;
        }

        fn initSingleGpuLayer(allocator: std.mem.Allocator, layer: *const Gemma4Q4DecoderLayer, i: usize) !Gemma4GpuLayer {
            const blocks = Gemma4GpuLayer.BlockSlices{
                .q = std.mem.sliceAsBytes(layer.self_attn.q_proj.blocks),
                .k = std.mem.sliceAsBytes(layer.self_attn.k_proj.blocks),
                .v = if (layer.self_attn.v_proj) |vp| std.mem.sliceAsBytes(vp.blocks) else null,
                .o = std.mem.sliceAsBytes(layer.self_attn.o_proj.blocks),
                .gate = std.mem.sliceAsBytes(layer.mlp.gate_proj.blocks),
                .up = std.mem.sliceAsBytes(layer.mlp.up_proj.blocks),
                .down = std.mem.sliceAsBytes(layer.mlp.down_proj.blocks),
            };
            return assembleGpuLayer(layer, i, try Gemma4GpuLayer.Projections.fromBlocks(allocator, config, i, blocks));
        }

        /// 为全模型所有 4-bit 量化层构建常驻 GPU 显存解码上下文 (GPU Model Context)。
        /// 已存在上下文时直接返回；构建成功后独立持有的 LM Head 所有权转移至上下文，失败时仍由模型持有。
        pub fn initGpuContext(self: *Self, allocator: std.mem.Allocator) !void {
            if (self.gpu_context != null) return;
            const lm_head = self.lm_head_weights orelse return error.LmHeadUnavailable;

            const gpu_layers = try allocator.alloc(Gemma4GpuLayer, self.layers.len);
            errdefer allocator.free(gpu_layers);
            var init_idx: usize = 0;
            errdefer {
                for (gpu_layers[0..init_idx]) |*gl| {
                    gl.deinit();
                }
            }

            const cpu_count = @max(1, std.Thread.getCpuCount() catch 1);
            const num_threads = @min(cpu_count, self.layers.len);

            if (num_threads <= 1 or self.layers.len <= 2) {
                for (self.layers, 0..) |*layer, i| {
                    gpu_layers[i] = try initSingleGpuLayer(allocator, layer, i);
                    init_idx += 1;
                }
            } else {
                const Worker = struct {
                    model: *Self,
                    alloc: std.mem.Allocator,
                    dest: []Gemma4GpuLayer,
                    start: usize,
                    end: usize,
                    /// 已成功构建的层数 (dest[start .. start + done])，出错时用于精确回收
                    done: usize = 0,
                    err: ?anyerror = null,

                    fn run(ctx: *@This()) void {
                        for (ctx.start..ctx.end) |i| {
                            ctx.dest[i] = initSingleGpuLayer(ctx.alloc, &ctx.model.layers[i], i) catch |err| {
                                ctx.err = err;
                                return;
                            };
                            ctx.done += 1;
                        }
                    }
                };

                var workers = try allocator.alloc(Worker, num_threads);
                defer allocator.free(workers);
                var threads = try allocator.alloc(std.Thread, num_threads);
                defer allocator.free(threads);

                const chunk = (self.layers.len + num_threads - 1) / num_threads;
                var spawned: usize = 0;

                for (0..num_threads) |t_idx| {
                    const s = t_idx * chunk;
                    const e = @min(s + chunk, self.layers.len);
                    if (s >= e) {
                        workers[t_idx] = .{ .model = self, .alloc = allocator, .dest = gpu_layers, .start = 0, .end = 0 };
                        continue;
                    }
                    workers[t_idx] = .{ .model = self, .alloc = allocator, .dest = gpu_layers, .start = s, .end = e };
                    threads[spawned] = std.Thread.spawn(.{}, Worker.run, .{&workers[t_idx]}) catch {
                        workers[t_idx].run();
                        continue;
                    };
                    spawned += 1;
                }

                for (threads[0..spawned]) |t| {
                    t.join();
                }

                var first_err: ?anyerror = null;
                for (workers[0..num_threads]) |w| {
                    if (w.err) |err| first_err = first_err orelse err;
                }
                if (first_err) |err| {
                    for (workers[0..num_threads]) |w| {
                        for (gpu_layers[w.start .. w.start + w.done]) |*gl| gl.deinit();
                    }
                    return err;
                }
                init_idx = self.layers.len;
            }

            self.gpu_context = try Gemma4GpuContext.init(
                allocator,
                config,
                gpu_layers,
                self.norm.weight.?.data,
                lm_head,
                config.max_position_embeddings,
            );
            // LM Head 所有权已转移至 GPU 上下文
            self.lm_head_weights = null;
            self.kv_owner = .cpu;
        }

        /// 为全模型所有 4-bit 量化层启用图形处理器 (Graphics Processing Unit, GPU) 硬件加速卸载
        pub fn enableHardwareOffload(self: *Self, allocator: std.mem.Allocator) !void {
            if (!backend.offload.isAvailable()) return;
            try self.initGpuContext(allocator);
        }

        /// 对最终隐状态 h_last 执行词表投影并按 sampler 选取下一个词元：截断后惩罚 (仅最近 repetition_window 个词元受罚)，
        /// 排除 `config.sampling_excluded_tokens`，贪婪采样或温度 / 前 K 截断 (Top-K) / 核采样 (Nucleus Sampling, Top-P) 随机采样
        fn sampleNextToken(
            self: *const Self,
            h_last: []const f32,
            embed_u16: []const u16,
            recent_tokens: []const usize,
            sampler: *Sampler,
            graph: *autodiff.Graph,
        ) !usize {
            const recent = gemma4.recentWindowFor(recent_tokens, sampler.options);
            // 1. 优先通过 GPU 常驻 LM Head 执行巨型词表投影与采样 (消除 2.3 秒延迟至 5~10 ms)；GPU 执行失败时回退 CPU。
            //    GPU 路径只在成功读回结果之后才消耗随机数，回退不会打乱采样器的随机序列
            if (graph.options.device.isGpu() and backend.offload.isAvailable()) {
                if (self.sampleNextTokenGpu(h_last, embed_u16, recent, sampler, graph)) |tok| {
                    return tok;
                } else |_| {}
            }

            // 2. CPU 确定性多核并行回退路径
            return gemma4.sampleLmHeadCpu(config, u16, graph, embed_u16, h_last, recent, sampler);
        }

        fn sampleNextTokenGpu(
            self: *const Self,
            h_last: []const f32,
            embed_u16: []const u16,
            recent_tokens: []const usize,
            sampler: *Sampler,
            graph: *autodiff.Graph,
        ) !usize {
            const stream = graph.options.mlx_stream;
            if (self.gpu_context) |*ctx| return ctx.sampleHidden(h_last, recent_tokens, sampler, stream);

            const arena = graph.arenaAllocator();
            var mask = try TokenExclusionMask.init(arena, config.vocab_size, config.sampling_excluded_tokens);
            defer mask.deinit();
            if (self.lm_head_weights) |*w| {
                return selectTokenFromHidden(arena, w, &mask, config, h_last, recent_tokens, sampler, stream);
            }
            // 未构建常驻 LM Head 时临时构建一次 BF16 词表矩阵
            var tmp = try LmHeadWeights.init(embed_u16, config.vocab_size, config.hidden_size, stream);
            defer tmp.deinit();
            return selectTokenFromHidden(arena, &tmp, &mask, config, h_last, recent_tokens, sampler, stream);
        }

        /// 从模型映射字节中切出 BF16 词表嵌入矩阵 [vocab_size, hidden_size]
        fn embedSliceFromMmap(mmap_bytes: []const u8) ![]const u16 {
            if (mmap_bytes.len < embed_offset + embed_bytes) return error.IncompleteModelFile;
            return @alignCast(std.mem.bytesAsSlice(u16, mmap_bytes[embed_offset .. embed_offset + embed_bytes]));
        }

        /// 无键值缓存的自回归前向推理单步：输入完整历史 token_ids 序列与 mmap 字节，前向传播后按 sampler 选取下一个词元。
        /// 词表投影通过 graph.executor() 并行，不派生临时 OS 线程
        pub fn generateNextToken(
            self: *const Self,
            graph: *autodiff.Graph,
            token_ids: []const usize,
            mmap_bytes: []const u8,
            sampler: *Sampler,
        ) !usize {
            const T = token_ids.len;
            if (T == 0) return error.EmptyPrompt;

            const embed_u16 = try embedSliceFromMmap(mmap_bytes);

            // 1. 词表嵌入查表
            const tok_emb = try gemma4.embedTokensScaled(u16, graph, embed_u16, config.hidden_size, token_ids);

            // 2. 逐层前向传播
            var h = tok_emb;
            for (self.layers) |*layer| {
                h = try layer.forward(graph, h);
            }

            // 3. 最终层 RMSNorm
            const h_norm = try self.norm.forward(graph, h);

            // 4. 获取最后一个位置 T-1 的隐状态向量 [config.hidden_size]
            const last_row = h_norm.data[(T - 1) * config.hidden_size .. T * config.hidden_size];

            // 5. 词表投影与采样 (CPU 路径为确定性多核并行，零临时 OS 线程派生)
            return self.sampleNextToken(last_row, embed_u16, token_ids, sampler, graph);
        }

        /// 预填充阶段 (Prefill)：处理 Prompt 词元序列，写入键值缓存 (Key-Value Cache, KVCache)，并按 sampler 返回首个预测词元
        pub fn prefill(
            self: *Self,
            graph: *autodiff.Graph,
            prompt_tokens: []const usize,
            cache: *Gemma4KVCache,
            mmap_bytes: []const u8,
            sampler: *Sampler,
        ) !usize {
            const T = prompt_tokens.len;
            if (T == 0) return error.EmptyPrompt;
            try tensor.checkIndices(prompt_tokens, config.vocab_size);
            if (T > cache.max_seq_len) return error.ContextWindowExceeded;
            cache.reset();

            const embed_u16 = try embedSliceFromMmap(mmap_bytes);

            if (graph.options.device.isGpu() and self.gpu_context != null) {
                const next_tok = try self.gpu_context.?.prefill(prompt_tokens, embed_u16, sampler, graph.options.mlx_stream);
                // 键值内容只存在于 GPU：CPU 层缓存仅推进游标，切回 CPU 解码前由 syncCpuCache 写回内容
                self.kv_owner = .gpu;
                for (cache.layers) |*cl| {
                    cl.core.curr_len = T;
                }
                return next_tok;
            }

            self.kv_owner = .cpu;

            // 1. 词表嵌入查表
            const tok_emb = try gemma4.embedTokensScaled(u16, graph, embed_u16, config.hidden_size, prompt_tokens);

            // 2. 逐层预填充前向传播并写入键值缓存
            var h = tok_emb;
            for (self.layers, 0..) |*layer, l_idx| {
                h = try layer.forwardPrefill(graph, h, &cache.layers[l_idx]);
            }

            // 3. 最终层 RMSNorm
            const h_norm = try self.norm.forward(graph, h);
            const last_row = h_norm.data[(T - 1) * config.hidden_size .. T * config.hidden_size];

            return self.sampleNextToken(last_row, embed_u16, prompt_tokens, sampler, graph);
        }

        /// GPU 解码前确保 GPU 键值缓存持有最新内容：上一次推进发生在 CPU 时先从 CPU 层缓存同步过去
        fn syncGpuCache(self: *Self, cache: *const Gemma4KVCache) !*Gemma4GpuContext {
            const gpu_ctx = &self.gpu_context.?;
            if (self.kv_owner == .cpu) try gpu_ctx.syncCacheFromCpu(cache);
            self.kv_owner = .gpu;
            return gpu_ctx;
        }

        /// CPU 解码前确保 CPU 层缓存持有最新内容：上一次推进发生在 GPU 时先把 GPU 键值缓存写回 CPU
        fn syncCpuCache(self: *Self, cache: *Gemma4KVCache) !void {
            if (self.kv_owner == .gpu) {
                const gpu_ctx = if (self.gpu_context) |*ctx| ctx else return error.KvCacheUnavailable;
                try gpu_ctx.syncCacheToCpu(cache);
            }
            self.kv_owner = .cpu;
        }

        /// 增量解码阶段 (Decode Step)：输入最新生成的单个词元 ID，基于键值缓存 (Key-Value Cache, KVCache) 高效自回归生成下一个词元
        pub fn decodeStep(
            self: *Self,
            graph: *autodiff.Graph,
            token: usize,
            cache: *Gemma4KVCache,
            mmap_bytes: []const u8,
            recent_tokens: []const usize,
            sampler: *Sampler,
        ) !usize {
            const tokens = [_]usize{token};
            try tensor.checkIndices(&tokens, config.vocab_size);
            const embed_u16 = try embedSliceFromMmap(mmap_bytes);

            if (graph.options.device.isGpu() and self.gpu_context != null) {
                const gpu_ctx = try self.syncGpuCache(cache);
                const next_tok = try gpu_ctx.decodeStep(token, embed_u16, recent_tokens, sampler, graph.options.mlx_stream);
                for (cache.layers) |*cl| {
                    cl.core.curr_len += 1;
                }
                return next_tok;
            }
            try self.syncCpuCache(cache);

            // 1. 单个词元嵌入查表 [1, 1, hidden_size]
            const tok_emb = try gemma4.embedTokensScaled(u16, graph, embed_u16, config.hidden_size, &tokens);

            // 2. 逐层增量解码传播
            var h = tok_emb;
            for (self.layers, 0..) |*layer, l_idx| {
                h = try layer.forwardDecode(graph, h, &cache.layers[l_idx]);
            }

            // 3. 最终层 RMSNorm
            const h_norm = try self.norm.forward(graph, h);
            const last_row = h_norm.data[0..config.hidden_size];

            return self.sampleNextToken(last_row, embed_u16, recent_tokens, sampler, graph);
        }

        /// 投机采样批量验证解码 (Speculative Draft Verification Step)
        /// 在 GPU 统一卸载加速下，一次前向传播批量验证候选词元序列，突破单字显存带宽高墙；
        /// 贪婪采样按精确匹配接受草稿，随机采样按标准投机采样规则接受 (见 `Gemma4GpuContext.decodeSpeculative`)
        pub fn decodeSpeculative(
            self: *Self,
            graph: *autodiff.Graph,
            input_tokens: []const usize,
            cache: *Gemma4KVCache,
            mmap_bytes: []const u8,
            recent_tokens: []const usize,
            sampler: *Sampler,
            accepted_buf: []usize,
        ) !usize {
            if (input_tokens.len == 0) return 0;
            if (accepted_buf.len == 0) return error.BufferTooSmall;
            try tensor.checkIndices(input_tokens, config.vocab_size);
            const embed_u16 = try embedSliceFromMmap(mmap_bytes);

            if (graph.options.device.isGpu() and self.gpu_context != null) {
                const gpu_ctx = try self.syncGpuCache(cache);
                const accepted_count = try gpu_ctx.decodeSpeculative(
                    input_tokens,
                    embed_u16,
                    gemma4.recentWindowFor(recent_tokens, sampler.options),
                    sampler,
                    graph.options.mlx_stream,
                    accepted_buf,
                );
                for (cache.layers) |*cl| {
                    cl.core.curr_len += accepted_count;
                }
                return accepted_count;
            }

            // CPU 单步回退保底：对第 1 个词元执行标准 decodeStep
            const single_tok = try self.decodeStep(graph, input_tokens[0], cache, mmap_bytes, recent_tokens, sampler);
            accepted_buf[0] = single_tok;
            return 1;
        }

        pub const GenerateOptions = struct {
            sampling: SamplingOptions = config.generation,
            max_new_tokens: usize = 256,
            stop_tokens: []const usize = &default_stop_tokens,
            enable_speculative: bool = true,
            max_seq_len: usize = 2048,
            mmap_bytes: ?[]const u8 = null,

            pub fn defaultOptions() GenerateOptions {
                return .{};
            }
        };

        pub const default_stop_tokens = [_]usize{ gemma4.Gemma4Vocabulary.EOS, gemma4.Gemma4Vocabulary.EOT };

        /// 开始一次自回归生成会话：统一封装预填充、键值缓存、投机验证与流式生成循环
        pub fn generate(
            self: *Self,
            allocator: std.mem.Allocator,
            prompt_tokens: []const usize,
            options: GenerateOptions,
        ) !Generation {
            if (prompt_tokens.len == 0) return error.EmptyPrompt;
            const mmap_slice = options.mmap_bytes orelse (self.mmap_bytes orelse return error.MissingMmapBytes);

            var history: std.ArrayList(usize) = .empty;
            errdefer history.deinit(allocator);
            try history.appendSlice(allocator, prompt_tokens);

            var cache = try gemma4.Gemma4KVCache.init(allocator, config, .{ .max_seq_len = options.max_seq_len });
            errdefer cache.deinit(allocator);

            const use_gpu = (self.gpu_context != null and backend.offload.isAvailable());
            var graph = autodiff.Graph.init(allocator, .{
                .enable_grad = false,
                .device = if (use_gpu) .metal else .cpu,
            });
            errdefer graph.deinit();

            var draft_engine: ?nn.SpeculativeDraftEngine = null;
            if (options.enable_speculative and use_gpu) {
                var de = nn.SpeculativeDraftEngine.initWithStopTokens(allocator, options.stop_tokens);
                de.seedFromTokens(prompt_tokens) catch {};
                draft_engine = de;
            }

            return Generation{
                .model = self,
                .allocator = allocator,
                .options = options,
                .sampler = Sampler.init(options.sampling),
                .cache = cache,
                .graph = graph,
                .draft_engine = draft_engine,
                .history = history,
                .prompt_len = prompt_tokens.len,
                .mmap_bytes = mmap_slice,
            };
        }

        /// 自回归生成会话：`next` 逐个返回生成的词元，生成结束 (遇到停止词元或达到 `max_new_tokens`) 后返回 null
        pub const Generation = struct {
            model: *Self,
            allocator: std.mem.Allocator,
            options: GenerateOptions,
            sampler: Sampler,
            cache: gemma4.Gemma4KVCache,
            graph: autodiff.Graph,
            draft_engine: ?nn.SpeculativeDraftEngine = null,
            history: std.ArrayList(usize),
            prompt_len: usize,
            started: bool = false,
            finished: bool = false,
            mmap_bytes: []const u8,

            accepted_buf: [16]usize = undefined,
            accepted_len: usize = 0,
            accepted_pos: usize = 0,
            speculative_hits: usize = 0,

            pub fn deinit(self: *Generation) void {
                if (self.draft_engine) |*de| de.deinit();
                self.cache.deinit(self.allocator);
                self.graph.deinit();
                self.history.deinit(self.allocator);
            }

            pub fn generated(self: *const Generation) []const usize {
                return self.history.items[self.prompt_len..];
            }

            pub fn tokens(self: *const Generation) []const usize {
                return self.history.items;
            }

            fn isStopToken(self: *const Generation, tok: usize) bool {
                for (self.options.stop_tokens) |st| {
                    if (tok == st) return true;
                }
                return false;
            }

            fn emit(self: *Generation, tok: usize) !?usize {
                try self.history.append(self.allocator, tok);
                if (self.isStopToken(tok) or self.history.items.len - self.prompt_len >= self.options.max_new_tokens) {
                    self.finished = true;
                }
                return tok;
            }

            pub fn next(self: *Generation) !?usize {
                if (self.finished) return null;
                if (self.options.max_new_tokens == 0) {
                    self.finished = true;
                    return null;
                }

                if (self.accepted_pos < self.accepted_len) {
                    const tok = self.accepted_buf[self.accepted_pos];
                    self.accepted_pos += 1;
                    return try self.emit(tok);
                }

                if (!self.started) {
                    self.graph.reset();
                    const first_tok = try self.model.prefill(
                        &self.graph,
                        self.history.items,
                        &self.cache,
                        self.mmap_bytes,
                        &self.sampler,
                    );
                    self.started = true;
                    if (self.draft_engine) |*de| {
                        const prev = self.history.items[self.history.items.len - 1];
                        de.recordTransition(prev, first_tok) catch {};
                    }
                    return try self.emit(first_tok);
                }

                self.graph.reset();
                const current_tok = self.history.items[self.history.items.len - 1];

                if (self.draft_engine) |*de| {
                    var cand_buf: [4]usize = undefined;
                    const draft = de.proposeDraft(self.history.items, cand_buf[0..3]);
                    if (draft) |cands| {
                        var input_spec: [8]usize = undefined;
                        input_spec[0] = current_tok;
                        const spec_k = @min(cands.len, 3);
                        for (0..spec_k) |idx| {
                            input_spec[idx + 1] = cands[idx];
                        }
                        const spec_total = 1 + spec_k;
                        var acc_buf: [8]usize = undefined;
                        const num_acc = try self.model.decodeSpeculative(
                            &self.graph,
                            input_spec[0..spec_total],
                            &self.cache,
                            self.mmap_bytes,
                            self.history.items,
                            &self.sampler,
                            acc_buf[0..spec_total],
                        );

                        if (num_acc > 0) {
                            if (num_acc > 1) {
                                self.speculative_hits += (num_acc - 1);
                            }
                            var prev = current_tok;
                            for (0..num_acc) |acc_i| {
                                const tok = acc_buf[acc_i];
                                de.recordTransition(prev, tok) catch {};
                                prev = tok;
                                self.accepted_buf[acc_i] = tok;
                            }
                            self.accepted_len = num_acc;
                            self.accepted_pos = 1;
                            return try self.emit(acc_buf[0]);
                        }
                    }
                }

                const next_tok = try self.model.decodeStep(
                    &self.graph,
                    current_tok,
                    &self.cache,
                    self.mmap_bytes,
                    self.history.items,
                    &self.sampler,
                );
                if (self.draft_engine) |*de| {
                    de.recordTransition(current_tok, next_tok) catch {};
                }
                return try self.emit(next_tok);
            }
        };
    };
}

/// 默认配置的 Gemma 4 12B 4-bit 量化模型类型
pub const DefaultGemma4Q4 = Gemma4Q4ForCausalLM(Gemma4Config.default);

/// 测试配置的微型 Q4 量化 Gemma 4 模型类型
pub const TinyQ4Gemma4 = Gemma4Q4ForCausalLM(Gemma4Config.tiny_test);

// ============================================================================
// Gemma 4 12B GPU 硬件加速层与整网自回归解码上下文
// ============================================================================

pub fn recentWindowFor(tokens: []const usize, opts: SamplingOptions) []const usize {
    const window = opts.repetition_window orelse return tokens;
    if (window == 0 or tokens.len <= window) return tokens;
    return tokens[tokens.len - window ..];
}

fn processorOpts(cfg: anytype) transformer_blocks.LogitsProcessorOptions {
    return .{
        .vocab_size = cfg.vocab_size,
        .final_logit_softcapping = cfg.final_logit_softcapping,
        .sampling_excluded_tokens = cfg.sampling_excluded_tokens,
    };
}

pub fn deviceCandidates(
    allocator: std.mem.Allocator,
    logits: offload.Array,
    cfg: anytype,
    mask: *const TokenExclusionMask,
    recent_tokens: []const usize,
    options: SamplingOptions,
    s: ?Stream,
) !DeviceCandidates {
    return transformer_blocks.deviceCandidates(allocator, logits, processorOpts(cfg), mask, recent_tokens, options, s);
}

pub fn selectToken(
    allocator: std.mem.Allocator,
    logits: offload.Array,
    cfg: anytype,
    mask: *const TokenExclusionMask,
    recent_tokens: []const usize,
    sampler: anytype,
    s: ?Stream,
) !usize {
    return transformer_blocks.selectToken(allocator, logits, processorOpts(cfg), mask, recent_tokens, sampler, s);
}

pub fn selectTokenFromHidden(
    allocator: std.mem.Allocator,
    lm_head: anytype,
    mask: *const TokenExclusionMask,
    cfg: anytype,
    h_last: []const f32,
    recent_tokens: []const usize,
    sampler: anytype,
    s: ?Stream,
) !usize {
    return transformer_blocks.selectTokenFromHidden(allocator, lm_head, processorOpts(cfg), mask, h_last, recent_tokens, sampler, s);
}

/// 第 layer_idx 层注意力投影的输出维度：全局注意力层 attention_k_eq_v 为 true，不存储独立的 V 投影 (v_dim = 0)
pub const ProjectionDims = struct {
    q_dim: usize,
    kv_dim: usize,
    v_dim: usize,

    pub fn of(cfg: anytype, layer_idx: usize) ProjectionDims {
        const spec = cfg.layerSpec(layer_idx);
        const kv_dim = spec.num_kv_heads * spec.head_dim;
        const is_full = @backingInt(spec.attn_type) == @backingInt(Gemma4AttentionType.full_attention);
        return .{
            .q_dim = cfg.num_attention_heads * spec.head_dim,
            .kv_dim = kv_dim,
            .v_dim = if (is_full) 0 else kv_dim,
        };
    }
};

/// 图形处理器 (Graphics Processing Unit, GPU) 加速解码单个 Gemma 4 Transformer 层的常驻权重与超参数
pub const Gemma4GpuLayer = struct {
    fused_qkv: Q4Weights,
    o_proj: Q4Weights,
    fused_gate_up: Q4Weights,
    down_proj: Q4Weights,

    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    intermediate_size: usize,
    attn_dim: usize,

    input_norm_w: offload.Array,
    q_norm_w: offload.Array,
    k_norm_w: offload.Array,
    post_attn_w: offload.Array,
    pre_mlp_w: offload.Array,
    post_mlp_w: offload.Array,

    layer_scalar: f32,
    attn_type: Gemma4AttentionType,
    sliding_window: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: GpuRope,
    rms_norm_eps: f32,
    hidden_activation: backend.activation.GeluApproximation,
    compiled_mlp: ?offload.Closure = null,

    pub const Projections = struct {
        fused_qkv: Q4Weights,
        o_proj: Q4Weights,
        fused_gate_up: Q4Weights,
        down_proj: Q4Weights,

        pub fn deinit(self: *Projections) void {
            self.fused_qkv.deinit();
            self.o_proj.deinit();
            self.fused_gate_up.deinit();
            self.down_proj.deinit();
        }

        pub fn fromBlocks(allocator: std.mem.Allocator, cfg: anytype, layer_idx: usize, blocks: BlockSlices) !Projections {
            const d = ProjectionDims.of(cfg, layer_idx);
            const H = cfg.hidden_size;
            if ((blocks.v != null) != (d.v_dim != 0)) return error.ShapeMismatch;

            var fused_qkv = if (blocks.v) |vb|
                try Q4Weights.initFromBlockSlices(allocator, &.{ blocks.q, blocks.k, vb }, H, &.{ d.q_dim, d.kv_dim, d.v_dim }, null)
            else
                try Q4Weights.initFromBlockSlices(allocator, &.{ blocks.q, blocks.k }, H, &.{ d.q_dim, d.kv_dim }, null);
            errdefer fused_qkv.deinit();
            var o_proj = try Q4Weights.initFromBlocks(allocator, blocks.o, d.q_dim, H, null);
            errdefer o_proj.deinit();
            var fused_gate_up = try Q4Weights.initFromBlockSlices(
                allocator,
                &.{ blocks.gate, blocks.up },
                H,
                &.{ cfg.intermediate_size, cfg.intermediate_size },
                null,
            );
            errdefer fused_gate_up.deinit();
            const down_proj = try Q4Weights.initFromBlocks(allocator, blocks.down, cfg.intermediate_size, H, null);
            return .{ .fused_qkv = fused_qkv, .o_proj = o_proj, .fused_gate_up = fused_gate_up, .down_proj = down_proj };
        }

        pub fn fromDirect(cfg: anytype, layer_idx: usize, direct: DirectSlices) !Projections {
            const d = ProjectionDims.of(cfg, layer_idx);
            const H = cfg.hidden_size;
            var fused_qkv = try Q4Weights.initFromDirectBuffers(direct.qkv.wq, direct.qkv.sc, direct.qkv.bi, H, d.q_dim + d.kv_dim + d.v_dim, null);
            errdefer fused_qkv.deinit();
            var o_proj = try Q4Weights.initFromDirectBuffers(direct.o.wq, direct.o.sc, direct.o.bi, d.q_dim, H, null);
            errdefer o_proj.deinit();
            var fused_gate_up = try Q4Weights.initFromDirectBuffers(direct.gate_up.wq, direct.gate_up.sc, direct.gate_up.bi, H, cfg.intermediate_size * 2, null);
            errdefer fused_gate_up.deinit();
            const down_proj = try Q4Weights.initFromDirectBuffers(direct.down.wq, direct.down.sc, direct.down.bi, cfg.intermediate_size, H, null);
            return .{ .fused_qkv = fused_qkv, .o_proj = o_proj, .fused_gate_up = fused_gate_up, .down_proj = down_proj };
        }
    };

    pub const BlockSlices = struct {
        q: []const u8,
        k: []const u8,
        v: ?[]const u8,
        o: []const u8,
        gate: []const u8,
        up: []const u8,
        down: []const u8,
    };

    pub const DirectMatrix = struct {
        wq: []const u8,
        sc: []const u8,
        bi: []const u8,
    };

    pub const DirectSlices = struct {
        qkv: DirectMatrix,
        o: DirectMatrix,
        gate_up: DirectMatrix,
        down: DirectMatrix,
    };

    pub const NormWeights = struct {
        input: []const f32,
        q: []const f32,
        k: []const f32,
        post_attn: []const f32,
        pre_mlp: []const f32,
        post_mlp: []const f32,
    };

    pub fn init(
        cfg: anytype,
        layer_idx: usize,
        projections: Projections,
        norms: NormWeights,
        layer_scalar: f32,
    ) !Gemma4GpuLayer {
        const spec = cfg.layerSpec(layer_idx);
        const d = ProjectionDims.of(cfg, layer_idx);
        if (norms.input.len != cfg.hidden_size or norms.post_attn.len != cfg.hidden_size or
            norms.pre_mlp.len != cfg.hidden_size or norms.post_mlp.len != cfg.hidden_size or
            norms.q.len != spec.head_dim or norms.k.len != spec.head_dim)
        {
            return error.ShapeMismatch;
        }

        var in_w = try offload.Array.fromSlice(norms.input, &.{norms.input.len});
        errdefer in_w.deinit();
        var qn_w = try offload.Array.fromSlice(norms.q, &.{norms.q.len});
        errdefer qn_w.deinit();
        var kn_w = try offload.Array.fromSlice(norms.k, &.{norms.k.len});
        errdefer kn_w.deinit();
        var pa_w = try offload.Array.fromSlice(norms.post_attn, &.{norms.post_attn.len});
        errdefer pa_w.deinit();
        var pm_w = try offload.Array.fromSlice(norms.pre_mlp, &.{norms.pre_mlp.len});
        errdefer pm_w.deinit();
        var pom_w = try offload.Array.fromSlice(norms.post_mlp, &.{norms.post_mlp.len});
        errdefer pom_w.deinit();
        const is_full = @backingInt(spec.attn_type) == @backingInt(Gemma4AttentionType.full_attention);
        const attn_type: Gemma4AttentionType = if (is_full) .full_attention else .sliding_attention;
        const rope_inst = try GpuRope.init(spec.head_dim, attn_type.ropeOptions());

        return .{
            .fused_qkv = projections.fused_qkv,
            .o_proj = projections.o_proj,
            .fused_gate_up = projections.fused_gate_up,
            .down_proj = projections.down_proj,
            .q_dim = d.q_dim,
            .k_dim = d.kv_dim,
            .v_dim = d.v_dim,
            .intermediate_size = cfg.intermediate_size,
            .attn_dim = d.q_dim,
            .input_norm_w = in_w,
            .q_norm_w = qn_w,
            .k_norm_w = kn_w,
            .post_attn_w = pa_w,
            .pre_mlp_w = pm_w,
            .post_mlp_w = pom_w,
            .layer_scalar = layer_scalar,
            .attn_type = attn_type,
            .sliding_window = cfg.sliding_window,
            .num_heads = cfg.num_attention_heads,
            .num_kv_heads = spec.num_kv_heads,
            .head_dim = spec.head_dim,
            .rope = rope_inst,
            .rms_norm_eps = cfg.rms_norm_eps,
            .hidden_activation = cfg.hidden_activation,
            .compiled_mlp = null,
        };
    }

    pub fn mlpResidual(self: *const Gemma4GpuLayer, x1: offload.Array, stream: ?Stream) !offload.Array {
        var post_mlp = try transformer_blocks.gatedMlp(
            x1,
            &self.fused_gate_up,
            null,
            &self.down_proj,
            self.hidden_activation,
            self.pre_mlp_w,
            self.post_mlp_w,
            self.rms_norm_eps,
            stream,
        );
        defer post_mlp.deinit();
        var x2 = try transformer_blocks.residualScale(x1, post_mlp, null, stream);
        if (self.layer_scalar == 1.0) return x2;
        defer x2.deinit();
        return x2.mulScalar(self.layer_scalar, stream);
    }

    fn runMlp(self: *const Gemma4GpuLayer, x1: offload.Array, stream: ?Stream) !offload.Array {
        if (self.compiled_mlp) |c_mlp| {
            var in_vec = offload.VectorArray.fromArrays(&.{x1});
            defer in_vec.deinit();
            var out_vec = try c_mlp.apply(in_vec);
            defer out_vec.deinit();
            return out_vec.get(0) orelse error.MlxOperationFailed;
        }
        return self.mlpResidual(x1, stream);
    }

    pub fn initCompiledMlp(self: *Gemma4GpuLayer) !void {
        if (!offload.isAvailable()) return;
        if (self.compiled_mlp != null) return;

        const mlpRun = struct {
            fn run(lw: *const Gemma4GpuLayer, in: [1]offload.Array) anyerror!offload.Array {
                return lw.mlpResidual(in[0], null);
            }
        }.run;

        self.compiled_mlp = try offload.compileFn(1, 1, Gemma4GpuLayer, mlpRun, false, self);
    }

    pub fn deinit(self: *Gemma4GpuLayer) void {
        if (self.compiled_mlp) |*c_cls| {
            c_cls.deinit();
            self.compiled_mlp = null;
        }
        self.fused_qkv.deinit();
        self.o_proj.deinit();
        self.fused_gate_up.deinit();
        self.down_proj.deinit();
        self.input_norm_w.deinit();
        self.q_norm_w.deinit();
        self.k_norm_w.deinit();
        self.post_attn_w.deinit();
        self.pre_mlp_w.deinit();
        self.post_mlp_w.deinit();
        self.rope.deinit();
    }
};

pub const Gemma4GpuContext = struct {
    allocator: std.mem.Allocator,
    layers: []Gemma4GpuLayer,
    final_norm_w: offload.Array,
    lm_head: LmHeadWeights,
    exclusion_mask: TokenExclusionMask,
    config: Gemma4Config,
    max_seq_len: usize,
    stream: Stream,

    gpu_kv: []?offload.GpuKvCache,
    curr_len: usize = 0,
    sqrt_d: f32 = 0.0,

    pub fn init(
        allocator: std.mem.Allocator,
        cfg: anytype,
        layers: []Gemma4GpuLayer,
        final_norm_data: []const f32,
        lm_head: LmHeadWeights,
        max_seq_len: usize,
    ) !Gemma4GpuContext {
        if (!offload.isAvailable()) return error.OffloadNotSupported;
        if (layers.len != cfg.num_hidden_layers or final_norm_data.len != cfg.hidden_size) return error.ShapeMismatch;
        if (lm_head.vocab_size != cfg.vocab_size or lm_head.hidden_size != cfg.hidden_size) return error.ShapeMismatch;

        var final_norm_w = try offload.Array.fromSlice(final_norm_data, &.{cfg.hidden_size});
        errdefer final_norm_w.deinit();
        final_norm_w.eval();

        var exclusion_mask = try TokenExclusionMask.init(allocator, cfg.vocab_size, cfg.sampling_excluded_tokens);
        errdefer exclusion_mask.deinit();

        const gpu_kv = try allocator.alloc(?offload.GpuKvCache, layers.len);
        errdefer allocator.free(gpu_kv);
        @memset(gpu_kv, null);

        const stream = try Stream.defaultGpu();

        offload.enableCompile();
        offload.setCompileMode(.enabled);

        for (layers) |*lw| {
            lw.initCompiledMlp() catch {};
        }

        const gemma_cfg: Gemma4Config = .{
            .vocab_size = cfg.vocab_size,
            .hidden_size = cfg.hidden_size,
            .intermediate_size = cfg.intermediate_size,
            .num_hidden_layers = cfg.num_hidden_layers,
            .num_attention_heads = cfg.num_attention_heads,
            .num_key_value_heads = cfg.num_key_value_heads,
            .num_global_key_value_heads = cfg.num_global_key_value_heads,
            .head_dim = cfg.head_dim,
            .global_head_dim = cfg.global_head_dim,
            .sliding_window = cfg.sliding_window,
            .rms_norm_eps = cfg.rms_norm_eps,
            .final_logit_softcapping = cfg.final_logit_softcapping,
            .max_position_embeddings = cfg.max_position_embeddings,
            .hidden_activation = cfg.hidden_activation,
            .sampling_excluded_tokens = cfg.sampling_excluded_tokens,
        };

        return .{
            .allocator = allocator,
            .layers = layers,
            .final_norm_w = final_norm_w,
            .lm_head = lm_head,
            .exclusion_mask = exclusion_mask,
            .config = gemma_cfg,
            .max_seq_len = max_seq_len,
            .stream = stream,
            .gpu_kv = gpu_kv,
            .curr_len = 0,
            .sqrt_d = @sqrt(@as(f32, @floatFromInt(cfg.hidden_size))),
        };
    }

    pub fn deinit(self: *Gemma4GpuContext) void {
        self.stream.synchronize();
        self.resetCache();
        for (self.layers) |*l| {
            l.deinit();
        }
        self.allocator.free(self.layers);
        self.final_norm_w.deinit();
        self.lm_head.deinit();
        self.exclusion_mask.deinit();
        self.stream.deinit();
        self.allocator.free(self.gpu_kv);
    }

    pub fn resetCache(self: *Gemma4GpuContext) void {
        for (self.gpu_kv) |*c_opt| {
            if (c_opt.*) |*c| c.deinit();
            c_opt.* = null;
        }
        self.curr_len = 0;
    }

    pub fn sampleHidden(
        self: *const Gemma4GpuContext,
        h_last: []const f32,
        recent_tokens: []const usize,
        sampler: *Sampler,
        stream: ?Stream,
    ) !usize {
        return selectTokenFromHidden(self.allocator, &self.lm_head, &self.exclusion_mask, self.config, h_last, recent_tokens, sampler, stream orelse self.stream);
    }

    fn embedTokens(self: *const Gemma4GpuContext, tokens: []const usize, embed_u16: []const u16, out: []f32) !void {
        const h = self.config.hidden_size;
        const table = embed_u16[0..@min(embed_u16.len, self.config.vocab_size * h)];
        for (tokens, 0..) |tok, i| {
            if (tok >= self.config.vocab_size) return error.IndexOutOfBounds;
            const row = table[tok * h .. (tok + 1) * h];
            const dest = out[i * h .. (i + 1) * h];
            for (row, dest) |val, *d| {
                d.* = vec_math.bf16ToF32(val) * self.sqrt_d;
            }
        }
    }

    fn checkCacheLayout(self: *const Gemma4GpuContext, cache: anytype) !void {
        if (cache.layers.len != self.layers.len) return error.ShapeMismatch;
        for (self.layers, cache.layers) |lw, cl| {
            if (cl.core.num_kv_heads != lw.num_kv_heads or cl.core.head_dim != lw.head_dim or @backingInt(cl.attn_type) != @backingInt(lw.attn_type)) {
                return error.ShapeMismatch;
            }
        }
    }

    pub fn syncCacheFromCpu(self: *Gemma4GpuContext, cache: anytype) !void {
        try self.checkCacheLayout(cache);
        self.resetCache();
        const cached_len = if (cache.layers.len > 0) cache.layers[0].core.curr_len else 0;
        if (cached_len == 0) return;
        if (cached_len >= self.max_seq_len) return error.ContextWindowExceeded;

        for (self.layers, cache.layers, 0..) |lw, *cl, l_idx| {
            if (cl.core.curr_len != cached_len) return error.ShapeMismatch;
            const win = cl.windowLen();
            const packed_len = lw.num_kv_heads * win * lw.head_dim;

            const pack_buf_k = try self.allocator.alloc(f32, packed_len);
            defer self.allocator.free(pack_buf_k);
            const pack_buf_v = try self.allocator.alloc(f32, packed_len);
            defer self.allocator.free(pack_buf_v);
            try cl.packWindow(pack_buf_k, pack_buf_v);

            if (self.gpu_kv[l_idx] == null) {
                const default_cap = if (lw.attn_type == .sliding_attention) lw.sliding_window else 256;
                const initial_cap = @max(default_cap, ((win + 255) / 256) * 256);
                self.gpu_kv[l_idx] = try offload.GpuKvCache.init(initial_cap, lw.num_kv_heads, lw.head_dim, .bfloat16, self.stream);
            }
            try self.gpu_kv[l_idx].?.importWindow(pack_buf_k, pack_buf_v, win, self.stream);
        }
        self.curr_len = cached_len;
    }

    pub fn syncCacheToCpu(self: *const Gemma4GpuContext, cache: anytype) !void {
        try self.checkCacheLayout(cache);
        for (self.layers, cache.layers, 0..) |lw, *cl, l_idx| {
            if (self.curr_len == 0) {
                cl.reset();
                continue;
            }
            const kv_cache = &(self.gpu_kv[l_idx] orelse return error.ShapeMismatch);
            const win = cl.windowLen();
            const packed_len = lw.num_kv_heads * win * lw.head_dim;

            const buf_k = try self.allocator.alloc(f32, packed_len);
            defer self.allocator.free(buf_k);
            const buf_v = try self.allocator.alloc(f32, packed_len);
            defer self.allocator.free(buf_v);
            try kv_cache.exportWindow(buf_k, buf_v, win, self.stream);
            try cl.writeWindow(buf_k, buf_v, win, self.curr_len);
        }
    }

    const QkvHeads = struct {
        q_rot: offload.Array,
        k_rot: offload.Array,
        v: offload.Array,

        fn deinit(self: *QkvHeads) void {
            self.q_rot.deinit();
            self.k_rot.deinit();
            self.v.deinit();
        }
    };

    fn projectQkv(lw: *const Gemma4GpuLayer, in_norm: offload.Array, t: usize, pos: usize, s: Stream) !QkvHeads {
        var qkv_raw = try lw.fused_qkv.forwardArray(in_norm, s);
        defer qkv_raw.deinit();

        var q_raw = try qkv_raw.sliceAxis(2, 0, lw.q_dim, s);
        defer q_raw.deinit();
        var k_raw = try qkv_raw.sliceAxis(2, lw.q_dim, lw.q_dim + lw.k_dim, s);
        defer k_raw.deinit();
        var v_raw = if (lw.v_dim > 0)
            try qkv_raw.sliceAxis(2, lw.q_dim + lw.k_dim, lw.q_dim + lw.k_dim + lw.v_dim, s)
        else
            try k_raw.sliceAxis(2, 0, lw.k_dim, s);
        defer v_raw.deinit();

        var q_rot = try transformer_blocks.projectHeads(q_raw, lw.q_norm_w, t, lw.num_heads, lw.head_dim, lw.rms_norm_eps, &lw.rope, pos, s);
        errdefer q_rot.deinit();
        var k_rot = try transformer_blocks.projectHeads(k_raw, lw.k_norm_w, t, lw.num_kv_heads, lw.head_dim, lw.rms_norm_eps, &lw.rope, pos, s);
        errdefer k_rot.deinit();
        const v_normed = try transformer_blocks.projectHeads(v_raw, null, t, lw.num_kv_heads, lw.head_dim, lw.rms_norm_eps, null, pos, s);

        return .{ .q_rot = q_rot, .k_rot = k_rot, .v = v_normed };
    }

    fn layerTail(lw: *const Gemma4GpuLayer, x_cur: offload.Array, attn_out: offload.Array, t: usize, use_compiled: bool, s: Stream) !offload.Array {
        var attn_t: ?offload.Array = if (t == 1) null else try attn_out.transposeAxes(&.{ 0, 2, 1, 3 }, s);
        defer if (attn_t) |*a| a.deinit();
        var attn_2d = try (attn_t orelse attn_out).reshape(&.{ 1, t, lw.attn_dim }, s);
        defer attn_2d.deinit();
        var o_out = try lw.o_proj.forwardArray(attn_2d, s);
        defer o_out.deinit();

        var post_attn = try offload.fastRmsNorm(o_out, lw.post_attn_w, lw.rms_norm_eps, s);
        defer post_attn.deinit();
        var x1 = try transformer_blocks.residualScale(x_cur, post_attn, null, s);
        defer x1.deinit();
        return if (use_compiled) lw.runMlp(x1, s) else lw.mlpResidual(x1, s);
    }

    fn finalNorm(self: *const Gemma4GpuContext, x_cur: offload.Array, t: usize, rows: usize, s: Stream) !offload.Array {
        _ = t;
        return transformer_blocks.finalHidden(x_cur, self.final_norm_w, self.config.rms_norm_eps, rows, self.config.hidden_size, s);
    }

    pub fn prefill(
        self: *Gemma4GpuContext,
        prompt_tokens: []const usize,
        embed_u16: []const u16,
        sampler: *Sampler,
        stream: ?Stream,
    ) !usize {
        const T = prompt_tokens.len;
        if (T == 0) return error.EmptyPrompt;
        if (T >= self.max_seq_len) return error.ContextWindowExceeded;

        const s = stream orelse self.stream;
        self.resetCache();

        const tok_emb_buf = try self.allocator.alloc(f32, T * self.config.hidden_size);
        defer self.allocator.free(tok_emb_buf);
        try self.embedTokens(prompt_tokens, embed_u16, tok_emb_buf);
        var x_cur = try offload.Array.fromSlice(tok_emb_buf, &.{ 1, T, self.config.hidden_size });
        defer x_cur.deinit();

        for (self.layers, 0..) |*lw, l| {
            var in_norm = try offload.fastRmsNorm(x_cur, lw.input_norm_w, lw.rms_norm_eps, s);
            defer in_norm.deinit();
            var qkv = try projectQkv(lw, in_norm, T, 0, s);
            defer qkv.deinit();

            if (self.gpu_kv[l] == null) {
                const default_cap = if (lw.attn_type == .sliding_attention) lw.sliding_window else 256;
                const initial_cap = @max(default_cap, ((T + 255) / 256) * 256);
                self.gpu_kv[l] = try offload.GpuKvCache.init(initial_cap, lw.num_kv_heads, lw.head_dim, .bfloat16, s);
            }
            _ = try self.gpu_kv[l].?.writeStep(qkv.k_rot, qkv.v, lw.attn_type == .sliding_attention, lw.sliding_window, 0, T, s);

            var attn_out = try offload.maskedAttention(self.allocator, qkv.q_rot, qkv.k_rot, qkv.v, 1.0, lw.attn_type.mask(lw.sliding_window), 0, 0, s);
            defer attn_out.deinit();

            const x_next = try layerTail(lw, x_cur, attn_out, T, false, s);
            x_cur.deinit();
            x_cur = x_next;
        }

        var h_norm = try self.finalNorm(x_cur, T, 1, s);
        defer h_norm.deinit();
        var logits = try self.lm_head.forwardArrayBf16(h_norm, s);
        defer logits.deinit();
        const next = try selectToken(self.allocator, logits, self.config, &self.exclusion_mask, recentWindowFor(prompt_tokens, sampler.options), sampler, s);
        self.curr_len = T;
        return next;
    }

    pub fn decodeStep(
        self: *Gemma4GpuContext,
        token: usize,
        embed_u16: []const u16,
        recent_tokens: []const usize,
        sampler: *Sampler,
        stream: ?Stream,
    ) !usize {
        const pos = self.curr_len;
        if (pos >= self.max_seq_len) return error.ContextWindowExceeded;
        const s = stream orelse self.stream;

        const tok_emb_buf = try self.allocator.alloc(f32, self.config.hidden_size);
        defer self.allocator.free(tok_emb_buf);
        try self.embedTokens(&.{token}, embed_u16, tok_emb_buf);
        var x_cur = try offload.Array.fromSlice(tok_emb_buf, &.{ 1, 1, self.config.hidden_size });
        defer x_cur.deinit();

        for (self.layers, 0..) |*lw, l| {
            var in_norm = try offload.fastRmsNorm(x_cur, lw.input_norm_w, lw.rms_norm_eps, s);
            defer in_norm.deinit();
            var qkv = try projectQkv(lw, in_norm, 1, pos, s);
            defer qkv.deinit();

            if (self.gpu_kv[l] == null) {
                const default_cap = if (lw.attn_type == .sliding_attention) lw.sliding_window else 256;
                const initial_cap = @max(default_cap, ((pos + 1 + 255) / 256) * 256);
                self.gpu_kv[l] = try offload.GpuKvCache.init(initial_cap, lw.num_kv_heads, lw.head_dim, .bfloat16, s);
            }
            const view = try self.gpu_kv[l].?.writeStep(qkv.k_rot, qkv.v, lw.attn_type == .sliding_attention, lw.sliding_window, pos, 1, s);
            defer if (view.is_slice) {
                var vk = view.k;
                var vv = view.v;
                vk.deinit();
                vv.deinit();
            };

            var attn_out = try offload.fastScaledDotProductAttention(qkv.q_rot, view.k, view.v, 1.0, .none, s);
            defer attn_out.deinit();

            const x_next = try layerTail(lw, x_cur, attn_out, 1, true, s);
            x_cur.deinit();
            x_cur = x_next;
        }

        var h_norm = try self.finalNorm(x_cur, 1, 1, s);
        defer h_norm.deinit();
        var logits = try self.lm_head.forwardArrayBf16(h_norm, s);
        defer logits.deinit();
        const next = try selectToken(self.allocator, logits, self.config, &self.exclusion_mask, recentWindowFor(recent_tokens, sampler.options), sampler, s);
        self.curr_len = pos + 1;
        return next;
    }

    pub fn decodeSpeculative(
        self: *Gemma4GpuContext,
        input_tokens: []const usize,
        embed_u16: []const u16,
        recent_tokens: []const usize,
        sampler: *Sampler,
        stream: ?Stream,
        accepted_tokens: []usize,
    ) !usize {
        const M = input_tokens.len;
        if (M == 0) return 0;
        if (accepted_tokens.len < M) return error.ShapeMismatch;
        const pos = self.curr_len;
        if (pos + M >= self.max_seq_len) return error.ContextWindowExceeded;

        const s = stream orelse self.stream;

        const tok_emb_buf = try self.allocator.alloc(f32, M * self.config.hidden_size);
        defer self.allocator.free(tok_emb_buf);
        try self.embedTokens(input_tokens, embed_u16, tok_emb_buf);
        var x_cur = try offload.Array.fromSlice(tok_emb_buf, &.{ 1, M, self.config.hidden_size });
        defer x_cur.deinit();

        for (self.layers, 0..) |*lw, l| {
            var in_norm = try offload.fastRmsNorm(x_cur, lw.input_norm_w, lw.rms_norm_eps, s);
            defer in_norm.deinit();
            var qkv = try projectQkv(lw, in_norm, M, pos, s);
            defer qkv.deinit();

            if (self.gpu_kv[l] == null) {
                const default_cap = if (lw.attn_type == .sliding_attention) lw.sliding_window else 256;
                const initial_cap = @max(default_cap, ((pos + M + 255) / 256) * 256);
                self.gpu_kv[l] = try offload.GpuKvCache.init(initial_cap, lw.num_kv_heads, lw.head_dim, .bfloat16, s);
            }
            const view = try self.gpu_kv[l].?.writeStep(qkv.k_rot, qkv.v, lw.attn_type == .sliding_attention, lw.sliding_window, pos, M, s);
            defer if (view.is_slice) {
                var vk = view.k;
                var vv = view.v;
                vk.deinit();
                vv.deinit();
            };

            var attn_out = try offload.cachedAttention(self.allocator, qkv.q_rot, view.k, view.v, 1.0, lw.attn_type == .sliding_attention, lw.sliding_window, pos, M, s);
            defer attn_out.deinit();

            const x_next = try layerTail(lw, x_cur, attn_out, M, true, s);
            x_cur.deinit();
            x_cur = x_next;
        }

        var h_norm = try self.finalNorm(x_cur, M, M, s);
        defer h_norm.deinit();
        var logits_raw = try self.lm_head.forwardArrayBf16(h_norm, s);
        defer logits_raw.deinit();

        var history = try std.ArrayList(usize).initCapacity(self.allocator, recent_tokens.len + M);
        defer history.deinit(self.allocator);
        try history.appendSlice(self.allocator, recent_tokens);
        try history.appendSlice(self.allocator, input_tokens[1..]);

        const num_accepted = if (sampler.isGreedy())
            try self.verifyGreedy(logits_raw, input_tokens, history.items, recent_tokens.len, sampler.options, accepted_tokens, s)
        else
            try self.verifySampled(logits_raw, input_tokens, history.items, recent_tokens.len, sampler, accepted_tokens, s);

        if (M > num_accepted) {
            for (self.gpu_kv) |*maybe_kv| {
                if (maybe_kv.*) |*kv| kv.truncate(pos + num_accepted);
            }
        }

        self.curr_len = pos + num_accepted;
        return num_accepted;
    }

    fn verifyGreedy(
        self: *const Gemma4GpuContext,
        logits_raw: offload.Array,
        input_tokens: []const usize,
        history: []const usize,
        recent_len: usize,
        sampling_opts: SamplingOptions,
        accepted_tokens: []usize,
        s: Stream,
    ) !usize {
        const M = input_tokens.len;
        const V = self.config.vocab_size;
        var logits = try softcapLogits(logits_raw, self.config.final_logit_softcapping, s);
        defer logits.deinit();
        const logits_slice = try logits.dataFloat32();
        if (logits_slice.len != M * V) return error.ShapeMismatch;

        for (0..M) |i| {
            const row = logits_slice[i * V .. (i + 1) * V];
            const window = recentWindowFor(history[0 .. recent_len + i], sampling_opts);
            const pred = tensor.argmaxWithPenalty(row, V, self.config.sampling_excluded_tokens, window, sampling_opts.repetition_penalty);
            accepted_tokens[i] = pred;
            if (i + 1 < M and pred != input_tokens[i + 1]) return i + 1;
        }
        return M;
    }

    fn verifySampled(
        self: *const Gemma4GpuContext,
        logits_raw: offload.Array,
        input_tokens: []const usize,
        history: []const usize,
        recent_len: usize,
        sampler: *Sampler,
        accepted_tokens: []usize,
        s: Stream,
    ) !usize {
        const M = input_tokens.len;
        const pending = try self.allocator.alloc(DeviceCandidates, M);
        defer self.allocator.free(pending);
        var built: usize = 0;
        defer for (pending[0..built]) |*p| p.deinit();
        for (0..M) |i| {
            var row = try logits_raw.sliceAxis(0, i, i + 1, s);
            defer row.deinit();
            const window = recentWindowFor(history[0 .. recent_len + i], sampler.options);
            pending[i] = try deviceCandidates(self.allocator, row, self.config, &self.exclusion_mask, window, sampler.options, s);
            built += 1;
        }

        const all_arrays = try self.allocator.alloc(offload.Array, 3 * M);
        defer self.allocator.free(all_arrays);
        var n_arrays: usize = 0;
        for (pending) |*p| n_arrays += p.arrays(all_arrays[n_arrays..]);
        offload.evalMultiple(all_arrays[0..n_arrays]);

        const buf = try self.allocator.alloc(Candidate, pending[0].len());
        defer self.allocator.free(buf);
        for (0..M) |i| {
            const host = try pending[i].read(buf);
            const dist = sampler.distribution(host.candidates, host.log_normalizer);
            if (i + 1 == M) {
                accepted_tokens[i] = dist.sample(sampler.random());
                return M;
            }
            const draft = input_tokens[i + 1];
            if (draft < self.config.vocab_size) {
                switch (sampler.verifyDraft(dist, @intCast(draft))) {
                    .accepted => {
                        accepted_tokens[i] = draft;
                        continue;
                    },
                    .rejected => |tok| accepted_tokens[i] = tok,
                }
            } else {
                accepted_tokens[i] = dist.sample(sampler.random());
            }
            return i + 1;
        }
        unreachable;
    }

    pub fn warmup(self: *Gemma4GpuContext, embed_u16: []const u16, stream: ?Stream) !void {
        if (self.config.vocab_size < 3) return;
        const dummy_prompt = [_]usize{ 1, 2 };
        var greedy = Sampler.init(.greedy);
        _ = try self.prefill(&dummy_prompt, embed_u16, &greedy, stream);
        _ = try self.decodeStep(1, embed_u16, &dummy_prompt, &greedy, stream);

        var accepted_dummy: [2]usize = undefined;
        _ = self.decodeSpeculative(&dummy_prompt, embed_u16, &dummy_prompt, &greedy, stream, &accepted_dummy) catch {};

        self.resetCache();
        (stream orelse self.stream).synchronize();
    }
};
