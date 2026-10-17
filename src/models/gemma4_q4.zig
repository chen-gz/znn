//! Gemma 4 专用 4-bit 量化 (Q4) 模型架构与推理实现 (Gemma 4 Q4 Architecture)
//!
//! 包含 Q4Linear 量化线性层、Gemma4Q4Attention 混合注意力、Gemma4Q4DecoderLayer 与 Gemma4Q4ForCausalLM。
//! 采用 split-half Q4Block，并通过 Executor 进行多核并行，热路径不派生临时 OS 线程。

const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const nn = @import("../nn.zig");
const gemma4 = @import("gemma4.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Q4Block = tensor.quant.Q4Block;
const Linear = nn.core.Linear;
const Embedding = nn.transformer.Embedding;
const GemmaRMSNorm = gemma4.GemmaRMSNorm;
const GemmaUnscaledRMSNorm = gemma4.GemmaUnscaledRMSNorm;
const Gemma4Config = gemma4.Gemma4Config;
const Gemma4AttentionType = gemma4.Gemma4AttentionType;
const ropeSplitHalf = gemma4.ropeSplitHalf;

const createPersistentTensor = nn.core.createPersistentTensor;
const freePersistentTensor = nn.core.freePersistentTensor;
const deinitModel = nn.core.deinitModel;
const enterModuleScope = nn.core.enterModuleScope;

/// 4-bit 量化线性层 (Q4Linear)
/// 权重存储为 Q4Block 结构体切片，内存体积仅为原始单精度 F32 的 1/8、BF16 的 1/4！
pub const Q4Linear = struct {
    in_features: usize,
    out_features: usize,
    blocks: []Q4Block, // 权重分块数据，总长度 = (in_features * out_features) / 32
    owns_blocks: bool = true,
    name: ?[]const u8 = null,
    module_type: []const u8 = "Q4Linear",

    pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize) !Q4Linear {
        std.debug.assert((in_features * out_features) % Q4Block.QK == 0);
        const num_blocks = (in_features * out_features) / Q4Block.QK;
        const blocks = try allocator.alloc(Q4Block, num_blocks);
        @memset(blocks, std.mem.zeroes(Q4Block));

        return Q4Linear{
            .in_features = in_features,
            .out_features = out_features,
            .blocks = blocks,
            .owns_blocks = true,
        };
    }

    pub fn fromBlocks(in_features: usize, out_features: usize, blocks: []const Q4Block) Q4Linear {
        return Q4Linear{
            .in_features = in_features,
            .out_features = out_features,
            .blocks = @constCast(blocks),
            .owns_blocks = false,
        };
    }

    pub fn deinit(self: *Q4Linear, allocator: std.mem.Allocator) void {
        if (self.owns_blocks) {
            allocator.free(self.blocks);
        }
    }

    /// 从原始 F32 权重矩阵原地量化填充当前层
    pub fn quantizeFromF32(self: *Q4Linear, f32_weights: []const f32) void {
        std.debug.assert(f32_weights.len == self.in_features * self.out_features);
        tensor.quant.quantizeSlice(self.blocks, f32_weights);
    }

    /// 从原始 BF16 权重矩阵原地量化填充当前层
    pub fn quantizeFromBF16(self: *Q4Linear, bf16_weights: []const u16) void {
        std.debug.assert(bf16_weights.len == self.in_features * self.out_features);
        const num_blocks = self.blocks.len;
        var f32_chunk: [32]f32 = undefined;
        for (0..num_blocks) |b| {
            for (0..32) |j| {
                const b_bits = bf16_weights[b * 32 + j];
                const val: tensor.bf16 = .{ .bits = b_bits };
                f32_chunk[j] = val.toF32();
            }
            self.blocks[b] = Q4Block.quantize(&f32_chunk);
        }
    }

    /// 前向传播：高效按块点积计算 x * W，通过 Graph.q4MatMul 注册算子
    pub fn forward(self: *const Q4Linear, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        return graph.q4MatMul(x, self.blocks, self.in_features, self.out_features);
    }
};

/// Gemma 4 门控前馈多层感知机 (Q4 量化版本)
pub const Gemma4Q4MLP = struct {
    gate_proj: Q4Linear,
    up_proj: Q4Linear,
    down_proj: Q4Linear,
    name: ?[]const u8 = null,
    module_type: []const u8 = "Gemma4Q4MLP",

    pub fn init(allocator: std.mem.Allocator, hidden_size: usize, intermediate_size: usize) !Gemma4Q4MLP {
        const gate_proj = try Q4Linear.init(allocator, hidden_size, intermediate_size);
        errdefer deinitModel(&gate_proj, allocator);
        const up_proj = try Q4Linear.init(allocator, hidden_size, intermediate_size);
        errdefer deinitModel(&up_proj, allocator);
        const down_proj = try Q4Linear.init(allocator, intermediate_size, hidden_size);
        errdefer deinitModel(&down_proj, allocator);

        return Gemma4Q4MLP{
            .gate_proj = gate_proj,
            .up_proj = up_proj,
            .down_proj = down_proj,
        };
    }

    pub fn deinit(self: *Gemma4Q4MLP, allocator: std.mem.Allocator) void {
        self.gate_proj.deinit(allocator);
        self.up_proj.deinit(allocator);
        self.down_proj.deinit(allocator);
    }

    pub fn forward(self: *const Gemma4Q4MLP, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        var x_2d = x;
        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            x_2d = try graph.reshape(x, &.{ B * T, D });
        }

        const gate = try self.gate_proj.forward(graph, x_2d);
        const up = try self.up_proj.forward(graph, x_2d);
        const gelu_gate = try graph.gelu(gate);
        const hidden = try graph.mul(gelu_gate, up);
        const out = try self.down_proj.forward(graph, hidden);

        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            return try graph.reshape(out, &.{ B, T, D });
        }
        return out;
    }
};

/// Gemma 4 混合注意力机制 (Q4 量化版本)
pub const Gemma4Q4Attention = struct {
    q_proj: Q4Linear,
    k_proj: Q4Linear,
    v_proj: ?Q4Linear,
    o_proj: Q4Linear,
    q_norm: GemmaRMSNorm,
    k_norm: GemmaRMSNorm,
    v_norm: GemmaUnscaledRMSNorm,

    attn_type: Gemma4AttentionType,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    rope_theta: f32,
    partial_rotary_factor: f32,

    name: ?[]const u8 = null,
    module_type: []const u8 = "Gemma4Q4Attention",

    pub fn init(
        allocator: std.mem.Allocator,
        attn_type: Gemma4AttentionType,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        rms_norm_eps: f32,
    ) !Gemma4Q4Attention {
        const q_dim = num_heads * head_dim;
        const kv_dim = num_kv_heads * head_dim;

        const q_proj = try Q4Linear.init(allocator, hidden_size, q_dim);
        const k_proj = try Q4Linear.init(allocator, hidden_size, kv_dim);
        const v_proj = if (attn_type == .full_attention) null else try Q4Linear.init(allocator, hidden_size, kv_dim);
        const o_proj = try Q4Linear.init(allocator, q_dim, hidden_size);

        const q_norm = try GemmaRMSNorm.init(allocator, head_dim, rms_norm_eps);
        const k_norm = try GemmaRMSNorm.init(allocator, head_dim, rms_norm_eps);
        const v_norm = try GemmaUnscaledRMSNorm.init(allocator, head_dim, rms_norm_eps);

        const rope_theta: f32 = switch (attn_type) {
            .sliding_attention => 10000.0,
            .full_attention => 1000000.0,
        };
        const partial_rotary_factor: f32 = switch (attn_type) {
            .sliding_attention => 1.0,
            .full_attention => 0.25,
        };

        return Gemma4Q4Attention{
            .q_proj = q_proj,
            .k_proj = k_proj,
            .v_proj = v_proj,
            .o_proj = o_proj,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .v_norm = v_norm,
            .attn_type = attn_type,
            .hidden_size = hidden_size,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .sliding_window = sliding_window,
            .rope_theta = rope_theta,
            .partial_rotary_factor = partial_rotary_factor,
        };
    }

    pub fn deinit(self: *Gemma4Q4Attention, allocator: std.mem.Allocator) void {
        self.q_proj.deinit(allocator);
        self.k_proj.deinit(allocator);
        if (self.v_proj) |*vp| vp.deinit(allocator);
        self.o_proj.deinit(allocator);
        deinitModel(&self.q_norm, allocator);
        deinitModel(&self.k_norm, allocator);
        self.v_norm.deinit(allocator);
    }

    pub fn forward(self: *const Gemma4Q4Attention, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const B = x.shape.dims[0];
        const T = x.shape.dims[1];
        const C = self.hidden_size;
        const nh = self.num_heads;
        const n_kv = self.num_kv_heads;
        const hd = self.head_dim;
        const groups = nh / n_kv;

        const x_2d = try graph.reshape(x, &.{ B * T, C });

        // 1. Q4 投影 Q, K, V (在 full_attention 下 attention_k_eq_v 为 true，V 与 K 共享投影权重)
        const q_raw = try self.q_proj.forward(graph, x_2d);
        const k_raw = try self.k_proj.forward(graph, x_2d);
        const v_raw = if (self.v_proj) |*vp| try vp.forward(graph, x_2d) else k_raw;

        // 2. Q-Norm 与 K-Norm，以及无参数 v_norm
        const q_heads = try graph.reshape(q_raw, &.{ B * T * nh, hd });
        const q_normed = try self.q_norm.forward(graph, q_heads);
        const k_heads = try graph.reshape(k_raw, &.{ B * T * n_kv, hd });
        const k_normed = try self.k_norm.forward(graph, k_heads);
        const v_heads = try graph.reshape(v_raw, &.{ B * T * n_kv, hd });
        const v_normed = try self.v_norm.forward(graph, v_heads);

        // 3. 转置为四维头张量
        const q_4d = try graph.reshape(q_normed, &.{ B, T, nh, hd });
        const k_4d = try graph.reshape(k_normed, &.{ B, T, n_kv, hd });
        const v_4d = try graph.reshape(v_normed, &.{ B, T, n_kv, hd });

        const q = try graph.transposeND(q_4d, 1, 2);
        const k_t_unrot = try graph.transposeND(k_4d, 1, 2);
        const v = try graph.transposeND(v_4d, 1, 2);

        // 4. 施加半切分 RoPE 旋转位置编码 (Split-Half RoPE, Gemma 官方标准)
        const q_rot = try ropeSplitHalf(graph, q, 0, self.partial_rotary_factor, self.rope_theta);
        const k_rot = try ropeSplitHalf(graph, k_t_unrot, 0, self.partial_rotary_factor, self.rope_theta);

        // 5. GQA 广播
        var k = k_rot;
        var v_final = v;
        if (groups > 1) {
            k = try graph.repeatKV(k_rot, groups);
            v_final = try graph.repeatKV(v, groups);
        }

        // 6. 注意力计算 (官方实现中 scaling = 1.0)
        const k_trans = try graph.transposeND(k, 2, 3);
        var scores = try graph.batchMatMul(q_rot, k_trans);

        const mask_node = try graph.tensorND(&.{ 1, 1, T, T }, false);
        mask_node.is_buffer = true;
        for (0..T) |i| {
            for (0..T) |j| {
                if (j > i) {
                    mask_node.data[i * T + j] = -1e9;
                } else if (self.attn_type == .sliding_attention and i >= j + self.sliding_window) {
                    mask_node.data[i * T + j] = -1e9;
                } else {
                    mask_node.data[i * T + j] = 0.0;
                }
            }
        }
        scores = try graph.add(scores, mask_node);

        const att_sm = try graph.softmax(scores);
        const y_4d = try graph.batchMatMul(att_sm, v_final);

        // 7. 转置并输出投影
        const y_trans = try graph.transposeND(y_4d, 1, 2);
        const y_2d = try graph.reshape(y_trans, &.{ B * T, nh * hd });
        const out_2d = try self.o_proj.forward(graph, y_2d);

        return try graph.reshape(out_2d, &.{ B, T, C });
    }
};

/// Gemma 4 解码器层 (Q4 量化版本)
pub const Gemma4Q4DecoderLayer = struct {
    input_layernorm: GemmaRMSNorm,
    self_attn: Gemma4Q4Attention,
    post_attention_layernorm: GemmaRMSNorm,
    pre_feedforward_layernorm: GemmaRMSNorm,
    mlp: Gemma4Q4MLP,
    post_feedforward_layernorm: GemmaRMSNorm,
    layer_scalar: *Tensor,

    name: ?[]const u8 = null,
    module_type: []const u8 = "Gemma4Q4DecoderLayer",

    pub fn init(
        allocator: std.mem.Allocator,
        attn_type: Gemma4AttentionType,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        rms_norm_eps: f32,
    ) !Gemma4Q4DecoderLayer {
        const input_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        const self_attn = try Gemma4Q4Attention.init(
            allocator,
            attn_type,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            rms_norm_eps,
        );
        const post_attention_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        const pre_feedforward_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        const mlp = try Gemma4Q4MLP.init(allocator, hidden_size, intermediate_size);
        const post_feedforward_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);

        const layer_scalar = try createPersistentTensor(allocator, 1, 1, true);
        layer_scalar.data[0] = 1.0;
        layer_scalar.shape = Shape.init(&.{1});
        layer_scalar.strides = tensor.computeContiguousStrides(layer_scalar.shape);

        return Gemma4Q4DecoderLayer{
            .input_layernorm = input_layernorm,
            .self_attn = self_attn,
            .post_attention_layernorm = post_attention_layernorm,
            .pre_feedforward_layernorm = pre_feedforward_layernorm,
            .mlp = mlp,
            .post_feedforward_layernorm = post_feedforward_layernorm,
            .layer_scalar = layer_scalar,
        };
    }

    pub fn deinit(self: *Gemma4Q4DecoderLayer, allocator: std.mem.Allocator) void {
        deinitModel(&self.input_layernorm, allocator);
        self.self_attn.deinit(allocator);
        deinitModel(&self.post_attention_layernorm, allocator);
        deinitModel(&self.pre_feedforward_layernorm, allocator);
        self.mlp.deinit(allocator);
        deinitModel(&self.post_feedforward_layernorm, allocator);
        freePersistentTensor(allocator, self.layer_scalar);
    }

    pub fn forward(self: *const Gemma4Q4DecoderLayer, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const in_norm = try self.input_layernorm.forward(graph, x);
        const attn_out = try self.self_attn.forward(graph, in_norm);
        const post_attn = try self.post_attention_layernorm.forward(graph, attn_out);
        const x1 = try graph.add(x, post_attn);

        const pre_mlp = try self.pre_feedforward_layernorm.forward(graph, x1);
        const mlp_out = try self.mlp.forward(graph, pre_mlp);
        const post_mlp = try self.post_feedforward_layernorm.forward(graph, mlp_out);
        const x2 = try graph.add(x1, post_mlp);
        return try graph.mulScalar(x2, self.layer_scalar.data[0]);
    }
};

/// 4-bit 量化因果语言模型 (Gemma4Q4ForCausalLM)
/// 整体内存降低至原始全精度的 1/4 到 1/8，使得 12B 模型能够在 16GB 设备上完整运行！
pub fn Gemma4Q4ForCausalLM(comptime cfg: Gemma4Config) type {
    return struct {
        pub const config: Gemma4Config = cfg;
        const Self = @This();

        embed_tokens: Embedding,
        layers: []Gemma4Q4DecoderLayer,
        norm: GemmaRMSNorm,

        name: ?[]const u8 = null,
        module_type: []const u8 = "Gemma4Q4ForCausalLM",

        pub fn init(allocator: std.mem.Allocator) !Self {
            const embed_tokens = try Embedding.init(allocator, config.vocab_size, config.hidden_size);
            const layers = try allocator.alloc(Gemma4Q4DecoderLayer, config.num_hidden_layers);

            for (0..config.num_hidden_layers) |i| {
                const attn_type: Gemma4AttentionType = if ((i + 1) % 6 == 0)
                    .full_attention
                else
                    .sliding_attention;

                layers[i] = try Gemma4Q4DecoderLayer.init(
                    allocator,
                    attn_type,
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.sliding_window,
                    config.rms_norm_eps,
                );
            }

            const norm = try GemmaRMSNorm.init(allocator, config.hidden_size, config.rms_norm_eps);

            return Self{
                .embed_tokens = embed_tokens,
                .layers = layers,
                .norm = norm,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
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
            const tok_emb = try self.embed_tokens.forward(graph, token_ids);
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
            const embed_t = try graph.transposeND(self.embed_tokens.weight, 0, 1);
            const logits_raw_2d = try graph.matmul(h_2d, embed_t);

            // 5. 对数几率截断 (Logit Softcapping, Gemma 官方标尺 30.0)
            const cap = config.final_logit_softcapping;
            const scaled = try graph.mulScalar(logits_raw_2d, 1.0 / cap);
            const capped = try graph.tanh(scaled);
            const capped_logits_2d = try graph.mulScalar(capped, cap);

            return try graph.reshape(capped_logits_2d, &.{ B, T, config.vocab_size });
        }

        /// 计算 Gemma 4 12B Q4 模型二进制文件的理论字节大小
        pub fn computeExpectedFileSize() usize {
            const header_magic_bytes: usize = 8; // "ZNNQ4G02"
            const final_norm_bytes: usize = config.hidden_size * 2; // BF16
            const embed_tokens_bytes: usize = config.vocab_size * config.hidden_size * 2; // BF16

            var layers_total: usize = 0;
            const head_dim = config.head_dim;
            const num_kv = config.num_key_value_heads;

            for (0..config.num_hidden_layers) |layer_idx| {
                const is_full = ((layer_idx + 1) % 6 == 0);
                const norms_bytes = (config.hidden_size * 4 + head_dim * 3 + 1) * 2; // BF16

                const q_blocks = (config.hidden_size * config.num_attention_heads * head_dim) / Q4Block.QK;
                const k_blocks = (config.hidden_size * num_kv * head_dim) / Q4Block.QK;
                const v_blocks = if (is_full) 0 else (config.hidden_size * num_kv * head_dim) / Q4Block.QK;
                const o_blocks = (config.num_attention_heads * head_dim * config.hidden_size) / Q4Block.QK;
                const mlp_blocks = (config.hidden_size * config.intermediate_size * 3) / Q4Block.QK;

                const total_blocks = q_blocks + k_blocks + v_blocks + o_blocks + mlp_blocks;
                layers_total += norms_bytes + total_blocks * @sizeOf(Q4Block);
            }

            return header_magic_bytes + final_norm_bytes + embed_tokens_bytes + layers_total;
        }

        /// 从 4-bit 量化二进制映射切片 (mmap_bytes) 零拷贝加载全部 48 层权重与层归一化参数
        /// 遵循魔数规范 ZNNQ4G02（无向后兼容旧格式分支）
        pub fn loadFromMmap(allocator: std.mem.Allocator, mmap_bytes: []const u8) !Self {
            if (mmap_bytes.len < 8 or !std.mem.eql(u8, mmap_bytes[0..8], "ZNNQ4G02")) {
                return error.InvalidModelSignature;
            }

            const expected_len = computeExpectedFileSize();
            if (mmap_bytes.len < expected_len) {
                return error.IncompleteModelFile;
            }

            var cursor: usize = 8;

            // 1. 最终层 RMSNorm 权重 (3840 个 BF16)
            const norm = try GemmaRMSNorm.init(allocator, config.hidden_size, config.rms_norm_eps);
            const norm_u16_slice: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, mmap_bytes[cursor .. cursor + config.hidden_size * 2]));
            for (norm_u16_slice, 0..) |bits, idx| {
                const b: tensor.bf16 = .{ .bits = bits };
                norm.weight.data[idx] = b.toF32();
            }
            cursor += config.hidden_size * 2;

            // 2. 词表嵌入矩阵 (262,144 x 3840 个 BF16)
            const embed_tokens = try Embedding.init(allocator, config.vocab_size, config.hidden_size);
            cursor += config.vocab_size * config.hidden_size * 2;

            // 3. 逐层加载 48 层 Transformer
            const layers = try allocator.alloc(Gemma4Q4DecoderLayer, config.num_hidden_layers);

            for (0..config.num_hidden_layers) |i| {
                const is_full = ((i + 1) % 6 == 0);
                const attn_type: Gemma4AttentionType = if (is_full) .full_attention else .sliding_attention;

                var layer = try Gemma4Q4DecoderLayer.init(
                    allocator,
                    attn_type,
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.sliding_window,
                    config.rms_norm_eps,
                );

                // 加载当前层 5 个 Layernorm 参数 (BF16)
                inline for ([_]*GemmaRMSNorm{
                    &layer.input_layernorm,
                    &layer.post_attention_layernorm,
                    &layer.pre_feedforward_layernorm,
                    &layer.post_feedforward_layernorm,
                }) |ln| {
                    const u16_s: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, mmap_bytes[cursor .. cursor + config.hidden_size * 2]));
                    for (u16_s, 0..) |bits, idx| {
                        const b: tensor.bf16 = .{ .bits = bits };
                        ln.weight.data[idx] = b.toF32();
                    }
                    cursor += config.hidden_size * 2;
                }

                // 加载 Q-Norm 与 K-Norm
                inline for ([_]*GemmaRMSNorm{ &layer.self_attn.q_norm, &layer.self_attn.k_norm }) |ln| {
                    const u16_s: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, mmap_bytes[cursor .. cursor + config.head_dim * 2]));
                    for (u16_s, 0..) |bits, idx| {
                        const b: tensor.bf16 = .{ .bits = bits };
                        ln.weight.data[idx] = b.toF32();
                    }
                    cursor += config.head_dim * 2;
                }

                // 加载 v_norm.weight (BF16) 与 layer_scalar (BF16)
                {
                    const u16_s: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, mmap_bytes[cursor .. cursor + config.head_dim * 2]));
                    for (u16_s, 0..) |bits, idx| {
                        const b: tensor.bf16 = .{ .bits = bits };
                        layer.self_attn.v_norm.weight.data[idx] = b.toF32();
                    }
                    cursor += config.head_dim * 2;

                    const sc_bits = std.mem.readInt(u16, mmap_bytes[cursor .. cursor + 2][0..2], .little);
                    const b_sc: tensor.bf16 = .{ .bits = sc_bits };
                    layer.layer_scalar.data[0] = b_sc.toF32();
                    cursor += 2;
                }

                // 零拷贝绑定 Q4Block 分块数组
                const q_in = config.hidden_size;
                const q_out = config.num_attention_heads * config.head_dim;
                const k_in = config.hidden_size;
                const k_out = config.num_key_value_heads * config.head_dim;
                const o_in = q_out;
                const o_out = config.hidden_size;
                const ffn_in = config.hidden_size;
                const ffn_mid = config.intermediate_size;

                layer.self_attn.q_proj.deinit(allocator);
                const q_blocks_count = (q_in * q_out) / Q4Block.QK;
                const q_bytes = q_blocks_count * @sizeOf(Q4Block);
                layer.self_attn.q_proj.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + q_bytes]))));
                layer.self_attn.q_proj.owns_blocks = false;
                cursor += q_bytes;

                layer.self_attn.k_proj.deinit(allocator);
                const k_blocks_count = (k_in * k_out) / Q4Block.QK;
                const k_bytes = k_blocks_count * @sizeOf(Q4Block);
                layer.self_attn.k_proj.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + k_bytes]))));
                layer.self_attn.k_proj.owns_blocks = false;
                cursor += k_bytes;

                if (!is_full) {
                    layer.self_attn.v_proj.?.deinit(allocator);
                    const v_blocks_count = (k_in * k_out) / Q4Block.QK;
                    const v_bytes = v_blocks_count * @sizeOf(Q4Block);
                    layer.self_attn.v_proj.?.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + v_bytes]))));
                    layer.self_attn.v_proj.?.owns_blocks = false;
                    cursor += v_bytes;
                }

                layer.self_attn.o_proj.deinit(allocator);
                const o_blocks_count = (o_in * o_out) / Q4Block.QK;
                const o_bytes = o_blocks_count * @sizeOf(Q4Block);
                layer.self_attn.o_proj.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + o_bytes]))));
                layer.self_attn.o_proj.owns_blocks = false;
                cursor += o_bytes;

                layer.mlp.gate_proj.deinit(allocator);
                const gate_blocks_count = (ffn_in * ffn_mid) / Q4Block.QK;
                const gate_bytes = gate_blocks_count * @sizeOf(Q4Block);
                layer.mlp.gate_proj.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + gate_bytes]))));
                layer.mlp.gate_proj.owns_blocks = false;
                cursor += gate_bytes;

                layer.mlp.up_proj.deinit(allocator);
                const up_blocks_count = (ffn_in * ffn_mid) / Q4Block.QK;
                const up_bytes = up_blocks_count * @sizeOf(Q4Block);
                layer.mlp.up_proj.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + up_bytes]))));
                layer.mlp.up_proj.owns_blocks = false;
                cursor += up_bytes;

                layer.mlp.down_proj.deinit(allocator);
                const down_blocks_count = (ffn_mid * ffn_in) / Q4Block.QK;
                const down_bytes = down_blocks_count * @sizeOf(Q4Block);
                layer.mlp.down_proj.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + down_bytes]))));
                layer.mlp.down_proj.owns_blocks = false;
                cursor += down_bytes;

                layers[i] = layer;
            }

            return Self{
                .embed_tokens = embed_tokens,
                .layers = layers,
                .norm = norm,
            };
        }

        const BestToken = struct {
            id: usize,
            val: f32,
        };

        const ArgmaxContext = struct {
            embed_u16: []const u16,
            h_last: []const f32,
            hidden_size: usize,
            recent_tokens: []const usize,
            repetition_penalty: f32,

            fn mapFn(ctx: @This(), start_v: usize, end_v: usize) BestToken {
                var max_val: f32 = -1e30;
                var max_id: usize = start_v;
                const hidden = ctx.hidden_size;

                for (start_v..end_v) |v| {
                    if (v == 0 or (v >= 258880 and v <= 258884)) continue;

                    const row = ctx.embed_u16[v * hidden .. (v + 1) * hidden];
                    var dot: f32 = 0.0;
                    for (0..hidden) |d| {
                        const b: tensor.bf16 = .{ .bits = row[d] };
                        dot += ctx.h_last[d] * b.toF32();
                    }

                    if (ctx.repetition_penalty > 1.0) {
                        for (ctx.recent_tokens) |rec| {
                            if (rec == v) {
                                if (dot > 0.0) {
                                    dot /= ctx.repetition_penalty;
                                } else {
                                    dot *= ctx.repetition_penalty;
                                }
                                break;
                            }
                        }
                    }

                    if (dot > max_val) {
                        max_val = dot;
                        max_id = v;
                    }
                }

                return .{ .id = max_id, .val = max_val };
            }

            fn reduceFn(a: BestToken, b: BestToken) BestToken {
                return if (b.val > a.val) b else a;
            }
        };

        /// 自回归前向推理单步：输入历史 token_ids 序列与 mmap 字节，直接前向传播并返回下一个预测 token_id
        pub fn generateNextToken(
            self: *const Self,
            graph: *autodiff.Graph,
            token_ids: []const usize,
            mmap_bytes: []const u8,
        ) !usize {
            return self.generateNextTokenWithPenalty(graph, token_ids, mmap_bytes, 1.25);
        }

        /// 带有重复惩罚的高效生成单步 (贪婪采样)
        /// 通过 graph.executor().parallelReduce 并行，不派生临时 OS 线程
        pub fn generateNextTokenWithPenalty(
            self: *const Self,
            graph: *autodiff.Graph,
            token_ids: []const usize,
            mmap_bytes: []const u8,
            repetition_penalty: f32,
        ) !usize {
            const T = token_ids.len;
            if (T == 0) return 0;

            const embed_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(
                u16,
                mmap_bytes[8 + config.hidden_size * 2 .. 8 + config.hidden_size * 2 + config.vocab_size * config.hidden_size * 2],
            ));

            // 1. 词表嵌入查表
            const tok_emb = try graph.tensorND(&.{ 1, T, config.hidden_size }, false);
            const sqrt_d: f32 = @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
            for (token_ids, 0..) |tok, t_idx| {
                const emb_row = embed_u16[tok * config.hidden_size .. (tok + 1) * config.hidden_size];
                const dest = tok_emb.data[t_idx * config.hidden_size .. (t_idx + 1) * config.hidden_size];
                for (0..config.hidden_size) |d| {
                    const b: tensor.bf16 = .{ .bits = emb_row[d] };
                    dest[d] = b.toF32() * sqrt_d;
                }
            }

            // 2. 逐层前向传播
            var h = tok_emb;
            for (self.layers) |*layer| {
                h = try layer.forward(graph, h);
            }

            // 3. 最终层 RMSNorm
            const h_norm = try self.norm.forward(graph, h);

            // 4. 获取最后一个位置 T-1 的隐状态向量 [config.hidden_size]
            const last_row = h_norm.data[(T - 1) * config.hidden_size .. T * config.hidden_size];

            // 5. 使用执行器执行确定性多核点积投影与贪婪采样 (零临时 OS 线程派生)
            const recent_window = if (token_ids.len > 64) token_ids[token_ids.len - 64 ..] else token_ids;
            const ctx = ArgmaxContext{
                .embed_u16 = embed_u16,
                .h_last = last_row,
                .hidden_size = config.hidden_size,
                .recent_tokens = recent_window,
                .repetition_penalty = repetition_penalty,
            };

            const exec = graph.executor();
            const grain: usize = (config.vocab_size + 7) / 8;
            const best = try exec.parallelReduce(
                BestToken,
                config.vocab_size,
                grain,
                ctx,
                ArgmaxContext.mapFn,
                ArgmaxContext.reduceFn,
                BestToken{ .id = 0, .val = -1e30 },
            );

            return best.id;
        }
    };
}

/// 默认配置的 Gemma 4 12B 4-bit 量化模型类型
pub const DefaultGemma4Q4 = Gemma4Q4ForCausalLM(Gemma4Config.default);

/// 测试配置的微型 Q4 量化 Gemma 4 模型类型
pub const TinyQ4Gemma4 = Gemma4Q4ForCausalLM(Gemma4Config.tiny_test);
