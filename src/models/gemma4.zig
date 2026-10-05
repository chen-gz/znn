const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const nn = @import("../nn.zig");
const core = nn.core;
const normalization = nn.normalization;
const transformer = nn.transformer;
const serialization = nn.serialization;

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Linear = core.Linear;
const Embedding = transformer.Embedding;
const RMSNorm = normalization.RMSNorm;
const createPersistentTensor = core.createPersistentTensor;
const freePersistentTensor = core.freePersistentTensor;
const deinitModel = core.deinitModel;
const enterModuleScope = core.enterModuleScope;

// ============================================================================
// Gemma 4 专用非通用算子 (Split-Half & Proportional RoPE)
// ============================================================================

/// 标准 GPT-NeoX / LLaMA / Gemma 规范的半切分旋转位置编码张量前向核函数 (Split-Half RoPE)
/// 算法原理：将输入维度分为前半段 x1 与后半段 x2，rotate_half(x) = [-x2, x1]
/// 对于每个旋转角 i (0 <= i < num_angles):
/// out[i] = x[i] * cos(theta_i) - x[i + half] * sin(theta_i)
/// out[i + half] = x[i + half] * cos(theta_i) + x[i] * sin(theta_i)
/// 当 partial_rotary_factor < 1.0 (例如 Gemma 4 的 proportional RoPE) 时，
/// 仅前 num_angles = int(partial_rotary_factor * D / 2) 个分量参与旋转，其余分量保持原始值 (cos=1, sin=0)。
pub fn tensorRopeSplitHalf(
    self: *Tensor,
    start_pos: usize,
    partial_rotary_factor: f32,
    rope_theta: f32,
    allocator: std.mem.Allocator,
) !*Tensor {
    return self.rope(start_pos, .{
        .mode = .split_half,
        .partial_rotary_factor = partial_rotary_factor,
        .base = rope_theta,
    }, allocator);
}

/// Gemma 4 计算图级半切分旋转位置编码算子
pub fn ropeSplitHalf(
    graph: *autodiff.Graph,
    X: *Tensor,
    start_pos: usize,
    partial_rotary_factor: f32,
    rope_theta: f32,
) !*Tensor {
    return graph.rope(X, start_pos, .{
        .mode = .split_half,
        .partial_rotary_factor = partial_rotary_factor,
        .base = rope_theta,
    });
}

// ============================================================================
// Gemma 4 架构定义与模型实现
// ============================================================================

/// Gemma 4 注意力层类型枚举
pub const Gemma4AttentionType = enum {
    sliding_attention,
    full_attention,
};

/// Gemma 4 文本模型与主干配置结构体
pub const Gemma4Config = struct {
    vocab_size: usize = 262144,
    hidden_size: usize = 3840,
    intermediate_size: usize = 15360,
    num_hidden_layers: usize = 48,
    num_attention_heads: usize = 16,
    num_key_value_heads: usize = 8,
    num_global_key_value_heads: usize = 1,
    head_dim: usize = 256,
    global_head_dim: usize = 512,
    sliding_window: usize = 1024,
    rms_norm_eps: f32 = 1e-6,
    final_logit_softcapping: f32 = 30.0,
    max_position_embeddings: usize = 262144,

    // 特殊词元 ID
    bos_token_id: u32 = 2,
    eos_token_id: u32 = 1,
    pad_token_id: u32 = 0,

    pub const default: Gemma4Config = .{};
    pub fn defaultConfig() Gemma4Config {
        return .{};
    }

    /// 小型测试配置（用于单测与快速验证，结构完全对齐 Gemma 4 规范）
    pub const tiny_test: Gemma4Config = .{
        .vocab_size = 128,
        .hidden_size = 32,
        .intermediate_size = 64,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .num_global_key_value_heads = 1,
        .head_dim = 8,
        .global_head_dim = 16,
        .sliding_window = 4,
        .rms_norm_eps = 1e-6,
        .final_logit_softcapping = 30.0,
        .max_position_embeddings = 64,
    };
};

/// Gemma RMSNorm (with learnable scale)
pub const GemmaRMSNorm = struct {
    weight: *Tensor, // [dim]
    eps: f32,
    name: ?[]const u8 = null,
    module_type: []const u8 = "GemmaRMSNorm",

    pub fn init(allocator: std.mem.Allocator, dim: usize, eps: f32) !GemmaRMSNorm {
        const weight = try createPersistentTensor(allocator, 1, dim, true);
        errdefer freePersistentTensor(allocator, weight);
        @memset(weight.data, 1.0);
        weight.shape = Shape.init(&.{dim});
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        return GemmaRMSNorm{
            .weight = weight,
            .eps = eps,
        };
    }

    pub const formula = "y = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum x_i^2 + \\epsilon}} \\odot w";

    pub fn forward(self: *const GemmaRMSNorm, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        return try graph.rmsNorm(x, self.weight, self.eps);
    }
};

/// Gemma 无权重缩放均方根层归一化 (GemmaUnscaledRMSNorm)
/// 数学公式：y = RMSNorm(x, eps) = x / sqrt(mean(x^2) + eps)
/// 专用于 Gemma 4 的 Value states 预注意力归一化 (v_norm, with_scale=False)
pub const GemmaUnscaledRMSNorm = struct {
    weight: *Tensor,
    eps: f32,
    name: ?[]const u8 = null,
    module_type: []const u8 = "GemmaUnscaledRMSNorm",

    pub fn init(allocator: std.mem.Allocator, dim: usize, eps: f32) !GemmaUnscaledRMSNorm {
        const weight = try createPersistentTensor(allocator, 1, dim, false);
        errdefer freePersistentTensor(allocator, weight);
        @memset(weight.data, 1.0);
        weight.shape = Shape.init(&.{dim});
        weight.strides = tensor.computeContiguousStrides(weight.shape);

        return GemmaUnscaledRMSNorm{
            .weight = weight,
            .eps = eps,
        };
    }

    pub fn deinit(self: *GemmaUnscaledRMSNorm, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
    }

    pub const formula = "y = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum x_i^2 + \\epsilon}}";

    pub fn forward(self: *const GemmaUnscaledRMSNorm, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        return try graph.rmsNorm(x, self.weight, self.eps);
    }
};

/// Gemma 4 门控前馈多层感知机 (Gemma4MLP: gate_proj, up_proj, down_proj)
/// 结构：(GELU_tanh(x * W_gate) * (x * W_up)) * W_down
pub const Gemma4MLP = struct {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    name: ?[]const u8 = null,
    module_type: []const u8 = "Gemma4MLP",

    pub fn init(allocator: std.mem.Allocator, hidden_size: usize, intermediate_size: usize) !Gemma4MLP {
        const gate_proj = try Linear.init(allocator, hidden_size, intermediate_size);
        errdefer deinitModel(&gate_proj, allocator);
        const up_proj = try Linear.init(allocator, hidden_size, intermediate_size);
        errdefer deinitModel(&up_proj, allocator);
        const down_proj = try Linear.init(allocator, intermediate_size, hidden_size);
        errdefer deinitModel(&down_proj, allocator);

        return Gemma4MLP{
            .gate_proj = gate_proj,
            .up_proj = up_proj,
            .down_proj = down_proj,
        };
    }

    pub const formula = "y = (\\text{GELU}(x W_{\\text{gate}}) \\odot (x W_{\\text{up}})) W_{\\text{down}}";

    pub fn forward(self: *const Gemma4MLP, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
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

/// Gemma 4 混合注意力机制 (Gemma4Attention: 支持滑动窗口注意力和全局注意力)
pub const Gemma4Attention = struct {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: ?Linear,
    o_proj: Linear,
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
    module_type: []const u8 = "Gemma4Attention",

    pub fn init(
        allocator: std.mem.Allocator,
        attn_type: Gemma4AttentionType,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        rms_norm_eps: f32,
    ) !Gemma4Attention {
        const q_dim = num_heads * head_dim;
        const kv_dim = num_kv_heads * head_dim;

        const q_proj = try Linear.init(allocator, hidden_size, q_dim);
        errdefer deinitModel(&q_proj, allocator);
        const k_proj = try Linear.init(allocator, hidden_size, kv_dim);
        errdefer deinitModel(&k_proj, allocator);
        const v_proj = if (attn_type == .full_attention) null else try Linear.init(allocator, hidden_size, kv_dim);
        errdefer if (v_proj) |*vp| deinitModel(vp, allocator);
        const o_proj = try Linear.init(allocator, q_dim, hidden_size);
        errdefer deinitModel(&o_proj, allocator);

        const q_norm = try GemmaRMSNorm.init(allocator, head_dim, rms_norm_eps);
        errdefer deinitModel(&q_norm, allocator);
        const k_norm = try GemmaRMSNorm.init(allocator, head_dim, rms_norm_eps);
        errdefer deinitModel(&k_norm, allocator);
        const v_norm = try GemmaUnscaledRMSNorm.init(allocator, head_dim, rms_norm_eps);
        errdefer deinitModel(&v_norm, allocator);

        const rope_theta: f32 = switch (attn_type) {
            .sliding_attention => 10000.0,
            .full_attention => 1000000.0,
        };
        const partial_rotary_factor: f32 = switch (attn_type) {
            .sliding_attention => 1.0,
            .full_attention => 0.25,
        };

        return Gemma4Attention{
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

    pub const formula = "\\text{Attn}(Q, K, V) = \\text{softmax}\\left(Q K^T + M\\right) V W_o";

    pub fn forward(self: *const Gemma4Attention, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
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

        // 1. 投影 Q, K, V (在 full_attention 下 attention_k_eq_v 为 true，V 与 K 共享投影权重)
        const q_raw = try self.q_proj.forward(graph, x_2d); // [B*T, nh * hd]
        const k_raw = try self.k_proj.forward(graph, x_2d); // [B*T, n_kv * hd]
        const v_raw = if (self.v_proj) |*vp| try vp.forward(graph, x_2d) else k_raw;

        // 2. 对每个头应用 Q-Norm 与 K-Norm，以及无参数 v_norm
        const q_heads = try graph.reshape(q_raw, &.{ B * T * nh, hd });
        const q_normed = try self.q_norm.forward(graph, q_heads);
        const k_heads = try graph.reshape(k_raw, &.{ B * T * n_kv, hd });
        const k_normed = try self.k_norm.forward(graph, k_heads);
        const v_heads = try graph.reshape(v_raw, &.{ B * T * n_kv, hd });
        const v_normed = try self.v_norm.forward(graph, v_heads);

        // 3. 转置为四维头张量: [B, nh, T, hd] 与 [B, n_kv, T, hd]
        const q_4d = try graph.reshape(q_normed, &.{ B, T, nh, hd });
        const k_4d = try graph.reshape(k_normed, &.{ B, T, n_kv, hd });
        const v_4d = try graph.reshape(v_normed, &.{ B, T, n_kv, hd });

        const q = try graph.transposeND(q_4d, 1, 2);
        const k_t_unrot = try graph.transposeND(k_4d, 1, 2);
        const v = try graph.transposeND(v_4d, 1, 2);

        // 4. 施加半切分 RoPE 旋转位置编码 (Split-Half RoPE, Gemma 官方标准)
        const q_rot = try ropeSplitHalf(graph, q, 0, self.partial_rotary_factor, self.rope_theta);
        const k_rot = try ropeSplitHalf(graph, k_t_unrot, 0, self.partial_rotary_factor, self.rope_theta);

        // 5. GQA 广播扩展至 nh 个头
        var k = k_rot;
        var v_final = v;
        if (groups > 1) {
            k = try graph.repeatKV(k_rot, groups);
            v_final = try graph.repeatKV(v, groups);
        }

        // 6. 注意力得分与因果 / 滑动窗口掩码计算 (官方实现中 scaling = 1.0)
        const k_trans = try graph.transposeND(k, 2, 3);
        var scores = try graph.batchMatMul(q_rot, k_trans);

        // 构造因果掩码 / 滑动窗口掩码
        const mask_node = try graph.tensorND(&.{ 1, 1, T, T }, false);
        mask_node.is_buffer = true;
        for (0..T) |i| {
            for (0..T) |j| {
                if (j > i) {
                    mask_node.data[i * T + j] = -1e9; // 未来位置因果遮蔽
                } else if (self.attn_type == .sliding_attention and i >= j + self.sliding_window) {
                    mask_node.data[i * T + j] = -1e9; // 超出局部滑动窗口遮蔽
                } else {
                    mask_node.data[i * T + j] = 0.0;
                }
            }
        }
        scores = try graph.add(scores, mask_node);

        const att_sm = try graph.softmax(scores);
        const y_4d = try graph.batchMatMul(att_sm, v_final);

        // 7. 转置并融合输出
        const y_trans = try graph.transposeND(y_4d, 1, 2);
        const y_2d = try graph.reshape(y_trans, &.{ B * T, nh * hd });
        const out_2d = try self.o_proj.forward(graph, y_2d);

        return try graph.reshape(out_2d, &.{ B, T, C });
    }
};

/// Gemma 4 变换器层 (Gemma4DecoderLayer)
pub const Gemma4DecoderLayer = struct {
    input_layernorm: GemmaRMSNorm,
    self_attn: Gemma4Attention,
    post_attention_layernorm: GemmaRMSNorm,
    pre_feedforward_layernorm: GemmaRMSNorm,
    mlp: Gemma4MLP,
    post_feedforward_layernorm: GemmaRMSNorm,
    layer_scalar: *Tensor, // [1]

    name: ?[]const u8 = null,
    module_type: []const u8 = "Gemma4DecoderLayer",

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
    ) !Gemma4DecoderLayer {
        const input_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&input_layernorm, allocator);

        const self_attn = try Gemma4Attention.init(
            allocator,
            attn_type,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            rms_norm_eps,
        );
        errdefer deinitModel(&self_attn, allocator);

        const post_attention_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&post_attention_layernorm, allocator);

        const pre_feedforward_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&pre_feedforward_layernorm, allocator);

        const mlp = try Gemma4MLP.init(allocator, hidden_size, intermediate_size);
        errdefer deinitModel(&mlp, allocator);

        const post_feedforward_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&post_feedforward_layernorm, allocator);

        const layer_scalar = try createPersistentTensor(allocator, 1, 1, true);
        errdefer freePersistentTensor(allocator, layer_scalar);
        layer_scalar.data[0] = 1.0;
        layer_scalar.shape = Shape.init(&.{1});
        layer_scalar.strides = tensor.computeContiguousStrides(layer_scalar.shape);

        return Gemma4DecoderLayer{
            .input_layernorm = input_layernorm,
            .self_attn = self_attn,
            .post_attention_layernorm = post_attention_layernorm,
            .pre_feedforward_layernorm = pre_feedforward_layernorm,
            .mlp = mlp,
            .post_feedforward_layernorm = post_feedforward_layernorm,
            .layer_scalar = layer_scalar,
        };
    }

    pub const formula = "\\begin{aligned} x_1 &= x + \\text{PostAttnLN}(\\text{Attn}(\\text{InputLN}(x))) \\\\ x_2 &= x_1 + \\text{PostMLPLN}(\\text{MLP}(\\text{PreMLPLN}(x_1))) \\end{aligned}";

    pub fn forward(self: *const Gemma4DecoderLayer, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        // 1. 注意力分支: InputLN -> Attn -> PostAttnLN -> + x
        const in_norm = try self.input_layernorm.forward(graph, x);
        const attn_out = try self.self_attn.forward(graph, in_norm);
        const post_attn = try self.post_attention_layernorm.forward(graph, attn_out);
        const x1 = try graph.add(x, post_attn);

        // 2. 前馈网络分支: PreMLPLN -> MLP -> PostMLPLN -> + x1
        const pre_mlp = try self.pre_feedforward_layernorm.forward(graph, x1);
        const mlp_out = try self.mlp.forward(graph, pre_mlp);
        const post_mlp = try self.post_feedforward_layernorm.forward(graph, mlp_out);
        return try graph.add(x1, post_mlp);
    }
};

/// Gemma 4 语言模型核心架构 (Gemma4ForCausalLM)
pub fn Gemma4ForCausalLM(comptime cfg: Gemma4Config) type {
    return struct {
        const Self = @This();
        pub const config: Gemma4Config = cfg;

        embed_tokens: Embedding,
        layers: []Gemma4DecoderLayer,
        norm: GemmaRMSNorm,
        name: ?[]const u8 = null,
        module_type: []const u8 = "Gemma4ForCausalLM",

        pub fn initDefault(allocator: std.mem.Allocator) !Self {
            return init(allocator);
        }

        pub fn init(allocator: std.mem.Allocator) !Self {
            const embed_tokens = try Embedding.init(allocator, config.vocab_size, config.hidden_size);
            errdefer deinitModel(&embed_tokens, allocator);

            const layers = try allocator.alloc(Gemma4DecoderLayer, config.num_hidden_layers);
            var initialized_layers: usize = 0;
            errdefer {
                for (0..initialized_layers) |idx| {
                    deinitModel(&layers[idx], allocator);
                }
                allocator.free(layers);
            }

            for (0..config.num_hidden_layers) |i| {
                // 每 6 层中最后一层为 full_attention，其余为 sliding_attention
                const is_full = ((i + 1) % 6 == 0);
                const attn_type: Gemma4AttentionType = if (is_full) .full_attention else .sliding_attention;
                const head_dim = if (is_full) config.global_head_dim else config.head_dim;
                const num_kv = if (is_full) config.num_global_key_value_heads else config.num_key_value_heads;

                layers[i] = try Gemma4DecoderLayer.init(
                    allocator,
                    attn_type,
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                    num_kv,
                    head_dim,
                    config.sliding_window,
                    config.rms_norm_eps,
                );
                initialized_layers += 1;
            }

            const norm = try GemmaRMSNorm.init(allocator, config.hidden_size, config.rms_norm_eps);
            errdefer deinitModel(&norm, allocator);

            return Self{
                .embed_tokens = embed_tokens,
                .layers = layers,
                .norm = norm,
            };
        }

        pub const formula = "\\text{logits} = \\text{Softcap}\\left(\\text{Norm}(\\text{Layers}(\\text{Embed}(x) \\cdot \\sqrt{d})) W_e^T, c\\right)";

        pub fn forward(self: *const Self, graph: *autodiff.Graph, token_ids: anytype) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const B = token_ids.shape.dims[0];
            const T = token_ids.shape.dims[1];

            // 1. 词嵌入与 Gemma 嵌入缩放 sqrt(hidden_size)
            const tok_emb = try self.embed_tokens.forward(graph, token_ids);
            const scale = @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
            var h = try graph.mulScalar(tok_emb, scale);

            // 2. 逐层前向传播
            for (self.layers) |*layer| {
                h = try layer.forward(graph, h);
            }

            // 3. 最终归一化
            const h_norm = try self.norm.forward(graph, h);

            // 4. 绑定权重投影 (Tie Word Embeddings: h * embed_tokens.weight^T)
            const h_2d = try graph.reshape(h_norm, &.{ B * T, config.hidden_size });
            const embed_t = try graph.transposeND(self.embed_tokens.weight, 0, 1);
            const logits_raw_2d = try graph.matmul(h_2d, embed_t);

            // 5. Logit Softcapping: 30 * tanh(logits / 30)
            const cap = config.final_logit_softcapping;
            const scaled_logits = try graph.mulScalar(logits_raw_2d, 1.0 / cap);
            const tanh_logits = try graph.tanh(scaled_logits);
            const capped_logits_2d = try graph.mulScalar(tanh_logits, cap);

            return try graph.reshape(capped_logits_2d, &.{ B, T, config.vocab_size });
        }
    };
}

// ============================================================================
// 4-bit (Q4_0) 分块量化格式与线性投影层 (4-bit Block Quantization & Q4Linear)
// ============================================================================

/// Q4_0 标准量化块：每 32 个权重为一组，包含 1 个 16-bit 浮点数比例因子 (scale) 与 16 字节 (32 个 4-bit 无符号量化值)
pub const Q4Block = extern struct {
    scale: f16, // 16 位浮点数缩放比例因子
    qs: [16]u8, // 16 个字节，每字节存储 2 个 4 位权重量化值 (低 4 位与高 4 位，偏移值为 8)

    pub const QK: usize = 32; // 每块量化权重数量

    /// 将 32 个 f32 浮点权重原地量化并打包为 Q4Block
    pub fn quantize(src: *const [32]f32) Q4Block {
        var amax: f32 = 0.0;
        for (src) |v| {
            const av = @abs(v);
            if (av > amax) amax = av;
        }
        const scale_f32 = amax / 7.0; // 4-bit 有符号映射至 [-7, +7]
        const inv_scale = if (scale_f32 > 0.0) 1.0 / scale_f32 else 0.0;
        var block: Q4Block = .{
            .scale = @floatCast(scale_f32),
            .qs = undefined,
        };

        for (0..16) |i| {
            const v0 = src[2 * i];
            const v1 = src[2 * i + 1];

            const q0_f = std.math.clamp(std.math.round(v0 * inv_scale) + 8.0, 0.0, 15.0);
            const q1_f = std.math.clamp(std.math.round(v1 * inv_scale) + 8.0, 0.0, 15.0);

            const q0: u8 = @intFromFloat(q0_f);
            const q1: u8 = @intFromFloat(q1_f);

            block.qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
        return block;
    }

    /// 反量化解包为 32 个 f32 浮点权重
    pub fn dequantize(self: Q4Block, dest: *[32]f32) void {
        const d: f32 = @floatCast(self.scale);
        for (0..16) |i| {
            const byte = self.qs[i];
            const q0: f32 = @as(f32, @floatFromInt(byte & 0x0F)) - 8.0;
            const q1: f32 = @as(f32, @floatFromInt(byte >> 4)) - 8.0;
            dest[2 * i] = q0 * d;
            dest[2 * i + 1] = q1 * d;
        }
    }
};

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
        const num_blocks = self.blocks.len;
        for (0..num_blocks) |b| {
            const chunk: *const [32]f32 = f32_weights[b * 32 .. (b + 1) * 32][0..32];
            self.blocks[b] = Q4Block.quantize(chunk);
        }
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

    const WorkerContext = struct {
        blocks: []const Q4Block,
        x_row: []const f32,
        out_row: []f32,
        blocks_per_in: usize,
        start_j: usize,
        end_j: usize,
    };

    fn dotWorker(ctx: WorkerContext) void {
        const V4 = @Vector(4, f32);
        const eight: V4 = @splat(8.0);

        for (ctx.start_j..ctx.end_j) |out_j| {
            var total_sum: f32 = 0.0;
            const base_b = out_j * ctx.blocks_per_in;

            for (0..ctx.blocks_per_in) |b| {
                const blk = ctx.blocks[base_b + b];
                const d: f32 = @floatCast(blk.scale);
                const x_chunk: *const [32]f32 = @ptrCast(ctx.x_row[b * 32 .. (b + 1) * 32].ptr);

                var acc: V4 = @splat(0.0);
                inline for (0..4) |g| {
                    const b0 = blk.qs[g * 4 + 0];
                    const b1 = blk.qs[g * 4 + 1];
                    const b2 = blk.qs[g * 4 + 2];
                    const b3 = blk.qs[g * 4 + 3];

                    const q_lo: V4 = .{
                        @floatFromInt(b0 & 0x0F),
                        @floatFromInt(b1 & 0x0F),
                        @floatFromInt(b2 & 0x0F),
                        @floatFromInt(b3 & 0x0F),
                    };
                    const q_hi: V4 = .{
                        @floatFromInt(b0 >> 4),
                        @floatFromInt(b1 >> 4),
                        @floatFromInt(b2 >> 4),
                        @floatFromInt(b3 >> 4),
                    };

                    const w0 = q_lo - eight;
                    const w1 = q_hi - eight;

                    const x0: V4 = .{
                        x_chunk[g * 8 + 0],
                        x_chunk[g * 8 + 2],
                        x_chunk[g * 8 + 4],
                        x_chunk[g * 8 + 6],
                    };
                    const x1: V4 = .{
                        x_chunk[g * 8 + 1],
                        x_chunk[g * 8 + 3],
                        x_chunk[g * 8 + 5],
                        x_chunk[g * 8 + 7],
                    };

                    acc += x0 * w0 + x1 * w1;
                }
                total_sum += @reduce(.Add, acc) * d;
            }
            ctx.out_row[out_j] = total_sum;
        }
    }

    /// 前向传播：高效按块点积计算 x * W，无需将整个权重矩阵完全反量化展开
    /// 输入 x: [B*T, in_features] -> 输出 [B*T, out_features]
    pub fn forward(self: *const Q4Linear, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const num_rows = x.data.len / self.in_features;
        const out_tensor = try graph.zeros(&.{ num_rows, self.out_features }, false);
        const blocks_per_in = self.in_features / Q4Block.QK;

        const num_threads: usize = if (self.out_features >= 2048) 8 else 1;

        for (0..num_rows) |r| {
            const x_row = x.data[r * self.in_features .. (r + 1) * self.in_features];
            const out_row = out_tensor.data[r * self.out_features .. (r + 1) * self.out_features];

            if (num_threads > 1) {
                var threads: [8]std.Thread = undefined;
                const chunk_size = (self.out_features + num_threads - 1) / num_threads;
                for (0..num_threads) |t| {
                    const start_j = t * chunk_size;
                    const end_j = @min(start_j + chunk_size, self.out_features);
                    threads[t] = try std.Thread.spawn(.{}, dotWorker, .{WorkerContext{
                        .blocks = self.blocks,
                        .x_row = x_row,
                        .out_row = out_row,
                        .blocks_per_in = blocks_per_in,
                        .start_j = start_j,
                        .end_j = end_j,
                    }});
                }
                for (0..num_threads) |t| {
                    threads[t].join();
                }
            } else {
                dotWorker(.{
                    .blocks = self.blocks,
                    .x_row = x_row,
                    .out_row = out_row,
                    .blocks_per_in = blocks_per_in,
                    .start_j = 0,
                    .end_j = self.out_features,
                });
            }
        }

        return out_tensor;
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
        const Self = @This();
        pub const config: Gemma4Config = cfg;

        embed_tokens: Embedding,
        layers: []Gemma4Q4DecoderLayer,
        norm: GemmaRMSNorm,
        name: ?[]const u8 = null,
        module_type: []const u8 = "Gemma4Q4ForCausalLM",

        pub fn init(allocator: std.mem.Allocator) !Self {
            const embed_tokens = try Embedding.init(allocator, config.vocab_size, config.hidden_size);
            const layers = try allocator.alloc(Gemma4Q4DecoderLayer, config.num_hidden_layers);

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
            for (self.layers) |*l| l.deinit(allocator);
            allocator.free(self.layers);
            deinitModel(&self.norm, allocator);
        }

        pub fn forward(self: *const Self, graph: *autodiff.Graph, token_ids: anytype) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const B = token_ids.shape.dims[0];
            const T = token_ids.shape.dims[1];

            const tok_emb = try self.embed_tokens.forward(graph, token_ids);
            const scale = @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
            var h = try graph.mulScalar(tok_emb, scale);

            for (self.layers) |*layer| {
                h = try layer.forward(graph, h);
            }

            const h_norm = try self.norm.forward(graph, h);
            const h_2d = try graph.reshape(h_norm, &.{ B * T, config.hidden_size });
            const embed_t = try graph.transposeND(self.embed_tokens.weight, 0, 1);
            const logits_raw_2d = try graph.matmul(h_2d, embed_t);
            const cap = config.final_logit_softcapping;
            const scaled_logits = try graph.mulScalar(logits_raw_2d, 1.0 / cap);
            const tanh_logits = try graph.tanh(scaled_logits);
            const capped_logits_2d = try graph.mulScalar(tanh_logits, cap);

            return try graph.reshape(capped_logits_2d, &.{ B, T, config.vocab_size });
        }

        /// 计算该模型配置下 4-bit 量化二进制文件的理论确切字节大小
        pub fn computeExpectedFileSize() usize {
            const magic_bytes = 8;
            const norm_bytes = config.hidden_size * 2;
            const embed_bytes = config.vocab_size * config.hidden_size * 2;

            var layers_total: usize = 0;
            for (0..config.num_hidden_layers) |i| {
                const is_full = ((i + 1) % 6 == 0);
                const head_dim = if (is_full) config.global_head_dim else config.head_dim;
                const num_kv = if (is_full) config.num_global_key_value_heads else config.num_key_value_heads;

                // 7 个 BF16 张量
                const norms_bytes = (config.hidden_size * 4 + head_dim * 2 + 1) * 2;

                const q_blocks = (config.hidden_size * config.num_attention_heads * head_dim) / Q4Block.QK;
                const k_blocks = (config.hidden_size * num_kv * head_dim) / Q4Block.QK;
                const v_blocks = if (is_full) 0 else (config.hidden_size * num_kv * head_dim) / Q4Block.QK;
                const o_blocks = (config.num_attention_heads * head_dim * config.hidden_size) / Q4Block.QK;
                const mlp_blocks = (config.hidden_size * config.intermediate_size * 3) / Q4Block.QK;

                const total_blocks = q_blocks + k_blocks + v_blocks + o_blocks + mlp_blocks;
                layers_total += norms_bytes + total_blocks * @sizeOf(Q4Block);
            }

            return magic_bytes + norm_bytes + embed_bytes + layers_total;
        }

        /// 从 4-bit 量化二进制映射切片 (mmap_bytes) 零拷贝加载全部 48 层权重与层归一化参数
        pub fn loadFromMmap(allocator: std.mem.Allocator, mmap_bytes: []const u8) !Self {
            if (mmap_bytes.len < 8 or !std.mem.eql(u8, mmap_bytes[0..8], "ZNNQ4G01")) {
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
            // 这里将 embed_tokens 的底层数据指针直接指向 mmap 切片（或保持空占位），因为实际推理使用 sampleNextToken 零拷贝点积
            const embed_bytes_len = config.vocab_size * config.hidden_size * 2;
            cursor += embed_bytes_len;

            // 3. 逐层加载 48 层 Transformer 解码器
            const layers = try allocator.alloc(Gemma4Q4DecoderLayer, config.num_hidden_layers);
            for (0..config.num_hidden_layers) |i| {
                const is_full = ((i + 1) % 6 == 0);
                const attn_type: Gemma4AttentionType = if (is_full) .full_attention else .sliding_attention;
                const head_dim = if (is_full) config.global_head_dim else config.head_dim;
                const num_kv = if (is_full) config.num_global_key_value_heads else config.num_key_value_heads;

                var layer = try Gemma4Q4DecoderLayer.init(
                    allocator,
                    attn_type,
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                    num_kv,
                    head_dim,
                    config.sliding_window,
                    config.rms_norm_eps,
                );

                // 加载当前层 Norm 参数 (7 个 BF16 张量)
                const norm_refs = [_]*Tensor{
                    layer.input_layernorm.weight,
                    layer.post_attention_layernorm.weight,
                    layer.pre_feedforward_layernorm.weight,
                    layer.post_feedforward_layernorm.weight,
                    layer.self_attn.q_norm.weight,
                    layer.self_attn.k_norm.weight,
                    layer.layer_scalar,
                };

                for (norm_refs) |t| {
                    const t_len = t.data.len;
                    const u16_s: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, mmap_bytes[cursor .. cursor + t_len * 2]));
                    for (u16_s, 0..) |bits, idx| {
                        const b: tensor.bf16 = .{ .bits = bits };
                        t.data[idx] = b.toF32();
                    }
                    cursor += t_len * 2;
                }

                // 零拷贝绑定 Q4Block 分块数组
                const q_in = config.hidden_size;
                const q_out = config.num_attention_heads * head_dim;
                const k_in = config.hidden_size;
                const k_out = num_kv * head_dim;
                const o_in = q_out;
                const o_out = config.hidden_size;
                const ffn_in = config.hidden_size;
                const ffn_mid = config.intermediate_size;

                // q_proj
                const q_blocks_count = (q_in * q_out) / Q4Block.QK;
                const q_bytes = q_blocks_count * @sizeOf(Q4Block);
                layer.self_attn.q_proj.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + q_bytes]))));
                layer.self_attn.q_proj.owns_blocks = false;
                cursor += q_bytes;

                // k_proj
                const k_blocks_count = (k_in * k_out) / Q4Block.QK;
                const k_bytes = k_blocks_count * @sizeOf(Q4Block);
                layer.self_attn.k_proj.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + k_bytes]))));
                layer.self_attn.k_proj.owns_blocks = false;
                cursor += k_bytes;

                // v_proj (仅滑动窗口层存在)
                if (attn_type == .sliding_attention) {
                    const v_blocks_count = (k_in * k_out) / Q4Block.QK;
                    const v_bytes = v_blocks_count * @sizeOf(Q4Block);
                    layer.self_attn.v_proj.?.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + v_bytes]))));
                    layer.self_attn.v_proj.?.owns_blocks = false;
                    cursor += v_bytes;
                }

                // o_proj
                const o_blocks_count = (o_in * o_out) / Q4Block.QK;
                const o_bytes = o_blocks_count * @sizeOf(Q4Block);
                layer.self_attn.o_proj.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + o_bytes]))));
                layer.self_attn.o_proj.owns_blocks = false;
                cursor += o_bytes;

                // gate_proj
                const gate_blocks_count = (ffn_in * ffn_mid) / Q4Block.QK;
                const gate_bytes = gate_blocks_count * @sizeOf(Q4Block);
                layer.mlp.gate_proj.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + gate_bytes]))));
                layer.mlp.gate_proj.owns_blocks = false;
                cursor += gate_bytes;

                // up_proj
                const up_blocks_count = (ffn_in * ffn_mid) / Q4Block.QK;
                const up_bytes = up_blocks_count * @sizeOf(Q4Block);
                layer.mlp.up_proj.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor .. cursor + up_bytes]))));
                layer.mlp.up_proj.owns_blocks = false;
                cursor += up_bytes;

                // down_proj
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

        const ArgmaxWorker = struct {
            embed_u16: []const u16,
            h_last: []const f32,
            hidden_size: usize,
            start_v: usize,
            end_v: usize,
            best_id: usize,
            best_val: f32,
            recent_tokens: []const usize,
            repetition_penalty: f32,
        };

        fn argmaxWorkerFn(ctx: *ArgmaxWorker) void {
            var max_val: f32 = -1e30;
            var max_id: usize = ctx.start_v;
            const hidden = ctx.hidden_size;

            for (ctx.start_v..ctx.end_v) |v| {
                // 抑制特殊系统/多模态控制符 (0: <pad>, 258880-258884: image/audio/video 标记)
                if (v == 0 or (v >= 258880 and v <= 258884)) continue;

                const row = ctx.embed_u16[v * hidden .. (v + 1) * hidden];
                var dot: f32 = 0.0;
                for (0..hidden) |d| {
                    const b: tensor.bf16 = .{ .bits = row[d] };
                    dot += ctx.h_last[d] * b.toF32();
                }

                // 重复惩罚 (Repetition Penalty)：降低最近已生成 Token 的 Logit 权重
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
            ctx.best_id = max_id;
            ctx.best_val = max_val;
        }

        /// 自回归前向推理单步：输入历史 token_ids 序列与 mmap 字节，直接前向传播并返回下一个预测 token_id
        pub fn generateNextToken(
            self: *const Self,
            graph: *autodiff.Graph,
            token_ids: []const usize,
            mmap_bytes: []const u8,
        ) !usize {
            return self.generateNextTokenWithPenalty(graph, token_ids, mmap_bytes, 1.25);
        }

        /// 带有重复惩罚 (Repetition Penalty) 的自回归前向推理
        pub fn generateNextTokenWithPenalty(
            self: *const Self,
            graph: *autodiff.Graph,
            token_ids: []const usize,
            mmap_bytes: []const u8,
            repetition_penalty: f32,
        ) !usize {
            const T = token_ids.len;
            const embed_start = 8 + config.hidden_size * 2;
            const embed_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, mmap_bytes[embed_start .. embed_start + config.vocab_size * config.hidden_size * 2]));

            // 1. 构建输入嵌入张量 [1, T, hidden_size]
            const tok_emb = try graph.zeros(&.{ 1, T, config.hidden_size }, false);
            const scale = @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
            for (token_ids, 0..) |tid, t| {
                const row = embed_u16[tid * config.hidden_size .. (tid + 1) * config.hidden_size];
                const dest = tok_emb.data[t * config.hidden_size .. (t + 1) * config.hidden_size];
                for (0..config.hidden_size) |d| {
                    const b: tensor.bf16 = .{ .bits = row[d] };
                    dest[d] = b.toF32() * scale;
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

            // 5. 多线程高效点积投影并执行带有重复惩罚的贪婪采样
            const recent_window = if (token_ids.len > 64) token_ids[token_ids.len - 64 ..] else token_ids;
            const num_threads: usize = 8;
            var workers: [num_threads]ArgmaxWorker = undefined;
            var threads: [num_threads]std.Thread = undefined;
            const chunk = (config.vocab_size + num_threads - 1) / num_threads;

            for (0..num_threads) |t| {
                const start_v = t * chunk;
                const end_v = @min(start_v + chunk, config.vocab_size);
                workers[t] = .{
                    .embed_u16 = embed_u16,
                    .h_last = last_row,
                    .hidden_size = config.hidden_size,
                    .start_v = start_v,
                    .end_v = end_v,
                    .best_id = start_v,
                    .best_val = -1e30,
                    .recent_tokens = recent_window,
                    .repetition_penalty = repetition_penalty,
                };
                threads[t] = try std.Thread.spawn(.{}, argmaxWorkerFn, .{&workers[t]});
            }
            for (0..num_threads) |t| {
                threads[t].join();
            }

            var best_id: usize = 0;
            var best_val: f32 = -1e30;
            for (workers) |w| {
                if (w.best_val > best_val) {
                    best_val = w.best_val;
                    best_id = w.best_id;
                }
            }

            return best_id;
        }
    };
}

/// 默认配置的 Gemma 4 12B 模型类型
pub const DefaultGemma4 = Gemma4ForCausalLM(Gemma4Config.default);

/// 默认配置的 Gemma 4 12B 4-bit 量化模型类型
pub const DefaultGemma4Q4 = Gemma4Q4ForCausalLM(Gemma4Config.default);

/// 测试配置的微型 Gemma 4 模型类型
pub const TinyGemma4 = Gemma4ForCausalLM(Gemma4Config.tiny_test);

/// 测试配置的微型 Q4 量化 Gemma 4 模型类型
pub const TinyQ4Gemma4 = Gemma4Q4ForCausalLM(Gemma4Config.tiny_test);

/// Gemma 4 完整 262K 词表零拷贝解码器 (Gemma4Vocabulary)
/// 支持直接映射二进制词表文件，以 O(1) 性能根据 Token ID 返回具体文本内容
pub const Gemma4Vocabulary = struct {
    mmap_ptr: []const u8,
    offsets: []const u32,
    str_data: []const u8,
    count: usize,

    pub const MAGIC: *const [8]u8 = "ZNNVOC01";

    pub fn loadFromMmap(mmap_bytes: []const u8) !Gemma4Vocabulary {
        if (mmap_bytes.len < 16) return error.InvalidVocabFile;
        if (!std.mem.eql(u8, mmap_bytes[0..8], MAGIC)) return error.InvalidVocabMagic;

        const count = std.mem.readInt(u32, mmap_bytes[8..12], .little);
        const total_str_bytes = std.mem.readInt(u32, mmap_bytes[12..16], .little);

        const offsets_start: usize = 16;
        const offsets_end: usize = offsets_start + count * @sizeOf(u32);
        if (mmap_bytes.len < offsets_end + total_str_bytes) return error.VocabTruncated;

        const offsets_bytes = mmap_bytes[offsets_start..offsets_end];
        const offsets = @as([]const u32, @alignCast(std.mem.bytesAsSlice(u32, offsets_bytes)));
        const str_data = mmap_bytes[offsets_end .. offsets_end + total_str_bytes];

        return Gemma4Vocabulary{
            .mmap_ptr = mmap_bytes,
            .offsets = offsets,
            .str_data = str_data,
            .count = count,
        };
    }

    pub fn openFile(io: std.Io, path: []const u8) !Gemma4Vocabulary {
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.openFile(io, path, .{});
        defer file.close(io);

        const stat = try file.stat(io);
        const mmap_ptr = try std.posix.mmap(
            null,
            @as(usize, @intCast(stat.size)),
            .{ .READ = true },
            .{ .TYPE = .SHARED },
            file.handle,
            0,
        );

        return loadFromMmap(mmap_ptr);
    }

    pub fn deinit(self: *Gemma4Vocabulary) void {
        std.posix.munmap(@alignCast(self.mmap_ptr));
    }

    /// O(1) 根据 token_id 获取对应解码文本切片
    pub fn decode(self: *const Gemma4Vocabulary, token_id: usize) []const u8 {
        if (token_id >= self.count) return "";
        const start = self.offsets[token_id];
        const end: usize = if (token_id + 1 < self.count) self.offsets[token_id + 1] else self.str_data.len;
        if (start > end or end > self.str_data.len) return "";
        return self.str_data[start..end];
    }
};
