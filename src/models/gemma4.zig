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

/// Gemma 4 专用键值缓存 (Key-Value Cache, KVCache)
/// 针对滑动窗口注意力 (Sliding Window Attention) 与全局全注意力 (Full Attention) 混合异构架构设计。
/// 每个解码层维护各自独立的键 (Key, K) 与值 (Value, V) 连续缓冲区。
pub const Gemma4KVCache = struct {
    pub const LayerCache = struct {
        k: []f32,
        v: []f32,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        attn_type: Gemma4AttentionType,
        curr_len: usize = 0,
        max_seq_len: usize,

        pub fn init(
            allocator: std.mem.Allocator,
            attn_type: Gemma4AttentionType,
            num_kv_heads: usize,
            head_dim: usize,
            sliding_window: usize,
            max_seq_len: usize,
        ) !LayerCache {
            const numel = num_kv_heads * max_seq_len * head_dim;
            const k = try allocator.alloc(f32, numel);
            errdefer allocator.free(k);
            const v = try allocator.alloc(f32, numel);
            @memset(k, 0.0);
            @memset(v, 0.0);
            return LayerCache{
                .k = k,
                .v = v,
                .num_kv_heads = num_kv_heads,
                .head_dim = head_dim,
                .sliding_window = sliding_window,
                .attn_type = attn_type,
                .curr_len = 0,
                .max_seq_len = max_seq_len,
            };
        }

        pub fn deinit(self: *LayerCache, allocator: std.mem.Allocator) void {
            allocator.free(self.k);
            allocator.free(self.v);
        }

        pub fn reset(self: *LayerCache) void {
            self.curr_len = 0;
        }
    };

    layers: []LayerCache,
    max_seq_len: usize,

    pub const Options = struct {
        max_seq_len: usize = 2048,

        pub const default: Options = .{};
        pub fn defaultOptions() Options {
            return .{};
        }
    };

    pub fn init(allocator: std.mem.Allocator, config: Gemma4Config, options: Options) !Gemma4KVCache {
        const layers = try allocator.alloc(LayerCache, config.num_hidden_layers);
        errdefer allocator.free(layers);

        var initialized_count: usize = 0;
        errdefer {
            for (layers[0..initialized_count]) |*l| {
                l.deinit(allocator);
            }
        }

        for (0..config.num_hidden_layers) |i| {
            const is_full = ((i + 1) % 6 == 0);
            const attn_type: Gemma4AttentionType = if (is_full) .full_attention else .sliding_attention;
            const head_dim = if (is_full) config.global_head_dim else config.head_dim;
            const num_kv = if (is_full) config.num_global_key_value_heads else config.num_key_value_heads;

            layers[i] = try LayerCache.init(
                allocator,
                attn_type,
                num_kv,
                head_dim,
                config.sliding_window,
                options.max_seq_len,
            );
            initialized_count += 1;
        }

        return Gemma4KVCache{
            .layers = layers,
            .max_seq_len = options.max_seq_len,
        };
    }

    pub fn initDefault(allocator: std.mem.Allocator, config: Gemma4Config) !Gemma4KVCache {
        return init(allocator, config, Options.default);
    }

    pub fn deinit(self: *Gemma4KVCache, allocator: std.mem.Allocator) void {
        for (self.layers) |*l| {
            l.deinit(allocator);
        }
        allocator.free(self.layers);
    }

    pub fn reset(self: *Gemma4KVCache) void {
        for (self.layers) |*l| {
            l.reset();
        }
    }
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
        // 当为全局注意力层 (full_attention) 或时间步不超过滑动窗口跨度时，等价于标准因果注意力，
        // 直接分发至 scaledDotProductAttention（无求导 GPU 模式直通融合 FlashAttention Metal 内核）；
        // 否则回退至带滑动窗口截断的分解计算图。
        const is_pure_causal = (self.attn_type == .full_attention or T <= self.sliding_window);
        const y_4d = if (is_pure_causal)
            try graph.scaledDotProductAttention(q_rot, k, v_final, 1.0, true)
        else blk: {
            const k_trans = try graph.transposeND(k, 2, 3);
            var scores = try graph.batchMatMul(q_rot, k_trans);

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
            break :blk try graph.batchMatMul(att_sm, v_final);
        };

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

/// 默认配置的 Gemma 4 12B 模型类型
pub const DefaultGemma4 = Gemma4ForCausalLM(Gemma4Config.default);

/// 测试配置的微型 Gemma 4 模型类型
pub const TinyGemma4 = Gemma4ForCausalLM(Gemma4Config.tiny_test);

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

    // 官方保留特殊控制词元 (Special Control Tokens)
    pub const PAD: usize = 0; // <pad>
    pub const EOS: usize = 1; // <eos> (End of Sequence)
    pub const BOS: usize = 2; // <bos> (Beginning of Sequence)
    pub const UNK: usize = 3; // <unk>
    pub const THINK: usize = 98; // <|think|>
    pub const SOC: usize = 100; // <|channel> (Start of Channel)
    pub const EOC: usize = 101; // <channel|> (End of Channel)
    pub const SOT: usize = 105; // <|turn> (Start of Turn)
    pub const EOT: usize = 106; // <turn|> (End of Turn)
    pub const NL: usize = 107; // \n
    pub const USER: usize = 2364; // user
    pub const MODEL: usize = 4368; // model
    pub const SYSTEM: usize = 9731; // system

    /// O(1) 根据 token_id 获取对应解码文本切片
    pub fn decode(self: *const Gemma4Vocabulary, token_id: usize) []const u8 {
        if (token_id >= self.count) return "";
        const start = self.offsets[token_id];
        const end: usize = if (token_id + 1 < self.count) self.offsets[token_id + 1] else self.str_data.len;
        if (start > end or end > self.str_data.len) return "";
        return self.str_data[start..end];
    }

    /// 在词表中根据确切文本反查对应的 Token ID (若不存在返回 null)
    pub fn findToken(self: *const Gemma4Vocabulary, text: []const u8) ?usize {
        for (0..self.count) |i| {
            if (std.mem.eql(u8, self.decode(i), text)) return i;
        }
        return null;
    }
};
