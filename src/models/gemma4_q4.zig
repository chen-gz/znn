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

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Executor = backend.Executor;
const Q4Block = tensor.quant.Q4Block;
const Linear = nn.core.Linear;
const Embedding = nn.transformer.Embedding;
const GemmaRMSNorm = gemma4.GemmaRMSNorm;
const GemmaUnscaledRMSNorm = gemma4.GemmaUnscaledRMSNorm;
const Gemma4Config = gemma4.Gemma4Config;
const Gemma4AttentionType = gemma4.Gemma4AttentionType;
const Gemma4KVCache = gemma4.Gemma4KVCache;
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
    q4_weights: ?backend.offload.Q4Weights = null,
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
            .q4_weights = null,
        };
    }

    pub fn fromBlocks(in_features: usize, out_features: usize, blocks: []const Q4Block) Q4Linear {
        return Q4Linear{
            .in_features = in_features,
            .out_features = out_features,
            .blocks = @constCast(blocks),
            .owns_blocks = false,
            .q4_weights = null,
        };
    }

    pub fn deinit(self: *Q4Linear, allocator: std.mem.Allocator) void {
        if (self.q4_weights) |*qw| {
            qw.deinit();
            self.q4_weights = null;
        }
        if (self.owns_blocks) {
            allocator.free(self.blocks);
        }
    }

    /// 零拷贝绑定 mmap 映射文件中的 Q4Block 权重块切片
    pub fn bindMmapBlocks(self: *Q4Linear, mmap_bytes: []const u8, cursor: *usize, allocator: std.mem.Allocator) void {
        self.deinit(allocator);
        const count = (self.in_features * self.out_features) / Q4Block.QK;
        const bytes = count * @sizeOf(Q4Block);
        self.blocks = @constCast(@as([]const Q4Block, @alignCast(std.mem.bytesAsSlice(Q4Block, mmap_bytes[cursor.* .. cursor.* + bytes]))));
        self.owns_blocks = false;
        cursor.* += bytes;
    }

    /// 启用图形处理器 (Graphics Processing Unit, GPU) 硬件加速卸载
    pub fn enableOffload(self: *Q4Linear, allocator: std.mem.Allocator, stream: ?backend.mlx.Stream) !void {
        if (!backend.offload.isAvailable()) return;
        if (self.q4_weights != null) return;
        const raw_bytes = std.mem.sliceAsBytes(self.blocks);
        self.q4_weights = try backend.offload.Q4Weights.initFromBlocks(
            allocator,
            raw_bytes,
            self.in_features,
            self.out_features,
            stream,
        );
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

    /// 前向传播：高效按块点积计算 x * W，在 GPU 硬件加速模式下直通 MLX 量化内核
    pub fn forward(self: *const Q4Linear, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        if (graph.options.device == .gpu and backend.offload.isAvailable() and !graph.enable_grad) {
            if (self.q4_weights) |*qw| {
                var total_elem: usize = 1;
                for (0..x.shape.len) |d| {
                    total_elem *= x.shape.dims[d];
                }
                const in_f = self.in_features;
                const out_f = self.out_features;
                const rows = total_elem / in_f;
                const out = if (x.shape.len == 2)
                    try graph.tensorND(&.{ rows, out_f }, false)
                else if (x.shape.len == 3)
                    try graph.tensorND(&.{ x.shape.dims[0], x.shape.dims[1], out_f }, false)
                else
                    try graph.tensorND(&.{ rows, out_f }, false);
                try qw.forward(x, out, graph.options.mlx_stream);
                return out;
            }
        }

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

        if (graph.options.device == .gpu and !graph.enable_grad) {
            if (self.gate_proj.q4_weights != null and self.up_proj.q4_weights != null and self.down_proj.q4_weights != null) {
                const out_2d = try graph.zeros(&.{ x_2d.shape.dims[0], self.gate_proj.in_features }, false);
                if (backend.offload.mlpForwardOffload(
                    &self.gate_proj.q4_weights.?,
                    &self.up_proj.q4_weights.?,
                    &self.down_proj.q4_weights.?,
                    x_2d,
                    out_2d,
                    graph.options.mlx_stream,
                )) {
                    if (is_3d) {
                        const B = old_shape.dims[0];
                        const T = old_shape.dims[1];
                        const D = old_shape.dims[2];
                        return try graph.reshape(out_2d, &.{ B, T, D });
                    }
                    return out_2d;
                } else |_| {}
            }
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
        return self.forwardPrefillOptional(graph, x, null);
    }

    /// 预填充阶段 (Prefill Phase)：处理初始输入的提示词元 (Prompt Tokens) 序列，
    /// 计算注意力并将该层所有的键值特征存入键值缓存 (Key-Value Cache, KVCache)
    pub fn forwardPrefill(
        self: *const Gemma4Q4Attention,
        graph: *autodiff.Graph,
        x: *Tensor,
        layer_cache: *Gemma4KVCache.LayerCache,
    ) !*Tensor {
        return self.forwardPrefillOptional(graph, x, layer_cache);
    }

    const QKVResult = struct {
        q: *Tensor,
        k: *Tensor,
        v: *Tensor,
    };

    /// 查询键值多头投影分发 (Query-Key-Value, QKV)
    /// 优先利用图形处理器 (Graphics Processing Unit, GPU) 批量分发量化矩阵乘法 (Quantized Matrix Multiplication, QMM)，
    /// 并执行单次多张量统一求值，降低主机-设备往返同步停顿。
    fn projectQKV(
        self: *const Gemma4Q4Attention,
        graph: *autodiff.Graph,
        x_2d: *Tensor,
    ) !QKVResult {
        const rows = x_2d.shape.dims[0];
        const nh = self.num_heads;
        const n_kv = self.num_kv_heads;
        const hd = self.head_dim;

        const q_raw = try graph.zeros(&.{ rows, nh * hd }, false);
        const k_raw = try graph.zeros(&.{ rows, n_kv * hd }, false);
        const v_raw = try graph.zeros(&.{ rows, n_kv * hd }, false);

        if (graph.options.device == .gpu and !graph.enable_grad) {
            if (self.q_proj.q4_weights != null and self.k_proj.q4_weights != null) {
                const v_qw = if (self.v_proj) |*vp| (if (vp.q4_weights) |*w| w else null) else null;
                if (backend.offload.qkvForwardOffload(
                    &self.q_proj.q4_weights.?,
                    &self.k_proj.q4_weights.?,
                    v_qw,
                    x_2d,
                    q_raw,
                    k_raw,
                    v_raw,
                    graph.options.mlx_stream,
                )) {
                    return .{ .q = q_raw, .k = k_raw, .v = v_raw };
                } else |_| {}
            }
        }

        const q_tmp = try self.q_proj.forward(graph, x_2d);
        @memcpy(q_raw.data, q_tmp.data);
        const k_tmp = try self.k_proj.forward(graph, x_2d);
        @memcpy(k_raw.data, k_tmp.data);
        if (self.v_proj) |*vp| {
            const v_tmp = try vp.forward(graph, x_2d);
            @memcpy(v_raw.data, v_tmp.data);
        } else {
            @memcpy(v_raw.data, k_tmp.data);
        }
        return .{ .q = q_raw, .k = k_raw, .v = v_raw };
    }

    fn forwardPrefillOptional(
        self: *const Gemma4Q4Attention,
        graph: *autodiff.Graph,
        x: *Tensor,
        layer_cache: ?*Gemma4KVCache.LayerCache,
    ) !*Tensor {
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
        const qkv = try self.projectQKV(graph, x_2d);
        const q_raw = qkv.q;
        const k_raw = qkv.k;
        const v_raw = qkv.v;

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

        // 5. 写入键值缓存 (Key-Value Cache, KVCache)
        if (layer_cache) |cache| {
            std.debug.assert(B == 1);
            std.debug.assert(T <= cache.max_seq_len);
            const max_len = cache.max_seq_len;
            for (0..n_kv) |kv_h| {
                for (0..T) |t| {
                    const dest_k = cache.k[(kv_h * max_len + t) * hd .. (kv_h * max_len + t + 1) * hd];
                    const dest_v = cache.v[(kv_h * max_len + t) * hd .. (kv_h * max_len + t + 1) * hd];
                    const src_k = k_rot.data[(kv_h * T + t) * hd .. (kv_h * T + t + 1) * hd];
                    const src_v = v.data[(kv_h * T + t) * hd .. (kv_h * T + t + 1) * hd];
                    @memcpy(dest_k, src_k);
                    @memcpy(dest_v, src_v);
                }
            }
            cache.curr_len = T;
        }

        // 6. GQA 广播
        var k = k_rot;
        var v_final = v;
        if (groups > 1) {
            k = try graph.repeatKV(k_rot, groups);
            v_final = try graph.repeatKV(v, groups);
        }

        // 7. 注意力得分与因果 / 滑动窗口掩码计算 (官方实现中 scaling = 1.0)
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
            break :blk try graph.batchMatMul(att_sm, v_final);
        };

        // 8. 转置并输出投影
        const y_trans = try graph.transposeND(y_4d, 1, 2);
        const y_2d = try graph.reshape(y_trans, &.{ B * T, nh * hd });
        const out_2d = try self.o_proj.forward(graph, y_2d);

        return try graph.reshape(out_2d, &.{ B, T, C });
    }

    /// 增量解码注意力特征图计算 (Decode Attention Feature Extraction)
    /// 输出投影前的原始注意力聚合特征图 y_2d [1, num_heads * head_dim]
    pub fn forwardDecodeAttention(
        self: *const Gemma4Q4Attention,
        graph: *autodiff.Graph,
        x: *Tensor,
        layer_cache: *Gemma4KVCache.LayerCache,
    ) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const C = self.hidden_size;
        const nh = self.num_heads;
        const n_kv = self.num_kv_heads;
        const hd = self.head_dim;
        const groups = nh / n_kv;
        const pos = layer_cache.curr_len;
        const max_len = layer_cache.max_seq_len;
        if (pos >= max_len) return error.ContextWindowExceeded;

        const x_2d = if (x.shape.len == 2) x else try graph.reshape(x, &.{ 1, C });

        // 1. Q4 投影单行向量 (GEMV)
        const qkv = try self.projectQKV(graph, x_2d);
        const q_raw = qkv.q;
        const k_raw = qkv.k;
        const v_raw = qkv.v;

        // 2. Q-Norm 与 K-Norm，以及无参数 v_norm
        const q_heads = try graph.reshape(q_raw, &.{ nh, hd });
        const q_normed = try self.q_norm.forward(graph, q_heads);
        const k_heads = try graph.reshape(k_raw, &.{ n_kv, hd });
        const k_normed = try self.k_norm.forward(graph, k_heads);
        const v_heads = try graph.reshape(v_raw, &.{ n_kv, hd });
        const v_normed = try self.v_norm.forward(graph, v_heads);

        // 3. 施加半切分旋转位置编码 (Split-Half RoPE, 当前时间步为 pos)
        const q_4d = try graph.reshape(q_normed, &.{ 1, nh, 1, hd });
        const k_4d = try graph.reshape(k_normed, &.{ 1, n_kv, 1, hd });
        const q_rot = try ropeSplitHalf(graph, q_4d, pos, self.partial_rotary_factor, self.rope_theta);
        const k_rot = try ropeSplitHalf(graph, k_4d, pos, self.partial_rotary_factor, self.rope_theta);

        // 4. 写入键值缓存 (Key-Value Cache, KVCache)
        for (0..n_kv) |kv_h| {
            const dest_k = layer_cache.k[(kv_h * max_len + pos) * hd .. (kv_h * max_len + pos + 1) * hd];
            const dest_v = layer_cache.v[(kv_h * max_len + pos) * hd .. (kv_h * max_len + pos + 1) * hd];
            const src_k = k_rot.data[kv_h * hd .. (kv_h + 1) * hd];
            const src_v = v_normed.data[kv_h * hd .. (kv_h + 1) * hd];
            @memcpy(dest_k, src_k);
            @memcpy(dest_v, src_v);
        }
        layer_cache.curr_len = pos + 1;
        const curr_len = layer_cache.curr_len;

        // 5. 单查询点积注意力 (Single Query Attention against Cached Keys & Values)
        const y_2d = try graph.zeros(&.{ 1, nh * hd }, false);
        const start_idx: usize = if (self.attn_type == .sliding_attention and curr_len > self.sliding_window)
            curr_len - self.sliding_window
        else
            0;
        const valid_len = curr_len - start_idx;
        const scores = try graph.arenaAllocator().alloc(f32, valid_len);

        for (0..nh) |h| {
            const kv_h = h / groups;
            const q_head = q_rot.data[h * hd .. (h + 1) * hd];
            var max_score: f32 = -1e30;

            for (start_idx..curr_len) |j| {
                const k_cached = layer_cache.k[(kv_h * max_len + j) * hd .. (kv_h * max_len + j + 1) * hd];
                var dot: f32 = 0.0;
                for (q_head, k_cached) |qv, kv| {
                    dot += qv * kv;
                }
                scores[j - start_idx] = dot;
                if (dot > max_score) max_score = dot;
            }

            var exp_sum: f32 = 0.0;
            for (scores) |*s| {
                const e = @exp(s.* - max_score);
                s.* = e;
                exp_sum += e;
            }
            const inv_sum = 1.0 / exp_sum;
            for (scores) |*s| {
                s.* *= inv_sum;
            }

            const y_head = y_2d.data[h * hd .. (h + 1) * hd];
            @memset(y_head, 0.0);
            for (start_idx..curr_len) |j| {
                const v_cached = layer_cache.v[(kv_h * max_len + j) * hd .. (kv_h * max_len + j + 1) * hd];
                const weight = scores[j - start_idx];
                for (y_head, v_cached) |*yv, vv| {
                    yv.* += weight * vv;
                }
            }
        }

        return y_2d;
    }

    /// 增量解码阶段 (Decode Phase)：基于键值缓存 (Key-Value Cache, KVCache) 处理单个增量词元 (Token)
    pub fn forwardDecode(
        self: *const Gemma4Q4Attention,
        graph: *autodiff.Graph,
        x: *Tensor,
        layer_cache: *Gemma4KVCache.LayerCache,
    ) !*Tensor {
        const y_2d = try self.forwardDecodeAttention(graph, x, layer_cache);
        const out_2d = try self.o_proj.forward(graph, y_2d);
        return try graph.reshape(out_2d, &.{ 1, 1, self.hidden_size });
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

    /// 为当前解码层所有 4-bit 量化投影启用 GPU 硬件加速卸载
    pub fn enableHardwareOffload(self: *Gemma4Q4DecoderLayer, allocator: std.mem.Allocator) !void {
        try self.self_attn.q_proj.enableOffload(allocator, null);
        try self.self_attn.k_proj.enableOffload(allocator, null);
        if (self.self_attn.v_proj) |*vp| try vp.enableOffload(allocator, null);
        try self.self_attn.o_proj.enableOffload(allocator, null);
        try self.mlp.gate_proj.enableOffload(allocator, null);
        try self.mlp.up_proj.enableOffload(allocator, null);
        try self.mlp.down_proj.enableOffload(allocator, null);
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

    pub fn forwardPrefill(
        self: *const Gemma4Q4DecoderLayer,
        graph: *autodiff.Graph,
        x: *Tensor,
        layer_cache: *Gemma4KVCache.LayerCache,
    ) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const in_norm = try self.input_layernorm.forward(graph, x);
        const attn_out = try self.self_attn.forwardPrefill(graph, in_norm, layer_cache);
        const post_attn = try self.post_attention_layernorm.forward(graph, attn_out);
        const x1 = try graph.add(x, post_attn);

        const pre_mlp = try self.pre_feedforward_layernorm.forward(graph, x1);
        const mlp_out = try self.mlp.forward(graph, pre_mlp);
        const post_mlp = try self.post_feedforward_layernorm.forward(graph, mlp_out);
        const x2 = try graph.add(x1, post_mlp);
        return try graph.mulScalar(x2, self.layer_scalar.data[0]);
    }

    pub fn forwardDecode(
        self: *const Gemma4Q4DecoderLayer,
        graph: *autodiff.Graph,
        x: *Tensor,
        layer_cache: *Gemma4KVCache.LayerCache,
    ) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const in_norm = try self.input_layernorm.forward(graph, x);
        const attn_out = try self.self_attn.forwardDecode(graph, in_norm, layer_cache);
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
        lm_head_weights: ?backend.offload.LmHeadWeights = null,
        gpu_context: ?backend.offload.GpuModelContext = null,
        gpu_needs_sync: bool = true,

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
                .lm_head_weights = null,
                .gpu_context = null,
                .gpu_needs_sync = true,
            };
        }

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

            for (0..config.num_hidden_layers) |layer_idx| {
                const is_full = ((layer_idx + 1) % 6 == 0);
                const head_dim = if (is_full) config.global_head_dim else config.head_dim;
                const num_kv = if (is_full) config.num_global_key_value_heads else config.num_key_value_heads;
                const norms_bytes = (config.hidden_size * 4 + head_dim * 2 + 1) * 2; // BF16

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

            // 2. 词表嵌入矩阵 (262,144 x 3840 个 BF16 保持在 mmap 与 GPU 统一显存中)
            // 避免在 CPU 堆内存上分配 4.03 GB 的冗余 F32 Tensor，自回归直接利用零拷贝映射
            const embed_tokens = try Embedding.init(allocator, 1, 1);
            cursor += config.vocab_size * config.hidden_size * 2;

            // 3. 逐层加载 48 层 Transformer
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

                // 加载当前层 4 个 Layernorm 参数 (BF16)
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
                    const u16_s: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, mmap_bytes[cursor .. cursor + head_dim * 2]));
                    for (u16_s, 0..) |bits, idx| {
                        const b: tensor.bf16 = .{ .bits = bits };
                        ln.weight.data[idx] = b.toF32();
                    }
                    cursor += head_dim * 2;
                }

                // 加载 layer_scalar (BF16)
                {
                    const sc_bits = std.mem.readInt(u16, mmap_bytes[cursor .. cursor + 2][0..2], .little);
                    const b_sc: tensor.bf16 = .{ .bits = sc_bits };
                    layer.layer_scalar.data[0] = b_sc.toF32();
                    cursor += 2;
                }

                // 零拷贝绑定 Q4Block 分块数组
                layer.self_attn.q_proj.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                layer.self_attn.k_proj.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                if (!is_full) {
                    layer.self_attn.v_proj.?.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                }
                layer.self_attn.o_proj.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                layer.mlp.gate_proj.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                layer.mlp.up_proj.bindMmapBlocks(mmap_bytes, &cursor, allocator);
                layer.mlp.down_proj.bindMmapBlocks(mmap_bytes, &cursor, allocator);

                layers[i] = layer;
            }

            var lm_head_weights: ?backend.offload.LmHeadWeights = null;
            if (backend.offload.isAvailable()) {
                const embed_u16_slice: []const u16 = @alignCast(std.mem.bytesAsSlice(
                    u16,
                    mmap_bytes[8 + config.hidden_size * 2 .. 8 + config.hidden_size * 2 + config.vocab_size * config.hidden_size * 2],
                ));
                lm_head_weights = backend.offload.LmHeadWeights.init(embed_u16_slice, config.vocab_size, config.hidden_size, null) catch null;

                for (layers) |*layer| {
                    layer.enableHardwareOffload(allocator) catch {};
                }

                // 通知操作系统内核丢弃已卸载至 GPU 显存的 4-bit 权重文件页缓存，防止 7.59 GB 物理内存重复驻留造成 macOS Swap 严重换页
                const q4_weights_start = 8 + config.hidden_size * 2 + config.vocab_size * config.hidden_size * 2;
                const page_size: usize = 16384;
                const aligned_start = (q4_weights_start / page_size) * page_size;
                const aligned_len = ((mmap_bytes.len - aligned_start) / page_size) * page_size;
                if (aligned_len > 0) {
                    const ptr: *align(16384) anyopaque = @ptrCast(@alignCast(@constCast(mmap_bytes.ptr + aligned_start)));
                    _ = std.posix.system.madvise(ptr, aligned_len, std.posix.MADV.DONTNEED);
                }
            }

            var res = Self{
                .embed_tokens = embed_tokens,
                .layers = layers,
                .norm = norm,
                .lm_head_weights = lm_head_weights,
            };
            res.initGpuContext(allocator) catch {};
            return res;
        }

        /// 初始化图形处理器 (Graphics Processing Unit, GPU) 端到端整网自回归增量解码上下文
        pub fn initGpuContext(self: *Self, allocator: std.mem.Allocator) !void {
            if (!backend.offload.isAvailable() or self.lm_head_weights == null) return;

            const gpu_layers = try allocator.alloc(backend.offload.GpuLayerWeights, self.layers.len);
            defer allocator.free(gpu_layers);

            for (self.layers, 0..) |*layer, i| {
                const q_qw = &(layer.self_attn.q_proj.q4_weights orelse return);
                const k_qw = &(layer.self_attn.k_proj.q4_weights orelse return);
                const v_qw = if (layer.self_attn.v_proj) |*vp| (if (vp.q4_weights) |*w| w else null) else null;
                const o_qw = &(layer.self_attn.o_proj.q4_weights orelse return);
                const gate_qw = &(layer.mlp.gate_proj.q4_weights orelse return);
                const up_qw = &(layer.mlp.up_proj.q4_weights orelse return);
                const down_qw = &(layer.mlp.down_proj.q4_weights orelse return);
                const hd = layer.self_attn.head_dim;

                gpu_layers[i] = try backend.offload.GpuLayerWeights.init(
                    q_qw,
                    k_qw,
                    v_qw,
                    o_qw,
                    gate_qw,
                    up_qw,
                    down_qw,
                    layer.input_layernorm.weight.data,
                    layer.self_attn.q_norm.weight.data[0..hd],
                    layer.self_attn.k_norm.weight.data[0..hd],
                    layer.post_attention_layernorm.weight.data,
                    layer.pre_feedforward_layernorm.weight.data,
                    layer.post_feedforward_layernorm.weight.data,
                    layer.layer_scalar.data[0],
                    if (layer.self_attn.attn_type == .full_attention) 1 else 0,
                    layer.self_attn.sliding_window,
                    layer.self_attn.num_heads,
                    layer.self_attn.num_kv_heads,
                    hd,
                    layer.self_attn.rope_theta,
                    layer.self_attn.partial_rotary_factor,
                    config.rms_norm_eps,
                );
            }

            self.gpu_context = try backend.offload.GpuModelContext.init(
                allocator,
                gpu_layers,
                self.norm.weight.data,
                config.rms_norm_eps,
                self.lm_head_weights.?,
                config.hidden_size,
                config.vocab_size,
                config.max_position_embeddings,
            );
            self.gpu_needs_sync = true;
        }

        /// 为全模型所有 4-bit 量化层启用图形处理器 (Graphics Processing Unit, GPU) 硬件加速卸载
        pub fn enableHardwareOffload(self: *Self, allocator: std.mem.Allocator) !void {
            if (!backend.offload.isAvailable()) return;
            for (self.layers) |*layer| {
                try layer.self_attn.q_proj.enableOffload(allocator, null);
                try layer.self_attn.k_proj.enableOffload(allocator, null);
                if (layer.self_attn.v_proj) |*vp| try vp.enableOffload(allocator, null);
                try layer.self_attn.o_proj.enableOffload(allocator, null);
                try layer.mlp.gate_proj.enableOffload(allocator, null);
                try layer.mlp.up_proj.enableOffload(allocator, null);
                try layer.mlp.down_proj.enableOffload(allocator, null);
            }
            try self.initGpuContext(allocator);
        }

        const BestToken = struct {
            id: usize,
            val: f32,
        };

        const ArgmaxContext = struct {
            embed_u16: []const u16,
            h_last: []const f32,
            hidden_size: usize,
            recent_bitset: *const [4096]u64,
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

                    if (ctx.repetition_penalty > 1.0 and (ctx.recent_bitset[v / 64] & (@as(u64, 1) << @intCast(v % 64))) != 0) {
                        if (dot > 0.0) {
                            dot /= ctx.repetition_penalty;
                        } else {
                            dot *= ctx.repetition_penalty;
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

        fn sampleNextToken(
            self: *const Self,
            h_last: []const f32,
            embed_u16: []const u16,
            recent_tokens: []const usize,
            repetition_penalty: f32,
            graph: *autodiff.Graph,
        ) !usize {
            const recent_window = if (recent_tokens.len > 64) recent_tokens[recent_tokens.len - 64 ..] else recent_tokens;

            // 1. 优先通过 GPU 卸载执行巨型词表投影与采样 (消除 2.3 秒延迟至 5~10 ms)
            if (graph.options.device == .gpu and backend.offload.isAvailable()) {
                if (self.lm_head_weights) |weights| {
                    if (weights.sampleNext(h_last, recent_window, repetition_penalty, graph.options.mlx_stream)) |tok| {
                        return tok;
                    } else |err| {
                        std.debug.print(" [LM_HEAD_WEIGHTS_ERROR: {}] ", .{err});
                    }
                } else {
                    std.debug.print(" [LM_HEAD_WEIGHTS_IS_NULL!] ", .{});
                    if (backend.offload.lmHeadArgmax(h_last, embed_u16, config.vocab_size, config.hidden_size, recent_window, repetition_penalty, graph.options.mlx_stream)) |tok| {
                        return tok;
                    } else |err| {
                        std.debug.print(" [LM_HEAD_ARGMAX_ERROR: {}] ", .{err});
                    }
                }
            }

            // 2. CPU 确定性多核并行回退路径
            var recent_bitset = [_]u64{0} ** (262144 / 64);
            if (repetition_penalty > 1.0) {
                for (recent_window) |rec| {
                    if (rec < config.vocab_size) {
                        recent_bitset[rec / 64] |= (@as(u64, 1) << @intCast(rec % 64));
                    }
                }
            }

            const ctx = ArgmaxContext{
                .embed_u16 = embed_u16,
                .h_last = h_last,
                .hidden_size = config.hidden_size,
                .recent_bitset = &recent_bitset,
                .repetition_penalty = repetition_penalty,
            };

            const grain: usize = 4096;
            const best = try graph.executor().parallelReduce(
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
            return self.sampleNextToken(last_row, embed_u16, token_ids, repetition_penalty, graph);
        }

        /// 预填充阶段 (Prefill)：处理 Prompt 词元序列，写入键值缓存 (Key-Value Cache, KVCache)，并返回首个预测词元
        pub fn prefill(
            self: *Self,
            graph: *autodiff.Graph,
            prompt_tokens: []const usize,
            cache: *Gemma4KVCache,
            mmap_bytes: []const u8,
            repetition_penalty: f32,
        ) !usize {
            const T = prompt_tokens.len;
            if (T == 0) return 0;
            cache.reset();
            self.gpu_needs_sync = true;

            const embed_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(
                u16,
                mmap_bytes[8 + config.hidden_size * 2 .. 8 + config.hidden_size * 2 + config.vocab_size * config.hidden_size * 2],
            ));

            // 1. 词表嵌入查表
            const tok_emb = try graph.tensorND(&.{ 1, T, config.hidden_size }, false);
            const sqrt_d: f32 = @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
            for (prompt_tokens, 0..) |tok, t_idx| {
                const emb_row = embed_u16[tok * config.hidden_size .. (tok + 1) * config.hidden_size];
                const dest = tok_emb.data[t_idx * config.hidden_size .. (t_idx + 1) * config.hidden_size];
                for (0..config.hidden_size) |d| {
                    const b: tensor.bf16 = .{ .bits = emb_row[d] };
                    dest[d] = b.toF32() * sqrt_d;
                }
            }

            // 2. 逐层预填充前向传播并写入键值缓存
            var h = tok_emb;
            for (self.layers, 0..) |*layer, l_idx| {
                h = try layer.forwardPrefill(graph, h, &cache.layers[l_idx]);
            }

            // 3. 最终层 RMSNorm
            const h_norm = try self.norm.forward(graph, h);
            const last_row = h_norm.data[(T - 1) * config.hidden_size .. T * config.hidden_size];

            return self.sampleNextToken(last_row, embed_u16, prompt_tokens, repetition_penalty, graph);
        }

        /// 增量解码阶段 (Decode Step)：输入最新生成的单个词元 ID，基于键值缓存 (Key-Value Cache, KVCache) 高效自回归生成下一个词元
        pub fn decodeStep(
            self: *Self,
            graph: *autodiff.Graph,
            token: usize,
            cache: *Gemma4KVCache,
            mmap_bytes: []const u8,
            recent_tokens: []const usize,
            repetition_penalty: f32,
        ) !usize {
            const embed_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(
                u16,
                mmap_bytes[8 + config.hidden_size * 2 .. 8 + config.hidden_size * 2 + config.vocab_size * config.hidden_size * 2],
            ));

            if (graph.options.device == .gpu and self.gpu_context != null) {
                var gpu_ctx = &self.gpu_context.?;
                if (self.gpu_needs_sync) {
                    const k_slices = try graph.arenaAllocator().alloc([]const f32, self.layers.len);
                    const v_slices = try graph.arenaAllocator().alloc([]const f32, self.layers.len);
                    for (cache.layers, 0..) |*cl, idx| {
                        k_slices[idx] = cl.k;
                        v_slices[idx] = cl.v;
                    }
                    try gpu_ctx.syncCacheFromCpu(k_slices, v_slices, cache.layers[0].curr_len);
                    self.gpu_needs_sync = false;
                }

                const sqrt_d: f32 = @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
                const emb_row = embed_u16[token * config.hidden_size .. (token + 1) * config.hidden_size];
                var tok_emb_buf: [config.hidden_size]f32 = undefined;
                for (0..config.hidden_size) |d| {
                    const b: tensor.bf16 = .{ .bits = emb_row[d] };
                    tok_emb_buf[d] = b.toF32() * sqrt_d;
                }

                const recent_window = if (recent_tokens.len > 64) recent_tokens[recent_tokens.len - 64 ..] else recent_tokens;
                const next_tok = try gpu_ctx.decodeStep(&tok_emb_buf, recent_window, repetition_penalty, graph.options.mlx_stream);
                for (cache.layers) |*cl| {
                    cl.curr_len += 1;
                }
                return next_tok;
            }

            // 1. 单个词元嵌入查表 [1, 1, hidden_size]
            const tok_emb = try graph.tensorND(&.{ 1, 1, config.hidden_size }, false);
            const sqrt_d: f32 = @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
            const emb_row = embed_u16[token * config.hidden_size .. (token + 1) * config.hidden_size];
            for (0..config.hidden_size) |d| {
                const b: tensor.bf16 = .{ .bits = emb_row[d] };
                tok_emb.data[d] = b.toF32() * sqrt_d;
            }

            // 2. 逐层增量解码传播
            var h = tok_emb;
            for (self.layers, 0..) |*layer, l_idx| {
                h = try layer.forwardDecode(graph, h, &cache.layers[l_idx]);
            }

            // 3. 最终层 RMSNorm
            const h_norm = try self.norm.forward(graph, h);
            const last_row = h_norm.data[0..config.hidden_size];

            return self.sampleNextToken(last_row, embed_u16, recent_tokens, repetition_penalty, graph);
        }
    };
}

/// 默认配置的 Gemma 4 12B 4-bit 量化模型类型
pub const DefaultGemma4Q4 = Gemma4Q4ForCausalLM(Gemma4Config.default);

/// 测试配置的微型 Q4 量化 Gemma 4 模型类型
pub const TinyQ4Gemma4 = Gemma4Q4ForCausalLM(Gemma4Config.tiny_test);
