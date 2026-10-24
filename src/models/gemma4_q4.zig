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
const gemma4_gpu = @import("gemma4_gpu.zig");

const Tensor = tensor.Tensor;
const Q4Block = tensor.quant.Q4Block;
const Embedding = nn.transformer.Embedding;
const GemmaRMSNorm = gemma4.GemmaRMSNorm;
const GemmaUnscaledRMSNorm = gemma4.GemmaUnscaledRMSNorm;
const Gemma4Config = gemma4.Gemma4Config;
const Gemma4AttentionType = gemma4.Gemma4AttentionType;
const Gemma4KVCache = gemma4.Gemma4KVCache;
const vec_math = backend.vec_math;
const LmHeadWeights = backend.offload.LmHeadWeights;
const Sampler = backend.sampling.Sampler;
const Gemma4GpuLayer = gemma4_gpu.Gemma4GpuLayer;
const Gemma4GpuContext = gemma4_gpu.Gemma4GpuContext;

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
            vec_math.widenBf16(&f32_chunk, bf16_weights[b * 32 ..][0..32], 1.0);
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
    /// 高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 形式 (Gemma 4 为双曲正切近似形式)
    hidden_activation: backend.GeluApproximation,
    name: ?[]const u8 = null,
    module_type: []const u8 = "Gemma4Q4MLP",

    pub fn init(
        allocator: std.mem.Allocator,
        hidden_size: usize,
        intermediate_size: usize,
        hidden_activation: backend.GeluApproximation,
    ) !Gemma4Q4MLP {
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
            .hidden_activation = hidden_activation,
        };
    }

    /// 构建无 CPU 权重实体的空 MLP 结构 (用于 GPU 硬件加速直通路径，避免 4.5 GB 内存分配与零填充)
    pub fn initEmpty(hidden_size: usize, intermediate_size: usize, hidden_activation: backend.GeluApproximation) Gemma4Q4MLP {
        return Gemma4Q4MLP{
            .gate_proj = Q4Linear.fromBlocks(hidden_size, intermediate_size, &.{}),
            .up_proj = Q4Linear.fromBlocks(hidden_size, intermediate_size, &.{}),
            .down_proj = Q4Linear.fromBlocks(intermediate_size, hidden_size, &.{}),
            .hidden_activation = hidden_activation,
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
                    self.hidden_activation,
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
        const gelu_gate = try graph.gelu(gate, self.hidden_activation);
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
    rope: tensor.RopeOptions,

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

        var q_proj = try Q4Linear.init(allocator, hidden_size, q_dim);
        errdefer q_proj.deinit(allocator);
        var k_proj = try Q4Linear.init(allocator, hidden_size, kv_dim);
        errdefer k_proj.deinit(allocator);
        var v_proj = if (attn_type == .full_attention) null else try Q4Linear.init(allocator, hidden_size, kv_dim);
        errdefer if (v_proj) |*vp| vp.deinit(allocator);
        var o_proj = try Q4Linear.init(allocator, q_dim, hidden_size);
        errdefer o_proj.deinit(allocator);

        const q_norm = try GemmaRMSNorm.init(allocator, head_dim, rms_norm_eps);
        errdefer deinitModel(&q_norm, allocator);
        const k_norm = try GemmaRMSNorm.init(allocator, head_dim, rms_norm_eps);
        errdefer deinitModel(&k_norm, allocator);
        const v_norm = GemmaUnscaledRMSNorm.init(rms_norm_eps);

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
            .rope = attn_type.ropeOptions(),
        };
    }

    /// 构建无 CPU 权重实体的空注意力机制结构 (用于 GPU 硬件加速直通路径，避免数 GB 内存分配与零填充)
    pub fn initEmpty(
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

        const q_proj = Q4Linear.fromBlocks(hidden_size, q_dim, &.{});
        const k_proj = Q4Linear.fromBlocks(hidden_size, kv_dim, &.{});
        const v_proj = if (attn_type == .full_attention) null else Q4Linear.fromBlocks(hidden_size, kv_dim, &.{});
        const o_proj = Q4Linear.fromBlocks(q_dim, hidden_size, &.{});

        const q_norm = try GemmaRMSNorm.init(allocator, head_dim, rms_norm_eps);
        errdefer deinitModel(&q_norm, allocator);
        const k_norm = try GemmaRMSNorm.init(allocator, head_dim, rms_norm_eps);
        errdefer deinitModel(&k_norm, allocator);
        const v_norm = GemmaUnscaledRMSNorm.init(rms_norm_eps);

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
            .rope = attn_type.ropeOptions(),
        };
    }

    pub fn deinit(self: *Gemma4Q4Attention, allocator: std.mem.Allocator) void {
        self.q_proj.deinit(allocator);
        self.k_proj.deinit(allocator);
        if (self.v_proj) |*vp| vp.deinit(allocator);
        self.o_proj.deinit(allocator);
        deinitModel(&self.q_norm, allocator);
        deinitModel(&self.k_norm, allocator);
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
        const q_rot = try graph.rope(q, 0, self.rope);
        const k_rot = try graph.rope(k_t_unrot, 0, self.rope);

        // 5. 写入键值缓存 (Key-Value Cache, KVCache)
        if (layer_cache) |cache| try gemma4.writePrefillKVCache(cache, k_rot.data, v.data, B, T);

        // 6. 注意力得分与因果 / 滑动窗口掩码计算 (官方实现中 scaling = 1.0，分组查询注意力直接按 kv_groups 索引)
        const y_4d = try graph.attention(q_rot, k_rot, v, .{
            .scale = 1.0,
            .mask = self.attn_type.mask(self.sliding_window),
            .kv_groups = groups,
        });

        // 7. 转置并输出投影
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
        const pos = layer_cache.curr_len;
        if (pos >= layer_cache.max_seq_len) return error.ContextWindowExceeded;

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
        const q_rot = try graph.rope(q_4d, pos, self.rope);
        const k_rot = try graph.rope(k_4d, pos, self.rope);

        // 4. 写入键值缓存并执行单查询点积注意力 (Single Query Attention against Cached Keys & Values)
        return gemma4.decodeAttention(graph, layer_cache, self.attn_type, self.sliding_window, nh, q_rot.data, k_rot.data, v_normed.data);
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
        hidden_activation: backend.GeluApproximation,
    ) !Gemma4Q4DecoderLayer {
        const input_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&input_layernorm, allocator);
        var self_attn = try Gemma4Q4Attention.init(
            allocator,
            attn_type,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            rms_norm_eps,
        );
        errdefer self_attn.deinit(allocator);
        const post_attention_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&post_attention_layernorm, allocator);
        const pre_feedforward_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&pre_feedforward_layernorm, allocator);
        var mlp = try Gemma4Q4MLP.init(allocator, hidden_size, intermediate_size, hidden_activation);
        errdefer mlp.deinit(allocator);
        const post_feedforward_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&post_feedforward_layernorm, allocator);

        const layer_scalar = try gemma4.createLayerScalar(allocator);

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

    /// 构建无 CPU 权重实体的空解码层结构 (用于 GPU 硬件加速直通路径，彻底消除 4.5 GB 内存分配与零填充)
    pub fn initEmpty(
        allocator: std.mem.Allocator,
        attn_type: Gemma4AttentionType,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        rms_norm_eps: f32,
        hidden_activation: backend.GeluApproximation,
    ) !Gemma4Q4DecoderLayer {
        const input_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&input_layernorm, allocator);
        var self_attn = try Gemma4Q4Attention.initEmpty(
            allocator,
            attn_type,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            rms_norm_eps,
        );
        errdefer self_attn.deinit(allocator);
        const post_attention_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&post_attention_layernorm, allocator);
        const pre_feedforward_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&pre_feedforward_layernorm, allocator);
        const mlp = Gemma4Q4MLP.initEmpty(hidden_size, intermediate_size, hidden_activation);
        const post_feedforward_layernorm = try GemmaRMSNorm.init(allocator, hidden_size, rms_norm_eps);
        errdefer deinitModel(&post_feedforward_layernorm, allocator);

        const layer_scalar = try gemma4.createLayerScalar(allocator);

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

    /// 注意力之后的公共残差尾部：PostAttnLN -> + x -> PreMLPLN -> MLP -> PostMLPLN -> + x1 -> * layer_scalar
    fn residualTail(self: *const Gemma4Q4DecoderLayer, graph: *autodiff.Graph, x: *Tensor, attn_out: *Tensor) !*Tensor {
        const post_attn = try self.post_attention_layernorm.forward(graph, attn_out);
        const x1 = try graph.add(x, post_attn);

        const pre_mlp = try self.pre_feedforward_layernorm.forward(graph, x1);
        const mlp_out = try self.mlp.forward(graph, pre_mlp);
        const post_mlp = try self.post_feedforward_layernorm.forward(graph, mlp_out);
        const x2 = try graph.add(x1, post_mlp);
        return try graph.mulScalar(x2, self.layer_scalar.data[0]);
    }

    pub fn forward(self: *const Gemma4Q4DecoderLayer, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const in_norm = try self.input_layernorm.forward(graph, x);
        const attn_out = try self.self_attn.forward(graph, in_norm);
        return self.residualTail(graph, x, attn_out);
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
        return self.residualTail(graph, x, attn_out);
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
        return self.residualTail(graph, x, attn_out);
    }
};

/// 4-bit 量化因果语言模型 (Gemma4Q4ForCausalLM)
/// 整体内存降低至原始全精度的 1/4 到 1/8，使得 12B 模型能够在 16GB 设备上完整运行！
pub fn Gemma4Q4ForCausalLM(comptime cfg: Gemma4Config) type {
    return struct {
        pub const config: Gemma4Config = cfg;
        const Self = @This();

        embed_tokens: Embedding,
        embed_u16_mmap: ?[]const u16 = null,
        layers: []Gemma4Q4DecoderLayer,
        norm: GemmaRMSNorm,
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

            const norm = try GemmaRMSNorm.init(allocator, config.hidden_size, config.rms_norm_eps);

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
                vec_math.widenBf16Bytes(ln.weight.data, bytes[cursor .. cursor + H * 2]);
                cursor += H * 2;
            }
            inline for (.{ &layer.self_attn.q_norm, &layer.self_attn.k_norm }) |ln| {
                vec_math.widenBf16Bytes(ln.weight.data[0..head_dim], bytes[cursor .. cursor + head_dim * 2]);
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
                .input = layer.input_layernorm.weight.data,
                .q = layer.self_attn.q_norm.weight.data[0..hd],
                .k = layer.self_attn.k_norm.weight.data[0..hd],
                .post_attn = layer.post_attention_layernorm.weight.data,
                .pre_mlp = layer.pre_feedforward_layernorm.weight.data,
                .post_mlp = layer.post_feedforward_layernorm.weight.data,
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
            const d = gemma4_gpu.ProjectionDims.of(config, i);
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
            const norm = try GemmaRMSNorm.init(allocator, config.hidden_size, config.rms_norm_eps);
            errdefer deinitModel(&norm, allocator);
            vec_math.widenBf16Bytes(norm.weight.data, mmap_bytes[magic_bytes..embed_offset]);

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
                    norm.weight.data,
                    lm_head_weights,
                    config.max_position_embeddings,
                );
                gpu_owned_here = false;
                var res = Self{
                    .embed_tokens = embed_tokens,
                    .embed_u16_mmap = embed_u16_slice,
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
                self.norm.weight.data,
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
            const recent = gemma4.recentWindow(recent_tokens);
            // 1. 优先通过 GPU 常驻 LM Head 执行巨型词表投影与采样 (消除 2.3 秒延迟至 5~10 ms)；GPU 执行失败时回退 CPU。
            //    GPU 路径只在成功读回结果之后才消耗随机数，回退不会打乱采样器的随机序列
            if (graph.options.device == .gpu and backend.offload.isAvailable()) {
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
            var mask = try backend.offload.TokenExclusionMask.init(arena, config.vocab_size, config.sampling_excluded_tokens);
            defer mask.deinit();
            if (self.lm_head_weights) |*w| {
                return gemma4_gpu.selectTokenFromHidden(arena, w, &mask, config, h_last, recent_tokens, sampler, stream);
            }
            // 未构建常驻 LM Head 时临时构建一次 BF16 词表矩阵
            var tmp = try LmHeadWeights.init(embed_u16, config.vocab_size, config.hidden_size, stream);
            defer tmp.deinit();
            return gemma4_gpu.selectTokenFromHidden(arena, &tmp, &mask, config, h_last, recent_tokens, sampler, stream);
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

            if (graph.options.device == .gpu and self.gpu_context != null) {
                const next_tok = try self.gpu_context.?.prefill(prompt_tokens, embed_u16, sampler, graph.options.mlx_stream);
                // 键值内容只存在于 GPU：CPU 层缓存仅推进游标，切回 CPU 解码前由 syncCpuCache 写回内容
                self.kv_owner = .gpu;
                for (cache.layers) |*cl| {
                    cl.curr_len = T;
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

            if (graph.options.device == .gpu and self.gpu_context != null) {
                const gpu_ctx = try self.syncGpuCache(cache);
                const next_tok = try gpu_ctx.decodeStep(token, embed_u16, recent_tokens, sampler, graph.options.mlx_stream);
                for (cache.layers) |*cl| {
                    cl.curr_len += 1;
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

            if (graph.options.device == .gpu and self.gpu_context != null) {
                const gpu_ctx = try self.syncGpuCache(cache);
                const accepted_count = try gpu_ctx.decodeSpeculative(
                    input_tokens,
                    embed_u16,
                    gemma4.recentWindow(recent_tokens),
                    sampler,
                    graph.options.mlx_stream,
                    accepted_buf,
                );
                for (cache.layers) |*cl| {
                    cl.curr_len += accepted_count;
                }
                return accepted_count;
            }

            // CPU 单步回退保底：对第 1 个词元执行标准 decodeStep
            const single_tok = try self.decodeStep(graph, input_tokens[0], cache, mmap_bytes, recent_tokens, sampler);
            accepted_buf[0] = single_tok;
            return 1;
        }
    };
}

/// 默认配置的 Gemma 4 12B 4-bit 量化模型类型
pub const DefaultGemma4Q4 = Gemma4Q4ForCausalLM(Gemma4Config.default);

/// 测试配置的微型 Q4 量化 Gemma 4 模型类型
pub const TinyQ4Gemma4 = Gemma4Q4ForCausalLM(Gemma4Config.tiny_test);
