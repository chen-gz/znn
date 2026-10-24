const std = @import("std");
const backend = @import("../backend.zig");
const vec_math = backend.vec_math;
const attention_mask = backend.attention_mask;
const tensor = @import("../tensor.zig");
const attention_kernels = tensor.attention_kernels;
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");
const deinitModel = core.deinitModel;
const enterModuleScope = core.enterModuleScope;
const normalization = @import("normalization.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Linear = core.Linear;
const RMSNorm = normalization.RMSNorm;
const InitOptions = core.InitOptions;
const createPersistentTensor = core.createPersistentTensor;
const createCacheTensor = core.createCacheTensor;
const freePersistentTensor = core.freePersistentTensor;

// ============================================================================
// 1. 键值缓存 (Key-Value Cache, KVCache & LayerKVCache)
// ============================================================================

/// 单层键值缓存 (Layer Key-Value Cache) 原语：
/// 维护当前层键 (Key, K) 与值 (Value, V) 的连续显存/内存缓冲区 [num_kv_heads, max_seq_len, head_dim]，
/// 支持追加单步词元 (append)、预填充写入 (writePrefill)、可见窗口打包 (packWindow) 与窗口覆写 (writeWindow)。
pub const LayerKVCache = struct {
    k: []f32,
    v: []f32,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    sliding_window: ?usize = null,
    curr_len: usize = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        sliding_window: ?usize,
    ) !LayerKVCache {
        const numel = num_kv_heads * max_seq_len * head_dim;
        const k = try allocator.alloc(f32, numel);
        errdefer allocator.free(k);
        const v = try allocator.alloc(f32, numel);
        @memset(k, 0.0);
        @memset(v, 0.0);
        return LayerKVCache{
            .k = k,
            .v = v,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .sliding_window = sliding_window,
            .curr_len = 0,
        };
    }

    pub fn deinit(self: *LayerKVCache, allocator: std.mem.Allocator) void {
        allocator.free(self.k);
        allocator.free(self.v);
    }

    pub fn reset(self: *LayerKVCache) void {
        self.curr_len = 0;
    }

    pub inline fn windowLen(self: *const LayerKVCache) usize {
        if (self.sliding_window) |w| return @min(self.curr_len, w);
        return self.curr_len;
    }

    pub fn visibleRange(self: *const LayerKVCache, pos: usize) attention_mask.KeyRange {
        const mask: attention_mask.Mask = if (self.sliding_window) |w|
            .{ .sliding = w }
        else
            .causal;
        return attention_mask.visibleRange(pos, 0, self.curr_len, mask);
    }

    pub fn append(self: *LayerKVCache, step_k: []const f32, step_v: []const f32) !void {
        if (self.curr_len >= self.max_seq_len) return error.ContextWindowExceeded;
        const hd = self.head_dim;
        const expected = self.num_kv_heads * hd;
        if (step_k.len != expected or step_v.len != expected) return error.ShapeMismatch;
        for (0..self.num_kv_heads) |kv_h| {
            const dest = (kv_h * self.max_seq_len + self.curr_len) * hd;
            const src = kv_h * hd;
            @memcpy(self.k[dest .. dest + hd], step_k[src .. src + hd]);
            @memcpy(self.v[dest .. dest + hd], step_v[src .. src + hd]);
        }
        self.curr_len += 1;
    }

    pub fn writePrefill(self: *LayerKVCache, k_seq: []const f32, v_seq: []const f32, seq_len: usize) !void {
        if (seq_len > self.max_seq_len) return error.ContextWindowExceeded;
        const hd = self.head_dim;
        const expected = self.num_kv_heads * seq_len * hd;
        if (k_seq.len != expected or v_seq.len != expected) return error.ShapeMismatch;
        for (0..self.num_kv_heads) |kv_h| {
            const dest = kv_h * self.max_seq_len * hd;
            const src = kv_h * seq_len * hd;
            @memcpy(self.k[dest .. dest + seq_len * hd], k_seq[src .. src + seq_len * hd]);
            @memcpy(self.v[dest .. dest + seq_len * hd], v_seq[src .. src + seq_len * hd]);
        }
        self.curr_len = seq_len;
    }

    pub fn packWindow(self: *const LayerKVCache, out_k: []f32, out_v: []f32) !void {
        const win = self.windowLen();
        const hd = self.head_dim;
        const n = self.num_kv_heads * win * hd;
        if (out_k.len != n or out_v.len != n) return error.ShapeMismatch;
        const start = self.curr_len - win;
        for (0..self.num_kv_heads) |kv_h| {
            const src = (kv_h * self.max_seq_len + start) * hd;
            const dst = kv_h * win * hd;
            @memcpy(out_k[dst .. dst + win * hd], self.k[src .. src + win * hd]);
            @memcpy(out_v[dst .. dst + win * hd], self.v[src .. src + win * hd]);
        }
    }

    pub fn writeWindow(self: *LayerKVCache, packed_k: []const f32, packed_v: []const f32, win_len: usize, len: usize) !void {
        if (len > self.max_seq_len) return error.ContextWindowExceeded;
        const required = if (self.sliding_window) |w| @min(len, w) else len;
        if (win_len > len or win_len < required) return error.ShapeMismatch;
        const hd = self.head_dim;
        const n = self.num_kv_heads * win_len * hd;
        if (packed_k.len != n or packed_v.len != n) return error.ShapeMismatch;
        const start = len - win_len;
        for (0..self.num_kv_heads) |kv_h| {
            const dst = (kv_h * self.max_seq_len + start) * hd;
            const src = kv_h * win_len * hd;
            @memcpy(self.k[dst .. dst + win_len * hd], packed_k[src .. src + win_len * hd]);
            @memcpy(self.v[dst .. dst + win_len * hd], packed_v[src .. src + win_len * hd]);
        }
        self.curr_len = len;
    }
};

/// 键值缓存 (Key-Value Cache, KVCache)，用于大语言模型 (Large Language Model, LLM) 自回归增量推理 (O(T) 生成复杂度)
pub const KVCache = struct {
    k: *Tensor, // 缓存的键 (Key, K) 张量 [batch_size, n_head, max_seq_len, head_dim]
    v: *Tensor, // 缓存的值 (Value, V) 张量 [batch_size, n_head, max_seq_len, head_dim]
    curr_len: usize = 0, // 当前已缓存的词元 (Token) 步长
    max_len: usize, // 最大支持上下文序列长度

    pub fn init(allocator: std.mem.Allocator, batch_size: usize, n_head: usize, max_len: usize, head_dim: usize) !KVCache {
        const dims = [_]usize{ batch_size, n_head, max_len, head_dim };
        const k = try createCacheTensor(allocator, &dims);
        errdefer freePersistentTensor(allocator, k);
        const v = try createCacheTensor(allocator, &dims);

        return KVCache{
            .k = k,
            .v = v,
            .curr_len = 0,
            .max_len = max_len,
        };
    }

    pub fn deinit(self: KVCache, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.k);
        freePersistentTensor(allocator, self.v);
    }

    pub fn reset(self: *KVCache) void {
        self.curr_len = 0;
    }

    pub fn asLayer(self: *KVCache) LayerKVCache {
        const n_head = self.k.shape.dims[1];
        const head_dim = self.k.shape.dims[3];
        return LayerKVCache{
            .k = self.k.data,
            .v = self.v.data,
            .num_kv_heads = n_head,
            .head_dim = head_dim,
            .max_seq_len = self.max_len,
            .sliding_window = null,
            .curr_len = self.curr_len,
        };
    }
};

// ============================================================================
// 2. 注意力机制：头拆分/合并辅助函数、缩放点积注意力 (SDPA)、
//    泛型分组查询注意力 (GQAAttention)、因果自注意力 (CausalSelfAttention) 与多头潜在注意力 (MLA)
// ============================================================================

/// 将 2D/3D 张量 [B, T, num_heads * head_dim] 拆分为 4D 头张量并转置为 [B, num_heads, T, head_dim]
pub fn splitHeads(graph: *autodiff.Graph, x: *Tensor, B: usize, T: usize, num_heads: usize, head_dim: usize) !*Tensor {
    const x_4d = try graph.reshape(x, &.{ B, T, num_heads, head_dim });
    return graph.transposeND(x_4d, 1, 2);
}

/// 将 4D 头张量 [B, num_heads, T, head_dim] 转置并合并为 3D 张量 [B, T, num_heads * head_dim]
pub fn mergeHeads(graph: *autodiff.Graph, y_4d: *Tensor, B: usize, T: usize, num_heads: usize, head_dim: usize) !*Tensor {
    const y_trans = try graph.transposeND(y_4d, 1, 2);
    return graph.reshape(y_trans, &.{ B, T, num_heads * head_dim });
}

/// 泛型分组查询注意力 (Grouped-Query Attention, GQA) 模块
/// 统一适用于全精度 Linear 与量化 Q4Linear，
/// 支持标准 MHA (num_kv_heads == num_heads)、GQA (num_kv_heads < num_heads) 与 MQA (num_kv_heads == 1)，
/// 可配置 qk_norm、无缩放 v_norm、RoPE、注意力缩放 scale、因果/滑动窗口掩码 mask 与 k_eq_v。
pub fn GQAAttention(comptime Proj: type) type {
    return struct {
        const Self = @This();

        q_proj: Proj,
        k_proj: Proj,
        v_proj: ?Proj,
        o_proj: Proj,

        q_norm: ?RMSNorm = null,
        k_norm: ?RMSNorm = null,
        v_norm: ?RMSNorm = null,

        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize = 0,
        rope: ?tensor.RopeOptions = null,
        scale: ?f32 = null,
        mask: tensor.AttentionMask = .causal,

        name: ?[]const u8 = null,
        module_type: []const u8 = "GQAAttention",

        pub const Options = struct {
            num_kv_heads: ?usize = null,
            head_dim: ?usize = null,
            sliding_window: usize = 0,
            bias: bool = false,
            k_eq_v: bool = false,
            qk_norm: bool = false,
            qk_norm_eps: f32 = 1e-6,
            v_norm: bool = false,
            v_norm_eps: f32 = 1e-6,
            rope: ?tensor.RopeOptions = null,
            scale: ?f32 = null,
            mask: tensor.AttentionMask = .causal,

            pub fn defaultOptions() Options {
                return .{};
            }
        };

        pub fn init(
            allocator: std.mem.Allocator,
            hidden_size: usize,
            num_heads: usize,
            options: Options,
        ) !Self {
            if (num_heads == 0) return error.InvalidConfiguration;
            const n_kv = options.num_kv_heads orelse num_heads;
            if (n_kv == 0 or num_heads % n_kv != 0) return error.InvalidConfiguration;

            const hd = options.head_dim orelse (hidden_size / num_heads);
            const q_dim = num_heads * hd;
            const kv_dim = n_kv * hd;

            const q_proj = if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, hidden_size, q_dim, .{ .bias = options.bias })
            else
                try Proj.init(allocator, hidden_size, q_dim);
            errdefer deinitModel(&q_proj, allocator);

            const k_proj = if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, hidden_size, kv_dim, .{ .bias = options.bias })
            else
                try Proj.init(allocator, hidden_size, kv_dim);
            errdefer deinitModel(&k_proj, allocator);

            const v_proj: ?Proj = if (options.k_eq_v)
                null
            else if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, hidden_size, kv_dim, .{ .bias = options.bias })
            else
                try Proj.init(allocator, hidden_size, kv_dim);
            errdefer if (v_proj) |*vp| deinitModel(vp, allocator);

            const o_proj = if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, q_dim, hidden_size, .{ .bias = options.bias })
            else
                try Proj.init(allocator, q_dim, hidden_size);
            errdefer deinitModel(&o_proj, allocator);

            const q_norm: ?RMSNorm = if (options.qk_norm) try RMSNorm.init(allocator, hd, options.qk_norm_eps) else null;
            errdefer if (q_norm) |*qn| deinitModel(qn, allocator);

            const k_norm: ?RMSNorm = if (options.qk_norm) try RMSNorm.init(allocator, hd, options.qk_norm_eps) else null;
            errdefer if (k_norm) |*kn| deinitModel(kn, allocator);

            const v_norm: ?RMSNorm = if (options.v_norm) RMSNorm.initUnscaled(.{ .eps = options.v_norm_eps }) else null;

            return Self{
                .q_proj = q_proj,
                .k_proj = k_proj,
                .v_proj = v_proj,
                .o_proj = o_proj,
                .q_norm = q_norm,
                .k_norm = k_norm,
                .v_norm = v_norm,
                .hidden_size = hidden_size,
                .num_heads = num_heads,
                .num_kv_heads = n_kv,
                .head_dim = hd,
                .sliding_window = options.sliding_window,
                .rope = options.rope,
                .scale = options.scale,
                .mask = options.mask,
            };
        }

        pub fn initEmpty(
            hidden_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            k_eq_v: bool,
            qk_norm_eps: f32,
            v_norm_eps: f32,
            sliding_window: usize,
            rope: ?tensor.RopeOptions,
            mask: tensor.AttentionMask,
        ) Self {
            if (Proj == core.Q4Linear) {
                const q_dim = num_heads * head_dim;
                const kv_dim = num_kv_heads * head_dim;
                return Self{
                    .q_proj = Proj.fromBlocks(hidden_size, q_dim, &.{}),
                    .k_proj = Proj.fromBlocks(hidden_size, kv_dim, &.{}),
                    .v_proj = if (k_eq_v) null else Proj.fromBlocks(hidden_size, kv_dim, &.{}),
                    .o_proj = Proj.fromBlocks(q_dim, hidden_size, &.{}),
                    .q_norm = RMSNorm.initUnscaled(.{ .eps = qk_norm_eps }),
                    .k_norm = RMSNorm.initUnscaled(.{ .eps = qk_norm_eps }),
                    .v_norm = RMSNorm.initUnscaled(.{ .eps = v_norm_eps }),
                    .hidden_size = hidden_size,
                    .num_heads = num_heads,
                    .num_kv_heads = num_kv_heads,
                    .head_dim = head_dim,
                    .sliding_window = sliding_window,
                    .rope = rope,
                    .scale = 1.0,
                    .mask = mask,
                };
            }
            @compileError("initEmpty only supported for Q4Linear");
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (@hasDecl(Proj, "deinit")) {
                self.q_proj.deinit(allocator);
                self.k_proj.deinit(allocator);
                if (self.v_proj) |*vp| vp.deinit(allocator);
                self.o_proj.deinit(allocator);
            } else {
                deinitModel(&self.q_proj, allocator);
                deinitModel(&self.k_proj, allocator);
                if (self.v_proj) |*vp| deinitModel(vp, allocator);
                deinitModel(&self.o_proj, allocator);
            }
            if (self.q_norm) |*qn| deinitModel(qn, allocator);
            if (self.k_norm) |*kn| deinitModel(kn, allocator);
        }

        pub fn resetParameters(self: *Self, random: std.Random, options: InitOptions) void {
            if (@hasDecl(Proj, "resetParameters")) {
                self.q_proj.resetParameters(random, options);
                self.k_proj.resetParameters(random, options);
                if (self.v_proj) |*vp| vp.resetParameters(random, options);
                self.o_proj.resetParameters(random, options);
            }
        }

        pub const formula = "\\text{Attn}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V W_o";

        pub fn forward(self: *const Self, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
            return self.forwardPrefillOptional(graph, x, null);
        }

        pub fn forwardPrefill(self: *const Self, graph: *autodiff.Graph, x: *Tensor, layer_cache: *LayerKVCache) !*Tensor {
            return self.forwardPrefillOptional(graph, x, layer_cache);
        }

        fn forwardPrefillOptional(self: *const Self, graph: *autodiff.Graph, x: *Tensor, layer_cache: ?*LayerKVCache) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const B = x.shape.dims[0];
            const T = x.shape.dims[1];
            const nh = self.num_heads;
            const n_kv = self.num_kv_heads;
            const hd = self.head_dim;
            const groups = nh / n_kv;

            const q_raw = try self.q_proj.forward(graph, x);
            const k_raw = try self.k_proj.forward(graph, x);
            const v_raw = if (self.v_proj) |*vp| try vp.forward(graph, x) else k_raw;

            const q_normed = if (self.q_norm) |*qn| blk: {
                const q_heads = try graph.reshape(q_raw, &.{ B * T * nh, hd });
                break :blk try qn.forward(graph, q_heads);
            } else q_raw;

            const k_normed = if (self.k_norm) |*kn| blk: {
                const k_heads = try graph.reshape(k_raw, &.{ B * T * n_kv, hd });
                break :blk try kn.forward(graph, k_heads);
            } else k_raw;

            const v_normed = if (self.v_norm) |*vn| blk: {
                const v_heads = try graph.reshape(v_raw, &.{ B * T * n_kv, hd });
                break :blk try vn.forward(graph, v_heads);
            } else v_raw;

            var q = try splitHeads(graph, q_normed, B, T, nh, hd);
            var k = try splitHeads(graph, k_normed, B, T, n_kv, hd);
            const v = try splitHeads(graph, v_normed, B, T, n_kv, hd);

            if (self.rope) |rope_opts| {
                q = try graph.rope(q, 0, rope_opts);
                k = try graph.rope(k, 0, rope_opts);
            }

            if (layer_cache) |cache| {
                if (B != 1) return error.InvalidBatchSize;
                try cache.writePrefill(k.data, v.data, T);
            }

            const scale = self.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(hd))));
            const y_4d = try graph.attention(q, k, v, .{
                .scale = scale,
                .mask = self.mask,
                .kv_groups = groups,
            });

            const y_3d = try mergeHeads(graph, y_4d, B, T, nh, hd);
            return try self.o_proj.forward(graph, y_3d);
        }
    };
}

/// 缩放点积注意力核心模块 (Scaled Dot-Product Attention, SDPA)
/// 负责对四维 (4-Dimensional, 4D) 多头张量 Q \in [B, H_q, T_q, d_k]、K \in [B, H_kv, T_k, d_k] 与 V \in [B, H_kv, T_k, d_v] 执行核心注意力运算：
/// \text{AttentionCore}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V
/// 掩码 M 由 `mask` 指定 (默认因果掩码)；H_q 为 H_kv 的整数倍时按分组查询注意力 (Grouped-Query Attention, GQA) 共享键值。
/// 计算交给单个计算图算子 `Graph.attention`，该模块既可作为独立无参数层直接调用，也作为 `CausalSelfAttention` 与 `MLALayer` 的内置计算核心复用。
pub const ScaledDotProductAttention = struct {
    mask: tensor.AttentionMask = .causal, // 注意力掩码 (默认自回归因果掩码)
    name: ?[]const u8 = null,
    module_type: []const u8 = "ScaledDotProductAttention",

    /// 缩放点积注意力配置选项 (Scaled Dot-Product Attention Options)
    pub const Options = struct {
        mask: tensor.AttentionMask = .causal,

        pub const default: Options = .{};
        pub fn defaultOptions() Options {
            return .{};
        }
    };

    pub const formula = "\\text{AttentionCore}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V";

    pub fn init(options: Options) ScaledDotProductAttention {
        return .{ .mask = options.mask };
    }

    pub fn initDefault() ScaledDotProductAttention {
        return init(Options.default);
    }

    /// 开启注意力核心子作用域 (显式模块名，未命名时为当前模块下的 "core") 并执行缩放点积注意力前向传播：
    /// 打分缩放系数为 1 / sqrt(d_k)，分组查询注意力的组大小由 H_q / H_kv 给出
    pub fn forward(self: *const ScaledDotProductAttention, graph: *autodiff.Graph, q: *Tensor, k: *Tensor, v: *Tensor) !*Tensor {
        const core_scope = if (self.name) |n|
            try graph.enterModule(n, self.module_type)
        else
            try graph.enterChildScope("core", self.module_type);
        defer core_scope.exit();
        if (core_scope.graph != null) try graph.setModuleFormula(graph.currentScope(), formula);
        if (q.shape.len != 4 or k.shape.len != 4 or k.shape.dims[1] == 0) return error.IncompatibleDimensions;
        return graph.attention(q, k, v, .{ .mask = self.mask, .kv_groups = q.shape.dims[1] / k.shape.dims[1] });
    }
};

/// 因果自注意力机制 (Causal Self-Attention / 掩码多头注意力 Masked Multi-Head Attention, MHA)
/// 变换器 (Transformer) 的核心机制，负责建模序列中不同位置的依赖关系。
///
/// 数学公式：
/// Q = X W_q, \quad K = X W_k, \quad V = X W_v
/// \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V
/// \text{Output} = \text{Attention}(Q, K, V) W_p
/// 其中 M 是因果掩码矩阵，上三角（未来位置）元素为 -\infty，其余为 0。
///
/// 包含以下关键设计：
/// 1. 多头注意力 (Multi-Head Attention, MHA)：将特征通道划分为 nh 个头，让模型在多个不同的投影子空间内并行关注信息。
/// 2. 因果掩码 (Causal Mask)：通过加上上三角矩阵（值为 -inf），阻止当前位置关注未来的位置，确保自回归生成时的因果律。
pub const CausalSelfAttention = struct {
    q_attn: Linear, // 查询 (Query, Q) 线性投影层
    k_attn: Linear, // 键 (Key, K) 线性投影层
    v_attn: Linear, // 值 (Value, V) 线性投影层
    core: ScaledDotProductAttention = .{}, // 缩放点积注意力核心子模块 (Scaled Dot-Product Attention, SDPA)
    c_proj: Linear, // 最终的多头输出融合与投影层 (Output Projection, c_proj)
    n_head: usize, // 查询注意力头数 (Query Heads, n_head)
    n_embd: usize, // 隐藏特征嵌入维度 (Embedding Dimension, n_embd)
    num_kv_heads: usize, // 键值头数 (Key-Value Heads)：1 为多查询注意力 (Multi-Query Attention, MQA)，< n_head 为分组查询注意力 (Grouped-Query Attention, GQA)，== n_head 为多头注意力 (Multi-Head Attention, MHA)
    name: ?[]const u8 = null,
    module_type: []const u8 = "CausalSelfAttention",

    /// 初始化支持分组查询注意力 (Grouped-Query Attention, GQA)、多查询注意力 (Multi-Query Attention, MQA) 与多头注意力 (Multi-Head Attention, MHA) 的自注意力层
    /// n_embd: 隐藏特征嵌入维度，必须能被 n_head 整除
    /// n_head: 查询 (Query) 注意力头数
    /// num_kv_heads: 键值 (Key-Value) 头数，必须能整除 n_head；不满足整除关系时返回 `error.InvalidConfiguration`
    pub fn initGQA(allocator: std.mem.Allocator, n_embd: usize, n_head: usize, num_kv_heads: usize) !CausalSelfAttention {
        if (n_head == 0 or num_kv_heads == 0 or n_embd % n_head != 0 or n_head % num_kv_heads != 0) {
            return error.InvalidConfiguration;
        }
        const hs = n_embd / n_head;
        const kv_dim = num_kv_heads * hs;

        const q_attn = try Linear.init(allocator, n_embd, n_embd);
        errdefer deinitModel(&q_attn, allocator);
        const k_attn = try Linear.init(allocator, n_embd, kv_dim);
        errdefer deinitModel(&k_attn, allocator);
        const v_attn = try Linear.init(allocator, n_embd, kv_dim);
        errdefer deinitModel(&v_attn, allocator);
        const c_proj = try Linear.init(allocator, n_embd, n_embd);
        errdefer deinitModel(&c_proj, allocator);

        return CausalSelfAttention{
            .q_attn = q_attn,
            .k_attn = k_attn,
            .v_attn = v_attn,
            .c_proj = c_proj,
            .n_head = n_head,
            .n_embd = n_embd,
            .num_kv_heads = num_kv_heads,
        };
    }

    /// 初始化标准多头自注意力层 (Multi-Head Attention, MHA: num_kv_heads == n_head)
    pub fn init(allocator: std.mem.Allocator, n_embd: usize, n_head: usize) !CausalSelfAttention {
        return initGQA(allocator, n_embd, n_head, n_head);
    }

    /// 模块标准数学变换公式
    pub const formula = "A = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V W_o + b_o";
    pub const core_formula = ScaledDotProductAttention.formula;

    /// 前向注意力计算流程
    /// 输入 x 的形状必须为三维张量 (3-Dimensional Tensor, 3D): [B, T, C]
    /// 其中 B 为批次大小 (Batch Size, B)，T 为时间步序列长度 (Sequence Length, T)，C 为通道特征维数 (Embedding Dimension, C / n_embd)
    pub fn forward(self: *const CausalSelfAttention, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const B = x.shape.dims[0];
        const T = x.shape.dims[1];
        const C = x.shape.dims[2];
        const nh = self.n_head;
        const n_kv = self.num_kv_heads;
        const hs = C / nh; // 每个注意力头的维度大小 (Head Size, hs)

        // 1. 投影计算查询 (Query, Q)、键 (Key, K) 与值 (Value, V)
        const q_raw = try self.q_attn.forward(graph, x);
        const k_raw = try self.k_attn.forward(graph, x);
        const v_raw = try self.v_attn.forward(graph, x);

        // 2. 拆分多头并转置: [B, T, nh*hs] -> [B, nh, T, hs]
        const q = try splitHeads(graph, q_raw, B, T, nh, hs);
        const k = try splitHeads(graph, k_raw, B, T, n_kv, hs);
        const v = try splitHeads(graph, v_raw, B, T, n_kv, hs);

        // 3. 委托给缩放点积注意力核心子模块 (ScaledDotProductAttention) 执行 softmax(Q K^T / sqrt(hs) + M) V；
        // n_kv < nh 时按分组查询注意力 (Grouped-Query Attention, GQA) 由每个键值头服务 nh / n_kv 个查询头，不复制键值
        const y_4d = try self.core.forward(graph, q, k, v);

        // 4. 将多头的输出转置回去并整合: [B, nh, T, hs] -> [B, T, C]
        const y_3d = try mergeHeads(graph, y_4d, B, T, nh, hs);

        // 5. 投影输出映射: [B, T, C] -> [B, T, C]
        return try self.c_proj.forward(graph, y_3d);
    }

    /// 基于键值缓存 (Key-Value Cache, KVCache) 的单步增量自回归推理 (O(1) 增量键值计算，O(T) 点积注意力)
    /// 输入 x 的形状为 [B, 1, C] 或 [B, C]；缓存已满时返回 `error.ContextWindowExceeded`
    pub fn forwardInference(self: *const CausalSelfAttention, allocator: std.mem.Allocator, x: *Tensor, cache: *KVCache) !*Tensor {
        const B = x.shape.dims[0];
        const C = self.n_embd;
        const nh = self.n_head;
        const n_kv = self.num_kv_heads;
        const hs = C / nh;
        const groups = nh / n_kv;
        const kv_dim = n_kv * hs;

        if (cache.curr_len >= cache.max_len) return error.ContextWindowExceeded;
        const cache_dims = cache.k.shape.dims;
        if (B > cache_dims[0] or cache_dims[1] != n_kv or cache_dims[3] != hs) return error.ShapeMismatch;

        // 单步推理使用局部无梯度计算图承载中间张量，函数返回时整体释放
        var step_g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
        defer step_g.deinit();

        // 1. 获取二维 (2-Dimensional, 2D) 输入 [B, C]
        const x_2d = if (x.shape.len != 2) try step_g.reshape(x, &.{ B, C }) else x;

        // 2. 投影当前词元 (Token) 的查询 (Query, Q)、键 (Key, K) 与值 (Value, V)
        const q_2d = try self.q_attn.forward(&step_g, x_2d);
        const k_step = try self.k_attn.forward(&step_g, x_2d);
        const v_step = try self.v_attn.forward(&step_g, x_2d);

        // 3. 写入键值缓存 (Key-Value Cache, KVCache)
        const t = cache.curr_len;
        for (0..B) |b| {
            for (0..n_kv) |kv_h| {
                const src_k = k_step.data[(b * kv_dim + kv_h * hs)..(b * kv_dim + (kv_h + 1) * hs)];
                const src_v = v_step.data[(b * kv_dim + kv_h * hs)..(b * kv_dim + (kv_h + 1) * hs)];
                const dest_k = cache.k.data[((b * n_kv + kv_h) * cache.max_len + t) * hs .. ((b * n_kv + kv_h) * cache.max_len + t + 1) * hs];
                const dest_v = cache.v.data[((b * n_kv + kv_h) * cache.max_len + t) * hs .. ((b * n_kv + kv_h) * cache.max_len + t + 1) * hs];
                @memcpy(dest_k, src_k);
                @memcpy(dest_v, src_v);
            }
        }
        cache.curr_len += 1;
        const curr_len = cache.curr_len;

        // 4. 注意力计算：对当前 1 个查询 (Query, Q) 与缓存中 [0..curr_len] 个键 (Key, K) 计算点积
        const y_2d = try step_g.zeros(&.{ B, C }, false);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hs)));
        const scores = try step_g.arenaAllocator().alloc(f32, curr_len);
        const range = attention_kernels.visibleRange(t, 0, curr_len, self.core.mask);

        for (0..B) |b| {
            for (0..nh) |h| {
                const head_base = (b * n_kv + h / groups) * cache.max_len * hs;
                attention_kernels.attendOneQuery(
                    q_2d.data[(b * C + h * hs)..(b * C + (h + 1) * hs)],
                    cache.k.data[head_base..][0 .. curr_len * hs],
                    cache.v.data[head_base..][0 .. curr_len * hs],
                    range,
                    scale,
                    scores,
                    y_2d.data[(b * C + h * hs)..(b * C + (h + 1) * hs)],
                );
            }
        }

        // 5. 投影输出，并拷贝到调用方分配器上 (局部计算图随函数返回释放)
        const out_proj = try self.c_proj.forward(&step_g, y_2d);
        const out_shape: []const usize = if (x.shape.len == 3) &.{ B, 1, C } else &.{ B, C };
        return try tensor.array(allocator, out_shape, out_proj.data);
    }
};

/// 多头潜在注意力缓存 (Multi-Head Latent Attention Cache, MLACache)
/// 对应 DeepSeek-V2 / V3 论文：
/// 仅存储低维联合压缩潜在键值向量 (Latent Key-Value Vector, c_t^{KV}) 与解耦旋转位置编码键 (Decoupled Rotary Position Embedding Key, k_t^R)，
/// 相比传统多头注意力 (Multi-Head Attention, MHA) 降低高达 93.3% 显存开销。
pub const MLACache = struct {
    c_kv: *Tensor, // 潜在键值缓存 (Latent Key-Value Cache) [batch_size, max_len, d_c]
    k_r: *Tensor, // 解耦旋转位置编码键缓存 (Decoupled Rotary Position Embedding Key Cache) [batch_size, max_len, d_r]
    curr_len: usize = 0,
    max_len: usize,
    d_c: usize,
    d_r: usize,

    pub fn init(allocator: std.mem.Allocator, batch_size: usize, max_len: usize, d_c: usize, d_r: usize) !MLACache {
        const c_kv = try createCacheTensor(allocator, &.{ batch_size, max_len, d_c });
        errdefer freePersistentTensor(allocator, c_kv);
        const k_r = try createCacheTensor(allocator, &.{ batch_size, max_len, d_r });

        return MLACache{
            .c_kv = c_kv,
            .k_r = k_r,
            .curr_len = 0,
            .max_len = max_len,
            .d_c = d_c,
            .d_r = d_r,
        };
    }

    pub fn deinit(self: MLACache, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.c_kv);
        freePersistentTensor(allocator, self.k_r);
    }

    pub fn reset(self: *MLACache) void {
        self.curr_len = 0;
    }
};

/// 多头潜在注意力机制 (Multi-Head Latent Attention, MLA / MLALayer)
/// 对应 DeepSeek-V2 / V3 核心注意力架构：
/// 采用键值 (Key-Value, KV) 低秩联合压缩、解耦旋转位置编码 (Rotary Position Embedding, RoPE) 以及推理期权重矩阵吸收 (Weight Matrix Absorption)。
pub const MLALayer = struct {
    dim: usize,
    n_head: usize,
    head_dim: usize,
    d_c: usize, // 键值 (Key-Value, KV) 潜在压缩维度 (如 512)
    d_r: usize, // 解耦旋转位置编码 (Rotary Position Embedding, RoPE) 维度 (如 64)
    q_proj: Linear, // 查询 (Query) 投影: dim -> n_head * (head_dim + d_r)
    w_dkv: Linear, // 键值 (Key-Value, KV) 下投影 (Down-Projection): dim -> d_c
    w_kr: Linear, // 旋转位置编码键 (Rotary Position Embedding Key) 投影: dim -> d_r
    w_uk: Linear, // 内容键 (Content Key) 上投影 (Up-Projection): d_c -> n_head * head_dim
    w_uv: Linear, // 内容值 (Content Value) 上投影 (Up-Projection): d_c -> n_head * head_dim
    core: ScaledDotProductAttention = .{}, // 缩放点积注意力核心子模块 (Scaled Dot-Product Attention, SDPA)
    o_proj: Linear, // 输出投影 (Output Projection): n_head * head_dim -> dim
    /// 解耦旋转位置编码配置，作用于每个 d_r 维旋转向量 (查询的旋转子空间与共享 k_r)；全序列前向与增量解码共用
    rope: tensor.RopeOptions = .{},
    name: ?[]const u8 = null,
    module_type: []const u8 = "MLALayer",

    pub const formula = "c_t^{KV} = x_t W^{DKV}, \\quad y = \\text{MLA}(Q, c^{KV}, k^R) W^O";

    pub fn init(
        allocator: std.mem.Allocator,
        dim: usize,
        n_head: usize,
        head_dim: usize,
        d_c: usize,
        d_r: usize,
    ) !MLALayer {
        const total_q_dim = n_head * (head_dim + d_r);
        const total_kv_dim = n_head * head_dim;

        const q_proj = try Linear.init(allocator, dim, total_q_dim);
        errdefer deinitModel(&q_proj, allocator);

        const w_dkv = try Linear.init(allocator, dim, d_c);
        errdefer deinitModel(&w_dkv, allocator);

        const w_kr = try Linear.init(allocator, dim, d_r);
        errdefer deinitModel(&w_kr, allocator);

        const w_uk = try Linear.init(allocator, d_c, total_kv_dim);
        errdefer deinitModel(&w_uk, allocator);

        const w_uv = try Linear.init(allocator, d_c, total_kv_dim);
        errdefer deinitModel(&w_uv, allocator);

        const o_proj = try Linear.init(allocator, total_kv_dim, dim);
        errdefer deinitModel(&o_proj, allocator);

        return MLALayer{
            .dim = dim,
            .n_head = n_head,
            .head_dim = head_dim,
            .d_c = d_c,
            .d_r = d_r,
            .q_proj = q_proj,
            .w_dkv = w_dkv,
            .w_kr = w_kr,
            .w_uk = w_uk,
            .w_uv = w_uv,
            .o_proj = o_proj,
        };
    }

    /// 全序列前向传播 (经由计算图执行，支持自动微分 (Automatic Differentiation, Autograd) 梯度回传)
    pub fn forward(self: *const MLALayer, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        const B = if (is_3d) old_shape.dims[0] else 1;
        const T = if (is_3d) old_shape.dims[1] else old_shape.dims[0];
        const C = if (is_3d) old_shape.dims[2] else old_shape.dims[1];
        const nh = self.n_head;
        const hd = self.head_dim;
        const dr = self.d_r;
        const q_head_dim = hd + dr;

        // 1. 投影查询 (Query, Q)、潜在键值向量 (Latent Key-Value, c_kv) 与解耦旋转位置编码键 (Decoupled Rotary Position Embedding Key, k_r)
        const q_all = try self.q_proj.forward(graph, x);

        const c_kv = try self.w_dkv.forward(graph, x);

        const k_r = try self.w_kr.forward(graph, x);

        // 2. 上投影还原内容键 (Content Key, Kc) 与内容值 (Content Value, Vc)
        const k_c = try self.w_uk.forward(graph, c_kv);

        const v_c = try self.w_uv.forward(graph, c_kv);

        // 3. 重塑并转置为四维 (4-Dimensional, 4D) 多头结构，对查询 (Query) 的旋转位置编码 (Rotary Position Embedding, RoPE) 子空间与共享 k_r 施加旋转位置编码
        const q_4d = try graph.reshape(q_all, &.{ B, T, nh, q_head_dim });
        const q_trans = try graph.transposeND(q_4d, 1, 2);
        var q_rope = self.rope;
        q_rope.rotary_offset += hd;
        const q_rot = try graph.rope(q_trans, 0, q_rope);

        const k_c_4d = try graph.reshape(k_c, &.{ B, T, nh, hd });
        const k_c_trans = try graph.transposeND(k_c_4d, 1, 2);

        const k_r_4d = try graph.reshape(k_r, &.{ B, 1, T, dr });
        const k_r_rot = try graph.rope(k_r_4d, 0, self.rope);

        const k_r_heads = if (nh > 1) try graph.repeatKV(k_r_rot, nh) else k_r_rot;

        const k_full = try graph.concat(&.{ k_c_trans, k_r_heads }, 3);

        const v_4d = try graph.reshape(v_c, &.{ B, T, nh, hd });
        const v = try graph.transposeND(v_4d, 1, 2);

        // 4. 委托给缩放点积注意力核心 (Scaled Dot-Product Attention, SDPA): Softmax((Q * K^T) / sqrt(hd + dr) + M) * V
        const y_4d = try self.core.forward(graph, q_rot, k_full, v);

        // 5. 合并多头并经输出投影 (Output Projection, o_proj) 输出特征
        const y_trans = try graph.transposeND(y_4d, 1, 2);
        const y_merged = try graph.reshape(y_trans, &.{ B, T, nh * hd });
        const out = try self.o_proj.forward(graph, y_merged);
        if (is_3d) return out;
        return try graph.reshape(out, &.{ T, C });
    }

    /// 多头潜在注意力 (Multi-Head Latent Attention, MLA) 推理期权重矩阵吸收 (Weight Matrix Absorption) 单步自回归生成
    /// 完全在低维潜在空间进行注意力计算与累加，绝不展开高维键值 (Key-Value, KV) 张量。
    /// 输入 x 的形状为 [B, 1, C] 或 [B, C]；缓存已满时返回 `error.ContextWindowExceeded`
    pub fn forwardInference(self: *const MLALayer, allocator: std.mem.Allocator, x: *Tensor, cache: *MLACache) !*Tensor {
        const B = x.shape.dims[0];
        const C = self.dim;
        const nh = self.n_head;
        const hd = self.head_dim;
        const dc = self.d_c;
        const dr = self.d_r;

        if (cache.curr_len >= cache.max_len) return error.ContextWindowExceeded;
        if (B > cache.c_kv.shape.dims[0] or cache.d_c != dc or cache.d_r != dr) return error.ShapeMismatch;

        // 单步推理使用局部无梯度计算图承载中间张量，函数返回时整体释放
        var step_g = autodiff.Graph.init(allocator, .{ .enable_grad = false });
        defer step_g.deinit();
        const step_alloc = step_g.arenaAllocator();

        const x_2d = if (x.shape.len != 2) try step_g.reshape(x, &.{ B, C }) else x;

        // 1. 投影当前词元 (Token) 的查询 (Query, Q)、潜在键值 (Latent Key-Value, c_kv) 与解耦旋转键 (Rotary Key, k_r)
        const q_all = try self.q_proj.forward(&step_g, x_2d);
        const c_kv_step = try self.w_dkv.forward(&step_g, x_2d);
        const k_r_step = try self.w_kr.forward(&step_g, x_2d);

        // 2. 施加旋转位置编码 (Rotary Position Embedding, RoPE) 并写入多头潜在注意力缓存 (Multi-Head Latent Attention Cache, MLACache)
        const t = cache.curr_len;
        const rope_table = try tensor.RopeTable.init(dr, self.rope);
        const rope_cos = try step_alloc.alloc(f32, rope_table.num_angles);
        const rope_sin = try step_alloc.alloc(f32, rope_table.num_angles);
        rope_table.angles(t, rope_cos, rope_sin);

        for (0..B) |b| {
            const k_r_vec = k_r_step.data[b * dr .. (b + 1) * dr];
            rope_table.applyRow(k_r_vec, rope_cos, rope_sin, false);

            const dest_c = cache.c_kv.data[(b * cache.max_len + t) * dc .. (b * cache.max_len + t + 1) * dc];
            const dest_r = cache.k_r.data[(b * cache.max_len + t) * dr .. (b * cache.max_len + t + 1) * dr];
            @memcpy(dest_c, c_kv_step.data[b * dc .. (b + 1) * dc]);
            @memcpy(dest_r, k_r_vec);
        }
        cache.curr_len += 1;
        const curr_len = cache.curr_len;

        // 3. 权重矩阵吸收 (Weight Matrix Absorption) 计算注意力
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd + dr)));
        const y_tensor = try step_g.zeros(&.{ B, nh * hd }, false);
        const y_concat = y_tensor.data;
        // 当前查询在缓存中的可见区间 (与全序列前向共用核心模块的掩码规则)
        const range = attention_kernels.visibleRange(t, 0, curr_len, self.core.mask);
        const scores = try step_alloc.alloc(f32, range.len());
        const q_absorbed = try step_alloc.alloc(f32, dc);
        const u_latent = try step_alloc.alloc(f32, dc);

        const q_stride = hd + dr;
        const kv_cols = nh * hd;

        for (0..B) |b| {
            for (0..nh) |h| {
                const q_offset = b * (nh * q_stride) + h * q_stride;
                const q_c_head = q_all.data[q_offset .. q_offset + hd];
                const q_r_head = q_all.data[q_offset + hd .. q_offset + q_stride];
                rope_table.applyRow(q_r_head, rope_cos, rope_sin, false);

                // 公式 (7) 矩阵吸收: \tilde{q}_{t,i} = (W_i^{UK})^T q_{t,i}^C \in R^{dc}
                // 内容键偏置 b^{UK} 对所有位置贡献同一常数 q^C \cdot b^{UK}，归一化指数函数 (Softmax) 对其不变，故可省略
                for (q_absorbed, 0..) |*qa, c_idx| {
                    const w_row = self.w_uk.weight.data[c_idx * kv_cols + h * hd .. c_idx * kv_cols + (h + 1) * hd];
                    qa.* = vec_math.dot(w_row, q_c_head);
                }

                // 在低维潜在空间直接与缓存中的 c_j^{KV} 计算点积打分
                for (scores, range.begin..) |*score, pos| {
                    const c_cached = cache.c_kv.data[(b * cache.max_len + pos) * dc .. (b * cache.max_len + pos + 1) * dc];
                    const k_r_cached = cache.k_r.data[(b * cache.max_len + pos) * dr .. (b * cache.max_len + pos + 1) * dr];
                    score.* = (vec_math.dot(q_absorbed, c_cached) + vec_math.dot(q_r_head, k_r_cached)) * scale;
                }
                vec_math.softmaxRowInPlace(scores);

                // 公式 (8) 潜在空间加权求和: u_h = \sum \alpha_{i,j} c_j^{KV} \in R^{dc}
                attention_kernels.weightedSum(scores, cache.c_kv.data[(b * cache.max_len + range.begin) * dc ..][0 .. range.len() * dc], u_latent);

                // 还原头输出: y_h = (W_i^{UV})^T u_h + b_i^{UV} \in R^{hd}
                // 注意力权重之和为 1，因此值偏置 b^{UV} 原样出现在输出中；按 c -> k 顺序连续读取 W^{UV} 的行
                const y_head_dest = y_concat[(b * nh + h) * hd .. (b * nh + h + 1) * hd];
                if (self.w_uv.bias) |bias| {
                    @memcpy(y_head_dest, bias.data[h * hd .. (h + 1) * hd]);
                } else {
                    @memset(y_head_dest, 0.0);
                }
                for (u_latent, 0..) |u, c_idx| {
                    vec_math.axpy(y_head_dest, u, self.w_uv.weight.data[c_idx * kv_cols + h * hd .. c_idx * kv_cols + (h + 1) * hd]);
                }
            }
        }

        // 4. 投影输出 (Output Projection, o_proj)，并拷贝到调用方分配器上 (局部计算图随函数返回释放)
        const out_proj = try self.o_proj.forward(&step_g, y_tensor);
        const out_shape: []const usize = if (x.shape.len == 3) &.{ B, 1, C } else &.{ B, C };
        return try tensor.array(allocator, out_shape, out_proj.data);
    }
};
