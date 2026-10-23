//! Gemma 4 图形处理器 (Graphics Processing Unit, GPU) 推理共享组件与 12B 端到端解码上下文
//!
//! 包含 E4B 与 12B 两条 Apple 机器学习拓展 (Machine Learning Explore, MLX) 推理路径共用的
//! 比例旋转位置编码 (Proportional Rotary Position Embedding, Proportional RoPE)、最终对数几率截断与词元选取，
//! 以及 12B Q4 模型的常驻显存解码层 (`Gemma4GpuLayer`) 与整网自回归解码上下文 (`Gemma4GpuContext`)。
//! backend 包只提供与模型无关的 MLX / 卸载原语，Gemma 4 专属的层结构、键值缓存管理与采样策略集中在本文件。

const std = @import("std");
const backend = @import("../backend.zig");
const gemma4 = @import("gemma4.zig");

const mlx = backend.mlx;
const offload = backend.offload;
const sampling = backend.sampling;
const Stream = mlx.Stream;
const Q4Weights = offload.Q4Weights;
const LmHeadWeights = offload.LmHeadWeights;
const TokenExclusionMask = offload.TokenExclusionMask;
const Gemma4Config = gemma4.Gemma4Config;
const Gemma4AttentionType = gemma4.Gemma4AttentionType;
const Gemma4KVCache = gemma4.Gemma4KVCache;

// ============================================================================
// 比例旋转位置编码 (Proportional RoPE)
// ============================================================================

/// 计算比例旋转位置编码的周期表，与 mlx-lm `ProportionalRoPE`、Hugging Face `proportional` RoPE 以及
/// F32 CPU 路径 (`tensor.rope` 的 split_half + partial_rotary_factor) 一致：在完整 head_dim 上以 (i, i + head_dim / 2) 配对，
/// 前 num_angles = floor(head_dim / 2 * partial_rotary_factor) 个角度的周期为 base^(2i / head_dim)，
/// 其余角度周期为 +inf (旋转角恒为 0，保持恒等)。`mlx.fast.rope` 以 theta_i = pos / freqs[i] 计算旋转角。
pub fn proportionalRopeFreqs(out: []f32, head_dim: usize, partial_rotary_factor: f32, base: f32) void {
    const half = head_dim / 2;
    std.debug.assert(out.len == half);
    const factor = std.math.clamp(partial_rotary_factor, 0.0, 1.0);
    const num_angles: usize = @intFromFloat(@as(f32, @floatFromInt(half)) * factor);
    const hd_f: f32 = @floatFromInt(head_dim);
    for (out, 0..) |*f, i| {
        f.* = if (i < num_angles)
            std.math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / hd_f)
        else
            std.math.inf(f32);
    }
}

/// 单个注意力层的 GPU 旋转位置编码：partial_rotary_factor 为 1 时为标准半切分 RoPE (base = rope_theta)，
/// 小于 1 时为在完整 head_dim 上旋转的比例旋转位置编码 (常驻 [head_dim / 2] 周期表)
pub const GpuRope = struct {
    head_dim: usize,
    base: f32,
    freqs: ?mlx.Array = null,

    /// 周期表在栈上构建时支持的最大角度数 (head_dim / 2)
    pub const max_angles: usize = 1024;

    pub fn init(head_dim: usize, partial_rotary_factor: f32, base: f32) !GpuRope {
        if (head_dim == 0 or head_dim % 2 != 0) return error.InvalidShape;
        if (partial_rotary_factor >= 1.0) return .{ .head_dim = head_dim, .base = base };
        const half = head_dim / 2;
        if (half > max_angles) return error.InvalidShape;
        var buf: [max_angles]f32 = undefined;
        proportionalRopeFreqs(buf[0..half], head_dim, partial_rotary_factor, base);
        return .{
            .head_dim = head_dim,
            .base = base,
            .freqs = try mlx.Array.fromSlice(buf[0..half], &.{half}),
        };
    }

    pub fn deinit(self: *GpuRope) void {
        if (self.freqs) |*f| f.deinit();
        self.freqs = null;
    }

    /// 对 [B, heads, T, head_dim] 张量施加旋转位置编码，第 t 个位置的绝对位置为 offset + t
    pub fn apply(self: *const GpuRope, x: mlx.Array, offset: usize, s: ?Stream) !mlx.Array {
        if (self.freqs) |f| return mlx.Array.fastRoPEWithFreqs(x, self.head_dim, false, 1.0, offset, f, s);
        return mlx.Array.fastRoPE(x, self.head_dim, false, self.base, 1.0, offset, s);
    }
};

// ============================================================================
// 对数几率截断与词元选取 (E4B 与 12B 共用)
// ============================================================================

/// 将 GPU 对数几率转换为单精度并施加最终对数几率截断 (Final Logit Softcapping)：cap * tanh(x / cap)；cap <= 0 时只做类型转换
pub fn softcapLogits(logits: mlx.Array, cap: f32, s: ?Stream) !mlx.Array {
    var logits_f32 = try logits.astype(.float32, s);
    if (cap <= 0.0) return logits_f32;
    defer logits_f32.deinit();
    var scaled = try logits_f32.mulScalar(1.0 / cap, s);
    defer scaled.deinit();
    var tanh_l = try mlx.Array.tanh(scaled, s);
    defer tanh_l.deinit();
    return tanh_l.mulScalar(cap, s);
}

/// 从单行 GPU 对数几率 [1, vocab_size] 中选取下一个词元：
/// - 无重复惩罚时在 GPU 上叠加排除掩码后直接求 argmax（最终对数几率截断单调递增，不改变最大值位置）；
/// - 有重复惩罚时先截断，再在 CPU 上对截断后的值施加惩罚并贪婪采样 (recent_tokens 中的词元全部受罚，窗口由调用方决定)。
/// 两条路径排除同一组词元 (`cfg.sampling_excluded_tokens`，mask 须由同一配置构建)。
pub fn selectToken(
    logits: mlx.Array,
    cfg: Gemma4Config,
    mask: *const TokenExclusionMask,
    recent_tokens: []const usize,
    repetition_penalty: f32,
    s: ?Stream,
) !usize {
    if (!sampling.penaltyActive(repetition_penalty)) {
        var best = try mask.argmax(logits, s);
        defer best.deinit();
        return @intCast(try best.itemUint32());
    }
    var capped = try softcapLogits(logits, cfg.final_logit_softcapping, s);
    defer capped.deinit();
    const data = try capped.dataFloat32();
    if (data.len != cfg.vocab_size) return error.ShapeMismatch;
    return sampling.argmaxWithPenalty(data, cfg.vocab_size, cfg.sampling_excluded_tokens, recent_tokens, repetition_penalty);
}

/// 以常驻 GPU 词表投影对最终隐状态 h_last [hidden_size] 打分并选取下一个词元
pub fn selectTokenFromHidden(
    lm_head: *const LmHeadWeights,
    mask: *const TokenExclusionMask,
    cfg: Gemma4Config,
    h_last: []const f32,
    recent_tokens: []const usize,
    repetition_penalty: f32,
    s: ?Stream,
) !usize {
    if (h_last.len != lm_head.hidden_size or lm_head.vocab_size != cfg.vocab_size) return error.ShapeMismatch;
    var x = try mlx.Array.fromSlice(h_last, &.{ 1, lm_head.hidden_size });
    defer x.deinit();
    var logits = try lm_head.forwardArrayBf16(x, s);
    defer logits.deinit();
    return selectToken(logits, cfg, mask, recent_tokens, repetition_penalty, s);
}

// ============================================================================
// Gemma 4 12B Q4 常驻显存解码层
// ============================================================================

/// 第 layer_idx 层注意力投影的输出维度：全局注意力层 attention_k_eq_v 为 true，不存储独立的 V 投影 (v_dim = 0)
pub const ProjectionDims = struct {
    q_dim: usize,
    kv_dim: usize,
    v_dim: usize,

    pub fn of(cfg: Gemma4Config, layer_idx: usize) ProjectionDims {
        const spec = cfg.layerSpec(layer_idx);
        const kv_dim = spec.num_kv_heads * spec.head_dim;
        return .{
            .q_dim = cfg.num_attention_heads * spec.head_dim,
            .kv_dim = kv_dim,
            .v_dim = if (spec.attn_type == .full_attention) 0 else kv_dim,
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

    input_norm_w: mlx.Array,
    q_norm_w: mlx.Array,
    k_norm_w: mlx.Array,
    post_attn_w: mlx.Array,
    pre_mlp_w: mlx.Array,
    post_mlp_w: mlx.Array,

    layer_scalar: f32,
    attn_type: Gemma4AttentionType,
    sliding_window: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: GpuRope,
    rms_norm_eps: f32,
    compiled_mlp: ?mlx.Closure = null,

    /// 单层四个融合量化投影 (QKV、O、Gate+Up、Down)
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

        /// 从 18 字节 Q4Block 原始切片构建并融合投影 (Q/K/V 与 Gate/Up 沿输出维度拼接)
        pub fn fromBlocks(allocator: std.mem.Allocator, cfg: Gemma4Config, layer_idx: usize, blocks: BlockSlices) !Projections {
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

        /// 从预重排直接内存布局 (Direct Layout) 构建投影，无需 CPU 解包
        pub fn fromDirect(cfg: Gemma4Config, layer_idx: usize, direct: DirectSlices) !Projections {
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

    /// 单层各投影的原始 Q4Block 字节切片；v 为 null 表示全局注意力层 K 与 V 共享投影
    pub const BlockSlices = struct {
        q: []const u8,
        k: []const u8,
        v: ?[]const u8,
        o: []const u8,
        gate: []const u8,
        up: []const u8,
        down: []const u8,
    };

    /// 直通布局中单个投影矩阵的三段式切片 (w_q, scales, biases)
    pub const DirectMatrix = struct {
        wq: []const u8,
        sc: []const u8,
        bi: []const u8,
    };

    /// 直通布局中单层四个融合投影矩阵
    pub const DirectSlices = struct {
        qkv: DirectMatrix,
        o: DirectMatrix,
        gate_up: DirectMatrix,
        down: DirectMatrix,
    };

    /// 单层六组均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm) 权重
    pub const NormWeights = struct {
        input: []const f32,
        q: []const f32,
        k: []const f32,
        post_attn: []const f32,
        pre_mlp: []const f32,
        post_mlp: []const f32,
    };

    /// 由已构建的投影权重与层归一化参数组装第 layer_idx 层。
    /// 成功时接管 `projections` 的所有权；失败时所有权仍归调用方。
    pub fn init(
        cfg: Gemma4Config,
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

        var in_w = try mlx.Array.fromSlice(norms.input, &.{norms.input.len});
        errdefer in_w.deinit();
        var qn_w = try mlx.Array.fromSlice(norms.q, &.{norms.q.len});
        errdefer qn_w.deinit();
        var kn_w = try mlx.Array.fromSlice(norms.k, &.{norms.k.len});
        errdefer kn_w.deinit();
        var pa_w = try mlx.Array.fromSlice(norms.post_attn, &.{norms.post_attn.len});
        errdefer pa_w.deinit();
        var pm_w = try mlx.Array.fromSlice(norms.pre_mlp, &.{norms.pre_mlp.len});
        errdefer pm_w.deinit();
        var pom_w = try mlx.Array.fromSlice(norms.post_mlp, &.{norms.post_mlp.len});
        errdefer pom_w.deinit();
        const rope = try GpuRope.init(spec.head_dim, spec.attn_type.partialRotaryFactor(), spec.attn_type.ropeTheta());

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
            .attn_type = spec.attn_type,
            .sliding_window = cfg.sliding_window,
            .num_heads = cfg.num_attention_heads,
            .num_kv_heads = spec.num_kv_heads,
            .head_dim = spec.head_dim,
            .rope = rope,
            .rms_norm_eps = cfg.rms_norm_eps,
            .compiled_mlp = null,
        };
    }

    fn heads(self: *const Gemma4GpuLayer) offload.AttentionHeads {
        return .{ .num_heads = self.num_heads, .num_kv_heads = self.num_kv_heads, .head_dim = self.head_dim };
    }

    /// 门控多层感知机残差块：x2 = (x1 + post_mlp_norm(down_proj(GeGLU(fused_gate_up(pre_mlp_norm(x1)))))) * layer_scalar
    pub fn mlpResidual(self: *const Gemma4GpuLayer, x1: mlx.Array, stream: ?Stream) !mlx.Array {
        var pre_mlp = try mlx.Array.fastRmsNorm(x1, self.pre_mlp_w, self.rms_norm_eps, stream);
        defer pre_mlp.deinit();
        var gate_up = try self.fused_gate_up.forwardArray(pre_mlp, stream);
        defer gate_up.deinit();
        var fused = try gate_up.geglu(self.intermediate_size, stream);
        defer fused.deinit();
        var down = try self.down_proj.forwardArray(fused, stream);
        defer down.deinit();
        var post_mlp = try mlx.Array.fastRmsNorm(down, self.post_mlp_w, self.rms_norm_eps, stream);
        defer post_mlp.deinit();
        var x2 = try mlx.Array.add(x1, post_mlp, stream);
        if (self.layer_scalar == 1.0) return x2;
        defer x2.deinit();
        return x2.mulScalar(self.layer_scalar, stream);
    }

    /// 执行门控多层感知机残差块：已编译宏内核闭包时直接派发单次融合调用，否则逐算子调度 `mlpResidual`
    fn runMlp(self: *const Gemma4GpuLayer, x1: mlx.Array, stream: ?Stream) !mlx.Array {
        if (self.compiled_mlp) |c_mlp| {
            var in_vec = mlx.VectorArray.fromArrays(&.{x1});
            defer in_vec.deinit();
            var out_vec = try c_mlp.apply(in_vec);
            defer out_vec.deinit();
            return out_vec.get(0) orelse error.MlxOperationFailed;
        }
        return self.mlpResidual(x1, stream);
    }

    /// 使用 Apple MLX Metal 即时编译 (Just-In-Time Compilation, JIT) 编译门控多层感知机 (Multi-Layer Perceptron, MLP) 宏内核闭包。
    /// 闭包持有指向本层的指针，调用方须保证本层在闭包存续期间地址不变 (`Gemma4GpuContext` 的层切片满足该要求)。
    pub fn initCompiledMlp(self: *Gemma4GpuLayer) !void {
        if (!mlx.has_mlx) return;
        if (self.compiled_mlp != null) return;

        const Helper = struct {
            fn compute(vres: [*c]mlx.c.mlx_vector_array, in: mlx.c.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
                if (!mlx.has_mlx) return 1;
                const lw: *const Gemma4GpuLayer = @ptrCast(@alignCast(payload orelse return 1));

                var x_handle: mlx.c.mlx_array = .{ .ctx = null };
                if (mlx.c.mlx_vector_array_get(&x_handle, in, 0) != 0 or x_handle.ctx == null) return 1;
                // mlx_vector_array_get 为取出的元素分配了新的数组句柄，用完后需释放
                var x1 = mlx.Array{ .handle = x_handle };
                defer x1.deinit();

                var res_arr = lw.mlpResidual(x1, null) catch return 1;
                defer res_arr.deinit();

                const res_vec = mlx.c.mlx_vector_array_new_data(&[_]mlx.c.mlx_array{res_arr.handle}, 1);
                _ = mlx.c.mlx_vector_array_set(vres, res_vec);
                _ = mlx.c.mlx_vector_array_free(res_vec);
                return 0;
            }
        };

        self.compiled_mlp = try mlx.compile(Helper.compute, self, false);
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

// ============================================================================
// Gemma 4 12B 端到端 GPU 解码上下文
// ============================================================================

/// 图形处理器 (Graphics Processing Unit, GPU) 端到端整网自回归增量解码上下文 (End-to-End GPU Decode Context)
/// 将全部 Transformer 解码层、均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)、
/// 旋转位置编码 (Rotary Position Embedding, RoPE)、自回归键值缓存 (Key-Value Cache, KVCache) 与
/// 语言模型输出头 (Language Model Head, LM Head) 完全常驻在 Apple Silicon 统一内存显存中。
/// 每生成一个词元仅触发一次最终求值 (Single Evaluation)，消除逐层 CPU-GPU 往返同步开销。
///
/// 所有权：上下文独占持有 `layers` 切片 (及其中各层权重) 与 `lm_head`，并在 `deinit` 中释放它们。
pub const Gemma4GpuContext = struct {
    allocator: std.mem.Allocator,
    layers: []Gemma4GpuLayer,
    final_norm_w: mlx.Array,
    /// 由上下文独占持有的语言模型输出头，在 `deinit` 中释放
    lm_head: LmHeadWeights,
    /// 与 `config.sampling_excluded_tokens` 一致的 GPU 词表排除掩码
    exclusion_mask: TokenExclusionMask,
    config: Gemma4Config,
    max_seq_len: usize,
    stream: Stream,

    gpu_k: []?mlx.Array,
    gpu_v: []?mlx.Array,
    curr_len: usize = 0,
    sqrt_d: f32 = 0.0,

    /// 构建 GPU 解码上下文。成功时接管 `layers` (须由 `allocator` 分配) 与 `lm_head` 的所有权；
    /// 失败时两者的所有权仍归调用方，由调用方负责释放。
    pub fn init(
        allocator: std.mem.Allocator,
        cfg: Gemma4Config,
        layers: []Gemma4GpuLayer,
        final_norm_data: []const f32,
        lm_head: LmHeadWeights,
        max_seq_len: usize,
    ) !Gemma4GpuContext {
        if (!offload.isAvailable()) return error.OffloadNotSupported;
        if (layers.len != cfg.num_hidden_layers or final_norm_data.len != cfg.hidden_size) return error.ShapeMismatch;
        if (lm_head.vocab_size != cfg.vocab_size or lm_head.hidden_size != cfg.hidden_size) return error.ShapeMismatch;

        var final_norm_w = try mlx.Array.fromSlice(final_norm_data, &.{cfg.hidden_size});
        errdefer final_norm_w.deinit();
        final_norm_w.eval();

        var exclusion_mask = try TokenExclusionMask.init(allocator, cfg.vocab_size, cfg.sampling_excluded_tokens);
        errdefer exclusion_mask.deinit();

        const gpu_k = try allocator.alloc(?mlx.Array, layers.len);
        errdefer allocator.free(gpu_k);
        const gpu_v = try allocator.alloc(?mlx.Array, layers.len);
        errdefer allocator.free(gpu_v);
        @memset(gpu_k, null);
        @memset(gpu_v, null);

        const stream = try Stream.defaultGpu();

        mlx.enableCompile();
        mlx.setCompileMode(.enabled);

        // 宏内核编译失败时该层退回逐算子调度 `mlpResidual`，结果一致
        for (layers) |*lw| {
            lw.initCompiledMlp() catch {};
        }

        return .{
            .allocator = allocator,
            .layers = layers,
            .final_norm_w = final_norm_w,
            .lm_head = lm_head,
            .exclusion_mask = exclusion_mask,
            .config = cfg,
            .max_seq_len = max_seq_len,
            .stream = stream,
            .gpu_k = gpu_k,
            .gpu_v = gpu_v,
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
        self.allocator.free(self.gpu_k);
        self.allocator.free(self.gpu_v);
    }

    /// 重置 GPU 显存中的全部键值缓存
    pub fn resetCache(self: *Gemma4GpuContext) void {
        for (self.gpu_k, self.gpu_v) |*k_opt, *v_opt| {
            if (k_opt.*) |*k| k.deinit();
            if (v_opt.*) |*v| v.deinit();
            k_opt.* = null;
            v_opt.* = null;
        }
        self.curr_len = 0;
    }

    /// 以常驻 LM Head 对最终隐状态 h_last 打分并选取下一个词元 (与上下文解码路径使用相同的排除与惩罚语义)
    pub fn sampleHidden(
        self: *const Gemma4GpuContext,
        h_last: []const f32,
        recent_tokens: []const usize,
        repetition_penalty: f32,
        stream: ?Stream,
    ) !usize {
        return selectTokenFromHidden(&self.lm_head, &self.exclusion_mask, self.config, h_last, recent_tokens, repetition_penalty, stream orelse self.stream);
    }

    /// 将词元序列的脑浮点 16 (Brain Floating Point 16, BF16) 嵌入行展开为单精度浮点并乘以 sqrt(hidden_size)，
    /// 写入 [tokens.len, hidden_size] 行主序缓冲区
    fn embedTokens(self: *const Gemma4GpuContext, tokens: []const usize, embed_u16: []const u16, out: []f32) !void {
        const h = self.config.hidden_size;
        std.debug.assert(out.len == tokens.len * h);
        for (tokens, 0..) |tok, t| {
            if (tok >= self.config.vocab_size or (tok + 1) * h > embed_u16.len) return error.TokenOutOfBounds;
            const emb_row = embed_u16[tok * h ..][0..h];
            for (out[t * h ..][0..h], emb_row) |*d, e| {
                d.* = @as(f32, @bitCast(@as(u32, e) << 16)) * self.sqrt_d;
            }
        }
    }

    /// 将新增键值 [1, num_kv_heads, t_new, head_dim] 沿序列轴 (axis 2) 追加至第 l 层 GPU 键值缓存
    fn appendKv(self: *Gemma4GpuContext, l: usize, k_new: mlx.Array, v_new: mlx.Array, s: Stream) !void {
        if (self.gpu_k[l]) |old_k| {
            var new_k = try mlx.Array.concatenateAxis(&.{ old_k, k_new }, 2, s);
            errdefer new_k.deinit();
            const new_v = try mlx.Array.concatenateAxis(&.{ self.gpu_v[l].?, v_new }, 2, s);
            self.replaceKv(l, new_k, new_v);
        } else {
            var cloned_k = try k_new.mulScalar(1.0, s);
            errdefer cloned_k.deinit();
            const cloned_v = try v_new.mulScalar(1.0, s);
            self.replaceKv(l, cloned_k, cloned_v);
        }
    }

    /// 仅保留第 l 层键值缓存序列轴上 [start, stop) 区间的词元
    fn sliceKv(self: *Gemma4GpuContext, l: usize, start: usize, stop: usize, s: Stream) !void {
        var new_k = try self.gpu_k[l].?.sliceAxis(2, start, stop, s);
        errdefer new_k.deinit();
        const new_v = try self.gpu_v[l].?.sliceAxis(2, start, stop, s);
        self.replaceKv(l, new_k, new_v);
    }

    /// 以新数组替换第 l 层键值缓存并释放旧数组
    fn replaceKv(self: *Gemma4GpuContext, l: usize, new_k: mlx.Array, new_v: mlx.Array) void {
        if (self.gpu_k[l]) |*old_k| old_k.deinit();
        if (self.gpu_v[l]) |*old_v| old_v.deinit();
        self.gpu_k[l] = new_k;
        self.gpu_v[l] = new_v;
    }

    /// 滑动窗口注意力层的键值缓存只保留最近 sliding_window 个词元（截取序列尾部）
    fn trimSlidingKv(self: *Gemma4GpuContext, l: usize, lw: *const Gemma4GpuLayer, s: Stream) !void {
        if (lw.attn_type != .sliding_attention) return;
        const len = self.gpu_k[l].?.dimSize(2);
        if (len > lw.sliding_window) try self.sliceKv(l, len - lw.sliding_window, len, s);
    }

    /// 校验 CPU 层缓存与 GPU 解码层的几何规格一致
    fn checkCacheLayout(self: *const Gemma4GpuContext, cache: *const Gemma4KVCache) !void {
        if (cache.layers.len != self.layers.len) return error.ShapeMismatch;
        for (self.layers, cache.layers) |lw, cl| {
            if (cl.num_kv_heads != lw.num_kv_heads or cl.head_dim != lw.head_dim or cl.attn_type != lw.attn_type) {
                return error.ShapeMismatch;
            }
        }
    }

    /// 将 CPU 预填充 / 解码得到的键值缓存 (各层可见窗口) 同步导入 GPU 显存，GPU 解码位置随之设为 CPU 缓存游标
    pub fn syncCacheFromCpu(self: *Gemma4GpuContext, cache: *const Gemma4KVCache) !void {
        try self.checkCacheLayout(cache);
        self.resetCache();
        const cached_len = if (cache.layers.len > 0) cache.layers[0].curr_len else 0;
        if (cached_len == 0) return;
        if (cached_len >= self.max_seq_len) return error.ContextWindowExceeded;

        for (self.layers, cache.layers, 0..) |lw, *cl, l_idx| {
            if (cl.curr_len != cached_len) return error.ShapeMismatch;
            const win = cl.windowLen();
            const packed_len = lw.num_kv_heads * win * lw.head_dim;

            const pack_buf_k = try self.allocator.alloc(f32, packed_len);
            defer self.allocator.free(pack_buf_k);
            const pack_buf_v = try self.allocator.alloc(f32, packed_len);
            defer self.allocator.free(pack_buf_v);
            try cl.packWindow(pack_buf_k, pack_buf_v);

            var k_arr = try mlx.Array.fromSlice(pack_buf_k, &.{ 1, lw.num_kv_heads, win, lw.head_dim });
            errdefer k_arr.deinit();
            k_arr.eval();
            var v_arr = try mlx.Array.fromSlice(pack_buf_v, &.{ 1, lw.num_kv_heads, win, lw.head_dim });
            v_arr.eval();

            self.replaceKv(l_idx, k_arr, v_arr);
        }
        self.curr_len = cached_len;
    }

    /// 将 GPU 显存中的键值缓存 (各层可见窗口) 写回 CPU 层缓存并把 CPU 游标设为 GPU 解码位置，
    /// 使 GPU 预填充 / 解码之后可以无缝切换到 CPU 增量解码
    pub fn syncCacheToCpu(self: *const Gemma4GpuContext, cache: *Gemma4KVCache) !void {
        try self.checkCacheLayout(cache);
        for (self.layers, cache.layers, 0..) |lw, *cl, l_idx| {
            if (self.curr_len == 0) {
                cl.reset();
                continue;
            }
            const k = self.gpu_k[l_idx] orelse return error.ShapeMismatch;
            const v = self.gpu_v[l_idx] orelse return error.ShapeMismatch;
            const win = k.dimSize(2);
            const packed_len = lw.num_kv_heads * win * lw.head_dim;

            const buf_k = try self.allocator.alloc(f32, packed_len);
            defer self.allocator.free(buf_k);
            const buf_v = try self.allocator.alloc(f32, packed_len);
            defer self.allocator.free(buf_v);
            try k.copyToSlice(buf_k);
            try v.copyToSlice(buf_v);
            try cl.writeWindow(buf_k, buf_v, win, self.curr_len);
        }
    }

    /// 单层注意力投影输出：旋转后的查询 [1, heads, t, head_dim] 与旋转后的键 / 归一化后的值 [1, kv_heads, t, head_dim]
    const QkvHeads = struct {
        q_rot: mlx.Array,
        k_rot: mlx.Array,
        v: mlx.Array,

        fn deinit(self: *QkvHeads) void {
            self.q_rot.deinit();
            self.k_rot.deinit();
            self.v.deinit();
        }
    };

    /// 将 [1, t, n * head_dim] 投影结果按头执行均方根层归一化 (weight 为 null 时为无参数归一化)，输出 [1, n, t, head_dim]
    fn normHeads(raw: mlx.Array, weight: ?mlx.Array, t: usize, n: usize, lw: *const Gemma4GpuLayer, s: Stream) !mlx.Array {
        if (t == 1) {
            var heads_4d = try raw.reshape(&.{ 1, n, 1, lw.head_dim }, s);
            defer heads_4d.deinit();
            return mlx.Array.fastRmsNorm(heads_4d, weight, lw.rms_norm_eps, s);
        }
        var heads_4d = try raw.reshape(&.{ 1, t, n, lw.head_dim }, s);
        defer heads_4d.deinit();
        var normed = try mlx.Array.fastRmsNorm(heads_4d, weight, lw.rms_norm_eps, s);
        defer normed.deinit();
        return normed.transposeAxes(&.{ 0, 2, 1, 3 }, s);
    }

    /// 对 [1, t, hidden_size] 输入执行融合 QKV 量化投影、Q/K/V 头归一化、转置与旋转位置编码 (起始绝对位置为 pos)
    fn projectQkv(lw: *const Gemma4GpuLayer, in_norm: mlx.Array, t: usize, pos: usize, s: Stream) !QkvHeads {
        var qkv_raw = try lw.fused_qkv.forwardArray(in_norm, s);
        defer qkv_raw.deinit();

        var q_raw = try qkv_raw.sliceAxis(2, 0, lw.q_dim, s);
        defer q_raw.deinit();
        var k_raw = try qkv_raw.sliceAxis(2, lw.q_dim, lw.q_dim + lw.k_dim, s);
        defer k_raw.deinit();
        // 全局注意力层 attention_k_eq_v：V 与 K 共享同一投影结果
        var v_raw = if (lw.v_dim > 0)
            try qkv_raw.sliceAxis(2, lw.q_dim + lw.k_dim, lw.q_dim + lw.k_dim + lw.v_dim, s)
        else
            try k_raw.sliceAxis(2, 0, lw.k_dim, s);
        defer v_raw.deinit();

        // 在连续内存上执行 Q-Norm / K-Norm / 无参数 V-Norm，再排布为 [1, heads, t, head_dim]：
        // t == 1 时 [1, 1, heads, head_dim] 与 [1, heads, 1, head_dim] 内存布局相同，直接重塑而无需转置
        var q_normed = try normHeads(q_raw, lw.q_norm_w, t, lw.num_heads, lw, s);
        defer q_normed.deinit();
        var k_normed = try normHeads(k_raw, lw.k_norm_w, t, lw.num_kv_heads, lw, s);
        defer k_normed.deinit();
        var v_normed = try normHeads(v_raw, null, t, lw.num_kv_heads, lw, s);
        errdefer v_normed.deinit();

        // 旋转位置编码：滑动窗口层为标准 RoPE，全局注意力层为在完整 head_dim 上旋转的比例旋转位置编码
        var q_rot = try lw.rope.apply(q_normed, pos, s);
        errdefer q_rot.deinit();
        const k_rot = try lw.rope.apply(k_normed, pos, s);
        return .{ .q_rot = q_rot, .k_rot = k_rot, .v = v_normed };
    }

    /// 注意力之后的公共尾部：输出投影 -> 注意力后层归一化 -> 残差相加 -> 门控多层感知机残差块
    fn layerTail(lw: *const Gemma4GpuLayer, x_cur: mlx.Array, attn_out: mlx.Array, t: usize, use_compiled: bool, s: Stream) !mlx.Array {
        // t == 1 时 [1, heads, 1, head_dim] 可直接重塑为 [1, 1, attn_dim]
        var attn_t: ?mlx.Array = if (t == 1) null else try attn_out.transposeAxes(&.{ 0, 2, 1, 3 }, s);
        defer if (attn_t) |*a| a.deinit();
        var attn_2d = try (attn_t orelse attn_out).reshape(&.{ 1, t, lw.attn_dim }, s);
        defer attn_2d.deinit();
        var o_out = try lw.o_proj.forwardArray(attn_2d, s);
        defer o_out.deinit();

        var post_attn = try mlx.Array.fastRmsNorm(o_out, lw.post_attn_w, lw.rms_norm_eps, s);
        defer post_attn.deinit();
        var x1 = try mlx.Array.add(x_cur, post_attn, s);
        defer x1.deinit();
        return if (use_compiled) lw.runMlp(x1, s) else lw.mlpResidual(x1, s);
    }

    /// 最终层归一化：取 [1, t, hidden_size] 的最后 rows 行，返回 [rows, hidden_size]
    fn finalNorm(self: *const Gemma4GpuContext, x_cur: mlx.Array, t: usize, rows: usize, s: Stream) !mlx.Array {
        var tail = try x_cur.sliceAxis(1, t - rows, t, s);
        defer tail.deinit();
        var tail_2d = try tail.reshape(&.{ rows, self.config.hidden_size }, s);
        defer tail_2d.deinit();
        return mlx.Array.fastRmsNorm(tail_2d, self.final_norm_w, self.config.rms_norm_eps, s);
    }

    /// 执行全 GPU 序列级并行预填充 (Prefill)，一次性完成整个 Prompt 词元序列的矩阵乘法与 KVCache 初始化，返回首个预测词元
    pub fn prefill(
        self: *Gemma4GpuContext,
        prompt_tokens: []const usize,
        embed_u16: []const u16,
        repetition_penalty: f32,
        stream: ?Stream,
    ) !usize {
        const T = prompt_tokens.len;
        if (T == 0) return error.EmptyPrompt;
        if (T >= self.max_seq_len) return error.ContextWindowExceeded;

        const s = stream orelse self.stream;
        self.resetCache();

        // 1. 构建 Prompt 词元嵌入 [1, T, hidden_size]
        const tok_emb_buf = try self.allocator.alloc(f32, T * self.config.hidden_size);
        defer self.allocator.free(tok_emb_buf);
        try self.embedTokens(prompt_tokens, embed_u16, tok_emb_buf);
        var x_cur = try mlx.Array.fromSlice(tok_emb_buf, &.{ 1, T, self.config.hidden_size });
        defer x_cur.deinit();

        for (self.layers, 0..) |*lw, l| {
            // (1) 输入均方根层归一化与 QKV 投影、头归一化、旋转位置编码
            var in_norm = try mlx.Array.fastRmsNorm(x_cur, lw.input_norm_w, lw.rms_norm_eps, s);
            defer in_norm.deinit();
            var qkv = try projectQkv(lw, in_norm, T, 0, s);
            defer qkv.deinit();

            // (2) 写入 GPU 键值缓存 (滑动窗口层只保留最近 sliding_window 个词元)
            if (lw.attn_type == .sliding_attention and T > lw.sliding_window) {
                var k_win = try qkv.k_rot.sliceAxis(2, T - lw.sliding_window, T, s);
                errdefer k_win.deinit();
                const v_win = try qkv.v.sliceAxis(2, T - lw.sliding_window, T, s);
                self.replaceKv(l, k_win, v_win);
            } else {
                var k_copy = try qkv.k_rot.mulScalar(1.0, s);
                errdefer k_copy.deinit();
                const v_copy = try qkv.v.mulScalar(1.0, s);
                self.replaceKv(l, k_copy, v_copy);
            }

            // (3) 带因果与滑动窗口掩码的缩放点积注意力 (scale = 1.0)
            var attn_out = if (lw.attn_type == .full_attention or T <= lw.sliding_window)
                try mlx.Array.fastScaledDotProductAttention(qkv.q_rot, qkv.k_rot, qkv.v, 1.0, true, s)
            else
                try offload.slidingWindowAttention(self.allocator, lw.heads(), qkv.q_rot, qkv.k_rot, qkv.v, 1.0, lw.sliding_window, T, T, 0, 0, s);
            defer attn_out.deinit();

            // (4) 输出投影、残差与门控多层感知机 (预填充序列长度可变，逐算子调度以避免形状特化闭包重编译)
            const x_next = try layerTail(lw, x_cur, attn_out, T, false, s);
            x_cur.deinit();
            x_cur = x_next;
        }

        // 最终层归一化与 LM Head 词表投影、词元选取
        var h_norm = try self.finalNorm(x_cur, T, 1, s);
        defer h_norm.deinit();
        var logits = try self.lm_head.forwardArrayBf16(h_norm, s);
        defer logits.deinit();
        const next = try selectToken(logits, self.config, &self.exclusion_mask, gemma4.recentWindow(prompt_tokens), repetition_penalty, s);
        self.curr_len = T;
        return next;
    }

    /// 执行单步端到端整网自回归增量解码 (Decode Step)：输入当前词元，全程在 GPU 计算图中惰性调度，返回下一个预测词元
    pub fn decodeStep(
        self: *Gemma4GpuContext,
        token: usize,
        embed_u16: []const u16,
        recent_tokens: []const usize,
        repetition_penalty: f32,
        stream: ?Stream,
    ) !usize {
        const pos = self.curr_len;
        if (pos >= self.max_seq_len) return error.ContextWindowExceeded;
        const s = stream orelse self.stream;

        const tok_emb_buf = try self.allocator.alloc(f32, self.config.hidden_size);
        defer self.allocator.free(tok_emb_buf);
        try self.embedTokens(&.{token}, embed_u16, tok_emb_buf);
        var x_cur = try mlx.Array.fromSlice(tok_emb_buf, &.{ 1, 1, self.config.hidden_size });
        defer x_cur.deinit();

        for (self.layers, 0..) |*lw, l| {
            var in_norm = try mlx.Array.fastRmsNorm(x_cur, lw.input_norm_w, lw.rms_norm_eps, s);
            defer in_norm.deinit();
            var qkv = try projectQkv(lw, in_norm, 1, pos, s);
            defer qkv.deinit();

            // 增量写入 GPU 键值缓存，滑动窗口层随后只保留最近 sliding_window 个词元
            try self.appendKv(l, qkv.k_rot, qkv.v, s);
            try self.trimSlidingKv(l, lw, s);

            // 单查询注意力：缓存中的全部键均可见 (滑动窗口层缓存已截取至窗口长度)
            var attn_out = try mlx.Array.fastScaledDotProductAttention(qkv.q_rot, self.gpu_k[l].?, self.gpu_v[l].?, 1.0, false, s);
            defer attn_out.deinit();

            // 单步解码形状固定，启用静态编译的门控多层感知机宏内核
            const x_next = try layerTail(lw, x_cur, attn_out, 1, true, s);
            x_cur.deinit();
            x_cur = x_next;
        }

        var h_norm = try self.finalNorm(x_cur, 1, 1, s);
        defer h_norm.deinit();
        var logits = try self.lm_head.forwardArrayBf16(h_norm, s);
        defer logits.deinit();
        const next = try selectToken(logits, self.config, &self.exclusion_mask, gemma4.recentWindow(recent_tokens), repetition_penalty, s);
        self.curr_len = pos + 1;
        return next;
    }

    /// 执行投机采样批量验证前向传播 (Speculative Draft Verification Step)
    /// 一次性将基础词元 (input_tokens[0]) 与投机候选词元 (input_tokens[1..M]) 在 GPU 上完成因果注意力与全网前向传播，
    /// 并逐位贪婪校验接受候选词元 (每一位均先截断再施加重复惩罚，排除词元与单步解码一致)。
    /// 返回值：实际被接受并写入 KVCache 的词元数量 (1 <= num_accepted <= input_tokens.len)
    pub fn decodeSpeculative(
        self: *Gemma4GpuContext,
        input_tokens: []const usize,
        embed_u16: []const u16,
        recent_tokens: []const usize,
        repetition_penalty: f32,
        stream: ?Stream,
        accepted_tokens: []usize,
    ) !usize {
        const M = input_tokens.len;
        if (M == 0) return 0;
        if (accepted_tokens.len < M) return error.ShapeMismatch;
        const pos = self.curr_len;
        if (pos + M >= self.max_seq_len) return error.ContextWindowExceeded;

        const s = stream orelse self.stream;
        const V = self.config.vocab_size;

        // 1. 构建 M 个词元的嵌入向量 [1, M, hidden_size]
        const tok_emb_buf = try self.allocator.alloc(f32, M * self.config.hidden_size);
        defer self.allocator.free(tok_emb_buf);
        try self.embedTokens(input_tokens, embed_u16, tok_emb_buf);
        var x_cur = try mlx.Array.fromSlice(tok_emb_buf, &.{ 1, M, self.config.hidden_size });
        defer x_cur.deinit();

        for (self.layers, 0..) |*lw, l| {
            var in_norm = try mlx.Array.fastRmsNorm(x_cur, lw.input_norm_w, lw.rms_norm_eps, s);
            defer in_norm.deinit();
            var qkv = try projectQkv(lw, in_norm, M, pos, s);
            defer qkv.deinit();

            // 增量追加至 GPU 键值缓存（滑动窗口层在本步结束后再截取最近 sliding_window 个词元）
            try self.appendKv(l, qkv.k_rot, qkv.v, s);
            const kv_len = self.gpu_k[l].?.dimSize(2);

            // 因果注意力：缓存中的第 j 个键对应绝对位置 pos + M - kv_len + j；
            // 滑动窗口层的缓存超过窗口长度时，较早的查询会看到窗口之外的键，必须改用显式滑动窗口掩码
            var attn_out = if (lw.attn_type == .full_attention or kv_len <= lw.sliding_window)
                try mlx.Array.fastScaledDotProductAttention(qkv.q_rot, self.gpu_k[l].?, self.gpu_v[l].?, 1.0, true, s)
            else
                try offload.slidingWindowAttention(self.allocator, lw.heads(), qkv.q_rot, self.gpu_k[l].?, self.gpu_v[l].?, 1.0, lw.sliding_window, M, kv_len, pos, pos + M - kv_len, s);
            defer attn_out.deinit();

            const x_next = try layerTail(lw, x_cur, attn_out, M, true, s);
            x_cur.deinit();
            x_cur = x_next;
        }

        // 2. 最终层 RMSNorm、LM Head 词表投影与最终对数几率截断
        var h_norm = try self.finalNorm(x_cur, M, M, s);
        defer h_norm.deinit();
        var logits_raw = try self.lm_head.forwardArrayBf16(h_norm, s);
        defer logits_raw.deinit();
        var logits = try softcapLogits(logits_raw, self.config.final_logit_softcapping, s);
        defer logits.deinit();
        const logits_slice = try logits.dataFloat32();
        if (logits_slice.len != M * V) return error.ShapeMismatch;

        // 3. 逐位贪婪校验候选词元 (Greedy Verification with Penalty)
        var verified_tokens = try std.ArrayList(usize).initCapacity(self.allocator, recent_tokens.len + M);
        defer verified_tokens.deinit(self.allocator);
        try verified_tokens.appendSlice(self.allocator, recent_tokens);

        var num_accepted: usize = 0;
        for (0..M) |step_i| {
            const row = logits_slice[step_i * V .. (step_i + 1) * V];
            const pred = sampling.argmaxWithPenalty(row, V, self.config.sampling_excluded_tokens, gemma4.recentWindow(verified_tokens.items), repetition_penalty);
            accepted_tokens[step_i] = pred;
            num_accepted += 1;
            try verified_tokens.append(self.allocator, pred);

            // 投机命中校验：若预测词元与下一个投机候选词元不一致，则停止接受后续投机词元
            if (step_i + 1 < M and pred != input_tokens[step_i + 1]) break;
        }

        // 4. 回滚未被接受的投机词元 KVCache（从缓存尾部丢弃，滑动窗口层的缓存并不从位置 0 开始），
        //    随后滑动窗口层只保留最近 sliding_window 个词元 (惰性图切片，零同步开销)
        const rejected = M - num_accepted;
        for (self.layers, 0..) |*lw, l| {
            if (rejected > 0) {
                const len = self.gpu_k[l].?.dimSize(2);
                try self.sliceKv(l, 0, len - rejected, s);
            }
            try self.trimSlidingKv(l, lw, s);
        }

        self.curr_len = pos + num_accepted;
        return num_accepted;
    }

    /// 执行 Metal 计算管线预热 (Pipeline Pre-warming)，通过微型前向传递触发 MSL JIT 编译并构建管线状态对象 (PSO)
    pub fn warmup(self: *Gemma4GpuContext, embed_u16: []const u16, stream: ?Stream) !void {
        if (self.config.vocab_size < 3) return;
        const dummy_prompt = [_]usize{ 1, 2 };
        _ = try self.prefill(&dummy_prompt, embed_u16, 1.0, stream);
        _ = try self.decodeStep(1, embed_u16, &dummy_prompt, 1.0, stream);

        var accepted_dummy: [2]usize = undefined;
        _ = self.decodeSpeculative(&dummy_prompt, embed_u16, &dummy_prompt, 1.0, stream, &accepted_dummy) catch {};

        self.resetCache();
        (stream orelse self.stream).synchronize();
    }
};
