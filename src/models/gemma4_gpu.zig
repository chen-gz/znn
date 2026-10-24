//! Gemma 4 图形处理器 (Graphics Processing Unit, GPU) 推理共享组件与 12B 端到端解码上下文
//!
//! 包含 E4B 与 12B 两条 Apple 机器学习拓展 (Machine Learning Explore, MLX) 推理路径共用的
//! 比例旋转位置编码 (Proportional Rotary Position Embedding, Proportional RoPE)、最终对数几率截断与词元选取
//! (贪婪 argmax 与设备端随机采样候选选择)，
//! 以及 12B Q4 模型的常驻显存解码层 (`Gemma4GpuLayer`) 与整网自回归解码上下文 (`Gemma4GpuContext`)。
//! backend 包只提供与模型无关的 MLX / 卸载原语，Gemma 4 专属的层结构、键值缓存管理与采样策略集中在本文件。

const std = @import("std");
const backend = @import("../backend.zig");
const tensor = @import("../tensor.zig");
const gemma4 = @import("gemma4.zig");

const mlx = backend.mlx;
const offload = backend.offload;
const sampling = backend.sampling;
const Sampler = sampling.Sampler;
const SamplingOptions = sampling.SamplingOptions;
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

pub const GpuRope = backend.mlx_blocks.Rope;

// ============================================================================
// 对数几率截断与词元选取 (E4B 与 12B 共用)
// ============================================================================

pub const softcapLogits = backend.logits_processor.softcapLogits;
pub const penalizeDevice = backend.logits_processor.penalizeDevice;
pub const DeviceCandidates = backend.logits_processor.DeviceCandidates;
pub const HostCandidates = backend.logits_processor.HostCandidates;

fn processorOpts(cfg: Gemma4Config) backend.LogitsProcessorOptions {
    return .{
        .vocab_size = cfg.vocab_size,
        .final_logit_softcapping = cfg.final_logit_softcapping,
        .sampling_excluded_tokens = cfg.sampling_excluded_tokens,
    };
}

pub fn deviceCandidates(
    allocator: std.mem.Allocator,
    logits: mlx.Array,
    cfg: Gemma4Config,
    mask: *const TokenExclusionMask,
    recent_tokens: []const usize,
    options: SamplingOptions,
    s: ?Stream,
) !DeviceCandidates {
    return backend.logits_processor.deviceCandidates(allocator, logits, processorOpts(cfg), mask, recent_tokens, options, s);
}

pub fn selectToken(
    allocator: std.mem.Allocator,
    logits: mlx.Array,
    cfg: Gemma4Config,
    mask: *const TokenExclusionMask,
    recent_tokens: []const usize,
    sampler: *Sampler,
    s: ?Stream,
) !usize {
    return backend.logits_processor.selectToken(allocator, logits, processorOpts(cfg), mask, recent_tokens, sampler, s);
}

pub fn selectTokenFromHidden(
    allocator: std.mem.Allocator,
    lm_head: anytype,
    mask: *const TokenExclusionMask,
    cfg: Gemma4Config,
    h_last: []const f32,
    recent_tokens: []const usize,
    sampler: *Sampler,
    s: ?Stream,
) !usize {
    const head: backend.offload.LmHead = switch (@TypeOf(lm_head.*)) {
        backend.offload.LmHead => lm_head.*,
        backend.offload.LmHeadWeights => .{ .dense = lm_head.* },
        backend.offload.QuantWeights => .{ .quant = lm_head.* },
        else => @compileError("Unsupported lm_head type"),
    };
    return backend.logits_processor.selectTokenFromHidden(allocator, &head, processorOpts(cfg), mask, h_last, recent_tokens, sampler, s);
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
    /// 门控多层感知机的高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 形式 (取自 `Gemma4Config.hidden_activation`)
    hidden_activation: backend.GeluApproximation,
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
        const rope = try GpuRope.init(spec.head_dim, spec.attn_type.ropeOptions());

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
            .hidden_activation = cfg.hidden_activation,
            .compiled_mlp = null,
        };
    }

    /// 门控多层感知机残差块：x2 = (x1 + post_mlp_norm(down_proj(GeGLU(fused_gate_up(pre_mlp_norm(x1)))))) * layer_scalar
    pub fn mlpResidual(self: *const Gemma4GpuLayer, x1: mlx.Array, stream: ?Stream) !mlx.Array {
        var pre_mlp = try mlx.Array.fastRmsNorm(x1, self.pre_mlp_w, self.rms_norm_eps, stream);
        defer pre_mlp.deinit();
        var gate_up = try self.fused_gate_up.forwardArray(pre_mlp, stream);
        defer gate_up.deinit();
        var fused = try gate_up.geglu(self.intermediate_size, self.hidden_activation, stream);
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

        const mlpRun = struct {
            fn run(lw: *const Gemma4GpuLayer, in: [1]mlx.Array) anyerror!mlx.Array {
                return lw.mlpResidual(in[0], null);
            }
        }.run;

        self.compiled_mlp = try mlx.compileFn(1, 1, Gemma4GpuLayer, mlpRun, false, self);
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

    gpu_kv: []?offload.GpuKvCache,
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

        const gpu_kv = try allocator.alloc(?offload.GpuKvCache, layers.len);
        errdefer allocator.free(gpu_kv);
        @memset(gpu_kv, null);

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

    /// 重置 GPU 显存中的全部键值缓存
    pub fn resetCache(self: *Gemma4GpuContext) void {
        for (self.gpu_kv) |*c_opt| {
            if (c_opt.*) |*c| c.deinit();
            c_opt.* = null;
        }
        self.curr_len = 0;
    }

    /// 以常驻 LM Head 对最终隐状态 h_last 打分并按 sampler 选取下一个词元 (与上下文解码路径使用相同的排除与惩罚语义)
    pub fn sampleHidden(
        self: *const Gemma4GpuContext,
        h_last: []const f32,
        recent_tokens: []const usize,
        sampler: *Sampler,
        stream: ?Stream,
    ) !usize {
        return selectTokenFromHidden(self.allocator, &self.lm_head, &self.exclusion_mask, self.config, h_last, recent_tokens, sampler, stream orelse self.stream);
    }

    /// 将词元序列的脑浮点 16 (Brain Floating Point 16, BF16) 嵌入行展开为单精度浮点并乘以 sqrt(hidden_size)，
    /// 写入 [tokens.len, hidden_size] 行主序缓冲区
    fn embedTokens(self: *const Gemma4GpuContext, tokens: []const usize, embed_u16: []const u16, out: []f32) !void {
        const h = self.config.hidden_size;
        const table = embed_u16[0..@min(embed_u16.len, self.config.vocab_size * h)];
        try tensor.embeddingLookupInto(u16, table, tokens, h, self.sqrt_d, out);
    }

    /// 校验 CPU 层缓存与 GPU 解码层的几何规格一致
    fn checkCacheLayout(self: *const Gemma4GpuContext, cache: *const Gemma4KVCache) !void {
        if (cache.layers.len != self.layers.len) return error.ShapeMismatch;
        for (self.layers, cache.layers) |lw, cl| {
            if (cl.core.num_kv_heads != lw.num_kv_heads or cl.core.head_dim != lw.head_dim or cl.attn_type != lw.attn_type) {
                return error.ShapeMismatch;
            }
        }
    }

    /// 将 CPU 预填充 / 解码得到的键值缓存 (各层可见窗口) 同步导入 GPU 显存，GPU 解码位置随之设为 CPU 缓存游标
    pub fn syncCacheFromCpu(self: *Gemma4GpuContext, cache: *const Gemma4KVCache) !void {
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

    /// 将 GPU 显存中的键值缓存 (各层可见窗口) 写回 CPU 层缓存并把 CPU 游标设为 GPU 解码位置，
    /// 使 GPU 预填充 / 解码之后可以无缝切换到 CPU 增量解码
    pub fn syncCacheToCpu(self: *const Gemma4GpuContext, cache: *Gemma4KVCache) !void {
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

        var q_rot = try backend.mlx_blocks.projectHeads(q_raw, lw.q_norm_w, t, lw.num_heads, lw.head_dim, lw.rms_norm_eps, &lw.rope, pos, s);
        errdefer q_rot.deinit();
        var k_rot = try backend.mlx_blocks.projectHeads(k_raw, lw.k_norm_w, t, lw.num_kv_heads, lw.head_dim, lw.rms_norm_eps, &lw.rope, pos, s);
        errdefer k_rot.deinit();
        const v_normed = try backend.mlx_blocks.projectHeads(v_raw, null, t, lw.num_kv_heads, lw.head_dim, lw.rms_norm_eps, null, pos, s);

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
        var x1 = try backend.mlx_blocks.residualScale(x_cur, post_attn, null, s);
        defer x1.deinit();
        return if (use_compiled) lw.runMlp(x1, s) else lw.mlpResidual(x1, s);
    }

    /// 最终层归一化：取 [1, t, hidden_size] 的最后 rows 行，返回 [rows, hidden_size]
    fn finalNorm(self: *const Gemma4GpuContext, x_cur: mlx.Array, t: usize, rows: usize, s: Stream) !mlx.Array {
        _ = t;
        return backend.mlx_blocks.finalHidden(x_cur, self.final_norm_w, self.config.rms_norm_eps, rows, self.config.hidden_size, s);
    }

    /// 执行全 GPU 序列级并行预填充 (Prefill)，一次性完成整个 Prompt 词元序列的矩阵乘法与 KVCache 初始化，按 sampler 返回首个预测词元
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

            // (2) 写入预分配 GPU 键值缓存
            if (self.gpu_kv[l] == null) {
                const default_cap = if (lw.attn_type == .sliding_attention) lw.sliding_window else 256;
                const initial_cap = @max(default_cap, ((T + 255) / 256) * 256);
                self.gpu_kv[l] = try offload.GpuKvCache.init(initial_cap, lw.num_kv_heads, lw.head_dim, .bfloat16, s);
            }
            _ = try self.gpu_kv[l].?.writeStep(qkv.k_rot, qkv.v, lw.attn_type == .sliding_attention, lw.sliding_window, 0, T, s);

            // (3) 带因果与滑动窗口掩码的缩放点积注意力 (scale = 1.0)：窗口覆盖整个序列时掩码化简为融合内核的因果模式
            var attn_out = try offload.maskedAttention(self.allocator, qkv.q_rot, qkv.k_rot, qkv.v, 1.0, lw.attn_type.mask(lw.sliding_window), 0, 0, s);
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
        const next = try selectToken(self.allocator, logits, self.config, &self.exclusion_mask, gemma4.recentWindow(prompt_tokens), sampler, s);
        self.curr_len = T;
        return next;
    }

    /// 执行单步端到端整网自回归增量解码 (Decode Step)：输入当前词元，全程在 GPU 计算图中惰性调度，按 sampler 返回下一个预测词元
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
        var x_cur = try mlx.Array.fromSlice(tok_emb_buf, &.{ 1, 1, self.config.hidden_size });
        defer x_cur.deinit();

        for (self.layers, 0..) |*lw, l| {
            var in_norm = try mlx.Array.fastRmsNorm(x_cur, lw.input_norm_w, lw.rms_norm_eps, s);
            defer in_norm.deinit();
            var qkv = try projectQkv(lw, in_norm, 1, pos, s);
            defer qkv.deinit();

            // 增量写入 GPU 键值缓存
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

            // 单查询注意力：缓存中的全部键均可见
            var attn_out = try mlx.Array.fastScaledDotProductAttention(qkv.q_rot, view.k, view.v, 1.0, .none, s);
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
        const next = try selectToken(self.allocator, logits, self.config, &self.exclusion_mask, gemma4.recentWindow(recent_tokens), sampler, s);
        self.curr_len = pos + 1;
        return next;
    }

    /// 执行投机采样批量验证前向传播 (Speculative Draft Verification Step)
    /// 一次性将基础词元 (input_tokens[0]) 与投机候选词元 (input_tokens[1..M]) 在 GPU 上完成因果注意力与全网前向传播，
    /// 再逐位校验候选词元。第 i 位的目标分布以「此前草稿全部被接受」为前提：重复惩罚窗口为
    /// recentWindow(recent_tokens ++ input_tokens[1 .. i + 1])，排除词元与单步解码一致。
    /// - 贪婪采样：第 i 位的预测词元 (先截断再施加重复惩罚后的 argmax) 与草稿 input_tokens[i + 1] 相同时接受；
    /// - 随机采样：按标准投机采样 (Speculative Sampling) 以目标概率 p(草稿) 接受确定性草稿，拒绝时从去掉草稿后重新归一化的
    ///   目标分布重采样；全部接受时从最后一位的目标分布抽样额外词元。每一位只把 k 个候选传回 CPU，输出分布与逐步采样一致 (无偏)。
    /// accepted_tokens[0 .. num_accepted] 为本步产出的词元 (最后一个是修正词元或额外词元，尚未写入 KVCache)。
    /// 返回值：实际被接受并写入 KVCache 的词元数量 (1 <= num_accepted <= input_tokens.len)
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

            // 增量写入预分配 GPU 键值缓存
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

        // 2. 最终层 RMSNorm 与 LM Head 词表投影 [M, vocab_size]
        var h_norm = try self.finalNorm(x_cur, M, M, s);
        defer h_norm.deinit();
        var logits_raw = try self.lm_head.forwardArrayBf16(h_norm, s);
        defer logits_raw.deinit();

        // 第 i 位的重复惩罚历史：recent_tokens 之后依次接上被假定接受的草稿词元
        var history = try std.ArrayList(usize).initCapacity(self.allocator, recent_tokens.len + M);
        defer history.deinit(self.allocator);
        try history.appendSlice(self.allocator, recent_tokens);
        try history.appendSlice(self.allocator, input_tokens[1..]);

        // 3. 逐位校验候选词元
        const num_accepted = if (sampler.isGreedy())
            try self.verifyGreedy(logits_raw, input_tokens, history.items, recent_tokens.len, sampler.options.repetition_penalty, accepted_tokens, s)
        else
            try self.verifySampled(logits_raw, input_tokens, history.items, recent_tokens.len, sampler, accepted_tokens, s);

        // 4. 回滚未被接受的投机词元 KVCache
        if (M > num_accepted) {
            for (self.gpu_kv) |*maybe_kv| {
                if (maybe_kv.*) |*kv| kv.truncate(pos + num_accepted);
            }
        }

        self.curr_len = pos + num_accepted;
        return num_accepted;
    }

    /// 贪婪校验：读回截断后的完整对数几率 [M, vocab_size]，逐位施加重复惩罚并求 argmax，预测与下一个草稿不一致时停止
    fn verifyGreedy(
        self: *const Gemma4GpuContext,
        logits_raw: mlx.Array,
        input_tokens: []const usize,
        history: []const usize,
        recent_len: usize,
        repetition_penalty: f32,
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
            const window = gemma4.recentWindow(history[0 .. recent_len + i]);
            const pred = sampling.argmaxWithPenalty(row, V, self.config.sampling_excluded_tokens, window, repetition_penalty);
            accepted_tokens[i] = pred;
            if (i + 1 < M and pred != input_tokens[i + 1]) return i + 1;
        }
        return M;
    }

    /// 随机采样校验：在 GPU 上为每一位构建候选集 (各自的重复惩罚窗口) 并一次性求值，随后在 CPU 上按投机采样规则逐位接受或重采样
    fn verifySampled(
        self: *const Gemma4GpuContext,
        logits_raw: mlx.Array,
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
            const window = gemma4.recentWindow(history[0 .. recent_len + i]);
            pending[i] = try deviceCandidates(self.allocator, row, self.config, &self.exclusion_mask, window, sampler.options, s);
            built += 1;
        }

        // 全部位置的候选一次性求值，只发生一次主机-设备同步
        const all_arrays = try self.allocator.alloc(mlx.Array, 3 * M);
        defer self.allocator.free(all_arrays);
        var n_arrays: usize = 0;
        for (pending) |*p| n_arrays += p.arrays(all_arrays[n_arrays..]);
        mlx.Array.evalMultiple(all_arrays[0..n_arrays]);

        const buf = try self.allocator.alloc(sampling.Candidate, pending[0].len());
        defer self.allocator.free(buf);
        for (0..M) |i| {
            const host = try pending[i].read(buf);
            const dist = sampler.distribution(host.candidates, host.log_normalizer);
            if (i + 1 == M) {
                // 全部草稿均被接受：从最后一位的目标分布抽样额外词元
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

    /// 执行 Metal 计算管线预热 (Pipeline Pre-warming)，通过微型前向传递触发 MSL JIT 编译并构建管线状态对象 (PSO)
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
