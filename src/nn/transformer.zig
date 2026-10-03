const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");
const deinitModel = core.deinitModel;
const enterModuleScope = core.enterModuleScope;
const normalization = @import("normalization.zig");
pub const attention = @import("attention.zig");
pub const llm = @import("llm.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Linear = core.Linear;
const RMSNorm = normalization.RMSNorm;
const createPersistentTensor = core.createPersistentTensor;
const freePersistentTensor = core.freePersistentTensor;
const initWeights = core.initWeights;
const InitMethod = core.InitMethod;

// 注意力子模块符号导出
pub const KVCache = attention.KVCache;
pub const ScaledDotProductAttention = attention.ScaledDotProductAttention;
pub const CausalSelfAttention = attention.CausalSelfAttention;
pub const applyRope1D = attention.applyRope1D;
pub const MLACache = attention.MLACache;
pub const MLALayer = attention.MLALayer;

// 大语言模型 (Large Language Model, LLM) 微调、对齐损失与采样子模块符号导出
pub const LoRALinear = llm.LoRALinear;
pub const maskedCrossEntropyLoss = llm.maskedCrossEntropyLoss;
pub const maskedCrossEntropyLossGraph = llm.maskedCrossEntropyLossGraph;
pub const dpoLoss = llm.dpoLoss;
pub const dpoLossGraph = llm.dpoLossGraph;
pub const computeGroupAdvantages = llm.computeGroupAdvantages;
pub const computeGRPOLoss = llm.computeGRPOLoss;
pub const grpoLoss = llm.grpoLoss;
pub const grpoLossGraph = llm.grpoLossGraph;
pub const sampleTopP = llm.sampleTopP;
pub const sampleTopK = llm.sampleTopK;

// ============================================================================
// 1. 嵌入层 (Embedding Layer)
// ============================================================================

/// 嵌入层 (Embedding Layer)
/// 用于将离散的词元标识符 (Token Identifier, Token ID)（例如整数索引）映射为连续的低维稠密向量。
/// 在数学上，这等价于使用独热编码 (One-Hot Encoding) 与权重矩阵相乘，而在实现上通过高效的查找表 (Lookup Table) 实现。
///
/// 权重形状：[vocab_size, embedding_dim]
pub const Embedding = struct {
    weight: *Tensor, // 嵌入层权重矩阵表 (Shape: [vocab_size, embedding_dim])
    name: ?[]const u8 = null,
    module_type: []const u8 = "Embedding",

    /// 嵌入层初始化选项
    pub const Options = struct {
        init_method: InitMethod = .{ .normal = .{ .mean = 0.0, .std = 0.02 } },

        pub const default: Options = .{};
        pub fn defaultOptions() Options {
            return .{};
        }
    };

    /// 构造嵌入层：只分配词表内存 (全零)，不做任何数值初始化
    pub fn init(allocator: std.mem.Allocator, vocab_size: usize, embedding_dim: usize) !Embedding {
        const weight = try createPersistentTensor(allocator, vocab_size, embedding_dim, true);
        return Embedding{
            .weight = weight,
        };
    }

    /// 库内标准参数初始化：在已分配的词表上按 options 重新填充，不分配内存，也不设置 is_custom_initialized 标记
    pub fn resetParameters(self: *Embedding, random: std.Random, options: Options) void {
        const vocab_size = self.weight.shape.dims[0];
        const embedding_dim = self.weight.shape.dims[1];
        initWeights(random, self.weight.data, vocab_size, embedding_dim, options.init_method);
    }

    /// 模块标准数学变换公式
    pub const formula = "y = \\text{Embedding}(\\text{indices}; W_e \\in \\mathbb{R}^{V \\times D})";

    /// 查找映射前向传播
    /// 输入 x 为包含词元标识符 (Token Identifier, Token ID) 的任意维度张量 (Tensor)、`GenericTensor(IntT)` 或整数切片，输出形状为 x.shape + [embedding_dim]
    pub fn forward(self: *const Embedding, graph: *autodiff.Graph, x: anytype) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        return try graph.embedding(self.weight, x);
    }
};

// ============================================================================
// 2. 前馈网络模块：多层感知机 (Multi-Layer Perceptron, MLP) 与 Swish 门控线性单元 (Swish-Gated Linear Unit, SwiGLU)
// ============================================================================

/// 多层感知机 (Multi-Layer Perceptron, MLP) / 前馈神经网络 (Feed-Forward Network, FFN) 模块
/// 变换器 (Transformer) 架构中的重要组件，紧跟在自注意力 (Self-Attention) 之后，
/// 用于在每个词元 (Token) 位置上独立地进行非线性特征投影与融合。
///
/// 数学公式：
/// \text{MLP}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2
/// 结构：
/// Linear(dim -> hidden_dim) -> 高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 激活函数 -> Linear(hidden_dim -> dim)
/// 其中 hidden_dim 通常设置为 4 * dim。
pub const MLP = struct {
    c_fc: Linear, // 升维全连接投影层 (Fully Connected Layer, c_fc: dim -> hidden_dim)
    c_proj: Linear, // 降维投影层 (Output Projection, c_proj: hidden_dim -> dim)
    name: ?[]const u8 = null,
    module_type: []const u8 = "MLP",

    /// 初始化多层感知机 (Multi-Layer Perceptron, MLP) 模块
    /// dim: 输入与输出隐藏维度
    /// hidden_dim: 中间隐藏维度 (一般为 4 * dim)
    pub fn init(allocator: std.mem.Allocator, dim: usize, hidden_dim: usize) !MLP {
        const c_fc = try Linear.init(allocator, dim, hidden_dim);
        errdefer deinitModel(&c_fc, allocator);
        const c_proj = try Linear.init(allocator, hidden_dim, dim);
        errdefer deinitModel(&c_proj, allocator);

        return MLP{
            .c_fc = c_fc,
            .c_proj = c_proj,
        };
    }

    /// 模块标准数学变换公式
    pub const formula = "y = \\text{GELU}(x W_{fc}^T + b_{fc}) W_{proj}^T + b_{proj}";

    /// 前向传播逻辑
    /// 支持输入二维张量 (2-Dimensional Tensor, 2D) [B*T, D] 或三维张量 (3-Dimensional Tensor, 3D) [B, T, D]
    pub fn forward(self: *const MLP, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        var x_2d = x;

        // 1. 如果输入是三维 (3-Dimensional, 3D) [B, T, D]，则将其打平为二维 (2-Dimensional, 2D) [B*T, D] 以满足线性层 (Linear) 矩阵乘法的输入规范
        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            x_2d = try graph.reshape(x, &.{ B * T, D });
        }

        // 2. 升维映射: [B*T, D] -> [B*T, hidden_dim]
        const h1 = try self.c_fc.forward(graph, x_2d);

        // 3. 高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 激活函数引入非线性
        const a1 = try graph.gelu(h1);
        if (self.name) |mod_name| a1.setNameFormatted("{s}.gelu", .{mod_name});

        // 4. 降维投射回原始特征维度: [B*T, hidden_dim] -> [B*T, D]
        const h2 = try self.c_proj.forward(graph, a1);

        // 5. 如果输入原本是三维 (3-Dimensional, 3D)，需要将输出再重新恢复成三维形状: [B, T, D]
        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            return try graph.reshape(h2, &.{ B, T, D });
        }
        return h2;
    }
};

/// 现代变换器 (Transformer) 门控前馈网络：Swish 门控线性单元 (Swish-Gated Linear Unit, SwiGLU / LLaMA-style Multi-Layer Perceptron, MLP)
/// 结构：(Sigmoid 线性单元 (Sigmoid Linear Unit, SiLU)(x * W_gate) * (x * W_up)) * W_down
/// 其中 hidden_dim 通常设置为 8/3 * dim
pub const SwiGLU = struct {
    w_gate: Linear, // 门控投影层 (dim -> hidden_dim)
    w_up: Linear, // 升维投影层 (dim -> hidden_dim)
    w_down: Linear, // 降维投影层 (hidden_dim -> dim)
    name: ?[]const u8 = null,
    module_type: []const u8 = "SwiGLU",

    pub fn init(allocator: std.mem.Allocator, dim: usize, hidden_dim: usize) !SwiGLU {
        const w_gate = try Linear.init(allocator, dim, hidden_dim);
        errdefer deinitModel(&w_gate, allocator);
        const w_up = try Linear.init(allocator, dim, hidden_dim);
        errdefer deinitModel(&w_up, allocator);
        const w_down = try Linear.init(allocator, hidden_dim, dim);
        errdefer deinitModel(&w_down, allocator);

        return SwiGLU{
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
        };
    }

    /// 模块标准数学变换公式
    pub const formula = "y = (\\text{SiLU}(x W_{\\text{gate}}) \\odot (x W_{\\text{up}})) W_{\\text{down}}";

    /// 前向传播逻辑：支持二维 (2-Dimensional, 2D) [B*T, D] 或三维 (3-Dimensional, 3D) [B, T, D]
    pub fn forward(self: *const SwiGLU, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
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

        // 1. 计算门控 (Gate) 投影: [B*T, D] -> [B*T, hidden_dim]
        const gate = try self.w_gate.forward(graph, x_2d);

        // 2. 计算升维 (Up) 投影: [B*T, D] -> [B*T, hidden_dim]
        const up = try self.w_up.forward(graph, x_2d);

        // 3. 计算 Sigmoid 线性单元 (Sigmoid Linear Unit, SiLU) 激活: SiLU(gate)
        const silu_gate = try graph.silu(gate);

        // 4. 逐元素乘法: SiLU(gate) * up
        const hidden = try graph.mul(silu_gate, up);

        // 5. 降维投射回原始特征维度: [B*T, hidden_dim] -> [B*T, D]
        const out = try self.w_down.forward(graph, hidden);

        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            return try graph.reshape(out, &.{ B, T, D });
        }
        return out;
    }
};

// ============================================================================
// 3. 混合专家前馈网络层 (Mixture of Experts Layer, MoELayer)
// ============================================================================

/// 混合专家前馈网络层 (Mixture of Experts Layer, MoELayer)
/// 对应现代前沿大模型与 DeepSeekMoE 细粒度专家路由架构：
/// 包含：
/// 1. 细粒度路由专家列表 (routed_experts, 激活 top_k)
/// 2. 可选隔离常驻共享专家列表 (shared_experts, 均无条件激活)
/// 3. 动态门控路由网络 gate: Linear(dim -> num_routed_experts)
/// 4. 门控概率归一化与稀疏加权聚合输出
/// 5. 基于计算图的前向与反向传播 (推理时传入 `Graph.initNoGrad` 构建的无梯度计算图)
pub const MoELayer = struct {
    dim: usize,
    num_routed_experts: usize,
    num_shared_experts: usize,
    top_k: usize,
    gate: Linear,
    routed_experts: []MLP,
    shared_experts: []MLP,
    name: ?[]const u8 = null,
    module_type: []const u8 = "MoELayer",

    pub const formula = "y = \\sum_{i \\in \\text{TopK}(g(x))} p_i(x) E_i(x) + \\sum_{j} E^{\\text{shared}}_j(x)";

    pub fn init(
        allocator: std.mem.Allocator,
        dim: usize,
        hidden_dim: usize,
        num_routed_experts: usize,
        num_shared_experts: usize,
        top_k: usize,
    ) !MoELayer {
        std.debug.assert(top_k > 0 and top_k <= num_routed_experts);

        const gate = try Linear.init(allocator, dim, num_routed_experts);
        errdefer deinitModel(&gate, allocator);

        const routed = try allocator.alloc(MLP, num_routed_experts);
        errdefer allocator.free(routed);

        var init_r: usize = 0;
        errdefer {
            for (0..init_r) |i| deinitModel(&routed[i], allocator);
        }
        for (0..num_routed_experts) |i| {
            routed[i] = try MLP.init(allocator, dim, hidden_dim);
            init_r += 1;
        }

        const shared = try allocator.alloc(MLP, num_shared_experts);
        errdefer allocator.free(shared);

        var init_s: usize = 0;
        errdefer {
            for (0..init_s) |i| deinitModel(&shared[i], allocator);
        }
        for (0..num_shared_experts) |i| {
            shared[i] = try MLP.init(allocator, dim, hidden_dim);
            init_s += 1;
        }

        return MoELayer{
            .dim = dim,
            .num_routed_experts = num_routed_experts,
            .num_shared_experts = num_shared_experts,
            .top_k = top_k,
            .gate = gate,
            .routed_experts = routed,
            .shared_experts = shared,
        };
    }

    pub fn forward(self: *const MoELayer, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
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

        const N = x_2d.shape.dims[0];
        const E = self.num_routed_experts;
        const K = self.top_k;

        // 1. 门控打分: [N, D] -> [N, E]
        const gate_logits = try self.gate.forward(graph, x_2d);

        // 2. 构造前 K 项 (Top-K) 掩码并执行归一化指数函数 (Softmax) 归一化
        const mask_node = try graph.tensorND(&.{ N, E }, false);
        mask_node.is_buffer = true;
        const mask_data = mask_node.data;
        @memset(mask_data, -1e9);

        const keep_node = try graph.tensorND(&.{ N, E }, false);
        keep_node.is_buffer = true;
        const keep_mask = keep_node.data;
        @memset(keep_mask, 0.0);

        const expert_active = try graph.arenaAllocator().alloc(bool, E);
        @memset(expert_active, false);

        // 对每一行寻找前 K 项 (Top-K) 个最大的索引
        for (0..N) |row| {
            const row_logits = gate_logits.data[row * E .. (row + 1) * E];

            var top_indices: [64]usize = undefined;
            var top_vals: [64]f32 = undefined;
            std.debug.assert(K <= 64);

            for (0..K) |k| {
                top_indices[k] = k;
                top_vals[k] = row_logits[k];
            }
            // 对初始 K 个排序
            for (0..K) |i| {
                for (i + 1..K) |j| {
                    if (top_vals[j] > top_vals[i]) {
                        const tmp_v = top_vals[i];
                        top_vals[i] = top_vals[j];
                        top_vals[j] = tmp_v;
                        const tmp_idx = top_indices[i];
                        top_indices[i] = top_indices[j];
                        top_indices[j] = tmp_idx;
                    }
                }
            }

            for (K..E) |e| {
                const val = row_logits[e];
                if (val > top_vals[K - 1]) {
                    top_vals[K - 1] = val;
                    top_indices[K - 1] = e;
                    var pos = K - 1;
                    while (pos > 0 and top_vals[pos] > top_vals[pos - 1]) : (pos -= 1) {
                        const tmp_v = top_vals[pos - 1];
                        top_vals[pos - 1] = top_vals[pos];
                        top_vals[pos] = tmp_v;
                        const tmp_idx = top_indices[pos - 1];
                        top_indices[pos - 1] = top_indices[pos];
                        top_indices[pos] = tmp_idx;
                    }
                }
            }

            for (0..K) |k| {
                const e_idx = top_indices[k];
                mask_data[row * E + e_idx] = 0.0;
                keep_mask[row * E + e_idx] = 1.0;
                expert_active[e_idx] = true;
            }
        }

        var total_routed: *Tensor = undefined;

        const masked_logits = try graph.add(gate_logits, mask_node);
        const raw_probs = try graph.softmax(masked_logits); // [N, E]
        const probs = try graph.mul(raw_probs, keep_node); // 严格置零非前 K 项 (Top-K) 概率
        const prob_cols = try graph.split(probs, E, 1); // E 个 [N, 1]

        var acc: ?*Tensor = null;
        for (self.routed_experts, 0..) |exp, e| {
            const exp_out = try exp.forward(graph, x_2d); // [N, D]
            const weighted = try graph.mul(exp_out, prob_cols[e]); // [N, D] * [N, 1] -> [N, D]
            if (acc) |a| {
                acc = try graph.add(a, weighted);
            } else {
                acc = weighted;
            }
        }
        total_routed = acc.?;

        // 3. 计算常驻共享专家 (Shared Experts)
        var total_shared: ?*Tensor = null;
        for (self.shared_experts) |exp| {
            const s_out = try exp.forward(graph, x_2d);

            if (total_shared) |s| {
                total_shared = try graph.add(s, s_out);
            } else {
                total_shared = s_out;
            }
        }

        // 4. 合并路由专家 (Routed Experts) 与共享专家 (Shared Experts)
        var final_2d = total_routed;
        if (total_shared) |s| {
            final_2d = try graph.add(total_routed, s);
        }

        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            return try graph.reshape(final_2d, &.{ B, T, D });
        }

        return final_2d;
    }
};

// ============================================================================
// 4. 变换器块 (Transformer Block) 与变换器解码器 (Transformer Decoder)
// ============================================================================

/// 变换器编码器/解码器块模块 (Transformer Block)
/// 采用前置层归一化 (Pre-Layer Normalization, Pre-LN) 架构进行组装：
/// 1. x_norm1 = 均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)(x)
/// 2. x_attn = 自注意力 (Self-Attention)(x_norm1)
/// 3. x1 = x + x_attn  (第一层残差连接)
/// 4. x_norm2 = 均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)(x1)
/// 5. x_mlp = 多层感知机 (Multi-Layer Perceptron, MLP)(x_norm2)
/// 6. out = x1 + x_mlp (第二层残差连接)
pub const TransformerBlock = struct {
    ln_1: RMSNorm, // 第一层均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)，在注意力 (Attention) 计算前执行
    attn: CausalSelfAttention, // 因果自注意力机制 (Causal Self-Attention) 层
    ln_2: RMSNorm, // 第二层均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)，在多层感知机 (Multi-Layer Perceptron, MLP) 计算前执行
    mlp: MLP, // 前馈多层感知机 (Multi-Layer Perceptron, MLP) 层
    name: ?[]const u8 = null,
    module_type: []const u8 = "TransformerBlock",

    /// 初始化变换器块 (Transformer Block)
    /// n_embd: 隐藏特征嵌入维度
    /// n_head: 注意力头数
    pub fn init(allocator: std.mem.Allocator, n_embd: usize, n_head: usize) !TransformerBlock {
        const ln_1 = try RMSNorm.init(allocator, n_embd, 1e-5);
        errdefer deinitModel(&ln_1, allocator);
        const attn = try CausalSelfAttention.init(allocator, n_embd, n_head);
        errdefer deinitModel(&attn, allocator);
        const ln_2 = try RMSNorm.init(allocator, n_embd, 1e-5);
        errdefer deinitModel(&ln_2, allocator);
        const mlp = try MLP.init(allocator, n_embd, 4 * n_embd);
        errdefer deinitModel(&mlp, allocator);

        return TransformerBlock{
            .ln_1 = ln_1,
            .attn = attn,
            .ln_2 = ln_2,
            .mlp = mlp,
        };
    }

    /// 模块标准数学变换公式
    pub const formula = "\\begin{aligned} h_l &= x_l + \\text{Attention}(\\text{RMSNorm}(x_l)) \\\\ x_{l+1} &= \\text{TransformerBlock}(x_l) = h_l + \\text{MLP}(\\text{RMSNorm}(h_l)) \\end{aligned}";

    /// 前向传播流程：x -> Block(x) -> out
    pub fn forward(self: *const TransformerBlock, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        // 1. 第一条支路: 均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm) -> 注意力 (Attention)
        const x_norm1 = try self.ln_1.forward(graph, x);

        const x_attn = try self.attn.forward(graph, x_norm1);

        // 2. 第一条残差混合: x1 = x_l + Attention(RMSNorm(x_l))
        const x1 = try graph.add(x, x_attn);
        if (self.name) |mod_name| {
            x1.setNameFormatted("{s}.residual_attn", .{mod_name});
            try graph.setModuleFormula(x1.name.?, "x_1 = x_l + \\text{Attention}(\\text{RMSNorm}(x_l))");
        }

        // 3. 第二条支路: 均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm) -> 多层感知机 (Multi-Layer Perceptron, MLP)
        const x_norm2 = try self.ln_2.forward(graph, x1);

        const x_mlp = try self.mlp.forward(graph, x_norm2);

        // 4. 第二条残差混合: out = x1 + MLP(RMSNorm(x1))
        const out = try graph.add(x1, x_mlp);
        if (self.name) |mod_name| {
            out.setNameFormatted("{s}.residual_mlp", .{mod_name});
            try graph.setModuleFormula(out.name.?, "x_{l+1} = x_1 + \\text{MLP}(\\text{RMSNorm}(x_1))");
        }
        return out;
    }
};

/// 堆叠多层变换器块 (Transformer Block) 的解码器主干网络 (Transformer Decoder)
pub fn TransformerDecoder(comptime n_layer: usize) type {
    return struct {
        const Self = @This();

        h: [n_layer]TransformerBlock, // 堆叠的变换器块 (Transformer Block) 数组
        ln_f: RMSNorm, // 骨架最末端用于规范化的均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm) 层

        name: ?[]const u8 = null,
        module_type: []const u8 = "TransformerDecoder",

        /// 初始化整个解码器组件
        pub fn init(allocator: std.mem.Allocator, n_embd: usize, n_head: usize) !Self {
            var h: [n_layer]TransformerBlock = undefined;
            var i: usize = 0;
            errdefer {
                for (0..i) |j| {
                    deinitModel(&h[j], allocator);
                }
            }
            // 循环初始化每一层变换器块 (TransformerBlock)
            while (i < n_layer) : (i += 1) {
                h[i] = try TransformerBlock.init(allocator, n_embd, n_head);
            }

            // 初始化最后的均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm) 层
            const ln_f = try RMSNorm.init(allocator, n_embd, 1e-5);

            return Self{
                .h = h,
                .ln_f = ln_f,
            };
        }

        /// 模块标准数学变换公式
        pub const formula = "x_L = \\text{DecoderStack}(x_0) = (\\text{Block}_L \\circ \\dots \\circ \\text{Block}_1)(x_0)";

        /// 解码器主干网络的前向传播流程
        pub fn forward(self: *const Self, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();
            var current_x = x;
            // 依次贯穿每一层变换器块 (Transformer Block)
            for (self.h) |layer| {
                current_x = try layer.forward(graph, current_x);
            }

            // 执行最后一层均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm) 映射输出
            return try self.ln_f.forward(graph, current_x);
        }
    };
}

// ============================================================================
// 5. 生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 模型定义
// ============================================================================

/// 生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 模型配置结构体
pub const GPTConfig = struct {
    vocab_size: usize = 50257, // 词表大小 (Vocabulary Size, vocab_size)，决定输入和输出层的映射维度
    block_size: usize = 1024, // 最大上下文长度/时间步长度 (Context Length / Block Size)
    n_embd: usize = 768, // 隐藏特征嵌入维度 (Embedding Dimension, n_embd)
    n_head: usize = 12, // 多头注意力头数 (Multi-Head Attention Heads, n_head)
    n_layer: usize = 12, // 变换器块 (Transformer Block) 堆叠的层数 (Number of Decoder Layers, n_layer)

    pub const default: GPTConfig = .{};
    pub fn defaultConfig() GPTConfig {
        return .{};
    }
};

/// 泛型生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 模型定义函数
pub fn GPT(comptime config: GPTConfig) type {
    return struct {
        token_embedding: Embedding, // 词元嵌入层 (Token Embedding, wte)
        position_embedding: Embedding, // 位置嵌入层 (Position Embedding, wpe)
        decoder: TransformerDecoder(config.n_layer), // 堆叠的变换器解码器 (Transformer Decoder) 层与最终归一化层
        lm_head: Linear, // 最终输出概率的语言模型线性分类投影头 (Language Model Head, lm_head)
        name: ?[]const u8 = null,
        module_type: []const u8 = "GPT",

        const Self = @This();

        /// 初始化默认配置的生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 模型实例
        pub fn initDefault(allocator: std.mem.Allocator) !Self {
            return init(allocator);
        }

        /// 初始化生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 模型中的所有网络层权重
        pub fn init(allocator: std.mem.Allocator) !Self {
            // 初始化词元 (Token) 嵌入矩阵 [vocab_size, n_embd]
            const token_embedding = try Embedding.init(allocator, config.vocab_size, config.n_embd);
            errdefer deinitModel(&token_embedding, allocator);

            // 初始化位置 (Position) 嵌入矩阵 [block_size, n_embd]
            const position_embedding = try Embedding.init(allocator, config.block_size, config.n_embd);
            errdefer deinitModel(&position_embedding, allocator);

            // 初始化变换器解码器 (Transformer Decoder) 主干网络
            const decoder = try TransformerDecoder(config.n_layer).init(allocator, config.n_embd, config.n_head);
            errdefer deinitModel(&decoder, allocator);

            // 初始化输出映射语言模型分类头 (Language Model Head, lm_head) [n_embd, vocab_size]
            const lm_head = try Linear.init(allocator, config.n_embd, config.vocab_size);

            return Self{
                .token_embedding = token_embedding,
                .position_embedding = position_embedding,
                .decoder = decoder,
                .lm_head = lm_head,
            };
        }

        /// 模块标准数学变换公式
        pub const formula = "\\text{logits} = \\text{GPT}(\\text{TokenIDs}; \\theta) \\rightarrow [B, T, V]";

        /// 前向推理传播流程
        /// 输入 x 为包含词元标识符 (Token Identifier, Token ID) 的二维张量 (2-Dimensional Tensor, 2D，支持 `*Tensor` 或 `*GenericTensor(IntT)`)，形状为 [B, T]
        /// 输出为未归一化的预测对数几率 (Logits)，形状为三维 (3-Dimensional, 3D): [B, T, vocab_size]
        pub fn forward(self: *const Self, graph: *autodiff.Graph, x: anytype) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();
            const B = x.shape.dims[0];
            const T = x.shape.dims[1];

            // 1. 获取词元 (Token) 嵌入向量: [B, T] -> [B, T, n_embd]
            const tok_emb = try self.token_embedding.forward(graph, x);

            // 2. 生成对应的时间/位置索引 [0, 1, 2, ... T-1]，并将其转换为二维 (2-Dimensional, 2D) 位置张量 (Tensor) [B, T]
            const pos_node = try graph.tensorND(&.{ B, T }, false);
            for (0..B) |b| {
                for (0..T) |t| {
                    pos_node.data[b * T + t] = @as(f32, @floatFromInt(t));
                }
            }
            // 位置索引由输入形状在模型内部生成，属于生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 自身的常量缓冲区，而非模型输入
            pos_node.is_buffer = true;
            if (self.name) |mod_name| {
                pos_node.setNameFormatted("{s}.pos_indices", .{mod_name});
            }

            // 3. 获取对应的可学习 (Learned) 位置嵌入向量: [B, T] -> [B, T, n_embd]
            const pos_emb = try self.position_embedding.forward(graph, pos_node);

            // 4. 将词元 (Token) 嵌入和位置嵌入进行求和融合，作为初始隐藏输入: h = tok_emb + pos_emb
            const h_x = try graph.add(tok_emb, pos_emb);
            if (self.name) |mod_name| {
                h_x.setNameFormatted("{s}.embeddings_sum", .{mod_name});
            }

            // 5. 将混合后的输入送进层叠的变换器解码器 (Transformer Decoder) 主干网络中依次计算
            // 输出形状保持为: [B, T, n_embd]
            const decoder_out = try self.decoder.forward(graph, h_x);

            // 6. 将输出展平为二维 (2-Dimensional, 2D)，以便进行最终分类头的全连接投影计算: [B, T, n_embd] -> [B*T, n_embd]
            const ln_x_2d = try graph.reshape(decoder_out, &.{ B * T, config.n_embd });

            // 7. 进行投影以获得词表空间未归一化的分类对数几率 (Logits): [B*T, n_embd] -> [B*T, vocab_size]
            const logits_2d = try self.lm_head.forward(graph, ln_x_2d);

            // 8. 将形状重塑还原成三维 (3-Dimensional, 3D) 形式返回: [B, T, vocab_size]
            return try graph.reshape(logits_2d, &.{ B, T, config.vocab_size });
        }
    };
}

/// 采用默认生成式预训练变换器配置 (GPTConfig) 的标准生成式预训练变换器 (Generative Pre-trained Transformer, GPT) 模型类型别名
pub const DefaultGPT = GPT(GPTConfig.default);
