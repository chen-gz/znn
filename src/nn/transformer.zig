const std = @import("std");
const backend = @import("../backend.zig");
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
const RowView = core.RowView;
const RMSNorm = normalization.RMSNorm;
const createPersistentTensor = core.createPersistentTensor;
const freePersistentTensor = core.freePersistentTensor;
const initWeights = core.initWeights;
const InitMethod = core.InitMethod;
const InitOptions = core.InitOptions;

// 注意力子模块符号导出
pub const LayerKVCache = attention.LayerKVCache;
pub const KVCache = attention.KVCache;
pub const ScaledDotProductAttention = attention.ScaledDotProductAttention;
pub const splitHeads = attention.splitHeads;
pub const mergeHeads = attention.mergeHeads;
pub const GQAAttention = attention.GQAAttention;
pub const CausalSelfAttention = attention.CausalSelfAttention;
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
// 2. 前馈网络模块：多层感知机 (MLP)、门控激活函数与泛型门控前馈网络 (GatedMLP / SwiGLU)
// ============================================================================

/// 门控前馈激活函数类型
pub const GatedActivation = enum {
    silu,
    gelu,
    gelu_tanh,

    pub fn apply(self: GatedActivation, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        return switch (self) {
            .silu => graph.silu(x),
            .gelu => graph.gelu(x, .none),
            .gelu_tanh => graph.gelu(x, .tanh),
        };
    }

    pub fn fromGelu(approx: backend.GeluApproximation) GatedActivation {
        return switch (approx) {
            .none => .gelu,
            .tanh => .gelu_tanh,
        };
    }

    pub fn geluApproximation(self: GatedActivation) ?backend.GeluApproximation {
        return switch (self) {
            .gelu => .none,
            .gelu_tanh => .tanh,
            .silu => null,
        };
    }
};

/// 多层感知机 (Multi-Layer Perceptron, MLP) / 前馈神经网络 (Feed-Forward Network, FFN) 模块
/// 变换器 (Transformer) 架构中的重要组件，紧跟在自注意力 (Self-Attention) 之后，
/// 用于在每个词元 (Token) 位置上独立地进行非线性特征投影与融合。
///
/// 数学公式：
/// \text{MLP}(x) = \text{Act}(x W_1 + b_1) W_2 + b_2
/// 结构：
/// Linear(dim -> hidden_dim) -> 激活函数 (默认 GELU) -> Linear(hidden_dim -> dim)
/// 其中 hidden_dim 通常设置为 4 * dim。
pub const MLP = struct {
    c_fc: Linear, // 升维全连接投影层 (Fully Connected Layer, c_fc: dim -> hidden_dim)
    c_proj: Linear, // 降维投影层 (Output Projection, c_proj: hidden_dim -> dim)
    activation: GatedActivation = .gelu,
    name: ?[]const u8 = null,
    module_type: []const u8 = "MLP",

    pub const Options = struct {
        bias: bool = true,
        activation: GatedActivation = .gelu,

        pub fn defaultOptions() Options {
            return .{};
        }
    };

    /// 初始化多层感知机 (Multi-Layer Perceptron, MLP) 模块
    /// dim: 输入与输出隐藏维度
    /// hidden_dim: 中间隐藏维度 (一般为 4 * dim)
    pub fn init(allocator: std.mem.Allocator, dim: usize, hidden_dim: usize) !MLP {
        return initWithOptions(allocator, dim, hidden_dim, .{});
    }

    pub fn initWithOptions(allocator: std.mem.Allocator, dim: usize, hidden_dim: usize, options: Options) !MLP {
        const c_fc = try Linear.initWithOptions(allocator, dim, hidden_dim, .{ .bias = options.bias });
        errdefer deinitModel(&c_fc, allocator);
        const c_proj = try Linear.initWithOptions(allocator, hidden_dim, dim, .{ .bias = options.bias });
        errdefer deinitModel(&c_proj, allocator);

        return MLP{
            .c_fc = c_fc,
            .c_proj = c_proj,
            .activation = options.activation,
        };
    }

    /// 库内标准参数初始化：在已分配的张量上按 options 重新填充权重与偏置，不分配内存，也不设置 is_custom_initialized 标记
    pub fn resetParameters(self: *MLP, random: std.Random, options: InitOptions) void {
        var fc_opts = options;
        fc_opts.nonlinearity = switch (self.activation) {
            .silu => .silu,
            .gelu, .gelu_tanh => .gelu,
        };
        self.c_fc.resetParameters(random, fc_opts);
        self.c_proj.resetParameters(random, options);
    }

    /// 模块标准数学变换公式
    pub const formula = "y = \\text{Act}(x W_{fc} + b_{fc}) W_{proj} + b_{proj}";

    /// 前向传播逻辑
    /// 支持输入任意形状张量 [..., dim]，输出与输入形状对齐
    pub fn forward(self: *const MLP, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();

        const h1 = try self.c_fc.forward(graph, x);
        const a1 = try self.activation.apply(graph, h1);
        if (self.name) |mod_name| {
            switch (self.activation) {
                .gelu => a1.setNameFormatted("{s}.gelu", .{mod_name}),
                .gelu_tanh => a1.setNameFormatted("{s}.gelu_tanh", .{mod_name}),
                .silu => a1.setNameFormatted("{s}.silu", .{mod_name}),
            }
        }
        return try self.c_proj.forward(graph, a1);
    }
};

/// 泛型门控前馈神经网络模块 (GatedMLP: gate_proj, up_proj, down_proj)
/// 统一适用于全精度 Linear 与量化 Q4Linear，支持 SiLU (SwiGLU / LLaMA) 与 GELU / GELU(tanh) (Gemma 4)
///
/// 数学公式：
/// y = (\text{Act}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}
pub fn GatedMLP(comptime Proj: type) type {
    return struct {
        const Self = @This();

        gate_proj: Proj,
        up_proj: Proj,
        down_proj: Proj,
        activation: GatedActivation = .silu,
        name: ?[]const u8 = null,
        module_type: []const u8 = "GatedMLP",

        pub const Options = struct {
            bias: bool = false,
            activation: GatedActivation = .silu,

            pub fn defaultOptions() Options {
                return .{};
            }
        };

        pub fn init(allocator: std.mem.Allocator, in_features: usize, intermediate_size: usize) !Self {
            return initWithOptions(allocator, in_features, intermediate_size, .{});
        }

        pub fn initWithOptions(allocator: std.mem.Allocator, in_features: usize, intermediate_size: usize, options: Options) !Self {
            const gate_proj = if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, in_features, intermediate_size, .{ .bias = options.bias })
            else
                try Proj.init(allocator, in_features, intermediate_size);
            errdefer deinitModel(&gate_proj, allocator);

            const up_proj = if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, in_features, intermediate_size, .{ .bias = options.bias })
            else
                try Proj.init(allocator, in_features, intermediate_size);
            errdefer deinitModel(&up_proj, allocator);

            const down_proj = if (@hasDecl(Proj, "initWithOptions"))
                try Proj.initWithOptions(allocator, intermediate_size, in_features, .{ .bias = options.bias })
            else
                try Proj.init(allocator, intermediate_size, in_features);
            errdefer deinitModel(&down_proj, allocator);

            return Self{
                .gate_proj = gate_proj,
                .up_proj = up_proj,
                .down_proj = down_proj,
                .activation = options.activation,
            };
        }

        pub fn initSilu(allocator: std.mem.Allocator, in_features: usize, intermediate_size: usize) !Self {
            return initWithOptions(allocator, in_features, intermediate_size, .{ .bias = true, .activation = .silu });
        }

        pub fn initGemma(allocator: std.mem.Allocator, in_features: usize, intermediate_size: usize, gelu_approx: backend.GeluApproximation) !Self {
            return initWithOptions(allocator, in_features, intermediate_size, .{
                .bias = false,
                .activation = GatedActivation.fromGelu(gelu_approx),
            });
        }

        pub fn initEmpty(hidden_size: usize, intermediate_size: usize, activation: GatedActivation) Self {
            return Self{
                .gate_proj = Proj.fromBlocks(hidden_size, intermediate_size, &.{}),
                .up_proj = Proj.fromBlocks(hidden_size, intermediate_size, &.{}),
                .down_proj = Proj.fromBlocks(intermediate_size, hidden_size, &.{}),
                .activation = activation,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (@hasDecl(Proj, "deinit")) {
                self.gate_proj.deinit(allocator);
                self.up_proj.deinit(allocator);
                self.down_proj.deinit(allocator);
            } else {
                deinitModel(&self.gate_proj, allocator);
                deinitModel(&self.up_proj, allocator);
                deinitModel(&self.down_proj, allocator);
            }
        }

        pub fn resetParameters(self: *Self, random: std.Random, options: InitOptions) void {
            if (@hasDecl(Proj, "resetParameters")) {
                var gate_opts = options;
                gate_opts.nonlinearity = switch (self.activation) {
                    .silu => .silu,
                    .gelu, .gelu_tanh => .relu,
                };
                self.gate_proj.resetParameters(random, gate_opts);
                self.up_proj.resetParameters(random, options);
                self.down_proj.resetParameters(random, options);
            }
        }

        pub const formula = "y = (\\text{Act}(x W_{\\text{gate}}) \\odot (x W_{\\text{up}})) W_{\\text{down}}";

        pub fn forward(self: *const Self, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
            const module_scope = try enterModuleScope(graph, self);
            defer module_scope.exit();

            const gate = try self.gate_proj.forward(graph, x);
            const up = try self.up_proj.forward(graph, x);
            const act_gate = try self.activation.apply(graph, gate);
            if (self.name) |mod_name| act_gate.setNameFormatted("{s}.act", .{mod_name});
            const hidden = try graph.mul(act_gate, up);
            return try self.down_proj.forward(graph, hidden);
        }
    };
}

/// 现代变换器 (Transformer) 门控前馈网络：Swish 门控线性单元 (Swish-Gated Linear Unit, SwiGLU / LLaMA-style Multi-Layer Perceptron, MLP)
pub const SwiGLU = GatedMLP(Linear);

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
/// 5. 基于计算图的前向与反向传播 (推理时传入 `Graph.init(alloc, .{ .enable_grad = false })` 构建的无梯度计算图)
pub const MoELayer = struct {
    dim: usize,
    num_routed_experts: usize,
    num_shared_experts: usize,
    top_k: usize,
    norm_topk_prob: bool = false,
    activation: GatedActivation = .gelu,
    gate: Linear,
    routed_experts: []MLP,
    shared_experts: []MLP,
    name: ?[]const u8 = null,
    module_type: []const u8 = "MoELayer",

    pub const Options = struct {
        norm_topk_prob: bool = false,
        activation: GatedActivation = .gelu,

        pub fn defaultOptions() Options {
            return .{};
        }
    };

    pub const formula = "y = \\sum_{i \\in \\text{TopK}(g(x))} p_i(x) E_i(x) + \\sum_{j} E^{\\text{shared}}_j(x)";

    pub fn init(
        allocator: std.mem.Allocator,
        dim: usize,
        hidden_dim: usize,
        num_routed_experts: usize,
        num_shared_experts: usize,
        top_k: usize,
    ) !MoELayer {
        return initWithOptions(allocator, dim, hidden_dim, num_routed_experts, num_shared_experts, top_k, .{});
    }

    pub fn initWithOptions(
        allocator: std.mem.Allocator,
        dim: usize,
        hidden_dim: usize,
        num_routed_experts: usize,
        num_shared_experts: usize,
        top_k: usize,
        options: Options,
    ) !MoELayer {
        if (dim == 0 or hidden_dim == 0 or num_routed_experts == 0 or top_k == 0 or top_k > num_routed_experts) {
            return error.InvalidConfiguration;
        }

        const gate = try Linear.init(allocator, dim, num_routed_experts);
        errdefer deinitModel(&gate, allocator);

        const routed = try allocator.alloc(MLP, num_routed_experts);
        errdefer allocator.free(routed);

        var init_r: usize = 0;
        errdefer {
            for (0..init_r) |i| deinitModel(&routed[i], allocator);
        }
        for (0..num_routed_experts) |i| {
            routed[i] = try MLP.initWithOptions(allocator, dim, hidden_dim, .{ .activation = options.activation });
            init_r += 1;
        }

        const shared = try allocator.alloc(MLP, num_shared_experts);
        errdefer allocator.free(shared);

        var init_s: usize = 0;
        errdefer {
            for (0..init_s) |i| deinitModel(&shared[i], allocator);
        }
        for (0..num_shared_experts) |i| {
            shared[i] = try MLP.initWithOptions(allocator, dim, hidden_dim, .{ .activation = options.activation });
            init_s += 1;
        }

        return MoELayer{
            .dim = dim,
            .num_routed_experts = num_routed_experts,
            .num_shared_experts = num_shared_experts,
            .top_k = top_k,
            .norm_topk_prob = options.norm_topk_prob,
            .activation = options.activation,
            .gate = gate,
            .routed_experts = routed,
            .shared_experts = shared,
        };
    }

    /// 库内标准参数初始化：在已分配的门控层与全部专家张量上按 options 重新填充权重与偏置
    pub fn resetParameters(self: *MoELayer, random: std.Random, options: InitOptions) void {
        self.gate.resetParameters(random, options);
        for (self.routed_experts) |*exp| {
            exp.resetParameters(random, options);
        }
        for (self.shared_experts) |*exp| {
            exp.resetParameters(random, options);
        }
    }

    pub fn forward(self: *const MoELayer, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        const old_shape = x.shape;
        if (old_shape.len != 2 and old_shape.len != 3) return error.IncompatibleDimensions;
        const is_3d = (old_shape.len == 3);
        var x_2d = x;

        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            if (D != self.dim) return error.ShapeMismatch;
            x_2d = try graph.reshape(x, &.{ B * T, D });
        } else if (old_shape.dims[1] != self.dim) {
            return error.ShapeMismatch;
        }

        const E = self.num_routed_experts;
        const K = self.top_k;
        const arena = graph.arenaAllocator();

        // 1. 门控打分: [N, D] -> [N, E]
        const gate_logits = try self.gate.forward(graph, x_2d);

        // 2. 先对全量路由专家得分执行归一化指数函数 (Softmax)，再通过无分支前 K 项选择算子 (TopK)
        // 提取每词元胜出的前 K 个门控概率 values [N, K] 与专家下标 indices [N, K]
        const raw_probs = try graph.softmax(gate_logits); // [N, E]
        const topk = try graph.topK(raw_probs, K);
        var topk_probs = topk.values; // [N, K]

        if (self.norm_topk_prob and K > 1) {
            const prob_sum = try graph.sum(topk_probs, 1, true);
            const eps_sum = try graph.addScalar(prob_sum, 1e-12);
            topk_probs = try graph.div(topk_probs, eps_sum);
        }

        // 3. 索引收集专家矩阵乘法 (GatherMatMul)：
        // 直接按 topk.indices [N, K] 寻址被选中的 K 个专家权重并执行升维、高斯误差线性单元 (Gaussian Error Linear Unit, GELU) 激活与降维，
        // 复杂度严格为 O(N * K)，零主机端或数据依赖 if 分支
        const fc_weights = try arena.alloc(*Tensor, E);
        const fc_biases = try arena.alloc(*Tensor, E);
        const proj_weights = try arena.alloc(*Tensor, E);
        const proj_biases = try arena.alloc(*Tensor, E);
        for (self.routed_experts, 0..) |exp, e| {
            fc_weights[e] = exp.c_fc.weight;
            fc_biases[e] = exp.c_fc.bias.?;
            proj_weights[e] = exp.c_proj.weight;
            proj_biases[e] = exp.c_proj.bias.?;
        }

        const routed_out = blk: {
            const routed_scope = try graph.enterChildScope("routed_experts", "ModuleList");
            defer routed_scope.exit();
            const h1 = try graph.gatherMatMul(x_2d, topk.indices, fc_weights, fc_biases); // [N, K, hidden_dim]
            const a1 = try self.activation.apply(graph, h1); // [N, K, hidden_dim]
            break :blk try graph.gatherMatMul(a1, topk.indices, proj_weights, proj_biases); // [N, K, D]
        };

        const probs_3d = try graph.unsqueeze(topk_probs, 2); // [N, K, 1]
        const weighted_routed = try graph.mul(routed_out, probs_3d); // [N, K, D]
        const total_routed = try graph.sum(weighted_routed, 1, false); // [N, D]

        // 4. 计算常驻共享专家 (Shared Experts)
        var total_shared: ?*Tensor = null;
        for (self.shared_experts) |exp| {
            const s_out = try exp.forward(graph, x_2d);

            if (total_shared) |s| {
                total_shared = try graph.add(s, s_out);
            } else {
                total_shared = s_out;
            }
        }

        // 5. 合并路由专家 (Routed Experts) 与共享专家 (Shared Experts)
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
