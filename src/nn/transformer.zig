const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");
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

// LLM 微调、对齐损失与采样子模块符号导出
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
/// 用于将离散的 Token ID（例如整数索引）映射为连续的低维稠密向量。
/// 在数学上，这等价于使用 One-hot 编码与权重矩阵相乘，而在实现上通过高效的查找表 (Lookup Table) 实现。
///
/// 权重形状：[vocab_size, embedding_dim]
pub const Embedding = struct {
    weight: *Tensor, // 嵌入层权重矩阵表 (Shape: [vocab_size, embedding_dim])
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "Embedding",

    /// 嵌入层初始化选项
    pub const Options = struct {
        init_method: InitMethod = .{ .normal = .{ .mean = 0.0, .std = 0.02 } },

        pub const default: Options = .{};
        pub fn defaultOptions() Options {
            return .{};
        }
    };

    /// 构造嵌入层：默认只分配词表张量形状和内存；若显式传入可选的 random: ?std.Random 则立即标记 customInit
    pub fn init(allocator: std.mem.Allocator, vocab_size: usize, embedding_dim: usize, random_opt: anytype) !Embedding {
        var emb = try initClean(allocator, vocab_size, embedding_dim);
        const ArgT = @TypeOf(random_opt);
        if (ArgT == std.Random) {
            emb.customInit(random_opt, Options.default);
        } else if (ArgT == ?std.Random) {
            if (random_opt) |rnd| {
                emb.customInit(rnd, Options.default);
            }
        }
        return emb;
    }

    /// 纯结构与内存初始化 (无随机数，交由 Graph.initWeights 自动探查推导)
    pub fn initClean(allocator: std.mem.Allocator, vocab_size: usize, embedding_dim: usize) !Embedding {
        const weight = try createPersistentTensor(allocator, vocab_size, embedding_dim, true);
        return Embedding{
            .weight = weight,
        };
    }

    /// 显式自定义初始化后门：由用户手动指定策略或在模型 customInit 中调用，
    /// 执行后标记 is_custom_initialized = true，Graph.initWeights 遍历时将绝对跳过，不会被重写！
    pub fn customInit(self: *Embedding, random: std.Random, options: Options) void {
        const vocab_size = self.weight.shape.dims[0];
        const embedding_dim = self.weight.shape.dims[1];
        initWeights(random, self.weight.data, vocab_size, embedding_dim, options.init_method);
        self.weight.is_custom_initialized = true;
    }

    /// 为嵌入层及权重张量设置人类可读的名称 (如传入 "wte"，自动设置 "wte.weight")
    pub fn setName(self: *Embedding, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.weight.setNameFormatted("{s}.weight", .{self.name.?});
    }

    /// 使用格式化模板为嵌入层设置人类可读的名称 (如 "{s}.wte", parent_name)
    pub fn setNameFormatted(self: *Embedding, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("embedding");
        }
    }

    /// 获取嵌入层的人类可读名称
    pub fn getName(self: *const Embedding) ?[]const u8 {
        return self.name;
    }

    /// 释放层内所有关联的 Tensor 内存资源
    pub fn deinit(self: Embedding, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
    }

    /// 清空权重对应的梯度
    pub fn zeroGrad(self: Embedding) void {
        self.weight.zeroGrad();
    }

    /// 模块标准数学变换公式
    pub const formula = "y = \\text{Embedding}(\\text{indices}; W_e \\in \\mathbb{R}^{V \\times D})";

    pub fn registerFormula(self: *const Embedding, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
        }
    }

    /// 查找映射前向传播
    /// 输入 x 为包含 Token ID 的任意维度 Tensor、`GenericTensor(IntT)` 或整数切片，输出形状为 x.shape + [embedding_dim]
    pub fn forward(self: Embedding, graph: *autodiff.Graph, x: anytype) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
        return try graph.embedding(self.weight, x);
    }
};

// ============================================================================
// 2. 前馈网络模块 (MLP & SwiGLU)
// ============================================================================

/// 多层感知机 (MLP) / 前馈网络 (Feed-forward Network) 模块
/// Transformer 架构中的重要组件，紧跟在 Self-Attention 之后，
/// 用于在每个 Token 位置上独立地进行非线性特征投影与融合。
///
/// 数学公式：
/// \text{MLP}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2
/// 结构：
/// Linear(dim -> hidden_dim) -> GELU 激活函数 -> Linear(hidden_dim -> dim)
/// 其中 hidden_dim 通常设置为 4 * dim。
pub const MLP = struct {
    c_fc: Linear, // 升维投影层 (dim -> hidden_dim)
    c_proj: Linear, // 降维投影层 (hidden_dim -> dim)
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "MLP",

    /// 初始化 MLP 模块
    /// dim: 输入与输出隐藏维度
    /// hidden_dim: 中间隐藏维度 (一般为 4 * dim)
    pub fn init(allocator: std.mem.Allocator, dim: usize, hidden_dim: usize, random: std.Random) !MLP {
        const c_fc = try Linear.init(allocator, dim, hidden_dim, random);
        errdefer c_fc.deinit(allocator);
        const c_proj = try Linear.init(allocator, hidden_dim, dim, random);
        errdefer c_proj.deinit(allocator);

        return MLP{
            .c_fc = c_fc,
            .c_proj = c_proj,
        };
    }

    /// 为 MLP 模块及子层统一设置人类可读的名称 (自动设置 "{name}.c_fc" 与 "{name}.c_proj")
    pub fn setName(self: *MLP, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.c_fc.setNameFormatted("{s}.c_fc", .{self.name.?});
        self.c_proj.setNameFormatted("{s}.c_proj", .{self.name.?});
    }

    pub fn setNameFormatted(self: *MLP, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("mlp");
        }
    }

    pub fn getName(self: *const MLP) ?[]const u8 {
        return self.name;
    }

    /// 释放子层的所有内存资源
    pub fn deinit(self: MLP, allocator: std.mem.Allocator) void {
        self.c_fc.deinit(allocator);
        self.c_proj.deinit(allocator);
    }

    /// 子层梯度全部清零
    pub fn zeroGrad(self: MLP) void {
        self.c_fc.zeroGrad();
        self.c_proj.zeroGrad();
    }

    /// 模块标准数学变换公式
    pub const formula = "y = \\text{GELU}(x W_{fc}^T + b_{fc}) W_{proj}^T + b_{proj}";

    pub fn registerFormula(self: *const MLP, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
        }
    }

    /// 前向传播逻辑
    /// 支持输入 2D Tensor [B*T, D] 或 3D Tensor [B, T, D]
    pub fn forward(self: MLP, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        var x_2d = x;

        // 1. 如果输入是 3D [B, T, D]，则将其打平为 2D [B*T, D] 以满足 Linear 矩阵乘法的输入规范
        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            x_2d = try graph.reshape(x, &.{ B * T, D });
        }

        // 2. 升维映射: [B*T, D] -> [B*T, hidden_dim]
        const h1 = try self.c_fc.forward(graph, x_2d);

        // 3. GELU 激活函数引入非线性
        const a1 = try graph.gelu(h1);
        if (self.name) |mod_name| a1.setNameFormatted("{s}.gelu", .{mod_name});

        // 4. 降维投射回原始特征维度: [B*T, hidden_dim] -> [B*T, D]
        const h2 = try self.c_proj.forward(graph, a1);

        // 5. 如果输入原本是 3D，需要将输出再重新恢复成 3D 形状: [B, T, D]
        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            return try graph.reshape(h2, &.{ B, T, D });
        }
        return h2;
    }
};

/// 现代 Transformer 门控前馈网络 (SwiGLU / LLaMA-style MLP)
/// 结构：(SiLU(x * W_gate) * (x * W_up)) * W_down
/// 其中 hidden_dim 通常设置为 8/3 * dim
pub const SwiGLU = struct {
    w_gate: Linear, // 门控投影层 (dim -> hidden_dim)
    w_up: Linear, // 升维投影层 (dim -> hidden_dim)
    w_down: Linear, // 降维投影层 (hidden_dim -> dim)
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "SwiGLU",

    pub fn init(allocator: std.mem.Allocator, dim: usize, hidden_dim: usize, random: std.Random) !SwiGLU {
        const w_gate = try Linear.init(allocator, dim, hidden_dim, random);
        errdefer w_gate.deinit(allocator);
        const w_up = try Linear.init(allocator, dim, hidden_dim, random);
        errdefer w_up.deinit(allocator);
        const w_down = try Linear.init(allocator, hidden_dim, dim, random);
        errdefer w_down.deinit(allocator);

        return SwiGLU{
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
        };
    }

    /// 为 SwiGLU 模块及子层统一设置人类可读的名称 (自动设置 "{name}.w_gate", "{name}.w_up", "{name}.w_down")
    pub fn setName(self: *SwiGLU, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.w_gate.setNameFormatted("{s}.w_gate", .{self.name.?});
        self.w_up.setNameFormatted("{s}.w_up", .{self.name.?});
        self.w_down.setNameFormatted("{s}.w_down", .{self.name.?});
    }

    pub fn setNameFormatted(self: *SwiGLU, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("swiglu");
        }
    }

    pub fn getName(self: *const SwiGLU) ?[]const u8 {
        return self.name;
    }

    pub fn deinit(self: SwiGLU, allocator: std.mem.Allocator) void {
        self.w_gate.deinit(allocator);
        self.w_up.deinit(allocator);
        self.w_down.deinit(allocator);
    }

    pub fn zeroGrad(self: SwiGLU) void {
        self.w_gate.zeroGrad();
        self.w_up.zeroGrad();
        self.w_down.zeroGrad();
    }

    /// 模块标准数学变换公式
    pub const formula = "y = (\\text{SiLU}(x W_{\\text{gate}}) \\odot (x W_{\\text{up}})) W_{\\text{down}}";

    pub fn registerFormula(self: *const SwiGLU, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
        }
    }

    /// 前向传播逻辑：支持 2D [B*T, D] 或 3D [B, T, D]
    pub fn forward(self: SwiGLU, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        var x_2d = x;

        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            x_2d = try graph.reshape(x, &.{ B * T, D });
        }

        // 1. 计算 gate 投影: [B*T, D] -> [B*T, hidden_dim]
        const gate = try self.w_gate.forward(graph, x_2d);

        // 2. 计算 up 投影: [B*T, D] -> [B*T, hidden_dim]
        const up = try self.w_up.forward(graph, x_2d);

        // 3. 计算 SiLU(gate) 激活
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
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "MoELayer",

    pub const formula = "y = \\sum_{i \\in \\text{TopK}(g(x))} p_i(x) E_i(x) + \\sum_{j} E^{\\text{shared}}_j(x)";

    pub fn init(
        allocator: std.mem.Allocator,
        dim: usize,
        hidden_dim: usize,
        num_routed_experts: usize,
        num_shared_experts: usize,
        top_k: usize,
        random: std.Random,
    ) !MoELayer {
        std.debug.assert(top_k > 0 and top_k <= num_routed_experts);

        const gate = try Linear.init(allocator, dim, num_routed_experts, random);
        errdefer gate.deinit(allocator);

        const routed = try allocator.alloc(MLP, num_routed_experts);
        errdefer allocator.free(routed);

        var init_r: usize = 0;
        errdefer {
            for (0..init_r) |i| routed[i].deinit(allocator);
        }
        for (0..num_routed_experts) |i| {
            routed[i] = try MLP.init(allocator, dim, hidden_dim, random);
            init_r += 1;
        }

        const shared = try allocator.alloc(MLP, num_shared_experts);
        errdefer allocator.free(shared);

        var init_s: usize = 0;
        errdefer {
            for (0..init_s) |i| shared[i].deinit(allocator);
        }
        for (0..num_shared_experts) |i| {
            shared[i] = try MLP.init(allocator, dim, hidden_dim, random);
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

    pub fn setName(self: *MoELayer, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.gate.setNameFormatted("{s}.gate", .{self.name.?});
        for (self.routed_experts, 0..) |*exp, i| {
            exp.setNameFormatted("{s}.routed_{d}", .{ self.name.?, i });
        }
        for (self.shared_experts, 0..) |*exp, i| {
            exp.setNameFormatted("{s}.shared_{d}", .{ self.name.?, i });
        }
    }

    pub fn setNameFormatted(self: *MoELayer, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("moe");
        }
    }

    pub fn getName(self: *const MoELayer) ?[]const u8 {
        return self.name;
    }

    pub fn registerFormula(self: *const MoELayer, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
            try graph.registerModuleType(n, self.module_type);
            try self.gate.registerFormula(graph);
            for (self.routed_experts) |*exp| {
                try exp.registerFormula(graph);
            }
            for (self.shared_experts) |*exp| {
                try exp.registerFormula(graph);
            }
        }
    }

    pub fn deinit(self: MoELayer, allocator: std.mem.Allocator) void {
        self.gate.deinit(allocator);
        for (self.routed_experts) |exp| exp.deinit(allocator);
        allocator.free(self.routed_experts);
        for (self.shared_experts) |exp| exp.deinit(allocator);
        allocator.free(self.shared_experts);
    }

    pub fn zeroGrad(self: MoELayer) void {
        self.gate.zeroGrad();
        for (self.routed_experts) |exp| exp.zeroGrad();
        for (self.shared_experts) |exp| exp.zeroGrad();
    }

    pub fn forward(self: MoELayer, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
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

        // 2. 构造 Top-K 掩码并执行 Softmax 归一化
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

        // 对每一行寻找 Top-K 个最大的索引
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
        const probs = try graph.mul(raw_probs, keep_node); // 严格置零非 Top-K 概率
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

        // 4. 合并 Routed 与 Shared 专家
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
// 4. Transformer Block 与 Decoder
// ============================================================================

/// Transformer 编码器/解码器 Block 模块 (Transformer Block)
/// 采用 Pre-LN (Layer Normalization Pre-activation) 架构进行组装：
/// 1. x_norm1 = RMSNorm(x)
/// 2. x_attn = SelfAttention(x_norm1)
/// 3. x1 = x + x_attn  (第一层残差连接)
/// 4. x_norm2 = RMSNorm(x1)
/// 5. x_mlp = MLP(x_norm2)
/// 6. out = x1 + x_mlp (第二层残差连接)
pub const TransformerBlock = struct {
    ln_1: RMSNorm, // 第一层归一化层，在 Attention 计算前执行
    attn: CausalSelfAttention, // 因果自注意力机制层
    ln_2: RMSNorm, // 第二层归一化层，在 MLP 计算前执行
    mlp: MLP, // 前馈多层感知机层
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "TransformerBlock",

    /// 初始化 Transformer 块
    /// n_embd: 隐藏特征特征维度
    /// n_head: 注意力头数
    pub fn init(allocator: std.mem.Allocator, n_embd: usize, n_head: usize, random: std.Random) !TransformerBlock {
        const ln_1 = try RMSNorm.init(allocator, n_embd, 1e-5);
        errdefer ln_1.deinit(allocator);
        const attn = try CausalSelfAttention.init(allocator, n_embd, n_head, random);
        errdefer attn.deinit(allocator);
        const ln_2 = try RMSNorm.init(allocator, n_embd, 1e-5);
        errdefer ln_2.deinit(allocator);
        const mlp = try MLP.init(allocator, n_embd, 4 * n_embd, random);
        errdefer mlp.deinit(allocator);

        return TransformerBlock{
            .ln_1 = ln_1,
            .attn = attn,
            .ln_2 = ln_2,
            .mlp = mlp,
        };
    }

    /// 为 Transformer Block 及内部各子层统一设置分层名称 (自动递归设置 ln_1, attn, ln_2, mlp)
    pub fn setName(self: *TransformerBlock, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.ln_1.setNameFormatted("{s}.ln_1", .{self.name.?});
        self.attn.setNameFormatted("{s}.attn", .{self.name.?});
        self.ln_2.setNameFormatted("{s}.ln_2", .{self.name.?});
        self.mlp.setNameFormatted("{s}.mlp", .{self.name.?});
    }

    pub fn setNameFormatted(self: *TransformerBlock, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("block");
        }
    }

    pub fn getName(self: *const TransformerBlock) ?[]const u8 {
        return self.name;
    }

    /// 释放所有内部子层的资源
    pub fn deinit(self: TransformerBlock, allocator: std.mem.Allocator) void {
        self.ln_1.deinit(allocator);
        self.attn.deinit(allocator);
        self.ln_2.deinit(allocator);
        self.mlp.deinit(allocator);
    }

    /// 块内所有子层的梯度清零
    pub fn zeroGrad(self: TransformerBlock) void {
        self.ln_1.zeroGrad();
        self.attn.zeroGrad();
        self.ln_2.zeroGrad();
        self.mlp.zeroGrad();
    }

    /// 模块标准数学变换公式
    pub const formula = "\\begin{aligned} h_l &= x_l + \\text{Attention}(\\text{RMSNorm}(x_l)) \\\\ x_{l+1} &= \\text{TransformerBlock}(x_l) = h_l + \\text{MLP}(\\text{RMSNorm}(h_l)) \\end{aligned}";

    pub fn registerFormula(self: *const TransformerBlock, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
            try graph.registerModuleType(n, self.module_type);
            try self.ln_1.registerFormula(graph);
            try self.attn.registerFormula(graph);
            try self.ln_2.registerFormula(graph);
            try self.mlp.registerFormula(graph);
        }
    }

    /// 前向传播流程：x -> Block(x) -> out
    pub fn forward(self: TransformerBlock, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
        // 1. 第一条支路: RMSNorm -> Attention
        const x_norm1 = try self.ln_1.forward(graph, x);

        const x_attn = try self.attn.forward(graph, x_norm1);

        // 2. 第一条残差混合: x1 = x_l + Attention(RMSNorm(x_l))
        const x1 = try graph.add(x, x_attn);
        if (self.name) |mod_name| {
            x1.setNameFormatted("{s}.residual_attn", .{mod_name});
            try graph.setModuleFormula(x1.name.?, "x_1 = x_l + \\text{Attention}(\\text{RMSNorm}(x_l))");
        }

        // 3. 第二条支路: RMSNorm -> MLP
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

/// 堆叠多层 Transformer 块的解码器主干网络 (Transformer Decoder)
pub fn TransformerDecoder(comptime n_layer: usize) type {
    return struct {
        const Self = @This();

        h: [n_layer]TransformerBlock, // 堆叠的 Blocks 数组
        ln_f: RMSNorm, // 骨架最末端用于规范化的归一化层

        name: ?[]const u8 = null,
        name_buf: [64]u8 = undefined,
        module_type: []const u8 = "TransformerDecoder",

        /// 为整个 Decoder 骨架及其包含的每层 Block 统一设置分层名称
        pub fn setName(self: *Self, name: []const u8) void {
            if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
                self.name = s;
            } else |_| {
                self.name = name;
            }
            for (&self.h, 0..) |*layer, i| {
                layer.setNameFormatted("{s}.{d}", .{ self.name.?, i });
            }
            self.ln_f.setNameFormatted("{s}.ln_f", .{self.name.?});
        }

        pub fn setNameFormatted(self: *Self, comptime fmt: []const u8, args: anytype) void {
            var buf: [64]u8 = undefined;
            if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
                self.setName(s);
            } else |_| {
                self.setName("decoder");
            }
        }

        pub fn getName(self: *const Self) ?[]const u8 {
            return self.name;
        }

        /// 初始化整个解码器组件
        pub fn init(allocator: std.mem.Allocator, n_embd: usize, n_head: usize, random: std.Random) !Self {
            var h: [n_layer]TransformerBlock = undefined;
            var i: usize = 0;
            errdefer {
                for (0..i) |j| {
                    h[j].deinit(allocator);
                }
            }
            // 循环初始化每一层 TransformerBlock
            while (i < n_layer) : (i += 1) {
                h[i] = try TransformerBlock.init(allocator, n_embd, n_head, random);
            }

            // 初始化最后的层归一化层
            const ln_f = try RMSNorm.init(allocator, n_embd, 1e-5);
            errdefer {
                for (0..n_layer) |j| {
                    h[j].deinit(allocator);
                }
                ln_f.deinit(allocator);
            }

            return Self{
                .h = h,
                .ln_f = ln_f,
            };
        }

        /// 释放整个骨架层及各 Block 的内存
        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            for (self.h) |layer| {
                layer.deinit(allocator);
            }
            self.ln_f.deinit(allocator);
        }

        /// 将所有 Block 和最末端 Norm 层的梯度全部清零
        pub fn zeroGrad(self: Self) void {
            for (self.h) |layer| {
                layer.zeroGrad();
            }
            self.ln_f.zeroGrad();
        }

        /// 模块标准数学变换公式
        pub const formula = "x_L = \\text{DecoderStack}(x_0) = (\\text{Block}_L \\circ \\dots \\circ \\text{Block}_1)(x_0)";

        pub fn registerFormula(self: *const Self, graph: *autodiff.Graph) !void {
            if (self.name) |n| {
                try graph.setModuleFormula(n, formula);
                try graph.registerModuleType(n, self.module_type);
                for (&self.h) |*layer| {
                    try layer.registerFormula(graph);
                }
                try self.ln_f.registerFormula(graph);
            }
        }

        /// 解码器主干网络的前向传播流程
        pub fn forward(self: *const Self, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
            const module_scope = try graph.enterModule(self.name, self.module_type);
            defer module_scope.exit();
            if (self.name) |n| try graph.setModuleFormula(n, formula);
            var current_x = x;
            // 依次贯穿每一层 Block
            for (self.h) |layer| {
                current_x = try layer.forward(graph, current_x);
            }

            // 执行最后一层 RMSNorm 映射输出
            return try self.ln_f.forward(graph, current_x);
        }
    };
}

// ============================================================================
// 5. GPT 模型定义
// ============================================================================

/// GPT 模型配置结构体
pub const GPTConfig = struct {
    vocab_size: usize = 50257, // 词表大小 (Vocab Size)，决定输入和输出层的映射维度
    block_size: usize = 1024, // 最大上下文长度/时间步长度 (Context Length / Block Size)
    n_embd: usize = 768, // 隐藏特征嵌入维度 (Embedding Dimension)
    n_head: usize = 12, // 多头注意力头数 (Attention Heads)
    n_layer: usize = 12, // Transformer 块堆叠的层数 (Number of Decoder Layers)

    pub const default: GPTConfig = .{};
    pub fn defaultConfig() GPTConfig {
        return .{};
    }
};

/// 泛型 GPT 模型定义函数
pub fn GPT(comptime config: GPTConfig) type {
    return struct {
        token_embedding: Embedding, // Token 嵌入层
        position_embedding: Embedding, // 位置嵌入层
        decoder: TransformerDecoder(config.n_layer), // 堆叠的解码器层与最终归一化层
        lm_head: Linear, // 最终输出概率的线性分类投影头
        name: ?[]const u8 = null,
        name_buf: [64]u8 = undefined,
        module_type: []const u8 = "GPT",

        const Self = @This();

        /// 为 GPT 顶层及其包含的 embedding、decoder、lm_head 统一设置分层命名
        pub fn setName(self: *Self, name: []const u8) void {
            if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
                self.name = s;
            } else |_| {
                self.name = name;
            }
            self.token_embedding.setNameFormatted("{s}.wte", .{self.name.?});
            self.position_embedding.setNameFormatted("{s}.wpe", .{self.name.?});
            self.decoder.setNameFormatted("{s}.layers", .{self.name.?});
            self.lm_head.setNameFormatted("{s}.lm_head", .{self.name.?});
        }

        pub fn setNameFormatted(self: *Self, comptime fmt: []const u8, args: anytype) void {
            var buf: [64]u8 = undefined;
            if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
                self.setName(s);
            } else |_| {
                self.setName("gpt");
            }
        }

        pub fn getName(self: *const Self) ?[]const u8 {
            return self.name;
        }

        /// 释放 GPT 模型所有子模块的内存资源
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.token_embedding.deinit(allocator);
            self.position_embedding.deinit(allocator);
            self.decoder.deinit(allocator);
            self.lm_head.deinit(allocator);
        }

        /// 模型所有参数梯度清零
        pub fn zeroGrad(self: *Self) void {
            self.token_embedding.zeroGrad();
            self.position_embedding.zeroGrad();
            self.decoder.zeroGrad();
            self.lm_head.zeroGrad();
        }

        /// 初始化默认配置的 GPT 模型实例
        pub fn initDefault(allocator: std.mem.Allocator, random: std.Random) !Self {
            return init(allocator, random);
        }

        /// 初始化 GPT 模型中的所有网络层权重
        pub fn init(allocator: std.mem.Allocator, random: std.Random) !Self {
            // 初始化 Token 嵌入矩阵 [vocab_size, n_embd]
            const token_embedding = try Embedding.init(allocator, config.vocab_size, config.n_embd, random);
            errdefer token_embedding.deinit(allocator);

            // 初始化位置嵌入矩阵 [block_size, n_embd]
            const position_embedding = try Embedding.init(allocator, config.block_size, config.n_embd, random);
            errdefer position_embedding.deinit(allocator);

            // 初始化 Decoder 主干网络
            const decoder = try TransformerDecoder(config.n_layer).init(allocator, config.n_embd, config.n_head, random);
            errdefer {
                token_embedding.deinit(allocator);
                position_embedding.deinit(allocator);
                decoder.deinit(allocator);
            }

            // 初始化输出映射分类头 [n_embd, vocab_size]
            const lm_head = try Linear.init(allocator, config.n_embd, config.vocab_size, random);
            errdefer {
                token_embedding.deinit(allocator);
                position_embedding.deinit(allocator);
                decoder.deinit(allocator);
                lm_head.deinit(allocator);
            }

            return Self{
                .token_embedding = token_embedding,
                .position_embedding = position_embedding,
                .decoder = decoder,
                .lm_head = lm_head,
            };
        }

        /// 模块标准数学变换公式
        pub const formula = "\\text{logits} = \\text{GPT}(\\text{TokenIDs}; \\theta) \\rightarrow [B, T, V]";

        pub fn registerFormula(self: *const Self, graph: *autodiff.Graph) !void {
            if (self.name) |n| {
                try graph.setModuleFormula(n, formula);
                try graph.registerModuleType(n, self.module_type);
                try self.token_embedding.registerFormula(graph);
                try self.position_embedding.registerFormula(graph);
                try self.decoder.registerFormula(graph);
                try self.lm_head.registerFormula(graph);
            }
        }

        /// 前向推理传播流程
        /// 输入 x 为包含 Token ID 的 2D 张量（支持 `*Tensor` 或 `*GenericTensor(IntT)`），形状为 [B, T]
        /// 输出为未归一化的预测对数 (Logits)，形状为 3D: [B, T, vocab_size]
        pub fn forward(self: *const Self, graph: *autodiff.Graph, x: anytype) !*Tensor {
            const module_scope = try graph.enterModule(self.name, self.module_type);
            defer module_scope.exit();
            if (self.name) |n| try graph.setModuleFormula(n, formula);
            const B = x.shape.dims[0];
            const T = x.shape.dims[1];

            // 1. 获取 Token 嵌入向量: [B, T] -> [B, T, n_embd]
            const tok_emb = try self.token_embedding.forward(graph, x);

            // 2. 生成对应的时间/位置索引 [0, 1, 2, ... T-1]，并将其转换为 2D 位置 Tensor [B, T]
            const pos_node = try graph.tensorND(&.{ B, T }, false);
            for (0..B) |b| {
                for (0..T) |t| {
                    pos_node.data[b * T + t] = @as(f32, @floatFromInt(t));
                }
            }
            // 位置索引由输入形状在模型内部生成，属于 GPT 自身的常量缓冲区，而非模型输入
            pos_node.is_buffer = true;
            if (self.name) |mod_name| {
                pos_node.setNameFormatted("{s}.pos_indices", .{mod_name});
            }

            // 3. 获取对应的 Learned 位置嵌入向量: [B, T] -> [B, T, n_embd]
            const pos_emb = try self.position_embedding.forward(graph, pos_node);

            // 4. 将 Token 嵌入和位置嵌入进行求和融合，作为初始隐藏输入: h = tok_emb + pos_emb
            const h_x = try graph.add(tok_emb, pos_emb);
            if (self.name) |mod_name| {
                h_x.setNameFormatted("{s}.embeddings_sum", .{mod_name});
            }

            // 5. 将混合后的输入送进层叠的 Decoder 主干网络中依次计算
            // 输出形状保持为: [B, T, n_embd]
            const decoder_out = try self.decoder.forward(graph, h_x);

            // 6. 将输出展平为 2D，以便进行最终分类头的全连接投影计算: [B, T, n_embd] -> [B*T, n_embd]
            const ln_x_2d = try graph.reshape(decoder_out, &.{ B * T, config.n_embd });

            // 7. 进行投影以获得词表空间未归一化的分类 Logits: [B*T, n_embd] -> [B*T, vocab_size]
            const logits_2d = try self.lm_head.forward(graph, ln_x_2d);

            // 8. 将形状重塑还原成 3D 形式返回: [B, T, vocab_size]
            return try graph.reshape(logits_2d, &.{ B, T, config.vocab_size });
        }
    };
}

/// 采用默认 GPTConfig 的标准 GPT 模型类型别名
pub const DefaultGPT = GPT(GPTConfig.default);
