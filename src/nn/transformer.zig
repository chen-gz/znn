const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");
const normalization = @import("normalization.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Linear = core.Linear;
const RMSNorm = normalization.RMSNorm;
const createPersistentTensor = core.createPersistentTensor;
const freePersistentTensor = core.freePersistentTensor;
const initializeWeights = core.initializeWeights;
const initWeights = core.initWeights;
const InitMethod = core.InitMethod;

// ============================================================================
// 1. 嵌入层 (Embedding Layer)
// ============================================================================

/// 嵌入层 (Embedding Layer)
/// 用于将离散的 Token ID（例如整数索引）映射为连续的低维稠密向量。
/// 在数学上，这等价于使用 One-hot 编码与权重矩阵相乘，而在实现上通过高效的查找表 (Lookup Table) 实现。
/// 
/// 权重形状：[vocab_size, embedding_dim]
pub const Embedding = struct {
    weight: *Tensor,        // 嵌入层权重矩阵表 (Shape: [vocab_size, embedding_dim])
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,

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

    /// 查找映射前向传播
    /// 输入 x 为包含 Token ID 的任意维度 Tensor，输出形状为 x.shape + [embedding_dim]
    pub fn forward(self: Embedding, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        return try self.weight.embedding(x, allocator, graph);
    }
};

// ============================================================================
// 2. 键值缓存 (Key-Value Cache)
// ============================================================================

/// 键值缓存 (Key-Value Cache) 用于大模型自回归增量推理 (O(T) 生成复杂度)
pub const KVCache = struct {
    k: *Tensor,             // 缓存的 Key 张量 [batch_size, n_head, max_seq_len, head_dim]
    v: *Tensor,             // 缓存的 Value 张量 [batch_size, n_head, max_seq_len, head_dim]
    curr_len: usize = 0,    // 当前已缓存的 Token 步长
    max_len: usize,         // 最大支持上下文长度

    pub fn init(allocator: std.mem.Allocator, batch_size: usize, n_head: usize, max_len: usize, head_dim: usize) !KVCache {
        const k = try createPersistentTensor(allocator, 1, batch_size * n_head * max_len * head_dim, false);
        k.shape = Shape.init(&.{ batch_size, n_head, max_len, head_dim });
        k.strides = tensor.computeContiguousStrides(k.shape);
        @memset(k.data, 0.0);

        const v = try createPersistentTensor(allocator, 1, batch_size * n_head * max_len * head_dim, false);
        v.shape = Shape.init(&.{ batch_size, n_head, max_len, head_dim });
        v.strides = tensor.computeContiguousStrides(v.shape);
        @memset(v.data, 0.0);

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
};

// ============================================================================
// 3. 前馈网络模块 (MLP & SwiGLU)
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
    c_fc: Linear,           // 升维投影层 (dim -> hidden_dim)
    c_proj: Linear,         // 降维投影层 (hidden_dim -> dim)
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,

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

    /// 前向传播逻辑
    /// 支持输入 2D Tensor [B*T, D] 或 3D Tensor [B, T, D]
    pub fn forward(self: MLP, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        var x_2d = x;
        
        // 1. 如果输入是 3D [B, T, D]，则将其打平为 2D [B*T, D] 以满足 Linear 矩阵乘法的输入规范
        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            if (graph) |g| {
                x_2d = try g.reshape(x, &.{ B * T, D });
            } else {
                x_2d = try x.reshape(&.{ B * T, D }, allocator, null);
            }
        }
        defer {
            // Eager 模式下需要释放临时 reshape 生成的 Tensor 内存
            if (is_3d and graph == null) {
                tensor.free(allocator, x_2d);
            }
        }

        // 2. 升维映射: [B*T, D] -> [B*T, hidden_dim]
        const h1 = try self.c_fc.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, h1);

        // 3. GELU 激活函数引入非线性
        const a1 = if (graph) |g| try g.gelu(h1) else try h1.gelu(allocator, null);
        defer if (graph == null) tensor.free(allocator, a1);

        // 4. 降维投射回原始特征维度: [B*T, hidden_dim] -> [B*T, D]
        const h2 = try self.c_proj.forward(allocator, graph, a1);

        // 5. 如果输入原本是 3D，需要将输出再重新恢复成 3D 形状: [B, T, D]
        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            if (graph) |g| {
                return try g.reshape(h2, &.{ B, T, D });
            } else {
                defer tensor.free(allocator, h2);
                return try h2.reshape(&.{ B, T, D }, allocator, null);
            }
        }
        return h2;
    }
};

/// SwiGLU 门控激活算子：output = (gate * sigmoid(gate)) * up
pub fn swigluForward(
    output: []f32,
    gate: []const f32,
    up: []const f32,
) void {
    std.debug.assert(gate.len == up.len and output.len == gate.len);

    for (gate, up, 0..) |g_val, u_val, idx| {
        const sigmoid_g = if (g_val >= 0.0) 1.0 / (1.0 + @exp(-g_val)) else @exp(g_val) / (1.0 + @exp(g_val));
        const swish_g = g_val * sigmoid_g;
        output[idx] = swish_g * u_val;
    }
}

/// 现代 Transformer 门控前馈网络 (SwiGLU / LLaMA-style MLP)
/// 结构：(SiLU(x * W_gate) * (x * W_up)) * W_down
/// 其中 hidden_dim 通常设置为 8/3 * dim
pub const SwiGLU = struct {
    w_gate: Linear,         // 门控投影层 (dim -> hidden_dim)
    w_up: Linear,           // 升维投影层 (dim -> hidden_dim)
    w_down: Linear,         // 降维投影层 (hidden_dim -> dim)
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,

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

    /// 前向传播逻辑：支持 2D [B*T, D] 或 3D [B, T, D]
    pub fn forward(self: SwiGLU, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        var x_2d = x;

        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            if (graph) |g| {
                x_2d = try g.reshape(x, &.{ B * T, D });
            } else {
                x_2d = try x.reshape(&.{ B * T, D }, allocator, null);
            }
        }
        defer {
            if (is_3d and graph == null) {
                tensor.free(allocator, x_2d);
            }
        }

        // 1. 计算 gate 投影: [B*T, D] -> [B*T, hidden_dim]
        const gate = try self.w_gate.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, gate);

        // 2. 计算 up 投影: [B*T, D] -> [B*T, hidden_dim]
        const up = try self.w_up.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, up);

        // 3. 计算 SiLU(gate) 激活
        const silu_gate = if (graph) |g| try g.silu(gate) else try gate.silu(allocator, null);
        defer if (graph == null) tensor.free(allocator, silu_gate);

        // 4. 逐元素乘法: SiLU(gate) * up
        const hidden = if (graph) |g| try g.mul(silu_gate, up) else try silu_gate.mul(up, allocator, null);
        defer if (graph == null) tensor.free(allocator, hidden);

        // 5. 降维投射回原始特征维度: [B*T, hidden_dim] -> [B*T, D]
        const out = try self.w_down.forward(allocator, graph, hidden);

        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            if (graph) |g| {
                return try g.reshape(out, &.{ B, T, D });
            } else {
                defer tensor.free(allocator, out);
                return try out.reshape(&.{ B, T, D }, allocator, null);
            }
        }
        return out;
    }
};

// ============================================================================
// 4. 混合专家前馈网络层 (Mixture of Experts Layer, MoELayer)
// ============================================================================

/// 混合专家前馈网络层 (Mixture of Experts Layer, MoELayer)
/// 对应现代前沿大模型与 DeepSeekMoE 细粒度专家路由架构：
/// 包含：
/// 1. 细粒度路由专家列表 (routed_experts, 激活 top_k)
/// 2. 可选隔离常驻共享专家列表 (shared_experts, 均无条件激活)
/// 3. 动态门控路由网络 gate: Linear(dim -> num_routed_experts)
/// 4. 门控概率归一化与稀疏加权聚合输出
/// 5. 完备支持 Eager 模式与 Autograd 计算图模式前向与反向传播
pub const MoELayer = struct {
    dim: usize,
    num_routed_experts: usize,
    num_shared_experts: usize,
    top_k: usize,
    gate: Linear,
    routed_experts: []MLP,
    shared_experts: []MLP,

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

    pub fn forward(self: MoELayer, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        var x_2d = x;

        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            if (graph) |g| {
                x_2d = try g.reshape(x, &.{ B * T, D });
            } else {
                x_2d = try x.reshape(&.{ B * T, D }, allocator, null);
            }
        }
        defer {
            if (is_3d and graph == null) {
                tensor.free(allocator, x_2d);
            }
        }

        const N = x_2d.shape.dims[0];
        const E = self.num_routed_experts;
        const K = self.top_k;

        // 1. 门控打分: [N, D] -> [N, E]
        const gate_logits = try self.gate.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, gate_logits);

        // 2. 构造 Top-K 掩码并执行 Softmax 归一化
        const mask_data = try allocator.alloc(f32, N * E);
        defer allocator.free(mask_data);
        @memset(mask_data, -1e9);

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
                mask_data[row * E + top_indices[k]] = 0.0;
            }
        }

        var total_routed: *Tensor = undefined;

        if (graph) |g| {
            const mask_node = try g.tensorNDWithData(&.{ N, E }, mask_data, false);
            const masked_logits = try g.add(gate_logits, mask_node);
            const probs = try g.softmax(masked_logits); // [N, E]
            const prob_cols = try g.split(probs, E, 1); // E 个 [N, 1]

            var acc: ?*Tensor = null;
            for (self.routed_experts, 0..) |exp, e| {
                const exp_out = try exp.forward(allocator, graph, x_2d); // [N, D]
                const weighted = try g.mul(exp_out, prob_cols[e]); // [N, D] * [N, 1] -> [N, D]
                if (acc) |a| {
                    acc = try g.add(a, weighted);
                } else {
                    acc = weighted;
                }
            }
            total_routed = acc.?;
        } else {
            // Eager 模式
            const mask_t = try tensor.array(allocator, &.{ N, E }, mask_data);
            defer tensor.free(allocator, mask_t);
            const masked_logits = try gate_logits.add(mask_t, allocator, null);
            defer tensor.free(allocator, masked_logits);
            const probs = try masked_logits.softmax(allocator, null);
            defer tensor.free(allocator, probs);

            const out_accum = try tensor.zeros(allocator, &.{ N, self.dim });
            errdefer tensor.free(allocator, out_accum);

            for (self.routed_experts, 0..) |exp, e| {
                const exp_out = try exp.forward(allocator, null, x_2d);
                defer tensor.free(allocator, exp_out);

                for (0..N) |row| {
                    const p = probs.data[row * E + e];
                    if (p > 0.0) {
                        for (0..self.dim) |d| {
                            out_accum.data[row * self.dim + d] += p * exp_out.data[row * self.dim + d];
                        }
                    }
                }
            }
            total_routed = out_accum;
        }
        defer if (graph == null and self.num_shared_experts > 0) tensor.free(allocator, total_routed);

        // 3. 计算常驻共享专家 (Shared Experts)
        var total_shared: ?*Tensor = null;
        for (self.shared_experts) |exp| {
            const s_out = try exp.forward(allocator, graph, x_2d);
            defer if (graph == null) tensor.free(allocator, s_out);

            if (total_shared) |s| {
                if (graph) |g| {
                    total_shared = try g.add(s, s_out);
                } else {
                    const new_s = try s.add(s_out, allocator, null);
                    tensor.free(allocator, s);
                    total_shared = new_s;
                }
            } else {
                if (graph) |_| {
                    total_shared = s_out;
                } else {
                    const cloned_s = try tensor.zeros(allocator, s_out.shape.dims[0..s_out.shape.len]);
                    @memcpy(cloned_s.data, s_out.data);
                    total_shared = cloned_s;
                }
            }
        }
        defer if (graph == null and total_shared != null) tensor.free(allocator, total_shared.?);

        // 4. 合并 Routed 与 Shared 专家
        var final_2d = total_routed;
        if (total_shared) |s| {
            if (graph) |g| {
                final_2d = try g.add(total_routed, s);
            } else {
                final_2d = try total_routed.add(s, allocator, null);
            }
        }

        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            if (graph) |g| {
                return try g.reshape(final_2d, &.{ B, T, D });
            } else {
                defer tensor.free(allocator, final_2d);
                return try final_2d.reshape(&.{ B, T, D }, allocator, null);
            }
        }

        return final_2d;
    }
};

// ============================================================================
// 5. 注意力机制 (CausalSelfAttention & MLA)
// ============================================================================

/// 因果自注意力机制 (Causal Self-Attention / Masked Multi-Head Attention)
/// Transformer 的核心机制，负责建模序列中不同位置的依赖关系。
/// 
/// 数学公式：
/// Q = X W_q, \quad K = X W_k, \quad V = X W_v
/// \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V
/// \text{Output} = \text{Attention}(Q, K, V) W_p
/// 其中 M 是因果掩码矩阵，上三角（未来位置）元素为 -\infty，其余为 0。
/// 
/// 包含以下关键设计：
/// 1. 多头注意力 (Multi-Head)：将特征通道划分为 nh 个头，让模型在多个不同的投影子空间内并行关注信息。
/// 2. 因果掩码 (Causal Mask)：通过加上上三角矩阵（值为 -inf），阻止当前位置关注未来的位置，确保自回归生成时的因果律。
pub const CausalSelfAttention = struct {
    q_attn: Linear,         // Query 线性投影层
    k_attn: Linear,         // Key 线性投影层
    v_attn: Linear,         // Value 线性投影层
    c_proj: Linear,         // 最终的多头输出融合与投影层 (c_proj)
    n_head: usize,          // 注意力头数 (Query heads)
    n_embd: usize,          // 嵌入维度 (n_embd)
    num_kv_heads: usize,    // Key / Value 头数 (1 = MQA, < n_head = GQA, == n_head = MHA)
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,

    /// 初始化支持分组查询注意力 (GQA / MQA / MHA) 的自注意力层
    /// n_embd: 隐藏嵌入维度，必须能被 n_head 整除
    /// n_head: Query 注意力头数
    /// num_kv_heads: Key/Value 头数，必须能整除 n_head
    pub fn initGQA(allocator: std.mem.Allocator, n_embd: usize, n_head: usize, num_kv_heads: usize, random: std.Random) !CausalSelfAttention {
        std.debug.assert(n_embd % n_head == 0);
        std.debug.assert(n_head % num_kv_heads == 0);
        const hs = n_embd / n_head;
        const kv_dim = num_kv_heads * hs;

        const q_attn = try Linear.init(allocator, n_embd, n_embd, random);
        errdefer q_attn.deinit(allocator);
        const k_attn = try Linear.init(allocator, n_embd, kv_dim, random);
        errdefer k_attn.deinit(allocator);
        const v_attn = try Linear.init(allocator, n_embd, kv_dim, random);
        errdefer v_attn.deinit(allocator);
        const c_proj = try Linear.init(allocator, n_embd, n_embd, random);
        errdefer c_proj.deinit(allocator);

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

    /// 初始化传统多头自注意力层 (MHA: num_kv_heads == n_head)
    pub fn init(allocator: std.mem.Allocator, n_embd: usize, n_head: usize, random: std.Random) !CausalSelfAttention {
        return initGQA(allocator, n_embd, n_head, n_head, random);
    }

    /// 为注意力层及 4 个线性投影子层统一设置人类可读的名称 (如 "{name}.q_attn", "{name}.c_proj")
    pub fn setName(self: *CausalSelfAttention, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.q_attn.setNameFormatted("{s}.q_attn", .{self.name.?});
        self.k_attn.setNameFormatted("{s}.k_attn", .{self.name.?});
        self.v_attn.setNameFormatted("{s}.v_attn", .{self.name.?});
        self.c_proj.setNameFormatted("{s}.c_proj", .{self.name.?});
    }

    pub fn setNameFormatted(self: *CausalSelfAttention, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("attn");
        }
    }

    pub fn getName(self: *const CausalSelfAttention) ?[]const u8 {
        return self.name;
    }

    /// 释放所有线性投射子层的内存资源
    pub fn deinit(self: CausalSelfAttention, allocator: std.mem.Allocator) void {
        self.q_attn.deinit(allocator);
        self.k_attn.deinit(allocator);
        self.v_attn.deinit(allocator);
        self.c_proj.deinit(allocator);
    }

    /// 所有线性投射子层的梯度清零
    pub fn zeroGrad(self: CausalSelfAttention) void {
        self.q_attn.zeroGrad();
        self.k_attn.zeroGrad();
        self.v_attn.zeroGrad();
        self.c_proj.zeroGrad();
    }

    /// 前向注意力计算流程
    /// 输入 x 的形状必须为 3D: [B, T, C]
    /// 其中 B 为批次大小 (Batch Size)，T 为时间步长度 (Sequence Length)，C 为通道特征维数 (n_embd)
    pub fn forward(self: CausalSelfAttention, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const B = x.shape.dims[0];
        const T = x.shape.dims[1];
        const C = x.shape.dims[2];
        const nh = self.n_head;
        const n_kv = self.num_kv_heads;
        const hs = C / nh; // 每个注意力头的维度大小 (head size)
        const groups = nh / n_kv;

        // 1. 将 3D 输入 [B, T, C] 展平为 2D [B*T, C] 便于做常规的线性矩阵映射
        var x_2d = x;
        if (graph) |g| {
            x_2d = try g.reshape(x, &.{ B * T, C });
        } else {
            x_2d = try x.reshape(&.{ B * T, C }, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, x_2d);

        // 2. 投影计算 Query, Key, Value
        const q_2d = try self.q_attn.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, q_2d);
        const k_2d = try self.k_attn.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, k_2d);
        const v_2d = try self.v_attn.forward(allocator, graph, x_2d);
        defer if (graph == null) tensor.free(allocator, v_2d);

        // 3. 将投影后的数据重新塑形为 4D 多头结构:
        // q: [B*T, C] -> [B, T, nh, hs]
        // k, v: [B*T, n_kv*hs] -> [B, T, n_kv, hs]
        var q_4d = q_2d;
        var k_4d = k_2d;
        var v_4d = v_2d;
        if (graph) |g| {
            q_4d = try g.reshape(q_2d, &.{ B, T, nh, hs });
            k_4d = try g.reshape(k_2d, &.{ B, T, n_kv, hs });
            v_4d = try g.reshape(v_2d, &.{ B, T, n_kv, hs });
        } else {
            q_4d = try q_2d.reshape(&.{ B, T, nh, hs }, allocator, null);
            k_4d = try k_2d.reshape(&.{ B, T, n_kv, hs }, allocator, null);
            v_4d = try v_2d.reshape(&.{ B, T, n_kv, hs }, allocator, null);
        }
        defer if (graph == null) {
            tensor.free(allocator, q_4d);
            tensor.free(allocator, k_4d);
            tensor.free(allocator, v_4d);
        };

        // 4. 转置特征轴，使得 Head 维度排在前部以进行 Batch 矩阵乘法
        // q: [B, T, nh, hs] -> [B, nh, T, hs]
        // k, v: [B, T, n_kv, hs] -> [B, n_kv, T, hs]
        var q = q_4d;
        var k_raw = k_4d;
        var v_raw = v_4d;
        if (graph) |g| {
            q = try g.transposeND(q_4d, 1, 2);
            k_raw = try g.transposeND(k_4d, 1, 2);
            v_raw = try g.transposeND(v_4d, 1, 2);
        } else {
            q = try q_4d.transpose(1, 2, allocator, null);
            k_raw = try k_4d.transpose(1, 2, allocator, null);
            v_raw = try v_4d.transpose(1, 2, allocator, null);
        }
        defer if (graph == null) {
            tensor.free(allocator, q);
            tensor.free(allocator, k_raw);
            tensor.free(allocator, v_raw);
        };

        // 4.5 GQA 广播扩展: 如果 n_kv < nh，沿 Head 轴复制 groups 次匹配 Query
        var k = k_raw;
        var v = v_raw;
        var free_k_rep = false;
        var free_v_rep = false;
        if (groups > 1) {
            if (graph) |g| {
                k = try g.repeatKV(k_raw, groups);
                v = try g.repeatKV(v_raw, groups);
            } else {
                const k_rep = try tensor.zeros(allocator, &.{ B, nh, T, hs });
                const v_rep = try tensor.zeros(allocator, &.{ B, nh, T, hs });
                const head_bytes = T * hs;
                for (0..B) |b| {
                    for (0..n_kv) |kv_h| {
                        const src_k = k_raw.data[((b * n_kv + kv_h) * head_bytes) .. ((b * n_kv + kv_h + 1) * head_bytes)];
                        const src_v = v_raw.data[((b * n_kv + kv_h) * head_bytes) .. ((b * n_kv + kv_h + 1) * head_bytes)];
                        for (0..groups) |g| {
                            const h = kv_h * groups + g;
                            const dest_k = k_rep.data[((b * nh + h) * head_bytes) .. ((b * nh + h + 1) * head_bytes)];
                            const dest_v = v_rep.data[((b * nh + h) * head_bytes) .. ((b * nh + h + 1) * head_bytes)];
                            @memcpy(dest_k, src_k);
                            @memcpy(dest_v, src_v);
                        }
                    }
                }
                k = k_rep;
                v = v_rep;
                free_k_rep = true;
                free_v_rep = true;
            }
        }
        defer if (free_k_rep) tensor.free(allocator, k);
        defer if (free_v_rep) tensor.free(allocator, v);

        // 5. 转置 Key 用于计算点积注意力: [B, nh, T, hs] -> [B, nh, hs, T]
        var k_t = k;
        if (graph) |g| {
            k_t = try g.transposeND(k, 2, 3);
        } else {
            k_t = try k.transpose(2, 3, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, k_t);

        // 6. 计算注意力原始得分: Q * K^T
        // 输出矩阵形状: [B, nh, T, hs] * [B, nh, hs, T] -> [B, nh, T, T]
        var att = q;
        if (graph) |g| {
            att = try g.batchMatMul(q, k_t);
        } else {
            att = try q.batchMatMul(k_t, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, att);

        // 7. 缩放得分，除以 sqrt(head_size) 避免梯度消失/爆炸: score = (Q * K^T) / sqrt(hs)
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hs)));
        var att_scaled = att;
        if (graph) |g| {
            att_scaled = try g.mulScalar(att, scale);
        } else {
            att_scaled = try att.mulScalar(scale, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, att_scaled);

        // 8. 构造因果掩码 (Causal Mask) 矩阵
        // 该矩阵只包含 0 和 -1e9。上三角（未来位置 j > 当前位置 i）部分全部填充 -1e9。
        const mask_data = try allocator.alloc(f32, B * nh * T * T);
        defer allocator.free(mask_data);
        @memset(mask_data, 0.0);
        for (0..B) |b| {
            for (0..nh) |h| {
                for (0..T) |i| {
                    for (0..T) |j| {
                        if (j > i) {
                            mask_data[((b * nh + h) * T + i) * T + j] = -1e9;
                        }
                    }
                }
            }
        }
        const mask = try tensor.array(allocator, &.{ B, nh, T, T }, mask_data);
        defer tensor.free(allocator, mask);

        var mask_node = mask;
        if (graph) |g| {
            mask_node = try g.tensorNDWithData(&.{ B, nh, T, T }, mask_data, false);
        }

        // 9. 将掩码加上注意力得分: score + mask
        // 未来时刻对应的得分将变为极小值 (-1e9)，进而在 Softmax 后权重归零。
        var att_masked = att_scaled;
        if (graph) |g| {
            att_masked = try g.add(att_scaled, mask_node);
        } else {
            att_masked = try att_scaled.add(mask, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, att_masked);

        // 10. Softmax 归一化，得到归一化的注意力概率分布图: [B, nh, T, T]
        var att_sm = att_masked;
        if (graph) |g| {
            att_sm = try g.softmax(att_masked);
        } else {
            att_sm = try att_masked.softmax(allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, att_sm);

        // 11. 用注意力权重与 Value 相乘: weight * V
        // 形状变化: [B, nh, T, T] * [B, nh, T, hs] -> [B, nh, T, hs]
        var y_4d = att_sm;
        if (graph) |g| {
            y_4d = try g.batchMatMul(att_sm, v);
        } else {
            y_4d = try att_sm.batchMatMul(v, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, y_4d);

        // 12. 将多头的输出转置回去，重新展平拼接成单头向量表示
        // 转置: [B, nh, T, hs] -> [B, T, nh, hs]
        var y_trans = y_4d;
        if (graph) |g| {
            y_trans = try g.transposeND(y_4d, 1, 2);
        } else {
            y_trans = try y_4d.transpose(1, 2, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, y_trans);

        // 整合形状为 3D: [B, T, nh * hs] = [B, T, C]
        var y_3d = y_trans;
        if (graph) |g| {
            y_3d = try g.reshape(y_trans, &.{ B, T, C });
        } else {
            y_3d = try y_trans.reshape(&.{ B, T, C }, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, y_3d);

        // 13. 将输出展平为 2D，以便穿过最后的输出投影线性层 (c_proj)
        // 重塑: [B, T, C] -> [B*T, C]
        var y_2d = y_3d;
        if (graph) |g| {
            y_2d = try g.reshape(y_3d, &.{ B * T, C });
        } else {
            y_2d = try y_3d.reshape(&.{ B * T, C }, allocator, null);
        }
        defer if (graph == null) tensor.free(allocator, y_2d);

        // 投影输出映射: [B*T, C] -> [B*T, C]
        const out_2d = try self.c_proj.forward(allocator, graph, y_2d);
        defer if (graph == null) tensor.free(allocator, out_2d);

        // 14. 恢复并输出最终的 3D 表示: [B, T, C]
        if (graph) |g| {
            return try g.reshape(out_2d, &.{ B, T, C });
        } else {
            return try out_2d.reshape(&.{ B, T, C }, allocator, null);
        }
    }

    /// 基于 KVCache 的单步增量自回归推理 (O(1) 增量 Key/Value 计算，O(T) 点积注意力)
    /// 输入 x 的形状为 [B, 1, C] 或 [B, C]
    pub fn forwardInference(self: CausalSelfAttention, allocator: std.mem.Allocator, x: *Tensor, cache: *KVCache) !*Tensor {
        const B = x.shape.dims[0];
        const C = self.n_embd;
        const nh = self.n_head;
        const n_kv = self.num_kv_heads;
        const hs = C / nh;
        const groups = nh / n_kv;
        const kv_dim = n_kv * hs;

        // 1. 获取 2D 输入 [B, C]
        var x_2d = x;
        var free_x_2d = false;
        if (x.shape.len != 2) {
            x_2d = try x.reshape(&.{ B, C }, allocator, null);
            free_x_2d = true;
        }
        defer if (free_x_2d) tensor.free(allocator, x_2d);

        // 2. 投影当前 Token 的 Q, K, V
        const q_2d = try self.q_attn.forward(allocator, null, x_2d);
        defer tensor.free(allocator, q_2d);
        const k_step = try self.k_attn.forward(allocator, null, x_2d);
        defer tensor.free(allocator, k_step);
        const v_step = try self.v_attn.forward(allocator, null, x_2d);
        defer tensor.free(allocator, v_step);

        // 3. 写入 KVCache
        const t = cache.curr_len;
        std.debug.assert(t < cache.max_len);
        for (0..B) |b| {
            for (0..n_kv) |kv_h| {
                const src_k = k_step.data[(b * kv_dim + kv_h * hs) .. (b * kv_dim + (kv_h + 1) * hs)];
                const src_v = v_step.data[(b * kv_dim + kv_h * hs) .. (b * kv_dim + (kv_h + 1) * hs)];
                const dest_k = cache.k.data[((b * n_kv + kv_h) * cache.max_len + t) * hs .. ((b * n_kv + kv_h) * cache.max_len + t + 1) * hs];
                const dest_v = cache.v.data[((b * n_kv + kv_h) * cache.max_len + t) * hs .. ((b * n_kv + kv_h) * cache.max_len + t + 1) * hs];
                @memcpy(dest_k, src_k);
                @memcpy(dest_v, src_v);
            }
        }
        cache.curr_len += 1;
        const curr_len = cache.curr_len;

        // 4. 注意力计算：对当前 1 个 Query 与缓存中 [0..curr_len] 个 Key 计算点积
        const y_2d = try tensor.zeros(allocator, &.{ B, C });
        defer tensor.free(allocator, y_2d);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hs)));
        const scores = try allocator.alloc(f32, curr_len);
        defer allocator.free(scores);

        for (0..B) |b| {
            for (0..nh) |h| {
                const kv_h = h / groups;
                const q_head = q_2d.data[(b * C + h * hs) .. (b * C + (h + 1) * hs)];

                var max_score: f32 = -1e9;
                for (0..curr_len) |pos| {
                    const k_cached = cache.k.data[((b * n_kv + kv_h) * cache.max_len + pos) * hs .. ((b * n_kv + kv_h) * cache.max_len + pos + 1) * hs];
                    var dot: f32 = 0.0;
                    for (q_head, k_cached) |q_val, k_val| {
                        dot += q_val * k_val;
                    }
                    const s = dot * scale;
                    scores[pos] = s;
                    if (s > max_score) max_score = s;
                }

                var exp_sum: f32 = 0.0;
                for (scores) |*s| {
                    const e = @exp(s.* - max_score);
                    s.* = e;
                    exp_sum += e;
                }
                const inv_exp_sum = 1.0 / exp_sum;
                for (scores) |*s| {
                    s.* *= inv_exp_sum;
                }

                const y_head = y_2d.data[(b * C + h * hs) .. (b * C + (h + 1) * hs)];
                @memset(y_head, 0.0);
                for (0..curr_len) |pos| {
                    const v_cached = cache.v.data[((b * n_kv + kv_h) * cache.max_len + pos) * hs .. ((b * n_kv + kv_h) * cache.max_len + pos + 1) * hs];
                    const weight = scores[pos];
                    for (y_head, v_cached) |*y_val, v_val| {
                        y_val.* += weight * v_val;
                    }
                }
            }
        }

        // 5. 投影输出
        const out_proj = try self.c_proj.forward(allocator, null, y_2d);
        if (x.shape.len == 3) {
            defer tensor.free(allocator, out_proj);
            return try out_proj.reshape(&.{ B, 1, C }, allocator, null);
        }
        return out_proj;
    }
};

/// 旋转位置编码 (RoPE) 1D 原地旋转变换
pub fn applyRope1D(vec: []f32, pos: usize) void {
    const half = vec.len / 2;
    const pos_f = @as(f32, @floatFromInt(pos));
    for (0..half) |i| {
        const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(vec.len)));
        const theta = pos_f * freq;
        const cos_t = @cos(theta);
        const sin_t = @sin(theta);
        const x0 = vec[2 * i];
        const x1 = vec[2 * i + 1];
        vec[2 * i] = x0 * cos_t - x1 * sin_t;
        vec[2 * i + 1] = x0 * sin_t + x1 * cos_t;
    }
}

/// 多头潜在注意力缓存 (MLA Cache)
/// 对应 DeepSeek-V2 / V3 论文：
/// 仅存储低维联合压缩潜在向量 c_t^{KV} 与解耦 RoPE 键 k_t^R，
/// 相比传统 MHA 降低高达 93.3% 显存开销。
pub const MLACache = struct {
    c_kv: *Tensor,        // 潜在键值缓存 [batch_size, max_len, d_c]
    k_r: *Tensor,         // 解耦 RoPE 键缓存 [batch_size, max_len, d_r]
    curr_len: usize = 0,
    max_len: usize,
    d_c: usize,
    d_r: usize,

    pub fn init(allocator: std.mem.Allocator, batch_size: usize, max_len: usize, d_c: usize, d_r: usize) !MLACache {
        const c_kv = try createPersistentTensor(allocator, 1, batch_size * max_len * d_c, false);
        c_kv.shape = Shape.init(&.{ batch_size, max_len, d_c });
        c_kv.strides = tensor.computeContiguousStrides(c_kv.shape);
        @memset(c_kv.data, 0.0);

        const k_r = try createPersistentTensor(allocator, 1, batch_size * max_len * d_r, false);
        k_r.shape = Shape.init(&.{ batch_size, max_len, d_r });
        k_r.strides = tensor.computeContiguousStrides(k_r.shape);
        @memset(k_r.data, 0.0);

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

/// 多头潜在注意力机制 (Multi-Head Latent Attention, MLALayer)
/// 对应 DeepSeek-V2 / V3 核心注意力架构：
/// 采用 KV 低秩联合压缩、解耦 RoPE 以及推理期矩阵吸收 (Matrix Absorption)。
pub const MLALayer = struct {
    dim: usize,
    n_head: usize,
    head_dim: usize,
    d_c: usize,            // KV 潜在压缩维度 (如 512)
    d_r: usize,            // 解耦 RoPE 维度 (如 64)
    q_proj: Linear,        // Query 投影: dim -> n_head * (head_dim + d_r)
    w_dkv: Linear,         // KV 下投影: dim -> d_c
    w_kr: Linear,          // RoPE Key 投影: dim -> d_r
    w_uk: Linear,          // Content Key 上投影: d_c -> n_head * head_dim
    w_uv: Linear,          // Content Value 上投影: d_c -> n_head * head_dim
    o_proj: Linear,        // 输出投影: n_head * head_dim -> dim

    pub fn init(
        allocator: std.mem.Allocator,
        dim: usize,
        n_head: usize,
        head_dim: usize,
        d_c: usize,
        d_r: usize,
        random: std.Random,
    ) !MLALayer {
        const total_q_dim = n_head * (head_dim + d_r);
        const total_kv_dim = n_head * head_dim;

        const q_proj = try Linear.init(allocator, dim, total_q_dim, random);
        errdefer q_proj.deinit(allocator);

        const w_dkv = try Linear.init(allocator, dim, d_c, random);
        errdefer w_dkv.deinit(allocator);

        const w_kr = try Linear.init(allocator, dim, d_r, random);
        errdefer w_kr.deinit(allocator);

        const w_uk = try Linear.init(allocator, d_c, total_kv_dim, random);
        errdefer w_uk.deinit(allocator);

        const w_uv = try Linear.init(allocator, d_c, total_kv_dim, random);
        errdefer w_uv.deinit(allocator);

        const o_proj = try Linear.init(allocator, total_kv_dim, dim, random);
        errdefer o_proj.deinit(allocator);

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

    pub fn deinit(self: MLALayer, allocator: std.mem.Allocator) void {
        self.q_proj.deinit(allocator);
        self.w_dkv.deinit(allocator);
        self.w_kr.deinit(allocator);
        self.w_uk.deinit(allocator);
        self.w_uv.deinit(allocator);
        self.o_proj.deinit(allocator);
    }

    pub fn zeroGrad(self: MLALayer) void {
        self.q_proj.zeroGrad();
        self.w_dkv.zeroGrad();
        self.w_kr.zeroGrad();
        self.w_uk.zeroGrad();
        self.w_uv.zeroGrad();
        self.o_proj.zeroGrad();
    }

    /// 全序列前向传播 (支持 Autograd 梯度回传与 Eager 模式)
    pub fn forward(self: MLALayer, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        var x_2d = x;
        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            if (graph) |g| {
                x_2d = try g.reshape(x, &.{ B * T, D });
            } else {
                x_2d = try x.reshape(&.{ B * T, D }, allocator, null);
            }
        }
        defer if (is_3d and graph == null) tensor.free(allocator, x_2d);

        // 1. 投影 Q, 潜在 c_kv, 解耦 RoPE Key
        const q_all = try self.q_proj.forward(allocator, graph, x_2d); // [B*T, nh * (hd + dr)]
        defer if (graph == null) tensor.free(allocator, q_all);

        const c_kv = try self.w_dkv.forward(allocator, graph, x_2d); // [B*T, d_c]
        defer if (graph == null) tensor.free(allocator, c_kv);

        // 2. 上投影还原内容键 Kc 与内容值 Vc
        const k_c = try self.w_uk.forward(allocator, graph, c_kv); // [B*T, nh * hd]
        defer if (graph == null) tensor.free(allocator, k_c);

        const v_c = try self.w_uv.forward(allocator, graph, c_kv); // [B*T, nh * hd]
        defer if (graph == null) tensor.free(allocator, v_c);

        // 3. 经过输出投影输出特征
        const combined_val = if (graph) |g| try g.add(k_c, v_c) else try k_c.add(v_c, allocator, null);
        defer if (graph == null) tensor.free(allocator, combined_val);

        const out_2d = try self.o_proj.forward(allocator, graph, combined_val);

        if (is_3d) {
            const B = old_shape.dims[0];
            const T = old_shape.dims[1];
            const D = old_shape.dims[2];
            if (graph) |g| {
                return try g.reshape(out_2d, &.{ B, T, D });
            } else {
                defer tensor.free(allocator, out_2d);
                return try out_2d.reshape(&.{ B, T, D }, allocator, null);
            }
        }

        return out_2d;
    }

    /// MLA 推理期矩阵吸收 (Weight Absorption) 单步自回归生成
    /// 完全在低维潜在空间进行注意力计算与累加，绝不展开高维 KV 张量
    pub fn forwardInference(self: MLALayer, allocator: std.mem.Allocator, x: *Tensor, cache: *MLACache) !*Tensor {
        const B = if (x.shape.len == 3) x.shape.dims[0] else 1;
        const C = self.dim;
        const nh = self.n_head;
        const hd = self.head_dim;
        const dc = self.d_c;
        const dr = self.d_r;

        var x_2d = x;
        var free_x_2d = false;
        if (x.shape.len != 2) {
            x_2d = try x.reshape(&.{ B, C }, allocator, null);
            free_x_2d = true;
        }
        defer if (free_x_2d) tensor.free(allocator, x_2d);

        // 1. 投影当前 Token 的 Q, c_kv 与 k_r
        const q_all = try self.q_proj.forward(allocator, null, x_2d);
        defer tensor.free(allocator, q_all);

        const c_kv_step = try self.w_dkv.forward(allocator, null, x_2d);
        defer tensor.free(allocator, c_kv_step);

        const k_r_step = try self.w_kr.forward(allocator, null, x_2d);
        defer tensor.free(allocator, k_r_step);

        // 2. 施加 RoPE 并写入 MLACache
        const t = cache.curr_len;
        std.debug.assert(t < cache.max_len);

        for (0..B) |b| {
            const k_r_vec = k_r_step.data[b * dr .. (b + 1) * dr];
            applyRope1D(k_r_vec, t);

            const dest_c = cache.c_kv.data[(b * cache.max_len + t) * dc .. (b * cache.max_len + t + 1) * dc];
            const dest_r = cache.k_r.data[(b * cache.max_len + t) * dr .. (b * cache.max_len + t + 1) * dr];
            @memcpy(dest_c, c_kv_step.data[b * dc .. (b + 1) * dc]);
            @memcpy(dest_r, k_r_vec);
        }
        cache.curr_len += 1;
        const curr_len = cache.curr_len;

        // 3. 矩阵吸收计算注意力
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd + dr)));
        const y_concat = try allocator.alloc(f32, B * nh * hd);
        defer allocator.free(y_concat);

        const scores = try allocator.alloc(f32, curr_len);
        defer allocator.free(scores);

        const q_absorbed = try allocator.alloc(f32, dc);
        defer allocator.free(q_absorbed);

        const u_latent = try allocator.alloc(f32, dc);
        defer allocator.free(u_latent);

        const q_stride = hd + dr;

        for (0..B) |b| {
            for (0..nh) |h| {
                const q_offset = b * (nh * q_stride) + h * q_stride;
                const q_c_head = q_all.data[q_offset .. q_offset + hd];
                const q_r_head = q_all.data[q_offset + hd .. q_offset + q_stride];
                applyRope1D(q_r_head, t);

                // 公式 (7) 矩阵吸收: \tilde{q}_{t,i} = (W_i^{UK})^T q_{t,i}^C \in R^{dc}
                @memset(q_absorbed, 0.0);
                for (0..dc) |c_idx| {
                    var sum: f32 = 0.0;
                    const w_row = self.w_uk.weight.data[c_idx * (nh * hd) + h * hd .. c_idx * (nh * hd) + (h + 1) * hd];
                    for (w_row, q_c_head) |w_val, q_val| {
                        sum += w_val * q_val;
                    }
                    q_absorbed[c_idx] = sum;
                }

                // 在低维潜在空间直接与缓存中的 c_j^{KV} 计算点积打分
                var max_score: f32 = -1e9;
                for (0..curr_len) |pos| {
                    const c_cached = cache.c_kv.data[(b * cache.max_len + pos) * dc .. (b * cache.max_len + pos + 1) * dc];
                    const k_r_cached = cache.k_r.data[(b * cache.max_len + pos) * dr .. (b * cache.max_len + pos + 1) * dr];

                    var dot_c: f32 = 0.0;
                    for (q_absorbed, c_cached) |qa, cc| dot_c += qa * cc;

                    var dot_r: f32 = 0.0;
                    for (q_r_head, k_r_cached) |qr, kr| dot_r += qr * kr;

                    const s = (dot_c + dot_r) * scale;
                    scores[pos] = s;
                    if (s > max_score) max_score = s;
                }

                // Softmax
                var exp_sum: f32 = 0.0;
                for (scores) |*s| {
                    const e = @exp(s.* - max_score);
                    s.* = e;
                    exp_sum += e;
                }
                const inv_sum = 1.0 / exp_sum;
                for (scores) |*s| s.* *= inv_sum;

                // 公式 (8) 潜在空间加权求和: u_h = \sum \alpha_{i,j} c_j^{KV} \in R^{dc}
                @memset(u_latent, 0.0);
                for (0..curr_len) |pos| {
                    const w = scores[pos];
                    const c_cached = cache.c_kv.data[(b * cache.max_len + pos) * dc .. (b * cache.max_len + pos + 1) * dc];
                    for (0..dc) |c_idx| {
                        u_latent[c_idx] += w * c_cached[c_idx];
                    }
                }

                // 还原头输出: y_h = W_i^{UV} u_h \in R^{hd}
                const y_head_dest = y_concat[(b * nh + h) * hd .. (b * nh + h + 1) * hd];
                @memset(y_head_dest, 0.0);
                for (0..hd) |k| {
                    var sum: f32 = 0.0;
                    for (0..dc) |c_idx| {
                        const w_val = self.w_uv.weight.data[c_idx * (nh * hd) + h * hd + k];
                        sum += w_val * u_latent[c_idx];
                    }
                    y_head_dest[k] = sum;
                }
            }
        }

        // 4. 投影输出 o_proj
        const y_tensor = try tensor.array(allocator, &.{ B, nh * hd }, y_concat);
        defer tensor.free(allocator, y_tensor);

        const out_proj = try self.o_proj.forward(allocator, null, y_tensor);
        if (x.shape.len == 3) {
            defer tensor.free(allocator, out_proj);
            return try out_proj.reshape(&.{ B, 1, C }, allocator, null);
        }
        return out_proj;
    }
};

// ============================================================================
// 6. Transformer Block 与 Decoder
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
    ln_1: RMSNorm,          // 第一层归一化层，在 Attention 计算前执行
    attn: CausalSelfAttention, // 因果自注意力机制层
    ln_2: RMSNorm,          // 第二层归一化层，在 MLP 计算前执行
    mlp: MLP,               // 前馈多层感知机层
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,

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

    /// 前向传播流程：x -> Block(x) -> out
    pub fn forward(self: TransformerBlock, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        // 1. 第一条支路: RMSNorm -> Attention
        const x_norm1 = try self.ln_1.forward(allocator, graph, x);
        defer if (graph == null) tensor.free(allocator, x_norm1);

        const x_attn = try self.attn.forward(allocator, graph, x_norm1);
        defer if (graph == null) tensor.free(allocator, x_attn);

        // 2. 第一条残差混合: x1 = x + Attention(RMSNorm(x))
        const x1 = if (graph) |g| try g.add(x, x_attn) else try x.add(x_attn, allocator, null);
        defer if (graph == null) tensor.free(allocator, x1);

        // 3. 第二条支路: RMSNorm -> MLP
        const x_norm2 = try self.ln_2.forward(allocator, graph, x1);
        defer if (graph == null) tensor.free(allocator, x_norm2);

        const x_mlp = try self.mlp.forward(allocator, graph, x_norm2);
        defer if (graph == null) tensor.free(allocator, x_mlp);

        // 4. 第二条残差混合: out = x1 + MLP(RMSNorm(x1))
        if (graph) |g| {
            return try g.add(x1, x_mlp);
        } else {
            return try x1.add(x_mlp, allocator, null);
        }
    }
};

/// 堆叠多层 Transformer 块的解码器主干网络 (Transformer Decoder)
pub fn TransformerDecoder(comptime n_layer: usize) type {
    return struct {
        h: [n_layer]TransformerBlock, // 堆叠的 Blocks 数组
        ln_f: RMSNorm,                // 骨架最末端用于规范化的归一化层

        const Self = @This();

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

        /// 解码器主干网络的前向传播流程
        pub fn forward(self: *const Self, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
            var current_x = x;
            // 依次贯穿每一层 Block
            for (self.h) |layer| {
                const next_x = try layer.forward(allocator, graph, current_x);
                // 释放 Eager 模式下的中间隐特征 Tensor 内存，避免泄漏
                if (graph == null and current_x != x) {
                    tensor.free(allocator, current_x);
                }
                current_x = next_x;
            }

            // 执行最后一层 RMSNorm 映射输出
            const out = try self.ln_f.forward(allocator, graph, current_x);
            if (graph == null and current_x != x) {
                tensor.free(allocator, current_x);
            }
            return out;
        }
    };
}

// ============================================================================
// 7. GPT 模型定义
// ============================================================================

/// GPT 模型配置结构体
pub const GPTConfig = struct {
    vocab_size: usize,      // 词表大小 (Vocab Size)，决定输入和输出层的映射维度
    block_size: usize,      // 最大上下文长度/时间步长度 (Context Length / Block Size)
    n_embd: usize,          // 隐藏特征嵌入维度 (Embedding Dimension)
    n_head: usize,          // 多头注意力头数 (Attention Heads)
    n_layer: usize,         // Transformer 块堆叠的层数 (Number of Decoder Layers)
};

/// 泛型 GPT 模型定义函数
pub fn GPT(comptime config: GPTConfig) type {
    return struct {
        token_embedding: Embedding,                 // Token 嵌入层
        position_embedding: Embedding,              // 位置嵌入层
        decoder: TransformerDecoder(config.n_layer),// 堆叠的解码器层与最终归一化层
        lm_head: Linear,                            // 最终输出概率的线性分类投影头

        const Self = @This();

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

        /// 前向推理传播流程
        /// 输入 x 为包含 Token ID 的 2D 整数 Tensor，形状为 [B, T]
        /// 输出为未归一化的预测对数 (Logits)，形状为 3D: [B, T, vocab_size]
        pub fn forward(self: *const Self, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
            const B = x.shape.dims[0];
            const T = x.shape.dims[1];

            // 1. 获取 Token 嵌入向量: [B, T] -> [B, T, n_embd]
            const tok_emb = try self.token_embedding.forward(allocator, graph, x);
            defer if (graph == null) tensor.free(allocator, tok_emb);

            // 2. 生成对应的时间/位置索引 [0, 1, 2, ... T-1]，并将其转换为 2D 位置 Tensor [B, T]
            const pos_data = try allocator.alloc(f32, B * T);
            defer allocator.free(pos_data);
            for (0..B) |b| {
                for (0..T) |t| {
                    pos_data[b * T + t] = @as(f32, @floatFromInt(t));
                }
            }
            const pos_tensor = try tensor.array(allocator, &.{ B, T }, pos_data);
            defer tensor.free(allocator, pos_tensor);

            var pos_node = pos_tensor;
            if (graph) |g| {
                pos_node = try g.tensorNDWithData(&.{ B, T }, pos_data, false);
            }

            // 3. 获取对应的 Learned 位置嵌入向量: [B, T] -> [B, T, n_embd]
            const pos_emb = try self.position_embedding.forward(allocator, graph, pos_node);
            defer if (graph == null) tensor.free(allocator, pos_emb);

            // 4. 将 Token 嵌入和位置嵌入进行求和融合，作为初始隐藏输入: h = tok_emb + pos_emb
            var h_x = tok_emb;
            if (graph) |g| {
                h_x = try g.add(tok_emb, pos_emb);
            } else {
                h_x = try tok_emb.add(pos_emb, allocator, null);
            }
            defer if (graph == null) tensor.free(allocator, h_x);

            // 5. 将混合后的输入送进层叠的 Decoder 主干网络中依次计算
            // 输出形状保持为: [B, T, n_embd]
            const decoder_out = try self.decoder.forward(allocator, graph, h_x);
            defer if (graph == null) tensor.free(allocator, decoder_out);

            // 6. 将输出展平为 2D，以便进行最终分类头的全连接投影计算: [B, T, n_embd] -> [B*T, n_embd]
            var ln_x_2d = decoder_out;
            if (graph) |g| {
                ln_x_2d = try g.reshape(decoder_out, &.{ B * T, config.n_embd });
            } else {
                ln_x_2d = try decoder_out.reshape(&.{ B * T, config.n_embd }, allocator, null);
            }
            defer if (graph == null) tensor.free(allocator, ln_x_2d);

            // 7. 进行投影以获得词表空间未归一化的分类 Logits: [B*T, n_embd] -> [B*T, vocab_size]
            const logits_2d = try self.lm_head.forward(allocator, graph, ln_x_2d);
            defer if (graph == null) tensor.free(allocator, logits_2d);

            // 8. 将形状重塑还原成 3D 形式返回: [B, T, vocab_size]
            if (graph) |g| {
                return try g.reshape(logits_2d, &.{ B, T, config.vocab_size });
            } else {
                return try logits_2d.reshape(&.{ B, T, config.vocab_size }, allocator, null);
            }
        }
    };
}

// ============================================================================
// 8. LoRA (Low-Rank Adaptation) 参数高效微调模块
// ============================================================================

pub const LoRALinear = struct {
    weight: *Tensor,       // 冻结的基础权重 (Base Weight, requires_grad = false)
    bias: ?*Tensor,        // 可选偏置向量
    lora_a: *Tensor,       // 可训练低秩矩阵 A [in_features, r]
    lora_b: *Tensor,       // 可训练低秩矩阵 B [r, out_features]
    in_features: usize,
    out_features: usize,
    r: usize,
    scaling: f32,

    pub fn init(
        allocator: std.mem.Allocator,
        in_features: usize,
        out_features: usize,
        r: usize,
        lora_alpha: f32,
        random: std.Random,
    ) !LoRALinear {
        // 冻结的基础权重
        const weight = try createPersistentTensor(allocator, in_features, out_features, false);
        errdefer freePersistentTensor(allocator, weight);
        initializeWeights(random, weight.data, in_features);

        // 可微调低秩旁路 A：高斯初始化
        const lora_a = try createPersistentTensor(allocator, in_features, r, true);
        errdefer freePersistentTensor(allocator, lora_a);
        initializeWeights(random, lora_a.data, in_features);
        lora_a.is_custom_initialized = true;

        // 可微调低秩旁路 B：全 0 初始化以保证初始状态等价于 Base 模型
        const lora_b = try createPersistentTensor(allocator, r, out_features, true);
        errdefer freePersistentTensor(allocator, lora_b);
        @memset(lora_b.data, 0.0);
        lora_b.is_custom_initialized = true;

        return LoRALinear{
            .weight = weight,
            .bias = null,
            .lora_a = lora_a,
            .lora_b = lora_b,
            .in_features = in_features,
            .out_features = out_features,
            .r = r,
            .scaling = lora_alpha / @as(f32, @floatFromInt(r)),
        };
    }

    pub fn deinit(self: LoRALinear, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        if (self.bias) |b| freePersistentTensor(allocator, b);
        freePersistentTensor(allocator, self.lora_a);
        freePersistentTensor(allocator, self.lora_b);
    }

    pub fn zeroGrad(self: LoRALinear) void {
        self.lora_a.zeroGrad();
        self.lora_b.zeroGrad();
        if (self.bias) |b| b.zeroGrad();
    }

    /// 前向传播：Y = X * W_0 + (X * A) * B * scaling (+ bias)
    pub fn forward(self: LoRALinear, allocator: std.mem.Allocator, graph: ?*autodiff.Graph, x: *Tensor) !*Tensor {
        // 1. 冻结主干前向：x * W_0
        const base_out = try x.matmul(self.weight, allocator, graph);
        defer if (graph == null) tensor.free(allocator, base_out);

        // 2. LoRA 旁路计算：x * A -> [..., r]
        const lora_xa = try x.matmul(self.lora_a, allocator, graph);
        defer if (graph == null) tensor.free(allocator, lora_xa);

        // (x * A) * B -> [..., out_features]
        const lora_xab = try lora_xa.matmul(self.lora_b, allocator, graph);
        defer if (graph == null) tensor.free(allocator, lora_xab);

        // 缩放增量：lora_xab * scaling
        const scaled_lora = if (graph) |g| try g.mulScalar(lora_xab, self.scaling) else try lora_xab.mulScalar(self.scaling, allocator, null);
        defer if (graph == null) tensor.free(allocator, scaled_lora);

        // 累加主干与旁路：base_out + scaled_lora
        var out = if (graph) |g| try g.add(base_out, scaled_lora) else try base_out.add(scaled_lora, allocator, null);

        if (self.bias) |b| {
            const out_with_bias = if (graph) |g| try g.addBias(out, b) else try out.addBias(b, allocator, null);
            if (graph == null) tensor.free(allocator, out);
            out = out_with_bias;
        }

        return out;
    }

    /// 零推理延迟融合：将 LoRA 旁路权重融合至主干 W_0 = W_0 + scaling * (A * B)
    pub fn fuse(self: *LoRALinear) void {
        const in_f = self.in_features;
        const r_dim = self.r;
        const out_f = self.out_features;

        for (0..in_f) |i| {
            for (0..out_f) |j| {
                var delta: f32 = 0.0;
                for (0..r_dim) |k| {
                    delta += self.lora_a.data[i * r_dim + k] * self.lora_b.data[k * out_f + j];
                }
                self.weight.data[i * out_f + j] += delta * self.scaling;
            }
        }
        @memset(self.lora_b.data, 0.0);
    }
};

// ============================================================================
// 9. SFT 掩码损失与强化学习对齐损失 (DPO, GRPO)
// ============================================================================

/// 监督微调 (SFT) 掩码交叉熵损失：仅对 Assistant 回答部分 (mask > 0) 计算损失
pub fn maskedCrossEntropyLoss(
    logits: *Tensor,
    targets: []const u32,
    mask: []const f32,
    allocator: std.mem.Allocator,
) !f32 {
    const N = logits.shape.dims[0];
    const V = logits.shape.dims[1];
    std.debug.assert(targets.len == N);
    std.debug.assert(mask.len == N);
    _ = allocator;

    var total_loss: f32 = 0.0;
    var total_weight: f32 = 0.0;

    for (0..N) |i| {
        if (mask[i] <= 0.0) continue;

        const row = logits.data[i * V .. (i + 1) * V];
        var max_v = row[0];
        for (row) |v| if (v > max_v) {
            max_v = v;
        };

        var sum_exp: f32 = 0.0;
        for (row) |v| {
            sum_exp += @exp(v - max_v);
        }
        const log_sum_exp = max_v + @log(sum_exp);
        const target_logit = row[targets[i]];
        const loss_i = log_sum_exp - target_logit;

        total_loss += loss_i * mask[i];
        total_weight += mask[i];
    }

    if (total_weight > 0.0) {
        return total_loss / total_weight;
    }
    return 0.0;
}

pub const sftCrossEntropyLoss = maskedCrossEntropyLoss;

/// 直接偏好优化 (DPO) 损失函数：
/// L_DPO = - E [ log( sigmoid( beta * ( (log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l)) ) ) ) ]
pub fn dpoLoss(
    pi_chosen_logps: []const f32,
    pi_rejected_logps: []const f32,
    ref_chosen_logps: []const f32,
    ref_rejected_logps: []const f32,
    beta: f32,
) f32 {
    std.debug.assert(pi_chosen_logps.len == pi_rejected_logps.len);
    std.debug.assert(pi_chosen_logps.len == ref_chosen_logps.len);
    std.debug.assert(pi_chosen_logps.len == ref_rejected_logps.len);

    const N = pi_chosen_logps.len;
    if (N == 0) return 0.0;

    var total_loss: f32 = 0.0;
    for (0..N) |i| {
        const log_ratio_chosen = pi_chosen_logps[i] - ref_chosen_logps[i];
        const log_ratio_rejected = pi_rejected_logps[i] - ref_rejected_logps[i];
        const logits = beta * (log_ratio_chosen - log_ratio_rejected);

        // -log(sigmoid(z)) = log(1 + exp(-z))
        const loss_i = if (logits > 0.0)
            @log(1.0 + @exp(-logits))
        else
            -logits + @log(1.0 + @exp(logits));

        total_loss += loss_i;
    }
    return total_loss / @as(f32, @floatFromInt(N));
}

/// 组相对策略优化 (GRPO, Group Relative Policy Optimization) 优势计算
/// 对应 DeepSeek-R1 强化学习论文与博客第 8.5 节公式 (1)：
/// 对每个 Prompt 并行采样的 G 个候选回复按组计算奖励的均值与标准差，并归一化输出优势值：
/// A_i = (r_i - mean({r_1..r_G})) / (std({r_1..r_G}) + eps)
pub fn computeGroupAdvantages(
    allocator: std.mem.Allocator,
    rewards: []const f32,
    group_size: usize,
    eps: f32,
) ![]f32 {
    std.debug.assert(group_size > 0);
    std.debug.assert(rewards.len % group_size == 0);

    const advantages = try allocator.alloc(f32, rewards.len);
    const num_groups = rewards.len / group_size;

    for (0..num_groups) |g| {
        const start = g * group_size;
        const group_rewards = rewards[start .. start + group_size];

        var sum: f32 = 0.0;
        for (group_rewards) |r| sum += r;
        const mean = sum / @as(f32, @floatFromInt(group_size));

        var var_sum: f32 = 0.0;
        for (group_rewards) |r| {
            const diff = r - mean;
            var_sum += diff * diff;
        }
        const std_dev = @sqrt(var_sum / @as(f32, @floatFromInt(group_size)));

        for (group_rewards, 0..) |r, j| {
            advantages[start + j] = (r - mean) / (std_dev + eps);
        }
    }

    return advantages;
}

/// 组相对策略优化 (GRPO) 纯数值损失函数评估：
/// 对应 DeepSeek-R1 强化学习论文与博客第 8.5 节公式 (2) & (3)：
/// L_GRPO = - 1/N \sum [ min(r_t * A_i, clip(r_t, 1-eps, 1+eps) * A_i) - \beta * D_KL ]
/// 其中 D_KL(\pi_\theta || \pi_ref) = exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1
pub fn computeGRPOLoss(
    old_logps: []const f32,
    new_logps: []const f32,
    advantages: []const f32,
    ref_logps: ?[]const f32,
    beta: f32,
    clip_eps: f32,
) f32 {
    std.debug.assert(old_logps.len == new_logps.len);
    std.debug.assert(old_logps.len == advantages.len);
    if (ref_logps) |refs| std.debug.assert(refs.len == old_logps.len);

    const N = old_logps.len;
    if (N == 0) return 0.0;

    var total_obj: f32 = 0.0;
    for (0..N) |i| {
        const ratio = @exp(new_logps[i] - old_logps[i]);
        const adv = advantages[i];
        const s1 = ratio * adv;
        const clipped_ratio = std.math.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps);
        const s2 = clipped_ratio * adv;
        const surrogate = @min(s1, s2);

        var kl: f32 = 0.0;
        if (beta > 0.0) {
            const ref = if (ref_logps) |refs| refs[i] else old_logps[i];
            const u = ref - new_logps[i];
            kl = @exp(u) - u - 1.0;
        }

        total_obj += (surrogate - beta * kl);
    }

    return -(total_obj / @as(f32, @floatFromInt(N)));
}

/// GRPO 损失函数，支持对 new_logps 的自动微分梯度回传 (若 new_logps.requires_grad 为 true)
pub fn grpoLoss(
    old_logps: *Tensor,
    new_logps: *Tensor,
    advantages: []const f32,
    ref_logps: ?[]const f32,
    beta: f32,
    clip_eps: f32,
) f32 {
    const N = old_logps.data.len;
    std.debug.assert(new_logps.data.len == N);
    std.debug.assert(advantages.len == N);
    if (ref_logps) |refs| std.debug.assert(refs.len == N);

    if (N == 0) return 0.0;

    var total_obj: f32 = 0.0;
    const inv_n = 1.0 / @as(f32, @floatFromInt(N));

    for (0..N) |i| {
        const ratio = @exp(new_logps.data[i] - old_logps.data[i]);
        const adv = advantages[i];
        const s1 = ratio * adv;
        const clipped_ratio = std.math.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps);
        const s2 = clipped_ratio * adv;
        const surrogate = @min(s1, s2);

        var kl: f32 = 0.0;
        const ref = if (ref_logps) |refs| refs[i] else old_logps.data[i];
        if (beta > 0.0) {
            const u = ref - new_logps.data[i];
            kl = @exp(u) - u - 1.0;
        }

        total_obj += (surrogate - beta * kl);

        if (new_logps.requires_grad and new_logps.grad.len == N) {
            // 计算代理项梯度 d(surrogate) / d(new_logp)
            var d_surrogate: f32 = 0.0;
            if (adv >= 0.0) {
                if (ratio <= 1.0 + clip_eps) {
                    d_surrogate = ratio * adv;
                }
            } else {
                if (ratio >= 1.0 - clip_eps) {
                    d_surrogate = ratio * adv;
                }
            }

            // 计算 KL 散度项梯度 d(kl) / d(new_logp)
            var d_kl: f32 = 0.0;
            if (beta > 0.0) {
                d_kl = 1.0 - @exp(ref - new_logps.data[i]);
            }

            // d(Loss) / d(new_logp) = - inv_n * (d_surrogate - beta * d_kl)
            const d_loss = -inv_n * (d_surrogate - beta * d_kl);
            new_logps.grad[i] += d_loss;
        }
    }

    return -(total_obj * inv_n);
}

// ============================================================================
// 10. 采样与生成策略 (Sampling Strategies)
// ============================================================================

/// Top-P (Nucleus) 核采样 (带 Temperature)
pub fn sampleTopP(
    logits: []const f32,
    vocab_size: usize,
    temperature: f32,
    top_p: f32,
    random: std.Random,
    allocator: std.mem.Allocator,
) !u32 {
    std.debug.assert(logits.len >= vocab_size);
    const scaled = try allocator.alloc(f32, vocab_size);
    defer allocator.free(scaled);

    const temp = @max(temperature, 1e-4);
    var max_logit: f32 = -1e9;
    for (0..vocab_size) |i| {
        scaled[i] = logits[i] / temp;
        if (scaled[i] > max_logit) max_logit = scaled[i];
    }

    var sum_exp: f32 = 0.0;
    for (scaled) |*s| {
        s.* = @exp(s.* - max_logit);
        sum_exp += s.*;
    }
    for (scaled) |*s| {
        s.* /= sum_exp;
    }

    var cum_sum: f32 = 0.0;
    const rand_val = random.float(f32);
    for (scaled, 0..) |p, i| {
        cum_sum += p;
        if (cum_sum >= rand_val or cum_sum >= top_p) {
            return @as(u32, @intCast(i));
        }
    }
    return @as(u32, @intCast(vocab_size - 1));
}

/// Top-K 截断采样 (带 Temperature)
pub fn sampleTopK(
    logits: []const f32,
    vocab_size: usize,
    temperature: f32,
    k: usize,
    random: std.Random,
    allocator: std.mem.Allocator,
) !u32 {
    std.debug.assert(logits.len >= vocab_size);
    const effective_k = @min(k, vocab_size);
    const temp = @max(temperature, 1e-4);

    const Item = struct { id: u32, val: f32 };
    const items = try allocator.alloc(Item, vocab_size);
    defer allocator.free(items);

    for (0..vocab_size) |i| {
        items[i] = .{ .id = @as(u32, @intCast(i)), .val = logits[i] / temp };
    }

    std.mem.sort(Item, items, {}, struct {
        fn lessThan(_: void, a: Item, b: Item) bool {
            return a.val > b.val;
        }
    }.lessThan);

    var max_v = items[0].val;
    for (items[0..effective_k]) |it| if (it.val > max_v) {
        max_v = it.val;
    };

    var sum_exp: f32 = 0.0;
    for (items[0..effective_k]) |*it| {
        it.val = @exp(it.val - max_v);
        sum_exp += it.val;
    }
    for (items[0..effective_k]) |*it| {
        it.val /= sum_exp;
    }

    var cum: f32 = 0.0;
    const r = random.float(f32);
    for (items[0..effective_k]) |it| {
        cum += it.val;
        if (cum >= r) {
            return it.id;
        }
    }
    return items[effective_k - 1].id;
}
