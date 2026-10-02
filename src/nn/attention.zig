const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");

const Tensor = tensor.Tensor;
const Shape = tensor.Shape;
const Linear = core.Linear;
const createPersistentTensor = core.createPersistentTensor;
const freePersistentTensor = core.freePersistentTensor;

// ============================================================================
// 1. 键值缓存 (Key-Value Cache)
// ============================================================================

/// 键值缓存 (Key-Value Cache) 用于大模型自回归增量推理 (O(T) 生成复杂度)
pub const KVCache = struct {
    k: *Tensor, // 缓存的 Key 张量 [batch_size, n_head, max_seq_len, head_dim]
    v: *Tensor, // 缓存的 Value 张量 [batch_size, n_head, max_seq_len, head_dim]
    curr_len: usize = 0, // 当前已缓存的 Token 步长
    max_len: usize, // 最大支持上下文长度

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
// 2. 注意力机制 (ScaledDotProductAttention, CausalSelfAttention & MLA)
// ============================================================================

/// 缩放点积注意力核心模块 (Scaled Dot-Product Attention)
pub const ScaledDotProductAttention = struct {
    module_type: []const u8 = "ScaledDotProductAttention",

    pub const formula = "\\text{AttentionCore}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V";
};

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
    q_attn: Linear, // Query 线性投影层
    k_attn: Linear, // Key 线性投影层
    v_attn: Linear, // Value 线性投影层
    c_proj: Linear, // 最终的多头输出融合与投影层 (c_proj)
    n_head: usize, // 注意力头数 (Query heads)
    n_embd: usize, // 嵌入维度 (n_embd)
    num_kv_heads: usize, // Key / Value 头数 (1 = MQA, < n_head = GQA, == n_head = MHA)
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "CausalSelfAttention",

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

    /// 模块标准数学变换公式
    pub const formula = "A = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V \\cdot W_o^T + b_o";
    pub const core_formula = ScaledDotProductAttention.formula;

    pub fn registerFormula(self: *const CausalSelfAttention, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
            try graph.registerModuleType(n, self.module_type);
            var buf: [128]u8 = undefined;
            if (std.fmt.bufPrint(&buf, "{s}.core", .{n})) |core_name| {
                try graph.setModuleFormula(core_name, core_formula);
                try graph.registerModuleType(core_name, "ScaledDotProductAttention");
            } else |_| {}
        }
    }

    /// 前向注意力计算流程
    /// 输入 x 的形状必须为 3D: [B, T, C]
    /// 其中 B 为批次大小 (Batch Size)，T 为时间步长度 (Sequence Length)，C 为通道特征维数 (n_embd)
    pub fn forward(self: CausalSelfAttention, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        try self.registerFormula(graph);

        const B = x.shape.dims[0];
        const T = x.shape.dims[1];
        const C = x.shape.dims[2];
        const nh = self.n_head;
        const n_kv = self.num_kv_heads;
        const hs = C / nh; // 每个注意力头的维度大小 (head size)
        const groups = nh / n_kv;

        // 1. 将 3D 输入 [B, T, C] 展平为 2D [B*T, C] 便于做常规的线性矩阵映射
        const x_2d = try graph.reshape(x, &.{ B * T, C });

        // 2. 投影计算 Query, Key, Value
        const q_2d = try self.q_attn.forward(graph, x_2d);
        const k_2d = try self.k_attn.forward(graph, x_2d);
        const v_2d = try self.v_attn.forward(graph, x_2d);

        // 3. 将投影后的数据重新塑形为 4D 多头结构:
        // q: [B*T, C] -> [B, T, nh, hs]
        // k, v: [B*T, n_kv*hs] -> [B, T, n_kv, hs]
        const q_4d = try graph.reshape(q_2d, &.{ B, T, nh, hs });
        const k_4d = try graph.reshape(k_2d, &.{ B, T, n_kv, hs });
        const v_4d = try graph.reshape(v_2d, &.{ B, T, n_kv, hs });

        // 4. 转置特征轴，使得 Head 维度排在前部以进行 Batch 矩阵乘法
        // q: [B, T, nh, hs] -> [B, nh, T, hs]
        // k, v: [B, T, n_kv, hs] -> [B, n_kv, T, hs]
        const q = try graph.transposeND(q_4d, 1, 2);
        const k_raw = try graph.transposeND(k_4d, 1, 2);
        const v_raw = try graph.transposeND(v_4d, 1, 2);

        // 4.5 GQA 广播扩展: 如果 n_kv < nh，沿 Head 轴复制 groups 次匹配 Query
        var k = k_raw;
        var v = v_raw;
        if (groups > 1) {
            k = try graph.repeatKV(k_raw, groups);
            v = try graph.repeatKV(v_raw, groups);
        }

        // 注意力核心 (ScaledDotProductAttention) 子作用域: K^T -> QK^T -> 缩放 -> 掩码 -> softmax -> ·V
        // head 切分与合并属于 CausalSelfAttention 本身，不计入核心子作用域
        const core_scope = try graph.enterChildScope("core", "ScaledDotProductAttention");
        var core_scope_open = true;
        defer if (core_scope_open) core_scope.exit();

        // 5. 转置 Key 用于计算点积注意力: [B, nh, T, hs] -> [B, nh, hs, T]
        const k_t = try graph.transposeND(k, 2, 3);

        // 6. 计算注意力原始得分: Q * K^T
        // 输出矩阵形状: [B, nh, T, hs] * [B, nh, hs, T] -> [B, nh, T, T]
        const att = try graph.batchMatMul(q, k_t);

        // 7. 缩放得分，除以 sqrt(head_size) 避免梯度消失/爆炸: score = (Q * K^T) / sqrt(hs)
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hs)));
        const att_scaled = try graph.mulScalar(att, scale);

        // 8. 构造因果掩码 (Causal Mask) 矩阵 [1, 1, T, T] (通过多维广播作用于 [B, nh, T, T])
        // 该矩阵只包含 0 和 -1e9。上三角（未来位置 j > 当前位置 i）部分全部填充 -1e9。
        const mask_node = try graph.tensorND(&.{ 1, 1, T, T }, false);
        mask_node.is_buffer = true;
        for (0..T) |i| {
            for (i + 1..T) |j| {
                mask_node.data[i * T + j] = -1e9;
            }
        }

        // 9. 将掩码加上注意力得分: score + mask
        // 未来时刻对应的得分将变为极小值 (-1e9)，进而在 Softmax 后权重归零。
        if (self.name) |mod_name| {
            mask_node.setNameFormatted("{s}.causal_mask", .{mod_name});
        }
        const att_masked = try graph.add(att_scaled, mask_node);

        // 10. Softmax 归一化，得到归一化的注意力概率分布图: [B, nh, T, T]
        const att_sm = try graph.softmax(att_masked);

        // 11. 用注意力权重与 Value 相乘: weight * V
        // 形状变化: [B, nh, T, T] * [B, nh, T, hs] -> [B, nh, T, hs]
        const y_4d = try graph.batchMatMul(att_sm, v);

        core_scope.exit();
        core_scope_open = false;

        // 12. 将多头的输出转置回去，重新展平拼接成单头向量表示
        // 转置: [B, nh, T, hs] -> [B, T, nh, hs]
        const y_trans = try graph.transposeND(y_4d, 1, 2);

        // 整合形状为 3D: [B, T, nh * hs] = [B, T, C]
        const y_3d = try graph.reshape(y_trans, &.{ B, T, C });

        // 13. 将输出展平为 2D，以便穿过最后的输出投影线性层 (c_proj)
        // 重塑: [B, T, C] -> [B*T, C]
        const y_2d = try graph.reshape(y_3d, &.{ B * T, C });

        // 投影输出映射: [B*T, C] -> [B*T, C]
        const out_2d = try self.c_proj.forward(graph, y_2d);

        // 14. 恢复并输出最终的 3D 表示: [B, T, C]
        return try graph.reshape(out_2d, &.{ B, T, C });
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

        // 单步推理使用局部无梯度计算图承载中间张量，函数返回时整体释放
        var step_g = autodiff.Graph.initNoGrad(allocator);
        defer step_g.deinit();

        // 1. 获取 2D 输入 [B, C]
        const x_2d = if (x.shape.len != 2) try step_g.reshape(x, &.{ B, C }) else x;

        // 2. 投影当前 Token 的 Q, K, V
        const q_2d = try self.q_attn.forward(&step_g, x_2d);
        const k_step = try self.k_attn.forward(&step_g, x_2d);
        const v_step = try self.v_attn.forward(&step_g, x_2d);

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
        const y_2d = try step_g.zeros(&.{ B, C }, false);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hs)));
        const scores = try step_g.arenaAllocator().alloc(f32, curr_len);

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

        // 5. 投影输出，并拷贝到调用方分配器上 (局部计算图随函数返回释放)
        const out_proj = try self.c_proj.forward(&step_g, y_2d);
        const out_shape: []const usize = if (x.shape.len == 3) &.{ B, 1, C } else &.{ B, C };
        return try tensor.array(allocator, out_shape, out_proj.data);
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
    c_kv: *Tensor, // 潜在键值缓存 [batch_size, max_len, d_c]
    k_r: *Tensor, // 解耦 RoPE 键缓存 [batch_size, max_len, d_r]
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
    d_c: usize, // KV 潜在压缩维度 (如 512)
    d_r: usize, // 解耦 RoPE 维度 (如 64)
    q_proj: Linear, // Query 投影: dim -> n_head * (head_dim + d_r)
    w_dkv: Linear, // KV 下投影: dim -> d_c
    w_kr: Linear, // RoPE Key 投影: dim -> d_r
    w_uk: Linear, // Content Key 上投影: d_c -> n_head * head_dim
    w_uv: Linear, // Content Value 上投影: d_c -> n_head * head_dim
    o_proj: Linear, // 输出投影: n_head * head_dim -> dim
    name: ?[]const u8 = null,
    name_buf: [64]u8 = undefined,
    module_type: []const u8 = "MLALayer",

    pub const formula = "c_t^{KV} = x_t W^{DKV}, \\quad y = \\text{MLA}(Q, c^{KV}, k^R) W^O";

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

    pub fn setName(self: *MLALayer, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.q_proj.setNameFormatted("{s}.q_proj", .{self.name.?});
        self.w_dkv.setNameFormatted("{s}.w_dkv", .{self.name.?});
        self.w_kr.setNameFormatted("{s}.w_kr", .{self.name.?});
        self.w_uk.setNameFormatted("{s}.w_uk", .{self.name.?});
        self.w_uv.setNameFormatted("{s}.w_uv", .{self.name.?});
        self.o_proj.setNameFormatted("{s}.o_proj", .{self.name.?});
    }

    pub fn setNameFormatted(self: *MLALayer, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("mla");
        }
    }

    pub fn getName(self: *const MLALayer) ?[]const u8 {
        return self.name;
    }

    pub fn registerFormula(self: *const MLALayer, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
            try graph.registerModuleType(n, self.module_type);
            try self.q_proj.registerFormula(graph);
            try self.w_dkv.registerFormula(graph);
            try self.w_kr.registerFormula(graph);
            try self.w_uk.registerFormula(graph);
            try self.w_uv.registerFormula(graph);
            try self.o_proj.registerFormula(graph);
        }
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

    /// 全序列前向传播 (经由计算图执行，支持 Autograd 梯度回传)
    pub fn forward(self: MLALayer, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
        const old_shape = x.shape;
        const is_3d = (old_shape.len == 3);
        const B = if (is_3d) old_shape.dims[0] else 1;
        const T = if (is_3d) old_shape.dims[1] else old_shape.dims[0];
        const C = if (is_3d) old_shape.dims[2] else old_shape.dims[1];
        const nh = self.n_head;
        const hd = self.head_dim;
        const dr = self.d_r;
        const q_head_dim = hd + dr;

        var x_2d = x;
        if (is_3d) {
            x_2d = try graph.reshape(x, &.{ B * T, C });
        }

        // 1. 投影 Q, 潜在 c_kv, 解耦 RoPE Key k_r
        const q_all = try self.q_proj.forward(graph, x_2d); // [B*T, nh * (hd + dr)]

        const c_kv = try self.w_dkv.forward(graph, x_2d); // [B*T, d_c]

        const k_r_2d = try self.w_kr.forward(graph, x_2d); // [B*T, d_r]

        // 2. 上投影还原内容键 Kc 与内容值 Vc
        const k_c_2d = try self.w_uk.forward(graph, c_kv); // [B*T, nh * hd]

        const v_c_2d = try self.w_uv.forward(graph, c_kv); // [B*T, nh * hd]

        // 3. 重塑并转置为 4D 多头结构，对 Query 的 RoPE 子空间与共享 k_r 施加旋转位置编码
        const q_4d = try graph.reshape(q_all, &.{ B, T, nh, q_head_dim });
        const q_trans = try graph.transposeND(q_4d, 1, 2);
        const q_rot = try graph.ropeOffset(q_trans, 0, hd);

        const k_c_4d = try graph.reshape(k_c_2d, &.{ B, T, nh, hd });
        const k_c = try graph.transposeND(k_c_4d, 1, 2);

        const k_r_4d = try graph.reshape(k_r_2d, &.{ B, 1, T, dr });
        const k_r_rot = try graph.rope(k_r_4d, 0);

        const k_r_heads = if (nh > 1) try graph.repeatKV(k_r_rot, nh) else k_r_rot;

        const k_full = try graph.concat(&.{ k_c, k_r_heads }, 3);

        const v_4d = try graph.reshape(v_c_2d, &.{ B, T, nh, hd });
        const v = try graph.transposeND(v_4d, 1, 2);

        // 4. 缩放点积因果注意力: Softmax((Q * K^T) / sqrt(hd + dr) + M) * V
        const k_t = try graph.transposeND(k_full, 2, 3);

        const att = try graph.batchMatMul(q_rot, k_t);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(q_head_dim)));
        const att_scaled = try graph.mulScalar(att, scale);

        const mask_node = try graph.tensorND(&.{ 1, 1, T, T }, false);
        mask_node.is_buffer = true;
        for (0..T) |i| {
            for (i + 1..T) |j| {
                mask_node.data[i * T + j] = -1e9;
            }
        }
        const att_masked = try graph.add(att_scaled, mask_node);

        const att_sm = try graph.softmax(att_masked);

        const y_4d = try graph.batchMatMul(att_sm, v);

        // 5. 合并多头并经输出投影 o_proj 输出特征
        const y_trans = try graph.transposeND(y_4d, 1, 2);

        const y_2d = try graph.reshape(y_trans, &.{ B * T, nh * hd });

        const out_2d = try self.o_proj.forward(graph, y_2d);

        if (is_3d) {
            return try graph.reshape(out_2d, &.{ B, T, C });
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

        // 单步推理使用局部无梯度计算图承载中间张量，函数返回时整体释放
        var step_g = autodiff.Graph.initNoGrad(allocator);
        defer step_g.deinit();
        const step_alloc = step_g.arenaAllocator();

        const x_2d = if (x.shape.len != 2) try step_g.reshape(x, &.{ B, C }) else x;

        // 1. 投影当前 Token 的 Q, c_kv 与 k_r
        const q_all = try self.q_proj.forward(&step_g, x_2d);
        const c_kv_step = try self.w_dkv.forward(&step_g, x_2d);
        const k_r_step = try self.w_kr.forward(&step_g, x_2d);

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
        const y_tensor = try step_g.zeros(&.{ B, nh * hd }, false);
        const y_concat = y_tensor.data;
        const scores = try step_alloc.alloc(f32, curr_len);
        const q_absorbed = try step_alloc.alloc(f32, dc);
        const u_latent = try step_alloc.alloc(f32, dc);

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

        // 4. 投影输出 o_proj，并拷贝到调用方分配器上 (局部计算图随函数返回释放)
        const out_proj = try self.o_proj.forward(&step_g, y_tensor);
        const out_shape: []const usize = if (x.shape.len == 3) &.{ B, 1, C } else &.{ B, C };
        return try tensor.array(allocator, out_shape, out_proj.data);
    }
};
