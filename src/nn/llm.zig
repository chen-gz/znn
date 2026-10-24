const std = @import("std");
const tensor = @import("../tensor.zig");
const vec_math = @import("../backend.zig").vec_math;
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");
const enterModuleScope = core.enterModuleScope;

const Tensor = tensor.Tensor;
const createPersistentTensor = core.createPersistentTensor;
const freePersistentTensor = core.freePersistentTensor;
const initWeights = core.initWeights;

// ============================================================================
// 1. 低秩自适应 (Low-Rank Adaptation, LoRA) 参数高效微调模块
// ============================================================================

/// 低秩自适应线性层 (Low-Rank Adaptation Linear Layer, LoRALinear)
pub const LoRALinear = struct {
    weight: *Tensor, // 冻结的基础权重 (Base Weight, requires_grad = false)
    bias: ?*Tensor, // 可选偏置向量
    lora_a: *Tensor, // 可训练低秩矩阵 A [in_features, r]
    lora_b: *Tensor, // 可训练低秩矩阵 B [r, out_features]
    in_features: usize,
    out_features: usize,
    r: usize,
    scaling: f32,
    name: ?[]const u8 = null,
    module_type: []const u8 = "LoRALinear",

    pub const formula = "y = x W_0 + \\frac{\\alpha}{r} (x A) B + b";

    pub const Options = struct {
        r: usize = 8,
        lora_alpha: f32 = 16.0,

        pub const default: Options = .{};
        pub fn defaultOptions() Options {
            return .{};
        }
    };

    pub fn initDefault(
        allocator: std.mem.Allocator,
        in_features: usize,
        out_features: usize,
    ) !LoRALinear {
        return init(allocator, in_features, out_features, Options.default.r, Options.default.lora_alpha);
    }

    pub fn init(
        allocator: std.mem.Allocator,
        in_features: usize,
        out_features: usize,
        r: usize,
        lora_alpha: f32,
    ) !LoRALinear {
        return initWithBias(allocator, in_features, out_features, r, lora_alpha, false);
    }

    pub fn initWithBias(
        allocator: std.mem.Allocator,
        in_features: usize,
        out_features: usize,
        r: usize,
        lora_alpha: f32,
        use_bias: bool,
    ) !LoRALinear {
        // 冻结的基础权重
        const weight = try createPersistentTensor(allocator, in_features, out_features, false);
        errdefer freePersistentTensor(allocator, weight);

        var bias: ?*Tensor = null;
        if (use_bias) {
            bias = try createPersistentTensor(allocator, 1, out_features, true);
        }
        errdefer if (bias) |b| freePersistentTensor(allocator, b);

        const lora_a = try createPersistentTensor(allocator, in_features, r, true);
        errdefer freePersistentTensor(allocator, lora_a);

        const lora_b = try createPersistentTensor(allocator, r, out_features, true);
        errdefer freePersistentTensor(allocator, lora_b);

        return LoRALinear{
            .weight = weight,
            .bias = bias,
            .lora_a = lora_a,
            .lora_b = lora_b,
            .in_features = in_features,
            .out_features = out_features,
            .r = r,
            .scaling = lora_alpha / @as(f32, @floatFromInt(r)),
        };
    }

    /// 库内标准参数初始化：冻结的基础权重按 He Normal 初始化 (加载预训练权重时由 loadModel 覆盖)；
    /// 低秩旁路 A 按 options 的权重策略初始化；旁路 B 全 0 以保证初始状态等价于基座 (Base) 模型；偏置按 options.bias_init 初始化
    pub fn resetParameters(self: *LoRALinear, random: std.Random, options: core.InitOptions) void {
        initWeights(random, self.weight.data, self.in_features, self.out_features, .{ .he_normal = .{} });
        initWeights(random, self.lora_a.data, self.in_features, self.r, options.resolveWeightInit());
        @memset(self.lora_b.data, 0.0);
        if (self.bias) |b| {
            initWeights(random, b.data, self.in_features, self.out_features, options.bias_init);
        }
    }

    /// 前向传播：Y = X * W_0 + (X * A) * B * scaling (+ bias)；与 `Linear` 一样支持任意秩输入 [..., in_features]
    pub fn forward(self: *const LoRALinear, graph: *autodiff.Graph, x_in: *Tensor) !*Tensor {
        const module_scope = try enterModuleScope(graph, self);
        defer module_scope.exit();
        const rows = try core.RowView.init(graph, x_in, self.in_features);
        const x = rows.matrix;

        // 1. 冻结主干前向：x * W_0
        const base_out = try graph.matmul(x, self.weight);

        // 2. 低秩自适应 (Low-Rank Adaptation, LoRA) 旁路计算：x * A -> [..., r]
        const lora_xa = try graph.matmul(x, self.lora_a);

        // (x * A) * B -> [..., out_features]
        const lora_xab = try graph.matmul(lora_xa, self.lora_b);

        // 缩放增量：lora_xab * scaling
        const scaled_lora = try graph.mulScalar(lora_xab, self.scaling);

        // 累加主干与旁路：base_out + scaled_lora
        const sum = try graph.add(base_out, scaled_lora);
        const out = if (self.bias) |b| try graph.addBias(sum, b) else sum;
        return rows.restore(graph, out, self.out_features);
    }

    /// 零推理延迟融合：将低秩自适应 (Low-Rank Adaptation, LoRA) 旁路权重融合至主干 W_0 = W_0 + scaling * (A * B)
    /// 按 i -> k -> j 顺序累加秩一更新 W_0[i, :] += (scaling * A[i, k]) * B[k, :]，最内层连续访问 W_0 与 B 的整行
    pub fn fuse(self: *LoRALinear) void {
        const in_f = self.in_features;
        const r_dim = self.r;
        const out_f = self.out_features;

        for (0..in_f) |i| {
            const w_row = self.weight.data[i * out_f .. (i + 1) * out_f];
            for (0..r_dim) |k| {
                const a_scaled = self.lora_a.data[i * r_dim + k] * self.scaling;
                const b_row = self.lora_b.data[k * out_f .. (k + 1) * out_f];
                for (w_row, b_row) |*w, b| w.* += a_scaled * b;
            }
        }
        @memset(self.lora_b.data, 0.0);
    }
};

// ============================================================================
// 2. 监督微调 (Supervised Fine-Tuning, SFT) 掩码损失与强化学习对齐损失：
//    直接偏好优化 (Direct Preference Optimization, DPO) 与组相对策略优化 (Group Relative Policy Optimization, GRPO)
// ============================================================================

/// 监督微调 (Supervised Fine-Tuning, SFT) 掩码交叉熵损失：仅对助手 (Assistant) 回答部分 (mask > 0) 计算损失
pub fn maskedCrossEntropyLoss(
    logits: *Tensor,
    targets: []const u32,
    mask: []const f32,
) !f32 {
    if (logits.shape.len != 2) return error.IncompatibleDimensions;
    const N = logits.shape.dims[0];
    const V = logits.shape.dims[1];
    if (targets.len != N or mask.len != N) return error.ShapeMismatch;
    if (V == 0 or logits.data.len < N * V) return error.ShapeMismatch;

    // 1. 先校验目标下标并累计总权重：梯度缩放 w_i / total_weight 需要在单遍前向中直接使用
    var total_weight: f32 = 0.0;
    for (targets, mask) |target, w| {
        if (w <= 0.0) continue;
        if (target >= V) return error.IndexOutOfBounds;
        total_weight += w;
    }
    if (total_weight <= 0.0) return 0.0;

    // 2. 单遍计算每行的对数和指数 (Log-Sum-Exp, LSE) 损失与归一化指数函数 (Softmax) 梯度
    const write_grad = logits.requires_grad and logits.grad.len == N * V;
    var total_loss: f32 = 0.0;
    for (0..N) |i| {
        const w = mask[i];
        if (w <= 0.0) continue;

        const row = logits.data[i * V .. (i + 1) * V];
        const label: usize = targets[i];
        const grad_row = if (write_grad) logits.grad[i * V .. (i + 1) * V] else null;
        const scale = if (write_grad) w / total_weight else 0.0;

        const row_loss = try tensor.losses.ceRow(row, label, null, grad_row, scale);
        total_loss += row_loss * w;
    }
    return total_loss / total_weight;
}

/// 监督微调 (Supervised Fine-Tuning, SFT) 掩码交叉熵损失 (自动微分 (Automatic Differentiation, Autograd) 计算图节点版本)
pub fn maskedCrossEntropyLossGraph(
    graph: *autodiff.Graph,
    logits: *Tensor,
    targets: anytype,
    mask: []const f32,
) !*Tensor {
    return graph.maskedCrossEntropyLoss(logits, targets, mask);
}

/// 直接偏好优化 (Direct Preference Optimization, DPO) 损失函数：
/// L_DPO = - E [ log( sigmoid( beta * ( (log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l)) ) ) ) ]
/// 四个对数概率切片长度不一致时返回 `error.ShapeMismatch`；`beta` 必须是有限正数，否则返回 `error.InvalidArgument`。
pub fn dpoLoss(
    pi_chosen_logps: []const f32,
    pi_rejected_logps: []const f32,
    ref_chosen_logps: []const f32,
    ref_rejected_logps: []const f32,
    beta: f32,
) !f32 {
    const N = pi_chosen_logps.len;
    if (pi_rejected_logps.len != N or ref_chosen_logps.len != N or ref_rejected_logps.len != N) {
        return error.ShapeMismatch;
    }
    try autodiff.graph_nn.validateDpoBeta(beta);
    if (N == 0) return 0.0;

    var total_loss: f32 = 0.0;
    for (0..N) |i| {
        total_loss += tensor.losses.dpoTerm(
            pi_chosen_logps[i],
            pi_rejected_logps[i],
            ref_chosen_logps[i],
            ref_rejected_logps[i],
            beta,
            null,
        );
    }
    return total_loss / @as(f32, @floatFromInt(N));
}

/// 直接偏好优化 (Direct Preference Optimization, DPO) 损失函数 (自动微分 (Automatic Differentiation, Autograd) 计算图节点版本)
pub fn dpoLossGraph(
    graph: *autodiff.Graph,
    pi_chosen_logps: *Tensor,
    pi_rejected_logps: *Tensor,
    ref_chosen_logps: []const f32,
    ref_rejected_logps: []const f32,
    beta: f32,
) !*Tensor {
    return graph.dpoLoss(pi_chosen_logps, pi_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta);
}

/// 组相对策略优化 (Group Relative Policy Optimization, GRPO) 优势计算
/// 对应 DeepSeek-R1 强化学习论文与博客第 8.5 节公式 (1)：
/// 对每个提示词 (Prompt) 并行采样的 G 个候选回复按组计算奖励的均值与标准差，并归一化输出优势值：
/// A_i = (r_i - mean({r_1..r_G})) / (std({r_1..r_G}) + eps)
pub fn computeGroupAdvantages(
    allocator: std.mem.Allocator,
    rewards: []const f32,
    group_size: usize,
    eps: f32,
) ![]f32 {
    if (group_size == 0 or rewards.len % group_size != 0) return error.InvalidArgument;

    const advantages = try allocator.alloc(f32, rewards.len);
    const num_groups = rewards.len / group_size;

    for (0..num_groups) |g| {
        const start = g * group_size;
        const group_rewards = rewards[start .. start + group_size];

        const mean = vec_math.sum(group_rewards) / @as(f32, @floatFromInt(group_size));
        const var_sum = vec_math.sumSquaredDeviations(group_rewards, mean);
        const std_dev = @sqrt(var_sum / @as(f32, @floatFromInt(group_size)));

        for (group_rewards, 0..) |r, j| {
            advantages[start + j] = (r - mean) / (std_dev + eps);
        }
    }

    return advantages;
}

/// 组相对策略优化 (Group Relative Policy Optimization, GRPO) 纯数值损失函数评估：
/// 对应 DeepSeek-R1 强化学习论文与博客第 8.5 节公式 (2) & (3)：
/// L_GRPO = - 1/N \sum [ min(r_t * A_i, clip(r_t, 1-eps, 1+eps) * A_i) - \beta * D_KL ]
/// 其中库尔贝克-莱布勒散度 (Kullback-Leibler Divergence, KL Divergence) D_KL(\pi_\theta || \pi_ref) = exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1
/// 输入长度不一致时返回 `error.ShapeMismatch`；`beta` 或 `clip_eps` 为负数或非有限值时返回 `error.InvalidArgument`。
pub fn computeGRPOLoss(
    old_logps: []const f32,
    new_logps: []const f32,
    advantages: []const f32,
    ref_logps: ?[]const f32,
    beta: f32,
    clip_eps: f32,
) !f32 {
    const N = old_logps.len;
    try validateGrpoInputs(N, new_logps.len, advantages.len, ref_logps, beta, clip_eps);
    if (N == 0) return 0.0;

    var total_obj: f32 = 0.0;
    for (0..N) |i| {
        const ref = if (ref_logps) |refs| refs[i] else null;
        total_obj += tensor.losses.grpoTerm(
            new_logps[i],
            old_logps[i],
            advantages[i],
            ref,
            beta,
            clip_eps,
            null,
        );
    }

    return -(total_obj / @as(f32, @floatFromInt(N)));
}

/// 组相对策略优化 (Group Relative Policy Optimization, GRPO) 损失函数，支持对 new_logps 的自动微分 (Automatic Differentiation, Autograd) 梯度回传 (若 new_logps.requires_grad 为 true)
/// 输入校验与 `computeGRPOLoss` 相同：长度不一致返回 `error.ShapeMismatch`，`beta` / `clip_eps` 非法返回 `error.InvalidArgument`。
pub fn grpoLoss(
    old_logps: *Tensor,
    new_logps: *Tensor,
    advantages: []const f32,
    ref_logps: ?[]const f32,
    beta: f32,
    clip_eps: f32,
) !f32 {
    const N = old_logps.data.len;
    try validateGrpoInputs(N, new_logps.data.len, advantages.len, ref_logps, beta, clip_eps);

    if (N == 0) return 0.0;

    var total_obj: f32 = 0.0;
    const inv_n = 1.0 / @as(f32, @floatFromInt(N));
    const write_grad = new_logps.requires_grad and new_logps.grad.len == N;

    for (0..N) |i| {
        const ref = if (ref_logps) |refs| refs[i] else null;
        var d_obj: f32 = undefined;
        const obj = tensor.losses.grpoTerm(
            new_logps.data[i],
            old_logps.data[i],
            advantages[i],
            ref,
            beta,
            clip_eps,
            if (write_grad) &d_obj else null,
        );
        total_obj += obj;

        if (write_grad) {
            new_logps.grad[i] += -inv_n * d_obj;
        }
    }

    return -(total_obj * inv_n);
}

/// 校验组相对策略优化 (Group Relative Policy Optimization, GRPO) 损失的输入长度与超参数
fn validateGrpoInputs(n: usize, new_len: usize, adv_len: usize, ref_logps: ?[]const f32, beta: f32, clip_eps: f32) !void {
    if (new_len != n or adv_len != n) return error.ShapeMismatch;
    if (ref_logps) |refs| {
        if (refs.len != n) return error.ShapeMismatch;
    }
    try autodiff.graph_nn.validateGrpoHyperparams(beta, clip_eps);
}

/// 组相对策略优化 (Group Relative Policy Optimization, GRPO) 损失函数 (自动微分 (Automatic Differentiation, Autograd) 计算图节点版本)
pub fn grpoLossGraph(
    graph: *autodiff.Graph,
    old_logps: *Tensor,
    new_logps: *Tensor,
    advantages: []const f32,
    ref_logps: ?[]const f32,
    beta: f32,
    clip_eps: f32,
) !*Tensor {
    return graph.grpoLoss(old_logps, new_logps, advantages, ref_logps, beta, clip_eps);
}

// ============================================================================
// 4. 投机采样推测草稿引擎 (Speculative Draft Engine & Prompt Lookup)
// ============================================================================

/// 在历史序列（Prompt 上下文 + 已生成词元）中反向查找当前尾部匹配的 N-gram，
/// 提取历史上紧随其后的后续词元作为推测候选序列 (Draft Candidates)。
/// 该算法零额外参数模型负担，完全基于自然语言/代码的局部重复性，与大模型一次性并行验证形成吞吐放大。
pub fn promptLookup(
    tokens: []const usize,
    min_n: usize,
    max_n: usize,
    max_speculate: usize,
) ?[]const usize {
    const lo = @max(min_n, 1);
    if (tokens.len < lo + 1) return null;
    const cur_len = tokens.len;

    var n = @min(max_n, cur_len - 1);
    while (n >= lo) : (n -= 1) {
        const pattern = tokens[cur_len - n .. cur_len];
        var search_idx: usize = cur_len - n;
        while (search_idx > 0) {
            search_idx -= 1;
            if (std.mem.eql(usize, tokens[search_idx .. search_idx + n], pattern)) {
                const draft_start = search_idx + n;
                if (draft_start < cur_len) {
                    const draft_end = @min(draft_start + max_speculate, cur_len);
                    if (draft_end > draft_start) {
                        return tokens[draft_start..draft_end];
                    }
                }
            }
        }
    }
    return null;
}

/// 投机采样推测草稿引擎 (Speculative Draft Engine)
/// 结合自适应多尺度 N-gram 历史回溯匹配与高频词元马尔可夫转移缓存 (Markov Transition Cache)
pub const SpeculativeDraftEngine = struct {
    allocator: std.mem.Allocator,
    transitions: std.AutoHashMapUnmanaged(usize, usize),
    stop_tokens: []const usize = &.{},

    pub fn init(allocator: std.mem.Allocator) SpeculativeDraftEngine {
        return .{
            .allocator = allocator,
            .transitions = .{},
        };
    }

    pub fn initWithStopTokens(allocator: std.mem.Allocator, stop_tokens: []const usize) SpeculativeDraftEngine {
        return .{
            .allocator = allocator,
            .transitions = .{},
            .stop_tokens = stop_tokens,
        };
    }

    pub fn deinit(self: *SpeculativeDraftEngine) void {
        self.transitions.deinit(self.allocator);
    }

    /// 根据 Prompt 与已知词元预填充转移图
    pub fn seedFromTokens(self: *SpeculativeDraftEngine, tokens: []const usize) !void {
        if (tokens.len < 2) return;
        for (0..tokens.len - 1) |i| {
            try self.transitions.put(self.allocator, tokens[i], tokens[i + 1]);
        }
    }

    /// 记录新生成的词元转移
    pub fn recordTransition(self: *SpeculativeDraftEngine, from: usize, to: usize) !void {
        try self.transitions.put(self.allocator, from, to);
    }

    fn isStopToken(self: *const SpeculativeDraftEngine, tok: usize) bool {
        for (self.stop_tokens) |st| {
            if (tok == st) return true;
        }
        return false;
    }

    /// 提出推测草稿候选词元 (Propose Draft Candidates)
    pub fn proposeDraft(
        self: *const SpeculativeDraftEngine,
        current_tokens: []const usize,
        out_buf: []usize,
    ) ?[]const usize {
        if (current_tokens.len == 0 or out_buf.len == 0) return null;

        // 1. 优先使用 N-gram 历史回溯匹配 (N=4, 3, 2)
        if (promptLookup(current_tokens, 2, 4, out_buf.len)) |draft| {
            return draft;
        }

        // 2. 马尔可夫多阶转移预测 (Token-to-Token Markov Chain)
        const cur_tok = current_tokens[current_tokens.len - 1];
        var count: usize = 0;
        var probe = cur_tok;
        while (count < out_buf.len) {
            if (self.transitions.get(probe)) |nxt| {
                if (nxt == probe or self.isStopToken(nxt)) break;
                out_buf[count] = nxt;
                count += 1;
                probe = nxt;
            } else {
                break;
            }
        }
        if (count > 0) {
            return out_buf[0..count];
        }
        return null;
    }
};
