const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const core = @import("core.zig");

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
    name_buf: [64]u8 = undefined,
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

    pub fn setName(self: *LoRALinear, name: []const u8) void {
        if (std.fmt.bufPrint(&self.name_buf, "{s}", .{name})) |s| {
            self.name = s;
        } else |_| {
            self.name = name;
        }
        self.weight.setNameFormatted("{s}.weight", .{self.name.?});
        self.lora_a.setNameFormatted("{s}.lora_a", .{self.name.?});
        self.lora_b.setNameFormatted("{s}.lora_b", .{self.name.?});
        if (self.bias) |b| b.setNameFormatted("{s}.bias", .{self.name.?});
    }

    pub fn setNameFormatted(self: *LoRALinear, comptime fmt: []const u8, args: anytype) void {
        var buf: [64]u8 = undefined;
        if (std.fmt.bufPrint(&buf, fmt, args)) |s| {
            self.setName(s);
        } else |_| {
            self.setName("lora_linear");
        }
    }

    pub fn getName(self: *const LoRALinear) ?[]const u8 {
        return self.name;
    }

    pub fn registerFormula(self: *const LoRALinear, graph: *autodiff.Graph) !void {
        if (self.name) |n| {
            try graph.setModuleFormula(n, formula);
            try graph.registerModuleType(n, self.module_type);
        }
    }

    pub fn initDefault(
        allocator: std.mem.Allocator,
        in_features: usize,
        out_features: usize,
        random: std.Random,
    ) !LoRALinear {
        return init(allocator, in_features, out_features, Options.default.r, Options.default.lora_alpha, random);
    }

    pub fn init(
        allocator: std.mem.Allocator,
        in_features: usize,
        out_features: usize,
        r: usize,
        lora_alpha: f32,
        random: std.Random,
    ) !LoRALinear {
        return initWithBias(allocator, in_features, out_features, r, lora_alpha, false, random);
    }

    pub fn initWithBias(
        allocator: std.mem.Allocator,
        in_features: usize,
        out_features: usize,
        r: usize,
        lora_alpha: f32,
        use_bias: bool,
        random: std.Random,
    ) !LoRALinear {
        // 冻结的基础权重
        const weight = try createPersistentTensor(allocator, in_features, out_features, false);
        errdefer freePersistentTensor(allocator, weight);
        initWeights(random, weight.data, in_features, out_features, .{ .he_normal = .{} });

        var bias: ?*Tensor = null;
        if (use_bias) {
            const b = try createPersistentTensor(allocator, 1, out_features, true);
            errdefer freePersistentTensor(allocator, b);
            @memset(b.data, 0.0);
            bias = b;
        }
        errdefer if (bias) |b| freePersistentTensor(allocator, b);

        // 可微调低秩旁路 A：高斯初始化
        const lora_a = try createPersistentTensor(allocator, in_features, r, true);
        errdefer freePersistentTensor(allocator, lora_a);
        initWeights(random, lora_a.data, in_features, r, .{ .he_normal = .{} });

        // 可微调低秩旁路 B：全 0 初始化以保证初始状态等价于基座 (Base) 模型
        const lora_b = try createPersistentTensor(allocator, r, out_features, true);
        errdefer freePersistentTensor(allocator, lora_b);
        @memset(lora_b.data, 0.0);

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

    /// 显式自定义初始化（仅限库外用户代码调用）：
    /// 执行后标记 is_custom_initialized = true，Graph.initWeights 遍历时将绝对跳过，不会被重写！
    pub fn customInit(self: *LoRALinear, random: std.Random, options: core.InitOptions) void {
        const w_init = options.resolveWeightInit();
        initWeights(random, self.lora_a.data, self.in_features, self.r, w_init);
        @memset(self.lora_b.data, 0.0);
        self.lora_a.is_custom_initialized = true;
        self.lora_b.is_custom_initialized = true;
        if (self.bias) |b| {
            initWeights(random, b.data, self.in_features, self.out_features, options.bias_init);
            b.is_custom_initialized = true;
        }
    }

    pub fn deinit(self: LoRALinear, allocator: std.mem.Allocator) void {
        freePersistentTensor(allocator, self.weight);
        if (self.bias) |b| freePersistentTensor(allocator, b);
        freePersistentTensor(allocator, self.lora_a);
        freePersistentTensor(allocator, self.lora_b);
    }

    pub fn zeroGrad(self: LoRALinear) void {
        self.weight.zeroGrad();
        self.lora_a.zeroGrad();
        self.lora_b.zeroGrad();
        if (self.bias) |b| b.zeroGrad();
    }

    /// 前向传播：Y = X * W_0 + (X * A) * B * scaling (+ bias)
    pub fn forward(self: LoRALinear, graph: *autodiff.Graph, x: *Tensor) !*Tensor {
        const module_scope = try graph.enterModule(self.name, self.module_type);
        defer module_scope.exit();
        if (self.name) |n| try graph.setModuleFormula(n, formula);
        // 1. 冻结主干前向：x * W_0
        const base_out = try graph.matmul(x, self.weight);

        // 2. 低秩自适应 (Low-Rank Adaptation, LoRA) 旁路计算：x * A -> [..., r]
        const lora_xa = try graph.matmul(x, self.lora_a);

        // (x * A) * B -> [..., out_features]
        const lora_xab = try graph.matmul(lora_xa, self.lora_b);

        // 缩放增量：lora_xab * scaling
        const scaled_lora = try graph.mulScalar(lora_xab, self.scaling);

        // 累加主干与旁路：base_out + scaled_lora
        const out = try graph.add(base_out, scaled_lora);
        if (self.bias) |b| return try graph.addBias(out, b);
        return out;
    }

    /// 零推理延迟融合：将低秩自适应 (Low-Rank Adaptation, LoRA) 旁路权重融合至主干 W_0 = W_0 + scaling * (A * B)
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
// 2. 监督微调 (Supervised Fine-Tuning, SFT) 掩码损失与强化学习对齐损失：
//    直接偏好优化 (Direct Preference Optimization, DPO) 与组相对策略优化 (Group Relative Policy Optimization, GRPO)
// ============================================================================

/// 监督微调 (Supervised Fine-Tuning, SFT) 掩码交叉熵损失：仅对助手 (Assistant) 回答部分 (mask > 0) 计算损失
pub fn maskedCrossEntropyLoss(
    logits: *Tensor,
    targets: []const u32,
    mask: []const f32,
    allocator: std.mem.Allocator,
) !f32 {
    const N = logits.shape.dims[0];
    const V = logits.shape.dims[1];
    if (targets.len != N or mask.len != N) return error.ShapeMismatch;
    _ = allocator;

    var total_loss: f32 = 0.0;
    var total_weight: f32 = 0.0;

    for (0..N) |i| {
        if (mask[i] <= 0.0) continue;
        if (targets[i] >= V) return error.IndexOutOfBounds;

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
        if (logits.requires_grad and logits.grad.len == N * V) {
            for (0..N) |i| {
                const w = mask[i];
                if (w <= 0.0) continue;
                const scale = w / total_weight;
                const row = logits.data[i * V .. (i + 1) * V];
                const grad_row = logits.grad[i * V .. (i + 1) * V];
                var max_v = row[0];
                for (row) |v| if (v > max_v) {
                    max_v = v;
                };
                var sum_exp: f32 = 0.0;
                for (row) |v| sum_exp += @exp(v - max_v);
                const label: usize = targets[i];
                for (0..V) |j| {
                    const p = @exp(row[j] - max_v) / sum_exp;
                    grad_row[j] += scale * (p - (if (j == label) @as(f32, 1.0) else 0.0));
                }
            }
        }
        return total_loss / total_weight;
    }
    return 0.0;
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

/// 组相对策略优化 (Group Relative Policy Optimization, GRPO) 纯数值损失函数评估：
/// 对应 DeepSeek-R1 强化学习论文与博客第 8.5 节公式 (2) & (3)：
/// L_GRPO = - 1/N \sum [ min(r_t * A_i, clip(r_t, 1-eps, 1+eps) * A_i) - \beta * D_KL ]
/// 其中库尔贝克-莱布勒散度 (Kullback-Leibler Divergence, KL Divergence) D_KL(\pi_\theta || \pi_ref) = exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1
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

/// 组相对策略优化 (Group Relative Policy Optimization, GRPO) 损失函数，支持对 new_logps 的自动微分 (Automatic Differentiation, Autograd) 梯度回传 (若 new_logps.requires_grad 为 true)
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

            // 计算库尔贝克-莱布勒散度 (Kullback-Leibler Divergence, KL Divergence) 项梯度 d(kl) / d(new_logp)
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
// 3. 采样与生成策略 (Sampling Strategies)
// ============================================================================

/// 核采样 / 累积概率阈值采样 (Nucleus Sampling, Top-P，带温度系数 Temperature)
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
