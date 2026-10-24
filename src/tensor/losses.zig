const std = @import("std");
const backend = @import("../backend.zig");
const vec_math = backend.vec_math;

/// 计算单行 logits 的对数和指数 (Log-Sum-Exp, LSE) 交叉熵损失：
/// loss = log(sum(exp(logits - max))) + max - logits[target]
/// - 若提供 out_probs，则将该行 Softmax 概率写入 out_probs；
/// - 若提供 out_grad，则将该行对数几率梯度累加：out_grad[j] += grad_scale * (prob[j] - 1_{j == target})。
pub fn ceRow(
    logits_row: []const f32,
    target: usize,
    out_probs: ?[]f32,
    out_grad: ?[]f32,
    grad_scale: f32,
) !f32 {
    if (logits_row.len == 0 or target >= logits_row.len) return error.IndexOutOfBounds;

    if (out_probs) |p| {
        if (p.len < logits_row.len) return error.ShapeMismatch;
        vec_math.softmaxRow(logits_row, p[0..logits_row.len]);
    }

    const lse = vec_math.logSumExpRow(logits_row, 1.0);
    const loss = lse - logits_row[target];

    if (out_grad) |g| {
        if (g.len < logits_row.len) return error.ShapeMismatch;
        for (g[0..logits_row.len], logits_row, 0..) |*grad_elem, v, j| {
            const p = if (out_probs) |probs| probs[j] else @exp(v - lse);
            grad_elem.* += grad_scale * (p - (if (j == target) @as(f32, 1.0) else 0.0));
        }
    }

    return loss;
}

/// 基于已缓存的 Softmax 概率分布累加单行交叉熵梯度：
/// out_grad[j] += scale * (probs[j] - 1_{j == target})
pub fn accumulateCeGradRow(
    probs: []const f32,
    target: usize,
    out_grad: []f32,
    scale: f32,
) void {
    const len = @min(probs.len, out_grad.len);
    for (out_grad[0..len], probs[0..len], 0..) |*g, p, j| {
        g.* += scale * (p - (if (j == target) @as(f32, 1.0) else 0.0));
    }
}

/// 直接偏好优化 (Direct Preference Optimization, DPO) 单样本损失项：
/// z = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
/// loss = -log(sigmoid(z)) = log(1 + exp(-z))
/// 若提供 out_dz，则将偏导 d(loss)/dz = -beta * sigmoid(-z) 写入 out_dz。
/// 注意：d(loss)/d(pi_chosen) = dz, d(loss)/d(pi_rejected) = -dz。
pub fn dpoTerm(
    pi_chosen: f32,
    pi_rejected: f32,
    ref_chosen: f32,
    ref_rejected: f32,
    beta: f32,
    out_dz: ?*f32,
) f32 {
    const log_ratio_chosen = pi_chosen - ref_chosen;
    const log_ratio_rejected = pi_rejected - ref_rejected;
    const z = beta * (log_ratio_chosen - log_ratio_rejected);

    const loss_i = if (z > 0.0)
        @log(1.0 + @exp(-z))
    else
        -z + @log(1.0 + @exp(z));

    if (out_dz) |dz_ptr| {
        const sig_neg_z = vec_math.sigmoid(-z);
        dz_ptr.* = -beta * sig_neg_z;
    }

    return loss_i;
}

/// 组相对策略优化 (Group Relative Policy Optimization, GRPO) 单样本代理目标与 KL 惩罚项：
/// ratio = exp(new_logp - old_logp)
/// surrogate = min(ratio * adv, clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
/// kl = exp(ref - new_logp) - (ref - new_logp) - 1
/// obj = surrogate - beta * kl
/// 目标值被最大化，因此损失对应 -obj / N。
/// 若提供 out_d_obj，则将 d(obj)/d(new_logp) 写入 out_d_obj。
/// 损失梯度对应 d(Loss)/d(new_logp) = -dL/N * d_obj。
pub fn grpoTerm(
    new_logp: f32,
    old_logp: f32,
    adv: f32,
    ref_logp: ?f32,
    beta: f32,
    clip_eps: f32,
    out_d_obj: ?*f32,
) f32 {
    const ratio = @exp(new_logp - old_logp);
    const s1 = ratio * adv;
    const clipped_ratio = std.math.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps);
    const s2 = clipped_ratio * adv;
    const surrogate = @min(s1, s2);

    var kl: f32 = 0.0;
    const ref = ref_logp orelse old_logp;
    if (beta > 0.0) {
        const u = ref - new_logp;
        kl = @exp(u) - u - 1.0;
    }
    const obj = surrogate - beta * kl;

    if (out_d_obj) |d_ptr| {
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

        var d_kl: f32 = 0.0;
        if (beta > 0.0) {
            d_kl = 1.0 - @exp(ref - new_logp);
        }

        d_ptr.* = d_surrogate - beta * d_kl;
    }

    return obj;
}
