//! 因果语言模型训练辅助器 (Causal Language Model Training Utilities)
//!
//! 封装自回归因果语言模型 (如 GPT, Gemma 4) 的单步训练、损失计算、反向传播与梯度裁剪。

const std = @import("std");
const tensor = @import("../tensor.zig");
const autodiff = @import("../autodiff.zig");
const optim = @import("../optim.zig");

pub const CausalLMStepResult = struct {
    loss: f32,
    batch_size: usize,
    seq_len: usize,
    num_tokens: usize,
};

/// 执行因果语言模型训练单步：
/// - model: 实现了 forward(graph, X) 的因果自回归模型 (如 nn.Module(models.GPT(cfg)) 或 models.GPT)
/// - optimizer: 优化器 (如 AdamWOptimizer)
/// - token_ids: 输入词元张量 [batch_size, seq_len]
/// - targets: 目标词元切片 [batch_size * seq_len] (每个位置的真实下一个词元)
/// - clip_config: 可选梯度裁剪配置
pub fn trainCausalLMStep(
    allocator: std.mem.Allocator,
    model: anytype,
    optimizer: anytype,
    token_ids: *tensor.Tensor,
    targets: []const u8,
    clip_config: ?optim.GradClipConfig,
) !CausalLMStepResult {
    const B = token_ids.shape.dims[0];
    const T = token_ids.shape.dims[1];
    const num_tokens = B * T;
    std.debug.assert(targets.len == num_tokens);

    var graph = autodiff.Graph.initDefault(allocator);
    defer graph.deinit();

    // 1. 前向传播计算对数几率 (Logits)
    const logits = try model.forward(&graph, token_ids);

    // 2. 将 [B, T, V] 转换为 [B*T, V] 以便计算交叉熵
    const V = logits.shape.dims[logits.shape.len - 1];
    const logits_2d = if (logits.shape.len != 2)
        try graph.reshape(logits, &.{ num_tokens, V })
    else
        logits;

    // 3. 计算交叉熵损失
    const loss = try graph.softmaxCrossEntropy(logits_2d, targets);
    const step_loss = loss.data[0];

    // 4. 清零梯度、反向传播与优化器更新
    model.zeroGrad();
    try graph.backward(loss);

    if (clip_config) |cfg| {
        _ = optim.clipGradients(optimizer.params, cfg);
    }

    try optimizer.step(&graph);

    return .{
        .loss = step_loss,
        .batch_size = B,
        .seq_len = T,
        .num_tokens = num_tokens,
    };
}
