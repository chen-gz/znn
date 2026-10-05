# Gemma 4 12B 模型支持与推理排查待办清单 (TODO)

## 1. 现状概述 (Status Quo)
在 `znn` 中完成了 Gemma 4 12B 模型的完整架构定义、流式 4-bit (Q4_0) 量化转换器、零拷贝内存映射（mmap）以及 262K 官方词表解码器 `Gemma4Vocabulary`。
经过对齐 Hugging Face 官方实现，修复了 RoPE 半切分（Split-Half）、Proportional RoPE 角度切片、无权重缩放 $V$ 归一化（`v_norm`）以及注意力分数缩放因子（`scaling = 1.0`）等关键机制。

模型在 ReleaseFast 模式下成功在 16GB Mac 设备上端到端运行，输入 `"hello world"` 时正确生成自然的语言回复：
```
User  : hello world
Model :hello world!

* 1.
* 2.
*
```

---

## 2. 核心问题与解决记录 (Resolved Architectural Alignment)

### [x] 1. RoPE (半切分 Split-Half 与 Proportional RoPE) 对齐 (已完成)
- **原因**：Gemma 4 预训练采用 GPT-NeoX / LLaMA 样式的半切分 RoPE（`[-x2, x1]`），而非相邻两两配对；且全局注意力层（`full_attention`）采用 Proportional RoPE（25% 角度旋转，前 64 个角度旋转后对应维度为 $[0\dots 63]$ 与 $[256\dots 319]$，其余维度不旋转）。
- **解决**：在 `tensor` 和 `autodiff.Graph` 中实现并引入 `ropeSplitHalf`，并在全局层与滑动层中正确应用。

---

### [x] 2. Attention Value 归一化 (`v_norm`) 补齐 (已完成)
- **原因**：Gemma 4 架构在对 $V$ 进行注意力加权之前，需先经过一层无参数学习的 RMSNorm（$y = x / \sqrt{\text{mean}(x^2) + \epsilon}$，`with_scale=False`）。
- **解决**：在 `Gemma4Attention` 与 `Gemma4Q4Attention` 中增加了 `GemmaUnscaledRMSNorm`，在投影出 $V$ 后先对各头应用 `v_norm`。

---

### [x] 3. Attention 得分缩放因子对齐 (`scaling = 1.0`) (已完成)
- **原因**：Gemma 4 由于 $Q$ 和 $K$ 已分别经过 `q_norm` 与 `k_norm` 单范数归一化，注意力点积 $QK^T$ 的缩放因子为 $1.0$，无需除以 $\sqrt{d_k}$。
- **解决**：移除了冗余的 $1/\sqrt{d}$ 缩放，与官方 `eager_attention_forward` 保持一致。

---

### [x] 4. Chat Template 词元规范化与多模态控制符过滤 (已完成)
- **解决**：
  - 修正了带词首空格的 Token（`29104: ' hello'`）。
  - 在采样器中过滤了特殊多模态控制符（`258880..258884`）与 `<pad>`（`0`）。

---

## 3. 后续性能与功能优化 (Upcoming Roadmap)

### [ ] 1. KV Cache (键值缓存) 加速支持
- **当前状态**：当前自回归生成采用朴素的全序列重计算（每次生成第 $t$ 个 token 均重新前向传播 $0 \dots t-1$）。
- **优化方案**：为 Gemma 4 添加增量 KV Cache，自回归步只需处理单个新 token ($T=1$)，单步生成延迟将从 ~8 秒大幅缩短至毫秒级。

### [ ] 2. 多轮对话与交互式 CLI (Chat REPL)
- 支持终端交互式多轮对话输入与流式 Token 实时打字机输出。
