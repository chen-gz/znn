#!/usr/bin/env python3
"""
Gemma 4 E4B (Effective 4B) 官方 Python MLX 对齐与精度对照验证脚本

用于直接对比 Python MLX 原生推理与 Zig (znn) 引擎的输出一致性，包含：
1. 逐层表征范数与特征统计 (Layer Norms)
2. Prefill 阶段末尾词元的 Logits 对数几率分布与 Top-10 词元排序
3. 自回归解码文本生成与生成速率基准

运行方法：
    /Users/guangzong/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-mac14-arm64@39/bin/python3 tools/verify_gemma4_e4b_mlx.py
或使用任意安装了 mlx_lm 的 Python 环境：
    python3 tools/verify_gemma4_e4b_mlx.py --prompt "who are you" --max-tokens 25
"""

import argparse
import sys
import time

try:
    import mlx.core as mx
    from mlx_lm import load, generate
except ImportError:
    print("错误: 当前 Python 环境未安装 mlx 或 mlx_lm。")
    print("请使用内置的 Apple MLX 环境运行，例如：")
    print("  /Users/guangzong/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-mac14-arm64@39/bin/python3 tools/verify_gemma4_e4b_mlx.py")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 4 E4B MLX 对齐验证工具")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="/Users/guangzong/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit",
        help="Gemma 4 E4B 模型权重目录路径",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="who are you",
        help="测试 Prompt 文本",
    )
    parser.add_argument(
        "-n",
        "--max-tokens",
        type=int,
        default=25,
        help="自回归最大生成词元数",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("🔍 Gemma 4 E4B (Effective 4B) Python MLX 原生官方基准验证")
    print("=" * 80)
    print(f"1. 加载模型: {args.model}")

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    t_load = time.perf_counter() - t0
    print(f"   - 架构: 42 层 Transformer (18 层 KV 共享)")
    print(f"   - 加载耗时: {t_load:.3f} 秒\n")

    # 1. 构建对话 Prompt 词元 (与 Zig znn 完全一致的 Gemma 4 对话模板)
    if args.prompt == "who are you":
        # 精确测试词元序列: <bos><|turn>user\n who are you<turn|>\n<|turn>model\n
        prompt_tokens = [2, 105, 2364, 107, 1015, 659, 611, 106, 107, 105, 4368, 107]
    else:
        user_tokens = tokenizer.encode(f" {args.prompt}")
        prompt_tokens = (
            [2, 105, 2364, 107]
            + user_tokens
            + [106, 107, 105, 4368, 107]
        )

    print(f"2. Prompt 词元 ({len(prompt_tokens)} 个):")
    print(f"   Tokens: {prompt_tokens}")
    print(f"   Decoded: {repr(tokenizer.decode(prompt_tokens))}\n")

    # 2. 手动执行 Transformer 前向传播以提取逐层范数与中间状态
    print("3. 手动前向传播与中间层特征比对:")
    x = mx.array([prompt_tokens])
    lm = model.language_model.model

    h = lm.embed_tokens(x) * lm.embed_scale
    print(f"   - Initial Embeddings norm: {mx.linalg.norm(h).item():.4f}")

    per_layer = lm._project_per_layer_inputs(h, lm._get_per_layer_inputs(x, h))
    print(f"   - Per-Layer Inputs (PLE) norm: {mx.linalg.norm(per_layer).item():.4f}")

    masks = lm._make_masks(h, [None] * len(lm.layers))
    intermediates = [(None, None)] * len(lm.layers)

    check_layers = [0, 1, 5, 21, 22, 23, 24, 25, 41]
    for idx, (layer, prev_idx) in enumerate(zip(lm.layers, lm.previous_kvs)):
        kvs, offset = intermediates[prev_idx]
        h, kvs, offset = layer(
            h,
            masks[idx],
            None,
            per_layer_input=per_layer[:, :, idx, :],
            shared_kv=kvs,
            offset=offset,
        )
        intermediates[idx] = (kvs, offset)

        if idx in check_layers:
            norm_val = mx.linalg.norm(h).item()
            tok_0_3 = h[0, -1, :3].tolist()
            print(f"   - Layer {idx:2d} norm: {norm_val:8.2f}, last token [0..3]: {[round(v, 4) for v in tok_0_3]}")

    final_h = lm.norm(h)
    print(f"   - Final LayerNorm norm: {mx.linalg.norm(final_h).item():.4f}")

    # 3. 计算最终词表 Logits
    logits = lm.embed_tokens.as_linear(final_h)
    if hasattr(model.language_model, "final_logit_softcapping") and model.language_model.final_logit_softcapping:
        cap = model.language_model.final_logit_softcapping
        logits = cap * mx.tanh(logits / cap)

    last_logits = logits[0, -1].astype(mx.float32)
    mx.eval(last_logits)

    # 4. 输出 Top-10 Prefill 预测词元与 Logit 数值
    print("\n4. Prefill 首字 Logits 统计与 Top-10 词元排序:")
    min_logit = mx.min(last_logits).item()
    max_logit = mx.max(last_logits).item()
    print(f"   [Logits Stats]: min={min_logit:.2f}, max={max_logit:.2f}, vocab_size={len(last_logits)}")
    print(f"     - Logit for ' I' (564): {last_logits[564].item():.2f}")
    print(f"     - Logit for ' am' (1006): {last_logits[1006].item():.2f}")
    print(f"     - Logit for '一' (237009): {last_logits[237009].item():.2f}")

    # 使用 argpartition 取 top 10
    top_indices = mx.argpartition(-last_logits, 10)[:10].tolist()
    top_indices = sorted(top_indices, key=lambda i: last_logits[i].item(), reverse=True)

    print("\n   [Top 10 Prefill Tokens]:")
    for rank, tid in enumerate(top_indices):
        val = last_logits[tid].item()
        text = tokenizer.decode([tid])
        print(f"     #{rank}: id={tid:6d}, val={val:6.2f}, text={repr(text)}")

    # 5. 端到端自回归生成测试
    print("\n5. 端到端自回归解码测试 (generate):")
    t_gen_start = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=prompt_tokens,
        max_tokens=args.max_tokens,
        verbose=False,
    )
    t_gen_total = time.perf_counter() - t_gen_start

    print(f"   Model : {response}")
    print("\n" + "=" * 80)
    print(f"🏁 生成验证完成 (耗时: {t_gen_total:.2f} 秒)")
    print("=" * 80)


if __name__ == "__main__":
    main()
