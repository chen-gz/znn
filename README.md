# Zig ML (`znn`)

[![Zig](https://img.shields.io/badge/Zig-0.16.0-orange.svg)](https://ziglang.org/)
[![Changelog](https://img.shields.io/badge/docs-CHANGELOG.md-brightgreen.svg)](CHANGELOG.md)

`znn` is a modular deep learning and LLM library written from scratch in pure **Zig 0.16.0**, with no runtime dependencies. The current version is recorded in [`build.zig.zon`](build.zig.zon); release notes are in [`CHANGELOG.md`](CHANGELOG.md).

- **Tensors**: N-dimensional strided tensors, NumPy-style broadcasting, generic dtypes (`f32`, `f64`, `bf16`, `i32`, `i64`, `bool`), zero-copy slices, and a compile-time shaped `StaticTensor`.
- **Autodiff**: dynamic reverse-mode graph with arena-allocated activations and gradients, and module scopes used by the model graph export.
- **Layers**: `Linear`, `Conv2D`, `ConvTranspose2D`, normalization and pooling layers, RNN / LSTM / GRU, and Transformer components (causal attention with GQA / MQA, `KVCache`, MLA, SwiGLU, MoE, LoRA, GPT).
- **Training**: SGD / Adam / AdamW, learning-rate schedulers, gradient clipping, Safetensors models and binary optimizer checkpoints, DPO / GRPO losses, BPE tokenizer.
- **Classical ML**: OLS, Ridge, Lasso, ElasticNet, K-fold cross-validation, t-SNE.
- **Performance**: macOS Accelerate (AMX) for GEMM, with a portable `@Vector` SIMD fallback.

## Documentation

| Topic | Location |
| :--- | :--- |
| Architecture: layers, tensor and autodiff internals, modules, optimizers, acceleration | [`doc/architecture.md`](doc/architecture.md) |
| Model graph export (scopes, local graphs, JSON Schema) | [`doc/model-graph-visualization.md`](doc/model-graph-visualization.md) |
| Documentation index | [`doc/README.md`](doc/README.md) |
| Roadmap and task list | [`plan/TODO.md`](plan/TODO.md) |
| Comparison with NumPy 2.x | [`plan/NUMPY_GAP_ANALYSIS.md`](plan/NUMPY_GAP_ANALYSIS.md) |
| Contributor and agent rules | [`AGENTS.md`](AGENTS.md) |

## Quick Start

```bash
zig build test                          # all unit tests
zig build coverage                      # kcov coverage report (-- --fail-under=90, -- --open)
just bench                              # benchmark suite in ReleaseFast
just bench --suite gemm --filter conv   # suites: gemm, ops, activations, layers, models, optimizers, tokenizer
```

Datasets are downloaded on demand:

```bash
zig build download-dataset                     # Fashion MNIST (default)
zig build download-dataset -- mnist            # MNIST
zig build download-dataset -- tinyshakespeare  # also: wikitext2, tinystories, alpaca, all_llm
```

## Examples

| Command | Source | Description |
| :--- | :--- | :--- |
| `zig build run -Doptimize=ReleaseFast` | [`fashion_mnist.zig`](examples/fashion_mnist.zig) | 3-layer MLP on Fashion MNIST |
| `zig build run-cnn` | [`cnn.zig`](examples/cnn.zig) | 2D CNN on Fashion MNIST |
| `zig build run-gan` | [`gan.zig`](examples/gan.zig) | GAN training with `BceWithLogitsLoss` |
| `zig build run-lr` | [`linear_regression.zig`](examples/linear_regression.zig) | OLS closed form vs. gradient descent |
| `zig build run-logr` | [`logistic_regression.zig`](examples/logistic_regression.zig) | Logistic regression |
| `zig build run-ridge` | [`ridge_regression.zig`](examples/ridge_regression.zig) | Ridge regression and multicollinearity |
| `zig build run-reg` | [`regularized_regression.zig`](examples/regularized_regression.zig) | Ridge vs. Lasso vs. ElasticNet |
| `zig build run-cv` | [`cross_validation.zig`](examples/cross_validation.zig) | 5-fold cross-validation grid search |
| `zig build run-emb` | [`transformer_embedding.zig`](examples/transformer_embedding.zig) | Token and position embeddings |
| `zig build run-att` | [`transformer_attention.zig`](examples/transformer_attention.zig) | Causal multi-head self-attention |
| `zig build run-block` | [`transformer_block.zig`](examples/transformer_block.zig) | Transformer block forward step |
| `zig build run-gpt` | [`transformer_gpt.zig`](examples/transformer_gpt.zig) | GPT initialization and forward step |
| `zig build run-shakespeare` | [`train_shakespeare.zig`](examples/train_shakespeare.zig) | Mini-GPT on TinyShakespeare with text generation |
| `zig build run-llm` | [`llm_training.zig`](examples/llm_training.zig) | BPE + SwiGLU + AdamW + SFT + LoRA + DPO pipeline |
| `zig build run-report` | [`export_model_report.zig`](examples/export_model_report.zig) | Model graph JSON export (`sample_model_graph.json`, `minimal_model_graph.json`) |
| `zig build run-bench` | [`benchmark.zig`](examples/benchmark.zig) | Benchmark CLI |

[`comptime_static_tensor.zig`](examples/comptime_static_tensor.zig) demonstrates the compile-time shaped tensor.
