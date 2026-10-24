# ZNN (Zig Neural Network) Rules & Guidelines

## 1. Version Control Guidelines (Jujutsu / jj)
- Always use `jj` (Jujutsu) for version control operations. Prefer `jj git clone <url>` over `git clone <url>` for initial setup.
- For both `gemini` and `antigravity`, always run `jj log` immediately after modifying any files to save the current work and maintain visibility of the version control state.
- Upon completing a logical task or a significant phase, use `jj describe -m "<type>(<scope>): <summary>"` (e.g. `feat(vis): ...`, `fix(autodiff): ...`, `docs(vis): ...`) with a body that explains what changed and why, so the history stays readable and meaningful.
- **Push Procedure (推送流程)**: work is pushed directly to `main`:
  1. `jj git fetch`
  2. `jj bookmark set main -r @`
  3. `jj git push`
  4. `jj log` to verify the state.
  - If the push is refused because `main` moved on the remote, run `jj rebase -d main`, resolve conflicts, re-run the tests, then push again.
  - Force push is forbidden (no `--force`, `-f`, or other non-fast-forward updates).

## 2. Versioning & Release Rules (版本更新规则)
遵循语义化版本规范 (Semantic Versioning: `MAJOR.MINOR.PATCH`):
- **日常改动记录在 `Unreleased`**：每个功能、重构与修复都记录在 [`CHANGELOG.md`](CHANGELOG.md) 的 `Unreleased` 小节 (遵循 [Keep a Changelog](https://keepachangelog.com/) 规范，分类记录 `Added`、`Changed`、`Fixed` 等条目)，不为单个提交递增版本号。
- **版本号由用户决定**：版本号只在用户明确要求或确认时更新，通常在积累了一批重大特性后作为里程碑发布。完成重大特性时可以向用户建议发布新版本。
- **关联文件同步**：更新版本号时，必须同步更新以下四个文件：
  1. [`build.zig.zon`](build.zig.zon) 中的 `.version = "X.Y.Z"`
  2. [`src/root.zig`](src/root.zig) 中的 `VERSION = "X.Y.Z"` 与 `version = .{ .major = X, .minor = Y, .patch = Z }`
  3. [`src/tests.zig`](src/tests.zig) 中的 `VERSION` 与 `version` 断言
  4. [`CHANGELOG.md`](CHANGELOG.md) 中把 `Unreleased` 的内容归入新版本小节
- **推送与 Tag 规则**：
  - **日常改动只推 main 分支**：日常迭代、优化与 Bug 修复只推送 commit 到 `main` 分支，不打 Tag。
  - **Tag 只用于里程碑**：只有在发布新版本时 (即用户要求或确认之后) 才创建并推送对应的 Git Tag。

## 3. Refactoring & No Backward Compatibility (重构优先，不保留向后兼容)
- **Refactor Freely (允许并鼓励重构)**: When a change exposes a flawed design, refactor the affected code, APIs, data formats and documents directly instead of layering patches on top. Restructuring modules, renaming public APIs and fields, and rewriting components are all allowed.
- **No Backward Compatibility (不需要考虑向后兼容)**: Do NOT keep compatibility layers for superseded designs, including:
  - fallback branches that read or write an older data or export format version,
  - deprecated aliases, re-exports, wrapper functions, or duplicated old/new code paths,
  - heuristics kept only to support a mechanism that has been replaced.
- **Delete What Is Superseded (删除被取代的内容)**: Remove replaced code, tests and plan documents in the same change, and update the affected tests to the new behavior instead of keeping the old assertions. Leave no commented-out code behind. Documents describe the current design only; the history of a change belongs in commit messages and `CHANGELOG.md`.
- **Upgrade Producer and Consumer Together (生产端与消费端同步升级)**: When a shared format changes (e.g. the model graph JSON produced by `graph_ir.serializeJson` / `nn.exportJson` and consumed by the `web/` visualizer), in the same change:
  - update the exporter and `src/nn/model_graph.schema.json` (field descriptions and `enum` lists matching the Zig enums),
  - regenerate `examples/sample_model_graph.json`, `examples/minimal_model_graph.json` (`zig build run-report`) and `examples/models/*.json` (`zig build run-book-models`),
  - update the `web/` visualizer consumers, tests (`cd web && npm test`), `doc/model-graph-visualization.md` and the full specification (`doc/visualization-model-edge-design.md`),
  - for breaking changes, bump the format version and record the change under `Changed` in the `CHANGELOG.md` `Unreleased` section.

## 4. Configuration Structs (配置结构体默认值)
- Every options / config / hyperparameter struct (e.g. `TSNEOptions`, `InitOptions`, `DataLoaderOptions`, `AdamWConfig`) gives each field a sensible default so that `.{}` is a valid configuration, and exposes `pub fn defaultOptions() Self` (or `defaultConfig()`) returning `.{}`.
- Models, optimizers and pipelines that take such a struct offer a convenient entry point that uses the defaults (e.g. `initDefault()`), so callers need no boilerplate.

## 5. Comment & Terminology Conventions (注释缩写规范)
- **Full Name Before Abbreviation (注释中使用缩写时先写全称再接缩写)**: Whenever an abbreviation or acronym is used in source code comments (`//` or `///`), always write out its full name first, followed by the abbreviation in parentheses — for example:
  - `缩放点积注意力 (Scaled Dot-Product Attention, SDPA)`
  - `多头潜在注意力 (Multi-head Latent Attention, MLA)`
  - `低秩自适应 (Low-Rank Adaptation, LoRA)`
  - `旋转位置编码 (Rotary Position Embedding, RoPE)`
  - `键值缓存 (Key-Value Cache, KVCache)`
  - `均方根层归一化 (Root Mean Square Layer Normalization, RMSNorm)`
  - `直接偏好优化 (Direct Preference Optimization, DPO)` / `群组相对策略优化 (Group Relative Policy Optimization, GRPO)`

## 6. Modularity & Layered Package Dependencies (模块化清晰与包级单向依赖)
- **Clear Modularity (保持文件模块化清晰)**: Organize source files by cohesive domain and single responsibility so module boundaries stay clear and maintainable; do not enforce an arbitrary per-file byte size limit (such as `< 60 KB`).
- Packages under `src/` form a strict one-way layering; a cross-package `@import` may only point to a lower layer:
  `backend` (0) ← `tensor`, `dataset` (1) ← `autodiff`, `regression`, `manifold` (2) ← `nn`, `cross_validation` (3) ← `optim`, `models` (4) ← `engine` (5) ← `bench` (6) ← `root` (7).
- `zig build check-layers` ([`tools/check_layers.zig`](tools/check_layers.zig)) enforces this rule and also runs as part of `zig build test`. When adding a new package, register its layer in the `layers` table of that tool.
- Imports between files of the same package are allowed. To break an upward dependency, move the shared code down to the lower layer or pass an opaque handle (e.g. `Tensor.creator: ?*tensor.OpHandle`, converted with `autodiff.Op.handle()` / `autodiff.Op.creatorOf()`); never re-export a higher-layer API from a lower-layer facade.

## 7. Model & Engine Design Principles (模型层与引擎层设计原则)
- **`models` (Layer 4) 是纯粹的模型定义与架构实现**：
  - `models` 仅包含模型配置（`*Config`）、网络层结构体、参数容器、前向计算（`forward`）与原生自回归会话生成（`generate()`）。
  - 严禁在 `models` 包中堆砌上层端到端控制台阶段打印、复杂的命令行参数解析或特定执行流引擎（如历史上的 `Gemma4*Engine`）。
- **`engine` (Layer 5) 专注于训练工作流与执行调度**：
  - 管理多步训练循环、梯度更新与评估流程（如 `ClassificationTrainer`、因果自回归模型单步训练 `trainCausalLMStep`）。
- **避免跨模型推理流水线的过早抽象 (No Premature Generalization)**：
  - 在尚未接入多个异构大模型（如 LLaMA、Mistral、Qwen 等）时，严禁生造一个内部写死特定模型词表或控制符的虚假“通用 Pipeline”。
  - 各具体模型推理通过各自的原生结构体与专属词表直接调用（如 `examples/` 直接调用 `Gemma4E4BModel` / `DefaultGemma4Q4` 与 `Gemma4Vocabulary`）。
  - 待后续积累了更多异构大模型并形成稳定的通用分词器与生成协议后，再在 `engine` 统一沉淀通用的多态推理流水线。
- **保持端到端大模型训练能力完备**：
  - 框架必须完整支持因果自回归语言模型的全生命周期训练（计算图构建、交叉熵损失、反向传播与优化器更新），确保可以直接加载 `models.GPT` 等模型编写预训练、SFT 与微调脚本。

