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
- **关联文件同步**：更新版本号时，必须同步更新以下三个文件：
  1. [`build.zig.zon`](build.zig.zon) 中的 `.version = "X.Y.Z"`
  2. [`src/root.zig`](src/root.zig) 中的 `VERSION = "X.Y.Z"` 与 `version = .{ .major = X, .minor = Y, .patch = Z }`
  3. [`CHANGELOG.md`](CHANGELOG.md) 中把 `Unreleased` 的内容归入新版本小节
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
- **Upgrade Producer and Consumer Together (生产端与消费端同步升级)**: When a shared format changes (e.g. the model graph JSON produced by `graph_ir.serializeJson` / `Graph.exportJson` and consumed by the chen-gz.github.io `/visualizer` page), in the same change:
  - update the exporter and `src/nn/model_graph.schema.json` (field descriptions and `enum` lists matching the Zig enums),
  - regenerate `examples/sample_model_graph.json` and `examples/minimal_model_graph.json` with `zig build run-report`,
  - copy the schema and both samples to `public/tools/visualizer/` in chen-gz.github.io and update its consumers,
  - update the tests, `doc/model-graph-visualization.md` and the full specification (`doc/visualization-model-edge-design.md` in chen-gz.github.io),
  - for breaking changes, bump the format version and record the change under `Changed` in the `CHANGELOG.md` `Unreleased` section.

## 4. Configuration Structs (配置结构体默认值)
- Every options / config / hyperparameter struct (e.g. `TSNEOptions`, `InitOptions`, `DataLoaderOptions`, `AdamWConfig`) gives each field a sensible default so that `.{}` is a valid configuration, and exposes `pub fn defaultOptions() Self` (or `defaultConfig()`) returning `.{}`.
- Models, optimizers and pipelines that take such a struct offer a convenient entry point that uses the defaults (e.g. `initDefault()`), so callers need no boilerplate.
