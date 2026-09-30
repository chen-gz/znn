# ZNN (Zig Neural Network) Rules & Guidelines

## 1. Version Control Guidelines (Jujutsu / jj)
- For both `gemini` and `antigravity`, always run `jj log` immediately after modifying any files to save the current work and maintain visibility of the version control state.
- Always use `jj` (Jujutsu) for version control operations.
- Prefer `jj git clone <url>` over `git clone <url>` for initial setup.
- Upon completing a logical task or a significant phase, always use `jj describe -m "..."` to provide a clear, structured summary of the changes made, ensuring the history is readable and meaningful.

## 2. Versioning & Release Rules (版本更新规则)
遵循语义化版本规范 (Semantic Versioning: `MAJOR.MINOR.PATCH`):
- **大 Feature 更新时必须更新版本号**：在完成重大特性 (Major Feature / Key Enhancement) 的开发与交付时，必须同步更新版本号。
- **关联文件同步**：每次更新版本号时，必须同步更新以下三个文件：
  1. [`build.zig.zon`](file:///Users/guangzong/Documents/znn/build.zig.zon) 中的 `.version = "X.Y.Z"`
  2. [`src/root.zig`](file:///Users/guangzong/Documents/znn/src/root.zig) 中的 `VERSION = "X.Y.Z"` 与 `version = .{ .major = X, .minor = Y, .patch = Z }`
  3. [`CHANGELOG.md`](file:///Users/guangzong/Documents/znn/CHANGELOG.md) 顶部追加对应新版本的详细更新日志 (遵循 [Keep a Changelog](https://keepachangelog.com/) 规范，分类记录 `Added`、`Changed`、`Fixed` 等条目)
- **版本更新权责划分**：
  - **大版本号 (`MAJOR`) 与小版本号 (`MINOR`)**：由用户决策或重大里程碑发布。未经用户明确要求或确认，日常迭代不频繁递增。
  - **日常修复与改进**：保持当前版本或记录在 `Unreleased` 中，避免每个小提交都频繁升级版本号与打标签。
- **推送与 Tag 规则**：
  - **日常小步改动只推 main 分支**：日常迭代、小步优化与 Bug 修复只推送代码 commit 到 `main` 分支（无需 Tag，直接使用 `jj git push` 或 `git push origin main`）。
  - **让 Tag 成为里程碑**：不要频繁打 Tag 或频繁递增版本号。只有在积累了一批功能、完成阶段性重要成果或由用户明确指示时，才作为里程碑创建并推送新的 Git Tag。

## 3. Refactoring & No Backward Compatibility (重构优先，不保留向后兼容)
- **Refactor Freely (允许并鼓励重构)**: When a change exposes a flawed design, refactor the affected code, APIs, data formats and documents directly instead of layering patches on top. Restructuring modules, renaming public APIs and fields, and rewriting components are all allowed.
- **No Backward Compatibility (不需要考虑向后兼容)**: Do NOT keep compatibility layers for superseded designs, including:
  - fallback branches for old data or export schema versions (e.g. the visualization JSON schema `1.0`),
  - deprecated aliases, re-exports, wrapper functions, or duplicated old/new code paths,
  - heuristics kept only to support a replaced mechanism (e.g. name-based scope inference after explicit module scopes landed).
- **Delete What Is Superseded (删除被取代的内容)**: Remove replaced code, tests and plan documents in the same change, and update the affected tests to the new behavior instead of keeping the old assertions. Leave no commented-out code behind.
- **Upgrade Producer and Consumer Together (生产端与消费端同步升级)**: When a shared format changes (e.g. `graph_ir` / `exportJson` output consumed by the chen-gz.github.io `/visualizer` page), update the exporter, all consumers, `examples/sample_model_graph.json`, tests and the design doc together, bump the format version, and record the breaking change under `Changed` in the `CHANGELOG.md` `Unreleased` section.
