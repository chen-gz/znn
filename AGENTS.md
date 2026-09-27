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
  - 用户确认或要求 push 后，直接推送代码 commit (`jj git push` 或 `git push origin main`)。
  - **小步修改与日常修复支持移动已有 Tag**：对于属于当前版本范围内的小步修改/修复，不频繁递增新版本号，可直接通过 `git tag -f -a <tag> <commit> -m "..."` 移动已有版本 Tag 指向最新提交，并使用 `git push origin <tag> --force` 覆盖同步远端 Tag。
  - **新 Tag 的创建**：仅在用户明确指示或完成阶段性正式发布里程碑时，才创建递增的新版本 Tag 并推送。

