# ZNN (Zig Neural Network) Rules & Guidelines

## 1. Version Control Guidelines (Jujutsu / jj)
- For both `gemini` and `antigravity`, always run `jj log` immediately after modifying any files to save the current work and maintain visibility of the version control state.
- Always use `jj` (Jujutsu) for version control operations.
- Prefer `jj git clone <url>` over `git clone <url>` for initial setup.
- Upon completing a logical task or a significant phase, always use `jj describe -m "..."` to provide a clear, structured summary of the changes made, ensuring the history is readable and meaningful.

## 2. Versioning & Release Rules (版本更新规则)
遵循语义化版本规范 (Semantic Versioning: `MAJOR.MINOR.PATCH`):
- **大 Feature 更新时必须更新版本号**：在完成重大特性 (Major Feature / Key Enhancement) 的开发与交付时，必须同步更新版本号。
- **关联文件同步**：每次更新版本号时，必须同步更新以下两个文件：
  1. [`build.zig.zon`](file:///Users/guangzong/Documents/znn/build.zig.zon) 中的 `.version = "X.Y.Z"`
  2. [`src/root.zig`](file:///Users/guangzong/Documents/znn/src/root.zig) 中的 `VERSION = "X.Y.Z"` 与 `version = .{ .major = X, .minor = Y, .patch = Z }`
- **版本更新权责划分**：
  - **大版本号 (`MAJOR`)**：由**用户决策**。未经用户明确要求或确认，AI 不得擅自升级大版本号。
  - **小版本号与补丁号 (`MINOR` / `PATCH`)**：可由 **AI 直接决定与递增**。当交付完整的大 Feature 或关键功能模块时，AI 自行升级次版本号（例如 `0.1.0 -> 0.2.0`），Bug 修复或小优化升级补丁号（例如 `0.2.0 -> 0.2.1`）。
