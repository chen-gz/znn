# Milestone 6: 端到端全量验证与对抗性加固 (Comprehensive E2E Verification & Hardening)

> **Status**: Queued (Milestone 6)  
> **Dependency**: Milestones 1 through 5  
> **Target Scope**: 全库测试、Schema 2.0 Web 对齐、压力与对抗性测试

---

## 1. 目标与背景 (Objective)

在完成前 5 个里程碑的深度架构重构后，本阶段作为最终收敛与加固门禁：
1. **全矩阵自动化验证**：确保从底层硬件驱动到高层大模型的每一个测试用例在全优化级别（Debug, ReleaseSafe, ReleaseFast）下均零故障运行；
2. **对抗性边界测试 (Adversarial Hardening)**：验证变长序列输入、上下文窗口截断溢出、空批次、显存 OOM 回退与极端数值梯度稳定性；
3. **前端 Web 可视化全契约对齐**：验证导出的 18 个经典模型 JSON 拓扑图与 Web 可视化工具完全对齐。

---

## 2. 核心验证维度 (Verification Vectors)

1. **层级与单向依赖严格性**：
   - 运行 `zig build check-layers`，确保全库文件保持 100% 单向拓扑，无任何私有子导入或 duck-typing 遗留。
2. **端到端测试与内存安全**：
   - 运行 `zig build test`，确保 390+ 单元测试全部通过，GPA 内存泄漏检测器汇报 0 leaks。
3. **多模型拓扑与 Web 可视化**：
   - 运行 `zig build run-book-models` 刷新 `examples/models/*.json`；
   - 运行 `cd web && npm test` 确保 Web 可视化前端测试完全通过。
4. **模型数值对齐验证**：
   - 验证 GPT 与 Gemma 4 自回归流式输出与基线严格位对齐。

---

## 3. 验收标准与归档 (Acceptance Criteria)

- [ ] `zig build check-layers` 100% pass；
- [ ] `zig build test` 100% pass（0 失败，0 泄漏）；
- [ ] `zig build run-book-models` 18 个模型全部成功生成；
- [ ] `cd web && npm test` 全部通过；
- [ ] 更新 `CHANGELOG.md`，将所有演进内容完整归入新版本预发布小节。
