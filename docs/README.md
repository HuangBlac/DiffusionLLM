# Documentation index

Start with **[current progress and handoff](progress.md)**. Read its newest dated entries first; older snapshots remain evidence, not current launch instructions. The two folders below organize purpose, not new research phases.

## Current - `current/`

Living scope, learning notes and still-relevant plans/handoffs. A dated plan or passing test is not proof of completed training.

- [真实文本续写的DeepSeek裁判方案与预算](current/coherence-judge-plan.md)
- [Diffusion LLM 项目交付目标](current/deliverables.md)
- [从自回归权重到掩码扩散语言模型](current/diffusion-llm-note.md)
- [Dream 7B与Qwen2.5 7B的HellaSwag质量与速度对照](current/dream-qwen-hellaswag-plan.md)
- [512更新学习运行：执行接续](current/learning-run-execution-2026-11-06.md)
- [首份学习交付：固定配方的0.5B训练与评价](current/learning-run-plan-2026-11-06.md)
- [P2下一轮：画布／长度提示的成对训练验证](current/p2-canvas-training-proposal-2026-11-05.md)
- [P2成对画布执行接续（2026-11-06）](current/p2-execution-2026-11-06.md)
- [P2接续：开发数据与成对训练准备](current/p2-readiness-2026-11-05.md)
- [Diffusion LLM训练与评测总方案](current/project-plan.md)
- [真实文本上的AR到扩散模型适配](current/real-text-study.md)
- [撤回与重掩码的实现核对](current/remasking-audit.md)
- [区分训练进展与采样误差的交叉实验](current/training-sampling-diagnosis.md)

## History - `history/`

Completed measurements and dated P0/P1/pilot process evidence, kept together without phase subfolders. Preserve their original scope and limitations; do not run old notebooks as current P2 entries.

- [检查点保存修复与AR恢复点核查](history/checkpoint-storage-repair-2026-11-04.md)
- [DLM重复生成：开发诊断](history/dlm-repetition-diagnosis-2026-11-04.md)
- [DLM200重复生成诊断结果](history/dlm-repetition-results-2026-11-05.md)
- [Small-model AR → masked-denoising pilot](history/experiment-design.md)
- [首轮准备与离线小测记录](history/first-round-readiness.md)
- [完整流程与第一轮小规模测试](history/first-round-runbook.md)
- [Colab自动执行与持久保存](history/managed-colab-workflow.md)
- [P0接续核验与P1执行边界](history/p0-grok-handoff.md)
- [P0资源检查：部分完成，已停机](history/p0-resource-check-results.md)
- [两组200步计算完成，AR200检查点未可靠保存](history/p1-200-results-2026-11-03.md)
- [P1费用与持久检查点准备](history/p1-readiness-2026-11-03.md)
- [P1本轮结果：20步状态已保存，200步尚未启动](history/p1-results-2026-11-03.md)
- [从已保存的第20步跨会话续训](history/p1-resume-entry-2026-11-03.md)
- [Qwen2.5 0.5B masked denoising pilot](history/pilot-results.md)
- [本地历史LoRA检查点：直接可见评分回撤](history/remasking-local-results-2026-11-06.md)
- [Fixed checkpoint sampling comparison](history/sampling-group-size-results.md)

Implementation: [script catalog](../experiments/README.md). Public evidence: [result index](../results/README.md). Local-only private materials are grouped as `private/application/`, `private/reports/` and `private/runs/`; each dated report/run retains its own folder. Do not auto-publish private contents.
