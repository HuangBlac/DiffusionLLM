## 通用策略
不要反复把资源用到进行讨论，比较之上，按照let it crash的原则，先动手操作，我批准了就可以直接执行。

## 子代理执行偏好

- 具体可执行的操作优先直接委派 subagent，使用 `gpt-terra` 或 `gpt-luna`；边界清晰的小任务可交给 `luna-worker`。主代理负责讨论、整合与验收。
- 委派沿用已有授权、权限和阶段预算；不得扩大实验范围或绕过权限要求。

## 项目接续与证据约束

- 先读 `docs/progress.md`，再按其索引查看研究总方案、P2规程和原始结果。该页是当前状态入口；日期化P0/P1文档和旧笔记本属于历史证据。
- 本地 `experiments/p2_canvas_ablation.py` 是行为原型，不是完整P2训练器。实现、微型测试、真实更新、服务器保存和独立恢复必须分别记录。
- 2026-11-05已批准P2成对后缀政策短训练独立4CU：开发筛选16、正式训练96，总180CU；同一调拨无需重问。实际阶段起点/费率须运行前实读，准备、失败、评估及保存恢复都计费；不得按历史余额直接启动。
- 大模型检查点留云端，以正式Drive API和本地暂存处理；只取小报告到本机。原AR200/DLM200、冻结P1数据与历史报告不覆盖，禁用旧DriveFS训练保存路径。
- 新P2实验身份必须包含后缀政策、画布容量及数据/源码摘要；两组共同信息条件评价。保留300条续写不参与开发选型；60条校准来源不是人工标签。
- `private/`含原文、账户账目及Drive定位，不自动发布。新接手者须保留继承的工作区改动；本轮整理不构成提交、推送或启动训练的指令。

## Documentation navigation

- Keep `docs/progress.md` as the current-status entry; [docs/README.md](docs/README.md) is the short two-section index. Read the newest dated progress entries before older snapshots.
- Use only `docs/current/` (living scope, notes, plans and handoffs) and flat `docs/history/` (dated evidence). Do not add phase subfolders merely for navigation; preserve phase names in filenames and text.
- [experiments/README.md](experiments/README.md) catalogs unchanged Python module locations; [results/README.md](results/README.md) links evidence to writeups. Historical notebooks/runbooks are not current launch instructions.
- Keep private materials in `private/application/`, `private/reports/` or `private/runs/`, retaining each dated report/run folder inside the latter two. The three existing helper scripts remain at private root. Private artifacts retain their scientific content; embedded local path prefixes now follow this layout. Inspect dated execution gates before any separately authorized reuse.
