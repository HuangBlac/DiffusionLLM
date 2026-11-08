# Public result evidence index

Start with [current progress](../docs/progress.md). These JSON/PNG artifacts retain their existing names and bytes; this index does not regenerate results or promote historical receipts into new experiment claims.

| Evidence family | Existing files | Explanation | Scope |
| --- | --- | --- | --- |
| Local LoRA pilot | [data_manifest.json](data_manifest.json), [qwen05b-same-200.json](qwen05b-same-200.json), [qwen05b-shift-200.json](qwen05b-shift-200.json) | [Writeup](../docs/history/pilot-results.md) | Pilot measurements, samples and data provenance; not full-parameter P1. |
| Fixed-checkpoint sampling | [sampling-group-size-200.json](sampling-group-size-200.json) | [Writeup](../docs/history/sampling-group-size-results.md) | Reveal-group comparison on the historical local adapters. |
| Hardware probes | [hardware-2026-10-29/](hardware-2026-10-29) | [Writeup](../README.md#historical-hardware-run) | Synthetic/random-weight hardware measurements; not pretrained 7B adaptation. |
| Managed workflow / P0 | [managed-transport-cloud-verified.json](managed-transport-cloud-verified.json), [managed-transport-local.json](managed-transport-local.json), [managed-workflow-validation-2026-11-02.json](managed-workflow-validation-2026-11-02.json), [p0-grok-verified-2026-11-03.json](p0-grok-verified-2026-11-03.json), [p0-managed-mount-failures-2026-11-02.json](p0-managed-mount-failures-2026-11-02.json), [p0-resource-audit-2026-11-02.json](p0-resource-audit-2026-11-02.json), [p0-retry-audit-2026-11-02.json](p0-retry-audit-2026-11-02.json) | [Writeup](../docs/history/p0-resource-check-results.md) | Dated transport/resource evidence; see also the historical managed workflow and handoff. |
| Full-parameter pilot inputs | [pipeline-pilot-data.json](pipeline-pilot-data.json) | [Writeup](../docs/history/first-round-readiness.md) | Development pilot manifest; not final formal-test release. |
| P1 progress and recovery | [p1-200-results-2026-11-03.json](p1-200-results-2026-11-03.json), [p1-bounded-progress-2026-11-03.json](p1-bounded-progress-2026-11-03.json), [p1-cpu-checkpoint-cloud-verified-2026-11-03.json](p1-cpu-checkpoint-cloud-verified-2026-11-03.json), [p1-cpu-compatible-cloud-verified-2026-11-03.json](p1-cpu-compatible-cloud-verified-2026-11-03.json), [p1-resume-transport-cloud-verified-2026-11-03.json](p1-resume-transport-cloud-verified-2026-11-03.json), [p1-training-objectives-2026-11-03.png](p1-training-objectives-2026-11-03.png) | [Writeup](../docs/history/p1-200-results-2026-11-03.md) | 200-step computation and earlier transport receipts; AR200 storage was repaired later, so read the next row too. |
| Storage repair / AR continuation | [ar-continuation-2026-11-04.json](ar-continuation-2026-11-04.json), [ar-recovery-confirmed-2026-11-04.json](ar-recovery-confirmed-2026-11-04.json), [checkpoint-storage-accepted-2026-11-04.json](checkpoint-storage-accepted-2026-11-04.json) | [Writeup](../docs/history/checkpoint-storage-repair-2026-11-04.md) | Server acknowledgement, fresh download and model/Adam/RNG restoration are distinct checks. |
| DLM200 diagnosis | [dlm-repetition-diagnosis-2026-11-05.json](dlm-repetition-diagnosis-2026-11-05.json), [dlm-repetition-diagnosis-2026-11-05.png](dlm-repetition-diagnosis-2026-11-05.png) | [Writeup](../docs/history/dlm-repetition-results-2026-11-05.md) | Fixed-weight generation/context/EOS diagnostics; lower repetition is not demonstrated coherence. |
| P2 data / EOS preparation | [p1-article-owner-audit-2026-11-05.json](p1-article-owner-audit-2026-11-05.json), [p2-data-candidates-2026-11-05.json](p2-data-candidates-2026-11-05.json), [p2-data-eos-gate-2026-11-05.json](p2-data-eos-gate-2026-11-05.json) | [Writeup](../docs/current/p2-readiness-2026-11-05.md) | Input/audit summaries, not evidence that P2 paired training completed. |

## Related explanations

- [Managed workflow](../docs/history/managed-colab-workflow.md) and [P0 handoff](../docs/history/p0-grok-handoff.md) explain the transport/resource history.
- [P1 readiness](../docs/history/p1-readiness-2026-11-03.md) and [resume entry](../docs/history/p1-resume-entry-2026-11-03.md) retain dated checkpoint constraints.
- The [local remasking report](../docs/history/remasking-local-results-2026-11-06.md) concerns historical LoRA weights, not cloud DLM200.
- The later 512-update learning-run and DLM512 evidence is indexed from [progress](../docs/progress.md); reports with source text and cloud locators remain in gitignored `private/`, not copied here.

Do not move or rewrite raw evidence merely to match a newer narrative. Public summaries are not permission to publish private artifacts.

Directory map: `docs/current/` holds living writeups; flat `docs/history/` holds dated evidence. Local-only private materials are grouped under `private/application/`, `private/reports/` and `private/runs/`, with dated inner folder names preserved. Private artifact content is preserved apart from path-prefix/link repairs for the merged layout.
