# Experiment script catalog

Read [current progress](../docs/progress.md) and the [documentation index](../docs/README.md) before choosing an entry. This is an inventory, not an instruction or authorization to execute scripts. Importing a module, passing tests, running a training update, saving to the server and independently restoring a checkpoint are separate evidence.

Python modules deliberately remain in this directory so imports, tests and historical commands keep working. Descriptions below are taken from the existing module documentation.

## Orientation

- Historical local pilot: [protocol](../docs/history/experiment-design.md) and [results](../docs/history/pilot-results.md).
- P0/P1: [historical workflow](../docs/history/managed-colab-workflow.md) and [storage/recovery evidence](../docs/history/checkpoint-storage-repair-2026-11-04.md). Old DriveFS/P0 paths are not current training instructions.
- P2: [readiness](../docs/current/p2-readiness-2026-11-05.md) and [execution handoff](../docs/current/p2-execution-2026-11-06.md). `p2_canvas_ablation.py` is a local behavioral prototype, not the full trainer.
- Learning run: [specification](../docs/current/learning-run-plan-2026-11-06.md) and [original handoff](../docs/current/learning-run-execution-2026-11-06.md); later observations live in progress.md.

## Operating boundaries

Data preparation may download public artifacts. Training/Colab entries require separate authorization, fresh resource observations and their documented budget gates. Do not run builders or historical notebooks merely to browse this catalog; generated handoffs may contain local/cloud bindings. Keep weights, source text and cloud/account locators out of public documentation.

`report_results.py` and `compare_sampling.py` write historical Markdown reports under `docs/history/`; raw JSON outputs remain in root `results/`. No report was regenerated during this reorganization.

## Data preparation

| Script | Existing responsibility |
| --- | --- |
| [prepare_data.py](prepare_data.py) | Download versioned public artifacts; keep weights and corpus out of Git. |
| [prepare_pipeline_pilot.py](prepare_pipeline_pilot.py) | Prepare a development-only full-parameter pilot from already downloaded data. |
| [prepare_p2_articles.py](prepare_p2_articles.py) | Prepare deterministic article candidates; token eligibility precedes final freeze. |
| [prepare_p2_examples.py](prepare_p2_examples.py) | Build exact-deduplicated development inputs from article candidates, without training. |

## Historical small local pilot and reports

| Script | Existing responsibility |
| --- | --- |
| [small_qwen.py](small_qwen.py) | Bounded AR-to-masked-denoising pilot; not a Dream reproduction. |
| [report_results.py](report_results.py) | Create a factual Markdown report from completed, matched pilot runs. |
| [compare_sampling.py](compare_sampling.py) | Inference-only comparison: reveal 1 vs 8 tokens with fixed pilot weights. |

## Colab / managed execution infrastructure

| Script | Existing responsibility |
| --- | --- |
| [build_colab_notebook.py](build_colab_notebook.py) | Build a self-contained, single-code-cell Colab entry; default is a CPU test. |
| [colab_pipeline.py](colab_pipeline.py) | Single invocation for transport checks or the bounded 0.5B P0 pipeline. |
| [managed_run.py](managed_run.py) | Run bounded child processes and keep evidence outside temporary compute storage. |
| [mount_contract.py](mount_contract.py) | Bind transport validation to the mount settings actually used by training. |
| [diagnose_drive_mount.py](diagnose_drive_mount.py) | CPU-only Drive mount diagnosis; never output credentials or raw DriveFS logs. |
| [import_p0_notebook.py](import_p0_notebook.py) | Recover verifiable P0 summaries from one saved notebook cell, not checkpoints. |

## P1 / checkpoint storage and transport

| Script | Existing responsibility |
| --- | --- |
| [full_parameter_pilot.py](full_parameter_pilot.py) | Bounded conditional AR/DLM pilot, separate from the historical LoRA runs. |
| [checkpoint_transport.py](checkpoint_transport.py) | CPU synthetic PyTorch checkpoint round trip; not 0.5B model training. |
| [api_checkpoint_entry.py](api_checkpoint_entry.py) | Bounded GPU continuation using verified API snapshots; no DriveFS writes. |
| [api_checkpoint_training.py](api_checkpoint_training.py) | Storage-only adapter for the unchanged training kernel. |
| [drive_checkpoint_store.py](drive_checkpoint_store.py) | Immutable checkpoint publication through Drive API, never a mounted cache. |
| [checkpoint_cadence.py](checkpoint_cadence.py) | Keep local 20-update saves; bound expensive server publications separately. |
| [checkpoint_metadata.py](checkpoint_metadata.py) | Bounded metadata inspection of our PyTorch checkpoints, without tensor reads. |
| [checkpoint_restore_check.py](checkpoint_restore_check.py) | Fresh CPU process: restore real model/Adam/RNG and run finite inference only. |
| [pinned_drive_artifact.py](pinned_drive_artifact.py) | Read a content-pinned JSON artifact without equating metadata versions with bytes. |

## Fixed-weight DLM diagnosis

| Script | Existing responsibility |
| --- | --- |
| [dlm_generation_diagnosis.py](dlm_generation_diagnosis.py) | Inference-only paired sampling and context diagnostics for the frozen P1 DLM. |
| [dlm_eos_probe.py](dlm_eos_probe.py) | P2 exploratory EOS/canvas check: same CPU weights, greedy C8 at three lengths. |
| [dlm512_diagnosis.py](dlm512_diagnosis.py) | CPU-only paired fixed-canvas diagnostics for the frozen DLM512 checkpoint. |

## P2 paired suffix-policy implementation

| Script | Existing responsibility |
| --- | --- |
| [p2_canvas_ablation.py](p2_canvas_ablation.py) | Local behavioral prototype for a proposed P2 ablation; this file does not train. |
| [build_p2_paired_notebook.py](build_p2_paired_notebook.py) | Build a locked P2 Colab handoff with a verifiable, local source bundle. |
| [p2_paired_entry.py](p2_paired_entry.py) | P2 paired run from fresh, externally observed resource and prepared data. |
| [p2_paired_training.py](p2_paired_training.py) | P2 paired-training primitives. Importing this module never starts a run. |
| [p2_restore_check.py](p2_restore_check.py) | Independent P2 restore check; this module never downloads or starts training. |

## 512-update learning-run implementation

| Script | Existing responsibility |
| --- | --- |
| [build_learning_run_notebook.py](build_learning_run_notebook.py) | Build the locked learning-run notebook and immutable source bundle. |
| [learning_run_entry.py](learning_run_entry.py) | Budget gates for the fixed 512-step learning run; importing never starts work. |
| [learning_run_training.py](learning_run_training.py) | Fixed-configuration learning run; import does not start training. |
| [learning_run_evaluation.py](learning_run_evaluation.py) | Bounded private development metrics and continuation samples. |
| [learning_run_restore.py](learning_run_restore.py) | Independent CPU audit of a learning-run checkpoint. |

## Historical resource / hardware probes

| Script | Existing responsibility |
| --- | --- |
| [colab_probe.py](colab_probe.py) | Colab resource and masked-language-model smoke test; no pretrained weights. |
| [colab_7b_benchmark.py](colab_7b_benchmark.py) | Qwen2.5-7B-shaped BF16 LoRA hardware benchmark, NOT pretrained adaptation. |
| [resource_resume_probe.py](resource_resume_probe.py) | P0 only: 20 updates, then replay update 20 in a fresh process from step 19. |

Directory map: `docs/current/` holds living writeups; flat `docs/history/` holds dated evidence. Local-only private materials are grouped under `private/application/`, `private/reports/` and `private/runs/`, with dated inner folder names preserved. Private artifact content is preserved apart from path-prefix/link repairs for the merged layout.
