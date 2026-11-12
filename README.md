# DiffusionLLM

**[阅读项目报告](Project%20Report.md)**（2026-11-11修订） · [已验证进度](docs/progress.md) · [512步学习运行与诊断结果](docs/history/learning-run-results-2026-11-08.md)

A learning and experimental project adapting Qwen2.5-0.5B Base to masked denoising, with AR controls and a planned mature-model evaluation.

## Start here - status recorded through 2026-11-08

**Read [project progress and handoff](docs/progress.md) first**, then the [documentation index](docs/README.md). The progress page is a dated evidence log: its newest entries supersede older status snapshots and next-step lists below them.

The 2026-11-07 entry records a single-seed learning run with AR/DLM each reaching 512 updates, both final checkpoints independently downloaded and CPU-restored with finite inference. The 2026-11-08 entry adds fixed-weight DLM512 diagnostics and report updates, not further training. High-mask prediction difficulty and generated-context degradation coexist; reduced repetition does not establish coherent generation or a quality/speed advantage. CUDA RNG was not GPU-replayed. This is not the planned multi-seed research or a reproduction of the original BYOD recipe.

Budget approvals, observed balances and verified model updates are separate evidence. The latest resource snapshot in the progress page is dated 2026-11-08; this documentation cleanup did not recheck live CU balances, billing rates or sessions and did not start compute. The approved total remains 180 CU; the P2 paired trial and the 512-update learning run have separate stage accounting. Large checkpoints remain cloud-side. **Do not run historical P0/P1 notebooks as the current P2 entry.**

## Repository map

| Folder / entry | Purpose |
| --- | --- |
| [docs/progress.md](docs/progress.md) | Current-status entry and dated handoff evidence |
| [docs/README.md](docs/README.md) | Two folders: `current/` for living writeups and `history/` for dated evidence |
| [experiments/README.md](experiments/README.md) | Script catalog; Python module paths remain unchanged |
| `tests/` | Unit, fake-integration and bounded process checks; test success is not a cloud experiment result |
| [results/README.md](results/README.md) | Public measurement files and their explanatory writeups |
| `notebooks/` | Historical managed P0 notebook; not the current training entry |
| `data/`, `checkpoints/` | Gitignored local corpus/weights/checkpoints; not reorganized |
| `private/` | Only `application/`, `reports/`, `runs/` plus root helper scripts; local-only, never auto-publish |

Useful next reads: [research and budget](docs/current/project-plan.md), [deliverables](docs/current/deliverables.md), [learning-run specification](docs/current/learning-run-plan-2026-11-06.md), [P2 execution handoff](docs/current/p2-execution-2026-11-06.md), and [learning-note draft](docs/current/diffusion-llm-note.md).

<details>
<summary>Historical P1/P2 README snapshot (2026-11-06; superseded by later progress entries)</summary>

P1 single-seed full-parameter AR/DLM runs reached 200 updates; AR200 passed independent model/Adam/RNG recovery. Fixed-weight DLM diagnosis completed 80 samples, 8 context trajectories and 24 additional EOS probes. Greedy sampling amplifies repetition; stochastic sampling still does not give reliable coherence.

P2 inputs are prepared in Drive (4096 training / 100 development / 300 reserved continuation / 60 calibration sources). Exact exclusions and 544 old token arrays were verified; near-duplicate audit, human calibration and formal test release remain pending. The paired suffix-policy trial has an approved independent 4 CU cap, within the unchanged 180 CU total. **Local regression passed 213 tests; one existing CPU child-process restore test was deselected after a prior manual interruption and remains unverified. Ruff and source review passed. The Colab notebook was uploaded and its full external byte readback matched the local artifact; its default remains `RUN=False`. No P2 training or model-checkpoint upload has occurred.**

The last recorded balance, 182.41 CU on 2026-11-05, predates job completion. Current balance, runtime state, actual rate and account-wide sessions remain unknown because browser control is unavailable and `mcporter` is missing. This turn did not start a new GPU and does not establish zero existing sessions or zero total CU use. Large checkpoints remain cloud-side. Do not run historical P0/P1 notebooks as the current P2 entry.

The balance/control limitations above describe that historical snapshot, not a fresh resource check. Later learning-run evidence must not be conflated with completion of the separate P2 suffix-policy trial.

</details>

The following experiment table and reproduction commands describe historical experiments. They do not launch the proposed P2 trial. All work remains subject to the progress page's evidence and budget boundaries.

## Historical experiments

| Experiment | Evidence | What it establishes |
| --- | --- | --- |
| 100-token absorbing-mask check and one synthetic GPU update | `results/hardware-2026-10-29/colab-check-results.json` | Empirical masking marginals and a working backward/update path |
| A100/G4, Qwen2.5-7B-shaped random weights, frozen BF16 base + FP32 LoRA | `results/hardware-2026-10-29/` | Short-run throughput and memory under the tested configuration |
| Qwen2.5-0.5B pretrained weights: shifted vs same-position denoising | [Protocol](docs/history/experiment-design.md), [results](docs/history/pilot-results.md) | Short adaptation reduces masked CE; repetitive generation remains a failure |
| Fixed 200-step checkpoints: reveal 1 vs 8 tokens per round | [All outputs and measurements](docs/history/sampling-group-size-results.md) | Smaller reveal groups change repetition patterns but do not resolve degeneration |

The hardware measurements used random weights and synthetic tokens. They must not be described as pretrained model training or a Dream reproduction. Colab billing rates were observations on 2026-10-29, not current quotations.

## Reproduce the small-model experiment

Use Python 3.12 and a CUDA-capable GPU. The original Windows host has PyTorch 2.6.0+cu124 and Transformers 4.57.6. Install the matching PyTorch CUDA build for your machine, then the remaining requirements.

```sh
python -m pip install -r requirements.txt
python -m pytest tests -q
python experiments/prepare_data.py
python experiments/small_qwen.py --alignment both --steps 200 --test-examples 32
python experiments/report_results.py
```

On the original Windows host, `python -s` excludes conflicting user-site packages. A clean environment is recommended elsewhere. Downloads are approximately 1 GB. The preparation script pins both model and dataset revisions and records selected examples and hashes. Weights, corpus text, local adapter checkpoints and private application documents are excluded from Git.

The training script performs behavioral attention checks, evaluates fixed held-out masks, records every training update, saves adapter checkpoints locally and exports metrics plus all fixed-prompt samples into `results/`. Failed runs must not be reported as completed results.

Verified locally: four correctness tests passed, including target alignment, hidden-target non-leakage, future-token attention behavior and trainable parameter scope. Selected train/validation/test sequences have no exact token-sequence overlap. The report builder verifies matched seeds, masks, supervised token counts, training budgets and provenance across both completed runs. Peak allocated GPU memory was approximately 1.29 GiB on an RTX 3050 Laptop GPU; no Colab compute was used for this small-model pilot.

To reproduce the inference-only sampling comparison after the two local checkpoints exist:

```sh
python -s -m experiments.compare_sampling --repeats 3
```

This verifies checkpoint hashes and exact historical sample reproduction before comparing 12 fixed prompts per checkpoint. Both conditions retain greedy, highest-confidence-first selection, 32 output tokens and the historical EOS exclusion. The 1-token groups require 32 forwards versus 4 for 8-token groups. All 8 objective/sampling tests passed; no training or Colab compute was used. Repeated 4-token fragments remained common (shift: 69.0% vs 82.2%; same: 78.7% vs 82.5%), so lower repetition must not be equated with coherent generation. The raw JSON includes reveal traces and three identical-output timing repetitions for every condition.

## Historical hardware run

`experiments/colab_probe.py` and `experiments/colab_7b_benchmark.py` preserve the original experiment source. The 7B-shaped benchmark was run with Transformers **5.20.2** and PyTorch **2.11.0+cu130**, so use a separate matching Colab environment; the small-model requirements do not reproduce that historical software stack.

At batch 4 and length 1,024, the recorded throughput was approximately 2,950 input tokens/s on A100 and 4,444 input tokens/s on G4. Timing excludes data loading, checkpointing and evaluation. See the result files for every tested shape, timing sample and limitation. Provenance hashes identify the archived files; private notebook links and account information were removed from public copies.

## Attribution and contribution

Owner: HuangBlac. This is a learning and experimental project prepared with substantial Codex assistance for design, implementation, execution, checks and documentation. User decisions and interpretation should be distinguished from agent-produced code and results. The repository is not evidence of an independently developed algorithm or independently mastered large-model training.

The model is [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), under Apache-2.0. The corpus is [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), under the licenses stated in its dataset card. The prediction-shift experiment is motivated by [Dream](https://arxiv.org/abs/2508.15487); this repository does not copy Dream's training code, implement CART, or claim its reported results.
