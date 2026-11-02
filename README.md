# DiffusionLLM

**Execution verification (2026-11-02):** the [single-cell Colab flow with durable results](docs/managed-colab-workflow.md) passed the CPU cloud test, including Linux descendant termination. Three saved files were fetched outside the runtime and matched their sizes/SHA-256; zero active sessions and zero CU/hour were verified. All 39 local tests pass; the execution manager has 93% statement coverage. This verifies JSON fixture persistence, not a trained neural checkpoint. The user approved a cumulative P0 cap of 4 CU by transferring 2 CU from the recovery allocation; the 180 CU overall cap is unchanged.

A small, reproducible exploration of adapting an autoregressive language model to masked denoising, with explicit experimental limits.

**Latest P0 status (2026-11-02):** the first attempt verified 20 full-parameter AR updates on A100 (8.71 GiB peak allocated memory), using approximately 0.54 CU. A subsequent attempt verified environment setup only; browser control failed and no new training/replay result was recovered from the saved notebook. The user reported zero active sessions afterward. The retry used approximately 1.28 CU, bringing cumulative P0 use to 1.82 of its 2 CU cap. GPU replay and DLM remain unverified; no checkpoint export is confirmed, and P1 has not started. [Partial results and evidence limits](docs/p0-resource-check-results.md).

**Overall scope:** [training and evaluation plan](docs/project-plan.md). The project includes both our own Qwen2.5-0.5B adaptation study and evaluation of mature pretrained diffusion models. The planned Dream/Qwen7B evaluation does not replace training; Clover-35B-A3B is an optional, unstarted extension.

**Current next run:** [first-round runbook](docs/first-round-runbook.md), a bounded full-parameter 0.5B AR/DLM development pilot (20 updates for resource/resume checks, then at most 200), with original AR and EOS-aware continuation baselines. DeepSeek Flash is selected for calibrated judging. The new entry point and local correctness tests are distinct from a completed real-weight training run; see [readiness evidence](docs/first-round-readiness.md). First-round limits are now 8 CU within the existing 180 CU, including all earlier P0 attempts, plus at most RMB 3 of the separate RMB 30 judge budget. The currently requested A100 run is P0 only; P1 requires completed GPU evidence.

**Project deliverables:** a coherent diffusion LLM note and a complete, reproducible real-text adaptation study. The current question is whether adapting pretrained AR weights produces a usable masked-denoising generator, followed by specific quality/cost or infilling comparisons. The existing 200-step comparison remains preliminary; the project does not claim a full-scale Dream reproduction. See [deliverables](docs/deliverables.md), the [real-text study design](docs/real-text-study.md), and the [opening learning draft](docs/diffusion-llm-note.md). The new study has only the partial P0 run described above.

**Planned mature-model evaluation (2026-11-02):** compare pretrained Dream-v0-Base-7B and Qwen2.5-7B on a fixed HellaSwag subset, with standard candidate scoring and a separately labeled full-ending generation variant. [Protocol and Colab budget](docs/dream-qwen-hellaswag-plan.md). This follows the first 0.5B pipeline pilot and has not run; its results will not be presented as our own 7B training outcomes.

**Current status (2026-11-01):** a real-weight Qwen2.5-0.5B comparison is complete: shifted and same-position denoising, 200 updates each. Held-out denoising loss decreased, while greedy iterative generation remained highly repetitive. No completed 7B adaptation or Dream reproduction is claimed.

At 50% masking, held-out CE changed from **17.36 to 5.23** nats for shifted prediction and **15.20 to 5.49** for same-position prediction. These single-seed results cover 32 sequences and do not establish general language quality. [Full results and all fixed-prompt samples](docs/pilot-results.md).

## Experiments

| Experiment | Evidence | What it establishes |
| --- | --- | --- |
| 100-token absorbing-mask check and one synthetic GPU update | `results/hardware-2026-10-29/colab-check-results.json` | Empirical masking marginals and a working backward/update path |
| A100/G4, Qwen2.5-7B-shaped random weights, frozen BF16 base + FP32 LoRA | `results/hardware-2026-10-29/` | Short-run throughput and memory under the tested configuration |
| Qwen2.5-0.5B pretrained weights: shifted vs same-position denoising | [Protocol](docs/experiment-design.md), [results](docs/pilot-results.md) | Short adaptation reduces masked CE; repetitive generation remains a failure |
| Fixed 200-step checkpoints: reveal 1 vs 8 tokens per round | [All outputs and measurements](docs/sampling-group-size-results.md) | Smaller reveal groups change repetition patterns but do not resolve degeneration |

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
