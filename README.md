# DiffusionLLM

A small, reproducible exploration of adapting an autoregressive language model to masked denoising, with explicit experimental limits.

**Current status (2026-11-01):** a real-weight Qwen2.5-0.5B comparison is complete: shifted and same-position denoising, 200 updates each. Held-out denoising loss decreased, while greedy iterative generation remained highly repetitive. No completed 7B adaptation or Dream reproduction is claimed.

At 50% masking, held-out CE changed from **17.36 to 5.23** nats for shifted prediction and **15.20 to 5.49** for same-position prediction. These single-seed results cover 32 sequences and do not establish general language quality. [Full results and all fixed-prompt samples](docs/pilot-results.md).

## Experiments

| Experiment | Evidence | What it establishes |
| --- | --- | --- |
| 100-token absorbing-mask check and one synthetic GPU update | `results/hardware-2026-10-29/colab-check-results.json` | Empirical masking marginals and a working backward/update path |
| A100/G4, Qwen2.5-7B-shaped random weights, frozen BF16 base + FP32 LoRA | `results/hardware-2026-10-29/` | Short-run throughput and memory under the tested configuration |
| Qwen2.5-0.5B pretrained weights: shifted vs same-position denoising | [Protocol](docs/experiment-design.md), [results](docs/pilot-results.md) | Short adaptation reduces masked CE; repetitive generation remains a failure |

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

## Historical hardware run

`experiments/colab_probe.py` and `experiments/colab_7b_benchmark.py` preserve the original experiment source. The 7B-shaped benchmark was run with Transformers **5.20.2** and PyTorch **2.11.0+cu130**, so use a separate matching Colab environment; the small-model requirements do not reproduce that historical software stack.

At batch 4 and length 1,024, the recorded throughput was approximately 2,950 input tokens/s on A100 and 4,444 input tokens/s on G4. Timing excludes data loading, checkpointing and evaluation. See the result files for every tested shape, timing sample and limitation. Provenance hashes identify the archived files; private notebook links and account information were removed from public copies.

## Attribution and contribution

Owner: HuangBlac. This is a learning and experimental project prepared with substantial Codex assistance for design, implementation, execution, checks and documentation. User decisions and interpretation should be distinguished from agent-produced code and results. The repository is not evidence of an independently developed algorithm or independently mastered large-model training.

The model is [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), under Apache-2.0. The corpus is [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), under the licenses stated in its dataset card. The prediction-shift experiment is motivated by [Dream](https://arxiv.org/abs/2508.15487); this repository does not copy Dream's training code, implement CART, or claim its reported results.
