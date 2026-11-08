# Qwen2.5 0.5B masked denoising pilot

> 历史实验／过程档案：正文状态、预算与操作指令仅对应文中日期，不作为现行执行依据。2026-11-06整理后的当前状态见[项目进度](../progress.md)，P2执行约束见[P2规程](../current/p2-readiness-2026-11-05.md)。原始测量和失败证据保留。


Completed on 2026-11-01. Single seed; 200 updates and 25,600 token exposures per condition. Both conditions start from the same real pretrained weights, use the same trainable parameter set, paragraph order and corruptions. The fixed final checkpoints are evaluated on 32 held-out 128-token sequences. Held out refers to adaptation, not to the unknown pretraining corpus.

## Denoising results

| Prediction | Mask probability | CE before | CE after | Accuracy before | Accuracy after | Masked test tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shift | 0.25 | 16.1457 | 4.1443 | 0.19% | 27.85% | 1027 |
| shift | 0.5 | 17.3628 | 5.2338 | 0.00% | 16.99% | 2072 |
| shift | 0.75 | 17.9398 | 6.1888 | 0.03% | 10.31% | 3035 |
| same | 0.25 | 12.1688 | 4.5300 | 1.07% | 22.78% | 1027 |
| same | 0.5 | 15.1964 | 5.4933 | 0.48% | 14.58% | 2072 |
| same | 0.75 | 17.3089 | 6.2732 | 0.03% | 9.13% | 3035 |

CE is cross-entropy in nats per masked target, not perplexity or exact likelihood. The prediction alignments have different initial behaviors. These numbers do not demonstrate general language quality.

## Runtime and controls

| Prediction | Training seconds | Peak allocated GiB | Trainable parameters |
| --- | ---: | ---: | ---: |
| shift | 92.3 | 1.289 | 4,400,000 |
| same | 62.3 | 1.286 | 4,400,000 |

Hardware: NVIDIA GeForce RTX 3050 Laptop GPU. Original causal-model next-token CE: 3.2420 nats. That reference sees different context from the denoising task and cannot be directly ranked against masked CE.

Timing covers the training loop, including logging and device synchronization; excludes loading and evaluation. Peak memory covers training. Equal update/token budgets are not a claim of equal wall-clock compute.

## Fixed prompt samples after adaptation

Each continuation uses 32 tokens and eight rounds of confidence-based greedy reveal. All three predetermined prompts are included for each condition; none was selected for fluency. Before-adaptation samples and per-round remaining-mask counts are retained in the raw JSON.

### shift

Prompt: The purpose of science is

```text
 to to to to understand the the nature , , , , , , , , , , , , , , , , , , , , , , , ,
```

Prompt: Once upon a time, a small bird

```text
 was was a , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

Prompt: A language model learns to

```text
 recognize the the language , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

### same

Prompt: The purpose of science is

```text
 to to the , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

Prompt: Once upon a time, a small bird

```text
 was caught , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

Prompt: A language model learns to

```text
 understand the the . . . . . . . . . The . . . . . . . . The language model is . . . . . . .
```

## Scope and next decision

This is an AI-assisted implementation and execution of a small adaptation pilot. It does not reproduce Dream/CART, validate 7B adaptation, establish superiority over AR, or demonstrate independent mastery by the repository owner. A training or held-out loss improvement can coexist with poor iterative generation. Before scaling, inspect the samples and explain the alignment/loss/sampling implementation, then repeat across seeds and larger held-out sets.

See [protocol](experiment-design.md), [shift raw results](../../results/qwen05b-shift-200.json), [same-position raw results](../../results/qwen05b-same-200.json), and [data manifest](../../results/data_manifest.json).
