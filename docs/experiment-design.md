# Small-model AR → masked-denoising pilot

Protocol fixed before training on 2026-11-01. The user selected a small-model comparison before deciding whether to extend to 7B.

## Question

With the same pretrained Qwen2.5-0.5B weights, data, masks and number of updates, how does preserving next-position prediction compare with same-position prediction during short masked-denoising adaptation?

The shift variant uses hidden state `i-1` to predict corrupted token `i`; the same-position variant uses hidden state `i`. Both use fully bidirectional attention. A fixed BOS precedes all 128 target tokens and is never masked. Corrupted targets are replaced at the embedding level with a trainable mask vector: their original identities cannot enter the network.

## Fixed settings

- Model: Qwen/Qwen2.5-0.5B, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- Data: Salesforce/wikitext, revision `b08601e04326c79dfdd32d625aee71d232d685c3`, wikitext-2-raw-v1.
- First 1,024 eligible training paragraphs, first 64 validation and 64 test paragraphs. Each has at least 128 tokens; use the first 128. Remove exact full-paragraph duplicates and identical 128-token prefixes across these selected splits. Record source row IDs and token-file hashes.
- Two fresh copies of the same frozen model; FP32 LoRA rank 8, alpha 16 on all seven attention/MLP linear projections, plus one trainable mask vector.
- Seed 20261101; batch size 1; 200 updates per condition; 25,600 training token exposures per condition. Identical random paragraph order and Bernoulli masks.
- AdamW, learning rate 1e-4, weight decay 0.01, gradient clipping 1.0. No early stopping or tuning against test results.
- Sample masking probability uniformly in [0.1,1.0); loss is summed masked CE divided by `128 * probability`. This truncated interval is a practical pilot objective, not the full diffusion ELBO.

## Measurements

1. Before/after masked CE and masked-token accuracy at masking probabilities 0.25, 0.50 and 0.75 on the first 32 selected test examples. The same test masks are reused across conditions and checkpoints.
2. Final-checkpoint validation metrics on 16 selected validation examples, reported separately. Validation does not select a checkpoint in this fixed-budget pilot.
3. Original causal model's next-token CE on the same test text, as context only: its conditioning differs, so it cannot be ranked directly against masked CE.
4. Three fixed prompts, 32 generated tokens, eight rounds of confidence-based greedy reveal, before and after adaptation. Preserve the prompt and report all samples; these are diagnostic examples, not a benchmark of language quality.
5. Actual GPU, versions, elapsed training time, peak allocated GPU memory, update verification and checkpoint hash.

## Interpretation limits

This is a single-seed, small-data pilot. WikiText may occur in the base model's pretraining data; “held out” refers to this adaptation run. Different prediction alignments have different initial behavior. Their equal step/token exposure is not equal measured wall-clock compute. A lower masked CE does not establish a better generator or superiority over autoregressive models. The project does not implement CART or reproduce Dream's full training recipe.

The next step depends on the results: inspect failures and obtain repeatable gains across seeds before increasing model size. No 7B adaptation is authorized by the small-model decision alone.

## Sources

- [Qwen model card and license](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [WikiText dataset card and licenses](https://huggingface.co/datasets/Salesforce/wikitext)
- [Dream paper, §4.1–4.2](https://arxiv.org/abs/2508.15487): motivation for preserving prediction shift; CART is outside this pilot.
- [Dream official implementation](https://github.com/DreamLM/Dream)

The experimental design and implementation were prepared with Codex assistance. A completed automated run is not evidence of the applicant's independent mastery.
