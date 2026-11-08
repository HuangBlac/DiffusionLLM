"""Inference-only comparison: reveal 1 vs 8 tokens with fixed pilot weights."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch
import transformers
from transformers import AutoTokenizer

from experiments.small_qwen import ROOT, generate as old_generate, load_model


# Fixed before inspecting the new outputs; first three are the original prompts.
PROMPTS = [
    "The purpose of science is",
    "Once upon a time, a small bird",
    "A language model learns to",
    "The city was founded in",
    "Water is essential for life because",
    "The researchers compared the two methods and found that",
    "Although the experiment was carefully designed,",
    "The main difference between a river and a lake is",
    "After the rain stopped, the children",
    "To understand the history of this region,",
    "The book describes how",
    "When the temperature falls below freezing,",
]


def file_hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def restore_adapter(model, state):
    """Require every trained tensor, then freeze the entire restored model."""
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(state) != set(expected):
        raise ValueError("Checkpoint keys do not exactly match trainable model tensors")
    for name, value in state.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"Checkpoint shape/dtype mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
    model.load_state_dict(state, strict=False)
    for name, parameter in expected.items():
        if not torch.equal(parameter.detach().cpu(), state[name]):
            raise RuntimeError(f"Restoration verification failed: {name}")
    model.requires_grad_(False)
    model.eval()
    return len(expected)


def repetition_metrics(ids):
    if not ids:
        raise ValueError("Expected a nonempty generated sequence")
    counts = Counter(ids)
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    longest = run = 1
    for previous, current in zip(ids, ids[1:]):
        run = run + 1 if previous == current else 1
        longest = max(longest, run)
    return {
        "repeated_4gram_fraction": 1 - len(set(grams)) / len(grams) if grams else None,
        "adjacent_equal_fraction": sum(a == b for a, b in zip(ids, ids[1:])) / (len(ids) - 1) if len(ids) > 1 else None,
        "dominant_token_fraction": max(counts.values()) / len(ids),
        "distinct_token_fraction": len(counts) / len(ids),
        "longest_identical_token_run": longest,
    }


@torch.inference_mode()
def decode(model, tokenizer, prompt, alignment, group_size, length=32):
    if alignment not in ("shift", "same") or group_size < 1 or length < 1:
        raise ValueError("Invalid alignment, group size or output length")
    device = next(model.parameters()).device
    bos = tokenizer.eos_token_id
    prefix = [bos] + tokenizer.encode(prompt, add_special_tokens=False)
    clean = torch.tensor([prefix + [bos] * length], dtype=torch.long, device=device)
    mask = torch.zeros_like(clean, dtype=torch.bool)
    mask[:, len(prefix):] = True
    trace = []
    while mask.any():
        positions = mask[0].nonzero().flatten()
        states = model.hidden(clean, mask)
        sources = positions - 1 if alignment == "shift" else positions
        logits = model.logits(states[0, sources])
        # Preserve the old pilot's special-token exclusions, including EOS.
        logits[:, tokenizer.all_special_ids] = -torch.inf
        logits[:, len(tokenizer):] = -torch.inf
        probabilities = logits.softmax(-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Nonfinite sampling probabilities")
        confidence, predictions = probabilities.max(-1)
        selected = confidence.topk(min(group_size, positions.numel())).indices
        chosen = positions[selected]
        clean[0, chosen] = predictions[selected]
        mask[0, chosen] = False
        trace.append({
            "round": len(trace) + 1,
            "output_positions": (chosen - len(prefix)).tolist(),
            "token_ids": predictions[selected].tolist(),
            "confidence": confidence[selected].tolist(),
            "remaining_masks": int(mask.sum()),
        })
    if clean[0, :len(prefix)].tolist() != prefix:
        raise RuntimeError("Prompt changed during sampling")
    if len(trace) != math.ceil(length / group_size):
        raise RuntimeError("Unexpected number of denoising forwards")
    ids = clean[0, len(prefix):].tolist()
    return {"completion": tokenizer.decode(ids), "generated_ids": ids,
            "forward_calls": len(trace), "trace": trace, "metrics": repetition_metrics(ids)}


def run_alignment(alignment, tokenizer, repeats):
    checkpoint = ROOT / "checkpoints" / f"{alignment}-200.pt"
    prior = json.loads((ROOT / "results" / f"qwen05b-{alignment}-200.json").read_text(encoding="utf-8"))
    original_hash = file_hash(checkpoint)
    if original_hash != prior["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash does not match the completed training record")
    model = load_model()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tensor_count = restore_adapter(model, state)
    # Verify that the reconstructed model reproduces all original fixed samples.
    old_samples = old_generate(model, tokenizer, tokenizer.eos_token_id, alignment)
    if old_samples != prior["samples_after"]:
        raise RuntimeError("Reloaded weights do not reproduce historical generation")
    for prompt, old in zip(PROMPTS[:3], old_samples):
        if decode(model, tokenizer, prompt, alignment, 4)["completion"] != old["completion"]:
            raise RuntimeError("New decoder does not reproduce the historical 4-token groups")
    print(f"{alignment}: checkpoint restored; historical samples reproduced", flush=True)
    for group in (1, 8):
        for _ in range(2):
            decode(model, tokenizer, PROMPTS[0], alignment, group)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    records = []
    for prompt_index, prompt in enumerate(PROMPTS):
        outputs, timings = {}, {1: [], 8: []}
        for repeat in range(repeats):
            order = (1, 8) if (prompt_index + repeat) % 2 == 0 else (8, 1)
            for group in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = decode(model, tokenizer, prompt, alignment, group)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                timings[group].append(elapsed)
                if group in outputs and result["generated_ids"] != outputs[group]["generated_ids"]:
                    raise RuntimeError("Greedy output changed between timing repetitions")
                outputs[group] = result
        for group in (1, 8):
            records.append({"prompt_id": prompt_index, "prompt": prompt, "tokens_per_round": group,
                            **outputs[group], "seconds": timings[group],
                            "median_seconds": statistics.median(timings[group])})
        print(f"{alignment}: prompt {prompt_index + 1}/{len(PROMPTS)} complete", flush=True)
    if file_hash(checkpoint) != original_hash:
        raise RuntimeError("Checkpoint changed during inference")
    for name, parameter in model.named_parameters():
        if name in state and not torch.equal(parameter.detach().cpu(), state[name]):
            raise RuntimeError("Adapter weights changed during inference")
    result = {"alignment": alignment, "checkpoint_sha256": original_hash,
              "restored_tensors": tensor_count, "historical_samples_reproduced": True,
              "checkpoint_and_adapter_unchanged": True,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "records": records}
    del model, state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def report(result):
    lines = ["# Fixed checkpoint sampling comparison", "",
             "Inference only on the existing 200-step checkpoints; no training or Colab compute. "
             "Twelve prompts were fixed before inspecting outputs; the first three are the historical prompts. "
             "Both methods greedily reveal the most confident masked positions. Only group size changes: 1 versus 8 tokens. "
             "Each output has 32 tokens, with all special tokens (including EOS) suppressed, matching the old fixed-length protocol.", "",
             "These are token groups, not necessarily words. This diagnostic cannot establish natural termination, "
             "general language quality, or whether more training would help. Repetition metrics are not coherence scores. "
             "Greedy timing repeats are identical generations, not independent quality samples.", "",
             "## Aggregate measurements", "",
             "| Checkpoint | Tokens per round | Forwards/output | Mean repeated 4-grams | Mean adjacent repetition | Mean dominant token share | Median per-prompt latency (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in result["runs"]:
        for group in (1, 8):
            rows = [r for r in run["records"] if r["tokens_per_round"] == group]
            avg = lambda key: statistics.mean(r["metrics"][key] for r in rows)
            lines.append(f"| {run['alignment']} | {group} | {rows[0]['forward_calls']} | "
                         f"{avg('repeated_4gram_fraction'):.1%} | {avg('adjacent_equal_fraction'):.1%} | "
                         f"{avg('dominant_token_fraction'):.1%} | {statistics.median(r['median_seconds'] for r in rows):.3f} |")
    lines += ["", "Latency includes tokenization, per-round trace extraction and synchronization in this diagnostic implementation; "
              "excludes model loading and warmup. It is not an optimized serving benchmark or a compute-matched quality comparison. "
              "The one-token condition uses 8x as many model forwards.", "", "## All outputs", ""]
    for run in result["runs"]:
        lines += [f"### {run['alignment']} checkpoint", ""]
        for prompt_index, prompt in enumerate(PROMPTS):
            lines += [f"#### {prompt_index + 1}. {prompt}", ""]
            for group in (1, 8):
                row = next(r for r in run["records"] if r["prompt_id"] == prompt_index and r["tokens_per_round"] == group)
                lines += [f"{group} token(s) per round:", "", "```text", row["completion"], "```", ""]
    lines += ["## Provenance and scope", "", "Both checkpoint hashes were checked against their original training records. "
              "Exact restoration of every adapter tensor and reproduction of all three historical samples were verified before measurement. "
              "The new decoder also reproduced the old four-token-per-round outputs. All parameters were frozen, no optimizer was created, "
              "and unchanged checkpoint/adapter values were verified after inference. Raw records contain token IDs, reveal order, confidence and all timing repeats.", "",
              "[Raw results](../../results/sampling-group-size-200.json).", ""]
    (ROOT / "docs" / "history" / "sampling-group-size-results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this timing experiment")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((ROOT / "results" / "data_manifest.json").read_text(encoding="utf-8"))
    weights_hash = file_hash(ROOT / "data" / "model" / "model.safetensors")
    if weights_hash != manifest["weights_sha256"]:
        raise RuntimeError("Base weights hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "data" / "model", local_files_only=True)
    result = {"date": "2026-11-01", "kind": "fixed-checkpoint inference diagnostic", "training_performed": False,
              "colab_used": False, "base_weights_sha256": weights_hash, "torch": torch.__version__,
              "transformers": transformers.__version__, "python": platform.python_version(),
              "gpu": torch.cuda.get_device_name(), "script_sha256": file_hash(__file__),
              "order": "highest confidence first", "token_selection": "greedy", "output_tokens": 32,
              "special_tokens": "all suppressed, including EOS, to match historical diagnostic",
              "timing_repeats": args.repeats, "warmups_per_group": 2, "prompts": PROMPTS, "runs": []}
    for alignment in ("shift", "same"):
        result["runs"].append(run_alignment(alignment, tokenizer, args.repeats))
        # Save completed checkpoint measurements even if a subsequent run fails.
        result["status"] = "completed" if len(result["runs"]) == 2 else "partial"
        (ROOT / "results" / "sampling-group-size-200.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report(result)
    print("Completed comparison and report", flush=True)


if __name__ == "__main__":
    main()
