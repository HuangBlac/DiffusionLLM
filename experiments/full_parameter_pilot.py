"""Bounded conditional AR/DLM pilot, separate from the historical LoRA runs.

The local smoke test uses a tiny random model. Real 0.5B runs require CUDA,
explicit time limits, and a prepared pilot manifest. This is not Dream training.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pack(rows, bos, device):
    """Padding is never a target or an attention key; prefix stays visible."""
    width = max(len(row["ids"]) for row in rows) + 1
    ids = torch.full((len(rows), width), bos, dtype=torch.long, device=device)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    targets = torch.zeros_like(valid)
    for i, row in enumerate(rows):
        n, p = len(row["ids"]), row["prefix_length"]
        if not 0 < p < n:
            raise ValueError("Each example needs a prefix and at least one target")
        ids[i, 1:n + 1] = torch.tensor(row["ids"], device=device)
        valid[i, :n + 1] = True
        targets[i, p + 1:n + 1] = True
    return ids, valid, targets


class ConditionalModel(nn.Module):
    def __init__(self, base, method):
        super().__init__()
        if method not in ("ar", "dlm"):
            raise ValueError(method)
        self.base, self.method = base, method
        base.requires_grad_(True)
        if method == "dlm":
            self.mask_embedding = nn.Parameter(
                base.get_input_embeddings().weight.detach().float().mean(0))

    def hidden(self, ids, valid, masked):
        embeds = self.base.get_input_embeddings()(ids)
        if self.method == "dlm":
            embeds = torch.where(masked[..., None], self.mask_embedding, embeds)
        elif masked.any():
            raise ValueError("AR inputs must be clean")
        length = ids.shape[1]
        allowed = valid[:, None, None, :].expand(-1, 1, length, -1)
        if self.method == "ar":
            allowed = allowed & torch.ones(length, length, device=ids.device,
                                           dtype=torch.bool).tril()
        attention = torch.zeros(allowed.shape, device=ids.device, dtype=embeds.dtype)
        attention.masked_fill_(~allowed, torch.finfo(embeds.dtype).min)
        return self.base.model(inputs_embeds=embeds,
                               attention_mask={"full_attention": attention},
                               use_cache=False).last_hidden_state

    def loss(self, batch, generator, probability=None):
        ids, valid, targets = batch
        if self.method == "dlm":
            p = (0.1 + 0.9 * torch.rand((ids.shape[0], 1), generator=generator)
                 if probability is None else torch.full((ids.shape[0], 1), probability))
            masked = (torch.rand(ids.shape, generator=generator) < p).to(ids.device) & targets
        else:
            p = torch.ones((ids.shape[0], 1))
            masked = torch.zeros_like(targets)
        selected = masked if self.method == "dlm" else targets
        states = self.hidden(ids, valid, masked)[:, :-1]
        selected = selected[:, 1:]
        logits = self.base.lm_head(states[selected]).float()
        labels = ids[:, 1:][selected]
        losses = F.cross_entropy(logits, labels, reduction="none")
        # Mean of per-example, length-normalized losses, also for unequal tails.
        weights = (1.0 / (targets.sum(1, keepdim=True).cpu() * p)).to(ids.device)
        weights = weights.expand_as(ids)[:, 1:][selected]
        objective = (losses * weights).sum() / ids.shape[0]
        return objective, {"ce_sum": losses.detach().sum().item(),
                           "correct": int((logits.argmax(-1) == labels).sum()),
                           "count": labels.numel(),
                           "target_tokens": int(targets.sum()),
                           "input_tokens": int(valid.sum()) - ids.shape[0]}


def precision(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def update(model, optimizer, rows, bos, order, masks, micro_batch, accumulation, lr):
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = lr
    total, inputs, targets, supervised = 0., 0, 0, 0
    for _ in range(accumulation):
        chosen = [rows[order.randrange(len(rows))] for _ in range(micro_batch)]
        with precision(device):
            loss, counts = model.loss(pack(chosen, bos, device), masks)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite objective")
        (loss / accumulation).backward()
        total += loss.item() / accumulation
        inputs += counts["input_tokens"]
        targets += counts["target_tokens"]
        supervised += counts["count"]
    grad = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return {"objective": total, "grad_norm": float(grad), "input_tokens": inputs,
            "target_tokens": targets, "supervised_tokens": supervised}


def save_checkpoint(path, model, optimizer, step, order, masks, identity, history):
    """Atomic replacement; include optimizer and RNG rather than weights only."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "order": order.getstate(), "masks": masks.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "identity": identity, "history": history}, temporary)
    os.replace(temporary, path)


def restore(path, model, optimizer, order, masks, identity):
    # Only load checkpoints created locally by this entry point.
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["identity"] != identity:
        raise ValueError("Resume configuration/data/code differs from checkpoint")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    order.setstate(state["order"])
    masks.set_state(state["masks"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"], state["history"]


@torch.no_grad()
def evaluate(model, rows, bos, deadline=math.inf):
    model.eval()
    device = next(model.parameters()).device
    result = {}
    for rate in ([.25, .5, .75, .9, 1.] if model.method == "dlm" else [1.]):
        masks = torch.Generator().manual_seed(20261102 + int(rate * 100))
        total, count, correct = 0., 0, 0
        evaluated = 0
        for row in rows:
            if time.monotonic() >= deadline:
                break
            with precision(device):
                _, stats = model.loss(pack([row], bos, device), masks, rate)
            total += stats["ce_sum"]
            count += stats["count"]
            correct += stats["correct"]
            evaluated += 1
        result[str(rate)] = {"ce_nats": total / count if count else None,
                             "accuracy": correct / count if count else None,
                             "tokens": count, "examples": evaluated}
    return {"kind": "conditional_ar_nll" if model.method == "ar" else "masked_ce_not_nll",
            "rates": result, "complete": all(r["examples"] == len(rows) for r in result.values())}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def continuation(model, tokenizer, row, length=128, steps=16):
    if not 1 <= steps <= length:
        raise ValueError("Need 1 <= denoising steps <= canvas length")
    model.eval()
    device = next(model.parameters()).device
    bos = tokenizer.eos_token_id
    prefix = [bos] + row["ids"][:row["prefix_length"]]
    ids = torch.tensor([prefix], device=device)
    synchronize(device)
    start = time.perf_counter()
    with precision(device):
        if model.method == "ar":
            forbidden = [i for i in tokenizer.all_special_ids if i != bos]
            forbidden += list(range(len(tokenizer), model.base.config.vocab_size))
            result = model.base.generate(ids, attention_mask=torch.ones_like(ids),
                                         do_sample=False, max_new_tokens=length,
                                         eos_token_id=bos, pad_token_id=bos, use_cache=True,
                                         suppress_tokens=forbidden)
            output = result[0, len(prefix):].tolist()
            forwards = len(output)
        else:
            ids = torch.tensor([prefix + [bos] * length], device=device)
            valid = torch.ones_like(ids, dtype=torch.bool)
            masked = torch.zeros_like(valid)
            masked[:, len(prefix):] = True
            forbidden = [i for i in tokenizer.all_special_ids if i != bos]
            for step in range(steps):
                positions = masked[0].nonzero().flatten()
                states = model.hidden(ids, valid, masked)
                logits = model.base.lm_head(states[0, positions - 1]).float()
                logits[:, forbidden] = -torch.inf
                logits[:, len(tokenizer):] = -torch.inf
                confidence, predictions = logits.softmax(-1).max(-1)
                k = math.ceil(len(positions) / (steps - step))
                chosen = confidence.topk(k).indices
                ids[0, positions[chosen]] = predictions[chosen]
                masked[0, positions[chosen]] = False
            if masked.any() or ids[0, :len(prefix)].tolist() != prefix:
                raise RuntimeError("Incomplete canvas or modified prefix")
            output = ids[0, len(prefix):].tolist()
            forwards = steps
    synchronize(device)
    seconds = time.perf_counter() - start
    eos = output.index(bos) if bos in output else None
    delivered = output if eos is None else output[:eos]
    return {"source_row": row["source_row"], "prompt": tokenizer.decode(prefix[1:]),
            "text": tokenizer.decode(delivered), "token_ids": output,
            "delivered_tokens": len(delivered), "eos_index": eos,
            "length_cap": length, "forwards": forwards, "seconds": seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "pipeline-pilot.json")
    parser.add_argument("--method", choices=["ar", "dlm"], required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--eval-examples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Original AR baseline only: no optimizer updates or checkpoint")
    args = parser.parse_args()
    if not (1 <= args.steps <= 200 and 1 <= args.eval_examples <= 32 and
            args.micro_batch > 0 and args.accumulation > 0 and
            math.isfinite(args.lr) and 0 < args.lr <= 1e-3 and
            math.isfinite(args.max_seconds) and args.max_seconds >= 180):
        parser.error("Invalid bounded pilot settings; need >=180 s for saving/exit")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Real pilot requires a BF16 CUDA GPU; use CPU unit tests locally")
    if args.evaluate_only:
        if args.method != "ar" or args.resume:
            parser.error("--evaluate-only is for a fresh original AR baseline")
        args.steps = 0
    start = time.monotonic()
    deadline = start + args.max_seconds
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "latest.pt"
    if not args.resume and any(args.output.iterdir()):
        raise FileExistsError("Fresh runs require an empty output directory")
    if checkpoint.exists() != args.resume:
        raise ValueError("Use a new output directory, or --resume for an existing checkpoint")
    data = json.loads(args.data.read_text(encoding="utf-8"))
    if data["status"] != "observed_development_pilot":
        raise ValueError("This entry is only for the separate development pilot")
    model_dir = ROOT / "data" / "model"
    if sha256(model_dir / "model.safetensors") != data["model_sha256"]:
        raise ValueError("Model weight digest mismatch")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                               dtype=torch.float32,
                                               attn_implementation="sdpa").to(device)
    model = ConditionalModel(base, args.method)
    if args.method == "dlm":
        model.mask_embedding.data = model.mask_embedding.data.to(device)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95),
                                  weight_decay=.01, foreach=False)
    order = random.Random(args.seed)
    masks = torch.Generator().manual_seed(args.seed + 1000)
    identity = {"method": args.method, "seed": args.seed, "lr": args.lr,
                "micro_batch": args.micro_batch, "accumulation": args.accumulation,
                "data_sha256": sha256(args.data), "script_sha256": sha256(__file__),
                "model_sha256": data["model_sha256"], "warmup_updates": 20,
                "torch": str(torch.__version__), "transformers": transformers.__version__,
                "objective": "conditional AR CE or truncated Uniform[.1,1) weighted masked CE"}
    step, history = (restore(checkpoint, model, optimizer, order, masks, identity)
                     if args.resume else (0, []))
    if step >= args.steps and not args.evaluate_only:
        raise ValueError("Requested step must exceed restored step")
    report = {"identity": identity, "start_step": step, "requested_step": args.steps,
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "python": platform.python_version(), "transformers": transformers.__version__,
              "precision": "FP32 trainable weights/Adam states, BF16 autocast",
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "status": "running", "stage": "observed_development_pilot"}
    dev = data["validation"][:args.eval_examples]
    torch.cuda.reset_peak_memory_stats()
    report["before"] = evaluate(model, dev, tokenizer.eos_token_id, deadline - 120)
    report["samples_before"] = []
    # Original AR is measured before its first update, never reconstructed from a trained model.
    if step == 0 and args.method == "ar" and not args.evaluate_only:
        for row in dev:
            if time.monotonic() > deadline - 120:
                break
            report["samples_before"].append(continuation(model, tokenizer, row))
    while step < args.steps and time.monotonic() < deadline - 120:
        synchronize(device)
        tick = time.perf_counter()
        stats = update(model, optimizer, data["train"], tokenizer.eos_token_id,
                       order, masks, args.micro_batch, args.accumulation,
                       args.lr * min((step + 1) / 20, 1.))
        synchronize(device)
        step += 1
        stats.update(step=step, seconds=time.perf_counter() - tick)
        history.append(stats)
        if step % 10 == 0 or step == 1:
            print(json.dumps(stats), flush=True)
        if step % 20 == 0:
            save_checkpoint(checkpoint, model, optimizer, step, order, masks, identity, history)
    optimizer.zero_grad(set_to_none=True)
    if not args.evaluate_only:
        save_checkpoint(checkpoint, model, optimizer, step, order, masks, identity, history)
    base.gradient_checkpointing_disable()
    report["end_step"] = step
    report["train_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["history"] = history
    report["samples_after"] = []
    if time.monotonic() < deadline - 90:
        report["after"] = evaluate(model, dev, tokenizer.eos_token_id, deadline - 60)
    for rounds in ([16, 128] if args.method == "dlm" else [1]):
        for row in dev:
            if time.monotonic() >= deadline - 60:
                break
            report["samples_after"].append(continuation(model, tokenizer, row, steps=rounds))
    expected = len(dev) * (2 if args.method == "dlm" else 1)
    report["status"] = ("completed" if step == args.steps and report["before"]["complete"] and
                         report.get("after", {}).get("complete") and
                         len(report["samples_after"]) == expected and
                         (args.evaluate_only or args.method != "ar" or report["start_step"] != 0 or
                          len(report["samples_before"]) == len(dev)) else "partial")
    report["checkpoint_sha256"] = None if args.evaluate_only else sha256(checkpoint)
    report["elapsed_seconds"] = time.monotonic() - start
    report["note"] = "Development only; time guard is best effort, not a Colab billing cutoff."
    destination = args.output / f"report-{report['start_step']}-{step}.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "step": step,
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
