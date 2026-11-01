"""Bounded AR-to-masked-denoising pilot; not a Dream reproduction."""
import argparse
import gc
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261101
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.normal_(self.a, std=0.02)
        self.scale = 2.0

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.a), self.b) * self.scale).to(x.dtype)


def target_states(states, clean, masked, alignment):
    """Select only hidden states predicting corrupted targets; BOS is excluded."""
    mask = masked[:, 1:]
    aligned = states[:, :-1] if alignment == "shift" else states[:, 1:]
    return aligned[mask], clean[:, 1:][mask]


class Denoiser(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        base.requires_grad_(False)
        for parent in list(base.modules()):
            for name, child in list(parent.named_children()):
                if name in TARGETS and isinstance(child, nn.Linear):
                    setattr(parent, name, LoRALinear(child))
        # Virtual mask input; no pretrained vocabulary entry is repurposed.
        self.mask_embedding = nn.Parameter(base.get_input_embeddings().weight.detach().float().mean(0))

    def hidden(self, clean, masked=None, causal=False):
        embeds = self.base.get_input_embeddings()(clean)
        if masked is not None:
            embeds = torch.where(masked.unsqueeze(-1), self.mask_embedding.to(embeds.dtype), embeds)
        length = clean.shape[1]
        attention = torch.zeros((1, 1, length, length), device=embeds.device, dtype=embeds.dtype)
        if causal:
            attention.masked_fill_(torch.ones(length, length, device=embeds.device, dtype=torch.bool).triu(1),
                                   torch.finfo(embeds.dtype).min)
        return self.base.model(inputs_embeds=embeds, attention_mask={"full_attention": attention},
                               use_cache=False).last_hidden_state

    def logits(self, states):
        return self.base.lm_head(states).float()


def seeded(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model():
    seeded(SEED)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(ROOT / "data" / "model", local_files_only=True,
                                               dtype=dtype, attn_implementation="sdpa").cuda()
    base.config.use_cache = False
    return Denoiser(base)


def batch(ids, bos):
    return torch.tensor([[bos] + row for row in ids], dtype=torch.long, device="cuda")


def corrupt(clean, probability, generator):
    masked = torch.rand(clean.shape, generator=generator) < probability
    masked[:, 0] = False
    return masked.to(clean.device)


@torch.no_grad()
def attention_check(model, clean):
    changed = clean.clone()
    changed[:, -1] = (changed[:, -1] + 19) % model.base.config.vocab_size
    model.eval()
    result = {}
    for causal in (True, False):
        delta = (model.hidden(clean, causal=causal)[:, 0] - model.hidden(changed, causal=causal)[:, 0]).abs().max().item()
        result["causal" if causal else "bidirectional"] = delta
    if result["causal"] > 1e-5 or result["bidirectional"] <= 1e-5:
        raise RuntimeError(f"Attention behavioral check failed: {result}")
    return result


@torch.no_grad()
def evaluate(model, rows, bos, alignment):
    model.eval()
    result = {}
    for probability in (0.25, 0.5, 0.75):
        generator = torch.Generator().manual_seed(SEED + int(probability * 100))
        total, correct, count = 0.0, 0, 0
        for row in rows:
            clean = batch([row], bos)
            mask = corrupt(clean, probability, generator)
            states, labels = target_states(model.hidden(clean, mask), clean, mask, alignment)
            if labels.numel() == 0:
                continue
            logits = model.logits(states)
            total += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(-1) == labels).sum().item()
            count += labels.numel()
        result[str(probability)] = {"masked_ce_nats": total / count, "masked_accuracy": correct / count,
                                    "masked_tokens": count}
    return result


@torch.no_grad()
def ar_nll(model, rows, bos):
    model.eval()
    total, count = 0.0, 0
    for row in rows:
        clean = batch([row], bos)
        states = model.hidden(clean, causal=True)[:, :-1]
        logits = model.logits(states).flatten(0, 1)
        labels = clean[:, 1:].flatten()
        total += F.cross_entropy(logits, labels, reduction="sum").item()
        count += labels.numel()
    return {"next_token_ce_nats": total / count, "tokens": count,
            "note": "Different conditioning from masked CE; not a head-to-head quality score."}


@torch.no_grad()
def generate(model, tokenizer, bos, alignment, steps=8, length=32):
    model.eval()
    outputs = []
    for prompt in ("The purpose of science is", "Once upon a time, a small bird", "A language model learns to"):
        prefix = [bos] + tokenizer.encode(prompt, add_special_tokens=False)
        clean = torch.tensor([prefix + [bos] * length], device="cuda")
        mask = torch.zeros_like(clean, dtype=torch.bool)
        mask[:, len(prefix):] = True
        trace = []
        for step in range(steps):
            positions = mask[0].nonzero().flatten()
            if not positions.numel():
                break
            states = model.hidden(clean, mask)
            source = positions - 1 if alignment == "shift" else positions
            logits = model.logits(states[0, source])
            logits[:, tokenizer.all_special_ids] = -torch.inf
            logits[:, len(tokenizer):] = -torch.inf
            probabilities = logits.softmax(-1)
            confidence, predictions = probabilities.max(-1)
            reveal = math.ceil(positions.numel() / (steps - step))
            selected = confidence.topk(reveal).indices
            clean[0, positions[selected]] = predictions[selected]
            mask[0, positions[selected]] = False
            trace.append(int(mask.sum()))
        if mask.any():
            raise RuntimeError("Sampler left mask tokens")
        if clean[0, :len(prefix)].tolist() != prefix:
            raise RuntimeError("Sampler changed the prompt")
        outputs.append({"prompt": prompt, "completion": tokenizer.decode(clean[0, len(prefix):]),
                        "remaining_masks": trace, "steps": steps, "generated_tokens": length})
    return outputs


def run(alignment, steps, test_examples):
    data = {split: json.loads((ROOT / "data" / f"{split}.json").read_text())
            for split in ("train", "validation", "test")}
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "data" / "model", local_files_only=True)
    bos = tokenizer.eos_token_id
    model = load_model()
    params = [p for p in model.parameters() if p.requires_grad]
    result = {"status": "running", "alignment": alignment, "seed": SEED, "steps": steps,
              "batch_size": 1, "sequence_tokens": 128, "training_tokens": steps * 128,
              "test_examples": test_examples, "model": "Qwen/Qwen2.5-0.5B",
              "trainable_parameters": sum(p.numel() for p in params),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "transformers": transformers.__version__, "python": platform.python_version(),
              "dtype": str(next(model.base.parameters()).dtype), "learning_rate": 1e-4,
              "lora_rank": 8, "lora_alpha": 16, "mask_probability": "Uniform[0.1,1.0)",
              "objective": "sum of masked target CE / (128 * probability)",
              "limitations": ["Single seed, short adaptation; no Dream/CART reproduction.",
                              "WikiText may be present in the base model's pretraining data.",
                              "Shift and same-position receive equal data order, masks, steps and token exposure.",
                              "Truncated noise interval: not an estimate of the full diffusion ELBO.",
                              "Original AR CE and denoising CE have different conditioning."]}
    result["attention_check"] = attention_check(model, batch([data["test"][0]], bos))
    evaluation_rows = data["test"][:test_examples]
    result["before"] = evaluate(model, evaluation_rows, bos, alignment)
    result["ar_reference_before"] = ar_nll(model, evaluation_rows, bos)
    result["samples_before"] = generate(model, tokenizer, bos, alignment)
    print(json.dumps({"alignment": alignment, "before": result["before"]}), flush=True)
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01, foreach=False)
    model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.base.enable_input_require_grads()
    generator = torch.Generator().manual_seed(SEED + 1000)
    order = random.Random(SEED)
    losses = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    watched = next(m.b for m in model.modules() if isinstance(m, LoRALinear))
    initial = watched.detach().clone()
    for step in range(1, steps + 1):
        model.train()
        clean = batch([data["train"][order.randrange(len(data["train"]))]], bos)
        probability = 0.1 + 0.9 * torch.rand((), generator=generator).item()
        mask = corrupt(clean, probability, generator)
        optimizer.zero_grad(set_to_none=True)
        states, labels = target_states(model.hidden(clean, mask), clean, mask, alignment)
        logits = model.logits(states)
        loss = F.cross_entropy(logits, labels, reduction="sum") / (128 * probability)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        grad = nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append({"step": step, "loss": loss.item(), "mask_probability": probability,
                       "masked_tokens": labels.numel(), "gradient_norm": grad.item()})
        if step == 1 or step % 25 == 0:
            print(json.dumps({"alignment": alignment, **losses[-1],
                              "elapsed_seconds": round(time.perf_counter() - start, 1)}), flush=True)
    torch.cuda.synchronize()
    result["training_seconds"] = time.perf_counter() - start
    result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    result["adapter_max_change"] = (watched.detach() - initial).abs().max().item()
    if result["adapter_max_change"] == 0:
        raise RuntimeError("No adapter update detected")
    optimizer.zero_grad(set_to_none=True)
    model.base.gradient_checkpointing_disable()
    result["after"] = evaluate(model, evaluation_rows, bos, alignment)
    result["validation_after"] = evaluate(model, data["validation"][:16], bos, alignment)
    result["samples_after"] = generate(model, tokenizer, bos, alignment)
    result["training_log"] = losses
    checkpoint = ROOT / "checkpoints" / f"{alignment}-{steps}.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save({name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}, checkpoint)
    result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result["status"] = "completed"
    path = ROOT / "results" / f"qwen05b-{alignment}-{steps}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path.name}", flush=True)
    del optimizer, model, params
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", choices=["shift", "same", "both"], default="both")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--test-examples", type=int, default=32)
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.test_examples <= 64:
        parser.error("steps must be positive; test-examples must be in [1,64]")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    for alignment in (["shift", "same"] if args.alignment == "both" else [args.alignment]):
        run(alignment, args.steps, args.test_examples)


if __name__ == "__main__":
    main()
