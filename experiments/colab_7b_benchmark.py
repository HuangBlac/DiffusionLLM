"""Qwen2.5-7B-shaped BF16 LoRA hardware benchmark, NOT pretrained adaptation."""

import gc
import hashlib
import json
import platform
import statistics
import time

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

SEED = 20261029
WARMUP = 3
MEASURED = 10
CASES = [(1, 512), (1, 1024), (4, 1024)]
RANK = 16
ALPHA = 32
CONFIG = dict(
    vocab_size=152064, hidden_size=3584, intermediate_size=18944,
    num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4,
    max_position_embeddings=131072, rms_norm_eps=1e-6,
    rope_theta=1000000.0, tie_word_embeddings=False, attention_dropout=0.0,
    bos_token_id=151643, eos_token_id=151643, use_cache=False,
    use_sliding_window=False,
)


def emit(kind: str, value: dict) -> None:
    print(kind + " " + json.dumps(value, ensure_ascii=False), flush=True)


class LoRALinear(nn.Module):
    """Frozen BF16 linear projection plus trainable FP32 low-rank matrices."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(RANK, base.in_features, device="cuda", dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, RANK, device="cuda", dtype=torch.float32))
        nn.init.normal_(self.a, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(x.float(), self.a), self.b)
        return self.base(x) + (update * (ALPHA / RANK)).to(x.dtype)


def build_model() -> tuple[nn.Module, list[nn.Parameter]]:
    config = Qwen2Config(**CONFIG)
    config._attn_implementation = "sdpa"
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            model = Qwen2ForCausalLM(config)
    finally:
        torch.set_default_dtype(old_dtype)
    model.requires_grad_(False)
    targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child))
    for layer in model.model.layers:
        layer.self_attn.is_causal = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model, [p for p in model.parameters() if p.requires_grad]


def hidden(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    # Transformers 5.x accepts an explicit mapping and skips causal mask construction.
    return model.model(
        input_ids=ids, attention_mask={"full_attention": None}, use_cache=False,
    ).last_hidden_state


def check_bidirectional(model: nn.Module) -> float:
    """Behavioral check: changing the final token must affect the first position."""
    model.eval()
    with torch.no_grad():
        ids = torch.arange(32, device="cuda").unsqueeze(0)
        initial = hidden(model, ids)[:, 0].clone()
        ids[:, -1] = 101
        delta = (hidden(model, ids)[:, 0] - initial).abs().max().item()
    if delta <= 1e-6:
        raise RuntimeError("Future-token influence check failed: attention may still be causal.")
    model.train()
    return delta


def run_case(model: nn.Module, trainable: list[nn.Parameter], optimizer, batch: int, length: int) -> dict:
    torch.manual_seed(SEED + batch + length)
    # The last vocabulary ID is a synthetic MASK; no language/tokenizer claims are made.
    clean = torch.randint(0, CONFIG["vocab_size"] - 1, (batch, length), device="cuda")
    mask = torch.rand(batch, length, device="cuda") < 0.5
    if not mask.any():
        raise RuntimeError("No masked targets in this draw.")
    damaged = clean.masked_fill(mask, CONFIG["vocab_size"] - 1)
    watched = next(m.b for m in model.modules() if isinstance(m, LoRALinear))
    before = watched.detach().clone()

    def step() -> tuple[float, float]:
        optimizer.zero_grad(set_to_none=True)
        states = hidden(model, damaged)
        # Only masked logits are materialized; the objective equals masking full logits.
        logits = model.lm_head(states[mask])
        loss = F.cross_entropy(logits.float(), clean[mask], reduction="sum") / (batch * length * 0.5)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach().item(), grad.item()

    for _ in range(WARMUP):
        step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(MEASURED):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss, grad = step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    change = (watched.detach() - before).abs().max().item()
    if not (change > 0 and torch.isfinite(torch.tensor(loss))):
        raise RuntimeError("Invalid loss or no adapter update.")
    result = dict(
        batch=batch, sequence_length=length, warmup_steps=WARMUP, measured_steps=MEASURED,
        median_step_seconds=statistics.median(timings),
        mean_step_seconds=statistics.mean(timings),
        min_step_seconds=min(timings), max_step_seconds=max(timings),
        input_tokens_per_second=batch * length * MEASURED / sum(timings),
        supervised_tokens_per_second=int(mask.sum()) * MEASURED / sum(timings),
        masked_targets=int(mask.sum()), final_loss=loss, gradient_norm=grad,
        adapter_max_change=change,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        step_times_seconds=timings,
    )
    optimizer.zero_grad(set_to_none=True)
    return result


def main() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a GPU runtime.")
    if not transformers.__version__.startswith("5."):
        raise RuntimeError("This benchmark's explicit attention mapping requires Transformers 5.x.")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    result = dict(
        benchmark="Qwen2.5-7B full architecture, random weights, BF16 base + FP32 LoRA",
        quality_claim="None: hardware benchmark only, synthetic inputs, not pretrained adaptation",
        gpu=torch.cuda.get_device_name(), total_gib=total / 2**30, initial_free_gib=free / 2**30,
        python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
        cuda=torch.version.cuda, compute_capability=list(torch.cuda.get_device_capability()),
        config=CONFIG, lora_rank=RANK, lora_alpha=ALPHA,
        optimizer="AdamW FP32 adapter states, foreach=False", attention="bidirectional SDPA",
        gradient_checkpointing="non-reentrant", mask_probability=0.5,
        config_sha256=hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest(),
        cases=[],
    )
    emit("HARDWARE", result)
    model, trainable = build_model()
    result["total_parameters"] = sum(p.numel() for p in model.parameters())
    result["trainable_parameters"] = sum(p.numel() for p in trainable)
    result["future_token_influence_max_delta"] = check_bidirectional(model)
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, foreach=False)
    emit("MODEL_READY", {k: result[k] for k in ("total_parameters", "trainable_parameters", "future_token_influence_max_delta")})
    for batch, length in CASES:
        emit("CASE_START", dict(batch=batch, sequence_length=length))
        try:
            case = run_case(model, trainable, optimizer, batch, length)
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            case = dict(batch=batch, sequence_length=length, status="out_of_memory")
        result["cases"].append(case)
        emit("CASE_RESULT", case)
        gc.collect()
        torch.cuda.empty_cache()
    emit("BENCHMARK_RESULT", result)
    del optimizer, model, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    benchmark_result = main()
