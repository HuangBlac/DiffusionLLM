"""Colab resource and masked-language-model smoke test; no pretrained weights."""

import gc
import json
import platform
import time

import torch
import torch.nn.functional as F
from torch import nn


def resource_report() -> dict:
    """Report the actual runtime, without collecting credentials or user files."""
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        report.update(
            gpu=torch.cuda.get_device_name(0),
            gpu_total_gib=round(total_bytes / 2**30, 2),
            gpu_free_gib=round(free_bytes / 2**30, 2),
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    print(json.dumps(report, indent=2))
    return report


def masking_experiment() -> dict:
    """Compare the marginal mask counts, not independent samples as one path."""
    gen = torch.Generator().manual_seed(20261029)
    n, length = 10000, 100
    direct = torch.rand(n, length, generator=gen) < 0.5
    first = torch.rand(n, length, generator=gen) < 0.2
    incremental = torch.rand(n, length, generator=gen) < 0.375
    staged = first | incremental
    report = {
        "samples": n,
        "length": length,
        "expected_mask_count": 50,
        "direct_mean": direct.sum(1).float().mean().item(),
        "staged_mean": staged.sum(1).float().mean().item(),
        "direct_std": direct.sum(1).float().std().item(),
        "staged_std": staged.sum(1).float().std().item(),
        "absorbing_property": bool(torch.all(staged[first]).item()),
        "note": "Empirical marginal check; samples are not paired trajectories.",
    }
    print(json.dumps(report, indent=2))
    return report


class MaskPredictor(nn.Module):
    """One bidirectional Transformer layer with tied input/output embeddings."""

    def __init__(self, vocab: int = 58706, length: int = 100, width: int = 768):
        super().__init__()
        self.vocab = vocab
        self.token = nn.Embedding(vocab + 1, width)
        nn.init.normal_(self.token.weight, std=0.02)
        self.position = nn.Parameter(torch.zeros(1, length, width))
        self.block = nn.TransformerEncoderLayer(
            width, nhead=12, dim_feedforward=1536, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.block(self.token(ids) + self.position))
        return F.linear(hidden, self.token.weight[:self.vocab])


def gpu_training_smoke() -> dict:
    """Execute one update on synthetic tokens; this is not language training."""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU allocated. Select a GPU runtime in Colab first.")
    torch.manual_seed(20261029)
    torch.cuda.manual_seed_all(20261029)
    torch.cuda.reset_peak_memory_stats()
    model = MaskPredictor().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    clean = torch.randint(0, model.vocab, (1, 100), device="cuda")
    t = 0.5
    masked = torch.rand(clean.shape, device="cuda") < t
    if not masked.any():
        raise RuntimeError("No masked positions in this random draw; rerun.")
    damaged = clean.masked_fill(masked, model.vocab)
    torch.cuda.synchronize()
    started = time.perf_counter()
    before = model.position.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    logits = model(damaged)
    loss = F.cross_entropy(logits[masked], clean[masked], reduction="sum") / (100 * t)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss.")
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    delta = (model.position.detach() - before).abs().max().item()
    if delta == 0:
        raise RuntimeError("No parameter update was detected.")
    report = {
        "kind": "one-update synthetic smoke test, NOT pretrained LLM fine-tuning",
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_shape": list(damaged.shape),
        "hidden_shape": [1, 100, 768],
        "output_shape": list(logits.shape),
        "masked_count": masked.sum().item(),
        "loss": loss.item(),
        "gradient_norm": grad_norm.item(),
        "position_parameter_max_change": delta,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }
    print(json.dumps(report, indent=2))
    del model, optimizer, logits, loss, before
    gc.collect()
    torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    resource_report()
    masking_experiment()
    gpu_training_smoke()
