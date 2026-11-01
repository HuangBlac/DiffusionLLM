"""P0 only: 20 updates, then replay update 20 in a fresh process from step 19.

Keeps the full FP32 optimizer checkpoint. No API judging or 200-step extension.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.full_parameter_pilot import (
    ROOT, ConditionalModel, evaluate, restore, save_checkpoint, sha256, synchronize, update,
)


def state_fingerprint(value):
    """Hash tensor values and nested optimizer state, not serialization metadata."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def fingerprints(model, optimizer, order, masks):
    return {"weights": state_fingerprint(model.state_dict()),
            "optimizer": state_fingerprint(optimizer.state_dict()),
            "data_order_rng": state_fingerprint(order.getstate()),
            "mask_rng": state_fingerprint(masks.get_state()),
            "torch_rng": state_fingerprint(torch.get_rng_state()),
            "cuda_rng": state_fingerprint(torch.cuda.get_rng_state_all())}


def write_report(path, value):
    path = Path(path)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["ar", "dlm"], required=True)
    parser.add_argument("--phase", choices=["train", "resume"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=420)
    args = parser.parse_args()
    if not 120 <= args.max_seconds <= 900:
        parser.error("Use a bounded 120–900 second P0 process budget")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("P0 requires BF16 CUDA")
    started = time.monotonic()
    deadline = started + args.max_seconds
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "latest.pt"
    report_file = args.output / f"{args.phase}.json"
    if report_file.exists() or (args.phase == "train" and any(args.output.iterdir())):
        raise FileExistsError("P0 outputs already exist; do not overwrite a prior attempt")
    data_path = ROOT / "data" / "pipeline-pilot.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if data["status"] != "observed_development_pilot":
        raise ValueError("Unexpected data status")
    model_dir = ROOT / "data" / "model"
    if sha256(model_dir / "model.safetensors") != data["model_sha256"]:
        raise ValueError("Base weight mismatch")
    torch.set_num_threads(4)
    torch.manual_seed(11)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    # A fixed math backend makes exact replay auditable; disclose it in timings.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    base = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                               dtype=torch.float32,
                                               attn_implementation="sdpa").cuda()
    model = ConditionalModel(base, args.method)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(.9, .95),
                                  weight_decay=.01, foreach=False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    order, masks = random.Random(11), torch.Generator().manual_seed(1011)
    identity = {"method": args.method, "seed": 11, "micro_batch": 2, "accumulation": 8,
                "lr": 2e-5, "warmup": 20, "betas": [.9, .95], "weight_decay": .01,
                "data_sha256": sha256(data_path), "weights_sha256": data["model_sha256"],
                "probe_sha256": sha256(__file__),
                "trainer_sha256": sha256(ROOT / "experiments" / "full_parameter_pilot.py"),
                "torch": str(torch.__version__), "transformers": transformers.__version__,
                "sdpa_backend": "math", "deterministic_algorithms": True}
    report = {"status": "running", "phase": args.phase, "identity": identity,
              "gpu": torch.cuda.get_device_name(), "python": platform.python_version(),
              "cuda": torch.version.cuda, "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "parameter_dtype": str(next(model.parameters()).dtype), "autocast": "bfloat16"}
    write_report(report_file, report)
    step, history = 0, []
    watched = model.base.model.layers[0].self_attn.q_proj.weight
    initial = watched.detach().clone()
    try:
        if args.phase == "resume":
            before_load = time.monotonic()
            step, history = restore(checkpoint, model, optimizer, order, masks, identity)
            report["restore_seconds"] = time.monotonic() - before_load
            if step != 19:
                raise ValueError("Replay must start from the step-19 checkpoint")
            expected = json.loads((args.output / "train.json").read_text())
            if expected["status"] != "completed":
                raise ValueError("Continuous reference run was not completed")
            report["restored_step"] = step
            report["restored_step19_matches"] = (
                fingerprints(model, optimizer, order, masks) == expected["step19_fingerprints"])
            if not report["restored_step19_matches"]:
                raise RuntimeError("Restored state differs from saved step 19")
        else:
            report["before"] = evaluate(model, data["validation"][:8], tokenizer.eos_token_id, deadline - 120)
            write_report(report_file, report)
        synchronize(torch.device("cuda"))
        torch.cuda.reset_peak_memory_stats()
        while step < 20:
            if time.monotonic() > deadline - 90:
                raise TimeoutError("P0 process guard: stop rather than exceed its allocation")
            tick = time.monotonic()
            stats = update(model, optimizer, data["train"], tokenizer.eos_token_id,
                           order, masks, 2, 8, 2e-5 * min((step + 1) / 20, 1))
            synchronize(torch.device("cuda"))
            step += 1
            stats.update(step=step, seconds=time.monotonic() - tick)
            history.append(stats)
            if step in (1, 5, 10, 15, 19, 20):
                print(json.dumps({"method": args.method, "phase": args.phase, **stats}), flush=True)
            if step == 19 and args.phase == "train":
                optimizer.zero_grad(set_to_none=True)
                report["step19_fingerprints"] = fingerprints(model, optimizer, order, masks)
                tick = time.monotonic()
                save_checkpoint(checkpoint, model, optimizer, step, order, masks, identity, history)
                report["save_step19_seconds"] = time.monotonic() - tick
                report["checkpoint_bytes"] = checkpoint.stat().st_size
                write_report(report_file, report)
        optimizer.zero_grad(set_to_none=True)
        report["train_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["train_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["step20_fingerprints"] = fingerprints(model, optimizer, order, masks)
        report["watched_parameter_max_change"] = (watched.detach() - initial).abs().max().item()
        report["history"] = history
        report["end_step"] = step
        if args.phase == "resume":
            report["step20_matches"] = report["step20_fingerprints"] == expected["step20_fingerprints"]
            keys = ("objective", "grad_norm", "input_tokens", "target_tokens", "supervised_tokens")
            report["step20_stats_match"] = all(history[-1][k] == expected["history"][-1][k] for k in keys)
            if not (report["step20_matches"] and report["step20_stats_match"]):
                raise RuntimeError("Step-20 replay differs from uninterrupted reference")
            tick = time.monotonic()
            save_checkpoint(checkpoint, model, optimizer, step, order, masks, identity, history)
            report["save_step20_seconds"] = time.monotonic() - tick
            report["checkpoint_bytes"] = checkpoint.stat().st_size
            report["checkpoint_sha256"] = sha256(checkpoint)
        base.gradient_checkpointing_disable()
        report["after"] = evaluate(model, data["validation"][:8], tokenizer.eos_token_id, deadline - 20)
        report["status"] = "completed" if report["after"]["complete"] else "partial"
        if report["watched_parameter_max_change"] == 0:
            raise RuntimeError("Watched pretrained parameter did not change")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), end_step=step, history=history)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(report_file, report)
        print(json.dumps({"method": args.method, "phase": args.phase,
                          "status": report["status"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
