"""Fixed-configuration learning run; import does not start training."""
import math
import time
from pathlib import Path

import torch
import transformers

from experiments import full_parameter_pilot as pilot
from experiments import p2_paired_training as paired


def identity(method, training_path, development_path, model_sha256, model_files=None):
    if method not in ('ar', 'dlm') or not model_sha256 or not model_files:
        raise ValueError('Method and verified base/tokenizer hashes are required')
    root = Path(__file__).parent
    return {'experiment': 'learning_run_512_v1', 'method': method,
            'base': 'Qwen/Qwen2.5-0.5B', 'model_sha256': model_sha256,
            'model_files_sha256': dict(model_files),
            'training_sha256': pilot.sha256(training_path),
            'development_sha256': pilot.sha256(development_path),
            'training_source_sha256': pilot.sha256(__file__),
            'evaluation_source_sha256': pilot.sha256(root / 'learning_run_evaluation.py'),
            'core_sha256': pilot.sha256(root / 'full_parameter_pilot.py'),
            'canvas_sha256': pilot.sha256(root / 'p2_canvas_ablation.py'),
            'paired_sha256': pilot.sha256(root / 'p2_paired_training.py'),
            'torch': str(torch.__version__), 'transformers': transformers.__version__,
            'canvas_policy': 'masked_padding_canvas' if method == 'dlm' else 'causal',
            'prefix_length': 128, 'target_capacity': 128, 'seed': 11, 'steps': 512,
            'micro_batch': 2, 'accumulation': 8, 'lr': 2e-5, 'warmup': 20,
            'betas': [0.9, 0.95], 'weight_decay': 0.01, 'clip': 1.0}


def run_training(model, optimizer, train, development, tokenizer, locked_identity, *,
                 deadline, reserve_seconds, checkpoint, evaluate, resume=None):
    """Callbacks own durable saving and bounded evaluation; deadline is monotonic.

    A timeout raises and never implies completion. A checkpoint is saved before each
    evaluation, leaving the caller responsible for saving on interruption as needed.
    """
    if not math.isfinite(deadline) or not math.isfinite(reserve_seconds) or reserve_seconds <= 0:
        raise ValueError('Finite deadline and positive reserve required')
    method = locked_identity.get('method')
    if (locked_identity.get('experiment') != 'learning_run_512_v1' or
        locked_identity.get('steps') != 512 or model.method != method or
        len(train) != 4096 or len(development) != 100 or
        any(r['prefix_length'] != 128 for r in train + development)):
        raise ValueError('Invalid learning run identity, model or data')
    if method == 'dlm' and not isinstance(model, paired.MaskedPaddingCanvasDLM):
        raise ValueError('DLM requires masked padding canvas')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Parameters must be FP32')
    work_deadline = deadline - reserve_seconds
    def guard():
        if time.monotonic() >= work_deadline:
            raise TimeoutError('Learning run stopped to preserve save/restore reserve')
    if resume is None:
        torch.manual_seed(11)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(11)
        order, masks = paired.fresh_generators(11)
        start, history = 0, []
        guard()
        evaluate(model, development, tokenizer, 0, work_deadline)
    else:
        start, history, order, masks = resume
        if start not in (128, 256) or len(history) != start:
            raise ValueError('Invalid checkpoint resume step/history')
        guard()
        evaluate(model, development, tokenizer, start, work_deadline)
    device = next(model.parameters()).device
    for step in range(start + 1, 513):
        guard()
        pilot.synchronize(device)
        begin = time.perf_counter()
        if method == 'dlm':
            stats = paired.update(model, optimizer, train, tokenizer.eos_token_id,
                                  order, masks, step)
        else:
            stats = pilot.update(model, optimizer, train, tokenizer.eos_token_id,
                                 order, masks, 2, 8, 2e-5 * min(step / 20, 1.))
            stats['step'] = step
        pilot.synchronize(device)
        stats['seconds'] = time.perf_counter() - begin
        history.append(stats)
        if step in (128, 256, 512):
            guard()
            checkpoint(model, optimizer, step, order, masks, locked_identity, history)
            guard()
            evaluate(model, development, tokenizer, step, work_deadline)
    return history
