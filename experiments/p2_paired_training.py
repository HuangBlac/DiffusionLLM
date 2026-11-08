"""P2 paired-training primitives. Importing this module never starts a run.

Cloud orchestration, live billing and independent Drive restore remain separate gates.
"""
import hashlib
import math
import random
import time
from pathlib import Path

import torch
import transformers
from torch import nn

from experiments import full_parameter_pilot as pilot
from experiments.p2_canvas_ablation import MaskedPaddingCanvasDLM, pack_canvas


POLICIES = ('actual_valid', 'masked_padding_canvas')


def identity(policy, training_path, development_path, model_sha256, *, steps=100, model_files=None):
    if policy not in POLICIES or type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError('Invalid P2 policy or prelocked update count')
    root = Path(__file__).parent
    return {'experiment': 'p2_paired_canvas_v1', 'method': 'dlm',
            'canvas_policy': policy, 'target_capacity': 128, 'prefix_length': 128,
            'training_sha256': pilot.sha256(training_path),
            'development_sha256': pilot.sha256(development_path),
            'model_sha256': model_sha256, 'steps': steps, 'seed': 11,
            'model_files_sha256': dict(model_files or {}),
            'micro_batch': 2, 'accumulation': 8, 'lr': 2e-5,
            'warmup_updates': 20, 'betas': [0.9, 0.95], 'weight_decay': 0.01,
            'torch': str(torch.__version__), 'transformers': transformers.__version__,
            'entry_sha256': pilot.sha256(__file__),
            'policy_sha256': pilot.sha256(root / 'p2_canvas_ablation.py'),
            'core_sha256': pilot.sha256(root / 'full_parameter_pilot.py'),
            'cloud_entry_sha256': pilot.sha256(root / 'p2_paired_entry.py'),
            'restore_check_sha256': pilot.sha256(root / 'p2_restore_check.py')}


def check_budget(start_balance, current_balance, cu_per_hour, reserve_cu, expected_seconds):
    """Reject a proposed operation unless its cost and reserve fit the 4 CU stage."""
    values = (start_balance, current_balance, cu_per_hour, reserve_cu, expected_seconds)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Budget values must be finite and nonnegative')
    if cu_per_hour <= 0 or reserve_cu <= 0 or current_balance > start_balance:
        raise ValueError('Need live rate, reserve and valid stage balances')
    spent = start_balance - current_balance
    projected = cu_per_hour * expected_seconds / 3600
    if spent + projected + reserve_cu > 4 or current_balance < projected + reserve_cu:
        raise RuntimeError('P2 budget or saving/restore reserve exhausted')
    return {'spent_cu': spent, 'projected_cu': projected, 'reserve_cu': reserve_cu}


def make_model(base, policy):
    if policy == 'actual_valid':
        return pilot.ConditionalModel(base, 'dlm')
    if policy == 'masked_padding_canvas':
        return MaskedPaddingCanvasDLM(base)
    raise ValueError('Unknown P2 policy')


def update(model, optimizer, rows, bos, order, masks, step, *, capacity=128,
           micro_batch=2, accumulation=8):
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = 2e-5 * min(step / 20, 1.)
    objective = 0.
    inputs = targets = supervised = 0
    for _ in range(accumulation):
        chosen = [rows[order.randrange(len(rows))] for _ in range(micro_batch)]
        with pilot.precision(device):
            loss, counts = model.loss(pack_canvas(chosen, bos, device,
                                                  target_capacity=capacity), masks)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite P2 objective')
        (loss / accumulation).backward()
        objective += loss.item() / accumulation
        inputs += counts['input_tokens']
        targets += counts['target_tokens']
        supervised += counts['count']
    grad = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return {'step': step, 'objective': objective, 'grad_norm': float(grad),
            'input_tokens': inputs, 'target_tokens': targets, 'supervised_tokens': supervised}


@torch.no_grad()
def evaluate(model, rows, bos, *, common_canvas=True, capacity=128, should_continue=None):
    """Fixed seeds pair the masks. Common canvas ignores the trained policy."""
    original = model.hidden
    if common_canvas:
        def hidden(ids, valid, masked):
            return pilot.ConditionalModel.hidden(model, ids, torch.ones_like(valid), masked | ~valid)
        model.hidden = hidden
    try:
        model.eval()
        device = next(model.parameters()).device
        result = {}
        for rate in (.25, .5, .75, .9, 1.):
            started = time.perf_counter()
            masks = torch.Generator().manual_seed(20261102 + int(rate * 100))
            total = count = correct = 0
            for row in rows:
                if should_continue is not None and not should_continue('evaluate'):
                    raise TimeoutError('Budget guard stopped evaluation')
                with pilot.precision(device):
                    _, stats = model.loss(pack_canvas([row], bos, device,
                                                       target_capacity=capacity), masks, rate)
                total += stats['ce_sum']
                count += stats['count']
                correct += stats['correct']
            result[str(rate)] = {'ce_nats': total / count if count else None,
                                 'accuracy': correct / count if count else None,
                                 'tokens': count, 'examples': len(rows),
                                 'seconds': time.perf_counter() - started}
        return result
    finally:
        if common_canvas:
            model.hidden = original


def selected_development(rows, count=8):
    if len(rows) < count:
        raise ValueError('Insufficient development examples')
    return sorted(rows, key=lambda row: hashlib.sha256(
        bytes(str(row['prefix_sha256']), 'utf-8')).digest())[:count]


def fresh_generators(seed=11):
    return random.Random(seed), torch.Generator().manual_seed(seed + 1000)


def run_policy(model, optimizer, train, development, tokenizer, locked_identity,
               checkpoint, should_continue):
    """Run one fresh group; caller enforces live budget, storage and independent restore.

    ``should_continue(phase)`` must fail closed on the live deadline/cost guard.
    ``checkpoint`` receives model, optimizer, step, generators, identity and history;
    it must persist the full state and establish its own server receipt.
    """
    if (locked_identity['canvas_policy'] not in POLICIES or len(train) != 4096
            or len(development) != 100 or any(row['prefix_length'] != 128
                                               for row in train + development)):
        raise ValueError('P2 requires the preselected training and development splits')
    torch.manual_seed(locked_identity['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(locked_identity['seed'])
    order, masks = fresh_generators(locked_identity['seed'])
    history = []
    device = next(model.parameters()).device
    bos = tokenizer.eos_token_id
    for step in range(1, locked_identity['steps'] + 1):
        if not should_continue('train'):
            raise TimeoutError('Budget guard stopped P2 before the locked update count')
        pilot.synchronize(device)
        started = time.perf_counter()
        stats = update(model, optimizer, train, bos, order, masks, step)
        pilot.synchronize(device)
        stats['seconds'] = time.perf_counter() - started
        history.append(stats)
    if not should_continue('save'):
        raise TimeoutError('Budget guard stopped before checkpoint')
    checkpoint(model, optimizer, locked_identity['steps'], order, masks, locked_identity, history)
    if not should_continue('evaluate'):
        raise TimeoutError('Budget guard stopped before evaluation')
    report = {'identity': locked_identity, 'history': history,
              'common_canvas_ce': evaluate(model, development, bos, should_continue=should_continue),
              'training_policy_ce': evaluate(model, development, bos, common_canvas=False,
                                              should_continue=should_continue),
              'samples': []}
    for length in (32, 64, 128):
        for row in selected_development(development):
            if not should_continue('generate'):
                raise TimeoutError('Budget guard stopped before all generation samples')
            sample = pilot.continuation(model, tokenizer, row, length=length, steps=length // 8)
            delivered = sample['token_ids'][:sample['delivered_tokens']]
            sample['empty'] = len(delivered) == 0
            sample['short_under_8_tokens'] = len(delivered) < 8
            sample['adjacent_repetitions'] = sum(a == b for a, b in zip(delivered, delivered[1:]))
            sample['adjacent_pairs'] = max(0, len(delivered) - 1)
            report['samples'].append(sample)
    return report
