"""CPU-only paired fixed-canvas diagnostics for the frozen DLM512 checkpoint."""
import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.p2_canvas_ablation import MaskedPaddingCanvasDLM, pack_canvas
from experiments.p2_paired_training import selected_development

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = 'c59e8d956edcd0da9b7ace94f39d7e22dfc4a65ce6c90b3709edc377208a03fa'
IDENTITY_SHA = '7696ae359720aac46f5d6ad22c621543f4f3ce99f032eb02c6d5537c37be849e'
RATES = (.25, .5, .75, .9, 1.)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def select_rows(rows, count):
    """Rank by full token content without using model outcomes."""
    if len(rows) < count:
        raise ValueError('Insufficient rows')
    return sorted(rows, key=lambda row: (hashlib.sha256(
        json.dumps(row['ids'], separators=(',', ':')).encode()).hexdigest(),
        str(row['source_row'])))[:count]


def guard(deadline):
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('Diagnostic deadline reached')


@torch.inference_mode()
def mask_diagnostics(model, rows, eos, deadline):
    device = next(model.parameters()).device
    result = {}
    for rate in RATES:
        generator = torch.Generator().manual_seed(20261102 + int(rate * 100))
        ce_sum = correct = tokens = targets = 0
        details = []
        for row in rows:
            guard(deadline)
            _, stats = model.loss(pack_canvas([row], eos, device, target_capacity=128), generator, rate)
            ce_sum += stats['ce_sum']
            correct += stats['correct']
            tokens += stats['count']
            targets += stats['target_tokens']
            details.append({'source_row': row['source_row'], 'prefix_sha256': row['prefix_sha256'],
                            'reference_length': len(row['ids']) - row['prefix_length'],
                            'ce_sum': stats['ce_sum'], 'masked_tokens': stats['count'],
                            'target_tokens': stats['target_tokens'], 'correct': stats['correct']})
        if tokens == 0:
            raise ValueError('No supervised tokens for mask condition')
        result[str(rate)] = {'ce_nats': ce_sum / tokens, 'accuracy': correct / tokens,
                             'masked_tokens': tokens, 'target_tokens': targets,
                             'examples': len(rows), 'seed': 20261102 + int(rate * 100),
                             'rows': details}
    return result


def paired_inputs(ids, masked, prefix_length, reference):
    """Substitute only revealed positions with real reference labels.

    Positions beyond the real continuation remain generated in both conditions;
    masks, attention and evaluated labels remain identical and expose no true-end
    attention cutoff. Return only remaining positions having reference labels.
    """
    start = prefix_length
    corrected = ids.clone()
    known = ~masked[0, start:start + len(reference)]
    labels = torch.as_tensor(reference, dtype=ids.dtype, device=ids.device)
    corrected[0, start:start + len(reference)][known] = labels[known]
    remaining = masked[0].nonzero().flatten()
    comparable = remaining[(remaining >= start) & (remaining < start + len(reference))]
    return corrected, comparable, int(known.sum().item())


@torch.inference_mode()
def context_diagnostic(model, tokenizer, row, deadline, *, capacity=128, checkpoints=(16, 32, 48), seed=101):
    if capacity < 1 or any(n < 1 or n >= capacity for n in checkpoints):
        raise ValueError('Invalid canvas or reveal points')
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    prefix = [eos] + row['ids'][:row['prefix_length']]
    reference = row['ids'][row['prefix_length']:]
    if not reference or len(reference) > capacity:
        raise ValueError('Invalid reference length')
    start = len(prefix)
    ids = torch.tensor([prefix + [eos] * capacity], device=device)
    valid = torch.ones_like(ids, dtype=torch.bool)
    masked = torch.zeros_like(valid)
    masked[:, start:] = True
    permutation = (torch.randperm(capacity, generator=torch.Generator().manual_seed(seed)) + start).to(device)
    draw = torch.Generator(device=device).manual_seed(seed + 10000)
    forbidden = [x for x in tokenizer.all_special_ids if x != eos]
    probes = []
    for revealed in range(1, max(checkpoints) + 1):
        guard(deadline)
        position = permutation[revealed - 1]
        hidden = model.hidden(ids, valid, masked)
        logits = model.base.lm_head(hidden[0, position - 1]).float()
        logits[forbidden] = -torch.inf
        logits[len(tokenizer):] = -torch.inf
        value = torch.multinomial(logits.softmax(-1), 1, generator=draw)
        ids[0, position] = value.item()
        masked[0, position] = False
        if revealed in checkpoints:
            corrected, comparable, substituted = paired_inputs(ids, masked, start, reference)
            known_total = revealed
            record = {'revealed': revealed, 'remaining_canvas_positions': capacity - revealed,
                      'remaining_reference_positions': comparable.numel(),
                      'reference_substituted_positions': substituted,
                      'unsubstituted_revealed_positions': known_total - substituted,
                      'reference_eos_exists': eos in reference,
                      'reference_eos_substituted': any(
                          token == eos and not bool(masked[0, start + i])
                          for i, token in enumerate(reference)),
                      'positions': (comparable - start).tolist(),
                      'generated_context_ce': None, 'reference_context_ce': None,
                      'same_mask_and_target_positions': True}
            if comparable.numel():
                labels = torch.tensor(reference, device=device)[comparable - start]
                for name, context in (('generated_context_ce', ids), ('reference_context_ce', corrected)):
                    guard(deadline)
                    states = model.hidden(context, valid, masked)
                    scores = model.base.lm_head(states[0, comparable - 1]).float()
                    record[name] = float(F.cross_entropy(scores, labels))
            probes.append(record)
    tokens = ids[0, start:].tolist()
    return {'source_row': row['source_row'], 'prefix_sha256': row['prefix_sha256'],
            'canvas_capacity': capacity, 'seed': seed, 'reference_length': len(reference),
            'partial_canvas': True, 'generation_complete': False,
            'revealed_count': max(checkpoints),
            'partially_revealed_canvas_token_ids': tokens,
            'still_masked_positions': (masked[0].nonzero().flatten() - start).tolist(),
            'revealed_positions': (~masked[0, start:]).nonzero().flatten().tolist(),
            'probes': probes}


def save(path, result):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'model-dir', 'training', 'development', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    guard(args.deadline)
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Need a new output and CPU-only environment')
    if sha(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Checkpoint digest differs')
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    identity = state['identity']
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (digest != IDENTITY_SHA or identity.get('experiment') != 'learning_run_512_v1'
            or identity.get('method') != 'dlm' or identity.get('canvas_policy') != 'masked_padding_canvas'
            or identity.get('target_capacity') != 128 or state.get('step') != 512
            or len(state.get('history', [])) != 512):
        raise ValueError('Checkpoint identity or step differs')
    for key, path in (('model_sha256', args.model_dir / 'model.safetensors'),
                      ('training_sha256', args.training), ('development_sha256', args.development),
                      ('training_source_sha256', ROOT / 'experiments/learning_run_training.py'),
                      ('evaluation_source_sha256', ROOT / 'experiments/learning_run_evaluation.py'),
                      ('core_sha256', ROOT / 'experiments/full_parameter_pilot.py'),
                      ('canvas_sha256', ROOT / 'experiments/p2_canvas_ablation.py'),
                      ('paired_sha256', ROOT / 'experiments/p2_paired_training.py')):
        guard(args.deadline)
        if sha(path) != identity[key]:
            raise ValueError('Identity digest differs: ' + key)
    for name, expected in identity['model_files_sha256'].items():
        if Path(name).name != name or sha(args.model_dir / name) != expected:
            raise ValueError('Model configuration/tokenizer differs')
    weights = state['model']
    del state
    gc.collect()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                                torch_dtype=torch.float32, attn_implementation='sdpa')
    model = MaskedPaddingCanvasDLM(base).cpu()
    model.load_state_dict(weights, strict=True)
    del weights
    gc.collect()
    model.requires_grad_(False).eval()
    parameter_versions = {name: parameter._version for name, parameter in model.named_parameters()}
    training = json.loads(args.training.read_text(encoding='utf-8'))
    development = json.loads(args.development.read_text(encoding='utf-8'))
    if len(training) != 4096 or len(development) != 100:
        raise ValueError('Frozen input counts differ')
    chosen = {'training': select_rows(training, 32), 'development': select_rows(development, 32)}
    contexts = selected_development(development, 8)
    result = {'status': 'running', 'training_performed': False, 'step': 512,
              'checkpoint_sha256': CHECKPOINT_SHA, 'identity_sha256': digest,
              'device': 'cpu', 'precision': 'fp32', 'canvas_policy': 'masked_padding_canvas',
              'target_capacity': 128, 'mask_diagnostics': {}, 'context_diagnostics': [],
              'selection': {key: [row['source_row'] for row in rows] for key, rows in chosen.items()},
              'context_source_rows': [row['source_row'] for row in contexts],
              'context_method': 'random positions, one per forward, temperature 1, seed 101',
              'p1_comparison': 'P1 used a clipped 64-position canvas; CE is not directly comparable across protocols.',
              'reference_tail_policy': 'Only revealed reference-bearing positions substituted; beyond-reference positions stay generated in both conditions. CE only on identical still-masked reference-bearing positions.',
              'limits': ['Reference advantage includes natural divergence; not causal attribution.',
                         'Different reveal counts may have different eligible reference positions.',
                         'Eight prompts and 32 examples per split are diagnostic samples.']}
    save(args.output, result)
    try:
        for split, rows in chosen.items():
            result['mask_diagnostics'][split] = mask_diagnostics(model, rows, tokenizer.eos_token_id, args.deadline)
            save(args.output, result)
            print('MASK_DIAGNOSTICS_COMPLETE', split, flush=True)
        for row in contexts:
            result['context_diagnostics'].append(context_diagnostic(model, tokenizer, row, args.deadline))
            save(args.output, result)
            print('CONTEXT_DIAGNOSTIC', len(result['context_diagnostics']), '/8', flush=True)
        result['status'] = 'completed'
    except BaseException as exc:
        result['status'] = 'partial'
        result['error_type'] = type(exc).__name__
        save(args.output, result)
        raise
    if any(parameter._version != parameter_versions[name] for name, parameter in model.named_parameters()):
        raise RuntimeError('Model parameter version changed during inference')
    result['weights_unchanged_by_inplace_version_check'] = True
    eligible = {count: [r['source_row'] for r in result['context_diagnostics']
                        if next(p for p in r['probes'] if p['revealed'] == count)['remaining_reference_positions'] > 0]
                for count in (16, 32, 48)}
    result['context_prompt_denominators'] = {str(k): v for k, v in eligible.items()}
    result['context_common_prompt_subset'] = sorted(set.intersection(*(set(v) for v in eligible.values())))
    save(args.output, result)
    print('DLM512_DIAGNOSIS_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
