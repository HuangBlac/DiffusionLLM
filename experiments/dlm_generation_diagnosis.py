"""Inference-only paired sampling and context diagnostics for the frozen P1 DLM."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from experiments.full_parameter_pilot import ConditionalModel, evaluate, precision, synchronize

ROOT = Path(__file__).resolve().parents[1]


def repetition(ids):
    grams = [tuple(ids[i:i+4]) for i in range(len(ids)-3)]
    return {'adjacent_equal_fraction': sum(a == b for a, b in zip(ids, ids[1:])) / (len(ids)-1)
            if len(ids) > 1 else None,
            'repeated_4gram_fraction': 1-len(set(grams))/len(grams) if grams else None,
            'dominant_token_fraction': max(Counter(ids).values())/len(ids) if ids else None,
            'empty': not ids, 'short_under_8': len(ids) < 8, 'delivered_tokens': len(ids)}


def select_rows(rows, count):
    """Freeze article-balanced examples by hashes, without looking at generations."""
    groups = {}
    for row in rows:
        groups.setdefault(row['article'], []).append(row)
    def rank(row):
        return hashlib.sha256(json.dumps(row['ids']).encode()).hexdigest()
    for group in groups.values():
        group.sort(key=rank)
    ordered = sorted(groups, key=lambda article: hashlib.sha256(article.encode()).hexdigest())
    result = []
    for depth in range(max(map(len, groups.values()))):
        for article in ordered:
            if depth < len(groups[article]):
                result.append(groups[article][depth])
                if len(result) == count:
                    return result
    if len(result) != count:
        raise ValueError('Insufficient diagnostic examples')
    return result


def choose_positions(probabilities, positions, order, group, permutation, revealed):
    if order == 'confidence':
        return positions[probabilities.max(-1).values.topk(min(group, len(positions))).indices]
    if order == 'random':
        return permutation[revealed:revealed+min(group, len(positions))]
    raise ValueError('Unknown position order')


@torch.inference_mode()
def decode(model, tokenizer, row, *, order, group, temperature, seed, length=64,
           feedback=False, deadline=float('inf')):
    if group not in (1, 8) or temperature not in (0, 1) or length < 1:
        raise ValueError('Unsupported diagnostic condition')
    model.eval()
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    prefix = [eos] + row['ids'][:row['prefix_length']]
    references = row['ids'][row['prefix_length']:]
    if feedback:
        length = min(length, len(references))
    ids = torch.tensor([prefix+[eos]*length], device=device)
    valid = torch.ones_like(ids, dtype=torch.bool)
    masked = torch.zeros_like(valid)
    masked[:, len(prefix):] = True
    permutation = (torch.randperm(length, generator=torch.Generator().manual_seed(seed))
                   + len(prefix)).to(device)
    draw = torch.Generator(device=device).manual_seed(seed+10000)
    forbidden = [x for x in tokenizer.all_special_ids if x != eos]
    trace, context, revealed, forwards = [], [], 0, 0
    synchronize(device)
    started = time.perf_counter()
    while masked.any():
        if time.monotonic() >= deadline:
            raise TimeoutError('Diagnostic deadline reached before canvas completion')
        positions = masked[0].nonzero().flatten()
        with precision(device):
            hidden = model.hidden(ids, valid, masked)
            logits = model.base.lm_head(hidden[0, positions-1]).float()
        logits[:, forbidden] = -torch.inf
        logits[:, len(tokenizer):] = -torch.inf
        probabilities = logits.softmax(-1)
        chosen = choose_positions(probabilities, positions, order, group, permutation, revealed)
        # Map absolute chosen positions to rows of the remaining-position distribution.
        indices = torch.searchsorted(positions, chosen)
        values = (probabilities[indices].argmax(-1) if temperature == 0 else
                  torch.multinomial(probabilities[indices], 1, generator=draw).squeeze(-1))
        ids[0, chosen] = values
        masked[0, chosen] = False
        trace.append({'positions': (chosen-len(prefix)).tolist(), 'tokens': values.tolist()})
        revealed += len(chosen)
        forwards += 1
        if feedback and revealed in (16, 32, 48) and masked.any():
            remaining = masked[0].nonzero().flatten()
            reference = torch.tensor(references[:length], device=device)
            corrected = ids.clone()
            known = ~masked[0, len(prefix):]
            corrected[0, len(prefix):][known] = reference[known]
            labels = reference[remaining-len(prefix)]
            scores = {}
            with precision(device):
                for name, input_ids in [('generated_context', ids), ('reference_context', corrected)]:
                    states = model.hidden(input_ids, valid, masked)
                    raw_logits = model.base.lm_head(states[0, remaining-1]).float()
                    scores[name] = float(torch.nn.functional.cross_entropy(raw_logits, labels))
            context.append({'revealed': revealed, 'remaining_positions': (remaining-len(prefix)).tolist(),
                            'reference_ce': scores, 'generated_ids': ids[0, len(prefix):].tolist(),
                            'reference_ids': reference.tolist(),
                            'known_matches_reference': float((ids[0, len(prefix):][known] == reference[known]).float().mean())})
    synchronize(device)
    if ids[0, :len(prefix)].tolist() != prefix:
        raise RuntimeError('Prefix changed')
    tokens = ids[0, len(prefix):].tolist()
    eos_index = tokens.index(eos) if eos in tokens else None
    delivered = tokens if eos_index is None else tokens[:eos_index]
    return {'source_row': row['source_row'], 'article': row['article'], 'order': order,
            'group': group, 'temperature': temperature, 'seed': seed, 'canvas_length': length,
            'prompt': tokenizer.decode(prefix[1:]), 'text': tokenizer.decode(delivered),
            'token_ids': tokens, 'eos_index': eos_index, 'metrics': repetition(delivered),
            'forwards_generation': forwards, 'seconds_including_context_probe': time.perf_counter()-started,
            'forwards_context_probe': 2*len(context),
            'trace': trace, 'context_probes': context,
            'reference_context_is_diagnostic_only': feedback}


def save_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=5400)
    parser.add_argument('--deadline', type=float, default=float('inf'))
    parser.add_argument('--from-config', action='store_true',
                        help='Initialize architecture only; all tensors come from the checked checkpoint')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.output.exists() or args.max_seconds <= 0:
        raise ValueError('Use a new output file and positive time limit')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    with args.weights.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != manifest['weights']['sha256'] or manifest['step'] != 200:
        raise ValueError('Unverified diagnostic weights')
    data_bytes = (ROOT/'data/pipeline-pilot.json').read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha256(data_bytes).hexdigest() != manifest['identity']['data_sha256']:
        raise ValueError('Diagnostic data differs from training')
    kernel = ROOT/'experiments/full_parameter_pilot.py'
    if hashlib.sha256(kernel.read_bytes()).hexdigest() != manifest['identity']['script_sha256']:
        raise ValueError('Frozen kernel changed')
    data = json.loads(data_bytes)
    selected = select_rows(data['validation'], 8)
    conditions = [(order, group, 1, seed) for order in ('random', 'confidence')
                  for group in (1, 8) for seed in (101, 202)]
    conditions += [('confidence', group, 0, 101) for group in (1, 8)]
    report = {'status': 'running', 'purpose': 'development_diagnosis_not_final_evaluation',
              'training_performed': False, 'step': 200, 'weights_sha256': digest,
              'identity': manifest['identity'],
              'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
              'device': args.device,
              'inference_precision': 'FP32 weights/BF16 autocast' if args.device == 'cuda' else 'FP32 CPU',
              'runtime_torch': str(torch.__version__), 'runtime_transformers': transformers.__version__,
              'diagnostic_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'selected_rows': [r['source_row'] for r in selected],
              'selected_articles': sorted({r['article'] for r in selected}),
              'conditions': conditions, 'samples': [], 'context_diagnostics': [],
              'limitations': ['Eight previously observed development paragraphs, not64 independent articles',
                              'One training seed; decoding seeds are not independent training seeds',
                              '20-step comparison uses saved greedy outputs; missing intermediate weights were not recreated',
                              'Reference context CE mixes error propagation and legitimate reference-branch divergence']}
    save_report(args.output, report)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(11)
        torch.backends.cuda.matmul.allow_tf32 = False
        deadline = min(time.monotonic()+args.max_seconds, args.deadline)
        if args.from_config:
            config = AutoConfig.from_pretrained(ROOT/'data/model', local_files_only=True)
            base = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32,
                                                   attn_implementation='sdpa')
        else:
            base = AutoModelForCausalLM.from_pretrained(ROOT/'data/model', torch_dtype=torch.float32,
                                                       attn_implementation='sdpa', local_files_only=True)
        model = ConditionalModel(base, 'dlm')
        state = torch.load(args.weights, map_location='cpu', mmap=True, weights_only=True)
        if state['identity'] != manifest['identity'] or state['step'] != 200:
            raise ValueError('Model export identity mismatch')
        model.load_state_dict(state['model'], strict=True)
        del state
        model.requires_grad_(False).eval().to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(ROOT/'data/model', local_files_only=True)
        # Same corruption schedule on fixed train and development selections.
        for split in ('train', 'validation'):
            report[split+'_mask_diagnostics'] = evaluate(model, select_rows(data[split], 32),
                                                         tokenizer.eos_token_id, deadline)
            save_report(args.output, report)
            if not report[split+'_mask_diagnostics']['complete']:
                raise TimeoutError('Incomplete fixed-mask diagnostic')
        for row in selected:
            for order, group, temperature, seed in conditions:
                result = decode(model, tokenizer, row, order=order, group=group,
                                temperature=temperature, seed=seed, deadline=deadline)
                report['samples'].append(result)
                save_report(args.output, report)
                print('SAMPLE', len(report['samples']), '/80', flush=True)
            report['context_diagnostics'].append(decode(model, tokenizer, row, order='random',
                group=1, temperature=1, seed=101, feedback=True, deadline=deadline))
            save_report(args.output, report)
        if len(report['samples']) != 80 or len(report['context_diagnostics']) != 8:
            raise RuntimeError('Incomplete diagnostic matrix')
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='partial', error_type=type(error).__name__)
        save_report(args.output, report)
        raise
    finally:
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_initialized() else None
        save_report(args.output, report)


if __name__ == '__main__':
    main()
