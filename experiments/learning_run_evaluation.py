"""Bounded private development metrics and continuation samples."""
import math
import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from experiments import full_parameter_pilot as pilot
from experiments.p2_canvas_ablation import pack_canvas
from experiments.p2_paired_training import selected_development


def _guard(deadline):
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('Evaluation deadline reached')


class _Deadline(StoppingCriteria):
    def __init__(self, deadline):
        self.deadline = deadline

    def __call__(self, input_ids, scores, **kwargs):
        _guard(self.deadline)
        return False


@torch.no_grad()
def evaluate_dev(model, rows, bos, deadline):
    if len(rows) != 100:
        raise ValueError('Expected 100 development examples')
    device = next(model.parameters()).device
    model.eval()
    results = {}
    for rate in ((0.5, 1.0) if model.method == 'dlm' else (1.0,)):
        rng = torch.Generator().manual_seed(20261102 + int(rate * 100))
        loss_sum = count = correct = 0
        pilot.synchronize(device)
        started = time.perf_counter()
        for row in rows:
            _guard(deadline)
            batch = (pack_canvas([row], bos, device) if model.method == 'dlm'
                     else pilot.pack([row], bos, device))
            with pilot.precision(device):
                _, stats = model.loss(batch, rng, rate)
            loss_sum += stats['ce_sum']
            count += stats['count']
            correct += stats['correct']
        pilot.synchronize(device)
        results[str(rate)] = {'ce_nats': loss_sum / count if count else None,
                              'accuracy': correct / count if count else None,
                              'tokens': count, 'examples': len(rows),
                              'seconds': time.perf_counter() - started}
    return {'kind': 'conditional_ar_nll' if model.method == 'ar' else 'masked_ce_not_nll',
            'rates': results}


def generation_schedule(step):
    if step not in (0, 128, 256, 512):
        raise ValueError('Invalid evaluation step')
    if step != 512:
        return [(32, 'sample', 101)]
    return [(n, mode, seed) for n in (16, 32, 128)
            for mode, seed in (('greedy', None), ('sample', 101), ('sample', 102))]


def _metrics(tokens, eos):
    index = tokens.index(eos) if eos in tokens else None
    delivered = tokens if index is None else tokens[:index]
    grams = [tuple(delivered[i:i+4]) for i in range(max(0, len(delivered)-3))]
    return {'token_ids': tokens, 'delivered_tokens': len(delivered), 'eos_index': index,
            'natural_eos': index is not None, 'empty': not delivered,
            'short_under_8_tokens': len(delivered) < 8,
            'adjacent_repetitions': sum(a == b for a, b in zip(delivered, delivered[1:])),
            'adjacent_pairs': max(0, len(delivered)-1),
            'repeated_4grams': len(grams)-len(set(grams)), 'total_4grams': len(grams)}


@torch.no_grad()
def generate(model, tokenizer, row, deadline, *, nfe=32, mode='sample', seed=101):
    _guard(deadline)
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    prefix = [eos] + row['ids'][:row['prefix_length']]
    model.eval()
    pilot.synchronize(device)
    started = time.perf_counter()
    with pilot.precision(device):
        if model.method == 'ar':
            ids = torch.tensor([prefix], device=device)
            forbidden = [i for i in tokenizer.all_special_ids if i != eos]
            forbidden += list(range(len(tokenizer), model.base.config.vocab_size))
            result = model.base.generate(ids, attention_mask=torch.ones_like(ids),
                do_sample=False, max_new_tokens=128, eos_token_id=eos, pad_token_id=eos,
                use_cache=True, suppress_tokens=forbidden,
                stopping_criteria=StoppingCriteriaList([_Deadline(deadline)]))
            tokens = result[0, len(prefix):].tolist()
            forwards = len(tokens)
        else:
            if nfe not in (16, 32, 128) or mode not in ('greedy', 'sample'):
                raise ValueError('Invalid generation configuration')
            ids = torch.tensor([prefix + [eos] * 128], device=device)
            valid = torch.ones_like(ids, dtype=torch.bool)
            masked = torch.zeros_like(valid)
            masked[:, len(prefix):] = True
            rng = torch.Generator(device=device).manual_seed(seed) if mode == 'sample' else None
            forbidden = [i for i in tokenizer.all_special_ids if i != eos]
            forbidden += list(range(len(tokenizer), model.base.config.vocab_size))
            for step in range(nfe):
                _guard(deadline)
                positions = masked[0].nonzero().flatten()
                logits = model.base.lm_head(model.hidden(ids, valid, masked)[0, positions - 1]).float()
                logits[:, forbidden] = -torch.inf
                probs = logits.softmax(-1)
                predictions = (probs.argmax(-1) if mode == 'greedy' else
                               torch.multinomial(probs, 1, generator=rng).squeeze(-1))
                confidence = probs.gather(1, predictions[:, None]).squeeze(1)
                k = math.ceil(len(positions) / (nfe-step))
                chosen = confidence.topk(k).indices
                ids[0, positions[chosen]] = predictions[chosen]
                masked[0, positions[chosen]] = False
            tokens = ids[0, len(prefix):].tolist()
            forwards = nfe
    pilot.synchronize(device)
    output = _metrics(tokens, eos)
    output.update({'source_row': row['source_row'], 'prompt': tokenizer.decode(prefix[1:]),
                   'text': tokenizer.decode(tokens[:output['delivered_tokens']]),
                   'forwards': forwards, 'seconds': time.perf_counter()-started,
                   'length_cap': 128, 'mode': mode if model.method == 'dlm' else 'greedy',
                   'seed': seed if model.method == 'dlm' and mode == 'sample' else None})
    return output


def evaluate_generations(model, tokenizer, rows, step, deadline):
    """Returns raw text: caller must keep the result private/cloud only."""
    chosen = selected_development(rows, 8)
    schedule = generation_schedule(step) if model.method == 'dlm' else [(None, 'greedy', None)]
    return [generate(model, tokenizer, row, deadline, nfe=nfe, mode=mode, seed=seed)
            for nfe, mode, seed in schedule for row in chosen]
