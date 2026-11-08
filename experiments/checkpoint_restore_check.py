"""Fresh CPU process: restore real model/Adam/RNG and run finite inference only."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def verify(path, config_dir, expected_identity, expected_history=None):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    from experiments.full_parameter_pilot import ConditionalModel
    import experiments.full_parameter_pilot as training

    if torch.cuda.is_available():
        raise RuntimeError('Restore audit must use CPU, not GPU')
    if (str(torch.__version__) != expected_identity['torch']
            or transformers.__version__ != expected_identity['transformers']):
        raise ValueError('Restore audit dependency versions differ')
    source_hash = hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest()
    if source_hash != expected_identity['script_sha256']:
        raise ValueError('Training source identity differs')
    torch.set_num_threads(2)
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    if state['identity'] != expected_identity:
        raise ValueError('Downloaded checkpoint identity differs')
    if state['history'][-1]['step'] != state['step']:
        raise ValueError('Step/history mismatch')
    if expected_history is not None and state['history'] != expected_history[:state['step']]:
        raise ValueError('Checkpoint history differs from verified training report')
    config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
    base = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32,
                                           attn_implementation='sdpa')
    model = ConditionalModel(base, state['identity']['method'])
    model.load_state_dict(state['model'], strict=True, assign=True)
    base.tie_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=state['identity']['lr'],
                                  betas=(.9, .95), weight_decay=.01, foreach=False)
    optimizer.load_state_dict(state['optimizer'])
    parameters = {p for group in optimizer.param_groups for p in group['params']}
    if state['step'] > 0 and set(optimizer.state) != parameters:
        raise ValueError('Adam state coverage is incomplete')
    for parameter, values in optimizer.state.items():
        for key in ('exp_avg', 'exp_avg_sq'):
            if values[key].shape != parameter.shape:
                raise ValueError('Adam state shape differs')
            if not bool(torch.isfinite(values[key]).all()):
                raise ValueError('Adam state contains non-finite values')
        if values['step'].item() != state['step']:
            raise ValueError('Adam update count differs')
    order = random.Random()
    order.setstate(state['order'])
    masks = torch.Generator()
    masks.set_state(state['masks'])
    torch.set_rng_state(state['torch_rng'])
    rng_ok = (order.getstate() == state['order']
              and torch.equal(masks.get_state(), state['masks'])
              and torch.equal(torch.get_rng_state(), state['torch_rng']))
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    valid = torch.ones_like(ids, dtype=torch.bool)
    masked = torch.zeros_like(valid)
    if state['identity']['method'] == 'dlm':
        masked[0, -1] = True
    with torch.no_grad():
        logits = base.lm_head(model.hidden(ids, valid, masked))
        finite = bool(torch.isfinite(logits).all())
    if not finite or not rng_ok:
        raise ValueError('CPU inference or RNG restore failed')
    identity_hash = hashlib.sha256(json.dumps(expected_identity, sort_keys=True,
                                              separators=(',', ':')).encode()).hexdigest()
    return {'restored': True, 'method': state['identity']['method'], 'step': state['step'],
            'identity_sha256': identity_hash, 'model_tensor_count': len(state['model']),
            'optimizer_state_count': len(optimizer.state), 'rng_state_restored': rng_ok,
            'inference_finite': finite, 'training_performed': False,
            'history_reference_matches': expected_history is not None,
            'cuda_rng_present': bool(state.get('cuda_rng')), 'gpu_step_replay_tested': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config-dir', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history-reference', type=Path)
    args = parser.parse_args()
    result = verify(args.checkpoint, args.config_dir,
                    json.loads(args.identity.read_text(encoding='utf-8')),
                    json.loads(args.history_reference.read_text(encoding='utf-8'))
                    if args.history_reference else None)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
