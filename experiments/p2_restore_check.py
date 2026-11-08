"""Independent P2 restore check; this module never downloads or starts training."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_restored(path, expected_identity, model, optimizer, order, masks, probe):
    """Restore full P2 state and run a caller-supplied finite inference probe."""
    import torch
    if (expected_identity.get('experiment') != 'p2_paired_canvas_v1'
            or expected_identity.get('canvas_policy') not in ('actual_valid', 'masked_padding_canvas')
            or expected_identity.get('target_capacity') != 128):
        raise ValueError('Invalid P2 identity')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('identity') != expected_identity or state.get('step') != expected_identity['steps']:
        raise ValueError('P2 identity or locked update count differs')
    if not state.get('optimizer', {}).get('state'):
        raise ValueError('Checkpoint lacks populated optimizer state')
    for key in ('model', 'order', 'masks', 'torch_rng', 'cuda_rng', 'history'):
        if key not in state:
            raise ValueError('Checkpoint lacks full state: ' + key)
    cuda_rng = state['cuda_rng']
    if (not isinstance(cuda_rng, list) or len(cuda_rng) != 1
            or any(not isinstance(t, torch.Tensor) or t.device.type != 'cpu'
                   or t.dtype != torch.uint8 or t.ndim != 1 or t.numel() == 0 for t in cuda_rng)):
        raise ValueError('Checkpoint CUDA RNG structure differs from one GPU')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    for group in optimizer.param_groups:
        for parameter in group['params']:
            values = optimizer.state.get(parameter, {})
            if not all(k in values for k in ('step', 'exp_avg', 'exp_avg_sq')):
                raise ValueError('Adam state is incomplete')
            for name in ('exp_avg', 'exp_avg_sq'):
                tensor = values[name]
                if tensor.shape != parameter.shape or not torch.isfinite(tensor).all():
                    raise ValueError('Adam moment shape or values differ')
            if int(values['step'].item()) != state['step']:
                raise ValueError('Adam step differs')
    if len(state['history']) != state['step']:
        raise ValueError('Checkpoint history length differs')
    order.setstate(state['order'])
    masks.set_state(state['masks'])
    torch.set_rng_state(state['torch_rng'])
    cuda_rng_present = bool(state['cuda_rng'])
    step = state['step']
    value = float(probe(model))
    if not math.isfinite(value):
        raise ValueError('P2 restored inference is not finite')
    return {'restored': True, 'step': step, 'method': 'dlm',
            'identity_sha256': identity_digest(expected_identity),
            'rng_state_restored': True, 'rng_restored_scope': 'cpu_order_masks',
            'cuda_rng_present': cuda_rng_present, 'cuda_rng_replayed': False,
            'inference_finite': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--development', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent restore requires CUDA_VISIBLE_DEVICES empty')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments import p2_paired_training as p2
    from experiments.full_parameter_pilot import sha256

    identity = json.loads(args.identity.read_text(encoding='utf-8'))
    root = Path(__file__).parent
    for key, path in [('entry_sha256', root / 'p2_paired_training.py'),
                      ('policy_sha256', root / 'p2_canvas_ablation.py'),
                      ('core_sha256', root / 'full_parameter_pilot.py'),
                      ('development_sha256', args.development),
                      ('model_sha256', args.model_dir / 'model.safetensors')]:
        if sha256(path) != identity[key]:
            raise ValueError('Identity digest differs: ' + key)
    for key, filename in [('cloud_entry_sha256', 'p2_paired_entry.py'),
                          ('restore_check_sha256', 'p2_restore_check.py')]:
        if sha256(root / filename) != identity[key]:
            raise ValueError('Identity digest differs: ' + key)
    for filename in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        if sha256(args.model_dir / filename) != identity['model_files_sha256'][filename]:
            raise ValueError('Model configuration/tokenizer digest differs: ' + filename)
    source_names = ('p2_paired_training.py', 'p2_canvas_ablation.py', 'full_parameter_pilot.py',
                    'p2_paired_entry.py', 'p2_restore_check.py', 'prepare_data.py',
                    'drive_checkpoint_store.py', 'api_checkpoint_training.py', 'managed_run.py')
    if set(identity['source_files_sha256']) != set(source_names):
        raise ValueError('Source manifest membership differs')
    for filename in source_names:
        if sha256(root / filename) != identity['source_files_sha256'][filename]:
            raise ValueError('Source manifest digest differs: ' + filename)
    rows = json.loads(args.development.read_text(encoding='utf-8'))
    base = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, torch_dtype=torch.float32)
    model = p2.make_model(base, identity['canvas_policy'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(.9, .95), weight_decay=.01)
    order, masks = p2.fresh_generators()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    def probe(restored):
        result = p2.evaluate(restored, rows[:1], tokenizer.eos_token_id)
        return result['1.0']['ce_nats']
    print(json.dumps(check_restored(args.checkpoint, identity, model, optimizer, order, masks, probe)))


if __name__ == '__main__':
    main()
