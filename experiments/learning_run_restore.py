"""Independent CPU audit of a learning-run checkpoint."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_restored(path, expected_identity, model, optimizer, order, masks, probe):
    import torch
    if (expected_identity.get('experiment') != 'learning_run_512_v1'
            or expected_identity.get('method') not in ('ar', 'dlm')
            or expected_identity.get('steps') != 512):
        raise ValueError('Invalid learning-run identity')
    state = torch.load(path, map_location='cpu', weights_only=True)
    step = state.get('step')
    if state.get('identity') != expected_identity or step not in (128, 256, 512):
        raise ValueError('Checkpoint identity or milestone differs')
    if len(state.get('history', [])) != step or not state.get('optimizer', {}).get('state'):
        raise ValueError('Incomplete history or optimizer state')
    cuda_rng = state.get('cuda_rng')
    if (not isinstance(cuda_rng, list) or len(cuda_rng) != 1 or any(
            not isinstance(t, torch.Tensor) or t.device.type != 'cpu'
            or t.dtype != torch.uint8 or t.ndim != 1 or not t.numel() for t in cuda_rng)):
        raise ValueError('Invalid CUDA RNG structure')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    for group in optimizer.param_groups:
        for parameter in group['params']:
            values = optimizer.state.get(parameter, {})
            if not all(k in values for k in ('step', 'exp_avg', 'exp_avg_sq')):
                raise ValueError('Incomplete Adam state')
            if int(values['step'].item()) != step:
                raise ValueError('Adam step differs')
            for name in ('exp_avg', 'exp_avg_sq'):
                if (values[name].dtype != torch.float32 or values[name].shape != parameter.shape
                        or not torch.isfinite(values[name]).all()):
                    raise ValueError('Invalid Adam moment')
    order.setstate(state['order'])
    masks.set_state(state['masks'])
    torch.set_rng_state(state['torch_rng'])
    if not math.isfinite(float(probe(model))):
        raise ValueError('Nonfinite CPU inference')
    return {'restored': True, 'step': step, 'method': expected_identity['method'],
            'identity_sha256': identity_digest(expected_identity), 'rng_state_restored': True,
            'rng_restored_scope': 'cpu_order_masks', 'cuda_rng_present': True,
            'cuda_rng_replayed': False, 'inference_finite': True}


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'identity', 'model-dir', 'development'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only restore required')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments import full_parameter_pilot as pilot, p2_paired_training as paired
    identity = json.loads(args.identity.read_text(encoding='utf-8'))
    root = Path(__file__).parent
    for key, path in [('model_sha256', args.model_dir / 'model.safetensors'),
                      ('development_sha256', args.development),
                      ('training_source_sha256', root / 'learning_run_training.py'),
                      ('evaluation_source_sha256', root / 'learning_run_evaluation.py'),
                      ('core_sha256', root / 'full_parameter_pilot.py'),
                      ('canvas_sha256', root / 'p2_canvas_ablation.py'),
                      ('paired_sha256', root / 'p2_paired_training.py')]:
        if pilot.sha256(path) != identity[key]:
            raise ValueError('Identity digest differs: ' + key)
    for name, digest in identity['model_files_sha256'].items():
        if pilot.sha256(args.model_dir / name) != digest:
            raise ValueError('Model configuration/tokenizer differs')
    for name, digest in identity.get('runtime_sources_sha256', {}).items():
        if Path(name).name != name or pilot.sha256(root / name) != digest:
            raise ValueError('Runtime source differs')
    base = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, torch_dtype=torch.float32)
    model = (pilot.ConditionalModel(base, 'ar') if identity['method'] == 'ar'
             else paired.MaskedPaddingCanvasDLM(base))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(.9, .95), weight_decay=.01)
    order, masks = paired.fresh_generators(11)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    rows = json.loads(args.development.read_text(encoding='utf-8'))
    def probe(restored):
        if identity['method'] == 'ar':
            return pilot.evaluate(restored, rows[:1], tokenizer.eos_token_id)['rates']['1.0']['ce_nats']
        return paired.evaluate(restored, rows[:1], tokenizer.eos_token_id)['1.0']['ce_nats']
    print(json.dumps(check_restored(args.checkpoint, identity, model, optimizer, order, masks, probe)))


if __name__ == '__main__':
    main()
