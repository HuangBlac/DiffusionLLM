"""CPU synthetic PyTorch checkpoint round trip; not 0.5B model training."""
import argparse
import os
from pathlib import Path
import random

import torch

from experiments.managed_run import atomic_json, digest


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def save(path, state):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(root: Path, phase: str):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = torch.nn.Linear(16, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    order, masks = random.Random(11), torch.Generator().manual_seed(1011)

    def update():
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(8, 16)
        mask = torch.rand(8, 16, generator=masks) > .5
        target = torch.randn(8, 16) + order.random()
        loss = ((model(x * mask) - target) ** 2).mean()
        loss.backward()
        optimizer.step()

    def state(step):
        return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'torch_rng': torch.get_rng_state(), 'order': order.getstate(),
                'masks': masks.get_state(), 'step': step, 'fixture_only': True}

    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / 'checkpoint.pt'
    reference = root / 'expected-step2.pt'
    if phase == 'save':
        if checkpoint.exists() or reference.exists():
            raise FileExistsError('Do not overwrite a checkpoint proof')
        update()
        save(checkpoint, state(1))
        update()
        save(reference, state(2))
        matched = None
        result_file = checkpoint
    else:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['step'] != 1 or saved['fixture_only'] is not True:
            raise ValueError('Unexpected checkpoint')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        order.setstate(saved['order'])
        masks.set_state(saved['masks'])
        update()
        matched = equal_state(state(2), torch.load(reference, map_location='cpu', weights_only=True))
        if not matched:
            raise RuntimeError('Restored next update differs')
        result_file = root / 'restored-step2.pt'
        if result_file.exists():
            raise FileExistsError(result_file)
        save(result_file, state(2))
    report = {'status': 'completed', 'kind': 'cpu_synthetic_torch_checkpoint',
              'phase': phase, 'next_update_equal': matched,
              'checkpoint_sha256': digest(result_file), 'checkpoint_bytes': result_file.stat().st_size,
              'reference_sha256': digest(reference), 'reference_bytes': reference.stat().st_size}
    atomic_json(root / f'{phase}.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=['save', 'restore'])
    args = parser.parse_args()
    print(run(args.root, args.phase), flush=True)
