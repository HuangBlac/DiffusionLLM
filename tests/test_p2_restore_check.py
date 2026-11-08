import random

import pytest
import torch

from experiments.full_parameter_pilot import save_checkpoint
from experiments.p2_restore_check import check_restored


def test_p2_restore_requires_exact_identity_and_populated_adam(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    order, masks = random.Random(11), torch.Generator().manual_seed(1011)
    identity = {'experiment': 'p2_paired_canvas_v1', 'canvas_policy': 'actual_valid',
                'target_capacity': 128, 'steps': 1}
    path = tmp_path / 'checkpoint.pt'
    with monkeypatch.context() as patched:
        patched.setattr(torch.cuda, 'is_available', lambda: True)
        patched.setattr(torch.cuda, 'get_rng_state_all', lambda: [torch.ones(16, dtype=torch.uint8)])
        save_checkpoint(path, model, optimizer, 1, order, masks, identity, [{'step': 1}])
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    receipt = check_restored(path, identity, restored, restored_optimizer,
                             random.Random(), torch.Generator(),
                             lambda m: m(torch.ones(1, 2)).item())
    assert receipt['restored'] and receipt['rng_state_restored']
    assert restored_optimizer.state
    with pytest.raises(ValueError, match='identity'):
        check_restored(path, {**identity, 'canvas_policy': 'masked_padding_canvas'},
                       restored, restored_optimizer, random.Random(), torch.Generator(), lambda m: 1)
    with pytest.raises(ValueError, match='finite'):
        check_restored(path, identity, restored, restored_optimizer,
                       random.Random(), torch.Generator(), lambda m: float('nan'))
