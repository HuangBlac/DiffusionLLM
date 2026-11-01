import torch

from experiments.resource_resume_probe import state_fingerprint


def test_fingerprint_ignores_dict_order_but_detects_weights_optimizer_and_rng_changes():
    value = {"weights": torch.tensor([1., 2.]), "optimizer": {0: {"step": torch.tensor(19.)}},
             "rng": torch.Generator().manual_seed(11).get_state()}
    reordered = dict(reversed(list(value.items())))
    assert state_fingerprint(value) == state_fingerprint(reordered)
    for key in value:
        changed = dict(value)
        if key == "weights":
            changed[key] = torch.tensor([1., 3.])
        elif key == "optimizer":
            changed[key] = {0: {"step": torch.tensor(20.)}}
        else:
            changed[key] = torch.Generator().manual_seed(12).get_state()
        assert state_fingerprint(value) != state_fingerprint(changed)
