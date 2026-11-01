import math

import pytest
import torch
from torch import nn

from experiments.compare_sampling import decode, repetition_metrics, restore_adapter


class TinyTokenizer:
    eos_token_id = 0
    all_special_ids = [0]

    def encode(self, prompt, add_special_tokens=False):
        return [1, 2]

    def __len__(self):
        return 6

    def decode(self, ids):
        return " ".join(map(str, ids))


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.seen_prefixes = []

    def hidden(self, clean, mask):
        self.seen_prefixes.append(clean[0, :3].tolist())
        assert not mask[0, :3].any()
        return torch.arange(clean.shape[1], dtype=torch.float32).view(1, -1, 1)

    def logits(self, states):
        logits = torch.zeros(len(states), 6)
        logits[:, 0] = 100  # EOS must be excluded by this fixed-length protocol.
        logits[:, 3] = 1 + states[:, 0] / 10
        return logits


@pytest.mark.parametrize("group", [1, 8])
def test_reveal_count_prompt_preservation_and_complete_output(group):
    model = TinyPredictor()
    result = decode(model, TinyTokenizer(), "prompt", "shift", group, length=10)
    assert result["forward_calls"] == math.ceil(10 / group)
    assert result["generated_ids"] == [3] * 10
    assert all(prefix == [0, 1, 2] for prefix in model.seen_prefixes)
    positions = [p for step in result["trace"] for p in step["output_positions"]]
    assert sorted(positions) == list(range(10))
    assert result["trace"][-1]["remaining_masks"] == 0


def test_restore_rejects_missing_state_and_freezes_verified_parameters():
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="keys"):
        restore_adapter(model, {"weight": model.weight.detach().clone()})
    state = {name: torch.ones_like(p) for name, p in model.named_parameters()}
    assert restore_adapter(model, state) == 2
    assert all(not p.requires_grad and torch.equal(p, state[name]) for name, p in model.named_parameters())


def test_repetition_metrics_are_defined_on_tokens_and_short_sequences():
    assert repetition_metrics([1, 2, 3])["repeated_4gram_fraction"] is None
    metrics = repetition_metrics([7] * 8)
    assert metrics["longest_identical_token_run"] == 8
    assert metrics["adjacent_equal_fraction"] == 1
    assert metrics["repeated_4gram_fraction"] == pytest.approx(0.8)
