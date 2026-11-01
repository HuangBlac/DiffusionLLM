import random

import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.full_parameter_pilot import (
    ConditionalModel, continuation, pack, restore, save_checkpoint, update,
)
from experiments.prepare_pipeline_pilot import examples


def tiny(method):
    torch.manual_seed(11)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=50, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, attn_implementation="sdpa", eos_token_id=0,
        bos_token_id=0, pad_token_id=0, attention_dropout=0.))
    return ConditionalModel(base, method)


ROWS = [{"ids": [2, 3, 4, 5, 6, 0], "prefix_length": 2, "source_row": 1},
        {"ids": [7, 8, 9, 0], "prefix_length": 2, "source_row": 2}]


def test_prefix_padding_and_hidden_targets_cannot_leak():
    model = tiny("dlm").eval()
    ids, valid, targets = pack(ROWS, 0, "cpu")
    assert targets.sum().item() == 6
    assert not targets[:, :3].any()
    assert not targets[1, 5:].any()
    changed = ids.clone()
    changed[targets | ~valid] = 33
    with torch.no_grad():
        assert torch.equal(model.hidden(ids, valid, targets)[valid],
                           model.hidden(changed, valid, targets)[valid])


def test_causal_and_bidirectional_attention_behave_differently():
    ids, valid, targets = pack(ROWS[:1], 0, "cpu")
    changed = ids.clone()
    changed[:, -1] = 30
    for method in ("ar", "dlm"):
        model = tiny(method).eval()
        with torch.no_grad():
            equal = torch.allclose(model.hidden(ids, valid, torch.zeros_like(targets))[:, 0],
                                   model.hidden(changed, valid, torch.zeros_like(targets))[:, 0], atol=1e-6)
        assert equal == (method == "ar")


@pytest.mark.parametrize("method", ["ar", "dlm"])
def test_full_parameters_update_and_resume_reproduces_uninterrupted_training(tmp_path, method):
    model = tiny(method)
    assert all(p.requires_grad for p in model.parameters())
    original = model.base.model.layers[0].self_attn.q_proj.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    order, masks = random.Random(11), torch.Generator().manual_seed(1011)
    history = []
    for _ in range(2):
        history.append(update(model, optimizer, ROWS, 0, order, masks, 1, 2, 1e-3))
    assert not torch.equal(original, model.base.model.layers[0].self_attn.q_proj.weight)
    path, identity = tmp_path / "state.pt", {"method": method}
    save_checkpoint(path, model, optimizer, 2, order, masks, identity, history)
    expected = update(model, optimizer, ROWS, 0, order, masks, 1, 2, 1e-3)
    expected_state = {n: p.detach().clone() for n, p in model.named_parameters()}
    resumed = tiny(method)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.5, foreach=False)
    resumed_order, resumed_masks = random.Random(), torch.Generator()
    step, log = restore(path, resumed, resumed_optimizer, resumed_order, resumed_masks, identity)
    assert step == 2 and log == history
    actual = update(resumed, resumed_optimizer, ROWS, 0, resumed_order, resumed_masks, 1, 2, 1e-3)
    assert actual == expected
    for name, value in resumed.named_parameters():
        assert torch.equal(value, expected_state[name])
    with pytest.raises(ValueError, match="differs"):
        restore(path, resumed, resumed_optimizer, resumed_order, resumed_masks, {"wrong": True})


def test_ar_and_dlm_receive_same_examples_and_target_exposure():
    logs = []
    for method in ("ar", "dlm"):
        model = tiny(method)
        logs.append(update(model, torch.optim.AdamW(model.parameters()), ROWS, 0,
                           random.Random(11), torch.Generator().manual_seed(1011), 2, 2, 1e-3))
    assert logs[0]["input_tokens"] == logs[1]["input_tokens"]
    assert logs[0]["target_tokens"] == logs[1]["target_tokens"]


class Tokenizer:
    eos_token_id = 0
    all_special_ids = [0, 1]

    def __len__(self):
        return 50

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [2] * int(text)


@pytest.mark.parametrize("method,steps", [("ar", 1), ("dlm", 1), ("dlm", 8)])
def test_eos_is_allowed_and_canvas_is_completed_before_interpretation(method, steps):
    model = tiny(method).eval()
    model.base.lm_head = nn.Linear(32, 50)
    with torch.no_grad():
        model.base.lm_head.weight.zero_()
        model.base.lm_head.bias.zero_()
        model.base.lm_head.bias[0] = 10
        model.base.lm_head.bias[1] = 20  # Must be forbidden in both methods.
    output = continuation(model, Tokenizer(), ROWS[0], length=8, steps=steps)
    assert output["text"] == "" and output["eos_index"] == 0
    assert output["token_ids"] == [0] * (1 if method == "ar" else 8)
    assert output["forwards"] == steps


def test_real_paragraph_end_has_eos_but_truncated_window_does_not():
    result = examples([" = Article = ", "140", "300"], Tokenizer(), 2)
    assert result[0]["natural_end"] and result[0]["ids"][-1] == 0
    assert not result[1]["natural_end"] and len(result[1]["ids"]) == 256
    assert result[0]["prefix_length"] == result[1]["prefix_length"] == 128
