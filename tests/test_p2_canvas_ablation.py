import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from experiments.full_parameter_pilot import ConditionalModel
from experiments.p2_canvas_ablation import MaskedPaddingCanvasDLM, pack_canvas


ROWS = [{'ids': [2, 3, 4, 5, 0], 'prefix_length': 2},
        {'ids': [2, 3, 6, 7, 8, 9, 0], 'prefix_length': 2}]


def models():
    torch.manual_seed(11)
    config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         attention_dropout=0., attn_implementation='sdpa')
    baseline = ConditionalModel(Qwen2ForCausalLM(config), 'dlm').eval()
    fixed = MaskedPaddingCanvasDLM(Qwen2ForCausalLM(config)).eval()
    fixed.load_state_dict(baseline.state_dict(), strict=True)
    return baseline, fixed


def test_common_canvas_excludes_padding_from_targets_and_preserves_real_counts():
    ids, valid, targets = pack_canvas(ROWS, 0, 'cpu', target_capacity=6)
    assert ids.shape == (2, 9)
    assert targets.sum().item() == 8
    assert not targets[~valid].any()
    assert not targets[:, :3].any()
    assert not targets[:, -1].any()


def test_fullmask_hides_true_end_cutoff_in_fixed_policy_tiny_model():
    baseline, fixed = models()
    ids, valid, targets = pack_canvas(ROWS, 0, 'cpu', target_capacity=6)
    with torch.no_grad():
        old = baseline.hidden(ids, valid, targets)
        new = fixed.hidden(ids, valid, targets)
    assert not torch.allclose(old[0], old[1], atol=1e-7)
    assert torch.allclose(new[0], new[1], atol=1e-7)
    changed = ids.clone()
    changed[targets | ~valid] = 27
    with torch.no_grad():
        assert torch.equal(new, fixed.hidden(changed, valid, targets))


def test_loss_uses_only_actual_labels_and_does_not_mutate_validity_masks():
    _, model = models()
    batch = pack_canvas(ROWS, 0, 'cpu', target_capacity=6)
    before = [value.clone() for value in batch]
    loss, stats = model.loss(batch, torch.Generator().manual_seed(1011), probability=1.)
    assert torch.isfinite(loss) and stats['count'] == 8 and stats['target_tokens'] == 8
    assert stats['input_tokens'] == sum(len(row['ids']) for row in ROWS)
    assert all(torch.equal(a, b) for a, b in zip(batch, before))


def test_capacity_and_prefix_changes_require_explicit_new_configuration():
    with pytest.raises(ValueError, match='exceeds'):
        pack_canvas(ROWS, 0, 'cpu', target_capacity=2)
    with pytest.raises(ValueError, match='shared prefix'):
        pack_canvas([ROWS[0], {**ROWS[1], 'prefix_length': 1}], 0, 'cpu')
