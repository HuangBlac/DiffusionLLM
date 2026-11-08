import time

import torch
import transformers.utils.import_utils as transformers_import_utils
transformers_import_utils._sklearn_available = False
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from experiments.dlm512_diagnosis import context_diagnostic, mask_diagnostics, paired_inputs, select_rows  # noqa: E402
from experiments.p2_canvas_ablation import MaskedPaddingCanvasDLM  # noqa: E402


class TinyTokenizer:
    eos_token_id = 0
    all_special_ids = [0]
    def __len__(self):
        return 32


def tiny_model():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         attention_dropout=0., attn_implementation='sdpa')
    return MaskedPaddingCanvasDLM(Qwen2ForCausalLM(config)).eval()


def test_reference_substitution_uses_same_remaining_label_positions():
    ids = torch.tensor([[0, 1, 2, 7, 0, 8, 9, 0, 0]])
    masked = torch.tensor([[False, False, False, False, True, False, False, True, True]])
    corrected, positions, count = paired_inputs(ids, masked, 3, [11, 12, 13])
    assert count == 2 and positions.tolist() == [4]
    assert corrected.tolist() == [[0, 1, 2, 11, 0, 13, 9, 0, 0]]
    assert ids[0, 3].item() == 7 and corrected[0, 6].item() == 9


def test_tiny_model_context_has_no_reference_length_leak_and_paired_positions():
    model, tokenizer = tiny_model(), TinyTokenizer()
    base = {'ids': [1, 2, 3, 4, 0], 'prefix_length': 2, 'source_row': 10, 'prefix_sha256': 'abc'}
    shorter = {**base, 'ids': [1, 2, 3]}
    deadline = time.monotonic() + 90
    with torch.inference_mode():
        full = context_diagnostic(model, tokenizer, base, deadline, capacity=6, checkpoints=(1, 2, 3))
        short = context_diagnostic(model, tokenizer, shorter, deadline, capacity=6, checkpoints=(1, 2, 3))
    assert full['partially_revealed_canvas_token_ids'] == short['partially_revealed_canvas_token_ids']
    assert len(full['still_masked_positions']) == 3
    assert len(full['revealed_positions']) == 3 and not full['generation_complete']
    for probe in full['probes']:
        assert probe['same_mask_and_target_positions']
        assert probe['remaining_reference_positions'] == len(probe['positions'])
        assert all(0 <= p < 3 for p in probe['positions'])
    assert all(p['remaining_reference_positions'] <= 1 for p in short['probes'])


def test_mask_rates_tiny_model_and_selection_determinism():
    model = tiny_model()
    rows = [{'ids': [1, 2, 3, 4, 0], 'prefix_length': 2, 'source_row': i, 'prefix_sha256': str(i)} for i in range(3)]
    assert [x['source_row'] for x in select_rows(rows, 2)] == [x['source_row'] for x in select_rows(rows[::-1], 2)]
    result = mask_diagnostics(model, rows, 0, time.monotonic() + 90)
    assert set(result) == {'0.25', '0.5', '0.75', '0.9', '1.0'}
    assert result['1.0']['masked_tokens'] == result['1.0']['target_tokens'] == 9
    assert all(x['examples'] == 3 and x['masked_tokens'] <= x['target_tokens'] for x in result.values())
