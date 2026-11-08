import torch
from torch import nn
from types import SimpleNamespace
from experiments.dlm_generation_diagnosis import decode, repetition, select_rows
from experiments.full_parameter_pilot import ConditionalModel, pack


class Tokenizer:
    eos_token_id = 0
    all_special_ids = [0, 1]

    def __len__(self):
        return 8

    def decode(self, ids):
        return ' '.join(map(str, ids))


class Oracle(nn.Module):
    """Exact conditionals for the two valid pairs (2,3) and (4,5)."""
    def __init__(self, eos_only=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.base = nn.Module()
        self.base.lm_head = nn.Identity()
        self.eos_only = eos_only
        self.inputs = []

    def hidden(self, ids, valid, masked):
        self.inputs.append((ids.clone(), masked.clone()))
        output = torch.full((*ids.shape, 8), -100.)
        if self.eos_only:
            output[..., 0] = 0.
            output[..., 1] = 10.  # Illegal special token must still be suppressed.
            return output
        first, second = int(ids[0, 2]), int(ids[0, 3])
        if masked[0, 3]:
            output[0, 1, [2, 4]] = 0.
        else:
            output[0, 1, 2 if second == 3 else 4] = 0.
        if masked[0, 2]:
            output[0, 2, [3, 5]] = 0.
        else:
            output[0, 2, 3 if first == 2 else 5] = 0.
        return output


ROW = {'ids': [7, 2, 3], 'prefix_length': 1, 'source_row': 1, 'article': 'a'}


def test_exact_joint_oracle_distinguishes_sequential_and_parallel_sampling():
    valid = {(2, 3), (4, 5)}
    sequential, parallel = [], []
    for seed in range(40):
        for group, results in [(1, sequential), (8, parallel)]:
            results.append(tuple(decode(Oracle(), Tokenizer(), ROW, order='random',
                group=group, temperature=1, seed=seed, length=2)['token_ids']))
    assert all(pair in valid for pair in sequential)
    assert any(pair not in valid for pair in parallel)


def test_eos_allowed_illegal_special_suppressed_and_full_canvas_filled():
    result = decode(Oracle(eos_only=True), Tokenizer(), ROW, order='confidence',
                    group=8, temperature=0, seed=101, length=8)
    assert result['token_ids'] == [0]*8
    assert result['eos_index'] == 0 and result['text'] == ''
    assert result['metrics']['repeated_4gram_fraction'] is None


def test_random_groups_share_permutation_and_prefix_never_changes():
    results = []
    for group in (1, 8):
        model = Oracle(eos_only=True)
        result = decode(model, Tokenizer(), ROW, order='random', group=group,
                        temperature=0, seed=202, length=8)
        results.append([p for turn in result['trace'] for p in turn['positions']])
        assert all(ids[0, :2].tolist() == [0, 7] for ids, _ in model.inputs)
        assert all(not mask[0, :2].any() for _, mask in model.inputs)
        assert model.inputs[0][1][0, 2:].all()
    assert results[0] == results[1] and sorted(results[0]) == list(range(8))


def test_reference_answer_is_not_visible_to_free_generation():
    changed = {**ROW, 'ids': [7, 6, 6]}
    args = dict(order='confidence', group=1, temperature=1, seed=101, length=2)
    assert decode(Oracle(), Tokenizer(), ROW, **args)['token_ids'] == decode(
        Oracle(), Tokenizer(), changed, **args)['token_ids']


def test_repetition_short_output_and_balanced_selection():
    assert repetition([])['adjacent_equal_fraction'] is None
    assert repetition([2, 2, 2])['repeated_4gram_fraction'] is None
    assert repetition([2]*8)['repeated_4gram_fraction'] == 0.8
    rows = [{'article': a, 'ids': [i], 'source_row': i} for a in ('a', 'b') for i in range(5)]
    picked = select_rows(rows, 4)
    assert [r['article'] for r in picked].count('a') == 2
    assert select_rows(list(reversed(rows)), 4) == picked


def test_reference_context_replaces_only_revealed_slots_on_a_clone():
    model = Oracle(eos_only=True)
    row = {**ROW, 'ids': [7]+[2]*33}
    result = decode(model, Tokenizer(), row, order='random', group=1,
                    temperature=0, seed=101, length=33, feedback=True)
    assert [p['revealed'] for p in result['context_probes']] == [16, 32]
    assert result['token_ids'] == [0]*33  # Truth corrections never affect the real trajectory.
    corrected_inputs = [(ids, mask) for ids, mask in model.inputs if (ids[0, 2:] == 2).any()]
    assert len(corrected_inputs) == 2
    for ids, mask in corrected_inputs:
        assert ids[0, :2].tolist() == [0, 7]
        assert (ids[0, 2:][~mask[0, 2:]] == 2).all()
        assert (ids[0, 2:][mask[0, 2:]] == 0).all()


def test_frozen_loss_matches_manual_shift_and_per_example_noise_normalization():
    class Probe:
        method = 'dlm'
        base = SimpleNamespace(lm_head=nn.Identity())

        def hidden(self, ids, valid, masked):
            return scores[:ids.shape[1]].unsqueeze(0).expand(len(ids), -1, -1)

    scores = torch.randn(10, 8, generator=torch.Generator().manual_seed(88))
    rows = [ROW, {**ROW, 'ids': [7, 3, 4, 5], 'source_row': 2}]
    batch = pack(rows, 0, 'cpu')
    ids, valid, targets = batch
    probability = 0.5
    mask = (torch.rand(ids.shape, generator=torch.Generator().manual_seed(17)) < probability) & targets
    actual, counts = ConditionalModel.loss(Probe(), batch,
        torch.Generator().manual_seed(17), probability=probability)
    manual = torch.tensor(0.)
    for row in range(len(rows)):
        for pos in mask[row].nonzero().flatten().tolist():
            manual += torch.nn.functional.cross_entropy(scores[pos-1:pos], ids[row, pos:pos+1]) / (
                targets[row].sum()*probability*len(rows))
    assert mask.any() and counts['count'] == int(mask.sum())
    assert torch.allclose(actual, manual, atol=1e-7)


def test_config_only_initialization_is_fully_replaced_by_checkpoint_state():
    from transformers import AutoModelForCausalLM, Qwen2Config

    config = Qwen2Config(vocab_size=8, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         tie_word_embeddings=True, attention_dropout=0.)
    torch.manual_seed(11)
    original = ConditionalModel(AutoModelForCausalLM.from_config(config, attn_implementation='sdpa'), 'dlm').eval()
    torch.manual_seed(22)
    restored = ConditionalModel(AutoModelForCausalLM.from_config(config, attn_implementation='sdpa'), 'dlm').eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    batch = pack([ROW], 0, 'cpu')
    with torch.no_grad():
        assert torch.equal(original.hidden(*batch), restored.hidden(*batch))
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in original.state_dict().items())
