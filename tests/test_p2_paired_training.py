import random

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments import full_parameter_pilot as pilot
from experiments import p2_paired_training as p2


def tiny_base():
    torch.manual_seed(3)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, attention_dropout=0.,
                         attn_implementation='sdpa')
    return Qwen2ForCausalLM(config)


def test_paired_masks_and_order_and_common_evaluation():
    rows = [{'ids': [2, 3, 4, 5, 0], 'prefix_length': 2},
            {'ids': [2, 3, 6, 7, 8, 0], 'prefix_length': 2}]
    models = [p2.make_model(tiny_base(), policy) for policy in p2.POLICIES]
    for name, tensor in models[0].state_dict().items():
        assert torch.equal(tensor, models[1].state_dict()[name])
    optimizers = [torch.optim.AdamW(model.parameters()) for model in models]
    generators = [p2.fresh_generators() for _ in models]
    stats = [p2.update(model, optimizer, rows, 0, order, masks, 1,
                       capacity=4, micro_batch=1, accumulation=1)
             for model, optimizer, (order, masks) in zip(models, optimizers, generators)]
    assert stats[0]['target_tokens'] == stats[1]['target_tokens']
    assert generators[0][0].getstate() == generators[1][0].getstate()
    assert torch.equal(generators[0][1].get_state(), generators[1][1].get_state())
    models[1].load_state_dict(models[0].state_dict())
    results = [p2.evaluate(model, rows, 0, capacity=4) for model in models]
    for result in results:
        for rate in result.values():
            rate.pop('seconds')
    assert results[0] == results[1]


def test_identity_guards_wrong_policy_restore(tmp_path):
    train, dev = tmp_path / 'training.json', tmp_path / 'development.json'
    train.write_text('train')
    dev.write_text('dev')
    a = p2.identity(p2.POLICIES[0], train, dev, 'base')
    b = p2.identity(p2.POLICIES[1], train, dev, 'base')
    model = p2.make_model(tiny_base(), p2.POLICIES[0])
    optimizer = torch.optim.AdamW(model.parameters())
    order, masks = p2.fresh_generators()
    path = tmp_path / 'snapshot.pt'
    pilot.save_checkpoint(path, model, optimizer, 0, order, masks, a, [])
    with pytest.raises(ValueError, match='configuration'):
        pilot.restore(path, model, optimizer, random.Random(), torch.Generator(), b)
    assert pilot.restore(path, model, optimizer, order, masks, a)[0] == 0


def test_budget_enforces_cumulative_cost_and_reserve():
    assert p2.check_budget(100, 99, 2, .5, 1800)['spent_cu'] == 1
    with pytest.raises(RuntimeError):
        p2.check_budget(100, 97, 2, .5, 1800)
    with pytest.raises(ValueError):
        p2.check_budget(100, 100, 0, .5, 60)
