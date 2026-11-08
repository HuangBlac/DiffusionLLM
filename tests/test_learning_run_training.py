import time
from types import SimpleNamespace

import pytest
import torch

from experiments import learning_run_training as run
from experiments import full_parameter_pilot as pilot


def test_paired_stream_and_bounded_callbacks(monkeypatch):
    draws = {}
    def fake_update(model, optimizer, rows, bos, order, masks, *args):
        draws.setdefault(model.method, []).extend(order.randrange(len(rows)) for _ in range(16))
        return {}
    monkeypatch.setattr(pilot, 'update', fake_update)
    monkeypatch.setattr(run.paired, 'update', fake_update)
    monkeypatch.setattr(run.paired, 'MaskedPaddingCanvasDLM', type('FakeDLM', (torch.nn.Module,), {
        '__init__': lambda self: (torch.nn.Module.__init__(self), setattr(self, 'method', 'dlm'),
                                  setattr(self, 'weight', torch.nn.Parameter(torch.zeros(1)))) and None}))
    class AR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.method = 'ar'
    rows = [{'ids': [1]*129, 'prefix_length': 128}] * 4096
    dev = rows[:100]
    for model in (AR(), run.paired.MaskedPaddingCanvasDLM()):
        checkpoints, evaluations = [], []
        identity = {'experiment': 'learning_run_512_v1', 'steps': 512, 'method': model.method}
        run.run_training(model, None, rows, dev, SimpleNamespace(eos_token_id=0), identity,
            deadline=time.monotonic()+30, reserve_seconds=1,
            checkpoint=lambda *args: checkpoints.append(args[2]),
            evaluate=lambda *args: evaluations.append(args[3]))
        assert checkpoints == [128, 256, 512]
        assert evaluations == [0, 128, 256, 512]
    assert draws['ar'] == draws['dlm'] and len(draws['ar']) == 8192


def test_deadline_stops_before_baseline():
    model = torch.nn.Linear(1, 1)
    model.method = 'ar'
    rows = [{'prefix_length': 128}] * 4096
    with pytest.raises(TimeoutError):
        run.run_training(model, None, rows, rows[:100], SimpleNamespace(eos_token_id=0),
          {'experiment':'learning_run_512_v1', 'steps':512, 'method':'ar'},
          deadline=time.monotonic()+0.1, reserve_seconds=1,
          checkpoint=lambda *args: None, evaluate=lambda *args: pytest.fail('evaluated'))


def test_identity_changes_with_data_and_method(tmp_path):
    train, dev = tmp_path/'train', tmp_path/'dev'
    train.write_text('a')
    dev.write_text('b')
    a = run.identity('ar', train, dev, 'hash', {'tokenizer':'hash'})
    b = run.identity('dlm', train, dev, 'hash', {'tokenizer':'hash'})
    train.write_text('changed')
    c = run.identity('ar', train, dev, 'hash', {'tokenizer':'hash'})
    assert a != b and a != c and a['experiment'] != 'p2_paired_canvas_v1'


def test_tiny_real_updates_and_masked_generation():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from experiments import p2_paired_training as paired
    from experiments import learning_run_evaluation as evaluation
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         attention_dropout=0., attn_implementation='sdpa', eos_token_id=0)
    rows = [{'ids':[2, 3, 4, 0], 'prefix_length':2, 'source_row':0}]
    for method in ('ar', 'dlm'):
        base = Qwen2ForCausalLM(config)
        model = (pilot.ConditionalModel(base, 'ar') if method == 'ar'
                 else paired.MaskedPaddingCanvasDLM(base))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(.9,.95), weight_decay=.01)
        order, masks = paired.fresh_generators()
        before = base.get_input_embeddings().weight.detach().clone()
        if method == 'ar':
            stats = pilot.update(model, optimizer, rows, 0, order, masks, 1, 1, 1e-6)
        else:
            stats = paired.update(model, optimizer, rows, 0, order, masks, 1,
                                  capacity=4, micro_batch=1, accumulation=1)
        assert stats['target_tokens'] == 2
        assert not torch.equal(before, base.get_input_embeddings().weight)
        if method == 'dlm':
            tokenizer = SimpleNamespace(eos_token_id=0, all_special_ids=[0],
                                        decode=lambda tokens: str(tokens))
            class Tokenizer:
                eos_token_id = tokenizer.eos_token_id
                all_special_ids = tokenizer.all_special_ids
                decode = staticmethod(tokenizer.decode)
                def __len__(self): return 16
            sample = evaluation.generate(model, Tokenizer(), rows[0], time.monotonic()+30,
                                         nfe=16, mode='sample', seed=101)
            assert len(sample['token_ids']) == 128 and sample['forwards'] == 16
