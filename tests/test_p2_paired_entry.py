from datetime import datetime, timezone

import pytest

from experiments.p2_paired_entry import resource_gate, lock_steps


def test_resource_freshness_and_paired_budget():
    now = datetime.now(timezone.utc)
    observed = {'observed_utc': now.isoformat(), 'start_balance': 200.,
                'current_balance': 199., 'cu_per_hour': 6.,
                'reserve_cu': .5, 'overhead_seconds': 100.,
                'gpu_name': 'test', 'session_id': 'test', 'other_active_sessions': 0}
    assert resource_gate(observed, now=now) == observed
    assert 1 <= lock_steps(observed, 10) <= 100
    with pytest.raises(ValueError):
        resource_gate({**observed, 'observed_utc': '2020-01-01T00:00:00Z'}, now=now)
    with pytest.raises(RuntimeError):
        lock_steps({**observed, 'current_balance': 196.1}, 10)


@pytest.mark.parametrize('failure', [None, 'first_train', 'expired'])
def test_pair_orchestration_offline(tmp_path, monkeypatch, failure):
    """Exercise both disposable benchmarks and fresh groups without CUDA or Drive."""
    import json
    from types import SimpleNamespace
    import torch
    from transformers import AutoModelForCausalLM
    from experiments import p2_paired_entry as entry, p2_paired_training as p2
    from experiments import full_parameter_pilot as pilot, api_checkpoint_training as api

    events = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'fake-gpu')
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(pilot, 'synchronize', lambda _: None)
    monkeypatch.setattr(pilot, 'sha256', lambda path: 'tok' if str(path).endswith('tokenizer.json') else 'digest')
    monkeypatch.setattr(entry, 'resource_gate', lambda observation, **_: observation)
    monkeypatch.setattr(entry, 'lock_steps', lambda *args: 1)
    monkeypatch.setattr(entry.ast, 'literal_eval', lambda _: 'digest')

    class Model:
        def __init__(self, policy):
            self.policy = policy
            self.parameter = torch.nn.Parameter(torch.ones(1))
        def to(self, _):
            return self
        def parameters(self):
            return iter([self.parameter])

    class Base:
        def gradient_checkpointing_enable(self, **_):
            pass

    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: Base())
    def make_model(base, policy):
        events.append(('model', policy))
        return Model(policy)
    monkeypatch.setattr(p2, 'make_model', make_model)
    monkeypatch.setattr(p2, 'update', lambda model, *a, **k: events.append(('benchmark', model.policy)))
    monkeypatch.setattr(p2, 'identity', lambda policy, *a, **k: {'method': 'dlm', 'canvas_policy': policy})

    class Store:
        service = SimpleNamespace(about=lambda: SimpleNamespace(get=lambda **_: SimpleNamespace(
            execute=lambda **_: {'storageQuota': {}})))
        def call(self, fn):
            return fn()
        def download_verified(self, file_id, path, expected):
            events.append(('download', file_id))
            count = 4096 if file_id == 'training' else 100
            path.write_text(json.dumps([{'prefix_length': 128}] * count), encoding='utf-8')
        def verify_snapshot(self, receipt, folder, identity_sha256, verification_dir, restore_check):
            events.append(('verify', receipt['policy']))
            restore_check(verification_dir / 'fresh.pt')
            return {'manifest': {}, 'manifest_id': receipt['policy']}

    class Writer:
        def __init__(self, original_save, store, folder, root, method):
            self.last_server_receipt = None
        def __call__(self, path, model, optimizer, step, order, masks, identity, history):
            events.append(('upload', model.policy))
            self.last_server_receipt = {'policy': model.policy}
    monkeypatch.setattr(api, 'ApiCheckpointWriter', Writer)

    def run_policy(model, optimizer, train, development, tokenizer, identity, checkpoint, guard):
        events.append(('train', model.policy))
        if not guard('train'):
            raise TimeoutError('expired')
        if failure == 'first_train':
            raise RuntimeError('first group failed')
        checkpoint(model, optimizer, 1, None, None, identity, [])
        return {}
    monkeypatch.setattr(p2, 'run_policy', run_policy)
    if failure == 'expired':
        monkeypatch.setattr(entry, 'budget_deadline', lambda *a: -1)

    observation = {'observed_utc': datetime.now(timezone.utc).isoformat(),
                   'start_balance': 100., 'current_balance': 100., 'cu_per_hour': 1.,
                   'reserve_cu': .5, 'overhead_seconds': 1., 'gpu_name': 'fake-gpu'}
    proof = {'counts': {'training': 4096, 'development': 100},
             'artifacts': {s: {'sha256': 'x'} for s in ('training', 'development')},
             'artifact_files': {s: {'server': {'id': s}, 'fingerprints': {'sha256': 'x'}}
                                for s in ('training', 'development')}, 'tokenizer_sha256': 'tok'}
    def restore(path, identity):
        events.append(('restore', identity['canvas_policy']))
        return {}
    def publish(policy, report):
        events.append(('publish', policy))
    kwargs = dict(observation=observation, store=Store(), folder_id='folder',
                  staging=tmp_path / 'staging', proof=proof, model_path=tmp_path,
                  tokenizer=SimpleNamespace(eos_token_id=0), independent_restore=restore,
                  publish_report=publish)
    if failure:
        with pytest.raises((RuntimeError, TimeoutError)):
            entry.run_pair(**kwargs)
    else:
        assert entry.run_pair(**kwargs)['status'] == 'completed'
        assert [event for event in events if event[0] == 'download'] == [
            ('download', 'training'), ('download', 'development')]
        assert [event for event in events if event[0] == 'model'] == [
            ('model', p) for p in p2.POLICIES for _ in range(1)] * 2
        assert [event[0] for event in events if event[0] in ('train', 'upload', 'verify', 'restore', 'publish')] == [
            'train', 'upload', 'verify', 'restore', 'publish'] * 2
    if failure == 'first_train':
        assert ('train', p2.POLICIES[1]) not in events
    if failure == 'expired':
        assert not any(event[0] == 'train' for event in events)
