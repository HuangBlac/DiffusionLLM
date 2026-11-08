from datetime import datetime, timezone, timedelta
import pytest
from experiments.learning_run_entry import resource_gate, budget_deadline, preflight_fit

NOW = datetime(2026, 11, 6, tzinfo=timezone.utc)

def observation():
    return dict(observed_utc=NOW.isoformat(), start_balance=180, current_balance=179,
                project_start_balance=199.7, cu_per_hour=2, gpu_name='L4',
                session_id='session', other_active_sessions=0)

def test_observation_requires_fresh_and_exclusive():
    resource_gate(observation(), now=NOW + timedelta(seconds=120))
    with pytest.raises(ValueError):
        resource_gate(observation(), now=NOW + timedelta(seconds=121))
    o = observation()
    o['other_active_sessions'] = 1
    with pytest.raises(ValueError):
        resource_gate(o, now=NOW)

def test_budget_enforces_stage_and_project():
    o = observation()
    assert budget_deadline(o, now_monotonic=100, now_utc=NOW) == 19900
    o['project_start_balance'] = 360
    with pytest.raises(RuntimeError):
        budget_deadline(o, now_monotonic=100, now_utc=NOW)

def test_512_preflight_never_shortens():
    kw = dict(now=100, update_seconds=1, evaluation_seconds=2,
              checkpoint_seconds=3, restore_seconds=4, reserve_seconds=20)
    result = preflight_fit(deadline=2000, **kw)
    assert result['projected_seconds'] == 1.5 * (1024 + 16 + 18 + 8)
    with pytest.raises(RuntimeError):
        preflight_fit(deadline=1500, **kw)


def test_cpu_restore_checks_identity_adam_and_rng(tmp_path):
    import random
    import torch
    from experiments.learning_run_restore import check_restored
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(.9, .95), weight_decay=.01)
    model(torch.ones(1, 2)).sum().backward()
    opt.step()
    for value in opt.state.values():
        value['step'] = torch.tensor(128.)
    order = random.Random(11)
    masks = torch.Generator().manual_seed(1011)
    identity = {'experiment':'learning_run_512_v1', 'method':'ar', 'steps':512}
    state = {'model':model.state_dict(), 'optimizer':opt.state_dict(), 'step':128,
             'order':order.getstate(), 'masks':masks.get_state(), 'torch_rng':torch.get_rng_state(),
             'cuda_rng':[torch.ones(12, dtype=torch.uint8)], 'identity':identity, 'history':[{}]*128}
    path = tmp_path / 'checkpoint.pt'
    torch.save(state, path)
    receipt = check_restored(path, identity, model, opt, order, masks, lambda m: m(torch.ones(1,2)).sum())
    assert receipt['restored'] and receipt['step'] == 128 and not receipt['cuda_rng_replayed']
    with pytest.raises(ValueError, match='identity'):
        check_restored(path, {**identity, 'method':'dlm'}, model, opt, order, masks, lambda _: 1.)
    state['optimizer']['state'][0]['exp_avg'][0, 0] = float('nan')
    torch.save(state, path)
    with pytest.raises(ValueError, match='moment'):
        check_restored(path, identity, model, opt, order, masks, lambda _: 1.)


def test_checkpoint_callback_checks_reserve_and_receipt(tmp_path):
    from experiments.learning_run_entry import checkpoint_callback
    class Writer:
        last_server_receipt = None
        def __call__(self, path, model, optimizer, step, order, masks, identity, history):
            self.last_server_receipt = {'step': step, 'status': 'server_verified'}
    reports = []
    callback = checkpoint_callback(writer=Writer(), group=tmp_path, method='ar', deadline=200,
                                   reserve=20, publish_report=lambda name, data: reports.append((name, data)),
                                   clock=lambda: 100)
    callback(None, None, 128, None, None, {'method':'ar'}, [])
    assert reports[0][0] == 'ar-checkpoint-128'
    with pytest.raises(ValueError):
        callback(None, None, 129, None, None, {'method':'ar'}, [])
    blocked = checkpoint_callback(writer=Writer(), group=tmp_path, method='ar', deadline=110,
                                  reserve=20, publish_report=lambda *_: None, clock=lambda: 100)
    with pytest.raises(TimeoutError):
        blocked(None, None, 128, None, None, {'method':'ar'}, [])


def test_lease_cannot_be_extended_or_nan():
    from experiments.learning_run_entry import validate_lease
    o = observation()
    # One hour of preparation is allowed only within the original timed lease.
    now = NOW + timedelta(hours=1)
    validate_lease(o, 19900, now_utc=now, now_monotonic=3700)
    with pytest.raises(ValueError):
        validate_lease(o, 19903, now_utc=now, now_monotonic=3700)
    with pytest.raises(ValueError):
        validate_lease(o, float('nan'), now_utc=now, now_monotonic=3700)


def test_cached_storage_rejects_changed_report_or_identity(monkeypatch):
    import base64
    import hashlib
    import json
    from pathlib import Path
    from experiments import learning_run_entry as entry
    payloads = {m: json.dumps({'save_seconds': 20, 'download_seconds': 10,
                               'cpu_load_seconds': 3, 'method': m}).encode()
                for m in ('ar', 'dlm')}
    monkeypatch.setattr(entry, 'CACHED_REPORT_SHA256',
                        {m: hashlib.sha256(data).hexdigest() for m, data in payloads.items()})
    cache = {m: base64.b64encode(data).decode() for m, data in payloads.items()}
    proof = {'artifacts': {key: {'sha256': entry.CACHED_DATA_SHA256[key]}
                           for key in ('training', 'development')},
             'tokenizer_sha256': entry.CACHED_DATA_SHA256['tokenizer']}
    result = entry.verified_cached_storage(cache, proof, Path('experiments'))
    assert result['ar']['save_seconds'] > 0 and result['dlm']['download_seconds'] > 0
    bad = dict(cache, ar=base64.b64encode(b'{}').decode())
    with pytest.raises(ValueError, match='digest'):
        entry.verified_cached_storage(bad, proof, Path('experiments'))
    proof['artifacts']['training']['sha256'] = 'other'
    with pytest.raises(ValueError, match='identity'):
        entry.verified_cached_storage(cache, proof, Path('experiments'))
