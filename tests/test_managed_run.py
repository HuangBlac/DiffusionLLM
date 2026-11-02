"""Exercise real child processes and durable files, without CUDA or downloads."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

from experiments.managed_run import Stage, execute, seconds_available, verify_receipt


CHILD = r'''
import argparse, hashlib, json, os, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('name')
p.add_argument('--behavior', default='ok')
a = p.parse_args()
a.root.mkdir(parents=True, exist_ok=True)
with (a.root / 'starts.txt').open('a') as f:
    f.write(a.name + '\n')
print('CHILD_STARTED ' + a.name, flush=True)
if a.behavior == 'hang':
    time.sleep(60)
if a.behavior == 'error':
    raise RuntimeError('injected child failure')
checkpoint = a.root / (a.name + '.pt')
checkpoint.write_bytes(b'checkpoint-' + a.name.encode())
if a.name == 'resume':
    assert (a.root / 'train.pt').read_bytes() == b'checkpoint-train'
    checkpoint.write_bytes((a.root / 'train.pt').read_bytes() + b'-restored')
report = {'status': 'completed', 'end_step': 20,
          'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
if a.behavior == 'partial':
    report['status'] = 'partial'
if a.behavior == 'wrong_digest':
    report['checkpoint_sha256'] = '0' * 64
if a.behavior == 'no_checkpoint':
    checkpoint.unlink()
(a.root / (a.name + '.json')).write_text(json.dumps(report))
print('CHILD_FINISHED ' + a.name, flush=True)
'''


@pytest.fixture
def make_stage(tmp_path):
    child = tmp_path / 'child.py'
    child.write_text(CHILD, encoding='utf-8')
    artifacts = tmp_path / 'persistent' / 'run'

    def make(name, behavior='ok', minimum_seconds=0):
        return Stage(name=name,
                     command=(sys.executable, '-u', str(child), str(artifacts), name,
                              '--behavior', behavior),
                     report=f'{name}.json', checkpoint=f'{name}.pt',
                     minimum_seconds=minimum_seconds, expected={'end_step': 20})

    return make, artifacts


def read_status(root):
    return json.loads((root / 'run.json').read_text(encoding='utf-8'))


def test_one_call_runs_train_then_fresh_process_restore_and_persists_evidence(make_stage):
    make, root = make_stage
    releases = []
    result = execute([make('train'), make('resume')], root,
                     budget_seconds=10, reserve_seconds=.1,
                     release=lambda: releases.append('released'))
    assert result['status'] == 'completed'
    assert (root / 'starts.txt').read_text().splitlines() == ['train', 'resume']
    assert [s['status'] for s in result['stages']] == ['completed', 'completed']
    assert releases == ['released']
    assert read_status(root)['status'] == 'completed'
    assert 'CHILD_FINISHED train' in (root / 'train.log').read_text()
    assert (root / 'resume.pt').read_bytes() == b'checkpoint-train-restored'
    assert verify_receipt(root)['verified'] is True


@pytest.mark.parametrize('behavior', ['error', 'partial', 'wrong_digest', 'no_checkpoint'])
def test_failure_never_starts_later_stage_and_still_releases(make_stage, behavior):
    make, root = make_stage
    releases = []
    result = execute([make('train', behavior), make('resume')], root,
                     budget_seconds=10, reserve_seconds=.1,
                     release=lambda: releases.append(True))
    assert result['status'] == 'failed'
    assert (root / 'starts.txt').read_text().splitlines() == ['train']
    assert result['stages'][-1]['status'] == 'not_started'
    assert releases == [True]
    assert read_status(root)['error']
    assert verify_receipt(root)['verified'] is False


def test_hung_child_is_killed_and_checkpoint_report_remain_unclaimed(make_stage):
    make, root = make_stage
    releases = []
    result = execute([make('train', 'hang'), make('resume')], root,
                     budget_seconds=1.2, reserve_seconds=.2,
                     release=lambda: releases.append(True))
    assert result['status'] == 'timed_out'
    assert result['elapsed_seconds'] < 6
    assert not (root / 'train.pt').exists()
    assert (root / 'starts.txt').read_text().splitlines() == ['train']
    assert releases == [True]


def test_missing_time_refuses_stage_instead_of_spending_save_reserve(make_stage):
    make, root = make_stage
    releases = []
    result = execute([make('train', minimum_seconds=120)], root,
                     budget_seconds=90, reserve_seconds=30,
                     release=lambda: releases.append(True))
    assert result['status'] == 'budget_exhausted'
    assert not (root / 'starts.txt').exists()
    assert releases == [True]


def test_existing_run_is_not_overwritten(make_stage):
    make, root = make_stage
    execute([make('train')], root, budget_seconds=10, reserve_seconds=.1)
    before = (root / 'run.json').read_bytes()
    with pytest.raises(FileExistsError):
        execute([make('train')], root, budget_seconds=10, reserve_seconds=.1)
    assert (root / 'run.json').read_bytes() == before


def test_checkpoint_corruption_and_missing_report_fail_readback(make_stage):
    make, root = make_stage
    execute([make('train')], root, budget_seconds=10, reserve_seconds=.1)
    (root / 'train.pt').write_bytes(b'corrupted')
    assert verify_receipt(root)['verified'] is False
    (root / 'train.json').unlink()
    assert verify_receipt(root)['verified'] is False


def test_release_failure_cannot_be_reported_as_confirmed_shutdown(make_stage):
    make, root = make_stage

    def fail_release():
        raise RuntimeError('release service unavailable')

    result = execute([make('train')], root, budget_seconds=10,
                     reserve_seconds=.1, release=fail_release)
    assert result['status'] == 'completed'
    assert result['release']['status'] == 'failed'
    assert 'release service unavailable' in read_status(root)['release']['error']


def test_budget_accounts_for_previous_attempts_and_rejects_exhaustion():
    assert seconds_available(2, 1.82, 6.73, reserve_cu=.1) == 42
    with pytest.raises(ValueError):
        seconds_available(2, 1.95, 6.73, reserve_cu=.1)
    for cap, used, rate in [(2, -1, 6.73), (2, 0, 0), (float('nan'), 0, 6.73)]:
        with pytest.raises(ValueError):
            seconds_available(cap, used, rate)


def test_artifact_paths_cannot_escape_durable_run_directory(make_stage, tmp_path):
    make, root = make_stage
    stage = make('train')
    malicious = Stage(name='escape', command=stage.command,
                      report='../outside.json', checkpoint='train.pt')
    with pytest.raises(ValueError):
        execute([malicious], root, budget_seconds=10, reserve_seconds=.1)
    assert not (root.parent / 'outside.json').exists()


def test_failure_to_create_durable_root_never_launches_child(make_stage, tmp_path):
    make, root = make_stage
    root.parent.mkdir(parents=True)
    root.write_text('A file blocks the intended directory')
    with pytest.raises(FileExistsError):
        execute([make('train')], root, budget_seconds=10, reserve_seconds=.1)
    assert root.read_text() == 'A file blocks the intended directory'
