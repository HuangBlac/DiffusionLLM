"""Exercise the actual Colab resume gate without mounting Drive or starting GPUs."""
import ast
import json

import pytest

from experiments.build_colab_notebook import build
from experiments.managed_run import digest, safe_path


def resume_gate():
    tree = ast.parse(''.join(build()['cells'][1]['source']))
    gate = next(node for node in ast.walk(tree)
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == 'RESUME_RUN_RELATIVE_PATH')
    return compile(ast.fix_missing_locations(ast.Module(body=[gate], type_ignores=[])),
                   '<actual-colab-resume-gate>', 'exec')


@pytest.fixture
def saved_run(tmp_path):
    root = tmp_path / 'previous'
    reports, checkpoints = {}, {}
    for method in ('ar', 'dlm', 'original-ar'):
        directory = root / method
        directory.mkdir(parents=True)
        report = {'status': 'completed', 'end_step': 0 if method == 'original-ar' else 20}
        if method != 'original-ar':
            checkpoint = directory / 'latest.pt'
            checkpoint.write_bytes(('saved-step-20-' + method).encode())
            checkpoints[method] = digest(checkpoint)
            report['checkpoint_sha256'] = checkpoints[method]
        filename = 'report-0-0.json' if method == 'original-ar' else 'report-0-20.json'
        path = directory / filename
        path.write_text(json.dumps(report), encoding='utf-8')
        reports[method] = digest(path)
    return {
        'MODE': 'p1', 'RESUME_RUN_RELATIVE_PATH': 'previous',
        'RESUME_CHECKPOINT_SHA256': checkpoints, 'RESUME_REPORT_SHA256': reports,
        'durable_base': tmp_path, 'safe_path': safe_path, 'digest': digest,
        'json': json, 'lifecycle': {}, 'run_id': 'new-control',
    }


def test_resume_accepts_verified_baseline_and_preserves_prior_reports(saved_run):
    prior = saved_run['durable_base'] / 'previous'
    before = {p: p.read_bytes() for p in prior.rglob('*') if p.is_file()}
    exec(resume_gate(), saved_run)
    assert saved_run['run_root'] == prior / 'new-control'
    assert saved_run['lifecycle']['verified_prior_report_sha256'] == saved_run['RESUME_REPORT_SHA256']
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize('method', ['ar', 'dlm', 'original-ar'])
def test_resume_rejects_changed_prior_report(saved_run, method):
    filename = 'report-0-0.json' if method == 'original-ar' else 'report-0-20.json'
    path = saved_run['durable_base'] / 'previous' / method / filename
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Prior report differs'):
        exec(resume_gate(), saved_run)
    assert not saved_run['lifecycle']


def test_resume_rejects_missing_original_baseline(saved_run):
    baseline = saved_run['durable_base'] / 'previous' / 'original-ar' / 'report-0-0.json'
    baseline.unlink()
    with pytest.raises(FileNotFoundError):
        exec(resume_gate(), saved_run)
    assert not saved_run['lifecycle']


def test_resume_rejects_partial_baseline_even_with_verified_digest(saved_run):
    baseline = saved_run['durable_base'] / 'previous' / 'original-ar' / 'report-0-0.json'
    baseline.write_text(json.dumps({'status': 'partial', 'end_step': 0}), encoding='utf-8')
    saved_run['RESUME_REPORT_SHA256']['original-ar'] = digest(baseline)
    with pytest.raises(ValueError, match='Prior stage is incomplete: original-ar'):
        exec(resume_gate(), saved_run)


def test_resume_rejects_changed_checkpoint(saved_run):
    checkpoint = saved_run['durable_base'] / 'previous' / 'ar' / 'latest.pt'
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Saved step-20 checkpoint differs: ar'):
        exec(resume_gate(), saved_run)


def test_resume_rejects_path_outside_durable_runs(saved_run):
    saved_run['RESUME_RUN_RELATIVE_PATH'] = '../outside'
    with pytest.raises(ValueError, match='Artifact must be a relative path'):
        exec(resume_gate(), saved_run)
