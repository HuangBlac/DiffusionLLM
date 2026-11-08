"""Single invocation for transport checks or the bounded 0.5B P0 pipeline.

Transport checks use JSON fixture state, not model weights or a GPU. P0 requires
a mounted Drive directory and sufficient explicitly supplied cumulative budget.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from experiments.managed_run import (
    Stage, artifact, atomic_json, digest, execute, seconds_available, utc_now, verify_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


def transport_worker(root: Path, action: str) -> None:
    """Real independent-process I/O fixture; explicitly not a training result."""
    root.mkdir(parents=True, exist_ok=True)
    print(f'TRANSPORT_WORKER {action}', flush=True)
    if action == 'error':
        raise RuntimeError('Deliberate transport failure for stop/release testing')
    if action == 'hang':
        if os.name == 'posix':
            descendant = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
            atomic_json(root / 'descendant.json', {'pid': descendant.pid})
        time.sleep(120)
    state = root / 'fixture-state.json'
    if action == 'write':
        atomic_json(state, {'fixture_only': True, 'counter': 19})
    elif action == 'restore':
        previous = json.loads(state.read_text(encoding='utf-8'))
        if previous != {'fixture_only': True, 'counter': 19}:
            raise ValueError('Previous-process fixture state differs')
        atomic_json(state, {'fixture_only': True, 'counter': 20})
    atomic_json(root / f'{action}.json',
                {'status': 'completed', 'fixture_only': True,
                 'checkpoint_sha256': digest(state)})


def transport_checks(root: Path, python: str = sys.executable) -> dict:
    """Exercise success, fresh-process restore, failure and timeout end to end."""
    root.mkdir(parents=True, exist_ok=False)

    def stage(directory: Path, action: str) -> Stage:
        return Stage(action, (python, '-u', '-m', 'experiments.colab_pipeline',
                              '--worker', action, '--worker-root', str(directory)),
                     report=f'{action}.json', checkpoint='fixture-state.json',
                     expected={'fixture_only': True})

    results = {}
    good = root / 'success'
    results['success'] = execute([stage(good, 'write'), stage(good, 'restore')], good,
                                 budget_seconds=30, reserve_seconds=1)
    receipt = verify_receipt(good)
    results['readback_verified'] = receipt['verified']
    bad = root / 'failure'
    results['failure'] = execute([stage(bad, 'error'), stage(bad, 'restore')], bad,
                                 budget_seconds=15, reserve_seconds=1)
    hung = root / 'timeout'
    results['timeout'] = execute([stage(hung, 'hang'), stage(hung, 'restore')], hung,
                                 budget_seconds=8, reserve_seconds=1)
    descendant_stopped = None
    if os.name == 'posix':
        descendant = json.loads((hung / 'descendant.json').read_text())['pid']
        proc_state = Path(f'/proc/{descendant}/stat')
        descendant_stopped = (not proc_state.exists()
                              or proc_state.read_text().split(') ', 1)[1].split()[0] == 'Z')
    checks = {
        'automatic_sequence': results['success']['status'] == 'completed',
        'independent_restore_readback': receipt['verified'],
        'failure_stops_next_stage': (results['failure']['status'] == 'failed'
                                     and results['failure']['stages'][1]['status'] == 'not_started'),
        'timeout_stops_next_stage': (results['timeout']['status'] == 'timed_out'
                                     and results['timeout']['stages'][1]['status'] == 'not_started'),
    }
    if os.name == 'posix':
        checks['timed_out_descendant_stopped'] = descendant_stopped
    parent = root / 'parent-resume'
    execute([stage(parent, 'write')], parent, budget_seconds=30, reserve_seconds=1)
    previous_manifest = (parent / 'run.json').read_bytes()
    control = parent / 'continuation'
    resumed = execute([stage(parent, 'restore')], control, budget_seconds=30,
                      reserve_seconds=1, artifacts_in_parent=True)
    checks['cross_session_parent_artifacts'] = (
        resumed['status'] == 'completed' and verify_receipt(control)['verified']
        and previous_manifest == (parent / 'run.json').read_bytes())
    tensor_root = root / 'torch-checkpoint'
    tensor_stages = [Stage(
        phase, (python, '-u', '-m', 'experiments.checkpoint_transport',
                '--root', str(tensor_root), '--phase', phase),
        report=f'{phase}.json', checkpoint=checkpoint, maximum_seconds=90,
        expected={'next_update_equal': True} if phase == 'restore' else {})
        for phase, checkpoint in [('save', 'checkpoint.pt'), ('restore', 'restored-step2.pt')]]
    tensor_result = execute(tensor_stages, tensor_root, budget_seconds=180, reserve_seconds=10)
    tensor_receipt = verify_receipt(tensor_root)
    checks['torch_checkpoint_fresh_process_restore'] = (
        tensor_result['status'] == 'completed' and tensor_receipt['verified'])
    tensor_artifacts = tensor_receipt.get('artifacts', [])
    if checks['torch_checkpoint_fresh_process_restore']:
        tensor_artifacts.append(artifact(tensor_root, 'expected-step2.pt', 'checkpoint_reference'))
    summary = {'schema': 1, 'kind': 'transport_fixture_not_model_training',
               'status': 'completed' if all(checks.values()) else 'failed',
               'time_utc': utc_now(), 'checks': checks,
               'success_artifacts': receipt.get('artifacts', []),
               'parent_resume_artifacts': verify_receipt(control).get('artifacts', []),
               'torch_checkpoint_artifacts': tensor_artifacts,
               'cloud_readback': 'not_yet_verified_from_outside_runtime',
               'cuda_training_performed': False}
    atomic_json(root / 'transport-receipt.json', summary)
    return summary


def p1_stages(root: Path, python: str = sys.executable) -> list[Stage]:
    """One fresh 0->20->200 trajectory per method; checkpoint stays on Drive."""
    stages = p0_stages(root, python)[:4]
    for method in ('ar', 'dlm'):
        stages.append(Stage(
            f'{method}-20',
            (python, '-u', '-m', 'experiments.full_parameter_pilot',
             '--method', method, '--steps', '20', '--eval-examples', '8',
             '--max-seconds', '{max_seconds}', '--output', str(root / method)),
            report=f'{method}/report-0-20.json', checkpoint=f'{method}/latest.pt',
            minimum_seconds=180, maximum_seconds=420,
            expected={'start_step': 0, 'end_step': 20}))
    stages.append(Stage(
        'original-ar', (python, '-u', '-m', 'experiments.full_parameter_pilot',
                        '--method', 'ar', '--evaluate-only', '--eval-examples', '32',
                        '--max-seconds', '{max_seconds}', '--output', str(root / 'original-ar')),
        report='original-ar/report-0-0.json', minimum_seconds=180, maximum_seconds=300,
        expected={'start_step': 0, 'end_step': 0}))
    for method in ('dlm', 'ar'):
        stages.append(Stage(
            f'{method}-200',
            (python, '-u', '-m', 'experiments.full_parameter_pilot',
             '--method', method, '--steps', '200', '--resume', '--eval-examples', '32',
             '--max-seconds', '{max_seconds}', '--output', str(root / method)),
            report=f'{method}/report-20-200.json', checkpoint=f'{method}/latest.pt',
            minimum_seconds=600, maximum_seconds=1000,
            expected={'start_step': 20, 'end_step': 200}))
    return stages


def p0_stages(pipeline_root: Path, python: str = sys.executable) -> list[Stage]:
    """Automatic setup -> DLM train/replay -> AR train/replay, no UI queue."""
    stages = [
        Stage('dependencies', (python, '-m', 'pip', 'install', '--quiet',
                               'transformers==4.57.6', 'huggingface-hub>=0.34,<1.0',
                               'pyarrow>=15'), minimum_seconds=30, maximum_seconds=180),
        Stage('environment', (python, '-c',
                              'import torch; assert torch.__version__.startswith("2.6.0"); '
                              'assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); '
                              'print(torch.__version__, torch.cuda.get_device_name())'),
              minimum_seconds=15, maximum_seconds=60),
        Stage('data', (python, '-u', '-m', 'experiments.prepare_data'),
              minimum_seconds=30, maximum_seconds=180),
        Stage('pilot-data', (python, '-u', '-m', 'experiments.prepare_pipeline_pilot'),
              minimum_seconds=15, maximum_seconds=60),
    ]
    for method in ('dlm', 'ar'):
        for phase in ('train', 'resume'):
            expected = {'end_step': 20}
            if phase == 'resume':
                expected.update(restored_step=19, restored_step19_matches=True,
                                step20_matches=True, step20_stats_match=True)
            stages.append(Stage(
                f'{method}-{phase}',
                (python, '-u', '-m', 'experiments.resource_resume_probe',
                 '--method', method, '--phase', phase,
                 '--output', str(pipeline_root / method), '--max-seconds', '{max_seconds}'),
                report=f'{method}/{phase}.json', checkpoint=f'{method}/latest.pt',
                minimum_seconds=120, maximum_seconds=300, expected=expected))
    return stages


def require_drive_storage(path: Path) -> None:
    mount = Path('/content/drive')
    target = path.resolve()
    if not os.path.ismount(mount) or not target.is_relative_to(mount / 'MyDrive'):
        raise ValueError('P0 requires a mounted /content/drive/MyDrive directory')
    # Atomic replacement needs temporary room in addition to retained checkpoints.
    if shutil.disk_usage(mount).free < 24 * 2**30:
        raise OSError('At least 24 GiB of reported Drive free space is required for P0')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['transport', 'p0'], default='transport')
    parser.add_argument('--durable-root', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--cap-cu', type=float, default=2)
    parser.add_argument('--spent-cu', type=float, default=1.82)
    parser.add_argument('--rate-cu-hour', type=float, default=6.73)
    parser.add_argument('--max-seconds', type=float, default=120)
    parser.add_argument('--worker', choices=['write', 'restore', 'error', 'hang'])
    parser.add_argument('--worker-root', type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.worker_root is None:
            parser.error('--worker-root is required')
        transport_worker(args.worker_root, args.worker)
        return
    if (args.durable_root is None or not args.run_id
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', args.run_id)):
        parser.error('Provide --durable-root and a safe unique --run-id')
    os.environ.update(USE_TF='0', USE_FLAX='0', HF_HUB_DISABLE_XET='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    root = args.durable_root / args.run_id
    if args.mode == 'transport':
        summary = transport_checks(root)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if summary['status'] != 'completed':
            raise SystemExit(1)
        return
    budget = min(args.max_seconds,
                 seconds_available(args.cap_cu, args.spent_cu, args.rate_cu_hour))
    if budget < 600:
        raise ValueError('P0 needs at least 600 bounded seconds; prior CU spending is retained')
    require_drive_storage(root)
    result = execute(p0_stages(root), root, budget_seconds=budget, reserve_seconds=90)
    print(json.dumps({'status': result['status'], 'readback': verify_receipt(root)}), flush=True)
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
