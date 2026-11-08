"""Build a self-contained, single-code-cell Colab entry; default is a CPU test."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('managed_run.py', 'colab_pipeline.py', 'mount_contract.py', 'checkpoint_transport.py', 'resource_resume_probe.py',
           'full_parameter_pilot.py', 'prepare_data.py', 'prepare_pipeline_pilot.py')


CELL = '''# Default: CPU transport check. This cell does not train a model.
MODE = "transport"
CAP_CU, SPENT_CU, RATE_CU_HOUR = 2.0, 1.82, 6.73
RELEASE_RESERVE_CU = 0.5
ACCOUNTING_RESERVE_CU = 0.1
# P0 stays locked until an outside-runtime readback has verified this receipt.
VERIFIED_TRANSPORT_RELATIVE_PATH = None
VERIFIED_TRANSPORT_SHA256 = None
RESUME_RUN_RELATIVE_PATH = None
RESUME_CHECKPOINT_SHA256 = None
RESUME_REPORT_SHA256 = None

import base64, hashlib, inspect, io, json, os, subprocess, sys, time, uuid, zipfile
from pathlib import Path
from google.colab import drive, runtime

mounted = False
lifecycle, lifecycle_path = {}, None
hard_release_deadline, flush_window_seconds = None, 0

def run():
    global mounted, lifecycle, lifecycle_path, hard_release_deadline, flush_window_seconds
    from datetime import datetime, timezone
    from math import floor, isfinite
    started = time.monotonic()
    if MODE not in ('transport', 'p0', 'p1'):
        raise ValueError('Unknown mode')
    if MODE != 'transport':
        raise RuntimeError('Legacy DriveFS training is disabled; use the restore-verified Drive API entry')
    values = (CAP_CU, SPENT_CU, RATE_CU_HOUR)
    if not all(isfinite(v) for v in values) or CAP_CU <= 0 or SPENT_CU < 0 or RATE_CU_HOUR <= 0:
        raise ValueError('Invalid cumulative budget')
    time_cap = 1800 if MODE == 'transport' else floor((CAP_CU-SPENT_CU-RELEASE_RESERVE_CU)*3600/RATE_CU_HOUR)
    hard_seconds = (1860 if MODE == 'transport' else
                    floor((CAP_CU-SPENT_CU-ACCOUNTING_RESERVE_CU)*3600/RATE_CU_HOUR))
    if not 0 < ACCOUNTING_RESERVE_CU < RELEASE_RESERVE_CU or hard_seconds <= time_cap:
        raise ValueError('Need distinct work, flush and accounting reserves')
    hard_release_deadline = started + hard_seconds
    flush_window_seconds = hard_seconds - time_cap
    if MODE != 'transport' and (time_cap < 600 or not VERIFIED_TRANSPORT_RELATIVE_PATH
                         or not VERIFIED_TRANSPORT_SHA256):
        raise ValueError('P0 locked: insufficient cumulative budget or missing external transport verification')
    # Separate process: no Drive I/O, survives a notebook-kernel reset.
    guard_code = "import sys,time; time.sleep(float(sys.argv[1])); from google.colab import runtime; runtime.unassign()"
    subprocess.Popen([sys.executable, '-c', guard_code,
                      str(max(0, hard_release_deadline - time.monotonic()))],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    source_bytes = base64.b64decode(__BUNDLE__)
    bundle_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_hashes = __HASHES__
    work = Path('/content') / ('dllm-source-' + uuid.uuid4().hex)
    work.mkdir()
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
        if set(archive.namelist()) != set(source_hashes):
            raise ValueError('Source bundle file list differs')
        for name in archive.namelist():
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != source_hashes[name]:
                raise ValueError('Source digest differs: ' + name)
            destination = work / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
    if any(n == 'experiments' or n.startswith('experiments.') for n in sys.modules):
        raise RuntimeError('Start a fresh runtime to avoid cached source from another run')
    sys.path.insert(0, str(work))
    os.chdir(work)
    from experiments.managed_run import Stage, atomic_json, digest, execute, safe_path, utc_now, verify_receipt
    from experiments.colab_pipeline import p0_stages, p1_stages, require_drive_storage, transport_checks
    from experiments.mount_contract import MOUNT_TIMEOUT_MS, mount_contract, require_mount_contract
    policy = mount_contract(inspect.getsource(drive))
    durable_base = Path('/content/drive/MyDrive/DiffusionLLM/runs')
    run_id = MODE + '-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    run_root = durable_base / run_id
    lifecycle_path = durable_base / (run_id + '-lifecycle.json')
    lifecycle = {'run_id': run_id, 'mode': MODE, 'status': 'starting',
                 'source_bundle_sha256': bundle_sha256, 'source_hashes': source_hashes,
                 'time_cap_seconds': time_cap, 'started_utc': utc_now()}
    lifecycle['hard_release_seconds'] = hard_seconds
    lifecycle['flush_window_seconds'] = flush_window_seconds
    if MODE == 'transport':
        try:
            gpu = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
        except FileNotFoundError:
            gpu = None
        if gpu is not None and gpu.returncode != 0:
            raise RuntimeError('GPU detection failed; cannot certify a CPU transport check')
        if gpu is not None and gpu.returncode == 0 and 'GPU' in gpu.stdout:
            raise RuntimeError('Use a CPU runtime for this transport check')
    mount_started = time.monotonic()
    print('MOUNT_STARTED', policy['timeout_ms'], flush=True)
    drive.mount('/content/drive', timeout_ms=MOUNT_TIMEOUT_MS)
    mounted = True
    lifecycle['mount_contract'] = policy
    lifecycle['mount_elapsed_seconds'] = time.monotonic() - mount_started
    if not os.path.ismount('/content/drive'):
        raise RuntimeError('Drive mount was not established')
    durable_base.mkdir(parents=True, exist_ok=True)
    atomic_json(lifecycle_path, lifecycle)
    print('DURABLE_RUN', str(run_root), flush=True)
    env_root = work / 'venv'
    python = str(env_root / 'bin' / 'python')
    setup = [
        Stage('create-env', (sys.executable, '-m', 'venv', '--without-pip', str(env_root)), maximum_seconds=60),
        Stage('install-env', (sys.executable, '-m', 'pip', '--python', python,
                              'install', '--disable-pip-version-check', 'pip',
                              'torch==2.6.0', 'transformers==4.57.6',
                              'huggingface-hub>=0.34,<1.0', 'pyarrow>=15'), maximum_seconds=600),
        Stage('check-env', (python, '-c', 'import torch,transformers; '
                            'assert torch.__version__.split("+")[0]=="2.6.0"; '
                            'assert transformers.__version__=="4.57.6"; '
                            'print(torch.__version__,transformers.__version__)'), maximum_seconds=60)]
    setup_budget = min(720, time_cap - (time.monotonic() - started) - 120)
    setup_result = execute(setup, durable_base / (run_id + '-environment'),
                           budget_seconds=setup_budget, reserve_seconds=30)
    if setup_result['status'] != 'completed':
        raise RuntimeError('Pinned environment preparation did not complete')
    policy['torch_version'] = '2.6.0'
    policy['transformers_version'] = '4.57.6'
    # Patch releases share the interpreter ABI; Colab CPU/GPU images may differ.
    policy['python_version'] = '.'.join(sys.version.split()[0].split('.')[:2])
    lifecycle['python_full_version'] = sys.version.split()[0]
    if MODE == 'transport':
        result = transport_checks(run_root, python)
        result['source_bundle_sha256'] = bundle_sha256
        result['mount_contract'] = policy
        result['mount_elapsed_seconds'] = lifecycle['mount_elapsed_seconds']
        result['python_full_version'] = sys.version.split()[0]
        atomic_json(run_root / 'transport-receipt.json', result)
    else:
        reference = (durable_base / VERIFIED_TRANSPORT_RELATIVE_PATH).resolve()
        if not reference.is_relative_to(durable_base.resolve()):
            raise ValueError('Transport receipt path escapes durable base')
        if digest(reference) != VERIFIED_TRANSPORT_SHA256:
            raise ValueError('Verified transport receipt has changed')
        receipt = json.loads(reference.read_text())
        require_mount_contract(receipt, policy)
        if receipt['status'] != 'completed' or receipt['source_bundle_sha256'] != bundle_sha256:
            raise ValueError('Transport check did not verify these exact source bytes')
        require_drive_storage(run_root)
        resume_root = None
        if RESUME_RUN_RELATIVE_PATH is not None:
            if (MODE != 'p1' or not isinstance(RESUME_CHECKPOINT_SHA256, dict)
                    or not isinstance(RESUME_REPORT_SHA256, dict)):
                raise ValueError('Resume requires P1 and explicit checkpoint digests')
            resume_root = safe_path(durable_base, RESUME_RUN_RELATIVE_PATH)
            for method, filename in (('ar', 'report-0-20.json'),
                                     ('dlm', 'report-0-20.json'),
                                     ('original-ar', 'report-0-0.json')):
                report_path = safe_path(resume_root, method + '/' + filename)
                if digest(report_path) != RESUME_REPORT_SHA256[method]:
                    raise ValueError('Prior report differs: ' + method)
                prior_report = json.loads(report_path.read_text())
                expected_step = 0 if method == 'original-ar' else 20
                if (prior_report.get('status') != 'completed'
                        or prior_report.get('end_step') != expected_step):
                    raise ValueError('Prior stage is incomplete: ' + method)
            for method in ('ar', 'dlm'):
                checkpoint = safe_path(resume_root, method + '/latest.pt')
                previous = json.loads(safe_path(resume_root, method + '/report-0-20.json').read_text())
                expected = RESUME_CHECKPOINT_SHA256[method]
                if (previous.get('status') != 'completed' or previous.get('end_step') != 20
                        or previous.get('checkpoint_sha256') != expected
                        or digest(checkpoint) != expected):
                    raise ValueError('Saved step-20 checkpoint differs: ' + method)
            lifecycle['resume_source'] = RESUME_RUN_RELATIVE_PATH
            run_root = resume_root / run_id
            lifecycle['control_directory'] = str(run_root)
            lifecycle['verified_prior_report_sha256'] = RESUME_REPORT_SHA256
            print('DURABLE_RESUME_CONTROL', str(run_root), flush=True)
        remaining = time_cap - (time.monotonic() - started)
        if remaining < 600:
            raise RuntimeError('Mount used the execution reserve; do not start model preparation')
        if resume_root is not None:
            from dataclasses import replace
            planned = p1_stages(resume_root, python)
            # Include repeated 6GB state saves and 32-example generation;
            # the global cumulative CU deadline remains the hard outer bound.
            stages = planned[:4] + [replace(s, maximum_seconds=1500) for s in planned[-2:]]
        else:
            stages = p1_stages(run_root, python) if MODE == 'p1' else p0_stages(run_root, python)
        result = execute(stages, run_root, budget_seconds=remaining,
                         reserve_seconds=90, artifacts_in_parent=resume_root is not None)
        lifecycle['checkpoint_readback'] = verify_receipt(run_root)
        if result['status'] == 'completed' and not lifecycle['checkpoint_readback']['verified']:
            raise RuntimeError('Final checkpoint readback failed')
    lifecycle['status'] = result['status']
    print('PIPELINE_STATUS', result['status'], flush=True)

try:
    run()
except BaseException as exc:
    lifecycle.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    raise
finally:
    lifecycle.update(release='requested_not_confirmed',
                     cloud_flush='sync_pending')
    try:
        if mounted:
            from experiments.managed_run import atomic_json, utc_now
            lifecycle['finished_utc'] = utc_now()
            atomic_json(lifecycle_path, lifecycle)
            # A same-mount read is insufficient: explicitly push outstanding writes.
            flush_seconds = min(flush_window_seconds,
                                int(hard_release_deadline - time.monotonic() - 30))
            if flush_seconds < 1:
                print('DRIVE_FLUSH_SKIPPED_BUDGET; saved cloud state unverified', flush=True)
                raise TimeoutError('No flush time remains before release margin')
            try:
                drive.flush_and_unmount(timeout_ms=1000 * flush_seconds)
            except BaseException:
                print('DRIVE_FLUSH_FAILED; saved cloud state unverified', flush=True)
                raise
            print('DRIVE_FLUSH_RETURNED; verify receipt externally', flush=True)
    finally:
        # The independent guard stays armed if this request fails.
        runtime.unassign()
'''


def build() -> dict:
    """Package a fixed whitelist of source, without data, weights or credentials."""
    output = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            relative = 'experiments/' + name
            payload = (ROOT / relative).read_bytes()
            hashes[relative] = hashlib.sha256(payload).hexdigest()
            info = zipfile.ZipInfo(relative, date_time=(2026, 11, 2, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    source = CELL.replace('__BUNDLE__', repr(base64.b64encode(output.getvalue()).decode()))
    source = source.replace('__HASHES__', repr(hashes))
    compile(source, '<managed-colab-cell>', 'exec')
    return {
        'nbformat': 4, 'nbformat_minor': 0,
        'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                     'colab': {}},
        'cells': [
            {'cell_type': 'markdown', 'metadata': {}, 'source': [
                '# DiffusionLLM: managed execution and durable results\n',
                'Default is **CPU transport verification**, with no model training.\n',
                'Run the single code cell. Grant Drive access yourself when prompted; '
                'this lets notebook code access your Drive files. Only the project run '
                'directory is used by this code.\n',
                'The cell runs all stages automatically, flushes Drive, then requests '
                'runtime release. Verify the saved receipt from outside Colab.\n',
                'Training stays locked until the same source bundle passes external '
                'readback and a sufficient cumulative CU budget is explicitly configured.\n']},
            {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
             'outputs': [], 'source': source.splitlines(keepends=True)},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'notebooks' / 'managed-p0.ipynb')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + '\n',
                           encoding='utf-8')
    print(f'CPU transport notebook written: {args.output.name}')


if __name__ == '__main__':
    main()
