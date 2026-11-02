"""Run bounded child processes and keep evidence outside temporary compute storage.

This module uses only the standard library. It does not connect to Colab, mount
Drive, acquire GPUs, download files, or change permissions on its own.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Callable, Mapping, Sequence
import uuid


@dataclass(frozen=True)
class Stage:
    """One command; a training command must also supply a checked JSON report."""

    name: str
    command: tuple[str, ...]
    report: str | None = None
    checkpoint: str | None = None
    minimum_seconds: float = 0
    maximum_seconds: float = 300
    expected: Mapping[str, object] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Flush, atomically replace, and read back; propagate all storage failures."""
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
               + '\n').encode('utf-8')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if path.read_bytes() != payload:
        raise OSError(f'Storage readback differs: {path.name}')


def seconds_available(cap_cu: float, spent_cu: float, rate_cu_hour: float,
                      reserve_cu: float = .1) -> int:
    """Carry prior attempts forward; reserve CU for connection/release overhead."""
    values = (cap_cu, spent_cu, rate_cu_hour, reserve_cu)
    if (not all(math.isfinite(v) for v in values) or cap_cu <= 0
            or spent_cu < 0 or rate_cu_hour <= 0 or reserve_cu < 0):
        raise ValueError('Budget, spending, rate and reserve must be finite and valid')
    seconds = math.floor((cap_cu - spent_cu - reserve_cu) * 3600 / rate_cu_hour)
    if seconds <= 0:
        raise ValueError('No execution budget remains after prior attempts and reserve')
    return seconds


def safe_path(root: Path, relative: str) -> Path:
    parts = relative.replace('\\', '/').split('/')
    if (not relative or ':' in relative or any(p in ('', '.', '..') for p in parts)
            or Path(relative).is_absolute()):
        raise ValueError(f'Artifact must be a relative path inside the run: {relative}')
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Artifact escapes the run directory')
    return target


def validate_stages(stages: Sequence[Stage], root: Path) -> None:
    names = set()
    for stage in stages:
        if (not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', stage.name)
                or stage.name in names or not stage.command):
            raise ValueError('Stages need unique safe names and nonempty argv commands')
        names.add(stage.name)
        if (not math.isfinite(stage.minimum_seconds) or stage.minimum_seconds < 0
                or not math.isfinite(stage.maximum_seconds)
                or stage.maximum_seconds <= 0):
            raise ValueError('Stage time limits must be finite and nonnegative')
        if stage.checkpoint and not stage.report:
            raise ValueError('A checkpoint requires a report from its producing stage')
        for relative in (stage.report, stage.checkpoint):
            if relative:
                safe_path(root, relative)
                if relative in ('run.json', 'storage-probe.json', 'heartbeat.json'):
                    raise ValueError('Artifact collides with a managed status file')
    if not stages:
        raise ValueError('An empty pipeline cannot establish a completed run')


def terminate_child(process: subprocess.Popen) -> None:
    """Kill the Linux process group, including descendants, on budget expiry."""
    if process.poll() is None:
        if os.name == 'posix':
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=10)


def artifact(root: Path, relative: str, kind: str) -> dict:
    path = safe_path(root, relative)
    size = path.stat().st_size
    if not path.is_file() or size == 0:
        raise ValueError(f'Missing or empty {kind}: {relative}')
    return {'path': relative, 'kind': kind, 'bytes': size, 'sha256': digest(path)}


def checked_artifacts(stage: Stage, root: Path) -> list[dict]:
    result = []
    if stage.report:
        report = json.loads(safe_path(root, stage.report).read_text(encoding='utf-8'))
        if report.get('status') != 'completed':
            raise ValueError(f'{stage.name}: child report is not completed')
        for key, expected in stage.expected.items():
            if report.get(key) != expected:
                raise ValueError(f'{stage.name}: required report field differs: {key}')
        result.append(artifact(root, stage.report, 'report'))
        if stage.checkpoint:
            checkpoint = artifact(root, stage.checkpoint, 'checkpoint')
            # The child must hash the checkpoint it actually wrote.
            if report.get('checkpoint_sha256') != checkpoint['sha256']:
                raise ValueError(f'{stage.name}: checkpoint digest does not match report')
            result.append(checkpoint)
    return result


def _execute(stages: Sequence[Stage], run_dir: Path, *, budget_seconds: float,
             reserve_seconds: float, heartbeat_seconds: float, started: float) -> dict:
    root = Path(run_dir).resolve()
    validate_stages(stages, root)
    if (not math.isfinite(budget_seconds) or not math.isfinite(reserve_seconds)
            or budget_seconds <= 0 or not 0 <= reserve_seconds < budget_seconds
            or not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0):
        raise ValueError('Execution needs a finite positive budget and save reserve')
    root.mkdir(parents=True, exist_ok=False)
    deadline = started + budget_seconds
    state = {'schema': 1, 'status': 'running', 'started_utc': utc_now(),
             'budget_seconds': budget_seconds, 'reserve_seconds': reserve_seconds,
             'stages': [{'name': s.name, 'status': 'not_started'} for s in stages],
             'release': {'status': 'not_requested'}}
    status_path = root / 'run.json'
    try:
        atomic_json(root / 'storage-probe.json', {'nonce': uuid.uuid4().hex})
        atomic_json(status_path, state)
        for stage, record in zip(stages, state['stages']):
            available = min(stage.maximum_seconds,
                            deadline - time.monotonic() - reserve_seconds)
            if available <= stage.minimum_seconds:
                state.update(status='budget_exhausted', error=f'Insufficient time for {stage.name}')
                break
            record.update(status='running', started_utc=utc_now())
            atomic_json(status_path, state)
            child_deadline = min(deadline - reserve_seconds,
                                 time.monotonic() + stage.maximum_seconds)
            available = child_deadline - time.monotonic()
            if available <= stage.minimum_seconds:
                state.update(status='budget_exhausted', error=f'Insufficient time for {stage.name}')
                record['status'] = 'not_started'
                break
            argv = [str(max(1, math.floor(available))) if arg == '{max_seconds}' else arg
                    for arg in stage.command]
            record['command'] = argv
            log_path = root / (stage.name + '.log')
            with log_path.open('xb', buffering=0) as log:
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=(os.name == 'posix'))
                heartbeat_at = 0
                try:
                    while process.poll() is None:
                        now = time.monotonic()
                        if now >= child_deadline:
                            raise subprocess.TimeoutExpired(argv, available)
                        if now >= heartbeat_at:
                            atomic_json(root / 'heartbeat.json',
                                        {'stage': stage.name, 'time_utc': utc_now(),
                                         'elapsed_seconds': now - started})
                            heartbeat_at = now + heartbeat_seconds
                        time.sleep(min(.1, max(.001, child_deadline - now)))
                    record['returncode'] = process.returncode
                    if process.returncode != 0:
                        raise RuntimeError(f'{stage.name}: child exited {process.returncode}; see log')
                finally:
                    terminate_child(process)
                    os.fsync(log.fileno())
            record['artifacts'] = checked_artifacts(stage, root)
            record.update(status='completed', finished_utc=utc_now())
            atomic_json(status_path, state)
            print(f'STAGE_COMPLETED {stage.name}', flush=True)
        else:
            state['status'] = 'completed'
    except subprocess.TimeoutExpired as exc:
        state.update(status='timed_out', error=str(exc))
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    except BaseException as exc:
        state.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for record in state['stages']:
            if record['status'] == 'running':
                record['status'] = state['status']
        state.update(elapsed_seconds=time.monotonic() - started, finished_utc=utc_now())
        atomic_json(status_path, state)
    return state


def execute(stages: Sequence[Stage], run_dir: Path, *, budget_seconds: float,
            reserve_seconds: float = 30, release: Callable[[], None] | None = None,
            heartbeat_seconds: float = 10) -> dict:
    """Run once; release on preflight failure or blocked storage at the deadline.

    The watchdog never touches storage. A returned release request does not
    prove zero active Colab sessions; read the resource panel separately.
    """
    started = time.monotonic()
    release_state = {'status': 'not_requested'}
    release_lock = threading.Lock()

    def request_release() -> None:
        with release_lock:
            if release_state['status'] != 'not_requested':
                return
            release_state.update(status='requested', time_utc=utc_now())
        try:
            release()
        except Exception as exc:
            release_state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        else:
            release_state['status'] = 'request_returned'

    guard = None
    result = None
    try:
        if release is not None and math.isfinite(budget_seconds) and budget_seconds > 0:
            guard = threading.Timer(budget_seconds, request_release)
            guard.daemon = True
            guard.start()
        result = _execute(stages, run_dir, budget_seconds=budget_seconds,
                          reserve_seconds=reserve_seconds,
                          heartbeat_seconds=heartbeat_seconds, started=started)
        return result
    finally:
        if guard is not None:
            guard.cancel()
        if release is not None:
            request_release()
        if result is not None:
            result['release'] = dict(release_state)
            atomic_json(Path(run_dir).resolve() / 'run.json', result)


def verify_receipt(run_dir: Path) -> dict:
    """Reopen saved evidence independently, including after the compute VM exits."""
    root = Path(run_dir).resolve()
    try:
        state = json.loads((root / 'run.json').read_text(encoding='utf-8'))
        if state.get('status') != 'completed' or not state.get('stages'):
            raise ValueError('Run did not complete')
        latest = {}
        for record in state['stages']:
            if record['status'] != 'completed':
                raise ValueError('A stage is incomplete')
            for entry in record.get('artifacts', []):
                latest[entry['path']] = entry
        for entry in latest.values():
            actual = artifact(root, entry['path'], entry['kind'])
            if actual != entry:
                raise ValueError(f"Artifact changed: {entry['path']}")
        return {'verified': True, 'artifacts': list(latest.values())}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'verified': False, 'error': f'{type(exc).__name__}: {exc}'}
