"""CPU-only Drive mount diagnosis; never output credentials or raw DriveFS logs."""
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def mount_attempt(drive, timeout_ms):
    started = time.monotonic()
    result = {'timeout_ms': timeout_ms, 'auth_events': [], 'drivefs_cases': []}
    original_request = drive._message.blocking_request
    original_trace = sys.gettrace()
    had_debug = hasattr(drive.mount, '_DEBUG')
    original_debug = getattr(drive.mount, '_DEBUG', False)
    drive.mount._DEBUG = False
    mount_code = drive._mount.__code__

    def request(*args, **kwargs):
        if not args or args[0] != 'request_auth':
            return original_request(*args, **kwargs)
        event = {'started_seconds': round(time.monotonic() - started, 3)}
        result['auth_events'].append(event)
        try:
            reply = original_request(*args, **kwargs)
            event['status'] = 'returned'
            return reply
        except Exception as exc:
            event.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            event['ended_seconds'] = round(time.monotonic() - started, 3)

    def trace(frame, event, arg):
        if frame.f_code is mount_code:
            case = frame.f_locals.get('case')
            if type(case) is int and 0 <= case <= 5:
                cases = result['drivefs_cases']
                if not cases or cases[-1]['case'] != case:
                    cases.append({'case': case, 'seconds': round(time.monotonic() - started, 3)})
            if event == 'exception':
                result['exception_line'] = frame.f_lineno
                result['root_timeout_hint'] = 'timeout' in frame.f_locals.get('extra_reason', '')
            return trace
        return None

    drive._message.blocking_request = request
    sys.settrace(trace)
    try:
        drive.mount('/content/drive', timeout_ms=timeout_ms)
        result['status'] = 'mounted'
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__,
                      mount_failed_message='mount failed' in str(exc))
    finally:
        sys.settrace(original_trace)
        drive._message.blocking_request = original_request
        if had_debug:
            drive.mount._DEBUG = original_debug
        else:
            del drive.mount._DEBUG
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return result


def log_indicators(drive):
    """Count a fixed vocabulary; never return log lines, URLs, or account IDs."""
    terms = ['timeout', 'timed out', 'credential', 'permission denied',
             'unauthorized', 'connection refused', 'connection reset',
             'failed to fetch', 'oauth', 'dns', 'token', 'error']
    counts = {term: 0 for term in terms}
    total = 0
    try:
        paths = sorted(Path(drive._logs_dir()).glob('*'))
        for path in paths:
            if not path.is_file():
                continue
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 512000))
                content = stream.read(512000).decode('utf-8', errors='replace').lower()
            total += len(content)
            for term in terms:
                counts[term] += content.count(term)
        return {'inspected_characters': total, 'counts': counts}
    except OSError as exc:
        return {'read_error_type': type(exc).__name__}


def main():
    from google.colab import drive, runtime
    # Independent fallback also covers a stuck mount or notebook-kernel reset.
    guard = "import time; time.sleep(1000); from google.colab import runtime; runtime.unassign()"
    report = {'kind': 'cpu_drive_mount_diagnosis', 'attempts': [], 'training_started': False}
    mounted = False
    try:
        subprocess.Popen([sys.executable, '-c', guard], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            gpu = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
        except FileNotFoundError:
            gpu = None
        if gpu is not None and gpu.returncode != 0:
            raise RuntimeError('GPU detection failed; cannot certify CPU-only diagnosis')
        if gpu is not None and gpu.returncode == 0 and 'GPU' in gpu.stdout:
            raise RuntimeError('CPU runtime required; refusing GPU diagnosis')
        report['gpu_absent'] = True
        report['drive_module_sha256'] = hashlib.sha256(inspect.getsource(drive).encode()).hexdigest()
        # Preserve only source code (not runtime values) for interpreting branch numbers.
        lines, first = inspect.getsourcelines(drive._mount)
        report['failure_branch_source'] = [
            {'line': first+i, 'source': line.rstrip()} for i, line in enumerate(lines)
            if re.search(r'case ==|case in|mount failed|timeout_pattern|extra_reason =', line)]
        for timeout in (120000, 600000):
            print('MOUNT_ATTEMPT_START', timeout, flush=True)
            attempt = mount_attempt(drive, timeout)
            attempt['log_indicators'] = log_indicators(drive)
            report['attempts'].append(attempt)
            print('MOUNT_ATTEMPT_RESULT', json.dumps(attempt), flush=True)
            mounted = attempt['status'] == 'mounted'
            if mounted:
                break
        report['status'] = 'mounted' if mounted else 'failed'
        if mounted:
            folder = Path('/content/drive/MyDrive/DiffusionLLM/diagnostics')
            folder.mkdir(parents=True, exist_ok=True)
            name = 'mount-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '.json'
            destination = folder / name
            payload = json.dumps(report, indent=2) + '\n'
            destination.write_text(payload)
            if destination.read_text() != payload:
                raise OSError('Diagnostic receipt readback differs')
            print('DIAGNOSTIC_RECEIPT', name, flush=True)
        print('MOUNT_DIAGNOSIS', json.dumps(report), flush=True)
    finally:
        try:
            if mounted:
                drive.flush_and_unmount(timeout_ms=45000)
                print('DRIVE_FLUSH_RETURNED', flush=True)
        finally:
            runtime.unassign()


if __name__ == '__main__':
    main()
