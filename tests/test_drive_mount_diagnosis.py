import json
from types import SimpleNamespace
import sys
import pytest
import types

from experiments.diagnose_drive_mount import log_indicators, mount_attempt


def test_diagnosis_captures_branch_and_auth_duration_without_credentials():
    message = SimpleNamespace(blocking_request=lambda *a, **k: 'secret-credential')
    original_request = message.blocking_request

    def mount(*a, **k):
        assert mount._DEBUG is False
        secret = message.blocking_request('request_auth')
        assert secret == 'secret-credential'
        case = 2
        assert case == 2
        raise ValueError('mount failed secret-credential')

    drive = SimpleNamespace(_message=message, _mount=mount, mount=mount)
    mount._DEBUG = True
    original_trace = sys.gettrace()
    result = mount_attempt(drive, 120000)
    assert result['status'] == 'failed'
    assert result['auth_events'][0]['status'] == 'returned'
    assert result['drivefs_cases'][-1]['case'] == 2
    assert 'secret-credential' not in json.dumps(result)
    assert message.blocking_request is original_request and sys.gettrace() is original_trace
    assert mount._DEBUG is True


def test_logs_only_return_counts(tmp_path):
    (tmp_path / 'driver.log').write_text('oauth secret-credential token error https://private.invalid')
    result = log_indicators(SimpleNamespace(_logs_dir=lambda: tmp_path))
    assert result['counts']['oauth'] == 1
    assert 'secret-credential' not in json.dumps(result)
    assert 'private.invalid' not in json.dumps(result)


def test_guard_start_failure_still_releases(monkeypatch):
    import experiments.diagnose_drive_mount as module
    calls = []
    colab = types.ModuleType('google.colab')
    colab.drive = SimpleNamespace()
    colab.runtime = SimpleNamespace(unassign=lambda: calls.append('release'))
    monkeypatch.setitem(sys.modules, 'google.colab', colab)

    def fail(*a, **k):
        raise OSError('cannot start guard')

    monkeypatch.setattr(module.subprocess, 'Popen', fail)
    with pytest.raises(OSError, match='cannot start guard'):
        module.main()
    assert calls == ['release']


def test_failed_gpu_detection_does_not_start_mount(monkeypatch):
    import experiments.diagnose_drive_mount as module
    calls = []
    colab = types.ModuleType('google.colab')
    colab.drive = SimpleNamespace(mount=lambda *a, **k: calls.append('mount'))
    colab.runtime = SimpleNamespace(unassign=lambda: calls.append('release'))
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: None)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match='GPU detection failed'):
        module.main()
    assert calls == ['release']
