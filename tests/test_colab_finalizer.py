"""Exercise actual notebook shutdown code with a clock and Drive substitutes."""
import ast
import copy
import sys
from types import SimpleNamespace

import pytest

from experiments.build_colab_notebook import build
import experiments.managed_run as managed


def entry_tree():
    return ast.parse(''.join(build()['cells'][1]['source']))


def compile_nodes(nodes):
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                   '<actual-colab-shutdown>', 'exec')


def budget_setup(cap=8, spent=3, clock=(1000, 1007)):
    function = next(n for n in entry_tree().body
                    if isinstance(n, ast.FunctionDef) and n.name == 'run')
    source_index = next(i for i, node in enumerate(function.body)
                        if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'source_bytes'
                                for t in node.targets))
    ticks = iter(clock)
    commands = []
    namespace = {
        'MODE': 'p1', 'CAP_CU': cap, 'SPENT_CU': spent, 'RATE_CU_HOUR': 6.73,
        'RELEASE_RESERVE_CU': .5, 'ACCOUNTING_RESERVE_CU': .1,
        'VERIFIED_TRANSPORT_RELATIVE_PATH': 'receipt.json',
        'VERIFIED_TRANSPORT_SHA256': 'verified',
        'time': SimpleNamespace(monotonic=lambda: next(ticks)), 'sys': sys,
        'subprocess': SimpleNamespace(DEVNULL=-3,
                                     Popen=lambda argv, **kwargs: commands.append(argv)),
    }
    # Isolate the retired entry's arithmetic; test its real refusal separately.
    nodes = function.body[:source_index]
    retirement_gates = [node for node in nodes if isinstance(node, ast.If)
                        and isinstance(node.test, ast.Compare)
                        and isinstance(node.test.left, ast.Name)
                        and node.test.left.id == 'MODE'
                        and any(isinstance(child, ast.Constant)
                                and isinstance(child.value, str)
                                and 'Legacy DriveFS training is disabled' in child.value
                                for child in ast.walk(node))]
    assert len(retirement_gates) == 1
    exec(compile_nodes([node for node in nodes if node not in retirement_gates]), namespace)
    return namespace, commands


def test_work_flush_and_accounting_boundaries_and_absolute_watchdog():
    namespace, commands = budget_setup()
    rate = namespace['RATE_CU_HOUR']
    work = namespace['time_cap']
    hard = namespace['hard_seconds']
    full = (8 - 3) * 3600 / rate
    assert work * rate / 3600 <= 4.5
    assert hard * rate / 3600 <= 4.9
    assert work < hard < full
    assert namespace['flush_window_seconds'] == hard - work
    assert float(commands[0][-1]) == hard - 7


def test_remaining_point_68_cu_cannot_launch_a_new_training_attempt():
    with pytest.raises(ValueError, match='P0 locked'):
        budget_setup(cap=8, spent=7.32)


@pytest.mark.parametrize('mode', ['p0', 'p1'])
def test_actual_legacy_entry_refuses_training_before_mount_or_watchdog(monkeypatch, mode):
    calls = []
    colab = SimpleNamespace(
        drive=SimpleNamespace(mount=lambda *a, **kw: calls.append('mount')),
        runtime=SimpleNamespace(unassign=lambda: calls.append('release')))
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    source = ''.join(build()['cells'][1]['source']).replace(
        'MODE = "transport"', f'MODE = "{mode}"')
    with pytest.raises(RuntimeError, match='Legacy DriveFS training is disabled'):
        exec(compile(source, '<real-retired-entry>', 'exec'), {})
    assert calls == ['release']


@pytest.fixture
def finalizer(monkeypatch):
    outer = next(n for n in entry_tree().body if isinstance(n, ast.Try))
    code = compile_nodes(outer.finalbody)

    def run(now=1000, hard=1300, window=210, mounted=True, failure=None):
        events = []
        live_mount = [mounted]

        def persist(path, value):
            assert live_mount[0], 'Must not write into an unmounted Drive path'
            events.append(('persist', copy.deepcopy(value)))

        def flush(*, timeout_ms):
            events.append(('flush', timeout_ms))
            if failure is not None:
                raise failure
            live_mount[0] = False

        monkeypatch.setattr(managed, 'atomic_json', persist)
        monkeypatch.setattr(managed, 'utc_now', lambda: 'clock-time')
        namespace = {
            'mounted': mounted, 'lifecycle': {'status': 'completed'},
            'lifecycle_path': 'Drive/lifecycle.json',
            'hard_release_deadline': hard, 'flush_window_seconds': window,
            'time': SimpleNamespace(monotonic=lambda: now),
            'drive': SimpleNamespace(flush_and_unmount=flush),
            'runtime': SimpleNamespace(unassign=lambda: events.append(('release',))),
        }
        try:
            exec(code, namespace)
        except BaseException as exc:
            return namespace, events, exc
        return namespace, events, None

    return run


def test_flush_uses_reserve_and_never_writes_after_successful_unmount(finalizer, capsys):
    namespace, events, error = finalizer()
    assert error is None
    assert [event[0] for event in events] == ['persist', 'flush', 'release']
    assert events[0][1]['cloud_flush'] == 'sync_pending'
    assert events[1] == ('flush', 210000)
    assert namespace['lifecycle']['cloud_flush'] != 'verified'
    assert 'DRIVE_FLUSH_RETURNED' in capsys.readouterr().out


def test_elapsed_work_reduces_flush_timeout_instead_of_resetting_it(finalizer):
    _, events, error = finalizer(now=1250)
    assert error is None
    assert ('flush', 20000) in events


def test_exhausted_reserve_skips_flush_and_releases(finalizer, capsys):
    _, events, error = finalizer(now=1271)
    assert isinstance(error, TimeoutError)
    assert [event[0] for event in events] == ['persist', 'release']
    assert 'DRIVE_FLUSH_SKIPPED_BUDGET' in capsys.readouterr().out


def test_flush_failure_remains_visible_and_always_releases(finalizer, capsys):
    failure = ValueError('upload timeout')
    namespace, events, error = finalizer(failure=failure)
    assert error is failure
    assert [event[0] for event in events] == ['persist', 'flush', 'release']
    assert namespace['lifecycle']['cloud_flush'] == 'sync_pending'
    output = capsys.readouterr().out
    assert 'DRIVE_FLUSH_FAILED' in output
    assert 'DRIVE_FLUSH_RETURNED' not in output


def test_unmounted_early_failure_only_requests_release(finalizer):
    _, events, error = finalizer(mounted=False, hard=None)
    assert error is None
    assert events == [('release',)]
