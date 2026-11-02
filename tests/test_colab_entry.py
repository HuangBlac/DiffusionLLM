"""Check the generated entry and its early-failure shutdown paths without Colab."""
import ast
import base64
import hashlib
import io
import sys
import types
import zipfile

import pytest

from experiments.build_colab_notebook import SOURCES, build


def test_notebook_has_one_code_cell_defaults_to_cpu_and_packs_only_reviewed_source():
    notebook = build()
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 1
    assert not notebook['metadata'].get('accelerator')
    source = ''.join(cells[0]['source'])
    tree = ast.parse(source)
    assert 'MODE = "transport"' in source
    assert 'VERIFIED_TRANSPORT_SHA256 = None' in source
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    payload = next(n.value.args[0].value for n in ast.walk(function)
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                       and t.id == 'source_bytes' for t in n.targets))
    hashes = ast.literal_eval(next(n.value for n in ast.walk(function)
                                  if isinstance(n, ast.Assign)
                                  and any(isinstance(t, ast.Name) and t.id == 'source_hashes'
                                          for t in n.targets)))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        assert set(archive.namelist()) == {'experiments/' + n for n in SOURCES}
        for name in archive.namelist():
            assert hashlib.sha256(archive.read(name)).hexdigest() == hashes[name]


@pytest.mark.parametrize('mode', ['invalid', 'p0'])
def test_early_configuration_failure_releases_without_mount_or_gpu(monkeypatch, mode):
    calls = []
    colab = types.ModuleType('google.colab')
    colab.drive = types.SimpleNamespace(mount=lambda *a, **k: calls.append('mount'))
    colab.runtime = types.SimpleNamespace(unassign=lambda: calls.append('release'))
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    source = ''.join(build()['cells'][1]['source']).replace(
        'MODE = "transport"', f'MODE = "{mode}"')
    with pytest.raises(ValueError, match='Unknown mode|P0 locked'):
        exec(compile(source, '<notebook-test>', 'exec'), {})
    assert calls == ['release']


def test_transport_and_training_exercise_identical_mount_timeout_policy():
    """A CPU pass must not certify a different GPU mount timeout."""
    source = ''.join(build()['cells'][1]['source'])
    tree = ast.parse(source)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'mount')
    timeout = next(k.value for k in call.keywords if k.arg == 'timeout_ms')
    values = [eval(compile(ast.Expression(timeout), '<mount-policy>', 'eval'),
                   {'MODE': mode, 'MOUNT_TIMEOUT_MS': 600000})
              for mode in ('transport', 'p0')]
    assert values[0] == values[1], 'CPU verification and GPU execution use different timeouts'
