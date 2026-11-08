from experiments.build_p2_paired_notebook import build


def test_notebook_is_locked_by_default():
    cell = ''.join(build()['cells'][0]['source'])
    exec(cell, {})
    assert 'RESOURCE_OBSERVATION = None' in cell
    assert 'runtime.unassign()' in cell


def test_enabled_without_observation_releases_runtime(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    import pytest

    calls = []
    colab = ModuleType('google.colab')
    colab.runtime = SimpleNamespace(unassign=lambda: calls.append('released'))
    colab.auth = SimpleNamespace(authenticate_user=lambda: calls.append('authenticated'))
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    cell = ''.join(build()['cells'][0]['source']).replace('RUN = False', 'RUN = True', 1)
    with pytest.raises(ValueError, match='resource observation'):
        exec(cell, {})
    assert calls == ['released']
