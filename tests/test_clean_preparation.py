"""A source-only Colab bundle has no pre-existing results directory."""
import hashlib
from types import SimpleNamespace

from experiments import prepare_data


def test_prepare_data_creates_reports_in_a_fresh_source_workspace(tmp_path, monkeypatch):
    weights = tmp_path / 'data' / 'model' / 'model.safetensors'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'fixture weights, not a neural model')
    monkeypatch.setattr(prepare_data, 'ROOT', tmp_path)
    monkeypatch.setattr(prepare_data, 'WEIGHTS_SHA256', hashlib.sha256(weights.read_bytes()).hexdigest())
    monkeypatch.setattr(prepare_data, 'snapshot_download', lambda *a, **k: None)
    monkeypatch.setattr(prepare_data, 'hf_hub_download', lambda *a, **k: a[1])
    tokenizer = SimpleNamespace(encode=lambda text, **k: [int(text)] * 128)
    monkeypatch.setattr(prepare_data.AutoTokenizer, 'from_pretrained', lambda *a, **k: tokenizer)

    def table(path, **kwargs):
        offset = 0 if 'train-' in path else 2000 if 'validation-' in path else 4000
        values = [str(offset + i) for i in range(1100)]
        return SimpleNamespace(column=lambda name: SimpleNamespace(to_pylist=lambda: values))

    monkeypatch.setattr(prepare_data.pq, 'read_table', table)
    assert not (tmp_path / 'results').exists()
    prepare_data.main()
    assert (tmp_path / 'results' / 'data_manifest.json').is_file()
    assert (tmp_path / 'data' / 'train.json').is_file()
