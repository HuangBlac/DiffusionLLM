import time
import pytest
from experiments import learning_run_evaluation as ev
from experiments.p2_canvas_ablation import pack_canvas


def test_schedule_and_suffix_targets():
    assert len(ev.generation_schedule(0)) == 1
    assert len(ev.generation_schedule(512)) * 8 == 72
    ids, valid, targets = pack_canvas([{'ids':[2,3,4,0], 'prefix_length':2}], 0, 'cpu', target_capacity=8)
    assert targets.sum() == 2 and not targets[~valid].any()
    assert ids.shape == (1, 11)


def test_generation_counts_and_deadline(monkeypatch):
    rows = [{'prefix_sha256': str(i), 'source_row': i} for i in range(100)]
    model = type('Model', (), {'method':'dlm'})()
    monkeypatch.setattr(ev, 'generate', lambda *a, **k: k)
    assert len(ev.evaluate_generations(model, None, rows, 512, time.monotonic()+1)) == 72
    model.method = 'ar'
    assert len(ev.evaluate_generations(model, None, rows, 512, time.monotonic()+1)) == 8
    with pytest.raises(TimeoutError):
        ev._guard(time.monotonic()-1)


def test_metrics_stop_at_first_eos():
    m = ev._metrics([3,3,3,3,3,0,3], 0)
    assert m['delivered_tokens'] == 5 and m['adjacent_repetitions'] == 4
    assert m['repeated_4grams'] == 1 and m['natural_eos']
