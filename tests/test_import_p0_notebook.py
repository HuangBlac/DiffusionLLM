import json
import pytest
from experiments.import_p0_notebook import extract


def notebook(bad=None):
    lines = []
    for method in ('dlm', 'ar'):
        for phase in ('train', 'resume'):
            value = {'status': 'completed', 'end_step': 20, 'returncode': 0,
                     'restored_step19_matches': True, 'step20_matches': True,
                     'step20_stats_match': True}
            if bad == method + '-' + phase:
                value['step20_matches'] = False
            lines.append(f'RESULT {method} {phase} {json.dumps(value)}\n')
    lines.append('P0_DONE {"dlm-train": {"history": [')
    return {'cells': [{'outputs': [{'output_type': 'stream', 'text': lines}]}]}


def test_truncated_full_report_does_not_invalidate_complete_phase_summaries():
    result = extract(notebook(), 0)
    assert len(result['phases']) == 4
    assert result['checkpoint_recovered'] is False
    assert result['full_report_recovered'] is False
    assert result['p1_resume_ready'] is False


def test_failed_replay_or_missing_phase_is_rejected():
    with pytest.raises(ValueError, match='replay'):
        extract(notebook('ar-resume'), 0)
    n = notebook()
    n['cells'][0]['outputs'][0]['text'].pop(0)
    with pytest.raises(ValueError, match='Missing'):
        extract(n, 0)
