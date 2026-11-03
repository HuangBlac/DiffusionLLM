"""Recover verifiable P0 summaries from one saved notebook cell, not checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def extract(notebook, cell_index):
    cell = notebook['cells'][cell_index]
    outputs = ''.join(''.join(o.get('text', [])) for o in cell.get('outputs', [])
                      if o.get('output_type') == 'stream')
    phases = {}
    for line in outputs.splitlines():
        match = re.fullmatch(r'RESULT (ar|dlm) (train|resume) (\{.*\})', line)
        if match:
            key = match[1] + '-' + match[2]
            value = json.loads(match[3])
            if key in phases and phases[key] != value:
                raise ValueError('Conflicting results within selected cell')
            phases[key] = value
    expected = {'ar-train', 'ar-resume', 'dlm-train', 'dlm-resume'}
    if phases.keys() != expected:
        raise ValueError('Missing one or more P0 phases')
    for key, value in phases.items():
        if (value.get('status') != 'completed' or value.get('end_step') != 20
                or value.get('returncode') != 0):
            raise ValueError(f'Incomplete/failed phase: {key}')
        if key.endswith('-resume'):
            flags = ('restored_step19_matches', 'step20_matches', 'step20_stats_match')
            if not all(value.get(flag) is True for flag in flags):
                raise ValueError(f'GPU replay verification failed: {key}')
    steps = {}
    for line in outputs.splitlines():
        line = line.strip()
        if line.startswith('{'):
            value = json.loads(line)
            if 'step' in value:
                key = f"{value['method']}-{value['phase']}-{value['step']}"
                if key in steps and steps[key] != value:
                    raise ValueError('Conflicting step logs')
                steps[key] = value
    # The wrapper explicitly slices the combined report to 6000 characters.
    return {'status': 'gpu_training_and_replay_verified_from_saved_output',
            'source_cell_index': cell_index, 'phases': phases,
            'selected_step_logs': list(steps.values()),
            'full_report_recovered': False, 'checkpoint_recovered': False,
            'p1_resume_ready': False,
            'limitations': ['P0_DONE is truncated; summaries are not complete training reports.',
                            'Saved stdout cannot replace model and optimizer checkpoint bytes.',
                            'No generation quality or matched AR/DLM loss comparison is established.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--cell', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.notebook.read_bytes()
    result = extract(json.loads(raw), args.cell)
    result['notebook_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'phases': list(result['phases']),
                      'checkpoint_recovered': False}))


if __name__ == '__main__':
    main()
