"""Build a locked P2 Colab handoff with a verifiable, local source bundle.

The notebook is deliberately a preparation artifact, not a training runner.
"""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('experiments/p2_paired_training.py', 'experiments/p2_canvas_ablation.py',
           'experiments/full_parameter_pilot.py', 'experiments/p2_paired_entry.py',
           'experiments/p2_restore_check.py', 'experiments/prepare_data.py',
           'experiments/drive_checkpoint_store.py', 'experiments/api_checkpoint_training.py',
           'experiments/managed_run.py')


def build(proof=None, parent_folder=None):
    module = ast.parse((ROOT / 'experiments/prepare_data.py').read_text(encoding='utf-8'))
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in module.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ('MODEL', 'MODEL_REVISION')}
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.writestr(name, (ROOT / name).read_bytes())
    bundle = base64.b64encode(stream.getvalue()).decode('ascii')
    cell = '''# Default is locked. Fill a freshly observed resource record before RUN=True.
RUN = False
RESOURCE_OBSERVATION = None
PARENT_FOLDER_ID = {parent!r}
INPUT_PROOF = {proof!r}
import base64, hashlib, io, json, zipfile, os, sys, subprocess, time, uuid, math
from pathlib import Path
from datetime import datetime, timezone
SOURCE_HASHES = {hashes}
BUNDLE = {bundle}
if RUN:
    from google.colab import runtime, auth
    try:
        o = RESOURCE_OBSERVATION
        if not isinstance(o, dict) or not PARENT_FOLDER_ID or not INPUT_PROOF:
            raise ValueError("Set fresh resource observation, proof and destination folder")
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(o['observed_utc'].replace('Z','+00:00'))).total_seconds()
        values = [o[k] for k in ('start_balance','current_balance','cu_per_hour','reserve_cu','overhead_seconds')]
        if not all(math.isfinite(v) and v > 0 for v in values) or o['current_balance'] > o['start_balance']:
            raise ValueError("Invalid resource values")
        remaining = min(4 - (o['start_balance'] - o['current_balance']), o['current_balance'])
        if not 0 <= age <= 120 or remaining <= o['reserve_cu']:
            raise ValueError("Stale observation or insufficient P2 budget")
        deadline = time.monotonic() - age + remaining * 3600 / o['cu_per_hour']
    except BaseException:
        runtime.unassign()
        raise
    guard_code = "import sys,time; time.sleep(float(sys.argv[1])); from google.colab import runtime; runtime.unassign()"
    def bounded(argv, **kwargs):
        remaining_seconds = deadline-time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError("P2 deadline expired")
        return subprocess.run(argv, check=True, timeout=remaining_seconds, **kwargs)
    try:
        subprocess.Popen([sys.executable, '-c', guard_code, str(max(0, deadline-time.monotonic()))],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        auth.authenticate_user()
        work = Path('/content') / ('p2-' + uuid.uuid4().hex)
        work.mkdir()
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as archive:
            if set(archive.namelist()) != set(SOURCE_HASHES):
                raise ValueError("Source bundle membership changed")
            for name in archive.namelist():
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != SOURCE_HASHES[name]:
                    raise ValueError("Source digest mismatch: " + name)
                dest = work / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        proof_path = work / 'proof.json'
        proof_path.write_text(json.dumps(INPUT_PROOF), encoding='utf-8')
        observation_path = work / 'observation.json'
        observation_path.write_text(json.dumps(o), encoding='utf-8')
        env = {{**os.environ, 'USE_TF':'0', 'USE_FLAX':'0', 'HF_HUB_DISABLE_XET':'1'}}
        venv = work / 'venv'
        bounded([sys.executable, '-m', 'venv', str(venv)])
        python = str(venv / 'bin' / 'python')
        bounded([python, '-m', 'pip', 'install', 'torch==2.6.0', 'transformers==4.57.6',
                 'google-api-python-client', 'google-auth-httplib2', 'huggingface_hub'], env=env)
        model_dir = work / 'model'
        download = "from huggingface_hub import snapshot_download; import sys; snapshot_download({model!r}, revision={revision!r}, local_dir=sys.argv[1], allow_patterns=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','model.safetensors'])"
        bounded([python, '-c', download, str(model_dir)], env=env, cwd=work)
        bounded([python, '-m', 'experiments.p2_paired_entry', '--observation', str(observation_path),
                 '--proof', str(proof_path), '--model-path', str(model_dir), '--staging', str(work / 'run'),
                 '--folder-id', PARENT_FOLDER_ID, '--deadline', str(deadline)], env=env, cwd=work)
    finally:
        runtime.unassign()
else:
    print("P2 is locked; no connection or training started")
'''.format(hashes=repr(hashes), bundle=repr(bundle), model=constants['MODEL'], revision=constants['MODEL_REVISION'], proof=proof, parent=parent_folder)
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
            'cells': [{'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                       'outputs': [], 'source': cell.splitlines(keepends=True)}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--parent-folder')
    args = parser.parse_args()
    proof = None
    if args.proof:
        original = json.loads(args.proof.read_text(encoding='utf-8'))
        proof = {key: {split: original[key][split] for split in ('training', 'development')}
                 for key in ('counts', 'artifacts', 'artifact_files')}
        proof['tokenizer_sha256'] = original['tokenizer_sha256']
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(build(proof, args.parent_folder), f, ensure_ascii=False, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
