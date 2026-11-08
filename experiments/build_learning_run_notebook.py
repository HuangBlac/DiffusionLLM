"""Build the locked learning-run notebook and immutable source bundle."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = tuple('experiments/' + name for name in (
    'learning_run_entry.py', 'learning_run_restore.py', 'learning_run_training.py',
    'learning_run_evaluation.py', 'p2_paired_training.py', 'p2_canvas_ablation.py',
    'full_parameter_pilot.py', 'prepare_data.py', 'drive_checkpoint_store.py',
    'api_checkpoint_training.py', 'managed_run.py', 'p2_paired_entry.py'))


def build(proof=None, parent_folder=None, storage_cache=None):
    if proof is not None:
        original = proof
        proof = {key: {split: original[key][split] for split in ('training', 'development')}
                 for key in ('counts', 'artifacts', 'artifact_files')}
        proof['tokenizer_sha256'] = original['tokenizer_sha256']
        if proof['counts'] != {'training': 4096, 'development': 100}:
            raise ValueError('Expected fixed learning data counts')
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.writestr(name, (ROOT / name).read_bytes())
    bundle = base64.b64encode(stream.getvalue()).decode('ascii')
    auth_cell = '''# CPU authentication and a bounded small-file Drive API round trip.
RUN_CPU_AUTH = False
PARENT_FOLDER_ID = __PARENT__
if RUN_CPU_AUTH:
    import sys, subprocess, io, uuid
    from google.colab import auth, runtime
    guard = "import sys,time; time.sleep(180); from google.colab import runtime; runtime.unassign()"
    subprocess.Popen([sys.executable, '-c', guard], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        try:
            gpu = subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], capture_output=True, text=True, timeout=10)
            if gpu.returncode == 0 and gpu.stdout.strip():
                raise RuntimeError('Use CPU for this probe')
        except FileNotFoundError:
            pass
        print('WATCHDOG_STARTED_180_SECONDS', flush=True)
        print('AUTH_START', flush=True)
        auth.authenticate_user(clear_output=False)
        import google.auth
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseUpload
        creds, _ = google.auth.default()
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        data = ('learning-auth-roundtrip-' + uuid.uuid4().hex).encode()
        result = service.files().create(body={'name':'learning-cpu-auth-proof.txt','parents':[PARENT_FOLDER_ID]}, media_body=MediaIoBaseUpload(io.BytesIO(data), mimetype='text/plain'), fields='id').execute(num_retries=2)
        if service.files().get_media(fileId=result['id']).execute(num_retries=2) != data:
            raise ValueError('Drive API roundtrip differs')
        print('CPU_API_ROUNDTRIP_OK', result['id'], flush=True)
    finally:
        print('RELEASE_REQUESTED', flush=True)
        runtime.unassign()
else:
    print('CPU auth locked')
'''.replace('__PARENT__', repr(parent_folder))
    run_cell = '''# Set a fresh observation and the CPU API roundtrip file ID before enabling.
RUN = False
RESOURCE_OBSERVATION = None
CPU_ROUNDTRIP_FILE_ID = None
PARENT_FOLDER_ID = __PARENT__
INPUT_PROOF = __PROOF__
STORAGE_CACHE = __CACHE__
SOURCE_HASHES = __HASHES__
BUNDLE = __BUNDLE__
if RUN:
    import base64, hashlib, io, json, math, os, subprocess, sys, time, uuid, zipfile
    from pathlib import Path
    from datetime import datetime, timezone
    from google.colab import auth, runtime
    try:
        o = RESOURCE_OBSERVATION
        if not isinstance(o, dict) or not CPU_ROUNDTRIP_FILE_ID or not PARENT_FOLDER_ID or not INPUT_PROOF or not STORAGE_CACHE:
            raise ValueError('Missing observation, CPU receipt, proof, storage cache or runs folder')
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(o['observed_utc'].replace('Z','+00:00'))).total_seconds()
        for key in ('start_balance','current_balance','project_start_balance','cu_per_hour'):
            if isinstance(o.get(key), bool) or not isinstance(o.get(key), (int,float)) or not math.isfinite(o[key]) or o[key] <= 0:
                raise ValueError('Invalid resource values')
        if not o['current_balance'] <= o['start_balance'] <= o['project_start_balance']:
            raise ValueError('Inconsistent balances')
        if not 0 <= age <= 120 or o['other_active_sessions'] != 0 or not o.get('gpu_name') or not o.get('session_id'):
            raise ValueError('Observation stale or other sessions active')
        remaining = min(12 - (o['start_balance'] - o['current_balance']), o['current_balance'], 180 - (o['project_start_balance'] - o['current_balance']))
        if remaining <= 0 or o['cu_per_hour'] <= 0:
            raise ValueError('Budget exhausted')
        deadline = time.monotonic() - age + remaining * 3600 / o['cu_per_hour']
        guard = "import sys,time; time.sleep(float(sys.argv[1])); from google.colab import runtime; runtime.unassign()"
        subprocess.Popen([sys.executable, '-c', guard, str(max(0, deadline-time.monotonic()))], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        def bounded(argv, **kwargs):
            left = deadline - time.monotonic()
            if left <= 0: raise TimeoutError('Stage budget deadline')
            return subprocess.run(argv, check=True, timeout=left, **kwargs)
        auth_guard = subprocess.Popen([sys.executable, '-c', guard, '180'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        print('AUTH_START', flush=True)
        try:
            auth.authenticate_user(clear_output=False)
        finally:
            if auth_guard.poll() is None:
                auth_guard.terminate()
                auth_guard.wait(timeout=5)
        print('AUTH_OK', flush=True)
        work = Path('/content') / ('learning-' + uuid.uuid4().hex)
        work.mkdir()
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as archive:
            if set(archive.namelist()) != set(SOURCE_HASHES): raise ValueError('Source bundle membership differs')
            for name in archive.namelist():
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != SOURCE_HASHES[name]: raise ValueError('Source digest differs')
                dest = work / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        (work / 'proof.json').write_text(json.dumps(INPUT_PROOF), encoding='utf-8')
        (work / 'storage-cache.json').write_text(json.dumps(STORAGE_CACHE), encoding='utf-8')
        (work / 'observation.json').write_text(json.dumps(o), encoding='utf-8')
        env = {**os.environ, 'USE_TF':'0', 'USE_FLAX':'0', 'HF_HUB_DISABLE_XET':'1', 'PYTHONUNBUFFERED':'1'}
        python = sys.executable
        probe = subprocess.run([python, '-c', 'import torch,transformers,googleapiclient,google_auth_httplib2,huggingface_hub; assert torch.__version__.split("+")[0]=="2.6.0" and transformers.__version__=="4.57.6"'], env=env, capture_output=True, timeout=max(1, deadline-time.monotonic()))
        if probe.returncode:
            print('ENVIRONMENT_SETUP_START', sys.version.split()[0], flush=True)
            venv = work / 'venv'
            bounded([sys.executable, '-m', 'venv', '--without-pip', str(venv)])
            python = str(venv / 'bin' / 'python')
            bounded([sys.executable, '-m', 'pip', '--python', python, 'install', 'torch==2.6.0', 'transformers==4.57.6', 'google-api-python-client', 'google-auth-httplib2', 'huggingface_hub'], env=env)
        bounded([python, '-c', 'import torch,transformers; assert torch.__version__.split("+")[0]=="2.6.0" and transformers.__version__=="4.57.6"; print("ENVIRONMENT_READY",torch.__version__,transformers.__version__,flush=True)'], env=env)
        model_dir = work / 'model'
        print('PINNED_MODEL_DOWNLOAD', flush=True)
        download = "from huggingface_hub import snapshot_download; import sys; snapshot_download('Qwen/Qwen2.5-0.5B', revision='060db6499f32faf8b98477b0a26969ef7d8b9987', local_dir=sys.argv[1], allow_patterns=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','model.safetensors'])"
        bounded([python, '-c', download, str(model_dir)], env=env, cwd=work)
        print('LEARNING_ENTRY_START', flush=True)
        bounded([python, '-m', 'experiments.learning_run_entry', '--observation', str(work / 'observation.json'),
                 '--proof', str(work / 'proof.json'), '--model-path', str(model_dir), '--staging', str(work / 'run'),
                 '--folder-id', PARENT_FOLDER_ID, '--deadline', str(deadline), '--cpu-receipt', CPU_ROUNDTRIP_FILE_ID,
                 '--storage-cache', str(work / 'storage-cache.json')],
                env=env, cwd=work)
    finally:
        print('RELEASE_REQUESTED', flush=True)
        runtime.unassign()
else:
    print('Learning run locked; no connection or training started')
'''.replace('__PARENT__', repr(parent_folder)).replace('__PROOF__', repr(proof)).replace('__CACHE__', repr(storage_cache)).replace('__HASHES__', repr(hashes)).replace('__BUNDLE__', repr(bundle))
    config_cell, run_cell = run_cell.split('PARENT_FOLDER_ID = ', 1)
    run_cell = 'PARENT_FOLDER_ID = ' + run_cell
    for cell in (auth_cell, config_cell, run_cell):
        compile(cell, '<learning-notebook-cell>', 'exec')
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name':'python3','display_name':'Python 3'}},
            'cells': [{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':cell.splitlines(keepends=True)} for cell in (auth_cell, config_cell, run_cell)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--parent-folder')
    parser.add_argument('--cached-storage-dir', type=Path)
    args = parser.parse_args()
    proof = None
    if args.proof:
        original = json.loads(args.proof.read_text(encoding='utf-8'))
        proof = {key: {split: original[key][split] for split in ('training', 'development')}
                 for key in ('counts', 'artifacts', 'artifact_files')}
        proof['tokenizer_sha256'] = original['tokenizer_sha256']
    cache = None
    if args.cached_storage_dir:
        from experiments.learning_run_entry import verified_cached_storage
        cache = {method: base64.b64encode((args.cached_storage_dir / ('benchmark-' + method + '.json')).read_bytes()).decode('ascii')
                 for method in ('ar', 'dlm')}
        verified_cached_storage(cache, proof, ROOT / 'experiments')
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(build(proof, args.parent_folder, cache), handle, ensure_ascii=False, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
