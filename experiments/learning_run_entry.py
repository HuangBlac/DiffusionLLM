"""Budget gates for the fixed 512-step learning run; importing never starts work.

The notebook owns the independent runtime release watchdog.
"""
import math
import time
from datetime import datetime, timezone


def resource_gate(observation, *, now=None):
    now = now or datetime.now(timezone.utc)
    observed = datetime.fromisoformat(observation['observed_utc'].replace('Z', '+00:00'))
    age = (now - observed).total_seconds()
    if not 0 <= age <= 120:
        raise ValueError('Resource observation must be at most 120 seconds old')
    names = ('start_balance', 'current_balance', 'project_start_balance', 'cu_per_hour')
    if any(not isinstance(observation.get(k), (int, float)) or isinstance(observation[k], bool)
           or not math.isfinite(observation[k]) or observation[k] <= 0 for k in names):
        raise ValueError('Missing positive resource values')
    if (observation['current_balance'] > observation['start_balance']
            or observation['start_balance'] > observation['project_start_balance']):
        raise ValueError('Inconsistent stage/project balances')
    if observation.get('other_active_sessions') != 0 or not observation.get('gpu_name') or not observation.get('session_id'):
        raise ValueError('Need observed GPU/session and no other active sessions')
    return age


def budget_deadline(observation, *, now_monotonic=None, now_utc=None):
    age = resource_gate(observation, now=now_utc)
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    remaining = min(12 - (observation['start_balance'] - observation['current_balance']),
                    180 - (observation['project_start_balance'] - observation['current_balance']),
                    observation['current_balance'])
    if remaining <= 0:
        raise RuntimeError('Stage or project budget exhausted')
    return now_monotonic - age + remaining * 3600 / observation['cu_per_hour']


def validate_lease(observation, deadline, *, now_utc=None, now_monotonic=None):
    # The caller validated freshness when the resource lease was created.
    now_utc = now_utc or datetime.now(timezone.utc)
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    observed = datetime.fromisoformat(observation['observed_utc'].replace('Z', '+00:00'))
    resource_gate(observation, now=observed)
    age = (now_utc - observed).total_seconds()
    if age < 0:
        raise ValueError('Observation is in the future')
    remaining = min(12 - (observation['start_balance'] - observation['current_balance']),
                    180 - (observation['project_start_balance'] - observation['current_balance']),
                    observation['current_balance'])
    allowed_deadline = now_monotonic - age + remaining * 3600 / observation['cu_per_hour']
    if not math.isfinite(deadline) or remaining <= 0 or deadline > allowed_deadline + 2 or deadline <= now_monotonic:
        raise ValueError('Deadline exceeds observed stage or project budget')


def preflight_fit(*, deadline, now, update_seconds, evaluation_seconds,
                  checkpoint_seconds, restore_seconds, reserve_seconds, safety_factor=1.5):
    """Require both complete 512 runs and measured overheads to fit.

    Evaluation and checkpoint inputs are representative per-event measurements;
    there are eight evaluation events, six saves and two final CPU audits.
    """
    values = (deadline, now, update_seconds, evaluation_seconds,
              checkpoint_seconds, restore_seconds, reserve_seconds, safety_factor)
    if any(not math.isfinite(v) or v <= 0 for v in values) or safety_factor < 1:
        raise ValueError('Positive measured times and safety factor required')
    projected = safety_factor * (1024 * update_seconds + 8 * evaluation_seconds
                                 + 6 * checkpoint_seconds + 2 * restore_seconds)
    if now + projected + reserve_seconds >= deadline:
        raise RuntimeError(f'Fixed 512-step pair requires {projected + reserve_seconds:.1f}s; only {deadline - now:.1f}s remains')
    return {'projected_seconds': projected, 'reserve_seconds': reserve_seconds,
            'safety_factor': safety_factor}



# The 2026-11-06 reports came from the prior full-size transfer benchmark.
# Digests lock exact locally retrieved report bytes and relevant storage/model code.
CACHED_REPORT_SHA256 = {
    'ar': '5782f45b425f6ad6d84f6346395876aea8fac5ebc9a1b709af80064ca0079ff5',
    'dlm': '7daf85ea038c942199d4b10469b0509168c0d39d404b93e69c32761d093ff590',
}
CACHED_SOURCE_SHA256 = {
    'prepare_data.py': '6c313c112abd6ccb7041fcbd9ce81e7622eacda7a59bda90bb48e6c07414ab50',
    'learning_run_training.py': 'ce4c6a36e8adfa5cc92e75c3c53b53cb0cadd55e9098a0000081e5cc299c8334',
    'p2_paired_training.py': '5be2ee8d9da7aa0b02da131b5a847ead3e0ff1c70cd7cfbe6069c25ede247195',
    'p2_canvas_ablation.py': '22e5cfaea76dd1eb49f186038ec0df2327473b56fab22ffc49c10ca6c7485f62',
    'full_parameter_pilot.py': 'c1fa177b3d314387a763c1b31d050f26a7f8d7c9036b6eb4da36bfd09d701897',
    'drive_checkpoint_store.py': '2fd96e0b53af139bfcac9a74f14ca157081bae699c98e9bc37203e4de3074b62',
    'api_checkpoint_training.py': 'c89d78791bb1aab7e46293c5d0a2aa3c4375e11b7b61e8a4a4d13b4d5b5ac5a7',
}
CACHED_DATA_SHA256 = {
    'training': '441fdb02dc692ecef396a32a987a39ed6e9a27e494baaf7fe38e4efb64704b02',
    'development': '902aa794853688f2a282d26e0a18d1c8b5acd3f67ccfca7e86e09c30d8428f90',
    'tokenizer': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
}


def verified_cached_storage(cache, proof, source_dir):
    """Verify immutable historical reports and the compatible data/storage code."""
    import base64
    import hashlib
    import json
    from pathlib import Path
    if not isinstance(cache, dict) or set(cache) != {'ar', 'dlm'}:
        raise ValueError('Both cached transfer reports are required')
    if any(proof['artifacts'][split]['sha256'] != CACHED_DATA_SHA256[split]
           for split in ('training', 'development')) or proof['tokenizer_sha256'] != CACHED_DATA_SHA256['tokenizer']:
        raise ValueError('Cached transfer data identity differs')
    for name, digest in CACHED_SOURCE_SHA256.items():
        if hashlib.sha256((Path(source_dir) / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Cached transfer source identity differs: ' + name)
    measurements = {}
    for method, encoded in cache.items():
        payload = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(payload).hexdigest() != CACHED_REPORT_SHA256[method]:
            raise ValueError('Cached transfer report digest differs: ' + method)
        report = json.loads(payload)
        fields = ('save_seconds', 'download_seconds', 'cpu_load_seconds')
        if any(isinstance(report.get(k), bool) or not isinstance(report.get(k), (int, float))
               or not math.isfinite(report[k]) or report[k] <= 0 for k in fields):
            raise ValueError('Cached transfer timing invalid')
        measurements[method] = {k: report[k] for k in fields}
    return measurements


def checkpoint_callback(*, writer, group, method, deadline, reserve, publish_report, clock=time.monotonic):
    """Save immutable server-verified milestones and publish their receipts."""
    def save(model, optimizer, step, order, masks, locked_identity, history):
        if step not in (128, 256, 512) or locked_identity.get('method') != method:
            raise ValueError('Checkpoint method or milestone differs')
        if clock() + reserve >= deadline:
            raise TimeoutError('Checkpoint would consume final reserve')
        writer(group / 'latest.pt', model, optimizer, step, order, masks, locked_identity, history)
        receipt = writer.last_server_receipt
        if not receipt or receipt.get('step') != step or receipt.get('status') != 'server_verified':
            raise RuntimeError('Checkpoint server verification missing')
        publish_report(method + '-checkpoint-' + str(step),
                       {'step': step, 'server_verified': True, 'receipt': receipt})
    return save


def run_pair(*, observation, deadline, store, folder_id, staging, proof, model_path,
             tokenizer, independent_restore, publish_report, cached_storage):
    """Measure a discarded full-state run, then execute the locked 512 pair."""
    import ast
    import gc
    import torch
    from pathlib import Path
    from transformers import AutoModelForCausalLM
    from experiments import full_parameter_pilot as pilot, p2_paired_training as paired
    from experiments import learning_run_training as training, learning_run_evaluation as evaluation
    from experiments.api_checkpoint_training import ApiCheckpointWriter
    from experiments.p2_paired_entry import verify_inputs
    from experiments.learning_run_restore import identity_digest

    root = Path(staging).resolve()
    if root.is_relative_to(Path('/content/drive').resolve()) or root.exists():
        raise ValueError('Need a fresh local staging directory')
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != observation['gpu_name']:
        raise ValueError('GPU differs from live observation')
    if not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 autocast is not supported')
    source = ast.parse((Path(__file__).parent / 'prepare_data.py').read_text(encoding='utf-8'))
    weights = [ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign)
               and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id == 'WEIGHTS_SHA256'][0]
    if pilot.sha256(Path(model_path) / 'model.safetensors') != weights:
        raise ValueError('Pinned base digest differs')
    model_files = {name: pilot.sha256(Path(model_path) / name) for name in
                   ('config.json', 'tokenizer.json', 'tokenizer_config.json')}
    if model_files['tokenizer.json'] != proof['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from data proof')
    runtime_files = ('learning_run_entry.py', 'learning_run_restore.py',
                     'drive_checkpoint_store.py', 'api_checkpoint_training.py',
                     'managed_run.py', 'p2_paired_entry.py', 'prepare_data.py')
    source_hashes = {name: pilot.sha256(Path(__file__).parent / name) for name in runtime_files}
    quota = store.call(lambda: store.service.about().get(fields='storageQuota').execute(num_retries=2))['storageQuota']
    if quota.get('limit') and int(quota['limit']) - int(quota.get('usage', 0)) < 50 * 2**30:
        raise OSError('Insufficient Drive space for immutable checkpoints and benchmark')
    root.mkdir(parents=True)
    inputs = verify_inputs(proof, store, root / 'inputs')
    train_path, train = inputs['training']
    dev_path, development = inputs['development']
    def create_model(method):
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                    torch_dtype=torch.float32, attn_implementation='sdpa')
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model = (pilot.ConditionalModel(base, 'ar') if method == 'ar'
                 else paired.MaskedPaddingCanvasDLM(base))
        return model.to('cuda')
    def optimizer_for(model):
        return torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(.9, .95), weight_decay=.01)
    def discard(model, optimizer):
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    cached = verified_cached_storage(cached_storage, proof, Path(__file__).parent)
    measurements = {}
    for method in ('ar', 'dlm'):
        start_load = time.monotonic()
        model = create_model(method)
        optimizer = optimizer_for(model)
        load_seconds = time.monotonic() - start_load
        order, masks = paired.fresh_generators(11)
        pilot.synchronize(next(model.parameters()).device)
        start = time.monotonic()
        if method == 'ar':
            pilot.update(model, optimizer, train, tokenizer.eos_token_id, order, masks, 2, 8, 1e-6)
        else:
            paired.update(model, optimizer, train, tokenizer.eos_token_id, order, masks, 1)
        pilot.synchronize(next(model.parameters()).device)
        update_seconds = time.monotonic() - start
        # Full 100-example loss evaluation and one worst-case generation.
        start = time.monotonic()
        evaluation.evaluate_dev(model, development, tokenizer.eos_token_id, deadline)
        dev_seconds = time.monotonic() - start
        start = time.monotonic()
        generation = evaluation.generate(model, tokenizer, development[0], deadline, nfe=128,
                                         mode='sample', seed=101)
        gen_seconds = time.monotonic() - start
        if method == 'ar':
            gen_seconds *= 128 / max(1, generation['forwards'])
        measurements[method] = {'update_seconds': update_seconds, 'dev_seconds': dev_seconds,
                                'gen128_seconds': gen_seconds, **cached[method],
                                'model_load_seconds': load_seconds,
                                'storage_timing_source': '2026-11-06-full-state-drive-api',
                                'storage_report_sha256': CACHED_REPORT_SHA256[method]}
        discard(model, optimizer)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        publish_report('benchmark-' + method, measurements[method])
    # Four evaluations per method; max generation count per event bounds final DLM.
    update = max(x['update_seconds'] for x in measurements.values())
    ar, dlm = measurements['ar'], measurements['dlm']
    # DLM: 3*8*32 plus 8*(3*16 + 3*32 + 3*128) forward passes.
    # Count two additional AR generations and four DLM warmups conservatively.
    eval_total = (4*(ar['dev_seconds'] + dlm['dev_seconds'])
                  + 18*ar['gen128_seconds'] + 43*dlm['gen128_seconds']
                  + 2*max(ar['model_load_seconds'], dlm['model_load_seconds']))
    eval_seconds = eval_total / 8
    save = max(x['save_seconds'] for x in measurements.values())
    restore = max(x['download_seconds'] + 3*x['cpu_load_seconds'] + 3*x['model_load_seconds']
                  for x in measurements.values())
    reserve = max(120., save + restore)
    fit = preflight_fit(deadline=deadline, now=time.monotonic(), update_seconds=update,
                        evaluation_seconds=eval_seconds, checkpoint_seconds=save,
                        restore_seconds=restore, reserve_seconds=reserve)
    publish_report('preflight', {'measurements': measurements, 'fit': fit})
    results = []
    for method in ('ar', 'dlm'):
        identity = training.identity(method, train_path, dev_path, weights, model_files)
        identity['runtime_sources_sha256'] = source_hashes
        group = root / method
        group.mkdir()
        model = create_model(method)
        optimizer = optimizer_for(model)
        writer = ApiCheckpointWriter(pilot.save_checkpoint, store, folder_id, group, method)
        checkpoint = checkpoint_callback(writer=writer, group=group, method=method,
                                         deadline=deadline, reserve=reserve,
                                         publish_report=publish_report)
        def evaluate(model, rows, tok, step, work_deadline):
            metrics = evaluation.evaluate_dev(model, rows, tok.eos_token_id, work_deadline)
            samples = []
            warmup_seconds = None
            if method == 'dlm' or step in (0, 512):
                start = time.monotonic()
                evaluation.generate(model, tok, rows[0], work_deadline, nfe=16, mode='greedy', seed=None)
                warmup_seconds = time.monotonic() - start
                samples = evaluation.evaluate_generations(model, tok, rows, step, work_deadline)
            publish_report(method + '-evaluation-' + str(step),
                           {'step': step, 'metrics': metrics, 'warmup_seconds': warmup_seconds,
                            'samples': samples})
        history = training.run_training(model, optimizer, train, development, tokenizer, identity,
                                        deadline=deadline, reserve_seconds=reserve,
                                        checkpoint=checkpoint, evaluate=evaluate)
        receipt = writer.last_server_receipt
        if not receipt or receipt['step'] != 512:
            raise RuntimeError('Final checkpoint missing')
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        verification = group / 'verification'
        verification.mkdir()
        verified = store.verify_snapshot(receipt, folder_id, identity_sha256=identity_digest(identity),
            verification_dir=verification, restore_check=lambda path: independent_restore(path, identity, dev_path))
        publish_report(method + '-complete', {'step': len(history), 'manifest': verified['manifest']})
        (verification / receipt['server']['name']).unlink()
        (group / 'latest.pt').unlink()
        results.append({'method': method, 'step': len(history), 'manifest_id': verified['manifest_id']})
    return {'status': 'completed', 'methods': results, 'measurements': measurements, 'fit': fit}


def main():
    import argparse
    import io
    import json
    import os
    import subprocess
    import sys
    import uuid
    from pathlib import Path
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    from googleapiclient.http import MediaIoBaseUpload
    from transformers import AutoTokenizer
    from experiments.drive_checkpoint_store import DriveCheckpointStore, build_drive_service
    parser = argparse.ArgumentParser()
    for name in ('observation', 'proof', 'model-path', 'staging', 'folder-id', 'deadline', 'cpu-receipt', 'storage-cache'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    deadline = float(args.deadline)
    observation = json.loads(Path(args.observation).read_text(encoding='utf-8'))
    proof = json.loads(Path(args.proof).read_text(encoding='utf-8'))
    cached_storage = json.loads(Path(args.storage_cache).read_text(encoding='utf-8'))
    validate_lease(observation, deadline)
    credentials, _ = google.auth.default()
    service = build_drive_service(credentials)
    store = DriveCheckpointStore(service, deadline=deadline, stream_session=AuthorizedSession(credentials))
    receipt_meta = store.metadata(args.cpu_receipt)
    receipt_bytes = store.call(lambda: service.files().get_media(fileId=args.cpu_receipt).execute(num_retries=2))
    if args.folder_id not in receipt_meta.get('parents', []) or not receipt_bytes.startswith(b'learning-auth-roundtrip-'):
        raise ValueError('CPU Drive roundtrip receipt differs')
    folder = store.call(lambda: service.files().create(body={'name': 'learning-512-' + uuid.uuid4().hex,
        'mimeType':'application/vnd.google-apps.folder', 'parents':[args.folder_id]}, fields='id').execute(num_retries=2))
    def publish_report(name, report):
        payload = json.dumps(report, ensure_ascii=False, allow_nan=False).encode()
        created = store.call(lambda: service.files().create(body={'name': name + '-' + uuid.uuid4().hex + '.json',
            'parents':[folder['id']]}, media_body=MediaIoBaseUpload(io.BytesIO(payload), mimetype='application/json'),
            fields='id').execute(num_retries=2))
        readback = store.call(lambda: service.files().get_media(fileId=created['id']).execute(num_retries=2))
        if readback != payload:
            raise ValueError('Report readback differs')
        print('REPORT_SAVED', name, created['id'], flush=True)
    def independent_restore(path, identity, dev_path):
        identity_path = path.parent / 'identity.json'
        identity_path.write_text(json.dumps(identity), encoding='utf-8')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('No independent restore time remains')
        proc = subprocess.run([sys.executable, '-m', 'experiments.learning_run_restore', '--checkpoint', str(path),
            '--identity', str(identity_path), '--model-dir', args.model_path, '--development', str(dev_path)],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES':''}, capture_output=True, text=True, timeout=remaining)
        if proc.returncode:
            safe_reasons = ('Invalid learning-run identity', 'Checkpoint identity or milestone differs',
                            'Incomplete history or optimizer state', 'Invalid CUDA RNG structure',
                            'Incomplete Adam state', 'Adam step differs', 'Invalid Adam moment',
                            'Nonfinite CPU inference', 'Model configuration/tokenizer differs',
                            'Runtime source differs', 'CPU-only restore required')
            final_line = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''
            reason = next((item for item in safe_reasons if final_line.endswith(item)),
                          'Unclassified restore failure')
            keys = ('model_sha256', 'development_sha256', 'training_source_sha256',
                    'evaluation_source_sha256', 'core_sha256', 'canvas_sha256', 'paired_sha256')
            for key in keys:
                if final_line.endswith('Identity digest differs: ' + key):
                    reason = 'Identity digest differs: ' + key
                    break
            reason += ' (exit ' + str(proc.returncode) + ')'
            print('CPU_RESTORE_FAILED', reason, flush=True)
            raise RuntimeError('Independent CPU restore failed: ' + reason)
        return json.loads(proc.stdout.splitlines()[-1])
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        result = run_pair(observation=observation, deadline=deadline, store=store, folder_id=folder['id'],
                          staging=args.staging, proof=proof, model_path=args.model_path, tokenizer=tokenizer,
                          independent_restore=independent_restore, publish_report=publish_report,
                          cached_storage=cached_storage)
        publish_report('summary', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        try:
            publish_report('failure', {'status':'failed', 'error_type':type(exc).__name__,
                                       'reason': (str(exc) if isinstance(exc, RuntimeError) and
                                                  str(exc).startswith(('Fixed 512-step pair requires',
                                                                       'Independent CPU restore failed:'))
                                                  else 'See execution output for the failure phase')})
        except Exception:
            print('FAILURE_REPORT_NOT_SAVED', flush=True)
        raise


if __name__ == '__main__':
    main()
