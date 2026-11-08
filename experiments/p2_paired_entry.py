"""P2 paired run from fresh, externally observed resource and prepared data.

Importing this module never acquires a GPU or starts a run. The caller owns the
independent runtime-release watchdog, including during preparation and failures.
"""
import json
import ast
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def resource_gate(observation, *, now=None, max_age_seconds=120):
    now = now or datetime.now(timezone.utc)
    observed = datetime.fromisoformat(observation['observed_utc'].replace('Z', '+00:00'))
    age = (now - observed).total_seconds()
    if not 0 <= age <= max_age_seconds:
        raise ValueError('Resource observation must be fresh and timestamped')
    for name in ('start_balance', 'current_balance', 'cu_per_hour', 'reserve_cu',
                 'overhead_seconds'):
        if not math.isfinite(observation[name]) or observation[name] <= 0:
            raise ValueError('Missing positive resource observation: ' + name)
    if observation['current_balance'] > observation['start_balance']:
        raise ValueError('Current balance exceeds recorded stage start')
    if (not observation.get('gpu_name') or not observation.get('session_id')
            or observation.get('other_active_sessions') != 0):
        raise ValueError('Need observed GPU, session identity and zero other active sessions')
    return observation


def lock_steps(observation, seconds_per_step):
    from experiments.p2_paired_training import check_budget
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise ValueError('A positive measured update time is required')
    for steps in range(100, 0, -1):
        try:
            check_budget(observation['start_balance'], observation['current_balance'],
                         observation['cu_per_hour'], observation['reserve_cu'],
                         observation['overhead_seconds'] + 2 * steps * seconds_per_step)
            return steps
        except RuntimeError:
            continue
    raise RuntimeError('Insufficient budget for paired updates and reserve')


def verify_inputs(proof, store, destination):
    """Fetch only approved training and development artifacts by immutable digest."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    result = {}
    for split, count in [('training', 4096), ('development', 100)]:
        entry = proof['artifact_files'][split]
        expected = entry['fingerprints']
        if (proof['counts'][split] != count
                or proof['artifacts'][split]['sha256'] != expected['sha256']):
            raise ValueError('Proof differs for ' + split)
        path = destination / (split + '.json')
        store.download_verified(entry['server']['id'], path, expected=expected)
        rows = json.loads(path.read_text(encoding='utf-8'))
        if len(rows) != count:
            raise ValueError('Downloaded row count differs for ' + split)
        result[split] = (path, rows)
    return result


def budget_deadline(observation, started):
    spent = observation['start_balance'] - observation['current_balance']
    remaining = min(4 - spent, observation['current_balance'])
    if remaining <= observation['reserve_cu']:
        raise ValueError('No budget remains after reserve')
    return started + remaining * 3600 / observation['cu_per_hour']


def run_pair(*, observation, store, folder_id, staging, proof, model_path,
             tokenizer, independent_restore, publish_report):
    """Run the pair; injected callbacks must use fresh local models and a new process.

    ``independent_restore(path, identity)`` must start a distinct process and
    return the restore receipt. The store independently downloads each snapshot.
    The surrounding Colab cell must start its independent release watchdog before
    any connection/preparation and release in a finally block.
    """
    import gc
    import torch
    from transformers import AutoModelForCausalLM
    from experiments import p2_paired_training as p2
    from experiments import full_parameter_pilot as pilot
    from experiments.api_checkpoint_training import ApiCheckpointWriter
    from experiments.p2_restore_check import identity_digest
    source = ast.parse((Path(__file__).parent / 'prepare_data.py').read_text(encoding='utf-8'))
    hashes = [ast.literal_eval(node.value) for node in source.body
              if isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'WEIGHTS_SHA256']
    if len(hashes) != 1:
        raise ValueError('Expected one frozen base weight digest')
    WEIGHTS_SHA256 = hashes[0]

    resource_gate(observation, max_age_seconds=3600)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != observation['gpu_name']:
        raise ValueError('Actual GPU differs from resource observation')
    observed = datetime.fromisoformat(observation['observed_utc'].replace('Z', '+00:00'))
    observed_age = (datetime.now(timezone.utc) - observed).total_seconds()
    started = time.monotonic() - observed_age
    deadline = budget_deadline(observation, started)
    reserve_seconds = observation['reserve_cu'] * 3600 / observation['cu_per_hour']
    root = Path(staging).resolve()
    if root.is_relative_to(Path('/content/drive').resolve()):
        raise ValueError('P2 staging must be local, not DriveFS')
    if pilot.sha256(Path(model_path) / 'model.safetensors') != WEIGHTS_SHA256:
        raise ValueError('Base weights differ from the fixed P2 base')
    model_files = {name: pilot.sha256(Path(model_path) / name) for name in
                   ('config.json', 'tokenizer.json', 'tokenizer_config.json')}
    if model_files['tokenizer.json'] != proof['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from prepared P2 data')
    source_files = {name: pilot.sha256(Path(__file__).parent / name) for name in
                    ('p2_paired_training.py', 'p2_canvas_ablation.py', 'full_parameter_pilot.py',
                     'p2_paired_entry.py', 'p2_restore_check.py', 'prepare_data.py',
                     'drive_checkpoint_store.py', 'api_checkpoint_training.py', 'managed_run.py')}
    quota = store.call(lambda: store.service.about().get(fields='storageQuota').execute(num_retries=2))['storageQuota']
    if quota.get('limit') and int(quota['limit']) - int(quota.get('usage', 0)) < 16 * 2**30:
        raise OSError('Insufficient Drive space for two immutable checkpoints')
    root.mkdir(parents=True, exist_ok=False)
    inputs = verify_inputs(proof, store, root / 'inputs')
    train_path, train = inputs['training']
    development_path, development = inputs['development']
    def optimizer_for(model):
        return torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(.9, .95), weight_decay=.01)
    def create_model(policy):
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                    torch_dtype=torch.float32,
                                                    attn_implementation='sdpa')
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        return p2.make_model(base, policy).to('cuda')
    # Benchmark has independent weights, optimizer and RNG, discarded before lock.
    timings = []
    for policy in p2.POLICIES:
        model = create_model(policy)
        optimizer = optimizer_for(model)
        order, masks = p2.fresh_generators()
        device = next(model.parameters()).device
        pilot.synchronize(device)
        measured = time.monotonic()
        p2.update(model, optimizer, train, tokenizer.eos_token_id, order, masks, 1)
        pilot.synchronize(device)
        timings.append(time.monotonic() - measured)
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    seconds = max(timings)
    # Account for all preparation and benchmark time since the observed balance.
    elapsed = time.monotonic() - started
    adjusted = dict(observation)
    adjusted['current_balance'] -= elapsed * observation['cu_per_hour'] / 3600
    steps = lock_steps(adjusted, seconds * 1.25)
    if time.monotonic() + observation['overhead_seconds'] + 2 * steps * seconds * 1.25 >= deadline - reserve_seconds:
        raise RuntimeError('Measured paired work exceeds deadline after preparation')
    results = []
    for policy in p2.POLICIES:
        identity = p2.identity(policy, train_path, development_path, WEIGHTS_SHA256,
                               steps=steps, model_files=model_files)
        identity['source_files_sha256'] = source_files
        model = create_model(policy)
        optimizer = optimizer_for(model)
        group = root / policy
        group.mkdir()
        writer = ApiCheckpointWriter(pilot.save_checkpoint, store, folder_id, group, 'dlm')
        def checkpoint(model, optimizer, step, order, masks, identity, history):
            writer(group / 'latest.pt', model, optimizer, step, order, masks, identity, history)
        def should_continue(phase):
            return time.monotonic() < deadline - reserve_seconds
        report = p2.run_policy(model, optimizer, train, development, tokenizer,
                               identity, checkpoint, should_continue)
        receipt = writer.last_server_receipt
        if receipt is None:
            raise RuntimeError('No verified server checkpoint')
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        verify_dir = group / 'verification'
        verify_dir.mkdir()
        verified = store.verify_snapshot(receipt, folder_id,
                                         identity_sha256=identity_digest(identity),
                                         verification_dir=verify_dir,
                                         restore_check=lambda path: independent_restore(path, identity))
        report['checkpoint_manifest'] = verified['manifest']
        publish_report(policy, report)
        results.append({'policy': policy, 'manifest_id': verified['manifest_id']})
    return {'status': 'completed', 'steps': steps, 'benchmark_seconds': seconds,
            'policies': results}


def main():
    import argparse
    import io
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    from googleapiclient.http import MediaIoBaseUpload
    from transformers import AutoTokenizer
    from experiments.drive_checkpoint_store import DriveCheckpointStore, build_drive_service
    parser = argparse.ArgumentParser()
    for name in ('observation', 'proof', 'model-path', 'staging', 'folder-id', 'deadline'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    observation = json.loads(Path(args.observation).read_text(encoding='utf-8'))
    proof = json.loads(Path(args.proof).read_text(encoding='utf-8'))
    credentials, _ = google.auth.default()
    service = build_drive_service(credentials)
    store = DriveCheckpointStore(service, deadline=float(args.deadline),
                                 stream_session=AuthorizedSession(credentials))
    folder = store.call(lambda: service.files().create(
        body={'name': 'p2-paired-' + __import__('uuid').uuid4().hex,
              'mimeType': 'application/vnd.google-apps.folder', 'parents': [args.folder_id]},
        fields='id').execute(num_retries=2))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    def independent_restore(path, identity):
        identity_path = path.parent / 'identity.json'
        identity_path.write_text(json.dumps(identity), encoding='utf-8')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
        remaining = float(args.deadline) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('No restore time remains')
        proc = subprocess.run([sys.executable, '-m', 'experiments.p2_restore_check',
                               '--checkpoint', str(path), '--identity', str(identity_path),
                               '--model-dir', args.model_path, '--development',
                               str(Path(args.staging) / 'inputs' / 'development.json')],
                              env=env, capture_output=True, text=True, check=True, timeout=remaining)
        return json.loads(proc.stdout.splitlines()[-1])
    def publish_report(policy, report):
        payload = json.dumps(report, ensure_ascii=False, allow_nan=False).encode()
        created = store.call(lambda: service.files().create(
            body={'name': 'p2-' + policy + '-report.json', 'parents': [folder['id']]},
            media_body=MediaIoBaseUpload(io.BytesIO(payload), mimetype='application/json'),
            fields='id').execute(num_retries=2))
        actual = store.call(lambda: service.files().get_media(fileId=created['id']).execute(num_retries=2))
        if actual != payload:
            raise ValueError('Report readback differs')
        print('REPORT_ID', policy, created['id'], flush=True)
    result = run_pair(observation=observation, store=store, folder_id=folder['id'],
                      staging=args.staging, proof=proof, model_path=args.model_path,
                      tokenizer=tokenizer, independent_restore=independent_restore,
                      publish_report=publish_report)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
