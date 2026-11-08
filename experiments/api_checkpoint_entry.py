"""Bounded GPU continuation using verified API snapshots; no DriveFS writes."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from experiments.api_checkpoint_training import resume_training
from experiments.checkpoint_cadence import CheckpointCadence
from experiments.drive_checkpoint_store import DriveCheckpointStore, build_drive_service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-id', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--parent-folder', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--publication-reserve-seconds', type=float, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--eval-examples', type=int, choices=[32], default=32)
    parser.add_argument('--cloud-checkpoint-every', type=int, choices=[20, 100], default=100)
    args = parser.parse_args()
    if not math.isfinite(args.deadline) or args.deadline <= time.monotonic():
        raise ValueError('No execution budget remains')
    if (not math.isfinite(args.publication_reserve_seconds)
            or args.publication_reserve_seconds <= 0
            or args.deadline - time.monotonic() - args.publication_reserve_seconds <= 180):
        raise ValueError('Insufficient bounded time after publication reserve')
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    from googleapiclient.http import MediaIoBaseUpload

    credentials, _ = google.auth.default()
    service = build_drive_service(credentials)
    last = {'phase': None, 'bytes': -1}

    def progress(event):
        if (event['phase'] != last['phase'] or event['bytes'] - last['bytes'] >= 256 * 2**20
                or event['bytes'] == event['total']):
            print('TRANSFER', json.dumps(event), flush=True)
            last.update(phase=event['phase'], bytes=event['bytes'])

    store = DriveCheckpointStore(service, deadline=args.deadline, progress=progress,
                                 stream_session=AuthorizedSession(credentials))
    root = Path('/content') / ('api-continuation-' + uuid.uuid4().hex[:8])
    root.mkdir()
    metadata = store.metadata(args.manifest_id)
    if int(metadata['size']) > 1 << 20:
        raise ValueError('Unexpectedly large checkpoint manifest')
    manifest_bytes = store.call(lambda: service.files().get_media(
        fileId=args.manifest_id).execute(num_retries=2))
    if hashlib.sha256(manifest_bytes).hexdigest() != args.manifest_sha256:
        raise ValueError('Input manifest checksum differs')
    manifest = json.loads(manifest_bytes)
    if manifest.get('status') != 'verified' or args.steps <= manifest['step']:
        raise ValueError('Need a verified input and a later target step')
    identity = json.loads(args.identity.read_text(encoding='utf-8'))
    if identity.get('method') not in ('ar', 'dlm'):
        raise ValueError('Invalid training method')
    # Keep all old snapshots. Refuse upfront when the account cannot hold this run.
    slots = args.steps // args.cloud_checkpoint_every - manifest['step'] // args.cloud_checkpoint_every + 1
    quota = store.call(lambda: service.about().get(fields='storageQuota').execute(num_retries=2))['storageQuota']
    if quota.get('limit') and int(quota['limit']) - int(quota.get('usage', 0)) < slots * manifest['bytes'] + (1 << 20):
        raise OSError('Insufficient Drive quota for immutable snapshots; do not train')
    manifest_path = root / 'input-manifest.json'
    manifest_path.write_bytes(manifest_bytes)
    source_root = Path(__file__).resolve().parents[1]
    for module in ('experiments.prepare_data', 'experiments.prepare_pipeline_pilot'):
        store.check_time()
        with (root / (module.rsplit('.', 1)[1] + '.log')).open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', module], cwd=source_root,
                           env={**os.environ, 'USE_TF': '0', 'USE_FLAX': '0', 'HF_HUB_DISABLE_XET': '1'},
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=max(1, min(300, args.deadline - time.monotonic())))
    folder = store.call(lambda: service.files().create(
        body={'name': identity['method'] + '-api-continuation-' + uuid.uuid4().hex[:8],
              'mimeType': 'application/vnd.google-apps.folder', 'parents': [args.parent_folder]},
        fields='id,name').execute(num_retries=2))
    print('CONTINUATION_FOLDER', json.dumps(folder), flush=True)
    staging = root / identity['method']
    receipt = resume_training(['--steps', str(args.steps), '--eval-examples', str(args.eval_examples)],
                              store, folder['id'], staging, manifest_path, args.manifest_sha256,
                              identity, publication_reserve_seconds=args.publication_reserve_seconds,
                              writer_wrapper=lambda writer, step: CheckpointCadence(
                                  writer, step, args.cloud_checkpoint_every))
    reports = list(staging.glob(f'report-{manifest["step"]}-*.json'))
    if len(reports) != 1:
        raise ValueError('Expected one continuation report')
    report_bytes = reports[0].read_bytes()
    report = json.loads(report_bytes)
    if (report['identity'] != identity or report['end_step'] != receipt['step']
            or report['checkpoint_sha256'] != receipt['fingerprints']['sha256']):
        raise ValueError('Report and server-verified checkpoint differ')

    def publish_json(name, payload):
        result = store.call(lambda: service.files().create(
            body={'name': name, 'parents': [folder['id']]},
            media_body=MediaIoBaseUpload(io.BytesIO(payload), mimetype='application/json'),
            fields='id,size,sha256Checksum').execute(num_retries=2))
        if store.call(lambda: service.files().get_media(fileId=result['id']).execute(num_retries=2)) != payload:
            raise ValueError('Report upload readback differs')
        return result

    report_file = publish_json(reports[0].name, report_bytes)
    receipt.update(training_status=report['status'], report_file=report_file,
                   input_manifest_id=args.manifest_id, input_manifest_sha256=args.manifest_sha256,
                   cloud_checkpoint_every=args.cloud_checkpoint_every,
                   cadence_source_sha256=hashlib.sha256(
                       (Path(__file__).parent / 'checkpoint_cadence.py').read_bytes()).hexdigest())
    receipt_file = publish_json('continuation-receipt.json',
                                (json.dumps(receipt, indent=2) + '\n').encode())
    print('CONTINUATION_RECEIPT', json.dumps(receipt_file), flush=True)
    print('TRAINING_STATUS', report['status'], 'END_STEP', report['end_step'],
          'INDEPENDENT_RESTORE_AUDIT_REQUIRED', flush=True)


if __name__ == '__main__':
    main()
