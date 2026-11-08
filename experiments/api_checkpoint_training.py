"""Storage-only adapter for the unchanged training kernel.

GPU execution is not started by importing this module. New snapshots are uploaded
through Drive API; server acknowledgement and independent restore are distinct.
"""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import uuid

from experiments.drive_checkpoint_store import fingerprints, TransferError
from experiments.managed_run import atomic_json


class ApiCheckpointWriter:
    def __init__(self, original_save, store, folder_id, local_root, method):
        self.original_save, self.store, self.folder_id = original_save, store, folder_id
        self.root = Path(local_root).resolve()
        if self.root.is_relative_to(Path('/content/drive').resolve()):
            raise ValueError('Training checkpoint staging must be local, not DriveFS')
        self.method = method
        self.last_server_receipt = None

    def __call__(self, path, model, optimizer, step, order, masks, identity, history):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root) or identity['method'] != self.method:
            raise ValueError('Checkpoint destination or method differs')
        self.store.check_time()
        self.original_save(path, model, optimizer, step, order, masks, identity, history)
        expected = fingerprints(path)
        previous = self.last_server_receipt
        if (previous and previous['step'] == step
                and previous['fingerprints'] == expected):
            return
        snapshots = self.root / 'immutable-local-snapshots'
        snapshots.mkdir(exist_ok=True)
        snapshot = snapshots / f'{self.method}-{step}-{uuid.uuid4().hex}.pt'
        os.link(path, snapshot)
        uploaded = self.store.upload_snapshot(snapshot, self.folder_id,
                                              method=self.method, step=step)
        if uploaded['fingerprints'] != expected or uploaded['status'] != 'server_verified':
            raise TransferError('Upload acknowledgement differs from saved checkpoint')
        receipt = {**uploaded, 'identity': identity,
                   'storage_adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'ready_for_resume': False, 'requires_independent_restore_audit': True}
        payload = (json.dumps(receipt, sort_keys=True, indent=2) + '\n').encode()
        _, _, bytes_upload = self.store.media()
        result = self.store.call(lambda: self.store.service.files().create(
            body={'name': uploaded['server']['name'] + '.server-verified.json',
                  'parents': [self.folder_id]},
            media_body=bytes_upload(io.BytesIO(payload), mimetype='application/json'),
            fields='id').execute(num_retries=2))
        readback = self.store.call(lambda: self.store.service.files().get_media(
            fileId=result['id']).execute(num_retries=2))
        if readback != payload:
            raise TransferError('Server receipt readback differs')
        receipt['receipt_id'] = result['id']
        atomic_json(self.root / 'last-server-verified.json', receipt)
        self.last_server_receipt = receipt
        # Only this invocation's local hard link is removed; original/cloud files stay.
        snapshot.unlink()
        print('CHECKPOINT_SERVER_VERIFIED', self.method, step, uploaded['server']['id'], flush=True)


def resume_training(argv, store, folder_id, local_root, manifest_path, manifest_sha256,
                    expected_identity, *, publication_reserve_seconds, writer_wrapper=None):
    """Prepare verified input, then run original math with the API storage adapter."""
    from experiments import full_parameter_pilot as training

    payload = Path(manifest_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest_sha256:
        raise ValueError('Verified manifest digest differs')
    manifest = json.loads(payload)
    identity_sha = hashlib.sha256(json.dumps(expected_identity, sort_keys=True,
                                             separators=(',', ':')).encode()).hexdigest()
    restored = manifest.get('restore', {})
    if (manifest.get('status') != 'verified' or manifest.get('fresh_download_verified') is not True
            or type(manifest.get('step')) is not int or manifest['step'] < 0
            or manifest.get('identity_sha256') != identity_sha
            or manifest.get('method') != expected_identity['method']
            or restored.get('restored') is not True
            or restored.get('rng_state_restored') is not True
            or restored.get('inference_finite') is not True
            or restored.get('step') != manifest.get('step')
            or restored.get('method') != expected_identity['method']
            or restored.get('identity_sha256') != identity_sha):
        raise ValueError('Input checkpoint is not restore-verified')
    if hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest() != expected_identity['script_sha256']:
        raise ValueError('Training kernel differs from checkpoint identity')
    if any(arg.split('=', 1)[0] in ('--output', '--method', '--resume', '--max-seconds') for arg in argv):
        raise ValueError('Output/method/resume are selected by the verified manifest')
    targets = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == '--steps']
    targets += [arg.split('=', 1)[1] for arg in argv if arg.startswith('--steps=')]
    if len(targets) != 1 or int(targets[0]) <= manifest['step']:
        raise ValueError('Specify one target step beyond the verified checkpoint')
    if not math.isfinite(publication_reserve_seconds) or publication_reserve_seconds <= 0:
        raise ValueError('Need an explicit measured publication time reserve')
    root = Path(local_root).resolve()
    if root.is_relative_to(Path('/content/drive').resolve()):
        raise ValueError('Training checkpoint staging must be local, not DriveFS')
    if root.exists():
        raise FileExistsError('Use a fresh local staging directory')
    root.mkdir(parents=True)
    current = store.metadata(manifest['checkpoint_id'])
    if current.get('version') != manifest['checkpoint_version']:
        raise ValueError('Verified checkpoint version changed')
    store.download_verified(manifest['checkpoint_id'], root / 'latest.pt',
                            expected={k: manifest[k] for k in ('bytes', 'sha256', 'md5')})
    writer = ApiCheckpointWriter(training.save_checkpoint, store, folder_id, root,
                                 expected_identity['method'])
    if writer_wrapper is not None:
        writer = writer_wrapper(writer, manifest['step'])
    old_save, old_argv = training.save_checkpoint, sys.argv
    try:
        remaining = math.floor(store.deadline - store.clock() - publication_reserve_seconds)
        if remaining <= 180:
            raise TimeoutError('Insufficient training time after publication reserve')
        training.save_checkpoint = writer
        sys.argv = ['full_parameter_pilot', *argv, '--method', expected_identity['method'],
                    '--resume', '--output', str(root), '--max-seconds', str(remaining)]
        training.main()
    finally:
        training.save_checkpoint, sys.argv = old_save, old_argv
    if writer.last_server_receipt is None:
        raise RuntimeError('No server-verified checkpoint was produced')
    return writer.last_server_receipt
