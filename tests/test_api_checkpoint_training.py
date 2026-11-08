"""Storage-hook and manifest-gate tests; no model training or network access."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import experiments
from experiments.api_checkpoint_training import ApiCheckpointWriter, resume_training
from experiments.drive_checkpoint_store import TransferError, fingerprints


class BytesMedia:
    def __init__(self, stream, **kwargs):
        self.payload = stream.read()


class FakeStore:
    def __init__(self, input_bytes=b'input checkpoint 40'):
        self.input_bytes = input_bytes
        self.events, self.receipts, self.snapshots = [], {}, []
        self.fail_upload = False
        self.version = '1'
        self.service = self
        self.deadline, self.now, self.download_seconds = 1000, 0, 0

    def clock(self):
        return self.now

    def check_time(self):
        return None

    def media(self):
        return None, None, BytesMedia

    def call(self, action):
        return action()

    def files(self):
        return self

    def create(self, *, body, media_body, fields):
        key = 'receipt-' + str(len(self.receipts))
        self.receipts[key] = media_body.payload
        self.events.append('receipt')
        return SimpleNamespace(execute=lambda **kwargs: {'id': key})

    def get_media(self, *, fileId):
        return SimpleNamespace(execute=lambda **kwargs: self.receipts[fileId])

    def metadata(self, file_id):
        self.events.append('metadata')
        return {'id': file_id, 'version': self.version}

    def download_verified(self, file_id, destination, *, expected):
        self.events.append('download')
        self.now += self.download_seconds
        with Path(destination).open('xb') as stream:
            stream.write(self.input_bytes)

    def upload_snapshot(self, path, folder_id, *, method, step):
        if self.fail_upload:
            raise TransferError('Injected upload failure')
        self.events.append('upload')
        self.snapshots.append((Path(path), Path(path).read_bytes()))
        return {'status': 'server_verified', 'server': {'id': 'snapshot-' + str(step),
                'name': f'{method}-{step}.pt'}, 'fingerprints': fingerprints(path),
                'method': method, 'step': step, 'restore_verified': False}


def save_local(path, model, optimizer, step, order, masks, identity, history):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(('saved step ' + str(step)).encode())
    os.replace(temporary, path)


def test_failed_upload_does_not_advance_last_server_receipt(tmp_path):
    store = FakeStore()
    writer = ApiCheckpointWriter(save_local, store, 'folder', tmp_path, 'ar')
    path, identity = tmp_path / 'latest.pt', {'method': 'ar'}
    writer(path, None, None, 60, None, None, identity, [])
    previous = copy.deepcopy(writer.last_server_receipt)
    saved_receipt = (tmp_path / 'last-server-verified.json').read_bytes()
    assert previous['ready_for_resume'] is False
    assert previous['requires_independent_restore_audit'] is True
    assert store.snapshots[0][0] != path
    assert not store.snapshots[0][0].exists()
    store.fail_upload = True
    with pytest.raises(TransferError, match='upload failure'):
        writer(path, None, None, 80, None, None, identity, [])
    assert writer.last_server_receipt == previous
    assert (tmp_path / 'last-server-verified.json').read_bytes() == saved_receipt
    assert store.snapshots[0][1] == b'saved step 60'
    assert path.read_bytes() == b'saved step 80'


def test_writer_rejects_path_escape_before_save(tmp_path):
    writer = ApiCheckpointWriter(save_local, FakeStore(), 'folder', tmp_path / 'local', 'ar')
    outside = tmp_path / 'outside.pt'
    with pytest.raises(ValueError, match='destination or method differs'):
        writer(outside, None, None, 60, None, None, {'method': 'ar'}, [])
    assert not outside.exists()


@pytest.fixture
def entry(tmp_path, monkeypatch):
    kernel = tmp_path / 'kernel.py'
    kernel.write_text('# unchanged training fixture\n', encoding='utf-8')
    identity = {'method': 'ar', 'script_sha256': hashlib.sha256(kernel.read_bytes()).hexdigest()}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                             separators=(',', ':')).encode()).hexdigest()
    original = tmp_path / 'original.pt'
    original.write_bytes(b'input checkpoint 40')
    manifest = {'status': 'verified', 'fresh_download_verified': True, 'method': 'ar',
                'step': 40, 'identity_sha256': identity_hash, 'checkpoint_id': 'original',
                'checkpoint_version': '1', **fingerprints(original),
                'restore': {'restored': True, 'step': 40, 'method': 'ar',
                            'identity_sha256': identity_hash, 'rng_state_restored': True,
                            'inference_finite': True}}
    path = tmp_path / 'verified.json'
    store = FakeStore()
    training = ModuleType('experiments.full_parameter_pilot')
    training.__file__ = str(kernel)
    training.save_checkpoint = save_local
    training.fail_after_save = False

    def main():
        assert isinstance(training.save_checkpoint, ApiCheckpointWriter)
        args = sys.argv
        training.assigned_max_seconds = int(args[args.index('--max-seconds') + 1])
        root = Path(args[args.index('--output') + 1])
        training.save_checkpoint(root / 'latest.pt', None, None, 60,
                                 None, None, identity, [])
        if training.fail_after_save:
            raise RuntimeError('Injected training interruption')

    training.main = main
    monkeypatch.setitem(sys.modules, 'experiments.full_parameter_pilot', training)
    monkeypatch.setattr(experiments, 'full_parameter_pilot', training, raising=False)

    def run(current=None, digest_override=None, argv=None, reserve=300):
        path.write_text(json.dumps(manifest if current is None else current), encoding='utf-8')
        expected_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return resume_training(['--steps', '60'] if argv is None else argv,
                               store, 'folder', tmp_path / 'local', path,
                               digest_override or expected_digest, identity,
                               publication_reserve_seconds=reserve)

    return SimpleNamespace(run=run, manifest=manifest, path=path, store=store,
                           training=training, root=tmp_path / 'local', original=original)


@pytest.mark.parametrize('fail', [False, True])
def test_entry_restores_training_globals_and_sys_argv(entry, fail):
    old_save, old_argv = entry.training.save_checkpoint, sys.argv
    entry.training.fail_after_save = fail
    if fail:
        with pytest.raises(RuntimeError, match='training interruption'):
            entry.run()
    else:
        result = entry.run()
        assert result['step'] == 60 and result['ready_for_resume'] is False
    assert entry.training.save_checkpoint is old_save
    assert sys.argv is old_argv
    assert entry.original.read_bytes() == b'input checkpoint 40'


@pytest.mark.parametrize('fault', ['status', 'fresh', 'restored', 'rng', 'finite',
                                  'restore-step', 'restore-method', 'restore-identity'])
def test_entry_rejects_semantically_invalid_manifest_before_download(entry, fault):
    manifest = copy.deepcopy(entry.manifest)
    if fault == 'status':
        manifest['status'] = 'server_verified'
    elif fault == 'fresh':
        manifest['fresh_download_verified'] = False
    elif fault in ('restored', 'rng', 'finite'):
        key = {'restored': 'restored', 'rng': 'rng_state_restored', 'finite': 'inference_finite'}[fault]
        manifest['restore'][key] = False
    elif fault == 'restore-step':
        manifest['restore']['step'] = 41
    elif fault == 'restore-method':
        manifest['restore']['method'] = 'dlm'
    else:
        manifest['restore']['identity_sha256'] = 'different'
    with pytest.raises(ValueError, match='restore-verified'):
        entry.run(current=manifest)
    assert not entry.store.events
    assert not entry.root.exists()


def test_entry_rejects_wrong_manifest_digest(entry):
    with pytest.raises(ValueError, match='manifest digest differs'):
        entry.run(digest_override='wrong')
    assert not entry.store.events


def test_entry_never_overwrites_existing_staging(entry):
    entry.root.mkdir()
    latest = entry.root / 'latest.pt'
    latest.write_bytes(b'keep existing local checkpoint')
    with pytest.raises(FileExistsError):
        entry.run()
    assert latest.read_bytes() == b'keep existing local checkpoint'
    assert not entry.store.events


def test_entry_rejects_changed_server_version(entry):
    entry.store.version = '2'
    with pytest.raises(ValueError, match='version changed'):
        entry.run()
    assert entry.store.events == ['metadata']


def test_failed_entry_upload_restores_globals_and_keeps_manifest(entry):
    old_save, old_argv = entry.training.save_checkpoint, sys.argv
    entry.store.fail_upload = True
    with pytest.raises(TransferError, match='upload failure'):
        entry.run()
    assert entry.training.save_checkpoint is old_save
    assert sys.argv is old_argv
    assert json.loads(entry.path.read_text()) == entry.manifest
    assert not (entry.root / 'last-server-verified.json').exists()


def test_publication_time_is_reserved_after_download(entry):
    entry.store.download_seconds = 250
    entry.run()
    assert entry.training.assigned_max_seconds == 450


def test_insufficient_time_after_download_refuses_training(entry):
    old_save, old_argv = entry.training.save_checkpoint, sys.argv
    entry.store.download_seconds = 550
    with pytest.raises(TimeoutError, match='publication reserve'):
        entry.run()
    assert entry.training.save_checkpoint is old_save
    assert sys.argv is old_argv
    assert 'upload' not in entry.store.events


@pytest.mark.parametrize('argv', [
    ['--output=outside', '--steps', '60'],
    ['--method', 'dlm', '--steps', '60'],
    ['--resume', '--steps', '60'],
    ['--max-seconds=9999', '--steps', '60'],
    ['--steps', '40'],
    ['--steps', '60', '--steps=80'],
    [],
])
def test_reserved_arguments_or_invalid_targets_refuse_before_download(entry, argv):
    with pytest.raises(ValueError):
        entry.run(argv=argv)
    assert not entry.store.events
    assert not entry.root.exists()


@pytest.mark.parametrize('reserve', [0, -1, float('nan'), float('inf')])
def test_invalid_publication_reserve_refuses_before_download(entry, reserve):
    with pytest.raises(ValueError, match='publication time reserve'):
        entry.run(reserve=reserve)
    assert not entry.store.events
