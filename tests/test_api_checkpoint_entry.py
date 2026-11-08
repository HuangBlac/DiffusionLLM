"""Exercise CLI preconditions and evidence publication without GPUs or network."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import experiments.api_checkpoint_entry as entry


class Media:
    def __init__(self, stream, **kwargs):
        self.payload = stream.read()


class Service:
    def __init__(self, manifest):
        self.data = {'manifest': json.dumps(manifest).encode()}
        self.quota = {'limit': str(1 << 50), 'usage': '0'}
        self.created = []
        self.corrupt_readback = False

    def files(self):
        return self

    def get_media(self, *, fileId):
        def read():
            return b'wrong' if self.corrupt_readback and fileId != 'manifest' else self.data[fileId]
        return SimpleNamespace(execute=lambda **kwargs: read())

    def create(self, *, body, fields, media_body=None):
        def write():
            key = 'new-' + str(len(self.created))
            self.created.append(body)
            if media_body is not None:
                self.data[key] = media_body.payload
            return {'id': key, 'name': body['name']}
        return SimpleNamespace(execute=lambda **kwargs: write())

    def about(self):
        return SimpleNamespace(get=lambda **kwargs: SimpleNamespace(
            execute=lambda **kw: {'storageQuota': self.quota}))


@pytest.fixture
def cli(tmp_path, monkeypatch):
    identity = {'method': 'ar', 'script_sha256': 'unchanged-source'}
    manifest = {'status': 'verified', 'step': 40, 'bytes': 6_000_000_000}
    service = Service(manifest)
    identity_file = tmp_path / 'identity.json'
    identity_file.write_text(json.dumps(identity), encoding='utf-8')
    preparation, resumes = [], []
    now = [1000]
    monkeypatch.setattr(entry, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(entry, 'Path', lambda value: tmp_path if value == '/content' else Path(value))
    monkeypatch.setattr(entry, 'build_drive_service', lambda credentials: service)

    class Store:
        def __init__(self, service, *, deadline, **kwargs):
            self.service, self.deadline = service, deadline

        def check_time(self):
            if now[0] >= self.deadline:
                raise TimeoutError('expired')

        def metadata(self, file_id):
            self.check_time()
            return {'size': str(len(service.data[file_id]))}

        def call(self, action):
            self.check_time()
            return action()

    monkeypatch.setattr(entry, 'DriveCheckpointStore', Store)
    monkeypatch.setattr(entry.subprocess, 'run', lambda *args, **kwargs: preparation.append((args, kwargs)))
    google, auth, transport = (ModuleType(name) for name in ('google', 'google.auth', 'google.auth.transport'))
    google.__path__, auth.__path__, transport.__path__ = [], [], []
    google.auth = auth
    auth.default = lambda: (object(), None)
    requests = ModuleType('google.auth.transport.requests')
    requests.AuthorizedSession = lambda credentials: object()
    sdk, http = ModuleType('googleapiclient'), ModuleType('googleapiclient.http')
    sdk.__path__ = []
    http.MediaIoBaseUpload = Media
    for name, module in [('google', google), ('google.auth', auth),
                         ('google.auth.transport', transport), ('google.auth.transport.requests', requests),
                         ('googleapiclient', sdk), ('googleapiclient.http', http)]:
        monkeypatch.setitem(sys.modules, name, module)

    fault = {'report': None}

    def resume(argv, store, folder, staging, path, digest, expected_identity, **kwargs):
        resumes.append((argv, kwargs))
        staging.mkdir()
        report = {'identity': identity, 'start_step': 40, 'end_step': 200,
                  'checkpoint_sha256': 'checkpoint-sha', 'status': 'completed'}
        if fault['report'] == 'identity':
            report['identity'] = {'method': 'dlm'}
        elif fault['report'] == 'step':
            report['end_step'] = 199
        elif fault['report'] == 'checksum':
            report['checkpoint_sha256'] = 'different'
        elif fault['report'] == 'partial':
            report['status'] = 'partial'
        if fault['report'] != 'missing':
            (staging / 'report-40-200.json').write_text(json.dumps(report), encoding='utf-8')
        return {'step': 200, 'fingerprints': {'sha256': 'checkpoint-sha'},
                'ready_for_resume': False, 'requires_independent_restore_audit': True}

    monkeypatch.setattr(entry, 'resume_training', resume)

    def run(*extra):
        digest = hashlib.sha256(service.data['manifest']).hexdigest()
        monkeypatch.setattr(sys, 'argv', ['api-entry', '--manifest-id', 'manifest',
            '--manifest-sha256', digest, '--identity', str(identity_file),
            '--parent-folder', 'parent', '--deadline', '2000',
            '--publication-reserve-seconds', '300', '--steps', '200', *extra])
        entry.main()

    return SimpleNamespace(run=run, service=service, preparation=preparation, resumes=resumes,
                           fault=fault, now=now, manifest=manifest)


@pytest.mark.parametrize('deadline', ['999', '1000', 'nan', 'inf'])
def test_expired_or_nonfinite_deadline_refuses_before_api_or_preparation(cli, deadline):
    with pytest.raises(ValueError, match='No execution budget'):
        cli.run('--deadline', deadline)
    assert not cli.preparation and not cli.resumes and not cli.service.created


@pytest.mark.parametrize('reserve', ['0', '-1', 'nan', 'inf', '820'])
def test_invalid_or_insufficient_publication_reserve_refuses(cli, reserve):
    with pytest.raises(ValueError, match='publication reserve'):
        cli.run('--publication-reserve-seconds', reserve)
    assert not cli.preparation and not cli.resumes and not cli.service.created


@pytest.mark.parametrize('every,slots', [(20, 9), (100, 3)])
def test_drive_quota_one_byte_short_blocks_before_preparation(cli, every, slots):
    required = slots * cli.manifest['bytes'] + (1 << 20)
    cli.service.quota = {'limit': str(required - 1), 'usage': '0'}
    with pytest.raises(OSError, match='Insufficient Drive quota'):
        cli.run('--cloud-checkpoint-every', str(every))
    assert not cli.preparation and not cli.resumes and not cli.service.created


@pytest.mark.parametrize('fault', ['identity', 'step', 'checksum', 'missing'])
def test_bad_continuation_report_is_never_published(cli, fault):
    cli.fault['report'] = fault
    with pytest.raises(ValueError, match='report|Report'):
        cli.run()
    assert len(cli.resumes) == 1
    assert len(cli.service.created) == 1  # Only the unique run folder exists.
    assert len(cli.service.data) == 1  # Original input manifest stays the only stored bytes.


@pytest.mark.parametrize('partial', [False, True])
def test_report_and_receipt_persist_with_independent_audit_required(cli, partial, capsys):
    if partial:
        cli.fault['report'] = 'partial'
    original_manifest = cli.service.data['manifest']
    cli.run()
    assert len(cli.preparation) == 2 and len(cli.resumes) == 1
    assert all(0 < kwargs['timeout'] <= 300 for _, kwargs in cli.preparation)
    assert cli.resumes[0][1]['publication_reserve_seconds'] == 300
    receipt = json.loads(cli.service.data['new-2'])
    assert receipt['ready_for_resume'] is False
    assert receipt['requires_independent_restore_audit'] is True
    assert receipt['training_status'] == ('partial' if partial else 'completed')
    assert cli.service.data['manifest'] == original_manifest
    assert 'INDEPENDENT_RESTORE_AUDIT_REQUIRED' in capsys.readouterr().out


def test_report_readback_failure_blocks_continuation_receipt(cli):
    cli.service.corrupt_readback = True
    with pytest.raises(ValueError, match='readback differs'):
        cli.run()
    assert len(cli.service.created) == 2  # Unique folder and first report, no receipt.
