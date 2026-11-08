import hashlib
from types import SimpleNamespace

import pytest
from experiments.pinned_drive_artifact import read_pinned_artifact
from experiments.drive_checkpoint_store import TransferError

PAYLOAD = b'{"dataset":"public"}\n'
EXPECTED = {'bytes': len(PAYLOAD), 'sha256': hashlib.sha256(PAYLOAD).hexdigest(),
            'md5': hashlib.md5(PAYLOAD, usedforsecurity=False).hexdigest()}
PROOF = {'candidate_fingerprints': EXPECTED, 'manifest_sha256': EXPECTED['sha256'],
         'candidate_file': {'id': 'candidate', 'version': '1'}}


class Store:
    def __init__(self, payload=PAYLOAD, before_version='2', after_version='3'):
        self.payload, self.reads = payload, 0
        metadata = {'id': 'candidate', 'size': str(len(PAYLOAD)), 'sha256Checksum': EXPECTED['sha256'],
                    'md5Checksum': EXPECTED['md5']}
        self.before = {**metadata, 'version': before_version}
        self.after = {**metadata, 'version': after_version}
        self.service = SimpleNamespace(files=lambda: SimpleNamespace(get_media=self.get_media))

    def get_media(self, **kwargs):
        self.reads += 1
        return SimpleNamespace(execute=lambda **kwargs: self.payload)

    def metadata(self, file_id):
        return self.before if self.reads == 0 else self.after

    def call(self, operation):
        return operation()


def test_metadata_version_changes_are_recorded_only_when_all_bytes_match():
    data, receipt = read_pinned_artifact(Store(), PROOF)
    assert data == PAYLOAD and receipt['content_verified']
    assert receipt['metadata_version_differs_from_proof']
    assert (receipt['proof_version'], receipt['version_before_read'], receipt['version_after_read']) == ('1', '2', '3')
    _, after_only = read_pinned_artifact(Store(before_version='1', after_version='2'), PROOF)
    assert after_only['metadata_version_differs_from_proof']


@pytest.mark.parametrize('field,value', [('id', 'other'), ('size', '999999999'),
                                      ('sha256Checksum', 'changed'), ('md5Checksum', 'changed')])
def test_changed_metadata_fails_before_downloading(field, value):
    store = Store()
    store.before[field] = value
    with pytest.raises((ValueError, TransferError)):
        read_pinned_artifact(store, PROOF)
    assert store.reads == 0


def test_download_or_after_metadata_change_cannot_pass():
    with pytest.raises(TransferError):
        read_pinned_artifact(Store(payload=b'changed'), PROOF)
    store = Store()
    store.after['sha256Checksum'] = 'changed'
    with pytest.raises(TransferError):
        read_pinned_artifact(store, PROOF)


def test_size_limit_and_inconsistent_proof_fail_closed():
    store = Store()
    with pytest.raises(ValueError):
        read_pinned_artifact(store, PROOF, max_bytes=2)
    with pytest.raises(ValueError):
        read_pinned_artifact(store, {**PROOF, 'manifest_sha256': 'changed'})
    assert store.reads == 0
