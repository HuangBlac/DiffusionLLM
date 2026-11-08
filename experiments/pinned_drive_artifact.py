"""Read a content-pinned JSON artifact without equating metadata versions with bytes."""
import hashlib

from experiments.drive_checkpoint_store import matches_server, TransferError


def read_pinned_artifact(store, proof, *, max_bytes=64 << 20):
    expected = proof['candidate_fingerprints']
    file_id = proof['candidate_file']['id']
    if expected['sha256'] != proof['manifest_sha256'] or not 0 < expected['bytes'] <= max_bytes:
        raise ValueError('Invalid pinned artifact proof')

    def check(metadata):
        if metadata.get('id') != file_id or metadata.get('sha256Checksum') != expected['sha256']:
            raise TransferError('Artifact ID or server SHA differs from proof')
        if metadata.get('md5Checksum') != expected['md5']:
            raise TransferError('Artifact server MD5 differs from proof')
        matches_server(metadata, expected)

    before = store.metadata(file_id)
    check(before)
    payload = store.call(lambda: store.service.files().get_media(fileId=file_id).execute(num_retries=1))
    if (len(payload) != expected['bytes'] or hashlib.sha256(payload).hexdigest() != expected['sha256']
            or hashlib.md5(payload, usedforsecurity=False).hexdigest() != expected['md5']):
        raise TransferError('Downloaded artifact bytes differ from proof')
    after = store.metadata(file_id)
    check(after)
    return payload, {'content_verified': True, 'sha256': expected['sha256'], 'bytes': len(payload),
                     'proof_version': proof['candidate_file'].get('version'),
                     'version_before_read': before.get('version'), 'version_after_read': after.get('version'),
                     'metadata_version_differs_from_proof': any(
                         metadata.get('version') != proof['candidate_file'].get('version')
                         for metadata in (before, after))}
