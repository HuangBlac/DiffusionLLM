"""Immutable checkpoint publication through Drive API, never a mounted cache.

A manifest is published only after server checks, a fresh full download and an
independent restore check. Existing files and the last good manifest are untouched.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid
from urllib.parse import quote


FIELDS = 'id,name,size,version,sha256Checksum,md5Checksum,mimeType,parents'


def build_drive_service(credentials, timeout=90):
    """Use the SDK transport: HTTP 308 is resumable progress, not a redirect."""
    import logging
    import google_auth_httplib2
    from googleapiclient.discovery import build
    from googleapiclient.http import build_http

    # SDK retry warnings can include credential-bearing resumable session URLs.
    # Our call wrapper still surfaces sanitized failures and progress explicitly.
    logging.getLogger('googleapiclient.http').disabled = True
    http = build_http()
    http.timeout = timeout
    # Keep this explicit even if a future SDK transport changes its defaults.
    http.redirect_codes = set(http.redirect_codes) - {308}
    authorized = google_auth_httplib2.AuthorizedHttp(credentials, http=http)
    return build('drive', 'v3', http=authorized, cache_discovery=False)


class TransferError(RuntimeError):
    pass


def fingerprints(path):
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    size = 0
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 << 20):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return {'bytes': size, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}


def matches_server(meta, expected):
    if int(meta.get('size', -1)) != expected['bytes']:
        raise TransferError('Server checkpoint size differs')
    sha = meta.get('sha256Checksum')
    md5 = meta.get('md5Checksum')
    if sha:
        if sha != expected['sha256']:
            raise TransferError('Server checkpoint SHA-256 differs')
    elif not md5 or md5 != expected['md5']:
        raise TransferError('Server checkpoint checksum missing or different')


class DriveCheckpointStore:
    def __init__(self, service, *, deadline, progress=None, clock=time.monotonic,
                 media_factories=None, chunk_bytes=64 << 20, stream_session=None):
        if type(chunk_bytes) is not int or not 0 < chunk_bytes <= 64 << 20 or chunk_bytes % (256 << 10):
            raise ValueError('Chunk size must be a 256KiB multiple, at most 64MiB')
        self.service = service
        self.deadline = deadline
        self.clock = clock
        self.progress = progress or (lambda event: None)
        self.media_factories = media_factories
        self.chunk_bytes = chunk_bytes
        self.stream_session = stream_session

    def _stream_download(self, file_id, stream, total):
        """Bound memory and reject ignored ranges before reading any response body."""
        import requests

        offset = 0
        url = 'https://www.googleapis.com/drive/v3/files/' + quote(file_id, safe='')
        while offset < total:
            stop = min(offset + self.chunk_bytes, total) - 1
            for attempt in range(3):
                self.check_time()
                try:
                    stream.seek(offset)
                    stream.truncate(offset)
                    with self.stream_session.get(
                        url, params={'alt': 'media'},
                        headers={'Range': f'bytes={offset}-{stop}', 'Accept-Encoding': 'identity'},
                        stream=True, allow_redirects=False, timeout=(15, 90),
                    ) as response:
                        if response.status_code in (408, 429, 500, 502, 503, 504):
                            raise requests.ConnectionError('Transient Drive response')
                        if response.status_code != 206:
                            raise TransferError('Download range refused or ignored')
                        if response.headers.get('Content-Range') != f'bytes {offset}-{stop}/{total}':
                            raise TransferError('Download Content-Range differs')
                        if response.headers.get('Content-Encoding', 'identity') != 'identity':
                            raise TransferError('Encoded download response refused')
                        received = 0
                        for block in response.iter_content(64 << 10):
                            self.check_time()
                            received += len(block)
                            if received > stop - offset + 1:
                                raise TransferError('Download body exceeds byte range')
                            stream.write(block)
                        if received != stop - offset + 1:
                            raise requests.ConnectionError('Truncated Drive byte range')
                    break
                except (requests.ConnectionError, requests.Timeout,
                        requests.exceptions.ChunkedEncodingError):
                    if attempt == 2:
                        raise TransferError('Streaming download failed after bounded retries') from None
                    self.check_time()
                    time.sleep(2 ** attempt)
            offset = stop + 1
            self.progress({'phase': 'download', 'bytes': offset, 'total': total})

    def check_time(self):
        if self.clock() >= self.deadline:
            raise TimeoutError('Checkpoint transfer deadline reached; no verified manifest published')

    def call(self, action):
        self.check_time()
        try:
            value = action()
        except Exception as exc:
            # Do not expose credential-bearing resumable session URLs in exceptions.
            status = getattr(getattr(exc, 'resp', None), 'status', None)
            raise TransferError(f'Drive operation failed ({type(exc).__name__}, status={status})') from None
        self.check_time()
        return value

    def media(self):
        if self.media_factories is None:
            from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload
            self.media_factories = (MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload)
        return self.media_factories

    def metadata(self, file_id):
        return self.call(lambda: self.service.files().get(
            fileId=file_id, fields=FIELDS).execute(num_retries=2))

    def download_verified(self, file_id, destination, *, expected=None):
        """Read committed bytes into a new local file; reject version changes."""
        started = self.clock()
        _, downloader_type, _ = self.media()
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = self.metadata(file_id)
        if int(before['size']) > 128 << 20 and self.stream_session is None:
            raise ValueError('Large checkpoints require bounded streaming download')
        if expected is not None:
            matches_server(before, expected)
        partial = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.partial')
        with partial.open('xb') as stream:
            if self.stream_session is not None:
                self._stream_download(file_id, stream, int(before['size']))
            else:
                downloader = downloader_type(stream, self.service.files().get_media(fileId=file_id),
                                             chunksize=self.chunk_bytes)
                done = False
                while not done:
                    status, done = self.call(lambda: downloader.next_chunk(num_retries=2))
                    if status is not None:
                        self.progress({'phase': 'download', 'bytes': status.resumable_progress,
                                       'total': int(before['size'])})
            stream.flush()
            os.fsync(stream.fileno())
        actual = fingerprints(partial)
        matches_server(before, actual)
        if expected is not None and actual != expected:
            raise TransferError('Fresh download does not match original local bytes')
        after = self.metadata(file_id)
        if any(before.get(k) != after.get(k) for k in
               ('id', 'size', 'version', 'sha256Checksum', 'md5Checksum')):
            raise TransferError('Cloud file changed during download')
        # Atomic no-overwrite publication on local storage. Never overwrite a checkpoint.
        os.link(partial, destination)
        partial.unlink()
        return {'path': str(destination), 'fingerprints': actual, 'server': after,
                'download_elapsed_seconds': self.clock() - started}

    def upload_snapshot(self, source, folder_id, *, method, step):
        """Server-verified immutable bytes, not yet a restore-verified manifest."""
        started = self.clock()
        if method not in ('ar', 'dlm') or not isinstance(step, int) or step < 0:
            raise ValueError('Invalid checkpoint identity')
        self.check_time()
        folder = self.metadata(folder_id)
        if folder.get('mimeType') != 'application/vnd.google-apps.folder':
            raise ValueError('Checkpoint destination must be a folder')
        source = Path(source)
        expected = fingerprints(source)
        upload_type, _, _ = self.media()
        name = f'{method}-step-{step:06d}-{expected["sha256"][:16]}-{uuid.uuid4().hex[:8]}.pt'
        media = upload_type(str(source), mimetype='application/octet-stream',
                            chunksize=self.chunk_bytes, resumable=True)
        request = self.service.files().create(
            body={'name': name, 'parents': [folder_id],
                  'appProperties': {'dllm_state': 'candidate', 'step': str(step)}},
            media_body=media, fields=FIELDS)
        response = None
        while response is None:
            status, response = self.call(lambda: request.next_chunk(num_retries=2))
            if status is not None:
                self.progress({'phase': 'upload', 'bytes': status.resumable_progress,
                               'total': expected['bytes']})
        candidate = self.metadata(response['id'])
        matches_server(candidate, expected)
        return {'status': 'server_verified', 'server': candidate, 'fingerprints': expected,
                'method': method, 'step': step, 'restore_verified': False,
                'upload_elapsed_seconds': self.clock() - started}

    def publish_verified(self, source, folder_id, *, method, step, identity_sha256,
                         verification_dir, restore_check):
        """Upload, then independently audit before publishing a verified manifest."""
        snapshot = self.upload_snapshot(source, folder_id, method=method, step=step)
        return self.verify_snapshot(snapshot, folder_id, identity_sha256=identity_sha256,
                                    verification_dir=verification_dir, restore_check=restore_check)

    def verify_snapshot(self, snapshot, folder_id, *, identity_sha256,
                        verification_dir, restore_check):
        """Audit an acknowledged snapshot, including after a prior runtime ended."""
        candidate, expected = snapshot['server'], snapshot['fingerprints']
        method, step, name = snapshot['method'], snapshot['step'], candidate['name']
        # A name returned by a server must not select a local destination path.
        if (not name or name in ('.', '..') or ':' in name or Path(name).name != name
                or '/' in name or '\\' in name):
            raise ValueError('Unsafe checkpoint filename')
        _, _, bytes_upload = self.media()
        downloaded = self.download_verified(candidate['id'], Path(verification_dir) / name,
                                             expected=expected)
        self.check_time()
        restored = restore_check(Path(downloaded['path']))
        if (restored.get('restored') is not True or restored.get('step') != step
                or restored.get('method') != method
                or restored.get('identity_sha256') != identity_sha256
                or restored.get('rng_state_restored') is not True
                or restored.get('inference_finite') is not True):
            raise TransferError('Independent checkpoint restore did not pass')
        current = self.metadata(candidate['id'])
        if current.get('version') != candidate.get('version'):
            raise TransferError('Checkpoint changed before manifest publication')
        matches_server(current, expected)
        manifest = {'schema': 1, 'status': 'verified', 'method': method, 'step': step,
                    'checkpoint_id': candidate['id'], 'checkpoint_version': current['version'],
                    'checkpoint_name': name, **expected, 'identity_sha256': identity_sha256,
                    'fresh_download_verified': True, 'restore': restored,
                    'verification_download_seconds': downloaded['download_elapsed_seconds']}
        payload = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
        result = self.call(lambda: self.service.files().create(
            body={'name': name + '.verified.json', 'parents': [folder_id]},
            media_body=bytes_upload(io.BytesIO(payload), mimetype='application/json'),
            fields=FIELDS).execute(num_retries=2))
        readback = self.call(lambda: self.service.files().get_media(
            fileId=result['id']).execute(num_retries=2))
        if readback != payload:
            raise TransferError('Published manifest readback differs')
        return {'manifest_id': result['id'], 'manifest': manifest}
