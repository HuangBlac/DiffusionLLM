"""Create-only API publication with fake media, byte readback, and restore gates."""
import copy
import hashlib
import io
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import requests

from experiments.drive_checkpoint_store import DriveCheckpointStore, TransferError, build_drive_service


class FakeMedia:
    def __init__(self, source, **kwargs):
        self.payload = Path(source).read_bytes() if isinstance(source, str) else source.read()


class FakeRequest:
    def __init__(self, action, file_id=None):
        self.action = action
        self.file_id = file_id

    def execute(self, **kwargs):
        return self.action()

    def next_chunk(self, **kwargs):
        return None, self.action()


class FakeDownloader:
    def __init__(self, stream, request, **kwargs):
        self.stream, self.request = stream, request

    def next_chunk(self, **kwargs):
        self.stream.write(self.request.execute())
        return None, True


class FakeDrive:
    def __init__(self):
        self.data = {'original': b'old checkpoint', 'old-manifest': b'old verified manifest'}
        self.records = {
            'folder': {'id': 'folder', 'mimeType': 'application/vnd.google-apps.folder'},
            'original': self.meta('original', self.data['original']),
            'old-manifest': self.meta('old-manifest', self.data['old-manifest']),
        }
        self.events = []
        self.metadata_calls = {}
        self.bad_checksum = False
        self.change_during_download = False
        self.upload_error = None

    @staticmethod
    def meta(file_id, payload, name='old'):
        return {'id': file_id, 'name': name, 'size': str(len(payload)), 'version': '1',
                'sha256Checksum': hashlib.sha256(payload).hexdigest(),
                'md5Checksum': hashlib.md5(payload, usedforsecurity=False).hexdigest()}

    def files(self):
        return self

    def get(self, *, fileId, fields):
        def action():
            self.events.append(('metadata', fileId))
            count = self.metadata_calls.get(fileId, 0) + 1
            self.metadata_calls[fileId] = count
            result = copy.deepcopy(self.records[fileId])
            if fileId.startswith('new-') and self.bad_checksum:
                result['sha256Checksum'] = 'bad'
            if fileId.startswith('new-') and self.change_during_download and count >= 3:
                result['version'] = '2'
            return result
        return FakeRequest(action)

    def create(self, *, body, media_body, fields):
        is_checkpoint = body['name'].endswith('.pt')

        def action():
            if is_checkpoint and self.upload_error:
                raise self.upload_error
            file_id = 'new-' + str(len(self.records))
            self.data[file_id] = media_body.payload
            self.records[file_id] = self.meta(file_id, media_body.payload, body['name'])
            self.events.append(('upload' if is_checkpoint else 'manifest', file_id))
            return copy.deepcopy(self.records[file_id])
        return FakeRequest(action)

    def get_media(self, *, fileId):
        def action():
            self.events.append(('download', fileId))
            return self.data[fileId]
        return FakeRequest(action, fileId)


@pytest.fixture
def publication(tmp_path):
    service = FakeDrive()
    source = tmp_path / 'original.pt'
    source.write_bytes(b'new model optimizer and RNG state')
    now = [0]
    store = DriveCheckpointStore(service, deadline=10, clock=lambda: now[0],
                                media_factories=(FakeMedia, FakeDownloader, FakeMedia))

    def restore(path):
        assert path != source
        assert path.read_bytes() == source.read_bytes()
        assert not any(event[0] == 'manifest' for event in service.events)
        service.events.append(('restore', str(path)))
        return {'restored': True, 'step': 200, 'method': 'ar', 'identity_sha256': 'identity',
                'rng_state_restored': True, 'inference_finite': True}

    arguments = dict(source=source, folder_id='folder', method='ar', step=200,
                     identity_sha256='identity', verification_dir=tmp_path / 'fresh',
                     restore_check=restore)
    return service, store, source, now, arguments


def assert_originals_untouched(service, source):
    assert service.data['original'] == b'old checkpoint'
    assert service.data['old-manifest'] == b'old verified manifest'
    assert source.read_bytes() == b'new model optimizer and RNG state'


def test_publish_orders_checks_fresh_download_restore_and_manifest(publication):
    service, store, source, _, arguments = publication
    result = store.publish_verified(**arguments)
    kinds = [event[0] for event in service.events]
    assert kinds.index('upload') < kinds.index('download') < kinds.index('restore') < kinds.index('manifest')
    assert result['manifest']['status'] == 'verified'
    assert result['manifest']['fresh_download_verified'] is True
    assert result['manifest']['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert service.data[result['manifest_id']].decode().find('"step": 200') >= 0
    assert_originals_untouched(service, source)


@pytest.mark.parametrize('fault, message', [
    ('bad_checksum', 'SHA-256 differs'),
    ('change_during_download', 'changed during download'),
])
def test_transfer_failures_do_not_publish_manifest(publication, fault, message):
    service, store, source, _, arguments = publication
    setattr(service, fault, True)
    with pytest.raises(TransferError, match=message):
        store.publish_verified(**arguments)
    assert not any(event[0] == 'manifest' for event in service.events)
    assert_originals_untouched(service, source)


def test_restore_failure_blocks_manifest(publication):
    service, store, source, _, arguments = publication
    arguments['restore_check'] = lambda path: {'restored': False}
    with pytest.raises(TransferError, match='restore did not pass'):
        store.publish_verified(**arguments)
    assert not any(event[0] == 'manifest' for event in service.events)
    assert_originals_untouched(service, source)


def test_version_change_after_restore_blocks_manifest(publication):
    service, store, source, _, arguments = publication
    original_restore = arguments['restore_check']

    def change_after_restore(path):
        restored = original_restore(path)
        candidate = next(k for k in service.records if k.startswith('new-'))
        service.records[candidate]['version'] = '2'
        return restored

    arguments['restore_check'] = change_after_restore
    with pytest.raises(TransferError, match='changed before manifest'):
        store.publish_verified(**arguments)
    assert not any(event[0] == 'manifest' for event in service.events)
    assert_originals_untouched(service, source)


def test_expired_deadline_has_no_api_side_effects(publication):
    service, store, source, now, arguments = publication
    now[0] = 10
    with pytest.raises(TimeoutError):
        store.publish_verified(**arguments)
    assert not service.events
    assert_originals_untouched(service, source)


def test_deadline_expiring_during_restore_blocks_manifest(publication):
    service, store, source, now, arguments = publication
    original_restore = arguments['restore_check']

    def slow_restore(path):
        restored = original_restore(path)
        now[0] = 11
        return restored

    arguments['restore_check'] = slow_restore
    with pytest.raises(TimeoutError):
        store.publish_verified(**arguments)
    assert not any(event[0] == 'manifest' for event in service.events)
    assert_originals_untouched(service, source)


def test_api_errors_hide_credential_bearing_session_urls(publication):
    service, store, source, _, arguments = publication
    error = RuntimeError('https://upload.example/session?secret=DO_NOT_DISCLOSE')
    error.resp = SimpleNamespace(status=503)
    service.upload_error = error
    with pytest.raises(TransferError) as captured:
        store.publish_verified(**arguments)
    assert 'status=503' in str(captured.value)
    assert 'secret' not in str(captured.value)
    assert 'upload.example' not in str(captured.value)
    assert not any(event[0] == 'manifest' for event in service.events)
    assert_originals_untouched(service, source)


def test_verified_download_does_not_replace_existing_destination(publication, tmp_path):
    service, store, _, _, _ = publication
    destination = tmp_path / 'existing.pt'
    destination.write_bytes(b'keep local checkpoint')
    with pytest.raises(FileExistsError):
        store.download_verified('original', destination)
    assert destination.read_bytes() == b'keep local checkpoint'
    assert not service.events


def test_service_factory_preserves_resumable_308_and_timeout(monkeypatch):
    """308 must reach googleapiclient's resumable upload code, not HTTP redirects."""
    http = SimpleNamespace(redirect_codes={301, 302, 303, 307, 308}, timeout=None)
    seen = {}
    auth = ModuleType('google_auth_httplib2')

    def authorize(credentials, *, http):
        seen.update(credentials=credentials, transport=http)
        return SimpleNamespace(transport=http)

    auth.AuthorizedHttp = authorize
    sdk = ModuleType('googleapiclient')
    sdk.__path__ = []
    discovery, transport = ModuleType('googleapiclient.discovery'), ModuleType('googleapiclient.http')
    transport.build_http = lambda: http

    def build(api, version, **kwargs):
        seen.update(api=api, version=version, kwargs=kwargs)
        return 'drive-service'

    discovery.build = build
    monkeypatch.setitem(sys.modules, 'google_auth_httplib2', auth)
    monkeypatch.setitem(sys.modules, 'googleapiclient', sdk)
    monkeypatch.setitem(sys.modules, 'googleapiclient.discovery', discovery)
    monkeypatch.setitem(sys.modules, 'googleapiclient.http', transport)
    logger = logging.getLogger('googleapiclient.http')
    monkeypatch.setattr(logger, 'disabled', False)
    credentials = object()
    assert build_drive_service(credentials, timeout=37) == 'drive-service'
    assert seen['credentials'] is credentials
    assert seen['transport'] is http
    assert http.redirect_codes == {301, 302, 303, 307}
    assert http.timeout == 37
    assert seen['api'] == 'drive' and seen['version'] == 'v3'
    assert seen['kwargs']['cache_discovery'] is False
    assert seen['kwargs']['http'].transport is http
    assert logger.disabled


@pytest.mark.parametrize('chunk_bytes', [0, -1, 1, 256 * 1024 + 1,
                                        65 << 20, True, 256 * 1024.0])
def test_invalid_chunk_sizes_refuse_before_any_transfer(chunk_bytes):
    with pytest.raises(ValueError, match='256KiB multiple'):
        DriveCheckpointStore(FakeDrive(), deadline=10, chunk_bytes=chunk_bytes)


def test_configured_chunk_size_reaches_upload_and_download(publication, tmp_path):
    service, _, source, _, arguments = publication
    sizes = []

    class TrackedUpload(FakeMedia):
        def __init__(self, source, **kwargs):
            sizes.append(('upload', kwargs['chunksize'], kwargs['resumable']))
            super().__init__(source, **kwargs)

    class TrackedDownload(FakeDownloader):
        def __init__(self, stream, request, **kwargs):
            sizes.append(('download', kwargs['chunksize']))
            super().__init__(stream, request, **kwargs)

    store = DriveCheckpointStore(service, deadline=10, clock=lambda: 0,
                                chunk_bytes=256 << 10,
                                media_factories=(TrackedUpload, TrackedDownload, FakeMedia))
    result = store.publish_verified(**arguments)
    assert result['manifest']['fresh_download_verified']
    assert sizes == [('upload', 256 << 10, True), ('download', 256 << 10)]
    assert_originals_untouched(service, source)


class StreamingResponse:
    def __init__(self, blocks, *, total=16, start=0, stop=15, status=206, headers=None,
                 on_block=None):
        self.status_code = status
        self.headers = {'Content-Range': f'bytes {start}-{stop}/{total}', **(headers or {})}
        self.blocks, self.on_block = blocks, on_block
        self.body_read, self.closed, self.requested_block_sizes = False, False, []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def iter_content(self, chunk_size):
        self.body_read = True
        self.requested_block_sizes.append(chunk_size)
        for block in self.blocks:
            if isinstance(block, BaseException):
                raise block
            if self.on_block:
                self.on_block()
            yield block


class StreamingSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.mark.parametrize('status,headers', [
    (200, {}),
    (403, {}),
    (206, {'Content-Range': 'bytes 0-14/16'}),
    (206, {'Content-Encoding': 'gzip'}),
])
def test_ignored_or_invalid_ranges_never_read_response_body(status, headers):
    response = StreamingResponse([AssertionError('body must never be consumed')],
                                 status=status, headers=headers)
    session = StreamingSession([response])
    store = DriveCheckpointStore(FakeDrive(), deadline=10, clock=lambda: 0,
                                stream_session=session)
    with pytest.raises(TransferError):
        store._stream_download('original', io.BytesIO(), 16)
    assert not response.body_read
    assert response.closed
    options = session.calls[0][1]
    assert options['stream'] is True and options['allow_redirects'] is False
    assert options['headers']['Accept-Encoding'] == 'identity'


def test_stream_overflow_fails_before_writing_excess_bytes():
    response = StreamingResponse([b'x' * 17])
    stream = io.BytesIO()
    store = DriveCheckpointStore(FakeDrive(), deadline=10, clock=lambda: 0,
                                stream_session=StreamingSession([response]))
    with pytest.raises(TransferError, match='body exceeds byte range'):
        store._stream_download('original', stream, 16)
    assert stream.getvalue() == b''
    assert response.closed


@pytest.mark.parametrize('fault', ['short', 'timeout-get', 'timeout-body', 'chunked-body'])
def test_transient_range_failure_retries_without_duplicate_partial_bytes(monkeypatch, fault):
    payload = b'0123456789abcdef'
    if fault == 'short':
        failed = StreamingResponse([payload[:5]])
    elif fault == 'timeout-get':
        failed = requests.Timeout('temporary request timeout')
    else:
        failure = (requests.Timeout('temporary body timeout') if fault == 'timeout-body'
                   else requests.exceptions.ChunkedEncodingError('temporary broken body'))
        failed = StreamingResponse([payload[:5], failure])
    succeeded = StreamingResponse([payload])
    session = StreamingSession([failed, succeeded])
    monkeypatch.setattr('experiments.drive_checkpoint_store.time.sleep', lambda seconds: None)
    stream = io.BytesIO()
    store = DriveCheckpointStore(FakeDrive(), deadline=10, clock=lambda: 0,
                                stream_session=session)
    store._stream_download('original', stream, len(payload))
    assert stream.getvalue() == payload
    assert len(session.calls) == 2
    assert [options['headers']['Range'] for _, options in session.calls] == ['bytes=0-15'] * 2
    assert succeeded.closed
    if isinstance(failed, StreamingResponse):
        assert failed.closed


def test_stream_deadline_during_body_stops_without_retry():
    now = [0]
    response = StreamingResponse([b'x' * 16], on_block=lambda: now.__setitem__(0, 10))
    session = StreamingSession([response])
    stream = io.BytesIO()
    store = DriveCheckpointStore(FakeDrive(), deadline=10, clock=lambda: now[0],
                                stream_session=session)
    with pytest.raises(TimeoutError):
        store._stream_download('original', stream, 16)
    assert stream.getvalue() == b''
    assert len(session.calls) == 1 and response.closed


def test_stream_retries_are_bounded_and_hide_underlying_request_urls(monkeypatch):
    session = StreamingSession([
        requests.Timeout('https://request.example/?credential=hidden') for _ in range(3)])
    monkeypatch.setattr('experiments.drive_checkpoint_store.time.sleep', lambda seconds: None)
    store = DriveCheckpointStore(FakeDrive(), deadline=10, clock=lambda: 0,
                                stream_session=session)
    with pytest.raises(TransferError, match='bounded retries') as captured:
        store._stream_download('original', io.BytesIO(), 16)
    assert len(session.calls) == 3
    assert 'credential' not in str(captured.value)
    assert 'request.example' not in str(captured.value)


def test_retry_keeps_previously_completed_ranges(monkeypatch):
    chunk = 256 << 10
    payload = b'A' * chunk + b'B' * 16
    first = StreamingResponse([b'A' * (64 << 10) for _ in range(4)],
                              total=len(payload), stop=chunk - 1)
    failed = StreamingResponse([b'B' * 5], total=len(payload), start=chunk, stop=len(payload) - 1)
    final = StreamingResponse([b'B' * 16], total=len(payload), start=chunk, stop=len(payload) - 1)
    session = StreamingSession([first, failed, final])
    monkeypatch.setattr('experiments.drive_checkpoint_store.time.sleep', lambda seconds: None)
    stream = io.BytesIO()
    store = DriveCheckpointStore(FakeDrive(), deadline=10, clock=lambda: 0,
                                chunk_bytes=chunk, stream_session=session)
    store._stream_download('original', stream, len(payload))
    assert stream.getvalue() == payload
    assert [options['headers']['Range'] for _, options in session.calls] == [
        f'bytes=0-{chunk-1}', f'bytes={chunk}-{len(payload)-1}', f'bytes={chunk}-{len(payload)-1}']


def test_real_multirange_streaming_download_matches_bytes_and_fingerprints(tmp_path):
    payload = bytes(range(256)) * 2300
    responses = []
    for start in range(0, len(payload), 256 << 10):
        stop = min(start + (256 << 10), len(payload))
        blocks = [payload[i:min(i + (64 << 10), stop)] for i in range(start, stop, 64 << 10)]
        responses.append(StreamingResponse(blocks, total=len(payload), start=start, stop=stop - 1))
    session, service, progress = StreamingSession(responses), FakeDrive(), []
    service.data['original'] = payload
    service.records['original'] = service.meta('original', payload)
    store = DriveCheckpointStore(service, deadline=10, clock=lambda: 0,
                                chunk_bytes=256 << 10, stream_session=session,
                                media_factories=(FakeMedia, FakeDownloader, FakeMedia),
                                progress=progress.append)
    destination = tmp_path / 'downloaded.pt'
    result = store.download_verified('original', destination)
    assert destination.read_bytes() == payload
    assert result['fingerprints']['sha256'] == hashlib.sha256(payload).hexdigest()
    assert [options['headers']['Range'] for _, options in session.calls] == [
        'bytes=0-262143', 'bytes=262144-524287', f'bytes=524288-{len(payload)-1}']
    assert [event['bytes'] for event in progress] == [262144, 524288, len(payload)]
    assert all(response.closed and response.requested_block_sizes == [64 << 10]
               for response in responses)
    assert not list(tmp_path.glob('*.partial'))


def test_large_file_refuses_unbounded_legacy_downloader(tmp_path):
    service = FakeDrive()
    service.records['original']['size'] = str(129 << 20)
    store = DriveCheckpointStore(service, deadline=10, clock=lambda: 0,
                                media_factories=(FakeMedia, FakeDownloader, FakeMedia))
    with pytest.raises(ValueError, match='require bounded streaming'):
        store.download_verified('original', tmp_path / 'large.pt')
    assert not (tmp_path / 'large.pt').exists()
    assert not any(event[0] == 'download' for event in service.events)
