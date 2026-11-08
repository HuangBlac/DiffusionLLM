import io

import pytest
import torch

from experiments.checkpoint_metadata import RangeReader, inspect_metadata


def test_inspection_skips_tensor_storage_and_reads_step():
    output = io.BytesIO()
    torch.save({'model': {'weight': torch.zeros(2_000_000)}, 'optimizer': {'state': {}},
                'step': 40, 'order': (), 'masks': torch.zeros(1), 'torch_rng': torch.zeros(1),
                'identity': {'method': 'ar'}, 'history': [{'step': 40}]}, output)
    payload = output.getvalue()
    reader = RangeReader(len(payload), lambda a, b: payload[a:b + 1],
                         block_bytes=65536, max_bytes=1 << 20)
    result = inspect_metadata(reader)
    assert result['step'] == 40
    assert result['metadata_only']
    assert reader.transferred < 1 << 20


def test_range_bound_and_ignored_range_fail_closed():
    r = RangeReader(100, lambda a, b: b'x' * 100, block_bytes=10, max_bytes=20)
    with pytest.raises(ValueError, match='honor'):
        r.read(1)
    r = RangeReader(100, lambda a, b: b'x' * (b - a + 1), block_bytes=10, max_bytes=20)
    with pytest.raises(ValueError, match='tensor-sized'):
        r.read(99)
    r.read(20)
    with pytest.raises(ValueError, match='limit'):
        r.read(1)


def test_many_distributed_tensor_headers_stay_bounded():
    output = io.BytesIO()
    torch.save({'model': {str(i): torch.zeros(262144) for i in range(64)},
                'optimizer': {'state': {}}, 'step': 40, 'order': (),
                'masks': torch.zeros(1), 'torch_rng': torch.zeros(1),
                'identity': {'method': 'ar'}, 'history': [{'step': 40}]}, output)
    payload = output.getvalue()
    reader = RangeReader(len(payload), lambda a, b: payload[a:b + 1], max_bytes=4 << 20)
    assert inspect_metadata(reader)['model_tensor_count'] == 64
    assert reader.transferred < 4 << 20
