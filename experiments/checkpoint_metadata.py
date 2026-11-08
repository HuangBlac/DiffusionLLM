"""Bounded metadata inspection of our PyTorch checkpoints, without tensor reads."""
import io
from collections import OrderedDict


class RangeReader(io.RawIOBase):
    """Seekable byte ranges with a strict transfer bound and small block cache."""

    def __init__(self, size, fetch_range, *, block_bytes=16 << 10, max_bytes=32 << 20):
        self.size = int(size)
        self.fetch_range = fetch_range
        self.block_bytes = block_bytes
        self.max_bytes = max_bytes
        self.position = 0
        self.transferred = 0
        self.cache = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        position = origin + offset
        if position < 0:
            raise ValueError('Negative checkpoint offset')
        self.position = position
        return position

    def read(self, count=-1):
        count = max(0, self.size - self.position) if count < 0 else count
        count = min(count, max(0, self.size - self.position))
        if count > self.max_bytes:
            raise ValueError('Metadata inspection requested tensor-sized read')
        chunks = []
        while count:
            start = self.position // self.block_bytes * self.block_bytes
            stop = min(start + self.block_bytes, self.size)
            if start not in self.cache:
                if self.transferred + stop - start > self.max_bytes:
                    raise ValueError('Checkpoint metadata transfer limit exceeded')
                payload = self.fetch_range(start, stop - 1)
                if len(payload) != stop - start:
                    raise ValueError('Server did not honor the requested byte range')
                self.transferred += len(payload)
                self.cache[start] = payload
                if len(self.cache) > 64:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(start)
            offset = self.position - start
            take = min(count, stop - self.position)
            chunks.append(self.cache[start][offset:offset + take])
            self.position += take
            count -= take
        return b''.join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def inspect_metadata(stream):
    """Metadata only; this does not establish tensor integrity or restorability."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        state = torch.load(stream, map_location='cpu', weights_only=True)
    required = {'model', 'optimizer', 'step', 'order', 'masks', 'torch_rng', 'identity', 'history'}
    if not required.issubset(state):
        raise ValueError('Incomplete checkpoint state')
    if not state['history'] or state['history'][-1]['step'] != state['step']:
        raise ValueError('Checkpoint step and recorded history differ')
    return {'step': state['step'], 'history_length': len(state['history']),
            'last_history_step': state['history'][-1]['step'], 'identity': state['identity'],
            'model_tensor_count': len(state['model']),
            'optimizer_state_count': len(state['optimizer']['state']),
            'tensor_integrity_verified': False, 'metadata_only': True}
