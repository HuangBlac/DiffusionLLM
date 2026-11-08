"""Local behavioral prototype for a proposed P2 ablation; this file does not train."""
import torch
from experiments.full_parameter_pilot import ConditionalModel, pack


def pack_canvas(rows, bos, device, *, target_capacity=128):
    if type(target_capacity) is not int or target_capacity <= 0:
        raise ValueError('Need a positive target canvas capacity')
    prefixes = {row['prefix_length'] for row in rows}
    if len(prefixes) != 1:
        raise ValueError('A canvas batch needs a shared prefix length')
    prefix = next(iter(prefixes))
    if any(len(row['ids']) > prefix+target_capacity for row in rows):
        raise ValueError('Source example exceeds the canvas')
    ids, valid, targets = pack(rows, bos, device)
    width = prefix+target_capacity+1
    canvas = torch.full((len(rows), width), bos, dtype=ids.dtype, device=ids.device)
    canvas_valid = torch.zeros_like(canvas, dtype=torch.bool)
    canvas_targets = torch.zeros_like(canvas_valid)
    canvas[:, :ids.shape[1]] = ids
    canvas_valid[:, :ids.shape[1]] = valid
    canvas_targets[:, :ids.shape[1]] = targets
    return canvas, canvas_valid, canvas_targets


class MaskedPaddingCanvasDLM(ConditionalModel):
    """Hide padding IDs and expose a common attention canvas; pad labels stay excluded.

    This removes the explicit true-end attention cutoff, not every possible
    statistical cue to length. Target/noise definitions remain in the frozen loss.
    """
    def __init__(self, base):
        super().__init__(base, 'dlm')

    def hidden(self, ids, valid, masked):
        return super().hidden(ids, torch.ones_like(valid), masked | ~valid)
