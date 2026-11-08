"""Bounded reads in logical row-major order, including noncontiguous tensors."""
from __future__ import annotations

from typing import Any


def flat_slice(tensor: Any, start: int, stop: int, step: int = 1):
    """Read a flat slice without first copying the complete tensor.

    Contiguous tensors use a view. Other layouts allocate coordinates and values
    only for the requested elements, with O(sample size * tensor rank) storage.
    Callers supply valid, nonnegative slice bounds and a positive step.
    """
    import torch

    tensor = tensor.detach()
    if tensor.is_contiguous():
        return tensor.view(-1)[start:stop:step]
    remaining = torch.arange(start, stop, step, device=tensor.device)
    coordinates = []
    for size in reversed(tensor.shape):
        coordinates.append(remaining.remainder(size))
        remaining = torch.div(remaining, size, rounding_mode="floor")
    return tensor[tuple(reversed(coordinates))]
