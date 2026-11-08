"""Explicit single-site activation capture/patch helpers for generic PyTorch.

These intentionally support modules returning a single Tensor only. For tuple,
sharded or cache-aware transformer modules use an architecture-aware adapter.
A causal claim requires aligned clean/corrupt inputs and controlled experiments.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any


def _resolve(model: Any, name: str):
    modules = dict(model.named_modules())
    if name not in modules:
        raise ValueError(f"unknown module {name!r}; available: {list(modules)[:15]}")
    return modules[name]


def capture_activation(model: Any, name: str, *args: Any, max_elements: int = 2_000_000, **kwargs: Any):
    """Run once and capture a detached activation. Fails on multiple calls/non-Tensor.

    The caller controls `torch.no_grad`, `eval()`, RNG, and device. Captured value
    stays on its original device; explicitly estimate memory before use.
    """
    import torch
    if max_elements < 1:
        raise ValueError("max_elements must be positive")
    module = _resolve(model, name)
    captured = []

    def hook(_module, _inputs, out):
        if not isinstance(out, torch.Tensor):
            raise TypeError("only single tensor module outputs are supported")
        if out.numel() > max_elements:
            raise MemoryError(f"activation {name} has {out.numel()} > max_elements={max_elements}")
        captured.append(out.detach().clone())

    handle = module.register_forward_hook(hook)
    try:
        result = model(*args, **kwargs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError(f"expected one invocation of {name}, observed {len(captured)}")
    return result, captured[0]


@contextmanager
def inject_activation(model: Any, name: str, source: Any, *, selection: Any = None,
                      alpha: float = 1.0):
    """Temporarily interpolate a matched source activation into selected site.

    `selection` indexes the *same region* of clean and corrupt activations.
    1.0 means full replacement. This is potentially off-distribution.
    A yielded dictionary records the hook call count; test after exit.
    """
    import torch
    if not isinstance(source, torch.Tensor):
        raise TypeError("source must be tensor")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    module = _resolve(model, name)
    info = {"calls": 0, "name": name, "alpha": alpha, "selection": repr(selection)}

    def hook(_module, _inputs, out):
        if not isinstance(out, torch.Tensor):
            raise TypeError("only single tensor module outputs are supported")
        if out.shape != source.shape or out.device != source.device:
            raise ValueError(f"patch shape/device mismatch: {tuple(out.shape)} {out.device} != "
                             f"{tuple(source.shape)} {source.device}")
        if not torch.isfinite(out).all().item() or not torch.isfinite(source).all().item():
            raise ValueError("cannot patch nonfinite activations")
        info["calls"] += 1
        patched = out.clone()
        ix = (slice(None),) * out.ndim if selection is None else selection
        original_part = out[ix]
        source_part = source[ix].to(dtype=out.dtype)
        patched[ix] = (1 - alpha) * original_part + alpha * source_part
        return patched

    handle = module.register_forward_hook(hook)
    try:
        yield info
    finally:
        handle.remove()


def run_patch_pair(model: Any, name: str, clean_args: tuple[Any, ...],
                   corrupt_args: tuple[Any, ...], *, selection: Any = None,
                   alpha: float = 1.0, kwargs: dict[str, Any] | None = None):
    """Return clean, corrupt, patched outputs plus a basic intervention record.

    Does not assert input/counterfactual validity; researcher must construct and
    document a valid matched pair and metric. Only deterministic inference runs.
    """
    import torch
    kwargs = kwargs or {}
    with torch.inference_mode():
        clean, source = capture_activation(model, name, *clean_args, **kwargs)
        corrupt = model(*corrupt_args, **kwargs)
        with inject_activation(model, name, source, selection=selection, alpha=alpha) as info:
            patched = model(*corrupt_args, **kwargs)
    if info["calls"] != 1:
        raise RuntimeError(f"patch expected exactly one hook execution: {info['calls']}")
    return {"clean": clean, "corrupt": corrupt, "patched": patched, "intervention": info}
