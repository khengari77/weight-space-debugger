"""Non-mutating in-memory sparse weight edits with explicit footprints.

Refuses tied/shared parameters, quantized/nonfloat tensors, nonfinite values and
unknown names. Uses torch.func.functional_call; input model remains unchanged.
Not a substitute for an architecture-specific intervention adapter.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from tensor_ops import flat_slice


def _edit_stats(original: Any, edited: Any, chunk_elements: int = 65_536):
    """Count in the source dtype; accumulate norms in bounded float64 blocks."""
    import torch

    changed, delta_l2, base_l2 = 0, 0.0, 0.0
    for start in range(0, original.numel(), chunk_elements):
        stop = min(original.numel(), start + chunk_elements)
        a = flat_slice(original, start, stop)
        b = flat_slice(edited, start, stop)
        changed += int(torch.count_nonzero(a != b).item())
        a64, b64 = a.to(torch.float64), b.to(torch.float64)
        delta_l2 = math.hypot(delta_l2, float(torch.linalg.vector_norm(b64 - a64).item()))
        base_l2 = math.hypot(base_l2, float(torch.linalg.vector_norm(a64).item()))
    return {"changed_entries": changed, "total_entries": original.numel(),
            "delta_l2": delta_l2, "relative_l2": delta_l2 / (base_l2 + 1e-12)}


@dataclass(frozen=True)
class TensorEdit:
    name: str
    index: Any
    operation: str  # 'set', 'add', or 'scale'
    value: Any


def _hash_tensor(tensor: Any) -> str:
    import torch
    # Hash the changed tensor; do not hash giant original checkpoint by default.
    x = tensor.detach().to("cpu").contiguous()
    if x.dtype is torch.bfloat16:
        x = x.view(torch.int16)
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def patch_state(model: Any, edits: list[TensorEdit], *, max_changed_fraction: float = 1.0):
    """Return (overrides, manifest) for `functional_call` without editing model.

    Name values are gathered with `remove_duplicate=False` and any named tied
    parameters are rejected because generic alias-preserving edits are risky.
    Only parameters are editable; unchanged parameters and buffers are aliased.
    Use inference without in-place state updates, or an isolated model instance.
    """
    import torch
    if not edits:
        raise ValueError("need at least one explicit edit")
    if not (0 < max_changed_fraction <= 1):
        raise ValueError("max_changed_fraction must be in (0,1]")
    all_params = list(model.named_parameters(remove_duplicate=False))
    by_id: dict[int, list[str]] = {}
    for name, p in all_params:
        by_id.setdefault(id(p), []).append(name)
    shared = {name for names in by_id.values() if len(names) > 1 for name in names}
    # Different Parameter objects may share storage. A conservative check:
    by_storage: dict[tuple[int, str], list[str]] = {}
    for name, p in all_params:
        by_storage.setdefault((p.untyped_storage().data_ptr(), str(p.device)), []).append(name)
    shared.update(name for names in by_storage.values() if len(names) > 1 for name in names)
    originals = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    overrides = dict(originals)
    overrides.update(buffers)
    changed_names = set()
    for edit in edits:
        if edit.name not in originals:
            raise ValueError(f"unknown/unsupported parameter: {edit.name}")
        if edit.name in shared:
            raise ValueError(f"tied/shared parameter {edit.name}; use an architecture-aware adapter")
        if edit.operation not in {"set", "add", "scale"}:
            raise ValueError(f"bad operation: {edit.operation}")
        original = originals[edit.name]
        if not original.is_floating_point() or original.is_sparse or original.is_quantized:
            raise ValueError(f"unsupported non-dense-float tensor: {edit.name}")
        if not torch.isfinite(original).all().item():
            raise ValueError(f"base tensor contains nonfinite values: {edit.name}")
        if edit.name not in changed_names:
            overrides[edit.name] = original.detach().clone()
            changed_names.add(edit.name)
        copy = overrides[edit.name]
        # Assignment is essential: indexed reads can return copies for advanced indexing.
        try:
            target = copy[edit.index]
            v = torch.as_tensor(edit.value, device=copy.device, dtype=copy.dtype)
            if not torch.isfinite(v).all().item():
                raise ValueError("edit value is nonfinite")
            if edit.operation == "set":
                replacement = v.expand_as(target)
            elif edit.operation == "add":
                replacement = target + v
            else:
                replacement = target * v
            if not torch.isfinite(replacement).all().item():
                raise ValueError(f"edit produced nonfinite values: {edit.name}")
            with torch.no_grad():
                copy[edit.index] = replacement
        except (IndexError, RuntimeError, TypeError) as exc:
            raise ValueError(f"invalid edit {edit.name}[{edit.index!r}]: {exc}") from exc
    total_params = sum(p.numel() for p in originals.values())
    records = []
    for name in sorted(changed_names):
        records.append({"name": name, "shape": list(originals[name].shape),
                        **_edit_stats(originals[name], overrides[name]),
                        "edited_tensor_sha256": _hash_tensor(overrides[name])})
    changed = sum(record["changed_entries"] for record in records)
    if changed / total_params > max_changed_fraction:
        raise ValueError(f"edit changes {changed}/{total_params} parameters, exceeds max_changed_fraction")
    return overrides, {"base_untouched": True, "changed_elements": changed,
                       "total_parameters": total_params, "changed_fraction": changed / total_params,
                       "edits": records, "risk_note": "functional_call may update passed tensors if forward mutates parameters"}


def forward_with_patch(model: Any, patched_state: dict[str, Any], *args: Any, **kwargs: Any):
    """Run `model` with supplied params/buffers without replacing its own state.

    Only safe when forward() does not mutate parameters or buffers; if it does,
    these supplied objects may mutate, and model-local state may also change.
    """
    from torch.func import functional_call
    return functional_call(model, patched_state, args, kwargs, strict=True, tie_weights=True)
