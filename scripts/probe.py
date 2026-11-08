"""Bounded, read-only PyTorch activation/gradient telemetry.

Example:
    with ModuleProbe(model, patterns=["0", "2"], max_events=5) as p:
        logits = model(inputs)
    print(p.summary())

This cannot assign semantics to units; it produces evidence for hypotheses.
"""
from __future__ import annotations

import fnmatch
import math
from typing import Any, Iterable

from tensor_ops import flat_slice


def first_tensor(obj: Any):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            result = first_tensor(item)
            if result is not None:
                return result
    if isinstance(obj, dict):
        for item in obj.values():
            result = first_tensor(item)
            if result is not None:
                return result
    return None


def tensor_summary(tensor: Any, sample_elements: int = 8192) -> dict[str, Any]:
    import torch
    if sample_elements < 1:
        raise ValueError("sample_elements must be >= 1")
    n = tensor.numel()
    if n == 0:
        return {"shape": list(tensor.shape), "numel": 0}
    step = max(1, math.ceil(n / sample_elements))
    x = flat_slice(tensor, 0, n, step)[:sample_elements]
    if x.is_complex():
        return {"shape": list(tensor.shape), "numel": n, "note": "complex stats unsupported"}
    float_x = x if x.dtype == torch.float64 else x.float()
    finite = torch.isfinite(float_x)
    good = float_x[finite]
    return {"shape": list(tensor.shape), "numel": n, "sampled": len(x),
            "dtype": str(tensor.dtype), "nonfinite_sample": int((~finite).sum().item()),
            "zero_fraction_sample": float((x == 0).float().mean().item()),
            "mean_sample": float(good.mean().item()) if good.numel() else None,
            "std_sample": float(good.std(unbiased=False).item()) if good.numel() else None,
            "l2_sample": float(good.norm().item()) if good.numel() else None,
            "min_sample": float(good.min().item()) if good.numel() else None,
            "max_sample": float(good.max().item()) if good.numel() else None}


class ModuleProbe:
    def __init__(self, model: Any, patterns: Iterable[str], *, max_events: int = 5,
                 sample_elements: int = 8192, gradients: bool = False):
        if max_events < 1 or sample_elements < 1:
            raise ValueError("max_events and sample_elements must be >= 1")
        self.model = model
        self.patterns = tuple(patterns)
        self.max_events = max_events
        self.sample_elements = sample_elements
        self.gradients = gradients
        self.records: dict[str, dict[str, list[dict]]] = {}
        self._handles: list[Any] = []

    def __enter__(self):
        if self._handles:
            raise RuntimeError("probe already active")
        for name, mod in self.model.named_modules():
            if name and any(fnmatch.fnmatchcase(name, p) for p in self.patterns):
                self.records[name] = {"activation": [], "gradient": []}
                self._handles.append(mod.register_forward_hook(self._mk_hook(name)))
        if not self._handles:
            raise ValueError(f"no modules matched {self.patterns!r}; check model.named_modules()")
        return self

    def _mk_hook(self, name: str):
        def hook(_module, _args, out):
            import torch
            t = first_tensor(out)
            if t is None:
                return None
            target = self.records[name]
            if len(target["activation"]) < self.max_events:
                with torch.no_grad():
                    target["activation"].append(tensor_summary(t, self.sample_elements))
            if self.gradients and t.requires_grad and len(target["gradient"]) < self.max_events:
                def grad_hook(g):
                    with torch.no_grad():
                        if len(target["gradient"]) < self.max_events:
                            target["gradient"].append(tensor_summary(g, self.sample_elements))
                t.register_hook(grad_hook)
            return None  # never alters the activation
        return hook

    def summary(self) -> dict[str, Any]:
        return {"sample_design": "stride_sample_first_flat_tensor_per_module",
                "warning": "sampling is partial and may miss rare/nonfinite events; not a causal explanation",
                "modules": self.records}

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False
