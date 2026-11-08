#!/usr/bin/env python3
"""Safe, bounded inspection of safetensors checkpoints. No pickle/model-code execution.

This is a *sampled* integrity survey, not a proof of semantics or full nonfinite scan
unless scanned_elements == numel on each tensor. Results explicitly say so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scan_slice(sliced: Any, shape: tuple[int, ...], limit: int, chunk_elements: int):
    """Yield axis-0 slices evenly dispersed across tensor; no full tensor allocations.

    A row larger than chunk_elements is omitted to prevent surprise allocations.
    """
    if len(shape) == 0:
        return  # handled via sf.get_tensor(name) in audit()
    if any(d == 0 for d in shape):
        return
    row_elems = math.prod(shape[1:]) if len(shape) > 1 else 1
    if row_elems > chunk_elements:
        return
    row_chunk = max(1, chunk_elements // row_elems)
    total_rows = shape[0]
    max_chunks = max(1, limit // (row_chunk * row_elems))
    full_chunks = math.ceil(total_rows / row_chunk)
    if full_chunks <= max_chunks:
        starts = list(range(0, total_rows, row_chunk))
    else:
        # Stratified sample of axis-0 intervals. Not a random/unbiased estimator.
        starts = sorted({round(i * (total_rows - row_chunk) / max(1, max_chunks - 1))
                         for i in range(max_chunks)})
    for start in starts:
        end = min(total_rows, start + row_chunk)
        yield sliced[start:end]


def summarize_tensor(sliced: Any, shape: tuple[int, ...], *, limit: int, chunk_elements: int,
                     scalar_tensor: Any = None) -> dict[str, Any]:
    import torch
    n = math.prod(shape)
    n_seen, n_zero, n_nan, n_inf = 0, 0, 0, 0
    running_sum = running_sq = 0.0
    minv = float('inf')
    maxv = float('-inf')
    pieces = [scalar_tensor] if scalar_tensor is not None else _scan_slice(sliced, shape, limit, chunk_elements)
    for piece in pieces:
        a = piece.detach().reshape(-1)
        if a.is_complex():
            return {"numel": n, "scanned_elements": 0, "complete_scan": False,
                    "note": "complex tensor: numeric sampling not implemented"}
        if not (a.is_floating_point() or a.dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                                                      torch.uint8, torch.bool)):
            return {"numel": n, "scanned_elements": 0, "complete_scan": False,
                    "note": f"statistics unsupported for dtype {a.dtype}"}
        finite = torch.isfinite(a)
        n_seen += a.numel()
        n_nan += int(torch.isnan(a).sum().item()) if a.is_floating_point() else 0
        n_inf += int(torch.isinf(a).sum().item()) if a.is_floating_point() else 0
        n_zero += int((a == 0).sum().item())
        good = a[finite].to(torch.float64)
        if good.numel():
            running_sum += good.sum().item()
            running_sq += good.square().sum().item()
            minv = min(minv, good.min().item())
            maxv = max(maxv, good.max().item())
    numfinite = n_seen - n_nan - n_inf
    if n_seen == 0:
        return {"numel": n, "scanned_elements": 0, "complete_scan": False,
                "note": "too large per row/slice to sample under current chunk limit, or empty tensor"}
    mean = running_sum / numfinite if numfinite else None
    var = max(0.0, running_sq / numfinite - mean * mean) if numfinite else None
    return {"numel": n, "scanned_elements": n_seen, "complete_scan": n_seen == n,
            "zero_fraction_of_sample": round(n_zero / n_seen, 8),
            "nan_in_sample": n_nan, "inf_in_sample": n_inf,
            "finite_min": minv if numfinite else None, "finite_max": maxv if numfinite else None,
            "finite_mean": mean, "finite_std": math.sqrt(var) if var is not None else None,
            "sampled_l2": math.sqrt(running_sq)}


def audit(path: Path, *, limit: int = 262_144, chunk_elements: int = 65_536,
          max_tensors: int = 0) -> dict[str, Any]:
    from safetensors import safe_open
    if path.suffix != ".safetensors":
        raise ValueError("Only .safetensors is supported; never auto-load pickle checkpoints")
    if not path.is_file():
        raise FileNotFoundError(path)
    if limit < chunk_elements or chunk_elements < 1:
        raise ValueError("--limit must be >= --chunk-elements >= 1")
    report: dict[str, Any] = {"file": str(path.resolve()), "bytes": path.stat().st_size,
                              "sha256": sha256_file(path), "sample_design": "axis0_stratified",
                              "sample_limit_per_tensor": limit, "chunk_elements": chunk_elements,
                              "tensors": [], "warnings": []}
    total_params = 0
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        report["tensor_count"] = len(keys)
        report["metadata"] = sf.metadata()
        for name in (keys[:max_tensors] if max_tensors else keys):
            s = sf.get_slice(name)
            shape = tuple(s.get_shape())
            total_params += math.prod(shape)
            stat = summarize_tensor(s, shape, limit=limit, chunk_elements=chunk_elements,
                                    scalar_tensor=sf.get_tensor(name) if not shape else None)
            stat.update({"name": name, "shape": list(shape), "dtype": s.get_dtype()})
            report["tensors"].append(stat)
            if stat.get("nan_in_sample", 0) or stat.get("inf_in_sample", 0):
                report["warnings"].append(f"Nonfinite sample values: {name}")
            if not stat["complete_scan"]:
                report["warnings"].append(f"Incomplete scan, NOT guaranteed finite: {name}")
    report["inspected_parameters"] = total_params
    report["skipped_tensor_count"] = max(0, report["tensor_count"] - len(report["tensors"]))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path, help="local .safetensors file (no code execution)")
    ap.add_argument("--limit", type=int, default=262_144, help="max sampled elements per tensor")
    ap.add_argument("--chunk-elements", type=int, default=65_536, help="max elements loaded per slice")
    ap.add_argument("--max-tensors", type=int, default=0, help="process at most this many tensors, 0=all")
    ap.add_argument("--output", type=Path, help="write JSON report to file instead of stdout")
    args = ap.parse_args()
    try:
        report = audit(args.checkpoint, limit=args.limit, chunk_elements=args.chunk_elements,
                       max_tensors=args.max_tensors)
    except Exception as exc:
        ap.exit(2, f"checkpoint audit failed: {exc}\n")
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Saved audit: {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
