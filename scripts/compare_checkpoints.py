#!/usr/bin/env python3
"""Sampled aligned-parameter diff of two local safetensors checkpoints.

Important: only compare aligned architectures/checkpoints with matching tensor
semantics (e.g., shared ancestor). A large delta does not imply causal relevance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from checkpoint_audit import sha256_file


def compare(base: Path, other: Path, *, limit: int = 262_144, chunk_elements: int = 65_536):
    import torch
    from safetensors import safe_open
    if base.suffix != ".safetensors" or other.suffix != ".safetensors":
        raise ValueError("only .safetensors is supported")
    if limit < chunk_elements or chunk_elements < 1:
        raise ValueError("limit must be >= chunk_elements >= 1")
    report = {"base": str(base.resolve()), "other": str(other.resolve()),
              "base_sha256": sha256_file(base), "other_sha256": sha256_file(other),
              "warning": "coordinate diffs meaningful only when tensors are aligned; sampled stats are not full scans",
              "tensors": [], "only_base": [], "only_other": [], "shape_mismatch": []}
    with safe_open(str(base), framework="pt", device="cpu") as sf1, safe_open(str(other), framework="pt", device="cpu") as sf2:
        k1, k2 = set(sf1.keys()), set(sf2.keys())
        report["only_base"] = sorted(k1 - k2)
        report["only_other"] = sorted(k2 - k1)
        for key in sorted(k1 & k2):
            s1, s2 = sf1.get_slice(key), sf2.get_slice(key)
            shp1, shp2 = tuple(s1.get_shape()), tuple(s2.get_shape())
            dtype1, dtype2 = s1.get_dtype(), s2.get_dtype()
            if shp1 != shp2:
                report["shape_mismatch"].append({"name": key, "base": shp1, "other": shp2})
                continue
            if not shp1:
                a = sf1.get_tensor(key).double()
                b = sf2.get_tensor(key).double()
                finite = bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())
                report["tensors"].append({"name": key, "shape": [],
                    "dtype_base": dtype1, "dtype_other": dtype2,
                    "numel": 1, "scanned_elements": 1, "complete_scan": True,
                    "all_sampled_finite": finite,
                    "different_sampled_entries": int((a != b).item()),
                    "delta_l2_sample": abs((b-a).item()) if finite else None,
                    "relative_l2_sample": abs((b-a).item())/(abs(a.item())+1e-12) if finite else None})
                continue
            if any(d == 0 for d in shp1):
                report["tensors"].append({"name": key, "shape": shp1, "note": "empty tensor"})
                continue
            # Use the same axis-0 segment locations for both checkpoints.
            row_elems = math.prod(shp1[1:]) if len(shp1) > 1 else 1
            if row_elems > chunk_elements:
                report["tensors"].append({"name": key, "shape": shp1, "note": "row too large for chunk budget"})
                continue
            n_rows = max(1, chunk_elements // row_elems)
            max_chunks = max(1, limit // (n_rows * row_elems))
            full_chunks = math.ceil(shp1[0] / n_rows)
            if full_chunks <= max_chunks:
                starts = range(0, shp1[0], n_rows)
            else:
                starts = sorted({round(i * (shp1[0] - n_rows) / max(1, max_chunks - 1))
                                 for i in range(max_chunks)})
            seen = num_different = 0
            sq = base_sq = 0.0
            finite = True
            for begin in starts:
                end = min(shp1[0], begin + n_rows)
                a = s1[begin:end].reshape(-1).to(torch.float64)
                b = s2[begin:end].reshape(-1).to(torch.float64)
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    finite = False
                    break
                d = b - a
                seen += a.numel()
                num_different += int(torch.count_nonzero(d).item())
                sq += d.square().sum().item()
                base_sq += a.square().sum().item()
            report["tensors"].append({"name": key, "shape": shp1,
                   "dtype_base": dtype1, "dtype_other": dtype2,
                   "numel": math.prod(shp1), "scanned_elements": seen,
                   "complete_scan": seen == math.prod(shp1),
                   "all_sampled_finite": finite,
                   "different_sampled_entries": num_different,
                   "delta_l2_sample": math.sqrt(sq) if finite else None,
                   "relative_l2_sample": math.sqrt(sq) / (math.sqrt(base_sq) + 1e-12) if finite else None})
    report["tensors"].sort(key=lambda r: r.get("relative_l2_sample") or 0, reverse=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", type=Path)
    ap.add_argument("other", type=Path)
    ap.add_argument("--limit", type=int, default=262_144)
    ap.add_argument("--chunk-elements", type=int, default=65_536)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    try:
        report = compare(args.base, args.other, limit=args.limit, chunk_elements=args.chunk_elements)
    except Exception as exc:
        ap.exit(2, f"comparison failed: {exc}\n")
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf8")
        print(f"Saved comparison: {args.output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
