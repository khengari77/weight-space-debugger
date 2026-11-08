"""Statistical helpers for paired controlled behavioral evaluation.

CIs are exploratory and assume independent *items/episodes*, NOT independent
measurements from repeated tokens/time steps within the same item.
"""
from __future__ import annotations

import math
import random
from typing import Sequence


def paired_bootstrap_difference(before: Sequence[float], after: Sequence[float], *,
                                n_resamples: int = 2000, seed: int = 7,
                                confidence: float = 0.95) -> dict[str, float | int]:
    if len(before) != len(after) or len(before) < 2:
        raise ValueError("before and after need >=2 matched observations")
    if n_resamples < 100 or not 0.5 < confidence < 1.0:
        raise ValueError("n_resamples >= 100 and confidence in (0.5,1) required")
    differences = [float(a) - float(b) for a, b in zip(after, before)]
    if not all(math.isfinite(v) for v in differences):
        raise ValueError("all paired differences must be finite")
    n = len(differences)
    rng = random.Random(seed)
    resampled = [sum(differences[rng.randrange(n)] for _ in range(n)) / n
                 for _ in range(n_resamples)]
    resampled.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = resampled[max(0, int(alpha * n_resamples))]
    hi = resampled[min(n_resamples - 1, int((1.0 - alpha) * n_resamples))]
    return {"n_pairs": n, "mean_difference_after_minus_before": sum(differences) / n,
            "confidence": confidence, "percentile_bootstrap_ci_low": lo,
            "percentile_bootstrap_ci_high": hi, "bootstrap_resamples": n_resamples,
            "caution": "percentile bootstrap; exploratory, assumes independent matched units"}
