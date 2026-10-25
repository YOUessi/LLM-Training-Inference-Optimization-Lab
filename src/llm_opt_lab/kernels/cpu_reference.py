"""Small standard-library correctness oracles; not performance baselines."""

from __future__ import annotations

import math
from collections.abc import Sequence


def rmsnorm_row(values: Sequence[float], eps: float = 1e-6) -> list[float]:
    """Root-mean-square normalization (no learned weight), fp64 reference."""
    if not values:
        raise ValueError("RMSNorm requires at least one element")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    xs = [float(v) for v in values]
    if not all(math.isfinite(x) for x in xs):
        raise ValueError("values must be finite")
    inv_rms = 1.0 / math.sqrt(math.fsum(x * x for x in xs) / len(xs) + eps)
    return [x * inv_rms for x in xs]


def softmax_row(values: Sequence[float]) -> list[float]:
    """Stable softmax reference; subtract max to avoid exp overflow."""
    if not values:
        raise ValueError("Softmax requires at least one element")
    xs = [float(v) for v in values]
    if not all(math.isfinite(x) for x in xs):
        raise ValueError("values must be finite")
    m = max(xs)
    exp_values = [math.exp(x - m) for x in xs]
    denom = math.fsum(exp_values)
    return [v / denom for v in exp_values]
