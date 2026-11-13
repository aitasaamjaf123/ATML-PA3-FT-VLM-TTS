"""Paired / question-level bootstrap statistics.

Task 1/2: bootstrap_indices, percentile_ci, bootstrap_stat, bootstrap_diff, bootstrap_many
Task 3:   bootstrap_mean_ci, paired_bootstrap (dict outputs with sample counts and `excludes_zero`)
All use the same index matrix convention [n_resamples, n_pairs], so resamples are matched across conditions.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def bootstrap_indices(n_pairs: int, n_resamples: int = 1000, seed: int = 6304) -> np.ndarray:
    """[n_resamples, n_pairs] pair indices drawn with replacement.

    The *same* index matrix must be reused for every condition/metric so that
    resamples are matched on pair IDs when computing differences.
    """
    if n_pairs <= 0:
        raise ValueError("Cannot bootstrap zero pairs")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_pairs, size=(n_resamples, n_pairs))


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_stat(values: np.ndarray, idx: np.ndarray) -> dict:
    """Mean of a per-pair statistic with a percentile 95% CI."""
    values = np.asarray(values, dtype=np.float64)
    samples = values[idx].mean(axis=1)
    lo, hi = percentile_ci(samples)
    return {"mean": float(values.mean()), "ci_low": lo, "ci_high": hi, "n_pairs": int(len(values))}


def bootstrap_diff(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> dict:
    """Mean of (a - b) per pair with a matched-resample percentile 95% CI."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Pair-aligned arrays required, got {a.shape} vs {b.shape}")
    return bootstrap_stat(a - b, idx)


def bootstrap_many(
    per_pair: Mapping[str, Sequence[float]],
    diffs: Sequence[tuple[str, str]] = (),
    n_resamples: int = 4000,
    seed: int = 6304,
) -> dict:
    """Bootstrap several pair-aligned statistics (and their differences) jointly."""
    names = list(per_pair)
    n = len(per_pair[names[0]])
    for k in names:
        if len(per_pair[k]) != n:
            raise ValueError("All statistics must be aligned over the same pairs")
    idx = bootstrap_indices(n, n_resamples, seed)
    out = {k: bootstrap_stat(np.asarray(v), idx) for k, v in per_pair.items()}
    for a, b in diffs:
        out[f"{a}-{b}"] = bootstrap_diff(np.asarray(per_pair[a]), np.asarray(per_pair[b]), idx)
    return out


# ----------------------------------------------------------------------------------------------
# Task 3
# ----------------------------------------------------------------------------------------------
def bootstrap_mean_ci(x, n_boot: int = 1000, seed: int = 6304, alpha: float = 0.05, idx=None) -> dict:
    x = np.asarray(x, dtype=float)
    if idx is None:
        idx = bootstrap_indices(len(x), n_boot, seed)
    boots = x[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(x)), "n_boot": int(idx.shape[0])}


def paired_bootstrap(a, b, n_boot: int = 1000, seed: int = 6304, alpha: float = 0.05, idx=None) -> dict:
    """Paired question-level bootstrap of mean(a - b); `a`, `b` are per-question values (already
    averaged over rollout orders)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = a - b
    if idx is None:
        idx = bootstrap_indices(len(d), n_boot, seed)
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n": int(len(d)),
        "n_boot": int(idx.shape[0]),
    }
