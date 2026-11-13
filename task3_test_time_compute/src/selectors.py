"""Answer selection rules: FirstAns, self-consistency, highest-score (Best-of-N), rank-weighted consensus.

Every rule works on the answer-group ids `g` of the revealed candidates IN CURRENT ORDER. Ties
(scores or votes) are resolved by the earliest supporting rollout in the current order.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import rankdata

TOL = 1e-9


def vote_stats(g: np.ndarray):
    """(unique group ids, counts, first position of each group in the current order)."""
    uniq, first, counts = np.unique(g, return_index=True, return_counts=True)
    return uniq, counts, first


def vote_winner(g: np.ndarray) -> int:
    """Self-consistency: one vote per rollout; ties -> group with the earliest supporting rollout."""
    uniq, counts, first = vote_stats(g)
    order = np.lexsort((first, -counts))
    return int(uniq[order[0]])


def unique_leader(g: np.ndarray):
    """Returns (winner gid, count, is_unique_leader). `is_unique_leader` is False on a count tie."""
    uniq, counts, first = vote_stats(g)
    order = np.lexsort((first, -counts))
    top = counts[order[0]]
    return int(uniq[order[0]]), int(top), bool((counts == top).sum() == 1)


def select_top(g: np.ndarray, score: np.ndarray) -> int:
    """Highest-score (Best-of-N): np.argmax returns the first (= earliest in order) maximum."""
    return int(g[int(np.argmax(score))])


def rank_weights(score: np.ndarray) -> np.ndarray:
    """w_i = (N - r_i)/(N - 1) with descending AVERAGE ranks r_i (rank 1 = best). Requires N > 1."""
    n = len(score)
    r = rankdata(-np.asarray(score, dtype=np.float64), method="average")
    return (n - r) / (n - 1)


def select_rwc(g: np.ndarray, score: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """Rank-weighted consensus. Returns (winner gid, group vote weights W, group ids).
    N = 1 returns the only candidate."""
    if len(g) == 1:
        return int(g[0]), np.array([1.0]), np.array([g[0]])
    w = rank_weights(score)
    uniq, inv = np.unique(g, return_inverse=True)
    W = np.bincount(inv.ravel(), weights=w, minlength=len(uniq))
    first = np.unique(g, return_index=True)[1]
    cand = np.where(W >= W.max() - TOL)[0]
    best = cand[np.argmin(first[cand])]
    return int(uniq[best]), W, uniq


def rwc_margin(W: np.ndarray) -> float:
    """(largest - second-largest group vote) / total vote weight; runner-up = 0 with a single group."""
    W = np.sort(np.asarray(W, dtype=np.float64))[::-1]
    second = W[1] if len(W) > 1 else 0.0
    total = W.sum()
    return float((W[0] - second) / total) if total > 0 else 0.0


# Method registry: name -> (score key or None, aggregation)
METHODS_BOTH = {
    "first": (None, "first"),
    "selfcons": (None, "vote"),
    "logp_top": ("logp", "top"),
    "logp_rwc": ("logp", "rwc"),
    "selfcert_top": ("selfcert", "top"),
    "selfcert_rwc": ("selfcert", "rwc"),
    "shortest": ("shortest", "top"),
}
METHODS_VISION = {
    "vision_top": ("vision", "top"),
    "vision_rwc": ("vision", "rwc"),
    "visconf_top": ("visconf", "top"),
    "visconf_rwc": ("visconf", "rwc"),
    "tokenfusion_top": ("tokenfusion", "top"),
    "tokenfusion_rwc": ("tokenfusion", "rwc"),
}


def method_specs(vision: bool) -> dict:
    specs = dict(METHODS_BOTH)
    if vision:
        specs.update(METHODS_VISION)
    return specs


def apply_method(agg: str, g: np.ndarray, score) -> int:
    if len(g) == 1:
        return int(g[0])  # at N = 1 every rule returns the only candidate
    if agg == "first":
        return int(g[0])
    if agg == "vote":
        return vote_winner(g)
    if agg == "top":
        return select_top(g, score)
    if agg == "rwc":
        return select_rwc(g, score)[0]
    raise ValueError(agg)
