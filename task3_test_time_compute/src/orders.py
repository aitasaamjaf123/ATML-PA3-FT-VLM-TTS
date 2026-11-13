"""Seeded random rollout orders with nested prefixes (shared by all methods and stopping policies)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from common.io import load_json, save_json
from common.seed import stable_hash


def question_orders(qid: str, n_orders: int = 10, n_rollouts: int = 32, seed: int = 6304) -> np.ndarray:
    """[n_orders, n_rollouts] permutations of rollout indices for one question.

    The generator depends only on (seed, qid), and permutations are drawn sequentially, so using
    fewer orders (smoke test) yields a prefix of the full ten orders."""
    rng = np.random.default_rng([int(seed), stable_hash(str(qid))])
    return np.stack([rng.permutation(n_rollouts) for _ in range(n_orders)]).astype(np.int16)


def make_orders(qids: Iterable[str], n_orders: int = 10, n_rollouts: int = 32, seed: int = 6304) -> Dict[str, np.ndarray]:
    return {str(q): question_orders(str(q), n_orders, n_rollouts, seed) for q in qids}


def prefix(perm: np.ndarray, n: int) -> np.ndarray:
    """Nested prefix of a permutation (first n candidates revealed)."""
    return perm[:n]


def save_orders(orders: Dict[str, np.ndarray], path) -> None:
    save_json({q: o.tolist() for q, o in orders.items()}, path, indent=None)


def load_orders(path) -> Dict[str, np.ndarray]:
    return {q: np.asarray(o, dtype=np.int16) for q, o in load_json(Path(path)).items()}
