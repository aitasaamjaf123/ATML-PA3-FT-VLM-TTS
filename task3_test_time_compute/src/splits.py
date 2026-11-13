"""Deterministic dev / held-out / live question splits (seed 6304)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from common.io import ensure_dir, load_json, save_json


def make_splits(sample_ids: Iterable, seed: int = 6304, n_dev: int = 200, n_live: int = 50) -> dict:
    """Sort IDs lexicographically (as strings), permute with default_rng(seed); first n_dev -> dev,
    the rest -> held-out evaluation; first n_live of the evaluation permutation -> live subset."""
    ids = sorted({str(x) for x in sample_ids})
    perm = np.random.default_rng(seed).permutation(len(ids))
    ordered = [ids[i] for i in perm]
    dev = ordered[:n_dev]
    evaluation = ordered[n_dev:]
    live = evaluation[:n_live]
    assert not set(dev) & set(evaluation), "dev and held-out splits overlap"
    assert len(dev) + len(evaluation) == len(ids)
    return {"seed": seed, "n_total": len(ids), "dev": dev, "eval": evaluation, "live": live}


def smoke_subset(splits: dict, n_dev: int, n_eval: int, n_live: int) -> dict:
    """Take prefixes of the full permutation-ordered splits (so smoke IDs are a subset of the full run)."""
    dev = splits["dev"][:n_dev]
    evaluation = splits["eval"][:n_eval]
    live = evaluation[: min(n_live, len(evaluation))]
    assert not set(dev) & set(evaluation)
    return {"seed": splits["seed"], "n_total": splits["n_total"], "dev": dev, "eval": evaluation, "live": live, "smoke": True}


def save_splits(splits: dict, directory, dataset: str) -> None:
    d = ensure_dir(directory)
    for k in ("dev", "eval", "live"):
        save_json(splits[k], Path(d) / f"{dataset}_{k}_ids.json")
    save_json({k: v for k, v in splits.items() if k not in ("dev", "eval", "live")}
              | {"counts": {k: len(splits[k]) for k in ("dev", "eval", "live")}}, Path(d) / f"{dataset}_meta.json")


def load_splits(directory, dataset: str) -> dict:
    d = Path(directory)
    out = {k: load_json(d / f"{dataset}_{k}_ids.json") for k in ("dev", "eval", "live")}
    out.update(load_json(d / f"{dataset}_meta.json"))
    return out
