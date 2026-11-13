"""Config loading (YAML + optional smoke overrides) and path resolution."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.io import REPO_ROOT, ensure_dir, read_yaml

DEFAULT_CONFIG = REPO_ROOT / "task2_vlm_grounding" / "configs" / "main_196tok.yaml"
VQA_TYPES = ("yesno", "number", "other")
VQA_TYPE_NAMES = {"yes/no": "yesno", "number": "number", "other": "other"}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, smoke: bool = False) -> dict:
    cfg = read_yaml(path or DEFAULT_CONFIG)
    smoke_block = cfg.pop("smoke", {}) or {}
    if smoke:
        cfg = deep_merge(cfg, smoke_block)
    cfg["smoke"] = bool(smoke)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    t = cfg["training"]
    assert t["microbatch"] * t["grad_accum"] == 16, "effective batch must be 16"
    assert len(t["stages"]) == 3
    probs = [sum(s["mix"].values()) for s in t["stages"].values()]
    for p in probs:
        assert abs(p - 1.0) < 1e-9, f"stage mixture must sum to 1, got {p}"


def stage_list(cfg: dict) -> list[dict]:
    """Ordered stage dicts with index, name, start/end global update."""
    stages, start = [], 0
    for i, key in enumerate(["stage1", "stage2", "stage3"]):
        s = dict(cfg["training"]["stages"][key])
        s["index"] = i
        s["key"] = key
        s["start"] = start
        s["end"] = start + int(s["updates"])
        start = s["end"]
        stages.append(s)
    return stages


def total_updates(cfg: dict) -> int:
    return stage_list(cfg)[-1]["end"]


def dev_eval_steps(cfg: dict) -> list[int]:
    total = total_updates(cfg)
    every = int(cfg["eval"]["dev_every"])
    steps = set(range(0, total + 1, every))
    steps.update(int(s) for s in cfg["eval"].get("dev_extra_steps", []))
    steps.update(s["end"] for s in stage_list(cfg))
    return sorted(s for s in steps if 0 <= s <= total)


@dataclass
class Paths:
    root: Path
    raw: Path
    splits: Path
    features: Path
    checkpoints: Path
    results: Path
    figures: Path

    @property
    def dev_dir(self) -> Path:
        return self.results / "dev"

    @property
    def final_dir(self) -> Path:
        return self.results / "final"

    @property
    def logs_dir(self) -> Path:
        return self.results / "training_logs"

    @property
    def qual_dir(self) -> Path:
        return self.results / "qualitative"


def get_paths(cfg: dict, create: bool = True) -> Paths:
    p = cfg["paths"]
    run = cfg["run_name"]
    out = REPO_ROOT / p["outputs_dir"]
    paths = Paths(
        root=REPO_ROOT,
        raw=REPO_ROOT / p["raw_dir"],
        splits=REPO_ROOT / p["splits_dir"],
        features=out / "features" / run,
        checkpoints=out / "checkpoints" / run,
        results=REPO_ROOT / p["results_dir"] / run,
        figures=REPO_ROOT / p["figures_dir"] / run,
    )
    if create:
        for d in (paths.splits, paths.features, paths.checkpoints, paths.dev_dir, paths.final_dir,
                  paths.logs_dir, paths.qual_dir, paths.figures):
            ensure_dir(d)
    return paths


def cfg_get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
