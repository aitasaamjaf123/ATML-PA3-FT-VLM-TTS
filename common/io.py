"""JSON / YAML / CSV helpers, hashing, config merging."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

import yaml


def save_json(obj, path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=str)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: dict, overrides: Optional[List[str]]) -> dict:
    """`a.b.c=value` overrides; value parsed as YAML (so 3, 1e-4, [1,2], true work)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        val = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class CsvLogger:
    """Append-only CSV with a fixed schema; flushes every row (safe against Kaggle kills)."""

    def __init__(self, path, fieldnames: Iterable[str], append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        exists = self.path.exists() and self.path.stat().st_size > 0
        self._f = open(self.path, "a" if append else "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames, extrasaction="ignore")
        if not (append and exists):
            self._w.writeheader()
            self._f.flush()

    def write(self, row: dict) -> None:
        self._w.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def write_rows_csv(path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
