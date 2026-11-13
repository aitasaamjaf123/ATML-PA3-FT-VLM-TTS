"""File-IO helpers (json / jsonl / csv / yaml / tables), hashing and config merging.

This module merges the helpers of Tasks 1, 2 and 3. Both naming conventions are kept:
  Task 1: save_json / load_json / load_yaml / save_yaml / write_rows_csv / CsvLogger / deep_update / apply_overrides / sha256_file
  Task 2: read_json / write_json / read_jsonl / write_jsonl / append_jsonl / write_csv / read_csv / read_yaml / REPO_ROOT
  Task 3: save_jsonl / load_jsonl (aliases of the jsonl pair) / save_csv / save_table / load_table / df_to_markdown / write_text
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------------------------------
def _json_default(o):
    """Strict serialiser (Task 2): numpy scalars/arrays and Paths, anything else is an error."""
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def _json_default_lenient(o):
    """Lenient serialiser (Tasks 1 and 3): numpy -> numbers (NaN -> null), sets -> sorted lists, else str()."""
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            v = float(o)
            return None if math.isnan(v) else v
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        return _json_default_lenient(o)


def read_json(path: os.PathLike | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json(path):  # Task 1 name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: os.PathLike | str, indent: int | None = 2) -> None:  # Task 2
    ensure_dir(Path(path).parent)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def save_json(obj, path, indent: int | None = 2) -> None:  # Tasks 1 and 3 (lenient: numpy-safe, never fails on odd types)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default_lenient, ensure_ascii=False)
    os.replace(tmp, path)


def read_jsonl(path: os.PathLike | str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(rows: Iterable[dict], path: os.PathLike | str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def append_jsonl(row: dict, path: os.PathLike | str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


load_jsonl = read_jsonl   # Task 3 names
save_jsonl = write_jsonl


# ----------------------------------------------------------------------------------------------
# YAML / config
# ----------------------------------------------------------------------------------------------
def read_yaml(path: os.PathLike | str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml(path):  # Task 1 name
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(obj, path) -> None:
    import yaml

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
    import yaml

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


# ----------------------------------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------------------------------
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


def write_rows_csv(path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:  # Task 1 (path first)
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


def write_csv(rows: Sequence[dict], path: os.PathLike | str, fieldnames: Sequence[str] | None = None) -> None:  # Task 2 (rows first)
    ensure_dir(Path(path).parent)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def read_csv(path: os.PathLike | str) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------------------------
# Task 3: text + DataFrame helpers (pandas is only needed when these are called)
# ----------------------------------------------------------------------------------------------
def write_text(text: str, path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p


def save_csv(df: "pd.DataFrame", path, index: bool = False) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index)
    return p


def save_table(df: "pd.DataFrame", stem) -> Path:
    """Save a (large) table as parquet when pyarrow is present, else csv.gz. `stem` has no suffix."""
    stem = str(stem)
    ensure_dir(Path(stem).parent)
    pq = Path(stem + ".parquet")
    try:
        df.to_parquet(pq, index=False)
        return pq
    except Exception:
        if pq.exists():
            pq.unlink()
        gz = Path(stem + ".csv.gz")
        df.to_csv(gz, index=False)
        return gz


def load_table(stem) -> Optional["pd.DataFrame"]:
    import pandas as pd

    stem = str(stem)
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv), (".csv", pd.read_csv)):
        p = Path(stem + suffix)
        if p.exists():
            return reader(p)
    return None


def df_to_markdown(df: "pd.DataFrame", floatfmt: str = "{:.4f}", index: bool = False) -> str:
    """Minimal markdown table writer (no `tabulate` dependency)."""
    import numpy as np

    if df is None or len(df) == 0:
        return "_(empty)_\n"
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        cells = []
        for v in row.tolist():
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if (isinstance(v, float) and math.isnan(v)) else floatfmt.format(v))
            else:
                cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
