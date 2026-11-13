"""Pair-level aggregation and paired bootstrap (resampling complete pairs) for final results."""
from __future__ import annotations

import zlib
from collections import defaultdict
from typing import Sequence

import numpy as np

from common.stats import bootstrap_diff, bootstrap_indices, bootstrap_stat
from task2_vlm_grounding.src.vqa_eval import CONDITIONS

PAIR_METRICS = ("acc", "exact", "pair_acc", "identical", "empty", "nll", "follow", "unsupported")
DIFF_METRICS = ("acc", "exact", "pair_acc", "nll")


def _crc(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))  # stable across processes (unlike hash())


def pair_table(rows: Sequence[dict]) -> dict:
    """rows -> {suite: {pair_ids, qtypes, cond: {condition: {metric: np.ndarray[n_pairs]}}, gen_ms}}.

    Pairs are sorted by pair_id so arrays are aligned across conditions (matched pair IDs).
    Each pair must have exactly two examples in every condition present.
    """
    grouped: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        grouped[r["suite"]][r["condition"]][r["pair_id"]].append(r)
    out: dict = {}
    for suite, conds in grouped.items():
        pair_ids = sorted(next(iter(conds.values())).keys())
        qtype_of = {pid: next(iter(conds.values()))[pid][0]["qtype"] for pid in pair_ids}
        suite_out = {"pair_ids": pair_ids, "qtypes": [qtype_of[p] for p in pair_ids], "cond": {}, "gen_ms": {}}
        for cond, by_pair in conds.items():
            assert sorted(by_pair.keys()) == pair_ids, f"{suite}/{cond}: pair ids differ across conditions"
            cols = {m: [] for m in PAIR_METRICS}
            gen = []
            for pid in pair_ids:
                two = sorted(by_pair[pid], key=lambda r: r["idx"])
                assert len(two) == 2, f"{pid} has {len(two)} examples"
                cols["acc"].append(np.mean([r["score"] for r in two]))
                cols["exact"].append(np.mean([r["exact"] for r in two]))
                cols["pair_acc"].append(float(all(r["exact"] == 1.0 for r in two)))
                cols["identical"].append(float(two[0]["pred_norm"] == two[1]["pred_norm"]))
                cols["empty"].append(np.mean([r["empty"] for r in two]))
                cols["nll"].append(np.mean([r["nll"] for r in two]))
                cols["follow"].append(np.mean([r["partner_follow"] for r in two]))
                cols["unsupported"].append(np.mean([1.0 - r["exact"] - r["partner_follow"] for r in two]))
                gen.extend(r["gen_ms"] for r in two)
            suite_out["cond"][cond] = {m: np.asarray(v, dtype=np.float64) for m, v in cols.items()}
            suite_out["gen_ms"][cond] = float(np.mean(gen))
        out[suite] = suite_out
    return out


def summarize(rows: Sequence[dict], n_resamples: int = 4000, seed: int = 6304) -> dict:
    """Full result structure plus flat table rows (suite, condition, subset, metric, ...).

    Subsets: 'overall' and every question type. Within a (suite, subset) the *same* resample index
    matrix is applied to every condition, so differences use matched pair IDs.
    """
    tables = pair_table(rows)
    flat: list[dict] = []
    for suite, t in tables.items():
        qtypes = np.asarray(t["qtypes"])
        subsets = {"overall": np.ones(len(qtypes), dtype=bool)}
        for q in sorted(set(t["qtypes"])):
            subsets[q] = qtypes == q
        conds = [c for c in CONDITIONS if c in t["cond"]]
        for sub, mask in subsets.items():
            n = int(mask.sum())
            if n == 0:
                continue
            idx = bootstrap_indices(n, n_resamples, seed + _crc(f"{suite}/{sub}"))
            for cond in conds:
                for m in PAIR_METRICS:
                    s = bootstrap_stat(t["cond"][cond][m][mask], idx)
                    flat.append({"suite": suite, "condition": cond, "subset": sub, "metric": m,
                                 "n_pairs": n, "mean": s["mean"], "ci_low": s["ci_low"], "ci_high": s["ci_high"]})
                flat.append({"suite": suite, "condition": cond, "subset": sub, "metric": "gen_ms_per_answer",
                             "n_pairs": n, "mean": t["gen_ms"][cond], "ci_low": "", "ci_high": ""})
            if "correct" in conds:
                for other in conds:
                    if other == "correct":
                        continue
                    for m in DIFF_METRICS:
                        a, b = t["cond"]["correct"][m][mask], t["cond"][other][m][mask]
                        # advantage of the correct image: acc/exact/pair_acc higher is better, NLL lower is better
                        d = bootstrap_diff(a, b, idx) if m != "nll" else bootstrap_diff(b, a, idx)
                        name = f"{m}_advantage_correct_vs_{other}" if m != "nll" else f"nll_penalty_{other}_minus_correct"
                        flat.append({"suite": suite, "condition": f"correct_vs_{other}", "subset": sub, "metric": name,
                                     "n_pairs": n, "mean": d["mean"], "ci_low": d["ci_low"], "ci_high": d["ci_high"]})
    return {"n_resamples": n_resamples, "seed": seed, "rows": flat,
            "n_pairs": {s: len(t["pair_ids"]) for s, t in tables.items()}}


def lookup(summary: dict, suite: str, condition: str, subset: str, metric: str) -> dict:
    for r in summary["rows"]:
        if (r["suite"], r["condition"], r["subset"], r["metric"]) == (suite, condition, subset, metric):
            return r
    raise KeyError((suite, condition, subset, metric))
