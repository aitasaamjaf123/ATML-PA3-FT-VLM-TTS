"""Offline evaluation over nested rollout prefixes: accuracy, oracle coverage, group counts, token cost,
sampling-vs-selection failure decomposition, paired question-level bootstrap."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common.stats import bootstrap_indices, bootstrap_mean_ci, paired_bootstrap
from task3_test_time_compute.src import selectors as SEL
from task3_test_time_compute.src.scores import prefix_quality


@dataclass
class OfflineResults:
    dataset: str
    qids: List[str]
    budgets: List[int]
    methods: List[str]
    sel_gid: Dict[str, np.ndarray]   # method -> [Q, O, B] selected answer-group id
    ok: Dict[str, np.ndarray]        # method -> [Q, O, B] bool correctness of selection
    coverage: np.ndarray             # [Q, O, B] oracle coverage (any correct in prefix)
    n_correct: np.ndarray            # [Q, O, B] number of correct candidates in prefix
    n_groups: np.ndarray             # [Q, O, B] distinct answer groups in prefix
    cost: np.ndarray                 # [Q, O, B] generated tokens of all consumed candidates
    cost_first: np.ndarray           # [Q, O, B] tokens of the first candidate only (FirstAns)

    def acc_qb(self, method: str) -> np.ndarray:
        """[Q, B] accuracy averaged over orders (within question first)."""
        return self.ok[method].mean(axis=1)

    def cost_qb(self, method: str) -> np.ndarray:
        return (self.cost_first if method == "first" else self.cost).mean(axis=1)


def eval_question(qd, perms: np.ndarray, budgets: Sequence[int], vision: bool):
    """Evaluate all methods for one question over O orders x B budgets."""
    specs = SEL.method_specs(vision)
    O, B = perms.shape[0], len(budgets)
    needed = {s for s, _ in specs.values() if s is not None}
    sel = {m: np.zeros((O, B), dtype=np.int16) for m in specs}
    cov = np.zeros((O, B), dtype=bool)
    ncorr = np.zeros((O, B), dtype=np.int16)
    ngrp = np.zeros((O, B), dtype=np.int16)
    cost = np.zeros((O, B), dtype=np.int64)
    cost_first = np.zeros((O, B), dtype=np.int64)
    correct = qd.correct
    for o in range(O):
        perm = perms[o]
        cum_t = np.cumsum(qd.T[perm])
        for b, N in enumerate(budgets):
            idx = perm[:N]
            g = qd.gid[idx]
            qs = prefix_quality(qd, idx, needed)
            for m, (skey, agg) in specs.items():
                sel[m][o, b] = SEL.apply_method(agg, g, qs[skey] if skey is not None else None)
            ncorr[o, b] = int(correct[idx].sum())
            cov[o, b] = ncorr[o, b] > 0
            ngrp[o, b] = len(np.unique(g))
            cost[o, b] = int(cum_t[N - 1])
            cost_first[o, b] = int(qd.T[perm[0]])
    return sel, cov, ncorr, ngrp, cost, cost_first


def evaluate_pool(pool, qids: Sequence[str], orders: Dict[str, np.ndarray], budgets: Sequence[int]) -> OfflineResults:
    vision = pool.has_vision
    specs = SEL.method_specs(vision)
    methods = list(specs)
    per_q = []
    for q in qids:
        per_q.append(eval_question(pool.questions[q], orders[q], budgets, vision))
    sel_gid = {m: np.stack([r[0][m] for r in per_q]) for m in methods}
    ok = {}
    for m in methods:
        arr = np.zeros(sel_gid[m].shape, dtype=bool)
        for qi, q in enumerate(qids):
            arr[qi] = pool.questions[q].gcorrect[sel_gid[m][qi]]
        ok[m] = arr
    return OfflineResults(
        dataset=pool.dataset, qids=list(qids), budgets=list(budgets), methods=methods, sel_gid=sel_gid, ok=ok,
        coverage=np.stack([r[1] for r in per_q]), n_correct=np.stack([r[2] for r in per_q]),
        n_groups=np.stack([r[3] for r in per_q]), cost=np.stack([r[4] for r in per_q]),
        cost_first=np.stack([r[5] for r in per_q]),
    )


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def method_table(res: OfflineResults, n_boot: int = 1000, seed: int = 6304) -> pd.DataFrame:
    """Per method x budget: accuracy (+ bootstrap CI), oracle coverage, cost, groups, failure decomposition."""
    idx = bootstrap_indices(len(res.qids), n_boot, seed)
    cov_qb = res.coverage.mean(axis=1)
    ng_qb = res.n_groups.mean(axis=1)
    rows = []
    for m in res.methods:
        acc_qb = res.acc_qb(m)
        cost_qb = res.cost_qb(m)
        okf = res.ok[m].astype(float)
        cov = res.coverage.astype(float)
        samp_fail = (1.0 - cov).mean(axis=1)                    # no correct candidate generated
        sel_fail = (cov * (1.0 - okf)).mean(axis=1)             # a correct candidate existed but was missed
        for b, N in enumerate(res.budgets):
            ci = bootstrap_mean_ci(acc_qb[:, b], idx=idx)
            rows.append({
                "dataset": res.dataset, "method": m, "N": N, "accuracy": ci["mean"],
                "acc_ci_low": ci["ci_low"], "acc_ci_high": ci["ci_high"],
                "oracle_coverage": cov_qb[:, b].mean(), "selection_gap": cov_qb[:, b].mean() - ci["mean"],
                "frac_sampling_failure": samp_fail[:, b].mean(), "frac_selection_failure": sel_fail[:, b].mean(),
                "mean_tokens": cost_qb[:, b].mean(), "mean_distinct_groups": ng_qb[:, b].mean(),
                "n_questions": len(res.qids),
            })
    return pd.DataFrame(rows)


def budget_table(res: OfflineResults, n_boot: int = 1000, seed: int = 6304) -> pd.DataFrame:
    """Method-independent budget statistics: coverage, distinct groups, duplicate-answer rate of added samples."""
    idx = bootstrap_indices(len(res.qids), n_boot, seed)
    cov_qb = res.coverage.mean(axis=1)
    ng_qb = res.n_groups.mean(axis=1)
    ncorr_qb = res.n_correct.mean(axis=1)
    rows = []
    for b, N in enumerate(res.budgets):
        ci = bootstrap_mean_ci(cov_qb[:, b], idx=idx)
        row = {"dataset": res.dataset, "N": N, "oracle_coverage": ci["mean"], "cov_ci_low": ci["ci_low"],
               "cov_ci_high": ci["ci_high"], "first_accuracy": res.acc_qb("first")[:, b].mean(),
               "mean_distinct_groups": ng_qb[:, b].mean(), "mean_correct_candidates": ncorr_qb[:, b].mean(),
               "mean_tokens_all": res.cost_qb("selfcons")[:, b].mean()}
        if b > 0:
            added = N - res.budgets[b - 1]
            new_groups = ng_qb[:, b].mean() - ng_qb[:, b - 1].mean()
            row["new_groups_per_added_sample"] = new_groups / added
            row["duplicate_rate_of_added"] = 1.0 - new_groups / added
            row["coverage_gain"] = cov_qb[:, b].mean() - cov_qb[:, b - 1].mean()
        rows.append(row)
    return pd.DataFrame(rows)


MAIN_COMPARISONS_BOTH = [
    ("selfcons", "first"), ("selfcert_top", "first"), ("logp_top", "first"), ("selfcert_rwc", "selfcons"),
    ("selfcert_rwc", "selfcert_top"), ("selfcert_top", "selfcons"), ("selfcert_top", "shortest"),
]
MAIN_COMPARISONS_VISION = [
    ("visconf_rwc", "visconf_top"), ("visconf_rwc", "selfcons"), ("visconf_rwc", "selfcert_rwc"),
    ("visconf_top", "selfcert_top"), ("visconf_top", "vision_top"), ("visconf_rwc", "tokenfusion_rwc"),
    ("visconf_top", "tokenfusion_top"),
]


def diff_table(res: OfflineResults, n_boot: int = 1000, seed: int = 6304) -> pd.DataFrame:
    idx = bootstrap_indices(len(res.qids), n_boot, seed)
    pairs = list(MAIN_COMPARISONS_BOTH) + (list(MAIN_COMPARISONS_VISION) if "visconf_rwc" in res.methods else [])
    rows = []
    for a, b_ in pairs:
        for bi, N in enumerate(res.budgets):
            if N == 1:
                continue  # all rules coincide at N = 1
            r = paired_bootstrap(res.acc_qb(a)[:, bi], res.acc_qb(b_)[:, bi], idx=idx)
            rows.append({"dataset": res.dataset, "method_a": a, "method_b": b_, "N": N, **r})
    return pd.DataFrame(rows)


def per_question_table(res: OfflineResults) -> pd.DataFrame:
    """Long table with one row per (question, order, budget): selected group ids, correctness, tokens."""
    Q, O, B = res.coverage.shape
    data = {
        "dataset": res.dataset,
        "qid": np.repeat(np.asarray(res.qids), O * B),
        "order": np.tile(np.repeat(np.arange(O), B), Q),
        "N": np.tile(np.asarray(res.budgets), Q * O),
        "oracle_coverage": res.coverage.ravel().astype(np.int8),
        "n_correct_in_prefix": res.n_correct.ravel(),
        "n_groups": res.n_groups.ravel(),
        "tokens_all": res.cost.ravel(),
        "tokens_first": res.cost_first.ravel(),
    }
    for m in res.methods:
        data[f"sel_{m}"] = res.sel_gid[m].ravel()
        data[f"ok_{m}"] = res.ok[m].ravel().astype(np.int8)
    return pd.DataFrame(data)


def groups_table(pool, qids: Sequence[str]) -> pd.DataFrame:
    rows = []
    for q in qids:
        qd = pool.questions[q]
        counts = np.bincount(qd.gid, minlength=len(qd.group_answers))
        for gid, (ans, ok) in enumerate(zip(qd.group_answers, qd.gcorrect)):
            rows.append({"dataset": pool.dataset, "qid": q, "gid": gid, "answer": ans, "correct": bool(ok),
                         "count_in_pool": int(counts[gid]), "gold": qd.gold})
    return pd.DataFrame(rows)


def decomposition_table(res: OfflineResults, methods: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Fractions of questions: correct selection / selection failure / sampling failure (sums to 1)."""
    methods = methods or res.methods
    rows = []
    cov = res.coverage.astype(float)
    for m in methods:
        okf = res.ok[m].astype(float)
        for bi, N in enumerate(res.budgets):
            rows.append({"dataset": res.dataset, "method": m, "N": N,
                         "selected_correct": okf[:, :, bi].mean(),
                         "selection_failure": (cov[:, :, bi] * (1 - okf[:, :, bi])).mean(),
                         "sampling_failure": (1 - cov[:, :, bi]).mean()})
    return pd.DataFrame(rows)


def selection_agreement(res: OfflineResults, a: str, b: str) -> pd.DataFrame:
    """How often two selectors return the same answer group, per budget."""
    return pd.DataFrame({"dataset": res.dataset, "method_a": a, "method_b": b, "N": res.budgets,
                         "agreement": [float((res.sel_gid[a][:, :, i] == res.sel_gid[b][:, :, i]).mean()) for i in range(len(res.budgets))]})
