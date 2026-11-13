"""Step 5: offline adaptive sampling. Stopping decisions use only revealed candidates (no gold, no future
rollouts). Both controllers return the current self-consistency answer, isolating the stopping rule."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common.stats import bootstrap_indices, paired_bootstrap
from task3_test_time_compute.src import selectors as SEL
from task3_test_time_compute.src.scores import prefix_quality

TOL = 1e-9


@dataclass
class Features:
    """Per (question, order) stopping evidence at each checkpoint, plus the fixed-32 outcome."""
    qids: List[str]
    checkpoints: List[int]
    final_n: int
    # [Q, O, C] evidence at checkpoints
    sc_gid: np.ndarray
    unique_leader: np.ndarray
    leader_frac: np.ndarray
    leader_is_na: np.ndarray
    rwc_gid: np.ndarray
    margin: np.ndarray
    # [Q, O, C+1] outcome if stopping at each checkpoint (last column = consume all 32)
    ok: np.ndarray
    cost: np.ndarray
    n_used: np.ndarray
    # [Q, O, C+1] selected group id at each stopping point
    sel_gid: np.ndarray
    p_correct_pool: np.ndarray  # [Q] fraction of correct rollouts in the full pool (retrospective only)


def build_features(pool, qids: Sequence[str], orders: Dict[str, np.ndarray], checkpoints=(4, 8, 16), final_n: int = 32) -> Features:
    vision = pool.has_vision
    qual_key = "visconf" if vision else "selfcert"
    cps = list(checkpoints)
    C = len(cps)
    Q = len(qids)
    O = orders[qids[0]].shape[0]
    sc_gid = np.zeros((Q, O, C), np.int16)
    uniq = np.zeros((Q, O, C), bool)
    frac = np.zeros((Q, O, C))
    isna = np.zeros((Q, O, C), bool)
    rwc_gid = np.zeros((Q, O, C), np.int16)
    margin = np.zeros((Q, O, C))
    ok = np.zeros((Q, O, C + 1), bool)
    cost = np.zeros((Q, O, C + 1), np.int64)
    n_used = np.tile(np.array(cps + [final_n]), (Q, O, 1))
    sel = np.zeros((Q, O, C + 1), np.int16)
    pcorr = np.zeros(Q)
    for qi, q in enumerate(qids):
        qd = pool.questions[q]
        pcorr[qi] = qd.correct.mean()
        for o in range(O):
            perm = orders[q][o]
            cum = np.cumsum(qd.T[perm])
            for ci, n in enumerate(cps):
                idx = perm[:n]
                g = qd.gid[idx]
                win, cnt, is_unique = SEL.unique_leader(g)
                sc_gid[qi, o, ci] = win
                uniq[qi, o, ci] = is_unique
                frac[qi, o, ci] = cnt / n
                isna[qi, o, ci] = win == 0
                score = prefix_quality(qd, idx, [qual_key])[qual_key]
                rw, W, _ = SEL.select_rwc(g, score)
                rwc_gid[qi, o, ci] = rw
                margin[qi, o, ci] = SEL.rwc_margin(W)
                sel[qi, o, ci] = win
                ok[qi, o, ci] = qd.gcorrect[win]
                cost[qi, o, ci] = cum[n - 1]
            g_all = qd.gid[perm[:final_n]]
            w32 = SEL.vote_winner(g_all)
            sel[qi, o, C] = w32
            ok[qi, o, C] = qd.gcorrect[w32]
            cost[qi, o, C] = cum[final_n - 1]
    return Features(list(qids), cps, final_n, sc_gid, uniq, frac, isna, rwc_gid, margin, ok, cost, n_used, sel, pcorr)


def stop_mask(F: Features, tau: float, gamma: Optional[float]) -> np.ndarray:
    """[Q, O, C] boolean: does the controller stop at each checkpoint?
    gamma=None -> agreement controller; otherwise quality-aware controller."""
    n = np.asarray(F.checkpoints, dtype=float)[None, None, :]
    agree = F.unique_leader & (~F.leader_is_na) & (F.leader_frac * n >= tau * n - TOL)
    if gamma is None:
        return agree
    return agree & (F.sc_gid == F.rwc_gid) & (F.margin >= gamma - TOL)


def run_policy(F: Features, tau: Optional[float], gamma: Optional[float]) -> Dict[str, np.ndarray]:
    """Simulate a stopping policy; tau=None means the fixed-32 fallback (never stop early)."""
    Q, O, C = F.sc_gid.shape
    if tau is None:
        stop_idx = np.full((Q, O), C, dtype=int)
    else:
        mask = stop_mask(F, tau, gamma)
        stop_idx = np.full((Q, O), C, dtype=int)
        done = np.zeros((Q, O), bool)
        for c in range(C):
            newly = mask[:, :, c] & ~done
            stop_idx[newly] = c
            done |= mask[:, :, c]
    take = lambda arr: np.take_along_axis(arr, stop_idx[..., None], axis=2)[..., 0]
    return {"stop_idx": stop_idx, "ok": take(F.ok), "cost": take(F.cost), "n": take(F.n_used), "sel_gid": take(F.sel_gid)}


def policy_summary(F: Features, res: Dict[str, np.ndarray]) -> dict:
    C = len(F.checkpoints)
    out = {"accuracy": float(res["ok"].mean()), "mean_tokens": float(res["cost"].mean()), "mean_rollouts": float(res["n"].mean())}
    for ci, n in enumerate(F.checkpoints + [F.final_n]):
        out[f"stop_frac_{n}"] = float((res["stop_idx"] == ci).mean())
    return out


def policy_grid(taus: Sequence[float], gammas: Sequence[float]) -> List[dict]:
    grid = [{"family": "agreement", "tau": t, "gamma": None} for t in taus]
    grid += [{"family": "quality_aware", "tau": t, "gamma": g} for t in taus for g in gammas]
    return grid


def evaluate_grid(F: Features, taus, gammas) -> pd.DataFrame:
    rows = []
    fixed = run_policy(F, None, None)
    rows.append({"family": "fixed_32", "tau": np.nan, "gamma": np.nan, **policy_summary(F, fixed)})
    for p in policy_grid(taus, gammas):
        res = run_policy(F, p["tau"], p["gamma"])
        rows.append({"family": p["family"], "tau": p["tau"], "gamma": np.nan if p["gamma"] is None else p["gamma"], **policy_summary(F, res)})
    return pd.DataFrame(rows)


def select_setting(grid_df: pd.DataFrame, family: str, tolerance: float = 0.01) -> dict:
    """Lowest mean token cost with dev accuracy >= fixed-32 SC accuracy - tolerance (the fixed-32 fallback is
    always a candidate). Ties: higher accuracy, then smaller tau, then smaller gamma."""
    acc32 = float(grid_df.loc[grid_df.family == "fixed_32", "accuracy"].iloc[0])
    cand = grid_df[(grid_df.family == family) | (grid_df.family == "fixed_32")].copy()
    cand = cand[cand.accuracy >= acc32 - tolerance - 1e-12].copy()
    cand["_tau"] = cand["tau"].fillna(np.inf)
    cand["_gamma"] = cand["gamma"].fillna(-1.0)
    cand["_cost"] = cand["mean_tokens"].round(9)
    cand["_acc"] = cand["accuracy"].round(12)
    cand = cand.sort_values(["_cost", "_acc", "_tau", "_gamma"], ascending=[True, False, True, True])
    best = cand.iloc[0]
    return {"family": family, "tau": None if np.isnan(best["tau"]) else float(best["tau"]),
            "gamma": None if np.isnan(best["gamma"]) else float(best["gamma"]),
            "is_fixed_32_fallback": bool(best["family"] == "fixed_32"), "dev_accuracy": float(best["accuracy"]),
            "dev_mean_tokens": float(best["mean_tokens"]), "dev_mean_rollouts": float(best["mean_rollouts"]),
            "dev_fixed32_accuracy": acc32}


# --------------------------------------------------------------------------------------
# Held-out analysis
# --------------------------------------------------------------------------------------
def nearest_fixed_budget(cost_target: float, budgets: Sequence[int], fixed_cost: np.ndarray) -> int:
    return int(np.argmin(np.abs(np.asarray(fixed_cost) - cost_target)))


def heldout_report(F: Features, selected: Dict[str, dict], offline_res, n_boot: int = 1000, seed: int = 6304):
    """Evaluate frozen settings on the held-out questions. `offline_res` provides the fixed-N self-consistency
    baselines computed on the SAME questions and orders."""
    assert list(offline_res.qids) == list(F.qids)
    idx = bootstrap_indices(len(F.qids), n_boot, seed)
    budgets = offline_res.budgets
    sc_acc_qb = offline_res.acc_qb("selfcons")
    sc_cost_b = offline_res.cost_qb("selfcons").mean(axis=0)
    rows, diffs, stop_rows, wrong_rows, bin_rows = [], [], [], [], []
    fixed_rows = [{"policy": f"fixed_N={N}", "family": "fixed", "N": N, "accuracy": float(sc_acc_qb[:, i].mean()),
                   "mean_tokens": float(sc_cost_b[i]), "mean_rollouts": float(N)} for i, N in enumerate(budgets)]
    fixed32 = run_policy(F, None, None)
    acc32_q = fixed32["ok"].mean(axis=1)
    details = {}
    for fam, sel in selected.items():
        res = run_policy(F, sel["tau"], sel["gamma"])
        details[fam] = res
        acc_q = res["ok"].mean(axis=1)
        s = policy_summary(F, res)
        rows.append({"policy": fam, "family": fam, "tau": sel["tau"], "gamma": sel["gamma"], **s,
                     "fallback_fixed_32": sel["is_fixed_32_fallback"]})
        j = nearest_fixed_budget(s["mean_tokens"], budgets, sc_cost_b)
        d32 = paired_bootstrap(acc_q, acc32_q, idx=idx)
        dnear = paired_bootstrap(acc_q, sc_acc_qb[:, j], idx=idx)
        diffs.append({"policy": fam, "baseline": "fixed_32_SC", **d32})
        diffs.append({"policy": fam, "baseline": f"cost_matched_fixed_N={budgets[j]}_SC", "baseline_mean_tokens": float(sc_cost_b[j]), **dnear})
        # stopping distribution
        for ci, n in enumerate(F.checkpoints + [F.final_n]):
            stop_rows.append({"policy": fam, "stop_at_n": n, "fraction": float((res["stop_idx"] == ci).mean())})
        # incorrect early stops
        early = res["stop_idx"] < len(F.checkpoints)
        ok_e, ok32 = res["ok"], fixed32["ok"]
        wrong_early = early & ~ok_e
        wrong_rows.append({
            "policy": fam, "n_decisions": int(early.size), "early_stop_rate": float(early.mean()),
            "early_stops": int(early.sum()), "early_correct": int((early & ok_e).sum()), "early_wrong": int(wrong_early.sum()),
            "early_wrong_rate_among_early": float(wrong_early.sum() / max(early.sum(), 1)),
            "early_wrong_but_fixed32_correct": int((wrong_early & ok32).sum()),
            "early_wrong_and_fixed32_wrong": int((wrong_early & ~ok32).sum()),
            "early_correct_but_fixed32_wrong": int((early & ok_e & ~ok32).sum()),
            "share_of_early_wrong_recoverable_by_fixed32": float((wrong_early & ok32).sum() / max(wrong_early.sum(), 1)),
            "mean_leader_frac_early_wrong": float(np.take_along_axis(F.leader_frac, np.minimum(res["stop_idx"], len(F.checkpoints) - 1)[..., None], 2)[..., 0][wrong_early].mean()) if wrong_early.any() else np.nan,
            "mean_leader_frac_early_correct": float(np.take_along_axis(F.leader_frac, np.minimum(res["stop_idx"], len(F.checkpoints) - 1)[..., None], 2)[..., 0][early & ok_e].mean()) if (early & ok_e).any() else np.nan,
        })
        # by fraction of correct candidates in the full pool (retrospective)
        edges = [(-1e-9, 1e-9, "0 (no correct rollout)"), (1e-9, 0.25, "(0, .25]"), (0.25, 0.5, "(.25, .5]"), (0.5, 0.75, "(.5, .75]"), (0.75, 1.0 + 1e-9, "(.75, 1]")]
        for lo, hi, lab in edges:
            m = (F.p_correct_pool > lo) & (F.p_correct_pool <= hi)
            if not m.any():
                continue
            bin_rows.append({"policy": fam, "pool_correct_fraction": lab, "n_questions": int(m.sum()),
                             "mean_rollouts_used": float(res["n"][m].mean()), "early_stop_rate": float(early[m].mean()),
                             "adaptive_accuracy": float(res["ok"][m].mean()), "fixed32_accuracy": float(fixed32["ok"][m].mean()),
                             "early_wrong_rate": float(wrong_early[m].mean())})
    return {
        "summary": pd.DataFrame(fixed_rows + rows), "diffs": pd.DataFrame(diffs), "stop_distribution": pd.DataFrame(stop_rows),
        "wrong_stops": pd.DataFrame(wrong_rows), "by_pool_fraction": pd.DataFrame(bin_rows), "details": details,
        "fixed32": fixed32,
    }
