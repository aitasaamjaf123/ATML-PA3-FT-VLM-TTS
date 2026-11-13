"""Step 3: what do the signals capture? Within-question quartile lifts, correlations, length control,
confidence-vision relationships and the tokenwise-vs-trajectory fusion identity (at N = 32)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.stats import bootstrap_indices, bootstrap_mean_ci
from task3_test_time_compute.src import scores as S


def _signals_for_question(qd, vision: bool) -> Dict[str, np.ndarray]:
    """All 32 rollouts revealed (N = 32): calibration is order independent."""
    sig = {"length": qd.T.astype(float), "logp": S.neginf(qd.L), "selfcert": S.neginf(qd.K)}
    if vision and qd.tok_k is not None:
        vc = S.prefix_visconf(qd.tok_k, qd.tok_v)
        sig.update(vision=S.neginf(qd.V), visconf=S.neginf(vc["Z"]), tokenfusion=S.neginf(vc["Zt"]),
                   token_minus_traj=np.nan_to_num(vc["cov"], nan=0.0))
    return sig


def _quartile_diff(score: np.ndarray, corr: np.ndarray, frac: float) -> float:
    """mean correctness in the top quartile minus bottom quartile of `score` (stable ties: lower index first)."""
    k = max(1, int(len(score) * frac))
    order = np.argsort(-score, kind="stable")
    return float(corr[order[:k]].mean() - corr[order[-k:]].mean())


def question_signal_table(pool, qids: Sequence[str], frac: float = 0.25) -> pd.DataFrame:
    rows = []
    for q in qids:
        qd = pool.questions[q]
        corr = qd.correct.astype(float)
        sig = _signals_for_question(qd, pool.has_vision)
        row = {"dataset": pool.dataset, "qid": q, "p_correct": corr.mean(), "mixed": bool(0 < corr.mean() < 1)}
        for name, s in sig.items():
            finite = s[np.isfinite(s)]
            varies = len(finite) > 1 and np.ptp(finite) > 0
            row[f"var_{name}"] = bool(varies)
            row[f"qdiff_{name}"] = _quartile_diff(s, corr, frac) if varies else np.nan
            if varies and 0 < corr.mean() < 1:
                rho = spearmanr(np.where(np.isfinite(s), s, np.nanmin(finite) - 1.0), corr)[0]
                row[f"rho_{name}"] = float(rho) if np.isfinite(rho) else np.nan
            else:
                row[f"rho_{name}"] = np.nan
        for name in ("logp", "selfcert", "vision", "visconf"):
            if name in sig and row.get(f"var_{name}", False) and row.get("var_length", False):
                rho = spearmanr(sig["length"], np.where(np.isfinite(sig[name]), sig[name], -1e9))[0]
                row[f"rho_length_{name}"] = float(rho) if np.isfinite(rho) else np.nan
        # control: shortest-rollout selection (ties -> lowest index), longest for reference
        row["acc_shortest"] = float(corr[int(np.argmin(qd.T))])
        row["acc_longest"] = float(corr[int(np.argmax(qd.T))])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_quartiles(tbl: pd.DataFrame, n_boot: int = 1000, seed: int = 6304) -> pd.DataFrame:
    """Average the within-question top-minus-bottom quartile difference across questions
    (all questions with score variance, and mixed-correctness questions only)."""
    signals = [c[len("qdiff_"):] for c in tbl.columns if c.startswith("qdiff_")]
    rows = []
    for name in signals:
        for subset, mask in (("all_questions", np.ones(len(tbl), bool)), ("mixed_questions", tbl["mixed"].to_numpy())):
            x = tbl.loc[mask, f"qdiff_{name}"].dropna().to_numpy()
            if len(x) == 0:
                continue
            ci = bootstrap_mean_ci(x, n_boot=n_boot, seed=seed)
            rho = tbl.loc[mask, f"rho_{name}"].dropna().to_numpy()
            rows.append({"dataset": tbl["dataset"].iloc[0], "signal": name, "subset": subset, "n_questions": len(x),
                         "top_minus_bottom_quartile": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                         "mean_within_q_spearman_with_correct": float(rho.mean()) if len(rho) else np.nan})
    return pd.DataFrame(rows)


def length_control_table(tbl: pd.DataFrame, offline_n32: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    rows = [{"dataset": tbl["dataset"].iloc[0], "selector": "shortest_rollout", "accuracy_N32": float(tbl["acc_shortest"].mean())},
            {"dataset": tbl["dataset"].iloc[0], "selector": "longest_rollout", "accuracy_N32": float(tbl["acc_longest"].mean())}]
    for k, v in (offline_n32 or {}).items():
        rows.append({"dataset": tbl["dataset"].iloc[0], "selector": k, "accuracy_N32": float(v)})
    return pd.DataFrame(rows)


def length_correlations(tbl: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in tbl.columns if c.startswith("rho_length_")]
    return pd.DataFrame([{"dataset": tbl["dataset"].iloc[0], "pair": c.replace("rho_length_", "length~"),
                          "mean_within_q_spearman": float(tbl[c].dropna().mean()), "n_questions": int(tbl[c].notna().sum())}
                         for c in cols])


# --------------------------------------------------------------------------------------
# Confidence-vision relationships and fusion identity (MathVista)
# --------------------------------------------------------------------------------------
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def confidence_vision(pool, qids: Sequence[str]) -> Dict[str, object]:
    """Within-rollout Pearson correlation between calibrated confidence c_t (and raw k_t) and visual
    engagement v_t, summarised per question and then across questions; plus the fusion identity check
    Z_token - Z == Cov_t(c, v) (population covariance)."""
    per_q = []
    max_dev = 0.0
    n_checked = 0
    for q in qids:
        qd = pool.questions[q]
        nonempty = [np.asarray(k, np.float64) for k in qd.tok_k if len(k)]
        if not nonempty:
            continue
        q05, q95 = S.calibration_bounds(np.concatenate(nonempty))
        vc = S.prefix_visconf(qd.tok_k, qd.tok_v)
        corr = qd.correct
        r_c, r_k, covs = [], [], []
        for i in range(qd.n_rollouts):
            if len(qd.tok_k[i]) == 0:
                continue
            c = S.calibrate(qd.tok_k[i], q05, q95)
            v = np.asarray(qd.tok_v[i], np.float64)
            r_c.append(_pearson(c, v))
            r_k.append(_pearson(np.asarray(qd.tok_k[i], np.float64), v))
            cov_direct = float(np.mean((c - c.mean()) * (v - v.mean())))
            max_dev = max(max_dev, abs((vc["Zt"][i] - vc["Z"][i]) - cov_direct))
            n_checked += 1
            covs.append(cov_direct)
        r_c = np.array(r_c)
        ok_idx = [i for i in range(qd.n_rollouts) if len(qd.tok_k[i])]
        cflags = corr[ok_idx]
        rc_correct = np.nanmean(r_c[cflags]) if cflags.any() and not np.isnan(r_c[cflags]).all() else np.nan
        rc_wrong = np.nanmean(r_c[~cflags]) if (~cflags).any() and not np.isnan(r_c[~cflags]).all() else np.nan
        per_q.append({"qid": q, "mean_corr_c_v": float(np.nanmean(r_c)) if not np.isnan(r_c).all() else np.nan,
                      "mean_corr_k_v": float(np.nanmean(r_k)) if not np.isnan(r_k).all() else np.nan,
                      "mean_cov_c_v": float(np.mean(covs)) if covs else np.nan,
                      "corr_correct_rollouts": rc_correct, "corr_wrong_rollouts": rc_wrong})
    df = pd.DataFrame(per_q)

    def summ(col):
        x = df[col].dropna().to_numpy()
        if len(x) == 0:
            return {}
        return {"mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std()),
                "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)),
                "frac_negative": float((x < 0).mean()), "n_questions": int(len(x))}

    both = df.dropna(subset=["corr_correct_rollouts", "corr_wrong_rollouts"])
    summary = {
        "corr_calibrated_confidence_vs_vision": summ("mean_corr_c_v"),
        "corr_raw_kl_vs_vision": summ("mean_corr_k_v"),
        "mean_token_covariance_c_v": summ("mean_cov_c_v"),
        "correct_minus_wrong_rollout_corr": {"mean": float((both["corr_correct_rollouts"] - both["corr_wrong_rollouts"]).mean()) if len(both) else None,
                                              "n_questions": int(len(both))},
        "fusion_identity": {"max_abs_deviation_Ztoken_minus_Z_minus_Cov": float(max_dev), "n_rollouts_checked": int(n_checked),
                            "holds_within_1e-8": bool(max_dev < 1e-8)},
    }
    return {"per_question": df, "summary": summary}


def select_illustrative_trajectories(pool, qids: Sequence[str], k_per_rule: int = 1) -> List[dict]:
    """Fixed, rule-based set of illustrative MathVista questions (N = 32, ties -> smallest qid):
      agree_correct   : VisConf-top, self-certainty-top and token-fusion-top all correct
      visconf_rescue  : VisConf-top correct, self-certainty-top wrong
      visconf_miss    : VisConf-top wrong, self-certainty-top correct
      fusion_disagree : token-fusion-top and VisConf-top select different answers
    Each entry lists the chosen rollout indices; token series are extracted in qualitative.py."""
    rules: Dict[str, List[dict]] = {"agree_correct": [], "visconf_rescue": [], "visconf_miss": [], "fusion_disagree": []}
    for q in sorted(qids):
        qd = pool.questions[q]
        sig = _signals_for_question(qd, True)
        i_vc, i_sc, i_tf = int(np.argmax(sig["visconf"])), int(np.argmax(sig["selfcert"])), int(np.argmax(sig["tokenfusion"]))
        corr = qd.correct
        info = {"qid": q, "visconf_idx": i_vc, "selfcert_idx": i_sc, "tokenfusion_idx": i_tf,
                "visconf_correct": bool(corr[i_vc]), "selfcert_correct": bool(corr[i_sc]), "tokenfusion_correct": bool(corr[i_tf])}
        if corr[i_vc] and corr[i_sc] and corr[i_tf] and len(rules["agree_correct"]) < k_per_rule:
            rules["agree_correct"].append(info)
        if corr[i_vc] and not corr[i_sc] and len(rules["visconf_rescue"]) < k_per_rule:
            rules["visconf_rescue"].append(info)
        if (not corr[i_vc]) and corr[i_sc] and len(rules["visconf_miss"]) < k_per_rule:
            rules["visconf_miss"].append(info)
        if qd.gid[i_tf] != qd.gid[i_vc] and len(rules["fusion_disagree"]) < k_per_rule:
            rules["fusion_disagree"].append(info)
    return [dict(rule=r, **e) for r, es in rules.items() for e in es]
