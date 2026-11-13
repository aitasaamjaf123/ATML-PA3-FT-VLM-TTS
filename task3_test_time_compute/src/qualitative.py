"""Step 6: deterministic qualitative examples selected by explicit programmatic rules."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from task3_test_time_compute.src import adaptive as AD
from task3_test_time_compute.src import selectors as SEL
from task3_test_time_compute.src.scores import prefix_quality

SNIPPET = 500

RULES = {
    "rwc_rescue": "Held-out question, order 0, N=16: self-consistency wrong but quality+RWC right. Ranked by fewest correct "
                  "candidates in the prefix (most 'minority' rescues first), then qid.",
    "rwc_harm": "Held-out question, order 0, N=16: self-consistency right but quality+RWC wrong. Ranked by most correct candidates, then qid.",
    "visconf_rescue": "MathVista, order 0, N=16: VisConf-highest right while self-certainty-highest wrong. Ranked by fewest correct candidates, then qid.",
    "refinement_wrong_to_correct": "Live MATH subset: initial answer wrong, revised answer right. Sorted by qid.",
    "refinement_correct_to_wrong": "Live MATH subset: initial answer right, revised answer wrong. Sorted by qid.",
    "premature_stop": "Held-out, order 0, frozen policy: stopped before 32 with a wrong answer while fixed-32 self-consistency is right. "
                      "Ranked by largest fraction of correct rollouts in the full pool (most recoverable first), then qid.",
}


def _snip(t: Optional[str], n: int = SNIPPET) -> Optional[str]:
    if t is None:
        return None
    t = str(t).strip().replace("\r", "")
    return t if len(t) <= n else t[:n] + " …"


def _answers_in_prefix(qd, idx) -> Dict[str, int]:
    g = qd.gid[idx]
    return {str(qd.group_answers[int(u)]) + (" [correct]" if qd.gcorrect[int(u)] else ""): int(c) for u, c in zip(*np.unique(g, return_counts=True))}


def _case(qd, idx, perm, extra: dict) -> dict:
    top = int(idx[0])
    return {"qid": qd.qid, "gold": qd.gold, "n_correct_in_prefix": int(qd.correct[idx].sum()), "answer_votes": _answers_in_prefix(qd, idx),
            "first_rollout_text": _snip(qd.texts[top]) if qd.texts else None, **extra}


def selection_examples(pool, qids: Sequence[str], orders: Dict[str, np.ndarray], n: int = 16, per_rule: int = 3) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {"rwc_rescue": [], "rwc_harm": [], "visconf_rescue": []}
    vision = pool.has_vision
    qual = "visconf" if vision else "selfcert"
    cands = {"rwc_rescue": [], "rwc_harm": [], "visconf_rescue": []}
    for q in qids:
        qd = pool.questions[q]
        perm = orders[q][0]
        idx = perm[:n]
        g = qd.gid[idx]
        names = {qual, "selfcert"}
        qs = prefix_quality(qd, idx, names)
        sc = SEL.vote_winner(g)
        rw = SEL.select_rwc(g, qs[qual])[0]
        ncor = int(qd.correct[idx].sum())
        info = {"selfcons_answer": qd.group_answers[sc], "rwc_answer": qd.group_answers[rw]}
        if qd.gcorrect[rw] and not qd.gcorrect[sc]:
            cands["rwc_rescue"].append((ncor, q, _case(qd, idx, perm, info)))
        if qd.gcorrect[sc] and not qd.gcorrect[rw]:
            cands["rwc_harm"].append((-ncor, q, _case(qd, idx, perm, info)))
        if vision:
            vc_ans = SEL.select_top(g, qs["visconf"])
            sc_top = SEL.select_top(g, qs["selfcert"])
            if qd.gcorrect[vc_ans] and not qd.gcorrect[sc_top]:
                cands["visconf_rescue"].append((ncor, q, _case(qd, idx, perm, {"visconf_top_answer": qd.group_answers[vc_ans],
                                                                              "selfcert_top_answer": qd.group_answers[sc_top]})))
    for k, lst in cands.items():
        lst.sort(key=lambda t: (t[0], t[1]))
        out[k] = [c for _, _, c in lst[:per_rule]]
    return out


def premature_stop_examples(pool, F: AD.Features, detail: Dict[str, np.ndarray], orders: Dict[str, np.ndarray], fixed32: Dict[str, np.ndarray],
                            per_rule: int = 3) -> List[dict]:
    cands = []
    C = len(F.checkpoints)
    for qi, q in enumerate(F.qids):
        stop = int(detail["stop_idx"][qi, 0])
        if stop < C and not detail["ok"][qi, 0] and fixed32["ok"][qi, 0]:
            qd = pool.questions[q]
            n = F.checkpoints[stop]
            idx = orders[q][0][:n]
            cands.append((-F.p_correct_pool[qi], q, _case(qd, idx, orders[q][0], {
                "stopped_at": n, "stop_leader_fraction": float(F.leader_frac[qi, 0, stop]), "rwc_margin_at_stop": float(F.margin[qi, 0, stop]),
                "pool_correct_fraction": float(F.p_correct_pool[qi]), "early_answer": qd.group_answers[int(detail["sel_gid"][qi, 0])],
                "fixed32_answer": qd.group_answers[int(fixed32["sel_gid"][qi, 0])]})))
    cands.sort(key=lambda t: (t[0], t[1]))
    return [c for _, _, c in cands[:per_rule]]


def refinement_examples(live_rows: List[dict], gens: Dict, per_rule: int = 3) -> Dict[str, List[dict]]:
    """`live_rows`: per-question dicts; `gens`: {(qid, role): generation row} (from generations.jsonl)."""
    out = {"refinement_wrong_to_correct": [], "refinement_correct_to_wrong": []}
    for r in sorted(live_rows, key=lambda r: str(r["qid"])):
        key = {"wrong->correct": "refinement_wrong_to_correct", "correct->wrong": "refinement_correct_to_wrong"}.get(r["transition"])
        if key and len(out[key]) < per_rule:
            q = str(r["qid"])
            out[key].append({"qid": q, "gold": r["gold"], "initial_answer": r["initial_answer"], "refined_answer": r["refined_answer"],
                             "initial_response": _snip(gens.get((q, "parallel_0"), {}).get("response")),
                             "revision_response": _snip(gens.get((q, "refine"), {}).get("response"))})
    return out


def trajectory_series(qd, rollout_idx: int, window: int = 25) -> Dict[str, List[float]]:
    """Token series (c_t, v_t, k_t) for one rollout, calibrated on the full 32-rollout pool, smoothed by a moving
    mean and resampled to at most 200 points for plotting."""
    from task3_test_time_compute.src import scores as S

    pooled = np.concatenate([np.asarray(k, np.float64) for k in qd.tok_k if len(k)])
    q05, q95 = S.calibration_bounds(pooled)
    k = np.asarray(qd.tok_k[rollout_idx], np.float64)
    c = S.calibrate(k, q05, q95)
    v = np.asarray(qd.tok_v[rollout_idx], np.float64)

    def smooth(x):
        if len(x) < 2:
            return x
        w = min(window, len(x))
        y = np.convolve(x, np.ones(w) / w, mode="valid")
        if len(y) > 200:
            y = y[np.linspace(0, len(y) - 1, 200).astype(int)]
        return y

    return {"confidence_c": smooth(c).tolist(), "vision_v": smooth(v).tolist(), "n_tokens": int(len(k))}
