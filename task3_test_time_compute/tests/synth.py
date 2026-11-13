"""Synthetic rollout data (no parquet / torch needed) mimicking the joined export frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from task3_test_time_compute.src import answers as A
from task3_test_time_compute.src import load_data as LD


def make_joined_df(dataset="math", n_q=30, R=32, vision=False, seed=0, with_problem=False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for qi in range(n_q):
        qid = f"q{qi:03d}"
        gold = str(int(rng.integers(0, 10)))
        p = rng.uniform(0.05, 0.95)
        for r in range(R):
            T = int(rng.integers(20, 120))
            correct = rng.random() < p
            if rng.random() < 0.05:
                ans = None
            elif correct:
                ans = gold
            else:
                ans = str(int(rng.integers(0, 10)))
            logp = -rng.exponential(0.6 if correct else 0.9, T)
            kl = rng.gamma(4.0 if correct else 3.0, 1.0, T)
            row = {"dataset": dataset, "sample_id": qid, "rollout_id": r, "text": f"reasoning {qid} {r} \\boxed{{{ans}}}",
                   "token_ids": list(range(T)), "length": T, "extracted_answer": ans, "gold_answer": gold,
                   "logp": logp, "kl_u_p": kl}
            if vision:
                img = rng.random(T) * (0.5 if correct else 0.3)
                row.update(image_attn=img, prompt_attn=rng.random(T) * 0.3, gen_attn=rng.random(T) * 0.3)
            if with_problem:
                row["problem"] = f"What is {qid}?"
            rows.append(row)
    return pd.DataFrame(rows)


def mathvista_ctx(n_q=30):
    meta = {f"q{i:03d}": {"question_type": "free_form", "answer_type": "integer", "choices": None, "precision": None} for i in range(n_q)}
    return A.MathVistaAnswerContext(meta)


def make_pool(dataset="math", n_q=30, vision=False, seed=0, with_problem=False) -> LD.Pool:
    raw = make_joined_df(dataset, n_q, 32, vision, seed, with_problem)
    gcols = ["dataset", "sample_id", "rollout_id", "text", "token_ids", "length", "extracted_answer", "gold_answer"]
    if with_problem:
        gcols.append("problem")
    gen = raw[gcols].copy()
    prob = raw[["dataset", "sample_id", "rollout_id", "logp", "kl_u_p"]].copy()
    attn = raw[["dataset", "sample_id", "rollout_id", "image_attn", "prompt_attn", "gen_attn"]].copy() if vision else None
    df = LD.join_frames(gen, prob, attn, vision)   # same path as the real parquet loader
    rep = LD.check_alignment(df, vision, strict=True)
    ctx = mathvista_ctx(n_q) if vision else A.MathAnswerContext(force_fallback=True)
    return LD.build_pool(df, dataset, ctx, vision, alignment=rep)
