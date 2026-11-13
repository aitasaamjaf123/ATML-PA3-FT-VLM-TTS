"""Load, join and validate the supplied rollout exports; build per-question pools with answer groups,
correctness labels and rollout-level quality signals."""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from task3_test_time_compute.src import answers as A
from task3_test_time_compute.src import scores as S

KEYS = ["dataset", "sample_id", "rollout_id"]

COLUMN_CANDIDATES = {
    "text": ["text", "generated_text", "response", "output_text", "generation"],
    "token_ids": ["token_ids", "generated_token_ids", "output_token_ids", "gen_token_ids"],
    "length": ["length", "num_tokens", "n_tokens", "num_generated_tokens", "generated_length", "output_length"],
    "answer": ["extracted_answer", "pred_answer", "predicted_answer", "answer_extracted", "extracted", "pred", "prediction", "extracted_answers"],
    "gold": ["gold_answer", "gold", "ground_truth", "reference_answer", "answer_gold", "gt_answer", "target", "answer"],
    "logp": ["logp", "token_logp", "sampled_logp", "logprobs"],
    "kl_u_p": ["kl_u_p", "kl_uniform", "kl_u_p_tokens"],
    "image_attn": ["image_attn", "image_attention", "attn_image", "image", "img_attn", "image_attn_mass", "A_image"],
    "prompt_attn": ["prompt_attn", "prompt_text_attn", "prompt_attention", "attn_prompt", "prompt", "prompt_text", "text_prompt_attn"],
    "gen_attn": ["gen_attn", "generated_attn", "generated_text_attn", "generated_attention", "attn_generated", "generated", "gen", "text_generated_attn"],
    "problem": ["problem", "question", "problem_text", "question_text"],
}
FUZZY = {"image_attn": ["image", "img"], "prompt_attn": ["prompt"], "gen_attn": ["gen"], "logp": ["logp"], "kl_u_p": ["kl"], "token_ids": ["token"]}


class AlignmentError(AssertionError):
    pass


def resolve_column(df: pd.DataFrame, role: str, override: Optional[str] = None, exclude: Iterable[str] = (), required: bool = True) -> Optional[str]:
    cols = [c for c in df.columns if c not in set(exclude)]
    if override:
        if override in df.columns:
            return override
        raise KeyError(f"configured column '{override}' for role '{role}' not found; columns: {list(df.columns)}")
    for c in COLUMN_CANDIDATES.get(role, []):
        if c in cols:
            return c
    for kw in FUZZY.get(role, []):
        hits = [c for c in cols if kw in c.lower()]
        if len(hits) == 1:
            return hits[0]
    if required:
        raise KeyError(f"cannot resolve column for role '{role}'. Available columns: {list(df.columns)}. "
                       f"Set it under `columns:` in configs/task3.yaml.")
    return None


# --------------------------------------------------------------------------------------
# README / directory discovery
# --------------------------------------------------------------------------------------
def read_readme(path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def parse_readme_revision(text: str) -> Optional[str]:
    """Best-effort: the 40-hex commit hash listed for Qwen/Qwen2.5-3B-Instruct in README.MD."""
    for m in re.finditer(r"Qwen/Qwen2\.5-3B-Instruct", text):
        window = text[max(0, m.start() - 200): m.end() + 400]
        h = re.search(r"\b[0-9a-f]{40}\b", window)
        if h:
            return h.group(0)
    h = re.search(r"revision[^0-9a-f\n]{0,20}([0-9a-f]{40})", text, flags=re.I)
    return h.group(1) if h else None


def discover_model_dirs(data_root) -> Dict[str, Path]:
    """Find export folders under data_root. Folders with attention_metrics.parquet are the VLM / MathVista
    export; the other folder with generations.parquet is the LLM / MATH export."""
    out: Dict[str, Path] = {}
    for gen in sorted(Path(data_root).rglob("generations.parquet")):
        d = gen.parent
        key = "mathvista" if (d / "attention_metrics.parquet").exists() else "math"
        out.setdefault(key, d)
    return out


def find_readme(data_root) -> Optional[Path]:
    for name in ("README.MD", "README.md", "readme.md", "Readme.md"):
        hits = list(Path(data_root).rglob(name))
        if hits:
            return sorted(hits, key=lambda p: len(p.parts))[0]
    return None


# --------------------------------------------------------------------------------------
# Joined frame
# --------------------------------------------------------------------------------------
def _key_str(df: pd.DataFrame) -> pd.DataFrame:
    for k in KEYS:
        df[k] = df[k].astype(str)
    return df


def load_joined(model_dir, vision: bool, columns: Optional[dict] = None, keep_qids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read generations / probability_metrics (/ attention_metrics) and join on (dataset, sample_id, rollout_id).
    Returns a frame with standardised column names."""
    columns = columns or {}
    d = Path(model_dir)
    gen = pd.read_parquet(d / "generations.parquet")
    prob = pd.read_parquet(d / "probability_metrics.parquet")
    attn = pd.read_parquet(d / "attention_metrics.parquet") if vision else None
    return join_frames(gen, prob, attn, vision, columns, keep_qids)


def join_frames(gen: pd.DataFrame, prob: pd.DataFrame, attn: Optional[pd.DataFrame], vision: bool,
                columns: Optional[dict] = None, keep_qids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    columns = columns or {}
    gen = _key_str(gen.copy())
    prob = _key_str(prob.copy())
    if keep_qids is not None:
        keep = {str(q) for q in keep_qids}
        gen = gen[gen["sample_id"].isin(keep)]
        prob = prob[prob["sample_id"].isin(keep)]
    g_map = {r: resolve_column(gen, r, columns.get(r), exclude=KEYS) for r in ("text", "length", "answer")}
    g_map["token_ids"] = resolve_column(gen, "token_ids", columns.get("token_ids"), exclude=KEYS, required=False)
    answer_col = g_map["answer"]
    gold_excl = KEYS + [answer_col]
    g_map["gold"] = resolve_column(gen, "gold", columns.get("gold"), exclude=gold_excl)
    g_map["problem"] = resolve_column(gen, "problem", columns.get("problem"), exclude=KEYS, required=False)
    p_map = {r: resolve_column(prob, r, columns.get(r), exclude=KEYS) for r in ("logp", "kl_u_p")}
    if g_map["token_ids"] is None:  # may live in the probability table
        g_map["token_ids"] = None
        tok_in_prob = resolve_column(prob, "token_ids", columns.get("token_ids"), exclude=KEYS, required=False)
    else:
        tok_in_prob = None

    left = gen[KEYS + [c for c in dict.fromkeys(g_map.values()) if c]].rename(columns={v: k for k, v in g_map.items() if v})
    right_cols = KEYS + list(p_map.values()) + ([tok_in_prob] if tok_in_prob else [])
    right = prob[right_cols].rename(columns={**{v: k for k, v in p_map.items()}, **({tok_in_prob: "token_ids"} if tok_in_prob else {})})
    df = left.merge(right, on=KEYS, how="inner", validate="one_to_one")
    if len(df) != len(left) or len(df) != len(right):
        raise AlignmentError(f"join lost rows: generations={len(left)} probability={len(right)} joined={len(df)}")
    if vision:
        assert attn is not None
        attn = _key_str(attn.copy())
        if keep_qids is not None:
            attn = attn[attn["sample_id"].isin(keep)]
        a_map = {r: resolve_column(attn, r, columns.get(r), exclude=KEYS) for r in ("image_attn", "prompt_attn", "gen_attn")}
        if len(set(a_map.values())) != 3:
            raise KeyError(f"attention columns resolved ambiguously: {a_map}; columns={list(attn.columns)}")
        a = attn[KEYS + list(a_map.values())].rename(columns={v: k for k, v in a_map.items()})
        n0 = len(df)
        df = df.merge(a, on=KEYS, how="inner", validate="one_to_one")
        if len(df) != n0 or len(a) != n0:
            raise AlignmentError(f"attention join lost rows: before={n0} attention={len(a)} after={len(df)}")
    df["qid"] = df["sample_id"]
    return df.reset_index(drop=True)


def _len(x) -> int:
    return 0 if x is None else len(x)


def check_alignment(df: pd.DataFrame, vision: bool, strict: bool = True, n_rollouts: Optional[int] = 32) -> dict:
    """Assert per rollout len(token_ids) == length == len(logp) == len(kl_u_p) (== attention lengths)."""
    bad = []
    has_tok = "token_ids" in df.columns
    for row in df.itertuples(index=False):
        L = int(row.length)
        lens = {"length": L, "logp": _len(row.logp), "kl_u_p": _len(row.kl_u_p)}
        if has_tok:
            lens["token_ids"] = _len(row.token_ids)
        if vision:
            lens.update(image_attn=_len(row.image_attn), prompt_attn=_len(row.prompt_attn), gen_attn=_len(row.gen_attn))
        if len(set(lens.values())) != 1:
            bad.append({"sample_id": row.sample_id, "rollout_id": row.rollout_id, **lens})
    counts = df.groupby("qid").size()
    report = {
        "n_rows": int(len(df)),
        "n_questions": int(counts.shape[0]),
        "rollouts_per_question_min": int(counts.min()) if len(counts) else 0,
        "rollouts_per_question_max": int(counts.max()) if len(counts) else 0,
        "n_misaligned": len(bad),
        "examples": bad[:5],
        "checked_token_ids": bool(has_tok),
    }
    if strict and bad:
        raise AlignmentError(f"{len(bad)} misaligned rollouts, e.g. {bad[:3]}")
    if strict and n_rollouts is not None and (report["rollouts_per_question_min"] != n_rollouts or report["rollouts_per_question_max"] != n_rollouts):
        raise AlignmentError(f"expected {n_rollouts} rollouts per question, got {report['rollouts_per_question_min']}..{report['rollouts_per_question_max']}")
    return report


# --------------------------------------------------------------------------------------
# Pools
# --------------------------------------------------------------------------------------
@dataclass
class QuestionData:
    qid: str
    gid: np.ndarray                 # [R] int16 answer-group id (0 = NO_ANSWER)
    gcorrect: np.ndarray            # [G] bool
    group_answers: List[str]        # [G] representative (normalised) answers
    gold: str
    gold_raw: str
    T: np.ndarray                   # [R] generated-token counts
    L: np.ndarray                   # [R] mean sampled-token log-prob
    K: np.ndarray                   # [R] self-certainty
    V: np.ndarray                   # [R] visual engagement (NaN for text-only)
    tok_k: Optional[List[np.ndarray]] = None
    tok_v: Optional[List[np.ndarray]] = None
    texts: Optional[List[str]] = None
    raw_answers: Optional[List] = None
    rollout_ids: Optional[List] = None
    problem: Optional[str] = None

    @property
    def n_rollouts(self) -> int:
        return len(self.gid)

    @property
    def correct(self) -> np.ndarray:
        return self.gcorrect[self.gid]


@dataclass
class Pool:
    dataset: str
    has_vision: bool
    questions: Dict[str, QuestionData]
    scorer: dict = field(default_factory=dict)
    alignment: dict = field(default_factory=dict)
    token_cap: int = 2048

    def qids(self) -> List[str]:
        return sorted(self.questions)

    def subset(self, qids: Iterable[str]) -> "Pool":
        return Pool(self.dataset, self.has_vision, {q: self.questions[q] for q in qids}, self.scorer, self.alignment, self.token_cap)


def _rollout_sort_key(values: pd.Series):
    num = pd.to_numeric(values, errors="coerce")
    if num.notna().all():
        return num.to_numpy()
    return values.astype(str).to_numpy()


def build_pool(df: pd.DataFrame, dataset: str, ctx, vision: bool, keep_text: bool = True, token_cap: int = 2048,
               alignment: Optional[dict] = None) -> Pool:
    questions: Dict[str, QuestionData] = {}
    for qid, g in df.groupby("qid", sort=False):
        g = g.iloc[np.argsort(_rollout_sort_key(g["rollout_id"]), kind="stable")]
        logps = [np.asarray(x, dtype=np.float32) for x in g["logp"]]
        kls = [np.asarray(x, dtype=np.float32) for x in g["kl_u_p"]]
        T = np.array([len(x) for x in logps], dtype=np.int64)
        tok_v = None
        if vision:
            tok_v = [S.visual_engagement_tokens(i, p, o).astype(np.float32)
                     for i, p, o in zip(g["image_attn"], g["prompt_attn"], g["gen_attn"])]
        L = np.empty(len(g))
        K = np.empty(len(g))
        V = np.full(len(g), np.nan)
        for i in range(len(g)):
            sc = S.rollout_scalars(logps[i], kls[i], tok_v[i] if tok_v is not None else None)
            L[i], K[i] = sc["L"], sc["K"]
            if vision:
                V[i] = sc["V"]
        meta = ctx.meta_for(str(qid))
        gold_raw = g["gold"].iloc[0]
        grp = A.build_question_groups(ctx, g["answer"].tolist(), gold_raw, meta)
        problem = g["problem"].iloc[0] if "problem" in g.columns else None
        questions[str(qid)] = QuestionData(
            qid=str(qid), gid=grp["gid"], gcorrect=grp["gcorrect"], group_answers=grp["reps"], gold=grp["gold"],
            gold_raw=str(gold_raw), T=T, L=L, K=K, V=V,
            tok_k=kls if vision else None, tok_v=tok_v,
            texts=g["text"].astype(str).tolist() if keep_text else None,
            raw_answers=g["answer"].tolist(), rollout_ids=g["rollout_id"].tolist(),
            problem=None if problem is None or (isinstance(problem, float) and np.isnan(problem)) else str(problem),
        )
    return Pool(dataset, vision, questions, getattr(ctx, "scorer_info", {}), alignment or {}, token_cap)


def describe_pool(pool: Pool) -> dict:
    """Headline statistics (rollout accuracy, NO_ANSWER rate, length, cap hits, coverage@32)."""
    qs = list(pool.questions.values())
    corr = np.stack([q.correct for q in qs])
    gid = np.stack([q.gid for q in qs])
    T = np.stack([q.T for q in qs])
    return {
        "dataset": pool.dataset,
        "n_questions": len(qs),
        "rollouts_per_question": int(corr.shape[1]),
        "mean_rollout_accuracy": float(corr.mean()),
        "oracle_coverage_all_rollouts": float(corr.any(axis=1).mean()),
        "no_answer_rate": float((gid == 0).mean()),
        "mean_tokens": float(T.mean()),
        "frac_at_token_cap": float((T >= pool.token_cap).mean()),
        "mean_distinct_groups": float(np.mean([len(np.unique(q.gid)) for q in qs])),
        "scorer": pool.scorer,
    }


def save_pool(pool: Pool, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pool(path) -> Pool:
    with open(path, "rb") as f:
        return pickle.load(f)
