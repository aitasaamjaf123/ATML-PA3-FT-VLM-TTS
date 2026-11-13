"""Step 4: live comparison on the fixed 50-question MATH subset with an equal 3,072-token output allowance:
single longer response vs. 3 x 1,024 parallel (self-consistency) vs. 1,024 + 2,048 sequential refinement.

The generation backend is injectable: `HFBackend` (Qwen/Qwen2.5-3B-Instruct, FP16, batch size 1) for real runs
and any object with `.generate(messages, max_new_tokens, seed) -> GenResult` for tests."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common.io import append_jsonl, ensure_dir, load_jsonl, save_csv, save_json
from common.logging_utils import get_logger, gpu_info, package_versions
from common.seed import derive_seed
from common.stats import bootstrap_mean_ci
from task3_test_time_compute.src import answers as A
from task3_test_time_compute.src import selectors as SEL


log = get_logger("task3")


@dataclass
class GenResult:
    text: str
    input_tokens: int
    output_tokens: int          # generated tokens excluding the terminating EOS token
    truncated: bool             # hit max_new_tokens without terminating
    seconds: float
    peak_mem_mb: float
    prompt_text: str = ""


class HFBackend:
    """Qwen2.5-3B-Instruct, FP16, no quantisation, batch size 1, native chat template."""

    def __init__(self, model_id: str, revision: Optional[str], sampling: dict, dtype: str = "float16", device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for the live experiment (FP16, runtime/memory measurement).")
        self.torch = torch
        self.device = device
        self.sampling = sampling
        self.model_id, self.revision = model_id, revision
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        kwargs = dict(revision=revision)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), **kwargs)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype), **kwargs)
        self.model.to(device).eval()
        eos = set()
        gce = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if isinstance(gce, (list, tuple)):
            eos.update(int(e) for e in gce)
        elif gce is not None:
            eos.add(int(gce))
        if self.tok.eos_token_id is not None:
            eos.add(int(self.tok.eos_token_id))
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0:
            eos.add(im_end)
        self.eos_ids = eos
        self.warmup()

    def warmup(self):
        """One tiny untimed generation so CUDA init does not distort the first measured call."""
        self.generate([{"role": "user", "content": "Say hi."}], 8, seed=0)

    def generate(self, messages: List[dict], max_new_tokens: int, seed: int) -> GenResult:
        torch = self.torch
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        n_in = int(enc["input_ids"].shape[1])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **enc, do_sample=True, temperature=self.sampling["temperature"], top_p=self.sampling["top_p"],
                top_k=self.sampling["top_k"], repetition_penalty=self.sampling["repetition_penalty"],
                max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id,
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
        gen = out[0, n_in:].tolist()
        ended = bool(gen) and gen[-1] in self.eos_ids
        while gen and gen[-1] in self.eos_ids:
            gen.pop()
        n_out = len(gen)
        text = self.tok.decode(gen, skip_special_tokens=True)
        return GenResult(text=text, input_tokens=n_in, output_tokens=n_out, truncated=(not ended) and n_out >= max_new_tokens,
                         seconds=dt, peak_mem_mb=peak, prompt_text=prompt)


# --------------------------------------------------------------------------------------
def build_messages(problem: str, system_prompt: str, instruction: str) -> List[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"{problem}\n\n{instruction}"}]


def refinement_messages(base: List[dict], initial_response: str, review_prompt: str) -> List[dict]:
    return base + [{"role": "assistant", "content": initial_response}, {"role": "user", "content": review_prompt}]


def _call(backend, cache: Dict, jsonl_path: Path, qid: str, role: str, messages, max_new: int, seed: int) -> dict:
    """Run (or resume) one seeded generation and log everything needed to audit it."""
    key = (qid, role)
    if key in cache:
        return cache[key]
    r = backend.generate(messages, max_new, seed)
    ans = A.extract_last_boxed(r.text)
    row = {"qid": qid, "role": role, "seed": seed, "max_new_tokens": max_new, "messages": messages, "response": r.text,
           "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "truncated": bool(r.truncated),
           "seconds": r.seconds, "peak_mem_mb": r.peak_mem_mb, "extracted": ans}
    append_jsonl(row, jsonl_path)
    cache[key] = row
    log.info("[live] q=%s %-10s in=%d out=%d%s %.1fs", qid, role, r.input_tokens, r.output_tokens,
             " (truncated)" if r.truncated else "", r.seconds)
    return row


def run_live(backend, questions: Sequence[dict], cfg_live: dict, ctx, out_dir, base_seed: int = 6304) -> Dict[str, object]:
    """`questions`: dicts with qid, problem, gold. Returns summary tables and also writes them to out_dir."""
    out_dir = ensure_dir(out_dir)
    jsonl = out_dir / "generations.jsonl"
    cache = {(r["qid"], r["role"]): r for r in load_jsonl(jsonl)}
    sysmsg, instr, review = cfg_live["system_prompt"], cfg_live["instruction"], cfg_live["review_prompt"]
    k_par, par_max = int(cfg_live["parallel_k"]), int(cfg_live["parallel_max_new_tokens"])
    single_max, ref_max = int(cfg_live["single_max_new_tokens"]), int(cfg_live["refine_max_new_tokens"])
    rows = []
    for q in questions:
        qid, gold = str(q["qid"]), q["gold"]
        base = build_messages(q["problem"], sysmsg, instr)
        single = _call(backend, cache, jsonl, qid, "single", base, single_max, derive_seed(base_seed, qid, "single"))
        par = [_call(backend, cache, jsonl, qid, f"parallel_{k}", base, par_max, derive_seed(base_seed, qid, "parallel", k)) for k in range(k_par)]
        ref = _call(backend, cache, jsonl, qid, "refine", refinement_messages(base, par[0]["response"], review), ref_max,
                    derive_seed(base_seed, qid, "refine"))

        def grade(a):
            grp = A.build_question_groups(ctx, [a], gold, None)
            return bool(grp["gcorrect"][grp["gid"][0]])

        ans_single = single["extracted"]
        ans_par = [p["extracted"] for p in par]
        grp = A.build_question_groups(ctx, ans_par, gold, None)       # groups without gold, then graded
        sc_gid = SEL.vote_winner(grp["gid"])                           # ties -> earliest response
        sc_ok = bool(grp["gcorrect"][sc_gid])
        ans_ref = ref["extracted"]
        init_ok = grade(ans_par[0])
        ref_ok = grade(ans_ref)
        # deployable variant: keep the initial answer when the revision has no extractable answer (no gold used)
        ref_fb_ans = ans_ref if not A.is_missing(ans_ref) else ans_par[0]
        ref_fb_ok = grade(ref_fb_ans)
        same_answer = (not A.is_missing(ans_par[0]) and not A.is_missing(ans_ref) and bool(ctx.same(ctx.normalize(ans_par[0]), ctx.normalize(ans_ref))))
        rows.append({
            "qid": qid, "gold": gold,
            "single_answer": ans_single, "single_ok": grade(ans_single),
            "parallel_answers": ans_par, "parallel_oks": [grade(a) for a in ans_par], "parallel_sc_answer": grp["reps"][sc_gid], "parallel_sc_ok": sc_ok,
            "initial_answer": ans_par[0], "initial_ok": init_ok, "refined_answer": ans_ref, "refined_ok": ref_ok,
            "refined_fallback_ok": ref_fb_ok, "answer_changed": not same_answer,
            "transition": ("wrong->correct" if (not init_ok and ref_ok) else "correct->wrong" if (init_ok and not ref_ok)
                           else "correct->correct" if init_ok else "wrong->wrong"),
            # per-strategy accounting (the shared initial response is charged to both parallel and refinement)
            "single_in": single["input_tokens"], "single_out": single["output_tokens"], "single_sec": single["seconds"],
            "single_trunc": single["truncated"], "single_peak": single["peak_mem_mb"],
            "par_in": sum(p["input_tokens"] for p in par), "par_out": sum(p["output_tokens"] for p in par),
            "par_sec": sum(p["seconds"] for p in par), "par_trunc_calls": sum(bool(p["truncated"]) for p in par),
            "par_peak": max(p["peak_mem_mb"] for p in par),
            "ref_in": par[0]["input_tokens"] + ref["input_tokens"], "ref_out": par[0]["output_tokens"] + ref["output_tokens"],
            "ref_sec": par[0]["seconds"] + ref["seconds"], "ref_initial_trunc": bool(par[0]["truncated"]),
            "ref_revision_trunc": bool(ref["truncated"]), "ref_peak": max(par[0]["peak_mem_mb"], ref["peak_mem_mb"]),
            "ref_revision_in": ref["input_tokens"], "ref_revision_out": ref["output_tokens"],
            "single_extracted": not A.is_missing(ans_single), "par_extracted_calls": sum(not A.is_missing(a) for a in ans_par),
            "ref_extracted": not A.is_missing(ans_ref),
        })
    per_q = pd.DataFrame(rows)
    summary = summarize_live(per_q, k_par)
    trans = transition_summary(per_q)
    save_csv(per_q.assign(parallel_answers=per_q["parallel_answers"].astype(str), parallel_oks=per_q["parallel_oks"].astype(str)),
             out_dir / "per_question.csv")
    save_csv(summary, out_dir / "summary.csv")
    save_json(trans, out_dir / "transitions.json")
    save_json({"gpu": gpu_info(), "versions": package_versions(), "scorer": getattr(ctx, "scorer_info", {}),
               "settings": cfg_live, "n_questions": len(per_q), "base_seed": base_seed}, out_dir / "meta.json")
    return {"per_question": per_q, "summary": summary, "transitions": trans}


def summarize_live(per_q: pd.DataFrame, k_par: int = 3) -> pd.DataFrame:
    n = len(per_q)
    specs = [
        ("single_3072", "single_ok", "single_in", "single_out", "single_sec", "single_peak", per_q["single_trunc"].mean(), per_q["single_trunc"].mean(),
         1.0 - per_q["single_extracted"].mean(), 1.0 - per_q["single_extracted"].mean()),
        ("parallel_3x1024_sc", "parallel_sc_ok", "par_in", "par_out", "par_sec", "par_peak", per_q["par_trunc_calls"].sum() / (n * k_par),
         (per_q["par_trunc_calls"] > 0).mean(), 1.0 - per_q["par_extracted_calls"].sum() / (n * k_par), np.nan),
        ("refine_1024_plus_2048", "refined_ok", "ref_in", "ref_out", "ref_sec", "ref_peak",
         (per_q["ref_initial_trunc"].sum() + per_q["ref_revision_trunc"].sum()) / (2 * n), (per_q["ref_initial_trunc"] | per_q["ref_revision_trunc"]).mean(),
         1.0 - per_q["ref_extracted"].mean(), 1.0 - per_q["ref_extracted"].mean()),
    ]
    rows = []
    for name, okc, inc, outc, sec, peak, trunc_calls, trunc_q, noext_calls, noext_final in specs:
        ci = bootstrap_mean_ci(per_q[okc].astype(float).to_numpy(), n_boot=1000)
        rows.append({"strategy": name, "n_questions": n, "accuracy": ci["mean"], "acc_ci_low": ci["ci_low"], "acc_ci_high": ci["ci_high"],
                     "mean_input_tokens": per_q[inc].mean(), "mean_output_tokens": per_q[outc].mean(),
                     "mean_total_tokens": (per_q[inc] + per_q[outc]).mean(), "mean_seconds": per_q[sec].mean(),
                     "total_seconds": per_q[sec].sum(), "peak_gpu_mem_mb": per_q[peak].max(),
                     "truncation_rate_per_generation": trunc_calls, "frac_questions_with_truncation": trunc_q,
                     "no_extraction_rate_per_generation": noext_calls})
    rows.append({"strategy": "refine_fallback_to_initial_if_no_answer", "n_questions": n, "accuracy": float(per_q["refined_fallback_ok"].mean()),
                 "mean_input_tokens": per_q["ref_in"].mean(), "mean_output_tokens": per_q["ref_out"].mean(),
                 "mean_total_tokens": (per_q["ref_in"] + per_q["ref_out"]).mean(), "mean_seconds": per_q["ref_sec"].mean(),
                 "total_seconds": per_q["ref_sec"].sum()})
    rows.append({"strategy": "initial_response_only_1024", "n_questions": n, "accuracy": float(per_q["initial_ok"].mean()),
                 "mean_input_tokens": per_q["par_in"].mean() / k_par, "mean_output_tokens": per_q["par_out"].mean() / k_par})
    return pd.DataFrame(rows)


def transition_summary(per_q: pd.DataFrame) -> dict:
    from scipy.stats import binomtest

    counts = per_q["transition"].value_counts().to_dict()
    w2c, c2w = int(counts.get("wrong->correct", 0)), int(counts.get("correct->wrong", 0))
    out = {"counts": {k: int(counts.get(k, 0)) for k in ("wrong->correct", "correct->wrong", "correct->correct", "wrong->wrong")},
           "net_gain": w2c - c2w, "answer_changed": int(per_q["answer_changed"].sum()),
           "revision_without_extractable_answer": int((~per_q["ref_extracted"]).sum())}
    if w2c + c2w > 0:
        out["sign_test_p_refined_vs_initial"] = float(binomtest(w2c, w2c + c2w, 0.5).pvalue)
    # paired differences between strategies (discordant pairs)
    def disc(a, b):
        a, b = per_q[a].astype(bool), per_q[b].astype(bool)
        n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        return {"a_only_correct": n10, "b_only_correct": n01, "p": float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0}

    out["parallel_sc_vs_single"] = disc("parallel_sc_ok", "single_ok")
    out["refined_vs_single"] = disc("refined_ok", "single_ok")
    out["refined_vs_parallel_sc"] = disc("refined_ok", "parallel_sc_ok")
    return out
