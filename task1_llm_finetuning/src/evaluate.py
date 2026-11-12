"""Step 5: benchmark evaluation with lm-evaluation-harness (zero-shot, FP16, dropout off).

GSM8K : gsm8k_cot, non-thinking chat template, greedy, 1,024 new tokens, stock stop rules / filters.
Others: hellaswag, boolq, arc_easy, stock completion prompts, no chat template.

Resumable: each (condition, task) writes its own files; finished ones are skipped unless --force.

results_root/benchmarks/<condition>/
    scores_<task>.csv            metric, filter, value, stderr   (all metrics lm-eval reported)
    samples_<task>.csv(.gz)      per-example outputs (GSM8K: raw response, extracted answers, truncation flag)
    results_<task>.json          raw lm-eval results block
    first_prompt_<task>.txt      first rendered prompt (check the empty <think></think> block for GSM8K)
    timing.csv                   wall-clock per task
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from .eval_utils import (ALL_CONDITIONS, SEED, Ctx, build_eval_model, free_gpu, load_tokenizer, patch_chat_template,
                         resolve_conditions, select_conditions)

TASKS = ["gsm8k_cot", "hellaswag", "boolq", "arc_easy"]
GEN_CAP = 1024
TRUNC_TOKENS = 1020  # >= this many generated tokens (re-tokenised) is flagged as hitting the 1,024 cap


def _first(x):
    while isinstance(x, (list, tuple)):
        if not x:
            return None
        x = x[0]
    return x


def score_rows(res_block: dict) -> pd.DataFrame:
    rows = {}
    for k, v in res_block.items():
        if not isinstance(v, (int, float)):
            continue
        name, _, filt = k.partition(",")
        stderr = name.endswith("_stderr")
        name = name[:-7] if stderr else name
        r = rows.setdefault((name, filt), dict(metric=name, filter=filt, value=None, stderr=None))
        r["stderr" if stderr else "value"] = float(v)
    return pd.DataFrame(rows.values())


def gsm8k_samples(samples, tok) -> pd.DataFrame:
    by = {}
    for s in samples:
        d = by.setdefault(s["doc_id"], dict(doc_id=s["doc_id"], target=s.get("target")))
        f = str(s.get("filter", "none")).replace("-", "_")
        d["raw_response"] = _first(s["resps"])
        d[f"extracted_{f}"] = _first(s["filtered_resps"])
        d[f"exact_match_{f}"] = s.get("exact_match")
    df = pd.DataFrame(by.values()).sort_values("doc_id")
    df["n_gen_tokens"] = [len(tok(str(r), add_special_tokens=False).input_ids) for r in df["raw_response"]]
    df["truncated"] = df["n_gen_tokens"] >= TRUNC_TOKENS
    return df


def loglik_samples(samples) -> pd.DataFrame:
    rows = []
    for s in samples:
        lls = [_first(r) for r in s["resps"]]
        rows.append(dict(doc_id=s["doc_id"], target=s.get("target"),
                         acc=s.get("acc"), acc_norm=s.get("acc_norm"),
                         pred_raw=max(range(len(lls)), key=lambda i: lls[i]) if lls else None,
                         loglikelihoods=json.dumps([float(x) for x in lls])))
    return pd.DataFrame(rows).sort_values("doc_id")


def run_task(model, tok, task, bs, limit):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs)
    # NOTE: HFLM.max_gen_toks is hard-coded to 256 and a constructor argument is silently ignored;
    # the cap is only changed through gen_kwargs (merged into the task's generation_kwargs).
    gen_kwargs = f"max_gen_toks={GEN_CAP}" if task == "gsm8k_cot" else None
    return simple_evaluate(model=lm, tasks=[task], num_fewshot=0, log_samples=True, limit=limit,
                           apply_chat_template=(task == "gsm8k_cot"), gen_kwargs=gen_kwargs,
                           random_seed=SEED, numpy_random_seed=SEED, torch_random_seed=SEED,
                           fewshot_random_seed=SEED)


def run_task_safe(model, tok, task, bs, limit):
    """Halve the batch size and retry on CUDA OOM (BoolQ passages are long and logits are 152k-wide)."""
    while True:
        try:
            return run_task(model, tok, task, bs, limit)
        except torch.cuda.OutOfMemoryError:
            free_gpu()
            if bs == "auto" or int(bs) <= 1:
                raise
            bs = max(1, int(bs) // 2)
            print(f"[oom] {task}: retrying with batch size {bs}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root")
    ap.add_argument("--conditions_file")
    ap.add_argument("--conditions", nargs="*", default=["all"], help=f"subset of {ALL_CONDITIONS}")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--results_root", default="task1_llm_finetuning/results")
    ap.add_argument("--bs_gen", type=int, default=32, help="GSM8K generation batch size")
    ap.add_argument("--bs_ll", default="32", help="likelihood-task batch size (int or 'auto')")
    ap.add_argument("--limit", type=int, default=None, help="DEBUG ONLY: first N docs per task")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    paths = resolve_conditions(a.run_root, a.conditions_file)
    conds = select_conditions(a.conditions, paths)
    base = Path(a.results_root) / ("benchmarks_debug" if a.limit else "benchmarks")
    ctx = Ctx(bases_cache=str(Path(a.results_root) / "diagnostics" / "svd_bases_k128.pt"))
    tok = patch_chat_template(load_tokenizer())

    for cond in conds:
        out = base / cond
        out.mkdir(parents=True, exist_ok=True)
        todo = [t for t in a.tasks if a.force or not (out / f"scores_{t}.csv").exists()]
        if not todo:
            print(f"[skip] {cond}: all requested tasks done")
            continue
        print(f"\n=== {cond}: {todo}  ({paths.get(cond)})")
        model = build_eval_model(cond, paths.get(cond), ctx)
        for task in todo:
            t0 = time.time()
            bs = a.bs_gen if task == "gsm8k_cot" else (a.bs_ll if a.bs_ll == "auto" else int(a.bs_ll))
            if task == "boolq" and bs != "auto":
                bs = min(bs, 8)  # long passages: batch 32 needs ~10 GB of logits on a 16 GB card
            res = run_task_safe(model, tok, task, bs, a.limit)
            samples = res["samples"][task]
            score_rows(res["results"][task]).to_csv(out / f"scores_{task}.csv", index=False)
            (out / f"results_{task}.json").write_text(json.dumps(res["results"][task], indent=1, default=str))
            (out / f"first_prompt_{task}.txt").write_text(str(samples[0].get("arguments"))[:4000])
            if task == "gsm8k_cot":
                gdf = gsm8k_samples(samples, tok)
                gdf.to_csv(out / f"samples_{task}.csv", index=False)
                spike = gdf.n_gen_tokens.between(254, 258).mean()
                print(f"[check] GSM8K generated tokens: max={gdf.n_gen_tokens.max()}, share at 254-258={spike:.3f}, "
                      f"share >={TRUNC_TOKENS} (truncated at cap)={gdf.truncated.mean():.3f}", flush=True)
                if gdf.n_gen_tokens.max() < 300 and spike > 0.05:
                    print("!! WARNING: responses pile up at ~256 tokens: the 1,024-token cap is NOT in effect !!", flush=True)
            else:
                loglik_samples(samples).to_csv(out / f"samples_{task}.csv.gz", index=False)
            secs = time.time() - t0
            pd.DataFrame([dict(condition=cond, task=task, seconds=secs, limit=a.limit)]).to_csv(
                out / "timing.csv", mode="a", header=not (out / "timing.csv").exists(), index=False)
            print(f"[done] {cond}/{task} in {secs / 60:.1f} min")
        del model
        free_gpu()


if __name__ == "__main__":
    main()
