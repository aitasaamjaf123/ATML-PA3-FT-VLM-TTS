"""Step 6 activation diagnostics (FP32).

1. Run the STARTING model on the probe set (32 GSM8K test prompts with the non-thinking chat template, 32 HellaSwag
   contexts as plain text) and cache the down_proj INPUT z at the last prompt token for layers 0, 13, 27.
2. For every condition apply its local update to those same cached z:  R(z) = ||dW z|| / (||W0 z|| + 1e-8).

Outputs (results_root/diagnostics/):
    cached_activations.pt         ids + z tensors (reused across conditions / sessions)
    activation_R_per_prompt.csv   condition, layer, domain, prompt_id, R, norm_dWz, norm_W0z
    activation_R_summary.csv      mean / median / p90 / max of R per (condition, layer, domain) + nonmath/math ratio
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from .eval_utils import (DEVICE, DIAG_CONDITIONS, DIAG_LAYERS, Ctx, DeltaSource, build_probe_prompts, free_gpu,
                         load_base, load_tokenizer, module_path, patch_chat_template, resolve_conditions,
                         select_conditions, upsert_csv)


@torch.no_grad()
def cache_activations(prompts: dict, tok) -> dict:
    model = load_base(torch.float32)
    store = {}
    cur = {}
    mods = dict(model.named_modules())
    handles = [mods[module_path(l, "down_proj")].register_forward_pre_hook(
        lambda m, inp, l=l: cur.__setitem__(l, inp[0][0, -1].detach().float().cpu())) for l in DIAG_LAYERS]
    for dom, items in prompts.items():
        store[dom] = dict(ids=[i for i, _ in items], z={l: [] for l in DIAG_LAYERS})
        for _, text in items:
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)  # 1 prompt at a time:
            model(ids, use_cache=False)                                                          # last token = last real one
            for l in DIAG_LAYERS:
                store[dom]["z"][l].append(cur[l])
        store[dom]["z"] = {l: torch.stack(v) for l, v in store[dom]["z"].items()}
    for h in handles:
        h.remove()
    del model
    free_gpu()
    return store


@torch.no_grad()
def R_rows(cond, src, ctx, store):
    rows = []
    for l in DIAG_LAYERS:
        W0 = ctx.start.get(l, "down_proj").to(DEVICE).float()
        dW = src.delta(l, "down_proj")
        for dom, d in store.items():
            Z = d["z"][l].to(DEVICE)
            num, den = (Z @ dW.T).norm(dim=1), (Z @ W0.T).norm(dim=1)
            R = num / (den + 1e-8)
            for pid, r, n, dd in zip(d["ids"], R.tolist(), num.tolist(), den.tolist()):
                rows.append(dict(condition=cond, layer=l, domain=dom, prompt_id=pid, R=r, norm_dWz=n, norm_W0z=dd))
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["condition", "layer", "domain"])["R"]
    s = g.agg(mean="mean", median="median", p90=lambda x: x.quantile(0.9), max="max", n="count").reset_index()
    piv = s.pivot_table(index=["condition", "layer"], columns="domain", values="mean")
    ratio = (piv["hellaswag"] / piv["gsm8k"]).rename("nonmath_over_math_mean_R").reset_index()
    return s.merge(ratio, on=["condition", "layer"], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root")
    ap.add_argument("--conditions_file")
    ap.add_argument("--conditions", nargs="*", default=DIAG_CONDITIONS)
    ap.add_argument("--results_root", default="task1_llm_finetuning/results")
    ap.add_argument("--probe_json", default="data/splits/task1/probe_set.json")
    a = ap.parse_args()

    out = Path(a.results_root) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    paths = resolve_conditions(a.run_root, a.conditions_file)
    conds = select_conditions(a.conditions, paths)
    ctx = Ctx(bases_cache=str(out / "svd_bases_k128.pt"))

    cache = out / "cached_activations.pt"
    if cache.exists():
        store = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"loaded cached activations from {cache}")
    else:
        tok = patch_chat_template(load_tokenizer())
        store = cache_activations(build_probe_prompts(a.probe_json, tok), tok)
        torch.save(store, cache)
    print({d: len(v["ids"]) for d, v in store.items()}, "prompts per domain")

    rows = []
    for c in conds:
        src = DeltaSource(c, paths[c], ctx)
        rows += R_rows(c, src, ctx, store)
        del src
    ctx.bases.save()
    df = upsert_csv(out / "activation_R_per_prompt.csv", pd.DataFrame(rows))
    s = summarize(df)
    s.to_csv(out / "activation_R_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
