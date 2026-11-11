"""Step 5 (merge part): check that merging adapters into a fresh backbone reproduces the unmerged model.

For each adapter run (LoRA-4, LoRA-16, OPLoRA-16; OPLoRA uses projected factors) and for FP32 and FP16:
  logits(unmerged: base + hook adding (alpha/r) B~A~ x)  vs  logits(merged weights), on a small batch of texts.
Also re-verifies OPLoRA's constraints (U_k^T B~ = 0, A~ V_k = 0) on all 196 modules.

Outputs (results_root/merge_check/): merge_check.csv, oplora_constraint_recheck.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from .eval_utils import (CONDITIONS, DEVICE, Ctx, attach_unmerged_hooks, build_probe_prompts, free_gpu,
                         load_adapter, load_base, load_tokenizer, merge_adapter_into, project,
                         resolve_conditions, select_conditions, upsert_csv)

FALLBACK_TEXTS = ["Natalia sold clips to 48 of her friends in April. How many clips did she sell in total?",
                  "A man is sitting on a roof. He starts pulling up roofing on a roof.",
                  "Question: Is the sky blue on a clear day?\nAnswer:",
                  "The capital of France is"]


@torch.no_grad()
def _logits(model, id_list):
    return [model(ids.to(DEVICE)).logits.float().cpu() for ids in id_list]


@torch.no_grad()
def check_condition(cond, path, ctx, dtype, id_list):
    spec = CONDITIONS[cond]
    adapter = load_adapter(path, spec["r"])
    model = load_base(dtype)
    base = _logits(model, id_list)
    handles = attach_unmerged_hooks(model, adapter, spec, ctx)
    unmerged = _logits(model, id_list)
    for h in handles:
        h.remove()
    merge_adapter_into(model, adapter, spec, ctx)
    merged = _logits(model, id_list)
    del model
    free_gpu()
    d = torch.cat([(a - b).abs().flatten() for a, b in zip(unmerged, merged)])
    eff = torch.cat([(a - b).abs().flatten() for a, b in zip(base, merged)])
    agree = torch.cat([(a.argmax(-1) == b.argmax(-1)).flatten() for a, b in zip(unmerged, merged)]).float().mean()
    return dict(condition=cond, dtype=str(dtype).replace("torch.", ""), n_texts=len(id_list),
                n_positions=int(sum(x.shape[1] for x in merged)),
                max_abs_diff_merged_vs_unmerged=d.max().item(), mean_abs_diff_merged_vs_unmerged=d.mean().item(),
                argmax_agreement=agree.item(),
                max_abs_change_vs_base=eff.max().item(), mean_abs_change_vs_base=eff.mean().item())


@torch.no_grad()
def oplora_recheck(path, ctx):
    spec = CONDITIONS["oplora16_final"]
    rows = []
    for (l, name), ab in sorted(load_adapter(path, spec["r"]).items()):
        Uk, Vk = ctx.bases.get(l, name)
        At, Bt = project(ab["A"].to(DEVICE), ab["B"].to(DEVICE), Uk, Vk)
        rows.append(dict(layer=l, module=name,
                         max_abs_UkT_Btilde=(Uk.T @ Bt).abs().max().item(),
                         max_abs_Atilde_Vk=(At @ Vk).abs().max().item(),
                         max_abs_B=ab["B"].abs().max().item(), max_abs_A=ab["A"].abs().max().item(),
                         frac_B_removed=1 - (Bt.norm() / ab["B"].to(DEVICE).norm()).item(),
                         frac_A_removed=1 - (At.norm() / ab["A"].to(DEVICE).norm()).item()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root")
    ap.add_argument("--conditions_file")
    ap.add_argument("--conditions", nargs="*", default=["lora4_final", "lora16_final", "oplora16_final"])
    ap.add_argument("--results_root", default="task1_llm_finetuning/results")
    ap.add_argument("--probe_json", default="data/splits/task1/probe_set.json")
    ap.add_argument("--max_tokens", type=int, default=128)
    a = ap.parse_args()

    out = Path(a.results_root) / "merge_check"
    out.mkdir(parents=True, exist_ok=True)
    paths = resolve_conditions(a.run_root, a.conditions_file)
    conds = select_conditions(a.conditions, paths)
    tok = load_tokenizer()
    try:
        pp = build_probe_prompts(a.probe_json, tok)
        texts = [p for _, p in pp["gsm8k"][:3]] + [p for _, p in pp["hellaswag"][:3]]
    except Exception as e:  # lm-eval / probe file unavailable: still a valid algebra check
        print(f"[warn] probe prompts unavailable ({type(e).__name__}: {e}); using fixed texts")
        texts = FALLBACK_TEXTS
    id_list = [tok(t, return_tensors="pt", add_special_tokens=False).input_ids[:, :a.max_tokens] for t in texts]

    ctx = Ctx(bases_cache=str(Path(a.results_root) / "diagnostics" / "svd_bases_k128.pt"))
    rows = []
    for c in conds:
        for dt in (torch.float32, torch.float16):
            r = check_condition(c, paths[c], ctx, dt, id_list)
            print(r)
            rows.append(r)
    upsert_csv(out / "merge_check.csv", pd.DataFrame(rows))
    if "oplora16_final" in conds:
        df = oplora_recheck(paths["oplora16_final"], ctx)
        df.to_csv(out / "oplora_constraint_recheck.csv", index=False)
        print(df[["max_abs_UkT_Btilde", "max_abs_Atilde_Vk"]].max())
    ctx.bases.save()


if __name__ == "__main__":
    main()
