"""Step 6 weight-space diagnostics (FP32).

down_proj in layers 0, 13, 27 for the five final models and Full FT midpoint:
    rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2      (protected OUTPUT subspace; the assignment's definition)
    eta   = ||dW||_F / ||W0||_F
plus, as extras: rho_in_k (same ratio on the INPUT side with V_k) and the random-subspace reference k/d.
Also the full-model displacement ||theta - theta0||_2 for the three full-parameter conditions.

Outputs (results_root/diagnostics/): weight_space.csv, full_model_displacement.csv, svd_bases_k128.pt
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from .eval_utils import (CONDITIONS, DEVICE, DIAG_CONDITIONS, DIAG_LAYERS, PROTECT_K, Ctx, DeltaSource,
                         normalize_keys, load_tensors, resolve_conditions, select_conditions, upsert_csv)


@torch.no_grad()
def weight_rows(cond, src, ctx):
    rows = []
    for l in DIAG_LAYERS:
        W0 = ctx.start.get(l, "down_proj").to(DEVICE).float()
        dW = src.delta(l, "down_proj")
        Uk, Vk = ctx.bases.get(l, "down_proj")
        fro = dW.norm()
        nz = fro.item() > 0
        rows.append(dict(
            condition=cond, layer=l, module="down_proj", k=PROTECT_K,
            shape=f"{tuple(dW.shape)}", fro_dW=fro.item(), fro_W0=W0.norm().item(),
            eta=(fro / W0.norm()).item(),
            rho_k=((Uk @ (Uk.T @ dW)).norm() ** 2 / fro ** 2).item() if nz else float("nan"),
            rho_in_k=(((dW @ Vk) @ Vk.T).norm() ** 2 / fro ** 2).item() if nz else float("nan"),
            rho_random_ref_out=PROTECT_K / dW.shape[0], rho_random_ref_in=PROTECT_K / dW.shape[1]))
    return rows


@torch.no_grad()
def displacement_row(cond, path, ctx):
    sd = normalize_keys(load_tensors(path))
    sq = ref = 0.0
    missing = 0
    for n in ctx.start.param_names:  # unique parameters; tied embedding counted once
        if n not in sd:
            missing += 1
            continue
        w0 = ctx.start.sd[n].float()
        sq += (sd[n].float() - w0).pow(2).sum().double().item()
        ref += w0.pow(2).sum().double().item()
    return dict(condition=cond, l2_displacement=sq ** 0.5, theta0_norm=ref ** 0.5,
                relative_displacement=(sq / ref) ** 0.5, n_params_missing_in_file=missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root")
    ap.add_argument("--conditions_file")
    ap.add_argument("--conditions", nargs="*", default=DIAG_CONDITIONS)
    ap.add_argument("--results_root", default="task1_llm_finetuning/results")
    a = ap.parse_args()

    out = Path(a.results_root) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    paths = resolve_conditions(a.run_root, a.conditions_file)
    conds = select_conditions(a.conditions, paths)
    ctx = Ctx(bases_cache=str(out / "svd_bases_k128.pt"))

    rows, disp = [], []
    for c in conds:
        src = DeltaSource(c, paths[c], ctx)
        rows += weight_rows(c, src, ctx)
        if CONDITIONS[c]["kind"] == "full":
            disp.append(displacement_row(c, paths[c], ctx))
        del src
    ctx.bases.save()
    df = upsert_csv(out / "weight_space.csv", pd.DataFrame(rows))
    dd = upsert_csv(out / "full_model_displacement.csv", pd.DataFrame(disp))
    print(df[["condition", "layer", "eta", "rho_k", "rho_in_k"]].to_string(index=False))
    print(dd.to_string(index=False))


if __name__ == "__main__":
    main()
