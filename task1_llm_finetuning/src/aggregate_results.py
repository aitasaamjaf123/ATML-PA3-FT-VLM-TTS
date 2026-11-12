"""Collate per-condition benchmark outputs into tables (CSV only; plotting comes later).

results_root/benchmarks/
    benchmark_scores_long.csv      condition, task, metric, filter, value, stderr
    benchmark_primary_table.csv    one row per condition: primary + secondary scores (%), delta vs baseline (pp)
    gsm8k_extraction_stats.csv     per condition: strict/flexible EM, invalid-extraction rate, truncation rate,
                                   mean generated tokens, EM among extractable answers
"""
import argparse
from pathlib import Path

import pandas as pd

ALL_CONDITIONS = ["baseline", "full_ft_mid", "full_ft_final", "l2sp_final", "lora4_final", "lora16_final", "oplora16_final"]

# column -> (task, metric, filter)
COLS = {
    "gsm8k_flexible_em": ("gsm8k_cot", "exact_match", "flexible-extract"),
    "gsm8k_strict_em": ("gsm8k_cot", "exact_match", "strict-match"),
    "hellaswag_acc_norm": ("hellaswag", "acc_norm", "none"),
    "hellaswag_acc": ("hellaswag", "acc", "none"),
    "boolq_acc": ("boolq", "acc", "none"),
    "arc_easy_acc_norm": ("arc_easy", "acc_norm", "none"),
    "arc_easy_acc": ("arc_easy", "acc", "none"),
}
PRIMARY = ["gsm8k_flexible_em", "hellaswag_acc_norm", "boolq_acc", "arc_easy_acc_norm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="task1_llm_finetuning/results")
    a = ap.parse_args()
    base = Path(a.results_root) / "benchmarks"

    long = []
    for c in ALL_CONDITIONS:
        for f in sorted((base / c).glob("scores_*.csv")):
            d = pd.read_csv(f)
            d.insert(0, "task", f.stem.replace("scores_", ""))
            d.insert(0, "condition", c)
            long.append(d)
    long = pd.concat(long, ignore_index=True)
    long.to_csv(base / "benchmark_scores_long.csv", index=False)

    long["filter"] = long["filter"].fillna("none")
    rows = []
    for c in ALL_CONDITIONS:
        r = dict(condition=c)
        for col, (t, m, f) in COLS.items():
            v = long[(long.condition == c) & (long.task == t) & (long.metric == m) & (long["filter"] == f)]
            r[col] = 100 * v["value"].iloc[0] if len(v) else None
            r[col + "_stderr"] = 100 * v["stderr"].iloc[0] if len(v) and pd.notna(v["stderr"].iloc[0]) else None
        rows.append(r)
    tab = pd.DataFrame(rows)
    if tab.loc[tab.condition == "baseline", PRIMARY[0]].notna().any():
        b = tab[tab.condition == "baseline"].iloc[0]
        for col in COLS:
            tab[f"delta_pp_{col}"] = tab[col] - b[col]
    tab.to_csv(base / "benchmark_primary_table.csv", index=False)
    print(tab[["condition"] + PRIMARY].round(2).to_string(index=False))

    ex = []
    for c in ALL_CONDITIONS:
        f = base / c / "samples_gsm8k_cot.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        inv = d["extracted_flexible_extract"].astype(str).eq("[invalid]")
        em = d["exact_match_flexible_extract"].astype(float)
        ex.append(dict(condition=c, n=len(d), em_flexible=em.mean(), em_strict=d["exact_match_strict_match"].astype(float).mean(),
                       invalid_extraction_rate=inv.mean(), truncation_rate=d["truncated"].mean(),
                       mean_gen_tokens=d["n_gen_tokens"].mean(), em_among_extractable=em[~inv].mean(),
                       em_when_truncated=em[d["truncated"]].mean() if d["truncated"].any() else None))
    pd.DataFrame(ex).to_csv(base / "gsm8k_extraction_stats.csv", index=False)
    print(pd.DataFrame(ex).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
