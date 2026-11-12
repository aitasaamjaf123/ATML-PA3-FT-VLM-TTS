"""Merge the results folders of the two Kaggle accounts into one (pandas only; no torch needed).

  python merge_accounts.py --a results_eval1/results_task1 --b results_eval2/results_task1 --out merged/results_task1

* per-condition folders (benchmarks/<cond>, training_logs/<run>) and qualitative/responses_<cond>.jsonl are copied
  (a condition present in BOTH inputs is an error: nothing is silently overwritten);
* diagnostics / merge_check CSVs are concatenated by `condition` (overlap is an error);
* benchmark tables (aggregate_results) and qualitative/responses_all.csv + annotation_subset_42.csv are regenerated.
Run it again after re-running any condition: point --a/--b at the newest outputs.
"""
import argparse, glob, os, shutil, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ALL = ["baseline", "full_ft_mid", "full_ft_final", "l2sp_final", "lora4_final", "lora16_final", "oplora16_final"]
CONCAT = ["diagnostics/weight_space.csv", "diagnostics/full_model_displacement.csv",
          "diagnostics/activation_R_per_prompt.csv", "diagnostics/activation_R_summary.csv", "merge_check/merge_check.csv"]
COPY_IF_PRESENT = ["merge_check/oplora_constraint_recheck.csv", "diagnostics/cached_activations.pt", "diagnostics/svd_bases_k128.pt"]


def collect_qualitative(qdir):
    files = sorted(f for f in glob.glob(f"{qdir}/responses_*.jsonl") if not f.endswith("responses_all.jsonl"))
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    df["cond_order"] = df["condition"].map({c: i for i, c in enumerate(ALL)})
    df["fmt_order"] = df["format"].map({"plain": 0, "json": 1, "template": 2})
    df = df.sort_values(["question_idx", "fmt_order", "cond_order"]).drop(columns=["cond_order", "fmt_order"])
    df.to_csv(f"{qdir}/responses_all.csv", index=False)
    sub = df[df.question_idx.isin([0, df.question_idx.max()])].copy()
    for c in ("format_compliance", "coherence_relevance", "repetition", "unnecessary_math_framing",
              "inappropriate_content", "notes", "excerpt"):
        sub[c] = ""
    sub.to_csv(f"{qdir}/annotation_subset_42.csv", index=False)
    print(f"qualitative: responses_all.csv {len(df)} rows; annotation_subset_42.csv {len(sub)} rows "
          f"({df.condition.nunique()} conditions)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ins, out = [a.a, a.b], a.out
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} exists and is not empty; choose a new folder")
    seen = set()
    for e in ins:
        for d in sorted(glob.glob(f"{e}/benchmarks/*/")):
            c = os.path.basename(d.rstrip("/"))
            if c in seen: sys.exit(f"condition {c} found in both inputs")
            seen.add(c); shutil.copytree(d, f"{out}/benchmarks/{c}")
        for f in glob.glob(f"{e}/qualitative/responses_*.jsonl"):
            os.makedirs(f"{out}/qualitative", exist_ok=True); shutil.copy(f, f"{out}/qualitative/")
        for d in glob.glob(f"{e}/training_logs/*/"):
            n = os.path.basename(d.rstrip("/"))
            if os.path.exists(f"{out}/training_logs/{n}"): sys.exit(f"training_logs/{n} in both inputs")
            shutil.copytree(d, f"{out}/training_logs/{n}")
    print("benchmark conditions:", sorted(seen))
    if set(ALL) - seen: print("!! missing conditions:", sorted(set(ALL) - seen))
    for rel in CONCAT:
        dfs = [pd.read_csv(f"{e}/{rel}") for e in ins if os.path.exists(f"{e}/{rel}")]
        if not dfs: print("!! missing", rel); continue
        conds = [set(d["condition"]) for d in dfs]
        if len(conds) == 2 and conds[0] & conds[1]: sys.exit(f"{rel}: overlapping conditions {conds[0] & conds[1]}")
        os.makedirs(os.path.dirname(f"{out}/{rel}"), exist_ok=True)
        pd.concat(dfs, ignore_index=True).to_csv(f"{out}/{rel}", index=False)
        print(f"{rel}: {sum(len(d) for d in dfs)} rows")
    for rel in COPY_IF_PRESENT:  # caches / single-account files: take the one from whichever input has it (b first)
        for e in reversed(ins):
            if os.path.exists(f"{e}/{rel}"):
                os.makedirs(os.path.dirname(f"{out}/{rel}"), exist_ok=True); shutil.copy(f"{e}/{rel}", f"{out}/{rel}"); break
    sys.path.insert(0, HERE)
    import importlib.util
    spec = importlib.util.spec_from_file_location("aggregate_results", f"{HERE}/aggregate_results.py")
    agg = importlib.util.module_from_spec(spec); spec.loader.exec_module(agg)
    sys.argv = ["aggregate_results", "--results_root", out]; agg.main()
    collect_qualitative(f"{out}/qualitative")


if __name__ == "__main__":
    main()
