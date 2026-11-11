"""Show which file each evaluation condition resolves to, and write an editable conditions.yaml.

  python -m task1_llm_finetuning.src.inspect_runs --run_root /content/drive/MyDrive/atml/task1_outputs \
      --write conditions.yaml
Edit conditions.yaml if a path is wrong / missing, then pass --conditions_file conditions.yaml everywhere.
--show_keys prints a few tensor key names of each resolved file (to debug adapter/weight key formats).
"""
import argparse
from pathlib import Path

import yaml

from .eval_utils import (ALL_CONDITIONS, CONDITIONS, candidate_files, find_run_dir, load_tensors,
                         resolve_conditions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--write", default=None)
    ap.add_argument("--show_keys", action="store_true")
    a = ap.parse_args()

    paths = resolve_conditions(a.run_root)
    for c in ALL_CONDITIONS:
        spec = CONDITIONS[c]
        print(f"\n== {c}")
        if spec["kind"] == "base":
            print("   (hub checkpoint)")
            continue
        print("   resolved:", paths[c])
        if paths[c] is None:
            rd = find_run_dir(Path(a.run_root), spec["run"])
            print("   run dir:", rd)
            if rd:
                for p in candidate_files(rd):
                    print(f"     candidate: {p.relative_to(rd)}  ({p.stat().st_size / 1e6:.1f} MB)")
        elif a.show_keys:
            ks = list(load_tensors(paths[c]).keys())
            print(f"   {len(ks)} tensors; first keys: {ks[:3]}")
    if a.write:
        yaml.safe_dump({c: (str(p) if p else None) for c, p in paths.items()}, open(a.write, "w"), sort_keys=False)
        print(f"\nwrote {a.write}")


if __name__ == "__main__":
    main()
