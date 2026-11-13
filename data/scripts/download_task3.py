"""Download auxiliary Task 3 data (needs network + `datasets`):

  * MathVista testmini question metadata (question_type, answer_type, choices, precision, answer) -> required
    for MathVista answer canonicalisation / grading.
  * MATH problem text, only needed for the live generation subtask when the supplied generations.parquet has no
    problem/question column.

The supplied ATML-PA3-Data rollouts themselves are NOT downloaded here (they are provided with the assignment).

    python data/scripts/download_task3.py                      # metadata + MATH problems
    python data/scripts/download_task3.py --math-dataset HuggingFaceH4/MATH-500 --math-id-field unique_id
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def download_mathvista(out: Path) -> None:
    from datasets import load_dataset

    ds = load_dataset("AI4Math/MathVista", split="testmini")
    keep = ("pid", "question", "question_type", "answer_type", "choices", "precision", "answer", "unit")
    records = {}
    for r in ds:
        rec = {k: r.get(k) for k in keep}
        rec["pid"] = str(rec["pid"])
        rec["choices"] = list(rec["choices"]) if rec.get("choices") is not None else None
        records[rec["pid"]] = rec
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"MathVista testmini metadata: {len(records)} questions -> {out}")


def download_math_problems(out: Path, name: str, split: str, id_field: str, problem_field: str) -> None:
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    probs = {}
    for i, r in enumerate(ds):
        key = str(r[id_field]) if id_field in r and r[id_field] is not None else str(i)
        probs[key] = r[problem_field]
        probs.setdefault(str(i), r[problem_field])  # also index by position
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(probs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"MATH problems from {name}: {len(ds)} questions -> {out} (keys: {id_field} and row index)")
    print("NOTE: verify that these sample_ids match the `sample_id` values in generations.parquet "
          "(see README.MD of the supplied data). If generations.parquet has a problem column this file is unused.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "task3"))
    ap.add_argument("--skip-mathvista", action="store_true")
    ap.add_argument("--skip-math", action="store_true")
    ap.add_argument("--math-dataset", default="HuggingFaceH4/MATH-500")
    ap.add_argument("--math-split", default="test")
    ap.add_argument("--math-id-field", default="unique_id")
    ap.add_argument("--math-problem-field", default="problem")
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    if not a.skip_mathvista:
        download_mathvista(raw / "mathvista_testmini_metadata.json")
    if not a.skip_math:
        try:
            download_math_problems(raw / "math_problems.json", a.math_dataset, a.math_split, a.math_id_field, a.math_problem_field)
        except Exception as e:  # not fatal: problems may be inside generations.parquet
            print(f"[warn] could not fetch MATH problems ({e}). Not needed if generations.parquet contains the problem text.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
