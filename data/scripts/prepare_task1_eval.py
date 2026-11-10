"""Prepare Task 1 EVALUATION inputs LOCALLY (run once, commit data/splits/task1/probe_set.json).

1. Samples the activation-probe set (32 GSM8K test + 32 HellaSwag validation, seed 6304) -> probe_set.json
2. Optionally warms a local HF-datasets cache with the 4 lm-eval tasks (gsm8k_cot, hellaswag, boolq, arc_easy),
   so evaluation can run offline on Kaggle:   python data/scripts/prepare_task1_eval.py --warm_cache

Usage (repo root):  python data/scripts/prepare_task1_eval.py [--warm_cache --cache_dir data/hf_cache]
Needs: pip install datasets lm-eval
(Put the course file at task1_llm_finetuning/prompts/first_plot_questions.yaml yourself.)
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--warm_cache", action="store_true")
ap.add_argument("--cache_dir", default=str(REPO_ROOT / "data/hf_cache"))
ap.add_argument("--out", default=str(REPO_ROOT / "data/splits/task1/probe_set.json"))
args = ap.parse_args()

if args.warm_cache:
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir  # must be set before importing datasets

from datasets import load_dataset  # noqa: E402

from common.io import save_json  # noqa: E402
from task1_llm_finetuning.src.probe_set import SEED, sample_ids  # noqa: E402

gsm = load_dataset("openai/gsm8k", "main", split="test")
hs = load_dataset("Rowan/hellaswag", split="validation")
assert len(gsm) == 1319 and len(hs) == 10042, (len(gsm), len(hs))

g_ids, h_ids = sample_ids(len(gsm)), sample_ids(len(hs))
save_json({
    "seed": SEED, "rule": "sorted(default_rng(6304).choice(N, 32, replace=False)) per dataset; idx = HF split order",
    "gsm8k_test": [{"idx": i, "question": gsm[i]["question"], "answer": gsm[i]["answer"]} for i in g_ids],
    "hellaswag_validation": [{"idx": i, "ind": hs[i]["ind"], "activity_label": hs[i]["activity_label"],
                              "ctx": hs[i]["ctx"], "endings": hs[i]["endings"], "label": hs[i]["label"]}
                             for i in h_ids],
}, args.out)
print("probe set saved:", args.out, "| gsm8k ids:", g_ids[:5], "... | hellaswag ids:", h_ids[:5], "...")

if args.warm_cache:
    from lm_eval.tasks import TaskManager, get_task_dict
    tasks = ["gsm8k_cot", "hellaswag", "boolq", "arc_easy"]
    td = get_task_dict(tasks, TaskManager())
    sizes = {}
    for name, t in td.items():
        t = getattr(t, "task", t)
        ds = t.dataset
        sizes[name] = {k: len(v) for k, v in ds.items()}
    print("lm-eval datasets cached in", args.cache_dir, "->", sizes)
    print("On Kaggle: upload this folder as a Dataset, copy it to /kaggle/working/hf_cache, then "
          "export HF_DATASETS_CACHE=/kaggle/working/hf_cache HF_DATASETS_OFFLINE=1")
