"""Final test evaluation of the Stage-3 model (greedy, thinking disabled, 16 new tokens, batch 8).

Usage: python -m task2_vlm_grounding.src.evaluate_final [--smoke] [--ckpt PATH]
The checkpoint is never selected using these results: the completed Stage-3 model is the final model.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common.io import write_csv, write_json, write_jsonl
from common.logging_utils import get_logger, peak_gpu_memory_mb, reset_peak_gpu_memory
from common.seed import set_seed
from task2_vlm_grounding.src.bootstrap import summarize
from task2_vlm_grounding.src.cache_features import FeatureStore
from task2_vlm_grounding.src.config import get_paths, load_config
from task2_vlm_grounding.src.evaluate_dev import generate_items, score_nll
from task2_vlm_grounding.src.manifest import load_manifest
from task2_vlm_grounding.src.qualitative import dump_qualitative
from task2_vlm_grounding.src.sampler import caption_to_example
from task2_vlm_grounding.src.vlm import SigLipQwenVLM
from task2_vlm_grounding.src.vqa_eval import CONDITIONS, eval_items, score_item

log = get_logger("evaluate_final")
SUITES = (("grounding_test", "grounding"), ("vqa_test", "vqa"))


def load_final_model(cfg: dict, paths, ckpt: str | None, device) -> SigLipQwenVLM:
    vlm = SigLipQwenVLM.from_pretrained(cfg, device)
    path = ckpt or str(paths.checkpoints / "stage3_final.pt")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    vlm.connector.load_state_dict(blob["connector"])
    if blob.get("lora") is not None:
        vlm.enable_lora(cfg["seed"] + 2)
        vlm.load_lora_state_dict(blob["lora"])
    vlm.eval_mode()
    log.info("Loaded %s (update %s)", path, blob.get("global_update"))
    return vlm


def evaluate_suite(vlm, store, pairs, suite: str, cfg: dict) -> tuple[list[dict], dict]:
    """Rows for every example x condition. Mismatch mapping is the one fixed in the manifest."""
    bs, max_new = cfg["eval"]["batch_size"], cfg["eval"]["max_new_tokens"]
    items = eval_items(pairs, suite)
    rows: list[dict] = []
    timing: dict = {}
    for cond in CONDITIONS:
        t0 = time.perf_counter()
        preds, ms = generate_items(vlm, store, items, cond, bs, max_new)
        nll = score_nll(vlm, store, items, cond, bs)
        timing[cond] = {"generation_ms_per_answer": float(np.mean(ms)),
                        "wall_s_gen_plus_nll": time.perf_counter() - t0, "n": len(items)}
        for it, pred, t_ms, n in zip(items, preds, ms, nll):
            sc = score_item(it, pred)
            ex = it["example"]
            rows.append({
                "suite": suite, "condition": cond, "pair_id": it["pair_id"], "idx": it["idx"],
                "example_id": ex["example_id"], "qtype": it["qtype"], "question": ex["question"],
                "reference": ex["answer"], "canonical": ex["canonical"],
                "all_answers": ex["all_answers"], "partner_reference": it["partner_answer"],
                "partner_canonical": it["partner_canonical"],
                "image_id_used": (None if cond == "no_image" else
                                  it[{"correct": "image_correct", "complementary": "image_complementary",
                                      "mismatched": "image_mismatched"}[cond]]),
                "image_correct": it["image_correct"], "image_complementary": it["image_complementary"],
                "prediction": pred, "pred_norm": sc["pred_norm"], "score": sc["score"], "exact": sc["exact"],
                "empty": sc["empty"], "partner_follow": sc["partner_follow"], "nll": float(n),
                "gen_ms": float(t_ms),
            })
        log.info("%s/%s: acc=%.3f nll=%.3f (%.1f ms/answer)", suite, cond,
                 np.mean([r["score"] for r in rows if r["condition"] == cond]),
                 float(np.mean(nll)), timing[cond]["generation_ms_per_answer"])
    return rows, timing


def caption_samples(vlm, store, manifest: dict, batch_size: int = 8, n: int = 8, max_new: int = 48) -> list[dict]:
    """Qualitative dev-caption generations (development images, not part of the final-test numbers)."""
    exs = [caption_to_example(e) for e in sorted(manifest["caption_dev"], key=lambda e: e["image_id"])[:n]]
    items = [{"example": e, "image_correct": e["image_id"], "image_complementary": None,
              "image_mismatched": None, "pair_id": e["example_id"]} for e in exs]
    preds, _ = generate_items(vlm, store, items, "correct", batch_size, max_new)
    return [{"image_id": e["image_id"], "reference": e["answer"], "generated": p} for e, p in zip(exs, preds)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    paths = get_paths(cfg)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(paths.splits)
    store = FeatureStore(paths.features / "siglip_features_fp16.pt")
    vlm = load_final_model(cfg, paths, args.ckpt, device)

    reset_peak_gpu_memory()
    t0 = time.perf_counter()
    all_rows: list[dict] = []
    timing: dict = {}
    for key, suite in SUITES:
        rows, tm = evaluate_suite(vlm, store, manifest[key], suite, cfg)
        all_rows += rows
        timing[suite] = tm
    total_s = time.perf_counter() - t0

    out = paths.final_dir
    write_jsonl(all_rows, out / "per_example_predictions.jsonl")
    write_csv([{k: v for k, v in r.items() if k != "all_answers"} for r in all_rows], out / "per_example_metrics.csv")
    summary = summarize(all_rows, cfg["eval"]["bootstrap_resamples"], cfg["seed"])
    write_json(summary, out / "final_summary.json")
    write_csv(summary["rows"], out / "final_summary.csv")
    dump_qualitative(all_rows, manifest, cfg, paths)
    write_json(caption_samples(vlm, store, manifest, cfg["eval"]["batch_size"]), paths.qual_dir / "dev_caption_samples.json")
    write_json({"timing": timing, "total_eval_wall_s": total_s, "peak_gpu_mb": peak_gpu_memory_mb(),
                "n_answers_generated": len(all_rows), "mean_generation_ms_per_answer":
                float(np.mean([r["gen_ms"] for r in all_rows]))}, out / "resources_eval.json")
    log.info("Final evaluation done in %.1fs -> %s", total_s, out)


if __name__ == "__main__":
    main()
