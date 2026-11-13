"""Build the deterministic split manifests (seed 6304) from the raw COCO / VQAv2 annotations.

Usage: python -m task2_vlm_grounding.src.build_manifest [--smoke]
Requires only the annotation JSON files (no images).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common.io import write_csv, write_json
from common.logging_utils import get_logger
from task2_vlm_grounding.src import data_coco, data_grounding, data_vqav2
from task2_vlm_grounding.src.config import get_paths, load_config
from task2_vlm_grounding.src.manifest import images_by_coco_split, write_manifest
from task2_vlm_grounding.src.split_checks import check_manifest

log = get_logger("build_manifest")


def sub_rng(seed: int, name: str) -> random.Random:
    """Independent, reproducible stream per component (string seeds are hash-randomisation safe)."""
    return random.Random(f"{seed}/{name}")


def assign_mismatch(pairs: list[dict], rng: random.Random) -> None:
    """Fix an unrelated image (from another pair of the same suite) for every example."""
    assert len(pairs) >= 2, "need >=2 pairs to draw mismatched images"
    for k, p in enumerate(pairs):
        own = {it["image_id"] for it in p["items"]}
        pool = sorted({it["image_id"] for j, q in enumerate(pairs) if j != k for it in q["items"]} - own)
        for it in p["items"]:
            it["mismatch_image_id"] = int(pool[rng.randrange(len(pool))])


def build(cfg: dict, raw_dir: Path, splits_dir: Path) -> dict:
    seed, d = cfg["seed"], cfg["data"]
    meta: dict = {"seed": seed, "run_name": cfg["run_name"], "smoke": cfg["smoke"], "vqa_filter_stats": {}}

    # ---------------- VQAv2 candidates ----------------
    vqa_cands = {}
    for split in ("train2014", "val2014"):
        log.info("Loading VQAv2 %s ...", split)
        q, a, pairs = data_vqav2.load_vqa_split(raw_dir, split)
        cands, stats = data_vqav2.build_candidate_pairs(q, a, pairs, split, d["min_annotator_agreement"])
        meta["vqa_filter_stats"][split] = stats
        log.info("  %s candidates: %s", split, stats)
        vqa_cands[split] = cands
        del q, a, pairs

    used_val: set[int] = set()  # global image-disjointness across every val-derived subset

    # ---------------- val: VQA test, VQA dev (most constrained first) ----------------
    vqa_test = data_vqav2.select_vqa_pairs(vqa_cands["val2014"], d["vqa_test"], sub_rng(seed, "vqa_test"), used_val)
    vqa_dev = data_vqav2.select_vqa_pairs(vqa_cands["val2014"], d["vqa_dev"], sub_rng(seed, "vqa_dev"), used_val)

    # ---------------- val: grounding test / dev ----------------
    log.info("Loading COCO instances (val2014) ...")
    cats, stats_val, areas_val = data_coco.load_instances(raw_dir, "val2014")
    idx_val = data_grounding.GroundingIndex(stats_val, cats, stats_val.keys())
    g_test, g_dev = [], []
    for kind in ("presence", "counting"):
        g_test += data_grounding.select_eval_pairs(idx_val, kind, d["grounding_test"][kind],
                                                   sub_rng(seed, f"grd_test_{kind}"), used_val, "test")
    for kind in ("presence", "counting"):
        g_dev += data_grounding.select_eval_pairs(idx_val, kind, d["grounding_dev"][kind],
                                                  sub_rng(seed, f"grd_dev_{kind}"), used_val, "dev")

    # ---------------- val: caption dev ----------------
    caps_val = data_coco.load_captions(raw_dir, "val2014")
    caption_dev = data_coco.select_caption_examples(
        caps_val, d["caption_dev_images"], d["caption_dev_per_image"], sub_rng(seed, "cap_dev"),
        "val2014", exclude=used_val, prompt=d["caption_prompt"])
    used_val |= {e["image_id"] for e in caption_dev}

    # ---------------- mismatch maps (test only) ----------------
    assign_mismatch(vqa_test, sub_rng(seed, "mismatch_vqa_test"))
    assign_mismatch(g_test, sub_rng(seed, "mismatch_grd_test"))

    # ---------------- train ----------------
    log.info("Loading COCO captions/instances (train2014) ...")
    caps_train = data_coco.load_captions(raw_dir, "train2014")
    caption_train = data_coco.select_caption_examples(
        caps_train, d["caption_train_images"], d["caption_train_per_image"], sub_rng(seed, "cap_train"),
        "train2014", prompt=d["caption_prompt"])
    cap_images = sorted({e["image_id"] for e in caption_train})
    _, stats_train, _ = data_coco.load_instances(raw_dir, "train2014")
    pool_rng = sub_rng(seed, "grd_pool")
    pool = list(cap_images)
    if d["grounding_pool_images"] > len(pool):
        pool_set = set(pool)
        extra = sorted(i for i, s in stats_train.items() if s and i not in pool_set)
        pool_rng.shuffle(extra)
        pool += extra[: d["grounding_pool_images"] - len(pool)]
    idx_train = data_grounding.GroundingIndex(stats_train, cats, pool)
    g_train_presence = data_grounding.generate_train_pairs(
        idx_train, "presence", d["grounding_train_presence_pairs"], sub_rng(seed, "grd_train_presence"))
    g_train_counting = data_grounding.generate_train_pairs(
        idx_train, "counting", d["grounding_train_counting_pairs"], sub_rng(seed, "grd_train_counting"))
    vqa_train = data_vqav2.select_vqa_pairs(vqa_cands["train2014"], d["vqa_train"], sub_rng(seed, "vqa_train"), None)

    manifest = {
        "caption_train": caption_train, "caption_dev": caption_dev,
        "grounding_train_presence": g_train_presence, "grounding_train_counting": g_train_counting,
        "grounding_dev": g_dev, "grounding_test": g_test,
        "vqa_train": vqa_train, "vqa_dev": vqa_dev, "vqa_test": vqa_test,
    }
    manifest["images"] = images_by_coco_split(manifest)
    meta["sizes"] = {k: len(v) for k, v in manifest.items() if k != "images"}
    meta["n_images"] = {k: len(v) for k, v in manifest["images"].items()}
    meta["grounding_train_pool_images"] = len(pool)
    meta["grounding_valid_categories"] = {
        "train_presence": len(idx_train.presence_categories()), "train_counting": len(idx_train.counting_categories()),
        "val_presence": len(idx_val.presence_categories()), "val_counting": len(idx_val.counting_categories()),
    }
    manifest["meta"] = meta

    report = check_manifest(manifest, cfg)  # raises on any violation
    meta["split_check"] = report
    write_manifest(manifest, splits_dir)

    # Label audit for evaluation grounding questions (independent of any model).
    write_csv(data_grounding.audit_rows(g_dev, stats_val, areas_val, cats), splits_dir / "grounding_label_audit_dev.csv")
    write_csv(data_grounding.audit_rows(g_test, stats_val, areas_val, cats), splits_dir / "grounding_label_audit_test.csv")
    log.info("Manifest written to %s: %s", splits_dir, meta["sizes"])
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    paths = get_paths(cfg)
    build(cfg, paths.raw, paths.splits)


if __name__ == "__main__":
    main()
