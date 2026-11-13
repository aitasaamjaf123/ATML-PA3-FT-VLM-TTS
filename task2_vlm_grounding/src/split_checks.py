"""Programmatic split-isolation and manifest sanity checks.

Usage: python -m task2_vlm_grounding.src.split_checks [--smoke]
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from common.io import write_json
from task2_vlm_grounding.src.config import get_paths, load_config
from task2_vlm_grounding.src.manifest import (EVAL_COMPONENTS, PAIR_COMPONENTS, TRAIN_COMPONENTS,
                                              component_images, images_by_coco_split, load_manifest)


def check_manifest(manifest: dict, cfg: dict | None = None) -> dict:
    """Raise AssertionError on any violation; return a report dict."""
    imgs = component_images(manifest)
    report: dict = {"component_image_counts": {k: len(v) for k, v in imgs.items()}}

    # 1. Eval subsets are pairwise image-disjoint.
    for a, b in itertools.combinations(EVAL_COMPONENTS, 2):
        inter = imgs[a] & imgs[b]
        assert not inter, f"Eval subsets {a} and {b} share {len(inter)} images, e.g. {sorted(inter)[:5]}"

    # 2. Eval subsets are disjoint from all training components (including replay pools).
    train_imgs = set().union(*(imgs[k] for k in TRAIN_COMPONENTS))
    eval_imgs = set().union(*(imgs[k] for k in EVAL_COMPONENTS))
    inter = train_imgs & eval_imgs
    assert not inter, f"{len(inter)} images shared by train and eval, e.g. {sorted(inter)[:5]}"
    report["n_train_images"] = len(train_imgs)
    report["n_eval_images"] = len(eval_imgs)

    # 3. COCO split provenance: training from train2014, evaluation from val2014.
    for key in TRAIN_COMPONENTS:
        splits = ({e["coco_split"] for e in manifest[key]} if key == "caption_train"
                  else {p["coco_split"] for p in manifest[key]})
        assert splits == {"train2014"}, f"{key} must be train2014 only, got {splits}"
    for key in EVAL_COMPONENTS:
        splits = ({e["coco_split"] for e in manifest[key]} if key == "caption_dev"
                  else {p["coco_split"] for p in manifest[key]})
        assert splits == {"val2014"}, f"{key} must be val2014 only, got {splits}"
    by_split = images_by_coco_split(manifest)
    assert not set(by_split["train2014"]) & set(by_split["val2014"])

    # 4. Pair structure.
    for key in PAIR_COMPONENTS:
        pair_ids = [p["pair_id"] for p in manifest[key]]
        assert len(pair_ids) == len(set(pair_ids)), f"duplicate pair ids in {key}"
        for p in manifest[key]:
            assert len(p["items"]) == 2, f"{p['pair_id']} must have two items"
            a, b = p["items"]
            assert a["image_id"] != b["image_id"], f"{p['pair_id']} uses the same image twice"
            assert a["canonical"] != b["canonical"], f"{p['pair_id']} has identical answers"
            assert a["canonical"] and b["canonical"], f"{p['pair_id']} has an empty canonical answer"
    for key in ("grounding_dev", "grounding_test", "vqa_dev", "vqa_test"):  # unique images within suite
        seen: set[int] = set()
        for p in manifest[key]:
            for it in p["items"]:
                assert it["image_id"] not in seen, f"{key}: image {it['image_id']} reused across pairs"
                seen.add(it["image_id"])

    # 5. Grounding label constraints.
    for key in ("grounding_train_presence", "grounding_dev", "grounding_test", "grounding_train_counting"):
        for p in manifest[key]:
            if p["kind"] == "presence":
                assert sorted(it["answer"] for it in p["items"]) == ["no", "yes"]
            else:
                assert all(it["answer"] in {"1", "2", "3", "4"} for it in p["items"])

    # 6. Mismatch mapping: from the same suite, never one of the pair's own images.
    for key in ("grounding_test", "vqa_test"):
        suite_imgs = {it["image_id"] for p in manifest[key] for it in p["items"]}
        for p in manifest[key]:
            own = {it["image_id"] for it in p["items"]}
            for it in p["items"]:
                m = it["mismatch_image_id"]
                assert m in suite_imgs and m not in own, f"bad mismatch mapping in {p['pair_id']}"

    # 7. Natural VQA: >=8/10 agreement on canonical answer.
    for key in ("vqa_train", "vqa_dev", "vqa_test"):
        from task2_vlm_grounding.src.data_vqav2 import normalize_vqa_answer
        for p in manifest[key]:
            for it in p["items"]:
                n = sum(normalize_vqa_answer(a) == it["canonical"] for a in it["all_answers"])
                assert n >= 8, f"{p['pair_id']} agreement {n} < 8"
                assert normalize_vqa_answer(it["answer"]) == it["canonical"]
                assert len(it["all_answers"]) == 10

    # 8. Requested sizes.
    if cfg is not None:
        d = cfg["data"]
        assert len(manifest["caption_train"]) == d["caption_train_images"] * d["caption_train_per_image"]
        assert len(manifest["caption_dev"]) == d["caption_dev_images"] * d["caption_dev_per_image"]
        assert len(manifest["grounding_dev"]) == d["grounding_dev"]["presence"] + d["grounding_dev"]["counting"]
        assert len(manifest["grounding_test"]) == d["grounding_test"]["presence"] + d["grounding_test"]["counting"]
        for key, cfgkey in (("vqa_train", "vqa_train"), ("vqa_dev", "vqa_dev"), ("vqa_test", "vqa_test")):
            for kind, n in d[cfgkey].items():
                got = sum(1 for p in manifest[key] if p["kind"] == kind)
                assert got == n, f"{key}/{kind}: {got} != {n}"
    report["status"] = "OK"
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    paths = get_paths(cfg)
    manifest = load_manifest(paths.splits)
    report = check_manifest(manifest, cfg)
    write_json(report, Path(paths.splits) / "split_check_report.json")
    print("Split checks passed:", report)


if __name__ == "__main__":
    main()
