"""Reading / writing the committed split manifests (data/splits/task2*/)."""
from __future__ import annotations

from pathlib import Path

from common.io import read_json, write_json

MANIFEST_FILES = {
    "caption_train": "caption_train.json",            # list of caption examples (train2014)
    "caption_dev": "caption_dev.json",                # list of caption examples (val2014)
    "grounding_train_presence": "grounding_train_presence.json",  # list of pairs (train2014 pool)
    "grounding_train_counting": "grounding_train_counting.json",
    "grounding_dev": "grounding_dev.json",            # list of pairs (val2014)
    "grounding_test": "grounding_test.json",          # list of pairs, items carry mismatch_image_id
    "vqa_train": "vqa_train.json",
    "vqa_dev": "vqa_dev.json",
    "vqa_test": "vqa_test.json",                      # items carry mismatch_image_id
}
PAIR_COMPONENTS = ("grounding_train_presence", "grounding_train_counting", "grounding_dev",
                   "grounding_test", "vqa_train", "vqa_dev", "vqa_test")
TRAIN_COMPONENTS = ("caption_train", "grounding_train_presence", "grounding_train_counting", "vqa_train")
EVAL_COMPONENTS = ("caption_dev", "grounding_dev", "vqa_dev", "grounding_test", "vqa_test")


def write_manifest(manifest: dict, splits_dir: Path) -> None:
    for key, fname in MANIFEST_FILES.items():
        write_json(manifest[key], Path(splits_dir) / fname, indent=None)
    if "meta" in manifest:
        write_json(manifest["meta"], Path(splits_dir) / "meta.json")
    if "images" in manifest:
        write_json(manifest["images"], Path(splits_dir) / "images.json", indent=None)


def load_manifest(splits_dir: Path) -> dict:
    m = {key: read_json(Path(splits_dir) / fname) for key, fname in MANIFEST_FILES.items()}
    meta = Path(splits_dir) / "meta.json"
    if meta.exists():
        m["meta"] = read_json(meta)
    return m


def component_images(manifest: dict) -> dict[str, set[int]]:
    """image ids used by every component (mismatch images are drawn from the same suite)."""
    out: dict[str, set[int]] = {}
    for key in ("caption_train", "caption_dev"):
        out[key] = {int(e["image_id"]) for e in manifest[key]}
    for key in PAIR_COMPONENTS:
        out[key] = {int(it["image_id"]) for p in manifest[key] for it in p["items"]}
    for key in ("grounding_test", "vqa_test"):
        for p in manifest[key]:
            for it in p["items"]:
                out[key].add(int(it["mismatch_image_id"]))
    return out


def images_by_coco_split(manifest: dict) -> dict[str, list[int]]:
    """{'train2014': [...], 'val2014': [...]} derived from component coco_split fields."""
    split_of: dict[int, str] = {}
    for key in ("caption_train", "caption_dev"):
        for e in manifest[key]:
            split_of[int(e["image_id"])] = e["coco_split"]
    for key in PAIR_COMPONENTS:
        for p in manifest[key]:
            for it in p["items"]:
                split_of[int(it["image_id"])] = p["coco_split"]
                if "mismatch_image_id" in it:
                    split_of[int(it["mismatch_image_id"])] = p["coco_split"]
    out: dict[str, list[int]] = {"train2014": [], "val2014": []}
    for img, sp in split_of.items():
        out[sp].append(img)
    return {k: sorted(v) for k, v in out.items()}
