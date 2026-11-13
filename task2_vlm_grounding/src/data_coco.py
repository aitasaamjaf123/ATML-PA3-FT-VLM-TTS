"""COCO 2014 captions + instance-annotation statistics (pure python)."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

COCO_SPLITS = ("train2014", "val2014")


def coco_image_filename(coco_split: str, image_id: int) -> str:
    return f"COCO_{coco_split}_{int(image_id):012d}.jpg"


def coco_image_path(raw_dir: Path, coco_split: str, image_id: int) -> Path:
    return Path(raw_dir) / "coco" / coco_split / coco_image_filename(coco_split, image_id)


def coco_image_url(coco_split: str, image_id: int) -> str:
    return f"http://images.cocodataset.org/{coco_split}/{coco_image_filename(coco_split, image_id)}"


def find_coco_json(raw_dir: Path, name: str) -> Path:
    hits = sorted(Path(raw_dir).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {raw_dir}")
    return hits[0]


def load_captions(raw_dir: Path, coco_split: str) -> dict[int, list[tuple[int, str]]]:
    """image_id -> [(caption_id, caption)] sorted by caption id."""
    with open(find_coco_json(raw_dir, f"captions_{coco_split}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    caps: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for a in data["annotations"]:
        caps[int(a["image_id"])].append((int(a["id"]), " ".join(a["caption"].strip().split())))
    return {k: sorted(v) for k, v in caps.items()}


def select_caption_examples(caps: dict[int, list[tuple[int, str]]], n_images: int, per_image: int,
                            rng: random.Random, coco_split: str, exclude: set[int] | None = None,
                            prompt: str = "Describe this image.") -> list[dict]:
    """Pick `n_images` images (seeded) with `per_image` captions each (seeded, sorted by id)."""
    exclude = exclude or set()
    eligible = sorted(i for i, v in caps.items() if len(v) >= per_image and i not in exclude)
    if len(eligible) < n_images:
        raise RuntimeError(f"Only {len(eligible)} eligible caption images, need {n_images}")
    rng.shuffle(eligible)
    out = []
    for img in eligible[:n_images]:
        chosen = sorted(rng.sample(caps[img], per_image))
        for cap_id, text in chosen:
            out.append({
                "example_id": f"cap_{coco_split[:-4]}_{img}_{cap_id}",
                "image_id": int(img),
                "coco_split": coco_split,
                "question": prompt,
                "answer": text,
            })
    return out


def load_instances(raw_dir: Path, coco_split: str) -> tuple[dict[int, str], dict[int, dict[int, list[float]]], dict[int, float]]:
    """Returns (categories, per-image stats, image areas).

    stats[image_id][category_id] = [n_noncrowd, n_crowd, total_noncrowd_area]
    Images present in the file but without any annotation are recorded with an empty dict.
    """
    with open(find_coco_json(raw_dir, f"instances_{coco_split}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    categories = {int(c["id"]): c["name"] for c in data["categories"]}
    stats: dict[int, dict[int, list[float]]] = {int(im["id"]): {} for im in data["images"]}
    areas = {int(im["id"]): float(im["width"]) * float(im["height"]) for im in data["images"]}
    for a in data["annotations"]:
        img, cat = int(a["image_id"]), int(a["category_id"])
        rec = stats[img].setdefault(cat, [0, 0, 0.0])
        if int(a.get("iscrowd", 0)) == 1:
            rec[1] += 1
        else:
            rec[0] += 1
            rec[2] += float(a.get("area", 0.0))
    del data
    return categories, stats, areas


def unique_sorted(ids: Sequence[int]) -> list[int]:
    return sorted(set(int(i) for i in ids))
