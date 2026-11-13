"""Annotation-derived grounding questions (presence / counting) from COCO instances.

All questions/answers are generated programmatically from `instances_*.json` (crowd
annotations are ignored for positives/counts; images with a crowd annotation of the
category are excluded from counting and from the "absent" side of presence pairs).
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Sequence

PLURALS = {
    "person": "people", "bicycle": "bicycles", "car": "cars", "motorcycle": "motorcycles",
    "airplane": "airplanes", "bus": "buses", "train": "trains", "truck": "trucks", "boat": "boats",
    "traffic light": "traffic lights", "fire hydrant": "fire hydrants", "stop sign": "stop signs",
    "parking meter": "parking meters", "bench": "benches", "bird": "birds", "cat": "cats", "dog": "dogs",
    "horse": "horses", "sheep": "sheep", "cow": "cows", "elephant": "elephants", "bear": "bears",
    "zebra": "zebras", "giraffe": "giraffes", "backpack": "backpacks", "umbrella": "umbrellas",
    "handbag": "handbags", "tie": "ties", "suitcase": "suitcases", "frisbee": "frisbees", "skis": "skis",
    "snowboard": "snowboards", "sports ball": "sports balls", "kite": "kites", "baseball bat": "baseball bats",
    "baseball glove": "baseball gloves", "skateboard": "skateboards", "surfboard": "surfboards",
    "tennis racket": "tennis rackets", "bottle": "bottles", "wine glass": "wine glasses", "cup": "cups",
    "fork": "forks", "knife": "knives", "spoon": "spoons", "bowl": "bowls", "banana": "bananas",
    "apple": "apples", "sandwich": "sandwiches", "orange": "oranges", "broccoli": "broccoli",
    "carrot": "carrots", "hot dog": "hot dogs", "pizza": "pizzas", "donut": "donuts", "cake": "cakes",
    "chair": "chairs", "couch": "couches", "potted plant": "potted plants", "bed": "beds",
    "dining table": "dining tables", "toilet": "toilets", "tv": "tvs", "laptop": "laptops",
    "mouse": "mice", "remote": "remotes", "keyboard": "keyboards", "cell phone": "cell phones",
    "microwave": "microwaves", "oven": "ovens", "toaster": "toasters", "sink": "sinks",
    "refrigerator": "refrigerators", "book": "books", "clock": "clocks", "vase": "vases",
    "scissors": "scissors", "teddy bear": "teddy bears", "hair drier": "hair driers",
    "toothbrush": "toothbrushes",
}
NO_ARTICLE = {"skis", "scissors", "broccoli"}  # plural / mass nouns
COUNT_VALUES = (1, 2, 3, 4)


def plural(name: str) -> str:
    return PLURALS[name]


def object_phrase(name: str) -> str:
    if name in NO_ARTICLE:
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def presence_question(name: str) -> str:
    return f"Does this image contain {object_phrase(name)}?"


def counting_question(name: str) -> str:
    return f"How many {plural(name)} can you see?"


class GroundingIndex:
    """Per-category image pools built from COCO instance statistics."""

    def __init__(self, stats: dict[int, dict[int, list[float]]], categories: dict[int, str],
                 image_ids: Iterable[int]) -> None:
        self.stats = stats
        self.categories = categories
        self.images = sorted(i for i in set(image_ids) if stats.get(i))  # images with >=1 annotation
        self.pos: dict[int, list[int]] = defaultdict(list)          # >=1 non-crowd instance
        self.any_count: dict[int, int] = defaultdict(int)           # any annotation (incl. crowd)
        self.count: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for img in self.images:
            for cat, (n, c, _area) in stats[img].items():
                self.any_count[cat] += 1
                if n >= 1:
                    self.pos[cat].append(img)
                if c == 0 and 1 <= n <= 4:
                    self.count[cat][int(n)].append(img)

    def presence_categories(self) -> list[int]:
        return sorted(c for c in self.categories
                      if self.pos.get(c) and self.any_count[c] < len(self.images))

    def counting_categories(self) -> list[int]:
        return sorted(c for c in self.categories
                      if sum(1 for k in COUNT_VALUES if self.count[c].get(k)) >= 2)


def _pick(pool: Sequence[int], rng: random.Random, used: set[int], banned: Iterable[int] = ()) -> int | None:
    banned = set(banned)
    if not pool:
        return None
    for _ in range(64):
        c = pool[rng.randrange(len(pool))]
        if c not in used and c not in banned:
            return c
    start = rng.randrange(len(pool))
    for off in range(len(pool)):
        c = pool[(start + off) % len(pool)]
        if c not in used and c not in banned:
            return c
    return None


def _pick_absent(idx: GroundingIndex, cat: int, rng: random.Random, used: set[int]) -> int | None:
    imgs = idx.images
    for _ in range(500):
        c = imgs[rng.randrange(len(imgs))]
        if cat not in idx.stats[c] and c not in used:
            return c
    start = rng.randrange(len(imgs))
    for off in range(len(imgs)):
        c = imgs[(start + off) % len(imgs)]
        if cat not in idx.stats[c] and c not in used:
            return c
    return None


def _item(pair_id: str, suffix: str, image_id: int, answer: str) -> dict:
    return {"example_id": f"{pair_id}_{suffix}", "question_id": None, "image_id": int(image_id),
            "answer": answer, "canonical": answer}


def make_presence_pair(idx: GroundingIndex, cat: int, rng: random.Random, used: set[int],
                       pair_id: str, coco_split: str) -> dict | None:
    pos = _pick(idx.pos[cat], rng, used)
    if pos is None:
        return None
    neg = _pick_absent(idx, cat, rng, used | {pos})
    if neg is None:
        return None
    name = idx.categories[cat]
    items = [_item(pair_id, "a", pos, "yes"), _item(pair_id, "b", neg, "no")]
    if rng.random() < 0.5:
        items.reverse()
        items[0]["example_id"], items[1]["example_id"] = f"{pair_id}_a", f"{pair_id}_b"
    return {"pair_id": pair_id, "suite": "grounding", "kind": "presence", "question": presence_question(name),
            "category": name, "coco_split": coco_split, "items": items}


def make_counting_pair(idx: GroundingIndex, cat: int, rng: random.Random, used: set[int],
                       pair_id: str, coco_split: str) -> dict | None:
    ks = [k for k in COUNT_VALUES if idx.count[cat].get(k)]
    if len(ks) < 2:
        return None
    k1, k2 = rng.sample(ks, 2)
    i1 = _pick(idx.count[cat][k1], rng, used)
    i2 = _pick(idx.count[cat][k2], rng, used, banned={i1})
    if i1 is None or i2 is None:
        return None
    name = idx.categories[cat]
    items = [_item(pair_id, "a", i1, str(k1)), _item(pair_id, "b", i2, str(k2))]
    return {"pair_id": pair_id, "suite": "grounding", "kind": "counting", "question": counting_question(name),
            "category": name, "coco_split": coco_split, "items": items}


def generate_train_pairs(idx: GroundingIndex, kind: str, n_pairs: int, rng: random.Random,
                         coco_split: str = "train2014") -> list[dict]:
    """Reusable training pool: unique (category, image, image) pairs, images may repeat."""
    cats = idx.presence_categories() if kind == "presence" else idx.counting_categories()
    if not cats:
        raise RuntimeError(f"No valid categories for {kind} pairs")
    maker = make_presence_pair if kind == "presence" else make_counting_pair
    seen, pairs, attempts = set(), [], 0
    while len(pairs) < n_pairs and attempts < n_pairs * 60:
        attempts += 1
        cat = cats[rng.randrange(len(cats))]
        p = maker(idx, cat, rng, set(), f"grd_train_{kind}_{len(pairs):05d}", coco_split)
        if p is None:
            continue
        key = (cat, tuple(sorted((it["image_id"], it["answer"]) for it in p["items"])))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(p)
    if len(pairs) < n_pairs:
        # The pool is "reusable": fewer unique pairs than requested is acceptable for tiny image pools.
        if not pairs:
            raise RuntimeError(f"Could not generate any {kind} pairs")
    return pairs


def select_eval_pairs(idx: GroundingIndex, kind: str, n_pairs: int, rng: random.Random,
                      used_images: set[int], tag: str, coco_split: str = "val2014") -> list[dict]:
    """Image-disjoint evaluation pairs (updates the global `used_images`)."""
    base = idx.presence_categories() if kind == "presence" else idx.counting_categories()
    maker = make_presence_pair if kind == "presence" else make_counting_pair
    pairs: list[dict] = []
    rounds = 0
    while len(pairs) < n_pairs and rounds < 200:
        rounds += 1
        cats = list(base)
        rng.shuffle(cats)  # spread categories: one pair per category per round
        for cat in cats:
            if len(pairs) == n_pairs:
                break
            p = maker(idx, cat, rng, used_images, f"grd_{tag}_{kind}_{len(pairs):04d}", coco_split)
            if p is None:
                continue
            used_images.update(it["image_id"] for it in p["items"])
            pairs.append(p)
    if len(pairs) < n_pairs:
        raise RuntimeError(f"Only {len(pairs)}/{n_pairs} {kind} eval pairs could be built")
    return pairs


def audit_rows(pairs: Sequence[dict], stats: dict, areas: dict, categories: dict[int, str]) -> list[dict]:
    """Label audit (independent of model predictions): counts, crowd flags, area fractions."""
    name_to_id = {v: k for k, v in categories.items()}
    rows = []
    for p in pairs:
        cat = name_to_id[p["category"]]
        for it in p["items"]:
            n, c, area = stats[it["image_id"]].get(cat, [0, 0, 0.0])
            rows.append({
                "pair_id": p["pair_id"], "example_id": it["example_id"], "kind": p["kind"],
                "category": p["category"], "question": p["question"], "answer": it["answer"],
                "image_id": it["image_id"], "n_noncrowd": n, "n_crowd": c,
                "category_area_fraction": round(area / max(areas[it["image_id"]], 1.0), 5),
                "n_categories_in_image": len(stats[it["image_id"]]),
            })
    return rows
