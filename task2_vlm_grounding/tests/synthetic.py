"""Synthetic COCO-2014 / VQAv2 annotation files in the real formats (for torch-free tests)."""
from __future__ import annotations

import json
import random
from pathlib import Path

CATS = ["person", "car", "dog", "cat", "bus", "knife", "sheep", "mouse", "chair", "cup", "bottle", "bench"]
COLORS = ["red", "blue", "green", "white", "black", "yellow"]


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def make_coco(raw: Path, split: str, first_id: int, n_images: int, rng: random.Random) -> list[int]:
    ids = list(range(first_id, first_id + n_images))
    images = [{"id": i, "width": 640, "height": 480} for i in ids]
    caps, anns, aid = [], [], 1
    for i in ids:
        for k in range(5):
            caps.append({"id": i * 10 + k, "image_id": i, "caption": f"  A photo number {i} variant {k} .  "})
        for cat in rng.sample(range(1, len(CATS) + 1), rng.randint(1, 4)):
            for _ in range(rng.choice([1, 1, 2, 3, 4, 5, 6])):
                anns.append({"id": aid + first_id * 1000, "image_id": i, "category_id": cat,
                             "iscrowd": 1 if rng.random() < 0.03 else 0, "area": rng.uniform(100, 5000)})
                aid += 1
    cats = [{"id": k + 1, "name": n} for k, n in enumerate(CATS)]
    d = raw / "coco" / "annotations"
    _write(d / f"captions_{split}.json", {"images": images, "annotations": caps})
    _write(d / f"instances_{split}.json", {"images": images, "annotations": anns, "categories": cats})
    return ids


def make_vqa(raw: Path, split: str, image_ids: list[int], n_pairs: int, rng: random.Random, qid0: int) -> None:
    questions, annotations, pairs = [], [], []
    qid = qid0
    for p in range(n_pairs):
        kind = ["yes/no", "number", "other"][p % 3]
        i1, i2 = rng.sample(image_ids, 2)
        if kind == "yes/no":
            q, (a1, a2) = "Is it sunny?", ("yes", "no")
        elif kind == "number":
            q, (a1, a2) = "How many things are there?", rng.sample(["1", "2", "3", "4"], 2)
        else:
            q, (a1, a2) = "What color is the object?", rng.sample(COLORS, 2)
        agree = rng.choice([10, 9, 8, 8, 6])            # 6 -> must be filtered out
        if p % 11 == 0:
            a2 = a1                                      # identical consensus -> must be filtered out
        for img, ans in ((i1, a1), (i2, a2)):
            qid += 1
            other = rng.choice([x for x in COLORS + ["1", "yes", "no"] if x != ans])
            answers = [ans] * agree + [other] * (10 - agree)
            if ans == "yes":
                answers[0] = "Yes"                       # original string differs from normalised form
            questions.append({"question_id": qid, "image_id": img, "question": q})
            annotations.append({"question_id": qid, "image_id": img, "answer_type": kind, "question_type": "x",
                                "multiple_choice_answer": ans,
                                "answers": [{"answer": a, "answer_confidence": "yes", "answer_id": k + 1}
                                            for k, a in enumerate(answers)]})
        pairs.append([qid - 1, qid])
    d = raw / "vqa"
    _write(d / f"v2_OpenEnded_mscoco_{split}_questions.json", {"questions": questions})
    _write(d / f"v2_mscoco_{split}_annotations.json", {"annotations": annotations})
    _write(d / f"v2_mscoco_{split}_complementary_pairs.json", pairs)


def make_raw(raw: Path, seed: int = 1, n_train: int = 700, n_val: int = 500) -> None:
    rng = random.Random(seed)
    tr = make_coco(raw, "train2014", 1000, n_train, rng)
    va = make_coco(raw, "val2014", 500000, n_val, rng)
    make_vqa(raw, "train2014", tr, 400, rng, 10_000_000)
    make_vqa(raw, "val2014", va, 400, rng, 20_000_000)


def tiny_cfg() -> dict:
    from task2_vlm_grounding.src.config import deep_merge, load_config

    cfg = load_config(smoke=True)
    return deep_merge(cfg, {"data": {
        "caption_train_images": 20, "caption_train_per_image": 2, "caption_dev_images": 4,
        "grounding_pool_images": 150, "grounding_train_presence_pairs": 30, "grounding_train_counting_pairs": 30,
        "grounding_dev": {"presence": 2, "counting": 2}, "grounding_test": {"presence": 3, "counting": 3},
        "vqa_train": {"yesno": 4, "number": 3, "other": 4}, "vqa_dev": {"yesno": 2, "number": 2, "other": 2},
        "vqa_test": {"yesno": 3, "number": 3, "other": 3}}})
