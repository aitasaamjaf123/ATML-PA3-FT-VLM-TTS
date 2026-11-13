"""Scoring: normalised exact match (grounding) and local VQA soft consensus accuracy.

Pure python (no torch). Also builds the flat evaluation items (correct / complementary /
mismatched image references) shared by dev and final evaluation.
"""
from __future__ import annotations

from typing import Sequence

from task2_vlm_grounding.src.data_vqav2 import normalize_vqa_answer
from task2_vlm_grounding.src.sampler import pair_to_examples

CONDITIONS = ("correct", "complementary", "mismatched", "no_image")
DEV_CONDITIONS = ("correct", "complementary", "no_image")


def vqa_soft_score(pred_norm: str, refs_norm: Sequence[str]) -> float:
    """s(a) = 1/10 * sum_i min(1, (1/3) * #{j != i : a == a_j}) (leave-one-annotator-out)."""
    n = len(refs_norm)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        matches = sum(1 for j in range(n) if j != i and refs_norm[j] == pred_norm)
        total += min(1.0, matches / 3.0)
    return total / n


def score_item(item: dict, pred: str) -> dict:
    """Score one generated answer for an evaluation item (see `eval_items`)."""
    ex = item["example"]
    pred_norm = normalize_vqa_answer(pred)
    own = normalize_vqa_answer(ex["canonical"])
    exact = float(pred_norm == own)
    if item["suite"] == "vqa":
        refs = [normalize_vqa_answer(a) for a in ex["all_answers"]]
        score = vqa_soft_score(pred_norm, refs)
    else:
        score = exact
    return {
        "pred_norm": pred_norm,
        "score": score,
        "exact": exact,
        "empty": float(pred_norm == ""),
        "partner_follow": float(pred_norm == normalize_vqa_answer(item["partner_canonical"])),
    }


def eval_items(pairs: Sequence[dict], suite: str) -> list[dict]:
    """Flatten pairs into per-example evaluation items (pair order preserved, 2 items per pair).

    `suite` is 'grounding' or 'vqa'. Mismatch images are present for final-test pairs only.
    """
    items = []
    for p in pairs:
        exs = pair_to_examples(p)
        for k, (ex, it) in enumerate(zip(exs, p["items"])):
            other = p["items"][1 - k]
            items.append({
                "pair_id": p["pair_id"], "suite": suite, "qtype": p["kind"], "idx": k, "example": ex,
                "image_correct": int(it["image_id"]),
                "image_complementary": int(other["image_id"]),
                "image_mismatched": int(it["mismatch_image_id"]) if "mismatch_image_id" in it else None,
                "partner_canonical": other["canonical"],
                "partner_answer": other["answer"],
            })
    return items


def image_for(item: dict, condition: str) -> int | None:
    if condition == "no_image":
        return None
    key = {"correct": "image_correct", "complementary": "image_complementary",
           "mismatched": "image_mismatched"}[condition]
    img = item[key]
    if img is None:
        raise KeyError(f"item {item['pair_id']} has no image for condition {condition}")
    return img
