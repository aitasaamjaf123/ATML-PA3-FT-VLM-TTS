import pytest

from task2_vlm_grounding.src.data_vqav2 import consensus_answer, normalize_vqa_answer as n
from task2_vlm_grounding.src.vqa_eval import eval_items, image_for, score_item, vqa_soft_score


@pytest.mark.parametrize("raw,expected", [
    ("The Two dogs!", "2 dogs"), ("A  cat\n", "cat"), ("1,000", "1000"), ("dont", "don't"), ("isnt", "isn't"),
    ("3.5 .", "3.5"), ("Ten", "10"), ("none", "0"), ("an apple", "apple"), ("Yes.", "yes"), ("", ""),
    ("hot-dog", "hot dog"), ("Zero", "0"), ("\tNo\t", "no"),
])
def test_normalizer(raw, expected):
    assert n(raw) == expected


def test_consensus_and_representative():
    top, k, rep = consensus_answer(["Yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "no", "maybe"])
    assert (top, k, rep) == ("yes", 8, "yes")
    top, k, rep = consensus_answer(["The dog"] * 4 + ["dog"] * 4 + ["cat"] * 2)
    assert top == "dog" and k == 8 and rep in {"The dog", "dog"}


def test_soft_score_formula():
    refs = ["yes"] * 3 + ["no"] * 7
    assert vqa_soft_score("yes", refs) == pytest.approx((3 * (2 / 3) + 7 * 1.0) / 10)  # = 0.9 (3 matching humans)
    assert vqa_soft_score("no", refs) == pytest.approx(1.0)
    assert vqa_soft_score("maybe", refs) == 0.0
    assert vqa_soft_score("x", ["x"] * 10) == 1.0
    assert vqa_soft_score("x", ["x"]) == 0.0  # a single annotator is never enough


def _pair(suite="vqa"):
    items = [{"example_id": "a", "image_id": 1, "answer": "yes", "canonical": "yes", "all_answers": ["yes"] * 10},
             {"example_id": "b", "image_id": 2, "answer": "no", "canonical": "no", "all_answers": ["no"] * 10,
              "mismatch_image_id": 9}]
    items[0]["mismatch_image_id"] = 8
    return {"pair_id": "p", "suite": "natural_vqa" if suite == "vqa" else "grounding", "kind": "yesno",
            "question": "Q?", "coco_split": "val2014", "items": items}


def test_items_and_scoring():
    items = eval_items([_pair()], "vqa")
    assert [i["image_correct"] for i in items] == [1, 2]
    assert [i["image_complementary"] for i in items] == [2, 1]
    assert image_for(items[0], "mismatched") == 8 and image_for(items[0], "no_image") is None
    s = score_item(items[0], "Yes.")
    assert s["score"] == 1.0 and s["exact"] == 1.0 and s["partner_follow"] == 0.0
    s = score_item(items[0], "no")
    assert s["score"] == 0.0 and s["partner_follow"] == 1.0
    assert score_item(items[0], "")["empty"] == 1.0
