"""End-to-end manifest construction on synthetic annotations (no torch / no network)."""
import copy
import json

import pytest

from task2_vlm_grounding.src.build_manifest import build
from task2_vlm_grounding.src.data_vqav2 import normalize_vqa_answer
from task2_vlm_grounding.src.manifest import component_images, load_manifest
from task2_vlm_grounding.src.split_checks import check_manifest
from task2_vlm_grounding.tests.synthetic import make_raw, tiny_cfg


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    make_raw(raw)
    cfg = tiny_cfg()
    out = tmp_path_factory.mktemp("splits")
    manifest = build(cfg, raw, out)
    return cfg, raw, out, manifest


def test_sizes_and_checks(built):
    cfg, _, out, m = built
    d = cfg["data"]
    assert len(m["caption_train"]) == 40 and len(m["caption_dev"]) == 4
    assert len(m["grounding_dev"]) == 4 and len(m["grounding_test"]) == 6
    assert len(m["vqa_train"]) == 11 and len(m["vqa_dev"]) == 6 and len(m["vqa_test"]) == 9
    check_manifest(load_manifest(out), cfg)


def test_split_isolation(built):
    _, _, _, m = built
    imgs = component_images(m)
    evals = ["caption_dev", "grounding_dev", "vqa_dev", "grounding_test", "vqa_test"]
    for i, a in enumerate(evals):
        for b in evals[i + 1:]:
            assert not imgs[a] & imgs[b], (a, b)
    train = set().union(*(imgs[k] for k in ("caption_train", "grounding_train_presence",
                                            "grounding_train_counting", "vqa_train")))
    assert not train & set().union(*(imgs[k] for k in evals))
    assert all(i < 500000 for i in train) and all(i >= 500000 for k in evals for i in imgs[k])


def test_isolation_violation_is_detected(built):
    cfg, _, _, m = built
    bad = copy.deepcopy(m)
    bad["grounding_dev"][0]["items"][0]["image_id"] = bad["vqa_dev"][0]["items"][0]["image_id"]
    with pytest.raises(AssertionError):
        check_manifest(bad, cfg)


def test_grounding_labels(built):
    _, raw, _, m = built
    ann = json.loads((raw / "coco" / "annotations" / "instances_val2014.json").read_text())
    names = {c["id"]: c["name"] for c in ann["categories"]}
    name_id = {v: k for k, v in names.items()}
    per = {}
    for a in ann["annotations"]:
        if not a["iscrowd"]:
            per[(a["image_id"], a["category_id"])] = per.get((a["image_id"], a["category_id"]), 0) + 1
    for p in m["grounding_test"] + m["grounding_dev"]:
        cat = name_id[p["category"]]
        for it in p["items"]:
            n = per.get((it["image_id"], cat), 0)
            if p["kind"] == "presence":
                assert (it["answer"] == "yes") == (n >= 1)
            else:
                assert it["answer"] == str(n) and 1 <= n <= 4
    cq = [p["question"] for p in m["grounding_test"] if p["kind"] == "counting"][0]
    assert cq.startswith("How many ") and cq.endswith(" can you see?")


def test_vqa_filters(built):
    _, _, _, m = built
    for key in ("vqa_train", "vqa_dev", "vqa_test"):
        for p in m[key]:
            a, b = p["items"]
            assert a["canonical"] != b["canonical"]
            for it in (a, b):
                assert sum(normalize_vqa_answer(x) == it["canonical"] for x in it["all_answers"]) >= 8
                assert len(it["all_answers"]) == 10
    yes_items = [it for p in m["vqa_train"] + m["vqa_test"] for it in p["items"] if it["canonical"] == "yes"]
    assert yes_items and all(it["answer"] in {"yes", "Yes"} for it in yes_items)


def test_vqa_stratification(built):
    cfg, _, _, m = built
    for key in ("vqa_train", "vqa_dev", "vqa_test"):
        for kind, n in cfg["data"][key].items():
            assert sum(p["kind"] == kind for p in m[key]) == n


def test_mismatch_mapping(built):
    _, _, _, m = built
    for key in ("grounding_test", "vqa_test"):
        suite = {it["image_id"] for p in m[key] for it in p["items"]}
        for p in m[key]:
            own = {it["image_id"] for it in p["items"]}
            for it in p["items"]:
                assert it["mismatch_image_id"] in suite - own


def test_deterministic(built, tmp_path):
    cfg, raw, out, m = built
    m2 = build(cfg, raw, tmp_path)
    for key in ("caption_train", "vqa_test", "grounding_test", "grounding_train_counting"):
        assert m[key] == m2[key]
