import math
import random
from collections import Counter

import numpy as np
import pytest

from common.stats import bootstrap_diff, bootstrap_indices, bootstrap_stat
from task2_vlm_grounding.src.bootstrap import summarize
from task2_vlm_grounding.src.config import dev_eval_steps, load_config, stage_list, total_updates
from task2_vlm_grounding.src.sampler import CurriculumSampler
from task2_vlm_grounding.src.schedule import lr_multiplier


def fake_manifest(n_caps=50):
    def pair(i, suite, kind):
        return {"pair_id": f"{kind}{i}", "suite": suite, "kind": kind, "question": f"q{i}", "coco_split": "train2014",
                "items": [{"example_id": f"{kind}{i}a", "image_id": 2 * i, "answer": "x", "canonical": "x"},
                          {"example_id": f"{kind}{i}b", "image_id": 2 * i + 1, "answer": "y", "canonical": "y"}]}
    return {"caption_train": [{"example_id": f"c{i}", "image_id": i, "question": "Describe this image.",
                               "answer": f"cap {i}"} for i in range(n_caps)],
            "grounding_train_presence": [pair(i, "grounding", "presence") for i in range(20)],
            "grounding_train_counting": [pair(i, "grounding", "counting") for i in range(20)],
            "vqa_train": [pair(i, "natural_vqa", "yesno") for i in range(20)]}


MIXES = [{"caption_pass": 1.0}, {"presence": 0.5, "counting": 0.25, "caption": 0.25},
         {"vqa": 0.7, "presence": 0.1, "counting": 0.1, "caption": 0.1}]


def test_budget_arithmetic():
    cfg = load_config()
    assert total_updates(cfg) == 4500 and 4500 * 16 == 72_000
    assert [(s["start"], s["end"]) for s in stage_list(cfg)] == [(0, 2250), (2250, 3750), (3750, 4500)]
    assert dev_eval_steps(cfg) == [0, 500, 1000, 1500, 2000, 2250, 2500, 3000, 3500, 3750, 4000, 4500]
    assert 2250 * 16 == 2 * 18_000
    smoke = load_config(smoke=True)
    assert total_updates(smoke) == 45 and dev_eval_steps(smoke)[:3] == [0, 5, 10]


def test_stage1_two_passes_cover_each_caption_twice():
    s = CurriculumSampler(fake_manifest(48), 6304, MIXES)
    seen = Counter()
    for _ in range(6):  # 6 * 16 = 96 = 2 passes of 48
        ex, _ = s.next_update(0)
        seen.update(e["example_id"] for e in ex)
    assert set(seen.values()) == {2} and len(seen) == 48


def test_mixed_updates_pairs_and_probabilities():
    s = CurriculumSampler(fake_manifest(), 6304, MIXES)
    tot = Counter()
    for _ in range(400):
        ex, counts = s.next_update(1)
        assert len(ex) == 16 and sum(counts.values()) == 8
        tot.update(counts)
        for k in range(0, 16, 2):               # every draw is two adjacent examples
            a, b = ex[k], ex[k + 1]
            if a["component"] == "caption":
                assert b["component"] == "caption" and a["example_id"] != b["example_id"]
            else:
                assert a["pair_id"] == b["pair_id"] and a["question"] == b["question"]
    n = sum(tot.values())
    assert tot["presence"] / n == pytest.approx(0.5, abs=0.04)
    assert tot["counting"] / n == pytest.approx(0.25, abs=0.04)
    assert tot["caption"] / n == pytest.approx(0.25, abs=0.04)
    s3 = CurriculumSampler(fake_manifest(), 6304, MIXES)
    t3 = Counter()
    for _ in range(400):
        t3.update(s3.next_update(2)[1])
    assert t3["vqa"] / sum(t3.values()) == pytest.approx(0.7, abs=0.04)


def test_resume_is_exact():
    a = CurriculumSampler(fake_manifest(), 6304, MIXES)
    for i in range(7):
        a.next_update(0)
    for i in range(5):
        a.next_update(1)
    for i in range(3):
        a.next_update(2)
    state = a.state_dict()
    expected = [(a.next_update(0), a.next_update(1), a.next_update(2)) for _ in range(5)]
    b = CurriculumSampler(fake_manifest(), 6304, MIXES)
    b.load_state_dict(state)
    got = [(b.next_update(0), b.next_update(1), b.next_update(2)) for _ in range(5)]
    assert expected == got


def test_lr_schedule():
    total = 1500
    warm = math.ceil(0.05 * total)
    m = [lr_multiplier(s, total) for s in range(total)]
    assert m[0] == pytest.approx(1 / warm) and m[warm - 1] == pytest.approx(1.0) and m[warm] == pytest.approx(1.0)
    assert m[-1] == pytest.approx(0.1)
    assert all(m[i] >= m[i + 1] - 1e-12 for i in range(warm, total - 1))
    assert lr_multiplier(0, 15) == 1.0 and lr_multiplier(14, 15) == pytest.approx(0.1)


def test_bootstrap_matched_and_ci_covers():
    rng = np.random.default_rng(0)
    a = rng.normal(0.6, 0.2, 96)
    b = a - 0.2 + rng.normal(0, 0.05, 96)
    idx = bootstrap_indices(96, 4000, 6304)
    assert idx.shape == (4000, 96)
    d = bootstrap_diff(a, b, idx)
    assert d["ci_low"] < 0.2 < d["ci_high"] and d["mean"] == pytest.approx(0.2, abs=0.02)
    assert np.array_equal(idx, bootstrap_indices(96, 4000, 6304))
    s = bootstrap_stat(np.ones(10), bootstrap_indices(10, 100, 1))
    assert s["ci_low"] == s["ci_high"] == 1.0
    with pytest.raises(ValueError):
        bootstrap_diff(a, b[:-1], idx)


def _rows():
    rows = []
    for k in range(10):
        pid = f"p{k}"
        for cond in ("correct", "complementary", "mismatched", "no_image"):
            for i in range(2):
                ok = cond == "correct" or (cond == "no_image" and i == 0)
                rows.append({"suite": "grounding", "condition": cond, "pair_id": pid, "idx": i, "qtype": "presence" if k < 5 else "counting",
                             "score": float(ok), "exact": float(ok), "empty": 0.0, "pred_norm": "yes" if ok == (i == 0) else "no",
                             "nll": 0.2 if cond == "correct" else 1.0, "partner_follow": 0.0 if ok else 1.0, "gen_ms": 5.0})
    return rows


def test_summarize():
    res = summarize(_rows(), 200, 6304)
    get = lambda c, s, m: [r for r in res["rows"] if (r["suite"], r["condition"], r["subset"], r["metric"]) == ("grounding", c, s, m)][0]
    assert get("correct", "overall", "acc")["mean"] == 1.0 and get("correct", "overall", "pair_acc")["mean"] == 1.0
    assert get("complementary", "overall", "acc")["mean"] == 0.0
    assert get("complementary", "overall", "follow")["mean"] == 1.0
    assert get("correct_vs_complementary", "overall", "acc_advantage_correct_vs_complementary")["mean"] == 1.0
    assert get("correct_vs_complementary", "overall", "nll_penalty_complementary_minus_correct")["mean"] == pytest.approx(0.8)
    assert get("correct", "presence", "acc")["n_pairs"] == 5
    assert res["n_pairs"] == {"grounding": 10}
