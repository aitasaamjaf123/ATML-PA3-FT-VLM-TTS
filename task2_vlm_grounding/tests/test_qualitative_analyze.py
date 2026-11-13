"""Qualitative selection, tables and plots from fabricated predictions (no torch)."""
import json
import random
from pathlib import Path
from types import SimpleNamespace

from common.io import write_json, write_jsonl
from task2_vlm_grounding.src import analyze
from task2_vlm_grounding.src.bootstrap import summarize
from task2_vlm_grounding.src.qualitative import CATEGORIES, build_records, dump_qualitative, select


def fake_rows(n_pairs=12, seed=0):
    rng, rows = random.Random(seed), []
    for suite, qt in (("grounding", ["presence", "counting"]), ("vqa", ["yesno", "number", "other"])):
        for k in range(n_pairs):
            pid, q = f"{suite}{k:02d}", qt[k % len(qt)]
            mode = k % 4                                   # 0 success, 1 identical, 2 partial, 3 unsupported
            refs = ["yes", "no"]
            preds_correct = [["yes", "no"], ["yes", "yes"], ["yes", "maybe"], ["foo", "bar"]][mode]
            for cond in ("correct", "complementary", "mismatched", "no_image"):
                for i in (0, 1):
                    pred = preds_correct[i] if cond == "correct" else preds_correct[1 - i] if cond == "complementary" else "yes"
                    rows.append({"suite": suite, "condition": cond, "pair_id": pid, "idx": i, "qtype": q,
                                 "question": "Q?", "reference": refs[i], "canonical": refs[i], "prediction": pred,
                                 "pred_norm": pred, "score": float(pred == refs[i]), "exact": float(pred == refs[i]),
                                 "empty": 0.0, "partner_follow": float(pred == refs[1 - i]),
                                 "nll": rng.uniform(0.1, 2.0), "gen_ms": 4.0, "image_correct": 100 + 2 * k + i,
                                 "image_complementary": 100 + 2 * k + 1 - i})
    return rows


def test_selection_rule_is_deterministic_and_categorised():
    rows = fake_rows()
    recs = build_records(rows, {})
    assert {r["category"] for r in recs["grounding"]} == set(CATEGORIES)
    a = select(recs["grounding"], 2)
    assert a == select(build_records(list(reversed(rows)), {})["grounding"], 2)
    succ = a["success"]
    assert [r["mean_nll_correct"] for r in succ] == sorted(r["mean_nll_correct"] for r in succ)
    assert all(r["category"] == "success" for r in succ)


def test_dump_tables_and_plots(tmp_path):
    rows = fake_rows()
    cfg = {"eval": {"qualitative_per_category": 2}}
    p = SimpleNamespace(root=tmp_path, raw=tmp_path / "raw", qual_dir=tmp_path / "qual", figures=tmp_path / "fig",
                        dev_dir=tmp_path / "dev", final_dir=tmp_path / "final", logs_dir=tmp_path / "logs",
                        results=tmp_path, features=tmp_path / "feat")
    for d in (p.qual_dir, p.figures, p.dev_dir, p.final_dir, p.logs_dir, p.features):
        d.mkdir(parents=True)
    info = dump_qualitative(rows, {}, cfg, p)
    assert info["category_counts"]["grounding"]["success"] == 3
    assert "identical_answer" in (p.qual_dir / "qualitative_examples.md").read_text()
    write_json(summarize(rows, 100, 6304), p.final_dir / "final_summary.json")
    write_jsonl([{"update": u, "stage": 1, "loss": 3.0 / (1 + u)} for u in range(1, 46)], p.logs_dir / "train_log.jsonl")
    dev = [{"step": s, "stage": 1, "caption_dev_loss": 3.0 - s / 20, **{f"{x}_{k}": 1.0 for x in ("grounding", "vqa") for k in
           ("nll_correct", "nll_complementary", "nll_no_image", "delta_nll", "acc_correct", "acc_complementary")}}
           for s in (0, 5, 5, 10)]                         # duplicate step 5: must be de-duplicated
    write_jsonl(dev, p.dev_dir / "dev_metrics.jsonl")
    write_json({"stages": {"stage1": {"name": "x", "updates": 15, "examples_presented": 240, "train_time_s": 1.0, "peak_gpu_mb": 10.0,
                "trainable_params": {"connector_trainable": 1, "lora_trainable": 0, "total_trainable": 1}}},
                "totals": {"updates": 15, "examples_presented": 240, "train_time_s": 1.0, "dev_eval_time_s": 0.1, "peak_gpu_mb": 10.0}},
               p.logs_dir / "resources_train.json")
    boundaries = [15, 30]
    analyze.plot_training(p, cfg, boundaries)
    dev_rows = analyze.plot_dev(p, cfg, boundaries)
    assert [d["step"] for d in dev_rows] == [0, 5, 10]
    analyze.final_tables(p)
    analyze.plot_final_bars(p)
    analyze.resource_table(p)
    assert "correct_vs_complementary" in (p.final_dir / "final_tables.md").read_text()
    for f in ("training_loss.png", "dev_caption_loss.png", "dev_image_swap_penalty.png", "final_accuracy_by_condition.png"):
        assert (p.figures / f).stat().st_size > 1000
    assert "Training" in (tmp_path / "resources.md").read_text()
