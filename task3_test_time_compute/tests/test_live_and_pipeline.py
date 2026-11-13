import json
import re
from pathlib import Path

import numpy as np
import pytest

from common.seed import derive_seed
from task3_test_time_compute.src import analyze
from task3_test_time_compute.src import answers as A
from task3_test_time_compute.src import live_generation as LG
from task3_test_time_compute.tests import synth

CFG_LIVE = {"system_prompt": "You are a helpful assistant.", "instruction": "Think step by step.", "review_prompt": "Review the solution above.",
            "single_max_new_tokens": 3072, "parallel_k": 3, "parallel_max_new_tokens": 1024, "refine_max_new_tokens": 2048,
            "sampling": {}}


class FakeBackend:
    """Deterministic stand-in for the HF model: answers depend on the seed; counts calls."""

    def __init__(self):
        self.calls = []

    def generate(self, messages, max_new_tokens, seed):
        self.calls.append((len(messages), max_new_tokens, seed))
        rng = np.random.default_rng(seed)
        ans = None if rng.random() < 0.1 else str(int(rng.integers(0, 3)))
        text = f"thinking... \\boxed{{{ans}}}" if ans is not None else "thinking without an answer"
        n_in = sum(len(m["content"]) for m in messages) // 4 + 1
        n_out = int(rng.integers(20, max_new_tokens))
        return LG.GenResult(text=text, input_tokens=n_in, output_tokens=n_out, truncated=bool(rng.random() < 0.1), seconds=float(rng.random()),
                            peak_mem_mb=1000.0 + float(rng.random()))


def _questions(n=6):
    return [{"qid": f"q{i:03d}", "problem": f"Compute {i}", "gold": str(i % 3)} for i in range(n)]


def test_live_accounting_and_resume(tmp_path):
    ctx = A.MathAnswerContext(force_fallback=True)
    be = FakeBackend()
    out = LG.run_live(be, _questions(), CFG_LIVE, ctx, tmp_path, 6304)
    assert len(be.calls) == 6 * 5                          # single + 3 parallel + 1 revision per question
    gens = {(r["qid"], r["role"]): r for r in map(json.loads, (tmp_path / "generations.jsonl").read_text().splitlines())}
    seeds = [g["seed"] for g in gens.values()]
    assert len(set(seeds)) == len(seeds)                   # independently seeded calls
    assert gens[("q000", "single")]["seed"] == derive_seed(6304, "q000", "single")
    pq = out["per_question"]
    r = pq.iloc[0]
    p0, rf = gens[("q000", "parallel_0")], gens[("q000", "refine")]
    # shared initial response charged to refinement; revision prompt tokens counted
    assert r.ref_in == p0["input_tokens"] + rf["input_tokens"] and r.ref_out == p0["output_tokens"] + rf["output_tokens"]
    assert r.par_out == sum(gens[("q000", f"parallel_{k}")]["output_tokens"] for k in range(3))
    # the revision prompt contains the initial response as an assistant turn plus the review instruction
    msgs = rf["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"] and msgs[2]["content"] == p0["response"]
    assert msgs[1]["content"] == "Compute 0\nThink step by step."
    assert gens[("q000", "refine")]["max_new_tokens"] == 2048 and gens[("q000", "single")]["max_new_tokens"] == 3072
    t = out["transitions"]["counts"]
    assert sum(t.values()) == 6
    assert set(out["summary"].strategy) >= {"single_3072", "parallel_3x1024_sc", "refine_1024_plus_2048"}
    # resume: a second run makes no new generation calls and gives the same answers
    be2 = FakeBackend()
    out2 = LG.run_live(be2, _questions(), CFG_LIVE, ctx, tmp_path, 6304)
    assert be2.calls == [] and out2["per_question"]["refined_ok"].tolist() == pq["refined_ok"].tolist()


def test_full_pipeline_on_synthetic_pools(tmp_path):
    cfg = analyze.load_config(smoke=False)
    cfg["paths"].update(results_dir=str(tmp_path / "results"), figures_dir=str(tmp_path / "figures"),
                        splits_dir=str(tmp_path / "splits"), processed_dir=str(tmp_path / "processed"))
    cfg.update(n_orders=3, n_bootstrap=50)
    cfg["split"] = {"n_dev": 10, "n_live": 4}
    cfg["answers"]["allow_fallback_scorer"] = True
    pools = {"math": synth.make_pool("math", n_q=36, seed=5, with_problem=True),
             "mathvista": synth.make_pool("mathvista", n_q=36, vision=True, seed=6)}
    # the injected MATH pool uses the fallback scorer; live grading must use the same one
    orig = analyze.make_answer_ctx
    analyze.make_answer_ctx = lambda c, ds: A.MathAnswerContext(force_fallback=True) if ds == "math" else orig(c, ds)
    try:
        state = analyze.run_pipeline(cfg, stages=analyze.ALL_STAGES, pools=pools, backend=FakeBackend())
    finally:
        analyze.make_answer_ctx = orig
    res = Path(cfg["paths"]["results_dir"])
    for rel in ["summary.md", "offline/math/method_by_budget.csv", "offline/mathvista/method_by_budget.csv", "signals/mathvista/confidence_vision_summary.json",
                "adaptive/math/selected_settings.json", "adaptive/mathvista/heldout_summary.csv", "live/summary.csv", "qualitative/math/examples.md"]:
        assert (res / rel).exists(), rel
    figs = {p.name for p in Path(cfg["paths"]["figures_dir"]).glob("*.png")}
    for f in ["accuracy_vs_budget_math.png", "accuracy_vs_budget_mathvista.png", "failure_decomposition_mathvista.png", "adaptive_pareto_math.png",
              "fusion_and_confidence_vision_mathvista.png", "live_strategies.png", "quartile_lift_math.png"]:
        assert f in figs, f
    sel = json.loads((res / "adaptive/math/selected_settings.json").read_text())
    assert set(sel) == {"agreement", "quality_aware"}
    sp = state["math"]["splits"]
    assert len(sp["dev"]) == 10 and not set(sp["dev"]) & set(sp["eval"]) and len(sp["live"]) == 4
    assert (Path(cfg["paths"]["splits_dir"]) / "math_dev_ids.json").exists()


def test_notebook_is_valid_and_compiles():
    nb_path = Path(__file__).resolve().parents[1] / "task3_kaggle_runner.ipynb"
    nb = json.loads(nb_path.read_text())
    assert nb["nbformat"] == 4
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    compile(src, "notebook", "exec")
    assert re.search(r"^SMOKE_TEST\s*=\s*(True|False)", src, re.M)
    assert "Kaggle" in "".join("".join(c["source"]) for c in nb["cells"])
