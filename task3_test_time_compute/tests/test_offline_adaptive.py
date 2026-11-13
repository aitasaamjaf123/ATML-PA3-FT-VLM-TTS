import numpy as np
import pandas as pd
import pytest

from task3_test_time_compute.src import adaptive as AD
from task3_test_time_compute.src import offline_eval as OE
from task3_test_time_compute.src import orders as OR
from task3_test_time_compute.src import selectors as SEL
from task3_test_time_compute.src import signal_analysis as SA
from task3_test_time_compute.tests import synth

BUDGETS = [1, 2, 4, 8, 16, 32]


@pytest.fixture(scope="module")
def math_pool():
    return synth.make_pool("math", n_q=24, seed=3)


@pytest.fixture(scope="module")
def vis_pool():
    return synth.make_pool("mathvista", n_q=16, vision=True, seed=4)


def _eval(pool, n_orders=3):
    qids = pool.qids()
    orders = OR.make_orders(qids, n_orders, 32, 6304)
    return OE.evaluate_pool(pool, qids, orders, BUDGETS), orders


def test_n1_all_methods_return_first_candidate(math_pool):
    res, _ = _eval(math_pool)
    for m in res.methods:
        assert (res.sel_gid[m][:, :, 0] == res.sel_gid["first"][:, :, 0]).all()


def test_oracle_dominates_and_is_monotone(math_pool):
    res, _ = _eval(math_pool)
    cov = res.coverage.mean(axis=(0, 1))
    assert (np.diff(cov) >= -1e-12).all()
    for m in res.methods:
        assert (res.ok[m] <= res.coverage).all()  # a selector cannot be correct without a correct candidate


def test_cost_accounting(math_pool):
    res, orders = _eval(math_pool)
    q = res.qids[0]
    qd = math_pool.questions[q]
    for o in range(3):
        perm = orders[q][o]
        for b, N in enumerate(BUDGETS):
            assert res.cost[0, o, b] == qd.T[perm[:N]].sum()
            assert res.cost_first[0, o, b] == qd.T[perm[0]]
    assert (res.cost[:, :, -1] == res.cost[:, 0:1, -1]).all()  # all 32 consumed -> order independent


def test_firstans_constant_in_budget_and_cost_only_first(math_pool):
    res, _ = _eval(math_pool)
    assert (res.sel_gid["first"] == res.sel_gid["first"][:, :, :1]).all()
    assert (res.cost_qb("first") == res.cost_qb("first")[:, :1]).all()


def test_decomposition_sums_to_one(math_pool):
    res, _ = _eval(math_pool)
    d = OE.decomposition_table(res)
    assert np.allclose(d.selected_correct + d.selection_failure + d.sampling_failure, 1.0)


def test_tables_build(math_pool):
    res, _ = _eval(math_pool)
    assert len(OE.method_table(res, 50)) == len(res.methods) * len(BUDGETS)
    assert len(OE.diff_table(res, 50)) > 0
    pq = OE.per_question_table(res)
    assert len(pq) == len(res.qids) * 3 * len(BUDGETS)
    assert len(OE.budget_table(res, 50)) == len(BUDGETS)


def test_vision_methods_present_and_consistent(vis_pool):
    res, _ = _eval(vis_pool)
    for m in ("vision_top", "vision_rwc", "visconf_top", "visconf_rwc", "tokenfusion_top", "tokenfusion_rwc"):
        assert m in res.methods
    assert (res.sel_gid["visconf_top"][:, :, 0] == res.sel_gid["first"][:, :, 0]).all()


def test_features_and_policy_logic(math_pool):
    qids = math_pool.qids()
    orders = OR.make_orders(qids, 3, 32, 6304)
    F = AD.build_features(math_pool, qids, orders)
    res, _ = _eval(math_pool)
    # fixed-32 outcome equals the offline self-consistency at N=32 (same orders)
    assert np.array_equal(F.ok[:, :, -1], res.ok["selfcons"][:, :, -1])
    full = AD.run_policy(F, None, None)
    assert (full["n"] == 32).all()
    # stricter thresholds can only stop later (cost non-decreasing in tau for the agreement controller)
    costs = [AD.run_policy(F, t, None)["cost"].mean() for t in (0.5, 0.625, 0.75, 0.875, 1.0)]
    assert all(a <= b + 1e-9 for a, b in zip(costs, costs[1:]))
    assert costs[-1] <= full["cost"].mean() + 1e-9
    # quality-aware stops no earlier than agreement with the same tau
    q = AD.run_policy(F, 0.5, 0.0)
    a = AD.run_policy(F, 0.5, None)
    assert (q["n"] >= a["n"]).all()
    # stopping distribution sums to one
    s = AD.policy_summary(F, a)
    assert np.isclose(sum(v for k, v in s.items() if k.startswith("stop_frac_")), 1.0)


def test_agreement_condition_manual():
    class Q:  # tiny hand-built feature set: one question, one order, checkpoints (4, 8, 16)
        pass

    F = AD.Features(["q"], [4, 8, 16], 32,
                    sc_gid=np.array([[[1, 1, 1]]], np.int16), unique_leader=np.array([[[True, True, False]]]),
                    leader_frac=np.array([[[0.5, 0.75, 1.0]]]), leader_is_na=np.array([[[False, False, False]]]),
                    rwc_gid=np.array([[[1, 2, 1]]], np.int16), margin=np.array([[[0.3, 0.3, 0.3]]]),
                    ok=np.ones((1, 1, 4), bool), cost=np.array([[[10, 20, 40, 80]]]), n_used=np.array([[[4, 8, 16, 32]]]),
                    sel_gid=np.ones((1, 1, 4), np.int16), p_correct_pool=np.array([0.5]))
    assert AD.run_policy(F, 0.5, None)["n"][0, 0] == 4
    assert AD.run_policy(F, 0.75, None)["n"][0, 0] == 8
    assert AD.run_policy(F, 1.0, None)["n"][0, 0] == 32      # at n=16 the leader is not unique -> no stop
    assert AD.run_policy(F, 0.5, 0.0)["n"][0, 0] == 4        # sc == rwc at n=4
    assert AD.run_policy(F, 0.75, 0.0)["n"][0, 0] == 32      # sc != rwc at 8, leader tied at 16
    assert AD.run_policy(F, 0.5, 0.5)["n"][0, 0] == 32       # margin below gamma everywhere
    F.leader_is_na[:] = True
    assert AD.run_policy(F, 0.5, None)["n"][0, 0] == 32      # NO_ANSWER can never trigger a stop


def test_select_setting_rule_and_tie_breaks():
    rows = [
        {"family": "fixed_32", "tau": np.nan, "gamma": np.nan, "accuracy": 0.80, "mean_tokens": 1000, "mean_rollouts": 32},
        {"family": "agreement", "tau": 0.5, "gamma": np.nan, "accuracy": 0.78, "mean_tokens": 300, "mean_rollouts": 6},    # 2 pp below: infeasible
        {"family": "agreement", "tau": 0.625, "gamma": np.nan, "accuracy": 0.795, "mean_tokens": 400, "mean_rollouts": 7},
        {"family": "agreement", "tau": 0.75, "gamma": np.nan, "accuracy": 0.795, "mean_tokens": 400, "mean_rollouts": 7},  # tie -> smaller tau
        {"family": "agreement", "tau": 0.875, "gamma": np.nan, "accuracy": 0.80, "mean_tokens": 400, "mean_rollouts": 7},  # tie cost -> higher acc
    ]
    sel = AD.select_setting(pd.DataFrame(rows), "agreement", 0.01)
    assert sel["tau"] == 0.875 and not sel["is_fixed_32_fallback"]
    only_fixed = AD.select_setting(pd.DataFrame(rows[:2]), "agreement", 0.01)
    assert only_fixed["is_fixed_32_fallback"] and only_fixed["tau"] is None


def test_grid_and_heldout_report(math_pool):
    qids = math_pool.qids()
    dev, ev = qids[:8], qids[8:]
    orders = OR.make_orders(qids, 3, 32, 6304)
    grid = AD.evaluate_grid(AD.build_features(math_pool, dev, orders), [0.5, 0.75, 1.0], [0.0, 0.25])
    assert (grid.family == "fixed_32").sum() == 1 and len(grid) == 1 + 3 + 6
    selected = {f: AD.select_setting(grid, f, 0.01) for f in ("agreement", "quality_aware")}
    res = OE.evaluate_pool(math_pool, ev, orders, BUDGETS)
    rep = AD.heldout_report(AD.build_features(math_pool, ev, orders), selected, res, 50)
    assert {"summary", "diffs", "stop_distribution", "wrong_stops", "by_pool_fraction"} <= set(rep)
    assert (rep["summary"].policy.str.startswith("fixed_N")).sum() == len(BUDGETS)


def test_signal_analysis_text_and_vision(math_pool, vis_pool):
    t = SA.question_signal_table(math_pool, math_pool.qids())
    s = SA.summarize_quartiles(t, 50)
    assert {"length", "logp", "selfcert"} <= set(s.signal)
    tv = SA.question_signal_table(vis_pool, vis_pool.qids())
    assert {"vision", "visconf", "tokenfusion"} <= set(SA.summarize_quartiles(tv, 50).signal)
    cv = SA.confidence_vision(vis_pool, vis_pool.qids())
    assert cv["summary"]["fusion_identity"]["holds_within_1e-8"]
    assert len(SA.length_control_table(tv, {"x": 0.5})) == 3
    assert isinstance(SA.select_illustrative_trajectories(vis_pool, vis_pool.qids()), list)


def test_quartile_diff_known():
    score = np.arange(32, dtype=float)
    corr = (score >= 24).astype(float)               # top 8 correct, bottom 8 wrong
    assert SA._quartile_diff(score, corr, 0.25) == 1.0
    assert SA._quartile_diff(-score, corr, 0.25) == -1.0
