import numpy as np
import pytest

from task3_test_time_compute.src import scores as S
from task3_test_time_compute.src import selectors as SEL


def test_vote_tie_breaks_to_earliest_supporting_rollout():
    assert SEL.vote_winner(np.array([2, 1, 1, 2])) == 2      # tie 2-2, group 2 appears first
    assert SEL.vote_winner(np.array([1, 2, 2, 1])) == 1
    assert SEL.vote_winner(np.array([3, 1, 1, 2])) == 1
    w, c, uniq = SEL.unique_leader(np.array([2, 1, 1, 2]))
    assert (w, c, uniq) == (2, 2, False)


def test_select_top_ties_earliest():
    assert SEL.select_top(np.array([5, 6, 7]), np.array([1.0, 3.0, 3.0])) == 6


def test_rwc_weights_known_example():
    score = np.array([3.0, 1.0, 2.0, 0.0])
    w = SEL.rank_weights(score)
    assert np.allclose(w, [1.0, 1 / 3, 2 / 3, 0.0])
    # groups A=1 (pos0), B=2 (pos1,2), C=3 (pos3): W(A)=1, W(B)=1, tie -> earliest supporting rollout (A)
    win, W, uniq = SEL.select_rwc(np.array([1, 2, 2, 3]), score)
    assert win == 1 and np.allclose(W, [1.0, 1.0, 0.0])


def test_rwc_average_ranks_for_tied_scores():
    w = SEL.rank_weights(np.array([1.0, 1.0, 0.0, 0.0]))
    assert np.allclose(w, [(4 - 1.5) / 3, (4 - 1.5) / 3, (4 - 3.5) / 3, (4 - 3.5) / 3])
    assert np.isclose(w.sum(), 4 / 2)  # total vote weight is N/2 for any score vector


def test_rwc_minority_rescue():
    # best-scored rollouts both say 7, which beats the 3 majority
    g = np.array([3, 3, 3, 7, 7])
    score = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    win, W, _ = SEL.select_rwc(g, score)
    assert win == 7
    assert SEL.vote_winner(g) == 3


def test_n1_returns_only_candidate_for_every_rule():
    g = np.array([4])
    for agg in ("first", "vote", "top", "rwc"):
        assert SEL.apply_method(agg, g, np.array([0.3])) == 4


def test_rwc_margin():
    assert SEL.rwc_margin(np.array([3.0])) == 1.0                       # single group: runner-up = 0
    assert np.isclose(SEL.rwc_margin(np.array([3.0, 1.0])), 0.5)
    assert np.isclose(SEL.rwc_margin(np.array([2.0, 2.0])), 0.0)


def test_visual_engagement_normalised_and_guarded():
    v = S.visual_engagement_tokens([1.0, 0.0], [1.0, 0.0], [2.0, 0.0])
    assert np.allclose(v, [0.25, 0.0])  # zero denominator guarded


def test_visconf_calibration_uses_only_candidate_prefix():
    rng = np.random.default_rng(0)
    tok_k = [rng.gamma(3, 1, 50) for _ in range(8)]
    tok_v = [rng.random(50) for _ in range(8)]
    a = S.prefix_visconf(tok_k[:4], tok_v[:4])
    tok_k2 = tok_k[:4] + [x * 100 + 1000 for x in tok_k[4:]]   # perturb the UNSEEN rollouts
    b = S.prefix_visconf(tok_k2[:4], tok_v[:4])
    assert np.allclose(a["Z"], b["Z"])
    full = S.prefix_visconf(tok_k2, tok_v)
    assert not np.allclose(full["Z"][:4], a["Z"])               # but calibration over the larger set differs


def test_visconf_matches_definition_and_linear_percentiles():
    rng = np.random.default_rng(1)
    tok_k = [rng.gamma(2, 1, n) for n in (30, 40, 25)]
    tok_v = [rng.random(len(k)) for k in tok_k]
    out = S.prefix_visconf(tok_k, tok_v)
    pooled = np.concatenate(tok_k)
    q05, q95 = np.percentile(pooled, [5, 95], method="linear")
    for i in range(3):
        c = np.clip((tok_k[i] - q05) / max(q95 - q05, 1e-8), 0, 1)
        assert np.isclose(out["C"][i], c.mean())
        assert np.isclose(out["Z"][i], tok_v[i].mean() * c.mean())


def test_tokenwise_fusion_identity_is_population_covariance():
    rng = np.random.default_rng(2)
    tok_k = [rng.gamma(2, 1, 60) for _ in range(6)]
    tok_v = [np.clip(0.3 + 0.2 * (k - k.mean()) / k.std() + rng.normal(0, 0.1, len(k)), 0, 1) for k in tok_k]
    out = S.prefix_visconf(tok_k, tok_v)
    pooled = np.concatenate(tok_k)
    q05, q95 = np.percentile(pooled, [5, 95])
    for i in range(6):
        c = np.clip((tok_k[i] - q05) / (q95 - q05), 0, 1)
        cov = np.mean((c - c.mean()) * (tok_v[i] - tok_v[i].mean()))
        assert np.isclose(out["Zt"][i] - out["Z"][i], cov)
        assert np.isclose(out["cov"][i], cov)


def test_empty_rollout_scores_are_neginf():
    assert np.isneginf(S.neginf(np.array([np.nan]))[0])
    assert np.isnan(S.rollout_scalars([], [])["L"])
