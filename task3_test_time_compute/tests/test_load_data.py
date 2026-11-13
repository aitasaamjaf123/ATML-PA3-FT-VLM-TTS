import numpy as np
import pandas as pd
import pytest

from task3_test_time_compute.src import load_data as LD
from task3_test_time_compute.tests import synth


def _gen_prob(df, vision=False):
    gen = df[["dataset", "sample_id", "rollout_id", "text", "token_ids", "length", "extracted_answer", "gold_answer"]].copy()
    prob = df[["dataset", "sample_id", "rollout_id", "logp", "kl_u_p"]].copy()
    attn = df[["dataset", "sample_id", "rollout_id", "image_attn", "prompt_attn", "gen_attn"]].copy() if vision else None
    return gen, prob, attn


def test_join_and_alignment_ok():
    df = synth.make_joined_df("math", n_q=3)
    gen, prob, _ = _gen_prob(df)
    j = LD.join_frames(gen, prob.sample(frac=1.0, random_state=1), None, False)
    rep = LD.check_alignment(j, False)
    assert rep["n_misaligned"] == 0 and rep["n_questions"] == 3 and rep["rollouts_per_question_min"] == 32


def test_alignment_detects_length_mismatch():
    df = synth.make_joined_df("math", n_q=2)
    gen, prob, _ = _gen_prob(df)
    prob.at[5, "kl_u_p"] = prob.at[5, "kl_u_p"][:-1]
    j = LD.join_frames(gen, prob, None, False)
    with pytest.raises(LD.AlignmentError):
        LD.check_alignment(j, False)


def test_alignment_checks_attention_lengths():
    df = synth.make_joined_df("mathvista", n_q=2, vision=True)
    gen, prob, attn = _gen_prob(df, True)
    attn.at[3, "gen_attn"] = attn.at[3, "gen_attn"][:-2]
    j = LD.join_frames(gen, prob, attn, True)
    with pytest.raises(LD.AlignmentError):
        LD.check_alignment(j, True)


def test_join_lost_rows_raises():
    df = synth.make_joined_df("math", n_q=2)
    gen, prob, _ = _gen_prob(df)
    with pytest.raises(LD.AlignmentError):
        LD.join_frames(gen, prob.iloc[:-3], None, False)


def test_missing_column_message():
    df = synth.make_joined_df("math", n_q=1).rename(columns={"extracted_answer": "zzz"})
    gen, prob, _ = _gen_prob(df.rename(columns={"zzz": "extracted_answer"}))
    gen = gen.rename(columns={"extracted_answer": "weird"})
    with pytest.raises(KeyError, match="Available columns"):
        LD.join_frames(gen, prob, None, False)


def test_build_pool_signals_and_correctness():
    pool = synth.make_pool("math", n_q=5)
    qd = pool.questions["q000"]
    assert qd.n_rollouts == 32 and qd.gid.shape == (32,)
    assert qd.correct.dtype == bool
    # NO_ANSWER rollouts are in group 0 and incorrect
    assert not qd.correct[qd.gid == 0].any()
    # mean log-prob / self-certainty are the per-rollout means of the token tables
    df = synth.make_joined_df("math", n_q=5)
    r0 = df[(df.sample_id == "q000") & (df.rollout_id == 0)].iloc[0]
    assert np.isclose(qd.L[0], np.mean(r0.logp), atol=1e-5)
    assert np.isclose(qd.K[0], np.mean(r0.kl_u_p), atol=1e-5)
    assert qd.T[0] == r0.length
    assert qd.tok_k is None


def test_build_pool_vision_has_token_arrays():
    pool = synth.make_pool("mathvista", n_q=3, vision=True)
    qd = pool.questions["q001"]
    assert len(qd.tok_k) == 32 and len(qd.tok_v) == 32
    assert np.all((qd.tok_v[0] >= 0) & (qd.tok_v[0] <= 1))
    assert np.isclose(qd.V[3], qd.tok_v[3].mean(), atol=1e-6)


def test_readme_revision_parse():
    h = "a" * 40
    assert LD.parse_readme_revision(f"Model: Qwen/Qwen2.5-3B-Instruct revision {h}") == h
    assert LD.parse_readme_revision("nothing here") is None


def test_describe_pool():
    d = LD.describe_pool(synth.make_pool("math", n_q=4))
    assert d["n_questions"] == 4 and 0 <= d["mean_rollout_accuracy"] <= 1 and d["no_answer_rate"] >= 0
