"""Sanity tests on a tiny random Qwen3 (28 layers, hidden 64). No internet / lm-eval needed.
Run:  python -m pytest task1_llm_finetuning/tests/test_eval_utils.py -q
"""
import json

import pytest
import torch
import yaml

from task1_llm_finetuning.src import eval_utils as eu
from task1_llm_finetuning.src import diag_weight
from task1_llm_finetuning.src.eval_utils import (Bases, Ctx, DeltaSource, attach_unmerged_hooks, load_adapter,
                                                  load_qualitative_prompts, load_probe_ids, merge_adapter_into,
                                                  module_path, project, resolve_conditions)

K = 16


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=28,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                      tie_word_embeddings=True)
    d = tmp_path_factory.mktemp("tiny")
    Qwen3ForCausalLM(cfg).save_pretrained(d)
    eu.MODEL_ID = str(d)
    eu.PROTECT_K = K
    return d


def make_adapter(ctx, r, path, peft_keys=True):
    sd = {}
    for l in range(28):
        for name in eu.TARGET_MODULES:
            out_f, in_f = ctx.start.get(l, name).shape
            p = module_path(l, name)
            pre = f"base_model.model.{p}.lora_" if peft_keys else f"{p}.lora_"
            suf = ".default.weight" if peft_keys else ""
            sd[f"{pre}A{suf}"] = torch.randn(r, in_f) * 0.05
            sd[f"{pre}B{suf}"] = torch.randn(out_f, r) * 0.05  # non-zero B so the update is non-trivial
    torch.save({"adapter": sd}, path)


@pytest.mark.parametrize("oplora", [False, True])
def test_merge_equals_unmerged_and_oplora_constraints(tiny, tmp_path, oplora):
    ctx = Ctx()
    spec = dict(kind="adapter", r=4, alpha=8, oplora=oplora)
    make_adapter(ctx, 4, tmp_path / "adapter_final.pt")
    ad = load_adapter(tmp_path / "adapter_final.pt", 4)
    assert len(ad) == 196
    model = eu.load_base(torch.float32)
    ids = torch.randint(0, 200, (1, 20)).to(eu.DEVICE)
    with torch.no_grad():
        base = model(ids).logits
        hs = attach_unmerged_hooks(model, ad, spec, ctx)
        un = model(ids).logits
        for h in hs:
            h.remove()
        merge_adapter_into(model, ad, spec, ctx)
        mg = model(ids).logits
    assert (un - mg).abs().max() < 1e-4
    assert (base - mg).abs().max() > 1e-4  # the adapter actually changes the model
    if oplora:
        for (l, name), ab in list(ad.items())[:20]:
            Uk, Vk = ctx.bases.get(l, name)
            At, Bt = project(ab["A"].to(eu.DEVICE), ab["B"].to(eu.DEVICE), Uk, Vk)
            assert (Uk.T @ Bt).abs().max() < 1e-5 and (At @ Vk).abs().max() < 1e-5


def test_weight_diag_rho_eta(tiny, tmp_path):
    ctx = Ctx()
    # full-FT-style file: perturbation lying exactly inside the protected output subspace -> rho == 1
    sd = {k: v.clone() for k, v in ctx.start.sd.items()}
    Uk, _ = ctx.bases.get(0, "down_proj")
    pert = Uk @ torch.randn(K, 128).to(eu.DEVICE) * 0.01
    key = module_path(0, "down_proj") + ".weight"
    sd[key] = sd[key] + pert.cpu()
    torch.save(sd, tmp_path / "model_final_fp32.pt")
    src = DeltaSource("full_ft_final", tmp_path / "model_final_fp32.pt", ctx)
    rows = diag_weight.weight_rows("full_ft_final", src, ctx)
    r0 = [r for r in rows if r["layer"] == 0][0]
    assert abs(r0["rho_k"] - 1) < 1e-4 and r0["eta"] > 0
    assert all(r["fro_dW"] == 0 or r["layer"] == 0 for r in rows)
    disp = diag_weight.displacement_row("full_ft_final", tmp_path / "model_final_fp32.pt", ctx)
    assert abs(disp["l2_displacement"] - pert.norm().item()) < 1e-3 and disp["n_params_missing_in_file"] == 0


def test_resolve_conditions(tmp_path):
    for run, files in {"full_ft": ["weights/mid_fp32.pt", "weights/final_fp32.pt", "recovery/recovery_u0625.pt"],
                       "l2sp": ["weights/final_fp32.pt"],
                       "lora4": ["adapter_final.pt", "recovery/recovery_u1250.pt"],
                       "lora16": ["adapter_final.pt"],
                       "oplora16": ["adapter_final.pt", "projected_factors_final.pt", "svd_bases.pt"]}.items():
        for f in files:
            p = tmp_path / run / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
    r = resolve_conditions(tmp_path)
    assert r["full_ft_mid"].name == "mid_fp32.pt" and r["full_ft_final"].name == "final_fp32.pt"
    assert r["l2sp_final"].parent.parent.name == "l2sp"
    assert r["lora16_final"].parent.name == "lora16" and r["oplora16_final"].name == "adapter_final.pt"
    assert r["oplora16_final"].parent.name == "oplora16" and r["lora4_final"].parent.name == "lora4"


def test_qualitative_parser(tmp_path):
    qs = [f"question {i}" for i in range(8)]
    shapes = [
        {f: [f"{f}: {q}" for q in qs] for f in ("plain", "json", "template")},
        [{f: f"{f}: {q}" for f in ("plain", "json", "template")} for q in qs],
        {"questions": [{"format": f, "question": f"{f}: {q}"} for q in qs for f in ("plain", "json", "template")]},
    ]
    for s in shapes:
        (tmp_path / "q.yaml").write_text(yaml.safe_dump(s))
        rows = load_qualitative_prompts(tmp_path / "q.yaml")
        assert len(rows) == 24 and rows[0]["prompt_id"] == "q00_plain" and rows[-1]["question_idx"] == 7


def test_probe_ids(tmp_path):
    for obj in ({"gsm8k": [1, 2], "hellaswag": [3, 4]},
                {"gsm8k_test_ids": ["gsm8k_test_1"], "hellaswag_val": [{"idx": 3}]},
                {"probe": {"gsm8k": {"ids": [1]}, "hellaswag": {"ids": [2]}}}):
        (tmp_path / "p.json").write_text(json.dumps(obj))
        ids = load_probe_ids(tmp_path / "p.json")
        assert ids["gsm8k"] and ids["hellaswag"]
