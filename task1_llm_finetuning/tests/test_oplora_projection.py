import torch
from task1_llm_finetuning.src.lora import inject_lora
from task1_llm_finetuning.src.oplora import compute_bases, export_projected, verify_constraints
from task1_llm_finetuning.tests.tiny import tiny_cfg, tiny_model

T = tiny_cfg("oplora")["lora"]["targets"]


def _setup(k=8, r=4):
    m = tiny_model()
    mods = inject_lora(m, T, r, 2 * r, 0.0, seed=6304)
    compute_bases(mods, k, "cpu", torch.device("cpu"))
    for mod in mods.values():  # make B non-zero so the projection is non-trivial
        mod.lora_B.data.normal_(0, 0.1)
    return m, mods


def test_constraints_hold_and_match_dense_formula():
    _, mods = _setup()
    for n, mod in mods.items():
        A, B = mod.lora_A.detach(), mod.lora_B.detach()
        Pu = torch.eye(mod.out_features) - mod.Uk @ mod.Uk.T
        Pv = torch.eye(mod.in_features) - mod.Vk @ mod.Vk.T
        dense = mod.scale * (Pu @ B) @ (A @ Pv)
        assert torch.allclose(mod.delta_weight(), dense, atol=1e-5)
        assert (mod.Uk.T @ mod.delta_weight()).abs().max() < 1e-4
        assert (mod.delta_weight() @ mod.Vk).abs().max() < 1e-4
    rows = verify_constraints(mods)
    assert max(r["max_abs_UkT_Btilde"] for r in rows) < 1e-5
    assert max(r["max_abs_Atilde_Vk"] for r in rows) < 1e-5
    assert max(r["rho_out"] for r in rows) < 1e-6


def test_export_reproduces_delta_and_init_matches_lora():
    _, mods = _setup()
    ex = export_projected(mods)
    for n, mod in mods.items():
        assert torch.allclose(mod.scale * ex[f"{n}.B_eff"] @ ex[f"{n}.A_eff"], mod.delta_weight(), atol=1e-6)
    # identical initial A for LoRA vs OPLoRA (same seed/code path)
    a1 = inject_lora(tiny_model(), T, 4, 8, 0.0, 6304)
    a2 = inject_lora(tiny_model(), T, 4, 8, 0.0, 6304)
    for n in a1:
        assert torch.equal(a1[n].lora_A, a2[n].lora_A) and (a1[n].lora_B == 0).all()


def test_forward_matches_merged_weight():
    m, mods = _setup()
    x = torch.randn(3, 5, 64)
    mod = next(iter(mods.values()))
    mod.eval()
    got = mod(x) if mod.in_features == 64 else None
    if got is not None:
        want = x @ (mod.base.weight + mod.delta_weight()).T
        assert torch.allclose(got, want, atol=1e-4)
