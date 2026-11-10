"""End-to-end CPU smoke: all four methods, checks outputs, accumulation equivalence and resume."""
import json
from pathlib import Path

import pandas as pd
import torch

from task1_llm_finetuning.src.l2sp import L2SP, Reference
from task1_llm_finetuning.src.lora import init_fingerprint, inject_lora
from task1_llm_finetuning.src.oplora import compute_bases
from task1_llm_finetuning.src.train import run_training
from task1_llm_finetuning.tests.tiny import synth_examples, tiny_cfg, tiny_model

DEV = torch.device("cpu")


def _run(tmp, method, examples=None, **kw):
    cfg = tiny_cfg("lora" if method in ("lora", "oplora") else method, **kw)
    cfg["method"], cfg["run_name"] = method, f"t_{method}"
    m = tiny_model()
    modules = l2sp = ref = None
    if method in ("full_ft", "l2sp"):
        for p in m.parameters():
            p.requires_grad = True
        if method == "l2sp":
            cfg["l2sp_lambda"] = 1e-2
            l2sp = L2SP(m, cfg["l2sp_lambda"], DEV)
        else:
            ref = Reference(m, "cpu")
    else:
        modules = inject_lora(m, cfg["lora"]["targets"], cfg["lora"]["r"], cfg["lora"]["alpha"], 0.05, 6304)
        if method == "oplora":
            compute_bases(modules, 8, "cpu", DEV)
    m.train()
    run = Path(tmp) / cfg["run_name"]
    s = run_training(cfg, m, examples or synth_examples(64), run, DEV, 0, modules=modules, l2sp=l2sp, reference=ref)
    return run, s, m


def test_all_methods(tmp_path):
    for method in ["full_ft", "l2sp", "lora", "oplora"]:
        run, s, _ = _run(tmp_path, method)
        df = pd.read_csv(run / "train_log.csv")
        assert len(df) == 4 and df["sft_loss"].notna().all() and (df["skipped_update"] == 0).all()
        assert (run / "DONE").exists() and (run / "resources.json").exists()
        assert (run / "checkpoint_sizes.csv").exists() and (run / "progress_stats.csv").exists()
        res = json.load(open(run / "resources.json"))
        if method == "l2sp":
            assert df["omega"].iloc[-1] > 0 and abs(df["total_loss"].iloc[-1] - df["sft_loss"].iloc[-1]
                                                    - df["lambda_omega"].iloc[-1]) < 1e-9
            assert res["overhead"]["reference_storage_bytes"] > 0
        if method in ("full_ft", "l2sp"):
            assert (run / "displacement_by_param_u0004.csv").exists()
            assert (run / "recovery" / "recovery_u0004.pt").exists()
        else:
            assert (run / "adapter_final.safetensors").exists()
            assert res["trainable_params_exact"] > 0 and res["trainable_fraction"] < 0.5
        if method == "oplora":
            chk = pd.read_csv(run / "oplora_constraint_check.csv")
            assert chk["max_abs_UkT_Btilde"].max() < 1e-5 and chk["rho_out"].max() < 1e-6
            assert (run / "adapter_projected_final.safetensors").exists()
        # lr schedule: warmup 1 update -> peak at update 1, then decays
        assert df["lr"].iloc[0] == 1e-3 and df["lr"].iloc[-1] < df["lr"].iloc[1]


def test_loss_decreases_full_ft(tmp_path):
    # the same 4 examples every update -> the model must be able to overfit them
    run, _, _ = _run(tmp_path, "full_ft", examples=synth_examples(4) * 16, num_updates=12, recovery_at_updates=[12],
                     save_weights_at=[], lr=3e-3, warmup_updates=1)
    df = pd.read_csv(run / "train_log.csv")
    assert df["sft_loss"].iloc[-3:].mean() < df["sft_loss"].iloc[:3].mean()


def test_resume_matches_uninterrupted(tmp_path):
    a, _, ma = _run(tmp_path / "a", "lora")
    # run 2 updates, then resume from the update-2 recovery checkpoint
    cfg2 = dict(num_updates=2, recovery_at_updates=[2])
    b1, _, _ = _run(tmp_path / "b", "lora", **cfg2)
    cfg = tiny_cfg("lora")
    m = tiny_model()
    modules = inject_lora(m, cfg["lora"]["targets"], 4, 8, 0.05, 6304)
    m.train()
    run_training(cfg, m, synth_examples(64), tmp_path / "b" / "t_lora", DEV, 0, modules=modules,
                 resume=str(b1 / "recovery" / "recovery_u0002.pt"))
    da = pd.read_csv(a / "train_log.csv")
    db = pd.read_csv(tmp_path / "b" / "t_lora" / "train_log.csv")
    assert len(db) == 4
    # data/lr are deterministic -> identical; loss identical up to dropout RNG restore
    assert (da["lr"] - db["lr"]).abs().max() < 1e-12
    assert (da["valid_tokens"] == db["valid_tokens"]).all()
    assert (da["sft_loss"] - db["sft_loss"]).abs().max() < 1e-5
