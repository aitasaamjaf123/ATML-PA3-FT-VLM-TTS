"""Task 1 training: Full FT / Full FT + L2-SP / LoRA / OPLoRA with one shared recipe.

Run from the repo root:
    python -m task1_llm_finetuning.src.train --config task1_llm_finetuning/configs/full_ft.yaml \
        --output_root /kaggle/working/task1_outputs
Smoke test (3 updates, tiny): add --smoke
Resume: add --resume <run_dir>/recovery/recovery_uXXXX.pt

Everything that is saved is listed in task1_llm_finetuning/README.md.
"""
import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.io import (CsvLogger, apply_overrides, deep_update, load_json, load_yaml, save_json, save_yaml,
                       sha256_file, write_rows_csv)
from common.logging_utils import env_info, get_git_commit, path_size_bytes, pip_freeze, tee_stdout
from common.seed import capture_rng_state, restore_rng_state, set_seed

from .data import IGNORE, collate, count_valid_targets, encode_dataset, formatted_length, load_jsonl, tokens_sha256
from .l2sp import L2SP, Reference
from .lora import (LoRALinear, adapter_delta_rows, adapter_parameters, adapter_state, init_fingerprint,
                   inject_lora, load_adapter_state)
from .model import (count_params, enable_checkpointing, load_model_fp32_cpu, load_tokenizer, resolved_revision,
                    weights_fingerprint)
from .oplora import bases_state, compute_bases, export_projected, verify_constraints

TRAIN_COLUMNS = [
    "run_name", "method", "update", "examples_seen", "lr", "sft_loss", "grad_norm_preclip", "grad_scale",
    "skipped_update", "loss_finite", "valid_tokens", "real_tokens", "padded_tokens", "update_time_sec",
    "train_time_cum_sec", "wall_time_sec", "peak_mem_alloc_gb", "peak_mem_reserved_gb",
    "omega", "lambda_omega", "total_loss", "displacement_pre_step", "penalty_time_sec",
]
STATS_COLUMNS = ["update", "examples_seen", "displacement_l2", "eta_mean", "eta_max", "delta_fro_total"]
CKPT_COLUMNS = ["update", "examples_seen", "artifact", "path", "bytes", "save_seconds"]
GB = 2 ** 30


# --------------------------------------------------------------------------------------------- helpers
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def lr_lambda_factory(warmup: int, total: int):
    """Linear warmup (update t uses (t+1)/warmup of peak) then cosine decay to 0."""
    def f(t):
        if t < warmup:
            return (t + 1) / warmup
        prog = (t - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return f


def sft_loss_sum(model, batch, chunk_tokens: int):
    """Sum of next-token CE over supervised tokens. Mathematically identical to full-vocab CE on all positions
    with labels masked, but lm_head/CE are only evaluated on supervised positions, in checkpointed chunks
    (keeps the 2048 x 151,936 logit tensor from ever being materialised)."""
    hid = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                      use_cache=False).last_hidden_state
    tgt = batch["labels"][:, 1:]
    sel = tgt != IGNORE
    h = hid[:, :-1][sel]
    t = tgt[sel]
    head = model.lm_head

    def ce(hc, tc):
        return F.cross_entropy(head(hc).float(), tc, reduction="sum")

    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, t.numel(), chunk_tokens):
        total = total + checkpoint(ce, h[i:i + chunk_tokens], t[i:i + chunk_tokens], use_reentrant=False)
    return total


def make_optimizer(named_params, cfg, device):
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or n.endswith(".bias")) else decay).append(p)
    o = cfg["optimizer"]
    groups = [{"params": decay, "weight_decay": o["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                            fused=bool(o.get("fused", True)) and device.type == "cuda")
    info = {"n_decay_tensors": len(decay), "n_no_decay_tensors": len(no_decay),
            "n_decay_params": sum(p.numel() for p in decay), "n_no_decay_params": sum(p.numel() for p in no_decay)}
    return opt, info


def make_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def atomic_torch_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def trainable_state(model, modules):
    if modules is None:
        return model.state_dict()
    return adapter_state(modules)


# --------------------------------------------------------------------------------------------- core loop
def run_training(cfg, model, examples, run_dir: Path, device, pad_id, tokenizer=None, modules=None, l2sp=None,
                 reference=None, resume=None, extra_meta=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    method = cfg["method"]
    full_param = modules is None
    use_amp = device.type == "cuda" and cfg.get("precision", "fp16") == "fp16"
    mb, accum, n_upd = cfg["micro_batch_size"], cfg["grad_accum"], cfg["num_updates"]
    eff_bs = mb * accum
    assert len(examples) >= n_upd * eff_bs, "not enough examples for the requested number of updates"
    recovery_at = set(cfg["recovery_at_updates"])
    weights_at = set(cfg.get("save_weights_at", []))
    stats_every = int(cfg.get("stats_every", 125))

    named = list(model.named_parameters())
    params = [p for _, p in named if p.requires_grad]
    optimizer, opt_info = make_optimizer(named, cfg, device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(cfg["warmup_updates"], n_upd))
    scaler = make_scaler(use_amp)

    start_update, train_time_cum, wall_offset, peak_alloc_prev, peak_res_prev = 0, 0.0, 0.0, 0.0, 0.0
    ckpt_time_cum = 0.0
    if resume:
        st = torch.load(resume, map_location="cpu", weights_only=False)
        if modules is None:
            model.load_state_dict(st["model"])
        else:
            load_adapter_state(modules, st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        restore_rng_state(st["rng"])
        start_update, train_time_cum = st["update"], st["train_time_cum"]
        lam_fn = lr_lambda_factory(cfg["warmup_updates"], n_upd)
        for g in optimizer.param_groups:  # re-derive LR from the schedule (never trust the stored value)
            g["lr"] = cfg["lr"] * lam_fn(start_update)
        wall_offset, peak_alloc_prev, peak_res_prev = st["wall_time"], st["peak_alloc_gb"], st["peak_reserved_gb"]
        ckpt_time_cum = st.get("ckpt_time_cum", 0.0)
        print(f"[resume] from {resume} at update {start_update}")

    append = start_update > 0
    train_log = CsvLogger(run_dir / "train_log.csv", TRAIN_COLUMNS, append=append)
    stats_log = CsvLogger(run_dir / "progress_stats.csv", STATS_COLUMNS, append=append)
    ckpt_log = CsvLogger(run_dir / "checkpoint_sizes.csv", CKPT_COLUMNS, append=append)
    (run_dir / "recovery").mkdir(exist_ok=True)

    skipped_updates, nonfinite_updates = [], []
    losses_tail = []
    last_stats = {}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_wall0 = time.perf_counter()
    model.train()
    print(f"[train] {method} | updates {start_update}->{n_upd} | eff batch {eff_bs} | trainable "
          f"{sum(p.numel() for p in params):,} | amp={use_amp}")

    def log_ckpt(u, artifact, path, secs):
        ckpt_log.write({"update": u, "examples_seen": u * eff_bs, "artifact": artifact, "path": str(path),
                        "bytes": path_size_bytes(path), "save_seconds": round(secs, 3)})

    def stats_now(u):
        row = {"update": u, "examples_seen": u * eff_bs}
        if full_param:
            _, disp = (reference or l2sp).displacement_rows()
            row["displacement_l2"] = disp
        else:
            rows = adapter_delta_rows(modules)
            etas = [r["eta"] for r in rows]
            row.update(eta_mean=sum(etas) / len(etas), eta_max=max(etas),
                       delta_fro_total=math.sqrt(sum(r["delta_fro"] ** 2 for r in rows)))
        stats_log.write(row)
        return row

    for u in range(start_update, n_upd):
        sync(device)
        t0 = time.perf_counter()
        lr_now = optimizer.param_groups[0]["lr"]
        base = u * eff_bs
        mbs = [collate(examples[base + i * mb: base + (i + 1) * mb], pad_id) for i in range(accum)]
        total_valid = sum(count_valid_targets(b["labels"]) for b in mbs)
        real_tokens = sum(int(b["attention_mask"].sum()) for b in mbs)
        padded_tokens = sum(b["attention_mask"].numel() for b in mbs) - real_tokens

        optimizer.zero_grad(set_to_none=True)
        loss_acc = torch.zeros((), device=device)
        for b in mbs:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss_sum = sft_loss_sum(model, b, cfg["loss_chunk_tokens"])
            loss = loss_sum / total_valid  # normalised over ALL valid response tokens of the effective batch
            scaler.scale(loss).backward()
            loss_acc += loss.detach()

        omega, pen_t = "", ""
        if l2sp is not None:
            omega = l2sp.add_penalty_grads(scaler.get_scale() if use_amp else 1.0)
            pen_t = l2sp.last_penalty_seconds

        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"]).item()
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        scheduler.step()

        sft = loss_acc.item()
        skipped = (not math.isfinite(gnorm)) or (new_scale < old_scale)
        if skipped:
            skipped_updates.append(u + 1)
        if not math.isfinite(sft):
            nonfinite_updates.append(u + 1)
        sync(device)
        dt = time.perf_counter() - t0
        train_time_cum += dt
        peak_a = max(peak_alloc_prev, torch.cuda.max_memory_allocated() / GB) if device.type == "cuda" else 0.0
        peak_r = max(peak_res_prev, torch.cuda.max_memory_reserved() / GB) if device.type == "cuda" else 0.0
        row = {
            "run_name": cfg["run_name"], "method": method, "update": u + 1, "examples_seen": (u + 1) * eff_bs,
            "lr": lr_now, "sft_loss": sft, "grad_norm_preclip": gnorm, "grad_scale": old_scale,
            "skipped_update": int(skipped), "loss_finite": int(math.isfinite(sft)), "valid_tokens": total_valid,
            "real_tokens": real_tokens, "padded_tokens": padded_tokens, "update_time_sec": dt,
            "train_time_cum_sec": train_time_cum, "wall_time_sec": wall_offset + time.perf_counter() - t_wall0,
            "peak_mem_alloc_gb": peak_a, "peak_mem_reserved_gb": peak_r,
        }
        if l2sp is not None:
            row.update(omega=omega, lambda_omega=l2sp.lam * omega, total_loss=sft + l2sp.lam * omega,
                       displacement_pre_step=math.sqrt(2.0 * omega), penalty_time_sec=pen_t)
        train_log.write(row)
        losses_tail.append(sft)
        if (u + 1) % 10 == 0 or u == start_update:
            print(f"[u {u + 1:4d}/{n_upd}] sft {sft:.4f} lr {lr_now:.2e} gn {gnorm:.2f} scale {old_scale:.0f} "
                  f"{'SKIP ' if skipped else ''}{dt:.1f}s mem {peak_a:.1f}G")

        done = u + 1
        do_ckpt = done in recovery_at
        if do_ckpt or (stats_every and done % stats_every == 0) or done == n_upd:
            last_stats = stats_now(done)
        if do_ckpt:
            tc = time.perf_counter()
            if full_param:
                rows, _ = (reference or l2sp).displacement_rows()
                write_rows_csv(run_dir / f"displacement_by_param_u{done:04d}.csv", rows)
            else:
                write_rows_csv(run_dir / f"adapter_delta_by_module_u{done:04d}.csv", adapter_delta_rows(modules))
            if done in weights_at and full_param:
                wdir = run_dir / f"weights_u{done:04d}"
                ts = time.perf_counter()
                model.save_pretrained(wdir, safe_serialization=True)
                if tokenizer is not None:
                    tokenizer.save_pretrained(wdir)
                log_ckpt(done, "model_weights_fp32", wdir, time.perf_counter() - ts)
            is_final = done == n_upd
            state = {
                "update": done, "examples_seen": done * eff_bs, "model": trainable_state(model, modules),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": capture_rng_state(),
                "train_time_cum": train_time_cum, "wall_time": wall_offset + time.perf_counter() - t_wall0,
                "peak_alloc_gb": peak_a, "peak_reserved_gb": peak_r, "ckpt_time_cum": ckpt_time_cum,
                "config": cfg, "data_position": {"next_example_index": done * eff_bs},
            }
            keep_opt = not (is_final and full_param and not cfg.get("save_final_optimizer", False))
            if keep_opt:
                state["optimizer"] = optimizer.state_dict()
            rp = run_dir / "recovery" / f"recovery_u{done:04d}.pt"
            ts = time.perf_counter()
            atomic_torch_save(state, rp)
            log_ckpt(done, "recovery_full_state" if keep_opt else "recovery_state_no_optimizer", rp,
                     time.perf_counter() - ts)
            if not cfg.get("keep_all_recovery", True):
                for old in sorted((run_dir / "recovery").glob("recovery_u*.pt")):
                    if old != rp:
                        old.unlink()
            ckpt_time_cum += time.perf_counter() - tc

    # ------------------------------------------------------------------ end-of-run artifacts
    overhead = dict((extra_meta or {}).get("overhead", {}))
    final_artifacts = {}
    if not full_param:
        ts = time.perf_counter()
        from safetensors.torch import save_file
        st = adapter_state(modules)
        p = run_dir / "adapter_final.safetensors"
        save_file(st, str(p))
        final_artifacts["adapter_raw_bytes"] = p.stat().st_size
        log_ckpt(n_upd, "adapter_raw_final", p, time.perf_counter() - ts)
        save_json({"method": method, "r": cfg["lora"]["r"], "alpha": cfg["lora"]["alpha"],
                   "dropout": cfg["lora"]["dropout"], "targets": cfg["lora"]["targets"],
                   "scale": cfg["lora"]["alpha"] / cfg["lora"]["r"], "module_names": list(modules.keys()),
                   "oplora_k": cfg.get("oplora", {}).get("k") if method == "oplora" else None},
                  run_dir / "adapter_config.json")
        write_rows_csv(run_dir / "adapter_delta_by_module_final.csv", adapter_delta_rows(modules))
        if method == "oplora":
            ts = time.perf_counter()
            pr = run_dir / "adapter_projected_final.safetensors"
            save_file(export_projected(modules), str(pr))
            overhead["export_projected_seconds"] = time.perf_counter() - ts
            overhead["export_projected_bytes"] = pr.stat().st_size
            log_ckpt(n_upd, "adapter_projected_final", pr, overhead["export_projected_seconds"])
            tv = time.perf_counter()
            vrows = verify_constraints(modules)
            overhead["verify_constraints_seconds"] = time.perf_counter() - tv
            write_rows_csv(run_dir / "oplora_constraint_check.csv", vrows)
            overhead["max_abs_UkT_Btilde"] = max(r["max_abs_UkT_Btilde"] for r in vrows)
            overhead["max_abs_Atilde_Vk"] = max(r["max_abs_Atilde_Vk"] for r in vrows)
    if l2sp is not None:
        overhead.update(reference_storage_bytes=l2sp.storage_bytes, reference_transfer_seconds=l2sp.transfer_seconds,
                        penalty_seconds_total=l2sp.penalty_seconds_total)
    if reference is not None and l2sp is None:
        overhead.update(reference_storage_bytes_cpu=reference.storage_bytes,
                        reference_copy_seconds=reference.transfer_seconds)

    import pandas as _pd  # light summary only; plotting code is NOT part of this script
    df = _pd.read_csv(run_dir / "train_log.csv")
    ck = _pd.read_csv(run_dir / "checkpoint_sizes.csv")
    wall = wall_offset + time.perf_counter() - t_wall0
    resources = {
        "run_name": cfg["run_name"], "method": method, **count_params(model),
        "trainable_params_exact": sum(p.numel() for p in params),
        "train_time_sec": float(df["update_time_sec"].sum()), "wall_time_sec": wall,
        "checkpointing_time_sec": ckpt_time_cum, "avg_update_time_sec": float(df["update_time_sec"].mean()),
        "real_tokens_total": int(df["real_tokens"].sum()),
        "real_tokens_per_sec": float(df["real_tokens"].sum() / max(df["update_time_sec"].sum(), 1e-9)),
        "peak_gpu_mem_allocated_gb": float(df["peak_mem_alloc_gb"].max()),
        "peak_gpu_mem_reserved_gb": float(df["peak_mem_reserved_gb"].max()),
        "checkpoint_sizes_bytes": {f"{r.artifact}@u{r.update}": int(r.bytes) for r in ck.itertuples()},
        "final_artifact_bytes": final_artifacts, "overhead": overhead, "optimizer_groups": opt_info,
        "gpu": env_info()["gpu"],
    }
    save_json(resources, run_dir / "resources.json")
    summary = {
        "completed": True, "updates_done": n_upd, "examples_seen": n_upd * eff_bs,
        "final_sft_loss": float(df["sft_loss"].iloc[-1]), "mean_sft_loss_last_50": float(df["sft_loss"].tail(50).mean()),
        "skipped_update_count": len(skipped_updates), "skipped_updates": skipped_updates,
        "nonfinite_loss_updates": nonfinite_updates, "final_lr": float(df["lr"].iloc[-1]), "final_stats": last_stats,
    }
    if l2sp is not None:
        summary["final_omega"] = float(df["omega"].iloc[-1])
    save_json(summary, run_dir / "run_summary.json")
    (run_dir / "DONE").write_text("ok\n")
    print("[done]", summary)
    return summary


# --------------------------------------------------------------------------------------------- entry point
def build_config(args):
    shared = load_yaml(REPO_ROOT / args.shared)
    method_cfg = load_yaml(args.config)
    cfg = deep_update(shared, method_cfg)
    if args.smoke:
        cfg = deep_update(cfg, {"num_updates": 3, "warmup_updates": 1, "recovery_at_updates": [2, 3],
                                "save_weights_at": [2, 3], "stats_every": 1, "run_name": cfg["run_name"] + "_smoke"})
    return apply_overrides(cfg, args.override)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="method yaml (full_ft/l2sp/lora4/lora16/oplora16)")
    ap.add_argument("--shared", default="task1_llm_finetuning/configs/shared.yaml")
    ap.add_argument("--output_root", default=str(REPO_ROOT / "task1_llm_finetuning" / "outputs" / "runs"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip_hash_check", action="store_true")
    ap.add_argument("--override", nargs="*", default=[], help="dotted.key=value (YAML-parsed)")
    args = ap.parse_args()

    cfg = build_config(args)
    run_dir = Path(args.output_root) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(run_dir / "train_stdout.log")
    save_yaml(cfg, run_dir / "config_resolved.yaml")
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = cfg["method"]
    assert method in ("full_ft", "l2sp", "lora", "oplora")
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[warn] {torch.cuda.device_count()} GPUs visible; using cuda:0 only (set CUDA_VISIBLE_DEVICES=0)")

    # ---- data (identical file + identical tokenization verified via hashes)
    tok = load_tokenizer(cfg["tokenizer_name_or_path"] or cfg["model_name_or_path"], cfg.get("model_revision"))
    train_path = REPO_ROOT / cfg["train_data"]
    records = load_jsonl(train_path)
    examples = encode_dataset(tok, records, enable_thinking=cfg["enable_thinking"])
    assert all(formatted_length(e["ids"]) <= cfg["max_seq_len"] for e in examples), "found > max_seq_len example"
    manifest = load_json(REPO_ROOT / cfg["train_manifest"])
    data_meta = {"n_examples": len(examples), "train_file_sha256": sha256_file(train_path),
                 "tokens_sha256": tokens_sha256(examples)}
    if not (args.smoke or args.skip_hash_check):
        assert len(examples) == cfg["num_train_examples"] == manifest["num_examples"], "example count mismatch"
        assert data_meta["train_file_sha256"] == manifest["train_file_sha256"], "train file differs from manifest"
        assert data_meta["tokens_sha256"] == manifest["tokens_sha256"], \
            "TOKENIZATION DIFFERS from the reference run (tokenizer/template/transformers version?)"
    save_json({"data_meta": data_meta, "manifest": manifest}, run_dir / "data_check.json")

    # ---- model
    model = load_model_fp32_cpu(cfg["model_name_or_path"], cfg.get("model_revision"))
    fingerprint = weights_fingerprint(model)
    revision = resolved_revision(model)
    print("[model] revision", revision, "| init weights fingerprint", fingerprint[:16])

    modules, l2sp, reference, overhead, lora_fp = None, None, None, {}, None
    if method in ("full_ft", "l2sp"):
        model.to(device)
        for p in model.parameters():
            p.requires_grad = True
        if method == "l2sp":
            ref_dev = cfg.get("l2sp_reference_device", "cpu")
            ref_dev = torch.device("cpu" if (ref_dev == "cpu" or device.type != "cuda") else "cuda")
            l2sp = L2SP(model, cfg["l2sp_lambda"], ref_dev)
        else:
            reference = Reference(model, "cpu")  # theta0 on host, only for displacement logging
    else:
        model.to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        lc = cfg["lora"]
        modules = inject_lora(model, lc["targets"], lc["r"], lc["alpha"], lc["dropout"], cfg["seed"])
        lora_fp = init_fingerprint(modules)
        if method == "oplora":
            oc = cfg["oplora"]
            rows, svd_t = compute_bases(modules, oc["k"], oc.get("svd_device", "cpu"), device)
            write_rows_csv(run_dir / "oplora_bases_meta.csv", rows)
            from safetensors.torch import save_file
            bp = run_dir / "oplora_bases.safetensors"
            save_file(bases_state(modules), str(bp))
            overhead.update(basis_preprocessing_seconds=svd_t, basis_file_bytes=bp.stat().st_size,
                            basis_gpu_bytes=sum(m.Uk.numel() * 4 + m.Vk.numel() * 4 for m in modules.values()))
    enable_checkpointing(model)
    pc = count_params(model)
    print("[params]", pc)

    save_json({
        "argv": sys.argv, "config": cfg, "git_commit": get_git_commit(REPO_ROOT), "env": env_info(),
        "model_resolved_revision": revision, "init_weights_fingerprint": fingerprint,
        "lora_A_init_abs_sum": lora_fp, "param_counts": pc, "data_meta": data_meta, "smoke": args.smoke,
        "n_target_modules": None if modules is None else len(modules),
    }, run_dir / "run_meta.json")
    (run_dir / "pip_freeze.txt").write_text(pip_freeze())

    run_training(cfg, model, examples, run_dir, device, tok.pad_token_id, tokenizer=tok, modules=modules,
                 l2sp=l2sp, reference=reference, resume=args.resume, extra_meta={"overhead": overhead})


if __name__ == "__main__":
    main()
