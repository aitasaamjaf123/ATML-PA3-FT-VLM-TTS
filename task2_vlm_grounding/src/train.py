"""Three-stage SigLIP-Qwen3 training: caption alignment -> visual instruction tuning -> natural VQA.

Usage:
  python -m task2_vlm_grounding.src.train [--smoke] [--resume auto|none|PATH]
"""
from __future__ import annotations

import argparse
import os
import platform
import random
import time
from pathlib import Path

import torch
import yaml

from common.io import append_jsonl, ensure_dir, write_json
from common.logging_utils import get_logger, peak_gpu_memory_mb, reset_peak_gpu_memory
from common.seed import get_rng_state, set_rng_state, set_seed
from task2_vlm_grounding.src.cache_features import FeatureStore
from task2_vlm_grounding.src.config import dev_eval_steps, get_paths, load_config, stage_list, total_updates
from task2_vlm_grounding.src.connector import calibrate_connector
from task2_vlm_grounding.src.evaluate_dev import run_dev_eval
from task2_vlm_grounding.src.manifest import load_manifest
from task2_vlm_grounding.src.sampler import CurriculumSampler, caption_to_example, pair_to_examples
from task2_vlm_grounding.src.schedule import lr_multiplier
from task2_vlm_grounding.src.split_checks import check_manifest
from task2_vlm_grounding.src.vlm import SigLipQwenVLM, autocast_ctx, make_grad_scaler

log = get_logger("train")


# ----------------------------------------------------------------------------- helpers
def build_optimizer(vlm: SigLipQwenVLM, stage: dict, tcfg: dict):
    groups = [{"params": list(vlm.connector.parameters()), "lr": stage["connector_lr"], "name": "connector"}]
    if stage["lora_lr"] is not None:
        groups.append({"params": vlm.lora_parameters(), "lr": stage["lora_lr"], "name": "lora"})
    opt = torch.optim.AdamW(groups, weight_decay=tcfg["weight_decay"])
    total, wf, ff = int(stage["updates"]), tcfg["warmup_frac"], tcfg["final_lr_frac"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lambda s: lr_multiplier(s, total, wf, ff)] * len(groups))
    return opt, sched


def save_atomic(obj, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    cands = sorted(ckpt_dir.glob("ckpt_*.pt"))
    return cands[-1] if cands else None


def prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    for old in sorted(ckpt_dir.glob("ckpt_*.pt"))[:-keep]:
        old.unlink()


def calibrate(vlm: SigLipQwenVLM, store: FeatureStore, manifest: dict, cfg: dict) -> dict:
    """Training-data-only magnitude calibration (see connector.CALIBRATION_RULE)."""
    cc = cfg["training"]["calibration"]
    rng = random.Random(f"{cfg['seed']}/calibration")
    caps = [caption_to_example(e) for e in rng.sample(manifest["caption_train"], min(cc["n_caption"], len(manifest["caption_train"])))]
    qa = []
    for key in ("grounding_train_presence", "grounding_train_counting", "vqa_train"):
        pool = manifest[key]
        for p in rng.sample(pool, min(cc["n_each_other"], len(pool))):
            qa.extend(pair_to_examples(p))
    text_norm = vlm.mean_text_embedding_norm(caps + qa)
    image_ids = sorted({e["image_id"] for e in caps + qa})
    info = calibrate_connector(vlm.connector, store.get(image_ids), text_norm)
    info["n_text_examples"] = len(caps) + len(qa)
    return info


def sanity_checks(vlm: SigLipQwenVLM, store: FeatureStore, manifest: dict, cfg: dict, results_dir: Path) -> dict:
    """Shapes, frozen parameter sets, connector gradients, label masking and split isolation."""
    report: dict = {"split_check": check_manifest(manifest, cfg)}
    ex = [caption_to_example(e) for e in manifest["caption_train"][:8]]
    feats = store.get([e["image_id"] for e in ex])
    assert feats.dtype == torch.float16 and tuple(feats.shape[1:]) == (196, 768)
    report["feature_shape"] = list(feats.shape)
    # frozen parameter sets (stage 1: connector only)
    assert not vlm.lora_enabled
    assert all(not p.requires_grad for p in vlm.qwen.parameters()), "Qwen must be frozen in stage 1"
    assert all(p.requires_grad and p.dtype == torch.float32 for p in vlm.connector.parameters())
    report["params_stage1"] = vlm.param_summary()
    # packing / masking
    report["packed_example"] = vlm.debug_sequence(ex[0], feats[0])
    pk = report["packed_example"]
    assert pk["n_visual_tokens"] == 196 and pk["positions_contiguous"] and pk["ends_with_im_end"]
    assert pk["visual_label_positions_masked"]
    assert pk["supervised_text"].endswith("<|im_end|>") and pk["n_supervised"] >= 2
    # gradients must reach the connector through the frozen backbone
    vlm.train_mode()
    with autocast_ctx(vlm.device_):
        loss0 = vlm.per_example_loss(ex, feats).mean()
    if not torch.isfinite(loss0):
        raise RuntimeError(f"forward loss is {float(loss0)} in FP16 autocast: activations overflow before any "
                           "gradient is computed (check feature cache / calibration scale)")
    report["initial_caption_loss_batch"] = float(loss0)
    grads, used_scale = None, None
    for scale in (1024.0, 64.0, 1.0):          # emulate GradScaler back-off: FP16 grads may overflow at large scales
        vlm.connector.zero_grad(set_to_none=True)
        with autocast_ctx(vlm.device_):
            loss = vlm.per_example_loss(ex, feats).mean()
        (loss * scale).backward()
        g = [p.grad for p in vlm.connector.parameters()]
        if all(x is not None and torch.isfinite(x).all() for x in g):
            grads, used_scale = g, scale
            break
    assert grads is not None, "connector gradients missing / non-finite at every loss scale (1024, 64, 1)"
    gmax = max(float(x.abs().max()) for x in grads)
    assert gmax > 0.0, "connector gradients are all zero"
    assert all(p.grad is None for p in vlm.qwen.parameters()), "frozen Qwen parameters received gradients"
    vlm.connector.zero_grad(set_to_none=True)
    report["connector_grad_abs_max"] = gmax / used_scale
    report["sanity_loss_scale_used"] = used_scale
    write_json(report, results_dir / "sanity_checks.json")
    return report


def env_info() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "platform": platform.platform()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 2**20
    try:
        import peft
        import transformers
        info["transformers"], info["peft"] = transformers.__version__, peft.__version__
    except ImportError:
        pass
    return info


# ----------------------------------------------------------------------------- main loop
def train(cfg: dict, resume: str = "auto") -> dict:
    paths = get_paths(cfg)
    seed, tcfg = cfg["seed"], cfg["training"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log.warning("CUDA is not available: running on CPU in FP32 (very slow).")
    stages, total = stage_list(cfg), total_updates(cfg)
    eval_steps = set(dev_eval_steps(cfg))
    ck_every, log_every = tcfg["checkpoint_every"], tcfg["log_every"]
    accum, mbs = tcfg["grad_accum"], tcfg["microbatch"]
    dev_path, train_log = paths.dev_dir / "dev_metrics.jsonl", paths.logs_dir / "train_log.jsonl"

    manifest = load_manifest(paths.splits)
    check_manifest(manifest, cfg)
    store = FeatureStore(paths.features / "siglip_features_fp16.pt")
    log.info("Loaded %d cached feature maps", len(store.image_ids))

    vlm = SigLipQwenVLM.from_pretrained(cfg, device)
    scaler = make_grad_scaler(device)
    sampler = CurriculumSampler(manifest, seed, [s["mix"] for s in stages])
    state = {"cum_draws": {s["key"]: {} for s in stages}, "stage_stats": {}, "dev_eval_time_s": 0.0}
    calib = None
    start, ck = 0, None
    if resume != "none":
        path = latest_checkpoint(paths.checkpoints) if resume == "auto" else Path(resume)
        if path is not None and Path(path).exists():
            ck = torch.load(path, map_location="cpu", weights_only=False)
            start = int(ck["global_update"])
            log.info("Resuming from %s (update %d)", path, start)
    if start >= total:
        log.info("Training already complete (%d updates).", total)
        return state

    if ck is not None:
        vlm.connector.load_state_dict(ck["connector"])
        calib, state = ck["calibration"], ck["state"]
        sampler.load_state_dict(ck["sampler"])
        scaler.load_state_dict(ck["scaler"])
    else:
        ensure_dir(paths.results)
        with open(paths.results / "config_used.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        write_json(env_info(), paths.results / "environment.json")
        calib = calibrate(vlm, store, manifest, cfg)
        write_json(calib, paths.results / "connector_calibration.json")
        log.info("Calibration: scale=%.5f (text norm %.4f, visual norm %.4f -> %.4f)", calib["scale_factor"],
                 calib["target_text_mean_l2_norm"], calib["visual_mean_l2_norm_before"], calib["visual_mean_l2_norm_after"])
        sanity_checks(vlm, store, manifest, cfg, paths.results)
        reset_peak_gpu_memory()
        rec = run_dev_eval(vlm, store, manifest, cfg, 0, 0, dev_path)
        log.info("dev@0: %s", {k: round(v, 4) for k, v in rec.items() if isinstance(v, float)})

    for stage in stages:
        if stage["end"] <= start:
            continue
        sidx, key = stage["index"], stage["key"]
        if sidx >= 1 and not vlm.lora_enabled:
            vlm.enable_lora(seed + sidx)
            if ck is not None and ck.get("lora") is not None:
                vlm.load_lora_state_dict(ck["lora"])
        opt, sched = build_optimizer(vlm, stage, tcfg)
        if ck is not None and start > stage["start"]:       # resuming mid-stage: restore optimizer/scheduler
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
        if ck is not None:                                  # RNG last, after every RNG-consuming setup step
            set_rng_state(ck["rng"])
            ck = None
        params = vlm.trainable_parameters()
        ss = state["stage_stats"].setdefault(key, {"train_time_s": 0.0, "updates": 0, "examples_presented": 0,
                                                    "peak_gpu_mb": 0.0})
        ss["trainable_params"] = vlm.param_summary()
        ss["name"] = stage["name"]
        log.info("== Stage %d (%s): updates %d-%d | trainable=%s", sidx + 1, stage["name"], stage["start"],
                 stage["end"], ss["trainable_params"])
        reset_peak_gpu_memory()
        vlm.train_mode()

        for upd in range(max(start, stage["start"]), stage["end"]):
            g = upd + 1
            t_upd = time.perf_counter()
            examples, counts = sampler.next_update(sidx)
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for k in range(accum):
                mb = examples[k * mbs:(k + 1) * mbs]
                feats = store.get([e["image_id"] for e in mb])
                with autocast_ctx(device):
                    per_ex = vlm.per_example_loss(mb, feats)          # example-weighted loss
                loss = per_ex.mean() / accum                           # accumulated grad == effective batch of 16
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
            scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            sched.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_upd
            ss["train_time_s"] += dt
            ss["updates"] += 1
            ss["examples_presented"] += len(examples)
            cd = state["cum_draws"][key]
            for c, n in counts.items():
                cd[c] = cd.get(c, 0) + n

            if g % log_every == 0 or g == stage["end"]:
                row = {"update": g, "stage": sidx + 1, "loss": loss_sum, "grad_norm": gnorm,
                       "lr_connector": opt.param_groups[0]["lr"],
                       "lr_lora": opt.param_groups[1]["lr"] if len(opt.param_groups) > 1 else None,
                       "loss_scale": float(scaler.get_scale()) if device.type == "cuda" else 1.0,
                       "draws": counts, "cum_draws": dict(cd), "update_time_s": dt}
                append_jsonl(row, train_log)
                if g % (log_every * 10) == 0 or g == stage["end"]:
                    log.info("upd %d/%d loss %.4f gn %.3f lr %.2e draws %s", g, total, loss_sum, gnorm,
                             row["lr_connector"], counts)

            if g in eval_steps or g == stage["end"]:
                t_e = time.perf_counter()
                rec = run_dev_eval(vlm, store, manifest, cfg, g, sidx + 1, dev_path)
                state["dev_eval_time_s"] += time.perf_counter() - t_e
                vlm.train_mode()
                log.info("dev@%d: %s", g, {k: round(v, 4) for k, v in rec.items() if isinstance(v, float)})

            ss["peak_gpu_mb"] = max(ss["peak_gpu_mb"], peak_gpu_memory_mb())
            is_stage_end = g == stage["end"]
            if is_stage_end or g % ck_every == 0 or g == total:
                payload = {"global_update": g, "stage_index": sidx, "connector": vlm.connector.state_dict(),
                           "lora": vlm.lora_state_dict() if vlm.lora_enabled else None, "calibration": calib,
                           "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                           "scaler": scaler.state_dict(), "sampler": sampler.state_dict(),
                           "rng": get_rng_state(), "state": state}
                if is_stage_end:
                    name = "stage3_final.pt" if sidx == 2 else f"stage{sidx + 1}_end.pt"
                    save_atomic({"global_update": g, "connector": payload["connector"], "lora": payload["lora"],
                                 "calibration": calib}, paths.checkpoints / name)
                save_atomic(payload, paths.checkpoints / f"ckpt_{g:05d}.pt")
                prune_checkpoints(paths.checkpoints, tcfg["keep_last_checkpoints"])
        start = stage["end"]

    totals = {"updates": sum(s["updates"] for s in state["stage_stats"].values()),
              "examples_presented": sum(s["examples_presented"] for s in state["stage_stats"].values()),
              "train_time_s": sum(s["train_time_s"] for s in state["stage_stats"].values()),
              "dev_eval_time_s": state["dev_eval_time_s"],
              "peak_gpu_mb": max(s["peak_gpu_mb"] for s in state["stage_stats"].values())}
    write_json({"stages": state["stage_stats"], "totals": totals, "cum_draws": state["cum_draws"],
                "environment": env_info()}, paths.logs_dir / "resources_train.json")
    log.info("Training finished: %s", totals)
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default="auto", help="auto | none | path to ckpt_*.pt")
    args = ap.parse_args()
    train(load_config(args.config, smoke=args.smoke), resume=args.resume)


if __name__ == "__main__":
    main()
