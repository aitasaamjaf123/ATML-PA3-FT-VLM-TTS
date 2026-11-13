"""Development evaluation: caption loss, reference-answer NLL (correct / complementary / no image),
image-swap penalty and correct-image generated accuracy. Also hosts helpers reused by final evaluation."""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch

from common.io import append_jsonl
from common.logging_utils import cuda_sync
from task2_vlm_grounding.src.sampler import caption_to_example
from task2_vlm_grounding.src.vlm import SigLipQwenVLM, autocast_ctx
from task2_vlm_grounding.src.vqa_eval import DEV_CONDITIONS, eval_items, image_for, score_item


def _feats_for(store, items: Sequence[dict], condition: str):
    imgs = [image_for(it, condition) for it in items]
    if imgs[0] is None:
        return None
    return store.get(imgs)


@torch.no_grad()
def score_nll(vlm: SigLipQwenVLM, store, items: Sequence[dict], condition: str, batch_size: int = 8) -> np.ndarray:
    """Teacher-forced reference-answer NLL per item (token-mean within example, <|im_end|> excluded)."""
    vlm.eval_mode()
    out = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        feats = _feats_for(store, chunk, condition)
        with autocast_ctx(vlm.device_):
            nll = vlm.per_example_loss([c["example"] for c in chunk], feats, supervise_end=False)
        out.extend(nll.float().cpu().tolist())
    return np.asarray(out, dtype=np.float64)


@torch.no_grad()
def generate_items(vlm: SigLipQwenVLM, store, items: Sequence[dict], condition: str, batch_size: int = 8,
                   max_new_tokens: int = 16) -> tuple[list[str], list[float]]:
    """Greedy predictions and per-answer generation time in ms (batch time split evenly)."""
    vlm.eval_mode()
    preds, ms = [], []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        feats = _feats_for(store, chunk, condition)
        cuda_sync()
        t0 = time.perf_counter()
        p = vlm.generate([c["example"] for c in chunk], feats, max_new_tokens)
        cuda_sync()
        dt = (time.perf_counter() - t0) * 1000.0
        preds.extend(p)
        ms.extend([dt / len(chunk)] * len(chunk))
    return preds, ms


@torch.no_grad()
def caption_dev_loss(vlm: SigLipQwenVLM, store, manifest: dict, batch_size: int = 8) -> float:
    """Mean per-example caption cross-entropy (including <|im_end|>) on the dev caption images."""
    vlm.eval_mode()
    exs = [caption_to_example(e) for e in manifest["caption_dev"]]
    vals = []
    for i in range(0, len(exs), batch_size):
        chunk = exs[i:i + batch_size]
        feats = store.get([c["image_id"] for c in chunk])
        with autocast_ctx(vlm.device_):
            vals.extend(vlm.per_example_loss(chunk, feats, supervise_end=True).float().cpu().tolist())
    return float(np.mean(vals))


def run_dev_eval(vlm: SigLipQwenVLM, store, manifest: dict, cfg: dict, step: int, stage: int,
                 out_path=None) -> dict:
    """Evaluate on the development sets (never the final-test sets) and append a JSON line."""
    t0 = time.perf_counter()
    bs = cfg["eval"]["batch_size"]
    rec: dict = {"step": int(step), "stage": int(stage), "caption_dev_loss": caption_dev_loss(vlm, store, manifest, bs)}
    for key, name in (("grounding_dev", "grounding"), ("vqa_dev", "vqa")):
        items = eval_items(manifest[key], name)
        means = {}
        for cond in DEV_CONDITIONS:
            nll = score_nll(vlm, store, items, cond, bs)
            means[cond] = float(nll.mean())
            rec[f"{name}_nll_{cond}"] = means[cond]
        rec[f"{name}_delta_nll"] = means["complementary"] - means["correct"]  # >0: correct image helps
        if cfg["eval"].get("dev_generation", True):
            for cond in ("correct", "complementary"):
                preds, _ = generate_items(vlm, store, items, cond, bs, cfg["eval"]["max_new_tokens"])
                scored = [score_item(it, p) for it, p in zip(items, preds)]
                rec[f"{name}_acc_{cond}"] = float(np.mean([s["score"] for s in scored]))
                if cond == "correct":
                    pn = [s["pred_norm"] for s in scored]
                    rec[f"{name}_identical_rate_correct"] = float(np.mean([pn[2 * k] == pn[2 * k + 1] for k in range(len(pn) // 2)]))
                    rec[f"{name}_empty_rate_correct"] = float(np.mean([s["empty"] for s in scored]))
    rec["eval_time_s"] = time.perf_counter() - t0
    if out_path is not None:
        append_jsonl(rec, out_path)
    return rec
