"""Cache the 196 final SigLIP patch features (FP16) for every manifest image, then free the encoder.

Usage: python -m task2_vlm_grounding.src.cache_features [--smoke] [--force]

SigLIP has no [CLS] token, so all 196 outputs of `last_hidden_state` are kept in their 14x14 row-major
order. Hugging Face's SiglipVisionTransformer already applies `post_layernorm` inside its forward pass, so
it is NOT applied again here.
"""
from __future__ import annotations

import argparse
import gc
import time

import torch
from PIL import Image

from common.io import write_json
from common.logging_utils import Timer, get_logger, peak_gpu_memory_mb, peak_gpu_reserved_mb, reset_peak_gpu_memory
from common.seed import set_seed
from task2_vlm_grounding.src.config import get_paths, load_config
from task2_vlm_grounding.src.data_coco import coco_image_path
from task2_vlm_grounding.src.manifest import images_by_coco_split, load_manifest

log = get_logger("cache_features")
N_TOKENS, DIM = 196, 768


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, entries, raw_dir, processor):
        self.entries, self.raw_dir, self.processor = entries, raw_dir, processor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        split, image_id = self.entries[i]
        with Image.open(coco_image_path(self.raw_dir, split, image_id)) as im:
            im = im.convert("RGB")
            # deterministic: resize to 224x224 + rescale + normalise (no crops / flips / augmentation)
            return self.processor(images=im, return_tensors="pt")["pixel_values"][0]


class FeatureStore:
    """All cached features in one FP16 CPU tensor; lookup by COCO image id."""

    def __init__(self, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.feats: torch.Tensor = blob["feats"]
        self.image_ids: list[int] = [int(i) for i in blob["image_ids"]]
        self.index = {i: k for k, i in enumerate(self.image_ids)}
        assert self.feats.dtype == torch.float16 and tuple(self.feats.shape[1:]) == (N_TOKENS, DIM), self.feats.shape
        assert len(self.index) == len(self.image_ids), "duplicate image ids"

    def get(self, image_ids) -> torch.Tensor:
        idx = torch.tensor([self.index[int(i)] for i in image_ids], dtype=torch.long)
        return self.feats.index_select(0, idx)

    def __contains__(self, image_id) -> bool:
        return int(image_id) in self.index


def cache_features(cfg: dict, force: bool = False) -> dict:
    from transformers import AutoImageProcessor, SiglipVisionModel

    paths = get_paths(cfg)
    out_path, meta_path = paths.features / "siglip_features_fp16.pt", paths.features / "cache_resources.json"
    manifest = load_manifest(paths.splits)
    by_split = images_by_coco_split(manifest)
    entries = [(s, i) for s in ("train2014", "val2014") for i in by_split[s]]
    all_ids = [i for _, i in entries]
    assert len(set(all_ids)) == len(all_ids), "COCO image ids must be unique across train2014/val2014"
    if out_path.exists() and meta_path.exists() and not force:
        store = FeatureStore(out_path)
        if sorted(store.image_ids) == sorted(all_ids):
            log.info("Features already cached for %d images (%s); use --force to recompute", len(all_ids), out_path)
            return {"skipped": True}

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    name = cfg["models"]["siglip"]
    processor = AutoImageProcessor.from_pretrained(name)
    try:
        model = SiglipVisionModel.from_pretrained(name, dtype=dtype)
    except TypeError:
        model = SiglipVisionModel.from_pretrained(name, torch_dtype=dtype)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    loader = torch.utils.data.DataLoader(
        ImageDataset(entries, paths.raw, processor), batch_size=cfg["models"]["cache_batch_size"],
        shuffle=False, num_workers=cfg["models"]["cache_num_workers"], pin_memory=device.type == "cuda")
    feats = torch.empty(len(entries), N_TOKENS, DIM, dtype=torch.float16)
    reset_peak_gpu_memory()
    timer = Timer().start()
    pos = 0
    with torch.no_grad():
        for px in loader:
            h = model(pixel_values=px.to(device, dtype=dtype)).last_hidden_state
            assert tuple(h.shape[1:]) == (N_TOKENS, DIM), f"unexpected SigLIP output {tuple(h.shape)}"
            feats[pos:pos + h.shape[0]] = h.to("cpu", torch.float16)
            pos += h.shape[0]
            if (pos // px.shape[0]) % 20 == 0:
                log.info("cached %d/%d", pos, len(entries))
    elapsed = timer.stop()
    assert pos == len(entries)
    assert torch.isfinite(feats).all(), "non-finite SigLIP features"
    peak, reserved = peak_gpu_memory_mb(), peak_gpu_reserved_mb()

    del model, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()  # the vision encoder must be gone before Qwen is loaded

    torch.save({"image_ids": all_ids, "feats": feats}, out_path)
    meta = {"n_images": len(entries), "wall_clock_s": elapsed, "images_per_s": len(entries) / max(elapsed, 1e-9),
            "peak_gpu_allocated_mb": peak, "peak_gpu_reserved_mb": reserved, "dtype": "float16",
            "shape_per_image": [N_TOKENS, DIM], "siglip": name, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "file_mb": out_path.stat().st_size / 2**20, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(meta, meta_path)
    log.info("Cached %d images in %.1fs (peak %.0f MB): %s", len(entries), elapsed, peak, out_path)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cache_features(load_config(args.config, smoke=args.smoke), force=args.force)


if __name__ == "__main__":
    main()
