"""Model loading, weight fingerprinting, parameter counting."""
import hashlib
from typing import Optional

import numpy as np
import torch
from packaging import version


def _dtype_kw(dtype) -> dict:
    import transformers

    key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56.0") else "torch_dtype"
    return {key: dtype}


def load_tokenizer(name_or_path: str, revision: Optional[str] = None):
    from transformers import AutoTokenizer

    kw = {"revision": revision} if revision else {}
    return AutoTokenizer.from_pretrained(name_or_path, **kw)


def load_model_fp32_cpu(name_or_path: str, revision: Optional[str] = None):
    """Original checkpoint upcast exactly to FP32 on CPU (Qwen3 ships in bf16; bf16->fp32 is exact)."""
    import transformers
    from transformers import AutoModelForCausalLM

    assert version.parse(transformers.__version__) >= version.parse("4.51.0"), "Qwen3 needs transformers>=4.51"
    kw = {"revision": revision} if revision else {}
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, attn_implementation="sdpa", **_dtype_kw(torch.float32), **kw
    )
    model.config.use_cache = False
    return model


def resolved_revision(model) -> Optional[str]:
    return getattr(model.config, "_commit_hash", None)


@torch.no_grad()
def weights_fingerprint(model) -> str:
    """sha256 over per-tensor (sum, abs-sum) in float64, in named_parameters order. Cheap, deterministic,
    and equal across machines iff the starting weights are identical."""
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        t = p.detach().double()
        h.update(name.encode())
        h.update(np.asarray([t.sum().item(), t.abs().sum().item()], dtype=np.float64).tobytes())
    return h.hexdigest()


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable,
            "trainable_fraction": trainable / max(total, 1)}


def enable_checkpointing(model):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
