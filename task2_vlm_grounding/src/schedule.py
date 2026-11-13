"""Per-stage learning-rate multiplier: 5% linear warm-up, then cosine decay to 10% of peak."""
from __future__ import annotations

import math


def lr_multiplier(step: int, total: int, warmup_frac: float = 0.05, final_frac: float = 0.10) -> float:
    """Multiplier (relative to the stage's peak LR) for the 0-indexed update `step` of `total`.

    * steps [0, warm):    linear ramp (step + 1) / warm, reaching 1.0 on the last warm-up step
    * steps [warm, total): cosine from 1.0 (step == warm) to `final_frac` (step == total - 1)
    """
    if total <= 0:
        raise ValueError("total must be positive")
    warm = max(1, math.ceil(warmup_frac * total))
    if step < warm:
        return (step + 1) / warm
    denom = max(1, total - 1 - warm)
    progress = min(1.0, (step - warm) / denom)
    return final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))
