"""Activation-probe set: 32 GSM8K-test questions + 32 HellaSwag-validation contexts, seed 6304.
Sampled ONCE (before any model results are inspected) with independent default_rng(6304) per dataset,
sorted indices into the HF split order. IDs + text are saved and reused for every model."""
import numpy as np

SEED = 6304


def sample_ids(n_total: int, k: int = 32, seed: int = SEED):
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_total, size=k, replace=False))
