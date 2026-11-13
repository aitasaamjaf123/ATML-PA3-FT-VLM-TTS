"""Seeding and RNG-state helpers (seed 6304).

Merged from Tasks 1-3:
  Task 1: DEFAULT_SEED, capture_rng_state / restore_rng_state   (state format: python, numpy, torch_cpu, torch_cuda list)
  Task 2: SEED, set_seed(seed, deterministic=True), get_rng_state / set_rng_state
  Task 3: stable_hash, derive_seed, set_global_seed (seeding without the determinism flags)
The two RNG-state pairs are each self-consistent; use matching get/set or capture/restore calls.
"""
from __future__ import annotations

import os
import random
import zlib

import numpy as np

SEED = 6304
DEFAULT_SEED = SEED


def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (if installed) everywhere."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pure-python usage (data building, tests)
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            # warn_only: a few kernels (e.g. index_add_) have no deterministic variant.
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover
            pass


def get_rng_state() -> dict:
    """Capture python / numpy / torch(cpu+cuda) RNG states (picklable)."""
    state = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    except ImportError:
        pass
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        import torch

        torch.set_rng_state(state["torch_cpu"])
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_rng_state() -> dict:  # Task 1 (requires torch)
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:  # Task 1 (requires torch)
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


# ----------------------------------------------------------------------------------------------
# Task 3
# ----------------------------------------------------------------------------------------------
def stable_hash(text: str) -> int:
    """Process-independent 32-bit hash (Python's hash() is salted per process)."""
    return zlib.crc32(str(text).encode("utf-8")) & 0xFFFFFFFF


def derive_seed(base: int, *parts) -> int:
    """Derive an independent, reproducible 31-bit seed from a base seed and labels."""
    entropy = [int(base)] + [stable_hash(str(p)) for p in parts]
    state = np.random.SeedSequence(entropy).generate_state(1)
    return int(state[0] & 0x7FFFFFFF)


def set_global_seed(seed: int = SEED) -> None:
    """Seed python / numpy / torch without touching determinism flags (used by the Task 3 pipeline)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
