"""Logging, wall-clock timing, GPU-memory helpers, environment info, stdout tee, git commit.

Merged from Tasks 1-3. torch is imported lazily so everything works without it (except env_info).
  Task 1: get_git_commit, pip_freeze, gpu_info, env_info, path_size_bytes, Tee, tee_stdout
  Task 2: get_logger, cuda_sync, reset_peak_gpu_memory, peak_gpu_memory_mb, peak_gpu_reserved_mb,
          current_gpu_memory_mb, Timer (.start/.stop/.elapsed), timed(label, logger=None)
  Task 3: package_versions, Timer.seconds, and timed() also accepts the argument order timed(logger, label)
"""
from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def get_logger(name: str = "atml", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------------------------- GPU / timing
def _cuda():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


def cuda_sync() -> None:
    t = _cuda()
    if t is not None:
        t.cuda.synchronize()


def reset_peak_gpu_memory() -> None:
    t = _cuda()
    if t is not None:
        t.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb() -> float:
    """Peak *allocated* GPU memory (MB) since the last reset (0.0 on CPU)."""
    t = _cuda()
    return float(t.cuda.max_memory_allocated() / 2**20) if t is not None else 0.0


def peak_gpu_reserved_mb() -> float:
    t = _cuda()
    return float(t.cuda.max_memory_reserved() / 2**20) if t is not None else 0.0


def current_gpu_memory_mb() -> float:
    t = _cuda()
    return float(t.cuda.memory_allocated() / 2**20) if t is not None else 0.0


class Timer:
    """Accumulating wall-clock timer (synchronises CUDA so GPU work is counted)."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._t0: float | None = None

    @property
    def seconds(self) -> float:  # Task 3 name
        return self.elapsed

    def start(self) -> "Timer":
        cuda_sync()
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> float:
        cuda_sync()
        if self._t0 is not None:
            self.elapsed += time.perf_counter() - self._t0
            self._t0 = None
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


@contextmanager
def timed(label, logger=None):
    """Time a block. Usage: `with timed("label", log):` (Task 2) or `with timed(log, "label"):` (Task 3)."""
    announce = False
    if isinstance(label, logging.Logger):  # Task 3 argument order
        label, logger = logger, label
        announce = True
    t = Timer().start()
    if announce and logger is not None:
        logger.info("%s ...", label)
    try:
        yield t
    finally:
        t.stop()
        if logger is not None:
            logger.info("%s took %.2fs", label, t.elapsed)


# ---------------------------------------------------------------------------------------------- environment
def get_git_commit(repo_dir=".") -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def pip_freeze() -> str:
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode()
    except Exception:
        return ""


def gpu_info() -> dict:
    t = _cuda()
    if t is None:
        return {"cuda_available": False}
    torch = t
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device_count_visible": torch.cuda.device_count(),
        "device_name": props.name,
        "total_memory_gb": round(props.total_memory / 2**30, 3),
        "compute_capability": f"{props.major}.{props.minor}",
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def env_info() -> dict:
    import torch
    import transformers

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "kaggle_url_base": os.environ.get("KAGGLE_URL_BASE"),
        "cpu_count": os.cpu_count(),
    }
    for pkg in ["safetensors", "numpy", "yaml", "datasets", "lm_eval"]:
        try:
            m = __import__(pkg)
            info[pkg] = getattr(m, "__version__", "unknown")
        except Exception:
            info[pkg] = None
    info.update({"gpu": gpu_info()})
    return info


def package_versions(names=("numpy", "pandas", "scipy", "torch", "transformers", "math-verify", "pyarrow", "matplotlib")) -> dict:
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except Exception:
            out[n] = None
    return out


def path_size_bytes(path) -> int:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class Tee:
    """Mirror stdout/stderr to a log file."""

    def __init__(self, path, stream):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", buffering=1)
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def isatty(self):
        return False


def tee_stdout(path):
    sys.stdout = Tee(path, sys.__stdout__)
    sys.stderr = Tee(path, sys.__stderr__)
