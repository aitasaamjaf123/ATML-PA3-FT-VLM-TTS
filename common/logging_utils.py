"""Environment/GPU info, file sizes, stdout tee, git commit."""
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import torch


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
    if not torch.cuda.is_available():
        return {"cuda_available": False}
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
