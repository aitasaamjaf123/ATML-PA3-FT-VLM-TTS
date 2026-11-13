"""CLI entry point:

    python task3_test_time_compute/scripts/run_task3.py --smoke                 # fast end-to-end check
    python task3_test_time_compute/scripts/run_task3.py                         # full specification
    python task3_test_time_compute/scripts/run_task3.py --stages offline adaptive report --data-root /path/to/ATML-PA3-Data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from task3_test_time_compute.src import analyze  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--stages", nargs="+", default=analyze.ALL_STAGES, choices=analyze.ALL_STAGES)
    ap.add_argument("--datasets", nargs="+", default=None, choices=["math", "mathvista"])
    a = ap.parse_args()
    cfg = analyze.load_config(a.config, smoke=a.smoke)
    if a.data_root:
        cfg["paths"]["data_root"] = a.data_root
    analyze.run_pipeline(cfg, stages=a.stages, datasets=a.datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
