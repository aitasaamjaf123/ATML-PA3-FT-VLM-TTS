#!/usr/bin/env bash
# Full specification run (4,500 updates). Run from the repository root.
set -euo pipefail
export PYTHONPATH="$PWD"
python data/scripts/download_task2.py annotations
python -m task2_vlm_grounding.src.build_manifest
python -m task2_vlm_grounding.src.split_checks
python data/scripts/download_task2.py images --splits-dir data/splits/task2
python -m task2_vlm_grounding.src.cache_features
python -m task2_vlm_grounding.src.train --resume auto     # resumable: re-run the same command after an interruption
python -m task2_vlm_grounding.src.evaluate_final
python -m task2_vlm_grounding.src.analyze
