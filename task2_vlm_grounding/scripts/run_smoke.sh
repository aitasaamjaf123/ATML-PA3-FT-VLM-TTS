#!/usr/bin/env bash
# End-to-end smoke test (tiny data, 45 updates). Run from the repository root.
set -euo pipefail
export PYTHONPATH="$PWD"
python data/scripts/download_task2.py annotations
python -m task2_vlm_grounding.src.build_manifest --smoke
python -m task2_vlm_grounding.src.split_checks --smoke
python data/scripts/download_task2.py images --splits-dir data/splits/task2_smoke
python -m task2_vlm_grounding.src.cache_features --smoke
python -m task2_vlm_grounding.src.train --smoke --resume none
python -m task2_vlm_grounding.src.evaluate_final --smoke
python -m task2_vlm_grounding.src.analyze --smoke
