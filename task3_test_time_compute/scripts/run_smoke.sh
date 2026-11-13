#!/usr/bin/env bash
# Fast end-to-end check (20 dev + 40 held-out questions, 3 orders, 100 bootstrap resamples, 4 live questions).
set -euo pipefail
cd "$(dirname "$0")/../.."
python task3_test_time_compute/scripts/run_task3.py --smoke "$@"
