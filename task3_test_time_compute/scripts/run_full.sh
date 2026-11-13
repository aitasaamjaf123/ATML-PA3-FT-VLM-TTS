#!/usr/bin/env bash
# Full specification: 200 dev / 800 held-out questions, 10 orders, 1,000 bootstrap resamples, 50 live questions.
set -euo pipefail
cd "$(dirname "$0")/../.."
python data/scripts/download_task3.py || true
python task3_test_time_compute/scripts/run_task3.py "$@"
