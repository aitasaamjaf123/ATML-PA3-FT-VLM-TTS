#!/usr/bin/env bash
# Usage (from repo root): bash task1_llm_finetuning/scripts/run_train.sh <full_ft|l2sp|lora4|lora16|oplora16> [output_root] [extra args]
set -euo pipefail
VARIANT="$1"; OUT="${2:-task1_llm_finetuning/outputs/runs}"; shift || true; shift || true
export CUDA_VISIBLE_DEVICES=0
python -m task1_llm_finetuning.src.train --config "task1_llm_finetuning/configs/${VARIANT}.yaml" --output_root "$OUT" "$@"
