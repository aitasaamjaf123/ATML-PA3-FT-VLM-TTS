# ATML PA3 — Fine-Tuning, VLM Grounding, and Test-Time Compute

EE-5102/CS-6304, Fall 2026. All experiments use seed **6304**.

## Layout

| Path | Contents |
|---|---|
| `common/` | Shared seeding, IO, logging, bootstrap stats, plotting |
| `data/` | `scripts/` download helpers, `splits/` saved IDs (committed), `raw/` + `processed/` (git-ignored) |
| `task1_llm_finetuning/` | Full FT, L2-SP, LoRA-4/16, OPLoRA, evals, diagnostics, qualitative |
| `task2_vlm_grounding/` | SigLIP–Qwen VLM, 3-stage curriculum, grounding evaluation |
| `task3_test_time_compute/` | Offline selection, signal analysis, live generation, adaptive sampling |
| `report/` | NeurIPS-style LaTeX report, figures, tables |

Each task folder has `src/` (code), `configs/` (YAML), `scripts/` (entry points), `tests/`, `results/` (machine-readable outputs, committed), and `figures/`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce
### Task 1
See `task1_llm_finetuning/README.md` (data prep, Kaggle commands, full list of saved outputs).
```bash
python data/scripts/prepare_task1_train.py && python data/scripts/prepare_task1_eval.py
bash task1_llm_finetuning/scripts/run_train.sh full_ft   # | l2sp | lora4 | lora16 | oplora16
python -m pytest task1_llm_finetuning/tests -q
```
### Task 2
```bash
bash task2_vlm_grounding/scripts/prepare_data.sh
bash task2_vlm_grounding/scripts/run_train.sh
bash task2_vlm_grounding/scripts/run_eval.sh
```
### Task 3
```bash
bash task3_test_time_compute/scripts/run_offline.sh
bash task3_test_time_compute/scripts/run_live.sh
bash task3_test_time_compute/scripts/run_adaptive.sh
```

## Attribution
List any materially reused external code here (library, URL, what was reused).

## Deviations / incomplete experiments
Record any deviations from the prescribed setup here.
