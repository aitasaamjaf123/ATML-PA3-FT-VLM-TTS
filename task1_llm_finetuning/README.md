# Task 1 — LLM fine-tuning (Full FT, L2-SP, LoRA-4/16, OPLoRA-16)

All runs: Qwen3-0.6B, thinking off, seed 6304, same 40,000 examples in the same order, same recipe
(`configs/shared.yaml`). Per-method files only set method-specific keys.

## 0. One-time local prep (then commit)
```bash
pip install datasets transformers pyyaml
python data/scripts/prepare_task1_train.py          # -> data/splits/task1/train_40k.jsonl.gz + manifest + ids
python data/scripts/prepare_task1_eval.py           # -> data/splits/task1/probe_set.json  (add --warm_cache for offline eval data)
cp <course file> task1_llm_finetuning/prompts/first_plot_questions.yaml
git add data/splits task1_llm_finetuning/prompts && git commit -m "task1 data" && git push
```
`train_40k_manifest.json` stores the file sha256 and a sha256 of the *token stream*. Every training run
re-tokenizes and asserts both -> a different tokenizer/transformers version on another account fails loudly.

## 1. Run on Kaggle (one variant per account; GPU on, internet on; use ONE GPU)
```python
!git clone https://github.com/<you>/<repo>.git /tmp/repo
%cd /tmp/repo
!pip install -q pyyaml safetensors pandas packaging "transformers>=4.51"
import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
!CUDA_VISIBLE_DEVICES=0 python -m task1_llm_finetuning.src.train \
    --config task1_llm_finetuning/configs/full_ft.yaml --output_root /kaggle/working/task1_outputs
```
Variants: `full_ft`, `l2sp`, `lora4`, `lora16`, `oplora16`. Do a quick check first with `--smoke` (3 updates).
Resume: `--resume /kaggle/working/task1_outputs/<run>/recovery/recovery_uXXXX.pt` (attach the previous output as an input).
Use the same GPU type for all runs if you want comparable time/memory (it is logged in `resources.json`).

## 2. Everything saved, per run: `<output_root>/<run_name>/`
| File | Contents |
|---|---|
| `train_log.csv` | per update: lr, sft_loss, grad_norm_preclip, grad_scale, skipped_update, loss_finite, valid/real/padded tokens, update time, cumulative train/wall time, peak GPU mem (alloc/reserved); L2-SP only: omega, lambda_omega, total_loss, displacement_pre_step, penalty_time_sec |
| `progress_stats.csv` | every 125 updates + checkpoints: full-param runs -> ‖θ−θ0‖₂; adapter runs -> mean/max η, total ‖ΔW‖ |
| `checkpoint_sizes.csv` | every artifact written: update, path, bytes, save seconds (kept even if the file is later deleted) |
| `resources.json` | trainable/total params, train & wall time, tokens/s, peak mem, GPU info, checkpoint sizes, method overhead (L2-SP: reference bytes/transfer/penalty seconds; OPLoRA: SVD seconds, basis bytes, export seconds/bytes, constraint residuals) |
| `run_summary.json` | final loss, skipped-update list, non-finite updates, final stats |
| `run_meta.json` | config, git commit, env/library versions, resolved model revision, **init-weights fingerprint**, LoRA-A init fingerprint, data/token hashes |
| `config_resolved.yaml`, `data_check.json`, `pip_freeze.txt`, `train_stdout.log` | reproducibility |
| `displacement_by_param_u{312,625,937,1250}.csv` | (Full FT, L2-SP) per-tensor ‖Δ‖, ‖W0‖, η |
| `weights_u0625/`, `weights_u1250/` | (Full FT) FP32 HF weights at midpoint + final; (L2-SP) final only |
| `recovery/recovery_uXXXX.pt` | model/adapters, optimizer, scheduler, scaler, RNG, data position, timers. Full-param runs keep only the latest (Kaggle 20 GB cap); final one has no optimizer state |
| `adapter_final.safetensors`, `adapter_config.json` | (LoRA/OPLoRA) raw A,B (FP32) + r/alpha/targets/module names |
| `adapter_delta_by_module_u*.csv`, `…_final.csv` | (LoRA/OPLoRA) per-module ‖A‖, ‖B‖, ‖ΔW‖, ‖W0‖, η |
| `oplora_bases.safetensors`, `oplora_bases_meta.csv` | (OPLoRA) U_k, V_k for all 196 modules (FP32), σ₁, σ_k, top-k energy — **reuse for diagnostics on every model** |
| `adapter_projected_final.safetensors`, `oplora_constraint_check.csv` | (OPLoRA) exported B̃, Ã for merging; max‖U_kᵀB̃‖, max‖ÃV_k‖, ρ_out, ρ_in per module |

## 3. Notes / deviations to state in the report
- Loss is computed by running `lm_head`+CE only on supervised positions in checkpointed chunks (memory only; unit-tested equal to full-vocab CE).
- LR: linear warmup (update t uses (t+1)/38 of peak) then cosine to 0. Skipped (inf/NaN) updates still advance schedule and data; they are logged.
- L2-SP θ0 is pinned host memory by default (GPU too full on 16 GB); the per-update host->device streaming cost is inside `penalty_time_sec`.
- Full-param final checkpoint stores weights, not optimizer state (disk cap).
- Adapter runs freeze the backbone in FP16 (bf16->fp16 rounding of W0); Full FT starts from the exact FP32 upcast.
- Expected trainable adapter params (hand-computed): LoRA-4 = 2,523,136; LoRA-16 / OPLoRA-16 = 10,092,544 — check `resources.json`.
