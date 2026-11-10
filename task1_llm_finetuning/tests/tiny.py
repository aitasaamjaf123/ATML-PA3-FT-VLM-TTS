"""Tiny random Qwen3 + synthetic examples for CPU tests (no downloads)."""
import random
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                      tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg).float()
    m.config.use_cache = False
    return m


def synth_examples(n, seed=0):
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        pl, rl = rnd.randint(4, 12), rnd.randint(5, 20)
        ids = [rnd.randint(1, 199) for _ in range(pl + rl)]
        out.append({"ids": ids, "prompt_len": pl, "src_index": i})
    return out


def tiny_cfg(method, **kw):
    cfg = {
        "run_name": f"t_{method}", "method": method, "seed": 6304, "lr": 1e-3, "l2sp_lambda": 0.0,
        "micro_batch_size": 2, "grad_accum": 2, "num_updates": 4, "warmup_updates": 1,
        "optimizer": {"name": "adamw", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "fused": False},
        "grad_clip": 1.0, "precision": "fp16", "loss_chunk_tokens": 16,
        "recovery_at_updates": [2, 4], "stats_every": 2, "save_weights_at": [2, 4], "keep_all_recovery": True,
        "save_final_optimizer": False,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.05,
                 "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        "oplora": {"k": 8, "svd_device": "cpu"},
    }
    cfg.update(kw)
    return cfg
