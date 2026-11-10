"""Runs train.main() (config merge, data load/encode, method setup, outputs) with a stub tokenizer + tiny model."""
import gzip
import json
import sys

import pandas as pd
import pytest

from task1_llm_finetuning.src import data as D
from task1_llm_finetuning.src import train as T
from task1_llm_finetuning.tests.tiny import tiny_model

END = 199


class StubTok:
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def _enc(self, s):
        out, parts = [], s.split("<|im_end|>")
        for i, p in enumerate(parts):
            out += [(ord(c) % 190) + 5 for c in p]
            if i < len(parts) - 1:
                out.append(END)
        return out

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [self._enc(t) for t in texts]}

    def convert_tokens_to_ids(self, t):
        assert t == "<|im_end|>"
        return END

    def save_pretrained(self, d):
        pass


def test_encode_batch_masks_and_end_token():
    tok = StubTok()
    (ids, plen), = D.encode_batch(tok, ["what is 2+2?"], ["it is 4."])
    assert ids[-1] == END and plen == len(tok._enc(tok.apply_chat_template([{"content": "what is 2+2?"}])))
    b = D.collate([{"ids": ids, "prompt_len": plen}], 0)
    assert (b["labels"][0, plen:] == b["input_ids"][0, plen:]).all() and b["labels"][0, plen - 1] == D.IGNORE
    assert b["labels"][0, -1] == END


@pytest.mark.parametrize("variant", ["full_ft", "l2sp", "lora4", "lora16", "oplora16"])
def test_main_smoke(tmp_path, monkeypatch, variant):
    recs = [{"src_index": i, "type": "x", "query": f"question number {i} ?", "original_question": "",
             "response": f"the answer to {i} is {i * 2}."} for i in range(100)]
    dp = tmp_path / "d.jsonl.gz"
    with gzip.open(dp, "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps({"num_examples": 100}))
    monkeypatch.setattr(T, "load_tokenizer", lambda *a, **k: StubTok())
    monkeypatch.setattr(T, "load_model_fp32_cpu", lambda *a, **k: tiny_model())
    monkeypatch.setattr(sys, "argv", ["train", "--config", f"task1_llm_finetuning/configs/{variant}.yaml",
                                      "--output_root", str(tmp_path / "out"), "--smoke", "--override",
                                      f"train_data={dp}", f"train_manifest={mp}", "oplora.k=8",
                                      "loss_chunk_tokens=16", "optimizer.fused=false"])
    monkeypatch.setattr(T, "tee_stdout", lambda p: None)
    T.main()
    run = tmp_path / "out" / f"{variant}_smoke"
    for f in ["train_log.csv", "run_meta.json", "resources.json", "run_summary.json", "config_resolved.yaml",
              "checkpoint_sizes.csv", "progress_stats.csv", "data_check.json", "DONE", "pip_freeze.txt"]:
        assert (run / f).exists(), f
    df = pd.read_csv(run / "train_log.csv")
    assert len(df) == 3 and df["sft_loss"].notna().all()
    meta = json.load(open(run / "run_meta.json"))
    assert len(meta["init_weights_fingerprint"]) == 64
    if variant in ("lora16", "oplora16"):
        assert meta["lora_A_init_abs_sum"] > 0
    if variant == "oplora16":
        assert (run / "oplora_bases.safetensors").exists() and (run / "oplora_bases_meta.csv").exists()
        res = json.load(open(run / "resources.json"))
        assert res["overhead"]["basis_preprocessing_seconds"] > 0 and res["overhead"]["export_projected_bytes"] > 0
    if variant in ("full_ft", "l2sp"):
        assert (run / "weights_u0003").exists()
