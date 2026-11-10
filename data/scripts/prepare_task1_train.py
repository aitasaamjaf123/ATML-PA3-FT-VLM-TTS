"""Build the Task 1 training set LOCALLY (run once, commit the output).

Selects the FIRST 40,000 *eligible* MetaMathQA examples in source order (so 40,000 remain AFTER filtering):
  excluded if: empty query/response | formatted length > 2048 tokens (Qwen3 chat template, thinking off)
               | query or original_question matches a GSM8K *test* question after normalization.
No truncation, no shuffling.

Outputs (data/splits/task1/):
  train_40k.jsonl.gz          src_index, type, query, original_question, response, n_tokens
  train_40k_manifest.json     counts, exclusion stats, file sha256, token-stream sha256 (consistency check)
  train_40k_ids.json          source row indices of the selected examples

Usage (repo root):  python data/scripts/prepare_task1_train.py
Needs: pip install datasets transformers
"""
import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.io import save_json, sha256_file  # noqa: E402
from task1_llm_finetuning.src.data import (build_prompt, encode_batch, formatted_length, normalize_question,  # noqa: E402
                                           tokens_sha256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "data/splits/task1"))
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()

    import transformers
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    assert len(gsm_test) == 1319
    test_norm = {normalize_question(q) for q in gsm_test["question"]}
    print(f"GSM8K test questions: {len(test_norm)} unique normalized")

    meta = load_dataset("meta-math/MetaMathQA", split="train")
    print("MetaMathQA rows:", len(meta))

    # template consistency sanity check (manual prompt+response+<|im_end|>\n vs full chat template)
    chk = meta[0]
    full = tok.apply_chat_template([{"role": "user", "content": chk["query"]},
                                    {"role": "assistant", "content": chk["response"]}],
                                   tokenize=False, enable_thinking=False)
    manual = build_prompt(tok, chk["query"], False) + chk["response"] + "<|im_end|>\n"
    print("template consistent with manual build:", full == manual)
    if full != manual:
        print("WARNING: template mismatch on first example\n", repr(full[:300]), "\n", repr(manual[:300]))

    kept, excl = [], Counter()
    scanned = 0
    for start in range(0, len(meta), args.chunk):
        rows = meta[start:start + args.chunk]
        n_rows = len(rows["query"])
        # cheap filters first
        status = []
        for i in range(n_rows):
            q, r, oq = rows["query"][i], rows["response"][i], rows["original_question"][i]
            if not (q or "").strip() or not (r or "").strip():
                status.append("empty")
            elif normalize_question(q) in test_norm or normalize_question(oq or "") in test_norm:
                status.append("gsm8k_test_match")
            else:
                status.append(None)
        surv = [i for i, s in enumerate(status) if s is None]
        enc = encode_batch(tok, [rows["query"][i] for i in surv], [rows["response"][i] for i in surv])
        enc_by_i = dict(zip(surv, enc))
        done = False
        for i in range(n_rows):
            scanned += 1
            if status[i] is not None:
                excl[status[i]] += 1
                continue
            ids, plen = enc_by_i[i]
            if formatted_length(ids) > args.max_len:
                excl["too_long"] += 1
                continue
            kept.append({"src_index": start + i, "type": rows["type"][i], "query": rows["query"][i],
                         "original_question": rows["original_question"][i], "response": rows["response"][i],
                         "n_tokens": formatted_length(ids), "_ids": ids, "_plen": plen})
            if len(kept) == args.n:
                done = True
                break
        print(f"scanned {scanned:,} | kept {len(kept):,} | excluded {dict(excl)}")
        if done:
            break
    assert len(kept) == args.n, f"only {len(kept)} eligible examples found"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jl = out / "train_40k.jsonl.gz"
    with gzip.open(jl, "wt", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps({x: k[x] for x in ("src_index", "type", "query", "original_question", "response",
                                                  "n_tokens")}, ensure_ascii=False) + "\n")
    tok_hash = tokens_sha256({"ids": k["_ids"], "prompt_len": k["_plen"]} for k in kept)
    lens = sorted(k["n_tokens"] for k in kept)
    manifest = {
        "num_examples": len(kept), "scanned_source_rows": scanned, "last_source_index": kept[-1]["src_index"],
        "excluded_counts_within_scanned": dict(excl), "max_len": args.max_len, "model_tokenizer": args.model,
        "transformers_version": transformers.__version__, "train_file": jl.name,
        "train_file_sha256": sha256_file(jl), "tokens_sha256": tok_hash,
        "token_len_min": lens[0], "token_len_median": lens[len(lens) // 2], "token_len_max": lens[-1],
        "token_len_mean": sum(lens) / len(lens), "type_counts": dict(Counter(k["type"] for k in kept)),
        "normalization": "NFKC, lowercase, non-alphanumeric->space, whitespace collapsed; query OR original_question "
                         "matched against all 1,319 GSM8K test questions",
        "length_rule": "len(prompt_ids)+len(response_ids)+<|im_end|> + 1 trailing newline <= max_len",
    }
    save_json(manifest, out / "train_40k_manifest.json")
    save_json({"src_indices": [k["src_index"] for k in kept]}, out / "train_40k_ids.json")
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {jl} ({jl.stat().st_size / 2**20:.1f} MB). Commit data/splits/task1/*.")


if __name__ == "__main__":
    main()
