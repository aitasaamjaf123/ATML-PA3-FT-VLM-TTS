"""Task 1 data: MetaMathQA formatting, GSM8K decontamination, tokenization, label masking, collation.

Format (Qwen3 chat template, thinking disabled):
    <|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{response}<|im_end|>
Loss is computed on `{response}<|im_end|>` only. The trailing "\n" the template adds after <|im_end|> is
not part of the training sequence (and not supervised) but IS counted in the "formatted length" filter.
"""
import gzip
import hashlib
import json
import re
import unicodedata
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

IM_END = "<|im_end|>"
IGNORE = -100


def normalize_question(s: str) -> str:
    """Text normalization for GSM8K-test decontamination: NFKC, lowercase, non-alnum -> space, collapse."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def build_prompt(tok, question: str, enable_thinking: bool = False) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def encode_batch(tok, questions: List[str], responses: List[str], enable_thinking: bool = False
                 ) -> List[Tuple[List[int], int]]:
    """Returns [(input_ids, prompt_len)] where input_ids = prompt_ids + response_ids + <|im_end|>."""
    prompts = [build_prompt(tok, q, enable_thinking) for q in questions]
    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    r_ids = tok([r + IM_END for r in responses], add_special_tokens=False)["input_ids"]
    end_id = tok.convert_tokens_to_ids(IM_END)
    out = []
    for p, r in zip(p_ids, r_ids):
        assert r[-1] == end_id, "response must end with <|im_end|> token"
        out.append((list(p) + list(r), len(p)))
    return out


def formatted_length(ids: List[int]) -> int:
    """Template-formatted length = training ids + the trailing newline token after <|im_end|>."""
    return len(ids) + 1


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def load_jsonl(path) -> List[dict]:
    with open_text(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_dataset(tok, records: List[dict], enable_thinking: bool = False, batch: int = 1000) -> List[dict]:
    out = []
    for i in range(0, len(records), batch):
        chunk = records[i:i + batch]
        enc = encode_batch(tok, [r["query"] for r in chunk], [r["response"] for r in chunk], enable_thinking)
        for r, (ids, plen) in zip(chunk, enc):
            out.append({"ids": ids, "prompt_len": plen, "src_index": r.get("src_index")})
    return out


def tokens_sha256(encoded: Iterable[dict]) -> str:
    """Hash of the full token stream (ids + prompt_len per example). Identical on every machine iff
    tokenizer/template/data are identical -> used as a cross-account consistency check."""
    h = hashlib.sha256()
    for ex in encoded:
        h.update(np.asarray(ex["ids"], dtype=np.int32).tobytes())
        h.update(np.asarray([ex["prompt_len"]], dtype=np.int32).tobytes())
    return h.hexdigest()


def collate(examples: List[dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """Right-pad; labels = ids on response+<|im_end|> positions, -100 on prompt and padding."""
    n = max(len(e["ids"]) for e in examples)
    ids = torch.full((len(examples), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(examples), n), dtype=torch.long)
    lab = torch.full((len(examples), n), IGNORE, dtype=torch.long)
    for i, e in enumerate(examples):
        L = len(e["ids"])
        t = torch.tensor(e["ids"], dtype=torch.long)
        ids[i, :L] = t
        att[i, :L] = 1
        lab[i, e["prompt_len"]:L] = t[e["prompt_len"]:]
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


def count_valid_targets(labels: torch.Tensor) -> int:
    """Number of supervised next-token targets (after the causal shift)."""
    return int((labels[:, 1:] != IGNORE).sum())
