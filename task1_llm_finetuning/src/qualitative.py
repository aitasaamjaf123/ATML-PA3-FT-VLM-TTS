"""Step 7: qualitative generations (greedy, thinking disabled, 512 new tokens) for the 24 prompts of
first_plot_questions.yaml under each evaluation condition; then collect everything.

Per condition (resumable):  results_root/qualitative/responses_<cond>.jsonl
Collect (--collect):        responses_all.csv            (168 rows when all 7 conditions are done)
                            annotation_subset_42.csv     (first & last plain question x 3 formats x 7 conditions,
                                                          with blank manual-annotation columns)
Automatic flags are only aids for the manual annotation: truncated, n_new_tokens, repetition_4gram,
math_framing_hits, has_answer_is_marker, json_valid.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch

from .eval_utils import (ALL_CONDITIONS, DEVICE, Ctx, build_eval_model, chat_prompt, free_gpu,
                         load_qualitative_prompts, load_tokenizer, patch_chat_template, resolve_conditions,
                         select_conditions)

MAX_NEW = 512
MATH_RE = re.compile(r"(\bsolve\b|\bequation\b|\bcalculate\b|\bstep \d|\btherefore\b|\\boxed|\bhow many\b|"
                     r"\d+\s*[+\-*/×=]\s*\d+)", re.I)


def repetition_4gram(text: str) -> float:
    w = text.split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)


def json_valid(text: str) -> bool:
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for cand in (t, t[t.find("{"):t.rfind("}") + 1] if "{" in t else ""):
        try:
            json.loads(cand)
            return True
        except Exception:
            pass
    return False


@torch.no_grad()
def generate(model, tok, prompts, bs=8):
    eos = set(int(x) for x in ([model.generation_config.eos_token_id] if isinstance(
        model.generation_config.eos_token_id, int) else (model.generation_config.eos_token_id or [])))
    eos.add(int(tok.eos_token_id))
    texts = [chat_prompt(tok, p) for p in prompts]
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None,
                           pad_token_id=tok.pad_token_id)
        for row in g[:, enc.input_ids.shape[1]:].tolist():
            cut = next((j for j, t in enumerate(row) if t in eos), None)
            ids = row if cut is None else row[:cut]
            outs.append(dict(response=tok.decode(ids, skip_special_tokens=True),
                             n_new_tokens=len(ids), truncated=cut is None and len(row) >= MAX_NEW))
    return outs


def collect(qdir: Path, prompts):
    dfs = [pd.read_json(p, lines=True) for p in sorted(qdir.glob("responses_*.jsonl")) if p.stem != "responses_all"]
    if not dfs:
        raise SystemExit("nothing to collect")
    df = pd.concat(dfs, ignore_index=True)
    df["cond_order"] = df["condition"].map({c: i for i, c in enumerate(ALL_CONDITIONS)})
    df = df.sort_values(["question_idx", "format", "cond_order"]).drop(columns="cond_order")
    df.to_csv(qdir / "responses_all.csv", index=False)
    last = max(p["question_idx"] for p in prompts)
    sub = df[df.question_idx.isin([0, last])].copy()
    for c in ("format_compliance", "coherence_relevance", "repetition", "unnecessary_math_framing",
              "inappropriate_content", "notes", "excerpt"):
        sub[c] = ""
    sub.to_csv(qdir / "annotation_subset_42.csv", index=False)
    print(f"responses_all.csv: {len(df)} rows ({df.condition.nunique()} conditions); "
          f"annotation_subset_42.csv: {len(sub)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root")
    ap.add_argument("--conditions_file")
    ap.add_argument("--conditions", nargs="*", default=["all"])
    ap.add_argument("--results_root", default="task1_llm_finetuning/results")
    ap.add_argument("--prompts", default="task1_llm_finetuning/prompts/first_plot_questions.yaml")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--collect", action="store_true", help="only (re)build responses_all.csv / annotation subset")
    ap.add_argument("--dry_run", action="store_true", help="print parsed prompts and exit")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    qdir = Path(a.results_root) / "qualitative"
    qdir.mkdir(parents=True, exist_ok=True)
    prompts = load_qualitative_prompts(a.prompts)
    if a.dry_run:
        for p in prompts:
            print(p["prompt_id"], "|", p["prompt"][:110].replace("\n", "\\n"))
        return
    if a.collect:
        return collect(qdir, prompts)

    paths = resolve_conditions(a.run_root, a.conditions_file)
    conds = select_conditions(a.conditions, paths)
    ctx = Ctx(bases_cache=str(Path(a.results_root) / "diagnostics" / "svd_bases_k128.pt"))
    tok = patch_chat_template(load_tokenizer())
    for c in conds:
        f = qdir / f"responses_{c}.jsonl"
        if f.exists() and not a.force:
            print(f"[skip] {c}")
            continue
        model = build_eval_model(c, paths.get(c), ctx)
        outs = generate(model, tok, [p["prompt"] for p in prompts], a.bs)
        with open(f, "w", encoding="utf-8") as fh:
            for p, o in zip(prompts, outs):
                r = dict(condition=c, **p, **o)
                r.update(repetition_4gram=repetition_4gram(o["response"]),
                         math_framing_hits=len(MATH_RE.findall(o["response"])),
                         has_answer_is_marker=bool(re.search(r"the answer is\s*:", o["response"], re.I)),
                         json_valid=json_valid(o["response"]) if p["format"] == "json" else None)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[done] {c}: {len(outs)} responses, {sum(o['truncated'] for o in outs)} truncated")
        del model
        free_gpu()
    collect(qdir, prompts)


if __name__ == "__main__":
    main()
