"""Shared helpers for Task 1 steps 5-7 (merge, evaluation, diagnostics, qualitative).

Everything that depends on *how the training runs saved their files* lives in
`resolve_conditions` / `find_weights` / `load_tensors` / `load_adapter`. If your run
layout differs from what is guessed here, pass --conditions_file (a YAML mapping
condition -> path) instead of editing code; see inspect_runs.py.
"""
from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path

import torch
import yaml

MODEL_ID = os.environ.get("TASK1_MODEL_ID", "Qwen/Qwen3-0.6B")  # env override is for tests only
SEED = 6304
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
DIAG_LAYERS = [0, 13, 27]
PROTECT_K = 128

# condition -> how to obtain the model
CONDITIONS = {
    "baseline":       dict(kind="base"),
    "full_ft_mid":    dict(kind="full", run="full_ft", stage="mid"),
    "full_ft_final":  dict(kind="full", run="full_ft", stage="final"),
    "l2sp_final":     dict(kind="full", run="l2sp", stage="final"),
    "lora4_final":    dict(kind="adapter", run="lora4", stage="final", r=4, alpha=8, oplora=False),
    "lora16_final":   dict(kind="adapter", run="lora16", stage="final", r=16, alpha=32, oplora=False),
    "oplora16_final": dict(kind="adapter", run="oplora16", stage="final", r=16, alpha=32, oplora=True),
}
ALL_CONDITIONS = list(CONDITIONS)
DIAG_CONDITIONS = [c for c in ALL_CONDITIONS if c != "baseline"]  # 5 finals + full_ft_mid


def module_path(layer: int, name: str) -> str:
    return f"model.layers.{layer}.{'self_attn' if name in ATTN_MODULES else 'mlp'}.{name}"


def upsert_csv(path, df, key="condition"):
    """Write df to path, keeping rows of OTHER conditions already in the file (so runs can be done in pieces/accounts)."""
    import pandas as pd
    path = Path(path)
    if df.empty:
        return df
    if path.exists():
        old = pd.read_csv(path)
        if key in old:
            df = pd.concat([old[~old[key].isin(df[key].unique())], df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- model / tokenizer
def _dtype_kw(dtype):
    import transformers
    p = transformers.__version__.split(".")
    major, minor = int(p[0]), int(re.match(r"\d+", p[1]).group())
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def load_base(dtype=torch.float16, device=DEVICE):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, **_dtype_kw(dtype))
    return m.to(device).eval()


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    return tok


def patch_chat_template(tok):
    """Make every apply_chat_template call default to enable_thinking=False (incl. calls made inside lm-eval)."""
    orig = tok.apply_chat_template

    def patched(*a, **k):
        k.setdefault("enable_thinking", False)
        return orig(*a, **k)

    tok.apply_chat_template = patched
    return tok


def chat_prompt(tok, user_text: str, system: str = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


# --------------------------------------------------------------------------- tensors on disk
def _flatten(obj, prefix=""):
    out = {}
    if torch.is_tensor(obj):
        out[prefix.rstrip(".")] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    return out


def load_tensors(path) -> dict:
    """Load a flat {key: tensor} dict from .safetensors / .pt / .bin (nested dicts are flattened)."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        out = {}
        with safe_open(str(path), "pt", device="cpu") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out
    return _flatten(torch.load(path, map_location="cpu", weights_only=False))


_KEY_RE = re.compile(r"(model\.(?:layers\.\d+\.|embed_tokens\.|norm\.).*|lm_head\.weight)$")


def normalize_keys(sd: dict) -> dict:
    """Strip wrapper prefixes ('module.', '_orig_mod.', 'base_model.model.' ...) to HF key names."""
    out = {}
    for k, v in sd.items():
        m = _KEY_RE.search(k)
        if m:
            out[m.group(1)] = v
    return out


_ADAPTER_RE = re.compile(
    r"layers\.(\d+)\.(?:self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"\.(?:lora_)?([ABab])(?:\.|$)")


def load_adapter(path, r: int) -> dict:
    """-> {(layer, module): {'A': (r,in) fp32, 'B': (out,r) fp32}}  (raw, i.e. UNprojected for OPLoRA)."""
    out = {}
    for k, v in load_tensors(path).items():
        m = _ADAPTER_RE.search(k)
        if m:
            out.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3).upper()] = v.float()
    if len(out) != 196:
        raise RuntimeError(f"{path}: found adapters for {len(out)} modules, expected 196 (28 layers x 7). "
                           f"Check the key names (inspect_runs.py --show_keys).")
    for key, ab in out.items():
        if set(ab) != {"A", "B"} or ab["A"].shape[0] != r or ab["B"].shape[1] != r:
            raise RuntimeError(f"{path}: bad adapter shapes at {key}: "
                               f"{ {k: tuple(v.shape) for k, v in ab.items()} } for r={r}")
    return out


# --------------------------------------------------------------------------- run / file discovery
_WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".bin")


def find_run_dir(root: Path, run: str):
    dirs = [d for d in Path(root).iterdir() if d.is_dir()]
    norm = lambda n: n.lower().replace("-", "_")
    exact = [d for d in dirs if norm(d.name) == run]
    if exact:
        return exact[0]
    rules = {
        "full_ft":  lambda n: "full" in n and "l2" not in n,
        "l2sp":     lambda n: "l2" in n,
        "lora4":    lambda n: re.search(r"(?<![a-z])lora_?4(?!\d)", n) is not None,
        "lora16":   lambda n: re.search(r"(?<![a-z])lora_?16", n) is not None,
        "oplora16": lambda n: "oplora" in n,
    }
    hits = [d for d in dirs if rules[run](norm(d.name))]
    if len(hits) > 1:
        raise RuntimeError(f"Several run dirs match '{run}': {[h.name for h in hits]}; use --conditions_file")
    return hits[0] if hits else None


def candidate_files(run_dir: Path):
    return sorted(p for p in Path(run_dir).rglob("*") if p.is_file() and p.suffix in _WEIGHT_SUFFIXES)


def _has(p: Path, base: Path, words) -> bool:
    s = str(p.relative_to(base)).lower()
    return any(w in s for w in words)


def find_weights(run_dir: Path, spec: dict):
    stage = spec["stage"]
    want = ("final", "1250") if stage == "final" else ("mid", "625")
    other = ("mid", "625", "312", "937") if stage == "final" else ("final", "1250", "312", "937")
    bad = ("recovery", "optim", "displacement", "basis", "bases", "svd", "proj", "tilde") + other
    must = ("adapter", "lora") if spec["kind"] == "adapter" else ("weight", "model", "fp32", "state", "theta", "full")
    sel = [p for p in candidate_files(run_dir) if _has(p, run_dir, want) and not _has(p, run_dir, bad)]
    sel = [p for p in sel if _has(p, run_dir, must)] or sel
    if spec["kind"] == "full":
        sel = [p for p in sel if not _has(p, run_dir, ("adapter", "lora"))]
    if len(sel) > 1:
        sel = sorted(sel, key=lambda p: -p.stat().st_size)
        print(f"[warn] {len(sel)} candidates in {run_dir.name} for {spec['kind']}/{stage}; using {sel[0].name}. "
              f"Others: {[s.name for s in sel[1:]]}")
    return sel[0] if sel else None


def resolve_conditions(run_root=None, conditions_file=None) -> dict:
    """condition -> Path (or None for baseline / not found)."""
    if conditions_file:
        raw = yaml.safe_load(open(conditions_file))
        return {c: (Path(raw[c]) if raw.get(c) else None) for c in ALL_CONDITIONS}
    out = {"baseline": None}
    for cond, spec in CONDITIONS.items():
        if spec["kind"] == "base":
            continue
        rd = find_run_dir(Path(run_root), spec["run"])
        out[cond] = find_weights(rd, spec) if rd else None
    return out


def select_conditions(requested, paths):
    conds = ALL_CONDITIONS if not requested or requested == ["all"] else requested
    for c in conds:
        if c not in CONDITIONS:
            raise SystemExit(f"Unknown condition {c}; choose from {ALL_CONDITIONS}")
        if CONDITIONS[c]["kind"] != "base" and not paths.get(c):
            raise SystemExit(f"No weights found for {c}. Run inspect_runs.py and pass --conditions_file.")
    return conds


# --------------------------------------------------------------------------- starting weights, OPLoRA bases
class StartingWeights:
    """FP32 copy of the starting checkpoint (CPU)."""

    def __init__(self):
        m = load_base(torch.float32, "cpu")
        self.sd = m.state_dict()  # references stay alive after `del m`
        self.param_names = [n for n, _ in m.named_parameters()]  # unique params (tied counted once)
        del m

    def get(self, layer, name):
        return self.sd[module_path(layer, name) + ".weight"]


class Bases:
    """Leading-k singular bases (U_k, V_k) of the starting weights, computed once in FP32 and cached."""

    def __init__(self, start: StartingWeights, k=PROTECT_K, cache_path=None):
        self.start, self.k, self.path, self.store = start, k, cache_path, {}
        if cache_path and Path(cache_path).exists():
            self.store = torch.load(cache_path, map_location="cpu", weights_only=True)

    def get(self, layer, name):
        key = f"{layer}.{name}"
        if key not in self.store:
            W = self.start.get(layer, name).to(DEVICE).float()
            U, _, Vh = torch.linalg.svd(W, full_matrices=False)
            self.store[key] = (U[:, :self.k].cpu().contiguous(), Vh[:self.k].T.cpu().contiguous())
        Uk, Vk = self.store[key]
        return Uk.to(DEVICE), Vk.to(DEVICE)

    def save(self):
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.store, self.path)


class Ctx:
    def __init__(self, bases_cache=None):
        self._start, self._bases, self.bases_cache = None, None, bases_cache

    @property
    def start(self):
        if self._start is None:
            self._start = StartingWeights()
        return self._start

    @property
    def bases(self):
        if self._bases is None:
            self._bases = Bases(self.start, PROTECT_K, self.bases_cache)
        return self._bases


def project(A, B, Uk, Vk):
    """OPLoRA: B~ = B - U_k(U_k^T B), A~ = A - (A V_k)V_k^T."""
    return A - (A @ Vk) @ Vk.T, B - Uk @ (Uk.T @ B)


# --------------------------------------------------------------------------- adapters: merge / unmerged hooks
def adapter_factors(ab, spec, ctx, layer, name):
    A, B = ab["A"].to(DEVICE).float(), ab["B"].to(DEVICE).float()
    if spec["oplora"]:
        Uk, Vk = ctx.bases.get(layer, name)
        A, B = project(A, B, Uk, Vk)
    return A, B


@torch.no_grad()
def merge_adapter_into(model, adapter, spec, ctx):
    """W <- W0 + (alpha/r) B~ A~ computed in FP32, written back in the model's dtype."""
    mods = dict(model.named_modules())
    scale = spec["alpha"] / spec["r"]
    for (l, name), ab in adapter.items():
        lin = mods[module_path(l, name)]
        A, B = adapter_factors(ab, spec, ctx, l, name)
        lin.weight.copy_((lin.weight.float() + scale * (B @ A)).to(lin.weight.dtype))


def attach_unmerged_hooks(model, adapter, spec, ctx):
    """Add (alpha/r) B~ A~ x on each target Linear's output (no merge, no dropout)."""
    mods = dict(model.named_modules())
    scale = spec["alpha"] / spec["r"]
    handles = []
    for (l, name), ab in adapter.items():
        A, B = adapter_factors(ab, spec, ctx, l, name)

        def hook(mod, inp, out, A=A, B=B):
            return out + (scale * ((inp[0].float() @ A.T) @ B.T)).to(out.dtype)

        handles.append(mods[module_path(l, name)].register_forward_hook(hook))
    return handles


def build_eval_model(cond, path, ctx: Ctx):
    """FP16 eval model on DEVICE for a condition (adapters merged into a fresh backbone copy)."""
    spec = CONDITIONS[cond]
    if spec["kind"] == "base":
        return load_base(torch.float16)
    if spec["kind"] == "full":
        m = load_base(torch.float32, "cpu")
        sd = normalize_keys(load_tensors(path))
        missing, unexpected = m.load_state_dict(sd, strict=False)
        bad = [k for k in missing if k != "lm_head.weight"]
        if unexpected or bad:
            raise RuntimeError(f"{path}: unexpected={unexpected[:3]} missing={bad[:3]}")
        return m.half().to(DEVICE).eval()
    m = load_base(torch.float16)
    merge_adapter_into(m, load_adapter(path, spec["r"]), spec, ctx)
    return m


class DeltaSource:
    """Gives dW = W_trained - W0 (full) or (alpha/r) B~A~ (adapters), FP32 on DEVICE."""

    def __init__(self, cond, path, ctx: Ctx):
        self.spec, self.ctx = CONDITIONS[cond], ctx
        if self.spec["kind"] == "full":
            self.sd = normalize_keys(load_tensors(path))
        else:
            self.adapter = load_adapter(path, self.spec["r"])

    def delta(self, layer, name):
        if self.spec["kind"] == "full":
            key = module_path(layer, name) + ".weight"
            return self.sd[key].to(DEVICE).float() - self.ctx.start.get(layer, name).to(DEVICE).float()
        A, B = adapter_factors(self.adapter[(layer, name)], self.spec, self.ctx, layer, name)
        return (self.spec["alpha"] / self.spec["r"]) * (B @ A)


# --------------------------------------------------------------------------- probe set (benchmark-format prompts)
def _ids(v):
    out = []
    for x in v:
        if isinstance(x, bool):
            continue
        if isinstance(x, int):
            out.append(x)
        elif isinstance(x, str) and re.search(r"(\d+)$", x):
            out.append(int(re.search(r"(\d+)$", x).group(1)))
        elif isinstance(x, dict):
            for k in ("doc_id", "idx", "index", "id", "example_id"):
                if k in x:
                    out += _ids([x[k]])
                    break
    return out


def load_probe_ids(path) -> dict:
    obj = json.load(open(path))
    found = {}

    def dom_of(s):
        s = str(s).lower()
        return "gsm8k" if "gsm" in s else "hellaswag" if "hella" in s else None

    def visit(o):
        if isinstance(o, dict):
            ds = o.get("dataset", o.get("task"))
            if ds and dom_of(ds) and any(k in o for k in ("id", "doc_id", "idx", "index", "example_id")):
                found.setdefault(dom_of(ds), []).extend(_ids([o]))
                return
            for k, v in o.items():
                d = dom_of(k)
                if d and isinstance(v, list) and d not in found:
                    found[d] = _ids(v)
                elif d and isinstance(v, dict) and d not in found:
                    for kk in ("ids", "example_ids", "doc_ids", "indices"):
                        if kk in v:
                            found[d] = _ids(v[kk])
                            break
                    else:
                        visit(v)
                else:
                    visit(v)
        elif isinstance(o, list):
            for x in o:
                visit(x)

    visit(obj)
    if not {"gsm8k", "hellaswag"} <= set(found):
        raise RuntimeError(f"Could not find gsm8k/hellaswag id lists in {path}; top-level: "
                           f"{list(obj)[:10] if isinstance(obj, dict) else type(obj)}")
    return found


def _task_docs(name, split):
    from lm_eval.tasks import TaskManager, get_task_dict
    task = get_task_dict([name], TaskManager())[name]
    return task, list(task.test_docs() if split == "test" else task.validation_docs())


def build_probe_prompts(probe_json, tok) -> dict:
    """{'gsm8k': [(id, prompt_str)], 'hellaswag': [...]} using the stock lm-eval prompt formats.
    ids are indices into the benchmark split (lm-eval doc_id). GSM8K uses the non-thinking chat template,
    HellaSwag the plain completion context."""
    ids = load_probe_ids(probe_json)
    out = {}
    task, docs = _task_docs("gsm8k_cot", "test")
    out["gsm8k"] = [(i, chat_prompt(tok, task.doc_to_text(docs[i]))) for i in ids["gsm8k"]]
    task, docs = _task_docs("hellaswag", "validation")
    out["hellaswag"] = [(i, task.doc_to_text(docs[i])) for i in ids["hellaswag"]]
    return out


# --------------------------------------------------------------------------- qualitative prompts
FORMATS = ("plain", "json", "template")


def _text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("prompt", "text", "content", "question", "input", "user"):
            if isinstance(x.get(k), str):
                return x[k]
    raise ValueError(f"Cannot read a prompt string from {x!r}")


def _load_id_schema(data) -> list:
    """List of {id, paraphrases, [system]}: format from the id suffix (_json / _template / none = plain);
    question_idx = position of the base id in the file's PLAIN-question order; JSON entries carry a system prompt."""
    def split_id(i):
        for suf, f in (("_json", "json"), ("_template", "template")):
            if i.endswith(suf):
                return i[:-len(suf)], f
        return i, "plain"

    parsed = [(*split_id(str(x["id"])), x) for x in data]
    plain_order = [b for b, f, _ in parsed if f == "plain"]
    rows = []
    for base, fmt, x in parsed:
        if base not in plain_order:
            raise ValueError(f"{x['id']}: no plain question '{base}' in the file")
        rows.append(dict(question_idx=plain_order.index(base), question_id=base, format=fmt,
                         prompt_id=str(x["id"]), system=x.get("system"), prompt=_text(x["paraphrases"][0])))
    rows.sort(key=lambda r: (r["question_idx"], FORMATS.index(r["format"])))
    per_fmt = {f: sum(r["format"] == f for r in rows) for f in FORMATS}
    if len(rows) != 24 or set(per_fmt.values()) != {8}:
        print(f"[warn] expected 24 prompts (8 per format); parsed {len(rows)}: {per_fmt}")
    return rows


def load_qualitative_prompts(path) -> list:
    """-> list of dict(question_idx, format, prompt_id, prompt), sorted by (question_idx, format).
    Handles: {plain:[..], json:[..], template:[..]}; list of {plain,json,template}; list of {format,question|prompt}."""
    data = yaml.safe_load(open(path, encoding="utf-8"))
    if isinstance(data, list) and data and all(isinstance(x, dict) and "id" in x and "paraphrases" in x for x in data):
        return _load_id_schema(data)  # the course's first_plot_questions.yaml
    while isinstance(data, dict) and len(data) == 1 and str(next(iter(data))).lower() not in FORMATS \
            and isinstance(next(iter(data.values())), (dict, list)):
        data = next(iter(data.values()))
    fk = lambda d: [k for k in d if str(k).lower() in FORMATS]
    items = []
    if isinstance(data, dict) and fk(data):
        for f in fk(data):
            for i, x in enumerate(data[f]):
                items.append((i, str(f).lower(), _text(x)))
    elif isinstance(data, dict):
        for i, (_, v) in enumerate(data.items()):
            if not (isinstance(v, dict) and fk(v)):
                raise ValueError(f"Unrecognised YAML structure near {v!r}")
            items += [(i, str(f).lower(), _text(v[f])) for f in fk(v)]
    elif isinstance(data, list):
        counter = {}
        for i, x in enumerate(data):
            if isinstance(x, dict) and fk(x):
                items += [(i, str(f).lower(), _text(x[f])) for f in fk(x)]
            elif isinstance(x, dict) and "format" in x:
                f = str(x["format"]).lower()
                q = x.get("question_idx", x.get("question_id", counter.get(f, 0)))
                counter[f] = counter.get(f, 0) + 1
                items.append((int(q) if str(q).isdigit() else counter[f] - 1, f, _text(x)))
            else:
                raise ValueError(f"Unrecognised YAML list element {x!r}")
    else:
        raise ValueError(f"Unrecognised YAML structure: top level is {type(data).__name__}, starts with {str(data)[:200]!r}")
    items.sort(key=lambda t: (t[0], FORMATS.index(t[1]) if t[1] in FORMATS else 9))
    rows = [dict(question_idx=q, format=f, prompt_id=f"q{q:02d}_{f}", prompt=p) for q, f, p in items]
    per_fmt = {f: sum(r["format"] == f for r in rows) for f in FORMATS}
    if len(rows) != 24 or set(per_fmt.values()) != {8}:
        print(f"[warn] expected 24 prompts (8 per format); parsed {len(rows)}: {per_fmt}")
    return rows
