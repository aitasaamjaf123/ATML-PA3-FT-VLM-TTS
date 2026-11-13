"""Answer extraction, normalisation, equivalence grouping and grading.

Design rules (assignment "Answers and correctness"):
  * Answer groups are formed pairwise WITHOUT consulting gold answers.
  * Grading uses the SAME equivalence function (group representative vs. gold).
  * Missing extractions map to one canonical NO_ANSWER group (gid 0), always graded incorrect,
    but they stay in voting, cost and accuracy accounting.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

NO_ANSWER = "NO_ANSWER"


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------
def is_missing(x: Any) -> bool:
    """True for None / NaN / empty (whitespace-only) answers. The literal string 'None' is a real answer."""
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 0:
        return True
    return str(x).strip() == ""


def extract_last_boxed(text: Optional[str]) -> Optional[str]:
    """Return the content of the LAST \\boxed{...} (balanced-brace rule). None if absent or the last
    occurrence is unbalanced (e.g. truncated generation)."""
    if not text:
        return None
    idx = text.rfind("\\boxed{")      # README.MD rule: the last literal \boxed{ (no \fbox, no numeric fallback)
    if idx < 0:
        return None
    start = text.index("{", idx)
    depth = 0
    for j in range(start, len(text)):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j].strip() or None   # an empty box counts as no answer
    return None


# --------------------------------------------------------------------------------------
# MATH equivalence (math-verify, with a documented simple fallback)
# --------------------------------------------------------------------------------------
_FRAC_RE = re.compile(r"^(-?)\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}$")
_SLASH_RE = re.compile(r"^-?\d+(?:\.\d+)?/\d+(?:\.\d+)?$")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


def _fallback_norm(s: str) -> str:
    s = s.strip()
    for a in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ "):
        s = s.replace(a, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    for a in ("^\\circ", "^{\\circ}", "\\%", "%", "\\$", "$"):
        s = s.replace(a, "")
    s = re.sub(r"\s+", "", s)
    return s.rstrip(".")


def _fallback_number(s: str) -> Optional[float]:
    m = _FRAC_RE.match(s)
    if m:
        den = float(m.group(3))
        if den == 0:
            return None
        val = float(m.group(2)) / den
        return -val if m.group(1) else val
    if _SLASH_RE.match(s):
        a, b = s.split("/")
        return float(a) / float(b) if float(b) != 0 else None
    if _THOUSANDS_RE.match(s):
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _fallback_equiv(a: str, b: str) -> bool:
    na, nb = _fallback_norm(a), _fallback_norm(b)
    if na == nb:
        return True
    xa, xb = _fallback_number(na), _fallback_number(nb)
    if xa is not None and xb is not None:
        return math.isclose(xa, xb, rel_tol=1e-6, abs_tol=1e-9)
    return False


class MathEquiv:
    """Mathematical equivalence for MATH answers. Uses `math-verify` when installed."""

    def __init__(self, timeout: int = 5, allow_fallback: bool = True, force_fallback: bool = False):
        self.timeout = timeout
        self._mv = None
        self.backend = "fallback"
        self.version = "fallback-simple-1"
        if not force_fallback:
            try:
                import math_verify  # type: ignore

                self._mv = math_verify
                self.backend = "math_verify"
                self.version = importlib.metadata.version("math-verify")
            except Exception:
                if not allow_fallback:
                    raise
        self._parse_cache: dict = {}
        self._pair_cache: dict = {}

    @staticmethod
    def norm(s: Any) -> str:
        s = str(s).strip()
        if s.startswith("\\boxed{") and s.endswith("}"):
            inner = extract_last_boxed(s)
            if inner is not None:
                s = inner
        s = s.strip().strip("$").strip()
        return re.sub(r"\s+", " ", s)

    def _parse(self, s: str):
        if s in self._parse_cache:
            return self._parse_cache[s]
        wrapped = "\\boxed{" + s + "}"
        try:
            try:
                parsed = self._mv.parse(wrapped, parsing_timeout=self.timeout)
            except TypeError:
                parsed = self._mv.parse(wrapped)
        except Exception:
            parsed = []
        self._parse_cache[s] = parsed
        return parsed

    def _verify(self, gold, target) -> bool:
        try:
            try:
                return bool(self._mv.verify(gold, target, timeout_seconds=self.timeout))
            except TypeError:
                return bool(self._mv.verify(gold, target))
        except Exception:
            return False

    def __call__(self, a: Any, b: Any) -> bool:
        na, nb = self.norm(a), self.norm(b)
        if na == nb:
            return True
        key = (na, nb) if na <= nb else (nb, na)
        if key in self._pair_cache:
            return self._pair_cache[key]
        if self._mv is None:
            res = _fallback_equiv(na, nb)
        else:
            pa, pb = self._parse(na), self._parse(nb)
            res = bool(pa) and bool(pb) and (self._verify(pa, pb) or self._verify(pb, pa))
        self._pair_cache[key] = res
        return res


class MathAnswerContext:
    """Answer handling for the MATH dataset."""

    name = "math"

    def __init__(self, timeout: int = 5, allow_fallback: bool = True, force_fallback: bool = False):
        self.eq = MathEquiv(timeout=timeout, allow_fallback=allow_fallback, force_fallback=force_fallback)

    @property
    def scorer_info(self) -> dict:
        return {"dataset": "math", "scorer": self.eq.backend, "version": self.eq.version}

    def meta_for(self, qid: str) -> dict:
        return {}

    def normalize(self, raw: Any, meta: Optional[dict] = None) -> str:
        if is_missing(raw):
            return NO_ANSWER
        s = MathEquiv.norm(raw)
        return s if s else NO_ANSWER

    def normalize_gold(self, raw: Any, meta: Optional[dict] = None) -> str:
        return self.normalize(raw, meta)

    def same(self, a: str, b: str, meta: Optional[dict] = None) -> bool:
        return self.eq(a, b)


# --------------------------------------------------------------------------------------
# MathVista canonicalisation (answer type / choices / precision)
# --------------------------------------------------------------------------------------
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|[-+]?\.\d+")
_LETTER_PREFIX_RE = re.compile(r"^\(?([A-Za-z])\)?[\.\):]?(?:\s|$)")


def _norm_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"^(the )?(final )?answer\s*(is)?\s*[:\-]?\s*", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_number(s: Any) -> Optional[float]:
    """Parse a number from an answer string ($, %, thousands separators, a single embedded number)."""
    t = str(s).strip().replace("$", "").replace("%", "").replace("\\%", "").strip()
    if _THOUSANDS_RE.match(t):
        t = t.replace(",", "")
    try:
        v = float(t)
        return v if math.isfinite(v) else None
    except ValueError:
        pass
    found = _NUM_RE.findall(t)
    if len(found) == 1:
        try:
            v = float(found[0].replace(",", ""))
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


def _choices_list(choices: Any) -> list:
    if choices is None:
        return []
    if isinstance(choices, float) and math.isnan(choices):
        return []
    try:
        return [str(c) for c in list(choices)]
    except TypeError:
        return []


def to_option_letter(ans: Any, choices: Any) -> Optional[str]:
    """Map an answer (letter, '(B)', 'B.', or full choice text) to an uppercase option letter."""
    ch = _choices_list(choices)
    s = str(ans).strip()
    if not s:
        return None
    n = len(ch)
    max_letter = chr(ord("A") + n - 1) if n else "Z"
    core = s.strip("() .:")
    if len(core) == 1 and core.isalpha() and core.upper() <= max_letter:
        return core.upper()
    m = _LETTER_PREFIX_RE.match(s)
    if m and m.group(1).isalpha() and m.group(1).upper() <= max_letter and (m.group(1).isupper() or len(core) == 1):
        rest = s[m.end():]
        # "A) 12" or "(B)": accept, unless it is a plain English word starting with a capital 'A'/'I'
        if not rest.strip() or s[0] in "(" or s[1:2] in (")", ".", ":"):
            return m.group(1).upper()
    ns = _norm_text(s)
    for i, c in enumerate(ch):
        if _norm_text(c) == ns and ns != "":
            return chr(ord("A") + i)
    return None


def canonical_mathvista(raw: Any, meta: Optional[dict]) -> str:
    """Canonical comparison key for a MathVista answer, or NO_ANSWER for missing extractions."""
    if is_missing(raw):
        return NO_ANSWER
    meta = meta or {}
    qtype = str(meta.get("question_type") or "").lower()
    atype = str(meta.get("answer_type") or "").lower()
    s = str(raw).strip()
    if qtype == "multi_choice":
        letter = to_option_letter(s, meta.get("choices"))
        if letter is not None:
            return "OPT:" + letter
        return "TXT:" + _norm_text(s)
    if atype == "integer":
        v = parse_number(s)
        if v is not None:
            return "NUM:" + str(int(v))  # official MathVista rule: int(float(x))
        return "TXT:" + _norm_text(s)
    if atype == "float":
        v = parse_number(s)
        prec = meta.get("precision")
        try:
            prec = 2 if prec is None or (isinstance(prec, float) and math.isnan(prec)) else int(prec)
        except (TypeError, ValueError):
            prec = 2
        if v is not None:
            return "NUM:" + repr(round(v, prec) + 0.0)
        return "TXT:" + _norm_text(s)
    t = _norm_text(s)
    return ("TXT:" + t) if t else NO_ANSWER


def load_mathvista_metadata(path) -> dict:
    """Load released MathVista metadata (JSON dict keyed by pid, or list of records with 'pid')."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    return {str(r["pid"]): r for r in obj}


class MathVistaAnswerContext:
    """Answer handling for MathVista (testmini) using released question metadata."""

    name = "mathvista"
    version = "mathvista-canon-1"

    def __init__(self, metadata: Optional[dict] = None):
        self.metadata = metadata or {}

    @property
    def scorer_info(self) -> dict:
        return {"dataset": "mathvista", "scorer": "mathvista-canonical", "version": self.version}

    def meta_for(self, qid: str) -> dict:
        return self.metadata.get(str(qid), {})

    def normalize(self, raw: Any, meta: Optional[dict] = None) -> str:
        return canonical_mathvista(raw, meta)

    def normalize_gold(self, raw: Any, meta: Optional[dict] = None) -> str:
        return canonical_mathvista(raw, meta)

    def same(self, a: str, b: str, meta: Optional[dict] = None) -> bool:
        return a == b


# --------------------------------------------------------------------------------------
# Grouping + grading
# --------------------------------------------------------------------------------------
def build_question_groups(ctx, raw_answers, gold_raw, meta: Optional[dict] = None) -> dict:
    """Group a question's rollout answers pairwise (no gold), then grade each group against gold.

    Group id 0 is always NO_ANSWER. Other ids are assigned in rollout-index order (not permutation
    order), so grouping is independent of the evaluation order. Returns dict with
      gid [R] int16, reps [G] (representative strings), gcorrect [G] bool, gold, norm [R].
    """
    norm = [ctx.normalize(a, meta) for a in raw_answers]
    reps = [NO_ANSWER]
    gid = np.zeros(len(norm), dtype=np.int16)
    cache: dict = {}
    for i, s in enumerate(norm):
        if s == NO_ANSWER:
            continue
        if s in cache:
            gid[i] = cache[s]
            continue
        found = None
        for j in range(1, len(reps)):
            if ctx.same(reps[j], s, meta):
                found = j
                break
        if found is None:
            reps.append(s)
            found = len(reps) - 1
        cache[s] = found
        gid[i] = found
    gold = ctx.normalize_gold(gold_raw, meta)
    gcorrect = np.array(
        [(j != 0) and (gold != NO_ANSWER) and bool(ctx.same(reps[j], gold, meta)) for j in range(len(reps))],
        dtype=bool,
    )
    return {"gid": gid, "reps": reps, "gcorrect": gcorrect, "gold": gold, "norm": norm}
