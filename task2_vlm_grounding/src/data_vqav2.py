"""VQAv2 complementary pairs: answer normalisation, consensus filtering, selection.

Pure python (no torch) so the manifest can be built on any machine.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from task2_vlm_grounding.src.config import VQA_TYPE_NAMES

# --------------------------------------------------------------------------------------
# Official VQA answer normalisation (port of the VQA evaluation helper). Keys lower-cased
# so that they can match lower-cased text.
# --------------------------------------------------------------------------------------
CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "id've": "i'd've", "i'dve": "i'd've", "im": "i'm",
    "ive": "i've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")  # drop '.' unless followed by a digit (decimals kept)
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def _process_punctuation(text: str) -> str:
    out = text
    has_num_comma = _COMMA_STRIP.search(text) is not None
    for p in _PUNCT:
        if (p + " " in text) or (" " + p in text) or has_num_comma:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _PERIOD_STRIP.sub("", out)


def normalize_vqa_answer(ans: str | None) -> str:
    """Lower-case, strip newlines/tabs/articles, map number words, fix contractions,
    drop commas inside numbers, keep decimal points, collapse whitespace."""
    if ans is None:
        return ""
    s = str(ans).replace("\n", " ").replace("\t", " ").strip().lower()
    s = _process_punctuation(s)
    words = []
    for w in s.split():
        w = MANUAL_MAP.get(w, w)
        if w not in ARTICLES:
            words.append(w)
    words = [CONTRACTIONS.get(w, w) for w in words]
    return " ".join(words)


def consensus_answer(raw_answers: Sequence[str]) -> tuple[str, int, str]:
    """(top normalised answer, #annotators agreeing, representative original string).

    The representative is the most frequent *original* string inside the consensus group
    (ties: alphabetical), used as the training target.
    """
    norms = [normalize_vqa_answer(a) for a in raw_answers]
    counts = Counter(norms)
    top, n = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    raw_counts = Counter(a for a, nrm in zip(raw_answers, norms) if nrm == top)
    rep = sorted(raw_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return top, n, rep


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _find_json(raw_dir: Path, exact: str, fallback_glob: str) -> Path:
    hits = sorted(Path(raw_dir).rglob(exact))
    if not hits:
        hits = sorted(Path(raw_dir).rglob(fallback_glob))
    if not hits:
        raise FileNotFoundError(f"Could not find {exact} (or {fallback_glob}) under {raw_dir}")
    return hits[0]


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pair_ids(path: Path) -> list[tuple[int, int]]:
    data = _load_json(path)
    pairs = []
    for item in data:
        if isinstance(item, dict):
            vals = list(item.values())
            pairs.append((int(vals[0]), int(vals[1])))
        else:
            pairs.append((int(item[0]), int(item[1])))
    return pairs


def load_vqa_split(raw_dir: Path, split: str) -> tuple[dict, dict, list[tuple[int, int]]]:
    """Return (questions, annotations, pairs) restricted to question ids used in pairs.

    split is 'train2014' or 'val2014'.
    """
    pairs = load_pair_ids(_find_json(raw_dir, f"v2_mscoco_{split}_complementary_pairs.json",
                                     f"*{split}*complementary*pairs*.json"))
    needed = {q for p in pairs for q in p}
    qs_raw = _load_json(_find_json(raw_dir, f"v2_OpenEnded_mscoco_{split}_questions.json",
                                   f"*{split}*questions*.json"))["questions"]
    questions = {int(q["question_id"]): {"question": q["question"], "image_id": int(q["image_id"])}
                 for q in qs_raw if int(q["question_id"]) in needed}
    del qs_raw
    an_raw = _load_json(_find_json(raw_dir, f"v2_mscoco_{split}_annotations.json",
                                   f"*{split}*annotations*.json"))["annotations"]
    annotations = {}
    for a in an_raw:
        qid = int(a["question_id"])
        if qid in needed:
            annotations[qid] = {
                "answers": [x["answer"] for x in a["answers"]],
                "answer_type": a["answer_type"],
                "image_id": int(a["image_id"]),
            }
    del an_raw
    return questions, annotations, pairs


# --------------------------------------------------------------------------------------
# Candidate construction and selection
# --------------------------------------------------------------------------------------
def build_candidate_pairs(questions: dict, annotations: dict, pairs: Iterable[tuple[int, int]],
                          coco_split: str, min_agreement: int = 8) -> tuple[list[dict], dict]:
    """Apply the task filters: >=min_agreement/10 annotators on the top normalised answer for BOTH
    images and strictly different top normalised answers. Returns (candidates, drop statistics)."""
    stats: Counter = Counter()
    cands = []
    short = "train" if coco_split.startswith("train") else "val"
    for q1, q2 in pairs:
        stats["pairs_total"] += 1
        if q1 not in questions or q2 not in questions or q1 not in annotations or q2 not in annotations:
            stats["drop_missing"] += 1
            continue
        a1, a2 = annotations[q1], annotations[q2]
        if len(a1["answers"]) != 10 or len(a2["answers"]) != 10:
            stats["drop_not_10_answers"] += 1
            continue
        if questions[q1]["question"].strip().lower() != questions[q2]["question"].strip().lower():
            stats["drop_question_mismatch"] += 1
            continue
        if a1["answer_type"] != a2["answer_type"] or a1["answer_type"] not in VQA_TYPE_NAMES:
            stats["drop_type_mismatch"] += 1
            continue
        if questions[q1]["image_id"] == questions[q2]["image_id"]:
            stats["drop_same_image"] += 1
            continue
        top1, n1, rep1 = consensus_answer(a1["answers"])
        top2, n2, rep2 = consensus_answer(a2["answers"])
        if n1 < min_agreement or n2 < min_agreement:
            stats["drop_agreement"] += 1
            continue
        if top1 == "" or top2 == "":
            stats["drop_empty_answer"] += 1
            continue
        if top1 == top2:
            stats["drop_same_answer"] += 1
            continue
        stats["kept"] += 1
        kind = VQA_TYPE_NAMES[a1["answer_type"]]
        items = []
        for qid, top, rep, ann in ((q1, top1, rep1, a1), (q2, top2, rep2, a2)):
            items.append({
                "example_id": f"q{qid}",
                "question_id": int(qid),
                "image_id": int(questions[qid]["image_id"]),
                "answer": rep,
                "canonical": top,
                "all_answers": list(ann["answers"]),
            })
        cands.append({
            "pair_id": f"vqa_{short}_{q1}_{q2}",
            "suite": "natural_vqa",
            "kind": kind,
            "question": questions[q1]["question"].strip(),
            "coco_split": coco_split,
            "items": items,
        })
        stats[f"kept_{kind}"] += 1
    return cands, dict(stats)


def select_vqa_pairs(cands: Sequence[dict], counts: dict[str, int], rng: random.Random,
                     used_images: set[int] | None = None) -> list[dict]:
    """Stratified (yesno/number/other) selection.

    If `used_images` is given (evaluation), both images of a chosen pair must be unused,
    and the images are added to the set (global image-disjointness, also within the suite).
    """
    by_kind: dict[str, list[dict]] = {k: [] for k in counts}
    for c in sorted(cands, key=lambda c: c["pair_id"]):
        if c["kind"] in by_kind:
            by_kind[c["kind"]].append(c)
    selected: list[dict] = []
    for kind in counts:  # dict order: yesno, number, other
        pool = by_kind[kind]
        rng.shuffle(pool)
        chosen = []
        for c in pool:
            if len(chosen) == counts[kind]:
                break
            imgs = {it["image_id"] for it in c["items"]}
            if used_images is not None:
                if imgs & used_images:
                    continue
                used_images |= imgs
            chosen.append(c)
        if len(chosen) < counts[kind]:
            raise RuntimeError(f"Only {len(chosen)}/{counts[kind]} '{kind}' VQA pairs available")
        selected.extend(chosen)
    return selected
