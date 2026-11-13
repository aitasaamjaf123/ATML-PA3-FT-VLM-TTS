"""Deterministic qualitative dump of complementary-image examples.

Selection rule (stated explicitly, no manual cherry-picking). For every suite, pairs are classified
using the predictions with the CORRECT image for both members:
  success            both predictions match their original canonical answers
  identical_answer   the two predictions are identical (the answer ignored the image difference)
  partial            predictions differ and exactly one matches its reference
  unsupported_change predictions differ but neither matches its reference
Within a category, pairs are ranked by mean reference NLL under the correct image (ascending for
`success`, descending for the failure categories; ties broken by pair_id) and the first K are shown.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

from common.io import ensure_dir, write_json

CATEGORIES = ("success", "identical_answer", "partial", "unsupported_change")


def classify(two_correct: Sequence[dict]) -> str:
    exact = [r["exact"] == 1.0 for r in two_correct]
    if all(exact):
        return "success"
    if two_correct[0]["pred_norm"] == two_correct[1]["pred_norm"]:
        return "identical_answer"
    return "partial" if any(exact) else "unsupported_change"


def build_records(rows: Sequence[dict], manifest: dict) -> dict[str, list[dict]]:
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        by[r["suite"]][r["pair_id"]][r["condition"]][r["idx"]] = r
    out: dict[str, list[dict]] = {}
    for suite, pairs in by.items():
        recs = []
        for pid, conds in sorted(pairs.items()):
            c, s = conds["correct"], conds["complementary"]
            two = [c[0], c[1]]
            recs.append({
                "pair_id": pid, "suite": suite, "qtype": two[0]["qtype"], "category": classify(two),
                "question": two[0]["question"],
                "mean_nll_correct": (two[0]["nll"] + two[1]["nll"]) / 2,
                "members": [{
                    "image_id": two[k]["image_correct"], "reference": two[k]["reference"],
                    "canonical": two[k]["canonical"],
                    "prediction_correct_image": two[k]["prediction"],
                    "prediction_swapped_image": s[k]["prediction"],   # this question + the partner's image
                    "prediction_no_image": conds["no_image"][k]["prediction"],
                    "prediction_mismatched_image": conds["mismatched"][k]["prediction"],
                    "nll_correct": two[k]["nll"], "nll_swapped": s[k]["nll"],
                    "follows_partner_after_swap": bool(s[k]["partner_follow"]),
                } for k in (0, 1)],
            })
        out[suite] = recs
    return out


def select(records: Sequence[dict], k: int) -> dict[str, list[dict]]:
    chosen: dict[str, list[dict]] = {}
    for cat in CATEGORIES:
        pool = [r for r in records if r["category"] == cat]
        pool.sort(key=lambda r: (r["mean_nll_correct"] if cat == "success" else -r["mean_nll_correct"], r["pair_id"]))
        chosen[cat] = pool[:k]
    return chosen


def to_markdown(selected: dict[str, dict[str, list[dict]]]) -> str:
    lines = ["# Qualitative complementary-pair examples", "", __doc__.split("\n", 1)[1].strip(), ""]
    for suite, cats in selected.items():
        lines += [f"## Suite: {suite}", ""]
        for cat, recs in cats.items():
            lines += [f"### {cat} ({len(recs)} shown)", ""]
            if not recs:
                lines += ["_no pairs in this category_", ""]
                continue
            for r in recs:
                lines += [f"**{r['pair_id']}** ({r['qtype']}) - Q: *{r['question']}*", "",
                          "| image | reference | correct image | partner image (swap) | no image | mismatched | follows partner? |",
                          "|---|---|---|---|---|---|---|"]
                for m in r["members"]:
                    lines.append(f"| {m['image_id']} | {m['reference']} | {m['prediction_correct_image']} | "
                                 f"{m['prediction_swapped_image']} | {m['prediction_no_image']} | "
                                 f"{m['prediction_mismatched_image']} | {m['follows_partner_after_swap']} |")
                lines.append("")
    return "\n".join(lines)


def render_figures(selected: dict[str, dict[str, list[dict]]], raw_dir: Path, out_dir: Path) -> int:
    """Side-by-side image figure for each selected pair (skipped silently if images are missing)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from task2_vlm_grounding.src.data_coco import coco_image_path

    ensure_dir(out_dir)
    n = 0
    for suite, cats in selected.items():
        for cat, recs in cats.items():
            for r in recs:
                fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
                ok = True
                for ax, m in zip(axes, r["members"]):
                    p = coco_image_path(raw_dir, "val2014", m["image_id"])
                    if not p.exists():
                        ok = False
                        break
                    ax.imshow(Image.open(p).convert("RGB"))
                    ax.axis("off")
                    ax.set_title(f"ref: {m['reference']}\npred: {m['prediction_correct_image']}", fontsize=9)
                if ok:
                    fig.suptitle(f"[{suite}/{cat}] {r['question']}", fontsize=9)
                    fig.tight_layout()
                    fig.savefig(out_dir / f"{suite}_{cat}_{r['pair_id']}.png", dpi=110)
                    n += 1
                plt.close(fig)
    return n


def dump_qualitative(rows: Sequence[dict], manifest: dict, cfg: dict, paths) -> dict:
    k = cfg["eval"]["qualitative_per_category"]
    records = build_records(rows, manifest)
    selected = {suite: select(recs, k) for suite, recs in records.items()}
    counts = {suite: {c: sum(1 for r in recs if r["category"] == c) for c in CATEGORIES}
              for suite, recs in records.items()}
    ensure_dir(paths.qual_dir)
    write_json({"selection_rule": __doc__, "category_counts": counts, "selected": selected},
               paths.qual_dir / "qualitative_examples.json")
    (paths.qual_dir / "qualitative_examples.md").write_text(to_markdown(selected), encoding="utf-8")
    write_json(records, paths.qual_dir / "all_pair_records.json", indent=None)
    n_fig = render_figures(selected, paths.raw, paths.figures / "qualitative")
    return {"category_counts": counts, "n_figures": n_fig}
