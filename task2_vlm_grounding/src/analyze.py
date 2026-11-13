"""Learning curves, final tables and resource summary from saved logs (no GPU needed).

Usage: python -m task2_vlm_grounding.src.analyze [--smoke]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.io import read_json, read_jsonl, write_csv  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.plotting import line_plot, smooth  # noqa: E402
from task2_vlm_grounding.src.config import get_paths, load_config, stage_list  # noqa: E402
from task2_vlm_grounding.src.vqa_eval import CONDITIONS  # noqa: E402

log = get_logger("analyze")
STAGE_LABELS = ["S2: LoRA + VQA-lite", "S3: natural VQA"]


def dedup_last(rows: list[dict], key: str) -> list[dict]:
    """Resumed runs may re-log an index: keep the last record per key, sorted."""
    return [r for _, r in sorted({r[key]: r for r in rows}.items())]


def plot_training(paths, cfg, boundaries) -> None:
    rows = dedup_last(read_jsonl(paths.logs_dir / "train_log.jsonl"), "update")
    if not rows:
        return
    x = [r["update"] for r in rows]
    y = np.array([r["loss"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(x, y, alpha=0.3, label="per update")
    ax.plot(x, smooth(y, max(1, len(y) // 40)), label="smoothed")
    for b in boundaries:
        ax.axvline(b, color="k", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set(title="Training loss (example-weighted CE)", xlabel="optimizer update", ylabel="loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures / "training_loss.png", dpi=150)
    plt.close(fig)


def plot_dev(paths, cfg, boundaries) -> list[dict]:
    dev = dedup_last(read_jsonl(paths.dev_dir / "dev_metrics.jsonl"), "step")
    if not dev:
        return dev
    steps = [d["step"] for d in dev]
    F = paths.figures
    line_plot({"caption dev CE": (steps, [d["caption_dev_loss"] for d in dev])}, F / "dev_caption_loss.png",
              "Dev caption reference loss", ylabel="CE (incl. <|im_end|>)", boundaries=boundaries, boundary_labels=STAGE_LABELS)
    for suite in ("grounding", "vqa"):
        if f"{suite}_nll_correct" not in dev[0]:
            continue
        line_plot({c: (steps, [d[f"{suite}_nll_{c}"] for d in dev]) for c in ("correct", "complementary", "no_image")},
                  F / f"dev_answer_nll_{suite}.png", f"Dev reference-answer NLL ({suite})", ylabel="token NLL",
                  boundaries=boundaries, boundary_labels=STAGE_LABELS)
    line_plot({s: (steps, [d[f"{s}_delta_nll"] for d in dev]) for s in ("grounding", "vqa")},
              F / "dev_image_swap_penalty.png", "Image-swap penalty  NLL(partner) - NLL(correct)",
              ylabel="Delta NLL", boundaries=boundaries, boundary_labels=STAGE_LABELS, hline=0.0)
    if f"grounding_acc_correct" in dev[0]:
        ser = {}
        for s in ("grounding", "vqa"):
            for c in ("correct", "complementary"):
                ser[f"{s} / {c}"] = (steps, [d[f"{s}_acc_{c}"] for d in dev])
        line_plot(ser, F / "dev_generated_accuracy.png", "Dev generated accuracy", ylabel="accuracy",
                  boundaries=boundaries, boundary_labels=STAGE_LABELS)
    write_csv(dev, paths.dev_dir / "dev_metrics.csv")
    return dev


def final_tables(paths) -> None:
    f = paths.final_dir / "final_summary.json"
    if not f.exists():
        return
    summary = read_json(f)
    rows = summary["rows"]
    fmt = lambda r: (f"{r['mean']:.3f}" if r["ci_low"] == "" else f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    lines = [f"# Final test results ({summary['n_resamples']} pair-bootstrap resamples, 95% percentile CI)", "",
             "n pairs: " + ", ".join(f"{k}={v}" for k, v in summary["n_pairs"].items()), ""]
    for suite in summary["n_pairs"]:
        lines += [f"## {suite}", ""]
        subsets = sorted({r["subset"] for r in rows if r["suite"] == suite}, key=lambda s: (s != "overall", s))
        metrics = ["acc", "pair_acc", "identical", "empty", "nll", "follow", "unsupported", "gen_ms_per_answer"]
        for sub in subsets:
            lines += [f"### subset: {sub}", "", "| condition | " + " | ".join(metrics) + " |",
                      "|---|" + "---|" * len(metrics)]
            for cond in CONDITIONS:
                cells = []
                for m in metrics:
                    hit = [r for r in rows if (r["suite"], r["condition"], r["subset"], r["metric"]) == (suite, cond, sub, m)]
                    cells.append(fmt(hit[0]) if hit else "-")
                lines.append(f"| {cond} | " + " | ".join(cells) + " |")
            lines.append("")
        lines += ["### correct-image advantages (overall)", "", "| comparison | metric | mean [95% CI] |", "|---|---|---|"]
        for r in rows:
            if r["suite"] == suite and r["subset"] == "overall" and r["condition"].startswith("correct_vs_"):
                lines.append(f"| {r['condition']} | {r['metric']} | {fmt(r)} |")
        lines.append("")
    (paths.final_dir / "final_tables.md").write_text("\n".join(lines), encoding="utf-8")


def resource_table(paths) -> None:
    out = ["# Resources", ""]
    cache = paths.features / "cache_resources.json"
    if cache.exists():
        c = read_json(cache)
        out += ["## Feature caching (separate from training)", "",
                f"- images: {c.get('n_images')}, wall-clock: {c.get('wall_clock_s', 0):.1f}s, "
                f"peak GPU alloc: {c.get('peak_gpu_allocated_mb', 0):.0f} MB, file: {c.get('file_mb', 0):.0f} MB, gpu: {c.get('gpu')}", ""]
    tr = paths.logs_dir / "resources_train.json"
    if tr.exists():
        t = read_json(tr)
        out += ["## Training", "", "| stage | updates | examples | train time (s) | peak GPU MB | trainable params (connector / LoRA / total) |",
                "|---|---|---|---|---|---|"]
        for key, s in t["stages"].items():
            p = s["trainable_params"]
            out.append(f"| {s.get('name', key)} | {s['updates']} | {s['examples_presented']} | {s['train_time_s']:.1f} | "
                       f"{s['peak_gpu_mb']:.0f} | {p['connector_trainable']:,} / {p['lora_trainable']:,} / {p['total_trainable']:,} |")
        tot = t["totals"]
        out += ["", f"Total: {tot['updates']} updates, {tot['examples_presented']} example presentations, "
                    f"{tot['train_time_s']:.1f}s training (+{tot['dev_eval_time_s']:.1f}s dev evaluation), "
                    f"peak {tot['peak_gpu_mb']:.0f} MB.", ""]
    ev = paths.final_dir / "resources_eval.json"
    if ev.exists():
        e = read_json(ev)
        out += ["## Final-test inference", "",
                f"- {e['n_answers_generated']} answers, mean generation {e['mean_generation_ms_per_answer']:.1f} ms/answer, "
                f"total {e['total_eval_wall_s']:.1f}s, peak GPU {e['peak_gpu_mb']:.0f} MB", ""]
    (paths.results / "resources.md").write_text("\n".join(out), encoding="utf-8")


def plot_final_bars(paths) -> None:
    f = paths.final_dir / "final_summary.json"
    if not f.exists():
        return
    rows = read_json(f)["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, suite in zip(axes, ("grounding", "vqa")):
        vals, lo, hi = [], [], []
        for c in CONDITIONS:
            r = [x for x in rows if (x["suite"], x["condition"], x["subset"], x["metric"]) == (suite, c, "overall", "acc")]
            if not r:
                vals.append(0), lo.append(0), hi.append(0)
                continue
            vals.append(r[0]["mean"]), lo.append(r[0]["mean"] - r[0]["ci_low"]), hi.append(r[0]["ci_high"] - r[0]["mean"])
        ax.bar(range(len(CONDITIONS)), vals, yerr=[lo, hi], capsize=3)
        ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=20)
        ax.set(title=f"{suite} accuracy (95% pair-bootstrap CI)", ylim=(0, 1))
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths.figures / "final_accuracy_by_condition.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    paths = get_paths(cfg)
    boundaries = [s["start"] for s in stage_list(cfg)[1:]]
    plot_training(paths, cfg, boundaries)
    plot_dev(paths, cfg, boundaries)
    final_tables(paths)
    plot_final_bars(paths)
    resource_table(paths)
    log.info("Analysis written to %s and %s", paths.figures, paths.results)


if __name__ == "__main__":
    main()
