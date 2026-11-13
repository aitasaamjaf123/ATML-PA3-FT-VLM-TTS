"""Matplotlib helpers (Agg backend).

Task 2: smooth, add_stage_lines, line_plot (vertical stage-boundary lines)
Task 3: PALETTE, METHOD_STYLE, apply_style, style_for, savefig (colourblind-safe, fixed-order palette)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.io import ensure_dir  # noqa: E402


def smooth(y: Sequence[float], window: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if window <= 1 or len(y) < 2:
        return y
    window = min(window, len(y))
    kernel = np.ones(window) / window
    pad = np.pad(y, (window - 1, 0), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


def add_stage_lines(ax, boundaries: Sequence[int], labels: Sequence[str] | None = None) -> None:
    for i, b in enumerate(boundaries):
        ax.axvline(b, color="k", linestyle="--", linewidth=0.9, alpha=0.6)
        if labels is not None and i < len(labels):
            ax.text(b, ax.get_ylim()[1], f" {labels[i]}", va="top", ha="left", fontsize=7, alpha=0.7)


def line_plot(
    series: dict[str, tuple[Sequence[float], Sequence[float]]],
    path: str | Path,
    title: str,
    xlabel: str = "optimizer update",
    ylabel: str = "",
    boundaries: Sequence[int] = (),
    boundary_labels: Sequence[str] | None = None,
    figsize: tuple[float, float] = (7, 4),
    hline: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    for name, (x, y) in series.items():
        ax.plot(x, y, marker="o" if len(x) < 40 else None, markersize=3, label=name)
    if hline is not None:
        ax.axhline(hline, color="gray", linewidth=0.8)
    add_stage_lines(ax, boundaries, boundary_labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if len(series) > 1 or any(series):
        ax.legend(fontsize=8)
    fig.tight_layout()
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Task 3: colourblind-safe fixed-order palette, thin marks, recessive grid
# ----------------------------------------------------------------------------------------------
# Okabe-Ito colourblind-safe palette, assigned in fixed order (never cycled).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A7A7A", "#000000"]

# Stable colour / style per method so a filter never repaints survivors.
METHOD_STYLE = {
    "first": dict(color="#7A7A7A", ls="-", marker="o", label="FirstAns"),
    "selfcons": dict(color="#0072B2", ls="-", marker="s", label="Self-consistency"),
    "logp_top": dict(color="#E69F00", ls="-", marker="^", label="Highest mean log-prob"),
    "logp_rwc": dict(color="#E69F00", ls="--", marker="^", label="Mean log-prob + RWC"),
    "selfcert_top": dict(color="#D55E00", ls="-", marker="v", label="Highest self-certainty"),
    "selfcert_rwc": dict(color="#009E73", ls="-", marker="D", label="Self-certainty + RWC"),
    "shortest": dict(color="#56B4E9", ls=":", marker="x", label="Shortest rollout"),
    "vision_top": dict(color="#56B4E9", ls="-", marker="<", label="Vision-only (highest)"),
    "vision_rwc": dict(color="#56B4E9", ls="--", marker="<", label="Vision-only + RWC"),
    "visconf_top": dict(color="#CC79A7", ls="-", marker="P", label="VisConf (highest)"),
    "visconf_rwc": dict(color="#000000", ls="-", marker="*", label="VisConf + RWC"),
    "tokenfusion_top": dict(color="#CC79A7", ls="--", marker="p", label="Token fusion (highest)"),
    "tokenfusion_rwc": dict(color="#000000", ls="--", marker="h", label="Token fusion + RWC"),
    "oracle": dict(color="#000000", ls=":", marker=None, label="Oracle coverage"),
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#888888",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.constrained_layout.use": True,
        }
    )


def style_for(method: str, i: int = 0) -> dict:
    if method in METHOD_STYLE:
        return dict(METHOD_STYLE[method])
    return dict(color=PALETTE[i % len(PALETTE)], ls="-", marker="o", label=method)


def savefig(fig, path_stem) -> list:
    """Save PNG + PDF; returns written paths."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = Path(str(stem) + "." + ext)
        fig.savefig(p, bbox_inches="tight")
        out.append(str(p))
    plt.close(fig)
    return out
