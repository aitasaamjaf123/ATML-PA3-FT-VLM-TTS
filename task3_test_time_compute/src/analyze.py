"""Pipeline orchestration + figures + summary report for Task 3.

    from task3_test_time_compute.src import analyze
    cfg = analyze.load_config(smoke=True)
    analyze.run_pipeline(cfg, stages=["offline", "signals", "adaptive", "live", "qualitative", "report"], smoke=True)
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common.io import (df_to_markdown, ensure_dir, load_json, load_jsonl, load_yaml, save_csv, save_json, save_table, write_text)
from common.logging_utils import get_logger, package_versions, timed
from common.plotting import PALETTE, apply_style, savefig, style_for
from common.seed import set_global_seed
from task3_test_time_compute.src import adaptive as AD
from task3_test_time_compute.src import answers as A
from task3_test_time_compute.src import load_data as LD
from task3_test_time_compute.src import live_generation as LG
from task3_test_time_compute.src import offline_eval as OE
from task3_test_time_compute.src import orders as OR
from task3_test_time_compute.src import qualitative as QL
from task3_test_time_compute.src import signal_analysis as SA
from task3_test_time_compute.src import splits as SP

log = get_logger("task3")
ALL_STAGES = ["offline", "signals", "adaptive", "live", "qualitative", "report"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else repo_root() / p


def load_config(path=None, smoke: bool = False, overrides: Optional[dict] = None) -> dict:
    cfg = load_yaml(path or (repo_root() / "task3_test_time_compute" / "configs" / "task3.yaml"))
    cfg["smoke_mode"] = bool(smoke)
    if smoke:
        s = cfg["smoke"]
        cfg["n_orders"], cfg["n_bootstrap"] = s["n_orders"], s["n_bootstrap"]
    for k, v in (overrides or {}).items():
        cfg[k] = v
    return cfg


def results_root(cfg) -> Path:
    base = _abs(cfg["paths"]["results_dir"])
    return base.parent / (base.name + "_smoke") if cfg.get("smoke_mode") else base


def figures_root(cfg) -> Path:
    base = _abs(cfg["paths"]["figures_dir"])
    return base.parent / (base.name + "_smoke") if cfg.get("smoke_mode") else base


def _sub(cfg, stage, ds=None) -> Path:
    p = results_root(cfg) / stage
    return ensure_dir(p / ds if ds else p)


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def make_answer_ctx(cfg, ds: str):
    if ds == "math":
        return A.MathAnswerContext(timeout=cfg["answers"]["math_verify_timeout"], allow_fallback=cfg["answers"]["allow_fallback_scorer"])
    meta_path = _abs(cfg["paths"]["mathvista_metadata"])
    if not meta_path.exists():
        raise FileNotFoundError(f"MathVista metadata not found at {meta_path}. Run `python data/scripts/download_task3.py` first.")
    return A.MathVistaAnswerContext(A.load_mathvista_metadata(meta_path))


def locate_data(cfg) -> dict:
    root = Path(cfg["paths"]["data_root"])
    found = LD.discover_model_dirs(root)
    for ds in cfg["datasets"]:
        if cfg["datasets"][ds].get("model_dir"):
            found[ds] = Path(cfg["datasets"][ds]["model_dir"])
    readme = Path(cfg["paths"]["readme"]) if cfg["paths"].get("readme") else LD.find_readme(root)
    return {"dirs": found, "readme": readme}


def get_pool(cfg, ds: str, needed_ids: Optional[Sequence[str]] = None):
    """Load + join + validate + build the pool. Returns (pool restricted to the needed ids, all sample ids)."""
    info = locate_data(cfg)
    if ds not in info["dirs"]:
        raise FileNotFoundError(f"no export folder found for '{ds}' under {cfg['paths']['data_root']}: {info['dirs']}")
    dcfg = cfg["datasets"][ds]
    vision = bool(dcfg["vision"])
    all_ids = LD.read_sample_ids(info["dirs"][ds])          # splits are defined over ALL ids, even in smoke mode
    with timed(log, f"[{ds}] load + join parquet"):
        df = LD.load_joined(info["dirs"][ds], vision, dcfg.get("columns"), keep_qids=needed_ids)
    with timed(log, f"[{ds}] alignment check"):
        report = LD.check_alignment(df, vision, strict=True, n_rollouts=cfg["n_rollouts"])
    ctx = make_answer_ctx(cfg, ds)
    with timed(log, f"[{ds}] answer groups + signals ({df['qid'].nunique()} questions)"):
        pool = LD.build_pool(df, ds, ctx, vision, token_cap=cfg["token_cap"], alignment=report)
    return pool, all_ids


def _strip_instruction(problem: str) -> str:
    """The live prompt appends the think/boxed instruction itself; drop it if the stored question already has it."""
    i = problem.find("Think step by step")
    return problem[:i].rstrip() if i > 0 else problem


def _problems_for(cfg, pool, qids) -> Dict[str, str]:
    probs = {q: _strip_instruction(pool.questions[q].problem) for q in qids if pool.questions[q].problem}
    missing = [q for q in qids if q not in probs]
    if missing:
        p = _abs(cfg["paths"]["math_problems"])
        if not p.exists():
            raise FileNotFoundError(
                f"Problem text for {len(missing)} live questions is not in generations.parquet and {p} is missing. "
                f"Run `python data/scripts/download_task3.py` (or set paths.math_problems).")
        extra = load_json(p)
        for q in missing:
            if q not in extra:
                raise KeyError(f"no problem text for sample_id {q} in {p}")
            probs[q] = _strip_instruction(extra[q])
    return probs


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
def run_pipeline(cfg, stages: Sequence[str] = ALL_STAGES, smoke: Optional[bool] = None, pools: Optional[Dict[str, object]] = None,
                 backend=None, datasets: Optional[Sequence[str]] = None) -> dict:
    """Run the requested stages. `pools` (dataset -> Pool) can be injected (tests); `backend` overrides the HF backend."""
    if smoke is not None:
        cfg = copy.deepcopy(cfg)
        if smoke and not cfg.get("smoke_mode"):
            cfg["smoke_mode"] = True
            cfg["n_orders"], cfg["n_bootstrap"] = cfg["smoke"]["n_orders"], cfg["smoke"]["n_bootstrap"]
    set_global_seed(cfg["seed"])
    smoke_mode = bool(cfg.get("smoke_mode"))
    datasets = list(datasets or cfg["datasets"].keys())
    state: Dict[str, dict] = {}
    root = results_root(cfg)
    save_json({"stages": list(stages), "smoke": smoke_mode, "config": cfg, "versions": package_versions()}, root / "run_config.json")

    for ds in datasets:
        st: dict = {}
        state[ds] = st
        if pools and ds in pools:
            pool_all, all_ids = pools[ds], pools[ds].qids()
        else:
            tag = "smoke" if smoke_mode else "full"
            cache = ensure_dir(_abs(cfg["paths"]["processed_dir"])) / f"{ds}_pool_{tag}.pkl"
            if cache.exists():
                pool_all = LD.load_pool(cache)
                all_ids = pool_all.alignment.get("all_sample_ids", pool_all.qids())
            else:
                # Splits need ALL question ids, so read the id column first; then build the pool only for the
                # questions that are needed (dev + held-out; a small subset in smoke mode).
                info = locate_data(cfg)
                probe_ids = LD.read_sample_ids(info["dirs"][ds])
                sp0 = SP.make_splits(probe_ids, cfg["seed"], cfg["split"]["n_dev"], cfg["split"]["n_live"])
                if smoke_mode:
                    sp0 = SP.smoke_subset(sp0, cfg["smoke"]["n_dev"], cfg["smoke"]["n_eval"], cfg["smoke"]["n_live"])
                pool_all, all_ids = get_pool(cfg, ds, sp0["dev"] + sp0["eval"])
                pool_all.alignment["all_sample_ids"] = all_ids
                LD.save_pool(pool_all, cache)
        sp_full = SP.make_splits(all_ids, cfg["seed"], cfg["split"]["n_dev"], cfg["split"]["n_live"])
        sp = SP.smoke_subset(sp_full, cfg["smoke"]["n_dev"], cfg["smoke"]["n_eval"], cfg["smoke"]["n_live"]) if smoke_mode else sp_full
        SP.save_splits(sp, _abs(cfg["paths"]["splits_dir"]) / ("smoke" if smoke_mode else ""), ds)
        pool = pool_all.subset(sp["dev"] + sp["eval"])
        orders = OR.make_orders(sp["dev"] + sp["eval"], cfg["n_orders"], cfg["n_rollouts"], cfg["seed"])
        OR.save_orders(orders, _sub(cfg, "offline", ds) / "orders.json")
        st.update(pool=pool, splits=sp, orders=orders)
        save_json({"description": LD.describe_pool(pool), "alignment": {k: v for k, v in pool.alignment.items() if k != "all_sample_ids"},
                   "n_dev": len(sp["dev"]), "n_eval": len(sp["eval"]), "n_live": len(sp["live"]), "scorer": pool.scorer},
                  _sub(cfg, "offline", ds) / "data_meta.json")

        if any(s in stages for s in ("offline", "signals", "adaptive", "qualitative")):
            st["offline"] = stage_offline(cfg, ds, pool, sp, orders)
        if "signals" in stages:
            stage_signals(cfg, ds, pool, sp, st["offline"])
        if "adaptive" in stages or "qualitative" in stages:
            st["adaptive"] = stage_adaptive(cfg, ds, pool, sp, orders, st["offline"])
    if "live" in stages and "math" in state:
        state["math"]["live"] = stage_live(cfg, state["math"]["pool"], state["math"]["splits"], backend)
    if "qualitative" in stages:
        for ds in datasets:
            stage_qualitative(cfg, ds, state[ds])
    if "report" in stages:
        make_report(cfg, datasets)
    return state


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------
def stage_offline(cfg, ds, pool, sp, orders) -> OE.OfflineResults:
    out = _sub(cfg, "offline", ds)
    with timed(log, f"[{ds}] offline evaluation ({len(sp['eval'])} held-out questions x {cfg['n_orders']} orders)"):
        res = OE.evaluate_pool(pool, sp["eval"], orders, cfg["budgets"])
    nb, seed = cfg["n_bootstrap"], cfg["seed"]
    save_csv(OE.method_table(res, nb, seed), out / "method_by_budget.csv")
    save_csv(OE.budget_table(res, nb, seed), out / "budget_summary.csv")
    save_csv(OE.diff_table(res, nb, seed), out / "paired_differences.csv")
    save_csv(OE.decomposition_table(res), out / "failure_decomposition.csv")
    save_table(OE.per_question_table(res), out / "per_question")
    save_table(OE.groups_table(pool, sp["eval"]), out / "groups")
    if pool.has_vision:
        save_csv(pd.concat([OE.selection_agreement(res, "visconf_top", "tokenfusion_top"),
                            OE.selection_agreement(res, "visconf_rwc", "tokenfusion_rwc"),
                            OE.selection_agreement(res, "visconf_top", "selfcert_top")]), out / "selector_agreement.csv")
    return res


def stage_signals(cfg, ds, pool, sp, res: OE.OfflineResults) -> None:
    out = _sub(cfg, "signals", ds)
    tbl = SA.question_signal_table(pool, sp["eval"], cfg["quartile_fraction"])
    save_table(tbl, out / "per_question_signals")
    save_csv(SA.summarize_quartiles(tbl, cfg["n_bootstrap"], cfg["seed"]), out / "quartile_summary.csv")
    last = len(res.budgets) - 1
    n32 = {m: float(res.acc_qb(m)[:, last].mean()) for m in res.methods if m != "first"}
    save_csv(SA.length_control_table(tbl, n32), out / "length_control.csv")
    save_csv(SA.length_correlations(tbl), out / "length_correlations.csv")
    if pool.has_vision:
        cv = SA.confidence_vision(pool, sp["eval"])
        save_csv(cv["per_question"], out / "confidence_vision_per_question.csv")
        save_json(cv["summary"], out / "confidence_vision_summary.json")
        picks = SA.select_illustrative_trajectories(pool, sp["eval"])
        for p in picks:
            qd = pool.questions[p["qid"]]
            p["series"] = {name: QL.trajectory_series(qd, p[key]) for name, key in
                           (("visconf_top", "visconf_idx"), ("selfcert_top", "selfcert_idx"))}
        save_json(picks, out / "trajectories.json")


def stage_adaptive(cfg, ds, pool, sp, orders, res: OE.OfflineResults) -> dict:
    out = _sub(cfg, "adaptive", ds)
    a = cfg["adaptive"]
    with timed(log, f"[{ds}] adaptive features (dev)"):
        F_dev = AD.build_features(pool, sp["dev"], orders, a["checkpoints"], a["final_n"])
    grid_dev = AD.evaluate_grid(F_dev, a["taus"], a["gammas"])
    save_csv(grid_dev, out / "dev_grid.csv")
    selected = {fam: AD.select_setting(grid_dev, fam, a["tolerance"]) for fam in ("agreement", "quality_aware")}
    save_json(selected, out / "selected_settings.json")  # frozen BEFORE held-out evaluation
    with timed(log, f"[{ds}] adaptive features (held-out)"):
        F_eval = AD.build_features(pool, sp["eval"], orders, a["checkpoints"], a["final_n"])
    rep = AD.heldout_report(F_eval, selected, res, cfg["n_bootstrap"], cfg["seed"])
    for k in ("summary", "diffs", "stop_distribution", "wrong_stops", "by_pool_fraction"):
        save_csv(rep[k], out / f"heldout_{k}.csv")
    save_csv(AD.evaluate_grid(F_eval, a["taus"], a["gammas"]), out / "heldout_grid_all_settings.csv")
    return {"selected": selected, "F_eval": F_eval, "report": rep}


def stage_live(cfg, pool, sp, backend=None) -> dict:
    out = _sub(cfg, "live")
    qids = sp["live"]
    probs = _problems_for(cfg, pool, qids)
    questions = [{"qid": q, "problem": probs[q], "gold": pool.questions[q].gold_raw} for q in qids]
    ctx = make_answer_ctx(cfg, "math")
    live_cfg = copy.deepcopy(cfg["live"])
    if backend is None:
        rev = live_cfg.get("revision")
        info = locate_data(cfg)
        if rev is None and info["readme"] is not None:
            rev = LD.parse_readme_revision(LD.read_readme(info["readme"]))
        if rev is None:
            log.warning("No revision found in README.MD or config; using the default branch (set live.revision for exact reproduction).")
        live_cfg["revision"] = rev
        backend = LG.HFBackend(live_cfg["model_id"], rev, live_cfg["sampling"], live_cfg["dtype"])
    with timed(log, f"[live] {len(questions)} questions x 5 generations"):
        return LG.run_live(backend, questions, live_cfg, ctx, out, cfg["seed"])


def stage_qualitative(cfg, ds, st) -> None:
    out = _sub(cfg, "qualitative", ds)
    pool, sp, orders = st["pool"], st["splits"], st["orders"]
    ex = QL.selection_examples(pool, sp["eval"], orders, n=16)
    if "adaptive" in st:
        F = st["adaptive"]["F_eval"]
        for fam, detail in st["adaptive"]["report"]["details"].items():
            ex[f"premature_stop_{fam}"] = QL.premature_stop_examples(pool, F, detail, orders, st["adaptive"]["report"]["fixed32"])
    if ds == "math":
        live_dir = _sub(cfg, "live")
        pq = live_dir / "per_question.csv"
        if pq.exists():
            rows = pd.read_csv(pq).to_dict("records")
            gens = {(str(r["qid"]), r["role"]): r for r in load_jsonl(live_dir / "generations.jsonl")}
            ex.update(QL.refinement_examples(rows, gens))
    save_json({"rules": QL.RULES, "examples": ex}, out / "examples.json")
    lines = ["# Qualitative examples\n"]
    for k, v in ex.items():
        lines.append(f"## {k}\n")
        lines.append(f"_Selection rule:_ {QL.RULES.get(k, QL.RULES['premature_stop'] if k.startswith('premature') else '')}\n")
        for c in v:
            lines.append("```json\n" + __import__("json").dumps(c, indent=1, ensure_ascii=False, default=str) + "\n```\n")
        if not v:
            lines.append("_No case satisfies this rule in this run._\n")
    write_text("\n".join(lines), out / "examples.md")


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _read(path) -> Optional[pd.DataFrame]:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def fig_accuracy(cfg, ds, fdir):
    import matplotlib.pyplot as plt

    mt = _read(_sub(cfg, "offline", ds) / "method_by_budget.csv")
    if mt is None:
        return
    methods = [m for m in ["first", "selfcons", "logp_top", "selfcert_top", "selfcert_rwc", "shortest", "vision_top", "vision_rwc",
                           "visconf_top", "visconf_rwc", "tokenfusion_rwc"] if m in set(mt.method)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in methods:
        d = mt[mt.method == m].sort_values("N")
        s = style_for(m)
        for ax, x in zip(axes, ("N", "mean_tokens")):
            xs = d[x] if m != "first" or x == "N" else d[x]
            ax.plot(xs, d.accuracy, color=s["color"], ls=s["ls"], marker=s["marker"], ms=4, label=s["label"])
    ref = mt[mt.method == methods[0]].sort_values("N")
    s = style_for("oracle")
    axes[0].plot(ref.N, ref.oracle_coverage, color=s["color"], ls=s["ls"], label=s["label"])
    sc = mt[mt.method == "selfcons"].sort_values("N")
    axes[1].plot(sc.mean_tokens, sc.oracle_coverage, color=s["color"], ls=s["ls"], label=s["label"])
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(cfg["budgets"])
    axes[0].set_xticklabels(cfg["budgets"])
    axes[1].set_xscale("log")
    axes[0].set_xlabel("candidates N")
    axes[1].set_xlabel("mean generated tokens per question")
    for ax in axes:
        ax.set_ylabel("accuracy / coverage")
    axes[1].legend(*axes[0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"{cfg['datasets'][ds]['pretty']}: accuracy vs. candidates and generated-token cost (held-out)")
    savefig(fig, fdir / f"accuracy_vs_budget_{ds}")


def fig_decomposition(cfg, ds, fdir):
    import matplotlib.pyplot as plt

    dec = _read(_sub(cfg, "offline", ds) / "failure_decomposition.csv")
    if dec is None:
        return
    methods = ["selfcons", "selfcert_rwc"] + (["visconf_rwc"] if ds == "mathvista" else [])
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.8), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), methods):
        d = dec[dec.method == m].sort_values("N")
        x = np.arange(len(d))
        ax.bar(x, d.selected_correct, color=PALETTE[2], label="selected correct", width=0.7)
        ax.bar(x, d.selection_failure, bottom=d.selected_correct, color=PALETTE[4], label="selection failure (correct candidate available)", width=0.7)
        ax.bar(x, d.sampling_failure, bottom=d.selected_correct + d.selection_failure, color=PALETTE[1], label="sampling failure (no correct candidate)", width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(d.N)
        ax.set_xlabel("candidates N")
        ax.set_title(style_for(m)["label"])
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("fraction of questions")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=7)
    savefig(fig, fdir / f"failure_decomposition_{ds}")


def fig_groups(cfg, ds, fdir):
    import matplotlib.pyplot as plt

    b = _read(_sub(cfg, "offline", ds) / "budget_summary.csv")
    if b is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
    ax[0].plot(b.N, b.mean_distinct_groups, marker="o", color=PALETTE[0], label="distinct answer groups")
    ax[0].plot(b.N, b.N, ls=":", color="#888888", label="N (all different)")
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log", base=2)
    ax[0].set_xlabel("candidates N")
    ax[0].legend()
    ax[1].plot(b.N, b.oracle_coverage, marker="o", color="#000000", label="oracle coverage")
    ax[1].plot(b.N, b.first_accuracy, marker="s", color=PALETTE[6], label="FirstAns accuracy")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("candidates N")
    ax[1].legend()
    savefig(fig, fdir / f"answer_diversity_{ds}")


def fig_quartiles(cfg, ds, fdir):
    import matplotlib.pyplot as plt

    q = _read(_sub(cfg, "signals", ds) / "quartile_summary.csv")
    if q is None:
        return
    q = q[q.subset == "mixed_questions"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    y = np.arange(len(q))
    ax.barh(y, q.top_minus_bottom_quartile, color=PALETTE[0], height=0.55)
    ax.errorbar(q.top_minus_bottom_quartile, y, xerr=[q.top_minus_bottom_quartile - q.ci_low, q.ci_high - q.top_minus_bottom_quartile],
                fmt="none", ecolor="#333333", lw=1)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(q.signal)
    ax.set_xlabel("correctness: top quartile - bottom quartile (within question, mixed-correctness questions)")
    savefig(fig, fdir / f"quartile_lift_{ds}")


def fig_fusion(cfg, ds, fdir):
    import matplotlib.pyplot as plt

    mt = _read(_sub(cfg, "offline", ds) / "method_by_budget.csv")
    cvq = _read(_sub(cfg, "signals", ds) / "confidence_vision_per_question.csv")
    if mt is None or "tokenfusion_top" not in set(mt.method):
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in ("visconf_top", "tokenfusion_top", "visconf_rwc", "tokenfusion_rwc", "selfcert_rwc"):
        d = mt[mt.method == m].sort_values("N")
        s = style_for(m)
        ax[0].plot(d.N, d.accuracy, color=s["color"], ls=s["ls"], marker=s["marker"], ms=4, label=s["label"])
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("candidates N")
    ax[0].set_ylabel("accuracy")
    ax[0].legend(fontsize=7)
    if cvq is not None and len(cvq):
        ax[1].hist(cvq.mean_corr_c_v.dropna(), bins=25, color=PALETTE[0], alpha=0.85, label="calibrated confidence c_t")
        ax[1].hist(cvq.mean_corr_k_v.dropna(), bins=25, color=PALETTE[4], alpha=0.6, label="raw KL k_t")
        ax[1].axvline(0, color="#555555", lw=0.8)
        ax[1].set_xlabel("within-rollout Pearson corr. with visual engagement (mean per question)")
        ax[1].set_ylabel("questions")
        ax[1].legend()
    savefig(fig, fdir / f"fusion_and_confidence_vision_{ds}")


def fig_trajectories(cfg, ds, fdir):
    import matplotlib.pyplot as plt

    p = _sub(cfg, "signals", ds) / "trajectories.json"
    if not p.exists():
        return
    picks = load_json(p)
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(8, 2.3 * len(picks)), squeeze=False)
    for ax, pk in zip(axes[:, 0], picks):
        s = pk["series"]["visconf_top"]
        ax.plot(np.linspace(0, 1, len(s["confidence_c"])), s["confidence_c"], color=PALETTE[0], label="confidence c_t")
        ax.plot(np.linspace(0, 1, len(s["vision_v"])), s["vision_v"], color=PALETTE[1], label="visual engagement v_t")
        ax.set_title(f"{pk['rule']} | qid {pk['qid']} | VisConf-top rollout (correct={pk['visconf_correct']})", fontsize=8)
        ax.set_xlabel("relative position in rollout")
    axes[0, 0].legend(fontsize=7)
    savefig(fig, fdir / f"illustrative_trajectories_{ds}")


def fig_adaptive(cfg, ds, fdir):
    import matplotlib.pyplot as plt

    a = _sub(cfg, "adaptive", ds)
    grid, summ, stops = _read(a / "heldout_grid_all_settings.csv"), _read(a / "heldout_summary.csv"), _read(a / "heldout_stop_distribution.csv")
    if grid is None or summ is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    fixed = summ[summ.family == "fixed"].sort_values("mean_tokens")
    ax[0].plot(fixed.mean_tokens, fixed.accuracy, color=PALETTE[6], marker="o", label="fixed-N self-consistency")
    for fam, col, mk in (("agreement", PALETTE[0], "s"), ("quality_aware", PALETTE[1], "D")):
        g = grid[grid.family == fam]
        ax[0].scatter(g.mean_tokens, g.accuracy, color=col, marker=mk, s=18, alpha=0.6, label=f"{fam} (all grid settings)")
        sel = summ[summ.policy == fam]
        ax[0].scatter(sel.mean_tokens, sel.accuracy, color=col, marker=mk, s=110, edgecolor="#000000", linewidth=1.2, label=f"{fam} (dev-selected)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("mean generated tokens per question")
    ax[0].set_ylabel("held-out accuracy")
    ax[0].legend(fontsize=7)
    if stops is not None:
        pols = list(stops.policy.unique())
        bottom = np.zeros(len(pols))
        for i, n in enumerate(sorted(stops.stop_at_n.unique())):
            vals = np.array([stops[(stops.policy == p) & (stops.stop_at_n == n)].fraction.sum() for p in pols])
            ax[1].bar(pols, vals, bottom=bottom, color=PALETTE[i], label=f"stop at n={n}", width=0.55)
            bottom += vals
        ax[1].set_ylabel("fraction of (question, order) pairs")
        ax[1].legend(fontsize=7)
    fig.suptitle(f"{cfg['datasets'][ds]['pretty']}: adaptive sampling (held-out)")
    savefig(fig, fdir / f"adaptive_pareto_{ds}")


def fig_live(cfg, fdir):
    import matplotlib.pyplot as plt

    live = _sub(cfg, "live")
    summ = _read(live / "summary.csv")
    if summ is None:
        return
    main = summ[summ.strategy.isin(["single_3072", "parallel_3x1024_sc", "refine_1024_plus_2048"])]
    names = ["single 3072", "3 x 1024\nself-consistency", "1024 + 2048\nrefinement"]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    x = np.arange(len(main))
    ax[0].bar(x, main.accuracy, color=PALETTE[:3], width=0.6)
    ax[0].errorbar(x, main.accuracy, yerr=[main.accuracy - main.acc_ci_low, main.acc_ci_high - main.accuracy], fmt="none", ecolor="#333333", lw=1)
    ax[0].set_ylabel(f"accuracy (n={int(main.n_questions.iloc[0])} questions, 95% bootstrap CI)")
    ax[1].bar(x, main.mean_input_tokens, color=PALETTE[5], width=0.6, label="input tokens")
    ax[1].bar(x, main.mean_output_tokens, bottom=main.mean_input_tokens, color=PALETTE[3], width=0.6, label="output tokens")
    ax[1].set_ylabel("mean tokens per question")
    ax[1].legend(fontsize=7)
    ax[2].bar(x, main.mean_seconds, color=PALETTE[4], width=0.6)
    ax[2].set_ylabel("mean inference seconds per question")
    for a_ in ax[:3]:
        a_.set_xticks(x)
        a_.set_xticklabels(names, fontsize=7)
        a_.grid(axis="x", visible=False)
    tr = load_json(live / "transitions.json")["counts"] if (live / "transitions.json").exists() else None
    if tr:
        ks = list(tr)
        ax[3].bar(ks, [tr[k] for k in ks], color=[PALETTE[2], PALETTE[1], "#B8B8B8", "#B8B8B8"], width=0.6)
        ax[3].set_ylabel("questions")
        ax[3].tick_params(axis="x", labelrotation=20, labelsize=7)
        ax[3].set_title("refinement transitions vs. initial answer", fontsize=8)
    savefig(fig, fdir / "live_strategies")


def make_report(cfg, datasets: Sequence[str]) -> None:
    apply_style()
    fdir = ensure_dir(figures_root(cfg))
    for ds in datasets:
        for fn in (fig_accuracy, fig_decomposition, fig_groups, fig_quartiles, fig_adaptive):
            fn(cfg, ds, fdir)
        if ds == "mathvista":
            fig_fusion(cfg, ds, fdir)
            fig_trajectories(cfg, ds, fdir)
    fig_live(cfg, fdir)
    lines = [f"# Task 3 results ({'SMOKE TEST - not for reporting' if cfg.get('smoke_mode') else 'full run'})\n"]
    for ds in datasets:
        off, sig, ada = _sub(cfg, "offline", ds), _sub(cfg, "signals", ds), _sub(cfg, "adaptive", ds)
        lines.append(f"## {cfg['datasets'][ds]['pretty']}\n")
        meta = off / "data_meta.json"
        if meta.exists():
            m = load_json(meta)
            lines.append("**Data / scorer**\n\n```json\n" + __import__("json").dumps({k: m[k] for k in ("description", "n_dev", "n_eval", "n_live")}, indent=1, default=str) + "\n```\n")
        mt = _read(off / "method_by_budget.csv")
        if mt is not None:
            piv = mt.pivot(index="method", columns="N", values="accuracy")
            cov = mt[mt.method == mt.method.iloc[0]].set_index("N")["oracle_coverage"]
            piv.loc["oracle_coverage"] = cov
            lines.append("**Held-out accuracy by method and candidate budget N**\n\n" + df_to_markdown(piv.reset_index()))
            dec = _read(off / "failure_decomposition.csv")
            if dec is not None:
                lines.append("**Failure decomposition at N=32**\n\n" + df_to_markdown(dec[dec.N == cfg["budgets"][-1]].drop(columns=["dataset", "N"])))
        for name, title in (("paired_differences.csv", "Paired question-level bootstrap differences"), ):
            d = _read(off / name)
            if d is not None:
                d = d[d.N.isin([8, 32])][["method_a", "method_b", "N", "mean_diff", "ci_low", "ci_high", "excludes_zero"]]
                lines.append(f"**{title}** (N=8, 32)\n\n" + df_to_markdown(d))
        for name, title in (("quartile_summary.csv", "Within-question quartile lift (N=32)"), ("length_control.csv", "Length-only control (N=32)")):
            d = _read(sig / name)
            if d is not None:
                lines.append(f"**{title}**\n\n" + df_to_markdown(d))
        if (sig / "confidence_vision_summary.json").exists():
            lines.append("**Confidence-vision relationships and fusion identity**\n\n```json\n" + __import__("json").dumps(load_json(sig / "confidence_vision_summary.json"), indent=1) + "\n```\n")
        for name, title in (("heldout_summary.csv", "Adaptive vs fixed-N (held-out)"), ("heldout_diffs.csv", "Adaptive paired differences"),
                            ("heldout_wrong_stops.csv", "Incorrect early stops"), ("heldout_by_pool_fraction.csv", "Stopping vs fraction of correct rollouts in pool")):
            d = _read(ada / name)
            if d is not None:
                lines.append(f"**{title}**\n\n" + df_to_markdown(d))
        if (ada / "selected_settings.json").exists():
            lines.append("**Dev-selected (frozen) settings**\n\n```json\n" + __import__("json").dumps(load_json(ada / "selected_settings.json"), indent=1) + "\n```\n")
    live = _sub(cfg, "live")
    if (live / "summary.csv").exists():
        lines.append("## Live MATH subset: single vs parallel vs refinement\n")
        lines.append(df_to_markdown(pd.read_csv(live / "summary.csv")))
        lines.append("```json\n" + __import__("json").dumps(load_json(live / "transitions.json"), indent=1) + "\n```\n")
    write_text("\n".join(lines), results_root(cfg) / "summary.md")
