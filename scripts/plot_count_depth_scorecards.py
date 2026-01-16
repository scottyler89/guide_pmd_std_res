from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str  # higher|lower


def _rank(values: pd.Series, *, direction: str) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'")
    ascending = direction == "lower"
    # Best rank = 1.0
    r = v.rank(ascending=ascending, method="average")
    return r


def _rank_and_score(values: pd.Series, *, direction: str) -> tuple[pd.Series, pd.Series]:
    v = pd.to_numeric(values, errors="coerce")
    r = _rank(v, direction=direction)
    n = int(v.notna().sum())
    if n <= 1:
        score = pd.Series([np.nan] * v.shape[0], index=v.index, dtype=float)
    else:
        score = 1.0 - (r - 1.0) / max(1.0, float(n - 1))
        score = score.where(v.notna(), np.nan)
    return r, score


def _savefig(fig, out_path: str) -> None:
    # `tight_layout()` can fail for long tick labels; `bbox_inches="tight"` ensures nothing is cut off.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=r"^Tight layout not applied\..*")
        fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)


def _dot_scorecard(
    df: pd.DataFrame,
    *,
    pipeline_col: str,
    metric_specs: list[MetricSpec],
    out_path: str,
    title: str,
    sort_mode: str,
) -> pd.DataFrame:
    plt = _require_matplotlib()

    if df.empty:
        raise ValueError("empty dataframe (nothing to plot)")

    needed = {pipeline_col}
    needed |= {m.name for m in metric_specs}
    missing = sorted([c for c in needed if c not in df.columns])
    if missing:
        raise ValueError(f"missing required column(s): {missing}")

    if sort_mode not in {"avg", "worst"}:
        raise ValueError("sort_mode must be one of: avg, worst")

    data = df[[pipeline_col, *[m.name for m in metric_specs]]].copy()
    for m in metric_specs:
        data[m.name] = pd.to_numeric(data[m.name], errors="coerce")

    # Rank within metric (best rank=1). Score is normalized rank in [0,1] (worst→best).
    for m in metric_specs:
        r, s = _rank_and_score(data[m.name], direction=m.direction)
        data[f"{m.name}__rank"] = r
        data[f"{m.name}__score"] = s

    score_cols = [f"{m.name}__score" for m in metric_specs]
    data["avg_score"] = data[score_cols].mean(axis=1, skipna=True)
    data["worst_score"] = data[score_cols].min(axis=1, skipna=True)

    if sort_mode == "avg":
        data = data.sort_values(["avg_score", "worst_score", pipeline_col], ascending=[False, False, True], kind="mergesort").reset_index(
            drop=True
        )
    else:
        data = data.sort_values(["worst_score", "avg_score", pipeline_col], ascending=[False, False, True], kind="mergesort").reset_index(
            drop=True
        )

    pipelines = data[pipeline_col].astype(str).tolist()
    metrics = [m.name for m in metric_specs]
    n_base_metrics = len(metrics)

    x: list[int] = []
    y: list[int] = []
    c: list[float] = []
    s: list[float] = []

    for i, _p in enumerate(pipelines):
        for j, m in enumerate(metric_specs):
            sc = float(data.loc[i, f"{m.name}__score"]) if np.isfinite(data.loc[i, f"{m.name}__score"]) else np.nan
            if not np.isfinite(sc):
                continue
            x.append(j)
            y.append(i)
            c.append(sc)
            s.append(60.0 + 240.0 * sc)

        # Summary panel (always present): average and worst scores across metrics.
        avg_score = float(data.loc[i, "avg_score"]) if np.isfinite(data.loc[i, "avg_score"]) else np.nan
        worst_score = float(data.loc[i, "worst_score"]) if np.isfinite(data.loc[i, "worst_score"]) else np.nan
        if np.isfinite(avg_score):
            x.append(n_base_metrics + 0)
            y.append(i)
            c.append(avg_score)
            s.append(60.0 + 240.0 * avg_score)
        if np.isfinite(worst_score):
            x.append(n_base_metrics + 1)
            y.append(i)
            c.append(worst_score)
            s.append(60.0 + 240.0 * worst_score)

    fig_h = float(max(2.0, 0.28 * len(pipelines) + 1.0))
    fig_w = float(max(6.0, 0.9 * (len(metrics) + 2) + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    sc = ax.scatter(x, y, c=c, s=s, cmap="viridis", vmin=0.0, vmax=1.0, edgecolors="none")

    ax.set_title(title)
    ax.set_xticks(range(len(metrics) + 2))
    ax.set_xticklabels([*metrics, "avg", "worst"], rotation=45, ha="right")
    ax.set_yticks(range(len(pipelines)))
    ax.set_yticklabels(pipelines, fontsize=8)
    ax.set_xlim(-0.5, len(metrics) + 1.5)
    ax.set_ylim(-0.5, len(pipelines) - 0.5)
    ax.invert_yaxis()
    ax.grid(True, axis="x", lw=0.5, alpha=0.2)
    ax.grid(True, axis="y", lw=0.5, alpha=0.2)
    if n_base_metrics > 0:
        ax.axvline(n_base_metrics - 0.5, color="black", lw=1.0, alpha=0.25)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Normalized rank (worst→best)")
    _savefig(fig, out_path)
    plt.close(fig)

    return data


def _method_families(long_df: pd.DataFrame) -> dict[str, pd.Series]:
    def _series_or_constant(col: str, default: float) -> pd.Series:
        if col in long_df.columns:
            return long_df[col]
        return pd.Series([default] * long_df.shape[0], index=long_df.index, dtype=float)

    frac_signal = pd.to_numeric(long_df["frac_signal"], errors="coerce").fillna(0.0)
    tdm = pd.to_numeric(_series_or_constant("treatment_depth_multiplier", 1.0), errors="coerce").fillna(1.0)
    ot = pd.to_numeric(_series_or_constant("offtarget_guide_frac", 0.0), errors="coerce").fillna(0.0)
    nb = pd.to_numeric(_series_or_constant("nb_overdispersion", 0.0), errors="coerce").fillna(0.0)

    return {
        "null": frac_signal == 0.0,
        "signal": frac_signal > 0.0,
        "signal_depth_confounded": (frac_signal > 0.0) & (tdm != 1.0),
        "signal_offtarget": (frac_signal > 0.0) & (ot > 0.0),
        "signal_overdispersed": (frac_signal > 0.0) & (nb > 0.0),
    }


def _method_grid_avg_rank(
    long_df: pd.DataFrame,
    *,
    pipeline_col: str,
    out_path: str,
    title: str,
    sort_mode: str,
) -> pd.DataFrame:
    plt = _require_matplotlib()

    families = _method_families(long_df)

    if sort_mode not in {"avg", "worst"}:
        raise ValueError("sort_mode must be one of: avg, worst")

    # Metrics that are broadly comparable across method pipelines (scenario families).
    alpha = float(pd.to_numeric(long_df.get("alpha", 0.05), errors="coerce").dropna().iloc[0]) if "alpha" in long_df.columns else 0.05
    fdr_q = float(pd.to_numeric(long_df.get("fdr_q", 0.1), errors="coerce").dropna().iloc[0]) if "fdr_q" in long_df.columns else 0.1

    tmp = long_df.copy()
    tmp["null_lambda_gc_dev"] = np.abs(pd.to_numeric(tmp.get("null_lambda_gc", np.nan), errors="coerce") - 1.0)
    tmp["alpha_fpr_dev"] = np.abs(pd.to_numeric(tmp.get("alpha_fpr", np.nan), errors="coerce") - alpha)
    tmp["q_fdr_excess"] = np.maximum(0.0, pd.to_numeric(tmp.get("q_fdr", np.nan), errors="coerce") - fdr_q)

    scenario_metric_specs: list[tuple[str, MetricSpec, str]] = [
        ("null", MetricSpec("null_lambda_gc_dev", "lower"), "null | lambda_gc_dev"),
        ("null", MetricSpec("alpha_fpr_dev", "lower"), "null | alpha_fpr_dev"),
        ("null", MetricSpec("null_ks", "lower"), "null | ks"),
        ("signal", MetricSpec("q_fdr_excess", "lower"), "signal | q_fdr_excess"),
        ("signal", MetricSpec("q_tpr", "higher"), "signal | q_tpr"),
        ("signal", MetricSpec("q_balanced_accuracy", "higher"), "signal | q_balanced_accuracy"),
        ("signal", MetricSpec("q_mcc", "higher"), "signal | q_mcc"),
        ("signal", MetricSpec("roc_auc", "higher"), "signal | roc_auc"),
        ("signal", MetricSpec("average_precision", "higher"), "signal | average_precision"),
        ("signal_depth_confounded", MetricSpec("q_fdr_excess", "lower"), "depth_confounded | q_fdr_excess"),
        ("signal_depth_confounded", MetricSpec("q_tpr", "higher"), "depth_confounded | q_tpr"),
        ("signal_depth_confounded", MetricSpec("q_balanced_accuracy", "higher"), "depth_confounded | q_balanced_accuracy"),
        ("signal_depth_confounded", MetricSpec("q_mcc", "higher"), "depth_confounded | q_mcc"),
        ("signal_offtarget", MetricSpec("q_fdr_excess", "lower"), "offtarget | q_fdr_excess"),
        ("signal_offtarget", MetricSpec("q_tpr", "higher"), "offtarget | q_tpr"),
        ("signal_offtarget", MetricSpec("q_balanced_accuracy", "higher"), "offtarget | q_balanced_accuracy"),
        ("signal_offtarget", MetricSpec("q_mcc", "higher"), "offtarget | q_mcc"),
        ("signal_overdispersed", MetricSpec("q_fdr_excess", "lower"), "overdispersed | q_fdr_excess"),
        ("signal_overdispersed", MetricSpec("q_tpr", "higher"), "overdispersed | q_tpr"),
        ("signal_overdispersed", MetricSpec("q_balanced_accuracy", "higher"), "overdispersed | q_balanced_accuracy"),
        ("signal_overdispersed", MetricSpec("q_mcc", "higher"), "overdispersed | q_mcc"),
        ("all", MetricSpec("runtime_sec", "lower"), "all | runtime_sec"),
    ]

    pipelines = sorted(tmp[pipeline_col].dropna().astype(str).unique().tolist())
    cols = [label for _fam, _spec, label in scenario_metric_specs]

    grid = pd.DataFrame({pipeline_col: pipelines})
    for fam, spec, label in scenario_metric_specs:
        if fam != "all" and fam not in families:
            raise ValueError(f"unknown family: {fam}")
        fam_df = tmp if fam == "all" else tmp.loc[families[fam]].copy()
        if fam_df.empty or spec.name not in fam_df.columns:
            grid[f"{label}__value"] = np.nan
            grid[f"{label}__rank"] = np.nan
            grid[f"{label}__score"] = np.nan
            continue
        agg = fam_df.groupby(pipeline_col, dropna=False)[spec.name].mean()
        values = pd.to_numeric(agg.reindex(pipelines), errors="coerce")
        r, score = _rank_and_score(values, direction=spec.direction)
        grid[f"{label}__value"] = values.to_numpy(dtype=float)
        grid[f"{label}__rank"] = pd.to_numeric(r, errors="coerce").to_numpy(dtype=float)
        grid[f"{label}__score"] = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)

    score_cols = [f"{c}__score" for c in cols]
    score_m = grid[score_cols].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        grid["avg_score"] = np.nanmean(score_m, axis=1)
        grid["worst_score"] = np.nanmin(score_m, axis=1)

    if sort_mode == "avg":
        grid = grid.sort_values(["avg_score", "worst_score", pipeline_col], ascending=[False, False, True], kind="mergesort").reset_index(
            drop=True
        )
    else:
        grid = grid.sort_values(["worst_score", "avg_score", pipeline_col], ascending=[False, False, True], kind="mergesort").reset_index(
            drop=True
        )

    heat_cols = [*score_cols, "avg_score", "worst_score"]
    mat = grid[heat_cols].to_numpy(dtype=float)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#d9d9d9")
    mat = np.ma.masked_invalid(mat)

    fig_h = float(max(2.0, 0.26 * len(pipelines) + 1.0))
    fig_w = float(max(10.0, 0.55 * len(heat_cols) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(heat_cols)))
    ax.set_xticklabels([*cols, "avg", "worst"], rotation=45, ha="right")
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(grid[pipeline_col].astype(str).tolist(), fontsize=7)
    ax.axvline(len(score_cols) - 0.5, color="black", lw=1.0, alpha=0.25)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Normalized rank (worst→best)")
    _savefig(fig, out_path)
    plt.close(fig)

    return grid


def _plot_pareto_runtime_vs_tpr(
    df: pd.DataFrame,
    *,
    out_path: str,
    title: str,
    fdr_q: float,
) -> None:
    plt = _require_matplotlib()

    sub = df.copy()
    sub["runtime_sec"] = pd.to_numeric(sub["runtime_sec"], errors="coerce")
    sub["q_tpr"] = pd.to_numeric(sub["q_tpr"], errors="coerce")
    sub["q_fdr"] = pd.to_numeric(sub["q_fdr"], errors="coerce")
    sub = sub.loc[sub["runtime_sec"].notna() & sub["q_tpr"].notna() & sub["q_fdr"].notna()].copy()
    if sub.empty:
        return

    sub["q_fdr_excess"] = np.maximum(0.0, sub["q_fdr"] - float(fdr_q))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    sc = ax.scatter(
        sub["runtime_sec"],
        sub["q_tpr"],
        c=sub["q_fdr_excess"],
        cmap="viridis_r",
        s=80,
        edgecolors="none",
    )
    ax.set_xlabel("Runtime (sec)")
    ax.set_ylabel("TPR at FDR q")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True, lw=0.5, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("FDR excess over q (lower is better)")
    _savefig(fig, out_path)
    plt.close(fig)


def _pipeline_label(row: pd.Series, *, method: str) -> str:
    rm = str(row.get("response_mode", ""))
    norm = str(row.get("normalization_mode", ""))
    lr = str(row.get("logratio_mode", ""))
    depth = str(row.get("depth_covariate_mode", ""))
    batch = int(bool(row.get("include_batch_covariate", False)))

    parts = [method, f"rm={rm}", f"norm={norm}", f"lr={lr}", f"depth={depth}", f"batch={batch}"]
    if method.startswith("lmm_"):
        scope = str(row.get("lmm_scope", ""))
        cap = row.get("lmm_max_genes_per_focal_var", None)
        cap_s = "0" if cap in (None, "", 0) else str(int(cap))
        parts.append(f"scope={scope}")
        parts.append(f"cap={cap_s}")
    return " | ".join(parts)


def _extract_long(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    base_cols = [
        "response_mode",
        "normalization_mode",
        "logratio_mode",
        "n_reference_genes",
        "depth_covariate_mode",
        "include_batch_covariate",
        "alpha",
        "fdr_q",
        "frac_signal",
        "effect_sd",
        "treatment_depth_multiplier",
        "offtarget_guide_frac",
        "nb_overdispersion",
        "lmm_scope",
        "lmm_q_meta",
        "lmm_q_het",
        "lmm_audit_n",
        "lmm_max_genes_per_focal_var",
    ]
    base_cols = [c for c in base_cols if c in df.columns]

    method_specs = [
        ("meta", "runtime_meta"),
        ("stouffer", "runtime_stouffer"),
        ("lmm_lrt", "runtime_lmm"),
        ("lmm_wald", "runtime_lmm"),
    ]

    for method, runtime_col in method_specs:
        prefix = method
        required_any = [f"{prefix}_q_tpr", f"{prefix}_alpha_fpr", f"{prefix}_null_lambda_gc"]
        if not any(c in df.columns for c in required_any):
            continue

        for r in df.itertuples(index=False):
            row = pd.Series(r._asdict())
            out: dict[str, object] = {c: row.get(c) for c in base_cols}
            out["method"] = method
            out["pipeline"] = _pipeline_label(row, method=method)
            out["runtime_sec"] = row.get(runtime_col, np.nan)

            out["null_lambda_gc"] = row.get(f"{prefix}_null_lambda_gc", np.nan)
            out["null_ks"] = row.get(f"{prefix}_null_ks", np.nan)
            out["alpha_fpr"] = row.get(f"{prefix}_alpha_fpr", np.nan)
            out["q_tpr"] = row.get(f"{prefix}_q_tpr", np.nan)
            out["q_fdr"] = row.get(f"{prefix}_q_fdr", np.nan)
            out["roc_auc"] = row.get(f"{prefix}_roc_auc", np.nan)
            out["average_precision"] = row.get(f"{prefix}_average_precision", np.nan)

            out["alpha_accuracy"] = row.get(f"{prefix}_alpha_accuracy", np.nan)
            out["alpha_balanced_accuracy"] = row.get(f"{prefix}_alpha_balanced_accuracy", np.nan)
            out["alpha_mcc"] = row.get(f"{prefix}_alpha_mcc", np.nan)
            out["alpha_f1"] = row.get(f"{prefix}_alpha_f1", np.nan)
            out["q_accuracy"] = row.get(f"{prefix}_q_accuracy", np.nan)
            out["q_balanced_accuracy"] = row.get(f"{prefix}_q_balanced_accuracy", np.nan)
            out["q_mcc"] = row.get(f"{prefix}_q_mcc", np.nan)
            out["q_f1"] = row.get(f"{prefix}_q_f1", np.nan)

            if method in {"meta", "lmm_lrt", "lmm_wald"}:
                out["theta_rmse_signal"] = row.get(f"{prefix}_theta_rmse_signal", np.nan)
                out["theta_corr_signal"] = row.get(f"{prefix}_theta_corr_signal", np.nan)
                out["theta_sign_acc_signal"] = row.get(f"{prefix}_theta_sign_acc_signal", np.nan)
            else:
                out["theta_rmse_signal"] = np.nan
                out["theta_corr_signal"] = np.nan
                out["theta_sign_acc_signal"] = np.nan

            if method == "lmm_lrt":
                out["ok_frac"] = row.get("lmm_lrt_ok_frac", np.nan)
            elif method == "lmm_wald":
                out["ok_frac"] = row.get("lmm_wald_ok_frac", np.nan)
            else:
                out["ok_frac"] = np.nan

            rows.append(out)

    return pd.DataFrame(rows)


def _write_tsv(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scorecards and method-grid figures for the count-depth benchmark (local).")
    parser.add_argument("--grid-tsv", required=True, type=str, help="Path to count_depth_grid_summary.tsv (raw or aggregated).")
    parser.add_argument("--out-dir", required=True, type=str, help="Directory to write figures and summary TSVs.")
    parser.add_argument(
        "--max-pipelines",
        type=int,
        default=0,
        help="If >0, keep only the top-N pipelines by overall average rank within each plot (default: 0 = keep all).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.grid_tsv, sep="\t")
    os.makedirs(args.out_dir, exist_ok=True)

    long_df = _extract_long(df)
    if long_df.empty:
        raise ValueError("no method metrics found in grid TSV (expected meta/stouffer/lmm_* columns)")

    # Aggregate across all runs for each pipeline for each plot's subset.
    families = _method_families(long_df)
    plots_made = 0

    # Null scorecard: calibration + runtime (common across methods).
    alpha = float(pd.to_numeric(long_df.get("alpha", 0.05), errors="coerce").dropna().iloc[0]) if "alpha" in long_df.columns else 0.05
    null_df = long_df.loc[families["null"]].copy()
    if not null_df.empty:
        null_df["null_lambda_gc_dev"] = np.abs(pd.to_numeric(null_df["null_lambda_gc"], errors="coerce") - 1.0)
        null_df["alpha_fpr_dev"] = np.abs(pd.to_numeric(null_df["alpha_fpr"], errors="coerce") - alpha)
        null_agg = null_df.groupby("pipeline", dropna=False).mean(numeric_only=True).reset_index()
        null_specs = [
            MetricSpec("null_lambda_gc_dev", "lower"),
            MetricSpec("alpha_fpr_dev", "lower"),
            MetricSpec("null_ks", "lower"),
            MetricSpec("runtime_sec", "lower"),
        ]
        null_ranked_avg = _dot_scorecard(
            null_agg,
            pipeline_col="pipeline",
            metric_specs=null_specs,
            out_path=os.path.join(args.out_dir, "scorecard_null.png"),
            title="Benchmark scorecard (null runs) — calibration + runtime (sorted by avg)",
            sort_mode="avg",
        )
        null_ranked_worst = _dot_scorecard(
            null_agg,
            pipeline_col="pipeline",
            metric_specs=null_specs,
            out_path=os.path.join(args.out_dir, "scorecard_null__sort=worst.png"),
            title="Benchmark scorecard (null runs) — calibration + runtime (sorted by worst)",
            sort_mode="worst",
        )
        if int(args.max_pipelines) > 0:
            null_ranked_avg = null_ranked_avg.head(int(args.max_pipelines))
            null_ranked_worst = null_ranked_worst.head(int(args.max_pipelines))
        _write_tsv(os.path.join(args.out_dir, "scorecard_null.tsv"), null_ranked_avg)
        _write_tsv(os.path.join(args.out_dir, "scorecard_null__sort=worst.tsv"), null_ranked_worst)
        plots_made += 1

    # Signal scorecard: detection + runtime (common across methods).
    fdr_q = float(pd.to_numeric(long_df.get("fdr_q", 0.1), errors="coerce").dropna().iloc[0]) if "fdr_q" in long_df.columns else 0.1
    sig_df = long_df.loc[families["signal"]].copy()
    if not sig_df.empty:
        sig_df["q_fdr_excess"] = np.maximum(0.0, pd.to_numeric(sig_df["q_fdr"], errors="coerce") - fdr_q)
        sig_agg = sig_df.groupby("pipeline", dropna=False).mean(numeric_only=True).reset_index()
        sig_specs = [
            MetricSpec("q_fdr_excess", "lower"),
            MetricSpec("q_tpr", "higher"),
            MetricSpec("roc_auc", "higher"),
            MetricSpec("average_precision", "higher"),
            MetricSpec("runtime_sec", "lower"),
        ]
        sig_ranked_avg = _dot_scorecard(
            sig_agg,
            pipeline_col="pipeline",
            metric_specs=sig_specs,
            out_path=os.path.join(args.out_dir, "scorecard_signal.png"),
            title="Benchmark scorecard (signal runs) — detection + runtime (sorted by avg)",
            sort_mode="avg",
        )
        sig_ranked_worst = _dot_scorecard(
            sig_agg,
            pipeline_col="pipeline",
            metric_specs=sig_specs,
            out_path=os.path.join(args.out_dir, "scorecard_signal__sort=worst.png"),
            title="Benchmark scorecard (signal runs) — detection + runtime (sorted by worst)",
            sort_mode="worst",
        )
        if int(args.max_pipelines) > 0:
            sig_ranked_avg = sig_ranked_avg.head(int(args.max_pipelines))
            sig_ranked_worst = sig_ranked_worst.head(int(args.max_pipelines))
        _write_tsv(os.path.join(args.out_dir, "scorecard_signal.tsv"), sig_ranked_avg)
        _write_tsv(os.path.join(args.out_dir, "scorecard_signal__sort=worst.tsv"), sig_ranked_worst)
        keep = sig_ranked_avg["pipeline"].astype(str).tolist() if (not sig_ranked_avg.empty) else []
        sig_for_pareto = sig_agg.loc[sig_agg["pipeline"].astype(str).isin(keep)].copy() if keep else sig_agg
        _plot_pareto_runtime_vs_tpr(
            sig_for_pareto,
            out_path=os.path.join(args.out_dir, "pareto_runtime_vs_tpr.png"),
            title="Pareto: runtime vs power (signal runs)",
            fdr_q=float(fdr_q),
        )
        plots_made += 1

    # Confusion-matrix scorecard: signal runs only (quadrant-derived metrics + runtime).
    if not sig_df.empty:
        conf_df = sig_df.copy()
        conf_df["q_fdr_excess"] = np.maximum(0.0, pd.to_numeric(conf_df["q_fdr"], errors="coerce") - fdr_q)
        conf_agg = conf_df.groupby("pipeline", dropna=False).mean(numeric_only=True).reset_index()
        conf_specs = [
            MetricSpec("q_fdr_excess", "lower"),
            MetricSpec("q_balanced_accuracy", "higher"),
            MetricSpec("q_mcc", "higher"),
            MetricSpec("q_f1", "higher"),
            MetricSpec("runtime_sec", "lower"),
        ]
        # Only plot if at least one confusion-derived metric is finite anywhere.
        has_any = False
        for c in ["q_balanced_accuracy", "q_mcc", "q_f1"]:
            if c not in conf_agg.columns:
                continue
            v = pd.to_numeric(conf_agg[c], errors="coerce")
            if bool(np.isfinite(v.to_numpy(dtype=float)).any()):
                has_any = True
                break
        if has_any:
            conf_ranked_avg = _dot_scorecard(
                conf_agg,
                pipeline_col="pipeline",
                metric_specs=conf_specs,
                out_path=os.path.join(args.out_dir, "scorecard_signal_confusion.png"),
                title="Benchmark scorecard (signal runs) — confusion-matrix metrics + runtime (sorted by avg)",
                sort_mode="avg",
            )
            conf_ranked_worst = _dot_scorecard(
                conf_agg,
                pipeline_col="pipeline",
                metric_specs=conf_specs,
                out_path=os.path.join(args.out_dir, "scorecard_signal_confusion__sort=worst.png"),
                title="Benchmark scorecard (signal runs) — confusion-matrix metrics + runtime (sorted by worst)",
                sort_mode="worst",
            )
            if int(args.max_pipelines) > 0:
                conf_ranked_avg = conf_ranked_avg.head(int(args.max_pipelines))
                conf_ranked_worst = conf_ranked_worst.head(int(args.max_pipelines))
            _write_tsv(os.path.join(args.out_dir, "scorecard_signal_confusion.tsv"), conf_ranked_avg)
            _write_tsv(os.path.join(args.out_dir, "scorecard_signal_confusion__sort=worst.tsv"), conf_ranked_worst)
            plots_made += 1

    # Estimation scorecard: only pipelines with theta metrics.
    if not sig_df.empty:
        est_df = sig_df.copy()
        est_df = est_df.loc[est_df["theta_rmse_signal"].notna()].copy()
        if not est_df.empty:
            est_agg = est_df.groupby("pipeline", dropna=False).mean(numeric_only=True).reset_index()
            est_specs = [
                MetricSpec("theta_rmse_signal", "lower"),
                MetricSpec("theta_corr_signal", "higher"),
                MetricSpec("theta_sign_acc_signal", "higher"),
                MetricSpec("runtime_sec", "lower"),
            ]
            est_ranked_avg = _dot_scorecard(
                est_agg,
                pipeline_col="pipeline",
                metric_specs=est_specs,
                out_path=os.path.join(args.out_dir, "scorecard_signal_estimation.png"),
                title="Benchmark scorecard (signal runs) — effect estimation (theta) (sorted by avg)",
                sort_mode="avg",
            )
            est_ranked_worst = _dot_scorecard(
                est_agg,
                pipeline_col="pipeline",
                metric_specs=est_specs,
                out_path=os.path.join(args.out_dir, "scorecard_signal_estimation__sort=worst.png"),
                title="Benchmark scorecard (signal runs) — effect estimation (theta) (sorted by worst)",
                sort_mode="worst",
            )
            if int(args.max_pipelines) > 0:
                est_ranked_avg = est_ranked_avg.head(int(args.max_pipelines))
                est_ranked_worst = est_ranked_worst.head(int(args.max_pipelines))
            _write_tsv(os.path.join(args.out_dir, "scorecard_signal_estimation.tsv"), est_ranked_avg)
            _write_tsv(os.path.join(args.out_dir, "scorecard_signal_estimation__sort=worst.tsv"), est_ranked_worst)
            plots_made += 1

    # Method-grid: average rank by scenario family.
    grid_rank_avg = _method_grid_avg_rank(
        long_df,
        pipeline_col="pipeline",
        out_path=os.path.join(args.out_dir, "method_grid_avg_rank.png"),
        title="Benchmark pipeline grid — scenario metric ranks (sorted by avg)",
        sort_mode="avg",
    )
    grid_rank_worst = _method_grid_avg_rank(
        long_df,
        pipeline_col="pipeline",
        out_path=os.path.join(args.out_dir, "method_grid_avg_rank__sort=worst.png"),
        title="Benchmark pipeline grid — scenario metric ranks (sorted by worst)",
        sort_mode="worst",
    )
    if int(args.max_pipelines) > 0:
        grid_rank_avg = grid_rank_avg.head(int(args.max_pipelines))
        grid_rank_worst = grid_rank_worst.head(int(args.max_pipelines))
    _write_tsv(os.path.join(args.out_dir, "method_grid_avg_rank.tsv"), grid_rank_avg)
    _write_tsv(os.path.join(args.out_dir, "method_grid_avg_rank__sort=worst.tsv"), grid_rank_worst)
    plots_made += 1

    if plots_made == 0:
        raise ValueError("no figures were produced (check that the grid TSV contains usable method metrics)")


if __name__ == "__main__":
    main()
