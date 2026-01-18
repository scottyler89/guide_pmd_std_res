from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from count_depth_scenarios import attach_scenarios
from count_depth_scenarios import make_scenario_table
from count_depth_pipelines import pipeline_label


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


def _worst_case_across_scenarios(
    df: pd.DataFrame,
    *,
    pipeline_col: str,
    scenario_col: str,
    metric_specs: list[MetricSpec],
) -> pd.DataFrame:
    """
    Reduce a long table to one row per pipeline by taking the *worst-case* metric value across scenarios.

    This avoids misleading averages across adversarial scenarios. "Worst-case" is direction-aware:
      - direction=="lower": worst = max
      - direction=="higher": worst = min
    """

    if df.empty:
        return pd.DataFrame(columns=[pipeline_col, *[m.name for m in metric_specs]])

    needed = {pipeline_col, scenario_col} | {m.name for m in metric_specs}
    missing = sorted([c for c in needed if c not in df.columns])
    if missing:
        raise ValueError(f"missing required column(s): {missing}")

    sub = df[[pipeline_col, scenario_col, *[m.name for m in metric_specs]]].copy()
    for m in metric_specs:
        sub[m.name] = pd.to_numeric(sub[m.name], errors="coerce")

    # First aggregate within scenario (e.g., across seeds) to keep scenario as the unit of evaluation.
    per = sub.groupby([pipeline_col, scenario_col], dropna=False).mean(numeric_only=True).reset_index()

    pipelines = sorted(per[pipeline_col].dropna().astype(str).unique().tolist())
    out = pd.DataFrame({pipeline_col: pipelines})

    for m in metric_specs:
        g = per.groupby(pipeline_col, dropna=False)[m.name]
        if m.direction == "lower":
            worst = g.max()
        elif m.direction == "higher":
            worst = g.min()
        else:
            raise ValueError("direction must be 'higher' or 'lower'")
        out[m.name] = pd.to_numeric(worst.reindex(pipelines), errors="coerce").to_numpy(dtype=float)

    return out


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
    # Used only for the high-level null-vs-signal scorecards. Method-grid figures should not
    # aggregate across heterogeneous scenarios.
    frac_signal = pd.to_numeric(long_df.get("frac_signal", 0.0), errors="coerce").fillna(0.0)
    return {"null": frac_signal == 0.0, "signal": frac_signal > 0.0}


def _method_grid_avg_rank(
    long_df: pd.DataFrame,
    *,
    pipeline_col: str,
    out_path: str,
    title: str,
    sort_mode: str,
) -> pd.DataFrame:
    plt = _require_matplotlib()

    if sort_mode not in {"avg", "worst"}:
        raise ValueError("sort_mode must be one of: avg, worst")

    # Metrics to compare across pipelines; computed per scenario (no cross-scenario pooling).
    alpha = float(pd.to_numeric(long_df.get("alpha", 0.05), errors="coerce").dropna().iloc[0]) if "alpha" in long_df.columns else 0.05
    fdr_q = float(pd.to_numeric(long_df.get("fdr_q", 0.1), errors="coerce").dropna().iloc[0]) if "fdr_q" in long_df.columns else 0.1

    tmp = long_df.copy()
    tmp["null_lambda_gc_dev"] = np.abs(pd.to_numeric(tmp.get("null_lambda_gc", np.nan), errors="coerce") - 1.0)
    tmp["alpha_fpr_dev"] = np.abs(pd.to_numeric(tmp.get("alpha_fpr", np.nan), errors="coerce") - alpha)
    tmp["q_fdr_excess"] = np.maximum(0.0, pd.to_numeric(tmp.get("q_fdr", np.nan), errors="coerce") - fdr_q)

    scenarios = make_scenario_table(tmp)
    scenario_cols = [c for c in scenarios.columns if c not in {"scenario", "scenario_id", "is_null"}]
    tmp = tmp.merge(scenarios, on=scenario_cols, how="left", validate="many_to_one")

    null_metrics = [
        (MetricSpec("null_lambda_gc_dev", "lower"), "lambda_gc_dev"),
        (MetricSpec("alpha_fpr_dev", "lower"), "alpha_fpr_dev"),
        (MetricSpec("null_ks", "lower"), "ks"),
        (MetricSpec("runtime_sec", "lower"), "runtime_sec"),
    ]
    signal_metrics = [
        (MetricSpec("q_fdr_excess", "lower"), "q_fdr_excess"),
        (MetricSpec("q_tpr", "higher"), "q_tpr"),
        (MetricSpec("q_balanced_accuracy", "higher"), "q_balacc"),
        (MetricSpec("q_mcc", "higher"), "q_mcc"),
        (MetricSpec("roc_auc", "higher"), "roc_auc"),
        (MetricSpec("average_precision", "higher"), "pr_auc"),
        (MetricSpec("runtime_sec", "lower"), "runtime_sec"),
    ]

    scenario_metric_specs: list[tuple[str, MetricSpec, str]] = []
    for s in scenarios.itertuples(index=False):
        label = str(getattr(s, "scenario"))
        is_null = bool(getattr(s, "is_null"))
        metrics = null_metrics if is_null else signal_metrics
        for spec, short_name in metrics:
            scenario_metric_specs.append((label, spec, f"{label} | {short_name}"))

    pipelines = sorted(tmp[pipeline_col].dropna().astype(str).unique().tolist())
    cols = [label for _fam, _spec, label in scenario_metric_specs]

    grid = pd.DataFrame({pipeline_col: pipelines})
    out_cols: dict[str, np.ndarray] = {}
    for scenario_label, spec, label in scenario_metric_specs:
        fam_df = tmp.loc[tmp["scenario"].astype(str) == str(scenario_label)].copy()
        if fam_df.empty or spec.name not in fam_df.columns:
            out_cols[f"{label}__value"] = np.full(len(pipelines), np.nan, dtype=float)
            out_cols[f"{label}__rank"] = np.full(len(pipelines), np.nan, dtype=float)
            out_cols[f"{label}__score"] = np.full(len(pipelines), np.nan, dtype=float)
            continue
        agg = fam_df.groupby(pipeline_col, dropna=False)[spec.name].mean()
        values = pd.to_numeric(agg.reindex(pipelines), errors="coerce")
        r, score = _rank_and_score(values, direction=spec.direction)
        out_cols[f"{label}__value"] = values.to_numpy(dtype=float)
        out_cols[f"{label}__rank"] = pd.to_numeric(r, errors="coerce").to_numpy(dtype=float)
        out_cols[f"{label}__score"] = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)

    if out_cols:
        grid = pd.concat([grid, pd.DataFrame(out_cols)], axis=1)

    is_null_by_scenario = {str(getattr(s, "scenario")): bool(getattr(s, "is_null")) for s in scenarios.itertuples(index=False)}
    null_cols = [label for scenario_label, _spec, label in scenario_metric_specs if bool(is_null_by_scenario.get(str(scenario_label), False))]
    signal_cols = [label for scenario_label, _spec, label in scenario_metric_specs if not bool(is_null_by_scenario.get(str(scenario_label), False))]

    score_cols = [f"{c}__score" for c in cols]
    null_score_cols = [f"{c}__score" for c in null_cols]
    signal_score_cols = [f"{c}__score" for c in signal_cols]

    def _summarize(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Returns: coverage_n, coverage_frac, avg_score, worst_score (with NaNs penalized to 0).
        n_cols = int(scores.shape[1])
        if n_cols == 0:
            nan = np.full(scores.shape[0], np.nan, dtype=float)
            return nan, nan, nan, nan
        mask = np.isfinite(scores)
        cov_n = np.sum(mask, axis=1).astype(float)
        cov_f = cov_n / float(n_cols)
        filled = np.where(mask, scores, 0.0)
        with np.errstate(all="ignore"):
            avg = np.mean(filled, axis=1).astype(float)
            worst = np.min(filled, axis=1).astype(float)
        return cov_n, cov_f, avg, worst

    score_m = grid[score_cols].to_numpy(dtype=float) if score_cols else np.empty((len(pipelines), 0), dtype=float)
    null_m = grid[null_score_cols].to_numpy(dtype=float) if null_score_cols else np.empty((len(pipelines), 0), dtype=float)
    sig_m = grid[signal_score_cols].to_numpy(dtype=float) if signal_score_cols else np.empty((len(pipelines), 0), dtype=float)

    with np.errstate(all="ignore"):
        cov_n, cov_f, avg_all, worst_all = _summarize(score_m)
        cov_n_null, cov_f_null, avg_null, worst_null = _summarize(null_m)
        cov_n_sig, cov_f_sig, avg_sig, worst_sig = _summarize(sig_m)

        grid["coverage_n"] = cov_n
        grid["coverage_frac"] = cov_f
        grid["coverage_n_null"] = cov_n_null
        grid["coverage_frac_null"] = cov_f_null
        grid["coverage_n_signal"] = cov_n_sig
        grid["coverage_frac_signal"] = cov_f_sig

        grid["avg_score_null"] = avg_null
        grid["worst_score_null"] = worst_null
        grid["avg_score_signal"] = avg_sig
        grid["worst_score_signal"] = worst_sig

        # Single-summary columns that do NOT allow null-vs-signal cancellation: use the weaker domain.
        grid["avg_score_min_domain"] = np.fmin(avg_null, avg_sig)
        grid["worst_score_min_domain"] = np.fmin(worst_null, worst_sig)
        grid["coverage_frac_min_domain"] = np.fmin(cov_f_null, cov_f_sig)

    if sort_mode == "avg":
        grid = grid.sort_values(
            ["coverage_frac_min_domain", "avg_score_min_domain", "worst_score_min_domain", pipeline_col],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        grid = grid.sort_values(
            ["coverage_frac_min_domain", "worst_score_min_domain", "avg_score_min_domain", pipeline_col],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    heat_cols = [*score_cols, "avg_score_min_domain", "worst_score_min_domain"]
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
    ax.set_xticklabels([*cols, "avg(min-domain)", "worst(min-domain)"], rotation=45, ha="right")
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


def _extract_long(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    base_cols = [
        # Pipeline knobs
        "response_mode",
        "normalization_mode",
        "logratio_mode",
        "n_reference_genes",
        "depth_covariate_mode",
        "include_batch_covariate",
        # Scenario knobs (simulation / design)
        "n_genes",
        "guides_per_gene",
        "n_control",
        "n_treatment",
        "depth_log_sd",
        "n_batches",
        "batch_confounding_strength",
        "batch_depth_log_sd",
        "guide_lambda_log_mean",
        "guide_lambda_log_sd",
        "gene_lambda_log_sd",
        "alpha",
        "fdr_q",
        "frac_signal",
        "effect_sd",
        "guide_slope_sd",
        "treatment_depth_multiplier",
        "offtarget_guide_frac",
        "offtarget_slope_sd",
        "nb_overdispersion",
        # LMM knobs (pipeline)
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
            out["pipeline"] = pipeline_label(row, method=method)
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

    # Attach explicit scenario labels (SSoT) so we never average across heterogeneous scenarios by accident.
    long_df_sc = attach_scenarios(long_df)
    plots_made = 0

    # Null scorecard: calibration + runtime (common across methods).
    alpha = float(pd.to_numeric(long_df_sc.get("alpha", 0.05), errors="coerce").dropna().iloc[0]) if "alpha" in long_df_sc.columns else 0.05
    null_df = long_df_sc.loc[long_df_sc["is_null"].astype(bool)].copy()
    if not null_df.empty:
        null_df["null_lambda_gc_dev"] = np.abs(pd.to_numeric(null_df["null_lambda_gc"], errors="coerce") - 1.0)
        null_df["alpha_fpr_dev"] = np.abs(pd.to_numeric(null_df["alpha_fpr"], errors="coerce") - alpha)
        null_specs = [
            MetricSpec("null_lambda_gc_dev", "lower"),
            MetricSpec("alpha_fpr_dev", "lower"),
            MetricSpec("null_ks", "lower"),
            MetricSpec("runtime_sec", "lower"),
        ]
        null_agg = _worst_case_across_scenarios(
            null_df,
            pipeline_col="pipeline",
            scenario_col="scenario_id",
            metric_specs=null_specs,
        )
        null_ranked_avg = _dot_scorecard(
            null_agg,
            pipeline_col="pipeline",
            metric_specs=null_specs,
            out_path=os.path.join(args.out_dir, "scorecard_null.png"),
            title="Benchmark scorecard (null scenarios; worst-case) — calibration + runtime (sorted by avg)",
            sort_mode="avg",
        )
        null_ranked_worst = _dot_scorecard(
            null_agg,
            pipeline_col="pipeline",
            metric_specs=null_specs,
            out_path=os.path.join(args.out_dir, "scorecard_null__sort=worst.png"),
            title="Benchmark scorecard (null scenarios; worst-case) — calibration + runtime (sorted by worst)",
            sort_mode="worst",
        )
        if int(args.max_pipelines) > 0:
            null_ranked_avg = null_ranked_avg.head(int(args.max_pipelines))
            null_ranked_worst = null_ranked_worst.head(int(args.max_pipelines))
        _write_tsv(os.path.join(args.out_dir, "scorecard_null.tsv"), null_ranked_avg)
        _write_tsv(os.path.join(args.out_dir, "scorecard_null__sort=worst.tsv"), null_ranked_worst)
        plots_made += 1

    # Signal scorecard: detection + runtime (common across methods).
    fdr_q = float(pd.to_numeric(long_df_sc.get("fdr_q", 0.1), errors="coerce").dropna().iloc[0]) if "fdr_q" in long_df_sc.columns else 0.1
    sig_df = long_df_sc.loc[~long_df_sc["is_null"].astype(bool)].copy()
    if not sig_df.empty:
        sig_df["q_fdr_excess"] = np.maximum(0.0, pd.to_numeric(sig_df["q_fdr"], errors="coerce") - fdr_q)
        sig_specs = [
            MetricSpec("q_fdr_excess", "lower"),
            MetricSpec("q_tpr", "higher"),
            MetricSpec("roc_auc", "higher"),
            MetricSpec("average_precision", "higher"),
            MetricSpec("runtime_sec", "lower"),
        ]
        sig_agg = _worst_case_across_scenarios(
            sig_df,
            pipeline_col="pipeline",
            scenario_col="scenario_id",
            metric_specs=sig_specs,
        )
        sig_ranked_avg = _dot_scorecard(
            sig_agg,
            pipeline_col="pipeline",
            metric_specs=sig_specs,
            out_path=os.path.join(args.out_dir, "scorecard_signal.png"),
            title="Benchmark scorecard (signal scenarios; worst-case) — detection + runtime (sorted by avg)",
            sort_mode="avg",
        )
        sig_ranked_worst = _dot_scorecard(
            sig_agg,
            pipeline_col="pipeline",
            metric_specs=sig_specs,
            out_path=os.path.join(args.out_dir, "scorecard_signal__sort=worst.png"),
            title="Benchmark scorecard (signal scenarios; worst-case) — detection + runtime (sorted by worst)",
            sort_mode="worst",
        )
        if int(args.max_pipelines) > 0:
            sig_ranked_avg = sig_ranked_avg.head(int(args.max_pipelines))
            sig_ranked_worst = sig_ranked_worst.head(int(args.max_pipelines))
        _write_tsv(os.path.join(args.out_dir, "scorecard_signal.tsv"), sig_ranked_avg)
        _write_tsv(os.path.join(args.out_dir, "scorecard_signal__sort=worst.tsv"), sig_ranked_worst)
        keep = sig_ranked_avg["pipeline"].astype(str).tolist() if (not sig_ranked_avg.empty) else []
        pareto_specs = [
            MetricSpec("runtime_sec", "lower"),
            MetricSpec("q_tpr", "higher"),
            MetricSpec("q_fdr", "lower"),
        ]
        pareto_agg = _worst_case_across_scenarios(
            sig_df,
            pipeline_col="pipeline",
            scenario_col="scenario_id",
            metric_specs=pareto_specs,
        )
        sig_for_pareto = pareto_agg.loc[pareto_agg["pipeline"].astype(str).isin(keep)].copy() if keep else pareto_agg
        _plot_pareto_runtime_vs_tpr(
            sig_for_pareto,
            out_path=os.path.join(args.out_dir, "pareto_runtime_vs_tpr.png"),
            title="Pareto: runtime vs power (signal scenarios; worst-case)",
            fdr_q=float(fdr_q),
        )
        plots_made += 1

    # Confusion-matrix scorecard: signal runs only (quadrant-derived metrics + runtime).
    if not sig_df.empty:
        conf_df = sig_df.copy()
        conf_df["q_fdr_excess"] = np.maximum(0.0, pd.to_numeric(conf_df["q_fdr"], errors="coerce") - fdr_q)
        conf_specs = [
            MetricSpec("q_fdr_excess", "lower"),
            MetricSpec("q_balanced_accuracy", "higher"),
            MetricSpec("q_mcc", "higher"),
            MetricSpec("q_f1", "higher"),
            MetricSpec("runtime_sec", "lower"),
        ]
        conf_agg = _worst_case_across_scenarios(
            conf_df,
            pipeline_col="pipeline",
            scenario_col="scenario_id",
            metric_specs=conf_specs,
        )
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
                title="Benchmark scorecard (signal scenarios; worst-case) — confusion-matrix metrics + runtime (sorted by avg)",
                sort_mode="avg",
            )
            conf_ranked_worst = _dot_scorecard(
                conf_agg,
                pipeline_col="pipeline",
                metric_specs=conf_specs,
                out_path=os.path.join(args.out_dir, "scorecard_signal_confusion__sort=worst.png"),
                title="Benchmark scorecard (signal scenarios; worst-case) — confusion-matrix metrics + runtime (sorted by worst)",
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
            est_specs = [
                MetricSpec("theta_rmse_signal", "lower"),
                MetricSpec("theta_corr_signal", "higher"),
                MetricSpec("theta_sign_acc_signal", "higher"),
                MetricSpec("runtime_sec", "lower"),
            ]
            est_agg = _worst_case_across_scenarios(
                est_df,
                pipeline_col="pipeline",
                scenario_col="scenario_id",
                metric_specs=est_specs,
            )
            est_ranked_avg = _dot_scorecard(
                est_agg,
                pipeline_col="pipeline",
                metric_specs=est_specs,
                out_path=os.path.join(args.out_dir, "scorecard_signal_estimation.png"),
                title="Benchmark scorecard (signal scenarios; worst-case) — effect estimation (theta) (sorted by avg)",
                sort_mode="avg",
            )
            est_ranked_worst = _dot_scorecard(
                est_agg,
                pipeline_col="pipeline",
                metric_specs=est_specs,
                out_path=os.path.join(args.out_dir, "scorecard_signal_estimation__sort=worst.png"),
                title="Benchmark scorecard (signal scenarios; worst-case) — effect estimation (theta) (sorted by worst)",
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
