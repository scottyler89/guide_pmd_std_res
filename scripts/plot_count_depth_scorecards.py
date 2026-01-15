from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

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


def _dot_scorecard(
    df: pd.DataFrame,
    *,
    pipeline_col: str,
    metric_specs: list[MetricSpec],
    out_path: str,
    title: str,
) -> pd.DataFrame:
    plt = _require_matplotlib()

    if df.empty:
        raise ValueError("empty dataframe (nothing to plot)")

    needed = {pipeline_col}
    needed |= {m.name for m in metric_specs}
    missing = sorted([c for c in needed if c not in df.columns])
    if missing:
        raise ValueError(f"missing required column(s): {missing}")

    data = df[[pipeline_col, *[m.name for m in metric_specs]]].copy()
    for m in metric_specs:
        data[m.name] = pd.to_numeric(data[m.name], errors="coerce")

    # Rank within metric (best rank=1). Average rank gives a stable ordering for y-axis.
    for m in metric_specs:
        data[f"{m.name}__rank"] = _rank(data[m.name], direction=m.direction)
    rank_cols = [f"{m.name}__rank" for m in metric_specs]
    data["avg_rank"] = data[rank_cols].mean(axis=1, skipna=True)
    data = data.sort_values(["avg_rank", pipeline_col], kind="mergesort").reset_index(drop=True)

    pipelines = data[pipeline_col].astype(str).tolist()
    metrics = [m.name for m in metric_specs]

    x: list[int] = []
    y: list[int] = []
    c: list[float] = []
    s: list[float] = []

    for i, _p in enumerate(pipelines):
        for j, m in enumerate(metric_specs):
            r = float(data.loc[i, f"{m.name}__rank"]) if np.isfinite(data.loc[i, f"{m.name}__rank"]) else np.nan
            if not np.isfinite(r):
                continue
            n = float(len(pipelines))
            rank_norm = 1.0 if n <= 1 else 1.0 - (r - 1.0) / max(1.0, n - 1.0)
            x.append(j)
            y.append(i)
            c.append(rank_norm)
            s.append(60.0 + 240.0 * rank_norm)

    fig_h = float(max(2.0, 0.28 * len(pipelines) + 1.0))
    fig_w = float(max(6.0, 0.9 * len(metrics) + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    sc = ax.scatter(x, y, c=c, s=s, cmap="viridis", vmin=0.0, vmax=1.0, edgecolors="none")

    ax.set_title(title)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(pipelines)))
    ax.set_yticklabels(pipelines, fontsize=8)
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylim(-0.5, len(pipelines) - 0.5)
    ax.invert_yaxis()
    ax.grid(True, axis="x", lw=0.5, alpha=0.2)
    ax.grid(True, axis="y", lw=0.5, alpha=0.2)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Rank (best→worst)")
    fig.tight_layout()
    fig.savefig(out_path)
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
) -> pd.DataFrame:
    plt = _require_matplotlib()

    families = _method_families(long_df)

    # Metrics that are broadly comparable across method pipelines.
    alpha = float(pd.to_numeric(long_df.get("alpha", 0.05), errors="coerce").dropna().iloc[0]) if "alpha" in long_df.columns else 0.05
    fdr_q = float(pd.to_numeric(long_df.get("fdr_q", 0.1), errors="coerce").dropna().iloc[0]) if "fdr_q" in long_df.columns else 0.1

    tmp = long_df.copy()
    tmp["null_lambda_gc_dev"] = np.abs(pd.to_numeric(tmp.get("null_lambda_gc", np.nan), errors="coerce") - 1.0)
    tmp["alpha_fpr_dev"] = np.abs(pd.to_numeric(tmp.get("alpha_fpr", np.nan), errors="coerce") - alpha)
    tmp["q_fdr_excess"] = np.maximum(0.0, pd.to_numeric(tmp.get("q_fdr", np.nan), errors="coerce") - fdr_q)

    family_specs: dict[str, list[MetricSpec]] = {
        "null": [
            MetricSpec("null_lambda_gc_dev", "lower"),
            MetricSpec("alpha_fpr_dev", "lower"),
            MetricSpec("null_ks", "lower"),
        ],
        "signal": [
            MetricSpec("q_fdr_excess", "lower"),
            MetricSpec("q_tpr", "higher"),
            MetricSpec("roc_auc", "higher"),
            MetricSpec("average_precision", "higher"),
        ],
        "signal_depth_confounded": [
            MetricSpec("q_fdr_excess", "lower"),
            MetricSpec("q_tpr", "higher"),
        ],
        "signal_offtarget": [
            MetricSpec("q_fdr_excess", "lower"),
            MetricSpec("q_tpr", "higher"),
        ],
        "signal_overdispersed": [
            MetricSpec("q_fdr_excess", "lower"),
            MetricSpec("q_tpr", "higher"),
        ],
    }

    pipelines = sorted(tmp[pipeline_col].dropna().astype(str).unique().tolist())
    fam_names = list(families.keys())
    mat = np.full((len(pipelines), len(fam_names)), np.nan, dtype=float)

    out_rows: list[dict[str, object]] = []
    for j, fam in enumerate(fam_names):
        fam_mask = families[fam]
        fam_df = tmp.loc[fam_mask].copy()
        if fam_df.empty:
            continue
        agg = fam_df.groupby(pipeline_col, dropna=False).mean(numeric_only=True).reset_index()
        agg[pipeline_col] = agg[pipeline_col].astype(str)

        specs = family_specs[fam]
        # Build average rank across the family's metric set.
        ranks: list[pd.Series] = []
        for spec in specs:
            if spec.name not in agg.columns:
                continue
            ranks.append(_rank(agg[spec.name], direction=spec.direction))
        if not ranks:
            continue
        avg_rank = pd.concat(ranks, axis=1).mean(axis=1, skipna=True)
        agg["avg_rank"] = avg_rank

        lookup = agg.set_index(pipeline_col)["avg_rank"].to_dict()
        for i, p in enumerate(pipelines):
            mat[i, j] = float(lookup.get(p, np.nan))
            out_rows.append({"family": fam, pipeline_col: p, "avg_rank": lookup.get(p, np.nan)})

    fig_h = float(max(2.0, 0.28 * len(pipelines) + 1.0))
    fig_w = float(max(7.0, 1.4 * len(fam_names) + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis_r")
    ax.set_title(title)
    ax.set_xticks(range(len(fam_names)))
    ax.set_xticklabels(fam_names, rotation=30, ha="right")
    ax.set_yticks(range(len(pipelines)))
    ax.set_yticklabels(pipelines, fontsize=8)
    ax.invert_yaxis()
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Average rank (lower is better)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return pd.DataFrame(out_rows)


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
    fig.tight_layout()
    fig.savefig(out_path)
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
        null_ranked = _dot_scorecard(
            null_agg,
            pipeline_col="pipeline",
            metric_specs=null_specs,
            out_path=os.path.join(args.out_dir, "scorecard_null.png"),
            title="Benchmark scorecard (null runs) — calibration + runtime",
        )
        if int(args.max_pipelines) > 0:
            null_ranked = null_ranked.head(int(args.max_pipelines))
        _write_tsv(os.path.join(args.out_dir, "scorecard_null.tsv"), null_ranked)
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
        sig_ranked = _dot_scorecard(
            sig_agg,
            pipeline_col="pipeline",
            metric_specs=sig_specs,
            out_path=os.path.join(args.out_dir, "scorecard_signal.png"),
            title="Benchmark scorecard (signal runs) — detection + runtime",
        )
        if int(args.max_pipelines) > 0:
            sig_ranked = sig_ranked.head(int(args.max_pipelines))
        _write_tsv(os.path.join(args.out_dir, "scorecard_signal.tsv"), sig_ranked)
        keep = sig_ranked["pipeline"].astype(str).tolist() if (not sig_ranked.empty) else []
        sig_for_pareto = sig_agg.loc[sig_agg["pipeline"].astype(str).isin(keep)].copy() if keep else sig_agg
        _plot_pareto_runtime_vs_tpr(
            sig_for_pareto,
            out_path=os.path.join(args.out_dir, "pareto_runtime_vs_tpr.png"),
            title="Pareto: runtime vs power (signal runs)",
            fdr_q=float(fdr_q),
        )
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
            est_ranked = _dot_scorecard(
                est_agg,
                pipeline_col="pipeline",
                metric_specs=est_specs,
                out_path=os.path.join(args.out_dir, "scorecard_signal_estimation.png"),
                title="Benchmark scorecard (signal runs) — effect estimation (theta)",
            )
            if int(args.max_pipelines) > 0:
                est_ranked = est_ranked.head(int(args.max_pipelines))
            _write_tsv(os.path.join(args.out_dir, "scorecard_signal_estimation.tsv"), est_ranked)
            plots_made += 1

    # Method-grid: average rank by scenario family.
    grid_rank = _method_grid_avg_rank(
        long_df,
        pipeline_col="pipeline",
        out_path=os.path.join(args.out_dir, "method_grid_avg_rank.png"),
        title="Benchmark method grid — average rank by scenario family",
    )
    _write_tsv(os.path.join(args.out_dir, "method_grid_avg_rank.tsv"), grid_rank)
    plots_made += 1

    if plots_made == 0:
        raise ValueError("no figures were produced (check that the grid TSV contains usable method metrics)")


if __name__ == "__main__":
    main()
