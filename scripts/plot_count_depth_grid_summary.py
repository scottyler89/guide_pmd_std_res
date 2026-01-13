from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _sorted_unique(df: pd.DataFrame, col: str) -> list[Any]:
    if col not in df.columns:
        return []
    vals = df[col].dropna().unique().tolist()
    try:
        return sorted(vals)
    except TypeError:
        return vals


def _plot_metric_grid(
    df: pd.DataFrame,
    *,
    metric_col: str,
    x_col: str,
    out_path: str,
    title: str,
    y_label: str,
    hline: float | None = None,
) -> None:
    plt = _require_matplotlib()

    if metric_col not in df.columns:
        raise ValueError(f"missing column: {metric_col}")
    if x_col not in df.columns:
        raise ValueError(f"missing column: {x_col}")

    depth_vals = _sorted_unique(df, "include_depth_covariate") or [None]
    batch_vals = _sorted_unique(df, "include_batch_covariate") or [None]

    nrows = len(depth_vals)
    ncols = len(batch_vals)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), dpi=150, sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape((nrows, ncols))

    for i, depth_cov in enumerate(depth_vals):
        for j, batch_cov in enumerate(batch_vals):
            ax = axes[i, j]
            sub = df
            if depth_cov is not None and "include_depth_covariate" in sub.columns:
                sub = sub.loc[sub["include_depth_covariate"] == depth_cov]
            if batch_cov is not None and "include_batch_covariate" in sub.columns:
                sub = sub.loc[sub["include_batch_covariate"] == batch_cov]

            if sub.empty:
                ax.set_axis_off()
                continue

            agg = (
                sub.groupby(x_col, dropna=False)[metric_col]
                .agg(["mean", "count"])
                .reset_index()
                .sort_values(x_col)
            )
            ax.plot(agg[x_col], agg["mean"], marker="o", lw=1)
            if hline is not None:
                ax.axhline(float(hline), color="black", lw=1, alpha=0.5)

            sub_title = []
            if depth_cov is not None:
                sub_title.append(f"depthcov={int(bool(depth_cov))}")
            if batch_cov is not None:
                sub_title.append(f"batchcov={int(bool(batch_cov))}")
            ax.set_title(", ".join(sub_title) if sub_title else "all")

    fig.suptitle(title)
    for ax in axes[-1, :]:
        ax.set_xlabel(x_col)
    for ax in axes[:, 0]:
        ax.set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot summary figures from count_depth_grid_summary.tsv.")
    parser.add_argument("--grid-tsv", required=True, type=str, help="Path to count_depth_grid_summary.tsv.")
    parser.add_argument("--out-dir", required=True, type=str, help="Directory to write figures.")
    args = parser.parse_args()

    df = pd.read_csv(args.grid_tsv, sep="\t")
    os.makedirs(args.out_dir, exist_ok=True)

    null_df = df.loc[df.get("frac_signal", 0.0) == 0.0].copy()
    sig_df = df.loc[df.get("frac_signal", 0.0) > 0.0].copy()

    x_tdm = "treatment_depth_multiplier" if "treatment_depth_multiplier" in df.columns else None
    x_esd = "effect_sd" if "effect_sd" in df.columns else None
    x_ngenes = "n_genes" if "n_genes" in df.columns else None

    plots_made = 0

    for prefix in ["meta", "lmm_lrt", "lmm_wald"]:
        if not null_df.empty and x_tdm is not None:
            metric = f"{prefix}_null_lambda_gc"
            if metric in null_df.columns:
                _plot_metric_grid(
                    null_df,
                    metric_col=metric,
                    x_col=x_tdm,
                    out_path=os.path.join(args.out_dir, f"null_lambda_gc__{prefix}.png"),
                    title=f"Null calibration (lambda_gc) — {prefix}",
                    y_label="lambda_gc",
                    hline=1.0,
                )
                plots_made += 1

            metric = f"{prefix}_alpha_fpr"
            if metric in null_df.columns:
                alpha = float(df["alpha"].iloc[0]) if "alpha" in df.columns and not df["alpha"].empty else None
                _plot_metric_grid(
                    null_df,
                    metric_col=metric,
                    x_col=x_tdm,
                    out_path=os.path.join(args.out_dir, f"null_fpr_alpha__{prefix}.png"),
                    title=f"Null FPR at alpha — {prefix}",
                    y_label="FPR",
                    hline=alpha,
                )
                plots_made += 1

        if not sig_df.empty and x_esd is not None:
            metric = f"{prefix}_q_tpr"
            if metric in sig_df.columns:
                _plot_metric_grid(
                    sig_df,
                    metric_col=metric,
                    x_col=x_esd,
                    out_path=os.path.join(args.out_dir, f"signal_tpr_fdrq__{prefix}.png"),
                    title=f"Power (TPR at FDR q) — {prefix}",
                    y_label="TPR",
                    hline=None,
                )
                plots_made += 1

    # Simple runtime scaling plots (when present).
    for runtime_col in ["runtime_meta", "runtime_lmm", "runtime_qc"]:
        if runtime_col in df.columns and x_ngenes is not None:
            _plot_metric_grid(
                df,
                metric_col=runtime_col,
                x_col=x_ngenes,
                out_path=os.path.join(args.out_dir, f"runtime__{runtime_col}.png"),
                title=f"Runtime — {runtime_col}",
                y_label="seconds",
                hline=None,
            )
            plots_made += 1

    if plots_made == 0:
        raise ValueError("no plots were generated (missing expected columns in the grid TSV)")


if __name__ == "__main__":
    main()
