from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

from count_depth_scenarios import attach_scenarios


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

    def _q(p: float):
        def f(s: pd.Series) -> float:
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return float("nan")
            return float(np.quantile(v, float(p)))

        return f

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
                .agg(median="median", q25=_q(0.25), q75=_q(0.75), n="count")
                .reset_index()
                .sort_values(x_col)
            )
            ax.plot(agg[x_col], agg["median"], marker="o", lw=1)
            ax.fill_between(agg[x_col], agg["q25"], agg["q75"], alpha=0.2, linewidth=0.0)
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


def _plot_selection_tradeoff(
    df: pd.DataFrame,
    *,
    metric_col: str,
    out_path: str,
    title: str,
    y_label: str,
) -> None:
    plt = _require_matplotlib()

    required = {"lmm_scope", "lmm_audit_n", "lmm_max_genes_per_focal_var"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"missing required column(s) for selection tradeoff plot: {sorted(missing)}")
    if metric_col not in df.columns:
        raise ValueError(f"missing metric column: {metric_col}")

    sub = df.copy()
    sub["cap"] = pd.to_numeric(sub["lmm_max_genes_per_focal_var"], errors="coerce").fillna(0).astype(int)
    sub["lmm_scope"] = sub["lmm_scope"].astype(str)
    sub["lmm_audit_n"] = pd.to_numeric(sub["lmm_audit_n"], errors="coerce").fillna(0).astype(int)

    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub.loc[sub[metric_col].notna()].copy()
    if sub.empty:
        raise ValueError(f"no finite values for metric: {metric_col}")

    agg = (
        sub.groupby(["lmm_scope", "lmm_audit_n", "cap"], dropna=False)[metric_col]
        .mean()
        .reset_index()
        .sort_values(["lmm_scope", "lmm_audit_n", "cap"])
    )

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for (scope, audit_n), g in agg.groupby(["lmm_scope", "lmm_audit_n"], sort=True):
        label = f"{scope}, audit_n={int(audit_n)}"
        ax.plot(g["cap"], g[metric_col], marker="o", lw=1, label=label)

    ax.set_xlabel("cap (0 = None/unlimited)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
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

    x_tdm = "treatment_depth_multiplier" if "treatment_depth_multiplier" in df.columns else None
    x_esd = "effect_sd" if "effect_sd" in df.columns else None
    x_ngenes = "n_genes" if "n_genes" in df.columns else None
    x_nsamples = "n_samples" if "n_samples" in df.columns else None
    x_gpg = "guides_per_gene" if "guides_per_gene" in df.columns else None

    plots_made = 0

    for prefix in ["meta", "stouffer", "lmm_lrt", "lmm_wald"]:
        # Null calibration panels: never pool across distinct simulation scenarios.
        if x_tdm is not None and "frac_signal" in df.columns:
            frac = pd.to_numeric(df["frac_signal"], errors="coerce").fillna(0.0)
            null_df = df.loc[frac == 0.0].copy()
            if not null_df.empty:
                null_df = attach_scenarios(null_df, exclude_cols=[x_tdm])
                group_cols = [c for c in ["scenario_id", "scenario", "response_mode"] if c in null_df.columns]
                for key, sub in null_df.groupby(group_cols, dropna=False, sort=True):
                    if not isinstance(key, tuple):
                        key = (key,)
                    key_values = {c: v for c, v in zip(group_cols, key)}
                    tag = f"sc={key_values.get('scenario_id','NA')}__rm={key_values.get('response_mode','NA')}"

                    metric = f"{prefix}_null_lambda_gc"
                    if metric in sub.columns:
                        _plot_metric_grid(
                            sub,
                            metric_col=metric,
                            x_col=x_tdm,
                            out_path=os.path.join(args.out_dir, f"null_lambda_gc__{prefix}__{tag}.png"),
                            title=f"Null calibration (lambda_gc) — {prefix}\n{key_values.get('scenario','')}",
                            y_label="lambda_gc",
                            hline=1.0,
                        )
                        plots_made += 1

                    metric = f"{prefix}_alpha_fpr"
                    if metric in sub.columns:
                        alpha = float(df["alpha"].iloc[0]) if "alpha" in df.columns and not df["alpha"].empty else None
                        _plot_metric_grid(
                            sub,
                            metric_col=metric,
                            x_col=x_tdm,
                            out_path=os.path.join(args.out_dir, f"null_fpr_alpha__{prefix}__{tag}.png"),
                            title=f"Null FPR at alpha — {prefix}\n{key_values.get('scenario','')}",
                            y_label="FPR",
                            hline=alpha,
                        )
                        plots_made += 1

        # Signal power panels: never pool across distinct simulation scenarios.
        if x_esd is not None and "frac_signal" in df.columns:
            frac = pd.to_numeric(df["frac_signal"], errors="coerce").fillna(0.0)
            sig_df = df.loc[frac > 0.0].copy()
            if not sig_df.empty:
                sig_df = attach_scenarios(sig_df, exclude_cols=[x_esd])
                group_cols = [c for c in ["scenario_id", "scenario", "response_mode"] if c in sig_df.columns]
                for key, sub in sig_df.groupby(group_cols, dropna=False, sort=True):
                    if not isinstance(key, tuple):
                        key = (key,)
                    key_values = {c: v for c, v in zip(group_cols, key)}
                    tag = f"sc={key_values.get('scenario_id','NA')}__rm={key_values.get('response_mode','NA')}"

                    metric = f"{prefix}_q_tpr"
                    if metric in sub.columns:
                        _plot_metric_grid(
                            sub,
                            metric_col=metric,
                            x_col=x_esd,
                            out_path=os.path.join(args.out_dir, f"signal_tpr_fdrq__{prefix}__{tag}.png"),
                            title=f"Power (TPR at FDR q) — {prefix}\n{key_values.get('scenario','')}",
                            y_label="TPR",
                            hline=None,
                        )
                        plots_made += 1

    # Simple runtime scaling plots (when present).
    for runtime_col in ["runtime_meta", "runtime_stouffer", "runtime_lmm", "runtime_qc"]:
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
        if runtime_col in df.columns and x_nsamples is not None:
            _plot_metric_grid(
                df,
                metric_col=runtime_col,
                x_col=x_nsamples,
                out_path=os.path.join(args.out_dir, f"runtime__{runtime_col}__vs_n_samples.png"),
                title=f"Runtime vs n_samples — {runtime_col}",
                y_label="seconds",
                hline=None,
            )
            plots_made += 1
        if runtime_col in df.columns and x_gpg is not None:
            _plot_metric_grid(
                df,
                metric_col=runtime_col,
                x_col=x_gpg,
                out_path=os.path.join(args.out_dir, f"runtime__{runtime_col}__vs_guides_per_gene.png"),
                title=f"Runtime vs guides_per_gene — {runtime_col}",
                y_label="seconds",
                hline=None,
            )
            plots_made += 1

    # LMM fit stability / failure rates (when present).
    for metric_col, title, ylab in [
        ("lmm_frac_attempted", "LMM attempted fraction (selected/total)", "fraction"),
        ("lmm_frac_method_lmm", "LMM success fraction among attempted", "fraction"),
        ("lmm_frac_method_meta_fallback", "LMM meta-fallback fraction among attempted", "fraction"),
        ("lmm_frac_method_failed", "LMM failed fraction among attempted", "fraction"),
        ("lmm_lrt_ok_frac_attempted", "LMM LRT ok fraction among attempted", "fraction"),
        ("lmm_wald_ok_frac_attempted", "LMM Wald ok fraction among attempted", "fraction"),
    ]:
        if metric_col in df.columns and x_ngenes is not None:
            _plot_metric_grid(
                df,
                metric_col=metric_col,
                x_col=x_ngenes,
                out_path=os.path.join(args.out_dir, f"lmm_fit__{metric_col}__vs_n_genes.png"),
                title=f"{title} vs n_genes",
                y_label=ylab,
                hline=None,
            )
            plots_made += 1
        if metric_col in df.columns and x_nsamples is not None:
            _plot_metric_grid(
                df,
                metric_col=metric_col,
                x_col=x_nsamples,
                out_path=os.path.join(args.out_dir, f"lmm_fit__{metric_col}__vs_n_samples.png"),
                title=f"{title} vs n_samples",
                y_label=ylab,
                hline=None,
            )
            plots_made += 1
        if metric_col in df.columns and x_gpg is not None:
            _plot_metric_grid(
                df,
                metric_col=metric_col,
                x_col=x_gpg,
                out_path=os.path.join(args.out_dir, f"lmm_fit__{metric_col}__vs_guides_per_gene.png"),
                title=f"{title} vs guides_per_gene",
                y_label=ylab,
                hline=None,
            )
            plots_made += 1

    # LMM selection tradeoffs (cap/scope/audit_n) when present.
    if {"lmm_scope", "lmm_audit_n", "lmm_max_genes_per_focal_var"}.issubset(set(df.columns)):
        if "runtime_lmm" in df.columns:
            _plot_selection_tradeoff(
                df,
                metric_col="runtime_lmm",
                out_path=os.path.join(args.out_dir, "selection_runtime_lmm_vs_cap.png"),
                title="LMM runtime vs selection cap",
                y_label="seconds",
            )
            plots_made += 1

        sig_rows = df.loc[df.get("frac_signal", 0.0) > 0.0].copy()
        if (not sig_rows.empty) and ("lmm_lrt_q_tpr" in sig_rows.columns):
            _plot_selection_tradeoff(
                sig_rows,
                metric_col="lmm_lrt_q_tpr",
                out_path=os.path.join(args.out_dir, "selection_power_lmm_lrt_q_tpr_vs_cap.png"),
                title="Selection tradeoff: LMM LRT power (TPR at FDR q) vs cap",
                y_label="TPR",
            )
            plots_made += 1

        null_rows = df.loc[df.get("frac_signal", 0.0) == 0.0].copy()
        if (not null_rows.empty) and ("lmm_lrt_alpha_fpr" in null_rows.columns):
            _plot_selection_tradeoff(
                null_rows,
                metric_col="lmm_lrt_alpha_fpr",
                out_path=os.path.join(args.out_dir, "selection_null_lmm_lrt_alpha_fpr_vs_cap.png"),
                title="Selection tradeoff: LMM LRT null FPR at alpha vs cap",
                y_label="FPR",
            )
            plots_made += 1

    if plots_made == 0:
        raise ValueError("no plots were generated (missing expected columns in the grid TSV)")


if __name__ == "__main__":
    main()
