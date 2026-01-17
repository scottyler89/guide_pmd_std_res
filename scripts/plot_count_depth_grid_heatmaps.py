from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd

from count_depth_scenarios import attach_scenarios


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


CONFIG_COLS = [
    "response_mode",
    "pmd_n_boot",
    "normalization_mode",
    "logratio_mode",
    "n_reference_genes",
    "qq_plots",
    "alpha",
    "fdr_q",
    "n_genes",
    "guides_per_gene",
    "guide_lambda_log_mean",
    "guide_lambda_log_sd",
    "gene_lambda_log_sd",
    "gene_lambda_family",
    "gene_lambda_mix_pi_high",
    "gene_lambda_mix_delta_log_mean",
    "gene_lambda_power_alpha",
    "guide_lambda_family",
    "guide_lambda_dirichlet_alpha0",
    "depth_log_sd",
    "treatment_depth_multiplier",
    "n_batches",
    "batch_confounding_strength",
    "batch_depth_log_sd",
    "depth_covariate_mode",
    "include_depth_covariate",
    "include_batch_covariate",
    "frac_signal",
    "effect_sd",
    "guide_slope_sd",
    "offtarget_guide_frac",
    "offtarget_slope_sd",
    "nb_overdispersion",
    "methods",
    "lmm_scope",
    "lmm_q_meta",
    "lmm_q_het",
    "lmm_audit_n",
    "lmm_audit_seed",
    "lmm_max_genes_per_focal_var",
    "seed",
]


def _sorted_unique(df: pd.DataFrame, col: str) -> list[object]:
    if col not in df.columns:
        return []
    vals = df[col].dropna().unique().tolist()
    try:
        return sorted(vals)
    except TypeError:
        return vals


def _strict_validate_group(
    sub: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    facet_cols: list[str],
) -> None:
    # Only enforce uniqueness over config columns (not metrics), and allow multiple seeds.
    cfg_cols = [c for c in CONFIG_COLS if c in sub.columns]
    ignore = set([x_col, y_col, "seed"]) | set(facet_cols)
    varying: list[str] = []
    for c in cfg_cols:
        if c in ignore:
            continue
        if int(sub[c].nunique(dropna=False)) > 1:
            varying.append(c)
    if varying:
        raise ValueError(
            "heatmap group is not uniquely specified; varying config column(s): "
            + ", ".join(varying)
            + ". Either filter the grid TSV or add column(s) to --facet-cols."
        )


def _pivot_mean(
    df: pd.DataFrame,
    *,
    metric_col: str,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    sub = df[[x_col, y_col, metric_col]].copy()
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub.loc[sub[metric_col].notna()].copy()
    if sub.empty:
        return pd.DataFrame()

    agg = sub.groupby([y_col, x_col], dropna=False)[metric_col].mean().reset_index()
    pivot = agg.pivot(index=y_col, columns=x_col, values=metric_col)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    return pivot


def _plot_heatmap(
    pivot: pd.DataFrame,
    *,
    out_path: str,
    title: str,
    fmt: str,
    cmap: str,
) -> None:
    plt = _require_matplotlib()
    if pivot.empty:
        return

    x_vals = [str(x) for x in pivot.columns.tolist()]
    y_vals = [str(y) for y in pivot.index.tolist()]
    mat = pivot.to_numpy(dtype=float)

    fig_w = float(max(4.5, 0.9 * len(x_vals) + 2.0))
    fig_h = float(max(3.5, 0.6 * len(y_vals) + 1.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_title(title)
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals, rotation=30, ha="right")
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals)

    # Annotate values for small grids.
    if len(x_vals) <= 8 and len(y_vals) <= 10:
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                v = mat[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _iter_groups(df: pd.DataFrame, facet_cols: list[str]) -> Iterable[tuple[tuple[object, ...], pd.DataFrame]]:
    if not facet_cols:
        yield tuple(), df
        return
    for key, sub in df.groupby(facet_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        yield key, sub


def _key_to_tag(facet_cols: list[str], key: tuple[object, ...]) -> str:
    parts: list[str] = []
    for c, v in zip(facet_cols, key):
        s = str(v)
        s = s.replace("/", "-").replace(" ", "")
        parts.append(f"{c}={s}")
    full = "__".join(parts) if parts else "all"
    if len(full) <= 120:
        return full
    h = hashlib.sha1(full.encode("utf-8")).hexdigest()[:12]
    short = "__".join(parts[:5])
    return f"{short}__h={h}" if short else f"h={h}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Heatmap panels for count-depth benchmark grids (local).")
    parser.add_argument("--grid-tsv", required=True, type=str, help="Path to count_depth_grid_summary.tsv (raw or aggregated).")
    parser.add_argument("--out-dir", required=True, type=str, help="Directory to write heatmap PNGs.")
    parser.add_argument(
        "--prefixes",
        type=str,
        nargs="+",
        default=["meta", "stouffer", "lmm_lrt", "lmm_wald"],
        help="Method prefixes to plot (default: meta stouffer lmm_lrt lmm_wald).",
    )
    parser.add_argument("--x-col", type=str, default="treatment_depth_multiplier", help="Heatmap x-axis column (default: treatment_depth_multiplier).")
    parser.add_argument("--y-col", type=str, default="depth_log_sd", help="Heatmap y-axis column (default: depth_log_sd).")
    parser.add_argument(
        "--facet-cols",
        type=str,
        nargs="+",
        default=[
            "scenario_id",
            "response_mode",
            "normalization_mode",
            "logratio_mode",
            "depth_covariate_mode",
            "include_batch_covariate",
            "n_reference_genes",
            "lmm_scope",
        ],
        help="Columns used to define independent heatmaps (default: key pipeline/scenario knobs).",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, fail if any non-faceted config columns vary within a heatmap group (default: enabled).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, skip plots whose output PNG already exists (useful for resuming a long run).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.grid_tsv, sep="\t")
    os.makedirs(args.out_dir, exist_ok=True)

    x_col = str(args.x_col)
    y_col = str(args.y_col)
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"missing required axis column(s): x={x_col!r} y={y_col!r}")

    # Tag each row with a scenario family identifier, excluding the heatmap axes so we can visualize
    # how performance changes across (x, y) while holding other simulation knobs fixed.
    df = attach_scenarios(df, exclude_cols=[x_col, y_col])

    facet_cols = [c for c in [str(c) for c in args.facet_cols] if c in df.columns]

    alpha = float(pd.to_numeric(df.get("alpha", 0.05), errors="coerce").dropna().iloc[0]) if "alpha" in df.columns else 0.05
    fdr_q = float(pd.to_numeric(df.get("fdr_q", 0.1), errors="coerce").dropna().iloc[0]) if "fdr_q" in df.columns else 0.1

    plots_made = 0
    for prefix in [str(p) for p in args.prefixes]:
        # Null calibration panels.
        null_df = df.loc[pd.to_numeric(df.get("frac_signal", 0.0), errors="coerce").fillna(0.0) == 0.0].copy()
        if not null_df.empty:
            if f"{prefix}_null_lambda_gc" in null_df.columns:
                null_df[f"{prefix}_null_lambda_gc_dev"] = np.abs(pd.to_numeric(null_df[f"{prefix}_null_lambda_gc"], errors="coerce") - 1.0)
            if f"{prefix}_alpha_fpr" in null_df.columns:
                null_df[f"{prefix}_alpha_fpr_dev"] = np.abs(pd.to_numeric(null_df[f"{prefix}_alpha_fpr"], errors="coerce") - alpha)

            null_metrics = [
                (f"{prefix}_null_lambda_gc_dev", "Null inflation: |lambda_gc-1|", ".3f", "viridis_r"),
                (f"{prefix}_alpha_fpr_dev", f"Null FPR dev: |FPR-alpha| (alpha={alpha})", ".3f", "viridis_r"),
                (f"{prefix}_null_ks", "Null p-value KS statistic vs Uniform(0,1)", ".3f", "viridis_r"),
            ]
            for metric_col, metric_label, fmt, cmap in null_metrics:
                if metric_col not in null_df.columns:
                    continue
                for key, sub in _iter_groups(null_df, facet_cols):
                    if bool(args.strict):
                        _strict_validate_group(sub, x_col=x_col, y_col=y_col, facet_cols=facet_cols)
                    pivot = _pivot_mean(sub, metric_col=metric_col, x_col=x_col, y_col=y_col)
                    if pivot.empty:
                        continue
                    tag = _key_to_tag(facet_cols, key)
                    metric_slug = metric_col.removeprefix(f"{prefix}_")
                    out_path = os.path.join(args.out_dir, f"heatmap__{prefix}__{metric_slug}__{tag}.png")
                    if bool(args.skip_existing) and os.path.isfile(out_path):
                        continue
                    _plot_heatmap(
                        pivot,
                        out_path=out_path,
                        title=f"{prefix} — {metric_label}\n{tag}",
                        fmt=fmt,
                        cmap=cmap,
                    )
                    plots_made += 1

        # Signal panels.
        sig_df = df.loc[pd.to_numeric(df.get("frac_signal", 0.0), errors="coerce").fillna(0.0) > 0.0].copy()
        if not sig_df.empty:
            if f"{prefix}_q_fdr" in sig_df.columns:
                sig_df[f"{prefix}_q_fdr_excess"] = np.maximum(0.0, pd.to_numeric(sig_df[f"{prefix}_q_fdr"], errors="coerce") - fdr_q)

            sig_metrics = [
                (f"{prefix}_q_fdr_excess", f"Signal FDR excess over q (q={fdr_q})", ".3f", "viridis_r"),
                (f"{prefix}_q_tpr", f"Signal power (TPR at q={fdr_q})", ".3f", "viridis"),
            ]
            for metric_col, metric_label, fmt, cmap in sig_metrics:
                if metric_col not in sig_df.columns:
                    continue
                for key, sub in _iter_groups(sig_df, facet_cols):
                    if bool(args.strict):
                        _strict_validate_group(sub, x_col=x_col, y_col=y_col, facet_cols=facet_cols)
                    pivot = _pivot_mean(sub, metric_col=metric_col, x_col=x_col, y_col=y_col)
                    if pivot.empty:
                        continue
                    tag = _key_to_tag(facet_cols, key)
                    metric_slug = metric_col.removeprefix(f"{prefix}_")
                    out_path = os.path.join(args.out_dir, f"heatmap__{prefix}__{metric_slug}__{tag}.png")
                    if bool(args.skip_existing) and os.path.isfile(out_path):
                        continue
                    _plot_heatmap(
                        pivot,
                        out_path=out_path,
                        title=f"{prefix} — {metric_label}\n{tag}",
                        fmt=fmt,
                        cmap=cmap,
                    )
                    plots_made += 1

    if plots_made == 0:
        raise ValueError("no heatmaps were generated (missing expected metric columns or empty subsets)")


if __name__ == "__main__":
    main()
