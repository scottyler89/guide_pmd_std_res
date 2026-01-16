from __future__ import annotations

import argparse
import hashlib
import os

import pandas as pd

from count_depth_scenarios import attach_scenarios


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _pick_metric_col(df: pd.DataFrame, base: str) -> str | None:
    if base in df.columns:
        return base
    alt = f"{base}_mean"
    if alt in df.columns:
        return alt
    return None


def _tag_from_values(values: dict[str, object]) -> str:
    parts: list[str] = []
    keys = list(values.keys())
    if "prefix" in keys:
        keys.remove("prefix")
        keys = ["prefix", *sorted(keys)]
    else:
        keys = sorted(keys)
    for k in keys:
        v = str(values[k]).replace("/", "-").replace(" ", "")
        parts.append(f"{k}={v}")
    full = "__".join(parts) if parts else "all"
    if len(full) <= 140:
        return full
    h = hashlib.sha1(full.encode("utf-8")).hexdigest()[:12]
    short = "__".join(parts[:5])
    return f"{short}__h={h}" if short else f"h={h}"


def _plot_one(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    out_path: str,
) -> None:
    plt = _require_matplotlib()

    sub = df.copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.loc[sub[x_col].notna() & sub[y_col].notna()].copy()
    if sub.empty:
        return

    colors = {"none": "#d95f02", "log_libsize": "#1b9e77"}
    default_color = "#7570b3"

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for mode, g in sub.groupby(color_col, dropna=False, sort=True):
        m = str(mode)
        ax.scatter(
            g[x_col],
            g[y_col],
            s=55,
            alpha=0.8,
            edgecolors="none",
            label=m,
            color=colors.get(m, default_color),
        )

    ax.axhline(0.0, color="black", lw=1, alpha=0.6)
    ax.set_xlabel("corr(treatment, log_libsize)  (observed)")
    ax.set_ylabel("mean theta_hat on null genes")
    ax.set_title(title)
    ax.grid(True, lw=0.5, alpha=0.3)
    ax.legend(fontsize=8, frameon=False, title=color_col)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth confounding diagnostics for the count-depth benchmark grid (local).")
    parser.add_argument("--grid-tsv", required=True, type=str, help="Path to count_depth_grid_summary.tsv (raw or aggregated).")
    parser.add_argument("--out-dir", required=True, type=str, help="Directory to write diagnostic figures.")
    parser.add_argument(
        "--prefixes",
        type=str,
        nargs="+",
        default=["meta", "lmm_lrt", "lmm_wald"],
        help="Which method prefixes to plot (default: meta lmm_lrt lmm_wald).",
    )
    parser.add_argument(
        "--only-null-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, restrict to frac_signal==0 runs (default: enabled).",
    )
    parser.add_argument(
        "--group-cols",
        type=str,
        nargs="+",
        default=[
            "scenario_id",
            "response_mode",
            "normalization_mode",
            "logratio_mode",
            "include_batch_covariate",
            "lmm_scope",
            "lmm_max_genes_per_focal_var",
        ],
        help="Columns used to define separate panels (default: key pipeline knobs).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.grid_tsv, sep="\t")
    os.makedirs(args.out_dir, exist_ok=True)

    if bool(args.only_null_runs) and "frac_signal" in df.columns:
        frac_col = _pick_metric_col(df, "frac_signal")
        if frac_col is not None:
            df = df.loc[pd.to_numeric(df[frac_col], errors="coerce").fillna(0.0) == 0.0].copy()
    if df.empty:
        raise ValueError("no rows selected after filtering")

    df = attach_scenarios(df)

    x_col = _pick_metric_col(df, "depth_corr_treatment_log_libsize")
    if x_col is None:
        raise ValueError("missing required column: depth_corr_treatment_log_libsize (or *_mean)")

    group_cols = [str(c) for c in args.group_cols if str(c) in df.columns]
    if not group_cols:
        group_cols = []

    plots_made = 0
    for prefix in [str(p) for p in args.prefixes]:
        y_base = f"{prefix}_theta_null_mean"
        y_col = _pick_metric_col(df, y_base)
        if y_col is None:
            continue

        for key, sub in df.groupby(group_cols, dropna=False, sort=True) if group_cols else [(tuple(), df)]:
            if not isinstance(key, tuple):
                key = (key,)
            key_values = {c: v for c, v in zip(group_cols, key)}
            tag = _tag_from_values({"prefix": prefix, **key_values})

            scenario_label = ""
            if "scenario" in sub.columns and int(sub["scenario"].nunique(dropna=False)) == 1:
                scenario_label = str(sub["scenario"].iloc[0])
            title = f"{prefix}: null-theta bias vs depth confounding\n{scenario_label}\n{tag}" if scenario_label else f"{prefix}: null-theta bias vs depth confounding\n{tag}"
            out_path = os.path.join(args.out_dir, f"theta_bias_null_mean__{tag}.png")

            color_col = "depth_covariate_mode" if "depth_covariate_mode" in sub.columns else "include_depth_covariate"
            _plot_one(sub, x_col=x_col, y_col=y_col, color_col=color_col, title=title, out_path=out_path)
            plots_made += 1

    if plots_made == 0:
        raise ValueError("no plots generated (missing expected theta_null_mean columns?)")


if __name__ == "__main__":
    main()
