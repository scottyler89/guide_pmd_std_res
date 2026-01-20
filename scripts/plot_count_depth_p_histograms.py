from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd

from count_depth_scenarios import attach_scenarios
from suite_paths import ReportPathResolver


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _series_or_constant(df: pd.DataFrame, col: str, default: object) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * df.shape[0], index=df.index)


def _apply_where(df: pd.DataFrame, where: list[str]) -> pd.DataFrame:
    out = df
    for clause in where:
        if "=" not in clause:
            raise ValueError(f"invalid --where (expected col=value): {clause!r}")
        col, raw = clause.split("=", 1)
        col = col.strip()
        raw = raw.strip()
        if col not in out.columns:
            raise ValueError(f"--where references missing column: {col!r}")

        series = out[col]
        # Try numeric compare if the column is numeric-ish and the value parses.
        value_num = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
        col_num = pd.to_numeric(series, errors="coerce")
        if np.isfinite(value_num) and col_num.notna().any():
            out = out.loc[col_num == float(value_num)]
        else:
            out = out.loc[series.astype(str) == raw]
    return out


def _stable_group_tag(values: dict[str, object]) -> str:
    parts: list[str] = []
    for k in sorted(values.keys()):
        v = str(values[k]).replace("/", "-").replace(" ", "")
        parts.append(f"{k}={v}")
    full = "__".join(parts) if parts else "all"
    if len(full) <= 120:
        return full
    h = hashlib.sha1(full.encode("utf-8")).hexdigest()[:12]
    short = "__".join(parts[:4])
    return f"{short}__h={h}" if short else f"h={h}"


def _aggregate_p_hists(
    report_paths: Iterable[str],
    *,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    edges: np.ndarray | None = None
    counts_sum: np.ndarray | None = None

    for path in report_paths:
        rep = _load_json(path)
        if prefix not in rep:
            continue
        hist = rep[prefix].get("p_hist_null", None)
        if not isinstance(hist, dict):
            continue
        bin_edges = np.asarray(hist.get("bin_edges", []), dtype=float)
        counts = np.asarray(hist.get("counts", []), dtype=float)
        if bin_edges.size < 2 or counts.size == 0:
            continue
        if counts.size != bin_edges.size - 1:
            continue

        if edges is None:
            edges = bin_edges
            counts_sum = np.zeros_like(counts, dtype=float)
        if edges.shape != bin_edges.shape or not np.allclose(edges, bin_edges, rtol=0.0, atol=0.0):
            raise ValueError("p-hist bin edges mismatch across reports (expected fixed binning)")
        counts_sum = counts_sum + counts

    if edges is None or counts_sum is None:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return edges, counts_sum


def _plot_group(
    report_paths: list[str],
    *,
    prefixes: list[str],
    out_path: str,
    title: str,
) -> None:
    plt = _require_matplotlib()

    n = len(prefixes)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(6.5, 2.2 * n), dpi=150, sharex=True)
    if n == 1:
        axes = [axes]

    plotted_any = False
    for ax, prefix in zip(axes, prefixes):
        edges, counts = _aggregate_p_hists(report_paths, prefix=prefix)
        if edges.size == 0 or counts.size == 0:
            ax.set_axis_off()
            continue
        mids = 0.5 * (edges[:-1] + edges[1:])
        total = float(np.sum(counts))
        frac = counts / total if total > 0 else counts

        ax.bar(mids, frac, width=float(edges[1] - edges[0]), align="center", edgecolor="none", alpha=0.85)
        ax.axhline(1.0 / float(counts.size), color="black", lw=1, alpha=0.6)
        ax.set_ylabel(prefix)
        plotted_any = True

    if not plotted_any:
        return

    axes[-1].set_xlabel("p-value bin")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot null p-value histograms from count-depth benchmark reports (local).")
    parser.add_argument("--grid-tsv", required=True, type=str, help="Path to count_depth_grid_summary.tsv (must include report_path).")
    parser.add_argument("--out-dir", required=True, type=str, help="Directory to write PNGs.")
    parser.add_argument(
        "--prefixes",
        type=str,
        nargs="+",
        default=["meta", "stouffer", "lmm_lrt", "lmm_wald"],
        help="Method prefixes to include (default: meta stouffer lmm_lrt lmm_wald).",
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
            "depth_covariate_mode",
            "include_batch_covariate",
            "lmm_scope",
            "lmm_max_genes_per_focal_var",
        ],
        help="Config columns to group reports before aggregating p-hists (default: key pipeline knobs).",
    )
    parser.add_argument(
        "--where",
        type=str,
        action="append",
        default=[],
        help="Filter rows with col=value (repeatable). Example: --where frac_signal=0.0",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.grid_tsv, sep="\t")
    if "report_path" not in df.columns:
        raise ValueError("grid TSV is missing report_path; use the raw count_depth_grid_summary.tsv (not the aggregated TSV)")

    df = _apply_where(df, [str(x) for x in args.where])

    # Default to null runs only unless the user explicitly includes signal runs.
    if "frac_signal" in df.columns and not any(w.startswith("frac_signal=") for w in args.where):
        df = df.loc[pd.to_numeric(df["frac_signal"], errors="coerce").fillna(0.0) == 0.0].copy()

    # Attach explicit scenario IDs so we never aggregate p-hists across heterogeneous simulation conditions.
    df = attach_scenarios(df)

    if df.empty:
        raise ValueError("no rows selected after filtering")

    group_cols = [c for c in [str(c) for c in args.group_cols] if c in df.columns]
    os.makedirs(args.out_dir, exist_ok=True)
    resolver = ReportPathResolver.from_grid_tsv(args.grid_tsv)

    plots_made = 0
    for key, sub in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        key_values = {c: v for c, v in zip(group_cols, key)}
        tag = _stable_group_tag(key_values)
        report_paths = [str(resolver.resolve_report_path(p)) for p in sub["report_path"].astype(str).tolist()]

        scenario_label = ""
        if "scenario" in sub.columns and int(sub["scenario"].nunique(dropna=False)) == 1:
            scenario_label = str(sub["scenario"].iloc[0])
        title = "Null p-value histograms\n" + (scenario_label + "\n" if scenario_label else "") + tag
        out_path = os.path.join(args.out_dir, f"p_hist_null__{tag}.png")
        _plot_group(report_paths, prefixes=[str(p) for p in args.prefixes], out_path=out_path, title=title)
        plots_made += 1

    if plots_made == 0:
        raise ValueError("no p-hist figures produced (missing report content?)")


if __name__ == "__main__":
    main()
