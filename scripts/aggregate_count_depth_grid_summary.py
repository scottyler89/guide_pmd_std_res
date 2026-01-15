from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


CONFIG_COLS = [
    "response_mode",
    "pmd_n_boot",
    "qq_plots",
    "alpha",
    "fdr_q",
    "n_genes",
    "guides_per_gene",
    "guide_lambda_log_mean",
    "guide_lambda_log_sd",
    "gene_lambda_log_sd",
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
]


def _flatten_cols(cols: pd.Index) -> list[str]:
    out: list[str] = []
    for c in cols:
        if not isinstance(c, tuple):
            out.append(str(c))
            continue
        if len(c) != 2:
            out.append("_".join(str(x) for x in c))
            continue
        out.append(f"{c[0]}_{c[1]}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate count_depth_grid_summary.tsv across seeds.")
    parser.add_argument("--grid-tsv", required=True, type=str, help="Path to count_depth_grid_summary.tsv.")
    parser.add_argument(
        "--out-tsv",
        type=str,
        default=None,
        help="Output path (default: alongside input as count_depth_grid_summary_agg.tsv).",
    )
    args = parser.parse_args()

    grid_path = Path(args.grid_tsv)
    if not grid_path.exists():
        raise FileNotFoundError(str(grid_path))

    out_tsv = args.out_tsv
    if out_tsv is None:
        out_tsv = str(grid_path.with_name("count_depth_grid_summary_agg.tsv"))

    df = pd.read_csv(grid_path, sep="\t")
    if "seed" not in df.columns:
        raise ValueError("expected column 'seed' in grid TSV")

    group_cols = [c for c in CONFIG_COLS if c in df.columns]
    missing_cfg = [c for c in CONFIG_COLS if c not in df.columns]
    if missing_cfg:
        # Keep deterministic behavior across versions: we still aggregate with the subset present.
        pass

    drop_cols = {"tag", "report_path", "seed"}
    metric_cols = [c for c in df.columns if (c not in set(group_cols)) and (c not in drop_cols)]

    # Coerce metrics to numeric where possible; non-numeric metrics are skipped explicitly.
    numeric_metrics: list[str] = []
    for col in metric_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().any():
            df[col] = coerced
            numeric_metrics.append(col)

    grouped = df.groupby(group_cols, dropna=False)

    agg_spec = {m: ["mean", "std", "count"] for m in numeric_metrics}
    agg = grouped.agg(agg_spec)
    agg.columns = _flatten_cols(agg.columns)
    agg = agg.reset_index()

    # Add group-level seed counts.
    n_seeds = grouped["seed"].nunique().rename("n_seeds").reset_index()
    out = agg.merge(n_seeds, on=group_cols, how="left", validate="one_to_one")

    # Stable ordering: group cols first.
    metric_out_cols = [c for c in out.columns if c not in set(group_cols)]
    out = out[group_cols + metric_out_cols]

    os.makedirs(os.path.dirname(out_tsv) or ".", exist_ok=True)
    out.to_csv(out_tsv, sep="\t", index=False)
    print(out_tsv)


if __name__ == "__main__":
    main()
