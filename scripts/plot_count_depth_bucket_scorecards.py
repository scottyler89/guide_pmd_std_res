from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from count_depth_scenarios import attach_scenarios
from plot_count_depth_scorecards import MetricSpec
from plot_count_depth_scorecards import _dot_scorecard
from plot_count_depth_scorecards import _worst_case_across_scenarios


def _pivot_bucket_metrics(df: pd.DataFrame) -> pd.DataFrame:
    required = {"pipeline", "bucket", "metric", "value", "seed"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"missing required column(s): {sorted(missing)}")

    idx = ["pipeline", "bucket", "scenario_id", "scenario", "is_null", "alpha", "fdr_q"]
    idx = [c for c in idx if c in df.columns]

    per = df.groupby([*idx, "metric"], dropna=False)["value"].mean().reset_index()
    wide = per.pivot(index=idx, columns="metric", values="value").reset_index()

    wide["null_lambda_gc_dev"] = np.abs(pd.to_numeric(wide.get("null_lambda_gc", np.nan), errors="coerce") - 1.0)
    wide["alpha_fpr_dev"] = np.abs(
        pd.to_numeric(wide.get("alpha_fpr", np.nan), errors="coerce") - pd.to_numeric(wide.get("alpha", np.nan), errors="coerce")
    )
    wide["q_fdr_excess"] = np.maximum(
        0.0,
        pd.to_numeric(wide.get("q_fdr", np.nan), errors="coerce") - pd.to_numeric(wide.get("fdr_q", np.nan), errors="coerce"),
    )

    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucket-stratified scorecards for count-depth benchmark suites (local).")
    parser.add_argument("--bucket-metrics-tsv", required=True, type=str, help="Path to count_depth_bucket_metrics.tsv (long format).")
    parser.add_argument("--out-dir", required=True, type=str, help="Directory to write scorecard PNGs + TSVs.")
    parser.add_argument(
        "--max-pipelines",
        type=int,
        default=0,
        help="If >0, limit plots to the top-N pipelines after sorting (default: 0 = no limit).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.bucket_metrics_tsv, sep="\t")
    if df.empty:
        raise ValueError("empty bucket metrics TSV")
    if "scenario_id" not in df.columns:
        df = attach_scenarios(df)

    wide = _pivot_bucket_metrics(df)

    os.makedirs(args.out_dir, exist_ok=True)

    buckets = sorted(wide["bucket"].dropna().astype(str).unique().tolist())
    if not buckets:
        raise ValueError("no buckets found in metrics table")

    null_specs = [
        MetricSpec("null_lambda_gc_dev", "lower"),
        MetricSpec("alpha_fpr_dev", "lower"),
        MetricSpec("null_ks", "lower"),
        MetricSpec("coverage_p_frac", "higher"),
    ]
    signal_specs = [
        MetricSpec("q_fdr_excess", "lower"),
        MetricSpec("q_tpr", "higher"),
        MetricSpec("q_mcc", "higher"),
        MetricSpec("coverage_p_frac", "higher"),
    ]

    for bucket in buckets:
        sub = wide.loc[wide["bucket"].astype(str) == str(bucket)].copy()
        if sub.empty:
            continue

        null_df = sub.loc[sub["is_null"].astype(bool)].copy() if "is_null" in sub.columns else pd.DataFrame()
        sig_df = sub.loc[~sub["is_null"].astype(bool)].copy() if "is_null" in sub.columns else pd.DataFrame()

        if not null_df.empty:
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
                out_path=os.path.join(args.out_dir, f"bucket={bucket}__scorecard_null__sort=avg.png"),
                title=f"Bucket scorecard (null scenarios; worst-case) — bucket={bucket} (sorted by avg)",
                sort_mode="avg",
            )
            null_ranked_worst = _dot_scorecard(
                null_agg,
                pipeline_col="pipeline",
                metric_specs=null_specs,
                out_path=os.path.join(args.out_dir, f"bucket={bucket}__scorecard_null__sort=worst.png"),
                title=f"Bucket scorecard (null scenarios; worst-case) — bucket={bucket} (sorted by worst)",
                sort_mode="worst",
            )
            if int(args.max_pipelines) > 0:
                null_ranked_avg = null_ranked_avg.head(int(args.max_pipelines))
                null_ranked_worst = null_ranked_worst.head(int(args.max_pipelines))
            null_ranked_avg.to_csv(os.path.join(args.out_dir, f"bucket={bucket}__scorecard_null__sort=avg.tsv"), sep="\t", index=False)
            null_ranked_worst.to_csv(os.path.join(args.out_dir, f"bucket={bucket}__scorecard_null__sort=worst.tsv"), sep="\t", index=False)

        if not sig_df.empty:
            sig_agg = _worst_case_across_scenarios(
                sig_df,
                pipeline_col="pipeline",
                scenario_col="scenario_id",
                metric_specs=signal_specs,
            )
            sig_ranked_avg = _dot_scorecard(
                sig_agg,
                pipeline_col="pipeline",
                metric_specs=signal_specs,
                out_path=os.path.join(args.out_dir, f"bucket={bucket}__scorecard_signal__sort=avg.png"),
                title=f"Bucket scorecard (signal scenarios; worst-case) — bucket={bucket} (sorted by avg)",
                sort_mode="avg",
            )
            sig_ranked_worst = _dot_scorecard(
                sig_agg,
                pipeline_col="pipeline",
                metric_specs=signal_specs,
                out_path=os.path.join(args.out_dir, f"bucket={bucket}__scorecard_signal__sort=worst.png"),
                title=f"Bucket scorecard (signal scenarios; worst-case) — bucket={bucket} (sorted by worst)",
                sort_mode="worst",
            )
            if int(args.max_pipelines) > 0:
                sig_ranked_avg = sig_ranked_avg.head(int(args.max_pipelines))
                sig_ranked_worst = sig_ranked_worst.head(int(args.max_pipelines))
            sig_ranked_avg.to_csv(os.path.join(args.out_dir, f"bucket={bucket}__scorecard_signal__sort=avg.tsv"), sep="\t", index=False)
            sig_ranked_worst.to_csv(os.path.join(args.out_dir, f"bucket={bucket}__scorecard_signal__sort=worst.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()

