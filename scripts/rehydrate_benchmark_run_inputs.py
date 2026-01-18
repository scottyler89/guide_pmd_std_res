from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_if_missing(
    path: str,
    df: pd.DataFrame,
    *,
    sep: str = "\t",
    index: bool = True,
    index_label: str | None = None,
    force: bool,
) -> bool:
    if os.path.exists(path) and not bool(force):
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, sep=sep, index=index, index_label=index_label)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rehydrate missing simulated benchmark inputs (counts/truth/model matrix) from benchmark_report.json config+seed. "
            "Does not recompute PMD standardized residuals by default."
        )
    )
    parser.add_argument("--run-dir", type=str, default=None, help="Benchmark run directory (contains benchmark_report.json).")
    parser.add_argument("--report-path", type=str, default=None, help="Path to benchmark_report.json.")
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, overwrite existing files (default: disabled).",
    )
    parser.add_argument(
        "--write-std-res",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, also regenerate sim_std_res.tsv (may be slow for response_mode=pmd_std_res).",
    )
    args = parser.parse_args()

    if args.run_dir is None and args.report_path is None:
        raise ValueError("must provide --run-dir or --report-path")
    if args.run_dir is not None and args.report_path is not None:
        raise ValueError("provide only one of --run-dir or --report-path")

    if args.report_path is None:
        run_dir = os.path.abspath(str(args.run_dir))
        report_path = os.path.join(run_dir, "benchmark_report.json")
    else:
        report_path = os.path.abspath(str(args.report_path))
        run_dir = os.path.dirname(report_path)

    if not os.path.isfile(report_path):
        raise FileNotFoundError(report_path)

    # Ensure scripts/ is importable (benchmark_count_depth.py lives here).
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from benchmark_count_depth import CountDepthBenchmarkConfig  # noqa: PLC0415
    from benchmark_count_depth import simulate_counts_and_std_res  # noqa: PLC0415
    from benchmark_count_depth import simulate_counts_and_truth  # noqa: PLC0415

    report = _load_json(report_path)
    cfg_dict = dict(report.get("config") or {})
    if not cfg_dict:
        raise ValueError(f"missing config in report: {report_path!r}")

    if "methods" in cfg_dict:
        cfg_dict["methods"] = tuple(cfg_dict["methods"])
    cfg = CountDepthBenchmarkConfig(**cfg_dict)

    wrote: dict[str, bool] = {}

    counts_df, ann_df, mm, truth_sample, truth_gene, truth_guide = simulate_counts_and_truth(cfg)
    counts_out = counts_df.copy()
    counts_out.insert(0, "gene_symbol", ann_df["gene_symbol"])
    counts_out.index.name = "guide_id"

    wrote["sim_counts.tsv"] = _write_if_missing(
        os.path.join(run_dir, "sim_counts.tsv"),
        counts_out,
        sep="\t",
        index=True,
        force=bool(args.force),
    )
    wrote["sim_model_matrix.tsv"] = _write_if_missing(
        os.path.join(run_dir, "sim_model_matrix.tsv"),
        mm,
        sep="\t",
        index=True,
        index_label="sample_id",
        force=bool(args.force),
    )
    wrote["sim_truth_sample.tsv"] = _write_if_missing(
        os.path.join(run_dir, "sim_truth_sample.tsv"),
        truth_sample,
        sep="\t",
        index=False,
        force=bool(args.force),
    )
    wrote["sim_truth_gene.tsv"] = _write_if_missing(
        os.path.join(run_dir, "sim_truth_gene.tsv"),
        truth_gene,
        sep="\t",
        index=False,
        force=bool(args.force),
    )
    wrote["sim_truth_guide.tsv"] = _write_if_missing(
        os.path.join(run_dir, "sim_truth_guide.tsv"),
        truth_guide,
        sep="\t",
        index=False,
        force=bool(args.force),
    )

    if bool(args.write_std_res):
        _counts_df, _ann_df, std_res_df, _mm, _ts, _tg, _tguide = simulate_counts_and_std_res(cfg)
        wrote["sim_std_res.tsv"] = _write_if_missing(
            os.path.join(run_dir, "sim_std_res.tsv"),
            std_res_df,
            sep="\t",
            index=True,
            force=bool(args.force),
        )

    out = {"run_dir": run_dir, "report_path": report_path, "wrote": wrote}
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
