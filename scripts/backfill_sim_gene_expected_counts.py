from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guide_pmd.expected_counts import collapse_guide_counts_to_gene_counts
from guide_pmd.expected_counts import compute_two_group_expected_count_quantifiability
from suite_paths import ReportPathResolver


def _iter_run_dirs_from_root(root: str) -> list[str]:
    run_dirs: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "benchmark_report.json" in set(filenames):
            run_dirs.append(dirpath)
    return sorted(set(run_dirs))


def _iter_run_dirs_from_grid_tsv(grid_tsv: str) -> list[str]:
    df = pd.read_csv(grid_tsv, sep="\t")
    if "report_path" not in df.columns:
        raise ValueError("grid TSV must contain a 'report_path' column")
    resolver = ReportPathResolver.from_grid_tsv(grid_tsv)
    paths = df["report_path"].astype(str).tolist()
    run_dirs = [str(resolver.resolve_run_dir(p)) for p in paths if p]
    return sorted(set(run_dirs))


def _load_sim_counts(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, sep="\t", index_col=0)
    if "gene_symbol" not in df.columns:
        raise ValueError(f"sim_counts missing required column 'gene_symbol': {path!r}")
    gene_ids = df["gene_symbol"].astype(str)
    counts_df = df.drop(columns=["gene_symbol"])
    return counts_df, gene_ids


def _load_model_matrix(path: str) -> pd.DataFrame:
    mm = pd.read_csv(path, sep="\t", index_col=0)
    if "treatment" not in mm.columns:
        raise ValueError(f"sim_model_matrix missing required column 'treatment': {path!r}")
    return mm


def _write_expected_counts(
    run_dir: str,
    *,
    resume: bool,
    force: bool,
    quantile: float,
    bucket_thresholds: tuple[float, float, float],
) -> dict[str, object]:
    out_summ_path = os.path.join(run_dir, "sim_gene_expected_counts.tsv")
    out_matrix_path = os.path.join(run_dir, "sim_gene_expected_counts_matrix.tsv.gz")

    if bool(resume) and (not bool(force)) and os.path.isfile(out_summ_path) and os.path.isfile(out_matrix_path):
        return {"run_dir": run_dir, "status": "skipped_existing"}

    counts_path = os.path.join(run_dir, "sim_counts.tsv")
    mm_path = os.path.join(run_dir, "sim_model_matrix.tsv")
    if not os.path.isfile(counts_path):
        raise FileNotFoundError(counts_path)
    if not os.path.isfile(mm_path):
        raise FileNotFoundError(mm_path)

    counts_df, gene_ids = _load_sim_counts(counts_path)
    mm = _load_model_matrix(mm_path)

    gene_counts = collapse_guide_counts_to_gene_counts(counts_df, gene_ids)
    gene_summ, expected_long = compute_two_group_expected_count_quantifiability(
        gene_counts,
        mm["treatment"],
        quantile=float(quantile),
        bucket_thresholds=bucket_thresholds,
    )

    if (not bool(force)) and bool(resume) and os.path.isfile(out_summ_path):
        # Keep additive semantics: if only one file is missing, write only that one.
        pass
    else:
        gene_summ.reset_index().to_csv(out_summ_path, sep="\t", index=False)

    if (not bool(force)) and bool(resume) and os.path.isfile(out_matrix_path):
        pass
    else:
        expected_long.to_csv(out_matrix_path, sep="\t", index=False, compression="gzip")

    return {"run_dir": run_dir, "status": "wrote"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill sim_gene_expected_counts.tsv (+ matrix) for existing benchmark run directories (no reruns)."
    )
    parser.add_argument("--grid-tsv", type=str, default=None, help="Path to count_depth_grid_summary.tsv (must include report_path).")
    parser.add_argument("--root", type=str, default=None, help="Root directory to scan for benchmark_report.json run dirs.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers for backfill (default: 1).")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip run dirs that already have expected-count artifacts (default: enabled).",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing expected-count artifacts (default: disabled).",
    )
    parser.add_argument("--quantile", type=float, default=0.1, help="Quantile used for the mincond driver (default: 0.1).")
    parser.add_argument(
        "--bucket-thresholds",
        type=float,
        nargs=3,
        default=(1.0, 3.0, 5.0),
        help="Bucket thresholds for the mincond driver (default: 1 3 5).",
    )
    args = parser.parse_args()

    if (args.grid_tsv is None) == (args.root is None):
        raise ValueError("provide exactly one of --grid-tsv or --root")

    if args.grid_tsv is not None:
        run_dirs = _iter_run_dirs_from_grid_tsv(str(args.grid_tsv))
    else:
        run_dirs = _iter_run_dirs_from_root(str(args.root))

    if not run_dirs:
        raise ValueError("no run dirs found")

    bucket_thresholds = tuple(float(x) for x in args.bucket_thresholds)

    results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for d in run_dirs:
            try:
                results.append(
                    _write_expected_counts(
                        d,
                        resume=bool(args.resume),
                        force=bool(args.force),
                        quantile=float(args.quantile),
                        bucket_thresholds=bucket_thresholds,  # type: ignore[arg-type]
                    )
                )
            except Exception as e:  # noqa: BLE001
                errors.append({"run_dir": d, "error": str(e)})
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {
                ex.submit(
                    _write_expected_counts,
                    d,
                    resume=bool(args.resume),
                    force=bool(args.force),
                    quantile=float(args.quantile),
                    bucket_thresholds=bucket_thresholds,  # type: ignore[arg-type]
                ): d
                for d in run_dirs
            }
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    errors.append({"run_dir": d, "error": str(e)})

    out = {
        "n_run_dirs": int(len(run_dirs)),
        "n_results": int(len(results)),
        "n_errors": int(len(errors)),
        "results": results,
        "errors": errors,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
