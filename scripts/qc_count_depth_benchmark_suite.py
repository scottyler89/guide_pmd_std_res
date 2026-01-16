from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


_CONFUSION_SENTINELS = [
    "meta_q_balanced_accuracy",
    "meta_q_mcc",
    "stouffer_q_balanced_accuracy",
    "stouffer_q_mcc",
    "lmm_lrt_q_balanced_accuracy",
    "lmm_lrt_q_mcc",
    "lmm_wald_q_balanced_accuracy",
    "lmm_wald_q_mcc",
]

_SUITE_SCORECARDS = [
    "scorecard_null.png",
    "scorecard_signal.png",
    "scorecard_signal_confusion.png",
    "scorecard_signal_estimation.png",
    "method_grid_avg_rank.png",
    "pareto_runtime_vs_tpr.png",
]


def _count_rows(tsv_path: Path) -> int:
    # Fast line count without materializing the full table.
    with tsv_path.open("r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def _any_png(run_dir: Path, names: list[str]) -> bool:
    for n in names:
        if (run_dir / n).is_file():
            return True
    return False


def _tsv_has_cols(path: Path, required: list[str]) -> bool | None:
    if not path.is_file():
        return None
    cols = set(pd.read_csv(path, sep="\t", nrows=0).columns.tolist())
    return bool(all(c in cols for c in required))


def main() -> None:
    parser = argparse.ArgumentParser(description="QC helper for scripts/run_count_depth_benchmark_suite.py output directories.")
    parser.add_argument("--suite-dir", required=True, type=str, help="Suite output directory (the --out-dir used for the suite).")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    if not suite_dir.exists():
        raise FileNotFoundError(str(suite_dir))

    report: dict[str, object] = {
        "suite_dir": str(suite_dir),
        "has_suite_manifest": (suite_dir / "suite_manifest.json").is_file(),
        "has_grid_root": (suite_dir / "grid_runs").is_dir(),
        "grid_variants": [],
        "figures": {},
    }

    grid_root = suite_dir / "grid_runs"
    variant_dirs = sorted([p for p in grid_root.iterdir() if p.is_dir()]) if grid_root.is_dir() else []

    variant_rows: list[dict[str, object]] = []
    for vdir in variant_dirs:
        tsv = vdir / "count_depth_grid_summary.tsv"
        row: dict[str, object] = {
            "variant_dir": str(vdir),
            "has_grid_tsv": tsv.is_file(),
            "n_rows": None,
            "has_confusion_metrics": None,
            "has_any_qq_png": None,
        }
        if tsv.is_file():
            row["n_rows"] = float(_count_rows(tsv))
            cols = set(pd.read_csv(tsv, sep="\t", nrows=0).columns.tolist())
            row["has_confusion_metrics"] = bool(all(c in cols for c in _CONFUSION_SENTINELS))

            # Spot-check existence of QQ PNGs: any run dir with the expected images.
            qq_png_names = [
                "qq_meta_p_null.png",
                "qq_stouffer_p_null.png",
                "qq_lmm_lrt_p_null.png",
                "qq_lmm_wald_p_null.png",
            ]
            has_any = False
            for run_dir in sorted([p for p in vdir.iterdir() if p.is_dir()])[:25]:
                if _any_png(run_dir / "figures", qq_png_names):
                    has_any = True
                    break
            row["has_any_qq_png"] = has_any

        variant_rows.append(row)

    report["grid_variants"] = variant_rows
    report["grid_variants_total"] = float(len(variant_rows))
    report["grid_variants_done"] = float(sum(1 for r in variant_rows if bool(r["has_grid_tsv"])))

    fig_root = suite_dir / "figures" / "scorecards"
    figs: dict[str, object] = {
        "scorecards_dir": str(fig_root),
        "present": {},
    }
    for name in _SUITE_SCORECARDS:
        figs["present"][name] = (fig_root / name).is_file()

    # Spot-check that the method-grid TSV includes the new domain-separated summary columns.
    required_cols = [
        "avg_score_min_domain",
        "worst_score_min_domain",
        "avg_score_null",
        "avg_score_signal",
        "coverage_frac_min_domain",
    ]
    figs["method_grid_has_min_domain_cols"] = _tsv_has_cols(fig_root / "method_grid_avg_rank.tsv", required_cols)
    report["figures"] = figs

    out_path = suite_dir / "suite_qc_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
