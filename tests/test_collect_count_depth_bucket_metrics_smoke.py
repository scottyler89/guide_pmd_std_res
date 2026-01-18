import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_collect_count_depth_bucket_metrics_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_count_depth.py",
            "--out-dir",
            str(run_dir),
            "--n-genes",
            "25",
            "--guides-per-gene",
            "4",
            "--n-control",
            "6",
            "--n-treatment",
            "6",
            "--frac-signal",
            "0.2",
            "--no-qq-plots",
            "--methods",
            "meta",
            "stouffer",
            "lmm",
            "--lmm-scope",
            "all",
            "--max-iter",
            "60",
        ],
        check=True,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    report_path = proc.stdout.strip().splitlines()[-1].strip()
    assert report_path
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    cfg = report["config"]

    grid_tsv = tmp_path / "grid.tsv"
    row = {
        "report_path": report_path,
        "seed": cfg["seed"],
        "alpha": cfg["alpha"],
        "fdr_q": cfg["fdr_q"],
        "response_mode": cfg["response_mode"],
        "normalization_mode": cfg["normalization_mode"],
        "logratio_mode": cfg["logratio_mode"],
        "n_reference_genes": cfg.get("n_reference_genes", 0),
        "depth_covariate_mode": cfg.get("depth_covariate_mode", "none"),
        "include_batch_covariate": cfg.get("include_batch_covariate", False),
        "lmm_scope": cfg.get("lmm_scope", "all"),
        "lmm_max_genes_per_focal_var": cfg.get("lmm_max_genes_per_focal_var", None),
        # minimal scenario cols used for labeling
        "n_genes": cfg["n_genes"],
        "guides_per_gene": cfg["guides_per_gene"],
        "n_control": cfg["n_control"],
        "n_treatment": cfg["n_treatment"],
        "depth_log_sd": cfg["depth_log_sd"],
        "treatment_depth_multiplier": cfg["treatment_depth_multiplier"],
        "frac_signal": cfg["frac_signal"],
        "effect_sd": cfg["effect_sd"],
        "guide_slope_sd": cfg["guide_slope_sd"],
        "offtarget_guide_frac": cfg["offtarget_guide_frac"],
        "offtarget_slope_sd": cfg["offtarget_slope_sd"],
        "nb_overdispersion": cfg["nb_overdispersion"],
    }
    pd.DataFrame([row]).to_csv(grid_tsv, sep="\t", index=False)

    out_tsv = tmp_path / "bucket_metrics.tsv"
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_count_depth_bucket_metrics.py",
            "--grid-tsv",
            str(grid_tsv),
            "--out-tsv",
            str(out_tsv),
            "--jobs",
            "1",
        ],
        check=True,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    df = pd.read_csv(out_tsv, sep="\t")
    assert not df.empty
    assert set(df.columns) >= {"pipeline", "bucket", "metric", "value", "n_genes", "scenario_id", "scenario", "is_null"}

