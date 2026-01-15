import json
from pathlib import Path
import subprocess
import sys


def test_benchmark_count_depth_report_has_qc_and_metrics(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "bench_out"
    cmd = [
        sys.executable,
        "scripts/benchmark_count_depth.py",
        "--out-dir",
        str(out_dir),
        "--n-genes",
        "25",
        "--guides-per-gene",
        "4",
        "--n-control",
        "6",
        "--n-treatment",
        "6",
        "--frac-signal",
        "0.0",
        "--no-qq-plots",
        "--methods",
        "meta",
        "stouffer",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(repo_root))
    report_path = proc.stdout.strip().splitlines()[-1].strip()

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert "counts_qc" in report
    assert "mean_dispersion" in report["counts_qc"]
    assert "depth_proxy" in report["counts_qc"]

    assert "design_matrix" in report
    assert "corr" in report["design_matrix"]
    assert report["design_matrix"]["corr"]["treatment"]["treatment"] == 1.0

    assert "outputs" in report
    md_path = report["outputs"]["counts_mean_dispersion_tsv"]
    assert md_path
    assert (out_dir / "sim_counts_mean_dispersion.tsv").exists()

    assert "meta" in report
    assert "ks_uniform_null" in report["meta"]
    assert "p_hist_null" in report["meta"]
    assert "roc_auc" in report["meta"]
    assert "average_precision" in report["meta"]
    assert "theta_metrics" in report["meta"]
    assert set(report["meta"]["ks_uniform_null"].keys()) >= {"n", "ks", "ks_p"}
    assert set(report["meta"]["p_hist_null"].keys()) >= {"n", "bin_edges", "counts"}
