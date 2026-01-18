import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def _run_benchmark(tmp_path: Path) -> Path:
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
    report_path = Path(proc.stdout.strip().splitlines()[-1].strip())
    assert report_path.exists()
    return out_dir


def test_backfill_sim_gene_expected_counts_recreates_missing_files(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = _run_benchmark(tmp_path)

    (out_dir / "sim_gene_expected_counts.tsv").unlink()
    (out_dir / "sim_gene_expected_counts_matrix.tsv.gz").unlink()

    cmd = [
        sys.executable,
        "scripts/backfill_sim_gene_expected_counts.py",
        "--root",
        str(tmp_path),
        "--jobs",
        "1",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(repo_root))
    payload = json.loads(proc.stdout.strip())
    assert payload["n_errors"] == 0

    assert (out_dir / "sim_gene_expected_counts.tsv").exists()
    assert (out_dir / "sim_gene_expected_counts_matrix.tsv.gz").exists()


def test_rehydrate_benchmark_run_inputs_restores_sim_counts(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = _run_benchmark(tmp_path)

    (out_dir / "sim_counts.tsv").unlink()
    (out_dir / "sim_model_matrix.tsv").unlink()

    cmd = [
        sys.executable,
        "scripts/rehydrate_benchmark_run_inputs.py",
        "--run-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(repo_root))
    payload = json.loads(proc.stdout.strip())
    assert payload["run_dir"] == str(out_dir)
    assert payload["wrote"]["sim_counts.tsv"] is True
    assert payload["wrote"]["sim_model_matrix.tsv"] is True

    sim_counts = pd.read_csv(out_dir / "sim_counts.tsv", sep="\t", index_col=0)
    assert "gene_symbol" in sim_counts.columns
