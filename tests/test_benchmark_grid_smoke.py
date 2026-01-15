import subprocess
import sys
from pathlib import Path


def test_run_count_depth_grid_resume_and_jobs_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "grid_runs"

    base_cmd = [
        sys.executable,
        "scripts/run_count_depth_grid.py",
        "--out-dir",
        str(out_dir),
        "--jobs",
        "2",
        "--progress-every",
        "0",
        "--seeds",
        "1",
        "2",
        "--n-genes",
        "10",
        "--guides-per-gene",
        "2",
        "--n-control",
        "4",
        "--n-treatment",
        "4",
        "--treatment-depth-multiplier",
        "1.0",
        "--no-include-depth-covariate",
        "--no-include-batch-covariate",
        "--methods",
        "meta",
    ]

    proc1 = subprocess.run(base_cmd, check=True, cwd=str(repo_root), capture_output=True, text=True)
    grid_tsv_1 = proc1.stdout.strip().splitlines()[-1].strip()
    assert Path(grid_tsv_1).is_file()

    proc2 = subprocess.run(
        [*base_cmd, "--resume"],
        check=True,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    # Should skip all runs (resume) and still write the same TSV.
    first_line = proc2.stdout.strip().splitlines()[0]
    assert "resumed=2" in first_line
    assert "to_run=0" in first_line

    grid_tsv_2 = proc2.stdout.strip().splitlines()[-1].strip()
    assert Path(grid_tsv_2).is_file()
