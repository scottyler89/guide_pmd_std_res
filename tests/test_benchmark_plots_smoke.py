import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_plot_count_depth_grid_heatmaps_smoke(tmp_path):
    pytest.importorskip("matplotlib")

    repo_root = Path(__file__).resolve().parents[1]
    grid_path = tmp_path / "grid.tsv"
    out_dir = tmp_path / "out"

    rows = []
    for frac_signal in [0.0, 0.2]:
        for depth_log_sd in [0.5, 1.0]:
            for tdm in [1.0, 2.0]:
                row = {
                    "response_mode": "log_counts",
                    "normalization_mode": "none",
                    "logratio_mode": "none",
                    "depth_covariate_mode": "none",
                    "include_batch_covariate": False,
                    "frac_signal": frac_signal,
                    "effect_sd": 0.5,
                    "depth_log_sd": depth_log_sd,
                    "treatment_depth_multiplier": tdm,
                    "alpha": 0.05,
                    "fdr_q": 0.1,
                }
                if frac_signal == 0.0:
                    row["meta_null_lambda_gc"] = 1.0 + 0.1 * depth_log_sd + 0.01 * tdm
                    row["meta_alpha_fpr"] = 0.05 + 0.005 * tdm
                    row["meta_null_ks"] = 0.1 * depth_log_sd
                else:
                    row["meta_q_tpr"] = 0.2 + 0.1 * tdm
                    row["meta_q_fdr"] = 0.1
                rows.append(row)

    pd.DataFrame(rows).to_csv(grid_path, sep="\t", index=False)

    cmd = [
        sys.executable,
        "scripts/plot_count_depth_grid_heatmaps.py",
        "--grid-tsv",
        str(grid_path),
        "--out-dir",
        str(out_dir),
        "--prefixes",
        "meta",
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root), capture_output=True, text=True)

    pngs = sorted(out_dir.glob("*.png"))
    assert pngs
    assert all(p.stat().st_size > 0 for p in pngs)


def test_plot_count_depth_scorecards_smoke(tmp_path):
    pytest.importorskip("matplotlib")

    repo_root = Path(__file__).resolve().parents[1]
    grid_path = tmp_path / "grid.tsv"
    out_dir = tmp_path / "out"

    rows = []
    for frac_signal in [0.0, 0.2]:
        for seed in [1, 2]:
            row = {
                "response_mode": "log_counts",
                "normalization_mode": "none",
                "logratio_mode": "none",
                "depth_covariate_mode": "none",
                "include_batch_covariate": False,
                "frac_signal": frac_signal,
                "effect_sd": 0.5,
                "alpha": 0.05,
                "fdr_q": 0.1,
                "seed": seed,
                "runtime_meta": 0.25 + 0.01 * seed,
            }
            if frac_signal == 0.0:
                row["meta_null_lambda_gc"] = 1.05
                row["meta_alpha_fpr"] = 0.05
                row["meta_null_ks"] = 0.1
            else:
                row["meta_q_tpr"] = 0.6
                row["meta_q_fdr"] = 0.1
                row["meta_roc_auc"] = 0.8
                row["meta_average_precision"] = 0.75
                row["meta_theta_rmse_signal"] = 0.2
                row["meta_theta_corr_signal"] = 0.7
                row["meta_theta_sign_acc_signal"] = 0.9
            rows.append(row)

    pd.DataFrame(rows).to_csv(grid_path, sep="\t", index=False)

    cmd = [
        sys.executable,
        "scripts/plot_count_depth_scorecards.py",
        "--grid-tsv",
        str(grid_path),
        "--out-dir",
        str(out_dir),
        "--max-pipelines",
        "5",
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root), capture_output=True, text=True)

    assert (out_dir / "pareto_runtime_vs_tpr.png").is_file()
    assert (out_dir / "pareto_runtime_vs_tpr.png").stat().st_size > 0


def test_plot_count_depth_p_histograms_smoke(tmp_path):
    pytest.importorskip("matplotlib")

    repo_root = Path(__file__).resolve().parents[1]
    grid_path = tmp_path / "grid.tsv"
    out_dir = tmp_path / "out"

    edges = [i / 10.0 for i in range(11)]
    report1 = tmp_path / "r1.json"
    report2 = tmp_path / "r2.json"
    report1.write_text(
        '{"meta":{"p_hist_null":{"n":10.0,"bin_edges":'
        + str(edges).replace(" ", "")
        + ',"counts":[1,1,1,1,1,1,1,1,1,1]}}}',
        encoding="utf-8",
    )
    report2.write_text(
        '{"meta":{"p_hist_null":{"n":10.0,"bin_edges":'
        + str(edges).replace(" ", "")
        + ',"counts":[2,0,1,1,1,1,1,1,1,1]}}}',
        encoding="utf-8",
    )

    df = pd.DataFrame(
        [
            {
                "response_mode": "log_counts",
                "normalization_mode": "none",
                "logratio_mode": "none",
                "depth_covariate_mode": "none",
                "include_batch_covariate": False,
                "frac_signal": 0.0,
                "depth_log_sd": 1.0,
                "treatment_depth_multiplier": 2.0,
                "lmm_scope": "all",
                "seed": 1,
                "report_path": str(report1),
            },
            {
                "response_mode": "log_counts",
                "normalization_mode": "none",
                "logratio_mode": "none",
                "depth_covariate_mode": "none",
                "include_batch_covariate": False,
                "frac_signal": 0.0,
                "depth_log_sd": 1.0,
                "treatment_depth_multiplier": 2.0,
                "lmm_scope": "all",
                "seed": 2,
                "report_path": str(report2),
            },
        ]
    )
    df.to_csv(grid_path, sep="\t", index=False)

    cmd = [
        sys.executable,
        "scripts/plot_count_depth_p_histograms.py",
        "--grid-tsv",
        str(grid_path),
        "--out-dir",
        str(out_dir),
        "--prefixes",
        "meta",
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root), capture_output=True, text=True)

    pngs = sorted(out_dir.glob("*.png"))
    assert pngs
    assert all(p.stat().st_size > 0 for p in pngs)
