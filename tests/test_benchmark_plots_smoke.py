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


def test_plot_count_depth_confounding_diagnostics_smoke(tmp_path):
    pytest.importorskip("matplotlib")

    repo_root = Path(__file__).resolve().parents[1]
    grid_path = tmp_path / "grid.tsv"
    out_dir = tmp_path / "out"

    df = pd.DataFrame(
        [
            {
                "response_mode": "log_counts",
                "normalization_mode": "none",
                "logratio_mode": "none",
                "depth_covariate_mode": "none",
                "include_batch_covariate": False,
                "frac_signal": 0.0,
                "depth_corr_treatment_log_libsize": 0.8,
                "meta_theta_null_mean": 0.4,
            },
            {
                "response_mode": "log_counts",
                "normalization_mode": "none",
                "logratio_mode": "none",
                "depth_covariate_mode": "log_libsize",
                "include_batch_covariate": False,
                "frac_signal": 0.0,
                "depth_corr_treatment_log_libsize": 0.8,
                "meta_theta_null_mean": 0.05,
            },
        ]
    )
    df.to_csv(grid_path, sep="\t", index=False)

    cmd = [
        sys.executable,
        "scripts/plot_count_depth_confounding_diagnostics.py",
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


def test_plot_benchmark_method_agreement_smoke(tmp_path):
    pytest.importorskip("matplotlib")

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir(parents=True, exist_ok=True)

    truth = pd.DataFrame(
        {
            "gene_id": [f"gene_{i:03d}" for i in range(6)],
            "is_signal": [False, False, False, False, True, True],
            "theta_true": [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
        }
    )
    truth.to_csv(run_dir / "sim_truth_gene.tsv", sep="\t", index=False)
    (run_dir / "benchmark_report.json").write_text('{"config":{"fdr_q":0.1,"alpha":0.05}}', encoding="utf-8")

    meta = pd.DataFrame(
        {
            "gene_id": truth["gene_id"],
            "focal_var": ["treatment"] * truth.shape[0],
            "theta": [0.0] * truth.shape[0],
            "p": [0.5, 0.6, 0.4, 0.3, 1e-6, 1e-4],
            "p_adj": [0.6, 0.6, 0.6, 0.6, 1e-5, 5e-4],
        }
    )
    meta.to_csv(run_dir / "PMD_std_res_gene_meta.tsv", sep="\t", index=False)

    lmm = pd.DataFrame(
        {
            "gene_id": truth["gene_id"],
            "focal_var": ["treatment"] * truth.shape[0],
            "theta": [0.0] * truth.shape[0],
            "lrt_p": [0.5, 0.5, 0.5, 0.5, 1e-7, 1e-3],
            "lrt_p_adj": [0.6, 0.6, 0.6, 0.6, 1e-6, 1e-2],
            "wald_p": [0.5, 0.5, 0.5, 0.5, 1e-6, 1e-2],
            "wald_p_adj": [0.6, 0.6, 0.6, 0.6, 1e-5, 5e-2],
        }
    )
    lmm.to_csv(run_dir / "PMD_std_res_gene_lmm.tsv", sep="\t", index=False)

    cmd = [
        sys.executable,
        "scripts/plot_benchmark_method_agreement.py",
        "--run-dir",
        str(run_dir),
        "--out-dir",
        str(out_dir),
        "--focal-var",
        "treatment",
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root), capture_output=True, text=True)

    assert (out_dir / "method_pair_agreement.tsv").is_file()
    pngs = sorted(out_dir.glob("*.png"))
    assert pngs
    assert all(p.stat().st_size > 0 for p in pngs)


def test_run_count_depth_benchmark_suite_smoke(tmp_path):
    pytest.importorskip("matplotlib")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "suite"
    grid_tsv = tmp_path / "grid.tsv"

    # Minimal raw grid TSV + minimal benchmark report JSONs for p-hist aggregation.
    edges = [i / 10.0 for i in range(11)]
    report1 = tmp_path / "r1.json"
    report2 = tmp_path / "r2.json"
    report1.write_text(
        '{"config":{"fdr_q":0.1,"alpha":0.05},"meta":{"p_hist_null":{"n":10.0,"bin_edges":'
        + str(edges).replace(" ", "")
        + ',"counts":[1,1,1,1,1,1,1,1,1,1]}}}',
        encoding="utf-8",
    )
    report2.write_text(
        '{"config":{"fdr_q":0.1,"alpha":0.05},"meta":{"p_hist_null":{"n":10.0,"bin_edges":'
        + str(edges).replace(" ", "")
        + ',"counts":[2,0,1,1,1,1,1,1,1,1]}}}',
        encoding="utf-8",
    )

    df = pd.DataFrame(
        [
            {
                "tag": "null",
                "report_path": str(report1),
                "seed": 1,
                "response_mode": "log_counts",
                "normalization_mode": "none",
                "logratio_mode": "none",
                "depth_covariate_mode": "none",
                "include_depth_covariate": False,
                "include_batch_covariate": False,
                "n_genes": 20,
                "guides_per_gene": 4,
                "n_control": 4,
                "n_treatment": 4,
                "n_samples": 8,
                "depth_log_sd": 1.0,
                "treatment_depth_multiplier": 2.0,
                "frac_signal": 0.0,
                "effect_sd": 0.5,
                "alpha": 0.05,
                "fdr_q": 0.1,
                "runtime_meta": 0.2,
                "meta_null_lambda_gc": 1.05,
                "meta_alpha_fpr": 0.05,
                "meta_null_ks": 0.1,
                "depth_corr_treatment_log_libsize": 0.8,
                "meta_theta_null_mean": 0.4,
            },
            {
                "tag": "signal",
                "report_path": str(report2),
                "seed": 1,
                "response_mode": "log_counts",
                "normalization_mode": "none",
                "logratio_mode": "none",
                "depth_covariate_mode": "none",
                "include_depth_covariate": False,
                "include_batch_covariate": False,
                "n_genes": 20,
                "guides_per_gene": 4,
                "n_control": 4,
                "n_treatment": 4,
                "n_samples": 8,
                "depth_log_sd": 1.0,
                "treatment_depth_multiplier": 2.0,
                "frac_signal": 0.2,
                "effect_sd": 0.5,
                "alpha": 0.05,
                "fdr_q": 0.1,
                "runtime_meta": 0.2,
                "meta_q_tpr": 0.6,
                "meta_q_fdr": 0.1,
                "meta_roc_auc": 0.8,
                "meta_average_precision": 0.75,
                "meta_theta_rmse_signal": 0.2,
                "meta_theta_corr_signal": 0.7,
                "meta_theta_sign_acc_signal": 0.9,
            },
        ]
    )
    df.to_csv(grid_tsv, sep="\t", index=False)

    cmd = [
        sys.executable,
        "scripts/run_count_depth_benchmark_suite.py",
        "--out-dir",
        str(out_dir),
        "--grid-tsv",
        str(grid_tsv),
    ]
    proc = subprocess.run(cmd, check=True, cwd=str(repo_root), capture_output=True, text=True)
    manifest_path = proc.stdout.strip().splitlines()[-1].strip()
    assert manifest_path
    assert (out_dir / "suite_manifest.json").is_file()
