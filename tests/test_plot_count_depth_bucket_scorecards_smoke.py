import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_plot_count_depth_bucket_scorecards_smoke(tmp_path: Path):
    pytest.importorskip("matplotlib")

    repo_root = Path(__file__).resolve().parents[1]
    metrics_path = tmp_path / "bucket_metrics.tsv"
    out_dir = tmp_path / "out"

    rows = []
    for seed in [1, 2]:
        for pipeline in ["meta | resp=log_counts | norm=none | lr=none | depthcov=none | batchcov=0", "lmm_wald | resp=log_counts | norm=none | lr=none | depthcov=none | batchcov=0 | scope=all | lmm_cap=0"]:
            for bucket in ["<1", ">=5"]:
                # Null scenario
                scenario_id = "deadbeef"
                scenario = "null; tdm=2"
                is_null = True
                alpha = 0.05
                fdr_q = 0.1
                base = {
                    "pipeline": pipeline,
                    "bucket": bucket,
                    "scenario_id": scenario_id,
                    "scenario": scenario,
                    "is_null": is_null,
                    "alpha": alpha,
                    "fdr_q": fdr_q,
                    "seed": seed,
                }
                rows.extend(
                    [
                        {**base, "metric": "null_lambda_gc", "value": 1.0},
                        {**base, "metric": "alpha_fpr", "value": 0.05},
                        {**base, "metric": "null_ks", "value": 0.1},
                        {**base, "metric": "coverage_p_frac", "value": 1.0},
                    ]
                )

                # Signal scenario
                scenario_id = "cafebabe"
                scenario = "signal; tdm=2"
                is_null = False
                base = {
                    **base,
                    "scenario_id": scenario_id,
                    "scenario": scenario,
                    "is_null": is_null,
                }
                rows.extend(
                    [
                        {**base, "metric": "q_fdr", "value": 0.1},
                        {**base, "metric": "q_tpr", "value": 0.5 if pipeline.startswith("meta") else 0.6},
                        {**base, "metric": "q_mcc", "value": 0.2 if pipeline.startswith("meta") else 0.3},
                        {**base, "metric": "coverage_p_frac", "value": 1.0},
                    ]
                )

    pd.DataFrame(rows).to_csv(metrics_path, sep="\t", index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_count_depth_bucket_scorecards.py",
            "--bucket-metrics-tsv",
            str(metrics_path),
            "--out-dir",
            str(out_dir),
            "--max-pipelines",
            "5",
        ],
        check=True,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    pngs = sorted(out_dir.glob("*.png"))
    assert pngs
    assert all(p.stat().st_size > 0 for p in pngs)

