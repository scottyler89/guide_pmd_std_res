import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import guide_pmd.pmd_std_res_guide_counts as mod
from guide_pmd.contrasts_posthoc import run_posthoc_contrasts


class FakePMD:
    def __init__(self, x, y=None, num_boot=1000, skip_posthoc=False):
        base = np.linspace(-1.0, 1.0, x.shape[1], dtype=float)
        arr = np.vstack([base + float(i) for i in range(x.shape[0])])
        self.z_scores = pd.DataFrame(arr, index=x.index, columns=x.columns)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contrasts_do_not_change_baseline_golden_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "pmd", FakePMD)

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "baseline_small"
    counts_path = fixture_root / "inputs" / "counts.tsv"
    model_matrix_path = fixture_root / "inputs" / "model_matrix.tsv"
    expected_dir = fixture_root / "expected"

    out_dir = tmp_path / "out"
    mod.pmd_std_res_and_stats(
        str(counts_path),
        str(out_dir),
        model_matrix_file=str(model_matrix_path),
        p_combine_idx=None,
        in_annotation_cols=2,
        pre_regress_vars=None,
        n_boot=2,
        seed=1,
        file_sep="tsv",
        contrasts=["treatment"],
    )

    expected_files = [
        "PMD_std_res.tsv",
        "PMD_std_res_stats.tsv",
        "PMD_std_res_stats_resids.tsv",
    ]
    for name in expected_files:
        assert _sha256(out_dir / name) == _sha256(expected_dir / name)

    assert (out_dir / "PMD_std_res_stats_contrasts.tsv").is_file()
    gene_level_dir = out_dir / "gene_level"
    assert (gene_level_dir / "PMD_std_res_gene_meta_contrasts.tsv").is_file()
    assert (gene_level_dir / "PMD_std_res_gene_stouffer_contrasts.tsv").is_file()


def test_posthoc_contrasts_writes_outputs(tmp_path):
    counts_path = tmp_path / "counts.tsv"
    std_res_path = tmp_path / "PMD_std_res.tsv"
    mm_path = tmp_path / "mm.tsv"
    out_dir = tmp_path / "out"

    counts = pd.DataFrame(
        {"gene": ["A", "A", "B"], "s1": [10, 11, 12], "s2": [13, 14, 15], "s3": [16, 17, 18], "s4": [19, 20, 21]},
        index=["g1", "g2", "g3"],
    )
    counts.to_csv(counts_path, sep="\t")

    std_res = pd.DataFrame(
        {"s1": [0.0, 0.1, -0.1], "s2": [0.0, 0.2, -0.2], "s3": [0.1, 0.0, 0.3], "s4": [0.2, 0.0, 0.4]},
        index=["g1", "g2", "g3"],
    )
    std_res.to_csv(std_res_path, sep="\t")

    mm = pd.DataFrame(
        {"treatment": [0.0, 0.0, 1.0, 1.0]},
        index=["s1", "s2", "s3", "s4"],
    )
    mm.to_csv(mm_path, sep="\t")

    run_posthoc_contrasts(
        counts_file=str(counts_path),
        std_res_file=str(std_res_path),
        model_matrix_file=str(mm_path),
        output_dir=str(out_dir),
        annotation_cols=2,
        pre_regress_vars=None,
        file_type="tsv",
        contrasts=["treatment"],
        gene_level=True,
        gene_id_col=1,
        gene_methods=["meta", "stouffer"],
        gene_out_dir=None,
        gene_progress=False,
    )

    assert (out_dir / "PMD_std_res_stats_contrasts.tsv").is_file()
    assert (out_dir / "gene_level" / "PMD_std_res_gene_meta_contrasts.tsv").is_file()
    assert (out_dir / "gene_level" / "PMD_std_res_gene_stouffer_contrasts.tsv").is_file()

