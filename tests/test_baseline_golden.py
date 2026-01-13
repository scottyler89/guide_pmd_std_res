import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import guide_pmd.pmd_std_res_guide_counts as mod


class FakePMD:
    def __init__(self, x, y=None, num_boot=1000, skip_posthoc=False):
        base = np.linspace(-1.0, 1.0, x.shape[1], dtype=float)
        arr = np.vstack([base + float(i) for i in range(x.shape[0])])
        self.z_scores = pd.DataFrame(arr, index=x.index, columns=x.columns)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_baseline_outputs_match_golden(tmp_path, monkeypatch):
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
    )

    expected_files = [
        "PMD_std_res.tsv",
        "PMD_std_res_stats.tsv",
        "PMD_std_res_stats_resids.tsv",
    ]
    for name in expected_files:
        assert _sha256(out_dir / name) == _sha256(expected_dir / name)

    # Gene-level outputs are additive and must not change the baseline TSV bytes.
    gene_level_dir = out_dir / "gene_level"
    assert gene_level_dir.is_dir()
    assert (gene_level_dir / "PMD_std_res_gene_meta.tsv").is_file()
    assert (gene_level_dir / "PMD_std_res_gene_lmm_selection.tsv").is_file()
    assert (gene_level_dir / "PMD_std_res_gene_lmm.tsv").is_file()
    assert (gene_level_dir / "PMD_std_res_gene_qc.tsv").is_file()
    assert (gene_level_dir / "PMD_std_res_gene_flagged.tsv").is_file()
    assert (gene_level_dir / "PMD_std_res_gene_mixture.tsv").is_file()
    assert (gene_level_dir / "PMD_std_res_gene_mixture_guides.tsv").is_file()
    assert (gene_level_dir / "PMD_std_res_gene_tmeta.tsv").is_file()
    assert (gene_level_dir / "PMD_std_res_gene_tmeta_guides.tsv").is_file()
