import hashlib

import numpy as np
import pandas as pd

import guide_pmd.pmd_std_res_guide_counts as mod


class FakePMD:
    def __init__(self, x, y=None, num_boot=1000, skip_posthoc=False):
        base = np.linspace(-1.0, 1.0, x.shape[1], dtype=float)
        arr = np.vstack([base + float(i) for i in range(x.shape[0])])
        self.z_scores = pd.DataFrame(arr, index=x.index, columns=x.columns)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gene_level_defaults_and_opt_out_preserves_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "pmd", FakePMD)

    input_path = tmp_path / "counts.tsv"
    model_matrix_path = tmp_path / "mm.tsv"

    counts = pd.DataFrame(
        {
            "gene": ["A", "A", "B"],
            "s1": [10, 11, 12],
            "s2": [13, 14, 15],
            "s3": [16, 17, 18],
            "s4": [19, 20, 21],
        },
        index=["g1", "g2", "g3"],
    )
    counts.to_csv(input_path, sep="\t")

    mm = pd.DataFrame(
        {"treatment": [0.0, 0.0, 1.0, 1.0]},
        index=["s1", "s2", "s3", "s4"],
    )
    mm.to_csv(model_matrix_path, sep="\t")

    out_default = tmp_path / "out_default"
    out_optout = tmp_path / "out_optout"

    mod.pmd_std_res_and_stats(
        str(input_path),
        str(out_default),
        model_matrix_file=str(model_matrix_path),
        p_combine_idx=None,
        in_annotation_cols=2,
        pre_regress_vars=None,
        n_boot=2,
        seed=1,
        file_sep="tsv",
    )

    mod.pmd_std_res_and_stats(
        str(input_path),
        str(out_optout),
        model_matrix_file=str(model_matrix_path),
        p_combine_idx=None,
        in_annotation_cols=2,
        pre_regress_vars=None,
        n_boot=2,
        seed=1,
        file_sep="tsv",
        gene_level=False,
        gene_figures=False,
    )

    baseline_files = [
        "PMD_std_res.tsv",
        "PMD_std_res_stats.tsv",
        "PMD_std_res_stats_resids.tsv",
    ]
    for name in baseline_files:
        assert (out_default / name).is_file()
        assert (out_optout / name).is_file()
        assert _sha256(out_default / name) == _sha256(out_optout / name)

    assert (out_default / "gene_level" / "PMD_std_res_gene_meta.tsv").is_file()
    assert (out_default / "gene_level" / "PMD_std_res_gene_lmm_selection.tsv").is_file()
    assert (out_default / "gene_level" / "PMD_std_res_gene_lmm.tsv").is_file()
    assert (out_default / "gene_level" / "PMD_std_res_gene_lmm_full.tsv").is_file()
    assert (out_default / "gene_level" / "PMD_std_res_gene_qc.tsv").is_file()
    assert (out_default / "gene_level" / "PMD_std_res_gene_flagged.tsv").is_file()
    assert (out_default / "gene_level" / "PMD_std_res_gene_mixture.tsv").is_file()
    assert (out_default / "gene_level" / "PMD_std_res_gene_mixture_guides.tsv").is_file()
    assert (out_default / "gene_level" / "PMD_std_res_gene_tmeta.tsv").is_file()
    assert (out_default / "gene_level" / "PMD_std_res_gene_tmeta_guides.tsv").is_file()
    assert not (out_optout / "gene_level").exists()
