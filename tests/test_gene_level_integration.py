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


def test_gene_level_meta_is_opt_in_and_baseline_is_unchanged(tmp_path, monkeypatch):
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

    out_baseline = tmp_path / "out_baseline"
    out_gene = tmp_path / "out_gene"

    mod.pmd_std_res_and_stats(
        str(input_path),
        str(out_baseline),
        model_matrix_file=str(model_matrix_path),
        p_combine_idx=None,
        in_annotation_cols=2,
        pre_regress_vars=None,
        n_boot=1,
        seed=1,
        file_sep="tsv",
        gene_level=False,
    )

    mod.pmd_std_res_and_stats(
        str(input_path),
        str(out_gene),
        model_matrix_file=str(model_matrix_path),
        p_combine_idx=None,
        in_annotation_cols=2,
        pre_regress_vars=None,
        n_boot=1,
        seed=1,
        file_sep="tsv",
        gene_level=True,
        focal_vars=["treatment"],
        gene_methods=["meta"],
        gene_id_col=1,
    )

    baseline_files = [
        "PMD_std_res.tsv",
        "PMD_std_res_stats.tsv",
        "PMD_std_res_stats_resids.tsv",
    ]
    for name in baseline_files:
        assert (out_baseline / name).is_file()
        assert (out_gene / name).is_file()
        assert _sha256(out_baseline / name) == _sha256(out_gene / name)

    assert not (out_baseline / "PMD_std_res_gene_meta.tsv").exists()
    assert (out_gene / "PMD_std_res_gene_meta.tsv").is_file()

