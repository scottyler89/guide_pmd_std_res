from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import guide_pmd.pmd_std_res_guide_counts as mod


class FakePMD:
    def __init__(self, x, y=None, num_boot=1000, skip_posthoc=False):
        base = np.linspace(-1.0, 1.0, x.shape[1], dtype=float)
        arr = np.vstack([base + float(i) for i in range(x.shape[0])])
        self.z_scores = pd.DataFrame(arr, index=x.index, columns=x.columns)


def test_gene_level_figures_are_written(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
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

    out_dir = tmp_path / "out"
    mod.pmd_std_res_and_stats(
        str(input_path),
        str(out_dir),
        model_matrix_file=str(model_matrix_path),
        p_combine_idx=None,
        in_annotation_cols=2,
        pre_regress_vars=None,
        n_boot=2,
        seed=1,
        file_sep="tsv",
        gene_level=True,
        focal_vars=["treatment"],
        gene_methods=["meta", "qc"],
        gene_figures=True,
    )

    figures_dir = out_dir / "gene_level_figures"
    assert figures_dir.is_dir()

    pngs = sorted(p.name for p in figures_dir.glob("*.png"))
    assert pngs
    for name in pngs:
        assert (figures_dir / name).stat().st_size > 0
