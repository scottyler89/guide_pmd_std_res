import numpy as np
import pandas as pd
import pytest

import guide_pmd.pmd_std_res_guide_counts as mod


class FakePMD:
    def __init__(self, x, y=None, num_boot=1000, skip_posthoc=False):
        self.z_scores = pd.DataFrame(
            np.zeros((x.shape[0], x.shape[1]), dtype=float),
            index=getattr(x, "index", None),
            columns=getattr(x, "columns", None),
        )


def test_get_pmd_std_res_honors_sep(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "pmd", FakePMD)

    input_path = tmp_path / "counts.csv"
    df = pd.DataFrame(
        {
            "gene": ["A", "B"],
            "s1": [10, 11],
            "s2": [12, 13],
        },
        index=["g1", "g2"],
    )
    df.to_csv(input_path, sep=",")

    std_res, ann = mod.get_pmd_std_res(str(input_path), in_annotation_cols=2, n_boot=1, seed=1, sep=",")

    assert list(ann.columns) == ["gene"]
    assert list(std_res.columns) == ["s1", "s2"]
    assert std_res.shape == (2, 2)


def test_pmd_std_res_and_stats_returns_nones_without_model_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "pmd", FakePMD)

    input_path = tmp_path / "counts.tsv"
    out_dir = tmp_path / "out"
    df = pd.DataFrame(
        {
            "gene": ["A", "B"],
            "s1": [10, 11],
            "s2": [12, 13],
        },
        index=["g1", "g2"],
    )
    df.to_csv(input_path, sep="\t")

    std_res, stats_df, resids_df, comb_stats = mod.pmd_std_res_and_stats(
        str(input_path),
        str(out_dir),
        model_matrix_file=None,
        p_combine_idx=None,
        in_annotation_cols=2,
        pre_regress_vars=None,
        n_boot=1,
        seed=1,
        file_sep="tsv",
    )

    assert std_res.shape == (2, 2)
    assert stats_df is None
    assert resids_df is None
    assert comb_stats is None
    assert (out_dir / "PMD_std_res.tsv").is_file()


def test_pmd_std_res_and_stats_rejects_invalid_file_sep(tmp_path):
    with pytest.raises(ValueError, match="file_sep"):
        mod.pmd_std_res_and_stats(
            str(tmp_path / "missing.tsv"),
            str(tmp_path / "out"),
            model_matrix_file=None,
            file_sep="nope",
        )


def test_run_glm_analysis_all_zero_variance_is_safe():
    normalized_matrix = pd.DataFrame(
        {"s1": [1.0, 1.0], "s2": [1.0, 1.0], "s3": [1.0, 1.0]},
        index=["f1", "f2"],
    )
    model_matrix = pd.DataFrame({"x": [0.0, 1.0, 0.0]}, index=["s1", "s2", "s3"])

    stats, resids = mod.run_glm_analysis(normalized_matrix, model_matrix)

    assert list(stats.index) == ["f1", "f2"]
    assert list(resids.index) == ["f1", "f2"]
    assert list(resids.columns) == ["s1", "s2", "s3"]
    assert stats.isna().all().all()
    assert resids.isna().all().all()

