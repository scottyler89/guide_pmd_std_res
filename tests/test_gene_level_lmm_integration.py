import hashlib

import numpy as np
import pandas as pd
import pytest

import guide_pmd.pmd_std_res_guide_counts as mod


class FakePMD:
    def __init__(self, x, y=None, num_boot=1000, skip_posthoc=False):
        base = np.linspace(-1.0, 1.0, x.shape[1], dtype=float)
        arr = np.vstack([base + float(i) for i in range(x.shape[0])])
        self.z_scores = pd.DataFrame(arr, index=x.index, columns=x.columns)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gene_level_lmm_wiring_does_not_change_baseline_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "pmd", FakePMD)

    import guide_pmd.gene_level_lmm as glmm

    def fake_compute_gene_lmm(*_args, **_kwargs):
        return pd.DataFrame(
            [
                {
                    "gene_id": "A",
                    "focal_var": "treatment",
                    "method": "lmm",
                    "model": "ri",
                    "theta": 0.0,
                    "se_theta": 1.0,
                    "wald_z": 0.0,
                    "wald_p": 1.0,
                    "wald_ok": True,
                    "wald_p_adj": 1.0,
                    "lrt_stat": 0.0,
                    "lrt_p": 1.0,
                    "lrt_ok": True,
                    "lrt_p_adj": 1.0,
                    "sigma_alpha": 0.0,
                    "tau": 0.0,
                    "converged_full": True,
                    "converged_null": True,
                    "m_guides_total": 1.0,
                    "m_guides_used": 1.0,
                    "n_samples": 1.0,
                    "n_obs": 1.0,
                    "fit_error": "",
                }
            ]
        )

    monkeypatch.setattr(glmm, "compute_gene_lmm", fake_compute_gene_lmm)

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
        n_boot=2,
        seed=1,
        file_sep="tsv",
    )

    mod.pmd_std_res_and_stats(
        str(input_path),
        str(out_gene),
        model_matrix_file=str(model_matrix_path),
        p_combine_idx=None,
        in_annotation_cols=2,
        pre_regress_vars=None,
        n_boot=2,
        seed=1,
        file_sep="tsv",
        gene_level=True,
        focal_vars=["treatment"],
        gene_methods=["lmm"],
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

    gene_level_dir = out_gene / "gene_level"
    assert not (gene_level_dir / "PMD_std_res_gene_meta.tsv").exists()
    assert (gene_level_dir / "PMD_std_res_gene_lmm.tsv").is_file()
    assert (gene_level_dir / "PMD_std_res_gene_lmm_full.tsv").is_file()


def test_gene_level_lmm_resume_skips_checkpointed_tasks(tmp_path, monkeypatch):
    import guide_pmd.gene_level_lmm as glmm

    input_path = tmp_path / "counts.tsv"
    model_matrix_path = tmp_path / "mm.tsv"
    std_res_path = tmp_path / "PMD_std_res.tsv"

    counts = pd.DataFrame(
        {
            "gene": ["A", "A", "B", "B"],
            "s1": [10, 11, 12, 13],
            "s2": [14, 15, 16, 17],
            "s3": [18, 19, 20, 21],
            "s4": [22, 23, 24, 25],
        },
        index=["g1", "g2", "g3", "g4"],
    )
    counts.to_csv(input_path, sep="\t")

    mm = pd.DataFrame(
        {"treatment": [0.0, 0.0, 1.0, 1.0]},
        index=["s1", "s2", "s3", "s4"],
    )
    mm.to_csv(model_matrix_path, sep="\t")

    std_res = pd.DataFrame(
        np.arange(16, dtype=float).reshape(4, 4),
        index=counts.index,
        columns=mm.index,
    )
    std_res.to_csv(std_res_path, sep="\t")

    out_dir = tmp_path / "out_resume"

    first_key = ("A", "treatment")
    second_key = ("B", "treatment")

    def fake_iter_gene_lmm_rows(*_args, **_kwargs):
        yield {"gene_id": first_key[0], "focal_var": first_key[1], "method": "lmm", "wald_p": 1.0, "lrt_p": 1.0}
        raise RuntimeError("stop early (test)")

    monkeypatch.setattr(glmm, "iter_gene_lmm_rows", fake_iter_gene_lmm_rows)

    with pytest.raises(RuntimeError):
        mod.pmd_std_res_and_stats(
            str(input_path),
            str(out_dir),
            model_matrix_file=str(model_matrix_path),
            p_combine_idx=None,
            in_annotation_cols=2,
            pre_regress_vars=None,
            file_sep="tsv",
            std_res_file=str(std_res_path),
            gene_level=True,
            gene_figures=False,
            gene_methods=["lmm"],
            focal_vars=["treatment"],
            gene_id_col=1,
            gene_lmm_scope="all",
            gene_lmm_resume=True,
            gene_lmm_checkpoint_every=1,
        )

    gene_level_dir = out_dir / "gene_level"
    assert (gene_level_dir / "PMD_std_res_gene_lmm.partial.tsv").is_file()
    assert (gene_level_dir / "PMD_std_res_gene_lmm.partial.meta.json").is_file()

    def fake_iter_gene_lmm_rows_resume(*_args, **kwargs):
        assert kwargs.get("skip_keys") == {first_key}
        yield {"gene_id": second_key[0], "focal_var": second_key[1], "method": "lmm", "wald_p": 1.0, "lrt_p": 1.0}

    monkeypatch.setattr(glmm, "iter_gene_lmm_rows", fake_iter_gene_lmm_rows_resume)

    mod.pmd_std_res_and_stats(
        str(input_path),
        str(out_dir),
        model_matrix_file=str(model_matrix_path),
        p_combine_idx=None,
        in_annotation_cols=2,
        pre_regress_vars=None,
        file_sep="tsv",
        std_res_file=str(std_res_path),
        gene_level=True,
        gene_figures=False,
        gene_methods=["lmm"],
        focal_vars=["treatment"],
        gene_id_col=1,
        gene_lmm_scope="all",
        gene_lmm_resume=True,
        gene_lmm_checkpoint_every=1,
    )

    gene_lmm = pd.read_csv(gene_level_dir / "PMD_std_res_gene_lmm.tsv", sep="\t")
    assert set(zip(gene_lmm["gene_id"].tolist(), gene_lmm["focal_var"].tolist(), strict=True)) == {first_key, second_key}
