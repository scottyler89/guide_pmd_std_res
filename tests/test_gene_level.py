import numpy as np
import pandas as pd

import guide_pmd.gene_level as gl


def test_dersimonian_laird_basic_case():
    beta = np.array([0.2, 0.0, 0.4])
    se = np.array([0.1, 0.1, 0.1])

    out = gl._dersimonian_laird(beta, se)

    assert out["m_guides_used"] == 3.0
    assert np.isclose(out["theta"], 0.2)
    assert np.isclose(out["tau2"], 0.03)
    assert np.isclose(out["I2"], 0.75)
    assert np.isclose(out["se_theta"], np.sqrt(1.0 / 75.0))
    assert 0.0 < out["p"] < 1.0


def test_dersimonian_laird_single_guide_passthrough():
    beta = np.array([0.5])
    se = np.array([0.2])

    out = gl._dersimonian_laird(beta, se)

    assert out["m_guides_used"] == 1.0
    assert out["tau2"] == 0.0
    assert out["tau"] == 0.0
    assert np.isclose(out["theta"], 0.5)
    assert np.isclose(out["se_theta"], 0.2)
    assert 0.0 < out["p"] < 1.0


def test_dersimonian_laird_all_invalid_returns_nans():
    beta = np.array([1.0, 2.0])
    se = np.array([0.0, np.nan])

    out = gl._dersimonian_laird(beta, se)

    assert out["m_guides_used"] == 0.0
    assert np.isnan(out["theta"])


def test_compute_gene_meta_groups_and_fdr(monkeypatch):
    response = pd.DataFrame(
        {"s1": [0.0, 0.0, 0.0], "s2": [0.0, 0.0, 0.0]},
        index=["g1", "g2", "g3"],
    )
    annotation_table = pd.DataFrame({"gene": ["A", "A", "B"]}, index=["g1", "g2", "g3"])
    model_matrix = pd.DataFrame({"treatment": [0.0, 1.0]}, index=["s1", "s2"])

    def fake_fit_per_guide_ols(*_args, **_kwargs):
        return pd.DataFrame(
            [
                {"guide_id": "g1", "focal_var": "treatment", "beta": 1.0, "se": 1.0, "t": 1.0, "p": 0.32},
                {"guide_id": "g2", "focal_var": "treatment", "beta": 1.0, "se": 1.0, "t": 1.0, "p": 0.32},
                {"guide_id": "g3", "focal_var": "treatment", "beta": 0.0, "se": 1.0, "t": 0.0, "p": 1.0},
            ]
        )

    monkeypatch.setattr(gl, "fit_per_guide_ols", fake_fit_per_guide_ols)

    out = gl.compute_gene_meta(
        response,
        annotation_table,
        model_matrix,
        focal_vars=["treatment"],
        gene_id_col=1,
        add_intercept=True,
    )

    assert out.columns.tolist() == [
        "gene_id",
        "focal_var",
        "theta",
        "se_theta",
        "z",
        "p",
        "p_adj",
        "tau",
        "tau2",
        "Q",
        "Q_df",
        "Q_p",
        "Q_p_adj",
        "I2",
        "m_guides_total",
        "m_guides_used",
        "sign_agreement",
    ]
    assert out["gene_id"].tolist() == ["A", "B"]
    assert out["focal_var"].unique().tolist() == ["treatment"]

    row_a = out.loc[out["gene_id"] == "A"].iloc[0]
    assert np.isclose(row_a["theta"], 1.0)
    assert row_a["sign_agreement"] == 1.0
    assert np.isclose(row_a["Q_df"], 1.0)
    assert np.isclose(row_a["Q_p"], 1.0)

    assert out["p_adj"].between(0.0, 1.0).all()
    assert out["Q_p_adj"].between(0.0, 1.0).all()


def test_compute_gene_meta_Q_p_edge_cases_m0_m1_m2():
    # This test exercises heterogeneity Q-stat edge cases:
    # - m_used=0 (all invalid SEs) -> Q_p is NaN; Q_p_adj is 1.0 after nan-aware FDR.
    # - m_used=1 -> Q_df=0 and Q_p is NaN; Q_p_adj is 1.0.
    # - m_used=2 -> Q is finite; Q_p is finite (and 1.0 when betas match).
    response = pd.DataFrame(
        {"s1": [0.0, 0.0, 0.0, 0.0, 0.0], "s2": [0.0, 0.0, 0.0, 0.0, 0.0]},
        index=["g0a", "g0b", "g1", "g2", "g2_b"],
    )
    annotation_table = pd.DataFrame(
        {"gene": ["G0", "G0", "G1", "G2", "G2"]},
        index=["g0a", "g0b", "g1", "g2", "g2_b"],
    )
    model_matrix = pd.DataFrame({"treatment": [0.0, 1.0]}, index=["s1", "s2"])

    per_guide = pd.DataFrame(
        [
            # G0: all invalid (se=0) => m_used=0
            {"guide_id": "g0a", "focal_var": "treatment", "beta": 0.1, "se": 0.0},
            {"guide_id": "g0b", "focal_var": "treatment", "beta": -0.1, "se": 0.0},
            # G1: single guide => m_used=1
            {"guide_id": "g1", "focal_var": "treatment", "beta": 0.2, "se": 0.1},
            # G2: two guides, identical betas => Q=0 => Q_p=1
            {"guide_id": "g2", "focal_var": "treatment", "beta": 0.3, "se": 0.1},
            {"guide_id": "g2_b", "focal_var": "treatment", "beta": 0.3, "se": 0.1},
        ]
    )

    out = gl.compute_gene_meta(
        response,
        annotation_table,
        model_matrix,
        focal_vars=["treatment"],
        gene_id_col=1,
        add_intercept=True,
        per_guide_ols=per_guide,
    )

    row_g0 = out.loc[out["gene_id"] == "G0"].iloc[0]
    assert row_g0["m_guides_total"] == 2.0
    assert row_g0["m_guides_used"] == 0.0
    assert np.isnan(row_g0["Q_p"])
    assert row_g0["Q_p_adj"] == 1.0

    row_g1 = out.loc[out["gene_id"] == "G1"].iloc[0]
    assert row_g1["m_guides_total"] == 1.0
    assert row_g1["m_guides_used"] == 1.0
    assert row_g1["Q_df"] == 0.0
    assert np.isnan(row_g1["Q_p"])
    assert row_g1["Q_p_adj"] == 1.0

    row_g2 = out.loc[out["gene_id"] == "G2"].iloc[0]
    assert row_g2["m_guides_total"] == 2.0
    assert row_g2["m_guides_used"] == 2.0
    assert row_g2["Q_df"] == 1.0
    assert np.isfinite(row_g2["Q"])
    assert np.isfinite(row_g2["Q_p"])
    assert np.isclose(row_g2["Q"], 0.0)
    assert np.isclose(row_g2["Q_p"], 1.0)
