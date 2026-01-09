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

    monkeypatch.setattr(gl, "_fit_per_guide_ols", fake_fit_per_guide_ols)

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

    assert out["p_adj"].between(0.0, 1.0).all()
