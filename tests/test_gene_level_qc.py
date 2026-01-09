import numpy as np
import pandas as pd

import guide_pmd.gene_level_qc as qc


def test_compute_gene_qc_metrics(monkeypatch):
    response = pd.DataFrame(
        {"s1": [0.0, 0.0, 0.0, 0.0], "s2": [0.0, 0.0, 0.0, 0.0]},
        index=["g1", "g2", "g3", "g4"],
    )
    annotation_table = pd.DataFrame({"gene": ["A", "A", "A", "B"]}, index=["g1", "g2", "g3", "g4"])
    model_matrix = pd.DataFrame({"treatment": [0.0, 1.0]}, index=["s1", "s2"])

    residuals = pd.DataFrame(
        {"s1": [0.0, 5.0, -1.0, 0.5], "s2": [0.0, 0.0, 0.0, -0.5]},
        index=["g1", "g2", "g3", "g4"],
    )

    def fake_fit_per_guide_ols(*_args, **_kwargs):
        return pd.DataFrame(
            [
                {"guide_id": "g1", "focal_var": "treatment", "beta": 2.0, "se": 1.0, "t": 2.0, "p": 0.1},
                {"guide_id": "g2", "focal_var": "treatment", "beta": -1.0, "se": 1.0, "t": -1.0, "p": 0.3},
                {"guide_id": "g3", "focal_var": "treatment", "beta": 1.0, "se": 1.0, "t": 1.0, "p": 0.3},
                {"guide_id": "g4", "focal_var": "treatment", "beta": 0.5, "se": 1.0, "t": 0.5, "p": 0.6},
            ]
        )

    monkeypatch.setattr(qc, "fit_per_guide_ols", fake_fit_per_guide_ols)

    out = qc.compute_gene_qc(
        response,
        annotation_table,
        model_matrix,
        focal_vars=["treatment"],
        gene_id_col=1,
        add_intercept=True,
        residual_matrix=residuals,
    )

    assert out["gene_id"].tolist() == ["A", "B"]
    assert out["focal_var"].unique().tolist() == ["treatment"]

    row_a = out.loc[out["gene_id"] == "A"].iloc[0]
    assert row_a["m_guides_total"] == 3.0
    assert row_a["m_guides_used"] == 3.0
    assert np.isclose(row_a["beta_median"], 1.0)
    assert row_a["majority_sign"] == 1.0
    assert np.isclose(row_a["sign_agreement"], 2.0 / 3.0)
    assert np.isclose(row_a["frac_opposite_sign"], 1.0 / 3.0)
    assert row_a["max_abs_beta"] == 2.0
    assert row_a["max_abs_z"] == 2.0
    assert row_a["max_abs_resid"] == 5.0
