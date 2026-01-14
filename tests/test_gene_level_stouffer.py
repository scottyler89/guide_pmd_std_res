import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control as fdr
from scipy.stats import t as student_t

from guide_pmd.gene_level_stouffer import compute_gene_stouffer


def test_compute_gene_stouffer_combines_t_values_and_fdr():
    response = pd.DataFrame(
        {
            "s1": [0.0, 0.0, 0.0],
            "s2": [0.0, 0.0, 0.0],
            "s3": [0.0, 0.0, 0.0],
            "s4": [0.0, 0.0, 0.0],
        },
        index=["g1", "g2", "g3"],
    )
    annotation = pd.DataFrame(
        {
            "gene_id": ["A", "A", "B"],
        },
        index=["g1", "g2", "g3"],
    )
    model_matrix = pd.DataFrame(
        {"treatment": [0.0, 0.0, 1.0, 1.0]},
        index=["s1", "s2", "s3", "s4"],
    )
    per_guide_ols = pd.DataFrame(
        {
            "guide_id": ["g1", "g2", "g3"],
            "focal_var": ["treatment", "treatment", "treatment"],
            "t": [1.0, 3.0, -2.0],
        }
    )

    out = compute_gene_stouffer(
        response,
        annotation,
        model_matrix,
        focal_vars=["treatment"],
        gene_id_col=1,
        add_intercept=True,
        per_guide_ols=per_guide_ols,
    )
    assert set(out.columns) >= {
        "gene_id",
        "focal_var",
        "stouffer_t",
        "p",
        "p_adj",
        "m_guides_total",
        "m_guides_used",
        "df_resid",
    }
    assert out.shape[0] == 2

    out = out.sort_values(["gene_id"], kind="mergesort").reset_index(drop=True)
    df_resid = int(out.loc[0, "df_resid"])
    assert df_resid == 2

    expected_t = {
        "A": float((1.0 + 3.0) / np.sqrt(2.0)),
        "B": float((-2.0) / np.sqrt(1.0)),
    }
    expected_p = {g: float(2 * student_t.sf(abs(t), df=df_resid)) for g, t in expected_t.items()}
    assert np.isclose(out.loc[out["gene_id"] == "A", "stouffer_t"].item(), expected_t["A"])
    assert np.isclose(out.loc[out["gene_id"] == "B", "stouffer_t"].item(), expected_t["B"])
    assert np.isclose(out.loc[out["gene_id"] == "A", "p"].item(), expected_p["A"])
    assert np.isclose(out.loc[out["gene_id"] == "B", "p"].item(), expected_p["B"])

    expected_adj = fdr(out["p"].to_numpy(dtype=float))
    assert np.allclose(out["p_adj"].to_numpy(dtype=float), expected_adj)

