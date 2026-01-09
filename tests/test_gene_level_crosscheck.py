import numpy as np
import pandas as pd

import guide_pmd.gene_level as meta
import guide_pmd.gene_level_lmm as lmm


def _make_synthetic(theta_a: float, theta_b: float, *, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_samples = 30
    sample_ids = [f"s{i}" for i in range(n_samples)]
    treatment = np.array([0.0] * (n_samples // 2) + [1.0] * (n_samples - n_samples // 2), dtype=float)
    model_matrix = pd.DataFrame({"treatment": treatment}, index=sample_ids)

    guides = []
    genes = []
    y_rows = []

    def add_gene(gene_id: str, theta: float, n_guides: int):
        nonlocal guides, genes, y_rows
        for j in range(n_guides):
            guide_id = f"{gene_id}_g{j+1}"
            intercept = rng.normal(0.0, 0.2)
            noise = rng.normal(0.0, 0.2, size=n_samples)
            y = intercept + theta * treatment + noise
            guides.append(guide_id)
            genes.append(gene_id)
            y_rows.append(y)

    add_gene("A", theta=theta_a, n_guides=4)
    add_gene("B", theta=theta_b, n_guides=4)

    response_matrix = pd.DataFrame(y_rows, index=guides, columns=sample_ids)
    annotation_table = pd.DataFrame({"gene": genes}, index=guides)
    return response_matrix, annotation_table, model_matrix


def test_plan_a_and_plan_b_agree_on_simple_synthetic():
    response, ann, mm = _make_synthetic(theta_a=1.0, theta_b=0.0)

    out_meta = meta.compute_gene_meta(
        response,
        ann,
        mm,
        focal_vars=["treatment"],
        gene_id_col=1,
        add_intercept=True,
    )
    out_lmm = lmm.compute_gene_lmm(
        response,
        ann,
        mm,
        focal_vars=["treatment"],
        gene_id_col=1,
        add_intercept=True,
        allow_random_slope=False,
        fallback_to_meta=False,
        max_iter=200,
    )

    m_a = out_meta.loc[out_meta["gene_id"] == "A"].iloc[0]
    l_a = out_lmm.loc[out_lmm["gene_id"] == "A"].iloc[0]
    assert np.sign(m_a["theta"]) == np.sign(l_a["theta"])
    assert abs(float(m_a["theta"]) - float(l_a["theta"])) < 0.25
    assert float(m_a["p"]) < 1e-4
    assert float(l_a["p_primary"]) < 1e-4

    m_b = out_meta.loc[out_meta["gene_id"] == "B"].iloc[0]
    l_b = out_lmm.loc[out_lmm["gene_id"] == "B"].iloc[0]
    assert abs(float(m_b["theta"])) < 0.25
    assert abs(float(l_b["theta"])) < 0.25
    assert float(m_b["p"]) > 0.01
    assert float(l_b["p_primary"]) > 0.01

