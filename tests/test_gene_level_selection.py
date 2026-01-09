import pandas as pd

import guide_pmd.gene_level_selection as sel


def test_compute_gene_lmm_feasibility_gates():
    response = pd.DataFrame(
        {
            "s1": [0.0, 0.0, 0.0],
            "s2": [0.0, 0.0, 1.0],
        },
        index=["g1", "g2", "g3"],
    )
    annotation_table = pd.DataFrame({"gene": ["A", "A", "B", "C"]}, index=["g1", "g2", "g3", "g4"])
    model_matrix = pd.DataFrame({"treatment": [0.0, 1.0]}, index=["s1", "s2"])

    out = sel.compute_gene_lmm_feasibility(
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
        "m_guides_total",
        "feasible",
        "skip_reason",
    ]
    assert out["gene_id"].tolist() == ["A", "B", "C"]
    assert out["focal_var"].unique().tolist() == ["treatment"]

    row_a = out.loc[out["gene_id"] == "A"].iloc[0]
    assert row_a["m_guides_total"] == 2.0
    assert bool(row_a["feasible"]) is False
    assert row_a["skip_reason"] == "degenerate_response"

    row_b = out.loc[out["gene_id"] == "B"].iloc[0]
    assert row_b["m_guides_total"] == 1.0
    assert bool(row_b["feasible"]) is False
    assert row_b["skip_reason"] == "insufficient_guides"

    row_c = out.loc[out["gene_id"] == "C"].iloc[0]
    assert row_c["m_guides_total"] == 0.0
    assert bool(row_c["feasible"]) is False
    assert row_c["skip_reason"] == "no_guides_in_response"


def test_compute_gene_lmm_selection_meta_or_het_and_audit_and_cap():
    feasibility = pd.DataFrame(
        [
            {"gene_id": "A", "focal_var": "treat", "m_guides_total": 4.0, "feasible": True, "skip_reason": ""},
            {"gene_id": "B", "focal_var": "treat", "m_guides_total": 4.0, "feasible": True, "skip_reason": ""},
            {"gene_id": "C", "focal_var": "treat", "m_guides_total": 4.0, "feasible": True, "skip_reason": ""},
            {"gene_id": "D", "focal_var": "treat", "m_guides_total": 4.0, "feasible": True, "skip_reason": ""},
        ]
    )
    meta = pd.DataFrame(
        [
            {
                "gene_id": "A",
                "focal_var": "treat",
                "theta": 1.0,
                "se_theta": 0.1,
                "p": 1e-6,
                "p_adj": 0.01,
                "Q_p": 0.9,
                "Q_p_adj": 0.9,
                "m_guides_used": 4.0,
            },
            {
                "gene_id": "B",
                "focal_var": "treat",
                "theta": 0.5,
                "se_theta": 0.2,
                "p": 0.02,
                "p_adj": 0.2,
                "Q_p": 0.01,
                "Q_p_adj": 0.05,
                "m_guides_used": 4.0,
            },
            {
                "gene_id": "C",
                "focal_var": "treat",
                "theta": 0.0,
                "se_theta": 0.2,
                "p": 0.8,
                "p_adj": 0.8,
                "Q_p": 0.9,
                "Q_p_adj": 0.9,
                "m_guides_used": 4.0,
            },
            {
                "gene_id": "D",
                "focal_var": "treat",
                "theta": 0.0,
                "se_theta": 0.2,
                "p": 0.9,
                "p_adj": 0.9,
                "Q_p": 0.9,
                "Q_p_adj": 0.9,
                "m_guides_used": 4.0,
            },
        ]
    )
    config = sel.GeneLmmSelectionConfig(
        scope="meta_or_het_fdr",
        q_meta=0.1,
        q_het=0.1,
        audit_n=1,
        audit_seed=7,
        max_genes_per_focal_var=2,
    )

    out = sel.compute_gene_lmm_selection(meta, feasibility, config=config)

    assert out["selected"].dtype == bool
    assert set(out["gene_id"].tolist()) == {"A", "B", "C", "D"}

    # Criteria selections.
    row_a = out.loc[out["gene_id"] == "A"].iloc[0]
    assert bool(row_a["selected"]) is True
    assert row_a["selection_reason"] == "meta_fdr"

    row_b = out.loc[out["gene_id"] == "B"].iloc[0]
    assert bool(row_b["selected"]) is True
    assert row_b["selection_reason"] == "het_fdr"

    # With cap=2, audit must be dropped.
    audit_rows = out.loc[out["selection_reason"].isin(["audit", "cap_dropped"])]
    assert audit_rows.shape[0] == 1
    assert audit_rows.iloc[0]["selection_reason"] == "cap_dropped"
    assert bool(audit_rows.iloc[0]["selected"]) is False

    assert out.loc[out["selected"], "gene_id"].tolist() == ["A", "B"]
