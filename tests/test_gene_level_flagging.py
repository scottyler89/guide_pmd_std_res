import pandas as pd

from guide_pmd.gene_level_flagging import GeneFlaggingConfig
from guide_pmd.gene_level_flagging import compute_gene_flag_table


def test_compute_gene_flag_table_flags_and_reasons():
    meta = pd.DataFrame(
        [
            {"gene_id": "A", "focal_var": "treatment", "Q_p_adj": 0.01, "m_guides_used": 4},
            {"gene_id": "B", "focal_var": "treatment", "Q_p_adj": 0.9, "m_guides_used": 4},
            {"gene_id": "C", "focal_var": "treatment", "Q_p_adj": 0.9, "m_guides_used": 2},
        ]
    )
    qc = pd.DataFrame(
        [
            # A: flagged by het_q
            {
                "gene_id": "A",
                "focal_var": "treatment",
                "m_guides_used": 4,
                "frac_opposite_sign": 0.0,
                "max_abs_z": 1.0,
                "max_abs_resid": 0.0,
            },
            # B: flagged by opposite_sign + max_abs_z
            {
                "gene_id": "B",
                "focal_var": "treatment",
                "m_guides_used": 4,
                "frac_opposite_sign": 0.5,
                "max_abs_z": 4.0,
                "max_abs_resid": 0.0,
            },
            # C: not enough guides (min_guides=3) => not flagged even though max_abs_z is large
            {
                "gene_id": "C",
                "focal_var": "treatment",
                "m_guides_used": 2,
                "frac_opposite_sign": 0.0,
                "max_abs_z": 10.0,
                "max_abs_resid": 0.0,
            },
        ]
    )

    cfg = GeneFlaggingConfig(
        scope="all",
        het_q_max=0.1,
        frac_opposite_min=0.25,
        max_abs_z_min=3.0,
        max_abs_resid_min=None,
        min_guides=3,
    )
    out = compute_gene_flag_table(meta, qc, config=cfg)

    a = out.loc[out["gene_id"] == "A"].iloc[0]
    assert bool(a["flagged"]) is True
    assert "het_q" in str(a["flag_reason"])

    b = out.loc[out["gene_id"] == "B"].iloc[0]
    assert bool(b["flagged"]) is True
    reason = str(b["flag_reason"])
    assert "opposite_sign" in reason
    assert "max_abs_z" in reason

    c = out.loc[out["gene_id"] == "C"].iloc[0]
    assert bool(c["flagged"]) is False
    assert str(c["flag_reason"]) == ""


def test_compute_gene_flag_table_scope_none():
    meta = pd.DataFrame([{"gene_id": "A", "focal_var": "treatment", "Q_p_adj": 0.01, "m_guides_used": 4}])
    qc = pd.DataFrame(
        [
            {
                "gene_id": "A",
                "focal_var": "treatment",
                "m_guides_used": 4,
                "frac_opposite_sign": 0.0,
                "max_abs_z": 1.0,
                "max_abs_resid": 0.0,
            }
        ]
    )
    cfg = GeneFlaggingConfig(scope="none")
    out = compute_gene_flag_table(meta, qc, config=cfg)
    assert out.columns.tolist() == ["gene_id", "focal_var", "flagged", "flag_reason"]
    assert bool(out.loc[0, "flagged"]) is False

