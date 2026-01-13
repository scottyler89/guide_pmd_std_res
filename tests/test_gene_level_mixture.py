import numpy as np
import pandas as pd

import guide_pmd.gene_level_mixture as mod


def test_compute_gene_mixture_downweights_outlier():
    per_guide_ols = pd.DataFrame(
        [
            {"guide_id": "g1", "focal_var": "treatment", "beta": 1.0, "se": 0.1},
            {"guide_id": "g2", "focal_var": "treatment", "beta": 1.0, "se": 0.1},
            {"guide_id": "g3", "focal_var": "treatment", "beta": 1.0, "se": 0.1},
            {"guide_id": "g4", "focal_var": "treatment", "beta": 10.0, "se": 0.1},
        ]
    )
    annotation_table = pd.DataFrame({"gene": ["A", "A", "A", "A"]}, index=["g1", "g2", "g3", "g4"])
    meta_results = pd.DataFrame([{"gene_id": "A", "focal_var": "treatment", "theta": 1.0, "tau2": 0.0}])
    flag_table = pd.DataFrame([{"gene_id": "A", "focal_var": "treatment", "flagged": True, "flag_reason": "het_q"}])

    config = mod.GeneMixtureConfig(scope="flagged", min_guides=3, bad_scale=10.0)
    gene, guides = mod.compute_gene_mixture(
        per_guide_ols,
        annotation_table,
        focal_vars=["treatment"],
        gene_id_col=1,
        meta_results=meta_results,
        flag_table=flag_table,
        config=config,
    )

    assert gene.shape[0] == 1
    row = gene.iloc[0]
    assert bool(row["fit_ok"]) is True
    assert float(row["tau2_meta"]) == 0.0
    assert float(row["sigma_bad2"]) > 0.0
    assert np.isfinite(float(row["theta"]))
    assert np.isfinite(float(row["se_theta"]))
    assert np.isfinite(float(row["wald_z"]))
    assert np.isfinite(float(row["wald_p"]))
    assert float(row["wald_p_adj"]) == float(row["wald_p"])

    assert guides.shape[0] == 4
    outlier_w = float(guides.loc[guides["guide_id"] == "g4", "mixture_prob_good"].iloc[0])
    inlier_w = guides.loc[guides["guide_id"] != "g4", "mixture_prob_good"].to_numpy(dtype=float)
    assert np.isfinite(outlier_w)
    assert np.all(np.isfinite(inlier_w))
    assert outlier_w < 0.5
    assert float(np.min(inlier_w)) > outlier_w


def test_compute_gene_mixture_insufficient_guides_reports_error():
    per_guide_ols = pd.DataFrame(
        [
            {"guide_id": "g1", "focal_var": "treatment", "beta": 0.5, "se": 0.2},
            {"guide_id": "g2", "focal_var": "treatment", "beta": 0.6, "se": 0.2},
        ]
    )
    annotation_table = pd.DataFrame({"gene": ["A", "A"]}, index=["g1", "g2"])
    flag_table = pd.DataFrame([{"gene_id": "A", "focal_var": "treatment", "flagged": True}])

    config = mod.GeneMixtureConfig(scope="flagged", min_guides=3)
    gene, guides = mod.compute_gene_mixture(
        per_guide_ols,
        annotation_table,
        focal_vars=["treatment"],
        gene_id_col=1,
        meta_results=None,
        flag_table=flag_table,
        config=config,
    )

    assert gene.shape[0] == 1
    row = gene.iloc[0]
    assert bool(row["fit_ok"]) is False
    assert str(row["fit_error"]) == "insufficient_guides"
    assert np.isnan(float(row["wald_p"]))
    assert guides.empty


def test_compute_gene_mixture_scope_flagged_skips_unflagged_genes():
    per_guide_ols = pd.DataFrame(
        [
            {"guide_id": "g1", "focal_var": "treatment", "beta": 0.5, "se": 0.2},
            {"guide_id": "g2", "focal_var": "treatment", "beta": 0.6, "se": 0.2},
            {"guide_id": "g3", "focal_var": "treatment", "beta": 0.7, "se": 0.2},
        ]
    )
    annotation_table = pd.DataFrame({"gene": ["A", "A", "A"]}, index=["g1", "g2", "g3"])
    flag_table = pd.DataFrame([{"gene_id": "A", "focal_var": "treatment", "flagged": False}])

    config = mod.GeneMixtureConfig(scope="flagged", min_guides=3)
    gene, guides = mod.compute_gene_mixture(
        per_guide_ols,
        annotation_table,
        focal_vars=["treatment"],
        gene_id_col=1,
        meta_results=None,
        flag_table=flag_table,
        config=config,
    )

    assert gene.empty
    assert guides.empty

