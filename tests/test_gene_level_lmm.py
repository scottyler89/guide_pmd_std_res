import numpy as np
import pandas as pd

import guide_pmd.gene_level_lmm as glmm


def _make_synthetic(n_samples: int = 20, *, seed: int = 123) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
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
            intercept = rng.normal(0.0, 0.5)
            noise = rng.normal(0.0, 0.5, size=n_samples)
            y = intercept + theta * treatment + noise
            guides.append(guide_id)
            genes.append(gene_id)
            y_rows.append(y)

    add_gene("A", theta=2.0, n_guides=4)
    add_gene("B", theta=0.0, n_guides=4)

    response_matrix = pd.DataFrame(y_rows, index=guides, columns=sample_ids)
    annotation_table = pd.DataFrame({"gene": genes}, index=guides)
    return response_matrix, annotation_table, model_matrix


def test_compute_gene_lmm_detects_signal_in_positive_control():
    response, ann, mm = _make_synthetic()
    out = glmm.compute_gene_lmm(
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

    assert out["focal_var"].unique().tolist() == ["treatment"]
    assert out["gene_id"].tolist() == ["A", "B"]
    assert out["method"].tolist() == ["lmm", "lmm"]
    assert out["model"].tolist() == ["ri", "ri"]

    row_a = out.loc[out["gene_id"] == "A"].iloc[0]
    row_b = out.loc[out["gene_id"] == "B"].iloc[0]

    assert row_a["theta"] > 1.0
    assert bool(row_a["lrt_ok"]) is True
    assert row_a["lrt_p"] < 1e-6
    assert bool(row_a["wald_ok"]) is True
    assert row_a["wald_p"] < 1e-6
    assert row_b["lrt_p"] > 0.01


def test_compute_gene_lmm_uses_ri_when_insufficient_guides():
    response, ann, mm = _make_synthetic()
    subset_guides = [g for g in response.index if g.startswith("A_")][:2]
    response = response.loc[subset_guides, :]
    ann = ann.loc[subset_guides, :]

    out = glmm.compute_gene_lmm(
        response,
        ann,
        mm,
        focal_vars=["treatment"],
        gene_id_col=1,
        add_intercept=True,
        allow_random_slope=True,
        min_guides_random_slope=3,
        fallback_to_meta=False,
        max_iter=200,
    )

    assert out["gene_id"].tolist() == ["A"]
    assert out["model"].tolist() == ["ri"]


def test_compute_gene_lmm_falls_back_to_meta_explicitly(monkeypatch):
    response, ann, mm = _make_synthetic()

    def always_fail(*_args, **_kwargs):
        return None, "forced failure"

    monkeypatch.setattr(glmm, "_fit_mixedlm", always_fail)

    out = glmm.compute_gene_lmm(
        response,
        ann,
        mm,
        focal_vars=["treatment"],
        gene_id_col=1,
        add_intercept=True,
        allow_random_slope=False,
        fallback_to_meta=True,
        max_iter=50,
    )

    assert set(out["method"]) == {"meta_fallback"}
    assert set(out["lrt_ok"]) == {False}
    assert set(out["wald_ok"]) == {True}
    assert out["fit_error"].str.contains("lmm_failed").all()


def test_compute_gene_lmm_falls_back_from_random_slope_to_ri(monkeypatch):
    response, ann, mm = _make_synthetic()

    orig_fit = glmm._fit_mixedlm

    def fail_random_slope_then_fit_ri(endog, exog, *, groups, exog_re, max_iter):
        if exog_re.shape[1] == 2:
            return None, "forced random-slope failure"
        return orig_fit(endog, exog, groups=groups, exog_re=exog_re, max_iter=max_iter)

    monkeypatch.setattr(glmm, "_fit_mixedlm", fail_random_slope_then_fit_ri)

    out = glmm.compute_gene_lmm(
        response,
        ann,
        mm,
        focal_vars=["treatment"],
        gene_id_col=1,
        add_intercept=True,
        allow_random_slope=True,
        min_guides_random_slope=3,
        fallback_to_meta=False,
        max_iter=200,
    )

    assert out["method"].tolist() == ["lmm", "lmm"]
    assert out["model"].tolist() == ["ri", "ri"]
