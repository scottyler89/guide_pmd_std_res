import pytest

import numpy as np
import pandas as pd
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_linear_model import GLM

from guide_pmd.contrasts import build_contrast_matrix
from guide_pmd.contrasts import parse_contrast_spec
import guide_pmd.pmd_std_res_guide_counts as guide_mod


def test_parse_contrast_spec_simple_difference():
    out = parse_contrast_spec("A - B")
    assert out.name == "A - B"
    assert out.weights == {"A": 1.0, "B": -1.0}


def test_parse_contrast_spec_named():
    out = parse_contrast_spec("ab=A - B")
    assert out.name == "ab"
    assert out.weights == {"A": 1.0, "B": -1.0}


def test_parse_contrast_spec_supports_backticks():
    out = parse_contrast_spec("`C-1` - B")
    assert out.weights == {"C-1": 1.0, "B": -1.0}


def test_parse_contrast_spec_rejects_nonzero_constant():
    with pytest.raises(ValueError, match="constant"):
        parse_contrast_spec("A + 1")


def test_parse_contrast_spec_rejects_non_linear_product():
    with pytest.raises(ValueError, match="linear"):
        parse_contrast_spec("A * B")


def test_build_contrast_matrix_validates_terms():
    with pytest.raises(ValueError, match="unknown term"):
        build_contrast_matrix(["A - MISSING"], design_cols=["A", "B"])


def test_run_glm_analysis_with_contrasts_matches_statsmodels_t_test():
    # 1 feature, 6 samples, 2 covariates.
    normalized = pd.DataFrame(
        {
            "s1": [0.1],
            "s2": [0.2],
            "s3": [0.4],
            "s4": [0.5],
            "s5": [0.0],
            "s6": [0.2],
        },
        index=["g1"],
    )
    model_matrix = pd.DataFrame(
        {
            "A": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "B": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        },
        index=["s1", "s2", "s3", "s4", "s5", "s6"],
    )

    _stats, _resids, contrasts = guide_mod.run_glm_analysis_with_contrasts(
        normalized,
        model_matrix,
        contrasts=["A - B"],
        add_intercept=True,
    )

    mm = model_matrix.copy()
    if "Intercept" not in mm.columns:
        mm.insert(0, "Intercept", 1.0)
    y = pd.DataFrame(normalized.loc["g1"].T)
    combined = pd.concat([y, mm], axis=1)
    fit = GLM(combined.iloc[:, 0], combined.loc[:, mm.columns.tolist()], family=Gaussian()).fit()
    L = build_contrast_matrix(["A - B"], mm.columns.tolist()).to_numpy(dtype=float)
    expected = fit.t_test(L)

    row = contrasts.loc[(contrasts["guide_id"] == "g1") & (contrasts["contrast"] == "A - B")].iloc[0]
    assert np.isclose(row["estimate"], float(np.asarray(expected.effect).ravel()[0]))
    assert np.isclose(row["se"], float(np.asarray(expected.sd).ravel()[0]))
    assert np.isclose(row["t"], float(np.asarray(expected.tvalue).ravel()[0]))
    assert np.isclose(row["p"], float(np.asarray(expected.pvalue).ravel()[0]))
