import numpy as np
import pandas as pd

from guide_pmd.expected_counts import bucket_expected_counts
from guide_pmd.expected_counts import chisq_expected_counts
from guide_pmd.expected_counts import collapse_guide_counts_to_gene_counts
from guide_pmd.expected_counts import summarize_expected_counts


def test_collapse_guide_counts_to_gene_counts_sums_guides():
    guide_counts = pd.DataFrame(
        {
            "s1": [1, 2, 10],
            "s2": [3, 4, 20],
        },
        index=["g1", "g2", "g3"],
    )
    gene_ids = pd.Series({"g1": "A", "g2": "A", "g3": "B"})

    collapsed = collapse_guide_counts_to_gene_counts(guide_counts, gene_ids)
    assert list(collapsed.index) == ["A", "B"]
    assert list(collapsed.columns) == ["s1", "s2"]
    assert collapsed.loc["A", "s1"] == 3
    assert collapsed.loc["A", "s2"] == 7
    assert collapsed.loc["B", "s1"] == 10
    assert collapsed.loc["B", "s2"] == 20


def test_chisq_expected_counts_matches_hand_calc():
    counts = pd.DataFrame([[10, 20], [30, 40]], index=["g1", "g2"], columns=["s1", "s2"])
    expected = chisq_expected_counts(counts)

    # Row sums = [30, 70]; col sums = [40, 60]; grand = 100.
    assert np.isclose(expected.loc["g1", "s1"], 12.0)
    assert np.isclose(expected.loc["g1", "s2"], 18.0)
    assert np.isclose(expected.loc["g2", "s1"], 28.0)
    assert np.isclose(expected.loc["g2", "s2"], 42.0)


def test_summarize_expected_counts_quantiles():
    expected = pd.DataFrame([[12, 18], [28, 42]], index=["g1", "g2"], columns=["s1", "s2"])
    summ = summarize_expected_counts(expected, quantiles=(0.5,))

    assert np.isclose(summ.loc["g1", "expected_min"], 12.0)
    assert np.isclose(summ.loc["g1", "expected_mean"], 15.0)
    assert np.isclose(summ.loc["g1", "expected_p50"], 15.0)

    assert np.isclose(summ.loc["g2", "expected_min"], 28.0)
    assert np.isclose(summ.loc["g2", "expected_mean"], 35.0)
    assert np.isclose(summ.loc["g2", "expected_p50"], 35.0)


def test_bucket_expected_counts_default_thresholds():
    buckets = bucket_expected_counts([0.5, 1.0, 2.9, 3.0, 4.9, 5.0, np.nan])
    assert list(buckets.astype(object)) == ["<1", "1-<3", "1-<3", "3-<5", "3-<5", ">=5", np.nan]

