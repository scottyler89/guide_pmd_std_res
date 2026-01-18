from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence

import numpy as np
import pandas as pd


def collapse_guide_counts_to_gene_counts(
    guide_counts: pd.DataFrame,
    gene_ids: pd.Series,
) -> pd.DataFrame:
    """
    Collapse a (guide x sample) counts matrix into a (gene x sample) counts matrix.

    Parameters
    ----------
    guide_counts:
        Rows are guide IDs; columns are sample IDs; values are non-negative counts.
    gene_ids:
        Series mapping guide_id -> gene_id (index must include all guide_counts rows).
    """
    if guide_counts.index.has_duplicates:
        raise ValueError("guide_counts index must not contain duplicates (guide_id)")
    if gene_ids.index.has_duplicates:
        raise ValueError("gene_ids index must not contain duplicates (guide_id)")

    gene_ids = gene_ids.astype(str).reindex(guide_counts.index)
    if gene_ids.isna().any():
        missing = gene_ids.index[gene_ids.isna()].astype(str).tolist()
        raise ValueError(f"missing gene ids for {len(missing)} guide(s), e.g. {missing[:5]}")

    x = guide_counts.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("guide_counts must contain only finite values")
    if (x < 0).any():
        raise ValueError("guide_counts must be non-negative")

    out = guide_counts.copy()
    out.insert(0, "__gene_id", gene_ids.to_numpy(dtype=str))
    collapsed = out.groupby("__gene_id", sort=True).sum(numeric_only=True)
    collapsed.index.name = "gene_id"
    return collapsed


def chisq_expected_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Chi-square expected counts under independence:
        E_ij = row_sum_i * col_sum_j / grand_sum

    Input rows/cols are preserved in the output.
    """
    x = counts.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("counts must contain only finite values")
    if (x < 0).any():
        raise ValueError("counts must be non-negative")

    row_sum = np.sum(x, axis=1, keepdims=True)
    col_sum = np.sum(x, axis=0, keepdims=True)
    grand = float(np.sum(col_sum))
    if grand <= 0.0:
        out = np.full_like(x, np.nan, dtype=float)
    else:
        out = (row_sum @ col_sum) / grand
    return pd.DataFrame(out, index=counts.index.copy(), columns=counts.columns.copy())


def summarize_expected_counts(
    expected_counts: pd.DataFrame,
    *,
    quantiles: Sequence[float] = (0.1,),
) -> pd.DataFrame:
    """
    Per-row summaries of an expected-count table.

    Returns a DataFrame indexed by row key with columns:
      - expected_min
      - expected_mean
      - expected_p{quantile*100:02d} for each requested quantile
    """
    qs = [float(q) for q in quantiles]
    bad = [q for q in qs if not (0.0 <= q <= 1.0)]
    if bad:
        raise ValueError(f"quantiles must be in [0, 1]; got: {bad}")

    x = expected_counts.to_numpy(dtype=float)
    if x.size == 0:
        cols = ["expected_min", "expected_mean"] + [f"expected_p{int(round(q * 100)):02d}" for q in qs]
        return pd.DataFrame(index=expected_counts.index.copy(), columns=cols, dtype=float)

    out: dict[str, np.ndarray] = {
        "expected_min": np.min(x, axis=1).astype(float),
        "expected_mean": np.mean(x, axis=1).astype(float),
    }
    for q in qs:
        key = f"expected_p{int(round(q * 100)):02d}"
        out[key] = np.quantile(x, q, axis=1).astype(float)
    return pd.DataFrame(out, index=expected_counts.index.copy())


def bucket_expected_counts(
    values: Iterable[float] | pd.Series,
    *,
    thresholds: Sequence[float] = (1.0, 3.0, 5.0),
) -> pd.Categorical:
    """
    Bucket values into classic chi-square "headroom" bands.

    Buckets (default):
      - <1
      - 1-<3
      - 3-<5
      - >=5
    """
    ts = [float(t) for t in thresholds]
    if len(ts) != 3:
        raise ValueError("thresholds must have length 3")
    if not (ts[0] < ts[1] < ts[2]):
        raise ValueError(f"thresholds must be strictly increasing; got: {ts}")

    s = pd.Series(values, dtype=float)
    bins = [-np.inf, ts[0], ts[1], ts[2], np.inf]
    labels = [f"<{ts[0]:g}", f"{ts[0]:g}-<{ts[1]:g}", f"{ts[1]:g}-<{ts[2]:g}", f">={ts[2]:g}"]
    return pd.cut(s, bins=bins, labels=labels, right=False, include_lowest=True)


def compute_two_group_expected_count_quantifiability(
    gene_counts: pd.DataFrame,
    treatment: pd.Series,
    *,
    quantile: float = 0.1,
    bucket_thresholds: Sequence[float] = (1.0, 3.0, 5.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expected-count quantifiability for two groups: control (treatment<=0) vs treatment (treatment>0).

    Returns
    -------
    gene_summary:
        Indexed by gene_id, with `y_*_sum`, expected-count summaries per group, and
        `expected_pXX_mincond` + bucket label.
    expected_long:
        Long table with columns: gene_id, sample_id, treatment, observed_count, expected_count.
    """
    if gene_counts.index.has_duplicates:
        raise ValueError("gene_counts index must not contain duplicates (gene_id)")

    treatment = pd.to_numeric(treatment, errors="coerce")
    treatment = treatment.reindex(gene_counts.columns)
    if treatment.isna().any():
        missing = treatment.index[treatment.isna()].astype(str).tolist()
        raise ValueError(f"treatment missing or non-numeric for {len(missing)} sample(s), e.g. {missing[:5]}")

    q = float(quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError("quantile must be in [0, 1]")
    q_name = int(round(q * 100))

    ctrl_cols = treatment.index[treatment.to_numpy(dtype=float) <= 0.0].astype(str).tolist()
    trt_cols = treatment.index[treatment.to_numpy(dtype=float) > 0.0].astype(str).tolist()

    counts_ctrl = gene_counts.reindex(columns=ctrl_cols)
    counts_trt = gene_counts.reindex(columns=trt_cols)
    expected_ctrl = chisq_expected_counts(counts_ctrl)
    expected_trt = chisq_expected_counts(counts_trt)

    ctrl_summ = summarize_expected_counts(expected_ctrl, quantiles=(q,)).add_prefix("ctrl_")
    trt_summ = summarize_expected_counts(expected_trt, quantiles=(q,)).add_prefix("trt_")

    summary = pd.DataFrame(index=gene_counts.index.copy())
    summary["y_ctrl_sum"] = counts_ctrl.sum(axis=1).astype(float)
    summary["y_trt_sum"] = counts_trt.sum(axis=1).astype(float)
    summary = summary.join(ctrl_summ, how="left").join(trt_summ, how="left")

    ctrl_p = f"ctrl_expected_p{q_name:02d}"
    trt_p = f"trt_expected_p{q_name:02d}"
    summary[f"expected_p{q_name:02d}_mincond"] = np.minimum(summary[ctrl_p], summary[trt_p]).astype(float)
    summary[f"expected_p{q_name:02d}_mincond_bucket"] = bucket_expected_counts(
        summary[f"expected_p{q_name:02d}_mincond"], thresholds=bucket_thresholds
    ).astype(object)

    def _stack(df: pd.DataFrame) -> pd.Series:
        try:
            return df.stack(future_stack=True)  # pandas>=2.1 (dropna/sort removed)
        except TypeError:  # pragma: no cover - pandas<2.1
            return df.stack(dropna=False)

    def _long(expected: pd.DataFrame, observed: pd.DataFrame, *, t: float) -> pd.DataFrame:
        out = (
            _stack(expected)
            .rename("expected_count")
            .rename_axis(index=["gene_id", "sample_id"])
            .to_frame()
            .join(
                _stack(observed)
                .rename("observed_count")
                .rename_axis(index=["gene_id", "sample_id"])
            )
            .reset_index()
        )
        out["treatment"] = float(t)
        return out

    expected_long = pd.concat(
        [
            _long(expected_ctrl, counts_ctrl, t=0.0),
            _long(expected_trt, counts_trt, t=1.0),
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    expected_long = expected_long[["gene_id", "sample_id", "treatment", "observed_count", "expected_count"]]

    return summary, expected_long
