from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .gene_level import _align_model_matrix
from .gene_level import _get_gene_ids
from .gene_level import _nan_fdr
from .gene_level import fit_per_guide_ols


def compute_gene_stouffer(
    response_matrix: pd.DataFrame,
    annotation_table: pd.DataFrame,
    model_matrix: pd.DataFrame,
    *,
    focal_vars: Sequence[str],
    gene_id_col: int = 1,
    add_intercept: bool = True,
    per_guide_ols: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute gene-level Stouffer-style combined t-statistics from per-guide OLS fits.

    This mirrors the repo's original combined-statistics approach: combine per-guide t-values as:

        t_combined = sum_j t_j / sqrt(m)

    and then convert to a two-sided p-value using the per-guide residual degrees of freedom.

    Notes:
    - This is a strict information-reduction relative to model-based approaches (meta/LMM).
    - It is included to provide continuity with the historical combined-stats workflow and
      for benchmark coverage.
    """
    if not isinstance(response_matrix, pd.DataFrame) or not isinstance(model_matrix, pd.DataFrame):
        raise ValueError("response_matrix and model_matrix must be pandas DataFrames")

    focal_vars = list(focal_vars)
    if not focal_vars:
        raise ValueError("focal_vars must not be empty")

    gene_ids = _get_gene_ids(annotation_table, gene_id_col)
    if response_matrix.index.has_duplicates:
        raise ValueError("response_matrix index must not contain duplicates (guide_id)")
    if gene_ids.index.has_duplicates:
        raise ValueError("annotation_table index must not contain duplicates (guide_id)")

    mm = _align_model_matrix(model_matrix, list(response_matrix.columns))
    mm = mm.copy()
    if add_intercept and "Intercept" not in mm.columns:
        mm.insert(0, "Intercept", 1.0)

    try:
        mm = mm.apply(pd.to_numeric)
    except Exception as exc:  # pragma: no cover
        raise ValueError("model_matrix must be numeric") from exc

    df_resid = int(mm.shape[0] - mm.shape[1])

    if per_guide_ols is None:
        per_guide = fit_per_guide_ols(
            response_matrix,
            mm,
            focal_vars=focal_vars,
            add_intercept=False,  # already handled above
        )
    else:
        required = {"guide_id", "focal_var", "t"}
        missing_cols = required.difference(set(per_guide_ols.columns))
        if missing_cols:
            raise ValueError(f"per_guide_ols missing required column(s): {sorted(missing_cols)}")
        per_guide = per_guide_ols.copy()

    per_guide = per_guide.copy()
    per_guide["guide_id"] = per_guide["guide_id"].astype(str)
    per_guide["focal_var"] = per_guide["focal_var"].astype(str)
    per_guide["t"] = pd.to_numeric(per_guide["t"], errors="coerce")

    # Attach gene id and compute combined statistics by (gene_id, focal_var).
    gene_map = gene_ids.astype(str).rename("gene_id").to_frame()
    per_guide = per_guide.merge(gene_map, left_on="guide_id", right_index=True, how="inner")
    if per_guide.empty:
        out = pd.DataFrame(columns=["gene_id", "focal_var", "stouffer_t", "p", "p_adj", "m_guides_total", "m_guides_used", "df_resid"])
        return out

    out_rows: list[dict[str, object]] = []
    for (gene_id, focal_var), sub in per_guide.groupby(["gene_id", "focal_var"], sort=True):
        t_vals = sub["t"].to_numpy(dtype=float)
        m_total = int(t_vals.size)
        t_vals = t_vals[np.isfinite(t_vals)]
        m_used = int(t_vals.size)
        if m_used == 0 or df_resid <= 0:
            t_combined = np.nan
            p_val = np.nan
        else:
            t_combined = float(np.sum(t_vals) / np.sqrt(float(m_used)))
            p_val = float(2 * student_t.sf(abs(t_combined), df=int(df_resid)))
        out_rows.append(
            {
                "gene_id": str(gene_id),
                "focal_var": str(focal_var),
                "stouffer_t": t_combined,
                "p": p_val,
                "p_adj": np.nan,
                "m_guides_total": float(m_total),
                "m_guides_used": float(m_used),
                "df_resid": float(df_resid),
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["focal_var", "gene_id"], kind="mergesort").reset_index(drop=True)
    if not out.empty:
        out["p_adj"] = out.groupby("focal_var", sort=False)["p"].transform(_nan_fdr)
    return out

