from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.robust.scale import Huber

from .gene_level import _align_model_matrix
from .gene_level import _get_gene_ids
from .gene_level import fit_per_guide_ols


def _trimmed_mean(values: np.ndarray, *, prop: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    if prop <= 0:
        return float(np.mean(values))
    values = np.sort(values)
    k = int(np.floor(prop * values.size))
    if k == 0:
        return float(np.mean(values))
    if values.size <= 2 * k:
        return float(np.mean(values))
    return float(np.mean(values[k:-k]))


def _winsorized_mean(values: np.ndarray, *, prop: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    if prop <= 0:
        return float(np.mean(values))
    values = np.sort(values)
    k = int(np.floor(prop * values.size))
    if k == 0:
        return float(np.mean(values))
    if values.size <= 2 * k:
        return float(np.mean(values))
    lo = values[k]
    hi = values[-k - 1]
    clipped = np.clip(values, lo, hi)
    return float(np.mean(clipped))


def _huber_location(
    values: np.ndarray,
    *,
    c: float,
    tol: float = 1e-8,
    max_iter: int = 30,
) -> tuple[float, str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, ""
    if values.size == 1:
        return float(values[0]), "single"

    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            mu, _scale = Huber(c=float(c), tol=float(tol), maxiter=int(max_iter))(values)
        return float(mu), "huber"
    except Exception:
        return float(np.median(values)), "median_fallback"


def compute_gene_qc(
    response_matrix: pd.DataFrame,
    annotation_table: pd.DataFrame,
    model_matrix: pd.DataFrame,
    *,
    focal_vars: Sequence[str],
    gene_id_col: int = 1,
    add_intercept: bool = True,
    trim_prop: float = 0.2,
    huber_c: float = 1.5,
    per_guide_ols: pd.DataFrame | None = None,
    residual_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute per-gene QC/robust summaries from per-guide OLS fits.

    This implements the diagnostic portion of Plan C (P3.5) without applying any
    hard thresholds or classification rules (see DEV_RUBRIC.md).
    """
    focal_vars = list(focal_vars)
    if not focal_vars:
        raise ValueError("focal_vars must not be empty")

    gene_ids = _get_gene_ids(annotation_table, gene_id_col)
    if response_matrix.index.has_duplicates:
        raise ValueError("response_matrix index must not contain duplicates (guide_id)")
    if gene_ids.index.has_duplicates:
        raise ValueError("annotation_table index must not contain duplicates (guide_id)")

    mm = _align_model_matrix(model_matrix, list(response_matrix.columns))
    if per_guide_ols is None:
        per_guide = fit_per_guide_ols(
            response_matrix,
            mm,
            focal_vars=focal_vars,
            add_intercept=add_intercept,
        )
    else:
        required = {"guide_id", "focal_var", "beta", "se"}
        missing_cols = required.difference(set(per_guide_ols.columns))
        if missing_cols:
            raise ValueError(f"per_guide_ols missing required column(s): {sorted(missing_cols)}")
        per_guide = per_guide_ols.copy()
        per_guide = per_guide[per_guide["focal_var"].isin([str(v) for v in focal_vars])]
    per_guide = per_guide.merge(gene_ids, left_on="guide_id", right_index=True, how="left")
    if per_guide["gene_id"].isna().any():
        missing_guides = per_guide.loc[per_guide["gene_id"].isna(), "guide_id"].unique().tolist()
        raise ValueError(f"missing gene ids for {len(missing_guides)} guide(s), e.g. {missing_guides[:5]}")

    if residual_matrix is not None:
        if residual_matrix.index.has_duplicates:
            raise ValueError("residual_matrix index must not contain duplicates (guide_id)")
        residual_matrix = residual_matrix.reindex(columns=list(response_matrix.columns))

    rows: list[dict[str, object]] = []
    for (gene_id, focal_var), sub in per_guide.groupby(["gene_id", "focal_var"], sort=True):
        beta = sub["beta"].to_numpy(dtype=float)
        se = sub["se"].to_numpy(dtype=float)
        z = np.full(beta.shape, np.nan, dtype=float)
        good = np.isfinite(beta) & np.isfinite(se) & (se > 0)
        z[good] = beta[good] / se[good]

        beta_good = beta[np.isfinite(beta)]
        m_total = int(sub.shape[0])
        m_used = int(beta_good.size)

        beta_mean = float(np.mean(beta_good)) if beta_good.size else np.nan
        beta_median = float(np.median(beta_good)) if beta_good.size else np.nan
        beta_sd = float(np.std(beta_good, ddof=1)) if beta_good.size >= 2 else np.nan
        beta_mad = float(np.median(np.abs(beta_good - beta_median))) if beta_good.size else np.nan

        majority_sign = float(np.sign(beta_median)) if np.isfinite(beta_median) else np.nan
        if beta_good.size and np.isfinite(majority_sign):
            sign_agreement = float(np.mean(np.sign(beta_good) == majority_sign))
            frac_opposite = float(np.mean(np.sign(beta_good) == (-majority_sign)))
        else:
            sign_agreement = np.nan
            frac_opposite = np.nan

        max_abs_beta = float(np.max(np.abs(beta_good))) if beta_good.size else np.nan
        max_abs_z = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else np.nan

        beta_huber, beta_huber_source = _huber_location(beta_good, c=huber_c)

        max_abs_resid = np.nan
        if residual_matrix is not None:
            guides = sub["guide_id"].tolist()
            guides = [g for g in guides if g in residual_matrix.index]
            if guides:
                max_abs_resid = float(np.nanmax(np.abs(residual_matrix.loc[guides].to_numpy(dtype=float))))

        rows.append(
            {
                "gene_id": gene_id,
                "focal_var": focal_var,
                "m_guides_total": float(m_total),
                "m_guides_used": float(m_used),
                "trim_prop": float(trim_prop),
                "huber_c": float(huber_c),
                "beta_mean": beta_mean,
                "beta_median": beta_median,
                "beta_sd": beta_sd,
                "beta_mad": beta_mad,
                "beta_trimmed_mean": _trimmed_mean(beta_good, prop=trim_prop),
                "beta_winsor_mean": _winsorized_mean(beta_good, prop=trim_prop),
                "beta_huber": beta_huber,
                "beta_huber_source": beta_huber_source,
                "majority_sign": majority_sign,
                "sign_agreement": sign_agreement,
                "frac_opposite_sign": frac_opposite,
                "max_abs_beta": max_abs_beta,
                "max_abs_z": max_abs_z,
                "max_abs_resid": max_abs_resid,
            }
        )

    out = pd.DataFrame(rows).sort_values(["focal_var", "gene_id"], kind="mergesort").reset_index(drop=True)
    return out
