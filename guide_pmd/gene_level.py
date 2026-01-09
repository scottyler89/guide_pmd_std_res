from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import false_discovery_control as fdr
from scipy.stats import norm


def _nan_fdr(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    good = np.isfinite(p_values)
    out = np.ones_like(p_values, dtype=float)
    if good.any():
        out[good] = fdr(p_values[good])
    return out


def _align_model_matrix(model_matrix: pd.DataFrame, sample_ids: Sequence[str]) -> pd.DataFrame:
    if model_matrix.index.has_duplicates:
        raise ValueError("model_matrix index must not contain duplicates")
    missing = [sample_id for sample_id in sample_ids if sample_id not in model_matrix.index]
    if missing:
        raise ValueError(f"model_matrix missing {len(missing)} sample id(s), e.g. {missing[:5]}")
    return model_matrix.reindex(sample_ids)


def _get_gene_ids(annotation_table: pd.DataFrame, gene_id_col: int) -> pd.Series:
    gene_id_col = int(gene_id_col)
    if gene_id_col < 1:
        raise ValueError(
            "gene_id_col must be >= 1 (0 is the guide id/index column and is not present in annotation_table)"
        )
    ann_idx = gene_id_col - 1
    if ann_idx >= annotation_table.shape[1]:
        raise ValueError(f"gene_id_col out of range for annotation_table: {gene_id_col}")
    gene_ids = annotation_table.iloc[:, ann_idx].astype(str)
    gene_ids.name = "gene_id"
    return gene_ids


def _fit_per_guide_ols(
    response_matrix: pd.DataFrame,
    model_matrix: pd.DataFrame,
    *,
    focal_vars: Sequence[str],
    add_intercept: bool = True,
) -> pd.DataFrame:
    if not isinstance(response_matrix, pd.DataFrame) or not isinstance(model_matrix, pd.DataFrame):
        raise ValueError("response_matrix and model_matrix must be pandas DataFrames")

    focal_vars = list(focal_vars)
    if not focal_vars:
        raise ValueError("focal_vars must not be empty")
    missing = [v for v in focal_vars if v not in model_matrix.columns]
    if missing:
        raise ValueError(f"focal var(s) missing from model_matrix: {missing}")

    mm = model_matrix.copy()
    if add_intercept and "Intercept" not in mm.columns:
        mm.insert(0, "Intercept", 1.0)

    try:
        mm = mm.apply(pd.to_numeric)
    except Exception as exc:  # pragma: no cover
        raise ValueError("model_matrix must be numeric") from exc

    X = mm.to_numpy(dtype=float)
    coef_names = list(mm.columns)

    rows: list[dict[str, object]] = []
    for guide_id, y in response_matrix.iterrows():
        y_arr = y.to_numpy(dtype=float)
        try:
            res = sm.OLS(y_arr, X).fit()
        except Exception:
            for focal_var in focal_vars:
                rows.append(
                    {
                        "guide_id": guide_id,
                        "focal_var": focal_var,
                        "beta": np.nan,
                        "se": np.nan,
                        "t": np.nan,
                        "p": np.nan,
                    }
                )
            continue

        for focal_var in focal_vars:
            if focal_var not in coef_names:
                raise ValueError(f"internal error: missing coef {focal_var}")  # pragma: no cover
            idx = coef_names.index(focal_var)
            rows.append(
                {
                    "guide_id": guide_id,
                    "focal_var": focal_var,
                    "beta": float(res.params[idx]),
                    "se": float(res.bse[idx]),
                    "t": float(res.tvalues[idx]),
                    "p": float(res.pvalues[idx]),
                }
            )

    return pd.DataFrame(rows)


def _dersimonian_laird(beta: np.ndarray, se: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    beta = beta[mask]
    se = se[mask]
    m = int(beta.size)

    if m == 0:
        return {
            "theta": np.nan,
            "se_theta": np.nan,
            "z": np.nan,
            "p": np.nan,
            "tau2": np.nan,
            "tau": np.nan,
            "Q": np.nan,
            "I2": np.nan,
            "m_guides_used": 0.0,
        }

    if m == 1:
        theta = float(beta[0])
        se_theta = float(se[0])
        z = theta / se_theta if se_theta > 0 else np.nan
        p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
        return {
            "theta": theta,
            "se_theta": se_theta,
            "z": float(z) if np.isfinite(z) else np.nan,
            "p": p,
            "tau2": 0.0,
            "tau": 0.0,
            "Q": 0.0,
            "I2": 0.0,
            "m_guides_used": 1.0,
        }

    w = 1.0 / (se**2)
    w_sum = float(w.sum())
    theta_fe = float(np.sum(w * beta) / w_sum)
    Q = float(np.sum(w * (beta - theta_fe) ** 2))
    df = m - 1
    C = float(w_sum - (np.sum(w**2) / w_sum))
    tau2 = float(max(0.0, (Q - df) / C)) if C > 0 else 0.0
    w_re = 1.0 / (se**2 + tau2)
    theta = float(np.sum(w_re * beta) / float(w_re.sum()))
    se_theta = float(np.sqrt(1.0 / float(w_re.sum())))
    z = theta / se_theta if se_theta > 0 else np.nan
    p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    I2 = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0
    return {
        "theta": theta,
        "se_theta": se_theta,
        "z": float(z) if np.isfinite(z) else np.nan,
        "p": p,
        "tau2": tau2,
        "tau": float(np.sqrt(tau2)),
        "Q": Q,
        "I2": I2,
        "m_guides_used": float(m),
    }


def compute_gene_meta(
    response_matrix: pd.DataFrame,
    annotation_table: pd.DataFrame,
    model_matrix: pd.DataFrame,
    *,
    focal_vars: Sequence[str],
    gene_id_col: int = 1,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """
    Compute gene-level random-effects meta-analysis from per-guide OLS fits.

    This is "Plan B" from docs/plans/gene_level_aggregation_plan.md.
    """
    gene_ids = _get_gene_ids(annotation_table, gene_id_col)
    if response_matrix.index.has_duplicates:
        raise ValueError("response_matrix index must not contain duplicates (guide_id)")
    if gene_ids.index.has_duplicates:
        raise ValueError("annotation_table index must not contain duplicates (guide_id)")

    mm = _align_model_matrix(model_matrix, list(response_matrix.columns))
    per_guide = _fit_per_guide_ols(
        response_matrix,
        mm,
        focal_vars=focal_vars,
        add_intercept=add_intercept,
    )
    per_guide = per_guide.merge(gene_ids, left_on="guide_id", right_index=True, how="left")
    if per_guide["gene_id"].isna().any():
        missing_guides = per_guide.loc[per_guide["gene_id"].isna(), "guide_id"].unique().tolist()
        raise ValueError(f"missing gene ids for {len(missing_guides)} guide(s), e.g. {missing_guides[:5]}")

    out_rows: list[dict[str, object]] = []
    for (gene_id, focal_var), sub in per_guide.groupby(["gene_id", "focal_var"], sort=True):
        beta = sub["beta"].to_numpy(dtype=float)
        se = sub["se"].to_numpy(dtype=float)
        meta = _dersimonian_laird(beta, se)

        if np.isfinite(meta["theta"]) and meta["theta"] != 0:
            sign = float(np.sign(meta["theta"]))
            sign_agreement = float(np.mean(np.sign(beta[np.isfinite(beta)]) == sign))
        elif np.isfinite(meta["theta"]):
            sign_agreement = float(np.mean(np.sign(beta[np.isfinite(beta)]) == 0.0))
        else:
            sign_agreement = np.nan

        out_rows.append(
            {
                "gene_id": gene_id,
                "focal_var": focal_var,
                "theta": meta["theta"],
                "se_theta": meta["se_theta"],
                "z": meta["z"],
                "p": meta["p"],
                "p_adj": np.nan,  # filled below per focal var
                "tau": meta["tau"],
                "tau2": meta["tau2"],
                "Q": meta["Q"],
                "I2": meta["I2"],
                "m_guides_total": float(sub.shape[0]),
                "m_guides_used": meta["m_guides_used"],
                "sign_agreement": sign_agreement,
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["focal_var", "gene_id"], kind="mergesort").reset_index(drop=True)
    if not out.empty:
        out["p_adj"] = out.groupby("focal_var", sort=False)["p"].transform(_nan_fdr)
    return out
