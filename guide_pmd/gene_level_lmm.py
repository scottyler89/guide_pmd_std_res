from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from scipy.stats import norm

from .gene_level import _align_model_matrix
from .gene_level import _get_gene_ids
from .gene_level import _nan_fdr


def _prepare_model_matrix(model_matrix: pd.DataFrame, *, add_intercept: bool) -> pd.DataFrame:
    mm = model_matrix.copy()
    if add_intercept and "Intercept" not in mm.columns:
        mm.insert(0, "Intercept", 1.0)
    try:
        mm = mm.apply(pd.to_numeric)
    except Exception as exc:  # pragma: no cover
        raise ValueError("model_matrix must be numeric") from exc
    return mm


def _wide_to_long(response_matrix: pd.DataFrame) -> pd.DataFrame:
    wide = response_matrix.copy()
    wide.index = wide.index.astype(str)
    wide.index.name = "guide_id"
    long_df = wide.reset_index().melt(id_vars=["guide_id"], var_name="sample_id", value_name="y")
    return long_df


def _fit_mixedlm(
    endog: pd.Series,
    exog: pd.DataFrame,
    *,
    groups: pd.Series,
    exog_re: pd.DataFrame,
    max_iter: int,
) -> tuple[sm.regression.mixed_linear_model.MixedLMResults | None, str | None]:
    try:
        model = sm.MixedLM(endog, exog, groups=groups, exog_re=exog_re)
        res = model.fit(reml=False, method="lbfgs", maxiter=max_iter, disp=False)
    except Exception as exc:
        return None, str(exc)
    return res, None


def _extract_re_sds(result: sm.regression.mixed_linear_model.MixedLMResults, *, random_slope: bool) -> tuple[float, float]:
    cov_re = result.cov_re
    if hasattr(cov_re, "to_numpy"):
        cov = cov_re.to_numpy(dtype=float)
    else:  # pragma: no cover
        cov = np.asarray(cov_re, dtype=float)

    if cov.shape == (1, 1):
        sigma_alpha = float(np.sqrt(max(0.0, cov[0, 0])))
        return sigma_alpha, 0.0
    if cov.shape == (2, 2) and random_slope:
        sigma_alpha = float(np.sqrt(max(0.0, cov[0, 0])))
        tau = float(np.sqrt(max(0.0, cov[1, 1])))
        return sigma_alpha, tau
    sigma_alpha = float(np.sqrt(max(0.0, cov[0, 0])))
    return sigma_alpha, np.nan


def compute_gene_lmm(
    response_matrix: pd.DataFrame,
    annotation_table: pd.DataFrame,
    model_matrix: pd.DataFrame,
    *,
    focal_vars: Sequence[str],
    gene_id_col: int = 1,
    add_intercept: bool = True,
    allow_random_slope: bool = True,
    min_guides_random_slope: int = 3,
    max_iter: int = 200,
    fallback_to_meta: bool = True,
    meta_results: pd.DataFrame | None = None,
    selection_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute gene-level mixed model results (Plan A) with explicit, recorded fallbacks.

    This function never mutates/writes baseline outputs; it returns a table only.
    """
    from . import gene_level as gene_level_mod

    focal_vars = list(focal_vars)
    if not focal_vars:
        raise ValueError("focal_vars must not be empty")

    gene_ids = _get_gene_ids(annotation_table, gene_id_col)
    if response_matrix.index.has_duplicates:
        raise ValueError("response_matrix index must not contain duplicates (guide_id)")
    if gene_ids.index.has_duplicates:
        raise ValueError("annotation_table index must not contain duplicates (guide_id)")

    mm = _align_model_matrix(model_matrix, list(response_matrix.columns))
    mm = _prepare_model_matrix(mm, add_intercept=add_intercept)

    if fallback_to_meta and meta_results is None:
        meta_results = gene_level_mod.compute_gene_meta(
            response_matrix,
            annotation_table,
            model_matrix,
            focal_vars=focal_vars,
            gene_id_col=gene_id_col,
            add_intercept=add_intercept,
        )

    genes_by_focal: dict[str, list[str]] | None = None
    if selection_table is not None:
        required = {"gene_id", "focal_var", "selected"}
        missing = required.difference(set(selection_table.columns))
        if missing:
            raise ValueError(f"selection_table missing required column(s): {sorted(missing)}")
        sel = selection_table.copy()
        sel["gene_id"] = sel["gene_id"].astype(str)
        sel["focal_var"] = sel["focal_var"].astype(str)
        sel = sel[sel["selected"].astype(bool)]
        genes_by_focal = {}
        for focal_var in sorted(focal_vars):
            focal_var = str(focal_var)
            genes = sel.loc[sel["focal_var"] == focal_var, "gene_id"].unique().tolist()
            genes_by_focal[focal_var] = sorted([str(g) for g in genes])

    out_rows: list[dict[str, object]] = []
    for focal_var in sorted(focal_vars):
        if focal_var not in mm.columns:
            raise ValueError(f"focal var missing from model_matrix: {focal_var}")

        fixed_cols_full = list(mm.columns)
        fixed_cols_null = [c for c in fixed_cols_full if c != focal_var]
        random_slope_allowed = bool(allow_random_slope)

        genes_to_fit = genes_by_focal.get(str(focal_var), []) if genes_by_focal is not None else None
        gene_iter = genes_to_fit if genes_to_fit is not None else sorted(gene_ids.unique().tolist())
        for gene_id in gene_iter:
            guides = gene_ids.index[gene_ids == gene_id].tolist()
            guides = [g for g in guides if g in response_matrix.index]
            sub = response_matrix.loc[guides, :]
            m_guides_total = int(sub.shape[0])
            n_samples = int(sub.shape[1])
            n_obs = int(m_guides_total * n_samples)

            if m_guides_total == 0:
                out_rows.append(
                    {
                        "gene_id": gene_id,
                        "focal_var": focal_var,
                        "method": "failed",
                        "model": "",
                        "theta": np.nan,
                        "se_theta": np.nan,
                        "wald_z": np.nan,
                        "wald_p": np.nan,
                        "wald_ok": False,
                        "wald_p_adj": np.nan,
                        "lrt_stat": np.nan,
                        "lrt_p": np.nan,
                        "lrt_ok": False,
                        "lrt_p_adj": np.nan,
                        "sigma_alpha": np.nan,
                        "tau": np.nan,
                        "converged_full": False,
                        "converged_null": False,
                        "m_guides_total": 0.0,
                        "m_guides_used": 0.0,
                        "n_samples": float(n_samples),
                        "n_obs": float(n_obs),
                        "fit_error": "no guides for gene in response_matrix",
                    }
                )
                continue

            long_df = _wide_to_long(sub)
            mm_reset = mm.reset_index().rename(columns={mm.index.name or "index": "sample_id"})
            long_df = long_df.merge(mm_reset, on="sample_id", how="left", validate="many_to_one")
            if long_df[fixed_cols_full].isna().any().any():
                raise ValueError(f"model_matrix missing sample ids after join for gene {gene_id}")

            endog = long_df["y"].astype(float)
            groups = long_df["guide_id"].astype(str)
            random_slope = random_slope_allowed and (m_guides_total >= int(min_guides_random_slope))

            def fit_with_structure(use_random_slope: bool) -> dict[str, object]:
                exog_full = long_df[fixed_cols_full].astype(float)
                exog_null = long_df[fixed_cols_null].astype(float)

                if use_random_slope:
                    exog_re = pd.DataFrame(
                        {
                            "Intercept_re": np.ones(long_df.shape[0], dtype=float),
                            f"{focal_var}_re": long_df[focal_var].astype(float).to_numpy(),
                        },
                        index=long_df.index,
                    )
                    model_name = "ri+rs"
                else:
                    exog_re = pd.DataFrame(
                        {"Intercept_re": np.ones(long_df.shape[0], dtype=float)},
                        index=long_df.index,
                    )
                    model_name = "ri"

                full_res, full_err = _fit_mixedlm(
                    endog,
                    exog_full,
                    groups=groups,
                    exog_re=exog_re,
                    max_iter=max_iter,
                )
                null_res, null_err = _fit_mixedlm(
                    endog,
                    exog_null,
                    groups=groups,
                    exog_re=exog_re,
                    max_iter=max_iter,
                )

                converged_full = bool(getattr(full_res, "converged", False)) if full_res is not None else False
                converged_null = bool(getattr(null_res, "converged", False)) if null_res is not None else False

                fit_error = "; ".join([e for e in [full_err, null_err] if e]) if (full_err or null_err) else ""
                if (full_res is None) or (null_res is None) or (not converged_full) or (not converged_null):
                    return {
                        "ok": False,
                        "model": model_name,
                        "full_res": full_res,
                        "null_res": null_res,
                        "converged_full": converged_full,
                        "converged_null": converged_null,
                        "fit_error": fit_error or "non-converged fit",
                    }

                theta = float(full_res.params.get(focal_var, np.nan))
                se_theta = float(full_res.bse.get(focal_var, np.nan))
                wald_z = float(theta / se_theta) if (np.isfinite(theta) and np.isfinite(se_theta) and se_theta > 0) else np.nan
                wald_p = float(2 * norm.sf(abs(wald_z))) if np.isfinite(wald_z) else np.nan

                ll_full = float(full_res.model.loglike(full_res.params))
                ll_null = float(null_res.model.loglike(null_res.params))
                lr = float(2.0 * (ll_full - ll_null)) if (np.isfinite(ll_full) and np.isfinite(ll_null)) else np.nan
                lrt_p = float(chi2.sf(lr, df=1)) if (np.isfinite(lr) and lr >= 0.0) else np.nan

                sigma_alpha, tau = _extract_re_sds(full_res, random_slope=use_random_slope)
                return {
                    "ok": True,
                    "model": model_name,
                    "theta": theta,
                    "se_theta": se_theta,
                    "wald_z": wald_z,
                    "wald_p": wald_p,
                    "lrt_stat": lr,
                    "lrt_p": lrt_p,
                    "lrt_ok": bool(np.isfinite(lrt_p)),
                    "sigma_alpha": sigma_alpha,
                    "tau": tau,
                    "converged_full": converged_full,
                    "converged_null": converged_null,
                    "fit_error": "",
                }

            res = fit_with_structure(random_slope)
            if (not res["ok"]) and random_slope:
                res = fit_with_structure(False)

            if res["ok"]:
                lrt_ok = bool(res["lrt_ok"])
                wald_ok = bool(np.isfinite(res["wald_p"]))
                fit_error = ""
                if not lrt_ok:
                    fit_error = "lrt_invalid"
                if (not wald_ok) and fit_error:
                    fit_error = f"{fit_error}; wald_invalid"
                elif not wald_ok:
                    fit_error = "wald_invalid"

                out_rows.append(
                    {
                        "gene_id": gene_id,
                        "focal_var": focal_var,
                        "method": "lmm",
                        "model": res["model"],
                        "theta": res["theta"],
                        "se_theta": res["se_theta"],
                        "wald_z": res["wald_z"],
                        "wald_p": res["wald_p"],
                        "wald_ok": wald_ok,
                        "wald_p_adj": np.nan,
                        "lrt_stat": res["lrt_stat"],
                        "lrt_p": res["lrt_p"],
                        "lrt_ok": lrt_ok,
                        "lrt_p_adj": np.nan,
                        "sigma_alpha": res["sigma_alpha"],
                        "tau": res["tau"],
                        "converged_full": res["converged_full"],
                        "converged_null": res["converged_null"],
                        "m_guides_total": float(m_guides_total),
                        "m_guides_used": float(m_guides_total),
                        "n_samples": float(n_samples),
                        "n_obs": float(n_obs),
                        "fit_error": fit_error,
                    }
                )
                continue

            if fallback_to_meta and (meta_results is not None):
                meta_row = meta_results[(meta_results["gene_id"] == gene_id) & (meta_results["focal_var"] == focal_var)]
                if meta_row.shape[0] == 1:
                    meta_row = meta_row.iloc[0]
                    theta = float(meta_row["theta"])
                    se_theta = float(meta_row["se_theta"])
                    z = float(meta_row["z"])
                    p = float(meta_row["p"])
                    out_rows.append(
                        {
                            "gene_id": gene_id,
                            "focal_var": focal_var,
                            "method": "meta_fallback",
                            "model": res["model"],
                            "theta": theta,
                            "se_theta": se_theta,
                            "wald_z": z,
                            "wald_p": p,
                            "wald_ok": bool(np.isfinite(p)),
                            "wald_p_adj": np.nan,
                            "lrt_stat": np.nan,
                            "lrt_p": np.nan,
                            "lrt_ok": False,
                            "lrt_p_adj": np.nan,
                            "sigma_alpha": np.nan,
                            "tau": float(meta_row["tau"]),
                            "converged_full": False,
                            "converged_null": False,
                            "m_guides_total": float(m_guides_total),
                            "m_guides_used": float(meta_row["m_guides_used"]),
                            "n_samples": float(n_samples),
                            "n_obs": float(n_obs),
                            "fit_error": f"lmm_failed: {res['fit_error']}",
                        }
                    )
                    continue

            out_rows.append(
                {
                    "gene_id": gene_id,
                    "focal_var": focal_var,
                    "method": "failed",
                    "model": res["model"],
                    "theta": np.nan,
                    "se_theta": np.nan,
                    "wald_z": np.nan,
                    "wald_p": np.nan,
                    "wald_ok": False,
                    "wald_p_adj": np.nan,
                    "lrt_stat": np.nan,
                    "lrt_p": np.nan,
                    "lrt_ok": False,
                    "lrt_p_adj": np.nan,
                    "sigma_alpha": np.nan,
                    "tau": np.nan,
                    "converged_full": bool(res.get("converged_full", False)),
                    "converged_null": bool(res.get("converged_null", False)),
                    "m_guides_total": float(m_guides_total),
                    "m_guides_used": float(m_guides_total),
                    "n_samples": float(n_samples),
                    "n_obs": float(n_obs),
                    "fit_error": str(res.get("fit_error", "")),
                }
            )

    columns = [
        "gene_id",
        "focal_var",
        "method",
        "model",
        "theta",
        "se_theta",
        "wald_z",
        "wald_p",
        "wald_ok",
        "wald_p_adj",
        "lrt_stat",
        "lrt_p",
        "lrt_ok",
        "lrt_p_adj",
        "sigma_alpha",
        "tau",
        "converged_full",
        "converged_null",
        "m_guides_total",
        "m_guides_used",
        "n_samples",
        "n_obs",
        "fit_error",
    ]
    out = pd.DataFrame(out_rows, columns=columns).sort_values(["focal_var", "gene_id"], kind="mergesort").reset_index(
        drop=True
    )
    if not out.empty:
        out["wald_p_adj"] = out.groupby("focal_var", sort=False)["wald_p"].transform(_nan_fdr)
        out["lrt_p_adj"] = out.groupby("focal_var", sort=False)["lrt_p"].transform(_nan_fdr)
    return out
