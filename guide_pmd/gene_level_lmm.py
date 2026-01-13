from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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
    errors: list[str] = []
    best: sm.regression.mixed_linear_model.MixedLMResults | None = None
    best_method: str | None = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"^Random effects covariance is singular.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"^The random effects covariance matrix is singular.*",
        )
        # SciPy optimizer warnings can be very noisy for some methods (Powell/NM)
        # even when the fit converges to a usable solution.
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy\.optimize\._optimize")

        model = sm.MixedLM(endog, exog, groups=groups, exog_re=exog_re)

        # Deterministic method order:
        # - `lbfgs` is typically fast and convergent.
        # - `bfgs`/`cg` are used when `lbfgs` hits an `llf=inf` boundary
        #   (variance=0) in statsmodels, which breaks LRT computation.
        # - `powell`/`nm` (Nelder–Mead) are slow but can converge in cases where
        #   gradient-based optimizers fail or raise `Singular matrix`.
        for method in ("lbfgs", "bfgs", "cg", "powell", "nm"):
            try:
                res = model.fit(reml=False, method=method, maxiter=max_iter, disp=False)
            except Exception as exc:
                errors.append(f"{method}: {exc}")
                continue

            setattr(res, "_guide_pmd_optimizer", method)

            converged = bool(getattr(res, "converged", False))
            llf = float(getattr(res, "llf", np.nan))
            llf_finite = bool(np.isfinite(llf))

            if converged and llf_finite:
                return res, None

            if converged and best is None:
                best = res
                best_method = method

            reason = "non-converged" if not converged else "llf_nonfinite"
            errors.append(f"{method}: {reason}")

    if best is not None:
        # Preserve a converged fit (e.g., for Wald inference), even if llf is non-finite.
        setattr(best, "_guide_pmd_optimizer", str(best_method or ""))
        return best, None

    return None, "; ".join(errors) if errors else "mixedlm fit failed"


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


def _fit_gene_lmm_task(
    *,
    response_matrix: pd.DataFrame,
    gene_to_guides: dict[str, list[str]],
    mm: pd.DataFrame,
    mm_reset: pd.DataFrame,
    gene_id: str,
    focal_var: str,
    allow_random_slope: bool,
    min_guides_random_slope: int,
    max_iter: int,
    fallback_to_meta: bool,
    meta_lookup: dict[tuple[str, str], dict[str, float]] | None,
) -> dict[str, object]:
    fixed_cols_full = list(mm.columns)
    if focal_var not in fixed_cols_full:
        raise ValueError(f"focal var missing from model_matrix: {focal_var}")
    fixed_cols_null = [c for c in fixed_cols_full if c != focal_var]

    guides = gene_to_guides.get(str(gene_id), [])
    sub = response_matrix.loc[guides, :] if guides else response_matrix.iloc[0:0, :]
    m_guides_total = int(sub.shape[0])
    n_samples = int(sub.shape[1])
    n_obs = int(m_guides_total * n_samples)

    if m_guides_total == 0:
        return {
            "gene_id": str(gene_id),
            "focal_var": str(focal_var),
            "method": "failed",
            "model": "",
            "optimizer_full": "",
            "optimizer_null": "",
            "theta": np.nan,
            "se_theta": np.nan,
            "wald_z": np.nan,
            "wald_p": np.nan,
            "wald_ok": False,
            "wald_p_adj": np.nan,
            "lrt_stat": np.nan,
            "lrt_stat_raw": np.nan,
            "lrt_p": np.nan,
            "lrt_ok": False,
            "lrt_p_adj": np.nan,
            "lrt_clipped": False,
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

    long_df = _wide_to_long(sub)
    long_df = long_df.merge(mm_reset, on="sample_id", how="left", validate="many_to_one")
    if long_df[fixed_cols_full].isna().any().any():
        raise ValueError(f"model_matrix missing sample ids after join for gene {gene_id}")

    endog = long_df["y"].astype(float)
    groups = long_df["guide_id"].astype(str)
    random_slope = bool(allow_random_slope) and (m_guides_total >= int(min_guides_random_slope))

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
                "optimizer_full": getattr(full_res, "_guide_pmd_optimizer", "") if full_res is not None else "",
                "optimizer_null": getattr(null_res, "_guide_pmd_optimizer", "") if null_res is not None else "",
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

        ll_full = float(getattr(full_res, "llf", np.nan))
        ll_null = float(getattr(null_res, "llf", np.nan))
        lr_raw = float(2.0 * (ll_full - ll_null)) if (np.isfinite(ll_full) and np.isfinite(ll_null)) else np.nan
        lr = lr_raw
        lrt_clipped = False
        if np.isfinite(lr_raw) and lr_raw < 0.0:
            lr = 0.0
            lrt_clipped = True
        lrt_p = float(chi2.sf(lr, df=1)) if np.isfinite(lr) else np.nan

        sigma_alpha, tau = _extract_re_sds(full_res, random_slope=use_random_slope)
        return {
            "ok": True,
            "model": model_name,
            "optimizer_full": getattr(full_res, "_guide_pmd_optimizer", ""),
            "optimizer_null": getattr(null_res, "_guide_pmd_optimizer", ""),
            "theta": theta,
            "se_theta": se_theta,
            "wald_z": wald_z,
            "wald_p": wald_p,
            "lrt_stat": lr,
            "lrt_stat_raw": lr_raw,
            "lrt_p": lrt_p,
            "lrt_ok": bool(np.isfinite(lrt_p)),
            "lrt_clipped": bool(lrt_clipped),
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

        return {
            "gene_id": str(gene_id),
            "focal_var": str(focal_var),
            "method": "lmm",
            "model": res["model"],
            "optimizer_full": res.get("optimizer_full", ""),
            "optimizer_null": res.get("optimizer_null", ""),
            "theta": res["theta"],
            "se_theta": res["se_theta"],
            "wald_z": res["wald_z"],
            "wald_p": res["wald_p"],
            "wald_ok": wald_ok,
            "wald_p_adj": np.nan,
            "lrt_stat": res["lrt_stat"],
            "lrt_stat_raw": res["lrt_stat_raw"],
            "lrt_p": res["lrt_p"],
            "lrt_ok": lrt_ok,
            "lrt_p_adj": np.nan,
            "lrt_clipped": bool(res.get("lrt_clipped", False)),
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

    if fallback_to_meta and (meta_lookup is not None):
        meta_row = meta_lookup.get((str(gene_id), str(focal_var)))
        if meta_row is not None:
            theta = float(meta_row["theta"])
            se_theta = float(meta_row["se_theta"])
            z = float(meta_row["z"])
            p = float(meta_row["p"])
            return {
                "gene_id": str(gene_id),
                "focal_var": str(focal_var),
                "method": "meta_fallback",
                "model": res["model"],
                "optimizer_full": res.get("optimizer_full", ""),
                "optimizer_null": res.get("optimizer_null", ""),
                "theta": theta,
                "se_theta": se_theta,
                "wald_z": z,
                "wald_p": p,
                "wald_ok": bool(np.isfinite(p)),
                "wald_p_adj": np.nan,
                "lrt_stat": np.nan,
                "lrt_stat_raw": np.nan,
                "lrt_p": np.nan,
                "lrt_ok": False,
                "lrt_p_adj": np.nan,
                "lrt_clipped": False,
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

    return {
        "gene_id": str(gene_id),
        "focal_var": str(focal_var),
        "method": "failed",
        "model": res["model"],
        "optimizer_full": res.get("optimizer_full", ""),
        "optimizer_null": res.get("optimizer_null", ""),
        "theta": np.nan,
        "se_theta": np.nan,
        "wald_z": np.nan,
        "wald_p": np.nan,
        "wald_ok": False,
        "wald_p_adj": np.nan,
        "lrt_stat": np.nan,
        "lrt_stat_raw": np.nan,
        "lrt_p": np.nan,
        "lrt_ok": False,
        "lrt_p_adj": np.nan,
        "lrt_clipped": False,
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
    n_jobs: int = 1,
    progress: bool = False,
    progress_every: int = 500,
    progress_min_seconds: float = 10.0,
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

    meta_lookup: dict[tuple[str, str], dict[str, float]] | None = None
    if fallback_to_meta and meta_results is not None and (not meta_results.empty):
        meta_lookup = {}
        for row in meta_results.itertuples(index=False):
            key = (str(getattr(row, "gene_id")), str(getattr(row, "focal_var")))
            meta_lookup[key] = {
                "theta": float(getattr(row, "theta")),
                "se_theta": float(getattr(row, "se_theta")),
                "z": float(getattr(row, "z")),
                "p": float(getattr(row, "p")),
                "tau": float(getattr(row, "tau")),
                "m_guides_used": float(getattr(row, "m_guides_used")),
            }

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

    n_jobs = int(n_jobs)
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    progress_every = int(progress_every)
    if progress_every < 1:
        progress_every = 1
    progress_min_seconds = float(progress_min_seconds)

    gene_to_guides: dict[str, list[str]] = {}
    guide_set = set(response_matrix.index.astype(str).tolist())
    for guide_id, gene_id in gene_ids.items():
        gid = str(guide_id)
        if gid not in guide_set:
            continue
        gene = str(gene_id)
        gene_to_guides.setdefault(gene, []).append(gid)

    mm_reset = mm.reset_index().rename(columns={mm.index.name or "index": "sample_id"})

    tasks: list[tuple[str, str]] = []
    all_gene_ids = sorted(gene_ids.unique().astype(str).tolist())
    for focal_var in sorted(focal_vars):
        genes_to_fit = genes_by_focal.get(str(focal_var), []) if genes_by_focal is not None else None
        gene_iter = genes_to_fit if genes_to_fit is not None else all_gene_ids
        for gene_id in gene_iter:
            tasks.append((str(focal_var), str(gene_id)))

    out_rows: list[dict[str, object]] = []
    n_total = int(len(tasks))
    t0 = time.monotonic()
    last_print = t0

    if progress and n_total:
        n_unique_genes = len(set([g for _, g in tasks]))
        print(
            f"gene-level lmm: fitting {n_total} task(s) ({n_unique_genes} gene(s) x {len(set([f for f, _ in tasks]))} focal var(s))",
            flush=True,
        )

    if n_jobs == 1 or n_total == 0:
        for idx, (focal_var, gene_id) in enumerate(tasks, start=1):
            row = _fit_gene_lmm_task(
                response_matrix=response_matrix,
                gene_to_guides=gene_to_guides,
                mm=mm,
                mm_reset=mm_reset,
                gene_id=gene_id,
                focal_var=focal_var,
                allow_random_slope=bool(allow_random_slope),
                min_guides_random_slope=int(min_guides_random_slope),
                max_iter=int(max_iter),
                fallback_to_meta=bool(fallback_to_meta),
                meta_lookup=meta_lookup,
            )
            out_rows.append(row)

            if progress and (idx % progress_every == 0 or idx == n_total):
                now = time.monotonic()
                if (now - last_print) >= progress_min_seconds or idx == n_total:
                    last_print = now
                    rate = idx / max(1e-9, (now - t0))
                    eta_s = (n_total - idx) / max(1e-9, rate)
                    print(
                        f"gene-level lmm: {idx}/{n_total} ({rate:.2f} task/s, eta~{eta_s/60.0:.1f} min)",
                        flush=True,
                    )
    else:
        def _fit_one(task: tuple[str, str]) -> dict[str, object]:
            focal_var, gene_id = task
            try:
                return _fit_gene_lmm_task(
                    response_matrix=response_matrix,
                    gene_to_guides=gene_to_guides,
                    mm=mm,
                    mm_reset=mm_reset,
                    gene_id=gene_id,
                    focal_var=focal_var,
                    allow_random_slope=bool(allow_random_slope),
                    min_guides_random_slope=int(min_guides_random_slope),
                    max_iter=int(max_iter),
                    fallback_to_meta=bool(fallback_to_meta),
                    meta_lookup=meta_lookup,
                )
            except Exception as exc:
                return {
                    "gene_id": str(gene_id),
                    "focal_var": str(focal_var),
                    "method": "failed",
                    "model": "",
                    "optimizer_full": "",
                    "optimizer_null": "",
                    "theta": np.nan,
                    "se_theta": np.nan,
                    "wald_z": np.nan,
                    "wald_p": np.nan,
                    "wald_ok": False,
                    "wald_p_adj": np.nan,
                    "lrt_stat": np.nan,
                    "lrt_stat_raw": np.nan,
                    "lrt_p": np.nan,
                    "lrt_ok": False,
                    "lrt_p_adj": np.nan,
                    "lrt_clipped": False,
                    "sigma_alpha": np.nan,
                    "tau": np.nan,
                    "converged_full": False,
                    "converged_null": False,
                    "m_guides_total": np.nan,
                    "m_guides_used": np.nan,
                    "n_samples": np.nan,
                    "n_obs": np.nan,
                    "fit_error": f"exception: {type(exc).__name__}: {exc}",
                }

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            for idx, row in enumerate(executor.map(_fit_one, tasks), start=1):
                out_rows.append(row)
                if progress and (idx % progress_every == 0 or idx == n_total):
                    now = time.monotonic()
                    if (now - last_print) >= progress_min_seconds or idx == n_total:
                        last_print = now
                        rate = idx / max(1e-9, (now - t0))
                        eta_s = (n_total - idx) / max(1e-9, rate)
                        print(
                            f"gene-level lmm: {idx}/{n_total} ({rate:.2f} task/s, eta~{eta_s/60.0:.1f} min)",
                            flush=True,
                        )

    columns = [
        "gene_id",
        "focal_var",
        "method",
        "model",
        "optimizer_full",
        "optimizer_null",
        "theta",
        "se_theta",
        "wald_z",
        "wald_p",
        "wald_ok",
        "wald_p_adj",
        "lrt_stat",
        "lrt_stat_raw",
        "lrt_p",
        "lrt_ok",
        "lrt_p_adj",
        "lrt_clipped",
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
