from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from .gene_level import _dersimonian_laird
from .gene_level import _get_gene_ids
from .gene_level import _nan_fdr


GeneMixtureScope = Literal["flagged", "all"]


@dataclass(frozen=True)
class GeneMixtureConfig:
    """
    Targeted mixture model on guide-level slopes (Plan C sensitivity analysis).

    We use a 2-component mixture to downweight "bad" / off-target guides:
    - Good component: beta ~ Normal(theta, se^2 + tau2_meta)
    - Bad component:  beta ~ Normal(theta0, se^2 + tau2_meta + sigma_bad^2)

    Only theta and pi are fit (EM); tau2_meta comes from Plan B meta-analysis (or is
    estimated per gene via DerSimonian-Laird if meta_results is not provided).
    """

    scope: GeneMixtureScope = "flagged"
    min_guides: int = 3
    theta0: float = 0.0
    bad_scale: float = 10.0
    max_iter: int = 50
    tol: float = 1e-6

    def validate(self) -> None:
        if self.scope not in ("flagged", "all"):
            raise ValueError(f"invalid scope: {self.scope}")
        if int(self.min_guides) < 1:
            raise ValueError("min_guides must be >= 1")
        if float(self.bad_scale) <= 0:
            raise ValueError("bad_scale must be > 0")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be >= 1")
        if float(self.tol) <= 0:
            raise ValueError("tol must be > 0")


def _logsumexp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def _fit_mixture_theta_pi(
    beta: np.ndarray,
    se: np.ndarray,
    *,
    theta_init: float,
    tau2_meta: float,
    theta0: float,
    bad_scale: float,
    max_iter: int,
    tol: float,
) -> dict[str, object]:
    beta = np.asarray(beta, dtype=float)
    se = np.asarray(se, dtype=float)
    if beta.shape != se.shape:
        raise ValueError("beta and se must have the same shape")

    var_good = se**2 + float(max(0.0, tau2_meta))
    if not np.all(np.isfinite(var_good)) or np.any(var_good <= 0):
        raise ValueError("non-positive or non-finite variance in mixture fit")

    sigma_bad2 = (float(bad_scale) ** 2) * float(np.median(var_good))
    var_bad = var_good + float(sigma_bad2)

    theta = float(theta_init) if np.isfinite(theta_init) else 0.0
    pi = 0.8
    converged = False

    for it in range(int(max_iter)):
        sd_good = np.sqrt(var_good)
        sd_bad = np.sqrt(var_bad)

        log1 = np.log(pi) + norm.logpdf(beta, loc=theta, scale=sd_good)
        log0 = np.log(1.0 - pi) + norm.logpdf(beta, loc=float(theta0), scale=sd_bad)
        denom = _logsumexp(log1, log0)
        r = np.exp(log1 - denom)

        denom_theta = np.sum(r / var_good)
        if denom_theta <= 0 or not np.isfinite(denom_theta):
            return {
                "fit_ok": False,
                "fit_error": "degenerate theta denominator",
                "theta": np.nan,
                "pi_good": np.nan,
                "r": r,
                "n_iter": it + 1,
                "converged": False,
            }

        theta_new = float(np.sum(r * beta / var_good) / denom_theta)
        pi_new = float(np.clip(np.mean(r), 1e-3, 1.0 - 1e-3))

        if (abs(theta_new - theta) < float(tol)) and (abs(pi_new - pi) < float(tol)):
            theta = theta_new
            pi = pi_new
            converged = True
            break

        theta = theta_new
        pi = pi_new

    return {
        "fit_ok": True,
        "fit_error": "",
        "theta": float(theta),
        "pi_good": float(pi),
        "r": r,
        "n_iter": int(it + 1),
        "converged": bool(converged),
        "tau2_meta": float(max(0.0, tau2_meta)),
        "sigma_bad2": float(sigma_bad2),
    }


def compute_gene_mixture(
    per_guide_ols: pd.DataFrame,
    annotation_table: pd.DataFrame,
    *,
    focal_vars: Sequence[str],
    gene_id_col: int = 1,
    meta_results: pd.DataFrame | None = None,
    flag_table: pd.DataFrame | None = None,
    config: GeneMixtureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute targeted mixture results (and per-guide posterior weights) per (gene_id, focal_var).

    Returns:
      - gene_mixture: per-gene mixture fit summary
      - guide_details: per-guide weights for each fitted gene
    """
    config.validate()
    focal_vars = [str(v) for v in focal_vars]
    if not focal_vars:
        raise ValueError("focal_vars must not be empty")

    required_pg = {"guide_id", "focal_var", "beta", "se"}
    missing_pg = required_pg.difference(set(per_guide_ols.columns))
    if missing_pg:
        raise ValueError(f"per_guide_ols missing required column(s): {sorted(missing_pg)}")

    gene_ids = _get_gene_ids(annotation_table, gene_id_col)
    pg = per_guide_ols.copy()
    pg["guide_id"] = pg["guide_id"].astype(str)
    pg["focal_var"] = pg["focal_var"].astype(str)
    pg = pg[pg["focal_var"].isin(focal_vars)]
    pg = pg.merge(gene_ids, left_on="guide_id", right_index=True, how="left")
    if pg["gene_id"].isna().any():
        missing = pg.loc[pg["gene_id"].isna(), "guide_id"].unique().tolist()
        raise ValueError(f"missing gene ids for {len(missing)} guide(s), e.g. {missing[:5]}")

    meta_map: dict[tuple[str, str], dict[str, float]] = {}
    if meta_results is not None and (not meta_results.empty):
        required_meta = {"gene_id", "focal_var", "theta", "tau2"}
        missing_meta = required_meta.difference(set(meta_results.columns))
        if missing_meta:
            raise ValueError(f"meta_results missing required column(s): {sorted(missing_meta)}")
        for row in meta_results.itertuples(index=False):
            key = (str(getattr(row, "gene_id")), str(getattr(row, "focal_var")))
            meta_map[key] = {
                "theta": float(getattr(row, "theta")),
                "tau2": float(getattr(row, "tau2")),
            }

    flagged_set: set[tuple[str, str]] | None = None
    flag_reason_map: dict[tuple[str, str], str] = {}
    if config.scope == "flagged":
        if flag_table is None:
            raise ValueError("flag_table is required when config.scope='flagged'")
        required_flag = {"gene_id", "focal_var", "flagged"}
        missing_flag = required_flag.difference(set(flag_table.columns))
        if missing_flag:
            raise ValueError(f"flag_table missing required column(s): {sorted(missing_flag)}")
        flagged = flag_table.copy()
        flagged["gene_id"] = flagged["gene_id"].astype(str)
        flagged["focal_var"] = flagged["focal_var"].astype(str)
        flagged = flagged.loc[flagged["flagged"].astype(bool)]
        flagged_set = set(zip(flagged["gene_id"].tolist(), flagged["focal_var"].tolist(), strict=True))
        if "flag_reason" in flagged.columns:
            for gene_id, focal_var, reason in zip(
                flagged["gene_id"].tolist(),
                flagged["focal_var"].tolist(),
                flagged["flag_reason"].astype(str).tolist(),
                strict=True,
            ):
                flag_reason_map[(gene_id, focal_var)] = str(reason)

    gene_rows: list[dict[str, object]] = []
    guide_rows: list[dict[str, object]] = []

    for (gene_id, focal_var), sub in pg.groupby(["gene_id", "focal_var"], sort=True):
        key = (str(gene_id), str(focal_var))
        if flagged_set is not None and key not in flagged_set:
            continue

        beta = pd.to_numeric(sub["beta"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(sub["se"], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(beta) & np.isfinite(se) & (se > 0)
        beta = beta[good]
        se = se[good]

        m_total = int(sub.shape[0])
        m_used = int(beta.size)

        if m_used < int(config.min_guides):
            gene_rows.append(
                {
                    "gene_id": str(gene_id),
                    "focal_var": str(focal_var),
                    "method": "mixture",
                    "scope": str(config.scope),
                    "fit_ok": False,
                    "fit_error": "insufficient_guides",
                    "theta": np.nan,
                    "se_theta": np.nan,
                    "wald_z": np.nan,
                    "wald_p": np.nan,
                    "wald_ok": False,
                    "wald_p_adj": np.nan,
                    "pi_good": np.nan,
                    "m_eff_guides": np.nan,
                    "w_min": np.nan,
                    "w_max": np.nan,
                    "tau2_meta": np.nan,
                    "sigma_bad2": np.nan,
                    "theta0": float(config.theta0),
                    "bad_scale": float(config.bad_scale),
                    "n_iter": 0,
                    "converged": False,
                    "m_guides_total": float(m_total),
                    "m_guides_used": float(m_used),
                    "flag_reason": flag_reason_map.get(key, ""),
                }
            )
            continue

        if key in meta_map:
            theta_init = meta_map[key]["theta"]
            tau2_meta = meta_map[key]["tau2"]
        else:
            meta = _dersimonian_laird(beta, se)
            theta_init = float(meta["theta"])
            tau2_meta = float(meta["tau2"]) if np.isfinite(meta["tau2"]) else 0.0

        fit = _fit_mixture_theta_pi(
            beta,
            se,
            theta_init=theta_init,
            tau2_meta=tau2_meta,
            theta0=float(config.theta0),
            bad_scale=float(config.bad_scale),
            max_iter=int(config.max_iter),
            tol=float(config.tol),
        )

        theta = float(fit["theta"]) if fit.get("fit_ok") else np.nan
        r = np.asarray(fit.get("r", np.full(beta.shape, np.nan)), dtype=float)
        var_good = se**2 + float(max(0.0, tau2_meta))
        denom = np.sum(r / var_good)
        se_theta = float(np.sqrt(1.0 / denom)) if denom > 0 and np.isfinite(denom) else np.nan
        wald_z = float(theta / se_theta) if np.isfinite(theta) and np.isfinite(se_theta) and se_theta > 0 else np.nan
        wald_p = float(2 * norm.sf(abs(wald_z))) if np.isfinite(wald_z) else np.nan
        wald_ok = bool(np.isfinite(wald_p))

        gene_rows.append(
            {
                "gene_id": str(gene_id),
                "focal_var": str(focal_var),
                "method": "mixture",
                "scope": str(config.scope),
                "fit_ok": bool(fit.get("fit_ok", False)),
                "fit_error": str(fit.get("fit_error", "")),
                "theta": theta,
                "se_theta": se_theta,
                "wald_z": wald_z,
                "wald_p": wald_p,
                "wald_ok": wald_ok,
                "wald_p_adj": np.nan,  # filled below
                "pi_good": float(fit.get("pi_good", np.nan)),
                "m_eff_guides": float(np.sum(r)) if np.isfinite(r).any() else np.nan,
                "w_min": float(np.nanmin(r)) if np.isfinite(r).any() else np.nan,
                "w_max": float(np.nanmax(r)) if np.isfinite(r).any() else np.nan,
                "tau2_meta": float(max(0.0, tau2_meta)),
                "sigma_bad2": float(fit.get("sigma_bad2", np.nan)),
                "theta0": float(config.theta0),
                "bad_scale": float(config.bad_scale),
                "n_iter": int(fit.get("n_iter", 0)) if fit.get("fit_ok") else 0,
                "converged": bool(fit.get("converged", False)),
                "m_guides_total": float(m_total),
                "m_guides_used": float(m_used),
                "flag_reason": flag_reason_map.get(key, ""),
            }
        )

        guides = sub.loc[good, "guide_id"].astype(str).tolist()
        for guide_id, b, s, w in zip(guides, beta.tolist(), se.tolist(), r.tolist(), strict=True):
            guide_rows.append(
                {
                    "gene_id": str(gene_id),
                    "focal_var": str(focal_var),
                    "guide_id": str(guide_id),
                    "beta": float(b),
                    "se": float(s),
                    "mixture_prob_good": float(w) if np.isfinite(w) else np.nan,
                }
            )

    gene_columns = [
        "gene_id",
        "focal_var",
        "method",
        "scope",
        "fit_ok",
        "fit_error",
        "theta",
        "se_theta",
        "wald_z",
        "wald_p",
        "wald_ok",
        "wald_p_adj",
        "pi_good",
        "m_eff_guides",
        "w_min",
        "w_max",
        "tau2_meta",
        "sigma_bad2",
        "theta0",
        "bad_scale",
        "n_iter",
        "converged",
        "m_guides_total",
        "m_guides_used",
        "flag_reason",
    ]
    gene_out = (
        pd.DataFrame(gene_rows, columns=gene_columns)
        .sort_values(["focal_var", "gene_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if not gene_out.empty:
        gene_out["wald_p_adj"] = gene_out.groupby("focal_var", sort=False)["wald_p"].transform(_nan_fdr)

    guide_columns = ["gene_id", "focal_var", "guide_id", "beta", "se", "mixture_prob_good"]
    guide_out = (
        pd.DataFrame(guide_rows, columns=guide_columns)
        .sort_values(["focal_var", "gene_id", "guide_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return gene_out, guide_out
