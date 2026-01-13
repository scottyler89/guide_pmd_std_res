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


GeneTMetaScope = Literal["flagged", "all"]


@dataclass(frozen=True)
class GeneTMetaConfig:
    """
    Targeted heavy-tailed sensitivity analysis on guide-level slopes (Plan C).

    This treats per-guide slopes as heavy-tailed around a gene-level mean theta_g:
      beta_j ~ approx Student-t_nu(theta, scale=sqrt(se_j^2 + tau2_meta))

    For simplicity and determinism, tau2_meta is taken from Plan B meta-analysis
    (or estimated per gene via DerSimonian-Laird if meta_results is not provided),
    and theta is fit via IRLS using the t scale-mixture weighting.
    """

    scope: GeneTMetaScope = "flagged"
    min_guides: int = 3
    nu: float = 4.0
    max_iter: int = 50
    tol: float = 1e-6

    def validate(self) -> None:
        if self.scope not in ("flagged", "all"):
            raise ValueError(f"invalid scope: {self.scope}")
        if int(self.min_guides) < 1:
            raise ValueError("min_guides must be >= 1")
        if float(self.nu) <= 0:
            raise ValueError("nu must be > 0")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be >= 1")
        if float(self.tol) <= 0:
            raise ValueError("tol must be > 0")


def _fit_tmeta_theta(
    beta: np.ndarray,
    se: np.ndarray,
    *,
    theta_init: float,
    tau2_meta: float,
    nu: float,
    max_iter: int,
    tol: float,
) -> dict[str, object]:
    beta = np.asarray(beta, dtype=float)
    se = np.asarray(se, dtype=float)
    if beta.shape != se.shape:
        raise ValueError("beta and se must have the same shape")

    var = se**2 + float(max(0.0, tau2_meta))
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        raise ValueError("non-positive or non-finite variance in t-meta fit")

    theta = float(theta_init) if np.isfinite(theta_init) else float(np.mean(beta))
    converged = False

    nu_f = float(nu)
    for it in range(int(max_iter)):
        r = (beta - theta) / np.sqrt(var)
        w = (nu_f + 1.0) / (nu_f + r**2)
        denom = np.sum(w / var)
        if denom <= 0 or not np.isfinite(denom):
            return {
                "fit_ok": False,
                "fit_error": "degenerate_theta_denominator",
                "theta": np.nan,
                "w": w,
                "n_iter": it + 1,
                "converged": False,
            }

        theta_new = float(np.sum(w * beta / var) / denom)
        if abs(theta_new - theta) < float(tol):
            theta = theta_new
            converged = True
            break
        theta = theta_new

    return {
        "fit_ok": True,
        "fit_error": "",
        "theta": float(theta),
        "w": w,
        "n_iter": int(it + 1),
        "converged": bool(converged),
        "tau2_meta": float(max(0.0, tau2_meta)),
    }


def compute_gene_tmeta(
    per_guide_ols: pd.DataFrame,
    annotation_table: pd.DataFrame,
    *,
    focal_vars: Sequence[str],
    gene_id_col: int = 1,
    meta_results: pd.DataFrame | None = None,
    flag_table: pd.DataFrame | None = None,
    config: GeneTMetaConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute targeted t-meta results (and per-guide weights) per (gene_id, focal_var).

    Returns:
      - gene_tmeta: per-gene fit summary
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

        beta_all = pd.to_numeric(sub["beta"], errors="coerce").to_numpy(dtype=float)
        se_all = pd.to_numeric(sub["se"], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(beta_all) & np.isfinite(se_all) & (se_all > 0)
        beta = beta_all[good]
        se = se_all[good]

        m_total = int(sub.shape[0])
        m_used = int(beta.size)

        if m_used < int(config.min_guides):
            gene_rows.append(
                {
                    "gene_id": str(gene_id),
                    "focal_var": str(focal_var),
                    "method": "tmeta",
                    "scope": str(config.scope),
                    "fit_ok": False,
                    "fit_error": "insufficient_guides",
                    "theta": np.nan,
                    "se_theta": np.nan,
                    "wald_z": np.nan,
                    "wald_p": np.nan,
                    "wald_ok": False,
                    "wald_p_adj": np.nan,
                    "nu": float(config.nu),
                    "n_iter": 0,
                    "converged": False,
                    "tau2_meta": np.nan,
                    "w_sum": np.nan,
                    "w_min": np.nan,
                    "w_max": np.nan,
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

        fit = _fit_tmeta_theta(
            beta,
            se,
            theta_init=theta_init,
            tau2_meta=tau2_meta,
            nu=float(config.nu),
            max_iter=int(config.max_iter),
            tol=float(config.tol),
        )

        fit_ok = bool(fit.get("fit_ok", False))
        theta = float(fit["theta"]) if fit_ok else np.nan
        w = np.asarray(fit.get("w", np.full(beta.shape, np.nan)), dtype=float)

        if fit_ok:
            var = se**2 + float(max(0.0, tau2_meta))
            denom = np.sum(w / var)
            se_theta = float(np.sqrt(1.0 / denom)) if denom > 0 and np.isfinite(denom) else np.nan
        else:
            se_theta = np.nan

        wald_z = float(theta / se_theta) if np.isfinite(theta) and np.isfinite(se_theta) and se_theta > 0 else np.nan
        wald_p = float(2 * norm.sf(abs(wald_z))) if np.isfinite(wald_z) else np.nan
        wald_ok = bool(np.isfinite(wald_p))

        gene_rows.append(
            {
                "gene_id": str(gene_id),
                "focal_var": str(focal_var),
                "method": "tmeta",
                "scope": str(config.scope),
                "fit_ok": fit_ok,
                "fit_error": str(fit.get("fit_error", "")),
                "theta": theta,
                "se_theta": se_theta,
                "wald_z": wald_z,
                "wald_p": wald_p,
                "wald_ok": wald_ok,
                "wald_p_adj": np.nan,  # filled below
                "nu": float(config.nu),
                "n_iter": int(fit.get("n_iter", 0)) if fit_ok else 0,
                "converged": bool(fit.get("converged", False)),
                "tau2_meta": float(max(0.0, tau2_meta)),
                "w_sum": float(np.sum(w)) if np.isfinite(w).any() else np.nan,
                "w_min": float(np.nanmin(w)) if np.isfinite(w).any() else np.nan,
                "w_max": float(np.nanmax(w)) if np.isfinite(w).any() else np.nan,
                "m_guides_total": float(m_total),
                "m_guides_used": float(m_used),
                "flag_reason": flag_reason_map.get(key, ""),
            }
        )

        guides = sub.loc[good, "guide_id"].astype(str).tolist()
        for guide_id, b, s, weight in zip(guides, beta.tolist(), se.tolist(), w.tolist(), strict=True):
            guide_rows.append(
                {
                    "gene_id": str(gene_id),
                    "focal_var": str(focal_var),
                    "guide_id": str(guide_id),
                    "beta": float(b),
                    "se": float(s),
                    "tmeta_weight": float(weight) if np.isfinite(weight) else np.nan,
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
        "nu",
        "n_iter",
        "converged",
        "tau2_meta",
        "w_sum",
        "w_min",
        "w_max",
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

    guide_columns = ["gene_id", "focal_var", "guide_id", "beta", "se", "tmeta_weight"]
    guide_out = (
        pd.DataFrame(guide_rows, columns=guide_columns)
        .sort_values(["focal_var", "gene_id", "guide_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return gene_out, guide_out

