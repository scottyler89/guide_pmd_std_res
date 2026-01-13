from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


GeneFlaggingScope = Literal["none", "all"]


@dataclass(frozen=True)
class GeneFlaggingConfig:
    """
    Consumer-layer policy for selecting genes for targeted robustness analyses (Plan C).

    Per DEV_RUBRIC.md:
    - Explicit and configurable.
    - No silent fallbacks: we return an inspectable table with reasons.

    This is not used to change primary results; it only controls which *additional*
    sensitivity analyses are computed.
    """

    scope: GeneFlaggingScope = "all"
    # Use meta-analysis heterogeneity (Q_p_adj) as a flag for discordant guides.
    het_q_max: float = 0.1
    # Flag when a substantial fraction of guides disagree in sign.
    frac_opposite_min: float = 0.25
    # Flag when any guide has a large |z| (per-guide OLS z=t/se).
    max_abs_z_min: float = 3.0
    # Flag when any guide has a large residual (requires residual_matrix input to QC).
    max_abs_resid_min: float | None = None
    # Minimum usable guides required for targeted fits.
    min_guides: int = 3

    def validate(self) -> None:
        if self.scope not in ("none", "all"):
            raise ValueError(f"invalid scope: {self.scope}")
        if not (0.0 < float(self.het_q_max) <= 1.0):
            raise ValueError("het_q_max must be in (0, 1]")
        if not (0.0 <= float(self.frac_opposite_min) <= 1.0):
            raise ValueError("frac_opposite_min must be in [0, 1]")
        if float(self.max_abs_z_min) < 0:
            raise ValueError("max_abs_z_min must be >= 0")
        if self.max_abs_resid_min is not None and float(self.max_abs_resid_min) < 0:
            raise ValueError("max_abs_resid_min must be >= 0 or None")
        if int(self.min_guides) < 1:
            raise ValueError("min_guides must be >= 1")


def compute_gene_flag_table(
    meta_results: pd.DataFrame,
    qc_results: pd.DataFrame,
    *,
    config: GeneFlaggingConfig,
) -> pd.DataFrame:
    """
    Return a per-(gene_id, focal_var) table indicating which genes should receive
    targeted robustness analyses and why.
    """
    config.validate()
    if config.scope == "none":
        out = qc_results[["gene_id", "focal_var"]].copy()
        out["flagged"] = False
        out["flag_reason"] = ""
        out = out.sort_values(["focal_var", "gene_id"], kind="mergesort").reset_index(drop=True)
        return out

    required_meta = {"gene_id", "focal_var", "Q_p_adj", "m_guides_used"}
    missing_meta = required_meta.difference(set(meta_results.columns))
    if missing_meta:
        raise ValueError(f"meta_results missing required column(s): {sorted(missing_meta)}")
    required_qc = {"gene_id", "focal_var", "m_guides_used", "frac_opposite_sign", "max_abs_z", "max_abs_resid"}
    missing_qc = required_qc.difference(set(qc_results.columns))
    if missing_qc:
        raise ValueError(f"qc_results missing required column(s): {sorted(missing_qc)}")

    meta = meta_results[["gene_id", "focal_var", "Q_p_adj", "m_guides_used"]].copy()
    meta["gene_id"] = meta["gene_id"].astype(str)
    meta["focal_var"] = meta["focal_var"].astype(str)

    qc_cols = ["gene_id", "focal_var", "m_guides_used", "frac_opposite_sign", "max_abs_z", "max_abs_resid"]
    qc = qc_results[qc_cols].copy()
    qc["gene_id"] = qc["gene_id"].astype(str)
    qc["focal_var"] = qc["focal_var"].astype(str)

    out = qc.merge(meta, on=["gene_id", "focal_var"], how="left", validate="one_to_one", suffixes=("_qc", "_meta"))
    out = out.sort_values(["focal_var", "gene_id"], kind="mergesort").reset_index(drop=True)

    # Use QC m_guides_used as the primary count (it matches per-guide OLS availability).
    m_guides = pd.to_numeric(out["m_guides_used_qc"], errors="coerce").to_numpy(dtype=float)
    enough_guides = np.isfinite(m_guides) & (m_guides >= float(config.min_guides))

    het_q = pd.to_numeric(out["Q_p_adj"], errors="coerce").to_numpy(dtype=float)
    frac_opp = pd.to_numeric(out["frac_opposite_sign"], errors="coerce").to_numpy(dtype=float)
    max_abs_z = pd.to_numeric(out["max_abs_z"], errors="coerce").to_numpy(dtype=float)
    max_abs_resid = pd.to_numeric(out["max_abs_resid"], errors="coerce").to_numpy(dtype=float)

    het_flag = np.isfinite(het_q) & (het_q <= float(config.het_q_max))
    opp_flag = np.isfinite(frac_opp) & (frac_opp >= float(config.frac_opposite_min))
    z_flag = np.isfinite(max_abs_z) & (max_abs_z >= float(config.max_abs_z_min))

    if config.max_abs_resid_min is None:
        resid_flag = np.zeros_like(het_flag, dtype=bool)
    else:
        resid_flag = np.isfinite(max_abs_resid) & (max_abs_resid >= float(config.max_abs_resid_min))

    flagged = enough_guides & (het_flag | opp_flag | z_flag | resid_flag)

    reasons = np.full(flagged.shape, "", dtype=object)
    for i in range(int(flagged.size)):
        if not bool(flagged[i]):
            continue
        parts: list[str] = []
        if bool(het_flag[i]):
            parts.append("het_q")
        if bool(opp_flag[i]):
            parts.append("opposite_sign")
        if bool(z_flag[i]):
            parts.append("max_abs_z")
        if bool(resid_flag[i]):
            parts.append("max_abs_resid")
        reasons[i] = "+".join(parts) if parts else "flagged"

    out["flagged"] = flagged.astype(bool)
    out["flag_reason"] = reasons.astype(str)
    return out[
        [
            "gene_id",
            "focal_var",
            "flagged",
            "flag_reason",
            "m_guides_used_qc",
            "Q_p_adj",
            "frac_opposite_sign",
            "max_abs_z",
            "max_abs_resid",
        ]
    ].rename(columns={"m_guides_used_qc": "m_guides_used"})

