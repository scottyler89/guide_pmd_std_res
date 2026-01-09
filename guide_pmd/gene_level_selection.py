from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .gene_level import _align_model_matrix
from .gene_level import _get_gene_ids


GeneLmmScope = Literal["all", "meta_fdr", "meta_or_het_fdr", "explicit", "none"]


@dataclass(frozen=True)
class GeneLmmSelectionConfig:
    """
    Policy for selecting which genes to fit with Plan A (mixed model).

    This is a consumer-layer policy object (see DEV_RUBRIC.md):
    - Explicit and configurable.
    - No silent fallbacks: selection decisions are returned as an inspectable table.
    """

    scope: GeneLmmScope = "meta_or_het_fdr"
    q_meta: float = 0.1
    q_het: float = 0.1
    audit_n: int = 50
    audit_seed: int = 123456
    max_genes_per_focal_var: int | None = None
    explicit_genes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.scope not in ("all", "meta_fdr", "meta_or_het_fdr", "explicit", "none"):
            raise ValueError(f"invalid scope: {self.scope}")
        if not (0.0 < float(self.q_meta) <= 1.0):
            raise ValueError(f"q_meta must be in (0, 1], got {self.q_meta}")
        if not (0.0 < float(self.q_het) <= 1.0):
            raise ValueError(f"q_het must be in (0, 1], got {self.q_het}")
        if int(self.audit_n) < 0:
            raise ValueError(f"audit_n must be >= 0, got {self.audit_n}")
        if self.max_genes_per_focal_var is not None and int(self.max_genes_per_focal_var) < 1:
            raise ValueError(f"max_genes_per_focal_var must be >= 1 or None, got {self.max_genes_per_focal_var}")


def _prepare_model_matrix(model_matrix: pd.DataFrame, *, add_intercept: bool) -> pd.DataFrame:
    mm = model_matrix.copy()
    if add_intercept and "Intercept" not in mm.columns:
        mm.insert(0, "Intercept", 1.0)
    try:
        mm = mm.apply(pd.to_numeric)
    except Exception as exc:  # pragma: no cover
        raise ValueError("model_matrix must be numeric") from exc
    return mm


def _is_focal_identifiable(mm: pd.DataFrame, *, focal_var: str) -> bool:
    if focal_var not in mm.columns:
        raise ValueError(f"focal var missing from model_matrix: {focal_var}")
    fixed_cols_full = list(mm.columns)
    fixed_cols_null = [c for c in fixed_cols_full if c != focal_var]
    X_full = mm[fixed_cols_full].to_numpy(dtype=float)
    X_null = mm[fixed_cols_null].to_numpy(dtype=float)
    rank_full = int(np.linalg.matrix_rank(X_full))
    rank_null = int(np.linalg.matrix_rank(X_null))
    return rank_full > rank_null


def compute_gene_lmm_feasibility(
    response_matrix: pd.DataFrame,
    annotation_table: pd.DataFrame,
    model_matrix: pd.DataFrame,
    *,
    focal_vars: list[str],
    gene_id_col: int = 1,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """
    Compute per-(gene, focal_var) feasibility gates for mixed-model fitting.

    These are identifiability checks (not "signal" heuristics):
    - Focal term is estimable from the model matrix.
    - Response is non-degenerate for the gene.
    - Gene has at least 2 guides in the response matrix.
    """
    if not focal_vars:
        raise ValueError("focal_vars must not be empty")

    response = response_matrix.copy()
    response.index = response.index.astype(str)
    if response.index.has_duplicates:
        raise ValueError("response_matrix index must not contain duplicates (guide_id)")

    gene_ids = _get_gene_ids(annotation_table, gene_id_col).copy()
    gene_ids.index = gene_ids.index.astype(str)
    if gene_ids.index.has_duplicates:
        raise ValueError("annotation_table index must not contain duplicates (guide_id)")

    mm = _align_model_matrix(model_matrix, list(response.columns))
    mm = _prepare_model_matrix(mm, add_intercept=add_intercept)

    genes = sorted(gene_ids.unique().tolist())

    guides_in_response = set(response.index.tolist())
    gene_ids_present = gene_ids[gene_ids.index.isin(guides_in_response)]

    guide_counts = gene_ids_present.value_counts().astype(int).to_dict()
    gene_has_guides = {g: (int(guide_counts.get(g, 0)) > 0) for g in genes}
    gene_m_guides = {g: int(guide_counts.get(g, 0)) for g in genes}

    # Precompute response degeneracy per gene.
    gene_deg = {}
    for gene_id in genes:
        m = gene_m_guides[gene_id]
        if m == 0:
            gene_deg[gene_id] = True
            continue
        guides = gene_ids_present.index[gene_ids_present == gene_id].tolist()
        sub = response.loc[guides, :]
        var = float(np.var(sub.to_numpy(dtype=float)))
        gene_deg[gene_id] = bool(var == 0.0)

    focal_identifiable = {str(v): _is_focal_identifiable(mm, focal_var=str(v)) for v in focal_vars}

    rows: list[dict[str, object]] = []
    for focal_var in sorted([str(v) for v in focal_vars]):
        identifiable = bool(focal_identifiable[focal_var])
        for gene_id in genes:
            m_guides = int(gene_m_guides[gene_id])
            skip_reason = ""
            if not identifiable:
                skip_reason = "focal_var_not_identifiable"
            elif not gene_has_guides[gene_id]:
                skip_reason = "no_guides_in_response"
            elif gene_deg[gene_id]:
                skip_reason = "degenerate_response"
            elif m_guides < 2:
                skip_reason = "insufficient_guides"
            rows.append(
                {
                    "gene_id": str(gene_id),
                    "focal_var": str(focal_var),
                    "m_guides_total": float(m_guides),
                    "feasible": bool(skip_reason == ""),
                    "skip_reason": skip_reason,
                }
            )

    out = pd.DataFrame(rows).sort_values(["focal_var", "gene_id"], kind="mergesort").reset_index(drop=True)
    return out


def compute_gene_lmm_selection(
    meta_results: pd.DataFrame,
    feasibility_table: pd.DataFrame,
    *,
    config: GeneLmmSelectionConfig,
) -> pd.DataFrame:
    """
    Select per-(gene, focal_var) rows for Plan A (mixed model) fitting.

    Returns a fully inspectable selection table suitable for writing as
    ``PMD_std_res_gene_lmm_selection.tsv``.
    """
    config.validate()

    required_meta = {"gene_id", "focal_var", "theta", "se_theta", "p", "p_adj", "Q_p", "Q_p_adj", "m_guides_used"}
    missing_meta = required_meta.difference(set(meta_results.columns))
    if missing_meta:
        raise ValueError(f"meta_results missing required column(s): {sorted(missing_meta)}")
    required_feas = {"gene_id", "focal_var", "feasible", "skip_reason", "m_guides_total"}
    missing_feas = required_feas.difference(set(feasibility_table.columns))
    if missing_feas:
        raise ValueError(f"feasibility_table missing required column(s): {sorted(missing_feas)}")

    meta_cols = ["gene_id", "focal_var", "theta", "se_theta", "p", "p_adj", "Q_p", "Q_p_adj", "m_guides_used"]
    meta = meta_results[meta_cols].copy()
    meta["gene_id"] = meta["gene_id"].astype(str)
    meta["focal_var"] = meta["focal_var"].astype(str)

    feas = feasibility_table[["gene_id", "focal_var", "m_guides_total", "feasible", "skip_reason"]].copy()
    feas["gene_id"] = feas["gene_id"].astype(str)
    feas["focal_var"] = feas["focal_var"].astype(str)

    out = feas.merge(meta, on=["gene_id", "focal_var"], how="left", validate="one_to_one")
    out = out.sort_values(["focal_var", "gene_id"], kind="mergesort").reset_index(drop=True)

    out["selected"] = False
    out["selection_reason"] = ""

    feasible = out["feasible"].astype(bool)
    if config.scope == "none":
        return out
    if config.scope == "all":
        out.loc[feasible, "selected"] = True
        out.loc[feasible, "selection_reason"] = "all"
        return out

    explicit = set(str(g) for g in config.explicit_genes)
    if config.scope == "explicit":
        sel = feasible & out["gene_id"].isin(explicit)
        out.loc[sel, "selected"] = True
        out.loc[sel, "selection_reason"] = "explicit"
        return out

    meta_q = pd.to_numeric(out["p_adj"], errors="coerce").to_numpy(dtype=float)
    het_q = pd.to_numeric(out["Q_p_adj"], errors="coerce").to_numpy(dtype=float)

    meta_hit = feasible.to_numpy(dtype=bool) & np.isfinite(meta_q) & (meta_q <= float(config.q_meta))
    het_hit = feasible.to_numpy(dtype=bool) & np.isfinite(het_q) & (het_q <= float(config.q_het))

    if config.scope == "meta_fdr":
        selected = meta_hit
    elif config.scope == "meta_or_het_fdr":
        selected = meta_hit | het_hit
    else:  # pragma: no cover
        raise ValueError(f"unhandled scope: {config.scope}")

    out.loc[selected, "selected"] = True
    out.loc[selected & meta_hit & het_hit, "selection_reason"] = "meta_and_het_fdr"
    out.loc[selected & meta_hit & ~het_hit, "selection_reason"] = "meta_fdr"
    out.loc[selected & ~meta_hit & het_hit, "selection_reason"] = "het_fdr"

    audit_n = int(config.audit_n)
    if audit_n > 0:
        for focal_var, sub_idx in out.groupby("focal_var", sort=True).groups.items():
            seed = int(config.audit_seed)
            focal_bytes = str(focal_var).encode("utf-8")
            for b in focal_bytes:
                seed = (seed * 131 + int(b)) % (2**32)
            rng = np.random.default_rng(seed)
            sub = out.loc[sub_idx]
            candidates = sub[(sub["feasible"].astype(bool)) & (~sub["selected"].astype(bool))]["gene_id"].tolist()
            candidates = sorted(set(str(g) for g in candidates))
            if not candidates:
                continue
            if audit_n >= len(candidates):
                audit_genes = candidates
            else:
                pick = rng.choice(len(candidates), size=audit_n, replace=False)
                audit_genes = [candidates[int(i)] for i in sorted(pick)]
            if audit_genes:
                mask = (out["focal_var"] == focal_var) & out["gene_id"].isin(audit_genes) & feasible
                out.loc[mask, "selected"] = True
                out.loc[mask, "selection_reason"] = "audit"

    cap = config.max_genes_per_focal_var
    if cap is not None:
        cap = int(cap)
        for focal_var, sub_idx in out.groupby("focal_var", sort=True).groups.items():
            sub = out.loc[sub_idx]
            sel_idx = sub.index[sub["selected"].astype(bool)].tolist()
            if len(sel_idx) <= cap:
                continue
            p_adj = pd.to_numeric(sub.loc[sel_idx, "p_adj"], errors="coerce").to_numpy(dtype=float)
            q_adj = pd.to_numeric(sub.loc[sel_idx, "Q_p_adj"], errors="coerce").to_numpy(dtype=float)
            score = np.fmin(p_adj, q_adj)
            score[~np.isfinite(score)] = 1.0
            ranked = [idx for _score, idx in sorted(zip(score.tolist(), sel_idx, strict=True))]
            keep = set(ranked[:cap])
            drop = [idx for idx in sel_idx if idx not in keep]
            if drop:
                out.loc[drop, "selected"] = False
                out.loc[drop, "selection_reason"] = "cap_dropped"

    return out
