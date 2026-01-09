from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import pandas as pd


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for gene-level figure generation; install with `pip install matplotlib`"
        ) from exc
    return plt


def _safe_neg_log10_p(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-300, 1.0)
    return -np.log10(p)


def _sanitize_filename_component(value: str) -> str:
    safe = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_.")
    return out or "value"


def _write_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    out_path: str,
) -> None:
    plt = _require_matplotlib()
    data = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)

    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, s=10, alpha=0.7, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_volcano(
    df: pd.DataFrame,
    *,
    effect_col: str,
    p_col: str,
    title: str,
    out_path: str,
) -> None:
    plt = _require_matplotlib()
    data = df[[effect_col, p_col]].replace([np.inf, -np.inf], np.nan).dropna()
    x = data[effect_col].to_numpy(dtype=float)
    y = _safe_neg_log10_p(data[p_col].to_numpy(dtype=float))

    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, s=10, alpha=0.7, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(effect_col)
    ax.set_ylabel(f"-log10({p_col})")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_gene_level_figures(
    output_dir: str,
    *,
    prefix: str,
    gene_meta: pd.DataFrame | None = None,
    gene_lmm: pd.DataFrame | None = None,
    gene_qc: pd.DataFrame | None = None,
) -> list[str]:
    """
    Write deterministic figures for the gene-level tables that are present.

    This is a consumer-only layer (no stats recomputation).
    """
    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []

    def _for_each_focal_var(df: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
        for focal_var, sub in df.groupby("focal_var", sort=True):
            sub = sub.sort_values("gene_id", kind="mergesort").reset_index(drop=True)
            yield str(focal_var), sub

    if gene_meta is not None and (not gene_meta.empty):
        for focal_var, sub in _for_each_focal_var(gene_meta):
            out_path = os.path.join(output_dir, f"{prefix}_gene_meta_volcano__{focal_var}.png")
            _write_volcano(
                sub,
                effect_col="theta",
                p_col="p",
                title=f"Gene-level meta volcano ({focal_var})",
                out_path=out_path,
            )
            written.append(out_path)

            out_path = os.path.join(output_dir, f"{prefix}_gene_meta_tau_vs_effect__{focal_var}.png")
            _write_scatter(
                sub.assign(abs_theta=np.abs(sub["theta"].to_numpy(dtype=float))),
                x_col="abs_theta",
                y_col="tau",
                title=f"Meta heterogeneity: tau vs |theta| ({focal_var})",
                x_label="|theta|",
                y_label="tau",
                out_path=out_path,
            )
            written.append(out_path)

            out_path = os.path.join(output_dir, f"{prefix}_gene_meta_sign_agreement_vs_p__{focal_var}.png")
            tmp = sub.assign(neglog10_p=_safe_neg_log10_p(sub["p"].to_numpy(dtype=float)))
            _write_scatter(
                tmp,
                x_col="sign_agreement",
                y_col="neglog10_p",
                title=f"Meta QC: sign agreement vs -log10(p) ({focal_var})",
                x_label="sign_agreement",
                y_label="-log10(p)",
                out_path=out_path,
            )
            written.append(out_path)

    if gene_lmm is not None and (not gene_lmm.empty):
        for focal_var, sub in _for_each_focal_var(gene_lmm):
            out_path = os.path.join(output_dir, f"{prefix}_gene_lmm_volcano__{focal_var}.png")
            _write_volcano(
                sub,
                effect_col="theta",
                p_col="p_primary",
                title=f"Gene-level LMM volcano ({focal_var})",
                out_path=out_path,
            )
            written.append(out_path)

            if "tau" in sub.columns:
                out_path = os.path.join(output_dir, f"{prefix}_gene_lmm_tau_vs_effect__{focal_var}.png")
                _write_scatter(
                    sub.assign(abs_theta=np.abs(sub["theta"].to_numpy(dtype=float))),
                    x_col="abs_theta",
                    y_col="tau",
                    title=f"LMM heterogeneity: tau vs |theta| ({focal_var})",
                    x_label="|theta|",
                    y_label="tau",
                    out_path=out_path,
                )
                written.append(out_path)

    if (
        (gene_meta is not None)
        and (gene_lmm is not None)
        and (not gene_meta.empty)
        and (not gene_lmm.empty)
    ):
        meta_cols = ["gene_id", "focal_var", "theta", "p"]
        lmm_cols = ["gene_id", "focal_var", "theta", "p_primary"]
        meta_sub = gene_meta[meta_cols].rename(columns={"theta": "theta_meta", "p": "p_meta"})
        lmm_sub = gene_lmm[lmm_cols].rename(columns={"theta": "theta_lmm", "p_primary": "p_lmm"})
        joined = meta_sub.merge(lmm_sub, on=["gene_id", "focal_var"], how="inner")
        if not joined.empty:
            for focal_var, sub in _for_each_focal_var(joined):
                out_path = os.path.join(output_dir, f"{prefix}_gene_compare_theta__{focal_var}.png")
                _write_scatter(
                    sub,
                    x_col="theta_meta",
                    y_col="theta_lmm",
                    title=f"Theta comparison: meta vs LMM ({focal_var})",
                    x_label="theta (meta)",
                    y_label="theta (lmm)",
                    out_path=out_path,
                )
                written.append(out_path)

                tmp = sub.assign(neglog10_p_meta=_safe_neg_log10_p(sub["p_meta"]), neglog10_p_lmm=_safe_neg_log10_p(sub["p_lmm"]))
                out_path = os.path.join(output_dir, f"{prefix}_gene_compare_p__{focal_var}.png")
                _write_scatter(
                    tmp,
                    x_col="neglog10_p_meta",
                    y_col="neglog10_p_lmm",
                    title=f"-log10(p) comparison: meta vs LMM ({focal_var})",
                    x_label="-log10(p) meta",
                    y_label="-log10(p) lmm",
                    out_path=out_path,
                )
                written.append(out_path)

    if gene_qc is not None and (not gene_qc.empty):
        for focal_var, sub in _for_each_focal_var(gene_qc):
            out_path = os.path.join(output_dir, f"{prefix}_gene_qc_sign_agreement__{focal_var}.png")
            _write_scatter(
                sub,
                x_col="beta_median",
                y_col="sign_agreement",
                title=f"QC: sign agreement vs beta_median ({focal_var})",
                x_label="beta_median",
                y_label="sign_agreement",
                out_path=out_path,
            )
            written.append(out_path)

    return written


def write_gene_forest_plots(
    per_guide_ols: pd.DataFrame,
    output_dir: str,
    *,
    prefix: str,
    forest_genes: list[str],
    focal_vars: list[str] | None = None,
) -> list[str]:
    """
    Write per-gene forest plots from per-guide OLS results.

    This requires explicit gene selection (no implicit "top-N" heuristics).
    """
    if not forest_genes:
        raise ValueError("forest_genes must not be empty")

    required_cols = {"guide_id", "gene_id", "focal_var", "beta", "se"}
    missing = required_cols.difference(per_guide_ols.columns)
    if missing:
        raise ValueError(f"per_guide_ols missing required column(s): {sorted(missing)}")

    if focal_vars is not None:
        focal_vars = list(focal_vars)
        per_guide_ols = per_guide_ols[per_guide_ols["focal_var"].isin(focal_vars)]

    per_guide_ols = per_guide_ols.copy()
    per_guide_ols["gene_id"] = per_guide_ols["gene_id"].astype(str)
    per_guide_ols = per_guide_ols[per_guide_ols["gene_id"].isin([str(g) for g in forest_genes])]

    os.makedirs(output_dir, exist_ok=True)
    plt = _require_matplotlib()
    written: list[str] = []

    for (gene_id, focal_var), sub in per_guide_ols.groupby(["gene_id", "focal_var"], sort=True):
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["beta", "se"])
        if sub.empty:
            continue
        sub = sub.sort_values(["beta", "guide_id"], kind="mergesort").reset_index(drop=True)

        beta = sub["beta"].to_numpy(dtype=float)
        se = sub["se"].to_numpy(dtype=float)
        ci_lo = beta - 1.96 * se
        ci_hi = beta + 1.96 * se
        y = np.arange(sub.shape[0], dtype=float)

        fig_h = max(3.0, 0.25 * float(sub.shape[0]) + 1.0)
        fig = plt.figure(figsize=(7, fig_h), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.hlines(y, ci_lo, ci_hi, color="black", linewidth=1)
        ax.scatter(beta, y, color="black", s=14)
        ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["guide_id"].astype(str).tolist())
        ax.set_xlabel("beta (per-guide OLS) with 95% CI")
        ax.set_title(f"Forest plot: {gene_id} ({focal_var})")
        ax.grid(True, axis="x", linewidth=0.3, alpha=0.5)
        fig.tight_layout()

        gene_safe = _sanitize_filename_component(gene_id)
        focal_safe = _sanitize_filename_component(focal_var)
        out_path = os.path.join(output_dir, f"{prefix}_gene_forest__{focal_safe}__{gene_safe}.png")
        fig.savefig(out_path)
        plt.close(fig)
        written.append(out_path)

    return written
