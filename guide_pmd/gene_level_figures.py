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

