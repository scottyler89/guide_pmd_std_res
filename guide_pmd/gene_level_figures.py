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


def _write_histogram(
    df: pd.DataFrame,
    *,
    x_col: str,
    title: str,
    x_label: str,
    out_path: str,
    bins: int = 50,
) -> None:
    plt = _require_matplotlib()
    data = df[[x_col]].replace([np.inf, -np.inf], np.nan).dropna()
    x = data[x_col].to_numpy(dtype=float)

    fig = plt.figure(figsize=(6, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x, bins=int(bins), color="#1f77b4", alpha=0.85, edgecolor="none")
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("count")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_confusion_matrix(
    counts: np.ndarray,
    *,
    title: str,
    x_label: str,
    y_label: str,
    out_path: str,
    x_ticklabels: tuple[str, str] = ("False", "True"),
    y_ticklabels: tuple[str, str] = ("False", "True"),
) -> None:
    plt = _require_matplotlib()
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (2, 2):
        raise ValueError("confusion matrix counts must be 2x2")

    fig = plt.figure(figsize=(4.6, 4.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(counts, cmap="Blues", vmin=0.0)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(list(x_ticklabels))
    ax.set_yticklabels(list(y_ticklabels))

    for (i, j), val in np.ndenumerate(counts):
        ax.text(j, i, f"{int(val)}", ha="center", va="center", color="black", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_scatter_categories(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    cat_x: pd.Series,
    cat_y: pd.Series,
    title: str,
    x_label: str,
    y_label: str,
    out_path: str,
) -> None:
    plt = _require_matplotlib()
    data = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return
    idx = data.index

    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    cx = cat_x.reindex(idx).fillna(False).astype(bool).to_numpy()
    cy = cat_y.reindex(idx).fillna(False).astype(bool).to_numpy()

    groups = [
        ("neither", (~cx) & (~cy), "#9aa0a6"),
        ("x_only", (cx) & (~cy), "#e67e22"),
        ("y_only", (~cx) & (cy), "#8e44ad"),
        ("both", (cx) & (cy), "#2ecc71"),
    ]

    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    for label, mask, color in groups:
        if not np.any(mask):
            continue
        ax.scatter(x[mask], y[mask], s=10, alpha=0.75, edgecolors="none", color=color, label=label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="best", fontsize=8, frameon=False)
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
    agreement_q: float = 0.1,
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
            if "lrt_p" in sub.columns:
                out_path = os.path.join(output_dir, f"{prefix}_gene_lmm_volcano_lrt__{focal_var}.png")
                _write_volcano(
                    sub,
                    effect_col="theta",
                    p_col="lrt_p",
                    title=f"Gene-level LMM volcano (LRT; {focal_var})",
                    out_path=out_path,
                )
                written.append(out_path)

            if "wald_p" in sub.columns:
                out_path = os.path.join(output_dir, f"{prefix}_gene_lmm_volcano_wald__{focal_var}.png")
                _write_volcano(
                    sub,
                    effect_col="theta",
                    p_col="wald_p",
                    title=f"Gene-level LMM volcano (Wald; {focal_var})",
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
        meta_cols = ["gene_id", "focal_var", "theta", "p", "p_adj"]
        lmm_cols = ["gene_id", "focal_var", "theta", "lrt_p", "wald_p", "lrt_p_adj", "wald_p_adj", "method"]
        meta_sub = gene_meta[meta_cols].rename(columns={"theta": "theta_meta", "p": "p_meta"})
        lmm_sub = gene_lmm[lmm_cols].rename(columns={"theta": "theta_lmm"})
        joined = meta_sub.merge(lmm_sub, on=["gene_id", "focal_var"], how="inner")
        if not joined.empty:
            q = float(agreement_q)
            q_safe = _sanitize_filename_component(f"q{q:g}")
            for focal_var, sub in _for_each_focal_var(joined):
                sub = sub.loc[sub["method"].astype(str) == "lmm"].copy()
                if sub.empty:
                    continue
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

                if "lrt_p" in sub.columns:
                    tmp = sub.assign(
                        neglog10_p_meta=_safe_neg_log10_p(sub["p_meta"]),
                        neglog10_p_lmm_lrt=_safe_neg_log10_p(sub["lrt_p"]),
                    )
                    out_path = os.path.join(output_dir, f"{prefix}_gene_compare_p_lrt__{focal_var}.png")
                    _write_scatter(
                        tmp,
                        x_col="neglog10_p_meta",
                        y_col="neglog10_p_lmm_lrt",
                        title=f"-log10(p) comparison: meta vs LMM LRT ({focal_var})",
                        x_label="-log10(p) meta",
                        y_label="-log10(p) lmm (lrt)",
                        out_path=out_path,
                    )
                    written.append(out_path)

                if "wald_p" in sub.columns:
                    tmp = sub.assign(
                        neglog10_p_meta=_safe_neg_log10_p(sub["p_meta"]),
                        neglog10_p_lmm_wald=_safe_neg_log10_p(sub["wald_p"]),
                    )
                    out_path = os.path.join(output_dir, f"{prefix}_gene_compare_p_wald__{focal_var}.png")
                    _write_scatter(
                        tmp,
                        x_col="neglog10_p_meta",
                        y_col="neglog10_p_lmm_wald",
                        title=f"-log10(p) comparison: meta vs LMM Wald ({focal_var})",
                        x_label="-log10(p) meta",
                        y_label="-log10(p) lmm (wald)",
                        out_path=out_path,
                    )
                    written.append(out_path)

                # Agreement / disagreement summaries (FDR thresholded; explicit q)
                if ("p_adj" in sub.columns) and ("lrt_p_adj" in sub.columns):
                    meta_sig = sub["p_adj"].astype(float) <= q
                    lrt_sig = sub["lrt_p_adj"].astype(float) <= q
                    counts = np.array(
                        [
                            [int((~meta_sig & ~lrt_sig).sum()), int((~meta_sig & lrt_sig).sum())],
                            [int((meta_sig & ~lrt_sig).sum()), int((meta_sig & lrt_sig).sum())],
                        ],
                        dtype=float,
                    )
                    out_path = os.path.join(
                        output_dir,
                        f"{prefix}_gene_agreement_confusion_meta_vs_lmm_lrt__{focal_var}__{q_safe}.png",
                    )
                    _write_confusion_matrix(
                        counts,
                        title=f"FDR agreement: meta vs LMM LRT ({focal_var}; q={q:g})",
                        x_label="LMM LRT significant",
                        y_label="Meta significant",
                        out_path=out_path,
                        x_ticklabels=("no", "yes"),
                        y_ticklabels=("no", "yes"),
                    )
                    written.append(out_path)

                    out_path = os.path.join(
                        output_dir,
                        f"{prefix}_gene_agreement_theta_sig_meta_vs_lmm_lrt__{focal_var}__{q_safe}.png",
                    )
                    _write_scatter_categories(
                        sub,
                        x_col="theta_meta",
                        y_col="theta_lmm",
                        cat_x=meta_sig,
                        cat_y=lrt_sig,
                        title=f"Theta: meta vs LMM (LRT sig; {focal_var}; q={q:g})",
                        x_label="theta (meta)",
                        y_label="theta (lmm)",
                        out_path=out_path,
                    )
                    written.append(out_path)

                if ("p_adj" in sub.columns) and ("wald_p_adj" in sub.columns):
                    meta_sig = sub["p_adj"].astype(float) <= q
                    wald_sig = sub["wald_p_adj"].astype(float) <= q
                    counts = np.array(
                        [
                            [int((~meta_sig & ~wald_sig).sum()), int((~meta_sig & wald_sig).sum())],
                            [int((meta_sig & ~wald_sig).sum()), int((meta_sig & wald_sig).sum())],
                        ],
                        dtype=float,
                    )
                    out_path = os.path.join(
                        output_dir,
                        f"{prefix}_gene_agreement_confusion_meta_vs_lmm_wald__{focal_var}__{q_safe}.png",
                    )
                    _write_confusion_matrix(
                        counts,
                        title=f"FDR agreement: meta vs LMM Wald ({focal_var}; q={q:g})",
                        x_label="LMM Wald significant",
                        y_label="Meta significant",
                        out_path=out_path,
                        x_ticklabels=("no", "yes"),
                        y_ticklabels=("no", "yes"),
                    )
                    written.append(out_path)

                    out_path = os.path.join(
                        output_dir,
                        f"{prefix}_gene_agreement_theta_sig_meta_vs_lmm_wald__{focal_var}__{q_safe}.png",
                    )
                    _write_scatter_categories(
                        sub,
                        x_col="theta_meta",
                        y_col="theta_lmm",
                        cat_x=meta_sig,
                        cat_y=wald_sig,
                        title=f"Theta: meta vs LMM (Wald sig; {focal_var}; q={q:g})",
                        x_label="theta (meta)",
                        y_label="theta (lmm)",
                        out_path=out_path,
                    )
                    written.append(out_path)

                if ("lrt_p_adj" in sub.columns) and ("wald_p_adj" in sub.columns):
                    lrt_sig = sub["lrt_p_adj"].astype(float) <= q
                    wald_sig = sub["wald_p_adj"].astype(float) <= q
                    counts = np.array(
                        [
                            [int((~lrt_sig & ~wald_sig).sum()), int((~lrt_sig & wald_sig).sum())],
                            [int((lrt_sig & ~wald_sig).sum()), int((lrt_sig & wald_sig).sum())],
                        ],
                        dtype=float,
                    )
                    out_path = os.path.join(
                        output_dir,
                        f"{prefix}_gene_agreement_confusion_lrt_vs_wald__{focal_var}__{q_safe}.png",
                    )
                    _write_confusion_matrix(
                        counts,
                        title=f"FDR agreement: LRT vs Wald ({focal_var}; q={q:g})",
                        x_label="Wald significant",
                        y_label="LRT significant",
                        out_path=out_path,
                        x_ticklabels=("no", "yes"),
                        y_ticklabels=("no", "yes"),
                    )
                    written.append(out_path)

                # Effect disagreements
                sub = sub.assign(theta_diff=(sub["theta_lmm"].astype(float) - sub["theta_meta"].astype(float)))
                out_path = os.path.join(output_dir, f"{prefix}_gene_agreement_theta_diff_hist__{focal_var}.png")
                _write_histogram(
                    sub,
                    x_col="theta_diff",
                    title=f"Theta difference (LMM - meta) ({focal_var})",
                    x_label="theta_lmm - theta_meta",
                    out_path=out_path,
                    bins=60,
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
