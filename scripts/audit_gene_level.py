from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from guide_pmd.gene_level import compute_gene_meta
from guide_pmd.gene_level_lmm import compute_gene_lmm


@dataclass(frozen=True)
class AuditConfig:
    n_genes: int
    guides_per_gene: int
    n_samples: int
    frac_signal: float
    effect_size: float
    intercept_sd: float
    noise_sd: float
    slope_sd: float
    seed: int
    alpha: float


def _balanced_treatment(n_samples: int) -> np.ndarray:
    n0 = n_samples // 2
    n1 = n_samples - n0
    return np.array([0.0] * n0 + [1.0] * n1, dtype=float)


def simulate_gene_level_dataset(cfg: AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    sample_ids = [f"s{i}" for i in range(cfg.n_samples)]
    treatment = _balanced_treatment(cfg.n_samples)
    model_matrix = pd.DataFrame({"treatment": treatment}, index=sample_ids)

    gene_ids = [f"gene_{i:04d}" for i in range(cfg.n_genes)]
    is_signal = rng.random(cfg.n_genes) < float(cfg.frac_signal)
    gene_truth = pd.DataFrame({"gene_id": gene_ids, "is_signal": is_signal.astype(bool)})

    guides: list[str] = []
    gene_for_guide: list[str] = []
    y_rows: list[np.ndarray] = []

    for gene_id, signal in zip(gene_ids, is_signal):
        theta = float(cfg.effect_size) if bool(signal) else 0.0
        for j in range(cfg.guides_per_gene):
            guide_id = f"{gene_id}_g{j+1}"
            intercept = rng.normal(0.0, float(cfg.intercept_sd))
            slope_dev = rng.normal(0.0, float(cfg.slope_sd)) if float(cfg.slope_sd) > 0 else 0.0
            noise = rng.normal(0.0, float(cfg.noise_sd), size=cfg.n_samples)
            y = intercept + (theta + slope_dev) * treatment + noise
            guides.append(guide_id)
            gene_for_guide.append(gene_id)
            y_rows.append(y)

    response_matrix = pd.DataFrame(y_rows, index=guides, columns=sample_ids)
    annotation_table = pd.DataFrame({"gene": gene_for_guide}, index=guides)
    return response_matrix, annotation_table, model_matrix, gene_truth


def _summarize_p(p: pd.Series, *, alpha: float) -> dict[str, float]:
    p_arr = p.to_numpy(dtype=float)
    finite = np.isfinite(p_arr)
    p_arr = p_arr[finite]
    if p_arr.size == 0:
        return {"n": 0.0, "nan_frac": 1.0, "mean": np.nan, "prop_lt_alpha": np.nan, "prop_lt_0p01": np.nan}
    return {
        "n": float(p_arr.size),
        "nan_frac": float(1.0 - (finite.mean() if finite.size else 0.0)),
        "mean": float(np.mean(p_arr)),
        "prop_lt_alpha": float(np.mean(p_arr < float(alpha))),
        "prop_lt_0p01": float(np.mean(p_arr < 0.01)),
    }


def run_audit(cfg: AuditConfig, out_dir: str) -> dict[str, object]:
    response, ann, mm, truth = simulate_gene_level_dataset(cfg)

    meta_df = compute_gene_meta(
        response,
        ann,
        mm,
        focal_vars=["treatment"],
        gene_id_col=1,
        add_intercept=True,
    )
    lmm_df = compute_gene_lmm(
        response,
        ann,
        mm,
        focal_vars=["treatment"],
        gene_id_col=1,
        add_intercept=True,
        allow_random_slope=True,
        fallback_to_meta=False,
        max_iter=200,
    )

    joined_meta = truth.merge(meta_df, on="gene_id", how="left")
    joined_lmm = truth.merge(lmm_df, on="gene_id", how="left")

    meta_null = joined_meta.loc[~joined_meta["is_signal"], "p"]
    meta_sig = joined_meta.loc[joined_meta["is_signal"], "p"]
    lmm_null = joined_lmm.loc[~joined_lmm["is_signal"], "p_primary"]
    lmm_sig = joined_lmm.loc[joined_lmm["is_signal"], "p_primary"]

    both = meta_df.merge(
        lmm_df[["gene_id", "theta", "p_primary", "p_primary_source"]],
        on="gene_id",
        how="inner",
        suffixes=("_meta", "_lmm"),
    )
    finite = np.isfinite(both["theta_meta"].to_numpy(dtype=float)) & np.isfinite(both["theta_lmm"].to_numpy(dtype=float))
    if finite.any():
        theta_corr = float(np.corrcoef(both.loc[finite, "theta_meta"], both.loc[finite, "theta_lmm"])[0, 1])
    else:
        theta_corr = np.nan

    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "audit_gene_meta.tsv")
    lmm_path = os.path.join(out_dir, "audit_gene_lmm.tsv")
    meta_df.to_csv(meta_path, sep="\t", index=False)
    lmm_df.to_csv(lmm_path, sep="\t", index=False)

    return {
        "config": asdict(cfg),
        "outputs": {"meta_tsv": meta_path, "lmm_tsv": lmm_path},
        "meta": {"null": _summarize_p(meta_null, alpha=cfg.alpha), "signal": _summarize_p(meta_sig, alpha=cfg.alpha)},
        "lmm": {"null": _summarize_p(lmm_null, alpha=cfg.alpha), "signal": _summarize_p(lmm_sig, alpha=cfg.alpha)},
        "theta_corr_meta_vs_lmm": theta_corr,
        "lmm_p_primary_source_counts": lmm_df["p_primary_source"].value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Small, deterministic gene-level audit harness (synthetic data).")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for audit artifacts.")
    parser.add_argument("--n-genes", type=int, default=200, help="Number of genes to simulate.")
    parser.add_argument("--guides-per-gene", type=int, default=4, help="Guides per gene.")
    parser.add_argument("--n-samples", type=int, default=40, help="Number of samples.")
    parser.add_argument("--frac-signal", type=float, default=0.2, help="Fraction of signal genes.")
    parser.add_argument("--effect-size", type=float, default=1.0, help="Signal effect size (mean slope).")
    parser.add_argument("--intercept-sd", type=float, default=0.5, help="Guide intercept SD.")
    parser.add_argument("--noise-sd", type=float, default=0.5, help="Observation noise SD.")
    parser.add_argument("--slope-sd", type=float, default=0.0, help="Guide slope SD (heterogeneity).")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha used for summary power/FPR.")
    args = parser.parse_args()

    cfg = AuditConfig(
        n_genes=args.n_genes,
        guides_per_gene=args.guides_per_gene,
        n_samples=args.n_samples,
        frac_signal=args.frac_signal,
        effect_size=args.effect_size,
        intercept_sd=args.intercept_sd,
        noise_sd=args.noise_sd,
        slope_sd=args.slope_sd,
        seed=args.seed,
        alpha=args.alpha,
    )

    report = run_audit(cfg, args.out_dir)
    report_path = os.path.join(args.out_dir, "audit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(report_path)


if __name__ == "__main__":
    main()

