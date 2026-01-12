from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guide_pmd.gene_level import compute_gene_meta
from guide_pmd.gene_level_lmm import compute_gene_lmm
from guide_pmd.gene_level_qc import compute_gene_qc


@dataclass(frozen=True)
class CountDepthBenchmarkConfig:
    n_genes: int
    guides_per_gene: int
    n_control: int
    n_treatment: int
    # Mean counts per guide at depth_factor=1.0 (log-normal parameters).
    guide_lambda_log_mean: float
    guide_lambda_log_sd: float
    # Per-sample depth variation (log-normal parameters for a multiplicative factor).
    depth_log_mean: float
    depth_log_sd: float
    # Optional extra Poisson noise layer on depth factors.
    depth_poisson_scale: float
    # Treatment can be confounded with depth (multiplier applied to treatment samples).
    treatment_depth_multiplier: float
    # True treatment effects (gene-level), and guide-level slope heterogeneity.
    frac_signal: float
    effect_sd: float
    guide_slope_sd: float
    # Construct a PMD-like response from simulated counts.
    pseudocount: float
    response_mode: str
    # Whether to include log-depth as a nuisance covariate in the model matrix.
    include_depth_covariate: bool
    # LMM options (keep small for speed by default).
    allow_random_slope: bool
    min_guides_random_slope: int
    max_iter: int
    seed: int
    alpha: float
    fdr_q: float

    def validate(self) -> None:
        if self.n_genes <= 0:
            raise ValueError("n_genes must be > 0")
        if self.guides_per_gene <= 0:
            raise ValueError("guides_per_gene must be > 0")
        if self.n_control <= 0 or self.n_treatment <= 0:
            raise ValueError("n_control and n_treatment must be > 0")
        if self.depth_poisson_scale < 0:
            raise ValueError("depth_poisson_scale must be >= 0")
        if not (0.0 <= self.frac_signal <= 1.0):
            raise ValueError("frac_signal must be in [0, 1]")
        if self.pseudocount <= 0:
            raise ValueError("pseudocount must be > 0")
        if self.response_mode not in {"log_counts", "guide_zscore_log_counts"}:
            raise ValueError("response_mode must be one of: log_counts, guide_zscore_log_counts")
        if self.min_guides_random_slope < 2:
            raise ValueError("min_guides_random_slope must be >= 2")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.fdr_q <= 0 or self.fdr_q >= 1:
            raise ValueError("fdr_q must be in (0, 1)")


def _simulate_depth_factors(
    rng: np.random.Generator,
    n_samples: int,
    *,
    depth_log_mean: float,
    depth_log_sd: float,
    depth_poisson_scale: float,
) -> np.ndarray:
    depth = rng.lognormal(mean=float(depth_log_mean), sigma=float(depth_log_sd), size=int(n_samples)).astype(float)
    if float(depth_poisson_scale) > 0:
        depth_counts = rng.poisson(depth * float(depth_poisson_scale)).astype(float)
        depth = np.maximum(depth_counts, 1.0) / float(depth_poisson_scale)
    return depth


def simulate_counts_and_std_res(
    cfg: CountDepthBenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg.validate()
    rng = np.random.default_rng(int(cfg.seed))

    n_control = int(cfg.n_control)
    n_treat = int(cfg.n_treatment)
    n_samples = n_control + n_treat

    sample_ids = [f"ctrl_{i:03d}" for i in range(n_control)] + [f"treat_{i:03d}" for i in range(n_treat)]
    treatment = np.array([0.0] * n_control + [1.0] * n_treat, dtype=float)

    depth_factor = _simulate_depth_factors(
        rng,
        n_samples,
        depth_log_mean=cfg.depth_log_mean,
        depth_log_sd=cfg.depth_log_sd,
        depth_poisson_scale=cfg.depth_poisson_scale,
    )
    depth_factor = depth_factor * np.where(treatment > 0, float(cfg.treatment_depth_multiplier), 1.0)
    log_depth = np.log(depth_factor)

    gene_ids = [f"gene_{i:05d}" for i in range(int(cfg.n_genes))]
    is_signal = rng.random(int(cfg.n_genes)) < float(cfg.frac_signal)
    gene_theta = rng.normal(loc=0.0, scale=float(cfg.effect_sd), size=int(cfg.n_genes)).astype(float)
    gene_theta = np.where(is_signal, gene_theta, 0.0)

    truth_gene = pd.DataFrame({"gene_id": gene_ids, "is_signal": is_signal.astype(bool), "theta_true": gene_theta.astype(float)})

    n_guides = int(cfg.n_genes) * int(cfg.guides_per_gene)
    # Per-guide baseline means (independent of sample depth).
    guide_lambda = rng.lognormal(mean=float(cfg.guide_lambda_log_mean), sigma=float(cfg.guide_lambda_log_sd), size=n_guides).astype(float)

    guides: list[str] = []
    gene_for_guide: list[str] = []
    counts_rows: list[np.ndarray] = []
    slope_dev_rows: list[float] = []

    guide_idx = 0
    for gene_id, theta in zip(gene_ids, gene_theta):
        for j in range(int(cfg.guides_per_gene)):
            guide_id = f"{gene_id}__g{j+1:02d}"
            slope_dev = float(rng.normal(loc=0.0, scale=float(cfg.guide_slope_sd))) if float(cfg.guide_slope_sd) > 0 else 0.0
            mu = guide_lambda[guide_idx] * depth_factor * np.exp((float(theta) + slope_dev) * treatment)
            mu = np.clip(mu, a_min=0.0, a_max=None)
            counts = rng.poisson(mu).astype(int)

            guides.append(guide_id)
            gene_for_guide.append(gene_id)
            counts_rows.append(counts)
            slope_dev_rows.append(slope_dev)
            guide_idx += 1

    counts_df = pd.DataFrame(counts_rows, index=guides, columns=sample_ids)
    annotation_df = pd.DataFrame({"gene_symbol": gene_for_guide}, index=guides)

    log_counts = np.log(counts_df.to_numpy(dtype=float) + float(cfg.pseudocount))
    if cfg.response_mode == "log_counts":
        response = log_counts
    else:
        # Per-guide z-scored log counts (a fast PMD-like surrogate).
        guide_mean = np.mean(log_counts, axis=1, keepdims=True)
        guide_sd = np.std(log_counts, axis=1, ddof=1, keepdims=True)
        guide_sd = np.where(guide_sd <= 0, 1.0, guide_sd)
        response = (log_counts - guide_mean) / guide_sd
    std_res_df = pd.DataFrame(response, index=guides, columns=sample_ids)

    model_matrix = pd.DataFrame({"treatment": treatment}, index=sample_ids)
    if bool(cfg.include_depth_covariate):
        model_matrix["log_depth"] = log_depth

    truth_guide = pd.DataFrame(
        {
            "guide_id": guides,
            "gene_id": gene_for_guide,
            "lambda_base": guide_lambda[: len(guides)].astype(float),
            "slope_dev": np.asarray(slope_dev_rows, dtype=float),
        }
    )
    return (
        counts_df,
        annotation_df,
        std_res_df,
        model_matrix,
        truth_gene.merge(truth_guide.groupby("gene_id").size().rename("m_guides"), on="gene_id"),
    )


def _summarize_p(p: pd.Series, *, alpha: float) -> dict[str, float]:
    p_arr = p.to_numpy(dtype=float)
    finite = np.isfinite(p_arr)
    p_arr = p_arr[finite]
    if p_arr.size == 0:
        return {"n": 0.0, "nan_frac": 1.0, "mean": np.nan, "prop_lt_alpha": np.nan, "prop_lt_0p01": np.nan}
    return {
        "n": float(p_arr.size),
        "nan_frac": float(1.0 - float(np.mean(finite)) if finite.size else 1.0),
        "mean": float(np.mean(p_arr)),
        "prop_lt_alpha": float(np.mean(p_arr < float(alpha))),
        "prop_lt_0p01": float(np.mean(p_arr < 0.01)),
    }


def _fdr_summary(p_adj: pd.Series, is_signal: pd.Series, *, q: float) -> dict[str, float]:
    p_adj_arr = p_adj.to_numpy(dtype=float)
    sig = is_signal.to_numpy(dtype=bool)
    called = np.isfinite(p_adj_arr) & (p_adj_arr < float(q))
    n_called = int(np.sum(called))
    if n_called == 0:
        return {"q": float(q), "n_called": 0.0, "fdr": np.nan, "tpr": np.nan}
    tp = int(np.sum(called & sig))
    fp = int(np.sum(called & ~sig))
    return {
        "q": float(q),
        "n_called": float(n_called),
        "fdr": float(fp / n_called),
        "tpr": float(tp / max(1, int(np.sum(sig)))),
    }


def run_benchmark(cfg: CountDepthBenchmarkConfig, out_dir: str) -> dict[str, object]:
    cfg.validate()
    os.makedirs(out_dir, exist_ok=True)

    counts_df, ann_df, std_res_df, mm, truth_gene = simulate_counts_and_std_res(cfg)

    # Write inputs for reproducibility.
    counts_path = os.path.join(out_dir, "sim_counts.tsv")
    mm_path = os.path.join(out_dir, "sim_model_matrix.tsv")
    std_res_path = os.path.join(out_dir, "sim_std_res.tsv")
    truth_path = os.path.join(out_dir, "sim_truth_gene.tsv")

    counts_out = counts_df.copy()
    counts_out.insert(0, "gene_symbol", ann_df["gene_symbol"])
    counts_out.index.name = "guide_id"
    counts_out.to_csv(counts_path, sep="\t")
    mm.to_csv(mm_path, sep="\t", index_label="sample_id")
    std_res_df.to_csv(std_res_path, sep="\t")
    truth_gene.to_csv(truth_path, sep="\t", index=False)

    focal_vars = ["treatment"]

    t0 = time.perf_counter()
    meta_df = compute_gene_meta(
        std_res_df,
        ann_df,
        mm,
        focal_vars=focal_vars,
        gene_id_col=1,
        add_intercept=True,
    )
    t_meta = time.perf_counter() - t0
    meta_out_path = os.path.join(out_dir, "PMD_std_res_gene_meta.tsv")
    meta_df.to_csv(meta_out_path, sep="\t", index=False)

    t0 = time.perf_counter()
    lmm_df = compute_gene_lmm(
        std_res_df,
        ann_df,
        mm,
        focal_vars=focal_vars,
        gene_id_col=1,
        add_intercept=True,
        allow_random_slope=bool(cfg.allow_random_slope),
        min_guides_random_slope=int(cfg.min_guides_random_slope),
        max_iter=int(cfg.max_iter),
        fallback_to_meta=False,
    )
    t_lmm = time.perf_counter() - t0
    lmm_out_path = os.path.join(out_dir, "PMD_std_res_gene_lmm.tsv")
    lmm_df.to_csv(lmm_out_path, sep="\t", index=False)

    t0 = time.perf_counter()
    qc_df = compute_gene_qc(
        std_res_df,
        ann_df,
        mm,
        focal_vars=focal_vars,
        gene_id_col=1,
        add_intercept=True,
        residual_matrix=None,
    )
    t_qc = time.perf_counter() - t0
    qc_out_path = os.path.join(out_dir, "PMD_std_res_gene_qc.tsv")
    qc_df.to_csv(qc_out_path, sep="\t", index=False)

    # Evaluate against truth at the gene level.
    meta_join = truth_gene.merge(meta_df, on="gene_id", how="left")
    lmm_join = truth_gene.merge(lmm_df, on="gene_id", how="left")

    meta_null = meta_join.loc[~meta_join["is_signal"], "p"]
    meta_sig = meta_join.loc[meta_join["is_signal"], "p"]

    lrt_p = lmm_join["lrt_p"] if "lrt_p" in lmm_join.columns else pd.Series(dtype=float)
    wald_p = lmm_join["wald_p"] if "wald_p" in lmm_join.columns else pd.Series(dtype=float)

    lrt_null = lrt_p.loc[~lmm_join["is_signal"]] if not lmm_join.empty else pd.Series(dtype=float)
    lrt_sig = lrt_p.loc[lmm_join["is_signal"]] if not lmm_join.empty else pd.Series(dtype=float)
    wald_null = wald_p.loc[~lmm_join["is_signal"]] if not lmm_join.empty else pd.Series(dtype=float)
    wald_sig = wald_p.loc[lmm_join["is_signal"]] if not lmm_join.empty else pd.Series(dtype=float)

    report = {
        "config": asdict(cfg),
        "outputs": {
            "counts_tsv": counts_path,
            "model_matrix_tsv": mm_path,
            "std_res_tsv": std_res_path,
            "truth_gene_tsv": truth_path,
            "gene_meta_tsv": meta_out_path,
            "gene_lmm_tsv": lmm_out_path,
            "gene_qc_tsv": qc_out_path,
        },
        "runtime_sec": {"meta": float(t_meta), "lmm": float(t_lmm), "qc": float(t_qc)},
        "meta": {
            "null": _summarize_p(meta_null, alpha=cfg.alpha),
            "signal": _summarize_p(meta_sig, alpha=cfg.alpha),
            "fdr": _fdr_summary(meta_join["p_adj"], meta_join["is_signal"], q=cfg.fdr_q),
        },
        "lmm_lrt": {
            "null": _summarize_p(lrt_null, alpha=cfg.alpha),
            "signal": _summarize_p(lrt_sig, alpha=cfg.alpha),
            "fdr": _fdr_summary(lmm_join["lrt_p_adj"], lmm_join["is_signal"], q=cfg.fdr_q) if "lrt_p_adj" in lmm_join.columns else {},
            "lrt_ok_frac": float(np.mean(lmm_join["lrt_ok"].fillna(False).astype(bool))) if "lrt_ok" in lmm_join.columns else np.nan,
        },
        "lmm_wald": {
            "null": _summarize_p(wald_null, alpha=cfg.alpha),
            "signal": _summarize_p(wald_sig, alpha=cfg.alpha),
            "fdr": _fdr_summary(lmm_join["wald_p_adj"], lmm_join["is_signal"], q=cfg.fdr_q) if "wald_p_adj" in lmm_join.columns else {},
            "wald_ok_frac": float(np.mean(lmm_join["wald_ok"].fillna(False).astype(bool))) if "wald_ok" in lmm_join.columns else np.nan,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Count-depth benchmark: Poisson counts + depth confounding → gene-level methods.")
    parser.add_argument("--out-dir", required=True, type=str, help="Output directory for benchmark artifacts.")
    parser.add_argument("--n-genes", type=int, default=500, help="Number of genes.")
    parser.add_argument("--guides-per-gene", type=int, default=4, help="Guides per gene.")
    parser.add_argument("--n-control", type=int, default=12, help="Number of control samples.")
    parser.add_argument("--n-treatment", type=int, default=12, help="Number of treatment samples.")
    parser.add_argument("--guide-lambda-log-mean", type=float, default=np.log(200.0), help="log-mean baseline guide lambda.")
    parser.add_argument("--guide-lambda-log-sd", type=float, default=0.8, help="log-sd baseline guide lambda.")
    parser.add_argument("--depth-log-mean", type=float, default=0.0, help="log-mean of depth factor.")
    parser.add_argument("--depth-log-sd", type=float, default=1.0, help="log-sd of depth factor (order-of-magnitude variation ~ 1).")
    parser.add_argument("--depth-poisson-scale", type=float, default=0.0, help="Optional Poisson noise on depth factors (0 disables).")
    parser.add_argument("--treatment-depth-multiplier", type=float, default=1.0, help="Depth multiplier applied to treatment samples.")
    parser.add_argument("--frac-signal", type=float, default=0.2, help="Fraction of truly non-null genes.")
    parser.add_argument("--effect-sd", type=float, default=0.5, help="SD of gene-level treatment effects.")
    parser.add_argument("--guide-slope-sd", type=float, default=0.2, help="SD of guide-level slope deviations.")
    parser.add_argument("--pseudocount", type=float, default=0.5, help="Pseudocount used in log transform.")
    parser.add_argument(
        "--response-mode",
        type=str,
        choices=["log_counts", "guide_zscore_log_counts"],
        default="log_counts",
        help="How to construct the response matrix from simulated counts (default: log_counts).",
    )
    parser.add_argument("--include-depth-covariate", action="store_true", help="Include log_depth in model matrix as a nuisance covariate.")
    parser.add_argument("--allow-random-slope", action=argparse.BooleanOptionalAction, default=True, help="Allow random slope in LMM (default: True).")
    parser.add_argument("--min-guides-random-slope", type=int, default=3, help="Minimum guides for RI+RS (default: 3).")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations per MixedLM fit (default: 200).")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha for summary metrics.")
    parser.add_argument("--fdr-q", type=float, default=0.1, help="FDR threshold for summary metrics.")
    args = parser.parse_args()

    cfg = CountDepthBenchmarkConfig(
        n_genes=args.n_genes,
        guides_per_gene=args.guides_per_gene,
        n_control=args.n_control,
        n_treatment=args.n_treatment,
        guide_lambda_log_mean=args.guide_lambda_log_mean,
        guide_lambda_log_sd=args.guide_lambda_log_sd,
        depth_log_mean=args.depth_log_mean,
        depth_log_sd=args.depth_log_sd,
        depth_poisson_scale=args.depth_poisson_scale,
        treatment_depth_multiplier=args.treatment_depth_multiplier,
        frac_signal=args.frac_signal,
        effect_sd=args.effect_sd,
        guide_slope_sd=args.guide_slope_sd,
        pseudocount=args.pseudocount,
        response_mode=str(args.response_mode),
        include_depth_covariate=bool(args.include_depth_covariate),
        allow_random_slope=bool(args.allow_random_slope),
        min_guides_random_slope=args.min_guides_random_slope,
        max_iter=args.max_iter,
        seed=args.seed,
        alpha=args.alpha,
        fdr_q=args.fdr_q,
    )

    report = run_benchmark(cfg, args.out_dir)
    report_path = os.path.join(args.out_dir, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(report_path)


if __name__ == "__main__":
    main()
