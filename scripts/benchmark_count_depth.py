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
    # Additional gene-level baseline heterogeneity on log(lambda).
    gene_lambda_log_sd: float
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
    # Guide-level on-target slope deviations (applied only for signal genes).
    guide_slope_sd: float
    # Optional "bad guide" off-target contamination (applied regardless of is_signal).
    offtarget_guide_frac: float
    offtarget_slope_sd: float
    # Construct a PMD-like response from simulated counts.
    pseudocount: float
    response_mode: str
    pmd_n_boot: int
    pmd_seed: int
    # Whether to include log-depth as a nuisance covariate in the model matrix.
    include_depth_covariate: bool
    # LMM options (keep small for speed by default).
    allow_random_slope: bool
    min_guides_random_slope: int
    max_iter: int
    methods: tuple[str, ...]
    seed: int
    alpha: float
    fdr_q: float
    qq_plots: bool

    def validate(self) -> None:
        if self.n_genes <= 0:
            raise ValueError("n_genes must be > 0")
        if self.guides_per_gene <= 0:
            raise ValueError("guides_per_gene must be > 0")
        if self.n_control <= 0 or self.n_treatment <= 0:
            raise ValueError("n_control and n_treatment must be > 0")
        if float(self.guide_lambda_log_sd) < 0:
            raise ValueError("guide_lambda_log_sd must be >= 0")
        if float(self.gene_lambda_log_sd) < 0:
            raise ValueError("gene_lambda_log_sd must be >= 0")
        if self.depth_poisson_scale < 0:
            raise ValueError("depth_poisson_scale must be >= 0")
        if not (0.0 <= self.frac_signal <= 1.0):
            raise ValueError("frac_signal must be in [0, 1]")
        if float(self.guide_slope_sd) < 0:
            raise ValueError("guide_slope_sd must be >= 0")
        if not (0.0 <= float(self.offtarget_guide_frac) <= 1.0):
            raise ValueError("offtarget_guide_frac must be in [0, 1]")
        if float(self.offtarget_slope_sd) < 0:
            raise ValueError("offtarget_slope_sd must be >= 0")
        if self.pseudocount <= 0:
            raise ValueError("pseudocount must be > 0")
        if self.response_mode not in {"log_counts", "guide_zscore_log_counts", "pmd_std_res"}:
            raise ValueError("response_mode must be one of: log_counts, guide_zscore_log_counts, pmd_std_res")
        if self.response_mode == "pmd_std_res":
            if int(self.pmd_n_boot) < 2:
                raise ValueError("pmd_n_boot must be >= 2 (required for valid PMD z-scores)")
            if int(self.pmd_seed) < 0:
                raise ValueError("pmd_seed must be >= 0")
        if self.min_guides_random_slope < 2:
            raise ValueError("min_guides_random_slope must be >= 2")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if not self.methods:
            raise ValueError("methods must not be empty")
        allowed_methods = {"meta", "lmm", "qc"}
        unknown = set(self.methods).difference(allowed_methods)
        if unknown:
            raise ValueError(f"unknown methods: {sorted(unknown)}")
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


def _compute_pmd_std_res(counts_df: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    from percent_max_diff.percent_max_diff import pmd

    # percent_max_diff uses numpy's global RNG; seed deterministically for reproducibility.
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        pmd_res = pmd(counts_df, num_boot=int(n_boot), skip_posthoc=True)
    finally:
        np.random.set_state(state)

    std_res = pmd_res.z_scores
    std_res = std_res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return std_res


def _confusion(called: np.ndarray, is_signal: np.ndarray) -> dict[str, float | int | None]:
    called = np.asarray(called, dtype=bool)
    is_signal = np.asarray(is_signal, dtype=bool)
    if called.shape != is_signal.shape:
        raise ValueError("called and is_signal must have the same shape")

    n_total = int(is_signal.size)
    n_signal = int(np.sum(is_signal))
    n_null = int(n_total - n_signal)

    tp = int(np.sum(called & is_signal))
    fp = int(np.sum(called & ~is_signal))
    tn = int(np.sum(~called & ~is_signal))
    fn = int(np.sum(~called & is_signal))

    n_called = int(tp + fp)

    tpr = (tp / n_signal) if n_signal else None
    fpr = (fp / n_null) if n_null else None
    fdr = (fp / n_called) if n_called else None
    ppv = (tp / n_called) if n_called else None

    return {
        "n_total": n_total,
        "n_signal": n_signal,
        "n_null": n_null,
        "n_called": n_called,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "fdr": fdr,
        "ppv": ppv,
    }


def simulate_counts_and_std_res(
    cfg: CountDepthBenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    gene_log_lambda = rng.normal(
        loc=float(cfg.guide_lambda_log_mean),
        scale=float(cfg.gene_lambda_log_sd),
        size=int(cfg.n_genes),
    ).astype(float)

    truth_gene = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "is_signal": is_signal.astype(bool),
            "theta_true": gene_theta.astype(float),
            "log_lambda_gene": gene_log_lambda.astype(float),
        }
    )

    guides: list[str] = []
    gene_for_guide: list[str] = []
    counts_rows: list[np.ndarray] = []
    slope_dev_rows: list[float] = []
    offtarget_dev_rows: list[float] = []
    is_offtarget_rows: list[bool] = []
    theta_guide_rows: list[float] = []
    log_lambda_guide_rows: list[float] = []

    for gene_i, (gene_id, theta) in enumerate(zip(gene_ids, gene_theta)):
        gene_is_signal = bool(is_signal[gene_i])
        gene_ll = float(gene_log_lambda[gene_i])
        for j in range(int(cfg.guides_per_gene)):
            guide_id = f"{gene_id}__g{j+1:02d}"
            log_lambda_guide = gene_ll + float(rng.normal(loc=0.0, scale=float(cfg.guide_lambda_log_sd)))
            lambda_base = float(np.exp(log_lambda_guide))
            slope_dev = (
                float(rng.normal(loc=0.0, scale=float(cfg.guide_slope_sd)))
                if gene_is_signal and float(cfg.guide_slope_sd) > 0
                else 0.0
            )
            is_offtarget = bool(rng.random() < float(cfg.offtarget_guide_frac)) if float(cfg.offtarget_guide_frac) > 0 else False
            offtarget_dev = (
                float(rng.normal(loc=0.0, scale=float(cfg.offtarget_slope_sd)))
                if is_offtarget and float(cfg.offtarget_slope_sd) > 0
                else 0.0
            )
            theta_guide = float(theta) + slope_dev + offtarget_dev

            mu = lambda_base * depth_factor * np.exp(theta_guide * treatment)
            mu = np.clip(mu, a_min=0.0, a_max=None)
            counts = rng.poisson(mu).astype(int)

            guides.append(guide_id)
            gene_for_guide.append(gene_id)
            counts_rows.append(counts)
            slope_dev_rows.append(slope_dev)
            offtarget_dev_rows.append(offtarget_dev)
            is_offtarget_rows.append(is_offtarget)
            theta_guide_rows.append(theta_guide)
            log_lambda_guide_rows.append(log_lambda_guide)

    counts_df = pd.DataFrame(counts_rows, index=guides, columns=sample_ids)
    annotation_df = pd.DataFrame({"gene_symbol": gene_for_guide}, index=guides)

    if cfg.response_mode == "pmd_std_res":
        std_res_df = _compute_pmd_std_res(counts_df, n_boot=int(cfg.pmd_n_boot), seed=int(cfg.pmd_seed))
    else:
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
            "log_lambda_gene": np.repeat(gene_log_lambda, int(cfg.guides_per_gene)).astype(float),
            "log_lambda_guide": np.asarray(log_lambda_guide_rows, dtype=float),
            "lambda_base": np.exp(np.asarray(log_lambda_guide_rows, dtype=float)).astype(float),
            "theta_gene": np.repeat(gene_theta, int(cfg.guides_per_gene)).astype(float),
            "slope_dev": np.asarray(slope_dev_rows, dtype=float),
            "offtarget_dev": np.asarray(offtarget_dev_rows, dtype=float),
            "theta_guide": np.asarray(theta_guide_rows, dtype=float),
            "is_offtarget": np.asarray(is_offtarget_rows, dtype=bool),
        }
    )
    return (
        counts_df,
        annotation_df,
        std_res_df,
        model_matrix,
        truth_gene.merge(truth_guide.groupby("gene_id").size().rename("m_guides"), on="gene_id"),
        truth_guide,
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


def _write_qq_plot(p: pd.Series, *, out_path: str, title: str) -> dict[str, float | int | None]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from scipy.stats import chi2

    p_arr = p.to_numpy(dtype=float)
    mask = np.isfinite(p_arr) & (p_arr > 0.0) & (p_arr <= 1.0)
    p_arr = p_arr[mask]
    n = int(p_arr.size)
    if n == 0:
        return {"n": 0, "lambda_gc": None}

    p_sorted = np.sort(np.clip(p_arr, np.finfo(float).tiny, 1.0))
    expected = (np.arange(1, n + 1, dtype=float) - 0.5) / float(n)

    x = -np.log10(expected)
    y = -np.log10(p_sorted)

    max_v = float(max(np.max(x), np.max(y))) if n else 1.0

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.scatter(x, y, s=8, alpha=0.6, edgecolors="none")
    ax.plot([0.0, max_v], [0.0, max_v], color="black", lw=1)
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    chi = chi2.isf(p_arr, 1)
    lambda_gc = float(np.median(chi) / float(chi2.ppf(0.5, 1)))
    return {"n": n, "lambda_gc": lambda_gc}


def _qq_stats(p: pd.Series) -> dict[str, float | int | None]:
    from scipy.stats import chi2

    p_arr = p.to_numpy(dtype=float)
    mask = np.isfinite(p_arr) & (p_arr > 0.0) & (p_arr <= 1.0)
    p_arr = p_arr[mask]
    n = int(p_arr.size)
    if n == 0:
        return {"n": 0, "lambda_gc": None}
    chi = chi2.isf(p_arr, 1)
    lambda_gc = float(np.median(chi) / float(chi2.ppf(0.5, 1)))
    return {"n": n, "lambda_gc": lambda_gc}


def _json_sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    return obj


def run_benchmark(cfg: CountDepthBenchmarkConfig, out_dir: str) -> dict[str, object]:
    cfg.validate()
    os.makedirs(out_dir, exist_ok=True)

    counts_df, ann_df, std_res_df, mm, truth_gene, truth_guide = simulate_counts_and_std_res(cfg)

    # Write inputs for reproducibility.
    counts_path = os.path.join(out_dir, "sim_counts.tsv")
    mm_path = os.path.join(out_dir, "sim_model_matrix.tsv")
    std_res_path = os.path.join(out_dir, "sim_std_res.tsv")
    truth_path = os.path.join(out_dir, "sim_truth_gene.tsv")
    truth_guide_path = os.path.join(out_dir, "sim_truth_guide.tsv")

    counts_out = counts_df.copy()
    counts_out.insert(0, "gene_symbol", ann_df["gene_symbol"])
    counts_out.index.name = "guide_id"
    counts_out.to_csv(counts_path, sep="\t")
    mm.to_csv(mm_path, sep="\t", index_label="sample_id")
    std_res_df.to_csv(std_res_path, sep="\t")
    truth_gene.to_csv(truth_path, sep="\t", index=False)
    truth_guide.to_csv(truth_guide_path, sep="\t", index=False)

    focal_vars = ["treatment"]

    runtime: dict[str, float] = {}
    meta_df = pd.DataFrame()
    lmm_df = pd.DataFrame()
    qc_df = pd.DataFrame()

    meta_out_path = os.path.join(out_dir, "PMD_std_res_gene_meta.tsv")
    if "meta" in cfg.methods:
        t0 = time.perf_counter()
        meta_df = compute_gene_meta(
            std_res_df,
            ann_df,
            mm,
            focal_vars=focal_vars,
            gene_id_col=1,
            add_intercept=True,
        )
        runtime["meta"] = float(time.perf_counter() - t0)
        meta_df.to_csv(meta_out_path, sep="\t", index=False)

    lmm_out_path = os.path.join(out_dir, "PMD_std_res_gene_lmm.tsv")
    if "lmm" in cfg.methods:
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
        runtime["lmm"] = float(time.perf_counter() - t0)
        lmm_df.to_csv(lmm_out_path, sep="\t", index=False)

    qc_out_path = os.path.join(out_dir, "PMD_std_res_gene_qc.tsv")
    if "qc" in cfg.methods:
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
        runtime["qc"] = float(time.perf_counter() - t0)
        qc_df.to_csv(qc_out_path, sep="\t", index=False)

    # Evaluate against truth at the gene level.
    meta_join = truth_gene.merge(meta_df, on="gene_id", how="left") if not meta_df.empty else truth_gene.copy()
    lmm_join = truth_gene.merge(lmm_df, on="gene_id", how="left") if not lmm_df.empty else truth_gene.copy()

    meta_null = meta_join.loc[~meta_join["is_signal"], "p"]
    meta_sig = meta_join.loc[meta_join["is_signal"], "p"]

    lrt_p = lmm_join["lrt_p"] if "lrt_p" in lmm_join.columns else pd.Series(dtype=float)
    wald_p = lmm_join["wald_p"] if "wald_p" in lmm_join.columns else pd.Series(dtype=float)

    lrt_null = lrt_p.loc[~lmm_join["is_signal"]] if not lmm_join.empty else pd.Series(dtype=float)
    lrt_sig = lrt_p.loc[lmm_join["is_signal"]] if not lmm_join.empty else pd.Series(dtype=float)
    wald_null = wald_p.loc[~lmm_join["is_signal"]] if not lmm_join.empty else pd.Series(dtype=float)
    wald_sig = wald_p.loc[lmm_join["is_signal"]] if not lmm_join.empty else pd.Series(dtype=float)

    report: dict[str, object] = {
        "config": asdict(cfg),
        "outputs": {
            "counts_tsv": counts_path,
            "model_matrix_tsv": mm_path,
            "std_res_tsv": std_res_path,
            "truth_gene_tsv": truth_path,
            "truth_guide_tsv": truth_guide_path,
            "gene_meta_tsv": meta_out_path if "meta" in cfg.methods else "",
            "gene_lmm_tsv": lmm_out_path if "lmm" in cfg.methods else "",
            "gene_qc_tsv": qc_out_path if "qc" in cfg.methods else "",
        },
        "runtime_sec": runtime,
    }

    if "meta" in cfg.methods and not meta_df.empty:
        meta_called_alpha = np.isfinite(meta_join["p"].to_numpy(dtype=float)) & (meta_join["p"].to_numpy(dtype=float) < float(cfg.alpha))
        meta_called_q = np.isfinite(meta_join["p_adj"].to_numpy(dtype=float)) & (meta_join["p_adj"].to_numpy(dtype=float) < float(cfg.fdr_q))
        report["meta"] = {
            "null": _summarize_p(meta_null, alpha=cfg.alpha),
            "signal": _summarize_p(meta_sig, alpha=cfg.alpha),
            "fdr": _fdr_summary(meta_join["p_adj"], meta_join["is_signal"], q=cfg.fdr_q),
            "confusion_alpha": _confusion(meta_called_alpha, meta_join["is_signal"].to_numpy(dtype=bool)),
            "confusion_fdr_q": _confusion(meta_called_q, meta_join["is_signal"].to_numpy(dtype=bool)),
        }
    if "lmm" in cfg.methods and not lmm_df.empty:
        lrt_p_arr = lrt_p.to_numpy(dtype=float) if not lrt_p.empty else np.asarray([], dtype=float)
        wald_p_arr = wald_p.to_numpy(dtype=float) if not wald_p.empty else np.asarray([], dtype=float)
        lrt_p_adj_arr = lmm_join["lrt_p_adj"].to_numpy(dtype=float) if "lrt_p_adj" in lmm_join.columns else np.asarray([], dtype=float)
        wald_p_adj_arr = lmm_join["wald_p_adj"].to_numpy(dtype=float) if "wald_p_adj" in lmm_join.columns else np.asarray([], dtype=float)
        is_signal_arr = lmm_join["is_signal"].to_numpy(dtype=bool)

        lrt_called_alpha = np.isfinite(lrt_p_arr) & (lrt_p_arr < float(cfg.alpha))
        wald_called_alpha = np.isfinite(wald_p_arr) & (wald_p_arr < float(cfg.alpha))
        lrt_called_q = np.isfinite(lrt_p_adj_arr) & (lrt_p_adj_arr < float(cfg.fdr_q)) if lrt_p_adj_arr.size else np.zeros_like(is_signal_arr, dtype=bool)
        wald_called_q = (
            np.isfinite(wald_p_adj_arr) & (wald_p_adj_arr < float(cfg.fdr_q)) if wald_p_adj_arr.size else np.zeros_like(is_signal_arr, dtype=bool)
        )

        report["lmm_lrt"] = {
            "null": _summarize_p(lrt_null, alpha=cfg.alpha),
            "signal": _summarize_p(lrt_sig, alpha=cfg.alpha),
            "fdr": _fdr_summary(lmm_join["lrt_p_adj"], lmm_join["is_signal"], q=cfg.fdr_q) if "lrt_p_adj" in lmm_join.columns else {},
            "lrt_ok_frac": float(np.mean(lmm_join["lrt_ok"].fillna(False).astype(bool))) if "lrt_ok" in lmm_join.columns else np.nan,
            "confusion_alpha": _confusion(lrt_called_alpha, is_signal_arr),
            "confusion_fdr_q": _confusion(lrt_called_q, is_signal_arr),
        }
        report["lmm_wald"] = {
            "null": _summarize_p(wald_null, alpha=cfg.alpha),
            "signal": _summarize_p(wald_sig, alpha=cfg.alpha),
            "fdr": _fdr_summary(lmm_join["wald_p_adj"], lmm_join["is_signal"], q=cfg.fdr_q) if "wald_p_adj" in lmm_join.columns else {},
            "wald_ok_frac": float(np.mean(lmm_join["wald_ok"].fillna(False).astype(bool))) if "wald_ok" in lmm_join.columns else np.nan,
            "confusion_alpha": _confusion(wald_called_alpha, is_signal_arr),
            "confusion_fdr_q": _confusion(wald_called_q, is_signal_arr),
        }
    qq: dict[str, object] = {}
    wrote_any_plot = False
    fig_dir = os.path.join(out_dir, "figures")

    if "meta" in cfg.methods and not meta_df.empty:
        if bool(cfg.qq_plots):
            os.makedirs(fig_dir, exist_ok=True)
            out_path = os.path.join(fig_dir, "qq_meta_p_null.png")
            report["outputs"]["meta_p_null_png"] = out_path
            qq["meta_p_null"] = _write_qq_plot(meta_null, out_path=out_path, title="Meta p (null)")
            wrote_any_plot = True
        else:
            qq["meta_p_null"] = _qq_stats(meta_null)

    if "lmm" in cfg.methods and not lmm_df.empty and "lrt_p" in lmm_join.columns:
        if bool(cfg.qq_plots):
            os.makedirs(fig_dir, exist_ok=True)
            out_path = os.path.join(fig_dir, "qq_lmm_lrt_p_null.png")
            report["outputs"]["lmm_lrt_p_null_png"] = out_path
            qq["lmm_lrt_p_null"] = _write_qq_plot(lrt_null, out_path=out_path, title="LMM LRT p (null)")
            wrote_any_plot = True
        else:
            qq["lmm_lrt_p_null"] = _qq_stats(lrt_null)

    if "lmm" in cfg.methods and not lmm_df.empty and "wald_p" in lmm_join.columns:
        if bool(cfg.qq_plots):
            os.makedirs(fig_dir, exist_ok=True)
            out_path = os.path.join(fig_dir, "qq_lmm_wald_p_null.png")
            report["outputs"]["lmm_wald_p_null_png"] = out_path
            qq["lmm_wald_p_null"] = _write_qq_plot(wald_null, out_path=out_path, title="LMM Wald p (null)")
            wrote_any_plot = True
        else:
            qq["lmm_wald_p_null"] = _qq_stats(wald_null)

    if qq:
        report["qq"] = qq
    if wrote_any_plot:
        report["outputs"]["figures_dir"] = fig_dir

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
    parser.add_argument("--gene-lambda-log-sd", type=float, default=0.5, help="Additional gene-level log-sd on lambda (default: 0.5).")
    parser.add_argument("--depth-log-mean", type=float, default=0.0, help="log-mean of depth factor.")
    parser.add_argument("--depth-log-sd", type=float, default=1.0, help="log-sd of depth factor (order-of-magnitude variation ~ 1).")
    parser.add_argument("--depth-poisson-scale", type=float, default=0.0, help="Optional Poisson noise on depth factors (0 disables).")
    parser.add_argument("--treatment-depth-multiplier", type=float, default=1.0, help="Depth multiplier applied to treatment samples.")
    parser.add_argument("--frac-signal", type=float, default=0.2, help="Fraction of truly non-null genes.")
    parser.add_argument("--effect-sd", type=float, default=0.5, help="SD of gene-level treatment effects.")
    parser.add_argument("--guide-slope-sd", type=float, default=0.2, help="SD of guide-level slope deviations.")
    parser.add_argument("--offtarget-guide-frac", type=float, default=0.0, help="Fraction of guides with off-target effects (default: 0).")
    parser.add_argument("--offtarget-slope-sd", type=float, default=0.0, help="SD of off-target slope deviations (default: 0).")
    parser.add_argument("--pseudocount", type=float, default=0.5, help="Pseudocount used in log transform.")
    parser.add_argument(
        "--response-mode",
        type=str,
        choices=["log_counts", "guide_zscore_log_counts", "pmd_std_res"],
        default="log_counts",
        help="How to construct the response matrix from simulated counts (default: log_counts).",
    )
    parser.add_argument("--pmd-n-boot", type=int, default=100, help="PMD num_boot (only used for response-mode=pmd_std_res).")
    parser.add_argument(
        "--pmd-seed",
        type=int,
        default=None,
        help="PMD RNG seed (only used for response-mode=pmd_std_res); defaults to --seed.",
    )
    parser.add_argument("--include-depth-covariate", action="store_true", help="Include log_depth in model matrix as a nuisance covariate.")
    parser.add_argument("--allow-random-slope", action=argparse.BooleanOptionalAction, default=True, help="Allow random slope in LMM (default: True).")
    parser.add_argument("--min-guides-random-slope", type=int, default=3, help="Minimum guides for RI+RS (default: 3).")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations per MixedLM fit (default: 200).")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha for summary metrics.")
    parser.add_argument("--fdr-q", type=float, default=0.1, help="FDR threshold for summary metrics.")
    parser.add_argument(
        "--qq-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write QQ plots for null p-values (default: enabled; requires matplotlib).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["meta", "lmm", "qc"],
        default=["meta", "lmm", "qc"],
        help="Which gene-level methods to run (default: meta lmm qc).",
    )
    args = parser.parse_args()

    cfg = CountDepthBenchmarkConfig(
        n_genes=args.n_genes,
        guides_per_gene=args.guides_per_gene,
        n_control=args.n_control,
        n_treatment=args.n_treatment,
        guide_lambda_log_mean=args.guide_lambda_log_mean,
        guide_lambda_log_sd=args.guide_lambda_log_sd,
        gene_lambda_log_sd=args.gene_lambda_log_sd,
        depth_log_mean=args.depth_log_mean,
        depth_log_sd=args.depth_log_sd,
        depth_poisson_scale=args.depth_poisson_scale,
        treatment_depth_multiplier=args.treatment_depth_multiplier,
        frac_signal=args.frac_signal,
        effect_sd=args.effect_sd,
        guide_slope_sd=args.guide_slope_sd,
        offtarget_guide_frac=args.offtarget_guide_frac,
        offtarget_slope_sd=args.offtarget_slope_sd,
        pseudocount=args.pseudocount,
        response_mode=str(args.response_mode),
        pmd_n_boot=int(args.pmd_n_boot),
        pmd_seed=int(args.seed if args.pmd_seed is None else args.pmd_seed),
        include_depth_covariate=bool(args.include_depth_covariate),
        allow_random_slope=bool(args.allow_random_slope),
        min_guides_random_slope=args.min_guides_random_slope,
        max_iter=args.max_iter,
        methods=tuple([str(m) for m in args.methods]),
        seed=args.seed,
        alpha=args.alpha,
        fdr_q=args.fdr_q,
        qq_plots=bool(args.qq_plots),
    )

    report = run_benchmark(cfg, args.out_dir)
    report_path = os.path.join(args.out_dir, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(report), f, indent=2, sort_keys=True, allow_nan=False)
    print(report_path)


if __name__ == "__main__":
    main()
