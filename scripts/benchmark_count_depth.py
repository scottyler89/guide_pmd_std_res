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
from guide_pmd.gene_level_stouffer import compute_gene_stouffer


def _ks_uniform(p: pd.Series) -> dict[str, float | None]:
    from scipy.stats import kstest

    p_arr = p.to_numpy(dtype=float)
    mask = np.isfinite(p_arr) & (p_arr >= 0.0) & (p_arr <= 1.0)
    p_arr = p_arr[mask]
    if p_arr.size == 0:
        return {"n": 0.0, "ks": None, "ks_p": None}
    res = kstest(p_arr, "uniform")
    return {"n": float(p_arr.size), "ks": float(res.statistic), "ks_p": float(res.pvalue)}


def _p_hist(p: pd.Series, *, n_bins: int = 20) -> dict[str, object]:
    p_arr = p.to_numpy(dtype=float)
    mask = np.isfinite(p_arr) & (p_arr >= 0.0) & (p_arr <= 1.0)
    p_arr = p_arr[mask]
    if p_arr.size == 0:
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=float)
        return {"n": 0.0, "bin_edges": edges.tolist(), "counts": [0.0] * int(n_bins)}
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=float)
    counts, _ = np.histogram(p_arr, bins=edges)
    return {"n": float(p_arr.size), "bin_edges": edges.tolist(), "counts": counts.astype(float).tolist()}


def _roc_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    from scipy.stats import rankdata

    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(np.sum(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(s, method="average")  # increasing
    sum_ranks_pos = float(np.sum(ranks[y]))
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def _average_precision(y_true: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(np.sum(y))
    if n_pos == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted, dtype=float)
    precision = tp / (np.arange(y_sorted.size, dtype=float) + 1.0)
    ap = float(np.sum(precision[y_sorted]) / float(n_pos))
    return ap


def _score_from_p(p: pd.Series) -> np.ndarray:
    p_arr = p.to_numpy(dtype=float)
    p_arr = np.clip(p_arr, 1e-300, 1.0)
    return -np.log10(p_arr)


def _score_from_p_impute_nan_as_1(p: pd.Series) -> np.ndarray:
    p_arr = p.to_numpy(dtype=float)
    p_arr = np.where(np.isfinite(p_arr), p_arr, 1.0)
    p_arr = np.clip(p_arr, 1e-300, 1.0)
    return -np.log10(p_arr)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        return None
    if float(np.std(x_arr)) <= 0.0 or float(np.std(y_arr)) <= 0.0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return None
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def _theta_metrics(joined: pd.DataFrame, *, theta_col: str) -> dict[str, float | None]:
    theta_true = pd.to_numeric(joined["theta_true"], errors="coerce").to_numpy(dtype=float)
    theta_hat = pd.to_numeric(joined[theta_col], errors="coerce").to_numpy(dtype=float)
    is_signal = joined["is_signal"].to_numpy(dtype=bool)

    corr_all = _safe_corr(theta_true, theta_hat)
    rmse_all = _rmse(theta_true, theta_hat)

    corr_signal = _safe_corr(theta_true[is_signal], theta_hat[is_signal])
    rmse_signal = _rmse(theta_true[is_signal], theta_hat[is_signal])

    mask_sig = is_signal & np.isfinite(theta_true) & np.isfinite(theta_hat) & (theta_true != 0.0)
    sign_acc_signal = None
    if int(np.sum(mask_sig)) > 0:
        sign_acc_signal = float(np.mean(np.sign(theta_true[mask_sig]) == np.sign(theta_hat[mask_sig])))

    return {
        "n_all": float(int(np.sum(np.isfinite(theta_true) & np.isfinite(theta_hat)))),
        "corr_all": corr_all,
        "rmse_all": rmse_all,
        "n_signal": float(int(np.sum(np.isfinite(theta_true[is_signal]) & np.isfinite(theta_hat[is_signal])))),
        "corr_signal": corr_signal,
        "rmse_signal": rmse_signal,
        "sign_acc_signal": sign_acc_signal,
    }


def _theta_bias_null(joined: pd.DataFrame, *, theta_col: str) -> dict[str, float | None]:
    theta_hat = pd.to_numeric(joined[theta_col], errors="coerce").to_numpy(dtype=float)
    is_signal = joined["is_signal"].to_numpy(dtype=bool)
    null_mask = (~is_signal) & np.isfinite(theta_hat)
    if int(np.sum(null_mask)) == 0:
        return {"n_null": 0.0, "mean_null": None, "median_null": None, "mean_abs_null": None}
    vals = theta_hat[null_mask]
    return {
        "n_null": float(int(vals.size)),
        "mean_null": float(np.mean(vals)),
        "median_null": float(np.median(vals)),
        "mean_abs_null": float(np.mean(np.abs(vals))),
    }


def _write_mean_dispersion_tables(
    counts_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    *,
    out_dir: str,
) -> tuple[str, dict[str, float | None]]:
    """
    Write per-guide mean/variance/dispersion summaries for simulated counts.

    Dispersion is a simple method-of-moments estimate:
        phi_hat = max(0, (var - mean) / mean^2)

    For NB counts (Var = mu + phi * mu^2), phi_hat targets phi in expectation.
    """
    guides = counts_df.index.astype(str).to_numpy()
    gene = annotation_df.iloc[:, 0].astype(str).reindex(counts_df.index).to_numpy()
    x = counts_df.to_numpy(dtype=float)
    mean = np.mean(x, axis=1)
    var = np.var(x, axis=1, ddof=1) if x.shape[1] > 1 else np.zeros_like(mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_hat = (var - mean) / np.maximum(mean, np.finfo(float).tiny) ** 2
    phi_hat = np.where(np.isfinite(phi_hat) & (phi_hat > 0.0), phi_hat, 0.0)

    out_path = os.path.join(out_dir, "sim_counts_mean_dispersion.tsv")
    df = pd.DataFrame(
        {
            "guide_id": guides,
            "gene_id": gene,
            "mean": mean.astype(float),
            "var": var.astype(float),
            "phi_hat": phi_hat.astype(float),
        }
    )
    df.to_csv(out_path, sep="\t", index=False)

    # Aggregate summaries for the report.
    mv_mask = np.isfinite(mean) & np.isfinite(var) & (mean > 0) & (var > 0)
    corr = None
    if int(np.sum(mv_mask)) > 2:
        corr = float(np.corrcoef(np.log(mean[mv_mask]), np.log(var[mv_mask]))[0, 1])
    qc = {
        "n_guides": float(df.shape[0]),
        "median_mean": float(np.nanmedian(mean)),
        "median_phi_hat": float(np.nanmedian(phi_hat)),
        "corr_log_mean_log_var": corr,
    }
    return out_path, qc


def _write_gene_mean_dispersion_tables(
    counts_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    *,
    out_dir: str,
) -> tuple[str, dict[str, float | None]]:
    """
    Write per-gene mean/variance/dispersion summaries from simulated counts.

    Aggregation: sum counts across guides within gene per sample, then compute
    mean/variance across samples.
    """
    gene = annotation_df.iloc[:, 0].astype(str).reindex(counts_df.index)
    gene_counts = counts_df.copy()
    gene_counts.insert(0, "_gene_id", gene.to_numpy(dtype=str))
    gene_counts = gene_counts.groupby("_gene_id", sort=True).sum(numeric_only=True)

    x = gene_counts.to_numpy(dtype=float)
    mean = np.mean(x, axis=1)
    var = np.var(x, axis=1, ddof=1) if x.shape[1] > 1 else np.zeros_like(mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_hat = (var - mean) / np.maximum(mean, np.finfo(float).tiny) ** 2
    phi_hat = np.where(np.isfinite(phi_hat) & (phi_hat > 0.0), phi_hat, 0.0)

    out_path = os.path.join(out_dir, "sim_counts_gene_mean_dispersion.tsv")
    df = pd.DataFrame(
        {
            "gene_id": gene_counts.index.astype(str).to_numpy(),
            "n_guides": annotation_df.iloc[:, 0].astype(str).value_counts().reindex(gene_counts.index).fillna(0).astype(int).to_numpy(),
            "mean": mean.astype(float),
            "var": var.astype(float),
            "phi_hat": phi_hat.astype(float),
        }
    )
    df.to_csv(out_path, sep="\t", index=False)

    mv_mask = np.isfinite(mean) & np.isfinite(var) & (mean > 0) & (var > 0)
    corr = None
    if int(np.sum(mv_mask)) > 2:
        corr = float(np.corrcoef(np.log(mean[mv_mask]), np.log(var[mv_mask]))[0, 1])
    qc = {
        "n_genes": float(df.shape[0]),
        "median_mean": float(np.nanmedian(mean)),
        "median_phi_hat": float(np.nanmedian(phi_hat)),
        "corr_log_mean_log_var": corr,
    }
    return out_path, qc


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
    # Optional batch structure + confounding (currently supports n_batches in {1, 2}).
    n_batches: int
    batch_confounding_strength: float
    batch_depth_log_sd: float
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
    normalization_mode: str
    logratio_mode: str
    n_reference_genes: int
    pmd_n_boot: int
    pmd_seed: int
    # Count distribution / overdispersion (Var = mu + nb_overdispersion * mu^2; 0 => Poisson).
    nb_overdispersion: float
    # Depth covariate mode (observed-only; derived from the simulated counts, not oracle depth_factor).
    depth_covariate_mode: str
    # Deprecated/compat: whether any depth covariate is included.
    include_depth_covariate: bool
    # Whether to include batch indicators as nuisance covariates in the model matrix.
    include_batch_covariate: bool
    # LMM options (keep small for speed by default).
    allow_random_slope: bool
    min_guides_random_slope: int
    max_iter: int
    # Plan A (LMM) selection policy (mirrors the main pipeline defaults).
    lmm_scope: str
    lmm_q_meta: float
    lmm_q_het: float
    lmm_audit_n: int
    lmm_audit_seed: int
    lmm_max_genes_per_focal_var: int | None
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
        if int(self.n_batches) not in {1, 2}:
            raise ValueError("n_batches must be in {1, 2}")
        if not (0.0 <= float(self.batch_confounding_strength) <= 1.0):
            raise ValueError("batch_confounding_strength must be in [0, 1]")
        if float(self.batch_depth_log_sd) < 0:
            raise ValueError("batch_depth_log_sd must be >= 0")
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
        if str(self.normalization_mode) not in {"none", "libsize_to_mean", "cpm", "median_ratio"}:
            raise ValueError("normalization_mode must be one of: none, libsize_to_mean, cpm, median_ratio")
        if str(self.logratio_mode) not in {"none", "clr_all", "alr_refset"}:
            raise ValueError("logratio_mode must be one of: none, clr_all, alr_refset")
        if int(self.n_reference_genes) < 0:
            raise ValueError("n_reference_genes must be >= 0")
        if str(self.logratio_mode) == "alr_refset" and int(self.n_reference_genes) < 1:
            raise ValueError("n_reference_genes must be >= 1 for logratio_mode=alr_refset")
        if self.response_mode == "pmd_std_res":
            if str(self.normalization_mode) != "none":
                raise ValueError("normalization_mode must be 'none' for response_mode=pmd_std_res")
            if str(self.logratio_mode) != "none":
                raise ValueError("logratio_mode must be 'none' for response_mode=pmd_std_res")
        if self.response_mode == "pmd_std_res":
            if int(self.pmd_n_boot) < 2:
                raise ValueError("pmd_n_boot must be >= 2 (required for valid PMD z-scores)")
            if int(self.pmd_seed) < 0:
                raise ValueError("pmd_seed must be >= 0")
        if float(self.nb_overdispersion) < 0:
            raise ValueError("nb_overdispersion must be >= 0")
        if str(self.depth_covariate_mode) not in {"none", "log_libsize"}:
            raise ValueError("depth_covariate_mode must be one of: none, log_libsize")
        include_depth_expected = str(self.depth_covariate_mode) != "none"
        if bool(self.include_depth_covariate) != include_depth_expected:
            raise ValueError("include_depth_covariate must match depth_covariate_mode (deprecated compat field)")
        if self.min_guides_random_slope < 2:
            raise ValueError("min_guides_random_slope must be >= 2")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if str(self.lmm_scope) not in {"all", "meta_fdr", "meta_or_het_fdr", "none"}:
            raise ValueError("lmm_scope must be one of: all, meta_fdr, meta_or_het_fdr, none")
        if not (0.0 < float(self.lmm_q_meta) <= 1.0):
            raise ValueError("lmm_q_meta must be in (0, 1]")
        if not (0.0 < float(self.lmm_q_het) <= 1.0):
            raise ValueError("lmm_q_het must be in (0, 1]")
        if int(self.lmm_audit_n) < 0:
            raise ValueError("lmm_audit_n must be >= 0")
        if int(self.lmm_audit_seed) < 0:
            raise ValueError("lmm_audit_seed must be >= 0")
        if self.lmm_max_genes_per_focal_var is not None and int(self.lmm_max_genes_per_focal_var) < 1:
            raise ValueError("lmm_max_genes_per_focal_var must be >= 1 or None")
        if not self.methods:
            raise ValueError("methods must not be empty")
        allowed_methods = {"meta", "stouffer", "lmm", "qc"}
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


def _assign_batches(
    rng: np.random.Generator,
    *,
    treatment: np.ndarray,
    n_batches: int,
    confounding_strength: float,
) -> np.ndarray:
    n_batches = int(n_batches)
    if n_batches == 1:
        return np.zeros(int(treatment.size), dtype=int)
    if n_batches != 2:
        raise ValueError("only n_batches in {1, 2} are supported")

    treat = np.asarray(treatment, dtype=float)
    is_treat = treat > 0
    n_treat = int(np.sum(is_treat))
    n_ctrl = int(treat.size - n_treat)

    strength = float(confounding_strength)
    p_treat_b1 = 0.5 + 0.5 * strength
    p_ctrl_b1 = 0.5 - 0.5 * strength

    n_treat_b1 = int(np.clip(np.round(p_treat_b1 * n_treat), 0, n_treat))
    n_ctrl_b1 = int(np.clip(np.round(p_ctrl_b1 * n_ctrl), 0, n_ctrl))

    ctrl_batches = np.array([1] * n_ctrl_b1 + [0] * (n_ctrl - n_ctrl_b1), dtype=int)
    treat_batches = np.array([1] * n_treat_b1 + [0] * (n_treat - n_treat_b1), dtype=int)

    ctrl_batches = rng.permutation(ctrl_batches)
    treat_batches = rng.permutation(treat_batches)

    out = np.empty(int(treat.size), dtype=int)
    out[~is_treat] = ctrl_batches
    out[is_treat] = treat_batches
    return out


def _draw_counts(rng: np.random.Generator, mu: np.ndarray, *, nb_overdispersion: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    phi = float(nb_overdispersion)
    if phi <= 0:
        return rng.poisson(mu).astype(int)

    # Gamma-Poisson mixture: if gamma ~ Gamma(shape=1/phi, scale=phi), then E[gamma]=1, Var[gamma]=phi and
    # counts ~ Poisson(mu * gamma) implies Var[counts] = mu + phi * mu^2.
    shape = 1.0 / phi
    scale = phi
    gamma = rng.gamma(shape=shape, scale=scale, size=mu.shape).astype(float)
    return rng.poisson(mu * gamma).astype(int)


def _normalize_counts(counts_df: pd.DataFrame, *, mode: str) -> np.ndarray:
    mode = str(mode)
    counts = counts_df.to_numpy(dtype=float)
    if mode == "none":
        return counts

    libsize = np.sum(counts, axis=0).astype(float)
    libsize = np.clip(libsize, 1.0, None)

    if mode == "libsize_to_mean":
        mean_libsize = float(np.mean(libsize))
        return counts * (mean_libsize / libsize)[None, :]
    if mode == "cpm":
        return (counts / libsize[None, :]) * 1e6
    if mode == "median_ratio":
        positive_rows = (counts > 0).all(axis=1)
        if not np.any(positive_rows):
            raise ValueError("median_ratio normalization requires at least one guide with all-positive counts")
        gmean = np.exp(np.mean(np.log(counts[positive_rows, :]), axis=1)).astype(float)  # (n_guides_pos,)
        ratios = counts[positive_rows, :] / gmean[:, None]
        size_factors = np.median(ratios, axis=0).astype(float)
        size_factors = np.clip(size_factors, np.finfo(float).tiny, None)
        return counts / size_factors[None, :]

    raise ValueError(f"unknown normalization_mode: {mode}")


def _apply_logratio(
    log_values: np.ndarray,
    *,
    mode: str,
    ref_mask: np.ndarray | None = None,
) -> np.ndarray:
    mode = str(mode)
    log_values = np.asarray(log_values, dtype=float)
    if mode == "none":
        return log_values
    if mode == "clr_all":
        return log_values - np.mean(log_values, axis=0, keepdims=True)
    if mode == "alr_refset":
        if ref_mask is None:
            raise ValueError("ref_mask is required for logratio_mode=alr_refset")
        ref_mask = np.asarray(ref_mask, dtype=bool)
        if ref_mask.ndim != 1 or ref_mask.shape[0] != log_values.shape[0]:
            raise ValueError("ref_mask must be 1D with length n_guides")
        if not np.any(ref_mask):
            raise ValueError("logratio_mode=alr_refset requires a non-empty reference guide set")
        ref_mean = np.mean(log_values[ref_mask, :], axis=0, keepdims=True)
        return log_values - ref_mean
    raise ValueError(f"unknown logratio_mode: {mode}")


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
    # For benchmarking/reporting treat "no calls" as FDR=0 (no discoveries => no false discoveries).
    fdr = (fp / n_called) if n_called else 0.0
    ppv = (tp / n_called) if n_called else None

    tnr = (tn / n_null) if n_null else None
    fnr = (fn / n_signal) if n_signal else None
    npv = (tn / (tn + fn)) if (tn + fn) else None
    acc = ((tp + tn) / n_total) if n_total else None
    f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) else None

    bal_acc = None
    if tpr is not None and tnr is not None:
        bal_acc = 0.5 * (float(tpr) + float(tnr))

    mcc = None
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom > 0:
        mcc = float((tp * tn - fp * fn) / float(np.sqrt(denom)))
    else:
        # Degenerate confusion matrices (all-positive or all-negative predictions) have undefined MCC.
        # For benchmarking treat this as 0.0 (no correlation).
        mcc = 0.0

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
        "tnr": tnr,
        "fnr": fnr,
        "npv": npv,
        "accuracy": acc,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
    }


def simulate_counts_and_std_res(
    cfg: CountDepthBenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg.validate()
    rng = np.random.default_rng(int(cfg.seed))

    n_control = int(cfg.n_control)
    n_treat = int(cfg.n_treatment)
    n_samples = n_control + n_treat

    sample_ids = [f"ctrl_{i:03d}" for i in range(n_control)] + [f"treat_{i:03d}" for i in range(n_treat)]
    treatment = np.array([0.0] * n_control + [1.0] * n_treat, dtype=float)

    batch_id = _assign_batches(
        rng,
        treatment=treatment,
        n_batches=int(cfg.n_batches),
        confounding_strength=float(cfg.batch_confounding_strength),
    )

    depth_factor = _simulate_depth_factors(
        rng,
        n_samples,
        depth_log_mean=cfg.depth_log_mean,
        depth_log_sd=cfg.depth_log_sd,
        depth_poisson_scale=cfg.depth_poisson_scale,
    )
    if int(cfg.n_batches) > 1 and float(cfg.batch_depth_log_sd) > 0:
        batch_shift = rng.normal(loc=0.0, scale=float(cfg.batch_depth_log_sd), size=int(cfg.n_batches)).astype(float)
        depth_factor = depth_factor * np.exp(batch_shift[batch_id])
    depth_factor = depth_factor * np.where(treatment > 0, float(cfg.treatment_depth_multiplier), 1.0)
    # "Observed" depth proxies derived from the simulated counts should be used for downstream adjustment.
    # Keep the oracle depth_factor for truth/auditing only.
    log_depth = np.log(depth_factor)

    n_target_genes = int(cfg.n_genes)
    n_ref_genes = int(cfg.n_reference_genes)
    gene_ids = [f"ref_{i:05d}" for i in range(n_ref_genes)] + [f"gene_{i:05d}" for i in range(n_target_genes)]
    is_reference = np.array([True] * n_ref_genes + [False] * n_target_genes, dtype=bool)
    is_signal_target = rng.random(n_target_genes) < float(cfg.frac_signal)
    gene_theta_target = rng.normal(loc=0.0, scale=float(cfg.effect_sd), size=n_target_genes).astype(float)
    gene_theta_target = np.where(is_signal_target, gene_theta_target, 0.0)
    gene_theta = np.concatenate([np.zeros(n_ref_genes, dtype=float), gene_theta_target]).astype(float)
    is_signal = np.concatenate([np.zeros(n_ref_genes, dtype=bool), is_signal_target]).astype(bool)

    gene_log_lambda = rng.normal(
        loc=float(cfg.guide_lambda_log_mean),
        scale=float(cfg.gene_lambda_log_sd),
        size=int(n_ref_genes + n_target_genes),
    ).astype(float)

    truth_gene = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "is_reference": is_reference.astype(bool),
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
        gene_is_signal = bool(is_signal[gene_i]) and (not bool(is_reference[gene_i]))
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
            counts = _draw_counts(rng, mu, nb_overdispersion=float(cfg.nb_overdispersion))

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

    libsize = counts_df.sum(axis=0).to_numpy(dtype=float)
    libsize = np.clip(libsize, 1.0, None)
    log_libsize = np.log(libsize)
    log_libsize_centered = log_libsize - float(np.mean(log_libsize))

    is_ref_guide = np.array([str(g).startswith("ref_") for g in gene_for_guide], dtype=bool)

    if cfg.response_mode == "pmd_std_res":
        std_res_df = _compute_pmd_std_res(counts_df, n_boot=int(cfg.pmd_n_boot), seed=int(cfg.pmd_seed))
    else:
        counts_norm = _normalize_counts(counts_df, mode=str(cfg.normalization_mode))
        log_vals = np.log(counts_norm + float(cfg.pseudocount))
        log_vals = _apply_logratio(log_vals, mode=str(cfg.logratio_mode), ref_mask=is_ref_guide)
        if cfg.response_mode == "log_counts":
            response = log_vals
        else:
            # Per-guide z-scored log values (optional; does not change per-guide OLS t/p with an intercept).
            guide_mean = np.mean(log_vals, axis=1, keepdims=True)
            guide_sd = np.std(log_vals, axis=1, ddof=1, keepdims=True)
            guide_sd = np.where(guide_sd <= 0, 1.0, guide_sd)
            response = (log_vals - guide_mean) / guide_sd
        std_res_df = pd.DataFrame(response, index=guides, columns=sample_ids)

    model_matrix = pd.DataFrame({"treatment": treatment}, index=sample_ids)
    if str(cfg.depth_covariate_mode) == "log_libsize":
        model_matrix["log_libsize_centered"] = log_libsize_centered
    if bool(cfg.include_batch_covariate) and int(cfg.n_batches) > 1:
        for b in range(1, int(cfg.n_batches)):
            model_matrix[f"batch_{b}"] = (batch_id == b).astype(float)

    truth_sample = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "treatment": treatment.astype(float),
            "batch_id": batch_id.astype(int),
            "depth_factor": depth_factor.astype(float),
            "log_depth": log_depth.astype(float),
            "libsize": libsize.astype(float),
            "log_libsize": log_libsize.astype(float),
            "log_libsize_centered": log_libsize_centered.astype(float),
        }
    )

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
        truth_sample,
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
        # For benchmarking and downstream aggregation, treat "no calls" as a valid outcome:
        #   - achieved FDR is 0 (no discoveries => no false discoveries)
        #   - achieved TPR is 0 (no signal recovered)
        return {"q": float(q), "n_called": 0.0, "fdr": 0.0, "tpr": 0.0}
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
    if isinstance(obj, np.generic):
        return _json_sanitize(obj.item())
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

    counts_df, ann_df, std_res_df, mm, truth_sample, truth_gene, truth_guide = simulate_counts_and_std_res(cfg)

    # Write inputs for reproducibility.
    counts_path = os.path.join(out_dir, "sim_counts.tsv")
    mm_path = os.path.join(out_dir, "sim_model_matrix.tsv")
    std_res_path = os.path.join(out_dir, "sim_std_res.tsv")
    truth_sample_path = os.path.join(out_dir, "sim_truth_sample.tsv")
    truth_path = os.path.join(out_dir, "sim_truth_gene.tsv")
    truth_guide_path = os.path.join(out_dir, "sim_truth_guide.tsv")

    counts_out = counts_df.copy()
    counts_out.insert(0, "gene_symbol", ann_df["gene_symbol"])
    counts_out.index.name = "guide_id"
    counts_out.to_csv(counts_path, sep="\t")
    mm.to_csv(mm_path, sep="\t", index_label="sample_id")
    std_res_df.to_csv(std_res_path, sep="\t")
    truth_sample.to_csv(truth_sample_path, sep="\t", index=False)
    truth_gene.to_csv(truth_path, sep="\t", index=False)
    truth_guide.to_csv(truth_guide_path, sep="\t", index=False)

    mean_disp_path, mean_disp_qc = _write_mean_dispersion_tables(counts_df, ann_df, out_dir=out_dir)
    mean_disp_gene_path, mean_disp_gene_qc = _write_gene_mean_dispersion_tables(counts_df, ann_df, out_dir=out_dir)
    depth_qc = {
        "n_samples": float(truth_sample.shape[0]),
        "log_libsize_mean": float(truth_sample["log_libsize"].mean()),
        "log_libsize_sd": float(truth_sample["log_libsize"].std(ddof=1)) if truth_sample.shape[0] > 1 else 0.0,
        "corr_treatment_log_libsize": _safe_corr(
            truth_sample["treatment"].to_numpy(dtype=float),
            truth_sample["log_libsize"].to_numpy(dtype=float),
        ),
    }
    counts_qc = {
        "mean_dispersion": mean_disp_qc,
        "mean_dispersion_gene": mean_disp_gene_qc,
        "depth_proxy": depth_qc,
    }

    mm_sanity = mm.copy()
    if "Intercept" not in mm_sanity.columns:
        mm_sanity.insert(0, "Intercept", 1.0)
    x = mm_sanity.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(x)) if x.size else 0
    cond = float(np.linalg.cond(x)) if x.size else None
    cols = [str(c) for c in mm_sanity.columns]
    corr_m = mm_sanity.corr().to_numpy(dtype=float)
    corr: dict[str, dict[str, float | None]] = {}
    for i, c1 in enumerate(cols):
        corr[c1] = {}
        for j, c2 in enumerate(cols):
            v = float(corr_m[i, j])
            corr[c1][c2] = v if np.isfinite(v) else None
    design_matrix = {
        "n_rows": float(mm_sanity.shape[0]),
        "n_cols": float(mm_sanity.shape[1]),
        "rank": float(rank),
        "cond": cond,
        "corr": corr,
    }

    truth_guide = truth_guide.copy()
    truth_guide["theta_dev"] = truth_guide["theta_guide"] - truth_guide["theta_gene"]
    true_het = truth_guide.groupby("gene_id")["theta_dev"].std(ddof=1).fillna(0.0).rename("theta_dev_sd").astype(float)

    focal_vars = ["treatment"]

    runtime: dict[str, float] = {}
    meta_df = pd.DataFrame()
    stouffer_df = pd.DataFrame()
    lmm_df = pd.DataFrame()
    qc_df = pd.DataFrame()

    meta_out_path = os.path.join(out_dir, "PMD_std_res_gene_meta.tsv")
    meta_needed_for_lmm = ("lmm" in cfg.methods) and (str(cfg.lmm_scope) != "all")
    if ("meta" in cfg.methods) or meta_needed_for_lmm:
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

    stouffer_out_path = os.path.join(out_dir, "PMD_std_res_gene_stouffer.tsv")
    if "stouffer" in cfg.methods:
        t0 = time.perf_counter()
        stouffer_df = compute_gene_stouffer(
            std_res_df,
            ann_df,
            mm,
            focal_vars=focal_vars,
            gene_id_col=1,
            add_intercept=True,
        )
        runtime["stouffer"] = float(time.perf_counter() - t0)
        stouffer_df.to_csv(stouffer_out_path, sep="\t", index=False)

    lmm_out_path = os.path.join(out_dir, "PMD_std_res_gene_lmm.tsv")
    lmm_sel_out_path = os.path.join(out_dir, "PMD_std_res_gene_lmm_selection.tsv")
    lmm_selection: pd.DataFrame | None = None
    if "lmm" in cfg.methods:
        if str(cfg.lmm_scope) != "all":
            if meta_df.empty:
                raise RuntimeError("internal error: meta_df is required for lmm selection but is empty")
            from guide_pmd import gene_level_selection as gene_level_selection_mod

            sel_cfg = gene_level_selection_mod.GeneLmmSelectionConfig(
                scope=str(cfg.lmm_scope),
                q_meta=float(cfg.lmm_q_meta),
                q_het=float(cfg.lmm_q_het),
                audit_n=int(cfg.lmm_audit_n),
                audit_seed=int(cfg.lmm_audit_seed),
                max_genes_per_focal_var=cfg.lmm_max_genes_per_focal_var,
            )
            feasibility = gene_level_selection_mod.compute_gene_lmm_feasibility(
                std_res_df,
                ann_df,
                mm,
                focal_vars=focal_vars,
                gene_id_col=1,
                add_intercept=True,
            )
            lmm_selection = gene_level_selection_mod.compute_gene_lmm_selection(
                meta_df,
                feasibility,
                config=sel_cfg,
            )
            lmm_selection.to_csv(lmm_sel_out_path, sep="\t", index=False)

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
            selection_table=lmm_selection,
        )
        runtime["lmm"] = float(time.perf_counter() - t0)
        lmm_df.to_csv(lmm_out_path, sep="\t", index=False)

    lmm_fit: dict[str, object] | None = None
    if "lmm" in cfg.methods:
        n_total = int(truth_gene.shape[0])
        n_attempted = int(lmm_df.shape[0]) if not lmm_df.empty else 0
        frac_attempted = float(n_attempted / n_total) if n_total > 0 else None
        method_counts: dict[str, int] = {}
        frac_method: dict[str, float] = {}
        if n_attempted > 0 and "method" in lmm_df.columns:
            vc = lmm_df["method"].astype(str).value_counts(dropna=False)
            method_counts = {str(k): int(v) for k, v in vc.to_dict().items()}
            frac_method = {str(k): float(int(v) / n_attempted) for k, v in vc.to_dict().items()}

        lrt_ok_frac_attempted = None
        if n_attempted > 0 and "lrt_ok" in lmm_df.columns:
            lrt_ok_frac_attempted = float(np.mean(lmm_df["lrt_ok"].fillna(False).astype(bool)))
        wald_ok_frac_attempted = None
        if n_attempted > 0 and "wald_ok" in lmm_df.columns:
            wald_ok_frac_attempted = float(np.mean(lmm_df["wald_ok"].fillna(False).astype(bool)))

        n_selected = None
        if lmm_selection is not None and (not lmm_selection.empty) and ("selected" in lmm_selection.columns):
            n_selected = int(np.sum(lmm_selection["selected"].astype(bool)))

        lmm_fit = {
            "n_total": float(n_total),
            "n_attempted": float(n_attempted),
            "frac_attempted": frac_attempted,
            "n_selected": float(n_selected) if n_selected is not None else None,
            "method_counts": method_counts,
            "method_fracs": frac_method,
            "lrt_ok_frac_attempted": lrt_ok_frac_attempted,
            "wald_ok_frac_attempted": wald_ok_frac_attempted,
        }

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
    meta_join = truth_gene.copy()
    stouffer_join = truth_gene.copy()
    lmm_join = truth_gene.copy()
    if not meta_df.empty:
        meta_join = truth_gene.merge(meta_df, on="gene_id", how="left")
    if not stouffer_df.empty:
        stouffer_join = truth_gene.merge(stouffer_df, on="gene_id", how="left")
    if not lmm_df.empty:
        lmm_join = truth_gene.merge(lmm_df, on="gene_id", how="left")

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
            "counts_mean_dispersion_tsv": mean_disp_path,
            "counts_gene_mean_dispersion_tsv": mean_disp_gene_path,
            "model_matrix_tsv": mm_path,
            "std_res_tsv": std_res_path,
            "truth_sample_tsv": truth_sample_path,
            "truth_gene_tsv": truth_path,
            "truth_guide_tsv": truth_guide_path,
            "gene_meta_tsv": meta_out_path if not meta_df.empty else "",
            "gene_stouffer_tsv": stouffer_out_path if not stouffer_df.empty else "",
            "gene_lmm_tsv": lmm_out_path if "lmm" in cfg.methods else "",
            "gene_lmm_selection_tsv": lmm_sel_out_path if lmm_selection is not None else "",
            "gene_qc_tsv": qc_out_path if "qc" in cfg.methods else "",
        },
        "counts_qc": counts_qc,
        "design_matrix": design_matrix,
        "runtime_sec": runtime,
    }
    if lmm_fit is not None:
        report["lmm_fit"] = lmm_fit

    if "meta" in cfg.methods and not meta_df.empty:
        meta_null = meta_join.loc[~meta_join["is_signal"], "p"]
        meta_sig = meta_join.loc[meta_join["is_signal"], "p"]
        meta_called_alpha = np.isfinite(meta_join["p"].to_numpy(dtype=float)) & (meta_join["p"].to_numpy(dtype=float) < float(cfg.alpha))
        meta_called_q = np.isfinite(meta_join["p_adj"].to_numpy(dtype=float)) & (meta_join["p_adj"].to_numpy(dtype=float) < float(cfg.fdr_q))
        report["meta"] = {
            "null": _summarize_p(meta_null, alpha=cfg.alpha),
            "signal": _summarize_p(meta_sig, alpha=cfg.alpha),
            "ks_uniform_null": _ks_uniform(meta_null),
            "p_hist_null": _p_hist(meta_null, n_bins=20),
            "roc_auc": _roc_auc(meta_join["is_signal"].to_numpy(dtype=bool), _score_from_p_impute_nan_as_1(meta_join["p"])),
            "average_precision": _average_precision(meta_join["is_signal"].to_numpy(dtype=bool), _score_from_p_impute_nan_as_1(meta_join["p"])),
            "theta_metrics": _theta_metrics(meta_join, theta_col="theta") if "theta" in meta_join.columns else {},
            "theta_bias_null": _theta_bias_null(meta_join, theta_col="theta") if "theta" in meta_join.columns else {},
            "fdr": _fdr_summary(meta_join["p_adj"], meta_join["is_signal"], q=cfg.fdr_q),
            "confusion_alpha": _confusion(meta_called_alpha, meta_join["is_signal"].to_numpy(dtype=bool)),
            "confusion_fdr_q": _confusion(meta_called_q, meta_join["is_signal"].to_numpy(dtype=bool)),
        }
    if "stouffer" in cfg.methods and not stouffer_df.empty:
        stouffer_null = stouffer_join.loc[~stouffer_join["is_signal"], "p"]
        stouffer_sig = stouffer_join.loc[stouffer_join["is_signal"], "p"]
        st_p = stouffer_join["p"].to_numpy(dtype=float)
        st_p_adj = stouffer_join["p_adj"].to_numpy(dtype=float)
        is_signal_arr = stouffer_join["is_signal"].to_numpy(dtype=bool)

        st_called_alpha = np.isfinite(st_p) & (st_p < float(cfg.alpha))
        st_called_q = np.isfinite(st_p_adj) & (st_p_adj < float(cfg.fdr_q))
        report["stouffer"] = {
            "null": _summarize_p(stouffer_null, alpha=cfg.alpha),
            "signal": _summarize_p(stouffer_sig, alpha=cfg.alpha),
            "ks_uniform_null": _ks_uniform(stouffer_null),
            "p_hist_null": _p_hist(stouffer_null, n_bins=20),
            "roc_auc": _roc_auc(is_signal_arr, _score_from_p_impute_nan_as_1(stouffer_join["p"])),
            "average_precision": _average_precision(is_signal_arr, _score_from_p_impute_nan_as_1(stouffer_join["p"])),
            "fdr": _fdr_summary(stouffer_join["p_adj"], stouffer_join["is_signal"], q=cfg.fdr_q),
            "confusion_alpha": _confusion(st_called_alpha, is_signal_arr),
            "confusion_fdr_q": _confusion(st_called_q, is_signal_arr),
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
            "ks_uniform_null": _ks_uniform(lrt_null),
            "p_hist_null": _p_hist(lrt_null, n_bins=20),
            "roc_auc": _roc_auc(is_signal_arr, _score_from_p_impute_nan_as_1(lmm_join["lrt_p"])),
            "average_precision": _average_precision(is_signal_arr, _score_from_p_impute_nan_as_1(lmm_join["lrt_p"])),
            "theta_metrics": _theta_metrics(lmm_join, theta_col="theta") if "theta" in lmm_join.columns else {},
            "theta_bias_null": _theta_bias_null(lmm_join, theta_col="theta") if "theta" in lmm_join.columns else {},
            "fdr": _fdr_summary(lmm_join["lrt_p_adj"], lmm_join["is_signal"], q=cfg.fdr_q) if "lrt_p_adj" in lmm_join.columns else {},
            "lrt_ok_frac": float(np.mean(lmm_join["lrt_ok"].fillna(False).astype(bool))) if "lrt_ok" in lmm_join.columns else np.nan,
            "confusion_alpha": _confusion(lrt_called_alpha, is_signal_arr),
            "confusion_fdr_q": _confusion(lrt_called_q, is_signal_arr),
        }
        report["lmm_wald"] = {
            "null": _summarize_p(wald_null, alpha=cfg.alpha),
            "signal": _summarize_p(wald_sig, alpha=cfg.alpha),
            "ks_uniform_null": _ks_uniform(wald_null),
            "p_hist_null": _p_hist(wald_null, n_bins=20),
            "roc_auc": _roc_auc(is_signal_arr, _score_from_p_impute_nan_as_1(lmm_join["wald_p"])),
            "average_precision": _average_precision(is_signal_arr, _score_from_p_impute_nan_as_1(lmm_join["wald_p"])),
            "theta_metrics": _theta_metrics(lmm_join, theta_col="theta") if "theta" in lmm_join.columns else {},
            "theta_bias_null": _theta_bias_null(lmm_join, theta_col="theta") if "theta" in lmm_join.columns else {},
            "fdr": _fdr_summary(lmm_join["wald_p_adj"], lmm_join["is_signal"], q=cfg.fdr_q) if "wald_p_adj" in lmm_join.columns else {},
            "wald_ok_frac": float(np.mean(lmm_join["wald_ok"].fillna(False).astype(bool))) if "wald_ok" in lmm_join.columns else np.nan,
            "confusion_alpha": _confusion(wald_called_alpha, is_signal_arr),
            "confusion_fdr_q": _confusion(wald_called_q, is_signal_arr),
        }

    heterogeneity: dict[str, object] = {
        "theta_dev_sd_median": float(np.nanmedian(true_het.to_numpy(dtype=float))) if not true_het.empty else None,
        "theta_dev_sd_mean": float(np.nanmean(true_het.to_numpy(dtype=float))) if not true_het.empty else None,
        "theta_dev_sd_tsv_note": "Use sim_truth_guide.tsv to recompute alternative heterogeneity summaries per gene.",
    }
    if ("meta" in cfg.methods) and (not meta_df.empty) and ("tau" in meta_join.columns):
        tau_hat = pd.to_numeric(meta_join["tau"], errors="coerce").to_numpy(dtype=float)
        tau_true = meta_join["gene_id"].astype(str).map(true_het).to_numpy(dtype=float)
        mask = np.isfinite(tau_hat) & np.isfinite(tau_true)
        heterogeneity["meta_tau_corr_true"] = _safe_corr(tau_true[mask], tau_hat[mask])
        heterogeneity["meta_tau_n"] = float(int(np.sum(mask)))
    if ("lmm" in cfg.methods) and (not lmm_df.empty) and ("tau" in lmm_join.columns):
        tau_hat = pd.to_numeric(lmm_join["tau"], errors="coerce").to_numpy(dtype=float)
        tau_true = lmm_join["gene_id"].astype(str).map(true_het).to_numpy(dtype=float)
        mask = np.isfinite(tau_hat) & np.isfinite(tau_true)
        heterogeneity["lmm_tau_corr_true"] = _safe_corr(tau_true[mask], tau_hat[mask])
        heterogeneity["lmm_tau_n"] = float(int(np.sum(mask)))
    report["heterogeneity"] = heterogeneity
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

    if "stouffer" in cfg.methods and not stouffer_df.empty:
        if bool(cfg.qq_plots):
            os.makedirs(fig_dir, exist_ok=True)
            out_path = os.path.join(fig_dir, "qq_stouffer_p_null.png")
            report["outputs"]["stouffer_p_null_png"] = out_path
            qq["stouffer_p_null"] = _write_qq_plot(stouffer_null, out_path=out_path, title="Stouffer p (null)")
            wrote_any_plot = True
        else:
            qq["stouffer_p_null"] = _qq_stats(stouffer_null)

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
    parser.add_argument("--n-genes", type=int, default=500, help="Number of target genes (excluding reference genes).")
    parser.add_argument("--guides-per-gene", type=int, default=4, help="Guides per gene.")
    parser.add_argument("--n-control", type=int, default=12, help="Number of control samples.")
    parser.add_argument("--n-treatment", type=int, default=12, help="Number of treatment samples.")
    parser.add_argument("--guide-lambda-log-mean", type=float, default=np.log(200.0), help="log-mean baseline guide lambda.")
    parser.add_argument("--guide-lambda-log-sd", type=float, default=0.8, help="log-sd baseline guide lambda.")
    parser.add_argument("--gene-lambda-log-sd", type=float, default=0.5, help="Additional gene-level log-sd on lambda (default: 0.5).")
    parser.add_argument("--depth-log-mean", type=float, default=0.0, help="log-mean of depth factor.")
    parser.add_argument("--depth-log-sd", type=float, default=1.0, help="log-sd of depth factor (order-of-magnitude variation ~ 1).")
    parser.add_argument("--depth-poisson-scale", type=float, default=0.0, help="Optional Poisson noise on depth factors (0 disables).")
    parser.add_argument("--n-batches", type=int, default=1, help="Number of batches (supports 1 or 2; default: 1).")
    parser.add_argument(
        "--batch-confounding-strength",
        type=float,
        default=0.0,
        help="Strength of treatment↔batch confounding in [0,1] (only used when n_batches=2).",
    )
    parser.add_argument(
        "--batch-depth-log-sd",
        type=float,
        default=0.0,
        help="Per-batch log-depth shift SD (default: 0; only meaningful when n_batches>1).",
    )
    parser.add_argument("--treatment-depth-multiplier", type=float, default=1.0, help="Depth multiplier applied to treatment samples.")
    parser.add_argument("--frac-signal", type=float, default=0.2, help="Fraction of truly non-null genes.")
    parser.add_argument("--effect-sd", type=float, default=0.5, help="SD of gene-level treatment effects.")
    parser.add_argument("--guide-slope-sd", type=float, default=0.2, help="SD of guide-level slope deviations.")
    parser.add_argument("--offtarget-guide-frac", type=float, default=0.0, help="Fraction of guides with off-target effects (default: 0).")
    parser.add_argument("--offtarget-slope-sd", type=float, default=0.0, help="SD of off-target slope deviations (default: 0).")
    parser.add_argument("--pseudocount", type=float, default=0.5, help="Pseudocount used in log transform.")
    parser.add_argument(
        "--n-reference-genes",
        type=int,
        default=0,
        help="Number of always-null reference genes (each with guides_per_gene guides). Required for logratio-mode=alr_refset.",
    )
    parser.add_argument(
        "--response-mode",
        type=str,
        choices=["log_counts", "guide_zscore_log_counts", "pmd_std_res"],
        default="log_counts",
        help="How to construct the response matrix from simulated counts (default: log_counts).",
    )
    parser.add_argument(
        "--normalization-mode",
        type=str,
        choices=["none", "libsize_to_mean", "cpm", "median_ratio"],
        default="none",
        help="Optional count normalization applied before log transform (default: none). Not supported for response-mode=pmd_std_res.",
    )
    parser.add_argument(
        "--logratio-mode",
        type=str,
        choices=["none", "clr_all", "alr_refset"],
        default="none",
        help="Optional compositional log-ratio transform applied in log-space (default: none). Not supported for response-mode=pmd_std_res.",
    )
    parser.add_argument("--pmd-n-boot", type=int, default=100, help="PMD num_boot (only used for response-mode=pmd_std_res).")
    parser.add_argument(
        "--pmd-seed",
        type=int,
        default=None,
        help="PMD RNG seed (only used for response-mode=pmd_std_res); defaults to --seed.",
    )
    parser.add_argument(
        "--include-depth-covariate",
        action="store_true",
        help="Include an observed depth proxy (log_libsize_centered = log(colsum(counts)) centered) in the model matrix as a nuisance covariate.",
    )
    parser.add_argument(
        "--depth-covariate-mode",
        type=str,
        choices=["none", "log_libsize"],
        default=None,
        help="Depth covariate mode (observed-only). If provided, overrides `--include-depth-covariate`.",
    )
    parser.add_argument(
        "--include-batch-covariate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include batch indicators in the model matrix (default: False).",
    )
    parser.add_argument(
        "--nb-overdispersion",
        type=float,
        default=0.0,
        help="Negative-binomial overdispersion phi (Var = mu + phi * mu^2; 0 => Poisson).",
    )
    parser.add_argument("--allow-random-slope", action=argparse.BooleanOptionalAction, default=True, help="Allow random slope in LMM (default: True).")
    parser.add_argument("--min-guides-random-slope", type=int, default=3, help="Minimum guides for RI+RS (default: 3).")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations per MixedLM fit (default: 200).")
    parser.add_argument(
        "--lmm-scope",
        type=str,
        choices=["all", "meta_fdr", "meta_or_het_fdr", "none"],
        default="all",
        help="Plan A (LMM) gene selection policy (default: all).",
    )
    parser.add_argument("--lmm-q-meta", type=float, default=0.1, help="LMM selection: meta FDR threshold q_meta (default: 0.1).")
    parser.add_argument("--lmm-q-het", type=float, default=0.1, help="LMM selection: heterogeneity FDR threshold q_het (default: 0.1).")
    parser.add_argument("--lmm-audit-n", type=int, default=50, help="LMM selection: audit sample size (default: 50).")
    parser.add_argument("--lmm-audit-seed", type=int, default=123456, help="LMM selection: audit seed (default: 123456).")
    parser.add_argument(
        "--lmm-max-genes-per-focal-var",
        type=int,
        default=None,
        help="LMM selection: cap selected genes per focal var (default: None).",
    )
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
        choices=["meta", "stouffer", "lmm", "qc"],
        default=["meta", "stouffer", "lmm", "qc"],
        help="Which gene-level methods to run (default: meta stouffer lmm qc).",
    )
    args = parser.parse_args()

    if args.depth_covariate_mode is None:
        depth_covariate_mode = "log_libsize" if bool(args.include_depth_covariate) else "none"
    else:
        depth_covariate_mode = str(args.depth_covariate_mode)
        if bool(args.include_depth_covariate) and depth_covariate_mode == "none":
            raise ValueError("Conflicting args: --include-depth-covariate with --depth-covariate-mode=none")

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
        n_batches=int(args.n_batches),
        batch_confounding_strength=float(args.batch_confounding_strength),
        batch_depth_log_sd=float(args.batch_depth_log_sd),
        treatment_depth_multiplier=args.treatment_depth_multiplier,
        frac_signal=args.frac_signal,
        effect_sd=args.effect_sd,
        guide_slope_sd=args.guide_slope_sd,
        offtarget_guide_frac=args.offtarget_guide_frac,
        offtarget_slope_sd=args.offtarget_slope_sd,
        pseudocount=args.pseudocount,
        response_mode=str(args.response_mode),
        normalization_mode=str(args.normalization_mode),
        logratio_mode=str(args.logratio_mode),
        n_reference_genes=int(args.n_reference_genes),
        pmd_n_boot=int(args.pmd_n_boot),
        pmd_seed=int(args.seed if args.pmd_seed is None else args.pmd_seed),
        nb_overdispersion=float(args.nb_overdispersion),
        depth_covariate_mode=str(depth_covariate_mode),
        include_depth_covariate=bool(depth_covariate_mode != "none"),
        include_batch_covariate=bool(args.include_batch_covariate),
        allow_random_slope=bool(args.allow_random_slope),
        min_guides_random_slope=args.min_guides_random_slope,
        max_iter=args.max_iter,
        lmm_scope=str(args.lmm_scope),
        lmm_q_meta=float(args.lmm_q_meta),
        lmm_q_het=float(args.lmm_q_het),
        lmm_audit_n=int(args.lmm_audit_n),
        lmm_audit_seed=int(args.lmm_audit_seed),
        lmm_max_genes_per_focal_var=args.lmm_max_genes_per_focal_var,
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
