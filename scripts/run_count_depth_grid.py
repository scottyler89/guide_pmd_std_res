from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from itertools import product

import pandas as pd


_CONFUSION_KEYS = [
    "n_total",
    "n_signal",
    "n_null",
    "n_called",
    "tp",
    "fp",
    "tn",
    "fn",
    "tpr",
    "fpr",
    "fdr",
    "ppv",
    "tnr",
    "fnr",
    "npv",
    "accuracy",
    "f1",
    "balanced_accuracy",
    "mcc",
]


def _add_confusion_metrics(row: dict[str, object], *, out_prefix: str, confusion: object) -> None:
    if not isinstance(confusion, dict):
        return
    for k in _CONFUSION_KEYS:
        row[f"{out_prefix}_{k}"] = confusion.get(k)


def _stable_hash(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _run_one(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    # benchmark_count_depth prints the report path.
    out = proc.stdout.strip().splitlines()
    if not out:
        raise RuntimeError(f"expected benchmark report path on stdout; got empty stdout\nstderr:\n{proc.stderr}")
    return out[-1].strip()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _report_path_for_run_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "benchmark_report.json")


def _row_from_report(*, tag: str, report_path: str, report: dict) -> dict[str, object]:
    cfg = report["config"]

    row: dict[str, object] = {
        "tag": tag,
        "report_path": report_path,
        "seed": int(cfg["seed"]),
        "pmd_n_boot": int(cfg["pmd_n_boot"]),
        "qq_plots": bool(cfg["qq_plots"]),
        "alpha": float(cfg["alpha"]),
        "fdr_q": float(cfg["fdr_q"]),
        "n_genes": int(cfg["n_genes"]),
        "guides_per_gene": int(cfg["guides_per_gene"]),
        "n_control": int(cfg["n_control"]),
        "n_treatment": int(cfg["n_treatment"]),
        "n_samples": int(cfg["n_control"]) + int(cfg["n_treatment"]),
        "normalization_mode": str(cfg.get("normalization_mode", "")),
        "logratio_mode": str(cfg.get("logratio_mode", "")),
        "n_reference_genes": int(cfg.get("n_reference_genes", 0)),
        "depth_log_sd": float(cfg["depth_log_sd"]),
        "n_batches": int(cfg["n_batches"]),
        "batch_confounding_strength": float(cfg["batch_confounding_strength"]),
        "batch_depth_log_sd": float(cfg["batch_depth_log_sd"]),
        "treatment_depth_multiplier": float(cfg["treatment_depth_multiplier"]),
        "include_depth_covariate": bool(cfg["include_depth_covariate"]),
        "depth_covariate_mode": str(cfg.get("depth_covariate_mode", "")),
        "include_batch_covariate": bool(cfg["include_batch_covariate"]),
        "response_mode": str(cfg["response_mode"]),
        "guide_lambda_log_mean": float(cfg["guide_lambda_log_mean"]),
        "guide_lambda_log_sd": float(cfg["guide_lambda_log_sd"]),
        "gene_lambda_log_sd": float(cfg["gene_lambda_log_sd"]),
        "gene_lambda_family": str(cfg.get("gene_lambda_family", "lognormal")),
        "gene_lambda_mix_pi_high": float(cfg.get("gene_lambda_mix_pi_high", 0.0)),
        "gene_lambda_mix_delta_log_mean": float(cfg.get("gene_lambda_mix_delta_log_mean", 0.0)),
        "gene_lambda_power_alpha": float(cfg.get("gene_lambda_power_alpha", 0.0)),
        "guide_lambda_family": str(cfg.get("guide_lambda_family", "lognormal_noise")),
        "guide_lambda_dirichlet_alpha0": float(cfg.get("guide_lambda_dirichlet_alpha0", 0.0)),
        "methods": ",".join([str(m) for m in cfg["methods"]]),
        "lmm_scope": str(cfg["lmm_scope"]),
        "lmm_q_meta": float(cfg["lmm_q_meta"]),
        "lmm_q_het": float(cfg["lmm_q_het"]),
        "lmm_audit_n": int(cfg["lmm_audit_n"]),
        "lmm_audit_seed": int(cfg["lmm_audit_seed"]),
        "lmm_max_genes_per_focal_var": cfg["lmm_max_genes_per_focal_var"],
        "frac_signal": float(cfg["frac_signal"]),
        "effect_sd": float(cfg["effect_sd"]),
        "guide_slope_sd": float(cfg["guide_slope_sd"]),
        "offtarget_guide_frac": float(cfg["offtarget_guide_frac"]),
        "offtarget_slope_sd": float(cfg["offtarget_slope_sd"]),
        "nb_overdispersion": float(cfg["nb_overdispersion"]),
    }

    runtime = report.get("runtime_sec", {})
    row.update({f"runtime_{k}": float(v) for k, v in runtime.items()})

    qq = report.get("qq", {})

    counts_qc = report.get("counts_qc", {})
    mean_disp = counts_qc.get("mean_dispersion", {})
    depth_proxy = counts_qc.get("depth_proxy", {})
    row["counts_median_mean"] = mean_disp.get("median_mean")
    row["counts_median_phi_hat"] = mean_disp.get("median_phi_hat")
    row["counts_corr_log_mean_log_var"] = mean_disp.get("corr_log_mean_log_var")
    row["depth_log_libsize_sd"] = depth_proxy.get("log_libsize_sd")
    row["depth_corr_treatment_log_libsize"] = depth_proxy.get("corr_treatment_log_libsize")

    design = report.get("design_matrix", {})
    row["design_rank"] = design.get("rank")
    row["design_cond"] = design.get("cond")
    design_corr = design.get("corr", {})
    row["design_corr_treatment_log_libsize_centered"] = design_corr.get("treatment", {}).get("log_libsize_centered")

    het = report.get("heterogeneity", {})
    row["theta_dev_sd_mean"] = het.get("theta_dev_sd_mean")
    row["theta_dev_sd_median"] = het.get("theta_dev_sd_median")
    row["meta_tau_corr_true"] = het.get("meta_tau_corr_true")
    row["meta_tau_n"] = het.get("meta_tau_n")
    row["lmm_tau_corr_true"] = het.get("lmm_tau_corr_true")
    row["lmm_tau_n"] = het.get("lmm_tau_n")

    if "meta" in report:
        row["meta_null_mean_p"] = report["meta"]["null"]["mean"]
        row["meta_null_prop_lt_alpha"] = report["meta"]["null"]["prop_lt_alpha"]
        row["meta_null_lambda_gc"] = qq.get("meta_p_null", {}).get("lambda_gc")
        row["meta_null_ks"] = report["meta"].get("ks_uniform_null", {}).get("ks")
        row["meta_null_ks_p"] = report["meta"].get("ks_uniform_null", {}).get("ks_p")
        row["meta_null_ks_n"] = report["meta"].get("ks_uniform_null", {}).get("n")
        row["meta_roc_auc"] = report["meta"].get("roc_auc")
        row["meta_average_precision"] = report["meta"].get("average_precision")
        theta = report["meta"].get("theta_metrics", {})
        row["meta_theta_corr_all"] = theta.get("corr_all")
        row["meta_theta_rmse_all"] = theta.get("rmse_all")
        row["meta_theta_corr_signal"] = theta.get("corr_signal")
        row["meta_theta_rmse_signal"] = theta.get("rmse_signal")
        row["meta_theta_sign_acc_signal"] = theta.get("sign_acc_signal")
        row["meta_theta_n_all"] = theta.get("n_all")
        row["meta_theta_n_signal"] = theta.get("n_signal")
        bias = report["meta"].get("theta_bias_null", {})
        row["meta_theta_null_mean"] = bias.get("mean_null")
        row["meta_theta_null_median"] = bias.get("median_null")
        row["meta_theta_null_abs_mean"] = bias.get("mean_abs_null")
        row["meta_theta_null_n"] = bias.get("n_null")
        _add_confusion_metrics(row, out_prefix="meta_alpha", confusion=report["meta"].get("confusion_alpha"))
        _add_confusion_metrics(row, out_prefix="meta_q", confusion=report["meta"].get("confusion_fdr_q"))
    if "stouffer" in report:
        row["stouffer_null_mean_p"] = report["stouffer"]["null"]["mean"]
        row["stouffer_null_prop_lt_alpha"] = report["stouffer"]["null"]["prop_lt_alpha"]
        row["stouffer_null_lambda_gc"] = qq.get("stouffer_p_null", {}).get("lambda_gc")
        row["stouffer_null_ks"] = report["stouffer"].get("ks_uniform_null", {}).get("ks")
        row["stouffer_null_ks_p"] = report["stouffer"].get("ks_uniform_null", {}).get("ks_p")
        row["stouffer_null_ks_n"] = report["stouffer"].get("ks_uniform_null", {}).get("n")
        row["stouffer_roc_auc"] = report["stouffer"].get("roc_auc")
        row["stouffer_average_precision"] = report["stouffer"].get("average_precision")
        _add_confusion_metrics(row, out_prefix="stouffer_alpha", confusion=report["stouffer"].get("confusion_alpha"))
        _add_confusion_metrics(row, out_prefix="stouffer_q", confusion=report["stouffer"].get("confusion_fdr_q"))
    if "lmm_lrt" in report:
        row["lmm_lrt_null_mean_p"] = report["lmm_lrt"]["null"]["mean"]
        row["lmm_lrt_null_prop_lt_alpha"] = report["lmm_lrt"]["null"]["prop_lt_alpha"]
        row["lmm_lrt_ok_frac"] = report["lmm_lrt"]["lrt_ok_frac"]
        row["lmm_lrt_null_lambda_gc"] = qq.get("lmm_lrt_p_null", {}).get("lambda_gc")
        row["lmm_lrt_null_ks"] = report["lmm_lrt"].get("ks_uniform_null", {}).get("ks")
        row["lmm_lrt_null_ks_p"] = report["lmm_lrt"].get("ks_uniform_null", {}).get("ks_p")
        row["lmm_lrt_null_ks_n"] = report["lmm_lrt"].get("ks_uniform_null", {}).get("n")
        row["lmm_lrt_roc_auc"] = report["lmm_lrt"].get("roc_auc")
        row["lmm_lrt_average_precision"] = report["lmm_lrt"].get("average_precision")
        theta = report["lmm_lrt"].get("theta_metrics", {})
        row["lmm_lrt_theta_corr_all"] = theta.get("corr_all")
        row["lmm_lrt_theta_rmse_all"] = theta.get("rmse_all")
        row["lmm_lrt_theta_corr_signal"] = theta.get("corr_signal")
        row["lmm_lrt_theta_rmse_signal"] = theta.get("rmse_signal")
        row["lmm_lrt_theta_sign_acc_signal"] = theta.get("sign_acc_signal")
        row["lmm_lrt_theta_n_all"] = theta.get("n_all")
        row["lmm_lrt_theta_n_signal"] = theta.get("n_signal")
        bias = report["lmm_lrt"].get("theta_bias_null", {})
        row["lmm_lrt_theta_null_mean"] = bias.get("mean_null")
        row["lmm_lrt_theta_null_median"] = bias.get("median_null")
        row["lmm_lrt_theta_null_abs_mean"] = bias.get("mean_abs_null")
        row["lmm_lrt_theta_null_n"] = bias.get("n_null")
        _add_confusion_metrics(row, out_prefix="lmm_lrt_alpha", confusion=report["lmm_lrt"].get("confusion_alpha"))
        _add_confusion_metrics(row, out_prefix="lmm_lrt_q", confusion=report["lmm_lrt"].get("confusion_fdr_q"))
    if "lmm_wald" in report:
        row["lmm_wald_null_mean_p"] = report["lmm_wald"]["null"]["mean"]
        row["lmm_wald_null_prop_lt_alpha"] = report["lmm_wald"]["null"]["prop_lt_alpha"]
        row["lmm_wald_ok_frac"] = report["lmm_wald"]["wald_ok_frac"]
        row["lmm_wald_null_lambda_gc"] = qq.get("lmm_wald_p_null", {}).get("lambda_gc")
        row["lmm_wald_null_ks"] = report["lmm_wald"].get("ks_uniform_null", {}).get("ks")
        row["lmm_wald_null_ks_p"] = report["lmm_wald"].get("ks_uniform_null", {}).get("ks_p")
        row["lmm_wald_null_ks_n"] = report["lmm_wald"].get("ks_uniform_null", {}).get("n")
        row["lmm_wald_roc_auc"] = report["lmm_wald"].get("roc_auc")
        row["lmm_wald_average_precision"] = report["lmm_wald"].get("average_precision")
        theta = report["lmm_wald"].get("theta_metrics", {})
        row["lmm_wald_theta_corr_all"] = theta.get("corr_all")
        row["lmm_wald_theta_rmse_all"] = theta.get("rmse_all")
        row["lmm_wald_theta_corr_signal"] = theta.get("corr_signal")
        row["lmm_wald_theta_rmse_signal"] = theta.get("rmse_signal")
        row["lmm_wald_theta_sign_acc_signal"] = theta.get("sign_acc_signal")
        row["lmm_wald_theta_n_all"] = theta.get("n_all")
        row["lmm_wald_theta_n_signal"] = theta.get("n_signal")
        bias = report["lmm_wald"].get("theta_bias_null", {})
        row["lmm_wald_theta_null_mean"] = bias.get("mean_null")
        row["lmm_wald_theta_null_median"] = bias.get("median_null")
        row["lmm_wald_theta_null_abs_mean"] = bias.get("mean_abs_null")
        row["lmm_wald_theta_null_n"] = bias.get("n_null")
        _add_confusion_metrics(row, out_prefix="lmm_wald_alpha", confusion=report["lmm_wald"].get("confusion_alpha"))
        _add_confusion_metrics(row, out_prefix="lmm_wald_q", confusion=report["lmm_wald"].get("confusion_fdr_q"))

    lmm_fit = report.get("lmm_fit", {})
    if isinstance(lmm_fit, dict) and lmm_fit:
        row["lmm_n_total"] = lmm_fit.get("n_total")
        row["lmm_n_attempted"] = lmm_fit.get("n_attempted")
        row["lmm_frac_attempted"] = lmm_fit.get("frac_attempted")
        row["lmm_lrt_ok_frac_attempted"] = lmm_fit.get("lrt_ok_frac_attempted")
        row["lmm_wald_ok_frac_attempted"] = lmm_fit.get("wald_ok_frac_attempted")
        row["lmm_n_selected"] = lmm_fit.get("n_selected")

        fracs = lmm_fit.get("method_fracs", {}) if isinstance(lmm_fit.get("method_fracs", {}), dict) else {}
        row["lmm_frac_method_lmm"] = fracs.get("lmm")
        row["lmm_frac_method_meta_fallback"] = fracs.get("meta_fallback")
        row["lmm_frac_method_failed"] = fracs.get("failed")

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small count-depth benchmark grid and collect summaries (local).")
    parser.add_argument("--out-dir", required=True, type=str, help="Root output directory for the grid.")
    parser.add_argument("--jobs", type=int, default=1, help="Max concurrent runs to execute (default: 1).")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, skip runs with an existing benchmark_report.json (default: disabled).",
    )
    parser.add_argument("--progress-every", type=int, default=25, help="Progress print interval in completed runs (default: 25).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Seeds to run (default: 1 2 3).")
    parser.add_argument("--n-genes", type=int, nargs="+", default=[500], help="n_genes values to run (default: 500).")
    parser.add_argument("--guides-per-gene", type=int, nargs="+", default=[4], help="Guides per gene (default: 4).")
    parser.add_argument("--n-control", type=int, nargs="+", default=[12], help="Number of control samples (default: 12).")
    parser.add_argument("--n-treatment", type=int, nargs="+", default=[12], help="Number of treatment samples (default: 12).")
    parser.add_argument(
        "--guide-lambda-log-mean",
        type=float,
        default=5.298317366548036,
        help="log-mean baseline guide lambda (default: log(200)).",
    )
    parser.add_argument(
        "--guide-lambda-log-sd",
        type=float,
        nargs="+",
        default=[0.8],
        help="Guide log-sd on lambda within gene (default: 0.8).",
    )
    parser.add_argument(
        "--gene-lambda-log-sd",
        type=float,
        nargs="+",
        default=[0.5],
        help="Gene-level log-sd on lambda (default: 0.5).",
    )
    parser.add_argument(
        "--gene-lambda-family",
        type=str,
        nargs="+",
        choices=["lognormal", "mixture_lognormal", "power_law"],
        default=["lognormal"],
        help="Gene-level abundance family values to sweep (default: lognormal).",
    )
    parser.add_argument(
        "--gene-lambda-mix-pi-high",
        type=float,
        nargs="+",
        default=[0.10],
        help="Mixture lognormal: high-component fraction values to sweep (default: 0.10).",
    )
    parser.add_argument(
        "--gene-lambda-mix-delta-log-mean",
        type=float,
        nargs="+",
        default=[2.0],
        help="Mixture lognormal: delta log-mean separation values to sweep (default: 2.0).",
    )
    parser.add_argument(
        "--gene-lambda-power-alpha",
        type=float,
        nargs="+",
        default=[2.0],
        help="Power-law: Pareto alpha values to sweep (>1; default: 2.0).",
    )
    parser.add_argument(
        "--guide-lambda-family",
        type=str,
        nargs="+",
        choices=["lognormal_noise", "dirichlet_weights"],
        default=["lognormal_noise"],
        help="Within-gene guide abundance family values to sweep (default: lognormal_noise).",
    )
    parser.add_argument(
        "--guide-lambda-dirichlet-alpha0",
        type=float,
        nargs="+",
        default=[1.0],
        help="Dirichlet weights: alpha0 values to sweep (default: 1.0).",
    )
    parser.add_argument("--depth-log-sd", type=float, nargs="+", default=[1.0], help="Depth log-sd values (default: 1.0).")
    parser.add_argument("--n-batches", type=int, nargs="+", default=[1], help="Number of batches (supports 1 or 2; default: 1).")
    parser.add_argument(
        "--batch-confounding-strength",
        type=float,
        nargs="+",
        default=[0.0],
        help="Treatment↔batch confounding strengths in [0,1] (default: 0).",
    )
    parser.add_argument(
        "--batch-depth-log-sd",
        type=float,
        nargs="+",
        default=[0.0],
        help="Per-batch log-depth shift SD values (default: 0).",
    )
    parser.add_argument(
        "--treatment-depth-multiplier",
        type=float,
        nargs="+",
        default=[1.0, 2.0],
        help="Treatment depth multipliers (default: 1.0 2.0).",
    )
    parser.add_argument(
        "--include-depth-covariate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If set, run only with or without the depth covariate; default runs both.",
    )
    parser.add_argument(
        "--include-batch-covariate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If set, run only with or without batch indicators; default runs both.",
    )
    parser.add_argument(
        "--response-mode",
        type=str,
        choices=["log_counts", "guide_zscore_log_counts", "pmd_std_res"],
        default="log_counts",
        help="Response construction mode (default: log_counts).",
    )
    parser.add_argument(
        "--normalization-mode",
        type=str,
        nargs="+",
        choices=["none", "libsize_to_mean", "cpm", "median_ratio"],
        default=["none"],
        help="Normalization mode(s) to sweep for non-PMD response modes (default: none).",
    )
    parser.add_argument(
        "--logratio-mode",
        type=str,
        nargs="+",
        choices=["none", "clr_all", "alr_refset"],
        default=["none"],
        help="Log-ratio mode(s) to sweep for non-PMD response modes (default: none).",
    )
    parser.add_argument(
        "--n-reference-genes",
        type=int,
        default=0,
        help="Number of always-null reference genes (required for logratio-mode=alr_refset).",
    )
    parser.add_argument("--pmd-n-boot", type=int, default=100, help="PMD num_boot (only used for response-mode=pmd_std_res).")
    parser.add_argument(
        "--qq-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write QQ plot PNGs for each run (default: disabled; stats are still computed).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["meta", "stouffer", "lmm", "qc"],
        default=["meta", "stouffer", "qc"],
        help="Which gene-level methods to run (default: meta stouffer qc).",
    )
    parser.add_argument(
        "--lmm-scope",
        type=str,
        choices=["all", "meta_fdr", "meta_or_het_fdr", "none"],
        nargs="+",
        default=["all"],
        help="Plan A (LMM) gene selection policy values to sweep (default: all).",
    )
    parser.add_argument("--lmm-q-meta", type=float, nargs="+", default=[0.1], help="LMM selection q_meta values to sweep (default: 0.1).")
    parser.add_argument("--lmm-q-het", type=float, nargs="+", default=[0.1], help="LMM selection q_het values to sweep (default: 0.1).")
    parser.add_argument("--lmm-audit-n", type=int, nargs="+", default=[50], help="LMM selection audit_n values to sweep (default: 50).")
    parser.add_argument("--lmm-audit-seed", type=int, default=123456, help="LMM selection audit_seed (default: 123456).")
    parser.add_argument(
        "--lmm-max-genes-per-focal-var",
        type=int,
        nargs="+",
        default=[0],
        help="LMM selection cap values to sweep; use 0 for None/unlimited (default: 0).",
    )
    parser.add_argument(
        "--frac-signal",
        type=float,
        nargs="+",
        default=[0.0],
        help="Fraction of signal genes (default: 0; null calibration runs).",
    )
    parser.add_argument("--effect-sd", type=float, nargs="+", default=[0.5], help="Effect SD for signal genes (default: 0.5).")
    parser.add_argument("--guide-slope-sd", type=float, nargs="+", default=[0.2], help="Guide slope SD (signal genes only).")
    parser.add_argument("--offtarget-guide-frac", type=float, nargs="+", default=[0.0], help="Off-target guide fraction (default: 0).")
    parser.add_argument("--offtarget-slope-sd", type=float, nargs="+", default=[0.0], help="Off-target slope SD (default: 0).")
    parser.add_argument(
        "--nb-overdispersion",
        type=float,
        nargs="+",
        default=[0.0],
        help="NB overdispersion phi values (Var = mu + phi*mu^2; default: 0 => Poisson).",
    )
    parser.add_argument("--max-iter", type=int, default=200, help="Max iter for MixedLM (only used when methods include lmm).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    include_depth_opts = [False, True] if args.include_depth_covariate is None else [bool(args.include_depth_covariate)]
    include_batch_opts = [False, True] if args.include_batch_covariate is None else [bool(args.include_batch_covariate)]

    rows: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    n_resumed = 0
    for (
        seed,
        n_genes,
        guides_per_gene,
        n_control,
        n_treatment,
        depth_log_sd,
        n_batches,
        batch_strength,
        batch_depth_log_sd,
        tdm,
        include_depth,
        include_batch,
        lmm_scope,
        lmm_q_meta,
        lmm_q_het,
        lmm_audit_n,
        lmm_max_genes,
        frac_signal,
        effect_sd,
        guide_slope_sd,
        gene_lambda_family,
        gene_lambda_mix_pi_high,
        gene_lambda_mix_delta_log_mean,
        gene_lambda_power_alpha,
        gene_lambda_log_sd,
        guide_lambda_log_sd,
        guide_lambda_family,
        guide_lambda_dirichlet_alpha0,
        ot_frac,
        ot_sd,
        nb_overdispersion,
        normalization_mode,
        logratio_mode,
    ) in product(
        [int(s) for s in args.seeds],
        [int(n) for n in args.n_genes],
        [int(x) for x in args.guides_per_gene],
        [int(x) for x in args.n_control],
        [int(x) for x in args.n_treatment],
        [float(x) for x in args.depth_log_sd],
        [int(x) for x in args.n_batches],
        [float(x) for x in args.batch_confounding_strength],
        [float(x) for x in args.batch_depth_log_sd],
        [float(x) for x in args.treatment_depth_multiplier],
        include_depth_opts,
        include_batch_opts,
        [str(x) for x in args.lmm_scope],
        [float(x) for x in args.lmm_q_meta],
        [float(x) for x in args.lmm_q_het],
        [int(x) for x in args.lmm_audit_n],
        [int(x) for x in args.lmm_max_genes_per_focal_var],
        [float(x) for x in args.frac_signal],
        [float(x) for x in args.effect_sd],
        [float(x) for x in args.guide_slope_sd],
        [str(x) for x in args.gene_lambda_family],
        [float(x) for x in args.gene_lambda_mix_pi_high],
        [float(x) for x in args.gene_lambda_mix_delta_log_mean],
        [float(x) for x in args.gene_lambda_power_alpha],
        [float(x) for x in args.gene_lambda_log_sd],
        [float(x) for x in args.guide_lambda_log_sd],
        [str(x) for x in args.guide_lambda_family],
        [float(x) for x in args.guide_lambda_dirichlet_alpha0],
        [float(x) for x in args.offtarget_guide_frac],
        [float(x) for x in args.offtarget_slope_sd],
        [float(x) for x in args.nb_overdispersion],
        [str(x) for x in args.normalization_mode],
        [str(x) for x in args.logratio_mode],
    ):
        lmm_max_genes_opt = None if int(lmm_max_genes) == 0 else int(lmm_max_genes)
        cap_tag = 0 if lmm_max_genes_opt is None else int(lmm_max_genes_opt)
        depth_tag = "logls" if bool(include_depth) else "none"
        batch_tag = int(bool(include_batch))

        full_cfg: dict[str, object] = {
            "seed": seed,
            "response_mode": str(args.response_mode),
            "pmd_n_boot": int(args.pmd_n_boot),
            "normalization_mode": str(normalization_mode),
            "logratio_mode": str(logratio_mode),
            "n_reference_genes": int(args.n_reference_genes),
            "n_genes": n_genes,
            "guides_per_gene": int(guides_per_gene),
            "n_control": int(n_control),
            "n_treatment": int(n_treatment),
            "guide_lambda_log_mean": float(args.guide_lambda_log_mean),
            "guide_lambda_log_sd": float(guide_lambda_log_sd),
            "gene_lambda_log_sd": float(gene_lambda_log_sd),
            "gene_lambda_family": str(gene_lambda_family),
            "gene_lambda_mix_pi_high": float(gene_lambda_mix_pi_high),
            "gene_lambda_mix_delta_log_mean": float(gene_lambda_mix_delta_log_mean),
            "gene_lambda_power_alpha": float(gene_lambda_power_alpha),
            "guide_lambda_family": str(guide_lambda_family),
            "guide_lambda_dirichlet_alpha0": float(guide_lambda_dirichlet_alpha0),
            "depth_log_sd": depth_log_sd,
            "n_batches": int(n_batches),
            "batch_confounding_strength": float(batch_strength),
            "batch_depth_log_sd": float(batch_depth_log_sd),
            "treatment_depth_multiplier": tdm,
            "include_depth_covariate": bool(include_depth),
            "depth_covariate_mode": "log_libsize" if bool(include_depth) else "none",
            "include_batch_covariate": bool(include_batch),
            "lmm_scope": str(lmm_scope),
            "lmm_q_meta": float(lmm_q_meta),
            "lmm_q_het": float(lmm_q_het),
            "lmm_audit_n": int(lmm_audit_n),
            "lmm_audit_seed": int(args.lmm_audit_seed),
            "lmm_max_genes_per_focal_var": lmm_max_genes_opt,
            "frac_signal": float(frac_signal),
            "effect_sd": float(effect_sd),
            "guide_slope_sd": float(guide_slope_sd),
            "offtarget_guide_frac": float(ot_frac),
            "offtarget_slope_sd": float(ot_sd),
            "nb_overdispersion": float(nb_overdispersion),
            "alpha": 0.05,
            "fdr_q": 0.1,
        }
        run_hash = _stable_hash(full_cfg)

        # Keep the directory name short to avoid filesystem path limits.
        glf_map = {"lognormal": "ln", "mixture_lognormal": "mln", "power_law": "pl"}
        guf_map = {"lognormal_noise": "lnn", "dirichlet_weights": "dir"}
        glf_tag = f"__glf={glf_map.get(str(gene_lambda_family), str(gene_lambda_family))}"
        guf_tag = f"__guf={guf_map.get(str(guide_lambda_family), str(guide_lambda_family))}"
        glf_param_tag = ""
        if str(gene_lambda_family) == "mixture_lognormal":
            glf_param_tag = f"__gpi={float(gene_lambda_mix_pi_high):g}__gdl={float(gene_lambda_mix_delta_log_mean):g}"
        if str(gene_lambda_family) == "power_law":
            glf_param_tag = f"__gpa={float(gene_lambda_power_alpha):g}"
        guf_param_tag = f"__ga0={float(guide_lambda_dirichlet_alpha0):g}" if str(guide_lambda_family) == "dirichlet_weights" else ""
        tag = (
            f"s={seed}"
            f"__rm={args.response_mode}"
            f"__norm={normalization_mode}"
            f"__lr={logratio_mode}"
            f"__dc={depth_tag}"
            f"__bc={batch_tag}"
            f"__ng={n_genes}"
            f"__gpg={int(guides_per_gene)}"
            f"__ns={int(n_control)+int(n_treatment)}"
            f"__tdm={tdm}"
            f"__fs={frac_signal}"
            f"{glf_tag}{glf_param_tag}{guf_tag}{guf_param_tag}"
            f"__lmm={lmm_scope}"
            f"__cap={cap_tag}"
            f"__h={run_hash}"
        )
        run_dir = os.path.join(args.out_dir, tag)
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "benchmark_count_depth.py"),
            "--out-dir",
            run_dir,
            "--n-genes",
            str(n_genes),
            "--guides-per-gene",
            str(int(guides_per_gene)),
            "--n-control",
            str(int(n_control)),
            "--n-treatment",
            str(int(n_treatment)),
            "--guide-lambda-log-mean",
            str(float(args.guide_lambda_log_mean)),
            "--guide-lambda-log-sd",
            str(float(guide_lambda_log_sd)),
            "--gene-lambda-log-sd",
            str(float(gene_lambda_log_sd)),
            "--gene-lambda-family",
            str(gene_lambda_family),
            "--gene-lambda-mix-pi-high",
            str(float(gene_lambda_mix_pi_high)),
            "--gene-lambda-mix-delta-log-mean",
            str(float(gene_lambda_mix_delta_log_mean)),
            "--gene-lambda-power-alpha",
            str(float(gene_lambda_power_alpha)),
            "--guide-lambda-family",
            str(guide_lambda_family),
            "--guide-lambda-dirichlet-alpha0",
            str(float(guide_lambda_dirichlet_alpha0)),
            "--depth-log-sd",
            str(depth_log_sd),
            "--n-batches",
            str(int(n_batches)),
            "--batch-confounding-strength",
            str(float(batch_strength)),
            "--batch-depth-log-sd",
            str(float(batch_depth_log_sd)),
            "--treatment-depth-multiplier",
            str(tdm),
            "--depth-covariate-mode",
            ("log_libsize" if bool(include_depth) else "none"),
            "--seed",
            str(seed),
            "--frac-signal",
            str(float(frac_signal)),
            "--effect-sd",
            str(float(effect_sd)),
            "--guide-slope-sd",
            str(float(guide_slope_sd)),
            "--offtarget-guide-frac",
            str(float(ot_frac)),
            "--offtarget-slope-sd",
            str(float(ot_sd)),
            "--response-mode",
            str(args.response_mode),
            "--normalization-mode",
            str(normalization_mode),
            "--logratio-mode",
            str(logratio_mode),
            "--n-reference-genes",
            str(int(args.n_reference_genes)),
            "--pmd-n-boot",
            str(int(args.pmd_n_boot)),
            "--nb-overdispersion",
            str(float(nb_overdispersion)),
            "--methods",
            *[str(m) for m in args.methods],
            "--max-iter",
            str(int(args.max_iter)),
            "--lmm-scope",
            str(lmm_scope),
            "--lmm-q-meta",
            str(float(lmm_q_meta)),
            "--lmm-q-het",
            str(float(lmm_q_het)),
            "--lmm-audit-n",
            str(int(lmm_audit_n)),
            "--lmm-audit-seed",
            str(int(args.lmm_audit_seed)),
        ]
        if lmm_max_genes_opt is not None:
            cmd += ["--lmm-max-genes-per-focal-var", str(int(lmm_max_genes_opt))]
        cmd.append("--include-batch-covariate" if include_batch else "--no-include-batch-covariate")
        cmd.append("--qq-plots" if bool(args.qq_plots) else "--no-qq-plots")

        existing_report_path = _report_path_for_run_dir(run_dir)
        if bool(args.resume) and os.path.isfile(existing_report_path):
            report = _load_json(existing_report_path)
            rows.append(_row_from_report(tag=tag, report_path=existing_report_path, report=report))
            n_resumed += 1
        else:
            tasks.append({"tag": tag, "cmd": cmd, "run_dir": run_dir})

    n_total = int(n_resumed + len(tasks))
    if n_total == 0:
        raise ValueError("no runs configured (empty grid)")

    jobs = int(args.jobs)
    if jobs < 1:
        raise ValueError("--jobs must be >= 1")

    print(
        f"grid: total={n_total} resumed={n_resumed} to_run={len(tasks)} jobs={jobs} resume={bool(args.resume)}",
        flush=True,
    )

    t0 = time.monotonic()
    n_done_from_tasks = 0
    if tasks:
        if jobs == 1:
            for task in tasks:
                report_path = _run_one(task["cmd"])
                report = _load_json(report_path)
                rows.append(_row_from_report(tag=str(task["tag"]), report_path=report_path, report=report))
                n_done_from_tasks += 1
                done = int(n_resumed + n_done_from_tasks)
                if int(args.progress_every) > 0 and (done % int(args.progress_every) == 0 or done == n_total):
                    elapsed = max(1e-9, time.monotonic() - t0)
                    rate = done / elapsed
                    eta_s = (n_total - done) / max(1e-9, rate)
                    print(f"grid: {done}/{n_total} done (eta~{eta_s/60.0:.1f} min)", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                fut_to_task = {ex.submit(_run_one, t["cmd"]): t for t in tasks}
                for fut in as_completed(fut_to_task):
                    task = fut_to_task[fut]
                    report_path = fut.result()
                    report = _load_json(report_path)
                    rows.append(_row_from_report(tag=str(task["tag"]), report_path=report_path, report=report))
                    n_done_from_tasks += 1
                    done = int(n_resumed + n_done_from_tasks)
                    if int(args.progress_every) > 0 and (done % int(args.progress_every) == 0 or done == n_total):
                        elapsed = max(1e-9, time.monotonic() - t0)
                        rate = done / elapsed
                        eta_s = (n_total - done) / max(1e-9, rate)
                        print(f"grid: {done}/{n_total} done (eta~{eta_s/60.0:.1f} min)", flush=True)

    sort_cols = [
        "response_mode",
        "pmd_n_boot",
        "normalization_mode",
        "logratio_mode",
        "n_reference_genes",
        "n_genes",
        "depth_log_sd",
        "guides_per_gene",
        "n_control",
        "n_treatment",
        "n_samples",
        "n_batches",
        "batch_confounding_strength",
        "batch_depth_log_sd",
        "treatment_depth_multiplier",
        "depth_covariate_mode",
        "include_depth_covariate",
        "include_batch_covariate",
        "lmm_scope",
        "lmm_q_meta",
        "lmm_q_het",
        "lmm_audit_n",
        "lmm_audit_seed",
        "frac_signal",
        "effect_sd",
        "guide_slope_sd",
        "gene_lambda_log_sd",
        "gene_lambda_family",
        "gene_lambda_mix_pi_high",
        "gene_lambda_mix_delta_log_mean",
        "gene_lambda_power_alpha",
        "guide_lambda_log_sd",
        "guide_lambda_family",
        "guide_lambda_dirichlet_alpha0",
        "offtarget_guide_frac",
        "offtarget_slope_sd",
        "nb_overdispersion",
        "seed",
    ]
    df = pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)
    out_path = os.path.join(args.out_dir, "count_depth_grid_summary.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    print(out_path)


if __name__ == "__main__":
    main()
