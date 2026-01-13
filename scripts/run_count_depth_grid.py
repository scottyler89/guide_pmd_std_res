from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from itertools import product

import pandas as pd


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small count-depth benchmark grid and collect summaries (local).")
    parser.add_argument("--out-dir", required=True, type=str, help="Root output directory for the grid.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Seeds to run (default: 1 2 3).")
    parser.add_argument("--n-genes", type=int, nargs="+", default=[500], help="n_genes values to run (default: 500).")
    parser.add_argument("--guides-per-gene", type=int, default=4, help="Guides per gene.")
    parser.add_argument("--n-control", type=int, default=12, help="Number of control samples.")
    parser.add_argument("--n-treatment", type=int, default=12, help="Number of treatment samples.")
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
    parser.add_argument("--pmd-n-boot", type=int, default=100, help="PMD num_boot (only used for response-mode=pmd_std_res).")
    parser.add_argument(
        "--qq-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write QQ plot PNGs for each run (default: disabled; stats are still computed).",
    )
    parser.add_argument("--methods", type=str, nargs="+", choices=["meta", "lmm", "qc"], default=["meta", "qc"])
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
    for (
        seed,
        n_genes,
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
        gene_lambda_log_sd,
        guide_lambda_log_sd,
        ot_frac,
        ot_sd,
        nb_overdispersion,
    ) in product(
        [int(s) for s in args.seeds],
        [int(n) for n in args.n_genes],
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
        [float(x) for x in args.gene_lambda_log_sd],
        [float(x) for x in args.guide_lambda_log_sd],
        [float(x) for x in args.offtarget_guide_frac],
        [float(x) for x in args.offtarget_slope_sd],
        [float(x) for x in args.nb_overdispersion],
    ):
        lmm_max_genes_opt = None if int(lmm_max_genes) == 0 else int(lmm_max_genes)
        cap_tag = 0 if lmm_max_genes_opt is None else int(lmm_max_genes_opt)

        full_cfg = {
            "seed": seed,
            "response_mode": str(args.response_mode),
            "pmd_n_boot": int(args.pmd_n_boot),
            "n_genes": n_genes,
            "guides_per_gene": int(args.guides_per_gene),
            "n_control": int(args.n_control),
            "n_treatment": int(args.n_treatment),
            "guide_lambda_log_mean": float(args.guide_lambda_log_mean),
            "guide_lambda_log_sd": float(guide_lambda_log_sd),
            "gene_lambda_log_sd": float(gene_lambda_log_sd),
            "depth_log_sd": depth_log_sd,
            "n_batches": int(n_batches),
            "batch_confounding_strength": float(batch_strength),
            "batch_depth_log_sd": float(batch_depth_log_sd),
            "treatment_depth_multiplier": tdm,
            "include_depth_covariate": bool(include_depth),
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
        tag = (
            f"s={seed}"
            f"__rm={args.response_mode}"
            f"__ng={n_genes}"
            f"__tdm={tdm}"
            f"__fs={frac_signal}"
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
            str(int(args.guides_per_gene)),
            "--n-control",
            str(int(args.n_control)),
            "--n-treatment",
            str(int(args.n_treatment)),
            "--guide-lambda-log-mean",
            str(float(args.guide_lambda_log_mean)),
            "--guide-lambda-log-sd",
            str(float(guide_lambda_log_sd)),
            "--gene-lambda-log-sd",
            str(float(gene_lambda_log_sd)),
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
        if include_depth:
            cmd.append("--include-depth-covariate")
        cmd.append("--include-batch-covariate" if include_batch else "--no-include-batch-covariate")
        cmd.append("--qq-plots" if bool(args.qq_plots) else "--no-qq-plots")

        report_path = _run_one(cmd)
        report = _load_json(report_path)

        row: dict[str, object] = {
            "tag": tag,
            "report_path": report_path,
            "seed": seed,
            "pmd_n_boot": int(report["config"]["pmd_n_boot"]),
            "qq_plots": bool(report["config"]["qq_plots"]),
            "alpha": float(report["config"]["alpha"]),
            "fdr_q": float(report["config"]["fdr_q"]),
            "n_genes": n_genes,
            "guides_per_gene": int(report["config"]["guides_per_gene"]),
            "depth_log_sd": depth_log_sd,
            "n_batches": int(report["config"]["n_batches"]),
            "batch_confounding_strength": float(report["config"]["batch_confounding_strength"]),
            "batch_depth_log_sd": float(report["config"]["batch_depth_log_sd"]),
            "treatment_depth_multiplier": tdm,
            "include_depth_covariate": include_depth,
            "include_batch_covariate": include_batch,
            "response_mode": report["config"]["response_mode"],
            "guide_lambda_log_mean": float(report["config"]["guide_lambda_log_mean"]),
            "guide_lambda_log_sd": float(report["config"]["guide_lambda_log_sd"]),
            "gene_lambda_log_sd": float(report["config"]["gene_lambda_log_sd"]),
            "methods": ",".join(report["config"]["methods"]),
            "lmm_scope": str(report["config"]["lmm_scope"]),
            "lmm_q_meta": float(report["config"]["lmm_q_meta"]),
            "lmm_q_het": float(report["config"]["lmm_q_het"]),
            "lmm_audit_n": int(report["config"]["lmm_audit_n"]),
            "lmm_audit_seed": int(report["config"]["lmm_audit_seed"]),
            "lmm_max_genes_per_focal_var": report["config"]["lmm_max_genes_per_focal_var"],
            "frac_signal": float(report["config"]["frac_signal"]),
            "effect_sd": float(report["config"]["effect_sd"]),
            "guide_slope_sd": float(report["config"]["guide_slope_sd"]),
            "offtarget_guide_frac": float(report["config"]["offtarget_guide_frac"]),
            "offtarget_slope_sd": float(report["config"]["offtarget_slope_sd"]),
            "nb_overdispersion": float(report["config"]["nb_overdispersion"]),
        }

        runtime = report.get("runtime_sec", {})
        row.update({f"runtime_{k}": float(v) for k, v in runtime.items()})

        qq = report.get("qq", {})

        if "meta" in report:
            row["meta_null_mean_p"] = report["meta"]["null"]["mean"]
            row["meta_null_prop_lt_alpha"] = report["meta"]["null"]["prop_lt_alpha"]
            row["meta_null_lambda_gc"] = qq.get("meta_p_null", {}).get("lambda_gc")
            row["meta_alpha_fp"] = report["meta"]["confusion_alpha"]["fp"]
            row["meta_alpha_fpr"] = report["meta"]["confusion_alpha"]["fpr"]
            row["meta_alpha_tpr"] = report["meta"]["confusion_alpha"]["tpr"]
            row["meta_alpha_fdr"] = report["meta"]["confusion_alpha"]["fdr"]
            row["meta_alpha_n_called"] = report["meta"]["confusion_alpha"]["n_called"]
            row["meta_q_fp"] = report["meta"]["confusion_fdr_q"]["fp"]
            row["meta_q_fdr"] = report["meta"]["confusion_fdr_q"]["fdr"]
            row["meta_q_tpr"] = report["meta"]["confusion_fdr_q"]["tpr"]
            row["meta_q_n_called"] = report["meta"]["confusion_fdr_q"]["n_called"]
        if "lmm_lrt" in report:
            row["lmm_lrt_null_mean_p"] = report["lmm_lrt"]["null"]["mean"]
            row["lmm_lrt_null_prop_lt_alpha"] = report["lmm_lrt"]["null"]["prop_lt_alpha"]
            row["lmm_lrt_ok_frac"] = report["lmm_lrt"]["lrt_ok_frac"]
            row["lmm_lrt_null_lambda_gc"] = qq.get("lmm_lrt_p_null", {}).get("lambda_gc")
            row["lmm_lrt_alpha_fp"] = report["lmm_lrt"]["confusion_alpha"]["fp"]
            row["lmm_lrt_alpha_fpr"] = report["lmm_lrt"]["confusion_alpha"]["fpr"]
            row["lmm_lrt_alpha_tpr"] = report["lmm_lrt"]["confusion_alpha"]["tpr"]
            row["lmm_lrt_alpha_fdr"] = report["lmm_lrt"]["confusion_alpha"]["fdr"]
            row["lmm_lrt_q_fp"] = report["lmm_lrt"]["confusion_fdr_q"]["fp"]
            row["lmm_lrt_q_fdr"] = report["lmm_lrt"]["confusion_fdr_q"]["fdr"]
            row["lmm_lrt_q_tpr"] = report["lmm_lrt"]["confusion_fdr_q"]["tpr"]
        if "lmm_wald" in report:
            row["lmm_wald_null_mean_p"] = report["lmm_wald"]["null"]["mean"]
            row["lmm_wald_null_prop_lt_alpha"] = report["lmm_wald"]["null"]["prop_lt_alpha"]
            row["lmm_wald_ok_frac"] = report["lmm_wald"]["wald_ok_frac"]
            row["lmm_wald_null_lambda_gc"] = qq.get("lmm_wald_p_null", {}).get("lambda_gc")
            row["lmm_wald_alpha_fp"] = report["lmm_wald"]["confusion_alpha"]["fp"]
            row["lmm_wald_alpha_fpr"] = report["lmm_wald"]["confusion_alpha"]["fpr"]
            row["lmm_wald_alpha_tpr"] = report["lmm_wald"]["confusion_alpha"]["tpr"]
            row["lmm_wald_alpha_fdr"] = report["lmm_wald"]["confusion_alpha"]["fdr"]
            row["lmm_wald_q_fp"] = report["lmm_wald"]["confusion_fdr_q"]["fp"]
            row["lmm_wald_q_fdr"] = report["lmm_wald"]["confusion_fdr_q"]["fdr"]
            row["lmm_wald_q_tpr"] = report["lmm_wald"]["confusion_fdr_q"]["tpr"]

        rows.append(row)

    sort_cols = [
        "response_mode",
        "pmd_n_boot",
        "n_genes",
        "depth_log_sd",
        "n_batches",
        "batch_confounding_strength",
        "batch_depth_log_sd",
        "treatment_depth_multiplier",
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
        "guide_lambda_log_sd",
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
