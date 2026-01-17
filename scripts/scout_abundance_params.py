from __future__ import annotations

import argparse
import itertools
import math

import numpy as np
import pandas as pd

from benchmark_count_depth import CountDepthBenchmarkConfig
from benchmark_count_depth import _compute_abundance_audit
from benchmark_count_depth import simulate_counts_and_std_res


def _get_nested(d: dict[str, object], path: list[str], default: float | None = None) -> float | None:
    cur: object = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    if cur is None:
        return default
    try:
        v = float(cur)  # type: ignore[arg-type]
    except Exception:
        return default
    return v if np.isfinite(v) else default


def _safe_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    if b <= 0:
        return None
    return float(a / b)


def _log10_ratio(a: float | None, b: float | None) -> float | None:
    r = _safe_ratio(a, b)
    if r is None or r <= 0:
        return None
    return float(math.log10(r))


def _scenario_grid() -> list[dict[str, object]]:
    """
    Small, intentionally adversarial grid to characterize dropout + dynamic range.

    This is not a performance benchmark; it's a parameter-scout to choose realistic stress-test defaults.
    """

    base = {
        "guide_lambda_log_mean": float(np.log(200.0)),
        "depth_log_sd": 1.0,
    }

    scenarios: list[dict[str, object]] = []

    # Baseline (close to current defaults).
    scenarios.append(
        {
            **base,
            "name": "baseline_lognormal",
            "gene_lambda_family": "lognormal",
            "gene_lambda_log_sd": 0.5,
            "guide_lambda_family": "lognormal_noise",
            "guide_lambda_log_sd": 0.8,
        }
    )

    # Dropout-heavy mixtures: many rare genes + a few dominant ones.
    for delta in [4.0, 5.0, 6.0]:
        scenarios.append(
            {
                **base,
                "name": f"mix_rare_d{delta:g}",
                "gene_lambda_family": "mixture_lognormal",
                "gene_lambda_log_sd": 0.6,
                "gene_lambda_mix_pi_high": 0.05,
                "gene_lambda_mix_delta_log_mean": float(delta),
                "guide_lambda_family": "lognormal_noise",
                "guide_lambda_log_sd": 1.0,
            }
        )

    # Many-dominant mixtures: most genes abundant; some rare.
    for pi_high in [0.5, 0.8]:
        scenarios.append(
            {
                **base,
                "name": f"mix_many_dominant_pi{pi_high:g}",
                "gene_lambda_family": "mixture_lognormal",
                "gene_lambda_log_sd": 0.6,
                "gene_lambda_mix_pi_high": float(pi_high),
                "gene_lambda_mix_delta_log_mean": 4.0,
                "guide_lambda_family": "lognormal_noise",
                "guide_lambda_log_sd": 1.0,
            }
        )

    # Power-law high-tail dominance (no extremely small values at the gene level; dropout can arise within gene).
    for alpha in [1.3, 1.6, 2.0]:
        scenarios.append(
            {
                **base,
                "name": f"powerlaw_a{alpha:g}",
                "gene_lambda_family": "power_law",
                "gene_lambda_power_alpha": float(alpha),
                "guide_lambda_family": "lognormal_noise",
                "guide_lambda_log_sd": 1.6,
            }
        )

    # Within-gene dominance via Dirichlet weights (creates within-gene dropout even when gene totals are moderate).
    for alpha0 in [0.1, 0.2, 0.5, 1.0]:
        scenarios.append(
            {
                **base,
                "name": f"dirichlet_a0{alpha0:g}",
                "gene_lambda_family": "lognormal",
                "gene_lambda_log_sd": 0.6,
                "guide_lambda_family": "dirichlet_weights",
                "guide_lambda_dirichlet_alpha0": float(alpha0),
            }
        )

    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Scout abundance-model parameters (dropout + dynamic range).")
    parser.add_argument("--out-tsv", type=str, default=None, help="Optional path to write a TSV summary.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Seeds to evaluate (default: 1 2 3).")
    parser.add_argument("--n-genes", type=int, default=500, help="Number of target genes (default: 500).")
    parser.add_argument("--guides-per-gene", type=int, default=4, help="Guides per gene (default: 4).")
    parser.add_argument("--n-control", type=int, default=12, help="Number of control samples (default: 12).")
    parser.add_argument("--n-treatment", type=int, default=12, help="Number of treatment samples (default: 12).")
    parser.add_argument("--depth-log-sd", type=float, default=1.0, help="Depth log-SD (default: 1.0).")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for seed, scenario in itertools.product([int(s) for s in args.seeds], _scenario_grid()):
        cfg = CountDepthBenchmarkConfig(
            n_genes=int(args.n_genes),
            guides_per_gene=int(args.guides_per_gene),
            n_control=int(args.n_control),
            n_treatment=int(args.n_treatment),
            guide_lambda_log_mean=float(scenario.get("guide_lambda_log_mean", np.log(200.0))),
            guide_lambda_log_sd=float(scenario.get("guide_lambda_log_sd", 0.8)),
            gene_lambda_log_sd=float(scenario.get("gene_lambda_log_sd", 0.5)),
            gene_lambda_family=str(scenario.get("gene_lambda_family", "lognormal")),
            gene_lambda_mix_pi_high=float(scenario.get("gene_lambda_mix_pi_high", 0.10)),
            gene_lambda_mix_delta_log_mean=float(scenario.get("gene_lambda_mix_delta_log_mean", 2.0)),
            gene_lambda_power_alpha=float(scenario.get("gene_lambda_power_alpha", 2.0)),
            guide_lambda_family=str(scenario.get("guide_lambda_family", "lognormal_noise")),
            guide_lambda_dirichlet_alpha0=float(scenario.get("guide_lambda_dirichlet_alpha0", 1.0)),
            depth_log_mean=0.0,
            depth_log_sd=float(args.depth_log_sd),
            depth_poisson_scale=0.0,
            n_batches=1,
            batch_confounding_strength=0.0,
            batch_depth_log_sd=0.0,
            treatment_depth_multiplier=1.0,
            frac_signal=0.0,
            effect_sd=0.5,
            guide_slope_sd=0.0,
            offtarget_guide_frac=0.0,
            offtarget_slope_sd=0.0,
            pseudocount=0.5,
            response_mode="log_counts",
            normalization_mode="none",
            logratio_mode="none",
            n_reference_genes=0,
            pmd_n_boot=100,
            pmd_seed=int(seed),
            nb_overdispersion=0.0,
            depth_covariate_mode="none",
            include_depth_covariate=False,
            include_batch_covariate=False,
            allow_random_slope=True,
            min_guides_random_slope=3,
            max_iter=50,
            lmm_scope="none",
            lmm_q_meta=0.1,
            lmm_q_het=0.1,
            lmm_audit_n=0,
            lmm_audit_seed=123456,
            lmm_max_genes_per_focal_var=None,
            methods=("qc",),
            seed=int(seed),
            alpha=0.05,
            fdr_q=0.1,
            qq_plots=False,
        )

        counts_df, _, _, _, _, truth_gene, truth_guide = simulate_counts_and_std_res(cfg)
        audit = _compute_abundance_audit(truth_gene, truth_guide, counts_df)

        guide_p01 = _get_nested(audit, ["guide_lambda", "p01"])
        guide_p99 = _get_nested(audit, ["guide_lambda", "p99"])
        gene_total_p01 = _get_nested(audit, ["gene_total_lambda", "p01"])
        gene_total_p99 = _get_nested(audit, ["gene_total_lambda", "p99"])

        row: dict[str, object] = {
            "name": str(scenario.get("name", "")),
            "seed": int(seed),
            "gene_lambda_family": cfg.gene_lambda_family,
            "guide_lambda_family": cfg.guide_lambda_family,
            "guide_lambda_log_mean": float(cfg.guide_lambda_log_mean),
            "gene_lambda_log_sd": float(cfg.gene_lambda_log_sd),
            "guide_lambda_log_sd": float(cfg.guide_lambda_log_sd),
            "gene_lambda_mix_pi_high": float(cfg.gene_lambda_mix_pi_high),
            "gene_lambda_mix_delta_log_mean": float(cfg.gene_lambda_mix_delta_log_mean),
            "gene_lambda_power_alpha": float(cfg.gene_lambda_power_alpha),
            "guide_lambda_dirichlet_alpha0": float(cfg.guide_lambda_dirichlet_alpha0),
            "depth_log_sd": float(cfg.depth_log_sd),
            # Dropout proxies.
            "counts_zero_frac_overall": _get_nested(audit, ["counts_zero_frac", "overall"]),
            "counts_zero_frac_per_sample_mean": _get_nested(audit, ["counts_zero_frac", "per_sample_mean"]),
            "guide_frac_lambda_lt_1": _get_nested(audit, ["guide_lambda", "frac_lt_1"]),
            # Dynamic range (orders of magnitude).
            "guide_lambda_log10_p99_over_p01": _log10_ratio(guide_p99, guide_p01),
            "gene_total_lambda_log10_p99_over_p01": _log10_ratio(gene_total_p99, gene_total_p01),
            # Dominance / compositional skew.
            "gene_total_lambda_gini": _get_nested(audit, ["gene_total_lambda", "gini"]),
            "gene_total_lambda_top_1pct_share": _get_nested(audit, ["gene_total_lambda", "top_1pct_share"]),
            "guide_total_lambda_top_1pct_share": _get_nested(audit, ["guide_total_lambda", "top_1pct_share"]),
            # Within-gene dominance summaries.
            "within_gene_max_over_mean_p50": _get_nested(audit, ["within_gene", "max_over_mean", "p50"]),
            "within_gene_max_over_mean_p90": _get_nested(audit, ["within_gene", "max_over_mean", "p90"]),
            "within_gene_log_lambda_sd_p50": _get_nested(audit, ["within_gene", "log_lambda_sd", "p50"]),
            "within_gene_log_lambda_sd_p90": _get_nested(audit, ["within_gene", "log_lambda_sd", "p90"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["name", "seed"], kind="mergesort").reset_index(drop=True)
    out_tsv = args.out_tsv
    if out_tsv is not None:
        df.to_csv(out_tsv, sep="\t", index=False)
        print(out_tsv)
    else:
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
            print(df)


if __name__ == "__main__":
    main()

