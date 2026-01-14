from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _subset_by_focal(df: pd.DataFrame, focal_var: str) -> pd.DataFrame:
    if "focal_var" not in df.columns:
        raise ValueError("expected column: focal_var")
    return df.loc[df["focal_var"].astype(str) == str(focal_var)].copy()


def _finite_series(x: pd.Series) -> np.ndarray:
    arr = x.to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _safe_median(x: pd.Series) -> float:
    arr = _finite_series(x)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _safe_quantile(x: pd.Series, q: float) -> float:
    arr = _finite_series(x)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def triage_gene_level_disagreements(
    gene_level_dir: str,
    *,
    q: float,
    top_n: int,
    min_abs_theta_diff: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_level_dir = str(gene_level_dir)

    meta_path = os.path.join(gene_level_dir, "PMD_std_res_gene_meta.tsv")
    lmm_path = os.path.join(gene_level_dir, "PMD_std_res_gene_lmm.tsv")
    qc_path = os.path.join(gene_level_dir, "PMD_std_res_gene_qc.tsv")

    meta_df = _read_tsv(meta_path)
    lmm_df = _read_tsv(lmm_path)
    qc_df = _read_tsv(qc_path)

    for required in ("gene_id", "focal_var"):
        if required not in meta_df.columns:
            raise ValueError(f"meta missing required column: {required}")
        if required not in lmm_df.columns:
            raise ValueError(f"lmm missing required column: {required}")
        if required not in qc_df.columns:
            raise ValueError(f"qc missing required column: {required}")

    if "p_adj" not in meta_df.columns:
        raise ValueError("meta missing required column: p_adj")
    if "theta" not in meta_df.columns:
        raise ValueError("meta missing required column: theta")
    if "theta" not in lmm_df.columns:
        raise ValueError("lmm missing required column: theta")
    if "method" not in lmm_df.columns:
        raise ValueError("lmm missing required column: method")

    if "lrt_p_adj" not in lmm_df.columns:
        raise ValueError("lmm missing required column: lrt_p_adj")
    if "wald_p_adj" not in lmm_df.columns:
        raise ValueError("lmm missing required column: wald_p_adj")

    focal_vars = sorted(set(meta_df["focal_var"].astype(str).unique().tolist()))

    summaries: list[dict[str, object]] = []
    examples: list[pd.DataFrame] = []

    for focal_var in focal_vars:
        meta_f = _subset_by_focal(meta_df, focal_var)
        lmm_f = _subset_by_focal(lmm_df, focal_var)
        qc_f = _subset_by_focal(qc_df, focal_var)

        key_cols = ["gene_id", "focal_var"]
        joined = meta_f.merge(lmm_f, on=key_cols, how="inner", suffixes=("_meta", "_lmm"))

        # Only consider LMM rows here (exclude meta_fallback/failed) for comparison.
        joined_lmm = joined.loc[joined["method"].astype(str) == "lmm"].copy()

        joined_lmm["theta_diff"] = joined_lmm["theta_lmm"].astype(float) - joined_lmm["theta_meta"].astype(float)
        joined_lmm["abs_theta_diff"] = joined_lmm["theta_diff"].abs()
        joined_lmm["sign_match"] = np.sign(joined_lmm["theta_lmm"].astype(float)) == np.sign(joined_lmm["theta_meta"].astype(float))

        # Significance (FDR) comparisons
        joined_lmm["meta_sig"] = joined_lmm["p_adj"].astype(float) <= float(q)
        joined_lmm["lrt_sig"] = joined_lmm["lrt_p_adj"].astype(float) <= float(q)
        joined_lmm["wald_sig"] = joined_lmm["wald_p_adj"].astype(float) <= float(q)

        meta_sig = joined_lmm["meta_sig"].sum()
        lrt_sig = joined_lmm["lrt_sig"].sum()
        wald_sig = joined_lmm["wald_sig"].sum()

        both_meta_lrt = (joined_lmm["meta_sig"] & joined_lmm["lrt_sig"]).sum()
        meta_only = (joined_lmm["meta_sig"] & ~joined_lmm["lrt_sig"]).sum()
        lrt_only = (~joined_lmm["meta_sig"] & joined_lmm["lrt_sig"]).sum()
        neither = (~joined_lmm["meta_sig"] & ~joined_lmm["lrt_sig"]).sum()

        both_lrt_wald = (joined_lmm["lrt_sig"] & joined_lmm["wald_sig"]).sum()
        lrt_only_vs_wald = (joined_lmm["lrt_sig"] & ~joined_lmm["wald_sig"]).sum()
        wald_only_vs_lrt = (~joined_lmm["lrt_sig"] & joined_lmm["wald_sig"]).sum()

        sign_mismatch = (~joined_lmm["sign_match"]).sum()

        # Attach QC columns if present.
        qc_join = joined_lmm.merge(qc_f, on=key_cols, how="left", suffixes=("", "_qc"))
        qc_cols = [c for c in ["beta_median", "beta_trimmed_mean", "beta_winsor_mean", "beta_huber"] if c in qc_join.columns]
        for c in qc_cols:
            qc_join[f"{c}_diff_vs_meta_theta"] = qc_join[c].astype(float) - qc_join["theta_meta"].astype(float)

        # Example rows: prioritize sign mismatches, then large abs theta diff.
        ex = qc_join.copy()
        if min_abs_theta_diff is not None:
            ex = ex.loc[ex["abs_theta_diff"].astype(float) >= float(min_abs_theta_diff)].copy()
        ex["example_rank"] = np.where(ex["sign_match"].fillna(True), 1, 0)  # mismatches first (0)
        ex = ex.sort_values(
            ["example_rank", "abs_theta_diff", "gene_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        if top_n > 0:
            ex = ex.head(int(top_n)).copy()

        keep_cols = [
            "gene_id",
            "focal_var",
            "theta_meta",
            "p_adj",
            "theta_lmm",
            "lrt_p_adj",
            "wald_p_adj",
            "sign_match",
            "abs_theta_diff",
        ]
        for extra in ["tau_meta", "Q_p_adj", "m_guides_used"]:
            if extra in ex.columns:
                keep_cols.append(extra)
        keep_cols += qc_cols
        for c in qc_cols:
            keep_cols.append(f"{c}_diff_vs_meta_theta")
        keep_cols = [c for c in keep_cols if c in ex.columns]
        examples.append(ex.loc[:, keep_cols])

        summaries.append(
            {
                "focal_var": focal_var,
                "q": float(q),
                "n_meta": int(meta_f.shape[0]),
                "n_lmm_rows": int(lmm_f.shape[0]),
                "n_lmm_success": int(joined_lmm.shape[0]),
                "sign_mismatch_frac_meta_vs_lmm": float(sign_mismatch / joined_lmm.shape[0]) if joined_lmm.shape[0] else float("nan"),
                "abs_theta_diff_median": _safe_median(joined_lmm["abs_theta_diff"]),
                "abs_theta_diff_p90": _safe_quantile(joined_lmm["abs_theta_diff"], 0.9),
                "meta_sig_n": int(meta_sig),
                "lrt_sig_n": int(lrt_sig),
                "wald_sig_n": int(wald_sig),
                "meta_lrt_both_sig_n": int(both_meta_lrt),
                "meta_only_sig_n": int(meta_only),
                "lrt_only_sig_n": int(lrt_only),
                "meta_lrt_neither_sig_n": int(neither),
                "lrt_wald_both_sig_n": int(both_lrt_wald),
                "lrt_only_sig_vs_wald_n": int(lrt_only_vs_wald),
                "wald_only_sig_vs_lrt_n": int(wald_only_vs_lrt),
            }
        )

    summary_df = pd.DataFrame(summaries).sort_values(["focal_var"], kind="mergesort").reset_index(drop=True)
    examples_df = pd.concat(examples, axis=0, ignore_index=True) if examples else pd.DataFrame()
    if not examples_df.empty:
        examples_df = examples_df.sort_values(["focal_var", "abs_theta_diff", "gene_id"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    return summary_df, examples_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage agreement/disagreement across gene-level methods (meta vs Plan A LMM).")
    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Run output directory that contains a `gene_level/` folder (or pass `--gene-level-dir`).",
    )
    parser.add_argument(
        "--gene-level-dir",
        default=None,
        type=str,
        help="Optional explicit gene-level directory (default: <out-dir>/gene_level).",
    )
    parser.add_argument(
        "--q",
        default=0.1,
        type=float,
        help="FDR threshold used to compare significance calls (default: 0.1).",
    )
    parser.add_argument(
        "--top-n",
        default=50,
        type=int,
        help="Example rows per focal var to write (default: 50).",
    )
    parser.add_argument(
        "--min-abs-theta-diff",
        default=None,
        type=float,
        help="Optional minimum |theta_lmm - theta_meta| to include in examples (default: no filter).",
    )
    parser.add_argument(
        "--write-summary-tsv",
        default=None,
        type=str,
        help="Optional path to write the summary TSV (default: <out-dir>/gene_level_method_disagreement_summary.tsv).",
    )
    parser.add_argument(
        "--write-examples-tsv",
        default=None,
        type=str,
        help="Optional path to write the examples TSV (default: <out-dir>/gene_level_method_disagreement_examples.tsv).",
    )
    args = parser.parse_args()

    gene_level_dir = args.gene_level_dir or os.path.join(args.out_dir, "gene_level")
    summary_df, examples_df = triage_gene_level_disagreements(
        gene_level_dir,
        q=float(args.q),
        top_n=int(args.top_n),
        min_abs_theta_diff=args.min_abs_theta_diff,
    )

    summary_path = args.write_summary_tsv or os.path.join(args.out_dir, "gene_level_method_disagreement_summary.tsv")
    examples_path = args.write_examples_tsv or os.path.join(args.out_dir, "gene_level_method_disagreement_examples.tsv")

    Path(os.path.dirname(summary_path)).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, sep="\t", index=False)
    examples_df.to_csv(examples_path, sep="\t", index=False)

    print(summary_path)
    print(examples_path)


if __name__ == "__main__":
    main()

