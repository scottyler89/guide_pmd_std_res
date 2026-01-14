from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _finite_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[m], y_arr[m]


def _pearson(x: pd.Series, y: pd.Series) -> float:
    x_arr, y_arr = _finite_pair(x, y)
    if x_arr.size < 2:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    x_arr, y_arr = _finite_pair(x, y)
    if x_arr.size < 2:
        return float("nan")
    # Stable Spearman via ranking.
    rx = pd.Series(x_arr).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y_arr).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _sign_match(x: pd.Series, y: pd.Series) -> float:
    x_arr, y_arr = _finite_pair(x, y)
    if x_arr.size == 0:
        return float("nan")
    return float(np.mean(np.sign(x_arr) == np.sign(y_arr)))


def _subset_by_focal(df: pd.DataFrame, focal_var: str) -> pd.DataFrame:
    if "focal_var" not in df.columns:
        raise ValueError("expected column: focal_var")
    return df.loc[df["focal_var"].astype(str) == str(focal_var)].copy()


def compare_gene_level_methods(gene_level_dir: str) -> tuple[pd.DataFrame, dict[str, object]]:
    gene_level_dir = str(gene_level_dir)
    meta_path = os.path.join(gene_level_dir, "PMD_std_res_gene_meta.tsv")
    stouffer_path = os.path.join(gene_level_dir, "PMD_std_res_gene_stouffer.tsv")
    lmm_path = os.path.join(gene_level_dir, "PMD_std_res_gene_lmm.tsv")
    qc_path = os.path.join(gene_level_dir, "PMD_std_res_gene_qc.tsv")

    meta_df = _read_tsv(meta_path)
    stouffer_df = _read_tsv(stouffer_path) if os.path.exists(stouffer_path) else pd.DataFrame()
    lmm_df = _read_tsv(lmm_path)
    qc_df = _read_tsv(qc_path)

    for required in ("gene_id", "focal_var"):
        if required not in meta_df.columns:
            raise ValueError(f"meta missing required column: {required}")
        if required not in lmm_df.columns:
            raise ValueError(f"lmm missing required column: {required}")
        if required not in qc_df.columns:
            raise ValueError(f"qc missing required column: {required}")

    focal_vars = sorted(set(meta_df["focal_var"].astype(str).unique().tolist()))

    summaries: list[dict[str, object]] = []
    notes: dict[str, object] = {
        "gene_level_dir": gene_level_dir,
        "meta_path": meta_path,
        "stouffer_path": stouffer_path if os.path.exists(stouffer_path) else "",
        "lmm_path": lmm_path,
        "qc_path": qc_path,
    }

    for focal_var in focal_vars:
        meta_f = _subset_by_focal(meta_df, focal_var)
        stouffer_f = _subset_by_focal(stouffer_df, focal_var) if not stouffer_df.empty else pd.DataFrame()
        lmm_f = _subset_by_focal(lmm_df, focal_var)
        qc_f = _subset_by_focal(qc_df, focal_var)

        key_cols = ["gene_id", "focal_var"]
        joined = meta_f.merge(lmm_f, on=key_cols, how="inner", suffixes=("_meta", "_lmm"))

        lmm_success = joined.loc[joined["method"].astype(str) == "lmm"].copy()
        lmm_fallback = joined.loc[joined["method"].astype(str) == "meta_fallback"].copy()

        def _col(name: str) -> pd.Series:
            if name not in joined.columns:
                return pd.Series(dtype=float)
            return joined[name]

        theta_corr = _pearson(lmm_success["theta_meta"], lmm_success["theta_lmm"]) if not lmm_success.empty else float("nan")
        theta_sign = _sign_match(lmm_success["theta_meta"], lmm_success["theta_lmm"]) if not lmm_success.empty else float("nan")
        p_lrt_corr = (
            _spearman(lmm_success["p"], lmm_success["lrt_p"]) if ("lrt_p" in lmm_success.columns and not lmm_success.empty) else float("nan")
        )
        p_wald_corr = (
            _spearman(lmm_success["p"], lmm_success["wald_p"])
            if ("wald_p" in lmm_success.columns and not lmm_success.empty)
            else float("nan")
        )

        max_abs_theta_diff_fallback = float("nan")
        max_abs_p_diff_fallback = float("nan")
        if not lmm_fallback.empty:
            if "theta_lmm" in lmm_fallback.columns:
                max_abs_theta_diff_fallback = float(np.nanmax(np.abs(lmm_fallback["theta_lmm"].to_numpy(dtype=float) - lmm_fallback["theta_meta"].to_numpy(dtype=float))))
            if "wald_p" in lmm_fallback.columns:
                max_abs_p_diff_fallback = float(np.nanmax(np.abs(lmm_fallback["wald_p"].to_numpy(dtype=float) - lmm_fallback["p"].to_numpy(dtype=float))))

        # QC comparisons (robust summaries vs meta theta)
        qc_theta_corr = float("nan")
        qc_theta_spearman = float("nan")
        if not qc_f.empty and "beta_median" in qc_f.columns:
            qc_join = meta_f.merge(qc_f, on=key_cols, how="inner")
            if not qc_join.empty:
                qc_theta_corr = _pearson(qc_join["theta"], qc_join["beta_median"])
                qc_theta_spearman = _spearman(qc_join["theta"], qc_join["beta_median"])

        # Stouffer comparisons (p-value correlations and direction agreement vs meta).
        st_p_spearman = float("nan")
        st_theta_sign_match = float("nan")
        if not stouffer_f.empty:
            st_join = meta_f.merge(stouffer_f, on=key_cols, how="inner", suffixes=("_meta", "_stouffer"))
            if not st_join.empty:
                st_p_spearman = _spearman(st_join["p_meta"], st_join["p_stouffer"]) if "p_stouffer" in st_join.columns else float("nan")
                if "stouffer_t" in st_join.columns:
                    st_theta_sign_match = _sign_match(st_join["theta_meta"], st_join["stouffer_t"])

        summaries.append(
            {
                "focal_var": focal_var,
                "n_meta": int(meta_f.shape[0]),
                "n_stouffer": int(stouffer_f.shape[0]) if not stouffer_f.empty else 0,
                "n_qc": int(qc_f.shape[0]),
                "n_lmm_rows": int(lmm_f.shape[0]),
                "n_lmm_success": int(lmm_f.loc[lmm_f["method"].astype(str) == "lmm"].shape[0]),
                "n_lmm_meta_fallback": int(lmm_f.loc[lmm_f["method"].astype(str) == "meta_fallback"].shape[0]),
                "n_lmm_failed": int(lmm_f.loc[lmm_f["method"].astype(str) == "failed"].shape[0]),
                "lrt_ok_frac_lmm": float(np.mean(lmm_f.loc[lmm_f["method"].astype(str) == "lmm", "lrt_ok"].fillna(False).astype(bool)))
                if ("lrt_ok" in lmm_f.columns and (lmm_f["method"].astype(str) == "lmm").any())
                else float("nan"),
                "wald_ok_frac_lmm": float(np.mean(lmm_f.loc[lmm_f["method"].astype(str) == "lmm", "wald_ok"].fillna(False).astype(bool)))
                if ("wald_ok" in lmm_f.columns and (lmm_f["method"].astype(str) == "lmm").any())
                else float("nan"),
                "theta_corr_meta_vs_lmm": theta_corr,
                "theta_sign_match_meta_vs_lmm": theta_sign,
                "spearman_p_meta_vs_lrt_p": p_lrt_corr,
                "spearman_p_meta_vs_wald_p": p_wald_corr,
                "max_abs_theta_diff_meta_fallback": max_abs_theta_diff_fallback,
                "max_abs_p_diff_meta_fallback": max_abs_p_diff_fallback,
                "qc_theta_corr_meta_vs_beta_median": qc_theta_corr,
                "qc_theta_spearman_meta_vs_beta_median": qc_theta_spearman,
                "spearman_p_meta_vs_stouffer_p": st_p_spearman,
                "sign_match_meta_theta_vs_stouffer_t": st_theta_sign_match,
            }
        )

    summary_df = pd.DataFrame(summaries).sort_values(["focal_var"], kind="mergesort").reset_index(drop=True)
    return summary_df, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gene-level methods (meta vs LMM vs QC) for a given output directory.")
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
        "--write-tsv",
        default=None,
        type=str,
        help="Optional path to write the summary TSV (default: <out-dir>/gene_level_method_comparison_summary.tsv).",
    )
    parser.add_argument(
        "--write-json",
        default=None,
        type=str,
        help="Optional path to write a small JSON note file (default: <out-dir>/gene_level_method_comparison_summary.json).",
    )
    args = parser.parse_args()

    gene_level_dir = args.gene_level_dir or os.path.join(args.out_dir, "gene_level")
    summary_df, notes = compare_gene_level_methods(gene_level_dir)

    tsv_path = args.write_tsv or os.path.join(args.out_dir, "gene_level_method_comparison_summary.tsv")
    json_path = args.write_json or os.path.join(args.out_dir, "gene_level_method_comparison_summary.json")

    Path(os.path.dirname(tsv_path)).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(tsv_path, sep="\t", index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"notes": notes, "tsv": tsv_path}, f, indent=2, sort_keys=True)

    print(tsv_path)


if __name__ == "__main__":
    main()
