from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from count_depth_pipelines import pipeline_label
from count_depth_scenarios import SCENARIO_CANDIDATE_COLS
from count_depth_scenarios import attach_scenarios


def _load_truth_gene(path: str, *, include_reference_genes: bool) -> pd.DataFrame:
    truth = pd.read_csv(path, sep="\t")
    if "gene_id" not in truth.columns:
        raise ValueError(f"truth gene table missing gene_id: {path!r}")
    truth["gene_id"] = truth["gene_id"].astype(str)
    truth["is_signal"] = truth.get("is_signal", False)
    truth["is_signal"] = truth["is_signal"].astype(bool)
    if not bool(include_reference_genes) and "is_reference" in truth.columns:
        truth = truth.loc[~truth["is_reference"].astype(bool)].copy()
    return truth[["gene_id", "is_signal"]].copy()


def _load_expected_gene(path: str) -> pd.DataFrame:
    exp = pd.read_csv(path, sep="\t")
    if "gene_id" not in exp.columns:
        raise ValueError(f"expected-count table missing gene_id: {path!r}")
    if "expected_p10_mincond_bucket" not in exp.columns:
        raise ValueError(f"expected-count table missing expected_p10_mincond_bucket: {path!r}")
    exp["gene_id"] = exp["gene_id"].astype(str)
    exp["expected_p10_mincond_bucket"] = exp["expected_p10_mincond_bucket"].astype(object)
    return exp[["gene_id", "expected_p10_mincond_bucket"]].copy()


def _read_gene_p_table(
    *,
    run_dir: str,
    method: str,
) -> pd.DataFrame | None:
    if method == "meta":
        path = os.path.join(run_dir, "PMD_std_res_gene_meta.tsv")
        if not os.path.isfile(path):
            return None
        df = pd.read_csv(path, sep="\t")
        return df
    if method == "stouffer":
        path = os.path.join(run_dir, "PMD_std_res_gene_stouffer.tsv")
        if not os.path.isfile(path):
            return None
        df = pd.read_csv(path, sep="\t")
        return df
    if method in {"lmm_lrt", "lmm_wald"}:
        path = os.path.join(run_dir, "PMD_std_res_gene_lmm.tsv")
        if not os.path.isfile(path):
            return None
        df = pd.read_csv(path, sep="\t")
        return df
    raise ValueError(f"unsupported method: {method!r}")


def _extract_p_columns(df: pd.DataFrame, *, method: str) -> tuple[pd.Series, pd.Series, pd.Series | None, pd.Series | None]:
    """
    Return: p_raw, p_adj, ok_flag, theta_hat
    """
    if "gene_id" not in df.columns:
        raise ValueError("missing gene_id column in method table")
    if "focal_var" in df.columns:
        df = df.loc[df["focal_var"].astype(str) == "treatment"].copy()
    df = df.copy()
    df["gene_id"] = df["gene_id"].astype(str)
    df = df.set_index("gene_id")

    if method in {"meta", "stouffer"}:
        if "p" not in df.columns or "p_adj" not in df.columns:
            raise ValueError(f"missing p/p_adj columns for method={method}")
        p = pd.to_numeric(df["p"], errors="coerce")
        p_adj = pd.to_numeric(df["p_adj"], errors="coerce")
        ok = pd.Series([True] * df.shape[0], index=df.index.copy(), dtype=bool)
        theta = pd.to_numeric(df["theta"], errors="coerce") if "theta" in df.columns else None
        return p, p_adj, ok, theta

    if method == "lmm_lrt":
        required = {"lrt_p", "lrt_p_adj", "lrt_ok"}
        missing = required.difference(set(df.columns))
        if missing:
            raise ValueError(f"missing lmm columns: {sorted(missing)}")
        p = pd.to_numeric(df["lrt_p"], errors="coerce")
        p_adj = pd.to_numeric(df["lrt_p_adj"], errors="coerce")
        ok = df["lrt_ok"].astype(bool)
        theta = pd.to_numeric(df["theta"], errors="coerce") if "theta" in df.columns else None
        return p, p_adj, ok, theta

    if method == "lmm_wald":
        required = {"wald_p", "wald_p_adj", "wald_ok"}
        missing = required.difference(set(df.columns))
        if missing:
            raise ValueError(f"missing lmm columns: {sorted(missing)}")
        p = pd.to_numeric(df["wald_p"], errors="coerce")
        p_adj = pd.to_numeric(df["wald_p_adj"], errors="coerce")
        ok = df["wald_ok"].astype(bool)
        theta = pd.to_numeric(df["theta"], errors="coerce") if "theta" in df.columns else None
        return p, p_adj, ok, theta

    raise ValueError(f"unsupported method: {method!r}")


def _load_report_config(report_path: str) -> dict[str, object]:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    cfg = report.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"missing config in report: {report_path!r}")
    return cfg


def _as_float(x: object) -> float | None:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if not np.isfinite(v):
        return None
    return float(v)


def _compute_bucket_metrics_for_method(
    joined: pd.DataFrame,
    *,
    method: str,
    alpha: float,
    fdr_q: float,
    theta_true: pd.Series | None,
    theta_hat: pd.Series | None,
    p: pd.Series,
    p_adj: pd.Series,
    ok: pd.Series | None,
) -> list[dict[str, object]]:
    # Import shared benchmark helpers (avoid re-implementing core metric semantics).
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from benchmark_count_depth import _confusion  # noqa: PLC0415
    from benchmark_count_depth import _ks_uniform  # noqa: PLC0415
    from benchmark_count_depth import _qq_stats  # noqa: PLC0415

    fixed_buckets = ["<1", "1-<3", "3-<5", ">=5"]
    rows: list[dict[str, object]] = []

    for bucket in fixed_buckets:
        sub = joined.loc[joined["expected_p10_mincond_bucket"].astype(object) == bucket].copy()
        n_total = int(sub.shape[0])

        is_signal = sub["is_signal"].astype(bool).to_numpy(dtype=bool)
        p_sub = pd.to_numeric(p.reindex(sub.index), errors="coerce")
        p_adj_sub = pd.to_numeric(p_adj.reindex(sub.index), errors="coerce")

        p_finite = np.isfinite(p_sub.to_numpy(dtype=float))
        coverage_n = int(np.sum(p_finite))
        coverage_frac = (coverage_n / n_total) if n_total else None

        if ok is not None:
            ok_sub = ok.reindex(sub.index).fillna(False).astype(bool)
            ok_frac = float(ok_sub.mean()) if n_total else None
        else:
            ok_frac = None

        # Null-only calibration (raw p-values).
        null_mask = (~is_signal) & p_finite
        null_p = pd.Series(p_sub.to_numpy(dtype=float)[null_mask])
        ks = _ks_uniform(null_p) if not null_p.empty else {"n": 0.0, "ks": None, "ks_p": None}
        qq = _qq_stats(null_p) if not null_p.empty else {"n": 0, "lambda_gc": None}

        null_prop_lt_alpha = None
        if null_p.size > 0:
            null_prop_lt_alpha = float(np.mean(null_p.to_numpy(dtype=float) < float(alpha)))

        # Signal-only power summaries.
        sig_mask = is_signal & p_finite
        sig_p = p_sub.to_numpy(dtype=float)[sig_mask]
        sig_tpr_alpha = float(np.mean(sig_p < float(alpha))) if sig_p.size else None

        sig_called_q = None
        if int(np.sum(is_signal)) > 0:
            sig_p_adj = p_adj_sub.to_numpy(dtype=float)[is_signal]
            sig_called_q = float(np.mean(sig_p_adj < float(fdr_q))) if sig_p_adj.size else None

        # Confusion matrices (bucket-local).
        called_alpha = (p_sub.to_numpy(dtype=float) < float(alpha)) & p_finite
        called_q = p_adj_sub.to_numpy(dtype=float) < float(fdr_q)
        alpha_conf = _confusion(called_alpha, is_signal) if n_total else {}
        q_conf = _confusion(called_q, is_signal) if n_total else {}

        # Optional theta metrics (signal-only).
        theta_corr_signal = None
        theta_rmse_signal = None
        if theta_true is not None and theta_hat is not None:
            t_true = pd.to_numeric(theta_true.reindex(sub.index), errors="coerce").to_numpy(dtype=float)
            t_hat = pd.to_numeric(theta_hat.reindex(sub.index), errors="coerce").to_numpy(dtype=float)
            mask = is_signal & np.isfinite(t_true) & np.isfinite(t_hat)
            if int(np.sum(mask)) >= 2:
                theta_corr_signal = float(np.corrcoef(t_true[mask], t_hat[mask])[0, 1])
            if int(np.sum(mask)) >= 1:
                theta_rmse_signal = float(np.sqrt(np.mean((t_hat[mask] - t_true[mask]) ** 2)))

        def _emit(metric: str, value: object, n_genes: int | None) -> None:
            rows.append(
                {
                    "method": method,
                    "bucket": bucket,
                    "metric": metric,
                    "value": _as_float(value),
                    "n_genes": float(n_genes) if n_genes is not None else None,
                }
            )

        _emit("n_total", n_total, n_total)
        _emit("coverage_p_n", coverage_n, n_total)
        _emit("coverage_p_frac", coverage_frac, n_total)
        if ok_frac is not None:
            _emit("ok_frac", ok_frac, n_total)

        _emit("null_n", int(np.sum(~is_signal)), n_total)
        _emit("null_ks", ks.get("ks"), int(ks.get("n") or 0))
        _emit("null_ks_p", ks.get("ks_p"), int(ks.get("n") or 0))
        _emit("null_lambda_gc", qq.get("lambda_gc"), int(qq.get("n") or 0))
        _emit("null_prop_lt_alpha", null_prop_lt_alpha, int(ks.get("n") or 0))

        _emit("signal_n", int(np.sum(is_signal)), n_total)
        _emit("signal_tpr_alpha", sig_tpr_alpha, int(np.sum(is_signal)))
        _emit("signal_tpr_q", sig_called_q, int(np.sum(is_signal)))
        _emit("signal_theta_corr", theta_corr_signal, int(np.sum(is_signal)))
        _emit("signal_theta_rmse", theta_rmse_signal, int(np.sum(is_signal)))

        for k, v in alpha_conf.items():
            _emit(f"alpha_{k}", v, n_total)
        for k, v in q_conf.items():
            _emit(f"q_{k}", v, n_total)

    return rows


def _run_one(
    row: pd.Series,
    *,
    methods: list[str],
    include_reference_genes: bool,
) -> list[dict[str, object]]:
    report_path = str(row.get("report_path", ""))
    if not report_path:
        raise ValueError("missing report_path")
    run_dir = os.path.dirname(report_path)

    cfg = _load_report_config(report_path)
    alpha = float(cfg.get("alpha", row.get("alpha", 0.05)))
    fdr_q = float(cfg.get("fdr_q", row.get("fdr_q", 0.1)))

    truth_path = os.path.join(run_dir, "sim_truth_gene.tsv")
    expected_path = os.path.join(run_dir, "sim_gene_expected_counts.tsv")
    if not os.path.isfile(truth_path):
        raise FileNotFoundError(truth_path)
    if not os.path.isfile(expected_path):
        raise FileNotFoundError(expected_path)

    truth = _load_truth_gene(truth_path, include_reference_genes=bool(include_reference_genes))
    expected = _load_expected_gene(expected_path)
    gene_base = truth.merge(expected, on="gene_id", how="left", validate="one_to_one")
    if gene_base["expected_p10_mincond_bucket"].isna().any():
        raise ValueError(f"missing bucket assignment for some genes in {expected_path!r}")

    theta_true = None
    truth_full = pd.read_csv(truth_path, sep="\t")
    if "theta_true" in truth_full.columns:
        truth_full["gene_id"] = truth_full["gene_id"].astype(str)
        if not bool(include_reference_genes) and "is_reference" in truth_full.columns:
            truth_full = truth_full.loc[~truth_full["is_reference"].astype(bool)].copy()
        theta_true = pd.to_numeric(truth_full.set_index("gene_id")["theta_true"], errors="coerce")

    base_cols = [
        # Pipeline knobs
        "response_mode",
        "normalization_mode",
        "logratio_mode",
        "n_reference_genes",
        "depth_covariate_mode",
        "include_batch_covariate",
        "lmm_scope",
        "lmm_max_genes_per_focal_var",
        # Thresholds (kept explicit; used in derived dev/excess metrics)
        "alpha",
        "fdr_q",
        # Scenario knobs
        *[c for c in SCENARIO_CANDIDATE_COLS if c in row.index],
        # Seed/repro
        "seed",
        "report_path",
    ]
    base_cols = [c for c in base_cols if c in row.index]

    out_rows: list[dict[str, object]] = []
    for method in methods:
        gene_table = _read_gene_p_table(run_dir=run_dir, method=method)
        if gene_table is None:
            continue

        p_raw, p_adj, ok, theta_hat = _extract_p_columns(gene_table, method=method)
        gene_table = gene_table.copy()
        gene_table["gene_id"] = gene_table["gene_id"].astype(str)
        if "focal_var" in gene_table.columns:
            gene_table = gene_table.loc[gene_table["focal_var"].astype(str) == "treatment"].copy()
        gene_table = gene_table.set_index("gene_id")

        joined = gene_base.set_index("gene_id").join(gene_table, how="left")
        joined = joined.reset_index().set_index("gene_id")

        metrics = _compute_bucket_metrics_for_method(
            joined,
            method=method,
            alpha=alpha,
            fdr_q=fdr_q,
            theta_true=theta_true,
            theta_hat=theta_hat.reindex(joined.index) if theta_hat is not None else None,
            p=p_raw.reindex(joined.index),
            p_adj=p_adj.reindex(joined.index),
            ok=ok.reindex(joined.index) if ok is not None else None,
        )

        pipe = pipeline_label(row, method=method)
        for m in metrics:
            out: dict[str, object] = {c: row.get(c) for c in base_cols}
            out["pipeline"] = pipe
            out.update(m)
            out_rows.append(out)

    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect bucket-stratified benchmark metrics (local; additive).")
    parser.add_argument("--grid-tsv", required=True, type=str, help="Path to count_depth_grid_summary.tsv (must include report_path).")
    parser.add_argument(
        "--out-tsv",
        required=True,
        type=str,
        help="Output TSV path (long format).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["meta", "stouffer", "lmm_lrt", "lmm_wald"],
        choices=["meta", "stouffer", "lmm_lrt", "lmm_wald"],
        help="Which method families to collect (default: meta stouffer lmm_lrt lmm_wald).",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers (default: 1).")
    parser.add_argument(
        "--include-reference-genes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, include reference genes in evaluation (default: disabled).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.grid_tsv, sep="\t")
    if "report_path" not in df.columns:
        raise ValueError("grid TSV must include report_path")

    methods = [str(m) for m in (args.methods or [])]
    jobs = max(1, int(args.jobs))

    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    if jobs == 1:
        for _i, r in df.iterrows():
            try:
                rows.extend(_run_one(r, methods=methods, include_reference_genes=bool(args.include_reference_genes)))
            except Exception as e:  # noqa: BLE001
                errors.append({"report_path": str(r.get("report_path", "")), "error": str(e)})
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {
                ex.submit(_run_one, r, methods=methods, include_reference_genes=bool(args.include_reference_genes)): str(r.get("report_path", ""))
                for _i, r in df.iterrows()
            }
            for fut in as_completed(futs):
                report_path = futs[fut]
                try:
                    rows.extend(fut.result())
                except Exception as e:  # noqa: BLE001
                    errors.append({"report_path": report_path, "error": str(e)})

    if errors:
        # Surface errors clearly (no silent partial outputs).
        raise RuntimeError(f"failed for {len(errors)} run(s), e.g. {errors[:3]}")

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("no bucket metrics produced (check methods and run dirs)")

    out = attach_scenarios(out)

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(args.out_tsv)


if __name__ == "__main__":
    main()
