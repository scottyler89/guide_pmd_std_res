from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _safe_logp(p: pd.Series) -> np.ndarray:
    p_arr = pd.to_numeric(p, errors="coerce").to_numpy(dtype=float)
    p_arr = np.where(np.isfinite(p_arr), p_arr, np.nan)
    p_arr = np.clip(p_arr, 1e-300, 1.0)
    return -np.log10(p_arr)


def _spearman_r(x: pd.Series, y: pd.Series) -> float | None:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        return None
    rx = pd.Series(x_arr).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y_arr).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _call_matrix(called_a: np.ndarray, called_b: np.ndarray) -> np.ndarray:
    a = np.asarray(called_a, dtype=bool)
    b = np.asarray(called_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("called arrays must have the same shape")
    # Rows = A called? [False, True]; Cols = B called? [False, True]
    m = np.zeros((2, 2), dtype=int)
    m[0, 0] = int(np.sum(~a & ~b))
    m[0, 1] = int(np.sum(~a & b))
    m[1, 0] = int(np.sum(a & ~b))
    m[1, 1] = int(np.sum(a & b))
    return m


def _jaccard_from_call_matrix(m: np.ndarray) -> float | None:
    a_or_b = float(m[0, 1] + m[1, 0] + m[1, 1])
    if a_or_b == 0.0:
        return None
    return float(m[1, 1] / a_or_b)


def _plot_scatter(
    df: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    p_col_a: str,
    p_col_b: str,
    out_path: str,
    title: str,
) -> dict[str, float | int | None]:
    plt = _require_matplotlib()

    x = _safe_logp(df[p_col_a])
    y = _safe_logp(df[p_col_b])
    sig = df["is_signal"].to_numpy(dtype=bool)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    sig = sig[mask]

    if x.size == 0:
        return {"n": 0, "spearman_r": None}

    max_v = float(max(np.nanmax(x), np.nanmax(y), 1.0))

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=150)
    ax.scatter(x[~sig], y[~sig], s=8, alpha=0.5, edgecolors="none", color="#1f77b4", label="null")
    ax.scatter(x[sig], y[sig], s=8, alpha=0.7, edgecolors="none", color="#d62728", label="signal")
    ax.plot([0.0, max_v], [0.0, max_v], color="black", lw=1, alpha=0.7)
    ax.set_xlim(0.0, max_v)
    ax.set_ylim(0.0, max_v)
    ax.set_xlabel(f"{method_a}  -log10(p)")
    ax.set_ylabel(f"{method_b}  -log10(p)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    r = _spearman_r(df[p_col_a], df[p_col_b])
    return {"n": int(x.size), "spearman_r": r}


def _plot_call_heatmap(
    df: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    p_adj_a: str,
    p_adj_b: str,
    q: float,
    out_path: str,
    title: str,
) -> dict[str, float | int | None]:
    plt = _require_matplotlib()

    a = pd.to_numeric(df[p_adj_a], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(df[p_adj_b], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return {"n": 0, "jaccard": None}

    called_a = a < float(q)
    called_b = b < float(q)
    m = _call_matrix(called_a, called_b)
    jac = _jaccard_from_call_matrix(m)

    fig, ax = plt.subplots(figsize=(3.4, 3.2), dpi=150)
    im = ax.imshow(m, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"{method_b}\nnot called", f"{method_b}\ncalled"])
    ax.set_yticklabels([f"{method_a}\nnot called", f"{method_a}\ncalled"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(m[i, j])), ha="center", va="center", fontsize=10, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return {"n": int(a.size), "jaccard": jac}


def main() -> None:
    parser = argparse.ArgumentParser(description="Agreement/disagreement plots across benchmark methods for a single run dir (local).")
    parser.add_argument("--run-dir", required=True, type=str, help="Benchmark run output directory (contains sim_truth_gene.tsv, gene tables).")
    parser.add_argument("--out-dir", default=None, type=str, help="Output directory for plots (default: <run-dir>/figures/method_agreement).")
    parser.add_argument("--focal-var", default="treatment", type=str, help="Focal var to plot (default: treatment).")
    parser.add_argument("--q", default=None, type=float, help="FDR q to use for call comparisons (default: from report config).")
    args = parser.parse_args()

    run_dir = str(args.run_dir)
    out_dir = args.out_dir or os.path.join(run_dir, "figures", "method_agreement")
    os.makedirs(out_dir, exist_ok=True)

    report_path = os.path.join(run_dir, "benchmark_report.json")
    q = float(args.q) if args.q is not None else None
    if q is None and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        q = float(rep.get("config", {}).get("fdr_q", 0.1))
    if q is None:
        q = 0.1

    truth_path = os.path.join(run_dir, "sim_truth_gene.tsv")
    truth = _read_tsv(truth_path)
    required_truth = {"gene_id", "is_signal"}
    if not required_truth.issubset(set(truth.columns)):
        raise ValueError(f"truth table missing required column(s): {sorted(required_truth.difference(set(truth.columns)))}")
    truth = truth[["gene_id", "is_signal"]].copy()
    truth["gene_id"] = truth["gene_id"].astype(str)
    truth["is_signal"] = truth["is_signal"].astype(bool)

    methods: dict[str, dict[str, str]] = {}
    tables: dict[str, pd.DataFrame] = {}

    meta_path = os.path.join(run_dir, "PMD_std_res_gene_meta.tsv")
    if os.path.exists(meta_path):
        df = _read_tsv(meta_path)
        df = df.loc[df.get("focal_var", "").astype(str) == str(args.focal_var)].copy()
        tables["meta"] = truth.merge(df, on="gene_id", how="left")
        methods["meta"] = {"p": "p", "p_adj": "p_adj"}

    stouffer_path = os.path.join(run_dir, "PMD_std_res_gene_stouffer.tsv")
    if os.path.exists(stouffer_path):
        df = _read_tsv(stouffer_path)
        df = df.loc[df.get("focal_var", "").astype(str) == str(args.focal_var)].copy()
        tables["stouffer"] = truth.merge(df, on="gene_id", how="left")
        methods["stouffer"] = {"p": "p", "p_adj": "p_adj"}

    lmm_path = os.path.join(run_dir, "PMD_std_res_gene_lmm.tsv")
    if os.path.exists(lmm_path):
        df = _read_tsv(lmm_path)
        df = df.loc[df.get("focal_var", "").astype(str) == str(args.focal_var)].copy()
        joined = truth.merge(df, on="gene_id", how="left")
        tables["lmm_lrt"] = joined
        tables["lmm_wald"] = joined
        methods["lmm_lrt"] = {"p": "lrt_p", "p_adj": "lrt_p_adj"}
        methods["lmm_wald"] = {"p": "wald_p", "p_adj": "wald_p_adj"}

    if not methods:
        raise ValueError("no method tables found in run dir")

    method_names = [m for m in ["meta", "stouffer", "lmm_lrt", "lmm_wald"] if m in methods]

    rows: list[dict[str, object]] = []
    for a, b in combinations(method_names, 2):
        df_a = tables[a]
        df_b = tables[b]
        p_a = methods[a]["p"]
        p_b = methods[b]["p"]
        p_adj_a = methods[a]["p_adj"]
        p_adj_b = methods[b]["p_adj"]

        col_p_a = f"{a}__p"
        col_p_adj_a = f"{a}__p_adj"
        col_p_b = f"{b}__p"
        col_p_adj_b = f"{b}__p_adj"

        df_pair = truth.copy()
        a_indexed = df_a.set_index("gene_id", drop=False)
        b_indexed = df_b.set_index("gene_id", drop=False)
        a_p = pd.to_numeric(a_indexed[p_a], errors="coerce") if p_a in a_indexed.columns else pd.Series(dtype=float)
        a_q = pd.to_numeric(a_indexed[p_adj_a], errors="coerce") if p_adj_a in a_indexed.columns else pd.Series(dtype=float)
        b_p = pd.to_numeric(b_indexed[p_b], errors="coerce") if p_b in b_indexed.columns else pd.Series(dtype=float)
        b_q = pd.to_numeric(b_indexed[p_adj_b], errors="coerce") if p_adj_b in b_indexed.columns else pd.Series(dtype=float)

        df_pair[col_p_a] = df_pair["gene_id"].map(a_p)
        df_pair[col_p_adj_a] = df_pair["gene_id"].map(a_q)
        df_pair[col_p_b] = df_pair["gene_id"].map(b_p)
        df_pair[col_p_adj_b] = df_pair["gene_id"].map(b_q)

        scatter_path = os.path.join(out_dir, f"scatter__{a}__vs__{b}__p.png")
        scatter_stats = _plot_scatter(
            df_pair,
            method_a=a,
            method_b=b,
            p_col_a=col_p_a,
            p_col_b=col_p_b,
            out_path=scatter_path,
            title=f"{a} vs {b}  (focal_var={args.focal_var})",
        )

        call_path = os.path.join(out_dir, f"calls_q__{a}__vs__{b}.png")
        call_stats = _plot_call_heatmap(
            df_pair,
            method_a=a,
            method_b=b,
            p_adj_a=col_p_adj_a,
            p_adj_b=col_p_adj_b,
            q=q,
            out_path=call_path,
            title=f"Calls at q={q:g}  (focal_var={args.focal_var})",
        )

        rows.append(
            {
                "focal_var": str(args.focal_var),
                "q": float(q),
                "method_a": a,
                "method_b": b,
                "scatter_n": scatter_stats.get("n"),
                "scatter_spearman_r": scatter_stats.get("spearman_r"),
                "calls_n": call_stats.get("n"),
                "calls_jaccard": call_stats.get("jaccard"),
                "scatter_png": scatter_path,
                "calls_png": call_path,
            }
        )

    out_tsv = os.path.join(out_dir, "method_pair_agreement.tsv")
    pd.DataFrame(rows).sort_values(["method_a", "method_b"], kind="mergesort").to_csv(out_tsv, sep="\t", index=False)
    print(out_tsv)


if __name__ == "__main__":
    main()
