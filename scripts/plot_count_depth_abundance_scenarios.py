from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from count_depth_scenarios import attach_scenarios
from count_depth_scenarios import make_scenario_table


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _flatten_dict(d: dict[str, object], *, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}__{k}"
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key))
        else:
            out[key] = v
    return out


def _safe_float(x: object) -> float | None:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if not np.isfinite(v):
        return None
    return float(v)


def _savefig(fig, out_path: str) -> None:
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)


def _pick_representative_row(df: pd.DataFrame) -> pd.Series:
    """
    Pick a deterministic representative row for a scenario group.

    Preference order:
      - lower seed
      - response_mode: log_counts, guide_zscore_log_counts, pmd_std_res
      - normalization/logratio modes (lexicographic)
    """

    if df.empty:
        raise ValueError("empty group (no rows)")

    sub = df.copy()
    if "seed" in sub.columns:
        sub["seed"] = pd.to_numeric(sub["seed"], errors="coerce")

    if "response_mode" in sub.columns:
        order = {"log_counts": 0, "guide_zscore_log_counts": 1, "pmd_std_res": 2}
        sub["_rm_rank"] = sub["response_mode"].astype(str).map(order).fillna(999).astype(int)
    else:
        sub["_rm_rank"] = 0

    sort_cols: list[str] = []
    for c in ["seed", "_rm_rank", "normalization_mode", "logratio_mode", "depth_covariate_mode", "include_batch_covariate"]:
        if c in sub.columns:
            sort_cols.append(c)
    if sort_cols:
        sub = sub.sort_values(sort_cols, kind="mergesort")

    return sub.iloc[0]


def _load_json(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_truth_tables(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth_gene = pd.read_csv(os.path.join(run_dir, "sim_truth_gene.tsv"), sep="\t")
    truth_guide = pd.read_csv(os.path.join(run_dir, "sim_truth_guide.tsv"), sep="\t")
    truth_sample = pd.read_csv(os.path.join(run_dir, "sim_truth_sample.tsv"), sep="\t")
    return truth_gene, truth_guide, truth_sample


def _scenario_figure(
    *,
    scenario_label: str,
    scenario_id: str,
    seed: int | None,
    truth_gene: pd.DataFrame,
    truth_guide: pd.DataFrame,
    truth_sample: pd.DataFrame,
    audit: dict[str, object] | None,
) -> tuple[object, dict[str, object]]:
    plt = _require_matplotlib()

    is_ref = truth_gene.get("is_reference", False)
    if isinstance(is_ref, pd.Series):
        is_ref_mask = is_ref.astype(bool).to_numpy()
    else:
        is_ref_mask = np.zeros(truth_gene.shape[0], dtype=bool)
    target_gene_ids = set(truth_gene.loc[~is_ref_mask, "gene_id"].astype(str).tolist())

    guide = truth_guide.copy()
    guide["gene_id"] = guide["gene_id"].astype(str)
    guide = guide[guide["gene_id"].isin(target_gene_ids)].copy()
    guide["lambda_base"] = pd.to_numeric(guide["lambda_base"], errors="coerce")
    guide = guide[np.isfinite(guide["lambda_base"].to_numpy(dtype=float))]

    gene_total = guide.groupby("gene_id", sort=True)["lambda_base"].sum()
    gene_total = gene_total[np.isfinite(gene_total.to_numpy(dtype=float))]
    gene_total_sorted = np.sort(gene_total.to_numpy(dtype=float))[::-1]

    within_gene = guide.groupby("gene_id", sort=True)["lambda_base"].agg(["mean", "max"])
    within_gene["max_over_mean"] = within_gene["max"] / within_gene["mean"].replace(0.0, np.nan)
    within_gene = within_gene.replace([np.inf, -np.inf], np.nan).dropna()

    out_metrics: dict[str, object] = {
        "n_genes_target": int(len(target_gene_ids)),
        "n_guides_target": int(guide.shape[0]),
        "gene_total_lambda_p50": float(np.median(gene_total_sorted)) if gene_total_sorted.size else np.nan,
        "within_gene_max_over_mean_p90": float(np.quantile(within_gene["max_over_mean"], 0.9))
        if not within_gene.empty
        else np.nan,
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax_rank, ax_hist, ax_dom, ax_lib = axes.flatten()

    # Rank-abundance (gene totals).
    if gene_total_sorted.size:
        ax_rank.plot(np.arange(1, gene_total_sorted.size + 1), gene_total_sorted, lw=1.0)
        ax_rank.set_yscale("log")
    ax_rank.set_title("Gene total λ rank-abundance")
    ax_rank.set_xlabel("Rank")
    ax_rank.set_ylabel("Total λ (sum over guides)")

    # Guide lambda histogram.
    if not guide.empty:
        log10_lam = np.log10(np.clip(guide["lambda_base"].to_numpy(dtype=float), 1e-300, None))
        ax_hist.hist(log10_lam, bins=50, color="#4C78A8", alpha=0.9)
    ax_hist.set_title("Guide λ distribution")
    ax_hist.set_xlabel("log10(λ)")
    ax_hist.set_ylabel("Count")

    # Within-gene dominance.
    if not within_gene.empty:
        dom = within_gene["max_over_mean"].to_numpy(dtype=float)
        dom = dom[np.isfinite(dom)]
        ax_dom.hist(np.log10(np.clip(dom, 1.0, None)), bins=50, color="#F58518", alpha=0.9)
    ax_dom.set_title("Within-gene dominance (max/mean)")
    ax_dom.set_xlabel("log10(max/mean)")
    ax_dom.set_ylabel("Genes")

    # Library size (observed from counts).
    if "log_libsize" in truth_sample.columns:
        log_ls = pd.to_numeric(truth_sample["log_libsize"], errors="coerce").to_numpy(dtype=float)
        log_ls = log_ls[np.isfinite(log_ls)]
        ax_lib.hist(log_ls, bins=30, color="#54A24B", alpha=0.9)
    ax_lib.set_title("Sample log(libsize)")
    ax_lib.set_xlabel("log(libsize)")
    ax_lib.set_ylabel("Samples")

    seed_txt = f"seed={seed}" if seed is not None else "seed=?"
    fig.suptitle(f"{scenario_label}\n[id={scenario_id}; {seed_txt}]", fontsize=11)

    if audit is not None:
        flat = _flatten_dict(audit)
        z_overall = _safe_float(flat.get("counts_zero_frac__overall"))
        frac_lt_1 = _safe_float(flat.get("guide_lambda__frac_lt_1"))
        gini = _safe_float(flat.get("gene_total_lambda__gini"))
        parts = []
        if z_overall is not None:
            parts.append(f"zero_frac={z_overall:.3g}")
        if frac_lt_1 is not None:
            parts.append(f"frac(λ<1)={frac_lt_1:.3g}")
        if gini is not None:
            parts.append(f"gini={gini:.3g}")
        if parts:
            fig.text(0.5, 0.01, " | ".join(parts), ha="center", va="bottom", fontsize=9)

    return fig, out_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario audit plots for abundance regimes (rank-abundance + distributions).")
    parser.add_argument("--grid-tsv", required=True, type=str, help="count_depth_grid_summary.tsv from the grid runner.")
    parser.add_argument("--out-dir", required=True, type=str, help="Output directory for figures and TSV summary.")
    parser.add_argument("--max-scenarios", type=int, default=0, help="If >0, limit number of scenarios plotted (debug).")
    args = parser.parse_args()

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(str(args.grid_tsv), sep="\t")
    df = attach_scenarios(df)

    scenario_table = make_scenario_table(df)
    scenario_table = scenario_table.sort_values(["is_null", "scenario"], kind="mergesort").reset_index(drop=True)
    if int(args.max_scenarios) > 0:
        scenario_table = scenario_table.iloc[: int(args.max_scenarios)].copy()

    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = os.path.join(out_dir, "abundance_scenario_audit.pdf")
    tsv_path = os.path.join(out_dir, "abundance_scenario_audit.tsv")

    rows: list[dict[str, object]] = []
    with PdfPages(pdf_path) as pdf:
        for _, srow in scenario_table.iterrows():
            scenario_id = str(srow["scenario_id"])
            scenario_label = str(srow["scenario"])
            sub = df[df["scenario_id"].astype(str) == scenario_id].copy()
            rep = _pick_representative_row(sub)

            report_path = str(rep["report_path"])
            run_dir = os.path.dirname(report_path)

            audit_path = os.path.join(run_dir, "sim_abundance_audit.json")
            audit = _load_json(audit_path) if os.path.isfile(audit_path) else None
            truth_gene, truth_guide, truth_sample = _load_truth_tables(run_dir)

            seed = None
            if "seed" in rep.index:
                try:
                    seed = int(pd.to_numeric(rep["seed"], errors="coerce"))
                except Exception:
                    seed = None

            fig, metrics = _scenario_figure(
                scenario_label=scenario_label,
                scenario_id=scenario_id,
                seed=seed,
                truth_gene=truth_gene,
                truth_guide=truth_guide,
                truth_sample=truth_sample,
                audit=audit,
            )

            png_path = os.path.join(out_dir, f"scenario_{scenario_id}.png")
            _savefig(fig, png_path)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
            plt = _require_matplotlib()
            plt.close(fig)

            row: dict[str, object] = {
                "scenario_id": scenario_id,
                "scenario": scenario_label,
                "report_path": report_path,
                "seed": seed,
                "png_path": png_path,
            }
            if audit is not None:
                row.update(_flatten_dict(audit))
            row.update(metrics)
            rows.append(row)

    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    print(pdf_path)
    print(tsv_path)


if __name__ == "__main__":
    main()

