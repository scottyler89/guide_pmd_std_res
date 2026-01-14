from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def _maybe_read_tsv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t")
    return df if not df.empty else df


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(
        description="Write gene-level figures from precomputed gene-level TSVs (no recomputation)."
    )
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
        "--figures-dir",
        default=None,
        type=str,
        help="Optional output directory for figures (default: <out-dir>/figures/gene_level).",
    )
    parser.add_argument(
        "--prefix",
        default="PMD_std_res",
        type=str,
        help="Filename prefix used for figures (default: PMD_std_res).",
    )
    parser.add_argument(
        "--agreement-q",
        default=0.1,
        type=float,
        help="FDR threshold for agreement/disagreement figures (default: 0.1).",
    )
    args = parser.parse_args()

    out_dir = str(args.out_dir)
    gene_level_dir = str(args.gene_level_dir or os.path.join(out_dir, "gene_level"))
    figures_dir = str(args.figures_dir or os.path.join(out_dir, "figures", "gene_level"))

    meta_path = os.path.join(gene_level_dir, "PMD_std_res_gene_meta.tsv")
    lmm_path = os.path.join(gene_level_dir, "PMD_std_res_gene_lmm.tsv")
    qc_path = os.path.join(gene_level_dir, "PMD_std_res_gene_qc.tsv")

    gene_meta = _maybe_read_tsv(meta_path)
    gene_lmm = _maybe_read_tsv(lmm_path)
    gene_qc = _maybe_read_tsv(qc_path)

    import guide_pmd.gene_level_figures as gene_level_figures_mod

    written = gene_level_figures_mod.write_gene_level_figures(
        figures_dir,
        prefix=str(args.prefix),
        gene_meta=gene_meta,
        gene_lmm=gene_lmm,
        gene_qc=gene_qc,
        agreement_q=float(args.agreement_q),
    )
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
