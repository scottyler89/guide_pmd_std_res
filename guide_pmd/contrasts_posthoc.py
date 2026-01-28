from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import gene_level as gene_level_mod
from . import pmd_std_res_guide_counts as guide_mod


def _sep_from_file_type(file_type: str) -> str:
    ft = str(file_type).lower().strip()
    if ft == "tsv":
        return "\t"
    if ft == "csv":
        return ","
    raise ValueError("file_type must be either 'tsv' or 'csv'")


def run_posthoc_contrasts(
    *,
    counts_file: str,
    std_res_file: str,
    model_matrix_file: str,
    output_dir: str,
    annotation_cols: int,
    pre_regress_vars: list[str] | None,
    file_type: str,
    contrasts: list[str],
    gene_level: bool,
    gene_id_col: int,
    gene_methods: list[str] | None,
    gene_out_dir: str | None,
    gene_progress: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sep = _sep_from_file_type(file_type)

    if pre_regress_vars is None:
        pre_regress_vars = []

    if not contrasts:
        raise ValueError("contrasts must not be empty")

    guides = pd.read_csv(counts_file, sep=sep, index_col=0)
    annotation_cols = int(annotation_cols)
    if annotation_cols < 1:
        raise ValueError("annotation_cols must be >= 1")
    annotation_table = guides.iloc[:, : (annotation_cols - 1)]
    if annotation_table.index.has_duplicates:
        raise ValueError("annotation_table index must not contain duplicates (guide_id)")

    std_res = pd.read_csv(std_res_file, sep="\t" if std_res_file.endswith(".tsv") else sep, index_col=0)
    std_res = std_res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if std_res.index.has_duplicates:
        raise ValueError("std_res index must not contain duplicates (guide_id)")
    if not annotation_table.index.isin(std_res.index).all():
        raise ValueError("std_res is missing one or more guide ids present in the input counts file")

    mm = pd.read_csv(model_matrix_file, sep=sep, index_col=0)
    mm = gene_level_mod._align_model_matrix(mm, list(std_res.columns))

    if len(pre_regress_vars) > 0:
        keep_cols = [c for c in mm.columns if c not in pre_regress_vars]
        if not all(var in mm.columns for var in pre_regress_vars):
            raise ValueError("Not all variables in pre_regress_vars are in the model matrix!")
        _, first_regressed = guide_mod.run_glm_analysis(std_res, mm[pre_regress_vars])
        response = first_regressed
        _, resids_df, contrasts_df = guide_mod.run_glm_analysis_with_contrasts(
            response,
            mm[keep_cols],
            contrasts=[str(c) for c in contrasts],
            add_intercept=False,
        )
    else:
        keep_cols = list(mm.columns)
        response = std_res
        _, resids_df, contrasts_df = guide_mod.run_glm_analysis_with_contrasts(
            response,
            mm[keep_cols],
            contrasts=[str(c) for c in contrasts],
        )

    out_path = os.path.join(output_dir, "PMD_std_res_stats_contrasts.tsv")
    contrasts_df.to_csv(out_path, sep="\t", index=False)

    if not gene_level:
        return

    # Gene-level contrast outputs are written under <out_dir>/gene_level by default.
    if gene_methods is None:
        gene_methods = ["meta", "stouffer", "qc", "flagged", "mixture", "tmeta"]
    gene_methods = [str(m) for m in gene_methods]
    if gene_out_dir is None:
        gene_out_dir = os.path.join(output_dir, "gene_level")

    if "lmm" in gene_methods:
        print("gene-level contrasts: skipping lmm (not implemented yet)", flush=True)

    gene_add_intercept = not (len(pre_regress_vars) > 0)
    gene_mm = mm[keep_cols]
    gene_mm_aligned = gene_level_mod._align_model_matrix(gene_mm, list(response.columns))

    per_guide_contrasts = gene_level_mod.fit_per_guide_contrasts(
        response,
        gene_mm_aligned,
        contrasts=[str(c) for c in contrasts],
        add_intercept=gene_add_intercept,
    )
    contrast_names = sorted(per_guide_contrasts["focal_var"].astype(str).unique().tolist())

    gene_meta_contrasts = None
    gene_qc_contrasts = None
    gene_flagged_contrasts = None

    needs_meta = any(m in gene_methods for m in ["meta", "flagged", "mixture", "tmeta"])
    if needs_meta:
        gene_meta_contrasts = gene_level_mod.compute_gene_meta(
            response,
            annotation_table,
            gene_mm,
            focal_vars=contrast_names,
            gene_id_col=gene_id_col,
            add_intercept=gene_add_intercept,
            per_guide_ols=per_guide_contrasts,
        )
        if "meta" in gene_methods:
            os.makedirs(gene_out_dir, exist_ok=True)
            out_path = os.path.join(gene_out_dir, "PMD_std_res_gene_meta_contrasts.tsv")
            gene_meta_contrasts.to_csv(out_path, sep="\t", index=False)

    if "stouffer" in gene_methods:
        from . import gene_level_stouffer as gene_level_stouffer_mod

        gene_stouffer_contrasts = gene_level_stouffer_mod.compute_gene_stouffer(
            response,
            annotation_table,
            gene_mm,
            focal_vars=contrast_names,
            gene_id_col=gene_id_col,
            add_intercept=gene_add_intercept,
            per_guide_ols=per_guide_contrasts,
        )
        os.makedirs(gene_out_dir, exist_ok=True)
        out_path = os.path.join(gene_out_dir, "PMD_std_res_gene_stouffer_contrasts.tsv")
        gene_stouffer_contrasts.to_csv(out_path, sep="\t", index=False)

    needs_qc = any(m in gene_methods for m in ["qc", "flagged", "mixture", "tmeta"])
    if needs_qc:
        from . import gene_level_qc as gene_level_qc_mod

        gene_qc_contrasts = gene_level_qc_mod.compute_gene_qc(
            response,
            annotation_table,
            gene_mm,
            focal_vars=contrast_names,
            gene_id_col=gene_id_col,
            add_intercept=gene_add_intercept,
            residual_matrix=resids_df,
            per_guide_ols=per_guide_contrasts,
        )
        if "qc" in gene_methods:
            os.makedirs(gene_out_dir, exist_ok=True)
            out_path = os.path.join(gene_out_dir, "PMD_std_res_gene_qc_contrasts.tsv")
            gene_qc_contrasts.to_csv(out_path, sep="\t", index=False)

    needs_flag = any(m in gene_methods for m in ["flagged", "mixture", "tmeta"])
    if needs_flag:
        from . import gene_level_flagging as gene_level_flagging_mod

        if gene_meta_contrasts is None:
            raise RuntimeError("internal error: gene_meta_contrasts is required for contrast flagging")
        if gene_qc_contrasts is None:
            raise RuntimeError("internal error: gene_qc_contrasts is required for contrast flagging")

        flag_cfg = gene_level_flagging_mod.GeneFlaggingConfig()
        flag_cfg.validate()
        gene_flagged_contrasts = gene_level_flagging_mod.compute_gene_flag_table(
            gene_meta_contrasts,
            gene_qc_contrasts,
            config=flag_cfg,
        )
        if "flagged" in gene_methods:
            os.makedirs(gene_out_dir, exist_ok=True)
            out_path = os.path.join(gene_out_dir, "PMD_std_res_gene_flagged_contrasts.tsv")
            gene_flagged_contrasts.to_csv(out_path, sep="\t", index=False)

    if "mixture" in gene_methods:
        from . import gene_level_mixture as gene_level_mixture_mod

        if gene_meta_contrasts is None or gene_flagged_contrasts is None:
            raise RuntimeError("internal error: missing meta/flag tables for contrast mixture")

        cfg = gene_level_mixture_mod.GeneMixtureConfig(scope="flagged")
        cfg.validate()
        gene_mixture_contrasts, gene_mixture_guides_contrasts = gene_level_mixture_mod.compute_gene_mixture(
            per_guide_contrasts,
            annotation_table,
            focal_vars=contrast_names,
            gene_id_col=gene_id_col,
            meta_results=gene_meta_contrasts,
            flag_table=gene_flagged_contrasts,
            config=cfg,
        )
        os.makedirs(gene_out_dir, exist_ok=True)
        out_path = os.path.join(gene_out_dir, "PMD_std_res_gene_mixture_contrasts.tsv")
        gene_mixture_contrasts.to_csv(out_path, sep="\t", index=False)
        out_path = os.path.join(gene_out_dir, "PMD_std_res_gene_mixture_guides_contrasts.tsv")
        gene_mixture_guides_contrasts.to_csv(out_path, sep="\t", index=False)

    if "tmeta" in gene_methods:
        from . import gene_level_tmeta as gene_level_tmeta_mod

        if gene_meta_contrasts is None or gene_flagged_contrasts is None:
            raise RuntimeError("internal error: missing meta/flag tables for contrast tmeta")

        cfg = gene_level_tmeta_mod.GeneTMetaConfig(scope="flagged")
        cfg.validate()
        gene_tmeta_contrasts, gene_tmeta_guides_contrasts = gene_level_tmeta_mod.compute_gene_tmeta(
            per_guide_contrasts,
            annotation_table,
            focal_vars=contrast_names,
            gene_id_col=gene_id_col,
            meta_results=gene_meta_contrasts,
            flag_table=gene_flagged_contrasts,
            config=cfg,
        )
        os.makedirs(gene_out_dir, exist_ok=True)
        out_path = os.path.join(gene_out_dir, "PMD_std_res_gene_tmeta_contrasts.tsv")
        gene_tmeta_contrasts.to_csv(out_path, sep="\t", index=False)
        out_path = os.path.join(gene_out_dir, "PMD_std_res_gene_tmeta_guides_contrasts.tsv")
        gene_tmeta_guides_contrasts.to_csv(out_path, sep="\t", index=False)

    if gene_progress:
        parts = []
        if gene_meta_contrasts is not None:
            parts.append(f"meta={gene_meta_contrasts.shape[0]}")
        if gene_qc_contrasts is not None:
            parts.append(f"qc={gene_qc_contrasts.shape[0]}")
        if gene_flagged_contrasts is not None:
            parts.append(f"flagged={int(gene_flagged_contrasts['flagged'].sum())}/{gene_flagged_contrasts.shape[0]}")
        if parts:
            print("gene-level contrast outputs:", ", ".join(parts), f"(dir={gene_out_dir})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute guide-level and gene-level contrast significance post hoc.")
    parser.add_argument("-in_file", "-i", type=str, required=True, help="Path to the original counts TSV/CSV file.")
    parser.add_argument("-out_dir", "-o", type=str, required=True, help="Output directory to write contrast TSVs.")
    parser.add_argument("-model_matrix_file", "-mm", type=str, required=True, help="Path to the model matrix TSV/CSV file.")
    parser.add_argument(
        "--std-res-file",
        dest="std_res_file",
        type=str,
        required=True,
        help="Path to a precomputed PMD_std_res.tsv (skips PMD bootstrap computation).",
    )
    parser.add_argument(
        "-annotation_cols",
        "-ann_cols",
        type=int,
        default=2,
        help="Number of annotation columns in the input file (including the ID/index column). Default=2",
    )
    parser.add_argument(
        "-pre_regress_vars",
        "-prv",
        type=str,
        nargs="+",
        default=None,
        help="Any variables to do full pre-regression rather than joint modeling.",
    )
    parser.add_argument("-file_type", type=str, default="tsv", help="tsv or csv (default: tsv)")
    parser.add_argument(
        "--contrast",
        dest="contrasts",
        type=str,
        nargs="+",
        required=True,
        help="Linear contrast expression(s), e.g. \"C1_high - C1_low\"",
    )
    parser.add_argument(
        "--gene-level",
        dest="gene_level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gene-level contrast outputs (default: enabled).",
    )
    parser.add_argument(
        "--gene-id-col",
        dest="gene_id_col",
        type=int,
        default=1,
        help="0-based column index in the original input file for the gene id (default=1; 0 is the guide id/index).",
    )
    parser.add_argument(
        "--gene-methods",
        dest="gene_methods",
        type=str,
        nargs="+",
        default=["meta", "stouffer", "qc", "flagged", "mixture", "tmeta"],
        help="Gene-level contrast methods to run (default: meta stouffer qc flagged mixture tmeta).",
    )
    parser.add_argument(
        "--gene-out-dir",
        dest="gene_out_dir",
        type=str,
        default=None,
        help="Optional output directory for gene-level contrast TSVs (default: <out_dir>/gene_level).",
    )
    parser.add_argument("--gene-progress", dest="gene_progress", action="store_true", help="Print a short contrast summary.")

    args = parser.parse_args()

    run_posthoc_contrasts(
        counts_file=args.in_file,
        std_res_file=args.std_res_file,
        model_matrix_file=args.model_matrix_file,
        output_dir=args.out_dir,
        annotation_cols=args.annotation_cols,
        pre_regress_vars=args.pre_regress_vars,
        file_type=args.file_type,
        contrasts=args.contrasts,
        gene_level=bool(args.gene_level),
        gene_id_col=int(args.gene_id_col),
        gene_methods=args.gene_methods,
        gene_out_dir=args.gene_out_dir,
        gene_progress=bool(args.gene_progress),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
