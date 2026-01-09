
# PMD Standardized Residuals Analysis Script

This Python script performs **Percent Maximum Difference (PMD)** standardized residuals analysis upstream of a Generalized Linear Model (GLM) specified by a user-provided model matrix. PMD is a robust method for analyzing differential abundance in count data and related datasets. This script processes input data, applies the PMD analysis pipeline, and outputs the results.

---

## Background

PMD, introduced in [bioRxiv 2021.11.15.468733v2](https://www.biorxiv.org/content/10.1101/2021.11.15.468733v2.full), is a statistical approach for analyzing differential abundance, with applications to count-based data (e.g., single-cell data, RNA sequencing). The use of standardized residuals in such analyses is further explored in [Nature Communications](https://www.nature.com/articles/s41467-023-43406-9).

This script calculates standardized residuals using PMD and performs statistical evaluations of differential abundance or other effects based on the provided model matrix.

---

## Installation
This script can be easily installed with pip:

`python -m pip install git+https://github.com/scottyler89/guide_pmd_std_res`

---
## Usage

### Command-Line Arguments

| Argument                  | Type    | Description                                                                                                                                          | Default      |
|---------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| `-in_file`               | `str`   | Path to the input TSV file.                                                                                                                          | **Required** |
| `-out_dir`               | `str`   | Path to the desired output directory.                                                                                                                | **Required** |
| `-model_matrix_file`     | `str`   | Path to the input model matrix TSV file.                                                                                                             | `None`       |
| `-pre_regress_vars`      | `list`  | Variables for pre-regression rather than joint modeling.                                                                                             | `[]`         |
| `-annotation_cols`       | `str`   | Number of annotation columns in the input file. The first column is treated as a unique ID (e.g., guide ID), followed by additional annotations.      | `2`          |
| `-p_combine_idx`         | `str`   | Zero-based column index for combining p-values using Stouffer's Method when variables have multiple measures across rows.                             | `None`       |
| `-n_boot`                | `int`   | Number of bootstrap shuffled nulls to run (must be >= 2).                                                                                            | `100`        |
| `-seed`                  | `int`   | Random seed for reproducibility.                                                                                                                     | `123456`     |
| `-file_type`             | `str`   | File type of the input data (`tsv` or `csv`).                                                                                                        | `tsv`        |
| `--gene-level`           | flag    | Enable **gene-level aggregation outputs** (opt-in; baseline outputs remain unchanged).                                                               | `False`      |
| `--focal-vars`           | `list`  | One or more model-matrix column names to compute gene-level effects for (**required** when `--gene-level`).                                          | `None`       |
| `--gene-id-col`          | `int`   | 0-based column index in the original input file for the gene id (0 is the guide id/index).                                                          | `1`          |
| `--gene-methods`         | `list`  | Gene-level methods to run (currently supports: `meta`, `lmm`, `qc`).                                                                                  | `["meta"]`   |
| `--gene-out-dir`         | `str`   | Optional output directory for gene-level files (default: same as `-out_dir`).                                                                        | `None`       |
| `--gene-figures`         | flag    | Generate gene-level figures (requires `matplotlib`).                                                                                                 | `False`      |
| `--gene-figures-dir`     | `str`   | Optional output directory for gene-level figures (default: `<gene_out_dir>/gene_level_figures`).                                                     | `None`       |

---

### Example Usage

```bash
python -m guide_pmd.pmd_std_res_guide_counts \
    -in_file data/input_data.tsv \
    -out_dir results/ \
    -model_matrix_file data/model_matrix.tsv \
    -annotation_cols 2 \
    -pre_regress_vars ["day"] \
    -p_combine_idx 2 \
    -n_boot 100 \
    -seed 123456 \
    -file_type tsv \
    --gene-level \
    --focal-vars treatment \
    --gene-methods lmm meta qc \
    --gene-figures
```

---

## Output

The script produces the following results:

1. **Standardized Residuals** (`std_res`): Residuals from the PMD analysis.
2. **Statistical Results** (`stats_res`): Statistical summaries for the analyzed variables.
3. **Residuals DataFrame** (`resids_df`): Detailed residuals per data point.
4. **Combined Statistics** (`comb_stats`): Aggregated statistics when combining p-values using Stouffer's method.

All outputs are saved in the specified output directory.

When gene-level outputs are enabled, additional files are created with new filenames (e.g., `PMD_std_res_gene_meta.tsv`, `PMD_std_res_gene_lmm.tsv`, `PMD_std_res_gene_qc.tsv`) and **no existing baseline files are modified**.

When `--gene-figures` is enabled, figures are written to `gene_level_figures/` under the gene output directory.

---

## Functionality Overview

The script wraps the `pmd_std_res_and_stats` function, which performs:

1. Reading and preprocessing of the input data.
2. Application of PMD residuals analysis with optional bootstrapping.
3. Optional handling of annotation columns for unique identifiers.
4. p-value combination across rows (if applicable).
5. Outputting the results.

---

## Citation

If you use this script, please cite the relevant PMD papers:

1. *PMD: Percent Maximum Difference* ([bioRxiv, 2021](https://www.biorxiv.org/content/10.1101/2021.11.15.468733v2.full))
2. *Standardized residuals in differential abundance analysis* ([Nature Communications, 2023](https://www.nature.com/articles/s41467-023-43406-9))

---

## License

This script is distributed under the MIT License.
