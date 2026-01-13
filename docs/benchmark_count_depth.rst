Count-Depth Benchmark (Ground Truth)
====================================

This benchmark simulates **guide-level count data** under a ground-truth generative model and evaluates the downstream
gene-level methods (Plan B meta-analysis, Plan A LMM, and QC diagnostics) against known truth.

The main entry points are:

- ``scripts/benchmark_count_depth.py`` (single-run benchmark; writes a full artifact bundle + strict JSON report)
- ``scripts/run_count_depth_grid.py`` (parameter sweep runner; writes ``count_depth_grid_summary.tsv``)
- ``scripts/plot_count_depth_grid_summary.py`` (summary figures from the grid TSV)


Generative Model (High-Level)
-----------------------------

For each gene ``g`` with guides ``j=1..m`` and samples ``k``:

1. **Baseline abundance (gene→guide hierarchy)**

- ``log_lambda_gene_g ~ Normal(guide_lambda_log_mean, gene_lambda_log_sd)``
- ``log_lambda_guide_gj = log_lambda_gene_g + Normal(0, guide_lambda_log_sd)``
- ``lambda_base_gj = exp(log_lambda_guide_gj)``

2. **Sample depth + confounding**

- ``depth_k ~ LogNormal(depth_log_mean, depth_log_sd)``
- optional depth noise via ``depth_poisson_scale``
- optional treatment-depth confounding via ``treatment_depth_multiplier``
- optional batch structure (currently ``n_batches in {1,2}``):
  - treatment↔batch association controlled by ``batch_confounding_strength``
  - batch-level depth shifts via ``batch_depth_log_sd``

3. **Treatment effects (gene + guide heterogeneity + optional off-target)**

- Gene-level truth: a fraction ``frac_signal`` have ``theta_gene_g ~ Normal(0, effect_sd)``, otherwise ``theta_gene_g = 0``.
- On-target guide slope deviations are applied **only for signal genes**:
  - ``slope_dev_gj ~ Normal(0, guide_slope_sd)`` if ``is_signal_g`` else ``0``
- Optional off-target contamination (independent of ``is_signal_g``):
  - with probability ``offtarget_guide_frac``, add ``offtarget_dev_gj ~ Normal(0, offtarget_slope_sd)``

The per-guide treatment effect is:

- ``theta_guide_gj = theta_gene_g + slope_dev_gj + offtarget_dev_gj``

4. **Counts**

The mean count for guide ``gj`` in sample ``k`` is:

- ``mu_gjk = lambda_base_gj * depth_k * exp(theta_guide_gj * treatment_k)``

Counts are drawn as:

- Poisson when ``nb_overdispersion = 0``
- Gamma-Poisson (Negative Binomial overdispersion) when ``nb_overdispersion > 0``, with
  ``Var[count] = mu + nb_overdispersion * mu^2``.


Response Construction Modes
---------------------------

``scripts/benchmark_count_depth.py`` can construct the downstream response matrix in three ways:

- ``log_counts``: ``log(count + pseudocount)``
- ``guide_zscore_log_counts``: per-guide z-scored log-counts (fast surrogate)
- ``pmd_std_res``: PMD standardized residuals computed on the simulated counts

For PMD runs, prefer ``--pmd-n-boot >= 10`` for stability.


Outputs (Per Run)
-----------------

Each run writes a reproducible artifact bundle under ``--out-dir``:

- Inputs/truth:
  - ``sim_counts.tsv``
  - ``sim_model_matrix.tsv``
  - ``sim_std_res.tsv``
  - ``sim_truth_sample.tsv``
  - ``sim_truth_gene.tsv``
  - ``sim_truth_guide.tsv``
- Gene-level results:
  - ``PMD_std_res_gene_meta.tsv``
  - ``PMD_std_res_gene_lmm.tsv``
  - ``PMD_std_res_gene_lmm_selection.tsv`` (when ``--lmm-scope != all``)
  - ``PMD_std_res_gene_qc.tsv``
- Report + figures:
  - ``benchmark_report.json`` (strict JSON: no ``NaN``)
  - optional QQ plot PNGs under ``figures/`` (controlled by ``--qq-plots``)


Recommended Workflows
---------------------

Quick null calibration (PMD mode, depth confounding on/off):

.. code-block:: bash

   python scripts/run_count_depth_grid.py \
     --out-dir .tmp/pmd_proto/null_depth \
     --seeds 1 \
     --n-genes 500 \
     --response-mode pmd_std_res --pmd-n-boot 10 \
     --methods meta lmm qc \
     --lmm-scope meta_or_het_fdr --lmm-audit-n 50 --lmm-max-genes-per-focal-var 150 \
     --frac-signal 0.0 \
     --treatment-depth-multiplier 1.0 2.0 \
     --no-include-batch-covariate

Then plot summary figures:

.. code-block:: bash

   python scripts/plot_count_depth_grid_summary.py \
     --grid-tsv .tmp/pmd_proto/null_depth/count_depth_grid_summary.tsv \
     --out-dir  .tmp/pmd_proto/null_depth/fig_summary

Batch confounding stress test:

.. code-block:: bash

   python scripts/run_count_depth_grid.py \
     --out-dir .tmp/pmd_proto/null_batch \
     --seeds 1 \
     --n-genes 500 \
     --response-mode pmd_std_res --pmd-n-boot 10 \
     --methods meta lmm qc \
     --lmm-scope meta_or_het_fdr --lmm-audit-n 50 --lmm-max-genes-per-focal-var 150 \
     --frac-signal 0.0 \
     --n-batches 2 --batch-confounding-strength 1.0 --batch-depth-log-sd 0.7 \
     --treatment-depth-multiplier 1.0

Signal power check:

.. code-block:: bash

   python scripts/run_count_depth_grid.py \
     --out-dir .tmp/pmd_proto/signal_depth \
     --seeds 1 \
     --n-genes 500 \
     --response-mode pmd_std_res --pmd-n-boot 10 \
     --methods meta lmm qc \
     --lmm-scope meta_or_het_fdr --lmm-audit-n 0 --lmm-max-genes-per-focal-var 150 \
     --frac-signal 0.2 --effect-sd 0.5 \
     --treatment-depth-multiplier 2.0 \
     --no-include-batch-covariate

Notes:

- When ``--lmm-scope != all``, Plan A fits only a selected subset of genes; confusion-matrix metrics treat non-fit genes as “not called”.
- For full-method calibration/power (no selection), set ``--lmm-scope all`` (but this can be much slower).

