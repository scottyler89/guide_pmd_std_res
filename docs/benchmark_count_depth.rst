Count-Depth Benchmark (Ground Truth)
====================================

This benchmark simulates **guide-level count data** under a ground-truth generative model and evaluates the downstream
gene-level methods (Plan B meta-analysis, Plan A LMM, and QC diagnostics) against known truth.

The main entry points are:

- ``scripts/benchmark_count_depth.py`` (single-run benchmark; writes a full artifact bundle + strict JSON report)
- ``scripts/run_count_depth_grid.py`` (parameter sweep runner; writes ``count_depth_grid_summary.tsv``)
- ``scripts/plot_count_depth_grid_summary.py`` (summary figures from the grid TSV)
- ``scripts/aggregate_count_depth_grid_summary.py`` (aggregate grid results across seeds)


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

Calibration Expectations & Interpretation
----------------------------------------

This benchmark is designed to answer two different questions:

1) **Statistical calibration**: are p-values approximately Uniform(0,1) under a *true null*?
2) **Practical robustness**: what happens when real-world assumptions are violated (confounding, contamination, overdispersion)?

Use the settings below to make those questions explicit.

True-null calibration runs (what "should" be uniform)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For null calibration, use a configuration that makes the gene-level truth genuinely null:

- ``--frac-signal 0``
- ``--offtarget-guide-frac 0`` (no contamination)
- ``--treatment-depth-multiplier 1`` (no depth confounding) **or** include ``--include-depth-covariate``
- ``--n-batches 1`` **or** include ``--include-batch-covariate``
- ``--nb-overdispersion 0`` (Poisson)

Under these settings, you should expect:

- ``lambda_gc`` near 1.0 for null p-values (QQ plots / QQ stats)
- empirical FPR near ``alpha`` (see the ``confusion_alpha`` block in ``benchmark_report.json``)

Confounding stress tests (what will inflate FPR if you omit covariates)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Depth confounding:

- Set ``--treatment-depth-multiplier 2`` and run with/without ``--include-depth-covariate``.
- Expect inflated null FPR without the covariate, and improved calibration when it is included.

Batch confounding:

- Set ``--n-batches 2 --batch-confounding-strength 1 --batch-depth-log-sd 0.5`` and run with/without
  ``--include-batch-covariate``.
- Expect inflated null FPR without batch indicators, and improved calibration when they are included.

Contamination runs (off-target guides)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``--offtarget-guide-frac > 0``, some guides have a nonzero treatment effect even when ``theta_gene=0``.
In this repo the primary gene-level truth label is:

- ``is_signal`` ⇢ **gene-level effect present** (``theta_gene != 0``)

so under contamination you should interpret “false positives” as “detections driven by off-target behavior”
rather than a failure of multiple testing per se. For pure calibration, keep off-target disabled.

Overdispersion (Negative Binomial)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``--nb-overdispersion > 0``, counts follow a Gamma-Poisson mixture:

- ``Var[count] = mu + nb_overdispersion * mu^2``

This is a realistic stress test for PMD and downstream Gaussian modeling. Use it to compare robustness across methods
and to see how much overdispersion breaks calibration under each response mode.

Response mode notes
^^^^^^^^^^^^^^^^^^^

- ``pmd_std_res`` is the intended mode for calibration/power benchmarking of downstream Gaussian methods.
- ``log_counts`` and ``guide_zscore_log_counts`` are faster stress-test modes; do not assume strict null calibration.
- For PMD mode, prefer ``--pmd-n-boot >= 10`` (and consider 20+) for stability.

Plan A selection policy (important for interpreting TPR/FPR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``--lmm-scope`` is not ``all``, Plan A (LMM) is fit only on a selected subset of genes. In that case:

- confusion matrices treat **non-fit genes as “not called”**
- reported TPR/FPR reflect the *pipeline policy* (selection + modeling), not the LMM fit in isolation

To benchmark the LMM method itself (without selection), set ``--lmm-scope all``.

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

Optionally aggregate across seeds (useful when ``--seeds`` has multiple values):

.. code-block:: bash

   python scripts/aggregate_count_depth_grid_summary.py \
     --grid-tsv .tmp/pmd_proto/null_depth/count_depth_grid_summary.tsv

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

Selection policy tradeoffs (power vs runtime)
--------------------------------------------

To evaluate Plan A selection-policy tradeoffs, sweep ``--lmm-scope`` and ``--lmm-max-genes-per-focal-var`` (and optionally
``--lmm-audit-n``) while holding the generative scenario fixed.

Example (signal scenario; compares full LMM vs selected/capped LMM):

.. code-block:: bash

   python scripts/run_count_depth_grid.py \
     --out-dir .tmp/pmd_proto/selection_tradeoff \
     --seeds 1 \
     --n-genes 200 \
     --response-mode pmd_std_res --pmd-n-boot 10 \
     --methods meta lmm \
     --frac-signal 0.2 --effect-sd 0.5 \
     --treatment-depth-multiplier 2.0 \
     --include-depth-covariate \
     --lmm-scope all meta_or_het_fdr \
     --lmm-audit-n 0 50 \
     --lmm-max-genes-per-focal-var 0 100

Then generate the tradeoff plots (including selection-runtime/power vs cap):

.. code-block:: bash

   python scripts/plot_count_depth_grid_summary.py \
     --grid-tsv .tmp/pmd_proto/selection_tradeoff/count_depth_grid_summary.tsv \
     --out-dir  .tmp/pmd_proto/selection_tradeoff/fig_summary

Off-target contamination stress test
-----------------------------------

To stress robustness to guide contamination, sweep ``offtarget_guide_frac`` and ``offtarget_slope_sd``.

Example (gene-level null, but with off-target guides present):

.. code-block:: bash

   python scripts/run_count_depth_grid.py \
     --out-dir .tmp/pmd_proto/offtarget_null \
     --seeds 1 \
     --n-genes 500 \
     --response-mode pmd_std_res --pmd-n-boot 10 \
     --methods meta lmm qc \
     --frac-signal 0.0 \
     --offtarget-guide-frac 0.00 0.05 0.10 \
     --offtarget-slope-sd 0.00 0.20 0.50 \
     --treatment-depth-multiplier 1.0 \
     --include-depth-covariate \
     --lmm-scope meta_or_het_fdr --lmm-audit-n 50 --lmm-max-genes-per-focal-var 150

Interpretation reminder: in this scenario, ``is_signal`` is still defined by the *gene-level* effect (``theta_gene != 0``).
Detections driven by off-target guides will appear as “false positives” in the confusion matrix relative to ``is_signal``.

Notes:

- When ``--lmm-scope != all``, Plan A fits only a selected subset of genes; confusion-matrix metrics treat non-fit genes as “not called”.
- For full-method calibration/power (no selection), set ``--lmm-scope all`` (but this can be much slower).
