Gene-Level Aggregation
======================

This repo can generate gene-level outputs that complement the existing guide-level analysis.

**Back-compat contract**

- Baseline outputs (existing ``PMD_std_res*.tsv`` files) remain **byte-for-byte identical** for the same inputs.
- Gene-level outputs write to **new filenames** only (they do not modify baseline TSV schemas).

Inputs
------

Gene-level methods use:

- the PMD standardized residual matrix (guides × samples)
- the guide annotation table (must include a gene identifier column)
- the model matrix (samples × covariates)

The focal covariate(s) can be specified explicitly via ``--focal-vars`` (or ``focal_vars=[...]`` in the API).
If omitted, focal vars default to **all** model-matrix columns except ``Intercept``.
The gene identifier is taken from the input annotation table column specified by ``--gene-id-col`` (default: 1,
the first non-index column in the input file).

Outputs are written with a deterministic sort order (``focal_var`` then ``gene_id``).

Output locations
----------------

By default:

- Gene-level TSVs are written under ``<out_dir>/gene_level/``.
- Gene-level figures are written under ``<out_dir>/figures/gene_level/``.

Estimands
---------

For a given gene ``g`` and focal covariate ``T``:

- Plan A (``lmm``): ``theta`` is the fixed-effect coefficient for ``T`` in a per-gene mixed model with guide
  random intercepts (and random slopes for ``T`` when supported).
- Plan B (``meta``): ``theta`` is the random-effects meta-analysis mean of per-guide OLS slopes.
- Plan C (``qc``): robust location summaries of per-guide slopes (no primary p-values).

Methods
-------

Plan B: Random-effects meta-analysis (``meta``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gene-level effects are computed by:

1. Fitting per-guide OLS for each focal covariate.
2. Aggregating per-guide effects using a random-effects meta-analysis (DerSimonian–Laird).

Output: ``PMD_std_res_gene_meta.tsv``

Columns include (per gene × focal covariate):

- ``theta``, ``se_theta``, ``z``, ``p``, ``p_adj``
- heterogeneity diagnostics: ``tau``, ``tau2``, ``Q``, ``Q_df``, ``Q_p``, ``Q_p_adj``, ``I2``
- QC: ``sign_agreement``, ``m_guides_total``, ``m_guides_used``

Notes on heterogeneity
^^^^^^^^^^^^^^^^^^^^^^

The Cochran’s Q test p-value (``Q_p``) is computed when ``m_guides_used >= 2`` (df = ``m_guides_used - 1``).
It is calibrated under a fixed-effect null and is used here as a **screening/diagnostic** signal to identify genes
where guide-level effects appear unusually heterogeneous.

Plan A: Observation-level mixed model (``lmm``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fits a per-gene linear mixed model using all observations ``y_{guide,sample}``:

- Random intercept per guide (RI)
- Random slope per guide (RI+RS) when enough guides exist (see rubric/fallback fields)

This output includes **both**:

- the likelihood ratio test (LRT) p-value (``lrt_p`` / ``lrt_p_adj``), and
- the Wald test p-value (``wald_p`` / ``wald_p_adj``)

They are kept in separate columns so they cannot be conflated. If the LRT is numerically invalid,
``lrt_ok`` is ``False`` and ``lrt_p`` is ``NaN`` (the Wald columns are still reported when available).

Output: ``PMD_std_res_gene_lmm.tsv``

Plan A scope selection
^^^^^^^^^^^^^^^^^^^^^^

Plan A is **not** fit for every gene by default. Instead, the pipeline computes an explicit selection table and
fits the mixed model only for the selected (and feasible) ``(gene_id, focal_var)`` rows.

Output: ``PMD_std_res_gene_lmm_selection.tsv``

This table records:

- feasibility gates (e.g., non-identifiable focal var, degenerate response, insufficient guides)
- selection decisions and reasons (meta FDR, heterogeneity FDR, deterministic audit sampling)

This keeps the mixed-model layer statistically grounded, inspectable, and computationally bounded.

Plan C: Diagnostics / robust summaries (``qc``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Produces per-gene summary diagnostics and robust effect summaries derived from per-guide fits.

This layer **does not** apply hard thresholds or make binary calls (policy belongs in consumer/report layers).

Output: ``PMD_std_res_gene_qc.tsv``

Columns include robust summaries:

- ``beta_median``, ``beta_trimmed_mean``, ``beta_winsor_mean``
- ``beta_huber`` (Huber M-estimator; ``huber_c`` recorded; failures fall back to the median and are labeled by
  ``beta_huber_source``)

Figures
-------

If enabled, figures are written under ``<out_dir>/figures/gene_level/`` with deterministic filenames.

Forest plots are generated only for gene ids explicitly provided via ``--gene-forest-genes``.

CLI usage
---------

Example (meta + mixed model + QC + figures):

.. code-block:: bash

   guide-pmd-std-res \
     -in_file counts.tsv \
     -out_dir out/ \
     -model_matrix_file model_matrix.tsv \
     -annotation_cols 2 \
     --focal-vars treatment \
     --gene-forest-genes A \
     --gene-progress

Plan A checkpoint/resume (recommended for large runs):

.. code-block:: bash

   guide-pmd-std-res \
     -in_file counts.tsv \
     -out_dir out/ \
     -model_matrix_file model_matrix.tsv \
     -annotation_cols 2 \
     --std-res-file PMD_std_res.tsv \
     --gene-lmm-resume \
     --gene-lmm-checkpoint-every 200 \
     --gene-lmm-jobs 8 \
     --gene-progress

Notes
-----

- ``-n_boot`` must be ``>= 2`` (smaller values produce degenerate null standard deviations in PMD).
- A small synthetic audit harness is available at ``scripts/audit_gene_level.py``.
