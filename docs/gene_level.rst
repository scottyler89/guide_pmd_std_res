Gene-Level Aggregation
======================

This repo includes **optional** gene-level outputs that complement the existing guide-level analysis.

**Back-compat contract**

- Baseline outputs (existing ``PMD_std_res*.tsv`` files) remain **byte-for-byte identical** for the same inputs.
- Gene-level outputs are **opt-in** and write to **new filenames** only.

Inputs
------

Gene-level methods use:

- the PMD standardized residual matrix (guides × samples)
- the guide annotation table (must include a gene identifier column)
- the model matrix (samples × covariates)

The focal covariate(s) are specified explicitly via ``--focal-vars`` (or ``focal_vars=[...]`` in the API).
The gene identifier is taken from the input annotation table column specified by ``--gene-id-col`` (default: 1,
the first non-index column in the input file).

Outputs are written with a deterministic sort order (``focal_var`` then ``gene_id``).

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
- heterogeneity diagnostics: ``tau``, ``tau2``, ``Q``, ``I2``
- QC: ``sign_agreement``, ``m_guides_total``, ``m_guides_used``

Plan A: Observation-level mixed model (``lmm``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fits a per-gene linear mixed model using all observations ``y_{guide,sample}``:

- Random intercept per guide (RI)
- Random slope per guide (RI+RS) when enough guides exist (see rubric/fallback fields)

Primary p-value is from a likelihood ratio test (LRT). If the LRT is numerically invalid,
``p_primary`` falls back to the Wald test and is explicitly labeled by ``p_primary_source``.

Output: ``PMD_std_res_gene_lmm.tsv``

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

If enabled, figures are written under ``gene_level_figures/`` with deterministic filenames.

Forest plots are generated only for gene ids explicitly provided via ``--gene-forest-genes``.

CLI usage
---------

Example (meta + mixed model + QC + figures):

.. code-block:: bash

   guide-pmd-std-res \
     -i counts.tsv \
     -o out/ \
     -mm model_matrix.tsv \
     --gene-level \
     --focal-vars treatment \
     --gene-methods meta lmm qc \
     --gene-figures \
     --gene-forest-genes A \
     --gene-progress

Notes
-----

- ``-n_boot`` must be ``>= 2`` (smaller values produce degenerate null standard deviations in PMD).
- A small synthetic audit harness is available at ``scripts/audit_gene_level.py``.
