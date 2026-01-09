# ROADMAP

Baseline tag: `baseline-2026-01-08` (commit `018cdd9` on `main`).
Current release tag: `v0.1.5`.

This roadmap tracks repo hygiene + modernization work for `guide_pmd_std_res`.

## Goals
- Make the CLI and Python API more robust (better input validation, fewer footguns).
- Add tests and CI so changes are safe to ship.
- Modernize packaging so installs/docs are reproducible.
- Add a strictly additive gene-level layer (aggregation + diagnostics) **without changing any existing outputs or statistics**.

## Back-Compat Contract (non-negotiable)
- Baseline outputs (existing `PMD_std_res*.tsv` files) remain **byte-for-byte identical** for the same inputs.
- New analyses write to **new filenames** (no extra columns added to existing outputs).
- Add golden/fixture tests to enforce the above.

## Work Plan

### P0 — Correctness / UX (start here)
- [x] Fix `pmd_std_res_and_stats()` returning undefined locals when `model_matrix_file=None`.
- [x] Fix `get_pmd_std_res()` to honor `sep` (currently always reads TSV).
- [x] Replace `raise "..."` with a real exception type.
- [x] Fix CLI arg typing/defaults (`annotation_cols` should be `int`).
- [x] Make `run_glm_analysis()` handle the “all-zero-variance features” case safely.
- [x] Reject degenerate `n_boot < 2` (prevents invalid PMD z-scores).
- [x] Add minimal smoke tests for the above.

Back-compat notes:
- Outputs remain `*.tsv` (tab-separated) as before.
- CLI default `-p_combine_idx` remains `1` as before.

### P1 — Tests / CI
- [x] Add `pytest` test suite (`tests/`) for pure functions and CLI parsing.
- [x] Add `pytest.ini` to standardize local/CI runs.
- [x] Add GitHub Actions workflow to run unit tests + lint on PRs.
- [x] Add a lightweight style tool (`ruff`) and baseline config.
- [x] Enforce `ruff` in CI.

### P1 — Packaging Modernization
- [x] Add `pyproject.toml` (PEP 621 metadata) and migrate away from `setup.py`-only packaging (kept `setup.py` for back-compat).
- [x] Declare extras: `dev` (tests/lint), `docs` (sphinx theme), and pin minimum supported Python (>=3.10).
- [x] Add console entry point (so users can run without `python -m ...`).

### P2 — Docs / Release Hygiene
- [x] Fix/expand docs build config (ensure `sphinx_rtd_theme` is declared).
- [x] Add `LICENSE` file (README says MIT but none is present).
- [x] Add `CHANGELOG.md`.
- [x] Tag releases aligned to versions (`v0.1.5`).

### P3 — Statistical Add-On (Gene-Level Aggregation)
Primary objective: add gene-level inference that **complements** (and never replaces) guide-level results.

Reference spec: `docs/plans/gene_level_aggregation_plan.md`.

#### P3.0 — Non-negotiables (Back-Compat + IO)
- [x] Define the “baseline outputs set” (which files must remain byte-identical; starting at `v0.1.5`).
  - [x] Fixture `tests/fixtures/baseline_small`: `PMD_std_res.tsv`, `PMD_std_res_stats.tsv`, `PMD_std_res_stats_resids.tsv`
- [x] Add fixture input(s) + committed golden baseline outputs; CI fails if any baseline file differs byte-for-byte.
- [x] Ensure gene-level outputs are **strictly additive** (baseline TSV schemas never change; new outputs have new filenames).
- [x] Ensure gene-level outputs write **only new files** (never modifies baseline TSV schemas; no extra columns).
- [x] Add a “baseline + gene-level default” test: baseline outputs are still byte-identical + gene-level artifacts appear in stable locations.
- [x] Add an opt-out test to guarantee gene-level can be disabled and baseline outputs are still byte-identical.

#### P3.1 — Definitions (Estimand, Contrast, and Inputs)
- [x] Explicitly define gene-level estimand(s) and map to model terms (Decision A in the plan doc).
- [x] Define how users specify focal contrasts:
  - [x] CLI: `--focal-vars` (one or more model-matrix columns)
  - [x] API: `focal_vars=[...]`
- [x] Decide whether to support:
  - [x] all covariates (gene-level outputs for every model term), and/or
  - [x] focal-only (users can restrict via `--focal-vars`).
- [x] Define “gene id” source:
  - [x] default: second column in the input file (typical gene target),
  - [x] configurable: `--gene-id-col` (0-based in original file; 0 is ID/index).
- [x] Define a canonical long-table layout for modeling: `gene_id, guide_id, sample_id, y, (X...)`.
- [x] Validate invariants: sample ordering, missing samples in model matrix, missing annotation fields, duplicates.

#### P3.2 — Data Plumbing (from current pipeline to add-on inputs)
- [x] Implement conversion from wide standardized residuals (`std_res`) to long-form (gene/guide/sample rows).
- [x] Add validation utilities:
  - [x] check that model-matrix rows match `std_res` columns (and align/reindex deterministically)
  - [x] check that `gene_id` exists for every guide row
- [x] Implement a stable deterministic ordering for outputs (gene sort, then guide sort).

#### P3.3 — Plan B (Fast + cross-check): Summary-level random-effects meta-analysis
Goal: low-overhead gene-level inference using guide-level fits; also used as a validation layer for Plan A (Decision B option 2).

- [x] Implement a per-guide OLS (Gaussian) fitter that returns: `beta`, `SE`, `t`, `p` for each focal var (avoid touching baseline stats tables).
- [x] Implement per-gene random-effects meta model:
  - [x] `beta_gj ~ N(theta_g, se_gj^2 + tau_g^2)`
  - [x] choose tau estimation method (start: DerSimonian–Laird; later: REML/EB).
- [x] Report heterogeneity diagnostics (Decision F scaffolding):
  - [x] `tau`, Cochran’s Q, I², guide sign-agreement fraction
- [x] Multiple testing:
  - [x] compute gene-level FDR for each focal var
- [x] Output spec (new file; stable schema + sorting):
  - [x] `PMD_std_res_gene_meta.tsv`

#### P3.4 — Plan A (Primary): Observation-level mixed model per gene (RI / RI+RS) + LRT
Goal: likelihood-based gene-level inference using all `y_{gjk}` (Decision B option 1; Decision C RI+RS).

- [x] Implement per-gene LMM with random intercepts and (when supported) random slopes for focal var(s) by guide:
  - [x] base: RI (random intercept only)
  - [x] preferred: RI+RS (random intercept + random slope)
- [x] Fit strategy:
  - [x] ML (`reml=False`) for LRT comparability
  - [x] convergence handling + iteration caps + clear diagnostics
- [x] Inference (Decision E):
  - [x] report LRT and Wald in separate columns (`lrt_*` vs `wald_*`)
  - [x] mark numerical validity explicitly (`lrt_ok`, `wald_ok`)
- [x] Rubric / fallbacks (Decision C + rubric section of plan doc):
  - [x] if `m_g < 3` -> RI only (configurable via `min_guides_random_slope`)
  - [x] if RI+RS singular/non-convergent -> RI
  - [x] if RI fails -> fallback to Plan B (explicitly labeled as `meta_fallback`)
- [x] Output spec:
  - [x] `PMD_std_res_gene_lmm.tsv` (theta, SE, Wald z/p, LRT p, tau, sigma_alpha, model_used, converged, n_samples, m_guides)
- [x] Multiple testing:
  - [x] compute gene-level FDR for each focal var

#### P3.4.1 — Plan A Scope Selection (Statistically grounded; default-on)
Goal: run Plan A where it is identifiable and informative, without wasting compute or producing misleading fits.

Principles (from `DEV_RUBRIC.md`):
- Core analysis code must not hide heuristics; policy lives in consumer/orchestration layers and must be explicit + configurable.
- Selection must be statistically grounded and inspectable (FDR control, formal tests, deterministic audit sampling).

Phase A — Add the missing statistics needed for principled selection
- [x] Add Cochran’s Q heterogeneity p-value to meta output (`Q_p`) with df=`m_guides_used-1` when `m_guides_used>=2`.
- [x] Add FDR-adjusted heterogeneity p-value per focal var (`Q_p_adj`).
- [x] Document interpretation + limitations (Q-test is calibrated under fixed-effect null; still useful as a screening metric).

Phase B — Define explicit feasibility gates (identifiability, not heuristics)
- [x] Implement design-matrix rank checks per focal var (skip if not identifiable).
- [x] Implement response degeneracy checks (skip if gene has all-zero variance across observations).
- [x] Implement minimum guide count gate for any mixed model (`m_guides >= 2`).
- [x] Keep these gates in shared utilities and surface the reason in outputs (no silent skipping).

Phase C — Selection policy (FDR-driven + deterministic audit)
- [x] Implement a selection config object (e.g., `GeneLmmSelectionConfig`) with explicit defaults:
  - [x] scope: `all` | `meta_fdr` | `meta_or_het_fdr` | `explicit` | `none`
  - [x] meta FDR threshold `q_meta` (per focal var; uses Plan B `p_adj`)
  - [x] heterogeneity FDR threshold `q_het` (per focal var; uses `Q_p_adj`)
  - [x] deterministic audit sample: `audit_n`, `audit_seed`
  - [x] optional compute cap: `max_genes_per_focal_var` (explicit budget; off by default)
- [x] Implement deterministic audit sampling from the complement set to monitor calibration and fit stability.
- [x] Ensure selection is reproducible (seeded RNG, stable sorting).

Phase D — Make selection inspectable via additive artifacts
- [ ] Add `PMD_std_res_gene_lmm_selection.tsv` with per `(gene_id, focal_var)`:
  - [ ] `selected` (bool), `selection_reason` (enum), and `skip_reason` (enum)
  - [ ] key inputs used for the decision (e.g., `meta_p_adj`, `Q_p_adj`, `m_guides_used`)
- [ ] Update gene-level progress reporting to include selection counts and fit outcome counts.

Phase E — Run Plan A only on the selected set (still default-on)
- [ ] Add `genes_to_fit` / `selection_table` support so LMM fitting does not iterate over all genes unnecessarily.
- [ ] Run Plan A per focal var on selected genes only; concatenate results with stable sorting.
- [ ] Preserve current explicit failure behavior (`meta_fallback` on fit failure; no silent fallbacks).

Phase F — Tests + real-data regression
- [ ] Unit tests for `Q_p` / `Q_p_adj` (edge cases: `m=0/1/2`, zero variance).
- [ ] Unit tests for selection determinism + reason labeling.
- [ ] Integration test: default selection produces `PMD_std_res_gene_lmm_selection.tsv` and does not change baseline TSV bytes.
- [ ] Add a local, non-committed st941c prototype runner doc/snippet that uses `--std-res-file` to validate behavior quickly.

#### P3.5 — Robustness / Contamination Handling (Plan C)
Goal: diagnostics first, robust methods as sensitivity / targeted follow-ups (Decision D).

- [x] Compute discordance metrics per gene:
  - [x] fraction opposite sign vs majority for focal var
  - [x] dispersion of guide slopes
  - [x] max |standardized residual| / influence proxy
- [x] Provide robust effect summaries (QC; not primary p-values):
  - [x] median of per-guide slopes
  - [x] trimmed mean / winsorized mean
  - [x] Huber M-estimator (optional)
- [ ] Define “flagged gene” criteria from rubric (thresholds + rationale + configurability).
- [ ] Optional targeted models for flagged genes:
  - [ ] heavy-tailed residual sensitivity (if feasible in chosen tooling)
  - [ ] 2-component mixture on guide slopes (good vs bad) with posterior guide weights
- [x] Output spec:
  - [x] `PMD_std_res_gene_qc.tsv`
  - [ ] optional: `PMD_std_res_gene_mixture.tsv`, `PMD_std_res_gene_guide_details.tsv`

#### P3.6 — Figures (Add-on only)
- [x] Volcano plot per focal var (Plan A and Plan B; consistent axes + labeling).
- [x] Plan A vs Plan B comparison scatter (effect size and -log10 p).
- [x] Heterogeneity QC plots (tau vs effect; sign agreement vs p).
- [x] Per-gene forest plot (per-guide slopes + SE) for explicit genes (no implicit top-N heuristic).
- [x] Output directory + naming convention (e.g., `figures/gene_level/` with deterministic filenames).

#### P3.7 — CLI + API (additive; no baseline changes)
- [x] Add CLI flags (additive; supports opt-out):
  - [x] `--gene-level` / `--no-gene-level`
  - [x] `--focal-vars ...`
  - [x] `--gene-id-col ...`
  - [x] `--gene-methods ...` (currently supports: `meta`, `lmm`, `qc`)
  - [x] `--gene-out-dir ...`
  - [x] `--gene-figures` / `--no-gene-figures`, `--gene-figures-dir`, and `--gene-forest-genes`
- [x] Add Python API entry point(s) that can run gene-level analysis using in-memory `std_res` + model matrix.
- [x] Ensure baseline pipeline path is unchanged when `--gene-level` is not set.

#### P3.8 — Testing + Validation (focus on invariants, calibration, and regressions)
- [x] Unit tests for meta-analysis math (tau estimator; edge cases `m_g=1/2`, zero variance).
- [x] Unit tests for LMM rubric behavior (RI vs RI+RS fallbacks).
- [x] Golden tests for baseline outputs (byte-for-byte).
- [x] Cross-check tests: Plan A and Plan B agree on simple simulated data when assumptions match.
- [x] Targeted simulation/audit harness (small; not “barrage of versions”) to sanity-check calibration and heterogeneity behavior.

#### P3.9 — Performance / UX
- [x] Add progress reporting (genes processed, failures, fallbacks).
- [ ] Add optional parallelization (careful: determinism + stable sorting).
- [ ] Add caching of per-gene fits (optional; keyed by inputs) to speed iteration.

#### P3.10 — Docs + Release
- [x] Add docs page describing Plans A/B/C, the rubric, and output file schemas.
- [x] Update `README.md` with gene-level usage examples.
- [ ] Version + changelog entry + tag once add-on ships.
