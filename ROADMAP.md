# ROADMAP

Baseline tag: `baseline-2026-01-08` (commit `018cdd9` on `main`).
Current release tag: `v0.1.5`.

This roadmap tracks repo hygiene + modernization work for `guide_pmd_std_res`.

## Goals
- Make the CLI and Python API more robust (better input validation, fewer footguns).
- Add tests and CI so changes are safe to ship.
- Modernize packaging so installs/docs are reproducible.
- Add a strictly additive “stats add-on” layer (gene-level aggregation + diagnostics) **without changing any existing outputs or statistics**.

## Back-Compat Contract (non-negotiable)
- Baseline outputs (existing `PMD_std_res*.tsv` files) remain **byte-for-byte identical** for the same inputs.
- New analyses must be **opt-in** and write to **new filenames** (no extra columns added to existing outputs).
- Add golden/fixture tests to enforce the above.

## Work Plan

### P0 — Correctness / UX (start here)
- [x] Fix `pmd_std_res_and_stats()` returning undefined locals when `model_matrix_file=None`.
- [x] Fix `get_pmd_std_res()` to honor `sep` (currently always reads TSV).
- [x] Replace `raise "..."` with a real exception type.
- [x] Fix CLI arg typing/defaults (`annotation_cols` should be `int`).
- [x] Make `run_glm_analysis()` handle the “all-zero-variance features” case safely.
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
- [x] Ensure gene-level outputs are **strictly opt-in** (default execution path must not import/execute gene-level code).
- [x] Ensure gene-level outputs write **only new files** (never modifies baseline TSV schemas; no extra columns).
- [x] Add a “baseline-only” test to guarantee gene-level flags default to disabled.
- [x] Add a “gene-level-enabled” test to guarantee baseline outputs are still byte-identical + only new files are created.

#### P3.1 — Definitions (Estimand, Contrast, and Inputs)
- [ ] Explicitly define gene-level estimand(s) and map to model terms (Decision A in the plan doc).
- [ ] Define how users specify focal contrasts:
  - [ ] CLI: `--focal-vars` (one or more model-matrix columns)
  - [ ] API: `focal_vars=[...]`
- [ ] Decide whether to support:
  - [ ] all covariates (gene-level outputs for every model term), or
  - [ ] focal-only (recommended for now).
- [ ] Define “gene id” source:
  - [ ] default: second column in the input file (typical gene target),
  - [ ] configurable: `--gene-id-col` (0-based in original file; 0 is ID/index).
- [ ] Define a canonical long-table layout for modeling: `gene_id, guide_id, sample_id, y, (X...)`.
- [ ] Validate invariants: sample ordering, missing samples in model matrix, missing annotation fields, duplicates.

#### P3.2 — Data Plumbing (from current pipeline to add-on inputs)
- [ ] Implement conversion from wide standardized residuals (`std_res`) to long-form (gene/guide/sample rows).
- [ ] Add validation utilities:
  - [ ] check that model-matrix rows match `std_res` columns (and align/reindex deterministically)
  - [ ] check that `gene_id` exists for every guide row
- [ ] Implement a stable deterministic ordering for outputs (gene sort, then guide sort).

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

- [ ] Implement per-gene LMM with random intercepts and (when supported) random slopes for focal var(s) by guide:
  - [ ] base: RI (random intercept only)
  - [ ] preferred: RI+RS (random intercept + random slope)
- [ ] Fit strategy:
  - [ ] ML (`reml=False`) for LRT comparability
  - [ ] convergence handling + iteration caps + clear diagnostics
- [ ] Inference (Decision E):
  - [ ] primary p-value: LRT of `theta_g=0` (full vs null; preserve RE structure)
  - [ ] secondary: Wald z/p for reporting convenience
- [ ] Rubric / fallbacks (Decision C + rubric section of plan doc):
  - [ ] if `m_g < 3` -> RI only
  - [ ] if RI+RS singular/non-convergent -> RI
  - [ ] if RI fails -> fallback to Plan B
- [ ] Output spec:
  - [ ] `PMD_std_res_gene_lmm.tsv` (theta, SE, Wald z/p, LRT p, tau, sigma_alpha, model_used, converged, n_samples, m_guides)
- [ ] Multiple testing:
  - [ ] compute gene-level FDR for each focal var

#### P3.5 — Robustness / Contamination Handling (Plan C)
Goal: diagnostics first, robust methods as sensitivity / targeted follow-ups (Decision D).

- [ ] Compute discordance metrics per gene:
  - [ ] fraction opposite sign vs majority for focal var
  - [ ] dispersion of guide slopes
  - [ ] max |standardized residual| / influence proxy
- [ ] Provide robust effect summaries (QC; not primary p-values):
  - [ ] median of per-guide slopes
  - [ ] trimmed mean / winsorized mean
  - [ ] Huber M-estimator (optional)
- [ ] Define “flagged gene” criteria from rubric (thresholds + rationale + configurability).
- [ ] Optional targeted models for flagged genes:
  - [ ] heavy-tailed residual sensitivity (if feasible in chosen tooling)
  - [ ] 2-component mixture on guide slopes (good vs bad) with posterior guide weights
- [ ] Output spec:
  - [ ] `PMD_std_res_gene_qc.tsv`
  - [ ] optional: `PMD_std_res_gene_mixture.tsv`, `PMD_std_res_gene_guide_details.tsv`

#### P3.6 — Figures (Add-on only)
- [ ] Volcano plot per focal var (Plan A and Plan B; consistent axes + labeling).
- [ ] Plan A vs Plan B comparison scatter (effect size and -log10 p).
- [ ] Heterogeneity QC plots (tau vs effect; sign agreement vs p).
- [ ] Per-gene forest plot (per-guide slopes + SE) for flagged/top genes.
- [ ] Output directory + naming convention (e.g., `gene_level_figures/` with deterministic filenames).

#### P3.7 — CLI + API (opt-in; no baseline changes)
- [x] Add CLI flags (additive):
  - [x] `--gene-level` (enable)
  - [x] `--focal-vars ...`
  - [x] `--gene-id-col ...`
  - [x] `--gene-methods ...` (currently supports: `meta`)
  - [x] `--gene-out-dir ...`
- [x] Add Python API entry point(s) that can run gene-level analysis using in-memory `std_res` + model matrix.
- [x] Ensure baseline pipeline path is unchanged when `--gene-level` is not set.

#### P3.8 — Testing + Validation (focus on invariants, calibration, and regressions)
- [x] Unit tests for meta-analysis math (tau estimator; edge cases `m_g=1/2`, zero variance).
- [ ] Unit tests for LMM rubric behavior (RI vs RI+RS fallbacks).
- [x] Golden tests for baseline outputs (byte-for-byte).
- [ ] Cross-check tests: Plan A and Plan B agree on simple simulated data when assumptions match.
- [ ] Targeted simulation/audit harness (small; not “barrage of versions”) to sanity-check calibration and heterogeneity behavior.

#### P3.9 — Performance / UX
- [ ] Add progress reporting (genes processed, failures, fallbacks).
- [ ] Add optional parallelization (careful: determinism + stable sorting).
- [ ] Add caching of per-gene fits (optional; keyed by inputs) to speed iteration.

#### P3.10 — Docs + Release
- [ ] Add docs page describing Plans A/B/C, the rubric, and output file schemas.
- [ ] Update `README.md` with add-on usage examples.
- [ ] Version + changelog entry + tag once add-on ships.
