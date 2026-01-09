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
Primary objective: add gene-level inference that complements (and never replaces) guide-level results.

**Back-compat enforcement**
- [ ] Add a small fixture dataset + golden outputs for the current pipeline; fail CI if any baseline output changes.
- [ ] Add a `--no-addon`/default mode test ensuring new code paths are never executed unless requested.

**Plan A (primary): observation-level mixed model per gene**
- [ ] Implement gene-level LMM `RI+RS` (random intercept + random slope for focal contrast) on the guide×sample standardized residuals.
- [ ] Use ML (`reml=False`) and compute both Wald summary + LRT p-value for the focal effect (`theta_g`).
- [ ] Implement the rubric/fallbacks: `m_g < 3 -> RI`, failed fit/singular -> fallback to RI, then to Plan B.
- [ ] Output: `PMD_std_res_gene_lmm.tsv` (theta, SE, Wald z/p, LRT p, tau, sigma_alpha, model_used, converged).

**Plan B (validation + fast mode): summary-level random-effects meta**
- [ ] Reuse per-guide GLM fits to extract per-guide beta/SE for focal contrasts (do not modify existing stats files).
- [ ] Fit random-effects meta model per gene: `beta_gj ~ N(theta_g, se_gj^2 + tau_g^2)`; report theta/SE/p and tau.
- [ ] Output: `PMD_std_res_gene_meta.tsv` (theta, SE, z/p, tau, m_guides).

**Plan C (targeted robustness + diagnostics)**
- [ ] Compute discordance metrics per gene (sign agreement, slope dispersion, outlier guide influence).
- [ ] Provide robust effect summaries (median/trimmed mean of guide slopes) as QC (not primary p-values).
- [ ] Optional: “good vs bad guide” mixture for flagged genes; output posterior guide weights.
- [ ] Output: `PMD_std_res_gene_qc.tsv` (+ optional `PMD_std_res_gene_mixture.tsv`).

**Figures (add-on)**
- [ ] Volcano plot for focal contrast (Plan A + Plan B side-by-side).
- [ ] Heterogeneity/discordance QC plots (tau vs effect, sign agreement vs p).
- [ ] Flagged-gene per-guide forest plots (effect + SE per guide).

**CLI + docs**
- [ ] Add new CLI flags to enable add-on and choose focal contrast(s) (e.g., `--gene-addon`, `--focal-vars`, `--methods lmm,meta`).
- [ ] Document the add-on estimands, outputs, and rubric in `docs/` (and link from `README.md`).
