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

### P3 — Gene-Level Aggregation (Complementary Statistics)
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

#### P3.2 — Data Plumbing (from current pipeline to gene-level inputs)
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

#### P3.3.1 — Historical combined-statistics (Stouffer; continuity + benchmark coverage)
- [x] Implement Stouffer-style combined statistics from per-guide OLS t-values (additive; no baseline changes).
- [x] Output spec (new file; stable schema + sorting): `PMD_std_res_gene_stouffer.tsv`

#### P3.4 — Plan A (Primary): Observation-level mixed model per gene (RI / RI+RS) + LRT
Goal: likelihood-based gene-level inference using all `y_{gjk}` (Decision B option 1; Decision C RI+RS).

- [x] Implement per-gene LMM with random intercepts and (when supported) random slopes for focal var(s) by guide:
  - [x] base: RI (random intercept only)
  - [x] preferred: RI+RS (random intercept + random slope)
- [x] Fit strategy:
  - [x] ML (`reml=False`) for LRT comparability
  - [x] convergence handling + iteration caps + clear diagnostics
  - [x] Robust optimizer fallback stack to avoid statsmodels `llf=inf` boundary + `Singular matrix` crashes (`lbfgs` → `bfgs` → `cg` → `powell` → `nm`)
- [x] Inference (Decision E):
  - [x] report LRT and Wald in separate columns (`lrt_*` vs `wald_*`)
  - [x] mark numerical validity explicitly (`lrt_ok`, `wald_ok`)
  - [x] record raw/adjusted LR stat (`lrt_stat_raw`, `lrt_stat`) + clipping flag (`lrt_clipped`) to diagnose negative LR stats
  - [x] record optimizer used for each fit (`optimizer_full`, `optimizer_null`)
- [x] Rubric / fallbacks (Decision C + rubric section of plan doc):
  - [x] if `m_g < 3` -> RI only (configurable via `min_guides_random_slope`)
  - [x] if RI+RS singular/non-convergent -> RI
  - [x] if RI fails -> fallback to Plan B (explicitly labeled as `meta_fallback`)
- [x] Output spec:
  - [x] `PMD_std_res_gene_lmm.tsv` (theta, SE, Wald z/p, LRT p, tau, sigma_alpha, model_used, converged, n_samples, m_guides)
  - [x] `PMD_std_res_gene_lmm_full.tsv` (one row per gene×focal-var; includes `meta_*` + selection fields + `lmm_*` where fit)
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
- [x] Add `PMD_std_res_gene_lmm_selection.tsv` with per `(gene_id, focal_var)`:
  - [x] `selected` (bool), `selection_reason` (enum), and `skip_reason` (enum)
  - [x] key inputs used for the decision (e.g., `meta_p_adj`, `Q_p_adj`, `m_guides_used`)
- [x] Update gene-level progress reporting to include selection counts and fit outcome counts.

Phase E — Run Plan A only on the selected set (still default-on)
- [x] Add `genes_to_fit` / `selection_table` support so LMM fitting does not iterate over all genes unnecessarily.
- [x] Run Plan A per focal var on selected genes only; concatenate results with stable sorting.
- [x] Preserve current explicit failure behavior (`meta_fallback` on fit failure; no silent fallbacks).

Phase F — Tests + real-data regression
- [x] Unit tests for `Q_p` / `Q_p_adj` (edge cases: `m=0/1/2`, zero variance).
- [x] Unit tests for selection determinism + reason labeling.
- [x] Integration test: default selection produces `PMD_std_res_gene_lmm_selection.tsv` and does not change baseline TSV bytes.
- [x] Add a local st941c prototype runner doc/snippet that uses `--std-res-file` to validate behavior quickly (`docs/plans/st941c_prototype_run.md`).

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
- [x] Define “flagged gene” criteria from rubric (thresholds + rationale + configurability).
- [x] Optional targeted models for flagged genes:
  - [x] heavy-tailed sensitivity on per-guide slopes (t-meta IRLS; weights returned per guide)
  - [x] 2-component mixture on per-guide slopes (good vs bad) with posterior guide weights
- [x] Output spec:
  - [x] `PMD_std_res_gene_qc.tsv`
  - [x] optional: `PMD_std_res_gene_flagged.tsv`
  - [x] optional: `PMD_std_res_gene_mixture.tsv`, `PMD_std_res_gene_mixture_guides.tsv`
  - [x] optional: `PMD_std_res_gene_tmeta.tsv`, `PMD_std_res_gene_tmeta_guides.tsv`

#### P3.6 — Figures (Gene-level only)
- [x] Volcano plot per focal var (Plan A and Plan B; consistent axes + labeling).
- [x] Stouffer volcano and p-value comparisons vs Plan B (never conflates effect sizes).
- [x] Plan A vs Plan B comparison scatter (effect size and -log10 p).
- [x] Agreement/disagreement figures (meta vs Plan A LMM): confusion matrices, significance-colored theta plots, theta-difference histograms.
- [x] Heterogeneity QC plots (tau vs effect; sign agreement vs p).
- [x] Per-gene forest plot (per-guide slopes + SE) for explicit genes (no implicit top-N heuristic).
- [x] Output directory + naming convention (e.g., `figures/gene_level/` with deterministic filenames).

#### P3.7 — CLI + API (additive; no baseline changes)
- [x] Add CLI flags (additive; supports opt-out):
  - [x] `--gene-level` / `--no-gene-level`
  - [x] `--focal-vars ...`
  - [x] `--gene-id-col ...`
  - [x] `--gene-methods ...` (supports: `meta`, `stouffer`, `lmm`, `qc`, `flagged`, `mixture`, `tmeta`)
  - [x] `--gene-out-dir ...`
  - [x] `--gene-figures` / `--no-gene-figures`, `--gene-figures-dir`, and `--gene-forest-genes`
- [x] Add Python API entry point(s) that can run gene-level analysis using in-memory `std_res` + model matrix.
- [x] Ensure baseline pipeline path is unchanged when `--gene-level` is not set.

#### P3.8 — Testing + Validation (focus on invariants, calibration, and regressions)
- [x] Unit tests for meta-analysis math (tau estimator; edge cases `m_g=1/2`, zero variance).
- [x] Unit tests for Stouffer math (combined t-statistic + p/FDR behavior).
- [x] Unit tests for LMM rubric behavior (RI vs RI+RS fallbacks).
- [x] Golden tests for baseline outputs (byte-for-byte).
- [x] Cross-check tests: Plan A and Plan B agree on simple simulated data when assumptions match.
- [x] Targeted simulation/audit harness (small; not “barrage of versions”) to sanity-check calibration and heterogeneity behavior.
- [x] Local method-comparison summary script (`scripts/compare_gene_level_methods.py`) for agreement/disagreement triage.

#### P3.9 — Performance / UX
- [x] Add progress reporting (genes processed, failures, fallbacks).
- [x] Add optional parallelization (careful: determinism + stable sorting).
  - [x] Plan A supports `--gene-lmm-jobs` (thread pool; preserves stable output sorting).
- [x] Add caching/checkpoint-resume for per-gene fits (optional; keyed by input fingerprints) to speed iteration.
  - [x] CLI: `--gene-lmm-resume` and `--gene-lmm-checkpoint-every N`
  - [x] Artifacts: `PMD_std_res_gene_lmm.partial.tsv` and `PMD_std_res_gene_lmm.partial.meta.json` (fail-fast on meta mismatch)

#### P3.10 — Docs + Release
- [x] Add docs page describing Plans A/B/C, the rubric, and output file schemas.
- [x] Update `README.md` with gene-level usage examples.
- [ ] Version + changelog entry + tag once this ships.

---

### P4 — Ground-Truth Benchmarks (Count Depth + Confounding)

#### P4.1 — Poisson Count-Depth Simulator (ground truth)
Goal: create a statistically grounded benchmark harness that simulates **Poisson counts** with **sample-depth variation**
(including depth confounded with treatment), then evaluates downstream gene-level methods against known truth.

Phase A — Minimal benchmark harness (deterministic; local)
- [x] Add `scripts/benchmark_count_depth.py`:
  - [x] simulate per-guide counts with per-sample depth factors (log-normal) + optional Poisson noise layer on depths
  - [x] allow treatment-confounded depth via `treatment_depth_multiplier`
  - [x] store truth (`theta_true`, `is_signal`) and inputs/outputs in an output directory
- [x] Ensure the null is *actually null* by default:
  - [x] apply on-target guide slope heterogeneity only for signal genes (no accidental signal leakage in `frac_signal=0` runs)
  - [x] add gene→guide hierarchy for baseline lambda (`log_lambda_gene + Normal(0, guide_lambda_log_sd)`)
  - [x] write `sim_truth_guide.tsv` for guide-level ground truth diagnostics
- [x] Support fast response construction modes (no PMD bootstrap):
  - [x] `log_counts`
  - [x] `guide_zscore_log_counts`
- [x] Support a full PMD response mode (small simulations only):
  - [x] `pmd_std_res` with `--pmd-n-boot` and deterministic `--pmd-seed`
- [x] Run Plan B / Plan A / Plan C **directly** on the simulated response matrix:
  - [x] write `PMD_std_res_gene_meta.tsv`, `PMD_std_res_gene_stouffer.tsv`, `PMD_std_res_gene_lmm.tsv`, `PMD_std_res_gene_qc.tsv`
  - [x] write a strict machine-readable `benchmark_report.json` (valid JSON; no `NaN`) with:
    - [x] runtime summaries
    - [x] confusion matrices at `p < alpha` and `p_adj < q` (TP/FP/TN/FN, FDR/TPR/FPR)
    - [x] null calibration summaries
    - [x] QQ stats (`lambda_gc`) and optional QQ plot PNGs

Phase B — Statistical realism upgrades (explicit configs; no silent heuristics)
- [x] Add optional batch covariate(s) and explicit confounding patterns (depth ↔ treatment ↔ batch) to stress identifiability.
- [x] Add optional “bad guide” contamination process (off-target mixture on guide effects) to stress robustness methods.
- [x] Add optional non-Gaussian noise/heteroskedasticity modes and document which downstream tests remain calibrated.
  - [x] Implement NB-style overdispersion via a Gamma-Poisson mixture (`nb_overdispersion`).
  - [x] Document calibration expectations (what stays uniform under null) for each response mode + confounding pattern.

Phase C — Performance benchmark grid + reporting
- [x] Add a small grid runner that shells out to `scripts/benchmark_count_depth.py` and writes `count_depth_grid_summary.tsv` (`scripts/run_count_depth_grid.py`).
- [x] Expand the grid runner to sweep key realism knobs:
  - [x] gene-level lambda heterogeneity (`gene_lambda_log_sd`)
  - [x] within-gene guide lambda heterogeneity (`guide_lambda_log_sd`)
  - [x] guide slope heterogeneity (`guide_slope_sd`)
  - [x] off-target fraction + magnitude (`offtarget_*`)
  - [x] depth variation + treatment depth confounding (`depth_log_sd`, `treatment_depth_multiplier`)
  - [x] optional depth covariate inclusion (`--include-depth-covariate`)
- [x] Capture calibration + confusion-matrix summaries in `count_depth_grid_summary.tsv` (incl. `lambda_gc`, FP/FPR, runtime).
- [x] Add summary figures (runtime vs size; null p-value calibration; power vs effect size) under a benchmark output directory.

Phase D — Tie benchmark back to selection policy
- [x] Evaluate Plan A selection policy tradeoffs (power vs runtime) on the benchmark grid (explicitly record selection settings).

#### P4.2 — Depth Covariate Realism (observed-only; no oracle adjustment)
Goal: ensure the benchmark reflects what we can do in real data, where we only have measured library sizes (count depth).

- [x] Remove oracle adjustment from the benchmark: `--include-depth-covariate` uses `log_libsize_centered = log(colsum(counts))` (not simulated `log_depth`).
- [x] Add explicit `depth_covariate_mode` to the benchmark config (no silent behavior):
  - [x] `none` (no depth adjustment)
  - [x] `log_libsize` (use `log(colsum(counts))` as a proxy for depth; real-data-compatible; center internally as an implementation detail)
- [x] Write the chosen depth covariate values to `sim_truth_sample.tsv` (and record in `benchmark_report.json`) so results are fully auditable.
- [x] Add a small check/plot in the benchmark report: `log_libsize` distributions and correlation with treatment/batch (confounding audit).

#### P4.3 — Normalization + Response Construction (PMD + common-sense baselines)
Goal: benchmark PMD standardized residuals **and** common-sense depth normalization approaches under the same confounding settings.

- [x] Implement explicit response-pipeline knobs in the benchmark config/CLI (recorded in `benchmark_report.json`; no silent behavior):
  - [x] `normalization_mode` (acts on counts before log; not supported for PMD response mode)
  - [x] `logratio_mode` (optional; operates across features within each sample; not supported for PMD response mode)
  - [x] `n_reference_genes` (adds an explicit always-null reference set for `alr_refset`)
- [x] Implement `normalization_mode` options (deterministic; no extra deps):
  - [x] `none` (raw counts)
  - [x] `libsize_to_mean` (scale counts by `mean(libsize)/libsize`)
  - [x] `cpm` (counts per million by sample libsize)
  - [x] `median_ratio` (DESeq-style size factors; document assumptions + failure modes under composition shifts)
- [x] Implement log transform (fixed for now): `log(norm_count + pseudocount)`
- [x] Implement `logratio_mode` options (compositional transforms that can change significance under depth confounding):
  - [x] `none`
  - [x] `clr_all` (Centered Log-Ratio: subtract per-sample mean log-count across all guides; depth-invariant under pure multiplicative sampling)
  - [x] `alr_refset` (Additive Log-Ratio to an explicit reference set: subtract per-sample mean log-count over reference guides, e.g., non-targeting controls)
  - [x] Do **not** add ILR by default (basis-dependent and hard to interpret at the per-guide level; consider only if a multivariate model truly requires it)
- [x] Implement `standardize_mode` variants (via `response_mode`; no separate knob needed):
  - [x] `none` (use `response_mode=log_counts`)
  - [x] `per_guide_zscore` (use `response_mode=guide_zscore_log_counts`; note: for per-guide OLS with an intercept this does **not** change t/p, so treat as LMM-only sensitivity / numerical-stability experiment)
- [x] Keep PMD standardized residuals as an additional (not exclusive) response construction:
  - [x] `response_mode=pmd_std_res` runs PMD on the count matrix (treat as a normalization layer)
  - [x] record `pmd_n_boot` and `pmd_seed` in `benchmark_report.json` for reproducibility
- [x] Ensure each response pipeline writes a clearly named artifact bundle in the output dir:
  - [x] `sim_std_res.tsv` (final response matrix used for downstream inference)
  - [x] a small JSON note of the pipeline (via `benchmark_report.json` → `config`)

#### P4.4 — Method Matrix (explicit; no conflation)
Goal: evaluate methods × response constructions × depth handling without mixing outputs.

- [x] Expand benchmark `methods` to cover the historical p-combine explicitly:
  - [x] `stouffer` (combined-t implementation; uses per-guide OLS t)
- [x] Keep Plan A results explicitly separated:
  - [x] LMM LRT metrics from `lrt_*` columns only
  - [x] LMM Wald metrics from `wald_*` columns only
- [x] Add a benchmark-side “design matrix sanity” section in the JSON:
  - [x] rank / condition number checks per run
  - [x] correlations among covariates (esp. treatment vs depth proxies vs batch)

#### P4.5 — Performance Metrics (beyond calibration)
Goal: quantify correctness along multiple axes, not just FDR.

- [x] Count realism / QC (pre-inference):
  - [x] mean–variance and mean–dispersion diagnostics (per-guide; writes `sim_counts_mean_dispersion.tsv`)
  - [x] mean–variance and mean–dispersion diagnostics (per-gene; writes `sim_counts_gene_mean_dispersion.tsv`)
  - [x] depth-proxy diagnostics: `log_libsize` distribution + correlation with treatment/batch
- [x] Calibration (null):
  - [x] QQ + `lambda_gc` (recorded numerically by default; optional QQ plot PNGs)
  - [x] KS distance vs Uniform(0,1) (recorded numerically)
  - [x] p-histograms (recorded in `benchmark_report.json` under `*_p_hist_null`; plotted via `scripts/plot_count_depth_p_histograms.py`)
- [x] Detection (signal):
  - [x] ROC-AUC and PR-AUC (using p-values as scores; deterministic)
  - [x] power curves vs effect size (`effect_sd`) at fixed FDR q (via `scripts/plot_count_depth_grid_summary.py`)
- [x] Confusion-matrix metrics (signal + null, when defined):
  - [x] record balanced accuracy and MCC alongside TP/FP/TN/FN (stored in `benchmark_report.json` and surfaced into `count_depth_grid_summary.tsv`)
- [x] Estimation quality:
  - [x] correlation and RMSE of `theta_hat` vs `theta_true` (meta/LMM)
  - [x] sign accuracy vs `theta_true` (signal genes)
- [x] Robustness/heterogeneity diagnostics:
  - [x] relationship between estimated `tau` (meta/LMM) and simulated guide heterogeneity (`theta_dev_sd` per gene from `sim_truth_guide.tsv`)
- [x] Runtime scaling:
  - [x] record per-method runtime per run in `benchmark_report.json` and `count_depth_grid_summary.tsv`
  - [x] runtime vs `n_genes` × `guides_per_gene` × sample size; plus “success/failure fractions” for Plan A (grid supports sweeps; plots via `scripts/plot_count_depth_grid_summary.py`).

#### P4.6 — Visualization Suite (for the full grid)
Goal: a small set of figures that makes tradeoffs obvious to a reader.

- [x] “Method grid” figure (single page, high-level):
  - [x] rows: method pipelines (explicitly separate `lmm_lrt` vs `lmm_wald`; include meta and stouffer; includes CLR/ALR variants when present in the grid)
  - [x] columns: scenario × metric pairs (e.g., `null | lambda_gc_dev`, `signal | q_tpr`, `depth_confounded | q_mcc`, …)
  - [x] cells: normalized rank (worst→best) so all metrics share a common 0–1 visual scale
  - [x] add summary columns without cross-domain cancellation:
    - [x] per-domain summaries in TSV (`avg_score_null`, `avg_score_signal`, etc.)
    - [x] sorting summaries use the weaker domain (`avg_score_min_domain`, `worst_score_min_domain`)
  - [x] produce 2 versions: sorted by `avg_score_min_domain` and by `worst_score_min_domain` (worst-case robustness view)
- [x] Rank scorecard (dot heatmap / circle plot):
  - [x] rows: pipelines; columns: key metrics (FDR@q excess, TPR@q, AUC/PR-AUC, lambda_gc deviation, runtime)
  - [x] circle size: normalized rank within each metric; color: worst→best with a single legend
  - [x] produce 2 versions: “null-only” and “signal-only” so calibration vs power is never conflated
  - [x] within each version, aggregate across scenarios by *worst-case* (direction-aware) rather than pooling averages across adversarial scenarios
  - [x] add right-side summary panel: `avg` and `worst` dots per pipeline row
  - [x] produce 2 sorted variants for each scorecard: by `avg` and by `worst`
  - [x] use tight layout + `bbox_inches="tight"` so long labels are never cut off
  - [x] include an additional “signal-only estimation” scorecard for theta metrics (meta/LMM only)
  - [x] include an additional “signal-only confusion” scorecard (balanced accuracy, MCC, F1; plus FDR excess and runtime)
- [x] Grid heatmaps (faceted by method/response):
  - [x] null inflation (`lambda_gc`) vs `depth_log_sd` and `treatment_depth_multiplier` (via `scripts/plot_count_depth_grid_heatmaps.py`)
  - [x] FDR at q vs same axes (via `scripts/plot_count_depth_grid_heatmaps.py`)
- [x] Power vs realism knobs:
  - [x] TPR at q vs `effect_sd` (one curve per method/response/depth-handling; via `scripts/plot_count_depth_grid_summary.py`)
- [x] Pareto front plots:
  - [x] runtime vs TPR at q (color by achieved FDR; via `scripts/plot_count_depth_scorecards.py`)
- [x] Agreement/disagreement plots on the *same simulated truth*:
  - [x] method-vs-method scatter of `-log10(p)` (never mixing LRT/Wald; via `scripts/plot_benchmark_method_agreement.py`)
  - [x] confusion matrices at q for method pairs (via `scripts/plot_benchmark_method_agreement.py`)
- [x] “Depth confounding” diagnostics:
  - [x] estimated treatment effect vs simulated depth proxy correlation (via `scripts/plot_count_depth_confounding_diagnostics.py`)

#### P4.7 — Reporting + Reproducibility
- [x] Standardize output directory naming to encode the benchmark pipeline (response + normalization + depth mode) without ambiguity.
- [x] Ensure every figure can be regenerated from the written TSV/JSON artifacts (scripts remain consumer-only; no hidden recomputation).
  - [x] Convenience runner: `scripts/run_count_depth_benchmark_suite.py` (grid → aggregate → figures).
  - [x] Suite manifest captures invocation, git, environment, and exact sub-commands (`suite_manifest.json`).
  - [x] Friction reducers: `--preset {quick,standard,full}` and `--resume` (reuse existing grid TSV; avoid mixing outputs by accident).
  - [x] Concurrency convenience: suite accepts `--jobs N` and forwards to the grid runner.
  - [x] Safer warm-starts: `--resume` refuses to continue across differing git HEADs unless `--force-resume`.
  - [x] Presets sweep common-sense processing permutations (response mode, normalization mode, log-ratio mode) where applicable; PMD stays normalization/log-ratio-free by construction.
  - [x] Warm-start + parallelism: `scripts/run_count_depth_grid.py --resume --jobs N` (skip completed runs; run multiple configs concurrently).
  - [x] Suite forwards `--resume` to the grid runner and streams subcommand output (background logs show progress).
- [x] Add a small QC helper for suite directories (`scripts/qc_count_depth_benchmark_suite.py`) to sanity-check completion + key artifacts.
- [x] Close benchmark “coverage gaps” for pipeline-vs-scenario comparisons: scenario presets must not hard-disable analysis-pipeline components (depth/batch covariates), and method-grid ranking must prioritize complete-coverage pipelines when gaps exist.

#### P4.8 — Abundance / Composition Stress Tests (Hierarchical λ regimes)
Goal: stress-test pipeline performance under realistic **compositional abundance regimes**, including heavy tails and within-gene dominance
(analogous to microbiome genus/species or cell-state/cell-type hierarchies).

Phase A — Codify the current abundance model (SSoT + audit)
- [x] Baseline today (already implemented): gene-level `log_lambda_gene ~ Normal(log_mean, gene_lambda_log_sd)` and within-gene guide `log_lambda_guide = log_lambda_gene + Normal(0, guide_lambda_log_sd)`.
- [x] Add a single, reusable “abundance audit” summary (written per run) that reports:
  - [x] gene-level: quantiles of `log_lambda_gene`, tail metrics (top-k share / Gini / entropy), and rare/zero fraction
  - [x] within-gene: distribution of per-gene guide dominance (e.g., max/mean, SD(log lambda) within gene)
  - [x] sample-level: library-size distribution and effective sparsity (zero fraction per sample)
  - [x] ensure the audit is derived from **simulated truth** (not downstream outputs), and is recorded in `benchmark_report.json`

Phase B — Gene-level abundance families (top layer: gene/species/cell-type)
- [x] Add explicit `gene_lambda_family` choices (default preserves current behavior):
  - [x] `lognormal` (current; compatibility path)
  - [x] `mixture_lognormal` (rare-vs-abundant mixture; tunable `pi_high`, `delta_log_mean`, `log_sds`)
  - [x] `power_law` / Pareto-like (few dominant, many rare; tunable tail index; avoid oracle scaling)
- [x] Define “battle-test” presets for the above that target edge-of-failure behavior (not “perfect data”):
  - [x] Parameter scouting: `scripts/scout_abundance_params.py` (choose regimes that land in/near dropout with ~3–4 orders of magnitude dynamic range).
  - [x] “few dominant, many rare” (microbiome-like; mixture lognormal at a sparse baseline):
    - [x] `guide_lambda_log_mean=log(20)`, `gene_lambda_log_sd=0.4`, `guide_lambda_log_sd=1.0`
    - [x] `gene_lambda_family=mixture_lognormal`, `mix_pi_high=0.05`, `mix_delta_log_mean=2.5`
  - [x] “many dominant, few rare” (broadly abundant classes; mixture lognormal at the same sparse baseline):
    - [x] `guide_lambda_log_mean=log(20)`, `gene_lambda_log_sd=0.4`, `guide_lambda_log_sd=1.0`
    - [x] `gene_lambda_family=mixture_lognormal`, `mix_pi_high=0.8`, `mix_delta_log_mean=2.5`
  - [x] Power-law heavy tail (additional “few dominant” regime without mixture assumptions):
    - [x] `guide_lambda_log_mean=log(20)`, `guide_lambda_log_sd=1.0`
    - [x] `gene_lambda_family=power_law`, `power_alpha=1.05`

Phase C — Within-gene guide abundance families (bottom layer: guide/genus/cell-state)
- [x] Add explicit `guide_lambda_family` choices (default preserves current behavior):
  - [x] `lognormal_noise` (current; compatibility path)
  - [x] `dirichlet_weights` (symmetric Dirichlet with tunable concentration to create within-gene dominance)
- [x] Ensure within-gene models preserve the intended interpretation of `log_lambda_gene` (mean per guide) while allowing dominance patterns.

Phase D — Benchmark grid + suite integration (without combinatorial explosion)
- [x] Add a dedicated suite preset focused on abundance regimes (keep other knobs minimal to avoid “metric column blow-up”):
  - [x] sweep a small set of (gene_lambda_family × guide_lambda_family) combinations
  - [x] keep depth/batch/offtarget/overdispersion scenarios as separate scenario columns (never pooled across null/signal)
  - [x] ensure scenario labeling (`scenario_id` / `scenario`) includes the abundance-family IDs + only the parameters that vary

Phase E — Visualization + reporting (statistician-first)
- [x] Add figure(s) that explicitly show the simulated abundance regime per scenario (rank-abundance + histograms), alongside performance:
  - [x] one-page “scenario audit” panel (rank-abundance + distributions): `scripts/plot_count_depth_abundance_scenarios.py`
  - [x] run by default for `--preset abundance`: `scripts/run_count_depth_benchmark_suite.py`
  - [x] keep pipelines in rows; scenario×metric in columns; avoid any null/signal averaging (benchmark-wide invariant)

#### P4.9 — Expected-Count Quantifiability (E) Buckets (Observed-only; depth-confounding-safe)
Goal: add a statistically grounded “is this gene/sample cell quantifiable?” layer based on **chi-square expected counts**,
so benchmark metrics can be stratified by low- vs high-information regimes (incl. treatment-confounded depth).

Key constraints:
- Use **observed counts only** (empirical library sizes from the simulated counts); never use oracle depth factors.
- Keep all existing benchmark artifacts stable; quantifiability outputs must be **additive** (new files / new optional metrics).
- Persist enough to support later **re-binning** and continuous association work without rerunning expensive benchmark grids.

Phase A — Definitions + output schema (SSoT)
- [ ] Define the expected-count construction per condition `c ∈ {control, treatment}`:
  - [ ] Collapse guide counts → gene counts: `y[g,s] = sum_{guides in g} counts[guide,s]`.
  - [ ] For each condition `c`, compute the chi-square expected table:
    - [ ] `Y_g(c) = sum_{s∈S_c} y[g,s]`
    - [ ] `d_s(c) = sum_g y[g,s]` (empirical libsize within `c`)
    - [ ] `D(c) = sum_{s∈S_c} d_s(c)`
    - [ ] `E[g,s|c] = Y_g(c) * d_s(c) / D(c)`
- [ ] Define a robust gene-level “quantifiability driver”:
  - [ ] `E_ctrl_p10(g)` and `E_trt_p10(g)` (10th percentile across samples within each condition)
  - [ ] `E_p10_mincond(g) = min(E_ctrl_p10(g), E_trt_p10(g))`
- [ ] Define reporting buckets (consumer-layer policy; parameterized):
  - [ ] `E_p10_mincond < 1`, `1–<3`, `3–<5`, `>=5` (classic chi-square headroom cutoffs).

Phase B — Core utilities (pure; testable)
- [ ] Add a small core module for:
  - [ ] collapsing guide→gene counts
  - [ ] chi-square expected-count computation
  - [ ] expected-count summaries (min/mean/quantiles) used by the benchmark

Phase C — Per-run artifacts (additive; no reruns required)
- [ ] Write per-run quantifiability artifacts alongside other `sim_*` inputs:
  - [ ] `sim_gene_expected_counts.tsv` (per-gene summaries + buckets; includes `Y_ctrl`, `Y_trt`, `E_*` summaries)
  - [ ] `sim_gene_expected_counts_matrix.tsv.gz` (long table: `gene_id, sample_id, treatment, observed_count, expected_count`)
- [ ] Ensure these never overwrite other artifacts unless explicitly forced.

Phase D — Backfill / resume (no benchmark reruns)
- [ ] Add a “backfill” script that scans existing run directories (via `--grid-tsv` or `--root`) and writes any missing
      expected-count artifacts with `--resume` semantics.
- [ ] Parallelize backfill safely (`--jobs N`); deterministic outputs; stable ordering.

Phase E — Rehydrate inputs (deterministic; avoid PMD recompute)
- [ ] Add a “rehydrate” script that can recreate missing `sim_counts.tsv`, `sim_truth_*.tsv`, and `sim_model_matrix.tsv`
      from `benchmark_report.json` config + seed, **without** recomputing PMD standardized residuals by default.
- [ ] Refuse to overwrite existing files unless `--force`.

Phase F — Bucket-stratified benchmark reporting (statistician-first)
- [ ] Compute benchmark metrics stratified by `E_p10_mincond` bucket:
  - [ ] calibration metrics on **null genes only** (e.g., KS/uniformity summaries, QQ `lambda_gc`)
  - [ ] power/TPR on **signal genes only**, plus FDR/FPR (avoid any null/signal pooling)
  - [ ] confusion-matrix metrics (balanced accuracy, MCC) as secondary summaries (clearly labeled)
- [ ] Emit a long-format TSV (to avoid column explosion) keyed by:
  - [ ] `pipeline_id`, `scenario_id`, `metric`, `bucket`, `value`, `n_genes`
- [ ] Update scorecards/heatmaps to treat **pipelines as rows** and **scenario×metric×bucket** as columns (no averaging).
