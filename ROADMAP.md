# ROADMAP

Baseline tag: `baseline-2026-01-08` (commit `018cdd9` on `main`).

This roadmap tracks repo hygiene + modernization work for `guide_pmd_std_res`.

## Goals
- Make the CLI and Python API more robust (better input validation, fewer footguns).
- Add tests and CI so changes are safe to ship.
- Modernize packaging so installs/docs are reproducible.

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
- [ ] Tag releases aligned to versions (done when merging back to `main`).
