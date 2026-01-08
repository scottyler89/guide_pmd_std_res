# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project aims to follow Semantic Versioning.

## [Unreleased]

## [0.1.5] - 2026-01-08

### Added
- `pytest` test suite and `pytest.ini` for consistent local/CI runs.
- `ruff` linting configuration and CI enforcement.
- GitHub Actions CI for tests, lint, and docs.
- `pyproject.toml` (PEP 621) with `dev` and `docs` extras.
- Console entry point `guide-pmd-std-res` and `python -m guide_pmd` support.
- API docs page (`docs/api.rst`) and improved docs index.
- MIT `LICENSE` file.

### Changed
- Minimum supported Python version is now 3.10+.

### Fixed
- `pmd_std_res_and_stats()` now returns `(std_res, None, None, None)` when `model_matrix_file=None`.
- `get_pmd_std_res()` now respects the provided delimiter (`sep`).
- Fixed invalid `raise "..."` usage (now raises a proper exception).
- `run_glm_analysis()` now safely handles the case where all features have zero variance.

## [0.1.4]
- Prior release baseline (pre-hygiene upgrades).
