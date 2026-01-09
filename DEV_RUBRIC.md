# DEV_RUBRIC – Engineering Principles & Review Rubric

This document is a **living** guide for how we design, implement, review, and evolve a scientific software repository. It is intended to:

- Make architectural decisions consistent and explainable.
- Prevent “quick fixes” from becoming permanent technical debt.
- Provide a shared rubric for grading proposed implementations (including experimental code and presentation pipelines).

If you’re proposing a change, treat this as the checklist that your PR/commit should satisfy.

---

## 1) Non‑Negotiables (Must Always Hold)

### 1.1 Single Source of Truth (SSoT)

- **One canonical place** defines each concept (data model, geometry, stats test, plot style, etc.).
- Downstream modules **consume** that canonical implementation; they do not re-implement “their own version”.
- If something must exist in multiple layers (e.g., analysis + plotting), define the data structure once (core) and build thin consumers (viz/reports).

### 1.2 No Silent Fallbacks, No Guessing

- Never “guess” missing metadata, fabricate coordinates, or auto-correct inconsistent inputs.
- Prefer **fail fast** with an explicit error message over running a likely-wrong analysis.
- If an input is ambiguous, require the caller to specify it (explicit config > implicit inference).

### 1.3 No Heuristics in Production Paths

We do not embed arbitrary thresholds or “if it looks like X then…” logic in core analysis utilities.

- Prefer formal, inspectable methods:
  - Cross-validation for model comparison.
  - Permutation / bootstrap for uncertainty and hypothesis testing.
  - Explicit priors only when stated and validated by synthetic controls.
- If a threshold is required by a downstream consumer (e.g., a report needs a binary call), it must be:
  - Documented as **consumer-layer policy**, not core analysis.
  - Parameterized and justified.

### 1.4 Shims / Bridges Are Allowed Only as Burn‑Down Tools

Temporary migration code is permitted only if it cannot leak into production or mask underlying issues.

Requirements for any shim:
- Must be **clearly labeled** (e.g., `DEPRECATED`, `MIGRATION_ONLY`, or `TEMP`) and include an explicit removal plan.
- Must be **isolated** (prefer a dedicated migration module or one-off scripts), not in the primary analysis path.
- Must not silently change behavior; if it transforms inputs, it must log/raise clearly or be gated behind an explicit flag.
- Must be deleted once the migration is complete (track removal via issues/TODOs and delete legacy code).

### 1.5 Reproducibility by Default

- All stochastic code must accept `rng: np.random.Generator | int | None` and use it deterministically.
- Benchmarks should record configs and seeds in machine-readable artifacts (e.g., JSON) so figures and tables can be regenerated.
- Deterministic failures are preferred over nondeterministic “flaky” behavior.

---

## 2) Architecture & Layering (How Code Should Be Organized)

### 2.1 Preferred Layer Boundaries

- **Data IO & validation**
  - Load/validate/slice datasets.
  - Enforce invariants and explicit geometry/units.
  - No plotting; minimal statistical logic.

- **Core analysis**
  - Pure functions and small dataclasses for modeling, summaries, and tests.
  - No IO; no “writing files”; no report/presentation logic.
  - All decisions should be inspectable (return risk curves, p-values, CIs, etc.).

- **Workflows / benchmarks**
  - Compose analysis primitives into benchmark runners and grids.
  - Prefer returning structured results rather than writing artifacts directly.

- **Visualization / reporting**
  - Thin wrappers around plotting libraries.
  - Consume analysis/workflow result objects; do not recompute statistics.
  - Centralize style (colors, legend placement, fonts) in one place.

- **Pipelines / scripts**
  - Orchestration only: call workflows, write outputs (JSON/CSV/PDF), and ensure reproducibility.
  - Should be “dumb consumers”: no core math, no new stats tests here.

### 2.2 Prefer Small, Explicit APIs Over “Frameworks”

- Use simple dataclasses/config objects with `.validate()` where appropriate.
- Avoid clever abstractions that hide important choices (especially statistical ones).
- Favor composability:
  - Make primitive functions reusable (e.g., one envelope helper used everywhere).
  - Keep orchestrators as the only place where “which scenarios to run” is chosen.

---

## 3) Testing Philosophy (Unit-Test Driven, With Controls)

### 3.1 Tests Must Include Positive and Negative Controls

For any new method that claims detection/estimation:
- **Negative controls**: scenarios where the true effect is zero (must not “hallucinate” signal).
- **Positive controls**: scenarios with known ground-truth effects (must detect/estimate correctly).

### 3.2 Targeted, Fast Tests by Default

- Prefer narrow unit tests (`pytest tests/test_<module>.py -q`) over running the full suite repeatedly.
- If a test is slow, make it explicit:
  - Mark it as slow (naming or `pytest` markers if introduced later).
  - Keep a small fast version and a longer validation version.

### 3.3 What to Assert

Prefer assertions that match the scientific claim:
- Shape and finiteness invariants (always).
- Monotonicity with effect size / noise when expected.
- Calibration under null (p-values approx Uniform(0,1) in synthetic nulls).
- CI coverage in controlled simulations (when feasible).

### 3.4 Determinism

- Use fixed seeds in tests.
- Avoid assertions that depend on fragile numeric coincidences; use robust inequalities and tolerances.

---

## 4) Statistical Guardrails (How We Keep Inference Honest)

- If a model includes additional parameters, we must quantify:
  - Bias/variance impact.
  - Overfitting risk (CV comparisons).
  - Null behavior (false positives).
- Any “decision rule” must be inspectable and grounded in:
  - CV risk, permutation p-values, bootstrap CIs, explicit Bayesian modeling with stated priors, or firmly statistically grounded.
- Treat presentation-ready narratives as **separate** from inference:
  - Slides can simplify language, but the underlying results must be reproducible and honest.

---

## 5) Git Hygiene (Working Style)

### 5.1 Commit Early and Often

- Prefer small commits with a single intent.
- Include tests and doc updates in the same commit when they are tightly coupled.
- Avoid “mega commits” that mix refactors and new features unless necessary.

### 5.2 Branching Guidance

- Use a branch when work spans multiple days or has a risk of breaking main.
- Keep branches short-lived; merge early once tests and docs are aligned.
- If a change is experimental, keep it explicitly labeled and isolated (do not ship “experimental defaults” in core APIs).

### 5.3 Commit Messages

- Use imperative, descriptive messages:
  - Good: “Add treatment synthetic generator”
  - Avoid: “updates”, “fix stuff”, “wip”

---

## 6) Documentation Discipline (Keep Planning + Code in Sync)

- Any non-trivial implementation must update the relevant project docs (README, method notes, experiment plans, etc.) so contributors can infer:
  - What exists in code today.
  - What’s missing.
  - What outputs are expected (artifact names, APIs, and how to reproduce them).
- If you maintain a roadmap or checklist, keep status markers accurate (e.g., `[x]/[ ]`) so a new contributor can pick up without guessing.

---

## 7) Review Rubric (How We Grade Architectural/Coding Ideas)

Use this rubric for PR review or when comparing multiple implementation options. Score each category 0–3, and treat “Non‑Negotiables” as hard gates.

### 7.1 Hard Gates (Fail Any = Do Not Merge)

- Violates SSoT (duplicate implementations of the same concept).
- Introduces silent fallback/guessing behavior.
- Adds production-path heuristics without explicit statistical justification.
- Adds a shim/bridge without isolation + removal plan.
- Breaks determinism / cannot reproduce results.
- No tests for new core behavior (especially missing negative controls).

### 7.2 Scored Categories (0–3)

1) **Correctness & Scientific Validity**
- 0: unclear or incorrect; not aligned with the stated model/biology
- 1: plausible but unvalidated; weak tests
- 2: validated with synthetic controls; behaves under null/effect
- 3: validated + calibration checks + clear failure modes

2) **Simplicity**
- 0: over-engineered; hard to reason about
- 1: unnecessary abstraction; unclear interfaces
- 2: straightforward; explicit config; small functions
- 3: minimal surface area; composable primitives

3) **SSoT & Reuse**
- 0: duplicates logic; inconsistent behavior across layers
- 1: partial reuse; still forks behavior
- 2: clean reuse of existing primitives
- 3: strengthens SSoT (removes duplication, consolidates style/config)

4) **Testability**
- 0: hard to test; relies on IO/globals
- 1: tests exist but are slow/flaky
- 2: fast, deterministic unit tests with controls
- 3: includes calibration/coverage or monotonicity tests where appropriate

5) **Performance & Scalability**
- 0: obviously inefficient (unbounded loops, O(n²) where avoidable)
- 1: acceptable for now but not measured
- 2: reasonable complexity; avoids redundant recomputation
- 3: measured hotspots; targeted optimization without harming clarity

6) **Clarity & Maintainability**
- 0: confusing naming; unclear docs; hidden coupling
- 1: partial docs; unclear invariants
- 2: clear naming; `.validate()`; docstrings match behavior
- 3: includes “how to use” examples and explicit constraints

7) **User/Consumer Impact (viz + slides + downstream)**
- 0: breaks downstream usage; inconsistent outputs
- 1: changes outputs without docs; ambiguous figure semantics
- 2: stable filenames/APIs; labels/legends clear
- 3: improves interpretability; consistent style system-wide

### 7.3 What “Good” Looks Like

A strong change usually:
- Adds one or two well-named primitives.
- Adds a workflow wrapper (if needed) that composes them.
- Adds fast tests with positive + negative controls.
- Updates the relevant docs/checklists so the story is coherent.
- Produces outputs (figures/tables/artifacts) via a script that stays a consumer, not a model.

---

## 8) Decision Records (Optional, but Encouraged)

For major architectural choices, consider adding a short “decision record” note (even as a section in a PR description) covering:

- Problem statement.
- Considered options.
- Chosen option and why (rubric-based).
- Risks and mitigation (tests, calibration, CV, etc.).
- Sunset plan for any temporary scaffolding.
