# Gene-level aggregation from multiple guide-level measurements: first-principles analysis plans

**Context (generic statistical framing).** For each gene \(g\), you have \(m_g\) guides \(j=1,\dots,m_g\). For each sample (or experimental unit) \(k\), you have a *Gaussian-stabilized* response \(y_{gjk}\) and a design matrix \(X_k\) containing the covariates of interest (e.g., treatment indicator, batch covariates). You want a **gene-level effect** for a specific contrast (e.g., treatment vs control) that appropriately aggregates guide-level information while being robust to guide idiosyncrasies (efficiency, off-target, mis-annotation), and improves on ad hoc combination of guide-level \(p\)-values (e.g., Stouffer).

This document expands items **#7–#8** from the previous discussion and provides a **decision rubric**, **option cross-comparisons**, and **recommended implementation plans**.

---

## 0) What you should *not* do (from first principles)

If you have access to the underlying sample-level data \(y_{gjk}\), then combining already-reduced guide-level test statistics is **strictly information-losing** unless:
- the guide-level statistics are sufficient for the gene-level parameter under a correctly specified model (rare in contamination/heterogeneity settings), **or**
- operational constraints force you to meta-analyze.

Therefore, your “upgrade” should be to **fit a single joint model per gene (or globally with gene indexing)** and then extract the gene-level contrast, with guide behavior modeled explicitly.

---

## 1) Decision map: the major modeling choices that matter most

Below are the key decisions that drive correctness, calibration, and robustness. For each decision, I give at least four viable options, compare them, and then recommend a best implementation (sometimes with a secondary robustness check).

### Decision A — What is the gene-level estimand?

You need to define the target parameter \(\theta_g\) precisely. Options:

1. **Common mean shift across guides (fixed guide effects):**  
   \[
   \mathbb{E}[y_{gjk}\mid X_k] = \alpha_{gj} + X_k^\top\beta + \theta_g \cdot T_k
   \]
   where \(T_k\) is the focal covariate (treatment).  
   **Interpretation:** gene effect is a *single* shift shared by all guides.

2. **Guide-weighted average shift (random guide slopes):**  
   \[
   \mathbb{E}[y_{gjk}] = \alpha_{gj} + X_k^\top\beta + (\theta_g + b_{gj})T_k
   \]
   \(\theta_g\) is the population mean slope; \(b_{gj}\) is guide deviation.  
   **Interpretation:** gene effect is the mean across a guide distribution.

3. **Median/robust location of guide slopes:**  
   \(\theta_g\) corresponds to a robust location (median or M-estimator) of guide slopes.  
   **Interpretation:** gene effect reflects the “typical” guide, resistant to outliers.

4. **Mixture-defined “on-target” slope:**  
   Guides are mixture of “good” and “bad” components; \(\theta_g\) is the good-component slope mean.  
   **Interpretation:** gene effect is the effect among on-target guides.

**Cross-comparison (A).**
- (1) is simplest and most powerful when all guides are valid and homogeneous; least robust.
- (2) allows heterogeneity and yields honest uncertainty; moderate robustness.
- (3) is robust but may sacrifice power and complicate inference.
- (4) is most aligned with “some guides are garbage” reality, but adds compute/complexity.

**Best implementation (A).** Use **(2)** as the primary estimand (population mean slope) because it is the most defensible “gene effect” under guide heterogeneity, and **(4)** as an optional robustness mode when contamination is non-trivial.

---

### Decision B — Where to aggregate: observation-level vs guide-level summaries?

Options:

1. **Observation-level hierarchical model (preferred):** model all \(y_{gjk}\) jointly with guide random effects. Uses full likelihood.

2. **Guide-level summary model:** fit per-guide GLMs, extract \(\hat\beta_{gj}\) and \(\mathrm{SE}_{gj}\), then fit a second-stage hierarchical model  
   \(\hat\beta_{gj}\sim \mathcal{N}(\theta_g, \mathrm{SE}_{gj}^2+\tau_g^2)\).

3. **Two-step with residualization:** regress out nuisance covariates at observation level, then model treatment residuals with guide hierarchy.

4. **Collapse data across guides first:** average \(y\) across guides per sample and fit gene-level GLM on the collapsed response.

**Cross-comparison (B).**
- (1) dominates statistically: correct propagation of uncertainty, avoids asymptotic approximations.
- (2) is reasonable if per-guide outputs are stable and independence holds; still information-losing.
- (3) can be numerically stable and fast; still close to (1) if done correctly.
- (4) is generally suboptimal unless guides are truly exchangeable and equally reliable; also loses guide diagnostics.

**Best implementation (B).** Choose **(1)** as primary. Keep **(2)** as a fast fallback / validation layer.

---

### Decision C — How to model guide heterogeneity (slopes and intercepts)?

Options:

1. **Random intercepts only (RI):** guide-specific baselines \(\alpha_{gj}\), common slope \(\theta_g\).  
2. **Random intercept + random slope (RI+RS):** \(\alpha_{gj}\) and slope deviations \(b_{gj}\).  
3. **Fixed guide intercepts (FI) + common slope:** treat \(\alpha_{gj}\) as fixed effects; slope common.  
4. **Fixed intercepts + random slopes:** often redundant; can be used if intercepts are uninteresting.

**Cross-comparison (C).**
- RI is often necessary (guides have different baselines).
- RI+RS captures heterogeneous efficacy/off-target in the *effect* itself; this is the key upgrade vs Stouffer.
- FI is robust to misspecification of intercept distribution but costs degrees of freedom.
- Random slopes without random intercepts is rarely sensible for guide data.

**Best implementation (C).** Use **RI+RS** when you have sufficient \(m_g\) (e.g., \(m_g\ge 3\)–4) and enough samples. If \(m_g\) is small or model is unstable, drop to **RI** (common slope) as a pragmatic fallback.

---

### Decision D — How to handle contamination / outlier guides?

Options:

1. **No special handling:** rely on Gaussian random effects and hope outliers are rare.  
2. **Robust likelihood (heavy-tailed):** Student-\(t\) residuals and/or \(t\) random effects on slopes.  
3. **Mixture model (“good vs bad” guides):** guide slopes drawn from a mixture; infer guide membership posteriors.  
4. **Influence-based trimming / winsorization:** fit RI+RS, compute standardized residuals or BLUP outliers, then refit excluding worst guides.

**Cross-comparison (D).**
- (1) is simplest; can fail badly if even one guide is extreme.
- (2) is principled robustness with modest overhead; inference remains likelihood-based.
- (3) is most explicit and interpretable for “on-target” vs “off-target”; can be compute-heavy and fragile for small \(m_g\).
- (4) is pragmatic but can distort p-values if not accounted for; best as sensitivity analysis, not primary.

**Best implementation (D).** Primary: **(2) heavy-tailed robustness** (least overhead, strong protection). Optional sensitivity: **(3) mixture** for genes flagged as heterogeneous/discordant.

---

### Decision E — What test statistic / inference procedure for the gene effect?

Options:

1. **Wald test on \(\theta_g\)** from the hierarchical model: \(z=\hat\theta_g/\mathrm{SE}(\hat\theta_g)\).  
2. **Likelihood ratio test (LRT)** comparing models with vs without \(\theta_g\).  
3. **Score test** at \(\theta_g=0\) (fast if fitting many genes).  
4. **Parametric bootstrap** for small-sample calibration (esp. with random slopes).

**Cross-comparison (E).**
- Wald is simple and fine with large-ish samples and stable SEs.
- LRT is usually more robust for variance components and boundary issues.
- Score is efficient but needs careful implementation.
- Bootstrap is gold-standard calibration but expensive.

**Best implementation (E).** Use **LRT** as primary (better behaved with mixed models), and compute Wald as a convenient effect-size summary. Use bootstrap only for a limited audit set.

---

### Decision F — Whether to share information across genes (global variance components)

Options:

1. **Per-gene variance components** \(\tau_g^2\) (random slope variance) estimated independently.  
2. **Global variance components** shared across genes (one \(\tau^2\) for all).  
3. **Empirical Bayes shrinkage** of \(\tau_g^2\) toward a global prior.  
4. **Stratified sharing** (e.g., by guide count or mean expression / variance bands).

**Cross-comparison (F).**
- (1) is flexible but unstable when \(m_g\) small.
- (2) is stable but may misfit specific genes.
- (3) is best compromise: stability + gene-specific adaptability.
- (4) is a practical approximation to (3).

**Best implementation (F).** If you are doing a screen across many genes, adopt **(3)** or **(4)**. If you are analyzing a small targeted set, per-gene (1) may be acceptable.

---

## 2) A practical rubric (how to pick model complexity gene-by-gene)

You can implement a rule-based rubric that chooses the simplest adequate model while preserving calibration.

### Inputs the rubric should consider
- **Guide count** \(m_g\)
- **Sample size** \(n\)
- **Guide agreement / discordance**
  - dispersion of per-guide slopes
  - fraction of guides with opposite sign vs the majority
- **Heterogeneity evidence**
  - likelihood gain of random slope vs no random slope
  - estimated \(\hat\tau_g^2\) (or its shrinkage estimate)
- **Outlier evidence**
  - max standardized residual / influence
  - posterior “bad-guide” probability (if mixture is used)
- **Model stability**
  - convergence diagnostics
  - singular fits (variance estimates ~0)

### Rubric: recommended decision rules

1. **Base model selection**
   - If \(m_g < 3\): fit **RI** only (random intercept, common slope).
   - If \(m_g \ge 3\): attempt **RI+RS**.
   - If RI+RS is singular or non-convergent: revert to **RI**.

2. **Robustness activation**
   - If discordance is high (e.g., ≥ 1 guide opposite sign and \(|z|\) large) or heterogeneity test is significant:
     - fit robust variant (Student-\(t\)) **or** mixture as a sensitivity check.

3. **Inference choice**
   - Prefer **LRT** for \(\theta_g\), especially when random slopes are present.
   - If LRT is unstable for a gene, fall back to Wald with robust SE (sandwich) as an emergency.

4. **Reporting**
   - Always report \(\hat\theta_g\) (effect size), \(\mathrm{SE}\), and a calibrated p-value (from LRT).
   - Report heterogeneity diagnostics (\(\hat\tau_g\), discordance metrics) as QC.

---

## 3) The best “upgrade” implementation: detailed plan

This plan assumes Gaussian outcomes and focuses on the gene-level contrast of interest (e.g., treatment effect).

### 3.1 Primary model (hierarchical mixed model; observation-level)

For each gene \(g\):

\[
y_{gjk} = \mu_g + \alpha_{gj} + X_k^\top \beta_g + (\theta_g + b_{gj})T_k + \varepsilon_{gjk}
\]

- \(\alpha_{gj}\sim \mathcal{N}(0,\sigma_{\alpha,g}^2)\) (random intercept per guide)
- \(b_{gj}\sim \mathcal{N}(0,\tau_g^2)\) (random slope deviation per guide)
- \(\varepsilon_{gjk}\sim \mathcal{N}(0,\sigma_g^2)\)

**Primary estimand:** \(\theta_g\) (mean treatment effect across guides for gene \(g\)).

**Test:** LRT of \(H_0:\theta_g=0\) comparing:
- Full: includes \(\theta_g\)
- Null: sets \(\theta_g=0\) (keep random effects structure the same)

**Why this is best from first principles**
- It is the likelihood-based estimator under a plausible generative model for guide heterogeneity.
- It uses all sample-level information.
- It yields a coherent uncertainty estimate and allows explicit modeling of guide deviations.

### 3.2 Robustness layer (recommended, modest overhead)

Implement **Student-\(t\) residuals** (or equivalently, scale-mixture-of-normals) to reduce influence of aberrant observations/guides.

Two practical versions:
- **t residuals**: \(\varepsilon_{gjk}\sim t_\nu(0,\sigma_g)\)
- **t random slopes**: \(b_{gj}\sim t_\nu(0,\tau_g)\)

**When to run it**
- Always, if compute allows; or
- Trigger via rubric only for genes with heterogeneity/discordance flags.

**Inference**
- Use approximate LRT (or Bayesian credible intervals if you go fully Bayesian).
- For frequentist calibration, treat this as sensitivity analysis unless you implement a formal t-likelihood fit.

### 3.3 Optional contamination mixture (only when justified)

For genes with strong discordance, fit a two-component mixture on guide slopes:

\[
(\theta_g + b_{gj}) \sim
\pi_g\,\mathcal{N}(\theta_g,\tau_g^2) + (1-\pi_g)\,\mathcal{N}(\theta_{0g},\kappa_g^2)
\]

Often \(\theta_{0g}=0\) is a reasonable “bad guide is null” anchor.

**Outputs**
- posterior guide weights \(w_{gj} = P(\text{good}\mid \text{data})\)
- a “clean” gene effect \(\theta_g\) supported by good guides

**Use case**
- This is not the default due to complexity, but it is the best aligned with explicit “on-target vs off-target” reasoning.

---

## 4) Cross-check plan: why and how (keeping overhead justified)

You asked for a good argument before adding overhead. Here is the minimal set that is usually worth it:

### Cross-check 1 (low overhead): second-stage meta-analysis from per-guide GLMs
- Fit per-guide GLM (as you already do) to get \(\hat\beta_{gj}, \mathrm{SE}_{gj}\).
- Fit random-effects meta model:
  \[
  \hat\beta_{gj}\sim \mathcal{N}(\theta_g, \mathrm{SE}_{gj}^2+\tau_g^2)
  \]
- Compare \(\hat\theta_g\) and p-values to the observation-level mixed model.

**Why keep it:** This detects implementation errors and confirms that the observation-level and summary-level approaches agree when assumptions match.

### Cross-check 2 (moderate overhead, targeted): robust refit for flagged genes
- Only for genes exceeding heterogeneity/discordance thresholds.
- Fit robust/heavy-tailed model or mixture.

**Why keep it:** Provides sensitivity assurance specifically where it matters, without running robust models for every gene.

---

## 5) Deliverables: outputs you should store per gene

Minimum gene-level outputs:
- \(\hat\theta_g\), \(\mathrm{SE}(\hat\theta_g)\), test statistic, p-value (LRT)
- guide heterogeneity \(\hat\tau_g\) (and \(\hat\sigma_{\alpha,g}\))
- per-guide BLUPs (optional) \( \hat b_{gj}\) and influence metrics
- discordance summary (e.g., sign agreement fraction, max \(|\text{standardized residual}|\))

These enable ranking, QC filtering, and downstream modeling.

---

## 6) Implementation sketch (language-agnostic)

### Data layout
A long table with columns:
- gene_id, guide_id, sample_id
- response y
- treatment indicator T
- nuisance covariates (batch, etc.)

### Per gene pipeline
1. Subset gene \(g\).
2. Fit RI+RS mixed model; if fails, fit RI.
3. Extract \(\hat\theta_g\), SE, diagnostics.
4. Compute LRT p-value for \(\theta_g\).
5. Apply rubric to decide if robust/mixture sensitivity should run.
6. Store outputs + QC metrics.

### Multiple testing
Apply FDR control on gene-level p-values from the primary model.

---

## 7) Why this beats Stouffer (conceptual)

Stouffer assumes guide \(z\)-scores are (approximately) independent draws from a common distribution under the null and aggregates *evidence* but not a coherent *effect model*. The mixed model upgrade:
- estimates the gene effect directly at the observation level,
- learns how much guides disagree (random slope variance),
- and can robustify against contamination.

---

## 8) Recommended “best laid” analysis plans (final)

### Plan A (Primary): Observation-level RI+RS mixed model with LRT
- Model: random intercepts and random slopes per guide
- Estimand: \(\theta_g\) mean slope
- Inference: LRT for \(\theta_g=0\)
- Outputs: effect size + heterogeneity + QC diagnostics
- Fallback: RI if RI+RS unstable

### Plan B (Low-overhead validation): Summary-level random-effects meta
- Inputs: per-guide \(\hat\beta_{gj}\), SE
- Model: \(\hat\beta_{gj}\sim \mathcal{N}(\theta_g, \mathrm{SE}_{gj}^2+\tau_g^2)\)
- Purpose: sanity check and faster approximate runs

### Plan C (Targeted robustness): Heavy-tailed or mixture model for flagged genes
- Trigger: heterogeneity/discordance thresholds
- Output: robust \(\theta_g\) and guide “goodness” weights
- Use: sensitivity report; optionally replace Plan A results for extreme cases if you pre-register that rule

---

## Appendix: decision table (quick reference)

| Decision | Option | Pros | Cons | Recommendation |
|---|---|---|---|---|
| Estimand | common slope | powerful, simple | fragile to heterogeneity | fallback only |
| Estimand | mean slope w/ random slopes | honest uncertainty, interpretable | needs enough guides | **primary** |
| Contamination | heavy-tailed | robust, modest overhead | needs implementation | **primary robustness** |
| Contamination | mixture | explicit on/off target | compute/fragility | targeted only |
| Inference | LRT | stable in mixed models | extra fit per gene | **primary** |
| Inference | Wald | cheap | less stable w/ RE | secondary |

---

**Notes.** This plan intentionally does not rely on CRISPR-specific conventions; it follows likelihood-based modeling of repeated imperfect measurements of a shared latent effect.
