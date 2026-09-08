# Onboarding Conflation Bias

**A structural treatment-misclassification problem in observational platform panels, and a diagnostic protocol to detect and correct it.**

> This repository contains the full, already-executed analysis pipeline — data-integrity verification, treatment-timing reconstruction, the discovery and formalization of *Onboarding Conflation Bias*, Monte Carlo validation, cross-type generalization, and two downstream demonstration analyses (Track 1 / Track 2) — together with every figure, table, and script that produced them.

---

## Table of contents

1. [Why this repository exists](#1-why-this-repository-exists)
2. [The core problem, in one picture](#2-the-core-problem-in-one-picture)
3. [Theoretical background and positioning](#3-theoretical-background-and-positioning)
4. [Definitions](#4-definitions)
5. [Research questions and propositions](#5-research-questions-and-propositions)
6. [Data and sample construction](#6-data-and-sample-construction)
7. [Treatment-timing reconstruction and left-censoring recovery](#7-treatment-timing-reconstruction-and-left-censoring-recovery)
8. [The Gap-Day Diagnostic and the Track 1 / Track 2 split](#8-the-gap-day-diagnostic-and-the-track-1--track-2-split)
9. [Monte Carlo validation](#9-monte-carlo-validation)
10. [Cross-type generalization and homogeneity testing](#10-cross-type-generalization-and-homogeneity-testing)
11. [Gap-threshold sensitivity](#11-gap-threshold-sensitivity)
12. [Pre-trend analysis](#12-pre-trend-analysis)
13. [Full robustness inventory](#13-full-robustness-inventory)
14. [Figures 1–11](#14-figures-1-11)
15. [Tables 1–8](#15-tables-1-8)
16. [Key results — one-paragraph summary](#16-key-results--one-paragraph-summary)
17. [Methodological and practical contributions](#17-methodological-and-practical-contributions)
18. [Limitations and future research](#18-limitations-and-future-research)
19. [Repository structure](#19-repository-structure)
20. [Code-to-result mapping](#20-code-to-result-mapping)
21. [Companion / exploratory pipeline (not part of the main paper)](#21-companion--exploratory-pipeline-not-part-of-the-main-paper)
22. [Reproducibility framework](#22-reproducibility-framework)

---

## 1. Why this repository exists

This project began as an attempt to estimate the causal effect of a single advertising-campaign type ("type 6") being **newly adopted** by existing customers, using a standard staggered-adoption difference-in-differences (DiD) design on a platform's customer-day panel. The first, entirely standard attempt collapsed:

```
32 nominal "new adopters" → 9 stacks with a usable pre-period → 7 within-customer switchers
```

Diagnosing *why* this collapse happened is the actual contribution of this repository. The short version: a stable-observation-window filter, applied for the ordinary purpose of guaranteeing a clean panel, systematically **misclassifies new-customer onboarding as an existing-customer treatment-adoption event**. We name this **Onboarding Conflation Bias**, build a statistic (the **Born-Treated Ratio**) and a diagnostic procedure (the **Gap-Day Diagnostic**) to detect it, validate the mechanism with a closed-form Monte Carlo simulation, and show the bias reproduces at a statistically indistinguishable rate across four different campaign types on this platform (mean 81.3%, homogeneity test p > 0.14 for every gap-day threshold tested, both under an asymptotic chi-square test and a margin-fixed Monte Carlo exact test).

The failed original question is not thrown away — it becomes the running example ("Track 1") that shows the protocol correctly isolates the small set of customers for whom within-customer causal inference is even possible, while the discarded majority ("Track 2") is repurposed into an honest, appropriately-hedged descriptive comparison.

## 2. The core problem, in one picture

<p align="center"><img src="figures/fig11_diagnostic_protocol_flowchart.png" width="480"></p>

Every stage of that flowchart corresponds to a script in this repository (see [§20](#20-code-to-result-mapping)) and to a section of this document.

## 3. Theoretical background and positioning

**Staggered-adoption DiD.** The empirical strategy this project originally set out to use is squarely in the tradition of Callaway & Sant'Anna (2021)-style staggered-adoption designs: define cohorts by first-treatment date, use never-treated units as the comparison group, and aggregate cohort-time ATTs. That literature has, in the last decade, thoroughly diagnosed the *estimation* pathologies of naive two-way-fixed-effects staggered designs (negative weighting, forbidden comparisons, etc.). It has *not*, to our knowledge, addressed a prior-stage problem: what happens when the **definition of the treatment-adoption event itself** is contaminated by how the observation panel was constructed.

**Treatment-timing misclassification.** A separate literature on measurement error in treatment timing (e.g., mismeasured adoption dates in policy-diffusion studies) shows that DiD estimates are highly sensitive to getting `g` (the cohort-defining date) wrong. That literature generally treats mistiming as classical noise. Onboarding Conflation Bias is not noise — it is a **directional, structural** misclassification produced by the *interaction* between an observation-window filter and a rolling-admission (continuously arriving new customers) panel.

**Rolling admission in observational platform data.** Platform panels differ from the closed cohorts common in policy-diffusion settings: new units (customers) enter continuously, and any window-based sample-construction rule (stability filters, minimum-observation-length filters, "clean sample" filters of the kind used throughout applied work) will, by construction, disproportionately retain units whose observation window happens to start near a portfolio milestone — including their very first day on the platform. We did not find a paper that names or formalizes this specific interaction; we treat this as the gap this paper fills.

**Positioning.** This is a **methodological discovery / design-science contribution**, not a substantive claim about advertising-campaign effectiveness. The type-6 case study is the empirical setting in which the phenomenon was *found and diagnosed* — it is not the object of study.

## 4. Definitions

**Onboarding Conflation Bias.** The systematic misclassification of a new customer's account-opening / onboarding event as an existing customer's treatment-adoption event, arising from the interaction between (a) an observation-window sample-selection rule and (b) continuous ("rolling-admission") customer entry into the panel. It results in inflated nominal adopter counts, collapsed effective (within-customer-identifiable) sample sizes, and, if unaddressed, standard-error understatement and/or non-causal "effects" driven by onboarding-specific dynamics rather than the treatment itself.

**Born-treated (customer).** A customer whose first appearance in the raw, unfiltered panel (`raw_panel_date_min`) coincides with (or is within a small tolerance of) their first-recorded activation of the focal treatment — i.e., there is no pre-treatment history to observe because the customer's panel history *begins* with the treatment already active. Formally, letting `gap_days = first_treated_date − raw_panel_date_min`, a customer is classified born-treated when `gap_days ≤ GAP_THRESHOLD` (adopted value: 1 day; see [§11](#11-gap-threshold-sensitivity) for the sensitivity of every downstream result to this choice).

**True switcher.** A customer with `gap_days > GAP_THRESHOLD` — i.e., a genuine pre-existing customer who later added the focal treatment, for whom a within-customer pre/post comparison is meaningful.

**Born-Treated Ratio.** The share of all diagnosed "adopters" (after left-censoring recovery) who are born-treated. The paper's central empirical fact is that this ratio is large (≈70–95% depending on campaign type) and **statistically indistinguishable across campaign types** ([Table 4](#table-4)).

**Gap-Day Diagnostic.** The procedure that computes `gap_days` for every nominal adopter recovered from left-censoring and routes them to Track 1 (true switchers, within-customer DiD) or Track 2 (born-treated, cross-sectional matched comparison) based on the threshold. See [Figure 11](#figure-11) and [§8](#8-the-gap-day-diagnostic-and-the-track-1--track-2-split).

## 5. Research questions and propositions

**RQ1.** When an observation-window-filtered panel is used to define staggered treatment adoption, to what extent does the "new adopter" cohort actually consist of customers for whom no pre-treatment period can exist (born-treated), rather than customers who genuinely added the treatment later (true switchers)?

**RQ2.** Is this phenomenon specific to one treatment/campaign type, or is it a structural property of the observation-window-plus-rolling-admission design that generalizes across treatment types?

**RQ3.** What are the inferential consequences of not diagnosing this — specifically, for (a) within-customer DiD estimators (bias vs. effective-sample-size / standard-error consequences) and (b) cross-sectional comparisons that unavoidably include born-treated units?

**P1 (Prevalence).** In an observation-window-filtered rolling-admission panel, the born-treated ratio among nominally "newly adopted" units will be large (we do not commit to an exact magnitude ex ante, but expect it to exceed 50%).

**P2 (Homogeneity / structural, not treatment-specific).** The born-treated ratio will not differ significantly across treatment (campaign) types, because the mechanism is a property of the sample-construction rule and the rolling-admission process, not of the treatment itself.

**P3 (Threshold robustness).** The qualitative conclusions in P1–P2 will be insensitive to the exact `gap_days` cutoff used to classify born-treated vs. true switcher.

**P4 (Track 1 consequence — effective-N collapse, not point-estimate bias).** Because within-customer DiD automatically drops born-treated units (they contribute no pre-period), the *point estimate* need not be biased by born-treated contamination, but the *effective identifying sample* will shrink sharply relative to the nominal adopter count, and using the nominal count to compute standard errors will understate uncertainty.

**P5 (Track 2 consequence — genuine bias risk).** Because a cross-sectional comparison of born-treated units against a matched control group cannot difference away an onboarding-specific ("novelty") effect, estimates from this design will be biased upward in the direction of any such effect and can appear spuriously "significant" at conventional thresholds even when no genuine treatment effect exists.

All five propositions are evaluated below and, for P1–P4, supported by both the empirical cross-type analysis and the Monte Carlo simulation; P5 motivates — but is intentionally not adjudicated as a causal claim by — the Track 2 demonstration.

## 6. Data and sample construction

**Inputs** (not included in this repository; paths configured via `config.py` / `AD_DATA_ROOT`): `customer_day_panel.csv`, `customer_day_campaign_type_panel.csv`, `customer_level_attributes.csv` — a daily customer-level panel plus a customer-by-campaign-type-by-day panel from a single advertising platform.

**Sample-selection funnel** (naive/stable-window construction used for the main paper's empirical setting):

| Stage | N remaining | N excluded |
|---|---:|---:|
| Registry-matched accounts | 263 | — |
| Continuous observation block | 108 | 155 |
| Stable window exists | 107 | 1 |
| Non-test / non-billing-anomalous | 104 | 3 |
| Minimum 30-day window rule | **98** | 6 |

Final analysis panel: **98 customers, 15,261 customer-day rows, 99 columns**. Data integrity (file hashes, stage-by-stage re-derivation, leverage-point / zero-spend checks, skewness diagnostics) is independently re-verified by a second script against the manifest emitted by the build script — see [Table 1](#table-1) and [§20](#20-code-to-result-mapping).

The 30-day minimum-window rule itself was added *after* a preliminary diagnostic found one customer with zero all-time spend inside the otherwise-clean sample; rather than an ad hoc exclusion, a principled minimum-observation-length rule was adopted, which also improved the log-spend distribution's standard deviation (+7.2%) at the cost of 0.4% of observations.

## 7. Treatment-timing reconstruction and left-censoring recovery

Naively defining "first adoption" as the first day `cost_type6 > 0` inside the stable window and comparing it to the customer's stable-window start date yields three cohorts for campaign type 6:

| Cohort (1st attempt) | N |
|---|---:|
| Adopted within window ("pure new adoption") | 7 |
| Already active at window start (left-censored) | 30 |
| Never treated within window | 61 |

Seven adopters is far below any usable threshold for cohort-based DiD. Rather than discard the 30 left-censored customers, the pipeline looks *behind* the stable window into the raw, unfiltered panel to recover a truer first-activation date, then runs a four-criterion pre-window safety check (test/anomaly flags, observation continuity, spend-volatility ratio vs. the stable window, extreme-value density) to decide whether the recovered pre-window history is safe to splice into the analysis panel:

| Recovery outcome | N |
|---|---:|
| Recovered, SAFE | 25 |
| Recovered, CAUTION | 3 |
| Still left-censored (unrecoverable) | 2 |

**Final cohort classification** (see [Table 2](#table-2) / [Figure 2](#figure-2)): 32 "primary adopters" (SAFE + within-window), 3 CAUTION-only adopters, 2 unrecoverable, 61 never-treated. A formal pre-trend test (linear trend in the pre-period, cluster-robust by customer, split into near-window [-15,-1] and far-window [<-15] sub-periods) finds **no significant pre-trend** in either the primary (n=32) or robust (n=35, including CAUTION) cohort definition, for either outcome variable — supporting the parallel-trends assumption for the estimation that follows, and confirming that the SAFE/CAUTION recovery classification itself is not driving results (the two cohort definitions converge to the same conclusion).

Running the (Callaway & Sant'Anna-style, stacked cohort-level 2×2 DiD) estimator on this 32-customer cohort against the 61 never-treated customers produces the pivotal diagnostic finding: only **9 of the 32 (28%)** nominal adopters contribute a valid, non-degenerate stack (i.e., have both a non-empty pre-period *and* a matching control observation in the same calendar window). A dedicated stack-dropout diagnostic (`step3b_stack_dropout_diagnosis.py`) shows that **100% of the tested dropout cases fail because their pre-treatment period is empty** — not because of calendar-coverage or control-availability issues — and that this exactly coincides with `gap_days ≈ 0`, i.e., born-treated status. This is the discovery moment of Onboarding Conflation Bias in this project.

## 8. The Gap-Day Diagnostic and the Track 1 / Track 2 split

Once every recovered adopter has a `gap_days` value, the diagnostic routes them:

- **`gap_days > threshold` → Track 1 (True Switchers).** A within-customer stacked DiD is run on this subgroup only. For campaign type 6 this leaves **n = 7**. Leave-one-out (LOO) sensitivity analysis shows the sign of the pooled ATT is reasonably stable (1/7 sign flips for cumulative log-spend, 0/7 for portfolio breadth), but the sample is explicitly reported as **case-level / exploratory evidence, not a formal causal estimate** — with all seven individual case-level ATTs shown transparently in [Table 7](#table-7) / [Figure 9](#figure-9). An alternative gap threshold (3 days instead of 1) was tested as a robustness check and, counter to the initial hypothesis that it would remove noisy edge cases, produced a comparably unstable (in fact slightly *more* LOO-unstable for one outcome) n=5 subsample with one coefficient flipping into nominal significance (p=0.022) purely as an artifact of having tried multiple thresholds on a tiny sample — this negative result is reported explicitly as a caution against over-interpreting small-sample threshold-shopping, not as a new finding.

- **`gap_days ≤ threshold` → Track 2 (Born-Treated).** These customers cannot supply a within-customer pre-period by construction, so they are instead compared cross-sectionally against never-treated customers who registered within a matched calendar window (`REG_MATCH_WINDOW_DAYS`, tested at 7/14/21/30 days). For campaign type 6 this covers **n = 28**, matched against up to 61 never-treated candidates. The comparison is explicitly framed throughout as a **descriptive association, not a causal effect** (association-only vocabulary is used in every script and in the paper). Early-window outcomes (cumulative log-spend, average active campaign-type count in the first 30 days) are strongly and significantly higher for born-treated customers (Welch's t, both p < 0.0001, Cohen's d = 1.34 and 2.10 respectively), robust across all four matching-window widths tested, and only partially explained by observed covariates — a regression-adjustment check (customer scale + device type) shrinks the coefficients by 34–56% but the portfolio-breadth result remains significant after adjustment ([Table 8](#table-8) / [Figure 10](#figure-10)).

A separate check confirmed that the covariate used for adjustment (`customer_total_cost_alltime`) partially overlapped in time with the outcome window itself (an outcome-contamination / post-treatment-bias risk); re-running the adjustment with a covariate that strictly excludes the outcome window produces nearly identical shrinkage percentages (54.7% vs. 56.5% for spend; 34.3% vs. 34.1% for breadth), confirming the original adjusted estimates were not an artifact of covariate contamination ([Table 8](#table-8)).

## 9. Monte Carlo validation

A single-platform empirical finding of N=98 customers cannot, by itself, establish that Onboarding Conflation Bias is a *general* phenomenon rather than an artifact of this dataset. `mc_onboarding_conflation_bias.py` formalizes the data-generating process implied by the mechanism and validates both the mechanism and the diagnostic's behavior under it.

**DGP.** `N_ADOPTERS=35` simulated adopters and `N_NEVER=61` never-treated customers (calibrated to the empirical robust cohort). A fraction `p_born` of adopters are born-treated (`gap = 0`, so their pre-period is structurally undefined); the remainder are true switchers with `gap` drawn from `[GAP_MIN, GAP_MAX]` days, set with an explicit safety margin (`GAP_MIN ≥ EVENT_WINDOW + NOVELTY_DURATION`) so that a true switcher's pre-period can never accidentally overlap the "novelty effect" window — a bug present in an earlier draft of this simulation is documented and fixed in the script's changelog, and a sanity check (`p_born=0` ⇒ Track 1 bias ≈ 0, within 3 Monte Carlo standard errors) is run automatically on every execution to guard against regression. `TRUE_EFFECT=0.15` is the ground-truth treatment effect; `NOVELTY_EFFECT=0.6` for the first `NOVELTY_DURATION=10` days after registration represents an onboarding-specific dynamic unrelated to treatment (e.g., a first-purchase / setup effect). 2,000 replications per grid point, `p_born ∈ {0, .2, .4, .6, .74, .8, .9, 1.0}` (0.74–0.80 matching the empirically observed range).

**Findings, at the empirically observed `p_born≈0.74–0.80`:**

- **Track 1 (within-customer DiD): bias stays ≈ 0** across the entire `p_born` grid (confirming P4's point-estimate claim), but the **effective sample collapses** from a nominal 35 to an average of 6.99–9.19 — closely mirroring the empirical 32→9 collapse — and the standard error computed from the (wrong) nominal N understates the correct standard error by **50–57%** at the empirical `p_born`, rising to 68% as `p_born→0.9` ([Figure 6](#figure-6), [Figure 7](#figure-7), [Figure 8](#figure-8)).
- **Track 2 (cross-sectional comparison): bias is stable at ≈ +0.20** for every `p_born > 0`, and matches a **closed-form theoretical prediction** exactly: `NOVELTY_EFFECT × (NOVELTY_DURATION / EVENT_WINDOW) = 0.6 × (10/30) = 0.200`, against a simulated range of [0.198, 0.208] — i.e., the simulator reproduces the intended data-generating mechanism to within Monte Carlo noise. The share of simulations in which this purely onboarding-driven, zero-causal-content bias registers as "statistically significant" at p<0.05 rises with `p_born`, reaching **43.7–47.8%** at the empirically observed range ([Table 6](#table-6)).

This is the theoretical anchor for the empirical Track 1 / Track 2 findings in §8: the simulation shows *why* Track 1's problem is an effective-sample/overconfidence problem while Track 2's problem is a genuine (novelty-driven) bias problem, and it validates that both consequences follow directly and predictably from the born-treated mechanism rather than being idiosyncratic to this dataset.

## 10. Cross-type generalization and homogeneity testing

Repeating the entire diagnostic pipeline (naive staggered adoption → left-censoring recovery → safety check → final cohort → gap-day diagnosis → Step-C stack validity) automatically across every campaign type detected in the master dataset (types 1, 2, 3, 6; types 4/5 have zero ever-active customers in this sample and are excluded on that documented basis) produces:

| Type | Ever active | Final adopters | Born-treated | True switchers | Born-treated ratio | Valid DiD stacks |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 89 | 87 | 70 | 17 | 80.5% | 20 |
| 2 | 19 | 19 | 18 | 1 | 94.7% | 1 |
| 3 | 10 | 10 | 7 | 3 | 70.0% | 3 |
| 6 | 37 | 35 | 28 | 7 | 80.0% | 9 |

Mean born-treated ratio: **81.3%**, standard deviation across types: **10.2%**. Because the standard deviation is well under a pre-registered 10% "structural pattern" heuristic, and because two independent homogeneity tests agree, the paper's central claim is that this is a **platform-wide structural pattern, not a type-6-specific artifact** ([Figure 4](#figure-4)).

**Homogeneity testing, done twice.** A 2×4 contingency-table chi-square test of homogeneity is run at every gap-day threshold, but several cells have expected counts below 5 (as low as 1.39 at threshold=7), so the asymptotic approximation is independently cross-checked with a margin-fixed Monte Carlo exact test (50,000 permutations per threshold, using a sequential-hypergeometric random-table generator that preserves both row and column margins). The two methods agree on the "homogeneous / not homogeneous" call at every threshold tested, and both always conclude homogeneous (p ranges 0.14–0.38 across both methods and all four thresholds) — see [Table 4](#table-4). This defends the paper's central generalization claim against a small-sample-chi-square objection before a reviewer can raise it.

## 11. Gap-threshold sensitivity

Because the born-treated/true-switcher classification hinges on an arbitrary-seeming `gap_days ≤ 1` cutoff, every headline number is recomputed at thresholds of **0, 1, 3, and 7 days**:

| Type | 0d | 1d | 3d | 7d |
|---|---:|---:|---:|---:|
| 1 | 77.0% | 80.5% | 81.6% | 85.1% |
| 2 | 94.7% | 94.7% | 94.7% | 100.0% |
| 3 | 70.0% | 70.0% | 70.0% | 70.0% |
| 6 | 74.3% | 80.0% | 85.7% | 85.7% |

The maximum within-type variation across all four thresholds is 11.4 percentage points (type 6), safely under a 15-point stability rule set in advance. The homogeneity conclusion (§10) is also confirmed to hold at every threshold ([Table 4](#table-4), [Figure 5](#figure-5)).

## 12. Pre-trend analysis

See §7. Reported here for completeness because it is a standard DiD identification check: no significant linear pre-trend in either outcome, in either the full pre-period or the near-adoption anticipation window, for either the primary (n=32) or robustness (n=35) cohort definition.

## 13. Full robustness inventory

| # | Check | Result |
|---|---|---|
| 1 | Data-integrity re-derivation (independent script re-verifies build manifest) | PASS — all stage counts, row counts, and zero-spend checks match |
| 2 | Pre-trend test (full period + near/far windows, cluster-robust) | No significant pre-trend, primary and robust cohorts converge |
| 3 | SAFE-only vs. SAFE+CAUTION left-censoring recovery | ATT sign and magnitude identical (CAUTION customers contribute 0 valid Step-C stacks) |
| 4 | Cross-type replication (types 1/2/3/6) | Homogeneous, 81.3% mean, SD 10.2% |
| 5 | Homogeneity test: asymptotic chi-square vs. margin-fixed Monte Carlo exact | Agree at every threshold tested |
| 6 | Gap-threshold sensitivity (0/1/3/7 days) | Max within-type variation 11.4pp, under the 15pp stability bound |
| 7 | Track 1 leave-one-out (LOO) sensitivity | 1/7 and 0/7 sign flips across the two outcomes at threshold=1 |
| 8 | Track 1 alternative gap threshold (3 vs. 1 day) | Does *not* improve stability; flags a multiple-comparisons risk at n=5 explicitly |
| 9 | Track 2 matching-window sensitivity (7/14/21/30 days) | Significant at every window, Cohen's d 1.18–1.34 (spend) and 2.10–2.23 (breadth) |
| 10 | Track 2 covariate balance (customer scale, device type) | Scale imbalanced (p=0.006) — motivates the regression-adjustment check |
| 11 | Track 2 regression adjustment (structural covariate) | Coefficients shrink 34–56%; breadth result survives, spend result does not |
| 12 | Track 2 outcome-contamination check (covariate excluding outcome window) | Confirms adjustment result is not a contamination artifact (54.7% vs. 56.5%; 34.3% vs. 34.1% shrinkage) |
| 13 | Monte Carlo sanity check (`p_born=0` ⇒ Track 1 bias ≈ 0) | PASS, within 3 Monte Carlo SEs |
| 14 | Monte Carlo theoretical-vs-simulated Track 2 bias | Closed-form 0.200 vs. simulated range [0.198, 0.208] |

## 14. Figures 1–11

Every figure is grayscale / hatch-and-marker-differentiated only (no color-dependent encoding), saved as both 300dpi PNG (`figures/*.png`, embedded below) and vector PDF for print production, generated by `make_paper_assets.py` from already-computed CSV/JSON outputs (it does **not** re-run the analysis pipeline — see [§22](#22-reproducibility-framework)).

### Figure 1
<a name="figure-1"></a>
**Sample-Selection Funnel (Naive Panel Construction)**
![Figure 1](figures/fig01_sample_selection_funnel.png)
- **Role in the paper:** Establishes the starting sample and shows exactly where each of the 165 excluded registry-matched accounts is lost.
- **Key result:** 263 registry-matched accounts → 98 in the final analysis sample, with the largest single loss (155 accounts) occurring at the "continuous observation block" stage.
- **Generated by:** `make_paper_assets.py::fig01_sample_funnel()`, reading `reproducibility_manifest.json`.
- **Underlying computation:** `step1_data_integrity_build.py` / `step1a_build_master_dataset.py`.
- **Reproduce:** run the step1 build script, then `python make_paper_assets.py`.

### Figure 2
<a name="figure-2"></a>
**Final Cohort Classification After Left-Censoring Recovery (Campaign Type 6)**
![Figure 2](figures/fig02_cohort_classification.png)
- **Role:** Shows the outcome of the left-censoring recovery + safety-check procedure.
- **Key result:** Of 33 nominally left-censored-or-adopted customers, 32 end up classified as usable adopters (SAFE + CAUTION + within-window), only 2 remain genuinely unrecoverable.
- **Generated by:** `make_paper_assets.py::fig02_cohort_classification()`, reading `staggered_adoption_FINAL_type6_summary.json`.
- **Underlying computation:** `step2_2_left_censoring_recovery_type6.py` / `step2b_left_censoring_recovery.py`.

### Figure 3
<a name="figure-3"></a>
**Distribution of Adoption Gap Days (Campaign Type 6) — the discovery figure**
![Figure 3](figures/fig03_gap_day_distribution.png)
- **Role:** The single most important figure in the paper — visualizes the discovery that the "adopter" distribution is bimodal, with a massive spike at `gap_days≈0` (born-treated, n=28) and a thin tail of genuine switchers (n=7).
- **Key result:** 80% born-treated at the adopted threshold.
- **Generated by:** `make_paper_assets.py::fig03_gap_day_distribution()`, reading `step3c_born_treated_diagnosis_type6.csv`.
- **Underlying computation:** `step3c_born_treated_diagnosis.py`.

### Figure 4
<a name="figure-4"></a>
**Born-Treated Ratio Is Homogeneous Across Campaign Types**
![Figure 4](figures/fig04_born_treated_ratio_by_type.png)
- **Role:** The generalization figure — shows the phenomenon is not type-6-specific.
- **Key result:** 70–95% across types 1/2/3/6, mean 81.3%.
- **Generated by:** `make_paper_assets.py::fig04_born_treated_ratio_by_type()`, reading `all_types/cohort_summary_all_types.csv`.
- **Underlying computation:** `step2e_all_types_generalization.py`.

### Figure 5
<a name="figure-5"></a>
**Robustness of the Born-Treated Classification to Threshold Choice**
![Figure 5](figures/fig05_gap_threshold_sensitivity.png)
- **Role:** Pre-empts the "why gap≤1 day?" reviewer objection.
- **Key result:** Ratios move by at most ~11pp across thresholds 0/1/3/7 days; ranking across types is stable.
- **Generated by:** `make_paper_assets.py::fig05_gap_threshold_sensitivity()`, reading `all_types/gap_threshold_sensitivity_summary.csv`.
- **Underlying computation:** `step2f_gap_threshold_sensitivity.py`.

### Figure 6
<a name="figure-6"></a>
**Monte Carlo Validation: Track 1 Remains Unbiased, Track 2 Inherits a Novelty-Effect Bias**
![Figure 6](figures/fig06_mc_bias_vs_pborn.png)
- **Role:** Central simulation figure — shows the two tracks' bias behavior diverging as `p_born` rises.
- **Key result:** Track 1 bias ≈ 0 for all `p_born`; Track 2 bias ≈ +0.20 for all `p_born > 0`, matching theory exactly.
- **Generated by:** `make_paper_assets.py::fig06_mc_bias_vs_pborn()`, reading `mc_output/mc_summary_by_p_born.csv`.
- **Underlying computation:** `mc_onboarding_conflation_bias.py`.

### Figure 7
<a name="figure-7"></a>
**Nominal Adopter Count vs. Effective Identifying Sample**
![Figure 7](figures/fig07_mc_effective_n.png)
- **Role:** Illustrates the effective-sample collapse mechanism behind the empirical 32→9 shrinkage.
- **Key result:** At the empirical `p_born≈0.74–0.80`, effective N falls to 7–9 from a nominal 35.
- **Generated by:** `make_paper_assets.py::fig07_mc_effective_n()`.
- **Underlying computation:** `mc_onboarding_conflation_bias.py`.

### Figure 8
<a name="figure-8"></a>
**Naive Standard-Error Understatement**
![Figure 8](figures/fig08_mc_se_overconfidence.png)
- **Role:** Shows the overconfidence consequence of not diagnosing the effective-N collapse.
- **Key result:** SE understated by 50–57% at the empirical `p_born` range, rising to 68% as `p_born→0.9`.
- **Generated by:** `make_paper_assets.py::fig08_mc_se_overconfidence()`.
- **Underlying computation:** `mc_onboarding_conflation_bias.py`.

### Figure 9
<a name="figure-9"></a>
**Track 1: Case-Level Treatment Effects Are Small-N and Sign-Mixed**
![Figure 9](figures/fig09_track1_case_level_att.png)
- **Role:** Full transparency on the n=7 Track 1 result — every individual customer's ATT and leave-one-out sensitivity is shown.
- **Key result:** Pooled mean ATT +0.057 (log-spend) / +0.069 (breadth); LOO sign flips 1/7 and 0/7 respectively.
- **Generated by:** `make_paper_assets.py::fig09_track1_forest_loo()`, reading `step3d_track1_switcher_cases_type6.csv` and `step3e_track1_leave_one_out_type6.csv`.
- **Underlying computation:** `step3d_two_track_analysis.py`, `step3e_robustness_checks.py`.

### Figure 10
<a name="figure-10"></a>
**Track 2: Born-Treated vs. Matched-Control Comparison**
![Figure 10](figures/fig10_track2_comparison.png)
- **Role:** The Track 2 headline descriptive-association result.
- **Key result:** Born-treated customers show significantly higher 30-day cumulative spend and portfolio breadth than matched never-treated controls (both p<0.0001).
- **Generated by:** `make_paper_assets.py::fig10_track2_comparison()`, reading `step3d_track2_born_treated_comparison_type6.csv`.
- **Underlying computation:** `step3d_two_track_analysis.py`.

### Figure 11
<a name="figure-11"></a>
**The Gap-Day Diagnostic Protocol**
![Figure 11](figures/fig11_diagnostic_protocol_flowchart.png)
- **Role:** The formalized, reusable diagnostic procedure — the paper's core methodological deliverable, illustrating reusability across settings.
- **Key result:** N/A (schematic).
- **Generated by:** `make_paper_assets.py::fig11_protocol_flowchart()` (hand-specified schematic, not data-driven).

## 15. Tables 1–8

### Table 1
<a name="table-1"></a>
**Sample selection funnel** — `tables/table01_sample_selection_stages.csv`. See §6. Generated by `make_paper_assets.py::table01_sample_selection()` from `reproducibility_manifest.json` (source: `step1_data_integrity_build.py`).

### Table 2
<a name="table-2"></a>
**Final cohort classification (type 6)** — `tables/table02_final_cohort_classification.csv`. See §7. Source: `step2_2_left_censoring_recovery_type6.py`.

### Table 3
<a name="table-3"></a>
**Cross-type generalization summary** — `tables/table03_cross_type_summary.csv`. See §10. Source: `step2e_all_types_generalization.py`.

### Table 4
<a name="table-4"></a>
**Homogeneity tests: asymptotic chi-square vs. Monte Carlo exact** — `tables/table04_homogeneity_tests.csv`. See §10. Source: `step2g_homogeneity_chisq_vs_exact.py`.

### Table 5
<a name="table-5"></a>
**Gap-threshold sensitivity by campaign type** — `tables/table05_gap_threshold_sensitivity.csv`. See §11. Source: `step2f_gap_threshold_sensitivity.py`.

### Table 6
<a name="table-6"></a>
**Monte Carlo summary by born-treated share** — `tables/table06_monte_carlo_summary.csv`. See §9. Source: `mc_onboarding_conflation_bias.py`.

### Table 7
<a name="table-7"></a>
**Track 1 case-level ATT** — `tables/table07_track1_case_level_att.csv`. See §8. Source: `step3d_two_track_analysis.py`.

### Table 8
<a name="table-8"></a>
**Track 2 regression comparison (unadjusted / structural covariate / outcome-window-excluded covariate)** — `tables/table08_track2_regression_comparison.csv`. See §8. Source: `step3e_robustness_checks.py`, `step3g_track2_clean_covariate.py`.

## 16. Key results — one-paragraph summary

A naive staggered-adoption DiD on this platform's data would report 32 "new adopters" of campaign type 6; in reality, only 9 of them have any usable pre-treatment period, and only 7 are genuine pre-existing customers who added the treatment (the rest were new customers whose very first day on the platform was already the treatment). This ratio (born-treated ≈ 80%) reproduces, statistically indistinguishably, across every other campaign type on the platform (mean 81.3%, homogeneity p > 0.14 under both an asymptotic chi-square and a margin-fixed Monte Carlo exact test, at every gap-day threshold tested from 0 to 7 days). A closed-form Monte Carlo simulation confirms the mechanism is structural rather than dataset-specific: within-customer DiD point estimates stay unbiased as the born-treated share rises, but the effective identifying sample collapses (mirroring the empirical 32→9 pattern) and naive standard errors understate uncertainty by 50%+; cross-sectional comparisons of the excluded born-treated majority, in contrast, inherit a genuine, theoretically-predicted bias from onboarding-specific dynamics that can register as spuriously significant at conventional thresholds in roughly half of simulated samples at the empirically observed contamination rate.

## 17. Methodological and practical contributions

**Methodological.** (1) Names and formalizes a previously undocumented interaction between observation-window sample construction and rolling-admission platform panels. (2) Introduces a named, reusable diagnostic statistic (Born-Treated Ratio) and a named, algorithmic diagnostic procedure (Gap-Day Diagnostic) that can be applied to any staggered-adoption design on continuously-enrolling panel data. (3) Provides a validated closed-form theoretical benchmark for the size of the resulting bias in cross-sectional ("Track 2"-style) designs, and a validated account of the effective-sample/overconfidence consequence in within-unit ("Track 1"-style) designs.

**Practical / design-science.** The bias is, in principle, entirely preventable at the data-collection stage: if a platform records a customer's **original/first-acquisition channel or campaign type** as a separate, immutable field at account creation, the born-treated/true-switcher distinction becomes directly observable rather than something that must be reverse-engineered from panel-entry timing. This is offered as a concrete system-design recommendation.

## 18. Limitations and future research

- **Single platform, N=98 core analysis sample.** The Monte Carlo simulation and the four-type internal replication partially compensate for external-validity concerns, but the finding should be replicated on other platforms before the "structural, general phenomenon" claim is treated as fully established.
- **Track 1 is underpowered by construction.** n=7 (or n=5 under the alternative threshold) does not support formal causal inference; it is reported as case-level, exploratory evidence only, and the paper is explicit that this is a *feature* of correctly applying the diagnostic (it would be dishonest to force a formal estimate out of a sample this size), not a shortcoming of the method.
- **Track 2 is a descriptive association, not a causal estimate.** Self-selection into a first campaign type is not addressed by matching on calendar registration window and observed covariates alone; the covariate-balance check finds a real imbalance in customer scale, and the regression-adjustment shrinkage (34–56%) should be read as an upper bound on how much of the raw comparison survives adjustment for observables, not as a causal effect size.
- **Types 4 and 5 were excluded** because zero customers in this sample ever had positive spend under those types — a data-availability limitation, documented rather than concealed.
- **Future work:** replicate the diagnostic on other platforms and treatment settings; formally extend the closed-form Track 2 bias result to a general novelty-effect functional form; explore whether a design-based fix (e.g., using platform-recorded first-acquisition-channel fields, where available) fully eliminates the need for the gap-day heuristic.

## 19. Repository structure

The pipeline is organized into six functional groups, executed strictly in order (Group A → Group F). Each group is self-contained: it reads only the outputs of earlier groups plus its own raw inputs, and writes outputs that later groups consume. A separate, non-overlapping companion pipeline (Group G) explores a secondary research question and is intentionally kept out of the main dependency chain (see [§21](#21-companion--exploratory-pipeline-not-part-of-the-main-paper)).

```
onboarding-conflation-bias/
├── README.md                              # this file
├── config.py                              # shared paths/env-vars (AD_DATA_ROOT, STEP2_OUT, MC_OUT, REFRAME_OUT, ...)
│
├── figures/                                # Figures 1–11 (PNG; PDF companions produced by make_paper_assets.py)
│   ├── fig01_sample_selection_funnel.png
│   ├── fig02_cohort_classification.png
│   ├── fig03_gap_day_distribution.png
│   ├── fig04_born_treated_ratio_by_type.png
│   ├── fig05_gap_threshold_sensitivity.png
│   ├── fig06_mc_bias_vs_pborn.png
│   ├── fig07_mc_effective_n.png
│   ├── fig08_mc_se_overconfidence.png
│   ├── fig09_track1_case_level_att.png
│   ├── fig10_track2_comparison.png
│   └── fig11_diagnostic_protocol_flowchart.png
│
├── tables/                                 # Tables 1–8 (CSV; .tex companions produced by make_paper_assets.py)
│   ├── table01_sample_selection_stages.csv
│   ├── table02_final_cohort_classification.csv
│   ├── table03_cross_type_summary.csv
│   ├── table04_homogeneity_tests.csv
│   ├── table05_gap_threshold_sensitivity.csv
│   ├── table06_monte_carlo_summary.csv
│   ├── table07_track1_case_level_att.csv
│   └── table08_track2_regression_comparison.csv
│
├── docs/                                   # supplementary notes (reserved for future manuscript drafts)
│
├── pipeline/                               # ALL CORE SCRIPTS — already executed and validated.
│   │                                        # Do not re-derive results by hand from this README;
│   │                                        # every number above was produced by the script cited next to it.
│   │
│   ├── group_a_data_integrity/             # A — Data integrity: build + independently re-verify the master panel
│   │   ├── step1_data_integrity_build.py       # (a.k.a. step1a_build_master_dataset.py)
│   │   │                                        #   builds df_analysis_master.csv (98 customers / 15,261 rows)
│   │   └── step1_data_integrity_verify.py      # (a.k.a. step1b_verify_master_dataset.py)
│   │                                            #   independently re-verifies the build manifest — PASS
│   │
│   ├── group_b_treatment_timing/           # B — Treatment-timing reconstruction (campaign type 6, the case study)
│   │   ├── step2_1_treatment_first_attempt.py  # (a.k.a. step2a_naive_staggered_adoption.py)
│   │   │                                        #   naive staggered-adoption attempt → 7 adopters, NOT_VIABLE
│   │   ├── step2_2_left_censoring_recovery.py  # (a.k.a. step2b_left_censoring_recovery.py)
│   │   │                                        #   left-censoring recovery + safety check → 32/3/2/61 final cohorts
│   │   └── step2_3_pretrend_test.py            # (a.k.a. step2c_pretrend_test.py)
│   │                                            #   formal pre-trend test → no significant pre-trend
│   │
│   ├── group_c_discovery_diagnostics/      # C — The discovery: gap-day diagnosis and the Track 1 / Track 2 split
│   │   ├── step3_stacked_did_estimation.py     # (a.k.a. step3a_stacked_did_estimation.py)
│   │   │                                        #   Step-C stacked-cohort DiD on 32-customer cohort → 9/32 valid stacks
│   │   ├── step3b_stack_dropout_diagnosis.py   # diagnoses WHY 23/32 stacks are invalid → 100% = empty pre-period
│   │   ├── step3c_born_treated_diagnosis.py    # computes gap_days for every adopter; confirms born-treated status
│   │   ├── step3d_two_track_analysis.py        # Track 1 (n=7 true switchers) + Track 2 (n=28 born-treated, matched)
│   │   ├── step3e_robustness_checks.py         # Track 1 LOO; Track 2 matching-window, covariate balance, adjustment
│   │   ├── step3f_track1_alternative_threshold.py  # alternative gap threshold (3 days) re-estimation + LOO comparison
│   │   └── step3g_track2_clean_covariate.py    # outcome-contamination-free covariate re-check (Track 2 adjustment)
│   │
│   ├── group_d_generalization_robustness/  # D — Cross-type generalization and threshold/homogeneity robustness
│   │   ├── step2e_all_types_generalization.py  # cross-type replication (types 1/2/3/6) → 81.3% mean born-treated ratio
│   │   ├── step2f_gap_threshold_sensitivity.py # gap-threshold (0/1/3/7 day) sensitivity of the born-treated ratio
│   │   └── step2g_homogeneity_chisq_vs_exact.py# chi-square vs. margin-fixed Monte Carlo exact homogeneity test
│   │
│   ├── group_e_monte_carlo/                # E — Theoretical validation via simulation
│   │   └── mc_onboarding_conflation_bias.py    # simulates the bias mechanism and validates the diagnostic under it
│   │
│   └── group_f_manuscript_assets/          # F — Read-only formatting layer (never re-derives numbers)
│       └── make_paper_assets.py                # turns saved CSV/JSON into Figures 1–11 / Tables 1–8
│
└── companion_pipeline/                     # G — Companion / exploratory pipeline (secondary research question — §21)
    │                                        #   NOT part of the Onboarding Conflation Bias manuscript
    ├── step0_reframe_sample_reconstruction.py
    ├── step0b_first_channel_assignment.py
    ├── step0c_cell_size_check.py
    ├── step0d_arm_v2_promotion.py
    ├── step1_omnibus_and_pairwise.py
    ├── step1b_gap_sensitivity_check.py
    ├── step2_heterogeneity_interactions.py
    └── step2a_diagnose_regmonth_cell_sizes.py
```

**Group summary**

| Group | Folder | Purpose | Key output |
|---|---|---|---|
| A | `group_a_data_integrity/` | Build the master panel and independently re-verify it | `df_analysis_master.csv` (98 × 15,261), manifest PASS |
| B | `group_b_treatment_timing/` | Reconstruct true treatment timing for campaign type 6 via left-censoring recovery | 32/3/2/61 final cohort split; no pre-trend |
| C | `group_c_discovery_diagnostics/` | Diagnose the stack-dropout problem, compute `gap_days`, run Track 1 / Track 2 | Born-Treated Ratio, Track 1 (n=7), Track 2 (n=28) |
| D | `group_d_generalization_robustness/` | Replicate the diagnostic across all campaign types; test threshold and homogeneity robustness | 81.3% mean ratio, homogeneous across types/thresholds |
| E | `group_e_monte_carlo/` | Validate the mechanism theoretically, independent of this dataset | Track 1 bias ≈ 0 / SE understatement; Track 2 bias ≈ +0.20 |
| F | `group_f_manuscript_assets/` | Format already-computed results into figures/tables | `figures/*.png`, `tables/*.csv` |
| G | `companion_pipeline/` | Secondary, non-overlapping research question (excluded from the manuscript) | Exploratory only — not cited in §14–15 |

## 20. Code-to-result mapping

**No result in this document was computed fresh for the README.** Every number, table, and figure was produced by re-reading the already-saved CSV/JSON outputs of the scripts below; `make_paper_assets.py` is explicitly designed to *not* re-run the underlying analysis pipeline, so that formatting figures/tables for the manuscript carries zero risk of silently re-deriving (and possibly changing) a result.

| Result | Group | Produced by | Reads from |
|---|---|---|---|
| §6 sample funnel / Table 1 / Figure 1 | A | `step1_data_integrity_build.py` (build) → `step1_data_integrity_verify.py` (independent re-check, PASS) | `customer_day_panel.csv`, `customer_day_campaign_type_panel.csv`, `customer_level_attributes.csv` |
| §7 naive attempt (7 adopters) | B | `step2_1_treatment_first_attempt.py` | `df_analysis_master.csv` |
| §7 left-censoring recovery / Table 2 / Figure 2 | B | `step2_2_left_censoring_recovery.py` | raw panel + `df_analysis_master.csv` |
| §7 pre-trend test | B | `step2_3_pretrend_test.py` | event-time panels from step2_2 |
| §7 Step-C stacked DiD (9/32 valid) | C | `step3_stacked_did_estimation.py` | `staggered_adoption_FINAL_type6.csv` |
| §7 stack-dropout diagnosis | C | `step3b_stack_dropout_diagnosis.py` | outputs of step3 |
| §8 born-treated diagnosis / gap_days / Figure 3 | C | `step3c_born_treated_diagnosis.py` | raw panel, `staggered_adoption_FINAL_type6.csv`, step3b output |
| §8 Track 1 & Track 2 / Table 7 / Figure 9 / Figure 10 | C | `step3d_two_track_analysis.py` | step3c output |
| §8 Track 1 LOO / Track 2 matching & covariate checks | C | `step3e_robustness_checks.py` | step3d output |
| §8 Track 1 alternative threshold | C | `step3f_track1_alternative_threshold.py` | step3c/step3d output |
| §8 Track 2 clean-covariate re-check / Table 8 | C | `step3g_track2_clean_covariate.py` | step3e output |
| §9 Monte Carlo simulation / Table 6 / Figures 6–8 | E | `mc_onboarding_conflation_bias.py` | (self-contained simulation) |
| §10 cross-type generalization / Table 3 / Figure 4 | D | `step2e_all_types_generalization.py` | `df_analysis_master.csv`, raw panel |
| §11 gap-threshold sensitivity / Table 5 / Figure 5 | D | `step2f_gap_threshold_sensitivity.py` | step2e output (`born_treated_diagnosis_type{T}.csv`) |
| §10 homogeneity chi-square vs. exact / Table 4 | D | `step2g_homogeneity_chisq_vs_exact.py` | step2f output |
| Figures 1–11 (formatting only) | F | `make_paper_assets.py` | all of the above (read-only; never re-runs analysis) |

## 21. Companion / exploratory pipeline (not part of the main paper)

During scoping, a second research question was explored — "does a new customer's *first-chosen* campaign type predict their early growth trajectory?" — using the same underlying panel but a completely re-derived, non-overlapping sample (`clean_onboarding_sample`, n=92, built without the 30-day stability filter that causes Onboarding Conflation Bias in the first place, since this design's outcome variable is measurement, not treatment-timing, and does not need it). This produced one robust finding (portfolio-breadth differences across first-campaign-type groups, robust across all four observation windows tested: 7/14/30/60 days) and one fragile, window-dependent finding (a 60-day cumulative-spend difference that is not robust to including a single outlier customer with a 34-day registration-to-activation gap). Because its strongest confirmed result is narrower than the Onboarding Conflation Bias discovery and its overall evidentiary profile is weaker (a single, largely-associational finding vs. a named, formalized, Monte-Carlo-validated, cross-type-replicated phenomenon), it was **deliberately scoped out of the target manuscript** and is retained in this repository only as a secondary, exploratory pipeline (Group G — `companion_pipeline/`) for transparency and potential future use — e.g., as a validated illustration that the Gap-Day-style diagnostic reasoning generalizes to sample-construction problems beyond the born-treated case specifically (this pipeline independently discovers and documents its own version of an outcome-contamination risk and its own small-cell/rank-deficiency diagnostics). **None of its figures or tables appear in Sections 14–15 above**, and none of its results should be cited as part of the Onboarding Conflation Bias findings.

## 22. Reproducibility framework

**Everything in `figures/` and `tables/` is downstream of already-executed and already-verified code.** To reproduce from scratch, run the groups strictly in order (A → F); Group G is independent and optional.

```bash
export AD_DATA_ROOT="/path/to/master_dataset"
export STEP2_OUT="$AD_DATA_ROOT/step2_treatment_output"      # default shown; override if needed
export MC_OUT="./mc_output"
export PAPER_ASSETS_OUT="./paper_assets"

# Group A — Data integrity
python pipeline/group_a_data_integrity/step1_data_integrity_build.py
python pipeline/group_a_data_integrity/step1_data_integrity_verify.py     # must print "PASS" before proceeding

# Group B — Treatment-timing reconstruction (campaign type 6, the discovery case study)
python pipeline/group_b_treatment_timing/step2_1_treatment_first_attempt.py   # confirms naive attempt is NOT_VIABLE (n=7)
python pipeline/group_b_treatment_timing/step2_2_left_censoring_recovery.py   # recovers 32/3/2/61 final cohorts
python pipeline/group_b_treatment_timing/step2_3_pretrend_test.py            # confirms no pre-trend

# Group C — The discovery: stack dropout, gap-day diagnosis, Track 1 / Track 2
python pipeline/group_c_discovery_diagnostics/step3_stacked_did_estimation.py        # Step-C DiD: only 9/32 stacks valid
python pipeline/group_c_discovery_diagnostics/step3b_stack_dropout_diagnosis.py      # WHY: 100% empty-pre-period
python pipeline/group_c_discovery_diagnostics/step3c_born_treated_diagnosis.py       # gap_days computed; born-treated confirmed
python pipeline/group_c_discovery_diagnostics/step3d_two_track_analysis.py           # Track 1 (n=7) / Track 2 (n=28)
python pipeline/group_c_discovery_diagnostics/step3e_robustness_checks.py            # LOO, matching-window, covariate balance, adjustment
python pipeline/group_c_discovery_diagnostics/step3f_track1_alternative_threshold.py
python pipeline/group_c_discovery_diagnostics/step3g_track2_clean_covariate.py

# Group D — Generalization and robustness
python pipeline/group_d_generalization_robustness/step2e_all_types_generalization.py    # replicate across types 1/2/3/6
python pipeline/group_d_generalization_robustness/step2f_gap_threshold_sensitivity.py   # 0/1/3/7-day threshold sensitivity
python pipeline/group_d_generalization_robustness/step2g_homogeneity_chisq_vs_exact.py  # chi-square vs. Monte Carlo exact

# Group E — Theoretical validation
python pipeline/group_e_monte_carlo/mc_onboarding_conflation_bias.py   # Monte Carlo — check "Sanity check ... PASS" in the log

# Group F — Manuscript assets (read-only formatting layer — never re-derives numbers)
python pipeline/group_f_manuscript_assets/make_paper_assets.py   # → paper_assets/figures, paper_assets/tables, generation_log.json

# Group G — Companion / exploratory pipeline (optional, not part of the main manuscript)
# python companion_pipeline/step0_reframe_sample_reconstruction.py
# ... (see §21; run independently, not required for Groups A–F)
```

Each script is idempotent given the same inputs and prints its own verification/sanity checks to the console (e.g., the data-integrity re-verify script prints an explicit `PASS`/`FAIL`; the Monte Carlo script prints an explicit sanity-check pass/fail before reporting any substantive numbers). All environment variables have documented defaults inside each script; none of the reported figures/tables require any manual post-processing beyond what `make_paper_assets.py` performs automatically.
