"""
================================================================================
Homogeneity Test Robustness: Asymptotic Chi-square vs. Margin-Fixed
Monte Carlo Exact Test
================================================================================
Purpose:
    Some campaign types have small cells (e.g., type2 with n=19, or
    switcher counts of 0-1 at high thresholds), so the chi-square test's
    asymptotic approximation may be unreliable. scipy has no RxC Fisher
    exact test, so we implement a margin-fixed Monte Carlo permutation
    test: fix the row totals (born-treated / true-switcher) and column
    totals (per-type adopter counts), redraw N_MC random tables from the
    corresponding multivariate hypergeometric distribution, and compute
    the share of random tables whose chi-square statistic is at least as
    extreme as the one observed.

Input:  step2_treatment_output/all_types/gap_threshold_sensitivity_summary.csv
        (produced by step2f_gap_threshold_sensitivity.py)
Output: step2_treatment_output/all_types/homogeneity_chisq_vs_mc_exact.csv
        step2_treatment_output/all_types/homogeneity_chisq_vs_mc_exact.json
================================================================================
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))
ALL_TYPES_DIR = STEP2_DIR / "all_types"

SENS_PATH = ALL_TYPES_DIR / "gap_threshold_sensitivity_summary.csv"
if not SENS_PATH.exists():
    raise FileNotFoundError(f"{SENS_PATH} not found — run step2f_gap_threshold_sensitivity.py first.")

sens_df = pd.read_csv(SENS_PATH)
N_MC = int(os.environ.get("N_MC_PERMUTATIONS", "50000"))
SEED = int(os.environ.get("MC_EXACT_SEED", "20260908"))
rng = np.random.default_rng(SEED)
print(f"Monte Carlo permutations: {N_MC:,}\n")


def random_table_fixed_margins(col_sums: np.ndarray, total_born: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a random 2xK table with fixed row/column margins via sequential
    hypergeometric sampling (equivalent to conditioning on both margins)."""
    K = len(col_sums)
    remaining_born, remaining_total = int(total_born), int(col_sums.sum())
    born_counts = np.empty(K, dtype=int)
    for i in range(K):
        n_i = int(col_sums[i])
        born_counts[i] = 0 if remaining_total <= 0 else rng.hypergeometric(
            ngood=remaining_born, nbad=remaining_total - remaining_born, nsample=n_i
        )
        remaining_born -= born_counts[i]
        remaining_total -= n_i
    return np.vstack([born_counts, col_sums - born_counts])


def chi2_statistic(table: np.ndarray) -> float:
    row_sums, col_sums = table.sum(axis=1, keepdims=True), table.sum(axis=0, keepdims=True)
    total = table.sum()
    if total == 0:
        return 0.0
    expected = row_sums @ col_sums / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    return float(terms.sum())


results = []
for threshold, g in sens_df.groupby("gap_threshold_days"):
    g = g.sort_values("campaign_type")
    table = g[["n_born_treated", "n_true_switcher"]].to_numpy().T
    col_sums, total_born, total_n = table.sum(axis=0), int(table[0].sum()), int(table.sum())
    if table.shape[1] < 2 or total_n == 0:
        continue

    chi2_obs, chi2_pval_asymp, dof, expected = sps.chi2_contingency(table.T)
    min_expected = expected.min()

    chi2_null = np.array([chi2_statistic(random_table_fixed_margins(col_sums, total_born, rng))
                           for _ in range(N_MC)])
    mc_pval = float((chi2_null >= chi2_obs - 1e-9).mean())
    mc_se = float(np.sqrt(mc_pval * (1 - mc_pval) / N_MC))
    agree = (chi2_pval_asymp >= 0.05) == (mc_pval >= 0.05)

    print(f"threshold={threshold}d: chi2={chi2_obs:.4f} (min expected cell={min_expected:.2f})")
    print(f"  asymptotic p={chi2_pval_asymp:.4f} | MC exact p={mc_pval:.4f} (+/-{1.96*mc_se:.4f}) "
          f"-> agree at alpha=.05: {'yes' if agree else 'NO'}")

    results.append({
        "gap_threshold_days": threshold, "chi2_observed": chi2_obs,
        "min_expected_cell": float(min_expected), "asymptotic_pvalue": float(chi2_pval_asymp),
        "mc_exact_pvalue": mc_pval, "mc_exact_pvalue_95ci_halfwidth": 1.96 * mc_se,
        "n_mc_permutations": N_MC, "agree_at_alpha_05": bool(agree),
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.round(4).to_string(index=False))

all_agree = bool(results_df["agree_at_alpha_05"].all()) if not results_df.empty else None
results_df.to_csv(ALL_TYPES_DIR / "homogeneity_chisq_vs_mc_exact.csv", index=False)
with open(ALL_TYPES_DIR / "homogeneity_chisq_vs_mc_exact.json", "w") as f:
    json.dump({"n_mc_permutations": N_MC, "seed": SEED, "results_by_threshold": results,
               "all_agree_at_alpha_05": all_agree}, f, indent=2, default=str)
print(f"\nAll thresholds agree (asymptotic vs. exact): {all_agree}")
print("Saved: homogeneity_chisq_vs_mc_exact.csv/.json")
