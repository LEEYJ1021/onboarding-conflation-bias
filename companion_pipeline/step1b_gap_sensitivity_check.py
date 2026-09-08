"""
================================================================================
Applied Paper - Step 1b: Gap-Day Sensitivity Check
================================================================================
Purpose:
    log_spend_cum is measured over [reg_date, reg_date+window). For
    customers whose reg_to_active_gap_days exceeds the window, that whole
    window is spuriously recorded as near-zero spend (out-of-frame
    activity, not genuinely low spend). This script dynamically excludes,
    for each window, any customer whose gap exceeds that window, and
    re-runs the omnibus + Holm-corrected pairwise tests on log_spend_cum
    only, to check whether the window=30 "significant" result depends on
    a single extreme-gap customer.

Output: reframe_output/step1b_gap_sensitivity_results.csv
================================================================================
"""
import os
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
OUT_DIR = Path(os.environ.get("REFRAME_OUT", str(ROOT / "reframe_output")))
ALPHA = float(os.environ.get("ALPHA", "0.05"))

outcomes_all = pd.read_csv(OUT_DIR / "step1_outcomes_by_window.csv")
level_order = list(outcomes_all["primary_arm_v2"].value_counts().index)

results = []
for window_days, grp in outcomes_all.groupby("window_days"):
    excluded_mask = grp["reg_to_active_gap_days"] >= window_days
    n_excluded = int(excluded_mask.sum())
    grp_f = grp.loc[~excluded_mask]
    print(f"window={window_days}d: excluding {n_excluded} customers with gap>=window "
          f"({len(grp)} -> {len(grp_f)})")
    if n_excluded:
        print(grp.loc[excluded_mask, ["customer_id", "primary_arm_v2", "reg_to_active_gap_days"]].to_string(index=False))

    groups = [grp_f.loc[grp_f["primary_arm_v2"] == lv, "log_spend_cum"].dropna().values for lv in level_order]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        continue
    h_stat, p_omni = stats.kruskal(*groups)
    print(f"  omnibus (post-exclusion): H={h_stat:.3f}, p={p_omni:.4f}")

    if p_omni < ALPHA:
        for lv_a, lv_b in combinations(level_order, 2):
            x = grp_f.loc[grp_f["primary_arm_v2"] == lv_a, "log_spend_cum"].dropna().values
            y = grp_f.loc[grp_f["primary_arm_v2"] == lv_b, "log_spend_cum"].dropna().values
            if len(x) < 2 or len(y) < 2:
                continue
            u_stat, p_raw = stats.mannwhitneyu(x, y, alternative="two-sided")
            pooled_std = np.sqrt(((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1)) / (len(x)+len(y)-2))
            d = float((np.mean(x) - np.mean(y)) / pooled_std) if pooled_std > 0 else np.nan
            results.append({"window_days": window_days, "level_a": lv_a, "level_b": lv_b,
                             "n_a": len(x), "n_b": len(y), "n_excluded_this_window": n_excluded,
                             "cohens_d": d, "p_raw": p_raw})

results_df = pd.DataFrame(results)
if not results_df.empty:
    reject, p_holm, _, _ = multipletests(results_df["p_raw"].values, alpha=ALPHA, method="holm")
    results_df["p_holm"] = p_holm
    results_df["significant_holm"] = reject
    print("\n" + results_df.round(4).to_string(index=False))

results_df.to_csv(OUT_DIR / "step1b_gap_sensitivity_results.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'step1b_gap_sensitivity_results.csv'}")
print("Compare significance flips window-by-window against step1_pairwise_results.csv "
      "before treating any single-window log_spend_cum result as confirmed.")
