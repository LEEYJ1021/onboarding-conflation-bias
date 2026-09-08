"""
================================================================================
Applied Paper - Step 1: Omnibus Test, Holm-Corrected Pairwise Comparisons,
and Regression, Across Multiple Observation Windows
================================================================================
Outcomes (post_day0 versions are primary; day0-inclusive versions are
appendix-only because they are definitionally entangled with primary_arm_v2,
which is itself defined by what happened on day 0):
    - log_spend_cum: cumulative log-spend in [reg_date, reg_date+window)
    - n_types_active_avg_post_day0: average daily active-type count,
      excluding day 0 (the day that defines the arm) and correspondingly
      shortening the denominator by one day
    - time_to_new_type_post_day0: days from day 0 to the first campaign
      type NOT present on day 0 (censored at window edge if none appears)

Design notes:
    - day 0 is anchored to each customer's own first_active_date (the date
      that actually defines primary_arm_v2), not the shared registration_date
      — these differ for customers with reg_to_active_gap_days > 0.
    - Omnibus test gates pairwise testing: Kruskal-Wallis is run first per
      (window, outcome); Holm-corrected Mann-Whitney U pairwise comparisons
      are only run for outcome/window combinations where the omnibus test
      is significant. This blocks the "try enough splits and something
      is p<0.05" failure mode already observed with Track 1 in the
      discovery paper.
    - Reference category is recomputed from live value_counts() each run
      (not trusted from CSV dtype round-trip).

Output: reframe_output/step1_outcomes_by_window.csv
        reframe_output/step1_omnibus_results.csv
        reframe_output/step1_pairwise_results.csv
        reframe_output/step1_regression_results.csv
        reframe_output/step1_manifest.json
================================================================================
"""
import os
import json
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
OUT_DIR = Path(os.environ.get("REFRAME_OUT", str(ROOT / "reframe_output")))
EARLY_WINDOWS = [int(x) for x in os.environ.get("EARLY_WINDOWS_DAYS", "7,14,30,60").split(",")]
ALPHA = float(os.environ.get("ALPHA", "0.05"))


def _detect_date_column(df):
    for c in ("date", "stat_date", "stat_dt", "dt", "ad_date", "report_date", "log_date"):
        if c in df.columns:
            return c
    raise KeyError("Could not auto-detect a date column")


ctp = pd.read_csv(ROOT / "customer_day_campaign_type_panel.csv")
attrs = pd.read_csv(ROOT / "customer_level_attributes.csv")
sample = pd.read_csv(OUT_DIR / "analysis_ready_sample_v2.csv")

DATE_COL_CTP = _detect_date_column(ctp)
ctp[DATE_COL_CTP] = pd.to_datetime(ctp[DATE_COL_CTP])
ctp["campaign_type"] = pd.to_numeric(ctp["campaign_type"], errors="coerce")
sample["registration_date"] = pd.to_datetime(sample["registration_date"])
sample["primary_arm_v2"] = sample["primary_arm_v2"].astype(str)  # ignore stale categorical dtype

sample_ids = set(sample["customer_id"].unique())
ctp_s = ctp[ctp["customer_id"].isin(sample_ids) & (ctp["cost"] > 0)].copy()
reg_map = sample.set_index("customer_id")["registration_date"].to_dict()

arm_size = sample["primary_arm_v2"].value_counts()
level_order = list(arm_size.index)
reference_level = level_order[0]
print(f"primary_arm_v2 levels (recomputed): {arm_size.to_dict()}")
print(f"Regression reference category: '{reference_level}'\n")

has_device = "device_type_mode" in attrs.columns
attrs_cols = ["customer_id"] + (["device_type_mode"] if has_device else [])
sample_full = sample.merge(attrs[attrs_cols], on="customer_id", how="left")
sample_full["reg_month"] = sample_full["registration_date"].dt.to_period("M").astype(str)
meta = sample_full[["customer_id", "primary_arm_v2", "reg_month"] + (["device_type_mode"] if has_device else [])]


def compute_outcomes_for_window(window_days: int) -> pd.DataFrame:
    rows = []
    for cid in sorted(sample_ids):
        reg_date = reg_map[cid]
        cust_ctp = ctp_s[ctp_s["customer_id"] == cid]
        arm_defining_date = cust_ctp[DATE_COL_CTP].min() if not cust_ctp.empty else reg_date

        day0_types = set(cust_ctp.loc[cust_ctp[DATE_COL_CTP] == arm_defining_date, "campaign_type"]
                          .dropna().unique().astype(int))

        window_end = reg_date + pd.Timedelta(days=window_days)
        full_sub = cust_ctp[(cust_ctp[DATE_COL_CTP] >= reg_date) & (cust_ctp[DATE_COL_CTP] < window_end)]
        log_spend_cum = float(np.log1p(full_sub["cost"].sum()))

        post_sub = full_sub[full_sub[DATE_COL_CTP] != arm_defining_date]
        if post_sub.empty:
            n_types_avg_post = 0.0
        else:
            daily = post_sub.groupby(DATE_COL_CTP)["campaign_type"].nunique()
            idx = pd.date_range(reg_date, periods=window_days, freq="D")
            idx = idx[idx != arm_defining_date]
            n_types_avg_post = float(daily.reindex(idx, fill_value=0).mean())

        new_type_sub = post_sub[~post_sub["campaign_type"].isin(day0_types)]
        if new_type_sub.empty:
            t_new_post, censored = float(window_days), True
        else:
            t_new_post = float((new_type_sub[DATE_COL_CTP].min() - arm_defining_date).days)
            censored = False

        rows.append({
            "customer_id": cid, "window_days": window_days, "log_spend_cum": log_spend_cum,
            "n_types_active_avg_post_day0": n_types_avg_post,
            "time_to_new_type_post_day0": t_new_post,
            "time_to_new_type_post_day0_censored": censored,
            "n_day0_types": len(day0_types),
            "reg_to_active_gap_days": (arm_defining_date - reg_date).days,
        })
    return pd.DataFrame(rows)


outcomes_all = pd.concat([compute_outcomes_for_window(w) for w in EARLY_WINDOWS], ignore_index=True)
outcomes_all = outcomes_all.merge(meta, on="customer_id", how="left")
outcomes_all.to_csv(OUT_DIR / "step1_outcomes_by_window.csv", index=False)

day0_check = outcomes_all[outcomes_all["window_days"] == EARLY_WINDOWS[0]].groupby("primary_arm_v2")["n_day0_types"].mean()
print("Diagnostic: mean day-0 active-type count by arm (should equal 1 for single_type1, 2 for multi levels):")
print(day0_check.to_string())

OUTCOME_VARS = ["log_spend_cum", "n_types_active_avg_post_day0", "time_to_new_type_post_day0"]

# --- Omnibus (Kruskal-Wallis) ---
omnibus_rows = []
for window_days in EARLY_WINDOWS:
    sub_w = outcomes_all[outcomes_all["window_days"] == window_days]
    for outcome in OUTCOME_VARS:
        groups = [sub_w.loc[sub_w["primary_arm_v2"] == lv, outcome].dropna().values for lv in level_order]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        h_stat, p_val = stats.kruskal(*groups)
        omnibus_rows.append({"window_days": window_days, "outcome": outcome, "H_stat": h_stat,
                              "p_value": p_val, "significant": bool(p_val < ALPHA)})
        print(f"  window={window_days:>2}d {outcome:<28} H={h_stat:.3f} p={p_val:.4f} "
              f"{'-> pairwise' if p_val < ALPHA else '-> skip'}")
omnibus_df = pd.DataFrame(omnibus_rows)
omnibus_df.to_csv(OUT_DIR / "step1_omnibus_results.csv", index=False)


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    pooled_std = np.sqrt(pooled_var)
    return float((np.mean(x) - np.mean(y)) / pooled_std) if pooled_std > 0 else np.nan


# --- Pairwise (Holm-corrected within outcome) ---
pairwise_records = []
for outcome in OUTCOME_VARS:
    sig_windows = omnibus_df.loc[(omnibus_df["outcome"] == outcome) & omnibus_df["significant"], "window_days"].tolist()
    for window_days in sig_windows:
        sub_w = outcomes_all[outcomes_all["window_days"] == window_days]
        for lv_a, lv_b in combinations(level_order, 2):
            x = sub_w.loc[sub_w["primary_arm_v2"] == lv_a, outcome].dropna().values
            y = sub_w.loc[sub_w["primary_arm_v2"] == lv_b, outcome].dropna().values
            if len(x) < 2 or len(y) < 2:
                continue
            u_stat, p_raw = stats.mannwhitneyu(x, y, alternative="two-sided")
            pairwise_records.append({"outcome": outcome, "window_days": window_days, "level_a": lv_a,
                                      "level_b": lv_b, "n_a": len(x), "n_b": len(y),
                                      "cohens_d": cohens_d(x, y), "p_raw": p_raw})
pairwise_df = pd.DataFrame(pairwise_records)
if not pairwise_df.empty:
    chunks = []
    for outcome, grp in pairwise_df.groupby("outcome"):
        reject, p_holm, _, _ = multipletests(grp["p_raw"].values, alpha=ALPHA, method="holm")
        grp = grp.copy(); grp["p_holm"] = p_holm; grp["significant_holm"] = reject
        chunks.append(grp)
    pairwise_df = pd.concat(chunks, ignore_index=True)
pairwise_df.to_csv(OUT_DIR / "step1_pairwise_results.csv", index=False)

# --- Regression (HC1) ---
regression_rows = []
for window_days in EARLY_WINDOWS:
    sub_w = outcomes_all[outcomes_all["window_days"] == window_days].copy()
    for outcome in OUTCOME_VARS:
        formula = f"{outcome} ~ C(primary_arm_v2, Treatment(reference='{reference_level}')) + C(reg_month)"
        if has_device:
            formula += " + C(device_type_mode)"
        model = smf.ols(formula, data=sub_w).fit(cov_type="HC1")
        for term in model.params.index:
            if "primary_arm_v2" not in term:
                continue
            regression_rows.append({"window_days": window_days, "outcome": outcome, "term": term,
                                     "coef": model.params[term], "se_hc1": model.bse[term],
                                     "p_value": model.pvalues[term]})
pd.DataFrame(regression_rows).to_csv(OUT_DIR / "step1_regression_results.csv", index=False)

with open(OUT_DIR / "step1_manifest.json", "w") as f:
    json.dump({"early_windows_days": EARLY_WINDOWS, "alpha": ALPHA, "level_order": level_order,
               "reference_level": reference_level,
               "n_omnibus_significant": int(omnibus_df["significant"].sum()) if len(omnibus_df) else 0},
              f, indent=2, default=str)
print("\nSaved step1_outcomes_by_window.csv, step1_omnibus_results.csv, "
      "step1_pairwise_results.csv, step1_regression_results.csv")
