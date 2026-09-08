"""
================================================================================
Step 3e -- Track 1 Leave-One-Out Sensitivity / Track 2 Matching-Window
           Sensitivity, Covariate Balance, and Regression Adjustment
================================================================================
Background (Step 3d results):
    Track 1 (n=7): the pooled ATT is small and non-significant, but two
    cases with large |gap_days| exert visible influence on the mean; a
    leave-one-out check is needed to see whether the (null) result is
    driven by a couple of outliers.

    Track 2 (n=28 vs 28): the Welch t-test is strong and highly significant
    (p<0.0001, d=1.3-2.1). Because this is a purely cross-sectional
    comparison, three follow-up checks are needed before it can be
    interpreted as anything beyond "a large, possibly confounded,
    association":
        1) Is the result stable to the registration-matching window
           (REG_MATCH_WINDOW_DAYS)?
        2) Are born-treated customers and their matched controls balanced
           on observable characteristics (customer scale, device type)?
        3) Does the association survive covariate-adjusted regression?

This script:
    A. Track 1 leave-one-out -- recompute the pooled mean ATT with each of
       the 7 switchers removed in turn, and report how many drops flip the
       sign of the mean (an instability signal).
    B. Track 2 matching-window sensitivity -- repeat the Welch t-test at
       REG_MATCH_WINDOW_DAYS in {7,14,21,30}.
    C. Track 2 covariate balance -- compare log(customer_total_cost_alltime)
       and device_type_mode between born-treated customers and their
       matched control pool (continuous: Welch t-test; categorical:
       chi-square).
    D. Track 2 regression adjustment -- outcome ~ is_born_treated +
       log(customer_total_cost_alltime) + C(device_type_mode), OLS with
       HC1 robust SEs, to see whether the effect survives covariate control.

Input:  step3d_track1_switcher_cases_type{T}.csv,
        staggered_adoption_FINAL_type{T}.csv,
        step3c_born_treated_diagnosis_type{T}.csv,
        customer_day_panel.csv, customer_day_campaign_type_panel.csv,
        df_analysis_master.csv
Output: step3e_track1_leave_one_out_type{T}.csv
        step3e_track2_window_sensitivity_type{T}.csv
        step3e_track2_covariate_balance_type{T}.csv
        step3e_track2_regression_adjusted_type{T}.json
================================================================================
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
import statsmodels.formula.api as smf

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))

TARGET_TYPE = int(os.environ.get("TARGET_TYPE", "6"))
GAP_THRESHOLD = int(os.environ.get("BORN_TREATED_GAP_THRESHOLD", "1"))
EARLY_WINDOW = int(os.environ.get("EARLY_OUTCOME_WINDOW_DAYS", "30"))

PANEL_PATH = ROOT / "customer_day_panel.csv"
CTP_PATH = ROOT / "customer_day_campaign_type_panel.csv"
MASTER_PATH = ROOT / "df_analysis_master.csv"
FINAL_COHORT_PATH = STEP2_DIR / f"staggered_adoption_FINAL_type{TARGET_TYPE}.csv"
BORN_DIAG_PATH = STEP2_DIR / f"step3c_born_treated_diagnosis_type{TARGET_TYPE}.csv"
TRACK1_CASES_PATH = STEP2_DIR / f"step3d_track1_switcher_cases_type{TARGET_TYPE}.csv"

for p in [PANEL_PATH, CTP_PATH, MASTER_PATH, FINAL_COHORT_PATH, BORN_DIAG_PATH, TRACK1_CASES_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p}")

panel = pd.read_csv(PANEL_PATH)
ctp = pd.read_csv(CTP_PATH)
master = pd.read_csv(MASTER_PATH)
final_cohort = pd.read_csv(FINAL_COHORT_PATH)
born_diag = pd.read_csv(BORN_DIAG_PATH)
track1_cases = pd.read_csv(TRACK1_CASES_PATH)


def _detect_date_column(df: pd.DataFrame) -> str:
    for c in ("date", "stat_date", "stat_dt", "dt", "ad_date", "report_date", "log_date"):
        if c in df.columns:
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c].dropna().iloc[:5])
            return c
        except (ValueError, TypeError):
            continue
    raise KeyError("Could not auto-detect a date column")


DATE_COL = _detect_date_column(panel)
DATE_COL_CTP = _detect_date_column(ctp)
panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])
ctp[DATE_COL_CTP] = pd.to_datetime(ctp[DATE_COL_CTP])
master["date"] = pd.to_datetime(master["date"])
final_cohort["final_first_treated_date"] = pd.to_datetime(final_cohort["final_first_treated_date"])
born_diag["final_first_treated_date"] = pd.to_datetime(born_diag["final_first_treated_date"])
born_diag["raw_panel_date_min"] = pd.to_datetime(born_diag["raw_panel_date_min"])

never_treated_ids = final_cohort.loc[
    final_cohort["final_cohort"] == "never_treated", "customer_id"
].unique().tolist()

# ============================================================================
# A. Track 1 leave-one-out
# ============================================================================
print("=" * 70)
print("A. Track 1 (true switchers) -- leave-one-out sensitivity")
print("=" * 70)

lo_rows = []
for ycol in track1_cases["outcome"].unique():
    sub = track1_cases[track1_cases["outcome"] == ycol].reset_index(drop=True)
    full_mean = sub["att_i"].mean()
    print(f"\n[{ycol}] full-sample mean ATT = {full_mean:+.4f} (n={len(sub)})")
    for _, drop_row in sub.iterrows():
        remaining = sub[sub["customer_id"] != drop_row["customer_id"]]
        loo_mean = remaining["att_i"].mean()
        shift = loo_mean - full_mean
        lo_rows.append({
            "outcome": ycol, "dropped_customer_id": drop_row["customer_id"],
            "dropped_att_i": drop_row["att_i"],
            "loo_mean_att": loo_mean, "shift_from_full_mean": shift,
        })
        flag = " NOTE: large shift" if abs(shift) > abs(full_mean) else ""
        print(f"  dropping {drop_row['customer_id']}: LOO mean={loo_mean:+.4f} "
              f"(shift from full mean = {shift:+.4f}){flag}")

loo_df = pd.DataFrame(lo_rows)
loo_path = STEP2_DIR / f"step3e_track1_leave_one_out_type{TARGET_TYPE}.csv"
loo_df.to_csv(loo_path, index=False)
print(f"\nLOO results saved: {loo_path}")

for ycol in track1_cases["outcome"].unique():
    sub_loo = loo_df[loo_df["outcome"] == ycol]
    sign_changes = (np.sign(sub_loo["loo_mean_att"]) != np.sign(sub_loo["loo_mean_att"].iloc[0])).sum()
    verdict = "-> result is driven by 1-2 individuals and is unstable" if sign_changes >= 2 else "-> comparatively stable"
    print(f"[{ycol}] LOO means with a different sign than the full sample: {sign_changes}/{len(sub_loo)} {verdict}")

# ============================================================================
# 0. Extended outcome panel + customer-level covariates (for Track 2)
# ============================================================================
panel_ext = panel[["customer_id", DATE_COL, "cost"]].copy()
panel_ext["log_spend_safe"] = np.log1p(panel_ext["cost"])
panel_ext = panel_ext.rename(columns={DATE_COL: "date"})

ctp["campaign_type"] = pd.to_numeric(ctp["campaign_type"], errors="coerce")
ctp_active = ctp[ctp["cost"] > 0].copy()
n_types_active = (
    ctp_active.groupby(["customer_id", DATE_COL_CTP])["campaign_type"]
    .nunique().rename("n_campaign_types_active").reset_index()
    .rename(columns={DATE_COL_CTP: "date"})
)
panel_ext = panel_ext.merge(n_types_active, on=["customer_id", "date"], how="left")
panel_ext["n_campaign_types_active"] = panel_ext["n_campaign_types_active"].fillna(0).astype(int)

master_outcomes = master[["customer_id", "date", "log_spend_safe", "n_campaign_types_active"]].copy()
master_outcomes["_source"] = "stable_window(master)"
panel_ext_labeled = panel_ext[["customer_id", "date", "log_spend_safe", "n_campaign_types_active"]].copy()
panel_ext_labeled["_source"] = "raw_panel(pre_or_post_window)"
combined_outcomes = pd.concat([master_outcomes, panel_ext_labeled], ignore_index=True)
combined_outcomes = (
    combined_outcomes.sort_values(
        "_source", key=lambda s: s.map({"stable_window(master)": 0, "raw_panel(pre_or_post_window)": 1})
    ).drop_duplicates(subset=["customer_id", "date"], keep="first").drop(columns="_source")
)

never_treated_reg = panel[panel["customer_id"].isin(never_treated_ids)].groupby("customer_id")[DATE_COL].min()
never_treated_reg = never_treated_reg.rename("registration_date").reset_index()

born_treated = born_diag[born_diag["gap_days"] <= GAP_THRESHOLD][
    ["customer_id", "raw_panel_date_min"]
].rename(columns={"raw_panel_date_min": "registration_date"}).copy()


def early_outcome(cid, reg_date, ycol):
    sub = combined_outcomes[
        (combined_outcomes["customer_id"] == cid)
        & (combined_outcomes["date"] >= reg_date)
        & (combined_outcomes["date"] < reg_date + pd.Timedelta(days=EARLY_WINDOW))
    ]
    if sub.empty:
        return np.nan
    return float(sub[ycol].sum()) if ycol == "log_spend_safe" else float(sub[ycol].mean())


OUTCOME_VARS = ["log_spend_safe", "n_campaign_types_active"]

# ============================================================================
# B. Track 2 matching-window sensitivity
# ============================================================================
print("\n" + "=" * 70)
print("B. Track 2 matching-window (REG_MATCH_WINDOW_DAYS) sensitivity")
print("=" * 70)

sens_rows = []
for win in [7, 14, 21, 30]:
    for ycol in OUTCOME_VARS:
        bt_vals, ctrl_vals = [], []
        for _, bt in born_treated.iterrows():
            cid_bt, reg_bt = bt["customer_id"], bt["registration_date"]
            candidates = never_treated_reg[
                (never_treated_reg["registration_date"] >= reg_bt - pd.Timedelta(days=win))
                & (never_treated_reg["registration_date"] <= reg_bt + pd.Timedelta(days=win))
            ]
            if candidates.empty:
                continue
            bt_val = early_outcome(cid_bt, reg_bt, ycol)
            ctrl_v = [early_outcome(c, reg_bt, ycol) for c in candidates["customer_id"]]
            ctrl_v = [v for v in ctrl_v if pd.notna(v)]
            if pd.notna(bt_val) and ctrl_v:
                bt_vals.append(bt_val)
                ctrl_vals.append(np.mean(ctrl_v))

        if len(bt_vals) >= 3 and len(ctrl_vals) >= 3:
            tstat, pval = sps.ttest_ind(bt_vals, ctrl_vals, equal_var=False)
            pooled_std = np.sqrt((np.var(bt_vals, ddof=1) + np.var(ctrl_vals, ddof=1)) / 2)
            d = (np.mean(bt_vals) - np.mean(ctrl_vals)) / pooled_std if pooled_std > 0 else np.nan
            sens_rows.append({
                "reg_match_window_days": win, "outcome": ycol,
                "n_born_treated": len(bt_vals), "mean_born_treated": np.mean(bt_vals),
                "mean_control": np.mean(ctrl_vals), "tstat": tstat, "pvalue": pval, "cohens_d": d,
            })
            print(f"  window={win:>2}d [{ycol}]: n={len(bt_vals)}, "
                  f"BT mean={np.mean(bt_vals):.3f}, control mean={np.mean(ctrl_vals):.3f}, "
                  f"p={pval:.4f}, d={d:.3f}")

sens_df = pd.DataFrame(sens_rows)
sens_path = STEP2_DIR / f"step3e_track2_window_sensitivity_type{TARGET_TYPE}.csv"
sens_df.to_csv(sens_path, index=False)
print(f"\nSensitivity results saved: {sens_path}")

for ycol in OUTCOME_VARS:
    sub = sens_df[sens_df["outcome"] == ycol]
    if not sub.empty:
        all_sig = (sub["pvalue"] < 0.05).all()
        d_range = f"{sub['cohens_d'].min():.2f} to {sub['cohens_d'].max():.2f}"
        print(f"[{ycol}] significant at every window: {'yes' if all_sig else 'no'}, Cohen's d range: {d_range}")

# ============================================================================
# C. Track 2 covariate balance
# ============================================================================
print("\n" + "=" * 70)
print("C. Track 2 covariate balance (born-treated vs. matched control pool)")
print("=" * 70)

REG_WINDOW_MAIN = int(os.environ.get("REG_MATCH_WINDOW_DAYS", "14"))
level_cols = ["customer_id", "customer_total_cost_alltime", "device_type_mode"]
level_cols = [c for c in level_cols if c in master.columns]
level_data = master.drop_duplicates(subset="customer_id")[level_cols].copy()
if "customer_total_cost_alltime" in level_data.columns:
    level_data["log_total_cost_alltime"] = np.log1p(level_data["customer_total_cost_alltime"])

bt_ids = born_treated["customer_id"].tolist()
matched_control_ids = set()
for _, bt in born_treated.iterrows():
    reg_bt = bt["registration_date"]
    candidates = never_treated_reg[
        (never_treated_reg["registration_date"] >= reg_bt - pd.Timedelta(days=REG_WINDOW_MAIN))
        & (never_treated_reg["registration_date"] <= reg_bt + pd.Timedelta(days=REG_WINDOW_MAIN))
    ]
    matched_control_ids.update(candidates["customer_id"].tolist())

bt_level = level_data[level_data["customer_id"].isin(bt_ids)]
ctrl_level = level_data[level_data["customer_id"].isin(matched_control_ids)]

balance_rows = []
if "log_total_cost_alltime" in level_data.columns:
    bt_cost = bt_level["log_total_cost_alltime"].dropna()
    ctrl_cost = ctrl_level["log_total_cost_alltime"].dropna()
    if len(bt_cost) >= 3 and len(ctrl_cost) >= 3:
        tstat, pval = sps.ttest_ind(bt_cost, ctrl_cost, equal_var=False)
        print(f"log_total_cost_alltime: BT mean={bt_cost.mean():.3f}(n={len(bt_cost)}), "
              f"control mean={ctrl_cost.mean():.3f}(n={len(ctrl_cost)}), t-test p={pval:.4f}"
              f"{'  NOTE: imbalanced' if pval < 0.05 else '  balanced'}")
        balance_rows.append({
            "variable": "log_total_cost_alltime", "type": "continuous",
            "born_treated_mean": bt_cost.mean(), "control_mean": ctrl_cost.mean(),
            "test": "welch_t", "pvalue": pval, "balanced": pval >= 0.05,
        })

if "device_type_mode" in level_data.columns:
    bt_dev = bt_level["device_type_mode"].dropna()
    ctrl_dev = ctrl_level["device_type_mode"].dropna()
    if len(bt_dev) > 0 and len(ctrl_dev) > 0:
        cats = sorted(set(bt_dev.unique()) | set(ctrl_dev.unique()))
        cont_table = pd.DataFrame({
            "born_treated": [((bt_dev == c).sum()) for c in cats],
            "control": [((ctrl_dev == c).sum()) for c in cats],
        }, index=cats)
        print(f"\ndevice_type_mode distribution:\n{cont_table}")
        if cont_table.shape[0] >= 2 and (cont_table.sum(axis=1) > 0).all():
            chi2, pval_chi, dof, _ = sps.chi2_contingency(cont_table.T)
            print(f"Chi-square test: chi2={chi2:.3f}, p={pval_chi:.4f}"
                  f"{'  NOTE: imbalanced' if pval_chi < 0.05 else '  balanced'}")
            balance_rows.append({
                "variable": "device_type_mode", "type": "categorical",
                "test": "chi_square", "pvalue": pval_chi, "balanced": pval_chi >= 0.05,
            })

balance_df = pd.DataFrame(balance_rows)
balance_path = STEP2_DIR / f"step3e_track2_covariate_balance_type{TARGET_TYPE}.csv"
balance_df.to_csv(balance_path, index=False)
print(f"\nCovariate balance results saved: {balance_path}")

# ============================================================================
# D. Track 2 regression adjustment
# ============================================================================
print("\n" + "=" * 70)
print("D. Track 2 regression adjustment (does the effect survive covariate control?)")
print("=" * 70)

reg_rows = []
for _, bt in born_treated.iterrows():
    reg_rows.append({"customer_id": bt["customer_id"], "registration_date": bt["registration_date"],
                      "is_born_treated": 1})
for _, ct in never_treated_reg[never_treated_reg["customer_id"].isin(matched_control_ids)].iterrows():
    reg_rows.append({"customer_id": ct["customer_id"], "registration_date": ct["registration_date"],
                      "is_born_treated": 0})
reg_df = pd.DataFrame(reg_rows)

for ycol in OUTCOME_VARS:
    reg_df[ycol] = reg_df.apply(
        lambda r: early_outcome(r["customer_id"], r["registration_date"], ycol), axis=1
    )

reg_df = reg_df.merge(level_data, on="customer_id", how="left")

regression_results = {}
for ycol in OUTCOME_VARS:
    print(f"\n--- outcome: {ycol} ---")
    formula_parts = [f"{ycol} ~ is_born_treated"]
    covariates_used = []
    if "log_total_cost_alltime" in reg_df.columns and reg_df["log_total_cost_alltime"].notna().sum() > 5:
        formula_parts[0] += " + log_total_cost_alltime"
        covariates_used.append("log_total_cost_alltime")
    if "device_type_mode" in reg_df.columns and reg_df["device_type_mode"].nunique() > 1:
        formula_parts[0] += " + C(device_type_mode)"
        covariates_used.append("device_type_mode")

    model_data = reg_df.dropna(subset=[ycol])
    if len(model_data) < 10:
        print("  insufficient sample -- regression skipped")
        continue

    m0 = smf.ols(f"{ycol} ~ is_born_treated", data=model_data).fit(cov_type="HC1")
    print(f"  [unadjusted] is_born_treated coef={m0.params['is_born_treated']:+.4f}, "
          f"p={m0.pvalues['is_born_treated']:.4f}")

    m1 = smf.ols(formula_parts[0], data=model_data).fit(cov_type="HC1")
    print(f"  [adjusted for {', '.join(covariates_used)}] is_born_treated coef="
          f"{m1.params['is_born_treated']:+.4f}, p={m1.pvalues['is_born_treated']:.4f}")

    coef_shrinkage = (
        (m0.params["is_born_treated"] - m1.params["is_born_treated"]) / m0.params["is_born_treated"]
        if m0.params["is_born_treated"] != 0 else np.nan
    )
    print(f"  coefficient shrinkage from covariate control: {coef_shrinkage:+.1%}"
          f"{'  NOTE: >30% shrinkage -- covariates explain a substantial share' if abs(coef_shrinkage) > 0.3 else ''}")

    regression_results[ycol] = {
        "unadjusted_coef": float(m0.params["is_born_treated"]),
        "unadjusted_pvalue": float(m0.pvalues["is_born_treated"]),
        "adjusted_coef": float(m1.params["is_born_treated"]),
        "adjusted_pvalue": float(m1.pvalues["is_born_treated"]),
        "covariates_used": covariates_used,
        "coef_shrinkage_pct": float(coef_shrinkage) if pd.notna(coef_shrinkage) else None,
        "still_significant_after_adjustment": bool(m1.pvalues["is_born_treated"] < 0.05),
    }

reg_out_path = STEP2_DIR / f"step3e_track2_regression_adjusted_type{TARGET_TYPE}.json"
with open(reg_out_path, "w", encoding="utf-8") as f:
    json.dump({
        "reg_match_window_days": REG_WINDOW_MAIN,
        "n_born_treated": len(born_treated),
        "n_matched_controls": len(matched_control_ids),
        "results": regression_results,
    }, f, ensure_ascii=False, indent=2, default=str)
print(f"\nRegression-adjusted results saved: {reg_out_path}")

print("\n" + "=" * 70)
print("Interpretation guide")
print("=" * 70)
print("Track 1: if the sign flips in >=2 of 7 LOO iterations, do not claim a directional effect --")
print("  report as individual case narratives (successes vs. failures) instead.")
print("Track 2: (B) significant at all windows + (C) covariates balanced + (D) still significant after")
print("  adjustment -> selection bias is unlikely to be the whole story (though this remains an")
print("  association, not a causal estimate). If (C) shows imbalance or (D) shows large shrinkage,")
print("  temper the claim to 'an association substantially explained by observed customer scale.'")
