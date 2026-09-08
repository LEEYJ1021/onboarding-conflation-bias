"""
================================================================================
Track 2 Regression Comparison: Outcome-Contamination-Free Covariate Check
================================================================================
Purpose:
    step3e's covariate adjustment used customer_total_cost_alltime (lifetime
    cumulative spend) to control for customer scale. But the Track 2 outcome
    is cumulative spend in the first EARLY_WINDOW days after registration —
    a subset of that same lifetime total. Using X to control for Y when X
    partly *contains* Y is a post-treatment / outcome-contamination bias:
    any shrinkage in the born-treated coefficient after "adjustment" could
    be mechanical rather than a genuine selection-bias correction.

    Fix: build a clean covariate, log_total_cost_excl_window, using only
    spend *outside* the outcome measurement window (lifetime total minus
    the EARLY_WINDOW-day window itself). This covariate cannot mechanically
    absorb outcome variance. Compare unadjusted vs. contaminated-covariate
    vs. clean-covariate regression coefficients side by side.

Input:  customer_day_panel.csv, customer_day_campaign_type_panel.csv,
        df_analysis_master.csv,
        step2_treatment_output/staggered_adoption_FINAL_type{T}.csv
        step2_treatment_output/step3c_born_treated_diagnosis_type{T}.csv
Output: step2_treatment_output/step3g_track2_clean_covariate_type{T}.csv
        step2_treatment_output/step3g_track2_regression_comparison_type{T}.json
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
REG_WINDOW_MAIN = int(os.environ.get("REG_MATCH_WINDOW_DAYS", "14"))

panel = pd.read_csv(ROOT / "customer_day_panel.csv")
ctp = pd.read_csv(ROOT / "customer_day_campaign_type_panel.csv")
master = pd.read_csv(ROOT / "df_analysis_master.csv")
final_cohort = pd.read_csv(STEP2_DIR / f"staggered_adoption_FINAL_type{TARGET_TYPE}.csv")
born_diag = pd.read_csv(STEP2_DIR / f"step3c_born_treated_diagnosis_type{TARGET_TYPE}.csv")


def _detect_date_column(df):
    for c in ("date", "stat_date", "stat_dt", "dt", "ad_date", "report_date", "log_date"):
        if c in df.columns:
            return c
    raise KeyError("Could not auto-detect a date column")


DATE_COL, DATE_COL_CTP = _detect_date_column(panel), _detect_date_column(ctp)
panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])
ctp[DATE_COL_CTP] = pd.to_datetime(ctp[DATE_COL_CTP])
master["date"] = pd.to_datetime(master["date"])
final_cohort["final_first_treated_date"] = pd.to_datetime(final_cohort["final_first_treated_date"])
born_diag["raw_panel_date_min"] = pd.to_datetime(born_diag["raw_panel_date_min"])
ctp["campaign_type"] = pd.to_numeric(ctp["campaign_type"], errors="coerce")

panel_ext = panel[["customer_id", DATE_COL, "cost"]].rename(columns={DATE_COL: "date"})
panel_ext["log_spend_safe"] = np.log1p(panel_ext["cost"])
ctp_active = ctp[ctp["cost"] > 0]
n_types = (ctp_active.groupby(["customer_id", DATE_COL_CTP])["campaign_type"].nunique()
           .rename("n_campaign_types_active").reset_index().rename(columns={DATE_COL_CTP: "date"}))
panel_ext = panel_ext.merge(n_types, on=["customer_id", "date"], how="left")
panel_ext["n_campaign_types_active"] = panel_ext["n_campaign_types_active"].fillna(0).astype(int)

master_o = master[["customer_id", "date", "log_spend_safe", "n_campaign_types_active"]].copy()
master_o["_s"] = 0
panel_o = panel_ext[["customer_id", "date", "log_spend_safe", "n_campaign_types_active"]].copy()
panel_o["_s"] = 1
combined = pd.concat([master_o, panel_o]).sort_values("_s").drop_duplicates(
    subset=["customer_id", "date"], keep="first").drop(columns="_s")

never_treated_ids = final_cohort.loc[final_cohort["final_cohort"] == "never_treated", "customer_id"].unique().tolist()
never_treated_reg = panel[panel["customer_id"].isin(never_treated_ids)].groupby("customer_id")[DATE_COL].min() \
    .rename("registration_date").reset_index()
born_treated = born_diag[born_diag["gap_days"] <= GAP_THRESHOLD][["customer_id", "raw_panel_date_min"]] \
    .rename(columns={"raw_panel_date_min": "registration_date"})


def early_outcome(cid, reg_date, ycol):
    sub = combined[(combined["customer_id"] == cid) & (combined["date"] >= reg_date)
                   & (combined["date"] < reg_date + pd.Timedelta(days=EARLY_WINDOW))]
    if sub.empty:
        return np.nan
    return float(sub[ycol].sum()) if ycol == "log_spend_safe" else float(sub[ycol].mean())


matched_control_ids = set()
for _, bt in born_treated.iterrows():
    reg_bt = bt["registration_date"]
    cand = never_treated_reg[
        (never_treated_reg["registration_date"] >= reg_bt - pd.Timedelta(days=REG_WINDOW_MAIN)) &
        (never_treated_reg["registration_date"] <= reg_bt + pd.Timedelta(days=REG_WINDOW_MAIN))]
    matched_control_ids.update(cand["customer_id"].tolist())

rows = [{"customer_id": r.customer_id, "registration_date": r.registration_date, "is_born_treated": 1}
        for r in born_treated.itertuples()]
rows += [{"customer_id": r.customer_id, "registration_date": r.registration_date, "is_born_treated": 0}
         for r in never_treated_reg[never_treated_reg["customer_id"].isin(matched_control_ids)].itertuples()]
reg_df = pd.DataFrame(rows)
print(f"born-treated: {int((reg_df['is_born_treated']==1).sum())}, matched controls: {int((reg_df['is_born_treated']==0).sum())}")

OUTCOME_VARS = ["log_spend_safe", "n_campaign_types_active"]
for ycol in OUTCOME_VARS:
    reg_df[ycol] = reg_df.apply(lambda r: early_outcome(r.customer_id, r.registration_date, ycol), axis=1)

# --- clean covariate: total spend excluding the outcome window ---
total_cost_full = panel.groupby("customer_id")["cost"].sum().rename("total_cost_full_history")
reg_df = reg_df.merge(total_cost_full, on="customer_id", how="left").fillna({"total_cost_full_history": 0.0})


def window_cost(cid, reg_date):
    sub = panel[(panel["customer_id"] == cid) & (panel[DATE_COL] >= reg_date)
                & (panel[DATE_COL] < reg_date + pd.Timedelta(days=EARLY_WINDOW))]
    return float(sub["cost"].sum())


reg_df["window_cost_raw"] = reg_df.apply(lambda r: window_cost(r.customer_id, r.registration_date), axis=1)
reg_df["total_cost_excl_window"] = (reg_df["total_cost_full_history"] - reg_df["window_cost_raw"]).clip(lower=0.0)
reg_df["log_total_cost_excl_window"] = np.log1p(reg_df["total_cost_excl_window"])

level_cols = [c for c in ["customer_id", "customer_total_cost_alltime", "device_type_mode"] if c in master.columns]
level_data = master.drop_duplicates("customer_id")[level_cols].copy()
if "customer_total_cost_alltime" in level_data.columns:
    level_data["log_total_cost_alltime_ORIG_CONTAMINATED"] = np.log1p(level_data["customer_total_cost_alltime"])
reg_df = reg_df.merge(level_data, on="customer_id", how="left")

detail_path = STEP2_DIR / f"step3g_track2_clean_covariate_type{TARGET_TYPE}.csv"
reg_df.to_csv(detail_path, index=False)
print(f"Detail saved: {detail_path}")

comparison_results = {}
for ycol in OUTCOME_VARS:
    print(f"\n--- outcome: {ycol} ---")
    model_data = reg_df.dropna(subset=[ycol]).copy()
    if len(model_data) < 10:
        continue

    m0 = smf.ols(f"{ycol} ~ is_born_treated", data=model_data).fit(cov_type="HC1")
    coef0, p0 = m0.params["is_born_treated"], m0.pvalues["is_born_treated"]
    print(f"  [unadjusted]              coef={coef0:+.4f}, p={p0:.4f}")

    def _fit(covar_col, label):
        covars, formula = [], f"{ycol} ~ is_born_treated"
        if covar_col in model_data.columns and model_data[covar_col].notna().sum() > 5:
            formula += f" + {covar_col}"; covars.append(covar_col)
        if "device_type_mode" in model_data.columns and model_data["device_type_mode"].nunique() > 1:
            formula += " + C(device_type_mode)"; covars.append("device_type_mode")
        m = smf.ols(formula, data=model_data).fit(cov_type="HC1")
        coef, p = m.params["is_born_treated"], m.pvalues["is_born_treated"]
        shrink = (coef0 - coef) / coef0 if coef0 != 0 else np.nan
        print(f"  [{label}] coef={coef:+.4f}, p={p:.4f}, shrinkage={shrink:+.1%} (covariates: {covars})")
        return coef, p, shrink, covars

    coef1, p1, shrink1, covars1 = _fit("log_total_cost_alltime_ORIG_CONTAMINATED", "contaminated covariate")
    coef2, p2, shrink2, covars2 = _fit("log_total_cost_excl_window", "clean covariate       ")

    comparison_results[ycol] = {
        "unadjusted": {"coef": float(coef0), "pvalue": float(p0)},
        "orig_contaminated_covariate": {"coef": float(coef1), "pvalue": float(p1),
                                          "shrinkage_pct": float(shrink1), "covariates": covars1},
        "clean_covariate": {"coef": float(coef2), "pvalue": float(p2),
                             "shrinkage_pct": float(shrink2), "covariates": covars2},
    }

out_path = STEP2_DIR / f"step3g_track2_regression_comparison_type{TARGET_TYPE}.json"
with open(out_path, "w") as f:
    json.dump({
        "target_type": TARGET_TYPE, "early_outcome_window_days": EARLY_WINDOW,
        "reg_match_window_days": REG_WINDOW_MAIN,
        "n_born_treated": int((reg_df["is_born_treated"] == 1).sum()),
        "n_matched_controls": int((reg_df["is_born_treated"] == 0).sum()),
        "results": comparison_results,
        "note": ("log_total_cost_excl_window measures spend outside the outcome window, "
                 "so it is structurally free of outcome contamination. Comparable shrinkage "
                 "between the contaminated and clean covariates indicates the original "
                 "adjustment result was not a circular-logic artifact."),
    }, f, indent=2, default=str)
print(f"\nSaved: {out_path}")
