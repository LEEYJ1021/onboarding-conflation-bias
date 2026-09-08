"""
================================================================================
Step 3b — Diagnosing Why Adopters Drop Out of the Valid-Stack Set
================================================================================
Background: step3_stacked_did_estimation.py found only 9 valid stacks out of
32 primary_adopter customers, and PRIMARY/ROBUST produced identical ATTs —
suggesting the 3 robustness_only_adopter customers never contributed at all.

Each adopter's compute_att() stack is valid only if all four conditions hold:
    1) treated pre_t (event_time<0) is non-empty
    2) treated post_t (event_time>=0) is non-empty
    3) never-treated controls have pre-period observations in the same
       calendar window (g +/- 30 days)
    4) controls have post-period observations in the same calendar window

This script checks all four conditions individually for every
primary_adopter (32) + robustness_only_adopter (3) customer and records
exactly which condition caused the drop. If (3)/(4) are the cause, that
would mean "the control group's observation period does not overlap the
adopter's adoption timing," which could require redesigning the Step-3
control-group definition itself.

Input:  customer_day_panel.csv, customer_day_campaign_type_panel.csv,
        df_analysis_master.csv, staggered_adoption_FINAL_type{T}.csv
Output: step3_stack_dropout_diagnosis_type{T}.csv
        step3_stack_dropout_summary_type{T}.json
================================================================================
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))

TARGET_TYPE = int(os.environ.get("TARGET_TYPE", "6"))
EVENT_WINDOW = int(os.environ.get("EVENT_WINDOW_DAYS", "30"))

PANEL_PATH = ROOT / "customer_day_panel.csv"
CTP_PATH = ROOT / "customer_day_campaign_type_panel.csv"
MASTER_PATH = ROOT / "df_analysis_master.csv"
FINAL_COHORT_PATH = STEP2_DIR / f"staggered_adoption_FINAL_type{TARGET_TYPE}.csv"

for p in [PANEL_PATH, CTP_PATH, MASTER_PATH, FINAL_COHORT_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p}")

panel = pd.read_csv(PANEL_PATH)
ctp = pd.read_csv(CTP_PATH)
master = pd.read_csv(MASTER_PATH)
final_cohort = pd.read_csv(FINAL_COHORT_PATH)


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

# ------------------------------------------------------------------
# 0. Rebuild the extended outcome panel (matches Step 3's definition)
# ------------------------------------------------------------------
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
    combined_outcomes
    .sort_values("_source", key=lambda s: s.map({"stable_window(master)": 0, "raw_panel(pre_or_post_window)": 1}))
    .drop_duplicates(subset=["customer_id", "date"], keep="first")
    .drop(columns="_source")
)

never_treated_ids = final_cohort.loc[
    final_cohort["final_cohort"] == "never_treated", "customer_id"
].unique().tolist()
control_panel = combined_outcomes[combined_outcomes["customer_id"].isin(never_treated_ids)].copy()

control_calendar_min = control_panel["date"].min()
control_calendar_max = control_panel["date"].max()
print(f"Control group (never-treated, n={len(never_treated_ids)}) calendar coverage: "
      f"{control_calendar_min.date()} to {control_calendar_max.date()}\n")

adopters_all = final_cohort.loc[
    final_cohort["final_cohort"].isin(["primary_adopter", "robustness_only_adopter"]),
    ["customer_id", "final_first_treated_date", "final_cohort"]
].drop_duplicates()

OUTCOME_VARS = ["log_spend_safe", "n_campaign_types_active"]

# ------------------------------------------------------------------
# 1. Check the four conditions individually for each adopter
# ------------------------------------------------------------------
print("=" * 70)
print("1. Checking valid-stack conditions per adopter")
print("=" * 70)

diag_rows = []
for _, row in adopters_all.iterrows():
    cid = row["customer_id"]
    g = row["final_first_treated_date"]
    cohort_label = row["final_cohort"]

    treated_series = combined_outcomes[combined_outcomes["customer_id"] == cid][["date"]].copy()
    treated_series["event_time"] = (treated_series["date"] - g).dt.days
    treated_series = treated_series[treated_series["event_time"].between(-EVENT_WINDOW, EVENT_WINDOW)]

    n_pre_treated = int((treated_series["event_time"] < 0).sum())
    n_post_treated = int((treated_series["event_time"] >= 0).sum())

    window_start = g - pd.Timedelta(days=EVENT_WINDOW)
    window_end = g + pd.Timedelta(days=EVENT_WINDOW)
    window_dates = pd.date_range(window_start, window_end)

    calendar_overlap = not (window_end < control_calendar_min or window_start > control_calendar_max)

    ctrl = control_panel[control_panel["date"].isin(window_dates)]
    ctrl = ctrl.copy()
    ctrl["event_time"] = (ctrl["date"] - g).dt.days
    n_pre_ctrl_custdays = int((ctrl["event_time"] < 0).shape[0])
    n_post_ctrl_custdays = int((ctrl["event_time"] >= 0).shape[0])
    n_ctrl_customers_pre = int(ctrl.loc[ctrl["event_time"] < 0, "customer_id"].nunique())
    n_ctrl_customers_post = int(ctrl.loc[ctrl["event_time"] >= 0, "customer_id"].nunique())

    cond1 = n_pre_treated > 0
    cond2 = n_post_treated > 0
    cond3 = n_pre_ctrl_custdays > 0
    cond4 = n_post_ctrl_custdays > 0
    is_valid = cond1 and cond2 and cond3 and cond4

    if is_valid:
        drop_reason = "VALID"
    elif not calendar_overlap:
        drop_reason = "no_calendar_overlap_with_control_coverage"
    elif not cond1:
        drop_reason = "treated_pre_period_empty"
    elif not cond2:
        drop_reason = "treated_post_period_empty"
    elif not cond3:
        drop_reason = "control_pre_period_empty_in_window"
    elif not cond4:
        drop_reason = "control_post_period_empty_in_window"
    else:
        drop_reason = "unknown"

    diag_rows.append({
        "customer_id": cid, "final_cohort": cohort_label,
        "first_treated_date": g,
        "n_pre_treated_days": n_pre_treated, "n_post_treated_days": n_post_treated,
        "calendar_overlap_with_control": calendar_overlap,
        "n_ctrl_customer_days_pre": n_pre_ctrl_custdays, "n_ctrl_customer_days_post": n_post_ctrl_custdays,
        "n_ctrl_customers_pre": n_ctrl_customers_pre, "n_ctrl_customers_post": n_ctrl_customers_post,
        "is_valid_stack": is_valid, "drop_reason": drop_reason,
    })

diag_df = pd.DataFrame(diag_rows)
diag_path = STEP2_DIR / f"step3_stack_dropout_diagnosis_type{TARGET_TYPE}.csv"
diag_df.to_csv(diag_path, index=False)

print(f"\n{diag_df['is_valid_stack'].sum()} valid / {(~diag_df['is_valid_stack']).sum()} dropped, "
      f"out of {len(diag_df)} total")
print("\nDrop-reason distribution:")
print(diag_df.loc[~diag_df["is_valid_stack"], "drop_reason"].value_counts())

print("\nValid/dropped by cohort:")
print(diag_df.groupby(["final_cohort", "is_valid_stack"]).size())

robust_only = diag_df[diag_df["final_cohort"] == "robustness_only_adopter"]
n_robust_only_valid = int(robust_only["is_valid_stack"].sum())
print(f"\nNOTE: robustness_only_adopter (n={len(robust_only)}) valid stacks: {n_robust_only_valid}")
if n_robust_only_valid == 0:
    print("  CONFIRMED: the 3 CAUTION customers never contributed to the Step-3 PRIMARY/ROBUST comparison.")
    print("    The earlier PRIMARY=ROBUST agreement was therefore not a genuine robustness")
    print("    check but a recomputation on the identical effective sample.")

# ------------------------------------------------------------------
# 2. Detail on no-calendar-overlap cases (actual date gaps)
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. Detail on no_calendar_overlap cases")
print("=" * 70)
no_overlap = diag_df[diag_df["drop_reason"] == "no_calendar_overlap_with_control_coverage"]
if not no_overlap.empty:
    no_overlap = no_overlap.copy()
    no_overlap["window_start"] = no_overlap["first_treated_date"] - pd.Timedelta(days=EVENT_WINDOW)
    no_overlap["window_end"] = no_overlap["first_treated_date"] + pd.Timedelta(days=EVENT_WINDOW)
    print(no_overlap[["customer_id", "final_cohort", "first_treated_date",
                       "window_start", "window_end"]].to_string(index=False))
    print(f"\n(reference) control-group calendar coverage: {control_calendar_min.date()} to {control_calendar_max.date()}")
else:
    print("None — no drops due to calendar non-overlap.")

# ------------------------------------------------------------------
# 3. Valid but thin cases — condition met but with very few observations
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. Valid but extremely thin observation cases (conditions met but noise risk)")
print("=" * 70)
THIN_DAYS_THRESHOLD = int(os.environ.get("THIN_DAYS_THRESHOLD", "3"))
valid_df = diag_df[diag_df["is_valid_stack"]].copy()
thin_valid = valid_df[
    (valid_df["n_pre_treated_days"] < THIN_DAYS_THRESHOLD) |
    (valid_df["n_post_treated_days"] < THIN_DAYS_THRESHOLD) |
    (valid_df["n_ctrl_customers_pre"] < THIN_DAYS_THRESHOLD) |
    (valid_df["n_ctrl_customers_post"] < THIN_DAYS_THRESHOLD)
]
if not thin_valid.empty:
    print(f"NOTE: {len(thin_valid)} of {len(valid_df)} valid stacks have "
          f"pre/post days or control customer counts below {THIN_DAYS_THRESHOLD}:")
    print(thin_valid[["customer_id", "n_pre_treated_days", "n_post_treated_days",
                       "n_ctrl_customers_pre", "n_ctrl_customers_post"]].to_string(index=False))
else:
    print(f"None — all valid stacks have >= {THIN_DAYS_THRESHOLD} days/customers")

# ------------------------------------------------------------------
# Save summary
# ------------------------------------------------------------------
summary = {
    "target_type": TARGET_TYPE,
    "event_window_days": EVENT_WINDOW,
    "n_adopters_checked": int(len(diag_df)),
    "n_valid_stacks": int(diag_df["is_valid_stack"].sum()),
    "n_dropped": int((~diag_df["is_valid_stack"]).sum()),
    "drop_reason_counts": diag_df.loc[~diag_df["is_valid_stack"], "drop_reason"].value_counts().to_dict(),
    "n_robustness_only_valid": n_robust_only_valid,
    "n_robustness_only_total": int(len(robust_only)),
    "control_calendar_coverage": {
        "min": str(control_calendar_min.date()), "max": str(control_calendar_max.date()),
    },
    "n_thin_valid_stacks": int(len(thin_valid)),
}
summary_path = STEP2_DIR / f"step3_stack_dropout_summary_type{TARGET_TYPE}.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print(f"\nDiagnostic table saved: {diag_path}")
print(f"Summary saved: {summary_path}")
print("\nAfter reviewing this, proceed with one of the following:")
print("  - If most drops are 'no_calendar_overlap': redefine the control group as customers")
print("    whose calendar overlaps each adopter, or shrink EVENT_WINDOW, and re-estimate.")
print("  - If most drops are 'control_*_empty_in_window': the control-group observation")
print("    density is low at particular times; consider stratifying by adoption era.")
print("  - If many valid stacks are thin: the Step-3 summary ATT is driven by a handful of")
print("    observations; scrutinize individual cases before trusting bootstrap CIs.")
