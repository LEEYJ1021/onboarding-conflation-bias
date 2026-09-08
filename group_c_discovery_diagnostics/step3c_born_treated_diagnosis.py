"""
================================================================================
Step 3c — "Born-Treated" Diagnosis: Panel-Wide Structural Left-Censoring
================================================================================
Hypothesis:
    Step 3b showed that 26 of 35 adopters (74%) drop out of the valid-stack
    set because their treated pre-period is empty. This script tests whether
    that emptiness reflects a deeper structural fact than the stable-window
    left-censoring already handled in Step 2.2: namely, that for many
    adopters the customer's very first row in the *raw* panel already shows
    the target campaign type active. If so, no amount of pre-window recovery
    can produce a pre-treatment observation for these customers, because
    pre-treatment history simply does not exist — they were "born treated."

    Step 2.2's coverage check (raw_panel_date_min < customer_window_start)
    only verified that data exists before the *stable window*; it never
    checked whether data exists before the *adoption date itself*. That
    distinction is the suspected source of this problem.

Test:
    For each adopter (primary_adopter + robustness_only_adopter):
        raw_panel_date_min = customer's first date in the raw panel
        gap_days = (true_first_active_date - raw_panel_date_min).days
        is_born_treated = gap_days <= BORN_TREATED_GAP_THRESHOLD (default 1)
            -> the panel-entry date IS effectively the adoption date, so no
               pre-period can exist by construction.
        gap_days > threshold but pre_period still empty in the event-time
        window -> a separate cause (e.g., window mismatch) worth inspecting
        individually.

Input:  customer_day_panel.csv,
        staggered_adoption_FINAL_type{T}.csv (Step 2.2 output)
        step3_stack_dropout_diagnosis_type{T}.csv (Step 3b output)
Output: step3c_born_treated_diagnosis_type{T}.csv
        step3c_born_treated_summary_type{T}.json
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
PANEL_PATH = ROOT / "customer_day_panel.csv"
FINAL_COHORT_PATH = STEP2_DIR / f"staggered_adoption_FINAL_type{TARGET_TYPE}.csv"
DROPOUT_DIAG_PATH = STEP2_DIR / f"step3_stack_dropout_diagnosis_type{TARGET_TYPE}.csv"

for p in [PANEL_PATH, FINAL_COHORT_PATH, DROPOUT_DIAG_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p}")

panel = pd.read_csv(PANEL_PATH)
final_cohort = pd.read_csv(FINAL_COHORT_PATH)
dropout_diag = pd.read_csv(DROPOUT_DIAG_PATH)


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
panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])
final_cohort["final_first_treated_date"] = pd.to_datetime(final_cohort["final_first_treated_date"])
dropout_diag["first_treated_date"] = pd.to_datetime(dropout_diag["first_treated_date"])

adopters_all = final_cohort.loc[
    final_cohort["final_cohort"].isin(["primary_adopter", "robustness_only_adopter"]),
    ["customer_id", "final_first_treated_date", "final_cohort"]
].drop_duplicates()

raw_date_min = panel.groupby("customer_id")[DATE_COL].min().rename("raw_panel_date_min")
raw_date_max = panel.groupby("customer_id")[DATE_COL].max().rename("raw_panel_date_max")
n_panel_rows = panel.groupby("customer_id").size().rename("n_raw_panel_rows")

# ------------------------------------------------------------------
# 1. Compute gap_days — the true gap between panel-entry date and adoption date
# ------------------------------------------------------------------
print("=" * 70)
print("1. Per-adopter gap between panel-entry date and adoption date")
print("=" * 70)

diag = adopters_all.merge(raw_date_min, on="customer_id", how="left")
diag = diag.merge(raw_date_max, on="customer_id", how="left")
diag = diag.merge(n_panel_rows, on="customer_id", how="left")
diag["gap_days"] = (diag["final_first_treated_date"] - diag["raw_panel_date_min"]).dt.days

BORN_TREATED_THRESHOLD = int(os.environ.get("BORN_TREATED_GAP_THRESHOLD", "1"))
diag["is_born_treated"] = diag["gap_days"] <= BORN_TREATED_THRESHOLD

diag = diag.merge(
    dropout_diag[["customer_id", "n_pre_treated_days", "is_valid_stack", "drop_reason"]],
    on="customer_id", how="left"
)

print(diag[["customer_id", "final_cohort", "raw_panel_date_min", "final_first_treated_date",
            "gap_days", "is_born_treated", "n_pre_treated_days", "drop_reason"]]
      .sort_values("gap_days").to_string(index=False))

n_born_treated = int(diag["is_born_treated"].sum())
n_has_real_gap = int((~diag["is_born_treated"]).sum())
print(f"\nOf {len(diag)} adopters:")
print(f"  born-treated (gap<={BORN_TREATED_THRESHOLD} day, panel entry = adoption date): {n_born_treated}")
print(f"  genuine gap exists (panel history predates adoption): {n_has_real_gap}")

# ------------------------------------------------------------------
# 2. Does treated_pre_period_empty (Step 3b) coincide with born-treated?
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. Hypothesis test: does treated_pre_period_empty == born-treated?")
print("=" * 70)

empty_pre = diag[diag["drop_reason"] == "treated_pre_period_empty"]
overlap = int((empty_pre["is_born_treated"]).sum())
print(f"Customers with drop_reason='treated_pre_period_empty': {len(empty_pre)}")
print(f"  of which confirmed born-treated (gap<={BORN_TREATED_THRESHOLD}): {overlap}")
if len(empty_pre) > 0:
    match_rate = overlap / len(empty_pre)
    print(f"  match rate: {match_rate:.1%}")
    if match_rate >= 0.9:
        print("  CONFIRMED: the empty pre-period is overwhelmingly explained by born-treated status "
              "(a structural, panel-wide left-censoring phenomenon), unrelated to Step 2.2's "
              "pre-window recovery procedure.")
    else:
        print(f"  {len(empty_pre) - overlap} cases do not fit the hypothesis and warrant individual "
              f"review (gap_days > threshold, yet pre-period still empty within the +/-30-day window).")

mismatch = empty_pre[~empty_pre["is_born_treated"]]
if not mismatch.empty:
    print("\n[Mismatch detail] gap_days>threshold but pre-period still empty:")
    print(mismatch[["customer_id", "gap_days", "raw_panel_date_min",
                     "final_first_treated_date", "n_pre_treated_days"]].to_string(index=False))

# ------------------------------------------------------------------
# 3. Distribution of gap_days
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. Distribution of gap_days")
print("=" * 70)
print(diag["gap_days"].describe().round(1))
bins = [-1, 0, 1, 7, 30, 9999]
labels = ["0 days (same day)", "1 day", "2-7 days", "8-30 days", "31+ days"]
diag["gap_bucket"] = pd.cut(diag["gap_days"], bins=bins, labels=labels)
print(f"\nCustomer count by gap_days bucket:")
print(diag["gap_bucket"].value_counts().sort_index())

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
out_path = STEP2_DIR / f"step3c_born_treated_diagnosis_type{TARGET_TYPE}.csv"
diag.to_csv(out_path, index=False)

summary = {
    "target_type": TARGET_TYPE,
    "born_treated_gap_threshold_days": BORN_TREATED_THRESHOLD,
    "n_adopters_total": int(len(diag)),
    "n_born_treated": n_born_treated,
    "n_has_real_gap": n_has_real_gap,
    "n_treated_pre_period_empty": int(len(empty_pre)),
    "n_treated_pre_period_empty_matched_born_treated": overlap,
    "gap_days_stats": {
        "mean": float(diag["gap_days"].mean()), "median": float(diag["gap_days"].median()),
        "min": float(diag["gap_days"].min()), "max": float(diag["gap_days"].max()),
    },
    "recommendation": (
        "Born-treated customers have no within-customer pre/post comparison by construction, so "
        "they carry zero identifying power for event-study DiD (Step 3). They should be split off "
        "explicitly as 'unidentifiable adopters,' and Step 3 should be re-estimated on the "
        "'identifiable adopters' subset (customers with a genuine pre-period) only. The born-treated "
        "group should instead be analyzed separately -- e.g., via a cross-sectional matched "
        "comparison against demographically similar never-treated customers -- and the resulting "
        "sample shrinkage should be reported candidly as a limitation of the study."
    ),
}
summary_path = STEP2_DIR / f"step3c_born_treated_summary_type{TARGET_TYPE}.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print(f"\nDiagnostic table saved: {out_path}")
print(f"Summary saved: {summary_path}")
print("\nNext step: step3d_two_track_analysis.py to split the sample into")
print("  Track 1 (true switchers, within-customer DiD) and")
print("  Track 2 (born-treated, cross-sectional matched comparison)")
