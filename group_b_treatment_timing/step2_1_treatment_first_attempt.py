"""
================================================================================
Step 2.1 — First Attempt at Staggered-Adoption Treatment Definition (Type 6)
================================================================================
Defines "first adoption" of the Type-6 campaign as the first observed day
with cost_type6 > 0, and classifies customers whose adoption predates the
start of their (30-day-minimum) stable observation window as left-censored.
This first attempt doubles as a diagnostic: we expect the stable-window
condition (long-run observability) to disproportionately retain customers
whose portfolios are already settled, leaving genuinely new adoptions rare.
This script checks that expectation against the data and, if confirmed,
motivates Step 2.2 (left-censoring recovery).

Input:  df_analysis_master.csv (post Section-1 PASS, N customers / rows)
Output: staggered_adoption_customer_level_type{T}.csv
        event_time_panel_type{T}_v1.csv
        event_study_type{T}_v1.png
        staggered_adoption_type{T}_v1_summary.json
================================================================================
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
OUT_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = ROOT / "df_analysis_master.csv"
if not MASTER_PATH.exists():
    raise FileNotFoundError(f"{MASTER_PATH} missing — run step1_data_integrity_build.py first.")

df = pd.read_csv(MASTER_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["customer_id", "date"]).reset_index(drop=True)

N_CUSTOMERS = df["customer_id"].nunique()
print(f"Loaded: {len(df):,} rows / {N_CUSTOMERS} customers\n")

TARGET_TYPE = int(os.environ.get("TARGET_TYPE", "6"))
cost_col = f"cost_type{TARGET_TYPE}"
if cost_col not in df.columns:
    raise ValueError(f"Column {cost_col} not found — check df_analysis_master.csv.")

# ------------------------------------------------------------------
# 1. Per-customer observation window + first active date for target type
# ------------------------------------------------------------------
print("=" * 70)
print(f"1. First-attempt staggered adoption timing for type {TARGET_TYPE}")
print("=" * 70)

window_start = df.groupby("customer_id")["date"].min().rename("customer_window_start")
window_end = df.groupby("customer_id")["date"].max().rename("customer_window_end")

active = df[df[cost_col] > 0]
first_active = active.groupby("customer_id")["date"].min().rename("first_treated_date")

n_type_users = active["customer_id"].nunique()
print(f"Customers with any type-{TARGET_TYPE} activity: {n_type_users} (of {N_CUSTOMERS} total)")

adoption_rows = []
for cid in sorted(df["customer_id"].unique()):
    w_start = window_start.loc[cid]
    w_end = window_end.loc[cid]
    if cid in first_active.index:
        f_date = first_active.loc[cid]
        if f_date <= w_start:
            cohort_type = "already_active_at_window_start"  # possible left-censoring
        else:
            cohort_type = "adopted_within_window"
    else:
        f_date = pd.NaT
        cohort_type = "never_treated_within_window"

    adoption_rows.append({
        "customer_id": cid,
        "campaign_type": TARGET_TYPE,
        "customer_window_start": w_start,
        "customer_window_end": w_end,
        "first_treated_date": f_date,
        "cohort_type": cohort_type,
        "cohort_period": (f_date.strftime("%Y-%m-%d") if pd.notna(f_date) else "never_treated"),
    })

adoption_df = pd.DataFrame(adoption_rows)
adoption_path = OUT_DIR / f"staggered_adoption_customer_level_type{TARGET_TYPE}.csv"
adoption_df.to_csv(adoption_path, index=False)

print(f"\n[type {TARGET_TYPE}] Cohort distribution:")
cohort_counts = adoption_df["cohort_type"].value_counts()
print(cohort_counts)

n_adopted = int(cohort_counts.get("adopted_within_window", 0))
n_censored = int(cohort_counts.get("already_active_at_window_start", 0))
n_never = int(cohort_counts.get("never_treated_within_window", 0))

if n_censored:
    print(f"\n  NOTE: {n_censored} customers were already active before the observation window "
          f"started (left-censored) — excluded from the 'pure new adoption' sample and are the "
          f"target of Step 2.2 (left-censoring recovery).")

print(f"\nCustomer-level adoption timing saved: {adoption_path}")

# ------------------------------------------------------------------
# 2. First-attempt event-time panel (pure new adoptions only) + pre-check plot
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. First-attempt event-time panel and pre-check event-study plot")
print("=" * 70)

EVENT_WINDOW = int(os.environ.get("EVENT_WINDOW_DAYS", "30"))
OUTCOME_VARS = [c for c in ["log_spend_safe", "n_campaign_types_active"] if c in df.columns]

coh = adoption_df[adoption_df["cohort_type"] == "adopted_within_window"][
    ["customer_id", "first_treated_date"]
]

event_summary = None
if coh.empty:
    print(f"[type {TARGET_TYPE}] No 'pure new adopter' customers — cannot build an event-time panel.")
else:
    panel_t = df.merge(coh, on="customer_id", how="inner")
    panel_t["event_time"] = (panel_t["date"] - panel_t["first_treated_date"]).dt.days
    panel_t = panel_t[panel_t["event_time"].between(-EVENT_WINDOW, EVENT_WINDOW)]

    ev_path = OUT_DIR / f"event_time_panel_type{TARGET_TYPE}_v1.csv"
    keep_cols = ["customer_id", "date", "first_treated_date", "event_time"] + OUTCOME_VARS
    share_col = f"cost_share_type{TARGET_TYPE}"
    if share_col in df.columns:
        keep_cols = keep_cols + [share_col]
    panel_t[keep_cols].to_csv(ev_path, index=False)
    print(f"[type {TARGET_TYPE}] Event-time panel saved: {ev_path} "
          f"({coh['customer_id'].nunique()} adopters, +/-{EVENT_WINDOW}-day window, {len(panel_t):,} rows)")

    n_series = len(OUTCOME_VARS)
    fig, axes = plt.subplots(1, n_series, figsize=(6 * n_series, 4), squeeze=False)
    for j, ycol in enumerate(OUTCOME_VARS):
        ev_mean = panel_t.groupby("event_time")[ycol].mean()
        ax = axes[0][j]
        ax.plot(ev_mean.index, ev_mean.values, marker="o", markersize=3, color="steelblue")
        ax.axvline(0, color="firebrick", linestyle="--", linewidth=1, label="Adoption date")
        ax.set_title(f"Type {TARGET_TYPE} (v1, n={coh['customer_id'].nunique()}): {ycol} around adoption")
        ax.set_xlabel("Event time (days from first adoption)")
        ax.set_ylabel(ycol)
        ax.legend()
    plt.tight_layout()
    fig_path = OUT_DIR / f"event_study_type{TARGET_TYPE}_v1.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Event-study plot saved: {fig_path}")

    event_summary = {
        "n_adopters": int(coh["customer_id"].nunique()),
        "n_rows": int(len(panel_t)),
    }

# ------------------------------------------------------------------
# 3. First-attempt verdict — is the sample large enough for cohort comparison?
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. First-attempt verdict")
print("=" * 70)

MIN_VIABLE_ADOPTERS = int(os.environ.get("MIN_VIABLE_ADOPTERS", "15"))
viable = n_adopted >= MIN_VIABLE_ADOPTERS

if viable:
    verdict = "VIABLE_PROCEED_TO_STEP_C"
    verdict_msg = (
        f"{n_adopted} pure new adopters meets the minimum threshold "
        f"({MIN_VIABLE_ADOPTERS}) for a cohort comparison. Step 2.2 (left-censoring "
        f"recovery) can be skipped; proceed directly to the Step-3 main estimation."
    )
else:
    verdict = "NOT_VIABLE_NEED_RECOVERY"
    verdict_msg = (
        f"{n_adopted} pure new adopters is insufficient for a cohort comparison "
        f"(minimum threshold {MIN_VIABLE_ADOPTERS}). Proceed to Step 2.2 (left-censoring "
        f"recovery) to check whether some of the {n_censored} left-censored customers "
        f"were dropped merely as an artifact of the stable-window definition."
    )

print(f"Verdict: {verdict}")
print(verdict_msg)

# ------------------------------------------------------------------
# 4. Save summary
# ------------------------------------------------------------------
summary = {
    "target_type": TARGET_TYPE,
    "n_customers_total": int(N_CUSTOMERS),
    "n_type_users_ever": int(n_type_users),
    "cohort_counts": {
        "adopted_within_window": n_adopted,
        "already_active_at_window_start": n_censored,
        "never_treated_within_window": n_never,
    },
    "event_window_days": EVENT_WINDOW,
    "event_time_panel_v1_summary": event_summary,
    "min_viable_adopters_threshold": MIN_VIABLE_ADOPTERS,
    "verdict": verdict,
    "verdict_message": verdict_msg,
}
summary_path = OUT_DIR / f"staggered_adoption_type{TARGET_TYPE}_v1_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print(f"\nSummary saved: {summary_path}")
print(f"\nAll Section-2.1 outputs saved to {OUT_DIR}.")
if not viable:
    print("Next step: step2_2_left_censoring_recovery.py "
          "(left-censoring recovery diagnosis + safety validation + final cohort construction)")
else:
    print("Next step: proceed directly to step3_callaway_santanna.py")
