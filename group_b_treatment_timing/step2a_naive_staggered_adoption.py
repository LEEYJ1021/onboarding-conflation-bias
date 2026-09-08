"""
================================================================================
Step 2a — Naive first attempt at staggered-adoption cohort construction
================================================================================
This is the "naive approach that anyone would try first," included
deliberately as the discovery narrative's opening move. It defines
"first adoption" of TARGET_TYPE as the first day with cost_type{T} > 0,
and classifies customers into:
    - adopted_within_window          (first-adoption date is after the
                                       customer's stable-window start)
    - already_active_at_window_start (left-censored: possibly already
                                       adopted before the window opened)
    - never_treated_within_window

The stable observation window (needed for long-run outcome measurement)
is expected to bias toward *already-settled* portfolios — i.e., toward
classifying genuine new adopters as left-censored. This script quantifies
that bias and issues a VIABLE / NOT_VIABLE verdict on whether the
resulting "pure new-adopter" cohort is large enough for formal DiD
inference (default threshold: 15 adopters).

Inputs:  df_analysis_master.csv (step1a/1b output)
Outputs: staggered_adoption_customer_level_type{T}.csv
         event_time_panel_type{T}_v1.csv
         event_study_type{T}_v1.png
         staggered_adoption_type{T}_v1_summary.json
================================================================================
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MASTER_PATH, STEP2_OUT, TARGET_TYPE

if not MASTER_PATH.exists():
    raise FileNotFoundError(f"{MASTER_PATH} missing — run step1a first.")

df = pd.read_csv(MASTER_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["customer_id", "date"]).reset_index(drop=True)

N_CUSTOMERS = df["customer_id"].nunique()
print(f"Loaded: {len(df):,} rows / {N_CUSTOMERS} customers\n")

cost_col = f"cost_type{TARGET_TYPE}"
if cost_col not in df.columns:
    raise ValueError(f"Column {cost_col} not found in df_analysis_master.csv")

print("=" * 70)
print(f"1. First-adoption timing for type {TARGET_TYPE} (naive, 1st attempt)")
print("=" * 70)

window_start = df.groupby("customer_id")["date"].min().rename("customer_window_start")
window_end = df.groupby("customer_id")["date"].max().rename("customer_window_end")
active = df[df[cost_col] > 0]
first_active = active.groupby("customer_id")["date"].min().rename("first_treated_date")
n_type_users = active["customer_id"].nunique()
print(f"Customers with any type-{TARGET_TYPE} activity: {n_type_users} (of {N_CUSTOMERS})")

rows = []
for cid in sorted(df["customer_id"].unique()):
    w_start, w_end = window_start.loc[cid], window_end.loc[cid]
    if cid in first_active.index:
        f_date = first_active.loc[cid]
        cohort_type = "already_active_at_window_start" if f_date <= w_start else "adopted_within_window"
    else:
        f_date, cohort_type = pd.NaT, "never_treated_within_window"
    rows.append({"customer_id": cid, "campaign_type": TARGET_TYPE,
                 "customer_window_start": w_start, "customer_window_end": w_end,
                 "first_treated_date": f_date, "cohort_type": cohort_type,
                 "cohort_period": f_date.strftime("%Y-%m-%d") if pd.notna(f_date) else "never_treated"})

adoption_df = pd.DataFrame(rows)
adoption_path = STEP2_OUT / f"staggered_adoption_customer_level_type{TARGET_TYPE}.csv"
adoption_df.to_csv(adoption_path, index=False)

print(f"\n[type {TARGET_TYPE}] cohort distribution:")
cohort_counts = adoption_df["cohort_type"].value_counts()
print(cohort_counts)
n_adopted = int(cohort_counts.get("adopted_within_window", 0))
n_censored = int(cohort_counts.get("already_active_at_window_start", 0))
n_never = int(cohort_counts.get("never_treated_within_window", 0))
if n_censored:
    print(f"\n{n_censored} customers were already active before the window opened (left-censored) — "
          f"excluded from the 'pure new-adopter' sample, subject to step2b recovery.")
print(f"\nSaved customer-level adoption table: {adoption_path}")

print("\n" + "=" * 70)
print("2. First-pass event-time panel + event-study plot")
print("=" * 70)
EVENT_WINDOW = int(os.environ.get("EVENT_WINDOW_DAYS", "30"))
OUTCOME_VARS = [c for c in ["log_spend_safe", "n_campaign_types_active"] if c in df.columns]
coh = adoption_df[adoption_df["cohort_type"] == "adopted_within_window"][["customer_id", "first_treated_date"]]

event_summary = None
if coh.empty:
    print(f"[type {TARGET_TYPE}] no pure new-adopters found — cannot build an event-time panel.")
else:
    panel_t = df.merge(coh, on="customer_id", how="inner")
    panel_t["event_time"] = (panel_t["date"] - panel_t["first_treated_date"]).dt.days
    panel_t = panel_t[panel_t["event_time"].between(-EVENT_WINDOW, EVENT_WINDOW)]

    ev_path = STEP2_OUT / f"event_time_panel_type{TARGET_TYPE}_v1.csv"
    keep_cols = ["customer_id", "date", "first_treated_date", "event_time"] + OUTCOME_VARS
    share_col = f"cost_share_type{TARGET_TYPE}"
    if share_col in df.columns:
        keep_cols += [share_col]
    panel_t[keep_cols].to_csv(ev_path, index=False)
    print(f"Saved event-time panel: {ev_path} ({coh['customer_id'].nunique()} adopters, "
          f"+/-{EVENT_WINDOW} days, {len(panel_t):,} rows)")

    n_series = len(OUTCOME_VARS)
    fig, axes = plt.subplots(1, n_series, figsize=(6 * n_series, 4), squeeze=False)
    for j, ycol in enumerate(OUTCOME_VARS):
        ev_mean = panel_t.groupby("event_time")[ycol].mean()
        ax = axes[0][j]
        ax.plot(ev_mean.index, ev_mean.values, marker="o", markersize=3, color="steelblue")
        ax.axvline(0, color="firebrick", linestyle="--", linewidth=1, label="Adoption date")
        ax.set_title(f"Type {TARGET_TYPE} (v1, n={coh['customer_id'].nunique()}): {ycol}")
        ax.set_xlabel("Event time (days from first adoption)")
        ax.set_ylabel(ycol)
        ax.legend()
    plt.tight_layout()
    fig_path = STEP2_OUT / f"event_study_type{TARGET_TYPE}_v1.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved event-study plot: {fig_path}")
    event_summary = {"n_adopters": int(coh["customer_id"].nunique()), "n_rows": int(len(panel_t))}

print("\n" + "=" * 70)
print("3. Viability verdict")
print("=" * 70)
MIN_VIABLE_ADOPTERS = int(os.environ.get("MIN_VIABLE_ADOPTERS", "15"))
viable = n_adopted >= MIN_VIABLE_ADOPTERS
if viable:
    verdict = "VIABLE_PROCEED_TO_STEP_C"
    msg = (f"{n_adopted} pure new-adopters meets the minimum viability threshold "
           f"({MIN_VIABLE_ADOPTERS}) — proceed directly to the DiD estimation step, "
           f"skipping left-censoring recovery.")
else:
    verdict = "NOT_VIABLE_NEED_RECOVERY"
    msg = (f"{n_adopted} pure new-adopters falls short of the viability threshold "
           f"({MIN_VIABLE_ADOPTERS}). Proceed to step2b to test whether some of the "
           f"{n_censored} left-censored customers can be recovered.")
print(f"Verdict: {verdict}\n{msg}")

summary = {
    "target_type": TARGET_TYPE, "n_customers_total": int(N_CUSTOMERS), "n_type_users_ever": int(n_type_users),
    "cohort_counts": {"adopted_within_window": n_adopted, "already_active_at_window_start": n_censored,
                       "never_treated_within_window": n_never},
    "event_window_days": EVENT_WINDOW, "event_time_panel_v1_summary": event_summary,
    "min_viable_adopters_threshold": MIN_VIABLE_ADOPTERS, "verdict": verdict, "verdict_message": msg,
}
summary_path = STEP2_OUT / f"staggered_adoption_type{TARGET_TYPE}_v1_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"\nSummary saved: {summary_path}")
print("Next step: step2b_left_censoring_recovery.py" if not viable else "Next step: step3a_stacked_did_estimation.py")
