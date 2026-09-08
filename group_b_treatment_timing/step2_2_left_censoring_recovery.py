"""
================================================================================
Step 2.2 — Left-Censoring Recovery, Safety Validation, and Final Cohort
================================================================================
Background (Step 2.1 result): of 37 type-6 users, only 7 were "pure new
adopters"; 30 were left-censored (already active at window start); verdict
was NOT_VIABLE_NEED_RECOVERY.

What this script does:
    A. Coverage check: for the 30 left-censored customers, verify whether the
       raw customer_day_panel.csv date range extends earlier than the stable
       window's start (structural censoring vs. an artifact of the
       stable-window filter).
    B. True-adoption-date recovery: search the raw campaign-type panel
       (beyond the stable window) for the true first date with cost_type6>0.
    C. Safety validation: decide whether the recovered pre-window segment is
       SAFE / CAUTION / UNSAFE to splice into the event-time panel, using four
       criteria (anomaly flags, continuity, volatility, extreme-value density).
    D. Final cohort assembly: combine original within-window adopters (7) with
       recovered SAFE and recovered CAUTION customers into
       primary_adopter / robustness_only_adopter / still_left_censored /
       never_treated.
    E. Outcome recomputation: inside the stable window use the master values;
       for the pre-window segment recompute the same outcome definitions from
       the raw panel/campaign-type panel, then build primary (SAFE only) and
       robust (SAFE+CAUTION) event-time panels and event-study plots.

Input:  customer_day_panel.csv, customer_day_campaign_type_panel.csv,
        df_analysis_master.csv,
        step2_treatment_output/staggered_adoption_customer_level_type{T}.csv
Output: left_censoring_diagnosis_type{T}.json
        pre_window_safety_check_type{T}.csv
        pre_window_safety_summary_type{T}.json
        staggered_adoption_FINAL_type{T}.csv
        event_time_panel_FINAL_type{T}.csv   (primary only)
        event_time_panel_ROBUST_type{T}.csv  (primary+caution)
        event_study_FINAL_type{T}.png
        event_study_ROBUST_type{T}.png
        staggered_adoption_FINAL_type{T}_summary.json
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
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))
STEP2_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TYPE = int(os.environ.get("TARGET_TYPE", "6"))

PANEL_PATH = ROOT / "customer_day_panel.csv"
CTP_PATH = ROOT / "customer_day_campaign_type_panel.csv"
MASTER_PATH = ROOT / "df_analysis_master.csv"
ORIG_ADOPTION_PATH = STEP2_DIR / f"staggered_adoption_customer_level_type{TARGET_TYPE}.csv"

for p in [PANEL_PATH, CTP_PATH, MASTER_PATH, ORIG_ADOPTION_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p} — run Step 2.1 first.")

panel = pd.read_csv(PANEL_PATH)
ctp = pd.read_csv(CTP_PATH)
master = pd.read_csv(MASTER_PATH)
orig = pd.read_csv(ORIG_ADOPTION_PATH)


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
orig["customer_window_start"] = pd.to_datetime(orig["customer_window_start"])
orig["first_treated_date"] = pd.to_datetime(orig["first_treated_date"])
ctp["campaign_type"] = pd.to_numeric(ctp["campaign_type"], errors="coerce")

LEFT_CENSORED = orig[orig["cohort_type"] == "already_active_at_window_start"][
    ["customer_id", "campaign_type", "customer_window_start"]
].drop_duplicates()

print(f"Left-censored customer count: {len(LEFT_CENSORED)} (type {TARGET_TYPE})\n")

# ============================================================================
# A. Does the raw panel extend earlier than the stable-window start?
# ============================================================================
print("=" * 70)
print("A. Raw panel.csv date coverage vs. stable-window start")
print("=" * 70)

raw_range = panel.groupby("customer_id")[DATE_COL].agg(["min", "max"]).rename(
    columns={"min": "raw_date_min", "max": "raw_date_max"}
)

coverage_check = LEFT_CENSORED.merge(raw_range, on="customer_id", how="left")
coverage_check["has_pre_window_data"] = (
    coverage_check["raw_date_min"] < coverage_check["customer_window_start"]
)
coverage_check["days_available_before_window"] = (
    coverage_check["customer_window_start"] - coverage_check["raw_date_min"]
).dt.days.clip(lower=0)

n_recoverable_candidates = int(coverage_check["has_pre_window_data"].sum())
print(f"Customers with raw data earlier than the stable-window start: "
      f"{n_recoverable_candidates} / {len(coverage_check)}")

if n_recoverable_candidates == 0:
    print("\nNOTE: no pre-window data exists even in the raw panel — structural "
          "left-censoring confirmed. Recovery not possible.")

# ============================================================================
# B. For recoverable candidates, search for the true first-adoption date
#    including the pre-window segment (based on cost_type6 > 0 in ctp)
# ============================================================================
print("\n" + "=" * 70)
print("B. Recovering the true first-adoption date beyond the stable window")
print("=" * 70)

recovered_rows = []
for _, row in LEFT_CENSORED.iterrows():
    cid, tcode = row["customer_id"], row["campaign_type"]
    sub = ctp[
        (ctp["customer_id"] == cid)
        & (ctp["campaign_type"] == tcode)
        & (ctp["cost"] > 0)
    ]
    if sub.empty:
        recovered_rows.append({
            "customer_id": cid, "campaign_type": tcode,
            "true_first_active_date": pd.NaT, "recovery_status": "no_ctp_cost_record",
        })
        continue

    true_first = sub[DATE_COL_CTP].min()
    w_start = row["customer_window_start"]
    if true_first < w_start:
        status = "recovered_earlier_than_stable_window"
    else:
        status = "consistent_with_stable_window"
    recovered_rows.append({
        "customer_id": cid, "campaign_type": tcode,
        "true_first_active_date": true_first, "recovery_status": status,
    })

recovered_df = pd.DataFrame(recovered_rows)
recovered_df["true_first_active_date"] = pd.to_datetime(recovered_df["true_first_active_date"])
n_actually_recovered = int((recovered_df["recovery_status"] == "recovered_earlier_than_stable_window").sum())
print(f"Cases where the ctp-based true adoption date is earlier than the window start: "
      f"{n_actually_recovered} / {len(recovered_df)}")
print(recovered_df["recovery_status"].value_counts())

merged_diag = LEFT_CENSORED.merge(recovered_df, on=["customer_id", "campaign_type"], how="left")

if n_recoverable_candidates == 0 and n_actually_recovered == 0:
    diag_verdict = "STRUCTURAL_LEFT_CENSORING_CONFIRMED"
    diag_msg = (
        "No pre-window segment exists in the raw data, or where it does exist the "
        "customer was already active. Structural left-censoring is confirmed and "
        "recovery is not possible — consider switching to a dose-response "
        "complementary strategy."
    )
elif n_actually_recovered > 0:
    diag_verdict = "PARTIALLY_RECOVERABLE"
    diag_msg = (
        f"{n_actually_recovered} case(s) can recover a true adoption date earlier "
        f"than the stable-window start from the raw ctp data. Since this segment "
        f"lies outside the stable window (is_stable_window), it should only be "
        f"spliced into the event-time panel after a separate data-quality "
        f"validation (Step C below)."
    )
else:
    diag_verdict = "NO_ADDITIONAL_RECOVERY"
    diag_msg = "Candidates existed but re-verification found no additional recovery."

print(f"\nVerdict: {diag_verdict}")
print(diag_msg)

diag_path = STEP2_DIR / f"left_censoring_diagnosis_type{TARGET_TYPE}.json"
with open(diag_path, "w", encoding="utf-8") as f:
    json.dump({
        "n_left_censored": int(len(LEFT_CENSORED)),
        "n_pre_window_data_available": n_recoverable_candidates,
        "n_actually_recovered_earlier_date": n_actually_recovered,
        "verdict": diag_verdict,
        "verdict_message": diag_msg,
    }, f, ensure_ascii=False, indent=2, default=str)
print(f"\nDiagnosis saved: {diag_path}")

# ============================================================================
# C. Safety validation of the recovered pre-window segment (SAFE/CAUTION/UNSAFE)
# ============================================================================
print("\n" + "=" * 70)
print("C. Safety validation of the recovered pre-window segment")
print("=" * 70)

TARGET_SAFETY = recovered_df[recovered_df["recovery_status"] == "recovered_earlier_than_stable_window"].copy()
TARGET_SAFETY = TARGET_SAFETY.merge(
    LEFT_CENSORED[["customer_id", "campaign_type", "customer_window_start"]],
    on=["customer_id", "campaign_type"], how="left"
)
print(f"Cases to validate: {len(TARGET_SAFETY)}\n")

safety_results = []
for _, row in TARGET_SAFETY.iterrows():
    cid = row["customer_id"]
    w_start = row["customer_window_start"]
    true_first = row["true_first_active_date"]

    pre = panel[
        (panel["customer_id"] == cid)
        & (panel[DATE_COL] >= true_first)
        & (panel[DATE_COL] < w_start)
    ].sort_values(DATE_COL)

    stable_ref = master[
        (master["customer_id"] == cid) & (master["date"] >= w_start)
    ].sort_values("date").head(30)

    if pre.empty:
        safety_results.append({"customer_id": cid, "campaign_type": row["campaign_type"],
                                "verdict": "UNSAFE", "reason": "pre_window_no_panel_rows"})
        continue

    # Criterion 1: anomaly-flag contamination
    flag_cols = [c for c in ["is_test_account", "billing_anomaly_flag"] if c in pre.columns]
    has_anomaly_flag = bool(pre[flag_cols].fillna(False).any().any()) if flag_cols else False

    # Criterion 2: observation continuity
    expected_days = (pre[DATE_COL].max() - pre[DATE_COL].min()).days + 1
    actual_days = pre[DATE_COL].nunique()
    continuity_ratio = actual_days / expected_days if expected_days > 0 else np.nan
    gap_flag = continuity_ratio < 0.8

    # Criterion 3: volatility comparison (log1p(cost))
    pre_log_cost = np.log1p(pre["cost"]) if "cost" in pre.columns else pd.Series(dtype=float)
    ref_log_cost = np.log1p(stable_ref["cost"]) if "cost" in stable_ref.columns and not stable_ref.empty else pd.Series(dtype=float)
    pre_std = pre_log_cost.std() if len(pre_log_cost) > 1 else np.nan
    ref_std = ref_log_cost.std() if len(ref_log_cost) > 1 else np.nan
    volatility_ratio = (pre_std / ref_std) if (ref_std and ref_std > 0 and pd.notna(pre_std)) else np.nan
    volatility_flag = pd.notna(volatility_ratio) and volatility_ratio >= 2.0

    # Criterion 4: extreme-value density
    if len(pre_log_cost) > 1 and pre_log_cost.std() > 0:
        z_pre = (pre_log_cost - pre_log_cost.mean()) / pre_log_cost.std()
        extreme_ratio_pre = (z_pre.abs() > 5).mean()
    else:
        extreme_ratio_pre = np.nan
    extreme_flag = pd.notna(extreme_ratio_pre) and extreme_ratio_pre > 0.05

    n_warn_flags = sum([gap_flag, volatility_flag, extreme_flag])
    if has_anomaly_flag:
        verdict, reason = "UNSAFE", "test_account_or_billing_anomaly_in_pre_window"
    elif n_warn_flags >= 2:
        verdict, reason = "UNSAFE", f"multiple_warning_flags({n_warn_flags})"
    elif n_warn_flags == 1:
        verdict, reason = "CAUTION", "single_warning_flag"
    else:
        verdict, reason = "SAFE", "no_flags"

    safety_results.append({
        "customer_id": cid, "campaign_type": row["campaign_type"],
        "true_first_active_date": true_first, "customer_window_start": w_start,
        "pre_window_days_span": expected_days, "pre_window_days_observed": actual_days,
        "continuity_ratio": round(continuity_ratio, 3) if pd.notna(continuity_ratio) else None,
        "has_anomaly_flag": has_anomaly_flag,
        "volatility_ratio_pre_vs_stable": round(volatility_ratio, 3) if pd.notna(volatility_ratio) else None,
        "extreme_value_ratio_pre": round(extreme_ratio_pre, 4) if pd.notna(extreme_ratio_pre) else None,
        "n_warning_flags": n_warn_flags, "verdict": verdict, "reason": reason,
    })

safety_df = pd.DataFrame(safety_results)
safety_path = STEP2_DIR / f"pre_window_safety_check_type{TARGET_TYPE}.csv"
safety_df.to_csv(safety_path, index=False)

print("Verdict distribution:")
print(safety_df["verdict"].value_counts())

n_safe = int((safety_df["verdict"] == "SAFE").sum())
n_caution = int((safety_df["verdict"] == "CAUTION").sum())
n_unsafe = int((safety_df["verdict"] == "UNSAFE").sum())

safety_summary = {
    "n_checked": int(len(safety_df)), "n_safe": n_safe, "n_caution": n_caution, "n_unsafe": n_unsafe,
}
safety_summary_path = STEP2_DIR / f"pre_window_safety_summary_type{TARGET_TYPE}.json"
with open(safety_summary_path, "w", encoding="utf-8") as f:
    json.dump(safety_summary, f, ensure_ascii=False, indent=2, default=str)

print(f"\nValidation results saved: {safety_path}")
print(f"Summary saved: {safety_summary_path}")
print(f"\nSAFE={n_safe} / CAUTION={n_caution} / UNSAFE={n_unsafe}")

# ============================================================================
# D. Final cohort classification
# ============================================================================
print("\n" + "=" * 70)
print("D. Final cohort classification")
print("=" * 70)

safety_map = safety_df.set_index(["customer_id", "campaign_type"])["verdict"].to_dict()
recovered_map = recovered_df.set_index(["customer_id", "campaign_type"])[
    ["true_first_active_date", "recovery_status"]
].to_dict("index")

final_rows = []
for _, row in orig.iterrows():
    cid, tcode = row["customer_id"], row["campaign_type"]
    if row["cohort_type"] == "adopted_within_window":
        final_rows.append({
            "customer_id": cid, "campaign_type": tcode,
            "final_first_treated_date": row["first_treated_date"],
            "final_cohort": "primary_adopter", "source": "original_within_window",
        })
    elif row["cohort_type"] == "never_treated_within_window":
        final_rows.append({
            "customer_id": cid, "campaign_type": tcode,
            "final_first_treated_date": pd.NaT,
            "final_cohort": "never_treated", "source": "original",
        })
    else:  # already_active_at_window_start
        verdict = safety_map.get((cid, tcode))
        rec = recovered_map.get((cid, tcode), {})
        rec_status = rec.get("recovery_status", "unknown")
        rec_date = rec.get("true_first_active_date", pd.NaT)

        if rec_status == "recovered_earlier_than_stable_window" and verdict == "SAFE":
            final_rows.append({
                "customer_id": cid, "campaign_type": tcode,
                "final_first_treated_date": rec_date,
                "final_cohort": "primary_adopter", "source": "recovered_safe",
            })
        elif rec_status == "recovered_earlier_than_stable_window" and verdict == "CAUTION":
            final_rows.append({
                "customer_id": cid, "campaign_type": tcode,
                "final_first_treated_date": rec_date,
                "final_cohort": "robustness_only_adopter", "source": "recovered_caution",
            })
        else:
            final_rows.append({
                "customer_id": cid, "campaign_type": tcode,
                "final_first_treated_date": pd.NaT,
                "final_cohort": "still_left_censored", "source": f"recovery_status={rec_status}",
            })

final_df = pd.DataFrame(final_rows)
final_df["final_first_treated_date"] = pd.to_datetime(final_df["final_first_treated_date"])
final_path = STEP2_DIR / f"staggered_adoption_FINAL_type{TARGET_TYPE}.csv"
final_df.to_csv(final_path, index=False)

print(f"\n[type {TARGET_TYPE}] Final cohort distribution:")
final_counts = final_df["final_cohort"].value_counts()
print(final_counts)
print(f"\nFinal cohort table saved: {final_path}")

n_primary = int(final_counts.get("primary_adopter", 0))
n_robust_only = int(final_counts.get("robustness_only_adopter", 0))
n_still_censored = int(final_counts.get("still_left_censored", 0))
n_never = int(final_counts.get("never_treated", 0))

# ============================================================================
# E. Extended outcome computation + primary/robust event-time panels & plots
# ============================================================================
print("\n" + "=" * 70)
print("E. Extended outcome computation + final event-time panels")
print("=" * 70)

panel_ext = panel[["customer_id", DATE_COL, "cost"]].copy()
panel_ext["log_spend_safe"] = np.log1p(panel_ext["cost"])
panel_ext = panel_ext.rename(columns={DATE_COL: "date"})

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
print(f"Combined outcome panel: {len(combined_outcomes):,} rows")

EVENT_WINDOW = int(os.environ.get("EVENT_WINDOW_DAYS", "30"))
OUTCOME_VARS = ["log_spend_safe", "n_campaign_types_active"]


def build_event_panel_and_plot(cohort_set, tag):
    coh = final_df[final_df["final_cohort"].isin(cohort_set)][
        ["customer_id", "final_first_treated_date"]
    ].dropna()

    if coh.empty:
        print(f"[{tag}] No target customers — skipping")
        return None

    panel_t = combined_outcomes.merge(coh, on="customer_id", how="inner")
    panel_t["event_time"] = (panel_t["date"] - panel_t["final_first_treated_date"]).dt.days
    panel_t = panel_t[panel_t["event_time"].between(-EVENT_WINDOW, EVENT_WINDOW)]

    ev_path = STEP2_DIR / f"event_time_panel_{tag}_type{TARGET_TYPE}.csv"
    panel_t.to_csv(ev_path, index=False)
    n_adopters = coh["customer_id"].nunique()
    print(f"[{tag}] Event-time panel saved: {ev_path} "
          f"({n_adopters} adopters, +/-{EVENT_WINDOW} days, {len(panel_t):,} rows)")

    fig, axes = plt.subplots(1, len(OUTCOME_VARS), figsize=(6 * len(OUTCOME_VARS), 4), squeeze=False)
    for j, ycol in enumerate(OUTCOME_VARS):
        ev_mean = panel_t.groupby("event_time")[ycol].mean()
        ax = axes[0][j]
        ax.plot(ev_mean.index, ev_mean.values, marker="o", markersize=3, color="steelblue")
        ax.axvline(0, color="firebrick", linestyle="--", linewidth=1, label="Adoption date")
        ax.set_title(f"Type {TARGET_TYPE} ({tag}, n_adopters={n_adopters}): {ycol}")
        ax.set_xlabel("Event time (days from first adoption)")
        ax.set_ylabel(ycol)
        ax.legend()
    plt.tight_layout()
    fig_path = STEP2_DIR / f"event_study_{tag}_type{TARGET_TYPE}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {fig_path}")
    return {"n_adopters": int(n_adopters), "n_rows": int(len(panel_t))}


r_primary = build_event_panel_and_plot({"primary_adopter"}, "FINAL")
r_robust = build_event_panel_and_plot({"primary_adopter", "robustness_only_adopter"}, "ROBUST")

# ============================================================================
# Save summary
# ============================================================================
summary = {
    "target_type": TARGET_TYPE,
    "left_censoring_diagnosis": {"verdict": diag_verdict, "n_recovered": n_actually_recovered},
    "safety_check": {"n_safe": n_safe, "n_caution": n_caution, "n_unsafe": n_unsafe},
    "final_cohort_counts": {
        "primary_adopter": n_primary,
        "robustness_only_adopter": n_robust_only,
        "still_left_censored": n_still_censored,
        "never_treated": n_never,
    },
    "event_window_days": EVENT_WINDOW,
    "event_time_panel": {"primary_FINAL": r_primary, "robust_ROBUST": r_robust},
}
summary_path = STEP2_DIR / f"staggered_adoption_FINAL_type{TARGET_TYPE}_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print("\n" + "=" * 70)
print("Done")
print("=" * 70)
print(f"First attempt (pure new adoption) 7 -> final primary_adopter {n_primary} "
      f"(robust version {n_primary + n_robust_only})")
print(f"Summary saved: {summary_path}")
print("\nNext step: step2_3_pretrend_test.py for formal pre-trend testing")
print("(uses event_time_panel_FINAL_type{T}.csv, includes near/far window split tests)")
