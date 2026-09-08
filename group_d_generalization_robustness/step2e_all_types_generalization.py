"""
================================================================================
Step 2e -- Generalization Across All Campaign Types
================================================================================
Background:
    The Type-6 analysis (Steps 2.1-3g) found that among 35 recovered
    adopters, 26 (74%) were "born-treated" -- customers whose panel-entry
    date coincides with their adoption date, i.e. new customers who started
    with Type 6 rather than existing customers adding it. Only 7 were
    genuine within-customer "true switchers."

Purpose:
    Test whether this phenomenon is specific to Type 6 or a structural
    pattern across the platform. This script repeats the full diagnostic
    pipeline (first-attempt cohort classification -> left-censoring
    recovery -> pre-window safety check -> final cohort classification ->
    born-treated diagnosis -> Step-3 stack validity check) for every
    campaign type auto-detected in df_analysis_master.csv, then compares
    results across types.

    Campaign types are auto-detected via a regex over cost_type{N} columns
    rather than hard-coded, so the script adapts automatically if the
    platform's campaign-type taxonomy changes.

Input:  customer_day_panel.csv, customer_day_campaign_type_panel.csv,
        df_analysis_master.csv (Step 1 output)
Output: step2_treatment_output/all_types/
            born_treated_diagnosis_type{T}.csv   -- per-type adopter detail
                                                     (gap_days, is_born_treated,
                                                      is_valid_stack, etc.)
            cohort_summary_all_types.csv          -- cross-type comparison (wide)
            cohort_summary_all_types.json
            born_treated_ratio_by_type.png        -- bar chart, born-treated ratio
                                                     and switcher counts by type
================================================================================
"""
import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))
OUT_DIR = STEP2_DIR / "all_types"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_WINDOW = int(os.environ.get("EVENT_WINDOW_DAYS", "30"))
BORN_TREATED_GAP_THRESHOLD = int(os.environ.get("BORN_TREATED_GAP_THRESHOLD", "1"))
SAFETY_Z_THRESHOLD = float(os.environ.get("SAFETY_Z_THRESHOLD", "5.0"))

PANEL_PATH = ROOT / "customer_day_panel.csv"
CTP_PATH = ROOT / "customer_day_campaign_type_panel.csv"
MASTER_PATH = ROOT / "df_analysis_master.csv"

for p in [PANEL_PATH, CTP_PATH, MASTER_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p} -- run step1_data_integrity_build.py first.")


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


print("Loading data...")
panel = pd.read_csv(PANEL_PATH)
ctp = pd.read_csv(CTP_PATH)
master = pd.read_csv(MASTER_PATH)

DATE_COL = _detect_date_column(panel)
DATE_COL_CTP = _detect_date_column(ctp)
panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])
ctp[DATE_COL_CTP] = pd.to_datetime(ctp[DATE_COL_CTP])
ctp["campaign_type"] = pd.to_numeric(ctp["campaign_type"], errors="coerce")
master["date"] = pd.to_datetime(master["date"])

# ------------------------------------------------------------------
# 0-1. Auto-detect campaign types (instead of hard-coding, e.g., type 6)
# ------------------------------------------------------------------
TYPE_PATTERN = re.compile(r"^cost_type(\d+)$")
TYPE_CODES = sorted({
    int(m.group(1)) for c in master.columns if (m := TYPE_PATTERN.match(c))
})
if not TYPE_CODES:
    raise ValueError("No cost_type{N} columns found in master.csv.")
print(f"Detected campaign types: {TYPE_CODES}\n")

# ------------------------------------------------------------------
# 0-2. Rebuild the extended outcome panel (same definition as Step 3,
#      type-agnostic -- total spend / portfolio breadth)
# ------------------------------------------------------------------
print("Building the extended outcome panel (including pre-window data)...")
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
raw_date_min_all = panel.groupby("customer_id")[DATE_COL].min().rename("raw_panel_date_min")
print(f"Combined outcome panel: {len(combined_outcomes):,} rows / {combined_outcomes['customer_id'].nunique()} customers\n")


# ============================================================================
# Per-type diagnostic pipeline (mirrors Steps 2.1 + 2.2[A-D] + 3b + 3c)
# ============================================================================
def run_type_diagnosis(T: int):
    cost_col = f"cost_type{T}"

    window_start = master.groupby("customer_id")["date"].min().rename("customer_window_start")

    active = master[master[cost_col] > 0]
    n_ever_active = int(active["customer_id"].nunique())
    if n_ever_active == 0:
        return None

    first_active = active.groupby("customer_id")["date"].min().rename("first_treated_date")

    # --- First-attempt cohort classification (Step 2.1 equivalent) ---
    rows = []
    for cid in sorted(master["customer_id"].unique()):
        w_start = window_start.loc[cid]
        if cid in first_active.index:
            f_date = first_active.loc[cid]
            cohort_type = "already_active_at_window_start" if f_date <= w_start else "adopted_within_window"
        else:
            f_date, cohort_type = pd.NaT, "never_treated_within_window"
        rows.append({"customer_id": cid, "customer_window_start": w_start,
                      "first_treated_date": f_date, "cohort_type": cohort_type})
    adoption_df = pd.DataFrame(rows)
    adoption_df["first_treated_date"] = pd.to_datetime(adoption_df["first_treated_date"])

    n_adopted = int((adoption_df["cohort_type"] == "adopted_within_window").sum())
    n_censored = int((adoption_df["cohort_type"] == "already_active_at_window_start").sum())
    n_never = int((adoption_df["cohort_type"] == "never_treated_within_window").sum())

    LEFT_CENSORED = adoption_df[adoption_df["cohort_type"] == "already_active_at_window_start"][
        ["customer_id", "customer_window_start"]
    ]

    # --- Left-censoring recovery diagnosis (Step 2.2 A/B equivalent) ---
    recovered_rows = []
    for _, row in LEFT_CENSORED.iterrows():
        cid = row["customer_id"]
        sub = ctp[(ctp["customer_id"] == cid) & (ctp["campaign_type"] == T) & (ctp["cost"] > 0)]
        if sub.empty:
            recovered_rows.append({"customer_id": cid, "true_first_active_date": pd.NaT,
                                    "recovery_status": "no_ctp_cost_record"})
            continue
        true_first = sub[DATE_COL_CTP].min()
        status = ("recovered_earlier_than_stable_window" if true_first < row["customer_window_start"]
                   else "consistent_with_stable_window")
        recovered_rows.append({"customer_id": cid, "true_first_active_date": true_first, "recovery_status": status})
    recovered_df = pd.DataFrame(recovered_rows)
    if not recovered_df.empty:
        recovered_df["true_first_active_date"] = pd.to_datetime(recovered_df["true_first_active_date"])
    n_recoverable = int((recovered_df["recovery_status"] == "recovered_earlier_than_stable_window").sum()) \
        if not recovered_df.empty else 0

    # --- Safety validation: SAFE / CAUTION / UNSAFE (Step 2.2 C equivalent) ---
    safety_rows = []
    if not recovered_df.empty:
        target_safety = recovered_df[recovered_df["recovery_status"] == "recovered_earlier_than_stable_window"].merge(
            LEFT_CENSORED, on="customer_id", how="left"
        )
        for _, row in target_safety.iterrows():
            cid, w_start, true_first = row["customer_id"], row["customer_window_start"], row["true_first_active_date"]
            pre = panel[(panel["customer_id"] == cid) & (panel[DATE_COL] >= true_first)
                        & (panel[DATE_COL] < w_start)].sort_values(DATE_COL)
            stable_ref = master[(master["customer_id"] == cid) & (master["date"] >= w_start)] \
                .sort_values("date").head(30)
            if pre.empty:
                safety_rows.append({"customer_id": cid, "verdict": "UNSAFE"})
                continue
            flag_cols = [c for c in ["is_test_account", "billing_anomaly_flag"] if c in pre.columns]
            has_anomaly = bool(pre[flag_cols].fillna(False).any().any()) if flag_cols else False
            expected_days = (pre[DATE_COL].max() - pre[DATE_COL].min()).days + 1
            actual_days = pre[DATE_COL].nunique()
            continuity_ratio = actual_days / expected_days if expected_days > 0 else np.nan
            gap_flag = continuity_ratio < 0.8
            pre_log_cost = np.log1p(pre["cost"]) if "cost" in pre.columns else pd.Series(dtype=float)
            ref_log_cost = (np.log1p(stable_ref["cost"]) if "cost" in stable_ref.columns and not stable_ref.empty
                             else pd.Series(dtype=float))
            pre_std = pre_log_cost.std() if len(pre_log_cost) > 1 else np.nan
            ref_std = ref_log_cost.std() if len(ref_log_cost) > 1 else np.nan
            volatility_ratio = (pre_std / ref_std) if (ref_std and ref_std > 0 and pd.notna(pre_std)) else np.nan
            volatility_flag = pd.notna(volatility_ratio) and volatility_ratio >= 2.0
            if len(pre_log_cost) > 1 and pre_log_cost.std() > 0:
                z_pre = (pre_log_cost - pre_log_cost.mean()) / pre_log_cost.std()
                extreme_ratio = (z_pre.abs() > SAFETY_Z_THRESHOLD).mean()
            else:
                extreme_ratio = np.nan
            extreme_flag = pd.notna(extreme_ratio) and extreme_ratio > 0.05
            n_warn = sum([gap_flag, volatility_flag, extreme_flag])
            if has_anomaly or n_warn >= 2:
                verdict = "UNSAFE"
            elif n_warn == 1:
                verdict = "CAUTION"
            else:
                verdict = "SAFE"
            safety_rows.append({"customer_id": cid, "verdict": verdict})
    safety_df = pd.DataFrame(safety_rows)
    n_safe = int((safety_df["verdict"] == "SAFE").sum()) if not safety_df.empty else 0
    n_caution = int((safety_df["verdict"] == "CAUTION").sum()) if not safety_df.empty else 0
    n_unsafe = int((safety_df["verdict"] == "UNSAFE").sum()) if not safety_df.empty else 0

    # --- Final cohort classification (Step 2.2 D equivalent) ---
    safety_map = safety_df.set_index("customer_id")["verdict"].to_dict() if not safety_df.empty else {}
    recovered_map = (recovered_df.set_index("customer_id")[["true_first_active_date", "recovery_status"]]
                      .to_dict("index") if not recovered_df.empty else {})

    final_rows = []
    for _, row in adoption_df.iterrows():
        cid = row["customer_id"]
        if row["cohort_type"] == "adopted_within_window":
            final_rows.append({"customer_id": cid, "final_first_treated_date": row["first_treated_date"],
                                "final_cohort": "primary_adopter"})
        elif row["cohort_type"] == "never_treated_within_window":
            final_rows.append({"customer_id": cid, "final_first_treated_date": pd.NaT,
                                "final_cohort": "never_treated"})
        else:
            verdict = safety_map.get(cid)
            rec = recovered_map.get(cid, {})
            rec_status, rec_date = rec.get("recovery_status"), rec.get("true_first_active_date", pd.NaT)
            if rec_status == "recovered_earlier_than_stable_window" and verdict == "SAFE":
                final_rows.append({"customer_id": cid, "final_first_treated_date": rec_date,
                                    "final_cohort": "primary_adopter"})
            elif rec_status == "recovered_earlier_than_stable_window" and verdict == "CAUTION":
                final_rows.append({"customer_id": cid, "final_first_treated_date": rec_date,
                                    "final_cohort": "robustness_only_adopter"})
            else:
                final_rows.append({"customer_id": cid, "final_first_treated_date": pd.NaT,
                                    "final_cohort": "still_left_censored"})
    final_df = pd.DataFrame(final_rows)
    final_df["final_first_treated_date"] = pd.to_datetime(final_df["final_first_treated_date"])

    final_counts = final_df["final_cohort"].value_counts()
    n_primary = int(final_counts.get("primary_adopter", 0))
    n_robust_only = int(final_counts.get("robustness_only_adopter", 0))
    n_still_censored = int(final_counts.get("still_left_censored", 0))
    n_never_final = int(final_counts.get("never_treated", 0))

    # --- Born-treated diagnosis (Step 3c equivalent) ---
    adopters_all = final_df[final_df["final_cohort"].isin(["primary_adopter", "robustness_only_adopter"])].merge(
        raw_date_min_all, on="customer_id", how="left"
    )
    if adopters_all.empty:
        n_final_adopters = n_born_treated = n_true_switcher = n_valid_stack = 0
        born_ratio = np.nan
        detail = adopters_all.copy()
    else:
        adopters_all["gap_days"] = (adopters_all["final_first_treated_date"]
                                     - adopters_all["raw_panel_date_min"]).dt.days
        adopters_all["is_born_treated"] = adopters_all["gap_days"] <= BORN_TREATED_GAP_THRESHOLD

        n_final_adopters = len(adopters_all)
        n_born_treated = int(adopters_all["is_born_treated"].sum())
        n_true_switcher = n_final_adopters - n_born_treated
        born_ratio = n_born_treated / n_final_adopters if n_final_adopters > 0 else np.nan

        # --- Step-C stack validity check (Step 3b equivalent) ---
        never_treated_ids = final_df.loc[final_df["final_cohort"] == "never_treated", "customer_id"].tolist()
        control_panel_T = combined_outcomes[combined_outcomes["customer_id"].isin(never_treated_ids)]

        valid_rows = []
        for _, r in adopters_all.iterrows():
            cid, g = r["customer_id"], r["final_first_treated_date"]
            ts = combined_outcomes[combined_outcomes["customer_id"] == cid][["date"]].copy()
            ts["event_time"] = (ts["date"] - g).dt.days
            ts = ts[ts["event_time"].between(-EVENT_WINDOW, EVENT_WINDOW)]
            n_pre, n_post = int((ts["event_time"] < 0).sum()), int((ts["event_time"] >= 0).sum())

            window_dates = pd.date_range(g - pd.Timedelta(days=EVENT_WINDOW), g + pd.Timedelta(days=EVENT_WINDOW))
            ctrl = control_panel_T[control_panel_T["date"].isin(window_dates)].copy()
            ctrl["event_time"] = (ctrl["date"] - g).dt.days
            n_ctrl_pre = int((ctrl["event_time"] < 0).shape[0])
            n_ctrl_post = int((ctrl["event_time"] >= 0).shape[0])

            is_valid = (n_pre > 0) and (n_post > 0) and (n_ctrl_pre > 0) and (n_ctrl_post > 0)
            valid_rows.append({"customer_id": cid, "n_pre_treated_days": n_pre,
                                "n_post_treated_days": n_post, "is_valid_stack": is_valid})
        valid_stack_df = pd.DataFrame(valid_rows)
        n_valid_stack = int(valid_stack_df["is_valid_stack"].sum()) if not valid_stack_df.empty else 0
        detail = adopters_all.merge(valid_stack_df, on="customer_id", how="left")

    summary = {
        "campaign_type": T,
        "n_ever_active": n_ever_active,
        "n_adopted_within_window_1st_attempt": n_adopted,
        "n_left_censored_1st_attempt": n_censored,
        "n_never_treated_1st_attempt": n_never,
        "n_recovered_earlier_than_window": n_recoverable,
        "n_safe": n_safe, "n_caution": n_caution, "n_unsafe": n_unsafe,
        "n_primary_adopter": n_primary, "n_robustness_only_adopter": n_robust_only,
        "n_still_left_censored": n_still_censored, "n_never_treated_final": n_never_final,
        "n_final_adopters_total": n_final_adopters,
        "n_born_treated": n_born_treated,
        "n_true_switcher": n_true_switcher,
        "born_treated_ratio": born_ratio,
        "n_valid_stack_for_DiD": n_valid_stack,
    }
    return summary, detail


# ============================================================================
# Iterate over all detected types
# ============================================================================
print("=" * 78)
print("Running per-type diagnosis")
print("=" * 78)

all_summaries = []
for T in TYPE_CODES:
    result = run_type_diagnosis(T)
    if result is None:
        print(f"[type {T}] no customers with any activity -- skipped")
        continue
    summary, detail = result
    detail.to_csv(OUT_DIR / f"born_treated_diagnosis_type{T}.csv", index=False)
    all_summaries.append(summary)

    ratio_str = f"{summary['born_treated_ratio']:.1%}" if pd.notna(summary["born_treated_ratio"]) else "N/A"
    print(f"[type{T:>2}] ever-active {summary['n_ever_active']:>3} | "
          f"1st-attempt adopted {summary['n_adopted_within_window_1st_attempt']:>3} / "
          f"left-censored {summary['n_left_censored_1st_attempt']:>3} | "
          f"final adopters {summary['n_final_adopters_total']:>3} (born-treated {summary['n_born_treated']:>3}, "
          f"ratio {ratio_str:>6}) | true switchers {summary['n_true_switcher']:>3} | "
          f"valid stacks {summary['n_valid_stack_for_DiD']:>3}")

if not all_summaries:
    raise RuntimeError("No campaign type had any activity history -- check master.csv.")

summary_df = pd.DataFrame(all_summaries).sort_values("campaign_type").reset_index(drop=True)

# ------------------------------------------------------------------
# Cross-type comparison verdict
# ------------------------------------------------------------------
print("\n" + "=" * 78)
print("Cross-type comparison -- is the born-treated phenomenon type-specific?")
print("=" * 78)

valid_ratio = summary_df.dropna(subset=["born_treated_ratio"])
if len(valid_ratio) >= 2:
    ratio_std = valid_ratio["born_treated_ratio"].std()
    ratio_mean = valid_ratio["born_treated_ratio"].mean()
    ratio_min_row = valid_ratio.loc[valid_ratio["born_treated_ratio"].idxmin()]
    ratio_max_row = valid_ratio.loc[valid_ratio["born_treated_ratio"].idxmax()]
    print(f"born_treated_ratio mean={ratio_mean:.1%}, SD={ratio_std:.1%}")
    print(f"lowest: type{int(ratio_min_row['campaign_type'])} ({ratio_min_row['born_treated_ratio']:.1%})")
    print(f"highest: type{int(ratio_max_row['campaign_type'])} ({ratio_max_row['born_treated_ratio']:.1%})")

    if ratio_std < 0.10:
        interp = (
            "The born-treated ratio is similar across all types -- consistent with a structural pattern "
            "across the platform (the stable-window filter systematically misclassifying new-customer "
            "onboarding as an incumbent's treatment adoption) rather than a Type-6-specific quirk. This "
            "supports the paper's contribution as a general methodological finding."
        )
    else:
        interp = (
            "The born-treated ratio varies meaningfully across types -- certain type(s) may be "
            "systematically used more often as onboarding channels. This heterogeneity itself could be "
            "an interesting empirical finding worth investigating (why do new customers gravitate to "
            "these particular types?)."
        )
    print(f"\nInterpretation: {interp}")
else:
    interp = "Fewer than two comparable types -- cross-type comparison not possible."
    print(interp)

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
summary_csv_path = OUT_DIR / "cohort_summary_all_types.csv"
summary_df.to_csv(summary_csv_path, index=False)

summary_json_path = OUT_DIR / "cohort_summary_all_types.json"
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "born_treated_gap_threshold_days": BORN_TREATED_GAP_THRESHOLD,
        "event_window_days": EVENT_WINDOW,
        "per_type_summary": all_summaries,
        "cross_type_interpretation": interp,
    }, f, ensure_ascii=False, indent=2, default=str)

# ------------------------------------------------------------------
# Plot: born-treated ratio + final-adopter/true-switcher counts by type
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
colors = ["firebrick" if t == 6 else "steelblue" for t in summary_df["campaign_type"]]
ax.bar(summary_df["campaign_type"].astype(str), summary_df["born_treated_ratio"].fillna(0), color=colors)
ax.set_title("Born-treated ratio by campaign type\n(share of final adopters whose panel-entry date = adoption date)")
ax.set_xlabel("campaign_type")
ax.set_ylabel("born-treated ratio")
ax.set_ylim(0, 1)
for i, v in enumerate(summary_df["born_treated_ratio"].fillna(0)):
    ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)

ax = axes[1]
width = 0.35
x = np.arange(len(summary_df))
ax.bar(x - width / 2, summary_df["n_born_treated"], width, label="born-treated", color="firebrick")
ax.bar(x + width / 2, summary_df["n_true_switcher"], width, label="true switcher", color="steelblue")
ax.set_xticks(x)
ax.set_xticklabels(summary_df["campaign_type"].astype(str))
ax.set_title("Born-treated vs. true switcher counts by campaign type")
ax.set_xlabel("campaign_type")
ax.set_ylabel("number of customers")
ax.legend()

plt.tight_layout()
plot_path = OUT_DIR / "born_treated_ratio_by_type.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"\nPer-type detail saved: {OUT_DIR}/born_treated_diagnosis_type{{T}}.csv")
print(f"Comparison summary saved: {summary_csv_path}")
print(f"Comparison summary (JSON) saved: {summary_json_path}")
print(f"Comparison plot saved: {plot_path}")
