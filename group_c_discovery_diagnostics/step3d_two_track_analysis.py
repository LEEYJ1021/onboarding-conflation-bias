"""
================================================================================
Step 3d — Track 1 (True Switchers) / Track 2 (Born-Treated) Split Analysis
================================================================================
Background (Step 3c findings):
    Of the 35 primary + robustness_only adopters, 26 (74%) are "born-treated"
    — customers whose panel-entry date coincides with (gap<=1 day) the
    target-type adoption date, meaning no within-customer pre-treatment
    observation can exist for them. The genuine "true switchers" (existing
    customers who later added the target type, gap>1 day) number only 7.

This script:
    Track 1 (True Switchers, DiD):
        - Re-labels customers with gap_days>threshold as "identifiable
          switchers" (irrespective of final_cohort) and re-runs the Step-3
          stacked-DiD ATT_i computation on this subset.
        - The sample is very small (N~7-9), so alongside the pooled
          estimate and a small-sample t-based CI, this script also prints
          the full case-level table -- signalling explicitly that this is
          closer to a case-study than a formal inferential result.

    Track 2 (Born-Treated, cross-sectional matched comparison):
        - For each born-treated customer, finds a matched "concurrent new"
          never-treated customer whose registration date falls within
          +/-REG_MATCH_WINDOW_DAYS of the born-treated customer's own.
        - Compares early-window outcomes (log_spend_safe cumulative,
          n_campaign_types_active) between born-treated and matched
          controls via Welch's t-test + effect size.
        - This is explicitly a descriptive/associational comparison, not a
          causal estimate; self-selection into the target type at
          registration is not addressed here (see Step 3e for covariate
          balance and adjusted-regression follow-ups).

Input:  customer_day_panel.csv, customer_day_campaign_type_panel.csv,
        df_analysis_master.csv, staggered_adoption_FINAL_type{T}.csv,
        step3c_born_treated_diagnosis_type{T}.csv
Output: step3d_track1_switcher_att_type{T}.json
        step3d_track1_switcher_cases_type{T}.csv
        step3d_track2_born_treated_comparison_type{T}.csv
        step3d_track2_summary_type{T}.json
================================================================================
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))

TARGET_TYPE = int(os.environ.get("TARGET_TYPE", "6"))
EVENT_WINDOW = int(os.environ.get("EVENT_WINDOW_DAYS", "30"))
GAP_THRESHOLD = int(os.environ.get("BORN_TREATED_GAP_THRESHOLD", "1"))
REG_WINDOW = int(os.environ.get("REG_MATCH_WINDOW_DAYS", "14"))
EARLY_WINDOW = int(os.environ.get("EARLY_OUTCOME_WINDOW_DAYS", "30"))

PANEL_PATH = ROOT / "customer_day_panel.csv"
CTP_PATH = ROOT / "customer_day_campaign_type_panel.csv"
MASTER_PATH = ROOT / "df_analysis_master.csv"
FINAL_COHORT_PATH = STEP2_DIR / f"staggered_adoption_FINAL_type{TARGET_TYPE}.csv"
BORN_DIAG_PATH = STEP2_DIR / f"step3c_born_treated_diagnosis_type{TARGET_TYPE}.csv"

for p in [PANEL_PATH, CTP_PATH, MASTER_PATH, FINAL_COHORT_PATH, BORN_DIAG_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p}")

panel = pd.read_csv(PANEL_PATH)
ctp = pd.read_csv(CTP_PATH)
master = pd.read_csv(MASTER_PATH)
final_cohort = pd.read_csv(FINAL_COHORT_PATH)
born_diag = pd.read_csv(BORN_DIAG_PATH)


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

OUTCOME_VARS = ["log_spend_safe", "n_campaign_types_active"]

never_treated_ids = final_cohort.loc[
    final_cohort["final_cohort"] == "never_treated", "customer_id"
].unique().tolist()
control_panel = combined_outcomes[combined_outcomes["customer_id"].isin(never_treated_ids)].copy()

# ============================================================================
# TRACK 1 -- True Switchers (gap_days > GAP_THRESHOLD)
# ============================================================================
print("=" * 70)
print(f"TRACK 1. True switchers (gap_days > {GAP_THRESHOLD}) -- re-estimated event-study DiD")
print("=" * 70)

switchers = born_diag[born_diag["gap_days"] > GAP_THRESHOLD][
    ["customer_id", "final_first_treated_date", "final_cohort", "gap_days",
     "raw_panel_date_min", "n_pre_treated_days"]
].copy()
print(f"Identifiable true switchers: {len(switchers)}")
print(switchers.to_string(index=False))

if len(switchers) < 5:
    print("\nNOTE: sample size below 5. Bootstrap CIs are close to meaningless at this scale, so a "
          "formal statistical test is deprioritized in favor of case-level narrative reporting.")


def get_control_values(dates, ycol, control_ids):
    sub = control_panel[control_panel["date"].isin(dates) & control_panel["customer_id"].isin(control_ids)]
    return sub[["customer_id", "date", ycol]].dropna(subset=[ycol])


def compute_att_case_level(adopters: pd.DataFrame, control_ids: list, ycol: str) -> pd.DataFrame:
    rows = []
    for _, r in adopters.iterrows():
        cid, g = r["customer_id"], r["final_first_treated_date"]
        ts = combined_outcomes[combined_outcomes["customer_id"] == cid][["date", ycol]].dropna(subset=[ycol]).copy()
        ts["event_time"] = (ts["date"] - g).dt.days
        ts = ts[ts["event_time"].between(-EVENT_WINDOW, EVENT_WINDOW)]
        pre_t, post_t = ts[ts["event_time"] < 0], ts[ts["event_time"] >= 0]
        if pre_t.empty or post_t.empty:
            continue
        treat_pre, treat_post = pre_t[ycol].mean(), post_t[ycol].mean()

        window_dates = pd.date_range(g - pd.Timedelta(days=EVENT_WINDOW), g + pd.Timedelta(days=EVENT_WINDOW))
        ctrl = get_control_values(window_dates, ycol, control_ids)
        if ctrl.empty:
            continue
        ctrl = ctrl.copy()
        ctrl["event_time"] = (ctrl["date"] - g).dt.days
        ctrl_pre_grp = ctrl.loc[ctrl["event_time"] < 0].groupby("customer_id")[ycol].mean()
        ctrl_post_grp = ctrl.loc[ctrl["event_time"] >= 0].groupby("customer_id")[ycol].mean()
        if ctrl_pre_grp.empty or ctrl_post_grp.empty:
            continue
        ctrl_pre, ctrl_post = ctrl_pre_grp.mean(), ctrl_post_grp.mean()

        att_i = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)
        rows.append({
            "customer_id": cid, "first_treated_date": g,
            "n_pre_days": len(pre_t), "n_post_days": len(post_t),
            "treat_pre_mean": treat_pre, "treat_post_mean": treat_post,
            "control_pre_mean": ctrl_pre, "control_post_mean": ctrl_post,
            "att_i": att_i,
        })
    return pd.DataFrame(rows)


track1_summary = {}
track1_cases_all = []
for ycol in OUTCOME_VARS:
    print(f"\n--- Track 1 outcome: {ycol} ---")
    cases = compute_att_case_level(switchers, never_treated_ids, ycol)
    if cases.empty:
        print("  no valid cases -- skipped")
        continue
    cases["outcome"] = ycol
    track1_cases_all.append(cases)

    n_valid = len(cases)
    mean_att = float(cases["att_i"].mean())
    print(f"  valid cases: {n_valid}")
    print(cases[["customer_id", "n_pre_days", "n_post_days", "att_i"]].to_string(index=False))
    print(f"  simple-average ATT = {mean_att:+.4f}")

    if n_valid >= 5:
        se = cases["att_i"].std(ddof=1) / np.sqrt(n_valid)
        tcrit = sps.t.ppf(0.975, df=n_valid - 1)
        ci_lo, ci_hi = mean_att - tcrit * se, mean_att + tcrit * se
        tstat = mean_att / se if se > 0 else np.nan
        pval = float(2 * (1 - sps.t.cdf(abs(tstat), df=n_valid - 1))) if pd.notna(tstat) else np.nan
        print(f"  t-distribution 95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}], p={pval:.4f}")
        track1_summary[ycol] = {
            "n_valid": n_valid, "mean_att": mean_att, "se": float(se),
            "ci_95_lo": float(ci_lo), "ci_95_hi": float(ci_hi), "pvalue": pval,
            "method": "t-distribution (small-sample, not bootstrap)",
        }
    else:
        print(f"  NOTE: n<5 -- formal CI omitted, case-level reporting only")
        track1_summary[ycol] = {
            "n_valid": n_valid, "mean_att": mean_att,
            "method": "case-level only (n<5, no formal inference)",
            "individual_att": cases[["customer_id", "att_i"]].to_dict("records"),
        }

if track1_cases_all:
    track1_cases_df = pd.concat(track1_cases_all, ignore_index=True)
    cases_path = STEP2_DIR / f"step3d_track1_switcher_cases_type{TARGET_TYPE}.csv"
    track1_cases_df.to_csv(cases_path, index=False)
    print(f"\nTrack 1 case table saved: {cases_path}")

track1_out_path = STEP2_DIR / f"step3d_track1_switcher_att_type{TARGET_TYPE}.json"
with open(track1_out_path, "w", encoding="utf-8") as f:
    json.dump({
        "n_switchers": len(switchers),
        "gap_threshold_days": GAP_THRESHOLD,
        "event_window_days": EVENT_WINDOW,
        "results": track1_summary,
        "caveat": (
            f"This track's sample is only {len(switchers)} customers. Interpretation should emphasize "
            "sign and cross-case consistency over statistical significance, and the paper should "
            "explicitly label this as exploratory / case-level evidence."
        ),
    }, f, ensure_ascii=False, indent=2, default=str)
print(f"Track 1 summary saved: {track1_out_path}")

# ============================================================================
# TRACK 2 -- Born-Treated Cross-Sectional Matched Comparison
# ============================================================================
n_born_total = int((born_diag["gap_days"] <= GAP_THRESHOLD).sum())
print("\n" + "=" * 70)
print(f"TRACK 2. Born-treated (gap_days<={GAP_THRESHOLD}, N={n_born_total}) matched comparison")
print("=" * 70)

born_treated = born_diag[born_diag["gap_days"] <= GAP_THRESHOLD][
    ["customer_id", "final_first_treated_date", "raw_panel_date_min"]
].rename(columns={"raw_panel_date_min": "registration_date"}).copy()

never_treated_reg = panel[panel["customer_id"].isin(never_treated_ids)].groupby("customer_id")[DATE_COL].min()
never_treated_reg = never_treated_reg.rename("registration_date").reset_index()

print(f"Born-treated customers: {len(born_treated)} / candidate never-treated pool: {len(never_treated_reg)}")


def early_outcome(cid: str, reg_date: pd.Timestamp, ycol: str):
    sub = combined_outcomes[
        (combined_outcomes["customer_id"] == cid)
        & (combined_outcomes["date"] >= reg_date)
        & (combined_outcomes["date"] < reg_date + pd.Timedelta(days=EARLY_WINDOW))
    ]
    if sub.empty:
        return np.nan, 0
    if ycol == "log_spend_safe":
        return float(sub[ycol].sum()), len(sub)   # cumulative early spend (log-summed)
    else:
        return float(sub[ycol].mean()), len(sub)  # average early portfolio breadth


match_rows = []
for _, bt in born_treated.iterrows():
    cid_bt, reg_bt = bt["customer_id"], bt["registration_date"]
    candidates = never_treated_reg[
        (never_treated_reg["registration_date"] >= reg_bt - pd.Timedelta(days=REG_WINDOW))
        & (never_treated_reg["registration_date"] <= reg_bt + pd.Timedelta(days=REG_WINDOW))
    ]
    n_matches = len(candidates)

    row = {"customer_id": cid_bt, "registration_date": reg_bt, "n_matched_controls": n_matches}
    for ycol in OUTCOME_VARS:
        bt_val, bt_n = early_outcome(cid_bt, reg_bt, ycol)
        row[f"{ycol}_born_treated"] = bt_val
        row[f"{ycol}_n_obs"] = bt_n
        if n_matches > 0:
            ctrl_vals = [early_outcome(c, reg_bt, ycol)[0] for c in candidates["customer_id"]]
            ctrl_vals = [v for v in ctrl_vals if pd.notna(v)]
            row[f"{ycol}_control_mean"] = float(np.mean(ctrl_vals)) if ctrl_vals else np.nan
            row[f"{ycol}_control_n"] = len(ctrl_vals)
        else:
            row[f"{ycol}_control_mean"] = np.nan
            row[f"{ycol}_control_n"] = 0
    match_rows.append(row)

match_df = pd.DataFrame(match_rows)
match_path = STEP2_DIR / f"step3d_track2_born_treated_comparison_type{TARGET_TYPE}.csv"
match_df.to_csv(match_path, index=False)
print(f"\nMatched comparison table saved: {match_path}")
print(f"Born-treated customers with zero matched controls: {(match_df['n_matched_controls']==0).sum()}")

track2_summary = {}
for ycol in OUTCOME_VARS:
    bt_vals = match_df[f"{ycol}_born_treated"].dropna()
    ctrl_vals = match_df[f"{ycol}_control_mean"].dropna()
    print(f"\n--- Track 2 outcome: {ycol} (early {EARLY_WINDOW}-day outcome post-registration) ---")
    print(f"  born-treated: n={len(bt_vals)}, mean={bt_vals.mean():.4f}" if len(bt_vals) else "  no born-treated observations")
    print(f"  matched-control mean distribution: n={len(ctrl_vals)}, mean={ctrl_vals.mean():.4f}" if len(ctrl_vals) else "  no control observations")

    if len(bt_vals) >= 3 and len(ctrl_vals) >= 3:
        tstat, pval = sps.ttest_ind(bt_vals, ctrl_vals, equal_var=False)  # Welch's t-test
        pooled_std = np.sqrt((bt_vals.var(ddof=1) + ctrl_vals.var(ddof=1)) / 2)
        cohens_d = (bt_vals.mean() - ctrl_vals.mean()) / pooled_std if pooled_std > 0 else np.nan
        print(f"  Welch t-test: t={tstat:.3f}, p={pval:.4f}, Cohen's d={cohens_d:.3f}")
        track2_summary[ycol] = {
            "n_born_treated": int(len(bt_vals)), "n_control": int(len(ctrl_vals)),
            "mean_born_treated": float(bt_vals.mean()), "mean_control": float(ctrl_vals.mean()),
            "welch_tstat": float(tstat), "welch_pvalue": float(pval), "cohens_d": float(cohens_d),
        }
    else:
        print("  insufficient sample -- test skipped")
        track2_summary[ycol] = {"status": "insufficient_sample"}

track2_summary_path = STEP2_DIR / f"step3d_track2_summary_type{TARGET_TYPE}.json"
with open(track2_summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "n_born_treated": len(born_treated),
        "reg_match_window_days": REG_WINDOW,
        "early_outcome_window_days": EARLY_WINDOW,
        "results": track2_summary,
        "caveat": (
            "This comparison is descriptive, not a causal estimate. The choice for a new customer to "
            "start with the target campaign type is not random (e.g., particular industries or budget "
            "sizes may prefer it), so selection bias is not controlled for. Report as an association, "
            "not a causal effect."
        ),
    }, f, ensure_ascii=False, indent=2, default=str)

print(f"\nTrack 2 summary saved: {track2_summary_path}")
print("\nNext step: step3e_track1_loo_track2_robustness.py")
