"""
================================================================================
Step 3f -- Track 1 (True Switchers) Alternative Gap-Threshold Re-Estimation
================================================================================
Background:
    Step 3c/3d used BORN_TREATED_GAP_THRESHOLD=1 day, yielding 7 true
    switchers for Type 6. The gap-threshold sensitivity analysis (Step 2f)
    shows that raising the threshold to 3 days shrinks the switcher count
    from 7 to 5 -- and this shrinkage is not arbitrary: it removes exactly
    the two customers flagged by Step 3e's leave-one-out check as having
    outsized influence (customer_id=1113201, gap=2 days, only 2 pre-period
    days; customer_id=2196844, gap=3 days, only 3 pre-period days).

    This script formally re-estimates Track 1 under the alternative
    threshold=3 specification and compares its case-level ATTs and
    leave-one-out stability against the original threshold=1 results,
    testing whether the alternative specification converges on a more
    stable, less outlier-driven answer.

    Decision rule: if the n=5 (threshold=3) leave-one-out sign-flip count
    is lower than the n=7 (threshold=1) count, that supports interpreting
    the threshold=1 instability as driven by a couple of extreme cases. If
    the alternative specification remains equally unstable, that instead
    supports the conclusion that Track 1's limitation is fundamentally one
    of sample size, not particular outliers.

Input:  customer_day_panel.csv, customer_day_campaign_type_panel.csv,
        df_analysis_master.csv, staggered_adoption_FINAL_type{T}.csv,
        step3c_born_treated_diagnosis_type{T}.csv,
        step3d_track1_switcher_att_type{T}.json (for reference, optional)
Output: step3f_track1_switcher_cases_gap{ALT}_type{T}.csv
        step3f_track1_leave_one_out_gap{ALT}_type{T}.csv
        step3f_track1_threshold_comparison_type{T}.json
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
ALT_GAP_THRESHOLD = int(os.environ.get("ALT_GAP_THRESHOLD_DAYS", "3"))
ORIG_GAP_THRESHOLD = int(os.environ.get("ORIG_GAP_THRESHOLD_DAYS", "1"))

PANEL_PATH = ROOT / "customer_day_panel.csv"
CTP_PATH = ROOT / "customer_day_campaign_type_panel.csv"
MASTER_PATH = ROOT / "df_analysis_master.csv"
FINAL_COHORT_PATH = STEP2_DIR / f"staggered_adoption_FINAL_type{TARGET_TYPE}.csv"
BORN_DIAG_PATH = STEP2_DIR / f"step3c_born_treated_diagnosis_type{TARGET_TYPE}.csv"
ORIG_TRACK1_PATH = STEP2_DIR / f"step3d_track1_switcher_att_type{TARGET_TYPE}.json"

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

OUTCOME_VARS = ["log_spend_safe", "n_campaign_types_active"]
never_treated_ids = final_cohort.loc[
    final_cohort["final_cohort"] == "never_treated", "customer_id"
].unique().tolist()
control_panel = combined_outcomes[combined_outcomes["customer_id"].isin(never_treated_ids)].copy()


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


def run_track1(gap_threshold: int, tag: str):
    switchers = born_diag[born_diag["gap_days"] > gap_threshold][
        ["customer_id", "final_first_treated_date", "final_cohort", "gap_days",
         "raw_panel_date_min", "n_pre_treated_days"]
    ].copy()
    print(f"\n[{tag}] gap_threshold={gap_threshold} days -> switchers: {len(switchers)}")
    if not switchers.empty:
        print(switchers.to_string(index=False))

    summary = {}
    cases_all = []
    loo_all = []
    for ycol in OUTCOME_VARS:
        cases = compute_att_case_level(switchers, never_treated_ids, ycol)
        if cases.empty:
            print(f"  [{tag}][{ycol}] no valid cases")
            summary[ycol] = {"n_valid": 0}
            continue
        cases["outcome"] = ycol
        cases_all.append(cases)

        n_valid = len(cases)
        mean_att = float(cases["att_i"].mean())
        print(f"  [{tag}][{ycol}] valid n={n_valid}, simple-average ATT={mean_att:+.4f}")

        entry = {"n_valid": n_valid, "mean_att": mean_att}
        if n_valid >= 3:
            se = cases["att_i"].std(ddof=1) / np.sqrt(n_valid)
            tcrit = sps.t.ppf(0.975, df=n_valid - 1) if n_valid > 1 else np.nan
            ci_lo, ci_hi = mean_att - tcrit * se, mean_att + tcrit * se
            tstat = mean_att / se if se > 0 else np.nan
            pval = float(2 * (1 - sps.t.cdf(abs(tstat), df=n_valid - 1))) if pd.notna(tstat) and n_valid > 1 else np.nan
            entry.update({"se": float(se), "ci_95_lo": float(ci_lo), "ci_95_hi": float(ci_hi), "pvalue": pval})
            print(f"    t-distribution 95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}], p={pval:.4f}")
        else:
            entry["note"] = "n<3, formal CI omitted"

        loo_rows = []
        for _, drop_row in cases.iterrows():
            remaining = cases[cases["customer_id"] != drop_row["customer_id"]]
            loo_mean = remaining["att_i"].mean() if not remaining.empty else np.nan
            loo_rows.append({
                "outcome": ycol, "dropped_customer_id": drop_row["customer_id"],
                "dropped_att_i": drop_row["att_i"], "loo_mean_att": loo_mean,
                "shift_from_full_mean": (loo_mean - mean_att) if pd.notna(loo_mean) else np.nan,
            })
        loo_df = pd.DataFrame(loo_rows)
        loo_all.append(loo_df)
        if not loo_df.empty:
            sign_changes = int((np.sign(loo_df["loo_mean_att"]) != np.sign(mean_att)).sum())
            entry["loo_sign_changes"] = sign_changes
            print(f"    LOO sign changes: {sign_changes}/{len(loo_df)}")

        summary[ycol] = entry

    cases_df = pd.concat(cases_all, ignore_index=True) if cases_all else pd.DataFrame()
    loo_df_all = pd.concat(loo_all, ignore_index=True) if loo_all else pd.DataFrame()
    return switchers, cases_df, loo_df_all, summary


# ============================================================================
# 1. Run original (threshold=1) and alternative (threshold=3) specifications
# ============================================================================
print("=" * 70)
print(f"Track 1 re-estimation -- original threshold={ORIG_GAP_THRESHOLD}d vs. alternative threshold={ALT_GAP_THRESHOLD}d")
print("=" * 70)

switchers_orig, cases_orig, loo_orig, summary_orig = run_track1(ORIG_GAP_THRESHOLD, "ORIG(threshold=1)")
switchers_alt, cases_alt, loo_alt, summary_alt = run_track1(ALT_GAP_THRESHOLD, f"ALT(threshold={ALT_GAP_THRESHOLD})")

# ------------------------------------------------------------------
# 2. Did raising the threshold actually remove the "problem" cases?
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. Customers removed by raising the threshold")
print("=" * 70)
removed_ids = set(switchers_orig["customer_id"]) - set(switchers_alt["customer_id"])
print(f"Removed by threshold {ORIG_GAP_THRESHOLD}->{ALT_GAP_THRESHOLD}: {sorted(removed_ids)}")
if not loo_orig.empty:
    flagged_in_orig_loo = loo_orig.loc[
        loo_orig["shift_from_full_mean"].abs() > loo_orig.groupby("outcome")["shift_from_full_mean"].transform(
            lambda s: s.abs().median() * 2
        ),
        "dropped_customer_id"
    ].unique().tolist()
    overlap = removed_ids & set(flagged_in_orig_loo)
    print(f"Customers flagged as high-influence in the original (threshold=1) LOO: {sorted(set(flagged_in_orig_loo))}")
    print(f"-> overlap with customers removed by raising the threshold: {sorted(overlap)}")
    if overlap:
        print("  CONFIRMED: raising the threshold removed at least one of the cases actually driving instability.")

# ------------------------------------------------------------------
# 3. Save
# ------------------------------------------------------------------
if not cases_alt.empty:
    cases_alt_path = STEP2_DIR / f"step3f_track1_switcher_cases_gap{ALT_GAP_THRESHOLD}_type{TARGET_TYPE}.csv"
    cases_alt.to_csv(cases_alt_path, index=False)
    print(f"\n[ALT] case table saved: {cases_alt_path}")
if not loo_alt.empty:
    loo_alt_path = STEP2_DIR / f"step3f_track1_leave_one_out_gap{ALT_GAP_THRESHOLD}_type{TARGET_TYPE}.csv"
    loo_alt.to_csv(loo_alt_path, index=False)
    print(f"[ALT] LOO results saved: {loo_alt_path}")

orig_from_file = None
if ORIG_TRACK1_PATH.exists():
    with open(ORIG_TRACK1_PATH, "r", encoding="utf-8") as f:
        orig_from_file = json.load(f)
    print(f"\n[reference] loaded original threshold=1 results from {ORIG_TRACK1_PATH.name}")

print("\n" + "=" * 70)
print(f"3. Comparison: threshold=1 (n={len(switchers_orig)}) vs. threshold={ALT_GAP_THRESHOLD} (n={len(switchers_alt)})")
print("=" * 70)
for ycol in OUTCOME_VARS:
    o = summary_orig.get(ycol, {})
    a = summary_alt.get(ycol, {})
    o_att, a_att = o.get("mean_att"), a.get("mean_att")
    if o_att is not None and a_att is not None:
        same_sign = np.sign(o_att) == np.sign(a_att)
        print(f"[{ycol}] threshold=1: ATT={o_att:+.4f} (n={o.get('n_valid')}, "
              f"LOO sign changes={o.get('loo_sign_changes', 'NA')}) | "
              f"threshold={ALT_GAP_THRESHOLD}: ATT={a_att:+.4f} (n={a.get('n_valid')}, "
              f"LOO sign changes={a.get('loo_sign_changes', 'NA')}) "
              f"-> sign match: {'yes' if same_sign else 'no'}")

verdict = None
o_loo_changes = [x for x in (summary_orig.get(y, {}).get("loo_sign_changes") for y in OUTCOME_VARS) if x is not None]
a_loo_changes = [x for x in (summary_alt.get(y, {}).get("loo_sign_changes") for y in OUTCOME_VARS) if x is not None]
if o_loo_changes and a_loo_changes:
    if sum(a_loo_changes) < sum(o_loo_changes):
        verdict = (
            f"The alternative specification (threshold={ALT_GAP_THRESHOLD}, n={len(switchers_alt)}) shows "
            f"fewer LOO sign changes than the original (threshold=1, n={len(switchers_orig)}) -- evidence "
            f"that much of the original instability came from a couple of extreme cases (short-gap "
            f"customers with very few pre-period days). Both specifications remain small-N (n<=7), so "
            f"the paper should still avoid claiming a directional effect and retain case-level reporting."
        )
    else:
        verdict = (
            f"Switching to the alternative threshold ({ALT_GAP_THRESHOLD} days) did not reduce LOO "
            f"instability -- Track 1's instability appears to be a fundamental small-sample limitation "
            f"rather than an artifact of any particular case."
        )
    print(f"\n{verdict}")

out_json_path = STEP2_DIR / f"step3f_track1_threshold_comparison_type{TARGET_TYPE}.json"
with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "target_type": TARGET_TYPE,
        "orig_gap_threshold_days": ORIG_GAP_THRESHOLD,
        "alt_gap_threshold_days": ALT_GAP_THRESHOLD,
        "n_switchers_orig": len(switchers_orig),
        "n_switchers_alt": len(switchers_alt),
        "removed_by_threshold_increase": sorted(int(x) for x in removed_ids),
        "summary_orig": summary_orig,
        "summary_alt": summary_alt,
        "orig_result_from_step3d_file": orig_from_file,
        "verdict_message": verdict,
    }, f, ensure_ascii=False, indent=2, default=str)

print(f"\nComparison saved: {out_json_path}")
print("\nCAUTION: if a result becomes significant only under the alternative threshold and not under the")
print("original, treat it as a likely artifact of searching across specifications (multiple comparisons)")
print("rather than a genuine discovery -- report both specifications side by side and let readers judge.")
