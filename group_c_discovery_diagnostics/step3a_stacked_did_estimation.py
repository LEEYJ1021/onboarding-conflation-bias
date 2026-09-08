"""
================================================================================
Step 3a — Main DiD estimation (stacked cohort-level 2x2 design)
================================================================================
For each primary_adopter customer i (adoption date g_i), builds a
"stack" of (a) customer i and (b) all never-treated customers, assigning
never-treated customers the pseudo-adoption date g_i so both share the
same event-time axis. Each stack's 2x2 DiD is:

    ATT_i = (post_i - pre_i) - (post_never_treated(g_i) - pre_never_treated(g_i))

The pooled ATT is the simple average of ATT_i across adopters. Dynamic
ATT(e) averages, at each event time e, the treated deviation from its
own pre-period mean minus the matched never-treated deviation.

This "stacked-regression DiD" avoids the negative-weighting problem of
naive two-way fixed effects while using only never-treated controls,
in the same spirit as Callaway & Sant'Anna (2021) — but it is NOT
numerically identical to the ATT(g,t) estimator from the `did`/`csdid`
packages; this must be stated explicitly in the paper's methods section.
Uncertainty is estimated by a customer-level paired cluster bootstrap
(distinct from the "wild bootstrap" sometimes used for this design).

Runs PRIMARY (32 adopters) and ROBUST (35, including CAUTION recoveries)
side by side to re-check step2c's robustness conclusion.

Inputs:  customer_day_panel.csv, customer_day_campaign_type_panel.csv,
         df_analysis_master.csv, staggered_adoption_FINAL_type{T}.csv
Outputs: step3_att_summary_type{T}.json
         step3_dynamic_att_{tag}_type{T}_{outcome}.csv
         step3_dynamic_att_plot_{tag}_type{T}_{outcome}.png
         step3_stack_level_att_{tag}_type{T}_{outcome}.csv
================================================================================
"""
import json

import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PANEL_PATH, CTP_PATH, MASTER_PATH, STEP2_OUT, TARGET_TYPE, detect_date_column

EVENT_WINDOW = 30
BOOTSTRAP_REPS = 1000
SEED = 20260908
rng = np.random.default_rng(SEED)

FINAL_COHORT_PATH = STEP2_OUT / f"staggered_adoption_FINAL_type{TARGET_TYPE}.csv"
for p in (PANEL_PATH, CTP_PATH, MASTER_PATH, FINAL_COHORT_PATH):
    if not p.exists():
        raise FileNotFoundError(f"Missing input: {p} — run step2b first.")

panel = pd.read_csv(PANEL_PATH)
ctp = pd.read_csv(CTP_PATH)
master = pd.read_csv(MASTER_PATH)
final_cohort = pd.read_csv(FINAL_COHORT_PATH)

DATE_COL = detect_date_column(panel)
DATE_COL_CTP = detect_date_column(ctp)
panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])
ctp[DATE_COL_CTP] = pd.to_datetime(ctp[DATE_COL_CTP])
master["date"] = pd.to_datetime(master["date"])
final_cohort["final_first_treated_date"] = pd.to_datetime(final_cohort["final_first_treated_date"])

print("=" * 70)
print("0. Rebuilding the extended outcome panel")
print("=" * 70)
panel_ext = panel[["customer_id", DATE_COL, "cost"]].copy()
panel_ext["log_spend_safe"] = np.log1p(panel_ext["cost"])
panel_ext = panel_ext.rename(columns={DATE_COL: "date"})
ctp["campaign_type"] = pd.to_numeric(ctp["campaign_type"], errors="coerce")
ctp_active = ctp[ctp["cost"] > 0].copy()
n_types_active = (ctp_active.groupby(["customer_id", DATE_COL_CTP])["campaign_type"].nunique()
                   .rename("n_campaign_types_active").reset_index().rename(columns={DATE_COL_CTP: "date"}))
panel_ext = panel_ext.merge(n_types_active, on=["customer_id", "date"], how="left")
panel_ext["n_campaign_types_active"] = panel_ext["n_campaign_types_active"].fillna(0).astype(int)
master_outcomes = master[["customer_id", "date", "log_spend_safe", "n_campaign_types_active"]].copy()
master_outcomes["_source"] = "stable_window(master)"
panel_ext_labeled = panel_ext[["customer_id", "date", "log_spend_safe", "n_campaign_types_active"]].copy()
panel_ext_labeled["_source"] = "raw_panel(pre_or_post_window)"
combined_outcomes = pd.concat([master_outcomes, panel_ext_labeled], ignore_index=True)
combined_outcomes = (combined_outcomes.sort_values(
        "_source", key=lambda s: s.map({"stable_window(master)": 0, "raw_panel(pre_or_post_window)": 1}))
    .drop_duplicates(subset=["customer_id", "date"], keep="first").drop(columns="_source"))
print(f"Combined outcome panel: {len(combined_outcomes):,} rows / {combined_outcomes['customer_id'].nunique()} customers")

OUTCOME_VARS = ["log_spend_safe", "n_campaign_types_active"]

never_treated_ids = final_cohort.loc[final_cohort["final_cohort"] == "never_treated", "customer_id"].unique().tolist()
primary_adopters = final_cohort.loc[final_cohort["final_cohort"] == "primary_adopter",
                                     ["customer_id", "final_first_treated_date"]].drop_duplicates()
robust_adopters = final_cohort.loc[final_cohort["final_cohort"].isin(["primary_adopter", "robustness_only_adopter"]),
                                    ["customer_id", "final_first_treated_date"]].drop_duplicates()
print(f"\nNever-treated controls: {len(never_treated_ids)}")
print(f"primary_adopter (PRIMARY): {len(primary_adopters)}")
print(f"primary + robustness_only (ROBUST): {len(robust_adopters)}")

control_panel = combined_outcomes[combined_outcomes["customer_id"].isin(never_treated_ids)].copy()


def get_control_values(dates, ycol, control_ids=None):
    ids = control_ids if control_ids is not None else never_treated_ids
    sub = control_panel[control_panel["date"].isin(dates) & control_panel["customer_id"].isin(ids)]
    return sub[["customer_id", "date", ycol]].dropna(subset=[ycol])


def compute_att(adopters, control_ids, ycol):
    g_map = adopters.set_index("customer_id")["final_first_treated_date"].to_dict()
    stack_rows = []
    dyn_treated, dyn_control = {}, {}
    for cid, g in g_map.items():
        treated_series = combined_outcomes[combined_outcomes["customer_id"] == cid][["date", ycol]].dropna(subset=[ycol]).copy()
        treated_series["event_time"] = (treated_series["date"] - g).dt.days
        treated_series = treated_series[treated_series["event_time"].between(-EVENT_WINDOW, EVENT_WINDOW)]
        pre_t, post_t = treated_series[treated_series["event_time"] < 0], treated_series[treated_series["event_time"] >= 0]
        if pre_t.empty or post_t.empty:
            continue
        treat_pre_mean, treat_post_mean = pre_t[ycol].mean(), post_t[ycol].mean()
        window_dates = pd.date_range(g - pd.Timedelta(days=EVENT_WINDOW), g + pd.Timedelta(days=EVENT_WINDOW))
        ctrl = get_control_values(window_dates, ycol, control_ids)
        if ctrl.empty:
            continue
        ctrl = ctrl.copy()
        ctrl["event_time"] = (ctrl["date"] - g).dt.days
        ctrl_pre, ctrl_post = ctrl[ctrl["event_time"] < 0], ctrl[ctrl["event_time"] >= 0]
        if ctrl_pre.empty or ctrl_post.empty:
            continue
        ctrl_pre_mean = ctrl_pre.groupby("customer_id")[ycol].mean().mean()
        ctrl_post_mean = ctrl_post.groupby("customer_id")[ycol].mean().mean()
        att_i = (treat_post_mean - treat_pre_mean) - (ctrl_post_mean - ctrl_pre_mean)
        stack_rows.append({"customer_id": cid, "first_treated_date": g, "treat_pre_mean": treat_pre_mean,
                            "treat_post_mean": treat_post_mean, "control_pre_mean": ctrl_pre_mean,
                            "control_post_mean": ctrl_post_mean, "att_i": att_i})
        ctrl_by_et = ctrl.groupby("event_time")[ycol].apply(
            lambda s: s.groupby(ctrl.loc[s.index, "customer_id"]).mean().mean())
        for et, y_val in treated_series.set_index("event_time")[ycol].items():
            if et not in ctrl_by_et.index:
                continue
            dyn_treated.setdefault(et, []).append(y_val - treat_pre_mean)
            dyn_control.setdefault(et, []).append(ctrl_by_et.loc[et] - ctrl_pre_mean)
    stack_level = pd.DataFrame(stack_rows)
    summary_att = float(stack_level["att_i"].mean()) if not stack_level.empty else np.nan
    dyn_rows = []
    for et in sorted(set(dyn_treated) & set(dyn_control)):
        td, cd = np.mean(dyn_treated[et]), np.mean(dyn_control[et])
        dyn_rows.append({"event_time": et, "att": td - cd, "n_contributing_customers": len(dyn_treated[et])})
    dynamic = pd.DataFrame(dyn_rows).sort_values("event_time").reset_index(drop=True)
    return stack_level, dynamic, summary_att


def bootstrap_ci(adopters, control_ids, ycol, n_reps, seed_rng):
    treated_ids_arr = adopters["customer_id"].to_numpy()
    control_ids_arr = np.array(control_ids)
    boot_atts = []
    for _ in range(n_reps):
        resampled_treated_ids = seed_rng.choice(treated_ids_arr, size=len(treated_ids_arr), replace=True)
        resampled_control_ids = seed_rng.choice(control_ids_arr, size=len(control_ids_arr), replace=True).tolist()
        resampled_adopters = adopters.set_index("customer_id").loc[resampled_treated_ids].reset_index()
        resampled_adopters = resampled_adopters.rename(columns={"index": "customer_id"})
        _, _, att_b = compute_att(resampled_adopters, resampled_control_ids, ycol)
        if pd.notna(att_b):
            boot_atts.append(att_b)
    boot_atts = np.array(boot_atts)
    se = float(boot_atts.std(ddof=1)) if len(boot_atts) > 1 else np.nan
    ci_lo, ci_hi = (np.percentile(boot_atts, [2.5, 97.5]) if len(boot_atts) > 1 else (np.nan, np.nan))
    return se, float(ci_lo), float(ci_hi), boot_atts


COHORT_DEFS = {"PRIMARY": primary_adopters, "ROBUST": robust_adopters}
all_summary = {}
for tag, adopters in COHORT_DEFS.items():
    print("\n" + "=" * 70)
    print(f"[{tag}] main DiD estimation (n_adopters={len(adopters)})")
    print("=" * 70)
    tag_summary = {}
    for ycol in OUTCOME_VARS:
        print(f"\n--- outcome: {ycol} ---")
        stack_level, dynamic, summary_att = compute_att(adopters, never_treated_ids, ycol)
        if stack_level.empty:
            print("  No valid stacks — skipping")
            continue
        print(f"  Valid stacks (customers): {len(stack_level)}")
        print(f"  Pooled ATT = {summary_att:+.4f}")
        se, ci_lo, ci_hi, _ = bootstrap_ci(adopters, never_treated_ids, ycol, BOOTSTRAP_REPS, rng)
        z = summary_att / se if (se and se > 0) else np.nan
        pval = float(2 * (1 - sps.norm.cdf(abs(z)))) if pd.notna(z) else np.nan
        print(f"  Cluster bootstrap (B={BOOTSTRAP_REPS}) SE={se:.4f}, 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], approx p={pval:.4f}")

        stack_path = STEP2_OUT / f"step3_stack_level_att_{tag}_type{TARGET_TYPE}_{ycol}.csv"
        stack_level.to_csv(stack_path, index=False)
        dyn_path = STEP2_OUT / f"step3_dynamic_att_{tag}_type{TARGET_TYPE}_{ycol}.csv"
        dynamic.to_csv(dyn_path, index=False)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(dynamic["event_time"], dynamic["att"], marker="o", markersize=3, color="darkred")
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.axvline(0, color="firebrick", linestyle="--", linewidth=1, label="Adoption date")
        ax.set_title(f"Type {TARGET_TYPE} ({tag}) dynamic ATT: {ycol}\n"
                      f"Pooled ATT={summary_att:+.4f} (bootstrap SE={se:.4f}, 95% CI=[{ci_lo:+.3f},{ci_hi:+.3f}])")
        ax.set_xlabel("Event time (days from adoption)")
        ax.set_ylabel(f"ATT({ycol})")
        ax.legend()
        plt.tight_layout()
        plot_path = STEP2_OUT / f"step3_dynamic_att_plot_{tag}_type{TARGET_TYPE}_{ycol}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved: {stack_path.name}, {dyn_path.name}, {plot_path.name}")
        tag_summary[ycol] = {"n_valid_stacks": int(len(stack_level)), "summary_att": summary_att,
                              "bootstrap_se": se, "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
                              "approx_pvalue": pval, "significant_at_05": bool(pval < 0.05) if pd.notna(pval) else None}
    all_summary[tag] = tag_summary

print("\n" + "=" * 70)
print("4. PRIMARY vs ROBUST comparison")
print("=" * 70)
for ycol in OUTCOME_VARS:
    p = all_summary.get("PRIMARY", {}).get(ycol, {})
    r = all_summary.get("ROBUST", {}).get(ycol, {})
    if p and r:
        same_sign = np.sign(p["summary_att"]) == np.sign(r["summary_att"])
        print(f"[{ycol}] PRIMARY ATT={p['summary_att']:+.4f}(p={p['approx_pvalue']:.3f}) vs "
              f"ROBUST ATT={r['summary_att']:+.4f}(p={r['approx_pvalue']:.3f}) -> sign match: {'YES' if same_sign else 'NO'}")

summary_path = STEP2_OUT / f"step3_att_summary_type{TARGET_TYPE}.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "target_type": TARGET_TYPE, "event_window_days": EVENT_WINDOW,
        "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": SEED,
        "method_note": ("Approximates Callaway & Sant'Anna (2021)'s never-treated-control, "
                         "cohort-then-aggregate identification strategy via a stacked cohort-level "
                         "2x2 DiD. Not numerically identical to the formal ATT(g,t) estimator. "
                         "Uncertainty via customer-level paired cluster bootstrap, distinct from "
                         "a wild bootstrap — both caveats must be stated in the methods section."),
        "n_never_treated": len(never_treated_ids), "n_primary_adopters": len(primary_adopters),
        "n_robust_adopters": len(robust_adopters), "results": all_summary,
    }, f, ensure_ascii=False, indent=2, default=str)
print(f"\nFinal summary saved: {summary_path}")
print("Next step: step3b_stack_dropout_diagnosis.py (why did most of the 32/35 adopters produce no valid stack?)")
