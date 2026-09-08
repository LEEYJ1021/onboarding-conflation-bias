"""
================================================================================
Step 3 — Main Staggered-Adoption DiD Estimation (Callaway & Sant'Anna style)
================================================================================
Methodology summary (uses Step 2.2 / 2.3 outputs as input):
    - For each primary_adopter customer i (adoption date g_i), construct a
      "stack" centered on the adoption date spanning +/-EVENT_WINDOW days.
      The stack contains (a) the adopter i and (b) all never-treated
      customers, who are assigned the same g_i as a placebo "adoption date"
      so both groups share the same event-time axis.
    - Stack-level 2x2 DiD: ATT_i = (i's post mean - i's pre mean)
                                  - (all never-treated's post-mean change
                                     - pre-mean change, same calendar window)
    - Overall summary ATT = simple average of ATT_i across i (since most
      cohorts contain a single adopter, the simple average essentially
      equals a cohort-size-weighted average).
    - Dynamic ATT(e): at each event time e, average (Y_i(e) - Y_i(pre mean))
      minus the corresponding never-treated change, over customers with an
      observation at e.
    - This "stacked-regression DiD" avoids the negative-weighting bias of
      standard TWFE while using only never-treated controls — in spirit this
      aligns with the identification strategy of Callaway & Sant'Anna
      (2021), though it is not numerically identical to the formal
      ATT(g,t) estimator implemented in the R `did` / Python `csdid`
      packages. This distinction must be stated explicitly in the methods
      section.
    - Uncertainty: paired customer-clustered bootstrap (resample both
      treated and control groups with replacement and recompute ATT B
      times) — distinct from the "wild bootstrap" mentioned in the original
      roadmap; this distinction is reported explicitly.
    - PRIMARY (32) and ROBUST (35, including CAUTION) are estimated side by
      side to re-confirm the robustness of the Step-2.2 recovery procedure.

Input:  customer_day_panel.csv, customer_day_campaign_type_panel.csv,
        df_analysis_master.csv, staggered_adoption_FINAL_type{T}.csv
Output: step3_att_summary_type{T}.json
        step3_dynamic_att_{tag}_type{T}_{outcome}.csv
        step3_dynamic_att_plot_{tag}_type{T}_{outcome}.png
        step3_stack_level_att_{tag}_type{T}_{outcome}.csv (diagnostic)
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
from scipy import stats as sps

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))

TARGET_TYPE = int(os.environ.get("TARGET_TYPE", "6"))
EVENT_WINDOW = int(os.environ.get("EVENT_WINDOW_DAYS", "30"))
BOOTSTRAP_REPS = int(os.environ.get("BOOTSTRAP_REPS", "1000"))
SEED = int(os.environ.get("BOOTSTRAP_SEED", "20260908"))
rng = np.random.default_rng(SEED)

PANEL_PATH = ROOT / "customer_day_panel.csv"
CTP_PATH = ROOT / "customer_day_campaign_type_panel.csv"
MASTER_PATH = ROOT / "df_analysis_master.csv"
FINAL_COHORT_PATH = STEP2_DIR / f"staggered_adoption_FINAL_type{TARGET_TYPE}.csv"

for p in [PANEL_PATH, CTP_PATH, MASTER_PATH, FINAL_COHORT_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p} — run Step 2.2 first.")

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
# 0. Rebuild the extended outcome panel (matches Step 2.2's definition)
# ------------------------------------------------------------------
print("=" * 70)
print("0. Rebuilding the extended outcome panel")
print("=" * 70)

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
print(f"Combined outcome panel: {len(combined_outcomes):,} rows / {combined_outcomes['customer_id'].nunique()} customers")

OUTCOME_VARS = ["log_spend_safe", "n_campaign_types_active"]

# ------------------------------------------------------------------
# 1. Cohort/control definitions
# ------------------------------------------------------------------
never_treated_ids = final_cohort.loc[
    final_cohort["final_cohort"] == "never_treated", "customer_id"
].unique().tolist()

primary_adopters = final_cohort.loc[
    final_cohort["final_cohort"] == "primary_adopter", ["customer_id", "final_first_treated_date"]
].drop_duplicates()
robust_adopters = final_cohort.loc[
    final_cohort["final_cohort"].isin(["primary_adopter", "robustness_only_adopter"]),
    ["customer_id", "final_first_treated_date"]
].drop_duplicates()

print(f"\nnever_treated controls: {len(never_treated_ids)}")
print(f"primary_adopter (PRIMARY): {len(primary_adopters)}")
print(f"primary+robustness_only (ROBUST): {len(robust_adopters)}")

control_panel = combined_outcomes[combined_outcomes["customer_id"].isin(never_treated_ids)].copy()


def get_control_values(dates: pd.Series, ycol: str, control_ids=None) -> pd.DataFrame:
    ids = control_ids if control_ids is not None else never_treated_ids
    sub = control_panel[control_panel["date"].isin(dates) & control_panel["customer_id"].isin(ids)]
    return sub[["customer_id", "date", ycol]].dropna(subset=[ycol])


# ------------------------------------------------------------------
# 2. Stack-level 2x2 DiD and dynamic ATT computation
# ------------------------------------------------------------------
def compute_att(adopters: pd.DataFrame, control_ids: list, ycol: str):
    treated_ids = adopters["customer_id"].tolist()
    g_map = adopters.set_index("customer_id")["final_first_treated_date"].to_dict()

    stack_rows = []
    dyn_treated = {}
    dyn_control = {}

    for cid, g in g_map.items():
        treated_series = combined_outcomes[
            (combined_outcomes["customer_id"] == cid)
        ][["date", ycol]].dropna(subset=[ycol]).copy()
        treated_series["event_time"] = (treated_series["date"] - g).dt.days
        treated_series = treated_series[treated_series["event_time"].between(-EVENT_WINDOW, EVENT_WINDOW)]

        pre_t = treated_series[treated_series["event_time"] < 0]
        post_t = treated_series[treated_series["event_time"] >= 0]
        if pre_t.empty or post_t.empty:
            continue
        treat_pre_mean = pre_t[ycol].mean()
        treat_post_mean = post_t[ycol].mean()

        window_dates = pd.date_range(g - pd.Timedelta(days=EVENT_WINDOW), g + pd.Timedelta(days=EVENT_WINDOW))
        ctrl = get_control_values(window_dates, ycol, control_ids)
        if ctrl.empty:
            continue
        ctrl = ctrl.copy()
        ctrl["event_time"] = (ctrl["date"] - g).dt.days

        ctrl_pre = ctrl[ctrl["event_time"] < 0]
        ctrl_post = ctrl[ctrl["event_time"] >= 0]
        if ctrl_pre.empty or ctrl_post.empty:
            continue
        ctrl_pre_mean = ctrl_pre.groupby("customer_id")[ycol].mean().mean()
        ctrl_post_mean = ctrl_post.groupby("customer_id")[ycol].mean().mean()

        att_i = (treat_post_mean - treat_pre_mean) - (ctrl_post_mean - ctrl_pre_mean)
        stack_rows.append({
            "customer_id": cid, "first_treated_date": g,
            "treat_pre_mean": treat_pre_mean, "treat_post_mean": treat_post_mean,
            "control_pre_mean": ctrl_pre_mean, "control_post_mean": ctrl_post_mean,
            "att_i": att_i,
        })

        ctrl_by_et = ctrl.groupby("event_time")[ycol].apply(
            lambda s: s.groupby(ctrl.loc[s.index, "customer_id"]).mean().mean()
        )
        for et, y_val in treated_series.set_index("event_time")[ycol].items():
            if et not in ctrl_by_et.index:
                continue
            treated_delta = y_val - treat_pre_mean
            control_delta = ctrl_by_et.loc[et] - ctrl_pre_mean
            dyn_treated.setdefault(et, []).append(treated_delta)
            dyn_control.setdefault(et, []).append(control_delta)

    stack_level = pd.DataFrame(stack_rows)
    summary_att = float(stack_level["att_i"].mean()) if not stack_level.empty else np.nan

    dyn_rows = []
    for et in sorted(set(dyn_treated) & set(dyn_control)):
        td = np.mean(dyn_treated[et])
        cd = np.mean(dyn_control[et])
        dyn_rows.append({
            "event_time": et, "att": td - cd, "n_contributing_customers": len(dyn_treated[et]),
        })
    dynamic = pd.DataFrame(dyn_rows).sort_values("event_time").reset_index(drop=True)

    return stack_level, dynamic, summary_att


def bootstrap_ci(adopters: pd.DataFrame, control_ids: list, ycol: str, n_reps: int, seed_rng):
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


# ------------------------------------------------------------------
# 3. Run PRIMARY / ROBUST estimation
# ------------------------------------------------------------------
COHORT_DEFS = {"PRIMARY": primary_adopters, "ROBUST": robust_adopters}
all_summary = {}

for tag, adopters in COHORT_DEFS.items():
    print("\n" + "=" * 70)
    print(f"[{tag}] Step 3 main estimation (n_adopters={len(adopters)})")
    print("=" * 70)

    tag_summary = {}
    for ycol in OUTCOME_VARS:
        print(f"\n--- outcome: {ycol} ---")
        stack_level, dynamic, summary_att = compute_att(adopters, never_treated_ids, ycol)
        if stack_level.empty:
            print("  No valid stacks — cannot estimate, skipping")
            continue

        print(f"  Valid stacks (customers): {len(stack_level)}")
        print(f"  Overall summary ATT = {summary_att:+.4f}")

        se, ci_lo, ci_hi, boot_dist = bootstrap_ci(
            adopters, never_treated_ids, ycol, BOOTSTRAP_REPS, rng
        )
        z = summary_att / se if (se and se > 0) else np.nan
        pval = float(2 * (1 - sps.norm.cdf(abs(z)))) if pd.notna(z) else np.nan
        print(f"  Clustered bootstrap (B={BOOTSTRAP_REPS}) SE={se:.4f}, "
              f"95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], approx p={pval:.4f}")

        stack_path = STEP2_DIR / f"step3_stack_level_att_{tag}_type{TARGET_TYPE}_{ycol}.csv"
        stack_level.to_csv(stack_path, index=False)
        dyn_path = STEP2_DIR / f"step3_dynamic_att_{tag}_type{TARGET_TYPE}_{ycol}.csv"
        dynamic.to_csv(dyn_path, index=False)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(dynamic["event_time"], dynamic["att"], marker="o", markersize=3, color="darkred")
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.axvline(0, color="firebrick", linestyle="--", linewidth=1, label="Adoption date")
        ax.set_title(f"Type {TARGET_TYPE} ({tag}) dynamic ATT: {ycol}\n"
                      f"Summary ATT={summary_att:+.4f} (bootstrap SE={se:.4f}, 95%CI=[{ci_lo:+.3f},{ci_hi:+.3f}])")
        ax.set_xlabel("Event time (days from adoption)")
        ax.set_ylabel(f"ATT({ycol})")
        ax.legend()
        plt.tight_layout()
        plot_path = STEP2_DIR / f"step3_dynamic_att_plot_{tag}_type{TARGET_TYPE}_{ycol}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved: {stack_path.name}, {dyn_path.name}, {plot_path.name}")

        tag_summary[ycol] = {
            "n_valid_stacks": int(len(stack_level)),
            "summary_att": summary_att,
            "bootstrap_se": se, "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
            "approx_pvalue": pval, "significant_at_05": bool(pval < 0.05) if pd.notna(pval) else None,
        }

    all_summary[tag] = tag_summary

# ------------------------------------------------------------------
# 4. Compare PRIMARY vs ROBUST + save final summary
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. Comparing PRIMARY vs ROBUST")
print("=" * 70)
for ycol in OUTCOME_VARS:
    p = all_summary.get("PRIMARY", {}).get(ycol, {})
    r = all_summary.get("ROBUST", {}).get(ycol, {})
    if p and r:
        same_sign = (np.sign(p["summary_att"]) == np.sign(r["summary_att"]))
        print(f"[{ycol}] PRIMARY ATT={p['summary_att']:+.4f}(p={p['approx_pvalue']:.3f}) vs "
              f"ROBUST ATT={r['summary_att']:+.4f}(p={r['approx_pvalue']:.3f}) "
              f"-> sign agreement: {'yes' if same_sign else 'no'}")

summary_path = STEP2_DIR / f"step3_att_summary_type{TARGET_TYPE}.json"
final_report = {
    "target_type": TARGET_TYPE,
    "event_window_days": EVENT_WINDOW,
    "bootstrap_reps": BOOTSTRAP_REPS,
    "bootstrap_seed": SEED,
    "method_note": (
        "Approximates the Callaway & Sant'Anna (2021) identification strategy "
        "(never-treated controls, per-cohort ATT(g,t) then aggregation) via a "
        "stacked cohort-level 2x2 DiD; not numerically identical to the formal "
        "ATT(g,t) estimator. Uncertainty comes from a paired customer-clustered "
        "bootstrap, distinct from the wild bootstrap mentioned in the original "
        "roadmap — this must be stated in the methods section."
    ),
    "n_never_treated": len(never_treated_ids),
    "n_primary_adopters": len(primary_adopters),
    "n_robust_adopters": len(robust_adopters),
    "results": all_summary,
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)

print(f"\nFinal summary saved: {summary_path}")
print("\nNext step: run step3b_stack_dropout_diagnosis.py to diagnose why so many")
print("adopters dropped out of the valid-stack set.")
