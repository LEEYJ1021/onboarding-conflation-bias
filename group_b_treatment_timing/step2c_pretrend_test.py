"""
================================================================================
Step 2c — Formal pre-trend test
================================================================================
The event-study plot from step2b showed some wobble near the adoption
date (t ~ -15..-1). Before interpreting that as a genuine pre-trend this
script formally tests it:
    A. Customer count by event_time — flag "thin" bins (<=30% of the median)
    B. Identify which customers drive local minima/maxima in the outcome
       series (to rule out a single-customer data artifact)
    C. Cluster-robust OLS of outcome ~ event_time on the pre-period only,
       run on the full pre-period and split into a near window (anticipation,
       default t in [-15,-1]) and a far window (t < -15), for BOTH the
       PRIMARY (SAFE-only) and ROBUST (SAFE+CAUTION) cohorts, to check that
       conclusions are insensitive to the CAUTION-flagged recoveries.

Inputs:  event_time_panel_FINAL_type{T}.csv, event_time_panel_ROBUST_type{T}.csv
         (step2b outputs)
Outputs: event_time_diagnostics_type{T}.json
================================================================================
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import STEP2_OUT, TARGET_TYPE

NEAR_WINDOW_CUTOFF = -15
SIG_ALPHA = 0.05

VERSIONS = {
    "PRIMARY": STEP2_OUT / f"event_time_panel_FINAL_type{TARGET_TYPE}.csv",
    "ROBUST": STEP2_OUT / f"event_time_panel_ROBUST_type{TARGET_TYPE}.csv",
}
all_results = {}

for tag, ev_path in VERSIONS.items():
    print("=" * 70)
    print(f"[{tag}] type {TARGET_TYPE} pre-trend diagnostics ({ev_path.name})")
    print("=" * 70)
    if not ev_path.exists():
        print(f"File not found: {ev_path} — skipping")
        continue
    ev = pd.read_csv(ev_path)
    outcome_cols = [c for c in ["log_spend_safe", "n_campaign_types_active"] if c in ev.columns]

    print("\nA. Customer count by event_time")
    n_by_et = ev.groupby("event_time")["customer_id"].nunique().rename("n_customers")
    thin_threshold = max(2, n_by_et.median() * 0.3)
    thin_bins = n_by_et[n_by_et <= thin_threshold]
    print(f"   event_time range: {n_by_et.index.min()}..{n_by_et.index.max()}, "
          f"median n={n_by_et.median():.0f}, min={n_by_et.min()}, max={n_by_et.max()}")
    if len(thin_bins):
        print(f"   Thin bins (threshold {thin_threshold:.1f}, {len(thin_bins)} bins): {sorted(thin_bins.index.tolist())}")
    else:
        print("   No unusually thin bins")

    contributor_report = {}
    for ycol in outcome_cols:
        et_mean = ev.groupby("event_time")[ycol].mean()
        min_et, max_et = et_mean.idxmin(), et_mean.idxmax()
        n_at_min = int(n_by_et.loc[min_et])
        is_min_thin = min_et in thin_bins.index
        print(f"\nB. [{ycol}] min mean at event_time={min_et} (mean={et_mean.loc[min_et]:.3f}, "
              f"n_customers={n_at_min}{' THIN' if is_min_thin else ''})")
        contributors_min = ev[ev["event_time"] == min_et][["customer_id", ycol]].sort_values(ycol)
        print(contributors_min.to_string(index=False))
        print(f"   [{ycol}] max mean at event_time={max_et} (mean={et_mean.loc[max_et]:.3f}, n_customers={int(n_by_et.loc[max_et])})")
        contributor_report[ycol] = {"min_event_time": int(min_et), "min_mean": float(et_mean.loc[min_et]),
                                     "n_customers_at_min": n_at_min, "min_is_thin_bin": bool(is_min_thin),
                                     "contributors_at_min": contributors_min.to_dict("records")}

    print("\nC. Formal pre-trend test")
    pre = ev[ev["event_time"] < 0].copy()
    pretrend_report = {}
    for ycol in outcome_cols:
        sub = pre[["customer_id", "event_time", ycol]].dropna()
        entry = {}
        if sub["event_time"].nunique() < 3 or sub["customer_id"].nunique() < 3:
            print(f"  [{ycol}] insufficient pre-period observations — skipping")
            pretrend_report[ycol] = {"status": "insufficient_data"}
            continue
        model = smf.ols(f"{ycol} ~ event_time", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["customer_id"]})
        coef, pval = float(model.params["event_time"]), float(model.pvalues["event_time"])
        sig = pval < SIG_ALPHA
        print(f"  [{ycol}] full pre-period slope={coef:+.4f}, p={pval:.4f} -> "
              f"{'SIGNIFICANT (possible pre-trend)' if sig else 'not significant'}")
        entry["full_period"] = {"slope": coef, "pvalue": pval, "significant": sig}

        near = sub[sub["event_time"] >= NEAR_WINDOW_CUTOFF]
        far = sub[sub["event_time"] < NEAR_WINDOW_CUTOFF]
        if near["event_time"].nunique() >= 3 and near["customer_id"].nunique() >= 3:
            m_near = smf.ols(f"{ycol} ~ event_time", data=near).fit(cov_type="cluster", cov_kwds={"groups": near["customer_id"]})
            near_coef, near_p = float(m_near.params["event_time"]), float(m_near.pvalues["event_time"])
            print(f"    -> near window (t in [{NEAR_WINDOW_CUTOFF},-1]) slope={near_coef:+.4f}, p={near_p:.4f}"
                  f"{'  possible anticipation' if near_p < SIG_ALPHA else ''}")
            entry["near_window"] = {"slope": near_coef, "pvalue": near_p, "significant": near_p < SIG_ALPHA,
                                     "cutoff": NEAR_WINDOW_CUTOFF}
        if far["event_time"].nunique() >= 3 and far["customer_id"].nunique() >= 3:
            m_far = smf.ols(f"{ycol} ~ event_time", data=far).fit(cov_type="cluster", cov_kwds={"groups": far["customer_id"]})
            far_coef, far_p = float(m_far.params["event_time"]), float(m_far.pvalues["event_time"])
            print(f"    -> far window (t < {NEAR_WINDOW_CUTOFF}) slope={far_coef:+.4f}, p={far_p:.4f}")
            entry["far_window"] = {"slope": far_coef, "pvalue": far_p, "significant": far_p < SIG_ALPHA}
        pretrend_report[ycol] = entry

    any_significant = any(v.get("full_period", {}).get("significant", False) or
                           v.get("near_window", {}).get("significant", False) for v in pretrend_report.values())
    version_verdict = "PRETREND_DETECTED" if any_significant else "NO_PRETREND_PARALLEL_TRENDS_SUPPORTED"
    print(f"\n[{tag}] Overall verdict: {version_verdict}")
    all_results[tag] = {"n_by_event_time_min": int(n_by_et.min()), "n_by_event_time_median": float(n_by_et.median()),
                         "thin_bins": {int(k): int(v) for k, v in thin_bins.items()},
                         "contributor_report": contributor_report, "pretrend_tests": pretrend_report,
                         "version_verdict": version_verdict}
    print()

print("=" * 70)
print("Final synthesis")
print("=" * 70)
primary_verdict = all_results.get("PRIMARY", {}).get("version_verdict")
robust_verdict = all_results.get("ROBUST", {}).get("version_verdict")
converged = (primary_verdict == robust_verdict) if (primary_verdict and robust_verdict) else None
if converged:
    print(f"PRIMARY ({primary_verdict}) and ROBUST ({robust_verdict}) agree — this supports the robustness "
          f"of the left-censoring recovery procedure (SAFE/CAUTION classification).")
elif converged is False:
    print(f"PRIMARY ({primary_verdict}) and ROBUST ({robust_verdict}) disagree — the CAUTION-flagged "
          f"customers materially affect the conclusion; adopt PRIMARY as the main result and note "
          f"this disagreement explicitly in the robustness section.")
else:
    print("One or more versions missing — cannot compare.")

out_path = STEP2_OUT / f"event_time_diagnostics_type{TARGET_TYPE}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"target_type": TARGET_TYPE, "near_window_cutoff": NEAR_WINDOW_CUTOFF, "significance_alpha": SIG_ALPHA,
               "versions": all_results, "primary_vs_robust_converged": converged}, f, ensure_ascii=False, indent=2, default=str)
print(f"\nDiagnostics saved: {out_path}")
