"""
================================================================================
Step 2.3 — Formal Pre-Trend Test
================================================================================
Background: the Step-2.2 event-study plot (event_study_FINAL_type6.png) shows
visible fluctuation in spend / portfolio diversity near the adoption date
(t ~ -15..-1). Before interpreting this as a "trend" we must formally test:
    1) whether this fluctuation is a real trend, or an artifact of a handful
       of customers' idiosyncratic values pulling down thin event-time bins;
    2) whether a statistically significant linear pre-trend (Ashenfelter
       dip / anticipation effect) exists — if so, the parallel-trends
       assumption is threatened.

Procedure:
    A. Tabulate the number of observed customers per event_time — flag "thin"
       bins at or below 30% of the median.
    B. List which customers produced the local min/max outcome values at each
       event_time (to directly inspect whether a spike/drop reflects a data
       anomaly).
    C. Regress outcome ~ event_time on the pre-period only (event_time < 0),
       with customer-clustered robust standard errors. Test the full
       pre-period as well as a near window (anticipation window, t in
       [-15,-1]) and a far window (t < -15) separately, to see whether any
       dip concentrates near the adoption date (anticipation).

    PRIMARY (32) is the main test target; the same tests are repeated on
    ROBUST (35) to check whether both converge on the same conclusion
    (significant / not significant) — this also serves as a robustness
    check on the Step-2.2 left-censoring recovery procedure (SAFE/CAUTION
    classification) itself.

Input:  event_time_panel_FINAL_type{T}.csv, event_time_panel_ROBUST_type{T}.csv
Output: event_time_diagnostics_type{T}.json
        detailed diagnostics printed to console (per version, A/B/C)
================================================================================
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))

TARGET_TYPE = int(os.environ.get("TARGET_TYPE", "6"))
NEAR_WINDOW_CUTOFF = int(os.environ.get("NEAR_WINDOW_CUTOFF", "-15"))  # anticipation-window boundary
SIG_ALPHA = 0.05

VERSIONS = {
    "PRIMARY": STEP2_DIR / f"event_time_panel_FINAL_type{TARGET_TYPE}.csv",
    "ROBUST": STEP2_DIR / f"event_time_panel_ROBUST_type{TARGET_TYPE}.csv",
}

all_results = {}

for tag, ev_path in VERSIONS.items():
    print("=" * 70)
    print(f"[{tag}] Type {TARGET_TYPE} pre-trend diagnosis ({ev_path.name})")
    print("=" * 70)

    if not ev_path.exists():
        print(f"File not found: {ev_path} — skipping")
        continue
    ev = pd.read_csv(ev_path)
    outcome_cols = [c for c in ["log_spend_safe", "n_campaign_types_active"] if c in ev.columns]

    # ------------------------------------------------------------------
    # A. Number of observed customers by event_time (thin-bin check)
    # ------------------------------------------------------------------
    print("\nA. Number of observed customers by event_time")
    n_by_et = ev.groupby("event_time")["customer_id"].nunique().rename("n_customers")
    thin_threshold = max(2, n_by_et.median() * 0.3)
    thin_bins = n_by_et[n_by_et <= thin_threshold]
    print(f"   event_time range: {n_by_et.index.min()} to {n_by_et.index.max()}, "
          f"median customers={n_by_et.median():.0f}, min={n_by_et.min()}, max={n_by_et.max()}")
    if len(thin_bins):
        print(f"   NOTE: thin bins (threshold <= {thin_threshold:.1f}, {len(thin_bins)} bins):")
        print(f"     {sorted(thin_bins.index.tolist())}")
    else:
        print("   No especially thin bins")

    # ------------------------------------------------------------------
    # B. Identify the customers driving spikes/drops
    # ------------------------------------------------------------------
    contributor_report = {}
    for ycol in outcome_cols:
        et_mean = ev.groupby("event_time")[ycol].mean()
        min_et = et_mean.idxmin()
        max_et = et_mean.idxmax()
        n_at_min = int(n_by_et.loc[min_et])
        is_min_thin = min_et in thin_bins.index

        print(f"\nB. [{ycol}] minimum-mean point: event_time={min_et} "
              f"(mean={et_mean.loc[min_et]:.3f}, n_customers={n_at_min}"
              f"{' THIN BIN' if is_min_thin else ''})")
        contributors_min = ev[ev["event_time"] == min_et][["customer_id", ycol]].sort_values(ycol)
        print(contributors_min.to_string(index=False))

        print(f"   [{ycol}] maximum-mean point: event_time={max_et} "
              f"(mean={et_mean.loc[max_et]:.3f}, n_customers={int(n_by_et.loc[max_et])})")

        contributor_report[ycol] = {
            "min_event_time": int(min_et),
            "min_mean": float(et_mean.loc[min_et]),
            "n_customers_at_min": n_at_min,
            "min_is_thin_bin": bool(is_min_thin),
            "contributors_at_min": contributors_min.to_dict("records"),
        }

    # ------------------------------------------------------------------
    # C. Formal pre-trend test — pre-period only (event_time<0),
    #    customer-clustered robust SE. Near vs. far window split.
    # ------------------------------------------------------------------
    print("\nC. Formal pre-trend test")
    pre = ev[ev["event_time"] < 0].copy()
    pretrend_report = {}
    for ycol in outcome_cols:
        sub = pre[["customer_id", "event_time", ycol]].dropna()
        entry = {}
        if sub["event_time"].nunique() < 3 or sub["customer_id"].nunique() < 3:
            print(f"  [{ycol}] insufficient pre-period observations — test skipped")
            pretrend_report[ycol] = {"status": "insufficient_data"}
            continue

        model = smf.ols(f"{ycol} ~ event_time", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["customer_id"]}
        )
        coef = float(model.params["event_time"])
        pval = float(model.pvalues["event_time"])
        sig = pval < SIG_ALPHA
        sig_label = "significant (p<0.05) - possible pre-trend" if sig else "not significant - no clear linear pre-trend"
        print(f"  [{ycol}] full pre-period slope={coef:+.4f}, p={pval:.4f} -> {sig_label}")
        entry["full_period"] = {"slope": coef, "pvalue": pval, "significant": sig}

        near = sub[sub["event_time"] >= NEAR_WINDOW_CUTOFF]
        far = sub[sub["event_time"] < NEAR_WINDOW_CUTOFF]
        if near["event_time"].nunique() >= 3 and near["customer_id"].nunique() >= 3:
            m_near = smf.ols(f"{ycol} ~ event_time", data=near).fit(
                cov_type="cluster", cov_kwds={"groups": near["customer_id"]}
            )
            near_coef = float(m_near.params["event_time"])
            near_p = float(m_near.pvalues["event_time"])
            print(f"    near window (t in [{NEAR_WINDOW_CUTOFF},-1]) slope={near_coef:+.4f}, p={near_p:.4f}"
                  f"{'  NOTE: possible anticipation' if near_p < SIG_ALPHA else ''}")
            entry["near_window"] = {"slope": near_coef, "pvalue": near_p, "significant": near_p < SIG_ALPHA,
                                     "cutoff": NEAR_WINDOW_CUTOFF}
        if far["event_time"].nunique() >= 3 and far["customer_id"].nunique() >= 3:
            m_far = smf.ols(f"{ycol} ~ event_time", data=far).fit(
                cov_type="cluster", cov_kwds={"groups": far["customer_id"]}
            )
            far_coef = float(m_far.params["event_time"])
            far_p = float(m_far.pvalues["event_time"])
            print(f"    far window (t < {NEAR_WINDOW_CUTOFF}) slope={far_coef:+.4f}, p={far_p:.4f}")
            entry["far_window"] = {"slope": far_coef, "pvalue": far_p, "significant": far_p < SIG_ALPHA}

        pretrend_report[ycol] = entry

    # ------------------------------------------------------------------
    # Per-version overall verdict
    # ------------------------------------------------------------------
    any_significant = any(
        v.get("full_period", {}).get("significant", False) or
        v.get("near_window", {}).get("significant", False)
        for v in pretrend_report.values()
    )
    version_verdict = "PRETREND_DETECTED" if any_significant else "NO_PRETREND_PARALLEL_TRENDS_SUPPORTED"
    print(f"\n[{tag}] Overall verdict: {version_verdict}")

    all_results[tag] = {
        "n_by_event_time_min": int(n_by_et.min()),
        "n_by_event_time_median": float(n_by_et.median()),
        "thin_bins": {int(k): int(v) for k, v in thin_bins.items()},
        "contributor_report": contributor_report,
        "pretrend_tests": pretrend_report,
        "version_verdict": version_verdict,
    }
    print()

# ------------------------------------------------------------------
# Final summary: do PRIMARY and ROBUST converge?
# ------------------------------------------------------------------
print("=" * 70)
print("Final overall verdict")
print("=" * 70)
primary_verdict = all_results.get("PRIMARY", {}).get("version_verdict")
robust_verdict = all_results.get("ROBUST", {}).get("version_verdict")
converged = (primary_verdict == robust_verdict) if (primary_verdict and robust_verdict) else None

if converged:
    print(f"CONVERGED: PRIMARY ({primary_verdict}) and ROBUST ({robust_verdict}) reach the same "
          f"conclusion — supports the robustness of the left-censoring recovery procedure "
          f"(SAFE/CAUTION classification).")
elif converged is False:
    print(f"DIVERGED: PRIMARY ({primary_verdict}) and ROBUST ({robust_verdict}) reach different "
          f"conclusions — including the 3 CAUTION customers affects the result, so the paper "
          f"should adopt PRIMARY as the main result and explicitly note this discrepancy in "
          f"the robustness section.")
else:
    print("Some version results missing — cannot compare")

out_path = STEP2_DIR / f"event_time_diagnostics_type{TARGET_TYPE}.json"
final_report = {
    "target_type": TARGET_TYPE,
    "near_window_cutoff": NEAR_WINDOW_CUTOFF,
    "significance_alpha": SIG_ALPHA,
    "versions": all_results,
    "primary_vs_robust_converged": converged,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
print(f"\nDiagnostic summary saved: {out_path}")

if all_results.get("PRIMARY", {}).get("version_verdict") == "NO_PRETREND_PARALLEL_TRENDS_SUPPORTED":
    print("\nNext step: proceed to step3_callaway_santanna.py for the main estimation")
    print("(parallel-trends assumption supported — primary_adopter 32 vs. never_treated 61)")
else:
    print("\nNOTE: if a significant pre-trend is detected, either use an anticipation")
    print("  parameter (recode the near-adoption window as already treated) in Step 3,")
    print("  or exclude the near window from the event window entirely.")
