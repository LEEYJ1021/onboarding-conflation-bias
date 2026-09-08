"""
================================================================================
Step 1b — Automated re-verification of df_analysis_master.csv
================================================================================
Cross-checks the three step1a outputs (master.csv, manifest.json,
audit.csv) against each other and against the raw sample-selection logic,
then issues a PASS/FAIL verdict plus a citable data-integrity statement
for the paper's Data section.

Checks:
    A. Cross-consistency of the three step1a outputs (customer counts,
       min-window-floor exclusion lists, and stage-by-stage headcounts)
    B. No customer with zero all-time spend remains in the final sample
    C. Leverage-point resolution — skewness/std of customer_total_cost_alltime
       (raw and log1p) in the final sample
    D. Panel-structure sanity — observation-day distribution, floor violations
    E. Overall verdict + auto-generated citation-ready integrity statement

Inputs:  df_analysis_master.csv, reproducibility_manifest.json,
         sample_selection_audit.csv (step1a outputs)
Outputs: data_integrity_final_report.json
================================================================================
"""
import json

import numpy as np
import pandas as pd
from scipy import stats as sps

from config import AD_DATA_ROOT

MASTER_PATH = AD_DATA_ROOT / "df_analysis_master.csv"
MANIFEST_PATH = AD_DATA_ROOT / "reproducibility_manifest.json"
AUDIT_PATH = AD_DATA_ROOT / "sample_selection_audit.csv"

for p in (MASTER_PATH, MANIFEST_PATH, AUDIT_PATH):
    if not p.exists():
        raise FileNotFoundError(f"Missing step1a output: {p} — run step1a_build_master_dataset.py first.")

master = pd.read_csv(MASTER_PATH)
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)
audit = pd.read_csv(AUDIT_PATH)

issues = []

# ------------------------------------------------------------------
# A. Cross-consistency of the three outputs
# ------------------------------------------------------------------
print("=" * 70)
print("A. Cross-consistency of step1a outputs")
print("=" * 70)

n_customers_master = master["customer_id"].nunique()
n_customers_manifest = manifest["sample_selection"]["n_clean_customers_final"]
print(f"master.csv customer count : {n_customers_master}")
print(f"manifest customer count   : {n_customers_manifest}")
if n_customers_master != n_customers_manifest:
    issues.append(f"[A-1] mismatch: master={n_customers_master} vs manifest={n_customers_manifest}")
else:
    print("  OK: match")

min_days_rule = manifest["sample_selection"]["min_stable_window_days_rule"]
manifest_excluded = set(manifest["sample_selection"]["excluded_by_min_days_rule"])
audit_excluded = set(audit.loc[audit["dropped_at_stage"] == "min_stable_window_days", "customer_id"])
print(f"\nmanifest min-window floor: {min_days_rule} days")
print(f"manifest excluded list   : {sorted(manifest_excluded)}")
print(f"audit.csv excluded list  : {sorted(audit_excluded)}")
if manifest_excluded != audit_excluded:
    issues.append(f"[A-2] mismatch: manifest={sorted(manifest_excluded)} vs audit={sorted(audit_excluded)}")
else:
    print("  OK: match")

stage_order = ["registry_match", "continuous_block", "stable_window_exists",
               "test_or_billing_anomaly", "min_stable_window_days"]
stage_labels = {
    "registry_match": "registry match", "continuous_block": "continuous block",
    "stable_window_exists": "stable window exists",
    "test_or_billing_anomaly": "non-test/non-billing-anomalous",
    "min_stable_window_days": f"min-{min_days_rule}-day floor",
}
n_start = manifest["sample_selection"]["n_registry_matched"]
print(f"\nStage-by-stage recomputation (registry-match start = {n_start}):")
expected_after = {
    "registry_match": n_start,
    "continuous_block": manifest["sample_selection"]["n_continuous_block"],
    "stable_window_exists": manifest["sample_selection"]["n_stable_window_exists"],
    "test_or_billing_anomaly": manifest["sample_selection"]["n_clean_pre_mindays"],
    "min_stable_window_days": manifest["sample_selection"]["n_clean_customers_final"],
}
prev_n = n_start
for stage in stage_order:
    n_dropped_this_stage = (audit["dropped_at_stage"] == stage).sum()
    n_after = prev_n - n_dropped_this_stage if stage != "registry_match" else n_start
    label = stage_labels[stage]
    exp = expected_after[stage]
    status = "OK" if (n_after == exp or stage == "registry_match") else "MISMATCH"
    print(f"  {label:32s}: dropped this stage {n_dropped_this_stage:>3} -> remaining {n_after:>3} "
          f"(manifest: {exp}) [{status}]")
    if stage != "registry_match" and n_after != exp:
        issues.append(f"[A-3] {label} stage count mismatch: recomputed={n_after} vs manifest={exp}")
    prev_n = n_after if stage != "registry_match" else n_start

# ------------------------------------------------------------------
# B. No zero-spend customers remain
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("B. Zero all-time-spend customers remaining in final sample")
print("=" * 70)
cust_level = master.drop_duplicates(subset="customer_id").set_index("customer_id")
zero_remaining = cust_level.loc[cust_level["customer_total_cost_alltime"] == 0]
if len(zero_remaining):
    issues.append(f"[B] zero-spend customers found: {zero_remaining.index.tolist()}")
    print(f"FAIL: found {zero_remaining.index.tolist()}")
else:
    print(f"OK: none (n={n_customers_master}, all customer_total_cost_alltime > 0)")
print(f"  min(customer_total_cost_alltime): {cust_level['customer_total_cost_alltime'].min():,.0f}")

# ------------------------------------------------------------------
# C. Leverage-point resolution — skewness/std of the final sample
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"C. Leverage-point check — final ({n_customers_master}) customer_total_cost_alltime distribution")
print("=" * 70)
tc_final = cust_level["customer_total_cost_alltime"].astype(float)
skew_raw = sps.skew(tc_final, bias=False)
skew_log = sps.skew(np.log1p(tc_final), bias=False)
std_log = np.log1p(tc_final).std()
print(f"n={len(tc_final)}, skewness_raw={skew_raw:.3f}, skewness_log1p={skew_log:.3f}, std_log1p={std_log:.3f}")

# ------------------------------------------------------------------
# D. Panel-structure sanity
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("D. Panel-structure sanity check")
print("=" * 70)
date_candidates = [c for c in master.columns if c.lower() in ("date", "stat_date", "stat_dt", "dt")]
date_col = date_candidates[0] if date_candidates else None
if date_col:
    master[date_col] = pd.to_datetime(master[date_col])
    days_per_cust = master.groupby("customer_id")[date_col].nunique()
else:
    days_per_cust = master.groupby("customer_id").size()
print(days_per_cust.describe().round(1))
n_below_rule = (days_per_cust < min_days_rule).sum()
print(f"\nCustomers observed < {min_days_rule} days (should be 0): {n_below_rule}")
if n_below_rule > 0:
    issues.append(f"[D] {n_below_rule} customers observed below the {min_days_rule}-day floor")
else:
    print(f"  OK: all customers observed >= {min_days_rule} days")

print(f"\nTotal rows: {len(master):,} / total columns: {master.shape[1]}")
n_master_rows_manifest = manifest.get("n_master_rows")
if n_master_rows_manifest is not None and n_master_rows_manifest != len(master):
    issues.append(f"[D] row-count mismatch: master.csv={len(master)} vs manifest={n_master_rows_manifest}")
else:
    print(f"  OK: matches manifest row count ({n_master_rows_manifest})")

# ------------------------------------------------------------------
# E. Overall verdict + citable integrity statement
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("E. Overall verdict")
print("=" * 70)
if not issues:
    verdict = "PASS"
    print("PASS — all A-D checks succeeded. Data integrity re-verification (Section 3.1 of the paper) "
          "is complete; proceed to treatment-event redefinition.")
else:
    verdict = "FAIL"
    print(f"FAIL — {len(issues)} issue(s) found. Resolve before proceeding:")
    for i in issues:
        print(f"   - {i}")

statement = (
    f"The final analysis sample starts from {n_start} registered accounts and applies four "
    f"sequential filters — (1) a continuous observation block, (2) a stable observation window, "
    f"(3) exclusion of test/billing-anomalous accounts, and (4) a minimum observation-window "
    f"length of {min_days_rule} days — yielding {n_customers_master} customers "
    f"({len(master):,} customer-day observations). A preliminary diagnostic identified one "
    f"clean-sample customer with zero all-time spend; rather than an ad-hoc exclusion, a "
    f"{min_days_rule}-day minimum-observation-window rule was formally adopted, which jointly "
    f"excluded {len(manifest['sample_selection']['excluded_by_min_days_rule'])} customers. "
    f"We re-confirm that no zero-spend account remains in the final sample."
)
print("\n" + "-" * 70)
print("Citable data-integrity statement (draft):")
print("-" * 70)
print(statement)

report_path = AD_DATA_ROOT / "data_integrity_final_report.json"
report = {
    "verdict": verdict, "issues": issues,
    "n_customers_final": int(n_customers_master), "n_rows_final": int(len(master)),
    "skewness_raw": float(skew_raw), "skewness_log1p": float(skew_log), "std_log1p": float(std_log),
    "min_stable_window_days_rule": int(min_days_rule),
    "paper_statement_draft": statement,
}
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\nFinal report saved: {report_path}")
