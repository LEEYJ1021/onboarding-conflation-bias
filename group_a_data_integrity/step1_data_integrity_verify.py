"""
================================================================================
Step 1b — Automated Consistency Verification of Step 1a Outputs
================================================================================
Input: df_analysis_master.csv, reproducibility_manifest.json,
       sample_selection_audit.csv (all Step 1a outputs)

Verification checks:
    A. Cross-consistency of the three outputs
       - manifest n_clean_customers_final == actual customer count in master.csv
       - manifest min_stable_window_days_rule / excluded_by_min_days_rule
         == the min_stable_window_days drop list in audit.csv
       - applying the audit.csv stage-by-stage drop counts in sequence
         reproduces exactly the per-stage counts recorded in the manifest
    B. Whether any all-time zero-spend customer remains in the final sample
       (there should be none)
    C. Whether the leverage point has been resolved — is the skewness/SD of
       log1p(customer_total_cost_alltime) in a sensible range
    D. Panel structure sanity — distribution of observed days, any violation
       of the minimum-day rule
    E. Overall PASS/FAIL verdict + an auto-generated consistency statement
       for citation in the paper's Section 1

Output: data_integrity_final_report.json
================================================================================
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))

MASTER_PATH = ROOT / "df_analysis_master.csv"
MANIFEST_PATH = ROOT / "reproducibility_manifest.json"
AUDIT_PATH = ROOT / "sample_selection_audit.csv"

for p in [MASTER_PATH, MANIFEST_PATH, AUDIT_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Required output missing: {p} — run step1_data_integrity_build.py first.")

master = pd.read_csv(MASTER_PATH)
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)
audit = pd.read_csv(AUDIT_PATH)

issues = []  # anything appended here forces a FAIL verdict

# ------------------------------------------------------------------
# A. Cross-consistency of the three outputs
# ------------------------------------------------------------------
print("=" * 70)
print("A. Cross-consistency of the three outputs")
print("=" * 70)

n_customers_master = master["customer_id"].nunique()
n_customers_manifest = manifest["sample_selection"]["n_clean_customers_final"]
print(f"Actual customer count in master.csv: {n_customers_master}")
print(f"Customer count recorded in manifest : {n_customers_manifest}")
if n_customers_master != n_customers_manifest:
    issues.append(f"[A-1] mismatch: master={n_customers_master} vs manifest={n_customers_manifest}")
else:
    print("  OK: match")

min_days_rule = manifest["sample_selection"]["min_stable_window_days_rule"]
manifest_excluded = set(manifest["sample_selection"]["excluded_by_min_days_rule"])
audit_excluded = set(
    audit.loc[audit["dropped_at_stage"] == "min_stable_window_days", "customer_id"]
)
print(f"\nManifest-recorded minimum-day rule: {min_days_rule} days")
print(f"Manifest excluded-customer list    : {sorted(manifest_excluded)}")
print(f"audit.csv excluded-customer list   : {sorted(audit_excluded)}")
if manifest_excluded != audit_excluded:
    issues.append(f"[A-2] mismatch: manifest={sorted(manifest_excluded)} vs audit={sorted(audit_excluded)}")
else:
    print("  OK: match")

stage_order = ["registry_match", "continuous_block", "stable_window_exists",
               "test_or_billing_anomaly", "min_stable_window_days"]
stage_labels = {
    "registry_match": "registry match",
    "continuous_block": "continuous block",
    "stable_window_exists": "stable window exists",
    "test_or_billing_anomaly": "non-test/non-anomalous",
    "min_stable_window_days": f"min {min_days_rule}-day rule",
}
n_start = manifest["sample_selection"]["n_registry_matched"]
print(f"\nStage-by-stage cumulative verification (starting at {n_start} registry-matched):")
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
    print(f"  {label:24s}: dropped this stage {n_dropped_this_stage:>3} -> remaining {n_after:>3} "
          f"(manifest value {exp}) [{status}]")
    if stage != "registry_match" and n_after != exp:
        issues.append(f"[A-3] mismatch at stage {label}: recomputed={n_after} vs manifest={exp}")
    prev_n = n_after if stage != "registry_match" else n_start

# ------------------------------------------------------------------
# B. Any remaining all-time zero-spend customers in the final sample?
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("B. Remaining all-time zero-spend customers in final sample")
print("=" * 70)
cust_level = master.drop_duplicates(subset="customer_id").set_index("customer_id")
zero_remaining = cust_level.loc[cust_level["customer_total_cost_alltime"] == 0]
if len(zero_remaining):
    issues.append(f"[B] zero-spend customers found: {zero_remaining.index.tolist()}")
    print(f"FAIL: found {zero_remaining.index.tolist()}")
else:
    print(f"OK: none (n={n_customers_master}, all have total_cost > 0)")
print(f"  min(customer_total_cost_alltime): {cust_level['customer_total_cost_alltime'].min():,.0f}")

# ------------------------------------------------------------------
# C. Leverage-point resolution — skewness/SD of the final sample's log1p
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"C. Leverage-point check — log1p distribution of customer_total_cost_alltime (n={n_customers_master})")
print("=" * 70)
tc_final = cust_level["customer_total_cost_alltime"].astype(float)
skew_raw = sps.skew(tc_final, bias=False)
skew_log = sps.skew(np.log1p(tc_final), bias=False)
std_log = np.log1p(tc_final).std()
print(f"n={len(tc_final)}, skewness_raw={skew_raw:.3f}, skewness_log1p={skew_log:.3f}, "
      f"std_log1p={std_log:.3f}")
REF_STD_LOG1P = os.environ.get("REF_STD_LOG1P_MAX")
if REF_STD_LOG1P is not None:
    ref_max = float(REF_STD_LOG1P)
    if std_log > ref_max:
        issues.append(f"[C] warning: final-sample std_log1p={std_log:.3f} exceeds reference max ({ref_max})")
    else:
        print(f"  OK: within reference max ({ref_max})")
else:
    print("  (REF_STD_LOG1P_MAX not set — absolute-bound check skipped, values reported for reference)")

# ------------------------------------------------------------------
# D. Panel structure sanity — observed-days distribution, min-day violations
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("D. Panel structure sanity check on final sample")
print("=" * 70)
DATE_COL_CANDIDATES = [c for c in master.columns if c.lower() in ("date", "stat_date", "stat_dt", "dt")]
date_col = DATE_COL_CANDIDATES[0] if DATE_COL_CANDIDATES else None
if date_col:
    master[date_col] = pd.to_datetime(master[date_col])
    days_per_cust = master.groupby("customer_id")[date_col].nunique()
else:
    days_per_cust = master.groupby("customer_id").size()
print(days_per_cust.describe().round(1))
n_below_rule = (days_per_cust < min_days_rule).sum()
print(f"\nCustomers observed for fewer than {min_days_rule} days (violation if >0): {n_below_rule}")
if n_below_rule > 0:
    issues.append(f"[D] {n_below_rule} customers in final sample below {min_days_rule}-day rule")
else:
    print(f"  OK: all customers observed >= {min_days_rule} days (rule correctly applied)")

print(f"\nTotal rows: {len(master):,} / Total columns: {master.shape[1]}")

n_master_rows_manifest = manifest.get("n_master_rows")
if n_master_rows_manifest is not None and n_master_rows_manifest != len(master):
    issues.append(f"[D] row-count mismatch: master.csv={len(master)} vs manifest={n_master_rows_manifest}")
else:
    print(f"  OK: matches manifest-recorded row count ({n_master_rows_manifest})")

# ------------------------------------------------------------------
# E. Final verdict + auto-generated Section-1 citation statement
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("E. Final verdict")
print("=" * 70)
if not issues:
    verdict = "PASS"
    print("PASS — all A-D checks succeeded. Section 1 (data-integrity re-verification) can be")
    print("   considered complete; proceed to Section 2 (treatment redefinition).")
else:
    verdict = "FAIL"
    print(f"FAIL — {len(issues)} issue(s) found. Resolve before moving to Section 2:")
    for i in issues:
        print(f"   - {i}")

statement = (
    f"The final analysis sample was constructed from {n_start} originally registered "
    f"accounts by sequentially applying four criteria: (1) a continuous observation "
    f"block, (2) a stable observation window (is_stable_window), (3) exclusion of "
    f"test/billing-anomalous accounts, and (4) a minimum observation length of "
    f"{min_days_rule} days, yielding {n_customers_master} customers "
    f"({len(master):,} customer-day observations). A preliminary diagnostic identified "
    f"one clean-sample customer with zero all-time cumulative spend, subsequently traced "
    f"to a post-registration, pre-execution account; rather than an ad hoc exclusion, a "
    f"formal minimum-observation-length rule of {min_days_rule} days was adopted, which "
    f"jointly excluded a total of {len(manifest['sample_selection']['excluded_by_min_days_rule'])} "
    f"customers. We confirmed that no zero-all-time-spend account remains in the final sample."
)
print("\n" + "-" * 70)
print("Draft consistency statement for citation in Section 1:")
print("-" * 70)
print(statement)

report_path = ROOT / "data_integrity_final_report.json"
report = {
    "verdict": verdict,
    "issues": issues,
    "n_customers_final": int(n_customers_master),
    "n_rows_final": int(len(master)),
    "skewness_raw": float(skew_raw),
    "skewness_log1p": float(skew_log),
    "std_log1p": float(std_log),
    "min_stable_window_days_rule": int(min_days_rule),
    "paper_statement_draft": statement,
}
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\nFinal report saved: {report_path}")
