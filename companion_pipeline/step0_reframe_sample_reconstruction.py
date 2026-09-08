"""
================================================================================
Applied Paper - Step 0: Clean Onboarding Sample Reconstruction
================================================================================
Background:
    The discovery paper's minimum-30-day stable-window rule was designed
    for long-run observability and, as shown there, systematically
    misclassifies genuinely new customers as left-censored. For the
    applied research question ("does a new advertiser's first-adopted
    campaign type predict early growth trajectory?") that filter is not
    just unnecessary but actively harmful, so it is dropped entirely.

Sample criteria (all four applied fresh, no 30-day rule):
    1. Registry match (customer_level_attributes.csv)
    2. Reliable first-observation date: raw_panel_date_min must be more
       than BUFFER_DAYS after the overall panel start date, otherwise it
       reflects "when data collection began" rather than "when the
       customer registered" (a real onboarding date cannot be identified).
    3. Observation persistence: at least EARLY_WINDOW days of panel data
       after registration (minimum requirement to measure outcomes).
    4. Not flagged test/billing-anomalous within the first EARLY_WINDOW
       days.

Output: reframe_output/clean_onboarding_sample.csv
        reframe_output/sample_reconstruction_audit.csv
        reframe_output/sample_reconstruction_manifest.json
        reframe_output/raw_panel_date_min_histogram.png
        reframe_output/buffer_sensitivity_precheck.csv
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
OUT_DIR = Path(os.environ.get("REFRAME_OUT", str(ROOT / "reframe_output")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUFFER_DAYS = int(os.environ.get("BUFFER_DAYS", "30"))
EARLY_WINDOW = int(os.environ.get("EARLY_OUTCOME_WINDOW_DAYS", "30"))
BUFFER_SENSITIVITY_GRID = [15, 30, 45, 60]


def _detect_date_column(df):
    for c in ("date", "stat_date", "stat_dt", "dt", "ad_date", "report_date", "log_date"):
        if c in df.columns:
            return c
    raise KeyError("Could not auto-detect a date column")


panel = pd.read_csv(ROOT / "customer_day_panel.csv")
ctp = pd.read_csv(ROOT / "customer_day_campaign_type_panel.csv")
attrs = pd.read_csv(ROOT / "customer_level_attributes.csv")

DATE_COL, DATE_COL_CTP = _detect_date_column(panel), _detect_date_column(ctp)
panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])
ctp[DATE_COL_CTP] = pd.to_datetime(ctp[DATE_COL_CTP])

audit_rows = []

# 1. Registry match
registry_customers = set(attrs["customer_id"].unique())
panel_registry = panel[panel["customer_id"].isin(registry_customers)].copy()
n_registry = len(registry_customers)
print(f"Registry-matched customers: {n_registry}")

# 2. First-observation reliability (buffer check)
raw_date_min = panel_registry.groupby("customer_id")[DATE_COL].min().rename("raw_panel_date_min")
overall_start, overall_end = panel_registry[DATE_COL].min(), panel_registry[DATE_COL].max()
print(f"Overall panel coverage: {overall_start.date()} to {overall_end.date()}")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(raw_date_min, bins=40, color="steelblue", edgecolor="white")
ax.axvline(overall_start + pd.Timedelta(days=BUFFER_DAYS), color="firebrick", linestyle="--",
           label=f"buffer cutoff (start+{BUFFER_DAYS}d)")
ax.set_title("Distribution of raw_panel_date_min\n(a spike at the panel start = 'data collection start', not registration)")
ax.set_xlabel("raw_panel_date_min"); ax.set_ylabel("Number of customers"); ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "raw_panel_date_min_histogram.png", dpi=150)
plt.close()
print("Histogram saved — inspect visually for clustering before proceeding.")

buffer_cutoff = overall_start + pd.Timedelta(days=BUFFER_DAYS)
buffer_reliable = set(raw_date_min[raw_date_min > buffer_cutoff].index)
print(f"Passed buffer check ({BUFFER_DAYS}d): {len(buffer_reliable)}/{n_registry}")

sens_rows = []
for b in BUFFER_SENSITIVITY_GRID:
    cutoff = overall_start + pd.Timedelta(days=b)
    n_pass = int((raw_date_min > cutoff).sum())
    sens_rows.append({"buffer_days": b, "n_pass": n_pass, "n_total_registry": n_registry})
pd.DataFrame(sens_rows).to_csv(OUT_DIR / "buffer_sensitivity_precheck.csv", index=False)

# 3. Observation persistence
date_max = panel_registry.groupby("customer_id")[DATE_COL].max().rename("raw_panel_date_max")
persistence = raw_date_min.to_frame().join(date_max)
persistence["days_available"] = (persistence["raw_panel_date_max"] - persistence["raw_panel_date_min"]).dt.days + 1
persistent = set(persistence[persistence["days_available"] >= EARLY_WINDOW].index)
after_persistence = buffer_reliable & persistent
print(f"Passed persistence check ({EARLY_WINDOW}d): {len(after_persistence)}/{len(buffer_reliable)}")

# 4. Not test/billing-anomalous in the early window
reg_date_map = raw_date_min.to_dict()
clean_flags = []
for cid in sorted(after_persistence):
    reg_date = reg_date_map[cid]
    window_end = reg_date + pd.Timedelta(days=EARLY_WINDOW)
    sub = panel_registry[(panel_registry["customer_id"] == cid) & (panel_registry[DATE_COL] >= reg_date)
                          & (panel_registry[DATE_COL] < window_end)]
    has_test = bool(sub.get("is_test_account", pd.Series(dtype=bool)).fillna(False).any())
    has_anom = bool(sub.get("billing_anomaly_flag", pd.Series(dtype=bool)).fillna(False).any())
    clean_flags.append({"customer_id": cid, "has_test": has_test, "has_anom": has_anom})
clean_df = pd.DataFrame(clean_flags)
clean_ids = set(clean_df.loc[~clean_df["has_test"] & ~clean_df["has_anom"], "customer_id"])
print(f"Final clean sample: {len(clean_ids)}/{len(after_persistence)}")

final_sample = pd.DataFrame({"customer_id": sorted(clean_ids)})
final_sample["registration_date"] = final_sample["customer_id"].map(reg_date_map)
final_sample = final_sample.merge(
    persistence[["days_available"]].reset_index().rename(columns={"index": "customer_id"}),
    on="customer_id", how="left")
final_sample.to_csv(OUT_DIR / "clean_onboarding_sample.csv", index=False)

manifest = {
    "buffer_days": BUFFER_DAYS, "early_window_days": EARLY_WINDOW,
    "n_registry_matched": n_registry, "n_buffer_reliable": len(buffer_reliable),
    "n_observation_persistent": len(after_persistence), "n_final_clean_sample": len(clean_ids),
    "buffer_sensitivity_precheck": sens_rows,
    "note": ("The 30-day stable-window rule from the discovery paper is dropped entirely here "
             "in favor of an EARLY_WINDOW-based criterion, since this study measures outcomes "
             "rather than identifying treatment events."),
}
with open(OUT_DIR / "sample_reconstruction_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print(f"\nSaved clean_onboarding_sample.csv ({len(clean_ids)} customers) and manifest.")
print("Next: step0b_first_channel_assignment.py")
