"""
================================================================================
Applied Paper - Step 0b: First-Adopted Campaign Type Labeling
================================================================================
Two competing definitions are computed for robustness comparison:
    (a) Same-day definition: campaign type(s) active on the first day with
        cost > 0. If multiple types are active the same day, the customer
        is labeled "co-first."
    (b) Cumulative-share definition: the type with the largest cumulative
        cost share over the first EARLY_LABEL_WINDOW days (default 3).

Definition (a) is the primary specification; (b) is a robustness check.
The agreement rate between the two is reported.

Output: reframe_output/first_channel_labels.csv
        reframe_output/first_channel_definition_agreement.json
        reframe_output/first_channel_cell_size_preview.csv
================================================================================
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
OUT_DIR = Path(os.environ.get("REFRAME_OUT", str(ROOT / "reframe_output")))

EARLY_LABEL_WINDOW = int(os.environ.get("EARLY_LABEL_WINDOW_DAYS", "3"))
REG_ACTIVE_GAP_FLAG_DAYS = int(os.environ.get("REG_ACTIVE_GAP_FLAG_DAYS", "3"))
MIN_CELL_SIZE = int(os.environ.get("MIN_CELL_SIZE", "8"))


def _detect_date_column(df):
    for c in ("date", "stat_date", "stat_dt", "dt", "ad_date", "report_date", "log_date"):
        if c in df.columns:
            return c
    raise KeyError("Could not auto-detect a date column")


ctp = pd.read_csv(ROOT / "customer_day_campaign_type_panel.csv")
sample = pd.read_csv(OUT_DIR / "clean_onboarding_sample.csv")

DATE_COL_CTP = _detect_date_column(ctp)
ctp[DATE_COL_CTP] = pd.to_datetime(ctp[DATE_COL_CTP])
ctp["campaign_type"] = pd.to_numeric(ctp["campaign_type"], errors="coerce")
sample["registration_date"] = pd.to_datetime(sample["registration_date"])

sample_ids = set(sample["customer_id"].unique())
ctp_sample = ctp[ctp["customer_id"].isin(sample_ids) & (ctp["cost"] > 0)].copy()
print(f"Sample: {len(sample_ids)} customers, with cost>0 records: {ctp_sample['customer_id'].nunique()}")

# --- Definition (a): same-day ---
first_active_date = ctp_sample.groupby("customer_id")[DATE_COL_CTP].min().rename("first_active_date")
rows_a = []
for cid, f_date in first_active_date.items():
    types = sorted(ctp_sample.loc[(ctp_sample["customer_id"] == cid) & (ctp_sample[DATE_COL_CTP] == f_date),
                                   "campaign_type"].dropna().unique().astype(int).tolist())
    rows_a.append({"customer_id": cid, "first_active_date": f_date,
                    "first_campaign_type_a": types[0] if types else np.nan,
                    "is_cofirst_a": len(types) > 1,
                    "cofirst_types_a": ",".join(map(str, types)) if len(types) > 1 else ""})
label_a = pd.DataFrame(rows_a)
n_cofirst = int(label_a["is_cofirst_a"].sum())
print(f"Definition (a) labeled: {len(label_a)}; co-first: {n_cofirst}")

# --- Definition (b): cumulative-share over first EARLY_LABEL_WINDOW days ---
reg_date_map = sample.set_index("customer_id")["registration_date"].to_dict()
rows_b = []
for cid in sorted(sample_ids):
    reg_date = reg_date_map.get(cid)
    if pd.isna(reg_date):
        continue
    window_end = reg_date + pd.Timedelta(days=EARLY_LABEL_WINDOW)
    sub = ctp_sample[(ctp_sample["customer_id"] == cid) & (ctp_sample[DATE_COL_CTP] >= reg_date)
                      & (ctp_sample[DATE_COL_CTP] < window_end)]
    if sub.empty:
        rows_b.append({"customer_id": cid, "first_campaign_type_b": np.nan})
        continue
    top_type = int(sub.groupby("campaign_type")["cost"].sum().idxmax())
    rows_b.append({"customer_id": cid, "first_campaign_type_b": top_type})
label_b = pd.DataFrame(rows_b)

merged = sample[["customer_id", "registration_date"]].merge(label_a, on="customer_id", how="left") \
    .merge(label_b, on="customer_id", how="left")
merged["reg_to_active_gap_days"] = (merged["first_active_date"] - merged["registration_date"]).dt.days
merged["large_reg_active_gap_flag"] = merged["reg_to_active_gap_days"] > REG_ACTIVE_GAP_FLAG_DAYS

valid_both = merged.dropna(subset=["first_campaign_type_a", "first_campaign_type_b"])
agree_mask = valid_both["first_campaign_type_a"] == valid_both["first_campaign_type_b"]
agreement_rate = float(agree_mask.mean()) if len(valid_both) else np.nan
print(f"(a) vs (b) agreement: {int(agree_mask.sum())}/{len(valid_both)} = {agreement_rate:.1%}")

n_large_gap = int(merged["large_reg_active_gap_flag"].sum())
print(f"Customers with registration-to-active gap > {REG_ACTIVE_GAP_FLAG_DAYS}d: {n_large_gap}")

merged.to_csv(OUT_DIR / "first_channel_labels.csv", index=False)
with open(OUT_DIR / "first_channel_definition_agreement.json", "w") as f:
    json.dump({"early_label_window_days_def_b": EARLY_LABEL_WINDOW,
               "reg_active_gap_flag_threshold_days": REG_ACTIVE_GAP_FLAG_DAYS,
               "n_labeled_total": int(len(merged)), "n_cofirst_def_a": n_cofirst,
               "n_compared_a_vs_b": int(len(valid_both)), "agreement_rate_a_vs_b": agreement_rate,
               "n_large_reg_active_gap": n_large_gap}, f, indent=2, default=str)

single = merged[~merged["is_cofirst_a"].fillna(False)]
cell_preview = single["first_campaign_type_a"].value_counts().rename_axis("campaign_type") \
    .reset_index(name="n_customers").sort_values("campaign_type")
cell_preview["below_min_cell_size"] = cell_preview["n_customers"] < MIN_CELL_SIZE
cell_preview.to_csv(OUT_DIR / "first_channel_cell_size_preview.csv", index=False)
print(cell_preview.to_string(index=False))
print("\nNext: step0c_cell_size_check.py to finalize pooling/exploratory decisions.")
