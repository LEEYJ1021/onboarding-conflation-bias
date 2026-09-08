"""
================================================================================
Step 1a — Master Analysis Dataset Construction (df_analysis_master.csv)
================================================================================
Builds the clean analysis panel from four raw sample-selection criteria
(registry match / continuous observation block / stable window exists /
non-test & non-billing-anomalous) plus a fifth criterion adopted after
preliminary diagnostics: a minimum stable-window length of
MIN_STABLE_WINDOW_DAYS (default 30) days.

This fifth rule was adopted to resolve a pilot-diagnostic finding: one
"clean" customer had zero all-time cumulative spend. That customer turned
out to hold a 4-day post-registration, pre-execution account. Excluding
this case ad hoc was judged less defensible than adding a general,
rule-based lower bound on observation length, which also improved the
post-log standard deviation (+7.2%) at a small cost in sample size (-0.4%
of the pool at that stage).

Outputs
-------
    df_analysis_master.csv
    reproducibility_manifest.json
    sample_selection_audit.csv
================================================================================
"""
import os
import json
import hashlib
import datetime as dt
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
OUT_DIR = ROOT

# Minimum stable-window length adopted after the zero-spend diagnostic (see docstring)
MIN_STABLE_WINDOW_DAYS = int(os.environ.get("MIN_STABLE_WINDOW_DAYS", "30"))

REQUIRED_FILES = [
    "customer_day_campaign_type_panel.csv",
    "customer_level_attributes.csv",
    "customer_day_panel.csv",
]
for fname in REQUIRED_FILES:
    if not (ROOT / fname).exists():
        raise FileNotFoundError(f"Required input file missing: {ROOT / fname}")


def _detect_date_column(df: pd.DataFrame, candidates=(
    "date", "stat_date", "stat_dt", "dt", "ad_date", "report_date", "log_date",
)) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c].dropna().iloc[:5])
            return c
        except (ValueError, TypeError):
            continue
    raise KeyError(f"Could not auto-detect a date column. Available columns: {list(df.columns)}")


ctp = pd.read_csv(ROOT / "customer_day_campaign_type_panel.csv")
attrs = pd.read_csv(ROOT / "customer_level_attributes.csv")
panel = pd.read_csv(ROOT / "customer_day_panel.csv")

DATE_COL = _detect_date_column(panel)
DATE_COL_CTP = _detect_date_column(ctp)
panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])
ctp[DATE_COL_CTP] = pd.to_datetime(ctp[DATE_COL_CTP])

# ------------------------------------------------------------------
# 1. Sample selection (registry match -> continuous block -> stable window ->
#    exclude test/billing-anomalous -> minimum-30-day rule)
# ------------------------------------------------------------------
audit_rows = []

registry_customers = set(attrs["customer_id"].unique())
panel_registry = panel[panel["customer_id"].isin(registry_customers)].copy()
dropped = set(panel["customer_id"].unique()) - registry_customers
for cid in sorted(dropped):
    audit_rows.append({"customer_id": cid, "dropped_at_stage": "registry_match",
                        "reason": "customer_id not registered in attrs"})

block_ids_per_customer = panel_registry.groupby("customer_id")["block_id"].unique()
continuous_block_customers = set(
    block_ids_per_customer[block_ids_per_customer.apply(lambda b: 2 in b)].index
)
panel_continuous = panel_registry[panel_registry["customer_id"].isin(continuous_block_customers)]
dropped = set(panel_registry["customer_id"].unique()) - continuous_block_customers
for cid in sorted(dropped):
    audit_rows.append({"customer_id": cid, "dropped_at_stage": "continuous_block",
                        "reason": "no block_id==2 segment"})

stable_customers = set(
    panel_continuous.groupby("customer_id")["is_stable_window"]
    .any().pipe(lambda s: s[s].index)
)
panel_stable = panel_continuous[panel_continuous["customer_id"].isin(stable_customers)]
dropped = set(panel_continuous["customer_id"].unique()) - stable_customers
for cid in sorted(dropped):
    audit_rows.append({"customer_id": cid, "dropped_at_stage": "stable_window_exists",
                        "reason": "no date with is_stable_window=True"})

flags_ = panel_stable.groupby("customer_id")[["is_test_account", "billing_anomaly_flag"]].any()
CLEAN_SAMPLE_PRE_MINDAYS = set(flags_[~(flags_["is_test_account"] | flags_["billing_anomaly_flag"])].index)
dropped = set(panel_stable["customer_id"].unique()) - CLEAN_SAMPLE_PRE_MINDAYS
for cid in sorted(dropped):
    audit_rows.append({"customer_id": cid, "dropped_at_stage": "test_or_billing_anomaly",
                        "reason": "is_test_account or billing_anomaly_flag=True"})
assert len(CLEAN_SAMPLE_PRE_MINDAYS) > 0, "Clean sample is empty."

n_stable_days = (
    panel[panel["customer_id"].isin(CLEAN_SAMPLE_PRE_MINDAYS) & panel["is_stable_window"]]
    .groupby("customer_id")[DATE_COL].nunique()
)
short_window_customers = set(n_stable_days[n_stable_days < MIN_STABLE_WINDOW_DAYS].index)
for cid in sorted(short_window_customers):
    audit_rows.append({
        "customer_id": cid, "dropped_at_stage": "min_stable_window_days",
        "reason": f"stable-window observed days={int(n_stable_days.loc[cid])} < minimum {MIN_STABLE_WINDOW_DAYS}",
    })

CLEAN_SAMPLE = CLEAN_SAMPLE_PRE_MINDAYS - short_window_customers
assert len(CLEAN_SAMPLE) > 0, "Sample is empty after the 30-day minimum rule."

print(f"Sample selection summary: registry {len(registry_customers)} -> continuous block "
      f"{len(continuous_block_customers)} -> stable window exists {len(stable_customers)} "
      f"-> non-test/non-anomalous {len(CLEAN_SAMPLE_PRE_MINDAYS)} "
      f"-> min-{MIN_STABLE_WINDOW_DAYS}-day rule applied -> final {len(CLEAN_SAMPLE)}")

audit_df = pd.DataFrame(audit_rows)
audit_path = OUT_DIR / "sample_selection_audit.csv"
audit_df.to_csv(audit_path, index=False)

# ------------------------------------------------------------------
# 2. cost_zero_category / usable_for_cpc labeling
# ------------------------------------------------------------------
def label_cost_zero(df: pd.DataFrame, has_campaign_type: bool = False) -> pd.DataFrame:
    d = df.copy()
    has_billing_flag = "billing_anomaly_flag" in d.columns
    has_type_col = has_campaign_type and "campaign_type" in d.columns
    ct_num = pd.to_numeric(d["campaign_type"], errors="coerce") if has_type_col else pd.Series(np.nan, index=d.index)

    def _categorize(row):
        if row["cost"] > 0:
            return "not_zero"
        if row["click"] == 0:
            return "undefined_0_over_0"
        if has_billing_flag and bool(row["billing_anomaly_flag"]):
            return "click_positive_billing_anom"
        if has_type_col and ct_num.loc[row.name] == 4:
            return "click_positive_type4"
        return "click_positive_other"

    d["cost_zero_category"] = d.apply(_categorize, axis=1)
    is_type4 = ct_num.eq(4) if has_type_col else pd.Series(False, index=d.index)
    d["usable_for_cpc"] = (d["cost"] > 0) & (~is_type4)
    return d


panel_labeled = label_cost_zero(panel, has_campaign_type=False)
ctp_labeled = label_cost_zero(ctp, has_campaign_type=True)

# ------------------------------------------------------------------
# 3. Customer-day skeleton + type-4 contamination check
# ------------------------------------------------------------------
day = panel_labeled[
    panel_labeled["customer_id"].isin(CLEAN_SAMPLE) & panel_labeled["is_stable_window"]
].copy()

ct_num_full = pd.to_numeric(ctp_labeled["campaign_type"], errors="coerce")
type4_by_day = (
    ctp_labeled.assign(_ct=ct_num_full)
    .groupby(["customer_id", DATE_COL_CTP])
    .apply(lambda g: bool(((g["_ct"] == 4) & (g["cost"] > 0)).any()))
    .rename("has_type4_cost")
    .reset_index()
    .rename(columns={DATE_COL_CTP: "_ctp_date_key"})
)

day = day.merge(
    type4_by_day,
    left_on=["customer_id", DATE_COL],
    right_on=["customer_id", "_ctp_date_key"],
    how="left",
)
day["cpc_type4_check_status"] = np.select(
    [day["has_type4_cost"].isna(), day["has_type4_cost"] == True],  # noqa: E712
    ["unknown_no_ctp_match", "type4_contaminated"],
    default="clean",
)
day["usable_for_cpc_verified"] = day["usable_for_cpc"] & (day["cpc_type4_check_status"] == "clean")
day = day.drop(columns=["_ctp_date_key", "has_type4_cost"])

day["log_spend_safe"] = np.log1p(day["cost"])
day["log_cpc_safe"] = np.where(day["usable_for_cpc_verified"], np.log(day["cpc"]), np.nan)

Z_THRESHOLD = 5.0
valid_mask = day["usable_for_cpc_verified"] & day["log_cpc_safe"].notna()
grp = day.loc[valid_mask].groupby("customer_id")["log_cpc_safe"]
mu, sd = grp.transform("mean"), grp.transform("std")
day["log_cpc_z_within_customer"] = np.nan
day.loc[valid_mask, "log_cpc_z_within_customer"] = (day.loc[valid_mask, "log_cpc_safe"] - mu) / sd
undefined_z_mask = valid_mask & (sd.isna() | (sd == 0))
day["is_extreme_value"] = (day["log_cpc_z_within_customer"].abs() > Z_THRESHOLD).astype("boolean")
day.loc[~valid_mask | undefined_z_mask, "is_extreme_value"] = pd.NA

DAY_COLUMNS = [
    "customer_id", DATE_COL, "impression", "click", "cost", "cpc", "mobile_share",
    "avg_ad_rank", "cost_zero_category", "usable_for_cpc_verified",
    "log_spend_safe", "log_cpc_safe", "log_cpc_z_within_customer", "is_extreme_value",
]
DAY_COLUMNS = [c for c in DAY_COLUMNS if c in day.columns]
day = day[DAY_COLUMNS].copy()

# ------------------------------------------------------------------
# 4. Campaign-type-level wide integration
# ------------------------------------------------------------------
if "is_stable_window" in ctp_labeled.columns:
    ctp_win = ctp_labeled.copy()
else:
    lookup = (
        panel[["customer_id", DATE_COL, "is_stable_window"]]
        .drop_duplicates(subset=["customer_id", DATE_COL])
        .rename(columns={DATE_COL: "_panel_date_key"})
    )
    ctp_win = ctp_labeled.merge(
        lookup, left_on=["customer_id", DATE_COL_CTP], right_on=["customer_id", "_panel_date_key"], how="left"
    )
    ctp_win["is_stable_window"] = ctp_win["is_stable_window"].fillna(False)
    ctp_win = ctp_win.drop(columns=["_panel_date_key"], errors="ignore")

ct = ctp_win[ctp_win["customer_id"].isin(CLEAN_SAMPLE) & ctp_win["is_stable_window"]].copy()
ct["campaign_type"] = pd.to_numeric(ct["campaign_type"], errors="coerce")
ct = ct[ct["campaign_type"].notna()].copy()
ct["campaign_type"] = ct["campaign_type"].astype(int)
ct = ct.rename(columns={DATE_COL_CTP: DATE_COL})

if "impression" in ct.columns:
    ct["ctr_safe"] = np.where(ct["impression"] > 0, ct["click"] / ct["impression"], np.nan)
if "cpc" in ct.columns:
    ct["cpc_safe"] = np.where(ct["usable_for_cpc"], ct["cpc"], np.nan)
elif {"cost", "click"}.issubset(ct.columns):
    ct["cpc_safe"] = np.where(ct["usable_for_cpc"], ct["cost"] / ct["click"], np.nan)
ct["log_cpc_safe"] = np.where(ct["usable_for_cpc"], np.log(ct["cpc_safe"]), np.nan) if "cpc_safe" in ct.columns else np.nan

SUM_METRICS = [c for c in ["impression", "click", "cost"] if c in ct.columns]
RATE_METRICS = [c for c in ["ctr_safe", "cpc_safe", "log_cpc_safe"] if c in ct.columns]

ct_wide_sum = ct.pivot_table(
    index=["customer_id", DATE_COL], columns="campaign_type", values=SUM_METRICS,
    aggfunc="sum", fill_value=0,
)
ct_wide_sum.columns = [f"{metric}_type{int(ctype)}" for metric, ctype in ct_wide_sum.columns]

ct_wide_rate = ct.pivot_table(
    index=["customer_id", DATE_COL], columns="campaign_type", values=RATE_METRICS,
    aggfunc="mean",
)
rate_name_map = {"ctr_safe": "ctr", "cpc_safe": "cpc", "log_cpc_safe": "log_cpc"}
ct_wide_rate.columns = [f"{rate_name_map.get(metric, metric)}_type{int(ctype)}" for metric, ctype in ct_wide_rate.columns]

ct_wide = ct_wide_sum.join(ct_wide_rate, how="outer").reset_index()

day = day.merge(ct_wide, on=["customer_id", DATE_COL], how="left")
sum_wide_cols = list(ct_wide_sum.columns)
rate_wide_cols = list(ct_wide_rate.columns)
day[sum_wide_cols] = day[sum_wide_cols].fillna(0.0)

TYPE_CODES = sorted({int(c.split("type")[-1]) for c in sum_wide_cols})
share_wide_cols = []
for tcode in TYPE_CODES:
    for base_col, share_name in [("cost", "cost_share"), ("click", "click_share"), ("impression", "impression_share")]:
        type_col = f"{base_col}_type{tcode}"
        if type_col in day.columns and base_col in day.columns:
            share_col = f"{share_name}_type{tcode}"
            day[share_col] = np.where(day[base_col] > 0, day[type_col] / day[base_col], np.nan)
            share_wide_cols.append(share_col)

type_wide_cols = sum_wide_cols + rate_wide_cols + share_wide_cols
day["n_campaign_types_active"] = sum((day[f"cost_type{tcode}"] > 0).astype(int) for tcode in TYPE_CODES)

# ------------------------------------------------------------------
# 5. Customer-level campaign-type shares / counts
# ------------------------------------------------------------------
customer_type_totals = ct.groupby(["customer_id", "campaign_type"])[SUM_METRICS].sum().reset_index()
customer_type_wide = customer_type_totals.pivot(
    index="customer_id", columns="campaign_type", values=SUM_METRICS
).fillna(0.0)
customer_type_wide.columns = [
    f"customer_{metric}_type{int(ctype)}_total" for metric, ctype in customer_type_wide.columns
]
customer_type_wide = customer_type_wide.reset_index()

for metric in SUM_METRICS:
    total_cols = [c for c in customer_type_wide.columns if c.startswith(f"customer_{metric}_type") and c.endswith("_total")]
    row_total = customer_type_wide[total_cols].sum(axis=1)
    for c in total_cols:
        tcode = c.replace(f"customer_{metric}_type", "").replace("_total", "")
        share_col = f"customer_{metric}_share_type{tcode}"
        customer_type_wide[share_col] = np.where(row_total > 0, customer_type_wide[c] / row_total, np.nan)

customer_active_days_total = day.groupby("customer_id")[DATE_COL].nunique().rename("customer_n_days_total")
active_day_wide = customer_type_wide[["customer_id"]].copy()
for tcode in TYPE_CODES:
    flag_col = f"_active_flag_type{tcode}"
    day[flag_col] = (day[f"cost_type{tcode}"] > 0).astype(int)
    n_active = day.groupby("customer_id")[flag_col].sum().rename(f"customer_n_active_days_type{tcode}")
    active_day_wide = active_day_wide.merge(n_active, on="customer_id", how="left")
    day = day.drop(columns=[flag_col])
active_day_wide = active_day_wide.merge(customer_active_days_total, on="customer_id", how="left")
customer_type_wide = customer_type_wide.merge(active_day_wide, on="customer_id", how="left")

# ------------------------------------------------------------------
# 6. Customer-level static attributes
# ------------------------------------------------------------------
level = attrs[attrs["customer_id"].isin(CLEAN_SAMPLE)].copy()
LEVEL_COLUMNS = ["customer_id", "total_cost", "all_time_ad_group_count",
                 "device_type_mode", "campaign_type_dominant"]
LEVEL_COLUMNS = [c for c in LEVEL_COLUMNS if c in level.columns]
level = level[LEVEL_COLUMNS].copy()

complete_case_cols = [c for c in ["device_type_mode", "campaign_type_dominant"] if c in level.columns]
level["is_complete_case_device_campaign"] = (
    level[complete_case_cols].notna().all(axis=1) if complete_case_cols else True
)
level = level.rename(columns={
    "total_cost": "customer_total_cost_alltime",
    "all_time_ad_group_count": "customer_ad_group_count_alltime",
})
level = level.merge(customer_type_wide, on="customer_id", how="left")

df_master = day.merge(level, on="customer_id", how="left")

remaining_zero_spend = set(level.loc[level["customer_total_cost_alltime"] == 0, "customer_id"])
if remaining_zero_spend:
    print(f"WARNING: zero-spend customers remain after 30-day rule: {sorted(remaining_zero_spend)}")
else:
    print("OK: no zero-spend customers remain after 30-day rule")

# ------------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------------
master_path = OUT_DIR / "df_analysis_master.csv"
df_master.to_csv(master_path, index=False)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


manifest = {
    "run_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "root_path": str(ROOT),
    "input_file_md5": {f: _md5(ROOT / f) for f in REQUIRED_FILES},
    "date_columns": {"panel": DATE_COL, "ctp": DATE_COL_CTP},
    "sample_selection": {
        "n_registry_matched": len(registry_customers),
        "n_continuous_block": len(continuous_block_customers),
        "n_stable_window_exists": len(stable_customers),
        "n_clean_pre_mindays": len(CLEAN_SAMPLE_PRE_MINDAYS),
        "min_stable_window_days_rule": MIN_STABLE_WINDOW_DAYS,
        "n_excluded_by_min_days_rule": len(short_window_customers),
        "excluded_by_min_days_rule": sorted(short_window_customers),
        "n_clean_customers_final": len(CLEAN_SAMPLE),
    },
    "n_clean_customers": len(CLEAN_SAMPLE),
    "n_master_rows": int(len(df_master)),
    "campaign_type_wide_columns_day": type_wide_cols + ["n_campaign_types_active"],
    "campaign_type_wide_columns_customer": [c for c in customer_type_wide.columns if c != "customer_id"],
    "extreme_value_definition": {
        "basis_column": "log_cpc_safe", "groupby": "customer_id", "z_threshold": Z_THRESHOLD,
    },
}
manifest_path = OUT_DIR / "reproducibility_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(f"\n{master_path.name:30s} {len(df_master):>7,} rows / {df_master['customer_id'].nunique()} customers "
      f"/ {len(df_master.columns)} columns")
print(f"{manifest_path.name}")
print(f"{audit_path.name}")
